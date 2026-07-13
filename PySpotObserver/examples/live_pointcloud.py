#!/usr/bin/env python3
"""
Live fused point cloud from ONE Spot's two front fisheye cameras.

Uses the fisheye calibration produced by calibration/calibrate_fisheye.py:
  * per-camera K + Kannala-Brandt D  -> correct fisheye deprojection of depth
  * frontleft<->frontright R,T        -> fuse both cameras into the frontleft frame

The result is a single live point cloud per robot, shown in an Open3D window.
Each robot's cloud lands in its own frontleft camera frame, so a later
robot-to-robot transform can line two of them up.

Run from the examples/ directory (needs common_cli + pyspotobserver), e.g.:

    # TUSKER, using its calibration:
    python live_pointcloud.py --robot-ip 128.148.138.22 \
        --calib calib/spot --user user --password ******

    # GOUGER:
    python live_pointcloud.py --robot-ip 128.148.138.21 \
        --calib calib/spot2 --user user --password ******

Controls: close the window to quit.

Notes
-----
* Depth must be registered to the fisheye visual frame (same HxW as the RGB
  image). The script checks this on the first frame and warns if it is not, in
  which case pixel<->depth do not correspond and the cloud will be wrong.
* Points beyond MAX_ANGLE_DEG from the optical axis are dropped -- fisheye
  unprojection is unstable at the extreme rim and can spit out garbage rays.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from pyspotobserver import SpotConnection
from common_cli import (
    add_common_connection_arguments,
    build_camera_mask,
    build_config_from_args,
    parse_camera_list,
)

# Drop rays past this incidence angle; the KB unprojection blows up at the rim.
MAX_ANGLE_DEG = 85.0
# Per-camera tint so you can see which camera a point came from (and thus how
# well the two overlap). Front cams are grayscale, so this tints the intensity.
TINT_LEFT = np.array([1.0, 0.55, 0.55])   # reddish
TINT_RIGHT = np.array([0.55, 0.65, 1.0])  # bluish
# Convert Spot camera optical axes (X right, Y down, Z forward) to a nicer
# Open3D viewing orientation (flip Y and Z so the scene sits upright).
VIEW_FLIP = np.array([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_connection_arguments(parser)
    parser.add_argument(
        "--calib",
        required=True,
        help="Robot calib dir or calibration.yaml (e.g. calib/spot).",
    )
    parser.add_argument("--cameras", default="frontleft,frontright",
                        help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="Per-frame retrieval timeout (s).")
    parser.add_argument("--stream-id", default="cloud_stream")
    parser.add_argument("--stride", type=int, default=2,
                        help="Pixel subsample step for speed (1=full res, 2=1/4 points).")
    parser.add_argument("--min-depth", type=float, default=0.2,
                        help="Discard points closer than this (m).")
    parser.add_argument("--max-depth", type=float, default=3.0,
                        help="Discard points farther than this (m); Spot depth is noisy far out.")
    parser.add_argument("--no-clean", dest="clean", action="store_false",
                        help="Disable statistical outlier removal (faster, noisier).")
    parser.set_defaults(clean=True)
    parser.add_argument("--point-size", type=float, default=3.0,
                        help="Render point size in px (bigger = denser-looking).")
    parser.add_argument("--fill-holes", action="store_true",
                        help="Fill small depth gaps with a local average of valid neighbors "
                             "(approximate: interpolates across holes).")
    parser.add_argument("--fill-ksize", type=int, default=7,
                        help="Neighborhood size for --fill-holes.")
    parser.add_argument("--plain", action="store_true",
                        help="Use raw image intensity for color instead of per-camera tint.")
    parser.add_argument("--only", choices=["frontleft", "frontright", "both"],
                        default="both", help="Show only one camera's cloud.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Grab one frame, print per-camera stats, dump .ply files, exit.")
    # build_config_from_args reads these two unconditionally.
    parser.add_argument("--dumps-enabled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save-dir", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_calibration(path_arg: str) -> dict:
    path = Path(path_arg)
    if path.is_dir():
        path = path / "calibration.yaml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit(f"Cannot open calibration: {path}")
    out = {}
    for side in ("frontleft", "frontright"):
        out[f"K_{side}"] = fs.getNode(f"K_{side}").mat().astype(np.float64)
        out[f"D_{side}"] = fs.getNode(f"D_{side}").mat().reshape(-1, 1).astype(np.float64)
    R = fs.getNode("R_left_to_right").mat()
    T = fs.getNode("T_left_to_right").mat()
    fs.release()
    if R is None or T is None:
        raise SystemExit(f"{path} has no R_left_to_right / T_left_to_right")
    out["R"] = R.astype(np.float64)
    out["T"] = T.reshape(3).astype(np.float64)
    return out


def to_gray_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) float image in [0, 1] for use as point colors."""
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr, 0.0, 1.0)


def fill_small_holes(dep: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Fill invalid depth pixels with the average of valid depth in a KxK window.

    Only fills where the window actually contains valid depth, so large empty
    regions stay empty (we don't invent geometry out of nothing). Approximate --
    it smooths across depth discontinuities -- but good for a denser viewer.
    """
    d = np.asarray(dep, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    valid = np.isfinite(d) & (d > 0)
    dv = np.where(valid, d, 0.0).astype(np.float32)
    vf = valid.astype(np.float32)
    k = np.ones((ksize, ksize), np.float32)
    num = cv2.filter2D(dv, -1, k, borderType=cv2.BORDER_CONSTANT)
    den = cv2.filter2D(vf, -1, k, borderType=cv2.BORDER_CONSTANT)
    # require at least a quarter of the window to be valid before filling
    fillable = (~valid) & (den >= (ksize * ksize) / 4.0)
    out = d.copy()
    out[fillable] = num[fillable] / den[fillable]
    return out


def backproject(
    depth: np.ndarray, color: np.ndarray, K: np.ndarray, D: np.ndarray,
    stride: int, min_d: float, max_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fisheye-unproject valid depth pixels -> (points Nx3, colors Nx3) in cam frame."""
    dep = np.asarray(depth)
    if dep.ndim == 3:
        dep = dep[..., 0]
    dep = dep[::stride, ::stride]
    col = color[::stride, ::stride]

    vs, us = np.indices(dep.shape)
    valid = np.isfinite(dep) & (dep >= min_d) & (dep <= max_d)
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3))

    z = dep[valid].astype(np.float64)
    # undistortPoints wants ORIGINAL pixel coords, so scale the strided indices.
    u = (us[valid] * stride).astype(np.float64)
    v = (vs[valid] * stride).astype(np.float64)
    pix = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
    rays = cv2.fisheye.undistortPoints(pix, K, D).reshape(-1, 2)  # (x/z, y/z)

    r = np.linalg.norm(rays, axis=1)
    keep = r < np.tan(np.radians(MAX_ANGLE_DEG))
    rays, z = rays[keep], z[keep]
    pts = np.column_stack((rays[:, 0] * z, rays[:, 1] * z, z))
    cols = col[valid][keep].reshape(-1, 3)
    return pts, cols


def _run_diagnose(per_cam: dict, out_dir: str = "cloud_diag") -> None:
    """Print per-camera stats and dump .ply files for offline inspection."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== diagnose ===")
    for nm, (pts, tcols, rgb, dep) in per_cam.items():
        d = dep[..., 0] if dep.ndim == 3 else dep
        valid = np.isfinite(d) & (d > 0)
        print(f"[{nm}] rgb shape={rgb.shape} dtype={rgb.dtype} "
              f"range[{np.asarray(rgb).min():.3f},{np.asarray(rgb).max():.3f}]  "
              f"depth shape={d.shape} dtype={d.dtype}")
        if valid.any():
            dv = d[valid]
            print(f"       depth: valid={int(valid.sum())} "
                  f"min={dv.min():.2f} median={np.median(dv):.2f} max={dv.max():.2f} m")
        print(f"       cloud pts={len(pts)}", end="")
        if len(pts):
            print(f"  centroid={np.round(pts.mean(0),3)} "
                  f"extent={np.round(pts.max(0)-pts.min(0),3)}")
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(tcols)
            o3d.io.write_point_cloud(f"{out_dir}/{nm}.ply", pc)
            print(f"       wrote {out_dir}/{nm}.ply")
        else:
            print()
    allp = [v[0] for v in per_cam.values() if len(v[0])]
    if allp:
        fused = o3d.geometry.PointCloud()
        fused.points = o3d.utility.Vector3dVector(np.vstack(allp))
        fused.colors = o3d.utility.Vector3dVector(
            np.vstack([v[1] for v in per_cam.values() if len(v[0])]))
        o3d.io.write_point_cloud(f"{out_dir}/fused.ply", fused)
        print(f"wrote {out_dir}/fused.ply ({len(fused.points)} pts)")
    print("=== end diagnose ===\n")


def main() -> int:
    args = parse_args()
    calib = load_calibration(args.calib)
    cameras = parse_camera_list(args.cameras)
    config = build_config_from_args(args)

    Rlr, Tlr = calib["R"], calib["T"]

    vis = None
    if not args.diagnose:
        vis = o3d.visualization.Visualizer()
        vis.create_window("Spot front cloud (frontleft frame)", width=1280, height=720)
        vis.get_render_option().point_size = args.point_size
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    pcd = o3d.geometry.PointCloud()
    added = False
    checked_shape = False
    diag_tries = 0

    with ExitStack() as stack:
        conn = stack.enter_context(SpotConnection(config))
        print("Connected:", conn)
        stream = conn.create_cam_stream(stream_id=args.stream_id)
        stream.start_streaming(build_camera_mask(cameras))
        order = [c.name.lower() for c in stream.get_camera_order()]
        print("Camera order:", order, "-- close the window to quit.")

        try:
            while True:
                rgb_images, depth_images = stream.get_current_images(
                    timeout=args.timeout, run_pipeline=False
                )
                imgs = {order[i]: (rgb_images[i], depth_images[i])
                        for i in range(len(order))}
                if "frontleft" not in imgs or "frontright" not in imgs:
                    print("Missing a front camera in stream; got", order)
                    break

                if not checked_shape:
                    for nm in ("frontleft", "frontright"):
                        rr = np.asarray(imgs[nm][0]); dd = np.asarray(imgs[nm][1])
                        if rr.shape[:2] != dd.shape[:2]:
                            print(f"WARNING: {nm} depth {dd.shape[:2]} != rgb "
                                  f"{rr.shape[:2]}; depth is NOT registered to the "
                                  f"fisheye frame -- cloud will be wrong.")
                    checked_shape = True

                per_cam = {}
                for nm, tint in (("frontleft", TINT_LEFT), ("frontright", TINT_RIGHT)):
                    if args.only != "both" and nm != args.only:
                        continue
                    rgb, dep = imgs[nm]
                    if args.fill_holes:
                        dep = fill_small_holes(dep, args.fill_ksize)
                    color = to_gray_rgb(rgb)
                    pts, cols = backproject(
                        dep, color, calib[f"K_{nm}"], calib[f"D_{nm}"],
                        args.stride, args.min_depth, args.max_depth,
                    )
                    if nm == "frontright" and len(pts):
                        # our R,T maps left->right (p_r = R p_l + T); bring the
                        # right cloud into the left frame: p_l = R^T (p_r - T).
                        pts = (pts - Tlr) @ Rlr
                    tcols = cols if args.plain else np.clip(cols * tint, 0, 1)
                    per_cam[nm] = (pts, tcols, np.asarray(rgb), np.asarray(dep))

                if args.diagnose:
                    diag_tries += 1
                    if any(len(v[0]) > 100 for v in per_cam.values()) or diag_tries >= 20:
                        _run_diagnose(per_cam)
                        break
                    continue

                pts_all = [v[0] for v in per_cam.values() if len(v[0])]
                cols_all = [v[1] for v in per_cam.values() if len(v[0])]
                if not pts_all:
                    if not vis.poll_events():
                        break
                    vis.update_renderer()
                    continue

                pts = np.vstack(pts_all) * VIEW_FLIP
                cols = np.vstack(cols_all)
                if args.clean and len(pts) > 50:
                    tmp = o3d.geometry.PointCloud()
                    tmp.points = o3d.utility.Vector3dVector(pts)
                    tmp.colors = o3d.utility.Vector3dVector(cols)
                    tmp, _ = tmp.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
                    pcd.points, pcd.colors = tmp.points, tmp.colors
                else:
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.colors = o3d.utility.Vector3dVector(cols)
                if not added:
                    vis.add_geometry(pcd)
                    added = True
                else:
                    vis.update_geometry(pcd)

                if not vis.poll_events():
                    break
                vis.update_renderer()
        finally:
            stream.stop_streaming()
            if vis is not None:
                vis.destroy_window()
            print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
