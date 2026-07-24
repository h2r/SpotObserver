#!/usr/bin/env python3
"""
REALTIME two-robot colored ICP — stream BOTH Spots and register on the fly.

Unlike test_realpair.py (offline, two saved .ply) this connects to both robots at once, builds a
fresh colored cloud from each frontleft+frontright pair every frame, and aligns robot-2 onto
robot-1 with PURE multi-scale COLORED ICP (no FPFH/RANSAC). The realtime win is the seed: each
frame warm-starts from the previous frame's transform, so once locked it stays locked and every
frame is a cheap refine. Frame 1 seeds from identity (or from a single bootstrap if you pass
--bootstrap-first, e.g. the robots start far apart in their own frontleft frames).

Colored ICP needs color to help, so pass --color (or env DIFFRENDER_COLOR=1); grayscale still runs
but only on intensity. XYZ is metres, RGB is [0,1] — the ingest handles both (see spot_ingest).

Run on a box that can reach BOTH robots. Open3D colored ICP is CPU-only; expect ~1-2 fps on ~20k
point clouds — "on the fly", not video rate.

Examples (from the diffrender_tracker/ directory):
    # TUSKER (spot) as robot-1, GOUGER (spot2) as robot-2, colored ICP, warm-started:
    python live_pair_icp.py \
        --robot-ip 128.148.138.22   --calib   calib/spot \
        --robot-ip-b 128.148.138.21 --calib-b calib/spot2 \
        --username user --password bigbubbabigbubba --color --frames 200

    # if identity can't acquire lock on frame 1, seed it once with FPFH/RANSAC then warm-start:
    python live_pair_icp.py ... --color --bootstrap-first

    # write the aligned merged cloud (robot1=red, robot2=blue) on exit to eyeball the fit:
    python live_pair_icp.py ... --color --save merged_live.ply
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PSO = os.path.join(HERE, "..", "PySpotObserver")
for p in (HERE, PSO, os.path.join(PSO, "examples")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common_cli import (                                    # noqa: E402  (needs sys.path above)
    add_common_connection_arguments,
    build_camera_mask,
    build_config_from_args,
    parse_camera_list,
)
from spot_ingest import (                                   # noqa: E402  (needs sys.path above)
    SpotCloudSource,
    env_color,
    load_fisheye_calib,
)
from bootstrap import bootstrap_register, colored_icp_register   # noqa: E402

# Optical frame is X-right/Y-down/Z-forward; flip Y,Z so the cloud sits upright under Open3D's
# default camera (same constant as examples/live_pointcloud.py).
VIEW_FLIP = np.array([1.0, -1.0, -1.0])
TINT_R1 = np.array([0.9, 0.2, 0.2])     # robot-1 = red
TINT_R2 = np.array([0.2, 0.4, 0.9])     # robot-2 = blue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_connection_arguments(p)                      # robot-1: --robot-ip/--username/...
    p.add_argument("--calib", required=True, help="Robot-1 calib dir or calibration.yaml.")
    p.add_argument("--robot-ip-b", required=True, help="Robot-2 IP address.")
    p.add_argument("--calib-b", required=True, help="Robot-2 calib dir or calibration.yaml.")
    p.add_argument("--username-b", help="Robot-2 username (defaults to --username).")
    p.add_argument("--password-b", help="Robot-2 password (defaults to --password).")
    p.add_argument("--cameras", default="frontleft,frontright", help=argparse.SUPPRESS)
    p.add_argument("--frames", type=int, default=200, help="ICP frames to run.")
    p.add_argument("--timeout", type=float, default=3.0, help="Per-frame retrieval timeout (s).")
    p.add_argument("--stride", type=int, default=2, help="Pixel subsample step (speed).")
    p.add_argument("--min-depth", type=float, default=0.2)
    p.add_argument("--max-depth", type=float, default=3.0)
    p.add_argument("--target", type=int, default=20000, help="Voxel-downsample point target.")
    p.add_argument("--voxel", type=float, default=0.05, help="ICP coarse voxel / scale (m).")
    p.add_argument("--color", action="store_true",
                   help="Keep CCM-corrected RGB (default grayscale). Also honours DIFFRENDER_COLOR=1.")
    p.add_argument("--no-warm-start", action="store_true",
                   help="Re-seed every frame from identity instead of the previous transform.")
    p.add_argument("--bootstrap-first", action="store_true",
                   help="Seed frame 1 with FPFH/RANSAC (bootstrap_register) to acquire lock, then "
                        "warm-start with pure colored ICP after. Use if identity is too far off.")
    p.add_argument("--min-overlap", type=float, default=0.15,
                   help="Frames below this overlap fitness aren't committed as the next seed.")
    p.add_argument("--view", action="store_true",
                   help="Open a live Open3D window of the merged aligned cloud (updates each frame).")
    p.add_argument("--view-rgb", action="store_true",
                   help="With --view, show each robot's true RGB instead of red/blue tint "
                        "(tint makes misalignment easier to see).")
    p.add_argument("--point-size", type=float, default=2.0, help="Live-view point size.")
    p.add_argument("--save", type=str, default=None,
                   help="On exit, write the aligned merged cloud (robot1=red, robot2=blue) here.")
    # build_config_from_args reads these two unconditionally.
    p.add_argument("--dumps-enabled", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--save-dir", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def resolve_calib(path_arg: str) -> str:
    """Accept calib/spot from any cwd (they live under PySpotObserver/examples)."""
    if os.path.exists(path_arg):
        return path_arg
    alt = os.path.join(PSO, "examples", path_arg)
    return alt if os.path.exists(alt) else path_arg


def camera_names(spec: str):
    """common_cli.parse_camera_list may return CameraType enums; SpotCloudSource wants names."""
    out = []
    for c in parse_camera_list(spec):
        out.append(c.name.lower() if hasattr(c, "name") else str(c).lower())
    return out


def build_config_b(config_a, args):
    """Robot-2 config = robot-1's config with the IP (and optionally creds) swapped. Deep-copied so
    the two SpotConnections never share mutable state."""
    config_b = copy.deepcopy(config_a)
    config_b.robot_ip = args.robot_ip_b
    if args.username_b:
        config_b.username = args.username_b
    if args.password_b:
        config_b.password = args.password_b
    return config_b


def save_merged(path, a_xyz, b_posed_xyz):
    """Aligned merged cloud tinted by robot — red on the same surfaces as blue => good alignment."""
    import open3d as o3d
    red = np.tile([0.9, 0.2, 0.2], (len(a_xyz), 1))
    blue = np.tile([0.2, 0.4, 0.9], (len(b_posed_xyz), 1))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.vstack([a_xyz, b_posed_xyz]).astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.vstack([red, blue]))
    o3d.io.write_point_cloud(path, pc)
    print(f"   saved {path}  (robot1=red, robot2=blue)")


def apply_T(T, xyz):
    T = np.asarray(T, np.float32)
    return xyz @ T[:3, :3].T + T[:3, 3]


def _make_view(args):
    """Open a live Open3D window. Returns (visualizer, merged PointCloud geometry)."""
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window("two-robot colored ICP (robot1=red, robot2=blue)", width=1280, height=720)
    # Add geometry BEFORE touching the render option: on macOS get_render_option() returns None
    # until the renderer is initialised by a first add_geometry, so setting point_size early
    # crashes with 'NoneType has no attribute point_size'.
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    ro = vis.get_render_option()
    if ro is not None:                                      # still guard: headless/GL failure -> None
        ro.point_size = args.point_size
    disp = o3d.geometry.PointCloud()                        # merged cloud, re-filled each frame
    return vis, disp


def _update_view(vis, disp, added, xyz_a, rgb_a, b_posed, rgb_b, view_rgb):
    """Push the merged aligned cloud into the window. Returns (alive, added) — alive is False if
    the user closed the window; `added` tracks the first add_geometry (pybind geometry can't hold a
    Python flag, so the caller owns it)."""
    import open3d as o3d
    if view_rgb:
        cols = np.vstack([np.clip(rgb_a, 0, 1), np.clip(rgb_b, 0, 1)])
    else:
        cols = np.vstack([np.tile(TINT_R1, (len(xyz_a), 1)), np.tile(TINT_R2, (len(b_posed), 1))])
    pts = np.vstack([xyz_a, b_posed]) * VIEW_FLIP           # flip into Open3D's view orientation
    disp.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    disp.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    if not added:
        vis.add_geometry(disp)                             # first frame frames the camera on the cloud
        added = True
    else:
        vis.update_geometry(disp)
    alive = vis.poll_events()
    vis.update_renderer()
    return alive, added


def main() -> int:
    args = parse_args()
    calib_a = load_fisheye_calib(resolve_calib(args.calib))
    calib_b = load_fisheye_calib(resolve_calib(args.calib_b))
    config_a = build_config_from_args(args)                 # robot-1 (uses --robot-ip)
    config_b = build_config_b(config_a, args)               # robot-2 (uses --robot-ip-b)
    mask = build_camera_mask(parse_camera_list(args.cameras))
    cams = camera_names(args.cameras)
    color = args.color or env_color()

    src_kw = dict(cameras=cams, stride=args.stride, min_depth=args.min_depth,
                  max_depth=args.max_depth, target=args.target, color=color)

    with SpotCloudSource(config_a, calib_a, mask, stream_id="pair_a", **src_kw) as srcA, \
         SpotCloudSource(config_b, calib_b, mask, stream_id="pair_b", **src_kw) as srcB:
        print(f"robot1 {config_a.robot_ip} order={srcA._order}  |  "
              f"robot2 {config_b.robot_ip} order={srcB._order}  |  "
              f"color={color}  warm_start={not args.no_warm_start}")

        vis, disp = _make_view(args) if args.view else (None, None)
        view_added = False

        T_good = np.eye(4, dtype=np.float32)
        have_lock = False
        last = None                                         # (xyz_a, xyz_b_posed) for --save
        n_locked = 0

        for k in range(args.frames):
            t0 = time.time()
            ca = srcA.latest(timeout=args.timeout)
            cb = srcB.latest(timeout=args.timeout)
            if ca is None or cb is None:
                print(f"[{k:03d}] no points ({'A' if ca is None else ''}{'B' if cb is None else ''})")
                continue
            xyz_a, rgb_a = ca
            xyz_b, rgb_b = cb

            if args.bootstrap_first and not have_lock:
                T_new, info = bootstrap_register(xyz_a, rgb_a, xyz_b, rgb_b, voxel=args.voxel)
                mode = "boot"
            else:
                seed = None if args.no_warm_start else T_good
                T_new, info = colored_icp_register(xyz_a, rgb_a, xyz_b, rgb_b,
                                                   T_init=seed, voxel=args.voxel)
                mode = "icp "

            ov = info["overlap_fitness"]
            accepted = ov >= args.min_overlap
            if accepted:
                T_good = T_new
                have_lock = True
                n_locked += 1
            dt = time.time() - t0
            note = "" if accepted else "  (low overlap, kept previous seed)"
            print(f"[{k:03d}] {mode} fit {info['icp_fitness']:.2f}  rmse {info['icp_rmse']*100:4.1f} cm"
                  f"  overlap {ov*100:3.0f}%  {len(xyz_a):5d}/{len(xyz_b):5d} pts"
                  f"  {dt*1000:5.0f} ms{note}")
            b_posed = apply_T(T_good, xyz_b)
            last = (xyz_a, b_posed)

            if vis is not None:
                alive, view_added = _update_view(vis, disp, view_added, xyz_a, rgb_a,
                                                 b_posed, rgb_b, args.view_rgb)
                if not alive:
                    print("   view window closed — stopping.")
                    break

        print(f"\ndone: {n_locked}/{args.frames} frames accepted (overlap >= {args.min_overlap}).")
        if vis is not None:
            vis.destroy_window()
        if args.save and last is not None:
            save_merged(args.save, last[0], last[1])
            print("   open it — red (robot1) should sit on the same surfaces as blue (robot2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
