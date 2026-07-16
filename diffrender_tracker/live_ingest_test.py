#!/usr/bin/env python3
"""
§5.2 LIVE ingest test — run ON THE 4060 WIRED TO A ROBOT.

Validates the streaming path SpotConnection -> stream -> fisheye backprojection -> grayscale
-> voxel-downsample, and (optionally) runs the full gsplat tracker on a freshly-streamed cloud.
Same connection flags as examples/live_pointcloud.py.

Modes:
  (default)   Stream N frames, print points / extent / FPS each frame. Confirms live ingest +
              drop-to-latest at rate.
  --save P    Grab the first good cloud, write it to P (.ply) with grayscale colors, and exit.
  --selftest  Grab one cloud and run the tracker on it (pose it by a known T, perturb, recover
              via gsplat). End-to-end proof on live real data. Needs CUDA + gsplat.

Examples (from the diffrender_tracker/ directory):
    # TUSKER, just watch the ingest rate:
    python live_ingest_test.py --robot-ip 128.148.138.22 --calib calib/spot \
        --user user --password ****** --frames 50

    # end-to-end tracker on a live cloud (on the 4060):
    python live_ingest_test.py --robot-ip 128.148.138.22 --calib calib/spot \
        --user user --password ****** --selftest

    # save a cloud to inspect:
    python live_ingest_test.py --robot-ip 128.148.138.22 --calib calib/spot \
        --user user --password ****** --save live_cloud.ply

Note: --calib paths (calib/spot, calib/spot2) live under PySpotObserver/examples, so either
pass an absolute path or run with that as the working dir; this script adds PySpotObserver and
its examples/ to sys.path so `pyspotobserver` and `common_cli` import regardless of cwd.
"""
from __future__ import annotations

import argparse
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
from spot_ingest import (                                  # noqa: E402  (needs sys.path above)
    SpotCloudSource,
    env_color,
    load_fisheye_calib,
    spot_frame_rig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_connection_arguments(p)
    p.add_argument("--calib", required=True, help="Robot calib dir or calibration.yaml.")
    p.add_argument("--cameras", default="frontleft,frontright", help=argparse.SUPPRESS)
    p.add_argument("--frames", type=int, default=30, help="Frames to stream in the default mode.")
    p.add_argument("--timeout", type=float, default=3.0, help="Per-frame retrieval timeout (s).")
    p.add_argument("--stride", type=int, default=2, help="Pixel subsample step (speed).")
    p.add_argument("--min-depth", type=float, default=0.2)
    p.add_argument("--max-depth", type=float, default=3.0)
    p.add_argument("--target", type=int, default=20000, help="Voxel-downsample point target.")
    p.add_argument("--color", action="store_true",
                   help="Keep CCM-corrected RGB (default collapses to grayscale). Also honours "
                        "env DIFFRENDER_COLOR=1.")
    p.add_argument("--save", type=str, default=None, help="Save first cloud to this .ply and exit.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the gsplat tracker on one live cloud (perturb -> recover).")
    # build_config_from_args reads these two unconditionally.
    p.add_argument("--dumps-enabled", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--save-dir", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def run_selftest(xyz, rgb):
    """Pose the live cloud by a known T_gt, perturb, and recover it with the gsplat tracker."""
    import torch
    from gaussians import cloud_to_gaussians
    from se3 import se3_exp, pose_error
    from tracker_core import run_fit
    from render_gsplat import gsplat_available

    if not gsplat_available():
        print("selftest wants CUDA+gsplat; not available here.")
        return
    dev = "cuda"
    xyz_t = torch.as_tensor(xyz, dtype=torch.float32, device=dev)
    rgb_t = torch.as_tensor(rgb, dtype=torch.float32, device=dev)
    T_gt = se3_exp(torch.tensor([0.12, -0.08, 0.10, *np.deg2rad((6.0, -5.0, 4.0))],
                                dtype=torch.float32, device=dev))
    R, t = T_gt[:3, :3], T_gt[:3, 3]
    xyz2 = (xyz_t - t) @ R
    rgb2 = torch.clamp(rgb_t * 1.2 - 0.05, 0, 1)          # exposure gap -> tests affine
    gA = cloud_to_gaussians(xyz_t, rgb_t, device=dev)
    gB = cloud_to_gaussians(xyz2, rgb2, device=dev)
    rig = spot_frame_rig(xyz, xyz2.cpu().numpy(), device=dev)
    perturb = torch.tensor([0.10, -0.08, 0.06, *np.deg2rad((5.0, 5.0, -6.0))],
                           dtype=torch.float32, device=dev)
    T_init = (T_gt @ se3_exp(perturb)).detach()
    r0, t0 = pose_error(T_init, T_gt)
    print(f"  selftest start {r0:.2f} deg / {t0*100:.1f} cm")
    T_final, hist = run_fit(gA, gB, T_gt, rig, T_init,
                            pyramid=[(0.25, 70), (0.5, 70), (1.0, 120)], lr=0.02,
                            affine=True, device=dev)
    rf, tf = pose_error(T_final, T_gt)
    ok = rf < 0.5 and tf < 0.01
    print(f"  selftest final {rf:.3f} deg / {tf*100:.2f} cm  coverage {hist['cov'][-1]*100:.0f}%"
          f"  -> {'PASS' if ok else 'FAIL'}")


def save_ply(path, xyz, rgb):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(path, pc)
    print(f"wrote {path} ({len(xyz)} pts)")


def main() -> int:
    args = parse_args()
    calib_path = args.calib                              # accept calib/spot from any cwd
    if not os.path.exists(calib_path):
        alt = os.path.join(PSO, "examples", args.calib)
        if os.path.exists(alt):
            calib_path = alt
    calib = load_fisheye_calib(calib_path)
    config = build_config_from_args(args)
    mask = build_camera_mask(parse_camera_list(args.cameras))

    color = args.color or env_color()
    with SpotCloudSource(config, calib, mask, cameras=parse_camera_list_names(args.cameras),
                         stride=args.stride, min_depth=args.min_depth,
                         max_depth=args.max_depth, target=args.target, color=color) as src:
        print(f"streaming; order = {src._order}  color = {color}")

        if args.save or args.selftest:
            cloud = None
            for _ in range(20):                            # wait for a frame with real points
                cloud = src.latest(timeout=args.timeout)
                if cloud is not None and len(cloud[0]) > 500:
                    break
            if cloud is None or len(cloud[0]) <= 500:
                print("no usable cloud received."); return 1
            xyz, rgb = cloud
            chan = "rgb" if color else "gray"
            print(f"cloud: {len(xyz)} pts, extent {np.round(xyz.max(0)-xyz.min(0),2)} m, "
                  f"{chan} range [{rgb.min():.2f},{rgb.max():.2f}]  "
                  f"chroma spread {float(rgb.max(1).mean()-rgb.min(1).mean()):.3f}")
            if args.save:
                save_ply(args.save, xyz, rgb)
            if args.selftest:
                run_selftest(xyz, rgb)
            return 0

        # default: measure live ingest rate
        n_ok = 0
        for k in range(args.frames):
            t0 = time.time()
            cloud = src.latest(timeout=args.timeout)
            dt = time.time() - t0
            if cloud is None:
                print(f"[{k:03d}] no points"); continue
            xyz, _ = cloud
            n_ok += 1
            print(f"[{k:03d}] {len(xyz):6d} pts  extent {np.round(xyz.max(0)-xyz.min(0),2)} m  "
                  f"{dt*1000:5.0f} ms  ({1.0/max(dt,1e-6):4.1f} fps)")
        print(f"done: {n_ok}/{args.frames} frames had points.")
    return 0


def parse_camera_list_names(spec: str):
    """common_cli.parse_camera_list may return CameraType enums; SpotCloudSource wants names to
    match against get_camera_order(). Normalize to lowercase strings either way."""
    out = []
    for c in parse_camera_list(spec):
        out.append(c.name.lower() if hasattr(c, "name") else str(c).lower())
    return out


if __name__ == "__main__":
    raise SystemExit(main())
