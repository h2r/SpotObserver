"""
§5.3 acceptance — BOOTSTRAP -> TRACKING on real Spot data, offline (Mac/soft or 4060/gsplat).

Simulates two independently-posed robots with PARTIAL overlap and a LARGE unknown relative
pose (~45deg/60cm — far outside the renderer's ~10-15deg basin, so the tracker alone cannot
cold-start). Then:
  1. BOOTSTRAP: bootstrap_register(A, B) with NO prior -> coarse T via FPFH+RANSAC+ICP.
     Must land within the tracker's basin (<~10deg/15cm).
  2. HANDOFF:  feed that T to run_fit (the photometric tracker) -> must refine to <0.5deg/1cm.

This is the missing cold-start piece: it proves the renderer, which can't find the first
inter-robot transform, gets a good enough seed from feature matching to take over.

    python test_bootstrap_offline.py [cloud.ply]     # default: cloud_diag/fused.ply
"""

import os
import sys

import numpy as np
import torch

from gaussians import cloud_to_gaussians
from se3 import se3_exp, pose_error
from tracker_core import run_fit
from spot_ingest import ply_to_cloud, voxel_downsample, spot_frame_rig
from bootstrap import bootstrap_register

_DEFAULT_PLY = os.path.join(os.path.dirname(__file__), "..", "PySpotObserver",
                            "examples", "cloud_diag", "fused.ply")
PLY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DIFFRENDER_PLY", _DEFAULT_PLY)
DEVICE = os.environ.get("DIFFRENDER_DEVICE", "cpu")
BASIN_DEG, BASIN_CM = 12.0, 18.0            # bootstrap must land inside this for the tracker


def main():
    if not os.path.exists(PLY):
        print(f"missing {PLY}"); return
    xyz, rgb = ply_to_cloud(PLY, grayscale=True)
    xyz, rgb = voxel_downsample(xyz, rgb, target=20000)
    print(f"scene: {len(xyz)} pts, extent {np.round(xyz.max(0)-xyz.min(0),2)} m")

    # --- build two partially-overlapping robots with a LARGE unknown relative pose ---
    T_gt = se3_exp(torch.tensor([0.30, -0.40, 0.35, *np.deg2rad((20.0, 25.0, -30.0))],
                                dtype=torch.float32)).numpy()      # ~45deg / 62cm (2->1)
    R, t = T_gt[:3, :3], T_gt[:3, 3]

    x = xyz[:, 0]                                                  # split along X for overlap
    lo, hi = np.quantile(x, 0.30), np.quantile(x, 0.70)
    mask_a = x <= hi                                              # robot 1 sees the left ~70%
    mask_b = x >= lo                                              # robot 2 sees the right ~70%
    xyz_a, rgb_a = xyz[mask_a], rgb[mask_a]
    xyz_bw, rgb_b = xyz[mask_b], np.clip(rgb[mask_b] * 1.2 - 0.05, 0, 1)   # + exposure gap
    xyz_b = (xyz_bw - t) @ R                                       # into robot-2's frame
    overlap = int((mask_a & mask_b).sum()) / len(xyz) * 100
    print(f"robot1 {len(xyz_a)} pts, robot2 {len(xyz_b)} pts, scene overlap ~{overlap:.0f}%")

    # --- 1. BOOTSTRAP (no prior) ---
    T_boot, info = bootstrap_register(xyz_a, rgb_a, xyz_b, rgb_b, voxel=0.05)
    rb, tb = pose_error(torch.as_tensor(T_boot), torch.as_tensor(T_gt))
    in_basin = rb < BASIN_DEG and tb * 100 < BASIN_CM
    print("\n1. BOOTSTRAP (FPFH+RANSAC+ICP, no prior)")
    print(f"   ransac fit {info['ransac_fitness']:.2f}  icp fit {info['icp_fitness']:.2f}  "
          f"rmse {info['icp_rmse']*100:.1f} cm  success={info['success']}")
    print(f"   error vs truth: {rb:.2f} deg / {tb*100:.2f} cm   "
          f"-> in basin (<{BASIN_DEG:.0f}deg/{BASIN_CM:.0f}cm)? {in_basin}")
    if not (info["success"] and in_basin):
        print("   BOOTSTRAP FAILED to reach the basin — tracker handoff would not converge.")
        return

    # --- 2. HANDOFF to the photometric tracker ---
    dev = DEVICE
    gA = cloud_to_gaussians(torch.as_tensor(xyz_a, dtype=torch.float32, device=dev),
                            torch.as_tensor(rgb_a, dtype=torch.float32, device=dev), device=dev)
    gB = cloud_to_gaussians(torch.as_tensor(xyz_b, dtype=torch.float32, device=dev),
                            torch.as_tensor(rgb_b, dtype=torch.float32, device=dev), device=dev)
    rig = spot_frame_rig(xyz_a, xyz_b, device=dev)
    T_init = torch.as_tensor(T_boot, dtype=torch.float32, device=dev)
    T_gt_t = torch.as_tensor(T_gt, dtype=torch.float32, device=dev)
    T_final, hist = run_fit(gA, gB, T_gt_t, rig, T_init,
                            pyramid=[(0.25, 70), (0.5, 70), (1.0, 120)], lr=0.02,
                            affine=True, device=dev)
    rf, tf = pose_error(T_final, T_gt_t)
    ok = rf < 0.5 and tf < 0.01
    print("\n2. HANDOFF -> TRACKING (run_fit from the bootstrap seed)")
    print(f"   {rb:.2f}deg/{tb*100:.1f}cm  ->  {rf:.3f}deg/{tf*100:.2f}cm  "
          f"coverage {hist['cov'][-1]*100:.0f}%")
    print(f"   FULL COLD-START (bootstrap+track): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
