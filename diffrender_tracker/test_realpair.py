"""
REAL two-robot alignment — bootstrap -> track -> merge, on two actual Spot clouds.

Unlike the self-tests (posed copies), this takes two clouds from two DIFFERENT robots — genuine
independent sampling, noise, and partial overlap — with NO ground-truth transform. So success is
judged without pose error:
  * bootstrap RANSAC/ICP fitness,
  * robot1->robot2 overlap = fraction of the SMALLER robot1 cloud landing within 5cm of robot2
    after alignment. (robot2 usually sees MORE of the room, so robot2->robot1 is low regardless
    of alignment — robot1->robot2 is the honest signal.)
  * a merged cloud saved to .ply, TINTED by robot (robot1 red, robot2 blue) — open it and see if
    the two colors sit on the same surfaces (aligned) or are offset (misaligned).

A §4.4-style health check gates the tracker: a photometric refine that REDUCES overlap is
rejected and we keep the bootstrap alignment (the tracker can wander on partial-overlap grayscale).

    python test_realpair.py robot1.ply robot2.ply     # DIFFRENDER_DEVICE=cuda on the GPU box
"""

import os
import sys

import numpy as np
import torch

from gaussians import cloud_to_gaussians
from tracker_core import run_fit
from spot_ingest import ply_to_cloud, voxel_downsample, spot_frame_rig, env_color
from bootstrap import bootstrap_register

DEVICE = os.environ.get("DIFFRENDER_DEVICE", "cpu")
COLOR = env_color()


def _o3d(xyz):
    import open3d as o3d
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(xyz, np.float64))
    return p


def apply_T(T, xyz):
    T = np.asarray(T, np.float32)
    return xyz @ T[:3, :3].T + T[:3, 3]


def frac_within(target_xyz, query_xyz, d=0.05):
    """Fraction of query points within `d` of any target point (nearest-neighbour, vectorised)."""
    dists = np.asarray(_o3d(query_xyz).compute_point_cloud_distance(_o3d(target_xyz)))
    return float((dists < d).mean()) if len(dists) else 0.0


def save_merged(path, a_xyz, b_posed_xyz):
    import open3d as o3d
    red = np.tile([0.9, 0.2, 0.2], (len(a_xyz), 1))
    blue = np.tile([0.2, 0.4, 0.9], (len(b_posed_xyz), 1))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.vstack([a_xyz, b_posed_xyz]).astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.vstack([red, blue]))
    o3d.io.write_point_cloud(path, pc)
    print(f"   saved {path}  (robot1=red, robot2=blue)")


def main():
    if len(sys.argv) < 3:
        print("usage: python test_realpair.py robot1.ply robot2.ply"); return
    a_ply, b_ply = sys.argv[1], sys.argv[2]
    xyz_a, rgb_a = voxel_downsample(*ply_to_cloud(a_ply, grayscale=not COLOR), target=20000)
    xyz_b, rgb_b = voxel_downsample(*ply_to_cloud(b_ply, grayscale=not COLOR), target=20000)
    print(f"robot1 {len(xyz_a)} pts, robot2 {len(xyz_b)} pts  ({'rgb' if COLOR else 'grayscale'})")

    # honest metric: fraction of the smaller robot1 cloud explained by robot2 after alignment.
    def r1_in_r2(T):
        bp = apply_T(T, xyz_b)
        return frac_within(bp, xyz_a), bp

    s_id, _ = r1_in_r2(np.eye(4))
    print(f"robot1->robot2 overlap, unaligned: {s_id*100:.0f}%")

    # --- 1. BOOTSTRAP (no prior) ---
    T_boot, info = bootstrap_register(xyz_a, rgb_a, xyz_b, rgb_b, voxel=0.05)
    s_boot, b_boot = r1_in_r2(T_boot)
    print("\n1. BOOTSTRAP (FPFH + best-of-N RANSAC + multiscale colored ICP + identity guard)")
    print(f"   ransac fit {info['ransac_fitness']:.2f}  icp fit {info['icp_fitness']:.2f}  "
          f"rmse {info['icp_rmse']*100:.1f} cm  -> chose {info.get('chosen','icp')}"
          + ("  (guard fired: refine was worse than identity)" if info.get('chosen') == 'identity' else ""))
    print(f"   robot1->robot2 overlap: {s_boot*100:.0f}%")

    # --- 2. TRACK (photometric refine from the bootstrap seed) ---
    dev = DEVICE
    gA = cloud_to_gaussians(torch.as_tensor(xyz_a, dtype=torch.float32, device=dev),
                            torch.as_tensor(rgb_a, dtype=torch.float32, device=dev), device=dev)
    gB = cloud_to_gaussians(torch.as_tensor(xyz_b, dtype=torch.float32, device=dev),
                            torch.as_tensor(rgb_b, dtype=torch.float32, device=dev), device=dev)
    rig = spot_frame_rig(xyz_a, b_boot, device=dev)                # frame the aligned overlap
    T_init = torch.as_tensor(T_boot, dtype=torch.float32, device=dev)
    # run_fit uses T_gt only for its (here meaningless) error log; pass the seed as a stand-in.
    T_track, hist = run_fit(gA, gB, T_init, rig, T_init,
                            pyramid=[(0.25, 70), (0.5, 70), (1.0, 120)], lr=0.02,
                            affine=True, device=dev)
    s_track, b_track = r1_in_r2(T_track.cpu().numpy())
    print("\n2. TRACK (photometric refine)")
    print(f"   final loss {hist['loss'][-1]:.4f}  coverage {hist['cov'][-1]*100:.0f}%")
    print(f"   robot1->robot2 overlap: {s_track*100:.0f}%")

    # --- health-check gate (§4.4): a refine that REDUCES overlap is rejected ---
    if s_track >= s_boot:
        which, b_final, s_final = "tracked", b_track, s_track
    else:
        which, b_final, s_final = "bootstrap", b_boot, s_boot
        print("   -> tracker reduced overlap; REJECTED (health check), keeping bootstrap.")
    save_merged("merged_final.ply", xyz_a, b_final)
    print(f"\nFINAL: {which} alignment, robot1->robot2 overlap {s_final*100:.0f}% "
          f"(unaligned was {s_id*100:.0f}%)")
    print("Open merged_final.ply — red (robot1) should sit on the same surfaces as blue (robot2).")


if __name__ == "__main__":
    main()
