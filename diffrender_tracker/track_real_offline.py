"""
§5.2 offline validation on REAL Spot data (runs on Mac/soft or 4060/gsplat — no robot).

Takes a real Spot cloud (cloud_diag/fused.ply, ~42k pts, grayscale), downsamples to ~20k,
and builds a two-robot problem from it: cloud_1 = the cloud, cloud_2 = the same cloud posed
by a known T_gt (2->1) with an exposure gap. Then runs the tracker (run_fit + 3-cam rig) from
a perturbed init and checks it recovers T_gt. This exercises the whole path — real geometry,
real grayscale texture, real point density — before any live streaming code.

Caveats (honest): cloud_2 is a posed COPY of cloud_1, so correspondence is perfect and overlap
is total — easier than two independent robots (different sampling/noise/partial overlap). It
validates ingest + render + loss + convergence on real data; it does NOT prove robustness to
cross-robot sampling differences (that needs live two-robot data).

Real Spot clouds are in the frontleft OPTICAL frame (X right, Y down, Z forward), so the virtual
rig looks along +Z (unlike the synthetic fixture's +Y). Camera placement is scene-dependent
(brief §7.4) — this `forward_rig` is the Spot-frame variant.
"""

import os
import sys

import numpy as np
import torch

from gaussians import cloud_to_gaussians
from se3 import se3_exp, pose_error
from tracker_core import run_fit
from spot_ingest import ply_to_cloud, voxel_downsample, spot_frame_rig, env_color

# Cloud to track: first CLI arg, else $DIFFRENDER_PLY, else the bundled cloud_diag fixture.
# Bridge flow: capture real_cloud.ply on the Mac, then on the 4060:
#   DIFFRENDER_DEVICE=cuda python track_real_offline.py real_cloud.ply
_DEFAULT_PLY = os.path.join(os.path.dirname(__file__), "..", "PySpotObserver",
                            "examples", "cloud_diag", "fused.ply")
PLY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DIFFRENDER_PLY", _DEFAULT_PLY)
DEVICE = os.environ.get("DIFFRENDER_DEVICE", "cpu")


def main():
    if not os.path.exists(PLY):
        print(f"missing {PLY}"); return
    color = env_color()
    xyz, rgb = ply_to_cloud(PLY, grayscale=not color)
    xyz, rgb = voxel_downsample(xyz, rgb, target=20000)
    print(f"real Spot cloud: {xyz.shape[0]} pts ({'rgb' if color else 'grayscale'}), extent "
          f"{np.round(xyz.max(0)-xyz.min(0),2)} m")

    xyz_t = torch.as_tensor(xyz, dtype=torch.float32, device=DEVICE)
    rgb_t = torch.as_tensor(rgb, dtype=torch.float32, device=DEVICE)

    # known ground-truth 2->1 transform, and cloud_2 = T_gt^{-1} @ cloud_1 (so T_gt aligns it)
    twist = torch.tensor([0.12, -0.08, 0.10, *np.deg2rad((6.0, -5.0, 4.0))],
                         dtype=torch.float32, device=DEVICE)
    T_gt = se3_exp(twist)
    R, t = T_gt[:3, :3], T_gt[:3, 3]
    xyz2 = (xyz_t - t) @ R                                 # T_gt^{-1} applied
    rgb2 = torch.clamp(rgb_t * 1.2 - 0.05, 0, 1)          # baked exposure gap -> tests affine

    gA = cloud_to_gaussians(xyz_t, rgb_t, device=DEVICE)
    gB = cloud_to_gaussians(xyz2, rgb2, device=DEVICE)
    rig = spot_frame_rig(xyz_t.cpu().numpy(), xyz2.cpu().numpy(), device=DEVICE)

    # perturbed start ~9deg/14cm from T_gt
    perturb = torch.tensor([0.10, -0.08, 0.06, *np.deg2rad((5.0, 5.0, -6.0))],
                           dtype=torch.float32, device=DEVICE)
    T_init = (T_gt @ se3_exp(perturb)).detach()
    r0, t0 = pose_error(T_init, T_gt)
    print(f"start error: {r0:.2f} deg, {t0*100:.1f} cm")

    T_final, hist = run_fit(gA, gB, T_gt, rig, T_init,
                            pyramid=[(0.25, 70), (0.5, 70), (1.0, 120)],
                            lr=0.02, affine=True, device=DEVICE)
    rf, tf = pose_error(T_final, T_gt)
    print(f"final error: {rf:.3f} deg, {tf*100:.2f} cm   coverage {hist['cov'][-1]*100:.0f}%")
    print(f"ACCEPTANCE (<0.5 deg & <1 cm): {'PASS' if (rf < 0.5 and tf < 0.01) else 'FAIL'}")


if __name__ == "__main__":
    main()
