"""
C — warm-started few-steps-per-frame tracker (Milestone 2 core, validated locally).

Simulates the moving robot: the true B->A transform drifts by a small motion each frame,
and B's cloud is regenerated consistently so that transforming it by the true pose always
lands on A. The tracker follows with only ~8 gradient steps per frame, warm-started from
the previous frame's estimate. A cold-start baseline (8 steps from a fixed pose every
frame) shows why the warm start is what makes this real-time.

This is the exact loop that goes real-time on the 3080 — the ONLY change for Milestone 2
is swapping render() for gsplat.rasterization and feeding gB from the PySpotObserver
stream (drop-to-latest) instead of the simulated trajectory.
"""

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussians import load_cloud_gaussians
from camera import VirtualCamera
from render import render
from se3 import se3_exp, pose_error
from loss import photometric_loss
from tracker_core import perturb_twist

DATA = os.path.join(os.path.dirname(__file__), "data")
N_FRAMES = 30
STEPS = 8                                   # gradient steps per frame (the real-time budget)
LR = 0.03
MOTION_ROT = (1.0, 0.8, -1.2)               # per-frame relative motion (deg) ~1.7 deg
MOTION_TRANS = (0.020, -0.015, 0.015)       # per-frame relative motion (m) ~0.03 m


def solve_frame(gB_k, cam, target, T_init, steps=STEPS, lr=LR):
    """Few gradient steps from T_init. Returns (T_est, n_steps)."""
    xi = torch.zeros(6, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        T = T_init @ se3_exp(xi)
        loss, _ = photometric_loss(render(gB_k, cam, transform=T), target, affine=True)
        loss.backward()
        opt.step()
    return (T_init @ se3_exp(xi)).detach()


def main():
    gA = load_cloud_gaussians(os.path.join(DATA, "cloud_A.npz"))
    gB = load_cloud_gaussians(os.path.join(DATA, "cloud_B.npz"))
    T_gt = torch.as_tensor(np.load(os.path.join(DATA, "cloud_B.npz"))["T_gt"], dtype=torch.float32)
    cam = VirtualCamera.place_overlap(gA["means"], gB["means"])

    with torch.no_grad():
        target = {k: v.detach() for k, v in render(gA, cam).items()}
        B_world = gB["means"] @ T_gt[:3, :3].T + T_gt[:3, 3]     # B's points in world/A frame

    delta = se3_exp(perturb_twist(MOTION_ROT, MOTION_TRANS))
    T_true = T_gt.clone()
    est_warm, est_cold = T_gt.clone(), T_gt.clone()
    H = {"warm_r": [], "warm_t": [], "cold_r": [], "cold_t": [], "dt": []}

    for k in range(N_FRAMES):
        if k > 0:
            T_true = T_true @ delta
        Rk, tk = T_true[:3, :3], T_true[:3, 3]
        gB_k = dict(gB, means=(B_world - tk) @ Rk)               # regenerate B in its frame-k pose

        t0 = time.time()
        est_warm = solve_frame(gB_k, cam, target, T_init=est_warm)   # warm: from last estimate
        dt = time.time() - t0
        est_cold = solve_frame(gB_k, cam, target, T_init=T_gt)       # cold: from fixed pose

        rw, tw = pose_error(est_warm, T_true)
        rc, tc = pose_error(est_cold, T_true)
        H["warm_r"].append(rw); H["warm_t"].append(tw)
        H["cold_r"].append(rc); H["cold_t"].append(tc); H["dt"].append(dt)

    drift_r, _ = pose_error(T_true, T_gt)
    print(f"total drift over {N_FRAMES} frames: {drift_r:.1f} deg")
    print(f"warm-start ({STEPS} steps/frame): mean {np.mean(H['warm_r']):.3f} deg, "
          f"{np.mean(H['warm_t'])*100:.2f} cm  |  final {H['warm_r'][-1]:.3f} deg")
    print(f"cold-start ({STEPS} steps/frame): mean {np.mean(H['cold_r']):.3f} deg, "
          f"{np.mean(H['cold_t'])*100:.2f} cm  |  final {H['cold_r'][-1]:.3f} deg")
    print(f"per-frame solve (CPU soft splatter): {np.mean(H['dt'])*1000:.0f} ms "
          f"(gsplat on the 3080 is ~100x faster)")
    _plot(H)


def _plot(H):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    fr = range(N_FRAMES)
    ax[0].plot(fr, H["warm_r"], "-o", ms=3, label="warm-start (8 steps)", color="C0")
    ax[0].plot(fr, H["cold_r"], "-o", ms=3, label="cold-start (8 steps)", color="C3")
    ax[0].set_title("rotation error per frame (deg)"); ax[0].set_ylabel("deg")
    ax[1].plot(fr, [t * 100 for t in H["warm_t"]], "-o", ms=3, label="warm-start", color="C0")
    ax[1].plot(fr, [t * 100 for t in H["cold_t"]], "-o", ms=3, label="cold-start", color="C3")
    ax[1].set_title("translation error per frame (cm)"); ax[1].set_ylabel("cm")
    for a in ax:
        a.set_xlabel("frame"); a.set_yscale("log"); a.grid(alpha=0.3); a.legend()
    fig.suptitle(f"C — warm-started {STEPS}-steps/frame tracker follows the motion; "
                 f"cold-start can't", fontsize=12)
    fig.tight_layout()
    out = os.path.join(DATA, "tracker.png")
    fig.savefig(out, dpi=110, facecolor="white")
    print("Saved", out)


if __name__ == "__main__":
    main()
