"""
Milestone 2 §5.1 acceptance gate — run this ON THE 3080 (needs CUDA + gsplat).

Checks the three acceptance criteria for swapping the renderer to gsplat, against the
same synthetic fixture (cloud_A/B.npz) that validated the soft splatter in Milestone 1:

  1. PARITY   — gsplat's render of A matches the soft splatter's over their shared coverage
                (advisory: the two composite differently, so expect close-but-not-identical).
  2. TIMING   — measured gsplat render time per frame (expect ~1-3 ms; the real-time budget).
  3. CONVERGE — fit_pose's cold solve, now running through gsplat, still recovers a 9.3deg/
                14cm perturbation to <0.5deg/1cm. THIS is the hard go/no-go.

Run:  DIFFRENDER_DEVICE unused here; this script drives CUDA directly.
      python test_gsplat_parity.py
"""

import os
import time

import numpy as np
import torch

from gaussians import load_cloud_gaussians
from camera import VirtualCamera
from render import render
from render_gsplat import gsplat_available
from se3 import se3_exp, pose_error
from loss import photometric_loss
from tracker_core import run_fit

DATA = os.path.join(os.path.dirname(__file__), "data")
TAU = 1e-3
PERTURB_ROT_DEG = (5.0, 5.0, -6.0)      # same start offset as fit_pose.py
PERTURB_TRANS = (0.10, -0.08, 0.06)
PYRAMID = [(0.25, 70), (0.5, 70), (1.0, 90)]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    if not gsplat_available():
        print("gsplat_available() is False — run this on the 3080 (CUDA + gsplat required). "
              "Nothing to test on a CUDA-less box.")
        return

    device = "cuda"
    gA = load_cloud_gaussians(os.path.join(DATA, "cloud_A.npz"), device=device)
    gB = load_cloud_gaussians(os.path.join(DATA, "cloud_B.npz"), device=device)
    T_gt = torch.as_tensor(np.load(os.path.join(DATA, "cloud_B.npz"))["T_gt"],
                           dtype=torch.float32, device=device)
    cam = VirtualCamera.place_overlap(gA["means"], gB["means"], device=device)

    # ---------------------------------------------------------------- 1. PARITY
    with torch.no_grad():
        r_soft = render(gA, cam, backend="soft")
        r_gs = render(gA, cam, backend="gsplat")
    mask = (r_soft["alpha"] > TAU) & (r_gs["alpha"] > TAU)
    diff = (r_soft["image"][mask] - r_gs["image"][mask]).abs()
    cov_soft = (r_soft["alpha"] > TAU).float().mean().item() * 100
    cov_gs = (r_gs["alpha"] > TAU).float().mean().item() * 100
    mae = diff.mean().item()
    print("1. PARITY (render of A, soft vs gsplat)")
    print(f"   coverage: soft {cov_soft:.0f}%  gsplat {cov_gs:.0f}%  shared px {int(mask.sum())}")
    print(f"   mean|Δcolor| over shared px: {mae:.4f}   (advisory; <0.05 is a good match)")

    # ---------------------------------------------------------------- 2. TIMING
    with torch.no_grad():
        for _ in range(5):                                   # warm up kernels/allocator
            render(gB, cam, transform=T_gt, backend="gsplat")
        _sync(); t0 = time.time(); N = 100
        for _ in range(N):
            render(gB, cam, transform=T_gt, backend="gsplat")
        _sync(); dt = (time.time() - t0) / N * 1000
    print("2. TIMING (gsplat render, full res)")
    print(f"   {dt:.2f} ms/render  ->  ~{1000/dt:.0f} renders/s single-stream")

    # ---------------------------------------------------------------- 3. CONVERGE
    perturb = torch.tensor([*PERTURB_TRANS, *np.deg2rad(PERTURB_ROT_DEG)],
                           dtype=torch.float32, device=device)
    T_init = (T_gt @ se3_exp(perturb)).detach()
    r0, t0 = pose_error(T_init, T_gt)
    os.environ["DIFFRENDER_BACKEND"] = "gsplat"              # force gsplat inside run_fit
    T_final, hist = run_fit(gA, gB, T_gt, cam, T_init, pyramid=PYRAMID,
                            lr=0.02, affine=True, device=device)
    rf, tf = pose_error(T_final, T_gt)
    ok = (rf < 0.5) and (tf < 0.01)
    print("3. CONVERGE (cold solve through gsplat, coarse-to-fine)")
    print(f"   start {r0:.2f} deg / {t0*100:.1f} cm  ->  final {rf:.3f} deg / {tf*100:.2f} cm")
    print(f"   ACCEPTANCE (<0.5 deg & <1 cm): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
