"""
gsplat divergence diagnostic — run ON THE 4060. Pure text output (no image viewing needed).

The §5.1 parity test showed gsplat renders blurrier than the soft splatter (coverage 38 vs
29%, Δcolor 0.19) and the cold solve DIVERGED (9.3->13 deg). This isolates the cause:

  A. RENDER   — coverage, sharpness, and a flip/orientation check (soft vs gsplat render of A).
                A gross flip would explain divergence outright; low sharpness points at blob size.
  B. CONVERGE — a matrix under gsplat: pyramid vs single-full-res, normal vs sharper blobs
                (smaller Gaussians), and big (9 deg) vs small (3 deg, the warm-start regime)
                start offsets. Whichever rows PASS tell us the fix.

Reader's guide to the matrix:
  * single-scale passes but pyramid fails  -> coarse levels blow up blob size; drop the pyramid.
  * sharper blobs (mult 0.6) pass, 1.5 fails-> Gaussians too big for gsplat; shrink scale.
  * small-offset passes, big fails         -> renderer is fine; only the cold basin shrank
                                              (warm-start tracking, the real use case, still works).
"""

import os

import numpy as np
import torch

os.environ["DIFFRENDER_BACKEND"] = "gsplat"        # force gsplat inside run_fit's render() calls

from gaussians import cloud_to_gaussians            # noqa: E402
from camera import VirtualCamera                    # noqa: E402
from render import render                           # noqa: E402
from render_gsplat import gsplat_available          # noqa: E402
from se3 import se3_exp, pose_error                 # noqa: E402
from tracker_core import run_fit                    # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
TAU = 1e-3
DEVICE = "cuda"


def load(scale_mult):
    a = np.load(os.path.join(DATA, "cloud_A.npz"))
    b = np.load(os.path.join(DATA, "cloud_B.npz"))
    gA = cloud_to_gaussians(a["xyz"], a["rgb"], scale_mult=scale_mult, device=DEVICE)
    gB = cloud_to_gaussians(b["xyz"], b["rgb"], scale_mult=scale_mult, device=DEVICE)
    T_gt = torch.as_tensor(b["T_gt"], dtype=torch.float32, device=DEVICE)
    return gA, gB, T_gt


def sharpness(img):
    g = img.mean(-1)                                # grayscale
    gx = (g[:, 1:] - g[:, :-1]).abs().mean()
    gy = (g[1:, :] - g[:-1, :]).abs().mean()
    return (gx + gy).item()


def section_A(cam, gA):
    print("A. RENDER (soft vs gsplat, cloud A)")
    with torch.no_grad():
        rs = render(gA, cam, backend="soft")
        rg = render(gA, cam, backend="gsplat")
    cs = (rs["alpha"] > TAU).float().mean().item() * 100
    cg = (rg["alpha"] > TAU).float().mean().item() * 100
    print(f"   coverage   soft {cs:4.0f}%   gsplat {cg:4.0f}%")
    print(f"   sharpness  soft {sharpness(rs['image']):.4f}   gsplat {sharpness(rg['image']):.4f}"
          "   (higher = crisper texture; if gsplat << soft, blobs too big)")
    si, gi = rs["image"], rg["image"]
    mae = (si - gi).abs().mean().item()
    mae_ud = (torch.flip(si, [0]) - gi).abs().mean().item()
    mae_lr = (torch.flip(si, [1]) - gi).abs().mean().item()
    print(f"   full-img MAE  as-is {mae:.4f}   flip-vert {mae_ud:.4f}   flip-horiz {mae_lr:.4f}")
    if min(mae_ud, mae_lr) < 0.8 * mae:
        print("   >> a FLIP matches better than as-is: orientation bug (gradient points wrong way).")
    else:
        print("   >> as-is is the best match: no flip. Difference is footprint/blur, not orientation.")


def conv(label, scale_mult, pyramid, rot, trans, affine=True):
    gA, gB, T_gt = load(scale_mult)
    cam = VirtualCamera.place_overlap(gA["means"], gB["means"], device=DEVICE)
    perturb = torch.tensor([*trans, *np.deg2rad(rot)], dtype=torch.float32, device=DEVICE)
    T_init = (T_gt @ se3_exp(perturb)).detach()
    r0, t0 = pose_error(T_init, T_gt)
    Tf, _ = run_fit(gA, gB, T_gt, cam, T_init, pyramid=pyramid, lr=0.02, affine=affine, device=DEVICE)
    rf, tf = pose_error(Tf, T_gt)
    ok = "PASS" if (rf < 0.5 and tf < 0.01) else "FAIL"
    print(f"   [{ok}] {label:38s} {r0:5.1f}deg/{t0*100:4.1f}cm -> {rf:6.2f}deg/{tf*100:6.2f}cm")


def main():
    if not gsplat_available():
        print("gsplat backend not available — run on the 4060.")
        return
    gA, _, _ = load(1.5)
    _, gBtmp, _ = load(1.5)
    cam = VirtualCamera.place_overlap(gA["means"], gBtmp["means"], device=DEVICE)
    section_A(cam, gA)

    BIG = ((5.0, 5.0, -6.0), (0.10, -0.08, 0.06))     # ~9.3 deg / 14 cm  (bootstrap regime)
    SMALL = ((1.5, 1.5, -2.0), (0.03, -0.02, 0.02))    # ~3 deg / 4 cm    (warm-start regime)
    PYR = [(0.25, 70), (0.5, 70), (1.0, 90)]
    FULL = [(1.0, 250)]

    print("\nB. CONVERGE matrix (all through gsplat)")
    conv("pyramid,  mult1.5, from 9deg", 1.5, PYR, *BIG)
    conv("fullres,  mult1.5, from 9deg", 1.5, FULL, *BIG)
    conv("fullres,  mult1.5, from 3deg", 1.5, FULL, *SMALL)
    conv("fullres,  mult0.6, from 9deg", 0.6, FULL, *BIG)
    conv("fullres,  mult0.6, from 3deg", 0.6, FULL, *SMALL)
    conv("pyramid,  mult0.6, from 9deg", 0.6, PYR, *BIG)

    print("\n   (normalize-off control, raw gsplat compositing)")
    os.environ["DIFFRENDER_GSPLAT_NORMALIZE"] = "0"
    conv("fullres,  mult0.6, from 9deg [norm off]", 0.6, FULL, *BIG)
    os.environ["DIFFRENDER_GSPLAT_NORMALIZE"] = "1"


if __name__ == "__main__":
    main()
