"""
Milestone 1, Step 7 — the convergence test (the go/no-go for the whole approach).

Start the pose at a deliberately-perturbed offset from the ground truth, then run
gradient descent on the se(3) twist to minimize the masked photometric loss between
render(B under current pose) and the cached render(A). If the pose error falls back to
~0, differentiable photometric alignment works for these scenes and Milestone 1 passes.

Coarse-to-fine: optimize at low resolution first (wide, smooth basin) then refine at
full resolution — the standard fix for the bumpy photometric landscape.

Outputs data/convergence.png (before / after / target renders + loss & error curves).
"""

import os
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

DATA = os.path.join(os.path.dirname(__file__), "data")

# --------------------------------------------------------------------------- config
PERTURB_ROT_DEG = (5.0, 5.0, -6.0)      # start this far (in rotation) from T_gt ...
PERTURB_TRANS   = (0.10, -0.08, 0.06)   # ... and this far in translation (m)
PYRAMID         = [(0.25, 70), (0.5, 70), (1.0, 90)]   # (resolution scale, iters)
LR              = 0.02
AFFINE          = True                  # correct the exposure gap baked into cloud B


def main():
    torch.manual_seed(0)
    device = os.environ.get("DIFFRENDER_DEVICE", "cpu")   # set "cuda" on the 3080 to exercise gsplat
    gA = load_cloud_gaussians(os.path.join(DATA, "cloud_A.npz"), device=device)
    gB = load_cloud_gaussians(os.path.join(DATA, "cloud_B.npz"), device=device)
    T_gt = torch.as_tensor(np.load(os.path.join(DATA, "cloud_B.npz"))["T_gt"],
                           dtype=torch.float32, device=device)
    cam_full = VirtualCamera.place_overlap(gA["means"], gB["means"], device=device)

    # initial guess = T_gt pushed off by a known perturbation
    perturb = torch.tensor([*PERTURB_TRANS, *np.deg2rad(PERTURB_ROT_DEG)],
                           dtype=torch.float32, device=device)
    T_init = (T_gt @ se3_exp(perturb)).detach()

    xi = torch.zeros(6, device=device, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=LR)

    r0, t0 = pose_error(T_init, T_gt)
    print(f"start error: {r0:.2f} deg, {t0*100:.1f} cm")

    hist = {"loss": [], "rot": [], "trans": [], "cov": []}
    for scale, iters in PYRAMID:
        cam = cam_full if scale >= 1.0 else cam_full.scaled(scale)
        with torch.no_grad():
            targetA = render(gA, cam)
            targetA = {k: v.detach() for k, v in targetA.items()}
        for _ in range(iters):
            opt.zero_grad()
            T = T_init @ se3_exp(xi)
            rB = render(gB, cam, transform=T)
            loss, info = photometric_loss(rB, targetA, affine=AFFINE)
            loss.backward()
            opt.step()
            rot, tr = pose_error((T_init @ se3_exp(xi)).detach(), T_gt)
            hist["loss"].append(loss.item()); hist["rot"].append(rot)
            hist["trans"].append(tr); hist["cov"].append(info["coverage"])

    T_final = (T_init @ se3_exp(xi)).detach()
    rf, tf = pose_error(T_final, T_gt)
    print(f"final error: {rf:.2f} deg, {tf*100:.2f} cm")
    print(f"rotation:    {r0:.2f} -> {rf:.2f} deg   ({r0/max(rf,1e-6):.0f}x reduction)")
    print(f"translation: {t0*100:.1f} -> {tf*100:.2f} cm ({t0/max(tf,1e-9):.0f}x reduction)")

    _plot(gA, gB, cam_full, T_init, T_final, hist, r0, t0, rf, tf)


def _plot(gA, gB, cam, T_init, T_final, hist, r0, t0, rf, tf):
    with torch.no_grad():
        img_tgt = render(gA, cam)["image"].clamp(0, 1).cpu().numpy()
        img_before = render(gB, cam, transform=T_init)["image"].clamp(0, 1).cpu().numpy()
        img_after = render(gB, cam, transform=T_final)["image"].clamp(0, 1).cpu().numpy()

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    ax[0, 0].imshow(img_before); ax[0, 0].set_title(f"BEFORE  ({r0:.1f}deg, {t0*100:.0f}cm off)")
    ax[0, 1].imshow(img_after);  ax[0, 1].set_title(f"AFTER  ({rf:.2f}deg, {tf*100:.1f}cm off)")
    ax[0, 2].imshow(img_tgt);    ax[0, 2].set_title("TARGET (render of A)")
    for a in ax[0]:
        a.set_xticks([]); a.set_yticks([])

    it = range(len(hist["loss"]))
    ax[1, 0].plot(it, hist["loss"]); ax[1, 0].set_title("photometric loss"); ax[1, 0].set_yscale("log")
    ax[1, 1].plot(it, hist["rot"]);  ax[1, 1].set_title("rotation error (deg)"); ax[1, 1].set_yscale("log")
    ax[1, 2].plot(it, hist["trans"]);ax[1, 2].set_title("translation error (m)"); ax[1, 2].set_yscale("log")
    for a in ax[1]:
        a.set_xlabel("iteration"); a.grid(alpha=0.3)
        for b in np.cumsum([n for _, n in PYRAMID])[:-1]:
            a.axvline(b, color="k", ls="--", alpha=0.3)   # pyramid level boundaries

    fig.suptitle("Milestone 1 — differentiable photometric pose convergence", fontsize=12)
    fig.tight_layout()
    out = os.path.join(DATA, "convergence.png")
    fig.savefig(out, dpi=110, facecolor="white")
    print("Saved", out)


if __name__ == "__main__":
    main()
