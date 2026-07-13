"""
Shared fit core — one warm-startable pose solve, reused by the convergence test,
the depth-vs-color ablation (A), the basin sweep (B), and the real-time tracker (C).

run_fit optimizes the se(3) twist so render(B under pose) matches a cached render(A),
with optional coarse-to-fine. Return the final transform and the per-iteration history.
"""

import numpy as np
import torch

from render import render
from se3 import se3_exp, pose_error
from loss import photometric_loss, depth_loss

DEFAULT_PYRAMID = [(0.25, 70), (0.5, 70), (1.0, 90)]


def perturb_twist(rot_deg, trans, device="cpu"):
    return torch.tensor([*trans, *np.deg2rad(rot_deg)], dtype=torch.float32, device=device)


def run_fit(gA, gB, T_gt, cam_full, T_init, pyramid=DEFAULT_PYRAMID, lr=0.02,
            affine=True, loss_mode="color", xi_init=None, device="cpu"):
    """loss_mode: 'color' (photometric) or 'depth' (geometric). Returns (T_final, hist)."""
    xi = (torch.zeros(6, device=device) if xi_init is None else xi_init.clone()).requires_grad_(True)
    opt = torch.optim.Adam([xi], lr=lr)
    hist = {"loss": [], "rot": [], "trans": [], "cov": []}

    for scale, iters in pyramid:
        cam = cam_full if scale >= 1.0 else cam_full.scaled(scale)
        with torch.no_grad():
            target = {k: v.detach() for k, v in render(gA, cam).items()}
        for _ in range(iters):
            opt.zero_grad()
            T = T_init @ se3_exp(xi)
            rB = render(gB, cam, transform=T)
            if loss_mode == "color":
                loss, info = photometric_loss(rB, target, affine=affine)
            else:
                loss, info = depth_loss(rB, target)
            loss.backward()
            opt.step()
            rot, tr = pose_error((T_init @ se3_exp(xi)).detach(), T_gt)
            hist["loss"].append(loss.item()); hist["rot"].append(rot)
            hist["trans"].append(tr); hist["cov"].append(info["coverage"])

    T_final = (T_init @ se3_exp(xi)).detach()
    return T_final, hist
