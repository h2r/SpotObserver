"""
Pure-torch soft splatter — the CPU/MPS fallback renderer (validated in Milestone 1).

A stand-in for gsplat so the whole render -> loss -> backprop -> converge loop runs and
is debuggable locally with no CUDA. Each Gaussian is splatted as a soft blob into a small
pixel window; contributions are accumulated and normalized. Because the blob weight is a
smooth function of the projected center, the image (and loss) changes SMOOTHLY as the pose
moves the means — exactly the gradient a hard z-buffer point renderer can't give you.

Kept behind the identical interface `render_soft(gaussians, camera, transform=None) ->
{"image","alpha","depth"}` so `render.py` can dispatch to it whenever gsplat/CUDA are
unavailable. `transform` is a (4,4) rigid matrix applied to the means ONLY (isotropic
Gaussians); gradients flow through it.

Simplifications vs. gsplat (fine for a corner facing the camera, no self-occlusion):
weighted-average compositing instead of front-to-back alpha, and a capped fixed window.
"""

import torch


def render_soft(gaussians, camera, transform=None, window_half=4,
                sigma_min=0.7, sigma_max=2.5, eps=1e-8):
    means = gaussians["means"]
    colors = gaussians["colors"]
    opac = gaussians["opacities"]
    scale = gaussians["scales"][:, 0]                       # isotropic radius (m)

    if transform is not None:
        means = means @ transform[:3, :3].T + transform[:3, 3]

    u, v, z, valid = camera.project(means)
    u, v, z = u[valid], v[valid], z[valid]
    colors, opac, scale = colors[valid], opac[valid], scale[valid]

    W, H = camera.width, camera.height
    fx = camera.K[0, 0]
    device = means.device

    # metric Gaussian radius -> pixel sigma (shrinks with depth), capped for cost.
    sigma = (fx * scale / z).clamp(sigma_min, sigma_max)    # (M,)

    # fixed integer window around each projected center; weight uses the FLOAT center
    # so gradients flow through u,v (round() only picks which pixels, not the weight).
    offs = torch.arange(-window_half, window_half + 1, device=device)
    du, dv = torch.meshgrid(offs, offs, indexing="xy")      # (K,K)
    cu = u.round().long()[:, None, None]                    # (M,1,1)
    cv = v.round().long()[:, None, None]
    pu = cu + du                                            # (M,K,K) pixel x
    pv = cv + dv                                            # (M,K,K) pixel y

    dx = pu.float() - u[:, None, None]
    dy = pv.float() - v[:, None, None]
    w = opac[:, None, None] * torch.exp(-0.5 * (dx * dx + dy * dy) / sigma[:, None, None] ** 2)

    inb = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
    w = w * inb                                             # zero weight outside the image
    idx = (pv * W + pu).clamp(0, H * W - 1)                 # clamp OOB (weight already 0 there)

    npix = H * W
    color_num = torch.zeros(npix, 3, device=device)
    depth_num = torch.zeros(npix, device=device)
    weight = torch.zeros(npix, device=device)

    idx_f = idx.reshape(-1)
    w_f = w.reshape(-1)
    color_num.index_add_(0, idx_f, (w[..., None] * colors[:, None, None, :]).reshape(-1, 3))
    depth_num.index_add_(0, idx_f, (w * z[:, None, None]).reshape(-1))
    weight.index_add_(0, idx_f, w_f)

    denom = weight.clamp_min(eps)
    image = (color_num / denom[:, None]).reshape(H, W, 3)
    depth = (depth_num / denom).reshape(H, W)
    alpha = weight.reshape(H, W)                            # coverage, for the overlap mask
    return {"image": image, "alpha": alpha, "depth": depth}
