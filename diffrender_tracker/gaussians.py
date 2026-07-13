"""
Milestone 1, Step 2 — colored point cloud -> isotropic 3D Gaussians.

A point cloud becomes a set of Gaussians so it can be *differentiably* rendered
(gsplat). Because the cloud carries no orientation, the Gaussians are ISOTROPIC:
one scalar radius per point, identity rotation. That is the whole reason the
autograd path stays short later — only the Gaussian *means* depend on the pose;
scales/rotations/opacities/colors are fixed constants.

Output is a plain dict of torch tensors, laid out to feed gsplat.rasterization
directly (means, quats, scales, opacities, colors) AND to be transformed by the
pose (only `means` moves).

Everything here is pure torch — runs on CPU/MPS, no CUDA/gsplat needed.
"""

import numpy as np
import torch


def estimate_spacing(means, sample=2000):
    """Median nearest-neighbour distance — a scale-free estimate of point spacing,
    used to size the Gaussians so neighbours just overlap (no holes, no mush)."""
    n = means.shape[0]
    idx = torch.randperm(n, device=means.device)[: min(sample, n)]
    pts = means[idx]
    d = torch.cdist(pts, pts)                      # (m, m)
    d.fill_diagonal_(float("inf"))
    return d.min(dim=1).values.median().item()


def cloud_to_gaussians(xyz, rgb, scale=None, scale_mult=1.0, opacity=0.99,
                       device="cpu", dtype=torch.float32):
    """
    xyz: (N,3) array/tensor of point positions (metres).
    rgb: (N,3) array/tensor of colors in [0,1].
    scale: fixed Gaussian radius (m). If None, derived from point spacing * scale_mult.
    Returns dict with:
        means     (N,3) float  -- the ONLY pose-dependent field
        quats     (N,4) float  -- identity [1,0,0,0]; irrelevant for isotropic
        scales    (N,3) float  -- isotropic: same value in x,y,z
        opacities (N,)  float
        colors    (N,3) float
        spacing   float        -- estimated point spacing (m), for reference
    """
    means = torch.as_tensor(np.asarray(xyz), device=device, dtype=dtype)
    colors = torch.as_tensor(np.asarray(rgb), device=device, dtype=dtype).clamp(0, 1)
    n = means.shape[0]

    spacing = estimate_spacing(means)
    if scale is None:
        scale = spacing * scale_mult

    quats = torch.zeros(n, 4, device=device, dtype=dtype)
    quats[:, 0] = 1.0                                   # identity rotation (w,x,y,z)
    scales = torch.full((n, 3), float(scale), device=device, dtype=dtype)
    opacities = torch.full((n,), float(opacity), device=device, dtype=dtype)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "spacing": spacing,
    }


def load_cloud_gaussians(npz_path, device="cpu", **kw):
    """Convenience: load a cloud_*.npz written by gen_synthetic_pair.py -> Gaussians."""
    d = np.load(npz_path)
    return cloud_to_gaussians(d["xyz"], d["rgb"], device=device, **kw)


if __name__ == "__main__":
    import os
    p = os.path.join(os.path.dirname(__file__), "data", "cloud_A.npz")
    g = load_cloud_gaussians(p)
    print(f"means   {tuple(g['means'].shape)}  {g['means'].dtype}")
    print(f"colors  {tuple(g['colors'].shape)}  range [{g['colors'].min():.2f}, {g['colors'].max():.2f}]")
    print(f"scales  {tuple(g['scales'].shape)}  radius = {g['scales'][0,0]:.4f} m")
    print(f"spacing {g['spacing']:.4f} m  ->  Gaussian radius {g['scales'][0,0]:.4f} m (1.5x)")
    print(f"quats   {tuple(g['quats'].shape)}  first = {g['quats'][0].tolist()} (identity)")
