"""
gsplat-backed differentiable render (Milestone 2, §5.1) — the real-time production path.

Same signature and return dict as the pure-torch soft splatter:
    render_gsplat(gaussians, camera, transform=None) -> {"image","alpha","depth"}
so it drops in behind `render.py`'s dispatcher with zero downstream changes. `transform`
is a (4,4) rigid matrix applied to the Gaussian MEANS only (isotropic Gaussians), and
gradients flow through it — the short autograd path `se(3) -> exp -> means -> render -> loss`
the tracker backprops through.

gsplat is CUDA-only, so this runs on the 3080, not the Mac dev box. The gaussians dict
(means/quats/scales/opacities/colors) already matches gsplat.rasterization's arg layout;
the camera is OpenCV-convention so K/viewmat drop straight in.

Compositing note (kept faithful to the validated soft splatter): gsplat's RGB and D
channels are alpha-ACCUMULATED (sum of transmittance*alpha*value), not normalized. The
soft splatter returns the alpha-NORMALIZED expected color/depth (color_num/weight). To
match that semantics exactly we divide both the color and depth channels by alpha here.
Loss/health-check thresholds calibrated on the soft splatter therefore transfer directly.
"""

import torch

_rasterization = None


def _get_rasterization():
    """Lazy import so this module can be imported on a CUDA-less box (it just won't run)."""
    global _rasterization
    if _rasterization is None:
        from gsplat import rasterization
        _rasterization = rasterization
    return _rasterization


def gsplat_available():
    """True only if gsplat imports AND a CUDA device exists (gsplat is CUDA-only)."""
    if not torch.cuda.is_available():
        return False
    try:
        _get_rasterization()
        return True
    except Exception:
        return False


def render_gsplat(gaussians, camera, transform=None, eps=1e-8,
                  near_plane=0.01, far_plane=1e10):
    rasterization = _get_rasterization()

    means = gaussians["means"]
    quats = gaussians["quats"]
    scales = gaussians["scales"]
    opacities = gaussians["opacities"]
    colors = gaussians["colors"]

    if transform is not None:                                  # pose acts on means only
        means = means @ transform[:3, :3].T + transform[:3, 3]

    device = means.device
    # camera tensors follow the gaussians onto the (CUDA) device; batch dim of 1 camera.
    viewmats = camera.viewmat.to(device=device, dtype=means.dtype)[None]   # (1,4,4) world->cam
    Ks = camera.K.to(device=device, dtype=means.dtype)[None]               # (1,3,3)
    W, H = camera.width, camera.height

    # colors is (N,3) precomputed RGB -> sh_degree=None (used directly, no SH eval).
    render_colors, render_alphas, _ = rasterization(
        means=means.contiguous(),
        quats=quats.contiguous(),
        scales=scales.contiguous(),
        opacities=opacities.contiguous(),
        colors=colors.contiguous(),
        viewmats=viewmats.contiguous(),
        Ks=Ks.contiguous(),
        width=W,
        height=H,
        render_mode="RGB+D",
        near_plane=near_plane,
        far_plane=far_plane,
        packed=False,
    )

    out = render_colors[0]                     # (H,W,4): RGB (accumulated) + D (accumulated)
    alpha = render_alphas[0, ..., 0]           # (H,W): accumulated coverage
    denom = alpha.clamp_min(eps)
    image = out[..., :3] / denom[..., None]    # -> expected color (matches soft splatter)
    depth = out[..., 3] / denom                # -> expected depth  (matches soft splatter)
    return {"image": image, "alpha": alpha, "depth": depth}
