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

Compositing (tuned on the 4060 via diag_gsplat.py, NOT blindly matched to the soft
splatter): gsplat's RGB/D channels are alpha-ACCUMULATED (sum of transmittance*alpha*value).
  * COLOR is left RAW (accumulated, composited over black) — this is what the pose optimizer
    minimizes. Alpha-NORMALIZING it (color/alpha = "expected color") looked more faithful to
    the soft splatter but injected high-frequency noise at low-alpha silhouette pixels that
    trapped the solver: from 9deg it stalled at ~7deg normalized vs converged to ~0.2deg raw.
    So we optimize on the smooth raw-composited image. Env DIFFRENDER_GSPLAT_NORMALIZE=1
    restores the old normalized-color behavior for comparison.
  * DEPTH is alpha-NORMALIZED (expected depth) — it's only a validator (§4.3), never in the
    optimizer path, so we keep it in metric units.
"""

import os

import torch

_rasterization = None


def _normalize_color():
    """Default False: optimize on RAW gsplat compositing (smoother pose landscape). Set env
    DIFFRENDER_GSPLAT_NORMALIZE=1 to divide color by alpha (old expected-color behavior)."""
    return os.environ.get("DIFFRENDER_GSPLAT_NORMALIZE", "0").strip().lower() in ("1", "true", "yes")


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
    image = out[..., :3] / denom[..., None] if _normalize_color() else out[..., :3]
    depth = out[..., 3] / denom                # expected (metric) depth — validator only
    return {"image": image, "alpha": alpha, "depth": depth}
