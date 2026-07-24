"""
Milestone 1, Step 6 — masked photometric loss.

Compares the moving render to the cached target render and returns a single scalar the
optimizer minimizes. Key choices, matching the plan:

  * COLOR, not depth. Your scenes are geometrically degenerate, so a depth loss can't
    lock the in-plane DoF; the wall/floor texture can. Loss is on the RGB image.
  * MASKED to the overlap. Only pixels both renders cover (alpha > tau) contribute, so
    the non-overlapping wings don't fight the alignment.
  * HUBER, not L2. Robust to the few mismatched pixels at silhouette edges / partial
    samples.
  * Optional AFFINE brightness correction. Two robots = two exposures. Fit per-channel
    gain+bias (closed form, detached) so a raw-intensity difference isn't mistaken for
    a pose error. Turn on when the exposure gap (PHOTOMETRIC_MISMATCH) is present.
"""

import torch


def _affine_match(src, tgt):
    """Per-channel gain a, bias b (least squares) mapping src -> tgt over given pixels.
    Detached: it removes an exposure gap, it should not supply pose gradient itself."""
    with torch.no_grad():
        s = src.reshape(-1, src.shape[-1])
        t = tgt.reshape(-1, t_shape := tgt.shape[-1])
        a = torch.ones(t_shape, device=src.device, dtype=src.dtype)
        b = torch.zeros(t_shape, device=src.device, dtype=src.dtype)
        for c in range(t_shape):
            sc, tc = s[:, c], t[:, c]
            var = sc.var(unbiased=False)
            if var > 1e-8:
                a[c] = ((sc * tc).mean() - sc.mean() * tc.mean()) / var
                b[c] = tc.mean() - a[c] * sc.mean()
    return a, b


def photometric_loss(moving, target, tau=1e-3, huber_delta=0.1, affine=False):
    """
    moving, target: dicts from render() with 'image' (H,W,3) and 'alpha' (H,W).
    Returns (loss_scalar, info_dict). `target` is expected to be detached (cached).
    """
    mask = (moving["alpha"] > tau) & (target["alpha"] > tau)     # overlap only
    coverage = mask.float().mean()
    if mask.sum() < 16:                                          # essentially no overlap
        # return a differentiable zero-grad-safe large loss + the no-overlap signal
        return (moving["image"].sum() * 0.0 + 1e3), {
            "coverage": coverage.item(), "n_pixels": int(mask.sum().item()),
            "overlap_ok": False}

    mimg = moving["image"][mask]                                 # (M,3)
    timg = target["image"][mask]

    if affine:
        a, b = _affine_match(mimg, timg)
        mimg = mimg * a + b

    resid = mimg - timg
    absr = resid.abs()
    quad = 0.5 * (resid ** 2)
    lin = huber_delta * (absr - 0.5 * huber_delta)
    huber = torch.where(absr <= huber_delta, quad, lin)
    loss = huber.mean()

    return loss, {"coverage": coverage.item(), "n_pixels": int(mask.sum().item()),
                  "overlap_ok": True, "rmse": resid.pow(2).mean().sqrt().item()}


def depth_loss(moving, target, tau=1e-3, huber_delta=0.05):
    """Masked Huber on DEPTH — the geometry-only counterpart, for the ablation. On a
    fronto-parallel wall/floor this is (correctly) near-blind to in-plane sliding, which
    is exactly what we expect it to fail to lock."""
    mask = (moving["alpha"] > tau) & (target["alpha"] > tau)
    coverage = mask.float().mean()
    if mask.sum() < 16:
        return (moving["depth"].sum() * 0.0 + 1e3), {
            "coverage": coverage.item(), "overlap_ok": False}
    resid = moving["depth"][mask] - target["depth"][mask]
    absr = resid.abs()
    huber = torch.where(absr <= huber_delta, 0.5 * resid ** 2,
                        huber_delta * (absr - 0.5 * huber_delta))
    return huber.mean(), {"coverage": coverage.item(), "overlap_ok": True,
                          "rmse": resid.pow(2).mean().sqrt().item()}
