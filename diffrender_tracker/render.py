"""
Differentiable render — backend dispatcher (Milestone 1 soft splatter + Milestone 2 gsplat).

`render(gaussians, camera, transform=None) -> {"image","alpha","depth"}` is THE stable
interface every downstream module (loss, tracker, tests) depends on — swap implementations,
never this signature (brief §7.5). `transform` is a (4,4) rigid matrix applied to the
Gaussian means only; gradients flow through it.

Two backends behind that one entry point:
  * "gsplat" (render_gsplat.py) — CUDA, real-time, the production path on the 3080.
  * "soft"   (render_soft.py)   — pure-torch soft splatter, the CPU/MPS fallback/test path
                                  that was validated in Milestone 1 (brief §7.1).

Selection order: explicit `backend=` arg  >  env DIFFRENDER_BACKEND  >  auto (gsplat when a
CUDA device + gsplat are present, else soft). So `from render import render` transparently
uses gsplat on the 3080 and the soft splatter on the Mac, with no code change downstream.
"""

import os

import torch

from render_soft import render_soft

_VALID = ("gsplat", "soft")
_auto_cache = None


def _resolve_backend(backend):
    if backend is not None:
        if backend not in _VALID:
            raise ValueError(f"backend must be one of {_VALID}, got {backend!r}")
        return backend
    env = os.environ.get("DIFFRENDER_BACKEND")
    if env:
        env = env.strip().lower()
        if env not in _VALID:
            raise ValueError(f"DIFFRENDER_BACKEND must be one of {_VALID}, got {env!r}")
        return env
    return _auto_backend()


def _auto_backend():
    """gsplat if it's importable AND CUDA is present, else the soft splatter. Cached."""
    global _auto_cache
    if _auto_cache is None:
        try:
            from render_gsplat import gsplat_available
            _auto_cache = "gsplat" if gsplat_available() else "soft"
        except Exception:
            _auto_cache = "soft"
    return _auto_cache


def render(gaussians, camera, transform=None, backend=None, **kw):
    """Render `gaussians` from `camera` (optionally posed by `transform`) via the selected
    backend. Returns {"image":(H,W,3), "alpha":(H,W), "depth":(H,W)}."""
    if _resolve_backend(backend) == "gsplat":
        from render_gsplat import render_gsplat
        return render_gsplat(gaussians, camera, transform, **kw)
    return render_soft(gaussians, camera, transform, **kw)


# ------------------------------------------------------------------------- demo / check
if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gaussians import load_cloud_gaussians
    from camera import VirtualCamera

    print("auto backend:", _auto_backend())
    data = os.path.join(os.path.dirname(__file__), "data")
    ga = load_cloud_gaussians(os.path.join(data, "cloud_A.npz"))
    gb = load_cloud_gaussians(os.path.join(data, "cloud_B.npz"))
    T_gt = torch.as_tensor(np.load(os.path.join(data, "cloud_B.npz"))["T_gt"], dtype=torch.float32)
    cam = VirtualCamera.place_overlap(ga["means"], gb["means"])

    rA = render(ga, cam)
    rB = render(gb, cam)
    rBa = render(gb, cam, transform=T_gt)

    for name, r in [("A", rA), ("B stored", rB), ("T_gt*B", rBa)]:
        cov = (r["alpha"] > 1e-3).float().mean().item() * 100
        print(f"render {name:9s}: image {tuple(r['image'].shape)}  "
              f"coverage {cov:.0f}%  depth {r['depth'][r['alpha']>1e-3].mean():.2f} m")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, (name, r) in zip(ax, [("A (reference)", rA),
                                 ("B (misaligned)", rB),
                                 ("T_gt*B (target)", rBa)]):
        a.imshow(r["image"].clamp(0, 1).numpy()); a.set_title(name, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("differentiable render check", fontsize=11)
    fig.tight_layout()
    out = os.path.join(data, "preview_render.png")
    fig.savefig(out, dpi=110, facecolor="white")
    print("Saved", out)
