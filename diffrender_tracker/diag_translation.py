"""
Translation-floor diagnostic — run ON THE 4060.

diag_gsplat showed gsplat nails rotation (~0.3deg) but translation stalls at a ~2.5cm
FLOOR from every start/scale. Constant-regardless-of-start => a biased minimum, not slow
convergence. Hypothesis: the residual lies along the camera OPTICAL AXIS (depth), which a
photometric loss on a fronto-parallel surface barely constrains. Test both claims:

  A. DIRECTION — solve with one camera, decompose the residual translation into that
                 camera's axes (right / down / DEPTH). If depth dominates, hypothesis holds.
  B. FIX       — re-solve with 1, 2, 3 virtual cameras at different azimuths (the depth axis
                 of one view is the in-plane axis of another). If translation drops <1cm with
                 more views, multi-camera is the production fix (brief §7.4).
  C. control   — 1 camera with many more iters, to rule out "just needs more steps".
"""

import os

import numpy as np
import torch

os.environ["DIFFRENDER_BACKEND"] = "gsplat"

from gaussians import cloud_to_gaussians          # noqa: E402
from camera import VirtualCamera                  # noqa: E402
from render import render                         # noqa: E402
from render_gsplat import gsplat_available        # noqa: E402
from se3 import se3_exp, pose_error               # noqa: E402
from loss import photometric_loss                 # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
DEVICE = "cuda"
SCALE = 1.0


def load():
    a = np.load(os.path.join(DATA, "cloud_A.npz"))
    b = np.load(os.path.join(DATA, "cloud_B.npz"))
    gA = cloud_to_gaussians(a["xyz"], a["rgb"], scale_mult=SCALE, device=DEVICE)
    gB = cloud_to_gaussians(b["xyz"], b["rgb"], scale_mult=SCALE, device=DEVICE)
    T_gt = torch.as_tensor(b["T_gt"], dtype=torch.float32, device=DEVICE)
    return gA, gB, T_gt


def overlap_geometry(gA, gB):
    a, b = gA["means"].detach().cpu(), gB["means"].detach().cpu()
    lo = torch.maximum(a.min(0).values, b.min(0).values)
    hi = torch.minimum(a.max(0).values, b.max(0).values)
    center = 0.5 * (lo + hi)
    extent = (hi - lo).max().item()
    return center, extent


def make_cameras(gA, gB, azimuths_deg, margin=1.6, fov=60.0):
    """Cameras ringed around the overlap centroid at the given azimuths (0deg == the
    original place_overlap view along +y). All aim at the centroid."""
    center, extent = overlap_geometry(gA, gB)
    cams = []
    for az in azimuths_deg:
        th = np.deg2rad(az)
        d = np.array([np.sin(th), np.cos(th), 0.0])          # 0deg -> +y, like place_overlap
        eye = center.numpy() + margin * extent * d
        eye[2] = center[2].item() + 0.3 * extent
        cams.append(VirtualCamera.look_at(eye.tolist(), center.tolist(), up=(0, 0, 1),
                                           fov_deg=fov, width=640, height=480, device=DEVICE))
    return cams


def solve(gA, gB, T_gt, cams, T_init, steps=400, lr=0.03, affine=True):
    xi = torch.zeros(6, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    with torch.no_grad():
        targets = [{k: v.detach() for k, v in render(gA, c).items()} for c in cams]
    for _ in range(steps):
        opt.zero_grad()
        T = T_init @ se3_exp(xi)
        loss = 0.0
        for c, tgt in zip(cams, targets):
            l, _ = photometric_loss(render(gB, c, transform=T), tgt, affine=affine)
            loss = loss + l
        loss.backward()
        opt.step()
    return (T_init @ se3_exp(xi)).detach()


def init_from(T_gt, rot, trans):
    p = torch.tensor([*trans, *np.deg2rad(rot)], dtype=torch.float32, device=DEVICE)
    return (T_gt @ se3_exp(p)).detach()


def main():
    if not gsplat_available():
        print("run on the 4060.")
        return
    gA, gB, T_gt = load()
    MID = ((3.0, 3.0, -3.5), (0.05, -0.04, 0.04))     # ~5 deg / 8 cm
    BIG = ((5.0, 5.0, -6.0), (0.10, -0.08, 0.06))     # ~9 deg / 14 cm

    # ---- A. direction of the residual (single camera) ----
    cam0 = make_cameras(gA, gB, [0.0])[0]
    T_init = init_from(T_gt, *MID)
    T_fin = solve(gA, gB, T_gt, [cam0], T_init, steps=400)
    T_err = torch.linalg.inv(T_gt) @ T_fin
    t_err_world = T_err[:3, 3]
    R_wc = cam0.viewmat[:3, :3]                        # world -> cam
    t_cam = (R_wc @ t_err_world).abs()                 # residual in cam axes
    rf, tf = pose_error(T_fin, T_gt)
    print("A. RESIDUAL DIRECTION (1 cam, from 5deg)")
    print(f"   final {rf:.2f}deg / {tf*100:.2f}cm")
    print(f"   residual in camera axes:  right {t_cam[0]*100:5.2f}cm   "
          f"down {t_cam[1]*100:5.2f}cm   DEPTH {t_cam[2]*100:5.2f}cm")
    dom = ["right", "down", "DEPTH"][int(t_cam.argmax())]
    print(f"   -> dominant axis: {dom}"
          + ("  (confirms depth-degeneracy)" if dom == "DEPTH" else "  (NOT depth — different cause)"))

    # ---- B. multi-camera fix ----
    print("\nB. MULTI-CAMERA (does adding views pin translation?)")
    rigs = {"1 cam  [0]": [0.0],
            "2 cams [0,50]": [0.0, 50.0],
            "3 cams [-50,0,50]": [-50.0, 0.0, 50.0]}
    for start_lbl, (rot, trans) in [("from 5deg", MID), ("from 9deg", BIG)]:
        for lbl, az in rigs.items():
            cams = make_cameras(gA, gB, az)
            T_fin = solve(gA, gB, T_gt, cams, init_from(T_gt, rot, trans), steps=400)
            rf, tf = pose_error(T_fin, T_gt)
            ok = "PASS" if (rf < 0.5 and tf < 0.01) else "FAIL"
            print(f"   [{ok}] {lbl:20s} {start_lbl}: {rf:.2f}deg / {tf*100:5.2f}cm")

    # ---- C. more iterations, single camera (rule out slow convergence) ----
    print("\nC. ITERATION control (1 cam, from 5deg)")
    for steps in (400, 1200, 3000):
        T_fin = solve(gA, gB, T_gt, [cam0], init_from(T_gt, *MID), steps=steps)
        rf, tf = pose_error(T_fin, T_gt)
        print(f"   {steps:5d} steps: {rf:.2f}deg / {tf*100:5.2f}cm")


if __name__ == "__main__":
    main()
