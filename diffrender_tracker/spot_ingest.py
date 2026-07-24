"""
§5.2 ingest helpers — turn Spot data into the tracker's (xyz, rgb) cloud format.

The tracker consumes clouds as (xyz Nx3 float32 [m], rgb Nx3 float32 [0,1]) — exactly what
`cloud_to_gaussians` wants. Two ways in:

  * `ply_to_cloud(path)` — load a saved .ply (e.g. cloud_diag/fused.ply) for OFFLINE dev/test.
  * a live SpotCloudSource (streaming) — added next; it reuses the same backprojection the
    PySpotObserver examples use, then feeds clouds through `voxel_downsample` + `to_grayscale`.

COLOR vs GRAYSCALE: these robots' front fisheye cameras are actually COLOR (camera_stream.py
requests PIXEL_FORMAT_RGB_U8 and applies a per-camera CCM — see pyspotobserver/color_correction).
The tracker historically collapsed that to grayscale, but grayscale indoor texture is self-similar
and aliases the photometric loss (the low-coverage tracker divergence). The ingest now keeps color
when asked: pass `color=True` / `grayscale=False`, or set env `DIFFRENDER_COLOR=1`. Default stays
grayscale so existing captures/tests are unchanged. Real clouds are ~40k points room-scale, so
`voxel_downsample` trims to the tracker's ~20k budget.
"""

import os

import numpy as np

# Drop rays past this incidence angle; the Kannala-Brandt unprojection blows up at the rim
# (same constant as examples/live_pointcloud.py).
_MAX_ANGLE_DEG = 85.0


def env_color(default=False):
    """Shared switch so the offline test scripts + capture agree on color vs grayscale.
    True iff env DIFFRENDER_COLOR is set truthy. Keeps color the OPT-IN (default grayscale)."""
    v = os.environ.get("DIFFRENDER_COLOR")
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "rgb", "color")


def to_grayscale(rgb):
    """(N,3) colors -> grayscale luma replicated to 3 channels. Collapses any per-camera
    debug tint / real color to a single intensity signal — what the photometric loss uses."""
    rgb = np.asarray(rgb, dtype=np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)   # (N,)
    return np.repeat(luma[:, None], 3, axis=1)


def balance_intensity(col_list):
    """Remove the per-camera luminance seam within one robot's cloud. Spot's frontleft/frontright
    have different exposure/gain (the CCM fixes colour balance but not luminance), so a point's
    value depends on WHICH camera saw it — which poisons the photometric loss. Normalise each
    camera to the pooled mean/std, PER CHANNEL, so the same surface reads the same regardless of
    camera. `col_list`: list of (Ni,3) color arrays (one per camera). Returns the balanced list.

    Per-channel so it works for both grayscale (all channels equal -> same as luma match) and RGB
    (each channel balanced independently, correcting a colour-cast seam, not just brightness).
    Caveat: this also flattens genuine content-brightness differences between the two views; it's a
    first-order gain/bias match, good enough to kill the seam that aliases the tracker."""
    arrs = [np.asarray(c, np.float32).reshape(-1, 3) for c in col_list]
    pooled = np.concatenate(arrs, axis=0)
    mt = pooled.mean(axis=0)                                   # (3,) per-channel target mean
    st = np.maximum(pooled.std(axis=0), 1e-6)                  # (3,) per-channel target std
    out = []
    for a in arrs:
        m = a.mean(axis=0)
        s = np.maximum(a.std(axis=0), 1e-6)
        out.append(np.clip((a - m) / s * st + mt, 0.0, 1.0))
    return out


# Shared reference exposure (exp_s * gain) so normalize_exposure puts BOTH robots on one
# radiance scale. ~the front cameras' measured exp*gain (probe_exposure.py). Must be the same
# constant for every SpotCloudSource in a run, or a per-robot offset creeps back in.
NOMINAL_EG = 0.19


def normalize_exposure(col_list, eg_list, ref=NOMINAL_EG):
    """Metadata-exact brightness match: scale each camera's colors by ref/(exp_s*gain), the
    inverse of the auto-exposure the camera actually applied. Two cameras (or two robots) that
    saw the same surface then land on one radiance scale — no free parameters, no gray-world
    assumption. This is the ALTERNATIVE to balance_intensity's luminance match (don't stack:
    balance_intensity re-matches means and would undo this). Unlike balance_intensity it does
    NOT touch colour cast — the CCM owns that. A camera reporting eg<=0 (e.g. hand cam, or a
    frame with no capture_params) passes through untouched.

    `col_list`: list of (Ni,3) colors, one per camera. `eg_list`: matching exp_s*gain per camera.
    Returns the scaled list."""
    out = []
    for c, e in zip(col_list, eg_list):
        c = np.asarray(c, np.float32)
        out.append(np.clip(c * (ref / e), 0.0, 1.0) if e and e > 0 else c)
    return out


def voxel_downsample(xyz, rgb, target=20000, min_voxel=0.01):
    """Voxel-grid downsample toward ~`target` points. Picks a voxel size from the bbox volume
    and point budget; returns (xyz, rgb) float32. Uses open3d if available, else passes through."""
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    if len(xyz) <= target:
        return xyz.astype(np.float32), rgb.astype(np.float32)
    try:
        import open3d as o3d
    except Exception:
        return xyz.astype(np.float32), rgb.astype(np.float32)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))

    # Spot clouds lie on 2D surfaces, so point count scales ~ 1/voxel^2 (area), not 1/voxel^3.
    # Seed the voxel from that area model, then iterate toward the target from EITHER side
    # (voxel too big -> too few points -> shrink; too small -> too many -> grow), keeping the
    # closest-to-target result as a fallback if we never land in the band.
    ext = np.maximum(xyz.max(0) - xyz.min(0), 1e-6)
    area = float(np.median(ext)) ** 2                      # rough surface-area proxy
    voxel = max((area / max(target, 1)) ** 0.5, min_voxel)
    best, best_gap = None, None
    for _ in range(12):
        d = pc.voxel_down_sample(voxel)
        n = len(d.points)
        if n == 0:
            voxel = max(voxel * 0.5, min_voxel)
            continue
        gap = abs(n - target)
        if best is None or gap < best_gap:
            best, best_gap = d, gap
        if 0.7 * target <= n <= 1.2 * target:
            break
        factor = np.clip((n / target) ** 0.5, 0.4, 2.5)    # surface scaling, clamped for stability
        new_voxel = max(voxel * factor, min_voxel)
        if new_voxel == voxel:                             # hit the floor; can't add more points
            break
        voxel = new_voxel
    return (np.asarray(best.points, dtype=np.float32),
            np.asarray(best.colors, dtype=np.float32))


def ply_to_cloud(path, grayscale=True, target=None):
    """Load a .ply -> (xyz Nx3 f32, rgb Nx3 f32 [0,1]). grayscale=True collapses color to
    intensity (matches Spot's grayscale front cams / undoes the cloud_diag debug tint).
    target: if set, voxel-downsample toward that many points."""
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pc.points, dtype=np.float32)
    rgb = np.asarray(pc.colors, dtype=np.float32)
    if rgb.size == 0:
        rgb = np.full_like(xyz, 0.5)
    if grayscale:
        rgb = to_grayscale(rgb)
    if target is not None:
        xyz, rgb = voxel_downsample(xyz, rgb, target=target)
    return xyz, np.clip(rgb, 0.0, 1.0)


# --------------------------------------------------------------------------- fisheye ingest
# These mirror examples/live_pointcloud.py so the tracker builds clouds the same way the viewer
# does. cv2 is imported lazily so the offline helpers above stay importable on a box without it.

def load_fisheye_calib(path_arg):
    """Load per-camera K + Kannala-Brandt D and the frontleft<->frontright R,T from a
    calibration.yaml (or a dir containing one). Returns a dict:
    {K_frontleft, D_frontleft, K_frontright, D_frontright, R, T}."""
    import cv2
    from pathlib import Path
    path = Path(path_arg)
    if path.is_dir():
        path = path / "calibration.yaml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit(f"Cannot open calibration: {path}")
    out = {}
    for side in ("frontleft", "frontright"):
        out[f"K_{side}"] = fs.getNode(f"K_{side}").mat().astype(np.float64)
        out[f"D_{side}"] = fs.getNode(f"D_{side}").mat().reshape(-1, 1).astype(np.float64)
    R = fs.getNode("R_left_to_right").mat()
    T = fs.getNode("T_left_to_right").mat()
    fs.release()
    if R is None or T is None:
        raise SystemExit(f"{path} has no R_left_to_right / T_left_to_right")
    out["R"] = R.astype(np.float64)
    out["T"] = T.reshape(3).astype(np.float64)
    return out


def _to_float_rgb_img(rgb):
    """(H,W)|(H,W,1)|(H,W,3) image -> (H,W,3) float [0,1]. PRESERVES color for a 3-channel
    (CCM-corrected RGB) image; replicates a single channel to 3 for a grayscale source. The
    grayscale collapse (if any) happens later in `latest()`/`ply_to_cloud`, not here."""
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr, 0.0, 1.0)


def backproject_fisheye(depth, color, K, D, stride=2, min_d=0.2, max_d=3.0):
    """Fisheye-unproject valid depth pixels -> (points Nx3, colors Nx3) in the camera frame.
    Depth must be registered to the visual frame (same HxW). Same math as live_pointcloud."""
    import cv2
    dep = np.asarray(depth)
    if dep.ndim == 3:
        dep = dep[..., 0]
    dep = dep[::stride, ::stride]
    col = color[::stride, ::stride]

    vs, us = np.indices(dep.shape)
    valid = np.isfinite(dep) & (dep >= min_d) & (dep <= max_d)
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3))

    z = dep[valid].astype(np.float64)
    u = (us[valid] * stride).astype(np.float64)          # undistortPoints wants original coords
    v = (vs[valid] * stride).astype(np.float64)
    pix = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
    rays = cv2.fisheye.undistortPoints(pix, K, D).reshape(-1, 2)   # (x/z, y/z)

    r = np.linalg.norm(rays, axis=1)
    keep = r < np.tan(np.radians(_MAX_ANGLE_DEG))
    rays, z = rays[keep], z[keep]
    pts = np.column_stack((rays[:, 0] * z, rays[:, 1] * z, z))
    cols = col[valid][keep].reshape(-1, 3)
    return pts, cols


def spot_frame_rig(xyz_a, xyz_b, azimuths=(-40.0, 0.0, 40.0), margin=1.4,
                   fov_deg=60.0, width=640, height=480, device="cpu"):
    """Virtual camera rig for a Spot OPTICAL-frame scene (X right, Y down, Z forward): cameras
    sit behind the overlap (toward the sensor, -Z), ringed in azimuth about the vertical axis,
    aimed at the overlap centroid (up = -Y). Multi-view breaks the depth-degeneracy (brief §7.4).
    Returns a list[VirtualCamera]. (place_overlap/rig_overlap assume the synthetic fixture's +Y
    orientation; this is the Spot-data variant.)"""
    from camera import VirtualCamera
    a, b = np.asarray(xyz_a, np.float64), np.asarray(xyz_b, np.float64)
    lo = np.maximum(a.min(0), b.min(0))
    hi = np.minimum(a.max(0), b.max(0))
    c = 0.5 * (lo + hi)
    e = float((hi - lo).max())
    cams = []
    for az in azimuths:
        th = np.deg2rad(az)
        d = np.array([np.sin(th), 0.0, -np.cos(th)])     # 0deg -> straight behind (-Z)
        eye = c + margin * e * d
        eye[1] = c[1] - 0.2 * e                          # lift (up is -Y)
        cams.append(VirtualCamera.look_at(eye.tolist(), c.tolist(), up=(0, -1, 0),
                                          fov_deg=fov_deg, width=width, height=height,
                                          device=device))
    return cams


class SpotCloudSource:
    """Live colored-cloud source for ONE robot: streams frontleft+frontright, fisheye-backprojects
    depth+color, fuses both into the frontleft frame, (optionally) collapses to grayscale, and
    voxel-downsamples to ~target points. Drop-to-latest (get_current_images returns the newest frame).

    Use as a context manager:
        with SpotCloudSource(config, calib, camera_mask, color=True) as src:
            xyz, rgb = src.latest()        # (N,3) f32 metres, (N,3) f32 [0,1] (RGB if color=True)

    `color=False` (default) collapses to grayscale, matching the historical behaviour; `color=True`
    keeps the CCM-corrected RGB (richer texture -> less aliasing for the photometric tracker).
    `config` is a SpotConfig (build it with common_cli.build_config_from_args), `calib` from
    load_fisheye_calib, `camera_mask` from common_cli.build_camera_mask. pyspotobserver is
    imported lazily so this module stays importable on a box without the SDK."""

    def __init__(self, config, calib, camera_mask, cameras=("frontleft", "frontright"),
                 stream_id="diffrender_ingest", stride=2, min_depth=0.2, max_depth=3.0,
                 target=20000, balance_lr=True, color=False, bright_mode=None,
                 exposure_ref=NOMINAL_EG):
        self.config = config
        self.calib = calib
        self.camera_mask = camera_mask
        self.cameras = [c.lower() for c in cameras]
        self.stream_id = stream_id
        self.stride = stride
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target = target
        self.balance_lr = balance_lr        # normalise frontleft/frontright luminance at fuse
        self.color = color                  # keep RGB (True) or collapse to grayscale (False)
        # How to reconcile brightness across cameras/robots at fuse time:
        #   "balance"  gray-world per-channel mean/std match (balance_intensity) — also fixes
        #              colour seam, but estimates brightness from pixels (content-confounded).
        #   "exposure" metadata-exact exp*gain normalisation (normalize_exposure) — luminance
        #              only, no free params; the shared exposure_ref ties the two robots together.
        #   "none"     leave each camera as-is.
        # Default preserves the old behaviour (balance if balance_lr else none).
        self.bright_mode = bright_mode or ("balance" if balance_lr else "none")
        self.exposure_ref = exposure_ref
        self._conn_cm = self._conn = self._stream = self._order = None
        self.last_b2w = None

    def __enter__(self):
        from pyspotobserver import SpotConnection
        self._conn_cm = SpotConnection(self.config)
        self._conn = self._conn_cm.__enter__()
        self._stream = self._conn.create_cam_stream(stream_id=self.stream_id)
        self._stream.start_streaming(self.camera_mask)
        self._order = [c.name.lower() for c in self._stream.get_camera_order()]
        return self

    def __exit__(self, *exc):
        try:
            if self._stream is not None:
                self._stream.stop_streaming()
        finally:
            if self._conn_cm is not None:
                self._conn_cm.__exit__(*exc)

    def latest(self, timeout=2.0):
        """Newest frame -> (xyz Nx3 f32, rgb Nx3 f32 [0,1]) in the frontleft frame, or None if
        no valid points this frame. RGB when color=True, else grayscale (channels equal). Stores
        body_to_world in self.last_b2w."""
        rgb_list, depth_list, b2w, eg = self._stream.get_current_images(
            timeout=timeout, run_pipeline=False, copy=True, include_exposure=True)
        self.last_b2w = b2w
        imgs = {self._order[i]: (rgb_list[i], depth_list[i]) for i in range(len(self._order))}
        eg_by_name = ({self._order[i]: eg[i] for i in range(min(len(self._order), len(eg)))}
                      if eg is not None else {})
        Rlr, Tlr = self.calib["R"], self.calib["T"]
        pts_all, col_all, eg_used = [], [], []
        for nm in self.cameras:
            if nm not in imgs:
                continue
            rgb, dep = imgs[nm]
            color = _to_float_rgb_img(rgb)                # preserves CCM-corrected RGB
            pts, cols = backproject_fisheye(dep, color, self.calib[f"K_{nm}"],
                                            self.calib[f"D_{nm}"], self.stride,
                                            self.min_depth, self.max_depth)
            if nm == "frontright" and len(pts):
                pts = (pts - Tlr) @ Rlr                   # right cloud -> left frame
            if len(pts):
                pts_all.append(pts)
                col_all.append(cols)
                eg_used.append(eg_by_name.get(nm, 0.0))   # exp*gain for this camera (0 = unknown)
        if not pts_all:
            return None
        if self.bright_mode == "exposure":
            col_all = normalize_exposure(col_all, eg_used, ref=self.exposure_ref)
        elif self.bright_mode == "balance" and len(col_all) > 1:
            col_all = balance_intensity(col_all)          # kill the frontleft/right seam (per-channel)
        xyz = np.vstack(pts_all)
        rgb = np.vstack(col_all)
        if not self.color:
            rgb = to_grayscale(rgb)                       # collapse to intensity unless color=True
        return voxel_downsample(xyz, rgb, target=self.target)


if __name__ == "__main__":
    import os
    here = os.path.dirname(__file__)
    ply = os.path.join(here, "..", "PySpotObserver", "examples", "cloud_diag", "fused.ply")
    if os.path.exists(ply):
        xyz0, rgb0 = ply_to_cloud(ply, grayscale=True)
        xyz, rgb = voxel_downsample(xyz0, rgb0, target=20000)
        print(f"loaded {ply}")
        print(f"  raw   {xyz0.shape[0]} pts")
        print(f"  voxel {xyz.shape[0]} pts (target 20000)  extent "
              f"{np.round(xyz.max(0)-xyz.min(0),2)} m")
        print(f"  gray rgb range [{rgb.min():.2f},{rgb.max():.2f}]  "
              f"channels equal? {np.allclose(rgb[:,0],rgb[:,1])}")
    else:
        print("no cloud_diag/fused.ply found; skipping demo")
