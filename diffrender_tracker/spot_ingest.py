"""
§5.2 ingest helpers — turn Spot data into the tracker's (xyz, rgb) cloud format.

The tracker consumes clouds as (xyz Nx3 float32 [m], rgb Nx3 float32 [0,1]) — exactly what
`cloud_to_gaussians` wants. Two ways in:

  * `ply_to_cloud(path)` — load a saved .ply (e.g. cloud_diag/fused.ply) for OFFLINE dev/test.
  * a live SpotCloudSource (streaming) — added next; it reuses the same backprojection the
    PySpotObserver examples use, then feeds clouds through `voxel_downsample` + `to_grayscale`.

Spot's front cameras are grayscale, so color = intensity (per the grayscale decision). Real
clouds are ~40k points room-scale, so `voxel_downsample` trims to the tracker's ~20k budget.
"""

import numpy as np

# Drop rays past this incidence angle; the Kannala-Brandt unprojection blows up at the rim
# (same constant as examples/live_pointcloud.py).
_MAX_ANGLE_DEG = 85.0


def to_grayscale(rgb):
    """(N,3) colors -> grayscale luma replicated to 3 channels. Collapses any per-camera
    debug tint / real color to a single intensity signal — what the photometric loss uses."""
    rgb = np.asarray(rgb, dtype=np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)   # (N,)
    return np.repeat(luma[:, None], 3, axis=1)


def balance_intensity(col_list):
    """Remove the per-camera luminance seam within one robot's cloud. Spot's frontleft/frontright
    have different exposure/gain (the CCM fixes colour balance but not luminance), so a point's
    grayscale value depends on WHICH camera saw it — which poisons the photometric loss. Normalise
    each camera's intensity to the pooled mean/std so the same surface reads the same regardless of
    camera. `col_list`: list of (Ni,3) grayscale arrays (one per camera). Returns the balanced list.

    Caveat: this also flattens genuine content-brightness differences between the two views; it's a
    first-order gain/bias match, good enough to kill the seam that aliases the tracker."""
    vals = [np.asarray(c, np.float32)[:, 0] for c in col_list]
    pooled = np.concatenate(vals)
    mt, st = float(pooled.mean()), float(max(pooled.std(), 1e-6))
    out = []
    for v in vals:
        m, s = float(v.mean()), float(max(v.std(), 1e-6))
        vn = np.clip((v - m) / s * st + mt, 0.0, 1.0)
        out.append(np.repeat(vn[:, None], 3, axis=1))
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


def _to_gray_rgb_img(rgb):
    """(H,W)|(H,W,1)|(H,W,3) image -> (H,W,3) float [0,1]. Spot front cams are grayscale."""
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
    depth+intensity, fuses both into the frontleft frame, converts to grayscale, and voxel-
    downsamples to ~target points. Drop-to-latest (get_current_images returns the newest frame).

    Use as a context manager:
        with SpotCloudSource(config, calib, camera_mask) as src:
            xyz, rgb = src.latest()        # (N,3) f32 metres, (N,3) f32 grayscale [0,1]

    `config` is a SpotConfig (build it with common_cli.build_config_from_args), `calib` from
    load_fisheye_calib, `camera_mask` from common_cli.build_camera_mask. pyspotobserver is
    imported lazily so this module stays importable on a box without the SDK."""

    def __init__(self, config, calib, camera_mask, cameras=("frontleft", "frontright"),
                 stream_id="diffrender_ingest", stride=2, min_depth=0.2, max_depth=3.0,
                 target=20000, balance_lr=True):
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
        """Newest frame -> (xyz Nx3 f32, gray_rgb Nx3 f32) in the frontleft frame, or None if
        no valid points this frame. Stores body_to_world in self.last_b2w."""
        rgb_list, depth_list, b2w = self._stream.get_current_images(
            timeout=timeout, run_pipeline=False, copy=True)
        self.last_b2w = b2w
        imgs = {self._order[i]: (rgb_list[i], depth_list[i]) for i in range(len(self._order))}
        Rlr, Tlr = self.calib["R"], self.calib["T"]
        pts_all, col_all = [], []
        for nm in self.cameras:
            if nm not in imgs:
                continue
            rgb, dep = imgs[nm]
            color = _to_gray_rgb_img(rgb)
            pts, cols = backproject_fisheye(dep, color, self.calib[f"K_{nm}"],
                                            self.calib[f"D_{nm}"], self.stride,
                                            self.min_depth, self.max_depth)
            if nm == "frontright" and len(pts):
                pts = (pts - Tlr) @ Rlr                   # right cloud -> left frame
            if len(pts):
                pts_all.append(pts)
                col_all.append(cols)
        if not pts_all:
            return None
        if self.balance_lr and len(col_all) > 1:
            col_all = balance_intensity(col_all)          # kill the frontleft/right luminance seam
        xyz = np.vstack(pts_all)
        rgb = to_grayscale(np.vstack(col_all))            # collapse to intensity (grayscale)
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
