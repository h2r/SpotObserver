"""Front camera stitching via point-cloud projection."""

from dataclasses import dataclass

import cv2
import numpy as np
from bosdyn.api import image_pb2  # type: ignore[import-untyped]
from bosdyn.client.frame_helpers import (  # type: ignore[import-untyped]
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)

STITCH_OUT_W = 1800
STITCH_OUT_H = 1000

# Exposure/gain compensation. The two front cameras auto-expose independently and
# point ~62 deg apart, so one (typically frontleft) comes out darker. Their static
# color-correction matrices bake in fixed per-camera gains and cannot track this, so
# the raw stitch shows a brightness step at the seam. We measure mean luminance over
# the overlapping FOV and scale the darker side up to the brighter one before blending.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # RGB (Rec.709)
_MIN_OVERLAP_PX = 200  # too little overlap -> gain estimate is noise; skip
_GAIN_LIMITS = (0.4, 4.0)  # clamp so a bad estimate can't blow out the image


def _mean_luma(rgb_pixels: np.ndarray) -> float:
    """Mean Rec.709 luminance of an (N, 3) or (H, W, 3) RGB float array."""
    if rgb_pixels.size == 0:
        return 0.0
    return float((rgb_pixels.reshape(-1, 3) * _LUMA_WEIGHTS).sum(axis=1).mean())


def _match_gain(
    rgb_l: np.ndarray,
    cov_l: np.ndarray,
    rgb_r: np.ndarray,
    cov_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale the darker of the two images up to the brighter, measured over the
    overlap (cov_l & cov_r). Returns the (possibly rescaled) images; the whole
    darker image is scaled so brightness is consistent across the panorama, not
    just in the seam. No-op when the overlap is too small or either side is black.
    """
    overlap = cov_l & cov_r
    if int(overlap.sum()) < _MIN_OVERLAP_PX:
        return rgb_l, rgb_r
    lum_l = _mean_luma(rgb_l[overlap])
    lum_r = _mean_luma(rgb_r[overlap])
    if lum_l <= 1e-4 or lum_r <= 1e-4:
        return rgb_l, rgb_r
    if lum_l < lum_r:
        gain = float(np.clip(lum_r / lum_l, *_GAIN_LIMITS))
        rgb_l = np.clip(rgb_l * gain, 0.0, 1.0)
    else:
        gain = float(np.clip(lum_l / lum_r, *_GAIN_LIMITS))
        rgb_r = np.clip(rgb_r * gain, 0.0, 1.0)
    return rgb_l, rgb_r


def _rasterize_frontal(
    pts: np.ndarray,
    cols: np.ndarray,
    out_w: int,
    out_h: int,
    f_virt: float,
    cx_v: float,
    cy_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one camera's body-frame points onto the virtual frontal canvas.

    Returns (rgb, dep) canvases; nearer points win (painted far-to-near). Keeping
    each camera on its own canvas lets us gain-match them before compositing.
    """
    rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
    dep = np.zeros((out_h, out_w), dtype=np.float32)

    fwd = pts[:, 0]
    keep = fwd > 0.1
    pts, cols, fwd = pts[keep], cols[keep], fwd[keep]
    if pts.shape[0] == 0:
        return rgb, dep

    u = (-pts[:, 1] * f_virt / pts[:, 0] + cx_v).astype(np.int32)
    v = (-pts[:, 2] * f_virt / pts[:, 0] + cy_v).astype(np.int32)
    in_bounds = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u, v, cols, fwd = u[in_bounds], v[in_bounds], cols[in_bounds], fwd[in_bounds]

    order = np.argsort(fwd)[::-1]  # far-to-near so closer points overwrite
    rgb[v[order], u[order]] = cols[order]
    dep[v[order], u[order]] = fwd[order]
    return rgb, dep


@dataclass
class CamStitchParams:
    rot: np.ndarray  # (3, 3) body_R_cam rotation matrix
    trans: np.ndarray  # (3,) body_t_cam translation vector
    fx: float
    fy: float
    cx: float
    cy: float
    # Fisheye (Kannala-Brandt) calibration. When D is set, _backproject uses the
    # fisheye unprojection (K + D) instead of the naive pinhole formula. Left None
    # for backward compatibility (falls back to pinhole using fx/fy/cx/cy).
    K: np.ndarray | None = None  # (3, 3) fisheye camera matrix
    D: np.ndarray | None = None  # (4, 1) Kannala-Brandt distortion coeffs


def load_fisheye_calibration(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load per-camera fisheye K + D from a calibrate_fisheye.py calibration.yaml.

    Returns {"frontleft": (K, D), "frontright": (K, D)}. Attach these to the
    CamStitchParams (see attach_fisheye_calibration) so _backproject undistorts.
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in ("frontleft", "frontright"):
        K = fs.getNode(f"K_{side}").mat()
        D = fs.getNode(f"D_{side}").mat()
        if K is None or D is None:
            fs.release()
            raise ValueError(f"calibration.yaml missing K_{side}/D_{side}: {path}")
        out[side] = (K.astype(np.float64), D.reshape(-1, 1).astype(np.float64))
    fs.release()
    return out


def attach_fisheye_calibration(
    params: CamStitchParams, K: np.ndarray, D: np.ndarray
) -> None:
    """Populate a CamStitchParams' K/D in place so _backproject uses fisheye math."""
    params.K = K
    params.D = D


def extract_stitch_params(response: image_pb2.ImageResponse) -> CamStitchParams:
    """Extract body-to-camera transform and pinhole intrinsics from an ImageResponse."""
    snapshot = response.shot.transforms_snapshot
    frame_name = response.shot.frame_name_image_sensor
    pose = get_a_tform_b(snapshot, BODY_FRAME_NAME, frame_name)
    if pose is None:
        raise ValueError(f"No transform from body to {frame_name!r}")
    M = pose.to_matrix()

    src = response.source
    which = src.WhichOneof("camera_models")
    if which == "pinhole":
        intr = src.pinhole.intrinsics
    elif which == "pinhole_brown_conrady":
        intr = src.pinhole_brown_conrady.intrinsics.pinhole_intrinsics
    elif which == "kannala_brandt":
        intr = src.kannala_brandt.intrinsics.pinhole_intrinsics
    else:
        raise ValueError(f"No pinhole intrinsics for {frame_name!r} (model={which!r})")

    return CamStitchParams(
        rot=M[:3, :3].copy(),
        trans=M[:3, 3].copy(),
        fx=intr.focal_length.x,
        fy=intr.focal_length.y,
        cx=intr.principal_point.x,
        cy=intr.principal_point.y,
    )


def extract_btw_params(response: image_pb2.ImageResponse) -> np.ndarray:
    """
    Extract the body pose in the vision/world frame for this image capture.

    This changes frame-to-frame. By contrast, body_T_camera is static for a
    mounted body camera and can usually be cached once.

    Spot docs describe the vision frame as:
        "An inertial frame that estimates the fixed location in the world
        (relative to where the robot is booted up), and is calculated using
        visual analysis of the world and the robot's odometry."
    """
    snapshot = response.shot.transforms_snapshot
    vision_T_body = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)

    if vision_T_body is None:
        raise ValueError(
            "No transform from body to vision/world in image transform snapshot. "
            "If this occurs frequently, consider using the previous frame instead of throwing exception."
        )
    b2w = vision_T_body.to_matrix()

    return b2w.copy()


def extract_ctb_params(response: image_pb2.ImageResponse) -> np.ndarray:
    """Extract the camera pose in the body frame for this image capture."""
    snapshot = response.shot.transforms_snapshot
    frame_name = response.shot.frame_name_image_sensor
    body_T_camera = get_a_tform_b(snapshot, BODY_FRAME_NAME, frame_name)

    if body_T_camera is None:
        raise ValueError(f"No transform from body to {frame_name!r}")

    return body_T_camera.to_matrix().copy()


def _backproject(
    rgb: np.ndarray, dep: np.ndarray, params: CamStitchParams
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid pixels to body-frame 3D points.

    If the camera has fisheye calibration (params.D set), each pixel is
    unprojected with the Kannala-Brandt model (cv2.fisheye.undistortPoints) so
    lens distortion -- strongest at the image edges -- is removed before the
    point is placed in 3D. Otherwise it falls back to the naive pinhole
    (u-cx)/fx*z formula, which mis-places fisheye pixels toward the periphery.
    """
    v_idx, u_idx = np.indices(dep.shape)
    valid = (dep > 0) & np.isfinite(dep)
    z = dep[valid]
    u = u_idx[valid].astype(np.float64)
    v = v_idx[valid].astype(np.float64)

    if params.D is not None:
        K = params.K
        if K is None:
            K = np.array([[params.fx, 0.0, params.cx],
                          [0.0, params.fy, params.cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        pix = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
        # undistortPoints with P=None -> normalized rays (x/z, y/z)
        rays = cv2.fisheye.undistortPoints(pix, K, params.D).reshape(-1, 2)
        x = rays[:, 0] * z
        y = rays[:, 1] * z
    else:
        x = (u - params.cx) * z / params.fx
        y = (v - params.cy) * z / params.fy

    cam_pts = np.stack((x, y, z), axis=-1)
    body_pts = cam_pts @ params.rot.T + params.trans
    return body_pts, rgb[valid]


def _stitch_pointcloud(
    l_rgb: np.ndarray,
    l_dep: np.ndarray,
    l_params: CamStitchParams,
    r_rgb: np.ndarray,
    r_dep: np.ndarray,
    r_params: CamStitchParams,
    out_rgb: np.ndarray,
    out_dep: np.ndarray,
    match_exposure: bool = True,
) -> None:
    """Depth-based stitcher: back-projects pixels to a body-frame point cloud, then
    re-projects. Each camera is rasterized onto its own canvas so their brightness can
    be gain-matched over the overlap before compositing; in the overlap the nearer
    point wins (same as the original single-canvas far-to-near paint).
    """
    out_h, out_w = out_rgb.shape[:2]
    out_rgb[:] = 0.0
    out_dep[:] = 0.0

    pts_l, cols_l = _backproject(l_rgb, l_dep, l_params)
    pts_r, cols_r = _backproject(r_rgb, r_dep, r_params)

    # Virtual frontal projection: u = -Y/X * f + cx_v,  v = -Z/X * f + cy_v
    # cy_v is anchored to the camera's own principal point so the horizon stays
    # at the same canvas row regardless of how tall the output buffer is.
    f_virt = l_params.fx
    cx_v, cy_v = out_w / 2.0, l_params.cy

    rgb_l, dep_l = _rasterize_frontal(pts_l, cols_l, out_w, out_h, f_virt, cx_v, cy_v)
    rgb_r, dep_r = _rasterize_frontal(pts_r, cols_r, out_w, out_h, f_virt, cx_v, cy_v)

    cov_l = dep_l > 0
    cov_r = dep_r > 0
    if match_exposure:
        rgb_l, rgb_r = _match_gain(rgb_l, cov_l, rgb_r, cov_r)

    # Composite: exclusive coverage copies straight over; in the overlap the nearer
    # (smaller depth) point wins, matching the original far-to-near semantics.
    only_l = cov_l & ~cov_r
    only_r = cov_r & ~cov_l
    both = cov_l & cov_r
    l_near = both & (dep_l <= dep_r)
    r_near = both & ~l_near

    for mask, rgb_src, dep_src in (
        (only_l, rgb_l, dep_l),
        (only_r, rgb_r, dep_r),
        (l_near, rgb_l, dep_l),
        (r_near, rgb_r, dep_r),
    ):
        out_rgb[mask] = rgb_src[mask]
        out_dep[mask] = dep_src[mask]


def _stitch_cylindrical(
    l_rgb: np.ndarray,
    l_dep: np.ndarray,
    l_params: CamStitchParams,
    r_rgb: np.ndarray,
    r_dep: np.ndarray,
    r_params: CamStitchParams,
    out_rgb: np.ndarray,
    out_dep: np.ndarray,
    match_exposure: bool = True,
) -> None:
    """
    Depth-free stitcher: for each virtual canvas pixel, back-computes the body-frame
    ray direction from the virtual K, rotates it into each camera frame using R only
    (translation ignored — assumes scene at infinity), then samples each camera with
    cv2.remap. Blends 50/50 in the overlap region. Depth is warped with the same
    maps using nearest-neighbour sampling to avoid blending across discontinuities.
    """
    out_h, out_w = out_rgb.shape[:2]

    f_virt = l_params.fx
    cx_v, cy_v = out_w / 2.0, l_params.cy

    # Body-frame ray for each virtual canvas pixel: d = [1, -(u-cx)/f, -(v-cy)/f]
    ug, vg = np.meshgrid(
        np.arange(out_w, dtype=np.float32),
        np.arange(out_h, dtype=np.float32),
    )
    d_body = np.stack(
        [
            np.ones((out_h, out_w), dtype=np.float32),
            -(ug - cx_v) / f_virt,
            -(vg - cy_v) / f_virt,
        ],
        axis=-1,
    )  # (out_h, out_w, 3)

    def _make_warp(params: CamStitchParams, img_h: int, img_w: int):
        # Rotate body ray into camera frame (row-vector form: d_cam = d_body @ rot,
        # since rot = body_R_cam so rot.T maps body→cam, and (rot.T @ col).T = row @ rot)
        d_cam = d_body @ params.rot  # (out_h, out_w, 3)
        dz = d_cam[..., 2]
        safe_dz = np.where(dz > 0, dz, 1.0)  # avoid div-by-zero for behind-camera rays
        map_x = np.where(dz > 0, params.fx * d_cam[..., 0] / safe_dz + params.cx, -1.0).astype(
            np.float32
        )
        map_y = np.where(dz > 0, params.fy * d_cam[..., 1] / safe_dz + params.cy, -1.0).astype(
            np.float32
        )
        valid = (dz > 0) & (map_x >= 0) & (map_x < img_w) & (map_y >= 0) & (map_y < img_h)
        return map_x, map_y, valid

    l_h, l_w = l_rgb.shape[:2]
    r_h, r_w = r_rgb.shape[:2]
    map_xl, map_yl, valid_l = _make_warp(l_params, l_h, l_w)
    map_xr, map_yr, valid_r = _make_warp(r_params, r_h, r_w)

    warped_l = cv2.remap(
        l_rgb, map_xl, map_yl, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    warped_r = cv2.remap(
        r_rgb, map_xr, map_yr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    if match_exposure:
        warped_l, warped_r = _match_gain(warped_l, valid_l, warped_r, valid_r)

    out_rgb[:] = 0.0
    out_rgb[valid_l & ~valid_r] = warped_l[valid_l & ~valid_r]
    out_rgb[valid_r & ~valid_l] = warped_r[valid_r & ~valid_l]
    both = valid_l & valid_r
    out_rgb[both] = 0.5 * (warped_l[both] + warped_r[both])

    # Warp depth with the same maps (nearest-neighbour avoids blending across discontinuities)
    warped_dl = cv2.remap(
        l_dep, map_xl, map_yl, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    warped_dr = cv2.remap(
        r_dep, map_xr, map_yr, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    dep_valid_l = valid_l & (warped_dl > 0)
    dep_valid_r = valid_r & (warped_dr > 0)
    out_dep[:] = 0.0
    out_dep[dep_valid_l & ~dep_valid_r] = warped_dl[dep_valid_l & ~dep_valid_r]
    out_dep[dep_valid_r & ~dep_valid_l] = warped_dr[dep_valid_r & ~dep_valid_l]
    dep_both = dep_valid_l & dep_valid_r
    out_dep[dep_both] = 0.5 * (warped_dl[dep_both] + warped_dr[dep_both])


def compute_stitch(
    l_rgb: np.ndarray,
    l_dep: np.ndarray,
    l_params: CamStitchParams,
    r_rgb: np.ndarray,
    r_dep: np.ndarray,
    r_params: CamStitchParams,
    out_rgb: np.ndarray,
    out_dep: np.ndarray,
    use_depth: bool = True,
    match_exposure: bool = True,
) -> None:
    """
    Stitch left and right front camera images onto a virtual forward-facing canvas.

    use_depth=True  — point-cloud projection (accurate at all ranges, requires depth).
    use_depth=False — direction-only inverse warp (no depth needed, parallax error
                      grows as objects get closer than ~3 m).
    match_exposure  — when True (default), scale the darker camera up to the brighter
                      over their overlap so the seam is not a brightness step.

    Results written in-place to out_rgb (H, W, 3) float32 and out_dep (H, W) float32.
    out_dep is zeroed when use_depth=False.
    """
    if use_depth:
        _stitch_pointcloud(
            l_rgb, l_dep, l_params, r_rgb, r_dep, r_params, out_rgb, out_dep, match_exposure
        )
    else:
        _stitch_cylindrical(
            l_rgb, l_dep, l_params, r_rgb, r_dep, r_params, out_rgb, out_dep, match_exposure
        )
