"""
Live comparison of the front cameras, color/light-corrected.

Default view is the STITCHED front panorama (frontleft + frontright fused so the
floor connects), one per robot, stacked for comparison. The stitch gain-matches
the two cameras over their overlap so the darker side (usually frontleft) no
longer shows a brightness step at the seam.

    [ TUSKER  stitched front ]
    [ GOUGER  stitched front ]

Use --view lr to instead see the left/right tiles, each rotated 90° clockwise (the
front cameras are mounted sideways), one row per robot with right-then-left:

    [ TUSKER R ][ TUSKER L ]
    [ GOUGER R ][ GOUGER L ]

RGB only (no depth). Keys: 'q' quit; 'c' toggle per-camera color calibration (CCM)
off/on (calibrated vs raw); 'm' toggle the live match of the right front camera to the
left (color + brightness), which cleans up the residual pink/exposure a fixed CCM can't.

Example:
    python examples/compare_front_rgb.py \
        --robot-ip 128.148.138.22 \
        --secondary-robot-ip 128.148.138.21 \
        --username user --password bigbubbabigbubba
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
from pyspotobserver import CameraType, SpotConfig, SpotConnection
from pyspotobserver.stitch import STITCH_OUT_H, STITCH_OUT_W

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WINDOW = "Front RGB comparison (q to quit)"
FRONT_CAMERAS = [CameraType.FRONTLEFT, CameraType.FRONTRIGHT]
BANNER_H = 28
_CALIB_ROOT = Path(__file__).with_name("calib")

# Friendly names + fisheye calibration folder for known robots.
ROBOT_INFO = {
    "128.148.138.22": ("TUSKER", "spot"),
    "128.148.138.21": ("GOUGER", "spot2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True, help="Primary robot IP.")
    parser.add_argument("--secondary-robot-ip", help="Optional second robot IP.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--duration", type=float, default=600.0, help="Max seconds to run.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Per-frame fetch timeout (s).")
    parser.add_argument(
        "--view",
        choices=["stitch", "lr"],
        default="stitch",
        help="'stitch': fused front panorama per robot (default). "
        "'lr': left/right tiles rotated 90° CW, one row per robot (right then left).",
    )
    parser.add_argument("--tile-width", type=int, default=None, help="Width per panel (px).")
    parser.add_argument(
        "--no-color-correct",
        action="store_true",
        help="Start with the per-camera color calibration (CCM) OFF so you see raw "
        "camera output. Press 'c' at any time to toggle calibrated vs raw live.",
    )
    parser.add_argument(
        "--no-match-lr",
        dest="match_lr",
        action="store_false",
        help="Disable the live per-frame match of the right front camera to the left "
        "(color + brightness). On by default in the 'lr' view; toggle live with 'm'.",
    )
    parser.add_argument(
        "--calib",
        help="Override fisheye calibration.yaml path (applies to all robots). "
        "By default each known robot uses examples/calib/<spot|spot2>/calibration.yaml.",
    )
    return parser.parse_args()


def robot_name(ip: str) -> str:
    info = ROBOT_INFO.get(ip)
    return info[0] if info else ip


def calib_path_for(ip: str, override: str | None) -> str | None:
    if override:
        return override
    info = ROBOT_INFO.get(ip)
    if info is None:
        return None
    path = _CALIB_ROOT / info[1] / "calibration.yaml"
    return str(path) if path.exists() else None


def to_bgr_uint8(rgb: np.ndarray) -> np.ndarray:
    """Corrected RGB float [0,1] (H,W,3) -> displayable BGR uint8."""
    img = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def match_right_to_left(
    left_bgr: np.ndarray | None, right_bgr: np.ndarray | None
) -> np.ndarray | None:
    """Scale the right front camera's per-channel means to the left's, so right matches
    the (well-calibrated) left in BOTH color and brightness. This fixes the residual
    per-camera cast (front-right's pink) and the live exposure difference that a fixed
    CCM can't track — the two front cameras are different sensors and auto-expose
    independently, so no static matrix makes them agree. Gray-world assumption: the two
    overlapping (~62°) front cams see roughly similar average color; approximate if they
    happen to view very different scenes. Left is left untouched (it's the anchor)."""
    if left_bgr is None or right_bgr is None:
        return right_bgr
    # Match over MID-TONE pixels only, via median. The fisheye frames have large black
    # floor/out-of-view wedges and blown highlights; a whole-frame mean is dominated by
    # those and matches badly. Median over 25<luma<235 keeps only usefully-lit pixels.
    luma = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR -> luminance

    def midtone_median(img: np.ndarray) -> np.ndarray:
        lum = img.astype(np.float32) @ luma
        mask = (lum > 25.0) & (lum < 235.0)
        px = img[mask] if int(mask.sum()) > 500 else img.reshape(-1, 3)
        return np.median(px.astype(np.float32), axis=0)

    gains = np.clip(midtone_median(left_bgr) / np.clip(midtone_median(right_bgr), 1e-3, None), 0.3, 3.0)
    return np.clip(right_bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def make_tile(img_bgr: np.ndarray | None, label: str, tile_w: int, tile_h: int) -> np.ndarray:
    """Resize to (tile_w, tile_h) and add a text banner on top."""
    if img_bgr is None:
        body = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        cv2.putText(
            body, "no signal", (10, tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )
    else:
        body = cv2.resize(img_bgr, (tile_w, tile_h))
    banner = np.zeros((BANNER_H, tile_w, 3), dtype=np.uint8)
    cv2.putText(
        banner, label, (8, BANNER_H - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([banner, body])


def fetch_rgb(stream, timeout: float) -> dict[CameraType, np.ndarray]:
    """Return {camera: corrected BGR uint8} for the latest frame, or {} on timeout."""
    try:
        rgb_images, _depth, _b2w = stream.get_current_images(timeout=timeout, run_pipeline=False)
    except Exception as exc:  # noqa: BLE001 - keep the viewer alive across transient errors
        logger.warning("fetch failed: %s", exc)
        return {}
    order = stream.get_camera_order()
    return {cam: to_bgr_uint8(rgb) for cam, rgb in zip(order, rgb_images)}


def render_stitch(streams, args, tile_w: int) -> np.ndarray:
    """One stitched panorama per robot, stacked vertically."""
    tile_h = int(tile_w * STITCH_OUT_H / STITCH_OUT_W)
    tiles = []
    for name, stream in streams:
        frames = fetch_rgb(stream, args.timeout)
        tiles.append(make_tile(frames.get(CameraType.FRONTSTITCHED), f"{name} stitched", tile_w, tile_h))
    return np.vstack(tiles)


def render_lr(streams, args, tile_w: int) -> np.ndarray:
    """Raw tiles, each rotated 90° CW (front cameras are mounted sideways), arranged
    as one row per robot with right-then-left; robots stacked top to bottom."""
    # Rotating 90° makes the 640x480 (WxH) frames portrait -> 480x640, so tiles are tall.
    tile_h = int(tile_w * 640 / 480)
    rows = []
    for name, stream in streams:
        frames = fetch_rgb(stream, args.timeout)
        left = frames.get(CameraType.FRONTLEFT)
        right = frames.get(CameraType.FRONTRIGHT)
        if getattr(args, "match_lr", False):
            right = match_right_to_left(left, right)
        robot_tiles = []
        for cam, img in ((CameraType.FRONTRIGHT, right), (CameraType.FRONTLEFT, left)):
            if img is not None:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            robot_tiles.append(make_tile(img, f"{name} {cam.name}", tile_w, tile_h))
        rows.append(np.hstack(robot_tiles))
    return np.vstack(rows)


def main() -> int:
    args = parse_args()

    robots = [args.robot_ip]
    if args.secondary_robot_ip:
        robots.append(args.secondary_robot_ip)

    stitch_view = args.view == "stitch"
    mask = CameraType(0)
    for cam in FRONT_CAMERAS:
        mask |= cam
    if stitch_view:
        mask |= CameraType.FRONTSTITCHED

    # Sensible default panel width per view (stitched panoramas are wide; rotated
    # lr tiles are portrait and stacked two-high, so keep them narrow).
    tile_w = args.tile_width or (960 if stitch_view else 360)

    with ExitStack() as stack:
        streams = []
        for ip in robots:
            config = SpotConfig(robot_ip=ip, username=args.username, password=args.password)
            conn = stack.enter_context(SpotConnection(config))
            stream = conn.create_cam_stream(stream_id=f"cmp_{ip.replace('.', '_')}")
            if stitch_view:
                calib = calib_path_for(ip, args.calib)
                if calib:
                    # Read by _cache_stitch_params so the stitch undistorts with the
                    # fisheye model instead of the naive pinhole fallback.
                    stream._stitch_calib_path = calib  # noqa: SLF001
                    logger.info("%s: using fisheye calibration %s", robot_name(ip), calib)
                else:
                    logger.warning(
                        "%s (%s): no calibration.yaml found; stitch falls back to pinhole "
                        "and the floor may not line up.", robot_name(ip), ip
                    )
            stream.start_streaming(mask)
            streams.append((robot_name(ip), stream))
            logger.info("Streaming front cameras from %s (%s)", robot_name(ip), ip)

        render = render_stitch if stitch_view else render_lr

        # Live A/B of the per-camera color calibration. We stash each stream's CCMs so
        # 'c' can toggle them off (raw) and back on (calibrated) on the same live scene.
        orig_ccms = {id(stream): stream._ccms for _name, stream in streams}  # noqa: SLF001
        color_on = not args.no_color_correct
        for _name, stream in streams:
            stream._ccms = orig_ccms[id(stream)] if color_on else None  # noqa: SLF001
        logger.info("Color calibration %s (press 'c' to toggle)", "ON" if color_on else "OFF (raw)")

        try:
            start = time.perf_counter()
            while time.perf_counter() - start < args.duration:
                cv2.imshow(WINDOW, render(streams, args, tile_w))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User requested quit")
                    break
                if key == ord("c"):
                    color_on = not color_on
                    for _name, stream in streams:
                        stream._ccms = orig_ccms[id(stream)] if color_on else None  # noqa: SLF001
                    logger.info("Color calibration %s", "ON" if color_on else "OFF (raw)")
                if key == ord("m"):
                    args.match_lr = not args.match_lr
                    logger.info("Live right->left match %s", "ON" if args.match_lr else "OFF")
        finally:
            for _name, stream in streams:
                stream.stop_streaming()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
