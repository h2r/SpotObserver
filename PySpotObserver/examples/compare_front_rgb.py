"""
Live side-by-side comparison of the color/light-corrected front RGB feeds.

Streams FRONTLEFT + FRONTRIGHT from one or two Spots and tiles the corrected
RGB images into a single window for easy comparison:

    [ robotA FRONTLEFT ][ robotA FRONTRIGHT ]
    [ robotB FRONTLEFT ][ robotB FRONTRIGHT ]

RGB only (no depth). Color correction + sRGB light correction are applied
inside the stream automatically for known robot IPs. Press 'q' to quit.

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

import cv2
import numpy as np
from pyspotobserver import CameraType, SpotConfig, SpotConnection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WINDOW = "Front RGB comparison (q to quit)"
FRONT_CAMERAS = [CameraType.FRONTLEFT, CameraType.FRONTRIGHT]
BANNER_H = 28

# Friendly names for known robots; falls back to the IP otherwise.
ROBOT_NAMES = {
    "128.148.138.22": "TUSKER",
    "128.148.138.21": "GOUGER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True, help="Primary robot IP.")
    parser.add_argument("--secondary-robot-ip", help="Optional second robot IP.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--duration", type=float, default=600.0, help="Max seconds to run.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Per-frame fetch timeout (s).")
    parser.add_argument("--tile-width", type=int, default=480, help="Width of each camera tile.")
    parser.add_argument(
        "--layout",
        choices=["row", "grid"],
        default="row",
        help="'row': single side-by-side strip [TUSKER L|R | GOUGER L|R]. "
        "'grid': one robot per row (2x2).",
    )
    return parser.parse_args()


def robot_label(ip: str) -> str:
    return ROBOT_NAMES.get(ip, ip)


def to_bgr_uint8(rgb: np.ndarray) -> np.ndarray:
    """Corrected RGB float [0,1] (H,W,3) -> displayable BGR uint8."""
    img = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


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


def fetch_front_rgb(stream, timeout: float) -> dict[CameraType, np.ndarray]:
    """Return {camera: corrected BGR uint8} for the latest frame, or {} on timeout."""
    try:
        rgb_images, _depth, _b2w = stream.get_current_images(timeout=timeout, run_pipeline=False)
    except Exception as exc:  # noqa: BLE001 - keep the viewer alive across transient errors
        logger.warning("fetch failed: %s", exc)
        return {}
    order = stream.get_camera_order()
    return {cam: to_bgr_uint8(rgb) for cam, rgb in zip(order, rgb_images)}


def main() -> int:
    args = parse_args()

    robots = [("primary", args.robot_ip)]
    if args.secondary_robot_ip:
        robots.append(("secondary", args.secondary_robot_ip))

    mask = CameraType(0)
    for cam in FRONT_CAMERAS:
        mask |= cam

    tile_w = args.tile_width
    tile_h = int(tile_w * 480 / 640)  # Spot fisheye frames are 640x480 (WxH)

    with ExitStack() as stack:
        streams = []
        for _label, ip in robots:
            config = SpotConfig(robot_ip=ip, username=args.username, password=args.password)
            conn = stack.enter_context(SpotConnection(config))
            stream = conn.create_cam_stream(stream_id=f"cmp_{ip.replace('.', '_')}")
            stream.start_streaming(mask)
            streams.append((robot_label(ip), stream))
            logger.info("Streaming front RGB from %s (%s)", robot_label(ip), ip)

        try:
            start = time.perf_counter()
            while time.perf_counter() - start < args.duration:
                rows = []
                for name, stream in streams:
                    frames = fetch_front_rgb(stream, args.timeout)
                    row_tiles = [
                        make_tile(
                            frames.get(cam),
                            f"{name} {cam.name}",
                            tile_w,
                            tile_h,
                        )
                        for cam in FRONT_CAMERAS
                    ]
                    rows.append(np.hstack(row_tiles))

                # 'row': one side-by-side strip; 'grid': one robot per row.
                canvas = np.hstack(rows) if args.layout == "row" else np.vstack(rows)
                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested quit")
                    break
        finally:
            for _name, stream in streams:
                stream.stop_streaming()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
