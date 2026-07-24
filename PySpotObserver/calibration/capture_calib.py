#!/usr/bin/env python3
"""
Step 2: Capture synchronized frontleft/frontright fisheye frames for calibration.

This reuses your existing pyspotobserver / common_cli plumbing (same imports as
your streaming example) but, instead of just displaying frames, it SAVES a
synchronized left/right pair to disk each time you press the space bar.

Run it ONCE PER ROBOT, pointing --output-dir at a per-robot folder:

    # robot A (spot):
    python capture_calib.py <your usual connection args> --output-dir calib/spot
    # robot B (spot2):  (add whatever arg selects the second robot's IP)
    python capture_calib.py <spot2 connection args> --output-dir calib/spot2

Controls (in the preview window):
    SPACE  save the current left/right pair
    q      quit

Coverage while capturing (aim for ~30-60 saved pairs):
    - push the board into all four corners and let it run off the frame edges
      (this is what constrains fisheye distortion -- do not keep it centered)
    - tilt it obliquely in every direction
    - roll it in-plane (rotate like a steering wheel)
    - vary distance, near and far
    - keep BOTH cameras seeing the board as much as possible (needed for the
      stereo extrinsic later)

Saved layout:
    <output-dir>/frontleft/frame_0000.png
    <output-dir>/frontright/frame_0000.png   (same index == same instant)
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from pyspotobserver import CameraType, SpotConnection
from common_cli import (
    add_common_connection_arguments,
    build_camera_mask,
    build_config_from_args,
    parse_camera_list,
)

from calib_config import make_board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_connection_arguments(parser)
    parser.add_argument(
        "--cameras",
        default="frontleft,frontright",
        help="Comma-separated cameras to capture (default: frontleft,frontright).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Per-robot output folder, e.g. calib/spot or calib/spot2.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Per-frame retrieval timeout in seconds.",
    )
    parser.add_argument(
        "--stream-id",
        default="calib_stream",
        help="Stream identifier.",
    )
    # build_config_from_args() reads these two unconditionally, so they must
    # exist on the namespace even though calibration capture never dumps.
    parser.add_argument(
        "--dumps-enabled", action="store_true", help="store debug information"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If --dumps-enabled, save data to this directory.",
    )
    return parser.parse_args()


def to_gray_uint8(rgb: np.ndarray) -> np.ndarray:
    """Frames arrive as float RGB in [0,1]; Spot front cams are grayscale fisheye.
    Convert to a plain uint8 grayscale image, which is what ChArUco detection and
    the fisheye calibrator want."""
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # frames are RGB from the SDK
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.ndim == 2:
        return arr
    # single-channel-with-trailing-dim or anything odd -> squeeze
    return arr.reshape(arr.shape[0], arr.shape[1])


def main() -> int:
    args = parse_args()
    cameras = parse_camera_list(args.cameras)

    # We specifically need the front stereo pair, in a known order.
    cam_names = [c.name.lower() for c in cameras]
    if not ("frontleft" in cam_names and "frontright" in cam_names):
        print("WARNING: expected frontleft and frontright in --cameras; got",
              cam_names)

    out_root = Path(args.output_dir)
    dirs = {c.name.lower(): out_root / c.name.lower() for c in cameras}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    _, _, detector = make_board()  # only used for a live "is the board seen?" overlay

    config = build_config_from_args(args)

    # Resume numbering: continue after any frames already in the output folders,
    # so re-running to add coverage APPENDS instead of overwriting existing pairs.
    existing = [int(p.stem.split("_")[1])
                for d in dirs.values() for p in d.glob("frame_*.png")]
    saved = max(existing) + 1 if existing else 0
    start_index = saved
    if saved:
        print(f"Found existing frames; appending from frame_{saved:04d}.")

    with ExitStack() as stack:
        conn = stack.enter_context(SpotConnection(config))
        print("Connected:", conn)
        stream = conn.create_cam_stream(stream_id=args.stream_id)
        stream.start_streaming(build_camera_mask(cameras))
        order = [c.name.lower() for c in stream.get_camera_order()]
        print("Camera order:", order)
        print("SPACE = save pair, q = quit")

        try:
            while True:
                # With run_pipeline=False this returns (rgb, depth). It returns a
                # third pipeline-output element ONLY when run_pipeline=True (that
                # is why basic_streaming.py unpacks three values in vision mode).
                rgb_images, _depth = stream.get_current_images(
                    timeout=args.timeout, run_pipeline=False
                )
                grays = {order[i]: to_gray_uint8(rgb_images[i])
                         for i in range(len(order))}

                # Build a side-by-side preview with a light detection overlay so
                # you can see whether the board is currently visible in each cam.
                previews = []
                counts = {}
                for name in order:
                    g = grays[name]
                    disp = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                    cc, ci, _, _ = detector.detectBoard(g)
                    n = 0 if ci is None else len(ci)
                    counts[name] = n
                    if cc is not None and n > 0:
                        cv2.aruco.drawDetectedCornersCharuco(disp, cc, ci)
                    color = (0, 255, 0) if n >= 6 else (0, 165, 255)
                    cv2.putText(disp, f"{name}: {n} corners", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    previews.append(disp)

                # match heights for hstack
                h = min(p.shape[0] for p in previews)
                previews = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
                            for p in previews]
                combo = np.hstack(previews)
                cv2.putText(combo,
                            f"pairs: {saved} ({saved - start_index} new)   SPACE=save  q=quit",
                            (10, combo.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Stereo hint: both front cams must see the board (>=6 corners each)
                # for a frame to constrain the frontleft<->frontright extrinsic.
                both_ok = all(counts.get(nm, 0) >= 6
                              for nm in ("frontleft", "frontright"))
                cv2.putText(combo,
                            "BOTH cams OK: good stereo pair" if both_ok
                            else "for stereo: get board into BOTH views",
                            (10, combo.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if both_ok else (0, 165, 255), 2)
                cv2.imshow("calib capture", combo)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    for name in order:
                        p = dirs[name] / f"frame_{saved:04d}.png"
                        cv2.imwrite(str(p), grays[name])
                    print(f"saved frame_{saved:04d}")
                    saved += 1
        finally:
            stream.stop_streaming()
            cv2.destroyAllWindows()
            print(f"Done. Saved {saved - start_index} new pairs "
                  f"({saved} total) to {out_root}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
