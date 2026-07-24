#!/usr/bin/env python3
"""Per-frame auto-white-balance comparison for VISUAL colour match (not ICP).

Grabs one frame of the front cameras from both robots and shows them under raw + three
standard AWB methods, so you can judge whether a per-frame white balance neutralises the
front-right pink cast and makes the two robots' colours agree.

Rows (top->bottom): raw / grayworld / shadesofgray / whitepatch
Cols (left->right): A-frontleft | A-frontright | B-frontleft | B-frontright

Also prints a numeric CROSS-CAMERA cast readout: each camera's chromaticity (R/G, B/G) and
the spread across the four cameras. Neutral gray => R/G≈B/G≈1; smaller spread => the cameras
agree better in colour (the actual "matched appearance" goal). Gray-world forces each image's
own mean to neutral, so judge it by the cross-camera SPREAD, not the per-image ratio.

AWB is applied in the images' delivered (gamma-encoded) space — standard for display AWB.
For strict linear-light correctness you'd decode->gain->encode; kept simple here for eyeballing.

Usage (same connection args as live_pair_icp.py):
    python3 compare_wb.py \
        --robot-ip 128.148.138.22 --robot-ip-b 128.148.138.21 \
        --username user --password bigbubbabigbubba --out wb_cmp.png
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PSO = os.path.join(HERE, "..", "PySpotObserver")
for p in (HERE, PSO, os.path.join(PSO, "examples")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common_cli import (                                    # noqa: E402
    add_common_connection_arguments,
    build_camera_mask,
    build_config_from_args,
    parse_camera_list,
)
from spot_ingest import _to_float_rgb_img                   # noqa: E402

CAMERAS = ("frontleft", "frontright")
METHODS = ("raw", "grayworld", "shadesofgray", "whitepatch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_connection_arguments(p)
    p.add_argument("--robot-ip-b", required=True, help="Robot-2 IP address.")
    p.add_argument("--username-b", help="Robot-2 username (defaults to --username).")
    p.add_argument("--password-b", help="Robot-2 password (defaults to --password).")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--white-pct", type=float, default=97.0,
                   help="Percentile treated as 'white' for whitepatch (default 97).")
    p.add_argument("--tile-width", type=int, default=320)
    p.add_argument("--out", type=str, default="wb_cmp.png")
    p.add_argument("--show", action="store_true")
    p.add_argument("--dumps-enabled", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--save-dir", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def build_config_b(config_a, args):
    config_b = copy.deepcopy(config_a)
    config_b.robot_ip = args.robot_ip_b
    if args.username_b:
        config_b.username = args.username_b
    if args.password_b:
        config_b.password = args.password_b
    return config_b


def grab_front(config, mask, timeout, stream_id):
    from pyspotobserver import SpotConnection
    with SpotConnection(config) as conn:
        stream = conn.create_cam_stream(stream_id=stream_id)
        stream.start_streaming(mask)
        try:
            order = [c.name.lower() for c in stream.get_camera_order()]
            rgb_list, _d, _b2w, _eg = stream.get_current_images(
                timeout=timeout, run_pipeline=False, copy=True, include_exposure=True)
        finally:
            stream.stop_streaming()
    return {order[i]: _to_float_rgb_img(rgb_list[i]) for i in range(len(order)) if order[i] in CAMERAS}


def illuminant(img, method, white_pct):
    """Estimate the per-channel illuminant (colour of 'gray'/'white') for an AWB method."""
    x = np.clip(img.reshape(-1, 3).astype(np.float64), 0, 1)
    if method == "grayworld":                    # Minkowski p=1
        e = x.mean(axis=0)
    elif method == "shadesofgray":               # Minkowski p=6 (robust middle ground)
        e = np.power(np.mean(np.power(x, 6), axis=0), 1.0 / 6.0)
    elif method == "whitepatch":                 # bright pixels are white (Retinex-ish)
        e = np.percentile(x, white_pct, axis=0)
    else:
        return np.ones(3)
    return np.maximum(e, 1e-6)


def white_balance(img, method, white_pct):
    if method == "raw":
        return img
    e = illuminant(img, method, white_pct)
    gains = e.mean() / e                          # normalise so overall level is preserved
    return np.clip(img * gains, 0.0, 1.0).astype(np.float32)


def chroma(img):
    """(R/G, B/G) of the image mean — a simple colour-cast readout."""
    m = np.clip(img.reshape(-1, 3).mean(axis=0), 1e-6, None)
    return m[0] / m[1], m[2] / m[1]


def build_montage(frames, args):
    import cv2
    rows = []
    for method in METHODS:
        cells = []
        for tag in ("A", "B"):
            for nm in CAMERAS:
                img = frames[tag][nm]
                wb = white_balance(img, method, args.white_pct)
                h, w = wb.shape[:2]
                tw, th = args.tile_width, int(round(h * args.tile_width / w))
                bgr = cv2.cvtColor((np.clip(wb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr = cv2.resize(bgr, (tw, th))
                rg, bg = chroma(wb)
                cv2.rectangle(bgr, (0, 0), (tw - 1, 22), (0, 0, 0), -1)
                cv2.putText(bgr, f"{method}:{tag}-{nm[:2]} rg={rg:.2f} bg={bg:.2f}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cells.append(bgr)
        rows.append(np.hstack(cells))
    return np.vstack(rows)


def main() -> int:
    args = parse_args()
    config_a = build_config_from_args(args)
    config_b = build_config_b(config_a, args)
    mask = build_camera_mask(parse_camera_list(",".join(CAMERAS)))

    print(f"grabbing frame: A={config_a.robot_ip}  B={config_b.robot_ip} ...")
    frames = {"A": grab_front(config_a, mask, args.timeout, "wb_a"),
              "B": grab_front(config_b, mask, args.timeout, "wb_b")}

    cams = [(tag, nm) for tag in ("A", "B") for nm in CAMERAS if nm in frames[tag]]
    print("\ncross-camera cast (R/G, B/G) and spread — neutral=1.00, smaller spread=cameras agree:")
    for method in METHODS:
        rgs, bgs = [], []
        for tag, nm in cams:
            rg, bg = chroma(white_balance(frames[tag][nm], method, args.white_pct))
            rgs.append(rg)
            bgs.append(bg)
        print(f"  {method:13s} "
              + "  ".join(f"{tag}-{nm[:2]}[{rg:.2f},{bg:.2f}]"
                          for (tag, nm), rg, bg in zip(cams, rgs, bgs))
              + f"   | spread R/G={np.std(rgs):.3f} B/G={np.std(bgs):.3f}")

    montage = build_montage(frames, args)
    import cv2
    cv2.imwrite(args.out, montage)
    print(f"\nwrote {args.out}  ({montage.shape[1]}x{montage.shape[0]})  "
          f"rows: {' / '.join(METHODS)}")
    if args.show:
        cv2.imshow("white balance comparison", montage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
