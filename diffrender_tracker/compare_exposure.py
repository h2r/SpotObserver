#!/usr/bin/env python3
"""Side-by-side 2D comparison of the brightness-reconciliation modes, so you can judge
exposure/colour matching on the ACTUAL camera images instead of a merged point cloud.

Grabs one frame of the front cameras from BOTH robots and writes a montage: three rows
(raw / balance / exposure), each row showing [A-frontleft | A-frontright | B-frontleft |
B-frontright]. Scan DOWN a column to see what a mode does to one camera; scan ACROSS a row
to see whether the two robots agree. Each tile is labelled with its exp*gain.

  * raw       — images untouched.
  * balance   — balance_intensity per robot (gray-world mean/std across that robot's FL/FR);
                also fixes colour seam but is estimated from pixels and does NOT tie the two
                robots together.
  * exposure  — normalize_exposure with a SHARED ref (NOMINAL_EG) across ALL four images;
                metadata-exact luminance match that puts both robots on one radiance scale.

Usage (same connection args as live_pair_icp.py):
    python3 compare_exposure.py \
        --robot-ip 128.148.138.22 --robot-ip-b 128.148.138.21 \
        --username user --password bigbubbabigbubba \
        --out exposure_cmp.png            # then open the PNG
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
from spot_ingest import (                                   # noqa: E402
    NOMINAL_EG,
    balance_intensity,
    normalize_exposure,
    _to_float_rgb_img,
)

CAMERAS = ("frontleft", "frontright")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_connection_arguments(p)                      # robot-1: --robot-ip/--username/...
    p.add_argument("--robot-ip-b", required=True, help="Robot-2 IP address.")
    p.add_argument("--username-b", help="Robot-2 username (defaults to --username).")
    p.add_argument("--password-b", help="Robot-2 password (defaults to --password).")
    p.add_argument("--timeout", type=float, default=10.0, help="Per-frame retrieval timeout (s).")
    p.add_argument("--exposure-ref", type=float, default=NOMINAL_EG,
                   help=f"Shared ref exp*gain for the exposure row (default {NOMINAL_EG}).")
    p.add_argument("--tile-width", type=int, default=320, help="Montage tile width (px).")
    p.add_argument("--out", type=str, default="exposure_cmp.png", help="Montage PNG path.")
    p.add_argument("--show", action="store_true", help="Also cv2.imshow the montage.")
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
    """Connect, stream the front cameras, return {name: (rgb_float[HxWx3], exp_gain)} for one frame."""
    from pyspotobserver import SpotConnection
    with SpotConnection(config) as conn:
        stream = conn.create_cam_stream(stream_id=stream_id)
        stream.start_streaming(mask)
        try:
            order = [c.name.lower() for c in stream.get_camera_order()]
            rgb_list, _depth, _b2w, eg = stream.get_current_images(
                timeout=timeout, run_pipeline=False, copy=True, include_exposure=True)
        finally:
            stream.stop_streaming()
    eg = eg or [0.0] * len(order)
    out = {}
    for i, nm in enumerate(order):
        if nm in CAMERAS:
            out[nm] = (_to_float_rgb_img(rgb_list[i]), float(eg[i]) if i < len(eg) else 0.0)
    return out


def apply_modes(imgsA, imgsB, ref):
    """Return {mode: [(label, rgb_float, eg), ...]} in tile order A-FL, A-FR, B-FL, B-FR."""
    def ordered(imgs, tag):
        return [(f"{tag}-{nm}", imgs[nm][0], imgs[nm][1]) for nm in CAMERAS if nm in imgs]

    tilesA, tilesB = ordered(imgsA, "A"), ordered(imgsB, "B")
    tiles = tilesA + tilesB
    H, W = tiles[0][1].shape[:2]

    def as_flat(t):
        return [img.reshape(-1, 3) for _, img, _ in t]

    def to_imgs(flats, ref_tiles):
        return [(lab, f.reshape(H, W, 3), eg) for (lab, _, eg), f in zip(ref_tiles, flats)]

    # raw
    raw = [(lab, img, eg) for lab, img, eg in tiles]
    # balance: per robot (across that robot's FL/FR), matching the tracker's per-source fuse
    balA = balance_intensity(as_flat(tilesA)) if len(tilesA) > 1 else [t[1].reshape(-1, 3) for t in tilesA]
    balB = balance_intensity(as_flat(tilesB)) if len(tilesB) > 1 else [t[1].reshape(-1, 3) for t in tilesB]
    balance = to_imgs(list(balA) + list(balB), tiles)
    # exposure: shared ref across ALL four -> both robots on one radiance scale
    egs = [eg for _, _, eg in tiles]
    expo = normalize_exposure(as_flat(tiles), egs, ref=ref)
    exposure = to_imgs(expo, tiles)
    return {"raw": raw, "balance": balance, "exposure": exposure}


def build_montage(modes, tile_w):
    import cv2
    rows = []
    for mode in ("raw", "balance", "exposure"):
        tiles = modes[mode]
        cells = []
        for lab, img, eg in tiles:
            h, w = img.shape[:2]
            tw, th = tile_w, int(round(h * tile_w / w))
            bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (tw, th))
            cv2.rectangle(bgr, (0, 0), (tw - 1, 22), (0, 0, 0), -1)
            cv2.putText(bgr, f"{mode}:{lab} eg={eg:.3f}", (4, 16),
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
    imgsA = grab_front(config_a, mask, args.timeout, "expo_cmp_a")
    imgsB = grab_front(config_b, mask, args.timeout, "expo_cmp_b")

    print("\nexp*gain per camera (the brightness each auto-exposure applied):")
    for tag, imgs in (("A", imgsA), ("B", imgsB)):
        for nm in CAMERAS:
            if nm in imgs:
                print(f"  {tag}-{nm:11s} eg={imgs[nm][1]:.4f}")
    print(f"  shared exposure ref = {args.exposure_ref:.4f}")

    modes = apply_modes(imgsA, imgsB, args.exposure_ref)
    montage = build_montage(modes, args.tile_width)

    import cv2
    cv2.imwrite(args.out, montage)
    print(f"\nwrote {args.out}  ({montage.shape[1]}x{montage.shape[0]})  rows: raw / balance / exposure")
    if args.show:
        cv2.imshow("exposure comparison (raw / balance / exposure)", montage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
