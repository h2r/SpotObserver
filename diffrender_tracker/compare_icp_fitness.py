#!/usr/bin/env python3
"""Controlled A/B/C of the brightness modes by ICP fitness — the metric that matters.

Captures ONE frame-pair from the two robots, then fuses+registers three times on that
IDENTICAL geometry, changing only the brightness reconciliation (raw / balance / exposure).
Because the xyz is byte-for-byte the same across modes, any difference in colored-ICP fitness
is purely the colour effect — no scene drift, no per-run confound. All three colored-ICP runs
seed from ONE common bootstrap pose so they start from the same place.

Higher overlap_fitness / lower icp_rmse = colours agree better on the overlap.

Usage (same connection args as live_pair_icp.py):
    python3 compare_icp_fitness.py \
        --robot-ip 128.148.138.22 --calib calib/spot \
        --robot-ip-b 128.148.138.21 --calib-b calib/spot2 \
        --username user --password bigbubbabigbubba
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
    backproject_fisheye,
    balance_intensity,
    load_fisheye_calib,
    normalize_exposure,
    voxel_downsample,
    _to_float_rgb_img,
)
from bootstrap import bootstrap_register, colored_icp_register   # noqa: E402

CAMERAS = ("frontleft", "frontright")
MODES = ("raw", "balance", "exposure")     # "raw" == bright_mode "none"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_connection_arguments(p)
    p.add_argument("--calib", required=True, help="Robot-1 calib dir or calibration.yaml.")
    p.add_argument("--robot-ip-b", required=True, help="Robot-2 IP address.")
    p.add_argument("--calib-b", required=True, help="Robot-2 calib dir or calibration.yaml.")
    p.add_argument("--username-b", help="Robot-2 username (defaults to --username).")
    p.add_argument("--password-b", help="Robot-2 password (defaults to --password).")
    p.add_argument("--timeout", type=float, default=10.0, help="Per-frame retrieval timeout (s).")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min-depth", type=float, default=0.2)
    p.add_argument("--max-depth", type=float, default=3.0)
    p.add_argument("--target", type=int, default=20000)
    p.add_argument("--voxel", type=float, default=0.05, help="ICP coarse voxel / scale (m).")
    p.add_argument("--exposure-ref", type=float, default=NOMINAL_EG)
    p.add_argument("--dumps-enabled", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--save-dir", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def resolve_calib(path_arg: str) -> str:
    if os.path.exists(path_arg):
        return path_arg
    alt = os.path.join(PSO, "examples", path_arg)
    return alt if os.path.exists(alt) else path_arg


def build_config_b(config_a, args):
    config_b = copy.deepcopy(config_a)
    config_b.robot_ip = args.robot_ip_b
    if args.username_b:
        config_b.username = args.username_b
    if args.password_b:
        config_b.password = args.password_b
    return config_b


def grab_raw(config, mask, timeout, stream_id):
    """One frame: return {name: (rgb_float, depth, eg)} for the front cameras."""
    from pyspotobserver import SpotConnection
    with SpotConnection(config) as conn:
        stream = conn.create_cam_stream(stream_id=stream_id)
        stream.start_streaming(mask)
        try:
            order = [c.name.lower() for c in stream.get_camera_order()]
            rgb_list, depth_list, _b2w, eg = stream.get_current_images(
                timeout=timeout, run_pipeline=False, copy=True, include_exposure=True)
        finally:
            stream.stop_streaming()
    eg = eg or [0.0] * len(order)
    out = {}
    for i, nm in enumerate(order):
        if nm in CAMERAS:
            out[nm] = (_to_float_rgb_img(rgb_list[i]), depth_list[i],
                       float(eg[i]) if i < len(eg) else 0.0)
    return out


def fuse(raw, calib, mode, args):
    """Replicate SpotCloudSource.latest() fuse for a captured frame under `mode`. Colour kept
    (colored ICP needs it). Returns (xyz, rgb) or None."""
    Rlr, Tlr = calib["R"], calib["T"]
    pts_all, col_all, eg_used = [], [], []
    for nm in CAMERAS:
        if nm not in raw:
            continue
        rgb_img, dep, eg = raw[nm]
        pts, cols = backproject_fisheye(dep, rgb_img, calib[f"K_{nm}"], calib[f"D_{nm}"],
                                        args.stride, args.min_depth, args.max_depth)
        if nm == "frontright" and len(pts):
            pts = (pts - Tlr) @ Rlr
        if len(pts):
            pts_all.append(pts)
            col_all.append(cols)
            eg_used.append(eg)
    if not pts_all:
        return None
    if mode == "exposure":
        col_all = normalize_exposure(col_all, eg_used, ref=args.exposure_ref)
    elif mode == "balance" and len(col_all) > 1:
        col_all = balance_intensity(col_all)
    # mode == "raw": leave as-is
    return voxel_downsample(np.vstack(pts_all), np.vstack(col_all), target=args.target)


def main() -> int:
    args = parse_args()
    config_a = build_config_from_args(args)
    config_b = build_config_b(config_a, args)
    calib_a = load_fisheye_calib(resolve_calib(args.calib))
    calib_b = load_fisheye_calib(resolve_calib(args.calib_b))
    mask = build_camera_mask(parse_camera_list(",".join(CAMERAS)))

    print(f"capturing one frame-pair: A={config_a.robot_ip}  B={config_b.robot_ip} ...")
    rawA = grab_raw(config_a, mask, args.timeout, "icpcmp_a")
    rawB = grab_raw(config_b, mask, args.timeout, "icpcmp_b")
    for tag, raw in (("A", rawA), ("B", rawB)):
        for nm in CAMERAS:
            if nm in raw:
                print(f"  {tag}-{nm:11s} eg={raw[nm][2]:.4f}")

    # Geometry is identical across modes, so fuse once per mode but bootstrap the seed ONCE
    # (on raw) and reuse it for all three colored-ICP runs -> same starting pose, fair compare.
    clouds = {}
    for m in MODES:
        ca = fuse(rawA, calib_a, m, args)
        cb = fuse(rawB, calib_b, m, args)
        if ca is None or cb is None:
            print(f"  [{m}] no valid points; aborting")
            return 1
        clouds[m] = (ca, cb)

    (xa, ra), (xb, rb) = clouds["raw"]
    print(f"\nclouds: A={len(xa)} pts  B={len(xb)} pts   (identical geometry across modes)")
    print("bootstrapping a common seed on raw geometry ...")
    T_init, binfo = bootstrap_register(xa, ra, xb, rb, voxel=args.voxel)
    print(f"  seed overlap_fitness={binfo.get('overlap_fitness', float('nan')):.4f}")

    print("\nmode      icp_fitness  icp_rmse(cm)  overlap_fitness")
    results = {}
    for m in MODES:
        (xa, ra), (xb, rb) = clouds[m]
        _T, info = colored_icp_register(xa, ra, xb, rb, T_init=T_init, voxel=args.voxel)
        results[m] = info
        print(f"{m:9s}   {info['icp_fitness']:.4f}      {info['icp_rmse']*100:5.2f}        "
              f"{info['overlap_fitness']:.4f}")

    best = max(results, key=lambda k: results[k]["overlap_fitness"])
    print(f"\nbest overlap_fitness: {best}  "
          f"({results[best]['overlap_fitness']:.4f})")
    print("Higher overlap_fitness / lower rmse = colours agree better on the overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
