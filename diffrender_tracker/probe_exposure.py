#!/usr/bin/env python3
"""Probe a Spot for exposure controls + read the live exposure/gain curve.

Answers two questions per camera source:
  1. Does the source advertise any settable custom_params (e.g. exposure)?
  2. What exposure_duration / gain did the camera actually use this frame?

Usage:
    python3 probe_exposure.py --robot-ip 128.148.138.22 \
        --username user --password bigbubbabigbubba
    # GOUGER 128.148.138.21 / TUSKER 128.148.138.22
"""
import argparse

from bosdyn.client import create_standard_sdk


def dur_s(d):
    return d.seconds + d.nanos * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    args = ap.parse_args()

    sdk = create_standard_sdk("expo-probe")
    robot = sdk.create_robot(args.robot_ip)
    robot.authenticate(args.username, args.password)
    img = robot.ensure_client("image")

    print("=== 1) settable custom_params per source ===")
    sources = img.list_image_sources()
    for s in sources:
        cp = s.custom_params
        specs = dict(cp.specs) if cp.specs else {}
        if specs:
            print(f"  {s.name:30s} SETTABLE -> {list(specs.keys())}")
        else:
            print(f"  {s.name:30s} (auto only, no custom params)")

    print("\n=== 2) live exposure_duration / gain per source ===")
    names = [s.name for s in sources]
    responses = img.get_image_from_sources(names)
    for r in responses:
        cp = r.shot.capture_params
        print(f"  {r.source.name:30s} exp_s={dur_s(cp.exposure_duration):.6f}  gain={cp.gain:.3f}")


if __name__ == "__main__":
    main()
