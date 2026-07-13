#!/usr/bin/env python3
"""
Step 1: Generate a ChArUco board to print.

Run this, print the PNG at 100% / actual-size (NO "fit to page"), mount it dead
flat on a rigid board, then MEASURE one printed square edge with calipers and put
that measured number into SQUARE_LENGTH_M in calib_config.py.

The board config here and in calib_config.py MUST match.
"""
import cv2
import numpy as np

from calib_config import (
    SQUARES_X,
    SQUARES_Y,
    SQUARE_LENGTH_M,
    MARKER_LENGTH_M,
    ARUCO_DICT,
)


def main() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )

    # Render big so it prints sharp. Aspect follows the board's square count.
    # These are just pixels for the PNG; real-world size comes from how you print
    # it and the caliper measurement you feed back into calib_config.py.
    px_per_square = 300
    w = SQUARES_X * px_per_square
    h = SQUARES_Y * px_per_square
    img = board.generateImage((w, h), marginSize=px_per_square // 2)

    out = "charuco_board.png"
    cv2.imwrite(out, img)
    print(f"Wrote {out}  ({SQUARES_X}x{SQUARES_Y} squares)")
    print("Print at 100% scale, mount flat, then measure a square and set")
    print("SQUARE_LENGTH_M in calib_config.py to the MEASURED value (in meters).")


if __name__ == "__main__":
    main()
