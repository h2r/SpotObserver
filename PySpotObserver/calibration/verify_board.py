#!/usr/bin/env python3
"""
Pre-flight check: confirm calib_config.py actually matches the physical board
BEFORE you spend time capturing 30-60 frames per robot.

Give it one image that clearly shows the whole ChArUco board -- either a phone
photo, or (better) one saved capture frame, e.g.:

    python verify_board.py calib/spot/frontleft/frame_0000.png
    python verify_board.py my_board_photo.jpg

What it does:
  1. Runs detection with EXACTLY the current calib_config.py settings and reports
     how many markers + interpolated chessboard corners it finds.
  2. Sweeps the two settings that silently break calib.io detection -- the legacy
     pattern flag and the DICT_5X5 variant -- plus the X/Y square-count order, and
     prints which combination detects best.

Read the table, then set calib_config.py (ARUCO_DICT, LEGACY_PATTERN, and if
needed SQUARES_X/SQUARES_Y) to the winning row. A good result finds ~all markers
and a large number of chessboard corners; 0/0 means the config is still wrong.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np

import calib_config as cfg
from calib_config import (
    SQUARES_X,
    SQUARES_Y,
    SQUARE_LENGTH_M,
    MARKER_LENGTH_M,
    make_board,
)

# Candidate settings to sweep. Family stays 5x5 (from the printed footer); only
# the variant (bit patterns differ per variant) and legacy flag are uncertain.
DICT_CANDIDATES = [
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_5X5_1000", cv2.aruco.DICT_5X5_1000),
]


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def try_detect(gray, squares_xy, dict_id, legacy):
    """Return (n_markers, n_charuco_corners) for one board configuration."""
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        squares_xy, SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary
    )
    board.setLegacyPattern(legacy)
    detector = cv2.aruco.CharucoDetector(board)
    cc, ci, mc, mi = detector.detectBoard(gray)
    n_markers = 0 if mi is None else len(mi)
    n_corners = 0 if ci is None else len(ci)
    return n_markers, n_corners


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python verify_board.py <image_of_the_board>")
        return 2
    gray = load_gray(Path(sys.argv[1]))
    print(f"image: {sys.argv[1]}  ({gray.shape[1]}x{gray.shape[0]})\n")

    # 1) exactly what calib_config.py is set to right now
    _, _, detector = make_board()
    cc, ci, mc, mi = detector.detectBoard(gray)
    n_m = 0 if mi is None else len(mi)
    n_c = 0 if ci is None else len(ci)
    print("== current calib_config.py ==")
    print(f"   squares=({SQUARES_X},{SQUARES_Y})  dict={cfg.ARUCO_DICT}  "
          f"legacy={cfg.LEGACY_PATTERN}")
    print(f"   -> {n_m} markers, {n_c} chessboard corners"
          + ("   (looks good)" if n_c >= 10 else "   (too few -- see sweep below)"))
    print()

    # 2) sweep the uncertain settings
    print("== sweep (markers / corners) ==")
    print(f"{'squares':>9}  {'legacy':>6}  {'dictionary':>14}  markers  corners")
    best = None
    for squares_xy in [(SQUARES_X, SQUARES_Y), (SQUARES_Y, SQUARES_X)]:
        for legacy, (dname, did) in product([True, False], DICT_CANDIDATES):
            nm, nc = try_detect(gray, squares_xy, did, legacy)
            flag = ""
            if best is None or nc > best[0]:
                best = (nc, squares_xy, legacy, dname)
                flag = "  <-- best so far"
            print(f"{str(squares_xy):>9}  {str(legacy):>6}  {dname:>14}"
                  f"  {nm:>7}  {nc:>7}{flag}")

    print()
    if best and best[0] >= 10:
        nc, sq, legacy, dname = best
        print("Recommended calib_config.py settings:")
        print(f"   SQUARES_X, SQUARES_Y = {sq[0]}, {sq[1]}")
        print(f"   ARUCO_DICT    = cv2.aruco.{dname}")
        print(f"   LEGACY_PATTERN = {legacy}")
    else:
        print("No configuration detected the board well. Re-shoot the image with the")
        print("whole board flat, filling the frame, no glare -- then re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
