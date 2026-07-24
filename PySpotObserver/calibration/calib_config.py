#!/usr/bin/env python3
"""
Shared configuration for the whole calibration pipeline.

generate_charuco.py, capture_calib.py, and calibrate_fisheye.py all import this
so the board definition is identical everywhere. If you change the board, change
it ONCE here.

>>> These values describe the EXISTING physical calib.io board <<<
    Printed footer on the board reads:
        www.calib.io | 9x12 | Checker Size: 60 mm | Marker Size: 45 mm
        Dictionary Aruco DICT_5X5.
"""
import cv2

# --- ChArUco board geometry -------------------------------------------------
# squaresX x squaresY = number of chessboard squares across / down.
# The calib.io footer says "9x12" (rows x cols). Held in its landscape design
# orientation the board is 12 squares wide x 9 tall, so X=12, Y=9.
#   NOTE: if verify_board.py finds 0 corners, try swapping these (9, 12). The
#   marker layout is orientation-specific, so only the correct pairing detects.
SQUARES_X = 12
SQUARES_Y = 9

# SQUARE_LENGTH_M: edge length of one chessboard square, in METERS.
#   >>> This is the number that sets metric scale for the whole pipeline. <<<
#   From the printed board: "Checker Size: 60 mm" -> 0.060 m.
#   (Optional sanity check: verify with calipers that a printed square really is
#   60.0 mm; a mis-scaled printout would silently bias every distance.)
SQUARE_LENGTH_M = 0.060

# MARKER_LENGTH_M: edge length of the aruco marker inside a square, in METERS.
#   From the printed board: "Marker Size: 45 mm" -> 0.045 m.
MARKER_LENGTH_M = 0.045

# ArUco dictionary. The board footer says "DICT_5X5" (5x5-bit markers).
#   IMPORTANT: DICT_5X5_50 / _100 / _250 / _1000 use DIFFERENT bit patterns for
#   the same marker id, so the variant must match what calib.io actually
#   generated -- not just be "big enough". calib.io's default for these boards
#   is _1000. If verify_board.py detects markers with a different variant, set
#   that one here.
ARUCO_DICT = cv2.aruco.DICT_5X5_1000

# LEGACY_PATTERN: calib.io ChArUco boards use the pre-OpenCV-4.6 marker layout.
#   On OpenCV >= 4.6 (this repo was validated on 4.13) you MUST set the board to
#   legacy mode, otherwise detected marker ids map to the wrong chessboard
#   corners and calibration is silently wrong. Keep True for the calib.io board.
#   verify_board.py sweeps this both ways and tells you which is right.
LEGACY_PATTERN = True


def make_board():
    """Return (dictionary, CharucoBoard, CharucoDetector) for the config above."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )
    # Must be set before detection so marker-id -> corner mapping matches the
    # physical (legacy-layout) calib.io board.
    board.setLegacyPattern(LEGACY_PATTERN)
    detector = cv2.aruco.CharucoDetector(board)
    return dictionary, board, detector
