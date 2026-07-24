#!/usr/bin/env python3
"""
Color calibration for Spot cameras from ColorChecker Classic (24-patch) images.

Fits a per-camera 3x3 color-correction matrix (CCM) that maps each camera's measured
color to the ColorChecker sRGB reference, in the EXACT convention the runtime applies
it (pyspotobserver/camera_stream.py::_apply_ccm_inplace):

    corrected_linear = measured_linear @ M       # row-vector, RGB order, LINEAR light

The fit is done entirely in linear light on RGB-ordered pixels, so it agrees with the
application. (A mismatch here — fitting in gamma/sRGB space, or in BGR order — is the
most likely reason the previous matrices imposed a yellow cast.)

Images live in calibration/Color_calibration/ as {g,t}-<camera>_fisheye_image.jpg
(g = GOUGER 128.148.138.21, t = TUSKER 128.148.138.22).

Per image you click the 4 CORNER patches of the chart, in printed orientation (use the
"colorchecker" logo as top-left), in this order:
    1) dark-skin  (brown)   = TOP-LEFT
    2) bluish-green         = TOP-RIGHT
    3) black                = BOTTOM-RIGHT
    4) white                = BOTTOM-LEFT
The 24 patch centers are recovered by a perspective homography from those 4 corners
(handles the oblique hand-held tilt); each patch color is the median of its center.
Clicked corners are cached to corners_cache.json so re-runs don't need re-clicking.

Outputs (in the images dir):
  - ccm_fit_results.json           raw matrices + per-camera error
  - new_ccms.py                    ready-to-paste _GOUGER_CCMS / _TUSKER_CCMS dicts
  - <robot>-<camera>_preview.png   measured vs corrected vs reference patch strip

Run (needs a display for clicking):
    python calibration/calibrate_color.py
    python calibration/calibrate_color.py --only t-frontleft   # redo one image
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ColorChecker Classic 24-patch reference sRGB (8-bit, D65), patch order 1..24
# row-major: row0 = patches 1-6, row1 = 7-12, row2 = 13-18, row3 = 19-24.
COLORCHECKER_SRGB = np.array(
    [
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
        [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
        [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
    ],
    dtype=np.float64,
)

N_COLS, N_ROWS = 6, 4  # ColorChecker layout

ROBOT_BY_PREFIX = {"g": ("GOUGER", "128.148.138.21"), "t": ("TUSKER", "128.148.138.22")}
CAMERA_BY_TOKEN = {
    "frontleft": "FRONTLEFT",
    "frontright": "FRONTRIGHT",
    "left": "LEFT",
    "right": "RIGHT",
    "back": "BACK",
    "hand": "HAND",
}


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> linear light. Matches _apply_ccm_inplace's decode exactly."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear light -> sRGB [0,1]. Matches _apply_ccm_inplace's encode exactly."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def parse_image_name(path: Path) -> tuple[str, str, str, str] | None:
    """'t-frontleft_fisheye_image.jpg' -> (robot_name, robot_ip, camera_const, token)."""
    stem = path.stem  # t-frontleft_fisheye_image
    if "-" not in stem:
        return None
    prefix, rest = stem.split("-", 1)
    if prefix not in ROBOT_BY_PREFIX:
        return None
    token = rest.split("_", 1)[0]  # frontleft
    if token not in CAMERA_BY_TOKEN:
        return None
    robot_name, robot_ip = ROBOT_BY_PREFIX[prefix]
    return robot_name, robot_ip, CAMERA_BY_TOKEN[token], token


def click_corners(img_rgb: np.ndarray, title: str) -> np.ndarray:
    """Show the image; return the 4 clicked corner-patch centers as (4,2) float xy."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        f"{title}\nClick 4 CORNER patch centers in order:  "
        "1) brown (TL)   2) bluish-green (TR)   3) black (BR)   4) white (BL)",
        fontsize=10,
    )
    ax.axis("off")
    pts = plt.ginput(4, timeout=0)
    plt.close(fig)
    if len(pts) != 4:
        raise RuntimeError(f"Expected 4 clicks, got {len(pts)}")
    return np.array(pts, dtype=np.float64)


def patch_centers_from_corners(corners: np.ndarray) -> np.ndarray:
    """Perspective-map the 4 corner-patch centers to all 24 patch centers (24,2)."""
    # Canonical grid coords of the 4 clicked corners: TL, TR, BR, BL.
    canon4 = np.array(
        [[0, 0], [N_COLS - 1, 0], [N_COLS - 1, N_ROWS - 1], [0, N_ROWS - 1]], dtype=np.float32
    )
    H = cv2.getPerspectiveTransform(canon4, corners.astype(np.float32))
    grid = np.array([[c, r] for r in range(N_ROWS) for c in range(N_COLS)], dtype=np.float32)
    mapped = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H).reshape(-1, 2)
    return mapped.astype(np.float64)


def sample_patches(img_rgb01: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Median RGB [0,1] over a small window at each of the 24 centers -> (24,3)."""
    # Window half-size from mean spacing between horizontally-adjacent centers.
    row0 = centers[:N_COLS]
    spacing = float(np.mean(np.linalg.norm(np.diff(row0, axis=0), axis=1)))
    half = max(3, int(round(0.18 * spacing)))
    h, w, _ = img_rgb01.shape
    out = np.zeros((24, 3), dtype=np.float64)
    for i, (x, y) in enumerate(centers):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - half), min(w, xi + half + 1)
        y0, y1 = max(0, yi - half), min(h, yi + half + 1)
        region = img_rgb01[y0:y1, x0:x1].reshape(-1, 3)
        out[i] = np.median(region, axis=0)
    return out


_LUMA_LIN = np.array([0.2126, 0.7152, 0.0722])  # Rec.709 luminance, linear light


def fit_ccm(measured01: np.ndarray, ref_srgb: np.ndarray, normalize_luma: bool = True) -> np.ndarray:
    """Least-squares 3x3 M with measured_linear @ M ~= reference_linear (runtime convention).

    - Patches with a clipped (>0.985) or crushed (<0.02) measured channel are dropped:
      a linear fit can't trust saturated data, and the front-left charts are overexposed.
    - normalize_luma scales M so the neutral ramp keeps its own luminance, i.e. the CCM
      does white-balance/color ONLY and adds no per-shot brightness gain. This is what
      lets the left/right cameras match: brightness is left to live auto-exposure (which
      already balances the two) instead of being baked differently into each matrix.
    """
    meas_lin = srgb_to_linear(measured01)               # (24,3)
    ref_lin = srgb_to_linear(ref_srgb / 255.0)          # (24,3)
    keep = np.all((measured01 > 0.02) & (measured01 < 0.985), axis=1)
    A, B = (meas_lin[keep], ref_lin[keep]) if int(keep.sum()) >= 8 else (meas_lin, ref_lin)
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    if normalize_luma:
        grays = srgb_to_linear(measured01[19:23])       # neutral 8/6.5/5/3.5 (skip white/black)
        in_l = float((grays @ _LUMA_LIN).mean())
        out_l = float(((grays @ M) @ _LUMA_LIN).mean())
        if out_l > 1e-6:
            M = M * (in_l / out_l)
    return M.astype(np.float64)


def apply_ccm(measured01: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Run measured sRGB [0,1] through the runtime pipeline -> corrected sRGB [0,1]."""
    return linear_to_srgb(srgb_to_linear(measured01) @ M)


def rms_srgb(a01: np.ndarray, ref_srgb: np.ndarray) -> float:
    """RMS error in 8-bit sRGB units between predicted [0,1] and reference 0-255."""
    return float(np.sqrt(np.mean((a01 * 255.0 - ref_srgb) ** 2)))


def save_preview(path: Path, measured01, corrected01, ref_srgb, title: str) -> None:
    """Stacked strips: measured / corrected / reference, 24 patches wide."""
    def strip(colors01):
        row = np.clip(colors01, 0, 1).reshape(N_ROWS, N_COLS, 3)
        return np.repeat(np.repeat(row, 40, axis=0), 40, axis=1)

    meas = strip(measured01)
    corr = strip(corrected01)
    ref = strip(ref_srgb / 255.0)
    gap = np.ones((10, meas.shape[1], 3))
    canvas = np.vstack([meas, gap, corr, gap, ref])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(f"{title}\ntop: measured   middle: corrected   bottom: reference", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fmt_matrix(M: np.ndarray) -> str:
    rows = ",\n".join(
        "            [" + ", ".join(f"{v: .7f}" for v in row) + "]" for row in M
    )
    return "np.array(\n        [\n" + rows + ",\n        ],\n        dtype=np.float32,\n    )"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = Path(__file__).with_name("Color_calibration")
    ap.add_argument("--images-dir", type=Path, default=default_dir)
    ap.add_argument("--only", help="Process only this image stem prefix, e.g. 't-frontleft'.")
    ap.add_argument("--reclick", action="store_true", help="Ignore cached corners and re-click.")
    args = ap.parse_args()

    img_dir: Path = args.images_dir
    images = sorted(p for p in img_dir.glob("*.jpg") if parse_image_name(p))
    if args.only:
        images = [p for p in images if p.stem.startswith(args.only)]
    if not images:
        print(f"No matching images in {img_dir}")
        return 1

    cache_path = img_dir / "corners_cache.json"
    corners_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    results: dict[str, dict] = {}  # robot_ip -> {camera_const: M.tolist()}
    errors: list[str] = []

    for path in images:
        parsed = parse_image_name(path)
        assert parsed is not None
        robot_name, robot_ip, camera_const, token = parsed
        label = f"{robot_name} {camera_const} ({path.name})"

        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"!! could not read {path}")
            continue
        img_rgb01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        if not args.reclick and path.name in corners_cache:
            corners = np.array(corners_cache[path.name], dtype=np.float64)
            print(f"[{label}] using cached corners")
        else:
            print(f"[{label}] click the 4 corner patches...")
            try:
                corners = click_corners((img_rgb01 * 255).astype(np.uint8), label)
            except RuntimeError as exc:
                print(f"   skipped: {exc}")
                continue
            corners_cache[path.name] = corners.tolist()
            cache_path.write_text(json.dumps(corners_cache, indent=2))

        centers = patch_centers_from_corners(corners)
        measured = sample_patches(img_rgb01, centers)

        # Exposure/clipping sanity on the neutral ramp (patches 19-24).
        white = measured[18]
        if white.max() > 0.98:
            print(f"   WARNING: white patch near clipping ({white.max():.2f}); fit may be off.")

        M = fit_ccm(measured, COLORCHECKER_SRGB)
        corrected = apply_ccm(measured, M)

        before = rms_srgb(measured, COLORCHECKER_SRGB)
        after = rms_srgb(corrected, COLORCHECKER_SRGB)
        # Neutral-cast check: mean B/G and R/G on the gray patches after correction.
        gray = corrected[18:24]
        rg = float(np.mean(gray[:, 0] / np.clip(gray[:, 1], 1e-6, None)))
        bg = float(np.mean(gray[:, 2] / np.clip(gray[:, 1], 1e-6, None)))
        print(
            f"   RMS sRGB error  before {before:6.1f}  ->  after {after:6.1f}   "
            f"| gray R/G {rg:.2f} B/G {bg:.2f} (target 1.00/1.00)"
        )

        results.setdefault(robot_ip, {})[camera_const] = M.tolist()
        errors.append(f"{robot_name:7s} {camera_const:11s} before {before:6.1f} after {after:6.1f}")
        save_preview(img_dir / f"{robot_name}-{token}_preview.png", measured, corrected, COLORCHECKER_SRGB, label)

    if not results:
        print("Nothing fit.")
        return 1

    (img_dir / "ccm_fit_results.json").write_text(json.dumps(results, indent=2))

    # Emit ready-to-paste dicts for color_correction.py.
    lines = ["import numpy as np", "from .config import CameraType", "", "_IDENTITY_3x3 = np.eye(3, dtype=np.float32)", ""]
    varname = {"128.148.138.21": "_GOUGER_CCMS", "128.148.138.22": "_TUSKER_CCMS"}
    for ip, cams in results.items():
        lines.append(f"{varname.get(ip, ip)}: dict = {{")
        for cam in ("LEFT", "RIGHT", "FRONTLEFT", "FRONTRIGHT", "BACK", "HAND"):
            if cam in cams:
                lines.append(f"    CameraType.{cam}: {fmt_matrix(np.array(cams[cam]))},")
            else:
                lines.append(f"    CameraType.{cam}: _IDENTITY_3x3,")
        lines.append("}")
        lines.append("")
    (img_dir / "new_ccms.py").write_text("\n".join(lines))

    print("\n=== summary (RMS sRGB error, lower is better) ===")
    for e in errors:
        print("  " + e)
    print(f"\nWrote:\n  {img_dir/'new_ccms.py'}  (paste into pyspotobserver/color_correction.py)")
    print(f"  {img_dir/'ccm_fit_results.json'}\n  {img_dir}/<robot>-<camera>_preview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
