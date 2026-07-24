#!/usr/bin/env python3
"""
Step 3: Calibrate one robot's front stereo pair from captured ChArUco frames.

Runs TWO things for the robot folder you point it at:
  (1) per-camera fisheye intrinsics  (K + 4 Kannala-Brandt distortion coeffs)
      for frontleft and frontright, independently  -> cv2.fisheye.calibrate
  (2) the frontleft<->frontright extrinsic (R, T), with intrinsics held fixed
      -> cv2.fisheye.stereoCalibrate

Run once per robot:
    python calibrate_fisheye.py calib/spot
    python calibrate_fisheye.py calib/spot2

Expects the layout produced by capture_calib.py:
    calib/spot/frontleft/frame_XXXX.png
    calib/spot/frontright/frame_XXXX.png   (matching index == same instant)

Writes results to <robot_dir>/calibration.yaml and prints a summary.
Use the SAME printed board (same SQUARE_LENGTH_M in calib_config.py) for both
robots so the two rigs come out in identical metric units.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

from calib_config import make_board, SQUARE_LENGTH_M

# A view must have at least this many ChArUco corners to be used.
MIN_CORNERS = 6
# For the stereo step, a frame must share at least this many corners across L/R.
MIN_SHARED = 6

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

_, BOARD, DETECTOR = make_board()
BOARD_OBJ = BOARD.getChessboardCorners().astype(np.float64)  # (n_charuco, 3), by id


# --------------------------------------------------------------------------- #
# detection helpers
# --------------------------------------------------------------------------- #
def detect(gray: np.ndarray):
    """Return (charuco_corners (N,1,2) float, charuco_ids (N,) int) or (None,None)."""
    cc, ci, _, _ = DETECTOR.detectBoard(gray)
    if ci is None or len(ci) < MIN_CORNERS:
        return None, None
    return cc, ci.flatten()


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def is_degenerate(obj_pts: np.ndarray) -> bool:
    """True if the planar board points are (near-)collinear.

    A view whose detected ChArUco corners all lie on one row or one column
    cannot define a board pose -- cv2.fisheye.calibrate aborts in InitExtrinsics
    ("fabs(norm_u1) > 0") and solvePnP is unreliable. We reject such views by
    checking the 2D spread of the object points: the ratio of the second to the
    first singular value collapses to ~0 for a line, but is O(0.1-1) for a real
    2D patch.
    """
    P = obj_pts.reshape(-1, 3)[:, :2].astype(np.float64)
    P = P - P.mean(axis=0)
    s = np.linalg.svd(P, compute_uv=False)
    return len(s) < 2 or s[0] <= 0 or (s[1] / s[0]) < 1e-4


# --------------------------------------------------------------------------- #
# per-camera fisheye intrinsics (robust to the occasional ill-conditioned view)
# --------------------------------------------------------------------------- #
def calibrate_intrinsics(cam_dir: Path):
    files = sorted(cam_dir.glob("frame_*.png"))
    if not files:
        raise SystemExit(f"No frames in {cam_dir}")

    obj_all, img_all, used_files = [], [], []
    image_size = None
    for f in files:
        g = load_gray(f)
        image_size = g.shape[::-1]  # (w, h)
        cc, ci = detect(g)
        if ci is None:
            continue
        op, ip = BOARD.matchImagePoints(cc, ci.reshape(-1, 1))
        if op is None or len(op) < MIN_CORNERS:
            continue
        if is_degenerate(op):
            continue  # collinear corners -> unusable, would crash InitExtrinsics
        obj_all.append(op.reshape(-1, 1, 3).astype(np.float64))
        img_all.append(ip.reshape(-1, 1, 2).astype(np.float64))
        used_files.append(f)

    if len(obj_all) < 5:
        raise SystemExit(
            f"Only {len(obj_all)} usable views in {cam_dir} (need >=5, more is better)."
        )

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
             | cv2.fisheye.CALIB_FIX_SKEW
             | cv2.fisheye.CALIB_CHECK_COND)

    keep = list(range(len(obj_all)))
    while True:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        objs = [obj_all[i] for i in keep]
        imgs = [img_all[i] for i in keep]
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                objs, imgs, image_size, K, D, flags=flags, criteria=CRITERIA
            )
            return dict(K=K, D=D, rms=rms, n_views=len(keep), image_size=image_size)
        except cv2.error as e:
            m = re.search(r"input array (\d+)", str(e))
            if m is None or len(keep) <= 5:
                # CHECK_COND couldn't localize (or too few left): retry without it
                K = np.zeros((3, 3)); D = np.zeros((4, 1))
                rms, K, D, _, _ = cv2.fisheye.calibrate(
                    objs, imgs, image_size, K, D,
                    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
                    criteria=CRITERIA,
                )
                return dict(K=K, D=D, rms=rms, n_views=len(keep), image_size=image_size)
            bad_local = int(m.group(1))
            dropped = keep.pop(bad_local)
            print(f"    dropped ill-conditioned view {used_files[dropped].name}; "
                  f"{len(keep)} left")


# --------------------------------------------------------------------------- #
# stereo extrinsic via per-frame board pose (solvePnP) + relative-pose averaging
#
# cv2.fisheye.stereoCalibrate is notoriously unstable (it frequently aborts with
# "abs_max < threshold"). Recovering each camera's board pose independently and
# averaging the left->right transform is far more robust and gives the same R,T.
# --------------------------------------------------------------------------- #
def _mat2quat(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                      (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                          0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _quat2mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _avg_rotation(Rs):
    """Markley quaternion averaging (sign-robust via outer-product eigenvector)."""
    M = np.zeros((4, 4))
    for R in Rs:
        q = _mat2quat(R)
        M += np.outer(q, q)
    _, vecs = np.linalg.eigh(M)
    return _quat2mat(vecs[:, -1])


def _board_pose(cc, ci, K, D):
    """Board pose (4x4, board->camera) via fisheye-undistort + solvePnP."""
    obj = BOARD_OBJ[ci].reshape(-1, 1, 3)
    imgp = cc.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.fisheye.undistortPoints(imgp, K, D, P=K)  # -> pinhole pixel coords
    ok, rvec, tvec = cv2.solvePnP(obj, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    Rm, _ = cv2.Rodrigues(rvec)
    Tm = np.eye(4)
    Tm[:3, :3] = Rm
    Tm[:3, 3] = tvec.flatten()
    return Tm


def calibrate_stereo(left_dir: Path, right_dir: Path, intr_l, intr_r):
    left_files = {p.name: p for p in left_dir.glob("frame_*.png")}
    right_files = {p.name: p for p in right_dir.glob("frame_*.png")}
    common = sorted(set(left_files) & set(right_files))

    rels = []  # left->right 4x4 transforms, one per usable pair
    for name in common:
        ccL, ciL = detect(load_gray(left_files[name]))
        ccR, ciR = detect(load_gray(right_files[name]))
        if ciL is None or ciR is None:
            continue
        if len(np.intersect1d(ciL, ciR)) < MIN_SHARED:
            continue
        if is_degenerate(BOARD_OBJ[ciL]) or is_degenerate(BOARD_OBJ[ciR]):
            continue  # collinear in one view -> solvePnP pose unreliable
        TL = _board_pose(ccL, ciL, intr_l["K"], intr_l["D"])
        TR = _board_pose(ccR, ciR, intr_r["K"], intr_r["D"])
        if TL is None or TR is None:
            continue
        rels.append(TR @ np.linalg.inv(TL))  # left camera -> right camera

    if len(rels) < 5:
        raise SystemExit(
            f"Only {len(rels)} frames where both cameras saw the board "
            f"(need >=5). Re-capture with the board more often in the shared view."
        )

    R = _avg_rotation([T[:3, :3] for T in rels])
    Ts = np.array([T[:3, 3] for T in rels])
    T = Ts.mean(axis=0).reshape(3, 1)

    # consistency spread across frames (should be small for a rigid rig)
    ang_spread = np.degrees(np.std([
        np.arccos(np.clip((np.trace(Ri[:3, :3] @ R.T) - 1) / 2, -1, 1)) for Ri in rels
    ]))
    t_spread_mm = float(np.linalg.norm(Ts.std(axis=0)) * 1000)
    return dict(R=R, T=T, n_pairs=len(rels),
                ang_spread_deg=ang_spread, t_spread_mm=t_spread_mm)


# --------------------------------------------------------------------------- #
def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python calibrate_fisheye.py <robot_dir>   (e.g. calib/spot)")
        return 2
    robot_dir = Path(sys.argv[1])
    left_dir = robot_dir / "frontleft"
    right_dir = robot_dir / "frontright"
    for d in (left_dir, right_dir):
        if not d.is_dir():
            print(f"missing {d}"); return 2

    print(f"== {robot_dir}  (square = {SQUARE_LENGTH_M} m) ==")
    print("Calibrating frontleft intrinsics...")
    intr_l = calibrate_intrinsics(left_dir)
    print(f"  frontleft:  rms={intr_l['rms']:.4f} px over {intr_l['n_views']} views")
    print("Calibrating frontright intrinsics...")
    intr_r = calibrate_intrinsics(right_dir)
    print(f"  frontright: rms={intr_r['rms']:.4f} px over {intr_r['n_views']} views")

    print("Calibrating stereo extrinsic (per-frame pose averaging)...")
    stereo = calibrate_stereo(left_dir, right_dir, intr_l, intr_r)
    baseline = float(np.linalg.norm(stereo["T"]))
    print(f"  stereo:     {stereo['n_pairs']} pairs, "
          f"baseline |T| = {baseline*1000:.1f} mm")
    print(f"  consistency across frames: "
          f"rotation std {stereo['ang_spread_deg']:.3f} deg, "
          f"translation std {stereo['t_spread_mm']:.2f} mm  (smaller = better)")

    # --- save ---
    out = robot_dir / "calibration.yaml"
    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)
    fs.write("model", "fisheye_kannala_brandt")
    fs.write("image_width", int(intr_l["image_size"][0]))
    fs.write("image_height", int(intr_l["image_size"][1]))
    fs.write("square_length_m", float(SQUARE_LENGTH_M))
    fs.write("K_frontleft", intr_l["K"]); fs.write("D_frontleft", intr_l["D"])
    fs.write("rms_frontleft", float(intr_l["rms"]))
    fs.write("K_frontright", intr_r["K"]); fs.write("D_frontright", intr_r["D"])
    fs.write("rms_frontright", float(intr_r["rms"]))
    fs.write("R_left_to_right", stereo["R"]); fs.write("T_left_to_right", stereo["T"])
    fs.write("stereo_rotation_std_deg", float(stereo["ang_spread_deg"]))
    fs.write("stereo_translation_std_mm", float(stereo["t_spread_mm"]))
    fs.release()
    print(f"Wrote {out}")

    # quick quality read
    if max(intr_l["rms"], intr_r["rms"]) > 1.0:
        print("\nNOTE: an intrinsic rms above ~1 px usually means thin coverage at the")
        print("image edges. Re-capture with more corner/edge and in-plane-roll views.")
    if stereo["ang_spread_deg"] > 0.5 or stereo["t_spread_mm"] > 5.0:
        print("\nNOTE: the stereo extrinsic varies a lot frame-to-frame, which points")
        print("to weak intrinsics or too few shared views. Improve those before trust.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
