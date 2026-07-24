"""
Milestone 1, Step 1 — synthetic colored point-cloud pair.

Generates two colored point clouds (XYZ + RGB) of the SAME scene seen from two
different robot frames, with a KNOWN ground-truth transform between them. This is
the test fixture for the differentiable photometric tracker: your optimizer should
recover T_gt (starting from a perturbed guess) by making B's render match A's.

The scene is deliberately a *textured corner* (one wall + floor):
  - Geometry alone is degenerate — you can slide along the wall / floor and the
    depth barely changes. A depth-only loss will NOT lock those DoF.
  - The color texture has strong spatial gradient in exactly those directions, so
    a photometric (color) loss CAN lock them.
That mismatch is the whole point of Milestone 1: it proves the color loss buys you
something depth cannot.

Conventions
-----------
World frame == Robot A's frame.
  - Cloud A is stored directly in the world/A frame.
  - Cloud B is stored in B's OWN frame. T_gt maps B -> A:  p_A = R_gt @ p_B + t_gt.
  - So the correct answer your tracker must recover is exactly T_gt.

The two clouds are sampled INDEPENDENTLY over an overlapping region (they do NOT
share point identities) and B gets a brightness offset, so the loss must work on
non-corresponding samples across two "cameras" — like the real problem.

Outputs (in ./data/):
  cloud_A.npz, cloud_B.npz   -> keys: xyz (N,3) float32, rgb (N,3) float32 in [0,1]
  cloud_B.npz also stores    -> R_gt (3,3), t_gt (3,), T_gt (4,4)   [the answer]
  cloud_A.ply, cloud_B.ply   -> for eyeballing in Open3D / CloudCompare / MeshLab
"""

import os
import numpy as np

# ----------------------------------------------------------------------------- config
SEED               = 0
N_POINTS           = 20_000      # per cloud (matches your ~20k onboard-downsample target)
WALL_FRACTION      = 0.5         # split of points between wall and floor

# Scene extents (metres). Wall is the back plane (y=0), floor extends forward (z=0).
X_RANGE            = (-3.0, 3.0) # along the corner
WALL_HEIGHT        = (0.0, 2.5)  # z on the wall
FLOOR_DEPTH        = (0.0, 3.0)  # y on the floor

# Partial overlap: A sees the left, B sees the right, they share the middle.
A_XMAX             =  1.5        # A covers x in [X_RANGE[0], A_XMAX]
B_XMIN             = -1.5        # B covers x in [B_XMIN, X_RANGE[1]]   -> overlap [-1.5, 1.5]

# Ground-truth transform B -> A. Modest: a few degrees + ~15 cm, warm-start scale.
GT_EULER_DEG       = (3.0, -5.0, 4.0)   # rotation about x, y, z (degrees)
GT_TRANSLATION     = (0.15, -0.08, 0.05)

# Realism knobs
POS_NOISE_STD      = 0.005       # 5 mm sensor noise on point positions
COLOR_NOISE_STD    = 0.01        # small per-point color noise
PHOTOMETRIC_MISMATCH = True      # give B a gain/bias so raw-intensity loss is wrong
B_GAIN             = 1.12        # simulates different auto-exposure between two Spots
B_BIAS             = -0.04

OUT_DIR            = os.path.join(os.path.dirname(__file__), "data")


# ------------------------------------------------------------------------- texture/geom
def texture(u, v):
    """Rich color as a function of 2D surface coords. Multi-frequency (smooth gradients
    for a wide basin) + a checker overlay (distinctive high-freq features to lock on)."""
    r = 0.5 + 0.5 * np.sin(2 * np.pi * u / 1.3)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * v / 1.1 + 1.0)
    b = 0.5 + 0.5 * np.sin(2 * np.pi * (u + v) / 0.7)
    checker = ((np.floor(u / 0.4) + np.floor(v / 0.4)) % 2).astype(np.float64)
    rgb = 0.7 * np.stack([r, g, b], axis=-1) + 0.3 * checker[..., None]
    return np.clip(rgb, 0.0, 1.0)


def euler_to_R(rx, ry, rz):
    """Rotation from XYZ Euler angles (radians): R = Rz @ Ry @ Rx."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def sample_cloud(x_min, x_max, n_points, rng):
    """Sample n_points over the wall+floor corner within [x_min, x_max], in world/A frame."""
    n_wall = int(n_points * WALL_FRACTION)
    n_floor = n_points - n_wall

    # Wall: plane at y=0, spanned by (x, z). Surface coords u=x, v=z.
    xw = rng.uniform(x_min, x_max, n_wall)
    zw = rng.uniform(*WALL_HEIGHT, n_wall)
    wall_xyz = np.stack([xw, np.zeros_like(xw), zw], axis=-1)
    wall_rgb = texture(xw, zw)

    # Floor: plane at z=0, spanned by (x, y). Surface coords u=x, v=y (offset so the
    # floor texture differs from the wall texture).
    xf = rng.uniform(x_min, x_max, n_floor)
    yf = rng.uniform(*FLOOR_DEPTH, n_floor)
    floor_xyz = np.stack([xf, yf, np.zeros_like(xf)], axis=-1)
    floor_rgb = texture(xf + 10.0, yf + 5.0)

    xyz = np.concatenate([wall_xyz, floor_xyz], axis=0)
    rgb = np.concatenate([wall_rgb, floor_rgb], axis=0)

    xyz = xyz + rng.normal(0.0, POS_NOISE_STD, xyz.shape)
    rgb = np.clip(rgb + rng.normal(0.0, COLOR_NOISE_STD, rgb.shape), 0.0, 1.0)
    return xyz.astype(np.float32), rgb.astype(np.float32)


# ----------------------------------------------------------------------------- ply i/o
def write_ply(path, xyz, rgb):
    rgb255 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb255):
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")


# --------------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Cloud A: reference, stored in world/A frame. Independent sampling & seed.
    A_xyz, A_rgb = sample_cloud(X_RANGE[0], A_XMAX, N_POINTS, rng)

    # Cloud B: same scene, its own independent samples over the overlapping region,
    # expressed (world frame) then pushed into B's own frame.
    B_world_xyz, B_rgb = sample_cloud(B_XMIN, X_RANGE[1], N_POINTS, rng)

    if PHOTOMETRIC_MISMATCH:
        B_rgb = np.clip(B_rgb * B_GAIN + B_BIAS, 0.0, 1.0).astype(np.float32)

    R_gt = euler_to_R(*np.deg2rad(GT_EULER_DEG))
    t_gt = np.array(GT_TRANSLATION, dtype=np.float64)
    T_gt = np.eye(4)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3] = t_gt

    # world (A frame) -> B frame:  p_B = R_gt^T (p_A - t_gt).  So T_gt maps B -> A.
    B_xyz = ((B_world_xyz.astype(np.float64) - t_gt) @ R_gt).astype(np.float32)

    np.savez(os.path.join(OUT_DIR, "cloud_A.npz"), xyz=A_xyz, rgb=A_rgb)
    np.savez(
        os.path.join(OUT_DIR, "cloud_B.npz"),
        xyz=B_xyz, rgb=B_rgb,
        R_gt=R_gt.astype(np.float32), t_gt=t_gt.astype(np.float32), T_gt=T_gt.astype(np.float32),
    )
    write_ply(os.path.join(OUT_DIR, "cloud_A.ply"), A_xyz, A_rgb)
    write_ply(os.path.join(OUT_DIR, "cloud_B.ply"), B_xyz, B_rgb)

    print(f"Wrote {len(A_xyz)}-pt cloud A and {len(B_xyz)}-pt cloud B to {OUT_DIR}/")
    print("Ground-truth T_gt (B -> A):")
    print(T_gt)
    print("\nSanity: applying T_gt to B should land back on the scene / cloud A.")
    print("Your tracker's job: recover this T_gt from a perturbed initial guess.")


if __name__ == "__main__":
    main()
