"""
Milestone 1, Step 3 — the virtual camera.

A pinhole camera you invent and fix in the scene. You render BOTH clouds from this
one viewpoint and compare the images. It uses the OpenCV convention (x right, y
down, z forward) because that is what gsplat's rasterization expects, so the same
K and view matrix drop straight into step 4.

`look_at` builds the camera from an eye/target; `place_overlap` auto-places it in
front of the overlap region so you don't have to hand-tune it. `project` pushes
world points to pixels — no rasterizer, so it runs anywhere and lets us preview
"what the camera sees" to validate placement before we ever call gsplat.

Pure torch — CPU/MPS, no CUDA needed.
"""

import numpy as np
import torch


def _normalize(v):
    return v / torch.linalg.norm(v)


class VirtualCamera:
    def __init__(self, K, viewmat, width, height):
        self.K = K                      # (3,3) intrinsics
        self.viewmat = viewmat          # (4,4) world -> camera (OpenCV)
        self.width = int(width)
        self.height = int(height)

    @property
    def device(self):
        return self.K.device

    # ----------------------------------------------------------------- constructors
    @classmethod
    def look_at(cls, eye, target, up=(0.0, 0.0, 1.0), fov_deg=60.0,
                width=640, height=480, device="cpu", dtype=torch.float32):
        eye = torch.as_tensor(eye, device=device, dtype=dtype)
        target = torch.as_tensor(target, device=device, dtype=dtype)
        up = torch.as_tensor(up, device=device, dtype=dtype)

        z = _normalize(target - eye)              # forward (+z, OpenCV)
        x = _normalize(torch.linalg.cross(z, up)) # right  (+x)
        y = torch.linalg.cross(z, x)              # down   (+y)

        R = torch.stack([x, y, z], dim=0)         # world -> cam (rows are cam axes)
        t = -R @ eye
        viewmat = torch.eye(4, device=device, dtype=dtype)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t

        f = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
        K = torch.tensor([[f, 0, width / 2],
                          [0, f, height / 2],
                          [0, 0, 1]], device=device, dtype=dtype)
        return cls(K, viewmat, width, height)

    @classmethod
    def place_overlap(cls, means_a, means_b, viewmat=None, margin=1.6,
                      fov_deg=60.0, width=640, height=480, device="cpu"):
        """Auto-place: aim at the centroid of the region A and B share, backing the
        camera off along +y (the open side of the corner) far enough to frame it."""
        a = torch.as_tensor(np.asarray(means_a), dtype=torch.float32)
        b = torch.as_tensor(np.asarray(means_b), dtype=torch.float32)
        lo = torch.maximum(a.min(0).values, b.min(0).values)
        hi = torch.minimum(a.max(0).values, b.max(0).values)
        center = 0.5 * (lo + hi)
        extent = (hi - lo).max().item()
        eye = center.clone()
        eye[1] = hi[1] + margin * extent          # back off along +y
        eye[2] = center[2] + 0.3 * extent         # lift slightly
        return cls.look_at(eye.tolist(), center.tolist(), up=(0, 0, 1),
                           fov_deg=fov_deg, width=width, height=height, device=device)

    def scaled(self, factor):
        """A lower-resolution copy of this camera (same viewpoint) for coarse-to-fine."""
        w = max(1, int(round(self.width * factor)))
        h = max(1, int(round(self.height * factor)))
        sx, sy = w / self.width, h / self.height
        K = self.K.clone()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        return VirtualCamera(K, self.viewmat, w, h)

    # --------------------------------------------------------------------- projection
    def project(self, points):
        """world points (N,3) -> (u, v, depth, valid_mask). Differentiable in points."""
        pts = torch.as_tensor(points, device=self.device, dtype=self.K.dtype)
        R = self.viewmat[:3, :3]
        t = self.viewmat[:3, 3]
        pc = pts @ R.T + t                         # camera coords
        z = pc[:, 2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        zc = z.clamp_min(1e-6)
        u = fx * pc[:, 0] / zc + cx
        v = fy * pc[:, 1] / zc + cy
        valid = (z > 1e-6) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, z, valid


if __name__ == "__main__":
    import os
    from gaussians import load_cloud_gaussians
    data = os.path.join(os.path.dirname(__file__), "data")
    ga = load_cloud_gaussians(os.path.join(data, "cloud_A.npz"))
    gb = load_cloud_gaussians(os.path.join(data, "cloud_B.npz"))
    cam = VirtualCamera.place_overlap(ga["means"], gb["means"])
    u, v, z, valid = cam.project(ga["means"])
    print("K =\n", cam.K.numpy())
    print("viewmat =\n", cam.viewmat.numpy())
    print(f"cloud A: {valid.float().mean()*100:.1f}% of points in frame, "
          f"depth {z[valid].min():.2f}-{z[valid].max():.2f} m")
