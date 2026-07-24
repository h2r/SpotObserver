"""
Milestone 1, Step 5 — se(3) pose parameterization.

The pose is a 6-vector twist xi = [rho(3 translation-ish), phi(3 rotation)] living in
the Lie algebra se(3). The exp map turns it into a 4x4 rigid transform. Optimizing in
the tangent space (6 unconstrained numbers) instead of directly on a 4x4 matrix keeps
the transform a valid rotation+translation at every step and gives clean gradients.

We parameterize the estimate as  T = T_init @ exp(xi)  with xi starting at 0, so the
optimizer moves a *delta* away from the initial guess. That is exactly the warm-start
structure the real tracker uses (follow a small motion from the previous pose).

The small-angle path is made autograd-safe by adding eps under the sqrt, so gradients
stay finite as the rotation angle -> 0.
"""

import torch


def skew(v):
    """(...,3) -> (...,3,3) skew-symmetric matrix (so that skew(v) @ x == cross(v, x))."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


def se3_exp(xi, eps=1e-8):
    """xi: (6,) = [rho(3), phi(3)]  ->  (4,4) rigid transform. Differentiable in xi."""
    rho, phi = xi[:3], xi[3:]
    device, dtype = xi.device, xi.dtype
    I = torch.eye(3, device=device, dtype=dtype)

    theta2 = (phi * phi).sum()
    theta = torch.sqrt(theta2 + eps)                 # eps keeps d/dphi finite at phi=0
    Phi = skew(phi)
    Phi2 = Phi @ Phi

    A = torch.sin(theta) / theta                     # -> 1   as theta->0
    B = (1 - torch.cos(theta)) / (theta2 + eps)      # -> 1/2
    C = (1 - A) / (theta2 + eps)                     # -> 1/6

    R = I + A * Phi + B * Phi2                        # Rodrigues
    V = I + B * Phi + C * Phi2                        # left Jacobian
    t = V @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose_error(T_est, T_gt):
    """Return (rotation error in degrees, translation error in metres) between two 4x4."""
    T_err = torch.linalg.inv(T_gt) @ T_est
    R = T_err[:3, :3]
    cos = ((torch.trace(R) - 1) * 0.5).clamp(-1.0, 1.0)
    rot_deg = torch.rad2deg(torch.arccos(cos))
    trans = torch.linalg.norm(T_err[:3, 3])
    return rot_deg.item(), trans.item()


if __name__ == "__main__":
    import numpy as np

    # exp(0) == I
    assert torch.allclose(se3_exp(torch.zeros(6)), torch.eye(4), atol=1e-6)

    # pure rotation about z by 30 deg, no translation
    xi = torch.tensor([0., 0., 0., 0., 0., np.deg2rad(30.)], dtype=torch.float32)
    T = se3_exp(xi)
    c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
    Rz = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.]], dtype=torch.float32)
    assert torch.allclose(T[:3, :3], Rz, atol=1e-5), T[:3, :3]
    assert torch.allclose(T[:3, 3], torch.zeros(3), atol=1e-6)

    # gradient flows to xi
    xi = torch.zeros(6, requires_grad=True)
    se3_exp(xi).sum().backward()
    assert xi.grad is not None and torch.isfinite(xi.grad).all()

    # pose_error sanity: identity vs 30deg-z rotation
    rot, tr = pose_error(se3_exp(torch.zeros(6)), T)
    print(f"self-tests passed. error(I, Rz30) = {rot:.2f} deg, {tr:.3f} m  (expect ~30 deg, 0 m)")
