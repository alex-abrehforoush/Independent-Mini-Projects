"""Quaternion and rotation utilities (torch, w-x-y-z convention, Hamilton product).

Everything works on arbitrary leading batch dimensions and on float32 or float64.
"""
from __future__ import annotations

import torch

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def qinv(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def qnormalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=eps)


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (...,3) by unit quaternions q (...,4)."""
    qv = q[..., 1:]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(qv, t, dim=-1)


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = angle[..., None] / 2.0
    return torch.cat([torch.cos(half), torch.sin(half) * axis], -1)


def euler_to_quat(e: torch.Tensor, order: str) -> torch.Tensor:
    """e (...,3) radians, applied as q = q[order[0]] * q[order[1]] * q[order[2]]."""
    qs = [axis_angle_to_quat(e.new_tensor(_AXES[c]), e[..., i]) for i, c in enumerate(order)]
    return qmul(qs[0], qmul(qs[1], qs[2]))


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = qnormalize(q).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Numerically stable matrix -> quaternion (picks the largest diagonal branch)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    cands = torch.stack(
        [
            torch.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1),
            torch.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1),
        ],
        -2,
    )
    diag = torch.stack([m00 + m11 + m22, m00, m11, m22], -1)
    best = diag.argmax(-1)
    q = torch.gather(cands, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = qnormalize(q)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0).to(q)


def mat_to_euler(m: torch.Tensor, order: str) -> torch.Tensor:
    """Inverse of euler_to_quat for the orders used by BVH files (returns radians, channel order)."""
    if order == "zyx":  # R = Rz(a) Ry(b) Rx(c)
        b = torch.asin((-m[..., 2, 0]).clamp(-1, 1))
        a = torch.atan2(m[..., 1, 0], m[..., 0, 0])
        c = torch.atan2(m[..., 2, 1], m[..., 2, 2])
        return torch.stack([a, b, c], -1)
    if order == "xyz":  # R = Rx(a) Ry(b) Rz(c)
        b = torch.asin(m[..., 0, 2].clamp(-1, 1))
        a = torch.atan2(-m[..., 1, 2], m[..., 2, 2])
        c = torch.atan2(-m[..., 0, 1], m[..., 0, 0])
        return torch.stack([a, b, c], -1)
    raise NotImplementedError(order)


def rot6d_to_mat(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 2019 continuous 6D representation (first two matrix columns)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], -1)


def mat_to_rot6d(m: torch.Tensor) -> torch.Tensor:
    return torch.cat([m[..., :, 0], m[..., :, 1]], -1)


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    return mat_to_rot6d(quat_to_mat(q))


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
    return mat_to_quat(rot6d_to_mat(d6))


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation; t broadcasts against a[..., 0]. Gradient-safe: near-parallel
    inputs take the linear branch, and the other branch never sees acos(1) (whose infinite
    derivative would otherwise leak NaNs through torch.where)."""
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(max=1.0)
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device)
    if t.dim() < a.dim():
        t = t[..., None]
    linear = (1.0 - dot) < 0.01
    omega = torch.acos(torch.where(linear, torch.zeros_like(dot), dot))
    so = torch.sin(omega)
    w0 = torch.where(linear, 1.0 - t, torch.sin((1.0 - t) * omega) / so)
    w1 = torch.where(linear, t, torch.sin(t * omega) / so)
    return qnormalize(w0 * a + w1 * b)


def remove_discontinuities(q: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Flip signs along a time dimension so consecutive quaternions lie in the same hemisphere."""
    q = q.movedim(dim, 0)
    dots = (q[1:] * q[:-1]).sum(-1)
    flips = torch.where(dots < 0, -1.0, 1.0).to(q)
    sign = torch.cat([torch.ones_like(flips[:1]), torch.cumprod(flips, 0)], 0)
    return (q * sign[..., None]).movedim(0, dim)


def quat_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest-arc rotation taking direction a to direction b."""
    w = torch.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + (a * b).sum(-1)
    return qnormalize(torch.cat([w[..., None], torch.linalg.cross(a, b, dim=-1)], -1))


def _levels(parents) -> list[list[int]]:
    depth = [0] * len(parents)
    for j in range(1, len(parents)):
        depth[j] = depth[parents[j]] + 1
    return [[j for j in range(len(parents)) if depth[j] == k] for k in range(1, max(depth) + 1)]


_LEVEL_CACHE: dict = {}


def fk(lq: torch.Tensor, lpos: torch.Tensor, parents) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics. lq (...,J,4) local rotations, lpos (...,J,3) with root position at
    joint 0 and bone offsets elsewhere. Returns global rotations and positions.
    Joints at the same tree depth are processed together (7 steps for LAFAN1, not 21)."""
    key = tuple(parents)
    if key not in _LEVEL_CACHE:
        _LEVEL_CACHE[key] = [(torch.tensor(l), torch.tensor([parents[j] for j in l])) for l in _levels(parents)]
    J = len(parents)
    gq = torch.zeros_like(lq)
    gp = torch.zeros_like(lpos)
    gq[..., 0, :] = lq[..., 0, :]
    gp[..., 0, :] = lpos[..., 0, :]
    for js, ps in _LEVEL_CACHE[key]:
        js, ps = js.to(lq.device), ps.to(lq.device)
        pq = gq.index_select(-2, ps)
        gp = gp.index_copy(-2, js, qrot(pq, lpos.index_select(-2, js)) + gp.index_select(-2, ps))
        gq = gq.index_copy(-2, js, qmul(pq, lq.index_select(-2, js)))
    return gq, gp


def ik(gq: torch.Tensor, gp: torch.Tensor, parents) -> tuple[torch.Tensor, torch.Tensor]:
    par = list(parents[1:])
    lq = torch.cat([gq[..., :1, :], qmul(qinv(gq[..., par, :]), gq[..., 1:, :])], -2)
    lp = torch.cat([gp[..., :1, :], qrot(qinv(gq[..., par, :]), gp[..., 1:, :] - gp[..., par, :])], -2)
    return lq, lp


def heading_quat(root_q: torch.Tensor, local_forward=(0.0, 1.0, 0.0)) -> torch.Tensor:
    """Yaw-only rotation (about +y) taking +x to the root's forward direction projected on xz.

    LAFAN1's hips joint points its local +y axis forward, matching the official rotate_at_frame."""
    fwd = qrot(root_q, root_q.new_tensor(local_forward).expand(*root_q.shape[:-1], 3))
    fwd = fwd * fwd.new_tensor([1.0, 0.0, 1.0])
    fwd = torch.nn.functional.normalize(fwd, dim=-1)
    x = torch.zeros_like(fwd)
    x[..., 0] = 1.0
    return quat_between(x, fwd)


def yaw_angle(hq: torch.Tensor) -> torch.Tensor:
    """Angle about +y of a yaw-only quaternion."""
    return 2.0 * torch.atan2(hq[..., 2], hq[..., 0])


def yaw_to_quat(angle: torch.Tensor) -> torch.Tensor:
    return axis_angle_to_quat(angle.new_tensor([0.0, 1.0, 0.0]), angle)
