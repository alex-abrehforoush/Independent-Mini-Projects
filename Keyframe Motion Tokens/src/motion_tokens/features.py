"""Motion representations shared by every tokenizer and generator.

A clip is (root positions (…,T,3), local quaternions (…,T,J,4)) in a window frame.

* canonicalize(): put an anchor frame at the xz origin, facing +x.
* relative_form(): for an increasing set of key frames k_0 < k_1 < …, describe key i relative to
  key i-1 by  [root height, Δxz in the heading frame of key i-1, Δyaw, root rotation relative
  to its own heading (6D), 21 local joint rotations (6D)] = 136 numbers.  With every frame as a
  key this is the usual velocity-style (HumanML3D-like) dense representation; with sparse keys
  it is a keyframe token's continuous content.  The two only differ in which frames are kept.
* absolute_from_relative(): integrate it back (differentiable).
"""
from __future__ import annotations

import math

import torch

from . import quat as Q

D_REL = 1 + 2 + 1 + 6 + 21 * 6  # 136
ROT_SLICE = slice(4, D_REL)  # root-relative + joint 6D rotations


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def heading_yaw(root_q: torch.Tensor) -> torch.Tensor:
    return Q.yaw_angle(Q.heading_quat(root_q))


def canonicalize(root: torch.Tensor, lq: torch.Tensor, anchor: int | torch.Tensor = 0):
    """Translate/rotate about y so frame `anchor` has root xz = 0 and faces +x."""
    B = root.shape[0]
    idx = torch.as_tensor(anchor, device=root.device).expand(B) if not torch.is_tensor(anchor) or anchor.dim() == 0 else anchor
    ar = root[torch.arange(B), idx]  # (B,3)
    aq = lq[torch.arange(B), idx, 0]  # (B,4)
    inv = Q.qinv(Q.heading_quat(aq))[:, None]  # (B,1,4)
    shift = ar * ar.new_tensor([1.0, 0.0, 1.0])
    root = Q.qrot(inv, root - shift[:, None])
    lq = lq.clone()
    lq[..., 0, :] = Q.qmul(inv, lq[..., 0, :])
    return root, lq


def relative_form(root: torch.Tensor, lq: torch.Tensor, keys: torch.Tensor | None = None) -> torch.Tensor:
    """root (B,T,3), lq (B,T,J,4); keys (B,K) long frame indices (default: all frames).
    Returns (B,K,136). The first key's Δ is measured from the window origin (heading +x)."""
    if keys is not None:
        b = torch.arange(root.shape[0], device=root.device)[:, None]
        root, lq = root[b, keys], lq[b, keys]
    hq = Q.heading_quat(lq[..., 0, :])  # (B,K,4)
    yaw = Q.yaw_angle(hq)
    prev_pos = torch.cat([torch.zeros_like(root[:, :1]), root[:, :-1]], 1)
    prev_yaw = torch.cat([torch.zeros_like(yaw[:, :1]), yaw[:, :-1]], 1)
    d = root - prev_pos
    d = Q.qrot(Q.qinv(Q.yaw_to_quat(prev_yaw)), d)
    rel_root = Q.qmul(Q.qinv(hq), lq[..., 0, :])
    rots = Q.quat_to_rot6d(torch.cat([rel_root[..., None, :], lq[..., 1:, :]], -2))
    return torch.cat([root[..., 1:2], d[..., [0, 2]], wrap(yaw - prev_yaw)[..., None], rots.flatten(-2)], -1)


def trajectory(root: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Root x, z and unwrapped heading yaw (B,K,3) in the window frame; f = relative_form(root, ...)."""
    return torch.stack([root[..., 0], root[..., 2], torch.cumsum(f[..., 3], 1)], -1)


def absolute_from_relative(f: torch.Tensor, n_joints: int = 22, traj: torch.Tensor | None = None):
    """Inverse of relative_form: (B,K,136) -> root (B,K,3), lq (B,K,J,4). If traj (B,K,3) is
    given (see trajectory), it replaces the trajectory integrated from the Δ channels."""
    if traj is None:
        dyaw = f[..., 3]
        yaw = torch.cumsum(dyaw, 1)
        prev_yaw = yaw - dyaw
        d = torch.stack([f[..., 1], torch.zeros_like(f[..., 1]), f[..., 2]], -1)
        d = Q.qrot(Q.yaw_to_quat(prev_yaw), d)
        xz = torch.cumsum(d, 1)
        x, z = xz[..., 0], xz[..., 2]
    else:
        x, z, yaw = traj.unbind(-1)
    root = torch.stack([x, f[..., 0], z], -1)
    rots = Q.rot6d_to_quat(f[..., ROT_SLICE].unflatten(-1, (n_joints, 6)))
    lq = rots.clone()
    lq[..., 0, :] = Q.qmul(Q.yaw_to_quat(yaw), rots[..., 0, :])
    return root, lq


def interpolate_keys(root: torch.Tensor, lq: torch.Tensor, keys: torch.Tensor, T: int):
    """Dense LERP (root) / SLERP (local rotations) through key frames.

    root (B,K,3), lq (B,K,J,4) are the key poses at frame indices keys (B,K), increasing, with
    keys[:,0] == 0 and keys[:,-1] == T-1. Returns dense (B,T,3), (B,T,J,4)."""
    B, K = keys.shape
    t = torch.arange(T, device=keys.device).expand(B, T)
    nxt = torch.searchsorted(keys.contiguous(), t.contiguous(), right=False).clamp(1, K - 1)
    prv = nxt - 1
    ka = torch.gather(keys, 1, prv)
    kb = torch.gather(keys, 1, nxt)
    w = ((t - ka) / (kb - ka).clamp(min=1)).to(root.dtype)
    b = torch.arange(B, device=keys.device)[:, None]
    r = root[b, prv] + w[..., None] * (root[b, nxt] - root[b, prv])
    qa, qb = lq[b, prv], lq[b, nxt]
    q = Q.slerp(qa, qb, w[..., None].expand(qa.shape[:-1]))
    return r, q
