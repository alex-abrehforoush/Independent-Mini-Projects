"""Choosing key frames.

* uniform: every `stride` frames (the fixed-rate control).
* adaptive: parallel Douglas-Peucker-style top-down simplification, the classical animation
  "keyframe reduction" recipe: keep splitting every segment at its worst-interpolated frame
  until the SLERP/LERP reconstruction is within `eps` cm everywhere (and no gap exceeds
  `max_gap`, so Δt fits a finite vocabulary).
Masks are (B,T) bools; the first and last frame are always keys.
"""
from __future__ import annotations

import torch

from . import quat as Q
from .lafan import local_positions


def neighbours(mask: torch.Tensor):
    """For every frame, the index of the key at or before it and the key strictly after it
    (the key itself for the last frame)."""
    B, T = mask.shape
    t = torch.arange(T, device=mask.device).expand(B, T)
    prv = torch.where(mask, t, torch.full_like(t, -1)).cummax(1).values
    big = torch.full_like(t, T)
    nxt_incl = torch.where(mask, t, big).flip(1).cummin(1).values.flip(1)
    # strictly-after key: shift by one
    nxt = torch.cat([nxt_incl[:, 1:], torch.full_like(t[:, :1], T)], 1)
    nxt = torch.where(nxt >= T, prv, nxt)
    return prv, nxt


def interpolate_mask(root: torch.Tensor, lq: torch.Tensor, mask: torch.Tensor):
    """LERP/SLERP reconstruction of a dense clip from the frames where mask is True."""
    prv, nxt = neighbours(mask)
    t = torch.arange(mask.shape[1], device=mask.device).expand_as(prv)
    w = ((t - prv) / (nxt - prv).clamp(min=1)).to(root.dtype)
    w = torch.where(mask, torch.zeros_like(w), w)
    b = torch.arange(mask.shape[0], device=mask.device)[:, None]
    r = root[b, prv] + w[..., None] * (root[b, nxt] - root[b, prv])
    qa, qb = lq[b, prv], lq[b, nxt]
    return r, Q.slerp(qa, qb, w[..., None].expand(qa.shape[:-1]))


def frame_errors(root, lq, mask, offsets, parents) -> torch.Tensor:
    """Max-over-joints global position error (cm) of the interpolated clip, per frame."""
    r, q = interpolate_mask(root, lq, mask)
    off = offsets.to(root)
    _, gp = Q.fk(lq, local_positions(root, off), parents)
    _, ip = Q.fk(q, local_positions(r, off), parents)
    return (gp - ip).norm(dim=-1).amax(-1)


def uniform(B: int, T: int, stride: int, device=None) -> torch.Tensor:
    m = torch.zeros(B, T, dtype=torch.bool, device=device)
    m[:, ::stride] = True
    m[:, -1] = True
    return m


def adaptive(root, lq, offsets, parents, eps: float, max_gap: int = 32, max_iter: int = 64) -> torch.Tensor:
    B, T = root.shape[:2]
    mask = torch.zeros(B, T, dtype=torch.bool, device=root.device)
    mask[:, 0] = mask[:, -1] = True
    t = torch.arange(T, device=root.device).expand(B, T)
    for _ in range(max_iter):
        err = frame_errors(root, lq, mask, offsets, parents)
        prv, nxt = neighbours(mask)
        seg_len = nxt - prv
        score = torch.where(mask, torch.full_like(err, -1.0), err + 1e-6 * (T - t))  # tie-break: earliest
        seg_max = torch.full_like(err, -1.0).scatter_reduce(1, prv, score, "amax", include_self=True)
        seg_err = torch.gather(seg_max, 1, prv)
        split_err = (score == seg_err) & ~mask & (err > eps)
        # long but well-interpolated segments are split in the middle so Δt stays <= max_gap
        split_gap = (t == (prv + nxt) // 2) & (seg_len > max_gap) & (seg_err <= eps) & ~mask
        split = split_err | split_gap
        if not split.any():
            break
        mask = mask | split
    return mask
