"""Non-learned in-betweening baselines from the LAFAN1 benchmark."""
from __future__ import annotations

import torch

from . import quat as Q


def zero_velocity(root: torch.Tensor, lq: torch.Tensor, n_past: int, n_trans: int):
    """Repeat the last context frame. root (B,T,3), lq (B,T,J,4) -> transition frames."""
    r = root[:, n_past - 1 : n_past].expand(-1, n_trans, -1)
    q = lq[:, n_past - 1 : n_past].expand(-1, n_trans, -1, -1)
    return r, q


def interpolation(root: torch.Tensor, lq: torch.Tensor, n_past: int, n_trans: int):
    """LERP the root position and SLERP every local quaternion between the last context frame
    and the target frame (index n_past + n_trans)."""
    a, b = n_past - 1, n_past + n_trans
    w = torch.linspace(0, 1, n_trans + 2, dtype=root.dtype, device=root.device)[1:-1]
    r = root[:, a : a + 1] + w[None, :, None] * (root[:, b : b + 1] - root[:, a : a + 1])
    qa = Q.qnormalize(lq[:, a : a + 1]).expand(-1, n_trans, -1, -1)
    qb = Q.qnormalize(lq[:, b : b + 1]).expand(-1, n_trans, -1, -1)
    q = Q.slerp(qa, qb, w[None, :, None].expand(qa.shape[:-1]))
    return r, q
