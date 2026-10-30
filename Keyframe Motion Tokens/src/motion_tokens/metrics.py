"""Evaluation metrics: the LAFAN1 benchmark trio (L2Q, L2P, NPSS) plus physical-plausibility
diagnostics (foot skate, ground penetration, jitter) that the benchmark trio does not see."""
from __future__ import annotations

import numpy as np
import torch

from . import quat as Q
from .lafan import FEET, local_positions


def l2q(pred_gq: torch.Tensor, gt_gq: torch.Tensor) -> torch.Tensor:
    """Per-window mean over frames of the L2 norm over all global quaternion components."""
    return (pred_gq - gt_gq).pow(2).sum((-1, -2)).sqrt().mean(-1)


def continuous_from(ref_lq: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """Choose quaternion signs so (ref, lq[0], lq[1], ...) is sign-continuous in time.

    q and -q are the same rotation, but L2Q and NPSS compare raw components. The ground truth
    is sign-continuous from the last context frame, so a prediction made continuous from that
    same (known) frame is compared on equal terms. Uses no ground-truth transition frames, and
    is a no-op for the official SLERP / zero-velocity baselines."""
    seq = torch.cat([ref_lq[:, None], lq], 1)
    return Q.remove_discontinuities(seq, dim=1)[:, 1:]


def l2p(pred_gp: torch.Tensor, gt_gp: torch.Tensor, stats: dict) -> torch.Tensor:
    B, T = gt_gp.shape[:2]
    p = (pred_gp.reshape(B, T, -1) - stats["x_mean"]) / stats["x_std"]
    g = (gt_gp.reshape(B, T, -1) - stats["x_mean"]) / stats["x_std"]
    return (p - g).pow(2).sum(-1).sqrt().mean(-1)


def npss(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Normalized Power Spectrum Similarity (Gopalakrishnan et al. 2019), as in the LAFAN1 code.
    gt/pred: (B, T, D). Returns the power-weighted average over sequences and dimensions."""
    gt, pred = gt.double().cpu().numpy(), pred.double().cpu().numpy()
    gp = np.square(np.real(np.fft.fft(gt, axis=1)))
    pp = np.square(np.real(np.fft.fft(pred, axis=1)))
    gt_tot, pr_tot = gp.sum(1), pp.sum(1)
    cdf_g = np.cumsum(gp / gt_tot[:, None], 1)
    cdf_p = np.cumsum(pp / pr_tot[:, None], 1)
    emd = np.linalg.norm(cdf_p - cdf_g, ord=1, axis=1)
    return float(np.average(emd, weights=gt_tot))


def benchmark_scores(pred_lq, pred_root, gt_lq, gt_root, offsets, parents, stats,
                     ref_lq: torch.Tensor | None = None) -> dict:
    """Scores on transition frames only: (B, n_trans, J, 4) / (B, n_trans, 3) tensors.
    ref_lq (B, J, 4): last context frame, used only to fix quaternion signs (see continuous_from)."""
    pred_lq = pred_lq.to(gt_lq)
    if ref_lq is not None:
        pred_lq = continuous_from(ref_lq.to(gt_lq), pred_lq)
    off = offsets.to(gt_root)
    gt_gq, gt_gp = Q.fk(gt_lq, local_positions(gt_root, off), parents)
    pr_gq, pr_gp = Q.fk(pred_lq, local_positions(pred_root.to(gt_root), off), parents)
    B, T = gt_lq.shape[:2]
    return {
        "L2Q": l2q(pr_gq, gt_gq).mean().item(),
        "L2P": l2p(pr_gp, gt_gp, stats).mean().item(),
        "NPSS": npss(gt_gq.reshape(B, T, -1), pr_gq.reshape(B, T, -1)),
    }


# ---------------------------------------------------------------- physical plausibility

def foot_contacts(gp: torch.Tensor, vel_thresh: float = 1.0, height_margin: float = 6.0) -> torch.Tensor:
    """Ground-truth-style contact labels for (…,T,J,3) global positions in cm at 30 fps.

    A foot joint is in contact when it moves less than `vel_thresh` cm/frame and is within
    `height_margin` cm of the lowest point that joint reaches in the clip (handles the
    uneven-terrain themes better than an absolute floor height)."""
    f = gp[..., list(FEET), :]
    v = torch.zeros_like(f[..., 0])
    v[..., 1:, :] = (f[..., 1:, :, :] - f[..., :-1, :, :]).norm(dim=-1)
    v[..., 0, :] = v[..., 1, :]
    h = f[..., 1]
    low = h <= h.amin(-2, keepdim=True) + height_margin
    return (v < vel_thresh) & low


def foot_skate(pred_gp: torch.Tensor, contacts: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """Mean horizontal foot speed (cm/s) on frames labelled as contact; per window."""
    f = pred_gp[..., list(FEET), :]
    hv = (f[..., 1:, :, :] - f[..., :-1, :, :])[..., [0, 2]].norm(dim=-1) * fps
    c = contacts[..., 1:, :] & contacts[..., :-1, :]
    num = (hv * c).sum((-1, -2))
    den = c.sum((-1, -2)).clamp(min=1)
    return num / den


def foot_skate_own(pred_gp: torch.Tensor, low: torch.Tensor, height_margin: float = 1.5, fps: float = 30.0) -> torch.Tensor:
    """Foot skate on the motion's *own* contacts, for generated motion: mean horizontal foot speed
    (cm/s) on frames where the foot is within `height_margin` cm of low (B, 4), each foot joint's
    lowest ground-truth height in the clip (the floor). Unlike foot_skate with ground-truth
    contacts, a step taken at a different time than in the ground truth is not counted as skating.
    1.5 cm recovers 87% of the ground-truth contact labels at 90% precision on the test windows
    (ground truth scores 8.5 cm/s); at 6 cm, swinging feet count and ground truth scores 54."""
    contacts = pred_gp[..., list(FEET), 1] <= low[..., None, :] + height_margin
    return foot_skate(pred_gp, contacts, fps)


def penetration(pred_gp: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    """Mean depth (cm) of toe joints below the clip's ground-truth floor; per window."""
    toes = pred_gp[..., [FEET[1], FEET[3]], 1]
    return (floor[..., None, None] - toes).clamp(min=0).mean((-1, -2))


def jitter(gp: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """Mean joint jerk magnitude (m/s^3) over all joints; per window."""
    j = gp[..., 3:, :, :] - 3 * gp[..., 2:-1, :, :] + 3 * gp[..., 1:-2, :, :] - gp[..., :-3, :, :]
    return j.norm(dim=-1).mean((-1, -2)) * fps**3 / 100.0
