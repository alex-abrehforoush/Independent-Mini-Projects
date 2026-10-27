"""Rollouts and the GRPO / DPO objectives, kept apart from the training loops for testing.

GRPO (Shao et al. 2024): for a prompt, G completions are sampled and scored;
each completion's advantage is its reward relative to the group. The
policy-gradient loss uses the PPO-clipped importance ratio per token, averaged
over each completion's tokens and then over completions, with an optional k3
KL penalty to a reference policy.

DPO (Rafailov et al. 2023): on (chosen, rejected) pairs, -log sigmoid of beta
times the difference of policy-vs-reference log-ratios of the two sequences.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from sensor_vlm import rewards
from sensor_vlm.model import generate


def rollout(model, processor, samples: list[dict], tasks_dir: Path, group: int, temperature: float = 1.0,
            max_new_tokens: int = 64) -> tuple[list[list[str]], list[list[dict]]]:
    """`group` sampled completions per sample and their reward dicts, grouped by sample."""
    was_training = model.training
    model.eval()
    comps = generate(model, processor, samples, tasks_dir, max_new_tokens=max_new_tokens, do_sample=True,
                     temperature=temperature, num_return_sequences=group)
    model.train(was_training)
    groups = [comps[i * group:(i + 1) * group] for i in range(len(samples))]
    return groups, [[rewards.score(c, s) for c in g] for g, s in zip(groups, samples)]


def group_advantages(scores: torch.Tensor, scale: str = "std", eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """scores (P, G) -> advantages (P, G) and a (P,) mask of groups with any reward spread.

    scale="std" is the original GRPO normalization; scale="none" only centers
    the rewards (Dr. GRPO), which avoids up-weighting near-uniform groups.
    """
    centered = scores - scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False)
    informative = std.squeeze(1) > eps
    adv = centered / (std + eps) if scale == "std" else centered
    return adv * informative.unsqueeze(1), informative


def policy_loss(logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
                clip: float = 0.2, ref_logp: torch.Tensor | None = None, beta: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Clipped surrogate for a micro-batch.

    logp, old_logp, ref_logp, mask: (B, T) per completion token; adv: (B,).
    Returns the loss summed over the micro-batch's completions (callers divide
    by the total number of completions) and diagnostics.
    """
    ratio = torch.exp(logp - old_logp)
    a = adv.unsqueeze(1)
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    per_token = -surrogate
    stats = {}
    if ref_logp is not None and beta > 0:
        d = ref_logp - logp
        kl = torch.exp(d) - d - 1  # k3 estimator, >= 0
        per_token = per_token + beta * kl
        stats["kl"] = ((kl * mask).sum() / mask.sum()).item()
    lengths = mask.sum(1).clamp(min=1)
    loss = ((per_token * mask).sum(1) / lengths).sum()
    clipped = ((ratio - 1).abs() > clip).float()
    stats["clip_frac"] = ((clipped * mask).sum() / mask.sum()).item()
    return loss, stats


def dpo_loss(pol_chosen: torch.Tensor, pol_rejected: torch.Tensor, ref_chosen: torch.Tensor,
             ref_rejected: torch.Tensor, beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """Sequence log-probs (B,) -> mean DPO loss and diagnostics (implicit reward margin, pair accuracy)."""
    margin = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, {"margin": (beta * margin).mean().item(), "accuracy": (margin > 0).float().mean().item()}
