"""Experiment 2: one autoregressive transformer backbone over either token family.

Task: given 10 context frames and a target frame `n` frames later, generate the transition.
Everything lives in the anchor frame (last context frame at the origin, facing +x).

Prefix (bidirectional): 10 context-frame tokens + 1 target token (+ gap embedding).
Sequence (causal):
  * "kf": keys k_1..k_m strictly inside the gap. Each step predicts the next key's Δt (masked so
          it never overshoots the target; landing on the target ends the sequence) and then, given
          Δt, its pose — FSQ codes (discrete) or the continuous key features via a flow head.
          Because keys are poses, each step also sees where the target is relative to the
          current key (free to compute from the decoded keys). With discrete codes it also sees
          each key's decoded features: a predictive code is a residual on the previous decoded
          key, so the code alone does not say what the pose is (codes-only: near-chance code loss).
  * "fr": the fixed-rate tokens of the anchor..target crop, one per 4 frames; codes or continuous
          latents via a flow head.
Heads: for FSQ, one small softmax per code dimension over its ordered levels (G x 4 per token;
a joint 1000-way softmax per group treats neighbouring levels as unrelated classes and barely
learned keyframe codes); G joint softmaxes for VQ codes; or a MAR-style per-token flow-matching MLP.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import features as F
from . import quat as Q
from .tokenizers import mlp, sinusoid

N_CTX = 10
MAX_GAP = 32


def pick(logits, temperature, gen=None):
    """Argmax at temperature 0, else a sample from softmax(logits / temperature)."""
    if temperature <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits / temperature, -1)
    return torch.multinomial(p.flatten(0, -2), 1, generator=gen).view(p.shape[:-1])


def target_features(root, lq, t_idx):
    """Target pose in the anchor frame: position (m), heading (sin, cos), rotations 6D. (B,137)"""
    b = torch.arange(root.shape[0], device=root.device)
    r, q = root[b, t_idx], lq[b, t_idx]
    hq = Q.heading_quat(q[:, 0])
    yaw = Q.yaw_angle(hq)
    rel = Q.qmul(Q.qinv(hq), q[:, 0])
    rots = Q.quat_to_rot6d(torch.cat([rel[:, None], q[:, 1:]], 1)).flatten(1)
    return torch.cat([r / 100.0, yaw.sin()[:, None], yaw.cos()[:, None], rots], -1)


def relative_target_state(key_root, key_lq, tgt_root, tgt_lq):
    """Where the target is from each key, in that key's heading frame: Δxz (m), Δyaw (sin, cos)."""
    hq = Q.heading_quat(key_lq[..., 0, :])
    d = Q.qrot(Q.qinv(hq), tgt_root[:, None] - key_root)
    dyaw = F.wrap(Q.yaw_angle(Q.heading_quat(tgt_lq[:, None, 0])) - Q.yaw_angle(hq))
    return torch.stack([d[..., 0] / 100.0, d[..., 2] / 100.0, dyaw.sin(), dyaw.cos()], -1)


class FlowHead(nn.Module):
    """Per-token conditional flow matching (rectified flow) on a small residual MLP."""

    def __init__(self, dim: int, cond: int, width: int = 512, depth: int = 3):
        super().__init__()
        self.dim = dim
        self.inp = nn.Linear(dim + 64, width)
        self.cond = nn.Linear(cond, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                                  nn.Linear(width, width)) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    def velocity(self, x, t, c):
        h = self.inp(torch.cat([x, sinusoid(t * 1000, 64)], -1)) + self.cond(c)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)

    def loss(self, x1, c, reps: int = 4):
        x1, c = x1.repeat(reps, 1), c.repeat(reps, 1)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        return (self.velocity(xt, t, c) - (x1 - x0)).pow(2).mean()

    @torch.no_grad()
    def sample(self, c, steps: int = 32, temperature: float = 1.0, gen=None):
        x = torch.randn(c.shape[0], self.dim, device=c.device, generator=gen) * temperature
        for i in range(steps):
            t = torch.full((c.shape[0],), i / steps, device=c.device)
            x = x + self.velocity(x, t, c) / steps
        return x


class Generator(nn.Module):
    def __init__(self, kind: str, continuous: bool, groups: int, codebook: int, cont_dim: int,
                 d: int = 256, layers: int = 6, heads: int = 4, levels=None):
        """levels: FSQ levels per group, e.g. (8, 5, 5, 5), for per-dimension code heads."""
        super().__init__()
        self.kind, self.continuous, self.groups, self.codebook = kind, continuous, groups, codebook
        self.levels = tuple(levels) if levels and not continuous else None
        self.d = d
        self.ctx_in = nn.Linear(F.D_REL, d)
        self.tgt_in = nn.Linear(137, d)
        self.type_emb = nn.Embedding(3, d)  # context, target, sequence
        self.gap_emb = nn.Embedding(256, d)
        self.bos = nn.Parameter(torch.zeros(d))
        if continuous:
            self.tok_in = nn.Linear(cont_dim, d)
            self.head = FlowHead(cont_dim, d)
        elif self.levels:
            lv = torch.tensor(self.levels)
            self.lmax, n_dim = int(lv.max()), groups * len(self.levels)
            self.register_buffer("lv", lv, persistent=False)
            self.register_buffer("basis", torch.cumprod(torch.cat([torch.ones(1, dtype=torch.long), lv[:-1]]), 0),
                                 persistent=False)
            self.code_emb = nn.Embedding(n_dim * self.lmax, d)
            self.code_head = nn.Linear(d, n_dim * self.lmax)
        else:
            self.code_emb = nn.Embedding(groups * codebook, d)
            self.code_head = nn.Linear(d, groups * codebook)
        if kind == "kf" and not continuous:
            self.key_in = nn.Linear(F.D_REL, d)
        if kind == "kf":
            self.dt_emb = nn.Embedding(MAX_GAP + 1, d)
            self.state_in = nn.Linear(4, d)
            self.dt_head = nn.Linear(d, MAX_GAP + 1)
            self.pose_cond = mlp(2 * d, d, d, 1)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    # ------------------------------------------------------------------ embeddings
    def prefix(self, ctx_feat, tgt_feat, gap):
        B = ctx_feat.shape[0]
        c = self.ctx_in(ctx_feat) + self.type_emb.weight[0] + sinusoid(torch.arange(-N_CTX + 1, 1, device=ctx_feat.device) + 512, self.d)
        t = self.tgt_in(tgt_feat) + self.type_emb.weight[1] + self.gap_emb(gap.clamp(max=255))
        return torch.cat([c, t[:, None]], 1)

    def token_inputs(self, tok, frame_t, remaining, dt=None, state=None, key_feat=None):
        """tok: codes (B,S,G) or continuous (B,S,C); frame_t/remaining (B,S) ints;
        key_feat: decoded normalised key features (B,S,136), kf with codes only."""
        if self.continuous:
            h = self.tok_in(tok)
        elif self.levels:
            dg = self.digits(tok).flatten(-2)
            h = self.code_emb(dg + torch.arange(dg.shape[-1], device=tok.device) * self.lmax).sum(-2)
        else:
            off = torch.arange(self.groups, device=tok.device) * self.codebook
            h = self.code_emb(tok + off).sum(-2)
        h = h + self.type_emb.weight[2] + sinusoid(frame_t + 512, self.d) + sinusoid(remaining.clamp(min=0) + 2048, self.d)
        if self.kind == "kf":
            h = h + self.dt_emb(dt.clamp(0, MAX_GAP)) + self.state_in(state)
            if not self.continuous:
                h = h + self.key_in(key_feat)
        return h

    def run(self, prefix, seq, pad=None):
        """prefix (B,P,d) bidirectional, seq (B,S,d) causal. Returns hidden states for seq."""
        P, S = prefix.shape[1], seq.shape[1]
        L = P + S
        mask = torch.zeros(L, L, dtype=torch.bool, device=seq.device)
        mask[:P, P:] = True  # prefix does not see the sequence
        mask[P:, P:] = torch.triu(torch.ones(S, S, dtype=torch.bool, device=seq.device), 1)
        kpm = None
        if pad is not None:
            kpm = torch.cat([torch.zeros(seq.shape[0], P, dtype=torch.bool, device=seq.device), pad], 1)
        h = self.backbone(torch.cat([prefix, seq], 1), mask=mask, src_key_padding_mask=kpm)
        return self.norm(h[:, P:])

    # ------------------------------------------------------------------ heads
    def digits(self, codes):
        """FSQ code indices (...,G) -> level per dimension (...,G,dims), FSQ.decode_indices order."""
        return (codes[..., None] // self.basis) % self.lv

    def code_logits(self, h):
        """(...,G,codebook) joint, or (...,G,dims,max level) per dimension (unused levels masked)."""
        if not self.levels:
            return self.code_head(h).unflatten(-1, (self.groups, self.codebook))
        lg = self.code_head(h).unflatten(-1, (self.groups, len(self.levels), self.lmax))
        return lg.masked_fill(torch.arange(self.lmax, device=h.device) >= self.lv[:, None], -1e9)

    def code_loss(self, h, codes):
        """Negative log-likelihood of codes (N,G) per group (per-dimension terms summed)."""
        lg = self.code_logits(h)
        tgt = self.digits(codes) if self.levels else codes
        nll = Fn.cross_entropy(lg.flatten(0, -2), tgt.flatten(), reduction="none").view(tgt.shape)
        return (nll.sum(-1) if self.levels else nll).mean()

    def sample_codes(self, h, temperature, gen=None):
        """Codes (...,G). Per-dimension heads at temperature 0 take the expected level, rounded
        (the levels are ordered, so this is the latent closest to the conditional mean)."""
        lg = self.code_logits(h)
        if not self.levels:
            return pick(lg, temperature, gen)
        if temperature <= 0:
            dg = (lg.softmax(-1) * torch.arange(self.lmax, device=lg.device)).sum(-1).round().long()
        else:
            dg = pick(lg, temperature, gen)
        return (dg * self.basis).sum(-1)

    def pose_condition(self, h, next_dt):
        return self.pose_cond(torch.cat([h, self.dt_emb(next_dt.clamp(0, MAX_GAP))], -1))

    @staticmethod
    def dt_mask(t_now, n_total):
        """Allowed Δt in 1..min(32, target - t_now). (B,S) -> (B,S,33) bool."""
        dts = torch.arange(MAX_GAP + 1, device=t_now.device)
        rem = (n_total[:, None] - t_now)[..., None]
        return (dts >= 1) & (dts <= rem)
