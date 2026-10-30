"""Quantizers with an explicit bit cost: grouped FSQ, grouped EMA-VQ, residual VQ."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023), G independent groups of len(levels) dims."""

    def __init__(self, levels=(8, 5, 5, 5), groups: int = 1):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32), persistent=False)
        basis = torch.cumprod(torch.tensor([1] + list(levels[:-1])), 0)
        self.register_buffer("basis", basis, persistent=False)
        self.groups, self.gdim = groups, len(levels)
        self.dim = groups * self.gdim
        self.codebook_size = int(math.prod(levels))
        self.bits = groups * math.log2(self.codebook_size)

    def bound(self, z):
        half = (self.levels - 1) * (1 - 1e-3) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half)
        return torch.tanh(z + shift) * half - offset

    def forward(self, z):
        """z (...,G*gdim) -> quantized values in [-1,1] (straight-through), indices (...,G), aux loss."""
        zg = z.unflatten(-1, (self.groups, self.gdim))
        b = self.bound(zg)
        q = b + (torch.round(b) - b).detach()
        half_w = torch.div(self.levels, 2, rounding_mode="floor")
        idx = ((torch.round(b) + half_w) * self.basis).sum(-1).long()
        return (q / half_w).flatten(-2), idx, z.new_zeros(())

    def decode_indices(self, idx):
        half_w = torch.div(self.levels, 2, rounding_mode="floor")
        digits = (idx[..., None] // self.basis) % self.levels.long()
        return ((digits.float() - half_w) / half_w).flatten(-2)


class VQ(nn.Module):
    """EMA vector quantizer (van den Oord 2017 / Razavi 2019) with dead-code restarts,
    applied independently to G groups."""

    def __init__(self, dim: int, codebook_size: int = 1024, groups: int = 1, decay: float = 0.99,
                 beta: float = 0.25):
        super().__init__()
        self.groups, self.gdim, self.K = groups, dim // groups, codebook_size
        self.dim, self.decay, self.beta = dim, decay, beta
        self.codebook_size = codebook_size
        self.bits = groups * math.log2(codebook_size)
        self.register_buffer("embed", torch.randn(groups, codebook_size, self.gdim) * 0.1)
        self.register_buffer("cluster_size", torch.ones(groups, codebook_size))
        self.register_buffer("embed_sum", self.embed.clone())

    def forward(self, z):
        shape = z.shape
        zg = z.reshape(-1, self.groups, self.gdim).transpose(0, 1)  # (G,N,d)
        d = (zg.pow(2).sum(-1, keepdim=True) - 2 * zg @ self.embed.transpose(1, 2)
             + self.embed.pow(2).sum(-1)[:, None])
        idx = d.argmin(-1)  # (G,N)
        q = torch.gather(self.embed, 1, idx[..., None].expand(-1, -1, self.gdim))
        if self.training:
            with torch.no_grad():
                onehot = Fn.one_hot(idx, self.K).type_as(zg)  # (G,N,K)
                self.cluster_size.mul_(self.decay).add_(onehot.sum(1), alpha=1 - self.decay)
                self.embed_sum.mul_(self.decay).add_(onehot.transpose(1, 2) @ zg, alpha=1 - self.decay)
                n = self.cluster_size.sum(-1, keepdim=True)
                size = (self.cluster_size + 1e-5) / (n + self.K * 1e-5) * n
                self.embed.copy_(self.embed_sum / size[..., None])
                dead = self.cluster_size < 1e-2
                if dead.any():
                    for g in range(self.groups):
                        nd = int(dead[g].sum())
                        if nd:
                            pick = zg[g][torch.randint(0, zg.shape[1], (nd,), device=z.device)]
                            self.embed[g, dead[g]] = pick
                            self.embed_sum[g, dead[g]] = pick
                            self.cluster_size[g, dead[g]] = 1.0
        commit = self.beta * Fn.mse_loss(zg, q.detach())
        q = zg + (q - zg).detach()
        q = q.transpose(0, 1).reshape(shape)
        idx = idx.transpose(0, 1).reshape(*shape[:-1], self.groups)
        return q, idx, commit


class RVQ(nn.Module):
    """Residual VQ: n_q stages, each a single-group EMA VQ on the running residual."""

    def __init__(self, dim: int, codebook_size: int = 1024, n_q: int = 2):
        super().__init__()
        self.stages = nn.ModuleList(VQ(dim, codebook_size) for _ in range(n_q))
        self.dim, self.codebook_size = dim, codebook_size
        self.bits = n_q * math.log2(codebook_size)

    def forward(self, z):
        res, out, idxs, loss = z, torch.zeros_like(z), [], z.new_zeros(())
        for st in self.stages:
            q, i, l = st(res)
            out = out + q
            res = res - q.detach()
            idxs.append(i)
            loss = loss + l
        return out, torch.cat(idxs, -1), loss


def make(kind: str, dim: int | None = None, groups: int = 1, levels=(8, 5, 5, 5), codebook_size=1024):
    if kind == "fsq":
        return FSQ(levels, groups)
    if kind == "vq":
        return VQ(dim, codebook_size, groups)
    if kind == "rvq":
        return RVQ(dim, codebook_size, groups)
    if kind == "none":
        return None
    raise ValueError(kind)
