"""GPU-resident random-crop sampler over LAFAN1 sequences."""
from __future__ import annotations

import numpy as np
import torch

from . import features as F
from . import lafan
from . import quat as Q


class Clips:
    def __init__(self, data: lafan.Lafan, actors, device="cuda"):
        seqs = data.select(actors)
        self.device = device
        self.root = torch.from_numpy(np.concatenate([s.root_pos for s in seqs])).float().to(device)
        self.lq = torch.from_numpy(np.concatenate([s.local_q for s in seqs])).float().to(device)
        lens = torch.tensor([len(s.root_pos) for s in seqs])
        self.starts = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]]).to(device)
        self.lens = lens.to(device)
        self.themes = [s.theme for s in seqs]
        self.parents = data.skeleton.parents
        self.offsets = torch.from_numpy(data.skeleton.offsets).float().to(device)
        # world-frame global joint positions, so ground truth needs no FK per step
        self.gp = torch.cat([Q.fk(self.lq[i:i + 65536], lafan.local_positions(self.root[i:i + 65536], self.offsets),
                                  self.parents)[1] for i in range(0, len(self.root), 65536)])
        self.key_eps: list[float] = []
        self.key_masks = None  # (n_eps, N) adaptive key masks computed on whole sequences

    def precompute_keys(self, eps_list, max_gap: int = 32, cache: str | None = None):
        """Adaptive (Douglas-Peucker) key masks on every whole sequence, for each eps. Crops take
        the keys inside them plus their two end frames (see crop_mask)."""
        import os

        from . import keyframes

        if cache and os.path.exists(cache):
            c = torch.load(cache, map_location=self.device)
            if c["eps"] == list(eps_list) and c["masks"].shape[1] == len(self.root):
                self.key_eps, self.key_masks = list(eps_list), c["masks"]
                return
        masks = torch.zeros(len(eps_list), len(self.root), dtype=torch.bool, device=self.device)
        for s0, n in zip(self.starts.tolist(), self.lens.tolist()):
            r, q = self.root[s0:s0 + n][None], self.lq[s0:s0 + n][None]
            for k, eps in enumerate(eps_list):
                masks[k, s0:s0 + n] = keyframes.adaptive(r, q, self.offsets, self.parents, eps, max_gap, max_iter=256)[0]
        self.key_eps, self.key_masks = list(eps_list), masks
        if cache:
            torch.save({"eps": list(eps_list), "masks": masks}, cache)

    def crop_mask(self, idx: torch.Tensor, k: int) -> torch.Tensor:
        m = self.key_masks[k][idx].clone()
        m[:, 0] = m[:, -1] = True
        return m

    def sample(self, B: int, T: int, anchor: int = 0, gen: torch.Generator | None = None):
        """B random crops of T frames, canonicalised so frame `anchor` is at the origin facing +x.
        Returns root (B,T,3), lq (B,T,J,4), global positions gp (B,T,J,3), frame indices (B,T)."""
        valid = (self.lens - T).clamp(min=0) + 1
        w = valid.float() / valid.sum()
        seq = torch.multinomial(w, B, replacement=True, generator=gen)
        off = (torch.rand(B, device=self.device, generator=gen) * valid[seq]).long()
        idx = (self.starts[seq] + off)[:, None] + torch.arange(T, device=self.device)
        root, lq, gp = self.root[idx], self.lq[idx], self.gp[idx]
        b = torch.arange(B, device=self.device)
        inv = Q.qinv(Q.heading_quat(lq[b, anchor, 0]))[:, None]
        shift = root[b, anchor] * root.new_tensor([1.0, 0.0, 1.0])
        root = Q.qrot(inv, root - shift[:, None])
        gp = Q.qrot(inv[:, None], gp - shift[:, None, None])
        lq = lq.clone()
        lq[..., 0, :] = Q.qmul(inv, lq[..., 0, :])
        return root, lq, gp, idx

    def global_positions(self, root, lq):
        return Q.fk(lq, lafan.local_positions(root, self.offsets), self.parents)[1]


@torch.no_grad()
def feature_stats(clips: Clips, n: int = 4096, T: int = 128, eps: float = 1e-3) -> dict:
    """Per-dimension mean/std of the dense relative form (velocity-style features)."""
    root, lq, _, _ = clips.sample(n, T)
    f = F.relative_form(root, lq)[:, 1:]  # skip the first frame (Δ from origin is zero)
    f = f.reshape(-1, f.shape[-1])
    return {"mean": f.mean(0), "std": f.std(0).clamp(min=eps)}
