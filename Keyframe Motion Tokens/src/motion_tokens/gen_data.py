"""Turning (context, transition, target) crops into generator inputs/targets for both token
families, and decoding generated tokens back to motion. Shared by training and evaluation."""
from __future__ import annotations

import torch

from . import features as F
from . import quat as Q
from .generator import N_CTX, relative_target_state, target_features
from .tokenizers import FixedRateTokenizer, KeyframeTokenizer, mask_to_index


def conditioning(tok, root, lq, n: int):
    """root/lq: (B, N_CTX + n + 1, ...) canonicalised at the last context frame."""
    ctx = tok.nrm.norm(F.relative_form(root[:, :N_CTX], lq[:, :N_CTX]))
    tgt = target_features(root, lq, torch.full((root.shape[0],), N_CTX + n, device=root.device))
    gap = torch.full((root.shape[0],), n, device=root.device, dtype=torch.long)
    return ctx, tgt, gap


def pad_to_multiple(root, lq, m: int = 4):
    T = root.shape[1]
    P = (-T) % m
    if P:
        root = torch.cat([root, root[:, -1:].expand(-1, P, -1)], 1)
        lq = torch.cat([lq, lq[:, -1:].expand(-1, P, -1, -1)], 1)
    return root, lq


# ------------------------------------------------------------------------------- keyframes

def kf_targets(tok: KeyframeTokenizer, crop_root, crop_lq, mask, continuous: bool, cont_stats=None):
    """Teacher-forcing tensors for keys inside the crop (crop frame 0 = anchor, last = target).
    Continuous tokens are the tokenizer's key latents normalised by cont_stats, or, without
    cont_stats, the raw normalised key features (the 136-number ablation)."""
    keys, valid = mask_to_index(mask)
    f, dt = tok._key_features(crop_root, crop_lq, keys)
    fn = tok.nrm.norm(f)
    if continuous and cont_stats is None:
        tokens = fn
        b = torch.arange(keys.shape[0], device=keys.device)[:, None]
        kr, kq = crop_root[b, keys], crop_lq[b, keys]
    else:
        z, codes, _, fd, dt = tok.encode_keys(crop_root, crop_lq, keys)
        tokens = (z - cont_stats["mean"]) / cont_stats["std"] if continuous else codes
        kr, kq = tok._absolute_keys(fd, dt)
        fn = tok.nrm.norm(fd)  # what the decoder made of each key: the generator's view of it
    return {"keys": keys, "valid": valid, "dt": dt, "tokens": tokens, "key_root": kr, "key_lq": kq, "key_feat": fn}


def kf_step_inputs(model, t, n_total, tgt_root, tgt_lq):
    """Inputs for positions 0..K-2 (key j predicts key j+1)."""
    keys, dt, tokens = t["keys"], t["dt"], t["tokens"]
    state = relative_target_state(t["key_root"], t["key_lq"], tgt_root, tgt_lq)
    remaining = n_total[:, None] - keys
    h_in = model.token_inputs(tokens[:, :-1], keys[:, :-1], remaining[:, :-1], dt[:, :-1], state[:, :-1],
                              t["key_feat"][:, :-1])
    return h_in


def kf_decode(tok: KeyframeTokenizer, anchor_feat, gen_feats, gen_dt, gen_keys, crop_root, crop_lq, n: int):
    """Absolute key poses from (exact anchor + generated) relative features, exact target appended,
    then the learned in-betweener. gen_* are lists over the batch of variable-length tensors."""
    B = crop_root.shape[0]
    outs_r, outs_q = [], []
    T = n + 2
    for b in range(B):
        feats = torch.cat([anchor_feat[b:b + 1], gen_feats[b]], 0)[None]
        dts = torch.cat([torch.zeros(1, dtype=torch.long, device=crop_root.device), gen_dt[b]])[None]
        kr, kq = tok._absolute_keys(feats, dts)
        kr = torch.cat([crop_root[b:b + 1, :1], kr[:, 1:], crop_root[b:b + 1, -1:]], 1)
        kq = torch.cat([crop_lq[b:b + 1, :1], kq[:, 1:], crop_lq[b:b + 1, -1:]], 1)
        keys = torch.cat([torch.zeros(1, dtype=torch.long, device=crop_root.device), gen_keys[b],
                          torch.tensor([T - 1], device=crop_root.device)])[None]
        r, q = tok.inbetween(kr, kq, keys, T)
        outs_r.append(r)
        outs_q.append(q)
    return torch.cat(outs_r), torch.cat(outs_q)


# ------------------------------------------------------------------------------- fixed rate

def fr_targets(tok: FixedRateTokenizer, crop_root, crop_lq, continuous: bool, cont_stats=None):
    r, q = pad_to_multiple(crop_root, crop_lq, tok.down)
    z = tok.encode(r, q)
    if continuous:
        return {"tokens": (z - cont_stats["mean"]) / cont_stats["std"], "z": z}
    zq, codes, _ = tok.quantize(z)
    return {"tokens": codes, "z": zq}


def fr_step_inputs(model, tokens, n_total):
    B, S = tokens.shape[:2]
    ft = (torch.arange(S, device=tokens.device) * 4).expand(B, S)
    rem = n_total[:, None] - ft
    h = model.token_inputs(tokens[:, :-1], ft[:, 1:], rem[:, 1:]) if S > 1 else None
    bos = (model.bos + model.type_emb.weight[2]).expand(B, 1, -1)
    from .tokenizers import sinusoid

    bos = bos + sinusoid(ft[:, :1] + 512, model.d) + sinusoid(rem[:, :1].clamp(min=0) + 2048, model.d)
    return bos if h is None else torch.cat([bos, h], 1)


def blend_endpoints(r, q, root0, lq0, root1, lq1):
    """Spread the start/end mismatch of a decoded crop linearly over it so both ends are exact."""
    T = r.shape[1]
    w = torch.linspace(0, 1, T, device=r.device)[None, :, None]
    e0, e1 = root0 - r[:, 0], root1 - r[:, -1]
    r = r + (1 - w) * e0[:, None] + w * e1[:, None]
    c0 = Q.qmul(lq0, Q.qinv(q[:, 0]))
    c1 = Q.qmul(lq1, Q.qinv(q[:, -1]))
    c = Q.slerp(c0[:, None].expand_as(q), c1[:, None].expand_as(q), w.expand(q.shape[:-1]))
    return r, Q.qnormalize(Q.qmul(c, q))
