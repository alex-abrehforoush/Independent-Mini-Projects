"""Why the better tokenizer can make the worse generator: diagnostics for the discrete generators
on random crops of held-out subject 5 (same FSQ code space, (8,5,5,5) x 4 groups, for both).

* cross-entropy rate: bits per second of transition the generator still has to resolve given the
  ground-truth prefix (codes, plus Δt for keyframes), next to the nominal bits it emits;
* code-error sensitivity: how far the decoded transition moves when one code dimension of one
  generated token steps one FSQ level, every other code held fixed (a lower bound on the cost of a
  sampling mistake, since later codes do not get to correct it);
* decode ceiling: transition error of the ground-truth codes decoded the way the generator decodes
  its own, next to SLERP and the generator's greedy output on the same crops.
Writes runs/gen/<run>/token_analysis.json."""
import argparse
import json
import math
import os
import sys
from importlib import import_module

import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(__file__))
from motion_tokens import baselines
from motion_tokens import data as D
from motion_tokens import features as F
from motion_tokens import gen_data as G
from motion_tokens import keyframes, lafan, placement
from motion_tokens import quat as Q
from motion_tokens.generator import N_CTX

ev = import_module("21_eval_generator")
DT_BITS = 5


def kf_decode(tok, codes, t, crop_r, crop_q):
    """Ground-truth key placement, given codes -> dense crop, as generate_kf decodes: closed-loop
    key decoding from the coded anchor, exact anchor and target poses, learned in-betweener."""
    B, K = codes.shape[:2]
    T = crop_r.shape[1]
    emb = tok.dt_emb(t["dt"])
    prev, feats = codes.new_zeros(B, F.D_REL, dtype=torch.float), []
    for i in range(K):
        prev = tok.dec_key(ev.codes_to_latent(tok.q, codes[:, i]), emb[:, i], prev)
        feats.append(prev)
    kr, kq = tok._absolute_keys(tok.nrm.denorm(torch.stack(feats, 1)), t["dt"])
    keep = t["valid"] & (t["keys"] < T - 1)
    kr = torch.where(keep[..., None], kr, crop_r[:, -1:])
    kq = torch.where(keep[..., None, None], kq, crop_q[:, -1:])
    kr = torch.cat([crop_r[:, :1], kr[:, 1:], crop_r[:, -1:]], 1)
    kq = torch.cat([crop_q[:, :1], kq[:, 1:], crop_q[:, -1:]], 1)
    idx = torch.cat([torch.where(keep, t["keys"], T - 1), torch.full((B, 1), T - 1, device=codes.device)], 1)
    return tok.inbetween(kr, kq, idx, T)


def fr_decode(tok, codes, crop_r, crop_q):
    T = crop_r.shape[1]
    r, q = tok.decode(ev.codes_to_latent(tok.q, codes), codes.shape[1] * tok.down)
    return G.blend_endpoints(r[:, :T], q[:, :T], crop_r[:, 0], crop_q[:, 0], crop_r[:, -1], crop_q[:, -1])


def nudge(model, codes, sel, gen):
    """Move one random code dimension of the selected tokens (B, S, G) one level (up or down,
    inwards at the edges). sel: (B, S) bool with at least one True per row."""
    B, S, Gr = codes.shape
    dev = codes.device
    s = torch.multinomial(sel.float(), 1, generator=gen)[:, 0]
    g = torch.randint(Gr, (B,), generator=gen, device=dev)
    b = torch.arange(B, device=dev)
    dg = model.digits(codes[b, s, g])  # (B, dims)
    k = torch.randint(dg.shape[-1], (B,), generator=gen, device=dev)
    lv = model.lv[k]
    d = dg[b, k]
    step = torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1, 1)
    step = torch.where(d + step < 0, 1, torch.where(d + step >= lv, -1, step))
    out = codes.clone()
    out[b, s, g] = codes[b, s, g] + step * model.basis[k]
    return out


@torch.no_grad()
def analyse(run, clips, gaps, B, dev):
    cfg, tok, m, _ = ev.load_generator(run, dev)
    assert m.levels, "per-dimension FSQ code heads only"
    off, P = clips.offsets, clips.parents
    eps_k = min(range(len(clips.key_eps)), key=lambda k: abs(clips.key_eps[k] - cfg["eps"])) if cfg["eps"] else None
    out = {}
    for n in gaps:
        g = torch.Generator(device=dev).manual_seed(n)
        root, lq, gp, idx = clips.sample(B, N_CTX + n + 1, anchor=N_CTX - 1, gen=g)
        ctx, tgt, gap = G.conditioning(tok, root, lq, n)
        prefix = m.prefix(ctx, tgt, gap)
        crop_r, crop_q = root[:, N_CTX - 1:], lq[:, N_CTX - 1:]
        T = n + 2
        n_total = torch.full((B,), T - 1, device=dev)
        if cfg["kind"] == "kf":
            mask = keyframes.uniform(B, T, cfg["stride"], dev) if cfg["stride"] else clips.crop_mask(idx[:, N_CTX - 1:], eps_k)
            t = G.kf_targets(tok, crop_r, crop_q, mask, False)
            h = m.run(prefix, G.kf_step_inputs(m, t, n_total, root[:, -1], lq[:, -1]), ~t["valid"][:, 1:])
            real, next_dt = t["valid"][:, 1:], t["dt"][:, 1:]
            lg_dt = m.dt_head(h).masked_fill(~m.dt_mask(t["keys"][:, :-1], n_total), -1e9)
            nll_dt = Fn.cross_entropy(lg_dt[real], next_dt[real], reduction="sum")
            j = torch.arange(h.shape[1], device=dev)[None]
            inner = j + 1 < (t["valid"].sum(1) - 1)[:, None]
            lg = m.code_logits(m.pose_condition(h, next_dt)[inner])
            nll_code = Fn.cross_entropy(lg.flatten(0, -2), m.digits(t["tokens"][:, 1:][inner]).flatten(), reduction="sum")
            n_tok = inner.sum().item()
            nominal = n_tok * tok.q.bits + (0 if cfg["stride"] else real.sum().item() * DT_BITS)  # uniform Δt is free
            codes = t["tokens"]
            sel = torch.cat([torch.zeros_like(inner[:, :1]), inner], 1)  # inner keys only (anchor is given)
            decode = lambda c: kf_decode(tok, c, t, crop_r, crop_q)
            generate = lambda: ev.generate_kf(m, tok, root, lq, n)
        else:
            t = G.fr_targets(tok, crop_r, crop_q, False)
            h = m.run(prefix, G.fr_step_inputs(m, t["tokens"], n_total))
            lg = m.code_logits(h)
            nll_dt = torch.zeros(())
            nll_code = Fn.cross_entropy(lg.flatten(0, -2), m.digits(t["tokens"]).flatten(), reduction="sum")
            n_tok = t["tokens"].shape[0] * t["tokens"].shape[1]
            nominal = n_tok * tok.q.bits
            codes = t["tokens"]
            sel = torch.ones(codes.shape[:2], dtype=torch.bool, device=dev)
            decode = lambda c: fr_decode(tok, c, crop_r, crop_q)
            generate = lambda: ev.generate_fr(m, tok, root, lq, n)

        def positions(r, q):
            return Q.fk(q, lafan.local_positions(r, off), P)[1]

        sec = B * n / 30.0
        gt = gp[:, N_CTX:N_CTX + n]
        clean = positions(*decode(codes))[:, 1:-1]
        has = sel.any(1)  # keyframe crops with no inner key have nothing to nudge
        moved = positions(*decode(nudge(m, codes, sel | ~has[:, None], g)))[:, 1:-1]
        delta = (moved - clean).norm(dim=-1).mean(-1)[has]  # (B', n) per-frame mean joint displacement
        r_g, q_g, _ = generate()
        r_s, q_s = baselines.interpolation(root, lq, N_CTX, n)
        out[n] = {
            "tokens_per_s": n_tok / sec,
            "nominal_bits_per_s": nominal / sec,
            "ce_bits_per_s": (nll_code + nll_dt).item() / math.log(2) / sec,
            "ce_dt_bits_per_s": nll_dt.item() / math.log(2) / sec,
            "ce_bits_per_token": nll_code.item() / math.log(2) / max(n_tok, 1),
            "nudge_cm": delta.mean().item(),
            "nudge_reach": (delta > 1.0).float().mean().item(),
            "nudge_crops": has.float().mean().item(),
            "ceiling_mpjpe_cm": (clean - gt).norm(dim=-1).mean().item(),
            "greedy_mpjpe_cm": (positions(r_g, q_g) - gt).norm(dim=-1).mean().item(),
            "slerp_mpjpe_cm": (positions(r_s, q_s) - gt).norm(dim=-1).mean().item(),
        }
        print(f"{run} n={n}: " + "  ".join(f"{k} {v:.3f}" for k, v in out[n].items()), flush=True)
    json.dump(out, open(os.path.join(run, "token_analysis.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["runs/gen/fr_fsq_g4", "runs/gen/kf_g4p"])
    ap.add_argument("--gaps", default="5,15,45,90")
    ap.add_argument("--n", type=int, default=512, help="crops per gap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = lafan.load("data/lafan1.npz")
    clips = D.Clips(d, lafan.TEST_ACTORS, a.device)
    eps = sorted({json.load(open(os.path.join(r, "config.json")))["eps"] for r in a.runs} - {None})
    clips.precompute_keys([min(placement.EPS_GRID, key=lambda e: abs(e - x)) for x in eps])
    for run in a.runs:
        analyse(run, clips, [int(x) for x in a.gaps.split(",")], a.n, a.device)


if __name__ == "__main__":
    main()
