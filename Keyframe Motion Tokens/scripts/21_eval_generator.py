"""Experiment 2 evaluation: in-betweening on held-out subject 5.

Official LAFAN1 windows (65 frames every 40) for n = 5/15/30/45, longer windows (101 frames
every 40) for n = 60/90.  Benchmark numbers decode at temperature 0: greedy codes (discrete) or the
flow ODE from zero noise (continuous), so neither family is scored on a random sample. Diversity
comes from several temperature-1 samples on a subset."""
import argparse
import json
import os
import sys
from importlib import import_module

import torch

sys.path.insert(0, os.path.dirname(__file__))
from motion_tokens import baselines, lafan, metrics
from motion_tokens import gen_data as G
from motion_tokens import quat as Q
from motion_tokens.generator import MAX_GAP, N_CTX, Generator, pick, relative_target_state
from motion_tokens.quantizers import FSQ

load_tok = import_module("11_eval_tokenizer").load_model


def load_generator(run, dev):
    cfg = json.load(open(os.path.join(run, "config.json")))
    _, tok = load_tok(cfg["tok"], dev)
    kind, cont = cfg["kind"], bool(cfg["continuous"])
    cont_stats = None
    if cont and os.path.exists(os.path.join(run, "cont_stats.pt")):  # latents (raw keyframe features have none)
        cont_stats = torch.load(os.path.join(run, "cont_stats.pt"), map_location=dev)
    if cont:
        groups, codebook, cont_dim = 1, 1, cont_stats["mean"].shape[0] if cont_stats is not None else 136
    else:
        groups, codebook, cont_dim = tok.q.groups, tok.q.codebook_size, 1
    levels = None
    if cfg.get("code_head", "joint") == "digits" and isinstance(tok.q, FSQ):
        levels = [int(x) for x in tok.q.levels.tolist()]
    m = Generator(kind, cont, groups, codebook, cont_dim, cfg["d"], cfg["layers"], levels=levels)
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev))
    return cfg, tok, m.to(dev).eval(), cont_stats


def codes_to_latent(q, codes):
    if hasattr(q, "decode_indices"):
        return q.decode_indices(codes)
    if hasattr(q, "stages"):  # RVQ
        return sum(st.embed[0][codes[..., i]] for i, st in enumerate(q.stages))
    G_ = q.groups  # grouped VQ
    return torch.cat([q.embed[g][codes[..., g]] for g in range(G_)], -1)


@torch.no_grad()
def generate_kf(model, tok, root, lq, n, temperature=0.0, gen=None, flow_steps=32, return_keys=False,
                cont_stats=None):
    """root/lq (B, N_CTX+n+1) in the anchor frame. Returns transition (B,n,3), (B,n,J,4), keys/s.
    cont_stats: continuous key latents' normalisation; without it, continuous tokens are raw features."""
    B, dev = root.shape[0], root.device
    ctx, tgt, gap = G.conditioning(tok, root, lq, n)
    prefix = model.prefix(ctx, tgt, gap)
    crop_r, crop_q = root[:, N_CTX - 1:], lq[:, N_CTX - 1:]
    T = n + 2
    n_total = torch.full((B,), T - 1, device=dev)
    tgt_r, tgt_q = root[:, -1], lq[:, -1]
    anchor_keys = torch.zeros(B, 1, dtype=torch.long, device=dev)
    f0, dt0 = tok._key_features(crop_r, crop_q, anchor_keys)
    fn0 = tok.nrm.norm(f0)
    if model.continuous and cont_stats is None:
        toks = fn0
        feats = fn0
    elif model.continuous:
        zero = torch.zeros_like(fn0)
        z0, _, _ = tok.code_key(fn0, tok.dt_emb(dt0), zero)
        toks = (z0 - cont_stats["mean"]) / cont_stats["std"]
        feats = tok.dec_key(z0, tok.dt_emb(dt0), zero)
    else:
        zero = torch.zeros_like(fn0)
        _, toks, _ = tok.code_key(fn0, tok.dt_emb(dt0), zero)
        feats = tok.dec_key(codes_to_latent(tok.q, toks), tok.dt_emb(dt0), zero)  # normalized features
    keys, dts = anchor_keys, dt0
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(T):
        kr, kq = tok._absolute_keys(tok.nrm.denorm(feats), dts)
        state = relative_target_state(kr, kq, tgt_r, tgt_q)
        rem = n_total[:, None] - keys
        h = model.run(prefix, model.token_inputs(toks, keys, rem, dts, state, feats))[:, -1]
        lg = model.dt_head(h).masked_fill(~model.dt_mask(keys[:, -1:], n_total)[:, 0], -1e9)
        dt = pick(lg, temperature, gen)
        new_t = keys[:, -1] + dt
        reached = (new_t >= T - 1) | done
        c = model.pose_condition(h, dt)
        if model.continuous:
            tok_new = model.head.sample(c, flow_steps, temperature, gen)
            f_new = tok_new if cont_stats is None else \
                tok.dec_key(tok_new * cont_stats["std"] + cont_stats["mean"], tok.dt_emb(dt), feats[:, -1])
        else:
            codes = model.sample_codes(c, temperature, gen)
            zq = codes_to_latent(tok.q, codes)
            f_new = tok.dec_key(zq, tok.dt_emb(dt), feats[:, -1])
            tok_new = codes
        # finished sequences repeat their last key with Δt = 0 (ignored later)
        dt = torch.where(reached, torch.zeros_like(dt), dt)
        new_t = torch.where(reached, keys[:, -1], new_t)
        keys = torch.cat([keys, new_t[:, None]], 1)
        dts = torch.cat([dts, dt[:, None]], 1)
        feats = torch.cat([feats, torch.where(reached[:, None], feats[:, -1], f_new)[:, None]], 1)
        toks = torch.cat([toks, torch.where(reached.view(-1, *[1] * (tok_new.dim() - 1)), toks[:, -1], tok_new)[:, None]], 1)
        done = reached
        if done.all():
            break
    # absolute keys; valid generated keys are those with dt > 0 (plus the anchor)
    kr, kq = tok._absolute_keys(tok.nrm.denorm(feats), dts)
    valid = dts > 0
    valid[:, 0] = True
    # build padded key list: anchor, generated..., target (padding repeats the target)
    t_idx = torch.where(valid, keys, torch.full_like(keys, T - 1))
    order = torch.where(valid, keys, keys + 10 * T).argsort(1)
    t_idx = torch.gather(t_idx, 1, order)
    b = torch.arange(B, device=dev)[:, None]
    kr, kq, val = kr[b, order], kq[b, order], valid[b, order]
    kr = torch.where(val[..., None], kr, tgt_r[:, None])
    kq = torch.where(val[..., None, None], kq, tgt_q[:, None])
    kr = torch.cat([crop_r[:, :1], kr[:, 1:], tgt_r[:, None]], 1)
    kq = torch.cat([crop_q[:, :1], kq[:, 1:], tgt_q[:, None]], 1)
    t_idx = torch.cat([t_idx, torch.full((B, 1), T - 1, device=dev)], 1)
    r, q = tok.inbetween(kr, kq, t_idx, T)
    n_keys = val.sum(1) - 1  # generated keys strictly inside the gap
    if return_keys:  # frame indices (crop frame 0 = last context frame) of the generated keys
        return r[:, 1:-1], q[:, 1:-1], n_keys, [t_idx[b_, 1:1 + int(n_keys[b_])].tolist() for b_ in range(B)]
    return r[:, 1:-1], q[:, 1:-1], n_keys


@torch.no_grad()
def generate_fr(model, tok, root, lq, n, temperature=0.0, gen=None, cont_stats=None, flow_steps=32):
    B, dev = root.shape[0], root.device
    ctx, tgt, gap = G.conditioning(tok, root, lq, n)
    prefix = model.prefix(ctx, tgt, gap)
    T = n + 2
    S = -(-T // tok.down)
    n_total = torch.full((B,), T - 1, device=dev)
    toks = None
    for i in range(S):
        dummy = torch.zeros(B, 1, *(() if not model.continuous else (cont_stats["mean"].shape[0],)),
                            dtype=torch.float if model.continuous else torch.long, device=dev)
        if not model.continuous:
            dummy = dummy.unsqueeze(-1).expand(B, 1, model.groups)
        seq = dummy if toks is None else torch.cat([toks, dummy], 1)
        h = model.run(prefix, G.fr_step_inputs(model, seq, n_total))[:, -1]
        if model.continuous:
            new = model.head.sample(h, flow_steps, temperature, gen)
        else:
            new = model.sample_codes(h, temperature, gen)
        toks = new[:, None] if toks is None else torch.cat([toks, new[:, None]], 1)
    if model.continuous:
        z = toks * cont_stats["std"] + cont_stats["mean"]
    else:
        z = codes_to_latent(tok.q, toks)
    r, q = tok.decode(z, S * tok.down)
    r, q = r[:, :T], q[:, :T]
    r, q = G.blend_endpoints(r, q, root[:, N_CTX - 1], lq[:, N_CTX - 1], root[:, -1], lq[:, -1])
    return r[:, 1:-1], q[:, 1:-1], torch.full((B,), S, device=dev)


def evaluate(method, windows, n, stats, off, P, bs=256):
    """method(root, lq, n) on anchor-frame crops -> transition. Returns scores + raw outputs."""
    root_w, lq_w = windows["root"], windows["lq"]
    shift = root_w[:, N_CTX - 1] * root_w.new_tensor([1.0, 0.0, 1.0])
    root_a = root_w - shift[:, None]
    rs, qs, ks = [], [], []
    for i in range(0, root_w.shape[0], bs):
        r, q, k = method(root_a[i:i + bs, :N_CTX + n + 1], lq_w[i:i + bs, :N_CTX + n + 1], n)
        rs.append(r + shift[i:i + bs, None])
        qs.append(q)
        ks.append(k.float())
    r, q, k = torch.cat(rs), torch.cat(qs), torch.cat(ks)
    s, e = N_CTX, N_CTX + n
    gt_r, gt_q = root_w[:, s:e], lq_w[:, s:e]
    sc = metrics.benchmark_scores(q.double(), r.double(), gt_q.double(), gt_r.double(), off.double(), P, stats,
                                  ref_lq=lq_w[:, s - 1].double())
    # physics on context tail + transition + target, so boundary pops count
    full_r = torch.cat([root_w[:, s - 3:s], r, root_w[:, e:e + 1]], 1)
    full_q = torch.cat([lq_w[:, s - 3:s], Q.qnormalize(q), lq_w[:, e:e + 1]], 1)
    full_q = torch.cat([full_q[:, :1], Q.remove_discontinuities(full_q, dim=1)[:, 1:]], 1)
    _, pp = Q.fk(full_q, lafan.local_positions(full_r, off), P)
    _, gp = Q.fk(lq_w[:, s - 3:e + 1], lafan.local_positions(root_w[:, s - 3:e + 1], off), P)
    _, gp_all = Q.fk(lq_w, lafan.local_positions(root_w, off), P)
    contacts = metrics.foot_contacts(gp_all)[:, s - 3:e + 1]
    floor = gp_all[..., [lafan.FEET[1], lafan.FEET[3]], 1].amin((1, 2))
    low = gp_all[..., list(lafan.FEET), 1].amin(1)
    sc.update({
        "skate_cm_s": metrics.foot_skate(pp, contacts).mean().item(),
        "skate_gt_cm_s": metrics.foot_skate(gp, contacts).mean().item(),
        "skate_own_cm_s": metrics.foot_skate_own(pp, low).mean().item(),
        "skate_own_gt_cm_s": metrics.foot_skate_own(gp, low).mean().item(),
        "penetration_cm": metrics.penetration(pp, floor).mean().item(),
        "jitter": metrics.jitter(pp).mean().item(),
        "jitter_gt": metrics.jitter(gp).mean().item(),
        "mpjpe_cm": (pp[:, 3:-1] - gp[:, 3:-1]).norm(dim=-1).mean().item(),
        "tokens_or_keys": k.mean().item(),
    })
    return sc, (r, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--gaps", default="5,15,30,45,60,90")
    ap.add_argument("--div_n", type=int, default=256, help="windows used for the diversity estimate")
    ap.add_argument("--div_k", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N windows (smoke tests)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    d = lafan.load("data/lafan1.npz")
    P = d.skeleton.parents
    off = torch.from_numpy(d.skeleton.offsets).to(dev)
    stats = {k: v.to(dev) for k, v in lafan.train_stats(d).items()}
    win = {"short": lafan.benchmark_windows(d, lafan.TEST_ACTORS, 65, 40, dtype=torch.float32),
           "long": lafan.benchmark_windows(d, lafan.TEST_ACTORS, 101, 40, dtype=torch.float32)}
    for w in win.values():
        sl = slice(0, a.limit or None)
        w["root"], w["lq"] = w["root"][sl].to(dev), w["lq"][sl].to(dev)
    gaps = [int(x) for x in a.gaps.split(",")]

    def windows_for(n):
        return win["short"] if n <= 45 else win["long"]

    offf = off.float()
    results = {}
    if a.baselines:
        def slerp(root, lq, n):
            r, q = baselines.interpolation(root, lq, N_CTX, n)
            return r, q, torch.zeros(root.shape[0], device=root.device)
        res = {}
        for n in gaps:
            res[n], _ = evaluate(slerp, windows_for(n), n, stats, offf, P)
            print(f"slerp n={n}: " + "  ".join(f"{k} {v:.4f}" for k, v in res[n].items()))
        results["slerp"] = res
    for run in a.runs:
        cfg, tok, model, cont_stats = load_generator(run, dev)
        res = {}
        for n in gaps:
            gen = torch.Generator(device=dev).manual_seed(0)
            if cfg["kind"] == "kf":
                fn = lambda r, q, n_: generate_kf(model, tok, r, q, n_, 0.0, gen, cont_stats=cont_stats)
            else:
                fn = lambda r, q, n_: generate_fr(model, tok, r, q, n_, 0.0, gen, cont_stats)
            res[n], _ = evaluate(fn, windows_for(n), n, stats, offf, P)
            # diversity: k samples at temperature 1 on a subset
            w = windows_for(n)
            sub = {"root": w["root"][: a.div_n], "lq": w["lq"][: a.div_n]}
            samples = []
            for k in range(a.div_k):
                g2 = torch.Generator(device=dev).manual_seed(100 + k)
                if cfg["kind"] == "kf":
                    fs = lambda r, q, n_: generate_kf(model, tok, r, q, n_, 1.0, g2, cont_stats=cont_stats)
                else:
                    fs = lambda r, q, n_: generate_fr(model, tok, r, q, n_, 1.0, g2, cont_stats)
                sc_k, (r_k, q_k) = evaluate(fs, sub, n, stats, offf, P)
                _, pk = Q.fk(q_k, lafan.local_positions(r_k, offf), P)
                samples.append((pk, sc_k))
            pos = torch.stack([s[0] for s in samples])  # (k,B,n,J,3)
            apd = torch.stack([(pos[i] - pos[j]).norm(dim=-1).mean() for i in range(a.div_k) for j in range(i + 1, a.div_k)]).mean()
            res[n]["apd_cm"] = apd.item()
            res[n]["sampled_L2P"] = sum(s[1]["L2P"] for s in samples) / a.div_k
            res[n]["sampled_skate"] = sum(s[1]["skate_cm_s"] for s in samples) / a.div_k
            print(f"{run} n={n}: " + "  ".join(f"{k} {v:.4f}" for k, v in res[n].items()), flush=True)
        results[run] = res
        json.dump(res, open(os.path.join(run, "eval_inbetween.json"), "w"), indent=1)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
