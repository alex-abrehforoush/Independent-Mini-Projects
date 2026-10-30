"""Experiment 1: rate-distortion of a trained tokenizer on held-out subject 5.

Windows of 128 frames every 64 frames, canonicalised at their first frame. A fixed-rate model
gives one operating point; a keyframe model is evaluated at every key placement in the sweep
(uniform strides and adaptive eps), plus the same keys decoded *without* quantization (the
in-betweener alone) and with plain SLERP (no learning)."""
import argparse
import json
import os

import torch

from motion_tokens import features as F
from motion_tokens import keyframes, lafan, metrics, quantizers, tokenizers
from motion_tokens import quat as Q

DT_BITS = 5  # fixed-length code for Δt in 1..32 (adaptive placements only)


def load_model(run, dev):
    cfg = json.load(open(os.path.join(run, "config.json")))
    stats = torch.load(os.path.join(run, "stats.pt"), map_location=dev)
    if cfg["kind"] == "fr":
        if cfg["quant"] == "fsq":
            q = quantizers.FSQ((8, 5, 5, 5), cfg["groups"])
            latent = q.dim
        elif cfg["quant"] == "none":
            q, latent = None, 16
        else:
            latent = 64 if cfg["quant"] == "rvq" else 32 * cfg["groups"]
            q = quantizers.make(cfg["quant"], latent, cfg["groups"])
        m = tokenizers.FixedRateTokenizer(stats, q, latent, cfg["d"], cfg["layers"],
                                          abs_traj=int(cfg.get("abs_traj", 0)))
    else:
        m = tokenizers.KeyframeTokenizer(stats, cfg["groups"], d=cfg["d"], layers=cfg["layers"],
                                         quantize=cfg["quant"] != "none",
                                         predictive=bool(cfg.get("predictive", 0)))
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev))
    return cfg, m.to(dev).eval()


def scores(r, q, gp, contacts, floor, off, P, themes, bits, n_keys=None, T=128):
    _, pp = Q.fk(q, lafan.local_positions(r, off), P)
    err = (pp - gp).norm(dim=-1)
    per_win = err.mean((1, 2))
    dur = T / 30.0
    out = {
        "bits_per_s": (bits / dur).mean().item(),
        "mpjpe_cm": per_win.mean().item(),
        "max_err_cm": err.amax((1, 2)).mean().item(),
        "root_err_cm": (r - gp[:, :, 0]).norm(dim=-1).mean().item(),
        # pose error with the root position subtracted: separates body pose from trajectory
        "mpjpe_local_cm": ((pp - pp[..., :1, :]) - (gp - gp[..., :1, :])).norm(dim=-1).mean().item(),
        "skate_cm_s": metrics.foot_skate(pp, contacts).mean().item(),
        "penetration_cm": metrics.penetration(pp, floor).mean().item(),
        "jitter": metrics.jitter(pp).mean().item(),
        "per_theme_mpjpe": {t: per_win[[i for i, x in enumerate(themes) if x == t]].mean().item()
                            for t in sorted(set(themes))},
        "per_window_mpjpe": per_win.tolist(),
        "per_window_bits_per_s": (bits / dur).tolist(),
    }
    if n_keys is not None:
        out["keys_per_s"] = (n_keys.float() / dur).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="data/lafan1.npz")
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    dev = "cuda"
    d = lafan.load(a.data)
    P = d.skeleton.parents
    off = torch.from_numpy(d.skeleton.offsets).float().to(dev)
    w = lafan.benchmark_windows(d, lafan.TEST_ACTORS, 128, 64, n_past=1, dtype=torch.float32)
    root, lq = F.canonicalize(w["root"].to(dev), w["lq"].to(dev), 0)
    themes = [n.rstrip("0123456789") for n in w["names"]]
    _, gp = Q.fk(lq, lafan.local_positions(root, off), P)
    contacts = metrics.foot_contacts(gp)
    floor = gp[..., [lafan.FEET[1], lafan.FEET[3]], 1].amin((1, 2))
    gt = {"skate_cm_s": metrics.foot_skate(gp, contacts).mean().item(),
          "penetration_cm": metrics.penetration(gp, floor).mean().item(), "jitter": metrics.jitter(gp).mean().item()}
    B, T = root.shape[:2]
    batches = [slice(i, i + a.bs) for i in range(0, B, a.bs)]

    def run_batched(fn):
        outs = [fn(s) for s in batches]
        return [torch.cat([o[k] for o in outs]) for k in range(len(outs[0]))]

    placements = [("uniform", s) for s in (2, 3, 4, 6, 8, 12, 16)] + \
                 [("adaptive", e) for e in (1, 2, 3, 5, 8, 12, 20, 30, 40)]
    masks = {}
    for kind, v in placements:
        if kind == "uniform":
            masks[(kind, v)] = keyframes.uniform(B, T, v, dev)
        else:
            masks[(kind, v)] = torch.cat([keyframes.adaptive(root[s], lq[s], off, P, v, 32) for s in batches])

    for run in a.runs:
        cfg, m = load_model(run, dev)
        res = {"config": cfg, "gt": gt, "points": []}
        with torch.no_grad():
            if cfg["kind"] == "fr":
                def fn(s):
                    o = m(root[s], lq[s])
                    return o["root"], o["lq"], o["bits"]
                r, q, bits = run_batched(fn)
                res["points"].append({"placement": "fixed", **scores(r, q, gp, contacts, floor, off, P, themes, bits)})
                p = res["points"][-1]
                print(f"{run}: {p['bits_per_s']:.0f} b/s  MPJPE {p['mpjpe_cm']:.2f}  skate {p['skate_cm_s']:.1f}  jitter {p['jitter']:.0f}")
            else:
                for (kind, v), mask in masks.items():
                    dtb = DT_BITS if kind == "adaptive" else 0

                    def fn(s):
                        o = m(root[s], lq[s], mask[s])
                        keys = o["keys"]
                        b_ = torch.arange(keys.shape[0], device=dev)[:, None]
                        # exact (unquantized) keys through the same in-betweener
                        r2, q2 = m.inbetween(root[s][b_, keys], lq[s][b_, keys], keys, T)
                        r3, q3 = keyframes.interpolate_mask(root[s], lq[s], mask[s])
                        bits = o["n_keys"] * (m.bits_per_key + dtb)
                        return o["root"], o["lq"], bits, o["n_keys"], r2, q2, r3, q3
                    r, q, bits, nk, r2, q2, r3, q3 = run_batched(fn)
                    for variant, (rr, qq, bb) in (("quantized", (r, q, bits)), ("exact_keys", (r2, q2, bits * float("nan"))),
                                                  ("slerp", (r3, q3, bits * float("nan")))):
                        pt = {"placement": kind, "param": v, "variant": variant,
                              **scores(rr, qq, gp, contacts, floor, off, P, themes, bb, nk)}
                        res["points"].append(pt)
                    pq, pe = res["points"][-3], res["points"][-2]
                    print(f"{run} {kind:8s} {v:>4}: {pq['keys_per_s']:5.2f} keys/s {pq['bits_per_s']:6.0f} b/s  "
                          f"MPJPE q {pq['mpjpe_cm']:.2f} exact {pe['mpjpe_cm']:.2f}  skate {pq['skate_cm_s']:.1f}")
        json.dump(res, open(os.path.join(run, "eval_recon.json"), "w"))
    print("GT", gt)


if __name__ == "__main__":
    main()
