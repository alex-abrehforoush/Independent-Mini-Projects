"""Train one tokenizer.  fr = fixed-rate (fsq|vq|rvq), kf = keyframe (FSQ per key, rate-agnostic)."""
import argparse
import json
import os
import random
import time

import torch

from motion_tokens import data as D
from motion_tokens import lafan, placement, quantizers, tokenizers

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["fr", "kf"], required=True)
ap.add_argument("--quant", default="fsq", choices=["fsq", "vq", "rvq", "none"])
ap.add_argument("--groups", type=int, default=2, help="FSQ/VQ groups, or RVQ stages")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--T", default="32,64,96,128", help="crop lengths, one drawn per step")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--d", type=int, default=192)
ap.add_argument("--layers", type=int, default=None, help="default: 3 (fr enc/dec) or 4 (kf refiner)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--abs_traj", type=int, default=0, choices=[0, 1, 2],
                help="fr trajectory: 0 integrated velocities, 1 absolute, 2 integrated + learned correction")
ap.add_argument("--predictive", type=int, default=0, help="kf: code each key relative to the previous decoded key")
ap.add_argument("--p_exact", type=float, default=0.25, help="kf: fraction of batches with exact keys")
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="data/lafan1.npz")
a = ap.parse_args()
if a.layers is None:
    a.layers = 3 if a.kind == "fr" else 4
T_choices = [int(x) for x in a.T.split(",")]

torch.manual_seed(a.seed)
rng = random.Random(a.seed)
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
d = lafan.load(a.data)
clips = D.Clips(d, lafan.TRAIN_ACTORS, dev)
stats = D.feature_stats(clips)
if a.kind == "kf":
    t0 = time.time()
    clips.precompute_keys(placement.EPS_GRID, cache="data/keymasks_train.pt")
    print(f"precomputed adaptive keys for eps {placement.EPS_GRID} in {time.time() - t0:.0f}s")
torch.save(stats, os.path.join(a.out, "stats.pt"))

if a.kind == "fr":
    if a.quant == "fsq":
        q = quantizers.FSQ((8, 5, 5, 5), a.groups)
        latent = q.dim
    elif a.quant == "none":
        q, latent = None, 16
    else:
        latent = 64 if a.quant == "rvq" else 32 * a.groups
        q = quantizers.make(a.quant, latent, a.groups)
    model = tokenizers.FixedRateTokenizer(stats, q, latent, a.d, a.layers, abs_traj=a.abs_traj)
else:
    model = tokenizers.KeyframeTokenizer(stats, a.groups, d=a.d, layers=a.layers, quantize=a.quant != "none",
                                         predictive=bool(a.predictive))
model.to(dev)
if a.kind == "kf":
    model.compile_key_step()  # the per-key coding loop is launch-bound; ~1.5x faster compiled
cfg = vars(a) | {"params": sum(p.numel() for p in model.parameters())}
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
print(cfg)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
log = open(os.path.join(a.out, "log.jsonl"), "w")
t0 = time.time()
n_bad = 0
for step in range(1, a.steps + 1):
    root, lq, gp, idx = clips.sample(a.bs, rng.choice(T_choices))
    if a.kind == "fr":
        out = model(root, lq)
    else:
        mask, mode = placement.sample_mask(clips, idx, rng)
        # a quarter of the batches in-between exact keys: the path user constraints take
        out = model(root, lq, mask, exact=a.quant != "none" and rng.random() < a.p_exact)
    loss, parts = tokenizers.recon_loss(out["root"], out["lq"], root, lq, gp, clips, model.nrm)
    loss = loss + out["aux"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gn):  # skip the update, but give up if it keeps happening
        n_bad += 1
        if n_bad > 50:
            raise RuntimeError("too many non-finite gradient steps")
        sched.step()
        continue
    opt.step()
    sched.step()
    if step % 100 == 0 or step == 1:
        rec = {"step": step, "loss": loss.item(), **parts, "aux": float(out["aux"]), "s": time.time() - t0}
        if out.get("idx") is not None:
            rec["codes_used"] = int(out["idx"].unique().numel())
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 1000 == 0 or step == 1:
            print(rec, flush=True)
    if step % 5000 == 0 or step == a.steps:
        torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
print("done", time.time() - t0)
