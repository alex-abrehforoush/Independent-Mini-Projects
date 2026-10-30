"""Train the autoregressive in-betweening generator over a frozen tokenizer.

  --tok runs/tok/kf_g4p   --continuous 0   keyframe tokens, FSQ codes
  --tok runs/tok/kf_none  --continuous 1   keyframe tokens, continuous key latents (flow head);
                                           --cont_space features: raw 136-d key features instead
  --tok runs/tok/fr_fsq_g4                 fixed-rate tokens, FSQ codes
  --tok runs/tok/fr_none  --continuous 1   fixed-rate continuous latents (flow head)
Gaps n ~ U{5..90}, one gap per batch."""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

from motion_tokens import data as D
from motion_tokens import gen_data as G
from motion_tokens import keyframes, lafan, placement
from motion_tokens.generator import MAX_GAP, N_CTX, Generator
from motion_tokens.quantizers import FSQ
from motion_tokens.tokenizers import mask_to_index

load_model = import_module("11_eval_tokenizer").load_model

ap = argparse.ArgumentParser()
ap.add_argument("--tok", required=True)
ap.add_argument("--continuous", type=int, default=0)
ap.add_argument("--eps", type=float, default=None, help="adaptive key eps (kf); default from tokenizer sweep")
ap.add_argument("--stride", type=int, default=0, help="uniform keys instead of adaptive (kf)")
ap.add_argument("--code_head", choices=("digits", "joint"), default="digits",
                help="FSQ codes: one softmax per dimension (digits) or one per group (joint)")
ap.add_argument("--cont_space", choices=("latent", "features"), default="latent",
                help="continuous keyframe tokens: the tokenizer's key latents, or raw key features (ablation)")
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--n_min", type=int, default=5)
ap.add_argument("--n_max", type=int, default=90)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.manual_seed(a.seed)
rng = random.Random(a.seed)
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
tcfg, tok = load_model(a.tok, dev)
for p in tok.parameters():
    p.requires_grad_(False)
kind = tcfg["kind"]
if kind == "kf" and (not a.continuous or a.cont_space == "latent"):
    tok.compile_key_step()  # keyframe targets come from the closed-loop key coder
d = lafan.load("data/lafan1.npz")
clips = D.Clips(d, lafan.TRAIN_ACTORS, dev)
eps_k = None
if kind == "kf" and not a.stride:
    clips.precompute_keys(placement.EPS_GRID, cache="data/keymasks_train.pt")
    eps = a.eps if a.eps is not None else 8.0
    eps_k = min(range(len(placement.EPS_GRID)), key=lambda k: abs(placement.EPS_GRID[k] - eps))
    a.eps = placement.EPS_GRID[eps_k]

cont_stats, groups, codebook, cont_dim = None, 1, 1, 1
if kind == "kf":
    if a.continuous and a.cont_space == "features":
        cont_dim = 136
    elif a.continuous:
        assert tok.q is None, "continuous keyframe generation needs an unquantized (kf_none) tokenizer"
        with torch.no_grad():
            zs = []
            for _ in range(8):
                root, lq, _, idx = clips.sample(256, 128)
                mask = keyframes.uniform(256, 128, a.stride, dev) if a.stride else clips.crop_mask(idx, eps_k)
                keys, valid = mask_to_index(mask)
                zs.append(tok.encode_keys(root, lq, keys)[0][valid])
            zs = torch.cat(zs)
        cont_stats = {"mean": zs.mean(0), "std": zs.std(0).clamp(min=1e-3)}
        cont_dim = zs.shape[-1]
    else:
        groups, codebook = tok.q.groups, tok.q.codebook_size
else:
    if a.continuous:
        assert tok.q is None, "continuous fixed-rate generation needs an unquantized (fr_none) tokenizer"
        with torch.no_grad():
            zs = torch.cat([tok.encode(*clips.sample(256, 128)[:2]).flatten(0, 1) for _ in range(8)])
        cont_stats = {"mean": zs.mean(0), "std": zs.std(0)}
        cont_dim = zs.shape[-1]
    else:
        groups, codebook = tok.q.groups, tok.q.codebook_size

levels = None
if not a.continuous and isinstance(tok.q, FSQ) and a.code_head == "digits":
    levels = [int(x) for x in tok.q.levels.tolist()]
else:
    a.code_head = "joint"
model = Generator(kind, bool(a.continuous), groups, codebook, cont_dim, a.d, a.layers, levels=levels).to(dev)
cfg = vars(a) | {"kind": kind, "params": sum(p.numel() for p in model.parameters()), "tok_config": tcfg}
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
if cont_stats is not None:
    torch.save(cont_stats, os.path.join(a.out, "cont_stats.pt"))
print({k: v for k, v in cfg.items() if k != "tok_config"})

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
log = open(os.path.join(a.out, "log.jsonl"), "w")
t0 = time.time()
for step in range(1, a.steps + 1):
    n = rng.randint(a.n_min, a.n_max)
    root, lq, gp, idx = clips.sample(a.bs, N_CTX + n + 1, anchor=N_CTX - 1)
    ctx, tgt, gap = G.conditioning(tok, root, lq, n)
    prefix = model.prefix(ctx, tgt, gap)
    crop_r, crop_q = root[:, N_CTX - 1:], lq[:, N_CTX - 1:]
    n_total = torch.full((a.bs,), n + 1, device=dev)
    parts = {}
    with torch.no_grad():
        if kind == "kf":
            if a.stride:
                mask = keyframes.uniform(a.bs, n + 2, a.stride, dev)
            else:
                mask = clips.crop_mask(idx[:, N_CTX - 1:], eps_k)
            t = G.kf_targets(tok, crop_r, crop_q, mask, bool(a.continuous), cont_stats)
        else:
            t = G.fr_targets(tok, crop_r, crop_q, bool(a.continuous), cont_stats)
    if kind == "kf":
        h_in = G.kf_step_inputs(model, t, n_total, root[:, -1], lq[:, -1])
        pad = ~t["valid"][:, 1:]  # position j is real iff key j+1 exists
        h = model.run(prefix, h_in, pad)
        real = ~pad
        # Δt of the next key
        next_dt = t["dt"][:, 1:]
        logits = model.dt_head(h).masked_fill(~model.dt_mask(t["keys"][:, :-1], n_total), -1e9)
        l_dt = Fn.cross_entropy(logits[real], next_dt[real])
        # pose of the next key, unless it is the (given) target
        nk = t["valid"].sum(1)
        j = torch.arange(h.shape[1], device=dev)[None]
        inner = j + 1 < (nk - 1)[:, None]
        c = model.pose_condition(h, next_dt)
        if a.continuous:
            l_pose = model.head.loss(t["tokens"][:, 1:][inner], c[inner])
        else:
            l_pose = model.code_loss(c[inner], t["tokens"][:, 1:][inner])
        loss = l_dt + l_pose
        parts = {"dt": l_dt.item(), "pose": l_pose.item(), "keys": nk.float().mean().item()}
    else:
        h_in = G.fr_step_inputs(model, t["tokens"], n_total)
        h = model.run(prefix, h_in)
        if a.continuous:
            loss = model.head.loss(t["tokens"].flatten(0, 1), h.flatten(0, 1))
        else:
            loss = model.code_loss(h.flatten(0, 1), t["tokens"].flatten(0, 1))
        parts = {"tokens": t["tokens"].shape[1]}
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 100 == 0 or step == 1:
        rec = {"step": step, "n": n, "loss": loss.item(), **parts, "s": time.time() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 1000 == 0 or step == 1:
            print(rec, flush=True)
    if step % 5000 == 0 or step == a.steps:
        torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
print("done", time.time() - t0)
