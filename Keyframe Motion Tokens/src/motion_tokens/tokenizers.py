"""The two tokenizer families compared in this project.

FixedRateTokenizer  — T2M-GPT/MoMask-style: dense velocity features -> conv x4 downsampling ->
                      transformer -> quantizer -> transformer -> x4 upsampling -> features.
                      One token per 4 frames (7.5 tokens/s at 30 fps), whatever the motion.
KeyframeTokenizer   — a token is a key pose (context-free MLP -> FSQ) plus its Δt; a learned
                      in-betweener (SLERP/LERP through the keys + transformer residual that is
                      zero at the keys) reconstructs the dense clip. Token rate follows the key
                      placement, so one model serves any rate.
Both decode to the same (root, local quaternion) output and are trained with the same loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import features as F
from . import quat as Q
from .keyframes import neighbours
from .quantizers import FSQ


def sinusoid(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[..., None] * freq
    return torch.cat([a.sin(), a.cos()], -1)


def transformer(d: int, layers: int, heads: int = 4, ff: int = 4, dropout: float = 0.0):
    layer = nn.TransformerEncoderLayer(d, heads, ff * d, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def mlp(i: int, h: int, o: int, n: int = 2):
    mods, d = [], i
    for _ in range(n):
        mods += [nn.Linear(d, h), nn.GELU()]
        d = h
    return nn.Sequential(*mods, nn.Linear(d, o))


class Normalizer(nn.Module):
    def __init__(self, stats: dict):
        super().__init__()
        self.register_buffer("mean", stats["mean"].float().clone())
        self.register_buffer("std", stats["std"].float().clone())

    def norm(self, f):
        return (f - self.mean) / self.std

    def denorm(self, f):
        return f * self.std + self.mean


# ----------------------------------------------------------------------------- fixed rate

class FixedRateTokenizer(nn.Module):
    """abs_traj: how the root trajectory is decoded.
    0: integrated from decoded velocities, as in HumanML3D-style features; quantization errors
       accumulate (~12 cm root error over 4 s at 300 b/s).
    1: the window-frame trajectory (x, z, unwrapped yaw) is also coded and decoded directly; a
       bounded token cannot hold an absolute position precisely over a long window.
    2: integrated, plus a learned correction (zero at init) decoded from the same coded channels."""
    down = 4
    traj_scale = (200.0, 200.0, math.pi)  # cm, cm, rad

    def __init__(self, stats: dict, quantizer: nn.Module | None, latent_dim: int, d: int = 256,
                 layers: int = 4, abs_traj: int = 0):
        super().__init__()
        D = F.D_REL + (3 if abs_traj else 0)
        self.nrm = Normalizer(stats)
        self.q = quantizer
        self.abs_traj = abs_traj
        self.latent_dim = latent_dim
        self.enc_in = nn.Linear(D, d)
        self.enc_conv = nn.Sequential(nn.Conv1d(d, d, 4, 2, 1), nn.GELU(), nn.Conv1d(d, d, 4, 2, 1))
        self.enc = transformer(d, layers)
        self.enc_out = nn.Linear(d, latent_dim)
        self.dec_in = nn.Linear(latent_dim, d)
        self.dec = transformer(d, layers)
        self.dec_up = nn.Sequential(nn.ConvTranspose1d(d, d, 4, 2, 1), nn.GELU(), nn.ConvTranspose1d(d, d, 4, 2, 1))
        self.dec2 = transformer(d, 2)
        self.dec_out = nn.Linear(d, D)
        if abs_traj == 2:
            nn.init.zeros_(self.dec_out.weight[F.D_REL:])
            nn.init.zeros_(self.dec_out.bias[F.D_REL:])
        self.d = d

    @property
    def bits_per_token(self) -> float:
        return self.q.bits if self.q is not None else float("nan")

    def encode(self, root, lq):
        f = F.relative_form(root, lq)
        x = self.nrm.norm(f)
        if self.abs_traj:
            x = torch.cat([x, F.trajectory(root, f) / x.new_tensor(self.traj_scale)], -1)
        h = self.enc_in(x).transpose(1, 2)
        h = self.enc_conv(h).transpose(1, 2)
        h = h + sinusoid(torch.arange(h.shape[1], device=h.device), self.d)
        return self.enc_out(self.enc(h))

    def quantize(self, z):
        if self.q is None:
            return z, None, z.new_zeros(())
        return self.q(z)

    def decode(self, zq, T: int):
        h = self.dec_in(zq)
        h = h + sinusoid(torch.arange(h.shape[1], device=h.device), self.d)
        h = self.dec(h)
        h = self.dec_up(h.transpose(1, 2)).transpose(1, 2)[:, :T]
        h = h + sinusoid(torch.arange(h.shape[1], device=h.device), self.d)
        out = self.dec_out(self.dec2(h))
        f = self.nrm.denorm(out[..., :F.D_REL])
        if not self.abs_traj:
            return F.absolute_from_relative(f)
        traj = out[..., F.D_REL:] * out.new_tensor(self.traj_scale)
        if self.abs_traj == 2:
            traj = traj + F.trajectory(F.absolute_from_relative(f)[0], f)
        return F.absolute_from_relative(f, traj=traj)

    def forward(self, root, lq):
        T = root.shape[1]
        assert T % self.down == 0
        z = self.encode(root, lq)
        zq, idx, aux = self.quantize(z)
        r, q = self.decode(zq, T)
        n_tokens = torch.full((root.shape[0],), z.shape[1], device=root.device)
        return {"root": r, "lq": q, "aux": aux, "idx": idx, "bits": n_tokens * self.bits_per_token}


# ----------------------------------------------------------------------------- keyframes

def mask_to_index(mask: torch.Tensor):
    """(B,T) bool -> padded increasing key indices (B,Kmax) (padding repeats T-1) and valid (B,Kmax)."""
    B, T = mask.shape
    t = torch.arange(T, device=mask.device).expand(B, T)
    order = torch.where(mask, t, t + T).sort(1).values
    K = int(mask.sum(1).max())
    idx = order[:, :K]
    valid = idx < T
    return torch.where(valid, idx, torch.full_like(idx, T - 1)), valid


class KeyframeTokenizer(nn.Module):
    max_gap = 32

    def __init__(self, stats: dict, groups: int = 4, levels=(8, 5, 5, 5), d: int = 256, layers: int = 4,
                 quantize: bool = True, predictive: bool = False):
        super().__init__()
        D = F.D_REL
        self.nrm = Normalizer(stats)
        self.q = FSQ(levels, groups) if quantize else None
        self.predictive = predictive
        self._compiled_step = None
        qdim = self.q.dim if quantize else 32
        ctx = D if predictive else 0
        self.key_enc = mlp(D + ctx + 32, 512, qdim, 3)
        self.key_dec = mlp(qdim + ctx + 32, 512, D, 3)
        self.dt_emb = nn.Embedding(self.max_gap + 1, 32)
        self.ref_in = nn.Linear(D + 5, d)
        self.ref = transformer(d, layers)
        self.ref_out = nn.Linear(d, D)
        nn.init.zeros_(self.ref_out.weight)
        nn.init.zeros_(self.ref_out.bias)
        self.d = d
        # residual scales: root height / xz offsets in cm, yaw in rad, 6D components
        scale = torch.cat([torch.tensor([5.0, 5.0, 5.0, 0.1]), torch.full((D - 4,), 0.1)])
        self.register_buffer("res_scale", scale)

    @property
    def bits_per_key(self) -> float:
        return self.q.bits if self.q is not None else float("nan")

    # key content <-> per-key features. Δxz and Δyaw are divided by Δt (velocity-like scale).
    def _key_features(self, root, lq, keys):
        f = F.relative_form(root, lq, keys)
        dt = torch.cat([torch.zeros_like(keys[:, :1]), keys[:, 1:] - keys[:, :-1]], 1)
        f = f.clone()
        f[..., 1:4] = f[..., 1:4] / dt.clamp(min=1)[..., None]
        return f, dt.clamp(max=self.max_gap)

    def _absolute_keys(self, f, dt):
        f = f.clone()
        f[..., 1:4] = f[..., 1:4] * dt.clamp(min=1)[..., None]
        f[..., 1:4] = torch.where(dt[..., None] == 0, torch.zeros_like(f[..., 1:4]), f[..., 1:4])
        return F.absolute_from_relative(f)

    # one key <-> latent. Context-free: a key is coded on its own (an I-frame), so a token is an
    # editable pose. Predictive: encoder and decoder also see the previous *decoded* key and the
    # decoder outputs a residual on it (a P-frame), so codes spend their bits on what changed.
    def code_key(self, fn, emb, prev_fn):
        """fn, prev_fn: normalised key features (..., 136); emb: Δt embedding. -> zq, code, aux."""
        x = [fn, prev_fn, emb] if self.predictive else [fn, emb]
        return self.quantize(self.key_enc(torch.cat(x, -1)))

    def dec_key(self, zq, emb, prev_fn):
        """-> normalised key features (..., 136)."""
        if not self.predictive:
            return self.key_dec(torch.cat([zq, emb], -1))
        return prev_fn + self.key_dec(torch.cat([zq, prev_fn, emb], -1))

    def encode_keys(self, root, lq, keys):
        """Closed-loop (DPCM) key coding. Each key's move — Δxz and Δyaw per frame of Δt — is
        measured from the *decoded* previous key, so quantization errors do not accumulate along
        the trajectory; height and rotations are absolute per key (or, predictive, a residual on
        the decoded previous key). An open-loop variant (moves measured between ground-truth keys)
        drifted ~15 cm over 4 s at every key rate.

        Returns latents zq (B,K,qdim), codes, aux loss, decoded features (B,K,136, un-normalised,
        moves per frame) and Δt (B,K)."""
        f, dt = self._key_features(root, lq, keys)
        B, K = keys.shape
        b = torch.arange(B, device=keys.device)[:, None]
        k_root = root[b, keys]
        k_yaw = F.heading_yaw(lq[b, keys, 0])
        state = (torch.zeros_like(k_root[:, 0]), torch.zeros_like(k_yaw[:, 0]), torch.zeros_like(f[:, 0]))
        zs, codes, decs, aux = [], [], [], f.new_zeros(())
        # key-major copies: per-key slices then have the same strides whatever K is (no recompiles)
        per_key = [x.transpose(0, 1).contiguous() for x in (f, self.dt_emb(dt), k_root, k_yaw, dt)]
        for i in range(K):
            zq, code, a, fd, state = self.key_step(*(x[i] for x in per_key), state)
            zs.append(zq)
            codes.append(code)
            decs.append(fd)
            aux = aux + a
        codes = torch.stack(codes, 1) if codes[0] is not None else None
        return torch.stack(zs, 1), codes, aux / K, torch.stack(decs, 1), dt

    def _key_step(self, f, emb, k_root, k_yaw, dt, state):
        """One closed-loop coding step. state = decoded previous key: root position, heading yaw
        and normalised features. Training swaps in a compiled copy (compile_key_step): the loop is
        launch-bound, ~100 tiny kernels per key."""
        prev_pos, prev_yaw, prev_fn = state
        dti = dt.clamp(min=1).to(f.dtype)
        moving = (dt > 0).to(f.dtype)
        d = Q.qrot(Q.qinv(Q.yaw_to_quat(prev_yaw)), k_root - prev_pos)
        traj = torch.stack([d[:, 0], d[:, 2], F.wrap(k_yaw - prev_yaw)], -1) / dti[:, None] * moving[:, None]
        fi = torch.cat([f[:, :1], traj, f[:, 4:]], -1)
        zq, code, a = self.code_key(self.nrm.norm(fi), emb, prev_fn)
        prev_fn = self.dec_key(zq, emb, prev_fn)
        fd = self.nrm.denorm(prev_fn)
        step = fd[:, 1:4] * (dti * moving)[:, None]
        prev_pos = prev_pos + Q.qrot(Q.yaw_to_quat(prev_yaw), torch.stack([step[:, 0], torch.zeros_like(step[:, 0]), step[:, 1]], -1))
        prev_yaw = prev_yaw + step[:, 2]
        return zq, code, a, fd, (prev_pos, prev_yaw, prev_fn)

    def key_step(self, *args):
        return (self._compiled_step or self._key_step)(*args)

    def compile_key_step(self):
        self._compiled_step = torch.compile(self._key_step, dynamic=False)

    def decode_keys(self, zq, dt):
        emb = self.dt_emb(dt)
        prev_fn = zq.new_zeros(*zq.shape[:-2], F.D_REL)
        fns = []
        for i in range(zq.shape[-2]):
            prev_fn = self.dec_key(zq[..., i, :], emb[..., i, :], prev_fn)
            fns.append(prev_fn)
        return self._absolute_keys(self.nrm.denorm(torch.stack(fns, -2)), dt)

    def quantize(self, z):
        if self.q is None:
            return z, None, z.new_zeros(())
        return self.q(z)

    def inbetween(self, key_root, key_lq, keys, T: int):
        """Dense clip from absolute key poses at frame indices keys (B,K)."""
        r, q = F.interpolate_keys(key_root, key_lq, keys, T)
        mask = torch.zeros(keys.shape[0], T, dtype=torch.bool, device=keys.device)
        mask.scatter_(1, keys, True)
        prv, nxt = neighbours(mask)
        t = torch.arange(T, device=keys.device).expand_as(prv)
        seg = (nxt - prv).float()
        phase = torch.where(mask, torch.zeros_like(seg), (t - prv).float() / seg.clamp(min=1))
        f = self.nrm.norm(F.relative_form(r, q))
        extra = torch.stack([mask.float(), phase, seg / self.max_gap, (t - prv).float() / self.max_gap,
                             (nxt - t).float() / self.max_gap], -1)
        h = self.ref_in(torch.cat([f, extra], -1)) + sinusoid(t, self.d)
        res = self.ref_out(self.ref(h)) * self.res_scale * (~mask)[..., None]
        return self._apply_residual(r, q, res)

    @staticmethod
    def _apply_residual(r, q, res):
        hq = Q.heading_quat(q[..., 0, :])
        yaw = Q.yaw_angle(hq)
        rel_root = Q.qmul(Q.qinv(hq), q[..., 0, :])
        rots = Q.quat_to_rot6d(torch.cat([rel_root[..., None, :], q[..., 1:, :]], -2))
        rots = rots + res[..., 4:].unflatten(-1, (rots.shape[-2], 6))
        new_rot = Q.rot6d_to_quat(rots)
        new_yaw = yaw + res[..., 3]
        dxz = torch.stack([res[..., 1], torch.zeros_like(res[..., 1]), res[..., 2]], -1)
        root = r + Q.qrot(Q.yaw_to_quat(yaw), dxz)
        root = root + torch.stack([torch.zeros_like(res[..., 0]), res[..., 0], torch.zeros_like(res[..., 0])], -1)
        lq = new_rot.clone()
        lq[..., 0, :] = Q.qmul(Q.yaw_to_quat(new_yaw), new_rot[..., 0, :])
        return root, lq

    def forward(self, root, lq, mask, exact: bool = False):
        """exact=True skips coding and in-betweens the ground-truth key poses (the constraint path)."""
        T = root.shape[1]
        keys, valid = mask_to_index(mask)
        if exact:
            b = torch.arange(keys.shape[0], device=keys.device)[:, None]
            kr, kq, idx, aux = root[b, keys], lq[b, keys], None, root.new_zeros(())
        else:
            _, idx, aux, fd, dt = self.encode_keys(root, lq, keys)
            kr, kq = self._absolute_keys(fd, dt)
        r, q = self.inbetween(kr, kq, keys, T)
        n_keys = valid.sum(1)
        return {"root": r, "lq": q, "aux": aux, "idx": idx, "keys": keys, "valid": valid,
                "key_root": kr, "key_lq": kq, "n_keys": n_keys, "bits_pose": n_keys * self.bits_per_key}


# ----------------------------------------------------------------------------- shared loss

def recon_loss(pred_root, pred_lq, root, lq, gp, clips, nrm: Normalizer, w_pos: float = 1.0, w_vel: float = 1.0,
               w_acc: float = 1.0):
    """L1 on normalised dense features + global joint positions, velocities and accelerations
    (decimetres). The acceleration term keeps decoders from adding high-frequency jitter."""
    fp = nrm.norm(F.relative_form(pred_root, pred_lq))
    ft = nrm.norm(F.relative_form(root, lq))
    gp_p = clips.global_positions(pred_root, pred_lq) / 10.0
    gp_t = gp / 10.0
    l_feat = (fp - ft).abs().mean()
    l_pos = (gp_p - gp_t).abs().mean()
    vp, vt = gp_p[:, 1:] - gp_p[:, :-1], gp_t[:, 1:] - gp_t[:, :-1]
    l_vel = (vp - vt).abs().mean() * 10.0
    l_acc = ((vp[:, 1:] - vp[:, :-1]) - (vt[:, 1:] - vt[:, :-1])).abs().mean() * 30.0
    return (l_feat + w_pos * l_pos + w_vel * l_vel + w_acc * l_acc,
            {"feat": l_feat.item(), "pos": l_pos.item(), "vel": l_vel.item(), "acc": l_acc.item()})
