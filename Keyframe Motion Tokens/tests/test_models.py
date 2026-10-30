import os
import sys

import pytest
import torch
import torch.nn.functional as Fn

from motion_tokens import data as D
from motion_tokens import features as F
from motion_tokens import gen_data as G
from motion_tokens import keyframes, lafan, metrics, quantizers, tokenizers
from motion_tokens import quat as Q
from motion_tokens.generator import N_CTX, Generator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
DATA = "data/lafan1.npz"
pytestmark = pytest.mark.skipif(not os.path.exists(DATA), reason="needs data/lafan1.npz")


@pytest.fixture(scope="module")
def clips():
    torch.manual_seed(0)
    return D.Clips(lafan.load(DATA), lafan.TEST_ACTORS, "cpu")


def same_rot(a, b, tol):
    return torch.minimum((a - b).abs().amax(-1), (a + b).abs().amax(-1)).max() < tol


def test_relative_form_roundtrip_dense_and_sparse(clips):
    root, lq, gp, idx = clips.sample(4, 64)
    r2, q2 = F.absolute_from_relative(F.relative_form(root, lq))
    assert (r2 - root).abs().max() < 1e-2 and same_rot(q2, lq, 1e-4)
    keys = torch.tensor([[0, 5, 9, 30, 63]] * 4)
    b = torch.arange(4)[:, None]
    r3, q3 = F.absolute_from_relative(F.relative_form(root, lq, keys))
    assert (r3 - root[b, keys]).abs().max() < 1e-2 and same_rot(q3, lq[b, keys], 1e-4)
    f = F.relative_form(root, lq)  # the absolute-trajectory path reproduces the same clip
    r4, q4 = F.absolute_from_relative(f, traj=F.trajectory(root, f))
    assert (r4 - root).abs().max() < 1e-2 and same_rot(q4, lq, 1e-4)


def test_sample_gp_matches_fk(clips):
    root, lq, gp, idx = clips.sample(4, 32, anchor=7)
    assert (clips.global_positions(root, lq) - gp).abs().max() < 1e-2
    assert root[:, 7, [0, 2]].abs().max() < 1e-3


def test_adaptive_keys_meet_tolerance(clips):
    root, lq, _, _ = clips.sample(8, 128)
    mask = keyframes.adaptive(root, lq, clips.offsets, clips.parents, 5.0, 32)
    err = keyframes.frame_errors(root, lq, mask, clips.offsets, clips.parents)
    assert err.max() <= 5.0 + 1e-3
    gaps = [torch.diff(torch.nonzero(m)[:, 0]).max().item() for m in mask]
    assert max(gaps) <= 32


def test_keyframe_tokenizer_hits_exact_keys(clips):
    stats = D.feature_stats(clips, n=64, T=64)
    tok = tokenizers.KeyframeTokenizer(stats, 2, d=32, layers=1)
    with torch.no_grad():  # make the refiner do something non-trivial
        tok.ref_out.weight.normal_(0, 0.1)
    root, lq, _, _ = clips.sample(3, 64)
    mask = keyframes.uniform(3, 64, 8)
    keys, _ = tokenizers.mask_to_index(mask)
    b = torch.arange(3)[:, None]
    r, q = tok.inbetween(root[b, keys], lq[b, keys], keys, 64)
    assert (r[b, keys] - root[b, keys]).abs().max() < 1e-3
    assert same_rot(q[b, keys], lq[b, keys], 1e-4)


@pytest.mark.parametrize("predictive", [False, True])
def test_key_codes_decode_like_the_encoder(clips, predictive):
    """Decoding codes alone (the generation path) reproduces the encoder's closed-loop keys."""
    stats = D.feature_stats(clips, n=64, T=64)
    tok = tokenizers.KeyframeTokenizer(stats, 2, d=32, layers=1, predictive=predictive).eval()
    root, lq, _, _ = clips.sample(3, 64)
    keys, _ = tokenizers.mask_to_index(keyframes.uniform(3, 64, 8))
    with torch.no_grad():
        _, codes, _, fd, dt = tok.encode_keys(root, lq, keys)
        r1, q1 = tok._absolute_keys(fd, dt)
        r2, q2 = tok.decode_keys(tok.q.decode_indices(codes), dt)
    assert (r1 - r2).abs().max() < 1e-3 and same_rot(q1, q2, 1e-4)


def test_fsq_index_roundtrip():
    q = quantizers.FSQ((8, 5, 5, 5), 3)
    v, idx, _ = q(torch.randn(10, q.dim) * 3)
    assert torch.allclose(q.decode_indices(idx), v)
    assert idx.max() < 1000


def test_per_dimension_code_head():
    """Digits follow FSQ.decode_indices; a head peaked on a code's levels returns that code; the
    per-group NLL of a perfectly peaked head is ~0."""
    q = quantizers.FSQ((8, 5, 5, 5), 3)
    m = Generator("fr", False, 3, 1000, 1, d=16, layers=1, levels=(8, 5, 5, 5))
    codes = torch.randint(0, 1000, (7, 3))
    hw = torch.tensor([4, 2, 2, 2])
    assert torch.equal(((m.digits(codes) - hw) / hw).flatten(-2), q.decode_indices(codes))
    onehot = Fn.one_hot(m.digits(codes), m.lmax).float() * 50
    with torch.no_grad():
        m.code_head.weight.zero_()
        m.code_head.bias.zero_()
    m.code_logits = lambda h: onehot.masked_fill(torch.arange(m.lmax) >= m.lv[:, None], -1e9)
    assert torch.equal(m.sample_codes(None, 0.0), codes) and torch.equal(m.sample_codes(None, 1.0), codes)
    assert m.code_loss(None, codes) < 1e-3


def test_generators_run_end_to_end(clips):
    ev = __import__("21_eval_generator")
    stats = D.feature_stats(clips, n=64, T=64)
    n = 23
    root, lq, _, _ = clips.sample(2, N_CTX + n + 1, anchor=N_CTX - 1)
    for pred in (False, True):
        kf = tokenizers.KeyframeTokenizer(stats, 2, d=32, layers=1, predictive=pred).eval()
        for cont, levels, temp in ((False, None, 1.0), (False, (8, 5, 5, 5), 1.0), (False, (8, 5, 5, 5), 0.0),
                                   (True, None, 1.0)):
            m = Generator("kf", cont, 2, 1000, 136, d=32, layers=1, levels=levels).eval()
            r, q, k = ev.generate_kf(m, kf, root, lq, n, temperature=temp)
            assert r.shape == (2, n, 3) and q.shape == (2, n, 22, 4) and torch.isfinite(r).all()
    # continuous keyframe tokens = the unquantized tokenizer's key latents, normalised
    kf = tokenizers.KeyframeTokenizer(stats, 2, d=32, layers=1, quantize=False).eval()
    cs = {"mean": torch.zeros(32), "std": torch.ones(32)}
    with torch.no_grad():
        t = G.kf_targets(kf, root[:, N_CTX - 1:], lq[:, N_CTX - 1:], keyframes.uniform(2, n + 2, 4), True, cs)
    assert t["tokens"].shape == (2, t["keys"].shape[1], 32)
    for temp in (0.0, 1.0):
        m = Generator("kf", True, 1, 1, 32, d=32, layers=1).eval()
        r, q, k = ev.generate_kf(m, kf, root, lq, n, temperature=temp, cont_stats=cs)
        assert r.shape == (2, n, 3) and torch.isfinite(r).all()
    fsq = quantizers.FSQ((8, 5, 5, 5), 2)
    for abs_traj, levels in ((0, None), (1, (8, 5, 5, 5)), (2, (8, 5, 5, 5))):
        fr = tokenizers.FixedRateTokenizer(stats, fsq, fsq.dim, d=32, layers=1, abs_traj=abs_traj).eval()
        m = Generator("fr", False, 2, 1000, 1, d=32, layers=1, levels=levels).eval()
        r, q, k = ev.generate_fr(m, fr, root, lq, n, temperature=1.0)
        assert r.shape == (2, n, 3) and torch.isfinite(r).all()
        if abs_traj == 2:  # the correction starts at zero: same clip as integrating velocities
            rr, _ = G.pad_to_multiple(root, lq)
            zq = fr.quantize(fr.encode(rr, G.pad_to_multiple(root, lq)[1]))[0]
            r2, _ = fr.decode(zq, rr.shape[1])
            fr.abs_traj = 0
            r0, _ = fr.decode(zq, rr.shape[1])
            fr.abs_traj = 2
            assert (r2 - r0).abs().max() < 1e-3
    # endpoint blending makes the decoded crop meet both boundary frames exactly
    rr, qq = G.blend_endpoints(r, q, root[:, 0], lq[:, 0], root[:, 1], lq[:, 1])
    assert (rr[:, 0] - root[:, 0]).abs().max() < 1e-3 and (rr[:, -1] - root[:, 1]).abs().max() < 1e-3


def test_own_contact_skate():
    """Only frames where the foot itself is down count: a planted foot scores 0, a foot sliding on
    the floor scores its speed, a foot moving in the air is ignored."""
    T, J = 10, 22
    gp = torch.zeros(1, T, J, 3)
    gp[..., 1] = 50.0
    t = torch.arange(T, dtype=torch.float)
    left, right = list(lafan.FEET[:2]), list(lafan.FEET[2:])
    gp[0, :, left, 1] = 0.0                    # left foot on the floor...
    gp[0, :, left, 0] = (t * 2.0)[:, None]     # ...sliding 2 cm/frame = 60 cm/s
    gp[0, :, right, 0] = (t * 5.0)[:, None]    # right foot moving, but 50 cm up
    low = torch.zeros(1, 4)
    assert metrics.foot_skate_own(gp, low).item() == pytest.approx(60.0)
    gp[0, :, left, 0] = 0.0
    assert metrics.foot_skate_own(gp, low).item() == pytest.approx(0.0)
