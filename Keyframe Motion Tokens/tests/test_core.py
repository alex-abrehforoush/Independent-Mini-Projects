import os
import tempfile

import numpy as np
import pytest
import torch

from motion_tokens import bvh, lafan
from motion_tokens import quat as Q

DATA = "data/lafan1.npz"


def rand_q(*shape):
    return Q.qnormalize(torch.randn(*shape, 4, dtype=torch.float64))


def same_rot(a, b, tol=1e-6):
    return torch.minimum((a - b).abs().amax(-1), (a + b).abs().amax(-1)).max() < tol


def test_matrix_and_6d_roundtrip():
    q = rand_q(1000)
    assert same_rot(Q.mat_to_quat(Q.quat_to_mat(q)), q)
    assert same_rot(Q.rot6d_to_quat(Q.quat_to_rot6d(q)), q)


def test_euler_roundtrip():
    q = rand_q(1000)
    for order in ("zyx", "xyz"):
        e = Q.mat_to_euler(Q.quat_to_mat(q), order)
        assert same_rot(Q.euler_to_quat(e, order), q)


def test_qrot_matches_matrix():
    q, v = rand_q(100), torch.randn(100, 3, dtype=torch.float64)
    assert torch.allclose(Q.qrot(q, v), (Q.quat_to_mat(q) @ v[..., None])[..., 0])


def test_fk_ik_roundtrip():
    parents = [-1, 0, 1, 2, 0, 4]
    lq, lp = rand_q(7, 6), torch.randn(7, 6, 3, dtype=torch.float64)
    gq, gp = Q.fk(lq, lp, parents)
    lq2, lp2 = Q.ik(gq, gp, parents)
    assert same_rot(lq2, lq) and torch.allclose(lp2, lp)


def test_remove_discontinuities():
    q = rand_q(50, 3)
    flips = torch.where(torch.rand(50, 3, 1) < 0.5, -1.0, 1.0).double()
    c = Q.remove_discontinuities(q * flips)
    assert ((c[1:] * c[:-1]).sum(-1) >= 0).all()


@pytest.mark.skipif(not os.path.exists(DATA), reason="needs data/lafan1.npz")
def test_bvh_write_read_roundtrip():
    d = lafan.load(DATA)
    s = d.sequences[0]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.bvh")
        bvh.write_bvh(p, d.skeleton, s.root_pos[:40], s.local_q[:40])
        b = bvh.read_bvh(p)
    assert np.allclose(b.root_pos, s.root_pos[:40], atol=1e-4)
    assert same_rot(torch.from_numpy(b.local_quats()), torch.from_numpy(s.local_q[:40]).double(), 1e-5)


def test_slerp_gradient_finite_for_identical_inputs():
    q = rand_q(5).requires_grad_(True)
    out = Q.slerp(q, q.detach().clone(), torch.full((5,), 0.3, dtype=torch.float64))
    out.sum().backward()
    assert torch.isfinite(q.grad).all()
