import torch

from sensor_vlm.rl import group_advantages, policy_loss


def test_group_advantages_center_scale_and_skip_flat_groups():
    r = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    adv, informative = group_advantages(r)
    assert informative.tolist() == [True, False]
    assert torch.allclose(adv[0], torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=1e-3)
    assert torch.all(adv[1] == 0)
    adv_none, _ = group_advantages(r, scale="none")
    assert torch.allclose(adv_none[0], torch.tensor([0.5, -0.5, 0.5, -0.5]))


def test_on_policy_gradient_is_advantage_weighted_logprob():
    logp = torch.tensor([[-1.0, -2.0], [-0.5, -0.1]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    adv = torch.tensor([2.0, -1.0])
    loss, stats = policy_loss(logp, logp.detach(), adv, mask)
    loss.backward()
    # d loss / d logp = -adv / length for each unmasked token
    expected = torch.tensor([[-1.0, -1.0], [1.0, 0.0]])
    assert torch.allclose(logp.grad, expected) and stats["clip_frac"] == 0


def test_clipping_stops_gradient_beyond_trust_region():
    old = torch.zeros(1, 1)
    logp = torch.full((1, 1), 0.5, requires_grad=True)  # ratio e^0.5 > 1.2
    loss, stats = policy_loss(logp, old, torch.tensor([1.0]), torch.ones(1, 1))
    loss.backward()
    assert logp.grad.abs().item() == 0 and stats["clip_frac"] == 1


def test_kl_penalty_zero_at_reference():
    logp = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    loss, stats = policy_loss(logp, logp.detach(), torch.tensor([0.0]), torch.ones(1, 2),
                              ref_logp=logp.detach(), beta=0.1)
    assert stats["kl"] == 0 and loss.item() == 0


def test_dpo_loss_at_reference_and_gradient_direction():
    from sensor_vlm.rl import dpo_loss
    pc = torch.tensor([-5.0], requires_grad=True)
    pr = torch.tensor([-4.0], requires_grad=True)
    loss, stats = dpo_loss(pc, pr, pc.detach(), pr.detach(), beta=0.1)
    assert abs(loss.item() - torch.log(torch.tensor(2.0)).item()) < 1e-6 and stats["margin"] == 0
    loss.backward()
    # raise the chosen sequence's log-prob, lower the rejected one's
    assert pc.grad.item() < 0 and pr.grad.item() > 0
