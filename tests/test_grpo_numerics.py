from __future__ import annotations

import pytest
import torch

from loop.trainer.grpo import compute_group_advantages, grpo_loss
from loop.trainer.weight_sync import assert_weights_synced, state_dict_sha256


def test_group_advantages_zero_variance_group_is_zero() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    adv = compute_group_advantages(rewards, group_size=4)
    assert torch.allclose(adv[:4], torch.zeros(4))
    assert adv[4] > 0
    assert torch.all(adv[5:] < 0)
    assert abs(float(adv[4:].mean().item())) < 1e-5


def test_observation_tokens_receive_strictly_zero_gradient() -> None:
    policy_lp = torch.tensor(
        [[-1.2, -0.8, -0.5, -0.4, -2.0], [-0.9, -1.1, -0.3, -0.7, -1.5]],
        requires_grad=True,
    )
    rollout_lp = policy_lp.detach() + 0.05
    # Positions 0, 1, and 4 are prompt/tool observations; 2 and 3 are assistant tokens.
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]])
    adv = torch.tensor([1.0, -1.0])

    loss, stats = grpo_loss(policy_lp, rollout_lp, adv, mask, use_mis=False)
    loss.backward()  # type: ignore[no-untyped-call]

    assert policy_lp.grad is not None
    assert torch.all(policy_lp.grad[:, [0, 1, 4]] == 0.0)
    assert torch.all(policy_lp.grad[:, [2, 3]] != 0.0)
    assert stats.active_tokens == 4
    assert stats.mean_abs_logprob_diff == pytest.approx(0.05, abs=1e-4)


def test_mis_masks_extreme_train_inference_mismatch_spikes() -> None:
    # Token 2 has huge TIM drift: log_ratio = +3.0 => ratio = exp(3) ~= 20.08 > mis_max_ratio=2.0.
    policy_lp = torch.tensor([[-0.5, -0.5, -0.1], [-0.5, -0.5, -0.5]], requires_grad=True)
    rollout_lp = torch.tensor([[-0.5, -0.5, -3.1], [-0.5, -0.5, -0.5]])
    mask = torch.ones_like(policy_lp)
    adv = torch.tensor([1.0, -1.0])

    loss_mis, stats_mis = grpo_loss(
        policy_lp, rollout_lp, adv, mask, use_mis=True, mis_max_ratio=2.0
    )
    loss_mis.backward()  # type: ignore[no-untyped-call]
    assert policy_lp.grad is not None
    # The spiked TIM token [0, 2] must be completely masked out by MIS.
    assert float(policy_lp.grad[0, 2].item()) == 0.0
    assert stats_mis.high_ratio_fraction == pytest.approx(1.0 / 6.0, abs=1e-5)
    assert stats_mis.masked_token_fraction == pytest.approx(1.0 / 6.0, abs=1e-5)


def test_weight_digest_detects_single_bit_staleness() -> None:
    w1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    w2 = w1.clone()
    d1 = state_dict_sha256({"layer.weight": w1})
    d2 = state_dict_sha256({"layer.weight": w2})
    assert_weights_synced(d1, d2, policy_version=1)

    w2[0] = torch.nextafter(w2[0], torch.tensor(float("inf"), dtype=torch.bfloat16))
    d3 = state_dict_sha256({"layer.weight": w2})
    with pytest.raises(RuntimeError, match="Weight staleness detected"):
        assert_weights_synced(d1, d3, policy_version=2)
