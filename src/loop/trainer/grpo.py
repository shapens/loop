from __future__ import annotations

import torch

from loop.types import TIMStats


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute GRPO group-relative normalized advantages across groups of size G.

    Args:
        rewards: 1D tensor of shape [B] where B is a multiple of `group_size`.
        group_size: Number of rollouts sampled per prompt (G).
        eps: Numerical stabilizer; groups with zero variance receive 0 advantage.
    """
    if rewards.ndim != 1:
        raise ValueError(f"Expected 1D rewards tensor, got shape {tuple(rewards.shape)}")
    if rewards.numel() % group_size != 0:
        raise ValueError(f"Batch size {rewards.numel()} not divisible by group_size={group_size}")

    grouped = rewards.float().view(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=False, keepdim=True)
    adv = torch.where(std > eps, (grouped - mean) / (std + eps), torch.zeros_like(grouped))
    return adv.view(-1)


def grpo_loss(
    policy_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    use_mis: bool = True,
    mis_max_ratio: float = 2.0,
    kl_coef: float = 0.0,
    ref_logprobs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TIMStats]:
    """From-scratch multi-turn masked GRPO loss with TIM telemetry and MIS ablation switch.

    Shapes:
        policy_logprobs:  [B, T] log pi_theta(a_t | s_t) from the PyTorch learner forward pass.
        rollout_logprobs: [B, T] log pi_vllm(a_t | s_t) captured directly from vLLM rollouts.
        advantages:       [B]    group-normalized scalar advantage per trajectory.
        loss_mask:        [B, T] binary mask (1 on assistant tokens, 0 on prompt/tool observations).
    """
    if policy_logprobs.shape != rollout_logprobs.shape or policy_logprobs.shape != loss_mask.shape:
        raise ValueError("policy_logprobs, rollout_logprobs, and loss_mask must share shape [B, T]")
    if advantages.ndim != 1 or advantages.shape[0] != policy_logprobs.shape[0]:
        raise ValueError("advantages must have shape [B]")

    mask = loss_mask.to(dtype=policy_logprobs.dtype)
    active_bool = mask > 0.5
    active_count = int(active_bool.sum().item())
    if active_count == 0:
        raise ValueError("loss_mask has zero active assistant tokens")

    # Zero out masked positions BEFORE exponentiation so prompt/tool tokens have strictly zero grad.
    log_ratio = (policy_logprobs - rollout_logprobs.detach()) * mask
    # Clamp exponent for overflow safety in fp16/bf16 while preserving true TIM diagnostics below.
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))

    with torch.no_grad():
        raw_delta = (policy_logprobs - rollout_logprobs)[active_bool].abs()
        raw_ratio = torch.exp((policy_logprobs - rollout_logprobs)[active_bool].clamp(-20.0, 20.0))
        high_ratio_frac = float((raw_ratio > 2.0).float().mean().item())
        w_sum = raw_ratio.sum()
        w_sq_sum = (raw_ratio * raw_ratio).sum().clamp_min(1e-12)
        ess = float(((w_sum * w_sum) / w_sq_sum).item())
        p99 = float(torch.quantile(raw_delta.float(), 0.99).item())
        mean_abs = float(raw_delta.float().mean().item())
        max_abs = float(raw_delta.float().max().item())

    # Masked Importance Sampling (MIS): drop extreme upper-tail tokens caused by TIM/staleness.
    if use_mis:
        mis_keep = ((ratio.detach() <= mis_max_ratio) & active_bool).to(dtype=policy_logprobs.dtype)
        masked_frac = float(1.0 - (mis_keep.sum().item() / active_count))
    else:
        mis_keep = mask
        masked_frac = 0.0

    adv = advantages.detach().unsqueeze(-1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_per_tok = -torch.minimum(surr1, surr2)

    if kl_coef > 0.0:
        if ref_logprobs is None:
            raise ValueError("ref_logprobs is required when kl_coef > 0")
        # Schulman k3 unbiased non-negative KL: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
        ref_diff = (ref_logprobs.detach() - policy_logprobs) * mask
        kl_per_tok = torch.exp(ref_diff.clamp(-20.0, 20.0)) - ref_diff - 1.0
        total_per_tok = pg_per_tok + kl_coef * kl_per_tok
    else:
        total_per_tok = pg_per_tok

    denom = mis_keep.sum().clamp_min(1.0)
    loss = (total_per_tok * mis_keep).sum() / denom

    stats = TIMStats(
        mean_abs_logprob_diff=mean_abs,
        p99_abs_logprob_diff=p99,
        max_abs_logprob_diff=max_abs,
        high_ratio_fraction=high_ratio_frac,
        masked_token_fraction=masked_frac,
        effective_sample_size=ess,
        active_tokens=active_count,
    )
    return loss, stats
