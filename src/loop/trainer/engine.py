from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from loop.trainer.grpo import compute_group_advantages, grpo_loss
from loop.trainer.triton_logprob import fused_shifted_logprobs
from loop.trainer.weight_sync import assert_weights_synced, state_dict_sha256
from loop.types import GraderMode, StepTiming, TIMStats, Trajectory


@dataclass(frozen=True)
class GRPOConfig:
    """Hyperparameters and ablation switches for multi-turn GRPO/RLVR."""

    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    steps: int = 15
    tasks_per_step: int = 4
    group_size: int = 8
    max_turns: int = 6
    max_seq_len: int = 4096
    micro_batch_size: int = 2
    clip_eps: float = 0.2
    use_mis: bool = True
    mis_max_ratio: float = 2.0
    stale_steps: int = 0
    use_triton_logprobs: bool = True
    kl_coef: float = 0.01
    temperature: float = 0.8
    max_new_tokens: int = 1024
    grader_mode: GraderMode = "hardened"
    eval_every: int = 5
    eval_tasks: int = 30
    eval_samples: int = 4


@dataclass(frozen=True)
class CollatedBatch:
    input_ids: torch.Tensor  # [B, L] int64
    attention_mask: torch.Tensor  # [B, L] int64
    loss_mask: torch.Tensor  # [B, L] float32 (1 on assistant tokens, 0 elsewhere)
    rollout_logprobs: torch.Tensor  # [B, L] float32


@dataclass(frozen=True)
class StepMetrics:
    step: int
    mean_reward: float
    pass_rate: float
    tamper_rate: float
    loss: float
    grad_norm: float
    tim: TIMStats
    timing: StepTiming
    adapter_sha256: str


def collate_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    pad_token_id: int,
    max_seq_len: int,
    device: torch.device | str = "cpu",
) -> CollatedBatch:
    """Pad and align trajectories into tensors without breaking token-ID / mask correspondence."""
    if not trajectories:
        raise ValueError("trajectories must be non-empty")

    truncated: list[tuple[list[int], list[int], list[float]]] = []
    max_len = 0
    for tr in trajectories:
        tr.validate()
        ids = tr.token_ids[:max_seq_len]
        mask = tr.loss_mask[:max_seq_len]
        lps = tr.rollout_logprobs[:max_seq_len]
        if len(ids) < 2:
            raise ValueError(f"Trajectory {tr.prompt_id} shorter than 2 tokens")
        truncated.append((ids, mask, lps))
        max_len = max(max_len, len(ids))

    bsz = len(truncated)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    loss_mask = torch.zeros((bsz, max_len), dtype=torch.float32)
    rollout_logprobs = torch.zeros((bsz, max_len), dtype=torch.float32)

    for i, (ids, mask, lps) in enumerate(truncated):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        loss_mask[i, :n] = torch.tensor(mask, dtype=torch.float32)
        rollout_logprobs[i, :n] = torch.tensor(lps, dtype=torch.float32)

    return CollatedBatch(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        loss_mask=loss_mask.to(device),
        rollout_logprobs=rollout_logprobs.to(device),
    )


def compute_shifted_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute exact per-token log P(x_t | x_<t) for t in 1..L-1 without full log_softmax HBM spike.

    Args:
        logits:    [B, L, V] raw causal LM logits.
        input_ids: [B, L]    target token IDs.
    Returns:
        [B, L - 1] float32 log-probabilities aligned to input_ids[:, 1:].
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits [B, L, V], got {tuple(logits.shape)}")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    bsz, seq_len, vocab = shift_logits.shape
    neg_nll = -F.cross_entropy(
        shift_logits.view(bsz * seq_len, vocab).float(),
        shift_labels.view(bsz * seq_len),
        reduction="none",
    )
    return neg_nll.view(bsz, seq_len)


def directory_sha256(path: Path) -> str:
    """Deterministic SHA-256 over all regular files in an exported adapter directory."""
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No adapter files found under {path}")
    for fp in files:
        rel = fp.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(fp.read_bytes())
    return h.hexdigest()


def verify_adapter_export(
    model: torch.nn.Module,
    adapter_dir: Path,
    *,
    policy_version: int,
) -> str:
    """Save trainable/PEFT adapter to `adapter_dir` and verify on-disk integrity via SHA-256."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_fn = getattr(model, "save_pretrained", None)
    if callable(save_fn):
        save_fn(str(adapter_dir))
        disk_digest = directory_sha256(adapter_dir)
        # Re-read digest to verify storage visibility before handing path to vLLM.
        verify_digest = directory_sha256(adapter_dir)
        assert_weights_synced(disk_digest, verify_digest, policy_version=policy_version)
        return disk_digest

    # Fallback for plain torch.nn.Module (used in unit tests).
    trainable = {k: v for k, v in model.state_dict().items()}
    mem_digest = state_dict_sha256(trainable)
    out_file = adapter_dir / "weights.pt"
    torch.save(trainable, out_file)
    reloaded = torch.load(out_file, map_location="cpu", weights_only=True)
    disk_digest = state_dict_sha256(reloaded)
    assert_weights_synced(mem_digest, disk_digest, policy_version=policy_version)
    return disk_digest


def grpo_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[Trajectory],
    *,
    pad_token_id: int,
    cfg: GRPOConfig,
    step: int,
    device: torch.device | str = "cpu",
    adapter_export_dir: Path | None = None,
    rollout_wall_ms: float | None = None,
    sandbox_wall_ms: float | None = None,
) -> StepMetrics:
    """Run one gradient-accumulated GRPO policy update over `trajectories`."""
    t_learn = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    rewards = torch.tensor(
        [tr.receipt.reward for tr in trajectories], dtype=torch.float32, device=device
    )
    advantages = compute_group_advantages(rewards, group_size=cfg.group_size)

    total_eps = len(trajectories)
    mb = max(1, min(cfg.micro_batch_size, total_eps))
    num_chunks = (total_eps + mb - 1) // mb

    accum_loss = 0.0
    tim_active = 0
    tim_abs_sum = 0.0
    tim_max = 0.0
    tim_p99_max = 0.0
    tim_high_ratio_num = 0.0
    tim_masked_num = 0.0
    tim_ess_sum = 0.0

    has_active_mb = 0
    for start in range(0, total_eps, mb):
        chunk = trajectories[start : start + mb]
        chunk_adv = advantages[start : start + mb]
        batch = collate_trajectories(
            chunk, pad_token_id=pad_token_id, max_seq_len=cfg.max_seq_len, device=device
        )
        shifted_mask = batch.loss_mask[:, 1:]
        if float(shifted_mask.sum().item()) == 0.0:
            continue

        out: Any = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        logits: torch.Tensor = out.logits if hasattr(out, "logits") else out
        policy_lp = (
            fused_shifted_logprobs(logits, batch.input_ids)
            if cfg.use_triton_logprobs
            else compute_shifted_logprobs(logits, batch.input_ids)
        )
        rollout_lp = batch.rollout_logprobs[:, 1:]

        ref_lp: torch.Tensor | None = None
        if cfg.kl_coef > 0.0:
            disable_ctx = (
                model.disable_adapter()  # type: ignore[operator]
                if callable(getattr(model, "disable_adapter", None))
                else nullcontext()
            )
            with torch.no_grad(), disable_ctx:
                ref_out: Any = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                ref_logits: torch.Tensor = ref_out.logits if hasattr(ref_out, "logits") else ref_out
                ref_lp = compute_shifted_logprobs(ref_logits, batch.input_ids)

        loss, tim = grpo_loss(
            policy_lp,
            rollout_lp,
            chunk_adv,
            shifted_mask,
            clip_eps=cfg.clip_eps,
            use_mis=cfg.use_mis,
            mis_max_ratio=cfg.mis_max_ratio,
            kl_coef=cfg.kl_coef,
            ref_logprobs=ref_lp,
        )
        (loss / num_chunks).backward()  # type: ignore[no-untyped-call]
        accum_loss += float(loss.item())
        has_active_mb += 1

        tim_active += tim.active_tokens
        tim_abs_sum += tim.mean_abs_logprob_diff * tim.active_tokens
        tim_max = max(tim_max, tim.max_abs_logprob_diff)
        tim_p99_max = max(tim_p99_max, tim.p99_abs_logprob_diff)
        tim_high_ratio_num += tim.high_ratio_fraction * tim.active_tokens
        tim_masked_num += tim.masked_token_fraction * tim.active_tokens
        tim_ess_sum += tim.effective_sample_size

    grad_norm_t = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
    )
    grad_norm = float(grad_norm_t.item()) if isinstance(grad_norm_t, torch.Tensor) else 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    learner_ms = (time.perf_counter() - t_learn) * 1000.0

    t_sync = time.perf_counter()
    if adapter_export_dir is not None:
        digest = verify_adapter_export(model, adapter_export_dir, policy_version=step)
    else:
        digest = state_dict_sha256(
            {k: v for k, v in model.named_parameters() if v.requires_grad}
            or dict(model.state_dict())
        )
    sync_ms = (time.perf_counter() - t_sync) * 1000.0

    denom = max(tim_active, 1)
    agg_tim = TIMStats(
        mean_abs_logprob_diff=tim_abs_sum / denom,
        p99_abs_logprob_diff=tim_p99_max,
        max_abs_logprob_diff=tim_max,
        high_ratio_fraction=tim_high_ratio_num / denom,
        masked_token_fraction=tim_masked_num / denom,
        effective_sample_size=tim_ess_sum / max(has_active_mb, 1),
        active_tokens=tim_active,
    )
    timing = StepTiming(
        rollout_ms=(
            rollout_wall_ms
            if rollout_wall_ms is not None
            else sum(tr.rollout_ms for tr in trajectories)
        ),
        sandbox_ms=(
            sandbox_wall_ms
            if sandbox_wall_ms is not None
            else sum(tr.sandbox_ms for tr in trajectories)
        ),
        learner_ms=learner_ms,
        weight_sync_ms=sync_ms,
    )
    return StepMetrics(
        step=step,
        mean_reward=float(rewards.mean().item()),
        pass_rate=sum(1 for tr in trajectories if tr.receipt.passed) / total_eps,
        tamper_rate=sum(1 for tr in trajectories if tr.receipt.visible_tests_tampered) / total_eps,
        loss=accum_loss / max(has_active_mb, 1),
        grad_norm=grad_norm,
        tim=agg_tim,
        timing=timing,
        adapter_sha256=digest,
    )
