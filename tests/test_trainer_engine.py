from __future__ import annotations

from pathlib import Path

import pytest
import torch

from loop.trainer.engine import (
    GRPOConfig,
    collate_trajectories,
    compute_shifted_logprobs,
    grpo_train_step,
)
from loop.trainer.triton_logprob import fused_shifted_logprobs
from loop.types import Trajectory, VerifierReceipt


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = attention_mask
        out: torch.Tensor = self.head(self.emb(input_ids))
        return out


def _dummy_receipt(passed: bool) -> VerifierReceipt:
    return VerifierReceipt(
        task_id="t0",
        passed=passed,
        reward=1.0 if passed else 0.0,
        grader_mode="hardened",
        visible_tests_tampered=False,
        unexpected_edits=(),
        timed_out=False,
        duration_ms=5.0,
        stdout="",
        stderr="",
        manifest_sha256="abc",
    )


def test_compute_shifted_logprobs_matches_explicit_log_softmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    input_ids = torch.randint(0, 11, (2, 6))

    fast = compute_shifted_logprobs(logits, input_ids)
    fused = fused_shifted_logprobs(logits, input_ids)
    expected = (
        torch.log_softmax(logits[:, :-1, :], dim=-1)
        .gather(-1, input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.allclose(fast, expected, atol=1e-5)
    assert torch.allclose(fused, expected, atol=1e-5)


def test_grpo_train_step_updates_weights_and_verifies_export(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = _TinyCausalLM(vocab=32, dim=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Create 4 trajectories in one group (2 pass, 2 fail) with rollout logprobs close to policy.
    trajs: list[Trajectory] = []
    with torch.no_grad():
        for i in range(4):
            ids = [1, 2, 3, 4 + i, 5 + i, 6]
            logits = model(torch.tensor([ids]))
            shifted = compute_shifted_logprobs(logits, torch.tensor([ids]))[0].tolist()
            # Align rollout_logprobs to full token_ids (index 0 unused/masked).
            lps = [0.0] + shifted
            mask = [0, 0, 0, 1, 1, 1]
            trajs.append(
                Trajectory(
                    task_id="t0",
                    prompt_id=f"t0#s{i}",
                    token_ids=ids,
                    loss_mask=mask,
                    rollout_logprobs=lps,
                    policy_version=0,
                    turns=2,
                    receipt=_dummy_receipt(passed=(i < 2)),
                    rollout_ms=20.0,
                    sandbox_ms=10.0,
                )
            )

    batch = collate_trajectories(trajs, pad_token_id=0, max_seq_len=32)
    assert batch.input_ids.shape == (4, 6)

    cfg = GRPOConfig(group_size=4, micro_batch_size=2, kl_coef=0.0, use_mis=True)
    export_dir = tmp_path / "step_1"
    metrics = grpo_train_step(
        model,
        opt,
        trajs,
        pad_token_id=0,
        cfg=cfg,
        step=1,
        adapter_export_dir=export_dir,
    )

    assert metrics.step == 1
    assert metrics.pass_rate == pytest.approx(0.5)
    assert metrics.mean_reward == pytest.approx(0.5)
    assert metrics.tim.mean_abs_logprob_diff < 1e-4
    assert metrics.grad_norm > 0.0
    assert (export_dir / "weights.pt").exists()
    assert len(metrics.adapter_sha256) == 64
