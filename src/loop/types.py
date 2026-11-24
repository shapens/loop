from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Split = Literal["train", "eval"]
GraderMode = Literal["hardened", "naive"]


@dataclass(frozen=True)
class TaskSpec:
    """Self-contained multi-file coding task with withheld hidden tests."""

    task_id: str
    family: str
    split: Split
    seed: int
    prompt: str
    repo_files: dict[str, str]
    hidden_tests: dict[str, str]
    solution_files: dict[str, str]
    allowed_edit_globs: tuple[str, ...] = ("src/*",)
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class VerifierReceipt:
    """Deterministic grading outcome and anti-cheat telemetry for one rollout."""

    task_id: str
    passed: bool
    reward: float
    grader_mode: GraderMode
    visible_tests_tampered: bool
    unexpected_edits: tuple[str, ...]
    timed_out: bool
    duration_ms: float
    stdout: str
    stderr: str
    manifest_sha256: str


@dataclass
class Trajectory:
    """Token-aligned multi-turn episode returned by a rollout worker."""

    task_id: str
    prompt_id: str
    token_ids: list[int]
    loss_mask: list[int]  # 1 = assistant token, 0 = prompt / tool observation
    rollout_logprobs: list[float]  # aligned to token_ids; 0.0 where loss_mask == 0
    policy_version: int
    turns: int
    receipt: VerifierReceipt
    rollout_ms: float = 0.0
    sandbox_ms: float = 0.0

    def validate(self) -> None:
        n = len(self.token_ids)
        if len(self.loss_mask) != n or len(self.rollout_logprobs) != n:
            raise ValueError(
                f"Misaligned trajectory lengths: ids={n}, "
                f"mask={len(self.loss_mask)}, logprobs={len(self.rollout_logprobs)}"
            )
        if any(m not in (0, 1) for m in self.loss_mask):
            raise ValueError("loss_mask must contain strictly 0 or 1")


@dataclass(frozen=True)
class TIMStats:
    """Train-Inference Mismatch (TIM) and importance-sampling diagnostics."""

    mean_abs_logprob_diff: float
    p99_abs_logprob_diff: float
    max_abs_logprob_diff: float
    high_ratio_fraction: float  # fraction of active tokens with r_t > 2.0
    masked_token_fraction: float  # fraction masked by MIS
    effective_sample_size: float
    active_tokens: int


@dataclass
class StepTiming:
    """Wall-clock breakdown per RL step to expose GPU starvation."""

    rollout_ms: float = 0.0
    sandbox_ms: float = 0.0
    learner_ms: float = 0.0
    weight_sync_ms: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.rollout_ms + self.sandbox_ms + self.learner_ms + self.weight_sync_ms

    def fractions(self) -> dict[str, float]:
        total = max(self.total_ms, 1e-9)
        return {
            "rollout": self.rollout_ms / total,
            "sandbox": self.sandbox_ms / total,
            "learner": self.learner_ms / total,
            "weight_sync": self.weight_sync_ms / total,
        }
