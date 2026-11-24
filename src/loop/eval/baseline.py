from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from loop.eval.stats import PassAtKCI, bootstrap_pass_at_k
from loop.types import TaskSpec, Trajectory


@dataclass(frozen=True)
class FamilySummary:
    family: str
    num_tasks: int
    pass_at_1: float
    tamper_rate: float
    mean_turns: float


@dataclass(frozen=True)
class EvalReport:
    model: str
    split: str
    grader_mode: str
    samples_per_task: int
    pass_at_k: dict[int, PassAtKCI]
    tamper_count: int
    total_episodes: int
    timeout_count: int
    mean_turns: float
    mean_rollout_ms: float
    mean_sandbox_ms: float
    by_family: list[FamilySummary]


def summarize_eval(
    *,
    model: str,
    split: str,
    grader_mode: str,
    tasks: list[TaskSpec],
    trajectories: list[Trajectory],
    ks: tuple[int, ...] = (1, 4),
    bootstrap_resamples: int = 2000,
    seed: int = 42,
) -> EvalReport:
    """Aggregate trajectory receipts into pass@k CIs, tamper rate, and timing telemetry."""
    task_by_id = {t.task_id: t for t in tasks}
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        grouped[traj.task_id].append(traj)

    if not grouped:
        raise ValueError("trajectories must be non-empty")

    samples_per_task = min(len(v) for v in grouped.values())
    outcomes = [
        (len(trajs), sum(1 for tr in trajs if tr.receipt.passed)) for trajs in grouped.values()
    ]

    valid_ks = [k for k in ks if k <= samples_per_task]
    if 1 not in valid_ks:
        valid_ks.insert(0, 1)

    pass_at_k = {
        k: bootstrap_pass_at_k(outcomes, k=k, num_resamples=bootstrap_resamples, seed=seed)
        for k in valid_ks
    }

    total_eps = len(trajectories)
    tamper_count = sum(1 for tr in trajectories if tr.receipt.visible_tests_tampered)
    timeout_count = sum(1 for tr in trajectories if tr.receipt.timed_out)
    mean_turns = sum(tr.turns for tr in trajectories) / total_eps
    mean_rollout_ms = sum(tr.rollout_ms for tr in trajectories) / total_eps
    mean_sandbox_ms = sum(tr.sandbox_ms for tr in trajectories) / total_eps

    fam_trajs: dict[str, list[Trajectory]] = defaultdict(list)
    fam_tasks: dict[str, set[str]] = defaultdict(set)
    for tr in trajectories:
        fam = task_by_id[tr.task_id].family if tr.task_id in task_by_id else "unknown"
        fam_trajs[fam].append(tr)
        fam_tasks[fam].add(tr.task_id)

    by_family: list[FamilySummary] = []
    for fam in sorted(fam_trajs.keys()):
        trs = fam_trajs[fam]
        n = len(trs)
        by_family.append(
            FamilySummary(
                family=fam,
                num_tasks=len(fam_tasks[fam]),
                pass_at_1=sum(1 for tr in trs if tr.receipt.passed) / n,
                tamper_rate=sum(1 for tr in trs if tr.receipt.visible_tests_tampered) / n,
                mean_turns=sum(tr.turns for tr in trs) / n,
            )
        )

    return EvalReport(
        model=model,
        split=split,
        grader_mode=grader_mode,
        samples_per_task=samples_per_task,
        pass_at_k=pass_at_k,
        tamper_count=tamper_count,
        total_episodes=total_eps,
        timeout_count=timeout_count,
        mean_turns=mean_turns,
        mean_rollout_ms=mean_rollout_ms,
        mean_sandbox_ms=mean_sandbox_ms,
        by_family=by_family,
    )
