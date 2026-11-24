from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PassAtKCI:
    k: int
    mean: float
    ci_low: float
    ci_high: float
    num_tasks: int


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n - c, k) / C(n, k)."""
    if k <= 0 or n <= 0 or k > n:
        raise ValueError(f"Require 0 < k <= n, got n={n}, k={k}")
    if c < 0 or c > n:
        raise ValueError(f"Require 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def bootstrap_pass_at_k(
    task_outcomes: list[tuple[int, int]],
    k: int,
    *,
    num_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> PassAtKCI:
    """Compute mean pass@k and deterministic percentile bootstrap CI across tasks.

    Args:
        task_outcomes: list of (n_samples, c_passed) per held-out task.
    """
    if not task_outcomes:
        raise ValueError("task_outcomes must be non-empty")
    scores = [estimate_pass_at_k(n, c, k) for (n, c) in task_outcomes]
    m = len(scores)
    point = sum(scores) / m

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(num_resamples):
        sample_sum = 0.0
        for _ in range(m):
            sample_sum += scores[rng.randrange(m)]
        means.append(sample_sum / m)
    means.sort()

    lo_idx = max(0, int((alpha / 2.0) * num_resamples))
    hi_idx = min(num_resamples - 1, int((1.0 - alpha / 2.0) * num_resamples))
    return PassAtKCI(
        k=k,
        mean=point,
        ci_low=means[lo_idx],
        ci_high=means[hi_idx],
        num_tasks=m,
    )
