from __future__ import annotations

import pytest

from loop.eval.stats import bootstrap_pass_at_k, estimate_pass_at_k
from loop.types import StepTiming


def test_unbiased_pass_at_k_known_values() -> None:
    assert estimate_pass_at_k(n=8, c=0, k=1) == 0.0
    assert estimate_pass_at_k(n=8, c=8, k=1) == 1.0
    assert estimate_pass_at_k(n=4, c=1, k=1) == pytest.approx(0.25)
    # 1 - C(3, 2)/C(4, 2) = 1 - 3/6 = 0.5
    assert estimate_pass_at_k(n=4, c=1, k=2) == pytest.approx(0.5)


def test_bootstrap_ci_deterministic_and_ordered() -> None:
    outcomes = [(4, 0), (4, 1), (4, 2), (4, 4), (4, 1), (4, 3)]
    ci1 = bootstrap_pass_at_k(outcomes, k=1, num_resamples=500, seed=7)
    ci2 = bootstrap_pass_at_k(outcomes, k=1, num_resamples=500, seed=7)
    assert ci1 == ci2
    assert 0.0 <= ci1.ci_low <= ci1.mean <= ci1.ci_high <= 1.0


def test_step_timing_fractions_sum_to_one() -> None:
    st = StepTiming(rollout_ms=500.0, sandbox_ms=300.0, learner_ms=150.0, weight_sync_ms=50.0)
    fracs = st.fractions()
    assert sum(fracs.values()) == pytest.approx(1.0)
    assert fracs["rollout"] == pytest.approx(0.5)
    assert fracs["sandbox"] == pytest.approx(0.3)
