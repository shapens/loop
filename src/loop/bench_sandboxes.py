from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from loop.envs.generator import generate_split
from loop.envs.sandbox import SandboxBackend
from loop.types import TaskSpec


@dataclass(frozen=True)
class ConcurrencyResult:
    backend: str
    concurrency: int
    total_gradings: int
    wall_ms: float
    throughput_per_sec: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    all_correct: bool


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def benchmark_sandbox_backend(
    name: str,
    sandbox: SandboxBackend,
    *,
    concurrency_levels: tuple[int, ...] = (1, 8, 16, 32),
    tasks_per_level: int = 32,
) -> list[ConcurrencyResult]:
    """Measure p50/p95/p99 latency and throughput for a SandboxBackend under concurrency."""
    pool_tasks: list[TaskSpec] = generate_split("eval", max(tasks_per_level, 15))
    results: list[ConcurrencyResult] = []

    for conc in concurrency_levels:
        batch = [pool_tasks[i % len(pool_tasks)] for i in range(tasks_per_level)]
        latencies: list[float] = []
        all_ok = True

        def _run_one(task: TaskSpec) -> tuple[float, bool]:
            solved = {**task.repo_files, **task.solution_files}
            t0 = time.perf_counter()
            receipt = sandbox.grade(task, solved, mode="hardened")
            return (time.perf_counter() - t0) * 1000.0, receipt.passed

        t_wall = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            for lat_ms, passed in ex.map(_run_one, batch):
                latencies.append(lat_ms)
                all_ok = all_ok and passed
        wall_ms = (time.perf_counter() - t_wall) * 1000.0
        latencies.sort()

        results.append(
            ConcurrencyResult(
                backend=name,
                concurrency=conc,
                total_gradings=len(batch),
                wall_ms=round(wall_ms, 1),
                throughput_per_sec=round(len(batch) / max(wall_ms / 1000.0, 1e-6), 2),
                p50_ms=round(_percentile(latencies, 0.50), 1),
                p95_ms=round(_percentile(latencies, 0.95), 1),
                p99_ms=round(_percentile(latencies, 0.99), 1),
                all_correct=all_ok,
            )
        )
    return results


def results_to_dicts(items: list[ConcurrencyResult]) -> list[dict[str, Any]]:
    return [asdict(x) for x in items]
