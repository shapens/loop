from __future__ import annotations

import json

from loop.bench_sandboxes import benchmark_sandbox_backend
from loop.envs.generator import FAMILIES, generate_split
from loop.envs.grader import grade_workspace, run_self_check
from loop.envs.sandbox import LocalSubprocessSandbox
from loop.rollout.protocol import TrajectoryBuilder, apply_tool_call, parse_tool_call


def test_disjoint_train_eval_splits() -> None:
    train = generate_split("train", 15)
    ev = generate_split("eval", 15)
    assert {t.seed for t in train}.isdisjoint({t.seed for t in ev})
    assert {t.task_id for t in train}.isdisjoint({t.task_id for t in ev})


def test_all_families_pass_self_check_and_block_reward_hacking() -> None:
    tasks = generate_split("train", len(FAMILIES) * 2) + generate_split("eval", len(FAMILIES) * 2)
    assert {t.family for t in tasks} == set(FAMILIES)
    summary = run_self_check(tasks)
    assert summary == {"checked": len(FAMILIES) * 4, "failed": 0}


def test_sandbox_benchmark_harness_runs_cleanly() -> None:
    res = benchmark_sandbox_backend(
        "local",
        LocalSubprocessSandbox(),
        concurrency_levels=(1, 4),
        tasks_per_level=4,
    )
    assert len(res) == 2
    assert all(r.all_correct and r.throughput_per_sec > 0 for r in res)


def test_multi_turn_tool_protocol_and_loss_mask() -> None:
    task = generate_split("eval", 1)[0]
    ws = dict(task.repo_files)
    sandbox = LocalSubprocessSandbox()

    call = parse_tool_call('I will inspect files.\n<tool>{"name": "list_files", "args": {}}</tool>')
    assert call is not None and call.name == "list_files"
    obs, done, _ = apply_tool_call(call, ws, sandbox)
    assert not done
    assert "tests/test_visible.py" in obs

    # Apply reference fix via real write_file tool calls.
    for path, content in task.solution_files.items():
        payload = json.dumps({"name": "write_file", "args": {"path": path, "content": content}})
        wcall = parse_tool_call(f"<tool>{payload}</tool>")
        assert wcall is not None and wcall.name == "write_file"
        wobs, wdone, _ = apply_tool_call(wcall, ws, sandbox)
        assert not wdone and wobs.startswith("wrote ")

    receipt = grade_workspace(task, ws, mode="hardened")
    assert receipt.passed
    assert receipt.reward == 1.0
    assert not receipt.visible_tests_tampered

    tb = TrajectoryBuilder()
    tb.append_observation([1, 2, 3])
    tb.append_assistant([10, 11], [-0.5, -0.2])
    tb.append_observation([4, 5])
    tb.append_assistant([12], [-0.1])
    assert tb.loss_mask == [0, 0, 0, 1, 1, 0, 0, 1]
    assert tb.rollout_logprobs == [0.0, 0.0, 0.0, -0.5, -0.2, 0.0, 0.0, -0.1]


def test_subtest_reward_shaping_and_tamper_gating() -> None:
    """Partial progress earns <= 0.5 shaped reward; full pass earns 1.0; tampering gets 0.0."""
    # keyset_paging has 3 source files + 4 total tests (1 pristine visible + 3 hidden).
    task = generate_split("eval", 1)[0]
    assert task.family == "keyset_paging"

    # Fixing 1 of 3 files should earn strictly > 0 and < 0.5 in subtest mode, 0.0 in binary mode.
    partial = dict(task.repo_files)
    partial["src/cursor.py"] = task.solution_files["src/cursor.py"]
    r_bin = grade_workspace(task, partial, mode="hardened", reward_mode="binary")
    r_sub = grade_workspace(task, partial, mode="hardened", reward_mode="subtest")
    assert not r_bin.passed and r_bin.reward == 0.0
    assert not r_sub.passed
    assert r_sub.subtests_total >= 4
    assert 0 < r_sub.subtests_passed < r_sub.subtests_total
    assert 0.0 < r_sub.reward < 0.5

    # Fixing all files earns 1.0 in both modes.
    solved = {**task.repo_files, **task.solution_files}
    r_full = grade_workspace(task, solved, mode="hardened", reward_mode="subtest")
    assert r_full.passed and r_full.reward == 1.0
    assert r_full.subtests_passed == r_full.subtests_total

    # Tampering strictly zeroes reward even in subtest mode.
    tampered = dict(solved)
    tampered["tests/test_visible.py"] = "import unittest\n"
    r_hack = grade_workspace(task, tampered, mode="hardened", reward_mode="subtest")
    assert not r_hack.passed and r_hack.visible_tests_tampered and r_hack.reward == 0.0
