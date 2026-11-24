from __future__ import annotations

import json

from loop.envs.generator import generate_split
from loop.envs.grader import grade_workspace, run_self_check
from loop.envs.sandbox import LocalSubprocessSandbox
from loop.rollout.protocol import TrajectoryBuilder, apply_tool_call, parse_tool_call


def test_disjoint_train_eval_splits() -> None:
    train = generate_split("train", 15)
    ev = generate_split("eval", 15)
    assert {t.seed for t in train}.isdisjoint({t.seed for t in ev})
    assert {t.task_id for t in train}.isdisjoint({t.task_id for t in ev})


def test_all_families_pass_self_check_and_block_reward_hacking() -> None:
    tasks = generate_split("train", 9) + generate_split("eval", 6)
    summary = run_self_check(tasks)
    assert summary == {"checked": 15, "failed": 0}


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
