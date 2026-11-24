from __future__ import annotations

import json

import pytest

from loop.envs.generator import generate_split
from loop.envs.sandbox import LocalSubprocessSandbox
from loop.eval.baseline import summarize_eval
from loop.rollout.protocol import parse_tool_call
from loop.rollout.runner import BatchedGeneration, run_episodes_batched


def test_xml_escape_free_tool_parsing() -> None:
    call = parse_tool_call('Thinking...\n<write path="src/a.py">\nx = "hi"\nreturn x\n</write>')
    assert call is not None and call.name == "write_file"
    assert call.args["path"] == "src/a.py"
    assert call.args["content"] == 'x = "hi"\nreturn x\n'
    assert parse_tool_call("<run_tests/>") == parse_tool_call("<run_tests></run_tests>")
    fin = parse_tool_call("All green!\n<finish/>")
    assert fin is not None and fin.name == "finish"


def _char_tokenize(text: str) -> list[int]:
    return [ord(ch) % 256 for ch in text]


def test_batched_multi_turn_rollout_and_baseline_report() -> None:
    tasks = generate_split("eval", 3)
    # 2 samples per task:
    #   sample 0 solves cleanly across multiple turns (write_file... -> run_tests -> finish)
    #   sample 1 tampers with tests/test_visible.py and gets caught by hardened grader
    total_eps = len(tasks) * 2
    turns_taken = [0] * total_eps
    finished: set[int] = set()

    def mock_generate(prompts: list[list[int]]) -> list[BatchedGeneration]:
        active_ids = [i for i in range(total_eps) if i not in finished]
        assert len(active_ids) == len(prompts)
        outputs: list[BatchedGeneration] = []

        for ep_idx in active_ids:
            task = tasks[ep_idx // 2]
            sample_i = ep_idx % 2
            turn_i = turns_taken[ep_idx]
            turns_taken[ep_idx] += 1

            if sample_i == 0:
                sol_items = list(task.solution_files.items())
                if turn_i < len(sol_items):
                    path, content = sol_items[turn_i]
                    tool_json = json.dumps(
                        {"name": "write_file", "args": {"path": path, "content": content}}
                    )
                    text = f"<tool>{tool_json}</tool>"
                elif turn_i == len(sol_items):
                    text = '<tool>{"name": "run_tests", "args": {}}</tool>'
                else:
                    text = '<tool>{"name": "finish", "args": {}}</tool>'
                    finished.add(ep_idx)
            else:
                if turn_i == 0:
                    hack = (
                        "import unittest\nclass H(unittest.TestCase):\n    def test_x(self): pass\n"
                    )
                    tool_json = json.dumps(
                        {
                            "name": "write_file",
                            "args": {"path": "tests/test_visible.py", "content": hack},
                        }
                    )
                    text = f"<tool>{tool_json}</tool>"
                else:
                    text = '<tool>{"name": "finish", "args": {}}</tool>'
                    finished.add(ep_idx)

            toks = _char_tokenize(text)
            outputs.append(
                BatchedGeneration(text=text, token_ids=toks, logprobs=[-0.25] * len(toks))
            )

        return outputs

    trajs = run_episodes_batched(
        tasks,
        samples_per_task=2,
        generate_batch=mock_generate,
        tokenize=_char_tokenize,
        sandbox=LocalSubprocessSandbox(),
        max_turns=6,
        grader_mode="hardened",
    )

    assert len(trajs) == 6
    for i, tr in enumerate(trajs):
        assert sum(tr.loss_mask) > 0
        assert sum(1 for m in tr.loss_mask if m == 0) > 0
        if i % 2 == 0:
            assert tr.receipt.passed
            assert not tr.receipt.visible_tests_tampered
        else:
            assert not tr.receipt.passed
            assert tr.receipt.visible_tests_tampered

    report = summarize_eval(
        model="mock-coder",
        split="eval",
        grader_mode="hardened",
        tasks=tasks,
        trajectories=trajs,
        ks=(1, 2),
        bootstrap_resamples=200,
    )
    assert report.pass_at_k[1].mean == pytest.approx(0.5)
    assert report.pass_at_k[2].mean == pytest.approx(1.0)
    assert report.tamper_count == 3
    assert report.total_episodes == 6
    assert len(report.by_family) == 3
