from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from loop.envs.sandbox import SandboxBackend

TOOL_BLOCK_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)
WRITE_TAG_RE = re.compile(r'<write\s+path=["\']([^"\']+)["\']\s*>\n?(.*?)\n?</write>', re.DOTALL)
READ_TAG_RE = re.compile(r'<read\s+path=["\']([^"\']+)["\']\s*/?>', re.DOTALL)
RUN_TESTS_TAG_RE = re.compile(r"<run_tests\s*/?>", re.DOTALL)
FINISH_TAG_RE = re.compile(r"<finish\s*/?>", re.DOTALL)

STOP_STRINGS: tuple[str, ...] = (
    "</tool>",
    "</write>",
    "</read>",
    "<run_tests/>",
    "</run_tests>",
    "<finish/>",
    "</finish>",
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, str]


def parse_tool_call(assistant_text: str) -> ToolCall | None:
    """Parse either an escape-free XML tool tag or a `<tool>{...}</tool>` JSON call."""
    m_write = WRITE_TAG_RE.search(assistant_text)
    if m_write:
        return ToolCall(
            name="write_file",
            args={"path": m_write.group(1).strip(), "content": m_write.group(2) + "\n"},
        )

    m_read = READ_TAG_RE.search(assistant_text)
    if m_read:
        return ToolCall(name="read_file", args={"path": m_read.group(1).strip()})

    if RUN_TESTS_TAG_RE.search(assistant_text):
        return ToolCall(name="run_tests", args={})

    if FINISH_TAG_RE.search(assistant_text):
        return ToolCall(name="finish", args={})

    match = TOOL_BLOCK_RE.search(assistant_text)
    if not match:
        return None
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
        return None
    args_obj = raw.get("args", {})
    if not isinstance(args_obj, dict):
        return None
    return ToolCall(
        name=raw["name"],
        args={str(k): str(v) for k, v in args_obj.items()},
    )


def apply_tool_call(
    call: ToolCall,
    workspace_files: dict[str, str],
    sandbox: SandboxBackend,
    timeout_seconds: float = 10.0,
) -> tuple[str, bool, float]:
    """Execute one coding tool on the workspace; returns (observation, done, sandbox_ms)."""
    if call.name == "list_files":
        return "\n".join(sorted(workspace_files.keys())), False, 0.0

    if call.name == "read_file":
        path = call.args.get("path", "")
        if path not in workspace_files:
            return f"error: file not found: {path}", False, 0.0
        return workspace_files[path], False, 0.0

    if call.name == "write_file":
        path = call.args.get("path", "")
        content = call.args.get("content", "")
        if not path or path.startswith("/") or ".." in path.split("/"):
            return f"error: invalid relative path: {path}", False, 0.0
        workspace_files[path] = content
        return f"wrote {path} ({len(content)} bytes)", False, 0.0

    if call.name == "run_tests":
        rc, out, err, ms = sandbox.run_command(
            workspace_files,
            [sys.executable, "-I", "-S", "-m", "unittest", "discover", "-s", "tests", "-t", "."],
            timeout_seconds=timeout_seconds,
        )
        obs = f"exit_code={rc}\nSTDOUT:\n{out[-2000:]}\nSTDERR:\n{err[-2000:]}"
        return obs, False, ms

    if call.name == "finish":
        return "finished", True, 0.0

    return f"error: unknown tool: {call.name}", False, 0.0


class TrajectoryBuilder:
    """Accumulates exact token IDs, rollout logprobs, and binary loss_mask across turns."""

    def __init__(self) -> None:
        self.token_ids: list[int] = []
        self.loss_mask: list[int] = []
        self.rollout_logprobs: list[float] = []

    def append_observation(self, token_ids: list[int]) -> None:
        """Append prompt or tool-output tokens (always masked out of policy gradient)."""
        self.token_ids.extend(token_ids)
        self.loss_mask.extend([0] * len(token_ids))
        self.rollout_logprobs.extend([0.0] * len(token_ids))

    def append_assistant(self, token_ids: list[int], logprobs: list[float]) -> None:
        """Append exact vLLM-generated assistant token IDs and their rollout logprobs."""
        if len(token_ids) != len(logprobs):
            raise ValueError(
                f"token_ids ({len(token_ids)}) and logprobs ({len(logprobs)}) must match"
            )
        self.token_ids.extend(token_ids)
        self.loss_mask.extend([1] * len(token_ids))
        self.rollout_logprobs.extend(logprobs)
