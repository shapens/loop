from __future__ import annotations

import base64
import contextlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

from loop.envs.grader import (
    _detect_tampering,
    compute_shaped_reward,
    grade_workspace,
    manifest_sha256,
    parse_unittest_counts,
)
from loop.types import GraderMode, RewardMode, TaskSpec, VerifierReceipt


class SandboxBackend(Protocol):
    """Pluggable execution/grading substrate (Local, Modal Sandboxes, or shapens/tri)."""

    def run_command(
        self, workspace_files: dict[str, str], command: list[str], timeout_seconds: float
    ) -> tuple[int, str, str, float]: ...

    def grade(
        self,
        task: TaskSpec,
        workspace_files: dict[str, str],
        mode: GraderMode = "hardened",
        reward_mode: RewardMode = "binary",
    ) -> VerifierReceipt: ...


class LocalSubprocessSandbox:
    """Fast local subprocess sandbox for unit tests, CI, and CPU debugging."""

    def run_command(
        self, workspace_files: dict[str, str], command: list[str], timeout_seconds: float
    ) -> tuple[int, str, str, float]:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="loop-cmd-") as tmp:
            root = Path(tmp)
            for rel, content in workspace_files.items():
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            try:
                proc = subprocess.run(
                    command,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                return (
                    proc.returncode,
                    proc.stdout,
                    proc.stderr,
                    (time.perf_counter() - t0) * 1000.0,
                )
            except subprocess.TimeoutExpired:
                return 124, "", "command timed out", (time.perf_counter() - t0) * 1000.0

    def grade(
        self,
        task: TaskSpec,
        workspace_files: dict[str, str],
        mode: GraderMode = "hardened",
        reward_mode: RewardMode = "binary",
    ) -> VerifierReceipt:
        return grade_workspace(task, workspace_files, mode=mode, reward_mode=reward_mode)


class ModalSandboxBackend:
    """Ephemeral Modal Sandbox backend for isolated, high-concurrency RL rollouts."""

    def __init__(self, app_name: str = "shapens-loop", timeout_s: int = 30) -> None:
        self.app_name = app_name
        self.timeout_s = timeout_s

    def _ensure_modal(self) -> Any:
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError("Install modal extra: `uv pip install -e '.[modal]'`") from exc
        return modal

    def run_command(
        self, workspace_files: dict[str, str], command: list[str], timeout_seconds: float
    ) -> tuple[int, str, str, float]:
        modal = self._ensure_modal()
        t0 = time.perf_counter()
        app = modal.App.lookup(self.app_name, create_if_missing=True)
        image = modal.Image.debian_slim(python_version="3.11")
        payload = json.dumps({"files": workspace_files, "cmd": command})
        runner = (
            "import json, pathlib, subprocess, sys\n"
            "data = json.loads(sys.stdin.read())\n"
            "root = pathlib.Path('/tmp/work')\n"
            "for rel, txt in data['files'].items():\n"
            "    p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(txt)\n"
            "r = subprocess.run(data['cmd'], cwd=root, capture_output=True, text=True)\n"
            "print(json.dumps({'rc': r.returncode, 'out': r.stdout, 'err': r.stderr}))\n"
        )
        sb = modal.Sandbox.create(
            "python3",
            "-c",
            runner,
            app=app,
            image=image,
            timeout=int(max(timeout_seconds, self.timeout_s)),
        )
        try:
            sb.stdin.write(payload)
            sb.stdin.write_eof()
            sb.stdin.drain()
            sb.wait()
            raw = sb.stdout.read().strip()
            parsed = json.loads(raw) if raw else {"rc": 1, "out": "", "err": sb.stderr.read()}
            return (
                int(parsed["rc"]),
                str(parsed["out"]),
                str(parsed["err"]),
                (time.perf_counter() - t0) * 1000.0,
            )
        finally:
            sb.terminate()

    def grade(
        self,
        task: TaskSpec,
        workspace_files: dict[str, str],
        mode: GraderMode = "hardened",
        reward_mode: RewardMode = "binary",
    ) -> VerifierReceipt:
        from loop.modal_app import grade_remote

        return grade_remote(task, workspace_files, mode=mode, reward_mode=reward_mode)


class TriSandboxBackend:
    """Executes isolated coding rollouts and grading via a `shapens/tri` HTTP service.

    When `use_batch=True` (default), collapses the full workspace lifecycle into a single
    `/v0/batch` HTTP call using `tri`'s atomic `files write-batch` transaction:
      1. `workspace create <project> --from <base_ref>` (copy-on-write snapshot fork)
      2. `files write-batch ${ws.data.id} --entries-json <json>` (single snapshot + tx)
      3. `run ${ws.data.id} --no-import-changes [--sandbox] [--runner remote] -- <cmd...>`
      4. `workspace discard ${ws.data.id}` (with fallback discard if batch aborts early)
    """

    def __init__(
        self,
        tri_remote: str,
        *,
        project: str = "loop",
        base_ref: str = "main",
        token: str | None = None,
        runner: str | None = None,
        require_sandbox: bool = False,
        use_batch: bool = True,
        http_timeout_s: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.tri_remote = tri_remote.rstrip("/")
        self.project = project
        self.base_ref = base_ref
        self.token = token
        self.runner = runner
        self.require_sandbox = require_sandbox
        self.use_batch = use_batch
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.Client(
            timeout=http_timeout_s,
            headers=headers,
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _command(self, argv: list[str]) -> dict[str, Any]:
        resp = self._client.post(f"{self.tri_remote}/v0/command", json={"argv": argv})
        resp.raise_for_status()
        env: dict[str, Any] = resp.json()
        if not env.get("ok"):
            diags = env.get("diagnostics") or []
            msg = diags[0].get("message") if diags else "unknown tri error"
            raise RuntimeError(f"tri command {argv[:2]} failed: {msg}")
        data: dict[str, Any] = env.get("data", {})
        return data

    def _batch(self, commands: list[dict[str, Any]]) -> dict[str, Any]:
        resp = self._client.post(f"{self.tri_remote}/v0/batch", json={"commands": commands})
        resp.raise_for_status()
        env: dict[str, Any] = resp.json()
        return env

    def _create_workspace(self) -> str:
        data = self._command(["workspace", "create", self.project, "--from", self.base_ref])
        return str(data["id"])

    def _write_files(self, ws_id: str, files: dict[str, str]) -> None:
        for rel, content in sorted(files.items()):
            b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
            self._command(["files", "write", ws_id, rel, "--base64", b64])

    def _discard_workspace(self, ws_id: str) -> None:
        with contextlib.suppress(Exception):
            self._command(["workspace", "discard", ws_id])

    def _build_run_argv(self, ws_ref: str, command: list[str], timeout_seconds: float) -> list[str]:
        argv = [
            "run",
            ws_ref,
            "--no-import-changes",
            "--no-env",
            "--timeout",
            str(timeout_seconds),
        ]
        if self.require_sandbox:
            argv.append("--sandbox")
        if self.runner:
            argv.extend(["--runner", self.runner])
        argv.append("--")
        argv.extend(command)
        return argv

    def _run_command_batched(
        self, workspace_files: dict[str, str], command: list[str], timeout_seconds: float
    ) -> tuple[int, str, str, float]:
        t0 = time.perf_counter()
        commands: list[dict[str, Any]] = [
            {"as": "ws", "argv": ["workspace", "create", self.project, "--from", self.base_ref]},
        ]
        if workspace_files:
            entries_payload = [
                {
                    "path": rel,
                    "base64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                }
                for rel, content in sorted(workspace_files.items())
            ]
            commands.append(
                {
                    "argv": [
                        "files",
                        "write-batch",
                        "${ws.data.id}",
                        "--entries-json",
                        json.dumps(entries_payload, separators=(",", ":")),
                    ]
                }
            )
        run_idx = len(commands)
        commands.append(
            {
                "as": "run",
                "argv": self._build_run_argv("${ws.data.id}", command, timeout_seconds),
            }
        )
        stdout_idx = len(commands)
        commands.append(
            {
                "as": "stdout",
                "argv": [
                    "checks",
                    "stream",
                    "${run.data.id}",
                    "--log",
                    "stdout",
                    "--limit",
                    "65536",
                ],
            }
        )
        stderr_idx = len(commands)
        commands.append(
            {
                "as": "stderr",
                "argv": [
                    "checks",
                    "stream",
                    "${run.data.id}",
                    "--log",
                    "stderr",
                    "--limit",
                    "65536",
                ],
            }
        )
        discard_idx = len(commands)
        commands.append({"argv": ["workspace", "discard", "${ws.data.id}"]})

        env = self._batch(commands)
        results: list[dict[str, Any]] = env.get("data", {}).get("results", [])
        created_ws_id: str | None = None
        if results and results[0].get("ok"):
            raw_id = results[0].get("data", {}).get("id")
            if raw_id:
                created_ws_id = str(raw_id)

        discarded_in_batch = len(results) > discard_idx and bool(results[discard_idx].get("ok"))
        try:
            if not env.get("ok") or len(results) <= run_idx or not results[run_idx].get("ok"):
                diags = env.get("diagnostics") or [
                    d for item in results for d in item.get("diagnostics", [])
                ]
                msg = diags[0].get("message") if diags else "unknown tri batch error"
                raise RuntimeError(f"tri batch failed: {msg}")
            data: dict[str, Any] = results[run_idx].get("data", {})
            logs = data.get("logs") or data.get("outputs", {}).get("logs") or {}
            stdout = (
                str(results[stdout_idx].get("data", {}).get("text", ""))
                if len(results) > stdout_idx and results[stdout_idx].get("ok")
                else str(logs.get("stdout", ""))
            )
            stderr = (
                str(results[stderr_idx].get("data", {}).get("text", ""))
                if len(results) > stderr_idx and results[stderr_idx].get("ok")
                else str(logs.get("stderr", ""))
            )
            exit_status = data.get("exit_status")
            rc = int(exit_status) if exit_status is not None else 1
            return rc, stdout, stderr, (time.perf_counter() - t0) * 1000.0
        finally:
            if created_ws_id and not discarded_in_batch:
                self._discard_workspace(created_ws_id)

    def _resolve_log_text(self, check_id: str | None, stream: str, raw_val: str) -> str:
        if check_id and raw_val.startswith("artifact_"):
            streamed = self._command(
                ["checks", "stream", check_id, "--log", stream, "--limit", "65536"]
            )
            return str(streamed.get("text", ""))
        return raw_val

    def run_command(
        self, workspace_files: dict[str, str], command: list[str], timeout_seconds: float
    ) -> tuple[int, str, str, float]:
        if self.use_batch:
            return self._run_command_batched(workspace_files, command, timeout_seconds)
        t0 = time.perf_counter()
        ws_id = self._create_workspace()
        try:
            self._write_files(ws_id, workspace_files)
            argv = self._build_run_argv(ws_id, command, timeout_seconds)
            data = self._command(argv)
            logs = data.get("logs") or data.get("outputs", {}).get("logs") or {}
            check_id = str(data["id"]) if data.get("id") else None
            stdout = self._resolve_log_text(check_id, "stdout", str(logs.get("stdout", "")))
            stderr = self._resolve_log_text(check_id, "stderr", str(logs.get("stderr", "")))
            exit_status = data.get("exit_status")
            rc = int(exit_status) if exit_status is not None else 1
            return rc, stdout, stderr, (time.perf_counter() - t0) * 1000.0
        finally:
            self._discard_workspace(ws_id)

    def grade(
        self,
        task: TaskSpec,
        workspace_files: dict[str, str],
        mode: GraderMode = "hardened",
        reward_mode: RewardMode = "binary",
    ) -> VerifierReceipt:
        t0 = time.perf_counter()
        digest = manifest_sha256(task)
        visible_tampered, unexpected_edits = _detect_tampering(task, workspace_files)

        if mode == "hardened" and (visible_tampered or unexpected_edits):
            return VerifierReceipt(
                task_id=task.task_id,
                passed=False,
                reward=0.0,
                grader_mode=mode,
                visible_tests_tampered=visible_tampered,
                unexpected_edits=unexpected_edits,
                timed_out=False,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                stdout="",
                stderr=(
                    f"anti-cheat rejected edits: tampered={visible_tampered}, "
                    f"unexpected={unexpected_edits}"
                ),
                manifest_sha256=digest,
            )

        overlay = dict(workspace_files)
        if mode == "hardened":
            overlay["_loop_hidden/__init__.py"] = ""
            for rel, content in task.repo_files.items():
                name = Path(rel).name
                if rel.startswith("tests/") and rel.endswith(".py") and name.startswith("test_"):
                    overlay[f"_loop_hidden/test_pristine_{name}"] = content
            for name, content in task.hidden_tests.items():
                overlay[f"_loop_hidden/{name}"] = content
            start_dir = "_loop_hidden"
        else:
            start_dir = "tests"

        cmd = [
            "python3",
            "-I",
            "-S",
            "-m",
            "unittest",
            "discover",
            "-s",
            start_dir,
            "-t",
            ".",
            "-v",
        ]
        rc, out, err, elapsed_ms = self.run_command(overlay, cmd, task.timeout_seconds)
        timed_out = rc == 124 or "timed out" in err.lower()
        passed = (not timed_out) and rc == 0 and "Ran 0 tests" not in err
        sub_passed, sub_total = (0, 0) if timed_out else parse_unittest_counts(err)
        reward = (
            0.0
            if timed_out
            else compute_shaped_reward(
                passed=passed,
                subtests_passed=sub_passed,
                subtests_total=sub_total,
                reward_mode=reward_mode,
            )
        )
        return VerifierReceipt(
            task_id=task.task_id,
            passed=passed,
            reward=reward,
            grader_mode=mode,
            visible_tests_tampered=visible_tampered,
            unexpected_edits=unexpected_edits,
            timed_out=timed_out,
            duration_ms=elapsed_ms,
            stdout=out,
            stderr=err,
            manifest_sha256=digest,
            subtests_passed=sub_passed,
            subtests_total=sub_total,
        )
