from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from loop.envs.grader import grade_workspace
from loop.types import GraderMode, TaskSpec, VerifierReceipt


class SandboxBackend(Protocol):
    """Pluggable execution/grading substrate (Local, Modal Sandboxes, or shapens/tri)."""

    def run_command(
        self, workspace_files: dict[str, str], command: list[str], timeout_seconds: float
    ) -> tuple[int, str, str, float]: ...

    def grade(
        self, task: TaskSpec, workspace_files: dict[str, str], mode: GraderMode = "hardened"
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
        self, task: TaskSpec, workspace_files: dict[str, str], mode: GraderMode = "hardened"
    ) -> VerifierReceipt:
        return grade_workspace(task, workspace_files, mode=mode)


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
        self, task: TaskSpec, workspace_files: dict[str, str], mode: GraderMode = "hardened"
    ) -> VerifierReceipt:
        # Delegates grading through the Modal app's remote grader function when deployed;
        # falls back cleanly to local isolated grading in dev/CI.
        from loop.modal_app import grade_remote

        return grade_remote(task, workspace_files, mode=mode)


class TriSandboxBackend:
    """Milestone 4 seam: delegates isolated copy-on-write workspaces to shapens/tri."""

    def __init__(self, tri_remote: str) -> None:
        self.tri_remote = tri_remote

    def run_command(
        self, workspace_files: dict[str, str], command: list[str], timeout_seconds: float
    ) -> tuple[int, str, str, float]:
        raise NotImplementedError(
            "Milestone 4: wire shapens/tri HTTP workspace create/edit/run/teardown"
        )

    def grade(
        self, task: TaskSpec, workspace_files: dict[str, str], mode: GraderMode = "hardened"
    ) -> VerifierReceipt:
        raise NotImplementedError("Milestone 4: wire shapens/tri gVisor/Kata runner pool grading")
