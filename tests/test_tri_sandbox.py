from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from loop.envs.generator import generate_split
from loop.envs.sandbox import LocalSubprocessSandbox, TriSandboxBackend


def test_tri_sandbox_backend_protocol_and_anti_cheat() -> None:
    """Verify TriSandboxBackend against an in-memory mock of tri's `/v0/command` HTTP API."""
    workspaces: dict[str, dict[str, str]] = {}
    discarded: list[str] = []
    seen_commands: list[list[str]] = []
    local = LocalSubprocessSandbox()
    ws_seq = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal ws_seq
        assert request.url.path == "/v0/command"
        assert request.headers.get("Authorization") == "Bearer tri_secret"
        payload: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        argv: list[str] = payload["argv"]
        seen_commands.append(argv)

        if argv[:2] == ["workspace", "create"]:
            ws_seq += 1
            ws_id = f"ws_test_{ws_seq}"
            workspaces[ws_id] = {}
            return httpx.Response(200, json={"ok": True, "data": {"id": ws_id}})

        if argv[:2] == ["files", "write"]:
            ws_id, path, flag, b64 = argv[2], argv[3], argv[4], argv[5]
            assert flag == "--base64"
            workspaces[ws_id][path] = base64.b64decode(b64).decode("utf-8")
            return httpx.Response(200, json={"ok": True, "data": {"path": path}})

        if argv[0] == "run":
            ws_id = argv[1]
            sep = argv.index("--")
            cmd = argv[sep + 1 :]
            rc, out, err, ms = local.run_command(workspaces[ws_id], cmd, timeout_seconds=10.0)
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": {
                        "exit_status": rc,
                        "duration_ms": ms,
                        "logs": {"stdout": out, "stderr": err},
                    },
                },
            )

        if argv[:2] == ["workspace", "discard"]:
            ws_id = argv[2]
            discarded.append(ws_id)
            workspaces.pop(ws_id, None)
            return httpx.Response(200, json={"ok": True, "data": {"id": ws_id}})

        return httpx.Response(400, json={"ok": False, "diagnostics": [{"message": "bad argv"}]})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer tri_secret"},
    )
    tri_sb = TriSandboxBackend(
        "http://tri.local",
        project="loop",
        base_ref="main",
        token="tri_secret",
        require_sandbox=True,
        client=client,
    )

    task = generate_split("eval", 1)[0]

    # 1. Untouched workspace fails cleanly and discards workspace.
    untouched = tri_sb.grade(task, dict(task.repo_files), mode="hardened")
    assert not untouched.passed
    assert untouched.reward == 0.0
    assert len(discarded) == 1
    assert any("--sandbox" in argv for argv in seen_commands)

    # 2. Reference solution passes via tri isolated overlay and discards workspace.
    solved = dict(task.repo_files)
    solved.update(task.solution_files)
    res_ok = tri_sb.grade(task, solved, mode="hardened")
    assert res_ok.passed
    assert res_ok.reward == 1.0
    assert len(discarded) == 2

    # 3. Tampering with visible tests is rejected before even creating a workspace.
    hacked = dict(task.repo_files)
    hacked["tests/test_visible.py"] = (
        "import unittest\nclass H(unittest.TestCase):\n    def test_ok(self): pass\n"
    )
    res_hack = tri_sb.grade(task, hacked, mode="hardened")
    assert not res_hack.passed
    assert res_hack.visible_tests_tampered
    assert len(discarded) == 2  # rejected locally before remote roundtrip
