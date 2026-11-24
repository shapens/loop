from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from loop.envs.generator import generate_split
from loop.envs.sandbox import LocalSubprocessSandbox, TriSandboxBackend


def test_tri_sandbox_backend_batched_and_legacy_and_anti_cheat() -> None:
    """Verify TriSandboxBackend (both `/v0/batch` and `/v0/command`) against an in-memory mock."""
    workspaces: dict[str, dict[str, str]] = {}
    check_logs: dict[str, dict[str, str]] = {}
    discarded: list[str] = []
    seen_commands: list[list[str]] = []
    batch_calls = 0
    local = LocalSubprocessSandbox()
    ws_seq = 0
    chk_seq = 0

    def exec_one(argv: list[str]) -> dict[str, Any]:
        nonlocal ws_seq, chk_seq
        seen_commands.append(argv)

        if argv[:2] == ["workspace", "create"]:
            ws_seq += 1
            ws_id = f"ws_test_{ws_seq}"
            workspaces[ws_id] = {}
            return {"ok": True, "data": {"id": ws_id}}

        if argv[:2] == ["files", "write"]:
            ws_id, path, flag, b64 = argv[2], argv[3], argv[4], argv[5]
            assert flag == "--base64"
            workspaces[ws_id][path] = base64.b64decode(b64).decode("utf-8")
            return {"ok": True, "data": {"path": path}}

        if argv[:2] == ["files", "write-batch"]:
            ws_id, flag, raw_json = argv[2], argv[3], argv[4]
            assert flag == "--entries-json"
            entries = json.loads(raw_json)
            paths: list[str] = []
            for item in entries:
                path = item["path"]
                workspaces[ws_id][path] = base64.b64decode(item["base64"]).decode("utf-8")
                paths.append(path)
            return {"ok": True, "data": {"operation": "op_1", "snapshot": "snap_1", "paths": paths}}

        if argv[0] == "run":
            ws_id = argv[1]
            sep = argv.index("--")
            cmd = argv[sep + 1 :]
            rc, out, err, ms = local.run_command(workspaces[ws_id], cmd, timeout_seconds=10.0)
            chk_seq += 1
            chk_id = f"check_test_{chk_seq}"
            check_logs[chk_id] = {"stdout": out, "stderr": err}
            return {
                "ok": True,
                "data": {
                    "id": chk_id,
                    "exit_status": rc,
                    "duration_ms": ms,
                    "logs": {
                        "stdout": f"artifact_out_{chk_seq}",
                        "stderr": f"artifact_err_{chk_seq}",
                    },
                },
            }

        if argv[:2] == ["checks", "stream"]:
            chk_id = argv[2]
            stream = argv[argv.index("--log") + 1]
            return {"ok": True, "data": {"text": check_logs[chk_id][stream]}}

        if argv[:2] == ["workspace", "discard"]:
            ws_id = argv[2]
            discarded.append(ws_id)
            workspaces.pop(ws_id, None)
            return {"ok": True, "data": {"id": ws_id}}

        return {"ok": False, "diagnostics": [{"message": f"bad argv: {argv[:2]}"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal batch_calls
        assert request.headers.get("Authorization") == "Bearer tri_secret"
        payload: dict[str, Any] = json.loads(request.content.decode("utf-8"))

        if request.url.path == "/v0/command":
            res = exec_one(payload["argv"])
            return httpx.Response(200, json=res)

        if request.url.path == "/v0/batch":
            batch_calls += 1
            bindings: dict[str, Any] = {}
            results: list[dict[str, Any]] = []
            for item in payload["commands"]:
                ws_val = bindings.get("ws", {}).get("data", {}).get("id", "")
                run_val = bindings.get("run", {}).get("data", {}).get("id", "")
                argv = [
                    part.replace("${ws.data.id}", ws_val).replace("${run.data.id}", run_val)
                    for part in item["argv"]
                ]
                out = exec_one(argv)
                if item.get("as"):
                    bindings[item["as"]] = out
                results.append(out)
                if not out.get("ok"):
                    break
            ok = all(r.get("ok") for r in results) and len(results) == len(payload["commands"])
            return httpx.Response(200, json={"ok": ok, "data": {"results": results}})

        return httpx.Response(404, json={"ok": False})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer tri_secret"},
    )
    tri_batched = TriSandboxBackend(
        "http://tri.local",
        project="loop",
        base_ref="main",
        token="tri_secret",
        require_sandbox=True,
        use_batch=True,
        client=client,
    )

    task = generate_split("eval", 1)[0]

    # 1. Batched untouched workspace fails cleanly in 1 HTTP call and discards workspace.
    untouched = tri_batched.grade(task, dict(task.repo_files), mode="hardened")
    assert not untouched.passed
    assert untouched.reward == 0.0
    assert batch_calls == 1
    assert len(discarded) == 1
    assert any(argv[:2] == ["files", "write-batch"] for argv in seen_commands)
    assert any("--sandbox" in argv for argv in seen_commands)

    # 2. Batched reference solution passes and discards workspace in 1 HTTP call.
    solved = dict(task.repo_files)
    solved.update(task.solution_files)
    res_ok = tri_batched.grade(task, solved, mode="hardened")
    assert res_ok.passed
    assert res_ok.reward == 1.0
    assert batch_calls == 2
    assert len(discarded) == 2

    # 3. Legacy non-batched path still works identically.
    tri_legacy = TriSandboxBackend(
        "http://tri.local",
        project="loop",
        base_ref="main",
        token="tri_secret",
        use_batch=False,
        client=client,
    )
    res_legacy = tri_legacy.grade(task, solved, mode="hardened")
    assert res_legacy.passed and res_legacy.reward == 1.0
    assert len(discarded) == 3

    # 4. Tampering with visible tests is rejected locally before any remote roundtrip.
    hacked = dict(task.repo_files)
    hacked["tests/test_visible.py"] = (
        "import unittest\nclass H(unittest.TestCase):\n    def test_ok(self): pass\n"
    )
    res_hack = tri_batched.grade(task, hacked, mode="hardened", reward_mode="subtest")
    assert not res_hack.passed
    assert res_hack.reward == 0.0
    assert res_hack.visible_tests_tampered
    assert batch_calls == 2
    assert len(discarded) == 3
