"""Executable and static contracts for docs/operators/agent-responses.md.

Phase 5 executable documentation contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from tests.conftest import binary_skip_reason, workspace_binary

ROOT = Path(__file__).resolve().parents[1]
BINARY = workspace_binary("kglite")
SKIP_REASON = binary_skip_reason("kglite CLI", BINARY, "cargo build -p kglite-cli")


def _run(args: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(BINARY), *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=30)


def test_agent_response_guide_names_the_implemented_contract() -> None:
    guide = (ROOT / "docs/operators/agent-responses.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in (
        "16,384",
        "4,096",
        "32 entries",
        "32 MiB",
        "ten minutes",
        "kglite response expand",
        "kglite response purge --all",
        "database-wide population is reported as `unknown`",
        "does not open the graph",
        "does not infer semantic groups",
    ):
        assert text in guide, text
    assert "--format agent" in readme
    assert "operators/agent-responses.html" in readme
    assert "Do not hard-code the expansion tool name or result ID" in guide
    assert "next.selected_value" in guide
    assert "json_pointer" in guide


@pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "")
def test_documented_cli_preview_generated_command_full_and_purge(tmp_path, monkeypatch) -> None:
    import kglite

    graph_path = tmp_path / "docs-agent.kgl"
    graph = kglite.KnowledgeGraph()
    body = "é" * 80
    graph.cypher(f"UNWIND range(0, 999) AS i CREATE (:Evidence {{id:i, body:'{body}'}})")
    graph.save(str(graph_path))
    cache = tmp_path / "cache"
    monkeypatch.setenv("KGLITE_AGENT_CACHE_DIR", str(cache))
    env = os.environ.copy()

    preview = _run(
        [
            "query",
            str(graph_path),
            "MATCH (n:Evidence) RETURN n.id ORDER BY n.id",
            "--format",
            "agent",
            "--response-max-bytes",
            "4096",
        ],
        cwd=tmp_path,
        env=env,
    )
    assert preview.returncode == 0, preview.stderr
    assert preview.stderr == ""
    assert len(preview.stdout.rstrip("\n").encode()) <= 4096
    value = json.loads(preview.stdout)
    budget = json.loads(value["content"][0]["text"])["response_budget"]
    command = next(entry["command"] for entry in budget["domain_commands"] if entry["json_pointer"] == "/rows")

    graph_path.unlink()
    args = shlex.split(command)
    assert args[:3] == ["kglite", "response", "expand"]
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    expanded = _run(args[1:], cwd=alternate, env=env)
    assert expanded.returncode == 0, expanded.stderr
    assert expanded.stderr == ""
    assert "rows" in expanded.stdout

    full = _run(
        ["response", "expand", budget["result_id"], "--response-full"],
        cwd=tmp_path,
        env=env,
    )
    assert full.returncode == 0, full.stderr
    restored = json.loads(full.stdout)["structuredContent"]
    assert len(restored["rows"]) == 1000
    assert restored["rows"][999][0] == 999

    purged = _run(["response", "purge", "--all"], cwd=tmp_path, env=env)
    assert purged.returncode == 0
    assert json.loads(purged.stdout) == {"ok": True, "purged": "all"}
