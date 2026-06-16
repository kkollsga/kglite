"""MCP (Python entry) server: explore tool + orchestration/views skills +
references_tools injection (operator proposal 2026-06-16; §2.1/2.2/2.3/2.4/4.1/4.4).

Drives the Python server (`python -m kglite.mcp_server`) over stdio with the
generic JSON-RPC client from the smoke suite, so it runs anywhere the server's
runtime deps are present (no embedder / fastembed needed).
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("mcp")
pytest.importorskip("yaml")

from tests.test_mcp_server_smoke import McpClient, _text_content  # noqa: E402


def _spawn_py(args: list[str]) -> McpClient:
    proc = subprocess.Popen(
        [sys.executable, "-m", "kglite.mcp_server", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    client = McpClient(proc)
    client.initialize()
    return client


def _code_graph_kgl(tmp_path: Path) -> Path:
    from kglite import code_tree

    src = tmp_path / "tiny_code"
    src.mkdir()
    (src / "main.py").write_text("def hub():\n    return leaf()\n\ndef leaf():\n    return 1\n\nclass Bar:\n    pass\n")
    g = code_tree.build(str(src))
    kgl = tmp_path / "tiny_code.kgl"
    g.save(str(kgl))
    return kgl


def _manifest(tmp_path: Path, name: str) -> Path:
    m = tmp_path / "m.yaml"
    m.write_text(f"name: {name}\nskills: true\n")
    return m


def _tools_by_name(client: McpClient) -> dict[str, str]:
    return {t["name"]: (t.get("description") or "") for t in client.list_tools()}


def test_explore_tool_registered_and_callable(tmp_path: Path) -> None:
    kgl = _code_graph_kgl(tmp_path)
    client = _spawn_py(["--graph", str(kgl), "--mcp-config", str(_manifest(tmp_path, "explore_test"))])
    try:
        assert "explore" in _tools_by_name(client)
        out = _text_content(client.call_tool("explore", {"query": "hub"}))
    finally:
        client.shutdown()
    assert "Entry points" in out, out
    assert "hub" in out


def test_explore_skill_discovered_not_orphaned(tmp_path: Path) -> None:
    """4.4: explore.md is now auto-discovered (was orphaned by the hardcoded
    allowlist) — its methodology injects into the explore tool description on
    a code graph."""
    kgl = _code_graph_kgl(tmp_path)
    client = _spawn_py(["--graph", str(kgl), "--mcp-config", str(_manifest(tmp_path, "explore_skill"))])
    try:
        desc = _tools_by_name(client)["explore"]
    finally:
        client.shutdown()
    assert "kglite-skill:explore" in desc
    assert "## Methodology" in desc


def test_cross_tool_skills_inject_via_references_tools(tmp_path: Path) -> None:
    """2.1/2.3/4.1: skills named after no tool reach cypher_query /
    graph_overview / explore through references_tools, and the TRIGGER/SKIP
    routing rides the live tool-description channel under '## When to use'."""
    kgl = _code_graph_kgl(tmp_path)
    client = _spawn_py(["--graph", str(kgl), "--mcp-config", str(_manifest(tmp_path, "orch"))])
    try:
        tools = _tools_by_name(client)
    finally:
        client.shutdown()

    for tool in ("cypher_query", "graph_overview", "explore"):
        desc = tools[tool]
        assert "kglite-skill:code_graph_analysis" in desc, f"{tool} missing orchestration skill"
        assert "## When to use" in desc, f"{tool} missing TRIGGER/SKIP routing (4.1)"
        assert "Never grep to discover" in desc, f"{tool} missing routing/body text"

    # code_graph_views attaches to cypher_query and teaches provenance filters.
    assert "kglite-skill:code_graph_views" in tools["cypher_query"]
    assert "is_benchmark" in tools["cypher_query"]


def test_code_graph_skills_silent_on_non_code_graph(tmp_path: Path) -> None:
    """applies_when [Function, Class] gating: on a graph with no code nodes the
    orchestration/views skills are inactive and don't pollute descriptions."""
    from tests.fixtures.build_tiny_graph import build_tiny_graph

    kgl = tmp_path / "domain.kgl"
    build_tiny_graph(kgl)  # Person/Article/etc — no Function/Class
    client = _spawn_py(["--graph", str(kgl), "--mcp-config", str(_manifest(tmp_path, "domain"))])
    try:
        tools = _tools_by_name(client)
    finally:
        client.shutdown()
    assert "kglite-skill:code_graph_analysis" not in tools.get("cypher_query", "")
    assert "kglite-skill:code_graph_views" not in tools.get("cypher_query", "")
