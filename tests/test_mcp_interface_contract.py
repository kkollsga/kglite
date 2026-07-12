"""Exact MCP tools/list contracts for each public server mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_mcp_server_smoke import BINARY, _build_fixture_graph, _spawn

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "api-baselines" / "mcp-tools.json"

pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="release kglite-mcp-server binary not built")


def _tool_contract(client) -> list[dict]:
    tools = client.list_tools()
    return sorted(
        (
            {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "inputSchema": tool.get("inputSchema") or {},
            }
            for tool in tools
        ),
        key=lambda tool: tool["name"],
    )


def _capture(args: list[str]) -> list[dict]:
    client = _spawn(args, env_remove=["GITHUB_TOKEN", "GH_TOKEN"])
    try:
        return _tool_contract(client)
    finally:
        client.shutdown()


def capture_mcp_contract(base: Path) -> dict[str, list[dict]]:
    base.mkdir(parents=True, exist_ok=True)
    graph = base / "fixture.kgl"
    _build_fixture_graph(graph)

    local_root = base / "local-root"
    local_root.mkdir()
    (local_root / "demo.py").write_text("print('hello')\n")
    local_manifest = base / "local_mcp.yaml"
    local_manifest.write_text(f"name: Local Contract\nworkspace:\n  kind: local\n  root: {local_root}\n")

    custom_manifest = base / "custom_mcp.yaml"
    custom_manifest.write_text(
        "name: Custom Contract\n"
        "tools:\n"
        "  - name: people_in_city\n"
        "    description: Find people in one city.\n"
        '    cypher: "MATCH (p:Person {city: $city}) RETURN p.title AS name"\n'
        "    parameters:\n"
        "      city:\n"
        "        type: string\n"
        "        description: Exact city name.\n"
        "        required: true\n"
    )

    return {
        "graph_readonly": _capture(["--graph", str(graph)]),
        "graph_writable": _capture(["--graph", str(graph), "--writable"]),
        "local_workspace": _capture(["--mcp-config", str(local_manifest)]),
        "manifest_tool": _capture(["--graph", str(graph), "--mcp-config", str(custom_manifest)]),
    }


@pytest.fixture(scope="module")
def mcp_contract(tmp_path_factory):
    return capture_mcp_contract(tmp_path_factory.mktemp("mcp-interface"))


def test_mcp_tools_list_matches_reviewed_mode_schemas(mcp_contract):
    expected = json.loads(BASELINE.read_text())
    assert mcp_contract == expected, (
        "MCP tool names/descriptions/input schemas drifted; review and refresh mcp-tools.json"
    )


def test_local_workspace_has_one_activation_tool(mcp_contract):
    tools = mcp_contract["local_workspace"]
    names = {tool["name"] for tool in tools}
    assert "set_root_dir" in names
    assert "repo_management" not in names
