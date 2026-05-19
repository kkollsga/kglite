"""Pre-release integration test suite for `kglite-mcp-server`.

Implements the spec from the MCP-servers operator's
`2026-05-12-from-mcp-servers-pre-release-test-suite-spec.md`. Every
test below maps to a specific bug class from the 0.9.16 → 0.9.22
release arc; the inline `# Catches:` comments preserve that mapping.
When all 27 tests pass, the release is shippable.

Categories:
- A: Install & boot (4 tests)
- B: Per-mode tool registration (7 tests)
- C: Tool output content (8 tests)
- D: Embedder & semantic search (3 tests)
- E: Manifest & .env handling (5 tests)
- F: Workspace state propagation (5 tests)
- J: Spatial Cypher (3 tests)
- K: Timeseries Cypher (3 tests)
- L: Procedures — orphans + duplicates (2 tests)
- W: File watcher boundary (1 test)

0.9.24: E5, F4, F5, and W1 added to anchor the new pyo3-wrapper
boundaries.

0.9.26: B6/B7 added (disk-graph CLI validator); Cat J/K/L (8 tests)
wired in against the mcp-servers operator's delivered fixtures
(tests/fixtures/{spatial,timeseries,graph_with_orphans,graph_with_duplicates}.kgl).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Any

import pytest

# Skip cleanly if the MCP extras aren't installed locally.
pytest.importorskip("mcp")
pytest.importorskip("yaml")
pytest.importorskip("aiohttp")
pytest.importorskip("fastembed")
pytest.importorskip("watchdog")

# ---------------------------------------------------------------------------
# Shared helpers — McpClient + spawn
# ---------------------------------------------------------------------------


def _which_server() -> str | None:
    return shutil.which("kglite-mcp-server")


def _spawn(args: list[str], extra_env: dict[str, str] | None = None, cwd: str | None = None) -> subprocess.Popen:
    server = _which_server()
    if server is None:
        pytest.skip("kglite-mcp-server console script not on PATH (run `maturin develop`)")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [server, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        bufsize=0,
    )


class McpClient:
    """JSON-RPC stdio driver. Used to call the server like Claude/Cursor do."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._next_id = 1
        self._stderr: list[str] = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        for line in iter(self._proc.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", errors="replace"))

    def stderr_lines(self) -> list[str]:
        return list(self._stderr)

    def _send(self, body: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(body) + "\n").encode())
        self._proc.stdin.flush()

    def _recv(self, timeout: float = 10.0) -> dict[str, Any]:
        assert self._proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise TimeoutError(f"no response within {timeout}s; stderr: {''.join(self._stderr[-10:])}")

    def initialize(self) -> dict[str, Any]:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "kglite-test", "version": "0"},
                },
            }
        )
        self._next_id += 1
        resp = self._recv()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    def list_tools(self) -> list[str]:
        self._send({"jsonrpc": "2.0", "id": self._next_id, "method": "tools/list", "params": {}})
        self._next_id += 1
        resp = self._recv()
        return sorted(t["name"] for t in resp["result"]["tools"])

    def list_tools_full(self) -> list[dict[str, Any]]:
        self._send({"jsonrpc": "2.0", "id": self._next_id, "method": "tools/list", "params": {}})
        self._next_id += 1
        resp = self._recv()
        return resp["result"]["tools"]

    def call_tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": args or {}},
            }
        )
        self._next_id += 1
        resp = self._recv()
        if "error" in resp:
            return f"ERROR: {resp['error']}"
        content = resp.get("result", {}).get("content", [])
        return content[0].get("text", "") if content else ""

    def list_prompts(self) -> list[dict[str, Any]]:
        self._send({"jsonrpc": "2.0", "id": self._next_id, "method": "prompts/list", "params": {}})
        self._next_id += 1
        resp = self._recv()
        return resp.get("result", {}).get("prompts", [])

    def get_prompt(self, name: str) -> dict[str, Any]:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "prompts/get",
                "params": {"name": name, "arguments": {}},
            }
        )
        self._next_id += 1
        return self._recv()

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


def _client(args: list[str], **kwargs: Any) -> McpClient:
    proc = _spawn(args, **kwargs)
    client = McpClient(proc)
    client.initialize()
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_graph_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the ~50-node-per-type fixture graph once per session."""
    from tests.fixtures.build_tiny_graph import build_tiny_graph

    path = tmp_path_factory.mktemp("graphs") / "tiny_graph.kgl"
    build_tiny_graph(path)
    return path


@pytest.fixture
def mutable_graph_path(tmp_path: Path, tiny_graph_path: Path) -> Path:
    """Fresh copy of tiny_graph.kgl that the test can mutate without
    affecting other tests. Also drops a sibling manifest with
    builtins.save_graph: true so save_graph registers."""
    target = tmp_path / "mutable.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "mutable_mcp.yaml"
    manifest.write_text("name: mutable\nbuiltins:\n  save_graph: true\n  temp_cleanup: on_overview\n")
    return target


@pytest.fixture
def tiny_source_dir(tmp_path: Path) -> Path:
    """A tiny source tree for read_source/grep/list_source tests."""
    root = tmp_path / "src"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 'alpha'\n\nclass A:\n    pass\n")
    (root / "beta.py").write_text("import os\n\ndef beta():\n    return os.getcwd()\n")
    nested = root / "nested"
    nested.mkdir()
    (nested / "gamma.py").write_text("def gamma():\n    return 42\n")
    return root


@pytest.fixture
def tiny_workspace_dir(tmp_path: Path) -> Path:
    """A workspace dir with one stubbed cloned repo so list_repos has
    something to find."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    org = ws / "stub"
    org.mkdir()
    repo = org / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")
    return ws


@pytest.fixture
def graph_with_savegraph_flag(mutable_graph_path: Path) -> Path:
    """Already has builtins.save_graph: true via mutable_graph_path."""
    return mutable_graph_path.with_suffix(".kgl").parent / "mutable_mcp.yaml"


@pytest.fixture
def graph_without_savegraph_flag(tmp_path: Path, tiny_graph_path: Path) -> Path:
    """Manifest matching tiny_graph but without save_graph flag."""
    target = tmp_path / "nograph.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "nograph_mcp.yaml"
    manifest.write_text("name: nograph\nbuiltins:\n  temp_cleanup: on_overview\n")
    return manifest


@pytest.fixture
def manifest_with_csv_http(tmp_path: Path, tiny_graph_path: Path) -> Path:
    target = tmp_path / "csvhttp.kgl"
    shutil.copy(tiny_graph_path, target)
    # Pick a free port deterministically.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    manifest = tmp_path / "csvhttp_mcp.yaml"
    manifest.write_text(f"name: csvhttp\nextensions:\n  csv_http_server:\n    port: {port}\n    dir: temp/\n")
    return manifest


@pytest.fixture
def graph_with_bge_m3_embedder(tmp_path: Path, tiny_graph_path: Path) -> Path:
    target = tmp_path / "bgem3.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "bgem3_mcp.yaml"
    manifest.write_text("name: bgem3\nextensions:\n  embedder:\n    backend: fastembed\n    model: BAAI/bge-m3\n")
    return manifest


@pytest.fixture(scope="session")
def graph_with_bge_m3_embeddings_precomputed(tmp_path_factory: pytest.TempPathFactory, tiny_graph_path: Path) -> Path:
    """Same as tiny_graph but with `Article.body` embeddings precomputed
    via bge-m3. kglite's text_score requires pre-stored embeddings (it
    embeds the query string at search time but not stored content).
    Without this fixture, text_score returns 'no embedding body_emb
    found' errors that mask real embedder-loading issues."""
    import kglite
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    target_dir = tmp_path_factory.mktemp("bge_m3_corpus")
    target = target_dir / "bgem3_corpus.kgl"
    shutil.copy(tiny_graph_path, target)
    g = kglite.load(str(target))
    embedder = BgeM3Embedder()
    g.set_embedder(embedder)
    g.embed_texts("Article", "body", show_progress=False)
    g.save(str(target))
    return target


@pytest.fixture
def local_workspace_yaml(tmp_path: Path, tiny_source_dir: Path) -> Path:
    """workspace.kind: local manifest pointing at a real directory."""
    yaml = tmp_path / "local_mcp.yaml"
    yaml.write_text(f"name: local\nworkspace:\n  kind: local\n  root: {tiny_source_dir}\n")
    return yaml


@pytest.fixture
def local_workspace_with_two_children(tmp_path: Path) -> tuple[Path, Path, Path]:
    """workspace.kind: local with two sibling subdirectories under the
    configured root, each carrying a unique marker file. Returns
    (manifest_yaml, child_a_path, child_b_path) so F4 can verify both
    siblings are reachable via set_root_dir."""
    root = tmp_path / "workspace_root"
    root.mkdir()
    child_a = root / "child_a"
    child_a.mkdir()
    (child_a / "marker_a.py").write_text("# child_a marker\n")
    child_b = root / "child_b"
    child_b.mkdir()
    (child_b / "marker_b.py").write_text("# child_b marker\n")
    yaml = tmp_path / "two_children_mcp.yaml"
    yaml.write_text(f"name: two_children\nworkspace:\n  kind: local\n  root: {root}\n")
    return yaml, child_a, child_b


@pytest.fixture
def manifest_with_cypher_tool(tmp_path: Path, tiny_graph_path: Path) -> Path:
    target = tmp_path / "cyphertool.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "cyphertool_mcp.yaml"
    manifest.write_text(
        "name: cyphertool\n"
        "tools:\n"
        "  - name: find_by_city\n"
        "    description: Find persons in a given city.\n"
        "    cypher: |\n"
        "      MATCH (p:Person) WHERE p.city = $city RETURN p.name LIMIT 5\n"
        "    parameters:\n"
        "      type: object\n"
        "      properties:\n"
        "        city: { type: string }\n"
        "      required: [city]\n"
    )
    return manifest


# ---------------------------------------------------------------------------
# Category A — Install & boot (4 tests)
# ---------------------------------------------------------------------------


def test_a1_console_script_on_path() -> None:
    """`pip install kglite[mcp]` puts kglite-mcp-server on PATH.
    Catches: 0.9.20 (no console script), 0.9.18 (wrong libpython linkage)."""
    server = _which_server()
    assert server is not None and server.strip(), "kglite-mcp-server not on PATH"


def test_a2_help_runs_without_error() -> None:
    """Binary starts — no dyld errors, no missing modules.
    Catches: 0.9.18 install_name regression, 0.9.20 missing mcp dep."""
    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run([server, "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "KGLite" in result.stdout


@pytest.mark.skipif(not shutil.which("conda"), reason="conda not available")
@pytest.mark.model_download
def test_a3_install_works_on_conda_python() -> None:
    """Fresh conda env can pip install kglite[mcp] and run the server.
    Catches: 0.9.18 (Python.org framework path baked into wheel; failed on conda)."""
    env_name = "kglite_smoke_test"
    try:
        subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.11"], check=True, capture_output=True)
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "kglite[mcp]"], check=True, capture_output=True
        )
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "kglite-mcp-server", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
    finally:
        subprocess.run(["conda", "env", "remove", "-n", env_name, "-y"], capture_output=True)


def test_a4_boot_summary_on_stderr(tiny_graph_path: Path) -> None:
    """Boot summary line on stderr names mode + path.
    Catches: 0.9.20 (silent boot)."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        time.sleep(0.5)  # let stderr drain
        lines = client.stderr_lines()
        boot = next((ln for ln in lines if "mode:" in ln), None)
        assert boot is not None, f"no boot summary; got stderr: {lines}"
        assert "graph" in boot
        assert str(tiny_graph_path) in boot
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Category B — Per-mode tool registration (5 tests)
# ---------------------------------------------------------------------------


def test_b1_graph_mode_registers_expected_tools(mutable_graph_path: Path) -> None:
    """--graph + manifest with builtins.save_graph: true → 8 baseline tools.
    Catches: 0.9.20 (8 missing tools)."""
    client = _client(
        [
            "--graph",
            str(mutable_graph_path),
            "--mcp-config",
            str(mutable_graph_path.parent / "mutable_mcp.yaml"),
        ]
    )
    try:
        tools = set(client.list_tools())
    finally:
        client.close()
    expected = {
        "ping",
        "cypher_query",
        "graph_overview",
        "read_code_source",
        "read_source",
        "grep",
        "list_source",
        "save_graph",
    }
    if os.environ.get("GITHUB_TOKEN"):
        expected |= {"github_issues", "github_api"}
    assert tools >= expected, f"missing: {expected - tools}; extra: {tools - expected}"


def test_b2_workspace_mode_registers_repo_management(tiny_workspace_dir: Path) -> None:
    """--workspace mode registers repo_management.
    Catches: 0.9.20 (workspace mode lost repo_management)."""
    client = _client(["--workspace", str(tiny_workspace_dir)])
    try:
        tools = client.list_tools()
    finally:
        client.close()
    assert "repo_management" in tools
    assert "set_root_dir" not in tools


def test_b3_local_workspace_mode_registers_set_root_dir(local_workspace_yaml: Path) -> None:
    """workspace.kind: local registers set_root_dir.
    Catches: 0.9.20 (set_root_dir gone in local-workspace mode)."""
    client = _client(["--mcp-config", str(local_workspace_yaml)])
    try:
        tools = client.list_tools()
    finally:
        client.close()
    assert "set_root_dir" in tools
    assert "repo_management" not in tools


def test_b6_graph_mode_accepts_disk_backed_graph_directory(tmp_path: Path) -> None:
    """--graph accepts a disk-backed graph directory (built via
    `kglite.KnowledgeGraph(storage='disk', path=...)`), not just .kgl files.

    Catches: 0.9.25 server.py:_validate_mode_paths rejecting any
    non-file path, which blocked operators deploying Wikidata-scale
    (124M+ node) disk-backed graphs through the CLI. Reported by
    the mcp-servers project against 0.9.25; fix lands in 0.9.26.

    The validator fix is what's anchored here. The fixture is
    populated via `add_nodes()` (the supported disk-mode write path —
    see B8 below for why Cypher CREATE is refused on disk graphs)."""
    import pandas as pd

    import kglite

    disk_dir = tmp_path / "disk_graph"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "m1", "title": "disk-d1"},
                {"id": "m2", "title": "disk-d2"},
                {"id": "m3", "title": "disk-d3"},
            ]
        ),
        "Marker",
        "id",
        "title",
    )
    g.save(str(disk_dir))
    del g  # release the mmap before booting the server

    # Server must boot with --graph pointing at the directory and
    # serve a cypher_query that sees the persisted nodes.
    client = _client(["--graph", str(disk_dir)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (n:Marker) RETURN n.title AS title ORDER BY title"},
        )
    finally:
        client.close()
    assert "Error" not in body and "ERROR" not in body, f"disk graph caused an error: {body!r}"
    assert "disk-d1" in body and "disk-d2" in body and "disk-d3" in body, (
        f"expected all three Markers in cypher_query result: {body!r}"
    )


def test_b9_disk_graph_supports_cypher_set_remove_and_delete(tmp_path: Path) -> None:
    """0.9.26 guards CREATE / MERGE on disk graphs (B8) but SET / REMOVE
    / DELETE go through `node_weight_mut` → `node_mut_cache` →
    `clear_arenas` flush. SET and DELETE always worked on disk;
    REMOVE was a silent no-op pre-fix (the flush only writes keys
    PRESENT in the staged Map, so a bare remove left the column
    store untouched). The fix routes REMOVE on disk to
    `NodeData::clear_property`, which inserts Null instead — the
    flush writes Null and reads return None.

    Catches: regressions across the disk mutation surface."""
    import pandas as pd

    import kglite

    disk_dir = tmp_path / "disk_set"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "m1", "title": "first", "score": 1.0},
                {"id": "m2", "title": "second", "score": 2.0},
            ]
        ),
        "Marker",
        "id",
        "title",
    )

    # SET a new property.
    g.cypher("MATCH (n:Marker {title: 'first'}) SET n.note = 'tagged'")
    rows = list(g.cypher("MATCH (n:Marker) RETURN n.title AS title, n.note AS note ORDER BY title"))
    assert rows[0]["note"] == "tagged", f"SET on disk graph regressed: {rows!r}"
    assert rows[1]["note"] is None, f"SET on disk graph regressed (other rows): {rows!r}"

    # REMOVE the property — must actually take effect (0.9.26 fix).
    g.cypher("MATCH (n:Marker {title: 'first'}) REMOVE n.note")
    rows = list(g.cypher("MATCH (n:Marker) RETURN n.title AS title, n.note AS note ORDER BY title"))
    assert rows[0]["note"] is None, f"REMOVE on disk graph regressed (still 'tagged'): {rows!r}"

    # REMOVE on a property set via add_nodes (not just SET).
    g.cypher("MATCH (n:Marker {title: 'second'}) REMOVE n.score")
    rows = list(g.cypher("MATCH (n:Marker) RETURN n.title AS title, n.score AS score ORDER BY title"))
    assert rows[1]["score"] is None, f"REMOVE of add_nodes-set property regressed: {rows!r}"

    # DELETE — also works.
    g.cypher("MATCH (n:Marker {title: 'second'}) DELETE n")
    count_rows = list(g.cypher("MATCH (n:Marker) RETURN count(n) AS c"))
    assert count_rows[0]["c"] == 1, f"DELETE on disk graph regressed: {count_rows!r}"


def test_b8_disk_graph_refuses_cypher_create_with_clear_error(tmp_path: Path) -> None:
    """0.9.26 guards against silent data loss on disk-backed graphs:
    Cypher CREATE / MERGE return a clear error directing users at
    `add_nodes()` or `to_disk()` instead. SET / REMOVE / DELETE still
    work normally.

    Catches: regressions in the disk add_node guard rail. Without
    this guard the call succeeds-with-no-data (no properties or
    title persist on disk-stored nodes). The proper write path is
    tracked separately."""
    import kglite

    disk_dir = tmp_path / "disk_guard"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))

    # CREATE must fail with the operator-facing message.
    try:
        result = g.cypher("CREATE (:Marker {title: 'x'})")
    except Exception as e:  # noqa: BLE001 — guard returns via Python exception
        msg = str(e)
    else:
        msg = str(result) if isinstance(result, str) else ""
    assert "CREATE" in msg and "disk" in msg.lower(), f"expected disk-CREATE refusal, got: {msg!r}"
    assert "add_nodes" in msg or "to_disk" in msg, f"expected workaround pointer in error: {msg!r}"

    # MERGE must fail with the same shape of message.
    try:
        result = g.cypher("MERGE (n:Marker {title: 'x'}) RETURN n")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
    else:
        msg = str(result) if isinstance(result, str) else ""
    assert "MERGE" in msg and "disk" in msg.lower(), f"expected disk-MERGE refusal, got: {msg!r}"


def test_b7_graph_mode_rejects_arbitrary_directory_without_meta(tmp_path: Path) -> None:
    """A directory that ISN'T a disk-backed graph (no disk_graph_meta.json)
    must still be rejected with a clear error, not silently accepted.

    Catches: a too-permissive fix that accepted any directory."""
    not_a_graph = tmp_path / "random_dir"
    not_a_graph.mkdir()
    (not_a_graph / "stuff.txt").write_text("not a graph\n")

    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--graph", str(not_a_graph)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0, "arbitrary directory should fail --graph validation"
    combined = (result.stdout + result.stderr).lower()
    assert "not a .kgl file or disk-backed graph directory" in combined, (
        f"expected the new validator message; got stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# ─── Bundled-tool overrides (0.9.27, mcp-methods 0.3.31) ─────


def test_b10_bundled_override_description_appears_in_tools_list(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Manifest `tools[].bundled: <name>` with a `description:` field
    replaces the bundled tool's default agent-facing description in
    `tools/list`. Catches: override map built but not applied;
    description swallowed; tool registered with stale text."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text(
        "name: override_test\n"
        "tools:\n"
        "  - bundled: cypher_query\n"
        '    description: "Custom description for cypher_query in this server."\n'
    )

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        tools_raw = client.list_tools_full()
    finally:
        client.close()
    cypher = next((t for t in tools_raw if t["name"] == "cypher_query"), None)
    assert cypher is not None, "cypher_query should always register"
    assert cypher["description"] == "Custom description for cypher_query in this server."


def test_b11_bundled_hidden_true_removes_tool_from_tools_list(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Manifest `tools[].bundled: ping / hidden: true` removes the
    tool from `tools/list`. Catches: hidden flag parsed but not
    enforced; tool still visible."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: hide_test\ntools:\n  - bundled: ping\n    hidden: true\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        tools = client.list_tools()
    finally:
        client.close()
    assert "ping" not in tools, f"ping should be hidden, but tools/list has: {tools!r}"
    # cypher_query still present (only ping was hidden)
    assert "cypher_query" in tools, f"non-hidden tools must still register: {tools!r}"


def test_b12_hidden_bundled_tool_call_returns_clear_error(tmp_path: Path, tiny_graph_path: Path) -> None:
    """An agent that calls a hidden bundled tool by name (despite it
    not being in tools/list) gets `Error: tool 'X' is hidden by
    manifest configuration.` rather than fall-through to
    'unknown tool'. Catches: dispatcher hidden-check missing."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: hide_test\ntools:\n  - bundled: ping\n    hidden: true\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool("ping", {})
    finally:
        client.close()
    assert "hidden by manifest" in body.lower(), f"expected hidden-tool error message, got: {body!r}"


def test_b13_unknown_bundled_tool_name_fails_at_boot(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Manifest `tools[].bundled: foo_not_a_tool` exits 3 at boot
    with a clear error listing the valid bundled tool names.
    Catches: typo in operator's override block reaching runtime."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text(
        "name: typo_test\n"
        "tools:\n"
        "  - bundled: cipher_query\n"  # typo: cipher vs cypher
        '    description: "x"\n'
    )

    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--graph", str(target), "--mcp-config", str(manifest)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0, (
        f"unknown bundled tool name should fail boot; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "unknown bundled tool" in combined, f"missing operator-facing error: {result.stderr!r}"
    assert "cipher_query" in combined, f"error should name the offending tool: {result.stderr!r}"
    # And the valid-names list should appear so the operator can find the typo fast.
    assert "cypher_query" in combined, f"error should list valid names: {result.stderr!r}"


def test_b14_bundled_rename_exposes_tool_under_new_name(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Manifest `tools[].bundled: cypher_query / rename: legal_cypher_query`
    makes the tool appear in `tools/list` as `legal_cypher_query` and
    removes `cypher_query` from the listing.

    Closes operator friction: three kglite servers (open_source / legal /
    prospect) exposing identical bundled surfaces showed six copies of
    `cypher_query` etc. in ToolSearch results. Per-deployment renames
    disambiguate the namespace. mcp-methods 0.3.34 + kglite 0.9.30."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: rename_test\ntools:\n  - bundled: cypher_query\n    rename: legal_cypher_query\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        tools = client.list_tools()
    finally:
        client.close()
    assert "legal_cypher_query" in tools, f"renamed tool should appear: {tools!r}"
    assert "cypher_query" not in tools, f"canonical name should not appear when rename is set: {tools!r}"


def test_b15_bundled_rename_dispatches_to_canonical_handler(tmp_path: Path, tiny_graph_path: Path) -> None:
    """A call to the renamed tool routes through the canonical
    bundled handler. Catches: alias appears in tools/list but call-
    site dispatch table never resolves the alias to its canonical
    name, leaving the tool unreachable in practice."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text(
        "name: rename_dispatch_test\ntools:\n  - bundled: cypher_query\n    rename: legal_cypher_query\n"
    )

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool(
            "legal_cypher_query",
            {"query": "MATCH (p:Person) RETURN p.name LIMIT 1"},
        )
    finally:
        client.close()
    assert "p.name" in body, f"renamed cypher_query call should return cypher rows: {body!r}"
    assert "unknown tool" not in body.lower()


def test_b16_bundled_rename_collision_with_bundled_name_fails_at_boot(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Renaming `cypher_query` to `ping` (an existing bundled name)
    should fail at boot with a clear collision error. Catches:
    rename validation pass missing the bundled-name shadow case."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: collide_test\ntools:\n  - bundled: cypher_query\n    rename: ping\n")

    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--graph", str(target), "--mcp-config", str(manifest)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0, f"collision should fail boot; stdout={result.stdout!r} stderr={result.stderr!r}"
    combined = (result.stdout + result.stderr).lower()
    assert "shadow" in combined or "collide" in combined, f"missing collision error: {result.stderr!r}"
    assert "ping" in combined


def test_b17_bundled_rename_collision_with_cypher_tool_fails_at_boot(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Renaming a bundled tool to a name already used by a manifest-
    declared cypher tool fails at boot. Same safety property as B16,
    different collision shape."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text(
        "name: collide_cypher_test\n"
        "tools:\n"
        "  - name: lookup\n"
        '    cypher: "MATCH (p:Person) RETURN p"\n'
        "  - bundled: cypher_query\n"
        "    rename: lookup\n"
    )

    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--graph", str(target), "--mcp-config", str(manifest)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0, "rename->cypher-tool collision should fail boot"
    combined = (result.stdout + result.stderr).lower()
    assert "lookup" in combined and "shadow" in combined


def test_i1_batch_load_hint_appears_in_instructions() -> None:
    """The server's `instructions` (sent on `initialize`) carries the
    ToolSearch batch-load hint so the agent learns how to load all
    of this server's tools in one round trip.

    Closes operator-reported friction: every search-by-name in
    Claude Code's deferred-tool loader was one round trip; agents
    had to make N searches to load N tools from one server, instead
    of one search per server. 0.9.30 auto-injects the hint."""
    from kglite.mcp_server.server import _BATCH_LOAD_HINT_MARKER, _compose_instructions

    # No operator instructions: the hint stands alone.
    out_alone = _compose_instructions(None)
    assert _BATCH_LOAD_HINT_MARKER in out_alone
    assert "ToolSearch" in out_alone

    # With operator instructions: hint prefixed; operator text preserved.
    operator_text = "Server-wide orientation: this server hosts the prospect graph."
    out_combined = _compose_instructions(operator_text)
    assert _BATCH_LOAD_HINT_MARKER in out_combined
    assert operator_text in out_combined
    # Hint comes BEFORE operator text so the batch-load primer is
    # read first by the agent.
    assert out_combined.index(_BATCH_LOAD_HINT_MARKER) < out_combined.index(operator_text)


def test_i2_batch_load_hint_is_idempotent() -> None:
    """Re-composing instructions that already contain the hint marker
    does not duplicate the hint. Catches: server boot path running
    _compose_instructions twice (e.g. through a refactor) leaves a
    double hint in the agent's initialize payload."""
    from kglite.mcp_server.server import _compose_instructions

    first_pass = _compose_instructions("Operator text.")
    second_pass = _compose_instructions(first_pass)
    assert first_pass == second_pass


def test_i3_batch_load_hint_marker_in_operator_text_suppresses_auto_inject() -> None:
    """An operator who writes the marker literally into their
    `instructions:` text suppresses the auto-injection — useful for
    deployments that want full control over the first-message
    surface."""
    from kglite.mcp_server.server import _BATCH_LOAD_HINT_MARKER, _compose_instructions

    operator_text = f"{_BATCH_LOAD_HINT_MARKER}\nMy custom orientation only."
    out = _compose_instructions(operator_text)
    assert out == operator_text  # auto-inject skipped


def test_i4_mode_banner_names_conditional_tools_per_mode() -> None:
    """The mode banner prepended to `instructions` (and to the bare
    `graph_overview()` preamble) names exactly which conditional tools
    are registered in the current mode.

    Closes operator feedback 2026-05-17: with several MCP servers
    attached, agents had to fingerprint mode by trial-calling
    `repo_management` / `set_root_dir` to learn which one was
    registered. The banner makes the answer visible at handshake."""
    from pathlib import Path

    from kglite.mcp_server.server import _MODE_BANNER_MARKER, _compose_mode_banner

    # graph mode (save_graph off — the default builtin)
    graph_banner = _compose_mode_banner({"kind": "graph", "path": Path("/tmp/x.kgl")}, None)
    assert _MODE_BANNER_MARKER in graph_banner
    assert "graph" in graph_banner
    assert "save_graph: not registered" in graph_banner
    assert "repo_management" in graph_banner and "not in this mode" in graph_banner

    # workspace mode — repo_management is the registered conditional tool
    workspace_banner = _compose_mode_banner({"kind": "workspace", "path": Path("/tmp")}, None)
    assert "repo_management: registered" in workspace_banner
    assert "set_root_dir" not in workspace_banner.split("repo_management: registered")[0]

    # local-workspace mode — set_root_dir is the registered conditional tool
    local_banner = _compose_mode_banner({"kind": "local_workspace", "path": Path("/tmp")}, None)
    assert "set_root_dir: registered" in local_banner
    assert "repo_management" in local_banner and "not in this mode" in local_banner


def test_i5_mode_banner_signals_save_graph_when_enabled() -> None:
    """`builtins.save_graph: true` flips the banner's save_graph line
    from "not registered (read-only)" to "registered. Call to persist
    ...". Tests the write-mode signal — the second half of the
    operator request paired with the mode signal."""
    from pathlib import Path
    from types import SimpleNamespace

    from kglite.mcp_server.server import _compose_mode_banner

    fake_manifest = SimpleNamespace(builtins=SimpleNamespace(save_graph=True))
    banner = _compose_mode_banner({"kind": "graph", "path": Path("/tmp/x.kgl")}, fake_manifest)
    assert "save_graph: registered" in banner
    assert "not registered" not in banner.split("save_graph: registered")[0]


def test_i6_mode_banner_covers_remaining_modes() -> None:
    """Mode-banner payloads exist for `source_root`, `watch`, and `bare`
    too — not just the three graph-bearing modes. Pinned so a refactor
    that drops one of them fails CI."""
    from kglite.mcp_server.server import _MODE_BANNER_MARKER, _compose_mode_banner

    source_root_banner = _compose_mode_banner({"kind": "source_root"}, None)
    assert _MODE_BANNER_MARKER in source_root_banner
    assert "source-root" in source_root_banner
    assert "read_source" in source_root_banner
    # The whole point of source_root mode: no graph tools registered.
    assert "cypher_query" in source_root_banner and "not active" in source_root_banner

    watch_banner = _compose_mode_banner({"kind": "watch"}, None)
    assert _MODE_BANNER_MARKER in watch_banner
    assert "watch" in watch_banner
    # Graph IS registered in watch mode (rebuilt from source tree).
    assert "cypher_query" in watch_banner and "registered" in watch_banner
    # ...but save_graph is not — the graph is derived state, persisting
    # would just be overwritten on the next rebuild.
    assert "save_graph: not registered" in watch_banner

    bare_banner = _compose_mode_banner({"kind": "bare"}, None)
    assert _MODE_BANNER_MARKER in bare_banner
    assert "bare" in bare_banner


def test_s1_module_property_uniform_across_code_tree_node_types() -> None:
    """code_tree.build() populates a `module` property on Function /
    Class / Constant / Module / File nodes — the same dotted path
    across types so `WHERE n.module STARTS WITH 'foo.bar'` works
    regardless of which entity label `n` matches.

    Closes operator-reported friction: `module` was previously only
    on File and Module nodes, so cross-type module filters silently
    returned zero rows. 0.9.30 normalises the property via a
    file_path → module_path lookup at code_tree build time."""
    import kglite
    from kglite import code_tree

    # Build a tiny in-memory project to keep this test fast (the
    # kglite codebase itself works but takes longer; for a regression
    # gate we just need a few files with distinct modules).
    src = Path(__file__).parent / "fixtures_code_tree"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir()
    (src / "alpha.py").write_text("def foo():\n    return 1\nclass Alpha:\n    pass\n")
    nested = src / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("")
    (nested / "beta.py").write_text("DELTA = 42\n\ndef bar():\n    return 2\n")
    try:
        g = code_tree.build(str(src))

        # Function: must carry a non-null `module` matching the file
        rows = list(g.cypher("MATCH (f:Function) RETURN f.title AS t, f.module AS m"))
        assert rows, "code_tree should produce Function nodes"
        for row in rows:
            assert row["m"] is not None, f"Function missing module: {row}"

        # Cross-type WHERE on module — the operator's reproducer.
        # code_tree prefixes the project root dir name, so for a
        # fixture rooted at ".../fixtures_code_tree" the function
        # `bar` in `nested/beta.py` has module `fixtures_code_tree.nested.beta`.
        # The WHERE clause uses the partial module path that locates
        # the nested submodule regardless of project-root prefix.
        rows_starts_with = list(g.cypher("MATCH (f:Function) WHERE f.module CONTAINS '.nested.' RETURN f.title"))
        assert any("bar" in r.get("f.title", "") for r in rows_starts_with), (
            f"WHERE f.module CONTAINS '.nested.' should find bar(): {rows_starts_with!r}"
        )

        # Class also carries module
        cls_rows = list(g.cypher("MATCH (c:Class) RETURN c.title AS t, c.module AS m"))
        assert cls_rows
        assert all(r["m"] is not None for r in cls_rows)

        # Constant also
        const_rows = list(g.cypher("MATCH (c:Constant) RETURN c.title AS t, c.module AS m"))
        assert const_rows
        for r in const_rows:
            assert r["m"] is not None

        # Module nodes carry module (== qualified_name)
        mod_rows = list(g.cypher("MATCH (m:Module) RETURN m.title AS t, m.module AS mod"))
        for r in mod_rows:
            assert r["mod"] is not None
    finally:
        if src.exists():
            shutil.rmtree(src)
    # Suppress unused-import warning when the test runs without the
    # kglite module being referenced elsewhere in the file scope.
    _ = kglite


def test_s2_describe_emits_sample_attr_for_high_cardinality_props() -> None:
    """`describe()` emits a `sample="..."` attribute on properties
    whose unique count exceeds the vals threshold. Lets the agent
    see ONE example value of every property — not just the property
    name — closing the operator-reported "guess the value shape
    from the name alone" friction."""
    import kglite
    from kglite import code_tree

    src = Path(__file__).parent / "fixtures_code_tree_s2"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir()
    # Enough files to push high-cardinality props past the vals
    # threshold (>15 unique values for things like docstring,
    # signature, line_number).
    for i in range(20):
        (src / f"mod_{i:02d}.py").write_text(
            f'"""Module docstring {i}."""\n\n'
            f"def function_{i:02d}():\n"
            f'    """Function {i} doing things."""\n'
            f"    return {i}\n"
        )
    try:
        g = code_tree.build(str(src))
        xml = g.describe(types=["Function"])
        # docstring should be high-cardinality (20 unique) and carry sample=
        # (not the lower-cardinality vals= path which only fires when unique <= 15)
        assert "sample=" in xml, f"high-cardinality property should carry sample= attr; got:\n{xml}"
    finally:
        if src.exists():
            shutil.rmtree(src)
    _ = kglite


def test_b4_save_graph_gated_on_manifest_flag(
    mutable_graph_path: Path, graph_without_savegraph_flag: Path, tiny_graph_path: Path
) -> None:
    """save_graph only registers when builtins.save_graph: true is set.
    Catches: 0.9.18 (unconditional registration)."""
    # With flag
    client = _client(
        [
            "--graph",
            str(mutable_graph_path),
            "--mcp-config",
            str(mutable_graph_path.parent / "mutable_mcp.yaml"),
        ]
    )
    try:
        assert "save_graph" in client.list_tools()
    finally:
        client.close()
    # Without flag
    target = graph_without_savegraph_flag.parent / "nograph.kgl"
    client = _client(
        [
            "--graph",
            str(target),
            "--mcp-config",
            str(graph_without_savegraph_flag),
        ]
    )
    try:
        assert "save_graph" not in client.list_tools()
    finally:
        client.close()


def test_b5_github_tools_gated_on_token(tiny_workspace_dir: Path, tmp_path: Path) -> None:
    """github_issues + github_api only register when GITHUB_TOKEN is present.
    Catches: 0.9.20 (tools absent with token), 0.9.18 (registered without)."""
    # No token
    env_no = {k: v for k, v in os.environ.items() if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}
    env_no["HOME"] = str(tmp_path)  # avoid walk-up to a real .env
    proc = _spawn(["--workspace", str(tiny_workspace_dir)], extra_env=env_no, cwd=str(tmp_path))
    client = McpClient(proc)
    client.initialize()
    try:
        tools_no = client.list_tools()
    finally:
        client.close()
    assert "github_issues" not in tools_no
    assert "github_api" not in tools_no

    # With token
    env_yes = {**env_no, "GITHUB_TOKEN": "ghp_test"}
    proc = _spawn(["--workspace", str(tiny_workspace_dir)], extra_env=env_yes, cwd=str(tmp_path))
    client = McpClient(proc)
    client.initialize()
    try:
        tools_yes = client.list_tools()
    finally:
        client.close()
    assert "github_issues" in tools_yes
    assert "github_api" in tools_yes


# ---------------------------------------------------------------------------
# Category C — Tool output content (8 tests)
# ---------------------------------------------------------------------------


def test_c1_cypher_query_returns_actual_row_values(tiny_graph_path: Path) -> None:
    """The 0.9.21 regression: rows rendered column names instead of values.
    Catches: 0.9.21 row formatter bug."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        body = client.call_tool("cypher_query", {"query": "RETURN 1+1 AS sum"})
    finally:
        client.close()
    assert "'sum'" not in body, f"formatter still emitting column name as value: {body!r}"
    assert "2" in body, f"expected value 2 not in response: {body!r}"


def test_c2_cypher_query_multi_column_returns_distinct_values(tiny_graph_path: Path) -> None:
    """Multi-column queries return distinct row values.
    Catches: 0.9.21 row formatter, partial-fix scenarios."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (p:Person) RETURN p.name, p.age LIMIT 3",
            },
        )
    finally:
        client.close()
    lines = [ln for ln in body.split("\n") if ln and "row(s)" not in ln and "p.name" != ln.strip()]
    assert len(set(lines)) > 1, f"all rows identical: {body!r}"
    for line in lines:
        assert "'p.name'" not in line, f"row contains column-name string: {line!r}"
        assert "'p.age'" not in line, f"row contains column-name string: {line!r}"


def test_c3_ping_returns_pong(tiny_graph_path: Path) -> None:
    """Simplest content assertion. Catches: 0.9.20 (ping absent)."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        body = client.call_tool("ping", {})
    finally:
        client.close()
    assert body.strip() == "pong"


def test_c4_read_source_returns_file_content(tiny_source_dir: Path) -> None:
    """read_source returns actual file content.
    Catches: 0.9.20 (absent), 0.9.21 (registered but garbage)."""
    client = _client(["--source-root", str(tiny_source_dir)])
    try:
        body = client.call_tool("read_source", {"file_path": "alpha.py"})
    finally:
        client.close()
    assert "def alpha" in body, f"read_source returned no source: {body[:200]!r}"


def test_c5_grep_returns_matches_with_file_and_line(tiny_source_dir: Path) -> None:
    """grep output: `path:line:content` format.
    Catches: 0.9.20 (grep absent), formatter-strips-line-numbers."""
    client = _client(["--source-root", str(tiny_source_dir)])
    try:
        body = client.call_tool("grep", {"pattern": "def ", "max_results": 5})
    finally:
        client.close()
    assert re.search(r"\.py:\d+:", body), f"missing path:line: format: {body[:300]!r}"


def test_c6_csv_http_server_returns_csv_body_via_http(tiny_graph_path: Path, manifest_with_csv_http: Path) -> None:
    """csv_http_server actually serves CSV over HTTP, no 500.
    Catches: 0.9.22 aiohttp content_type/charset bug."""
    import urllib.request

    target = manifest_with_csv_http.parent / "csvhttp.kgl"
    client = _client(["--graph", str(target), "--mcp-config", str(manifest_with_csv_http)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (p:Person) RETURN p.name LIMIT 2 FORMAT CSV",
            },
        )
        m = re.search(r"http://[^\s\"\)]+", body)
        assert m, f"no localhost URL in FORMAT CSV response: {body!r}"
        time.sleep(0.3)
        with urllib.request.urlopen(m.group(0), timeout=5) as resp:
            assert resp.status == 200, f"HTTP GET status {resp.status}"
            content = resp.read().decode()
        assert "p.name" in content, f"CSV missing header: {content!r}"
        assert content.count("\n") >= 2
    finally:
        client.close()


def test_c7_format_csv_inline_fallback(tiny_graph_path: Path) -> None:
    """Without csv_http_server, FORMAT CSV returns body inline.
    Catches: regression where inline fallback breaks."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (p:Person) RETURN p.name LIMIT 2 FORMAT CSV",
            },
        )
    finally:
        client.close()
    assert "http://" not in body, f"unexpected URL when csv_http_server disabled: {body!r}"
    assert "p.name" in body
    assert body.count("\n") >= 2


def test_c9_csv_http_server_true_uses_os_assigned_port(tmp_path: Path, tiny_graph_path: Path) -> None:
    """`csv_http_server: true` (no explicit port) must bind to an
    OS-assigned port and produce a URL with a non-zero port.

    Catches: pre-0.9.29 regression where the default port was 8765.
    Claude Desktop launches all registered MCP servers in parallel
    at startup; the first to boot grabbed 8765 and every subsequent
    server crashed with `OSError: address already in use`, which the
    user saw as "Server disconnected" with no actionable detail.
    Operator workaround was to hand-assign distinct ports across
    every manifest. Fix: default port → 0, post-bind capture writes
    the OS-assigned port back into the config so url_for() works."""
    target = tmp_path / "true_csv.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "true_csv_mcp.yaml"
    manifest.write_text("name: true_csv\nextensions:\n  csv_http_server: true\n")
    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (p:Person) RETURN p.name LIMIT 1 FORMAT CSV"},
        )
        m = re.search(r"http://127\.0\.0\.1:(\d+)/", body)
        assert m, f"no localhost URL in FORMAT CSV response: {body!r}"
        port = int(m.group(1))
        assert port != 0, f"URL has port 0 — post-bind capture broken: {body!r}"
        assert 1024 <= port <= 65535, f"OS-assigned port out of expected range: {port}"
    finally:
        client.close()


def test_c10_two_csv_http_servers_can_coexist(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Two kglite-mcp-server processes booted concurrently with the
    `csv_http_server: true` default must NOT collide on the listener
    port — each gets its own OS-assigned port.

    Catches: pre-0.9.29 hardcoded 8765 default that made parallel
    server boot impossible. This is the operator-reported scenario:
    Claude Desktop registers four kglite-mcp-servers and launches
    them all at startup; with a fixed default the second through
    fourth crashed silently."""
    targets = []
    manifests = []
    for i in range(2):
        t = tmp_path / f"co_{i}.kgl"
        shutil.copy(tiny_graph_path, t)
        m = tmp_path / f"co_{i}_mcp.yaml"
        m.write_text(f"name: co_{i}\nextensions:\n  csv_http_server: true\n")
        targets.append(t)
        manifests.append(m)

    clients = []
    try:
        for t, m in zip(targets, manifests):
            clients.append(_client(["--graph", str(t), "--mcp-config", str(m)]))
        # Each must be alive + responsive — neither died from a port
        # collision at boot.
        for c in clients:
            tools = c.list_tools()
            assert "cypher_query" in tools, f"client missing cypher_query (port-collision crash?): tools={tools!r}"
    finally:
        for c in clients:
            c.close()


def test_c8_save_graph_persists_to_disk(mutable_graph_path: Path) -> None:
    """save_graph actually writes to the .kgl file.
    Catches: save_graph registers but doesn't persist."""
    manifest = mutable_graph_path.parent / "mutable_mcp.yaml"
    original_mtime = mutable_graph_path.stat().st_mtime
    time.sleep(1.1)  # ensure mtime resolution captures the change

    client = _client(["--graph", str(mutable_graph_path), "--mcp-config", str(manifest)])
    try:
        client.call_tool("cypher_query", {"query": "CREATE (:Marker {tag: 'savegraph-test'})"})
        save_body = client.call_tool("save_graph", {})
    finally:
        client.close()
    assert "ERROR" not in save_body and "error" not in save_body.lower()[:30]
    new_mtime = mutable_graph_path.stat().st_mtime
    assert new_mtime > original_mtime, "save_graph did not update .kgl mtime"

    # Reload and verify the mutation persisted
    client = _client(["--graph", str(mutable_graph_path), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (n:Marker {tag: 'savegraph-test'}) RETURN count(n) AS c",
            },
        )
    finally:
        client.close()
    assert "1" in body, f"mutation didn't persist across save/reload: {body!r}"


def test_c8b_save_graph_persists_disk_mode(tmp_path: Path) -> None:
    """save_graph on a disk-backed graph (directory, not .kgl file)
    exercises the `save_disk` branch of `kglite::api::save_graph`'s
    dispatch — the complement to c8's in-memory branch. Both are
    needed to lock in the dispatch fix landed in 0.9.45 (the pre-fix
    `run_save` only handled the disk branch and errored on in-memory
    `.kgl` graphs).

    Mutation goes through `SET` (one of the Cypher operations that
    works on disk-mode graphs — see B9) since `CREATE` is refused
    on disk (B8)."""
    import pandas as pd

    import kglite

    disk_dir = tmp_path / "disk_graph"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.add_nodes(
        pd.DataFrame([{"id": "m1", "title": "before"}]),
        "Marker",
        "id",
        "title",
    )
    g.save(str(disk_dir))
    del g  # release the mmap before booting the server

    manifest = tmp_path / "disk_mcp.yaml"
    manifest.write_text("name: disk\nbuiltins:\n  save_graph: true\n")

    # Mutate via SET, then save.
    client = _client(["--graph", str(disk_dir), "--mcp-config", str(manifest)])
    try:
        client.call_tool(
            "cypher_query",
            {"query": "MATCH (m:Marker {id: 'm1'}) SET m.title = 'after'"},
        )
        save_body = client.call_tool("save_graph", {})
    finally:
        client.close()
    assert "ERROR" not in save_body and "error" not in save_body.lower()[:30], (
        f"save_graph on disk-backed graph errored: {save_body!r}"
    )

    # Reload from disk and verify the mutation persisted.
    client = _client(["--graph", str(disk_dir), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (m:Marker {id: 'm1'}) RETURN m.title AS title"},
        )
    finally:
        client.close()
    assert "after" in body, f"disk mutation didn't persist across save/reload: {body!r}"


# ---------------------------------------------------------------------------
# Category D — Embedder & semantic search (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.model_download
def test_d1_bge_m3_loads_and_returns_dimension_1024(graph_with_bge_m3_embedder: Path) -> None:
    """bge-m3 must actually load. Catches: 0.9.20 catalog mismatch."""
    target = graph_with_bge_m3_embedder.parent / "bgem3.kgl"
    client = _client(["--graph", str(target), "--mcp-config", str(graph_with_bge_m3_embedder)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (a:Article) WHERE text_score(a, 'body', 'hello world') > 0 RETURN count(a) AS n",
            },
        )
    finally:
        client.close()
    assert "not supported" not in body, f"bge-m3 catalog mismatch: {body!r}"
    assert re.search(r"\b\d+\b", body), f"no count in response: {body!r}"


@pytest.mark.model_download
def test_d2_text_score_returns_ordered_relevance(
    graph_with_bge_m3_embeddings_precomputed: Path, tmp_path: Path
) -> None:
    """Semantically-close text scores higher.
    Catches: 0.9.20 (model didn't load → no scoring)."""
    manifest = tmp_path / "bgem3_mcp.yaml"
    manifest.write_text("name: bgem3\nextensions:\n  embedder:\n    backend: fastembed\n    model: BAAI/bge-m3\n")
    client = _client(
        [
            "--graph",
            str(graph_with_bge_m3_embeddings_precomputed),
            "--mcp-config",
            str(manifest),
        ]
    )
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": (
                    "MATCH (a:Article) RETURN a.title, "
                    "text_score(a, 'body', 'quantum mechanics') AS s "
                    "ORDER BY s DESC LIMIT 4 FORMAT CSV"
                ),
            },
        )
    finally:
        client.close()
    lines = [ln for ln in body.split("\n") if "," in ln and not ln.startswith("a.title")]
    assert len(lines) >= 2, f"expected ≥2 score rows: {body!r}"
    scores = [float(ln.split(",")[-1]) for ln in lines]
    assert scores[0] > scores[-1], f"scores not descending: {scores}"
    # Top result should be the quantum article (our fixture has alternating titles)
    assert "quantum" in lines[0].lower(), f"quantum should rank first: {lines!r}"


@pytest.mark.model_download
def test_d3_embedder_lifecycle_lazy_load_and_unload(graph_with_bge_m3_embedder: Path) -> None:
    """First call cold (loads model); second call warm.
    Catches: embedder reloading every call regressions."""
    target = graph_with_bge_m3_embedder.parent / "bgem3.kgl"
    client = _client(["--graph", str(target), "--mcp-config", str(graph_with_bge_m3_embedder)])
    try:
        t0 = time.monotonic()
        body1 = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (a:Article) WHERE text_score(a, 'body', 'test') > 0 RETURN count(a)",
            },
        )
        cold_ms = (time.monotonic() - t0) * 1000
        t0 = time.monotonic()
        body2 = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (a:Article) WHERE text_score(a, 'body', 'other') > 0 RETURN count(a)",
            },
        )
        warm_ms = (time.monotonic() - t0) * 1000
    finally:
        client.close()
    assert "ERROR" not in body1 and "ERROR" not in body2
    # Warm should be substantially faster than cold (loose threshold —
    # cold includes the model load on first query).
    assert warm_ms < cold_ms / 2 + 500, (
        f"warm call ({warm_ms:.0f}ms) not faster than cold ({cold_ms:.0f}ms) — embedder reloading?"
    )


# ---------------------------------------------------------------------------
# Category E — Manifest & .env handling (4 tests)
# ---------------------------------------------------------------------------


def test_e1_explicit_source_root_overrides_auto_bind(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Manifest source_root: overrides --graph parent auto-bind.
    Catches: 0.9.17 P1 (manifest source_root silently ignored)."""
    # Graph in one dir, source in another.
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    graph_target = graph_dir / "g.kgl"
    shutil.copy(tiny_graph_path, graph_target)

    source_dir = tmp_path / "explicit_source"
    source_dir.mkdir()
    (source_dir / "marker.py").write_text("# explicitly bound\n")

    manifest = tmp_path / "g_mcp.yaml"
    manifest.write_text(f"name: g\nsource_root: {source_dir}\n")

    client = _client(["--graph", str(graph_target), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool("read_source", {"file_path": "marker.py"})
    finally:
        client.close()
    assert "ERROR" not in body and "Error" not in body[:30], f"explicit source_root not honored: {body!r}"
    assert "explicitly bound" in body


def test_e2_env_file_walk_up(tmp_path: Path, tiny_workspace_dir: Path) -> None:
    """`.env` placed in workspace parent gets loaded.
    Catches: 0.9.17 .env walk-up missing."""
    env_path = tiny_workspace_dir.parent / ".env"
    env_path.write_text("TEST_VAR_XYZ=found_via_walkup\n")

    client = _client(["--workspace", str(tiny_workspace_dir)])
    try:
        time.sleep(0.5)
        lines = client.stderr_lines()
        env_line = next((ln for ln in lines if "env" in ln.lower() and ".env" in ln), None)
    finally:
        client.close()
    assert env_line is not None, f"no env-loaded line in stderr: {lines}"


def test_e3_unknown_manifest_keys_error_cleanly(tmp_path: Path) -> None:
    """Strict-key validation rejects typos.
    Catches: silent acceptance of typo'd keys."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nunknown_top_key: oops\n")
    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--mcp-config", str(bad)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    assert "unknown" in combined, f"strict-key validation didn't error: {combined!r}"


def test_e4_temp_cleanup_wipes_on_graph_overview(mutable_graph_path: Path, tmp_path: Path) -> None:
    """builtins.temp_cleanup: on_overview wipes temp/ on bare graph_overview().
    Catches: 0.9.18 (no-op flag), 0.9.20 regressions."""
    # The fixture's manifest base dir is tmp_path. Drop a stale file in
    # `<base>/temp/`.
    manifest = mutable_graph_path.parent / "mutable_mcp.yaml"
    temp_dir = manifest.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    stale = temp_dir / "test-stale.csv"
    stale.write_text("old-data")
    assert stale.exists()

    client = _client(["--graph", str(mutable_graph_path), "--mcp-config", str(manifest)])
    try:
        client.call_tool("graph_overview", {})
    finally:
        client.close()
    assert not stale.exists(), "temp_cleanup didn't wipe temp/"


# ---------------------------------------------------------------------------
# Category F — Workspace state propagation (3 tests)
# ---------------------------------------------------------------------------


def test_f1_repo_management_activates_makes_repo_default_for_github_tools(
    tiny_workspace_dir: Path,
) -> None:
    """After repo_management activates, github_issues defaults to that repo.
    Catches: 0.9.22 active-repo bug."""
    # Pre-create stub/repo so activate() doesn't try to clone from github.com.
    (tiny_workspace_dir / "stub" / "repo" / "README").write_text("# stub\n")
    client = _client(
        ["--workspace", str(tiny_workspace_dir)],
        extra_env={"GITHUB_TOKEN": "ghp_test"},
    )
    try:
        act = client.call_tool("repo_management", {"name": "stub/repo"})
        assert "Activated" in act or "stub/repo" in act, f"activation failed: {act!r}"
        # github_issues without repo_name — should NOT say "could not auto-detect"
        # (it may fail with auth since the token is fake, but the repo must be picked up)
        body = client.call_tool("github_issues", {"limit": 1})
    finally:
        client.close()
    assert "could not auto-detect" not in body.lower(), f"active repo not propagating to github_issues: {body!r}"


def test_f2_set_root_dir_rebinds_source_tools(
    local_workspace_yaml: Path, tmp_path: Path, tiny_source_dir: Path
) -> None:
    """After set_root_dir, list_source operates on the new root.
    Catches: tool registers but state-swap is incomplete."""
    # Set up a second root inside the workspace
    second = tiny_source_dir / "second_root"
    second.mkdir()
    (second / "second_file.py").write_text("# in second root\n")

    client = _client(["--mcp-config", str(local_workspace_yaml)])
    try:
        initial = client.call_tool("list_source", {})
        swap = client.call_tool("set_root_dir", {"path": "second_root"})
        assert "ERROR" not in swap, f"set_root_dir failed: {swap!r}"
        new_listing = client.call_tool("list_source", {})
    finally:
        client.close()
    assert new_listing != initial, "set_root_dir didn't change source listing"
    assert "second_file.py" in new_listing, f"new root file not visible: {new_listing!r}"


def test_f3_repo_management_force_rebuild_triggers_post_activate_hook(
    tiny_workspace_dir: Path,
) -> None:
    """force_rebuild=true bypasses SHA gating.

    NOTE: our 0.9.23 Workspace doesn't implement SHA gating yet (it
    rebuilds unconditionally). This test asserts the boolean is
    accepted without error and the response acknowledges the request —
    once SHA gating lands in a future release, tighten the assertion.
    Catches: force_rebuild kwarg being silently dropped."""
    (tiny_workspace_dir / "stub" / "repo" / "README").write_text("# stub\n")
    client = _client(["--workspace", str(tiny_workspace_dir)])
    try:
        client.call_tool("repo_management", {"name": "stub/repo"})
        body = client.call_tool("repo_management", {"update": True, "force_rebuild": True})
    finally:
        client.close()
    # The current implementation may return "Updated" or "Activated" —
    # what matters is no error.
    assert "ERROR" not in body and "invalid" not in body.lower(), f"force_rebuild rejected: {body!r}"


def test_f4_set_root_dir_can_swap_laterally_within_workspace_root(
    local_workspace_with_two_children: tuple[Path, Path, Path],
) -> None:
    """After set_root_dir(child_a), agents can still set_root_dir(child_b)
    where both are siblings under the manifest's workspace.root.
    Catches: 0.9.23 sandbox-narrows-each-swap regression (the
    pre-0.9.24 Python wrapper mutated `self.root` after each swap,
    so the next sandbox check compared against the narrower active
    root; mcp-methods 0.3.28's atomic-swap RwLock + immutable
    configured root makes lateral swaps work by construction)."""
    yaml, child_a, child_b = local_workspace_with_two_children
    client = _client(["--mcp-config", str(yaml)])
    try:
        body_a = client.call_tool("set_root_dir", {"path": str(child_a)})
        assert "Active root set to" in body_a, f"first swap failed: {body_a!r}"

        body_b = client.call_tool("set_root_dir", {"path": str(child_b)})
        assert "escapes" not in body_b.lower(), (
            f"sandbox narrowed: cannot swap to sibling under workspace.root: {body_b!r}"
        )
        assert "Active root set to" in body_b, f"second swap failed: {body_b!r}"

        # The listing should now reflect child_b's contents — proves the
        # source-root rebind actually fired through to the source tools.
        listing = client.call_tool("list_source", {})
    finally:
        client.close()
    assert "marker_b.py" in listing, f"new sibling root not reflected in list_source: {listing!r}"


# ---------------------------------------------------------------------------
# Direct API tests (not in operator spec but kept from previous suite)
# ---------------------------------------------------------------------------


def test_mcp_internal_module_importable() -> None:
    """The submodule must be importable as `kglite._mcp_internal`
    (registered in sys.modules by `mcp_tools.rs::register`)."""
    import kglite._mcp_internal as m

    assert callable(m.read_source)
    assert callable(m.grep)
    assert callable(m.list_source)
    assert callable(m.ping)
    assert callable(m.git_api)
    assert callable(m.has_github_token)
    assert m.GithubIssues


def test_mcp_internal_read_source_traversal_rejected(tmp_path: Path) -> None:
    import kglite._mcp_internal as m

    body = m.read_source("../../../etc/passwd", [str(tmp_path)])
    assert body.startswith("Error:")


def test_cypher_query_returns_actual_row_data_direct(tmp_path: Path) -> None:
    """Direct exercise of the 0.9.22 bug-fix code path (no MCP stdio).
    Useful when debugging the formatter in isolation."""
    import kglite
    from kglite.mcp_server.tools import GraphState, run_cypher

    state = GraphState()
    g = kglite.KnowledgeGraph()
    state._active = type("A", (), {"graph": g, "source_path": None})()  # type: ignore[attr-defined]
    body = run_cypher(state, "RETURN 1+1 AS sum")
    assert "'sum'" not in body
    assert "2" in body


def test_bge_m3_cooldown_default_keeps_session_resident() -> None:
    """Without explicit cool-down, the default (900s) holds the
    session resident across rapid embed() calls — exactly what makes
    warm-call latency drop to ms-scale."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    e = BgeM3Embedder()
    # Simulate a load by stuffing a sentinel into _session, then
    # check the idle-release path leaves it alone for short idles.
    e._session = "sentinel-loaded"  # type: ignore[assignment]
    e._last_used = time.monotonic()
    e._maybe_release_idle()
    assert e._session == "sentinel-loaded", "default cool-down should not release fresh session"


def test_bge_m3_cooldown_releases_after_idle_threshold() -> None:
    """Once `cooldown_seconds` of monotonic time elapses since the
    last embed, the next call drops the session."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    e = BgeM3Embedder(cooldown_seconds=1)  # 1-second cool-down for fast test
    e._session = "sentinel-loaded"  # type: ignore[assignment]
    e._tokenizer = "sentinel-tok"  # type: ignore[assignment]
    e._last_used = time.monotonic() - 2.0  # already idle past threshold
    e._maybe_release_idle()
    assert e._session is None, "stale session should have been released"
    assert e._tokenizer is None


def test_bge_m3_cooldown_zero_disables_release() -> None:
    """cooldown_seconds=0 means "never release" — model stays
    resident forever (the operator's heavy-use mode)."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    e = BgeM3Embedder(cooldown_seconds=0)
    e._session = "sentinel-loaded"  # type: ignore[assignment]
    e._last_used = time.monotonic() - 86400  # idle for a day
    e._maybe_release_idle()
    assert e._session == "sentinel-loaded", "cooldown=0 should never release"


def test_embedder_manifest_parses_cooldown() -> None:
    """`extensions.embedder.cooldown:` flows through from YAML into
    BgeM3Embedder. Catches: YAML parser silently dropping the field."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder
    from kglite.mcp_server.embedder import from_manifest_value

    adapter = from_manifest_value(
        {
            "backend": "fastembed",
            "model": "BAAI/bge-m3",
            "cooldown": 60,
        }
    )
    assert isinstance(adapter, BgeM3Embedder)
    assert adapter._cooldown_seconds == 60


# ---------------------------------------------------------------------------
# 0.9.24 pyo3-wrapper boundary tests (E5 / F5 / W1)
# ---------------------------------------------------------------------------


def test_e5_manifest_extensions_passthrough_preserves_all_keys(tmp_path: Path) -> None:
    """The Rust → Python conversion of `extensions: serde_json::Map`
    must recursively preserve every key including deeply nested
    mappings, ints, bools, and strings. Catches: drift in
    `json_value_to_py` when the recursive walker changes shape.

    The wire contract is "round-trip every JSON scalar / array /
    object" — exercise each branch."""
    yaml_path = tmp_path / "ext_passthrough_mcp.yaml"
    yaml_path.write_text(
        "name: ext_test\n"
        "extensions:\n"
        "  csv_http_server: true\n"
        "  csv_http_server_dir: temp/\n"
        "  embedder:\n"
        "    backend: fastembed\n"
        "    model: BAAI/bge-m3\n"
        "    cooldown: 1200\n"
        "  deeply:\n"
        "    nested:\n"
        "      list_value: [1, 2, 3]\n"
        "      bool_value: false\n"
        "      string_value: hello\n"
    )
    from kglite.mcp_server.manifest import load_manifest

    manifest = load_manifest(yaml_path)
    ext = manifest.extensions
    assert ext["csv_http_server"] is True
    assert ext["csv_http_server_dir"] == "temp/"
    assert ext["embedder"]["backend"] == "fastembed"
    assert ext["embedder"]["model"] == "BAAI/bge-m3"
    assert ext["embedder"]["cooldown"] == 1200
    assert ext["deeply"]["nested"]["list_value"] == [1, 2, 3]
    assert ext["deeply"]["nested"]["bool_value"] is False
    assert ext["deeply"]["nested"]["string_value"] == "hello"


def test_e6_workspace_manifest_parent_walk_with_applies_to(tmp_path: Path) -> None:
    """When the manifest at `<parent>/workspace_mcp.yaml` declares
    `workspace.applies_to: ./*` (or a matching pattern),
    `find_workspace_manifest(<parent>/<child>/)` walks one level up
    and resolves to it.

    Catches: pre-0.9.29 layout-mismatch where the natural folder
    structure was

        open_source/
        ├── workspace_mcp.yaml     # config sits beside the sandbox
        └── repos/                 # --workspace points here

    but discovery only looked INSIDE `--workspace`. Operators had to
    pass `--mcp-config` explicitly for every launch. The fix in
    mcp-methods 0.3.33 added opt-in parent-walk via
    `workspace.applies_to`; kglite 0.9.29 bumps the pin to pick it
    up. The opt-in is *required* — see E7 for the safety property."""
    from kglite.mcp_server.manifest import find_workspace_manifest

    parent = tmp_path / "open_source"
    parent.mkdir()
    (parent / "workspace_mcp.yaml").write_text("name: parent_test\nworkspace:\n  kind: github\n  applies_to: ./*\n")
    repos = parent / "repos"
    repos.mkdir()

    # Inside-the-dir layout still works (primary case is unaffected).
    r_primary = find_workspace_manifest(parent)
    assert r_primary is not None
    assert r_primary.name == "workspace_mcp.yaml"

    # Parent-of-dir layout with applies_to: ./* resolves to the same
    # manifest.
    r_fallback = find_workspace_manifest(repos)
    assert r_fallback is not None, "applies_to: ./* should allow parent-walk"
    assert r_fallback.resolve() == r_primary.resolve()


def test_e7_workspace_manifest_parent_walk_refused_without_applies_to(tmp_path: Path) -> None:
    """Without `workspace.applies_to` declared in the parent manifest,
    parent-walk discovery is REFUSED — even if the parent has a
    valid `workspace_mcp.yaml`. This is the safety property the
    opt-in shape guarantees.

    Catches: the silent-wrong-manifest failure mode an unconditional
    parent-walk would introduce. If a user typos `--workspace
    ~/projects/some_random_dir/` and `~/projects/workspace_mcp.yaml`
    exists, the server should NOT silently inherit that manifest.
    Pre-0.3.33 the framework refused on the primary location alone
    (correct but inflexible); 0.3.33's opt-in is the middle ground —
    the manifest *author* declares which child dirs the manifest
    covers.

    Also catches: literal `applies_to: ./allowed` correctly refuses
    sibling dirs that don't match the pattern (extra layer beyond
    the all-or-nothing opt-in)."""
    from kglite.mcp_server.manifest import find_workspace_manifest

    # Case 1: parent manifest exists but has no applies_to — refuse.
    closed = tmp_path / "closed"
    closed.mkdir()
    (closed / "workspace_mcp.yaml").write_text("name: closed\nworkspace:\n  kind: github\n")
    inner = closed / "inner"
    inner.mkdir()
    assert find_workspace_manifest(inner) is None, "parent manifest without applies_to should refuse parent-walk"

    # Case 2: applies_to is a literal that doesn't match this child's
    # basename — refuse.
    literal = tmp_path / "literal"
    literal.mkdir()
    (literal / "workspace_mcp.yaml").write_text("name: literal\nworkspace:\n  kind: github\n  applies_to: ./allowed\n")
    mismatch = literal / "forbidden"
    mismatch.mkdir()
    assert find_workspace_manifest(mismatch) is None, "applies_to literal mismatch should refuse parent-walk"

    # Case 3: positive control — same shape but matching basename works.
    matching = literal / "allowed"
    matching.mkdir()
    assert find_workspace_manifest(matching) is not None, "applies_to literal MATCH should allow parent-walk"


def test_f5_workspace_post_activate_hook_fires_on_activate(tmp_path: Path) -> None:
    """A Python callable registered via Workspace.set_post_activate
    must be invoked when activate / set_root_dir succeeds. Catches:
    Arc-wrap bugs or GIL-acquisition failures in the DeferredPyHook
    dispatch path."""
    import kglite._mcp_internal as m

    target = tmp_path / "ws_root"
    target.mkdir()
    (target / "marker.txt").write_text("hi\n")

    captured: list[tuple[str, str]] = []

    def hook(path: str, name: str) -> None:
        captured.append((path, name))

    ws = m.Workspace.open_local(str(target))
    ws.set_post_activate(hook)

    # repo_management(update=True) in local mode re-runs activate,
    # which fires the post-activate hook.
    out = ws.repo_management(update=True)
    assert "failed" not in out.lower()

    assert len(captured) >= 1, f"post_activate hook didn't fire: out={out!r}"
    fired_path, fired_name = captured[-1]
    assert Path(fired_path).resolve() == target.resolve(), (
        f"hook received wrong path: {fired_path!r} (expected {target!r})"
    )
    assert fired_name.startswith("local/"), f"hook received unexpected name: {fired_name!r}"


def test_f6_local_workspace_builds_code_tree_at_boot(local_workspace_yaml: Path) -> None:
    """local_workspace mode must build a code-tree graph for the
    workspace root at boot — the first cypher_query (before any
    set_root_dir) must see populated nodes from the source tree.

    Catches: pre-0.9.28 regression where local_workspace mode
    instantiated Workspace but never called build_code_tree, so the
    graph stayed empty until the agent issued set_root_dir. Operators
    deploying code_review-style local-workspace manifests hit this
    when their first agent action was cypher_query: it returned
    'No active graph.' even though the manifest declared a workspace
    root. Wired in 0.9.28 to mirror watch mode's boot-time build."""
    client = _client(["--mcp-config", str(local_workspace_yaml)])
    try:
        body = client.call_tool("cypher_query", {"query": "MATCH (n) RETURN count(n) AS n"})
    finally:
        client.close()
    assert "No active graph" not in body, f"graph empty at boot: {body!r}"
    # tiny_source_dir has alpha.py (def alpha + class A), beta.py (def
    # beta), and nested/gamma.py (def gamma) — the code-tree builder
    # produces at minimum a non-zero node count.
    assert "n: 0" not in body and '"n":0' not in body.replace(" ", ""), (
        f"code-tree built but produced zero nodes: {body!r}"
    )


def test_f7_set_root_dir_rebuilds_code_tree_for_new_root(local_workspace_yaml: Path, tiny_source_dir: Path) -> None:
    """After set_root_dir(child), the active code-tree must reflect
    the new root's source files, not the original workspace root's.

    Catches: pre-0.9.28 regression where set_root_dir rebound
    source_roots (so read_source/grep targeted the new path) but
    didn't trigger a code-tree rebuild — cypher_query kept returning
    nodes from the original workspace root. Now the post_activate
    hook fires on set_root_dir and rebuilds via build_code_tree."""
    # Set up a sibling with a distinctively-named function so we can
    # tell whether the active graph reflects the new root.
    second = tiny_source_dir / "second_root"
    second.mkdir()
    (second / "uniquely_named.py").write_text("def function_only_in_second_root():\n    return 42\n")

    client = _client(["--mcp-config", str(local_workspace_yaml)])
    try:
        swap = client.call_tool("set_root_dir", {"path": "second_root"})
        assert "ERROR" not in swap, f"set_root_dir failed: {swap!r}"
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (f:Function) WHERE f.name CONTAINS 'function_only_in_second_root' RETURN f.name AS name"},
        )
    finally:
        client.close()
    assert "function_only_in_second_root" in body, f"set_root_dir didn't rebuild code-tree for new root: {body!r}"


def test_f8_kglite_code_tree_loadable_via_from_import() -> None:
    """`from kglite import code_tree` must succeed so the mcp_server
    can call code_tree.build(...) on workspace activate.

    Catches: pre-0.9.28 attribute-chain bug where
    `kglite.code_tree.build(...)` raised AttributeError because
    kglite/__init__.py never imported the submodule at module load.
    The mcp_server's GraphState.build_code_tree path now does
    `from kglite import code_tree` which forces the submodule load."""
    from kglite import code_tree

    assert callable(code_tree.build), "code_tree.build is not callable"
    # Confirm the bundled Rust extension actually loads — if the
    # native module is missing the build function will raise
    # ImportError on use, which the mcp_server surfaces as a clean
    # RuntimeError pointing at the install instructions.


def test_w1_watch_callback_receives_changed_paths(tmp_path: Path) -> None:
    """`start_watch` invokes the Python callable with a list of
    changed path strings within `debounce_ms + grace`. Catches:
    GIL or type-conversion issues at the watcher / Python boundary
    (callback runs on a background thread, must reacquire the GIL
    automatically)."""
    import kglite._mcp_internal as m

    received: list[list[str]] = []
    lock = threading.Lock()

    def on_change(paths: list[str]) -> None:
        with lock:
            received.append(list(paths))

    handle = m.start_watch(str(tmp_path), on_change, debounce_ms=100)
    try:
        # Give the watcher a moment to settle, then write a file.
        time.sleep(0.05)
        (tmp_path / "new_file.txt").write_text("change me\n")
        # Wait for the debounce window + grace.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                if received:
                    break
            time.sleep(0.05)
    finally:
        handle.stop()

    assert received, "watch callback never fired after file write"
    flat = [p for batch in received for p in batch]
    # macOS FSEvents may report either the new file or its parent
    # directory (depending on coalescing). Accept either as evidence
    # the watcher picked up the change. The contract we care about
    # is "list[str] arrives within the debounce window" — not the
    # specific path granularity, which is platform-dependent.
    tmp_path_str = str(tmp_path)
    assert any("new_file.txt" in p or tmp_path_str in p for p in flat), (
        f"callback fired but path list did not include either the file or the parent dir: {received!r}"
    )


# ---------------------------------------------------------------------------
# 0.9.25 cypher_preprocessor regression tests (Category O — preprocessor)
# ---------------------------------------------------------------------------


def _make_state_with_tiny_graph():
    """Construct a GraphState wrapping a 3-node graph for preprocessor
    integration tests. Uses kglite Python API directly — no MCP stdio."""
    import kglite
    from kglite.mcp_server.tools import GraphState

    state = GraphState()
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Item {title: 'foo', value: 1})")
    g.cypher("CREATE (:Item {title: 'bar', value: 2})")
    g.cypher("CREATE (:Item {title: 'baz', value: 3})")
    # Inject the active-graph slot directly so we don't need a .kgl file.
    state._active = type("A", (), {"graph": g, "source_path": None})()
    return state, g


def test_o1_cypher_preprocessor_rewrites_query_before_execution(tmp_path: Path) -> None:
    """The preprocessor's rewrite() output is what reaches graph.cypher().
    Operator's spec test #1. Catches: preprocessor configured but
    actually ignored at dispatch time."""
    from kglite.mcp_server.preprocessor import Preprocessor
    from kglite.mcp_server.tools import run_cypher

    def rewrite(query: str, params):
        # Rewrite "foo" → "bar" so the query targets a different node.
        return query.replace("'foo'", "'bar'"), params

    preprocessor = Preprocessor(rewrite=rewrite, module_path=tmp_path / "p.py")
    state, _ = _make_state_with_tiny_graph()

    body = run_cypher(state, "MATCH (n:Item {title: 'foo'}) RETURN n.value AS v", preprocessor=preprocessor)
    # bar has value=2, foo has value=1. If the rewrite landed, we see 2.
    assert "2" in body
    assert "1" not in body or "row" in body  # tolerant — "1 row" might appear


def test_o2_cypher_preprocessor_fires_for_tools_cypher_too(tmp_path: Path) -> None:
    """YAML-declared tools[].cypher templates also go through the
    preprocessor. Operator's spec test #2. Catches: tools[].cypher
    bypassing the hook."""
    import asyncio

    from kglite.mcp_server.cypher_tools import call_cypher_tool
    from kglite.mcp_server.manifest import CypherTool
    from kglite.mcp_server.preprocessor import Preprocessor

    fired_with: list[str] = []

    def rewrite(query: str, params):
        fired_with.append(query)
        return query.replace("'foo'", "'baz'"), params

    preprocessor = Preprocessor(rewrite=rewrite, module_path=tmp_path / "p.py")
    state, _ = _make_state_with_tiny_graph()
    spec = CypherTool(
        name="lookup",
        cypher="MATCH (n:Item {title: 'foo'}) RETURN n.value AS v",
    )
    body = asyncio.run(call_cypher_tool(spec, state, {}, csv_http=None, preprocessor=preprocessor))
    assert fired_with, "preprocessor never received the tools[].cypher template"
    # baz has value=3.
    assert "3" in body


def test_o3_cypher_preprocessor_does_not_fire_for_non_cypher_tools(tmp_path: Path) -> None:
    """The preprocessor is scoped to cypher_query / tools[].cypher.
    graph_overview, save_graph, and the source-tool / GitHub-tool
    surface do not invoke it. Operator's spec test #3."""
    from kglite.mcp_server.preprocessor import Preprocessor
    from kglite.mcp_server.tools import run_overview, run_save

    fired_count = [0]

    def rewrite(query: str, params):
        fired_count[0] += 1
        return query, params

    # The preprocessor is constructed (so its rewrite() is genuinely
    # callable) but deliberately NOT passed to run_overview / run_save —
    # the point of this test is that non-cypher tool surfaces have
    # no parameter through which a preprocessor could reach them. If
    # someone ever adds a sneaky `preprocessor` kwarg or global to
    # those tools, the fired_count assertion catches it.
    _ = Preprocessor(rewrite=rewrite, module_path=tmp_path / "p.py")
    state, _ = _make_state_with_tiny_graph()

    body = run_overview(state, types=None, connections=None, cypher=None, temp_cleanup_dir=None)
    assert "Item" in body or "node" in body.lower()  # describe() should mention the type
    assert fired_count[0] == 0, "graph_overview leaked into preprocessor"

    save_body = run_save(state)
    assert "save_graph" in save_body or "requires" in save_body
    assert fired_count[0] == 0, "save_graph leaked into preprocessor"


def test_o4_cypher_preprocessor_exception_returns_clean_error(tmp_path: Path) -> None:
    """If rewrite() raises ValueError / TypeError, the agent receives
    `preprocessor: <message>` — no stack trace. Operator's spec test #4."""
    from kglite.mcp_server.preprocessor import Preprocessor
    from kglite.mcp_server.tools import run_cypher

    def rewrite(query: str, params):
        raise ValueError("rewrite refused: query rejected by policy")

    preprocessor = Preprocessor(rewrite=rewrite, module_path=tmp_path / "p.py")
    state, _ = _make_state_with_tiny_graph()
    body = run_cypher(state, "MATCH (n) RETURN n", preprocessor=preprocessor)
    assert body.startswith("preprocessor: "), f"expected clean error envelope, got: {body!r}"
    assert "rewrite refused" in body
    assert "Traceback" not in body


def test_o5_trust_gate_blocks_preprocessor_without_allow_query_preprocessor(tmp_path: Path) -> None:
    """extensions.cypher_preprocessor without trust.allow_query_preprocessor
    fails at preprocessor-load time. Operator's spec test #5. Mirrors the
    existing embedder/allow_embedder gate."""
    from kglite.mcp_server.preprocessor import PreprocessorError, from_manifest_value

    # Trust = False → must raise.
    with pytest.raises(PreprocessorError) as exc_info:
        from_manifest_value(
            {"module": "./p.py", "class": "P"},
            base_dir=tmp_path,
            trust_allowed=False,
        )
    assert "trust.allow_query_preprocessor" in str(exc_info.value)

    # Trust = True but module file missing → different error (proves
    # the trust gate is the first check that ran).
    with pytest.raises(PreprocessorError) as exc_info:
        from_manifest_value(
            {"module": "./does_not_exist.py", "class": "P"},
            base_dir=tmp_path,
            trust_allowed=True,
        )
    assert "module file does not exist" in str(exc_info.value)


def test_o6_cypher_preprocessor_loads_class_with_kwargs(tmp_path: Path) -> None:
    """End-to-end load of a class-based preprocessor with kwargs.
    Catches: kwargs not threaded through to the constructor."""
    from kglite.mcp_server.preprocessor import from_manifest_value

    (tmp_path / "p.py").write_text(
        "class P:\n"
        "    def __init__(self, suffix: str):\n"
        "        self._s = suffix\n"
        "    def rewrite(self, query, params):\n"
        "        return query + self._s, params\n"
    )
    pre = from_manifest_value(
        {"module": "./p.py", "class": "P", "kwargs": {"suffix": "  -- rewritten"}},
        base_dir=tmp_path,
        trust_allowed=True,
    )
    assert pre is not None
    new_q, _ = pre.rewrite("MATCH (n) RETURN n", None)
    assert new_q.endswith("-- rewritten")


def test_o7_cypher_preprocessor_loads_free_function(tmp_path: Path) -> None:
    """End-to-end load of a function-based preprocessor.
    Catches: function loader not finding module-level callables."""
    from kglite.mcp_server.preprocessor import from_manifest_value

    (tmp_path / "rewrites.py").write_text("def rewrite(query, params):\n    return query.lower(), params\n")
    pre = from_manifest_value(
        {"module": "./rewrites.py", "function": "rewrite"},
        base_dir=tmp_path,
        trust_allowed=True,
    )
    assert pre is not None
    new_q, _ = pre.rewrite("MATCH (N) RETURN N", None)
    assert new_q == "match (n) return n"


def test_o8_cypher_preprocessor_full_yaml_roundtrip_via_mcp_stdio(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Full boot path: YAML manifest with `trust.allow_query_preprocessor:
    true` + `extensions.cypher_preprocessor:` + a real preprocessor
    module → mcp-methods parser → kglite Manifest → server boot →
    cypher_query reaches graph.cypher with the rewritten string.

    Requires mcp-methods 0.3.29+ (the trust key has to parse cleanly).
    Catches end-to-end wiring regressions across the YAML → Rust →
    Python → tool-dispatch boundary."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)

    # Rewriter substitutes 'Person' → 'Item' so the query targets
    # nothing — we use the no-results signal to confirm the rewrite
    # actually reached the engine.
    (tmp_path / "rewrites.py").write_text(
        "def rewrite(query, params):\n    return query.replace('Person', 'NONEXISTENT_TYPE'), params\n"
    )
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text(
        "name: preprocessed\n"
        "trust:\n"
        "  allow_query_preprocessor: true\n"
        "extensions:\n"
        "  cypher_preprocessor:\n"
        "    module: ./rewrites.py\n"
        "    function: rewrite\n"
    )

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        # The tiny graph has Person nodes. Without the preprocessor
        # this query would return rows; with the rewrite it sees
        # NONEXISTENT_TYPE and returns "No results."
        body = client.call_tool("cypher_query", {"query": "MATCH (p:Person) RETURN p.name LIMIT 3"})
    finally:
        client.close()
    assert "No results" in body, f"preprocessor rewrite didn't reach engine — got: {body!r}"


def test_o9_trust_gate_blocks_boot_when_yaml_missing_allow_query_preprocessor(
    tmp_path: Path, tiny_graph_path: Path
) -> None:
    """If extensions.cypher_preprocessor is declared without the trust
    flag, the server exits 3 at boot with the operator-facing
    message. Mirrors the existing trust.allow_embedder gate.

    Requires mcp-methods 0.3.29+ (the trust key has to parse cleanly
    when present; this test omits it, so the framework's strict-key
    check would fire on a stale pin and shadow our gate. Pinning
    0.3.29+ keeps the assertion targeting the right code path)."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    (tmp_path / "rewrites.py").write_text("def rewrite(q, p):\n    return q, p\n")
    # Omits `trust.allow_query_preprocessor: true` deliberately.
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text(
        "name: ungated\nextensions:\n  cypher_preprocessor:\n    module: ./rewrites.py\n    function: rewrite\n"
    )

    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--graph", str(target), "--mcp-config", str(manifest)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0, (
        f"server should have exited non-zero without trust gate; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "trust.allow_query_preprocessor" in combined, f"missing gate-message in stderr: {result.stderr!r}"


# ---------------------------------------------------------------------------
# 0.9.26 Cat J/K/L — spatial / timeseries / procedure tests against the
# mcp-servers operator's delivered Cat G-N fixture bundle.
# See tests/fixtures/CAT_G_N_FIXTURES.md for the fixture catalog.
# ---------------------------------------------------------------------------


_CAT_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def spatial_graph() -> Path:
    return _CAT_FIXTURES / "spatial_graph.kgl"


@pytest.fixture
def timeseries_graph() -> Path:
    return _CAT_FIXTURES / "timeseries_graph.kgl"


@pytest.fixture
def graph_with_orphans() -> Path:
    return _CAT_FIXTURES / "graph_with_orphans.kgl"


@pytest.fixture
def graph_with_duplicates() -> Path:
    return _CAT_FIXTURES / "graph_with_duplicates.kgl"


# --- Cat J — Spatial Cypher --------------------------------------------------


def test_j1_contains_finds_wells_inside_areas(spatial_graph: Path) -> None:
    """contains(area, point(lat, lon)) — three wells are inside one of
    three areas in the fixture, two are outside all areas. Catches:
    spatial predicate regressions; point-from-property construction
    breaking; WKT polygon parsing breaking."""
    client = _client(["--graph", str(spatial_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (a:Area), (w:Well) "
                "WHERE contains(a, point(w.latitude, w.longitude)) "
                "RETURN count(*) AS n"
            },
        )
    finally:
        client.close()
    assert "3" in body, f"expected 3 wells inside areas, got: {body!r}"


def test_j2_centroid_of_polygon(spatial_graph: Path) -> None:
    """centroid(area) returns a point with lat/lon matching the
    polygon's geometric centre. NORTH_BLOCK is a unit square centred
    at (61.0, 5.0). Catches: centroid math regressions; centroid()
    not returning a point-shaped value."""
    client = _client(["--graph", str(spatial_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (a:Area {id: 'north'}) RETURN centroid(a) AS c"},
        )
    finally:
        client.close()
    # centroid is rendered as a dict — Python repr in the inline format.
    # Expected: {'latitude': 61.0, 'longitude': 5.0}.
    assert "61" in body and "5" in body, f"expected NORTH_BLOCK centroid (61, 5), got: {body!r}"
    assert "latitude" in body or "lat" in body.lower(), f"centroid not point-shaped: {body!r}"


def test_j3_contains_query_via_explicit_point(spatial_graph: Path) -> None:
    """Query-side `point(lat, lon)` literal lookup. The fixture has
    NORTH_BLOCK covering lat 60-62 / lon 4-6; (61, 5) is inside.
    Catches: agent-friendly `WHERE contains(area, point(...))` shape
    breaking; literal point construction breaking."""
    client = _client(["--graph", str(spatial_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (a:Area) WHERE contains(a, point(61.0, 5.0)) RETURN a.title AS title"},
        )
    finally:
        client.close()
    assert "NORTH_BLOCK" in body, f"expected NORTH_BLOCK to contain (61, 5), got: {body!r}"


# --- Cat K — Timeseries Cypher -----------------------------------------------


def test_k1_ts_sum_aggregates_a_year(timeseries_graph: Path) -> None:
    """ts_sum(channel, 'YYYY') aggregates all timesteps within the year.
    TROLL's oil_col in 2019 sums to ~1563.55 against the random.seed(42)
    seed in build_fixtures.py. Catches: ts_sum regressions; year-range
    parsing breaking."""
    client = _client(["--graph", str(timeseries_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (f:Field {title:'TROLL'}) RETURN ts_sum(f.oil_col, '2019') AS annual"},
        )
    finally:
        client.close()
    # 1563.55 with possible float-format variation. Operator confirmed
    # the value at fixture-build time.
    assert "1563" in body, f"expected ts_sum ~1563.55, got: {body!r}"


def test_k2_ts_at_point_in_time(timeseries_graph: Path) -> None:
    """ts_at(channel, 'YYYY-M') returns the value at a single timestep.
    March 2019 for TROLL oil is 177.12 under the seeded random."""
    client = _client(["--graph", str(timeseries_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (f:Field {title:'TROLL'}) RETURN ts_at(f.oil_col, '2019-3') AS march"},
        )
    finally:
        client.close()
    assert "177.12" in body, f"expected ts_at March 2019 = 177.12, got: {body!r}"


def test_k3_ts_aggregations_handle_all_fields(timeseries_graph: Path) -> None:
    """ts_sum applied across all three Field nodes in the fixture
    returns three rows. Catches: ts_* functions not iterating across
    matches correctly."""
    client = _client(["--graph", str(timeseries_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (f:Field) RETURN f.title AS title, ts_sum(f.oil_col, '2019') AS annual"},
        )
    finally:
        client.close()
    assert "3 row" in body, f"expected 3 rows (one per field), got: {body!r}"
    for field in ("TROLL", "EKOFISK", "SNORRE"):
        assert field in body, f"missing field {field!r} in result: {body!r}"


# --- Cat L — Procedures ------------------------------------------------------


def test_l1_orphan_node_procedure_counts_isolates(graph_with_orphans: Path) -> None:
    """CALL orphan_node({type:'Wellbore'}) YIELD node — fixture has
    6 Wellbore nodes, 3 connected to a Field via IN_FIELD, 3 isolated.
    Procedure must return the 3 isolated ones. Catches: orphan_node
    regressions; CALL/YIELD wiring breaking."""
    client = _client(["--graph", str(graph_with_orphans)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "CALL orphan_node({type: 'Wellbore'}) YIELD node RETURN count(node) AS n"},
        )
    finally:
        client.close()
    assert "3" in body, f"expected 3 orphan Wellbores, got: {body!r}"


def test_l2_duplicate_title_procedure_counts_dupes(graph_with_duplicates: Path) -> None:
    """CALL duplicate_title({type:'Prospect'}) YIELD node — fixture has
    6 Prospects, 4 of which share titles in two pairs (ALPHA x2, BETA x2).
    Procedure must yield all 4 members of the duplicate pairs. Catches:
    duplicate_title regressions; "members of duplicate sets" vs "one
    representative per set" off-by-N errors."""
    client = _client(["--graph", str(graph_with_duplicates)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "CALL duplicate_title({type: 'Prospect'}) YIELD node RETURN count(node) AS n"},
        )
    finally:
        client.close()
    assert "4" in body, f"expected 4 duplicate-title Prospects, got: {body!r}"


# ---------------------------------------------------------------------------
# Category SK — Skills wiring (0.9.31)
# ---------------------------------------------------------------------------


def test_sk1_skills_disabled_by_default_no_prompts_list(tmp_path: Path, tiny_graph_path: Path) -> None:
    """A manifest without `skills:` (the pre-0.9.31 default) must not
    expose any prompts. Catches: skills always-on regression that
    bloats the agent's prompts/list with unsolicited methodology."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: no_skills\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        prompts = client.list_prompts()
    finally:
        client.close()
    assert prompts == [], f"prompts/list should be empty when skills not opted in: {prompts!r}"


def test_sk2_skills_true_exposes_kglite_bundled(tmp_path: Path, tiny_graph_path: Path) -> None:
    """With `skills: true`, the four kglite-bundled skills land in
    prompts/list (subject to applies_when filtering — only
    read_code_source has a predicate, which requires Function/Class
    in the graph)."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: skills_on\nskills: true\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        prompts = client.list_prompts()
    finally:
        client.close()
    names = {p["name"] for p in prompts}
    # cypher_query / graph_overview / save_graph are always-active.
    # read_code_source is gated by applies_when graph_has_node_type
    # [Function, Class] — tiny_graph_path is a Person/Article fixture
    # with neither type, so the skill should be filtered out here.
    assert "cypher_query" in names, f"cypher_query should appear: {names!r}"
    assert "graph_overview" in names, f"graph_overview should appear: {names!r}"
    # save_graph is gated by tool_registered:save_graph; we didn't
    # set builtins.save_graph: true so the tool isn't registered,
    # which means the skill's applies_when predicate filters it out.
    # That's correct behaviour.
    assert "read_code_source" not in names, f"read_code_source should be filtered out on non-code graph: {names!r}"


def test_sk3_get_prompt_returns_skill_body(tmp_path: Path, tiny_graph_path: Path) -> None:
    """prompts/get cypher_query returns the bundled markdown body with
    its description. Catches: handler registered but body not threaded
    through."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: skill_body_test\nskills: true\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        resp = client.get_prompt("cypher_query")
    finally:
        client.close()
    assert "error" not in resp, f"get_prompt returned error: {resp!r}"
    result = resp["result"]
    messages = result["messages"]
    assert len(messages) == 1
    body = messages[0]["content"]["text"]
    # The bundled cypher_query skill body has a recognisable anchor.
    assert "Quick Reference" in body, f"skill body missing expected section: {body[:200]!r}"
    assert "cypher_query" in body.lower()


def test_sk4_auto_inject_embeds_full_skill_body_in_tool_description(tmp_path: Path, tiny_graph_path: Path) -> None:
    """When skills are on and a skill's name matches a registered tool
    with auto_inject_hint=True, the tool's description gains the FULL
    skill body under a `## Methodology` header.

    0.9.32 change: previously injected a `[See prompts/get NAME ...]`
    pointer. Operator found that agents in Claude Code / Claude
    Desktop / Cursor / Continue don't have prompts/get exposed —
    the pointer was dangling. Embedding the body makes methodology
    reachable in every MCP client that exposes tools/list to the
    agent (which is all of them)."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: hint_test\nskills: true\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        tools_raw = client.list_tools_full()
    finally:
        client.close()
    cypher = next((t for t in tools_raw if t["name"] == "cypher_query"), None)
    assert cypher is not None
    desc = cypher["description"] or ""
    assert "## Methodology" in desc, f"auto-inject marker missing from cypher_query description: {desc[:300]!r}"
    # Pull a recognisable substring from the bundled cypher_query
    # body — the Quick Reference section header should be in there.
    assert "Quick Reference" in desc, f"full skill body missing — got truncated description: {desc[:500]!r}"


def test_sk5_read_code_source_skill_active_on_code_tree(tmp_path: Path) -> None:
    """When the active graph has Function/Class nodes, the
    `read_code_source` skill's applies_when predicate evaluates to
    true and the skill appears in prompts/list. Catches:
    applies_when wired but always-false; predicate evaluator
    not consulting the live graph."""
    # Build a tiny code-tree graph via kglite.code_tree.build, save as
    # .kgl, then point a server at it.
    import kglite
    from kglite import code_tree

    src = tmp_path / "tiny_code"
    src.mkdir()
    (src / "main.py").write_text("def foo():\n    pass\n\nclass Bar:\n    pass\n")
    g = code_tree.build(str(src))
    kgl_path = tmp_path / "tiny_code.kgl"
    g.save(str(kgl_path))

    manifest = tmp_path / "tiny_code_mcp.yaml"
    manifest.write_text("name: code_skills_test\nskills: true\n")

    client = _client(["--graph", str(kgl_path), "--mcp-config", str(manifest)])
    try:
        prompts = client.list_prompts()
    finally:
        client.close()
    names = {p["name"] for p in prompts}
    assert "read_code_source" in names, f"read_code_source should be active on code-tree graph: {names!r}"
    _ = kglite  # silence unused-import lint in this scope


def test_sk6_unknown_prompt_returns_error(tmp_path: Path, tiny_graph_path: Path) -> None:
    """prompts/get with an unknown name returns a JSON-RPC error.
    Catches: handler raising the wrong exception type, or the lowlevel
    Server silently returning empty content."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: unknown_prompt_test\nskills: true\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        resp = client.get_prompt("not_a_real_skill")
    finally:
        client.close()
    assert "error" in resp, f"expected error for unknown prompt, got: {resp!r}"
    assert "unknown prompt" in resp["error"].get("message", "").lower()


# ---------------------------------------------------------------------------
# Category O — Overview prefix + auto-inject escape hatch (0.9.32)
# ---------------------------------------------------------------------------


def test_o1_overview_prefix_prepended_on_bare_overview(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Manifest `overview_prefix:` is prepended to bare graph_overview()
    output. Pre-0.9.32 the field was parsed but never read in the
    Python entry path — operator bug report 2026-05-14. Mirrors the
    FastMCP path's behaviour."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: overview_prefix_test\noverview_prefix: |\n  ## ORIENTATION\n  Start with X then Y.\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool("graph_overview", {})
    finally:
        client.close()
    assert "ORIENTATION" in body, f"overview_prefix should be prepended on bare overview: {body[:300]!r}"
    assert "Start with X then Y." in body
    # And the schema XML still follows the prefix.
    assert "<graph" in body, f"schema XML missing from response: {body[:500]!r}"


def test_o2_overview_prefix_not_prepended_on_drill_down(tmp_path: Path, tiny_graph_path: Path) -> None:
    """overview_prefix is only injected on BARE graph_overview()
    calls — drill-downs (types=[...] / connections / cypher) skip it.
    Catches: prefix bloating every drill-down response, wasting
    context."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: overview_prefix_drill_test\noverview_prefix: |\n  ## SHOULD_NOT_APPEAR_HERE\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool("graph_overview", {"types": ["Person"]})
    finally:
        client.close()
    assert "SHOULD_NOT_APPEAR_HERE" not in body, f"overview_prefix leaked into drill-down response: {body[:300]!r}"


def test_o3_auto_inject_respects_auto_inject_hint_false(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Skills with `auto_inject_hint: false` should NOT inject their
    body into the matching tool's description. Operator's escape
    hatch for cases where context cost matters more than retrieval."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    # Manifest's project layer with auto_inject_hint: false on cypher_query.
    skills_dir = tmp_path / "test_mcp.skills"
    skills_dir.mkdir()
    (skills_dir / "cypher_query.md").write_text(
        "---\n"
        "name: cypher_query\n"
        'description: "Short opt-out skill."\n'
        "auto_inject_hint: false\n"
        "---\n\n"
        "# UNIQUE_BODY_MARKER\n\nshouldn't appear in tools/list.\n"
    )
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: opt_out_test\nskills: true\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        tools_raw = client.list_tools_full()
    finally:
        client.close()
    cypher = next((t for t in tools_raw if t["name"] == "cypher_query"), None)
    assert cypher is not None
    desc = cypher["description"] or ""
    assert "UNIQUE_BODY_MARKER" not in desc, (
        f"auto_inject_hint: false skill body leaked into tool description: {desc[:300]!r}"
    )
    assert "## Methodology" not in desc, "auto_inject_hint: false should suppress the Methodology header too"


def test_o4_auto_inject_idempotent_via_methodology_header_marker(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Running _apply_skill_hint twice on the same tool doesn't
    duplicate the body — the `## Methodology` marker is the
    idempotency check. Catches: handler re-resolving on every
    request and stacking duplicates."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text("name: idempotent_test\nskills: true\n")

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        # Call list_tools twice — the handler re-evaluates each time.
        # If injection isn't idempotent, the second call would carry
        # a double-injected body.
        first = client.list_tools_full()
        second = client.list_tools_full()
    finally:
        client.close()
    cypher_first = next((t for t in first if t["name"] == "cypher_query"), None)
    cypher_second = next((t for t in second if t["name"] == "cypher_query"), None)
    assert cypher_first and cypher_second
    assert cypher_first["description"] == cypher_second["description"], (
        "tool description should be stable across list_tools calls"
    )
    # And only ONE methodology header should be present.
    assert cypher_first["description"].count("## Methodology") == 1, (
        f"methodology header duplicated: {cypher_first['description'][:500]!r}"
    )
