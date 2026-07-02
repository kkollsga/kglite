"""End-to-end smoke tests for the kglite-mcp-server Rust binary.

Drives the binary over JSON-RPC stdio (the way Claude Desktop / Cursor
do) and exercises every tool the server exposes — so we catch boot
failures, missing tools, and per-tool argument-shape regressions
before users do.

Tests are skipped when the binary isn't built. Build it with::

    cargo build -p kglite-mcp-server --release

GitHub-token-gated tools (`github_issues`, `github_api`) are exercised
when ``GITHUB_TOKEN`` is set in the environment, OR when a sibling
``mcp-methods/.env`` exists with one (the same walk-up mcp-server does).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Optional

import pandas as pd
import pytest

import kglite

BINARY = Path(__file__).resolve().parent.parent / "target" / "release" / "kglite-mcp-server"


pytestmark = pytest.mark.skipif(
    not BINARY.exists(),
    reason=f"kglite-mcp-server binary not built (missing at {BINARY}). "
    f"Build with `cargo build -p kglite-mcp-server --release`.",
)


def _discover_github_token() -> Optional[str]:
    """Look for a GitHub token in env, then fall back to the sibling
    `mcp-methods/.env` (the same lookup the binary itself does at boot)."""
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "mcp-methods" / ".env",
    ]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("GITHUB_TOKEN", "GH_TOKEN"):
                return value.strip().strip("\"'")
    return None


GITHUB_TOKEN = _discover_github_token()


# ── Fixture builders ──────────────────────────────────────────────────────


def _build_fixture_graph(path: Path) -> None:
    """Build a small Person/KNOWS graph, save to ``path``."""
    g = kglite.KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Alice", "Bob", "Carol", "Dave"],
            "city": ["Oslo", "Bergen", "Oslo", "Trondheim"],
        }
    )
    g.add_nodes(nodes, "Person", "id", "title")
    edges = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 3, 4]})
    g.add_connections(edges, "KNOWS", "Person", "src", "Person", "dst")
    g.save(str(path))


def _write_savegraph_manifest(path: Path) -> Path:
    """Drop a minimal manifest with `builtins.save_graph: true` so the
    server registers the save_graph tool. Matches the canonical
    opt-in pattern in `test_mcp_server_python_entry.py` (`ff5cc91`
    made save_graph opt-in to avoid exposing a destructive operation
    on every server boot)."""
    manifest = path / "smoke_mcp.yaml"
    manifest.write_text("name: smoke\nbuiltins:\n  save_graph: true\n")
    return manifest


# ── JSON-RPC stdio client ─────────────────────────────────────────────────


class McpClient:
    """Minimal JSON-RPC 2.0 / NDJSON client for an MCP stdio server."""

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self.proc = proc
        self._next_id = 0
        # Drain stderr in the background so the subprocess buffer doesn't fill up
        # if the server logs verbosely. We don't assert against stderr — just
        # collect it for diagnostics on failure.
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in iter(self.proc.stderr.readline, b""):
            self._stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())

    def _allocate_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, payload: dict[str, Any]) -> None:
        line = (json.dumps(payload) + "\n").encode("utf-8")
        assert self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def _recv(self, expected_id: int, timeout_s: float = 30.0) -> dict[str, Any]:
        """Read NDJSON responses from stdout until one matching `expected_id`
        comes back. Notifications and other ids are buffered/ignored."""
        deadline = time.monotonic() + timeout_s
        assert self.proc.stdout is not None
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                # EOF — server died; surface stderr in the assertion msg.
                stderr_tail = "\n".join(self._stderr_lines[-20:])
                raise RuntimeError(f"Server exited unexpectedly. Last stderr:\n{stderr_tail}")
            try:
                msg = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                # Skip non-JSON lines (server may emit log lines on stdout
                # by mistake; we'd rather not assume).
                continue
            if msg.get("id") == expected_id:
                return msg
            # Otherwise it's a notification or unrelated response — drop it.
        raise TimeoutError(f"Timed out waiting for response id={expected_id}")

    def initialize(self) -> dict[str, Any]:
        rid = self._allocate_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "kglite-smoke-test", "version": "0"},
                },
            }
        )
        resp = self._recv(rid)
        # Stash so callers can assert on serverInfo / instructions after _spawn.
        self.init_result = resp
        # Initialized notification (no id).
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    def list_tools(self) -> list[dict[str, Any]]:
        rid = self._allocate_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": "tools/list"})
        resp = self._recv(rid)
        if "error" in resp:
            raise RuntimeError(f"tools/list errored: {resp['error']}")
        return resp["result"]["tools"]

    def call_tool(self, name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        rid = self._allocate_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            }
        )
        resp = self._recv(rid)
        if "error" in resp:
            raise RuntimeError(f"tools/call({name}) errored: {resp['error']}")
        return resp["result"]

    def shutdown(self) -> None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


def _spawn(
    args: list[str],
    cwd: Optional[Path] = None,
    env_extra: Optional[dict[str, str]] = None,
    env_remove: Optional[list[str]] = None,
) -> McpClient:
    env = os.environ.copy()
    if env_remove:
        for key in env_remove:
            env.pop(key, None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(
        [str(BINARY), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    client = McpClient(proc)
    client.initialize()
    return client


def _text_content(result: dict[str, Any]) -> str:
    """Extract the joined text from a tools/call result envelope."""
    parts = result.get("content", [])
    text_parts = [p["text"] for p in parts if p.get("type") == "text"]
    return "\n".join(text_parts)


# ── Test: --graph mode (kglite-side tools + ping) ─────────────────────────


@pytest.fixture
def graph_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture.kgl"
    _build_fixture_graph(fixture)
    return fixture


class TestGraphMode:
    """`--graph X.kgl` registers kglite tools + auto-binds the .kgl's parent
    directory as a source root, so source tools are also live."""

    def test_lists_expected_tools(self, graph_fixture: Path, tmp_path: Path):
        # save_graph is opt-in via `builtins.save_graph: true` since
        # `ff5cc91` (May 17, 2026). Without the manifest the server
        # registers everything except save_graph.
        manifest = _write_savegraph_manifest(tmp_path)
        client = _spawn(["--graph", str(graph_fixture), "--mcp-config", str(manifest)])
        try:
            tools = client.list_tools()
            names = {t["name"] for t in tools}
            # kglite-side tools (always registered)
            assert "ping" in names
            assert "cypher_query" in names
            assert "graph_overview" in names
            assert "save_graph" in names
            # Source tools are auto-registered because --graph mode binds
            # the .kgl's parent directory as a source root (see main.rs).
            assert "read_source" in names
            assert "grep" in names
            assert "list_source" in names
        finally:
            client.shutdown()

    def test_ping(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("ping")
            assert "pong" in _text_content(r).lower()
        finally:
            client.shutdown()

    def test_no_discovery_steer_in_graph_mode(self, graph_fixture: Path):
        """The lazy-tool-discovery steer is scoped to workspace modes; a
        single-graph deployment doesn't get it appended (nothing to steer —
        no code-tree exploration surface)."""
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            instructions = client.init_result["result"].get("instructions") or ""
            assert "ALWAYS registered" not in instructions
        finally:
            client.shutdown()

    def test_cypher_query_count(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("cypher_query", {"query": "MATCH (p:Person) RETURN count(p) AS n"})
            text = _text_content(r)
            assert "4" in text  # 4 Person nodes in fixture
        finally:
            client.shutdown()

    def test_cypher_query_traversal(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool(
                "cypher_query",
                {"query": "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title AS who, b.title AS knows ORDER BY who"},
            )
            text = _text_content(r)
            # 3 KNOWS edges in fixture
            assert "Alice" in text and "Bob" in text
            assert "3 row(s)" in text
        finally:
            client.shutdown()

    def test_cypher_query_format_csv(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool(
                "cypher_query",
                {"query": "MATCH (p:Person) RETURN p.title AS name, p.city AS city ORDER BY name FORMAT CSV"},
            )
            text = _text_content(r)
            # CSV output — header line + data rows
            assert "name" in text and "city" in text
            assert "Alice" in text and "Oslo" in text
        finally:
            client.shutdown()

    def test_graph_overview_inventory(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("graph_overview")
            text = _text_content(r)
            # describe() returns XML — at minimum it should reference our type.
            assert "Person" in text
        finally:
            client.shutdown()

    def test_graph_overview_drill_down(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("graph_overview", {"types": ["Person"]})
            text = _text_content(r)
            assert "Person" in text
            # Drill-down should mention property names from our fixture.
            assert "city" in text or "title" in text
        finally:
            client.shutdown()

    def test_readonly_rejects_mutation(self, graph_fixture: Path):
        # Default (no --writable): mutations are rejected.
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("cypher_query", {"query": "CREATE (:Task {id: 't1'})"})
            text = _text_content(r)
            assert "not allowed" in text.lower() or "mutation" in text.lower()
        finally:
            client.shutdown()


class TestWritableMode:
    """`--graph X.kgl --writable` = the agent graph workbench: cypher_query
    accepts (scoped) mutations + the lifecycle tools are registered."""

    def test_lists_lifecycle_tools(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture), "--writable"])
        try:
            names = {t["name"] for t in client.list_tools()}
            assert {"load_graph", "create_graph", "save_graph_as"} <= names
            # save_graph is implied by --writable (no manifest needed).
            assert "save_graph" in names
        finally:
            client.shutdown()

    def test_accepts_scoped_mutation_and_reads_back(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture), "--writable"])
        try:
            client.call_tool(
                "cypher_query",
                {"query": "CREATE (:Task {id: 't1', status: 'todo'})", "write_scope": ["Task"]},
            )
            r = client.call_tool("cypher_query", {"query": "MATCH (t:Task) RETURN count(t) AS n"})
            assert "1" in _text_content(r)
        finally:
            client.shutdown()

    def test_write_scope_blocks_out_of_scope(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture), "--writable"])
        try:
            r = client.call_tool(
                "cypher_query",
                {"query": "CREATE (:Algorithm {id: 'a1'})", "write_scope": ["Task"]},
            )
            assert "write scope" in _text_content(r).lower()
        finally:
            client.shutdown()

    def test_save_graph_round_trip(self, tmp_path: Path):
        # Use a copy so we don't churn the shared fixture across tests.
        src = tmp_path / "saveable.kgl"
        _build_fixture_graph(src)
        mtime_before = src.stat().st_mtime

        # save_graph is opt-in — see _write_savegraph_manifest. This
        # is the in-memory `.kgl` round-trip; the disk-mode variant
        # is in `test_mcp_server_python_entry.py`. Both are needed to
        # cover the dispatch in `kglite::api::save_graph`.
        manifest = _write_savegraph_manifest(tmp_path)
        client = _spawn(["--graph", str(src), "--mcp-config", str(manifest)])
        try:
            time.sleep(0.05)  # so save mtime is detectably newer
            r = client.call_tool("save_graph")
            text = _text_content(r)
            assert "Saved" in text and "node" in text  # message format check
            assert src.stat().st_mtime > mtime_before
        finally:
            client.shutdown()


# ── Test: --graph + --source-root (adds source tools) ─────────────────────


class TestSourceRootMode:
    """`--source-root <dir>` (no graph) gives just the file-tooling surface
    plus the always-on kglite tools (which respond with 'no active graph')."""

    @pytest.fixture
    def source_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "src"
        d.mkdir()
        (d / "hello.py").write_text(
            "def greet(name):\n    return f'Hello, {name}'\n\ndef shout(name):\n    return greet(name).upper()\n"
        )
        (d / "README.md").write_text("# Sample\n\nA tiny demo.\n")
        sub = d / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested file content\n")
        return d

    def test_lists_source_tools(self, source_dir: Path):
        client = _spawn(["--source-root", str(source_dir)])
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "read_source" in names
            assert "grep" in names
            assert "list_source" in names
        finally:
            client.shutdown()

    def test_read_source(self, source_dir: Path):
        client = _spawn(["--source-root", str(source_dir)])
        try:
            r = client.call_tool("read_source", {"file_path": "hello.py"})
            text = _text_content(r)
            assert "def greet" in text and "def shout" in text
        finally:
            client.shutdown()

    def test_read_source_line_range(self, source_dir: Path):
        client = _spawn(["--source-root", str(source_dir)])
        try:
            r = client.call_tool("read_source", {"file_path": "hello.py", "start_line": 1, "end_line": 2})
            text = _text_content(r)
            assert "def greet" in text
            assert "def shout" not in text
        finally:
            client.shutdown()

    def test_read_source_grep(self, source_dir: Path):
        client = _spawn(["--source-root", str(source_dir)])
        try:
            r = client.call_tool("read_source", {"file_path": "hello.py", "grep": r"def\s+\w+"})
            text = _text_content(r)
            assert "def greet" in text and "def shout" in text
        finally:
            client.shutdown()

    def test_grep_across_files(self, source_dir: Path):
        client = _spawn(["--source-root", str(source_dir)])
        try:
            r = client.call_tool("grep", {"pattern": "Hello"})
            text = _text_content(r)
            assert "hello.py" in text
        finally:
            client.shutdown()

    def test_grep_glob_filter(self, source_dir: Path):
        client = _spawn(["--source-root", str(source_dir)])
        try:
            r = client.call_tool("grep", {"pattern": "demo", "glob": "*.md"})
            text = _text_content(r)
            assert "README.md" in text
            # Non-md files shouldn't be searched.
            assert "hello.py" not in text
        finally:
            client.shutdown()

    def test_list_source(self, source_dir: Path):
        client = _spawn(["--source-root", str(source_dir)])
        try:
            r = client.call_tool("list_source", {"path": ".", "depth": 2})
            text = _text_content(r)
            assert "hello.py" in text
            assert "README.md" in text
            assert "sub" in text
        finally:
            client.shutdown()

    def test_cypher_without_graph_returns_no_graph(self, source_dir: Path):
        """kglite tools register unconditionally; without a graph they
        return the standard 'no active graph' message."""
        client = _spawn(["--source-root", str(source_dir)])
        try:
            r = client.call_tool("cypher_query", {"query": "MATCH (n) RETURN n"})
            text = _text_content(r)
            assert "No active graph" in text
        finally:
            client.shutdown()


# ── Test: GITHUB_TOKEN-gated tools ────────────────────────────────────────


class TestGithubTools:
    """`github_issues` / `github_api` boot-register only when GITHUB_TOKEN is reachable.

    The binary's `.env` walk-up means a sibling `mcp-methods/.env` will leak
    a token even if we clear our own env; we run unauthorized tests in an
    isolated tmp working directory above which no `.env` lives.
    """

    def test_unauthorized_hides_github_tools(self, graph_fixture: Path, tmp_path: Path):
        # Walk-up looks at cwd, source-root, workspace, watch — pick a tmp
        # dir that has none of these "leaking" a token, and unset the env
        # vars entirely (the framework's `auth_token()` accepts "" as set,
        # which is technically a framework quirk but matches the lib).
        isolated_cwd = tmp_path / "no_env_here"
        isolated_cwd.mkdir()
        client = _spawn(
            ["--graph", str(graph_fixture)],
            cwd=isolated_cwd,
            env_remove=["GITHUB_TOKEN", "GH_TOKEN"],
        )
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "github_issues" not in names, (
                "github_issues registered without a token — the .env walk-up "
                "may have found one in an unexpected location."
            )
            assert "github_api" not in names
        finally:
            client.shutdown()

    @pytest.mark.skipif(
        GITHUB_TOKEN is None,
        reason="No GITHUB_TOKEN reachable (env or sibling mcp-methods/.env).",
    )
    def test_authorized_lists_github_tools(self, graph_fixture: Path):
        client = _spawn(
            ["--graph", str(graph_fixture)],
            env_extra={"GITHUB_TOKEN": GITHUB_TOKEN or ""},
        )
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "github_issues" in names
            assert "github_api" in names
        finally:
            client.shutdown()

    @pytest.mark.skipif(
        GITHUB_TOKEN is None,
        reason="No GITHUB_TOKEN reachable.",
    )
    def test_github_api_call(self, graph_fixture: Path):
        """Live GitHub call against a stable public endpoint."""
        client = _spawn(
            ["--graph", str(graph_fixture)],
            env_extra={"GITHUB_TOKEN": GITHUB_TOKEN or ""},
        )
        try:
            # 'octocat' is GitHub's mascot account — stable since forever.
            r = client.call_tool("github_api", {"path": "users/octocat"})
            text = _text_content(r)
            assert "octocat" in text.lower()
        finally:
            client.shutdown()

    @pytest.mark.skipif(
        GITHUB_TOKEN is None,
        reason="No GITHUB_TOKEN reachable.",
    )
    def test_github_issues_search(self, graph_fixture: Path):
        client = _spawn(
            ["--graph", str(graph_fixture)],
            env_extra={"GITHUB_TOKEN": GITHUB_TOKEN or ""},
        )
        try:
            # Search a stable, popular repo for any open issue.
            r = client.call_tool(
                "github_issues",
                {"query": "bug", "repo_name": "rust-lang/rust", "limit": 3},
            )
            text = _text_content(r)
            # Search response should mention the repo or at least produce
            # non-empty output (not the no-token error).
            assert text.strip(), "github_issues returned empty body"
            assert "error" not in text.lower()[:80]
        finally:
            client.shutdown()


# ── Test: .env auto-discovery (walk-up from mode dir + env_file: override) ──


class TestEnvFileLoading:
    """The shim calls `mcp_server::load_env_for_mode` after picking the mode,
    which walks up from the mode dir looking for `.env`. Explicit `env_file:`
    YAML key overrides walk-up. Regression coverage for A3 in the operator
    feedback: pre-fix, the shim never invoked the loader, so .env was never
    found and GitHub tools were silently hidden."""

    def test_walk_up_from_workspace_finds_env(self, tmp_path: Path):
        """Putting `.env` one dir above the workspace should be discovered."""
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / ".env").write_text("GITHUB_TOKEN=ghp_walkup_test_token_not_real\n")
        ws = outer / "workspace"
        ws.mkdir()
        # Use --source-root mode (workspace mode would try to read inventory).
        client = _spawn(
            ["--source-root", str(ws)],
            cwd=tmp_path,  # neutral cwd that has no .env
            env_remove=["GITHUB_TOKEN", "GH_TOKEN"],
        )
        try:
            # If the walk-up worked, github tools register.
            names = {t["name"] for t in client.list_tools()}
            assert "github_issues" in names, (
                "github_issues missing — .env walk-up from --source-root parent "
                "didn't fire. Tools listed: " + str(sorted(names))
            )
        finally:
            client.shutdown()

    def test_explicit_env_file_yaml_key(self, tmp_path: Path):
        """`env_file:` in the manifest overrides walk-up — points at an
        explicit path relative to the manifest's directory."""
        env_dir = tmp_path / "stash"
        env_dir.mkdir()
        (env_dir / "my.env").write_text("GITHUB_TOKEN=ghp_explicit_test_token_not_real\n")
        manifest = tmp_path / "explicit_mcp.yaml"
        manifest.write_text("name: Explicit Env Test\nenv_file: stash/my.env\n")
        client = _spawn(
            ["--mcp-config", str(manifest)],
            cwd=tmp_path,
            env_remove=["GITHUB_TOKEN", "GH_TOKEN"],
        )
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "github_issues" in names, "explicit env_file: didn't load the token. Tools listed: " + str(
                sorted(names)
            )
        finally:
            client.shutdown()

    def test_existing_env_var_not_overwritten(self, tmp_path: Path):
        """If GITHUB_TOKEN is already in the environment, the .env walk-up
        must not overwrite it (matches mcp-methods `apply_env_file` semantics).
        Verified indirectly: pass a real token via env, ensure github tools
        register even if no .env exists at the walk-up location."""
        ws = tmp_path / "workspace_no_env"
        ws.mkdir()
        client = _spawn(
            ["--source-root", str(ws)],
            cwd=ws,
            env_extra={"GITHUB_TOKEN": "ghp_via_env_not_real"},
        )
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "github_issues" in names
        finally:
            client.shutdown()


# ── Test: YAML manifest (parameterised Cypher tools + overview_prefix) ─────


class TestYamlManifest:
    """A `<basename>_mcp.yaml` next to the .kgl auto-extends the tool surface."""

    @pytest.fixture
    def graph_with_manifest(self, tmp_path: Path) -> Path:
        kgl = tmp_path / "demo.kgl"
        _build_fixture_graph(kgl)
        manifest = tmp_path / "demo_mcp.yaml"
        manifest.write_text(
            "name: Demo Smoke Test\n"
            "tools:\n"
            "  - name: people_in_city\n"
            "    description: Find Person nodes whose city matches the parameter.\n"
            "    cypher: |\n"
            "      MATCH (p:Person {city: $city}) RETURN p.title AS name ORDER BY name\n"
            "    parameters:\n"
            "      city:\n"
            "        type: string\n"
            "        description: City name to filter by.\n"
        )
        return kgl

    def test_yaml_tool_registered(self, graph_with_manifest: Path):
        client = _spawn(["--graph", str(graph_with_manifest)])
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "people_in_city" in names
        finally:
            client.shutdown()

    def test_yaml_tool_runs(self, graph_with_manifest: Path):
        client = _spawn(["--graph", str(graph_with_manifest)])
        try:
            r = client.call_tool("people_in_city", {"city": "Oslo"})
            text = _text_content(r)
            assert "Alice" in text and "Carol" in text
            assert "Bob" not in text
            assert "Dave" not in text
        finally:
            client.shutdown()


# ── Test: workspace.kind: local (new in 0.3.22) ───────────────────────────


class TestLocalWorkspace:
    """`workspace.kind: local` registers `set_root_dir` for runtime root swap."""

    @pytest.fixture
    def local_workspace(self, tmp_path: Path) -> tuple[Path, Path]:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "demo.py").write_text("print('hello')\n")
        manifest = tmp_path / "ws_mcp.yaml"
        manifest.write_text(f"name: Local WS Test\nworkspace:\n  kind: local\n  root: {ws}\n")
        return manifest, ws

    def test_set_root_dir_registered(self, local_workspace):
        manifest, _ws = local_workspace
        client = _spawn(["--mcp-config", str(manifest)])
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "set_root_dir" in names
            assert "read_source" in names
        finally:
            client.shutdown()

    def test_read_source_via_workspace(self, local_workspace):
        manifest, _ws = local_workspace
        client = _spawn(["--mcp-config", str(manifest)])
        try:
            r = client.call_tool("read_source", {"file_path": "demo.py"})
            text = _text_content(r)
            assert "hello" in text
        finally:
            client.shutdown()

    def test_discovery_steer_in_instructions(self, local_workspace):
        """Workspace modes fold the lazy-tool-discovery steer into the
        `initialize` instructions by default (no per-manifest copy-paste),
        so code-mode / tool-search clients learn to search the registry for
        `cypher` before falling back to grep."""
        manifest, _ws = local_workspace
        client = _spawn(["--mcp-config", str(manifest)])
        try:
            instructions = client.init_result["result"].get("instructions", "")
            assert "ALWAYS registered" in instructions
            assert "cypher" in instructions
        finally:
            client.shutdown()


# ── Test: read_code_source (qualified_name → file slice) ───────────────────


class TestReadCodeSource:
    """The kglite shim adds `read_code_source(qualified_name)` to bridge
    the code-graph qualified-name → file-slice lookup that the framework's
    file-only `read_source` can't do alone. Reported as A1 in the
    MCP-operator feedback after 0.9.16 dropped the qualified_name surface."""

    @pytest.fixture
    def code_graph_fixture(self, tmp_path: Path) -> tuple[Path, Path]:
        """Build a code-tree graph from a tiny Python module and save it."""
        try:
            from kglite import code_tree
        except ImportError:
            pytest.skip("kglite.code_tree (tree-sitter) not installed")
        project = tmp_path / "demo_proj"
        project.mkdir()
        src = project / "demo_mod.py"
        src.write_text(
            "def greet(name):\n"
            '    """Return a greeting."""\n'
            "    return f'Hello, {name}'\n"
            "\n"
            "\n"
            "def shout(name):\n"
            '    """Greet then upper-case."""\n'
            "    return greet(name).upper()\n"
        )
        g = code_tree.build(str(project))
        # Save the .kgl inside the project dir so the --graph mode's
        # auto-bound source root (parent of the .kgl) lines up with
        # the project root that code_tree's file_path entries are
        # relative to.
        kgl = project / "demo_code.kgl"
        g.save(str(kgl))
        return kgl, project

    def test_lists_read_code_source(self, code_graph_fixture):
        kgl, _ = code_graph_fixture
        client = _spawn(["--graph", str(kgl)])
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "read_code_source" in names
        finally:
            client.shutdown()

    def test_resolves_qualified_name(self, code_graph_fixture):
        kgl, _ = code_graph_fixture
        client = _spawn(["--graph", str(kgl)])
        try:
            r = client.call_tool("read_code_source", {"qualified_name": "demo_proj.demo_mod.greet"})
            text = _text_content(r)
            assert "demo_mod" in text
            assert "greet" in text
            assert "Hello," in text
        finally:
            client.shutdown()

    def test_grep_filter(self, code_graph_fixture):
        kgl, _ = code_graph_fixture
        client = _spawn(["--graph", str(kgl)])
        try:
            r = client.call_tool(
                "read_code_source",
                {
                    "qualified_name": "demo_proj.demo_mod.greet",
                    "grep": r"return",
                },
            )
            text = _text_content(r)
            assert "return" in text
        finally:
            client.shutdown()

    def test_missing_qualified_name_arg(self, code_graph_fixture):
        kgl, _ = code_graph_fixture
        client = _spawn(["--graph", str(kgl)])
        try:
            # Tool returns a friendly error body (success envelope) on
            # missing required arg, OR rmcp rejects at protocol level —
            # either is acceptable as long as the operator sees a clear
            # message about the missing parameter.
            try:
                r = client.call_tool("read_code_source", {})
                text = _text_content(r)
                assert "missing" in text.lower() or "qualified_name" in text
            except RuntimeError as e:
                assert "qualified_name" in str(e)
        finally:
            client.shutdown()


# ── Test: no-graph framework boot (just `ping`) ───────────────────────────


class TestBareBoot:
    """`kglite-mcp-server` with no graph + no source root → still boots,
    kglite tools are registered (always) but report 'no active graph'."""

    def test_boots_with_minimal_manifest(self, tmp_path: Path):
        manifest = tmp_path / "bare_mcp.yaml"
        manifest.write_text("name: Bare Smoke Test\n")
        client = _spawn(["--mcp-config", str(manifest)])
        try:
            names = {t["name"] for t in client.list_tools()}
            # Always-on framework tool
            assert "ping" in names
            # kglite tools register unconditionally (the shim calls
            # tools::register before any mode-specific binding); they
            # respond with "no active graph" until a graph is loaded.
            assert "cypher_query" in names
            assert "graph_overview" in names

            r = client.call_tool("cypher_query", {"query": "MATCH (n) RETURN n"})
            assert "No active graph" in _text_content(r)
        finally:
            client.shutdown()


# ── Test: explore tool + skills (the single Rust server) ──────────────────


class TestExploreAndSkills:
    """`explore` is registered and callable, and bundled skills inject their
    methodology into the matching tool description (mcp-methods serve_prompts).
    These exercise the now-single MCP server (the Rust binary) on a code
    graph; the Python server was retired in 0.10.25."""

    @pytest.fixture
    def code_graph_fixture(self, tmp_path: Path) -> Path:
        try:
            from kglite import code_tree
        except ImportError:
            pytest.skip("kglite.code_tree (tree-sitter) not installed")
        project = tmp_path / "demo_proj"
        project.mkdir()
        (project / "demo_mod.py").write_text(
            "def hub():\n    return leaf()\n\ndef leaf():\n    return 1\n\nclass Bar:\n    pass\n"
        )
        g = code_tree.build(str(project))
        kgl = project / "demo_code.kgl"
        g.save(str(kgl))
        manifest = project / "demo_mcp.yaml"
        manifest.write_text("name: explore_smoke\nskills: true\n")
        return kgl

    def test_explore_registered_and_callable(self, code_graph_fixture):
        client = _spawn(["--graph", str(code_graph_fixture)])
        try:
            assert "explore" in {t["name"] for t in client.list_tools()}
            text = _text_content(client.call_tool("explore", {"query": "hub"}))
        finally:
            client.shutdown()
        assert "Entry points" in text and "hub" in text

    def test_explore_skill_injected_into_tool_description(self, code_graph_fixture):
        """skills: true → the bundled `explore` skill body is injected into the
        explore tool's description (serve_prompts auto-inject)."""
        kgl = code_graph_fixture
        client = _spawn(["--graph", str(kgl), "--mcp-config", str(kgl.parent / "demo_mcp.yaml")])
        try:
            tools = {t["name"]: (t.get("description") or "") for t in client.list_tools()}
        finally:
            client.shutdown()
        assert "## Methodology" in tools["explore"], tools["explore"][:200]

    def test_cross_tool_skills_inject_via_references_tools(self, code_graph_fixture):
        """code_graph_analysis / code_graph_views are cross-tool skills (named
        after no tool). With mcp-methods >=0.3.42 serve_prompts honors
        references_tools and injects the `description` under `## When to use`,
        so they surface on the tools they reference — on a code graph."""
        kgl = code_graph_fixture
        client = _spawn(["--graph", str(kgl), "--mcp-config", str(kgl.parent / "demo_mcp.yaml")])
        try:
            tools = {t["name"]: (t.get("description") or "") for t in client.list_tools()}
        finally:
            client.shutdown()
        # code_graph_analysis references cypher_query / graph_overview / explore.
        for tool in ("cypher_query", "graph_overview", "explore"):
            assert "mcp-skill:code_graph_analysis" in tools[tool], f"{tool} missing orchestration skill"
            assert "## When to use" in tools[tool], f"{tool} missing description routing (0.3.42)"
            assert "Never grep to discover" in tools[tool], f"{tool} missing skill body"
        # code_graph_views references cypher_query.
        assert "mcp-skill:code_graph_views" in tools["cypher_query"]
        assert "is_benchmark" in tools["cypher_query"]


# ── Test: runtime graph-over-grep steering footers (mcp-methods 0.3.46 hook) ──


class TestSteeringFooters:
    """The result-postprocess hook appends a graph-steering footer at the moment
    of a likely misuse (petekSuite skill-steering field report). Fires only when
    a code graph is active."""

    @pytest.fixture
    def code_proj(self, tmp_path: Path) -> Path:
        try:
            from kglite import code_tree
        except ImportError:
            pytest.skip("kglite.code_tree (tree-sitter) not installed")
        project = tmp_path / "steer_proj"
        project.mkdir()
        (project / "m.py").write_text(
            "def hub():\n    return leaf()\n\ndef leaf():\n    return 1\n\nclass Bar:\n    pass\n"
        )
        code_tree.build(str(project)).save(str(project / "code.kgl"))
        return project

    def test_cypher_result_suggests_read_code_source(self, code_proj):
        """A cypher result carrying qualified_name gets a read_code_source tip."""
        client = _spawn(["--graph", str(code_proj / "code.kgl")])
        try:
            text = _text_content(
                client.call_tool(
                    "cypher_query",
                    {"query": "MATCH (f:Function) RETURN f.qualified_name LIMIT 1"},
                )
            )
        finally:
            client.shutdown()
        assert "read_code_source(qualified_name" in text, text

    def test_definition_shaped_grep_steers_to_graph(self, code_proj):
        """`--watch` builds the graph AND sets a source root, so grep runs with a
        code graph active. A `def `-shaped pattern gets the cypher_query tip."""
        client = _spawn(["--watch", str(code_proj)])
        try:
            text = _text_content(client.call_tool("grep", {"pattern": "def "}))
        finally:
            client.shutdown()
        assert "definition search" in text and "cypher_query" in text, text

    def test_zero_match_grep_steers_to_graph(self, code_proj):
        client = _spawn(["--watch", str(code_proj)])
        try:
            text = _text_content(client.call_tool("grep", {"pattern": "zzzz_no_such_symbol"}))
        finally:
            client.shutdown()
        assert "No grep matches" in text or "No matches" in text, text
        assert "graph_overview" in text, text

    def test_literal_grep_not_over_steered(self, code_proj):
        """A plain literal grep that matches should NOT get the definition tip."""
        client = _spawn(["--watch", str(code_proj)])
        try:
            text = _text_content(client.call_tool("grep", {"pattern": "return"}))
        finally:
            client.shutdown()
        assert "definition search" not in text, text


# ── Test: extensions.value_codecs (position-scoped literal codecs) ────────


class TestValueCodecs:
    """`extensions.value_codecs` decode query-side literals bound to a codec'd
    property (`'Q1'` → `1`) and encode result columns back (`1` → `'Q1'`) — the
    safe, post-parse replacement for the retired cypher_preprocessor."""

    def test_prefix_codec_decode_and_encode_round_trip(self, graph_fixture: Path, tmp_path: Path):
        manifest = tmp_path / "vc_mcp.yaml"
        manifest.write_text(
            "name: vc\n"
            "extensions:\n"
            "  value_codecs:\n"
            "    - property: id\n"
            "      kind: prefix\n"
            '      prefix: "Q"\n'
            "      stored_type: int\n"
        )
        client = _spawn(["--graph", str(graph_fixture), "--mcp-config", str(manifest)])
        try:
            # Agent types the Q-form; the codec decodes 'Q1' → 1 to match the
            # integer id, then encodes the projected id 1 → 'Q1' on the way back.
            out = _text_content(
                client.call_tool(
                    "cypher_query",
                    {"query": "MATCH (p:Person {id: 'Q1'}) RETURN p.title AS name, p.id AS id"},
                )
            )
        finally:
            client.shutdown()
        assert "Alice" in out, out  # decode matched node id=1 (Alice)
        assert "Q1" in out, out  # encode round-tripped the projected id

    def test_codec_leaves_other_property_uncoerced(self, graph_fixture: Path, tmp_path: Path):
        """A `Q`-codec on `id` must not touch a `'Q1'` compared against a
        different (string) property — no coercion, no error, just 0 rows."""
        manifest = tmp_path / "vc2_mcp.yaml"
        manifest.write_text(
            "name: vc2\n"
            "extensions:\n"
            "  value_codecs:\n"
            '    - property: id\n      kind: prefix\n      prefix: "Q"\n      stored_type: int\n'
        )
        client = _spawn(["--graph", str(graph_fixture), "--mcp-config", str(manifest)])
        try:
            out = _text_content(
                client.call_tool(
                    "cypher_query",
                    {"query": "MATCH (p:Person) WHERE p.city = 'Q1' RETURN p.title"},
                )
            )
        finally:
            client.shutdown()
        assert "error" not in out.lower()[:40], out
        assert "Alice" not in out, out  # 'Q1' is not a city, so nothing matches

    def test_malformed_codec_refuses_to_boot(self, graph_fixture: Path, tmp_path: Path):
        """A malformed value_codecs block (here: a non-bijective map) is a boot
        error — the server must not start and silently ignore it."""
        manifest = tmp_path / "vc_bad_mcp.yaml"
        manifest.write_text(
            "name: vc_bad\n"
            "extensions:\n"
            "  value_codecs:\n"
            "    - property: status\n      kind: map\n      map: { a: 1, b: 1 }\n"
        )
        proc = subprocess.Popen(
            [str(BINARY), "--graph", str(graph_fixture), "--mcp-config", str(manifest)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("server should have exited (malformed value_codecs) but kept running")
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        assert rc != 0, "server should fail to boot on a malformed value_codecs block"
        assert "value_codecs" in stderr or "bijective" in stderr, stderr[:400]


class TestEmbedderLibrary:
    """`extensions.embedder.library` selection — the engine the user names."""

    def test_python_library_rejected_by_cargo_binary(self, graph_fixture: Path, tmp_path: Path):
        """A Python embedding library (`library: sentence-transformers`) needs a
        Python interpreter to host it. The standalone cargo binary has none, so
        it must refuse to boot with a clear message pointing at the wheel or
        `library: fastembed-rs`."""
        manifest = tmp_path / "py_embed_mcp.yaml"
        manifest.write_text(
            "name: py_embed\nextensions:\n  embedder:\n    library: sentence-transformers\n    model: BAAI/bge-m3\n"
        )
        proc = subprocess.Popen(
            [str(BINARY), "--graph", str(graph_fixture), "--mcp-config", str(manifest)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("server should have exited (Python library on the binary) but kept running")
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        assert rc != 0, "cargo binary should fail to boot on a Python embedder library"
        assert "Python embedding library" in stderr or "fastembed-rs" in stderr, stderr[:400]


# ── Cleanup safety: ensure no orphaned binaries ───────────────────────────


def teardown_module(_module):
    # Best-effort: kill any leaked kglite-mcp-server children. pytest's
    # subprocess management should handle this, but be defensive.
    if shutil.which("pkill"):
        subprocess.run(
            ["pkill", "-f", str(BINARY)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
