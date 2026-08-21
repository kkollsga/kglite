"""End-to-end smoke tests for the kglite-mcp-server Rust binary.

Drives the binary over JSON-RPC stdio (the way Claude Desktop / Cursor
do) and exercises every tool the server exposes — so we catch boot
failures, missing tools, and per-tool argument-shape regressions
before users do.

Tests are skipped when the binary isn't built. Build it with::

    cargo build -p kglite-mcp-server --release

GitHub-token-gated tool registration is exercised with synthetic tokens.
Requests to the live GitHub API additionally require the explicit
``KGLITE_GITHUB_INTEGRATION=1`` opt-in and a reachable ``GITHUB_TOKEN``
(including the sibling ``mcp-methods/.env`` lookup the server performs).
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

# Newest built profile (release or debug); skip with the rebuild command when
# nothing fresh is built. See tests/conftest.py::workspace_binary.
from tests.conftest import binary_skip_reason, workspace_binary

BINARY = workspace_binary("kglite-mcp-server")
_SKIP_REASON = binary_skip_reason("kglite-mcp-server", BINARY, "cargo build -p kglite-mcp-server --release")

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


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
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("GITHUB_TOKEN", "GH_TOKEN"):
                return value.strip().strip("\"'")
    return None


GITHUB_TOKEN = _discover_github_token()


def _github_live_enabled() -> bool:
    """Live GitHub calls require an explicit opt-in as well as a token."""
    return os.environ.get("KGLITE_GITHUB_INTEGRATION") == "1" and GITHUB_TOKEN is not None


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
    manifest.write_text("name: smoke\nbuiltins:\n  save_graph: true\n", encoding="utf-8")
    return manifest


def _build_code_graph_kgl(kgl: Path, entities: list[dict], calls: list[tuple[str, str]] | None = None) -> None:
    """Hand-build a code-schema graph and save it to ``kgl``.

    Mirrors what a code-graph build emits for the code-aware MCP tools
    (``read_code_source`` / ``explore`` / the graph-steering footers):
    Function/Class nodes whose ``id`` is the fully-qualified name and whose
    ``file_path``/``line_number``/``end_line`` point at real source on disk,
    wired by CALLS edges. ``file_path`` values are relative to the ``.kgl``'s
    parent dir — the source root the ``--graph`` mode auto-binds.

    Each entity dict needs ``id`` and ``name``; ``kind`` (default
    ``Function``), ``file_path``, ``line_number``, ``end_line`` and
    ``signature`` are optional.
    """
    g = kglite.KnowledgeGraph()
    for ent in entities:
        props = {
            "id": ent["id"],
            "name": ent["name"],
            # `qualified_name` mirrors the node id — code-graph tools read it
            # back on projections (the steering footer keys off the column).
            "qualified_name": ent["id"],
            "file_path": ent.get("file_path"),
            "line_number": ent.get("line_number"),
            "end_line": ent.get("end_line"),
            "signature": ent.get("signature"),
        }
        keys = [k for k, v in props.items() if v is not None]
        assignments = ", ".join(f"{k}: ${k}" for k in keys)
        kind = ent.get("kind", "Function")
        g.cypher(
            f"CREATE (n:{kind} {{{assignments}}})",
            params={k: props[k] for k in keys},
        )
    for src, dst in calls or []:
        g.cypher(
            "MATCH (a {id: $src}), (b {id: $dst}) CREATE (a)-[:CALLS]->(b)",
            params={"src": src, "dst": dst},
        )
    g.save(str(kgl))


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

    def list_prompts(self) -> list[dict[str, Any]]:
        rid = self._allocate_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": "prompts/list"})
        resp = self._recv(rid)
        if "error" in resp:
            raise RuntimeError(f"prompts/list errored: {resp['error']}")
        return resp["result"]["prompts"]

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


def _validate_github_user_response(result: dict[str, Any]) -> dict[str, Any]:
    """Validate the stable subset of the ``github_api`` user response contract.

    This is deliberately a response-shape validator, not an HTTP mock: the
    network client lives in the external ``mcp-methods`` crate.
    """
    payload = json.loads(_text_content(result))
    assert isinstance(payload.get("login"), str) and payload["login"]
    assert isinstance(payload.get("id"), int)
    assert isinstance(payload.get("html_url"), str) and payload["html_url"].startswith("https://")
    return payload


def _validate_github_issues_response(result: dict[str, Any]) -> str:
    """Validate the text contract exposed by ``github_issues``."""
    text = _text_content(result).strip()
    assert text
    assert "error" not in text.lower()[:80]
    return text


# ── Test: --graph mode (kglite-side tools + ping) ─────────────────────────


@pytest.fixture
def graph_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture.kgl"
    _build_fixture_graph(fixture)
    return fixture


def _build_directed_fixture_graph(path: Path) -> None:
    """Two types wired one way only, so a reversed arrow is expressible.

    The Person/KNOWS fixture above cannot show the eval's §3b defect: with the
    same type at both ends of the relationship, no orientation is wrong.
    """
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2], "title": ["Bergen", "Oslo"]}), "Port", "id", "title")
    g.add_nodes(
        pd.DataFrame({"id": [10, 11], "title": ["V-1", "V-2"], "imo_number": ["9123456", "9234567"]}),
        "Voyage",
        "id",
        "title",
    )
    g.add_connections(pd.DataFrame({"src": [10, 11], "dst": [1, 2]}), "ARRIVES_AT", "Voyage", "src", "Port", "dst")
    g.save(str(path))


@pytest.fixture
def directed_graph_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "directed.kgl"
    _build_directed_fixture_graph(fixture)
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

    def test_cypher_query_typo_reports_the_engine_warning(self, graph_fixture: Path):
        """The engine has always diagnosed `MATCH (p:person)` on a graph of
        `Person`s; before D1p the diagnosis went to the server's stderr and the
        agent read a confident "No results." """
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("cypher_query", {"query": "MATCH (p:person) RETURN count(p) AS n"})
            text = _text_content(r)
            assert "warnings:" in text
            assert "unknown node label 'person'" in text
            assert "Did you mean 'Person'?" in text
        finally:
            client.shutdown()

    def test_cypher_query_binds_the_params_argument(self, graph_fixture: Path):
        """`params` did not exist on the tool, so `$name` was unbindable over
        MCP by construction — while describe()'s own examples teach the inline
        `{prop: $p}` spelling. Both spellings bind from the same argument."""
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            inline = client.call_tool(
                "cypher_query",
                {
                    "query": "MATCH (p:Person {city: $city}) RETURN p.title AS who ORDER BY who",
                    "params": {"city": "Oslo"},
                },
            )
            assert "Alice" in _text_content(inline) and "Carol" in _text_content(inline)
            assert "2 row(s)" in _text_content(inline)

            where = client.call_tool(
                "cypher_query",
                {
                    "query": "MATCH (p:Person) WHERE p.city = $city RETURN p.title AS who ORDER BY who",
                    "params": {"city": "Oslo"},
                },
            )
            assert _text_content(where) == _text_content(inline)
        finally:
            client.shutdown()

    def test_cypher_query_unbound_param_is_an_error_not_an_empty_result(self, graph_fixture: Path):
        """The eval's finding, over the wire: the inline-map spelling used to
        answer "No results." — the agent concluded the city had no people."""
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("cypher_query", {"query": "MATCH (p:Person {city: $city}) RETURN p.title AS who"})
            assert "Missing parameter: $city" in _text_content(r)
            # Twice in a row: the plan cache must not serve the second call a
            # silently-empty answer.
            again = client.call_tool("cypher_query", {"query": "MATCH (p:Person {city: $city}) RETURN p.title AS who"})
            assert "Missing parameter: $city" in _text_content(again)
        finally:
            client.shutdown()

    def test_cypher_query_clean_has_no_warnings_block(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("cypher_query", {"query": "MATCH (p:Person) RETURN count(p) AS n"})
            assert "warnings:" not in _text_content(r)
        finally:
            client.shutdown()

    def test_cypher_query_reversed_arrow_reports_the_direction_warning(self, directed_graph_fixture: Path):
        """The eval's zeros table, over the wire: `(p:Port)-[:ARRIVES_AT]->(v:Voyage)`
        counts 0 with no error, and the agent had nothing to correct against."""
        client = _spawn(["--graph", str(directed_graph_fixture)])
        try:
            r = client.call_tool(
                "cypher_query",
                {"query": "MATCH (p:Port)-[:ARRIVES_AT]->(v:Voyage) RETURN count(*) AS n"},
            )
            text = _text_content(r)
            assert "warnings:" in text
            assert "'ARRIVES_AT'" in text
            assert "Voyage \u2192 Port" in text
            assert "Reverse the arrow?" in text
        finally:
            client.shutdown()

    def test_cypher_query_absent_projection_reports_the_null_column_warning(self, directed_graph_fixture: Path):
        client = _spawn(["--graph", str(directed_graph_fixture)])
        try:
            r = client.call_tool("cypher_query", {"query": "MATCH (v:Voyage) RETURN v.imo, v.imo_numbr"})
            text = _text_content(r)
            assert "warnings:" in text
            assert "RETURN projects property 'imo'" in text
            # `imo` is 7 edits from `imo_number` so the measured suggestion rule
            # stays silent on it; a real typo of the same property gets the hint.
            assert "Did you mean 'imo_number'?" in text
        finally:
            client.shutdown()

    def test_graph_overview_unknown_type_suggests_a_near_miss(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("graph_overview", {"types": ["person"]})
            text = _text_content(r)
            assert "Did you mean 'Person'?" in text
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

    def test_cypher_query_format_csv_is_capped_with_a_notice(self, graph_fixture: Path):
        """`FORMAT CSV` over MCP was the largest response this server could
        produce — an external eval measured ~71k tokens from one call — on a
        tool that recommended it for exactly that case."""
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("cypher_query", {"query": "UNWIND range(1, 640) AS n RETURN n FORMAT CSV"})
            text = _text_content(r)
            body, _, notice = text.partition("\nFORMAT CSV truncated:")
            assert notice, f"no truncation notice: {text[-400:]}"
            # Header + exactly 200 data rows.
            rows = [line for line in body.splitlines() if line]
            assert rows[0] == "n"
            assert len(rows) == 201, f"expected header + 200 rows, got {len(rows)}"
            assert rows[-1] == "200"
            # The notice must name the true total and the escape hatch, or the
            # agent re-runs the same query hoping for more.
            assert "first 200 of 640 row(s)" in notice
            assert "extensions.csv_http_server" in notice
        finally:
            client.shutdown()

    def test_cypher_query_format_csv_under_the_cap_has_no_notice(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            r = client.call_tool("cypher_query", {"query": "UNWIND range(1, 12) AS n RETURN n FORMAT CSV"})
            assert "FORMAT CSV truncated" not in _text_content(r)
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

    def test_graph_overview_truncates_long_sample_values(self, tmp_path: Path):
        """MCP passed `sample_truncate=None` — no truncation at all — while
        Python's `describe()` has clipped at 40 chars since it shipped. The
        surface with the tightest token budget had the weakest cap."""
        long_value = "L" * 300
        fixture = tmp_path / "long.kgl"
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "blob": [long_value, long_value + "z"]}),
            "Doc",
            "id",
            "title",
        )
        g.save(str(fixture))

        client = _spawn(["--graph", str(fixture)])
        try:
            text = _text_content(client.call_tool("graph_overview"))
            assert long_value not in text, "a 300-char property value reached the agent unclipped"
            assert "L" * 37 + "..." in text, f"expected a 40-char clipped sample in:\n{text}"
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

    def test_readonly_server_leaves_the_graph_lockable(self, graph_fixture: Path):
        """A read-only server must not hold the served file's writer lease.

        It never writes that file, and while it held the lease an external
        rebuilder's ``kglite.open(path)`` — which locks by default — was refused
        for the server's whole lifetime. The cross-process claim is the point,
        so this drives the real binary rather than a state object: the boot
        wiring is the part a unit test cannot see.
        """
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            rebuilder = kglite.open(str(graph_fixture))
            try:
                assert list(rebuilder.cypher("MATCH (n) RETURN count(n) AS c"))[0]["c"] == 4
            finally:
                rebuilder.close()
            # The server is unaffected by the external open and keeps serving.
            r = client.call_tool("cypher_query", {"query": "MATCH (n) RETURN count(n) AS c"})
            assert "4" in _text_content(r)
        finally:
            client.shutdown()

    def test_writable_server_keeps_the_graph_locked(self, graph_fixture: Path):
        """The other half of the contract — and the proof the test above is not
        passing because locking stopped working. A `--writable` server owns the
        file it serves, so a second writer is still refused."""
        client = _spawn(["--graph", str(graph_fixture), "--writable"])
        try:
            with pytest.raises(Exception) as caught:
                kglite.open(str(graph_fixture))
            assert "only one process may write" in str(caught.value)
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

    def test_write_tool_binds_params_on_reads_and_mutations(self, graph_fixture: Path):
        """The write path built its own empty parameter map and ignored the
        caller's, so a parameterised mutation was unreachable too."""
        client = _spawn(["--graph", str(graph_fixture), "--writable"])
        try:
            client.call_tool(
                "cypher_query",
                {
                    "query": "CREATE (:Task {id: $id, status: $status})",
                    "params": {"id": "t9", "status": "todo"},
                    "write_scope": ["Task"],
                },
            )
            r = client.call_tool(
                "cypher_query",
                {"query": "MATCH (t:Task {id: $id}) RETURN t.status AS status", "params": {"id": "t9"}},
            )
            assert "todo" in _text_content(r)
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
            "def greet(name):\n    return f'Hello, {name}'\n\ndef shout(name):\n    return greet(name).upper()\n",
            encoding="utf-8",
        )
        (d / "README.md").write_text("# Sample\n\nA tiny demo.\n", encoding="utf-8")
        sub = d / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested file content\n", encoding="utf-8")
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
    """`github_issues` / `github_api` boot-register behind *two* gates.

    Since mcp-methods 0.4.5: the manifest must opt in with
    `builtins.github: true`, and only then does a reachable GITHUB_TOKEN
    decide whether the tools are listed. Every test here supplies the opt-in
    manifest so it isolates the token gate — without it they would all pass
    on the manifest gate alone and prove nothing about tokens.

    The binary's `.env` walk-up means a sibling `mcp-methods/.env` will leak
    a token even if we clear our own env; we run unauthorized tests in an
    isolated tmp working directory above which no `.env` lives.
    """

    @staticmethod
    def _optin_manifest(tmp_path: Path) -> Path:
        """A manifest whose only job is `builtins.github: true`."""
        manifest = tmp_path / "github_optin_mcp.yaml"
        manifest.write_text("name: github test\nbuiltins:\n  github: true\n", encoding="utf-8")
        return manifest

    def test_unauthorized_hides_github_tools(self, graph_fixture: Path, tmp_path: Path):
        # Walk-up looks at cwd, source-root, workspace, watch — pick a tmp
        # dir that has none of these "leaking" a token, and unset the env
        # vars entirely (the framework's `auth_token()` accepts "" as set,
        # which is technically a framework quirk but matches the lib).
        isolated_cwd = tmp_path / "no_env_here"
        isolated_cwd.mkdir()
        client = _spawn(
            # Opted in, so the only thing left to hide the tools is the
            # missing token — which is what this test is about.
            ["--graph", str(graph_fixture), "--mcp-config", str(self._optin_manifest(tmp_path))],
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
    def test_authorized_lists_github_tools(self, graph_fixture: Path, tmp_path: Path):
        client = _spawn(
            ["--graph", str(graph_fixture), "--mcp-config", str(self._optin_manifest(tmp_path))],
            env_extra={"GITHUB_TOKEN": GITHUB_TOKEN or ""},
        )
        try:
            names = {t["name"] for t in client.list_tools()}
            assert "github_issues" in names
            assert "github_api" in names
        finally:
            client.shutdown()

    @pytest.mark.skipif(
        not _github_live_enabled(),
        reason="set KGLITE_GITHUB_INTEGRATION=1 and provide GITHUB_TOKEN to run live GitHub calls",
    )
    def test_github_api_call(self, graph_fixture: Path, tmp_path: Path):
        """Live GitHub call against a stable public endpoint."""
        client = _spawn(
            ["--graph", str(graph_fixture), "--mcp-config", str(self._optin_manifest(tmp_path))],
            env_extra={"GITHUB_TOKEN": GITHUB_TOKEN or ""},
        )
        try:
            # 'octocat' is GitHub's mascot account — stable since forever.
            r = client.call_tool("github_api", {"path": "users/octocat"})
            payload = _validate_github_user_response(r)
            assert payload["login"].lower() == "octocat"
        finally:
            client.shutdown()

    @pytest.mark.skipif(
        not _github_live_enabled(),
        reason="set KGLITE_GITHUB_INTEGRATION=1 and provide GITHUB_TOKEN to run live GitHub calls",
    )
    def test_github_issues_search(self, graph_fixture: Path, tmp_path: Path):
        client = _spawn(
            ["--graph", str(graph_fixture), "--mcp-config", str(self._optin_manifest(tmp_path))],
            env_extra={"GITHUB_TOKEN": GITHUB_TOKEN or ""},
        )
        try:
            # Search a stable, popular repo for any open issue.
            r = client.call_tool(
                "github_issues",
                {"query": "bug", "repo_name": "rust-lang/rust", "limit": 3},
            )
            text = _validate_github_issues_response(r)
            # Search response should mention the repo or at least produce
            # non-empty output (not the no-token error).
            assert text
        finally:
            client.shutdown()


# ── Test: extensions.tools_allow (closed-by-default tool surface) ─────────


class TestToolsAllowlist:
    """`extensions.tools_allow:` pins the whole agent-visible tool surface.

    The deployment shape is a domain embedder serving one graph and wanting
    three tools. Without the key its surface is the union of everything that
    registered — but as of mcp-methods 0.4.5 an *ambient credential* is no
    longer part of that union: `github_api` / `github_issues` /
    `screen_stargazers` register only when the manifest opts in with
    `builtins.github: true`, so a `GITHUB_TOKEN` exported for unrelated
    reasons can no longer widen a music server's surface on its own.

    Two independent things are asserted below, because either alone would be
    weak. Default-off: the token adds nothing to a surface that never opted
    in. Opt-in works: a manifest that *does* say `builtins.github: true`,
    with the same token, registers all three — which is what stops the
    default-off assertion from passing merely because GitHub registration
    broke entirely. The allowlist arm is now belt-and-braces on top.
    """

    ALLOWED = {"cypher_query", "graph_overview", "ping"}
    GITHUB_TOOLS = {"github_api", "github_issues", "screen_stargazers"}

    @pytest.fixture
    def deployment(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        """An isolated graph + three manifests: allowlisted, open, GitHub opt-in."""
        root = tmp_path / "domain_server"
        root.mkdir()
        kgl = root / "domain.kgl"
        _build_fixture_graph(kgl)
        allowed = root / "allowlist_mcp.yaml"
        allowed.write_text(
            "name: domain\nextensions:\n  tools_allow:\n    - cypher_query\n    - graph_overview\n    - ping\n",
            encoding="utf-8",
        )
        open_surface = root / "open_mcp.yaml"
        open_surface.write_text("name: domain\n", encoding="utf-8")
        github_optin = root / "github_optin_mcp.yaml"
        github_optin.write_text("name: domain\nbuiltins:\n  github: true\n", encoding="utf-8")
        return kgl, allowed, open_surface, github_optin

    @staticmethod
    def _tool_names(kgl: Path, manifest: Path, token: Optional[str] = None) -> set[str]:
        # `env_remove` runs before `env_extra` in `_spawn`, so the tokenless arm
        # is genuinely tokenless and the token arm carries exactly one token.
        client = _spawn(
            ["--graph", str(kgl), "--mcp-config", str(manifest)],
            cwd=kgl.parent,
            env_extra={"GITHUB_TOKEN": token} if token else None,
            env_remove=["GITHUB_TOKEN", "GH_TOKEN"],
        )
        try:
            return {t["name"] for t in client.list_tools()}
        finally:
            client.shutdown()

    def test_allowlist_is_the_entire_tool_surface(self, deployment: tuple[Path, Path, Path, Path]):
        kgl, allowed, open_surface, _ = deployment
        # Control: the same deployment without the key serves far more than
        # three tools, so the assertion below cannot pass vacuously.
        open_names = self._tool_names(kgl, open_surface)
        assert self.ALLOWED < open_names
        assert {"grep", "read_source", "list_source"} <= open_names

        assert self._tool_names(kgl, allowed) == self.ALLOWED

    def test_a_dropped_tool_is_rejected_on_call_not_merely_unlisted(self, deployment: tuple[Path, Path, Path, Path]):
        kgl, allowed, _, _ = deployment
        client = _spawn(
            ["--graph", str(kgl), "--mcp-config", str(allowed)],
            cwd=kgl.parent,
            env_remove=["GITHUB_TOKEN", "GH_TOKEN"],
        )
        try:
            assert "pong" in _text_content(client.call_tool("ping")).lower()
            with pytest.raises(RuntimeError, match="tool not found"):
                client.call_tool("grep", {"pattern": "anything"})
        finally:
            client.shutdown()

    def test_an_ambient_github_token_widens_no_surface(self, deployment: tuple[Path, Path, Path, Path]):
        """mcp-methods 0.4.5: a reachable token is not an intent to register.

        Pre-0.4.5 this same deployment gained `github_api` / `github_issues` /
        `screen_stargazers` from the token alone, with no manifest saying so.
        Now the surface is identical either way — and the opt-in test below is
        what proves that identity is default-off rather than GitHub
        registration being broken outright.
        """
        kgl, allowed, open_surface, _ = deployment

        # No allowlist at all: the token must not add a thing.
        open_tokenless = self._tool_names(kgl, open_surface)
        open_with_token = self._tool_names(kgl, open_surface, token="dummy-value")
        assert open_with_token == open_tokenless, (
            "an ambient GITHUB_TOKEN changed a non-opted-in surface "
            f"(added: {sorted(open_with_token - open_tokenless)}, "
            f"removed: {sorted(open_tokenless - open_with_token)})"
        )
        assert not (self.GITHUB_TOOLS & open_with_token), (
            "GitHub tools registered without `builtins.github: true` in the manifest: "
            f"{sorted(self.GITHUB_TOOLS & open_with_token)}"
        )

        # Belt-and-braces: the allowlist independently pins the surface, so it
        # holds even if the framework default ever regresses back to token-keyed.
        assert self._tool_names(kgl, allowed, token="dummy-value") == self._tool_names(kgl, allowed)
        assert self._tool_names(kgl, allowed, token="dummy-value") == self.ALLOWED

    def test_builtins_github_opt_in_registers_the_github_tools(self, deployment: tuple[Path, Path, Path, Path]):
        """The other half of default-off: opting in must actually work.

        Without this, `test_an_ambient_github_token_widens_no_surface` would
        pass just as happily if GitHub registration were removed entirely.
        Both of 0.4.5's boot-time gates are exercised here: the manifest
        opt-in, and token reachability (registration keys off token
        *presence* — `has_git_token()` is an env/.env lookup, no live
        validation — so a dummy value is enough).
        """
        kgl, _, _, github_optin = deployment

        optin_with_token = self._tool_names(kgl, github_optin, token="dummy-value")
        assert self.GITHUB_TOOLS <= optin_with_token, (
            "`builtins.github: true` + a reachable token did not register the GitHub tools "
            f"(missing: {sorted(self.GITHUB_TOOLS - optin_with_token)})"
        )

        # Gate 2 in isolation: opted in, but no token reachable → still hidden,
        # so the agent never sees a tool that cannot succeed.
        optin_tokenless = self._tool_names(kgl, github_optin)
        assert not (self.GITHUB_TOOLS & optin_tokenless), (
            "GitHub tools registered with `builtins.github: true` but no token reachable: "
            f"{sorted(self.GITHUB_TOOLS & optin_tokenless)}"
        )


# ── Test: .env auto-discovery (walk-up from mode dir + env_file: override) ──


class TestEnvFileLoading:
    """The shim calls `mcp_server::load_env_for_mode` after picking the mode,
    which walks up from the mode dir looking for `.env`. Explicit `env_file:`
    YAML key overrides walk-up. Regression coverage for A3 in the operator
    feedback: pre-fix, the shim never invoked the loader, so .env was never
    found and GitHub tools were silently hidden.

    The observable for "the token was loaded" is still GitHub-tool
    registration, but since mcp-methods 0.4.5 that also requires the manifest
    to opt in with `builtins.github: true` — so each case below pairs its
    `.env` arrangement with an opt-in manifest. `--mcp-config` composes with
    `--source-root` and does not move the walk-up anchor, which
    `resolve_env_start_dir` derives from the mode alone."""

    GITHUB_OPT_IN = "name: env test\nbuiltins:\n  github: true\n"

    def test_walk_up_from_workspace_finds_env(self, tmp_path: Path):
        """Putting `.env` one dir above the workspace should be discovered."""
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / ".env").write_text("GITHUB_TOKEN=ghp_walkup_test_token_not_real\n", encoding="utf-8")
        ws = outer / "workspace"
        ws.mkdir()
        # No `env_file:` key here — that would override the walk-up under test.
        manifest = tmp_path / "optin_mcp.yaml"
        manifest.write_text(self.GITHUB_OPT_IN, encoding="utf-8")
        # Use --source-root mode (workspace mode would try to read inventory).
        client = _spawn(
            ["--source-root", str(ws), "--mcp-config", str(manifest)],
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
        (env_dir / "my.env").write_text("GITHUB_TOKEN=ghp_explicit_test_token_not_real\n", encoding="utf-8")
        manifest = tmp_path / "explicit_mcp.yaml"
        manifest.write_text(
            "name: Explicit Env Test\nenv_file: stash/my.env\nbuiltins:\n  github: true\n",
            encoding="utf-8",
        )
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
        manifest = tmp_path / "optin_mcp.yaml"
        manifest.write_text(self.GITHUB_OPT_IN, encoding="utf-8")
        client = _spawn(
            ["--source-root", str(ws), "--mcp-config", str(manifest)],
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
            "skills: true\n"
            "overview_prefix: |\n"
            "  smoke overview prefix\n"
            "builtins:\n"
            "  temp_cleanup: on_overview\n"
            "extensions:\n"
            "  cypher_recipes:\n"
            "    smoke:\n"
            "      description: Smoke-test recipe operations.\n"
            "      queries:\n"
            "        count_people:\n"
            "          description: Count Person nodes.\n"
            "          parameters:\n"
            "            type: object\n"
            "            properties: {}\n"
            "            required: []\n"
            "            additionalProperties: false\n"
            "          cypher: MATCH (p:Person) RETURN count(p) AS people ORDER BY people\n"
            "tools:\n"
            "  - name: people_in_city\n"
            "    description: Find Person nodes whose city matches the parameter.\n"
            "    cypher: |\n"
            "      MATCH (p:Person {city: $city}) RETURN p.title AS name ORDER BY name\n"
            "    parameters:\n"
            "      type: object\n"
            "      properties:\n"
            "        city:\n"
            "          type: string\n"
            "          description: City name to filter by.\n"
            "      required: [city]\n"
            "      additionalProperties: false\n",
            encoding="utf-8",
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

    def test_recipe_tools_list_and_run_with_identical_structured_fallback(self, graph_with_manifest: Path):
        client = _spawn(["--graph", str(graph_with_manifest)])
        try:
            listing = client.call_tool("list_recipe_queries", {"recipe": "smoke"})
            result = client.call_tool(
                "run_recipe_query",
                {
                    "recipe": "smoke",
                    "query": "count_people",
                    "variables": {},
                },
            )
        finally:
            client.shutdown()

        for envelope in (listing, result):
            assert envelope["structuredContent"] == json.loads(_text_content(envelope))
        assert listing["structuredContent"]["recipes"][0]["query_count"] == 1
        assert result["structuredContent"]["result"] == {
            "columns": ["people"],
            "rows": [[4]],
            "row_count": 1,
        }

    def test_bare_overview_adds_prefix_then_catalog_hint(self, graph_with_manifest: Path):
        client = _spawn(["--graph", str(graph_with_manifest)])
        try:
            text = _text_content(client.call_tool("graph_overview"))
        finally:
            client.shutdown()

        prefix = text.index("smoke overview prefix")
        active_graph = text.index("<active_graph")
        catalog = text.index(
            '<query-catalog recipes="1" queries="1" list-tool="list_recipe_queries" run-tool="run_recipe_query"/>'
        )
        assert prefix < active_graph < catalog

    def test_focused_overview_has_no_prefix_hint_or_cleanup(self, graph_with_manifest: Path):
        temp_dir = graph_with_manifest.parent / "temp"
        temp_dir.mkdir()
        marker = temp_dir / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        client = _spawn(["--graph", str(graph_with_manifest)])
        try:
            focused = _text_content(client.call_tool("graph_overview", {"types": []}))
            assert marker.exists(), "focused overview must not run temp cleanup"
            bare = _text_content(client.call_tool("graph_overview"))
        finally:
            client.shutdown()

        assert "smoke overview prefix" not in focused
        assert "<query-catalog" not in focused
        assert not marker.exists(), "bare overview must run temp cleanup"
        assert "smoke overview prefix" in bare
        assert "<query-catalog" in bare

    def test_no_active_graph_still_exposes_bare_discovery(self, graph_with_manifest: Path):
        manifest = graph_with_manifest.with_name("demo_mcp.yaml")
        client = _spawn(["--mcp-config", str(manifest)])
        try:
            text = _text_content(client.call_tool("graph_overview"))
        finally:
            client.shutdown()

        assert text.startswith("smoke overview prefix\nNo active graph.")
        assert text.endswith(
            '<query-catalog recipes="1" queries="1" list-tool="list_recipe_queries" run-tool="run_recipe_query"/>'
        )

    def test_present_catalog_exposes_recipe_prompt_and_injects_referenced_tools(self, graph_with_manifest: Path):
        client = _spawn(["--graph", str(graph_with_manifest)])
        try:
            prompts = {prompt["name"] for prompt in client.list_prompts()}
            tools = {tool["name"]: (tool.get("description") or "") for tool in client.list_tools()}
        finally:
            client.shutdown()

        assert "recipe_queries" in prompts
        for name in ("list_recipe_queries", "run_recipe_query", "cypher_query"):
            assert "mcp-skill:recipe_queries" in tools[name], name
        injected = " ".join(tools["run_recipe_query"].split())
        assert "fall back to raw `cypher_query`" in injected

    def test_absent_and_empty_catalogs_expose_no_recipe_skill(self, graph_with_manifest: Path):
        for label, catalog_yaml in (
            ("absent", ""),
            ("empty", "extensions:\n  cypher_recipes: {}\n"),
        ):
            manifest = graph_with_manifest.parent / f"{label}_mcp.yaml"
            manifest.write_text(f"name: {label}\nskills: true\n{catalog_yaml}", encoding="utf-8")
            client = _spawn(["--graph", str(graph_with_manifest), "--mcp-config", str(manifest)])
            try:
                prompts = {prompt["name"] for prompt in client.list_prompts()}
                tools = {tool["name"]: (tool.get("description") or "") for tool in client.list_tools()}
            finally:
                client.shutdown()

            assert "recipe_queries" not in prompts
            assert "list_recipe_queries" not in tools
            assert "run_recipe_query" not in tools
            assert "mcp-skill:recipe_queries" not in tools["cypher_query"]


# ── Test: workspace.kind: local (new in 0.3.22) ───────────────────────────


class TestLocalWorkspace:
    """`workspace.kind: local` registers `set_root_dir` for runtime root swap."""

    @pytest.fixture
    def local_workspace(self, tmp_path: Path) -> tuple[Path, Path]:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "demo.py").write_text("print('hello')\n", encoding="utf-8")
        manifest = tmp_path / "ws_mcp.yaml"
        manifest.write_text(f"name: Local WS Test\nworkspace:\n  kind: local\n  root: {ws}\n", encoding="utf-8")
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
        """Hand-build a code-schema graph over a tiny Python module and save it."""
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
            "    return greet(name).upper()\n",
            encoding="utf-8",
        )
        # Save the .kgl inside the project dir so the --graph mode's
        # auto-bound source root (parent of the .kgl) lines up with the
        # project root the Function file_path entries are relative to.
        # line_number/end_line point at the real `greet`/`shout` bodies so
        # read_code_source slices the correct lines off disk.
        kgl = project / "demo_code.kgl"
        _build_code_graph_kgl(
            kgl,
            [
                {
                    "id": "demo_proj.demo_mod.greet",
                    "name": "greet",
                    "file_path": "demo_mod.py",
                    "line_number": 1,
                    "end_line": 3,
                    "signature": "def greet(name)",
                },
                {
                    "id": "demo_proj.demo_mod.shout",
                    "name": "shout",
                    "file_path": "demo_mod.py",
                    "line_number": 6,
                    "end_line": 8,
                    "signature": "def shout(name)",
                },
            ],
            calls=[("demo_proj.demo_mod.shout", "demo_proj.demo_mod.greet")],
        )
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
        manifest.write_text("name: Bare Smoke Test\n", encoding="utf-8")
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
        project = tmp_path / "demo_proj"
        project.mkdir()
        (project / "demo_mod.py").write_text(
            "def hub():\n    return leaf()\n\ndef leaf():\n    return 1\n\nclass Bar:\n    pass\n", encoding="utf-8"
        )
        kgl = project / "demo_code.kgl"
        _build_code_graph_kgl(
            kgl,
            [
                {
                    "id": "demo_mod.hub",
                    "name": "hub",
                    "file_path": "demo_mod.py",
                    "line_number": 1,
                    "end_line": 2,
                    "signature": "def hub()",
                },
                {
                    "id": "demo_mod.leaf",
                    "name": "leaf",
                    "file_path": "demo_mod.py",
                    "line_number": 4,
                    "end_line": 5,
                    "signature": "def leaf()",
                },
                {
                    "id": "demo_mod.Bar",
                    "name": "Bar",
                    "kind": "Class",
                    "file_path": "demo_mod.py",
                    "line_number": 7,
                    "end_line": 8,
                },
            ],
            calls=[("demo_mod.hub", "demo_mod.leaf")],
        )
        manifest = project / "demo_mcp.yaml"
        manifest.write_text("name: explore_smoke\nskills: true\n", encoding="utf-8")
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


# ── Test: code-tool gating on non-code graphs ─────────────────────────────


CODE_TOOLS = ("explore", "read_code_source")


class TestCodeToolGating:
    """`explore` and `read_code_source` are code-graph tools: `explore` pins its
    entry types and traversal edges to code node types (so it is structurally
    dead on a graph without them), and `read_code_source(node_type=...)` would
    otherwise read arbitrary `file_path` properties off disk. A read-only
    `--graph` server whose graph has no Function/Class nodes therefore hides
    both. The gate is a *boot-time* decision, scoped to `--graph` mode and
    exempting `--writable` (see `code_tools_are_dead` in server_run.rs)."""

    @pytest.fixture
    def code_graph(self, tmp_path: Path) -> Path:
        """A minimal Function/Class graph — the negative control for the gate."""
        project = tmp_path / "gated_proj"
        project.mkdir()
        (project / "m.py").write_text("def hub():\n    return 1\n\nclass Bar:\n    pass\n", encoding="utf-8")
        kgl = project / "code.kgl"
        _build_code_graph_kgl(
            kgl,
            [
                {"id": "m.hub", "name": "hub", "file_path": "m.py", "line_number": 1, "end_line": 2},
                {"id": "m.Bar", "name": "Bar", "kind": "Class", "file_path": "m.py", "line_number": 4, "end_line": 5},
            ],
        )
        return kgl

    def _names(self, args: list[str]) -> set[str]:
        client = _spawn(args)
        try:
            return {t["name"] for t in client.list_tools()}
        finally:
            client.shutdown()

    def test_non_code_readonly_graph_hides_both_code_tools(self, graph_fixture: Path):
        """The Person/KNOWS fixture has no Function/Class — both tools go."""
        names = self._names(["--graph", str(graph_fixture)])
        for tool in CODE_TOOLS:
            assert tool not in names, f"{tool} must be hidden on a non-code read-only graph"
        # The gate must be surgical: everything else still registers.
        assert {"cypher_query", "graph_overview", "read_source", "grep"} <= names

    def test_code_graph_keeps_both_code_tools(self, code_graph: Path):
        """Negative control — the same read-only mode over a code graph."""
        names = self._names(["--graph", str(code_graph)])
        for tool in CODE_TOOLS:
            assert tool in names, f"{tool} must stay on a code graph"

    def test_writable_non_code_graph_keeps_both_code_tools(self, graph_fixture: Path):
        """Writable servers are exempt: `load_graph` can swap a code graph in at
        runtime and there is no router access from a tool handler to re-enable a
        disabled route afterwards."""
        names = self._names(["--graph", str(graph_fixture), "--writable"])
        for tool in CODE_TOOLS:
            assert tool in names, f"{tool} must stay on a writable server"

    def test_local_workspace_keeps_both_code_tools(self, tmp_path: Path):
        """Over-broad-gate canary. Local-workspace mode has no graph at boot, so
        a mode-blind gate would strip `explore` from exactly the deployments
        that need it (the code-review / open-source servers)."""
        root = tmp_path / "ws-root"
        root.mkdir()
        (root / "demo.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        manifest = tmp_path / "ws_mcp.yaml"
        manifest.write_text(f"name: Gating Canary\nworkspace:\n  kind: local\n  root: {root}\n", encoding="utf-8")
        names = self._names(["--mcp-config", str(manifest)])
        for tool in CODE_TOOLS:
            assert tool in names, f"{tool} must stay in local-workspace mode"

    def test_manifest_may_override_a_gated_route(self, graph_fixture: Path, tmp_path: Path):
        """The gate disables the routes instead of skipping registration, so a
        manifest `bundled:` override naming one still resolves at boot. Removing
        the route instead would make this manifest a hard boot failure
        ("override targets unknown route"). The Rust unit test in
        bundled_overrides.rs proves the *mechanism* on a stub router; this
        proves the real boot ordering — the gate runs before
        `apply_bundled_tool_overrides`, with the real route names."""
        manifest = tmp_path / "override_mcp.yaml"
        manifest.write_text("name: Gated Override\ntools:\n  - bundled: explore\n    hidden: true\n", encoding="utf-8")
        names = self._names(["--graph", str(graph_fixture), "--mcp-config", str(manifest)])
        # Boot survived (a dead server raises in _spawn), and the tool stays hidden.
        assert "explore" not in names
        assert "cypher_query" in names


# ── Test: reload_graph (the --graph refresh affordance) ───────────────────


class TestReloadGraph:
    """`reload_graph` re-reads the served `.kgl` from disk, replacing the
    in-memory graph. It is the refresh affordance for a file rebuilt by an
    external process, and is registered only in `--graph` mode (no other mode
    has a single source file to re-read)."""

    def test_reload_picks_up_an_external_rewrite(self, graph_fixture: Path):
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            assert "reload_graph" in {t["name"] for t in client.list_tools()}
            counted = _text_content(
                client.call_tool("cypher_query", {"query": "MATCH (p:Person) RETURN count(p) AS n"})
            )
            assert "4" in counted, counted

            # An external producer rebuilds the file (kglite's save is a
            # rename-over, so this is exactly the atomic-republish shape).
            g = kglite.KnowledgeGraph()
            g.add_nodes(pd.DataFrame({"id": [9], "title": ["Zoe"], "city": ["Tromso"]}), "Person", "id", "title")
            g.save(str(graph_fixture))

            reloaded = _text_content(client.call_tool("reload_graph"))
            assert "1 nodes" in reloaded and "0 edges" in reloaded, reloaded
            assert str(graph_fixture) in reloaded, reloaded
            # The generation bump is what distinguishes a completed re-read
            # from a reply about the graph the server already had.
            assert "Graph generation 2." in reloaded, reloaded

            after = _text_content(client.call_tool("cypher_query", {"query": "MATCH (p:Person) RETURN p.title AS t"}))
            assert "Zoe" in after and "Alice" not in after, after
        finally:
            client.shutdown()

    def test_absent_outside_graph_mode(self, tmp_path: Path):
        """No other mode serves a single source file, so the tool would have
        nothing to re-read — it must not be advertised there."""
        manifest = tmp_path / "bare_reload_mcp.yaml"
        manifest.write_text("name: Bare Reload\n", encoding="utf-8")
        client = _spawn(["--mcp-config", str(manifest)])
        try:
            assert "reload_graph" not in {t["name"] for t in client.list_tools()}
        finally:
            client.shutdown()


class TestGraphWatch:
    """`extensions.graph_watch: true` arms a filesystem watch on the served
    `.kgl`: an external rewrite marks the graph, and the *next* graph tool call
    serves the new bytes without anyone calling `reload_graph`. Opt-in — the
    control test below pins that a server without the key never moves."""

    @staticmethod
    def _rewrite(kgl: Path) -> None:
        """Republish the served file with different contents. kglite's save
        writes a sibling temp file and renames it over, so this is exactly the
        atomic-republish event shape a real producer emits (and the sibling
        churn the watch callback has to filter out)."""
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [9], "title": ["Zoe"], "city": ["Tromso"]}), "Person", "id", "title")
        g.save(str(kgl))

    @staticmethod
    def _titles(client: McpClient) -> str:
        return _text_content(client.call_tool("cypher_query", {"query": "MATCH (p:Person) RETURN p.title AS t"}))

    def test_manifest_opt_in_reloads_after_an_external_rewrite(self, graph_fixture: Path, tmp_path: Path):
        # `--graph X.kgl` auto-finds `X_mcp.yaml` beside it, so the manifest
        # needs no CLI flag — the deployment shape an operator actually uses.
        (tmp_path / "fixture_mcp.yaml").write_text(
            "name: Graph Watch\nextensions:\n  graph_watch: true\n", encoding="utf-8"
        )
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            before = self._titles(client)
            assert "Alice" in before and "Zoe" not in before, before

            self._rewrite(graph_fixture)

            # The watch debounces (500 ms) and the reload is lazy, so poll the
            # tool instead of guessing one sleep long enough to be non-flaky.
            deadline = time.monotonic() + 10.0
            after = ""
            while time.monotonic() < deadline:
                after = self._titles(client)
                if "Zoe" in after:
                    break
                time.sleep(0.25)
            assert "Zoe" in after and "Alice" not in after, after
        finally:
            client.shutdown()

    def test_without_the_manifest_key_a_rewrite_is_not_picked_up(self, graph_fixture: Path):
        """The control arm: the same rewrite against a server with no
        `graph_watch` key must leave the served graph alone (`reload_graph` is
        the only refresh there). A fixed short wait is right here — the
        assertion is an absence, and 2 s is well past the 500 ms debounce."""
        client = _spawn(["--graph", str(graph_fixture)])
        try:
            assert "Alice" in self._titles(client)
            self._rewrite(graph_fixture)
            time.sleep(2.0)
            after = self._titles(client)
            assert "Alice" in after and "Zoe" not in after, after
        finally:
            client.shutdown()


# ── Test: runtime graph-over-grep steering footers (mcp-methods 0.3.46 hook) ──


class TestSteeringFooters:
    """The result-postprocess hook appends a graph-steering footer at the moment
    of a likely misuse (petekSuite skill-steering field report). Fires only when
    a code graph is active."""

    @pytest.fixture
    def code_kgl(self, tmp_path: Path) -> Path:
        """A hand-built code-schema graph saved next to its source file. The
        ``--graph`` mode auto-binds the source root to the .kgl's parent dir,
        so grep runs against ``m.py`` with the code graph active — the same
        "code graph active + source root bound" condition the steering hook
        fires on."""
        project = tmp_path / "steer_proj"
        project.mkdir()
        (project / "m.py").write_text(
            "def hub():\n    return leaf()\n\ndef leaf():\n    return 1\n\nclass Bar:\n    pass\n", encoding="utf-8"
        )
        kgl = project / "code.kgl"
        _build_code_graph_kgl(
            kgl,
            [
                {"id": "m.hub", "name": "hub", "file_path": "m.py", "line_number": 1, "end_line": 2},
                {"id": "m.leaf", "name": "leaf", "file_path": "m.py", "line_number": 4, "end_line": 5},
                {"id": "m.Bar", "name": "Bar", "kind": "Class", "file_path": "m.py", "line_number": 7, "end_line": 8},
            ],
            calls=[("m.hub", "m.leaf")],
        )
        return kgl

    def test_cypher_result_suggests_read_code_source(self, code_kgl):
        """A cypher result carrying qualified_name gets a read_code_source tip."""
        client = _spawn(["--graph", str(code_kgl)])
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

    def test_definition_shaped_grep_steers_to_graph(self, code_kgl):
        """`--graph` loads the code graph AND auto-binds the .kgl's parent as a
        source root, so grep runs with a code graph active. A `def `-shaped
        pattern gets the cypher_query tip."""
        client = _spawn(["--graph", str(code_kgl)])
        try:
            text = _text_content(client.call_tool("grep", {"pattern": "def "}))
        finally:
            client.shutdown()
        assert "definition search" in text and "cypher_query" in text, text

    def test_zero_match_grep_steers_to_graph(self, code_kgl):
        client = _spawn(["--graph", str(code_kgl)])
        try:
            text = _text_content(client.call_tool("grep", {"pattern": "zzzz_no_such_symbol"}))
        finally:
            client.shutdown()
        assert "No grep matches" in text or "No matches" in text, text
        assert "graph_overview" in text, text

    def test_literal_grep_not_over_steered(self, code_kgl):
        """A plain literal grep that matches should NOT get the definition tip."""
        client = _spawn(["--graph", str(code_kgl)])
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
            "      stored_type: int\n",
            encoding="utf-8",
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
            '    - property: id\n      kind: prefix\n      prefix: "Q"\n      stored_type: int\n',
            encoding="utf-8",
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
            "    - property: status\n      kind: map\n      map: { a: 1, b: 1 }\n",
            encoding="utf-8",
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
            "name: py_embed\ntrust:\n  allow_embedder: true\nextensions:\n"
            "  embedder:\n    library: sentence-transformers\n    model: BAAI/bge-m3\n",
            encoding="utf-8",
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


# ── Test: --selftest handshake harness ─────────────────────────────────────


def _run_selftest(args: list[str], timeout_s: float = 60.0) -> tuple[int, str]:
    """Run `kglite-mcp-server --selftest <args>` to completion; return
    (exit_code, combined_stdout+stderr)."""
    proc = subprocess.run(
        [str(BINARY), "--selftest", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    out = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    return proc.returncode, out


def _write_local_workspace_manifest(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny local workspace and its MCP manifest."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "demo.py").write_text("def greet(name):\n    return name\n", encoding="utf-8")
    manifest = tmp_path / "ws_mcp.yaml"
    manifest.write_text(f"name: WS Selftest\nworkspace:\n  kind: local\n  root: {ws}\n", encoding="utf-8")
    return manifest, ws


class TestSelftest:
    """`--selftest` re-spawns the binary with the operator's flags, drives a
    real MCP handshake, and prints green/red per capability — the positive
    "did I set it up right?" step the operator feedback asked for."""

    def test_graph_mode_passes(self, graph_fixture: Path):
        rc, out = _run_selftest(["--graph", str(graph_fixture)])
        assert rc == 0, out
        assert "Selftest PASSED" in out
        assert "server initializes" in out
        assert "graph tools registered" in out
        assert "graph hydrates" in out

    def test_bad_graph_fails_nonzero(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.kgl"
        rc, out = _run_selftest(["--graph", str(missing)])
        assert rc != 0, out
        assert "Selftest FAILED" in out

    def test_local_workspace_passes(self, tmp_path: Path):
        manifest, _ws = _write_local_workspace_manifest(tmp_path)
        rc, out = _run_selftest(["--mcp-config", str(manifest)])
        assert rc == 0, out
        assert "Selftest PASSED" in out
        assert "workspace activation" in out

    def test_missing_explicit_selftest_path_fails_activation(self, tmp_path: Path):
        manifest, ws = _write_local_workspace_manifest(tmp_path)
        missing = ws / "does_not_exist"

        rc, out = _run_selftest(["--selftest-path", str(missing), "--mcp-config", str(manifest)])

        assert rc != 0, out
        assert "✗ workspace activation" in out
        assert "Path does not exist or is not a directory" in out
        assert "– graph hydrates" in out
        assert "Selftest FAILED" in out

    def test_explicit_path_without_builder_fails_hydration(self, tmp_path: Path):
        manifest, ws = _write_local_workspace_manifest(tmp_path)

        rc, out = _run_selftest(["--selftest-path", str(ws), "--mcp-config", str(manifest)])

        assert rc != 0, out
        assert "✓ workspace activation" in out
        assert "✗ graph hydrates: No active graph" in out
        assert "Selftest FAILED" in out

    def test_nonempty_recipe_catalog_reports_both_routes(self, graph_fixture: Path, tmp_path: Path):
        manifest = tmp_path / "recipes_mcp.yaml"
        manifest.write_text(
            "name: Recipe Selftest\n"
            "extensions:\n"
            "  cypher_recipes:\n"
            "    smoke:\n"
            "      description: Selftest operations.\n"
            "      queries:\n"
            "        count_people:\n"
            "          description: Count Person nodes.\n"
            "          parameters:\n"
            "            type: object\n"
            "            properties: {}\n"
            "            required: []\n"
            "            additionalProperties: false\n"
            "          cypher: MATCH (p:Person) RETURN count(p) AS people ORDER BY people\n",
            encoding="utf-8",
        )
        rc, out = _run_selftest(["--graph", str(graph_fixture), "--mcp-config", str(manifest)])
        assert rc == 0, out
        assert "recipe catalog tools" in out
        assert "list_recipe_queries + run_recipe_query present" in out


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
