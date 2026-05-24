"""Bolt v5 wire-protocol smoke tests against `target/release/kglite-bolt-server`.

These 8 tests are the **failing-by-design contract** for Phase C of
`bolt_implementation.md`. Each is `pytest.mark.xfail(strict=True)` and
tagged with the Phase C sub-phase that retires it. Strict mode does two
things:

1. **Every test must fail in Phase B.** The bolt-server binary boots and
   binds a port, but every `BoltBackend` method body is
   `unimplemented!("phase C.X ...")` — so the first real Bolt message
   panics the connection task and the driver sees a broken pipe. The
   driver's `verify_connectivity` raises, the test catches via `xfail`,
   pytest reports XFAIL, the suite is green.

2. **Each test must turn green on exactly its retiring sub-phase.** If
   Phase C.2 (RUN/PULL scalars) accidentally fixes test #4 (Node return)
   before Phase C.4 lands, strict mode flips XFAIL → XPASS → fail, and
   CI alerts the maintainer to the surprise regression-in-reverse.

The suite is gated three ways:

- `pytest.importorskip("neo4j")` — silent skip if the driver isn't
  installed (the `[neo4j]` extra is opt-in for the conformance runner).
- `pytestmark = pytest.mark.skipif(not BINARY.exists(), ...)` — silent
  skip if the release binary hasn't been built (the CI binary-build step
  is what makes this active).
- `pytest.mark.bolt` — excluded from the default test run via
  `pyproject.toml` `addopts`; opt in via `pytest -m bolt`.
"""

from pathlib import Path
import socket
import subprocess
import time

import pandas as pd
import pytest

import kglite

neo4j = pytest.importorskip("neo4j")


BINARY = Path(__file__).resolve().parent.parent / "target" / "release" / "kglite-bolt-server"

pytestmark = [
    pytest.mark.bolt,
    pytest.mark.skipif(
        not BINARY.exists(),
        reason=(
            f"kglite-bolt-server binary not built (missing at {BINARY}). "
            f"Build with `cargo build -p kglite-bolt-server --release`."
        ),
    ),
]


# ── Fixture builders ──────────────────────────────────────────────────────


def _build_fixture_graph(path: Path) -> None:
    """Build a small Person/KNOWS graph, save to ``path``.

    Mirrors `tests/test_mcp_server_smoke.py::_build_fixture_graph` —
    intentional copy rather than shared helper. The two smoke suites
    serve different protocols (MCP stdio vs Bolt TCP) and may drift
    independently; sharing the 15-liner adds cross-file coupling for
    little gain. Revisit in Phase D.
    """
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


def _find_free_port() -> int:
    """Bind a socket to port 0, read the OS-assigned port, close.

    Brief race window between close() and the spawned server's bind() —
    a concurrent process could grab the port in between. Acceptable for
    test isolation; the failure mode is a clean spawn-time error, not
    a silent hang.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_listener(host: str, port: int, deadline_s: float) -> None:
    """Poll-connect a raw TCP socket until the listener answers.

    Used *before* attempting the Bolt handshake — we want to know the
    listener is up regardless of whether the Bolt handshake succeeds
    (in Phase B, handshake itself is stubbed and panics). This is the
    minimum signal that the binary spawned successfully.
    """
    deadline = time.monotonic() + deadline_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError) as e:
            last_err = e
            time.sleep(0.1)
    raise RuntimeError(f"bolt server never started listening on {host}:{port}: {last_err}")


@pytest.fixture
def bolt_server(tmp_path):
    """Spawn `kglite-bolt-server` on an ephemeral port; yield the URL.

    Kills the subprocess on test exit. Captures stdout+stderr so a
    spawn failure surfaces in the test report.
    """
    fixture_path = tmp_path / "fixture.kgl"
    _build_fixture_graph(fixture_path)
    port = _find_free_port()

    proc = subprocess.Popen(
        [
            str(BINARY),
            "--graph",
            str(fixture_path),
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"bolt://127.0.0.1:{port}"

    try:
        _wait_for_listener("127.0.0.1", port, deadline_s=10.0)
    except Exception:
        proc.kill()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else "<no stderr>"
        raise RuntimeError(f"server failed to start. stderr:\n{stderr}")

    yield url

    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=2)


# ── The 8 tests — each xfail(strict=True), tagged to its retiring sub-phase ──


def test_bolt_handshake_and_verify_connectivity(bolt_server):
    """Phase C.1 ✓: HELLO/LOGON/GOODBYE + a `verify_connectivity()` ping.

    The backend's `create_session` / `get_server_info` / `close_session`
    + the auto-no-op `set_session_auth` (boltr skips it when no
    `AuthValidator` is wired) are enough to satisfy the neo4j Python
    driver's handshake. Driver opens a connection, sends HELLO + LOGON,
    receives SUCCESS for both, sends GOODBYE, closes. No queries run.
    """
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        driver.verify_connectivity()


def test_bolt_run_returns_scalar_rows(bolt_server):
    """Phase C.2 ✓: RUN a trivial scalar query, PULL all, check the rows."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("MATCH (n:Person) RETURN n.title AS name ORDER BY name")
            names = [record["name"] for record in result]
    assert names == ["Alice", "Bob", "Carol", "Dave"]


def test_bolt_run_supports_parameters(bolt_server):
    """Phase C.3 ✓: RUN with `$param` map decoded from PackStream."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run(
                "MATCH (n:Person {city: $c}) RETURN n.title AS name ORDER BY name",
                c="Oslo",
            )
            names = [record["name"] for record in result]
    assert names == ["Alice", "Carol"]


@pytest.mark.xfail(strict=True, reason="retired by Phase C.4 — Node/Rel/Path RETURN (needs A.1 ✓)")
def test_bolt_return_node_yields_node_struct(bolt_server):
    """Phase C.4: `RETURN n` maps Value::Node → BoltNode PackStream struct (0x4E)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("MATCH (n:Person {title: 'Alice'}) RETURN n")
            record = result.single()
            assert record is not None
            node = record["n"]
            assert isinstance(node, neo4j.graph.Node)
            assert "Person" in node.labels
            assert node["title"] == "Alice"
            assert node["city"] == "Oslo"


@pytest.mark.xfail(strict=True, reason="retired by Phase C.4 — Node/Rel/Path RETURN (needs A.1 ✓)")
def test_bolt_return_relationship_yields_rel_struct(bolt_server):
    """Phase C.4: `RETURN r` maps Value::Relationship → BoltRelationship (0x52)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("MATCH (:Person {title: 'Alice'})-[r:KNOWS]->(:Person {title: 'Bob'}) RETURN r")
            record = result.single()
            assert record is not None
            rel = record["r"]
            assert isinstance(rel, neo4j.graph.Relationship)
            assert rel.type == "KNOWS"


@pytest.mark.xfail(strict=True, reason="retired by Phase C.5 — BEGIN/COMMIT/ROLLBACK")
def test_bolt_transaction_commit_and_rollback(bolt_server):
    """Phase C.5: explicit `tx.run()` + `tx.commit()` / `tx.rollback()`."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # Commit a mutation, verify it's visible.
            tx = session.begin_transaction()
            tx.run("CREATE (:Person {id: 99, title: 'Eve', city: 'Stavanger'})")
            tx.commit()
            after_commit = session.run("MATCH (n:Person {title: 'Eve'}) RETURN count(n) AS c").single()["c"]
            assert after_commit == 1

            # Rollback another mutation, verify it's discarded.
            tx2 = session.begin_transaction()
            tx2.run("CREATE (:Person {id: 100, title: 'Frank', city: 'Tromsø'})")
            tx2.rollback()
            after_rb = session.run("MATCH (n:Person {title: 'Frank'}) RETURN count(n) AS c").single()["c"]
            assert after_rb == 0


@pytest.mark.xfail(strict=True, reason="retired by Phase C.5 — --readonly enforcement")
def test_bolt_rejects_writes_when_readonly(bolt_server, tmp_path):
    """Phase C.5: `--readonly` flag rejects mutations with a Bolt FAILURE.

    NOTE: this test currently uses the shared `bolt_server` fixture
    which spawns without `--readonly`. Phase C.5 will adjust either this
    test to spawn its own readonly instance, or the fixture to be
    parametrized. The xfail covers either intermediate state.
    """
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError):
                session.run("CREATE (:Person {id: 999, title: 'Bad'})").consume()


@pytest.mark.xfail(strict=True, reason="retired by Phase C.6 — KgError → Bolt FAILURE mapping (needs A.2 ✓)")
def test_bolt_returns_failure_on_parse_error(bolt_server):
    """Phase C.6: a syntactically invalid Cypher returns Bolt FAILURE
    with a `Neo.ClientError.Statement.SyntaxError` code (the canonical
    Neo4j status code for this case), driven by the KgErrorCode
    enum that Phase A.2 shipped."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError) as exc_info:
                session.run("MATCH NOT VALID CYPHER").consume()
            assert "Syntax" in str(exc_info.value.code)
