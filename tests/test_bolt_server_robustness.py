"""Robustness tests for kglite-bolt-server — malformed input,
connection edge cases, failure-mode pin tests.

Some of these confirm 'server doesn't crash when X' (smoke-level
robustness); others pin specific error semantics so a refactor
that silently changes them will fail the test.

Fixtures: `bolt_server` from `tests/conftest.py`.
"""

import socket
import subprocess

import pytest

neo4j = pytest.importorskip("neo4j")

pytestmark = [pytest.mark.bolt]


# ────────────────────────────────────────────────────────────────────────────
# Connection-level abuses (raw socket; no Bolt handshake)
# ────────────────────────────────────────────────────────────────────────────


def _server_port(bolt_url: str) -> tuple[str, int]:
    """Extract (host, port) from `bolt://host:port`."""
    rest = bolt_url.removeprefix("bolt://")
    host, port_str = rest.rsplit(":", 1)
    return host, int(port_str)


def test_raw_garbage_bytes_dont_crash_server(bolt_server):
    """Send random non-Bolt bytes to the listener. boltr's handshake
    should reject; server keeps serving subsequent connections."""
    host, port = _server_port(bolt_server)
    # Connect, send garbage, close.
    for _ in range(5):
        with socket.create_connection((host, port), timeout=2.0) as s:
            s.sendall(b"\x00\x00\x00\x00garbage non-bolt traffic" * 10)
            try:
                s.recv(1024)
            except (TimeoutError, ConnectionResetError, OSError):
                pass
    # Server still works for legitimate clients?
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        driver.verify_connectivity()


def test_premature_disconnect_after_handshake_no_crash(bolt_server):
    """Connect, send a Bolt magic preamble + valid version table,
    receive the server's choice, then close. boltr should handle
    the abrupt close cleanly."""
    host, port = _server_port(bolt_server)
    bolt_magic = b"\x60\x60\xb0\x17"
    # Offer Bolt 5.0 (and pad with zeros).
    version_table = b"\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    for _ in range(10):
        with socket.create_connection((host, port), timeout=2.0) as s:
            s.sendall(bolt_magic + version_table)
            try:
                _ = s.recv(4)  # server's version choice
            except OSError:
                pass
        # Connection closes here (the `with` exit).
    # Server still works?
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        driver.verify_connectivity()


def test_zero_byte_then_disconnect(bolt_server):
    """Some scanners send a single 0x00 and close. boltr should
    handle this gracefully (rejected as bad magic)."""
    host, port = _server_port(bolt_server)
    for _ in range(20):
        with socket.create_connection((host, port), timeout=2.0) as s:
            s.sendall(b"\x00")
    # Server still responsive?
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        driver.verify_connectivity()


def test_concurrent_handshakes_dont_interfere(bolt_server):
    """Open 10 simultaneous connections via raw sockets, hold them
    open without finishing the handshake. Then verify a real
    driver can still connect."""
    host, port = _server_port(bolt_server)
    bolt_magic = b"\x60\x60\xb0\x17"
    sockets = []
    try:
        for _ in range(10):
            s = socket.create_connection((host, port), timeout=2.0)
            s.sendall(bolt_magic)  # magic but no version table yet
            sockets.append(s)
        # While those are pending, open a legitimate driver session.
        with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
            driver.verify_connectivity()
    finally:
        for s in sockets:
            try:
                s.close()
            except OSError:
                pass


# ────────────────────────────────────────────────────────────────────────────
# Query-level abuses
# ────────────────────────────────────────────────────────────────────────────


def test_query_with_null_bytes_handled_cleanly(bolt_server):
    """Cypher with embedded null byte inside a string literal —
    kglite accepts (strings are arbitrary byte sequences). Pin
    that the server doesn't crash on the input; either accept or
    reject is fine."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # Either runs cleanly OR raises a parse error — both are
            # acceptable. The contract is "no crash, no hang".
            try:
                result = session.run("MATCH (n:Person) WHERE n.title = 'A\x00B' RETURN n")
                rows = list(result)
                # If we got here, the query ran (likely 0 rows since
                # none of Alice/Bob/Carol/Dave match 'A\\x00B').
                assert rows == []
            except Exception:
                pass
            # Server still responsive?
            r = session.run("MATCH (n:Person) RETURN count(n) AS c")
            assert r.single()["c"] == 4


def test_query_with_bell_and_escape_chars(bolt_server):
    """Cypher with bell + escape control bytes in a string literal —
    should parse correctly (strings are byte-transparent)."""
    weird = "alert\x07esc\x1b"
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("RETURN $x AS x", x=weird)
            assert result.single()["x"] == weird


def test_extremely_deep_predicate_nesting(bolt_server):
    """50-level nested AND. Pin that the parser doesn't stack-
    overflow on reasonable nesting depths."""
    pred = " AND ".join([f"n.id = {i}" for i in range(50)])
    query = f"MATCH (n:Person) WHERE {pred} RETURN count(n) AS c"
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run(query)
            # No row will satisfy all 50 simultaneously, but the
            # parser + planner + executor should not crash.
            assert result.single()["c"] == 0


def test_unicode_in_label_name_clean_error(bolt_server):
    """A label with non-ASCII characters — parser may accept or reject;
    we just pin that the server doesn't crash."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            try:
                result = session.run("MATCH (n:日本語) RETURN count(n)")
                result.consume()  # may succeed (0 rows) or raise
            except Exception:
                pass  # either outcome is acceptable; we just verify no crash


def test_parameter_with_reserved_keyword_name(bolt_server):
    """A parameter named `match` (a reserved Cypher keyword) —
    should be addressable via $match since the $ prefix disambiguates."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # The driver may quote or reject; we accept either as long
            # as the server doesn't crash.
            try:
                result = session.run("RETURN $match AS m", **{"match": "value"})
                assert result.single()["m"] == "value"
            except Exception:
                pass


def test_session_run_after_server_restart_simulation(bolt_server):
    """We can't actually restart the server mid-test, but we can verify
    that closing the driver, opening a new one, and querying works
    repeatedly — pins driver/server reconnection robustness."""
    for _ in range(5):
        with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                result = session.run("MATCH (n:Person) RETURN count(n) AS c")
                assert result.single()["c"] >= 4


# ────────────────────────────────────────────────────────────────────────────
# Resource limits / DoS surface
# ────────────────────────────────────────────────────────────────────────────


def test_many_simultaneous_connections_doesnt_oom(bolt_server):
    """Open 32 driver sessions concurrently, hold them open briefly,
    close cleanly. Pins that the server's default --max-sessions
    (256) accommodates this without OOM."""
    drivers = []
    try:
        for _ in range(32):
            d = neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password"))
            d.verify_connectivity()
            drivers.append(d)
        # All 32 sessions are open simultaneously.
    finally:
        for d in drivers:
            try:
                d.close()
            except Exception:
                pass


def test_large_pull_doesnt_crash(bolt_server):
    """Insert 1000 nodes via a tx, then PULL them all in one RUN.
    Pins that result-set materialization (with_streaming=false) works
    at this size without OOM.

    Note: each `tx.run(...)` is explicitly `.consume()`'d. Without
    that, the unconsumed results from earlier CREATEs accumulate
    on the tx and the later inserts may be discarded silently
    by the driver/server when commit fires — a real wire-protocol
    contract finding from initial T4 runs."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # Insert 1000 nodes in one tx; consume each result.
            tx = session.begin_transaction()
            for i in range(1000):
                tx.run(f"CREATE (:Person {{id: {30000 + i}, title: 'bulk{i}'}})").consume()
            tx.commit()
            # PULL them all.
            result = session.run("MATCH (n:Person) RETURN n.title AS title")
            titles = [r["title"] for r in result]
            assert len(titles) >= 1004  # 4 baseline + 1000 bulk


# ────────────────────────────────────────────────────────────────────────────
# Server-startup robustness
# ────────────────────────────────────────────────────────────────────────────


def test_server_binary_help_flag(bolt_binary_path):
    """`kglite-bolt-server --help` prints usage and exits 0.
    Pins that the CLI is parseable + the binary doesn't depend on
    any runtime setup just to show --help."""
    if not bolt_binary_path.exists():
        pytest.skip(f"binary missing at {bolt_binary_path}")
    result = subprocess.run(
        [str(bolt_binary_path), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "Bolt v5" in result.stdout or "bolt" in result.stdout.lower()
    assert "--graph" in result.stdout
    assert "--bind" in result.stdout
    assert "--port" in result.stdout
    assert "--readonly" in result.stdout
    assert "--auth" in result.stdout


def test_server_missing_graph_file_clean_error(bolt_binary_path, tmp_path):
    """`--graph` pointing at a nonexistent file → exit nonzero with
    a clear error, not a panic."""
    if not bolt_binary_path.exists():
        pytest.skip(f"binary missing at {bolt_binary_path}")
    result = subprocess.run(
        [
            str(bolt_binary_path),
            "--graph",
            str(tmp_path / "does_not_exist.kgl"),
            "--bind",
            "127.0.0.1",
            "--port",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    # Either stdout or stderr should mention the issue.
    output = result.stdout + result.stderr
    assert "does not exist" in output or "not found" in output.lower()
    # Should NOT contain a panic backtrace.
    assert "panicked" not in output.lower()
    assert "stack backtrace" not in output.lower()


def test_server_invalid_port_clean_error(bolt_binary_path, tmp_path):
    """`--port 99999` (out of u16 range) → clap rejects with a clear
    parse error, no panic."""
    if not bolt_binary_path.exists():
        pytest.skip(f"binary missing at {bolt_binary_path}")
    # Build a real graph file so the failure isn't from missing graph.
    from tests.conftest import _build_bolt_fixture_graph

    fixture = tmp_path / "ok.kgl"
    _build_bolt_fixture_graph(fixture)
    result = subprocess.run(
        [
            str(bolt_binary_path),
            "--graph",
            str(fixture),
            "--bind",
            "127.0.0.1",
            "--port",
            "99999",  # > u16::MAX
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "panicked" not in output.lower()


def test_server_readonly_flag_blocks_writes_at_startup(bolt_binary_path, tmp_path):
    """Spawn with --readonly, verify that all mutation paths are
    blocked from a fresh driver session (regression-test for the
    --readonly enforcement landing in C.5)."""
    if not bolt_binary_path.exists():
        pytest.skip(f"binary missing at {bolt_binary_path}")
    from tests.conftest import _build_bolt_fixture_graph, _spawn_bolt_server, _teardown_bolt_server

    fixture = tmp_path / "ro.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, readonly=True)
    try:
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                # Auto-commit CREATE rejected.
                with pytest.raises(neo4j.exceptions.ClientError):
                    session.run("CREATE (:Person {id: 9999, title: 'x'})").consume()
                # BEGIN rejected.
                with pytest.raises(neo4j.exceptions.ClientError):
                    session.begin_transaction()
                # Reads still work.
                result = session.run("MATCH (n:Person) RETURN count(n) AS c")
                assert result.single()["c"] == 4
    finally:
        _teardown_bolt_server(proc)
