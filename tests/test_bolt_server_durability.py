"""Durability surface of kglite-bolt-server — `--save-on-exit` and the
signals that trigger it.

The server's writes are process-local: they live in the served graph's
in-memory state and reach the `.kgl` file only when something checkpoints
them. `test_write_over_bolt_does_not_reach_the_file_without_save_on_exit`
pins that baseline — every other test here measures against it, and without
it a passing save-on-exit test could just be reading a write the server had
persisted anyway.

POSIX-gated: the graceful-stop helper delivers SIGINT/SIGTERM, which have no
Windows equivalent that means "shut down cleanly".
"""

import os
import signal
import subprocess

import pytest

import kglite

neo4j = pytest.importorskip("neo4j")

from tests.conftest import (  # noqa: E402
    _BOLT_SKIP_REASON,
    _bolt_binary_available,
    _build_bolt_fixture_graph,
    _graceful_stop_bolt_server,
    _spawn_bolt_server,
    _teardown_bolt_server,
)

pytestmark = [
    pytest.mark.bolt,
    pytest.mark.skipif(os.name != "posix", reason="SIGINT/SIGTERM shutdown is POSIX-only"),
]

MARKER_QUERY = "MATCH (p:Person {title: 'Zed'}) RETURN count(p) AS c"


def _require_binary():
    if not _bolt_binary_available():
        pytest.skip(_BOLT_SKIP_REASON)


def _write_marker(url: str) -> None:
    """Commit one node through an explicit transaction, so the write is
    definitely committed (not merely sent) before the server is signalled."""
    with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            tx.run("CREATE (:Person {id: 99, title: 'Zed', city: 'Tromso'})")
            tx.commit()
        # Read it back on a fresh session: the write is live in the server.
        with driver.session() as session:
            assert session.run(MARKER_QUERY).single()["c"] == 1


def _marker_count_on_disk(path) -> int:
    """Reopen the served `.kgl` in-process and count the marker node."""
    g = kglite.open(str(path))
    return g.cypher(MARKER_QUERY).scalar()


def _run_server_binary(bolt_binary_path, args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(bolt_binary_path), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, **env} if env else None,
    )


# ────────────────────────────────────────────────────────────────────────────
# Baseline: without the flag, a committed write never reaches the file
# ────────────────────────────────────────────────────────────────────────────


def test_write_over_bolt_does_not_reach_the_file_without_save_on_exit(tmp_path):
    """The process-local baseline. A graceful SIGINT shutdown with no
    `--save-on-exit` leaves the served file exactly as it was."""
    _require_binary()
    fixture = tmp_path / "baseline.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture)
    try:
        _write_marker(url)
        status = _graceful_stop_bolt_server(proc, signal.SIGINT)
    finally:
        if proc.poll() is None:
            _teardown_bolt_server(proc)
    assert status == 0, "a SIGINT shutdown is a clean stop"
    assert _marker_count_on_disk(fixture) == 0, (
        "without --save-on-exit the server must not write the graph back — "
        "if this ever passes with 1 the save-on-exit tests below prove nothing"
    )


# ────────────────────────────────────────────────────────────────────────────
# --save-on-exit round-trips, on both shutdown signals
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_save_on_exit_persists_a_committed_write(tmp_path, sig):
    """Spawn with `--save-on-exit`, commit over Bolt, signal, and find the
    write in the file. SIGTERM is the one a supervisor actually sends."""
    _require_binary()
    fixture = tmp_path / f"exit-save-{sig}.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=["--save-on-exit"])
    try:
        _write_marker(url)
        status = _graceful_stop_bolt_server(proc, sig)
    finally:
        if proc.poll() is None:
            _teardown_bolt_server(proc)
    assert status == 0, "a successful exit save exits zero"
    assert _marker_count_on_disk(fixture) == 1, f"--save-on-exit must write the committed graph back on {sig.name}"


def test_save_on_exit_env_mirror_persists_a_committed_write(tmp_path):
    """`KGLITE_BOLT_SAVE_ON_EXIT=1` does what the flag does — the spelling a
    Compose file or unit file can set."""
    _require_binary()
    fixture = tmp_path / "exit-save-env.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, env={"KGLITE_BOLT_SAVE_ON_EXIT": "1"})
    try:
        _write_marker(url)
        status = _graceful_stop_bolt_server(proc, signal.SIGTERM)
    finally:
        if proc.poll() is None:
            _teardown_bolt_server(proc)
    assert status == 0
    assert _marker_count_on_disk(fixture) == 1


def test_exit_save_reports_the_saved_graph_version(tmp_path):
    """The version is logged next to the path: shutdown does not drain
    connections, so an operator needs to be able to tell a save that ran from
    a commit that landed after it."""
    _require_binary()
    fixture = tmp_path / "exit-save-log.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=["--save-on-exit"])
    try:
        _write_marker(url)
        _graceful_stop_bolt_server(proc, signal.SIGINT)
    finally:
        if proc.poll() is None:
            _teardown_bolt_server(proc)
    # `tracing_subscriber::fmt()` writes to stdout; read both so the assertion
    # does not silently depend on which stream the subscriber picks.
    logs = proc.stdout.read().decode("utf-8", errors="replace") + proc.stderr.read().decode("utf-8", errors="replace")
    assert "save-on-exit complete" in logs, logs
    assert "graph_version" in logs, logs


# ────────────────────────────────────────────────────────────────────────────
# Startup refusals
# ────────────────────────────────────────────────────────────────────────────


def test_save_on_exit_conflicts_with_readonly_flag(bolt_binary_path, tmp_path):
    """clap rejects the flag pair before anything is opened."""
    _require_binary()
    fixture = tmp_path / "ro.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0", "--readonly", "--save-on-exit"],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    # "cannot be used with" is clap's conflict phrasing — asserting on the two
    # flag names alone would also pass against a build where `--save-on-exit`
    # does not exist at all (its "unexpected argument" error names both).
    assert "cannot be used with" in combined, combined
    assert "--readonly" in combined and "--save-on-exit" in combined, combined


def test_save_on_exit_env_mirror_is_refused_with_readonly(bolt_binary_path, tmp_path):
    """The environment spelling is invisible to clap's `conflicts_with`, so
    the server checks it itself — and fails startup rather than serving a
    read-only server that silently never saves."""
    _require_binary()
    fixture = tmp_path / "ro-env.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0", "--readonly"],
        env={"KGLITE_BOLT_SAVE_ON_EXIT": "1"},
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "--readonly" in combined and "nothing to write back" in combined, combined


def test_save_on_exit_is_refused_for_a_disk_graph(bolt_binary_path, tmp_path):
    """Disk graphs are excluded: every disk save publishes a new generation
    and nothing prunes them, so exit checkpoints would grow the directory."""
    _require_binary()
    disk_dir = tmp_path / "disk-graph"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.cypher("CREATE (:Person {id: 1, title: 'Alice'})")
    g.save(str(disk_dir))
    del g

    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(disk_dir), "--port", "0", "--save-on-exit"],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "disk-mode" in combined and "generation" in combined, combined
