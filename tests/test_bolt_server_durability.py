"""Durability surface of kglite-bolt-server — `--save-on-exit`, the signals
that trigger it, the `CALL db.checkpoint()` verb, and `--checkpoint-interval`.

The server's writes are process-local: they live in the served graph's
in-memory state and reach the `.kgl` file only when something checkpoints
them. `test_write_over_bolt_does_not_reach_the_file_without_save_on_exit`
pins that baseline — every other test here measures against it, and without
it a passing save-on-exit test could just be reading a write the server had
persisted anyway.

POSIX-gated: the graceful-stop helper delivers SIGINT/SIGTERM, which have no
Windows equivalent that means "shut down cleanly".
"""

import hashlib
import os
import shutil
import signal
import subprocess
import time
from uuid import uuid4

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


def _named_count_query(title: str) -> str:
    return f"MATCH (p:Person {{title: '{title}'}}) RETURN count(p) AS c"


def _write_marker(url: str, title: str = "Zed", node_id: int = 99) -> None:
    """Commit one node through an explicit transaction, so the write is
    definitely committed (not merely sent) before the server is signalled.

    `node_id` is distinct per marker wherever two markers must coexist *after a
    WAL replay*: the log records a mutation as an upsert keyed by `(type, id)`,
    so two `CREATE`s sharing an id are two nodes live and one node replayed.
    """
    with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            tx.run(f"CREATE (:Person {{id: {node_id}, title: '{title}', city: 'Tromso'}})")
            tx.commit()
        # Read it back on a fresh session: the write is live in the server.
        with driver.session() as session:
            assert session.run(_named_count_query(title)).single()["c"] == 1


def _marker_count_on_disk(path, title: str = "Zed") -> int:
    """Reopen the served `.kgl` in-process and count the marker node."""
    g = kglite.open(str(path))
    return g.cypher(_named_count_query(title)).scalar()


def _snapshot(served, tmp_path):
    """Copy the served file so it can be opened while the server still runs.

    The running server holds the graph's writer lease, so `kglite.open` on the
    served path itself would contend with it. Every checkpoint is an atomic
    temp+rename, so a copy is always a whole file, never a half-written one.
    """
    snap = tmp_path / f"snapshot-{uuid4().hex}.kgl"
    shutil.copyfile(served, snap)
    return snap


def _marker_count_in_snapshot(served, tmp_path, title: str = "Zed") -> int:
    return _marker_count_on_disk(_snapshot(served, tmp_path), title)


def _digest(path) -> str:
    """Content hash of the served file.

    Content, not mtime: this suite runs on filesystems whose timestamp
    granularity is coarse enough that two writes a tick apart can share an
    mtime, which would turn an idle-skip assertion into a coin flip.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wait_until(predicate, timeout: float, what: str):
    """Poll `predicate` until it returns a truthy value, or fail saying what
    was being waited for. Bounded well inside the suite's 120 s ceiling."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.2)
    raise AssertionError(f"timed out after {timeout}s waiting for {what} (last value: {last!r})")


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


# ────────────────────────────────────────────────────────────────────────────
# CALL db.checkpoint() — the on-demand verb (bolt-only; see p2-red.txt for
# what this query did before the intercept existed)
# ────────────────────────────────────────────────────────────────────────────

CHECKPOINT = "CALL db.checkpoint()"


def _checkpoint(url: str, query: str = CHECKPOINT):
    """Run the verb on a fresh driver session and return the single record."""
    with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            return session.run(query).single()


def test_checkpoint_returns_the_neo4j_result_shape(tmp_path):
    """Columns `success, message` with one record — the shape Neo4j's own
    `db.checkpoint()` yields, so a client can consume it identically."""
    _require_binary()
    fixture = tmp_path / "shape.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture)
    try:
        record = _checkpoint(url)
        assert list(record.keys()) == ["success", "message"]
        assert record["success"] is True
        assert record["message"].startswith("checkpoint written: version "), record["message"]
        # An explicit YIELD projects exactly what it asked for, in its order.
        yielded = _checkpoint(url, "CALL db.checkpoint() YIELD message, success")
        assert list(yielded.keys()) == ["message", "success"]
        assert yielded["success"] is True
    finally:
        _teardown_bolt_server(proc)


def test_checkpoint_writes_the_committed_graph_to_the_served_file(tmp_path):
    """The point of the verb. The server is then SIGKILLed — a killed process
    runs no exit hook, so the write can only have reached the file through the
    checkpoint."""
    _require_binary()
    fixture = tmp_path / "verb-write.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture)
    try:
        _write_marker(url)
        assert _checkpoint(url)["success"] is True
    finally:
        _teardown_bolt_server(proc)
    assert _marker_count_on_disk(fixture) == 1, "the checkpoint must write the committed graph back"


def test_checkpoint_bounds_the_loss_window_across_a_sigkill(tmp_path):
    """The documented loss window, proven in both directions: a write made
    *before* `db.checkpoint()` survives a SIGKILL and a restart on the same
    file; a write made *after* it is gone. Neither half alone is evidence —
    the first without the second would also pass on a server that saved
    everything continuously."""
    _require_binary()
    fixture = tmp_path / "loss-window.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture)
    try:
        _write_marker(url, "Before")
        assert _checkpoint(url)["success"] is True
        _write_marker(url, "After")
    finally:
        _teardown_bolt_server(proc)  # SIGKILL: no shutdown path, no exit save

    # Restart on the same file — the server is the reader, so this also proves
    # the checkpointed file is servable and the dead process's lease is gone.
    restarted, restarted_url = _spawn_bolt_server(fixture)
    try:
        with neo4j.GraphDatabase.driver(restarted_url, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                before = session.run(_named_count_query("Before")).single()["c"]
                after = session.run(_named_count_query("After")).single()["c"]
    finally:
        _teardown_bolt_server(restarted)
    assert before == 1, "a write committed before the checkpoint must survive the crash"
    assert after == 0, "a write committed after the checkpoint is the documented loss window"


def test_checkpoint_skips_when_the_graph_has_not_changed(tmp_path):
    """Digest-skip: the second checkpoint of an unchanged graph reports the
    skip and leaves the file untouched (mtime, byte-for-byte)."""
    _require_binary()
    fixture = tmp_path / "skip.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture)
    try:
        _write_marker(url)
        first = _checkpoint(url)
        assert first["message"].startswith("checkpoint written: version "), first["message"]
        stat_after_write = fixture.stat()

        second = _checkpoint(url)
        assert second["success"] is True
        assert second["message"].startswith("skipped: graph unchanged since version "), second["message"]
        assert fixture.stat().st_mtime_ns == stat_after_write.st_mtime_ns, (
            "a skipped checkpoint must not rewrite the served file"
        )

        # Mutation check: a further write makes the next checkpoint save again.
        _write_marker(url, "Later")
        third = _checkpoint(url)
        assert third["message"].startswith("checkpoint written: version "), third["message"]
    finally:
        _teardown_bolt_server(proc)
    assert _marker_count_on_disk(fixture, "Later") == 1


def test_checkpoint_is_refused_on_a_readonly_server(tmp_path):
    """`--readonly` refuses the verb with the exact Neo4j security code, and
    writes nothing."""
    _require_binary()
    fixture = tmp_path / "readonly.kgl"
    _build_bolt_fixture_graph(fixture)
    before = fixture.stat().st_mtime_ns
    proc, url = _spawn_bolt_server(fixture, readonly=True)
    try:
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                with pytest.raises(neo4j.exceptions.Forbidden) as excinfo:
                    session.run(CHECKPOINT).single()
    finally:
        _teardown_bolt_server(proc)
    assert excinfo.value.code == "Neo.ClientError.Security.Forbidden"
    assert "--readonly" in excinfo.value.message
    assert fixture.stat().st_mtime_ns == before, "a refused checkpoint writes nothing"


def test_checkpoint_is_refused_for_a_disk_graph_over_the_wire(tmp_path):
    """A disk graph serves normally; only the checkpoint verb is refused, for
    the same unbounded-generation reason `--save-on-exit` refuses at startup."""
    _require_binary()
    disk_dir = tmp_path / "disk-verb"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.cypher("CREATE (:Person {id: 1, title: 'Alice'})")
    g.save(str(disk_dir))
    del g

    proc, url = _spawn_bolt_server(disk_dir)
    try:
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                # The graph itself is perfectly queryable — the refusal is
                # specific to the verb, not to serving a disk graph.
                assert session.run("MATCH (n:Person) RETURN count(n) AS c").single()["c"] == 1
                with pytest.raises(neo4j.exceptions.Forbidden) as excinfo:
                    session.run(CHECKPOINT).single()
    finally:
        _teardown_bolt_server(proc)
    assert excinfo.value.code == "Neo.ClientError.Security.Forbidden"
    assert "generation" in excinfo.value.message


def test_checkpoint_is_refused_inside_an_explicit_transaction(tmp_path):
    """A checkpoint writes the *committed* graph, which by definition excludes
    the calling transaction's uncommitted writes — so running it there would
    report success over a file missing the work just done. Refused, and the
    transaction is still usable afterwards."""
    _require_binary()
    fixture = tmp_path / "in-tx.kgl"
    _build_bolt_fixture_graph(fixture)
    before = fixture.stat().st_mtime_ns
    proc, url = _spawn_bolt_server(fixture)
    try:
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                tx = session.begin_transaction()
                tx.run("CREATE (:Person {id: 99, title: 'Zed'})")
                with pytest.raises(neo4j.exceptions.ClientError) as excinfo:
                    tx.run(CHECKPOINT).single()
                tx.rollback()
        assert fixture.stat().st_mtime_ns == before, "a refused checkpoint writes nothing"
        # COMMIT-then-checkpoint is the supported spelling, and still works.
        _write_marker(url)
        assert _checkpoint(url)["success"] is True
    finally:
        _teardown_bolt_server(proc)
    assert excinfo.value.code == "Neo.ClientError.Request.Invalid"
    assert "explicit transaction" in excinfo.value.message
    assert _marker_count_on_disk(fixture) == 1


# ────────────────────────────────────────────────────────────────────────────
# --checkpoint-interval — the periodic task
#
# Every test here uses a 1-second interval and a bounded poll: the point is
# that a tick happens and what it does, never how fast it is.
# ────────────────────────────────────────────────────────────────────────────

INTERVAL_ARGS = ["--checkpoint-interval", "1"]


def test_checkpoint_interval_writes_a_committed_write_without_any_other_trigger(tmp_path):
    """A write committed over Bolt reaches the file on its own — no verb call,
    no shutdown. The server is SIGKILLed afterwards, so nothing but a tick can
    have written it (the baseline test at the top of this file pins that a
    server without any checkpointing writes nothing at all)."""
    _require_binary()
    fixture = tmp_path / "interval-write.kgl"
    _build_bolt_fixture_graph(fixture)
    assert _marker_count_on_disk(fixture) == 0
    proc, url = _spawn_bolt_server(fixture, extra_args=INTERVAL_ARGS)
    try:
        _write_marker(url)
        _wait_until(
            lambda: _marker_count_in_snapshot(fixture, tmp_path) == 1,
            timeout=25.0,
            what="the periodic checkpoint to write the committed graph",
        )
    finally:
        _teardown_bolt_server(proc)  # SIGKILL: no exit save, no shutdown path
    assert _marker_count_on_disk(fixture) == 1


def test_checkpoint_interval_env_mirror_writes_a_committed_write(tmp_path):
    """`KGLITE_BOLT_CHECKPOINT_INTERVAL=1` does what the flag does."""
    _require_binary()
    fixture = tmp_path / "interval-env.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, env={"KGLITE_BOLT_CHECKPOINT_INTERVAL": "1"})
    try:
        _write_marker(url)
        _wait_until(
            lambda: _marker_count_in_snapshot(fixture, tmp_path) == 1,
            timeout=25.0,
            what="the env-configured periodic checkpoint to write",
        )
    finally:
        _teardown_bolt_server(proc)


def test_checkpoint_interval_skips_ticks_while_the_graph_is_idle(tmp_path):
    """After the graph is on disk, further ticks on an unchanged graph write
    nothing — the file is byte-identical across several tick windows."""
    _require_binary()
    fixture = tmp_path / "interval-idle.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=INTERVAL_ARGS)
    try:
        _write_marker(url)
        _wait_until(
            lambda: _marker_count_in_snapshot(fixture, tmp_path) == 1,
            timeout=25.0,
            what="the first checkpoint of the written graph",
        )
        settled = _digest(fixture)
        # No client writes from here: at least three further ticks pass.
        time.sleep(3.5)
        assert _digest(fixture) == settled, "ticks over an unchanged graph must not rewrite the served file"
        # Mutation check on the skip itself: a further write makes the next
        # tick save again, so the stability above is a skip, not a dead task.
        _write_marker(url, "Later")
        _wait_until(
            lambda: _digest(fixture) != settled,
            timeout=25.0,
            what="a tick after a new write to rewrite the file",
        )
    finally:
        _teardown_bolt_server(proc)
    assert _marker_count_on_disk(fixture, "Later") == 1


def test_first_tick_rewrites_a_file_that_predates_the_process(tmp_path):
    """The first tick saves unconditionally. The served file is replaced *on
    disk* with a different graph after the server loaded it — the state the
    server serves and the state on disk have now diverged with no version bump
    to notice — and the first tick must restore what is being served. A
    digest-skip seeded from the graph as loaded would leave the stale file
    there forever."""
    _require_binary()
    fixture = tmp_path / "interval-stale.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=INTERVAL_ARGS)
    try:
        # A graph the server has never seen, written under the served path.
        stale = tmp_path / "stale-source.kgl"
        _build_bolt_fixture_graph(stale)
        g = kglite.open(str(stale))
        g.cypher("CREATE (:Person {id: 77, title: 'Stale', city: 'Nowhere'})")
        g.save(str(stale))
        del g
        os.replace(shutil.copyfile(stale, tmp_path / "stale-copy.kgl"), fixture)
        assert _marker_count_on_disk(_snapshot(fixture, tmp_path), "Stale") == 1

        # No client writes at all — only the unconditional first tick can undo
        # this, and it must, because the file no longer matches what is served.
        _wait_until(
            lambda: _marker_count_in_snapshot(fixture, tmp_path, "Stale") == 0,
            timeout=25.0,
            what="the first tick to overwrite the stale file with the served state",
        )
        snapshot = _snapshot(fixture, tmp_path)
        assert kglite.open(str(snapshot)).cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 4, (
            "the rewritten file must be the served state, not a merge of the two"
        )
    finally:
        _teardown_bolt_server(proc)


def test_shutdown_stays_prompt_and_ordered_with_the_interval_armed(tmp_path):
    """The periodic task cannot delay or outlive shutdown: SIGINT still exits
    cleanly and well inside the timeout, and the task is stopped *before* the
    exit save runs — the exit save must be the last write to the file."""
    _require_binary()
    fixture = tmp_path / "interval-shutdown.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=[*INTERVAL_ARGS, "--save-on-exit"])
    try:
        _write_marker(url)
        started = time.monotonic()
        status = _graceful_stop_bolt_server(proc, signal.SIGINT, timeout=20.0)
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:
            _teardown_bolt_server(proc)
    assert status == 0, "an armed checkpoint task must not break the clean stop"
    assert elapsed < 10.0, f"shutdown took {elapsed:.1f}s with the checkpoint task armed"
    logs = proc.stdout.read().decode("utf-8", errors="replace") + proc.stderr.read().decode("utf-8", errors="replace")
    stopped = logs.find("checkpoint-interval: stopped")
    saved = logs.find("save-on-exit complete")
    assert stopped != -1 and saved != -1, logs
    assert stopped < saved, f"the periodic task must be stopped before the exit save runs:\n{logs}"
    assert _marker_count_on_disk(fixture) == 1


# ────────────────────────────────────────────────────────────────────────────
# --checkpoint-interval — startup validation
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["0", "abc", "5s", "-1"], ids=["zero", "junk", "units", "negative"])
def test_checkpoint_interval_rejects_a_value_it_cannot_honour(bolt_binary_path, tmp_path, value):
    """A mistyped interval fails startup rather than starting a server that
    silently never checkpoints.

    `--flag=value` rather than `--flag value` so every case reaches the value
    parser: a bare `-1` in the next argv slot is read by clap as an unknown
    *flag*, which is still a refusal but a different one. The spaced spelling
    is what every other test here spawns with.
    """
    _require_binary()
    fixture = tmp_path / "bad-interval.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0", f"--checkpoint-interval={value}"],
    )
    assert result.returncode != 0, f"--checkpoint-interval {value} must not start a server"
    combined = result.stdout + result.stderr
    assert "--checkpoint-interval" in combined, combined
    if value == "0":
        assert "at least 1 second" in combined, combined


def test_checkpoint_interval_env_mirror_rejects_a_bad_value(bolt_binary_path, tmp_path):
    """The environment spelling gets the same parse — and the error names the
    variable, since the operator never typed a flag."""
    _require_binary()
    fixture = tmp_path / "bad-interval-env.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0"],
        env={"KGLITE_BOLT_CHECKPOINT_INTERVAL": "0"},
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "KGLITE_BOLT_CHECKPOINT_INTERVAL" in combined and "at least 1 second" in combined, combined


def test_checkpoint_interval_conflicts_with_readonly_flag(bolt_binary_path, tmp_path):
    """clap rejects the flag pair before anything is opened."""
    _require_binary()
    fixture = tmp_path / "ro-interval.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0", "--readonly", "--checkpoint-interval", "60"],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "cannot be used with" in combined, combined
    assert "--readonly" in combined and "--checkpoint-interval" in combined, combined


def test_checkpoint_interval_env_mirror_is_refused_with_readonly(bolt_binary_path, tmp_path):
    """clap cannot see the environment, so the server checks it itself."""
    _require_binary()
    fixture = tmp_path / "ro-interval-env.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0", "--readonly"],
        env={"KGLITE_BOLT_CHECKPOINT_INTERVAL": "60"},
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "--readonly" in combined and "nothing to write back" in combined, combined


def test_checkpoint_interval_is_refused_for_a_disk_graph(bolt_binary_path, tmp_path):
    """Disk graphs are excluded for the same unbounded-generation reason
    `--save-on-exit` refuses them — and periodic checkpointing is the shape
    that would grow the directory fastest."""
    _require_binary()
    disk_dir = tmp_path / "disk-interval"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.cypher("CREATE (:Person {id: 1, title: 'Alice'})")
    g.save(str(disk_dir))
    del g

    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(disk_dir), "--port", "0", "--checkpoint-interval", "60"],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "disk-mode" in combined and "generation" in combined, combined


def test_checkpoint_interval_and_the_verb_share_one_skip_state(tmp_path):
    """One recorded version, two routes to it: a `db.checkpoint()` that wrote
    version N makes the next tick a skip, and a tick that wrote version N makes
    the next verb call report the skip. Two independent counters would each
    re-save what the other had just written."""
    _require_binary()
    fixture = tmp_path / "shared-state.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=INTERVAL_ARGS)
    try:
        _write_marker(url)
        # The verb writes first; the tick then finds nothing to do, so the file
        # stays byte-identical across several tick windows.
        assert _checkpoint(url)["message"].startswith("checkpoint written: version ")
        after_verb = _digest(fixture)
        time.sleep(3.5)
        assert _digest(fixture) == after_verb, "a tick must skip what the verb just wrote"

        # Now the other direction: a tick writes the next version, and the verb
        # that follows reports the skip instead of re-saving.
        _write_marker(url, "Later")
        _wait_until(
            lambda: _digest(fixture) != after_verb,
            timeout=25.0,
            what="a tick to write the new version",
        )
        message = _checkpoint(url)["message"]
        assert message.startswith("skipped: graph unchanged since version "), message
    finally:
        _teardown_bolt_server(proc)


# ────────────────────────────────────────────────────────────────────────────
# --durability: the write-ahead log
#
# The three features above bound the loss window by *rewriting the whole
# graph*; the log bounds it per commit. What every test here measures is what
# survives a SIGKILL with **no checkpoint of any kind in between** — the
# baseline test at the top of this file is the control: at `off`, that write is
# gone.
# ────────────────────────────────────────────────────────────────────────────

# magic (4 bytes) + format version (1 byte). A truncated log is exactly this
# long; anything more is at least one frame.
WAL_HEADER_BYTES = 5


def _wal_path(served):
    """The sidecar beside `served`, derived the way the engine derives it."""
    return served.parent / (served.name + "-wal")


def _wal_size(served) -> int:
    path = _wal_path(served)
    return path.stat().st_size if path.exists() else 0


def _count_over_bolt(url: str, title: str) -> int:
    with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            return session.run(_named_count_query(title)).single()["c"]


@pytest.mark.parametrize("level", ["full", "normal"])
def test_a_logged_commit_survives_a_sigkill_with_no_checkpoint(tmp_path, level):
    """The rung's headline. Commit over Bolt, SIGKILL the server — no exit
    hook, no `db.checkpoint()`, no interval — and restart at the same level:
    the write comes back out of the log.

    `normal` is tested against a *process* kill, which is exactly what it
    promises (the frame is in the kernel's page cache). Its OS-crash/power-loss
    window is documented, not tested: nothing in a user-space test can take the
    page cache away.
    """
    _require_binary()
    fixture = tmp_path / f"wal-{level}.kgl"
    _build_bolt_fixture_graph(fixture)
    untouched = _digest(fixture)

    proc, url = _spawn_bolt_server(fixture, extra_args=["--durability", level])
    try:
        _write_marker(url)
        assert _wal_size(fixture) > WAL_HEADER_BYTES, "the commit must reach the log"
    finally:
        _teardown_bolt_server(proc)  # SIGKILL: no shutdown path, no exit save

    assert _digest(fixture) == untouched, (
        "no checkpoint ran, so the .kgl must be byte-identical — otherwise this test would prove nothing about the log"
    )

    restarted, restarted_url = _spawn_bolt_server(fixture, extra_args=["--durability", level])
    try:
        assert _count_over_bolt(restarted_url, "Zed") == 1, (
            f"--durability {level} must replay the committed write at startup"
        )
    finally:
        _teardown_bolt_server(restarted)


def test_the_env_mirror_logs_the_same_way(tmp_path):
    """`KGLITE_BOLT_DURABILITY=full` does what the flag does — the spelling a
    Compose file or unit file can set."""
    _require_binary()
    fixture = tmp_path / "wal-env.kgl"
    _build_bolt_fixture_graph(fixture)
    env = {"KGLITE_BOLT_DURABILITY": "full"}

    proc, url = _spawn_bolt_server(fixture, env=env)
    try:
        _write_marker(url)
    finally:
        _teardown_bolt_server(proc)

    restarted, restarted_url = _spawn_bolt_server(fixture, env=env)
    try:
        assert _count_over_bolt(restarted_url, "Zed") == 1
    finally:
        _teardown_bolt_server(restarted)


def test_without_the_log_the_same_commit_is_lost(tmp_path):
    """The control for the two tests above, and the pinned default: `off` — the
    level an invocation with no `--durability` gets — loses a committed write to
    a SIGKILL, because nothing but a checkpoint ever writes the graph back."""
    _require_binary()
    fixture = tmp_path / "wal-off.kgl"
    _build_bolt_fixture_graph(fixture)

    proc, url = _spawn_bolt_server(fixture, extra_args=["--durability", "off"])
    try:
        _write_marker(url)
        assert _wal_size(fixture) == 0, "level off must not create a log at all"
    finally:
        _teardown_bolt_server(proc)

    restarted, restarted_url = _spawn_bolt_server(fixture, extra_args=["--durability", "off"])
    try:
        assert _count_over_bolt(restarted_url, "Zed") == 0
    finally:
        _teardown_bolt_server(restarted)


def test_a_default_server_does_not_log(tmp_path):
    """The default is `off`: an invocation that says nothing about durability
    behaves exactly as it did before the flag existed. Pinned separately from
    the `off` test above because flipping the default is a decision, not an
    implementation detail — see `DEFAULT_DURABILITY` in main.rs."""
    _require_binary()
    fixture = tmp_path / "wal-default.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture)
    try:
        _write_marker(url)
        assert _wal_size(fixture) == 0, "the default must not attach a log"
    finally:
        _teardown_bolt_server(proc)
    assert _marker_count_on_disk(fixture) == 0


# ── Recovery on open is unconditional ───────────────────────────────────────


def test_starting_at_off_over_an_unreplayed_log_is_refused(tmp_path, bolt_binary_path):
    """A logged commit that no checkpoint has folded in is *data*, and starting
    a server that neither replays nor keeps it would lose it at the next
    checkpoint. Both spellings of "no log" are refused: the explicit `off` and
    the default."""
    _require_binary()
    fixture = tmp_path / "unreplayed.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=["--durability", "full"])
    try:
        _write_marker(url)
    finally:
        _teardown_bolt_server(proc)

    for args in (["--durability", "off"], []):
        result = _run_server_binary(
            bolt_binary_path,
            ["--graph", str(fixture), "--port", "0", *args],
        )
        assert result.returncode != 0, f"args={args} must refuse: {result.stdout}{result.stderr}"
        combined = result.stdout + result.stderr
        assert "-wal" in combined, combined
        assert "'full' or 'normal'" in combined, combined

    # Non-vacuity: the refusal is about the level, not a broken file — the same
    # path serves fine at a logging level, with the write.
    recovered, recovered_url = _spawn_bolt_server(fixture, extra_args=["--durability", "full"])
    try:
        assert _count_over_bolt(recovered_url, "Zed") == 1
    finally:
        _teardown_bolt_server(recovered)


# ── Checkpoints and the log ─────────────────────────────────────────────────


def test_a_checkpoint_truncates_the_log_and_later_commits_start_a_fresh_one(tmp_path):
    """The four-step checkpoint order, observed from outside: flush → stamp →
    write the `.kgl` → truncate the log. After `db.checkpoint()` the file holds
    the write and the log is back to its header; the next commit lands in the
    fresh log and is itself recovered by a restart."""
    _require_binary()
    fixture = tmp_path / "checkpoint-wal.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=["--durability", "full"])
    try:
        _write_marker(url, "Before", node_id=101)
        assert _wal_size(fixture) > WAL_HEADER_BYTES

        assert _checkpoint(url)["success"] is True
        assert _wal_size(fixture) == WAL_HEADER_BYTES, "a checkpoint folds the log into the .kgl and truncates it"
        # The truncated sidecar is also what makes the file plainly openable
        # again: nothing is left in front of the checkpoint.
        assert _marker_count_on_disk(_snapshot(fixture, tmp_path), "Before") == 1

        _write_marker(url, "After", node_id=102)
        assert _wal_size(fixture) > WAL_HEADER_BYTES, "post-checkpoint commits use the fresh log"
    finally:
        _teardown_bolt_server(proc)  # SIGKILL: only the log can carry "After"

    restarted, restarted_url = _spawn_bolt_server(fixture, extra_args=["--durability", "full"])
    try:
        assert _count_over_bolt(restarted_url, "Before") == 1, "the checkpointed write"
        assert _count_over_bolt(restarted_url, "After") == 1, (
            "with a log, the post-checkpoint write is no longer the loss window"
        )
    finally:
        _teardown_bolt_server(restarted)


def test_save_on_exit_flushes_and_truncates_the_log(tmp_path):
    """A graceful stop under a log: the exit path flushes the log, the exit
    save folds it into the `.kgl`, and the log is left truncated — so the next
    start needs no recovery and the graph opens with no durability argument at
    all."""
    _require_binary()
    fixture = tmp_path / "exit-save-wal.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=["--durability", "normal", "--save-on-exit"])
    try:
        _write_marker(url)
        status = _graceful_stop_bolt_server(proc, signal.SIGTERM)
    finally:
        if proc.poll() is None:
            _teardown_bolt_server(proc)
    assert status == 0, "a successful flush + exit save exits zero"
    logs = proc.stdout.read().decode("utf-8", errors="replace") + proc.stderr.read().decode("utf-8", errors="replace")
    assert "write-ahead log flushed" in logs, logs
    assert _wal_size(fixture) == WAL_HEADER_BYTES
    assert _marker_count_on_disk(fixture) == 1


# ── Refusal matrix ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", ["full", "normal"])
def test_a_logging_level_is_refused_with_readonly(bolt_binary_path, tmp_path, level):
    """A read-only server never commits, so a log would only ever be empty."""
    _require_binary()
    fixture = tmp_path / "ro-wal.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0", "--readonly", "--durability", level],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "--readonly" in combined and "nothing to log" in combined, combined


def test_the_env_mirror_is_refused_with_readonly(bolt_binary_path, tmp_path):
    """The environment spelling too — the level is what conflicts, so a clap
    `conflicts_with` on the argument could not express this rule anyway."""
    _require_binary()
    fixture = tmp_path / "ro-wal-env.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0", "--readonly"],
        env={"KGLITE_BOLT_DURABILITY": "full"},
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "--readonly" in combined and "KGLITE_BOLT_DURABILITY" in combined, combined


def test_readonly_serves_fine_at_off(tmp_path):
    """The other side of that rule: `off` is not a conflict, it is what every
    read-only server already runs at."""
    _require_binary()
    fixture = tmp_path / "ro-off.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, readonly=True, extra_args=["--durability", "off"])
    try:
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                assert session.run("MATCH (p:Person) RETURN count(p) AS c").single()["c"] == 4
    finally:
        _teardown_bolt_server(proc)


def test_durability_is_refused_for_a_disk_graph(bolt_binary_path, tmp_path):
    """A disk graph commits by publishing an immutable generation, so it keeps
    no logical log at any level. The refusal is the engine's; what matters here
    is that it reaches the operator as a clean startup error."""
    _require_binary()
    disk_dir = tmp_path / "disk-graph"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.cypher("CREATE (:Person {id: 1, title: 'Alice'})")
    g.save(str(disk_dir))
    del g

    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(disk_dir), "--port", "0", "--durability", "full"],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "storage='disk'" in combined, combined
    assert "panicked" not in combined.lower(), combined


@pytest.mark.parametrize("value", ["sometimes", "1", "true", "fsync", ""])
def test_an_unknown_durability_level_is_a_startup_error(bolt_binary_path, tmp_path, value):
    """A mistyped level fails startup rather than quietly logging nothing —
    the same treatment `--checkpoint-interval` gives a malformed number."""
    _require_binary()
    fixture = tmp_path / "bad-level.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0", "--durability", value],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "full" in combined and "normal" in combined and "off" in combined, combined


def test_an_unknown_env_durability_level_is_a_startup_error(bolt_binary_path, tmp_path):
    _require_binary()
    fixture = tmp_path / "bad-level-env.kgl"
    _build_bolt_fixture_graph(fixture)
    result = _run_server_binary(
        bolt_binary_path,
        ["--graph", str(fixture), "--port", "0"],
        env={"KGLITE_BOLT_DURABILITY": "sometimes"},
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "KGLITE_BOLT_DURABILITY" in combined, combined


def test_a_storage_conversion_keeps_the_log_recoverable(tmp_path):
    """`--storage` converts the graph *in memory*, before the session exists, so
    the `.kgl` on disk — and the `checkpoint_lsn` the replay is gated on — is
    untouched until the first checkpoint. A converted server therefore logs and
    recovers exactly like an unconverted one, and the checkpoint writes the
    converted mode and truncates the log together."""
    _require_binary()
    fixture = tmp_path / "converted.kgl"
    _build_bolt_fixture_graph(fixture)  # saved in memory mode
    args = ["--storage", "mapped", "--durability", "full"]

    proc, url = _spawn_bolt_server(fixture, extra_args=args)
    try:
        _write_marker(url)
        assert _wal_size(fixture) > WAL_HEADER_BYTES
    finally:
        _teardown_bolt_server(proc)  # SIGKILL: only the log carries the write

    restarted, restarted_url = _spawn_bolt_server(fixture, extra_args=args)
    try:
        assert _count_over_bolt(restarted_url, "Zed") == 1
        assert _checkpoint(restarted_url)["success"] is True
    finally:
        _teardown_bolt_server(restarted)
    assert _wal_size(fixture) == WAL_HEADER_BYTES
    assert _marker_count_on_disk(fixture) == 1
    assert kglite.open(str(fixture)).graph_info()["storage_mode"] == "mapped", (
        "the checkpoint must record the mode the operator converted to"
    )
