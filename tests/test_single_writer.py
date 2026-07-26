"""Cross-process single-writer guard for ``kglite.open``.

``open()`` is the write-back entry point — it remembers its path, logs to a
WAL by default, and checkpoints on close — so two processes holding the same
path both produce full snapshots at ``save()``, and whichever publishes last
wins outright. Before the guard, both exited 0 and one writer's work vanished
with no error, no warning, and nothing in the file to say it had happened.

The guard is an advisory OS file lock on a ``<path>.lock`` sidecar, taken at
``open()`` and held until ``close()`` / ``__exit__`` / drop. Three properties
are load-bearing and each has a test here:

- **It blocks a second writer, loudly**, naming the holding pid — an error a
  reader of the traceback can act on without guessing.
- **A crashed holder does not brick the graph.** The lock is owned by the OS,
  not by the file's existence, so ``SIGKILL`` releases it even though the
  sidecar stays on disk. If this ever regressed, users would learn to delete
  lock files by reflex and the guard would be worth nothing.
- **Readers are never blocked.** ``load()`` takes no lease, so read replicas
  and analytics jobs keep working alongside a writer.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

import kglite

PYBIN = sys.executable

#: See ``tests/test_durability.py`` — ``SIGKILL`` is POSIX-only, and a crash
#: test that silently degrades into a graceful-shutdown test on Windows is
#: worse than one that does not run. The crashed-holder case is the whole
#: reason this guard is safe to ship, so it must not be faked.
requires_sigkill = pytest.mark.skipif(
    not hasattr(signal, "SIGKILL"),
    reason="SIGKILL is POSIX-only; this is a crash test and must not degrade into a graceful-shutdown test on Windows",
)

#: Long enough that the child is unambiguously still holding when the parent
#: makes its assertions, short enough that a leaked child cannot outlive the
#: suite's 120 s per-test ceiling.
HOLD_SECONDS = 30


def _seed(path: str) -> None:
    """Create *path* as a real saved graph, releasing the lease afterwards."""
    graph = kglite.open(path)
    graph.cypher("CREATE (:Task {name: 'seed'})")
    graph.save()
    graph.close()


def _write_script(directory: str, name: str, body: str) -> str:
    script = os.path.join(directory, name)
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(body))
    return script


def _spawn_holder(directory: str, db: str) -> tuple[subprocess.Popen, str]:
    """Start a child that opens *db* for writing and holds it.

    Returns once the child has signalled that it owns the lease, so the
    parent's assertions race nothing.
    """
    ready = os.path.join(directory, "ready")
    script = _write_script(
        directory,
        "holder.py",
        """
        import sys, time, kglite
        graph = kglite.open(sys.argv[1])
        with open(sys.argv[2], "w") as handle:
            handle.write("ready")
        time.sleep(int(sys.argv[3]))
        """,
    )
    child = subprocess.Popen([PYBIN, script, db, ready, str(HOLD_SECONDS)])
    deadline = time.monotonic() + 30
    while not os.path.exists(ready) and time.monotonic() < deadline:
        if child.poll() is not None:
            raise AssertionError(f"holder exited early with {child.returncode}")
        time.sleep(0.02)
    assert os.path.exists(ready), "holder never acquired the writer lease"
    return child, ready


def _reap(child: subprocess.Popen) -> None:
    if child.poll() is None:
        child.kill()
    child.wait()


def test_second_process_open_fails_instead_of_losing_writes(tmp_path):
    """The audit's scenario: two processes, 20 writes each, 40 expected.

    Before the guard both children exited 0 and the reopened graph held 20 —
    one writer's work gone with no signal. The fix does not merge the two
    writers; it makes the loser *fail loudly* instead of failing silently,
    which is the difference between a bug report and silent data loss.
    """
    db = str(tmp_path / "app.kgl")
    _seed(db)

    script = _write_script(
        str(tmp_path),
        "writer.py",
        """
        import sys, kglite
        db, tag, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
        graph = kglite.open(db)
        for index in range(count):
            graph.cypher("CREATE (:Task {name: $nm})", params={"nm": f"{tag}-{index}"})
        graph.save()
        graph.close()
        """,
    )
    first = subprocess.Popen([PYBIN, script, db, "A", "20"], stderr=subprocess.PIPE, text=True)
    second = subprocess.Popen([PYBIN, script, db, "B", "20"], stderr=subprocess.PIPE, text=True)
    _, first_err = first.communicate()
    _, second_err = second.communicate()

    codes = sorted([first.returncode, second.returncode])
    assert codes == [0, 1], (
        f"exactly one writer must win and the other must fail; got {codes} (this is the silent-loss regression)"
    )
    loser_err = first_err if first.returncode else second_err
    assert "is open for writing by pid" in loser_err, loser_err

    # The winner's 20 rows are intact and the loser wrote nothing — no
    # partial interleaving, and no snapshot built on a stale read.
    reopened = kglite.load(db)
    total = reopened.cypher("MATCH (t:Task) RETURN count(t) AS c").to_dicts()[0]["c"]
    assert total == 21, f"expected seed + one writer's 20 rows, got {total}"


def test_blocked_open_names_the_holding_process(tmp_path):
    """``KgError: app.kgl is open for writing by pid 4711`` — the message the
    audit asked for. A pid the operator can look up is the whole difference
    between an actionable error and a mystery."""
    db = str(tmp_path / "app.kgl")
    _seed(db)
    child, _ = _spawn_holder(str(tmp_path), db)
    try:
        with pytest.raises(Exception) as caught:
            kglite.open(db)
        message = str(caught.value)
        assert f"pid {child.pid}" in message, message
        assert os.path.basename(db) in message, message
        # A timestamp turns "some process has it" into "and it has had it
        # since 09:15", which is what distinguishes a live writer from a
        # forgotten one.
        assert "since" in message, message
    finally:
        _reap(child)


def test_reader_is_never_blocked_by_a_writer(tmp_path):
    """``load()`` takes no lease.

    The durability unit is ``save()``, which republishes the whole file, so a
    reader either sees the previous consistent snapshot or the next one. Making
    reads exclusive would break read-replica and analytics deployments to fix a
    problem that only writers have.
    """
    db = str(tmp_path / "app.kgl")
    _seed(db)
    child, _ = _spawn_holder(str(tmp_path), db)
    try:
        reader = kglite.load(db)
        rows = reader.cypher("MATCH (t:Task) RETURN count(t) AS c").to_dicts()
        assert rows[0]["c"] == 1
        # A second concurrent reader is fine too — readers do not contend
        # with each other any more than they contend with the writer.
        assert kglite.load(db).cypher("MATCH (t:Task) RETURN count(t) AS c").to_dicts()
    finally:
        _reap(child)


@requires_sigkill
def test_sigkill_holder_releases_the_lease(tmp_path):
    """A crashed writer must not brick the graph.

    ``SIGKILL`` is uncatchable, so no Rust ``Drop``, no ``atexit``, no
    interpreter teardown runs — the sidecar lock file is left on disk exactly
    as a crash leaves it. The lock is nevertheless released, because it is an
    OS file-descriptor lock rather than a claim staked by the file's presence.
    This is asserted, not assumed: if it regressed, the guard would turn every
    crash into a permanent outage.
    """
    db = str(tmp_path / "app.kgl")
    _seed(db)
    child, _ = _spawn_holder(str(tmp_path), db)
    lock_file = db + ".lock"

    with pytest.raises(Exception):
        kglite.open(db)

    os.kill(child.pid, signal.SIGKILL)
    assert child.wait() == -signal.SIGKILL, "child must have died on SIGKILL"

    # The stale file survives the crash. That is expected and harmless, and
    # is precisely why the error message tells users not to delete it.
    assert os.path.exists(lock_file), "lock sidecar is persistent by design"

    recovered = kglite.open(db)
    recovered.cypher("CREATE (:Task {name: 'after-crash'})")
    recovered.close()

    reopened = kglite.load(db)
    total = reopened.cypher("MATCH (t:Task) RETURN count(t) AS c").to_dicts()[0]["c"]
    assert total == 2


def test_close_releases_so_the_path_can_be_reopened(tmp_path):
    """Two sequential ``with`` blocks over one path, in one process.

    The graph object stays alive and bound after its ``with`` block, so if the
    lease were held until garbage collection this everyday pattern would
    deadlock against itself.
    """
    db = str(tmp_path / "app.kgl")
    with kglite.open(db) as graph:
        graph.cypher("CREATE (:Task {name: 'first'})")
    with kglite.open(db) as graph:
        graph.cypher("CREATE (:Task {name: 'second'})")

    reopened = kglite.load(db)
    total = reopened.cypher("MATCH (t:Task) RETURN count(t) AS c").to_dicts()[0]["c"]
    assert total == 2


def test_second_open_in_one_process_reports_itself(tmp_path):
    """Overlapping opens inside one process are the same lost-update bug, and
    are blocked the same way — but the message says so, rather than pointing at
    a phantom "other process" with the caller's own pid."""
    db = str(tmp_path / "app.kgl")
    _seed(db)
    held = kglite.open(db)
    try:
        with pytest.raises(Exception) as caught:
            kglite.open(db)
        assert "this same process" in str(caught.value), str(caught.value)
        assert str(os.getpid()) in str(caught.value)
    finally:
        held.close()


def test_lock_false_opts_out(tmp_path):
    """The documented escape hatch, for callers coordinating externally.

    Explicit and never the default — the point is that opting out of the guard
    should be something a reviewer can see in the diff.
    """
    db = str(tmp_path / "app.kgl")
    _seed(db)
    first = kglite.open(db, lock=False)
    second = kglite.open(db, lock=False)
    assert first is not second
    first.close()
    second.close()


def test_lock_false_neither_takes_nor_checks_the_lease(tmp_path):
    """The escape hatch is a full opt-out, and is documented as one.

    ``lock=False`` does not consult the lease either, so it will happily open
    alongside a live holder — including one in another process. That is the
    whole point (an external supervisor may be doing the coordinating), and it
    is also exactly why it must stay explicit: a caller that reaches for it
    without a coordination story has re-armed the original data-loss bug.
    """
    db = str(tmp_path / "app.kgl")
    _seed(db)
    child, _ = _spawn_holder(str(tmp_path), db)
    try:
        # A guarded open is refused...
        with pytest.raises(Exception):
            kglite.open(db)
        # ...and an explicitly unguarded one is not.
        bypass = kglite.open(db, lock=False)
        assert bypass.cypher("MATCH (t:Task) RETURN count(t) AS c").to_dicts()[0]["c"] == 1
        bypass.close()
    finally:
        _reap(child)


def test_lock_false_still_allows_a_guarded_writer(tmp_path):
    """An unguarded opener must not leave a stale lease behind that blocks the
    next honest writer."""
    db = str(tmp_path / "app.kgl")
    _seed(db)
    unguarded = kglite.open(db, lock=False)
    unguarded.close()
    guarded = kglite.open(db)
    guarded.close()


def test_disk_mode_is_guarded_at_open(tmp_path):
    """Disk graphs previously failed only at the first mutation, with a raw
    ``Resource temporarily unavailable`` errno. They now fail at ``open()``
    with the same named message as every other storage mode."""
    db = str(tmp_path / "diskgraph")
    seed = kglite.open(db, storage="disk")
    seed.cypher("CREATE (:Task {name: 'seed'})")
    seed.save()
    seed.close()
    del seed

    held = kglite.open(db, storage="disk")
    try:
        with pytest.raises(Exception) as caught:
            kglite.open(db, storage="disk")
        assert "is open for writing by" in str(caught.value), str(caught.value)
    finally:
        held.close()
