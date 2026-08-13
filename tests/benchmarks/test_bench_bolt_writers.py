"""Contended-writer load test for kglite-bolt-server (Track G, G4a).

Answers the only question the write path can actually be asked: all Bolt
writes are *explicit* transactions (auto-commit mutations are rejected at
`backend.rs`), so what matters is how committed throughput, conflict rate,
and latency move as N driver-managed writers contend for the single graph.

Two sweeps live here: `test_contended_writer_sweep` varies the writer count
at one write per transaction, and `test_batch_size_sweep` varies the writes
per transaction at a fixed writer count — the measurement behind the
operator page's "batch more work into each transaction rather than adding
writers". A third cell, `test_checkpoint_under_contention`, holds the
writer count fixed and adds a `CALL db.checkpoint()` thread — the price of
durability under load, since a checkpoint pauses committing writers. A
fourth, `test_durability_sweep`, spawns the server at each `--durability`
level and prices the write-ahead log: `full` puts a device barrier inside
the session lock every commit already serializes on, so it is the level
whose cost decides which one ships as the default.

The unit of work runs through `session.execute_write`, i.e. the
driver-managed path that retries a `Neo.TransientError.*` failure by
itself — the system a caller actually gets after the conflict code became
retriable. A per-worker attempt counter is incremented *inside* the unit of
work, so `attempts - committed` is the number of driver retries.

Design notes that matter for reading the numbers:

- **Fixed wall-clock window per cell.** Committed throughput is the metric;
  a fixed op count would let a heavily-retried cell run far longer and turn
  throughput into a function of the slowest worker.
- **Committed-in-window is the throughput cell, not committed/elapsed.**
  Each worker keeps running its current transaction after the window closes,
  so the harness's `elapsed` carries a drain tail whose length depends on
  how slow that last transaction was — at `--durability full` the tail is a
  whole fsync. Counting the commits that *completed inside* the fixed window
  divides by a constant instead, which is why the recorded rows use it.
- **Fresh server + fresh copy of the fixture per cell**, so every cell
  starts from an identical 10k-node graph. A write transaction materializes
  a working graph, so starting size is part of the cost being measured.
- **Fast driver backoff.** With the driver's default 1 s initial retry
  delay, every conflicting writer sleeps a second and the curve measures
  the client's backoff policy rather than the server. The retry delays are
  tightened (see `_RETRY_KW`) and recorded alongside the numbers.
- **Nothing about the numbers is asserted.** Only sanity invariants are:
  every committed write is present in the graph afterwards, no non-conflict
  error escaped, throughput > 0. Thresholds here would be flaky; the
  recorded curve lives in `dev-docs/bench/results/results.csv` and the
  qualitative contract in `docs/operators/bolt-server.md`.

Run with:

    uv run --no-sync pytest tests/benchmarks/test_bench_bolt_writers.py \
        -m "benchmark and bolt_stress" -q -s

Recorded numbers require a **release**-built `kglite-bolt-server`
(`cargo build -p kglite-bolt-server --release`); rebuild debug afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import threading
import time

import pytest

from tests.conftest import (
    _BOLT_BINARY,
    _spawn_bolt_server,
    _teardown_bolt_server,
)

neo4j = pytest.importorskip("neo4j")

pytestmark = [pytest.mark.benchmark, pytest.mark.bolt_stress]

#: Binary every cell spawns. Defaults to the shared newest-of-profile
#: resolution; setting `KGLITE_BENCH_BOLT_BINARY` pins one explicit
#: executable for the whole run. Recorded numbers use the pinned form with
#: a *copy* of the release build, because a concurrent `cargo build` of the
#: other profile would otherwise flip newest-of-profile mid-run and the
#: recorded row could not say which profile produced it.
_PINNED_BINARY = os.environ.get("KGLITE_BENCH_BOLT_BINARY")
_BENCH_BINARY = Path(_PINNED_BINARY) if _PINNED_BINARY else _BOLT_BINARY


#: OCC conflict status code. Classification is by code — never by exception
#: class (the conflict already moved class once) and never by message.
OCC_CONFLICT_CODE = "Neo.TransientError.Transaction.Outdated"

#: Writer counts swept.
WRITER_COUNTS = (1, 2, 4, 8)

#: Batch sweep: (writers, creates-per-transaction) cells. N=4 is the
#: contended series the operator page's batching advice is about; the two
#: N=1 cells are the uncontended control that separates "batching amortizes
#: per-transaction cost" from "batching wins because it dodges conflicts".
BATCH_CELLS = ((4, 1), (4, 10), (4, 100), (1, 1), (1, 100))

#: Measurement window per cell, seconds.
CELL_SECONDS = 5.0

#: Durability sweep: `(--durability level, writer count)` cells. N=1 is the
#: uncontended control — it isolates the log's per-commit cost from what the
#: session lock does with it once writers queue behind each other. Ordered
#: level-minor so a machine-drift trend shows up as a within-N pattern.
DURABILITY_CELLS = tuple((level, n) for n in (1, 4) for level in ("off", "normal", "full"))

#: Checkpoint-under-contention cell: writers, and how often the checkpoint
#: thread fires. N=4 is the contended series the batch sweep is about; ~1 s
#: over a 5 s window gives four calls, each landing on a graph with writes
#: since the last one (so the digest-skip never answers).
CHECKPOINT_WRITERS = 4
CHECKPOINT_PERIOD_S = 1.0

#: Sham-thread control for the checkpoint cell. The checkpointed arm
#: measured *faster* than its baseline; a third arm whose extra session
#: issues a cheap read at the same period separates "saving consolidates
#: something" from "an extra session on this driver changes the load
#: pattern" — if the read arm speeds up too, the speed-up is the thread,
#: not the checkpoint.
READ_PROBE_PERIOD_S = CHECKPOINT_PERIOD_S
READ_PROBE_QUERY = "RETURN 1 AS one"

#: The level every cell *outside* the durability sweep pins explicitly.
#:
#: Those three sweeps are the write-path curve, and their rows in
#: `dev-docs/bench/results/results.csv` go back to before the write-ahead log
#: existed. The server's default is now `normal`, so spawning without the flag
#: would silently re-point the cells at a different system and make the old
#: rows incomparable. What logging costs is `test_durability_sweep`'s question,
#: and it answers it against this same `off` reference.
RECORDED_LEVEL = "off"

#: Warmup cell (discarded): pays first-write / page-cache / JIT-ish costs.
WARMUP_WRITERS = 2
WARMUP_SECONDS = 1.5

#: Driver retry policy for every cell. Recorded with the results: the
#: default (1 s initial delay, doubling, 30 s budget) makes the client's
#: sleep the dominant term at N=8.
_RETRY_KW = dict(
    max_transaction_retry_time=30.0,
    initial_retry_delay=0.05,
    retry_delay_multiplier=1.5,
    retry_delay_jitter_factor=0.2,
)


@dataclass
class CellResult:
    writers: int
    committed: int
    attempts: int
    elapsed_s: float
    latencies_ms: list[float] = field(default_factory=list)
    landed: int = 0
    hard_errors: list[str] = field(default_factory=list)
    exhausted_conflicts: int = 0
    #: Most attempts any single unit of work needed. This is what
    #: attributes a multi-second worst-case latency: a starved writer
    #: losing N conflict rounds pays the driver's compounding backoff,
    #: which is a client-policy cost, not a server stall.
    max_op_attempts: int = 1
    #: Attempts made by the single slowest committed unit of work.
    slowest_op_attempts: int = 1
    #: CREATEs per committed transaction (the unit of work's batch size).
    batch: int = 1
    #: `CALL db.checkpoint()` round-trip times, ms — one per call the
    #: checkpoint thread made (empty in an un-checkpointed cell). This is
    #: the writer-pause upper bound: `Session::save` holds the session lock
    #: for its whole duration, so a contended writer waits at most this.
    checkpoint_ms: list[float] = field(default_factory=list)
    #: Calls that wrote the file vs. calls the digest-skip answered without
    #: writing, classified from the verb's `message` column.
    checkpoints_written: int = 0
    checkpoints_skipped: int = 0
    #: Anything the checkpoint thread saw that was neither: a raised
    #: exception or an unrecognized message. Asserted empty.
    checkpoint_errors: list[str] = field(default_factory=list)
    #: `--durability` level the server was spawned at ("off" when the flag
    #: was not passed, i.e. whatever the binary's default is).
    durability: str = "off"
    #: `perf_counter` stamp of every committed unit of work, used to count
    #: the commits that landed inside the fixed window (see the module
    #: docstring) rather than dividing by a drain-inflated elapsed.
    commit_times: list[float] = field(default_factory=list)
    #: End of the fixed measurement window, on the same clock.
    window_end: float = 0.0
    #: Length of that window, seconds — the fixed denominator.
    window_s: float = 0.0
    #: Size of the `<graph>-wal` sidecar when the window closed, bytes. 0
    #: when the level keeps no log (the file does not exist).
    wal_bytes: int = 0
    #: Round-trip times of the sham read-probe thread, ms (empty unless the
    #: read-probe arm ran), and anything it raised.
    probe_ms: list[float] = field(default_factory=list)
    probe_errors: list[str] = field(default_factory=list)

    @property
    def committed_per_s(self) -> float:
        return self.committed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def committed_in_window(self) -> int:
        """Commits that completed before the window closed — the metric.

        See the module docstring: `committed / elapsed` charges a cell for
        its own drain tail, which is a whole fsync at `--durability full`.
        """
        return sum(1 for t in self.commit_times if t <= self.window_end)

    @property
    def window_per_s(self) -> float:
        return self.committed_in_window / self.window_s if self.window_s > 0 else 0.0

    @property
    def wal_bytes_per_commit(self) -> float:
        """Sidecar bytes per committed transaction — the growth-rate datum.

        Paired with the cell's *total* commits, not the in-window count:
        the size is read after the writers drain, so every committed
        transaction contributed a frame to it.
        """
        return self.wal_bytes / self.committed if self.committed else 0.0

    @property
    def ops_per_s(self) -> float:
        """Committed *node creations* per second — the decision metric.

        `committed_per_s` counts transactions, which is the wrong unit for
        comparing batch sizes: a K=100 cell that commits a tenth as often
        still does ten times the work.
        """
        return self.committed_per_s * self.batch

    @property
    def retries(self) -> int:
        return self.attempts - self.committed

    @property
    def retry_rate(self) -> float:
        return self.retries / self.attempts if self.attempts else 0.0

    @property
    def mean_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile — no interpolation, no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


@pytest.fixture(scope="module")
def _writer_fixture(tmp_path_factory):
    """Build the 10k-node graph once; yield the path each cell copies."""
    if not _BENCH_BINARY.exists():
        pytest.skip(f"bolt-server binary not built at {_BENCH_BINARY}")
    import pandas as pd

    import kglite

    tmp = tmp_path_factory.mktemp("bolt_writers")
    fixture_path = tmp / "writers.kgl"

    n = 10_000
    g = kglite.KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "pid": list(range(n)),
            "name": [f"P{i}" for i in range(n)],
            "age": [20 + (i % 60) for i in range(n)],
            "city": [f"city_{i % 100}" for i in range(n)],
        }
    )
    g.add_nodes(nodes, "Person", "pid", "name")
    edges = pd.DataFrame(
        {
            "s": [i % n for i in range(3 * n)],
            "d": [(i * 13 + 7) % n for i in range(3 * n)],
        }
    )
    g.add_connections(edges, "KNOWS", "Person", "s", "Person", "d")
    g.save(str(fixture_path))
    return fixture_path


#: Batched unit of work. One query, one server round-trip, `$rows` rows —
#: the idiomatic way a driver user "batches more work into a transaction",
#: and the shape used at *every* K including K=1 so that batch size is the
#: only thing that varies across the sweep.
_BATCH_QUERY = "UNWIND $rows AS row CREATE (:Person {id: row.id, title: row.title, city: row.city})"


def _run_cell(
    fixture_path,
    tmp_dir,
    writers: int,
    seconds: float,
    tag: str,
    batch: int | None = None,
    checkpoint_every: float | None = None,
    durability: str | None = None,
    read_probe_every: float | None = None,
) -> CellResult:
    """One cell: `writers` threads hammering `execute_write` for `seconds`.

    Each thread owns its own session and a disjoint id range, so the final
    graph count is an exact oracle for "every committed write landed".

    `batch` selects the unit of work. `None` (the writer-count sweep) is a
    bare single-node `CREATE`. An integer K runs `_BATCH_QUERY` with K rows,
    including at K=1 — so the batch sweep's cells differ *only* in K, and
    the K=1 cell carries the same `UNWIND`/parameter overhead as K=100.

    `checkpoint_every`, when set, adds one more thread on its own driver
    session firing `CALL db.checkpoint()` at that period for the whole
    window — the durability-under-load arm. It is deliberately *not* in the
    start barrier: the writers' window must not wait on it, and its first
    call lands one period into the window with writes already in flight.
    The verb writes the served path, which is this cell's private copy.

    `durability` spawns the server with `--durability <level>`; `None`
    passes no flag, so the cell measures whatever the binary defaults to.
    `read_probe_every` is the sham-thread control for `checkpoint_every`:
    the same extra session at the same period running a trivial read.
    """
    rows_per_tx = 1 if batch is None else batch
    cell_graph = tmp_dir / f"cell_{tag}.kgl"
    shutil.copy(fixture_path, cell_graph)
    wal_file = Path(f"{cell_graph}-wal")

    stop = threading.Event()
    start_barrier = threading.Barrier(writers + 1, timeout=60)
    lock = threading.Lock()
    result = CellResult(
        writers=writers,
        committed=0,
        attempts=0,
        elapsed_s=0.0,
        batch=rows_per_tx,
        durability=durability or "default",
        window_s=seconds,
    )

    def worker(driver, worker_id: int) -> None:
        attempts = 0
        committed = 0
        latencies: list[float] = []
        commit_times: list[float] = []
        hard: list[str] = []
        exhausted = 0
        max_op_attempts = 1
        slowest_op_attempts = 1
        slowest_ms = -1.0
        # Wide enough that a fast K=100 cell cannot walk into the next
        # worker's ids and break the exact landed-count oracle.
        base_id = 1_000_000 + worker_id * 100_000_000
        seq = 0
        try:
            with driver.session() as session:
                start_barrier.wait()
                while not stop.is_set():
                    node_id = base_id + seq * rows_per_tx
                    seq += 1
                    # Built once per unit of work, not once per attempt: a
                    # retry should re-run the *transaction*, not re-pay a
                    # client-side list build the timing would then charge
                    # to the server.
                    rows = (
                        None
                        if batch is None
                        else [
                            {
                                "id": node_id + j,
                                "title": f"{tag}-w{worker_id}-{node_id + j}",
                                "city": "loadtest",
                            }
                            for j in range(batch)
                        ]
                    )

                    def unit_of_work(tx, node_id=node_id, worker_id=worker_id, rows=rows):
                        nonlocal attempts
                        attempts += 1
                        if batch is None:
                            tx.run(
                                "CREATE (:Person {id: $id, title: $title, city: $city})",
                                id=node_id,
                                title=f"{tag}-w{worker_id}-{node_id}",
                                city="loadtest",
                            ).consume()
                            return
                        tx.run(_BATCH_QUERY, rows=rows).consume()

                    attempts_before = attempts
                    t0 = time.perf_counter()
                    try:
                        session.execute_write(unit_of_work)
                    except Exception as e:  # noqa: BLE001
                        # Classify by status code, never by class or message.
                        if getattr(e, "code", None) == OCC_CONFLICT_CODE:
                            # Retry budget exhausted while still conflicting.
                            exhausted += 1
                            continue
                        hard.append(repr(e))
                        break
                    t_end = time.perf_counter()
                    elapsed_ms = (t_end - t0) * 1000.0
                    op_attempts = attempts - attempts_before
                    latencies.append(elapsed_ms)
                    commit_times.append(t_end)
                    committed += 1
                    max_op_attempts = max(max_op_attempts, op_attempts)
                    if elapsed_ms > slowest_ms:
                        slowest_ms = elapsed_ms
                        slowest_op_attempts = op_attempts
        except threading.BrokenBarrierError as e:  # noqa: BLE001
            hard.append(f"barrier: {e!r}")
        except Exception as e:  # noqa: BLE001
            hard.append(f"session: {e!r}")
        with lock:
            result.attempts += attempts
            result.committed += committed
            result.latencies_ms.extend(latencies)
            result.commit_times.extend(commit_times)
            result.hard_errors.extend(hard)
            result.exhausted_conflicts += exhausted
            result.max_op_attempts = max(result.max_op_attempts, max_op_attempts)
            if latencies and max(latencies) >= max(result.latencies_ms or [0.0]):
                result.slowest_op_attempts = slowest_op_attempts

    def checkpointer(driver) -> None:
        """Fire `CALL db.checkpoint()` every `checkpoint_every` seconds until
        the window closes, timing each round-trip.

        `stop.wait(period)` returns True the moment the window ends, so a
        checkpoint is never issued after `stop.set()` — the timed calls all
        overlap live writers, which is the point of the arm. The call is
        auto-commit: the verb refuses to run inside an explicit transaction.
        """
        errors: list[str] = []
        durations: list[float] = []
        written = 0
        skipped = 0
        try:
            with driver.session() as session:
                while not stop.wait(checkpoint_every):
                    t0 = time.perf_counter()
                    try:
                        record = session.run("CALL db.checkpoint() YIELD success, message").single()
                    except Exception as e:  # noqa: BLE001
                        errors.append(repr(e))
                        continue
                    durations.append((time.perf_counter() - t0) * 1000.0)
                    message = record["message"]
                    if not record["success"]:
                        errors.append(f"success=false: {message!r}")
                    elif message.startswith("checkpoint written"):
                        written += 1
                    elif message.startswith("skipped"):
                        skipped += 1
                    else:
                        errors.append(f"unrecognized message: {message!r}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"session: {e!r}")
        with lock:
            result.checkpoint_ms.extend(durations)
            result.checkpoints_written += written
            result.checkpoints_skipped += skipped
            result.checkpoint_errors.extend(errors)

    def read_prober(driver) -> None:
        """The checkpoint thread's sham twin: same extra session, same
        period, a trivial read instead of the verb.

        Whatever the checkpointed arm gains from merely having one more
        session on the driver, this arm gains too — so the difference
        between the two arms is what saving actually did.
        """
        errors: list[str] = []
        durations: list[float] = []
        try:
            with driver.session() as session:
                while not stop.wait(read_probe_every):
                    t0 = time.perf_counter()
                    try:
                        session.run(READ_PROBE_QUERY).consume()
                    except Exception as e:  # noqa: BLE001
                        errors.append(repr(e))
                        continue
                    durations.append((time.perf_counter() - t0) * 1000.0)
        except Exception as e:  # noqa: BLE001
            errors.append(f"session: {e!r}")
        with lock:
            result.probe_ms.extend(durations)
            result.probe_errors.extend(errors)

    extra_args = ["--durability", durability] if durability is not None else None
    proc, url = _spawn_bolt_server(cell_graph, binary=_BENCH_BINARY, extra_args=extra_args)
    try:
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password"), **_RETRY_KW) as driver:
            threads = [threading.Thread(target=worker, args=(driver, i), daemon=True) for i in range(writers)]
            for t in threads:
                t.start()
            try:
                start_barrier.wait()
            except threading.BrokenBarrierError:
                pass
            t_start = time.perf_counter()
            result.window_end = t_start + seconds
            ckpt = None
            if checkpoint_every is not None:
                ckpt = threading.Thread(target=checkpointer, args=(driver,), daemon=True)
                ckpt.start()
            probe = None
            if read_probe_every is not None:
                probe = threading.Thread(target=read_prober, args=(driver,), daemon=True)
                probe.start()
            time.sleep(seconds)
            stop.set()
            for t in threads:
                t.join(timeout=120)
            # Elapsed is read before joining the checkpoint thread on
            # purpose. A checkpoint in flight when the window closes would
            # otherwise be added to the denominator and charge the
            # checkpointed arm a throughput dip it did not have.
            result.elapsed_s = time.perf_counter() - t_start
            # Read once every writer has drained, so the sidecar holds one
            # frame per *committed* transaction of this cell — the pairing
            # `wal_bytes_per_commit` assumes. The cell's copy starts with
            # no sidecar and nothing truncates it unless the cell asked for
            # a checkpoint, so this is the cell's own growth.
            result.wal_bytes = wal_file.stat().st_size if wal_file.exists() else 0
            if ckpt is not None:
                ckpt.join(timeout=120)
            if probe is not None:
                probe.join(timeout=120)

            with driver.session() as session:
                record = session.run(
                    "MATCH (n:Person) WHERE n.city = 'loadtest' RETURN count(n) AS c",
                ).single()
                result.landed = record["c"]
    finally:
        _teardown_bolt_server(proc)
    return result


def _format_table(cells: list[CellResult]) -> str:
    lines = [
        "",
        f"bolt contended writers — binary: {_BENCH_BINARY}",
        f"{'N':>3} {'committed/s':>12} {'committed':>10} {'attempts':>9} {'retry_rate':>11} "
        f"{'p50_ms':>8} {'p95_ms':>8} {'p99_ms':>8} {'max_ms':>9} {'mean_ms':>8} "
        f"{'max_att':>8} {'slow_att':>9} {'exhaust':>8}",
    ]
    for c in cells:
        lines.append(
            f"{c.writers:>3} {c.committed_per_s:>12.1f} {c.committed:>10} {c.attempts:>9} "
            f"{c.retry_rate:>11.3f} {_percentile(c.latencies_ms, 50):>8.2f} "
            f"{_percentile(c.latencies_ms, 95):>8.2f} {_percentile(c.latencies_ms, 99):>8.2f} "
            f"{max(c.latencies_ms or [0.0]):>9.2f} {c.mean_ms:>8.2f} "
            f"{c.max_op_attempts:>8} {c.slowest_op_attempts:>9} {c.exhausted_conflicts:>8}"
        )
    return "\n".join(lines)


def _format_batch_table(cells: list[CellResult]) -> str:
    lines = [
        "",
        f"bolt batch-size sweep — binary: {_BENCH_BINARY}",
        f"{'N':>3} {'K':>5} {'ops/s':>10} {'tx/s':>9} {'committed':>10} {'attempts':>9} "
        f"{'retry_rate':>11} {'p50_ms':>8} {'p95_ms':>8} {'max_ms':>9} {'exhaust':>8}",
    ]
    for c in cells:
        lines.append(
            f"{c.writers:>3} {c.batch:>5} {c.ops_per_s:>10.1f} {c.committed_per_s:>9.1f} "
            f"{c.committed:>10} {c.attempts:>9} {c.retry_rate:>11.3f} "
            f"{_percentile(c.latencies_ms, 50):>8.2f} {_percentile(c.latencies_ms, 95):>8.2f} "
            f"{max(c.latencies_ms or [0.0]):>9.2f} {c.exhausted_conflicts:>8}"
        )
    return "\n".join(lines)


def _format_durability_table(cells: list[CellResult]) -> str:
    lines = [
        "",
        f"bolt durability sweep — binary: {_BENCH_BINARY}",
        f"{'level':>7} {'N':>3} {'win_tx/s':>9} {'in_window':>10} {'committed':>10} "
        f"{'elapsed_tx/s':>13} {'retry_rate':>11} {'p50_ms':>8} {'p95_ms':>8} "
        f"{'mean_ms':>8} {'max_ms':>9} {'wal_KiB':>9} {'wal_B/tx':>9} {'wal_KiB/s':>10}",
    ]
    for c in cells:
        wal_kib = c.wal_bytes / 1024.0
        lines.append(
            f"{c.durability:>7} {c.writers:>3} {c.window_per_s:>9.1f} {c.committed_in_window:>10} "
            f"{c.committed:>10} {c.committed_per_s:>13.1f} {c.retry_rate:>11.3f} "
            f"{_percentile(c.latencies_ms, 50):>8.2f} {_percentile(c.latencies_ms, 95):>8.2f} "
            f"{c.mean_ms:>8.2f} {max(c.latencies_ms or [0.0]):>9.2f} "
            f"{wal_kib:>9.1f} {c.wal_bytes_per_commit:>9.1f} "
            f"{(wal_kib / c.elapsed_s if c.elapsed_s else 0.0):>10.1f}"
        )
    by_key = {(c.durability, c.writers): c for c in cells}
    for n in sorted({c.writers for c in cells}):
        off = by_key.get(("off", n))
        if off is None or off.committed_in_window == 0:
            continue
        for level in ("normal", "full"):
            cell = by_key.get((level, n))
            if cell is None:
                continue
            cost = (off.committed_in_window - cell.committed_in_window) / off.committed_in_window * 100.0
            lines.append(f"  N={n} {level} costs {cost:+.1f}% of off's in-window committed throughput")
    return "\n".join(lines)


def _format_checkpoint_table(cells: list[tuple[str, CellResult]]) -> str:
    lines = [
        "",
        f"bolt checkpoint-under-contention (N={CHECKPOINT_WRITERS}) — binary: {_BENCH_BINARY}",
        f"{'arm':>14} {'win_tx/s':>9} {'in_window':>10} {'committed/s':>12} {'committed':>10} "
        f"{'retry_rate':>11} {'p50_ms':>8} {'p95_ms':>8} {'max_ms':>9} "
        f"{'ckpt_w':>7} {'ckpt_skip':>10} {'ckpt_p50_ms':>12} {'ckpt_max_ms':>12}",
    ]
    for arm, c in cells:
        lines.append(
            f"{arm:>14} {c.window_per_s:>9.1f} {c.committed_in_window:>10} "
            f"{c.committed_per_s:>12.1f} {c.committed:>10} {c.retry_rate:>11.3f} "
            f"{_percentile(c.latencies_ms, 50):>8.2f} {_percentile(c.latencies_ms, 95):>8.2f} "
            f"{max(c.latencies_ms or [0.0]):>9.2f} "
            f"{c.checkpoints_written:>7} {c.checkpoints_skipped:>10} "
            f"{_percentile(c.checkpoint_ms, 50):>12.2f} {max(c.checkpoint_ms or [0.0]):>12.2f}"
        )
    # Every arm is compared against the first one (the un-augmented
    # baseline) on in-window commits, the fixed-denominator cell.
    base = cells[0][1].committed_in_window if cells else 0
    for arm, c in cells[1:]:
        dip = (base - c.committed_in_window) / base * 100.0 if base else 0.0
        lines.append(f"  committed-throughput dip, {arm} vs baseline: {dip:+.1f}%")
    return "\n".join(lines)


def test_checkpoint_under_contention(_writer_fixture, tmp_path):
    """What `CALL db.checkpoint()` costs the writers it interrupts.

    `Session::save` holds the session lock for its whole duration (it
    mutates the graph: metadata stamp, index keys, columnar consolidation),
    so a checkpoint pauses committing writers rather than running beside
    them. This cell prices that pause the only way that matters to an
    operator: two otherwise-identical windows of N contended managed
    writers, one with a thread firing the verb about once a second and one
    without, and the difference in committed throughput.

    A third arm is the sham-thread control: the same extra session at the
    same period running `RETURN 1`. The first measurement of this cell had
    the *checkpointed* arm committing more than its baseline, reproducibly
    and in a reversed-order run; the control separates a real effect of
    saving from anything an extra session on the driver does to the load
    pattern.

    Nothing numeric is asserted — same as every cell in this module. The
    sanity invariants are the ones that would invalidate the number:
    the landed==committed oracle survives concurrent saving, no
    non-conflict error escaped, and the checkpointed arm really did write
    the file more than once (a run where every call hit the digest-skip
    would be a baseline wearing a checkpoint thread's clothes).

    Recorded in `dev-docs/bench/results/results.csv` under
    `bolt_checkpoint:*`; the docs' qualitative claim about checkpoint cost
    is this measurement.
    """
    # Warmup cell — discarded.
    _run_cell(_writer_fixture, tmp_path, WARMUP_WRITERS, WARMUP_SECONDS, "ckptwarm", durability=RECORDED_LEVEL)

    baseline = _run_cell(
        _writer_fixture, tmp_path, CHECKPOINT_WRITERS, CELL_SECONDS, "ckptbase", durability=RECORDED_LEVEL
    )
    checkpointed = _run_cell(
        _writer_fixture,
        tmp_path,
        CHECKPOINT_WRITERS,
        CELL_SECONDS,
        "ckpton",
        checkpoint_every=CHECKPOINT_PERIOD_S,
        durability=RECORDED_LEVEL,
    )
    read_probed = _run_cell(
        _writer_fixture,
        tmp_path,
        CHECKPOINT_WRITERS,
        CELL_SECONDS,
        "ckptprobe",
        read_probe_every=READ_PROBE_PERIOD_S,
        durability=RECORDED_LEVEL,
    )

    arms = [("baseline", baseline), ("checkpointed", checkpointed), ("read-probe", read_probed)]
    print(_format_checkpoint_table(arms))

    for arm, c in arms:
        assert c.hard_errors == [], f"{arm}: non-conflict errors: {c.hard_errors[:3]}"
        assert c.committed > 0, f"{arm}: no writer committed"
        # The oracle the whole arm rests on: a save running concurrently
        # with commits must not lose or duplicate a committed write.
        assert c.landed == c.committed, f"{arm}: landed {c.landed} != committed {c.committed}"
        assert c.committed_per_s > 0

    assert baseline.checkpoint_ms == [], "baseline arm must not have checkpointed"
    assert checkpointed.checkpoint_errors == [], f"checkpoint errors: {checkpointed.checkpoint_errors[:3]}"
    # The control is only a control if its extra session really ran.
    assert read_probed.checkpoint_ms == [], "read-probe arm must not have checkpointed"
    assert read_probed.probe_errors == [], f"read-probe errors: {read_probed.probe_errors[:3]}"
    assert len(read_probed.probe_ms) >= 2, f"read-probe thread barely ran: {read_probed.probe_ms}"
    # Under continuous writes every call should find a new version, so a
    # skip here means the writers stalled or the digest logic changed.
    assert checkpointed.checkpoints_written >= 2, (
        f"expected >=2 real checkpoint writes, got {checkpointed.checkpoints_written} "
        f"written / {checkpointed.checkpoints_skipped} skipped"
    )


def test_contended_writer_sweep(_writer_fixture, tmp_path):
    """Sweep N ∈ {1,2,4,8} managed writers; record the curve, assert only
    that the system stayed correct (see module docstring)."""
    # Warmup cell — discarded.
    _run_cell(_writer_fixture, tmp_path, WARMUP_WRITERS, WARMUP_SECONDS, "warm", durability=RECORDED_LEVEL)

    cells = [
        _run_cell(_writer_fixture, tmp_path, n, CELL_SECONDS, f"n{n}", durability=RECORDED_LEVEL) for n in WRITER_COUNTS
    ]

    print(_format_table(cells))

    for c in cells:
        assert c.hard_errors == [], f"N={c.writers}: non-conflict errors: {c.hard_errors[:3]}"
        assert c.committed > 0, f"N={c.writers}: no writer committed"
        assert c.attempts >= c.committed, f"N={c.writers}: attempts {c.attempts} < committed {c.committed}"
        # Every committed unit of work is present in the graph, and nothing
        # else is: a retried transaction must not leave a partial write.
        assert c.landed == c.committed, f"N={c.writers}: landed {c.landed} != committed {c.committed}"
        assert c.committed_per_s > 0


def test_batch_size_sweep(_writer_fixture, tmp_path):
    """Sweep K ∈ {1,10,100} creations per managed transaction at N=4, with
    an uncontended N=1 control at K ∈ {1,100}.

    This is the measurement behind the operator page's "batch more work
    into each transaction rather than adding writers": committed *ops*/s
    (K × tx/s) is the decision metric, since transactions/s necessarily
    falls as K grows. As everywhere in this module, nothing numeric is
    asserted — the recorded curve lives in
    `dev-docs/bench/results/results.csv`.
    """
    # Warmup cell — discarded.
    _run_cell(
        _writer_fixture, tmp_path, WARMUP_WRITERS, WARMUP_SECONDS, "batchwarm", batch=10, durability=RECORDED_LEVEL
    )

    cells = [
        _run_cell(_writer_fixture, tmp_path, n, CELL_SECONDS, f"b{k}n{n}", batch=k, durability=RECORDED_LEVEL)
        for n, k in BATCH_CELLS
    ]

    print(_format_batch_table(cells))

    for c in cells:
        label = f"N={c.writers} K={c.batch}"
        assert c.hard_errors == [], f"{label}: non-conflict errors: {c.hard_errors[:3]}"
        assert c.committed > 0, f"{label}: no writer committed"
        assert c.attempts >= c.committed, f"{label}: attempts {c.attempts} < committed {c.committed}"
        # A batched transaction is all-or-nothing: every committed unit of
        # work contributed exactly K nodes, and a retried or conflicted one
        # contributed none.
        assert c.landed == c.committed * c.batch, f"{label}: landed {c.landed} != committed*K {c.committed * c.batch}"
        assert c.ops_per_s > 0


def test_durability_sweep(_writer_fixture, tmp_path):
    """What the write-ahead log costs a Bolt writer, per level and per N.

    Every Bolt commit already serializes on the session lock. `full` puts a
    device barrier (`F_FULLFSYNC` on macOS) *inside* that lock, so its cost
    is paid by every writer waiting behind the one committing, not just by
    the committer; `normal` writes the same frame but stops at the page
    cache. This cell prices both against `off` at an uncontended N=1 and a
    contended N=4, and records how fast the `<graph>-wal` sidecar grows,
    which is what tells an operator how often to checkpoint.

    Committed-in-window is the throughput cell here (see the module
    docstring): a `full` cell's drain tail after the window closes is a
    whole fsync per writer, which `committed / elapsed` would charge to the
    level being measured.

    Nothing numeric is asserted — the sanity invariants are the ones that
    would invalidate the numbers: the landed==committed oracle holds at
    every level (a logged commit must be exactly as present as an unlogged
    one), no non-conflict error escaped, and a logging level really did
    write a sidecar while `off` really did not.
    """
    # Warmup cell — discarded.
    _run_cell(_writer_fixture, tmp_path, WARMUP_WRITERS, WARMUP_SECONDS, "durwarm", durability="off")

    cells = [
        _run_cell(_writer_fixture, tmp_path, n, CELL_SECONDS, f"dur{level}n{n}", durability=level)
        for level, n in DURABILITY_CELLS
    ]

    print(_format_durability_table(cells))

    for c in cells:
        label = f"--durability {c.durability} N={c.writers}"
        assert c.hard_errors == [], f"{label}: non-conflict errors: {c.hard_errors[:3]}"
        assert c.committed > 0, f"{label}: no writer committed"
        # The oracle: logging must not lose, duplicate or delay a write.
        assert c.landed == c.committed, f"{label}: landed {c.landed} != committed {c.committed}"
        assert c.committed_in_window > 0, f"{label}: nothing committed inside the window"
        if c.durability == "off":
            assert c.wal_bytes == 0, f"{label}: an unlogged server wrote a {c.wal_bytes}-byte sidecar"
        else:
            assert c.wal_bytes > 0, f"{label}: a logging server wrote no sidecar"
