"""Contended-writer load test for kglite-bolt-server (Track G, G4a).

Answers the only question the write path can actually be asked: all Bolt
writes are *explicit* transactions (auto-commit mutations are rejected at
`backend.rs`), so what matters is how committed throughput, conflict rate,
and latency move as N driver-managed writers contend for the single graph.

Two sweeps live here: `test_contended_writer_sweep` varies the writer count
at one write per transaction, and `test_batch_size_sweep` varies the writes
per transaction at a fixed writer count — the measurement behind the
operator page's "batch more work into each transaction rather than adding
writers".

The unit of work runs through `session.execute_write`, i.e. the
driver-managed path that retries a `Neo.TransientError.*` failure by
itself — the system a caller actually gets after the conflict code became
retriable. A per-worker attempt counter is incremented *inside* the unit of
work, so `attempts - committed` is the number of driver retries.

Design notes that matter for reading the numbers:

- **Fixed wall-clock window per cell.** Committed throughput is the metric;
  a fixed op count would let a heavily-retried cell run far longer and turn
  throughput into a function of the slowest worker.
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

    @property
    def committed_per_s(self) -> float:
        return self.committed / self.elapsed_s if self.elapsed_s > 0 else 0.0

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
    if not _BOLT_BINARY.exists():
        pytest.skip(f"bolt-server binary not built at {_BOLT_BINARY}")
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
) -> CellResult:
    """One cell: `writers` threads hammering `execute_write` for `seconds`.

    Each thread owns its own session and a disjoint id range, so the final
    graph count is an exact oracle for "every committed write landed".

    `batch` selects the unit of work. `None` (the writer-count sweep) is a
    bare single-node `CREATE`. An integer K runs `_BATCH_QUERY` with K rows,
    including at K=1 — so the batch sweep's cells differ *only* in K, and
    the K=1 cell carries the same `UNWIND`/parameter overhead as K=100.
    """
    rows_per_tx = 1 if batch is None else batch
    cell_graph = tmp_dir / f"cell_{tag}.kgl"
    shutil.copy(fixture_path, cell_graph)

    stop = threading.Event()
    start_barrier = threading.Barrier(writers + 1, timeout=60)
    lock = threading.Lock()
    result = CellResult(writers=writers, committed=0, attempts=0, elapsed_s=0.0, batch=rows_per_tx)

    def worker(driver, worker_id: int) -> None:
        attempts = 0
        committed = 0
        latencies: list[float] = []
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
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    op_attempts = attempts - attempts_before
                    latencies.append(elapsed_ms)
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
            result.hard_errors.extend(hard)
            result.exhausted_conflicts += exhausted
            result.max_op_attempts = max(result.max_op_attempts, max_op_attempts)
            if latencies and max(latencies) >= max(result.latencies_ms or [0.0]):
                result.slowest_op_attempts = slowest_op_attempts

    proc, url = _spawn_bolt_server(cell_graph)
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
            time.sleep(seconds)
            stop.set()
            for t in threads:
                t.join(timeout=120)
            result.elapsed_s = time.perf_counter() - t_start

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
        f"bolt contended writers — binary: {_BOLT_BINARY}",
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
        f"bolt batch-size sweep — binary: {_BOLT_BINARY}",
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


def test_contended_writer_sweep(_writer_fixture, tmp_path):
    """Sweep N ∈ {1,2,4,8} managed writers; record the curve, assert only
    that the system stayed correct (see module docstring)."""
    # Warmup cell — discarded.
    _run_cell(_writer_fixture, tmp_path, WARMUP_WRITERS, WARMUP_SECONDS, "warm")

    cells = [_run_cell(_writer_fixture, tmp_path, n, CELL_SECONDS, f"n{n}") for n in WRITER_COUNTS]

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
    _run_cell(_writer_fixture, tmp_path, WARMUP_WRITERS, WARMUP_SECONDS, "batchwarm", batch=10)

    cells = [_run_cell(_writer_fixture, tmp_path, n, CELL_SECONDS, f"b{k}n{n}", batch=k) for n, k in BATCH_CELLS]

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
