"""Does one `add_nodes` call write a write-ahead-log frame quadratic in its
own row count?

Measured 2026-07-27 against mapped mode: **payload ~= 0.1496 * n^2 bytes** at
116-byte strings, fitted exponent **2.00 twice**, three points to 0.1%. Because
the 4 GiB ceiling is **per call** (`MAX_WAL_FRAME_BYTES = u32::MAX`,
`wal.rs:115`), a large bulk load fails with `payload is 4920896785 bytes` and
chunking the same rows into smaller calls is a complete workaround — the shape
of a bug, not of a limit.

────────────────────────────────────────────────────────────────────────────
The attribution in the original report is probably wrong, and these cells
are built to settle it rather than to inherit it
────────────────────────────────────────────────────────────────────────────

It was recorded as a **mapped-mode** defect, because that is where it was hit.
Reading the source says the storage mode is incidental and the real ingredient
is the **WAL**:

* `add_nodes` -> `apply_node_batch` -> `commit_wal` (`kg_mutation.rs:1123`) ->
  `flush_wal` (`kglite-py/src/graph/mod.rs:404-437`), which drains the capture
  buffer, resolves it, and appends **one** `WalFrame`.
* The op count is inflated by two loops in the columnar bulk-append path:
  `Batch::detach_columnar_stores` (`batch.rs:354-390`, the iteration at
  **`batch.rs:367`**) and `Batch::reattach_columnar_stores` (`batch.rs:396-437`,
  at **`batch.rs:417`**). Both iterate *every existing node of the affected
  type* calling `GraphWrite::node_weight_mut`, and `RecordingGraph::
  node_weight_mut` (`recording.rs:568-579`) pushes a `RawOp::UpsertNode` on
  every call. Each resolved op then carries the node's whole property map
  (`recording.rs:195-200`).
* With `LARGE_BATCH_CHUNK_SIZE = 1000` (`batch.rs:24`), an `n`-row call records
  roughly `2 * sum_k 1000k ~= n^2/1000` ops. At ~150 bytes per op with
  116-byte strings that is **0.15 * n^2 bytes** — which is the measured
  0.1496 * n^2, including the constant.

Two ingredients are required: the type must already own a column store (so the
detach/reattach loops run at all), and the WAL must be on (so the ops are
recorded rather than discarded). The competitive run had both without intending
either — `kglite.open(path, storage="mapped")` resolves `durable=None` to
**`Full`**, silently turning the log on, whereas `KnowledgeGraph(storage=
"mapped")` has no WAL at all.

**SETTLED 2026-07-30, and the guess above was wrong.** The paragraph that stood
here argued "neither ingredient is mapped" and predicted that an ordinary
`kglite.open(...)` application would hit this after any `save()`. It does not.
The sweeps are storage-mode gated at `batch.rs:186`::

    let mapped = graph.graph.is_mapped() || graph.graph.is_disk();

and the detach/reattach pair runs only under that flag, so a memory-backed
durable graph never enters the amplifying path. Demonstrated rather than read:
reverting the reattach call site to the recorded borrow turns `mapped_normal`
red at ratio 2.18 while `default_normal` passes untouched.

The correct scope is **mapped or disk, plus a WAL** — which is exactly how the
shipped CHANGELOG entry phrases it. That mis-attribution propagated into the
backlog, where it was recorded as "THIS IS NOT A MAPPED-MODE BUG"; it is not a
*mapped-mode* bug in the narrow sense, because disk hits it too, but it is
firmly a not-memory bug, and the difference matters when choosing which arm
guards it.

**RESOLVED — and the fix was exactly this.** `GraphWrite::node_weight_mut_silent`
existed for the case already (`storage/mod.rs`, impl in `recording.rs`), doc
reading *"Bypass recording — internal bookkeeping (columnar handle refresh),
not a logical mutation."* Both loops were precisely that: the detach writes a
temporary empty map that reattach restores before the flush, and genuinely-new
nodes already got their op from `RecordingGraph::add_node`. Since `resolve_ops`
reads *final* state, switching both call sites collapsed the payload to O(n).
Shipped in v0.15.0 as `3bf9ef00`; measured 1k-row 2,000 ops / 84 B per row and
4k-row 20,000 ops / 213 B per row, both now 1 op/row at flat byte cost.

The file also moved: the two sweeps are now `graph/mutation/batch.rs:393`
(detach) and `:449` (reattach), not the `batch.rs:367`/`:417` cited above. The
remaining *recorded* `node_weight_mut` in that file is a genuine logical
mutation and is correctly still recorded.

────────────────────────────────────────────────────────────────────────────
Why payload bytes, and not just wall time
────────────────────────────────────────────────────────────────────────────

The WAL sidecar is `<path>-wal` (`wal.rs:520-524`), so
`os.path.getsize(path + "-wal")` is a **deterministic, noise-free** measurement
of the thing that is actually broken. An exponent fitted from byte counts needs
no idle machine, no warmup, no `min`-vs-`mean` argument and no thermal settle —
it is either 1.0 or 2.0. Wall time is reported too, because that is what a user
feels, but the byte count is the evidence.

Each cell records `wal_bytes` and `wal_bytes_per_row` into
`benchmark.extra_info`, so both land in `--benchmark-json` output. Read
`wal_bytes_per_row` across the row counts: **flat = linear = fixed; growing
proportionally to `rows` = the quadratic is still there.**

The defect this file was built to measure was **fixed in v0.15.0** by
`3bf9ef00`, which switched both `batch.rs` sweeps to the silent accessor. The
follow-up that fix set up — a real assertion — is
`test_wal_bytes_per_row_are_flat` below.

That guard is deliberately **not** marked `benchmark`. Everything else here is,
and `-m 'not benchmark'` is in the default `addopts`, so a guard living behind
that marker would never run in the ordinary suite — a gate nobody executes. It
needs no `benchmark` fixture either: the payload measurement is an untimed pair
of `stat` calls, exact on the first try, with nothing to average.

⚠ Historical note, kept because it cost real time: between 07:14 and 11:00 on
2026-07-27 this docstring said "**the defect is live** and a guard would fail
on `main`". The fix landed at 11:00 and the docstring was not updated. On
2026-07-30 an investigating agent read it, trusted it over the code, and
reported the bug as still live — while `batch.rs:393` and `:449` had been
calling `node_weight_mut_silent` for three days. A file that asserts nothing
cannot correct a stale claim about itself; only an assertion can.

Row counts stop at 32k on purpose: `postcard::to_stdvec` materialises the whole
payload before the limit is checked (`postcard_v1.rs:27-31`), so approaching
the 4 GiB ceiling means a multi-gigabyte allocation and then an error. 32k rows
is ~150 MB — enough to fit the curve to several decimal places, nowhere near
enough to reproduce the crash. **Do not raise this to "see the failure"**; the
failure is arithmetic, and the curve already predicts it.

Nothing here is in the `make bench-check` tracked set (`Makefile:85`).

Run with::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_wal_payload.py \\
        -m benchmark -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

# Six geometric points. A quadratic is a straight line of slope 2 on a log-log
# plot, and six points make the fit obvious by eye without any curve-fitting
# machinery — the original finding needed only three to reach 0.1%.
#
# The top of the range is bounded by the payload it produces, not by runtime:
# 32k rows is ~0.1496 * 32000^2 ~= 153 MB. See the module docstring before
# raising it.
ROW_COUNTS = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]

#: Seed rows written and checkpointed before the measured call, so the type
#: already owns a column store when `add_nodes` runs. Without a pre-existing
#: column store the detach/reattach loops never execute (`batch.rs:402-404`
#: early-returns when nothing was detached) and the cell would measure a code
#: path the defect does not live on.
SEED_ROWS = 1_000

#: 116-byte strings, matching the payload constant quoted in the report. The
#: property payload is what each amplified op carries, so string width is a
#: direct multiplier on the measured bytes — a narrower fixture would show the
#: same exponent with a different constant and would not be comparable to the
#: 0.1496 figure.
FILLER = "x" * 116

# The three arms that decide storage-mode versus WAL. `normal` rather than
# `full`: the amplification is in the frame's *size*, and `full` would add one
# F_FULLFSYNC per commit (3.37 ms on this machine) to a wall-time number
# without changing a single payload byte.
ARMS = ["default_off", "default_normal", "mapped_normal"]

# Few rounds, and deliberately so. Each round rebuilds the fixture graph from
# scratch in an untimed setup, and at the top row count each round writes
# ~150 MB of WAL. The payload measurement — the part that matters — is exact
# and needs no repetition at all; the rounds exist only to give the wall-time
# figure a little stability.
ROUNDS = 3
WARMUP_ROUNDS = 1


def _frame(rows: int, offset: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(offset, offset + rows),
            "name": [f"item-{i}" for i in range(rows)],
            "body": [FILLER] * rows,
        }
    )


def _wal_bytes(path: str) -> int:
    """Size of the `<path>-wal` sidecar, or 0 when there is no log.

    At `durable="off"` the file is never created, which is why the control arm
    reports 0 rather than a small number — that is the correct reading, not a
    missing measurement.
    """
    try:
        return os.path.getsize(path + "-wal")
    except FileNotFoundError:
        return 0


def _reset(path: str) -> None:
    """Remove a previous round's graph and its sidecars.

    Called before every rebuild so that at most one graph's files exist at a
    time. Without this, a cell at the top row count would leave ~150 MB per
    round behind, and the file as a whole would leave gigabytes in the pytest
    temp directory — the kind of ungated accumulation CLAUDE.md requires an
    owner for.
    """
    for suffix in ("", "-wal", ".lock"):
        try:
            os.unlink(path + suffix)
        except FileNotFoundError:
            pass


def _seeded_columnar_graph(path: str, arm: str) -> KnowledgeGraph:
    """A file-backed graph whose `Item` type already owns a column store.

    `save()` is what consolidates `DirGraph.column_stores` — and it also
    truncates the WAL back to its 5-byte
    header (`wal.rs:557-566`), which is what makes the post-save sidecar size a
    clean zero point for the delta the cells measure.
    """
    _reset(path)
    if arm == "default_off":
        graph = kglite.open(path, durable="off", lock=False)
    elif arm == "default_normal":
        graph = kglite.open(path, durable="normal", lock=False)
    else:
        graph = kglite.open(path, storage="mapped", durable="normal", lock=False)
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(_frame(SEED_ROWS, 0), "Item", "id", "name")
    graph.save()
    return graph


@pytest.mark.benchmark
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_bench_add_nodes_wal_payload(benchmark, tmp_path, rows, arm):
    """One `add_nodes` call of `rows` rows — wall time, and the bytes it logged.

    Defends the measured **0.1496 * n^2** payload with exponent **2.00**
    (2026-07-27) and the per-call 4 GiB ceiling it eventually overflows.

    Read `wal_bytes_per_row` from `extra_info` across the `rows` parameter, at
    a fixed `arm`. Flat means the amplification is gone; proportional to `rows`
    means it is still there. Then read the same row count across arms to decide
    whether this belongs to mapped mode or to the WAL — see the module
    docstring for what each answer implies.
    """
    path = str(tmp_path / "ingest.kgl")

    # The payload measurement is exact, so it is taken once and untimed rather
    # than averaged over rounds. Taking it inside the timed callable would put
    # two `stat` calls inside the measurement for no gain.
    probe = _seeded_columnar_graph(path, arm)
    before = _wal_bytes(path)
    probe.add_nodes(_frame(rows, SEED_ROWS), "Item", "id", "name")
    logged = _wal_bytes(path) - before
    benchmark.extra_info["arm"] = arm
    benchmark.extra_info["rows"] = rows
    benchmark.extra_info["wal_bytes"] = logged
    benchmark.extra_info["wal_bytes_per_row"] = logged / rows
    del probe

    offsets = iter(range(SEED_ROWS, 1 << 30, rows))

    def setup():
        graph = _seeded_columnar_graph(path, arm)
        return (graph, _frame(rows, next(offsets))), {}

    def ingest(graph, frame):
        graph.add_nodes(frame, "Item", "id", "name")
        # Returned so the graph is unambiguously alive for the whole timed
        # call and its deallocation cannot land inside the next round.
        return graph

    benchmark.pedantic(ingest, setup=setup, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)
    _reset(path)


@pytest.mark.benchmark
@pytest.mark.parametrize("arm", ARMS)
def test_bench_add_nodes_chunked_workaround(benchmark, tmp_path, arm):
    """The same rows ingested in 1k-row calls instead of one large call.

    The ceiling is **per call**, so chunking is the documented workaround — and
    if the amplification is real, chunking is not merely a way to stay under
    4 GiB but a large throughput win, because it caps the quadratic term at the
    chunk size. That makes this cell the measurement of what the workaround is
    worth, and the natural before/after for a fix: once the two `batch.rs` call
    sites stop recording bookkeeping writes, this cell and the 32k row of
    `test_bench_add_nodes_wal_payload` should converge.

    Same total rows as the largest cell above, so the two are directly
    comparable.
    """
    path = str(tmp_path / "chunked.kgl")
    total = ROW_COUNTS[-1]
    chunk = 1_000

    probe = _seeded_columnar_graph(path, arm)
    before = _wal_bytes(path)
    for start in range(SEED_ROWS, SEED_ROWS + total, chunk):
        probe.add_nodes(_frame(chunk, start), "Item", "id", "name")
    logged = _wal_bytes(path) - before
    benchmark.extra_info["arm"] = arm
    benchmark.extra_info["rows"] = total
    benchmark.extra_info["chunk"] = chunk
    benchmark.extra_info["wal_bytes"] = logged
    benchmark.extra_info["wal_bytes_per_row"] = logged / total
    del probe

    offsets = iter(range(SEED_ROWS, 1 << 30, total))

    def setup():
        graph = _seeded_columnar_graph(path, arm)
        return (graph, next(offsets)), {}

    def ingest(graph, offset):
        for start in range(offset, offset + total, chunk):
            graph.add_nodes(_frame(chunk, start), "Item", "id", "name")
        return graph

    benchmark.pedantic(ingest, setup=setup, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)
    _reset(path)


# ---------------------------------------------------------------------------
# The guard. Unmarked on purpose — see the module docstring.
# ---------------------------------------------------------------------------

#: Two row counts an octave apart. The defect was `payload ~= 0.1496 * n^2`, so
#: an 8x row count meant ~8x the bytes *per row*; linear means the ratio sits at
#: 1.0. Anything between those is not a shape this code can produce, which is
#: why a single loose threshold separates them cleanly.
FLAT_ROW_COUNTS = (1_000, 8_000)

#: Slack over 1.0. The per-row cost is not perfectly constant — the frame header
#: and the postcard varint widths grow with row *count*, not row count squared —
#: so a few percent of drift is real and expected. 1.25 sits far above that and
#: far below the ~8x a reintroduced quadratic would produce.
FLAT_TOLERANCE = 1.25

#: 32k is used by the benchmark cells above but not here: it writes ~150 MB per
#: measurement, and this guard runs in the DEFAULT suite under a 120 s timeout.
#: 8k is enough to separate n from n^2 by 8x.
#:
#: ONLY the arms that actually execute the sweeps. `batch.rs:186` reads
#: `let mapped = graph.graph.is_mapped() || graph.graph.is_disk();` and the
#: detach/reattach pair runs only under that flag — so a memory-backed durable
#: graph never enters the amplifying path at all. `default_normal` was in this
#: tuple for one revision and is the reason it is now documented: with the
#: reattach site deliberately reverted to the recorded borrow, `mapped_normal`
#: went red at ratio 2.18 while `default_normal` sailed through. An arm that
#: cannot fail is not a guard, whatever it is measuring.
#:
#: Disk shares the identical `is_disk()` branch. It is not covered here only
#: because a disk fixture is materially heavier to stand up; the mapped arm
#: exercises the same two call sites.
SWEEP_ARMS = ("mapped_normal",)


def _wal_bytes_per_row(tmp_path, arm: str, rows: int) -> float:
    """Bytes the WAL grew by during one `add_nodes`, divided by rows written.

    Mirrors the untimed measurement in `test_bench_add_nodes_wal_payload` — the
    seeded graph gives the type a column store so the detach/reattach sweeps
    actually execute, and the post-`save()` sidecar is a clean zero point.
    """
    path = str(tmp_path / f"flat-{arm}-{rows}.kgl")
    probe = _seeded_columnar_graph(path, arm)
    before = _wal_bytes(path)
    probe.add_nodes(_frame(rows, SEED_ROWS), "Item", "id", "name")
    logged = _wal_bytes(path) - before
    del probe
    _reset(path)
    return logged / rows


@pytest.mark.parametrize("arm", SWEEP_ARMS)
def test_wal_bytes_per_row_are_flat(tmp_path, arm):
    """One `add_nodes` must log a constant number of bytes per row.

    Guards the v0.15.0 fix (`3bf9ef00`): the columnar detach/reattach sweeps in
    `batch.rs` once iterated every existing node of the type through the
    *recorded* `node_weight_mut`, pushing a `RawOp::UpsertNode` carrying the
    whole property map per call — `~n^2/1000` ops at `LARGE_BATCH_CHUNK_SIZE`.

    Byte counts are deterministic, so this needs no idle machine, no warmup and
    no min-vs-mean argument. It is a correctness assertion that happens to live
    in a benchmark file, not a performance gate.
    """
    small = _wal_bytes_per_row(tmp_path, arm, FLAT_ROW_COUNTS[0])
    large = _wal_bytes_per_row(tmp_path, arm, FLAT_ROW_COUNTS[1])

    # A zero here would make the ratio meaningless (0/0) and pass vacuously —
    # exactly the failure this file spent three days demonstrating. These arms
    # are the LOGGED ones; if nothing was logged, the measurement is broken,
    # not the code under test.
    assert small > 0, f"{arm}: no WAL bytes logged at {FLAT_ROW_COUNTS[0]} rows"
    assert large > 0, f"{arm}: no WAL bytes logged at {FLAT_ROW_COUNTS[1]} rows"

    ratio = large / small
    assert ratio <= FLAT_TOLERANCE, (
        f"{arm}: WAL payload is not linear in row count. "
        f"{FLAT_ROW_COUNTS[0]} rows logged {small:.1f} B/row, "
        f"{FLAT_ROW_COUNTS[1]} rows logged {large:.1f} B/row "
        f"(ratio {ratio:.2f} > {FLAT_TOLERANCE}). "
        f"An {FLAT_ROW_COUNTS[1] // FLAT_ROW_COUNTS[0]}x ratio means the "
        f"quadratic amplification is back: check that both sweeps in "
        f"graph/mutation/batch.rs still call node_weight_mut_silent."
    )


def test_durable_off_logs_nothing(tmp_path):
    """Control. Without it, a broken measurement reads as a pass.

    `durable="off"` never creates the sidecar, so `_wal_bytes` returns 0 by the
    FileNotFoundError path. If this ever reports bytes, the arm above is
    measuring something other than the WAL and its ratio proves nothing.
    """
    assert _wal_bytes_per_row(tmp_path, "default_off", FLAT_ROW_COUNTS[0]) == 0


def test_memory_backed_durable_graph_never_runs_the_sweeps(tmp_path):
    """Pins WHY `default_normal` is absent from `SWEEP_ARMS` — it cannot fail.

    This is documentation with an assertion attached, not a regression guard.
    The columnar detach/reattach pair is gated on
    `graph.graph.is_mapped() || graph.graph.is_disk()` (`batch.rs:186`), so a
    memory-backed durable graph never enters the path that once amplified the
    payload. Its bytes-per-row is therefore flat by construction, and would
    stay flat with the fix reverted — verified by doing exactly that.

    If this ever goes red, the mode gate at `batch.rs:186` changed and
    `SWEEP_ARMS` must grow to match, or the real guard silently stops covering
    the ordinary durable path.
    """
    small = _wal_bytes_per_row(tmp_path, "default_normal", FLAT_ROW_COUNTS[0])
    large = _wal_bytes_per_row(tmp_path, "default_normal", FLAT_ROW_COUNTS[1])
    assert small > 0 and large > 0, "expected a memory-backed durable graph to log"
    assert large / small <= FLAT_TOLERANCE
