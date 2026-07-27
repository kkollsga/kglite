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

Two ingredients are therefore required, and **neither of them is "mapped"**:
the type must already own a column store (so the detach/reattach loops run at
all), and the WAL must be on (so the ops are recorded rather than discarded).
The competitive run had both without intending either — `kglite.open(path,
storage="mapped")` resolves `durable=None` to **`Full`** (`lib.rs:422-431`),
silently turning the log on, whereas `KnowledgeGraph(storage="mapped")` has no
WAL at all.

The three arms below are chosen to decide this. If `default_normal` matches
`mapped_normal`, the mapped attribution is refuted and the bug belongs to the
WAL path, where an ordinary `kglite.open(...)` application after a `save()`
would hit it too. If `default_off` stays linear while both logged arms go
quadratic, the WAL is confirmed as the carrier.

**This looks like a defect with a fix already sitting in the codebase.**
`GraphWrite::node_weight_mut_silent` exists for exactly this
(`storage/mod.rs:448`, impl `recording.rs:582-586`) and its doc reads *"Bypass
recording — internal bookkeeping (columnar handle refresh), not a logical
mutation."* Both loops are precisely that: the detach writes a temporary empty
map that reattach restores before the flush, and genuinely-new nodes already
got their op from `RecordingGraph::add_node` (`recording.rs:597-601`). Since
`resolve_ops` reads *final* state, switching both call sites to the silent
variant should collapse the payload to O(n). Not attempted here — this branch
only adds benchmarks. Filed as a finding.

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

⚠ This file asserts nothing about the exponent, because at the time of writing
**the defect is live** and a guard would fail on `main`. It is a marker for a
known open issue in the sense `test_bench_write_scaling.py::test_bench_id_
index_invalidation_on_create` is one. Once the two `batch.rs` call sites are
switched to the silent accessor, `wal_bytes_per_row` becomes flat and a real
assertion can be added here — that is the follow-up this file is setting up.

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

    `save()` is what populates `DirGraph.column_stores`, via `enable_columnar()`
    (`io/file.rs:1495`) — and it also truncates the WAL back to its 5-byte
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
