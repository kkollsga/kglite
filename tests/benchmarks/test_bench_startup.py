"""How long until an embedded database can answer its first question?

For a library that starts with the process, this is a cost paid on every cold
request, every CLI invocation, every worker respawn — and it is the one number
where the 2026-07-27 competitive benchmark found the largest structural gap:

| graph  | kglite open -> first query | SQLite `wal_normal` |
|--------|---------------------------:|--------------------:|
| 100k   |                   113.5 ms |             1.44 ms |

**79x**, and unlike a per-query gap it cannot be amortised by doing more work
per process. SQLite opens a file handle and reads a page; kglite deserializes
the entire graph.

Nothing in this suite measured it. The nearest existing cell,
`test_bench_phase19.py::test_bench_disk_reopen_query_mutate_promote`, covers
**disk** mode only and bundles a query, a write and a `save()` into one number.
There was no benchmark timing `kglite.open()` or `kglite.load()` of a `.kgl`
file in memory mode at all.

────────────────────────────────────────────────────────────────────────────
⚠ The `storage="mapped"` reading from that run is void. Do not repeat it.
────────────────────────────────────────────────────────────────────────────

The competitive run recorded `storage="mapped"` at **116.5 ms against Default's
113.5 ms** and concluded "mapped does not help — so this is not a Default-mode
artifact". The conclusion is right; the evidence is not evidence. Reading the
source (2026-07-27), those two cells ran **identical code**, for two
independent reasons:

1. `storage=` is **ignored when opening an existing file**. `kglite.open`
   passes it to the constructor only on the file-does-not-exist branch
   (`lib.rs:397-403`), and both the docstring (`__init__.pyi:757-759`) and
   `lib.rs:264-265` say so.
2. Even if it were honoured, it could not matter: `impl Deserialize for
   GraphBackend` (`storage/backend.rs:494-499`) **always** yields
   `GraphBackend::Memory`, and `io/file.rs` contains no occurrence of `Mapped`.
   Mapped-ness is not a property of a loaded `.kgl`. The only mmap decision on
   load is per-column and size-driven — a column spills to a temp file at
   `MMAP_THRESHOLD` = 256 KB (`column_store.rs:137,2294-2306`) — identically
   for both "modes".

So 3 ms out of 115 ms was noise between two runs of the same code path, and no
mapped cell appears in this file. A future mapped-vs-default startup comparison
would have to build the graph mapped *and* teach the loader to preserve the
mode; until then such a cell can only mislead. `graph_info()["columnar_is_
mapped"]` is the field that would actually tell you whether mapped did
anything.

────────────────────────────────────────────────────────────────────────────
What the three arms separate
────────────────────────────────────────────────────────────────────────────

The load is fully eager — `open()` does not return before the work is done, so
timing `open()` *is* timing the load. Four O(graph) passes run in sequence
(`io/file.rs:2395-2422`): a full Postcard deserialize of the `StableDiGraph`
plus `rebuild_type_indices_and_compact()`; per-type column decompress; a
per-node sweep in `attach_portable_column_stores` plus
`rebuild_indices_from_keys()`; then embeddings, timeseries, secondary labels
and the vector index. Nothing is deferred.

* **`load`** — `kglite.load()`. The deserialize alone: no writer lease, no WAL
  recovery, no header barriers.
* **`open_off`** — `kglite.open(durable="off")`. Adds the cross-process writer
  lease. `load` -> `open_off` is the lease cost.
* **`open_full`** — the default. Adds WAL creation, which pays a `sync_all()`
  **and** a `sync_parent_dir()` (`wal.rs:598-606`), plus recovery/replay of any
  existing frames. `open_off` -> `open_full` is the cost of durability at
  startup, and on macOS those barriers are `F_FULLFSYNC` — measured on this
  machine at 3.37 ms each.

Read every arm across sizes. All three are expected to be O(graph) today; the
cells exist so that the *constant* is tracked and so that any future lazy-load
work has a before number to point at.

Nothing here is in the `make bench-check` tracked set (`Makefile:85`).

Run with::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_startup.py \\
        -m benchmark -v
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

# Three decades, because the finding IS the slope. A single size would report a
# large constant and say nothing about whether it is O(graph) — which is the
# whole claim being defended. 1M is omitted: at ~1.1 s per open it would
# dominate the file's runtime while only confirming a line already drawn by
# three points.
SIZES = [1_000, 10_000, 100_000]

# Explicit rounds. At 100k an open is ~113 ms, so auto-calibration would
# schedule a round count from a first call that is not representative (cold
# page cache) and could run for minutes; `-m benchmark` is exempt from the
# 120 s hang ceiling, so nothing would stop it.
#
# 20 rounds x 113 ms x 9 cells is ~25 s for the file, which is the right budget
# for a number quoted this often.
ROUNDS = 20
WARMUP_ROUNDS = 2

#: `open()` variants under test. See the module docstring for what each adds.
OPEN_MODES = ["load", "open_off", "open_full"]


def _saved_graph_path(tmp_dir, size: int) -> str:
    """Write a `size`-node `.kgl` once and return its path.

    A primary key is declared because a real application graph has one, and
    because `rebuild_indices_from_keys()` is one of the O(graph) load passes
    being measured — a keyless graph would skip work that every real open pays.
    """
    graph = KnowledgeGraph()
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(size),
                "name": [f"item-{i}" for i in range(size)],
                "code": [f"code-{i}" for i in range(size)],
                "qty": [i % 977 for i in range(size)],
            }
        ),
        "Item",
        "id",
        "name",
        columns=["code", "qty"],
    )
    graph.add_connections(
        pd.DataFrame({"src": range(size), "dst": [(i * 7 + 13) % size for i in range(size)]}),
        "LINKS",
        "Item",
        "src",
        "Item",
        "dst",
    )
    path = str(tmp_dir / f"startup-{size}.kgl")
    graph.save(path)
    return path


@pytest.fixture(scope="module")
def startup_paths(tmp_path_factory) -> dict[int, str]:
    """One saved `.kgl` per size, under a pytest-managed temp dir.

    `tmp_path_factory` rather than `tmp_path`: this is module-scoped, and
    nothing in this file may write outside a temp directory pytest owns and
    cleans.
    """
    tmp_dir = tmp_path_factory.mktemp("startup")
    return {size: _saved_graph_path(tmp_dir, size) for size in SIZES}


def _open_graph(mode: str, path: str) -> KnowledgeGraph:
    """Open `path` under one of the three arms.

    `lock=False` on both `open` arms is a measurement decision, not a
    convenience. The writer lease is held until `close()`, and `close()`
    *persists the graph to its origin path* — a full O(graph) serialize. A
    benchmark that opened with the lease would therefore have to choose between
    leaking a lock per round and paying a whole save inside the loop, and the
    save is far larger than the open being measured. Opting out of the lease
    keeps the timed region to the load.

    The lease cost is consequently **not measured anywhere in this file**. It
    is a file creation plus an advisory lock — small next to a 113 ms load, but
    unquantified, and that should be stated rather than assumed.
    """
    if mode == "load":
        return kglite.load(path)
    if mode == "open_off":
        return kglite.open(path, durable="off", lock=False)
    return kglite.open(path, lock=False)


@pytest.mark.benchmark
@pytest.mark.parametrize("mode", OPEN_MODES)
@pytest.mark.parametrize("size", SIZES)
def test_bench_open_to_first_query(benchmark, startup_paths, size, mode):
    """Open a saved graph and answer one point query — time to first answer.

    The headline cell. Defends **113.5 ms at 100k against SQLite's 1.44 ms**
    (2026-07-27), a cost paid on every process start.

    The query is included because "open to first query" is the number a caller
    actually experiences, and excluded from interpretation because it is
    negligible: a cold plan-cache miss is tens of microseconds against a
    hundred-plus milliseconds of load.
    `test_bench_first_query_on_open_graph` measures it alone so that claim is
    checked rather than asserted.

    The returned graph is handed back to pytest-benchmark rather than dropped,
    so deallocation of a 100k-node graph cannot land inside the next round's
    timing.
    """
    path = startup_paths[size]

    def open_and_query():
        graph = _open_graph(mode, path)
        graph.cypher("MATCH (n:Item {id: 7}) RETURN n.name").to_list()
        return graph

    benchmark.pedantic(open_and_query, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_first_query_on_open_graph(benchmark, startup_paths, size):
    """The first point query on an already-open graph — the control.

    Exists to keep `test_bench_open_to_first_query` honest. If this is ever a
    meaningful fraction of the open cell, then "startup is O(graph)" has become
    the wrong description of that number and the attribution above needs
    revisiting. Expect microseconds, flat across sizes.
    """
    graph = kglite.load(startup_paths[size])
    counter = iter(range(1 << 30))

    def query():
        node_id = next(counter) % 1000
        return graph.cypher("MATCH (n:Item {id: $i}) RETURN n.name", params={"i": node_id}).to_list()

    benchmark.pedantic(query, rounds=ROUNDS * 5, iterations=1, warmup_rounds=WARMUP_ROUNDS)
