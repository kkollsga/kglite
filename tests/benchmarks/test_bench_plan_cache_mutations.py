"""What does a writer pay for a plan cache it can never read from?

``session/execute.rs::prepare`` decides cacheability with three terms — empty
params, no disabled passes, no value codecs — and **no ``is_mutation`` term**.
So every write that clears those three does a cache ``get`` on the way in and
an ``insert`` on the way out. The key carries the graph version
(``cypher/plan_cache.rs``), and a successful write bumps that version
(``execute.rs``, ``bump_version``) *after* the insert — so a serial writer's
entry is stale the instant it lands. Per write, that is an ``RwLock`` read, an
``RwLock`` write, a ``VecDeque`` push, a FIFO eviction once the 512-entry cache
is full, and an ``Arc<CypherQuery>`` kept alive until it is evicted.

**The Rust side owns the "is it dead?" question; this file owns "what does it
cost?".** ``crates/kglite/src/graph/session/plan_cache_cost_tests.rs`` counts
the events exactly — 600 identical serial writes produce 600 lookups, **0
hits**, 600 insertions, 88 evictions and leave the shared cache at its full 512
entries, all of them mutation-keyed. Those counters are ``#[cfg(test)]``
thread-locals and are not reachable from Python by any route, so nothing here
can assert them. What it can do is time the write.

────────────────────────────────────────────────────────────────────────────
Why the statement text is identical every round, and why that is the hard case
────────────────────────────────────────────────────────────────────────────

An identical, param-less statement is *exactly* what a plan cache exists to
serve, and it is the shape a real write loop has (the parameters that vary live
in ``$params``, which makes a statement uncacheable and is therefore not the
shape at issue). A varying statement text would additionally miss the parse
cache and turn a plan-cache measurement into a parser measurement.

────────────────────────────────────────────────────────────────────────────
Why a bare ``CREATE`` is the headline write shape here
────────────────────────────────────────────────────────────────────────────

The opposite choice from ``test_bench_fast_write_path.py``, for the opposite
reason. That file needs a statement that opens a rollback checkpoint, because
the checkpoint is what it measures. This file measures a **fixed per-statement
overhead**, and a fixed overhead is most visible against the *cheapest* write
in the engine — the single-node ``CREATE`` that ``can_skip_rollback_checkpoint``
(``execute.rs``) lets through with no checkpoint at all. A ``SET`` cell is kept
alongside it: it opens a real checkpoint and costs ~2.4x as much per statement
(measured 2026-08-09: 1.37 us vs 3.21 us min at 1k), so it shows the same fixed
overhead diluted, which is closer to what an application experiences.

The **read cell is not decoration** — it is the control that any change to
plan-cache policy must not regress. It repeats one point-lookup against a graph
that is never written, so every round after the first is a cache *hit*: it
measures the hit path itself.

Nothing here is in the ``make bench-check`` tracked set (that gate runs
``tests/benchmarks/test_bench_core.py`` only, ``Makefile:85``), so this file
cannot perturb it.

Run with::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_plan_cache_mutations.py \\
        -m benchmark -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

#: Two decades. 1k is where a fixed per-statement overhead is the largest share
#: of a write, so it is the most sensitive cell; 100k confirms the overhead is
#: flat in graph size (it must be — the cache key is four integers) and gives
#: the diluted, application-shaped number.
SIZES = [1_000, 100_000]

# `benchmark.pedantic` with an explicit round count, never plain `benchmark(fn)`
# — pytest-benchmark's auto-calibration sizes the round count from a first call
# and this file deliberately spans cells that differ by ~2.5x, so calibration
# performed on the cheap cell schedules the wrong count for the others. Every
# cell here is sub-10 us, which is where the Performance protocol asks for 200
# rounds and says to read `min` (median drifts upward with system load).
ROUNDS = 200
WARMUP_ROUNDS = 20


def _base_graph(size: int) -> KnowledgeGraph:
    """``size`` ``Item`` nodes with a primary key, plus a warmed id index.

    The primary key matters for the ``SET`` cell: without one, each write
    invalidates the type's id index and the next ``MATCH ... {id: 0}`` pays an
    O(V) rebuild — a per-statement cost two orders of magnitude larger than the
    one under measurement, which would swamp it completely (see
    ``test_bench_write_scaling.py``).
    """
    graph = KnowledgeGraph()
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(size),
                "name": [f"item-{i}" for i in range(size)],
                "qty": [i % 977 for i in range(size)],
            }
        ),
        "Item",
        "id",
        "name",
        columns=["qty"],
    )
    # First-touch index build must not land in an arbitrary measured round.
    graph.cypher("MATCH (n:Item {id: 0}) RETURN n.id")
    # Vacuity guard. Every cell in this file is O(1) in graph size by
    # construction — a point lookup through a primary key, a single-node
    # insert, four integers of cache key — so the 1k and 100k cells are
    # *expected* to read the same, and a fixture that silently built the wrong
    # number of nodes would produce exactly that agreement for the wrong
    # reason. Assert the size rather than infer it from a plausible number.
    built = graph.cypher("MATCH (n:Item) RETURN count(n) AS c").to_dicts()[0]["c"]
    assert built == size, f"fixture built {built} nodes, expected {size}"
    return graph


@pytest.fixture(scope="module")
def create_graphs() -> dict[int, KnowledgeGraph]:
    """One graph per size for the ``CREATE`` cell. It grows by ~220 nodes."""
    return {size: _base_graph(size) for size in SIZES}


@pytest.fixture(scope="module")
def set_graphs() -> dict[int, KnowledgeGraph]:
    """Separate graphs for the ``SET`` cell, so the two writers cannot
    interleave version bumps and change each other's node counts."""
    return {size: _base_graph(size) for size in SIZES}


@pytest.fixture(scope="module")
def read_graphs() -> dict[int, KnowledgeGraph]:
    """Graphs that are **never written after construction**.

    That is the entire point of the read control: an unchanged version means
    every round after the first is a plan-cache hit. A stray write in this
    fixture would silently convert the cell into a miss-path measurement and it
    would still look like a plausible number.
    """
    return {size: _base_graph(size) for size in SIZES}


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_repeated_create_write(benchmark, create_graphs, size):
    """The cheapest write in the engine, repeated with identical text.

    Checkpoint-free (``can_skip_rollback_checkpoint``), single node, no
    property evaluation worth the name — so the plan-cache lookup + insert is
    the largest fixed term that is *not* the write itself. If the writer's
    cache traffic is ever skipped, this is the cell that should move first.
    """
    graph = create_graphs[size]

    def write():
        graph.cypher("CREATE (:Note {body: 'x'})")

    benchmark.pedantic(write, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_repeated_set_write(benchmark, set_graphs, size):
    """A checkpointed write, repeated with identical text.

    Same cache traffic, much larger denominator — this is the ratio an
    application sees, and it is here so that a change measured on the ``CREATE``
    cell is not mistaken for an application-visible one.
    """
    graph = set_graphs[size]

    def write():
        graph.cypher("MATCH (n:Item {id: 0}) SET n.qty = 7")

    benchmark.pedantic(write, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_repeated_read_cache_hit(benchmark, read_graphs, size):
    """The control: a point lookup on an unchanged graph — a cache **hit**.

    Every round but the first skips parse, validate and optimize entirely and
    ``Arc``-clones a stored plan. Any policy change that touches the lookup path
    is bounded by this cell.
    """
    graph = read_graphs[size]

    def read():
        graph.cypher("MATCH (n:Item {id: 0}) RETURN n.name")

    benchmark.pedantic(read, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_repeated_read_uncacheable_control(benchmark, read_graphs, size):
    """The same read, made **uncacheable** by one unused parameter.

    ⚠ This cell is the only thing standing between this file and a completely
    vacuous read control. ``prepare`` refuses to cache a statement with a
    non-empty param map, so this is the identical query on the identical graph
    with the plan cache switched off — nothing else differs.

    It must come out **slower** than
    ``test_bench_repeated_read_cache_hit`` at the same size. If the two ever
    converge, the "hit" cell is not hitting (a param the binding injects, a
    codec, a disabled pass — any of which would make the cell measure the miss
    path under a label promising the opposite), and every conclusion drawn from
    it is void. Measured 2026-08-09 at 1k: 1.79 us cacheable vs 2.54 us with a
    param, so the cache is worth ~0.75 us on this shape.
    """
    graph = read_graphs[size]

    def read():
        graph.cypher("MATCH (n:Item {id: 0}) RETURN n.name", params={"unused_probe": 1})

    benchmark.pedantic(read, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)
