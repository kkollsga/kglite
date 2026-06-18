"""Concurrency-path benchmarks — the cost model behind the Session pyclass.

The Session concurrency story (snapshot reads + serialized CoW writes) has
exactly one non-trivial cost: a write that overlaps an outstanding read
snapshot cannot mutate in place (``Arc::make_mut`` sees refcount > 1) and so
**deep-clones the whole graph** before mutating. This is inherent to snapshot
isolation — every MVCC store pays it under a long-lived reader — but the
"in-memory is the gate" rule means we capture the number rather than wave at
it.

These benchmarks isolate that cost:

- ``write_no_snapshot`` — the baseline mutation path: no snapshot held, so
  ``make_mut`` mutates in place (refcount 1). This is what an existing
  single-owner ``KnowledgeGraph`` user pays today and must NOT regress.
- ``write_under_snapshot`` — a fresh ``freeze()`` is taken before each write
  so the graph is shared (refcount > 1) and the write deep-clones. The delta
  vs the baseline is the deep-clone cost (≈ O(graph size); transient memory
  ≈ 2× the graph footprint).
- ``read_snapshot`` — a read against a held ``freeze()`` snapshot, the
  lock-free read path the Session hands out. Baseline for read latency.

Run with: pytest tests/benchmarks/test_bench_concurrency.py -m benchmark
(release build only — ``maturin develop --release``).
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

N = 50_000


@pytest.fixture
def cc_graph():
    """50k nodes / 100k edges — large enough that a full deep-clone is
    clearly visible against a single-node mutation."""
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(N)),
            "name": [f"Node_{i}" for i in range(N)],
            "value": [float(i) for i in range(N)],
            "bucket": [i % 100 for i in range(N)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")
    edges = pd.DataFrame(
        {
            "from_id": [i % N for i in range(2 * N)],
            "to_id": [(i * 7 + 13) % N for i in range(2 * N)],
        }
    )
    graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id")
    return graph


# ---------------------------------------------------------------------------
# Write path: in-place vs deep-clone-under-snapshot
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_write_no_snapshot(benchmark, cc_graph):
    """Mutation with no outstanding snapshot → in-place make_mut.

    This is the existing single-owner write cost. The no-regression gate:
    Session must not change this number for code that doesn't use Session.
    """
    counter = {"n": 0}

    def mutate():
        counter["n"] += 1
        cc_graph.cypher(
            "MATCH (n:Item {nid: 0}) SET n.touched = $v",
            params={"v": counter["n"]},
        )

    benchmark(mutate)


@pytest.mark.benchmark
def test_write_under_snapshot(benchmark, cc_graph):
    """Mutation while a fresh freeze() snapshot is held → deep-clone path.

    A new freeze() each round keeps the graph shared (refcount > 1) so every
    write deep-clones. Delta vs test_write_no_snapshot = the clone cost.
    """
    counter = {"n": 0}

    def mutate_under_snapshot():
        counter["n"] += 1
        snapshot = cc_graph.freeze()  # refcount bump → next write must clone
        cc_graph.cypher(
            "MATCH (n:Item {nid: 0}) SET n.touched = $v",
            params={"v": counter["n"]},
        )
        # keep the snapshot alive across the mutation, then drop it
        del snapshot

    benchmark(mutate_under_snapshot)


# ---------------------------------------------------------------------------
# Read path: lock-free read against a held snapshot
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_read_snapshot(benchmark, cc_graph):
    """Read against a held freeze() snapshot — the lock-free read path."""
    snapshot = cc_graph.freeze()

    def read():
        return snapshot.cypher("MATCH (n:Item) WHERE n.bucket = 7 RETURN count(n) AS c")

    benchmark(read)
