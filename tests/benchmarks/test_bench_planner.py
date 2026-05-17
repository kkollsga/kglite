"""Cypher planner benchmarks.

Covers the 0.9.35 label-pair selectivity branch of `reorder_match_clauses`.
The fixture is intentionally label-skewed (one rare label + one common
one) so the new cost branch picks a meaningfully different driving side
than the old "sum of edge-type totals" proxy.

Numbers ride alongside the existing core benchmarks under
`make bench-save` / `make bench-compare`.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


def _build_skewed_graph(n_common: int = 5_000, n_rare: int = 50) -> KnowledgeGraph:
    """Two label classes (`Common`, `Rare`) joined by a shared `LINKS` edge type.

    Common: n_common nodes, all interconnected (n_common * 3 LINKS edges).
    Rare:   n_rare nodes, each linked to a single Common (n_rare LINKS edges).

    Result: total LINKS = n_common*3 + n_rare. Per-label-pair:
      (Common, LINKS, Common) = 3 * n_common
      (Rare,   LINKS, Common) = n_rare

    Querying `MATCH (a:Rare)-[:LINKS]->(b:Common)` should drive from Rare;
    with only the edge-type total, the planner sees `LINKS = ~15050` and
    can't distinguish.
    """
    kg = KnowledgeGraph()
    common = pd.DataFrame({"cid": list(range(n_common)), "name": [f"C{i}" for i in range(n_common)]})
    kg.add_nodes(common, "Common", "cid", "name")
    rare = pd.DataFrame({"rid": list(range(n_rare)), "name": [f"R{i}" for i in range(n_rare)]})
    kg.add_nodes(rare, "Rare", "rid", "name")

    # Common→Common edges (the bulk of the LINKS population)
    cc_src = []
    cc_tgt = []
    for i in range(n_common):
        for delta in (1, 7, 13):
            cc_src.append(i)
            cc_tgt.append((i + delta) % n_common)
    kg.add_connections(
        pd.DataFrame({"src": cc_src, "tgt": cc_tgt}),
        "LINKS",
        "Common",
        "src",
        "Common",
        "tgt",
    )

    # Rare→Common edges (the selective slice the planner should pick)
    rc = pd.DataFrame(
        {
            "src": list(range(n_rare)),
            "tgt": [(i * 91) % n_common for i in range(n_rare)],
        }
    )
    kg.add_connections(rc, "LINKS", "Rare", "src", "Common", "tgt")

    # Warm the planner caches so the first-bench-iteration cost isn't a
    # one-off O(E) scan. Mirrors how `make bench-compare` invokes
    # production queries that have already warmed these caches.
    kg.label_pair_counts()
    kg.cypher("MATCH (r:Rare)-[:LINKS]->(c:Common) RETURN count(*)").to_list()
    return kg


@pytest.fixture
def skewed_graph():
    return _build_skewed_graph()


@pytest.mark.benchmark
def test_bench_label_pair_counts_compute(benchmark):
    """Cold-cache compute of the label-pair triples. Captures the O(E)
    walk that the planner amortises into one-time cost per mutation
    epoch. Builds a fresh graph each round so the cache is always cold."""

    def setup():
        kg = _build_skewed_graph(n_common=2000, n_rare=20)
        # Invalidate so each timed iteration hits the cold path.
        kg.cypher("MATCH (a:Common {cid: 0}), (b:Common {cid: 1}) CREATE (a)-[:LINKS]->(b)").to_list()
        return (kg,), {}

    benchmark.pedantic(lambda kg: kg.label_pair_counts(), setup=setup, rounds=5, iterations=1)


@pytest.mark.benchmark
def test_bench_planner_two_match_with_skewed_labels(benchmark, skewed_graph):
    """Two MATCH clauses where one is selective on `Rare` and the other
    on `Common`. The new selectivity branch should drive from the Rare
    side. Measures end-to-end query latency with optimiser on."""
    g = skewed_graph
    benchmark(
        lambda: g.cypher(
            "MATCH (r:Rare {rid: 0})-[:LINKS]->(c:Common) "
            "MATCH (r2:Rare {rid: 1})-[:LINKS]->(c2:Common) "
            "RETURN c.name, c2.name"
        ).to_list()
    )


@pytest.mark.benchmark
def test_bench_label_pair_counts_warm_read(benchmark, skewed_graph):
    """Warm-cache read — must be O(triples), essentially free. If this
    is more than a few microseconds something has regressed in the
    Arc<RwLock<Option<...>>> read path."""
    g = skewed_graph
    benchmark(lambda: g.label_pair_counts())
