#!/usr/bin/env python3
"""Sustained hot-path workload for sampling profilers (samply / py-spy).

Runs the Phase 3 suspect query shapes in a tight loop for ~20s each of
wall time per shape, on the same 50k-node graph the hot-path benchmark
suite uses (tests/benchmarks/test_bench_hotpaths.py), so profile
rankings map 1:1 onto benchmark names.

Usage:
    samply record --save-only -o /tmp/kglite_hotpaths.json \
        python scripts/profile_hotpaths.py [shape ...]

With no args, all shapes run. Pass shape names (distinct, return_n,
unwind, where, topk) to profile a subset.
"""

import sys
import time

import pandas as pd

from kglite import KnowledgeGraph

N = 50_000
SECONDS_PER_SHAPE = 20.0


def build_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(N)),
            "name": [f"Node_{i}" for i in range(N)],
            "high_card": [f"hc_{i % (N // 2)}" for i in range(N)],
            "mid_card": [f"mc_{i % 1000}" for i in range(N)],
            "low_card": [f"lc_{i % 10}" for i in range(N)],
            "value": [float(i) for i in range(N)],
            "rank_val": [(i * 7919) % N for i in range(N)],
            "flag": [i % 2 == 0 for i in range(N)],
            "p1": [f"a{i % 97}" for i in range(N)],
            "p2": [f"b{i % 89}" for i in range(N)],
            "p3": [float(i % 83) for i in range(N)],
            "p4": [i % 79 for i in range(N)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")
    edges = pd.DataFrame(
        {
            "from_id": [i % N for i in range(2 * N)],
            "to_id": [(i * 7 + 13) % N for i in range(2 * N)],
            "weight": [float(i % 100) for i in range(2 * N)],
        }
    )
    graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])
    return graph


SHAPES = {
    "distinct": "MATCH (n:Item) RETURN count(DISTINCT n.high_card)",
    "return_n": "MATCH (n:Item) WHERE n.rank_val < 5000 RETURN n",
    "unwind": ("MATCH (n:Item) WHERE n.rank_val < 5000 UNWIND [1,2,3,4,5,6,7,8,9,10] AS x RETURN count(x)"),
    "where": ("MATCH (n:Item) WHERE n.value > 25000.0 AND n.p4 < 40 AND n.flag RETURN count(n)"),
    "topk": "MATCH (n:Item) RETURN n.name, n.value ORDER BY n.value DESC LIMIT 25",
    "distinct_rows": "MATCH (n:Item) RETURN DISTINCT n.mid_card, n.low_card",
}


def main() -> None:
    wanted = sys.argv[1:] or list(SHAPES)
    unknown = [w for w in wanted if w not in SHAPES]
    if unknown:
        sys.exit(f"unknown shape(s) {unknown}; available: {list(SHAPES)}")

    print(f"building {N}-node graph ...", flush=True)
    graph = build_graph()

    for name in wanted:
        query = SHAPES[name]
        # Warm up so one-time costs (index builds, caches) don't pollute
        # the sampled window.
        for _ in range(3):
            graph.cypher(query)
        print(f"shape {name}: ~{SECONDS_PER_SHAPE:.0f}s of {query!r}", flush=True)
        iterations = 0
        start = time.perf_counter()
        while time.perf_counter() - start < SECONDS_PER_SHAPE:
            graph.cypher(query)
            iterations += 1
        elapsed = time.perf_counter() - start
        print(f"  {iterations} iterations, {elapsed / iterations * 1e3:.2f} ms/query", flush=True)


if __name__ == "__main__":
    main()
