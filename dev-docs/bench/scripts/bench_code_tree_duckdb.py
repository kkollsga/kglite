"""Manual smoke benchmark: clone duckdb and build its KnowledgeGraph.

Not run in CI. Expected to print wall-clock time and node/edge totals
so the numbers can go into the release notes.

Usage::

    git clone --depth 1 https://github.com/duckdb/duckdb /tmp/duckdb
    python bench/bench_code_tree_duckdb.py /tmp/duckdb
"""

from __future__ import annotations

import sys
import time

from kglite.code_tree import build


def main(src: str) -> None:
    t0 = time.perf_counter()
    graph = build(src, verbose=True)
    elapsed = time.perf_counter() - t0

    nodes = graph.cypher("MATCH (n) RETURN labels(n)[0] AS t, count(*) AS c ORDER BY c DESC").to_list()
    edges = graph.cypher("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC").to_list()

    print()
    print(f"Elapsed: {elapsed:.2f}s")
    print("\nNodes by type:")
    for row in nodes:
        print(f"  {row['t']:14}{row['c']:>10}")
    print("\nEdges by type:")
    for row in edges:
        print(f"  {row['t']:14}{row['c']:>10}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: bench_code_tree_duckdb.py <src-dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
