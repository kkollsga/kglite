"""
Benchmark: Default (heap) vs Mapped (mmap) vs Disk (CSR) storage modes.
"""

import gc
import os
import sys
import tempfile
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kglite import KnowledgeGraph, load


def make_node_df(n, n_props=5):
    data = {"nid": list(range(n)), "name": [f"Node_{i}" for i in range(n)]}
    for p in range(n_props):
        if p % 3 == 0:
            data[f"int_prop_{p}"] = [i * (p + 1) for i in range(n)]
        elif p % 3 == 1:
            data[f"float_prop_{p}"] = [float(i) * 1.1 for i in range(n)]
        else:
            data[f"str_prop_{p}"] = [f"val_{i}_{p}" for i in range(n)]
    return pd.DataFrame(data)


def make_edge_df(n_nodes, n_edges):
    return pd.DataFrame(
        {
            "from_id": [i % n_nodes for i in range(n_edges)],
            "to_id": [(i * 7 + 3) % n_nodes for i in range(n_edges)],
        }
    )


def bench(label, fn):
    gc.collect()
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    return result, t1 - t0


def build_graph(mode, node_df, edge_df):
    if mode == "mapped":
        g = KnowledgeGraph(storage="mapped")
    else:
        g = KnowledgeGraph()
    g.add_nodes(node_df, "Item", "nid", "name")
    g.add_connections(edge_df, "LINKS", "Item", "from_id", "Item", "to_id")
    if mode == "disk":
        g.enable_disk_mode()
    return g


def dir_size(path):
    """Total size of all files in a directory tree."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


MODES = ["default", "default+col", "mapped", "disk"]


def run_suite(n_nodes, n_edges, rows):
    node_df = make_node_df(n_nodes, n_props=8)
    edge_df = make_edge_df(n_nodes, n_edges)
    threshold = n_nodes // 2

    for mode in MODES:

        def add_row(op, t):
            rows.append((n_nodes, mode, op, t))

        # Build (includes enable_disk_mode for disk)
        _, t = bench("build", lambda: build_graph(mode, node_df, edge_df))
        add_row("Build graph", t)

        g = build_graph(mode, node_df, edge_df)

        # Count
        _, t = bench("count", lambda: g.cypher("MATCH (n:Item) RETURN count(n)").to_df())
        add_row("Cypher COUNT", t)

        # Filter
        _, t = bench(
            "filter",
            lambda: g.cypher(f"MATCH (n:Item) WHERE n.int_prop_0 > {threshold} RETURN n.name LIMIT 100").to_df(),
        )
        add_row("Cypher WHERE+LIMIT", t)

        # Aggregation
        _, t = bench(
            "agg",
            lambda: g.cypher("MATCH (n:Item) RETURN avg(n.float_prop_1) AS a, max(n.int_prop_0) AS m").to_df(),
        )
        add_row("Cypher AVG/MAX", t)

        # 1-hop
        _, t = bench(
            "1hop",
            lambda: g.cypher("MATCH (a:Item)-[:LINKS]->(b:Item) RETURN a.name, b.name LIMIT 100").to_df(),
        )
        add_row("Cypher 1-hop LIMIT", t)

        # 2-hop
        _, t = bench(
            "2hop",
            lambda: g.cypher(
                "MATCH (a:Item)-[:LINKS]->(b:Item)-[:LINKS]->(c:Item) RETURN a.name, c.name LIMIT 50"
            ).to_df(),
        )
        add_row("Cypher 2-hop LIMIT", t)

        # ORDER BY
        _, t = bench(
            "order",
            lambda: g.cypher("MATCH (n:Item) RETURN n.name, n.int_prop_0 ORDER BY n.int_prop_0 DESC LIMIT 50").to_df(),
        )
        add_row("Cypher ORDER BY", t)

        # Fluent select all
        _, t = bench("select", lambda: g.select("Item").to_df())
        add_row("Fluent select to_df", t)

        # Fluent where
        _, t = bench(
            "where",
            lambda: g.select("Item").where({"int_prop_0": (">", threshold)}).to_df(),
        )
        add_row("Fluent where to_df", t)

        # graph_info
        _, t = bench("info", lambda: g.graph_info())
        add_row("graph_info()", t)

        # Save + Load
        with tempfile.TemporaryDirectory() as tmpdir:
            if mode == "disk":
                path = os.path.join(tmpdir, "bench_disk")
            else:
                path = os.path.join(tmpdir, "bench.kgl")
            _, t = bench("save", lambda: g.save(path))
            add_row("Save", t)

            if mode == "disk":
                fsize = dir_size(path)
            else:
                fsize = os.path.getsize(path)
            add_row("Save size (bytes)", fsize)

            _, t = bench("load", lambda: load(path))
            add_row("Load", t)

        # SET
        _, t = bench(
            "set",
            lambda: g.cypher("MATCH (n:Item) WHERE n.int_prop_0 < 10 SET n.int_prop_0 = n.int_prop_0 + 1"),
        )
        add_row("Cypher SET (update)", t)

        # DELETE
        g2 = build_graph(mode, node_df, edge_df)
        _, t = bench(
            "del",
            lambda: g2.cypher(f"MATCH (n:Item) WHERE n.int_prop_0 > {n_nodes - 11} DETACH DELETE n"),
        )
        add_row("Cypher DETACH DELETE", t)


def main():
    sizes = [
        (1_000, 3_000),
        (10_000, 30_000),
        (100_000, 300_000),
    ]

    print("KGLite Storage Mode Benchmark (4 modes)")
    print("=" * 60)

    rows = []
    for n, e in sizes:
        print(f"\n--- {n:,} nodes, {e:,} edges ---")
        run_suite(n, e, rows)

    # Print results
    from collections import defaultdict

    grouped = defaultdict(dict)
    for n, mode, op, t in rows:
        grouped[(n, op)][mode] = t

    w = 155
    print("\n" + "=" * w)
    hdr = (
        f"{'SIZE':>10} | {'OPERATION':<25}"
        f" | {'DEFAULT':>12} | {'DEF+COL':>12}"
        f" | {'MAPPED':>12} | {'DISK':>12}"
        f" | {'MAP/DEF':>8} | {'DISK/DEF':>8}"
        f" | {'FASTEST':>10}"
    )
    print(hdr)
    print("-" * w)

    prev = None
    for (size, op), modes in sorted(grouped.items()):
        if prev is not None and size != prev:
            print("-" * w)
        prev = size

        td = modes.get("default", 0)
        tc = modes.get("default+col", 0)
        tm = modes.get("mapped", 0)
        tk = modes.get("disk", 0)

        if "size" in op.lower():
            print(
                f"{size:>10,} | {op:<25}"
                f" | {td:>12,.0f} | {tc:>12,.0f}"
                f" | {tm:>12,.0f} | {tk:>12,.0f}"
                f" | {'':>8} | {'':>8} | {'':>10}"
            )
            continue

        rm = tm / td if td > 0 else float("nan")
        rk = tk / td if td > 0 else float("nan")

        candidates = [
            (td, "DEFAULT"),
            (tc, "DEF+COL"),
            (tm, "MAPPED"),
            (tk, "DISK"),
        ]
        best = min(candidates, key=lambda x: x[0])

        print(
            f"{size:>10,} | {op:<25}"
            f" | {td:>12.6f} | {tc:>12.6f}"
            f" | {tm:>12.6f} | {tk:>12.6f}"
            f" | {rm:>7.2f}x | {rk:>7.2f}x"
            f" | {best[1]:>10}"
        )

    print("=" * w)
    print("\nMAP/DEF = mapped/default, DISK/DEF = disk/default. <1 means faster than default.\n")


if __name__ == "__main__":
    main()
