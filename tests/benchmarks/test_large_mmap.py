"""Large-graph stress test for v3 columnar file format.

Creates a graph that approaches or exceeds available RAM, saves it as a v3
.kgl file, reloads it, and measures query performance on the loaded graph.

Usage (standalone — not part of normal test suite):
    python tests/benchmarks/test_large_mmap.py                 # default ~2 GB
    python tests/benchmarks/test_large_mmap.py --target-gb 8   # 8 GB
    python tests/benchmarks/test_large_mmap.py --target-gb 20  # exceed 16 GB RAM
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import kglite  # noqa: E402
from tests.conftest import rss_mb  # noqa: E402

# -- Helpers -------------------------------------------------------------------


def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def fmt_mb(mb):
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def section(title):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# -- Constants -----------------------------------------------------------------

# Each node has ~20 string properties averaging ~30 chars each.
# Estimated per-node footprint: ~800 bytes raw data + overhead ~ 1 KB in compact,
# less in columnar.  So 1M nodes ~ 1 GB, 2M ~ 2 GB, etc.
PROPS_PER_NODE = 20
BATCH_SIZE = 100_000
EDGE_RATIO = 2  # edges per node


# -- Build ---------------------------------------------------------------------


def build_graph(target_gb):
    """Build a large KnowledgeGraph in batches, returning (graph, stats)."""
    target_nodes = int(target_gb * 1_000_000)  # ~1 GB per 1M nodes
    num_batches = max(1, target_nodes // BATCH_SIZE)
    total_nodes = num_batches * BATCH_SIZE
    total_edges = total_nodes * EDGE_RATIO

    section(f"Building graph: {total_nodes:,} nodes, {total_edges:,} edges  (target ~ {target_gb} GB)")

    kg = kglite.KnowledgeGraph()

    # -- Nodes --
    t0 = time.perf_counter()
    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        ids = list(range(start, end))

        data = {"nid": ids, "name": [f"Node_{i}" for i in ids]}
        for p in range(PROPS_PER_NODE):
            data[f"prop_{p}"] = [f"val_{p}_{i % 10000}_{i}" for i in ids]

        df = pd.DataFrame(data)
        kg.add_nodes(df, "Item", "nid", "name")

        elapsed = time.perf_counter() - t0
        pct = (batch_idx + 1) / num_batches * 100
        print(
            f"  Nodes: batch {batch_idx + 1}/{num_batches}  ({pct:.0f}%)  {fmt_time(elapsed)}  RSS={fmt_mb(rss_mb())}",
            end="\r",
        )
    print()

    node_time = time.perf_counter() - t0
    print(f"  Nodes done: {total_nodes:,} in {fmt_time(node_time)}  RSS={fmt_mb(rss_mb())}")

    # -- Edges --
    t1 = time.perf_counter()
    edge_batches = max(1, total_edges // BATCH_SIZE)
    for batch_idx in range(edge_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total_edges)
        count = end - start
        from_ids = [i % total_nodes for i in range(start, end)]
        to_ids = [(i * 7 + 13) % total_nodes for i in range(start, end)]
        weights = [float(i % 1000) for i in range(count)]

        edf = pd.DataFrame({"src": from_ids, "dst": to_ids, "weight": weights})
        kg.add_connections(edf, "LINKS", "Item", "src", "Item", "dst", columns=["weight"])

        elapsed = time.perf_counter() - t1
        pct = (batch_idx + 1) / edge_batches * 100
        print(
            f"  Edges: batch {batch_idx + 1}/{edge_batches}  ({pct:.0f}%)  {fmt_time(elapsed)}  RSS={fmt_mb(rss_mb())}",
            end="\r",
        )
    print()

    edge_time = time.perf_counter() - t1
    print(f"  Edges done: {total_edges:,} in {fmt_time(edge_time)}  RSS={fmt_mb(rss_mb())}")

    stats = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "node_time": node_time,
        "edge_time": edge_time,
        "build_rss_mb": rss_mb(),
    }
    return kg, stats


# -- Save / Load ---------------------------------------------------------------


def save_and_load(kg, kgl_path):
    results = {}

    # -- save v3 .kgl --
    section("Saving v3 .kgl file")
    t0 = time.perf_counter()
    kg.save(kgl_path)
    results["save_kgl_time"] = time.perf_counter() - t0
    results["kgl_size_mb"] = os.path.getsize(kgl_path) / (1024 * 1024)
    print(f"  save(): {fmt_time(results['save_kgl_time'])}")
    print(f"  File size: {fmt_mb(results['kgl_size_mb'])}")

    # -- load v3 .kgl --
    section("Loading from v3 .kgl file")
    t0 = time.perf_counter()
    kg2 = kglite.load(kgl_path)
    results["load_kgl_time"] = time.perf_counter() - t0
    print(f"  load(): {fmt_time(results['load_kgl_time'])}  RSS={fmt_mb(rss_mb())}")
    print(f"  is_columnar: {kg2.is_columnar}")

    return kg2, results


# -- Queries -------------------------------------------------------------------


def run_queries(kg, label, total_nodes):
    """Run a battery of queries and return timing dict."""
    section(f"Queries on {label}")
    timings = {}

    queries = [
        ("point lookup (id=42)", "MATCH (n:Item) WHERE n.nid = 42 RETURN n.name, n.prop_0"),
        ("range scan (nid > N-100)", f"MATCH (n:Item) WHERE n.nid > {total_nodes - 100} RETURN n.name"),
        ("string filter", "MATCH (n:Item) WHERE n.prop_0 STARTS WITH 'val_0_0_' RETURN count(n) AS cnt"),
        ("aggregation", "MATCH (n:Item) RETURN count(n) AS cnt"),
        ("traversal", "MATCH (a:Item)-[:LINKS]->(b:Item) WHERE a.nid = 0 RETURN b.name LIMIT 10"),
        ("LIMIT 1000", "MATCH (n:Item) RETURN n.name, n.prop_0, n.prop_5 LIMIT 1000"),
    ]

    for name, query in queries:
        # Warm-up
        kg.cypher(query)

        # Timed run (3 iterations, take min)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = kg.cypher(query)
            # Force materialisation
            _ = result.to_list()
            times.append(time.perf_counter() - t0)

        best = min(times)
        timings[name] = best
        print(f"  {name:40s} {fmt_time(best):>10s}")

    return timings


# -- Disk-lookup stress (cold reads) ------------------------------------------


def disk_lookup_test(kgl_path, total_nodes):
    """Load v3 graph and do random-ish point lookups to exercise disk paging."""
    section("Disk lookup stress test (cold reads after fresh load)")

    # Fresh load — pages are cold
    kg = kglite.load(kgl_path)
    lookup_ids = [i * (total_nodes // 50) for i in range(50)]  # 50 evenly-spaced lookups

    t0 = time.perf_counter()
    for nid in lookup_ids:
        result = kg.cypher(f"MATCH (n:Item) WHERE n.nid = {nid} RETURN n.name, n.prop_0, n.prop_10").to_list()
        assert len(result) == 1, f"Expected 1 result for nid={nid}, got {len(result)}"
    elapsed = time.perf_counter() - t0

    per_lookup = elapsed / len(lookup_ids)
    print(f"  {len(lookup_ids)} point lookups: {fmt_time(elapsed)} total, {fmt_time(per_lookup)} avg")
    print(f"  RSS after lookups: {fmt_mb(rss_mb())}")
    return {"cold_lookups": len(lookup_ids), "cold_total": elapsed, "cold_avg": per_lookup}


# -- Summary -------------------------------------------------------------------


def print_summary(build_stats, io_results, query_timings, disk_stats):
    section("SUMMARY")
    n = build_stats["total_nodes"]
    e = build_stats["total_edges"]

    print(f"  Graph:          {n:>12,} nodes, {e:>12,} edges")
    print(f"  Build time:     {fmt_time(build_stats['node_time'] + build_stats['edge_time']):>12s}")
    print(f"  Peak RSS:       {fmt_mb(build_stats['build_rss_mb']):>12s}")
    print()
    print(f"  {'Save time':40s} {fmt_time(io_results['save_kgl_time']):>12s}")
    print(f"  {'File size':40s} {fmt_mb(io_results['kgl_size_mb']):>12s}")
    print(f"  {'Load time':40s} {fmt_time(io_results['load_kgl_time']):>12s}")
    print()

    print(f"  {'Query':40s} {'v3 columnar':>12s}")
    for name, t in query_timings.items():
        print(f"  {name:40s} {fmt_time(t):>12s}")

    print()
    print(f"  Cold disk lookups:  {disk_stats['cold_lookups']} lookups in {fmt_time(disk_stats['cold_total'])}")
    print(f"  Cold avg per lookup: {fmt_time(disk_stats['cold_avg'])}")


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Large-graph v3 stress test")
    parser.add_argument(
        "--target-gb",
        type=float,
        default=2.0,
        help="Target graph size in GB (default: 2.0)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep temp files after completion (default: clean up)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory for output (default: temp dir)",
    )
    args = parser.parse_args()

    print(f"Target: ~{args.target_gb} GB graph on {os.cpu_count()} cores, RSS baseline={fmt_mb(rss_mb())}")

    tmpdir = args.dir or tempfile.mkdtemp(prefix="kglite_stress_")
    kgl_path = os.path.join(tmpdir, "graph.kgl")

    try:
        # Build
        kg, build_stats = build_graph(args.target_gb)

        # Save & load (v3 auto-enables columnar)
        kg_loaded, io_results = save_and_load(kg, kgl_path)

        # Free the original graph to reclaim memory
        del kg

        # Query performance on v3-loaded graph
        query_timings = run_queries(kg_loaded, "v3-loaded graph", build_stats["total_nodes"])

        # Free loaded graph
        del kg_loaded

        # Disk lookup stress (fresh load, cold pages)
        disk_stats = disk_lookup_test(kgl_path, build_stats["total_nodes"])

        # Summary
        print_summary(build_stats, io_results, query_timings, disk_stats)

    finally:
        if not args.keep and not args.dir:
            print(f"\n  Cleaning up {tmpdir}")
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"\n  Files kept at: {tmpdir}")


if __name__ == "__main__":
    main()
