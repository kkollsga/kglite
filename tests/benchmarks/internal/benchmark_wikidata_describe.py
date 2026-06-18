"""Benchmark describe() + rebuild_caches() on wikidata disk graph.

Records detailed timings from each stage and writes results to CSV.
Run: python examples/benchmark_wikidata_describe.py

Output: benchmark_wikidata_describe.csv
"""

import csv
import os
import sys
import time

import kglite

WIKIDATA_PATH = "/Volumes/EksternalHome/Data/Wikidata/wikidata_disk_graph"
CSV_PATH = "benchmark_wikidata_describe.csv"


def timed(label, fn):
    """Run fn, print and return (label, duration, result)."""
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    return label, dt, result


def main():
    if not os.path.exists(WIKIDATA_PATH):
        print(f"Wikidata graph not found at {WIKIDATA_PATH}")
        return 1

    rows = []

    def record(stage, duration, detail=""):
        rows.append({"stage": stage, "seconds": f"{duration:.3f}", "detail": detail})
        print(f"  {stage}: {duration:.3f}s  {detail}")

    print("=" * 70)
    print("  Wikidata Describe Benchmark")
    print("=" * 70)

    # ── Load ──
    t0 = time.time()
    g = kglite.load(WIKIDATA_PATH)
    record("load", time.time() - t0)

    schema = g.schema()
    record(
        "schema",
        0,
        f"nodes={schema['node_count']:,} edges={schema['edge_count']:,} "
        f"types={len(schema['node_types'])} conn_types={len(schema['connection_types'])}",
    )

    # ── describe() cold cache ──
    print("\n-- describe() cold cache --")
    label, dt, desc = timed("describe_cold", lambda: g.describe())
    record(label, dt, f"chars={len(desc):,}")

    # ── rebuild_caches() — the single O(E) pass ──
    print("\n-- rebuild_caches() --")
    label, dt, _ = timed("rebuild_caches", lambda: g.rebuild_caches())
    record(label, dt, "single O(E) pass: type connectivity + edge counts + endpoints")

    # ── describe() warm cache ──
    print("\n-- describe() warm cache --")
    label, dt, desc = timed("describe_warm", lambda: g.describe())
    record(label, dt, f"chars={len(desc):,}")

    # ── type_search warm ──
    print("\n-- type_search warm cache --")
    for term in ["software", "galaxy", "human", "article", "painting"]:
        label, dt, ts = timed(f"type_search_{term}", lambda t=term: g.describe(type_search=t))
        record(label, dt, f"chars={len(ts):,}")

    # ── describe(types=[...]) warm ──
    print("\n-- describe(types=[...]) warm cache --")
    for t in ["human", "software", "street", "taxon"]:
        label, dt, td = timed(f"types_{t}", lambda t=t: g.describe(types=[t]))
        record(label, dt, f"chars={len(td):,}")

    # ── describe(connections=True) ──
    print("\n-- describe(connections=True) --")
    label, dt, dc = timed("connections_overview", lambda: g.describe(connections=True))
    record(label, dt, f"chars={len(dc):,}")

    # ── Write CSV ──
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "seconds", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 70}")
    print(f"  Results written to {CSV_PATH}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
