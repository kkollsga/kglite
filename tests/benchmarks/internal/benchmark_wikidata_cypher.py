"""Wikidata Cypher benchmark — tests all optimized code paths and known pain points.

Loads the disk-backed Wikidata graph and runs queries across categories:
  1. Basics         — ID lookups, fused counts (should be instant)
  2. Anchored hops  — single/multi-hop from known entity (our sweet spot)
  3. LIMIT          — tests LIMIT push-down on large types
  4. Edge-type      — tests sorted CSR binary search
  5. Aggregation    — tests fused aggregation paths
  6. Property scan  — tests direct columnar access (dangerous on disk)
  7. Hub nodes      — high-degree fan-in/out (Q5, Q515, etc.)
  8. Pain points    — queries expected to be slow or fail (future work)

Each query has a 20s timeout. Failures are recorded, not fatal.

Run:  python benchmarks/benchmark_wikidata_cypher.py
"""

import csv
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from kglite import load

GRAPH_DIR = "/Volumes/EksternalHome/Data/Wikidata/wikidata_disk_graph_v2"
SCRIPT_DIR = Path(__file__).parent
CSV_OUT = str(SCRIPT_DIR / "benchmark_wikidata_cypher.csv")
TIMEOUT_MS = 20_000  # 20s per query


# ═══════════════════════════════════════════════════════════════════
# Benchmark definitions: (name, category, query)
# ═══════════════════════════════════════════════════════════════════

BENCHMARKS = [
    # ── 1. BASICS — instant queries ─────────────────────────────────
    # Note: Wikidata Q-numbers are stored as UniqueId in the `id` field.
    # Use {id: 42} for Q42 (O(1) index lookup), NOT {nid: 'Q42'} (full scan).
    ("count_all_nodes", "basics", "MATCH (n) RETURN count(n) AS c"),
    ("count_by_type", "basics", "MATCH (n) RETURN n.type, count(n) AS c ORDER BY c DESC LIMIT 20"),
    ("count_edges_by_type", "basics", "MATCH ()-[r]->() RETURN type(r), count(*)"),
    ("lookup_Q42", "basics", "MATCH (n {id: 42}) RETURN n.title, n.description"),
    ("lookup_Q64", "basics", "MATCH (n {id: 64}) RETURN n.title, n.description"),
    ("lookup_Q76", "basics", "MATCH (n {id: 76}) RETURN n.title, n.description"),
    ("lookup_Q5", "basics", "MATCH (n {id: 5}) RETURN n.title, n.description"),
    ("lookup_Q515", "basics", "MATCH (n {id: 515}) RETURN n.title, n.description"),
    # ── 2. ANCHORED HOPS — the MCP sweet spot ──────────────────────
    ("Q42_outgoing", "anchored", "MATCH ({id: 42})-[r]->(m) RETURN type(r), m.title LIMIT 20"),
    ("Q42_incoming", "anchored", "MATCH (m)-[r]->({id: 42}) RETURN type(r), m.title LIMIT 20"),
    ("Q76_outgoing", "anchored", "MATCH ({id: 76})-[r]->(m) RETURN type(r), m.title LIMIT 20"),
    ("Q42_P31", "anchored", "MATCH ({id: 42})-[:P31]->(m) RETURN m.title"),
    ("Q64_P17", "anchored", "MATCH ({id: 64})-[:P17]->(m) RETURN m.title"),
    ("Q42_to_Q5_direct", "anchored", "MATCH ({id: 42})-[r]->({id: 5}) RETURN type(r)"),
    ("Q42_2hop", "anchored", "MATCH ({id: 42})-[]->(b)-[]->(c) RETURN b.title, c.title LIMIT 20"),
    ("Q76_2hop", "anchored", "MATCH ({id: 76})-[]->(b)-[]->(c) RETURN b.title, c.title LIMIT 20"),
    ("Q42_optional", "anchored", "MATCH (n {id: 42}) OPTIONAL MATCH (n)-[r]->(m) RETURN type(r), m.title LIMIT 30"),
    # ── 3. LIMIT PUSH-DOWN — should be fast on any type ────────────
    ("limit_10_P31", "limit", "MATCH (a)-[:P31]->(b) RETURN a.title, b.title LIMIT 10"),
    ("limit_50_P31", "limit", "MATCH (a)-[:P31]->(b) RETURN a.title, b.title LIMIT 50"),
    ("limit_100_any_edge", "limit", "MATCH (a)-[r]->(b) RETURN a.title, type(r), b.title LIMIT 100"),
    ("limit_10_2hop", "limit", "MATCH (a)-[:P31]->(b)-[:P279]->(c) RETURN a.title, c.title LIMIT 10"),
    # ── 4. EDGE-TYPE FILTERING (sorted CSR binary search) ──────────
    # Q5 "human" — massive P31 hub. Binary search should skip millions.
    ("Q5_incoming_P31_lim50", "edge_type", "MATCH (n)-[:P31]->({id: 5}) RETURN n.title LIMIT 50"),
    ("Q515_incoming_P31_lim50", "edge_type", "MATCH (n)-[:P31]->({id: 515}) RETURN n.title LIMIT 50"),
    ("Q6256_incoming_P31", "edge_type", "MATCH (n)-[:P31]->({id: 6256}) RETURN n.title LIMIT 50"),
    ("Q5_count_P31_incoming", "edge_type", "MATCH (n)-[:P31]->({id: 5}) RETURN count(n) AS humans"),
    ("Q515_count_P31", "edge_type", "MATCH (n)-[:P31]->({id: 515}) RETURN count(n) AS cities"),
    ("born_in_Berlin", "edge_type", "MATCH (p)-[:P19]->({id: 64}) RETURN p.title LIMIT 20"),
    ("citizens_of_Germany", "edge_type", "MATCH (p)-[:P27]->({id: 183}) RETURN p.title LIMIT 20"),
    # ── 5. AGGREGATION (fused paths) ───────────────────────────────
    # FusedMatchReturnAggregate: group by one node, count the other
    (
        "agg_P31_by_target",
        "aggregation",
        "MATCH (a)-[:P31]->(b) RETURN b.title, count(a) AS instances ORDER BY instances DESC LIMIT 20",
    ),
    (
        "agg_P27_by_country",
        "aggregation",
        "MATCH (p)-[:P27]->(c) RETURN c.title, count(p) AS citizens ORDER BY citizens DESC LIMIT 20",
    ),
    # FusedNodeScanAggregate with WHERE (Phase 3 optimization)
    ("agg_count_by_type_top20", "aggregation", "MATCH (n) RETURN n.type, count(n) AS c ORDER BY c DESC LIMIT 20"),
    # ── 6. PROPERTY SCANS (direct columnar access) ─────────────────
    # These scan large types filtering by property — tests our columnar fast path
    ("contains_Einstein", "prop_scan", "MATCH (n) WHERE n.title CONTAINS 'Einstein' RETURN n.id, n.title LIMIT 20"),
    ("contains_Norway", "prop_scan", "MATCH (n) WHERE n.title CONTAINS 'Norway' RETURN n.id, n.title LIMIT 20"),
    ("startswith_Albert", "prop_scan", "MATCH (n) WHERE n.title STARTS WITH 'Albert' RETURN n.id, n.title LIMIT 20"),
    # ── 7. HUB NODES — stress test high-degree vertices ────────────
    ("Q5_all_outgoing", "hub", "MATCH ({id: 5})-[r]->(m) RETURN type(r), m.title LIMIT 50"),
    ("Q5_all_incoming_lim100", "hub", "MATCH (m)-[r]->({id: 5}) RETURN type(r), m.title LIMIT 100"),
    ("Q515_all_incoming_lim100", "hub", "MATCH (m)-[r]->({id: 515}) RETURN type(r), m.title LIMIT 100"),
    # ── 8. PAIN POINTS — expected slow, measuring for future work ──
    # Full type scan without index — scans all entities
    ("full_scan_contains_2024", "pain", "MATCH (n) WHERE n.description CONTAINS '2024' RETURN n.id, n.title LIMIT 20"),
    # Unanchored expansion — no start anchor, relies on LIMIT
    ("unanchored_P31_count", "pain", "MATCH (a)-[:P31]->(b) RETURN count(*) AS total_P31"),
    # Variable-length path from hub
    ("varpath_Q42_1_2", "pain", "MATCH ({id: 42})-[*1..2]->(b) RETURN count(*) AS reachable"),
    # Cross-join risk (both typed, no anchor)
    ("cross_type_limited", "pain", "MATCH (a)-[:P31]->(b), (b)-[:P279]->(c) RETURN a.title, c.title LIMIT 10"),
    # Aggregation without fused path (HAVING)
    (
        "agg_having",
        "pain",
        "MATCH (a)-[:P31]->(b) RETURN b.title, count(a) AS cnt HAVING cnt > 100000 ORDER BY cnt DESC",
    ),
    # Hub incoming no type filter (all edge types)
    ("Q5_incoming_all_count", "pain", "MATCH (m)-[r]->({id: 5}) RETURN count(m) AS total"),
    # Multi-hop from hub
    ("Q5_2hop_out", "pain", "MATCH ({id: 5})-[]->(b)-[]->(c) RETURN b.title, c.title LIMIT 20"),
]


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════


def main():
    print("Loading Wikidata disk graph...")
    t0 = time.perf_counter()
    g = load(GRAPH_DIR)
    load_time = time.perf_counter() - t0
    info = g.graph_info()
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Nodes: {info.get('node_count', 0):,}")
    print(f"  Edges: {info.get('edge_count', 0):,}")
    print(f"  Types: {info.get('type_count', 0):,}")
    print()

    # Set default timeout
    g.set_default_timeout(TIMEOUT_MS)

    # Open CSV at start and flush per row so partial runs (SIGKILL, OOM, Ctrl-C)
    # still leave a populated CSV behind. The previous end-of-main writerows path
    # lost everything on crash.
    csv_file = open(CSV_OUT, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["name", "category", "status", "rows", "time_ms"])
    writer.writeheader()
    csv_file.flush()

    results = []
    current_category = None

    for name, category, query in BENCHMARKS:
        if category != current_category:
            current_category = category
            print(f"\n  ── {category.upper()} {'─' * (55 - len(category))}")

        t0 = time.perf_counter()
        try:
            result = g.cypher(query)
            elapsed = time.perf_counter() - t0
            rows = len(result)
            status = "ok"
            # Show first result value for verification
            preview = ""
            if rows > 0:
                first = result[0]
                vals = list(first.values())[:2]
                preview = str(vals[0])[:30] if vals else ""
        except Exception as e:
            elapsed = time.perf_counter() - t0
            err_str = str(e)
            if "timed out" in err_str.lower():
                status = "TIMEOUT"
                rows = 0
                preview = f">{TIMEOUT_MS / 1000:.0f}s"
            else:
                status = "ERROR"
                rows = 0
                preview = err_str[:40]

        row_record = {
            "name": name,
            "category": category,
            "status": status,
            "rows": rows,
            "time_ms": round(elapsed * 1000, 1),
        }
        results.append(row_record)
        writer.writerow(row_record)
        csv_file.flush()

        if status == "ok":
            time_str = f"{elapsed * 1000:>8.1f} ms"
        elif status == "TIMEOUT":
            time_str = f"{'TIMEOUT':>8s}   "
        else:
            time_str = f"{'ERROR':>8s}   "

        row_str = f"{rows:>6,}" if isinstance(rows, int) and rows > 0 else f"{'—':>6s}"
        print(f"    {name:36s} {time_str}  {row_str}  {preview}")

    csv_file.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"ok": 0, "timeout": 0, "error": 0, "total_ms": 0.0}
        if r["status"] == "ok":
            by_category[cat]["ok"] += 1
            by_category[cat]["total_ms"] += r["time_ms"]
        elif r["status"] == "TIMEOUT":
            by_category[cat]["timeout"] += 1
        else:
            by_category[cat]["error"] += 1

    print(f"\n  {'Category':15s} {'OK':>4s} {'TMO':>4s} {'ERR':>4s} {'Avg ms':>10s}")
    print("  " + "-" * 45)
    for cat, stats in sorted(by_category.items()):
        avg = f"{stats['total_ms'] / max(stats['ok'], 1):.1f}" if stats["ok"] else "—"
        print(f"  {cat:15s} {stats['ok']:>4d} {stats['timeout']:>4d} {stats['error']:>4d} {avg:>10s}")

    total_ok = sum(s["ok"] for s in by_category.values())
    total_tmo = sum(s["timeout"] for s in by_category.values())
    total_err = sum(s["error"] for s in by_category.values())
    print(f"  {'TOTAL':15s} {total_ok:>4d} {total_tmo:>4d} {total_err:>4d}")

    # CSV was streamed row-by-row during the benchmark (see earlier loop).
    print(f"\n  Results written to {CSV_OUT}")

    return total_tmo + total_err


if __name__ == "__main__":
    sys.exit(main())
