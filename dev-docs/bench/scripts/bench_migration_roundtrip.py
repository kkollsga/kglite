"""Migration round-trip across the existing pre-built Wikidata-slice graphs.

For each graph:
  1. Load (legacy `*.bin.zst` format if present)
  2. Run the cypher query suite — capture results as the baseline
  3. Save in-place (emits new `id_indices.bin` and `type_indices.bin`)
  4. Reload (now uses mmap-resident format)
  5. Re-run the cypher suite — assert results identical to baseline
  6. Time everything

Reports per-graph: load_legacy_ms, save_ms, load_new_ms, query times before/after,
plus a result-equality check that flags any divergence.

Run each graph in a fresh subprocess so RSS / page-cache state don't leak.

Usage:
    python bench/bench_migration_roundtrip.py
    python bench/bench_migration_roundtrip.py --graphs /Volumes/EksternalHome/Data/Wikidata/graph_5.0,/Volumes/EksternalHome/Data/Wikidata/graph_50.0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path("/Volumes/EksternalHome/Data/Wikidata")

# Default selection — small to medium. Skip graph_500.0+ unless asked.
DEFAULT_GRAPHS = [
    DATA_DIR / "graph_0.5",
    DATA_DIR / "graph_1.0",
    DATA_DIR / "graph_5.0",
    DATA_DIR / "graph_50.0",
    DATA_DIR / "graph_100.0",
]

# Anchored queries — should run in tens of ms on small graphs. The benchmark
# compares result_count + first-row content before save vs after reload.
QUERIES = [
    ("count_all", "MATCH (n) RETURN count(n) AS c"),
    ("count_by_type", "MATCH (n) RETURN labels(n)[0] AS t, count(n) AS c ORDER BY c DESC LIMIT 5"),
    ("sample_titles", "MATCH (n) RETURN n.title AS t ORDER BY t LIMIT 10"),
    ("edge_count", "MATCH ()-[r]->() RETURN count(r) AS c"),
    (
        "edge_types",
        "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC LIMIT 5",
    ),
]

CHILD_SCRIPT = r"""
import json, os, sys, time, kglite

graph_path = sys.argv[1]
queries = json.loads(sys.argv[2])

def run_queries(g, label):
    out = {}
    for name, q in queries:
        t0 = time.perf_counter()
        try:
            rows = list(g.cypher(q))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out[name] = {
                "ok": True,
                "ms": round(elapsed_ms, 2),
                "row_count": len(rows),
                # Stringify rows so they compare cleanly (Value objects don't pickle)
                "first_row": str(rows[0]) if rows else None,
            }
        except Exception as e:
            out[name] = {"ok": False, "error": str(e)[:200]}
    return out

# Phase 1: load (legacy or new — whatever's on disk)
t0 = time.perf_counter()
g = kglite.load(graph_path)
load_legacy_ms = (time.perf_counter() - t0) * 1000.0

baseline = run_queries(g, "before_save")

# Phase 2: save in place (emits new format)
t0 = time.perf_counter()
g.save(graph_path)
save_ms = (time.perf_counter() - t0) * 1000.0

# Drop the in-process graph so reload starts cold-ish (page cache stays warm)
del g

# Phase 3: reload (now uses new mmap format)
t0 = time.perf_counter()
g = kglite.load(graph_path)
load_new_ms = (time.perf_counter() - t0) * 1000.0

after = run_queries(g, "after_reload")

# Diff query results
divergences = []
for name, _ in queries:
    b = baseline.get(name, {})
    a = after.get(name, {})
    if b.get("row_count") != a.get("row_count"):
        divergences.append(f"{name}: row_count {b.get('row_count')} -> {a.get('row_count')}")
    elif b.get("first_row") != a.get("first_row"):
        divergences.append(f"{name}: first_row diverged")

# RSS at end of process (peak)
import resource
rss_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    rss_mb = rss_b / 1024 / 1024
else:
    rss_mb = rss_b / 1024

print("__BENCH_RESULT__=" + json.dumps({
    "load_legacy_ms": round(load_legacy_ms, 1),
    "save_ms": round(save_ms, 1),
    "load_new_ms": round(load_new_ms, 1),
    "rss_peak_mb": round(rss_mb, 1),
    "queries_before": baseline,
    "queries_after": after,
    "divergences": divergences,
}))
"""


def fmt_ms(ms):
    if ms is None:
        return "    —"
    if ms >= 1000:
        return f"{ms / 1000:6.2f} s"
    return f"{ms:6.0f} ms"


def fmt_bytes(b):
    if b is None:
        return "    —"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:6.1f} {unit}"
        b /= 1024
    return f"{b:6.1f} TB"


def run_one(graph_path: Path) -> dict:
    """Run the round-trip for one graph; returns the parsed JSON result."""
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT, str(graph_path), json.dumps(QUERIES)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"child failed (exit {proc.returncode}) on {graph_path}")
    for line in proc.stdout.splitlines():
        if line.startswith("__BENCH_RESULT__="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"no __BENCH_RESULT__ line in stdout for {graph_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="Comma-separated paths; default = pre-built test slices under /Volumes/.../Wikidata/",
    )
    args = p.parse_args()

    if args.graphs:
        graphs = [Path(g.strip()) for g in args.graphs.split(",") if g.strip()]
    else:
        graphs = [g for g in DEFAULT_GRAPHS if (g / "disk_graph_meta.json").exists()]

    if not graphs:
        print("error: no graphs to test", file=sys.stderr)
        return 2

    print(f"testing {len(graphs)} graphs:")
    for g in graphs:
        try:
            with (g / "disk_graph_meta.json").open() as f:
                meta = json.load(f)
            print(
                f"  {g.name:<20} {meta['node_count']:>10} nodes, {meta['edge_count']:>10} edges"
            )
        except Exception:
            print(f"  {g.name:<20} ?")

    print()
    print(
        f"{'graph':<20} {'load_legacy':>12} {'save':>10} {'load_new':>10} {'speedup':>8} {'RSS':>10} {'regressions':>12}"
    )
    print("-" * 100)

    overall_regressions = []
    rows = []
    for g in graphs:
        t0 = time.perf_counter()
        try:
            r = run_one(g)
        except Exception as e:
            print(f"{g.name:<20} ERROR: {e}")
            continue
        wall = (time.perf_counter() - t0) * 1000.0
        speedup = (
            r["load_legacy_ms"] / r["load_new_ms"] if r["load_new_ms"] > 0 else float("inf")
        )
        regs = len(r["divergences"])
        if regs:
            overall_regressions.append((g.name, r["divergences"]))
        print(
            f"{g.name:<20} "
            f"{fmt_ms(r['load_legacy_ms']):>12} "
            f"{fmt_ms(r['save_ms']):>10} "
            f"{fmt_ms(r['load_new_ms']):>10} "
            f"{speedup:>7.1f}x "
            f"{r['rss_peak_mb']:>7.1f} MB "
            f"{regs:>12}"
        )
        rows.append((g.name, r))

    print()
    if overall_regressions:
        print(f"REGRESSIONS in {len(overall_regressions)} graphs:")
        for name, divs in overall_regressions:
            print(f"  {name}:")
            for d in divs:
                print(f"    - {d}")
        return 1
    else:
        print("All graphs pass roundtrip with identical query results.")

    # Per-query timing breakdown
    print()
    print("per-query timings (median across graphs, before save / after reload):")
    print(f"{'query':<20} {'before_ms':>10} {'after_ms':>10}  {'change':>10}")
    print("-" * 60)
    for name, _ in QUERIES:
        before_times = [
            r["queries_before"].get(name, {}).get("ms")
            for _, r in rows
            if r["queries_before"].get(name, {}).get("ok")
        ]
        after_times = [
            r["queries_after"].get(name, {}).get("ms")
            for _, r in rows
            if r["queries_after"].get(name, {}).get("ok")
        ]
        if not before_times or not after_times:
            continue
        before_med = sorted(before_times)[len(before_times) // 2]
        after_med = sorted(after_times)[len(after_times) // 2]
        change = after_med - before_med
        sign = "+" if change >= 0 else ""
        print(
            f"{name:<20} {before_med:>9.2f} ms {after_med:>9.2f} ms  {sign}{change:>7.2f} ms"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
