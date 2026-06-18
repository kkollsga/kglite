"""
Benchmark: compare standard, mapped, and disk storage modes on 5M Wikidata triples.

Usage: python examples/benchmark_storage_modes.py
"""

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kglite import KnowledgeGraph, load

NT_FILE = "/Volumes/EksternalHome/Data/Wikidata/test_5M.nt.zst"
TEMP_DIR = "/tmp/kglite_bench_modes"

# Enable build sub-step debug output (CSR step timings, etc.)
os.environ["KGLITE_BUILD_DEBUG"] = "1"

CYPHER_QUERIES = [
    ("Count nodes", "MATCH (n) RETURN count(n) AS c"),
    ("Count Entity edges", "MATCH (n:Entity)-[r]->() RETURN count(r) AS c"),
    ("Count Entity", "MATCH (n:Entity) RETURN count(n) AS c"),
    ("Entity titles L10", "MATCH (n:Entity) RETURN n.title LIMIT 10"),
    ("Entity edges L10", "MATCH (n:Entity)-[r]->(m) RETURN n.title, type(r), m.title LIMIT 10"),
    ("Entity -[:P31]-> L10", "MATCH (n:Entity)-[:P31]->(m) RETURN n.title, m.title LIMIT 10"),
    ("Incoming to Entity L10", "MATCH (n)-[r]->(m:Entity) RETURN n.title, type(r) LIMIT 10"),
    ("2-hop L10", "MATCH (a:Entity)-[]->(b)-[]->(c) RETURN a.title, b.title, c.title LIMIT 10"),
    ("WHERE id=42", "MATCH (n) WHERE id(n) = 42 RETURN n.title, labels(n)"),
    ("CONTAINS Einstein", "MATCH (n) WHERE n.title CONTAINS 'Einstein' RETURN n.title LIMIT 10"),
]


def build_and_benchmark(mode):
    """Build a graph in the given mode, run queries, return timing dict."""
    results = {"mode": mode}
    graph_dir = os.path.join(TEMP_DIR, f"graph_{mode}")

    # Clean up previous
    if os.path.exists(graph_dir):
        shutil.rmtree(graph_dir)

    # Build
    print(f"\n{'=' * 60}", flush=True)
    print(f"  MODE: {mode}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if mode == "disk":
        os.makedirs(graph_dir, exist_ok=True)
        g = KnowledgeGraph(storage="disk", path=graph_dir)
    elif mode == "mapped":
        os.makedirs(graph_dir, exist_ok=True)
        g = KnowledgeGraph(storage="mapped", path=graph_dir)
    else:
        g = KnowledgeGraph()

    # Build with 2-minute timeout
    import signal

    class BuildTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise BuildTimeout()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(120)
    try:
        t0 = time.perf_counter()
        g.load_ntriples(NT_FILE, languages=["en"], verbose=True)
        results["build_time"] = time.perf_counter() - t0
        signal.alarm(0)
    except BuildTimeout:
        results["build_time"] = 120.0
        results["skipped"] = True
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        print(f"  BUILD TIMED OUT after 120s — skipping tests for {mode}", flush=True)
        results["nodes"] = 0
        results["edges"] = 0
        results["save_time"] = 0
        results["load_time"] = 0
        results["disk_mb"] = 0
        results["cypher"] = {label: 0 for label, _ in CYPHER_QUERIES}
        results["cypher_total"] = 0
        results["fluent"] = {}
        results["fluent_total"] = 0
        return results
    signal.signal(signal.SIGALRM, old_handler)

    info = g.graph_info()
    results["nodes"] = info.get("node_count", 0)
    results["edges"] = info.get("edge_count", 0)
    print(
        f"  Built: {results['nodes']:,} nodes, {results['edges']:,} edges in {results['build_time']:.1f}s", flush=True
    )

    # Save + reload
    if mode == "disk":
        # Disk mode: already saved during build, just reload
        t0 = time.perf_counter()
        g.save(graph_dir)
        results["save_time"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        g = load(graph_dir)
        results["load_time"] = time.perf_counter() - t0
    else:
        # Standard and mapped: save to file, reload
        save_path = os.path.join(TEMP_DIR, f"graph_{mode}.kglite")
        t0 = time.perf_counter()
        g.save(save_path)
        results["save_time"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        g = load(save_path)
        results["load_time"] = time.perf_counter() - t0
    print(f"  Save: {results['save_time']:.2f}s, Load: {results['load_time']:.2f}s", flush=True)

    # Disk size
    if mode == "disk":
        total_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, files in os.walk(graph_dir) for f in files)
    else:
        save_path = os.path.join(TEMP_DIR, f"graph_{mode}.kglite")
        total_bytes = os.path.getsize(save_path) if os.path.exists(save_path) else 0
    results["disk_mb"] = total_bytes / (1024 * 1024)
    print(f"  Disk: {results['disk_mb']:.1f} MB", flush=True)

    # Cypher queries (30s timeout each)
    print(f"\n  {'Query':35s} {'Time':>8s}  {'Rows':>6s}", flush=True)
    print(f"  {'-' * 55}", flush=True)
    cypher_times = {}
    for label, query in CYPHER_QUERIES:
        t0 = time.perf_counter()
        try:
            df = g.cypher(query, timeout_ms=30000).to_df()
            rows = len(df)
        except Exception as e:
            rows = f"ERR: {str(e)[:30]}"
        elapsed = time.perf_counter() - t0
        cypher_times[label] = elapsed
        r = f"{rows:,}" if isinstance(rows, int) else rows
        print(f"  {label:35s} {elapsed:>7.3f}s  {r:>6s}", flush=True)
    results["cypher"] = cypher_times
    results["cypher_total"] = sum(cypher_times.values())

    # Fluent API
    print(f"\n  {'Fluent':35s} {'Time':>8s}  {'Result':>8s}", flush=True)
    print(f"  {'-' * 55}", flush=True)
    fluent_times = {}

    fluent_ops = [
        ("select(Entity).len()", lambda: g.select("Entity").len()),
        ("traverse P31 out L100", lambda: g.select("Entity").traverse("P31", direction="outgoing", limit=100).len()),
        ("traverse P31 in L50", lambda: g.select("Entity").traverse("P31", direction="incoming", limit=50).len()),
        (
            "where title contains Berlin",
            lambda: g.select("Entity").where_({"title": ("contains", "Berlin")}, limit=20).len(),
        ),
        ("describe() length", lambda: len(g.describe())),
    ]

    for label, fn in fluent_ops:
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as e:
            result = f"ERR: {str(e)[:30]}"
        elapsed = time.perf_counter() - t0
        fluent_times[label] = elapsed
        r = f"{result:,}" if isinstance(result, int) else str(result)[:20]
        print(f"  {label:35s} {elapsed:>7.3f}s  {r:>8s}", flush=True)
    results["fluent"] = fluent_times
    results["fluent_total"] = sum(fluent_times.values())

    return results


def print_comparison(all_results):
    modes = [r["mode"] for r in all_results]

    print(f"\n\n{'=' * 80}")
    print("  COMPARISON TABLE — 5M Wikidata triples")
    print(f"{'=' * 80}\n")

    # Build / Save / Load / Disk
    print(f"  {'Metric':35s}", end="")
    for m in modes:
        print(f" {m:>12s}", end="")
    print()
    print(f"  {'-' * 35}", end="")
    for _ in modes:
        print(f" {'-' * 12}", end="")
    print()

    for key, label, fmt in [
        ("nodes", "Nodes", lambda v: f"{v:>12,}"),
        ("edges", "Edges", lambda v: f"{v:>12,}"),
        ("build_time", "Build (s)", lambda v: f"{v:>11.2f}s"),
        ("save_time", "Save (s)", lambda v: f"{v:>11.2f}s"),
        ("load_time", "Load (s)", lambda v: f"{v:>11.2f}s"),
        ("disk_mb", "Disk (MB)", lambda v: f"{v:>11.1f}M"),
    ]:
        print(f"  {label:35s}", end="")
        for r in all_results:
            print(f" {fmt(r.get(key, 0))}", end="")
        print()

    # Cypher queries
    print(f"\n  {'Cypher Query':35s}", end="")
    for m in modes:
        print(f" {m:>12s}", end="")
    print()
    print(f"  {'-' * 35}", end="")
    for _ in modes:
        print(f" {'-' * 12}", end="")
    print()

    for label, _ in CYPHER_QUERIES:
        print(f"  {label:35s}", end="")
        for r in all_results:
            t = r["cypher"].get(label, 0)
            print(f" {t:>11.3f}s", end="")
        print()
    print(f"  {'TOTAL':35s}", end="")
    for r in all_results:
        print(f" {r['cypher_total']:>11.3f}s", end="")
    print()

    # Fluent
    print(f"\n  {'Fluent API':35s}", end="")
    for m in modes:
        print(f" {m:>12s}", end="")
    print()
    print(f"  {'-' * 35}", end="")
    for _ in modes:
        print(f" {'-' * 12}", end="")
    print()

    for label in all_results[0]["fluent"]:
        print(f"  {label:35s}", end="")
        for r in all_results:
            t = r["fluent"].get(label, 0)
            print(f" {t:>11.3f}s", end="")
        print()
    print(f"  {'TOTAL':35s}", end="")
    for r in all_results:
        print(f" {r['fluent_total']:>11.3f}s", end="")
    print()


if __name__ == "__main__":
    os.makedirs(TEMP_DIR, exist_ok=True)

    all_results = []
    for mode in ["standard", "mapped", "disk"]:
        results = build_and_benchmark(mode)
        all_results.append(results)

    print_comparison(all_results)
