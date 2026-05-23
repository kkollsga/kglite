"""Pre/post-fix performance audit harness.

Re-runnable script that captures the Bolt-relevant micro-benchmarks
(beyond the tracked pytest-benchmark suite) into a JSON file so we
can produce clean before/after diffs.

Usage:
    python scripts/perf_audit.py before > .perf_audit_before.json
    python scripts/perf_audit.py after  > .perf_audit_after.json
    python scripts/perf_audit.py diff   # reads both files, prints table

The bench function reports min, median, mean across N samples after
warmup. Sub-millisecond benches use rigorous round counts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import pandas as pd

import kglite


def bench(fn, *, n: int = 200, warmup: int = 20) -> dict:
    gc.collect()
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return {
        "min_ns": min(samples),
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
        "n": n,
    }


def fmt(ns: float) -> str:
    if ns < 1000:
        return f"{ns:.0f}ns"
    if ns < 1e6:
        return f"{ns / 1000:.1f}µs"
    return f"{ns / 1e6:.2f}ms"


def _build_perf_graph(size: int) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(size)),
                "title": [f"P{i}" for i in range(size)],
                "age": [20 + (i % 60) for i in range(size)],
                "city": [["Oslo", "Bergen", "Trondheim", "Stavanger"][i % 4] for i in range(size)],
                "salary": [50000 + (i * 100) for i in range(size)],
            }
        ),
        "Person",
        "id",
        "title",
    )
    # 10% edges
    e = size // 10
    if e > 0:
        g.add_connections(
            pd.DataFrame(
                {
                    "src": [i for i in range(e)],
                    "tgt": [(i * 7) % size for i in range(e)],
                }
            ),
            "KNOWS",
            "Person",
            "src",
            "Person",
            "tgt",
        )
    return g


def collect() -> dict:
    out: dict = {"timestamp": time.time()}

    # ── Tier 1: begin() clone cost vs graph size ────────────────────
    sz_results = {}
    for size in (1_000, 10_000, 100_000):
        g = _build_perf_graph(size)
        sz_results[f"begin_commit_{size}"] = bench(
            lambda: (lambda tx: (tx, tx.commit()))(g.begin()),
            n=50,
            warmup=5,
        )
    out["begin_clone_scaling"] = sz_results

    # ── Tier 2: per-call cypher() cost (parse + execute baseline) ──
    g = _build_perf_graph(1_000)
    out["cypher_calls"] = {
        "return_1": bench(lambda: g.cypher("RETURN 1 AS n")),
        "match_1row_indexed": bench(lambda: g.cypher("MATCH (p:Person {id: 1}) RETURN p.title")),
        "match_inside_begin_read": bench(
            lambda: (
                (tx := g.begin_read()),
                tx.cypher("MATCH (p:Person {id: 1}) RETURN p.title"),
                tx.commit(),
            )
        ),
        "match_inside_begin_rw": bench(
            lambda: (
                (tx := g.begin()),
                tx.cypher("MATCH (p:Person {id: 1}) RETURN p.title"),
                tx.commit(),
            ),
            n=50,  # slower because of clone
            warmup=5,
        ),
        "begin_read_commit_noop": bench(lambda: (lambda tx: (tx, tx.commit()))(g.begin_read())),
        "begin_rw_commit_noop_1k": bench(
            lambda: (lambda tx: (tx, tx.commit()))(g.begin()),
            n=50,
            warmup=5,
        ),
    }

    # ── Tier 2: parse cache effectiveness ──────────────────────────
    queries_unique = [f"MATCH (p:Person {{id: {i}}}) RETURN p.title" for i in range(100)]
    counter = [0]

    def unique_query():
        q = queries_unique[counter[0] % 100]
        counter[0] += 1
        g.cypher(q)

    out["parse_cache_test"] = {
        "same_query_repeat": bench(lambda: g.cypher("MATCH (p:Person {id: 500}) RETURN p.title")),
        "unique_query_each": bench(unique_query),
    }

    # ── Tier 2: Concurrent read scaling ────────────────────────────
    big = _build_perf_graph(5_000)
    QUERY = "MATCH (p:Person) WHERE p.city = 'Oslo' AND p.age > 30 RETURN count(p)"

    def one_query():
        big.cypher(QUERY)

    # Warmup
    for _ in range(50):
        one_query()

    def measure_threads(threads: int) -> dict:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futs = [pool.submit(one_query) for _ in range(threads * 100)]
                for f in futs:
                    f.result()
            times.append(time.perf_counter() - t0)
        wall = min(times)
        return {
            "wall_seconds": wall,
            "total_queries": threads * 100,
            "per_query_ns": wall / (threads * 100) * 1e9,
        }

    # Sequential baseline
    t0 = time.perf_counter()
    for _ in range(100):
        one_query()
    seq_100 = time.perf_counter() - t0
    out["concurrent_scaling"] = {"sequential_100q_seconds": seq_100}
    for t in (1, 2, 4, 8, 16, 32):
        out["concurrent_scaling"][f"threads_{t}"] = measure_threads(t)

    # ── Tier 3: result serialization ───────────────────────────────
    g100k = _build_perf_graph(100_000)
    ser = {}
    for n_rows in (1, 10, 100, 1000, 10000):
        q = f"MATCH (p:Person) RETURN p.title, p.age, p.city LIMIT {n_rows}"
        ser[f"limit_{n_rows}"] = bench(lambda q=q: g100k.cypher(q), n=50, warmup=5)
    out["result_serialization"] = ser

    # ── Tier 3: full-scan throughput ───────────────────────────────
    out["full_scan_100k"] = bench(
        lambda: g100k.cypher("MATCH (p:Person) WHERE p.salary > 5000000 RETURN count(p)"),
        n=20,
        warmup=3,
    )

    # ── Tier 3: columnar_enable ────────────────────────────────────
    g_col = _build_perf_graph(1_000)
    # Warmup
    for _ in range(5):
        g_col.enable_columnar()
        g_col.disable_columnar()

    en_samples, dis_samples = [], []
    for _ in range(50):
        g_col.disable_columnar()
        t0 = time.perf_counter_ns()
        g_col.enable_columnar()
        en_samples.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        g_col.disable_columnar()
        dis_samples.append(time.perf_counter_ns() - t0)
    out["columnar_cycle"] = {
        "enable": {
            "min_ns": min(en_samples),
            "median_ns": statistics.median(en_samples),
            "mean_ns": statistics.mean(en_samples),
            "n": len(en_samples),
        },
        "disable": {
            "min_ns": min(dis_samples),
            "median_ns": statistics.median(dis_samples),
            "mean_ns": statistics.mean(dis_samples),
            "n": len(dis_samples),
        },
    }

    return out


def diff(before: dict, after: dict) -> None:
    """Print a delta table."""

    def walk(prefix: str, b: dict, a: dict, rows: list):
        for k in sorted(set(b.keys()) | set(a.keys())):
            if k == "timestamp":
                continue
            bv, av = b.get(k), a.get(k)
            if isinstance(bv, dict) and isinstance(av, dict):
                if "min_ns" in bv:
                    rows.append(
                        (
                            f"{prefix}.{k}",
                            bv["min_ns"],
                            av["min_ns"],
                            bv.get("median_ns"),
                            av.get("median_ns"),
                        )
                    )
                else:
                    walk(f"{prefix}.{k}", bv, av, rows)
            elif isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                rows.append(
                    (
                        f"{prefix}.{k}",
                        bv * 1e9 if "seconds" in k else bv,
                        av * 1e9 if "seconds" in k else av,
                        None,
                        None,
                    )
                )

    rows: list = []
    walk("", before, after, rows)

    print(f"{'Metric':<55} {'before':>12} {'after':>12} {'Δ min':>8} {'Δ med':>8}")
    print("-" * 105)
    for name, b_min, a_min, b_med, a_med in rows:
        if b_min is None or a_min is None or b_min == 0:
            continue
        d_min = (a_min - b_min) / b_min * 100
        if b_med is not None and a_med is not None and b_med != 0:
            d_med = (a_med - b_med) / b_med * 100
            d_med_str = f"{d_med:+.1f}%"
        else:
            d_med_str = "—"
        flag = ""
        if d_min < -20:
            flag = "  🟢"
        elif d_min > 20:
            flag = "  🔴"
        print(f"{name:<55} {fmt(b_min):>12} {fmt(a_min):>12} {d_min:+7.1f}% {d_med_str:>8}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["before", "after", "diff"])
    args = ap.parse_args()

    if args.mode == "diff":
        before_path = Path(".perf_audit_before.json")
        after_path = Path(".perf_audit_after.json")
        if not before_path.exists() or not after_path.exists():
            print("Missing .perf_audit_before.json or .perf_audit_after.json", file=sys.stderr)
            sys.exit(1)
        diff(json.loads(before_path.read_text()), json.loads(after_path.read_text()))
    else:
        data = collect()
        Path(f".perf_audit_{args.mode}.json").write_text(json.dumps(data, indent=2))
        print(f"Captured {args.mode} → .perf_audit_{args.mode}.json")


if __name__ == "__main__":
    main()
