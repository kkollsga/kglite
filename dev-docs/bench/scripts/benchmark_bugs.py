"""Pre-bugfix performance baseline for BUG-01 through BUG-20.

Self-contained benchmark covering every Cypher engine code path affected
by the 20 bugs in TODO.md.  Two graph sizes (small 8-node, bench 1000-node)
ensure both correctness-path coverage and realistic load.

Run:  python bench/benchmark_bugs.py
"""

import csv
import os
from pathlib import Path
import re
import statistics
import time

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kw):
        return it


import kglite
from kglite import KnowledgeGraph

SCRIPT_DIR = Path(__file__).parent
CSV_OUT = str(SCRIPT_DIR / "benchmark_bugs.csv")

ITERATIONS = 10
WARMUP = 2
BENCH_ITERATIONS = 5
BENCH_WARMUP = 1


# ═══════════════════════════════════════════════════════════════════
# Graph builders  (exact replicas of test fixtures)
# ═══════════════════════════════════════════════════════════════════


def build_cypher_graph() -> KnowledgeGraph:
    """Replica of tests/test_cypher.py cypher_graph fixture."""
    g = KnowledgeGraph()

    people = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [30, 25, 35, 28, 40],
            "city": ["Oslo", "Bergen", "Oslo", "Bergen", "Oslo"],
            "salary": [70000, 55000, 80000, 65000, 90000],
            "email": ["alice@test.com", None, "charlie@test.com", None, "eve@test.com"],
        }
    )
    g.add_nodes(people, "Person", "person_id", "name")

    products = pd.DataFrame(
        {
            "product_id": [101, 102, 103],
            "name": ["Laptop", "Phone", "Tablet"],
            "price": [999.99, 699.99, 349.99],
        }
    )
    g.add_nodes(products, "Product", "product_id", "name")

    knows = pd.DataFrame({"from_id": [1, 1, 2, 3, 4], "to_id": [2, 3, 3, 4, 5]})
    g.add_connections(knows, "KNOWS", "Person", "from_id", "Person", "to_id")

    purchased = pd.DataFrame({"person_id": [1, 1, 2, 3], "product_id": [101, 102, 103, 101]})
    g.add_connections(purchased, "PURCHASED", "Person", "person_id", "Product", "product_id")

    return g


def build_bench_graph() -> KnowledgeGraph:
    """Replica of tests/benchmarks/test_bench_core.py bench_graph fixture."""
    g = KnowledgeGraph()

    nodes = pd.DataFrame(
        {
            "nid": list(range(1000)),
            "name": [f"Node_{i}" for i in range(1000)],
            "value": [float(i) for i in range(1000)],
            "category": [f"cat_{i % 10}" for i in range(1000)],
        }
    )
    g.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % 1000 for i in range(2000)],
            "to_id": [(i * 7 + 13) % 1000 for i in range(2000)],
            "weight": [float(i % 100) for i in range(2000)],
        }
    )
    g.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])

    return g


# ═══════════════════════════════════════════════════════════════════
# Timing harness
# ═══════════════════════════════════════════════════════════════════


def bench(fn, iterations=ITERATIONS, warmup=WARMUP):
    """Run fn() multiple times, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    return statistics.median(times)


# ═══════════════════════════════════════════════════════════════════
# Benchmark definitions  (name, category, graph_key, cypher_query)
# ═══════════════════════════════════════════════════════════════════

BENCHMARKS: list[tuple[str, str, str, str]] = [
    # ── A. Planner: WHERE pushdown + fusion (BUG-01, BUG-17) ──────
    ("planner_eq_pushdown_small", "planner", "small", "MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.title, n.age"),
    (
        "planner_eq_groupby_small",
        "planner",
        "small",
        "MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.city, count(*) AS cnt",
    ),
    (
        "planner_in_pushdown_small",
        "planner",
        "small",
        "MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen'] RETURN n.title",
    ),
    (
        "planner_contains_no_push_small",
        "planner",
        "small",
        "MATCH (n:Person) WHERE n.title CONTAINS 'li' RETURN n.title",
    ),
    ("planner_range_pushdown_small", "planner", "small", "MATCH (n:Person) WHERE n.age > 30 RETURN n.title, n.age"),
    (
        "planner_fusion_count_small",
        "planner",
        "small",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, count(b) AS friends",
    ),
    ("planner_fusion_count_only_small", "planner", "small", "MATCH (n:Person) RETURN count(*) AS total"),
    (
        "planner_pushdown_plus_fusion_small",
        "planner",
        "small",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.city = 'Oslo' RETURN a.title, count(b) AS friends",
    ),
    (
        "planner_eq_pushdown_bench",
        "planner",
        "bench",
        "MATCH (n:Item) WHERE n.category = 'cat_0' RETURN n.title, n.value",
    ),
    (
        "planner_eq_groupby_bench",
        "planner",
        "bench",
        "MATCH (n:Item) WHERE n.category = 'cat_0' RETURN n.category, count(*) AS cnt",
    ),
    ("planner_range_pushdown_bench", "planner", "bench", "MATCH (n:Item) WHERE n.value > 500 RETURN n.title, n.value"),
    (
        "planner_fusion_count_bench",
        "planner",
        "bench",
        "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN a.title, count(b) AS links",
    ),
    ("planner_fusion_count_only_bench", "planner", "bench", "MATCH (n:Item) RETURN count(*) AS total"),
    ("planner_unlabeled_type_eq_small", "planner", "small", "MATCH (n) WHERE n.type = 'Person' RETURN count(n) AS cnt"),
    # ── B. Top-K: ORDER BY + LIMIT (BUG-02) ──────────────────────
    (
        "topk_int_agg_small",
        "top_k",
        "small",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WITH a, count(b) AS friends RETURN a.title, friends ORDER BY friends DESC LIMIT 3",
    ),
    ("topk_float_small", "top_k", "small", "MATCH (n:Person) RETURN n.title, n.salary ORDER BY n.salary DESC LIMIT 3"),
    ("topk_no_limit_small", "top_k", "small", "MATCH (n:Person) RETURN n.title, n.age ORDER BY n.age DESC"),
    ("topk_float_bench", "top_k", "bench", "MATCH (n:Item) RETURN n.title, n.value ORDER BY n.value DESC LIMIT 10"),
    ("topk_float_bench_k50", "top_k", "bench", "MATCH (n:Item) RETURN n.title, n.value ORDER BY n.value DESC LIMIT 50"),
    ("topk_float_bench_k1", "top_k", "bench", "MATCH (n:Item) RETURN n.title, n.value ORDER BY n.value DESC LIMIT 1"),
    (
        "topk_agg_bench",
        "top_k",
        "bench",
        "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN a.title, count(b) AS links ORDER BY links DESC LIMIT 10",
    ),
    ("topk_no_limit_bench", "top_k", "bench", "MATCH (n:Item) RETURN n.title, n.value ORDER BY n.value"),
    # ── C. Aggregation + HAVING (BUG-03, BUG-07, BUG-18) ─────────
    ("agg_count_groupby_small", "aggregation", "small", "MATCH (n:Person) RETURN n.city AS city, count(*) AS cnt"),
    ("agg_sum_small", "aggregation", "small", "MATCH (n:Person) RETURN sum(n.salary) AS total"),
    ("agg_avg_small", "aggregation", "small", "MATCH (n:Person) RETURN avg(n.age) AS avg_age"),
    (
        "agg_min_max_small",
        "aggregation",
        "small",
        "MATCH (n:Person) RETURN min(n.age) AS youngest, max(n.age) AS oldest",
    ),
    (
        "agg_collect_small",
        "aggregation",
        "small",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, collect(b.title) AS friends",
    ),
    (
        "agg_having_small",
        "aggregation",
        "small",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, count(b) AS friends HAVING friends >= 2",
    ),
    (
        "agg_labels_groupby_small",
        "aggregation",
        "small",
        "MATCH (n) RETURN labels(n) AS lbl, count(n) AS cnt ORDER BY cnt DESC",
    ),
    (
        "agg_count_groupby_bench",
        "aggregation",
        "bench",
        "MATCH (n:Item) RETURN n.category, count(*) AS cnt ORDER BY cnt DESC",
    ),
    (
        "agg_sum_groupby_bench",
        "aggregation",
        "bench",
        "MATCH (n:Item) RETURN n.category, sum(n.value) AS total ORDER BY total DESC",
    ),
    ("agg_avg_bench", "aggregation", "bench", "MATCH (n:Item) RETURN avg(n.value) AS avg_val"),
    (
        "agg_collect_bench",
        "aggregation",
        "bench",
        "MATCH (n:Item) WHERE n.category = 'cat_0' RETURN collect(n.title) AS items",
    ),
    ("agg_having_bench", "aggregation", "bench", "MATCH (n:Item) RETURN n.category, count(*) AS cnt HAVING cnt > 90"),
    # ── D. EXISTS subquery (BUG-04) ───────────────────────────────
    (
        "exists_simple_small",
        "exists",
        "small",
        "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:KNOWS]->(other:Person) } RETURN p.title",
    ),
    (
        "exists_with_where_small",
        "exists",
        "small",
        "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:PURCHASED]->(pr:Product) WHERE pr.price > 500 } RETURN p.title",
    ),
    (
        "exists_negated_small",
        "exists",
        "small",
        "MATCH (p:Person) WHERE NOT EXISTS { MATCH (p)-[:PURCHASED]->(pr:Product) } RETURN p.title",
    ),
    (
        "exists_simple_bench",
        "exists",
        "bench",
        "MATCH (n:Item) WHERE EXISTS { MATCH (n)-[:LINKS]->(other:Item) } RETURN count(n) AS cnt",
    ),
    (
        "exists_with_where_bench",
        "exists",
        "bench",
        "MATCH (n:Item) WHERE EXISTS { MATCH (n)-[:LINKS]->(other:Item) WHERE other.value > 900 } RETURN count(n) AS cnt",
    ),
    # ── E. Expression evaluation (BUG-05, BUG-08-10, BUG-13-14, BUG-19) ──
    ("expr_return_star_small", "expression", "small", "MATCH (n:Person) WHERE n.name = 'Alice' RETURN *"),
    (
        "expr_arithmetic_small",
        "expression",
        "small",
        "MATCH (n:Person) RETURN n.title, n.salary * 1.1 AS raised, n.salary / 12 AS monthly",
    ),
    (
        "expr_string_functions_small",
        "expression",
        "small",
        "MATCH (n:Person) RETURN toUpper(n.title) AS upper, toLower(n.title) AS lower, substring(n.title, 0, 3) AS prefix",
    ),
    (
        "expr_coalesce_small",
        "expression",
        "small",
        "MATCH (n:Person) RETURN n.title, coalesce(n.email, 'none') AS contact",
    ),
    (
        "expr_case_small",
        "expression",
        "small",
        "MATCH (n:Person) RETURN n.title, CASE WHEN n.age > 35 THEN 'senior' ELSE 'junior' END AS level",
    ),
    ("expr_date_parse_small", "expression", "small", "RETURN date('2024-06-15') AS d"),
    (
        "expr_arithmetic_bench",
        "expression",
        "bench",
        "MATCH (n:Item) RETURN n.title, n.value * 2.0 AS doubled, n.value / 3.0 AS third LIMIT 100",
    ),
    (
        "expr_string_bench",
        "expression",
        "bench",
        "MATCH (n:Item) RETURN toUpper(n.title) AS upper, left(n.title, 5) AS prefix LIMIT 100",
    ),
    # ── F. Pattern matching (BUG-06, BUG-11) ─────────────────────
    ("pattern_1hop_small", "pattern", "small", "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, b.title"),
    (
        "pattern_2hop_explicit_small",
        "pattern",
        "small",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:PURCHASED]->(pr:Product) RETURN a.title, b.title, pr.title",
    ),
    (
        "pattern_2hop_path_var_small",
        "pattern",
        "small",
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:PURCHASED]->(pr:Product) RETURN length(p) AS hops LIMIT 1",
    ),
    ("pattern_varlen_small", "pattern", "small", "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.title, b.title"),
    ("pattern_inline_props_small", "pattern", "small", "MATCH (n:Person {city: 'Oslo'}) RETURN n.title"),
    ("pattern_1hop_bench", "pattern", "bench", "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN a.title, b.title LIMIT 100"),
    ("pattern_varlen_bench", "pattern", "bench", "MATCH (a:Item {id: 0})-[:LINKS*1..2]->(b:Item) RETURN b.title"),
    ("pattern_inline_props_bench", "pattern", "bench", "MATCH (n:Item {category: 'cat_0'}) RETURN n.title"),
    # ── G. Parser throughput (BUG-10, BUG-12, BUG-15, BUG-16, BUG-19, BUG-20) ──
    ("parser_simple_small", "parser", "small", "MATCH (n:Person) RETURN n.title"),
    (
        "parser_complex_where_small",
        "parser",
        "small",
        "MATCH (n:Person) WHERE n.age > 25 AND n.city = 'Oslo' AND n.salary > 60000 RETURN n.title, n.age",
    ),
    (
        "parser_multi_clause_small",
        "parser",
        "small",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WITH a, count(b) AS friends WHERE friends > 1 RETURN a.title, friends ORDER BY friends DESC LIMIT 5",
    ),
    (
        "parser_long_predicate_small",
        "parser",
        "small",
        "MATCH (n:Person) WHERE n.age > 20 AND n.age < 50 AND n.city = 'Oslo' AND n.salary > 50000 AND n.title STARTS WITH 'A' RETURN n.title",
    ),
    (
        "parser_unwind_match_small",
        "parser",
        "small",
        "UNWIND ['Alice', 'Bob', 'Charlie'] AS name MATCH (n:Person {name: name}) RETURN n.title, n.age",
    ),
    (
        "parser_case_complex_small",
        "parser",
        "small",
        "MATCH (n:Person) RETURN n.title, CASE WHEN n.age > 35 THEN 'senior' WHEN n.age > 28 THEN 'mid' ELSE 'junior' END AS level, CASE WHEN n.city = 'Oslo' THEN 'capital' ELSE 'other' END AS region",
    ),
    # ── H. Projection (BUG-05, BUG-16, BUG-20) ──────────────────
    (
        "proj_properties_small",
        "projection",
        "small",
        "MATCH (n:Person) RETURN n.title, n.age, n.city, n.salary, n.email",
    ),
    (
        "proj_map_literal_small",
        "projection",
        "small",
        "MATCH (n:Person) WHERE n.name = 'Alice' RETURN {name: n.title, age: n.age} AS info",
    ),
    (
        "proj_map_projection_small",
        "projection",
        "small",
        "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n {.title, .age, .city} AS info",
    ),
    ("proj_distinct_small", "projection", "small", "MATCH (n:Person) RETURN DISTINCT n.city"),
    ("proj_multi_bench", "projection", "bench", "MATCH (n:Item) RETURN n.title, n.value, n.category LIMIT 200"),
    ("proj_distinct_bench", "projection", "bench", "MATCH (n:Item) RETURN DISTINCT n.category"),
    # ── I. Core baselines (from test_bench_core.py) ───────────────
    ("core_match_bench", "core", "bench", "MATCH (n:Item) RETURN n.title, n.value LIMIT 100"),
    ("core_where_bench", "core", "bench", "MATCH (n:Item) WHERE n.value > 500 RETURN n.title, n.value"),
    (
        "core_shortest_path_bench",
        "core",
        "bench",
        "MATCH p = shortestPath((a:Item {id: 0})-[*]-(b:Item {id: 500})) RETURN length(p)",
    ),
]


# ═══════════════════════════════════════════════════════════════════
# CSV persistence  (column-append pattern from benchmark_sodir.py)
# ═══════════════════════════════════════════════════════════════════


def load_existing_csv() -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
    """Load existing CSV -> (benchmark_names, col_names, data)."""
    if not os.path.exists(CSV_OUT):
        return [], [], {}

    with open(CSV_OUT, newline="") as f:
        reader = csv.DictReader(f)
        bm_key = next((c for c in reader.fieldnames if c.startswith("benchmark")), "benchmark")
        col_names = [c for c in reader.fieldnames if c != bm_key]
        data: dict[str, dict[str, str]] = {}
        benchmark_names: list[str] = []
        for row in reader:
            bm = row[bm_key]
            benchmark_names.append(bm)
            data[bm] = {c: row[c] for c in col_names}

    return benchmark_names, col_names, data


def next_column_name(existing_cols: list[str], version: str) -> str:
    """Determine column name: v0.6.10_pre, v0.6.10_pre_2, ..."""
    base = f"v{version}_pre"
    if base not in existing_cols:
        return base

    pattern = re.compile(re.escape(base) + r"(?:_(\d+))?$")
    max_suffix = 1
    for col in existing_cols:
        m = pattern.match(col)
        if m:
            suffix = int(m.group(1)) if m.group(1) else 1
            max_suffix = max(max_suffix, suffix)

    return f"{base}_{max_suffix + 1}"


def save_to_csv(version: str, results: dict[str, float]):
    """Append results as a new column in benchmark_bugs.csv."""
    benchmark_names, col_names, data = load_existing_csv()

    new_col = next_column_name(col_names, version)
    col_names.append(new_col)

    for bm in results:
        if bm not in data:
            data[bm] = {}
            if bm not in benchmark_names:
                benchmark_names.append(bm)
        data[bm][new_col] = str(results[bm])

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark (ms)"] + col_names)
        for bm in benchmark_names:
            row = [bm] + [data.get(bm, {}).get(c, "") for c in col_names]
            writer.writerow(row)

    print(f"  Results written to {CSV_OUT}  (column: {new_col})")


# ═══════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════


def run_benchmarks():
    version = kglite.__version__
    results: dict[str, float] = {}
    errors: list[tuple[str, str]] = []

    print(f"KGLite v{version} — Bug-path performance baseline")
    print()

    # Build graphs
    print("  Building graphs...")
    t0 = time.perf_counter()
    small = build_cypher_graph()
    bench_g = build_bench_graph()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Graphs built in {build_ms:.0f} ms  (small: 8 nodes, bench: 1000 nodes)")
    print()

    graphs = {"small": small, "bench": bench_g}

    # Run benchmarks
    pbar = tqdm(BENCHMARKS, desc="Benchmarking", unit="bench", ncols=90)
    for name, category, graph_key, query in pbar:
        pbar.set_postfix_str(name, refresh=True)
        g = graphs[graph_key]
        iters = ITERATIONS if graph_key == "small" else BENCH_ITERATIONS
        wu = WARMUP if graph_key == "small" else BENCH_WARMUP

        try:
            ms = bench(lambda q=query, g=g: list(g.cypher(q)), iterations=iters, warmup=wu)
            results[name] = round(ms, 4)
        except Exception as e:
            results[name] = -1
            errors.append((name, str(e)))

    # Summary
    print()
    categories = {}
    for name, cat, _, _ in BENCHMARKS:
        v = results.get(name, -1)
        categories.setdefault(cat, []).append((name, v))

    total_ok = 0
    total_ms = 0.0
    for cat in dict.fromkeys(c for _, c, _, _ in BENCHMARKS):
        items = categories[cat]
        ok = [(n, v) for n, v in items if v >= 0]
        err = [n for n, v in items if v < 0]
        cat_ms = sum(v for _, v in ok)
        total_ok += len(ok)
        total_ms += cat_ms
        status = f"{cat_ms:>8.2f} ms  ({len(ok)}/{len(items)} ok)"
        if err:
            status += f"  [{len(err)} errors]"
        print(f"  {cat:<14s} {status}")

    print(f"  {'TOTAL':<14s} {total_ms:>8.2f} ms  ({total_ok}/{len(BENCHMARKS)} ok)")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for name, msg in errors:
            short = msg[:80] + "..." if len(msg) > 80 else msg
            print(f"    {name}: {short}")

    print()
    return version, results


if __name__ == "__main__":
    version, results = run_benchmarks()
    save_to_csv(version, results)
