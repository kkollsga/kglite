"""Cypher scalability baseline — measures all code paths affected by
upcoming performance work on pattern expansion, WHERE push-down, index
usage, LIMIT propagation, and degree-aware traversal.

Three graph tiers:
  - small:  ~50 nodes   (correctness-scale, sub-ms expected)
  - medium: 10K nodes   (legal-graph-scale, <100 ms expected)
  - large:  100K nodes  (stress-scale, tests scaling behavior)

All graphs are in-memory (storage="default"). This is the regression
gate — in-memory must not regress.

Run:  python benchmarks/benchmark_cypher_scalability.py
"""

import csv
from pathlib import Path
import statistics
import sys
import time

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kw):
        return it


from kglite import KnowledgeGraph

SCRIPT_DIR = Path(__file__).parent
CSV_OUT = str(SCRIPT_DIR / "benchmark_cypher_scalability.csv")

ITERATIONS = 10
WARMUP = 2


# ═══════════════════════════════════════════════════════════════════
# Graph builders
# ═══════════════════════════════════════════════════════════════════


def build_small() -> KnowledgeGraph:
    """~50 nodes, ~80 edges. Two types, mixed degree."""
    g = KnowledgeGraph()

    people = pd.DataFrame(
        {
            "pid": list(range(30)),
            "name": [f"Person_{i}" for i in range(30)],
            "age": [20 + (i % 50) for i in range(30)],
            "city": [["Oslo", "Bergen", "Trondheim"][i % 3] for i in range(30)],
        }
    )
    g.add_nodes(people, "Person", "pid", "name")

    orgs = pd.DataFrame(
        {
            "oid": list(range(100, 120)),
            "name": [f"Org_{i}" for i in range(20)],
            "sector": [["Tech", "Finance", "Health"][i % 3] for i in range(20)],
        }
    )
    g.add_nodes(orgs, "Organization", "oid", "name")

    # Person->Person edges (social graph, moderate degree)
    knows = pd.DataFrame(
        {
            "from_id": [i % 30 for i in range(60)],
            "to_id": [(i * 7 + 3) % 30 for i in range(60)],
        }
    )
    g.add_connections(knows, "KNOWS", "Person", "from_id", "Person", "to_id")

    # Person->Org edges
    works = pd.DataFrame(
        {
            "pid": [i % 30 for i in range(30)],
            "oid": [100 + (i % 20) for i in range(30)],
        }
    )
    g.add_connections(works, "WORKS_AT", "Person", "pid", "Organization", "oid")

    return g


def build_medium() -> KnowledgeGraph:
    """~10K nodes, ~30K edges. Multiple types, power-law-ish degree."""
    g = KnowledgeGraph()

    n_people = 5000
    n_orgs = 3000
    n_docs = 2000

    people = pd.DataFrame(
        {
            "pid": list(range(n_people)),
            "name": [f"Person_{i}" for i in range(n_people)],
            "age": [20 + (i % 60) for i in range(n_people)],
            "city": [["Oslo", "Bergen", "Trondheim", "Stavanger", "Tromsø"][i % 5] for i in range(n_people)],
            "score": [float(i % 100) for i in range(n_people)],
        }
    )
    g.add_nodes(people, "Person", "pid", "name")

    orgs = pd.DataFrame(
        {
            "oid": list(range(10000, 10000 + n_orgs)),
            "name": [f"Org_{i}" for i in range(n_orgs)],
            "sector": [["Tech", "Finance", "Health", "Energy", "Education"][i % 5] for i in range(n_orgs)],
            "revenue": [float(i * 1000) for i in range(n_orgs)],
        }
    )
    g.add_nodes(orgs, "Organization", "oid", "name")

    docs = pd.DataFrame(
        {
            "did": list(range(20000, 20000 + n_docs)),
            "name": [f"Doc_{i}" for i in range(n_docs)],
            "year": [2000 + (i % 25) for i in range(n_docs)],
            "category": [["Legal", "Technical", "Financial"][i % 3] for i in range(n_docs)],
        }
    )
    g.add_nodes(docs, "Document", "did", "name")

    # Person->Person: 10K edges (avg degree 2)
    knows_from = [i % n_people for i in range(10000)]
    knows_to = [(i * 7 + 13) % n_people for i in range(10000)]
    g.add_connections(
        pd.DataFrame({"f": knows_from, "t": knows_to}),
        "KNOWS",
        "Person",
        "f",
        "Person",
        "t",
    )

    # Person->Org: 8K edges
    works_from = [i % n_people for i in range(8000)]
    works_to = [10000 + (i * 3 + 7) % n_orgs for i in range(8000)]
    g.add_connections(
        pd.DataFrame({"f": works_from, "t": works_to}),
        "WORKS_AT",
        "Person",
        "f",
        "Organization",
        "t",
    )

    # Person->Doc: 10K edges
    authored_from = [i % n_people for i in range(10000)]
    authored_to = [20000 + (i * 11 + 5) % n_docs for i in range(10000)]
    g.add_connections(
        pd.DataFrame({"f": authored_from, "t": authored_to}),
        "AUTHORED",
        "Person",
        "f",
        "Document",
        "t",
    )

    # Doc->Doc: 5K citation edges
    cites_from = [20000 + (i % n_docs) for i in range(5000)]
    cites_to = [20000 + ((i * 13 + 7) % n_docs) for i in range(5000)]
    g.add_connections(
        pd.DataFrame({"f": cites_from, "t": cites_to}),
        "CITES",
        "Document",
        "f",
        "Document",
        "t",
    )

    return g


def build_large() -> KnowledgeGraph:
    """~100K nodes, ~300K edges. Tests scaling behavior.
    Includes a high-degree hub node (node 0) to test fan-out."""
    g = KnowledgeGraph()

    n_entities = 80000
    n_categories = 20000

    entities = pd.DataFrame(
        {
            "eid": list(range(n_entities)),
            "name": [f"Entity_{i}" for i in range(n_entities)],
            "value": [float(i % 1000) for i in range(n_entities)],
            "group": [f"grp_{i % 100}" for i in range(n_entities)],
            "flag": [i % 2 == 0 for i in range(n_entities)],
        }
    )
    g.add_nodes(entities, "Entity", "eid", "name")

    categories = pd.DataFrame(
        {
            "cid": list(range(100000, 100000 + n_categories)),
            "name": [f"Cat_{i}" for i in range(n_categories)],
            "level": [i % 5 for i in range(n_categories)],
        }
    )
    g.add_nodes(categories, "Category", "cid", "name")

    # Entity->Entity: 150K edges (avg degree ~2)
    rel_from = [i % n_entities for i in range(150000)]
    rel_to = [(i * 7 + 13) % n_entities for i in range(150000)]
    g.add_connections(
        pd.DataFrame({"f": rel_from, "t": rel_to}),
        "RELATED",
        "Entity",
        "f",
        "Entity",
        "t",
    )

    # Entity->Category: 100K edges (many-to-one-ish)
    typed_from = [i % n_entities for i in range(100000)]
    typed_to = [100000 + (i * 3 + 1) % n_categories for i in range(100000)]
    g.add_connections(
        pd.DataFrame({"f": typed_from, "t": typed_to}),
        "HAS_TYPE",
        "Entity",
        "f",
        "Category",
        "t",
    )

    # Hub node: entity 0 gets 5000 extra incoming RELATED edges
    # This simulates a power-law hub like Wikidata Q5 ("human")
    hub_from = list(range(1, 5001))
    hub_to = [0] * 5000
    g.add_connections(
        pd.DataFrame({"f": hub_from, "t": hub_to}),
        "RELATED",
        "Entity",
        "f",
        "Entity",
        "t",
    )

    return g


# ═══════════════════════════════════════════════════════════════════
# Timing harness
# ═══════════════════════════════════════════════════════════════════


def bench(fn, iterations=ITERATIONS, warmup=WARMUP):
    """Run fn() multiple times, return (median_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    return statistics.median(times), min(times), max(times)


# ═══════════════════════════════════════════════════════════════════
# Benchmark definitions
#   (name, category, graph_key, query, [params])
# ═══════════════════════════════════════════════════════════════════

BENCHMARKS: list[tuple] = [
    # ── 1. INITIAL CANDIDATE GENERATION (find_matching_nodes) ────────
    # Tests type_index scan vs index-accelerated lookup
    # 1a. ID lookup — should be O(1) via id_indices
    ("id_lookup_small", "candidates", "small", "MATCH (n:Person {id: 5}) RETURN n.title"),
    ("id_lookup_medium", "candidates", "medium", "MATCH (n:Person {id: 2500}) RETURN n.title"),
    ("id_lookup_large", "candidates", "large", "MATCH (n:Entity {id: 40000}) RETURN n.title"),
    # 1b. Full type scan — returns all nodes of a type
    ("type_scan_small", "candidates", "small", "MATCH (n:Person) RETURN count(n)"),
    ("type_scan_medium", "candidates", "medium", "MATCH (n:Person) RETURN count(n)"),
    ("type_scan_large", "candidates", "large", "MATCH (n:Entity) RETURN count(n)"),
    # 1c. Untyped scan — all nodes, worst case
    ("untyped_scan_small", "candidates", "small", "MATCH (n) RETURN count(n)"),
    ("untyped_scan_medium", "candidates", "medium", "MATCH (n) RETURN count(n)"),
    ("untyped_scan_large", "candidates", "large", "MATCH (n) RETURN count(n)"),
    # 1d. Property filter on type — scan + filter
    ("prop_filter_small", "candidates", "small", "MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.title"),
    ("prop_filter_medium", "candidates", "medium", "MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.title"),
    ("prop_filter_large", "candidates", "large", "MATCH (n:Entity) WHERE n.group = 'grp_0' RETURN n.title"),
    # 1e. Range filter
    ("range_filter_medium", "candidates", "medium", "MATCH (n:Person) WHERE n.age > 60 RETURN n.title"),
    ("range_filter_large", "candidates", "large", "MATCH (n:Entity) WHERE n.value > 990 RETURN n.title"),
    # 1f. IN filter
    ("in_filter_medium", "candidates", "medium", "MATCH (n:Person {id: 100}) RETURN n.title"),
    (
        "in_filter_large",
        "candidates",
        "large",
        "MATCH (n:Entity) WHERE n.id IN [1, 100, 1000, 10000, 50000] RETURN n.title",
    ),
    # ── 2. PATTERN EXPANSION (expand_from_node) ─────────────────────
    # Tests edge traversal fan-out
    # 2a. Single-hop traversal — typical case
    ("single_hop_small", "expansion", "small", "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, b.title"),
    ("single_hop_medium", "expansion", "medium", "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, b.title"),
    ("single_hop_large", "expansion", "large", "MATCH (a:Entity)-[:RELATED]->(b:Entity) RETURN count(*)"),
    # 2b. Single-hop from specific node (anchored) — typical MCP query
    ("anchored_hop_small", "expansion", "small", "MATCH (a:Person {id: 0})-[:KNOWS]->(b:Person) RETURN b.title"),
    ("anchored_hop_medium", "expansion", "medium", "MATCH (a:Person {id: 0})-[:KNOWS]->(b:Person) RETURN b.title"),
    ("anchored_hop_large", "expansion", "large", "MATCH (a:Entity {id: 0})-[:RELATED]->(b:Entity) RETURN b.title"),
    # 2c. Hub node incoming — high degree fan-in (entity 0 in large)
    ("hub_incoming_large", "expansion", "large", "MATCH (a:Entity)-[:RELATED]->(b:Entity {id: 0}) RETURN count(a)"),
    # 2d. Multi-hop traversal
    ("two_hop_small", "expansion", "small", "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.title"),
    ("two_hop_medium", "expansion", "medium", "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.title"),
    ("two_hop_large", "expansion", "large", "MATCH (a:Entity {id: 0})-[:RELATED]->(b)-[:RELATED]->(c) RETURN count(*)"),
    # 2e. Cross-type traversal
    (
        "cross_type_medium",
        "expansion",
        "medium",
        "MATCH (p:Person)-[:AUTHORED]->(d:Document) RETURN p.title, d.title LIMIT 100",
    ),
    (
        "cross_type_large",
        "expansion",
        "large",
        "MATCH (e:Entity)-[:HAS_TYPE]->(c:Category) RETURN e.title, c.title LIMIT 100",
    ),
    # ── 3. LIMIT PUSH-DOWN ──────────────────────────────────────────
    # Tests that LIMIT stops expansion early
    # 3a. LIMIT on full type scan — should be fast regardless of type size
    ("limit_type_scan_small", "limit", "small", "MATCH (n:Person) RETURN n.title LIMIT 5"),
    ("limit_type_scan_medium", "limit", "medium", "MATCH (n:Person) RETURN n.title LIMIT 5"),
    ("limit_type_scan_large", "limit", "large", "MATCH (n:Entity) RETURN n.title LIMIT 5"),
    # 3b. LIMIT on edge expansion
    (
        "limit_expansion_medium",
        "limit",
        "medium",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, b.title LIMIT 10",
    ),
    (
        "limit_expansion_large",
        "limit",
        "large",
        "MATCH (a:Entity)-[:RELATED]->(b:Entity) RETURN a.title, b.title LIMIT 10",
    ),
    # 3c. LIMIT on multi-hop
    (
        "limit_two_hop_large",
        "limit",
        "large",
        "MATCH (a:Entity)-[:RELATED]->(b)-[:RELATED]->(c) RETURN a.title, c.title LIMIT 10",
    ),
    # ── 4. WHERE PUSH-DOWN EFFECTIVENESS ────────────────────────────
    # Tests that WHERE predicates are pushed into MATCH patterns
    # 4a. Equality pushed down
    (
        "where_eq_pushed_medium",
        "where_pushdown",
        "medium",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.city = 'Oslo' RETURN a.title, b.title",
    ),
    (
        "where_eq_pushed_large",
        "where_pushdown",
        "large",
        "MATCH (a:Entity)-[:RELATED]->(b:Entity) WHERE a.group = 'grp_0' RETURN a.title, b.title LIMIT 100",
    ),
    # 4b. Non-pushable WHERE (function-based) — must scan then filter
    (
        "where_contains_medium",
        "where_pushdown",
        "medium",
        "MATCH (n:Person) WHERE n.name CONTAINS 'Person_1' RETURN n.title",
    ),
    (
        "where_startswith_medium",
        "where_pushdown",
        "medium",
        "MATCH (n:Person) WHERE n.name STARTS WITH 'Person_10' RETURN n.title",
    ),
    # 4c. WHERE on second pattern variable (b, not a)
    (
        "where_on_target_medium",
        "where_pushdown",
        "medium",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.city = 'Bergen' RETURN a.title, b.title",
    ),
    (
        "where_on_target_large",
        "where_pushdown",
        "large",
        "MATCH (a:Entity)-[:RELATED]->(b:Entity) WHERE b.group = 'grp_0' RETURN count(*)",
    ),
    # ── 5. PATTERN START NODE OPTIMIZATION ──────────────────────────
    # Tests that the planner picks the more selective end
    # 5a. Anchor on last node (should reverse pattern)
    (
        "reverse_pattern_medium",
        "start_node",
        "medium",
        "MATCH (a:Person)-[:AUTHORED]->(d:Document {id: 20001}) RETURN a.title",
    ),
    (
        "reverse_pattern_large",
        "start_node",
        "large",
        "MATCH (a:Entity)-[:HAS_TYPE]->(c:Category {id: 100005}) RETURN a.title",
    ),
    # 5b. Both ends have filters — planner should pick more selective
    (
        "both_filtered_medium",
        "start_node",
        "medium",
        "MATCH (p:Person {id: 0})-[:KNOWS]->(q:Person {id: 100}) RETURN p.title, q.title",
    ),
    # ── 6. AGGREGATION ──────────────────────────────────────────────
    # 6a. Fused count (should be O(1))
    ("fused_count_all_large", "aggregation", "large", "MATCH (n) RETURN count(n)"),
    ("fused_count_typed_large", "aggregation", "large", "MATCH (n:Entity) RETURN count(n)"),
    ("fused_count_by_type_large", "aggregation", "large", "MATCH (n) RETURN n.type, count(n) AS cnt"),
    # 6b. GROUP BY aggregation
    ("groupby_medium", "aggregation", "medium", "MATCH (n:Person) RETURN n.city, count(*) AS cnt ORDER BY cnt DESC"),
    ("groupby_large", "aggregation", "large", "MATCH (n:Entity) RETURN n.group, count(*) AS cnt ORDER BY cnt DESC"),
    # 6c. COLLECT
    (
        "collect_medium",
        "aggregation",
        "medium",
        "MATCH (n:Person) WHERE n.city = 'Oslo' RETURN collect(n.title) AS names",
    ),
    (
        "collect_large",
        "aggregation",
        "large",
        "MATCH (n:Entity) WHERE n.group = 'grp_0' RETURN collect(n.title) AS names",
    ),
    # ── 7. VARIABLE-LENGTH PATHS ────────────────────────────────────
    ("var_path_1_2_small", "var_length", "small", "MATCH (a:Person {id: 0})-[:KNOWS*1..2]->(b) RETURN b.title"),
    ("var_path_1_2_medium", "var_length", "medium", "MATCH (a:Person {id: 0})-[:KNOWS*1..2]->(b) RETURN b.title"),
    ("var_path_1_3_medium", "var_length", "medium", "MATCH (a:Person {id: 0})-[:KNOWS*1..3]->(b) RETURN count(*)"),
    ("var_path_1_2_large", "var_length", "large", "MATCH (a:Entity {id: 0})-[:RELATED*1..2]->(b) RETURN count(*)"),
    # ── 8. FUSED OPTIMIZATIONS ──────────────────────────────────────
    # 8a. FusedCountEdgesByType
    ("fused_edge_count_medium", "fused", "medium", "MATCH ()-[r]->() RETURN type(r), count(*) AS cnt"),
    ("fused_edge_count_large", "fused", "large", "MATCH ()-[r]->() RETURN type(r), count(*) AS cnt"),
    # 8b. Optional match aggregate
    (
        "optional_agg_medium",
        "fused",
        "medium",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:AUTHORED]->(d:Document) RETURN p.title, count(d) AS docs",
    ),
    # ── 9. CARTESIAN PRODUCT RISK ───────────────────────────────────
    # These queries have explosion potential — the benchmark
    # verifies they complete in bounded time
    # 9a. Two unanchored typed scans (cross-join risk with LIMIT safety)
    (
        "cartesian_limited_medium",
        "cartesian",
        "medium",
        "MATCH (a:Person), (b:Organization) RETURN a.title, b.title LIMIT 20",
    ),
    # 9b. Large expansion with LIMIT
    (
        "expansion_limited_large",
        "cartesian",
        "large",
        "MATCH (a:Entity)-[:RELATED]->(b:Entity) RETURN a.title, b.title LIMIT 20",
    ),
    # 9c. Full expansion on large (no LIMIT) — stress test
    # Only run 1 iteration, measure if it stays sane
    ("full_expansion_large", "cartesian", "large", "MATCH (a:Entity)-[:RELATED]->(b:Entity) RETURN count(*)"),
]


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════


def main():
    print("Building graphs...")
    t0 = time.perf_counter()
    graphs = {
        "small": build_small(),
        "medium": build_medium(),
        "large": build_large(),
    }
    build_time = time.perf_counter() - t0
    for name, g in graphs.items():
        nc = g.cypher("MATCH (n) RETURN count(n)")[0]["count(n)"]
        ec = g.cypher("MATCH ()-[r]->() RETURN count(r)")[0]["count(r)"]
        print(f"  {name}: {nc:,} nodes, {ec:,} edges")
    print(f"  Built in {build_time:.1f}s\n")

    results = []
    errors = []

    for entry in tqdm(BENCHMARKS, desc="Benchmarks"):
        name, category, graph_key, query = entry[0], entry[1], entry[2], entry[3]
        params = entry[4] if len(entry) > 4 else None
        g = graphs[graph_key]

        try:
            # Quick sanity check
            if params:
                g.cypher(query, params=params)
            else:
                g.cypher(query)

            # Benchmark
            if name.startswith("full_expansion"):
                median, mn, mx = bench(
                    lambda q=query, p=params: g.cypher(q, params=p) if p else g.cypher(q),
                    iterations=3,
                    warmup=1,
                )
            else:
                median, mn, mx = bench(
                    lambda q=query, p=params: g.cypher(q, params=p) if p else g.cypher(q),
                )

            results.append(
                {
                    "name": name,
                    "category": category,
                    "graph": graph_key,
                    "median_ms": round(median, 3),
                    "min_ms": round(mn, 3),
                    "max_ms": round(mx, 3),
                }
            )
            status = f"{median:8.3f} ms"
        except Exception as e:
            errors.append((name, str(e)))
            status = f"ERROR: {e}"

        print(f"  {name:45s} {status}")

    # Write CSV
    if results:
        with open(CSV_OUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "category", "graph", "median_ms", "min_ms", "max_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {CSV_OUT}")

    # Summary by category
    print("\n" + "=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, items in sorted(categories.items()):
        print(f"\n  {cat}:")
        for r in items:
            print(f"    {r['name']:45s} {r['median_ms']:8.3f} ms  ({r['graph']})")

    if errors:
        print(f"\n{'=' * 70}")
        print(f"ERRORS ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}: {err}")

    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
