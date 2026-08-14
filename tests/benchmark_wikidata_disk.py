"""
Wikidata full-scale benchmark: disk graph mode.
Builds or loads full Wikidata, runs 30 Cypher + 20 Fluent queries.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kglite import KnowledgeGraph, load

WIKIDATA_ZST = "/Volumes/EksternalHome/Data/Wikidata/latest-truthy.nt.zst"
GRAPH_DIR = "/Volumes/EksternalHome/Data/Wikidata/wikidata_disk_graph"


def timed(label, fn):
    sys.stdout.write(f"  {label}...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f" ERROR in {elapsed:.2f}s: {e}")
        return None, elapsed
    elapsed = time.perf_counter() - t0
    print(f" {elapsed:.2f}s", flush=True)
    return result, elapsed


def build_graph():
    print("\n" + "=" * 70)
    print("PHASE 1: BUILD DISK GRAPH FROM WIKIDATA")
    print("=" * 70)

    t_total = time.perf_counter()
    print(f"\n  Source: {WIKIDATA_ZST}")
    print(f"  Target: {GRAPH_DIR}")

    g = KnowledgeGraph(storage="disk", path=GRAPH_DIR)

    print("\n  Loading N-Triples...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    g.load_ntriples(WIKIDATA_ZST, languages=["en"], verbose=True)
    t_load = time.perf_counter() - t0
    print(f"  Load time: {t_load:.1f}s ({t_load / 60:.1f} min)")

    info = g.graph_info()
    print(f"  Nodes: {info.get('node_count', 'N/A'):,}")
    print(f"  Edges: {info.get('edge_count', 'N/A'):,}")

    t_total = time.perf_counter() - t_total
    print(f"\n  TOTAL BUILD: {t_total:.1f}s ({t_total / 60:.1f} min)")

    total_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, files in os.walk(GRAPH_DIR) for f in files)
    print(f"  Disk size: {total_bytes / (1024**3):.2f} GB")
    return g


def load_graph():
    print("\n" + "=" * 70)
    print("PHASE 2: LOAD DISK GRAPH")
    print("=" * 70)
    g, _ = timed("Load from disk", lambda: load(GRAPH_DIR))
    info = g.graph_info()
    print(f"  Nodes: {info.get('node_count', 'N/A'):,}")
    print(f"  Edges: {info.get('edge_count', 'N/A'):,}")
    return g


def run_benchmarks(g):
    print("\n" + "=" * 70)
    print("PHASE 3: CYPHER BENCHMARKS (30 queries)")
    print("=" * 70)

    results = []

    def q(label, query):
        t0 = time.perf_counter()
        try:
            df = g.cypher(query).to_df()
            rows = len(df)
        except Exception as e:
            rows = f"ERR: {e}"
        elapsed = time.perf_counter() - t0
        results.append((label, elapsed, rows))
        r = f"{rows:,}" if isinstance(rows, int) else str(rows)[:40]
        print(f"  {label:50s} {elapsed:>8.3f}s  {r:>12s}")
        sys.stdout.flush()

    hdr = f"\n  {'Query':50s} {'Time':>8s}  {'Rows':>12s}"
    print(hdr)
    print("  " + "-" * 74)

    # --- Counts ---
    q("COUNT all nodes", "MATCH (n) RETURN count(n) AS c")

    # --- ID lookups ---
    q("Q1 Universe", "MATCH (n {nid: 'Q1'}) RETURN n.label, n.description")
    q("Q5 Human", "MATCH (n {nid: 'Q5'}) RETURN n.label, n.description")
    q("Q42 Douglas Adams", "MATCH (n {nid: 'Q42'}) RETURN n.label, n.description")
    q("Q64 Berlin", "MATCH (n {nid: 'Q64'}) RETURN n.label, n.description")
    q("Q76 Barack Obama", "MATCH (n {nid: 'Q76'}) RETURN n.label, n.description")
    q("Q183 Germany", "MATCH (n {nid: 'Q183'}) RETURN n.label, n.description")
    q("Q515 city (concept)", "MATCH (n {nid: 'Q515'}) RETURN n.label, n.description")

    # --- Property filters ---
    q(
        "WHERE label CONTAINS 'Einstein'",
        "MATCH (n) WHERE n.label CONTAINS 'Einstein' RETURN n.nid, n.label LIMIT 20",
    )
    q(
        "WHERE label STARTS WITH 'Albert'",
        "MATCH (n) WHERE n.label STARTS WITH 'Albert' RETURN n.nid, n.label LIMIT 20",
    )
    q(
        "WHERE label CONTAINS 'Norway'",
        "MATCH (n) WHERE n.label CONTAINS 'Norway' RETURN n.nid, n.label LIMIT 20",
    )

    # --- 1-hop traversals ---
    q("Q42 outgoing LIMIT 20", "MATCH ({nid:'Q42'})-[r]->(m) RETURN type(r), m.label LIMIT 20")
    q("Q42 incoming LIMIT 20", "MATCH (m)-[r]->({nid:'Q42'}) RETURN type(r), m.label LIMIT 20")
    q("Q64 outgoing LIMIT 20", "MATCH ({nid:'Q64'})-[r]->(m) RETURN type(r), m.label LIMIT 20")
    q("Q76 outgoing LIMIT 20", "MATCH ({nid:'Q76'})-[r]->(m) RETURN type(r), m.label LIMIT 20")
    q("Q183 outgoing LIMIT 20", "MATCH ({nid:'Q183'})-[r]->(m) RETURN type(r), m.label LIMIT 20")
    q(
        "Instances of Human LIMIT 50",
        "MATCH (n)-[:P31]->({nid:'Q5'}) RETURN n.label LIMIT 50",
    )
    q(
        "Instances of City LIMIT 50",
        "MATCH (n)-[:P31]->({nid:'Q515'}) RETURN n.label LIMIT 50",
    )
    q(
        "Instances of Country LIMIT 50",
        "MATCH (n)-[:P31]->({nid:'Q6256'}) RETURN n.label LIMIT 50",
    )

    # --- 2-hop ---
    q(
        "Q42 2-hop LIMIT 20",
        "MATCH ({nid:'Q42'})-[]->(b)-[]->(c) RETURN b.label, c.label LIMIT 20",
    )
    q("Born in Berlin LIMIT 20", "MATCH (p)-[:P19]->({nid:'Q64'}) RETURN p.label LIMIT 20")
    q("Died in London LIMIT 20", "MATCH (p)-[:P20]->({nid:'Q84'}) RETURN p.label LIMIT 20")

    # --- Aggregation ---
    q(
        "Top 20 most linked entities",
        "MATCH (n)-[r]->() RETURN n.nid, n.label, count(r) AS d ORDER BY d DESC LIMIT 20",
    )
    q(
        "Edge type distribution",
        "MATCH ()-[r]->() RETURN type(r), count(r) AS c ORDER BY c DESC LIMIT 20",
    )

    # --- Path / connection ---
    q("Q42->Q5 direct?", "MATCH ({nid:'Q42'})-[r]->({nid:'Q5'}) RETURN type(r)")
    q("Q76->Q30 direct?", "MATCH ({nid:'Q76'})-[r]->({nid:'Q30'}) RETURN type(r)")

    # --- OPTIONAL MATCH ---
    q(
        "Q42 optional edges",
        "MATCH (n {nid:'Q42'}) OPTIONAL MATCH (n)-[r]->(m) RETURN type(r), m.label LIMIT 30",
    )

    # --- Fluent API ---
    print("\n" + "=" * 70)
    print("PHASE 4: FLUENT API BENCHMARKS (20 queries)")
    print("=" * 70)
    print(f"\n  {'Operation':50s} {'Time':>8s}  {'Result':>12s}")
    print("  " + "-" * 74)

    fluent_results = []

    def fl(label, fn):
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as e:
            result = f"ERR: {e}"
        elapsed = time.perf_counter() - t0
        fluent_results.append((label, elapsed, result))
        r = f"{result:,}" if isinstance(result, int) else str(result)[:40]
        print(f"  {label:50s} {elapsed:>8.3f}s  {r:>12s}")
        sys.stdout.flush()

    fl("select(Entity).len()", lambda: g.select("Entity").len())
    fl("graph_info()", lambda: len(str(g.graph_info())))
    fl("node_types", lambda: g.node_types)

    # Traversal via fluent
    fl(
        "select(Entity).traverse(P31).len()",
        lambda: g.select("Entity").traverse("P31", limit=100).len(),
    )
    fl(
        "select(Entity).traverse(P31,out).to_df LIMIT 20",
        lambda: len(g.select("Entity").traverse("P31", direction="outgoing", limit=20).to_df()),
    )
    fl(
        "select(Entity).traverse(P31,in).len()",
        lambda: g.select("Entity").traverse("P31", direction="incoming", limit=50).len(),
    )

    # Where filters via fluent
    fl(
        "select.where(label contains 'Berlin')",
        lambda: g.select("Entity").where({"label": ("contains", "Berlin")}, limit=20).len(),
    )
    fl(
        "select.where(label starts 'United')",
        lambda: g.select("Entity").where({"label": ("starts_with", "United")}, limit=20).len(),
    )
    fl(
        "select.where(P31 = 'Q5')",
        lambda: g.select("Entity").where({"P31": "Q5"}, limit=20).len(),
    )
    fl(
        "select.where(P31 = 'Q515')",
        lambda: g.select("Entity").where({"P31": "Q515"}, limit=20).len(),
    )
    fl(
        "select.where(P31 = 'Q6256')",
        lambda: g.select("Entity").where({"P31": "Q6256"}, limit=20).len(),
    )

    # Sorting
    fl(
        "select.where(P31='Q5', limit=50).to_df()",
        lambda: len(g.select("Entity").where({"P31": "Q5"}, limit=50).to_df()),
    )

    # Multiple traversals
    fl(
        "traverse P31 then P17 LIMIT 10",
        lambda: g.select("Entity").traverse("P31", limit=10).traverse("P17", limit=10).len(),
    )
    fl(
        "traverse P31 then P30 LIMIT 10",
        lambda: g.select("Entity").traverse("P31", limit=10).traverse("P30", limit=10).len(),
    )

    # Selection length
    fl("select all .len()", lambda: g.select("Entity").len())

    # Edge info
    fl("edge_types (via graph_info)", lambda: len(g.graph_info().get("edge_types", [])))

    # Describe
    fl("describe() length", lambda: len(g.describe()))

    # Node count via select
    fl(
        "select(Entity).where(is_not_null label).len()",
        lambda: g.select("Entity").where({"label": "is_not_null"}, limit=100).len(),
    )
    fl(
        "select(Entity).where(is_null description).len()",
        lambda: g.select("Entity").where({"description": "is_null"}, limit=100).len(),
    )

    # Export
    fl(
        "select.to_list() LIMIT 10",
        lambda: len(g.select("Entity").where({"P31": "Q5"}, limit=10).to_list()),
    )
    fl(
        "select.to_df() LIMIT 20",
        lambda: len(g.select("Entity").where({"P31": "Q515"}, limit=20).to_df()),
    )

    # Count connected
    fl(
        "select.where_connected(P31).len()",
        lambda: g.select("Entity").where_connected("P31").len(),
    )
    fl(
        "select.where_connected(P17).len()",
        lambda: g.select("Entity").where_connected("P17").len(),
    )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_cypher = sum(r[1] for r in results)
    total_fluent = sum(r[1] for r in fluent_results)
    print(
        f"  Cypher: {total_cypher:.2f}s across {len(results)} queries (avg {total_cypher / max(len(results), 1):.3f}s)"
    )
    print(f"  Fluent: {total_fluent:.2f}s across {len(fluent_results)} queries")
    if results:
        fastest = min(results, key=lambda r: r[1])
        slowest = max(results, key=lambda r: r[1])
        print(f"  Fastest Cypher: {fastest[0]} ({fastest[1]:.3f}s)")
        print(f"  Slowest Cypher: {slowest[0]} ({slowest[1]:.3f}s)")


def main():
    print("=" * 70)
    print("WIKIDATA FULL-SCALE DISK GRAPH BENCHMARK")
    print("=" * 70)
    print(f"Source: {WIKIDATA_ZST}")
    print(f"Graph:  {GRAPH_DIR}")

    if os.path.exists(os.path.join(GRAPH_DIR, "disk_graph_meta.json")):
        print("\nExisting graph found — loading...")
        g = load_graph()
    else:
        print("\nBuilding from N-Triples...")
        g = build_graph()

    # Reload test (if graph was built or loaded)
    if os.path.exists(os.path.join(GRAPH_DIR, "disk_graph_meta.json")):
        print("\n  Reload benchmark:")
        timed("Reload", lambda: load(GRAPH_DIR))

    run_benchmarks(g)


if __name__ == "__main__":
    main()
