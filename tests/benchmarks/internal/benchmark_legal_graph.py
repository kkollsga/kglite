"""
Benchmark: Norwegian legal knowledge graph query performance.
Source: /Volumes/EksternalHome/Koding/MCP servers/legal/norwegian_law.kgl
158K nodes, 1M edges, 8 types.

Usage: python examples/benchmark_legal_graph.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kglite import load

SOURCE_GRAPH = "/Volumes/EksternalHome/Koding/MCP servers/legal/norwegian_law.kgl"

CYPHER_QUERIES = [
    ("Count nodes", "MATCH (n) RETURN count(n)"),
    ("Count CourtDecision", "MATCH (n:CourtDecision) RETURN count(n)"),
    ("Count Law", "MATCH (n:Law) RETURN count(n)"),
    ("Count LawSection", "MATCH (n:LawSection) RETURN count(n)"),
    ("Law titles L10", "MATCH (n:Law) RETURN n.title LIMIT 10"),
    ("Decision titles L10", "MATCH (d:CourtDecision) RETURN d.title LIMIT 10"),
    ("Decision edges L10", "MATCH (d:CourtDecision)-[r]->(t) RETURN d.title, type(r), t.title LIMIT 10"),
    ("Decision -[:CITES]-> L10", "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection) RETURN d.title, s.title LIMIT 10"),
    (
        "Decision -[:JUDGED_BY]-> L10",
        "MATCH (d:CourtDecision)-[:JUDGED_BY]->(j:Judge) RETURN d.title, j.title LIMIT 10",
    ),
    (
        "Decision -[:HAS_KEYWORD]-> L10",
        "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) RETURN d.title, k.title LIMIT 10",
    ),
    ("Incoming to Law L10", "MATCH (n)-[r]->(l:Law) RETURN n.title, type(r) LIMIT 10"),
    (
        "2-hop dec->sec->law L10",
        "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law)"
        " RETURN d.title, s.title, l.title LIMIT 10",
    ),
    (
        "Most cited sections TOP10",
        "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection) RETURN s.title, count(d) AS c ORDER BY c DESC LIMIT 10",
    ),
    ("Supreme Court L10", "MATCH (d:CourtDecision) WHERE d.court_level = 'Høyesterett' RETURN d.title LIMIT 10"),
    (
        "Keyword strafferett",
        "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword {title: 'strafferett'}) RETURN d.title LIMIT 10",
    ),
    ("Law by korttittel", "MATCH (l:Law {korttittel: 'straffeloven'}) RETURN l.title"),
    (
        "2-hop law->sec<-decision",
        "MATCH (l:Law {korttittel: 'straffeloven'})<-[:SECTION_OF]"
        "-(s:LawSection)<-[:CITES]-(d:CourtDecision) RETURN s.title, d.title LIMIT 10",
    ),
    (
        "Case progression",
        "MATCH (d1:CourtDecision)-[:CASE_PROGRESSION]->(d2:CourtDecision) RETURN d1.title, d2.title LIMIT 10",
    ),
    ("WHERE id=1", "MATCH (n) WHERE id(n) = 1 RETURN n.title, labels(n)"),
    ("Count all CITES edges", "MATCH (d:CourtDecision)-[:CITES]->() RETURN count(d)"),
]

FLUENT_OPS = [
    ("select(CourtDecision).len()", lambda g: g.select("CourtDecision").len()),
    ("select(Law).len()", lambda g: g.select("Law").len()),
    ("select(LawSection).len()", lambda g: g.select("LawSection").len()),
    (
        "traverse CITES out L100",
        lambda g: g.select("CourtDecision").traverse("CITES", direction="outgoing", limit=100).len(),
    ),
    ("traverse CITES in L50", lambda g: g.select("LawSection").traverse("CITES", direction="incoming", limit=50).len()),
    ("where court=Høyesterett", lambda g: g.select("CourtDecision").where_({"court_level": "Høyesterett"}).len()),
    ("where_connected CITES", lambda g: g.select("CourtDecision").where_connected("CITES").len()),
    (
        "2-hop CITES->SECTION_OF",
        lambda g: g.select("CourtDecision").traverse("CITES", limit=50).traverse("SECTION_OF", limit=50).len(),
    ),
    ("describe() length", lambda g: len(g.describe())),
]


if __name__ == "__main__":
    print(f"Loading {SOURCE_GRAPH}...", flush=True)
    t0 = time.perf_counter()
    g = load(SOURCE_GRAPH)
    load_time = time.perf_counter() - t0
    info = g.graph_info()
    nc, ec, tc = info["node_count"], info["edge_count"], info["type_count"]
    print(
        f"Loaded in {load_time:.2f}s — {nc:,} nodes, {ec:,} edges, {tc} types\n",
        flush=True,
    )

    # Cypher queries
    print(f"{'Query':40s} {'Time':>8s}  {'Rows':>8s}")
    print("=" * 62)
    total = 0.0
    for label, query in CYPHER_QUERIES:
        t0 = time.perf_counter()
        try:
            df = g.cypher(query, timeout_ms=30000).to_df()
            rows = len(df)
        except Exception as e:
            rows = f"ERR: {str(e)[:30]}"
        elapsed = time.perf_counter() - t0
        total += elapsed
        r = f"{rows:,}" if isinstance(rows, int) else rows
        print(f"{label:40s} {elapsed:>7.3f}s  {r:>8s}", flush=True)
    print(f"{'TOTAL':40s} {total:>7.3f}s")

    # Fluent API
    print(f"\n{'Fluent API':40s} {'Time':>8s}  {'Result':>8s}")
    print("=" * 62)
    total = 0.0
    for label, fn in FLUENT_OPS:
        t0 = time.perf_counter()
        try:
            result = fn(g)
        except Exception as e:
            result = f"ERR: {str(e)[:30]}"
        elapsed = time.perf_counter() - t0
        total += elapsed
        r = f"{result:,}" if isinstance(result, int) else str(result)[:20]
        print(f"{label:40s} {elapsed:>7.3f}s  {r:>8s}", flush=True)
    print(f"{'TOTAL':40s} {total:>7.3f}s")
