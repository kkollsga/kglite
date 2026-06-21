"""Baseline profiler for the top-10 suspected performance bottlenecks.

Sequential, min-over-rounds, release build only. Produces a JSON baseline
(baselines_top10.json) so later fixes can be gated on "measurably faster".

Run:  .venv/bin/python tests/benchmarks/internal/profile_top10.py
"""

import gc
import json
import subprocess
import time
from pathlib import Path

import kglite

HERE = Path(__file__).parent
OUT = HERE / "baselines_top10.json"


def bench(fn, rounds=50, warmup=3):
    for _ in range(warmup):
        fn()
    gc.disable()
    best = float("inf")
    for _ in range(rounds):
        s = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - s)
    gc.enable()
    return best * 1000.0  # ms


def build_users(g, n=30000, edges_per=3):
    rows = [
        {
            "id": i,
            "age": 20 + (i % 80),          # ndv ~80
            "city": f"c{i % 50}",          # ndv ~50 (non-indexed)
            "active": (i % 2 == 0),        # ndv 2
            "a": i, "b": i + 1, "c": f"s{i}", "d": (i % 7 == 0),
        }
        for i in range(n)
    ]
    g.cypher("UNWIND $r AS r CREATE (:User {id:r.id, age:r.age, city:r.city, "
             "active:r.active, a:r.a, b:r.b, c:r.c, d:r.d})", params={"r": rows})
    pairs = [{"s": i, "t": (i + k + 1) % n} for i in range(n) for k in range(edges_per)]
    g.cypher("UNWIND $p AS e MATCH (a:User {id:e.s}), (b:User {id:e.t}) "
             "CREATE (a)-[:KNOWS]->(b)", params={"p": pairs})
    return g


results = {}


def record(key, ms, note=""):
    results[key] = {"min_ms": round(ms, 4), "note": note}
    print(f"  {key:42s} {ms:10.3f} ms   {note}")


print("Building fixture (30k users, 90k KNOWS)…")
G = build_users(kglite.KnowledgeGraph())
N = 30000
items = list(range(2000))

print("\n=== Baselines (min over rounds, release) ===")

# M5 — planner fixed overhead (parse+plan+trivial exec)
record("M5_planner_overhead_trivial",
       bench(lambda: G.cypher("RETURN 1 AS x"), rounds=300),
       "RETURN 1 — pure parse+plan floor")

# M4 — Value clone / node materialization (full node vs scalar, same cardinality)
full = bench(lambda: G.cypher("MATCH (u:User) RETURN u LIMIT 10000").to_list(), rounds=30)
scal = bench(lambda: G.cypher("MATCH (u:User) RETURN u.id LIMIT 10000").to_list(), rounds=30)
record("M4_return_full_node_10k", full, "RETURN n (clones Value::Node)")
record("M4_return_scalar_10k", scal, "RETURN n.id (scalar)")
record("M4_node_clone_delta", full - scal, "delta ≈ node clone+marshal")

# M6 — property access (interning + hashing): 4 props vs 1, full scan
p4 = bench(lambda: G.cypher("MATCH (u:User) RETURN u.a, u.b, u.c, u.d").to_list(), rounds=20)
p1 = bench(lambda: G.cypher("MATCH (u:User) RETURN u.id").to_list(), rounds=20)
record("M6_props_x4_fullscan", p4, "4 property accesses × 30k")
record("M6_props_x1_fullscan", p1, "1 property access × 30k")
record("M6_per_extra_prop_delta", (p4 - p1) / 3, "≈ per-property interning+hash")

# M7 — compare / DISTINCT / ORDER BY dedup
record("M7_where_distinct_orderby",
       bench(lambda: G.cypher("MATCH (u:User) WHERE u.age > 40 "
                              "RETURN DISTINCT u.city AS c ORDER BY c").to_list(), rounds=30),
       "filter + DISTINCT + ORDER BY")

# M3 — multi-MATCH fan-out (intermediate materialization), time proxy
record("M3_two_hop_fanout",
       bench(lambda: G.cypher("MATCH (a:User)-[:KNOWS]->(b:User)-[:KNOWS]->(c:User) "
                              "RETURN count(*) AS n"), rounds=10),
       "2-hop count (large intermediate)")

# M9 — traversal Vec-per-node allocation at increasing depth
for k in (2, 3, 4):
    record(f"M9_varlen_depth_{k}",
           bench(lambda k=k: G.cypher(f"MATCH (a:User {{id:0}})-[:KNOWS*1..{k}]-(b) "
                                      "RETURN count(DISTINCT b) AS n"), rounds=20),
           f"-[*1..{k}]- from one node")

# M10 — py_out marshalling: materialize vs count, same scan
mat = bench(lambda: G.cypher("MATCH (u:User) RETURN u.id, u.age, u.city").to_list(), rounds=20)
cnt = bench(lambda: G.cypher("MATCH (u:User) RETURN count(u) AS n"), rounds=20)
record("M10_materialize_30k_x3", mat, "to_list 30k×3 cells")
record("M10_count_only", cnt, "count() — no marshalling")
record("M10_marshal_delta", mat - cnt, "delta ≈ py_out marshalling")

# M2 — NDV cold (after mutation, whole-map invalidated) vs warm (cached)
ndv_q = ("MATCH (u:User {active:true})-[:KNOWS]->(v:User {city:'c0'}) "
         "RETURN count(*) AS n")
G.cypher(ndv_q)  # warm the cache once
warm = bench(lambda: G.cypher(ndv_q), rounds=20)
def cold():
    G.cypher("CREATE (:Tmp {id: 1})")          # bumps version → drops NDV cache
    G.cypher("MATCH (t:Tmp) DELETE t")          # keep graph stable
    G.cypher(ndv_q)                             # re-scans NDV
record("M2_ndv_warm", warm, "equality query, NDV cached")
record("M2_ndv_cold_after_mutation", bench(cold, rounds=20), "incl. mutation + NDV re-scan")

# M1 / M8 — FOREACH per-element flush, memory vs disk
def foreach_mem():
    g = kglite.KnowledgeGraph()
    g.cypher("FOREACH (i IN $it | MERGE (n:K {id:i}) SET n.t = i)", params={"it": items})
def unwind_mem():
    g = kglite.KnowledgeGraph()
    g.cypher("UNWIND $it AS i MERGE (n:K {id:i}) SET n.t = i", params={"it": items})
record("M1_foreach_merge_set_mem", bench(foreach_mem, rounds=15), f"FOREACH {len(items)} (per-elem flush)")
record("M1_unwind_merge_set_mem", bench(unwind_mem, rounds=15), f"UNWIND {len(items)} (batch flush)")

import tempfile
try:
    def foreach_disk():
        d = tempfile.mkdtemp()
        g = kglite.KnowledgeGraph(storage="disk", path=d)
        g.cypher("FOREACH (i IN $it | MERGE (n:K {id:i}) SET n.t = i)", params={"it": items})
    def unwind_disk():
        d = tempfile.mkdtemp()
        g = kglite.KnowledgeGraph(storage="disk", path=d)
        g.cypher("UNWIND $it AS i MERGE (n:K {id:i}) SET n.t = i", params={"it": items})
    record("M1_foreach_merge_set_disk", bench(foreach_disk, rounds=5), "FOREACH disk (per-elem flush+sync)")
    record("M1_unwind_merge_set_disk", bench(unwind_disk, rounds=5), "UNWIND disk (batch flush+sync)")
except Exception as e:  # noqa: BLE001
    record("M1_disk", -1.0, f"disk mode skipped: {e}")

sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
OUT.write_text(json.dumps({"sha": sha, "results": results}, indent=2))
print(f"\nBaseline written to {OUT}  (HEAD {sha})")
