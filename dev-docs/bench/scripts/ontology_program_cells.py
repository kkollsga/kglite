# Phase 10 A/B cells: P-1 supertype equality, P-2 dematerialize, P-3 alternation.
# min-of-rounds for sub-ms cells; dematerialize = single-shot per size (once-per-event
# cost class -> report the value itself, not min of repeats on same graph).
import kglite, pandas as pd, time, sys

def build(n_per=100_000, ontology=True):
    g = kglite.KnowledgeGraph()
    for t, lo in (("Student", 0), ("Teacher", n_per)):
        g.add_nodes(data=pd.DataFrame({"pid": range(lo, lo+n_per),
                                       "email": [f"e{i}@x.no" for i in range(lo, lo+n_per)],
                                       "name": [f"P{i}" for i in range(lo, lo+n_per)]}),
                    node_type=t, unique_id_field="pid", node_title_field="name")
        g.create_index(t, "email")
    if ontology:
        g.define_ontology({"classes": {"Person": {"abstract": True},
                                       "Student": {"is_a": "Person"}, "Teacher": {"is_a": "Person"}}})
        g.materialize_ontology()
    return g

def cell(g, q, rounds=200):
    best = float("inf")
    list(g.cypher(q))
    for _ in range(rounds):
        t0 = time.perf_counter(); list(g.cypher(q)); dt = time.perf_counter()-t0
        if dt < best: best = dt
    return best*1000

g = build()
cells = [
    ("P1_supertype_email", "MATCH (p:Person {email: 'e55555@x.no'}) RETURN p.id"),
    ("P1_supertype_id",    "MATCH (p:Person {id: 55555}) RETURN p.id"),
    ("CTRL_subtype_email", "MATCH (p:Student {email: 'e55555@x.no'}) RETURN p.id"),
    ("P3_alt_count",       "MATCH (n:Student|Teacher) RETURN count(n)"),
    ("P3_alt_email",       "MATCH (n:Student|Teacher {email: 'e55555@x.no'}) RETURN n.id"),
    ("CTRL_single_count",  "MATCH (n:Student) RETURN count(n)"),
    ("CTRL_scan_where",    "MATCH (n:Student) WHERE n.name STARTS WITH 'P5555' RETURN count(n)"),
]
for name, q in cells:
    print(f"{name} {cell(g, q):.4f} ms")
# P-2: fresh graph per size, single-shot dematerialize
for n in (100_000, 200_000):
    g2 = build(n_per=n//2)
    t0 = time.perf_counter(); g2.dematerialize_ontology(); dt = (time.perf_counter()-t0)*1000
    print(f"P2_dematerialize_{n} {dt:.1f} ms")
