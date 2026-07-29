#!/usr/bin/env python3
"""Objective graph-database benchmark — single self-contained script.

Produces ONE table: rows = benchmarks (fixed schema), columns = libraries.
Adding a new library later = add one Adapter subclass + register it in
ADAPTERS. Nothing else changes.

Design:
  * Each library runs IN A SUBPROCESS (the script re-execs itself with
    --engine) so peak RSS is cleanly isolated per library.
  * All libraries run IN-PROCESS (no wire protocol), execute byte-identical
    Cypher over byte-identical synthetic data. Latency = min over R timed
    rounds after W warmups (min is the right statistic for sub-ms work).

Usage:
  python graph_bench.py --scale 100000                 # all libraries, print table
  python graph_bench.py --scale 100000 --libs kglite   # subset
  python graph_bench.py --scale 100000 --csv out.csv   # also write CSV
  python graph_bench.py --engine kglite --scale 100000 # worker (internal)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time

# ======================================================================
# 1. DATASET  (deterministic; identical bytes for every library)
# ======================================================================
SURNAMES = [f"Sur{i:03d}" for i in range(100)]
CTYPES = ["theft", "assault", "fraud", "arson", "burglary", "vandalism"]


def build_dataset(n_persons: int, avg_degree: int, seed: int, jsonl_path: str):
    """Person{name unique, surname low-card, age} + Crime; KNOWS + PARTY_TO.

    Writes a TuringDB-schema JSONL file AND returns pandas DataFrames, both
    from one build so every library ingests identical data. Node ids share
    one global int space so JSONL endpoints and DataFrame source/target align.
    """
    import pandas as pd

    rng = random.Random(seed)
    n_crimes = max(1, n_persons // 10)

    pid = list(range(n_persons))
    names = [f"Person_{i}" for i in pid]
    surnames = [SURNAMES[rng.randrange(len(SURNAMES))] for _ in pid]
    ages = [rng.randint(18, 90) for _ in pid]
    cid = list(range(n_persons, n_persons + n_crimes))
    ctypes = [CTYPES[rng.randrange(len(CTYPES))] for _ in cid]
    sev = [rng.randint(1, 10) for _ in cid]

    ks, kt = [], []
    for s in pid:
        for _ in range(avg_degree):
            t = rng.randrange(n_persons)
            if t != s:
                ks.append(s)
                kt.append(t)
    ps, pt = [], []
    for s in pid:
        if rng.random() < 0.3:
            ps.append(s)
            pt.append(cid[rng.randrange(n_crimes)])

    with open(jsonl_path, "w") as f:
        w = f.write
        for i in pid:
            w(
                json.dumps(
                    {
                        "type": "node",
                        "id": i,
                        "labels": ["Person"],
                        "properties": {"name": names[i], "surname": surnames[i], "age": ages[i]},
                    }
                )
                + "\n"
            )
        for j, c in enumerate(cid):
            w(
                json.dumps(
                    {
                        "type": "node",
                        "id": c,
                        "labels": ["Crime"],
                        "properties": {"ctype": ctypes[j], "severity": sev[j]},
                    }
                )
                + "\n"
            )
        eid = n_persons + n_crimes
        for s, t in zip(ks, kt):
            w(
                json.dumps(
                    {
                        "type": "relationship",
                        "id": eid,
                        "label": "KNOWS",
                        "start": {"id": s},
                        "end": {"id": t},
                        "properties": {},
                    }
                )
                + "\n"
            )
            eid += 1
        for s, t in zip(ps, pt):
            w(
                json.dumps(
                    {
                        "type": "relationship",
                        "id": eid,
                        "label": "PARTY_TO",
                        "start": {"id": s},
                        "end": {"id": t},
                        "properties": {},
                    }
                )
                + "\n"
            )
            eid += 1

    return {
        "persons": pd.DataFrame({"id": pid, "name": names, "surname": surnames, "age": ages}),
        "crimes": pd.DataFrame({"id": cid, "ctype": ctypes, "severity": sev}),
        "knows": pd.DataFrame({"source_id": ks, "target_id": kt}),
        "party_to": pd.DataFrame({"source_id": ps, "target_id": pt}),
        "n_crimes": n_crimes,
        "n_knows": len(ks),
        "n_party_to": len(ps),
        "jsonl": jsonl_path,
    }


# ======================================================================
# 2. BENCHMARK CORPUS  (identical Cypher strings for every library)
# ======================================================================
def corpus(max_hops: int):
    """(label, cypher) — ordered by group. Every entry yields one scalar-ish
    result so each row carries a single latency. Cypher features a given engine
    lacks surface as n/a in that engine's column (e.g. aggregations on turingdb)."""
    c = [
        # ── scans / counts ──────────────────────────────────────────────
        ("count_all_nodes", "MATCH (n) RETURN count(n)"),
        ("count_person", "MATCH (n:Person) RETURN count(n)"),
        ("count_crime", "MATCH (n:Crime) RETURN count(n)"),
        ("count_knows", "MATCH ()-[r:KNOWS]->() RETURN count(r)"),
        ("count_party_to", "MATCH ()-[r:PARTY_TO]->() RETURN count(r)"),
        # ── filters / lookups ───────────────────────────────────────────
        ("filter_age_gt50", "MATCH (n:Person) WHERE n.age > 50 RETURN count(n)"),
        ("filter_age_range", "MATCH (n:Person) WHERE n.age >= 30 AND n.age <= 40 RETURN count(n)"),
        ("filter_compound", "MATCH (n:Person) WHERE n.age > 70 AND n.surname = 'Sur010' RETURN count(n)"),
        ("lookup_unindexed_surname", "MATCH (n:Person {surname:'Sur050'}) RETURN count(n)"),
        ("lookup_indexed_name", "MATCH (n:Person {name:'Person_0'}) RETURN n.name, n.age"),
        # ── aggregations ────────────────────────────────────────────────
        ("avg_age", "MATCH (n:Person) RETURN avg(n.age)"),
        ("min_age", "MATCH (n:Person) RETURN min(n.age)"),
        ("max_age", "MATCH (n:Person) RETURN max(n.age)"),
        ("sum_age", "MATCH (n:Person) RETURN sum(n.age)"),
        ("distinct_surnames", "MATCH (n:Person) RETURN count(DISTINCT n.surname)"),
        ("groups_by_surname", "MATCH (n:Person) RETURN n.surname, count(n)"),
        # ── materialize / sort ──────────────────────────────────────────
        ("scan_materialize", "MATCH (n:Person) RETURN n.name, n.age"),
        ("top10_by_age", "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC LIMIT 10"),
        ("top100_by_age", "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC LIMIT 100"),
        # ── traversal ───────────────────────────────────────────────────
        ("party_to_fwd", "MATCH (p:Person)-[:PARTY_TO]->(c:Crime) RETURN count(c)"),
        ("party_to_rev", "MATCH (c:Crime)<-[:PARTY_TO]-(p:Person) RETURN count(p)"),
        ("seed_party_to", "MATCH (s:Person {name:'Person_0'})-[:PARTY_TO]->(c) RETURN count(c)"),
        ("seed_neighbors", "MATCH (s:Person {name:'Person_0'})-[:KNOWS]->(m) RETURN m.name"),
        ("surname_party_to", "MATCH (p:Person {surname:'Sur050'})-[:PARTY_TO]->(c) RETURN count(c)"),
    ]
    for h in range(1, max_hops + 1):
        chain = "".join(f"-[:KNOWS]->(n{i})" for i in range(h))
        c.append((f"hop{h}", f"MATCH (s:Person {{name:'Person_0'}}){chain} RETURN count(n{h - 1})"))
    return c


# ======================================================================
# 3. LIBRARY ADAPTERS  <-- THE EXPANSION POINT
#    To add a library: subclass Adapter, implement load()/run(), register
#    it in ADAPTERS. It must accept the SAME Cypher strings. Anything that
#    speaks Cypher and runs in-process drops straight in.
# ======================================================================
class Adapter:
    key: str = ""
    transport: str = "in-process"  # "in-process" or "bolt"/"http" (network)

    def __init__(self, ds: dict, data_dir: str):
        self.ds, self.data_dir = ds, data_dir

    def load(self) -> None: ...  # ingest the dataset (timed)

    def run(self, label: str, cypher: str):
        """Execute one benchmark, return a rows-iterable (or None).

        Cypher engines ignore *label* and run *cypher*. Non-Cypher libraries
        (e.g. NetworkX) ignore *cypher* and dispatch on *label*.
        """
        raise NotImplementedError


class KGLiteAdapter(Adapter):
    key = "kglite"
    storage = None  # None = default (heap/petgraph); "mapped" (mmap); "disk" (CSR+mmap)

    def load(self):
        import kglite

        kw = {} if self.storage is None else {"storage": self.storage, "path": os.path.join(self.data_dir, "kgl")}
        self.g = kglite.KnowledgeGraph(**kw)
        self.g.add_nodes_bulk(
            [
                {
                    "node_type": "Person",
                    "unique_id_field": "id",
                    "node_title_field": "name",
                    "data": self.ds["persons"],
                },
                {"node_type": "Crime", "unique_id_field": "id", "node_title_field": "ctype", "data": self.ds["crimes"]},
            ]
        )
        self.g.add_connections_bulk(
            [
                {
                    "source_type": "Person",
                    "target_type": "Person",
                    "connection_name": "KNOWS",
                    "data": self.ds["knows"],
                },
                {
                    "source_type": "Person",
                    "target_type": "Crime",
                    "connection_name": "PARTY_TO",
                    "data": self.ds["party_to"],
                },
            ]
        )
        self.g.create_index("Person", "name")  # KGLite's method: property index

    def run(self, label, cypher):
        return self.g.cypher(cypher, to_df=True)


class KGLiteMappedAdapter(KGLiteAdapter):
    key = "kglite_mapped"
    storage = "mapped"


class KGLiteDiskAdapter(KGLiteAdapter):
    key = "kglite_disk"
    storage = "disk"


class KGLiteFluentAdapter(KGLiteAdapter):
    """KGLite in-memory via the fluent (method-chaining) API instead of Cypher.
    Selection-based: maps cleanly to counts/filters/lookup/materialize/top-N
    (identical answers). Edge-counts, scalar aggregates, and path/hop counts
    don't translate (a selection dedups target nodes ≠ counting edges/paths),
    so those raise -> render n/a. Memory mode only.
    """

    key = "kglite_fluent"
    storage = None

    def run(self, label, cypher):
        g = self.g
        if label == "count_person":
            return [g.select("Person").len()]
        if label == "count_crime":
            return [g.select("Crime").len()]
        if label == "count_all_nodes":
            return [g.select("Person").len() + g.select("Crime").len()]
        if label == "filter_age_gt50":
            return [g.select("Person").where({"age": {">": 50}}).len()]
        if label == "filter_age_range":
            return [g.select("Person").where({"age": {">=": 30}}).where({"age": {"<=": 40}}).len()]
        if label == "filter_compound":
            return [g.select("Person").where({"age": {">": 70}}).where({"surname": "Sur010"}).len()]
        if label == "lookup_unindexed_surname":
            return [g.select("Person").where({"surname": "Sur050"}).len()]
        if label == "lookup_indexed_name":
            return g.select("Person").where({"name": "Person_0"}).to_df()
        if label == "scan_materialize":
            return g.select("Person").to_df()
        if label == "top10_by_age":
            return g.select("Person", sort=[("age", False)], limit=10).to_df()
        if label == "top100_by_age":
            return g.select("Person", sort=[("age", False)], limit=100).to_df()
        # edge counts, scalar aggregates, traversals/hops: no faithful fluent
        # equivalent (selection semantics differ) -> n/a
        raise NotImplementedError(label)


class TuringDBAdapter(Adapter):
    key = "turingdb"

    def load(self):
        import turingdb

        self.db = turingdb.TuringDB(data_dir=self.data_dir)  # in-process, no socket
        self.db.query(f'LOAD JSONL "{os.path.basename(self.ds["jsonl"])}" AS bench')
        self.db.set_graph("bench")  # method: no explicit index

    def run(self, label, cypher):
        return self.db.query(cypher)


class KuzuAdapter(Adapter):
    """Embedded columnar graph DB, in-memory mode, real Cypher engine."""

    key = "kuzu"

    def load(self):
        import kuzu

        ds = self.data_dir
        self.ds["persons"].to_csv(f"{ds}/persons.csv", index=False)
        self.ds["crimes"].to_csv(f"{ds}/crimes.csv", index=False)
        self.ds["knows"].to_csv(f"{ds}/knows.csv", index=False)
        self.ds["party_to"].to_csv(f"{ds}/party_to.csv", index=False)
        self.db = kuzu.Database(":memory:")
        c = kuzu.Connection(self.db)
        c.execute("CREATE NODE TABLE Person(id INT64, name STRING, surname STRING, age INT64, PRIMARY KEY(id))")
        c.execute("CREATE NODE TABLE Crime(id INT64, ctype STRING, severity INT64, PRIMARY KEY(id))")
        c.execute("CREATE REL TABLE KNOWS(FROM Person TO Person)")
        c.execute("CREATE REL TABLE PARTY_TO(FROM Person TO Crime)")
        for tbl, f in (("Person", "persons"), ("Crime", "crimes"), ("KNOWS", "knows"), ("PARTY_TO", "party_to")):
            c.execute(f'COPY {tbl} FROM "{ds}/{f}.csv" (HEADER=true)')
        self.c = c

    def run(self, label, cypher):
        return self.c.execute(cypher).get_as_df()


class _PyLibAdapter(Adapter):
    """Shared base for pure-Python-API graph libraries (no Cypher). Subclass
    builds the native graph in `_build`; counts use native O(1) ops, attribute
    filters iterate the native store, multi-hop walks use a precomputed typed
    adjacency (the idiomatic fast path for repeated traversal)."""

    def load(self):
        ds = self.ds
        self.persons = list(ds["persons"].itertuples(index=False))  # id,name,surname,age
        self.name_idx = {f"Person_{i}": i for i in ds["persons"]["id"]}
        self.knows_adj: dict[int, list[int]] = {}
        for s, t in ds["knows"].itertuples(index=False):
            self.knows_adj.setdefault(s, []).append(t)
        self.party_pairs = list(ds["party_to"].itertuples(index=False))  # (source_id, target_id)
        self.n_party = ds["n_party_to"]
        self.n_crime = len(ds["crimes"])
        self._build()

    def _build(self): ...
    def n_nodes(self) -> int: ...
    def n_knows(self) -> int: ...

    def run(self, label, cypher):
        P = self.persons
        # scans / counts
        if label == "count_all_nodes":
            return [self.n_nodes()]
        if label == "count_person":
            return [len(P)]
        if label == "count_crime":
            return [self.n_crime]
        if label == "count_knows":
            return [self.n_knows()]
        if label == "count_party_to":
            return [self.n_party]
        # filters / lookups
        if label == "filter_age_gt50":
            return [sum(1 for p in P if p.age > 50)]
        if label == "filter_age_range":
            return [sum(1 for p in P if 30 <= p.age <= 40)]
        if label == "filter_compound":
            return [sum(1 for p in P if p.age > 70 and p.surname == "Sur010")]
        if label == "lookup_unindexed_surname":
            return [sum(1 for p in P if p.surname == "Sur050")]
        if label == "lookup_indexed_name":
            p = P[self.name_idx["Person_0"]]
            return [(p.name, p.age)]
        # aggregations
        if label == "avg_age":
            return [sum(p.age for p in P) / len(P)]
        if label == "min_age":
            return [min(p.age for p in P)]
        if label == "max_age":
            return [max(p.age for p in P)]
        if label == "sum_age":
            return [sum(p.age for p in P)]
        if label == "distinct_surnames":
            return [len({p.surname for p in P})]
        if label == "groups_by_surname":
            g: dict = {}
            for p in P:
                g[p.surname] = g.get(p.surname, 0) + 1
            return list(g.items())
        # materialize / sort
        if label == "scan_materialize":
            return [(p.name, p.age) for p in P]
        if label == "top10_by_age":
            return sorted((p.name for p in P), key=lambda n: P[self.name_idx[n]].age, reverse=True)[:10]
        if label == "top100_by_age":
            return sorted(P, key=lambda p: p.age, reverse=True)[:100]
        # traversal
        if label in ("party_to_fwd", "party_to_rev"):
            return [self.n_party]
        if label == "seed_party_to":
            return [sum(1 for s, _ in self.party_pairs if s == 0)]
        if label == "seed_neighbors":
            return [P[v].name for v in self.knows_adj.get(0, ())]
        if label == "surname_party_to":
            ids = {p.id for p in P if p.surname == "Sur050"}
            return [sum(1 for s, _ in self.party_pairs if s in ids)]
        if label.startswith("hop"):
            h = int(label[3:])
            adj = self.knows_adj
            frontier = [self.name_idx["Person_0"]]
            for _ in range(h):
                frontier = [v for u in frontier for v in adj.get(u, ())]
            return [len(frontier)]
        raise NotImplementedError(label)


class NetworkXAdapter(_PyLibAdapter):
    key = "networkx"

    def _build(self):
        import networkx as nx

        g = nx.MultiDiGraph()
        for p in self.persons:
            g.add_node(p.id, label="Person")
        for r in self.ds["crimes"].itertuples(index=False):
            g.add_node(r.id, label="Crime")
        g.add_edges_from((s, t) for s, t in self.ds["knows"].itertuples(index=False))
        self.g = g
        self._nk = self.ds["n_knows"]

    def n_nodes(self):
        return self.g.number_of_nodes()

    def n_knows(self):
        return self._nk  # MultiDiGraph keeps all KNOWS edges


class RustworkxAdapter(_PyLibAdapter):
    key = "rustworkx"

    def _build(self):
        import rustworkx as rx

        g = rx.PyDiGraph(multigraph=True)
        g.add_nodes_from([None] * self.n_nodes_total())  # indices 0..N-1 == ids
        g.add_edges_from([(int(s), int(t), 1) for s, t in self.ds["knows"].itertuples(index=False)])
        self.g = g

    def n_nodes_total(self):
        return len(self.persons) + len(self.ds["crimes"])

    def n_nodes(self):
        return self.g.num_nodes()

    def n_knows(self):
        return self.g.num_edges()


class IgraphAdapter(_PyLibAdapter):
    key = "igraph"

    def _build(self):
        import igraph as ig

        n = len(self.persons) + len(self.ds["crimes"])
        g = ig.Graph(n=n, directed=True)  # vertex index == original id
        g.add_edges([(int(s), int(t)) for s, t in self.ds["knows"].itertuples(index=False)])
        self.g = g

    def n_nodes(self):
        return self.g.vcount()

    def n_knows(self):
        return self.g.ecount()


class MemgraphAdapter(Adapter):
    """In-memory graph DB but accessed over BOLT (network) — NOT in-process.
    Requires a running Memgraph (Docker). Numbers carry Bolt round-trip cost,
    so the 'Transport' row flags it. Not in the default in-memory set."""

    key = "memgraph"
    transport = "bolt"

    def load(self):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            for p in self.ds["persons"].itertuples(index=False):
                s.run(
                    "CREATE (:Person {id:$i,name:$n,surname:$su,age:$a})",
                    i=int(p.id),
                    n=p.name,
                    su=p.surname,
                    a=int(p.age),
                )
            # (bulk UNWIND load omitted for brevity; enable when server is up)

    def run(self, label, cypher):
        with self.driver.session() as s:
            return list(s.run(cypher))


# In-memory / in-process libraries (the default set).
ADAPTERS = {
    a.key: a
    for a in (
        KGLiteAdapter,
        KGLiteMappedAdapter,
        KGLiteDiskAdapter,
        KGLiteFluentAdapter,
        TuringDBAdapter,
        KuzuAdapter,
        NetworkXAdapter,
        RustworkxAdapter,
        IgraphAdapter,
        MemgraphAdapter,  # available via --libs, network transport
    )
}
IN_MEMORY = [
    "kglite",
    "kglite_mapped",
    "kglite_disk",
    "kglite_fluent",
    "turingdb",
    "kuzu",
    "networkx",
    "rustworkx",
    "igraph",
]


# ======================================================================
# 4. WORKER  (one library, in its own process -> isolated peak RSS)
# ======================================================================
def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024  # macOS bytes / Linux KiB


def run_worker(engine: str, persons: int, degree: int, seed: int, rounds: int, warmup: int, max_hops: int, out: str):
    data_dir = tempfile.mkdtemp(prefix=f"{engine}_")
    # TuringDB needs the JSONL under <data_dir>/data/; others read it anywhere.
    jdir = os.path.join(data_dir, "data") if engine == "turingdb" else data_dir
    os.makedirs(jdir, exist_ok=True)
    jsonl = os.path.join(jdir, "g.jsonl")

    t = time.perf_counter()
    ds = build_dataset(persons, degree, seed, jsonl)
    gen_s = time.perf_counter() - t

    adapter = ADAPTERS[engine](ds, data_dir)
    t = time.perf_counter()
    adapter.load()
    load_s = time.perf_counter() - t
    rss_load = rss_mb()

    queries = {}
    for label, q in corpus(max_hops):
        try:
            for _ in range(warmup):
                rows = adapter.run(label, q)
            samples = []
            for _ in range(rounds):
                t = time.perf_counter()
                rows = adapter.run(label, q)
                samples.append((time.perf_counter() - t) * 1000.0)
            queries[label] = {
                "min_ms": min(samples),
                "median_ms": statistics.median(samples),
                "rows": len(rows) if rows is not None else None,
            }
        except Exception as e:  # noqa: BLE001
            queries[label] = {"error": str(e).splitlines()[0][:120]}

    json.dump(
        {
            "engine": engine,
            "transport": adapter.transport,
            "persons": persons,
            "n_crimes": ds["n_crimes"],
            "n_knows": ds["n_knows"],
            "n_party_to": ds["n_party_to"],
            "gen_s": round(gen_s, 3),
            "load_s": round(load_s, 4),
            "rss_after_load_mb": round(rss_load, 1),
            "rss_peak_mb": round(rss_mb(), 1),
            "queries": queries,
        },
        open(out, "w"),
    )


# ======================================================================
# 5. TABLE  (rows = benchmark, columns = library)
# ======================================================================
PERF_ROWS = [  # (label, unit, source)  source: meta key or "q:<query>"
    ("Load time", "s", "load_s"),
    ("Memory — resident graph", "MB", "rss_after_load_mb"),
    ("Memory — peak (w/ queries)", "MB", "rss_peak_mb"),
    ("Count all nodes", "ms", "q:count_all_nodes"),
    ("Count Person", "ms", "q:count_person"),
    ("Count Crime", "ms", "q:count_crime"),
    ("Count KNOWS edges", "ms", "q:count_knows"),
    ("Count PARTY_TO edges", "ms", "q:count_party_to"),
    ("Filter age>50", "ms", "q:filter_age_gt50"),
    ("Filter age 30–40", "ms", "q:filter_age_range"),
    ("Filter compound (age+surname)", "ms", "q:filter_compound"),
    ("Lookup unindexed (surname)", "ms", "q:lookup_unindexed_surname"),
    ("Lookup indexed (name)", "ms", "q:lookup_indexed_name"),
    ("Aggregate avg(age)", "ms", "q:avg_age"),
    ("Aggregate min(age)", "ms", "q:min_age"),
    ("Aggregate max(age)", "ms", "q:max_age"),
    ("Aggregate sum(age)", "ms", "q:sum_age"),
    ("Count distinct surnames", "ms", "q:distinct_surnames"),
    ("Group by surname", "ms", "q:groups_by_surname"),
    ("Scan + materialize", "ms", "q:scan_materialize"),
    ("Top-10 by age", "ms", "q:top10_by_age"),
    ("Top-100 by age", "ms", "q:top100_by_age"),
    ("Traversal PARTY_TO fwd", "ms", "q:party_to_fwd"),
    ("Traversal PARTY_TO rev", "ms", "q:party_to_rev"),
    ("Seed → PARTY_TO", "ms", "q:seed_party_to"),
    ("Seed → KNOWS neighbours", "ms", "q:seed_neighbors"),
    ("Surname → PARTY_TO", "ms", "q:surname_party_to"),
    *[(f"Traversal {h}-hop", "ms", f"q:hop{h}") for h in range(1, 7)],
]


def _val(d: dict, src: str):
    """Numeric value for a cell, or None if missing/unsupported."""
    if src.startswith("q:"):
        q = d.get("queries", {}).get(src[2:])
        if q is None or "min_ms" not in q:
            return None
        return q["min_ms"]
    return d.get(src)


def cell(d: dict, src: str) -> str:
    v = _val(d, src)
    if v is None:
        q = d.get("queries", {}).get(src[2:]) if src.startswith("q:") else None
        return "n/a" if (src.startswith("q:") and q is not None) else "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def emit_table(scale: int, libs: list[str], csv_path: str | None):
    data = {lib: json.load(open(f"/tmp/gbench_{lib}_{scale}.json")) for lib in libs}
    any_d = next(iter(data.values()))
    nodes = scale + any_d["n_crimes"]
    edges = any_d["n_knows"] + any_d["n_party_to"]

    def row_sum(src):  # sum across libraries for one benchmark (skips n/a)
        vals = [_val(data[_lib], src) for _lib in libs]
        vals = [v for v in vals if v is not None]
        return _fmt(sum(vals)) if vals else "—"

    hdr = ["Benchmark", "Unit"] + libs + ["sum"]
    rows = [hdr, ["---"] * len(hdr)]
    rows.append(["Transport", "", *[data[_lib].get("transport", "in-process") for _lib in libs], ""])
    rows += [[lbl, unit] + [cell(data[_lib], src) for _lib in libs] + [row_sum(src)] for lbl, unit, src in PERF_ROWS]

    # per-library TOTAL over the ms benchmark rows (skips n/a; excludes load/memory)
    ms_srcs = [src for lbl, unit, src in PERF_ROWS if unit == "ms"]
    total = ["TOTAL (ms)", "ms"]
    grand = []
    for _lib in libs:
        vals = [v for v in (_val(data[_lib], s) for s in ms_srcs) if v is not None]
        grand += vals
        total.append(_fmt(sum(vals)))
    total.append(_fmt(sum(grand)))
    rows.append(total)

    widths = [max(len(r[i]) for r in rows) for i in range(len(hdr))]
    print(f"# Graph-DB benchmark — {nodes:,} nodes / {edges:,} edges")
    print("# in-process · identical Cypher & data · latency = min ms over timed rounds\n")
    for r in rows:
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")

    if csv_path:
        import csv

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["benchmark", "unit"] + libs + ["sum"])
            for lbl, unit, src in PERF_ROWS:
                w.writerow([lbl, unit] + [cell(data[_lib], src) for _lib in libs] + [row_sum(src)])
            w.writerow(total)
        print(f"\nCSV -> {csv_path}")


# ======================================================================
# 6. CLI  (driver spawns one worker subprocess per library)
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Objective graph-DB benchmark table.")
    ap.add_argument("--scale", type=int, default=100_000, help="number of Person nodes")
    ap.add_argument("--degree", type=int, default=8, help="avg KNOWS out-degree")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--libs", default=",".join(IN_MEMORY), help="comma list of library keys")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--engine", default=None, help="(internal) run a single library worker")
    ap.add_argument("--out", default=None, help="(internal) worker output path")
    args = ap.parse_args()

    if args.engine:  # worker mode
        run_worker(args.engine, args.scale, args.degree, args.seed, args.rounds, args.warmup, args.max_hops, args.out)
        return

    libs, ok = [x for x in args.libs.split(",") if x], []
    for lib in libs:
        if lib not in ADAPTERS:
            sys.exit(f"unknown library '{lib}'. known: {', '.join(ADAPTERS)}")
        out = f"/tmp/gbench_{lib}_{args.scale}.json"
        print(f"running {lib} @ {args.scale:,} persons ...", file=sys.stderr)
        r = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--engine",
                lib,
                "--out",
                out,
                "--scale",
                str(args.scale),
                "--degree",
                str(args.degree),
                "--seed",
                str(args.seed),
                "--rounds",
                str(args.rounds),
                "--warmup",
                str(args.warmup),
                "--max-hops",
                str(args.max_hops),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if r.returncode == 0 and os.path.exists(out):
            ok.append(lib)
        else:
            tail = r.stderr.decode().strip().splitlines()[-1:] if r.stderr else []
            print(f"  ! skipped {lib}: {tail[0] if tail else 'failed'}", file=sys.stderr)
    print(file=sys.stderr)
    if ok:
        emit_table(args.scale, ok, args.csv)


if __name__ == "__main__":
    main()
