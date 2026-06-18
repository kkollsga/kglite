"""kglite adapters: in-memory via Cypher, via the fluent API, and over Bolt.

All three drive the *same* in-memory kglite engine; they differ only in
the surface used to reach it (Cypher string / fluent builder / Bolt wire
protocol). Connected-components and PageRank are intentionally skipped —
kglite is a knowledge-graph engine, not a graph-analytics library, and
has no native WCC/PageRank primitive (those belong to igraph/rustworkx).
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import kglite

from .base import Adapter, Skip
from .dataset import Dataset

# edge type -> (source node type, target node type)
EDGE_ENDPOINTS = {
    "KNOWS": ("Person", "Person"),
    "WORKS_AT": ("Person", "Company"),
    "CONTRIBUTES_TO": ("Person", "Project"),
    "HAS_SKILL": ("Person", "Skill"),
    "OWNS": ("Company", "Project"),
    "DEPENDS_ON": ("Project", "Project"),
    "LOCATED_IN": ("Company", "City"),
}


def build_kglite_graph(ds: Dataset, storage: str | None = None, path: str | None = None) -> kglite.KnowledgeGraph:
    """Shared builder used by every kglite surface and storage mode.

    `storage` is ``None`` (heap, default), ``"mapped"`` (mmap-backed
    columnar) or ``"disk"`` (fully disk-backed; needs `path`)."""
    if storage == "disk":
        g = kglite.KnowledgeGraph(storage="disk", path=path)
    elif storage == "mapped":
        g = kglite.KnowledgeGraph(storage="mapped")
    else:
        g = kglite.KnowledgeGraph()
    for ntype, rows in ds.nodes.items():
        df = ds.node_frame(ntype)
        g.add_nodes(df, ntype, "gid", "name")
    for etype, (st, tt) in EDGE_ENDPOINTS.items():
        df = ds.edge_frame(etype)
        propcols = [c for c in df.columns if c not in ("src", "dst")]
        g.add_connections(df, etype, st, "src", tt, "dst", columns=propcols or None)
    return g


class KgliteCypher(Adapter):
    name = "kglite-memory-cypher"

    def version(self) -> str:
        return kglite.__version__

    def build(self, ds: Dataset) -> None:
        self.g = build_kglite_graph(ds)
        self._mut = 0

    def _q(self, query, **params):
        return self.g.cypher(query, params=params or None)

    def g_node_scan(self, ds):
        c = self._q("MATCH (n:Person) RETURN count(n) AS c").scalar()
        ids = self._q("MATCH (n:Person) RETURN n.id AS id").column("id")
        assert len(ids) == c
        return len(ids)

    def g_point_lookup(self, ds):
        found = 0
        for gid in ds.params["lookup_ids"]:
            if self.g.node("Person", gid) is not None:
                found += 1
        return found

    def g_property_filter(self, ds):
        ids = self._q(
            "MATCH (n:Person) WHERE n.age > $age AND n.city = $city RETURN n.id AS id",
            age=ds.params["filter_age"],
            city=ds.params["filter_city"],
        ).column("id")
        return frozenset(ids)

    def g_group_aggregation(self, ds):
        rows = self._q("MATCH (n:Person) RETURN n.city AS city, count(n) AS c, avg(n.age) AS a").to_list()
        return {r["city"]: (r["c"], r["a"]) for r in rows}

    # Seeded traversals use the UNWIND-anchored form so the id index drives
    # the scan. (The `... WHERE p.id IN $ids` form does NOT anchor on the
    # index in the current planner and is ~240x slower — see README findings.)
    # k-hop neighbourhood = distinct nodes reachable within k hops. Each
    # engine uses its idiomatic form; counts agree to <1% (the only delta is
    # walk-vs-trail handling of paths that return to a seed — see README).
    def g_one_hop(self, ds):
        return frozenset(
            self._q(
                "UNWIND $ids AS sid MATCH (p:Person {id:sid})-[:KNOWS]-(f:Person) RETURN DISTINCT f.id AS id",
                ids=ds.params["seed_persons"],
            ).column("id")
        )

    def g_two_hop(self, ds):
        return frozenset(
            self._q(
                "UNWIND $ids AS sid MATCH (p:Person {id:sid})-[:KNOWS*1..2]-(f:Person) RETURN DISTINCT f.id AS id",
                ids=ds.params["seed_persons_small"],
            ).column("id")
        )

    def g_three_hop(self, ds):
        return frozenset(
            self._q(
                "UNWIND $ids AS sid MATCH (p:Person {id:sid})-[:KNOWS*1..3]-(f:Person) RETURN DISTINCT f.id AS id",
                ids=ds.params["seed_persons_tiny"],
            ).column("id")
        )

    def g_filtered_traversal(self, ds):
        return frozenset(
            self._q(
                "UNWIND $ids AS sid MATCH (p:Person {id:sid})-[:KNOWS]-(f:Person) "
                "WHERE f.age < 30 RETURN DISTINCT f.id AS id",
                ids=ds.params["seed_persons"],
            ).column("id")
        )

    def g_deep_traversal(self, ds):
        return frozenset(
            self._q(
                "UNWIND $ids AS sid MATCH (p:Project {id:sid})-[:DEPENDS_ON*1..15]->(d:Project) "
                "RETURN DISTINCT d.id AS id",
                ids=ds.params["seed_projects"],
            ).column("id")
        )

    def g_shortest_path(self, ds):
        lengths = []
        for a, b in ds.params["sp_pairs"]:
            r = self._q(
                "MATCH path = shortestPath((a:Person {id:$a})-[:KNOWS*]-(b:Person {id:$b})) RETURN length(path) AS L",
                a=a,
                b=b,
            ).to_list()
            lengths.append(r[0]["L"] if r and r[0].get("L") is not None else None)
        return tuple(lengths)

    def g_pattern_match(self, ds):
        return self._q(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:OWNS]->(pr:Project)"
            "<-[:CONTRIBUTES_TO]-(p) RETURN count(*) AS c"
        ).scalar()

    def g_degree_topk(self, ds):
        rows = self._q(
            "MATCH (p:Person) WITH p, COUNT { (p)-[:KNOWS]-() } AS deg "
            "RETURN p.id AS id, deg ORDER BY deg DESC LIMIT $k",
            k=ds.params["topk"],
        ).to_list()
        return tuple(r["deg"] for r in rows)

    def g_connected_components(self, ds):
        # Scoped WCC over the Person/KNOWS subgraph — `node_type` makes every
        # Person the universe (isolated persons = singletons) and `relationship`
        # restricts the unioning edges, matching the graph-algo libraries.
        rows = self._q(
            "CALL connected_components({node_type: 'Person', relationship: 'KNOWS'}) "
            "YIELD node, component RETURN component AS c, count(*) AS size"
        ).to_list()
        num = len(rows)
        largest = max((r["size"] for r in rows), default=0)
        return (num, largest)

    def g_mutations(self, ds):
        off = ds.params["mut_new_base"] + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        rows = [{"id": off + i, "age": 30 + (i % 40), "name": f"M{off + i}"} for i in range(n)]
        self._q("UNWIND $rows AS r CREATE (n:Person {id:r.id, age:r.age, name:r.name})", rows=rows)
        # connect each new node to its predecessor
        pairs = [{"a": off + i, "b": off + i - 1} for i in range(1, n)]
        self._q(
            "UNWIND $pairs AS p MATCH (a:Person {id:p.a}), (b:Person {id:p.b}) CREATE (a)-[:KNOWS]->(b)",
            pairs=pairs,
        )
        # update ages on the just-created nodes
        ups = [{"id": off + i, "age": 99} for i in range(n)]
        self._q("UNWIND $rows AS r MATCH (n:Person {id:r.id}) SET n.age = r.age", rows=ups)
        # delete a subset
        dels = [off + i for i in range(0, n, 3)]
        self._q("UNWIND $ids AS i MATCH (n:Person {id:i}) DETACH DELETE n", ids=dels)
        return len(dels)


class KgliteFluent(Adapter):
    """Same engine via the fluent select/where/traverse builder.

    Skips groups the fluent surface cannot express directly (shortest
    path, multi-edge cyclic pattern match, WCC)."""

    name = "kglite-memory-fluent"

    def version(self) -> str:
        return kglite.__version__

    def build(self, ds: Dataset) -> None:
        self.g = build_kglite_graph(ds)
        self._mut = 0

    def g_node_scan(self, ds):
        return self.g.select("Person").len()

    def g_point_lookup(self, ds):
        found = 0
        for gid in ds.params["lookup_ids"]:
            if self.g.node("Person", gid) is not None:
                found += 1
        return found

    def g_property_filter(self, ds):
        return frozenset(
            self.g.select("Person")
            .where({"age": {">": ds.params["filter_age"]}, "city": ds.params["filter_city"]})
            .ids()
        )

    def g_group_aggregation(self, ds):
        # statistics(group_by) returns {city: {count, mean, ...}} — projected to
        # {city: (count, mean)} to match the Cypher count + avg(age) result.
        stats = self.g.select("Person").statistics("age", group_by="city")
        return {c: (s["count"], s["mean"]) for c, s in stats.items()}

    def g_one_hop(self, ds):
        sel = self.g.select("Person").where({"id": {"in": ds.params["seed_persons"]}}).traverse("KNOWS")
        return frozenset(sel.ids())

    def g_two_hop(self, ds):
        l1 = self.g.select("Person").where({"id": {"in": ds.params["seed_persons_small"]}}).traverse("KNOWS")
        seen = set(l1.ids())
        seen.update(l1.traverse("KNOWS").ids())
        return frozenset(seen)

    def g_three_hop(self, ds):
        l1 = self.g.select("Person").where({"id": {"in": ds.params["seed_persons_tiny"]}}).traverse("KNOWS")
        seen = set(l1.ids())
        l2 = l1.traverse("KNOWS")
        seen.update(l2.ids())
        seen.update(l2.traverse("KNOWS").ids())
        return frozenset(seen)

    def g_filtered_traversal(self, ds):
        sel = (
            self.g.select("Person")
            .where({"id": {"in": ds.params["seed_persons"]}})
            .traverse("KNOWS", where={"age": {"<": 30}})
        )
        return frozenset(sel.ids())

    def g_deep_traversal(self, ds):
        sel = self.g.select("Project").where({"id": {"in": ds.params["seed_projects"]}})
        seen: set = set()
        # iterative deepening via repeated traverse (fluent has no var-length);
        # accumulate the distinct reachable set to match the Cypher closure.
        for _ in range(15):
            sel = sel.traverse("DEPENDS_ON", direction="outgoing")
            ids = sel.ids()
            if not ids:
                break
            seen.update(ids)
        return frozenset(seen)

    def g_mutations(self, ds):
        import pandas as pd

        off = ds.params["mut_new_base"] + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        df = pd.DataFrame(
            {
                "gid": [off + i for i in range(n)],
                "name": [f"M{off + i}" for i in range(n)],
                "age": [30 + (i % 40) for i in range(n)],
            }
        )
        self.g.add_nodes(df, "Person", "gid", "name")
        edf = pd.DataFrame({"src": [off + i for i in range(1, n)], "dst": [off + i - 1 for i in range(1, n)]})
        self.g.add_connections(edf, "KNOWS", "Person", "src", "Person", "dst")
        # update via fluent
        self.g.select("Person").where({"id": {"in": [off + i for i in range(n)]}}).update({"age": 99})
        return n

    # explicitly unsupported on the fluent surface
    def g_degree_topk(self, ds):
        raise Skip("fluent API has no degree+rank primitive")

    def g_shortest_path(self, ds):
        raise Skip("fluent API has no shortestPath")

    def g_pattern_match(self, ds):
        raise Skip("fluent API cannot express the cyclic triangle pattern")

    def g_connected_components(self, ds):
        raise Skip("fluent API can't CALL procedures; use Cypher connected_components")


class KgliteBolt(KgliteCypher):
    """Same Cypher workloads sent over the Bolt wire protocol via the
    neo4j Python driver. Reveals the wire/serialisation tax vs the direct
    in-process Cypher adapter. Point-lookup and shortest-path issue one
    round-trip per id/pair to expose per-query latency."""

    name = "kglite-memory-bolt"

    def version(self) -> str:
        return kglite.__version__

    def available(self) -> tuple[bool, str]:
        try:
            from tests.conftest import _BOLT_BINARY  # noqa
        except Exception as e:  # pragma: no cover
            return False, f"conftest import failed: {e}"
        if not _BOLT_BINARY.exists():
            return False, f"bolt binary not built at {_BOLT_BINARY}"
        try:
            import neo4j  # noqa
        except Exception as e:
            return False, f"neo4j driver missing: {e}"
        return True, ""

    def build(self, ds: Dataset) -> None:
        import neo4j

        from tests.conftest import _spawn_bolt_server, _teardown_bolt_server

        self._teardown_fn = _teardown_bolt_server
        g = build_kglite_graph(ds)
        self._tmpdir = tempfile.mkdtemp(prefix="graphsuite_bolt_")
        fixture = Path(self._tmpdir) / "graph.kgl"
        g.save(str(fixture))
        self._proc, url = _spawn_bolt_server(fixture)
        self._driver = neo4j.GraphDatabase.driver(url, auth=("neo4j", "password"))
        self._session = self._driver.session()
        self._mut = 0
        self._session.run("RETURN 1").consume()  # warm

    def teardown(self) -> None:
        try:
            self._session.close()
            self._driver.close()
        finally:
            self._teardown_fn(self._proc)

    def _q(self, query, **params):
        # mimic the ResultView surface used by the parent class
        recs = list(self._session.run(query, **params))

        class _R:
            def __init__(self, rows):
                self._rows = rows

            def to_list(self):
                return [dict(r) for r in self._rows]

            def column(self, name):
                return [r[name] for r in self._rows]

            def scalar(self):
                if not self._rows:
                    return None
                r = self._rows[0]
                return r[r.keys()[0]]

        return _R(recs)

    def g_point_lookup(self, ds):
        found = 0
        for gid in ds.params["lookup_ids"]:
            rows = list(self._session.run("MATCH (n:Person) WHERE n.id = $id RETURN n.age AS a", id=gid))
            if rows:
                found += 1
        return found

    def g_mutations(self, ds):
        # kglite-bolt-server rejects auto-commit writes — mutations must run
        # inside an explicit transaction.
        off = ds.params["mut_new_base"] + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        rows = [{"id": off + i, "age": 30 + (i % 40), "name": f"M{off + i}"} for i in range(n)]
        pairs = [{"a": off + i, "b": off + i - 1} for i in range(1, n)]
        ups = [{"id": off + i, "age": 99} for i in range(n)]
        dels = [off + i for i in range(0, n, 3)]
        tx = self._session.begin_transaction()
        try:
            tx.run("UNWIND $rows AS r CREATE (n:Person {id:r.id, age:r.age, name:r.name})", rows=rows)
            tx.run(
                "UNWIND $pairs AS p MATCH (a:Person {id:p.a}), (b:Person {id:p.b}) CREATE (a)-[:KNOWS]->(b)",
                pairs=pairs,
            )
            tx.run("UNWIND $rows AS r MATCH (n:Person {id:r.id}) SET n.age = r.age", rows=ups)
            tx.run("UNWIND $ids AS i MATCH (n:Person {id:i}) DETACH DELETE n", ids=dels)
            tx.commit()
        except Exception:
            tx.close()
            raise
        return len(dels)


class KgliteMapped(KgliteCypher):
    """Same Cypher workloads on an mmap-backed columnar graph
    (``storage='mapped'``). The Cypher planner/executor is shared with the
    in-memory mode, so the delta vs `kglite-memory-cypher` is purely the
    columnar/mmap storage tax. Sized for graphs approaching RAM limits."""

    name = "kglite-mapped-cypher"

    def build(self, ds: Dataset) -> None:
        self.g = build_kglite_graph(ds, storage="mapped")
        self._mut = 0


class KgliteDisk(KgliteCypher):
    """Same Cypher workloads on a fully disk-backed graph
    (``storage='disk'``) — the large-graph (100M+ node) exploration mode.
    The directory IS the graph (mmap CSR). Build writes through to disk."""

    name = "kglite-disk-cypher"

    def build(self, ds: Dataset) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="graphsuite_disk_")
        self.g = build_kglite_graph(ds, storage="disk", path=self._tmpdir)
        self._mut = 0

    def teardown(self) -> None:
        import shutil

        shutil.rmtree(getattr(self, "_tmpdir", ""), ignore_errors=True)

    def g_mutations(self, ds):
        # Cypher CREATE is unsupported on disk-backed graphs; the bulk
        # add_nodes/add_connections loaders are the supported write path.
        # SET/DELETE via Cypher work as normal.
        import pandas as pd

        off = ds.params["mut_new_base"] + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        df = pd.DataFrame(
            {
                "gid": [off + i for i in range(n)],
                "name": [f"M{off + i}" for i in range(n)],
                "age": [30 + (i % 40) for i in range(n)],
            }
        )
        self.g.add_nodes(df, "Person", "gid", "name")
        edf = pd.DataFrame({"src": [off + i for i in range(1, n)], "dst": [off + i - 1 for i in range(1, n)]})
        self.g.add_connections(edf, "KNOWS", "Person", "src", "Person", "dst")
        self._q("MATCH (n:Person) WHERE n.id >= $off SET n.age = 99", off=off)
        dels = [off + i for i in range(0, n, 3)]
        self._q("MATCH (n:Person) WHERE n.id IN $ids DETACH DELETE n", ids=dels)
        return len(dels)
