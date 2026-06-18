"""Neo4j adapter — connects over Bolt to a running Neo4j server.

Neo4j has no true Python-embedded mode (the embedded API is JVM-only),
so this adapter talks to a server via the Bolt driver. It is **opt-in**:
set ``GRAPHSUITE_NEO4J_URI`` (and optionally ``GRAPHSUITE_NEO4J_USER`` /
``GRAPHSUITE_NEO4J_PASSWORD``) to point at a server; otherwise the
adapter reports itself unavailable and is skipped — so a no-server run
stays clean. The wired-up adapter lets a Neo4j run slot into the same
datafile later.

Nodes carry a `gid` property with a uniqueness constraint (the index
that anchors the seeded UNWIND traversals). WCC is skipped (needs the
GDS plugin).
"""

from __future__ import annotations

import os

from .base import Adapter, Skip
from .dataset import DEGREE_MIN, GEO_BBOX, SCORE_MIN, SCORE_RANGE, Dataset

NODE_PROPS = {
    "Person": ["gid", "name", "age", "city", "joined_year", "active", "score"],
    "Company": ["gid", "name", "industry", "size"],
    "Project": ["gid", "name", "budget", "status"],
    "Skill": ["gid", "name", "category"],
    "City": ["gid", "name", "population", "region", "latitude", "longitude"],
}
REL_ENDPOINTS = {
    "KNOWS": ("Person", "Person"),
    "WORKS_AT": ("Person", "Company"),
    "CONTRIBUTES_TO": ("Person", "Project"),
    "HAS_SKILL": ("Person", "Skill"),
    "OWNS": ("Company", "Project"),
    "DEPENDS_ON": ("Project", "Project"),
    "LOCATED_IN": ("Company", "City"),
}
BATCH = 5000


class Neo4jAdapter(Adapter):
    name = "neo4j"

    def version(self) -> str:
        try:
            rec = self._session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()
            return rec["v"] if rec else "server"
        except Exception:
            import neo4j

            return f"driver-{neo4j.__version__}"

    def available(self) -> tuple[bool, str]:
        uri = os.environ.get("GRAPHSUITE_NEO4J_URI")
        if not uri:
            return False, "set GRAPHSUITE_NEO4J_URI to benchmark a Neo4j server"
        try:
            import neo4j  # noqa
        except Exception as e:
            return False, f"neo4j driver missing: {e}"
        try:
            import neo4j

            user = os.environ.get("GRAPHSUITE_NEO4J_USER", "neo4j")
            pw = os.environ.get("GRAPHSUITE_NEO4J_PASSWORD", "password")
            drv = neo4j.GraphDatabase.driver(uri, auth=(user, pw))
            drv.verify_connectivity()
            drv.close()
            return True, ""
        except Exception as e:
            return False, f"cannot reach Neo4j at {uri}: {e}"

    def build(self, ds: Dataset) -> None:
        import neo4j

        uri = os.environ["GRAPHSUITE_NEO4J_URI"]
        user = os.environ.get("GRAPHSUITE_NEO4J_USER", "neo4j")
        pw = os.environ.get("GRAPHSUITE_NEO4J_PASSWORD", "password")
        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, pw))
        self._session = self._driver.session()
        s = self._session
        s.run("MATCH (n) DETACH DELETE n").consume()  # clean slate
        for label in NODE_PROPS:
            s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.gid IS UNIQUE").consume()
        for ntype, props in NODE_PROPS.items():
            rows = [{k: r[k] for k in props} for r in ds.nodes[ntype]]
            for i in range(0, len(rows), BATCH):
                s.run(f"UNWIND $rows AS r CREATE (n:{ntype}) SET n = r", rows=rows[i : i + BATCH]).consume()
        for etype, (ft, tt) in REL_ENDPOINTS.items():
            edges = [{"s": e["src"], "d": e["dst"]} for e in ds.edges[etype]]
            for i in range(0, len(edges), BATCH):
                s.run(
                    f"UNWIND $e AS x MATCH (a:{ft} {{gid:x.s}}), (b:{tt} {{gid:x.d}}) CREATE (a)-[:{etype}]->(b)",
                    e=edges[i : i + BATCH],
                ).consume()
        self._mut = 0

    def teardown(self) -> None:
        try:
            self._session.close()
            self._driver.close()
        except Exception:
            pass

    def _scalar(self, q, **p):
        rec = self._session.run(q, **p).single()
        return rec[0] if rec else None

    def _col(self, q, col, **p):
        return [r[col] for r in self._session.run(q, **p)]

    def g_node_scan(self, ds):
        return self._scalar("MATCH (n:Person) RETURN count(n)")

    def g_point_lookup(self, ds):
        return self._scalar("MATCH (n:Person) WHERE n.gid IN $ids RETURN count(n)", ids=ds.params["lookup_ids"])

    def g_property_filter(self, ds):
        return frozenset(
            self._col(
                "MATCH (n:Person) WHERE n.age > $age AND n.city = $city RETURN n.gid AS g",
                "g",
                age=ds.params["filter_age"],
                city=ds.params["filter_city"],
            )
        )

    def g_group_aggregation(self, ds):
        rows = self._session.run("MATCH (n:Person) RETURN n.city AS city, count(*) AS c, avg(n.age) AS a")
        return {r["city"]: (r["c"], r["a"]) for r in rows}

    def g_edge_scan(self, ds):
        return self._scalar("MATCH ()-[r:KNOWS]->() RETURN count(r)")

    def g_range_filter(self, ds):
        lo, hi = SCORE_RANGE
        return frozenset(
            self._col(
                "MATCH (n:Person) WHERE n.score >= $lo AND n.score <= $hi RETURN n.gid AS g",
                "g",
                lo=lo,
                hi=hi,
            )
        )

    def g_year_aggregation(self, ds):
        rows = self._session.run("MATCH (n:Person) RETURN n.joined_year AS y, count(*) AS c, avg(n.score) AS a")
        return {r["y"]: (r["c"], r["a"]) for r in rows}

    def g_score_filtered_traversal(self, ds):
        return frozenset(
            self._col(
                "UNWIND $ids AS sid MATCH (p:Person {gid:sid})-[:KNOWS]-(f:Person) "
                "WHERE f.score > $mn RETURN DISTINCT f.gid AS g",
                "g",
                ids=ds.params["seed_persons"],
                mn=SCORE_MIN,
            )
        )

    def g_degree_filter(self, ds):
        return self._scalar(
            "MATCH (p:Person)-[:KNOWS]-(x) WITH p, count(*) AS deg WHERE deg >= $k RETURN count(p)",
            k=DEGREE_MIN,
        )

    def g_bulk_update(self, ds):
        return self._scalar(
            "UNWIND $ids AS i MATCH (n:Person {gid:i}) SET n.active = true RETURN count(n)",
            ids=ds.params["lookup_ids"],
        )

    def _anchored(self, pat, ids):
        return frozenset(
            self._col(
                f"UNWIND $ids AS sid MATCH (p:Person {{gid:sid}})-{pat}-(f:Person) RETURN DISTINCT f.gid AS g",
                "g",
                ids=ids,
            )
        )

    def g_one_hop(self, ds):
        return self._anchored("[:KNOWS]", ds.params["seed_persons"])

    def g_two_hop(self, ds):
        return self._anchored("[:KNOWS*1..2]", ds.params["seed_persons_small"])

    def g_three_hop(self, ds):
        return self._anchored("[:KNOWS*1..3]", ds.params["seed_persons_tiny"])

    def g_filtered_traversal(self, ds):
        return frozenset(
            self._col(
                "UNWIND $ids AS sid MATCH (p:Person {gid:sid})-[:KNOWS]-(f:Person) "
                "WHERE f.age < 30 RETURN DISTINCT f.gid AS g",
                "g",
                ids=ds.params["seed_persons"],
            )
        )

    def g_deep_traversal(self, ds):
        return frozenset(
            self._col(
                "UNWIND $ids AS sid MATCH (p:Project {gid:sid})-[:DEPENDS_ON*1..15]->(d:Project) "
                "RETURN DISTINCT d.gid AS g",
                "g",
                ids=ds.params["seed_projects"],
            )
        )

    def g_shortest_path(self, ds):
        lengths = []
        for a, b in ds.params["sp_pairs"]:
            r = self._session.run(
                "MATCH p = shortestPath((a:Person {gid:$a})-[:KNOWS*]-(b:Person {gid:$b})) RETURN length(p) AS L",
                a=a,
                b=b,
            ).single()
            lengths.append(r["L"] if r and r["L"] is not None else None)
        return tuple(lengths)

    def g_pattern_match(self, ds):
        return self._scalar(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:OWNS]->(pr:Project)"
            "<-[:CONTRIBUTES_TO]-(p) RETURN count(*) AS c"
        )

    def g_industry_aggregation(self, ds):
        rows = self._session.run("MATCH (n:Company) RETURN n.industry AS ind, count(n) AS c, avg(n.size) AS a")
        return {r["ind"]: (r["c"], r["a"]) for r in rows}

    def g_two_step_join(self, ds):
        return self._scalar("MATCH (:Person)-[:WORKS_AT]->(:Company)-[:OWNS]->(:Project) RETURN count(*) AS c")

    def g_geo_within(self, ds):
        lat0, lat1, lon0, lon1 = GEO_BBOX
        return frozenset(
            self._col(
                "MATCH (c:City) WHERE c.latitude >= $lat0 AND c.latitude <= $lat1 "
                "AND c.longitude >= $lon0 AND c.longitude <= $lon1 RETURN c.gid AS g",
                "g",
                lat0=lat0,
                lat1=lat1,
                lon0=lon0,
                lon1=lon1,
            )
        )

    def g_degree_topk(self, ds):
        return tuple(
            self._col(
                "MATCH (p:Person)-[:KNOWS]-() WITH p, count(*) AS deg "
                "RETURN p.gid AS gid, deg ORDER BY deg DESC LIMIT $k",
                "deg",
                k=ds.params["topk"],
            )
        )

    def g_connected_components(self, ds):
        raise Skip("WCC needs the Neo4j GDS plugin")

    def g_mutations(self, ds):
        off = ds.params["mut_new_base"] + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        s = self._session
        rows = [{"gid": off + i, "age": 30 + (i % 40)} for i in range(n)]
        s.run("UNWIND $rows AS r CREATE (n:Person {gid:r.gid, age:r.age})", rows=rows).consume()
        pairs = [{"a": off + i, "b": off + i - 1} for i in range(1, n)]
        s.run(
            "UNWIND $p AS x MATCH (a:Person {gid:x.a}),(b:Person {gid:x.b}) CREATE (a)-[:KNOWS]->(b)", p=pairs
        ).consume()
        s.run("MATCH (n:Person) WHERE n.gid >= $off SET n.age = 99", off=off).consume()
        dels = [off + i for i in range(0, n, 3)]
        s.run("UNWIND $ids AS i MATCH (n:Person {gid:i}) DETACH DELETE n", ids=dels).consume()
        return len(dels)
