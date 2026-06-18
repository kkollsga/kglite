"""DuckDB adapter — the relational (SQL) baseline.

Nodes and edges live in columnar tables. SQL is excellent at scans,
property filters, group-by aggregation and the multi-edge join
(`pattern_match`), and expresses k-hop traversal and the DEPENDS_ON
closure as recursive CTEs. Pure-SQL single-pair shortest path and
weakly-connected-components are iterative/impractical, so those two
groups are skipped (honest for a relational engine).

KNOWS is materialised undirected in table `kn(a,b)` (both directions)
so neighbourhood/degree queries match the graph backends' semantics.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from .base import Adapter, Skip
from .dataset import DEGREE_MIN, GEO_BBOX, SCORE_MIN, SCORE_RANGE, Dataset

# every edge type is loaded (full-dataset build, fair vs the graph stores);
# only a subset is queried by the groups.
EDGE_TABLES = {
    "knows": "KNOWS",
    "works_at": "WORKS_AT",
    "owns": "OWNS",
    "contributes": "CONTRIBUTES_TO",
    "depends": "DEPENDS_ON",
    "has_skill": "HAS_SKILL",
    "located_in": "LOCATED_IN",
}
NODE_TABLES = ("Person", "Company", "Project", "Skill", "City")


class DuckDBAdapter(Adapter):
    name = "duckdb"

    def version(self) -> str:
        return duckdb.__version__

    def build(self, ds: Dataset) -> None:
        con = duckdb.connect(":memory:")
        # node tables (full dataset)
        for ntype in NODE_TABLES:
            df = ds.node_frame(ntype)  # noqa: F841 (referenced by name in SQL)
            con.execute(f"CREATE TABLE {ntype.lower()} AS SELECT * FROM df")
        # edge tables
        for tbl, etype in EDGE_TABLES.items():
            df = pd.DataFrame(ds.edges[etype])[["src", "dst"]]  # noqa: F841
            con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM df")
        # undirected KNOWS view as a materialised table
        con.execute(
            "CREATE TABLE kn AS SELECT src AS a, dst AS b FROM knows UNION ALL SELECT dst AS a, src AS b FROM knows"
        )
        con.execute("CREATE INDEX kn_a ON kn(a)")
        con.execute("CREATE INDEX person_pk ON person(gid)")
        con.execute("CREATE INDEX dep_src ON depends(src)")
        self.con = con
        self._mut = 0

    def teardown(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    def _seeds(self, ids):
        df = pd.DataFrame({"id": ids})  # noqa: F841
        self.con.execute("CREATE OR REPLACE TEMP TABLE seeds AS SELECT * FROM df")

    def g_node_scan(self, ds):
        c = self.con.execute("SELECT count(*) FROM person").fetchone()[0]
        rows = self.con.execute("SELECT gid FROM person").fetchall()
        assert len(rows) == c
        return len(rows)

    def g_point_lookup(self, ds):
        self._seeds(ds.params["lookup_ids"])
        return self.con.execute("SELECT count(*) FROM person WHERE gid IN (SELECT id FROM seeds)").fetchone()[0]

    def g_property_filter(self, ds):
        rows = self.con.execute(
            "SELECT gid FROM person WHERE age > ? AND city = ?",
            [ds.params["filter_age"], ds.params["filter_city"]],
        ).fetchall()
        return frozenset(r[0] for r in rows)

    def g_group_aggregation(self, ds):
        rows = self.con.execute("SELECT city, count(*) AS c, avg(age) AS a FROM person GROUP BY city").fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def g_edge_scan(self, ds):
        return self.con.execute("SELECT count(*) FROM knows").fetchone()[0]

    def g_range_filter(self, ds):
        lo, hi = SCORE_RANGE
        rows = self.con.execute("SELECT gid FROM person WHERE score >= ? AND score <= ?", [lo, hi]).fetchall()
        return frozenset(r[0] for r in rows)

    def g_year_aggregation(self, ds):
        rows = self.con.execute(
            "SELECT joined_year, count(*) AS c, avg(score) AS a FROM person GROUP BY joined_year"
        ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def g_score_filtered_traversal(self, ds):
        self._seeds(ds.params["seed_persons"])
        rows = self.con.execute(
            "SELECT DISTINCT kn.b FROM kn JOIN person p ON kn.b = p.gid "
            "WHERE kn.a IN (SELECT id FROM seeds) AND p.score > ?",
            [SCORE_MIN],
        ).fetchall()
        return frozenset(r[0] for r in rows)

    def g_degree_filter(self, ds):
        return self.con.execute(
            "SELECT count(*) FROM (SELECT a FROM kn GROUP BY a HAVING count(*) >= ?)",
            [DEGREE_MIN],
        ).fetchone()[0]

    def g_bulk_update(self, ds):
        self._seeds(ds.params["lookup_ids"])
        self.con.execute("UPDATE person SET active = true WHERE gid IN (SELECT id FROM seeds)")
        return self.con.execute("SELECT count(*) FROM person WHERE gid IN (SELECT id FROM seeds)").fetchone()[0]

    def _khop(self, seeds, k):
        self._seeds(seeds)
        # recursive BFS to depth k; every row is at distance >=1 from the seed
        # set, so DISTINCT node matches the graph backends' k-hop union.
        rows = self.con.execute(
            f"""
            WITH RECURSIVE reach(node, depth) AS (
                SELECT b, 1 FROM kn WHERE a IN (SELECT id FROM seeds)
                UNION
                SELECT kn.b, reach.depth + 1 FROM reach JOIN kn ON kn.a = reach.node
                WHERE reach.depth < {k}
            )
            SELECT DISTINCT node FROM reach
            """
        ).fetchall()
        return frozenset(r[0] for r in rows)

    def g_one_hop(self, ds):
        return self._khop(ds.params["seed_persons"], 1)

    def g_two_hop(self, ds):
        return self._khop(ds.params["seed_persons_small"], 2)

    def g_three_hop(self, ds):
        return self._khop(ds.params["seed_persons_tiny"], 3)

    def g_filtered_traversal(self, ds):
        self._seeds(ds.params["seed_persons"])
        rows = self.con.execute(
            "SELECT DISTINCT kn.b FROM kn JOIN person p ON kn.b = p.gid "
            "WHERE kn.a IN (SELECT id FROM seeds) AND p.age < 30"
        ).fetchall()
        return frozenset(r[0] for r in rows)

    def g_deep_traversal(self, ds):
        self._seeds(ds.params["seed_projects"])
        rows = self.con.execute(
            """
            WITH RECURSIVE reach(node) AS (
                SELECT dst FROM depends WHERE src IN (SELECT id FROM seeds)
                UNION
                SELECT d.dst FROM reach r JOIN depends d ON d.src = r.node
            )
            SELECT DISTINCT node FROM reach
            """
        ).fetchall()
        return frozenset(r[0] for r in rows)

    def g_pattern_match(self, ds):
        return self.con.execute(
            "SELECT count(*) FROM works_at w "
            "JOIN owns o ON w.dst = o.src "
            "JOIN contributes c ON c.src = w.src AND c.dst = o.dst"
        ).fetchone()[0]

    def g_industry_aggregation(self, ds):
        rows = self.con.execute(
            "SELECT industry, count(*) AS c, avg(size) AS a FROM company GROUP BY industry"
        ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def g_two_step_join(self, ds):
        return self.con.execute("SELECT count(*) FROM works_at w JOIN owns o ON w.dst = o.src").fetchone()[0]

    def g_geo_within(self, ds):
        lat0, lat1, lon0, lon1 = GEO_BBOX
        rows = self.con.execute(
            "SELECT gid FROM city WHERE latitude >= ? AND latitude <= ? AND longitude >= ? AND longitude <= ?",
            [lat0, lat1, lon0, lon1],
        ).fetchall()
        return frozenset(r[0] for r in rows)

    def g_degree_topk(self, ds):
        rows = self.con.execute(
            "SELECT a, count(*) AS deg FROM kn GROUP BY a ORDER BY deg DESC LIMIT ?",
            [ds.params["topk"]],
        ).fetchall()
        return tuple(r[1] for r in rows)

    def g_shortest_path(self, ds):
        raise Skip("single-pair shortest path is impractical in pure SQL")

    def g_connected_components(self, ds):
        raise Skip("weakly-connected-components is impractical in pure SQL")

    def g_mutations(self, ds):
        off = ds.params["mut_new_base"] + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        con = self.con
        new = pd.DataFrame(  # noqa: F841
            {"gid": [off + i for i in range(n)], "age": [30 + (i % 40) for i in range(n)]}
        )
        con.execute("INSERT INTO person (gid, age) SELECT gid, age FROM new")
        edf = pd.DataFrame(  # noqa: F841
            {"a": [off + i for i in range(1, n)], "b": [off + i - 1 for i in range(1, n)]}
        )
        con.execute("INSERT INTO kn SELECT a, b FROM edf")
        con.execute(f"UPDATE person SET age = 99 WHERE gid >= {off}")
        con.execute(f"DELETE FROM person WHERE gid >= {off} AND (gid - {off}) % 3 = 0")
        return n
