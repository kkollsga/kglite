"""Kùzu adapter — embedded, columnar, Cypher-speaking graph database.

The closest "embedded graph DB" peer to kglite. Same logical Cypher
workloads as the kglite Cypher adapter (translated to Kùzu's `gid`
property + `* SHORTEST` path syntax). Bulk-loaded from CSV (no
pyarrow/parquet dependency by design — this repo deliberately avoids
pyarrow). Weakly-connected-components needs the optional algo extension
and is skipped here, matching kglite's no-native-WCC stance.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import kuzu
import pandas as pd

from .base import Adapter, Skip
from .dataset import Dataset

# node type -> (PRIMARY KEY-first ordered (column, kuzu_type) list)
NODE_SCHEMA = {
    "Person": [
        ("gid", "INT64"),
        ("name", "STRING"),
        ("age", "INT64"),
        ("city", "STRING"),
        ("joined_year", "INT64"),
        ("active", "INT64"),
        ("score", "DOUBLE"),
    ],
    "Company": [("gid", "INT64"), ("name", "STRING"), ("industry", "STRING"), ("size", "INT64")],
    "Project": [("gid", "INT64"), ("name", "STRING"), ("budget", "DOUBLE"), ("status", "STRING")],
    "Skill": [("gid", "INT64"), ("name", "STRING"), ("category", "STRING")],
    "City": [("gid", "INT64"), ("name", "STRING"), ("population", "INT64"), ("region", "STRING")],
}
# rel type -> (FROM node type, TO node type)
REL_SCHEMA = {
    "KNOWS": ("Person", "Person"),
    "WORKS_AT": ("Person", "Company"),
    "CONTRIBUTES_TO": ("Person", "Project"),
    "HAS_SKILL": ("Person", "Skill"),
    "OWNS": ("Company", "Project"),
    "DEPENDS_ON": ("Project", "Project"),
    "LOCATED_IN": ("Company", "City"),
}


class KuzuAdapter(Adapter):
    name = "kuzu"

    def version(self) -> str:
        return kuzu.__version__

    def build(self, ds: Dataset) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="graphsuite_kuzu_")
        csvdir = Path(self._tmpdir) / "csv"
        csvdir.mkdir()
        db = kuzu.Database(str(Path(self._tmpdir) / "kz"))
        con = kuzu.Connection(db)

        for ntype, cols in NODE_SCHEMA.items():
            df = ds.node_frame(ntype).copy()
            if "active" in df.columns:
                df["active"] = df["active"].astype(int)
            colnames = [c for c, _ in cols]
            csv = csvdir / f"{ntype}.csv"
            df[colnames].to_csv(csv, index=False)
            decl = ", ".join(f"{c} {t}" for c, t in cols)
            con.execute(f"CREATE NODE TABLE {ntype}({decl}, PRIMARY KEY(gid))")
            con.execute(f"COPY {ntype} FROM '{csv}' (HEADER=true)")

        for etype, (ft, tt) in REL_SCHEMA.items():
            df = pd.DataFrame(ds.edges[etype])[["src", "dst"]]
            csv = csvdir / f"{etype}.csv"
            df.to_csv(csv, index=False)
            con.execute(f"CREATE REL TABLE {etype}(FROM {ft} TO {tt})")
            con.execute(f"COPY {etype} FROM '{csv}' (HEADER=true)")

        self._db = db
        self.con = con
        self._mut = 0

    def teardown(self) -> None:
        try:
            self.con = None
            self._db = None
        finally:
            shutil.rmtree(getattr(self, "_tmpdir", ""), ignore_errors=True)

    def _scalar(self, q, params=None):
        return self.con.execute(q, params).get_as_df().iloc[0, 0]

    def _rows(self, q, params=None):
        return self.con.execute(q, params).get_as_df().shape[0]

    def _df(self, q, params=None):
        return self.con.execute(q, params).get_as_df()

    def g_node_scan(self, ds):
        c = self._scalar("MATCH (n:Person) RETURN count(n)")
        ids = self._rows("MATCH (n:Person) RETURN n.gid")
        assert ids == c
        return int(ids)

    def g_point_lookup(self, ds):
        return int(
            self._scalar(
                "MATCH (n:Person) WHERE n.gid IN $ids RETURN count(n)",
                {"ids": ds.params["lookup_ids"]},
            )
        )

    def g_property_filter(self, ds):
        df = self._df(
            "MATCH (n:Person) WHERE n.age > $age AND n.city = $city RETURN n.gid AS gid",
            {"age": ds.params["filter_age"], "city": ds.params["filter_city"]},
        )
        return frozenset(int(x) for x in df["gid"])

    def g_group_aggregation(self, ds):
        df = self._df("MATCH (n:Person) RETURN n.city AS city, count(*) AS c, avg(n.age) AS a")
        return {row.city: (int(row.c), float(row.a)) for row in df.itertuples()}

    def _anchored(self, label_pat, ids):
        df = self._df(
            f"UNWIND $ids AS sid MATCH (p:Person {{gid:sid}})-{label_pat}-(f:Person) RETURN DISTINCT f.gid AS gid",
            {"ids": ids},
        )
        return frozenset(int(x) for x in df["gid"])

    def g_one_hop(self, ds):
        return self._anchored("[:KNOWS]", ds.params["seed_persons"])

    def g_two_hop(self, ds):
        return self._anchored("[:KNOWS*1..2]", ds.params["seed_persons_small"])

    def g_three_hop(self, ds):
        return self._anchored("[:KNOWS*1..3]", ds.params["seed_persons_tiny"])

    def g_filtered_traversal(self, ds):
        df = self._df(
            "UNWIND $ids AS sid MATCH (p:Person {gid:sid})-[:KNOWS]-(f:Person) "
            "WHERE f.age < 30 RETURN DISTINCT f.gid AS gid",
            {"ids": ds.params["seed_persons"]},
        )
        return frozenset(int(x) for x in df["gid"])

    def g_deep_traversal(self, ds):
        df = self._df(
            "UNWIND $ids AS sid MATCH (p:Project {gid:sid})-[:DEPENDS_ON*1..15]->(d:Project) "
            "RETURN DISTINCT d.gid AS gid",
            {"ids": ds.params["seed_projects"]},
        )
        return frozenset(int(x) for x in df["gid"])

    def g_shortest_path(self, ds):
        lengths = []
        for a, b in ds.params["sp_pairs"]:
            df = self._df(
                "MATCH p = (a:Person {gid:$a})-[:KNOWS* SHORTEST 1..30]-(b:Person {gid:$b}) RETURN length(p) AS L",
                {"a": a, "b": b},
            )
            lengths.append(int(df.iloc[0, 0]) if len(df) else None)
        return tuple(lengths)

    def g_pattern_match(self, ds):
        return int(
            self._scalar(
                "MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:OWNS]->(pr:Project)<-[:CONTRIBUTES_TO]-(p) RETURN count(*)"
            )
        )

    def g_degree_topk(self, ds):
        df = self._df(
            "MATCH (p:Person)-[:KNOWS]-(x) RETURN p.gid AS gid, count(*) AS deg ORDER BY deg DESC LIMIT $k",
            {"k": ds.params["topk"]},
        )
        return tuple(int(x) for x in df["deg"])

    def g_connected_components(self, ds):
        raise Skip("WCC needs the kuzu algo extension; not enabled")

    def g_mutations(self, ds):
        off = ds.params["mut_new_base"] + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        rows = [{"gid": off + i, "age": 30 + (i % 40)} for i in range(n)]
        self.con.execute("UNWIND $rows AS r CREATE (n:Person {gid:r.gid, age:r.age})", {"rows": rows})
        pairs = [{"a": off + i, "b": off + i - 1} for i in range(1, n)]
        self.con.execute(
            "UNWIND $pairs AS p MATCH (a:Person {gid:p.a}), (b:Person {gid:p.b}) CREATE (a)-[:KNOWS]->(b)",
            {"pairs": pairs},
        )
        self.con.execute("MATCH (n:Person) WHERE n.gid >= $off SET n.age = 99", {"off": off})
        dels = [off + i for i in range(0, n, 3)]
        self.con.execute("UNWIND $ids AS i MATCH (n:Person {gid:i}) DETACH DELETE n", {"ids": dels})
        return len(dels)
