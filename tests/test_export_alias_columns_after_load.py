"""A reloaded graph exports the same columns as the graph that was saved.

Past 50 single-typed nodes, `to_df()` and `collect()` take their property
columns from the type's schema rather than scanning every node. A schema
rebuilt on load comes from the type's metadata, which records the loader's id
and title column names (`add_nodes(df, 'T', 'code', 'name')`); no node stores
them — they read back as `id` / `title` — so a reloaded graph gained all-None
`code` / `name` columns while Cypher read the values.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

ROWS = [2, 51, 100]


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"code": [f"c{i}" for i in range(n)], "name": [f"N{i}" for i in range(n)], "x": range(n)})


def _reloaded(storage, n: int, tmp_path) -> tuple[kglite.KnowledgeGraph, kglite.KnowledgeGraph]:
    if storage == "disk":
        path = str(tmp_path / "g")
        g = kglite.KnowledgeGraph(storage="disk", path=path)
        g.add_nodes(_frame(n), "T", "code", "name")
        g.save()
        return g, kglite.load(path)
    g = kglite.KnowledgeGraph(storage=storage) if storage else kglite.KnowledgeGraph()
    g.add_nodes(_frame(n), "T", "code", "name")
    file = str(tmp_path / "g.kgl")
    g.save(file)
    return g, kglite.load(file)


@pytest.mark.parametrize("storage", [None, "mapped", "disk"], ids=["memory", "mapped", "disk"])
@pytest.mark.parametrize("n", ROWS)
def test_the_alias_columns_stay_out_after_a_reload(storage, n, tmp_path) -> None:
    fresh, reloaded = _reloaded(storage, n, tmp_path)
    for graph in (fresh, reloaded):
        frame = graph.select("T").to_df()
        assert list(frame.columns) == ["type", "title", "id", "x"]
        assert frame.iloc[0].to_dict() == {"type": "T", "title": "N0", "id": "c0", "x": 0}
        assert graph.select("T").to_df(include_id=False).columns.tolist() == ["type", "title", "x"]
        assert graph.select("T").collect()[0] == {"type": "T", "title": "N0", "id": "c0", "x": 0}
        # The alias spellings still read the values.
        assert graph.cypher("MATCH (t:T {code: 'c0'}) RETURN t.code AS c, t.name AS n").to_list() == [
            {"c": "c0", "n": "N0"}
        ]


def test_a_traversal_target_exports_no_alias_column(tmp_path) -> None:
    g = kglite.KnowledgeGraph()
    g.add_nodes(_frame(60), "T", "code", "name")
    # Sixty distinct targets: past the schema fast path's threshold.
    g.add_nodes(
        pd.DataFrame({"pid": [f"p{i}" for i in range(60)], "pname": [f"P{i}" for i in range(60)]}), "P", "pid", "pname"
    )
    g.add_relationships(
        pd.DataFrame({"s": [f"c{i}" for i in range(60)], "t": [f"p{i}" for i in range(60)]}), "IN", "T", "s", "P", "t"
    )
    file = str(tmp_path / "g.kgl")
    g.save(file)
    h = kglite.load(file)
    targets = h.select("T").traverse("IN").to_df()
    assert len(targets) == 60
    assert list(targets.columns) == ["type", "title", "id"]
