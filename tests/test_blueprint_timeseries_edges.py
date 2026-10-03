"""Timeseries sub-nodes: one parent edge per distinct (src, tgt, props), not per CSV row."""

import json

import pandas as pd
import pytest

from kglite.blueprint import from_blueprint

TS = {
    "time_key": {"year": "yr", "month": "mo"},
    "resolution": "month",
    "channels": {"out": "val"},
    "units": {"out": "u"},
}


def _csv(path, df):
    df.to_csv(path, index=False)


def _build(tmp_path, nodes):
    bp = {"settings": {"root": str(tmp_path)}, "nodes": nodes}
    with open(tmp_path / "bp.json", "w", encoding="utf-8") as f:
        json.dump(bp, f)
    return from_blueprint(tmp_path / "bp.json", save=False)


def _readings(n_per=3):
    rows = []
    for pid, base in ((1, 1.0), (2, 10.0)):
        for m in range(1, n_per + 1):
            rows.append({"plant_id": pid, "sensor": f"S{pid}", "yr": 2020, "mo": m, "val": base * m})
    return pd.DataFrame(rows)  # plant 1 total 6.0, plant 2 total 60.0


def _plants(tmp_path):
    _csv(tmp_path / "plants.csv", pd.DataFrame({"plant_id": [1, 2], "name": ["Alpha", "Beta"]}))


def _reading_spec(**extra):
    spec = {
        "csv": "readings.csv",
        "pk": "plant_id",
        "title": "sensor",
        "parent_fk": "plant_id",
        "properties": {},
        "skipped": ["plant_id", "sensor"],
        "timeseries": TS,
        "connections": {"fk_edges": {"OF_PLANT": {"target": "Plant", "fk": "plant_id"}}},
    }
    spec.update(extra)
    return spec


def _plant_node(sub):
    return {
        "Plant": {
            "csv": "plants.csv",
            "pk": "plant_id",
            "title": "name",
            "properties": {},
            "skipped": [],
            "sub_nodes": sub,
        }
    }


def _join_sums(g, rel):
    rows = g.cypher(
        f"MATCH (pl:Plant)<-[r:{rel}]-(x) RETURN pl.title AS t, count(r) AS c, sum(ts_sum(x.out)) AS s ORDER BY t"
    ).to_list()
    return [(r["t"], r["c"], r["s"]) for r in rows]


def test_variant_a_timeseries_is_only_loader_of_edge(tmp_path):
    _plants(tmp_path)
    _csv(tmp_path / "readings.csv", _readings())
    g = _build(tmp_path, _plant_node({"Reading": _reading_spec()}))
    assert g.cypher("MATCH ()-[r:OF_PLANT]->() RETURN count(r) AS c").to_list()[0]["c"] == 2
    out = _join_sums(g, "OF_PLANT")
    assert [(t, c) for t, c, _ in out] == [("Alpha", 1), ("Beta", 1)]
    assert [s for *_, s in out] == [pytest.approx(6.0), pytest.approx(60.0)]


def test_variant_b_another_subnode_loads_the_edge_type_first(tmp_path):
    _plants(tmp_path)
    _csv(tmp_path / "readings.csv", _readings())
    _csv(tmp_path / "sites.csv", pd.DataFrame({"site_id": [10, 20], "name": ["X", "Y"], "plant_id": [1, 2]}))
    site = {
        "csv": "sites.csv",
        "pk": "site_id",
        "title": "name",
        "parent_fk": "plant_id",
        "properties": {},
        "skipped": ["plant_id"],
        "connections": {"fk_edges": {"OF_PLANT": {"target": "Plant", "fk": "plant_id"}}},
    }
    g = _build(tmp_path, _plant_node({"Site": site, "Reading": _reading_spec()}))
    assert g.cypher("MATCH (:Reading)-[r:OF_PLANT]->() RETURN count(r) AS c").to_list()[0]["c"] == 2
    assert g.cypher("MATCH (:Site)-[r:OF_PLANT]->() RETURN count(r) AS c").to_list()[0]["c"] == 2
    out = g.cypher(
        "MATCH (pl:Plant)<-[r:OF_PLANT]-(x:Reading) "
        "RETURN pl.title AS t, count(r) AS c, sum(ts_sum(x.out)) AS s ORDER BY t"
    ).to_list()
    assert [(r["t"], r["c"]) for r in out] == [("Alpha", 1), ("Beta", 1)]
    assert [r["s"] for r in out] == [pytest.approx(6.0), pytest.approx(60.0)]


def test_pk_differs_from_fk(tmp_path):
    _plants(tmp_path)
    df = _readings()
    df["rid"] = df["plant_id"].astype(str) + "-" + df["sensor"]
    _csv(tmp_path / "readings.csv", df)
    spec = _reading_spec(pk="rid", skipped=["plant_id", "sensor", "rid"])
    g = _build(tmp_path, _plant_node({"Reading": spec}))
    assert g.cypher("MATCH (r:Reading) RETURN count(r) AS c").to_list()[0]["c"] == 2
    out = _join_sums(g, "OF_PLANT")
    assert [(t, c) for t, c, _ in out] == [("Alpha", 1), ("Beta", 1)]
    assert [s for *_, s in out] == [pytest.approx(6.0), pytest.approx(60.0)]


def test_top_level_spec_with_parent_gets_one_implicit_edge(tmp_path):
    _plants(tmp_path)
    _csv(tmp_path / "readings.csv", _readings())
    spec = _reading_spec(parent="Plant")
    spec.pop("connections")
    nodes = _plant_node({})
    nodes["Reading"] = spec
    g = _build(tmp_path, nodes)
    assert g.cypher("MATCH (:Reading)-[r:OF_PLANT]->(:Plant) RETURN count(r) AS c").to_list()[0]["c"] == 2
    out = _join_sums(g, "OF_PLANT")
    assert [(t, c) for t, c, _ in out] == [("Alpha", 1), ("Beta", 1)]
    assert [s for *_, s in out] == [pytest.approx(6.0), pytest.approx(60.0)]


def test_changing_fk_keeps_both_targets(tmp_path):
    _plants(tmp_path)
    df = pd.DataFrame(
        {
            "rid": ["R"] * 4,
            "plant_id": [1, 1, 2, 2],
            "yr": 2020,
            "mo": [1, 2, 3, 4],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )
    _csv(tmp_path / "readings.csv", df)
    spec = _reading_spec(pk="rid", title="rid", skipped=["plant_id", "rid"])
    g = _build(tmp_path, _plant_node({"Reading": spec}))
    assert g.cypher("MATCH (:Reading)-[r:OF_PLANT]->() RETURN count(r) AS c").to_list()[0]["c"] == 2
    rows = g.cypher("MATCH (:Reading)-[:OF_PLANT]->(pl:Plant) RETURN pl.title AS t ORDER BY t").to_list()
    assert [r["t"] for r in rows] == ["Alpha", "Beta"]


def test_differing_edge_property_keeps_both_edges(tmp_path):
    _plants(tmp_path)
    df = pd.DataFrame(
        {
            "rid": ["R"] * 4,
            "plant_id": [1] * 4,
            "role": ["a", "a", "b", "b"],
            "yr": 2020,
            "mo": [1, 2, 3, 4],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )
    _csv(tmp_path / "readings.csv", df)
    spec = _reading_spec(pk="rid", title="rid", skipped=["plant_id", "rid", "role"])
    spec["connections"]["fk_edges"]["OF_PLANT"]["properties"] = ["role"]
    g = _build(tmp_path, _plant_node({"Reading": spec}))
    rows = g.cypher("MATCH (:Reading)-[r:OF_PLANT]->() RETURN r.role AS role ORDER BY role").to_list()
    assert [r["role"] for r in rows] == ["a", "b"]
