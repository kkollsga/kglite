"""A sub-node with ``parent_fk`` gets its ``OF_<PARENT>`` edge automatically."""

import json

import pandas as pd

from kglite.blueprint import from_blueprint


def _csv(path, df):
    df.to_csv(path, index=False, encoding="utf-8")


def _build(tmp_path, nodes):
    bp = {"settings": {"root": str(tmp_path)}, "nodes": nodes}
    (tmp_path / "bp.json").write_text(json.dumps(bp), encoding="utf-8")
    return from_blueprint(tmp_path / "bp.json", save=False)


def _employees(tmp_path):
    _csv(
        tmp_path / "employees.csv",
        pd.DataFrame({"employee_id": [1, 2, 3], "name": ["Ann", "Bo", "Cy"]}),
    )
    _csv(
        tmp_path / "reviews.csv",
        pd.DataFrame(
            {
                "review_id": [10, 11, 12, 13],
                "employee_id": [1, 1, 2, 3],
                "summary": ["a", "b", "c", "d"],
                "rating": [4, 3, 5, 2],
            }
        ),
    )


def _review(**extra):
    spec = {
        "csv": "reviews.csv",
        "pk": "review_id",
        "title": "summary",
        "parent_fk": "employee_id",
        "properties": {"rating": "int"},
        "skipped": ["employee_id"],
    }
    spec.update(extra)
    return spec


def _employee(sub):
    return {"Employee": {"csv": "employees.csv", "pk": "employee_id", "title": "name", "sub_nodes": {"Review": sub}}}


def _rows(g, q):
    return [tuple(r.values()) for r in g.cypher(q)]


def test_docs_review_example_links_each_review_to_its_employee(tmp_path):
    _employees(tmp_path)
    g = _build(tmp_path, _employee(_review()))
    rows = _rows(
        g,
        "MATCH (e:Employee)<-[:OF_EMPLOYEE]-(r:Review) RETURN e.name AS n, r.rating AS x ORDER BY r.rating DESC",
    )
    assert rows == [("Bo", 5), ("Ann", 4), ("Ann", 3), ("Cy", 2)]


def test_parent_fk_alone_gives_one_edge_per_node_with_auto_pk(tmp_path):
    _employees(tmp_path)
    g = _build(tmp_path, _employee(_review(pk="auto")))
    assert _rows(g, "MATCH (r:Review)-[x:OF_EMPLOYEE]->(:Employee) RETURN count(x) AS c") == [(4,)]
    assert _rows(g, "MATCH (r:Review) RETURN count(r) AS c") == [(4,)]
    assert _rows(g, "MATCH (r:Review)-[x]->() RETURN count(x) AS c") == [(4,)]


def test_explicit_same_name_fk_edge_wins_without_duplicate(tmp_path):
    _employees(tmp_path)
    sub = _review(connections={"fk_edges": {"OF_EMPLOYEE": {"target": "Employee", "fk": "employee_id"}}})
    g = _build(tmp_path, _employee(sub))
    assert _rows(g, "MATCH (:Review)-[x:OF_EMPLOYEE]->(:Employee) RETURN count(x) AS c") == [(4,)]


def test_differently_named_explicit_edge_coexists_with_implicit(tmp_path):
    _employees(tmp_path)
    sub = _review(connections={"fk_edges": {"REVIEWS": {"target": "Employee", "fk": "employee_id"}}})
    g = _build(tmp_path, _employee(sub))
    assert _rows(g, "MATCH (:Review)-[x:OF_EMPLOYEE]->(:Employee) RETURN count(x) AS c") == [(4,)]
    assert _rows(g, "MATCH (:Review)-[x:REVIEWS]->(:Employee) RETURN count(x) AS c") == [(4,)]


def test_timeseries_sub_node_with_only_parent_fk_gets_one_edge_per_node(tmp_path):
    _csv(tmp_path / "fields.csv", pd.DataFrame({"field_id": [1, 2], "name": ["Troll", "Ekofisk"]}))
    _csv(
        tmp_path / "prod.csv",
        pd.DataFrame(
            {
                "field_id": [1, 1, 1, 2, 2, 2],
                "year": [2020] * 6,
                "month": [1, 2, 3, 1, 2, 3],
                "oil": [1.0, 1.5, 2.0, 0.5, 0.6, 0.7],
            }
        ),
    )
    sub = {
        "csv": "prod.csv",
        "pk": "field_id",
        "parent_fk": "field_id",
        "properties": {},
        "skipped": [],
        "timeseries": {
            "time_key": {"year": "year", "month": "month"},
            "resolution": "month",
            "channels": {"oil": "oil"},
        },
    }
    nodes = {
        "Field": {"csv": "fields.csv", "pk": "field_id", "title": "name", "sub_nodes": {"Production": sub}},
    }
    g = _build(tmp_path, nodes)
    assert _rows(g, "MATCH (p:Production)-[x:OF_FIELD]->(:Field) RETURN count(x) AS c") == [(2,)]
    assert _rows(g, "MATCH (p:Production) RETURN count(p) AS c") == [(2,)]
