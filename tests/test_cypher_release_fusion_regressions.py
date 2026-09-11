"""Release-review regressions for the count and retrieval fusion boundaries."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture(params=["default", "mapped", "disk"])
def graph(request, tmp_path):
    options = {"storage": request.param}
    if request.param == "disk":
        options["path"] = str(tmp_path / "graph")
    g = KnowledgeGraph(**options)
    g.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "wanted": [2, 3, 1]}), "A", "id", "name")
    g.add_nodes(pd.DataFrame({"id": [4], "name": ["d"]}), "B", "id", "name")
    g.add_connections(pd.DataFrame({"s": [1, 1], "t": [2, 3]}), "R", "A", "s", "A", "t")
    g.add_connections(pd.DataFrame({"s": [1, 2], "t": [4, 4]}), "S", "A", "s", "B", "t")
    return g


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [("(a:A)-[:S]-(b)", 2), ("(a:A)<-[:S]-(b)", 0), ("(a:B)<-[:S]-(b:A)", 2)],
)
def test_typed_count_respects_endpoint_orientation(graph, pattern, expected):
    query = f"MATCH {pattern} RETURN count(*) AS c"
    assert graph.cypher(query, disable_optimizer=True).scalar() == expected
    assert graph.cypher(query).scalar() == expected


@pytest.mark.parametrize("disable_optimizer", [False, True])
@pytest.mark.parametrize(
    ("prefix", "pattern", "expected"),
    [
        ("MATCH (a:A {id:1})", "(a {id:999})-[:R]->()", [0]),
        ("MATCH (a:A {id:1})", "(a:B)-[:R]->()", [0]),
        ("MATCH (a:A {id:1}) WITH a, 2 AS k", "(a)-[:R]->({id:k})", [1]),
        ("MATCH (a:A {id:1})", "(a)-[:R]->({id:a.wanted})", [1]),
        ("MATCH (a:A)-[r:R]->(b)", "(a)-[r:R]->()", [1, 1]),
        (
            "MATCH (a:A {id:1}) OPTIONAL MATCH (a)-[r:R]->(:A {id:999}) WITH a,r",
            "(a)-[r:R]->()",
            [0],
        ),
        ("MATCH (a:A {id:1}) OPTIONAL MATCH (b:A {id:999}) WITH a,b", "(a)-[:R]->(b)", [0]),
        (
            "MATCH (a:A {id:1}) MATCH (b:A {id:2}) WITH a, collect(b) AS bs UNWIND bs AS b",
            "(a)-[:R]->(b)",
            [1],
        ),
    ],
)
def test_count_incident_scan_preserves_correlated_constraints(graph, prefix, pattern, expected, disable_optimizer):
    query = f"{prefix} RETURN COUNT {{ {pattern} }} AS c"
    assert [r["c"] for r in graph.cypher(query, disable_optimizer=disable_optimizer)] == expected


def test_with_rename_preserves_count_correlation(graph):
    query = "MATCH (a:A) WITH a AS x RETURN x.id AS id, COUNT { (x)-[:R]->() } AS c ORDER BY id"
    expected = [{"id": 1, "c": 2}, {"id": 2, "c": 0}, {"id": 3, "c": 0}]
    assert graph.cypher(query, disable_optimizer=True).to_list() == expected
    assert graph.cypher(query).to_list() == expected


@pytest.mark.parametrize("predicate", ["EXISTS { (a:B)-[:R]-() }", "EXISTS { (a {id:999})-[:R]-() }"])
def test_exists_incident_scan_preserves_anchor_constraints(graph, predicate):
    assert graph.cypher(f"MATCH (a:A {{id:1}}) RETURN {predicate} AS e").scalar() is False


def test_exists_incident_scan_preserves_null_relationship(graph):
    query = "MATCH (a:A {id:1}) OPTIONAL MATCH (a)-[r:R]->(:A {id:999}) WITH a,r RETURN EXISTS { (a)-[r:R]->() } AS e"
    assert graph.cypher(query).scalar() is False


@pytest.mark.parametrize("hop", ["(a)-[r:R]->()", "(a)-[r:R]-()", "()<-[r:R]-(a)"])
def test_exists_where_can_read_the_local_relationship(graph, hop):
    query = f"MATCH (a:A {{id:1}}) RETURN EXISTS {{ {hop} WHERE type(r) = 'R' }} AS e"
    assert graph.cypher(query).scalar() is True


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("alias", ["s", 'vector_score(d, Literal(String("text_emb")), Parameter("ordered"))'])
def test_retrieval_orders_by_its_own_query_vector(indexed, alias):
    g = KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2], "text": ["a", "b"]}), "Doc", "id", "text", ["text"])
    g.set_embeddings("Doc", "text", {1: [1.0, 0.0], 2: [0.0, 1.0]})
    if indexed:
        g.build_vector_index("Doc", "text")
    query = (
        f"MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'text_emb', $projected) AS `{alias}` "
        "ORDER BY vector_score(d, 'text_emb', $ordered) DESC LIMIT 1"
    )
    params = {"projected": [1.0, 0.0], "ordered": [0.0, 1.0]}
    expected = [{"id": 2, alias: 0.0}]
    assert g.cypher(query, params=params, disable_optimizer=True).to_list() == expected
    assert g.cypher(query, params=params).to_list() == expected


def test_bm25_orders_by_its_own_query_text():
    g = KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2], "text": ["alpha", "beta"]}), "Doc", "id", "text", ["text"])
    g.build_text_index("Doc", "text")
    query = (
        "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'text', 'alpha') AS s "
        "ORDER BY text_bm25(d, 'text', 'beta') DESC LIMIT 1"
    )
    expected = [{"id": 2, "s": 0.0}]
    assert g.cypher(query, disable_optimizer=True).to_list() == expected
    assert g.cypher(query).to_list() == expected
