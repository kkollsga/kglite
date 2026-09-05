"""Exact DBSCAN labels and input order, without canonical relabeling."""

import pytest

from kglite import KnowledgeGraph


@pytest.mark.parametrize("reverse", [False, True])
def test_dbscan_shared_border_follows_first_seed(reverse):
    values = [0.0, 1.0, 1.1, 1.2, 1.3, -1.0, -1.1, -1.2, -1.3]
    order = [0, 5, 6, 7, 8, 1, 2, 3, 4] if reverse else list(range(9))
    graph = KnowledgeGraph()
    graph.cypher(
        "UNWIND $points AS p CREATE (:Point {id: p.id, f: p.f})",
        params={"points": [{"id": i, "f": values[i]} for i in order]},
    )
    rows = graph.cypher(
        "MATCH (p:Point) CALL cluster({method: 'dbscan', properties: ['f'], "
        "eps: 1.0, min_points: 3, normalize: false}) "
        "YIELD node, cluster RETURN node.id AS id, cluster"
    ).to_list()
    assert rows == [{"id": i, "cluster": 0 if position < 5 else 1} for position, i in enumerate(order)]
    assert all(type(row["id"]) is int and type(row["cluster"]) is int for row in rows)


def test_dbscan_core_chain_expands_beyond_seed():
    graph = KnowledgeGraph()
    values = [0.0, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 5.0]
    graph.cypher(
        "UNWIND $points AS p CREATE (:Point {id: p.id, f: p.f})",
        params={"points": [{"id": i, "f": value} for i, value in enumerate(values)]},
    )
    rows = graph.cypher(
        "MATCH (p:Point) CALL cluster({method: 'dbscan', properties: ['f'], "
        "eps: 0.31, min_points: 2, normalize: false}) "
        "YIELD node, cluster RETURN node.id AS id, cluster"
    ).to_list()
    assert rows == [{"id": i, "cluster": 0 if i < 8 else -1} for i in range(9)]


def test_fluent_dbscan_keeps_representative_and_children_order():
    graph = KnowledgeGraph()
    graph.cypher("UNWIND range(0, 7) AS i CREATE (:Point {id: i, f: 0.0})")
    selected = graph.select("Point").compare(
        "Point", {"type": "cluster", "algorithm": "dbscan", "features": ["f"], "eps": 0.01, "min_samples": 2}
    )
    # The fluent hierarchy uses the first member as parent and returns children.
    assert [row["id"] for row in selected.collect()] == list(range(1, 8))
