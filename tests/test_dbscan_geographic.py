"""Geographic DBSCAN keeps valid-row mapping and fluent spatial-field semantics."""

import pandas as pd
import pytest

from kglite import CypherTimeoutError, KnowledgeGraph


def test_geographic_dbscan_preserves_original_indices_when_fields_are_missing():
    graph = KnowledgeGraph()
    graph.set_spatial("Geo", location=("lat", "lon"))
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": [10, 11, 12, 13, 14, 15, 16],
                "lat": [0.0, None, 0.0, "bad", 0.0, 1.0, 1.0],
                "lon": [0.0, 0.0, 0.00001, 0.0, 0.00002, 1.0, 1.00001],
            }
        ),
        "Geo",
        "id",
        columns=["lat", "lon"],
    )
    rows = graph.cypher(
        "MATCH (g:Geo) CALL cluster({method: 'dbscan', eps: 3.0, min_points: 2}) "
        "YIELD node, cluster RETURN node.id AS id, cluster"
    ).to_list()
    assert rows == [
        {"id": 10, "cluster": 0},
        {"id": 12, "cluster": 0},
        {"id": 14, "cluster": 0},
        {"id": 15, "cluster": -1},
        {"id": 16, "cluster": -1},
    ]
    assert all(type(v) is int for row in rows for v in row.values())


def test_fluent_geographic_dbscan_uses_configured_fields_in_reversed_feature_order():
    graph = KnowledgeGraph()
    graph.set_spatial("Geo", location=("lat", "lon"))
    graph.add_nodes(
        pd.DataFrame({"id": range(6), "lat": [89.0] * 6, "lon": [0.0, 0.001, 0.002, 10.0, 10.001, 10.002]}),
        "Geo",
        "id",
        columns=["lat", "lon"],
    )
    selected = graph.select("Geo").compare(
        "Geo", {"type": "cluster", "algorithm": "dbscan", "features": ["lon", "lat"], "eps": 5.0, "min_samples": 2}
    )
    # Each cluster keeps its first member as representative parent. Swapped
    # latitude/longitude would make all six points noise and yield five children.
    assert sorted(row["id"] for row in selected.collect()) == [1, 2, 4, 5]


def test_geographic_dbscan_deadline_returns_no_partial_result():
    graph = KnowledgeGraph()
    graph.set_spatial("Geo", location=("lat", "lon"))
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(2048),
                "lat": [-70.0 + (i // 64) * 4 for i in range(2048)],
                "lon": [-170.0 + (i % 64) * 5 for i in range(2048)],
            }
        ),
        "Geo",
        "id",
        columns=["lat", "lon"],
    )
    with pytest.raises(CypherTimeoutError):
        graph.cypher(
            "MATCH (g:Geo) CALL cluster({method: 'dbscan', eps: 3.0, min_points: 2}) "
            "YIELD node, cluster RETURN node.id AS id, cluster",
            timeout_ms=1,
        ).to_list()
    assert graph.cypher("MATCH (g:Geo) RETURN count(g) AS n").to_list() == [{"n": 2048}]
