"""Regression coverage for bound-anchor count fusion."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture(params=["default", "mapped", "disk"])
def graph(request, tmp_path):
    options = {"storage": request.param}
    if request.param == "disk":
        options["path"] = str(tmp_path / "graph")
    g = KnowledgeGraph(**options)
    g.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}), "A", "id", "name")
    g.add_nodes(pd.DataFrame({"id": [4], "name": ["d"]}), "B", "id", "name")
    g.add_connections(pd.DataFrame({"s": [1, 1], "t": [2, 3]}), "R", "A", "s", "A", "t")
    g.add_connections(pd.DataFrame({"s": [1], "t": [4]}), "S", "A", "s", "B", "t")
    return g


@pytest.mark.parametrize("constraint", ["{id:999}", "{name:'missing'}", ":B"])
@pytest.mark.parametrize(
    "tail",
    [
        "RETURN a.id AS id, count(r) AS c",
        "WITH a, count(r) AS c RETURN a.id AS id, c",
    ],
)
def test_optional_count_honors_constraints_on_bound_anchor(graph, constraint, tail):
    query = f"MATCH (a:A {{id:1}}) OPTIONAL MATCH (a {constraint})-[r:R]->(b) {tail}"
    assert graph.cypher(query, disable_optimizer=True).to_list() == [{"id": 1, "c": 0}]
    assert graph.cypher(query).to_list() == [{"id": 1, "c": 0}]


@pytest.mark.parametrize("constraint", ["{id:999}", "{name:'missing'}", ":B"])
def test_two_match_count_honors_constraints_on_shared_anchor(graph, constraint):
    query = f"MATCH (a:A)-[:S]->(b:B) MATCH (a {constraint})-[r:R]->() WITH a, count(r) AS c RETURN a.id AS id, c"
    assert graph.cypher(query, disable_optimizer=True).to_list() == []
    assert graph.cypher(query).to_list() == []
