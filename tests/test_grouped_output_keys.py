"""Presentation labels cannot discard distinct parents, nodes or edges."""

import pytest

import kglite


@pytest.fixture
def grouped_graph():
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:P {id:1,title:'A'}),(b:P {id:2,title:'A'}),(c:P {id:3,title:'A_2'}),"
        "(x:C {id:11,title:'X',v:11}),(y:C {id:12,title:'Y',v:12}),(z:C {id:13,title:'Z',v:13}) "
        "CREATE (a)-[:R]->(x),(b)-[:R]->(y),(c)-[:R]->(z)"
    )
    return graph


@pytest.mark.parametrize("parent_info", [False, True])
def test_collect_grouped_reserves_real_suffixes(grouped_graph, parent_info):
    view = grouped_graph.select("P").traverse("R")
    result = view.collect_grouped("P", parent_info=parent_info, flatten_single_parent=False)
    groups = [row["children"] if parent_info else row for row in result.values()]
    assert sorted(child["id"] for group in groups for child in group) == [11, 12, 13]
    assert set(result) == {"A", "A_2", "A_3"}
    if parent_info:
        assert result["A_2"]["id"] == 3
    assert result == view.collect_grouped("P", parent_info=parent_info, flatten_single_parent=False)


@pytest.mark.parametrize("reader", ["titles", "properties", "unique", "connections", "count", "calculate"])
def test_grouped_readers_keep_all_parent_groups(grouped_graph, reader):
    view = grouped_graph.select("P").traverse("R")
    read = {
        "titles": lambda: view.titles(flatten_single_parent=False),
        "properties": lambda: view.get_properties(["id"], flatten_single_parent=False),
        "unique": lambda: view.unique_values("id"),
        "connections": lambda: view.connections(parent_info=True, flatten_single_parent=False),
        "count": lambda: view.count(group_by_parent=True),
        "calculate": lambda: view.calculate("sum(v)"),
    }[reader]
    result = read()
    assert set(result) == {"A", "A_2", "A_3"}
    if reader == "titles":
        assert sorted(value for values in result.values() for value in values) == ["X", "Y", "Z"]
    elif reader == "properties":
        assert sorted(value[0] for values in result.values() for value in values) == [11, 12, 13]
    elif reader == "unique":
        assert sorted(value for values in result.values() for value in values) == [11, 12, 13]
    elif reader == "connections":
        assert sorted(value["parent_id"] for value in result.values()) == [1, 2, 3]
    elif reader == "count":
        assert sorted(result.values()) == [1, 1, 1]
    else:
        assert sorted(result.values()) == [11.0, 12.0, 13.0]
    assert result == read()


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("flatten", [False, True])
def test_connections_keep_duplicate_endpoint_titles(direction, flatten):
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:P {id:1,title:'A'}),(b:P {id:2,title:'A'}),(c:P {id:3,title:'A_2'}),"
        "(x:C {id:11,title:'X'}) CREATE (x)-[:R {rank:1}]->(a),"
        "(x)-[:R {rank:2}]->(b),(x)-[:R {rank:3}]->(c)"
    )
    if direction == "incoming":
        graph.cypher("MATCH (x:C)-[r:R]->(p:P) CREATE (p)-[:BACK {rank:r.rank}]->(x)")
    result = graph.select("C").connections(flatten_single_parent=flatten)
    if not flatten:
        result = next(iter(result.values()))["connections"]
    edges = result["X"][direction]["BACK" if direction == "incoming" else "R"]
    assert set(edges) == {"A", "A_2", "A_3"}
    assert sorted((edge["node_id"], edge["connection_properties"]["rank"]) for edge in edges.values()) == [
        (1, 1),
        (2, 2),
        (3, 3),
    ]


@pytest.mark.parametrize("parent_info", [False, True])
def test_connections_selected_nodes_and_reserved_metadata(parent_info):
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:P {id:1,title:'A'}),(b:P {id:2,title:'A'}),(c:P {id:3,title:'A_2'}),"
        "(d:P {id:4,title:'parent_title'}),(x:C {id:11,title:'X'}) "
        "CREATE (a)-[:R]->(x),(b)-[:R]->(x),(c)-[:R]->(x),(d)-[:R]->(x)"
    )
    result = graph.select("P").connections(parent_info=parent_info)
    nodes = [value for value in result.values() if isinstance(value, dict)]
    assert sorted(node["node_id"] for node in nodes) == [1, 2, 3, 4]
    if parent_info:
        assert isinstance(result["parent_title"], str)


def test_grouped_single_parent_shape_is_unchanged(grouped_graph):
    view = grouped_graph.select("P").where({"id": 1}).traverse("R")
    assert view.titles() == ["X"]
    assert view.get_properties(["id"]) == [(11,)]
    assert view.collect_grouped("P")[0]["id"] == 11
    assert view.collect_grouped("P", parent_info=True)["id"] == 1


def test_connections_typed_ids_and_parallel_edges_remain_separate():
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:P {id:1,title:'A'}),(b:Q {id:1,title:'A'}),(x:C {id:11,title:'X'}) "
        "CREATE (x)-[:R {rank:1}]->(a),(x)-[:R {rank:2}]->(b),(x)-[:R {rank:3}]->(a)"
    )
    result = graph.select("C").connections()["X"]["outgoing"]["R"]
    assert len(result) == 3
    assert sorted((row["node_id"], row["connection_properties"]["rank"]) for row in result.values()) == [
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    assert result == graph.select("C").connections()["X"]["outgoing"]["R"]


@pytest.mark.parametrize("titles", [["", "", "no_title"], ["A_2", "A", "A"], ["A", "A_2", "A"]])
def test_grouped_label_permutations_preserve_parent_identity(titles):
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "UNWIND $rows AS row CREATE (p:P {id:row.id,title:row.title}) CREATE (c:C {id:row.id+10}) CREATE (p)-[:R]->(c)",
        params={"rows": [{"id": i + 1, "title": title} for i, title in enumerate(titles)]},
    )
    view = graph.select("P").traverse("R")
    result = view.collect_grouped("P", parent_info=True, flatten_single_parent=False)
    assert sorted((row["id"], row["children"][0]["id"]) for row in result.values()) == [(1, 11), (2, 12), (3, 13)]
    assert result == view.collect_grouped("P", parent_info=True, flatten_single_parent=False)


def test_selected_connections_keep_same_id_from_distinct_types():
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:P {id:1,title:'A'}),(b:Q {id:1,title:'A'}),(x:C {id:11,title:'X'}) "
        "CREATE (x)-[:R]->(a),(x)-[:R]->(b)"
    )
    result = graph.select("C").traverse("R").connections()
    assert sorted((node["type"], node["node_id"]) for node in result.values()) == [("P", 1), ("Q", 1)]
