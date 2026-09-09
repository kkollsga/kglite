"""Absolute mutation contracts for Cypher 25 INSERT."""

from __future__ import annotations

import pytest

import kglite


def _graph(mode: str, tmp_path) -> kglite.KnowledgeGraph:
    if mode == "memory":
        return kglite.KnowledgeGraph()
    if mode == "disk":
        return kglite.KnowledgeGraph(storage=mode, path=str(tmp_path / "insert-disk"))
    return kglite.KnowledgeGraph(storage=mode)


@pytest.mark.parametrize("mode", ("memory", "mapped", "disk"))
def test_insert_nodes_relationships_bindings_and_stats_in_every_storage_mode(mode, tmp_path):
    graph = _graph(mode, tmp_path)
    rows = graph.cypher(
        "INSERT (a IS Person&Actor {id: 1, name: 'Ada'}), "
        "(b:Person {id: 2, name: 'Bob'}), "
        "(a)-[r IS KNOWS {since: 2020}]->(b) "
        "RETURN a.id AS source, b.id AS target, labels(a) AS labels, "
        "type(r) AS kind, r.since AS since"
    ).to_list()

    assert rows == [
        {
            "source": 1,
            "target": 2,
            "labels": ["Person", "Actor"],
            "kind": "KNOWS",
            "since": 2020,
        }
    ]
    assert graph.last_mutation_stats["nodes_created"] == 2
    assert graph.last_mutation_stats["relationships_created"] == 1
    assert graph.cypher(
        "MATCH (a:Person {id: 1})-[r:KNOWS]->(b:Person {id: 2}) "
        "RETURN a.name AS source, b.name AS target, r.since AS since"
    ).to_list() == [{"source": "Ada", "target": "Bob", "since": 2020}]


def test_insert_is_soft_and_is_can_still_be_a_pattern_name():
    graph = kglite.KnowledgeGraph()
    assert graph.cypher("INSERT (insert:insert {insert: 1}) RETURN insert.insert AS insert").to_list() == [
        {"insert": 1}
    ]
    graph.cypher("INSERT (IS {id: 2}), (IS IS {id: 3})").to_list()
    assert graph.cypher("MATCH (n:IS) RETURN n.id AS id").to_list() == [{"id": 3}]


def test_insert_rejects_create_only_forms_before_any_mutation():
    graph = kglite.KnowledgeGraph()
    invalid = (
        "INSERT (:A:B)",
        "INSERT (:A&B:C)",
        "INSERT (n:$label)",
        "INSERT (n $properties)",
        "INSERT path = ()-[:R]->()",
        "INSERT ()-[r]->()",
        "INSERT ()-[:$type]->()",
        "INSERT ()-[:A&B]->()",
        "INSERT ()-[:R]-()",
        "INSERT ()-[:R*2]->()",
    )
    for query in invalid:
        with pytest.raises(kglite.CypherSyntaxError):
            graph.cypher(query, params={"label": "A", "type": "R", "properties": {"id": 1}}).to_list()
        assert graph.cypher("MATCH (n) RETURN count(n) AS n").to_list() == [{"n": 0}]


def test_create_static_and_dynamic_forms_are_unchanged():
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:Person:Actor {id: 1}), (b:$label {id: 2}), (a)-[:$type]->(b)",
        params={"label": "Target", "type": "KNOWS"},
    ).to_list()
    assert graph.cypher(
        "MATCH (a:Actor {id: 1})-[r:KNOWS]->(b:Target {id: 2}) RETURN labels(a) AS labels, type(r) AS kind"
    ).to_list() == [{"labels": ["Person", "Actor"], "kind": "KNOWS"}]


def test_insert_is_present_in_compact_and_detailed_introspection():
    graph = kglite.KnowledgeGraph()
    assert '<clause name="INSERT">' in graph.describe(cypher=True)
    detail = graph.describe(cypher=["INSERT"])
    assert "<INSERT>" in detail
    assert "INSERT (n IS Person&amp;Actor" in detail
