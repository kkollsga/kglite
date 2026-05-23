"""Phase A.1 / C6 — pin the new RETURN shapes introduced by the
`Value::{Node, Relationship, Path, List, Map}` enum extension.

These tests are the post-A.1 contract: `RETURN n` yields a dict with
``id``/``labels``/``properties`` (not a title string); `labels(n)` /
`properties(n)` / `nodes(p)` / `relationships(p)` / `collect(n)` /
list/map literals all yield native Python lists/dicts (not JSON strings
re-parsed by an inference hack).

Cross-cuts all three storage modes (memory / mapped / disk) via a
parametrised fixture — every test runs 3× so a backend-specific
projection regression surfaces immediately.

The file is the **canonical reference** for the new shape — when in
doubt about how a RETURN clause materialises, search this file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

import kglite

STORAGE_MODES = ("memory", "mapped", "disk")


def _new_graph(mode: str, tmp_path: Path) -> kglite.KnowledgeGraph:
    if mode == "memory":
        return kglite.KnowledgeGraph()
    if mode == "mapped":
        return kglite.KnowledgeGraph(storage="mapped")
    if mode == "disk":
        return kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "disk_graph"))
    raise ValueError(f"unknown mode {mode!r}")


def _build_person_knows_graph(g: kglite.KnowledgeGraph) -> None:
    """Small Person+KNOWS+LIKES graph fixture used across most tests.

    Three people (alice, bob, carol), two KNOWS edges (alice→bob,
    bob→carol), one LIKES edge (alice→carol). Properties on nodes
    let us test both the canonical (id/title) and user-set
    (name/age/city) shapes.
    """
    persons = pd.DataFrame(
        [
            {"id": "alice", "name": "Alice", "age": 30, "city": "Oslo"},
            {"id": "bob", "name": "Bob", "age": 35, "city": "Bergen"},
            {"id": "carol", "name": "Carol", "age": 28, "city": "Oslo"},
        ]
    )
    g.add_nodes(persons, "Person", "id", "name")

    knows = pd.DataFrame(
        [
            {"src": "alice", "tgt": "bob", "since": 2015},
            {"src": "bob", "tgt": "carol", "since": 2020},
        ]
    )
    g.add_connections(knows, "KNOWS", "Person", "src", "Person", "tgt")

    likes = pd.DataFrame([{"src": "alice", "tgt": "carol"}])
    g.add_connections(likes, "LIKES", "Person", "src", "Person", "tgt")


@pytest.fixture(params=STORAGE_MODES, ids=STORAGE_MODES)
def graph_all_modes(request, tmp_path: Path) -> Iterator[kglite.KnowledgeGraph]:
    """Per-mode Person+KNOWS+LIKES fixture. Every test that uses this
    fixture runs three times — once per storage backend. The disk
    mode goes through a real save/load via the path argument."""
    g = _new_graph(request.param, tmp_path)
    _build_person_knows_graph(g)
    yield g


# ── RETURN n → Value::Node → Python dict ─────────────────────────────────────


class TestReturnNode:
    """RETURN n yields a dict with id/labels/properties (not a title string)."""

    def test_return_node_yields_dict(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person {id: 'alice'}) RETURN n"))
        assert len(rows) == 1
        node = rows[0]["n"]
        assert isinstance(node, dict), f"expected dict, got {type(node).__name__}"

    def test_return_node_has_id_labels_properties_keys(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person {id: 'alice'}) RETURN n"))
        node = rows[0]["n"]
        assert set(node.keys()) >= {"id", "labels", "properties"}

    def test_return_node_labels_is_single_element_list(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person {id: 'alice'}) RETURN n"))
        node = rows[0]["n"]
        assert isinstance(node["labels"], list)
        assert node["labels"] == ["Person"]

    def test_return_node_properties_carries_user_set_columns(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person {id: 'alice'}) RETURN n"))
        node = rows[0]["n"]
        # User set: id (via add_nodes), name (title), age, city
        assert node["properties"]["age"] == 30
        assert node["properties"]["city"] == "Oslo"


class TestReturnRelationship:
    """RETURN r yields a dict with id/start/end/type/properties."""

    def test_return_relationship_yields_dict(self, graph_all_modes):
        # Anchor via the source node to avoid edge-property WHERE
        # filter quirks (those have their own tests).
        rows = list(graph_all_modes.cypher("MATCH (a:Person {id: 'alice'})-[r:KNOWS]->() RETURN r LIMIT 1"))
        assert len(rows) == 1
        rel = rows[0]["r"]
        assert isinstance(rel, dict)

    def test_return_relationship_has_endpoints_and_type(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (a:Person {id: 'alice'})-[r:KNOWS]->() RETURN r LIMIT 1"))
        rel = rows[0]["r"]
        assert set(rel.keys()) >= {"id", "start", "end", "type", "properties"}
        assert rel["type"] == "KNOWS"


class TestReturnPath:
    """RETURN p (variable-length path) yields a dict with nodes/relationships."""

    def test_shortest_path_yields_dict_with_nodes_and_rels(self, graph_all_modes):
        rows = list(
            graph_all_modes.cypher(
                "MATCH p = shortestPath((a:Person {id: 'alice'})-[*]->(c:Person {id: 'carol'})) RETURN p"
            )
        )
        assert len(rows) == 1
        path = rows[0]["p"]
        assert isinstance(path, dict)
        assert "nodes" in path and "relationships" in path
        assert isinstance(path["nodes"], list)
        assert isinstance(path["relationships"], list)
        # shortest alice→carol is one hop via LIKES
        assert len(path["nodes"]) == 2
        assert len(path["relationships"]) == 1


# ── Scalar functions → native lists/maps (not JSON strings) ─────────────────


class TestScalarFunctionsNative:
    def test_labels_returns_list_not_json_string(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person {id: 'alice'}) RETURN labels(n) AS L"))
        labels = rows[0]["L"]
        assert isinstance(labels, list), f"expected list, got {type(labels).__name__}"
        assert labels == ["Person"]

    def test_keys_returns_list_of_strings(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person {id: 'alice'}) RETURN keys(n) AS K"))
        keys = rows[0]["K"]
        assert isinstance(keys, list)
        assert all(isinstance(k, str) for k in keys)
        # Includes virtual id/title/type plus user-set columns
        assert {"id", "title", "type", "age", "city"} <= set(keys)

    def test_properties_returns_dict_not_json_string(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person {id: 'alice'}) RETURN properties(n) AS P"))
        props = rows[0]["P"]
        assert isinstance(props, dict), f"expected dict, got {type(props).__name__}"
        assert props["age"] == 30
        assert props["city"] == "Oslo"

    def test_nodes_returns_list_of_node_dicts(self, graph_all_modes):
        rows = list(
            graph_all_modes.cypher("MATCH p = (a:Person {id: 'alice'})-[*1..2]->(z) RETURN nodes(p) AS ns LIMIT 1")
        )
        ns = rows[0]["ns"]
        assert isinstance(ns, list)
        assert all(isinstance(n, dict) for n in ns)
        assert all("labels" in n for n in ns)

    def test_relationships_returns_list_of_rel_dicts(self, graph_all_modes):
        rows = list(
            graph_all_modes.cypher(
                "MATCH p = (a:Person {id: 'alice'})-[*1..2]->(z) RETURN relationships(p) AS rs LIMIT 1"
            )
        )
        rs = rows[0]["rs"]
        assert isinstance(rs, list)
        assert all(isinstance(r, dict) for r in rs)
        assert all("type" in r for r in rs)


# ── List/map literals + aggregations + comprehensions ────────────────────────


class TestNativeCollections:
    def test_list_literal_yields_python_list(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("RETURN [1, 2, 3] AS xs"))
        assert rows[0]["xs"] == [1, 2, 3]

    def test_map_literal_yields_python_dict(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("RETURN {a: 1, b: 'two'} AS m"))
        assert rows[0]["m"] == {"a": 1, "b": "two"}

    def test_collect_yields_list_of_node_dicts(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person) RETURN collect(n) AS ns"))
        ns = rows[0]["ns"]
        assert isinstance(ns, list)
        assert len(ns) == 3
        assert all(isinstance(n, dict) and "labels" in n for n in ns)

    def test_collect_scalar_yields_list_of_scalars(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("MATCH (n:Person) RETURN collect(n.name) AS names"))
        names = rows[0]["names"]
        assert sorted(names) == ["Alice", "Bob", "Carol"]

    def test_range_yields_python_list(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("RETURN range(1, 4) AS xs"))
        assert rows[0]["xs"] == [1, 2, 3, 4]

    def test_list_comprehension_yields_python_list(self, graph_all_modes):
        # List comprehensions don't accept aggregates in the source —
        # use a literal range / list to feed the comprehension.
        rows = list(graph_all_modes.cypher("RETURN [x IN [10, 20, 30, 40] WHERE x > 15 | x * 2] AS doubled"))
        doubled = rows[0]["doubled"]
        assert isinstance(doubled, list)
        assert doubled == [40, 60, 80]

    def test_list_slice_yields_python_list(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("RETURN [1, 2, 3, 4, 5][1..3] AS s"))
        assert rows[0]["s"] == [2, 3]


# ── Cross-feature: chained Node access through WITH / WHERE / ORDER BY ─────


class TestNodeChaining:
    def test_with_node_then_property_access(self, graph_all_modes):
        rows = list(
            graph_all_modes.cypher("MATCH (n:Person) WITH n WHERE n.age > 28 RETURN n.name AS name ORDER BY name")
        )
        names = [r["name"] for r in rows]
        assert names == ["Alice", "Bob"]

    def test_any_over_collected_nodes_with_property_filter(self, graph_all_modes):
        # Pattern that broke after C2 before parse_list_value learned
        # the native Value::List path — the WITH wraps the collected
        # nodes, then any(x IN xs WHERE x.prop ...) iterates.
        rows = list(
            graph_all_modes.cypher(
                "MATCH (n:Person) WITH collect(n) AS xs RETURN any(x IN xs WHERE x.age > 30) AS has_over_30"
            )
        )
        assert rows[0]["has_over_30"] is True


# ── DETACH DELETE survives same-query RETURN ────────────────────────────────


class TestDeleteSurvivesCountInSameQuery:
    """Cypher semantics: count(n) in the same query as DELETE n must
    return the matched-row count, not the post-delete count. Phase A.1 / C2
    preserves this via the tombstone Node path in the Variable resolver."""

    def test_detach_delete_count_in_same_query(self, graph_all_modes):
        rows = list(
            graph_all_modes.cypher("MATCH (n:Person) WHERE n.age >= 30 DETACH DELETE n RETURN count(n) AS deleted")
        )
        assert rows[0]["deleted"] == 2

    def test_remaining_nodes_after_delete(self, graph_all_modes):
        # First delete...
        list(graph_all_modes.cypher("MATCH (n:Person) WHERE n.age >= 30 DETACH DELETE n"))
        # ...then verify what's left
        rows = list(graph_all_modes.cypher("MATCH (n:Person) RETURN n.name AS name"))
        assert [r["name"] for r in rows] == ["Carol"]


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_list_literal(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("RETURN [] AS empty"))
        assert rows[0]["empty"] == []

    def test_empty_map_literal(self, graph_all_modes):
        rows = list(graph_all_modes.cypher("RETURN {} AS empty"))
        assert rows[0]["empty"] == {}

    def test_nested_list_of_nodes_in_map(self, graph_all_modes):
        rows = list(
            graph_all_modes.cypher(
                "MATCH (n:Person) WITH collect(n) AS people RETURN {people: people, count: size(people)} AS bundle"
            )
        )
        bundle = rows[0]["bundle"]
        assert isinstance(bundle, dict)
        assert bundle["count"] == 3
        assert isinstance(bundle["people"], list)
        assert len(bundle["people"]) == 3
        assert all(isinstance(p, dict) and "labels" in p for p in bundle["people"])
