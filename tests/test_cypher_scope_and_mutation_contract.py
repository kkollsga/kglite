"""Independent scope, map-mutation, and relationship identity contracts."""

from __future__ import annotations

import pytest

import kglite


def _graph():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Person {id: 1, name: 'Ada', old: true})").to_list()
    graph.cypher("CREATE (:Person {id: 2, name: 'Bob'})").to_list()
    graph.cypher(
        "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:KNOWS {since: 2020, old: true}]->(b)"
    ).to_list()
    return graph


@pytest.mark.parametrize(
    "query",
    (
        "RETURN missing AS value",
        "MATCH (n:Person) RETURN missing.name AS value",
        "MATCH (n:Person) WITH n.name AS name RETURN n",
        "MATCH (n:Person) SET missing.value = 1",
    ),
)
def test_undefined_variables_are_rejected_before_execution(query):
    with pytest.raises(kglite.SchemaError, match="Undefined variable"):
        _graph().cypher(query).to_list()


def test_merge_rejects_null_property_without_mutating():
    graph = _graph()
    before = graph.cypher("MATCH (n:Person) RETURN count(n) AS count").to_list()
    with pytest.raises(kglite.CypherExecutionError, match="MERGE.*null.*id"):
        graph.cypher("MERGE (:Person {id: null})").to_list()
    assert graph.cypher("MATCH (n:Person) RETURN count(n) AS count").to_list() == before


def test_set_map_merge_preserves_unspecified_node_properties():
    graph = _graph()
    graph.cypher("MATCH (n:Person {id: 1}) SET n += {city: 'Oslo', active: true}").to_list()
    assert graph.cypher(
        "MATCH (n:Person {id: 1}) RETURN n.name AS name, n.old AS old, n.city AS city, n.active AS active"
    ).to_list() == [{"name": "Ada", "old": True, "city": "Oslo", "active": True}]


def test_set_map_replace_removes_unspecified_mutable_node_properties():
    graph = _graph()
    graph.cypher("MATCH (n:Person {id: 1}) SET n = {name: 'Ada Lovelace', city: 'London'}").to_list()
    assert graph.cypher(
        "MATCH (n:Person {id: 1}) RETURN n.id AS id, n.name AS name, n.old AS old, n.city AS city"
    ).to_list() == [{"id": 1, "name": "Ada Lovelace", "old": None, "city": "London"}]


def test_set_map_replace_clears_unspecified_title_alias():
    graph = _graph()
    graph.cypher("MATCH (n:Person {id: 1}) SET n = {city: 'Oslo'}").to_list()
    assert graph.cypher("MATCH (n:Person {id: 1}) RETURN n.name AS name, n.city AS city").to_list() == [
        {"name": None, "city": "Oslo"}
    ]


def test_set_map_forms_apply_to_relationships():
    graph = _graph()
    graph.cypher("MATCH ()-[r:KNOWS]->() SET r += {weight: 2}").to_list()
    graph.cypher("MATCH ()-[r:KNOWS]->() SET r = {since: 2024}").to_list()
    assert graph.cypher(
        "MATCH ()-[r:KNOWS]->() RETURN r.since AS since, r.old AS old, r.weight AS weight"
    ).to_list() == [{"since": 2024, "old": None, "weight": None}]


@pytest.mark.parametrize("mode", ("memory", "mapped", "disk"))
def test_set_map_forms_are_storage_mode_independent(tmp_path, mode):
    if mode == "memory":
        graph = kglite.KnowledgeGraph()
    elif mode == "mapped":
        graph = kglite.KnowledgeGraph(storage="mapped")
    else:
        graph = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "disk-graph"))

    graph.cypher("CREATE (:Item {id: 1, old: true})").to_list()
    graph.cypher("MATCH (n:Item) SET n += {added: 2}").to_list()
    graph.cypher("MATCH (n:Item) SET n = {kept: 3}").to_list()
    assert graph.cypher(
        "MATCH (n:Item) RETURN n.id AS id, n.old AS old, n.added AS added, n.kept AS kept"
    ).to_list() == [{"id": 1, "old": None, "added": None, "kept": 3}]


def test_relationship_id_is_stable_across_variable_and_materialized_value():
    graph = _graph()
    rows = graph.cypher("MATCH ()-[r:KNOWS]->() WITH r, id(r) AS direct RETURN direct, id(r) AS carried").to_list()
    assert len(rows) == 1
    assert isinstance(rows[0]["direct"], int)
    assert rows[0]["direct"] == rows[0]["carried"]


# ── Node identity through projected node values ─────────────────────────
#
# A node variable re-used in a later MATCH pins the pattern to exactly
# that node (openCypher re-MATCH identity) — including when the row
# carries it only as a projected Value::Node (`UNWIND collect(n) AS n`),
# not a live binding. Parallel to the relationship-identity contract
# above.


def _paired_graph():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:A {id: 1}), (:A {id: 2}), (:B {id: 10}), (:B {id: 20})").to_list()
    graph.cypher("MATCH (a:A {id: 1}), (b:B {id: 10}) CREATE (a)-[:R]->(b)").to_list()
    graph.cypher("MATCH (a:A {id: 2}), (b:B {id: 20}) CREATE (a)-[:R]->(b)").to_list()
    return graph


def test_node_identity_pinned_through_with():
    graph = _paired_graph()
    rows = graph.cypher("MATCH (n:A) WITH n MATCH (n)-[:R]->(m) RETURN n.id AS nid, m.id AS mid ORDER BY nid").to_list()
    assert rows == [{"nid": 1, "mid": 10}, {"nid": 2, "mid": 20}]


def test_node_identity_pinned_through_projected_node_value():
    # UNWIND over collect(n) carries `n` as a projected Value::Node; the
    # later MATCH must bind exactly that node, not cartesian-join against
    # every :R edge (which produced 4 rows with mismatched pairs).
    graph = _paired_graph()
    rows = graph.cypher(
        "MATCH (n:A) WITH collect(n) AS ns UNWIND ns AS n "
        "MATCH (n)-[:R]->(m) RETURN n.id AS nid, m.id AS mid ORDER BY nid"
    ).to_list()
    assert rows == [{"nid": 1, "mid": 10}, {"nid": 2, "mid": 20}]


def test_node_identity_pinned_in_exists_through_projected_node_value():
    graph = _paired_graph()
    rows = graph.cypher(
        "MATCH (n:A) WITH collect(n) AS ns UNWIND ns AS n "
        "RETURN n.id AS nid, EXISTS { (n)-[:R]->(:B {id: 10}) } AS to10 ORDER BY nid"
    ).to_list()
    assert rows == [{"nid": 1, "to10": True}, {"nid": 2, "to10": False}]


# ── Writes on NULL targets are no-ops (openCypher) ──────────────────────
#
# An OPTIONAL MATCH miss leaves the variable null-valued; SET / REMOVE on
# it must skip that row (mirroring DELETE's existing null handling), while
# a truly undefined variable still fails scope validation before execution
# (covered by test_undefined_variables_are_rejected_before_execution).


def _optional_miss_graph():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Person {id: 1, name: 'Ada', keep: 1})").to_list()
    return graph


@pytest.mark.parametrize(
    "write_clause",
    (
        "SET m.x = 1",
        "SET m = {x: 1}",
        "SET m += {x: 1}",
        "SET m:Flagged",
        "REMOVE m.keep",
        "REMOVE m:Flagged",
    ),
)
def test_write_on_null_optional_variable_is_noop(write_clause):
    graph = _optional_miss_graph()
    rows = graph.cypher(
        f"MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) {write_clause} RETURN n.name AS name"
    ).to_list()
    assert rows == [{"name": "Ada"}]
    # The anchor node is untouched.
    assert graph.cypher("MATCH (n:Person) RETURN n.keep AS keep").to_list() == [{"keep": 1}]


def test_set_on_matched_optional_variable_still_writes():
    graph = _optional_miss_graph()
    graph.cypher("CREATE (:Person {id: 2, name: 'Bob'})").to_list()
    graph.cypher("MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:KNOWS]->(b)").to_list()
    graph.cypher("MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) SET m.x = 7").to_list()
    # Bob (the only KNOWS target) got the write; Ada's miss row was a no-op.
    assert graph.cypher("MATCH (p:Person) RETURN p.name AS name, p.x AS x ORDER BY name").to_list() == [
        {"name": "Ada", "x": None},
        {"name": "Bob", "x": 7},
    ]


# ── DELETE is statement-atomic ──────────────────────────────────────────


def test_delete_edge_and_node_in_one_statement():
    graph = _graph()
    graph.cypher("MATCH (a:Person {id: 1})-[r:KNOWS]->(b) DELETE r, a").to_list()
    assert graph.cypher("MATCH (n:Person) RETURN count(n) AS n").to_list() == [{"n": 1}]
    assert graph.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS n").to_list() == [{"n": 0}]


def test_plain_delete_still_rejects_node_with_surviving_edges():
    graph = _graph()
    with pytest.raises(kglite.CypherExecutionError, match="still has relationships"):
        graph.cypher("MATCH (a:Person {id: 1}) DELETE a").to_list()


def test_plain_delete_error_names_node_without_debug_formatting():
    graph = _graph()
    with pytest.raises(kglite.CypherExecutionError, match="node 'Ada'"):
        graph.cypher("MATCH (a:Person {id: 1}) DELETE a").to_list()
