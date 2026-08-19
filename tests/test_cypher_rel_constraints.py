"""Relationship constraints end to end, through the wheel.

`CREATE CONSTRAINT FOR ()-[r:TYPE]-() REQUIRE r.p IS NOT NULL | IS :: TYPE` —
declared, validated against existing relationships, enforced on every write
path, listed, dropped, and persisted. The Rust suites cover the engine; this
file is the binding's contract: the typed exception a caller catches, the
relationship vocabulary the message is written in, and the surfaces
(`SHOW CONSTRAINTS`, `db.constraints()`, `describe()`) a caller reads.

`IS UNIQUE` / `IS RELATIONSHIP KEY` on a relationship stay refused — the
deferral is asserted here too, because "not yet" and "not ever under this data
model" are different promises and the message has to make that clear.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

RELATIONSHIP_WORDS = "relationship"


def _graph(storage: str = "memory", path: str | None = None) -> kglite.KnowledgeGraph:
    """Two `Person`s joined by one `KNOWS` carrying `since` and `weight`."""
    if storage == "memory":
        graph = kglite.KnowledgeGraph()
    elif storage == "mapped":
        graph = kglite.KnowledgeGraph(storage="mapped")
    else:
        graph = kglite.KnowledgeGraph(storage="disk", path=path)
    graph.add_nodes(
        pd.DataFrame({"person_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]}),
        "Person",
        "person_id",
        "name",
    )
    graph.add_connections(
        pd.DataFrame({"s": [1], "t": [2], "since": [2020], "weight": [3]}),
        "KNOWS",
        "Person",
        "s",
        "Person",
        "t",
    )
    return graph


def _rows(graph) -> list[dict]:
    return graph.cypher("SHOW CONSTRAINTS").to_list()


# ── declaration ──────────────────────────────────────────────────────


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
def test_both_kinds_install_over_clean_relationship_data(storage, tmp_path):
    graph = _graph(storage, str(tmp_path / "kg"))
    graph.cypher("CREATE CONSTRAINT knows_since FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    graph.cypher("CREATE CONSTRAINT knows_weight FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    by_name = {row["name"]: row for row in _rows(graph)}
    assert by_name["knows_since"]["type"] == "RELATIONSHIP_PROPERTY_EXISTENCE"
    assert by_name["knows_weight"]["type"] == "RELATIONSHIP_PROPERTY_TYPE"
    assert by_name["knows_weight"]["propertyType"] == "INTEGER"


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
def test_a_declaration_the_stored_relationships_violate_is_refused(storage, tmp_path):
    graph = _graph(storage, str(tmp_path / "kg"))
    graph.add_connections(
        pd.DataFrame({"s": [1], "t": [3]}),  # no `since`
        "KNOWS",
        "Person",
        "s",
        "Person",
        "t",
    )
    with pytest.raises(kglite.ConstraintCreationError) as exc:
        graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    message = str(exc.value)
    assert "existing relationship" in message, message
    assert "node" not in message, message
    assert _rows(graph) == [], "a refused declaration installs nothing"


def test_a_violating_stored_type_is_refused_with_relationship_prose():
    graph = _graph()
    graph.cypher(
        "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 3}) CREATE (a)-[:KNOWS {weight: 'heavy'}]->(b)"
    )
    with pytest.raises(kglite.ConstraintCreationError) as exc:
        graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    message = str(exc.value)
    assert "STRING" in message and "INTEGER" in message, message
    assert RELATIONSHIP_WORDS in message, message


# ── enforcement through Cypher ───────────────────────────────────────


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
def test_create_set_and_remove_are_all_gated(storage, tmp_path):
    graph = _graph(storage, str(tmp_path / "kg"))
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")

    with pytest.raises(kglite.ConstraintViolationError):
        graph.cypher(
            "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 3}) "
            "CREATE (a)-[:KNOWS {weight: 1}]->(b)"  # no `since`
        )
    with pytest.raises(kglite.ConstraintViolationError):
        graph.cypher("MATCH ()-[r:KNOWS]->() SET r.weight = 'heavy'")
    with pytest.raises(kglite.ConstraintViolationError):
        graph.cypher("MATCH ()-[r:KNOWS]->() SET r.since = null")
    with pytest.raises(kglite.ConstraintViolationError):
        graph.cypher("MATCH ()-[r:KNOWS]->() REMOVE r.since")

    # Nothing above landed: one relationship, unchanged.
    rows = graph.cypher("MATCH ()-[r:KNOWS]->() RETURN r.since AS since, r.weight AS weight").to_list()
    assert rows == [{"since": 2020, "weight": 3}]


def test_the_typed_exception_carries_relationship_prose():
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    with pytest.raises(kglite.ConstraintViolationError) as exc:
        graph.cypher("MATCH ()-[r:KNOWS]->() REMOVE r.since")
    message = str(exc.value)
    assert "a relationship of type 'KNOWS'" in message, message
    assert "node" not in message, message
    # The node advice would be actively wrong here — a relationship MERGE needs
    # both endpoints bound — so it must not appear.
    assert "MERGE" not in message, message
    assert isinstance(exc.value, kglite.ConstraintError)


def test_map_assignment_spellings_are_gated():
    for statement in (
        "MATCH ()-[r:KNOWS]->() SET r = {since: 2020, weight: 'heavy'}",
        "MATCH ()-[r:KNOWS]->() SET r += {weight: 'heavy'}",
    ):
        graph = _graph()
        graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
        with pytest.raises(kglite.ConstraintViolationError):
            graph.cypher(statement)


# ── enforcement through the bulk loader, per conflict mode ───────────


def _load(graph, rows: pd.DataFrame, mode: str | None = None):
    return graph.add_connections(rows, "KNOWS", "Person", "s", "Person", "t", conflict_handling=mode)


@pytest.mark.parametrize(
    "mode,accepted",
    [
        # The row's value wins, so a bad one lands — and is refused.
        ("update", False),
        # The stored value wins: the row's bad value is discarded, so refusing
        # it would reject a write the engine never performs.
        ("preserve", True),
        # The stored properties are dropped and rebuilt from the row.
        ("replace", False),
    ],
)
def test_bulk_conflict_modes_are_judged_on_the_state_they_produce(mode, accepted):
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    frame = pd.DataFrame({"s": [1], "t": [2], "weight": ["heavy"]})
    if accepted:
        _load(graph, frame, mode)
        stored = graph.cypher("MATCH ()-[r:KNOWS]->() RETURN r.weight AS w").to_list()
        assert stored == [{"w": 3}], "preserve keeps the stored value"
    else:
        with pytest.raises(kglite.ConstraintViolationError):
            _load(graph, frame, mode)


def test_sum_refuses_an_addition_that_changes_the_type():
    """The mode that produces a value neither side wrote: 3 + 1.5 is a FLOAT,
    and an INTEGER declaration has to catch it. A gate judging the row alone
    would pass this."""
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    with pytest.raises(kglite.ConstraintViolationError) as exc:
        _load(graph, pd.DataFrame({"s": [1], "t": [2], "weight": [1.5]}), "sum")
    assert "FLOAT" in str(exc.value), str(exc.value)
    # And an addition that stays in type is accepted.
    _load(graph, pd.DataFrame({"s": [1], "t": [2], "weight": [4]}), "sum")
    assert graph.cypher("MATCH ()-[r:KNOWS]->() RETURN r.weight AS w").to_list() == [{"w": 7}]


def test_a_partial_update_frame_keeps_the_required_property():
    """The bulk contract: a frame that does not carry the required column
    leaves the stored value alone, so it must not be refused."""
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    _load(graph, pd.DataFrame({"s": [1], "t": [2], "weight": [9]}), "update")
    assert graph.cypher("MATCH ()-[r:KNOWS]->() RETURN r.since AS s").to_list() == [{"s": 2020}]


def test_a_refused_frame_leaves_the_existing_relationships_alone():
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    with pytest.raises(kglite.ConstraintViolationError):
        _load(graph, pd.DataFrame({"s": [1], "t": [3]}), "update")
    assert graph.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS n").to_list() == [{"n": 1}]


# ── DROP ─────────────────────────────────────────────────────────────


def test_drop_by_name_and_by_descriptor():
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT knows_since FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    graph.cypher("DROP CONSTRAINT knows_since")
    assert _rows(graph) == []
    # Enforcement goes with it.
    graph.cypher("MATCH ()-[r:KNOWS]->() REMOVE r.since")

    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    assert [row["name"] for row in _rows(graph)] == ["KNOWS.weight"]
    graph.cypher("DROP CONSTRAINT `KNOWS.weight`")
    assert _rows(graph) == []


# ── surfaces ─────────────────────────────────────────────────────────


def test_show_constraints_and_db_constraints_report_the_same_rows():
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT knows_since FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    columns = "name, type, entityType, labelsOrTypes, properties, propertyType"
    assert _rows(graph) == graph.cypher(f"CALL db.constraints() YIELD {columns}").to_list()
    row = _rows(graph)[0]
    assert row["entityType"] == "RELATIONSHIP"
    assert row["labelsOrTypes"] == ["KNOWS"]
    assert row["properties"] == ["since"]


def test_describe_annotates_the_constrained_edge_property():
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    described = graph.describe(connections=["KNOWS"])
    assert 'constraint="not_null"' in described, described
    assert 'declared_type="INTEGER"' in described, described


# ── persistence ──────────────────────────────────────────────────────


def test_a_saved_graph_reloads_still_enforcing(tmp_path):
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT knows_since FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    path = str(tmp_path / "rel.kgl")
    graph.save(path)

    reloaded = kglite.load(path)
    names = {row["name"] for row in _rows(reloaded)}
    assert names == {"knows_since", "KNOWS.weight"}
    with pytest.raises(kglite.ConstraintViolationError):
        reloaded.cypher("MATCH ()-[r:KNOWS]->() REMOVE r.since")
    with pytest.raises(kglite.ConstraintViolationError):
        reloaded.cypher("MATCH ()-[r:KNOWS]->() SET r.weight = 'heavy'")
    # And the name still drops what it named.
    reloaded.cypher("DROP CONSTRAINT knows_since")
    reloaded.cypher("MATCH ()-[r:KNOWS]->() REMOVE r.since")


# ── refusals ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("requirement", ["IS UNIQUE", "IS RELATIONSHIP KEY"])
def test_relationship_uniqueness_names_the_data_model_reason(requirement):
    graph = _graph()
    with pytest.raises(kglite.CypherExecutionError) as exc:
        graph.cypher(f"CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since {requirement}")
    message = str(exc.value)
    assert "KNOWS" in message, message
    assert "parallel edges" in message, message
    assert "IS NOT NULL" in message, "the message must name what *is* served"


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS RELATIONSHIP UNIQUE",
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS RELATIONSHIP KEY",
        "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NODE UNIQUE",
        "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NODE KEY",
    ],
)
def test_a_scope_word_contradicting_the_pattern_is_refused(statement):
    graph = _graph()
    with pytest.raises((kglite.CypherSyntaxError, kglite.CypherExecutionError)) as exc:
        graph.cypher(statement)
    assert "does not match the FOR pattern" in str(exc.value), str(exc.value)


def test_a_locked_schema_gates_the_type_but_not_the_property():
    graph = _graph()
    graph.lock_schema()
    with pytest.raises(kglite.CypherExecutionError) as exc:
        graph.cypher("CREATE CONSTRAINT FOR ()-[r:MISSING]-() REQUIRE r.p IS NOT NULL")
    assert "schema is locked" in str(exc.value)
    # The lock does not check edge property names on write, so constraining an
    # unseen one must stay legal — refusing it would make DDL stricter than the
    # lock it is agreeing with.
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.unseen IS :: INTEGER")


# ── change capture ───────────────────────────────────────────────────


def test_a_refused_relationship_write_publishes_no_change_event():
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    graph.cypher("CALL db.cdc.enable()")
    cursor = graph.cypher("CALL db.cdc.current() YIELD id RETURN id").to_list()[0]["id"]

    with pytest.raises(kglite.ConstraintViolationError):
        graph.cypher("MATCH ()-[r:KNOWS]->() SET r.weight = 'heavy'")
    events = graph.cypher("CALL db.cdc.query({from: $c})", params={"c": cursor}).to_list()
    assert events == [], events


def test_an_accepted_relationship_write_still_publishes():
    """The other half of the no-phantom claim: the gate must not have made the
    path silent for writes that do land."""
    graph = _graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    graph.cypher("CALL db.cdc.enable()")
    cursor = graph.cypher("CALL db.cdc.current() YIELD id RETURN id").to_list()[0]["id"]

    graph.cypher("MATCH ()-[r:KNOWS]->() SET r.weight = 42")
    events = graph.cypher("CALL db.cdc.query({from: $c})", params={"c": cursor}).to_list()
    assert [event["elementType"] for event in events] == ["relationship"], events
