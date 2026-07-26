"""Cypher constraint DDL — `CREATE CONSTRAINT` / `DROP CONSTRAINT` / `SHOW CONSTRAINTS`.

Sibling of `test_cypher_ddl_indexes.py`, and the same acceptance bar:
*portability*. A vanilla Neo4j 5 schema-setup script must run against KGLite and
either take effect or fail with a specific unsupported-feature message. Never a
syntax error, and — the failure mode that matters most here — never a statement
that reports success without enforcing anything. Users build data-integrity
assumptions on a `CREATE CONSTRAINT` that returned cleanly.

`test_every_declared_constraint_actually_rejects_a_violating_write` is the
load-bearing test: for each supported form it asserts the constraint is *enforced*
afterwards, not merely listed.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

# Constraint violations raised through Cypher arrive as `CypherExecutionError`,
# not as the typed `ConstraintViolationError`: the Cypher executor's error
# channel is `Result<_, String>`, so the structured violation is rendered to text
# before any binding sees it. That is a pre-existing gap (Sprint 3's
# `tests/test_constraints.py` matches on message text for the same reason), not
# something constraint DDL changed — the enforcement itself is identical. These
# tests therefore assert on the message, which is the contract that actually
# holds today.
UNIQUE_ERROR = "rejects the duplicate"
NOT_NULL_ERROR = "must have the property"
DECLARATION_ERROR = "cannot declare"


def _graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [28, 35, 42],
                "city": ["Oslo", "Bergen", "Oslo"],
            }
        ),
        "Person",
        "person_id",
        "name",
    )
    return g


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    return _graph()


def _constraint_rows(g: kglite.KnowledgeGraph) -> list[dict]:
    return g.cypher("SHOW CONSTRAINTS").to_list()


def _names(g: kglite.KnowledgeGraph) -> set[str]:
    return {row["name"] for row in _constraint_rows(g)}


# ── CREATE CONSTRAINT ────────────────────────────────────────────────


def test_unique_constraint_is_declared_and_enforced(graph) -> None:
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE")
    assert graph.last_mutation_stats["constraints_added"] == 1
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        graph.cypher("CREATE (p:Person {person_id: 9, name: 'Alice'})")


def test_not_null_constraint_is_declared_and_enforced(graph) -> None:
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL")
    assert graph.last_mutation_stats["constraints_added"] == 1
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        graph.cypher("MATCH (p:Person {person_id: 1}) REMOVE p.name")


def test_node_key_enforces_both_uniqueness_and_presence(graph) -> None:
    graph.cypher("CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE p.name IS NODE KEY")
    # Presence half.
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        graph.cypher("MATCH (p:Person {person_id: 1}) REMOVE p.name")
    # Uniqueness half.
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        graph.cypher("CREATE (p:Person {person_id: 9, name: 'Alice'})")


def test_composite_unique_constraint_constrains_the_tuple_not_each_property(graph) -> None:
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE (p.city, p.age) IS UNIQUE")
    # `city` alone repeats in the fixture, so a plain per-property reading would
    # already have rejected the declaration.
    graph.cypher("CREATE (p:Person {person_id: 9, name: 'D', city: 'Oslo', age: 99})")
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        graph.cypher("CREATE (p:Person {person_id: 10, name: 'E', city: 'Oslo', age: 99})")


def test_declaring_against_dirty_data_is_rejected_and_names_the_value(graph) -> None:
    """`city` has two 'Oslo' rows, so uniqueness cannot be declared."""
    with pytest.raises(Exception, match=DECLARATION_ERROR) as exc:
        graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.city IS UNIQUE")
    message = str(exc.value)
    assert "Oslo" in message, message
    # Nothing installed, so the graph is as permissive as before.
    assert _constraint_rows(graph) == []
    graph.cypher("CREATE (p:Person {person_id: 9, name: 'D', city: 'Oslo'})")


def test_declaring_not_null_against_missing_values_is_rejected(graph) -> None:
    graph.cypher("CREATE (p:Person {person_id: 9, name: 'D'})")
    with pytest.raises(Exception, match=DECLARATION_ERROR):
        graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.city IS NOT NULL")
    assert _constraint_rows(graph) == []


def test_if_not_exists_makes_a_duplicate_declaration_a_no_op(graph) -> None:
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE")
    with pytest.raises(kglite.CypherExecutionError, match="already exists"):
        graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE")
    graph.cypher("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
    assert graph.last_mutation_stats["constraints_added"] == 0


def test_property_type_constraint_is_rejected_not_silently_accepted(graph) -> None:
    """The one outcome worse than an error is a success that enforces nothing."""
    with pytest.raises(kglite.CypherExecutionError) as exc:
        graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    message = str(exc.value)
    assert "is not supported" in message, message
    assert "lock_schema" in message, message
    assert _constraint_rows(graph) == []


@pytest.mark.parametrize("property_name", ["person_id", "id"])
def test_uniqueness_on_the_id_field_is_rejected_not_silently_unenforced(
    property_name: str,
) -> None:
    """`id` is a NodeData field, not a stored property, so the write path never
    produces a claim for it — the constraint admitted duplicates while reporting
    success. It is now refused and points at the route that does enforce.

    (`person_id` is the fixture's unique_id_field, so it aliases `id`.)
    """
    g = _graph()
    with pytest.raises(kglite.CypherExecutionError) as exc:
        g.cypher(f"CREATE CONSTRAINT c FOR (p:Person) REQUIRE p.{property_name} IS UNIQUE")
    message = str(exc.value)
    assert "is not supported" in message, message
    assert "primary_key" in message, message
    assert _constraint_rows(g) == []


@pytest.mark.parametrize("property_name", ["name", "title", "age"])
def test_uniqueness_on_title_and_ordinary_properties_stays_allowed(property_name: str) -> None:
    """The guard above must not over-refuse. `name` is the fixture's
    node_title_field and `title` is the structural field itself; both enforce
    correctly, so refusing them would cost the very common
    `REQUIRE p.name IS UNIQUE`."""
    g = _graph()
    g.cypher(f"CREATE CONSTRAINT c FOR (p:Person) REQUIRE p.{property_name} IS UNIQUE")
    assert g.last_mutation_stats["constraints_added"] == 1
    existing = g.cypher(f"MATCH (p:Person) RETURN p.{property_name} AS v").to_list()[0]["v"]
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        g.cypher(f"CREATE (p:Person {{{property_name}: $v}})".replace("$v", repr(existing)))


def test_node_key_on_the_id_field_is_rejected_too(graph) -> None:
    """A node key carries a uniqueness half, so it inherits the restriction."""
    with pytest.raises(kglite.CypherExecutionError, match="is not supported"):
        graph.cypher("CREATE CONSTRAINT c FOR (p:Person) REQUIRE p.person_id IS NODE KEY")
    assert _constraint_rows(graph) == []


def test_not_null_on_the_id_field_is_still_accepted(graph) -> None:
    """`id` is present by construction, so the requirement is genuinely satisfied
    rather than ignored — no reason to refuse it."""
    graph.cypher("CREATE CONSTRAINT c FOR (p:Person) REQUIRE p.person_id IS NOT NULL")
    assert graph.last_mutation_stats["constraints_added"] == 1


def test_the_primary_key_route_actually_enforces_identity_uniqueness(graph) -> None:
    """The route the rejection message names must work, under both spellings."""
    graph.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    for duplicate in ["CREATE (p:Person {id: 1})", "CREATE (p:Person {person_id: 1})"]:
        with pytest.raises(Exception, match="duplicate primary key"):
            graph.cypher(duplicate)


def test_relationship_constraint_is_rejected_by_name(graph) -> None:
    with pytest.raises(kglite.CypherExecutionError, match="KNOWS"):
        graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS UNIQUE")


# ── names ────────────────────────────────────────────────────────────


def test_the_common_ported_script_shape_round_trips(graph) -> None:
    """Declare under a name, then drop by it — what a Neo4j script actually does."""
    graph.cypher("CREATE CONSTRAINT person_name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE")
    assert "person_name_unique" in _names(graph)
    graph.cypher("DROP CONSTRAINT person_name_unique")
    assert _constraint_rows(graph) == []
    # Enforcement is gone with it.
    graph.cypher("CREATE (p:Person {person_id: 9, name: 'Alice'})")


def test_an_unnamed_constraint_is_addressable_by_its_descriptor(graph) -> None:
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE")
    assert _names(graph) == {"Person.name"}
    graph.cypher("DROP CONSTRAINT `Person.name`")
    assert _constraint_rows(graph) == []


def test_reusing_a_name_for_a_different_constraint_is_rejected(graph) -> None:
    graph.cypher("CREATE CONSTRAINT dup FOR (p:Person) REQUIRE p.name IS UNIQUE")
    with pytest.raises(kglite.CypherExecutionError, match="unique per"):
        graph.cypher("CREATE CONSTRAINT dup FOR (p:Person) REQUIRE p.age IS UNIQUE")


def test_dropping_an_unknown_constraint_lists_what_exists(graph) -> None:
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE")
    with pytest.raises(kglite.CypherExecutionError) as exc:
        graph.cypher("DROP CONSTRAINT nope")
    assert "Person.name" in str(exc.value)
    # IF EXISTS is a truthful no-op.
    graph.cypher("DROP CONSTRAINT nope IF EXISTS")
    assert graph.last_mutation_stats["constraints_removed"] == 0


def test_dropping_a_node_key_withdraws_both_halves(graph) -> None:
    graph.cypher("CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE p.name IS NODE KEY")
    graph.cypher("DROP CONSTRAINT person_key")
    # Neither half remains: a duplicate and a removal both succeed now.
    graph.cypher("CREATE (p:Person {person_id: 9, name: 'Alice'})")
    graph.cypher("MATCH (p:Person {person_id: 1}) REMOVE p.name")


# ── SHOW CONSTRAINTS ─────────────────────────────────────────────────


def test_show_constraints_columns_and_types(graph) -> None:
    graph.cypher("CREATE CONSTRAINT u FOR (p:Person) REQUIRE p.name IS UNIQUE")
    graph.cypher("CREATE CONSTRAINT e FOR (p:Person) REQUIRE p.age IS NOT NULL")
    rows = _constraint_rows(graph)
    assert set(rows[0]) == {"name", "type", "entityType", "labelsOrTypes", "properties"}
    by_name = {row["name"]: row for row in rows}
    assert by_name["u"]["type"] == "UNIQUENESS"
    assert by_name["e"]["type"] == "NODE_PROPERTY_EXISTENCE"
    assert by_name["u"]["entityType"] == "NODE"
    assert by_name["u"]["labelsOrTypes"] == ["Person"]
    assert by_name["u"]["properties"] == ["name"]


def test_a_node_key_is_reported_as_one_row(graph) -> None:
    graph.cypher("CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE p.name IS NODE KEY")
    rows = _constraint_rows(graph)
    assert len(rows) == 1, rows
    assert rows[0]["type"] == "NODE_KEY"


def test_show_constraints_matches_db_constraints(graph) -> None:
    """One collector, two surfaces — they can never drift."""
    graph.cypher("CREATE CONSTRAINT u FOR (p:Person) REQUIRE p.name IS UNIQUE")
    procedure = graph.cypher(
        "CALL db.constraints() YIELD name, type, entityType, labelsOrTypes, properties "
        "RETURN name, type, entityType, labelsOrTypes, properties"
    ).to_list()
    assert procedure == _constraint_rows(graph)


def test_show_constraints_rejects_yield_and_points_at_the_right_procedure(graph) -> None:
    with pytest.raises(kglite.CypherSyntaxError) as exc:
        graph.cypher("SHOW CONSTRAINTS YIELD name")
    message = str(exc.value)
    assert "db.constraints()" in message, message
    assert "db.indexes()" not in message, message


def test_show_constraints_works_on_a_read_only_graph(graph) -> None:
    """It is a read, so the read-only guard must not block it."""
    graph.cypher("CREATE CONSTRAINT u FOR (p:Person) REQUIRE p.name IS UNIQUE")
    graph.read_only(True)
    assert _names(graph) == {"u"}


# ── guards ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "DROP CONSTRAINT person_name_unique",
    ],
)
def test_constraint_ddl_is_blocked_on_a_read_only_graph(graph, statement: str) -> None:
    graph.read_only(True)
    with pytest.raises(Exception, match="read-only"):
        graph.cypher(statement)


def test_constraint_ddl_respects_the_write_scope(graph) -> None:
    """A constraint is schema state for one node type, so the whitelist applies."""
    with pytest.raises(Exception, match="write scope"):
        graph.cypher(
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
            write_scope=["Other"],
        )
    graph.cypher(
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
        write_scope=["Person"],
    )


def test_schema_lock_rejects_constraining_an_undeclared_property(graph) -> None:
    graph.lock_schema()
    with pytest.raises(kglite.CypherExecutionError) as exc:
        graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.nickname IS UNIQUE")
    message = str(exc.value)
    assert "schema is locked" in message, message
    assert "cannot be constrained" in message, message


@pytest.mark.parametrize(
    "statement",
    [
        "MATCH (n) CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE RETURN 1",
    ],
)
def test_constraint_commands_must_stand_alone(statement: str) -> None:
    with pytest.raises(kglite.CypherSyntaxError):
        _graph().cypher(statement)


# ── persistence ──────────────────────────────────────────────────────


def test_declared_constraints_and_their_names_survive_save_load(tmp_path) -> None:
    g = _graph()
    g.cypher("CREATE CONSTRAINT person_name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE")
    g.cypher("CREATE CONSTRAINT person_age_e FOR (p:Person) REQUIRE p.age IS NOT NULL")
    path = tmp_path / "constraints.kgl"
    g.save(str(path))

    loaded = kglite.load(str(path))
    assert _names(loaded) == {"person_name_unique", "person_age_e"}

    # Both are still enforced. The presence constraint fires first when `age` is
    # omitted, so each half is provoked separately.
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        loaded.cypher("CREATE (p:Person {person_id: 9, name: 'Dana'})")
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        loaded.cypher("CREATE (p:Person {person_id: 9, name: 'Alice', age: 50})")

    # And still droppable by the name they were declared under.
    loaded.cypher("DROP CONSTRAINT person_name_unique")
    assert "person_name_unique" not in _names(loaded)
    loaded.cypher("CREATE (p:Person {person_id: 9, name: 'Alice', age: 50})")


def test_a_dropped_name_does_not_come_back_after_a_reload(tmp_path) -> None:
    """The registry is pruned at save time, so a stale name cannot resurrect."""
    g = _graph()
    g.cypher("CREATE CONSTRAINT person_name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE")
    g.cypher("DROP CONSTRAINT person_name_unique")
    path = tmp_path / "dropped.kgl"
    g.save(str(path))

    loaded = kglite.load(str(path))
    assert _constraint_rows(loaded) == []
    loaded.cypher("CREATE (p:Person {person_id: 9, name: 'Alice'})")


# ── the portability bar ──────────────────────────────────────────────

# A representative vanilla Neo4j 5 constraint-setup script. `None` means the
# statement must succeed; a string is a substring the unsupported-feature message
# must contain. Nothing in this list may raise a syntax error.
NEO4J_CONSTRAINT_SCRIPT: list[tuple[str, str | None]] = [
    # `person_id` is the fixture's id field, so uniqueness on it is refused with
    # the primary-key route rather than silently not enforced.
    (
        "CREATE CONSTRAINT person_id_u IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE",
        "primary_key",
    ),
    ("CREATE CONSTRAINT person_age_u IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS UNIQUE", None),
    ("CREATE CONSTRAINT person_name_e IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS NOT NULL", None),
    ("CREATE CONSTRAINT person_nk IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS NODE KEY", None),
    (
        "CREATE CONSTRAINT person_ck IF NOT EXISTS FOR (p:Person) REQUIRE (p.age, p.city) IS NODE KEY",
        None,
    ),
    # The optional NODE / RELATIONSHIP scope word before UNIQUE.
    ("CREATE CONSTRAINT person_nu IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS NODE UNIQUE", None),
    # Neo4j 4 spellings.
    ("CREATE CONSTRAINT person_id_u4 FOR (p:Person) ASSERT p.age IS UNIQUE", None),
    # Not served, and rejected rather than silently accepted.
    ("CREATE CONSTRAINT person_t IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS :: INTEGER", "is not supported"),
    ("CREATE CONSTRAINT person_t2 IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS TYPED STRING", "is not supported"),
    (
        "CREATE CONSTRAINT knows_u IF NOT EXISTS FOR ()-[r:KNOWS]-() REQUIRE r.since IS UNIQUE",
        "KNOWS",
    ),
    ("SHOW CONSTRAINTS", None),
    ("DROP CONSTRAINT person_id_u IF EXISTS", None),
    ("DROP CONSTRAINT does_not_exist IF EXISTS", None),
]


@pytest.mark.parametrize(
    "statement,expected_error",
    NEO4J_CONSTRAINT_SCRIPT,
    ids=[" ".join(s.split()[:4]) for s, _ in NEO4J_CONSTRAINT_SCRIPT],
)
def test_neo4j_constraint_script_statements_are_all_actionable(statement: str, expected_error: str | None) -> None:
    """Every statement either executes or fails with a specific
    unsupported-feature message. A `CypherSyntaxError` is always a failure: it
    tells a porting user "you typo'd" when the truth is "we lack this"."""
    g = _graph()
    try:
        g.cypher(statement)
    except kglite.CypherSyntaxError as exc:  # pragma: no cover - failure path
        pytest.fail(f"`{statement}` raised a syntax error, not a feature error: {exc}")
    except kglite.CypherExecutionError as exc:
        assert expected_error is not None, f"`{statement}` should have succeeded: {exc}"
        assert expected_error in str(exc), f"`{statement}`: {exc}"
    else:
        assert expected_error is None, f"`{statement}` should have been rejected"


SUPPORTED_FORMS: list[tuple[str, str, str]] = [
    # (declaration, a write it must reject, why)
    (
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE (p:Person {person_id: 9, name: 'Alice'})",
        "duplicate name",
    ),
    (
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NODE UNIQUE",
        "CREATE (p:Person {person_id: 9, name: 'Alice'})",
        "duplicate name under the NODE spelling",
    ),
    (
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL",
        "CREATE (p:Person {person_id: 9})",
        "missing required name",
    ),
    (
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NODE KEY",
        "CREATE (p:Person {person_id: 9})",
        "node key requires presence",
    ),
    (
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE (p.name, p.age) IS UNIQUE",
        "CREATE (p:Person {person_id: 9, name: 'Alice', age: 28})",
        "duplicate composite tuple",
    ),
    (
        "CREATE CONSTRAINT FOR (p:Person) ASSERT p.name IS UNIQUE",
        "CREATE (p:Person {person_id: 9, name: 'Alice'})",
        "Neo4j 4 ASSERT spelling still enforces",
    ),
]


@pytest.mark.parametrize(
    "declaration,violating_write,reason",
    SUPPORTED_FORMS,
    ids=[reason for _, _, reason in SUPPORTED_FORMS],
)
def test_every_declared_constraint_actually_rejects_a_violating_write(
    declaration: str, violating_write: str, reason: str
) -> None:
    """The load-bearing test. A `CREATE CONSTRAINT` that returns cleanly must
    enforce something — a success that enforces nothing is the worst outcome,
    because users build data-integrity assumptions on it."""
    g = _graph()
    g.cypher(declaration)
    assert g.last_mutation_stats["constraints_added"] == 1, declaration
    assert _constraint_rows(g), f"`{declaration}` declared nothing visible"
    with pytest.raises(Exception, match="constraint on"):
        g.cypher(violating_write)
