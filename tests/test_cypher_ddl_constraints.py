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

# Constraint violations raised through Cypher arrive as the typed
# `ConstraintViolationError` (and a failed *declaration* as
# `ConstraintCreationError`): the executor's error channel is
# `Result<_, String>`, but the structured violation is parked on the graph
# alongside the message it produced and re-attached when the binding raises, so
# the type survives the string channel. Message matching below is therefore
# about the *prose* being actionable, not about the type being unavailable —
# the property-type tests assert both.
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


def test_an_unmappable_property_type_is_rejected_not_silently_accepted(graph) -> None:
    """The one outcome worse than an error is a success that enforces nothing.

    The accept-list is closed: a type name with no exact KGLite value
    counterpart is refused by name rather than approximated to a nearby one.
    """
    for declared in ("LIST<INTEGER>", "ZONED DATETIME", "NUMBER"):
        with pytest.raises(kglite.CypherExecutionError) as exc:
            graph.cypher(f"CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: {declared}")
        message = str(exc.value)
        assert "is not supported" in message, message
        # The message names the set that does work, so the statement is fixable
        # without leaving the terminal.
        assert "INTEGER" in message, message
        assert "LOCAL DATETIME" in message, message
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
    """The route the rejection message names must work — under the `id` spelling."""
    graph.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    with pytest.raises(Exception, match="duplicate primary key"):
        graph.cypher("CREATE (p:Person {id: 1})")


def test_the_id_alias_spelling_reaches_the_identity_field_on_create(graph) -> None:
    """The aliased spelling *is* the identity on `CREATE`, so the primary key
    enforces it.

    `add_nodes(unique_id_field="person_id")` makes `person_id` the identity, and
    every read honours that: `MATCH (p:Person {person_id: 1})` finds Alice and
    `p.person_id` resolves through the alias to the identity. Until 0.16.1 the
    write path did not agree — `CREATE (:Person {person_id: 99})` stored 99 as an
    ordinary property beside an engine-minted id, and because the dot read
    resolves the alias to the identity, `p.person_id` then answered with the
    minted id while `properties(p)` still showed 99.

    Now the alias is promoted to the identity, so the id `99` is really the
    node's, and a repeat of it collides with the declared primary key.
    """
    graph.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    graph.cypher("CREATE (p:Person {person_id: 99})")

    ids = sorted(d["id"] for d in graph.cypher("MATCH (p:Person) RETURN p.id AS id").to_dicts())
    assert len(set(ids)) == len(ids), "the engine must not mint a duplicate identity"
    assert ids == [1, 2, 3, 99], "the aliased spelling must land in the identity field"
    # …and the primary key now covers the spelling the type actually uses.
    with pytest.raises(Exception, match="duplicate primary key"):
        graph.cypher("CREATE (p:Person {person_id: 99})")


def test_the_id_alias_and_the_literal_id_spelling_cannot_disagree(graph) -> None:
    """Both spellings name one field, so two different values is a request the
    node cannot satisfy — refused rather than silently resolved one way."""
    with pytest.raises(Exception, match="two different identities"):
        graph.cypher("CREATE (p:Person {person_id: 7, id: 8})")
    # Agreeing values are fine, and store one identity.
    graph.cypher("CREATE (p:Person {person_id: 7, id: 7})")
    assert graph.cypher("MATCH (p:Person {person_id: 7}) RETURN p.id AS id").to_list() == [{"id": 7}]


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
    assert set(rows[0]) == {
        "name",
        "type",
        "entityType",
        "labelsOrTypes",
        "properties",
        "propertyType",
    }
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
        "CALL db.constraints() YIELD name, type, entityType, labelsOrTypes, properties, propertyType "
        "RETURN name, type, entityType, labelsOrTypes, properties, propertyType"
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
    # Property types: declared and enforced for the accepted names, refused by
    # name for the rest.
    ("CREATE CONSTRAINT person_t IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS :: INTEGER", None),
    ("CREATE CONSTRAINT person_t2 IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS TYPED STRING", None),
    ("CREATE CONSTRAINT person_t3 IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS :: LIST<INTEGER>", "is not supported"),
    # A declaration the stored data already violates is refused, like every
    # other kind.
    ("CREATE CONSTRAINT person_t4 IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS :: STRING", "cannot declare"),
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
    except (kglite.CypherExecutionError, kglite.ConstraintError) as exc:
        # A declaration refused against dirty data raises the typed
        # `ConstraintCreationError`, which is an actionable failure too — the
        # bar here is "specific and fixable", not "one exception class".
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


# ── Property-type constraints (IS :: T) ──────────────────────────────


def test_a_declared_property_type_raises_the_typed_error_on_a_violating_write(graph) -> None:
    """The type matters as much as the message: a binding's `except` clause is
    what a user actually writes, and it must catch the same class every other
    constraint raises."""
    graph.cypher("CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    assert graph.last_mutation_stats["constraints_added"] == 1

    for statement in (
        "CREATE (p:Person {person_id: 9, age: 'old'})",
        "MERGE (p:Person {person_id: 10, age: 'old'})",
        "MATCH (p:Person) SET p.age = 'old'",
        "MATCH (p:Person) SET p += {age: 'old'}",
    ):
        with pytest.raises(kglite.ConstraintViolationError) as exc:
            graph.cypher(statement)
        message = str(exc.value)
        assert "INTEGER" in message, f"{statement}: {message}"
        assert "STRING" in message, f"{statement}: {message}"

    # A conforming write is untouched.
    graph.cypher("CREATE (p:Person {person_id: 11, age: 44})")


def test_a_declaration_against_violating_data_raises_the_creation_error() -> None:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"person_id": [1, 2], "name": ["A", "B"], "nickname": ["x", "y"]}),
        "Person",
        "person_id",
        "name",
    )
    with pytest.raises(kglite.ConstraintCreationError, match="cannot declare"):
        g.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.nickname IS :: INTEGER")
    assert _constraint_rows(g) == [], "a refused declaration must install nothing"


def test_null_and_absent_values_satisfy_a_declared_type(graph) -> None:
    """Neo4j semantics: a type constraint is not an existence constraint."""
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    graph.cypher("MATCH (p:Person) SET p.age = null")
    graph.cypher("MATCH (p:Person) REMOVE p.age")
    graph.cypher("CREATE (p:Person {person_id: 12})")


def test_dropping_a_property_type_constraint_restores_the_write(graph) -> None:
    graph.cypher("CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    with pytest.raises(kglite.ConstraintViolationError):
        graph.cypher("MATCH (p:Person) SET p.age = 'old'")

    graph.cypher("DROP CONSTRAINT age_typed")
    assert graph.last_mutation_stats["constraints_removed"] == 1
    assert _constraint_rows(graph) == []
    graph.cypher("MATCH (p:Person) SET p.age = 'old'")


def test_an_unnamed_property_type_constraint_drops_by_its_descriptor(graph) -> None:
    """`SHOW CONSTRAINTS` output must paste straight into `DROP CONSTRAINT`."""
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    assert _names(graph) == {"Person.age"}
    graph.cypher("DROP CONSTRAINT `Person.age`")
    assert _constraint_rows(graph) == []


def test_show_constraints_reports_the_property_type_column(graph) -> None:
    graph.cypher("CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    graph.cypher("CREATE CONSTRAINT name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE")

    rows = {row["name"]: row for row in _constraint_rows(graph)}
    assert rows["age_typed"]["type"] == "NODE_PROPERTY_TYPE"
    assert rows["age_typed"]["propertyType"] == "INTEGER"
    # Present-but-null on every other kind, exactly as in Neo4j 5.
    assert "propertyType" in rows["name_unique"]
    assert rows["name_unique"]["propertyType"] is None


def test_db_constraints_yields_the_same_property_type(graph) -> None:
    """One collector, two surfaces — they cannot be allowed to drift."""
    graph.cypher("CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    yielded = graph.cypher("CALL db.constraints() YIELD name, type, propertyType").to_list()
    assert yielded == [{"name": "age_typed", "type": "NODE_PROPERTY_TYPE", "propertyType": "INTEGER"}]


def test_a_declared_property_type_survives_save_and_load(graph, tmp_path) -> None:
    """The declaration has no second home — if the file loses it, the reloaded
    graph silently stops enforcing."""
    graph.cypher("CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    path = tmp_path / "typed.kgl"
    graph.save(str(path))

    reloaded = kglite.load(str(path))
    rows = {row["name"]: row for row in _constraint_rows(reloaded)}
    assert rows["age_typed"]["propertyType"] == "INTEGER"
    with pytest.raises(kglite.ConstraintViolationError):
        reloaded.cypher("MATCH (p:Person) SET p.age = 'old'")


def test_a_declared_type_beats_the_schema_lock_validation_message() -> None:
    """Two checks cover the same property; the one the user *wrote* wins, so the
    error names their constraint rather than a type they never declared."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"person_id": [1], "name": ["A"], "nickname": ["x"]}),
        "Person",
        "person_id",
        "name",
    )
    g.cypher("CREATE CONSTRAINT nick_typed FOR (p:Person) REQUIRE p.nickname IS :: STRING")
    g.lock_schema()

    with pytest.raises(kglite.ConstraintViolationError) as exc:
        g.cypher("MATCH (p:Person) SET p.nickname = 7")
    message = str(exc.value)
    assert "PROPERTY TYPE constraint" in message, message
    assert "schema is locked" not in message, message
    # And the declaration's own verdict is what applies to an accepted value.
    g.cypher("MATCH (p:Person) SET p.nickname = 'nick'")


def test_a_declared_type_does_not_exempt_an_unknown_property_from_the_lock() -> None:
    """The exemption skips the schema lock's *type* verdict, never its typo guard.

    A type constraint can be declared on a property no node holds (nothing
    violates it yet), which leaves the property absent from the observed schema
    the lock validates against. The exemption then waved the write through, so a
    locked graph accepted a property it does not know — the exact write
    `lock_schema()` exists to refuse.
    """
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"person_id": [1], "name": ["A"], "age": [30]}),
        "Person",
        "person_id",
        "name",
    )
    # No Person holds `nickname`, so this installs cleanly.
    g.cypher("CREATE CONSTRAINT nick_typed FOR (p:Person) REQUIRE p.nickname IS :: INTEGER")
    g.lock_schema()

    with pytest.raises(kglite.CypherExecutionError) as exc:
        g.cypher("MATCH (p:Person) SET p.nickname = 7")
    message = str(exc.value)
    assert "Unknown property 'nickname'" in message, message

    # Control: an ordinary typo hits the same guard, so the assertion above is
    # not passing for some unrelated reason.
    with pytest.raises(kglite.CypherExecutionError, match="Unknown property 'nickanme'"):
        g.cypher("MATCH (p:Person) SET p.nickanme = 7")

    # And a property the schema DOES know keeps the declaration's typed verdict
    # (the exemption's whole purpose) — see the test above for the full case.
    with pytest.raises(kglite.ConstraintViolationError):
        g.cypher("CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER")
        g.cypher("MATCH (p:Person) SET p.age = 'old'")

    # The CREATE path carries no such exemption and never did; pinned here so
    # the two write paths cannot drift into disagreeing about the same graph.
    with pytest.raises(kglite.KgError, match="Unknown property 'nickname'"):
        g.cypher("CREATE (:Person {person_id: 2, name: 'B', nickname: 7})")


def test_describe_annotates_a_declared_property_type(graph) -> None:
    """An agent planning a write should see the requirement before attempting
    it, which is what the describe() annotation is for."""
    assert "declared_type" not in graph.describe()
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    assert 'declared_type="INTEGER"' in graph.describe()


def test_a_bulk_load_is_gated_by_a_declared_type(graph) -> None:
    """`add_nodes` never touches the Cypher executor, so it is its own choke
    point — a constraint the bulk path bypassed would be theatre."""
    graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    # Conforming first: a refused load leaves the observed column-type metadata
    # behind (pre-existing `add_nodes` behaviour, unrelated to constraints), and
    # ordering around it keeps this test about the gate rather than that.
    graph.add_nodes(
        pd.DataFrame({"person_id": [51], "name": ["Y"], "age": [61]}),
        "Person",
        "person_id",
        "name",
        columns=["age"],
    )
    with pytest.raises(kglite.ConstraintViolationError, match="INTEGER"):
        graph.add_nodes(
            pd.DataFrame({"person_id": [50], "name": ["Z"], "age": ["old"]}),
            "Person",
            "person_id",
            "name",
            columns=["age"],
        )


# ── structural fields ────────────────────────────────────────────────


@pytest.mark.parametrize("populated", [False, True], ids=["empty-label", "populated-label"])
@pytest.mark.parametrize(
    ("field", "declared", "accepted"),
    [
        # `type` reads as the node's primary type — always a string, and never a
        # stored property, so the write path cannot see it at all.
        ("type", "STRING", True),
        ("type", "INTEGER", False),
        ("type", "DATE", False),
        # `id` is the structural identity: always an integer.
        ("id", "INTEGER", True),
        ("id", "STRING", False),
        # `title` is the structural title: always a string.
        ("title", "STRING", True),
        ("title", "INTEGER", False),
    ],
)
def test_a_type_constraint_on_a_structural_field_is_decided_structurally(
    field: str, declared: str, accepted: bool, populated: bool
) -> None:
    """A structural field's type is fixed by the data model, so the verdict is
    the same with or without rows — it must never be read off scanned data.

    Scanning made an empty label say yes to anything: `p.type IS :: INTEGER`
    installed, reported success, and then enforced nothing, because the label is
    not a stored property the write path checks. The mirror failure is a
    declaration that *can* be seen but can never be satisfied — `p.id IS ::
    STRING` installed and then refused every subsequent write, bricking the
    node type.
    """
    g = kglite.KnowledgeGraph()
    if populated:
        g.add_nodes(pd.DataFrame({"id": [1], "title": ["a"]}), "Person", "id", "title")
    statement = f"CREATE CONSTRAINT c FOR (p:Person) REQUIRE p.{field} IS :: {declared}"

    if accepted:
        g.cypher(statement)
        # An accepted declaration is one the field always satisfies, so the node
        # type stays writable — the constraint is true, not merely installed.
        g.cypher("CREATE (:Person {id: 99, title: 'z'})")
        assert g.cypher("MATCH (p:Person) RETURN count(p) AS c").to_dicts()[0]["c"] >= 1
    else:
        with pytest.raises(kglite.KgError) as exc:
            g.cypher(statement)
        message = str(exc.value)
        assert "structural" in message, message
        assert declared in message, message
        # Nothing was installed, so the node type is untouched and writable.
        assert g.cypher("CALL db.constraints() YIELD name RETURN name").to_dicts() == []
        g.cypher("CREATE (:Person {id: 99, title: 'z'})")


def test_a_type_constraint_on_an_aliased_structural_field_resolves_through_the_alias() -> None:
    """`add_nodes(df, 'Person', 'pid', 'pname')` maps `pid`/`pname` onto the
    structural id/title, so a declaration on the alias must get the structural
    verdict rather than the stored-property one."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"pid": [1], "pname": ["a"]}), "Person", "pid", "pname")

    with pytest.raises(kglite.KgError, match="structural"):
        g.cypher("CREATE CONSTRAINT c1 FOR (p:Person) REQUIRE p.pid IS :: STRING")
    with pytest.raises(kglite.KgError, match="structural"):
        g.cypher("CREATE CONSTRAINT c2 FOR (p:Person) REQUIRE p.pname IS :: INTEGER")

    # The satisfiable spellings still install.
    g.cypher("CREATE CONSTRAINT c3 FOR (p:Person) REQUIRE p.pid IS :: INTEGER")
    g.cypher("CREATE CONSTRAINT c4 FOR (p:Person) REQUIRE p.pname IS :: STRING")


def test_an_ordinary_property_still_gets_the_data_scan() -> None:
    """The structural rule must not swallow the pre-existing-data refusal that
    protects ordinary stored properties."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1], "title": ["a"], "age": ["thirty"]}), "Person", "id", "title")
    with pytest.raises(kglite.ConstraintCreationError, match="existing"):
        g.cypher("CREATE CONSTRAINT c FOR (p:Person) REQUIRE p.age IS :: INTEGER")
