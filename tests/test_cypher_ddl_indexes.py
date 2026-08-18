"""Cypher index DDL — `CREATE INDEX` / `DROP INDEX` / `SHOW INDEXES`.

The acceptance bar for this surface is *portability*: a vanilla Neo4j 5
schema-setup script must run against KGLite and either take effect or fail
with a specific unsupported-feature message. Never a syntax error, and never
a no-op that reports success.

`test_neo4j_schema_script_statements_are_all_actionable` is the load-bearing
test — it walks a representative Neo4j schema script statement by statement
and asserts that outcome for every one.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


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
    g.add_connections(
        pd.DataFrame({"from_id": [1, 2], "to_id": [2, 3], "since": [2020, 2019]}),
        "KNOWS",
        "Person",
        "from_id",
        "Person",
        "to_id",
    )
    return g


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    return _graph()


def _index_rows(g: kglite.KnowledgeGraph) -> list[dict]:
    return g.cypher("SHOW INDEXES").to_list()


def _names(g: kglite.KnowledgeGraph) -> set[str]:
    return {row["name"] for row in _index_rows(g)}


# ── CREATE INDEX ─────────────────────────────────────────────────────


def test_create_index_makes_the_property_indexed(graph) -> None:
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
    assert graph.has_index("Person", "city")
    assert graph.last_mutation_stats["indexes_added"] == 1
    assert "Person.city" in _names(graph)


def test_create_index_matches_the_python_api_result(graph) -> None:
    """DDL must route to the same machinery as `create_index`, not a parallel
    implementation — so the two must be indistinguishable afterwards."""
    other = _graph()
    other.create_index("Person", "city")
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
    assert _index_rows(graph) == _index_rows(other)
    assert graph.index_stats("Person", "city") == other.index_stats("Person", "city")


def test_composite_create_index_builds_a_composite_index(graph) -> None:
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city, p.age)")
    assert graph.has_composite_index("Person", ["city", "age"])
    assert "Person.(city,age)" in _names(graph)


def test_range_index_builds_both_structures(graph) -> None:
    """Neo4j's RANGE index serves equality *and* range, which in KGLite takes
    the hash index plus the B-tree — the documented two-structure mapping."""
    graph.cypher("CREATE RANGE INDEX person_age FOR (p:Person) ON (p.age)")
    assert graph.last_mutation_stats["indexes_added"] == 2
    assert graph.has_index("Person", "age")
    kinds = {row["type"] for row in _index_rows(graph) if row["name"] == "Person.age"}
    assert kinds == {"PROPERTY", "RANGE"}


def test_bare_create_index_does_not_build_the_btree(graph) -> None:
    """The bare form is deliberately equality-only: building both for every
    ported `CREATE INDEX` would silently double index memory."""
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.age)")
    assert {row["type"] for row in _index_rows(graph)} == {"PROPERTY"}


def test_if_not_exists_is_idempotent_and_the_bare_form_is_not(graph) -> None:
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")

    graph.cypher("CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.city)")
    assert graph.last_mutation_stats["indexes_added"] == 0

    with pytest.raises(kglite.CypherExecutionError, match="already exists"):
        graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")


def test_created_index_is_populated_and_the_lookup_still_answers(graph) -> None:
    """The point of the statement: a real, populated index, and queries that
    still return the right rows through it."""
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
    assert graph.index_stats("Person", "city") == {
        "node_type": "Person",
        "property": "city",
        "unique_values": 2,
        "total_entries": 3,
        "avg_entries_per_value": 1.5,
    }
    rows = graph.cypher("MATCH (p:Person {city: 'Oslo'}) RETURN p.name AS n").to_list()
    assert sorted(r["n"] for r in rows) == ["Alice", "Charlie"]


# ── Index content stays in step with writes ──────────────────────────


def _aliased_graph() -> kglite.KnowledgeGraph:
    """`Term` names its identity columns itself: `term_id` is the id field and
    `term_name` the title field, so neither lives in the property map — an
    index on either spelling is built from the node's id / title, which is also
    what a `MATCH` on that name compares against."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "term_id": [1, 2, 3],
                "term_name": ["term-1", "term-2", "term-3"],
                "city": ["Oslo", "Bergen", "Oslo"],
            }
        ),
        "Term",
        "term_id",
        "term_name",
    )
    return g


def _matched(g: kglite.KnowledgeGraph, predicate: str) -> int:
    """Row counts for the indexed spelling and the scanning spelling of one
    predicate. An index that has drifted from the data makes them disagree."""
    inline = g.cypher(f"MATCH (t:Term {{{predicate}}}) RETURN t").to_list()
    where_key, where_value = predicate.split(": ", 1)
    scanned = g.cypher(f"MATCH (t:Term) WHERE t.{where_key} = {where_value} RETURN t").to_list()
    assert len(inline) == len(scanned), (
        f"the indexed and scanning spellings of `{predicate}` disagree: {len(inline)} vs {len(scanned)}"
    )
    return len(inline)


@pytest.mark.parametrize("indexed_property", ["city", "term_name"])
def test_a_cypher_create_keeps_the_index_in_step_with_the_scan(
    indexed_property: str,
) -> None:
    """Incremental maintenance has to file a written node the way a rebuild
    would, or the index answers with rows a scan cannot produce."""
    g = _aliased_graph()
    g.cypher(f"CREATE INDEX FOR (t:Term) ON (t.{indexed_property})")
    g.cypher("CREATE (:Term {term_id: 99, term_name: 'created-x', city: 'Oslo'})")

    _matched(g, "term_name: 'created-x'")
    _matched(g, "term_name: 'term-1'")
    assert _matched(g, "city: 'Oslo'") == 3


def test_a_cypher_set_keeps_the_index_in_step_with_the_scan() -> None:
    g = _aliased_graph()
    g.cypher("CREATE INDEX FOR (t:Term) ON (t.term_name)")
    g.cypher("CREATE INDEX FOR (t:Term) ON (t.city)")
    g.cypher("MATCH (t:Term {term_id: 1}) SET t.city = 'Trondheim'")
    g.cypher("MATCH (t:Term {term_id: 2}) SET t.term_name = 'renamed'")

    assert _matched(g, "city: 'Trondheim'") == 1
    assert _matched(g, "city: 'Oslo'") == 1
    _matched(g, "term_name: 'renamed'")
    _matched(g, "term_name: 'term-2'")


def test_a_range_index_does_not_double_count_a_created_node() -> None:
    """A RANGE index installs both structures on one property, so the write
    path sees the same key twice — the node must still join each bucket once."""
    g = _aliased_graph()
    g.cypher("CREATE RANGE INDEX term_city FOR (t:Term) ON (t.city)")
    g.cypher("CREATE (:Term {term_id: 99, term_name: 'Dana', city: 'Tromso'})")

    assert _matched(g, "city: 'Tromso'") == 1
    assert len(g.cypher("MATCH (t:Term) WHERE t.city > 'Trom' RETURN t").to_list()) == 1


# ── SHOW INDEXES ─────────────────────────────────────────────────────


def test_show_indexes_columns_match_db_indexes(graph) -> None:
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
    show = graph.cypher("SHOW INDEXES")
    assert show.columns == [
        "name",
        "type",
        "entityType",
        "labelsOrTypes",
        "properties",
        "state",
    ]
    call = graph.cypher("CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state")
    assert show.to_list() == call.to_list()


def test_show_indexes_is_a_read_not_a_mutation(graph) -> None:
    graph.read_only(True)
    assert graph.cypher("SHOW INDEXES").to_list() == []
    graph.read_only(False)


@pytest.mark.parametrize("statement", ["SHOW INDEXES", "SHOW INDEX", "SHOW ALL INDEXES"])
def test_show_indexes_spellings(graph, statement) -> None:
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
    assert len(graph.cypher(statement).to_list()) == 1


def test_show_indexes_modifiers_point_at_db_indexes(graph) -> None:
    with pytest.raises(kglite.CypherSyntaxError, match=r"db\.indexes\(\)"):
        graph.cypher("SHOW INDEXES YIELD name WHERE name = 'x'")


# ── DROP INDEX ───────────────────────────────────────────────────────


def test_drop_index_accepts_the_canonical_name_unquoted(graph) -> None:
    """`SHOW INDEXES` output must be pastable into `DROP INDEX` — the dotted
    canonical name works without backticks."""
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
    graph.cypher("DROP INDEX Person.city")
    assert not graph.has_index("Person", "city")
    assert graph.last_mutation_stats["indexes_removed"] == 1


def test_drop_index_accepts_a_backticked_name(graph) -> None:
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
    graph.cypher("DROP INDEX `Person.city`")
    assert not graph.has_index("Person", "city")


def test_drop_index_accepts_the_composite_canonical_name(graph) -> None:
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city, p.age)")
    graph.cypher("DROP INDEX Person.(city,age)")
    assert not graph.has_composite_index("Person", ["city", "age"])


def test_drop_index_by_descriptor_needs_no_name(graph) -> None:
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city, p.age)")
    graph.cypher("DROP INDEX FOR (p:Person) ON (p.city, p.age)")
    assert not graph.has_composite_index("Person", ["city", "age"])


def test_drop_index_removes_both_structures_of_a_range_index(graph) -> None:
    graph.cypher("CREATE RANGE INDEX FOR (p:Person) ON (p.age)")
    graph.cypher("DROP INDEX Person.age")
    assert graph.last_mutation_stats["indexes_removed"] == 2
    assert _index_rows(graph) == []


def test_dropping_a_neo4j_style_name_explains_the_naming_rule(graph) -> None:
    """A name given to `CREATE INDEX` is not stored. Failing loudly beats
    letting the caller believe the drop worked."""
    graph.cypher("CREATE INDEX person_city FOR (p:Person) ON (p.city)")
    assert _names(graph) == {"Person.city"}

    with pytest.raises(kglite.CypherExecutionError) as excinfo:
        graph.cypher("DROP INDEX person_city")
    message = str(excinfo.value)
    assert "canonical" in message
    assert "Person.city" in message

    # IF EXISTS is a no-op: there genuinely is no index under that name.
    graph.cypher("DROP INDEX person_city IF EXISTS")
    assert graph.has_index("Person", "city")


def test_neo4j_3_drop_syntax_is_named_as_such(graph) -> None:
    with pytest.raises(kglite.CypherSyntaxError, match="Neo4j 3.x"):
        graph.cypher("DROP INDEX ON :Person(city)")


# ── Guards: read-only, schema lock, standalone-statement ─────────────


def test_index_ddl_is_blocked_on_a_read_only_graph(graph) -> None:
    graph.read_only(True)
    for statement in (
        "CREATE INDEX FOR (p:Person) ON (p.city)",
        "DROP INDEX Person.city",
    ):
        with pytest.raises(kglite.CypherExecutionError, match="read-only mode"):
            graph.cypher(statement)
    graph.read_only(False)
    assert not graph.has_index("Person", "city")


def test_index_ddl_is_blocked_in_a_read_only_transaction(graph) -> None:
    with graph.begin_read() as tx:
        assert tx.is_read_only
        with pytest.raises(Exception, match="read"):
            tx.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
    assert not graph.has_index("Person", "city")


def test_schema_lock_rejects_indexing_an_undeclared_property(graph) -> None:
    graph.lock_schema()
    try:
        with pytest.raises(kglite.CypherExecutionError, match="schema is locked"):
            graph.cypher("CREATE INDEX FOR (p:Person) ON (p.nickname)")
        # A declared property still indexes under the lock.
        graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
        assert graph.has_index("Person", "city")
    finally:
        graph.unlock_schema()


@pytest.mark.parametrize(
    "statement",
    [
        "MATCH (p:Person) CREATE INDEX FOR (p:Person) ON (p.city)",
        "CREATE INDEX FOR (p:Person) ON (p.city) RETURN 1",
        "CALL { CREATE INDEX FOR (p:Person) ON (p.city) RETURN 1 } RETURN 1",
    ],
)
def test_schema_commands_must_stand_alone(graph, statement) -> None:
    with pytest.raises(kglite.CypherSyntaxError):
        graph.cypher(statement)
    assert not graph.has_index("Person", "city")


def test_failed_ddl_leaves_no_partial_index(graph) -> None:
    """DDL is a mutation, so a rejected statement rolls back like any other."""
    with pytest.raises(kglite.CypherExecutionError, match="OPTIONS"):
        graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city) OPTIONS {indexProvider: 'x'}")
    assert not graph.has_index("Person", "city")
    assert _index_rows(graph) == []


# ── Persistence ──────────────────────────────────────────────────────


def test_ddl_created_indexes_survive_save_and_load(graph, tmp_path) -> None:
    graph.cypher("CREATE RANGE INDEX FOR (p:Person) ON (p.age)")
    graph.cypher("CREATE INDEX FOR (p:Person) ON (p.city, p.age)")
    before = _index_rows(graph)

    path = tmp_path / "ddl.kgl"
    graph.save(str(path))
    reloaded = kglite.load(str(path))
    assert _index_rows(reloaded) == before


# ── The portability bar ──────────────────────────────────────────────

# A representative vanilla Neo4j 5 schema-setup script. `None` means the
# statement must succeed; a string is a substring the unsupported-feature
# message must contain. Nothing in this list may raise a syntax error.
NEO4J_SCHEMA_SCRIPT: list[tuple[str, str | None]] = [
    ("CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)", None),
    ("CREATE INDEX FOR (p:Person) ON (p.city)", None),
    ("CREATE INDEX person_city_age IF NOT EXISTS FOR (p:Person) ON (p.city, p.age)", None),
    ("CREATE RANGE INDEX person_age IF NOT EXISTS FOR (p:Person) ON (p.age)", None),
    ("SHOW INDEXES", None),
    ("CREATE TEXT INDEX pn_text IF NOT EXISTS FOR (p:Person) ON (p.name)", "TEXT INDEX"),
    (
        "CREATE FULLTEXT INDEX p_ft IF NOT EXISTS FOR (p:Person) ON EACH [p.name, p.city]",
        "FULLTEXT INDEX",
    ),
    ("CREATE POINT INDEX p_loc IF NOT EXISTS FOR (p:Person) ON (p.location)", "POINT INDEX"),
    (
        "CREATE VECTOR INDEX p_emb IF NOT EXISTS FOR (p:Person) ON (p.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 128}}",
        "build_vector_index",
    ),
    ("CREATE LOOKUP INDEX node_labels IF NOT EXISTS FOR (n) ON EACH labels(n)", "LOOKUP INDEX"),
    ("CREATE INDEX knows_since IF NOT EXISTS FOR ()-[r:KNOWS]-() ON (r.since)", "KNOWS"),
    # Constraint DDL is served as of Sprint 4b: uniqueness, presence, and node
    # keys all route to real enforcement. `person_id` is unique across the
    # fixture and `name` is present on every row, so each declaration holds.
    # `person_id` is this fixture's unique_id_field, so it resolves to the
    # structural `id`. Uniqueness DDL over `id` is refused and points at the
    # primary-key route, rather than declaring a constraint that would admit
    # duplicates while reporting success.
    (
        "CREATE CONSTRAINT person_id_u IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE",
        "primary_key",
    ),
    (
        "CREATE CONSTRAINT person_city_u IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
        None,
    ),
    (
        "CREATE CONSTRAINT person_name_e IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS NOT NULL",
        None,
    ),
    (
        "CREATE CONSTRAINT person_key IF NOT EXISTS FOR (p:Person) REQUIRE (p.name, p.age) IS NODE KEY",
        None,
    ),
    # Property-type constraints are declared and enforced; a type name with no
    # exact value counterpart is still rejected rather than approximated.
    (
        "CREATE CONSTRAINT person_typed IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS :: INTEGER",
        None,
    ),
    (
        "CREATE CONSTRAINT person_listed IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS :: LIST<INTEGER>",
        "is not supported",
    ),
    # Neo4j 4 spelled REQUIRE as ASSERT; a 4.x-era script must reach the same
    # enforcement rather than a parse error.
    (
        "CREATE CONSTRAINT person_id_u4 IF NOT EXISTS FOR (p:Person) ASSERT p.name IS UNIQUE",
        None,
    ),
    ("SHOW CONSTRAINTS", None),
    ("DROP CONSTRAINT person_id_u IF EXISTS", None),
    ("DROP INDEX Person.city IF EXISTS", None),
    ("DROP INDEX FOR (p:Person) ON (p.city, p.age)", None),
]


@pytest.mark.parametrize(
    "statement,expected_error",
    NEO4J_SCHEMA_SCRIPT,
    ids=[s.split()[:4] and " ".join(s.split()[:4]) for s, _ in NEO4J_SCHEMA_SCRIPT],
)
def test_neo4j_schema_script_statements_are_all_actionable(statement: str, expected_error: str | None) -> None:
    """Every statement either executes or fails with a specific
    unsupported-feature message. A `CypherSyntaxError` is always a failure:
    it tells a porting user "you typo'd" when the truth is "we lack this"."""
    g = _graph()
    # Preload the state the DROP statements at the tail expect.
    if statement.startswith("DROP INDEX"):
        g.cypher("CREATE INDEX FOR (p:Person) ON (p.city)")
        g.cypher("CREATE INDEX FOR (p:Person) ON (p.city, p.age)")

    try:
        g.cypher(statement)
    except kglite.CypherSyntaxError as exc:  # pragma: no cover - failure path
        pytest.fail(f"`{statement}` raised a syntax error, not a feature error: {exc}")
    except kglite.CypherExecutionError as exc:
        assert expected_error is not None, f"`{statement}` should have succeeded: {exc}"
        assert expected_error in str(exc), f"`{statement}`: {exc}"
    else:
        assert expected_error is None, f"`{statement}` should have been rejected"


# ── Disk mode: DDL must make the same backend decision as the API ────


def _disk_graph(path: str) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph(storage="disk", path=path)
    g.add_nodes(
        pd.DataFrame(
            {
                "nid": [f"Q{i}" for i in range(1, 6)],
                "label": ["Norway", "Sweden", "Denmark", "Finland", "Iceland"],
            }
        ),
        "Country",
        "nid",
        "label",
    )
    return g


def test_disk_ddl_builds_the_persistent_index_not_the_heap_map(tmp_path) -> None:
    """On a disk graph the equality index must be the mmap-backed one.

    `DirGraph::create_index` builds an in-memory HashMap that would need
    multiple GB of heap for a large type — which is why `create_index` routes
    to the persistent index on disk. Cypher DDL has to make the same call, so
    the two are compared directly here.
    """
    via_api = _disk_graph(str(tmp_path / "api"))
    api_info = via_api.create_index("Country", "label")
    assert api_info["persistent"] is True

    via_ddl = _disk_graph(str(tmp_path / "ddl"))
    via_ddl.cypher("CREATE INDEX FOR (c:Country) ON (c.label)")

    # The persistent index is visible to `has_index`; the heap map is not
    # populated on either path.
    assert via_ddl.has_index("Country", "label") == via_api.has_index("Country", "label")
    assert via_ddl.list_indexes() == via_api.list_indexes()
    assert (
        via_ddl.cypher("MATCH (c:Country {label: 'Norway'}) RETURN c.nid AS nid").to_list()
        == via_api.cypher("MATCH (c:Country {label: 'Norway'}) RETURN c.nid AS nid").to_list()
    )


def test_disk_ddl_if_not_exists_sees_the_persistent_index(tmp_path) -> None:
    """The duplicate check must consult `has_any_index`, or a disk graph would
    silently rebuild the index on every `IF NOT EXISTS` statement."""
    g = _disk_graph(str(tmp_path / "g"))
    g.create_index("Country", "label")
    g.cypher("CREATE INDEX IF NOT EXISTS FOR (c:Country) ON (c.label)")
    assert g.last_mutation_stats["indexes_added"] == 0
    with pytest.raises(kglite.CypherExecutionError, match="already exists"):
        g.cypher("CREATE INDEX FOR (c:Country) ON (c.label)")


def test_disk_ddl_refuses_to_claim_a_zero_entry_index(tmp_path) -> None:
    """A disk graph's persistent index covers string columns only; a numeric
    property yields zero entries. Reporting success there would leave the
    caller believing their lookups are indexed."""
    g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    g.add_nodes(
        pd.DataFrame({"nid": ["Q1", "Q2"], "label": ["Norway", "Sweden"], "pop": [5, 10]}),
        "Country",
        "nid",
        "label",
    )
    with pytest.raises(kglite.CypherExecutionError, match="indexed no values"):
        g.cypher("CREATE INDEX FOR (c:Country) ON (c.pop)")

    # The string column still indexes. `has_index` reads the in-memory map
    # only, so probe the persistent index the disk-aware way: a second
    # create reports `created=False` (it consults `has_any_index`).
    g.cypher("CREATE INDEX FOR (c:Country) ON (c.label)")
    assert g.create_index("Country", "label")["created"] is False
    assert g.cypher("MATCH (c:Country {label: 'Norway'}) RETURN c.nid AS nid").to_list() == [{"nid": "Q1"}]


def test_disk_ddl_allows_an_empty_type(tmp_path) -> None:
    """Zero entries is legitimate when the type has no nodes — the emptiness
    check, not the count, gates the error."""
    g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    g.add_nodes(pd.DataFrame({"nid": ["Q1"], "label": ["Norway"]}), "Country", "nid", "label")
    g.cypher("MATCH (c:Country) DETACH DELETE c")
    g.cypher("CREATE INDEX FOR (c:Country) ON (c.label)")
