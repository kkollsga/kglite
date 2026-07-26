"""SQLite export round-trip: export → ingest with stdlib sqlite3 → verify data.

The acceptance test for the no-lock-in exit. It does not check that "a file
appeared" — it ingests the dump with Python's bundled ``sqlite3`` (no new
dependency anywhere) and asserts per-type node counts, per-link-table edge
counts, specific property *values*, SQLite's own view of each value's type, and
how a missing property is represented.

Type fidelity is the part most easily broken by a careless exporter: integers
must not arrive as strings, floats must keep full precision, booleans must be
storable, and a property one node lacks must be NULL rather than an empty
string (which would be indistinguishable from a genuine empty value).
"""

from __future__ import annotations

import sqlite3

import pytest

import kglite


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    """Two node types, edges with properties, and deliberate gaps.

    ``Person`` carries an integer, a float needing full precision, a boolean, a
    string with a SQL-hostile apostrophe, and — on one node only — an ``email``,
    so the other node's ``email`` must come back NULL.
    """
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Person {id: 1, title: 'Ada', age: 36, "
        "ratio: 0.1234567890123456, active: true, email: 'ada@example.com'})"
    )
    # No email, no ratio, and active is explicitly false (not missing).
    g.cypher('CREATE (:Person {id: 2, title: "O\'Brien", age: 41, active: false})')
    g.cypher("CREATE (:Company {id: 10, title: 'Acme', founded: 1999})")
    g.cypher("CREATE (:Company {id: 11, title: 'Globex', founded: 2004})")
    g.cypher("MATCH (p:Person {id: 1}), (c:Company {id: 10}) CREATE (p)-[:WORKS_AT {since: 2019, fte: 0.8}]->(c)")
    g.cypher("MATCH (p:Person {id: 2}), (c:Company {id: 11}) CREATE (p)-[:WORKS_AT {since: 2021}]->(c)")
    g.cypher("MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:KNOWS]->(b)")
    return g


def ingest(graph: kglite.KnowledgeGraph, tmp_path) -> sqlite3.Connection:
    """Export the graph and load the dump into a real SQLite database."""
    dump = tmp_path / "dump.sql"
    graph.export(str(dump))  # .sql infers format="sqlite"
    connection = sqlite3.connect(":memory:")
    connection.executescript(dump.read_text(encoding="utf-8"))
    return connection


def test_dump_ingests_into_sqlite_with_expected_tables(graph, tmp_path):
    connection = ingest(graph, tmp_path)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # One table per node type, one link table per connection type.
    assert tables == {"Person", "Company", "WORKS_AT", "KNOWS"}


def test_node_counts_per_type_round_trip(graph, tmp_path):
    connection = ingest(graph, tmp_path)
    assert connection.execute("SELECT count(*) FROM Person").fetchone()[0] == 2
    assert connection.execute("SELECT count(*) FROM Company").fetchone()[0] == 2


def test_edge_counts_per_link_table_round_trip(graph, tmp_path):
    connection = ingest(graph, tmp_path)
    assert connection.execute("SELECT count(*) FROM WORKS_AT").fetchone()[0] == 2
    assert connection.execute("SELECT count(*) FROM KNOWS").fetchone()[0] == 1


def test_scalar_values_and_types_survive(graph, tmp_path):
    connection = ingest(graph, tmp_path)
    row = connection.execute(
        "SELECT id, title, age, ratio, active, email, "
        "typeof(id), typeof(age), typeof(ratio), typeof(active), typeof(title) "
        "FROM Person WHERE id = 1"
    ).fetchone()

    assert row[0] == 1
    assert row[1] == "Ada"
    assert row[2] == 36
    # Full double precision, not a rounded or reformatted value.
    assert row[3] == 0.1234567890123456
    assert row[4] == 1, "booleans become SQLite integers 0/1"
    assert row[5] == "ada@example.com"

    # SQLite's own view of the storage class: integers did not arrive as text
    # and the float did not collapse to an integer.
    assert row[6] == "integer", "id must not become a string"
    assert row[7] == "integer", "age must not become a string"
    assert row[8] == "real", "ratio must stay a float"
    assert row[9] == "integer"
    assert row[10] == "text"


def test_sql_hostile_string_survives_verbatim(graph, tmp_path):
    connection = ingest(graph, tmp_path)
    title = connection.execute("SELECT title FROM Person WHERE id = 2").fetchone()[0]
    assert title == "O'Brien", "the apostrophe must be escaped, not dropped or doubled"


def test_missing_properties_become_null_not_empty_string(graph, tmp_path):
    connection = ingest(graph, tmp_path)
    email, ratio = connection.execute("SELECT email, ratio FROM Person WHERE id = 2").fetchone()
    assert email is None, "a property this node never had must be NULL"
    assert ratio is None

    # And a genuinely false boolean is 0, distinguishable from a missing one.
    active = connection.execute("SELECT active FROM Person WHERE id = 2").fetchone()[0]
    assert active == 0

    # SQL agrees: exactly one Person row has a NULL email.
    missing = connection.execute("SELECT count(*) FROM Person WHERE email IS NULL").fetchone()[0]
    assert missing == 1


def test_edge_properties_and_endpoints_round_trip(graph, tmp_path):
    connection = ingest(graph, tmp_path)
    rows = connection.execute(
        "SELECT source_type, source_id, target_type, target_id, since, fte FROM WORKS_AT ORDER BY source_id"
    ).fetchall()
    assert rows == [
        ("Person", "1", "Company", "10", 2019, 0.8),
        # The second edge has no `fte`, so that column is NULL for it.
        ("Person", "2", "Company", "11", 2021, None),
    ]


def test_link_tables_join_back_to_node_tables(graph, tmp_path):
    """The exported schema is actually usable as a relational graph."""
    connection = ingest(graph, tmp_path)
    pairs = connection.execute(
        "SELECT p.title, c.title FROM WORKS_AT w "
        "JOIN Person p ON p.id = w.source_id "
        "JOIN Company c ON c.id = w.target_id "
        "ORDER BY p.title"
    ).fetchall()
    assert pairs == [("Ada", "Acme"), ("O'Brien", "Globex")]


def test_export_is_deterministic(graph, tmp_path):
    first = graph.export_string("sqlite")
    second = graph.export_string("sqlite")
    assert first == second, "the same graph must always produce byte-identical SQL"


def test_unknown_export_format_names_sqlite(graph):
    with pytest.raises(ValueError, match="sqlite"):
        graph.export_string("nonsense")


def test_export_survives_a_kgl_save_load_cycle(graph, tmp_path):
    """A reloaded graph exports the same SQL as the original.

    This is the cross-check that the export reads through the canonical node
    projection: after a save/load the properties may live in a different
    internal store, and an exporter reading the wrong path would silently emit
    empty tables.
    """
    before = graph.export_string("sqlite")
    path = tmp_path / "g.kgl"
    graph.save(str(path))
    reloaded = kglite.load(str(path))
    assert reloaded.export_string("sqlite") == before
