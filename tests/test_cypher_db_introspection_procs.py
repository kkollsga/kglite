"""db.* schema-introspection Cypher procedures.

Procedures every binding can call via cypher_query:

  CALL db.graph_stats() YIELD node_count, edge_count, label_count, relationship_type_count
  CALL db.property_stats(node_type, property) YIELD value_count, null_count, distinct_count
  CALL db.property_uniqueness(node_type, property) YIELD is_unique, violation_count, distinct_count
  CALL db.propertyKeys() YIELD propertyKey
  CALL db.schema() YIELD nodeType, properties

The first three (2026-05-25 Batch 6) answer "how big / how unique"; the last
two (2026-06-25) make property keys + the per-type schema reachable from
Cypher itself, not just the Python describe() path.

Real use case: agent's first "what's in this graph?" query, and
pre-flight before declaring a uniqueness constraint.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def small_graph():
    g = kglite.KnowledgeGraph()
    # `title` is the natural-key field (the 4th add_nodes arg) — it's
    # auto-uniqued by the graph and not stored as a regular property.
    # `city` and `email` ARE stored as regular properties — those are
    # what db.property_stats sees.
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["alice_1", "bob", "alice_2", "diana"],
                "city": ["Oslo", "Oslo", "Oslo", "Bergen"],
                "email": ["a@x.com", "b@x.com", "c@x.com", None],
            }
        ),
        "Person",
        "id",
        "name",
    )
    g.add_nodes(
        pd.DataFrame({"id": [10, 11], "name": ["Acme", "Beta"]}),
        "Company",
        "id",
        "name",
    )
    g.add_connections(
        pd.DataFrame({"src": [1, 2, 3], "dst": [10, 10, 11]}),
        "WORKS_AT",
        "Person",
        "src",
        "Company",
        "dst",
    )
    return g


# ── db.graph_stats ─────────────────────────────────────────────────────


def test_graph_stats_basic(small_graph):
    rows = small_graph.cypher(
        "CALL db.graph_stats() YIELD node_count, edge_count, label_count, relationship_type_count"
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["node_count"] == 6  # 4 people + 2 companies
    assert r["edge_count"] == 3
    assert r["label_count"] == 2  # Person, Company
    assert r["relationship_type_count"] == 1  # WORKS_AT


def test_graph_stats_partial_yield(small_graph):
    """Only yield the fields the user asks for."""
    rows = small_graph.cypher("CALL db.graph_stats() YIELD node_count")
    assert len(rows) == 1
    assert rows[0]["node_count"] == 6


def test_graph_stats_empty_graph():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("CALL db.graph_stats() YIELD node_count, edge_count")
    assert rows[0]["node_count"] == 0
    assert rows[0]["edge_count"] == 0


# ── db.property_stats ──────────────────────────────────────────────────


def test_property_stats_with_duplicates(small_graph):
    """city has 4 values, distinct={Oslo, Bergen} = 2."""
    rows = small_graph.cypher(
        "CALL db.property_stats({node_type: 'Person', property: 'city'}) YIELD value_count, null_count, distinct_count"
    )
    assert rows[0]["value_count"] == 4
    assert rows[0]["null_count"] == 0
    assert rows[0]["distinct_count"] == 2  # Oslo x3, Bergen x1


def test_property_stats_with_nulls(small_graph):
    """email: 3 non-null, 1 null."""
    rows = small_graph.cypher(
        "CALL db.property_stats({node_type: 'Person', property: 'email'}) YIELD value_count, null_count, distinct_count"
    )
    assert rows[0]["value_count"] == 3
    assert rows[0]["null_count"] == 1
    assert rows[0]["distinct_count"] == 3


def test_property_stats_unknown_node_type(small_graph):
    rows = small_graph.cypher(
        "CALL db.property_stats({node_type: 'NoSuchType', property: 'x'}) YIELD value_count, null_count, distinct_count"
    )
    assert rows[0]["value_count"] == 0
    assert rows[0]["null_count"] == 0
    assert rows[0]["distinct_count"] == 0


def test_property_stats_missing_param(small_graph):
    with pytest.raises(Exception, match="requires a `node_type`"):
        small_graph.cypher("CALL db.property_stats({property: 'name'}) YIELD value_count")


# ── db.property_uniqueness ─────────────────────────────────────────────


def test_property_uniqueness_unique_field(small_graph):
    """id is unique on Person."""
    rows = small_graph.cypher(
        "CALL db.property_uniqueness({node_type: 'Person', property: 'id'}) "
        "YIELD is_unique, violation_count, distinct_count"
    )
    assert rows[0]["is_unique"] is True
    assert rows[0]["violation_count"] == 0
    assert rows[0]["distinct_count"] == 4


def test_property_uniqueness_non_unique_field(small_graph):
    """city on Person: Oslo appears 3x, Bergen 1x."""
    rows = small_graph.cypher(
        "CALL db.property_uniqueness({node_type: 'Person', property: 'city'}) "
        "YIELD is_unique, violation_count, distinct_count"
    )
    assert rows[0]["is_unique"] is False
    assert rows[0]["violation_count"] == 2  # 4 - 2 = 2 dupes
    assert rows[0]["distinct_count"] == 2


def test_property_uniqueness_unknown_node_type(small_graph):
    rows = small_graph.cypher(
        "CALL db.property_uniqueness({node_type: 'NoSuchType', property: 'x'}) "
        "YIELD is_unique, violation_count, distinct_count"
    )
    # Empty: is_unique is false (no values to be unique over)
    assert rows[0]["is_unique"] is False
    assert rows[0]["violation_count"] == 0
    assert rows[0]["distinct_count"] == 0


# ── db.propertyKeys ────────────────────────────────────────────────────


def test_property_keys_basic(small_graph):
    """Every declared property name across all node/relationship types, sorted
    and de-duplicated. Reflects node_type_metadata, which records every declared
    column (incl. `id` and the `name` natural-key/title field), unioned across
    Person {id,name,city,email} + Company {id,name}."""
    rows = small_graph.cypher("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey")
    keys = [r["propertyKey"] for r in rows]
    assert keys == ["city", "email", "id", "name"]


def test_property_keys_postfilter(small_graph):
    """YIELD feeds downstream WHERE like any other procedure stream."""
    rows = small_graph.cypher(
        "CALL db.propertyKeys() YIELD propertyKey WITH propertyKey WHERE propertyKey STARTS WITH 'c' RETURN propertyKey"
    )
    assert [r["propertyKey"] for r in rows] == ["city"]


# ── db.schema ──────────────────────────────────────────────────────────


def test_schema_basic(small_graph):
    """One row per node type with its sorted property-name list."""
    rows = small_graph.cypher(
        "CALL db.schema() YIELD nodeType, properties RETURN nodeType, properties ORDER BY nodeType"
    )
    by_type = {r["nodeType"]: r["properties"] for r in rows}
    assert set(by_type) == {"Person", "Company"}
    assert by_type["Person"] == ["city", "email", "id", "name"]
    assert by_type["Company"] == ["id", "name"]


def test_schema_and_property_keys_listed(small_graph):
    """Both new procedures are discoverable via list_procedures."""
    rows = small_graph.cypher("CALL list_procedures() YIELD name, yield_columns RETURN name, yield_columns")
    names = {r["name"]: r["yield_columns"] for r in rows}
    assert names.get("db.propertyKeys") == "propertyKey"
    assert names.get("db.schema") == "nodeType, properties"


# ── Standalone CALL + the result-column contract (2026-08-15) ──────────────


def test_bare_call_returns_all_declared_columns(small_graph):
    """`CALL db.labels()` with no YIELD — the form every Neo4j client sends —
    expands to the procedure's declared columns."""
    rows = small_graph.cypher("CALL db.labels()").to_dicts()
    assert rows and all(set(r) == {"label"} for r in rows)


def test_bare_call_rejected_mid_pipeline(small_graph):
    with pytest.raises(Exception, match="CALL requires a YIELD clause"):
        small_graph.cypher("MATCH (n) CALL db.labels()")


def test_call_columns_follow_yield_order(small_graph):
    """Columns come back in YIELD order (Neo4j semantics), not sorted
    alphabetically — pre-fix `YIELD type, name` answered [name, type]."""
    df = small_graph.cypher("CALL db.indexes() YIELD type, name", to_df=True)
    assert list(df.columns) == ["type", "name"]


def test_zero_row_call_keeps_declared_columns(small_graph):
    """A CALL that yields no rows still reports its columns — pre-fix a Bolt
    client's result.keys() came back empty on `CALL db.indexes()` against a
    fresh graph."""
    df = small_graph.cypher("CALL db.indexes()", to_df=True)
    assert len(df) == 0
    assert list(df.columns) == [
        "name",
        "type",
        "entityType",
        "labelsOrTypes",
        "properties",
        "state",
    ]


# ── SHOW PROCEDURES (2026-08-15) ───────────────────────────────────────────


def test_show_procedures_default_columns(small_graph):
    df = small_graph.cypher("SHOW PROCEDURES", to_df=True)
    assert list(df.columns) == ["name", "description", "mode", "worksOnSystem"]
    assert (df["mode"] == "READ").all()
    names = set(df["name"])
    assert {"pagerank", "db.labels", "list_procedures"} <= names


def test_show_procedures_yield_projection_and_alias(small_graph):
    df = small_graph.cypher("SHOW PROCEDURES YIELD name AS proc", to_df=True)
    assert list(df.columns) == ["proc"]


def test_show_procedures_agrees_with_list_procedures(small_graph):
    """SHOW PROCEDURES and CALL list_procedures() read the same registry —
    the drift this table replaced (list_procedures advertised db.labels as
    yielding `name`; the validator said `label`) must stay impossible."""
    shown = {r["name"] for r in small_graph.cypher("SHOW PROCEDURES YIELD name").to_dicts()}
    listed = {r["name"] for r in small_graph.cypher("CALL list_procedures() YIELD name").to_dicts()}
    assert shown == listed


def test_show_procedures_rejects_unknown_yield_and_where(small_graph):
    with pytest.raises(Exception, match="does not yield"):
        small_graph.cypher("SHOW PROCEDURES YIELD nope")
    with pytest.raises(Exception, match="YIELD projection"):
        small_graph.cypher("SHOW PROCEDURES WHERE name = 'x'")


def test_list_procedures_advertises_true_columns(small_graph):
    """The registry fixed a drift: db.labels yields `label`, not `name`."""
    rows = small_graph.cypher("CALL list_procedures() YIELD name, yield_columns").to_dicts()
    by = {r["name"]: r["yield_columns"] for r in rows}
    assert by["db.labels"] == "label"
    assert by["db.relationshipTypes"] == "relationshipType"


# ── db.schema.visualization() (2026-08-15) ─────────────────────────────────


def test_schema_visualization_shape():
    """One row: virtual nodes (one per label, name/indexes/constraints
    properties) + virtual relationships per (src, type, tgt) combination —
    the shape Neo4j Browser's schema tab renders."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'ann'})-[:LIVES_IN]->(:City {name: 'Oslo'})")
    g.cypher("CREATE (:Person {name: 'bob'})-[:KNOWS]->(:Person {name: 'cec'})")
    g.cypher("CREATE INDEX FOR (p:Person) ON (p.name)")

    rows = g.cypher("CALL db.schema.visualization()").to_dicts()
    assert len(rows) == 1
    nodes = rows[0]["nodes"]
    by_label = {n["labels"][0]: n for n in nodes}
    assert set(by_label) == {"Person", "City"}
    assert by_label["Person"]["properties"]["name"] == "Person"
    assert by_label["Person"]["properties"]["indexes"] == ["Person.name"]
    assert by_label["City"]["properties"]["indexes"] == []

    rels = rows[0]["relationships"]
    triples = {(nodes_id_label(nodes, r["start"]), r["type"], nodes_id_label(nodes, r["end"])) for r in rels}
    assert triples == {("Person", "KNOWS", "Person"), ("Person", "LIVES_IN", "City")}


def nodes_id_label(nodes, vid):
    return next(n["labels"][0] for n in nodes if n["id"] == vid)


def test_schema_visualization_yield_subset():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:A {x: 1})")
    rows = g.cypher("CALL db.schema.visualization() YIELD nodes").to_dicts()
    assert list(rows[0].keys()) == ["nodes"]


def test_schema_visualization_listed_in_registry():
    g = kglite.KnowledgeGraph()
    names = {r["name"] for r in g.cypher("SHOW PROCEDURES YIELD name").to_dicts()}
    assert "db.schema.visualization" in names


# ── db.schema.nodeTypeProperties / relTypeProperties (2026-08-15) ──────────
# Measured as the calls G.V()'s data-model load makes; Neo4j Browser's
# schema code path uses the same pair.


@pytest.fixture
def typed_graph():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'ann', age: 28})-[:KNOWS {since: 2020}]->(:Person {name: 'bob', age: 35})")
    g.cypher("CREATE (:Empty)")
    return g


def test_node_type_properties_shape(typed_graph):
    rows = typed_graph.cypher("CALL db.schema.nodeTypeProperties()").to_dicts()
    person = [r for r in rows if r["nodeType"] == ":`Person`"]
    by_prop = {r["propertyName"]: r for r in person}
    assert by_prop["age"]["propertyTypes"] == ["Long"]
    assert by_prop["name"]["propertyTypes"] == ["String"]
    assert by_prop["age"]["nodeLabels"] == ["Person"]
    assert all(r["mandatory"] is False for r in person)
    # A property-less label still appears — one row, null propertyName.
    empty = [r for r in rows if r["nodeType"] == ":`Empty`"]
    assert len(empty) == 1 and empty[0]["propertyName"] is None


def test_rel_type_properties_shape(typed_graph):
    rows = typed_graph.cypher("CALL db.schema.relTypeProperties()").to_dicts()
    assert rows == [
        {
            "relType": ":`KNOWS`",
            "propertyName": "since",
            "propertyTypes": ["Long"],
            "mandatory": False,
        }
    ]


def test_show_procedures_signature_yield(typed_graph):
    """The exact YIELD G.V() sends (measured 2026-08-15). `signature` is
    yieldable but not in the default column set, matching Neo4j."""
    rows = typed_graph.cypher("SHOW PROCEDURES YIELD name, description, signature").to_dicts()
    by = {r["name"]: r["signature"] for r in rows}
    assert by["db.labels"] == "db.labels(config = {} :: MAP?) :: (label :: ANY?)"
    default = typed_graph.cypher("SHOW PROCEDURES", to_df=True)
    assert "signature" not in default.columns


# ── apoc.meta.* compatibility shims (2026-08-15) ───────────────────────────


def test_apoc_meta_rel_type_properties_carries_endpoints():
    """The APOC shape's whole reason to exist: sourceNodeLabels /
    targetNodeLabels per observed (source, type, target) pairing — the
    columns schema-graph clients (G.V(), measured) draw their edges from,
    absent from the db.schema.* contract."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'a'})-[:LIVES_IN {since: 2020}]->(:City {name: 'O'})")
    g.cypher("CREATE (:Company {name: 'x'})-[:LIVES_IN]->(:City {name: 'B'})")
    rows = g.cypher("CALL apoc.meta.relTypeProperties() YIELD relType, sourceNodeLabels, targetNodeLabels").to_dicts()
    pairs = {(r["sourceNodeLabels"][0], r["relType"], r["targetNodeLabels"][0]) for r in rows}
    assert pairs == {("Person", ":`LIVES_IN`", "City"), ("Company", ":`LIVES_IN`", "City")}


def test_apoc_meta_node_type_properties_matches_db_schema():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'a', age: 3})")
    apoc = g.cypher("CALL apoc.meta.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes").to_dicts()
    native = g.cypher("CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes").to_dicts()
    assert apoc == native
    counts = g.cypher("CALL apoc.meta.nodeTypeProperties() YIELD totalObservations").to_dicts()
    assert all(r["totalObservations"] == 1 for r in counts)


def test_no_other_apoc_name_resolves():
    """The shim is scoped to exactly two names — the apoc namespace stays
    otherwise closed (no accidental 'we have APOC' impression)."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:A {x: 1})")
    with pytest.raises(Exception, match="Unknown procedure"):
        g.cypher("CALL apoc.meta.data()")
    with pytest.raises(Exception, match="Unknown procedure"):
        g.cypher("CALL apoc.version()")


# ── SHOW FUNCTIONS (2026-08-15) ────────────────────────────────────────────
# G.V() sends `SHOW FUNCTIONS YIELD name, description, signature` on connect;
# without it the IDE's function autocomplete is dead. The listing is gated
# against the real dispatcher in Rust
# (scalar_functions::function_registry::tests::every_registry_name_dispatches),
# so these tests only need to check the statement's shape.


def test_show_functions_default_columns(small_graph):
    df = small_graph.cypher("SHOW FUNCTIONS", to_df=True)
    assert list(df.columns) == ["name", "category", "description"]
    names = set(df["name"])
    assert {"toUpper", "coalesce", "labels", "count"} <= names
    # Rows are sorted by name and carry no empty descriptions.
    assert list(df["name"]) == sorted(df["name"])
    assert all(d for d in df["description"])


def test_show_functions_signature_yield_is_the_gv_query(small_graph):
    """The exact query G.V() sends (measured 2026-08-15)."""
    rows = small_graph.cypher("SHOW FUNCTIONS YIELD name, description, signature").to_dicts()
    assert rows
    by = {r["name"]: r["signature"] for r in rows}
    assert by["toUpper"] == "toUpper(input :: STRING) :: STRING?"
    assert by["randomUUID"] == "randomUUID() :: STRING"
    # `signature` is yieldable but not in the default column set, as in Neo4j.
    assert "signature" not in small_graph.cypher("SHOW FUNCTIONS", to_df=True).columns


def test_show_functions_yield_alias_and_aliases_column(small_graph):
    df = small_graph.cypher("SHOW FUNCTIONS YIELD name AS fn", to_df=True)
    assert list(df.columns) == ["fn"]
    rows = small_graph.cypher("SHOW FUNCTIONS YIELD name, aliases").to_dicts()
    by = {r["name"]: r["aliases"] for r in rows}
    # An accepted alternate spelling is a field of its canonical row, not a
    # row of its own — Neo4j lists canonical names only.
    assert by["toUpper"] == ["toUpperCase"]
    assert by["log"] == ["ln"]
    assert by["floor"] == []
    assert "toUpperCase" not in by


def test_show_functions_listing_case_matches_the_callable(small_graph):
    """Names are listed in their conventional camelCase spelling and the
    parser lowercases before dispatch, so what the listing shows is callable
    verbatim."""
    assert small_graph.cypher("RETURN toUpper('a') AS v").to_dicts()[0]["v"] == "A"
    assert small_graph.cypher("RETURN TOUPPER('a') AS v").to_dicts()[0]["v"] == "A"
    assert small_graph.cypher("RETURN toUpperCase('a') AS v").to_dicts()[0]["v"] == "A"


def test_show_functions_rejects_unknown_yield_and_where(small_graph):
    with pytest.raises(Exception, match="does not yield"):
        small_graph.cypher("SHOW FUNCTIONS YIELD nope")
    with pytest.raises(Exception, match="YIELD projection"):
        small_graph.cypher("SHOW FUNCTIONS WHERE name = 'toUpper'")


def test_functions_and_procedures_are_separate_registries(small_graph):
    """Two registries, two namespaces. `degree` is the one deliberate name in
    both — `RETURN degree(n)` is a scalar function, `CALL degree()` is the
    centrality procedure — and pinning the intersection keeps a third
    collision from arriving unnoticed."""
    functions = {r["name"] for r in small_graph.cypher("SHOW FUNCTIONS YIELD name").to_dicts()}
    procedures = {r["name"] for r in small_graph.cypher("SHOW PROCEDURES YIELD name").to_dicts()}
    assert functions & procedures == {"degree"}
    assert "pagerank" not in functions
    assert "toUpper" not in procedures


def test_no_registry_function_panics_on_zero_args():
    """Every listed function called with no arguments must ERROR, never
    panic — pre-fix 26 single-arg functions indexed args[0] unguarded and
    a bare `RETURN size()` killed the process (over Bolt: the connection)."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:A {x: 1})")
    for row in g.cypher("SHOW FUNCTIONS YIELD name, category").to_dicts():
        if row["category"] == "aggregate":
            continue
        try:
            g.cypher(f"RETURN {row['name']}() AS v")
        except BaseException as exc:  # noqa: BLE001 — panics surface as BaseException
            assert type(exc).__name__ != "PanicException", row["name"]


def test_bulk_loaded_edge_properties_are_typed():
    """add_connections must record observed property types like the Cypher
    CREATE path — pre-fix every bulk-loaded edge property registered as
    'Unknown' (all 59 sodir rel properties showed `unknown` in G.V())."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}), "P", "id", "name")
    g.add_connections(
        pd.DataFrame({"src": [1], "tgt": [2], "since": [2020], "w": [0.5], "tag": ["x"]}),
        "KNOWS",
        "P",
        "src",
        "P",
        "tgt",
    )
    rows = g.cypher("CALL db.schema.relTypeProperties() YIELD propertyName, propertyTypes").to_dicts()
    types = {r["propertyName"]: r["propertyTypes"] for r in rows}
    assert types == {"since": ["Long"], "tag": ["String"], "w": ["Double"]}
