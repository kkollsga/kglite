"""Tests for the structural-validator CALL procedures.

Replaces ``tests/test_rules.py`` (deleted in 0.8.19). Each procedure is
exercised in isolation against a tiny graph plus end-to-end against a
realistic edge-direction setup. Procedures verified:

- ``orphan_node({type})``
- ``self_loop({type, edge})``
- ``cycle_2step({type, edge})``
- ``missing_required_edge({type, edge})``
- ``missing_inbound_edge({type, edge})``
- ``duplicate_title({type})``

Plus the direction validator that fires before iteration when a caller
passes the wrong (type, edge) pair for ``missing_required_edge`` or
``missing_inbound_edge``.
"""

import pandas as pd
import pytest

import kglite


@pytest.fixture
def integrity_graph():
    """4 LawSections (one orphaned, two share a title) + 1 Law."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "sec1", "title": "§1"},
                {"id": "sec2", "title": "§2"},
                {"id": "sec3", "title": "§3-orphan"},  # zero-degree
                {"id": "sec4", "title": "§1"},  # duplicates sec1's title
            ]
        ),
        "LawSection",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame([{"id": "lov1", "title": "Test Law"}]),
        "Law",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"s": "sec1", "t": "lov1"},
                {"s": "sec2", "t": "lov1"},
                {"s": "sec4", "t": "lov1"},
            ]
        ),
        "SECTION_OF",
        "LawSection",
        "s",
        "Law",
        "t",
    )
    return g


# ---------------------------------------------------------------------
# orphan_node


def test_orphan_node_finds_zero_degree(integrity_graph):
    rows = list(integrity_graph.cypher("CALL orphan_node({type: 'LawSection'}) YIELD node RETURN node.id AS id"))
    assert len(rows) == 1
    assert rows[0]["id"] == "sec3"


def test_orphan_node_clean_type_returns_zero(integrity_graph):
    rows = list(integrity_graph.cypher("CALL orphan_node({type: 'Law'}) YIELD node RETURN count(node) AS c"))
    assert rows[0]["c"] == 0


def test_orphan_node_unknown_type_errors(integrity_graph):
    with pytest.raises(kglite.KgError, match="no nodes"):
        integrity_graph.cypher("CALL orphan_node({type: 'NoSuchType'}) YIELD node RETURN node")


def test_orphan_node_missing_param_errors(integrity_graph):
    with pytest.raises(kglite.KgError, match="missing required parameter"):
        integrity_graph.cypher("CALL orphan_node({}) YIELD node RETURN node")


def test_orphan_node_missing_param_error_lists_accepted_params(integrity_graph):
    """Error message includes the accepted-parameters hint so first-time
    users don't have to brute-force-guess the param name (issue #13)."""
    try:
        integrity_graph.cypher("CALL orphan_node({wrong: 'X'}) YIELD node RETURN node")
    except kglite.KgError as exc:
        msg = str(exc)
        assert "Accepted parameters" in msg, f"missing schema hint in error: {msg}"
        assert "type" in msg, f"required param 'type' must be listed in error: {msg}"
    else:
        pytest.fail("expected a ValueError/RuntimeError")


def test_orphan_node_link_type_in_finds_section_with_no_inbound(integrity_graph):
    """`direction='in'` + `link_type` = "no node has an inbound matching edge".

    None of the LawSections have inbound SECTION_OF edges (the edges
    flow LawSection -[SECTION_OF]-> Law), so all four sections must be
    reported as orphans by an inbound-CALLS-style query.
    """
    rows = list(
        integrity_graph.cypher(
            "CALL orphan_node({type: 'LawSection', link_type: 'SECTION_OF', direction: 'in'}) "
            "YIELD node RETURN node.id AS id ORDER BY id"
        )
    )
    ids = [r["id"] for r in rows]
    assert ids == ["sec1", "sec2", "sec3", "sec4"]


def test_orphan_node_link_type_out_excludes_connected_sections(integrity_graph):
    """`direction='out'` + `link_type='SECTION_OF'` reports only sections
    with NO outbound SECTION_OF edge — i.e. the orphan sec3 (no edges
    at all).
    """
    rows = list(
        integrity_graph.cypher(
            "CALL orphan_node({type: 'LawSection', link_type: 'SECTION_OF', direction: 'out'}) "
            "YIELD node RETURN node.id AS id"
        )
    )
    assert [r["id"] for r in rows] == ["sec3"]


def test_orphan_node_link_type_filters_to_specific_edge_type(integrity_graph):
    """A non-existent `link_type` makes every node an orphan because
    none have edges of that connection type — proves the filter is
    actually applied (vs. silently falling back to "any edge").
    """
    rows = list(
        integrity_graph.cypher(
            "CALL orphan_node({type: 'LawSection', link_type: 'NON_EXISTENT_EDGE'}) YIELD node RETURN count(node) AS c"
        )
    )
    assert rows[0]["c"] == 4


def test_orphan_node_invalid_direction_errors(integrity_graph):
    with pytest.raises(kglite.KgError, match="invalid direction"):
        integrity_graph.cypher("CALL orphan_node({type: 'LawSection', direction: 'sideways'}) YIELD node RETURN node")


def test_cardinality_violation_error_lists_optional_params(integrity_graph):
    """Optional params (min, max) should also appear in the schema hint."""
    try:
        integrity_graph.cypher("CALL cardinality_violation({}) YIELD node RETURN node")
    except kglite.KgError as exc:
        msg = str(exc)
        assert "required: [type, edge]" in msg, f"required list missing/wrong: {msg}"
        assert "optional: [min, max]" in msg, f"optional list missing/wrong: {msg}"
    else:
        pytest.fail("expected a ValueError/RuntimeError")


# ---------------------------------------------------------------------
# self_loop + cycle_2step


@pytest.fixture
def loop_graph():
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "a", "title": "A"},
                {"id": "b", "title": "B"},
                {"id": "c", "title": "C-self"},
            ]
        ),
        "T",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"s": "a", "t": "b"},  # a→b
                {"s": "b", "t": "a"},  # b→a (forms a 2-cycle)
                {"s": "c", "t": "c"},  # self-loop on c
            ]
        ),
        "E",
        "T",
        "s",
        "T",
        "t",
    )
    return g


def test_self_loop_finds_node(loop_graph):
    rows = list(loop_graph.cypher("CALL self_loop({type: 'T', edge: 'E'}) YIELD node RETURN node.id AS id"))
    assert [r["id"] for r in rows] == ["c"]


def test_cycle_2step_finds_pair(loop_graph):
    rows = list(
        loop_graph.cypher(
            "CALL cycle_2step({type: 'T', edge: 'E'}) YIELD node_a, node_b RETURN node_a.id AS a, node_b.id AS b"
        )
    )
    assert len(rows) == 1
    pair = (rows[0]["a"], rows[0]["b"])
    assert pair == ("a", "b") or pair == ("b", "a")


def test_cycle_2step_yields_named_node_a_node_b(loop_graph):
    """Variables must be node_a / node_b — `end` is reserved by Cypher (CASE)."""
    rows = list(loop_graph.cypher("CALL cycle_2step({type: 'T', edge: 'E'}) YIELD node_a, node_b RETURN count(*) AS c"))
    assert rows[0]["c"] == 1


# ---------------------------------------------------------------------
# missing_required_edge / missing_inbound_edge + direction validation


@pytest.fixture
def directional_graph():
    """Wellbore -[IN_LICENCE]-> Licence (outbound from Wellbore)."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "w1", "title": "W1"},
                {"id": "w2", "title": "W2"},
                {"id": "w3", "title": "W3-no-licence"},
            ]
        ),
        "Wellbore",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame([{"id": "l1", "title": "PL001"}]),
        "Licence",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"s": "w1", "t": "l1"},
                {"s": "w2", "t": "l1"},
            ]
        ),
        "IN_LICENCE",
        "Wellbore",
        "s",
        "Licence",
        "t",
    )
    return g


def test_missing_required_edge_finds_no_outbound(directional_graph):
    rows = list(
        directional_graph.cypher(
            "CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node RETURN node.id AS id"
        )
    )
    assert [r["id"] for r in rows] == ["w3"]


def test_missing_inbound_edge_finds_no_inbound(directional_graph):
    """All licences have inbound IN_LICENCE; an unreferenced licence wouldn't."""
    directional_graph.add_nodes(
        pd.DataFrame([{"id": "l_orphan", "title": "PL999"}]),
        "Licence",
        "id",
        "title",
    )
    rows = list(
        directional_graph.cypher(
            "CALL missing_inbound_edge({type: 'Licence', edge: 'IN_LICENCE'}) YIELD node RETURN node.id AS id"
        )
    )
    assert [r["id"] for r in rows] == ["l_orphan"]


def test_direction_validator_blocks_outbound_misuse(directional_graph):
    """Calling missing_required_edge with type=Licence flips on direction validator."""
    with pytest.raises(kglite.KgError, match="DirectionMismatch"):
        directional_graph.cypher(
            "CALL missing_required_edge({type: 'Licence', edge: 'IN_LICENCE'}) YIELD node RETURN node"
        )


def test_direction_validator_blocks_inbound_misuse(directional_graph):
    """Calling missing_inbound_edge with type=Wellbore flips on direction validator."""
    with pytest.raises(kglite.KgError, match="DirectionMismatch"):
        directional_graph.cypher(
            "CALL missing_inbound_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node RETURN node"
        )


# ---------------------------------------------------------------------
# duplicate_title


def test_duplicate_title_emits_one_row_per_duplicate(integrity_graph):
    rows = list(integrity_graph.cypher("CALL duplicate_title({type: 'LawSection'}) YIELD node RETURN node.id AS id"))
    # sec1 + sec4 share '§1' — both flagged
    assert sorted(r["id"] for r in rows) == ["sec1", "sec4"]


def test_duplicate_title_aggregates_downstream(integrity_graph):
    rows = list(
        integrity_graph.cypher(
            "CALL duplicate_title({type: 'LawSection'}) YIELD node "
            "WITH node.title AS title, collect(node) AS dups "
            "WITH title, size(dups) AS dup_count "
            "RETURN title, dup_count"
        )
    )
    assert len(rows) == 1
    assert rows[0]["title"] == "§1"
    assert rows[0]["dup_count"] == 2


# ---------------------------------------------------------------------
# Composability — rule procedures flow into surrounding Cypher


def test_call_composes_with_where_and_order_by(integrity_graph):
    """Rule output flows naturally into WHERE / ORDER BY / LIMIT."""
    rows = list(
        integrity_graph.cypher(
            "CALL duplicate_title({type: 'LawSection'}) YIELD node WHERE node.id <> 'sec1' RETURN node.id AS id"
        )
    )
    assert [r["id"] for r in rows] == ["sec4"]


def test_call_composes_with_match_and_aggregation(directional_graph):
    """Cross-reference query results against rule output in a single Cypher pass."""
    # Add wellbore w4 with no IN_LICENCE so missing_required_edge has 2 hits
    directional_graph.add_nodes(
        pd.DataFrame([{"id": "w4", "title": "W4"}]),
        "Wellbore",
        "id",
        "title",
    )
    rows = list(
        directional_graph.cypher(
            "CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node "
            "WHERE node.title CONTAINS 'W' "
            "RETURN count(node) AS c"
        )
    )
    assert rows[0]["c"] == 2


# ---------------------------------------------------------------------
# Procedure catalogue + YIELD validation


def test_list_procedures_includes_rule_procedures(integrity_graph):
    rows = list(integrity_graph.cypher("CALL list_procedures() YIELD name RETURN name"))
    names = {r["name"] for r in rows}
    for p in (
        "orphan_node",
        "self_loop",
        "cycle_2step",
        "missing_required_edge",
        "missing_inbound_edge",
        "duplicate_title",
        "null_property",
        "inverse_violation",
        "transitivity_violation",
        "cardinality_violation",
        "type_domain_violation",
        "type_range_violation",
        "parallel_edges",
    ):
        assert p in names, f"missing procedure: {p}"


def test_invalid_yield_column_rejected(integrity_graph):
    with pytest.raises(kglite.KgError, match="does not yield"):
        integrity_graph.cypher("CALL orphan_node({type: 'LawSection'}) YIELD bogus RETURN bogus")


# ---------------------------------------------------------------------
# 0.8.20 — rule packs v2: validators


@pytest.fixture
def schema_graph():
    """Two-tier graph: A→B→C parent_of chain; partial child_of inverse;
    extra Q node connects to P with a wrong source type."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": 1, "name": "A", "email": "a@x"},
                {"id": 2, "name": "B", "email": ""},
                {"id": 3, "name": "C", "email": None},
                {"id": 4, "name": "D"},
            ]
        ),
        "P",
        "id",
        "name",
    )
    # parent_of: A→B, B→C
    g.add_connections(
        pd.DataFrame([{"f": 1, "t": 2}, {"f": 2, "t": 3}]),
        "parent_of",
        "P",
        "f",
        "P",
        "t",
    )
    # child_of: only B→A (so A→B has inverse, B→C does not)
    g.add_connections(
        pd.DataFrame([{"f": 2, "t": 1}]),
        "child_of",
        "P",
        "f",
        "P",
        "t",
    )
    return g


class TestInverseViolation:
    def test_finds_unidirectional(self, schema_graph):
        rows = list(
            schema_graph.cypher(
                "CALL inverse_violation({rel_a: 'parent_of', rel_b: 'child_of'}) YIELD a, b "
                "RETURN a.id AS av, b.id AS bv"
            )
        )
        # B→C parent_of has no C→B child_of inverse; A→B does have inverse
        ids = {(r["av"], r["bv"]) for r in rows}
        assert (2, 3) in ids
        assert (1, 2) not in ids


class TestTransitivityViolation:
    def test_finds_chain(self, schema_graph):
        rows = list(
            schema_graph.cypher(
                "CALL transitivity_violation({rel: 'parent_of'}) YIELD a, b, c "
                "RETURN a.id AS av, b.id AS bv, c.id AS cv"
            )
        )
        # A→B→C with no A→C
        triples = {(r["av"], r["bv"], r["cv"]) for r in rows}
        assert (1, 2, 3) in triples


class TestCardinalityViolation:
    def test_max_zero_flags_all(self, schema_graph):
        # max=0 means no edges allowed; A and B both have 1 outgoing parent_of
        rows = list(
            schema_graph.cypher(
                "CALL cardinality_violation({type: 'P', edge: 'parent_of', max: 0}) YIELD node, count "
                "RETURN node.id AS nv, count"
            )
        )
        flagged = {(r["nv"], r["count"]) for r in rows}
        assert (1, 1) in flagged
        assert (2, 1) in flagged

    def test_min_two_flags_zero_count(self, schema_graph):
        rows = list(
            schema_graph.cypher(
                "CALL cardinality_violation({type: 'P', edge: 'parent_of', min: 2}) YIELD node, count "
                "RETURN node.id AS nv, count"
            )
        )
        # All 4 nodes have <2 outgoing parent_of
        ids = {r["nv"] for r in rows}
        assert ids == {1, 2, 3, 4}


class TestNullProperty:
    def test_finds_missing_email(self, schema_graph):
        rows = list(
            schema_graph.cypher(
                "CALL null_property({type: 'P', property: 'email'}) YIELD node RETURN node.id AS nv ORDER BY nv"
            )
        )
        ids = [r["nv"] for r in rows]
        # B has empty string, C has null, D has no email property
        assert 2 in ids
        assert 3 in ids
        assert 4 in ids
        assert 1 not in ids


class TestTypeDomainAndRangeViolation:
    def test_wrong_source_type(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame([{"id": 1, "n": "p1"}]), "P", "id", "n")
        g.add_nodes(pd.DataFrame([{"id": 100, "n": "q1"}]), "Q", "id", "n")
        g.add_connections(pd.DataFrame([{"f": 100, "t": 1}]), "LINK", "Q", "f", "P", "t")
        rows = list(
            g.cypher(
                "CALL type_domain_violation({edge: 'LINK', expected_source: 'P'}) "
                "YIELD source, target RETURN source.id AS sv"
            )
        )
        assert any(r["sv"] == 100 for r in rows)

    def test_wrong_target_type(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame([{"id": 1, "n": "p1"}]), "P", "id", "n")
        g.add_nodes(pd.DataFrame([{"id": 100, "n": "q1"}]), "Q", "id", "n")
        g.add_connections(pd.DataFrame([{"f": 1, "t": 100}]), "LINK", "P", "f", "Q", "t")
        rows = list(
            g.cypher(
                "CALL type_range_violation({edge: 'LINK', expected_target: 'P'}) "
                "YIELD source, target RETURN target.id AS tv"
            )
        )
        assert any(r["tv"] == 100 for r in rows)


class TestParallelEdges:
    def test_detects_duplicates(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame([{"id": 1, "n": "a"}, {"id": 2, "n": "b"}]), "P", "id", "n")
        # CREATE bypasses dedup
        for _ in range(3):
            g.cypher("MATCH (a:P {id:1}), (b:P {id:2}) CREATE (a)-[:LINK]->(b)")
        rows = list(
            g.cypher("CALL parallel_edges({edge: 'LINK'}) YIELD a, b, count RETURN a.id AS av, b.id AS bv, count")
        )
        assert len(rows) == 1
        assert rows[0]["count"] == 3

    def test_no_duplicates_returns_empty(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame([{"id": 1, "n": "a"}, {"id": 2, "n": "b"}]), "P", "id", "n")
        g.cypher("MATCH (a:P {id:1}), (b:P {id:2}) CREATE (a)-[:LINK]->(b)")
        rows = list(g.cypher("CALL parallel_edges({edge: 'LINK'}) YIELD a, b, count RETURN count"))
        assert rows == []


# ============================================================================
# Phase A.3 — db.* schema-introspection procedures.
# ============================================================================


@pytest.fixture
def db_proc_graph():
    """Small graph with 2 node types + 2 relationship types for db.* tests."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1, 2], "title": ["Alice", "Bob"]}),
        "Person",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame({"id": [10], "title": ["Acme"]}),
        "Company",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame({"p": [1], "c": [10]}),
        "WORKS_AT",
        "Person",
        "p",
        "Company",
        "c",
    )
    g.add_connections(
        pd.DataFrame({"p": [2], "c": [10]}),
        "CONSULTS_FOR",
        "Person",
        "p",
        "Company",
        "c",
    )
    return g


class TestDbLabels:
    """CALL db.labels() — Neo4j-compatible node-type enumeration."""

    def test_returns_all_node_types(self, db_proc_graph):
        rows = list(db_proc_graph.cypher("CALL db.labels() YIELD label RETURN label"))
        names = {r["label"] for r in rows}
        assert names == {"Person", "Company"}

    def test_alphabetical_order(self, db_proc_graph):
        rows = list(db_proc_graph.cypher("CALL db.labels() YIELD label RETURN label"))
        names = [r["label"] for r in rows]
        assert names == sorted(names), "db.labels() must return labels in alphabetical order"

    def test_yield_alias(self, db_proc_graph):
        rows = list(db_proc_graph.cypher("CALL db.labels() YIELD label RETURN label"))
        assert {r["label"] for r in rows} == {"Person", "Company"}

    def test_empty_graph_returns_zero_rows(self):
        g = kglite.KnowledgeGraph()
        rows = list(g.cypher("CALL db.labels() YIELD label RETURN label"))
        assert rows == []

    def test_invalid_yield_column_rejected(self, db_proc_graph):
        with pytest.raises(kglite.KgError, match="does not yield"):
            db_proc_graph.cypher("CALL db.labels() YIELD bogus RETURN bogus")

    def test_post_create_visibility(self, db_proc_graph):
        """A node type added via CREATE must appear in db.labels() output."""
        db_proc_graph.cypher("CREATE (n:Product {id: 999, title: 'Widget'})")
        rows = list(db_proc_graph.cypher("CALL db.labels() YIELD label RETURN label"))
        names = {r["label"] for r in rows}
        assert "Product" in names

    def test_composes_with_where(self, db_proc_graph):
        rows = list(db_proc_graph.cypher("CALL db.labels() YIELD label WHERE label STARTS WITH 'P' RETURN label"))
        assert [r["label"] for r in rows] == ["Person"]


class TestDbRelationshipTypes:
    """CALL db.relationshipTypes() — Neo4j-compatible edge-type enumeration."""

    def test_returns_all_relationship_types(self, db_proc_graph):
        rows = list(db_proc_graph.cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"))
        names = {r["relationshipType"] for r in rows}
        assert names == {"WORKS_AT", "CONSULTS_FOR"}

    def test_alphabetical_order(self, db_proc_graph):
        rows = list(db_proc_graph.cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"))
        names = [r["relationshipType"] for r in rows]
        assert names == sorted(names)

    def test_camel_case_procedure_name(self, db_proc_graph):
        """Procedure dispatch is case-insensitive on the name (Neo4j convention)."""
        rows_camel = list(
            db_proc_graph.cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        )
        rows_lower = list(
            db_proc_graph.cypher("CALL db.relationshiptypes() YIELD relationshipType RETURN relationshipType")
        )
        assert {r["relationshipType"] for r in rows_camel} == {r["relationshipType"] for r in rows_lower}

    def test_yield_alias(self, db_proc_graph):
        rows = list(db_proc_graph.cypher("CALL db.relationshipTypes() YIELD relationshipType AS rel RETURN rel"))
        assert {r["rel"] for r in rows} == {"WORKS_AT", "CONSULTS_FOR"}

    def test_empty_graph_returns_zero_rows(self):
        g = kglite.KnowledgeGraph()
        rows = list(g.cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"))
        assert rows == []

    def test_post_create_visibility(self, db_proc_graph):
        db_proc_graph.cypher("MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:FRIEND_OF]->(b)")
        rows = list(db_proc_graph.cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"))
        names = {r["relationshipType"] for r in rows}
        assert "FRIEND_OF" in names

    def test_invalid_yield_column_rejected(self, db_proc_graph):
        with pytest.raises(kglite.KgError, match="does not yield"):
            db_proc_graph.cypher("CALL db.relationshipTypes() YIELD bogus RETURN bogus")


class TestNamespacedProcedureParsing:
    """The parser must accept `<namespace>.<procedure>` qualified names."""

    def test_unknown_dotted_procedure_errors_cleanly(self, db_proc_graph):
        """A typo in a dotted name should produce 'Unknown procedure', not a parse error."""
        with pytest.raises(kglite.KgError, match="Unknown procedure"):
            db_proc_graph.cypher("CALL db.bogus() YIELD name RETURN name")

    def test_dotted_name_with_no_yield_errors(self, db_proc_graph):
        with pytest.raises(kglite.KgError):
            db_proc_graph.cypher("CALL db.labels()")

    def test_list_procedures_advertises_db_namespace(self, db_proc_graph):
        rows = list(db_proc_graph.cypher("CALL list_procedures() YIELD name RETURN name"))
        names = {r["name"] for r in rows}
        assert "db.labels" in names
        assert "db.relationshipTypes" in names
        assert "db.indexes" in names


@pytest.fixture
def indexed_graph():
    """Graph with equality, composite, and range indexes for db.indexes() tests."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Alice", "Bob", "Carol"],
                "age": [30, 25, 40],
                "city": ["Oslo", "Bergen", "Oslo"],
            }
        ),
        "Person",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame({"id": [10], "title": ["Acme"]}),
        "Company",
        "id",
        "title",
    )
    g.create_index("Person", "city")  # equality
    g.create_composite_index("Person", ["age", "city"])  # composite
    g.create_range_index("Person", "age")  # range
    return g


class TestDbIndexes:
    """CALL db.indexes() — Neo4j-compatible index enumeration with structured columns."""

    def test_all_columns(self, indexed_graph):
        rows = list(
            indexed_graph.cypher(
                "CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state "
                "RETURN name, type, entityType, labelsOrTypes, properties, state ORDER BY name"
            )
        )
        # 3 indexes: Person.city (equality), Person.(age,city) (composite), Person.age (range)
        assert len(rows) == 3
        # All cover NODE entities, all ONLINE.
        for r in rows:
            assert r["entityType"] == "NODE"
            assert r["state"] == "ONLINE"
            assert r["labelsOrTypes"] == ["Person"]

    def test_equality_index_columns(self, indexed_graph):
        rows = list(
            indexed_graph.cypher(
                "CALL db.indexes() YIELD name, type, properties WHERE name = 'Person.city' RETURN type, properties"
            )
        )
        assert len(rows) == 1
        assert rows[0]["type"] == "PROPERTY"
        assert rows[0]["properties"] == ["city"]

    def test_composite_index_lists_multiple_properties(self, indexed_graph):
        rows = list(
            indexed_graph.cypher(
                "CALL db.indexes() YIELD name, type, properties "
                "WHERE name STARTS WITH 'Person.(' "
                "RETURN type, properties"
            )
        )
        assert len(rows) == 1
        assert rows[0]["type"] == "PROPERTY"
        assert rows[0]["properties"] == ["age", "city"]

    def test_range_index_marked_with_range_type(self, indexed_graph):
        """KGLite-specific: range indexes report type='RANGE' (not Neo4j's 'PROPERTY')."""
        rows = list(indexed_graph.cypher("CALL db.indexes() YIELD name, type WHERE type = 'RANGE' RETURN name"))
        assert len(rows) == 1
        assert rows[0]["name"] == "Person.age"

    def test_partial_yield_works(self, indexed_graph):
        """Users can YIELD a subset of columns."""
        rows = list(indexed_graph.cypher("CALL db.indexes() YIELD name RETURN name ORDER BY name"))
        names = [r["name"] for r in rows]
        assert names == sorted(names)
        assert len(names) == 3

    def test_yield_alias(self, indexed_graph):
        rows = list(
            indexed_graph.cypher(
                "CALL db.indexes() YIELD name AS idx_name, type AS idx_type RETURN idx_name, idx_type ORDER BY idx_name"
            )
        )
        assert {r["idx_name"] for r in rows} == {
            "Person.city",
            "Person.(age,city)",
            "Person.age",
        }

    def test_empty_graph_returns_zero_rows(self):
        g = kglite.KnowledgeGraph()
        rows = list(g.cypher("CALL db.indexes() YIELD name RETURN name"))
        assert rows == []

    def test_graph_with_no_indexes_returns_zero_rows(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [1], "title": ["A"]}), "T", "id", "title")
        rows = list(g.cypher("CALL db.indexes() YIELD name RETURN name"))
        assert rows == []

    def test_invalid_yield_column_rejected(self, indexed_graph):
        with pytest.raises(kglite.KgError, match="does not yield"):
            indexed_graph.cypher("CALL db.indexes() YIELD bogus RETURN bogus")

    def test_alphabetical_order(self, indexed_graph):
        rows = list(indexed_graph.cypher("CALL db.indexes() YIELD name RETURN name"))
        names = [r["name"] for r in rows]
        assert names == sorted(names)

    def test_schema_method_still_reports_same_indexes(self, indexed_graph):
        """Phase A.3 refactored compute_schema() to share collect_indexes_structured —
        ensure the existing schema()['indexes'] output shape is preserved."""
        schema_indexes = indexed_graph.schema()["indexes"]
        # The schema() API formats names with " [range]" suffix and comma+space
        # in composite property lists; verify both shapes survive the refactor.
        assert "Person.city" in schema_indexes
        assert "Person.(age, city)" in schema_indexes
        assert "Person.age [range]" in schema_indexes
