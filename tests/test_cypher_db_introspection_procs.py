"""db.* schema-introspection Cypher procedures — 2026-05-25 Batch 6.

Three procedures every binding can call via cypher_query:

  CALL db.graph_stats() YIELD node_count, edge_count, label_count, relationship_type_count
  CALL db.property_stats(node_type, property) YIELD value_count, null_count, distinct_count
  CALL db.property_uniqueness(node_type, property) YIELD is_unique, violation_count, distinct_count

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
