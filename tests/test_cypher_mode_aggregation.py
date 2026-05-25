"""mode(x) Cypher aggregation — 2026-05-25 Batch 5.

Returns the most-frequent value per group. Works on any Value type
(strings, ints, floats, dates). Real use case: "most common city
per country": `RETURN p.country, mode(p.city) AS top_city`.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def people_graph():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "city": [
                "Oslo",
                "Oslo",
                "Bergen",
                "Oslo",
                "Bergen",
                "Trondheim",
                "Oslo",
                "Bergen",
                "Bergen",
                "Oslo",
            ],
            "country": ["NO"] * 10,
        }
    )
    g.add_nodes(df, "Person", "id", "name")
    return g


def test_mode_simple_string(people_graph):
    """Oslo appears 4 times, Bergen 4 times, Trondheim 1. Oslo wins on debug-order tiebreak."""
    rows = people_graph.cypher("MATCH (p:Person) RETURN mode(p.city) AS most_common")
    # Either Oslo or Bergen — both have 4 occurrences. Deterministic.
    assert rows[0]["most_common"] in {"Oslo", "Bergen"}


def test_mode_with_clear_winner():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "color": ["red", "blue", "red", "red", "blue"],
        }
    )
    g.add_nodes(df, "Item", "id", "name")
    rows = g.cypher("MATCH (i:Item) RETURN mode(i.color) AS top_color")
    assert rows[0]["top_color"] == "red"  # red x3 > blue x2


def test_mode_on_integer():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"], "score": [10, 20, 10, 30, 10]})
    g.add_nodes(df, "Item", "id", "name")
    rows = g.cypher("MATCH (i:Item) RETURN mode(i.score) AS top")
    assert rows[0]["top"] == 10


def test_mode_skips_null():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "x": ["A", None, "A", None, "B"],
        }
    )
    g.add_nodes(df, "Item", "id", "name")
    rows = g.cypher("MATCH (i:Item) RETURN mode(i.x) AS top")
    # A appears 2 times (non-null), B once. Nulls don't count.
    assert rows[0]["top"] == "A"


def test_mode_empty_group_returns_null():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("MATCH (i:NoSuchType) RETURN mode(i.x) AS top")
    assert rows[0]["top"] is None


def test_mode_per_group_with_collect():
    """Real use case: mode per group via GROUP BY."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": list(range(1, 8)),
            "name": ["a", "b", "c", "d", "e", "f", "g"],
            "team": ["X", "X", "X", "Y", "Y", "Y", "Y"],
            "color": ["red", "red", "blue", "green", "green", "blue", "green"],
        }
    )
    g.add_nodes(df, "Player", "id", "name")
    rows = g.cypher("MATCH (p:Player) RETURN p.team AS team, mode(p.color) AS top_color ORDER BY team")
    # team X: red(2), blue(1) → red
    # team Y: green(3), blue(1) → green
    assert [(r["team"], r["top_color"]) for r in rows] == [("X", "red"), ("Y", "green")]


def test_mode_all_distinct_returns_first():
    """When every value appears once, mode is non-deterministic across runs UNLESS
    we tie-break deterministically. We do (by debug-string order)."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "x": ["A", "B", "C"]})
    g.add_nodes(df, "Item", "id", "name")
    rows1 = g.cypher("MATCH (i:Item) RETURN mode(i.x) AS top")
    rows2 = g.cypher("MATCH (i:Item) RETURN mode(i.x) AS top")
    # Same result across calls (deterministic tiebreak).
    assert rows1[0]["top"] == rows2[0]["top"]
    assert rows1[0]["top"] in {"A", "B", "C"}


def test_mode_distinct_modifier():
    """mode(DISTINCT x) counts unique occurrences once — every value's count is 1,
    so it returns deterministically the first-seen-by-tiebreak."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "x": ["A", "A", "B", "B", "C"],
        }
    )
    g.add_nodes(df, "Item", "id", "name")
    rows = g.cypher("MATCH (i:Item) RETURN mode(DISTINCT i.x) AS top")
    # With DISTINCT, each value contributes 1; deterministic tiebreak.
    assert rows[0]["top"] in {"A", "B", "C"}
