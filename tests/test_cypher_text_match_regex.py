"""text_match_regex Cypher function — 2026-05-25 broad-scan lift, Batch 3.

Server-side pattern filtering for large-graph queries. Compiled
patterns are cached in `regex_cache.rs` so repeated use in a hot
loop is fast.

Real use case: filter rows by pattern without shipping all rows to
the client — `MATCH (n:Person) WHERE text_match_regex(n.email,
'^[a-z]+@example\\.com$') RETURN n`.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def named_graph():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "code": ["A1", "BB22", "CCC333", "D4D4", "E5"],
            "email": [
                "alice@example.com",
                "BOB@example.com",
                "charlie@other.net",
                "diana@example.com",
                "eve@nope",
            ],
        }
    )
    g.add_nodes(df, "Person", "id", "name")
    return g


# ── basic match ────────────────────────────────────────────────────────


def test_match_returns_boolean(named_graph):
    rows = named_graph.cypher(
        "MATCH (n:Person) WHERE text_match_regex(n.code, '^[A-Z]+\\\\d+$') RETURN n.name AS name ORDER BY n.id"
    )
    # A1, BB22, CCC333, E5 match (single-prefix); D4D4 doesn't (alternating)
    assert [r["name"] for r in rows] == ["Alice", "Bob", "Charlie", "Eve"]


def test_no_match_returns_empty(named_graph):
    rows = named_graph.cypher("MATCH (n:Person) WHERE text_match_regex(n.name, '^xyzzy$') RETURN n.name")
    assert len(rows) == 0


def test_anchored_email_pattern(named_graph):
    rows = named_graph.cypher(
        "MATCH (n:Person) WHERE text_match_regex(n.email, '^[a-z]+@example\\\\.com$') "
        "RETURN n.name AS name ORDER BY n.id"
    )
    # alice + diana match; BOB doesn't (case-sensitive); charlie wrong domain; eve no domain.
    assert [r["name"] for r in rows] == ["Alice", "Diana"]


# ── flags arg ──────────────────────────────────────────────────────────


def test_case_insensitive_via_i_flag(named_graph):
    rows = named_graph.cypher(
        "MATCH (n:Person) WHERE text_match_regex(n.email, '^[a-z]+@example\\\\.com$', 'i') "
        "RETURN n.name AS name ORDER BY n.id"
    )
    # With i flag, BOB now matches too.
    assert [r["name"] for r in rows] == ["Alice", "Bob", "Diana"]


def test_case_insensitive_via_inline_flag(named_graph):
    rows = named_graph.cypher(
        "MATCH (n:Person) WHERE text_match_regex(n.email, '(?i)^[a-z]+@example\\\\.com$') "
        "RETURN n.name AS name ORDER BY n.id"
    )
    # Inline (?i) is equivalent to the 3rd-arg 'i' flag.
    assert [r["name"] for r in rows] == ["Alice", "Bob", "Diana"]


def test_unknown_flag_errors():
    g = kglite.KnowledgeGraph()
    with pytest.raises(Exception, match="unknown flag"):
        g.cypher("RETURN text_match_regex('hi', '.', 'z') AS r")


# ── null + edge cases ──────────────────────────────────────────────────


def test_null_text_returns_null():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN text_match_regex(null, '.*') AS r")
    assert rows[0]["r"] is None


def test_null_pattern_returns_null():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN text_match_regex('hello', null) AS r")
    assert rows[0]["r"] is None


def test_invalid_pattern_errors():
    g = kglite.KnowledgeGraph()
    with pytest.raises(Exception, match="invalid pattern"):
        g.cypher("RETURN text_match_regex('hi', '(?P<bad') AS r")


def test_wrong_arg_count_errors():
    g = kglite.KnowledgeGraph()
    with pytest.raises(Exception, match="requires 2 or 3 args"):
        g.cypher("RETURN text_match_regex('hi') AS r")


# ── cache hit performance signal (best-effort) ─────────────────────────


def test_repeated_pattern_use_doesnt_explode(named_graph):
    """If pattern compilation weren't cached, this would be slow. Smoke test."""
    # Repeat the same query 50 times — same pattern compiles once.
    for _ in range(50):
        rows = named_graph.cypher("MATCH (n:Person) WHERE text_match_regex(n.code, '^[A-Z]+\\\\d+$') RETURN n.id")
        assert len(rows) == 4
