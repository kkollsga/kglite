"""0.9.0 gate item §1 — better Cypher error messages.

Pre-§1, the parser emitted token-level errors with no source
positions and no distinction between "you typo'd" and "feature not
yet supported". The agent's recent UX feedback singled this out as
the gating item that compounds with §2/§3/§6 — when the user can't
tell "wrong" from "not implemented", every gap feels twice as rough.

Post-§1 baseline (this iteration):

1. Every parse error now carries `(line N col M)` plus a source
   excerpt with a `^` caret pointing at the failing column.
2. `§2/§3/§6` queries that previously errored with token-level
   messages now succeed (because those features are implemented) —
   the intent-message UX they targeted is moot for those exact
   queries. Future unimplemented features will pick up the
   `intent_level_rewrite` machinery in `parser/mod.rs`.

Positions are byte-precise as of 0.9.0 Cluster 3 — the tokenizer
attaches a char offset to every token, the parser threads them
through, and `format_parse_error` walks `input.chars()` to convert
to (line, col) on the error path. Earlier shipped variant used a
re-walk estimator; that's gone.
"""

from __future__ import annotations

import pytest

import kglite


def _try_cypher_capture_error(g: kglite.KnowledgeGraph, query: str) -> str:
    """Run a query expected to fail; return the error message string."""
    try:
        list(g.cypher(query))
    except Exception as e:  # noqa: BLE001 — we want any error
        return str(e)
    pytest.fail(f"expected an error from query: {query!r}")


# ---------------------------------------------------------------------------
# Source offsets — every parser error should know where it is.
# ---------------------------------------------------------------------------


def test_error_carries_line_number():
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "MATCH (n)\nRETURN n.))")
    assert "line 2" in msg.lower() or "2:" in msg, f"missing line info: {msg!r}"


def test_error_carries_column_number():
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "RETURN ))")
    assert "col" in msg.lower() or "column" in msg.lower() or "^" in msg, f"missing column info: {msg!r}"


def test_error_carries_caret_excerpt():
    """Errors include a single-line source excerpt with a `^` caret."""
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "MATCH (n)\nRETURN n.))")
    assert "^" in msg, f"missing caret marker: {msg!r}"


# ---------------------------------------------------------------------------
# Locked-in: §2/§3/§6 queries now succeed (intent-message tests
# converted from "must error with intent message" to "must succeed",
# since the features they probed are implemented).
# ---------------------------------------------------------------------------


def test_nulls_last_query_now_succeeds():
    """§2 landed — `ORDER BY x DESC NULLS LAST` parses and runs."""
    g = kglite.KnowledgeGraph()
    # Empty graph, but the parser path is what we're locking in.
    list(g.cypher("MATCH (n) RETURN n ORDER BY n.title DESC NULLS LAST"))


def test_datetime_accessor_query_now_succeeds():
    """§3 landed — `n.d.year` parses and runs (returns Null on
    nodes without a `joined` property; the parser path is what we're
    locking in)."""
    g = kglite.KnowledgeGraph()
    list(g.cypher("MATCH (n:X) RETURN n.joined.year AS y"))


def test_size_pattern_expression_query_now_succeeds():
    """§6 landed — `size((:A)-[:R]->(:B))` parses and runs."""
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN size((:A)-[:R]->(:B)) AS n"))
    assert rows[0]["n"] == 0  # empty graph


# ---------------------------------------------------------------------------
# Caret pointing for typos.
# ---------------------------------------------------------------------------


def test_caret_or_column_for_typo_extra_paren():
    """Typo'd extra `)` in CASE — error locates the bad `)`."""
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "MATCH (n:X) RETURN CASE WHEN count(n)) THEN 1 END")
    assert "^" in msg or "col" in msg.lower() or "column" in msg.lower(), f"no positional info on typo error: {msg!r}"


# Intent-level rewrites — `intent_level_rewrite` in `parser/mod.rs`
# is the hook for future features. Currently no stable
# "not-yet-implemented" feature to lock in (the named candidates —
# variable-length paths, quantified relationships — already parse
# without error). The hook returns None for everything today; new
# §X work can plug in detection there as features land.


# ---------------------------------------------------------------------------
# The message body itself — no Rust shapes, and the escape hatch is named.
# ---------------------------------------------------------------------------


def test_parse_error_names_the_token_the_way_the_user_wrote_it():
    """`SET 1 = 2` used to report `got Some(IntLit(1))`.

    `Option` and enum-variant spellings are Rust internals; a user reading
    the message has no way to map `IntLit(1)` back to what they typed.
    """
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "MATCH (n) SET 1 = 2")
    assert "Expected variable name in SET, got 1" in msg, msg
    assert "Some(" not in msg and "IntLit" not in msg, msg

    msg = _try_cypher_capture_error(g, "MATCH (n) SET")
    assert "got end of input" in msg, msg


def test_reserved_keyword_in_a_name_position_points_at_backticks():
    """Backticks make a reserved word usable as a name; the error says so.

    `MATCH (match:P)` reported only `Unexpected token in MATCH pattern:
    Match`, which states the rule and hides the one-character fix.
    """
    g = kglite.KnowledgeGraph()
    for query in (
        "MATCH (match:P) RETURN match",
        "MATCH (n) RETURN n.match",
        "CREATE (n:P {match: 1})",
    ):
        msg = _try_cypher_capture_error(g, query)
        assert "reserved keyword" in msg and "backtick" in msg, f"{query}: {msg}"

    # The hint is not sprayed onto errors it does not apply to...
    assert "backtick" not in _try_cypher_capture_error(g, "MATCH (n) SET 1 = 2")
    # ...and the fix it advertises works.
    list(g.cypher("MATCH (`match`:P) RETURN `match`"))
    list(g.cypher("MATCH (n) RETURN n.`match`"))


def test_block_comment_is_rejected_by_name():
    """`/* ... */` is not supported; the message says so and points at `//`.

    It used to tokenize as a division sign, so the user saw
    `Unexpected token at start of clause: Slash`.
    """
    g = kglite.KnowledgeGraph()
    for query in (
        "/* hi */ RETURN 1 AS a",
        "RETURN 1 AS a /* hi */",
        "RETURN /* hi */ 1 AS a",
    ):
        msg = _try_cypher_capture_error(g, query)
        assert "Block comments" in msg, f"{query}: {msg}"
        assert "//" in msg, f"{query}: {msg}"
        assert "^" in msg and "col" in msg.lower(), f"{query}: {msg}"

    # Line comments, `/*` inside a string, and division are unaffected.
    assert list(g.cypher("// leading\nRETURN 1 AS a"))[0]["a"] == 1
    assert list(g.cypher("RETURN '/*' AS a"))[0]["a"] == "/*"
    assert list(g.cypher("RETURN 6 / 2 AS a"))[0]["a"] == 3
