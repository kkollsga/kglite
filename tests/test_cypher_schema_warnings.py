"""Non-fatal schema warnings for MATCH typos (DX / Lever 2).

A MATCH against an unknown node label or relationship type is legal Cypher
(it returns zero rows — a valid existence check), so kglite does *not* error.
But an unknown type is almost always a typo, the most common "why is my query
empty?" foot-gun, so the engine emits a non-fatal `warning:` to stderr with an
edit-distance "did you mean?" hint. These tests use `capfd` because the
warning is emitted at the fd level from the Rust extension.
"""

from __future__ import annotations

import pandas as pd

import kglite


def _graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"gid": [1, 2], "name": ["a", "b"]}), "Person", "gid", "name")
    g.add_connections(pd.DataFrame({"s": [1], "d": [2]}), "KNOWS", "Person", "s", "Person", "d")
    return g


def test_unknown_label_warns_with_hint(capfd):
    g = _graph()
    result = g.cypher("MATCH (n:Persn) RETURN n")
    assert len(result.to_list()) == 0  # still a valid, zero-row query
    err = capfd.readouterr().err
    assert "unknown node label 'Persn'" in err
    assert "Did you mean 'Person'?" in err


def test_unknown_relationship_warns_with_hint(capfd):
    g = _graph()
    g.cypher("MATCH (a:Person)-[:KNOWZ]->(b) RETURN a")
    err = capfd.readouterr().err
    assert "unknown relationship type 'KNOWZ'" in err
    assert "Did you mean 'KNOWS'?" in err


def test_valid_query_emits_no_warning(capfd):
    g = _graph()
    g.cypher("MATCH (a:Person)-[:KNOWS]->(b) RETURN a")
    err = capfd.readouterr().err
    assert "unknown node label" not in err
    assert "unknown relationship type" not in err


# --- structured warnings via diagnostics() (agent-visible; no stderr needed) ---


def test_diagnostics_exposes_warnings():
    g = _graph()
    diag = g.cypher("MATCH (n:Persn) RETURN n").diagnostics
    assert diag is not None
    warnings = diag["warnings"]
    assert any("unknown node label 'Persn'" in w and "Did you mean 'Person'?" in w for w in warnings)


def test_diagnostics_warnings_empty_for_clean_query():
    g = _graph()
    diag = g.cypher("MATCH (a:Person)-[:KNOWS]->(b) RETURN a").diagnostics
    assert diag is not None
    assert diag["warnings"] == []
