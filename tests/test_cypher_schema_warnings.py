"""Non-fatal schema warnings for MATCH typos (DX / Lever 2).

A MATCH against an unknown node label or relationship type is legal Cypher
(it returns zero rows — a valid existence check), so kglite does *not* error.
But an unknown type is almost always a typo, the most common "why is my query
empty?" foot-gun, so the engine emits a non-fatal `warning:` to stderr with an
edit-distance "did you mean?" hint. These tests use `capfd` because the
warning is emitted at the fd level from the Rust extension.

The same warnings are also carried structurally on ``ResultView.diagnostics``
— one computation in the engine, two consumers (stderr for interactive users,
the field for programmatic and agent callers).
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


def test_mixed_alternation_warning_does_not_claim_no_rows(capfd):
    """An alternation matches through ANY branch, so one unknown branch is not "no rows"."""
    g = _graph()
    result = g.cypher("MATCH (a:Person)-[:MENTORS|KNOWS]->(b) RETURN a")
    assert len(result.to_list()) == 1  # the KNOWS branch matches
    err = capfd.readouterr().err
    assert "unknown relationship type 'MENTORS'" in err
    assert "returns no rows" not in err
    assert "can still return rows via 'KNOWS'" in err


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


def test_diagnostics_shape_is_unchanged():
    """The dict keys are a pinned contract; the engine populating `warnings`
    must not have moved anything else."""
    g = _graph()
    diag = g.cypher("MATCH (n:Persn) RETURN n").diagnostics
    assert set(diag) == {"elapsed_ms", "timed_out", "timeout_ms", "warnings"}
    assert isinstance(diag["elapsed_ms"], int)
    assert diag["timed_out"] is False
    # The wheel still reports the caller-configured deadline (the in-memory
    # default is 180 s), not the instant core derived it from.
    assert diag["timeout_ms"] == 180_000
    assert all(isinstance(w, str) for w in diag["warnings"])


def test_diagnostics_warnings_survive_the_plan_cache():
    """The second run of a query is served from the plan cache, which returns
    before the schema pass. The warning must survive that — this is where the
    engine-side fix could regress silently, since run 1 keeps working."""
    g = _graph()
    query = "MATCH (n:Persn) RETURN n"
    first = g.cypher(query).diagnostics["warnings"]
    second = g.cypher(query).diagnostics["warnings"]
    assert first, first
    assert first == second


def test_mutation_diagnostics_carry_the_read_pattern_warning():
    """`MATCH (n:typo) SET ...` silently updates nothing — the same foot-gun in
    write clothing. A CREATE of an unseen type is not a typo and stays clean."""
    g = _graph()
    diag = g.cypher("MATCH (n:Persn) SET n.flag = true").diagnostics
    assert any("unknown node label 'Persn'" in w for w in diag["warnings"])
    clean = g.cypher("CREATE (:BrandNewType {gid: 99})").diagnostics
    assert clean["warnings"] == []


def test_procedure_scope_warning_reaches_diagnostics(capfd):
    """Procedure scoping is validated during execution, not at plan time, so
    its warning travels a different route to the same field."""
    g = _graph()
    diag = g.cypher("CALL pagerank({relationship: 'KNOWZ'}) YIELD node RETURN count(*) AS c").diagnostics
    capfd.readouterr()
    assert any("unknown relationship type 'KNOWZ'" in w for w in diag["warnings"]), diag
