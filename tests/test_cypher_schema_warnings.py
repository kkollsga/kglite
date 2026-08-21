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


# --- projection + direction warnings (eval 2026-08-20 §3a / §3b) ---


def _maritime() -> kglite.KnowledgeGraph:
    """The eval's shape: voyages arriving at ports, vessels whose IMO lives
    under `imo_number`, and a `flag` set on exactly one vessel."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"pid": [1, 2], "name": ["Bergen", "Oslo"]}), "Port", "pid", "name")
    g.add_nodes(pd.DataFrame({"vid": [10, 11], "name": ["V-1", "V-2"]}), "Voyage", "vid", "name")
    g.add_nodes(
        pd.DataFrame(
            {
                "sid": [100, 101, 102],
                "name": ["Nordic", "Baltic", "Arctic"],
                "imo_number": ["9123456", "9234567", "9345678"],
                "flag": ["NO", None, None],
            }
        ),
        "Vessel",
        "sid",
        "name",
    )
    g.add_connections(pd.DataFrame({"s": [10, 11], "d": [1, 2]}), "ARRIVES_AT", "Voyage", "s", "Port", "d")
    return g


def test_projection_of_absent_property_warns(capfd):
    """The eval's `RETURN v.imo`: three rows of nulls, and before this the only
    signal was that the column was empty."""
    g = _maritime()
    result = g.cypher("MATCH (v:Vessel) RETURN v.name, v.imo")
    assert [row["v.imo"] for row in result.to_list()] == [None, None, None]
    warnings = result.diagnostics["warnings"]
    assert any("RETURN projects property 'imo'" in w and "no Vessel node has" in w for w in warnings), warnings
    assert "warning: RETURN projects property 'imo'" in capfd.readouterr().err
    # `imo` is not a typo for `imo_number` by the measured suggestion rule
    # (7 edits), and the rule's whole point is "genuinely close, or silent".
    # A real typo does get the hint.
    typo = g.cypher("MATCH (v:Vessel) RETURN v.imo_numbr").diagnostics["warnings"]
    assert any("Did you mean 'imo_number'?" in w for w in typo), typo
    capfd.readouterr()


def test_sparse_property_does_not_warn():
    """`flag` is set on one vessel of three — sparse, not absent. The metadata
    records it, so the projection is silent (this is the false-positive guard
    that lets the warning be trusted)."""
    g = _maritime()
    diag = g.cypher("MATCH (v:Vessel) RETURN v.flag").diagnostics
    assert diag["warnings"] == []


def test_reversed_relationship_direction_warns(capfd):
    """The eval's zeros table, verbatim: `(p:Port)-[:ARRIVES_AT]->(v:Voyage)`
    counts 0 because every ARRIVES_AT edge runs Voyage→Port."""
    g = _maritime()
    result = g.cypher("MATCH (p:Port)-[:ARRIVES_AT]->(v:Voyage) RETURN count(*) AS n")
    assert result.to_list()[0]["n"] == 0
    warnings = result.diagnostics["warnings"]
    assert any("'ARRIVES_AT'" in w and "Voyage → Port" in w and "matches no edges" in w for w in warnings), warnings
    assert "Reverse the arrow?" in capfd.readouterr().err


def test_correct_relationship_direction_does_not_warn():
    g = _maritime()
    result = g.cypher("MATCH (v:Voyage)-[:ARRIVES_AT]->(p:Port) RETURN count(*) AS n")
    assert result.to_list()[0]["n"] == 2
    assert result.diagnostics["warnings"] == []


def test_new_warnings_survive_the_plan_cache():
    """Same cache-hit contract D1p established, for the two new families."""
    g = _maritime()
    for query in (
        "MATCH (v:Vessel) RETURN v.imo",
        "MATCH (p:Port)-[:ARRIVES_AT]->(v:Voyage) RETURN count(*) AS n",
    ):
        first = g.cypher(query).diagnostics["warnings"]
        second = g.cypher(query).diagnostics["warnings"]
        assert first, query
        assert first == second, query
