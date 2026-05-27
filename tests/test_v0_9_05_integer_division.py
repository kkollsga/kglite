"""0.9.0 gate item §5 — integer division correctness.

Currently `1967 / 10 → 196.7` (always promoted to Float64). Neo4j /
openCypher: int÷int = int (truncated, → 196). Promote to float only
when either operand is float.

These tests are xfail-strict: when the implementation lands, pytest
flips them to XPASS and the strict marker forces removal of the
`@pytest.mark.xfail` decorator. That is the signal to update the
gate-item status in `dev-documentation/archive/0.9.0-readiness.md` §5
(local-only; 0.9.0 has shipped).
"""

from __future__ import annotations

import pytest

import kglite

NOT_IMPLEMENTED = "0.9.0 §5 — int÷int currently promotes to float; flip when fixed."


def test_int_div_int_returns_int():
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN 1967 / 10 AS d"))
    assert rows[0]["d"] == 196
    assert isinstance(rows[0]["d"], int), f"expected int, got {type(rows[0]['d']).__name__}"


def test_decade_bucketing_pattern():
    """The Sodir creaming-curve query shape: bucket year into decade."""
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN 1967 / 10 * 10 AS decade"))
    assert rows[0]["decade"] == 1960


def test_int_div_negative():
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN -7 / 2 AS d"))
    # Neo4j: truncates toward zero, so -7/2 = -3 (not -4)
    assert rows[0]["d"] == -3


def test_int_modulo():
    """Already passes — `%` returns int for int operands. Locks the
    behavior so the int÷int fix doesn't accidentally regress modulo.
    """
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN 7 % 3 AS m"))
    assert rows[0]["m"] == 1
    assert isinstance(rows[0]["m"], int)


def test_float_div_int_returns_float():
    """Already passes — included as the canonical 'opposite' shape so
    we lock in the cross-type behavior alongside the int÷int fix.
    """
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN 1967.0 / 10 AS d"))
    assert rows[0]["d"] == pytest.approx(196.7)


def test_int_div_explicit_float_via_tofloat():
    """Already passes — the documented current workaround. Locks it
    in as the supported path so any int-division fix doesn't
    accidentally regress this.
    """
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN 1967 / toFloat(10) AS d"))
    assert rows[0]["d"] == pytest.approx(196.7)
