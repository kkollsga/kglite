"""Golden assertions for Cypher boolean-operator precedence.

openCypher binds **NOT tighter than AND tighter than OR** (and comparison
tighter than all three). These are *correctness* assertions (expected result
sets, not optimised-vs-naive consistency) — the parser layer the differential
corpus structurally can't cover, since a precedence bug is present in every
pass-config / storage-mode identically.

Guards the investigation of the reported "NOT binds too loosely vs AND" bug
(suspected `NOT a = b AND c = d` parsing as `NOT (a = b AND c = d)`): a
26-shape stress sweep on 0.11.2 found it *not reproducible* — every shape below
parses per the openCypher spec. This file keeps it that way.

Run: pytest tests/test_cypher_operator_precedence.py
"""

from __future__ import annotations

import pytest

from kglite import KnowledgeGraph


@pytest.fixture(scope="module")
def grid() -> KnowledgeGraph:
    """3×3 grid of (x, y) ∈ {1,2,3}² — one node per combination."""
    g = KnowledgeGraph()
    nid = 0
    for x in (1, 2, 3):
        for y in (1, 2, 3):
            g.cypher("CREATE (:N {id: $i, x: $x, y: $y})", params={"i": nid, "x": x, "y": y})
            nid += 1
    return g


# (where_clause, expected match count) — expected computed by hand per the
# openCypher precedence NOT > comparison > AND > OR over the 9-node grid.
PRECEDENCE_CASES = [
    ("NOT n.x = 1 AND n.y = 2", 2),  # (NOT x=1) AND y=2  — the reported shape
    ("n.x = 1 AND NOT n.y = 2", 2),  # NOT on the right
    ("NOT n.x = 1 OR n.y = 2", 7),  # (NOT x=1) OR y=2
    ("NOT n.x = 1 AND NOT n.y = 2", 4),
    ("NOT (n.x = 1 AND n.y = 2)", 8),  # explicit grouping (control)
    ("NOT n.x IN [1] AND n.y = 2", 2),  # NOT … IN …
    ("NOT n.x = 1 AND n.y = 2 OR n.x = 3", 4),  # ((NOT x=1) AND y=2) OR x=3
    ("NOT n.x < 2", 6),  # NOT over a comparison
    ("NOT NOT n.x = 1", 3),  # double negation
    ("NOT n.x = 1 OR n.y = 2 AND n.x = 3", 6),  # AND binds tighter than OR
    ("(NOT n.x = 1 OR n.y = 1) AND n.x = 2", 3),
]


@pytest.mark.parametrize("where,expected", PRECEDENCE_CASES, ids=[c[0] for c in PRECEDENCE_CASES])
def test_boolean_operator_precedence(grid, where, expected):
    got = grid.cypher(f"MATCH (n:N) WHERE {where} RETURN count(n) AS c").scalar()
    assert got == expected, f"`WHERE {where}` returned {got}, expected {expected} (openCypher NOT>AND>OR)"


def test_reported_not_and_string_shape():
    """The exact shape from the bug report (string equality + AND), confirmed
    to parse as `(NOT level='hoyesterett') AND year=2024` — i.e. it excludes
    hoyesterett rows, not keeps them."""
    g = KnowledgeGraph()
    g.cypher("CREATE (:C {id: 0, level: 'hoyesterett', year: 2024})")  # excluded by NOT
    g.cypher("CREATE (:C {id: 1, level: 'lagmannsrett', year: 2024})")  # matches
    g.cypher("CREATE (:C {id: 2, level: 'lagmannsrett', year: 2020})")  # excluded by year
    levels = g.cypher("MATCH (c:C) WHERE NOT c.level = 'hoyesterett' AND c.year = 2024 RETURN c.level AS lvl").column(
        "lvl"
    )
    assert levels == ["lagmannsrett"]
