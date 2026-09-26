"""Cypher `valid_at` / `valid_during` follow a type's declared interval.

They used to read every interval closed, whatever the declaration said, so on a
registry whose periods end on their successor's start day (`half_open`) the
boundary day counted both the old and the new entity — 51/127 yearly counts
right where the fluent filters got 115. Now:

- `valid_at(x, t)` / `valid_during(x, a, b)` read the declared bounds and
  convention, and raise on a type with no declaration;
- the forms that name the bounds follow the declaration when it names the same
  two properties, and read an undeclared pair closed;
- a relationship takes its source type's keyed declaration first.
"""

from __future__ import annotations

import pytest

import kglite

MODES = [None, "mapped", "disk"]
MODE_IDS = ["memory", "mapped", "disk"]


def _graph(storage, tmp_path, convention: str) -> kglite.KnowledgeGraph:
    if storage == "disk":
        g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / f"g-{convention}"))
    elif storage == "mapped":
        g = kglite.KnowledgeGraph(storage="mapped")
    else:
        g = kglite.KnowledgeGraph()
    # Scheemda (0039) ends 2010-01-01, the day its successor Oldambt (1895)
    # starts; Adorp (0001) ended in 1990. `a`/`b` are a second, undeclared pair.
    g.cypher(
        "CREATE (:M {code: '0039', vf: date('1900-01-01'), vt: date('2010-01-01'),"
        "            a: date('1900-01-01'), b: date('2010-01-01')}),"
        "       (:M {code: '1895', vf: date('2010-01-01'), a: date('2010-01-01')}),"
        "       (:M {code: '0001', vf: date('1900-01-01'), vt: date('1990-01-01'),"
        "            a: date('1900-01-01'), b: date('1990-01-01')})"
    ).to_list()
    g.cypher(f"CALL db.temporal.declare({{node: 'M', from: 'vf', to: 'vt', convention: '{convention}'}})").to_list()
    return g


def _codes(g, where: str, disable_optimizer: bool = False) -> list[str]:
    rows = g.cypher(
        f"MATCH (m:M) WHERE {where} RETURN m.code AS c ORDER BY c", disable_optimizer=disable_optimizer
    ).to_list()
    return [r["c"] for r in rows]


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("disable_optimizer", [False, True], ids=["optimized", "naive"])
@pytest.mark.parametrize(
    ("convention", "on_boundary"),
    [("half_open", ["1895"]), ("closed", ["0039", "1895"])],
)
def test_the_boundary_day_follows_the_declared_convention(
    storage, disable_optimizer, convention, on_boundary, tmp_path
) -> None:
    g = _graph(storage, tmp_path, convention)
    day = "date('2010-01-01')"
    two_arg = _codes(g, f"valid_at(m, {day})", disable_optimizer)
    four_arg = _codes(g, f"valid_at(m, {day}, 'vf', 'vt')", disable_optimizer)
    assert two_arg == four_arg == on_boundary
    assert g.date("2010-01-01").select("M").len() == len(on_boundary)
    assert _codes(g, "valid_at(m, date('2009-12-31'))", disable_optimizer) == ["0039"]
    during = _codes(g, "valid_during(m, date('2010-01-01'), date('2010-06-30'))", disable_optimizer)
    assert during == on_boundary
    assert during == _codes(g, "valid_during(m, date('2010-01-01'), date('2010-06-30'), 'vf', 'vt')")
    count = g.cypher(
        f"MATCH (m:M) WHERE valid_at(m, {day}) RETURN count(*) AS c", disable_optimizer=disable_optimizer
    ).to_list()
    assert count == [{"c": len(on_boundary)}]


def test_an_undeclared_pair_reads_closed(tmp_path) -> None:
    g = _graph(None, tmp_path, "half_open")
    assert _codes(g, "valid_at(m, date('2010-01-01'), 'a', 'b')") == ["0039", "1895"]


def test_the_short_form_needs_a_declaration(tmp_path) -> None:
    g = _graph(None, tmp_path, "half_open")
    g.cypher("CREATE (:P {vf: date('2000-01-01')})").to_list()
    with pytest.raises(
        kglite.CypherExecutionError, match=r"node type 'P' has no declared validity interval.*db\.temporal\.declare"
    ):
        g.cypher("MATCH (p:P) WHERE valid_at(p, date('2010-01-01')) RETURN count(*)").to_list()
    with pytest.raises(kglite.CypherExecutionError, match="valid_during"):
        g.cypher("MATCH (p:P) WHERE valid_during(p, date('2010-01-01'), date('2011-01-01')) RETURN p").to_list()


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
def test_a_relationship_takes_its_source_types_keyed_declaration_first(storage, tmp_path) -> None:
    if storage == "disk":
        g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "rel"))
    elif storage == "mapped":
        g = kglite.KnowledgeGraph(storage="mapped")
    else:
        g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (f:Field {name: 'f'}), (w:Well {name: 'w'}), (c:Company {name: 'c'}),"
        " (f)-[:HAS {vf: date('2000-01-01'), vt: date('2010-01-01')}]->(c),"
        " (w)-[:HAS {vf: date('2000-01-01'), vt: date('2010-01-01')}]->(c)"
    ).to_list()
    g.cypher(
        "CALL db.temporal.declare({relationship: 'HAS', source_type: 'Field', from: 'vf', to: 'vt',"
        " convention: 'half_open'})"
    ).to_list()
    g.cypher("CALL db.temporal.declare({relationship: 'HAS', from: 'vf', to: 'vt', convention: 'closed'})").to_list()
    for form in ("valid_at(r, date('2010-01-01'))", "valid_at(r, date('2010-01-01'), 'vf', 'vt')"):
        rows = g.cypher(f"MATCH (s)-[r:HAS]->() WHERE {form} RETURN s.name AS s ORDER BY s").to_list()
        # The Field edge is half-open (keyed), the Well edge closed (unkeyed).
        assert [r["s"] for r in rows] == ["w"], form
