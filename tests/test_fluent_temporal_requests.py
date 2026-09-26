"""An explicit fluent validity request raises where Cypher's does.

`valid_at()` / `valid_during()` and `traverse(at=, during=)` on a type with no
declared interval, or with a misspelled bound name, used to read the missing
property as an open bound and keep every element. They now resolve the bounds
through the one core rule Cypher's `valid_at` uses: a named bound must exist on
the type, an unnamed one comes from the declaration (or, for nodes, the
documented `date_from` / `date_to` default, which must then exist), and the
error names the type and the fix. The ambient `date()` context is not a
request: it still filters only the declared types.
"""

from __future__ import annotations

import pytest

import kglite

MODES = [None, "mapped", "disk"]
MODE_IDS = ["memory", "mapped", "disk"]


def _graph(storage, tmp_path) -> kglite.KnowledgeGraph:
    if storage == "disk":
        g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    elif storage == "mapped":
        g = kglite.KnowledgeGraph(storage="mapped")
    else:
        g = kglite.KnowledgeGraph()
    # F is undeclared: one row 2000–2005, one from 2010. P is a target.
    g.cypher(
        "CREATE (a:F {name: 'a', valid_from: date('2000-01-01'), valid_to: date('2005-01-01')}),"
        "       (b:F {name: 'b', valid_from: date('2010-01-01')}), (p:P {name: 'p'}),"
        "       (a)-[:IN {valid_from: date('2000-01-01'), valid_to: date('2005-01-01')}]->(p),"
        "       (b)-[:IN {valid_from: date('2010-01-01')}]->(p)"
    ).to_list()
    return g


def _names(kg) -> list[str]:
    return sorted(n["name"] for n in kg.collect())


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
def test_an_undeclared_type_raises_naming_the_fix(storage, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    undeclared = r"node type 'F' has no declared validity interval.*set_temporal\('F'"
    with pytest.raises(ValueError, match=undeclared):
        g.select("F").valid_at("2003")
    with pytest.raises(ValueError, match=undeclared):
        g.select("F").valid_during("2001", "2002")
    # Cypher agrees.
    with pytest.raises(kglite.CypherExecutionError, match="node type 'F' has no declared validity interval"):
        g.cypher("MATCH (f:F) WHERE valid_at(f, '2003') RETURN f").to_list()


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
def test_a_misspelled_bound_raises(storage, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    missing = r"property 'validfrom' does not exist on node type 'F'"
    with pytest.raises(ValueError, match=missing):
        g.select("F").valid_at("2003", date_from_field="validfrom", date_to_field="valid_to")
    with pytest.raises(ValueError, match=missing):
        g.select("F").valid_during("2001", "2002", date_from_field="validfrom", date_to_field="valid_to")
    with pytest.raises(kglite.CypherExecutionError, match=missing):
        g.cypher("MATCH (f:F) WHERE valid_at(f, '2003', 'validfrom', 'valid_to') RETURN f").to_list()
    # One side named, the other defaulting to a property the type lacks.
    with pytest.raises(ValueError, match="no 'valid_from' / 'date_to' properties"):
        g.select("F").valid_at("2003", date_from_field="valid_from")


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
def test_named_and_declared_bounds_filter(storage, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    named = {"date_from_field": "valid_from", "date_to_field": "valid_to"}
    assert _names(g.select("F").valid_at("2003", **named)) == ["a"]
    assert _names(g.select("F").valid_during("2011", "2012", **named)) == ["b"]
    g.set_temporal("F", "valid_from", "valid_to", convention="half_open")
    # `temporal=False`: the ambient context (today) would otherwise drop `a`.
    everything = g.select("F", temporal=False)
    assert _names(everything.valid_at("2003")) == ["a"]
    # Half-open: the `to` day is not valid.
    assert _names(everything.valid_at("2005-01-01")) == []
    assert _names(everything.valid_during("2004", "2010")) == ["a", "b"]


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
def test_traverse_with_a_date_on_an_undeclared_relationship_raises(storage, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    undeclared = r"traverse\(at=\.\.\.\): relationship type 'IN' has no declared validity interval"
    with pytest.raises(kglite.ArgumentError, match=undeclared):
        g.select("F", temporal=False).traverse("IN", at="2003")
    with pytest.raises(kglite.ArgumentError, match=r"traverse\(during=\.\.\.\)"):
        g.select("F", temporal=False).traverse("IN", during=("2001", "2002"))
    # A traversal that visits no edge of the type has nothing to refuse.
    empty = g.select("F").where({"name": "zzz"})
    assert empty.traverse("IN", at="2003").len() == 0
    # Declared, the date filters.
    g.set_temporal("IN", "valid_from", "valid_to", convention="half_open")
    assert g.select("F", temporal=False).traverse("IN", at="2003").len() == 1
    assert g.select("F", temporal=False).traverse("IN", at="2012").len() == 1
    assert g.select("F", temporal=False).traverse("IN", at="2007").len() == 0


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
def test_the_ambient_date_context_leaves_undeclared_types_alone(storage, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    assert _names(g.date("2003").select("F")) == ["a", "b"]
    assert g.date("2003").select("F").traverse("IN").len() == 2
