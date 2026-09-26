"""`valid_at` / `valid_during` on an element that arrives as a value.

The two-argument forms resolved the element's declaration only from a matched
variable binding, so an element taken from a list — `collect()`, `UNWIND`,
`nodes(p)`, `relationships(p)`, a variable-length relationship list — answered
`true` whatever its interval, and an undeclared type there answered `true`
instead of raising. A null element (an unmatched `OPTIONAL MATCH`) answered
`true` in both forms. Now the declaration is read from the value's own type,
exactly as for a bound variable, and a null element gives null.
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
    # `old` ends 2005, the day `new` starts. `a` holds an R edge valid
    # 2000–2005 to `b` and one valid from 2010 to `c`; U is undeclared.
    g.cypher(
        "CREATE (o:M {name: 'old', vf: date('2000-01-01'), vt: date('2005-01-01')}),"
        "       (n:M {name: 'new', vf: date('2005-01-01')}), (o)-[:SUCC]->(n),"
        "       (a:A {name: 'a'}), (b:B {name: 'b'}), (c:B {name: 'c'}),"
        "       (a)-[:R {vf: date('2000-01-01'), vt: date('2005-01-01')}]->(b),"
        "       (a)-[:R {vf: date('2010-01-01')}]->(c),"
        "       (b)-[:R {vf: date('2000-01-01')}]->(c),"
        "       (a)-[:U {x: 1}]->(b)"
    ).to_list()
    g.cypher("CALL db.temporal.declare({node: 'M', from: 'vf', to: 'vt', convention: 'half_open'})").to_list()
    g.cypher("CALL db.temporal.declare({relationship: 'R', from: 'vf', to: 'vt', convention: 'half_open'})").to_list()
    return g


FORMS = {
    "declared": ("valid_at({x}, '{t}')", "valid_during({x}, '{t}', '{t}')"),
    "named": ("valid_at({x}, '{t}', 'vf', 'vt')", "valid_during({x}, '{t}', '{t}', 'vf', 'vt')"),
}


def _q(g, query: str, naive: bool) -> list[dict]:
    return g.cypher(query, disable_optimizer=naive).to_list()


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("naive", [False, True], ids=["optimized", "naive"])
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("fn", [0, 1], ids=["valid_at", "valid_during"])
def test_a_listed_element_reads_its_own_declaration(storage, naive, form, fn, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    call = FORMS[form][fn]
    at = lambda x, t: call.format(x=x, t=t)  # noqa: E731

    direct = _q(g, f"MATCH (m:M) RETURN m.name AS n, {at('m', '2010')} AS v ORDER BY n", naive)
    expected = [{"n": "new", "v": True}, {"n": "old", "v": False}]
    assert direct == expected
    collected = _q(
        g,
        f"MATCH (m:M) WITH collect(m) AS ms UNWIND ms AS n RETURN n.name AS n, {at('n', '2010')} AS v ORDER BY n",
        naive,
    )
    assert collected == expected
    path_nodes = _q(
        g,
        f"MATCH p = (:M)-[:SUCC]->(:M) UNWIND nodes(p) AS n RETURN n.name AS n, {at('n', '2010')} AS v ORDER BY n",
        naive,
    )
    assert path_nodes == expected
    comprehension = _q(g, f"MATCH p = (:M)-[:SUCC]->(:M) RETURN [n IN nodes(p) | {at('n', '2010')}] AS v", naive)
    assert comprehension == [{"v": [False, True]}]

    # relationships(p) under ALL / ANY, UNWIND and a comprehension.
    all_rels = _q(
        g,
        f"MATCH p = (:A)-[:R]->(x) WHERE ALL(r IN relationships(p) WHERE {at('r', '2003')}) RETURN x.name AS x",
        naive,
    )
    assert all_rels == [{"x": "b"}]
    any_rels = _q(
        g,
        f"MATCH p = (:A)-[:R]->(x) WHERE ANY(r IN relationships(p) WHERE {at('r', '2012')}) RETURN x.name AS x",
        naive,
    )
    assert any_rels == [{"x": "c"}]
    unwound = _q(
        g,
        f"MATCH p = (:A)-[:R]->(x) UNWIND relationships(p) AS r RETURN x.name AS x, {at('r', '2003')} AS v ORDER BY x",
        naive,
    )
    assert unwound == [{"x": "b", "v": True}, {"x": "c", "v": False}]

    # A variable-length relationship list, one and two hops.
    var_length = _q(
        g,
        f"MATCH (:A)-[rs:R*1..2]->(x) RETURN x.name AS x, size(rs) AS h, [r IN rs | {at('r', '2003')}] AS v "
        "ORDER BY x, h",
        naive,
    )
    assert var_length == [
        {"x": "b", "h": 1, "v": [True]},
        {"x": "c", "h": 1, "v": [False]},
        {"x": "c", "h": 2, "v": [True, True]},
    ]


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("naive", [False, True], ids=["optimized", "naive"])
def test_an_undeclared_listed_element_raises_like_a_bound_one(storage, naive, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    undeclared = r"relationship type 'U' has no declared validity interval"
    for query in (
        "MATCH p = (:A)-[:U]->() RETURN [r IN relationships(p) | valid_at(r, '2003')] AS v",
        "MATCH p = (:A)-[:U]->() WHERE ALL(r IN relationships(p) WHERE valid_during(r, '2003', '2004')) RETURN 1",
        "MATCH (:A)-[r:U]->() WITH collect(r) AS rs UNWIND rs AS r RETURN valid_at(r, '2003') AS v",
    ):
        with pytest.raises(kglite.CypherExecutionError, match=undeclared):
            _q(g, query, naive)
    with pytest.raises(kglite.CypherExecutionError, match="node type 'A' has no declared validity interval"):
        _q(g, "MATCH p = (:A)-[:U]->() UNWIND nodes(p) AS n RETURN valid_at(n, '2003') AS v", naive)
    # A value that is no element at all has no type to read.
    with pytest.raises(kglite.CypherExecutionError, match="first argument must be a node or relationship"):
        _q(g, "WITH {vf: date('2000-01-01')} AS m RETURN valid_at(m, '2003') AS v", naive)
    # The named form still reads a map's two keys.
    assert _q(g, "WITH {vf: date('2000-01-01')} AS m RETURN valid_at(m, '2003', 'vf', 'vt') AS v", naive) == [
        {"v": True}
    ]


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("naive", [False, True], ids=["optimized", "naive"])
def test_a_null_element_gives_null(storage, naive, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    miss = "MATCH (a:A) OPTIONAL MATCH (a)-[r:R]->(:B {name: 'zzz'})"
    rows = _q(
        g,
        f"{miss} RETURN valid_at(r, '2003') AS v2, valid_at(r, '2003', 'vf', 'vt') AS v4,"
        " valid_during(r, '2003', '2004') AS d3, valid_during(r, '2003', '2004', 'vf', 'vt') AS d5",
        naive,
    )
    assert rows == [{"v2": None, "v4": None, "d3": None, "d5": None}]
    for form in ("valid_at(r, '2003')", "valid_at(r, '2003', 'vf', 'vt')"):
        assert _q(g, f"{miss} WITH a, r WHERE {form} RETURN count(*) AS n", naive) == [{"n": 0}]
        assert _q(g, f"{miss} WITH a, r WHERE NOT {form} RETURN count(*) AS n", naive) == [{"n": 0}]
    assert _q(g, "OPTIONAL MATCH (z:B {name: 'nope'}) RETURN valid_at(z, '2003', 'vf', 'vt') AS v", naive) == [
        {"v": None}
    ]
    assert _q(g, "UNWIND [null] AS n RETURN valid_at(n, '2003') AS v", naive) == [{"v": None}]


def test_the_appingedam_shape(tmp_path) -> None:
    """The report's registry shape: a place's municipality chain, filtered by
    every relationship on the path being valid in 2015 — the 2021 merger's
    municipality must not appear."""
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (w:Woonplaats {title: 'Appingedam'}),"
        " (old:Municipality {name: 'Appingedam'}), (new:Municipality {name: 'Eemsdelta'}),"
        " (p:Province {name: 'Groningen'}),"
        " (w)-[:IN_MUNICIPALITY {valid_from: date('1900-01-01'), valid_to: date('2021-01-01')}]->(old),"
        " (w)-[:IN_MUNICIPALITY {valid_from: date('2021-01-01')}]->(new),"
        " (old)-[:IN_PROVINCE {valid_from: date('1900-01-01'), valid_to: date('2021-01-01')}]->(p),"
        " (new)-[:IN_PROVINCE {valid_from: date('2021-01-01')}]->(p)"
    ).to_list()
    for rel in ("IN_MUNICIPALITY", "IN_PROVINCE"):
        g.cypher(
            f"CALL db.temporal.declare({{relationship: '{rel}', from: 'valid_from', to: 'valid_to',"
            " convention: 'half_open'})"
        ).to_list()
    query = (
        "MATCH path = (w:Woonplaats {title: 'Appingedam'})-[:IN_MUNICIPALITY]->(m)-[:IN_PROVINCE]->(p) "
        "WHERE ALL(r IN relationships(path) WHERE valid_at(r, '{t}')) RETURN m.name AS m"
    )
    assert g.cypher(query.replace("{t}", "2015")).to_list() == [{"m": "Appingedam"}]
    assert g.cypher(query.replace("{t}", "2022")).to_list() == [{"m": "Eemsdelta"}]
