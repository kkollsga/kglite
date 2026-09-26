"""Golden results for NULL handling in predicates pushed into MATCH patterns.

A pushed node matcher is the only enforcement of its conjunct once another
conjunct keeps the residual WHERE alive, and a fused scan drops a fully
subsumed WHERE altogether. So every rule here must answer exactly what the
unoptimised WHERE answers, in every storage mode.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

STORAGE_MODES = ("memory", "mapped", "disk")


def _new_graph(mode: str, tmp_path) -> kglite.KnowledgeGraph:
    if mode == "memory":
        return kglite.KnowledgeGraph()
    if mode == "mapped":
        return kglite.KnowledgeGraph(storage="mapped")
    return kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))


@pytest.fixture(params=STORAGE_MODES, ids=STORAGE_MODES)
def people(request, tmp_path):
    g = _new_graph(request.param, tmp_path)
    df = pd.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "name": ["a", "b", "c", "d"],
            "age": [20, 30, 40, 50],
        }
    )
    g.add_nodes(df, "P", "pid", "name")
    return g


# ── A NULL comparison value is never true ────────────────────────────────────
#
# `n.age > null` is null for every row. The unpushable regex conjunct keeps a
# residual WHERE, so the pushed matcher alone decides the comparison.

NULL_COMPARISONS = [
    ("MATCH (n:P) WHERE n.age > $x AND toString(n.id) =~ '.*' RETURN count(*) AS c", {"x": None}),
    ("MATCH (n:P) WHERE n.age >= null AND toString(n.id) =~ '.*' RETURN count(*) AS c", None),
    ("MATCH (n:P) WHERE $x < n.age AND toString(n.id) =~ '.*' RETURN count(*) AS c", {"x": None}),
    ("MATCH (n:P) WHERE n.age <= $x AND toString(n.id) =~ '.*' RETURN count(*) AS c", {"x": None}),
    ("MATCH (n:P) WHERE n.age > $x RETURN count(*) AS c", {"x": None}),
    ("MATCH (n:P) WHERE n.age > 10 AND n.age < $x RETURN count(*) AS c", {"x": None}),
]


@pytest.mark.parametrize("query,params", NULL_COMPARISONS)
def test_null_comparison_value_matches_nothing(people, query, params):
    assert people.cypher(query, params=params).to_list() == [{"c": 0}]


def test_null_comparison_grouped_aggregate_is_empty(people):
    rows = people.cypher("MATCH (n:P) WHERE n.age > null RETURN n.name AS name, count(n) AS k").to_list()
    assert rows == []


# ── `x.p IS NULL OR x.p <op> c` and `coalesce(x.p, c0) <op> c` ───────────────
#
# Six spans at t = 150: vt is NULL on 3, absent on 4, and the string 'open' on
# 5 — `'open' >= 150` is null, so 5 is never an "IS NULL" row nor a match.

SPANS = [
    # (k, vf, vt)            vt: absent when the key is missing
    {"k": 1, "vf": 0, "vt": 99},
    {"k": 2, "vf": 100, "vt": 199},
    {"k": 3, "vf": 200, "vt": None},
    {"k": 4, "vf": 150},
    {"k": 5, "vf": 120, "vt": "open"},
    {"k": 6, "vf": 50, "vt": 150},
]


def _create_spans(g: kglite.KnowledgeGraph) -> None:
    for span in SPANS:
        props = ", ".join(f"{key}: ${key}" for key in span)
        g.cypher(f"CREATE (:Span {{{props}}})", params=span)
    g.cypher("CREATE (:Anchor {k: 0})")
    for span in SPANS:
        props = ", ".join(f"{key}: ${key}" for key in span if key != "k")
        g.cypher(
            f"MATCH (a:Anchor), (s:Span {{k: $k}}) CREATE (a)-[:NEXT {{{props}}}]->(s)",
            params=span,
        )


@pytest.fixture(
    params=[(mode, indexed) for mode in STORAGE_MODES for indexed in (False, True)],
    ids=[f"{mode}-{'ranged' if indexed else 'plain'}" for mode in STORAGE_MODES for indexed in (False, True)],
)
def spans(request, tmp_path):
    mode, indexed = request.param
    g = _new_graph(mode, tmp_path)
    _create_spans(g)
    if indexed:
        g.create_range_index("Span", "vf")
        g.create_range_index("Span", "vt")
    return g


T = {"t": 150}

NODE_SHAPES = [
    ("s.vf <= 150 AND (s.vt IS NULL OR s.vt >= 150)", None, [2, 4, 6]),
    ("s.vf <= 150 AND (150 <= s.vt OR s.vt IS NULL)", None, [2, 4, 6]),
    ("s.vf <= 150 AND (s.vt IS NULL OR s.vt >= $t)", T, [2, 4, 6]),
    ("s.vf <= 150 AND ($t <= s.vt OR s.vt IS NULL)", T, [2, 4, 6]),
    ("s.vt IS NULL OR s.vt >= 150", None, [2, 3, 4, 6]),
    ("(s.vt IS NULL OR s.vt > 150) AND s.vf <= 150", None, [2, 4]),
    ("s.vf <= 150 AND (s.vt IS NULL OR s.vt < 150)", None, [1, 4]),
    ("s.vf <= 150 AND coalesce(s.vt, 1000) >= 150", None, [2, 4, 6]),
    ("s.vf <= 150 AND 150 <= coalesce(s.vt, $max)", {"max": 1000}, [2, 4, 6]),
    ("coalesce(s.vt, 1000) >= 150", None, [2, 3, 4, 6]),
    ("s.vf <= 150 AND coalesce(s.vt, 0) >= 150", None, [2, 6]),
    ("s.vf <= 150 AND coalesce(s.vt, null) <= 150", None, [1, 6]),
    ("s.vf <= 150 AND coalesce(s.vt, 'x') >= 150", None, [2, 6]),
    ("s.vf <= 150 AND NOT (s.vt IS NULL OR s.vt >= 150)", None, [1]),
    ("s.vf <= 150 AND (s.vt IS NOT NULL OR s.vt >= 150)", None, [1, 2, 5, 6]),
    ("s.vt <= 150 AND (s.vt IS NULL OR s.vt >= 150)", None, [6]),
    ("s.vf <= 150 AND (s.vt IS NULL OR s.vt >= $t)", {"t": None}, [4]),
]


@pytest.mark.parametrize("where,params,expected", NODE_SHAPES)
def test_node_null_or_shapes(spans, where, params, expected):
    rows = spans.cypher(f"MATCH (s:Span) WHERE {where} RETURN s.k AS k ORDER BY k", params=params).to_list()
    assert [r["k"] for r in rows] == expected


@pytest.mark.parametrize("where,params,expected", NODE_SHAPES)
def test_node_null_or_shapes_through_the_fused_aggregate(spans, where, params, expected):
    rows = spans.cypher(
        f"MATCH (s:Span) WHERE {where} RETURN s.k % 2 AS parity, count(s) AS n ORDER BY parity",
        params=params,
    ).to_list()
    counts = {}
    for k in expected:
        counts[k % 2] = counts.get(k % 2, 0) + 1
    assert rows == [{"parity": p, "n": n} for p, n in sorted(counts.items())]


@pytest.mark.parametrize("where,params,expected", NODE_SHAPES)
def test_node_null_or_shapes_behind_an_anchor(spans, where, params, expected):
    rows = spans.cypher(
        f"MATCH (a:Anchor)-[:NEXT]->(s:Span) WHERE {where} RETURN s.k AS k ORDER BY k",
        params=params,
    ).to_list()
    assert [r["k"] for r in rows] == expected


REL_SHAPES = [(where.replace("s.v", "r.v"), params, expected) for where, params, expected in NODE_SHAPES] + [
    ("r.vt IS NULL", None, [3, 4]),
    ("r.vt IS NOT NULL", None, [1, 2, 5, 6]),
    ("NOT (coalesce(r.vt, 0) >= 150)", None, [1, 3, 4]),
    ("NOT (coalesce(r.vt, null) >= 150)", None, [1]),
    ("NOT (coalesce(r.vt, 1000) >= 150)", None, [1]),
]


@pytest.mark.parametrize("where,params,expected", REL_SHAPES)
@pytest.mark.parametrize(
    "pattern",
    ["(a:Anchor {k: 0})-[r:NEXT]->(s)", "()-[r:NEXT]->(s)", "(s)<-[r:NEXT]-(:Anchor)"],
    ids=["anchored", "unanchored", "reversed"],
)
def test_relationship_null_or_shapes(spans, pattern, where, params, expected):
    rows = spans.cypher(f"MATCH {pattern} WHERE {where} RETURN s.k AS k ORDER BY k", params=params).to_list()
    assert [r["k"] for r in rows] == expected


def test_relationship_null_or_count(spans):
    rows = spans.cypher(
        "MATCH (:Anchor)-[r:NEXT]->() WHERE r.vf <= 150 AND (r.vt IS NULL OR r.vt >= 150) RETURN count(*) AS n"
    ).to_list()
    assert rows == [{"n": 3}]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_date_bounds_with_a_date_parameter(mode, tmp_path):
    import datetime

    g = _new_graph(mode, tmp_path)
    df = pd.DataFrame(
        {
            "k": [1, 2, 3],
            "vf": [datetime.date(2020, 1, 1)] * 3,
            "vt": [datetime.date(2020, 6, 1), datetime.date(2021, 6, 1), None],
        }
    )
    g.add_nodes(df, "D", "k", "k", column_types={"vf": "date", "vt": "date"})
    g.create_range_index("D", "vt")
    t = datetime.date(2021, 1, 1)
    for where, params in [
        ("d.vf <= $t AND (d.vt IS NULL OR d.vt >= $t)", {"t": t}),
        ("d.vf <= $t AND coalesce(d.vt, $max) >= $t", {"t": t, "max": datetime.date(9999, 12, 31)}),
    ]:
        rows = g.cypher(f"MATCH (d:D) WHERE {where} RETURN d.k AS k ORDER BY k", params=params).to_list()
        assert [r["k"] for r in rows] == [2, 3], where
