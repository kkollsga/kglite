"""Independent regression cases for three-valued and list expressions."""

from __future__ import annotations

import pytest

import kglite


@pytest.fixture
def graph():
    return kglite.KnowledgeGraph()


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("null IN [1, null]", None),
        ("1 IN [1, null]", True),
        ("2 IN [1, null]", None),
        ("NOT (null IN [1])", None),
        ("true AND null", None),
        ("false AND null", False),
        ("true OR null", True),
        ("false OR null", None),
        ("true XOR null", None),
        ("NOT null", None),
        ("true OR false AND false", True),
    ],
)
def test_boolean_expressions_preserve_unknown_and_precedence(graph, expression, expected):
    assert graph.cypher(f"RETURN {expression} AS value").to_list() == [{"value": expected}]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("any(x IN [null, false] WHERE x)", None),
        ("any(x IN [null, true] WHERE x)", True),
        ("all(x IN [true, null] WHERE x)", None),
        ("all(x IN [false, null] WHERE x)", False),
        ("none(x IN [false, null] WHERE x)", None),
        ("none(x IN [true, null] WHERE x)", False),
        ("single(x IN [true, null] WHERE x)", None),
        ("single(x IN [true, true, null] WHERE x)", False),
        ("single(x IN [true, false] WHERE x)", True),
    ],
)
def test_list_quantifiers_preserve_unknown(graph, expression, expected):
    assert graph.cypher(f"RETURN {expression} AS value").to_list() == [{"value": expected}]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("[1, 2] + [3, 4]", [1, 2, 3, 4]),
        ("0 + [1, 2]", [0, 1, 2]),
        ("[1, 2] + 3", [1, 2, 3]),
        ("[1] + null", [1, None]),
    ],
)
def test_plus_composes_lists_and_elements(graph, expression, expected):
    assert graph.cypher(f"RETURN {expression} AS value").to_list() == [{"value": expected}]


@pytest.mark.parametrize(
    ("expression", "message"),
    (("[0]['x']", "String index requires"), ("[0][1.2]", "index.*integer"), ("[0][true]", "index.*integer")),
)
def test_list_index_rejects_non_integer_types(graph, expression, message):
    with pytest.raises(kglite.CypherExecutionError, match=message):
        graph.cypher(f"RETURN {expression} AS value").to_list()


# ── Cross-type ordering: no rule for the pair of types → null ──────────────
#
# `<`, `<=`, `>`, `>=` used to collapse "these two types have no ordering rule"
# to `false`, in both directions. That made `NOT (a < b)` answer `true` and keep
# rows openCypher drops, and it contradicted KGLite's own declared total order
# (which places `'a'` before `1` for `ORDER BY` while `'a' < 1` was false).
# Equality across types stays `false` / `<>` stays `true` — only the ordering
# operators moved.

ORDERING_OPS = ("<", "<=", ">", ">=")

INCOMPARABLE_PAIRS = (
    ("1", "'a'"),
    ("1.5", "'a'"),
    ("true", "1"),
    ("[1]", "2"),
    ("{a: 1}", "1"),
    ("date('2024-03-15')", "'not-a-date'"),
    # Composite ordering: Neo4j answers null for list/list and map/map too.
    ("[1]", "[2]"),
    ("{a: 1}", "{a: 2}"),
)


@pytest.mark.parametrize(("left", "right"), INCOMPARABLE_PAIRS)
@pytest.mark.parametrize("op", ORDERING_OPS)
def test_cross_type_ordering_is_null_in_both_directions(graph, left, right, op):
    assert graph.cypher(f"RETURN {left} {op} {right} AS value").to_list() == [{"value": None}]
    assert graph.cypher(f"RETURN {right} {op} {left} AS value").to_list() == [{"value": None}]


@pytest.mark.parametrize(("left", "right"), INCOMPARABLE_PAIRS)
def test_cross_type_equality_stays_two_valued(graph, left, right):
    assert graph.cypher(f"RETURN {left} = {right} AS value").to_list() == [{"value": False}]
    assert graph.cypher(f"RETURN {left} <> {right} AS value").to_list() == [{"value": True}]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("NOT (1 < 'a')", None),
        ("NOT ('a' < 1)", None),
        ("1 < 'a' AND true", None),
        ("1 < 'a' AND false", False),
        ("1 < 'a' OR true", True),
        ("1 < 'a' OR false", None),
        ("1 < 'a' XOR true", None),
        ("(1 < 'a') IS NULL", True),
        # A date parses the string when it can; only an unparseable one is
        # cross-type.
        ("date('2024-03-15') < '2024-03-16'", True),
    ],
)
def test_cross_type_ordering_propagates_through_boolean_composition(graph, expression, expected):
    assert graph.cypher(f"RETURN {expression} AS value").to_list() == [{"value": expected}]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("CASE WHEN 1 < 'a' THEN 'y' ELSE 'n' END", "n"),
        ("CASE WHEN 1 < 'a' THEN 'y' END", None),
    ],
)
def test_null_comparison_takes_the_case_else_branch(graph, expression, expected):
    assert graph.cypher(f"RETURN {expression} AS value").to_list() == [{"value": expected}]


# NaN is a *number*, not an incomparable type: IEEE and Neo4j both answer
# `false` for every ordering comparison against it, and CYPHER.md's ordering
# section declares NaN sortable above every other number. These stay `false`
# while the pairings above move to `null`, which is the whole reason the two
# reasons `compare_values` declines a pair are told apart.
@pytest.mark.parametrize("op", ORDERING_OPS)
def test_nan_comparisons_are_false_not_null(graph, op):
    rows = graph.cypher(
        f"RETURN toFloat('nan') {op} 1 AS a, 1 {op} toFloat('nan') AS b,"
        f" toFloat('nan') {op} toFloat('nan') AS c,"
        f" NOT (toFloat('nan') {op} 1) AS negated"
    ).to_list()
    assert rows == [{"a": False, "b": False, "c": False, "negated": True}]


@pytest.mark.parametrize("op", ORDERING_OPS)
def test_a_stored_nan_keeps_its_row_under_negation(op):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:N {id: 1, v: toFloat('nan')}), (:N {id: 2, v: 'text'})")
    # The NaN row survives `NOT (...)` because its comparison is false; the
    # string row does not, because its comparison is null.
    assert g.cypher(f"MATCH (n:N) WHERE NOT (n.v {op} 1) RETURN n.id AS id").to_list() == [{"id": 1}]
    assert g.cypher(f"MATCH (n:N) WHERE n.v {op} 1 RETURN n.id AS id").to_list() == []


@pytest.fixture
def mixed_property_graph():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:P {id: 1, v: 'alpha'}), (b:P {id: 2, v: 'beta'}), (c:P {id: 3, v: 7})")
    return g


@pytest.mark.parametrize("op", ORDERING_OPS)
def test_where_drops_the_rows_whose_comparison_is_null(mixed_property_graph, op):
    rows = mixed_property_graph.cypher(f"MATCH (n:P) WHERE n.v {op} 5 RETURN n.id AS id").to_list()
    # Only the numeric row (v = 7) can answer; the two strings are null
    # against 5, and a null keeps no row.
    assert rows == ([{"id": 3}] if op in (">", ">=") else [])


@pytest.mark.parametrize("op", ORDERING_OPS)
def test_negated_where_also_drops_them(mixed_property_graph, op):
    rows = mixed_property_graph.cypher(f"MATCH (n:P) WHERE NOT (n.v {op} 5) RETURN n.id AS id").to_list()
    # `NOT null` is still null, so the strings stay dropped — that is the fix.
    assert rows == ([{"id": 3}] if op in ("<", "<=") else [])


@pytest.mark.parametrize("op", ORDERING_OPS)
def test_a_range_index_gives_the_same_rows_as_the_scan(op):
    scanned = []
    for indexed in (False, True):
        g = kglite.KnowledgeGraph()
        g.cypher("CREATE (a:P {id: 1, v: 'alpha'}), (b:P {id: 2, v: 7}), (c:P {id: 3, v: 3})")
        if indexed:
            g.create_index("P", "v")
        scanned.append(g.cypher(f"MATCH (n:P) WHERE n.v {op} 5 RETURN n.id AS id ORDER BY id").to_list())
        scanned.append(g.cypher(f"MATCH (n:P) WHERE NOT (n.v {op} 5) RETURN n.id AS id ORDER BY id").to_list())
    assert scanned[0:2] == scanned[2:4]


@pytest.mark.parametrize("op", ORDERING_OPS)
def test_relationship_property_predicates_follow_the_same_rule(op):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:P {id: 1})-[:K {w: 'x'}]->(b:P {id: 2}), (c:P {id: 3})-[:K {w: 9}]->(d:P {id: 4})")
    kept = g.cypher(f"MATCH (a)-[r:K]->(b) WHERE r.w {op} 1 RETURN a.id AS id").to_list()
    dropped = g.cypher(f"MATCH (a)-[r:K]->(b) WHERE NOT (r.w {op} 1) RETURN a.id AS id").to_list()
    # The string edge answers null for both forms; only the numeric edge is
    # ever kept, and exactly one of the two forms keeps it.
    assert kept == ([{"id": 3}] if op in (">", ">=") else [])
    assert dropped == ([] if op in (">", ">=") else [{"id": 3}])


@pytest.mark.parametrize("op", ORDERING_OPS)
def test_inline_relationship_property_pattern_is_unaffected(op):
    # A pattern-inline property is a positive filter: unknown and false both
    # reject the edge, so this shape is unchanged by the tristate move.
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:P {id: 1})-[:K {w: 'x'}]->(b:P {id: 2})")
    assert g.cypher("MATCH (a)-[r:K {w: 'x'}]->(b) RETURN a.id AS id").to_list() == [{"id": 1}]


def test_order_by_still_places_every_value(mixed_property_graph):
    # The total order `ORDER BY`, `min` and `max` use is deliberately separate
    # from the partial comparison `WHERE` uses, and this fix does not touch it:
    # strings still rank below numbers even though `'alpha' < 7` is null.
    rows = mixed_property_graph.cypher("MATCH (n:P) RETURN n.id AS id ORDER BY n.v").to_list()
    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert mixed_property_graph.cypher("MATCH (n:P) RETURN min(n.v) AS lo, max(n.v) AS hi").to_list() == [
        {"lo": "alpha", "hi": 7}
    ]
