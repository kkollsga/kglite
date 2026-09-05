"""Absolute nullable-container oracles, separate from structural DISTINCT."""

import pytest

import kglite


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([None], [None], None),
        ([None], [1], None),
        ([None, 1], [2, 3], False),
        ([1, None], [3, 2], False),
        ([None], [], False),
        ({"a": None}, {"a": 1}, None),
        ({"a": [None]}, {"a": [None]}, None),
        ({"a": None}, {"b": None}, False),
        ({"a": None, "b": 1}, {"a": None, "b": 2}, False),
        ([[1]], [[1.0]], True),
    ],
)
def test_recursive_equality_truth_table(left, right, expected, disabled):
    graph = kglite.KnowledgeGraph()
    params = {"left": left, "right": right}
    actual = graph.cypher(
        "RETURN $left=$right AS eq,$left<>$right AS ne,NOT($left=$right) AS neg, "
        "CASE $left WHEN $right THEN 'match' ELSE 'other' END AS branch",
        params=params,
        disable_optimizer=disabled,
    ).to_list()
    inverse = None if expected is None else not expected
    assert actual == [
        {"eq": expected, "ne": inverse, "neg": inverse, "branch": "match" if expected is True else "other"}
    ]
    actual = graph.cypher(
        "UNWIND [1] AS x WITH x WHERE $left<>$right RETURN x",
        params=params,
        disable_optimizer=disabled,
    ).to_list()
    assert actual == ([{"x": 1}] if expected is False else [])


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("count", [8, 9])
def test_nested_null_membership_literal_prepared_and_dynamic(count, disabled):
    graph = kglite.KnowledgeGraph()
    literal = ",".join(f"[{i}]" for i in range(1, count + 1))
    actual = graph.cypher(
        f"RETURN [null] IN [{literal}] AS unknown,[1] IN [[null],{literal}] AS hit,"
        f"[] IN [{literal}] AS miss,NOT([null] IN [{literal}]) AS neg",
        disable_optimizer=disabled,
    ).to_list()
    assert actual == [{"unknown": None, "hit": True, "miss": False, "neg": None}]
    actual = graph.cypher(
        "UNWIND $rows AS r RETURN r.id AS id,r.probe IN r.items AS hit ORDER BY id",
        params={
            "rows": [
                {"id": 0, "probe": [None], "items": [[i] for i in range(1, count + 1)]},
                {"id": 1, "probe": [1], "items": [[None], [1]]},
                {"id": 2, "probe": [], "items": [[None]]},
            ]
        },
        disable_optimizer=disabled,
    ).to_list()
    assert actual == [{"id": 0, "hit": None}, {"id": 1, "hit": True}, {"id": 2, "hit": False}]


@pytest.mark.parametrize("disabled", [False, True])
def test_recursive_predicates_preserve_structural_distinct_grouping(disabled):
    graph = kglite.KnowledgeGraph()
    rows = graph.cypher(
        "UNWIND [[null],[null],[1]] AS x RETURN x,count(*) AS n ORDER BY n DESC",
        disable_optimizer=disabled,
    ).to_list()
    assert rows == [{"x": [None], "n": 2}, {"x": [1], "n": 1}]
    result = graph.cypher(
        "UNWIND [[null],[null],[1]] AS x RETURN collect(DISTINCT x) AS xs",
        disable_optimizer=disabled,
    ).scalar()
    assert result == [[None], [1]]


@pytest.mark.parametrize("disabled", [False, True])
def test_nested_null_relationship_predicate_pushdown(disabled):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(a:N{id:0}),(b:N{id:1}),(a)-[:R{v:[null]}]->(b)")
    for predicate in ["r.v=$v", "r.v<>$v", "NOT(r.v=$v)", "NOT(r.v<>$v)"]:
        actual = graph.cypher(
            f"MATCH(a:N)-[r:R]->(b:N) WHERE {predicate} RETURN b.id AS id",
            params={"v": [1]},
            disable_optimizer=disabled,
        ).to_list()
        assert actual == [], predicate
    assert graph.cypher(
        "MATCH(a:N)-[r:R]->(b:N) WHERE r.v=$v OR b.id=1 RETURN b.id AS id",
        params={"v": [1]},
        disable_optimizer=disabled,
    ).to_list() == [{"id": 1}]


@pytest.mark.parametrize("disabled", [False, True])
def test_nested_null_node_scan_comparisons(disabled):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(:N{id:0,v:[null]}),(:N{id:1,v:[1]}),(:N{id:2,v:[2]})")
    for predicate, expected in [
        ("n.v=$v", [1]),
        ("n.v<>$v", [2]),
        ("NOT(n.v=$v)", [2]),
        ("NOT(n.v<>$v)", [1]),
    ]:
        actual = graph.cypher(
            f"MATCH(n:N) WHERE {predicate} RETURN n.id AS id ORDER BY id",
            params={"v": [1]},
            disable_optimizer=disabled,
        ).to_list()
        assert actual == [{"id": value} for value in expected], predicate


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("count", [8, 9])
@pytest.mark.parametrize("parameterized", [False, True])
def test_prepared_membership_not_where_preserves_nested_unknown(disabled, count, parameterized):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(:N{id:0,v:[null]}),(:N{id:1,v:[1]}),(:N{id:2,v:[99]}),(:N{id:3,v:[]})")
    items = [[i] for i in range(1, count + 1)]
    literal = "[" + ",".join(f"[{i}]" for i in range(1, count + 1)) + "]"
    rhs = "$items" if parameterized else literal
    params = {"items": items} if parameterized else {}
    for predicate, expected in [(f"n.v IN {rhs}", [1]), (f"NOT(n.v IN {rhs})", [2, 3])]:
        rows = graph.cypher(
            f"MATCH(n:N) WHERE {predicate} RETURN n.id AS id ORDER BY id",
            params=params,
            disable_optimizer=disabled,
        ).to_list()
        assert rows == [{"id": value} for value in expected]

    # A later real match absorbs nested UNKNOWN, while a definite shape mismatch stays false.
    rhs = "$items" if parameterized else "[[null]," + literal[1:]
    params = {"items": [[None], *items]} if parameterized else {}
    rows = graph.cypher(
        f"MATCH(n:N) WHERE NOT(n.v IN {rhs}) RETURN n.id AS id ORDER BY id",
        params=params,
        disable_optimizer=disabled,
    ).to_list()
    assert rows == [{"id": 3}]
    rows = graph.cypher(
        f"MATCH(n:N) WHERE n.v IN {rhs} RETURN n.id AS id ORDER BY id",
        params=params,
        disable_optimizer=disabled,
    ).to_list()
    assert rows == [{"id": 1}]
