"""Golden semantics for the fused node-scan operators' compiled property routes.

`FusedNodeScanAggregate` / `FusedNodeScanTopK` evaluate their group keys, sort
keys, aggregate arguments and surviving WHERE predicate through a plan compiled
once per scan (`executor/scan_eval.rs`): property *routes* resolved per node
type, and a borrowed string route for comparisons against a constant.

The compiled route is only ever a faster spelling of
`resolve_node_property` + `evaluate_predicate_tristate`, so every test here
pins the **value**, not the plan — an absolute golden, which is the one thing
the differential corpus cannot give (it compares the two paths against each
other, and a shared defect stays green in both).

The cases are the ones where a "just read the column" shortcut is wrong:
identity aliases, the structural soft aliases, absent and stored-`Null`
properties, non-string columns compared against string literals, mixed columns,
rows relocated out of a string column by a widening write, mapped storage, and
scans that cross node types mid-loop.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph


def _rows(graph: KnowledgeGraph, query: str, params: dict | None = None) -> list[dict]:
    return graph.cypher(query, params=params).to_list() if params else graph.cypher(query).to_list()


def _scalar(graph: KnowledgeGraph, query: str):
    return graph.cypher(query).scalar()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alias_graph() -> KnowledgeGraph:
    """A type whose id and title live under user-chosen names.

    `person_id` is the id alias, `full_name` the title alias, so `p.person_id`
    must read the id virtual and `p.full_name` the title virtual — while
    `p.city` is a plain stored column.
    """
    graph = KnowledgeGraph()
    people = pd.DataFrame(
        {
            "person_id": ["p1", "p2", "p3"],
            "full_name": ["Ada", "Bo", "Cyd"],
            "city": ["Oslo", "Bergen", "Oslo"],
        }
    )
    graph.add_nodes(people, "Person", "person_id", "full_name")
    return graph


@pytest.fixture
def sparse_graph() -> KnowledgeGraph:
    """A nullable string column and a nullable numeric column."""
    graph = KnowledgeGraph()
    rows = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "name": ["A", "B", "C", "D"],
            "email": ["a@x.io", None, "c@x.io", None],
            "score": [10, None, 30, None],
        }
    )
    graph.add_nodes(rows, "Rec", "id", "name")
    return graph


# ---------------------------------------------------------------------------
# Identity + soft aliases
# ---------------------------------------------------------------------------


def test_id_and_title_aliases_read_the_identity_columns(alias_graph):
    """The compiled route must resolve the type's declared aliases, not read a
    stored column of the same name (there is none)."""
    assert _rows(alias_graph, "MATCH (p:Person) RETURN p.person_id AS v ORDER BY v") == [
        {"v": "p1"},
        {"v": "p2"},
        {"v": "p3"},
    ]
    assert _rows(alias_graph, "MATCH (p:Person) RETURN p.full_name AS v ORDER BY v") == [
        {"v": "Ada"},
        {"v": "Bo"},
        {"v": "Cyd"},
    ]
    # …and the canonical spellings resolve to the same virtuals.
    assert _rows(alias_graph, "MATCH (p:Person) RETURN p.id AS v ORDER BY v") == [
        {"v": "p1"},
        {"v": "p2"},
        {"v": "p3"},
    ]
    assert _scalar(alias_graph, "MATCH (p:Person) WHERE p.title = 'Bo' RETURN count(*) AS c") == 1


def test_alias_predicates_take_the_borrowed_string_route(alias_graph):
    """A comparison against a constant string on an aliased identity field."""
    assert _scalar(alias_graph, "MATCH (p:Person) WHERE p.full_name = 'Bo' RETURN count(*) AS c") == 1
    assert _scalar(alias_graph, "MATCH (p:Person) WHERE p.full_name <> 'Bo' RETURN count(*) AS c") == 2
    assert _scalar(alias_graph, "MATCH (p:Person) WHERE p.person_id > 'p1' RETURN count(*) AS c") == 2
    assert _scalar(alias_graph, "MATCH (p:Person) WHERE p.full_name STARTS WITH 'A' RETURN count(*) AS c") == 1
    assert _scalar(alias_graph, "MATCH (p:Person) WHERE p.full_name ENDS WITH 'd' RETURN count(*) AS c") == 1
    assert _scalar(alias_graph, "MATCH (p:Person) WHERE p.full_name CONTAINS 'o' RETURN count(*) AS c") == 1


def test_soft_alias_name_prefers_a_stored_property_then_the_title():
    """`name` is a structural soft alias: a stored `name` wins (KG-1), and only
    a type without one falls back to the node title."""
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": ["t1", "t2"], "label_text": ["TitleOne", "TitleTwo"]}),
        "Titled",
        "id",
        "label_text",
    )
    graph.add_nodes(
        pd.DataFrame({"id": ["s1"], "heading": ["HeadOne"], "name": ["StoredName"]}),
        "Stored",
        "id",
        "heading",
    )
    assert _rows(graph, "MATCH (n:Titled) RETURN n.name AS v ORDER BY v") == [
        {"v": "TitleOne"},
        {"v": "TitleTwo"},
    ]
    assert _rows(graph, "MATCH (n:Stored) RETURN n.name AS v") == [{"v": "StoredName"}]
    # The same two resolutions, through the borrowed comparison route.
    assert _scalar(graph, "MATCH (n:Titled) WHERE n.name = 'TitleTwo' RETURN count(*) AS c") == 1
    assert _scalar(graph, "MATCH (n:Stored) WHERE n.name = 'StoredName' RETURN count(*) AS c") == 1
    assert _scalar(graph, "MATCH (n:Stored) WHERE n.name = 'HeadOne' RETURN count(*) AS c") == 0


def test_soft_alias_label_falls_back_to_the_type_string():
    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": ["a", "b"], "t": ["A", "B"]}), "Widget", "id", "t")
    assert _rows(graph, "MATCH (n:Widget) RETURN DISTINCT n.label AS v") == [{"v": "Widget"}]
    assert _scalar(graph, "MATCH (n:Widget) WHERE n.label = 'Widget' RETURN count(*) AS c") == 2
    assert _scalar(graph, "MATCH (n:Widget) WHERE n.label STARTS WITH 'Wid' RETURN count(*) AS c") == 2
    assert _scalar(graph, "MATCH (n:Widget) WHERE n.node_type = 'Widget' RETURN count(*) AS c") == 2


def test_stored_property_named_label_wins_over_the_type_string():
    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": ["a"], "t": ["A"], "label": ["user-set"]}), "Widget", "id", "t")
    assert _rows(graph, "MATCH (n:Widget) RETURN n.label AS v") == [{"v": "user-set"}]
    assert _scalar(graph, "MATCH (n:Widget) WHERE n.label = 'user-set' RETURN count(*) AS c") == 1
    assert _scalar(graph, "MATCH (n:Widget) WHERE n.label = 'Widget' RETURN count(*) AS c") == 0


# ---------------------------------------------------------------------------
# NULL semantics
# ---------------------------------------------------------------------------


def test_absent_string_property_is_unknown_not_false(sparse_graph):
    """`<>` against a NULL operand is UNKNOWN, so the row drops — and it must
    still drop under `NOT`, which is where a two-valued shortcut shows up."""
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.email <> 'zz' RETURN count(*) AS c") == 2
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.email = 'a@x.io' RETURN count(*) AS c") == 1
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE NOT (n.email CONTAINS 'x') RETURN count(*) AS c") == 0
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE NOT (n.email = 'a@x.io') RETURN count(*) AS c") == 1
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.email IS NULL RETURN count(*) AS c") == 2
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.email IS NOT NULL RETURN count(*) AS c") == 2


def test_absent_property_never_named_by_the_scan_is_null(sparse_graph):
    assert _rows(sparse_graph, "MATCH (n:Rec) RETURN DISTINCT n.nope AS v") == [{"v": None}]
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.nope = 'x' RETURN count(*) AS c") == 0
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.nope <> 'x' RETURN count(*) AS c") == 0


def test_null_group_keys_and_aggregate_inputs(sparse_graph):
    assert sorted(
        (r["v"] is None, r["c"]) for r in _rows(sparse_graph, "MATCH (n:Rec) RETURN n.email AS v, count(*) AS c")
    ) == [(False, 1), (False, 1), (True, 2)]
    # sum/avg skip NULL inputs; count(prop) counts only non-NULL.
    assert _scalar(sparse_graph, "MATCH (n:Rec) RETURN sum(n.score) AS s") == 40
    assert _scalar(sparse_graph, "MATCH (n:Rec) RETURN count(n.score) AS s") == 2
    assert _scalar(sparse_graph, "MATCH (n:Rec) RETURN avg(n.score) AS s") == 20.0
    # count(*) and count(<bound var>) count every row regardless.
    assert _scalar(sparse_graph, "MATCH (n:Rec) RETURN count(*) AS s") == 4
    assert _scalar(sparse_graph, "MATCH (n:Rec) RETURN count(n) AS s") == 4


def test_null_sort_keys_order_identically_to_the_unfused_pipeline(sparse_graph):
    fused = _rows(sparse_graph, "MATCH (n:Rec) RETURN n.id AS i ORDER BY n.score DESC LIMIT 4")
    unfused = sparse_graph.cypher(
        "MATCH (n:Rec) RETURN n.id AS i ORDER BY n.score DESC LIMIT 4", disable_optimizer=True
    ).to_list()
    assert fused == unfused


# ---------------------------------------------------------------------------
# Non-string values under a string comparison
# ---------------------------------------------------------------------------


def test_numeric_column_compared_against_a_string_literal(sparse_graph):
    """The borrowed route cannot answer a non-string column; it must fall back
    to the values the interpreter would have compared, not answer `false` for
    a shape the interpreter answers differently."""
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.score = '10' RETURN count(*) AS c") == 0
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.score <> '10' RETURN count(*) AS c") == 2
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.score CONTAINS '1' RETURN count(*) AS c") == 0
    assert _scalar(sparse_graph, "MATCH (n:Rec) WHERE n.score STARTS WITH '1' RETURN count(*) AS c") == 0


def test_date_column_compared_against_a_date_string():
    """`compare_values` parses a date out of the literal — a semantics the
    borrowed string route must not swallow."""
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "t": ["A", "B"],
                "happened": pd.to_datetime(["2024-01-01", "2025-06-01"]),
            }
        ),
        "Ev",
        "id",
        "t",
    )
    # `<` on a property is pushed into the pattern, so the surviving-WHERE
    # spelling is the OR — which is exactly the shape the compiled route owns.
    query = "MATCH (n:Ev) WHERE n.happened < '2025-01-01' OR n.id = 'zz' RETURN count(*) AS c"
    assert _scalar(graph, query) == graph.cypher(query, disable_optimizer=True).scalar() == 1


def test_json_single_element_list_equals_its_inner_string():
    """`values_equal`'s JSON-list equivalence, which the borrowed route
    reproduces rather than replacing with a byte compare.

    Spelled with predicates the planner leaves in the WHERE clause. A bare
    `n.tag = 'Oslo'` is *consumed* by the index-selection pushdown and
    answered by the pattern matcher's byte fast arm instead — a different
    route, pinned by
    `test_bare_equality_on_a_stored_json_list_agrees_with_in_and_not_equals`.
    """
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": ["a", "b"], "t": ["A", "B"], "tag": ['["Oslo"]', "Bergen"]}),
        "T",
        "id",
        "t",
    )
    for query, expected in (
        ("MATCH (n:T) WHERE n.tag <> 'Oslo' RETURN count(*) AS c", 1),
        ("MATCH (n:T) WHERE n.tag <> 'Bergen' RETURN count(*) AS c", 1),
        ("MATCH (n:T) WHERE n.tag = 'Oslo' OR n.id = 'zz' RETURN count(*) AS c", 1),
        ("MATCH (n:T) WHERE n.tag = '[\"Oslo\"]' OR n.id = 'zz' RETURN count(*) AS c", 1),
    ):
        assert _scalar(graph, query) == graph.cypher(query, disable_optimizer=True).scalar(), query
        assert _scalar(graph, query) == expected, query


def test_bare_equality_on_a_stored_json_list_agrees_with_in_and_not_equals():
    """One question, three spellings, one answer — an absolute golden.

    A bare `n.tag = 'Oslo'` is consumed by the index-selection pushdown and
    answered by `prop_matches`' byte-equality fast arm, which used to be the
    only route that skipped `values_equal`'s single-element-JSON-list rule.
    A row storing `'["Oslo"]'` therefore satisfied **neither** `=` nor `<>`
    against `'Oslo'`, while `IN ['Oslo']` matched it: `=` disagreed with `IN`,
    and the `=`/`<>` partition lost a row.

    Both directions are pinned (plain literal against a stored list, list
    literal against a stored plain string), on the optimized and the naive
    plan, because the defect lived in a route both plans share.
    """
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": ["a", "b"], "t": ["A", "B"], "tag": ['["Oslo"]', "Oslo"]}),
        "T",
        "id",
        "t",
    )
    for query, expected in (
        ("MATCH (n:T) WHERE n.tag = 'Oslo' RETURN count(*) AS c", 2),
        ("MATCH (n:T) WHERE n.tag <> 'Oslo' RETURN count(*) AS c", 0),
        ("MATCH (n:T) WHERE n.tag IN ['Oslo'] RETURN count(*) AS c", 2),
        ("MATCH (n:T) WHERE n.tag = '[\"Oslo\"]' RETURN count(*) AS c", 2),
        ("MATCH (n:T) WHERE n.tag IN ['[\"Oslo\"]'] RETURN count(*) AS c", 2),
        # The inline-property spelling reaches the same fast arm.
        ("MATCH (n:T {tag: 'Oslo'}) RETURN count(*) AS c", 2),
        # A genuine non-match must stay a non-match on every route.
        ("MATCH (n:T) WHERE n.tag = 'Bergen' RETURN count(*) AS c", 0),
    ):
        assert _scalar(graph, query) == expected, query
        assert graph.cypher(query, disable_optimizer=True).scalar() == expected, query

    # `=` and `<>` must partition the rows: none may satisfy neither.
    eq = _scalar(graph, "MATCH (n:T) WHERE n.tag = 'Oslo' RETURN count(*) AS c")
    ne = _scalar(graph, "MATCH (n:T) WHERE n.tag <> 'Oslo' RETURN count(*) AS c")
    assert eq + ne == 2, f"= matched {eq}, <> matched {ne}, of 2 rows"


def test_mixed_type_column_under_string_and_numeric_predicates():
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": ["a", "b", "c"], "t": ["A", "B", "C"], "v": ["x", 2, None]}),
        "T",
        "id",
        "t",
    )
    for query in (
        "MATCH (n:T) WHERE n.v = 'x' RETURN count(*) AS c",
        "MATCH (n:T) WHERE n.v <> 'x' RETURN count(*) AS c",
        "MATCH (n:T) WHERE n.v STARTS WITH 'x' RETURN count(*) AS c",
        "MATCH (n:T) WHERE n.v > 1 RETURN count(*) AS c",
        "MATCH (n:T) RETURN n.v AS v, count(*) AS c",
    ):
        assert _rows(graph, query) == graph.cypher(query, disable_optimizer=True).to_list(), query


# ---------------------------------------------------------------------------
# Storage shapes: relocated rows, mapped mode, cross-type scans
# ---------------------------------------------------------------------------


def test_relocated_string_rows_read_through_the_overlay():
    """A widening SET moves a row out of the packed string column into the
    relocation overlay; both the borrowed read and the owned read must follow
    it."""
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": [f"n{i}" for i in range(20)], "t": [f"t{i}" for i in range(20)], "s": ["ab"] * 20}),
        "T",
        "id",
        "t",
    )
    graph.cypher("MATCH (n:T) WHERE n.id = 'n7' SET n.s = 'a much longer relocated value'")
    assert _scalar(graph, "MATCH (n:T) WHERE n.s = 'ab' RETURN count(*) AS c") == 19
    assert _scalar(graph, "MATCH (n:T) WHERE n.s STARTS WITH 'a much' RETURN count(*) AS c") == 1
    assert _rows(graph, "MATCH (n:T) WHERE n.s <> 'ab' RETURN n.id AS i") == [{"i": "n7"}]
    assert sorted(r["c"] for r in _rows(graph, "MATCH (n:T) RETURN n.s AS s, count(*) AS c")) == [1, 19]


def test_mapped_storage_reads_the_same_values(tmp_path):
    def build(storage):
        graph = KnowledgeGraph(storage=storage) if storage else KnowledgeGraph()
        graph.add_nodes(
            pd.DataFrame(
                {
                    "id": ["a", "b", "c"],
                    "t": ["A", "B", "C"],
                    "city": ["Oslo", "Bergen", None],
                    "n": [1, 2, 3],
                }
            ),
            "T",
            "id",
            "t",
        )
        return graph

    memory, mapped = build(None), build("mapped")
    for query in (
        "MATCH (n:T) WHERE n.city = 'Oslo' RETURN count(*) AS c",
        "MATCH (n:T) WHERE n.city <> 'Oslo' RETURN count(*) AS c",
        "MATCH (n:T) WHERE n.city CONTAINS 'e' RETURN count(*) AS c",
        "MATCH (n:T) RETURN n.city AS c, count(*) AS k ORDER BY k, c",
        "MATCH (n:T) RETURN sum(n.n + 1) AS s",
        "MATCH (n:T) RETURN n.id AS i ORDER BY n.n DESC LIMIT 2",
        "MATCH (n:T) RETURN n.name AS v ORDER BY v",
    ):
        assert _rows(memory, query) == _rows(mapped, query), query


def test_scan_crossing_node_types_reresolves_each_type():
    """An untyped scan visits two types whose title alias differs; the route
    table has to follow the type change mid-loop."""
    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"aid": ["a1"], "aname": ["Alpha"]}), "A", "aid", "aname")
    graph.add_nodes(pd.DataFrame({"bid": ["b1"], "bname": ["Beta"]}), "B", "bid", "bname")
    for query in (
        "MATCH (n) RETURN n.name AS v ORDER BY v",
        "MATCH (n) RETURN n.label AS v, count(*) AS c ORDER BY v",
        "MATCH (n) WHERE n.name STARTS WITH 'A' RETURN count(*) AS c",
        "MATCH (n) RETURN n.name AS v ORDER BY v DESC LIMIT 1",
    ):
        assert _rows(graph, query) == graph.cypher(query, disable_optimizer=True).to_list(), query


# ---------------------------------------------------------------------------
# Compiled arithmetic + folding
# ---------------------------------------------------------------------------


def test_compiled_arithmetic_matches_the_interpreter(sparse_graph):
    for query in (
        "MATCH (n:Rec) RETURN sum(n.score + 1) AS s",
        "MATCH (n:Rec) RETURN sum(n.score * 2 - 1) AS s",
        "MATCH (n:Rec) RETURN sum(n.score / 2) AS s",
        "MATCH (n:Rec) RETURN sum(n.score % 3) AS s",
        "MATCH (n:Rec) RETURN sum(-n.score) AS s",
        "MATCH (n:Rec) RETURN max(n.name + '!') AS s",
        "MATCH (n:Rec) RETURN sum(1 + 2 + 3) AS s",
        "MATCH (n:Rec) RETURN n.id AS i ORDER BY n.score + 0 DESC LIMIT 2",
    ):
        assert _rows(sparse_graph, query) == sparse_graph.cypher(query, disable_optimizer=True).to_list(), query


def test_arithmetic_overflow_still_errors():
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": ["a"], "t": ["A"], "big": [2**62]}),
        "T",
        "id",
        "t",
    )
    with pytest.raises(Exception):
        graph.cypher("MATCH (n:T) RETURN sum(n.big * 8) AS s").to_list()


def test_or_and_xor_kleene_logic_over_a_nullable_column(sparse_graph):
    for query in (
        "MATCH (n:Rec) WHERE n.email = 'a@x.io' OR n.score > 20 RETURN count(*) AS c",
        "MATCH (n:Rec) WHERE n.email <> 'a@x.io' AND n.score > 5 RETURN count(*) AS c",
        "MATCH (n:Rec) WHERE n.email = 'a@x.io' XOR n.score > 5 RETURN count(*) AS c",
        "MATCH (n:Rec) WHERE n.email IN ['a@x.io', 'c@x.io'] RETURN count(*) AS c",
        "MATCH (n:Rec) WHERE NOT (n.email STARTS WITH 'a') RETURN count(*) AS c",
    ):
        assert _rows(sparse_graph, query) == sparse_graph.cypher(query, disable_optimizer=True).to_list(), query


# ---------------------------------------------------------------------------
# Aggregates over a property holding mixed types
# ---------------------------------------------------------------------------
#
# `avg()` is `sum() / count-of-numeric-values`. The fused scan's inline
# accumulator divided the numeric sum by the count of every *non-null* value
# instead, so one string cell in an otherwise numeric property dragged every
# average down — `[10, 20, 'hello']` averaged 10.0 (30/3) rather than 15.0.
# `sum()` on the same input already counted only numerics, which is what made
# the two disagree.
#
# Absolute goldens, deliberately: the four aggregation paths below must all
# answer the same, and "same as each other" is not enough — a shared defect
# would keep the comparison green.

_MIXED_VALUES_QUERY = "avg({x}) AS a, sum({x}) AS s, count({x}) AS c"
_MIXED_EXPECTED = {"a": 15.0, "s": 30, "c": 3}


def _mixed_node_graph() -> KnowledgeGraph:
    """`:S` nodes whose `v` holds `[10, 20, 'hello']`.

    Written through Cypher `CREATE`: the bulk loader types a column once, so a
    pandas object column of mixed values arrives as all strings (pinned
    separately by `test_loader_stringified_column_aggregates_to_null_average`).
    """
    graph = KnowledgeGraph()
    graph.cypher(
        "CREATE (:S {id: 1, name: 'a', v: 10}),"
        "       (:S {id: 2, name: 'b', v: 20}),"
        "       (:S {id: 3, name: 'c', v: 'hello'})"
    )
    return graph


def _mixed_rel_graph() -> KnowledgeGraph:
    """Three `:R` edges whose `w` property holds `[10, 20, 'hello']`."""
    graph = KnowledgeGraph()
    graph.cypher(
        "CREATE (:P {id: 1, name: 'a'}), (:P {id: 2, name: 'b'}),"
        "       (:P {id: 3, name: 'c'}), (:P {id: 4, name: 'd'})"
    )
    for target, weight in ((2, "10"), (3, "20"), (4, "'hello'")):
        graph.cypher(f"MATCH (a:P {{id: 1}}), (b:P {{id: {target}}}) CREATE (a)-[:R {{w: {weight}}}]->(b)")
    return graph


def test_avg_divides_by_the_numeric_count_on_every_aggregation_path():
    node_graph = _mixed_node_graph()
    rel_graph = _mixed_rel_graph()
    cases = (
        # Fused node-scan aggregate — the path that was wrong.
        (node_graph, "MATCH (n:S) RETURN " + _MIXED_VALUES_QUERY.format(x="n.v")),
        # Same shape with the node carried through a WITH.
        (node_graph, "MATCH (n:S) WITH n RETURN " + _MIXED_VALUES_QUERY.format(x="n.v")),
        # Projected to a scalar first — the materialized path.
        (node_graph, "MATCH (n:S) WITH n.v AS x RETURN " + _MIXED_VALUES_QUERY.format(x="x")),
        # No scan at all.
        (node_graph, "UNWIND [10, 20, 'hello'] AS x RETURN " + _MIXED_VALUES_QUERY.format(x="x")),
        # Relationship property.
        (rel_graph, "MATCH ()-[r:R]->() RETURN " + _MIXED_VALUES_QUERY.format(x="r.w")),
    )
    for graph, query in cases:
        assert _rows(graph, query) == [_MIXED_EXPECTED], query
        # avg is sum over the numeric count, on the same row it was computed.
        row = _rows(graph, query)[0]
        assert row["a"] == row["s"] / 2, query
        # …and the unoptimized plan agrees, so neither path is alone.
        assert graph.cypher(query, disable_optimizer=True).to_list() == [_MIXED_EXPECTED], query


def test_grouped_avg_divides_each_group_by_its_own_numeric_count():
    graph = KnowledgeGraph()
    graph.cypher(
        "CREATE (:S {id: 1, name: 'a', g: 'x', v: 10}),"
        "       (:S {id: 2, name: 'b', g: 'x', v: 20}),"
        "       (:S {id: 3, name: 'c', g: 'x', v: 'hello'}),"
        "       (:S {id: 4, name: 'd', g: 'y', v: 4}),"
        "       (:S {id: 5, name: 'e', g: 'y', v: 8})"
    )
    query = "MATCH (n:S) RETURN n.g AS g, avg(n.v) AS a, sum(n.v) AS s, count(n.v) AS c ORDER BY g"
    expected = [
        {"g": "x", "a": 15.0, "s": 30, "c": 3},
        # Control group: no string cell, so this one was right all along.
        {"g": "y", "a": 6.0, "s": 12, "c": 2},
    ]
    assert _rows(graph, query) == expected
    assert graph.cypher(query, disable_optimizer=True).to_list() == expected


def test_aggregates_over_zero_numeric_values_are_null_avg_and_zero_sum():
    """No numeric input at all: `avg` is null (not 0.0) and `sum` is 0."""
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:S {id: 1, name: 'a', v: 'x'}), (:S {id: 2, name: 'b', v: 'y'})")
    expected = [{"a": None, "s": 0, "c": 2}]
    for query in (
        "MATCH (n:S) RETURN avg(n.v) AS a, sum(n.v) AS s, count(n.v) AS c",
        "MATCH (n:S) WITH n.v AS x RETURN avg(x) AS a, sum(x) AS s, count(x) AS c",
        "UNWIND ['x', 'y'] AS x RETURN avg(x) AS a, sum(x) AS s, count(x) AS c",
    ):
        assert _rows(graph, query) == expected, query
        assert graph.cypher(query, disable_optimizer=True).to_list() == expected, query


def test_loader_stringified_column_aggregates_to_null_average():
    """The loader shape: a pandas object column of `[10, 20, 'N/A']`.

    `add_nodes` types the column once, so all three values are stored as
    strings — every value non-null, none of them numeric. That made the fused
    path answer `avg` = 0.0 (a numeric sum of 0 over 3 values) while the
    unfused path answered null. Both now say null, and `sum` says 0.
    """
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "v": [10, 20, "N/A"]}),
        "L",
        "id",
        "name",
        columns=["v"],
    )
    # The stringification itself, pinned: it is why there is no numeric value.
    assert _rows(graph, "MATCH (n:L) RETURN n.v AS v ORDER BY n.id") == [
        {"v": "10"},
        {"v": "20"},
        {"v": "N/A"},
    ]
    query = "MATCH (n:L) RETURN avg(n.v) AS a, sum(n.v) AS s, count(n.v) AS c"
    expected = [{"a": None, "s": 0, "c": 3}]
    assert _rows(graph, query) == expected
    assert graph.cypher(query, disable_optimizer=True).to_list() == expected


def test_sum_keeps_its_numeric_type_when_a_string_cell_is_present():
    """`sum()`'s Int64-vs-Float64 choice must not read `min()`.

    A string outranks every number in the cross-type order, so deriving the
    integer-ness of the sum from the running minimum turned `sum` over an
    integer property into a float the moment one string cell appeared — on the
    fused path only.
    """
    graph = _mixed_node_graph()
    for query, expected in (
        ("MATCH (n:S) RETURN sum(n.v) AS s", 30),
        ("MATCH (n:S) RETURN min(n.v) AS m", "hello"),
        ("MATCH (n:S) RETURN max(n.v) AS m", 20),
    ):
        assert _rows(graph, query) == [{query.split(" AS ")[1]: expected}], query
        assert graph.cypher(query, disable_optimizer=True).to_list() == [{query.split(" AS ")[1]: expected}], query
    # Int64, not Float64 — `str()` is what the differential corpus compares on.
    assert isinstance(_scalar(graph, "MATCH (n:S) RETURN sum(n.v) AS s"), int)


# `sum()`'s Int64-vs-Float64 result type is one rule for every internal
# aggregation path: integer iff every numeric input was an `Int64` and the
# total is whole. The materialized executor instead probed the *first* row of
# the group — a leading string or null says nothing about the numerics behind
# it — so identical data summed to `10` through the streaming and fused-scan
# paths and `10.0` through the materialized one, and which one a query got
# depended on whether the streaming aggregate happened to bail.
#
# Absolute goldens across all three paths. The route levers, verified against
# the pre-fix build:
#   * default kwargs on a bare scan — the fused node-scan aggregate;
#   * a projected scalar (`WITH n.v AS x`) — the streaming aggregate;
#   * the same with `streaming=False` — the materialized executor's
#     `evaluate_aggregate_with_rows`;
#   * a literal grouping key with the optimizer off — the materialized
#     executor's `try_fused_numeric_aggregation`;
#   * `median` alongside `sum` — an aggregate the streaming recognizer and
#     the fused scan both refuse, so it lands on the materialized executor
#     whatever else is enabled.

_SUM_TYPE_CASES = (
    # A leading non-numeric, then an integer: pre-fix Float64(10.0).
    (["'x'", "10"], 10, int),
    # A leading null, then integers: pre-fix Float64(3.0).
    (["null", "1", "2"], 3, int),
    # A float anywhere makes the sum a float, wherever it sits.
    (["1.5", "2"], 3.5, float),
    (["2", "1.5"], 3.5, float),
    # Controls: all-integer stays integer, no numerics at all sums to 0.
    (["1", "2"], 3, int),
    (["'x'", "'y'"], 0, int),
)

_SUM_TYPE_ROUTES = (
    ("MATCH (n:S) RETURN sum(n.v) AS s", {}),
    ("MATCH (n:S) WITH n.v AS x RETURN sum(x) AS s", {}),
    ("MATCH (n:S) WITH n.v AS x RETURN sum(x) AS s", {"streaming": False}),
    ("MATCH (n:S) RETURN 1 AS k, sum(n.v) AS s", {"streaming": False, "disable_optimizer": True}),
    ("MATCH (n:S) RETURN sum(n.v) AS s, median(n.v) AS m", {}),
    ("MATCH (n:S) RETURN sum(n.v) AS s, median(n.v) AS m", {"streaming": False}),
)


def _sum_type_graph(values: list[str]) -> KnowledgeGraph:
    """`:S` nodes whose `v` holds `values`; the literal `null` omits `v`."""
    graph = KnowledgeGraph()
    parts = []
    for i, value in enumerate(values, start=1):
        props = f"id: {i}, name: 's{i}'"
        if value != "null":
            props += f", v: {value}"
        parts.append(f"(:S {{{props}}})")
    graph.cypher("CREATE " + ", ".join(parts))
    return graph


@pytest.mark.parametrize(("values", "expected", "expected_type"), _SUM_TYPE_CASES)
def test_sum_result_type_is_the_same_on_every_aggregation_path(values, expected, expected_type):
    graph = _sum_type_graph(values)
    for query, kwargs in _SUM_TYPE_ROUTES:
        row = graph.cypher(query, **kwargs).to_list()[0]
        label = f"{values} :: {query} {kwargs}"
        assert row["s"] == expected, label
        # `bool` is an `int` subclass and `10 == 10.0`, so the type is the
        # assertion — equality alone cannot see this bug.
        assert type(row["s"]) is expected_type, label
