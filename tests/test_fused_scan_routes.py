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
    `n.tag = 'Oslo'` is *consumed* by the index-selection pushdown, and the
    pattern matcher answers stored-property equality with a deliberate byte
    compare (`prop_matches`' fast arm) — so that spelling measures the
    matcher, not this route.
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
