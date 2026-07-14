"""Regression coverage for query-wide Cypher execution budgets."""

import pandas as pd
import pytest

import kglite


def graph_with_types() -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": ["seed", "other"],
                "title": ["Seed", "Other"],
                "flag": [False, False],
            }
        ),
        "T",
        "id",
        "title",
    )
    graph.add_nodes(pd.DataFrame({"id": ["u"]}), "U", "id")
    return graph


@pytest.mark.parametrize("streaming", [False, True])
def test_max_rows_covers_unwind_and_is_inclusive(streaming: bool) -> None:
    graph = graph_with_types()
    query = "UNWIND [1, 2, 3] AS x RETURN x"

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher(query, max_rows=2, streaming=streaming)

    assert graph.cypher(query, max_rows=3, streaming=streaming).to_list() == [
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ]


def test_max_rows_covers_union_all_and_procedure_rows() -> None:
    graph = graph_with_types()

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher("RETURN 1 AS x UNION ALL RETURN 2 AS x", max_rows=1)

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher("CALL db.labels() YIELD label RETURN label", max_rows=1)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("disable_optimizer", [False, True])
def test_budget_is_identical_for_fused_and_naive_plans(streaming: bool, disable_optimizer: bool) -> None:
    graph = graph_with_types()
    query = "MATCH (n) RETURN labels(n) AS kind, count(*) AS n"

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher(
            query,
            max_rows=1,
            streaming=streaming,
            disable_optimizer=disable_optimizer,
        )

    assert (
        len(
            graph.cypher(
                query,
                max_rows=3,
                streaming=streaming,
                disable_optimizer=disable_optimizer,
            )
        )
        == 2
    )


def test_limit_physically_pushed_to_match_succeeds_at_cap() -> None:
    graph = graph_with_types()
    assert len(graph.cypher("MATCH (n:T) RETURN n.id AS id LIMIT 1", max_rows=1)) == 1


def test_budget_counts_retained_aggregate_collection_items() -> None:
    graph = graph_with_types()
    query = "UNWIND [1, 2] AS x RETURN collect(x) AS a, collect(x) AS b"

    with pytest.raises(kglite.CypherExecutionError, match="collection items"):
        graph.cypher(query, max_rows=2)

    row = graph.cypher(query, max_rows=4).to_list()[0]
    assert row == {"a": [1, 2], "b": [1, 2]}


def test_max_rows_covers_correlated_subquery_join() -> None:
    graph = graph_with_types()
    query = """
    UNWIND [1, 2] AS x
    CALL { WITH x UNWIND [10, 20] AS y RETURN y }
    RETURN x, y
    """

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher(query, max_rows=3)


def test_max_rows_covers_count_subquery_patterns_and_cross_joins() -> None:
    graph = graph_with_types()

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher("RETURN COUNT { (n:T) } AS c", max_rows=1)
    assert graph.cypher("RETURN COUNT { (n:T) } AS c", max_rows=2).to_list() == [{"c": 2}]

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher("RETURN COUNT { (:T), (:U) } AS c", max_rows=1)
    assert graph.cypher("RETURN COUNT { (:T), (:U) } AS c", max_rows=2).to_list() == [{"c": 2}]


def test_count_subquery_budget_error_rolls_back_earlier_mutation() -> None:
    graph = graph_with_types()
    query = """
    MATCH (n:T {id: 'seed'})
    SET n.flag = true
    WITH n
    RETURN COUNT { (m:T) } AS c
    """

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher(query, max_rows=1)

    assert graph.cypher("MATCH (n:T {id: 'seed'}) RETURN n.flag AS flag").to_list() == [{"flag": False}]


def test_mutation_budget_error_rolls_back_earlier_clause() -> None:
    graph = graph_with_types()
    query = """
    MATCH (n:T {id: 'seed'})
    SET n.flag = true
    WITH [1, 2, 3] AS xs
    UNWIND xs AS x
    RETURN x
    """

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher(query, max_rows=2)

    assert graph.cypher("MATCH (n:T {id: 'seed'}) RETURN n.flag AS flag").to_list() == [{"flag": False}]


def test_session_mutation_budget_matches_live_graph_and_rolls_back() -> None:
    session = graph_with_types().session()
    query = """
    MATCH (n:T {id: 'seed'})
    SET n.flag = true
    WITH [1, 2, 3] AS xs
    UNWIND xs AS x
    RETURN x
    """

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        session.execute(query, max_rows=2)

    assert session.cypher("MATCH (n:T {id: 'seed'}) RETURN n.flag AS flag").to_list() == [{"flag": False}]


def test_transaction_mutation_budget_rolls_back_only_failed_statement() -> None:
    graph = graph_with_types()
    tx = graph.begin()
    tx.cypher("MATCH (n:T {id: 'other'}) SET n.flag = true")

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        tx.cypher(
            """
            MATCH (n:T {id: 'seed'})
            SET n.flag = true
            WITH [1, 2, 3] AS xs
            UNWIND xs AS x
            RETURN x
            """,
            max_rows=2,
        )

    assert tx.cypher("MATCH (n:T) RETURN n.id AS id, n.flag AS flag ORDER BY id").to_list() == [
        {"id": "other", "flag": True},
        {"id": "seed", "flag": False},
    ]
    tx.commit()
