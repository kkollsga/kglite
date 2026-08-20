"""Regression coverage for query-wide Cypher execution budgets."""

import time

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


# ---------------------------------------------------------------------------
# Default path (no max_rows): the absolute backstop inside ExecutionBudget.
#
# `max_rows` is opt-in, so every check above used to be inert on the path
# almost all callers take. An unbounded cross-product could therefore
# materialize until the OS killed the *host* process. Ceiling and rationale:
# `MAX_UNBOUNDED_ROWS` in crates/kglite/src/graph/languages/cypher/executor/
# budget.rs. These queries are sized to cross it via a pre-sized or
# accumulating check, so they fail without first materializing 10M rows.
# ---------------------------------------------------------------------------

BACKSTOP_ROWS = 10_000_000

# 3200 x 3200 = 10,240,000 combined rows. The uncorrelated CALL subquery join
# knows the product before it allocates, so the backstop stops it while only
# 6,400 rows exist.
UNBOUNDED_CROSS_PRODUCT = """
UNWIND range(1, 3200) AS a
CALL { UNWIND range(1, 3200) AS b RETURN b }
RETURN count(*) AS n
"""


def test_default_path_backstops_an_unbounded_cross_product() -> None:
    graph = kglite.KnowledgeGraph()

    start = time.perf_counter()
    with pytest.raises(kglite.CypherExecutionError) as excinfo:
        graph.cypher(UNBOUNDED_CROSS_PRODUCT)
    elapsed = time.perf_counter() - start

    message = str(excinfo.value)
    assert str(BACKSTOP_ROWS) in message, message
    assert "max_rows" in message, message
    assert "10240000" in message, message
    # Incremental/pre-sized checks mean this must fail early, not after the
    # cross-product has been built.
    assert elapsed < 30.0, f"backstop took {elapsed:.1f}s — it is not failing early"


def test_default_path_backstops_accumulated_collection_items() -> None:
    """Rows stay small, but the collections built to produce them do not."""
    graph = kglite.KnowledgeGraph()
    query = "UNWIND range(1, 4000) AS a WITH size(range(1, 3000 + a)) AS s RETURN sum(s) AS total"

    with pytest.raises(kglite.CypherExecutionError) as excinfo:
        graph.cypher(query)

    message = str(excinfo.value)
    assert "collection items" in message, message
    assert str(BACKSTOP_ROWS) in message, message


def test_explicit_max_rows_still_governs_the_same_query() -> None:
    """An explicit max_rows replaces the backstop — smaller or larger."""
    graph = kglite.KnowledgeGraph()

    with pytest.raises(kglite.CypherExecutionError, match="max_rows limit of 1000"):
        graph.cypher(UNBOUNDED_CROSS_PRODUCT, max_rows=1000)

    graph.set_default_max_rows(1000)
    with pytest.raises(kglite.CypherExecutionError, match="max_rows limit of 1000"):
        graph.cypher(UNBOUNDED_CROSS_PRODUCT)
    graph.set_default_max_rows(None)


def test_default_path_below_the_ceiling_is_unaffected() -> None:
    graph = graph_with_types()

    assert graph.cypher("UNWIND range(1, 200) AS a UNWIND range(1, 200) AS b RETURN count(*) AS n").to_list() == [
        {"n": 40000}
    ]
    # A single collection an order of magnitude past any realistic result set
    # is still built: the backstop is not a default max_rows.
    assert graph.cypher("RETURN size(range(1, 1000000)) AS n").to_list() == [{"n": 1000000}]
    # Scan work is exempt: a fused count charges the whole graph and allocates
    # nothing, so it must never see the ceiling.
    assert graph.cypher("MATCH (n) RETURN count(n) AS n").to_list() == [{"n": 3}]


def test_oversized_range_still_reports_its_own_byte_ceiling() -> None:
    """The range()-specific message stays the one a caller sees."""
    graph = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherExecutionError, match="256 MiB"):
        graph.cypher("RETURN range(0, 100000000) AS r")
