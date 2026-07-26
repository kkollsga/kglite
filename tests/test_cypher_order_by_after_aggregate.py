"""ORDER BY after an aggregating RETURN.

Aggregation rebuilds its output rows from scratch, so a variable only
survives onto them if the executor deliberately carries its binding
forward. Three implementations produce those rows — the streaming
aggregate, the materialized aggregate, and the fused
`OPTIONAL MATCH` + `count()` operator — and they used to disagree about
which variables survive. Where a variable was dropped,
`execute_order_by` evaluated the sort key to NULL on *every* row, every
key tied, and the stable sort handed back insertion order: the ORDER BY
clause was silently ignored.

The contract these tests pin down:

* A variable read by a **grouping key** keeps a value on the aggregated
  row, so ordering by any of its properties works — regardless of which
  aggregation path the planner picked.
* A variable that appears **only inside an aggregate argument** has no
  single value per group. Ordering by it is rejected with a message
  naming the fix, never silently ignored.
* Ordering by a projected alias keeps working.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

# priority is a total order with no ties, so the expected sequence below
# is the only correct answer — an assertion on it cannot pass by luck.
_TASKS = [
    ("t1", 5),
    ("t2", 2),
    ("t3", 7),
    ("t4", 4),
    ("t5", 3),
    ("t6", 8),
    ("t7", 9),
]
# Insertion order, which is what a silently-dropped ORDER BY returns.
INSERTION_ORDER = [title for title, _ in _TASKS]
PRIORITY_DESC = [title for title, _ in sorted(_TASKS, key=lambda t: -t[1])]
PRIORITY_ASC = list(reversed(PRIORITY_DESC))

assert PRIORITY_DESC != INSERTION_ORDER, "fixture must distinguish sorted from insertion order"


def build_task_graph() -> kglite.KnowledgeGraph:
    """Tasks with a comment count — the list-view shape this bug broke."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "tid": list(range(1, len(_TASKS) + 1)),
                "title": [title for title, _ in _TASKS],
                "priority": [priority for _, priority in _TASKS],
            }
        ),
        "Task",
        "tid",
        "title",
    )
    g.add_nodes(
        pd.DataFrame({"cid": [1, 2, 3], "label": ["c1", "c2", "c3"]}),
        "Comment",
        "cid",
        "label",
    )
    # t1 has two comments, t3 has one, the rest have none — so the
    # OPTIONAL MATCH genuinely produces null-padded rows.
    g.add_connections(
        pd.DataFrame({"from_id": [1, 1, 3], "to_id": [1, 2, 3]}),
        "X",
        "Task",
        "from_id",
        "Comment",
        "to_id",
    )
    return g


@pytest.fixture
def task_graph() -> kglite.KnowledgeGraph:
    return build_task_graph()


def titles(graph: kglite.KnowledgeGraph, query: str, **kwargs) -> list[str]:
    return [row["title"] for row in graph.cypher(query, **kwargs).to_list()]


# ── The reported bug ────────────────────────────────────────────────
#
# OPTIONAL MATCH + aggregate + ORDER BY on a non-projected expression.
# Asserts the row *order*, not just the row set — the bug returned every
# correct row in the wrong sequence.

REPRO = "MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) RETURN t.title AS title, count(c) AS n ORDER BY t.priority DESC"


def test_optional_match_aggregate_orders_by_non_projected_key(task_graph):
    assert titles(task_graph, REPRO) == PRIORITY_DESC


def test_repro_is_not_merely_insertion_order(task_graph):
    """Guards the guard: a fix that ignored ORDER BY would return this."""
    assert titles(task_graph, REPRO) != INSERTION_ORDER


def test_optional_match_aggregate_orders_ascending(task_graph):
    query = REPRO.replace("DESC", "ASC")
    assert titles(task_graph, query) == PRIORITY_ASC


def test_repro_matches_with_optimizer_disabled(task_graph):
    """The optimizer picked a different aggregation operator for the
    OPTIONAL shape; both must now agree, order included."""
    assert titles(task_graph, REPRO) == titles(task_graph, REPRO, disable_optimizer=True)


# ── Pagination ──────────────────────────────────────────────────────
#
# The practical damage: with the order unspecified, SKIP/LIMIT pages
# could repeat rows and skip others.


def test_pagination_pages_are_disjoint_and_ordered(task_graph):
    page_size = 3
    pages = [titles(task_graph, f"{REPRO} SKIP {skip} LIMIT {page_size}") for skip in range(0, len(_TASKS), page_size)]
    seen = [title for page in pages for title in page]

    assert seen == PRIORITY_DESC, "concatenated pages must reproduce the full ordering"
    assert len(seen) == len(set(seen)), f"pages overlap: {pages}"
    assert sorted(seen) == sorted(INSERTION_ORDER), "pagination dropped or duplicated rows"


# ── Controls: shapes that already worked must keep working ──────────


def test_order_by_projected_alias(task_graph):
    query = (
        "MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) "
        "RETURN t.title AS title, t.priority AS p, count(c) AS n ORDER BY p DESC"
    )
    assert titles(task_graph, query) == PRIORITY_DESC


def test_order_by_aggregate_alias(task_graph):
    query = (
        "MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) RETURN t.title AS title, count(c) AS n ORDER BY n DESC, title ASC"
    )
    assert titles(task_graph, query) == ["t1", "t3", "t2", "t4", "t5", "t6", "t7"]


def test_plain_match_aggregate_still_correct(task_graph):
    """The audit's working counter-example — must not regress."""
    query = "MATCH (t:Task) RETURN t.title AS title, count(*) AS n ORDER BY t.priority DESC"
    assert titles(task_graph, query) == PRIORITY_DESC


def test_projection_without_aggregate_still_correct(task_graph):
    query = "MATCH (t:Task) RETURN t.title AS title ORDER BY t.priority DESC"
    assert titles(task_graph, query) == PRIORITY_DESC


def test_group_by_bare_variable(task_graph):
    query = (
        "MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) "
        "RETURN t.title AS title, t AS node, count(c) AS n ORDER BY t.priority DESC"
    )
    assert titles(task_graph, query) == PRIORITY_DESC


def test_unaliased_aggregate_ordered_by_its_expression_form(task_graph):
    """`ORDER BY count(c)` resolves against the identically-named column
    when the aggregate is projected without an alias."""
    query = (
        "MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) "
        "RETURN t.title AS title, count(c) ORDER BY count(c) DESC, title ASC"
    )
    assert titles(task_graph, query) == ["t1", "t3", "t2", "t4", "t5", "t6", "t7"]


# ── The same defect outside OPTIONAL MATCH ──────────────────────────
#
# `collect()` is not computable by the streaming aggregate, so it falls
# to the materialized path — which dropped the binding too. This shape
# was silently unordered with no OPTIONAL MATCH and no optimizer.


@pytest.mark.parametrize(
    "aggregate",
    ["collect(t.tid)", "count(*)", "max(t.tid)", "avg(t.tid)"],
    ids=["collect", "count_star", "max", "avg"],
)
def test_every_aggregate_kind_honours_order_by(task_graph, aggregate):
    query = f"MATCH (t:Task) RETURN t.title AS title, {aggregate} AS agg ORDER BY t.priority DESC"
    assert titles(task_graph, query) == PRIORITY_DESC
    assert titles(task_graph, query) == titles(task_graph, query, disable_optimizer=True)


@pytest.mark.parametrize(
    "aggregate",
    ["collect(c.label)", "count(c)"],
    ids=["collect", "count"],
)
def test_every_aggregate_kind_honours_order_by_under_optional(task_graph, aggregate):
    query = (
        f"MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) "
        f"RETURN t.title AS title, {aggregate} AS agg ORDER BY t.priority DESC"
    )
    assert titles(task_graph, query) == PRIORITY_DESC
    assert titles(task_graph, query) == titles(task_graph, query, disable_optimizer=True)


# ── The ambiguous half: rejected, never silently reordered ──────────


def test_order_by_variable_only_inside_aggregate_is_rejected(task_graph):
    """`c` collapses into `count(c)`, so `c.label` has no single value
    per group. Neo4j rejects this shape; so do we."""
    query = "MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) RETURN t.title AS title, count(c) AS n ORDER BY c.label DESC"
    with pytest.raises(kglite.SchemaError) as excinfo:
        task_graph.cypher(query).to_list()

    message = str(excinfo.value)
    assert "'c'" in message, message
    assert "grouping keys" in message, message
    # The message must name a way out, not just refuse.
    assert "sort_key" in message or "projected column" in message, message


def test_rejection_is_identical_with_optimizer_disabled(task_graph):
    query = "MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) RETURN t.title AS title, count(c) AS n ORDER BY c.label DESC"
    with pytest.raises(kglite.SchemaError):
        task_graph.cypher(query, disable_optimizer=True).to_list()


def test_order_by_unprojected_aggregate_is_rejected(task_graph):
    query = "MATCH (t:Task) RETURN t.title AS title, count(*) AS n ORDER BY max(t.priority) DESC"
    with pytest.raises(kglite.SchemaError) as excinfo:
        task_graph.cypher(query).to_list()
    assert "max(t.priority)" in str(excinfo.value)


def test_order_by_aliased_aggregate_expression_names_the_alias(task_graph):
    """`count(*) AS n` moves the value to `n`; ordering by the expression
    form can no longer resolve, so the error points at the alias."""
    query = "MATCH (t:Task) RETURN t.title AS title, count(*) AS n ORDER BY count(*) DESC"
    with pytest.raises(kglite.SchemaError) as excinfo:
        task_graph.cypher(query).to_list()
    assert "ORDER BY n" in str(excinfo.value)


def test_ordering_by_grouping_variable_expression_is_allowed(task_graph):
    """Derived expressions over a grouping variable stay well defined."""
    query = (
        "MATCH (t:Task) OPTIONAL MATCH (t)-[:X]->(c) "
        "RETURN t.title AS title, count(c) AS n ORDER BY t.priority * -1 ASC"
    )
    assert titles(task_graph, query) == PRIORITY_DESC
