"""Cross-type ORDER BY / min / max — the total order, pinned absolutely.

A sort key that holds more than one type is ordinary: a `CASE` that returns a
number on some rows and a string on others, `coalesce` over differently-typed
properties, a property read across two node types. The comparator used to
*skip* a pair it could not compare, which made it intransitive — string-vs-
number reported "equal" while number-vs-number ordered — and Rust's `sort_by`
detects exactly that:

    PanicException: user-provided comparison function does not correctly
    implement a total order

deterministically from 21 rows up. Through pyo3 that surfaces as a
`BaseException` (not catchable by `except Exception`); the Bolt server has no
`catch_unwind`, so there it is a network-facing availability bug. The bounded
top-K heap has no such check and silently disagreed with the full sort
instead, and `min`/`max` rejected every candidate of a different type from the
incumbent, making their answer depend on which row arrived first.

The fix is one total order over every value — Neo4j 5's cross-type ranking —
shared by ORDER BY, the top-K heap and the aggregates. These tests pin the
rank sequence absolutely, both directions; the differential corpus cannot see
any of it, because it compares row *sets* and because the optimised and
unoptimised paths shared the defect.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

# The ascending rank table, as documented in CYPHER.md and in
# `graph/core/filtering.rs::type_rank`.
ASCENDING_RANKS = [
    "map",
    "node",
    "rel",
    "list",
    "path",
    "date",
    "datetime",
    "duration",
    "point",
    "string",
    "bool",
    "float",
    "int",
    "null",
]


@pytest.fixture
def tiny_graph() -> kglite.KnowledgeGraph:
    """Two nodes and one relationship, so a query can produce a node, a
    relationship and a path value to sort alongside the scalars."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame([{"id": 1, "nm": "A"}, {"id": 2, "nm": "B"}]), "N", "id", "nm")
    g.add_connections(pd.DataFrame([{"src": 1, "dst": 2}]), "L", "N", "src", "N", "dst")
    return g


def _kind(value: object) -> str:
    """Name the rank class of a returned value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "date" if value == "2024-01-01" else "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        for key, kind in (("labels", "node"), ("start", "rel"), ("nodes", "path"), ("months", "duration")):
            if key in value:
                return kind
        if "lat" in value or "latitude" in value:
            return "point"
        return "map"
    return type(value).__name__


# One value per rank class, deliberately *not* in rank order, so a sort that
# does nothing cannot pass.
_ONE_PER_CLASS = (
    "true, [1, 2], point(1.0, 2.0), datetime('2024-01-01T12:00:00'), -1.5, 7, a, r, "
    "'zzz', p, {z: 1}, duration({days: 1}), null, date('2024-01-01')"
)
_EVERY_CLASS_QUERY = f"MATCH p = (a:N)-[r:L]->(b:N) UNWIND [{_ONE_PER_CLASS}] AS v RETURN v ORDER BY v"


def test_every_rank_class_orders_ascending(tiny_graph):
    rows = tiny_graph.cypher(f"{_EVERY_CLASS_QUERY} ASC").to_list()
    assert [_kind(row["v"]) for row in rows] == ASCENDING_RANKS


def test_every_rank_class_orders_descending(tiny_graph):
    """DESC is the exact reverse, NULL included — DESC defaults to NULLS
    FIRST, which puts NULL at the head rather than dropping it."""
    rows = tiny_graph.cypher(f"{_EVERY_CLASS_QUERY} DESC").to_list()
    assert [_kind(row["v"]) for row in rows] == list(reversed(ASCENDING_RANKS))


def test_nulls_placement_overrides_the_type_rank(tiny_graph):
    """NULL's rank (last ascending) is the *default*; an explicit
    `NULLS FIRST/LAST` still wins."""
    rows = tiny_graph.cypher("UNWIND [3, null, 'a'] AS v RETURN v ORDER BY v ASC NULLS FIRST").to_list()
    assert [_kind(row["v"]) for row in rows] == ["null", "string", "int"]
    rows = tiny_graph.cypher("UNWIND [3, null, 'a'] AS v RETURN v ORDER BY v DESC NULLS LAST").to_list()
    assert [_kind(row["v"]) for row in rows] == ["int", "string", "null"]


def test_numbers_order_numerically_across_int_and_float(tiny_graph):
    """One rank for all three numeric variants — never `all ints, then all
    floats`."""
    rows = tiny_graph.cypher("UNWIND [3, 1.5, 2, -0.5, 10] AS v RETURN v ORDER BY v ASC").to_list()
    assert [row["v"] for row in rows] == [-0.5, 1.5, 2, 3, 10]


def test_an_integer_past_float_precision_orders_exactly(tiny_graph):
    """2**53 and 2**53+1 collapse onto one `f64`; comparing the integer
    *through* that float is intransitive as well as wrong."""
    rows = tiny_graph.cypher(
        "UNWIND [9007199254740993, 9007199254740992.0, 9007199254740992] AS v RETURN v ORDER BY v ASC"
    ).to_list()
    values = [row["v"] for row in rows]
    assert values[-1] == 9007199254740993
    assert set(values[:2]) == {9007199254740992, 9007199254740992.0}


def test_lists_order_element_wise_then_by_length(tiny_graph):
    rows = tiny_graph.cypher("UNWIND [[1, 2], [1], [1, 1, 9], [2]] AS v RETURN v ORDER BY v ASC").to_list()
    assert [row["v"] for row in rows] == [[1], [1, 1, 9], [1, 2], [2]]


def test_a_date_and_a_timestamp_share_one_rank_and_order_chronologically(tiny_graph):
    """Dates and timestamps are one rank, not two: a date compares as
    midnight, so a mixed column still reads chronologically."""
    rows = tiny_graph.cypher(
        "UNWIND [datetime('2024-01-01T12:00:00'), date('2024-01-02'), date('2024-01-01'), "
        "datetime('2023-12-31T23:00:00')] AS v RETURN v ORDER BY v ASC"
    ).to_list()
    assert [str(row["v"]) for row in rows] == [
        "2023-12-31 23:00:00",
        "2024-01-01",
        "2024-01-01 12:00:00",
        "2024-01-02",
    ]


# ── the panic: a mixed sort key over real rows ───────────────────────


def _mixed_key_graph(n: int) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame([{"id": i + 1, "nm": f"n{i:03}", "k": i % 7} for i in range(n)]),
        "S",
        "id",
        "nm",
    )
    return g


# A key that is a number on even `k` and a string on odd `k`, interleaved in
# scan order — the arrangement `sort_by` inspects for totality.
MIXED_KEY = "CASE WHEN n.k % 2 = 0 THEN n.k ELSE n.nm END"


@pytest.mark.parametrize("n", [21, 24, 32, 40, 64])
@pytest.mark.parametrize("direction", ["ASC", "DESC"])
def test_a_mixed_type_sort_key_does_not_abort_the_query(n, direction):
    """Regression: this raised `PanicException` (a BaseException, so user code
    could not even catch it) for every n >= 21."""
    graph = _mixed_key_graph(n)
    rows = graph.cypher(f"MATCH (n:S) RETURN n.nm AS nm ORDER BY {MIXED_KEY} {direction}").to_list()
    assert len(rows) == n


@pytest.mark.parametrize("n", [21, 64])
def test_a_mixed_type_sort_key_groups_strings_before_numbers(n):
    """The order is not merely defined, it is the documented one: strings
    (rank 9) sort ahead of every number (rank 11) ascending."""
    graph = _mixed_key_graph(n)
    rows = graph.cypher(f"MATCH (n:S) RETURN n.nm AS nm, {MIXED_KEY} AS k ORDER BY {MIXED_KEY} ASC").to_list()
    kinds = [_kind(row["k"]) for row in rows]
    assert set(kinds) == {"string", "int"}, "the fixture must actually mix types"
    assert kinds == sorted(kinds, key=lambda kind: 0 if kind == "string" else 1)
    strings = [row["k"] for row in rows if isinstance(row["k"], str)]
    assert strings == sorted(strings)


@pytest.mark.parametrize("limit", [1, 3, 17, 64])
@pytest.mark.parametrize("direction", ["ASC", "DESC"])
def test_top_k_agrees_with_the_full_sort_on_a_mixed_column(limit, direction):
    """The bounded heap has no totality check of its own — it just returned
    different rows. It must select and order exactly the full sort's prefix,
    on the optimised and unoptimised paths alike."""
    graph = _mixed_key_graph(64)
    limited = f"MATCH (n:S) RETURN n.nm AS nm ORDER BY {MIXED_KEY} {direction}, n.id ASC LIMIT {limit}"
    full = f"MATCH (n:S) RETURN n.nm AS nm ORDER BY {MIXED_KEY} {direction}, n.id ASC"
    expected = [row["nm"] for row in graph.cypher(full).to_list()][:limit]
    assert [row["nm"] for row in graph.cypher(limited).to_list()] == expected
    assert [row["nm"] for row in graph.cypher(limited, disable_optimizer=True).to_list()] == expected


def test_a_property_read_across_two_node_types_orders_by_rank():
    """No expression needed: two types, one property name, two types of
    value."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame([{"id": i, "nm": f"a{i}", "k": i} for i in range(1, 4)]), "A", "id", "nm")
    g.add_nodes(pd.DataFrame([{"id": i, "nm": f"b{i}", "k": f"s{i}"} for i in range(1, 4)]), "B", "id", "nm")
    rows = g.cypher("MATCH (n) RETURN n.nm AS nm ORDER BY n.k ASC, n.nm ASC").to_list()
    assert [row["nm"] for row in rows] == ["b1", "b2", "b3", "a1", "a2", "a3"]
    rows = g.cypher("MATCH (n) RETURN n.nm AS nm ORDER BY n.k DESC, n.nm ASC").to_list()
    # Every `k` is distinct, so the leading key decides every pair: numbers
    # (rank 11) lead descending, then the strings, each block reversed.
    assert [row["nm"] for row in rows] == ["a3", "a2", "a1", "b3", "b2", "b1"]


# ── min / max ────────────────────────────────────────────────────────


def _mixed_value_graph(order: list[int]) -> kglite.KnowledgeGraph:
    """Six nodes whose `k` is a string for three of them and a number for the
    other three, *inserted* in `order` — so `order` controls the arrival
    sequence the aggregates see, which is the whole point: the answer must not
    move when it changes."""
    rows = [
        {"id": 1, "nm": "a", "k": 4, "grp": "g"},
        {"id": 2, "nm": "b", "k": "m", "grp": "g"},
        {"id": 3, "nm": "c", "k": 9, "grp": "g"},
        {"id": 4, "nm": "d", "k": "z", "grp": "g"},
        {"id": 5, "nm": "e", "k": 1, "grp": "g"},
        {"id": 6, "nm": "f", "k": "a", "grp": "g"},
    ]
    ordered = [rows[index] for index in order]
    g = kglite.KnowledgeGraph()
    # `add_nodes` pins a property's type from the first batch, so the mixed
    # column has to arrive through the untyped Cypher write path.
    g.add_nodes(
        pd.DataFrame([{"id": r["id"], "nm": r["nm"], "grp": r["grp"]} for r in ordered]),
        "S",
        "id",
        "nm",
    )
    for row in ordered:
        literal = f"'{row['k']}'" if isinstance(row["k"], str) else row["k"]
        g.cypher(f"MATCH (n:S) WHERE n.id = {row['id']} SET n.k = {literal}")
    return g


FORWARD = [0, 1, 2, 3, 4, 5]
REVERSED = [5, 4, 3, 2, 1, 0]


@pytest.mark.parametrize("order", [FORWARD, REVERSED, [3, 0, 5, 2, 1, 4]])
def test_min_and_max_on_a_mixed_column_do_not_depend_on_row_order(order):
    """`min`/`max` used to keep the first-arriving value whenever the next
    candidate was of another type, so the answer changed with insertion
    order. Under the total order the strings (rank 9) sit below every number
    (rank 11): min is the smallest string, max the largest number."""
    graph = _mixed_value_graph(order)
    rows = graph.cypher("MATCH (n:S) RETURN min(n.k) AS mn, max(n.k) AS mx").to_list()
    assert rows == [{"mn": "a", "mx": 9}]


@pytest.mark.parametrize("order", [FORWARD, REVERSED])
def test_grouped_min_and_max_take_the_same_order(order):
    """The grouped/columnar aggregate path is a separate implementation of
    the same rule."""
    graph = _mixed_value_graph(order)
    rows = graph.cypher("MATCH (n:S) RETURN n.grp AS g, min(n.k) AS mn, max(n.k) AS mx").to_list()
    assert rows == [{"g": "g", "mn": "a", "mx": 9}]


@pytest.mark.parametrize("order", [FORWARD, REVERSED])
def test_min_and_max_take_the_same_order_off_the_optimised_path(order):
    """Three executors compute min/max — the fused scan, the materialized
    row loop and the streaming hash aggregate. They must not disagree, so
    each shape is asserted against the same expected answer."""
    graph = _mixed_value_graph(order)
    expected = [{"mn": "a", "mx": 9}]
    assert (
        graph.cypher("MATCH (n:S) RETURN min(n.k) AS mn, max(n.k) AS mx", disable_optimizer=True).to_list() == expected
    )
    assert graph.cypher("MATCH (n:S) WITH n WHERE n.id > 0 RETURN min(n.k) AS mn, max(n.k) AS mx").to_list() == expected
    assert graph.cypher(
        "MATCH (n:S) RETURN n.grp AS g, min(n.k) AS mn, max(n.k) AS mx",
        disable_optimizer=True,
    ).to_list() == [{"g": "g", "mn": "a", "mx": 9}]
    # A `collect` companion puts the whole RETURN on the materialized
    # aggregate executor — the third implementation, and the only shape that
    # reaches its min/max.
    rows = graph.cypher("MATCH (n:S) RETURN min(n.k) AS mn, max(n.k) AS mx, collect(n.nm) AS all").to_list()
    assert [{"mn": row["mn"], "mx": row["mx"]} for row in rows] == expected
    rows = graph.cypher("MATCH (n:S) RETURN n.grp AS g, min(n.k) AS mn, max(n.k) AS mx, collect(n.nm) AS all").to_list()
    assert [{"mn": row["mn"], "mx": row["mx"]} for row in rows] == expected


@pytest.mark.parametrize("order", [FORWARD, REVERSED])
def test_min_and_max_agree_with_order_by(order):
    """One order, three consumers: whatever `ORDER BY ... LIMIT 1` returns is
    what `min`/`max` return."""
    graph = _mixed_value_graph(order)
    first = graph.cypher("MATCH (n:S) RETURN n.k AS k ORDER BY n.k ASC LIMIT 1").to_list()
    last = graph.cypher("MATCH (n:S) RETURN n.k AS k ORDER BY n.k DESC NULLS LAST LIMIT 1").to_list()
    aggregate = graph.cypher("MATCH (n:S) RETURN min(n.k) AS mn, max(n.k) AS mx").to_list()
    assert first[0]["k"] == aggregate[0]["mn"]
    assert last[0]["k"] == aggregate[0]["mx"]


def test_min_and_max_still_ignore_nulls():
    """NULL ranks last ascending, but aggregates skip it entirely — a NULL
    must not become the answer of either aggregate."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame([{"id": 1, "nm": "a", "k": 3.0}, {"id": 2, "nm": "b", "k": None}, {"id": 3, "nm": "c", "k": 8.0}]),
        "S",
        "id",
        "nm",
    )
    assert g.cypher("MATCH (n:S) RETURN min(n.k) AS mn, max(n.k) AS mx").to_list() == [{"mn": 3.0, "mx": 8.0}]
