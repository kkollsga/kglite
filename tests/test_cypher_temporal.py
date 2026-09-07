"""Tests for Cypher temporal filtering functions: valid_at, valid_during."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def temporal_graph():
    """Graph with temporal properties on both nodes and edges.

    Employees:
      Alice:   hire_date=2015-01-01, end_date=2020-12-31
      Bob:     hire_date=2018-06-01, end_date=NULL (still employed)
      Charlie: hire_date=2020-03-15, end_date=NULL (still employed)
      Diana:   hire_date=NULL,       end_date=NULL (unknown bounds)
      Eve:     hire_date=2010-01-01, end_date=2019-06-30

    Companies: Acme, Globex

    WORKS_AT edges (with start_date/end_date):
      Alice  -> Acme   (2015-01-01 to 2020-12-31)
      Bob    -> Acme   (2018-06-01 to NULL)
      Charlie-> Globex (2020-03-15 to NULL)
      Diana  -> Globex (NULL to NULL)
      Eve    -> Acme   (2010-01-01 to 2019-06-30)
      Alice  -> Globex (2021-01-01 to NULL)
    """
    graph = KnowledgeGraph()

    employees = pd.DataFrame(
        {
            "emp_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "hire_date": ["2015-01-01", "2018-06-01", "2020-03-15", None, "2010-01-01"],
            "end_date": ["2020-12-31", None, None, None, "2019-06-30"],
        }
    )
    graph.add_nodes(employees, "Employee", "emp_id", "name")

    companies = pd.DataFrame(
        {
            "comp_id": [10, 20],
            "name": ["Acme", "Globex"],
        }
    )
    graph.add_nodes(companies, "Company", "comp_id", "name")

    employment = pd.DataFrame(
        {
            "emp_id": [1, 2, 3, 4, 5, 1],
            "comp_id": [10, 10, 20, 20, 10, 20],
            "start_date": ["2015-01-01", "2018-06-01", "2020-03-15", None, "2010-01-01", "2021-01-01"],
            "end_date": ["2020-12-31", None, None, None, "2019-06-30", None],
        }
    )
    graph.add_connections(
        employment, "WORKS_AT", "Employee", "emp_id", "Company", "comp_id", columns=["start_date", "end_date"]
    )

    return graph


# ── valid_at on nodes ────────────────────────────────────────────────────────


class TestValidAtNodes:
    def test_basic_match(self, temporal_graph):
        """Mid-2019: Alice, Bob, Eve, Diana active; Charlie not yet hired."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee) "
            "WHERE valid_at(e, '2019-01-01', 'hire_date', 'end_date') "
            "RETURN e.title ORDER BY e.title"
        )
        names = [r["e.title"] for r in rows]
        assert "Alice" in names
        assert "Bob" in names
        assert "Eve" in names
        assert "Diana" in names  # NULL/NULL = always valid
        assert "Charlie" not in names  # starts 2020

    def test_null_from_open_start(self, temporal_graph):
        """Diana (NULL hire_date) should be valid at any date."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee) WHERE valid_at(e, '2000-01-01', 'hire_date', 'end_date') RETURN e.title"
        )
        names = {r["e.title"] for r in rows}
        assert "Diana" in names

    def test_null_to_open_end(self, temporal_graph):
        """Far future: only people with NULL end_date still valid."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee) WHERE valid_at(e, '2025-01-01', 'hire_date', 'end_date') RETURN e.title"
        )
        names = {r["e.title"] for r in rows}
        assert "Bob" in names  # 2018-NULL
        assert "Charlie" in names  # 2020-NULL
        assert "Diana" in names  # NULL-NULL
        assert "Alice" not in names  # ended 2020
        assert "Eve" not in names  # ended 2019

    def test_both_null_always_valid(self, temporal_graph):
        """Diana (both NULL) matches even very old dates."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee {title: 'Diana'}) "
            "WHERE valid_at(e, '1900-01-01', 'hire_date', 'end_date') "
            "RETURN e.title"
        )
        assert len(rows) == 1
        assert rows[0]["e.title"] == "Diana"

    def test_outside_range(self, temporal_graph):
        """Alice ended 2020-12-31, not valid in 2021."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee {title: 'Alice'}) "
            "WHERE valid_at(e, '2021-06-01', 'hire_date', 'end_date') "
            "RETURN e.title"
        )
        assert len(rows) == 0


# ── valid_at on edges ────────────────────────────────────────────────────────


class TestValidAtEdges:
    def test_basic_edge_match(self, temporal_graph):
        """Mid-2019: Alice@Acme, Bob@Acme, Eve@Acme, Diana@Globex active."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee)-[r:WORKS_AT]->(c:Company) "
            "WHERE valid_at(r, '2019-01-01', 'start_date', 'end_date') "
            "RETURN e.title, c.title ORDER BY e.title"
        )
        pairs = {(r["e.title"], r["c.title"]) for r in rows}
        assert ("Alice", "Acme") in pairs
        assert ("Bob", "Acme") in pairs
        assert ("Eve", "Acme") in pairs
        assert ("Diana", "Globex") in pairs
        # Charlie starts 2020, Alice@Globex starts 2021
        assert ("Charlie", "Globex") not in pairs
        assert ("Alice", "Globex") not in pairs

    def test_edge_null_end(self, temporal_graph):
        """Far future: only edges with NULL end_date active."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee)-[r:WORKS_AT]->(c:Company) "
            "WHERE valid_at(r, '2025-01-01', 'start_date', 'end_date') "
            "RETURN e.title, c.title"
        )
        pairs = {(r["e.title"], r["c.title"]) for r in rows}
        assert ("Bob", "Acme") in pairs  # NULL end
        assert ("Charlie", "Globex") in pairs  # NULL end
        assert ("Alice", "Globex") in pairs  # 2021-NULL
        assert ("Diana", "Globex") in pairs  # NULL-NULL
        assert ("Alice", "Acme") not in pairs  # ended 2020
        assert ("Eve", "Acme") not in pairs  # ended 2019

    def test_edge_with_node_filter(self, temporal_graph):
        """Combine temporal edge filter with node property filter."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee)-[r:WORKS_AT]->(c:Company {title: 'Acme'}) "
            "WHERE valid_at(r, '2019-01-01', 'start_date', 'end_date') "
            "RETURN e.title ORDER BY e.title"
        )
        names = [r["e.title"] for r in rows]
        assert names == ["Alice", "Bob", "Eve"]


# ── valid_during ─────────────────────────────────────────────────────────────


class TestValidDuring:
    def test_basic_overlap(self, temporal_graph):
        """2019 range: overlaps Alice (2015-2020), Bob (2018-NULL), Eve (2010-2019), Diana (NULL-NULL)."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee) "
            "WHERE valid_during(e, '2019-01-01', '2019-12-31', 'hire_date', 'end_date') "
            "RETURN e.title"
        )
        names = {r["e.title"] for r in rows}
        assert "Alice" in names  # 2015-2020 overlaps 2019
        assert "Bob" in names  # 2018-NULL overlaps 2019
        assert "Eve" in names  # 2010-2019 overlaps 2019
        assert "Diana" in names  # NULL-NULL overlaps everything
        assert "Charlie" not in names  # 2020-NULL starts after 2019

    def test_edge_overlap(self, temporal_graph):
        """2020 range on edges."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee)-[r:WORKS_AT]->(c:Company) "
            "WHERE valid_during(r, '2020-01-01', '2020-12-31', 'start_date', 'end_date') "
            "RETURN e.title, c.title"
        )
        pairs = {(r["e.title"], r["c.title"]) for r in rows}
        assert ("Alice", "Acme") in pairs  # 2015-2020 overlaps 2020
        assert ("Bob", "Acme") in pairs  # 2018-NULL overlaps 2020
        assert ("Charlie", "Globex") in pairs  # 2020-NULL overlaps 2020
        assert ("Diana", "Globex") in pairs  # NULL-NULL overlaps everything

    def test_no_overlap(self, temporal_graph):
        """Eve (2010-2019) doesn't overlap 2020-2025."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee {title: 'Eve'}) "
            "WHERE valid_during(e, '2020-01-01', '2025-12-31', 'hire_date', 'end_date') "
            "RETURN e.title"
        )
        assert len(rows) == 0


# ── DateTime property values ─────────────────────────────────────────────────


class TestValidAtWithDatetime:
    """petroleum_graph has Estimate nodes with DateTime date_from/date_to."""

    def test_estimate_datetime_fields(self, petroleum_graph):
        """First 25 estimates: 2020-01-01 to 2020-12-31. Mid-2020 should match all 25."""
        rows = petroleum_graph.cypher(
            "MATCH (e:Estimate) WHERE valid_at(e, '2020-06-15', 'date_from', 'date_to') RETURN count(*) AS n"
        )
        assert rows[0]["n"] == 25

    def test_estimate_with_date_function(self, petroleum_graph):
        """date() function should also work."""
        rows = petroleum_graph.cypher(
            "MATCH (e:Estimate) WHERE valid_at(e, date('2020-06-15'), 'date_from', 'date_to') RETURN count(*) AS n"
        )
        assert rows[0]["n"] == 25

    def test_prospect_string_fields(self, petroleum_graph):
        """Prospect date_from/date_to are strings. All 20 active mid-2022."""
        rows = petroleum_graph.cypher(
            "MATCH (p:Prospect) WHERE valid_at(p, '2022-06-15', 'date_from', 'date_to') RETURN count(*) AS n"
        )
        # First 10: 2020-01-01 to 2025-12-31 ✓, Last 10: 2019-01-01 to 2023-12-31 ✓
        assert rows[0]["n"] == 20


# ── Error handling ───────────────────────────────────────────────────────────


class TestTemporalErrors:
    def test_wrong_arg_count_valid_at(self, temporal_graph):
        with pytest.raises(Exception, match="4 arguments"):
            temporal_graph.cypher("MATCH (e:Employee) WHERE valid_at(e, '2020-01-01', 'hire_date') RETURN e")

    def test_wrong_arg_count_valid_during(self, temporal_graph):
        with pytest.raises(Exception, match="5 arguments"):
            temporal_graph.cypher(
                "MATCH (e:Employee) WHERE valid_during(e, '2020-01-01', '2020-12-31', 'hire_date') RETURN e"
            )

    def test_first_arg_not_variable(self, temporal_graph):
        with pytest.raises(Exception, match="variable"):
            temporal_graph.cypher(
                "MATCH (e:Employee) WHERE valid_at('not_a_var', '2020-01-01', 'hire_date', 'end_date') RETURN e"
            )


# ── Combined with other predicates ──────────────────────────────────────────


class TestTemporalCombined:
    def test_with_and(self, temporal_graph):
        """Temporal filter AND string filter."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee) "
            "WHERE valid_at(e, '2019-06-01', 'hire_date', 'end_date') "
            "  AND e.title STARTS WITH 'A' "
            "RETURN e.title"
        )
        assert len(rows) == 1
        assert rows[0]["e.title"] == "Alice"

    def test_with_or(self, temporal_graph):
        """Temporal filter OR exact match."""
        rows = temporal_graph.cypher(
            "MATCH (e:Employee) "
            "WHERE valid_at(e, '2025-01-01', 'hire_date', 'end_date') "
            "  OR e.title = 'Alice' "
            "RETURN e.title ORDER BY e.title"
        )
        names = [r["e.title"] for r in rows]
        assert "Alice" in names  # matched via OR
        assert "Bob" in names  # matched via valid_at


# ──────────────────────────────────────────────────────────────────────────
# A date is midnight on that date — for equality as well as ordering.
#
# CYPHER.md's ordering section declares "Dates and datetimes share one rank
# and compare chronologically, a date counting as midnight on that date", and
# `<`/`<=`/`>`/`>=` implemented it. `=`, `<>` and `IN` did not: they fell
# through to structural `Value` equality, so the same pair answered
# `<= -> true` and `= -> false` at once. These goldens pin the trichotomy.
# ──────────────────────────────────────────────────────────────────────────

MIDNIGHT = "datetime('2024-03-15T00:00:00')"
NOON = "datetime('2024-03-15T12:00:00')"
DAY = "date('2024-03-15')"


@pytest.fixture
def temporal_pair_graph():
    graph = KnowledgeGraph()
    graph.cypher(
        """
        CREATE (:T {id: 1, d: date('2024-03-15'), t: datetime('2024-03-15T00:00:00')}),
               (:T {id: 2, d: date('2024-03-16'), t: datetime('2024-03-15T12:00:00')})
        """
    )
    return graph


@pytest.mark.parametrize("disable_optimizer", [False, True])
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (f"{MIDNIGHT} = {DAY}", True),
        (f"{DAY} = {MIDNIGHT}", True),
        (f"{MIDNIGHT} <> {DAY}", False),
        (f"{DAY} <> {MIDNIGHT}", False),
        (f"{MIDNIGHT} < {DAY}", False),
        (f"{MIDNIGHT} <= {DAY}", True),
        (f"{MIDNIGHT} > {DAY}", False),
        (f"{MIDNIGHT} >= {DAY}", True),
        (f"{NOON} = {DAY}", False),
        (f"{NOON} <> {DAY}", True),
        (f"{NOON} > {DAY}", True),
        (f"{NOON} >= {DAY}", True),
        (f"{NOON} < {DAY}", False),
        (f"{DAY} = date('2024-03-16')", False),
        (f"{MIDNIGHT} IN [{DAY}]", True),
        (f"{DAY} IN [{MIDNIGHT}]", True),
        (f"{NOON} IN [{DAY}]", False),
        (f"{DAY} IN [{NOON}, {MIDNIGHT}]", True),
        (f"coalesce(null, {DAY}) = {MIDNIGHT}", True),
        (f"CASE WHEN {MIDNIGHT} = {DAY} THEN 'y' ELSE 'n' END", "y"),
        (f"CASE WHEN {NOON} = {DAY} THEN 'y' ELSE 'n' END", "n"),
        # A string is a different type family: `=` stays false, and only the
        # ordering comparison parses a date string.
        (f"{DAY} = '2024-03-15'", False),
        (f"{DAY} <> '2024-03-15'", True),
        (f"{MIDNIGHT} = '2024-03-15T00:00:00'", False),
        # Null still propagates through both.
        (f"{DAY} = null", None),
        (f"null = {MIDNIGHT}", None),
    ],
)
def test_date_equals_midnight_on_that_date(expression, expected, disable_optimizer):
    graph = KnowledgeGraph()
    got = graph.cypher(f"RETURN {expression} AS value", disable_optimizer=disable_optimizer).to_list()
    assert got == [{"value": expected}]


@pytest.mark.parametrize("disable_optimizer", [False, True])
@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        (f"WHERE n.d = {MIDNIGHT}", [1]),
        (f"WHERE n.d <> {MIDNIGHT}", [2]),
        (f"WHERE n.t = {DAY}", [1]),
        (f"WHERE n.t <> {DAY}", [2]),
        (f"WHERE n.d IN [{MIDNIGHT}]", [1]),
        (f"WHERE n.t IN [{DAY}, {NOON}]", [1, 2]),
        (f"WHERE NOT (n.t = {DAY})", [2]),
        (f"WHERE n.d <= {MIDNIGHT}", [1]),
        (f"WHERE n.d >= {MIDNIGHT}", [1, 2]),
    ],
)
def test_stored_temporal_properties_equate_across_date_and_datetime(
    temporal_pair_graph, clause, expected, disable_optimizer
):
    got = temporal_pair_graph.cypher(
        f"MATCH (n:T) {clause} RETURN n.id AS id ORDER BY id",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert [r["id"] for r in got] == expected


@pytest.mark.parametrize("disable_optimizer", [False, True])
def test_property_index_agrees_with_the_scan_on_temporal_equality(temporal_pair_graph, disable_optimizer):
    """A property index answers `=` without re-verifying, so its key set has to
    carry the same rule the scan does.
    """
    query = f"MATCH (n:T) WHERE n.d = {MIDNIGHT} RETURN n.id AS id ORDER BY id"
    before = temporal_pair_graph.cypher(query, disable_optimizer=disable_optimizer).to_list()
    temporal_pair_graph.create_index("T", "d")
    after = temporal_pair_graph.cypher(query, disable_optimizer=disable_optimizer).to_list()
    assert [r["id"] for r in before] == [1]
    assert after == before


@pytest.mark.parametrize("disable_optimizer", [False, True])
def test_temporal_trichotomy_holds_for_every_pair(disable_optimizer):
    """Exactly one of `<`, `=`, `>` is true for any date/datetime pair."""
    graph = KnowledgeGraph()
    values = [
        "date('2024-03-14')",
        "date('2024-03-15')",
        "datetime('2024-03-14T00:00:00')",
        "datetime('2024-03-15T00:00:00')",
        "datetime('2024-03-15T12:00:00')",
    ]
    for left in values:
        for right in values:
            row = graph.cypher(
                f"RETURN {left} < {right} AS lt, {left} = {right} AS eq, {left} > {right} AS gt",
                disable_optimizer=disable_optimizer,
            ).to_list()[0]
            assert sum(bool(row[k]) for k in ("lt", "eq", "gt")) == 1, (
                left,
                right,
                row,
            )


@pytest.mark.parametrize("disable_optimizer", [False, True])
def test_order_by_min_max_and_distinct_are_unchanged(disable_optimizer):
    """`ORDER BY`, `min`/`max` use `total_order` and `DISTINCT` uses structural
    `Value` identity — none of them routes through predicate equality, so a
    date and its midnight stay two distinct grouping keys while comparing
    equal under `=`.
    """
    graph = KnowledgeGraph()
    rows = graph.cypher(
        f"UNWIND [{NOON}, {DAY}, {MIDNIGHT}] AS v "
        "RETURN count(DISTINCT v) AS distinct_count, min(v) AS lo, max(v) AS hi",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows[0]["distinct_count"] == 3
    ordered = graph.cypher(
        f"UNWIND [{NOON}, {DAY}, {MIDNIGHT}] AS v RETURN v ORDER BY v",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert len(ordered) == 3
    assert str(ordered[-1]["v"]).startswith("2024-03-15 12:00")
