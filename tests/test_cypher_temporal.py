"""Tests for Cypher temporal filtering functions: valid_at, valid_during."""

import pandas as pd
import pytest

import kglite
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
        # A string is compared by parsing it, for `=` exactly as for `<`: a
        # value read back as ISO text finds the date it came from.
        (f"{DAY} = '2024-03-15'", True),
        (f"{DAY} <> '2024-03-15'", False),
        (f"{MIDNIGHT} = '2024-03-15T00:00:00'", True),
        (f"{DAY} = '2024-03-16'", False),
        (f"{DAY} <= '2024-03-15'", True),
        # Text that is not a date is another type family: `=` is false.
        (f"{DAY} = 'garbage'", False),
        (f"{DAY} <> 'garbage'", True),
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


# ──────────────────────────────────────────────────────────────────────────
# One instant rule for valid_at / valid_during.
#
# The query instant is read the way `date()` / `datetime()` read it: a
# date-only string through the `date()` parser (so '2009' is 2009-01-01), a
# string with a time part through the `datetime()` parser (an offset applied,
# normalised to UTC). Anything else is an error, never a silent false. Stored
# bounds may be dates, datetimes or ISO strings and answer alike.
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def bound_types_graph():
    """`a`, `s`, `t`: [2005-01-01, 2012-12-31] stored as date, string and
    datetime; `b`: [2009-01-01, open); `c`: [2010-01-01, open)."""
    graph = KnowledgeGraph()
    graph.cypher(
        """
        CREATE (:V {id: 'a', vf: date('2005-01-01'), vt: date('2012-12-31')}),
               (:V {id: 'b', vf: date('2009-01-01')}),
               (:V {id: 's', vf: '2005-01-01', vt: '2012-12-31'}),
               (:V {id: 't', vf: datetime('2005-01-01T00:00:00'), vt: datetime('2012-12-31T12:00:00')}),
               (:V {id: 'c', vf: date('2010-01-01')})
        """
    )
    return graph


def _valid_ids(graph, instant, disable_optimizer=False):
    rows = graph.cypher(
        f"MATCH (n:V) WHERE valid_at(n, {instant}, 'vf', 'vt') RETURN n.id AS id ORDER BY id",
        disable_optimizer=disable_optimizer,
    ).to_list()
    return [r["id"] for r in rows]


@pytest.mark.parametrize("disable_optimizer", [False, True])
@pytest.mark.parametrize(
    "instant",
    [
        "'2009'",
        "'2009-06'",
        "'2009-06-30'",
        "date('2009')",
        "date('2009-06-30')",
        "datetime('2009-06-30T12:00:00')",
        "'2009-06-30T12:00:00'",
    ],
)
def test_valid_at_reads_every_instant_form_the_same_way(bound_types_graph, instant, disable_optimizer):
    assert _valid_ids(bound_types_graph, instant, disable_optimizer) == ["a", "b", "s", "t"]


def test_valid_at_year_string_is_the_first_day_of_the_year(bound_types_graph):
    """'2010' is 2010-01-01: `c` starts that day, and the closed bound keeps it."""
    assert _valid_ids(bound_types_graph, "'2010'") == ["a", "b", "c", "s", "t"]
    assert _valid_ids(bound_types_graph, "'2004'") == []


@pytest.mark.parametrize("instant", ["'garbage'", "2009", "null", "datetime('garbage')", "[2009]"])
def test_valid_at_rejects_an_instant_it_cannot_read(bound_types_graph, instant):
    with pytest.raises(kglite.CypherExecutionError) as info:
        _valid_ids(bound_types_graph, instant)
    message = str(info.value)
    assert "n.vf" in message
    assert "date('2009')" in message


def test_valid_at_instant_error_names_both_types(bound_types_graph):
    with pytest.raises(kglite.CypherExecutionError) as info:
        _valid_ids(bound_types_graph, "2009")
    message = str(info.value)
    assert "INTEGER" in message
    assert "DATE" in message


@pytest.mark.parametrize("instant", ["'garbage'", "2009", "null"])
def test_valid_during_rejects_an_instant_it_cannot_read(bound_types_graph, instant):
    with pytest.raises(kglite.CypherExecutionError, match=r"date\('2009'\)"):
        bound_types_graph.cypher(f"MATCH (n:V) WHERE valid_during(n, {instant}, '2010', 'vf', 'vt') RETURN n.id")


def test_valid_during_reads_year_strings(bound_types_graph):
    rows = bound_types_graph.cypher(
        "MATCH (n:V) WHERE valid_during(n, '2000', '2009', 'vf', 'vt') RETURN n.id AS id ORDER BY id"
    ).to_list()
    assert [r["id"] for r in rows] == ["a", "b", "s", "t"]


def test_valid_at_rejects_an_unreadable_stored_bound():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:V {id: 'bad', vf: 'sometime', vt: date('2012-12-31')})")
    with pytest.raises(kglite.CypherExecutionError) as info:
        graph.cypher("MATCH (n:V) WHERE valid_at(n, date('2009'), 'vf', 'vt') RETURN n.id")
    message = str(info.value)
    assert "'bad'" in message
    assert "vf" in message
    assert "STRING" in message


def test_valid_at_rejects_an_unreadable_bound_on_a_relationship():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:P {id: 'p1'})-[:R {vf: 2005, vt: date('2012-12-31')}]->(:P {id: 'p2'})")
    with pytest.raises(kglite.CypherExecutionError) as info:
        graph.cypher("MATCH ()-[r:R]->() WHERE valid_at(r, date('2009'), 'vf', 'vt') RETURN 1")
    message = str(info.value)
    assert "r.vf" in message
    assert "INTEGER" in message
    assert "p1" in message


@pytest.fixture
def zoned_bounds_graph():
    graph = KnowledgeGraph()
    graph.cypher(
        """
        CREATE (:Z {id: 'ts_after', vf: date('2000-01-01'), vt: datetime('2009-06-29T23:30')}),
               (:Z {id: 'ts_before', vf: date('2000-01-01'), vt: datetime('2009-06-29T22:30')}),
               (:Z {id: 'day', vf: date('2000-01-01'), vt: date('2009-06-29')})
        """
    )
    return graph


@pytest.mark.parametrize("disable_optimizer", [False, True])
@pytest.mark.parametrize(
    "instant",
    ["datetime('2009-06-30T01:00+02:00')", "datetime('2009-06-30T01:00:00+02:00')", "'2009-06-30T01:00:00+02:00'"],
)
def test_valid_at_applies_the_instant_offset(zoned_bounds_graph, instant, disable_optimizer):
    """01:00+02:00 on the 30th is 23:00 UTC on the 29th: inside a 23:30 bound,
    outside a 22:30 bound, and inside a closed bound on its UTC date."""
    rows = zoned_bounds_graph.cypher(
        f"MATCH (n:Z) WHERE valid_at(n, {instant}, 'vf', 'vt') RETURN n.id AS id ORDER BY id",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert [r["id"] for r in rows] == ["day", "ts_after"]


def test_datetime_accepts_an_offset_without_seconds():
    graph = KnowledgeGraph()
    rows = graph.cypher(
        "RETURN datetime('2009-06-30T01:00+02:00') = datetime('2009-06-29T23:00:00') AS same, "
        "datetime('2009-06-30T01:00Z') = datetime('2009-06-30T01:00:00') AS utc"
    ).to_list()
    assert rows == [{"same": True, "utc": True}]
