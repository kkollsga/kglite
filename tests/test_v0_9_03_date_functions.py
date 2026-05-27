"""0.9.0 gate item §3 — stable date function set.

Currently `n.d.year` (and friends) is rejected at parse on a
`Value::DateTime` column with "Unexpected token at start of clause:
Dot". The Sodir prospect-graph enhance script works around this by
`toString()` + substring, which the agent flagged as a hack.

Target behavior:
- Field accessors on Value::DateTime: .year, .month, .day,
  .dayOfWeek, .hour, .minute, .second, .epochSeconds
- duration() constructor + arithmetic with DateTime
- duration accessors (.days, .seconds, .months)
- duration.between(d1, d2)

These tests are xfail-strict — see test_v0_9_05_integer_division.py
header for the workflow.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

NOT_IMPLEMENTED = "0.9.0 §3 — datetime accessor / duration not implemented; flip when fixed."


@pytest.fixture
def datetime_graph():
    """Date-only fixture — Value::DateTime is currently NaiveDate
    (date-only precision). Time-bearing strings like
    "2023-04-15T10:30:45" don't ingest cleanly today and would require
    promoting Value::DateTime → NaiveDateTime (200+ Value-match sites
    + storage format change). Deferred from §3 v1; tracked in
    dev-documentation/archive/0.9.0-readiness.md §3 (local-only)."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        [
            {"id": 1, "title": "A", "joined": "2023-04-15"},
            {"id": 2, "title": "B", "joined": "2024-12-31"},
            {"id": 3, "title": "C", "joined": "2020-01-01"},
        ]
    )
    g.add_nodes(df, "X", "id", "title", column_types={"joined": "datetime"})
    return g


# ---------------------------------------------------------------------------
# Field accessors on Value::DateTime
# ---------------------------------------------------------------------------


def test_datetime_year_accessor(datetime_graph):
    rows = list(datetime_graph.cypher("MATCH (n:X) RETURN n.id AS id, n.joined.year AS y ORDER BY n.id"))
    assert [(r["id"], r["y"]) for r in rows] == [(1, 2023), (2, 2024), (3, 2020)]


def test_datetime_month_day(datetime_graph):
    rows = list(datetime_graph.cypher("MATCH (n:X) WHERE n.id = 1 RETURN n.joined.month AS m, n.joined.day AS d"))
    assert rows[0]["m"] == 4
    assert rows[0]["d"] == 15


def test_datetime_time_accessors_return_zero_on_date_only(datetime_graph):
    """0.9.0 §3 v1 — Value::DateTime is NaiveDate (date-only). The
    .hour/.minute/.second accessors are wired but always return 0
    until Value::DateTime is promoted to NaiveDateTime (deferred —
    200+ Value-match sites). Locks in the v1 behavior."""
    rows = list(
        datetime_graph.cypher(
            "MATCH (n:X) WHERE n.id = 1 RETURN n.joined.hour AS h, n.joined.minute AS m, n.joined.second AS s"
        )
    )
    assert rows[0]["h"] == 0 and rows[0]["m"] == 0 and rows[0]["s"] == 0


def test_datetime_year_in_filter(datetime_graph):
    """Decade-bucket pattern from the Sodir creaming-curve queries."""
    rows = list(datetime_graph.cypher("MATCH (n:X) WHERE n.joined.year >= 2023 RETURN n.id AS id ORDER BY n.id"))
    assert [r["id"] for r in rows] == [1, 2]


def test_datetime_year_in_aggregation(datetime_graph):
    rows = list(datetime_graph.cypher("MATCH (n:X) RETURN n.joined.year AS year, count(n) AS c ORDER BY year"))
    pairs = [(r["year"], r["c"]) for r in rows]
    assert pairs == [(2020, 1), (2023, 1), (2024, 1)]


# ---------------------------------------------------------------------------
# duration() and arithmetic
# ---------------------------------------------------------------------------


def test_duration_days_constructor():
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN duration({days: 30}).days AS d"))
    assert rows[0]["d"] == 30


def test_datetime_plus_duration():
    """date + 30 days arithmetic — the natural shape for "expires in N
    days" patterns."""
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN datetime('2023-01-15T00:00:00') + duration({days: 30}) AS d"))
    # 2023-01-15 + 30 = 2023-02-14
    d = rows[0]["d"]
    # Accept any datetime representation; check year/month/day if dict-shaped
    if isinstance(d, dict):
        assert d.get("year") == 2023 and d.get("month") == 2 and d.get("day") == 14
    else:
        assert "2023-02-14" in str(d), f"got {d!r}"


def test_datetime_minus_datetime_returns_duration():
    g = kglite.KnowledgeGraph()
    rows = list(
        g.cypher(
            "RETURN duration.between(datetime('2023-01-01T00:00:00'), datetime('2023-02-01T00:00:00')).days AS days"
        )
    )
    assert rows[0]["days"] == 31


def test_duration_components_stay_separate():
    """0.9.0 Cluster 2 — calendar units (months) and clock units
    (days/seconds) are stored separately. Pre-Cluster-2 the
    soft-Duration folded everything to days, so
    duration({months: 1}).months returned 30 (collapsed) instead of 1.
    """
    g = kglite.KnowledgeGraph()
    rows = list(
        g.cypher(
            "RETURN duration({months: 1, days: 5}).months AS m, "
            "duration({months: 1, days: 5}).days AS d, "
            "duration({months: 1, days: 5}).seconds AS s"
        )
    )
    assert rows[0]["m"] == 1
    assert rows[0]["d"] == 5
    assert rows[0]["s"] == 0


def test_duration_seconds_component():
    """Sub-day inputs (hours/minutes/seconds) are folded into the
    `seconds` component without collapsing to days."""
    g = kglite.KnowledgeGraph()
    rows = list(
        g.cypher("RETURN duration({hours: 1, minutes: 30}).seconds AS s, duration({hours: 1, minutes: 30}).hours AS h")
    )
    assert rows[0]["s"] == 5400  # 3600 + 1800
    assert rows[0]["h"] == 1


def test_duration_addition_component_wise():
    """Duration + Duration sums components separately. Bound via
    WITH because chained `.field` access on a parenthesised
    expression isn't parsed yet."""
    g = kglite.KnowledgeGraph()
    rows = list(
        g.cypher(
            "WITH duration({months: 1, days: 5}) + duration({months: 2, days: 10}) AS d "
            "RETURN d.months AS m, d.days AS dd"
        )
    )
    assert rows[0]["m"] == 3
    assert rows[0]["dd"] == 15


def test_duration_between_cypher_md_example():
    """0.9.1 — anchor test for the (2024-08-12, 2026-05-02) example
    used in CYPHER.md "Duration semantics" section. Ensures the
    documented behaviour can't drift unnoticed.

    `duration.between` only fills in `days` (Value::DateTime is
    NaiveDate; calendar-month-aware diff is deferred).
    """
    g = kglite.KnowledgeGraph()
    rows = list(
        g.cypher(
            "RETURN duration.between(date('2024-08-12'), date('2026-05-02')).days AS d, "
            "duration.between(date('2024-08-12'), date('2026-05-02')).months AS m"
        )
    )
    assert rows[0]["d"] == 628
    assert rows[0]["m"] == 0


def test_estimate_is_stale_pattern(datetime_graph):
    """The Sodir 'estimate is stale (>5y old)' pattern, expressed
    naturally with duration arithmetic instead of toString hacking."""
    rows = list(
        datetime_graph.cypher(
            """
            MATCH (n:X)
            WITH n, datetime() AS now
            WHERE duration.between(n.joined, now).days > 365 * 5
            RETURN n.id AS id ORDER BY n.id
            """
        )
    )
    # Whichever rows are >5 years old — assertion adapts to current date,
    # but the query must parse and execute. n.id=3 (2020) will be stale by
    # any reasonable current date >2025-01-01.
    assert 3 in [r["id"] for r in rows]
