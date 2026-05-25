"""Temporal arithmetic Cypher functions — 2026-05-25 broad-scan lift.

Adds add_days / add_months / add_years / date_truncate as scalar
functions so wrappers reach these capabilities through the universal
cypher_query interface without per-binding date math.

Real use cases:
- `MATCH (e:Event) WHERE e.date <= add_days(date(), 30)` (next 30d)
- `RETURN date_truncate(e.ts, 'month'), count(e)` (group by month)
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def datetime_graph():
    """Small graph with date-typed properties for temporal queries."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            # Note: 2025-02-28 (not 29 — 2025 isn't a leap year).
            "ts": pd.to_datetime(["2024-01-15", "2024-06-30", "2025-02-28"]),
        }
    )
    g.add_nodes(df, "Event", "id", "name", column_types={"ts": "date"})
    return g


# ── add_days ───────────────────────────────────────────────────────────


def test_add_days_positive(datetime_graph):
    rows = datetime_graph.cypher("RETURN add_days(date('2024-01-15'), 30) AS d")
    assert str(rows[0]["d"]) == "2024-02-14"


def test_add_days_negative(datetime_graph):
    rows = datetime_graph.cypher("RETURN add_days(date('2024-01-15'), -15) AS d")
    assert str(rows[0]["d"]) == "2023-12-31"


def test_add_days_zero(datetime_graph):
    rows = datetime_graph.cypher("RETURN add_days(date('2024-01-15'), 0) AS d")
    assert str(rows[0]["d"]) == "2024-01-15"


def test_add_days_null_propagates(datetime_graph):
    rows = datetime_graph.cypher("RETURN add_days(null, 10) AS d")
    assert rows[0]["d"] is None


def test_add_days_on_property(datetime_graph):
    rows = datetime_graph.cypher("MATCH (e:Event {name: 'a'}) RETURN add_days(e.ts, 7) AS d")
    assert str(rows[0]["d"]) == "2024-01-22"


# ── add_months ─────────────────────────────────────────────────────────


def test_add_months_positive(datetime_graph):
    rows = datetime_graph.cypher("RETURN add_months(date('2024-01-15'), 2) AS d")
    assert str(rows[0]["d"]) == "2024-03-15"


def test_add_months_negative(datetime_graph):
    rows = datetime_graph.cypher("RETURN add_months(date('2024-03-15'), -2) AS d")
    assert str(rows[0]["d"]) == "2024-01-15"


def test_add_months_handles_short_month():
    """Jan 31 + 1 month = Feb 28 (or Feb 29 in leap year), not crash."""
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN add_months(date('2024-01-31'), 1) AS d")
    # 2024 is a leap year
    assert str(rows[0]["d"]) == "2024-02-29"


def test_add_months_null_propagates():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN add_months(null, 5) AS d")
    assert rows[0]["d"] is None


# ── add_years ──────────────────────────────────────────────────────────


def test_add_years_positive():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN add_years(date('2024-06-15'), 5) AS d")
    assert str(rows[0]["d"]) == "2029-06-15"


def test_add_years_leap_to_non_leap():
    """Feb 29 in leap year + 1 year = Feb 28 in non-leap year."""
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN add_years(date('2024-02-29'), 1) AS d")
    assert str(rows[0]["d"]) == "2025-02-28"


def test_add_years_negative():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN add_years(date('2024-06-15'), -10) AS d")
    assert str(rows[0]["d"]) == "2014-06-15"


# ── date_truncate ──────────────────────────────────────────────────────


def test_date_truncate_year():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN date_truncate(date('2024-06-15'), 'year') AS d")
    assert str(rows[0]["d"]) == "2024-01-01"


def test_date_truncate_month():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN date_truncate(date('2024-06-15'), 'month') AS d")
    assert str(rows[0]["d"]) == "2024-06-01"


def test_date_truncate_day_is_noop():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN date_truncate(date('2024-06-15'), 'day') AS d")
    assert str(rows[0]["d"]) == "2024-06-15"


def test_date_truncate_week_iso_monday():
    """ISO week starts Monday. 2024-06-15 is a Saturday — truncate to Mon 2024-06-10."""
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN date_truncate(date('2024-06-15'), 'week') AS d")
    assert str(rows[0]["d"]) == "2024-06-10"


def test_date_truncate_unknown_unit_errors():
    g = kglite.KnowledgeGraph()
    with pytest.raises(Exception, match="unit must be year/month/week/day"):
        g.cypher("RETURN date_truncate(date('2024-01-15'), 'second') AS d")


def test_date_truncate_null_propagates():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("RETURN date_truncate(null, 'month') AS d")
    assert rows[0]["d"] is None


# ── group-by use case (the documented real query pattern) ──────────────


def test_group_by_month_using_truncate(datetime_graph):
    """The real query pattern: monthly histogram of events."""
    rows = datetime_graph.cypher("MATCH (e:Event) RETURN date_truncate(e.ts, 'month') AS m, count(*) AS n ORDER BY m")
    # 3 events: Jan 2024, Jun 2024, Feb 2025 — three distinct months.
    assert len(rows) == 3
    months = [str(r["m"]) for r in rows]
    assert months == ["2024-01-01", "2024-06-01", "2025-02-01"]
    counts = [r["n"] for r in rows]
    assert counts == [1, 1, 1]
