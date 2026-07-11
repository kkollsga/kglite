"""Regression tests for checked range and temporal arithmetic."""

import pytest

import kglite

I64_MIN = -(2**63)
I64_MAX = 2**63 - 1


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    return kglite.KnowledgeGraph()


@pytest.mark.parametrize(
    ("start", "end", "step", "expected"),
    [
        (I64_MAX, I64_MAX, 1, [I64_MAX]),
        (I64_MAX, I64_MAX - 1, -1, [I64_MAX, I64_MAX - 1]),
        (I64_MIN, I64_MIN + 1, 1, [I64_MIN, I64_MIN + 1]),
        (I64_MIN, I64_MIN, -1, [I64_MIN]),
        (0, I64_MIN, I64_MIN, [0, I64_MIN]),
        (5, 1, 1, []),
        (1, 5, -1, []),
    ],
)
def test_range_handles_extrema_and_direction_mismatch(
    graph: kglite.KnowledgeGraph,
    start: int,
    end: int,
    step: int,
    expected: list[int],
) -> None:
    row = graph.cypher(
        "RETURN range($start, $end, $step) AS r",
        params={"start": start, "end": end, "step": step},
    ).to_list()[0]
    assert row["r"] == expected


def test_range_rejects_huge_cardinality_before_allocation(
    graph: kglite.KnowledgeGraph,
) -> None:
    with pytest.raises(kglite.CypherExecutionError, match=r"range\(\)"):
        graph.cypher(
            "RETURN range($start, $end, 1) AS r",
            params={"start": I64_MIN, "end": I64_MAX},
        )

    with pytest.raises(kglite.CypherExecutionError, match="max_rows"):
        graph.cypher("RETURN range(0, 10) AS r", max_rows=5)
    with pytest.raises(kglite.CypherExecutionError, match="256 MiB"):
        graph.cypher("RETURN range(0, 100000000) AS r")
    assert graph.cypher("RETURN range(0, 4) AS r", max_rows=5).to_list() == [{"r": [0, 1, 2, 3, 4]}]


@pytest.mark.parametrize("amount", [I64_MIN, I64_MAX])
def test_add_days_extrema_return_null(graph: kglite.KnowledgeGraph, amount: int) -> None:
    assert graph.cypher(
        "RETURN add_days(date('2024-01-01'), $amount) AS d",
        params={"amount": amount},
    ).to_list() == [{"d": None}]


@pytest.mark.parametrize("fn", ["add_months", "add_years"])
@pytest.mark.parametrize("amount", [I64_MIN, I64_MAX, 2**32, -(2**32)])
def test_calendar_shift_unsupported_magnitudes_return_null(graph: kglite.KnowledgeGraph, fn: str, amount: int) -> None:
    row = graph.cypher(
        f"RETURN {fn}(date('2024-01-31'), $amount) AS d",
        params={"amount": amount},
    ).to_list()[0]
    assert row["d"] is None


def test_add_years_preserves_leap_day_policy(graph: kglite.KnowledgeGraph) -> None:
    assert graph.cypher("RETURN add_years(date('2024-02-29'), 1) AS d").to_list() == [{"d": "2025-02-28"}]
    assert graph.cypher("RETURN add_years(date('2024-02-29'), -1) AS d").to_list() == [{"d": "2023-02-28"}]


@pytest.mark.parametrize(
    "query",
    [
        "RETURN duration({years: $n}) AS d",
        "RETURN duration({hours: $n}) AS d",
        "RETURN duration({months: 2147483648}) AS d",
        "RETURN duration({days: 2147483648}) AS d",
        "RETURN duration({seconds: $n, minutes: 1}) AS d",
    ],
)
def test_duration_construction_rejects_component_overflow(graph: kglite.KnowledgeGraph, query: str) -> None:
    with pytest.raises(kglite.CypherExecutionError, match=r"duration\(\)"):
        graph.cypher(query, params={"n": I64_MAX})


def test_duration_addition_and_multiplication_are_checked(
    graph: kglite.KnowledgeGraph,
) -> None:
    row = graph.cypher(
        "WITH duration({months: 2, days: 3, seconds: 4}) * 3 AS d "
        "RETURN d.months AS m, d.days AS days, d.seconds AS seconds"
    ).to_list()[0]
    assert row == {"m": 6, "days": 9, "seconds": 12}

    with pytest.raises(kglite.CypherExecutionError, match="Duration month"):
        graph.cypher("RETURN duration({months: 2147483647}) + duration({months: 1}) AS d")

    with pytest.raises(kglite.CypherExecutionError, match="multiplication"):
        graph.cypher("RETURN duration({months: 2147483647}) * 2 AS d")

    # A large factor is valid when every resulting component still fits.
    row = graph.cypher(
        "WITH duration({seconds: 0}) * $factor AS d RETURN d.seconds AS seconds",
        params={"factor": I64_MAX},
    ).to_list()[0]
    assert row == {"seconds": 0}

    with pytest.raises(kglite.CypherExecutionError, match="subtraction"):
        graph.cypher("RETURN duration({months: -2147483648}) - duration({months: 1}) AS d")


def test_duration_rejects_fractional_or_non_finite_components(
    graph: kglite.KnowledgeGraph,
) -> None:
    for value in (1.5, float("inf"), float("nan")):
        with pytest.raises(kglite.CypherExecutionError, match="expects a number"):
            graph.cypher("RETURN duration({seconds: $value}) AS d", params={"value": value})


def test_date_and_timestamp_arithmetic_return_null_on_unrepresentable_shift(
    graph: kglite.KnowledgeGraph,
) -> None:
    assert graph.cypher("RETURN date('2024-01-01') + $days AS d", params={"days": I64_MAX}).to_list() == [{"d": None}]
    assert graph.cypher(
        "RETURN datetime('2024-01-01T00:00:00') + duration({seconds: $seconds}) AS d",
        params={"seconds": I64_MAX},
    ).to_list() == [{"d": None}]
