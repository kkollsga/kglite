"""``row_limit`` — a result-row retention cap with a mandatory truncation signal.

The cap is the deliberate opposite number to ``max_work_units``: that one
bounds *work* and raises, this one bounds *retained rows* and truncates. The
tests that matter most here are the ones proving the truncation is never
silent — a caller who reads only ``diagnostics``, only ``warnings``, or only
the announcement must still learn its result was cut, and must be able to say
"showing 5,000 of N" from the result alone.
"""

from __future__ import annotations

import warnings as pywarnings

import pandas as pd
import pytest

import kglite


@pytest.fixture(autouse=True)
def _restore_policy():
    """``set_query_warning_policy`` is process-global — never leak a choice."""
    try:
        yield
    finally:
        kglite.set_query_warning_policy("stderr")


def _items(n: int) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": list(range(1, n + 1)), "seq": list(range(1, n + 1))}),
        "Item",
        "id",
    )
    return g


ALL_ITEMS = "MATCH (n:Item) RETURN n.seq AS seq ORDER BY seq"


# ── the cap itself ─────────────────────────────────────────────────────────


def test_cap_truncates_and_reports_the_exact_total():
    g = _items(50)
    rv = g.cypher(ALL_ITEMS, row_limit=5)

    assert [row["seq"] for row in rv.to_list()] == [1, 2, 3, 4, 5]
    assert rv.diagnostics["row_limit"] == 5
    assert rv.diagnostics["total_rows"] == 50


def test_under_cap_result_carries_no_truncation_signal():
    g = _items(5)
    rv = g.cypher(ALL_ITEMS, row_limit=500)

    assert len(rv.to_list()) == 5
    assert rv.diagnostics["row_limit"] == 500
    assert rv.diagnostics["total_rows"] is None
    assert rv.warnings == []


def test_no_cap_leaves_both_diagnostics_fields_none():
    g = _items(5)
    rv = g.cypher(ALL_ITEMS)
    assert rv.diagnostics["row_limit"] is None
    assert rv.diagnostics["total_rows"] is None


def test_cap_of_zero_retains_nothing_and_still_counts():
    g = _items(7)
    rv = g.cypher(ALL_ITEMS, row_limit=0)

    assert rv.to_list() == []
    assert rv.diagnostics["row_limit"] == 0
    assert rv.diagnostics["total_rows"] == 7


def test_cap_keeps_the_real_top_n_of_an_ordered_result():
    g = _items(50)
    rv = g.cypher("MATCH (n:Item) RETURN n.seq AS seq ORDER BY seq DESC", row_limit=3)
    assert [row["seq"] for row in rv.to_list()] == [50, 49, 48]


def test_query_limit_and_row_limit_take_the_minimum():
    g = _items(50)

    capped = g.cypher(ALL_ITEMS + " LIMIT 20", row_limit=5)
    assert len(capped.to_list()) == 5
    assert capped.diagnostics["total_rows"] == 20

    uncapped = g.cypher(ALL_ITEMS + " LIMIT 3", row_limit=5)
    assert len(uncapped.to_list()) == 3
    assert uncapped.diagnostics["total_rows"] is None


def test_streaming_path_is_capped_too():
    """``streaming=True`` returns rows through the lazy descriptor."""
    g = _items(40)
    rv = g.cypher(ALL_ITEMS, row_limit=6, streaming=True)
    assert len(rv.to_list()) == 6
    assert rv.diagnostics["total_rows"] == 40


def test_mutation_return_is_capped_but_every_write_lands():
    g = _items(12)
    rv = g.cypher("MATCH (n:Item) SET n.touched = true RETURN n.seq AS seq", row_limit=3)

    assert len(rv.to_list()) == 3
    assert rv.diagnostics["total_rows"] == 12
    still_written = g.cypher("MATCH (n:Item) WHERE n.touched = true RETURN count(*) AS c")
    assert still_written.to_list()[0]["c"] == 12


def test_row_limit_does_not_rescue_a_work_budget_overrun():
    """The two knobs are orthogonal: the budget still raises."""
    g = _items(50)
    with pytest.raises(kglite.CypherExecutionError, match="max_work_units"):
        g.cypher(ALL_ITEMS, max_work_units=2, row_limit=1)


# ── the mandatory signal ───────────────────────────────────────────────────


def test_truncation_is_announced_as_a_query_warning():
    g = _items(50)
    rv = g.cypher(ALL_ITEMS, row_limit=5)

    assert len(rv.warnings) == 1
    assert "row_limit" in rv.warnings[0]
    assert "5 of 50" in rv.warnings[0]
    assert rv.warnings == rv.diagnostics["warnings"]


def test_truncation_reaches_the_pywarn_announcement_channel():
    """The `warning_policy` echo, not just the structured field."""
    kglite.set_query_warning_policy("pywarn")
    g = _items(50)
    with pywarnings.catch_warnings(record=True) as caught:
        pywarnings.simplefilter("always")
        g.cypher(ALL_ITEMS, row_limit=5)
    messages = [str(w.message) for w in caught]
    assert any("row_limit" in m and "5 of 50" in m for m in messages), messages


def test_truncation_is_announced_even_when_the_result_shape_cannot_carry_it():
    """``to_df=True`` returns a DataFrame with nowhere to put diagnostics."""
    kglite.set_query_warning_policy("pywarn")
    g = _items(50)
    with pywarnings.catch_warnings(record=True) as caught:
        pywarnings.simplefilter("always")
        df = g.cypher(ALL_ITEMS, row_limit=5, to_df=True)
    assert len(df) == 5
    assert any("row_limit" in str(w.message) for w in caught)


# ── the session-level default ──────────────────────────────────────────────


def test_default_row_limit_round_trips():
    g = _items(5)
    assert g.get_default_row_limit() is None
    g.set_default_row_limit(2)
    assert g.get_default_row_limit() == 2
    g.set_default_row_limit(None)
    assert g.get_default_row_limit() is None


def test_default_row_limit_applies_to_cypher():
    g = _items(50)
    g.set_default_row_limit(4)
    rv = g.cypher(ALL_ITEMS)
    assert len(rv.to_list()) == 4
    assert rv.diagnostics["total_rows"] == 50


def test_per_query_row_limit_overrides_the_default():
    g = _items(50)
    g.set_default_row_limit(4)
    assert len(g.cypher(ALL_ITEMS, row_limit=9).to_list()) == 9
    # ...including overriding it downward to zero, which is a real cap.
    assert g.cypher(ALL_ITEMS, row_limit=0).to_list() == []


# ── the other query entry points ───────────────────────────────────────────


def test_session_cypher_and_execute_take_the_cap():
    g = _items(30)
    s = g.session()

    read = s.cypher(ALL_ITEMS, row_limit=5)
    assert len(read.to_list()) == 5
    assert read.diagnostics["total_rows"] == 30

    written = s.execute("MATCH (n:Item) SET n.seen = true RETURN n.seq AS seq", row_limit=2)
    assert len(written.to_list()) == 2
    assert written.diagnostics["total_rows"] == 30


def test_frozen_graph_cypher_takes_the_cap():
    g = _items(30)
    rv = g.freeze().cypher(ALL_ITEMS, row_limit=5)
    assert len(rv.to_list()) == 5
    assert rv.diagnostics["total_rows"] == 30


def test_transaction_cypher_takes_the_cap():
    g = _items(30)
    with g.begin() as tx:
        rv = tx.cypher(ALL_ITEMS, row_limit=5)
        assert len(rv.to_list()) == 5
        assert rv.diagnostics["total_rows"] == 30
