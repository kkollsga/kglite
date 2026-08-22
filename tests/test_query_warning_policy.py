"""``ResultView.warnings`` and the process-global query-warning policy.

Query warnings ride two independent channels:

* the **structured** one — ``ResultView.diagnostics["warnings"]``, and its
  shortcut ``ResultView.warnings``. Unconditional: no policy touches it.
* the **echo** — where a warning is *announced*. ``"stderr"`` (the default,
  and what every release before this one did), ``"silent"``, or ``"pywarn"``
  (a ``UserWarning`` through the :mod:`warnings` module instead of stderr).

The default-behaviour pin lives in ``test_cypher_schema_warnings.py`` (capfd
against an untouched policy); this module owns the accessor and the knob.
"""

from __future__ import annotations

import warnings as pywarnings

import pandas as pd
import pytest

import kglite


@pytest.fixture(autouse=True)
def _restore_policy():
    """The policy is process-global — never leak one test's choice."""
    try:
        yield
    finally:
        kglite.set_query_warning_policy("stderr")


def _maritime() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"vid": [1, 2], "vname": ["Aurora", "Bravo"], "imo_number": [11, 22]}),
        "Vessel",
        "vid",
        "vname",
    )
    return g


TYPO_QUERY = "MATCH (v:Vessel) RETURN v.imo"
CLEAN_QUERY = "MATCH (v:Vessel) RETURN v.imo_number"


# ── ResultView.warnings ────────────────────────────────────────────────────


def test_warnings_property_is_empty_on_a_clean_query():
    g = _maritime()
    assert g.cypher(CLEAN_QUERY).warnings == []


def test_warnings_property_reports_a_property_typo():
    g = _maritime()
    rv = g.cypher(TYPO_QUERY)
    assert rv.warnings == rv.diagnostics["warnings"]
    assert len(rv.warnings) == 1
    assert "RETURN projects property 'imo'" in rv.warnings[0]


def test_warnings_property_reports_a_label_typo():
    g = _maritime()
    rv = g.cypher("MATCH (v:Vessl) RETURN v")
    assert any("unknown node label 'Vessl'" in w for w in rv.warnings), rv.warnings
    assert any("Did you mean 'Vessel'?" in w for w in rv.warnings), rv.warnings


def test_warnings_property_is_empty_for_a_view_that_did_not_come_from_a_query():
    """``head()`` slices carry no diagnostics — the accessor returns ``[]``,
    not ``None`` and not an attribute error."""
    g = _maritime()
    sliced = g.cypher(TYPO_QUERY).head(1)
    assert sliced.diagnostics is None
    assert sliced.warnings == []


# ── the policy knob ────────────────────────────────────────────────────────


def test_default_policy_is_stderr():
    assert kglite.get_query_warning_policy() == "stderr"


def test_policy_roundtrips():
    for policy in ("silent", "pywarn", "stderr"):
        kglite.set_query_warning_policy(policy)
        assert kglite.get_query_warning_policy() == policy


def test_unknown_policy_is_refused_and_leaves_the_policy_alone():
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.set_query_warning_policy("shout")
    assert "silent" in str(excinfo.value) and "pywarn" in str(excinfo.value)
    assert kglite.get_query_warning_policy() == "stderr"


def test_silent_suppresses_stderr_but_keeps_the_structured_channel(capfd):
    g = _maritime()
    kglite.set_query_warning_policy("silent")
    rv = g.cypher(TYPO_QUERY)
    assert "warning:" not in capfd.readouterr().err
    assert len(rv.warnings) == 1  # the structured channel is unconditional


def test_stderr_is_what_the_default_does(capfd):
    g = _maritime()
    kglite.set_query_warning_policy("stderr")
    rv = g.cypher(TYPO_QUERY)
    assert "warning: RETURN projects property 'imo'" in capfd.readouterr().err
    assert len(rv.warnings) == 1


def test_pywarn_routes_through_the_warnings_module_instead_of_stderr(capfd):
    g = _maritime()
    kglite.set_query_warning_policy("pywarn")
    with pytest.warns(UserWarning, match="RETURN projects property 'imo'"):
        rv = g.cypher(TYPO_QUERY)
    # `pywarn` *replaces* the stderr echo — warnings.warn does its own routing.
    assert "warning:" not in capfd.readouterr().err
    assert len(rv.warnings) == 1


def test_pywarn_is_an_error_under_filterwarnings_error():
    """The reason `pywarn` is never the default: `-W error` turns it into a
    raise out of `cypher()`."""
    g = _maritime()
    kglite.set_query_warning_policy("pywarn")
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("error")
        with pytest.raises(UserWarning):
            g.cypher(TYPO_QUERY)


def test_clean_query_warns_nothing_under_pywarn():
    g = _maritime()
    kglite.set_query_warning_policy("pywarn")
    with pywarnings.catch_warnings(record=True) as caught:
        pywarnings.simplefilter("always")
        g.cypher(CLEAN_QUERY)
    assert caught == []


# ── the policy reaches every path diagnostics do ───────────────────────────


def test_policy_reaches_session_execute():
    g = _maritime()
    session = g.session()
    kglite.set_query_warning_policy("pywarn")
    with pytest.warns(UserWarning, match="RETURN projects property 'imo'"):
        rv = session.execute(TYPO_QUERY)
    assert len(rv.warnings) == 1


def test_policy_reaches_session_cypher():
    g = _maritime()
    session = g.session()
    kglite.set_query_warning_policy("silent")
    rv = session.cypher(TYPO_QUERY)
    assert len(rv.warnings) == 1


def test_policy_reaches_a_transaction():
    g = _maritime()
    kglite.set_query_warning_policy("pywarn")
    with pytest.warns(UserWarning, match="RETURN projects property 'imo'"):
        with g.begin() as tx:
            rv = tx.cypher(TYPO_QUERY)
    assert len(rv.warnings) == 1


def test_policy_reaches_a_frozen_graph():
    g = _maritime()
    frozen = g.freeze()
    kglite.set_query_warning_policy("pywarn")
    with pytest.warns(UserWarning, match="RETURN projects property 'imo'"):
        frozen.cypher(TYPO_QUERY)


def test_policy_reaches_a_mutation():
    g = _maritime()
    kglite.set_query_warning_policy("pywarn")
    with pytest.warns(UserWarning, match="unknown node label 'Vessl'"):
        g.cypher("MATCH (v:Vessl) SET v.flag = 1")


# ── to_df / FORMAT CSV: no structured channel, but the echo still fires ────


def test_to_df_carries_no_diagnostics_but_still_warns_under_pywarn():
    """A DataFrame has nowhere to hang diagnostics, so the policy channel is
    the only one a `to_df=True` caller has. It fires."""
    g = _maritime()
    kglite.set_query_warning_policy("pywarn")
    with pytest.warns(UserWarning, match="RETURN projects property 'imo'"):
        df = g.cypher(TYPO_QUERY, to_df=True)
    assert isinstance(df, pd.DataFrame)
    assert not hasattr(df, "diagnostics")


def test_to_df_warns_on_stderr_by_default(capfd):
    g = _maritime()
    g.cypher(TYPO_QUERY, to_df=True)
    assert "warning: RETURN projects property 'imo'" in capfd.readouterr().err


def test_csv_output_still_warns_under_pywarn():
    g = _maritime()
    kglite.set_query_warning_policy("pywarn")
    with pytest.warns(UserWarning, match="RETURN projects property 'imo'"):
        csv = g.cypher(TYPO_QUERY + " FORMAT CSV")
    assert isinstance(csv, str)


def test_csv_output_is_silent_under_silent(capfd):
    g = _maritime()
    kglite.set_query_warning_policy("silent")
    g.cypher(TYPO_QUERY + " FORMAT CSV")
    assert "warning:" not in capfd.readouterr().err
