"""Characterization (golden) tests for the KnowledgeGraph decomposition.

These pin the *cross-field* semantics that the storage / lifecycle / cursor
decomposition could silently break — Clone-vs-copy field preservation, fluent
selection inheritance, temporal-context propagation, reports accumulation,
last_mutation_stats lifecycle, source_path on derived views, and the update()
special case. They observe only **public behavior**, so they must stay green
byte-for-byte through every internal refactoring phase.

If one of these changes, the decomposition changed observable behavior — that
is a regression unless explicitly intended and re-pinned.
"""

import pandas as pd
import pytest

import kglite


def _people(n: int = 5) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(n)),
                "name": [f"p{i}" for i in range(n)],
                "team": ["A" if i % 2 == 0 else "B" for i in range(n)],
            }
        ),
        "Person",
        "id",
        "name",
    )
    return g


def _count(view, q: str = "MATCH (n:Person) RETURN count(n) AS c") -> int:
    return view.cypher(q).to_list()[0]["c"]


# ── copy(): independent deep copy, resets cursor + save target ───────────────


def test_copy_is_independent_both_directions():
    g = _people(3)
    c = g.copy()
    c.cypher("CREATE (n:Person {id: 99, name: 'x'})")
    assert _count(g) == 3, "mutating the copy must not affect the original"
    g.cypher("CREATE (n:Person {id: 100, name: 'y'})")
    assert _count(c) == 4, "mutating the original must not affect the copy"
    assert _count(g) == 4


def test_copy_resets_selection():
    g = _people(4)
    selected = g.select("Person").where({"team": "A"})
    # copy() of a narrowed view starts from a fresh (full) selection.
    c = selected.copy()
    assert len(c.select("Person").to_df()) == 4


def test_copy_has_no_save_target():
    # copy() resets source_path → save() with no path has nowhere to write.
    c = _people(2).copy()
    with pytest.raises((ValueError, OSError)):
        c.save()


# ── Clone / derived views: preserve identity fields ──────────────────────────


def test_derived_view_preserves_source_path(tmp_path):
    p = tmp_path / "g.kgl"
    _people(3).save(str(p))
    g = kglite.load(str(p))
    # A fluent-derived view (a Clone) inherits the origin path: save() with no
    # arg writes back to the same file.
    g.select("Person").save()
    reloaded = kglite.load(str(p))
    assert _count(reloaded) == 3


def test_default_timeout_inherited_by_derived_view():
    g = _people(3)
    g.set_default_timeout(1234)
    assert g.get_default_timeout() == 1234
    # A derived view inherits the configured default.
    assert g.select("Person").get_default_timeout() == 1234


# ── selection inheritance through a fluent chain ─────────────────────────────


def test_select_narrows_and_chain_inherits():
    g = _people(6)  # teams: A,B,A,B,A,B → 3 A, 3 B
    team_a = g.select("Person").where({"team": "A"})
    assert len(team_a.to_df()) == 3
    # chaining narrows further off the inherited selection
    narrowed = team_a.where({"id": {"<": 3}})  # ids 0,2 are team A and < 3
    assert len(narrowed.to_df()) == 2


# ── temporal_context propagates to derived views ────────────────────────────


def test_temporal_context_propagates_through_derived_view():
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["early", "mid", "late"],
                "date_from": ["2000-01-01", "2010-06-01", "2020-01-01"],
                "date_to": ["2010-05-31", "2019-12-31", None],
            }
        ),
        "Status",
        "id",
        "title",
        column_types={"date_from": "validFrom", "date_to": "validTo"},
    )
    # date() sets temporal context and returns a derived handle; the context
    # survives into a subsequent fluent op on that handle.
    as_of_2015 = g.date("2015-06-01")
    rows = as_of_2015.select("Status").to_df()
    # only the "mid" status (valid 2010-2019) is active as of 2015
    assert list(rows["title"]) == ["mid"]


# ── reports accumulate; last_report reflects the latest op ───────────────────


def test_reports_accumulate_across_operations():
    g = kglite.KnowledgeGraph()
    start = g.operation_index()
    g.add_nodes(pd.DataFrame({"id": [1], "name": ["a"]}), "Person", "id", "name")
    after_first = g.operation_index()
    assert after_first > start
    g.add_nodes(pd.DataFrame({"id": [2], "name": ["b"]}), "Person", "id", "name")
    assert g.operation_index() > after_first
    # last_report is a dict describing the most recent operation
    assert isinstance(g.last_report(), dict)


# ── last_mutation_stats lifecycle ────────────────────────────────────────────


def test_last_mutation_stats_lifecycle():
    g = _people(2)
    # last_mutation_stats is a property reflecting only the most recent cypher
    # mutation (add_nodes is not a cypher mutation).
    g.cypher("CREATE (n:Person {id: 50, name: 'new'})")
    stats = g.last_mutation_stats
    assert stats is not None
    assert stats["nodes_created"] == 1
    g.cypher("MATCH (n:Person {id: 50}) SET n.name = 'edited'")
    stats2 = g.last_mutation_stats
    assert stats2["properties_set"] >= 1
    assert stats2["nodes_created"] == 0  # fresh stats per mutation, not cumulative


# ── update(): mutates the graph and returns a usable handle ──────────────────


def test_update_on_chained_view_returns_mutated_graph():
    g = _people(4)
    # update() on a chained view applies to a CoW clone: the RETURNED graph sees
    # the change, the source `g` does not (the well-documented chained-view
    # mutation semantics). update() returns {graph, nodes_updated, report_index}.
    r = g.select("Person").where({"team": "A"}).update({"flagged": True})
    assert r["nodes_updated"] == 2  # ids 0, 2 are team A
    returned = r["graph"]
    seen = returned.cypher("MATCH (n:Person) WHERE n.flagged = true RETURN count(n) AS c").to_list()[0]["c"]
    assert seen == 2
    # the source graph is unchanged (mutation landed on the chained clone)
    src = g.cypher("MATCH (n:Person) WHERE n.flagged = true RETURN count(n) AS c").to_list()[0]["c"]
    assert src == 0
