"""Regression: updating a String property on a columnar-backed node
must not corrupt the offsets table of adjacent rows.

Bug shape (kglite-docs 2026-05-28): updating `s.verification_status`
from "unverified" → "verified" on a Summary node panicked with
``slice index starts at 108 but ends at 30`` from
``mmap_vec.rs:746``. Root cause: `TypedColumn::Str::set` appended the
new bytes at the end of the data buffer and rewrote both
``offsets[idx]`` and ``offsets[idx+1]``. But ``offsets[idx+1]`` is
also the START of the NEXT row, so the next row's start jumped to
the appended end, leaving its (unchanged) ``offsets[idx+2]`` end
earlier in the buffer.
"""

from pathlib import Path

import pandas as pd
import pytest

import kglite


@pytest.fixture
def graph_with_summaries(tmp_path: Path) -> kglite.KnowledgeGraph:
    """Five summaries with String properties, saved + reloaded so that
    properties live in PropertyStorage::Columnar."""
    df = pd.DataFrame(
        {
            "id": [f"sum_{i}" for i in range(5)],
            "title": [f"Summary {i}" for i in range(5)],
            "status": ["unverified"] * 5,
            "verifier": ["agent_alice"] * 5,
            "notes": ["initial review pending"] * 5,
        }
    )
    g = kglite.KnowledgeGraph()
    g.add_nodes(df, "Summary", "id", "title")
    save_path = tmp_path / "summaries.kgl"
    g.save(str(save_path))
    return kglite.load(str(save_path))


def test_second_string_write_on_same_property(graph_with_summaries: kglite.KnowledgeGraph):
    """First SET succeeds; SECOND SET on same property used to panic
    in mmap_vec.rs:746."""
    g = graph_with_summaries

    g.cypher(
        "MATCH (s:Summary {id: $id}) SET s.status = $v",
        params={"id": "sum_2", "v": "verified"},
    )
    g.cypher(
        "MATCH (s:Summary {id: $id}) SET s.status = $v",
        params={"id": "sum_2", "v": "rejected"},
    )

    rows = g.cypher(
        "MATCH (s:Summary {id: $id}) RETURN s.status AS st",
        params={"id": "sum_2"},
    ).to_list()
    assert rows[0]["st"] == "rejected"


def test_multi_property_set_in_one_statement(
    graph_with_summaries: kglite.KnowledgeGraph,
):
    """Cypher SET assigning multiple String properties on one node
    in a single statement (the original kglite-docs trigger)."""
    g = graph_with_summaries

    g.cypher(
        "MATCH (s:Summary {id: $id}) SET s.status = $v, s.verifier = $w, s.notes = $n",
        params={
            "id": "sum_1",
            "v": "verified",
            "w": "agent_bob",
            "n": "reviewed in second pass",
        },
    )

    rows = g.cypher(
        "MATCH (s:Summary {id: $id}) RETURN s.status AS st, s.verifier AS vr, s.notes AS nt",
        params={"id": "sum_1"},
    ).to_list()
    assert rows[0]["st"] == "verified"
    assert rows[0]["vr"] == "agent_bob"
    assert rows[0]["nt"] == "reviewed in second pass"


def test_adjacent_rows_unaffected_by_string_update(
    graph_with_summaries: kglite.KnowledgeGraph,
):
    """Updating row N's String property must not corrupt rows N-1 or N+1."""
    g = graph_with_summaries

    g.cypher(
        "MATCH (s:Summary {id: $id}) SET s.status = $v",
        params={"id": "sum_2", "v": "verified"},
    )

    rows = g.cypher("MATCH (s:Summary) RETURN s.id AS id, s.status AS st, s.notes AS nt ORDER BY s.id").to_list()

    assert [r["id"] for r in rows] == [f"sum_{i}" for i in range(5)]
    assert rows[1]["st"] == "unverified"
    assert rows[2]["st"] == "verified"
    assert rows[3]["st"] == "unverified"
    for r in rows:
        assert r["nt"] == "initial review pending"


def test_string_update_then_save_then_reload(graph_with_summaries: kglite.KnowledgeGraph, tmp_path: Path):
    """Modified Strings must round-trip cleanly through save+load."""
    g = graph_with_summaries

    g.cypher(
        "MATCH (s:Summary {id: $id}) SET s.status = $v",
        params={"id": "sum_3", "v": "verified-and-archived"},
    )
    out_path = tmp_path / "after_update.kgl"
    g.save(str(out_path))

    g2 = kglite.load(str(out_path))
    rows = g2.cypher("MATCH (s:Summary) RETURN s.id AS id, s.status AS st ORDER BY s.id").to_list()
    assert rows[3]["st"] == "verified-and-archived"
    assert rows[0]["st"] == "unverified"
    assert rows[4]["st"] == "unverified"
