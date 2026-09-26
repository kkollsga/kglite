"""Date cells read the same through `add_nodes` and a blueprint.

Registry data writes dates as ISO 8601 basic `YYYYMMDD`. A blueprint `"date"`
column used to read those eight digits as epoch milliseconds — every value
became 1970-01-01, silently — while `add_nodes` stored NULL. Both routes now
read `YYYYMMDD` (as text, an integer, or a whole-number float from a pandas
column with a gap) as the date it spells, and report the cells that are not a
date. An empty cell is a missing value and is not reported.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import kglite

FRAME = pd.DataFrame(
    {
        "id": ["a", "b", "c", "d"],
        # Text: a valid basic date, a missing cell, a blank one, an impossible
        # month, and (below) a small integer epoch milliseconds would put on
        # 1970-01-01.
        "vf": ["19650701", "", "  ", "19651301"],
        # A pandas integer column with a gap arrives as float.
        "vt": [19910201.0, np.nan, 20100101.0, 7.0],
    }
)

EXPECTED = [
    {"id": "a", "vf": "1965-07-01", "vt": "1991-02-01"},
    {"id": "b", "vf": None, "vt": None},
    {"id": "c", "vf": None, "vt": "2010-01-01"},
    {"id": "d", "vf": None, "vt": None},
]

ROWS = "MATCH (m:M) RETURN m.id AS id, m.vf AS vf, m.vt AS vt ORDER BY id"


def _messages(caught) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def test_add_nodes_reads_basic_dates_and_reports_only_real_failures() -> None:
    graph = kglite.KnowledgeGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph.add_nodes(FRAME, "M", "id", column_types={"vf": "date", "vt": "date"})
    assert graph.cypher(ROWS).to_list() == EXPECTED
    messages = _messages(caught)
    vf = [m for m in messages if "Column 'vf'" in m]
    vt = [m for m in messages if "Column 'vt'" in m]
    # The blank and empty cells are missing values, not parse failures.
    assert len(vf) == 1 and "1 value could not be parsed" in vf[0] and "'19651301'" in vf[0], vf
    assert len(vt) == 1 and "1 value could not be parsed" in vt[0] and "7.0" in vt[0], vt
    assert "YYYYMMDD" in vf[0]


def test_add_nodes_reads_an_integer_column_of_basic_dates() -> None:
    graph = kglite.KnowledgeGraph()
    frame = pd.DataFrame({"id": ["a"], "vf": np.array([19650701], dtype="int64")})
    graph.add_nodes(frame, "M", "id", column_types={"vf": "date"})
    assert graph.cypher("MATCH (m:M) RETURN m.vf AS vf").to_list() == [{"vf": "1965-07-01"}]


def test_blueprint_reads_basic_dates_like_add_nodes(tmp_path) -> None:
    FRAME.to_csv(tmp_path / "m.csv", index=False)
    blueprint = {
        "settings": {"root": str(tmp_path)},
        "nodes": {"M": {"csv": "m.csv", "pk": "id", "properties": {"vf": "date", "vt": "date"}}},
    }
    (tmp_path / "bp.json").write_text(json.dumps(blueprint), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph = kglite.from_blueprint(str(tmp_path / "bp.json"))
    assert graph.cypher(ROWS).to_list() == EXPECTED
    messages = _messages(caught)
    vf = [m for m in messages if "column 'vf' is declared date" in m]
    vt = [m for m in messages if "column 'vt' is declared date" in m]
    assert len(vf) == 1 and "1 cell(s) are not a date" in vf[0] and "'19651301'" in vf[0], messages
    assert len(vt) == 1 and "1 cell(s) are not a date" in vt[0] and "'7.0'" in vt[0], messages


def test_blueprint_keeps_epoch_milliseconds(tmp_path) -> None:
    (tmp_path / "m.csv").write_text("id,d\na,1609459200000\n", encoding="utf-8")
    blueprint = {
        "settings": {"root": str(tmp_path)},
        "nodes": {"M": {"csv": "m.csv", "pk": "id", "properties": {"d": "date"}}},
    }
    (tmp_path / "bp.json").write_text(json.dumps(blueprint), encoding="utf-8")
    graph = kglite.from_blueprint(str(tmp_path / "bp.json"))
    assert graph.cypher("MATCH (m:M) RETURN m.d AS d").to_list() == [{"d": "2021-01-01"}]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("19650701", "1965-07-01"), ("1965-07-01", "1965-07-01"), ("19651301", None)],
)
def test_cypher_date_reads_the_basic_format(text: str, expected: str | None) -> None:
    graph = kglite.KnowledgeGraph()
    assert graph.cypher("RETURN date($t) AS d", params={"t": text}).to_list() == [{"d": expected}]
