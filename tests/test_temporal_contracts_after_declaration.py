"""The temporal contracts a declared type keeps after it is declared.

- Writes onto a declared type are not re-validated (the declaration costs the
  write path nothing). A non-date bound written later raises from the next
  temporal filter that reads it, naming the element and the property; an
  inverted interval is valid on no date — `valid_during` over any range
  included, which used to report it as overlapping a range that covered both
  bounds. A property-type constraint refuses such a write up front.
- The fluent date arguments take a `datetime.date`, a `datetime.datetime` or a
  datetime string, as `cypher()` parameters do, at date grain.
- `traverse()` under a date context filters the relationships, not the target
  nodes by their own declaration; `.valid_at()` does that.
- Load errors number rows by 0-based position and say so.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import kglite

MODES = [None, "mapped", "disk"]
MODE_IDS = ["memory", "mapped", "disk"]


def _graph(storage, tmp_path) -> kglite.KnowledgeGraph:
    if storage == "disk":
        g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    elif storage == "mapped":
        g = kglite.KnowledgeGraph(storage="mapped")
    else:
        g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"code": ["1", "2"], "vf": ["2000-01-01", "2001-01-01"], "vt": ["2005-01-01", None]}),
        "M",
        "code",
        column_types={"vf": "validFrom", "vt": "validTo"},
        convention="half_open",
    )
    return g


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
def test_a_later_non_date_bound_raises_naming_the_element(storage, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    # Accepted: writes onto a declared type are not re-validated.
    g.cypher("MATCH (m:M {code: '2'}) SET m.vt = 20210101").to_list()
    named = r"(vt on node '2'|node '2', property 'vt').*20210101 \(INTEGER\) is not a date"
    for query in (
        "MATCH (m:M) WHERE valid_at(m, '2003') RETURN m.code",
        "MATCH (m:M) WHERE valid_during(m, '2003', '2004') RETURN count(*) AS c",
    ):
        with pytest.raises(kglite.CypherExecutionError, match=named):
            g.cypher(query).to_list()
    with pytest.raises(ValueError, match=named):
        g.select("M", temporal=False).valid_at("2003")
    with pytest.raises(ValueError, match=named):
        g.date("2003").select("M")


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
def test_an_inverted_interval_is_valid_on_no_date(storage, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    g.cypher("MATCH (m:M {code: '1'}) SET m.vt = date('1990-01-01')").to_list()
    for year in ("1980", "1995", "2003"):
        rows = g.cypher(f"MATCH (m:M {{code: '1'}}) WHERE valid_at(m, '{year}') RETURN m").to_list()
        assert rows == []
    overlaps = g.cypher("MATCH (m:M {code: '1'}) WHERE valid_during(m, '1900', '2100') RETURN m").to_list()
    assert overlaps == []
    assert g.select("M", temporal=False).valid_during("1900", "2100").len() == 1  # only '2'


def test_a_property_type_constraint_refuses_the_write_up_front() -> None:
    g = _graph(None, None)
    g.cypher("CREATE CONSTRAINT FOR (m:M) REQUIRE m.vt IS :: DATE").to_list()
    with pytest.raises(kglite.ConstraintViolationError):
        g.cypher("MATCH (m:M {code: '2'}) SET m.vt = 20210101").to_list()


DATE_FORMS = [
    "2003-06-30",
    "2003-06-30T12:00",
    dt.date(2003, 6, 30),
    dt.datetime(2003, 6, 30, 12, 0),
    dt.datetime(2003, 6, 30, 23, 30, tzinfo=dt.timezone(dt.timedelta(hours=2))),
]


@pytest.mark.parametrize("storage", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("when", DATE_FORMS, ids=["text", "datetime-text", "date", "datetime", "aware"])
def test_fluent_date_arguments_take_dates_and_datetimes(storage, when, tmp_path) -> None:
    g = _graph(storage, tmp_path)
    everything = g.select("M", temporal=False)
    assert sorted(n["id"] for n in everything.valid_at(when).collect()) == ["1", "2"]
    assert everything.valid_during(when, dt.date(2004, 1, 1)).len() == 2
    assert g.date(when).select("M").len() == 2
    assert g.date(when, "2004").select("M").len() == 2
    later = dt.date(2006, 1, 1)
    assert g.select("M", temporal=False).valid_at(later).len() == 1
    assert g.date(later).select("M").len() == 1


def test_fluent_date_arguments_refuse_other_types() -> None:
    g = _graph(None, None)
    with pytest.raises(TypeError, match="date must be a date string, a datetime.date or a datetime.datetime"):
        g.select("M", temporal=False).valid_at(2003)
    with pytest.raises(kglite.ArgumentError, match="also accepted"):
        g.select("M", temporal=False).valid_at("garbage")
    assert g.date("all").select("M").len() == 2


def test_traverse_under_a_date_filters_relationships_not_targets() -> None:
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (a:A {name: 'a', vf: date('2000-01-01'), vt: date('2005-01-01')}), (p:P {name: 'p'}),"
        " (a)-[:IN {vf: date('2000-01-01'), vt: date('2030-01-01')}]->(p)"
    ).to_list()
    g.set_temporal("A", "vf", "vt", convention="half_open")
    g.set_temporal("IN", "vf", "vt", convention="half_open")
    targets = g.date("2010").select("P").traverse("IN", direction="incoming")
    # The relationship is valid in 2010, so `a` is reached although its own
    # interval ended in 2005 ...
    assert [n["title"] for n in targets.collect()] == ["a"]
    # ... and `.valid_at()` under the same context filters it.
    assert targets.valid_at().len() == 0
    # Relationship dates accept dates too.
    assert g.select("P", temporal=False).traverse("IN", direction="incoming", at=dt.date(2010, 1, 1)).len() == 1


def test_load_errors_number_rows_by_zero_based_position() -> None:
    g = kglite.KnowledgeGraph()
    frame = pd.DataFrame({"code": ["a", "b"], "vf": ["2000-01-01", "2010-01-01"], "vt": ["2005-01-01", "2001-01-01"]})
    with pytest.raises(kglite.ArgumentError, match=r"row 1 \(0-based\) of the load"):
        g.add_nodes(frame, "M", "code", column_types={"vf": "validFrom", "vt": "validTo"})
    with pytest.warns(UserWarning, match=r"row 1 \(0-based\) holds 'garbage'"):
        g.add_nodes(
            pd.DataFrame({"code": ["a", "b"], "d": ["2000-01-01", "garbage"]}),
            "N",
            "code",
            column_types={"d": "datetime"},
        )
    with pytest.raises(kglite.ArgumentError, match=r"row 1 \(0-based\) has"):
        g.add_nodes(pd.DataFrame({"code": ["a", None]}), "Q", "code", on_invalid="error")
