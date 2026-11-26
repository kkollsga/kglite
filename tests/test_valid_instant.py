"""valid_instant(): the instant of the statement's valid-time context.

A date for a date context, a datetime for a datetime context, today (UTC) under
the default context; an error where no instant exists (ALL, writes, no
context). It is not a valid_at() call, so a statement that uses it still gets
the default context.
"""

import datetime

import pytest

import kglite


def _graph(stored_default=None):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Plant {id: 1, title: 'A', vf: date('2000-01-01'), vt: date('2999-01-01')})")
    g.cypher("CREATE (:Plant {id: 2, title: 'B', vf: date('2000-01-01'), vt: date('2010-01-01')})")
    g.cypher("CALL db.temporal.declare({node: 'Plant', from: 'vf', to: 'vt', convention: 'closed'})")
    g.set_timeseries("Plant", resolution="month", channels=["ch"])
    keys = [f"2015-{m:02d}" for m in range(1, 13)]
    g.set_time_index(1, keys)
    g.add_ts_channel(1, "ch", [float(m) for m in range(1, 13)])
    if stored_default is not None:
        g.set_valid_time_default(stored_default, persist=True)
    return g


@pytest.fixture
def g():
    return _graph()


def one(g, query, **kw):
    return g.cypher(query, **kw).to_list()[0]["v"]


def test_literal_date_instant(g):
    assert one(g, "FOR VALID_TIME AS OF date('2015-06-15') MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v") == (
        datetime.date(2015, 6, 15)
    )


def test_parameter_instant(g):
    q = "FOR VALID_TIME AS OF $d MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v"
    assert one(g, q, params={"d": "2015-06-15"}) == datetime.date(2015, 6, 15)
    assert one(g, q, params={"d": datetime.date(2012, 2, 29)}) == datetime.date(2012, 2, 29)


def test_datetime_instant_stays_a_datetime(g):
    v = one(
        g, "FOR VALID_TIME AS OF datetime('2015-06-15T13:45:00') MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v"
    )
    assert v == datetime.datetime(2015, 6, 15, 13, 45)


def test_valid_at_parameter(g):
    assert one(g, "MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v", valid_at="2015-06-15") == datetime.date(
        2015, 6, 15
    )


def test_default_context_is_today(g):
    # Read the clock on both sides: a run that crosses midnight UTC sees either day.
    before = datetime.datetime.now(datetime.timezone.utc).date()
    instant = one(g, "MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v")
    after = datetime.datetime.now(datetime.timezone.utc).date()
    assert instant in {before, after}
    # The default context applied: B (valid to 2010) is hidden.
    assert g.cypher("MATCH (p:Plant) RETURN count(p) AS v").to_list()[0]["v"] == 1
    assert g.cypher("MATCH (p:Plant) WHERE valid_instant() = date() RETURN count(p) AS v").to_list()[0]["v"] == 1


def test_stored_date_default(g):
    g = _graph("2015-06-15")
    assert one(g, "MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v") == datetime.date(2015, 6, 15)


def test_stored_all_default_is_an_error():
    g = _graph("all")
    with pytest.raises(Exception, match="valid_instant"):
        one(g, "MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v")


def test_all_is_an_error(g):
    with pytest.raises(Exception, match=r"valid_instant\(\).*FOR VALID_TIME AS OF"):
        one(g, "FOR VALID_TIME ALL MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v")


def test_write_is_an_error(g):
    with pytest.raises(Exception, match="valid_instant"):
        g.cypher("MATCH (p:Plant {id: 1}) SET p.vt = valid_instant()")
    with pytest.raises(Exception, match="valid_instant"):
        g.cypher("CREATE (:Plant {id: 9, title: 'Z', vf: valid_instant()})")


def test_no_context_is_an_error():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Thing {id: 1})")
    with pytest.raises(Exception, match="valid_instant"):
        one(g, "MATCH (t:Thing) RETURN valid_instant() AS v")
    # An explicit context on a graph that declares nothing still has an instant.
    assert one(g, "FOR VALID_TIME ALL MATCH (t:Thing) RETURN 1 AS v") == 1


def test_arguments_are_refused(g):
    with pytest.raises(Exception, match="no arguments"):
        one(g, "FOR VALID_TIME AS OF date('2015-06-15') MATCH (p:Plant {id: 1}) RETURN valid_instant(1) AS v")


AS_OF = "FOR VALID_TIME AS OF date('2015-06-15') "
EXPECTED = datetime.date(2015, 6, 15)


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "body",
    [
        "MATCH (p:Plant {id: 1}) CALL { WITH p RETURN valid_instant() AS v } RETURN v",
        "MATCH (p:Plant {id: 1}) CALL { RETURN valid_instant() AS v } RETURN v",
        "MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v UNION MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v",
        "MATCH (p:Plant {id: 1}) WITH p, valid_instant() AS t WITH p, t RETURN t AS v",
        "MATCH (p:Plant) WHERE valid_instant() = date('2015-06-15') RETURN valid_instant() AS v",
        "UNWIND [1, 2] AS i MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v LIMIT 1",
        "MATCH (p:Plant {id: 1}) RETURN [x IN [1] | valid_instant()][0] AS v",
        "MATCH (p:Plant {id: 1}) RETURN p.title AS k, valid_instant() AS v ORDER BY valid_instant()",
    ],
)
def test_nested_scopes_and_pipeline(g, body, streaming):
    rows = g.cypher(AS_OF + body, streaming=streaming).to_list()
    assert rows and all(r["v"] == EXPECTED for r in rows)


def test_inline_property_map_is_per_execution(g):
    g.cypher("CREATE (:Marker {id: 1, at: date('2015-06-15')})")
    q = "MATCH (m:Marker {at: valid_instant()}) RETURN count(m) AS v"
    assert one(g, "FOR VALID_TIME AS OF date('2015-06-15') " + q) == 1
    assert one(g, "FOR VALID_TIME AS OF date('2015-06-16') " + q) == 0
    assert one(g, "FOR VALID_TIME AS OF date('2015-06-15') " + q) == 1


def test_plan_cache_does_not_freeze_the_instant(g):
    q = "FOR VALID_TIME AS OF $d MATCH (p:Plant {id: 1}) RETURN valid_instant() AS v"
    assert one(g, q, params={"d": "2015-01-01"}) == datetime.date(2015, 1, 1)
    assert one(g, q, params={"d": "2015-02-01"}) == datetime.date(2015, 2, 1)


def test_statement_with_valid_instant_still_gets_the_default(g):
    echo = g.cypher("MATCH (p:Plant) RETURN valid_instant() AS v").diagnostics["temporal"]
    assert echo["source"] == "default"
    skipped = g.cypher("MATCH (p:Plant) WHERE valid_at(p, date('2005-01-01')) RETURN count(p) AS v").diagnostics[
        "temporal"
    ]
    assert skipped["source"] == "skipped:valid_at"


def test_ts_at_at_the_instant(g):
    q = AS_OF + "MATCH (p:Plant {id: 1}) RETURN ts_at(p.ch, valid_instant()) AS v"
    assert one(g, q) == 6.0
    assert (
        one(
            g,
            "FOR VALID_TIME AS OF date('2015-09-30') MATCH (p:Plant {id: 1}) RETURN ts_at(p.ch, valid_instant()) AS v",
        )
        == 9.0
    )


def test_year_to_date(g):
    q = (
        "FOR VALID_TIME AS OF date('2015-06-15') MATCH (p:Plant {id: 1}) "
        "RETURN ts_sum(p.ch, date_truncate(valid_instant(), 'year'), valid_instant()) AS v"
    )
    assert one(g, q) == 1 + 2 + 3 + 4 + 5 + 6.0


def test_trailing_twelve_months(g):
    q = (
        "FOR VALID_TIME AS OF date('2015-09-15') MATCH (p:Plant {id: 1}) "
        "RETURN ts_sum(p.ch, add_months(date_truncate(valid_instant(), 'month'), -11), valid_instant()) AS v"
    )
    assert one(g, q) == sum(range(1, 10)) * 1.0


def test_datetime_instant_composes_through_date(g):
    q = (
        "FOR VALID_TIME AS OF datetime('2015-06-15T13:45:00') MATCH (p:Plant {id: 1}) "
        "RETURN ts_at(p.ch, valid_instant()) AS a, "
        "ts_sum(p.ch, date_truncate(date(valid_instant()), 'year'), valid_instant()) AS b"
    )
    row = g.cypher(q).to_list()[0]
    assert row["a"] == 6.0 and row["b"] == 21.0
