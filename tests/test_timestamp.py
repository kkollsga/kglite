"""Timestamp (date + time-of-day) value type — 0.12 Cluster 1.

Complements the date-only `DateTime`: a `datetime.datetime` property
round-trips with full second precision, `datetime()` / `localdatetime()`
return real timestamps, and timestamps persist through `.kgl` save/load.
"""

import datetime

import kglite


def test_datetime_property_roundtrips_as_datetime():
    g = kglite.KnowledgeGraph()
    ts = datetime.datetime(2024, 3, 15, 10, 30, 45)
    g.cypher("CREATE (:Event {id: 1, ts: $ts})", params={"ts": ts})
    got = g.cypher("MATCH (e:Event) RETURN e.ts AS ts")[0]["ts"]
    assert got == ts
    assert isinstance(got, datetime.datetime)


def test_date_property_stays_date_only():
    # A pure date (no time) maps to the date-only DateTime, not Timestamp.
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Event {id: 1, d: $d})", params={"d": datetime.date(2024, 3, 15)})
    got = g.cypher("MATCH (e:Event) RETURN e.d AS d")[0]["d"]
    assert got == "2024-03-15"  # DateTime renders as an ISO date string


def test_datetime_constructor_returns_timestamp():
    g = kglite.KnowledgeGraph()
    v = g.cypher("RETURN datetime('2024-03-15T10:30:45') AS v")[0]["v"]
    assert v == datetime.datetime(2024, 3, 15, 10, 30, 45)


def test_timestamp_plus_duration_applies_seconds():
    g = kglite.KnowledgeGraph()
    v = g.cypher("RETURN datetime('2024-03-15T10:30:00') + duration({hours: 2, minutes: 15}) AS v")[0]["v"]
    assert v == datetime.datetime(2024, 3, 15, 12, 45, 0)


def test_duration_between_timestamps_has_second_precision():
    g = kglite.KnowledgeGraph()
    secs = g.cypher(
        "RETURN duration.between(datetime('2024-03-15T10:00:00'), datetime('2024-03-15T10:00:30')).seconds AS s"
    )[0]["s"]
    assert secs == 30


def test_timestamp_comparison_filter():
    g = kglite.KnowledgeGraph()
    g.cypher(
        "UNWIND $rows AS r CREATE (:Event {id: r.id, ts: r.ts})",
        params={
            "rows": [
                {"id": 1, "ts": datetime.datetime(2024, 1, 1, 9, 0, 0)},
                {"id": 2, "ts": datetime.datetime(2024, 1, 1, 17, 30, 0)},
            ]
        },
    )
    rows = g.cypher("MATCH (e:Event) WHERE e.ts > datetime('2024-01-01T12:00:00') RETURN e.id AS id").to_list()
    assert [r["id"] for r in rows] == [2]


def test_timestamp_survives_save_load(tmp_path):
    ts = datetime.datetime(2024, 3, 15, 10, 30, 45)
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Event {id: 1, ts: $ts})", params={"ts": ts})
    path = str(tmp_path / "ts.kgl")
    g.save(path)

    reloaded = kglite.load(path)
    got = reloaded.cypher("MATCH (e:Event) RETURN e.ts AS ts")[0]["ts"]
    assert got == ts
