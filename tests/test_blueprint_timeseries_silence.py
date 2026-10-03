"""The blueprint timeseries loader says what it drops, coerces or ignores.

Aggregate rows (`month = 0`), float-formatted time components (pandas writes
`2020.0` whenever a column holds a NaN), unknown keys inside a `timeseries`
block and a repeated pk each used to pass without a word.
"""

import json
import subprocess
import sys
import textwrap
import warnings

from kglite.blueprint import from_blueprint


def _build(tmp_path, nodes, csvs):
    for name, text in csvs.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    bp = {"settings": {"root": str(tmp_path)}, "nodes": nodes}
    (tmp_path / "bp.json").write_text(json.dumps(bp), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = from_blueprint(tmp_path / "bp.json", save=False)
    return g, [str(w.message) for w in caught]


def _series(extra_ts=None, **kw):
    ts = {
        "time_key": {"year": "yr", "month": "mo"},
        "resolution": "month",
        "channels": {"out": "val", "inv": "stock"},
    }
    ts.update(extra_ts or {})
    spec = {"csv": "r.csv", "pk": "pid", "title": "pid", "properties": {}, "timeseries": ts}
    spec.update(kw)
    return {"Plant": spec}


def _scalar(g, q):
    return g.cypher(q).to_list()[0]["v"]


def test_dropped_aggregate_rows_are_reported_with_the_channel_only_they_fill(tmp_path):
    csv = "pid,yr,mo,val,stock\n1,2020,1,1.0,\n1,2020,2,2.0,\n1,2020,0,3.0,500.0\n"
    g, msgs = _build(tmp_path, _series(), {"r.csv": csv})
    hits = [m for m in msgs if "aggregate row" in m]
    assert len(hits) == 1, msgs
    m = hits[0]
    assert "[Plant] dropped 1 aggregate row(s)" in m
    assert "channel 'inv' has values only" in m and "'out'" not in m
    assert '"filter"' in m
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_sum(p.out) AS v") == 3.0


def test_aggregate_rows_warn_once_when_the_spec_also_feeds_fk_edges(tmp_path):
    nodes = _series()
    nodes["Plant"]["connections"] = {"fk_edges": {"OF_SITE": {"target": "Site", "fk": "site"}}}
    nodes["Site"] = {"csv": "s.csv", "pk": "site", "title": "site", "properties": {}}
    csv = "pid,yr,mo,val,stock,site\n1,2020,1,1.0,1.0,a\n1,2020,0,3.0,5.0,a\n"
    _, msgs = _build(tmp_path, nodes, {"r.csv": csv, "s.csv": "site\na\n"})
    assert len([m for m in msgs if "aggregate row" in m]) == 1, msgs


def test_no_aggregate_rows_no_warning(tmp_path):
    csv = "pid,yr,mo,val,stock\n1,2020,1,1.0,1.0\n"
    _, msgs = _build(tmp_path, _series(), {"r.csv": csv})
    assert not [m for m in msgs if "aggregate row" in m], msgs


def test_float_formatted_time_components_land_on_the_right_dates(tmp_path):
    rows = ["1,2020.0,1.0,1.0,1.0", "1,2020.0,2.0,2.0,2.0", "1,2020.0,12.0,4.0,4.0", "1,2020.0,0.0,99.0,99.0"]
    csv = "pid,yr,mo,val,stock\n" + "\n".join(rows) + "\n"
    g, msgs = _build(tmp_path, _series(), {"r.csv": csv})
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_count(p.out) AS v") == 3
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_at(p.out, '2020-12') AS v") == 4.0
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_at(p.out, '2020-2') AS v") == 2.0
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_sum(p.out, '2020') AS v") == 7.0
    assert any("dropped 1 aggregate row(s)" in m for m in msgs), msgs


def test_a_non_whole_time_component_drops_the_row_and_says_so_once(tmp_path):
    rows = ["1,2020.5,1,5.0,1.0", "1,abc,2,6.0,1.0", "1,2020,3,7.0,1.0", "1,2021,1.5,8.0,1.0", "1,,4,9.0,1.0"]
    csv = "pid,yr,mo,val,stock\n" + "\n".join(rows) + "\n"
    g, msgs = _build(tmp_path, _series(), {"r.csv": csv})
    # Only the well-formed row is in the series; nothing is filed under year 0 or a coerced year.
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_count(p.out) AS v") == 1
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_sum(p.out, '2020') AS v") == 7.0
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_sum(p.out, '0') AS v") in (0.0, None)
    hits = [m for m in msgs if "not a whole number" in m]
    assert len(hits) == 1, msgs
    m = hits[0]
    assert "[Plant] dropped 4 row(s)" in m
    assert "yr" in m and "mo" in m
    assert "'2020.5'" in m and "'abc'" in m and "'1.5'" in m


def test_an_unknown_key_in_a_timeseries_block_warns(tmp_path):
    csv = "pid,yr,mo,val,stock\n1,2020,1,1.0,1.0\n"
    _, msgs = _build(tmp_path, _series({"aggregates": "keep_as_year"}), {"r.csv": csv})
    hits = [m for m in msgs if "unknown key 'aggregates'" in m]
    assert len(hits) == 1 and "timeseries" in hits[0], msgs


def test_known_timeseries_keys_do_not_warn(tmp_path):
    csv = "pid,yr,mo,val,stock\n1,2020,1,1.0,1.0\n"
    _, msgs = _build(tmp_path, _series({"units": {"out": "MW"}}), {"r.csv": csv})
    assert not [m for m in msgs if "unknown key" in m], msgs


def test_a_missing_channel_column_still_loads_empty_without_a_new_warning(tmp_path):
    csv = "pid,yr,mo,val\n1,2020,1,1.0\n"
    g, msgs = _build(tmp_path, _series(), {"r.csv": csv})
    assert _scalar(g, "MATCH (p:Plant) RETURN ts_count(p.out) AS v") == 1
    assert not [m for m in msgs if "aggregate" in m or "unknown key" in m], msgs


SCRIPT = textwrap.dedent(
    """
    import json, sys, pathlib
    from kglite.blueprint import from_blueprint
    root = pathlib.Path(sys.argv[1])
    (root / "s.csv").write_text("sid,v\\ns1,1\\ns1,2\\ns1,3\\ns2,4\\n", encoding="utf-8")
    bp = {"settings": {"root": str(root)},
          "nodes": {"Sample": {"csv": "s.csv", "pk": "sid", "title": "sid", "properties": {"v": "int"}}}}
    (root / "bp.json").write_text(json.dumps(bp), encoding="utf-8")
    g = from_blueprint(root / "bp.json", save=False)
    print("COUNT", g.cypher("MATCH (n:Sample) RETURN count(n) AS c").to_list()[0]["c"])
    """
)


def test_a_repeated_pk_warns_and_still_creates_every_node(tmp_path):
    # The duplicate-id warning's counter is process-global and rate-limited: fresh interpreter.
    proc = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "COUNT 4" in proc.stdout
    assert "duplicate id(s) on type 'Sample'" in proc.stderr, proc.stderr
