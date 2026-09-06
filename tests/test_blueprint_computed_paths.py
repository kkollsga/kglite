"""Distinct logical compute outputs never overwrite another source or output."""

import csv
import json

import pytest

from kglite.blueprint import from_blueprint


def load(root, nodes, operations, sources):
    root.mkdir(parents=True, exist_ok=True)
    for name, content in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    spec = {"settings": {"root": str(root)}, "nodes": nodes, "compute": operations}
    path = root / "blueprint.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    graph = from_blueprint(path, save=False)
    assert all((root / name).read_text(encoding="utf-8") == content for name, content in sources.items())
    return graph


def node(csv_path):
    return {"csv": csv_path, "pk": "id", "properties": {"v": "int", "g": "string"}}


@pytest.mark.parametrize("names", [("A-B", "A_B"), ("Name", "name")])
@pytest.mark.parametrize("kind", ["derive", "filter", "aggregate", "chain", "calendar"])
def test_colliding_computed_names_keep_exact_logical_values(tmp_path, names, kind):
    left, right = names
    sources = {"a.csv": "id,v,g\n1,10,x\n2,11,x\n", "b.csv": "id,v,g\n3,20,x\n4,21,x\n"}
    nodes = {"S1": node("a.csv"), "S2": node("b.csv")}
    if kind == "derive":
        nodes = {left: node("a.csv"), right: node("b.csv")}
        ops = [{"op": kind, "from": name, "set": {"d": "v + 1"}} for name in names]
        query = "MATCH(n) RETURN labels(n) AS labels,n.id AS id,n.v AS v,n.d AS d ORDER BY id"
        expected = [
            {"labels": [label], "id": i, "v": value, "d": value + 1}
            for label, pairs in [(left, [(1, 10), (2, 11)]), (right, [(3, 20), (4, 21)])]
            for i, value in pairs
        ]
    elif kind == "filter":
        ops = [
            {"op": kind, "from": source, "into": name, "where": "v % 10 == 0"}
            for source, name in zip(["S1", "S2"], names)
        ]
        query = f"MATCH(n) WHERE n:`{left}` OR n:`{right}` RETURN labels(n) AS labels,n.id AS id,n.v AS v ORDER BY id"
        expected = [{"labels": [left], "id": 1, "v": 10}, {"labels": [right], "id": 3, "v": 20}]
    elif kind == "aggregate":
        ops = [
            {"op": kind, "from": source, "into": name, "group_by": ["g"], "agg": {"total": "sum(v)"}}
            for source, name in zip(["S1", "S2"], names)
        ]
        query = (
            f"MATCH(n) WHERE n:`{left}` OR n:`{right}` "
            "RETURN labels(n) AS labels,n.g AS g,n.total AS total ORDER BY total"
        )
        expected = [{"labels": [left], "g": "x", "total": 21}, {"labels": [right], "g": "x", "total": 41}]
    elif kind == "chain":
        ops = [
            {"op": kind, "from": source, "edge": name, "group_by": ["g"], "order_by": "v"}
            for source, name in zip(["S1", "S2"], names)
        ]
        query = "MATCH(a)-[r]->(b) RETURN a.id AS a,b.id AS b,type(r) AS kind,r.step_index AS step ORDER BY a"
        expected = [{"a": 1, "b": 2, "kind": left, "step": 0}, {"a": 3, "b": 4, "kind": right, "step": 0}]
    else:
        sources = {}
        nodes = {}
        ops = [
            {"op": kind, "type": name, "start": date, "end": date}
            for name, date in zip(names, ["2026-01-01", "2026-02-01"])
        ]
        query = "MATCH(n) RETURN labels(n) AS labels,n.month AS month,n.day AS day ORDER BY month"
        expected = [{"labels": [left], "month": 1, "day": 1}, {"labels": [right], "month": 2, "day": 1}]
    emitted = []
    for order, operations in [("forward", ops), ("reverse", ops[::-1])]:
        root = tmp_path / order
        graph = load(root, nodes, operations, sources)
        assert graph.cypher(query).to_list() == expected
        files = {}
        for path in sorted((root / "computed").glob("*.csv")):
            with path.open(newline="", encoding="utf-8") as stream:
                files[path.name] = list(csv.reader(stream))
        assert len(files) == (4 if kind == "calendar" else 2)
        assert len({name.lower() for name in files}) == len(files)
        emitted.append(files)
    assert emitted[0] == emitted[1]


def test_existing_output_and_active_source_survive_repeated_steps_and_rerun(tmp_path):
    sources = {"computed/T_derived.csv": "id,v,g\n1,10,x\n", "computed/compute_0.csv": "unrelated retained bytes\n"}
    nodes = {"T": node("computed/T_derived.csv")}
    ops = [{"op": "derive", "from": "T", "set": {"a": "v+1"}}, {"op": "derive", "from": "T", "set": {"b": "a+1"}}]
    first = load(tmp_path, nodes, ops, sources)
    expected = [{"id": 1, "v": 10, "a": 11, "b": 12}]
    query = "MATCH(n:T) RETURN n.id AS id,n.v AS v,n.a AS a,n.b AS b"
    assert first.cypher(query).to_list() == expected
    completed = {p.name: p.read_bytes() for p in (tmp_path / "computed").iterdir()}
    assert len(completed) == 3
    second = from_blueprint(tmp_path / "blueprint.json", save=False)
    assert second.cypher(query).to_list() == expected
    assert all((tmp_path / "computed" / name).read_bytes() == content for name, content in completed.items())
    assert len(list((tmp_path / "computed").iterdir())) == 4


def test_calendar_hierarchy_loads_every_exact_endpoint(tmp_path):
    graph = load(
        tmp_path,
        {},
        [
            {
                "op": "calendar",
                "type": "Date",
                "start": "2026-01-30",
                "end": "2026-02-01",
                "next_edge": "NEXT_DAY",
                "in_month_edge": "IN_MONTH",
                "in_quarter_edge": "IN_QUARTER",
            }
        ],
        {},
    )
    actual = graph.cypher(
        "MATCH(a:Date)-[r]->(b) RETURN toString(a.id) AS a,type(r) AS edge,toString(b.id) AS b ORDER BY a,edge,b"
    ).to_list()
    expected = []
    dates = ["2026-01-30", "2026-01-31", "2026-02-01"]
    for i, date in enumerate(dates):
        expected.extend(
            [{"a": date, "edge": "IN_MONTH", "b": date[:7]}, {"a": date, "edge": "IN_QUARTER", "b": "2026-Q1"}]
        )
        if i + 1 < len(dates):
            expected.append({"a": date, "edge": "NEXT_DAY", "b": dates[i + 1]})
    assert actual == sorted(expected, key=lambda row: (row["a"], row["edge"], row["b"]))


def test_multiple_calendars_union_only_their_owned_hierarchy_types(tmp_path):
    graph = load(
        tmp_path,
        {},
        [
            {
                "op": "calendar",
                "type": name,
                "start": date,
                "end": date,
                "in_month_edge": "IN_MONTH",
                "in_quarter_edge": "IN_QUARTER",
            }
            for name, date in [("DateA", "2026-01-01"), ("DateB", "2026-04-01")]
        ],
        {},
    )
    assert graph.cypher("MATCH(n:Month) RETURN n.id AS id ORDER BY id").to_list() == [
        {"id": "2026-01"},
        {"id": "2026-04"},
    ]
    assert graph.cypher("MATCH(n:Quarter) RETURN n.id AS id ORDER BY id").to_list() == [
        {"id": "2026-Q1"},
        {"id": "2026-Q2"},
    ]
    assert graph.cypher(
        "MATCH(a)-[r]->(b) RETURN labels(a)[0] AS source,type(r) AS edge,toString(b.id) AS target ORDER BY source,edge"
    ).to_list() == [
        {"source": "DateA", "edge": "IN_MONTH", "target": "2026-01"},
        {"source": "DateA", "edge": "IN_QUARTER", "target": "2026-Q1"},
        {"source": "DateB", "edge": "IN_MONTH", "target": "2026-04"},
        {"source": "DateB", "edge": "IN_QUARTER", "target": "2026-Q2"},
    ]


@pytest.mark.parametrize("hierarchy,option", [("Month", "in_month_edge"), ("Quarter", "in_quarter_edge")])
@pytest.mark.parametrize("existing", [False, True])
def test_calendar_hierarchy_identity_collision_refuses_before_output(tmp_path, hierarchy, option, existing):
    source = tmp_path / "retained.csv"
    content = "id,v\nretained,42\n"
    source.write_text(content, encoding="utf-8")
    nodes = {hierarchy: {"csv": "retained.csv", "pk": "id", "properties": {"v": "int"}}} if existing else {}
    spec = {
        "settings": {"root": str(tmp_path)},
        "nodes": nodes,
        "compute": [
            {
                "op": "calendar",
                "type": "Date" if existing else hierarchy,
                "start": "2026-01-01",
                "end": "2026-01-01",
                option: "IN_HIER",
            }
        ],
    }
    path = tmp_path / "blueprint.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="collides with"):
        from_blueprint(path, save=False)
    assert source.read_text(encoding="utf-8") == content
    assert path.read_bytes() == before
    assert list((tmp_path / "computed").glob("*.csv")) == []


@pytest.mark.parametrize("intervening", ["derive", "chain"])
def test_calendar_refuses_to_replace_an_edited_owned_hierarchy(tmp_path, intervening):
    first = {"op": "calendar", "type": "DateA", "start": "2026-01-31", "end": "2026-02-01", "in_month_edge": "IN_MONTH"}
    middle = (
        {"op": "derive", "from": "Month", "set": {"tag": '"kept"'}}
        if intervening == "derive"
        else {"op": "chain", "from": "Month", "group_by": ["month_iso"], "order_by": "month_iso", "edge": "NEXT_MONTH"}
    )
    last = {"op": "calendar", "type": "DateB", "start": "2026-04-01", "end": "2026-04-01", "in_month_edge": "IN_MONTH"}
    with pytest.raises(ValueError, match="collides with"):
        load(tmp_path, {}, [first, middle, last], {})
    assert not (tmp_path / "computed/calendar_DateB.csv").exists()
    with (tmp_path / "computed/calendar_Month.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [["month_iso"], ["2026-01"], ["2026-02"]]
    if intervening == "derive":
        with (tmp_path / "computed/Month_derived.csv").open(encoding="utf-8", newline="") as stream:
            assert list(csv.reader(stream)) == [["month_iso", "tag"], ["2026-01", "kept"], ["2026-02", "kept"]]
    else:
        with (tmp_path / "computed/chain_NEXT_MONTH.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows == [["month_iso_prev", "month_iso_next", "step_index"]]


@pytest.mark.parametrize(
    "source,error", [("id,other\n1,2026-02-01\n", "date_col"), ("id,date\n1,2026-02-01\n2,2026-02-02,extra\n", "row:")]
)
def test_bad_calendar_link_preserves_prior_hierarchy(tmp_path, source, error):
    ops = [
        {"op": "calendar", "type": "DateA", "start": "2026-01-01", "end": "2026-01-01", "in_month_edge": "IN_MONTH"},
        {
            "op": "calendar",
            "type": "DateB",
            "start": "2026-02-01",
            "end": "2026-02-01",
            "in_month_edge": "IN_MONTH",
            "links": [{"from": "S", "date_col": "date", "edge": "ON_DATE"}],
        },
    ]
    with pytest.raises(ValueError, match=error):
        load(tmp_path, {"S": {"csv": "s.csv", "pk": "id"}}, ops, {"s.csv": source})
    assert (tmp_path / "s.csv").read_text(encoding="utf-8") == source
    assert not (tmp_path / "computed/calendar_DateB.csv").exists()
    with (tmp_path / "computed/calendar_Month.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [["month_iso"], ["2026-01"]]


def test_calendar_can_link_from_same_step_generated_sources(tmp_path):
    graph = load(
        tmp_path,
        {},
        [
            {
                "op": "calendar",
                "type": "Date",
                "start": "2026-01-01",
                "end": "2026-01-02",
                "in_month_edge": "IN_MONTH",
                "links": [
                    {"from": "Date", "date_col": "iso", "edge": "SELF_DATE"},
                    {"from": "Month", "date_col": "month_iso", "edge": "MONTH_DATE"},
                ],
            }
        ],
        {},
    )
    assert graph.cypher(
        "MATCH(a:Date)-[:SELF_DATE]->(b:Date) RETURN toString(a.id) AS a,toString(b.id) AS b ORDER BY a"
    ).to_list() == [
        {"a": "2026-01-01", "b": "2026-01-01"},
        {"a": "2026-01-02", "b": "2026-01-02"},
    ]
    assert graph.cypher("MATCH(:Month)-[r:MONTH_DATE]->() RETURN count(r) AS n").to_list() == [{"n": 0}]


@pytest.mark.parametrize("pk", ["iso", "date_iso", "id"])
def test_calendar_links_keep_distinct_source_and_target_columns(tmp_path, pk):
    graph = load(
        tmp_path,
        {"T": {"csv": "t.csv", "pk": pk, "properties": {"day": "string"}}},
        [
            {
                "op": "calendar",
                "type": "Date",
                "start": "2026-01-01",
                "end": "2026-01-02",
                "links": [{"from": "T", "date_col": "day", "edge": "ON_DATE"}],
            }
        ],
        {"t.csv": f"{pk},day\nrow-a,2026-01-01\nrow-b,2026-01-02\n"},
    )
    assert graph.cypher(
        "MATCH(a:T)-[:ON_DATE]->(b:Date) RETURN a.id AS a,toString(b.id) AS b ORDER BY a"
    ).to_list() == [{"a": "row-a", "b": "2026-01-01"}, {"a": "row-b", "b": "2026-01-02"}]
    with (tmp_path / "computed/calendar_link_T_ON_DATE.csv").open(encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            [pk, "date_iso" if pk == "iso" else "iso"],
            ["row-a", "2026-01-01"],
            ["row-b", "2026-01-02"],
        ]
