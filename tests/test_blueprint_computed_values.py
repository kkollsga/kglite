"""Computed identities and property declarations survive CSV and graph loading."""

import csv
import json

import pytest

from kglite.blueprint import from_blueprint


def _load(root, headers, rows, properties, compute, extra_nodes=None):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "t.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    nodes = {"T": {"csv": "t.csv", "pk": "id", "properties": properties}}
    nodes.update(extra_nodes or {})
    blueprint = {"settings": {"root": str(root)}, "nodes": nodes, "compute": compute}
    path = root / "blueprint.json"
    path.write_text(json.dumps(blueprint), encoding="utf-8")
    return from_blueprint(path, save=False)


def _csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _group_id(parts):
    return "group:" + json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def test_aggregate_ids_are_independent_of_rows_and_other_groups(tmp_path):
    rows = [[1, "a_b", "c", 10], [2, "a", "b_c", 20], [3, "001", "x", 30], [4, "1", "x", 40]]
    operation = {"op": "aggregate", "from": "T", "group_by": ["a", "b"], "into": "Summary", "agg": {"total": "sum(v)"}}
    expected = {_group_id([a, b]): value for _, a, b, value in rows}
    for label, input_rows in [("forward", rows), ("reverse", rows[::-1]), ("extra", rows + [[5, "雪", "z", 50]])]:
        root = tmp_path / label
        graph = _load(root, ["id", "a", "b", "v"], input_rows, {"a": "string", "b": "string", "v": "int"}, [operation])
        loaded = graph.cypher("MATCH(n:Summary) RETURN n.id AS id,n.total AS total").to_list()
        actual = {row["id"]: row["total"] for row in loaded}
        assert len(actual) == len(input_rows) == len(loaded)
        assert {key: actual[key] for key in expected} == expected
        for identity, total in expected.items():
            assert (
                graph.cypher(
                    "MATCH(n:Summary) WHERE n.id=$id RETURN n.total AS total", params={"id": identity}
                ).scalar()
                == total
            )
        emitted = _csv_rows(root / "computed/aggregate_Summary.csv")
        assert {row["summary_id"]: int(row["total"]) for row in emitted} == actual
        assert {(row["a"], row["b"]) for row in emitted} == {(row[1], row[2]) for row in input_rows}


def test_single_component_ids_do_not_coerce_numeric_strings(tmp_path):
    rows = [[1, "001"], [2, "1"], [3, ""]]
    operation = {"op": "aggregate", "from": "T", "group_by": ["g"], "into": "Summary", "agg": {"n": "count(*)"}}
    graph = _load(tmp_path, ["id", "g"], rows, {"g": "string"}, [operation])
    actual = graph.cypher("MATCH(n:Summary) RETURN n.id AS id,n.n AS count ORDER BY id").to_list()
    expected = sorted(({"id": _group_id([group]), "count": 1} for _, group in rows), key=lambda row: row["id"])
    assert actual == expected
    assert all(type(row["id"]) is str for row in actual)


def test_chain_and_aggregate_fk_edges_keep_raw_groups(tmp_path):
    rows = [[1, "a_b", "c", 1], [2, "a", "b_c", 1], [3, "a_b", "c", 2], [4, "a", "b_c", 2]]
    with (tmp_path / "owners.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id"])
        writer.writerows([["a_b"], ["a"]])
    operations = [
        {"op": "chain", "from": "T", "group_by": ["a", "b"], "order_by": "v", "edge": "NEXT"},
        {
            "op": "aggregate",
            "from": "T",
            "group_by": ["a", "b"],
            "into": "Summary",
            "agg": {"n": "count(*)"},
            "edges": [{"edge": "OWNER", "to": "Owner", "fk": "a"}],
        },
    ]
    graph = _load(
        tmp_path,
        ["id", "a", "b", "v"],
        rows,
        {"a": "string", "b": "string", "v": "int"},
        operations,
        {"Owner": {"csv": "owners.csv", "pk": "id"}},
    )
    assert graph.cypher("MATCH(a:T)-[:NEXT]->(b:T) RETURN a.id AS source,b.id AS target ORDER BY source").to_list() == [
        {"source": 1, "target": 3},
        {"source": 2, "target": 4},
    ]
    edges = graph.cypher("MATCH(s:Summary)-[:OWNER]->(o:Owner) RETURN s.id AS source,o.id AS target").to_list()
    assert {(row["source"], row["target"]) for row in edges} == {
        (_group_id(["a_b", "c"]), "a_b"),
        (_group_id(["a", "b_c"]), "a"),
    }


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "expression,expected,kind",
    [
        ("if(v == 1, 1, 1.5)", [1.0, 1.5], float),
        ("if(v == 1, null, 2)", [None, 2], int),
        ("if(v == 1, null, true)", [None, True], bool),
        ("if(v == 1, '001', '1')", ["001", "1"], str),
        ("if(v == 1, true, 2)", ["true", "2"], str),
        ("if(v == 1, '1', 2)", ["1", "2"], str),
        ("null", [None, None], None),
        ("v", [1, 2], int),
        ("v / 2.0", [0.5, 1.0], float),
        ("if(v == 1, 9007199254740993, 0.5)", [float(2**53 + 1), 0.5], float),
        ("[v]", ["[1]", "[2]"], str),
    ],
)
def test_derived_type_reconciles_every_expression_value(tmp_path, reverse, expression, expected, kind):
    rows = [[1, 1], [2, 2]]
    graph = _load(
        tmp_path,
        ["id", "v"],
        rows[::-1] if reverse else rows,
        {"v": "int"},
        [{"op": "derive", "from": "T", "set": {"result": expression}}],
    )
    values = [row["result"] for row in graph.cypher("MATCH(n:T) RETURN n.result AS result ORDER BY n.id").to_list()]
    assert values == expected
    assert all(value is None or type(value) is kind for value in values)
    emitted = _csv_rows(tmp_path / "computed/T_derived.csv")
    assert len(emitted) == 2
    if expression == "if(v == 1, 1, 1.5)":
        assert {row["id"]: row["result"] for row in emitted} == {"1": "1", "2": "1.5"}


@pytest.mark.parametrize("reverse_groups", [False, True])
def test_aggregate_column_widens_before_loading(tmp_path, reverse_groups):
    first, second = ("B", "A") if reverse_groups else ("A", "B")
    rows = [[1, first, 1], [2, second, 1.5]]
    operation = {"op": "aggregate", "from": "T", "group_by": ["g"], "into": "Summary", "agg": {"total": "sum(v)"}}
    graph = _load(tmp_path, ["id", "g", "v"], rows, {"g": "string", "v": "float"}, [operation])
    actual = graph.cypher("MATCH(n:Summary) RETURN n.g AS g,n.total AS total").to_list()
    assert {row["g"]: row["total"] for row in actual} == {first: 1.0, second: 1.5}
    assert all(type(row["total"]) is float for row in actual)
    emitted = _csv_rows(tmp_path / "computed/aggregate_Summary.csv")
    assert {row["g"]: row["total"] for row in emitted} == {first: "1", second: "1.5"}


def test_finalized_derived_type_is_used_by_next_compute_step(tmp_path):
    operations = [
        {"op": "derive", "from": "T", "set": {"result": "if(v == 1, 1, 1.5)"}},
        {"op": "aggregate", "from": "T", "group_by": ["g"], "into": "Summary", "agg": {"total": "sum(result)"}},
    ]
    graph = _load(tmp_path, ["id", "g", "v"], [[1, "A", 1], [2, "A", 2]], {"g": "string", "v": "int"}, operations)
    assert graph.cypher("MATCH(n:Summary) RETURN n.total AS total").scalar() == 2.5


@pytest.mark.parametrize(
    "expression,expected,kind",
    [
        ("if(v == 1, null, 2)", [None, 2], int),
        ("if(v == 1, null, true)", [None, True], bool),
        ("if(v == 1, true, 2)", ["true", "2"], str),
        ("if(v == 1, '001', '1')", ["001", "1"], str),
        ("null", [None, None], None),
    ],
)
def test_aggregate_expression_types_remain_null_neutral(tmp_path, expression, expected, kind):
    operation = {"op": "aggregate", "from": "T", "group_by": ["g"], "into": "Summary", "agg": {"result": expression}}
    graph = _load(tmp_path, ["id", "g", "v"], [[1, "A", 1], [2, "B", 2]], {"g": "string", "v": "int"}, [operation])
    rows = graph.cypher("MATCH(n:Summary) RETURN n.result AS result ORDER BY n.g").to_list()
    assert [row["result"] for row in rows] == expected
    assert all(row["result"] is None or type(row["result"]) is kind for row in rows)


def test_successive_derive_preserves_all_records_beyond_reader_buffer(tmp_path):
    count = 5000
    rows = [[i, i, f"ordinary-record-{i:05d}"] for i in range(count)]
    graph = _load(
        tmp_path,
        ["id", "v", "text"],
        rows,
        {"v": "int", "text": "string"},
        [
            {"op": "derive", "from": "T", "set": {"first": "v + 1"}},
            {"op": "derive", "from": "T", "set": {"second": "first + 1"}},
        ],
    )
    actual = graph.cypher(
        "MATCH(n:T) RETURN n.id AS id,n.v AS v,n.text AS text,n.first AS first,n.second AS second ORDER BY id"
    ).to_list()
    expected = [{"id": i, "v": i, "text": text, "first": i + 1, "second": i + 2} for i, _, text in rows]
    assert actual == expected
    source = tmp_path / "t.csv"
    assert source.stat().st_size > 100000
    assert _csv_rows(source) == [{"id": str(i), "v": str(i), "text": text} for i, _, text in rows]
    emitted = _csv_rows(tmp_path / "computed/T_derived.csv")
    assert list(emitted[0]) == ["id", "v", "text", "first", "second"]
    assert emitted == [{key: str(value) for key, value in row.items()} for row in expected]
    assert {path.name for path in (tmp_path / "computed").iterdir()} == {"T_derived.csv"}


def test_successive_filter_preserves_all_records_beyond_reader_buffer(tmp_path):
    rows = [[i, i, f"ordinary-record-{i:05d}"] for i in range(5000)]
    graph = _load(
        tmp_path,
        ["id", "v", "text"],
        rows,
        {"v": "int", "text": "string"},
        [
            {"op": "filter", "from": "T", "where": "v >= 0"},
            {"op": "filter", "from": "T", "where": "v < 5000"},
        ],
    )
    expected = [{"id": i, "v": v, "text": text} for i, v, text in rows]
    assert graph.cypher("MATCH(n:T) RETURN n.id AS id,n.v AS v,n.text AS text ORDER BY id").to_list() == expected
    assert (tmp_path / "t.csv").stat().st_size > 100000
    original = [{key: str(value) for key, value in row.items()} for row in expected]
    assert _csv_rows(tmp_path / "computed/T_filtered.csv") == original
    assert _csv_rows(tmp_path / "t.csv") == original
    assert {path.name for path in (tmp_path / "computed").iterdir()} == {"T_filtered.csv"}
