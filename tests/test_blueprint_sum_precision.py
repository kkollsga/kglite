"""Blueprint SUM must preserve integer input through CSV and loaded nodes."""

import csv
import json

import pytest

from kglite.blueprint import from_blueprint


@pytest.mark.parametrize(
    "values,expected", [([2**53 + 1], 2**53 + 1), ([2**53 + 1, -(2**53)], 1), ([2**63 - 1, 1, -(2**63 - 1)], 1)]
)
def test_blueprint_integer_sum_roundtrip(tmp_path, values, expected):
    with (tmp_path / "t.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "group", "value"])
        writer.writerows((i, "A", v) for i, v in enumerate(values))
    blueprint = {
        "settings": {"root": str(tmp_path)},
        "nodes": {"T": {"csv": "t.csv", "pk": "id", "properties": {"group": "string", "value": "int"}}},
        "compute": [
            {"op": "aggregate", "from": "T", "group_by": ["group"], "into": "Summary", "agg": {"total": "sum(value)"}}
        ],
    }
    path = tmp_path / "bp.json"
    path.write_text(json.dumps(blueprint))
    graph = from_blueprint(path, save=False)
    actual = graph.cypher("MATCH(n:Summary) RETURN n.total AS total").scalar()
    assert type(actual) is int and actual == expected
    with (tmp_path / "computed/aggregate_Summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["total"] == str(expected)
