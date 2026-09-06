"""Declared integer properties preserve exact decimal input or become NULL."""

import csv
import json

from kglite.blueprint import from_blueprint


def test_declared_integer_cells_never_round_or_saturate(tmp_path):
    cases = [
        ("9223372036854775807", 2**63 - 1),
        ("9223372036854775807.0", 2**63 - 1),
        ("-9223372036854775808.0", -(2**63)),
        ("9.223372036854775807e18", 2**63 - 1),
        ("-9.223372036854775808e18", -(2**63)),
        ("9007199254740993.0", 2**53 + 1),
        (" +001.000e+3 ", 1000),
        ("9223372036854775808", None),
        ("-9223372036854775809", None),
        ("9223372036854775808.0", None),
        ("-9223372036854775809.0", None),
        ("9007199254740993.1", None),
        ("1.5", None),
        ("inf", None),
        ("NaN", None),
        ("", None),
    ]
    with (tmp_path / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "value"])
        writer.writerows((i, text) for i, (text, _) in enumerate(cases))
    blueprint = {
        "settings": {"root": str(tmp_path)},
        "nodes": {"Cell": {"csv": "cells.csv", "pk": "id", "properties": {"value": "int"}}},
    }
    path = tmp_path / "blueprint.json"
    path.write_text(json.dumps(blueprint), encoding="utf-8")
    graph = from_blueprint(path, save=False)
    rows = graph.cypher("MATCH(n:Cell) RETURN n.id AS id,n.value AS value ORDER BY id").to_list()
    assert rows == [{"id": i, "value": expected} for i, (_, expected) in enumerate(cases)]
    assert all(row["value"] is None or type(row["value"]) is int for row in rows)
