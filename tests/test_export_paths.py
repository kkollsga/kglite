"""Ordinary exports remain inside owned output directories with distinct files."""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from kglite import KnowledgeGraph
from kglite.blueprint import from_blueprint


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("suffix", ["", ".csv", ".CSV", ".data"])
@pytest.mark.parametrize("with_edge", [False, True])
def test_explicit_csv_uses_two_distinct_sibling_paths(tmp_path, suffix, with_edge):
    parent = tmp_path / "parent.csv.folder"
    parent.mkdir()
    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [10, 20], "title": ["first", "second"]}), "Doc", "id", "title")
    if with_edge:
        graph.cypher("MATCH(a:Doc),(b:Doc) WHERE a.id=10 AND b.id=20 CREATE(a)-[:LINK]->(b)")
    graph.export(str(parent / f"graph{suffix}"), format="csv")
    assert {path.name for path in parent.iterdir()} == {"graph_nodes.csv", "graph_edges.csv"}
    assert _read_csv(parent / "graph_nodes.csv") == [
        {"id": "0", "type": "Doc", "title": "first"},
        {"id": "1", "type": "Doc", "title": "second"},
    ]
    assert _read_csv(parent / "graph_edges.csv") == (
        [{"source": "0", "target": "1", "type": "LINK"}] if with_edge else []
    )


def test_case_distinct_types_and_parent_folders_roundtrip(tmp_path):
    graph = KnowledgeGraph()
    types = ["Case", "case", "Parent", "Parent.csv", "Child"]
    for identity, logical in enumerate(types, 1):
        graph.add_nodes(
            pd.DataFrame({"id": [identity], "title": [logical], "note": [f"note-{logical}"]}), logical, "id", "title"
        )
    graph.set_parent_type("Child", "Parent.csv")
    graph.add_connections(
        pd.DataFrame({"source": [1], "target": [2], "note": ["upper"]}), "LINK", "Case", "source", "case", "target"
    )
    graph.add_connections(
        pd.DataFrame({"source": [2], "target": [5], "note": ["lower"]}), "link", "case", "source", "Child", "target"
    )
    node_query = "MATCH(n) RETURN n.id AS id,n.note AS note ORDER BY id"
    edge_query = "MATCH(a)-[r]->(b) RETURN a.id AS source,b.id AS target,type(r) AS type,r.note AS note ORDER BY source"
    expected_nodes = [{"id": i, "note": f"note-{logical}"} for i, logical in enumerate(types, 1)]
    expected_edges = [
        {"source": 1, "target": 2, "type": "LINK", "note": "upper"},
        {"source": 2, "target": 5, "type": "link", "note": "lower"},
    ]
    assert graph.cypher(node_query).to_list() == expected_nodes
    assert graph.cypher(edge_query).to_list() == expected_edges
    output = tmp_path / "export"
    summary = graph.export_csv(str(output))
    blueprint = json.loads((output / "blueprint.json").read_text(encoding="utf-8"))
    assert set(blueprint["nodes"]) == set(types)
    assert blueprint["nodes"]["Child"]["parent"] == "Parent.csv"
    relative_paths = [node["csv"] for node in blueprint["nodes"].values()]
    for node in blueprint["nodes"].values():
        relative_paths.extend(edge["csv"] for edge in node.get("connections", {}).get("junction_edges", {}).values())
    assert len(relative_paths) == len({name.lower() for name in relative_paths}) == 7
    for relative in relative_paths:
        path = Path(relative)
        assert not path.is_absolute() and all(part not in {".", ".."} for part in path.parts)
        assert (output / path).is_file()
        assert (output / path).resolve().is_relative_to(output.resolve())
    assert summary["files_written"] == 8
    loaded = from_blueprint(output / "blueprint.json", save=False)
    assert set(loaded.node_types) == set(types)
    assert loaded.cypher(node_query).to_list() == expected_nodes
    assert loaded.cypher(edge_query).to_list() == expected_edges
