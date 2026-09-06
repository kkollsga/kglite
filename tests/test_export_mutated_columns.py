"""Export and persistence consume effective values after bulk column mutations."""

import json

import pandas as pd
import pytest

import kglite
from kglite.blueprint import from_blueprint


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_bulk_typed_mutation_exports_and_saves_complete_values(tmp_path, mode):
    graph = kglite.KnowledgeGraph(storage=mode, path=str(tmp_path / "disk") if mode == "disk" else None)
    graph.add_nodes(
        pd.DataFrame(
            {"id": list(range(8)), "title": [f"row-{i}" for i in range(8)], "a": list(range(8)), "b": [0.5] * 8}
        ),
        "Row",
        "id",
        "title",
    )
    if mode == "disk":
        graph.save(str(tmp_path / "disk"))
    # All rows cross the original integer column's representation together.
    graph.cypher("MATCH(n:Row) SET n.a='updated-'+toString(n.id),n.b=n.b+0.25").to_list()
    query = "MATCH(n:Row) RETURN n.id AS id,n.title AS title,n.a AS a,n.b AS b ORDER BY id"
    expected = [{"id": i, "title": f"row-{i}", "a": f"updated-{i}", "b": 0.75} for i in range(8)]
    assert graph.cypher(query).to_list() == expected
    exported = json.loads(graph.export_string("json"))
    assert [
        {key: node[key] for key in expected[0]} for node in sorted(exported["nodes"], key=lambda n: n["id"])
    ] == expected
    assert exported["links"] == []
    folder = tmp_path / "csv-export"
    graph.export_csv(str(folder))
    reloaded_csv = from_blueprint(folder / "blueprint.json", save=False)
    assert reloaded_csv.cypher(query).to_list() == expected
    saved = tmp_path / "saved.kgl"
    graph.save(str(saved))
    reloaded = kglite.load(str(saved))
    assert reloaded.cypher(query).to_list() == expected
