from pathlib import Path

import pytest

import kglite


def test_parse_json_keeps_out_of_range_float_null():
    rows = kglite.KnowledgeGraph().cypher("RETURN parse_json($raw) AS value", params={"raw": "1e400"}).to_list()
    assert rows == [{"value": None}]


def test_blueprint_keeps_out_of_range_number_refusal(tmp_path: Path):
    path = tmp_path / "blueprint.json"
    path.write_text('{"settings":{"unknown_number":1e400},"nodes":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid blueprint JSON: number out of range"):
        kglite.from_blueprint(str(path), save=False)


def test_arbitrary_precision_loads_and_resaves_pre_feature_portable_file(tmp_path: Path):
    fixture = Path(__file__).parent / "fixtures" / "json_number_contract_pre_feature.kgl"
    loaded = kglite.load(str(fixture))
    expected = [{"a": 1, "score": 1.5, "weight": 3.5, "b": 2}]
    query = "MATCH (a:T)-[r:R]->(b:T) RETURN a.id AS a, a.score AS score, r.weight AS weight, b.id AS b"
    assert loaded.cypher(query).to_list() == expected

    resaved = tmp_path / "resaved.kgl"
    loaded.save(str(resaved))
    assert kglite.load(str(resaved)).cypher(query).to_list() == expected
