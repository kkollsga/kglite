"""Disk unified identity columns preserve values and their streaming consumers."""

import datetime

import pytest

import kglite

CASES = {
    "small_int": list(range(8)),
    "signed_bounds": [-(2**63), -(2**31) - 1, -1, 0, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1],
    "int_then_null": [1, None, 2],
    "null_then_int": [None, 1, 2],
    "all_null": [None, None],
    "mixed": [1, "two", -3],
    "string": ["first", "", "third"],
    "float": [1.5, -2.5, 0.0],
    "bool": [True, False],
    "date": [datetime.date(2020, 1, 1), datetime.date(1970, 1, 1)],
}


def identity_rows(graph):
    return graph.cypher("MATCH(n:Item) RETURN n.id AS id,n.title AS title,n.rank AS rank ORDER BY rank").to_list()


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
@pytest.mark.parametrize("case", CASES)
def test_identity_values_survive_save_reload_and_streaming_subset(tmp_path, storage, case):
    path = str(tmp_path / "graph")
    graph = kglite.KnowledgeGraph(**({"storage": storage, "path": path} if storage == "disk" else {"storage": storage}))
    values = CASES[case]
    expected = [{"id": value, "title": f"row-{i}", "rank": i} for i, value in enumerate(values)]
    for row in expected:
        graph.cypher("CREATE (:Item {id:$id,title:$title,rank:$rank})", params=row)
    expected = [
        {**row, "id": row["id"].isoformat() if isinstance(row["id"], datetime.date) else row["id"]} for row in expected
    ]
    assert identity_rows(graph) == expected
    graph.save(path)
    loaded = kglite.load(path)
    assert identity_rows(loaded) == expected
    # A disk source streams borrowed IDs into a writer selected by id_type_str.
    subset_path = str(tmp_path / "subset")
    loaded.select("Item").save_subset(subset_path)
    subset = kglite.load(subset_path)
    assert identity_rows(subset) == expected
    subset.save(subset_path)
    assert identity_rows(kglite.load(subset_path)) == expected


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
def test_non_string_and_null_titles_keep_the_general_roundtrip(tmp_path, storage):
    path = str(tmp_path / "graph")
    graph = kglite.KnowledgeGraph(**({"storage": storage, "path": path} if storage == "disk" else {"storage": storage}))
    expected = [{"id": str(i), "title": value, "rank": i} for i, value in enumerate([12, None, 2.5, True, "end"])]
    for row in expected:
        graph.cypher("CREATE (:Item {id:$id,title:$title,rank:$rank})", params=row)
    graph.save(path)
    assert identity_rows(kglite.load(path)) == expected
