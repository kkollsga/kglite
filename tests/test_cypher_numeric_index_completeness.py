"""Predicate-equivalent index candidates must match absolute scan oracles."""

import pandas as pd
import pytest

import kglite


def numeric_graph(graph=None):
    if graph is None:
        graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(:N{id:-1,v:'sentinel',tag:'same'})")
    graph.cypher(
        "UNWIND $rows AS r CREATE(:N{id:r.id,v:r.v,tag:'same'})",
        params={
            "rows": [
                {"id": 0, "v": 1},
                {"id": 1, "v": 1.0},
                {"id": 2, "v": 2},
                {"id": 3, "v": 1.5},
                {"id": 4, "v": 2**53 + 1},
            ]
        },
    )
    graph.cypher("MATCH(n:N{id:-1}) DELETE n")
    rows = graph.cypher("MATCH(n:N) RETURN n.id AS id,n.v AS v ORDER BY id").to_list()
    assert [(row["id"], type(row["v"])) for row in rows] == [(0, int), (1, float), (2, int), (3, float), (4, int)]
    return graph


def ids(graph, predicate, params, disabled=False):
    return graph.cypher(
        f"MATCH(n:N) WHERE {predicate} RETURN n.id AS id ORDER BY id", params=params, disable_optimizer=disabled
    ).to_list()


@pytest.mark.parametrize("kind", ["equality", "range"])
@pytest.mark.parametrize("disabled", [False, True])
def test_indexed_numeric_queries_match_absolute_unindexed_results(kind, disabled):
    graph = numeric_graph()
    cases = [
        ("n.v=$v", {"v": 1}, [0, 1]),
        ("n.v=$v", {"v": 1.0}, [0, 1]),
        ("n.v IN $vs", {"vs": [1.0, 2]}, [0, 1, 2]),
        ("n.v IN $vs", {"vs": [1, 2**53 + 1]}, [0, 1, 4]),
        ("n.v IN $vs", {"vs": [1, [None]]}, [0, 1]),
        ("n.v >= $a AND n.v <= $b", {"a": 1.0, "b": 1.0}, [0, 1]),
        ("n.v > $a AND n.v < $b", {"a": 1, "b": 2.0}, [3]),
        ("n.v < $v", {"v": 1.5}, [0, 1]),
        ("n.v > $v", {"v": 1.5}, [2, 4]),
        ("n.v > $a AND n.v < $b", {"a": 2, "b": 1}, []),
        ("n.v=$v", {"v": 2**53 + 1}, [4]),
        ("n.v >= $v", {"v": 2**53}, [4]),
    ]
    for indexed in [False, True]:
        if indexed:
            if kind == "equality":
                graph.create_index("N", "v")
            else:
                graph.create_range_index("N", "v")
        for predicate, params, expected in cases:
            assert ids(graph, predicate, params, disabled) == [{"id": i} for i in expected], (
                indexed,
                predicate,
                params,
            )


def test_numeric_index_stays_complete_after_mutations_and_slot_reuse():
    graph = numeric_graph()
    graph.create_index("N", "v")
    graph.create_range_index("N", "v")
    for mutation, expected in [
        ("MATCH(n:N{id:0}) SET n.v=3", [1]),
        ("MATCH(n:N{id:1}) REMOVE n.v", []),
        ("MATCH(n:N{id:2}) SET n.v=1.0", [2]),
        ("MATCH(n:N{id:2}) DELETE n", []),
        ("CREATE(:N{id:5,v:1})", [5]),
    ]:
        graph.cypher(mutation)
        assert ids(graph, "n.v=$v", {"v": 1.0}) == [{"id": i} for i in expected]
        assert ids(graph, "n.v >= $v AND n.v <= $v", {"v": 1.0}) == [{"id": i} for i in expected]


def test_composite_structural_hit_does_not_hide_other_numeric_variants():
    graph = numeric_graph()
    graph.cypher("CREATE INDEX FOR(n:N) ON(n.tag,n.v)")
    assert ids(graph, "n.tag='same' AND n.v=$v", {"v": 1.0}) == [{"id": 0}, {"id": 1}]


@pytest.mark.parametrize("disabled", [False, True])
def test_composite_numeric_and_string_families_preserve_exact_matches(disabled):
    graph = numeric_graph()
    graph.cypher('CREATE(:N{id:5,v:1,tag:\'["same"]\'}),(:N{id:6,v:1,tag:\'["["same"]"]\'})')
    graph.cypher("CREATE INDEX FOR(n:N) ON(n.v,n.tag)")
    for tag, expected in [("same", [0, 1, 5]), ('["same"]', [0, 1, 5, 6])]:
        assert ids(graph, "n.tag=$tag AND n.v=$v", {"tag": tag, "v": 1.0}, disabled) == [{"id": i} for i in expected]


def test_composite_merge_reuses_equivalent_numeric_and_string_tuple():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(:N{id:7,v:1,tag:'[\"same\"]'})")
    graph.cypher("CREATE INDEX FOR(n:N) ON(n.tag,n.v)")
    assert graph.cypher("MERGE(n:N{v:1.0,tag:'same'}) RETURN n.id AS id").to_list() == [{"id": 7}]
    assert graph.cypher("MATCH(n:N) RETURN count(*) AS n").to_list() == [{"n": 1}]


@pytest.mark.parametrize("index", ["none", "single", "composite"])
@pytest.mark.parametrize("kind", ["numeric", "wrapped", "name", "title_alias"])
def test_merge_non_id_property_predicates_match_existing_nodes(index, kind):
    graph = kglite.KnowledgeGraph()
    if kind == "title_alias":
        graph.add_nodes(pd.DataFrame({"rid": [7], "caption": ["Shown"], "tag": ["same"]}), "N", "rid", "caption")
        field, probe = "caption", "Shown"
    elif kind == "name":
        graph.add_nodes(
            pd.DataFrame({"id": [7], "title": ["Other"], "name": ["Ann"], "tag": ["same"]}), "N", "id", "title"
        )
        assert graph.cypher("MATCH(n:N) RETURN n.name AS name,n.title AS title").to_list() == [
            {"name": "Ann", "title": "Other"}
        ]
        field, probe = "name", "Ann"
    else:
        graph.cypher(
            "CREATE(:N{id:7,v:$value,tag:'same',name:'Ann',title:'Other'})",
            params={"value": '["same"]' if kind == "wrapped" else 1},
        )
        field, probe = "v", "same" if kind == "wrapped" else 1.0
    if index == "single":
        graph.create_index("N", field)
    elif index == "composite":
        graph.cypher(f"CREATE INDEX FOR(n:N) ON(n.{field},n.tag)")
    extra = "" if index == "single" else ",tag:'same'"
    assert graph.cypher(
        f"MERGE(n:N{{{field}:$value{extra}}}) RETURN n.id AS id", params={"value": probe}
    ).to_list() == [{"id": 7}]
    assert graph.cypher("MATCH(n:N) RETURN count(*) AS n").to_list() == [{"n": 1}]


@pytest.mark.parametrize("indexed", [False, True])
def test_merge_stored_name_does_not_match_an_unrelated_title(indexed):
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [7], "title": ["Other"], "name": ["Ann"]}), "N", "id", "title")
    assert graph.cypher("MATCH(n:N) RETURN n.name AS name,n.title AS title").to_list() == [
        {"name": "Ann", "title": "Other"}
    ]
    if indexed:
        graph.create_index("N", "name")
    assert graph.cypher("MERGE(n:N{name:'Other'}) RETURN n.id AS id").to_list() == [{"id": 8}]
    assert graph.cypher("MATCH(n:N) RETURN count(*) AS n").to_list() == [{"n": 2}]


def test_merge_property_fallbacks_use_the_candidates_primary_type():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(:N{id:7,title:'Shown',v:1})")
    graph.cypher("MATCH(n:N) SET n:Secondary")
    for query in [
        "MERGE(n:N{name:'Shown',v:1.0}) RETURN n.id AS id",
        "MERGE(n:Secondary{type:'N',v:1.0}) RETURN n.id AS id",
    ]:
        assert graph.cypher(query).to_list() == [{"id": 7}]
    assert graph.cypher("MATCH(n) RETURN count(*) AS n").to_list() == [{"n": 1}]


def test_composite_soft_alias_preserves_missing_stored_property():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(:N{id:1,name:'Ann',v:1}),(:N{id:2,title:'Ann',v:1})")
    query = "MATCH(n:N) WHERE n.name='Ann' AND n.v=1.0 RETURN n.id AS id ORDER BY id"
    assert graph.cypher(query).to_list() == [{"id": 1}, {"id": 2}]
    graph.cypher("CREATE INDEX FOR(n:N) ON(n.name,n.v)")
    assert graph.cypher(query).to_list() == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("storage", ["memory", "mapped"])
@pytest.mark.parametrize("deferred", [False, True])
def test_numeric_index_roundtrip_controls(tmp_path, storage, deferred):
    graph = numeric_graph()
    graph.create_index("N", "v")
    graph.create_range_index("N", "v")
    path = tmp_path / "numeric.kgl"
    graph.save(str(path))
    loaded = kglite.load(str(path), storage=storage, defer_index_rebuild=deferred)
    assert ids(loaded, "n.v=$v", {"v": 1.0}) == [{"id": 0}, {"id": 1}]
    assert ids(loaded, "n.v >= $v AND n.v <= $v", {"v": 1.0}) == [{"id": 0}, {"id": 1}]


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
def test_string_wrapper_equivalence_in_property_indexes(tmp_path, storage):
    graph = kglite.KnowledgeGraph(storage=storage, path=str(tmp_path / "strings") if storage == "disk" else None)
    graph.cypher("CREATE(:N{id:0,v:'Oslo'}),(:N{id:1,v:'[\"Oslo\"]'}),(:N{id:2,v:'Bergen'})")
    graph.create_index("N", "v")
    for value in ["Oslo", '["Oslo"]']:
        assert ids(graph, "n.v=$v", {"v": value}) == [{"id": 0}, {"id": 1}]


def test_primary_id_alias_and_secondary_label_controls():
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"rid": [1, 2], "title": ["one", "two"], "v": [1, 2]}), "N", "rid", "title")
    graph.create_index("N", "v")
    for name in ["id", "rid"]:
        assert ids(graph, f"n.{name}=$v", {"v": 1.0}) == [{"id": 1}]
    graph.cypher("MATCH(n:N{id:1}) SET n:Secondary")
    assert graph.cypher("MATCH(n:Secondary) WHERE n.v=1.0 RETURN n.id AS id").to_list() == [{"id": 1}]


def test_disk_numeric_range_controls(tmp_path):
    graph = numeric_graph(kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "numeric")))
    graph.create_range_index("N", "v")
    assert ids(graph, "n.v=$v", {"v": 1.0}) == [{"id": 0}, {"id": 1}]
    assert ids(graph, "n.v >= $v AND n.v <= $v", {"v": 1.0}) == [{"id": 0}, {"id": 1}]


def test_fluent_range_requires_every_selected_type():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(:A{id:0,v:1}),(:B{id:1,v:1}),(:B{id:2,v:3})")
    graph.create_range_index("A", "v")
    assert sorted(row["id"] for row in graph.where({"v": {"<": 2}}).collect()) == [0, 1]
    graph.create_range_index("B", "v")
    assert sorted(row["id"] for row in graph.where({"v": {"<": 2}}).collect()) == [0, 1]


def test_soft_alias_range_preserves_title_fallback_candidates():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(:T{id:1,name:'Ann'}),(:T{id:2,title:'Ann'})")
    query = "MATCH(n:T) WHERE n.name >= 'A' RETURN n.id AS id ORDER BY id"
    expected = [{"id": 1}, {"id": 2}]
    assert graph.cypher(query).to_list() == expected
    graph.create_range_index("T", "name")
    assert graph.cypher(query).to_list() == expected
