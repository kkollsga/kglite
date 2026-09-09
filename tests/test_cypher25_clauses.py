"""Absolute contracts for the supported Cypher 25 clause spellings."""

from __future__ import annotations

import pytest

import kglite


def _graph(storage: str = "memory", path=None):
    kwargs = {} if storage == "memory" else {"storage": storage}
    if path is not None:
        kwargs["path"] = str(path)
    graph = kglite.KnowledgeGraph(**kwargs)
    graph.cypher("CREATE (:Person {id: 1, age: 30}), (:Person {id: 2, age: 40})").to_list()
    graph.cypher("MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:KNOWS]->(b)").to_list()
    return graph


@pytest.mark.parametrize("disable_optimizer", (False, True))
def test_filter_is_post_optional_match_and_removes_null_extended_rows(disable_optimizer):
    graph = _graph()
    rows = graph.cypher(
        "UNWIND [30, 99] AS wanted "
        "OPTIONAL MATCH (p:Person) WHERE p.age = wanted "
        "FILTER p.age = wanted "
        "RETURN wanted AS age ORDER BY age",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == [{"age": 30}]


@pytest.mark.parametrize("disable_optimizer", (False, True))
def test_leading_filter_consumes_only_the_implicit_initial_row(disable_optimizer):
    graph = kglite.KnowledgeGraph()
    assert graph.cypher("FILTER true RETURN 1 AS x", disable_optimizer=disable_optimizer).to_list() == [{"x": 1}]
    for predicate in ("false", "null"):
        assert (
            graph.cypher(
                f"FILTER {predicate} RETURN 1 AS x",
                disable_optimizer=disable_optimizer,
            ).to_list()
            == []
        )
    assert (
        graph.cypher(
            "UNWIND [] AS x FILTER true RETURN x",
            disable_optimizer=disable_optimizer,
        ).to_list()
        == []
    )


@pytest.mark.parametrize("mode", ("memory", "mapped", "disk"))
def test_leading_filter_controls_writes_in_every_storage_mode(tmp_path, mode):
    path = tmp_path / "filter-write-disk" if mode == "disk" else None
    graph = kglite.KnowledgeGraph(
        **({} if mode == "memory" else {"storage": mode}),
        **({} if path is None else {"path": str(path)}),
    )
    for predicate, expected in (("true", 1), ("false", 0)):
        result = graph.cypher(f"FILTER {predicate} CREATE (:Item {{accepted: {predicate}}}) FINISH")
        assert result.to_list() == []
        assert graph.last_mutation_stats["nodes_created"] == expected
    assert graph.cypher("MATCH (n:Item) RETURN count(n) AS n").to_list() == [{"n": 1}]


def test_filter_diagnostics_name_the_standalone_clause():
    graph = _graph()
    warning = graph.cypher("MATCH (p:Person) FILTER p.agee > 1 RETURN p").diagnostics["warnings"]
    assert warning[0].startswith("FILTER references property 'agee'"), warning

    graph.cypher("CREATE CONSTRAINT person_age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER").to_list()
    mismatch = graph.cypher("MATCH (p:Person) FILTER p.age > 'forty' RETURN p").diagnostics["warnings"]
    assert mismatch[0].startswith("FILTER compares Person.age"), mismatch

    graph.lock_schema()
    with pytest.raises(kglite.SchemaError, match="referenced in FILTER"):
        graph.cypher("MATCH (p:Person) FILTER p.agee > 1 RETURN p").to_list()
    with pytest.raises(kglite.SchemaError, match="FILTER compares Person.age"):
        graph.cypher("MATCH (p:Person) FILTER p.age > 'forty' RETURN p").to_list()


@pytest.mark.parametrize(
    ("skip", "offset"),
    (
        (
            "MATCH (n:Person) ORDER BY n.id SKIP 1 RETURN n.id AS id",
            "MATCH (n:Person) ORDER BY n.id OFFSET 1 RETURN n.id AS id",
        ),
        (
            "UNWIND [1,2,3,4] AS x RETURN x ORDER BY x SKIP 1 LIMIT 2",
            "UNWIND [1,2,3,4] AS x RETURN x ORDER BY x OFFSET 1 LIMIT 2",
        ),
    ),
)
def test_offset_is_exact_skip_synonym(skip, offset):
    graph = _graph()
    assert graph.cypher(offset).to_list() == graph.cypher(skip).to_list()


def test_finish_read_has_no_rows_or_columns_and_keeps_profile_and_warnings():
    graph = _graph()
    result = graph.cypher("PROFILE MATCH (n:Persn) FINISH")
    assert result.to_list() == []
    assert result.columns == []
    assert result.profile[-1]["clause"] == "Finish"
    assert result.profile[-1]["rows_in"] == result.profile[-1]["rows_out"] == 0
    assert any("unknown node label 'Persn'" in warning for warning in result.diagnostics["warnings"])


@pytest.mark.parametrize("mode", ("memory", "mapped", "disk"))
def test_finish_preserves_write_and_stats_in_every_storage_mode(tmp_path, mode):
    path = tmp_path / "finish-disk" if mode == "disk" else None
    graph = _graph(mode, path)
    result = graph.cypher("PROFILE CREATE (:Item {id: 1}) FINISH")
    assert result.to_list() == []
    assert result.columns == []
    assert result.profile[-1]["clause"] == "Finish"
    assert graph.last_mutation_stats["nodes_created"] == 1
    assert graph.cypher("MATCH (n:Item) RETURN n.id AS id").to_list() == [{"id": 1}]


@pytest.mark.parametrize("mode", ("memory", "mapped", "disk"))
def test_nodetach_delete_is_plain_delete_in_every_storage_mode(tmp_path, mode):
    path = tmp_path / "nodetach-disk" if mode == "disk" else None
    graph = _graph(mode, path)
    with pytest.raises(kglite.CypherExecutionError, match="still has relationships"):
        graph.cypher("MATCH (n:Person {id: 1}) NODETACH DELETE n FINISH").to_list()
    assert graph.cypher("MATCH (n:Person) RETURN count(n) AS n").to_list() == [{"n": 2}]
    assert graph.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS n").to_list() == [{"n": 1}]

    graph.cypher("MATCH ()-[r:KNOWS]->() NODETACH DELETE r FINISH").to_list()
    graph.cypher("MATCH (n:Person {id: 1}) NODETACH DELETE n FINISH").to_list()
    assert graph.cypher("MATCH (n:Person) RETURN n.id AS id").to_list() == [{"id": 2}]


@pytest.mark.parametrize(
    "query",
    (
        "FINISH",
        "MATCH (n) FINISH RETURN n",
        "MATCH (n) FINISH LIMIT 1",
        "MATCH (n) RETURN n FINISH",
        "CREATE (:N) FINISH FORMAT CSV",
        "CALL db.graph_stats() FINISH",
        "CALL { MATCH (n) FINISH } RETURN 1 AS x",
    ),
)
def test_finish_rejects_invalid_forms_before_execution(query):
    graph = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherSyntaxError):
        graph.cypher(query).to_list()
    assert graph.cypher("MATCH (n) RETURN count(n) AS n").to_list() == [{"n": 0}]


def test_finish_accepts_yielding_procedure_and_soft_identifiers():
    graph = _graph()
    assert graph.cypher("CALL db.graph_stats() YIELD node_count FINISH").to_list() == []
    graph.cypher("CREATE (:filter {offset: 1, finish: 2, nodetach: 3})").to_list()
    assert graph.cypher(
        "MATCH (filter:filter {offset: 1}) "
        "RETURN filter.offset AS finish, filter.finish AS offset, filter.nodetach AS nodetach"
    ).to_list() == [{"finish": 1, "offset": 2, "nodetach": 3}]
