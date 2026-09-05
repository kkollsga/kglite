"""Absolute NULL-winner contracts for BM25 postings coverage."""

import pytest

import kglite


@pytest.mark.parametrize("unindexed", ["missing", "numeric"])
@pytest.mark.parametrize("null_first", [False, True], ids=["null-last-slot", "null-first-slot"])
@pytest.mark.parametrize("k,nulls", [(1, ""), (2, ""), (1, " NULLS LAST")])
@pytest.mark.parametrize("freshness", ["clean", "stale-read-only"])
def test_equal_cardinality_subset_preserves_null_winner(unindexed, null_first, k, nulls, freshness):
    graph = kglite.KnowledgeGraph()
    absent = "(:Doc {id:2})" if unindexed == "missing" else "(:Doc {id:2,body:42})"
    indexed = ["(:Doc {id:0,body:'needle'})", "(:Doc {id:1,body:'other'})"]
    nodes = [absent, *indexed] if null_first else [*indexed, absent]
    graph.cypher("CREATE " + ",".join(nodes))
    assert graph.build_text_index("Doc", "body") == {"indexed": 2, "skipped": 1, "terms": 2}
    if freshness == "stale-read-only":
        graph.cypher("MATCH (d:Doc) WHERE d.id = 1 SET d.body = 'different'")
        graph.read_only(True)
    query = (
        "MATCH (d:Doc) WHERE d.id <> 1 RETURN d.id AS id, "
        f"text_bm25(d, 'body', 'needle') AS score ORDER BY score DESC{nulls} LIMIT {k}"
    )
    expected_ids = [0] if nulls else [2, 0][:k]
    expected = graph.cypher(query, disable_optimizer=True).to_list()
    assert [row["id"] for row in expected] == expected_ids
    if not nulls:
        assert expected[0] == {"id": 2, "score": None}
    for prefix in ("", "PROFILE "):
        result = graph.cypher(prefix + query)
        assert result.to_list() == expected
        if freshness != "clean":
            assert any("read-only" in warning for warning in result.diagnostics["warnings"])


def test_complete_indexed_corpus_keeps_numeric_zero_and_stable_ties():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Doc {id:0,body:'needle'}),(:Doc {id:1,body:'needle'}),(:Doc {id:2,body:'other'})")
    graph.build_text_index("Doc", "body")
    for k in (1, 2, 3):
        query = (
            f"MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', 'needle') AS score ORDER BY score DESC LIMIT {k}"
        )
        plan = graph.cypher("EXPLAIN " + query).to_list()
        assert any(row["operation"] == "FusedTextBm25TopK" for row in plan)
        expected = graph.cypher(query, disable_optimizer=True).to_list()
        assert [row["id"] for row in expected] == list(range(k))
        if k == 3:
            assert expected[-1]["score"] == 0.0
        assert graph.cypher(query).to_list() == expected
        assert graph.cypher("PROFILE " + query).to_list() == expected


def entry_graph():
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (:Doc {id:0,body:'alpha beta',other:'omega'}),"
        "(:Doc {id:1,body:'alpha beta',other:'zeta'}),"
        "(:Doc {id:2,body:'beta gamma',other:'zeta zeta'}),"
        "(:Doc {id:3,body:'omega',other:'omega'}),(:Doc {id:4,body:'',other:'zeta'})"
    )
    graph.build_text_index("Doc", "body")
    graph.build_text_index("Doc", "other")
    return graph


def bm25_routes(graph, query, params=None, *, unfused=True):
    ordinary = graph.cypher(query, params=params)
    profiled = graph.cypher("PROFILE " + query, params=params)
    assert ordinary.to_list() == profiled.to_list()
    assert ordinary.diagnostics["warnings"] == profiled.diagnostics["warnings"]
    assert ordinary.diagnostics["retrieval"] == profiled.diagnostics["retrieval"] == []
    if unfused:
        scalar = graph.cypher(query, params=params, disable_optimizer=True)
        assert ordinary.to_list() == scalar.to_list()
        assert ordinary.diagnostics["warnings"] == scalar.diagnostics["warnings"]
    return ordinary.to_list()


@pytest.mark.parametrize("term", ["alpha", "beta", "unknown", None])
@pytest.mark.parametrize("k", [0, 1, 2, 10, 9_223_372_036_854_775_807])
def test_bm25_entry_exact_scores_ties_underfill_and_large_limits(term, k):
    graph = entry_graph()
    query = f"MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, $property, $q) AS score ORDER BY score DESC LIMIT {k}"
    result = bm25_routes(graph, query, {"property": "body", "q": term})
    assert [row["id"] for row in result] == list(range(min(k, 5)))
    if term is None:
        assert all(row["score"] is None for row in result)
    elif term == "unknown":
        assert all(row["score"] == 0.0 for row in result)


@pytest.mark.parametrize("sort", ["first", "second"])
def test_bm25_entry_independent_property_and_query_call_sites(sort):
    graph = entry_graph()
    query = (
        "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, $p1, $q1) AS first, "
        f"text_bm25(d, $p2, $q2) AS second ORDER BY {sort} DESC LIMIT 2"
    )
    result = bm25_routes(graph, query, {"p1": "body", "q1": "alpha", "p2": "other", "q2": "zeta"})
    assert [row["id"] for row in result] == ([0, 1] if sort == "first" else [2, 1])
    assert result[0]["second" if sort == "first" else "first"] == 0.0


@pytest.mark.parametrize("shape", ["missing", "numeric", "list", "secondary", "slots", "filtered", "imported"])
def test_bm25_entry_population_changes_preserve_scalar_contract(shape):
    graph = entry_graph()
    query = "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', 'alpha') AS score ORDER BY score DESC LIMIT 2"
    if shape in {"missing", "numeric", "list"}:
        body = {"missing": "", "numeric": ",body:42", "list": ",body:['alpha',null,'beta']"}[shape]
        graph.cypher(f"CREATE (:Doc {{id:9{body}}})")
    elif shape == "secondary":
        graph.cypher("CREATE (d:Other {id:9,body:'alpha'}) SET d:Doc")
        graph.build_text_index("Other", "body")
        query = query.replace("LIMIT 2", "LIMIT 10")
    elif shape == "slots":
        graph.cypher("MATCH (d:Doc) WHERE d.id=0 DELETE d")
        graph.cypher("CREATE (:Doc {id:9,body:'alpha beta'})")
    elif shape == "filtered":
        query = query.replace("RETURN", "WHERE d.id >= 1 RETURN")
    elif shape == "imported":
        query = (
            "MATCH (d:Doc) WHERE d.id IN [0,3] CALL { WITH d MATCH (d:Doc) RETURN d.id AS id, "
            "text_bm25(d, 'body', 'alpha') AS score ORDER BY score DESC LIMIT 1 } RETURN id, score ORDER BY id"
        )
    result = bm25_routes(graph, query)
    if shape in {"missing", "numeric"}:
        assert result[0] == {"id": 9, "score": None}
    elif shape == "slots":
        assert [row["id"] for row in result] == [1, 9]
    elif shape == "secondary":
        assert len(result) == 6 and any(row["id"] == 9 for row in result)
    elif shape == "imported":
        assert [row["id"] for row in result] == [0, 3] and result[1]["score"] == 0.0


def test_bm25_entry_snapshot_and_persisted_index_keep_their_own_scores(tmp_path):
    graph = entry_graph()
    frozen = graph.freeze()
    query = "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', 'alpha') AS score ORDER BY score DESC LIMIT 2"
    before = bm25_routes(frozen, query, unfused=False)
    graph.cypher("MATCH (d:Doc) WHERE d.id=0 SET d.body='omega'")
    assert bm25_routes(frozen, query, unfused=False) == before
    after = bm25_routes(graph, query)
    assert after != before
    path = tmp_path / "bm25.kgl"
    graph.save(str(path))
    loaded = kglite.load(str(path))
    loaded.read_only(True)
    assert bm25_routes(loaded, query) == after


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("expression", ["text_bm25(d, 'absent', 'alpha')", "text_bm25(d, 'body', 42)"])
def test_bm25_entry_empty_and_zero_limit_preserve_argument_evaluation(empty, expression):
    graph = entry_graph()
    if empty:
        graph.cypher("MATCH (d:Doc) DELETE d")
    query = f"MATCH (d:Doc) RETURN {expression} AS score ORDER BY score DESC LIMIT {1 if empty else 0}"
    for prefix in ("", "PROFILE "):
        if empty:
            result = graph.cypher(prefix + query)
            assert result.to_list() == [] and result.diagnostics["retrieval"] == []
        else:
            with pytest.raises(Exception, match="(?i)(text index|query string)"):
                graph.cypher(prefix + query)
