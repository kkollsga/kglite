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
