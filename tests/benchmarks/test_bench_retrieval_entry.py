"""Cypher HNSW entry and unchanged-route controls on identical vector stores.

Run with a release extension. PROFILE retains clause-attributed execution and
provides a same-index route oracle; approximate/exact recall is checked outside
measurement. All cells consume their complete result.
"""

import pytest

from tests.benchmarks.test_bench_vector_index import _build_corpus, _ensure_index

QUERY = "MATCH (n:Doc) RETURN n.id AS id, vector_score(n, 'summary_emb', $q) AS score ORDER BY score DESC LIMIT 10"


@pytest.fixture(scope="module", params=[300, 1_000, 10_000, 50_000], ids=lambda n: f"n{n}")
def entry_corpus(request):
    corpus = _build_corpus(request.param)
    _ensure_index(corpus)
    query = corpus.query(corpus.query_ids[len(corpus.query_ids) // 2])
    params = {"q": query.tolist()}
    rows = corpus.graph.cypher(QUERY, params=params).to_list()
    profiled = corpus.graph.cypher("PROFILE " + QUERY, params=params).to_list()
    assert rows == profiled
    expected = corpus.oracle_ids(corpus.query_ids[len(corpus.query_ids) // 2])
    recall = len({row["id"] for row in rows} & set(expected)) / 10
    assert recall >= 0.8
    return corpus, query, params, rows, recall


@pytest.mark.benchmark
@pytest.mark.parametrize("route", ["cypher", "profile", "filtered", "direct", "exact", "property_control"])
def test_bench_retrieval_entry(benchmark, entry_corpus, route):
    corpus, query, params, expected, recall = entry_corpus
    graph = corpus.graph
    if route == "direct":
        selection = graph.select("Doc")

        def run():
            return selection.vector_search("summary", query, top_k=10)

    elif route == "property_control":

        def run():
            return graph.cypher("MATCH (n:Doc) WHERE n.id % 3 = 0 RETURN sum(n.id) AS s").to_list()

    else:
        statement = QUERY
        if route == "profile":
            statement = "PROFILE " + statement
        elif route == "filtered":
            statement = statement.replace("RETURN", "WHERE n.id >= 0 RETURN")
        elif route == "exact":
            statement = statement.replace("$q)", "$q, {exact:true})")

        def run():
            return graph.cypher(statement, params=params).to_list()

    result = benchmark.pedantic(run, rounds=200, iterations=1, warmup_rounds=20)
    assert result
    if route in {"cypher", "profile", "filtered"}:
        assert result == expected
    elif route == "property_control":
        assert result == [{"s": sum(i for i in corpus.selected_ids if i % 3 == 0)}]
    elif route == "exact":
        query_id = corpus.query_ids[len(corpus.query_ids) // 2]
        assert [row["id"] for row in result] == corpus.oracle_ids(query_id)
    benchmark.extra_info.update(vectors=len(corpus.selected_ids), route=route, recall=recall)
