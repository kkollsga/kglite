"""Exact-vector entry creation/consumption and unchanged scalar-route controls.

Release only. Fixtures retain independent NumPy ordered-ID oracles and exact
PROFILE score values; every measured query consumes or verifies its whole result.
The Phase1 retrieval harness remains a separate unchanged-route control surface.
"""

import pytest

from tests.benchmarks.test_bench_retrieval_entry import QUERY
from tests.benchmarks.test_bench_vector_index import _build_corpus, _ensure_index


@pytest.fixture(scope="module", params=[300, 1_000, 10_000, 50_000], ids=lambda n: f"n{n}")
def exact_corpora(request):
    unindexed = _build_corpus(request.param)
    indexed = _build_corpus(request.param)
    _ensure_index(indexed)
    return {"absent": unindexed, "present": indexed}


@pytest.mark.benchmark
@pytest.mark.parametrize("index", ["absent", "present"])
@pytest.mark.parametrize("route", ["consume", "create", "profile", "direct_exact"])
def test_bench_exact_retrieval_entry(benchmark, exact_corpora, index, route):
    corpus = exact_corpora[index]
    query_id = corpus.query_ids[len(corpus.query_ids) // 2]
    vector = corpus.query(query_id)
    params = {"q": vector.tolist()}
    statement = QUERY if index == "absent" else QUERY.replace("$q)", "$q, {exact:true})")
    oracle = corpus.graph.cypher("PROFILE " + statement, params=params).to_list()
    assert [row["id"] for row in oracle] == corpus.oracle_ids(query_id)
    if route == "direct_exact":
        selection = corpus.graph.select("Doc")

        def run():
            return selection.vector_search("summary", vector, top_k=10, exact=True)

    elif route == "create":

        def run():
            return corpus.graph.cypher(statement, params=params)

    else:
        if route == "profile":
            statement = "PROFILE " + statement

        def run():
            return corpus.graph.cypher(statement, params=params).to_list()

    result = benchmark.pedantic(run, rounds=200, iterations=1, warmup_rounds=20)
    if route == "create":
        result = result.to_list()
    if route == "direct_exact":
        assert [row["id"] for row in result] == corpus.oracle_ids(query_id)
    else:
        assert result == oracle
    benchmark.extra_info.update(vectors=len(corpus.selected_ids), index=index, route=route)
