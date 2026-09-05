"""BM25 entry over the existing Zipf corpus, with scalar ordered-value oracles.

Release only. Raw captures belong in dev-docs/bench/out and are compressed by
this program's capture driver. First-after-delta cells use mean, all other cells
use min unless their own distribution is heavy tailed.
"""

import pytest

from tests.benchmarks.test_bench_text_index import CORPUS_SEED, _build_corpus

QUERY = "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', $q) AS score ORDER BY score DESC LIMIT $k"


@pytest.fixture(scope="module", params=[500, 20_000, 100_000], ids=lambda n: f"n{n}")
def bm25_entry_corpus(request):
    corpus = _build_corpus(request.param, CORPUS_SEED)
    corpus.ensure_index(auto_refresh_limit=100)
    queries = {"rare": "w07999", "medium": corpus.queries["selective"], "common": corpus.queries["mixed"]}
    oracles = {}
    for name, query in queries.items():
        # No fused consumer: this evaluates the scalar for every document and
        # Python sorts exact returned scores, retaining original slot ties.
        rows = corpus.graph.cypher(
            "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', $q) AS score",
            params={"q": query},
            disable_optimizer=True,
        ).to_list()
        assert all(row["score"] is not None for row in rows)
        oracles[name] = sorted(rows, key=lambda row: -row["score"])
    return corpus, queries, oracles


CASES = [(frequency, k) for frequency in ("rare", "medium", "common") for k in (10, 100)]
CASES += [(route, 10) for route in ("underfilled", "unknown", "filtered", "profile", "create", "property")]


@pytest.mark.benchmark
@pytest.mark.parametrize("route,k", CASES, ids=[f"{route}-k{k}" for route, k in CASES])
def test_bench_bm25_retrieval_entry(benchmark, bm25_entry_corpus, route, k):
    corpus, queries, oracles = bm25_entry_corpus
    frequency = route if route in queries else "medium"
    statement = QUERY
    params = {"q": queries[frequency], "k": k}
    if route == "underfilled":
        frequency = "rare"
        k = sum(row["score"] > 0 for row in oracles[frequency]) + 1
        params = {"q": queries[frequency], "k": k}
    elif route == "unknown":
        params["q"] = "zzabsentterm"
    elif route == "filtered":
        statement = statement.replace("RETURN", "WHERE d.id >= 0 RETURN")
    elif route == "profile":
        statement = "PROFILE " + statement
    elif route == "property":
        statement = "MATCH (d:Doc) WHERE d.id % 3 = 0 RETURN sum(d.id) AS total"
    if route == "property":
        expected = [{"total": sum(range(0, corpus.docs, 3))}]
    elif route == "unknown":
        expected = [{"id": i, "score": 0.0} for i in range(min(k, corpus.docs))]
    else:
        expected = oracles[frequency][:k]

    statement = statement.replace("$k", str(k))
    if route != "property":
        plan = corpus.graph.cypher("EXPLAIN " + statement.removeprefix("PROFILE "), params=params).to_list()
        assert any(row["operation"] == "FusedTextBm25TopK" for row in plan)

    def run():
        result = corpus.graph.cypher(statement, params=params)
        return result if route == "create" else result.to_list()

    result = benchmark.pedantic(run, rounds=200, iterations=1, warmup_rounds=20)
    if route == "create":
        result = result.to_list()
    assert result == expected
    benchmark.extra_info.update(
        documents=corpus.docs,
        route=route,
        k=k,
        positive_hits=sum(row["score"] > 0 for row in oracles[frequency]),
        statistic="min unless heavy tailed",
    )


@pytest.mark.benchmark
def test_bench_bm25_entry_first_after_delta(benchmark, bm25_entry_corpus):
    corpus, queries, _ = bm25_entry_corpus
    params = {"q": queries["medium"], "k": 10}
    generation = 0
    plan = corpus.graph.cypher("EXPLAIN " + QUERY.replace("$k", "10"), params=params).to_list()
    assert any(row["operation"] == "FusedTextBm25TopK" for row in plan)

    def dirty():
        nonlocal generation
        generation += 1
        corpus.graph.cypher(
            "MATCH (d:Doc) WHERE d.id = 0 SET d.body = $body",
            params={"body": f"{queries['medium']} revision{generation}"},
        )
        return (), {}

    def run():
        return corpus.graph.cypher(QUERY.replace("$k", "10"), params=params).to_list()

    result = benchmark.pedantic(run, setup=dirty, rounds=20, iterations=1)
    expected = corpus.graph.cypher(QUERY.replace("$k", "10"), params=params, disable_optimizer=True).to_list()
    assert result == expected
    benchmark.extra_info.update(documents=corpus.docs, route="first_after_delta", statistic="mean")
