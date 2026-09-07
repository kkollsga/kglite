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
CASES += [(route, 10) for route in ("equal", "underfilled", "unknown", "filtered", "profile", "create", "property")]


@pytest.mark.benchmark
@pytest.mark.parametrize("route,k", CASES, ids=[f"{route}-k{k}" for route, k in CASES])
def test_bench_bm25_retrieval_entry(benchmark, bm25_entry_corpus, route, k):
    corpus, queries, oracles = bm25_entry_corpus
    frequency = route if route in queries else "medium"
    statement = QUERY
    params = {"q": queries[frequency], "k": k}
    if route in {"equal", "underfilled"}:
        frequency = "rare"
        k = sum(row["score"] > 0 for row in oracles[frequency]) + (route == "underfilled")
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
        assert any(row["operation"] == "FusedTextBm25TopK" for row in plan) == (k > 0)

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


@pytest.mark.benchmark
def test_bench_bm25_entry_missing_document(benchmark, bm25_entry_corpus):
    corpus, queries, _ = bm25_entry_corpus
    missing_id = corpus.docs + 1
    corpus.graph.cypher("CREATE (:Doc {id:$id})", params={"id": missing_id})
    statement = QUERY.replace("$k", "10")
    params = {"q": queries["medium"]}
    # The untimed scalar oracle also consumes the creation delta, leaving a
    # clean index with one fewer document than the type's candidate population.
    expected = corpus.graph.cypher(statement, params=params, disable_optimizer=True).to_list()
    assert expected[0] == {"id": missing_id, "score": None}
    plan = corpus.graph.cypher("EXPLAIN " + statement, params=params).to_list()
    assert any(row["operation"] == "FusedTextBm25TopK" for row in plan)

    def run():
        return corpus.graph.cypher(statement, params=params).to_list()

    result = benchmark.pedantic(run, rounds=200, iterations=1, warmup_rounds=20)
    assert result == expected
    benchmark.extra_info.update(documents=corpus.docs, route="missing_document", statistic="min unless heavy tailed")


def _min_seconds(run, rounds=40, warmup=5):
    """Min of `rounds` timed calls — the statistic this file's other cells use,
    taken by hand because one test needs two shapes measured against each
    other rather than against a stored baseline."""
    import time

    for _ in range(warmup):
        run()
    return min(_timed(run, time) for _ in range(rounds))


def _timed(run, time):
    start = time.perf_counter()
    run()
    return time.perf_counter() - start


#: The underfilled top-k may cost this much more than the filled one. The
#: shapes differ by a single `k`, so anything beyond a small constant is the
#: operator having declined and replayed the corpus per row — 43-64x before
#: the tail was filled from the proven population.
UNDERFILL_RATIO_CEILING = 2.0


@pytest.mark.benchmark
def test_bench_bm25_underfilled_top_k_costs_what_the_filled_one_costs():
    # Its own corpus, not the module fixture: the missing-document cell adds an
    # unindexed Doc to that one, which makes the population a superset of the
    # index and takes every route here off the operator being measured.
    corpus = _build_corpus(20_000, CORPUS_SEED)
    corpus.ensure_index(auto_refresh_limit=100)
    params = {"q": "w07999"}
    ranked = sorted(
        corpus.graph.cypher(
            "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', $q) AS score",
            params=params,
            disable_optimizer=True,
        ).to_list(),
        key=lambda row: -row["score"],
    )
    positives = sum(row["score"] > 0 for row in ranked)
    assert 0 < positives < 100, f"the gate needs a rare term, not {positives} of {corpus.docs}"

    def route(k, fused=True):
        statement = QUERY.replace("$k", str(k))
        expected = ranked[:k]
        passes = {} if fused else {"disabled_passes": ["fuse_text_bm25_order_limit"]}

        def run():
            return corpus.graph.cypher(statement, params=params, **passes).to_list()

        # Non-vacuity: a cell that answered nothing, or answered from a
        # different ranking, would be fast for the wrong reason.
        assert run() == expected
        return _min_seconds(run)

    filled = route(positives)
    underfilled = route(positives + 1)
    assert underfilled <= UNDERFILL_RATIO_CEILING * filled, (
        f"underfilled top-k (k={positives + 1}) cost {underfilled * 1e3:.2f} ms against "
        f"{filled * 1e3:.2f} ms for the filled one (k={positives}) — "
        f"{underfilled / filled:.1f}x, ceiling {UNDERFILL_RATIO_CEILING}x"
    )
    # The ceiling above can only fail if the operator declines, so pin that the
    # instrument can still see a decline: the same query with the fusion pass
    # off is the pre-fill cost, and it must stay far above what we just timed.
    declined = route(positives + 1, fused=False)
    assert declined >= 5 * underfilled, (
        f"per-row ranking of the same query cost {declined * 1e3:.2f} ms against "
        f"{underfilled * 1e3:.2f} ms fused — the ratio ceiling above cannot go red"
    )
