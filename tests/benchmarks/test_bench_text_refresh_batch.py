"""First-query text refresh events, verified against independent bulk builds.

Release only. Every timed call follows a real body change or append. Means are
used for event costs; clean controls use min unless their distribution is heavy
tailed. Raw captures belong in dev-docs/bench/out and must be compressed.
"""

import sys
import time

import numpy as np
import pandas as pd
import pytest

from tests.benchmarks.test_bench_text_index import (
    CORPUS_SEED,
    VOCABULARY,
    _build_corpus,
    _term_probabilities,
    _vocabulary,
)

QUERY = "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', $q) AS score ORDER BY score DESC LIMIT 10"
LIMIT = 200_000


def _process_peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


@pytest.fixture(scope="module", params=[500, 20_000, 100_000], ids=lambda n: f"n{n}")
def refresh_batch_corpus(request):
    corpus = _build_corpus(request.param, CORPUS_SEED)
    corpus.ensure_index(auto_refresh_limit=LIMIT)
    return corpus


def _query(corpus):
    return corpus.graph.cypher(QUERY, params={"q": corpus.query}).to_list()


def _replace(corpus, delta, body):
    corpus.graph.cypher("MATCH (d:Doc) WHERE d.id < $delta SET d.body = $body", params={"delta": delta, "body": body})


def _bodies(shape, delta):
    rng = np.random.default_rng(CORPUS_SEED + delta)
    if shape == "disjoint":
        return [" ".join(f"replacement{side}token{i}" for i in range(200)) for side in ("a", "b")]
    bodies = [" ".join(_vocabulary()[rng.choice(VOCABULARY, size=200, p=_term_probabilities())]) for _ in range(2)]
    return [[body[: len(body) // 2], None, body[len(body) // 2 :]] for body in bodies] if shape == "list" else bodies


def _rebuilt_oracle(corpus):
    start = time.perf_counter()
    report = corpus.graph.build_text_index("Doc", "body", auto_refresh_limit=LIMIT)
    elapsed = time.perf_counter() - start
    assert report["indexed"] == corpus.docs
    # Scalar evaluation and Python's stable sort avoid sharing the retrieval
    # shortcut with the measured query. The index itself uses TextIndex::build.
    rows = corpus.graph.cypher(
        "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', $q) AS score",
        params={"q": corpus.query},
        disable_optimizer=True,
    ).to_list()
    assert len(rows) == corpus.docs and all(row["score"] is not None for row in rows)
    expected = sorted(rows, key=lambda row: -row["score"])[:10]
    assert _query(corpus) == expected
    return expected, elapsed


CASES = [("zipf", delta) for delta in (1, 10, 100, 1000, 1500)] + [("disjoint", 1000), ("list", 1000)]


@pytest.mark.benchmark
@pytest.mark.parametrize("shape,delta", CASES, ids=[f"{shape}-d{delta}" for shape, delta in CASES])
def test_bench_text_refresh_batch(benchmark, refresh_batch_corpus, shape, delta):
    corpus = refresh_batch_corpus
    if delta > corpus.docs:
        pytest.skip("Delta exceeds this tiny control corpus")
    bodies = _bodies(shape, delta)
    oracles = []
    rebuild_seconds = []
    for body in bodies:
        _replace(corpus, delta, body)
        oracle, elapsed = _rebuilt_oracle(corpus)
        oracles.append(oracle)
        rebuild_seconds.append(elapsed)
    state = 1
    result = None
    updates = []
    clean = []

    def setup():
        nonlocal state
        state = 1 - state
        start = time.perf_counter()
        _replace(corpus, delta, bodies[state])
        updates.append(time.perf_counter() - start)

    def run():
        nonlocal result
        result = _query(corpus)
        return result

    def verify():
        assert result == oracles[state]
        start = time.perf_counter()
        clean_result = _query(corpus)
        clean.append(time.perf_counter() - start)
        assert clean_result == oracles[state]

    result = benchmark.pedantic(run, setup=setup, teardown=verify, rounds=20, iterations=1, warmup_rounds=2)
    assert result == oracles[state]
    if benchmark.disabled:
        verify()
    offset = 0 if benchmark.disabled else 2
    benchmark.extra_info.update(
        documents=corpus.docs,
        delta=delta,
        shape=shape,
        statistic="mean (first event)",
        clean_mean_seconds=float(np.mean(clean[offset:])),
        update_mean_seconds=float(np.mean(updates[offset:])),
        oracle_rebuild_seconds=rebuild_seconds,
        process_peak_rss_bytes=_process_peak_rss_bytes(),
        oracle="independent bulk build + scalar stable sorted exact values, both states, every event",
    )


@pytest.mark.benchmark
def test_bench_text_refresh_clean(benchmark, refresh_batch_corpus):
    corpus = refresh_batch_corpus
    expected = _query(corpus)
    result = benchmark.pedantic(lambda: _query(corpus), rounds=200, warmup_rounds=20, iterations=1)
    assert result == expected
    benchmark.extra_info.update(documents=corpus.docs, statistic="min unless heavy tailed", route="clean")


@pytest.mark.benchmark
@pytest.mark.parametrize("delta", [1, 1000], ids=lambda n: f"d{n}")
def test_bench_text_refresh_append(benchmark, refresh_batch_corpus, delta):
    corpus = refresh_batch_corpus
    if delta > corpus.docs:
        pytest.skip("Large append is not a tiny control")
    original = corpus.graph
    original_docs = corpus.docs
    frame = pd.DataFrame(
        {
            "id": np.arange(original_docs, original_docs + delta),
            "title": ["appended"] * delta,
            "body": [_bodies("zipf", delta)[0]] * delta,
        }
    )

    def append():
        corpus.graph.add_nodes(frame, "Doc", "id", "title", ["body"])

    corpus.graph = original.copy()
    corpus.docs += delta
    append()
    appended = corpus.graph.cypher(
        "MATCH (d:Doc) WHERE d.id >= $first RETURN d.body AS body",
        params={"first": original_docs},
    ).to_list()
    assert appended == [{"body": frame["body"].iloc[0]}] * delta
    expected, rebuild_seconds = _rebuilt_oracle(corpus)
    updates = []
    clean = []
    result = None

    def setup():
        corpus.graph = original.copy()
        start = time.perf_counter()
        append()
        updates.append(time.perf_counter() - start)

    def run():
        nonlocal result
        result = _query(corpus)
        return result

    def verify():
        assert result == expected
        start = time.perf_counter()
        clean_result = _query(corpus)
        clean.append(time.perf_counter() - start)
        assert clean_result == expected

    try:
        result = benchmark.pedantic(run, setup=setup, teardown=verify, rounds=10, iterations=1, warmup_rounds=1)
        assert result == expected
        if benchmark.disabled:
            verify()
        offset = 0 if benchmark.disabled else 1
        benchmark.extra_info.update(
            documents=original_docs,
            delta=delta,
            route="append",
            statistic="mean (first event)",
            clean_mean_seconds=float(np.mean(clean[offset:])),
            update_mean_seconds=float(np.mean(updates[offset:])),
            oracle_rebuild_seconds=rebuild_seconds,
            process_peak_rss_bytes=_process_peak_rss_bytes(),
            oracle="independent bulk build + scalar exact sorted values, every event",
        )
    finally:
        corpus.graph = original
        corpus.docs = original_docs


@pytest.mark.benchmark
def test_bench_text_refresh_foreign_gap(benchmark, refresh_batch_corpus):
    _foreign_gap(benchmark, refresh_batch_corpus, min(1000, refresh_batch_corpus.docs))


@pytest.mark.benchmark
def test_bench_text_refresh_foreign_gap_few(benchmark, refresh_batch_corpus):
    _foreign_gap(benchmark, refresh_batch_corpus, 2)


def _foreign_gap(benchmark, corpus, changed):
    original = corpus.graph
    original_docs = corpus.docs
    foreign = 20_000
    body = _bodies("zipf", changed)[0]
    doc_frame = pd.DataFrame({"id": [original_docs], "title": ["appended"], "body": [body]})
    foreign_frame = pd.DataFrame({"id": np.arange(foreign), "title": ["foreign"] * foreign})

    def edit():
        _replace(corpus, changed, body)
        corpus.graph.add_nodes(doc_frame, "Doc", "id", "title", ["body"])
        corpus.graph.add_nodes(foreign_frame, "Foreign", "id", "title")

    corpus.graph = original.copy()
    corpus.docs += 1
    edit()
    expected, _ = _rebuilt_oracle(corpus)
    result = None
    clean = []

    def setup():
        corpus.graph = original.copy()
        edit()

    def run():
        nonlocal result
        result = _query(corpus)
        return result

    def verify():
        assert result == expected
        start = time.perf_counter()
        clean_result = _query(corpus)
        clean.append(time.perf_counter() - start)
        assert clean_result == expected

    try:
        result = benchmark.pedantic(run, setup=setup, teardown=verify, rounds=10, warmup_rounds=1, iterations=1)
        assert result == expected
        if benchmark.disabled:
            verify()
        benchmark.extra_info.update(
            documents=original_docs,
            changed=changed,
            foreign=foreign,
            route="mixed foreign gap",
            statistic="mean (first event)",
            clean_mean_seconds=float(np.mean(clean[0 if benchmark.disabled else 1 :])),
        )
    finally:
        corpus.graph = original
        corpus.docs = original_docs


@pytest.mark.benchmark
@pytest.mark.parametrize("delta", [1501, 5000], ids=lambda n: f"d{n}")
def test_bench_text_refresh_crossover(benchmark, refresh_batch_corpus, delta):
    if refresh_batch_corpus.docs < 20_000:
        pytest.skip("Crossover probe targets the two representative corpus sizes")
    test_bench_text_refresh_batch(benchmark, refresh_batch_corpus, "zipf", delta)


@pytest.fixture(scope="module")
def small_full_refresh_corpus():
    corpus = _build_corpus(5000, CORPUS_SEED)
    corpus.ensure_index(auto_refresh_limit=LIMIT)
    return corpus


@pytest.mark.benchmark
def test_bench_text_refresh_small_full(benchmark, small_full_refresh_corpus):
    test_bench_text_refresh_batch(benchmark, small_full_refresh_corpus, "zipf", 5000)
