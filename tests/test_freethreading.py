"""Concurrency correctness + free-threading (no-GIL) validation.

The concurrency body runs on every build and exercises the supported
shared-access model: many threads hammering one ``Session`` (lock-free
reads, serialized composable writes) plus concurrent ``FrozenGraph``
readers. On a free-threaded (``Py_GIL_DISABLED``) interpreter it also
asserts the GIL is actually off — backing the module's
``gil_used = false`` declaration with a real run rather than a claim.

Run under free-threaded CPython:
    uv venv --python cpython-3.14t /tmp/ftvenv
    uv pip install --python /tmp/ftvenv maturin pandas
    VIRTUAL_ENV=/tmp/ftvenv maturin develop --no-default-features \
        --features python-extension
    /tmp/ftvenv/bin/python -m pytest tests/test_freethreading.py -v
"""

import concurrent.futures as cf
import sys

import kglite


def _gil_disabled() -> bool:
    # sys._is_gil_enabled exists on 3.13+; False only on a free-threaded build.
    getter = getattr(sys, "_is_gil_enabled", None)
    return getter is not None and not getter()


def _seed(n_people: int = 2000) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher(f"UNWIND range(1, {n_people}) AS i CREATE (:Person {{id: i}})")
    g.cypher("MATCH (a:Person), (b:Person) WHERE b.id = a.id + 1 CREATE (a)-[:KNOWS]->(b)")
    return g


def test_concurrent_session_reads():
    """Many threads share one Session; every read sees the full graph."""
    g = _seed()
    s = g.session()
    expected = s.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"]

    def worker(_):
        total = 0
        for _ in range(20):
            rows = s.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()
            total += rows[0]["c"]
        return total

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(8)))

    assert all(r == expected * 20 for r in results)


def test_concurrent_writes_compose():
    """Concurrent Session.execute() writes serialize without lost updates."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Counter {id: 1})")
    s = g.session()
    n_threads, per_thread = 8, 25

    def writer(t):
        for k in range(per_thread):
            s.execute(f"CREATE (:Item {{tid: {t}, k: {k}}})")

    with cf.ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(writer, range(n_threads)))

    got = s.cypher("MATCH (n:Item) RETURN count(n) AS c").to_list()[0]["c"]
    assert got == n_threads * per_thread, f"lost updates: {got}"


def test_concurrent_frozen_readers():
    """A single FrozenGraph snapshot is safe for many concurrent readers."""
    g = _seed()
    fz = g.freeze()

    def worker(_):
        return fz.cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(*) AS c").to_list()[0]["c"]

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(16)))
    assert len(set(results)) == 1  # all readers agree


def test_gil_actually_disabled_when_freethreaded():
    """On a free-threaded build, the GIL must really be off — otherwise the
    module's `gil_used = false` declaration would be silently re-enabling it."""
    if not _gil_disabled():
        import pytest

        pytest.skip("not a free-threaded interpreter")
    # Running queries from threads must not have re-enabled the GIL.
    g = _seed(200)
    s = g.session()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda _: s.cypher("MATCH (n) RETURN count(n)").to_list(), range(4)))
    assert not sys._is_gil_enabled(), "GIL was re-enabled under a query load"
