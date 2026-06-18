"""Adversarial concurrency stress harness for `Session` (opt-in: -m stress).

These run the Session concurrency scenarios under sustained heavy load — many
threads, thousands of iterations — and assert hard invariants every time. The
point is to surface *timing-dependent* races that the low-iteration functional
tests in test_session.py cannot: a single missed lock, a torn read, or a lost
update shows up here as a wrong count, not a clean pass.

Run: pytest tests/test_session_stress.py -m stress
Default suite excludes them (slow). Treat a failure here as a real concurrency
bug, not flakiness — the invariants are exact, not statistical.
"""

import threading

import pandas as pd
import pytest

import kglite

THREADS = 16


def _counter_graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1], "title": ["ctr"], "n": [0]}), "C", "id", "title")
    return g


def _docs(n: int) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(n)),
                "title": [f"d{i}" for i in range(n)],
                "team": ["A" if i % 2 == 0 else "B" for i in range(n)],
            }
        ),
        "Doc",
        "id",
        "title",
    )
    return g


@pytest.mark.stress
def test_write_compose_heavy_contention():
    """The strongest lost-update detector: THREADS writers each do PER
    read-modify-write increments of one shared counter through the Session.
    Every increment must land — final must equal THREADS*PER exactly."""
    per = 1500
    s = _counter_graph().session()

    def worker():
        for _ in range(per):
            s.execute("MATCH (n:C {id: 1}) SET n.n = n.n + 1")

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = s.cypher("MATCH (n:C {id: 1}) RETURN n.n AS n").to_list()[0]["n"]
    assert final == THREADS * per, f"lost updates: {final} != {THREADS * per}"


@pytest.mark.stress
def test_mixed_read_write_invariants():
    """Writers append nodes while readers count concurrently. Readers must
    never see an impossible count (outside the monotonic range), never error,
    and the final count must be exact."""
    start_n = 50
    per_writer = 300
    writers = 8
    readers = 8
    s = _docs(start_n).session()

    errors: list[Exception] = []
    torn: list[int] = []
    done = threading.Event()

    def writer(base: int):
        for i in range(per_writer):
            try:
                s.execute(
                    "CREATE (n:Doc {id: $id, title: 'w'})",
                    params={"id": 10_000 + base * per_writer + i},
                )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    def reader():
        # read until all writers finish; counts only ever grow
        while not done.is_set():
            try:
                c = s.cypher("MATCH (n:Doc) RETURN count(n) AS c").to_list()[0]["c"]
                if not (start_n <= c <= start_n + writers * per_writer):
                    torn.append(c)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    wts = [threading.Thread(target=writer, args=(b,)) for b in range(writers)]
    rts = [threading.Thread(target=reader) for _ in range(readers)]
    for t in rts:
        t.start()
    for t in wts:
        t.start()
    for t in wts:
        t.join()
    done.set()
    for t in rts:
        t.join()

    assert errors == [], f"errors under mixed load: {errors[:3]}"
    assert torn == [], f"readers observed impossible counts: {torn[:3]}"
    final = s.cypher("MATCH (n:Doc) RETURN count(n) AS c").to_list()[0]["c"]
    assert final == start_n + writers * per_writer


@pytest.mark.stress
def test_snapshot_isolation_under_writes():
    """A held snapshot must NEVER observe writes that land after it was taken,
    no matter how much concurrent write pressure there is."""
    s = _docs(40).session()
    bad: list[int] = []
    errors: list[Exception] = []
    stop = threading.Event()

    def writer(base: int):
        i = 0
        while not stop.is_set():
            try:
                s.execute(
                    "CREATE (n:Doc {id: $id, title: 'w'})",
                    params={"id": 100_000 + base * 1_000_000 + i},
                )
                i += 1
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    wts = [threading.Thread(target=writer, args=(b,)) for b in range(4)]
    for t in wts:
        t.start()
    try:
        # repeatedly snapshot and immediately re-read it many times while writes
        # rain down; each snapshot is frozen at its capture count.
        for _ in range(2000):
            fz = s.snapshot()
            c0 = fz.cypher("MATCH (n:Doc) RETURN count(n) AS c").to_list()[0]["c"]
            c1 = fz.cypher("MATCH (n:Doc) RETURN count(n) AS c").to_list()[0]["c"]
            if c0 != c1:
                bad.append(c1)
    finally:
        stop.set()
        for t in wts:
            t.join()

    assert errors == [], f"errors during snapshot isolation stress: {errors[:3]}"
    assert bad == [], f"a frozen snapshot changed under concurrent writes: {bad[:3]}"


@pytest.mark.stress
def test_per_thread_cursor_fluent_under_load():
    """THREADS threads each repeatedly take their own cursor off one shared
    Session and run a full fluent chain. Every chain must return the exact
    expected count, with zero errors — no single-owner borrow conflict."""
    n = 200  # 100 team A, 100 team B
    s = _docs(n).session()
    errors: list[Exception] = []
    bad: list[int] = []

    def worker():
        for _ in range(200):
            try:
                got = len(s.cursor().select("Doc").where({"team": "A"}).sort("id").to_df())
                if got != n // 2:
                    bad.append(got)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"per-thread cursor errors: {errors[:3]}"
    assert bad == [], f"cursor fluent chains returned wrong counts: {bad[:3]}"
