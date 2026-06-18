"""Session — thread-safe, shareable concurrency handle (Phase 1: reads).

`graph.session()` returns a `Session` that wraps the engine's
`Mutex<Arc<DirGraph>>`. Unlike a live `KnowledgeGraph` (single-owner,
borrow-guarded), a `Session` exposes only `&self` methods, so many threads
can read it at once lock-free. This module covers the read surface; writes
are covered in the Phase 2 / Phase 4 sections of the suite.
"""

import threading

import pandas as pd

import kglite


def _graph(n: int = 3) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": list(range(n)), "title": [f"d{i}" for i in range(n)]}),
        "Doc",
        "id",
        "title",
    )
    return g


def _count(view, q: str = "MATCH (n:Doc) RETURN count(n) AS c") -> int:
    return view.cypher(q).to_list()[0]["c"]


def test_session_reads_match_source():
    g = _graph(3)
    s = g.session()
    assert _count(s) == 3
    assert s.node_count() == 3
    assert s.node_types == ["Doc"]


def test_session_cypher_with_params_and_df():
    s = _graph(5).session()
    rows = s.cypher("MATCH (n:Doc) WHERE n.id >= $lo RETURN n.id AS id", params={"lo": 3}).to_list()
    assert {r["id"] for r in rows} == {3, 4}
    df = s.cypher("MATCH (n:Doc) RETURN n.id AS id", to_df=True)
    assert len(df) == 5


def test_session_cypher_rejects_mutations():
    s = _graph(2).session()
    for q in [
        "CREATE (n:Doc {id: 99})",
        "MATCH (n:Doc) SET n.x = 1",
        "MATCH (n:Doc) DELETE n",
        "MERGE (n:Doc {id: 1})",
    ]:
        try:
            s.cypher(q)
            raise AssertionError(f"mutation not rejected: {q}")
        except ValueError as e:
            assert "read-only" in str(e) or "execute()" in str(e)


def test_session_snapshot_is_frozen_and_stable():
    g = _graph(3)
    s = g.session()
    fz = s.snapshot()
    assert _count(fz) == 3
    # The snapshot is a FrozenGraph — immutable.
    assert "FrozenGraph" in repr(fz)


def test_session_version_reflects_state():
    s = _graph(3).session()
    # add_nodes bumped the source version before session(); a read doesn't move it.
    v0 = s.version()
    s.cypher("MATCH (n:Doc) RETURN count(n) AS c")
    assert s.version() == v0


def test_session_execute_basic_write():
    s = _graph(2).session()
    s.execute("CREATE (n:Doc {id: 99, title: 'new'})")
    assert _count(s) == 3
    assert s.version() > 0
    s.execute("MATCH (n:Doc {id: 99}) SET n.title = 'edited'")
    title = s.cypher("MATCH (n:Doc {id: 99}) RETURN n.title AS t").to_list()[0]["t"]
    assert title == "edited"
    s.execute("MATCH (n:Doc {id: 99}) DELETE n")
    assert _count(s) == 2


def test_session_execute_read_fastpath():
    """A read passed to execute() is fast-pathed to the read path (no
    working-copy materialisation) and returns rows."""
    s = _graph(3).session()
    rows = s.execute("MATCH (n:Doc) RETURN count(n) AS c").to_list()
    assert rows[0]["c"] == 3


def test_session_snapshot_isolation_under_write():
    """A snapshot taken before a write does not observe the write (CoW)."""
    s = _graph(3).session()
    fz = s.snapshot()
    s.execute("CREATE (n:Doc {id: 100, title: 'x'})")
    # the live session sees 4; the snapshot is frozen at 3
    assert _count(s) == 4
    assert _count(fz) == 3


def test_concurrent_writes_compose_no_lost_updates():
    """The headline write guarantee: N threads each incrementing a shared
    counter must compose — every increment lands, none is lost. This is the
    serialized-writer property that a naive shared mutable handle lacks."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1], "title": ["ctr"], "n": [0]}), "Doc", "id", "title"
    )
    s = g.session()

    def worker():
        for _ in range(100):
            s.execute("MATCH (n:Doc {id: 1}) SET n.n = n.n + 1")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = s.cypher("MATCH (n:Doc {id: 1}) RETURN n.n AS n").to_list()[0]["n"]
    assert final == 800, f"lost updates: {final} != 800"


def test_many_threads_read_one_session_concurrently():
    """Headline guarantee for reads: N threads querying the SAME Session at once
    never raise the single-owner borrow error and always get the right answer."""
    s = _graph(200).session()

    start = threading.Event()
    errors: list[Exception] = []
    bad_counts: list[int] = []

    def worker():
        start.wait()
        for _ in range(50):
            try:
                c = _count(s)
                if c != 200:
                    bad_counts.append(c)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent reads on a Session must not error: {errors[:3]}"
    assert bad_counts == [], f"concurrent reads returned wrong counts: {bad_counts[:3]}"
