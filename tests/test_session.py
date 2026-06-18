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
