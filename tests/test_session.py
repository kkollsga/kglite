"""Session — thread-safe, shareable concurrency handle.

`graph.session()` returns a `Session` that wraps the engine's
`Mutex<Arc<DirGraph>>`. Unlike a live `KnowledgeGraph` (single-owner,
borrow-guarded), a `Session` exposes only `&self` methods: `cypher()` reads
run lock-free against momentary snapshots, while `execute()` writes serialize
behind an internal writer lock and compose (no lost updates). Covers reads,
snapshot isolation, serialized writes, and mixed concurrent read/write — the
shared-handle failure mode that panicked on a shared live `KnowledgeGraph`.
"""

import threading

import pandas as pd
import pytest

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


def test_open_session_one_call_shared_handle(tmp_path):
    """kglite.open_session(path) loads a saved graph directly as a Session —
    the low-friction shared entry point (== load(path).session())."""
    p = tmp_path / "g.kgl"
    _graph(5).save(str(p))
    s = kglite.open_session(str(p))
    assert type(s).__name__ == "Session"
    assert _count(s) == 5
    # full fluent surface via cursor()
    assert len(s.cursor().select("Doc").to_df()) == 5


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


def test_disk_session_reuses_writer_lineage_and_composes(tmp_path):
    path = str(tmp_path / "disk-session")
    g = kglite.KnowledgeGraph(storage="disk", path=path)
    g.cypher("CREATE (:Doc {id: 1, title: 'base'})")
    s = g.session()
    old = s.snapshot()

    s.execute("CREATE (:Doc {id: 2, title: 'second'})")
    s.execute("CREATE (:Doc {id: 3, title: 'third'})")

    assert _count(s) == 3
    assert _count(old) == 1
    held = s.snapshot()
    s.execute("MATCH (n:Doc {id: 2}) SET n.title = 'edited'")
    assert held.cypher("MATCH (n:Doc {id: 2}) RETURN n.title AS t").to_list()[0]["t"] == "second"

    with pytest.raises(Exception):
        s.execute(
            "MATCH (n:Doc {id: 2}) SET n.title = 'leaked' WITH [1, 2, 3] AS xs UNWIND xs AS x RETURN x",
            max_rows=1,
        )
    s.execute("CREATE (:Doc {id: 4, title: 'after-error'})")
    assert _count(s) == 4
    title = s.cypher("MATCH (n:Doc {id: 2}) RETURN n.title AS t").to_list()[0]["t"]
    assert title == "edited"


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
    g.add_nodes(pd.DataFrame({"id": [1], "title": ["ctr"], "n": [0]}), "Doc", "id", "title")
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


def test_mixed_readers_and_writers_one_session():
    """The shared-handle failure mode, fixed: many threads reading AND writing the
    SAME shared Session at once — what tripped a process-killing borrow panic
    on a live KnowledgeGraph. With a Session it must never raise, and the
    writes must all land (readers see only consistent, never torn, state)."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": list(range(100)), "title": [f"d{i}" for i in range(100)]}),
        "Doc",
        "id",
        "title",
    )
    s = g.session()

    start = threading.Event()
    errors: list[Exception] = []
    torn: list[int] = []

    def reader():
        start.wait()
        for _ in range(80):
            try:
                c = _count(s)
                # count must always be one of the consistent values in [100, 200]
                if not (100 <= c <= 200):
                    torn.append(c)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    def writer(base: int):
        start.wait()
        for i in range(25):
            try:
                s.execute(
                    "CREATE (n:Doc {id: $id, title: 'w'})",
                    params={"id": 1000 + base * 25 + i},
                )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads += [threading.Thread(target=writer, args=(b,)) for b in range(4)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    assert errors == [], f"mixed read/write on a Session must not error: {errors[:3]}"
    assert torn == [], f"readers observed torn counts: {torn[:3]}"
    # 4 writers x 25 creates = 100 new nodes on top of the initial 100.
    assert _count(s) == 200


def test_cursor_exposes_full_fluent_surface():
    """Session.cursor() returns a snapshot-bound handle with the WHOLE fluent
    chain (not just cypher()) — select/where/sort/to_df all work."""
    s = _graph(6).session()
    cur = s.cursor()
    # full fluent chain off the cursor
    df = cur.select("Doc").where({"id": {"<": 3}}).sort("id").to_df()
    assert len(df) == 3
    # cypher read also works
    assert _count(cur) == 6


def test_cursor_snapshot_isolation_and_cow():
    """A cursor is bound to the snapshot at call time; mutating it is isolated
    (copy-on-write) and never writes back to the Session."""
    s = _graph(3).session()
    cur = s.cursor()
    cur.cypher("CREATE (n:Doc {id: 99, title: 'x'})")  # mutate the cursor
    assert _count(cur) == 4  # the cursor saw its own write
    assert _count(s) == 3  # the session is untouched (CoW isolation)
    # a fresh cursor reflects the session, not the mutated one
    assert _count(s.cursor()) == 3


def test_per_thread_cursors_off_one_session():
    """The roadmap's 'flexible' goal: N threads each take their own cursor off
    one shared Session and run full fluent chains in parallel, lock-free, with
    no single-owner borrow conflict."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(100)),
                "title": [f"d{i}" for i in range(100)],
                "team": ["A" if i % 2 == 0 else "B" for i in range(100)],
            }
        ),
        "Doc",
        "id",
        "title",
    )
    s = g.session()

    start = threading.Event()
    errors: list[Exception] = []
    bad: list[int] = []

    def worker():
        start.wait()
        for _ in range(40):
            try:
                n = len(s.cursor().select("Doc").where({"team": "A"}).to_df())
                if n != 50:
                    bad.append(n)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    assert errors == [], f"per-thread cursors must not error: {errors[:3]}"
    assert bad == [], f"per-thread cursor fluent chains returned wrong counts: {bad[:3]}"


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
