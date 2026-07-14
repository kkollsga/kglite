"""FrozenGraph — immutable, concurrently-readable snapshot (R2).

Concurrency Tier 2 from the operator's roadmap: `graph.freeze()` returns a
read-only view that

- answers the same read queries as the source graph;
- rejects mutations (it's immutable);
- is stable under copy-on-write — mutating the source after freezing does not
  change the snapshot;
- can be queried from many threads at once without the single-owner borrow
  error a live KnowledgeGraph raises.
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


def _count(view, q: str) -> int:
    return view.cypher(q).to_list()[0]["c"]


def _disk_graph(path, n: int = 2) -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph(storage="disk", path=str(path))
    graph.add_nodes(
        pd.DataFrame({"id": list(range(n)), "title": [f"d{i}" for i in range(n)]}),
        "Doc",
        "id",
        "title",
    )
    graph.save(str(path))
    return graph


def test_freeze_reads_match_source():
    g = _graph(3)
    fz = g.freeze()
    assert _count(fz, "MATCH (n:Doc) RETURN count(n) AS c") == 3
    assert fz.node_count() >= 3
    assert fz.node_types == ["Doc"]


def test_freeze_rejects_mutations():
    fz = _graph(2).freeze()
    for q in [
        "CREATE (n:Doc {id: 99})",
        "MATCH (n:Doc) SET n.x = 1",
        "MATCH (n:Doc) DELETE n",
        "MERGE (n:Doc {id: 1})",
    ]:
        with pytest.raises(ValueError, match="immutable|frozen|snapshot"):
            fz.cypher(q)


def test_freeze_is_stable_under_source_mutation():
    g = _graph(3)
    fz = g.freeze()
    # Mutate the source after freezing.
    g.add_nodes(pd.DataFrame({"id": [100, 101], "title": ["x", "y"]}), "Doc", "id", "title")
    # The snapshot is unchanged (copy-on-write); the source grew.
    assert _count(fz, "MATCH (n:Doc) RETURN count(n) AS c") == 3
    assert _count(g, "MATCH (n:Doc) RETURN count(n) AS c") == 5


@pytest.mark.parametrize(
    "hold_snapshot",
    [lambda graph: graph.freeze(), lambda graph: graph.select("Doc")],
    ids=["freeze", "fluent-view"],
)
def test_disk_snapshot_keeps_stable_reads_while_source_mutates_and_saves(tmp_path, hold_snapshot):
    path = tmp_path / "graph"
    graph = _disk_graph(path)
    snapshot = hold_snapshot(graph)

    graph.cypher("CREATE (:Doc {id: 10, title: 'source'})")
    assert _count(snapshot, "MATCH (n:Doc) RETURN count(n) AS c") == 2
    assert _count(graph, "MATCH (n:Doc) RETURN count(n) AS c") == 3
    graph.save()

    assert _count(snapshot, "MATCH (n:Doc) RETURN count(n) AS c") == 2
    assert _count(kglite.open(str(path)), "MATCH (n:Doc) RETURN count(n) AS c") == 3
    assert not list(path.glob(".working-*"))


def test_disk_save_with_held_freeze_reuses_writer_lineage(tmp_path):
    path = tmp_path / "graph"
    graph = _disk_graph(path)
    snapshot = graph.freeze()
    current_before = (path / "CURRENT").read_text(encoding="utf-8")

    graph.save()

    current_after = (path / "CURRENT").read_text(encoding="utf-8")
    assert current_after != current_before
    assert _count(snapshot, "MATCH (n:Doc) RETURN count(n) AS c") == 2
    assert _count(kglite.open(str(path)), "MATCH (n:Doc) RETURN count(n) AS c") == 2
    assert not list(path.glob(".working-*"))


def test_freeze_cypher_with_params_and_df():
    g = _graph(5)
    fz = g.freeze()
    rows = fz.cypher("MATCH (n:Doc) WHERE n.id >= $lo RETURN n.id AS id", params={"lo": 3}).to_list()
    assert {r["id"] for r in rows} == {3, 4}
    df = fz.cypher("MATCH (n:Doc) RETURN n.id AS id", to_df=True)
    assert len(df) == 5


def test_freeze_preserves_embeddings_for_vector_read():
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2, 3], "title": ["a", "b", "c"]}), "Doc", "id", "title")
    g.add_embeddings(
        "Doc",
        "summary",
        {1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0], 3: [0.9, 0.1, 0.0, 0.0]},
    )
    fz = g.freeze()
    # vector_score() against the frozen snapshot ranks by similarity to a query
    # vector — a read-only semantic search, no mutation, no embedder rebind.
    rows = fz.cypher(
        "MATCH (n:Doc) RETURN n.id AS id, vector_score(n, 'summary_emb', [1.0, 0.0, 0.0, 0.0]) AS s "
        "ORDER BY s DESC LIMIT 1"
    ).to_list()
    assert rows[0]["id"] == 1


def test_many_threads_read_one_frozen_snapshot_concurrently():
    """The headline guarantee: N threads querying the SAME FrozenGraph at once
    never raise the single-owner borrow error and always get the right answer."""
    g = _graph(200)
    fz = g.freeze()

    start = threading.Event()
    errors: list[Exception] = []
    bad_counts: list[int] = []

    def worker():
        start.wait()
        for _ in range(50):
            try:
                c = _count(fz, "MATCH (n:Doc) RETURN count(n) AS c")
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

    assert errors == [], f"concurrent reads on a frozen snapshot must not error: {errors[:3]}"
    assert bad_counts == [], f"concurrent reads returned wrong counts: {bad_counts[:3]}"


def test_freeze_while_source_mutated_in_another_thread():
    """A frozen snapshot stays correct even while the source graph is being
    mutated on the owning thread — the COW snapshot is fully decoupled."""
    g = _graph(100)
    fz = g.freeze()

    errors: list[Exception] = []

    def reader():
        for _ in range(100):
            try:
                assert _count(fz, "MATCH (n:Doc) RETURN count(n) AS c") == 100
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    # Meanwhile mutate the source on this thread — does not touch the snapshot.
    for i in range(100, 150):
        g.add_nodes(pd.DataFrame({"id": [i], "title": [str(i)]}), "Doc", "id", "title")
    t.join()
    assert errors == []
    # Snapshot still 100; source grew to 150.
    assert _count(fz, "MATCH (n:Doc) RETURN count(n) AS c") == 100
    assert _count(g, "MATCH (n:Doc) RETURN count(n) AS c") == 150
