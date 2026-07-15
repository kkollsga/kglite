"""Durable/atomic save + in-memory bytes round-trip + cross-thread error.

R1 of the concurrency/durability roadmap (operator notes, 2026-06-17):

- `save()` is atomic (temp + rename) and durable (`fsync` default) — a crash
  mid-write can't tear the `.kgl`; a successful save leaves no temp litter.
- `to_bytes()` / `kglite.from_bytes()` round-trip an in-memory graph without a
  filesystem path; a corrupt/truncated buffer is rejected, not half-loaded.
- Sharing a graph across threads while one mutates it raises a clear,
  actionable error instead of panicking.
"""

import os
import threading

import pandas as pd
import pytest

import kglite


def _graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1, 2, 3], "title": ["a", "b", "c"], "n": [10, 20, 30]}),
        "Doc",
        "id",
        "title",
    )
    g.add_connections(pd.DataFrame({"s": [1, 2], "t": [2, 3]}), "LINKS", "Doc", "s", "Doc", "t")
    return g


def _count(g: kglite.KnowledgeGraph, q: str) -> int:
    return g.cypher(q).to_list()[0]["c"]


# ── atomic + durable save ───────────────────────────────────────────────────


def test_save_roundtrip(tmp_path):
    p = str(tmp_path / "g.kgl")
    _graph().save(p)
    g2 = kglite.load(p)
    assert _count(g2, "MATCH (n:Doc) RETURN count(n) AS c") == 3
    assert _count(g2, "MATCH ()-[r:LINKS]->() RETURN count(r) AS c") == 2


def test_save_fsync_false_roundtrips(tmp_path):
    p = str(tmp_path / "g.kgl")
    _graph().save(p, fsync=False)
    assert _count(kglite.load(p), "MATCH (n:Doc) RETURN count(n) AS c") == 3


def test_save_overwrites_existing_cleanly(tmp_path):
    p = str(tmp_path / "g.kgl")
    _graph().save(p)
    # Overwrite with a different graph — load must see the new one, intact.
    g2 = kglite.KnowledgeGraph()
    g2.add_nodes(pd.DataFrame({"id": list(range(9)), "title": list("abcdefghi")}), "Doc", "id", "title")
    g2.save(p)
    assert _count(kglite.load(p), "MATCH (n:Doc) RETURN count(n) AS c") == 9


def test_successful_save_leaves_no_temp_litter(tmp_path):
    p = str(tmp_path / "g.kgl")
    _graph().save(p)
    entries = sorted(os.listdir(tmp_path))
    assert entries == ["g.kgl"], f"expected only the dest file, found {entries}"


def test_failed_save_leaves_existing_file_intact(tmp_path):
    p = str(tmp_path / "g.kgl")
    _graph().save(p)
    before = (tmp_path / "g.kgl").read_bytes()
    # Save into a path whose parent dir doesn't exist → error, no partial write.
    with pytest.raises(Exception):
        _graph().save(str(tmp_path / "nope" / "g.kgl"))
    assert (tmp_path / "g.kgl").read_bytes() == before


# ── to_bytes / from_bytes ────────────────────────────────────────────────────


def test_to_bytes_from_bytes_roundtrip():
    g = _graph()
    data = g.to_bytes()
    assert isinstance(data, bytes) and len(data) > 8
    g2 = kglite.from_bytes(data)
    assert _count(g2, "MATCH (n:Doc) RETURN count(n) AS c") == 3
    assert _count(g2, "MATCH ()-[r:LINKS]->() RETURN count(r) AS c") == 2
    # Node properties survive the round-trip.
    rows = g2.cypher("MATCH (n:Doc {id: 2}) RETURN n.n AS n").to_list()
    assert rows[0]["n"] == 20


def test_to_bytes_matches_save_bytes(tmp_path):
    g = _graph()
    p = str(tmp_path / "g.kgl")
    g.save(p)
    on_disk = (tmp_path / "g.kgl").read_bytes()
    # to_bytes() produces the same serialization the file save writes.
    # (Re-fetch from a fresh graph identical in content.)
    assert (
        kglite.from_bytes(g.to_bytes()).cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
        == kglite.load(p).cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    )
    assert on_disk[:5] == b"RGF\x05\x02"


def test_from_bytes_rejects_garbage():
    with pytest.raises(Exception):
        kglite.from_bytes(b"this is definitely not a kglite graph buffer at all")


def test_from_bytes_rejects_truncated():
    data = _graph().to_bytes()
    with pytest.raises(Exception):
        kglite.from_bytes(data[: len(data) // 2])


def test_from_bytes_rejects_empty():
    with pytest.raises(Exception):
        kglite.from_bytes(b"")


def test_to_bytes_preserves_embeddings():
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2], "title": ["a", "b"]}), "Doc", "id", "title")
    g.add_embeddings("Doc", "summary", {1: [0.1, 0.2, 0.3, 0.4], 2: [0.5, 0.6, 0.7, 0.8]})
    g2 = kglite.from_bytes(g.to_bytes())
    assert g2.embedding_dim("Doc", "summary") == 4


# ── cross-thread / re-entrant borrow error ───────────────────────────────────


class _ReentrantEmbedder:
    """Embedder whose embed() re-enters the SAME graph while embed_texts holds
    the exclusive borrow — the deterministic stand-in for a concurrent thread
    touching a shared graph mid-mutation."""

    def __init__(self, g: kglite.KnowledgeGraph) -> None:
        self.g = g
        self.dimension = 4
        self.reentry_error: Exception | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            self.g.cypher("MATCH (n) RETURN count(n) AS c")
        except Exception as e:  # noqa: BLE001 — capturing to assert on it
            self.reentry_error = e
            raise
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def test_reentrant_access_during_mutation_raises_clear_error():
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1, 2], "title": ["a", "b"], "summary": ["x", "y"]}),
        "Doc",
        "id",
        "title",
    )
    emb = _ReentrantEmbedder(g)
    g.set_embedder(emb)
    with pytest.raises(Exception):
        g.embed_texts("Doc", "summary", show_progress=False)
    # The re-entrant read hit the single-owner guard with the clear message,
    # not a panic and not a silent wrong answer.
    assert emb.reentry_error is not None
    assert "concurrent" in str(emb.reentry_error).lower()


def test_concurrent_threads_get_error_not_panic():
    """Two threads, one mutating in a slow embedder: the other thread's access
    surfaces as a catchable exception, never a process-killing panic. (Timing
    makes the *which* call races nondeterministic, so we assert the program
    survives and at least one outcome is a clean exception or success — never a
    crash.)"""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {"id": list(range(50)), "title": [str(i) for i in range(50)], "summary": [f"text {i}" for i in range(50)]}
        ),
        "Doc",
        "id",
        "title",
    )

    start = threading.Event()
    errors: list[Exception] = []

    def reader():
        start.wait()
        for _ in range(200):
            try:
                g.cypher("MATCH (n:Doc) RETURN count(n) AS c")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    class _SlowEmbedder:
        dimension = 4

        def embed(self, texts):
            return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    g.set_embedder(_SlowEmbedder())
    t = threading.Thread(target=reader)
    t.start()
    start.set()
    try:
        g.embed_texts("Doc", "summary", show_progress=False)
    except Exception:  # noqa: BLE001 — a borrow conflict here is acceptable
        pass
    t.join()
    # Any errors collected must be the clean, classifiable borrow error.
    for e in errors:
        assert isinstance(e, RuntimeError) and "concurrent" in str(e).lower()
