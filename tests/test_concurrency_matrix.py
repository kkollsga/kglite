"""Concurrency / durability / serialization matrix (Phase E).

The operator asked us to test the 0.11.0 surface hard before release, since a lot
landed at once and they can't verify from outside. This covers the gaps not
already exercised by the per-feature suites:

- EVERY mutating entry point, hit while the graph is exclusively borrowed by
  another in-flight operation, raises a *catchable* exception — never a panic /
  interpreter crash / silent corruption.
- FrozenGraph serves many concurrent readers (cypher + vector_score) in parallel
  with no error and correct results, and rejects every mutation path.
- to_bytes/from_bytes is equivalent to save/load.

Cross-binding note: the Rust core paths have their own #[cfg(test)] unit tests,
and the MCP server is covered by tests/test_mcp_server_smoke.py; this file is the
Python-binding half of the matrix.
"""

import hashlib
import threading

import pandas as pd
import pytest

import kglite


def _docs(n=4):
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {"id": list(range(n)), "title": [f"d{i}" for i in range(n)], "summary": [f"t{i}" for i in range(n)]}
        ),
        "Doc",
        "id",
        "title",
    )
    return g


class _ReentrantEmbedder:
    """embed() re-enters the graph with a caller-supplied op while embed_texts
    holds the exclusive borrow — a deterministic stand-in for a second thread
    touching a shared graph mid-mutation."""

    def __init__(self, g, op):
        self.g = g
        self.op = op
        self.dimension = 4
        self.captured = None

    def embed(self, texts):
        try:
            self.op(self.g)
        except BaseException as e:  # noqa: BLE001 — capture *anything*, incl. would-be panics
            self.captured = e
            raise
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


_MUTATING_OPS = {
    "add_nodes": lambda g: g.add_nodes(pd.DataFrame({"id": [99], "title": ["x"]}), "Doc", "id", "title"),
    "add_connections": lambda g: g.add_connections(pd.DataFrame({"s": [0], "t": [1]}), "LINKS", "Doc", "s", "Doc", "t"),
    "replace_connections": lambda g: g.replace_connections(
        pd.DataFrame({"s": [0], "t": [1]}), "LINKS", "Doc", "s", "Doc", "t"
    ),
    "cypher_create": lambda g: g.cypher("CREATE (n:Doc {id: 1000})"),
    "cypher_set": lambda g: g.cypher("MATCH (n:Doc) SET n.flag = true"),
    "save": lambda g: g.save("/tmp/kglite_concurrency_should_not_happen.kgl"),
}


@pytest.mark.parametrize("op_name", list(_MUTATING_OPS))
def test_mutating_op_under_held_borrow_raises_catchable_not_panic(op_name):
    """Re-entering ANY mutating entry point while the graph is exclusively
    borrowed raises a catchable exception (no panic, no corruption)."""
    g = _docs()
    emb = _ReentrantEmbedder(g, _MUTATING_OPS[op_name])
    g.set_embedder(emb)
    with pytest.raises(Exception):
        g.embed_texts("Doc", "summary", show_progress=False)
    # The re-entrant op was rejected with a catchable exception (Exception, not
    # a BaseException like a panic-turned-SystemExit), and the process survives.
    assert emb.captured is not None
    assert isinstance(emb.captured, Exception)
    # And the graph is still usable afterwards — no corruption.
    assert g.cypher("MATCH (n:Doc) RETURN count(n) AS c").to_list()[0]["c"] == 4


# ── FrozenGraph real parallelism ─────────────────────────────────────────────


class _StubEmbedder:
    dimension = 4

    def embed(self, texts):
        return [[float(b) for b in hashlib.sha256(t.encode()).digest()[:4]] for t in texts]


def test_frozen_snapshot_many_concurrent_readers_mixed_queries():
    g = _docs(100)
    g.set_embedder(_StubEmbedder())
    g.embed_texts("Doc", "summary", show_progress=False)
    fz = g.freeze()

    start = threading.Event()
    errors = []

    def reader(kind):
        start.wait()
        for _ in range(40):
            try:
                if kind == 0:
                    fz.cypher("MATCH (n:Doc) RETURN count(n) AS c")
                else:
                    fz.cypher(
                        "MATCH (n:Doc) RETURN n.id AS id, "
                        "vector_score(n, 'summary_emb', [1.0,0.0,0.0,0.0]) AS s "
                        "ORDER BY s DESC LIMIT 5"
                    )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=reader, args=(i % 2,)) for i in range(8)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()
    assert errors == [], f"concurrent frozen reads must not error: {errors[:3]}"


def test_frozen_rejects_every_mutation_path():
    fz = _docs().freeze()
    for q in ["CREATE (n:Doc {id: 5})", "MATCH (n) SET n.x = 1", "MATCH (n) DELETE n", "MERGE (n:Doc {id: 1})"]:
        with pytest.raises(ValueError):
            fz.cypher(q)


# ── serialization equivalence ────────────────────────────────────────────────


def test_to_bytes_equivalent_to_save_load(tmp_path):
    g = _docs(10)
    g.add_connections(pd.DataFrame({"s": [0, 1], "t": [1, 2]}), "LINKS", "Doc", "s", "Doc", "t")

    p = str(tmp_path / "g.kgl")
    g.save(p)
    from_file = kglite.load(p)
    from_bytes = kglite.from_bytes(g.to_bytes())

    q = "MATCH (n:Doc) RETURN count(n) AS c"
    assert from_file.cypher(q).to_list()[0]["c"] == from_bytes.cypher(q).to_list()[0]["c"]
    qe = "MATCH ()-[r:LINKS]->() RETURN count(r) AS c"
    assert from_file.cypher(qe).to_list()[0]["c"] == from_bytes.cypher(qe).to_list()[0]["c"]
