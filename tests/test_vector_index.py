"""HNSW vector-index lifecycle + auto-use tests.

The index is opt-in (``build_vector_index``), auto-used by ``vector_search`` /
``search_text`` for whole-corpus queries on large stores, overridable with
``exact=True``, and dropped automatically whenever the store's vectors change.
These tests pin recall vs the exact path, the auto-use/exact dispatch, and the
invalidation lifecycle.
"""

import os
import random
import tempfile

import pandas as pd
import pytest

import kglite


def _build_graph(n=3000, d=64, seed=11, metric="cosine"):
    rng = random.Random(seed)
    rows = {
        "id": list(range(n)),
        "title": [f"n{i}" for i in range(n)],
        "summary": [f"text {i}" for i in range(n)],
    }
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame(rows), "Doc", "id", "title")
    emb = {i: [rng.gauss(0, 1) for _ in range(d)] for i in range(n)}
    g.set_embeddings("Doc", "summary", emb, metric=metric)
    return g, emb


def _query(d, seed=99):
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(d)]


def _ids(rows):
    return [r["id"] for r in rows]


class TestIndexLifecycle:
    def test_build_drop_has(self):
        g, _ = _build_graph(n=500)
        assert g.has_vector_index("Doc", "summary") is False
        info = g.build_vector_index("Doc", "summary")
        assert info["indexed"] == 500
        assert info["metric"] == "cosine"
        assert g.has_vector_index("Doc", "summary") is True
        assert g.drop_vector_index("Doc", "summary") is True
        assert g.has_vector_index("Doc", "summary") is False
        # Dropping again is a no-op.
        assert g.drop_vector_index("Doc", "summary") is False

    def test_build_missing_store_raises(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [1], "title": ["a"]}), "Doc", "id", "title")
        with pytest.raises(ValueError):
            g.build_vector_index("Doc", "summary")

    def test_poincare_rejected(self):
        g, _ = _build_graph(n=300, metric="poincare")
        with pytest.raises(ValueError):
            g.build_vector_index("Doc", "summary", metric="poincare")

    def test_mutation_invalidates_index(self):
        g, _ = _build_graph(n=500)
        g.build_vector_index("Doc", "summary")
        assert g.has_vector_index("Doc", "summary")
        # add_embeddings touches the store -> index drops.
        g.add_embeddings("Doc", "summary", {0: [0.0] * 64})
        assert g.has_vector_index("Doc", "summary") is False

    def test_vacuum_invalidates_index(self):
        # vacuum() remaps embedding slots -> the index's slot ids go stale, so
        # it must be dropped.
        g, emb = _build_graph(n=500)
        g.build_vector_index("Doc", "summary")
        g.cypher("MATCH (d:Doc) WHERE d.id = 7 DETACH DELETE d")
        g.vacuum()
        assert g.has_vector_index("Doc", "summary") is False

    def test_delete_without_vacuum_excludes_dead_node(self):
        # A plain DELETE keeps the index (slots unchanged); the dead node is
        # excluded from results via the selection membership filter, so results
        # stay correct even with a stale-but-valid index.
        g, _ = _build_graph(n=2000)
        g.build_vector_index("Doc", "summary")
        q = _query(64)
        # Find a node that would otherwise rank; delete it, then confirm it
        # never appears in results.
        top = _ids(g.select("Doc").vector_search("summary", q, top_k=5))
        victim = top[0]
        g.cypher("MATCH (d:Doc) WHERE d.id = $v DETACH DELETE d", params={"v": victim})
        after = _ids(g.select("Doc").vector_search("summary", q, top_k=5))
        assert victim not in after


class TestAutoUseAndRecall:
    @staticmethod
    def _exact_topk(g, q, k=10):
        return _ids(g.select("Doc").vector_search("summary", q, top_k=k, exact=True))

    def test_recall_vs_exact(self):
        g, _ = _build_graph(n=3000, d=64)
        q = _query(64)
        truth = set(self._exact_topk(g, q, k=10))
        g.build_vector_index("Doc", "summary")
        approx = _ids(g.select("Doc").vector_search("summary", q, top_k=10))
        recall = len(truth.intersection(approx)) / 10.0
        assert recall >= 0.8, f"recall too low: {recall}"

    def test_exact_flag_forces_bruteforce(self):
        # With exact=True the index is bypassed -> identical to no-index result.
        g, _ = _build_graph(n=2000)
        q = _query(64)
        before = self._exact_topk(g, q, k=10)
        g.build_vector_index("Doc", "summary")
        after = self._exact_topk(g, q, k=10)
        assert before == after

    def test_scores_on_same_scale_as_exact(self):
        # The ANN step only narrows which nodes are scored; surviving scores
        # match the exact cosine value for the same node.
        g, emb = _build_graph(n=2000, d=48)
        q = _query(48)
        exact = {r["id"]: r["score"] for r in g.select("Doc").vector_search("summary", q, top_k=10, exact=True)}
        g.build_vector_index("Doc", "summary")
        for r in g.select("Doc").vector_search("summary", q, top_k=10):
            if r["id"] in exact:
                assert abs(r["score"] - exact[r["id"]]) < 1e-4

    def test_filtered_subset_still_correct(self):
        # A selective filter falls back to an exact scan -> exact results even
        # though an index exists.
        g, _ = _build_graph(n=2000)
        g.build_vector_index("Doc", "summary")
        q = _query(64)
        # Restrict to a small id range; results must equal the exact scan over
        # that same subset.
        sub = g.select("Doc").where({"id": {"<": 50}})
        got = _ids(sub.vector_search("summary", q, top_k=5))
        sub2 = g.select("Doc").where({"id": {"<": 50}})
        exact = _ids(sub2.vector_search("summary", q, top_k=5, exact=True))
        assert got == exact

    def test_euclidean_index(self):
        g, _ = _build_graph(n=2000, metric="euclidean")
        q = _query(64)
        truth = set(_ids(g.select("Doc").vector_search("summary", q, top_k=10, metric="euclidean", exact=True)))
        g.build_vector_index("Doc", "summary", metric="euclidean")
        approx = set(_ids(g.select("Doc").vector_search("summary", q, top_k=10, metric="euclidean")))
        recall = len(truth.intersection(approx)) / 10.0
        assert recall >= 0.8


class TestIndexRoundTrip:
    def test_index_persists_across_save_load(self):
        # V4: the HNSW index rides in the .kgl, so a reloaded graph keeps it and
        # the approximate results are identical (same topology + vectors).
        g, _ = _build_graph(n=2000)
        g.build_vector_index("Doc", "summary")
        q = _query(64)
        before = [(r["id"], round(r["score"], 6)) for r in g.select("Doc").vector_search("summary", q, top_k=10)]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.kgl")
            g.save(p)
            g2 = kglite.load(p)
        assert g2.has_vector_index("Doc", "summary") is True
        after = [(r["id"], round(r["score"], 6)) for r in g2.select("Doc").vector_search("summary", q, top_k=10)]
        assert before == after

    def test_no_index_no_section(self):
        # Embeddings but no index round-trips fine (no index after).
        g, _ = _build_graph(n=500)
        q = _query(64)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.kgl")
            g.save(p)
            g2 = kglite.load(p)
        assert g2.has_vector_index("Doc", "summary") is False
        assert len(g2.select("Doc").vector_search("summary", q, top_k=5)) == 5

    def test_to_bytes_from_bytes_preserves_index(self):
        g, _ = _build_graph(n=1500)
        g.build_vector_index("Doc", "summary")
        blob = g.to_bytes()
        g2 = kglite.from_bytes(blob)
        assert g2.has_vector_index("Doc", "summary") is True
