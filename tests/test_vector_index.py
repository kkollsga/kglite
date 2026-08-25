"""HNSW vector-index lifecycle + auto-use tests.

The index is opt-in (``build_vector_index``), auto-used by ``vector_search`` /
``search_text`` for whole-corpus queries on large stores, and overridable with
``exact=True``. Later vector writes do not drop it: they are recorded and folded
in at query entry while the outstanding delta stays under ``auto_refresh_limit``
(0.16.10). Only a change to the *slot layout* — a delete's prune, ``vacuum()`` —
drops it. These tests pin recall vs the exact path, the auto-use/exact dispatch,
and the catch-up/invalidation lifecycle.
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


def _build_multi_type_graph(n=2000, d=48, seed=11, notes=1500):
    """``n`` embedded ``Doc`` nodes beside ``notes`` un-embedded ``Note`` nodes.

    The shape any realistic whole-graph search has: one embedded type among
    several. Only ``Doc`` carries the ``summary`` store.
    """
    g, emb = _build_graph(n=n, d=d, seed=seed)
    g.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(notes)),
                "title": [f"note{i}" for i in range(notes)],
                "body": [f"note text {i}" for i in range(notes)],
            }
        ),
        "Note",
        "id",
        "title",
    )
    return g, emb


def _query(d, seed=99):
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(d)]


def _ids(rows):
    return [r["id"] for r in rows]


def _vector_row(g, name="Doc.summary"):
    """The ``SHOW INDEXES`` row for a built vector index, or ``None``."""
    for row in g.cypher("SHOW INDEXES"):
        if row["name"] == name and row["type"] == "VECTOR":
            return row
    return None


def _two_type_metric_selection(node_order, store_order, metrics):
    graph = kglite.KnowledgeGraph()
    vectors = {"A": {"a": [100.0, 0.0]}, "B": {"b": [1.0, 1.0]}}
    for node_type in node_order:
        graph.add_nodes(
            pd.DataFrame({"id": [node_type.lower()], "title": [node_type], "summary": [node_type]}),
            node_type,
            "id",
            "title",
        )
    for node_type in store_order:
        graph.set_embeddings(node_type, "summary", vectors[node_type], metric=metrics[node_type])
    return graph.select("A").union(graph.select("B"))


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

    def test_vector_write_is_a_catch_up_delta_not_an_invalidation(self):
        # Neither arm of a vector write moves an existing slot, so the index
        # survives and the write becomes a delta the next query folds in.
        g, _ = _build_graph(n=500)
        g.build_vector_index("Doc", "summary")
        assert g.has_vector_index("Doc", "summary")
        g.add_embeddings("Doc", "summary", {0: [0.0] * 64})
        assert g.has_vector_index("Doc", "summary") is True
        assert _vector_row(g)["stale"] is True
        assert _vector_row(g)["delta"] == 1
        assert g.refresh_vector_index("Doc", "summary") == 1
        assert _vector_row(g)["stale"] is False

    def test_vacuum_invalidates_index(self):
        # vacuum() remaps embedding slots -> the index's slot ids go stale, so
        # it must be dropped.
        g, emb = _build_graph(n=500)
        g.build_vector_index("Doc", "summary")
        g.cypher("MATCH (d:Doc) WHERE d.id = 7 DETACH DELETE d")
        g.vacuum()
        assert g.has_vector_index("Doc", "summary") is False

    def test_delete_without_vacuum_excludes_dead_node(self):
        # A plain DELETE prunes the deleted node's vector, which tail-swaps the
        # store's slot layout and therefore drops the index. Results stay
        # correct because the exact scan takes over — the point of the test is
        # that the dead node never comes back, index or no index.
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


def _plant_cluster(g, emb, around, ids, seed=5):
    """Overwrite ``ids``' vectors with tight perturbations of ``emb[around]``.

    Gaussian fixtures have no structure, so an HNSW build's recall against
    exact is dominated by build-to-build graph variance (measured in the R9
    sweep: an ef step moves recall less than rebuilding does). A recall floor
    needs a *structured* neighborhood to recover: after planting, the exact
    top-k for ``emb[around]`` is the cluster itself, which HNSW finds at
    ~1.0 recall, giving the assertion real margin.
    """
    rng = random.Random(seed)
    cluster = {i: [v + rng.gauss(0, 0.01) for v in emb[around]] for i in ids}
    g.add_embeddings("Doc", "summary", cluster)
    emb.update(cluster)


class TestAutoUseAndRecall:
    @staticmethod
    def _exact_topk(g, q, k=10):
        return _ids(g.select("Doc").vector_search("summary", q, top_k=k, exact=True))

    def test_recall_vs_exact(self):
        # Query with a STORED vector — it has a real nearest-neighbourhood, the
        # case ANN is designed for. (A fresh *random* query has no structure in
        # high dimensions, so recall is inherently noisy — see the "benchmark on
        # real embeddings, not random vectors" note in the semantic-search guide;
        # the concurrent build adds run-to-run variance on top, which makes a
        # tight random-query threshold flaky.)
        g, emb = _build_graph(n=3000, d=64)
        q = emb[0]
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

    def test_mixed_embedded_types_use_global_exact_ranking(self):
        g = kglite.KnowledgeGraph()
        for node_type in ("A", "B"):
            g.add_nodes(
                pd.DataFrame(
                    {
                        "id": list(range(300)),
                        "title": [f"{node_type}{i}" for i in range(300)],
                        "summary": [f"text {node_type}{i}" for i in range(300)],
                    }
                ),
                node_type,
                "id",
                "title",
            )
        g.set_embeddings("A", "summary", {i: [float(2 * i), 0.0] for i in range(300)}, metric="euclidean")
        g.set_embeddings("B", "summary", {i: [float(2 * i + 1), 0.0] for i in range(300)}, metric="euclidean")

        expected = [("A", 0), ("B", 0), ("A", 1), ("B", 1), ("A", 2), ("B", 2), ("A", 3), ("B", 3), ("A", 4), ("B", 4)]
        selected = g.select("A").union(g.select("B"))
        exact = selected.vector_search("summary", [0.0, 0.0], top_k=10, metric="euclidean", exact=True)
        assert [(row["type"], row["id"]) for row in exact] == expected

        g.build_vector_index("A", "summary", metric="euclidean")
        g.build_vector_index("B", "summary", metric="euclidean")
        approximate = (
            g.select("A").union(g.select("B")).vector_search("summary", [0.0, 0.0], top_k=10, metric="euclidean")
        )
        assert [(row["type"], row["id"]) for row in approximate] == expected

        # Same guarantee for the whole-graph shape: two types carry `summary`,
        # so neither type's index may serve the search on its own — that would
        # silently drop the other type's rows. It stays a global exact ranking.
        whole_graph = g.vector_search("summary", [0.0, 0.0], top_k=10, metric="euclidean")
        assert [(row["type"], row["id"]) for row in whole_graph] == expected

    @pytest.mark.parametrize("node_order", [("A", "B"), ("B", "A")], ids=["nodes-a-b", "nodes-b-a"])
    @pytest.mark.parametrize("store_order", [("A", "B"), ("B", "A")], ids=["stores-a-b", "stores-b-a"])
    def test_mixed_stored_metrics_require_explicit_metric(self, node_order, store_order):
        selected = _two_type_metric_selection(
            node_order,
            store_order,
            {"A": "cosine", "B": "euclidean"},
        )
        with pytest.raises(ValueError, match="multiple stored metrics.*metric"):
            selected.vector_search("summary", [1.0, 0.0], top_k=2, exact=True)

    def test_same_stored_metric_ranks_multi_type_selection(self):
        selected = _two_type_metric_selection(
            ("B", "A"),
            ("A", "B"),
            {"A": "euclidean", "B": "euclidean"},
        )
        rows = selected.vector_search("summary", [1.0, 0.0], top_k=2, exact=True)
        assert [(row["type"], row["id"]) for row in rows] == [("B", "b"), ("A", "a")]

    def test_unembedded_selected_type_does_not_contribute_its_store_metric(self):
        graph = kglite.KnowledgeGraph()
        graph.add_nodes(
            pd.DataFrame({"id": ["a"], "title": ["A"], "summary": ["A"]}),
            "A",
            "id",
            "title",
        )
        graph.add_nodes(
            pd.DataFrame(
                {
                    "id": ["selected", "stored"],
                    "title": ["Selected", "Stored"],
                    "summary": ["Selected", "Stored"],
                }
            ),
            "B",
            "id",
            "title",
        )
        graph.set_embeddings("A", "summary", {"a": [100.0, 0.0]}, metric="cosine")
        graph.set_embeddings("B", "summary", {"stored": [1.0, 1.0]}, metric="euclidean")
        selected = graph.select("A").union(graph.select("B").where({"id": "selected"}))

        rows = selected.vector_search("summary", [1.0, 0.0], top_k=2, exact=True)
        assert [(row["type"], row["id"]) for row in rows] == [("A", "a")]

    def test_explicit_metric_overrides_mixed_stored_metrics(self):
        selected = _two_type_metric_selection(
            ("A", "B"),
            ("B", "A"),
            {"A": "cosine", "B": "euclidean"},
        )
        cosine = selected.vector_search("summary", [1.0, 0.0], top_k=2, metric="cosine", exact=True)
        euclidean = selected.vector_search("summary", [1.0, 0.0], top_k=2, metric="euclidean", exact=True)
        assert [(row["type"], row["id"]) for row in cosine] == [("A", "a"), ("B", "b")]
        assert [(row["type"], row["id"]) for row in euclidean] == [("B", "b"), ("A", "a")]

    def test_euclidean_index(self):
        # Stored-vector query (see test_recall_vs_exact) for a stable recall gate.
        g, emb = _build_graph(n=2000, metric="euclidean")
        q = emb[0]
        truth = set(_ids(g.select("Doc").vector_search("summary", q, top_k=10, metric="euclidean", exact=True)))
        g.build_vector_index("Doc", "summary", metric="euclidean")
        approx = set(_ids(g.select("Doc").vector_search("summary", q, top_k=10, metric="euclidean")))
        recall = len(truth.intersection(approx)) / 10.0
        assert recall >= 0.8

    def test_never_selected_whole_graph_rides_the_index(self):
        # A never-narrowed selection resolves to the whole graph, which for a
        # single embedded type is the whole store — so the index fast path is
        # still eligible (the candidate-resolution half is pinned by
        # `never_selected_candidates_are_the_whole_graph_in_index_order` in the
        # engine's vector tests; this pins the end-to-end result).
        g, emb = _build_graph(n=2000, d=48)
        _plant_cluster(g, emb, around=7, ids=range(100, 112))
        q = emb[7]
        g.build_vector_index("Doc", "summary")
        selected = _ids(g.select("Doc").vector_search("summary", q, top_k=10))
        unselected = _ids(g.vector_search("summary", q, top_k=10))
        assert unselected == selected
        exact = set(_ids(g.select("Doc").vector_search("summary", q, top_k=10, exact=True)))
        recall = len(exact.intersection(unselected)) / 10.0
        assert recall >= 0.8

    def test_whole_graph_search_on_a_multi_type_graph_rides_the_index(self):
        # A second, un-embedded type used to disqualify the whole-graph search
        # from the index: eligibility was proven by *type homogeneity*, which
        # the first Note node kills, so the search fell to a scan-only path and
        # ran orders of magnitude slower than the identical `.select('Doc')`
        # call. Only one type carries the store, so routing on store uniqueness
        # is lossless — foreign candidates miss the store and are skipped.
        # (The routing decision itself is pinned in the engine's
        # `one_store_routes_a_mixed_candidate_set_to_that_store`; this pins the
        # end-to-end answer.)
        g, emb = _build_multi_type_graph()
        _plant_cluster(g, emb, around=7, ids=range(100, 112))
        q = emb[7]
        exact_whole = _ids(g.vector_search("summary", q, top_k=10, exact=True))
        exact_selected = _ids(g.select("Doc").vector_search("summary", q, top_k=10, exact=True))
        assert exact_whole == exact_selected

        g.build_vector_index("Doc", "summary")
        whole = _ids(g.vector_search("summary", q, top_k=10))
        selected = _ids(g.select("Doc").vector_search("summary", q, top_k=10))
        # Both routes now make the same dispatch decision over the same store.
        assert whole == selected
        assert len(set(exact_whole).intersection(whole)) / 10.0 >= 0.8

    def test_whole_graph_search_raises_when_no_type_carries_the_store(self):
        # Routing on store uniqueness must not swallow the caller mistake the
        # multi-store scan raises: a selection of types that do not carry the
        # store is an error, never a silent [].
        g, _ = _build_multi_type_graph(n=600, notes=600)
        g.build_vector_index("Doc", "summary")
        with pytest.raises(ValueError, match="'Note'"):
            g.select("Note").vector_search("summary", _query(48), top_k=10)


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


class TestHnswInCypher:
    """0.11.1 (E): the Cypher fused top-k (RETURN vector_score(...) ORDER BY s
    DESC LIMIT k) auto-uses the HNSW index when one is built — same opt-in
    approximate behaviour as the fluent API. Without an index it stays exact."""

    @staticmethod
    def _g(n=1500, d=48, seed=21, metric="cosine"):
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

    Q = "MATCH (d:Doc) RETURN d.id AS id, vector_score(d,'summary_emb',$q) AS s ORDER BY s DESC LIMIT 10"

    def test_cypher_fused_topk_uses_index_with_high_recall(self):
        g, emb = self._g()
        query_ids = [0, 1, 3, 5, 8, 13, 21, 34, 55, 89]
        exact_by_query = {}
        for query_id in query_ids:
            # Stored vectors give a clear nearest-neighbour structure. Capture
            # exact truth before the index exists.
            exact = [r["id"] for r in g.cypher(self.Q, params={"q": emb[query_id]})]
            assert exact[0] == query_id
            exact_by_query[query_id] = exact

        g.build_vector_index("Doc", "summary")

        # Concurrent HNSW construction is intentionally nondeterministic. A
        # single query can occasionally land at 0.7 even while corpus recall
        # is healthy, so gate the Cypher dispatch on aggregate recall like the
        # Rust HNSW tests do rather than on one scheduling-sensitive sample.
        hits = 0
        for query_id in query_ids:
            approx = [r["id"] for r in g.cypher(self.Q, params={"q": emb[query_id]})]
            assert len(approx) == 10
            hits += len(set(exact_by_query[query_id]) & set(approx))
        recall = hits / (10.0 * len(query_ids))
        assert recall >= 0.8, f"cypher HNSW recall too low: {recall}"

    def test_cypher_scores_match_exact_scale(self):
        g, emb = self._g()
        q = emb[3]
        exact = {r["id"]: r["s"] for r in g.cypher(self.Q, params={"q": q})}
        g.build_vector_index("Doc", "summary")
        for r in g.cypher(self.Q, params={"q": q}):
            if r["id"] in exact:
                assert abs(r["s"] - exact[r["id"]]) < 1e-4

    def test_cypher_no_index_is_exact(self):
        # Without an index, the fused path is unchanged (exact) — sanity that the
        # dispatch is truly opt-in.
        g, emb = self._g(n=300)
        q = emb[0]
        assert g.has_vector_index("Doc", "summary") is False
        rows = g.cypher(self.Q, params={"q": q})
        assert rows[0]["id"] == 0  # exact top-1 is the query vector itself

    def test_cypher_where_filtered_correct(self):
        # A selective WHERE before the top-k must stay correct (membership filter
        # / exact fallback), matching the exact-over-subset result.
        g, emb = self._g(n=1500)
        q = emb[0]
        QF = (
            "MATCH (d:Doc) WHERE d.id < 40 RETURN d.id AS id, "
            "vector_score(d,'summary_emb',$q) AS s ORDER BY s DESC LIMIT 5"
        )
        exact = [r["id"] for r in g.cypher(QF, params={"q": q})]
        g.build_vector_index("Doc", "summary")
        got = [r["id"] for r in g.cypher(QF, params={"q": q})]
        assert got == exact  # filtered subset → exact result preserved

    def test_cypher_mixed_unembedded_rows_do_not_suppress_exact_fallback(self):
        embedded = 320
        unembedded = 320
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame(
                {
                    "id": list(range(embedded + unembedded)),
                    "title": [f"n{i}" for i in range(embedded + unembedded)],
                    "summary": [f"text {i}" for i in range(embedded + unembedded)],
                }
            ),
            "Doc",
            "id",
            "title",
        )
        vectors = {i: [float(i), 0.0] for i in range(embedded)}
        g.set_embeddings("Doc", "summary", vectors, metric="euclidean")
        query = [0.0, 0.0]
        mixed_query = (
            "MATCH (d:Doc) WHERE d.id >= 32 "
            "RETURN d.id AS id, vector_score(d,'summary_emb',$q,'euclidean') AS s "
            "ORDER BY s DESC LIMIT 5"
        )

        exact = [row["id"] for row in g.cypher(mixed_query, params={"q": query})]
        assert exact == [32, 33, 34, 35, 36]

        g.build_vector_index("Doc", "summary", metric="euclidean")
        approximate = [row["id"] for row in g.cypher(mixed_query, params={"q": query})]
        assert approximate == exact

    def test_cypher_bypasses_index_built_for_a_different_metric(self):
        # Above HNSW_AUTO_MIN, so the metric guard is the *only* reason the
        # index is bypassed here — a smaller corpus would pass vacuously on
        # the size gate instead.
        n = 512
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame(
                {
                    "id": list(range(n)),
                    "title": [f"n{i}" for i in range(n)],
                    "summary": [f"text {i}" for i in range(n)],
                }
            ),
            "Doc",
            "id",
            "title",
        )
        vectors = {i: ([1.0, 0.0] if i == 0 else [100.0, 100.0] if i == 1 else [1.0, 0.1]) for i in range(n)}
        g.set_embeddings("Doc", "summary", vectors, metric="cosine")
        query = (
            "MATCH (d:Doc) "
            "RETURN d.id AS id, vector_score(d,'summary_emb',$q,'dot_product') AS s "
            "ORDER BY s DESC LIMIT 1"
        )

        exact = g.cypher(query, params={"q": [1.0, 0.0]})
        assert [row["id"] for row in exact] == [1]
        g.build_vector_index("Doc", "summary", metric="cosine")
        automatic = g.cypher(query, params={"q": [1.0, 0.0]})
        assert [(row["id"], row["s"]) for row in automatic] == [(row["id"], row["s"]) for row in exact]

    def test_text_score_uses_index(self):
        # text_score rewrites to vector_score, so it rides the same fast path.
        import hashlib

        class _Emb:
            dimension = 16
            model_id = "fake"

            def embed(self, texts):
                return [[float(b) for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]

        g = kglite.KnowledgeGraph()
        n = 800
        g.add_nodes(
            pd.DataFrame(
                {"id": list(range(n)), "title": [f"n{i}" for i in range(n)], "summary": [f"doc {i}" for i in range(n)]}
            ),
            "Doc",
            "id",
            "title",
        )
        g.set_embedder(_Emb())
        g.embed_texts("Doc", "summary", show_progress=False)
        TQ = "MATCH (d:Doc) RETURN d.id AS id, text_score(d,'summary','doc 5') AS s ORDER BY s DESC LIMIT 10"
        exact = [r["id"] for r in g.cypher(TQ)]
        g.build_vector_index("Doc", "summary")
        approx = [r["id"] for r in g.cypher(TQ)]
        assert len(approx) == 10
        recall = len(set(exact) & set(approx)) / 10.0
        assert recall >= 0.8


class TestFreshnessUx:
    """0.16.10: vector indexes adopt the shared catch-up framework — the same
    stale/delta reporting and threshold-bounded inline refresh the BM25 lane
    uses, with one deliberate difference: a stale *vector* index falls back to
    the exact scan, which is the oracle the approximate path is measured
    against, so staleness costs latency and never accuracy."""

    def test_show_indexes_lists_a_built_vector_index(self):
        g, _ = _build_graph(n=500)
        assert _vector_row(g) is None, "vectors alone are not an installed index"
        g.build_vector_index("Doc", "summary")
        row = _vector_row(g)
        assert row is not None
        assert row["type"] == "VECTOR"
        assert row["entityType"] == "NODE"
        assert row["labelsOrTypes"] == ["Doc"]
        assert row["properties"] == ["summary"], "keyed on the source column"
        assert row["stale"] is False
        assert row["delta"] == 0
        assert row["unembedded"] == 0
        g.drop_vector_index("Doc", "summary")
        assert _vector_row(g) is None

    def test_unembedded_nodes_are_counted_and_never_embedded(self):
        g, _ = _build_graph(n=500)
        g.build_vector_index("Doc", "summary")
        g.cypher("CREATE (:Doc {id: 9001, title: 'fresh', summary: 'never embedded'})")
        row = _vector_row(g)
        assert row["unembedded"] == 1
        assert row["delta"] == 0, "a node with no vector is not a catch-up delta"
        assert row["stale"] is False
        # Running a query must not change that: catch-up indexes, it never embeds.
        g.select("Doc").vector_search("summary", _query(64), top_k=5)
        assert _vector_row(g)["unembedded"] == 1

    def test_a_query_catches_up_an_under_limit_delta(self):
        g, emb = _build_graph(n=600)
        g.build_vector_index("Doc", "summary")
        g.add_embeddings("Doc", "summary", {600: [0.0] * 64, 0: [1.0] * 64})
        # id 600 matches no node, so only the in-place replacement lands.
        assert _vector_row(g)["stale"] is True
        g.select("Doc").vector_search("summary", _query(64), top_k=10)
        assert _vector_row(g)["stale"] is False, "the query folded the delta in"
        assert _vector_row(g)["delta"] == 0

    def test_an_over_limit_delta_stays_stale_and_serves_the_exact_answer(self):
        g, emb = _build_graph(n=600)
        g.build_vector_index("Doc", "summary", auto_refresh_limit=2)
        rewritten = {i: [float(i % 7), 1.0] + [0.0] * 62 for i in range(10)}
        g.add_embeddings("Doc", "summary", rewritten)
        assert _vector_row(g)["delta"] == 10

        # Doc 3's rewritten vector is the query itself, so the exact answer is
        # unambiguous — and it is the answer we must get, from a scan the stale
        # index stepped aside for.
        target = rewritten[3]
        rows = g.select("Doc").vector_search("summary", target, top_k=1)
        assert _vector_row(g)["stale"] is True, "over the ceiling, nothing is folded in"
        assert rows[0]["id"] == 3
        assert rows[0]["score"] == pytest.approx(1.0, abs=1e-6)

    def test_refresh_is_available_explicitly(self):
        g, _ = _build_graph(n=600)
        g.build_vector_index("Doc", "summary", auto_refresh_limit=1)
        g.add_embeddings("Doc", "summary", {i: [0.5] * 64 for i in range(5)})
        assert g.refresh_vector_index("Doc", "summary") == 5
        assert _vector_row(g)["stale"] is False
        assert g.refresh_vector_index("Doc", "summary") == 0
        assert g.refresh_vector_index("Doc", "nope") == 0

    def test_a_partly_covered_index_survives_save_and_load(self):
        g, _ = _build_graph(n=600)
        g.build_vector_index("Doc", "summary", auto_refresh_limit=2)
        g.add_embeddings("Doc", "summary", {i: [0.25] * 64 for i in range(5)})
        assert _vector_row(g)["delta"] == 5
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.kgl")
            g.save(p)
            g2 = kglite.load(p)
        assert g2.has_vector_index("Doc", "summary") is True
        assert _vector_row(g2)["delta"] == 5, "the outstanding delta rides along"
        assert g2.refresh_vector_index("Doc", "summary") == 5
        assert _vector_row(g2)["stale"] is False

    def test_the_fused_cypher_top_k_regains_the_index_path_after_catch_up(self):
        g, _ = _build_graph(n=800)
        g.build_vector_index("Doc", "summary")
        g.add_embeddings("Doc", "summary", {5: [0.75] * 64})
        assert _vector_row(g)["stale"] is True
        rows = g.cypher(
            "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', $q) AS s ORDER BY s DESC LIMIT 5",
            params={"q": _query(64)},
        )
        assert len(rows) == 5
        assert _vector_row(g)["stale"] is False, "the fused top-k's coverage seam is where the catch-up happens"

    def test_bulk_node_ingest_never_dirties_a_vector_index(self):
        # The design promise behind putting freshness in *store*-slot space:
        # creating nodes cannot make a vector index stale, because a node with
        # no vector is not a document. Bulk ingest therefore pays nothing.
        g, _ = _build_graph(n=500)
        g.build_vector_index("Doc", "summary")
        g.add_nodes(
            pd.DataFrame({"id": list(range(5000, 9000)), "title": [f"x{i}" for i in range(4000)]}),
            "Doc",
            "id",
            "title",
        )
        row = _vector_row(g)
        assert row["stale"] is False
        assert row["delta"] == 0
        assert row["unembedded"] == 4000
