"""Embedding provenance + incremental re-embedding (R3).

From the operator's embedding note — "bind each vector to the text-hash + model
that produced it":

- `embed_texts(mode='changed')` re-embeds exactly the nodes whose text changed
  (or are missing), via a per-node content hash;
- the store records the embedder's model id, surfaced by `embedding_info()`;
- both the text-hashes and the model id survive a save/load round-trip, so the
  disposable-cache rebuild workflow doesn't re-embed everything after a reload.
"""

import hashlib

import pandas as pd
import pytest

import kglite


class _Embedder:
    """Deterministic stub; text → vector, with an optional model id."""

    def __init__(self, dim: int = 4, model_id: str | None = None) -> None:
        self.dimension = dim
        if model_id is not None:
            self.model_id = model_id
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(b) for b in hashlib.sha256(t.encode()).digest()[: self.dimension]] for t in texts]

    def embedded_count(self) -> int:
        return sum(len(c) for c in self.calls)


def _docs(n: int = 3) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {"id": list(range(n)), "title": [f"d{i}" for i in range(n)], "summary": [f"text {i}" for i in range(n)]}
        ),
        "Doc",
        "id",
        "title",
    )
    return g


# ── model provenance ─────────────────────────────────────────────────────────


def test_embed_texts_stamps_model_id():
    g = _docs()
    g.set_embedder(_Embedder(model_id="my/model-v1"))
    g.embed_texts("Doc", "summary", show_progress=False)
    info = g.embedding_info("Doc", "summary")
    assert info["model"] == "my/model-v1"
    assert info["dimension"] == 4
    assert info["count"] == 3
    assert info["hashed"] == 3


def test_embed_texts_without_model_id_leaves_model_none():
    g = _docs()
    g.set_embedder(_Embedder())  # no model_id attribute
    g.embed_texts("Doc", "summary", show_progress=False)
    assert g.embedding_info("Doc", "summary")["model"] is None


def test_embedding_info_none_for_missing_store():
    assert _docs().embedding_info("Doc", "summary") is None


def test_add_embeddings_store_has_no_model_or_hashes():
    g = _docs(2)
    g.add_embeddings("Doc", "summary", {0: [0.0] * 4, 1: [1.0] * 4})
    info = g.embedding_info("Doc", "summary")
    assert info["model"] is None
    assert info["hashed"] == 0  # raw vectors carry no source-text hash
    assert info["count"] == 2


# ── mode='changed' incremental re-embed ──────────────────────────────────────


def test_mode_changed_reembeds_only_changed_text():
    g = _docs(3)
    emb = _Embedder()
    g.set_embedder(emb)
    r1 = g.embed_texts("Doc", "summary", show_progress=False)
    assert r1["embedded"] == 3
    first_pass = emb.embedded_count()

    # Change exactly one node's text.
    g.cypher("MATCH (n:Doc {id: 1}) SET n.summary = 'rewritten'")

    r2 = g.embed_texts("Doc", "summary", show_progress=False, mode="changed")
    assert r2["embedded"] == 1
    assert r2["reembedded_changed"] == 1
    assert r2["skipped_existing"] == 2
    # Only one extra text was sent to the model.
    assert emb.embedded_count() == first_pass + 1


def test_mode_changed_noop_when_nothing_changed():
    g = _docs(3)
    g.set_embedder(_Embedder())
    g.embed_texts("Doc", "summary", show_progress=False)
    r = g.embed_texts("Doc", "summary", show_progress=False, mode="changed")
    assert r["embedded"] == 0
    assert r["reembedded_changed"] == 0
    assert r["skipped_existing"] == 3


def test_mode_missing_default_skips_existing():
    g = _docs(3)
    g.set_embedder(_Embedder())
    g.embed_texts("Doc", "summary", show_progress=False)
    # Add a 4th doc; default mode embeds only the new one.
    g.add_nodes(pd.DataFrame({"id": [3], "title": ["d3"], "summary": ["text 3"]}), "Doc", "id", "title")
    r = g.embed_texts("Doc", "summary", show_progress=False)  # mode='missing'
    assert r["embedded"] == 1
    assert r["skipped_existing"] == 3


def test_mode_all_reembeds_everything():
    g = _docs(3)
    g.set_embedder(_Embedder())
    g.embed_texts("Doc", "summary", show_progress=False)
    r = g.embed_texts("Doc", "summary", show_progress=False, mode="all")
    assert r["embedded"] == 3
    assert r["skipped_existing"] == 0


def test_invalid_mode_raises():
    g = _docs()
    g.set_embedder(_Embedder())
    with pytest.raises(ValueError, match="mode"):
        g.embed_texts("Doc", "summary", show_progress=False, mode="bogus")


# ── persistence: hashes + model id survive save/load ─────────────────────────


def test_text_hashes_survive_roundtrip_so_changed_is_a_noop(tmp_path):
    g = _docs(3)
    g.set_embedder(_Embedder(model_id="m/1"))
    g.embed_texts("Doc", "summary", show_progress=False)
    p = str(tmp_path / "g.kgl")
    g.save(p)

    g2 = kglite.load(p)
    # Provenance persisted.
    info = g2.embedding_info("Doc", "summary")
    assert info["model"] == "m/1"
    assert info["hashed"] == 3
    # With the hashes restored, a 'changed' pass with the same text re-embeds
    # nothing — the rebuild-from-cache workflow doesn't redo work.
    g2.set_embedder(_Embedder(model_id="m/1"))
    r = g2.embed_texts("Doc", "summary", show_progress=False, mode="changed")
    assert r["embedded"] == 0
    assert r["reembedded_changed"] == 0


def test_roundtrip_via_to_bytes_preserves_provenance():
    g = _docs(2)
    g.set_embedder(_Embedder(model_id="m/2"))
    g.embed_texts("Doc", "summary", show_progress=False)
    g2 = kglite.from_bytes(g.to_bytes())
    info = g2.embedding_info("Doc", "summary")
    assert info["model"] == "m/2"
    assert info["hashed"] == 2


# ── 0.11.1 (A): embedding_info reports the *effective* metric ─────────────────


class TestEffectiveMetricReporting:
    """embedding_info().metric must report what search actually uses — the
    explicit metric if set, else the cosine default — never a bare None for an
    existing store (operator: the None was confusing though ranking was correct)."""

    def test_embed_texts_store_reports_cosine_default(self):
        g = _docs()
        g.set_embedder(_Embedder(model_id="m"))
        g.embed_texts("Doc", "summary", show_progress=False)
        # No explicit metric was set -> effective default is cosine.
        assert g.embedding_info("Doc", "summary")["metric"] == "cosine"

    def test_set_embeddings_without_metric_reports_cosine(self):
        g = _docs()
        g.set_embeddings("Doc", "summary", {0: [1.0, 0.0], 1: [0.0, 1.0]})
        assert g.embedding_info("Doc", "summary")["metric"] == "cosine"

    def test_explicit_metric_reported_verbatim(self):
        g = _docs()
        g.set_embeddings("Doc", "summary", {0: [1.0, 0.0], 1: [0.0, 1.0]}, metric="euclidean")
        assert g.embedding_info("Doc", "summary")["metric"] == "euclidean"
        # Consistent with list_embeddings.
        row = next(r for r in g.list_embeddings() if r["text_column"] == "summary")
        assert row["metric"] == "euclidean"


# ── 0.11.1 (B): .kgle export/import carries provenance ────────────────────────


class TestKgleProvenanceRoundTrip:
    """export_embeddings/import_embeddings must carry model_id + per-node
    text_hashes + metric (KGLE v2), so a rebuild-from-.kgle pipeline keeps
    provenance and embed_texts(mode='changed') re-embeds nothing unchanged."""

    def test_roundtrip_preserves_model_metric_and_hashes(self, tmp_path):
        import os

        src = _docs(3)
        src.set_embedder(_Embedder(model_id="prov/model-v2"))
        src.embed_texts("Doc", "summary", show_progress=False)
        src.set_embeddings(  # set an explicit metric on a second store
            "Doc", "title", {0: [1.0, 0.0], 1: [0.0, 1.0], 2: [1.0, 1.0]}, metric="euclidean"
        )
        kgle = os.path.join(tmp_path, "prov.kgle")
        src.export_embeddings(kgle)

        # Fresh graph with the same nodes; import the .kgle.
        dst = _docs(3)
        dst.import_embeddings(kgle)

        info = dst.embedding_info("Doc", "summary")
        assert info["model"] == "prov/model-v2"
        assert info["metric"] == "cosine"  # effective default, carried as None
        assert info["hashed"] == 3  # per-node text hashes survived

        title_info = dst.embedding_info("Doc", "title")
        assert title_info["metric"] == "euclidean"  # explicit metric survived

    def test_imported_provenance_enables_mode_changed(self, tmp_path):
        import os

        src = _docs(3)
        src.set_embedder(_Embedder(model_id="prov/model-v2"))
        src.embed_texts("Doc", "summary", show_progress=False)
        kgle = os.path.join(tmp_path, "prov.kgle")
        src.export_embeddings(kgle)

        dst = _docs(3)
        dst.import_embeddings(kgle)
        # The whole point: with hashes carried, mode='changed' re-embeds nothing
        # (no text changed since export).
        dst.set_embedder(_Embedder(model_id="prov/model-v2"))
        res = dst.embed_texts("Doc", "summary", mode="changed", show_progress=False)
        assert res.get("reembedded_changed", 0) == 0
        assert res.get("embedded", 0) == 0
