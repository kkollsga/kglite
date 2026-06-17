"""Embedding-store integrity — operator feedback B4/B5 (2026-06-17).

- `add_embeddings` / `embed_texts` reject a dimension that mismatches the
  existing store (mixing dims silently corrupts similarity search).
- `embedding_dim(node_type, text_column)` exposes the store's dimension so a
  model change is detectable without bookkeeping.
- `embed_texts(replace=True)` is deterministic across a dimension change
  (rebuilds a fresh store at the new dimension).
"""

import hashlib

import pandas as pd
import pytest

import kglite


class _Stub:
    """Deterministic embedder of a given dimension (no network/model)."""

    def __init__(self, dim: int) -> None:
        self.dimension = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(b) for b in hashlib.sha256(t.encode()).digest()[: self.dimension]] for t in texts]


def _docs() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1, 2, 3], "title": ["a", "b", "c"], "summary": ["alpha", "beta", "gamma"]}),
        "Doc",
        "id",
        "title",
    )
    return g


def test_embedding_dim_accessor() -> None:
    g = _docs()
    assert g.embedding_dim("Doc", "summary") is None  # no store yet
    g.add_embeddings("Doc", "summary", {1: [0.1, 0.2, 0.3, 0.4]})
    assert g.embedding_dim("Doc", "summary") == 4
    assert g.embedding_dim("Doc", "missing") is None


def test_add_embeddings_rejects_dimension_mismatch() -> None:
    g = _docs()
    g.add_embeddings("Doc", "summary", {1: [0.0] * 4})
    with pytest.raises(ValueError, match="dimension"):
        g.add_embeddings("Doc", "summary", {2: [0.0] * 8})


def test_embed_texts_upsert_rejects_dimension_change() -> None:
    """embed_texts(replace=False) into a store of a different dimension must
    error with a recipe, not silently mix dims."""
    g = _docs()
    g.set_embedder(_Stub(4))
    g.embed_texts("Doc", "summary", show_progress=False)
    assert g.embedding_dim("Doc", "summary") == 4

    g.set_embedder(_Stub(8))  # model swap
    with pytest.raises(ValueError, match="replace=True"):
        g.embed_texts("Doc", "summary", show_progress=False)
    # The store is untouched by the rejected upsert.
    assert g.embedding_dim("Doc", "summary") == 4


def test_embed_texts_replace_rebuilds_at_new_dimension() -> None:
    """replace=True is deterministic across a dimension change — it rebuilds a
    fresh store at the new model's dimension (B5)."""
    g = _docs()
    g.set_embedder(_Stub(4))
    g.embed_texts("Doc", "summary", show_progress=False)
    assert g.embedding_dim("Doc", "summary") == 4

    g.set_embedder(_Stub(8))
    g.embed_texts("Doc", "summary", replace=True, show_progress=False)
    assert g.embedding_dim("Doc", "summary") == 8
