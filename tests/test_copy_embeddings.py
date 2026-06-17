"""copy_embeddings_from — id-keyed cross-graph vector carry (Phase B).

Operator embedding note #2: the disposable-cache workflow rebuilds a fresh graph
from source on each load, so vectors must be carried across. This verifies a
single `new.copy_embeddings_from(old)` carries vectors (matched by node id) plus
provenance, and composes with `embed_texts(mode='changed')` to fill only the new
nodes.
"""

import hashlib

import pandas as pd

import kglite


class _Embedder:
    def __init__(self, dim: int = 4, model_id: str = "m/1") -> None:
        self.dimension = dim
        self.model_id = model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(b) for b in hashlib.sha256(t.encode()).digest()[: self.dimension]] for t in texts]


def _old() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1, 2, 3], "title": ["a", "b", "c"], "summary": ["x", "y", "z"]}),
        "Doc",
        "id",
        "title",
    )
    g.set_embedder(_Embedder())
    g.embed_texts("Doc", "summary", show_progress=False)
    return g


def _fresh_rebuild(ids=(1, 2, 3)) -> kglite.KnowledgeGraph:
    """A fresh graph rebuilt from 'source' — same ids, NO vectors yet."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": list(ids), "title": [f"t{i}" for i in ids], "summary": [f"s{i}" for i in ids]}),
        "Doc",
        "id",
        "title",
    )
    return g


def test_copy_carries_vectors_by_id():
    old = _old()
    new = _fresh_rebuild()
    assert new.embedding_dim("Doc", "summary") is None  # fresh, no vectors

    report = new.copy_embeddings_from(old)
    assert report["stores_copied"] == 1
    assert report["vectors_copied"] == 3
    assert report["vectors_skipped"] == 0
    assert new.embedding_dim("Doc", "summary") == 4


def test_copy_carries_provenance_and_hashes():
    old = _old()
    new = _fresh_rebuild()
    new.copy_embeddings_from(old)
    info = new.embedding_info("Doc", "summary")
    assert info["model"] == "m/1"
    assert info["hashed"] == 3  # text hashes carried


def test_copy_then_changed_only_embeds_new_nodes():
    """The headline workflow: carry old vectors, then embed_texts(mode='changed')
    fills ONLY the genuinely-new node — not the carried ones."""
    old = _old()
    # Fresh rebuild has an extra doc (id 4) that old never embedded.
    new = _fresh_rebuild(ids=(1, 2, 3, 4))
    # Make the carried nodes' text identical to what 'old' embedded, so their
    # hashes match and they're not re-embedded.
    new.cypher("MATCH (n:Doc {id: 1}) SET n.summary = 'x'")
    new.cypher("MATCH (n:Doc {id: 2}) SET n.summary = 'y'")
    new.cypher("MATCH (n:Doc {id: 3}) SET n.summary = 'z'")

    new.copy_embeddings_from(old)
    new.set_embedder(_Embedder())
    r = new.embed_texts("Doc", "summary", show_progress=False, mode="changed")
    # Only id 4 (never carried) gets embedded; 1/2/3 are unchanged → skipped.
    assert r["embedded"] == 1
    assert r["skipped_existing"] == 3


def test_copy_skips_ids_with_no_matching_node():
    old = _old()  # ids 1,2,3
    new = _fresh_rebuild(ids=(1, 2))  # id 3 absent here
    report = new.copy_embeddings_from(old)
    assert report["vectors_copied"] == 2
    assert report["vectors_skipped"] == 1


def test_copy_survives_save_load_roundtrip(tmp_path):
    old = _old()
    new = _fresh_rebuild()
    new.copy_embeddings_from(old)
    p = str(tmp_path / "new.kgl")
    new.save(p)
    reloaded = kglite.load(p)
    info = reloaded.embedding_info("Doc", "summary")
    assert info["count"] == 3
    assert info["model"] == "m/1"
    assert info["hashed"] == 3
