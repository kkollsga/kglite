"""Regression: set_embeddings must see nodes loaded from disk plus nodes
just added via add_nodes in the same session.

Bug shape (kglite-docs 2026-05-28): a 50-LOC repro showed that after
`kglite.load(path)` of a graph with 140 Chunk embeddings, calling
`add_nodes(one new Chunk)` then `set_embeddings(merged_dict)` reported
`skipped: 140` for ids that demonstrably existed as Chunk nodes. Root
cause: BatchProcessor wrote the new chunk's id into `id_indices`
incrementally, creating a partial entry that subsequent `build_id_index`
calls short-circuited on. The save+load roundtrip worked around it by
clearing `id_indices` entirely.
"""

from pathlib import Path

import pandas as pd

import kglite


def _make_corpus(n: int, dim: int = 4) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    ids = [f"chunk_{i:03d}" for i in range(n)]
    df = pd.DataFrame(
        {
            "id": ids,
            "title": ids,
            "text": [f"text body for {i}" for i in range(n)],
        }
    )
    embeddings = {cid: [float(i + k) for k in range(dim)] for i, cid in enumerate(ids)}
    return df, embeddings


def test_set_embeddings_sees_disk_and_session_nodes(tmp_path: Path):
    """The kglite-docs 50-LOC repro: load 140 chunks, add one, set merged dict."""
    df, original_embeddings = _make_corpus(140)

    # Build + save the corpus on first ingest.
    g = kglite.KnowledgeGraph()
    g.add_nodes(df, "Chunk", "id", "title")
    g.set_embeddings("Chunk", "text", original_embeddings)
    save_path = tmp_path / "corpus.kgl"
    g.save(str(save_path))

    # Second ingest: reload, add one more, merge + set.
    g2 = kglite.load(str(save_path))
    pre = g2.embeddings("Chunk", "text")
    assert len(pre) == 140

    new_id = "chunk_140"
    new_df = pd.DataFrame([{"id": new_id, "title": new_id, "text": "the 141st chunk"}])
    g2.add_nodes(new_df, "Chunk", "id", "title")

    merged = dict(pre)
    merged[new_id] = [0.1, 0.2, 0.3, 0.4]
    result = g2.set_embeddings("Chunk", "text", merged)

    assert result["skipped"] == 0, (
        f"set_embeddings dropped {result['skipped']} ids that exist as Chunk nodes — "
        f"id_indices was stale after add_nodes."
    )
    assert result["embeddings_stored"] == 141
    assert len(g2.embeddings("Chunk", "text")) == 141


def test_set_embeddings_after_repeated_add_nodes(tmp_path: Path):
    """Repeated load → add_nodes → set_embeddings cycles must not drift."""
    df, original_embeddings = _make_corpus(50)
    g = kglite.KnowledgeGraph()
    g.add_nodes(df, "Chunk", "id", "title")
    g.set_embeddings("Chunk", "text", original_embeddings)
    save_path = tmp_path / "corpus.kgl"
    g.save(str(save_path))

    expected = 50
    for batch_idx in range(3):
        g2 = kglite.load(str(save_path))
        pre = g2.embeddings("Chunk", "text")
        assert len(pre) == expected, f"Round {batch_idx}: load() saw {len(pre)} embeddings, expected {expected}"

        new_ids = [f"chunk_b{batch_idx}_{i}" for i in range(5)]
        new_df = pd.DataFrame({"id": new_ids, "title": new_ids, "text": [f"body {n}" for n in new_ids]})
        g2.add_nodes(new_df, "Chunk", "id", "title")

        merged = dict(pre)
        for nid in new_ids:
            merged[nid] = [0.0, 1.0, 2.0, 3.0]
        result = g2.set_embeddings("Chunk", "text", merged)
        assert result["skipped"] == 0
        assert result["embeddings_stored"] == expected + 5

        g2.save(str(save_path))
        expected += 5


def test_lookup_by_id_after_add_nodes_into_loaded_graph(tmp_path: Path):
    """Direct id_indices invariant: lookup after add_nodes must find both
    the pre-existing rows and the newly-added one."""
    df, _ = _make_corpus(20)
    g = kglite.KnowledgeGraph()
    g.add_nodes(df, "Chunk", "id", "title")
    save_path = tmp_path / "small.kgl"
    g.save(str(save_path))

    g2 = kglite.load(str(save_path))
    new_df = pd.DataFrame([{"id": "chunk_new", "title": "chunk_new", "text": "fresh"}])
    g2.add_nodes(new_df, "Chunk", "id", "title")

    # Both an old id and the new id should be reachable via cypher MATCH-by-id.
    old_hit = g2.cypher(
        "MATCH (c:Chunk {id: $id}) RETURN c.id AS id",
        params={"id": "chunk_005"},
    ).to_list()
    new_hit = g2.cypher(
        "MATCH (c:Chunk {id: $id}) RETURN c.id AS id",
        params={"id": "chunk_new"},
    ).to_list()
    assert old_hit and old_hit[0]["id"] == "chunk_005"
    assert new_hit and new_hit[0]["id"] == "chunk_new"
