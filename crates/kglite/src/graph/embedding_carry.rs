//! Cross-graph embedding carry — `DirGraph::copy_embeddings_from`.
//!
//! Extracted from `dir_graph.rs` to keep that file under the god-file LoC
//! ceiling. The method lives on `DirGraph` (an `impl` block here is fine —
//! same crate/module tree); it's the core behind the Python
//! `KnowledgeGraph.copy_embeddings_from`.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::graph::dir_graph::node_remap::NodeRemap;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::EmbeddingStore;
use crate::graph::storage::GraphRead;

impl DirGraph {
    /// Copy every embedding store from `src` into this graph, matching
    /// vectors by **node id** (not internal index — those differ across a
    /// rebuild). For the dominant embedding workflow: rebuild a fresh graph
    /// from a source of truth on each load, then `new.copy_embeddings_from(old)`
    /// to carry the vectors forward in one call — instead of the manual
    /// `embeddings()` snapshot → `add_embeddings()` restore dance. Carries the
    /// store's `dimension`, `metric`, `model_id`, and per-node `text_hashes`
    /// (so a subsequent `embed_texts(mode='changed')` re-embeds only what
    /// actually changed).
    ///
    /// A vector whose source id has no matching node here gets one more
    /// chance: a node carrying a `chunk_hash` — the property a vault's
    /// `structure:` block derives (VAULT.md §7.1) — matches the node of the
    /// same label and the same hash, when exactly one node on each side
    /// carries it. Derived ids move whenever the prose around them does, so
    /// without this a rebuild would re-embed a chunk that merely shifted down
    /// a section. The fallback decides *which* old node this is, never whether
    /// to re-embed: the carried text hash still drives `embed_texts`'
    /// changed-mode pass. Anything still unmatched is skipped. Returns
    /// `(stores_copied, vectors_copied, vectors_skipped)`.
    pub fn copy_embeddings_from(&mut self, src: &DirGraph) -> (usize, usize, usize) {
        let mut stores_copied = 0usize;
        let mut vectors_copied = 0usize;
        let mut vectors_skipped = 0usize;

        // Snapshot the store keys + node types first so we can build each
        // type's id index (a `&mut self` op) before the immutable id lookups.
        let store_keys: Vec<(String, String)> = src.embeddings.keys().cloned().collect();

        for (node_type, prop) in store_keys {
            let Some(src_store) = src.embeddings.get(&(node_type.clone(), prop.clone())) else {
                continue;
            };
            // Build this type's id index once so lookups are O(1).
            self.build_id_index(&node_type);

            let mut dst_store = EmbeddingStore::new(src_store.dimension);
            dst_store.metric = src_store.metric.clone();
            dst_store.model_id = src_store.model_id.clone();

            // Ids first, in one pass; whatever they missed goes to the
            // hash fallback below, which needs both graphs' whole
            // `chunk_hash` census and is therefore built once, not per node.
            let mut unmatched: Vec<usize> = Vec::new();
            for &src_idx in src_store.node_to_slot.keys() {
                let Some(src_node) = src.graph.node_view(NodeIndex::new(src_idx)) else {
                    vectors_skipped += 1;
                    continue;
                };
                let id = src_node.id().into_owned();
                let Some(embedding) = src_store.get_embedding(src_idx) else {
                    vectors_skipped += 1;
                    continue;
                };
                match self.lookup_by_id_readonly(&node_type, &id) {
                    Some(dst_idx) => {
                        dst_store.set_embedding(dst_idx.index(), embedding);
                        if let Some(&h) = src_store.text_hashes.get(&src_idx) {
                            dst_store.set_text_hash(dst_idx.index(), h);
                        }
                        vectors_copied += 1;
                    }
                    None => unmatched.push(src_idx),
                }
            }
            let (carried, skipped) =
                carry_by_chunk_hash(self, src, &node_type, &unmatched, src_store, &mut dst_store);
            vectors_copied += carried;
            vectors_skipped += skipped;

            // A carried store keeps the scan precision it was built with.
            if let Some(quant) = src_store.quant_store() {
                let _ = dst_store.quantize(quant.precision(), Some(quant.oversample()));
            }
            let text_column = crate::graph::embeddings::text_column_of(&prop)
                .unwrap_or(&prop)
                .to_string();
            self.set_embedding_store(&node_type, &text_column, dst_store);
            stores_copied += 1;
        }

        if stores_copied > 0 {
            self.bump_version();
        }

        (stores_copied, vectors_copied, vectors_skipped)
    }

    /// Remap every embedding store's internal node indices through `old_to_new`
    /// after a `vacuum()` rebuilds the graph with contiguous indices. Drops
    /// vectors whose node was deleted (absent from the map), compacts the data
    /// buffer to the surviving slots, and resyncs the cached-norm column.
    /// Extracted from `vacuum()` to keep `dir_graph.rs` under the god-file
    /// ceiling.
    pub(crate) fn remap_embedding_slots(&mut self, old_to_new: &NodeRemap) {
        for store in self.embeddings.values_mut() {
            let mut new_node_to_slot = HashMap::with_capacity(store.node_to_slot.len());
            let mut new_slot_to_node = Vec::with_capacity(store.slot_to_node.len());
            let mut new_data = Vec::with_capacity(store.data.len());

            for (&old_node_raw, &slot) in &store.node_to_slot {
                let old_idx = NodeIndex::new(old_node_raw);
                if let Some(new_idx) = old_to_new.get(old_idx) {
                    let new_slot = new_slot_to_node.len();
                    new_node_to_slot.insert(new_idx.index(), new_slot);
                    new_slot_to_node.push(new_idx.index());
                    let start = slot * store.dimension;
                    let end = start + store.dimension;
                    new_data.extend_from_slice(&store.data[start..end]);
                }
                // Deleted nodes (not in old_to_new) are dropped.
            }

            store.node_to_slot = new_node_to_slot;
            store.slot_to_node = new_slot_to_node;
            store.data = new_data.into();
            // Slots were remapped wholesale; resync the cached-norm column and
            // drop any HNSW index (its slot ids are now stale — rebuild on demand).
            store.rebuild_norms();
            store.requantize();
            store.invalidate_index();
        }
    }
}

/// Carry the vectors the id pass missed, by `chunk_hash` (VAULT.md §7.1, §12).
///
/// Both sides must be unambiguous: exactly one old node and exactly one new
/// node of this label may carry the hash. Two chunks with identical text are a
/// real case — a repeated boilerplate paragraph — and guessing between them
/// would hand a node a vector computed from another node's context.
fn carry_by_chunk_hash(
    dst: &DirGraph,
    src: &DirGraph,
    node_type: &str,
    unmatched: &[usize],
    src_store: &EmbeddingStore,
    dst_store: &mut EmbeddingStore,
) -> (usize, usize) {
    if unmatched.is_empty() {
        return (0, 0);
    }
    let dst_by_hash = chunk_hash_index(dst, node_type);
    if dst_by_hash.is_empty() {
        return (0, unmatched.len());
    }
    let src_by_hash = chunk_hash_index(src, node_type);
    let mut copied = 0usize;
    let mut skipped = 0usize;
    for &src_idx in unmatched {
        let target = chunk_hash_of(src, src_idx)
            // Ambiguous on the old side: `None` is the marker this index uses
            // for a hash more than one node carried.
            .filter(|hash| matches!(src_by_hash.get(hash), Some(Some(_))))
            .and_then(|hash| dst_by_hash.get(&hash).copied().flatten())
            // A node the id pass already filled is not up for a second
            // vector, whichever pass would have written it.
            .filter(|idx| !dst_store.node_to_slot.contains_key(&idx.index()));
        match (target, src_store.get_embedding(src_idx)) {
            (Some(idx), Some(embedding)) => {
                dst_store.set_embedding(idx.index(), embedding);
                if let Some(&hash) = src_store.text_hashes.get(&src_idx) {
                    dst_store.set_text_hash(idx.index(), hash);
                }
                copied += 1;
            }
            _ => skipped += 1,
        }
    }
    (copied, skipped)
}

/// `chunk_hash` → the one node of `node_type` carrying it, or `None` where two
/// or more do.
fn chunk_hash_index(graph: &DirGraph, node_type: &str) -> HashMap<String, Option<NodeIndex>> {
    let mut out: HashMap<String, Option<NodeIndex>> = HashMap::new();
    let Some(nodes) = graph.type_indices.get(node_type) else {
        return out;
    };
    for idx in nodes.iter() {
        if let Some(hash) = chunk_hash_of(graph, idx.index()) {
            out.entry(hash)
                .and_modify(|found| *found = None)
                .or_insert(Some(idx));
        }
    }
    out
}

fn chunk_hash_of(graph: &DirGraph, idx: usize) -> Option<String> {
    match graph
        .graph
        .node_view(NodeIndex::new(idx))?
        .get_property_value(CHUNK_HASH_PROPERTY)
    {
        Some(crate::datatypes::Value::String(hash)) if !hash.is_empty() => Some(hash),
        _ => None,
    }
}

/// The property a derived node's text hash lives under (VAULT.md §7.1). Named
/// once here because the carry is the only reader outside the vault builder.
const CHUNK_HASH_PROPERTY: &str = "chunk_hash";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::schema::NodeData;
    use crate::graph::session::Session;
    use crate::graph::storage::GraphWrite;
    use std::collections::HashMap;

    fn graph_with_docs(ids: &[i64]) -> DirGraph {
        let mut g = DirGraph::new();
        for &id in ids {
            let nd = NodeData::new(
                Value::Int64(id),
                Value::String(format!("d{id}")),
                "Doc".to_string(),
                HashMap::new(),
                &mut g.interner,
            );
            let idx = GraphWrite::add_node(&mut g.graph, nd);
            g.type_indices.entry_or_default("Doc".to_string()).push(idx);
        }
        g.build_id_index("Doc");
        g
    }

    /// Vectors carry by node id (not internal index), with dimension, model_id
    /// and text-hashes preserved; ids absent in the destination are skipped.
    #[test]
    fn copies_vectors_by_id_with_provenance() {
        let mut src = graph_with_docs(&[1, 2, 3]);
        let mut store = EmbeddingStore::new(2);
        store.model_id = Some("m/1".to_string());
        for &id in &[1i64, 2, 3] {
            let idx = src.lookup_by_id_readonly("Doc", &Value::Int64(id)).unwrap();
            store.set_embedding(idx.index(), &[id as f32, 0.0]);
            store.set_text_hash(idx.index(), EmbeddingStore::text_hash(&format!("t{id}")));
        }
        src.embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);

        // Destination is a fresh rebuild missing id 3.
        let mut dst = graph_with_docs(&[1, 2]);
        let (stores, vectors, skipped) = dst.copy_embeddings_from(&src);
        assert_eq!((stores, vectors, skipped), (1, 2, 1));

        let dst_store = dst
            .embeddings
            .get(&("Doc".to_string(), "summary_emb".to_string()))
            .unwrap();
        assert_eq!(dst_store.dimension, 2);
        assert_eq!(dst_store.model_id.as_deref(), Some("m/1"));
        assert_eq!(dst_store.len(), 2);
        assert_eq!(dst_store.text_hashes.len(), 2);
        // The carried vector landed on the dst node with the matching id.
        let dst_idx = dst.lookup_by_id_readonly("Doc", &Value::Int64(2)).unwrap();
        assert_eq!(
            dst_store.get_embedding(dst_idx.index()),
            Some(&[2.0f32, 0.0][..])
        );
    }

    /// `Session::transact` publishes a working graph only when its version
    /// changed. Carrying a store through that public path must therefore mark
    /// the mutation, or the closure reports success while its new store is
    /// discarded with the transaction fork.
    #[test]
    fn copy_embeddings_from_publishes_inside_session_transaction() {
        let mut src = graph_with_docs(&[1]);
        let src_idx = src.lookup_by_id_readonly("Doc", &Value::Int64(1)).unwrap();
        let mut store = EmbeddingStore::new(2);
        store.set_embedding(src_idx.index(), &[1.0, 2.0]);
        src.embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);

        let session = Session::new(graph_with_docs(&[1]));
        let before = session.version();
        let copied = session
            .transact::<_, ()>(|working| Ok(working.copy_embeddings_from(&src)))
            .unwrap();

        assert_eq!(copied, (1, 1, 0));
        assert_eq!(session.version(), before + 1);
        let snapshot = session.snapshot();
        let dst_idx = snapshot
            .lookup_by_id_readonly("Doc", &Value::Int64(1))
            .unwrap();
        assert_eq!(
            snapshot.embeddings[&("Doc".to_string(), "summary_emb".to_string())]
                .get_embedding(dst_idx.index()),
            Some(&[1.0, 2.0][..])
        );
    }

    #[test]
    fn copying_no_stores_is_a_true_no_op() {
        let src = graph_with_docs(&[1]);
        let mut dst = graph_with_docs(&[1]);
        let before = dst.version();

        assert_eq!(dst.copy_embeddings_from(&src), (0, 0, 0));
        assert_eq!(dst.version(), before);
    }

    /// A `vacuum` moves every surviving node to a new index, and this is the
    /// one consumer that has to follow it: each vector must end up on the node
    /// that owned it, and a deleted node's vector must be dropped rather than
    /// re-attached to whichever node inherited its slot.
    ///
    /// Nothing pinned this before — the suite only asserted that a vacuum
    /// *invalidates the HNSW index*, which stays true however wrongly the
    /// vectors are remapped.
    #[test]
    fn a_vacuum_moves_each_vector_to_its_own_nodes_new_index() {
        let mut g = graph_with_docs(&[1, 2, 3, 4]);
        let mut store = EmbeddingStore::new(2);
        for &id in &[1i64, 2, 3, 4] {
            let idx = g.lookup_by_id_readonly("Doc", &Value::Int64(id)).unwrap();
            store.set_embedding(idx.index(), &[id as f32, 0.0]);
        }
        g.embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);

        // Delete the second node, leaving a tombstone for the vacuum to close.
        let victim = g.lookup_by_id_readonly("Doc", &Value::Int64(2)).unwrap();
        g.graph.remove_node(victim);
        g.type_indices
            .entry_or_default("Doc".to_string())
            .retain(|idx| *idx != victim);
        g.id_indices.clear();

        let remap = g.vacuum();
        assert_eq!(remap.len(), 3, "the rebuild must actually have happened");

        let store = g
            .embeddings
            .get(&("Doc".to_string(), "summary_emb".to_string()))
            .unwrap();
        assert_eq!(store.len(), 3, "the deleted node's vector must be dropped");
        for &id in &[1i64, 3, 4] {
            let idx = g.lookup_by_id_readonly("Doc", &Value::Int64(id)).unwrap();
            assert_eq!(
                store.get_embedding(idx.index()),
                Some(&[id as f32, 0.0][..]),
                "doc {id} kept a vector that is not its own"
            );
        }
    }

    /// A graph of chunks: `(id, chunk_hash)` pairs, the shape a vault's
    /// `structure:` block derives (VAULT.md §7.1).
    fn graph_with_chunks(chunks: &[(&str, &str)]) -> DirGraph {
        let mut g = DirGraph::new();
        for (id, hash) in chunks {
            let mut props = HashMap::new();
            props.insert("chunk_hash".to_string(), Value::String((*hash).to_string()));
            let nd = NodeData::new(
                Value::String((*id).to_string()),
                Value::String((*id).to_string()),
                "Chunk".to_string(),
                props,
                &mut g.interner,
            );
            let idx = GraphWrite::add_node(&mut g.graph, nd);
            g.type_indices
                .entry_or_default("Chunk".to_string())
                .push(idx);
        }
        g.build_id_index("Chunk");
        g
    }

    fn store_over(g: &DirGraph, vectors: &[(&str, f32)]) -> EmbeddingStore {
        let mut store = EmbeddingStore::new(2);
        for (id, value) in vectors {
            let idx = g
                .lookup_by_id_readonly("Chunk", &Value::String((*id).to_string()))
                .unwrap();
            store.set_embedding(idx.index(), &[*value, 0.0]);
            store.set_text_hash(idx.index(), EmbeddingStore::text_hash(&format!("t{value}")));
        }
        store
    }

    fn vector_of(g: &DirGraph, id: &str) -> Option<Vec<f32>> {
        let store = g
            .embeddings
            .get(&("Chunk".to_string(), "text_emb".to_string()))?;
        let idx = g.lookup_by_id_readonly("Chunk", &Value::String(id.to_string()))?;
        store.get_embedding(idx.index()).map(<[f32]>::to_vec)
    }

    /// A heading rename moves every chunk id under it — the ids embed the
    /// heading text — so an id match finds nothing at all. The `chunk_hash`
    /// carries across the chunks that only moved; the one whose *text* changed
    /// hashes differently and is left to be re-embedded (VAULT.md §12).
    #[test]
    fn a_chunk_that_only_moved_keeps_its_vector() {
        let mut old =
            graph_with_chunks(&[("n#A~chunk1", "hash-intro"), ("n#A~chunk2", "hash-body")]);
        let store = store_over(&old, &[("n#A~chunk1", 1.0), ("n#A~chunk2", 2.0)]);
        old.embeddings
            .insert(("Chunk".to_string(), "text_emb".to_string()), store);

        // The rebuild: `## A` was renamed `## B`, and the body was edited.
        let mut new = graph_with_chunks(&[
            ("n#B~chunk1", "hash-intro"),
            ("n#B~chunk2", "hash-body-edited"),
        ]);
        let (stores, copied, skipped) = new.copy_embeddings_from(&old);
        assert_eq!((stores, copied, skipped), (1, 1, 1));
        assert_eq!(
            vector_of(&new, "n#B~chunk1"),
            Some(vec![1.0, 0.0]),
            "the intro is the same chunk under a new id"
        );
        assert_eq!(
            vector_of(&new, "n#B~chunk2"),
            None,
            "an edited chunk hashes differently and is re-embedded"
        );
        let store = new
            .embeddings
            .get(&("Chunk".to_string(), "text_emb".to_string()))
            .unwrap();
        let idx = new
            .lookup_by_id_readonly("Chunk", &Value::String("n#B~chunk1".to_string()))
            .unwrap();
        assert!(
            store.text_hashes.contains_key(&idx.index()),
            "the carried text hash is what keeps `mode=changed` from re-embedding it"
        );
    }

    /// The id is still the primary key (VAULT.md §12), and a chunk id the
    /// rebuild *reused* for different prose takes the old node's vector and
    /// **its stored text hash** — which is what makes the changed-mode pass
    /// re-embed it. The fallback is a second rung, not a replacement.
    #[test]
    fn a_reused_id_still_wins_over_the_hash() {
        let mut old = graph_with_chunks(&[("n#A~chunk1", "hash-intro")]);
        let store = store_over(&old, &[("n#A~chunk1", 1.0)]);
        old.embeddings
            .insert(("Chunk".to_string(), "text_emb".to_string()), store);
        let mut new = graph_with_chunks(&[
            ("n#A~chunk1", "hash-inserted"),
            ("n#A~chunk2", "hash-intro"),
        ]);
        let (_, copied, skipped) = new.copy_embeddings_from(&old);
        assert_eq!((copied, skipped), (1, 0));
        assert_eq!(vector_of(&new, "n#A~chunk1"), Some(vec![1.0, 0.0]));
        assert_eq!(vector_of(&new, "n#A~chunk2"), None);
    }

    /// Two chunks of identical text are a real case — repeated boilerplate —
    /// and guessing between them would hand a node another node's vector.
    #[test]
    fn an_ambiguous_hash_carries_nothing() {
        let mut old = graph_with_chunks(&[("n#A~chunk1", "same"), ("n#B~chunk1", "same")]);
        let store = store_over(&old, &[("n#A~chunk1", 1.0), ("n#B~chunk1", 2.0)]);
        old.embeddings
            .insert(("Chunk".to_string(), "text_emb".to_string()), store);
        let mut new = graph_with_chunks(&[("n#A~chunk2", "same"), ("n#B~chunk2", "same")]);
        let (_, copied, skipped) = new.copy_embeddings_from(&old);
        assert_eq!((copied, skipped), (0, 2));
    }

    /// The fallback decides *which* old node this is; it never overwrites a
    /// node the id pass already answered for.
    #[test]
    fn an_id_match_wins_over_a_hash_match() {
        let mut old = graph_with_chunks(&[("n#A~chunk1", "hash-a"), ("n#A~chunk2", "hash-b")]);
        let store = store_over(&old, &[("n#A~chunk1", 1.0), ("n#A~chunk2", 2.0)]);
        old.embeddings
            .insert(("Chunk".to_string(), "text_emb".to_string()), store);
        // `~chunk1` kept its id but now holds what `~chunk2` used to say.
        let mut new = graph_with_chunks(&[("n#A~chunk1", "hash-b")]);
        let (_, copied, _) = new.copy_embeddings_from(&old);
        assert_eq!(copied, 1);
        assert_eq!(
            vector_of(&new, "n#A~chunk1"),
            Some(vec![1.0, 0.0]),
            "the id match stands; the hash match finds the node already filled"
        );
    }

    /// A graph with no `chunk_hash` anywhere behaves exactly as it did before
    /// the fallback existed.
    #[test]
    fn without_chunk_hashes_nothing_changes() {
        let mut src = graph_with_docs(&[1, 2]);
        let mut store = EmbeddingStore::new(2);
        for &id in &[1i64, 2] {
            let idx = src.lookup_by_id_readonly("Doc", &Value::Int64(id)).unwrap();
            store.set_embedding(idx.index(), &[id as f32, 0.0]);
        }
        src.embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);
        let mut dst = graph_with_docs(&[1]);
        assert_eq!(dst.copy_embeddings_from(&src), (1, 1, 1));
    }
}
