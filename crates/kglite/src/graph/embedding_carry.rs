//! Cross-graph embedding carry — `DirGraph::copy_embeddings_from`.
//!
//! Extracted from `dir_graph.rs` to keep that file under the god-file LoC
//! ceiling. The method lives on `DirGraph` (an `impl` block here is fine —
//! same crate/module tree); it's the core behind the Python
//! `KnowledgeGraph.copy_embeddings_from`.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

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
    /// actually changed). A vector whose source id has no matching node here is
    /// skipped. Returns `(stores_copied, vectors_copied, vectors_skipped)`.
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

            for &src_idx in src_store.node_to_slot.keys() {
                let Some(src_node) = src.graph.node_weight(NodeIndex::new(src_idx)) else {
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
                    None => vectors_skipped += 1,
                }
            }

            self.embeddings.insert((node_type, prop), dst_store);
            stores_copied += 1;
        }

        (stores_copied, vectors_copied, vectors_skipped)
    }

    /// Remap every embedding store's internal node indices through `old_to_new`
    /// after a `vacuum()` rebuilds the graph with contiguous indices. Drops
    /// vectors whose node was deleted (absent from the map), compacts the data
    /// buffer to the surviving slots, and resyncs the cached-norm column.
    /// Extracted from `vacuum()` to keep `dir_graph.rs` under the god-file
    /// ceiling.
    pub(crate) fn remap_embedding_slots(&mut self, old_to_new: &HashMap<NodeIndex, NodeIndex>) {
        for store in self.embeddings.values_mut() {
            let mut new_node_to_slot = HashMap::with_capacity(store.node_to_slot.len());
            let mut new_slot_to_node = Vec::with_capacity(store.slot_to_node.len());
            let mut new_data = Vec::with_capacity(store.data.len());

            for (&old_node_raw, &slot) in &store.node_to_slot {
                let old_idx = NodeIndex::new(old_node_raw);
                if let Some(&new_idx) = old_to_new.get(&old_idx) {
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
            store.data = new_data;
            // Slots were remapped wholesale; resync the cached-norm column.
            store.rebuild_norms();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::schema::NodeData;
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
}
