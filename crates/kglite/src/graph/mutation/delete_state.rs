//! Node-deletion's above-storage half.
//!
//! `GraphWrite::remove_node` takes the node out of the backend. Four pieces
//! of state that a deleted node owns live one layer *above* the backend, on
//! `DirGraph` itself, so nothing inside storage can see them go — and each is
//! readable only while the node still exists. They are removed here, in the
//! same loop as the backend removal and in the order that keeps each read
//! valid.
//!
//! Extracted from `maintain.rs` to keep that file under the god-file LoC
//! ceiling. The index/bucket sweeps that follow a deletion stay there, in
//! `detach_delete_nodes`, which is this module's only caller.

use std::collections::HashSet;

use petgraph::graph::NodeIndex;

use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphWrite;

/// Drop `node_idx`'s vector from every embedding store that holds one, and
/// journal each removal so a statement rollback can put it back.
///
/// **Why deletion must reach this map.** `EmbeddingStore` is keyed by the
/// global `NodeIndex`, and `StableDiGraph` hands a freed index straight to
/// the next node created. A vector left behind is therefore not merely stale
/// bookkeeping: the next node to land on that slot — of *any* type, embedded
/// or not — inherits it and comes back as a full-similarity top hit from
/// `vector_search`, on both the scan and the HNSW path.
///
/// Costs one hash probe per store, and stores are per `(node_type, property)`
/// — a handful, independent of graph size. The `is_empty` guard keeps the
/// overwhelmingly common un-embedded graph at zero cost per deleted node.
fn prune_doomed_embeddings(graph: &mut DirGraph, node_idx: NodeIndex) {
    if graph.embeddings.is_empty() {
        return;
    }
    let node = node_idx.index();
    let removed: Vec<((String, String), _)> = graph
        .embeddings
        .iter_mut()
        .filter_map(|(key, store)| Some((key.clone(), store.remove_embedding(node)?)))
        .collect();
    if removed.is_empty() {
        return;
    }
    if let Some(journal) = graph.graph.undo_journal_mut() {
        for (store_key, prior) in removed {
            journal.note_embedding_removed(store_key, node, prior);
        }
    }
}

/// Drop `node_idx`'s document from every text index that holds one.
///
/// **Why deletion must reach this map**, and why it is not journalled the way
/// [`prune_doomed_embeddings`] is. A text index addresses documents *by*
/// `NodeIndex`, and `StableDiGraph` hands a freed index straight to the next
/// node created, so a document left behind is inherited: the new node — of any
/// type, of any content — scores as the deleted one's text. That is a wrong
/// answer, so the prune is unconditional.
///
/// A rolled-back delete therefore leaves the node unindexed rather than
/// restored. That is the *safe* direction of the two: an unindexed row scores
/// as absent, which is exactly what a node created after the build does, and
/// what an explicit rebuild fixes. Restoring the document would need its text
/// re-tokenized, which the graph can do at rebuild time and the journal cannot
/// do at all without carrying a second copy of every deleted document.
///
/// Costs one hash probe per index, and indexes are per `(node_type, property)`
/// — a handful, independent of graph size. The `is_empty` guard keeps the
/// overwhelmingly common un-indexed graph at zero cost per deleted node.
fn prune_doomed_text_docs(graph: &mut DirGraph, node_idx: NodeIndex) {
    if graph.text_indexes.is_empty() {
        return;
    }
    for store in graph.text_indexes.values_mut() {
        store.remove_node(node_idx);
    }
}

/// Remove each doomed node from storage, carrying the four pieces of state
/// that live *above* storage and so cannot be recovered afterwards.
///
/// All four are read while the node still exists and are lost the moment it
/// does not, which is why they are here rather than in `detach_delete_nodes`'
/// sweeps:
///
/// - **The change-capture before-image's labels.** The capture wrapper reads
///   the node's properties and title as it removes it, but secondary labels
///   live in `DirGraph::secondary_label_index`, one layer above the backend.
///   A delete is the one event whose only informative half is `before`, so an
///   image missing its labels is the whole loss.
/// - **The dropped timeseries entry.** `timeseries_store` is O(V) and so is
///   deliberately not part of the checkpoint's schema clone; statement
///   rollback recovers it from the undo journal instead.
/// - **The node's embeddings.** Same ownership story as the timeseries, with a
///   sharper failure mode: the freed `NodeIndex` is reused, so a vector left
///   behind is inherited rather than merely orphaned. See
///   [`prune_doomed_embeddings`].
/// - **The node's text-index documents.** The same inheritance hazard, one
///   layer further up: a BM25 document is addressed by `NodeIndex` directly.
///   See [`prune_doomed_text_docs`].
pub(super) fn remove_doomed_nodes(graph: &mut DirGraph, nodes_to_delete: &HashSet<NodeIndex>) {
    let captures_before = graph.graph.captures_before_images();
    for &node_idx in nodes_to_delete {
        let doomed_labels = captures_before.then(|| graph.secondary_label_names(node_idx));
        GraphWrite::remove_node(&mut graph.graph, node_idx);
        if let Some(labels) = doomed_labels {
            graph.graph.backfill_node_before_labels(node_idx, labels);
        }
        prune_doomed_embeddings(graph, node_idx);
        prune_doomed_text_docs(graph, node_idx);
        let Some(prior) = graph.timeseries_store.remove(&node_idx.index()) else {
            continue;
        };
        if let Some(journal) = graph.graph.undo_journal_mut() {
            journal.note_timeseries_removed(node_idx.index(), prior);
        }
    }
}
