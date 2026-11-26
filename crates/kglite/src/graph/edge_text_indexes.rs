//! Relationship BM25 text indexes — the node lane's twin, keyed by edge slot.
//!
//! A child of [`crate::graph::text_indexes`] so it reuses [`TextIndexStore`]
//! whole: the same inverted index, the same generation counter, the same
//! [`IndexFreshness`] tracker. What differs is only what a slot *is* and how a
//! document is read, so those are the only things this module adds:
//!
//! - **Slot = `EdgeIndex::index()`.** The same identity convention as nodes,
//!   and the same reason: no second mapping to fall out of step. Freshness
//!   runs over the graph's *edge* bound.
//! - **Document = the relationship's own property**, read by interned key with
//!   no alias resolution (relationships have no id/title alias). A string, or a
//!   list of strings/nulls joined by the shared node rule.
//! - **Deletion prunes at the one edge-removal choke point**
//!   (`edge_embeddings::remove_edge_with_embeddings`), and journals
//!   `UndoEntry::EdgeTextDocPruned` so a rolled-back delete re-marks the slot,
//!   exactly as `prune_doomed_text_docs` does for nodes.
//! - **Lifecycle runs inside Cypher** (`db.relationship_text_index.*`), so unlike the
//!   node lane a build or drop can be followed by a failing clause:
//!   both journal `UndoEntry::EdgeTextIndexReplaced`, moving (not cloning) the
//!   prior store into the journal.
//!
//! Memory and mapped only; disk refuses with the node lane's reason.
//! WAL-recorded like the node text index (`MutationOp::SetTextIndex`): replay
//! rebuilds it from the recovered rows.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

use petgraph::graph::EdgeIndex;

use super::{
    index_key, join_text_list, rebuild_beats_folding, TextIndexRead, TextIndexReport,
    TextIndexStore, BATCH_MIN_CHANGES,
};
use crate::datatypes::values::Value;
use crate::graph::algorithms::text_index::bm25::PreparedQuery;
use crate::graph::algorithms::text_index::TextIndex;
use crate::graph::dir_graph::DirGraph;
use crate::graph::index_freshness::IndexFreshness;
use crate::graph::schema::{EdgeData, InternedKey};
use crate::graph::storage::{GraphRead, RecordRef};

#[inline]
fn edge_slot(edge: EdgeIndex) -> u32 {
    edge.index() as u32
}

/// The graph's edge-slot bound, as the document slot space sees it.
#[inline]
fn edge_bound(graph: &DirGraph) -> u32 {
    GraphRead::edge_bound(&graph.graph) as u32
}

/// The document of a record held by handle: borrowed when the record is lent,
/// owned when it was built for the read.
fn record_document(edge: RecordRef<'_, EdgeData>, field: InternedKey) -> Option<Cow<'_, str>> {
    match edge {
        RecordRef::Borrowed(edge) => edge_document(edge, field),
        RecordRef::Owned(edge) => edge_document(&edge, field).map(|d| Cow::Owned(d.into_owned())),
    }
}

fn edge_document(edge: &EdgeData, field: InternedKey) -> Option<Cow<'_, str>> {
    let value = edge
        .properties
        .iter()
        .find(|(key, _)| *key == field)
        .map(|(_, value)| value)?;
    match value {
        Value::String(text) => Some(Cow::Borrowed(text.as_str())),
        Value::List(items) => join_text_list(items).map(Cow::Owned),
        _ => None,
    }
}

/// The document `slot` holds for an index over `(type_key, field)`: `None`
/// when the slot is vacant, holds another type, or carries no text.
fn slot_document(
    graph: &DirGraph,
    slot: u32,
    type_key: InternedKey,
    field: InternedKey,
) -> Option<Cow<'_, str>> {
    let edge = graph.graph.edge_weight(EdgeIndex::new(slot as usize))?;
    if edge.connection_type != type_key {
        return None;
    }
    record_document(edge, field)
}

/// Build a fresh index over every relationship of `type_key`. Returns the
/// index, how many relationships of the type exist, and how many yielded no
/// document.
fn build_over_type(
    graph: &DirGraph,
    type_key: InternedKey,
    field: InternedKey,
) -> (TextIndex, usize, usize) {
    let mut members = 0usize;
    let mut skipped = 0usize;
    let index = TextIndex::build(graph.graph.edge_indices().filter_map(|edge| {
        let data = graph.graph.edge_weight(edge)?;
        if data.connection_type != type_key {
            return None;
        }
        members += 1;
        match record_document(data, field) {
            Some(text) => Some((edge_slot(edge), text)),
            None => {
                skipped += 1;
                None
            }
        }
    }));
    (index, members, skipped)
}

impl TextIndexRead<'_> {
    /// BM25 score of one relationship, or `None` when it has no document.
    pub fn score_relationship(&self, edge: EdgeIndex, query: &PreparedQuery) -> Option<f64> {
        let slot = edge_slot(edge);
        self.0.contains_doc(slot).then(|| self.0.score(slot, query))
    }
}

impl TextIndexStore {
    /// Documents the next [`Self::refresh_relationships`] would re-read — the edge
    /// twin of [`Self::delta_size`].
    pub fn relationship_delta_size(&self, graph: &DirGraph) -> usize {
        self.freshness.delta_size(edge_bound(graph))
    }

    /// Whether the graph's relationships have moved past what this index
    /// covers.
    pub fn relationship_is_stale(&self, graph: &DirGraph) -> bool {
        self.freshness.is_stale(edge_bound(graph))
    }

    /// Whether the outstanding relationship delta is within the inline-refresh
    /// ceiling.
    pub fn relationship_can_auto_refresh(&self, graph: &DirGraph) -> bool {
        self.freshness.within_limit(edge_bound(graph))
    }

    /// Drop a relationship's document. See [`Self::remove_node`] for why every
    /// deletion must reach this.
    pub(crate) fn remove_edge(&mut self, edge: EdgeIndex) -> bool {
        self.index
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .remove_doc(edge_slot(edge))
    }

    /// Mark a relationship slot for re-reading — the undo of a pruning delete.
    pub(crate) fn note_edge_slot_changed(&self, edge: EdgeIndex) {
        self.freshness.note_changed(edge_slot(edge));
    }

    /// Fold every outstanding relationship change into the index. The edge
    /// twin of [`Self::refresh`]: the same lock order, the same fold / batch /
    /// rebuild arms and crossover, reading each slot's relationship instead of
    /// its node.
    pub fn refresh_relationships(&self, graph: &DirGraph, rel_type: &str) -> usize {
        if graph.read_only {
            return 0;
        }
        let mut index = self.index.write().unwrap_or_else(|e| e.into_inner());
        let Some(delta) = self.freshness.take_delta(edge_bound(graph)) else {
            return 0;
        };
        let field = InternedKey::from_str(&self.resolved_field);
        let type_key = InternedKey::from_str(rel_type);
        let changes = delta
            .slots()
            .filter(|slot| {
                graph
                    .graph
                    .edge_weight(EdgeIndex::new(*slot as usize))
                    .is_some_and(|edge| edge.connection_type == type_key)
                    || index.contains_doc(*slot)
            })
            .count();
        let seen = if rebuild_beats_folding(changes, index.total_docs()) {
            let (rebuilt, members, skipped) = build_over_type(graph, type_key, field);
            *index = rebuilt;
            self.skipped.store(skipped, Ordering::Relaxed);
            members
        } else if changes < BATCH_MIN_CHANGES {
            let mut seen = 0usize;
            for slot in delta.slots() {
                seen += 1;
                match slot_document(graph, slot, type_key, field) {
                    Some(text) => index.add_doc(slot, text.as_ref()),
                    None => {
                        index.remove_doc(slot);
                    }
                }
            }
            seen
        } else {
            index.replace_batch(
                delta
                    .slots()
                    .map(|slot| (slot, slot_document(graph, slot, type_key, field))),
            )
        };
        // Bumped for every claimed delta; see `refresh` for why.
        self.generation.fetch_add(1, Ordering::Release);
        debug_assert!(
            index.validate().is_ok(),
            "a refreshed relationship text index must satisfy its own invariants: {:?}",
            index.validate()
        );
        seen
    }
}

fn store_of(
    index: TextIndex,
    freshness: IndexFreshness,
    property: &str,
    skipped: usize,
) -> TextIndexStore {
    TextIndexStore {
        index: RwLock::new(index),
        generation: AtomicU64::new(0),
        freshness,
        resolved_field: property.to_string(),
        skipped: AtomicUsize::new(skipped),
    }
}

/// Install `store` under `key` (or remove the key when `None`), journalling
/// the store it displaces so a failed statement puts it back.
fn replace_store(graph: &mut DirGraph, key: (String, String), store: Option<TextIndexStore>) {
    let prior = match store {
        Some(store) => graph.edge_text_indexes.insert(key.clone(), store),
        None => graph.edge_text_indexes.remove(&key),
    };
    if let Some(journal) = graph.graph.undo_journal_mut() {
        journal.note_edge_text_index_replaced(key, prior);
    }
}

/// Build (or rebuild) a BM25 index over `property` for every relationship of
/// `rel_type`. The relationship twin of
/// [`build_text_index`](super::build_text_index): same document rule, same
/// "a type whose every member yields nothing is a misspelling" error, same
/// kept-unless-given `auto_refresh_limit`.
pub(crate) fn build_edge_text_index(
    graph: &mut DirGraph,
    rel_type: &str,
    property: &str,
    auto_refresh_limit: Option<usize>,
) -> Result<TextIndexReport, String> {
    let report = build_edge_text_index_structure(graph, rel_type, property, auto_refresh_limit)?;
    // The resolved ceiling, for the reason `build_text_index` gives.
    let limit =
        edge_text_index_store(graph, rel_type, property).map(TextIndexStore::auto_refresh_limit);
    graph.note_declaration(crate::graph::wal::MutationOp::SetTextIndex {
        entity: crate::graph::wal::TextIndexEntity::Relationship,
        type_name: rel_type.to_string(),
        property: property.to_string(),
        auto_refresh_limit: limit,
        present: true,
    });
    Ok(report)
}

/// The build half of [`build_edge_text_index`], without the log — the WAL
/// replay entry, for the reason `build_text_index_structure` gives.
pub(crate) fn build_edge_text_index_structure(
    graph: &mut DirGraph,
    rel_type: &str,
    property: &str,
    auto_refresh_limit: Option<usize>,
) -> Result<TextIndexReport, String> {
    if GraphRead::is_disk(&graph.graph) {
        return Err(format!(
            "db.relationship_text_index.build({{type: '{rel_type}', text_column: '{property}'}}) is not \
             supported on a disk-backed graph: the BM25 index is heap-resident, and building one \
             over a graph sized for the disk backend is the memory cliff that backend exists to \
             avoid. Use the default (in-memory) or 'mapped' storage mode."
        ));
    }
    if !graph.has_connection_type(rel_type) {
        return Err(format!(
            "Unknown relationship type '{rel_type}'. db.relationship_text_index.build indexes one \
             relationship type's property; CALL db.relationshipTypes() lists the types that exist."
        ));
    }
    let field = InternedKey::from_str(property);
    let type_key = InternedKey::from_str(rel_type);
    let (index, members, skipped) = build_over_type(graph, type_key, field);
    if index.total_docs() == 0 && members > 0 {
        return Err(format!(
            "No '{rel_type}' relationship carries text or a string/null list for '{property}' — \
             all {members} were absent or not text documents, so there is nothing to index. \
             Check the spelling, and note that numbers and lists containing non-text members \
             are not indexable."
        ));
    }
    let key = index_key(rel_type, property);
    let limit = auto_refresh_limit.or_else(|| {
        graph
            .edge_text_indexes
            .get(&key)
            .map(TextIndexStore::auto_refresh_limit)
    });
    let store = store_of(
        index,
        IndexFreshness::covering(edge_bound(graph), limit),
        property,
        skipped,
    );
    let report = TextIndexReport {
        indexed: store.documents(),
        skipped,
        terms: store.terms(),
    };
    replace_store(graph, key, Some(store));
    graph.bump_version();
    Ok(report)
}

/// Fold every outstanding change into the relationship index over
/// `(rel_type, property)`. `None` when no such index exists.
pub(crate) fn refresh_edge_text_index(
    graph: &DirGraph,
    rel_type: &str,
    property: &str,
) -> Option<usize> {
    let store = edge_text_index_store(graph, rel_type, property)?;
    Some(store.refresh_relationships(graph, rel_type))
}

/// Drop the relationship text index over `(rel_type, property)`. Returns
/// whether one existed.
pub(crate) fn drop_edge_text_index(graph: &mut DirGraph, rel_type: &str, property: &str) -> bool {
    let removed = drop_edge_text_index_structure(graph, rel_type, property);
    if removed {
        graph.forget_index_names_for(
            crate::graph::constraints::EntityKind::Relationship,
            rel_type,
            &[property.to_string()],
            &[crate::graph::dir_graph::index_names::IndexClass::Fulltext],
        );
        graph.note_declaration(crate::graph::wal::MutationOp::SetTextIndex {
            entity: crate::graph::wal::TextIndexEntity::Relationship,
            type_name: rel_type.to_string(),
            property: property.to_string(),
            auto_refresh_limit: None,
            present: false,
        });
    }
    removed
}

/// The drop half of [`drop_edge_text_index`], without the log.
pub(crate) fn drop_edge_text_index_structure(
    graph: &mut DirGraph,
    rel_type: &str,
    property: &str,
) -> bool {
    let key = index_key(rel_type, property);
    if !graph.edge_text_indexes.contains_key(&key) {
        return false;
    }
    replace_store(graph, key, None);
    graph.bump_version();
    true
}

/// The relationship text index over `(rel_type, property)`, if built. Scans
/// rather than hashing, for the reason
/// [`text_index_store`](super::text_index_store) gives.
pub(crate) fn edge_text_index_store<'a>(
    graph: &'a DirGraph,
    rel_type: &str,
    property: &str,
) -> Option<&'a TextIndexStore> {
    graph
        .edge_text_indexes
        .iter()
        .find(|((indexed_type, indexed_property), _)| {
            indexed_type == rel_type && indexed_property == property
        })
        .map(|(_, store)| store)
}

/// Every relationship text index, sorted by `(rel_type, property)`.
pub(crate) fn list_edge_text_indexes(graph: &DirGraph) -> Vec<(&str, &str, &TextIndexStore)> {
    let mut out: Vec<(&str, &str, &TextIndexStore)> = graph
        .edge_text_indexes
        .iter()
        .map(|((rel_type, property), store)| (rel_type.as_str(), property.as_str(), store))
        .collect();
    out.sort_unstable_by_key(|(rel_type, property, _)| (*rel_type, *property));
    out
}

/// Install an index restored from a `.kgl` section. Persistence-only; the
/// decoder validates the payload first.
pub(crate) fn attach_persisted_edge_text_index(
    graph: &mut DirGraph,
    rel_type: &str,
    property: &str,
    index: TextIndex,
    freshness: IndexFreshness,
    skipped: usize,
) {
    graph.edge_text_indexes.insert(
        index_key(rel_type, property),
        store_of(index, freshness, property, skipped),
    );
}

/// A relationship of `rel_type` was created at `edge` — the recycled-slot
/// check. Reached only through `index_freshness::write_hooks`, past its gate.
pub(crate) fn note_edge_created(graph: &DirGraph, edge: EdgeIndex, rel_type: InternedKey) {
    let slot = edge_slot(edge);
    for ((indexed_type, _), store) in &graph.edge_text_indexes {
        store
            .freshness
            .note_created(slot, InternedKey::from_str(indexed_type) == rel_type);
    }
}

/// A property of the relationship at `edge` was written. `field: None` is a
/// caller that replaced several properties at once.
pub(crate) fn note_edge_property_written(
    graph: &DirGraph,
    edge: EdgeIndex,
    rel_type: InternedKey,
    field: Option<InternedKey>,
) {
    let slot = edge_slot(edge);
    for ((indexed_type, property), store) in &graph.edge_text_indexes {
        if InternedKey::from_str(indexed_type) != rel_type {
            continue;
        }
        if field.is_none_or(|written| written == InternedKey::from_str(property)) {
            store.freshness.note_changed(slot);
        }
    }
}

/// Prune `edge`'s documents before its slot is freed, journalling each prune
/// so a rollback re-marks the slot (the relationship twin of
/// `delete_state::prune_doomed_text_docs`).
pub(crate) fn prune_edge_text_docs(graph: &mut DirGraph, edge: EdgeIndex) {
    if graph.edge_text_indexes.is_empty() {
        return;
    }
    let pruned: Vec<(String, String)> = graph
        .edge_text_indexes
        .iter_mut()
        .filter_map(|(key, store)| store.remove_edge(edge).then(|| key.clone()))
        .collect();
    let Some(journal) = graph.graph.undo_journal_mut() else {
        return;
    };
    for store_key in pruned {
        journal.note_edge_text_doc_pruned(store_key, edge.index());
    }
}

#[cfg(test)]
#[path = "edge_text_indexes_tests.rs"]
mod tests;
