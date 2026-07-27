//! Statement-scoped undo journal — the inverse-op buffer that replaces the
//! whole-graph rollback clone.
//!
//! ## Why a journal and not a clone
//!
//! Statement atomicity used to be bought with `DirGraph::fork_transaction()`:
//! a deep clone of the whole graph taken *before* every mutating Cypher
//! statement, discarded on success and swapped back on failure. Correct, but
//! it makes the cost of writing one property proportional to the size of the
//! whole graph — a `Compact` `NodeData` clone deep-copies every
//! `Value::String` it holds — which is disqualifying for a primary store.
//!
//! A journal inverts the trade: capture what changed, on the way through.
//! Every edit pushes the state needed to reverse it, so opening a checkpoint
//! costs O(changes) instead of O(V+E). Rollback — the exception, not the
//! rule — replays the journal backwards and is allowed to be the expensive
//! direction.
//!
//! ## Two capture seams
//!
//! The state one statement can mutate lives in two layers, so capture does
//! too:
//!
//! - the **storage backend** (petgraph nodes/edges), captured inside
//!   `MemoryGraph`'s `GraphWrite` impl — the same choke point the WAL's
//!   [`crate::graph::storage::recording::RecordingGraph`] hooks, and the one
//!   every Cypher write path funnels through;
//! - **`DirGraph`'s inverted indexes** (`type_indices`,
//!   `secondary_label_index`, and the user-created `property_indices` /
//!   `range_indices` / `composite_indices`) and `timeseries_store`, captured
//!   at their documented choke-point APIs, because those structures sit
//!   *above* storage and a backend cannot see them.
//!
//! Everything else a statement can touch is O(schema)-sized and is restored
//! verbatim from a cheap shell clone — see
//! [`crate::graph::dir_graph::rollback`], which owns the restore half and
//! documents exactly which fields are handled which way.
//!
//! ## Reverse replay restores index identity, not just content
//!
//! Entries replay in reverse capture order. That is what lets the journal
//! restore petgraph *slot identity* rather than merely logical content:
//! `StableGraph` keeps freed node/edge slots on a LIFO free list
//! (`free_node`/`free_edge`), so re-inserting removals in reverse order
//! hands every entity back the slot it vacated. `rollback.rs` pins that
//! petgraph behaviour with a dedicated test so a dependency bump cannot
//! silently break the guarantee.
//!
//! Two consequences for what gets recorded:
//!
//! - **Structural** edits (add/remove) are recorded unconditionally, because
//!   their order is what drives free-list reuse.
//! - **Weight** pre-images are recorded at most once per entity — first
//!   touch wins, and the first touch is by definition the pre-statement
//!   state. A five-property `SET n.a=…, n.b=…` therefore costs one
//!   `NodeData` clone, not five.

use std::collections::HashSet;
use std::sync::Arc;

use petgraph::graph::{EdgeIndex, NodeIndex};

use crate::datatypes::Value;
use crate::graph::features::timeseries::NodeTimeseries;
use crate::graph::schema::{
    CompositeIndexKey, CompositeValue, EdgeData, IndexKey, InternedKey, NodeData,
};
use crate::graph::storage::column_store::ColumnStore;

#[cfg(test)]
thread_local! {
    /// Node pre-images actually cloned into an undo journal since the last
    /// reset.
    ///
    /// The complement of `BACKEND_CLONE_NODES` (`storage/backend.rs`), which
    /// counts nodes copied by a *backend* clone and is therefore blind to the
    /// journal path by construction: the whole point of the journal checkpoint
    /// is that it clones no backend. A statement that captures a pre-image per
    /// node of a type rather than per node it changed is O(type)-per-write,
    /// which is the cost class the journal exists to remove, and this is the
    /// only counter that can see it.
    static JOURNAL_NODE_PRE_IMAGES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_journal_node_pre_images() {
    JOURNAL_NODE_PRE_IMAGES.set(0);
}

/// Node pre-images cloned into an undo journal since the last reset.
#[cfg(test)]
pub(crate) fn journal_node_pre_images() -> usize {
    JOURNAL_NODE_PRE_IMAGES.get()
}

/// A `Vec<NodeIndex>` bucket in one of `DirGraph`'s inverted indexes.
/// Bucket edits are journalled with a *position* so a rollback restores the
/// original ordering, not just the original membership — scan order is what
/// an un-`ORDER BY`'d `MATCH` returns, and a failed statement must not
/// perturb it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BucketId {
    /// `DirGraph::type_indices`, keyed by node-type name.
    NodeType(String),
    /// `DirGraph::secondary_label_index`, keyed by interned label.
    SecondaryLabel(InternedKey),
    /// One value bucket of a user-created single-property index
    /// (`DirGraph::property_indices`).
    PropertyValue { key: IndexKey, value: Value },
    /// One value bucket of a user-created range index
    /// (`DirGraph::range_indices`). Same key shape as
    /// [`PropertyValue`](Self::PropertyValue); the bucket lives in a B-tree,
    /// which orders the *values* but not the node indices inside a bucket.
    RangeValue { key: IndexKey, value: Value },
    /// One tuple bucket of a user-created composite index
    /// (`DirGraph::composite_indices`).
    CompositeTuple {
        key: CompositeIndexKey,
        value: CompositeValue,
    },
}

/// One reversible edit. See the module docs for the replay contract.
#[derive(Debug)]
pub enum UndoEntry {
    /// A node was inserted at `idx`. Undo removes it; `node_type` is kept so
    /// the restore can name the `type_indices` bucket without reading a node
    /// that is about to disappear.
    NodeAdded {
        idx: NodeIndex,
        node_type: InternedKey,
    },
    /// `idx`'s weight as it stood before the statement first touched it.
    NodeWeight { idx: NodeIndex, prior: NodeData },
    /// The node at `idx` was removed. Undo re-inserts `prior`, which lands
    /// back on slot `idx` under reverse replay.
    NodeRemoved { idx: NodeIndex, prior: NodeData },
    /// An edge was inserted at `idx`. Undo removes it.
    EdgeAdded { idx: EdgeIndex },
    /// `idx`'s weight as it stood before the statement first touched it.
    EdgeWeight { idx: EdgeIndex, prior: EdgeData },
    /// The edge at `idx` was removed. Undo re-connects `src -> tgt`, which
    /// lands back on slot `idx` under reverse replay.
    EdgeRemoved {
        idx: EdgeIndex,
        src: NodeIndex,
        tgt: NodeIndex,
        prior: EdgeData,
    },
    /// `idx` was appended to an inverted-index bucket. Undo removes it — and
    /// removes the bucket itself when `bucket_was_new`, because an
    /// emptied-but-present bucket is still observable (a zero-count type in
    /// `describe()`).
    BucketAppended {
        bucket: BucketId,
        idx: NodeIndex,
        bucket_was_new: bool,
    },
    /// `idx` was removed from position `pos` of a bucket. Undo re-inserts it
    /// there.
    BucketRemoved {
        bucket: BucketId,
        idx: NodeIndex,
        pos: usize,
    },
    /// A node's timeseries was dropped with the node. Undo restores it.
    /// Boxed: `NodeTimeseries` is far larger than every other variant.
    TimeseriesRemoved {
        node: usize,
        prior: Box<NodeTimeseries>,
    },
    /// `execute_set`'s columnar fast path wrote through the master
    /// `Arc<ColumnStore>` for `node_type`, forking it, and then re-pointed
    /// every node of that type at the fork. `prior` is the master as it stood
    /// before the statement's first such write; undo re-points them all back.
    ///
    /// **One entry per type per statement, not per node.** That is the whole
    /// reason this variant exists: the refresh sweep touches every node of the
    /// type, so journalling it through the ordinary weight-capture seam would
    /// cost a `NodeData` clone per node of the type on every single-row `SET`.
    /// The store itself needs no copy — `ColumnStore` has no interior
    /// mutability and the fork already left `prior` pristine, so holding the
    /// handle is the entire pre-image.
    ColumnarHandles {
        node_type: String,
        prior: Arc<ColumnStore>,
    },
}

/// Buffer of inverse operations for exactly one statement.
///
/// Installed on the in-memory backend for the duration of a mutating
/// statement and taken back out when the statement commits or rolls back;
/// `None` is the steady state, so a graph that is not mid-statement pays one
/// `Option` discriminant check per write call and nothing at all on reads.
#[derive(Debug, Default)]
pub struct UndoJournal {
    entries: Vec<UndoEntry>,
    /// Nodes whose pre-statement weight is already captured (or that were
    /// created by this statement, whose pre-state is "absent").
    weighed_nodes: HashSet<NodeIndex>,
    /// Edge counterpart of `weighed_nodes`.
    weighed_edges: HashSet<EdgeIndex>,
    /// Node types whose pre-statement master column store is already captured.
    /// First touch wins, exactly like `weighed_nodes`: a statement can write
    /// through the same master in several clauses, and only the first one saw
    /// the pre-statement store.
    forked_columnar_types: HashSet<String>,
}

impl UndoJournal {
    /// A fresh, empty journal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume the journal, yielding entries in replay (reverse capture)
    /// order.
    pub fn into_replay_order(self) -> impl Iterator<Item = UndoEntry> {
        self.entries.into_iter().rev()
    }

    // ── backend-seam capture ────────────────────────────────────────────

    /// A node was created at `idx`.
    #[inline]
    pub fn note_node_added(&mut self, idx: NodeIndex, node_type: InternedKey) {
        self.entries.push(UndoEntry::NodeAdded { idx, node_type });
        // Its pre-statement state is "absent", which `NodeAdded` already
        // encodes — later weight captures for this slot are redundant.
        self.weighed_nodes.insert(idx);
    }

    /// A node's weight is about to be mutated. `prior` is only invoked when
    /// this is the statement's first touch of `idx`, so the `NodeData` clone
    /// is paid once per node rather than once per property.
    #[inline]
    pub fn note_node_weight(&mut self, idx: NodeIndex, prior: impl FnOnce() -> Option<NodeData>) {
        if self.weighed_nodes.insert(idx) {
            if let Some(prior) = prior() {
                #[cfg(test)]
                JOURNAL_NODE_PRE_IMAGES.set(JOURNAL_NODE_PRE_IMAGES.get() + 1);
                self.entries.push(UndoEntry::NodeWeight { idx, prior });
            }
        }
    }

    /// The node at `idx` was removed, carrying `prior` out with it.
    #[inline]
    pub fn note_node_removed(&mut self, idx: NodeIndex, prior: NodeData) {
        self.entries.push(UndoEntry::NodeRemoved { idx, prior });
    }

    /// An edge was created at `idx`.
    #[inline]
    pub fn note_edge_added(&mut self, idx: EdgeIndex) {
        self.entries.push(UndoEntry::EdgeAdded { idx });
        self.weighed_edges.insert(idx);
    }

    /// An edge's weight is about to be mutated. See
    /// [`note_node_weight`](Self::note_node_weight) for the laziness
    /// contract.
    #[inline]
    pub fn note_edge_weight(&mut self, idx: EdgeIndex, prior: impl FnOnce() -> Option<EdgeData>) {
        if self.weighed_edges.insert(idx) {
            if let Some(prior) = prior() {
                self.entries.push(UndoEntry::EdgeWeight { idx, prior });
            }
        }
    }

    /// The edge at `idx` (`src -> tgt`) was removed, carrying `prior` out.
    #[inline]
    pub fn note_edge_removed(
        &mut self,
        idx: EdgeIndex,
        src: NodeIndex,
        tgt: NodeIndex,
        prior: EdgeData,
    ) {
        self.entries.push(UndoEntry::EdgeRemoved {
            idx,
            src,
            tgt,
            prior,
        });
    }

    // ── DirGraph-seam capture ───────────────────────────────────────────

    /// `idx` was appended to `bucket`. `bucket_was_new` records whether the
    /// bucket itself came into existence with this append.
    #[inline]
    pub fn note_bucket_appended(&mut self, bucket: BucketId, idx: NodeIndex, bucket_was_new: bool) {
        self.entries.push(UndoEntry::BucketAppended {
            bucket,
            idx,
            bucket_was_new,
        });
    }

    /// `idx` was removed from position `pos` of `bucket`.
    #[inline]
    pub fn note_bucket_removed(&mut self, bucket: BucketId, idx: NodeIndex, pos: usize) {
        self.entries
            .push(UndoEntry::BucketRemoved { bucket, idx, pos });
    }

    /// Journal the removal of every doomed member of `bucket_contents`.
    ///
    /// Positions are recorded **descending**, so reverse replay re-inserts
    /// them ascending — the only order under which each recorded position is
    /// still the right one as the bucket regrows.
    pub fn note_bucket_retain(
        &mut self,
        bucket: &BucketId,
        members: impl Iterator<Item = NodeIndex>,
        doomed: &HashSet<NodeIndex>,
    ) {
        let mut hits: Vec<(usize, NodeIndex)> = members
            .enumerate()
            .filter(|(_, idx)| doomed.contains(idx))
            .collect();
        hits.reverse();
        for (pos, idx) in hits {
            self.note_bucket_removed(bucket.clone(), idx, pos);
        }
    }

    /// The master column store for `node_type` is about to be forked by a
    /// columnar write. `prior` is only taken on the statement's first fork of
    /// that type, so the captured handle is the pre-statement one.
    ///
    /// Returns whether the capture happened, so the caller can skip the
    /// `Arc::clone` on later writes of the same statement.
    #[inline]
    pub fn note_columnar_fork(
        &mut self,
        node_type: &str,
        prior: impl FnOnce() -> Option<Arc<ColumnStore>>,
    ) -> bool {
        if !self.forked_columnar_types.contains(node_type) {
            self.forked_columnar_types.insert(node_type.to_string());
            if let Some(prior) = prior() {
                self.entries.push(UndoEntry::ColumnarHandles {
                    node_type: node_type.to_string(),
                    prior,
                });
                return true;
            }
        }
        false
    }

    /// A node's timeseries was dropped.
    #[inline]
    pub fn note_timeseries_removed(&mut self, node: usize, prior: NodeTimeseries) {
        self.entries.push(UndoEntry::TimeseriesRemoved {
            node,
            prior: Box::new(prior),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::schema::StringInterner;
    use std::collections::HashMap;

    /// Journal contents in replay order.
    fn entries(journal: UndoJournal) -> Vec<UndoEntry> {
        journal.into_replay_order().collect()
    }

    fn node(interner: &mut StringInterner, id: i64) -> NodeData {
        NodeData::new(
            Value::Int64(id),
            Value::String(format!("n{id}")),
            "T".to_string(),
            HashMap::new(),
            interner,
        )
    }

    #[test]
    fn weight_capture_is_once_per_entity() {
        let mut interner = StringInterner::new();
        let mut journal = UndoJournal::new();
        let idx = NodeIndex::new(4);
        let mut calls = 0;
        for _ in 0..5 {
            journal.note_node_weight(idx, || {
                calls += 1;
                Some(node(&mut interner, 1))
            });
        }
        assert_eq!(calls, 1, "the pre-image must be cloned exactly once");
        assert_eq!(entries(journal).len(), 1);
    }

    #[test]
    fn created_nodes_skip_later_weight_capture() {
        let mut interner = StringInterner::new();
        let mut journal = UndoJournal::new();
        let idx = NodeIndex::new(0);
        journal.note_node_added(idx, InternedKey::from_str("T"));
        let mut calls = 0;
        journal.note_node_weight(idx, || {
            calls += 1;
            Some(node(&mut interner, 1))
        });
        assert_eq!(calls, 0, "a node created this statement needs no pre-image");
        assert_eq!(entries(journal).len(), 1);
    }

    #[test]
    fn structural_entries_are_never_deduplicated() {
        let mut interner = StringInterner::new();
        let mut journal = UndoJournal::new();
        let idx = NodeIndex::new(2);
        journal.note_node_removed(idx, node(&mut interner, 1));
        journal.note_node_added(idx, InternedKey::from_str("T"));
        journal.note_node_removed(idx, node(&mut interner, 2));
        assert_eq!(
            entries(journal).len(),
            3,
            "free-list reuse depends on every structural edit being replayed"
        );
    }

    #[test]
    fn replay_order_is_reverse_of_capture() {
        let mut journal = UndoJournal::new();
        journal.note_edge_added(EdgeIndex::new(0));
        journal.note_edge_added(EdgeIndex::new(1));
        journal.note_edge_added(EdgeIndex::new(2));
        let seen: Vec<usize> = journal
            .into_replay_order()
            .map(|e| match e {
                UndoEntry::EdgeAdded { idx } => idx.index(),
                other => panic!("unexpected entry: {other:?}"),
            })
            .collect();
        assert_eq!(seen, vec![2, 1, 0]);
    }

    #[test]
    fn bucket_retain_records_positions_descending() {
        let mut journal = UndoJournal::new();
        let bucket = BucketId::NodeType("T".to_string());
        let contents: Vec<NodeIndex> = (0..8).map(NodeIndex::new).collect();
        let doomed: HashSet<NodeIndex> =
            [NodeIndex::new(2), NodeIndex::new(6)].into_iter().collect();
        journal.note_bucket_retain(&bucket, contents.iter().copied(), &doomed);
        let positions: Vec<usize> = journal
            .into_replay_order()
            .map(|e| match e {
                UndoEntry::BucketRemoved { pos, .. } => pos,
                other => panic!("unexpected entry: {other:?}"),
            })
            .collect();
        // Captured descending → replayed ascending, which is the order in
        // which each stored position is still valid as the bucket refills.
        assert_eq!(positions, vec![2, 6]);
    }
}
