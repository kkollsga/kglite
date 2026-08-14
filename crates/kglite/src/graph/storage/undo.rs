//! Statement-scoped undo journal — the inverse-op buffer that replaces the
//! whole-graph rollback clone.
//!
//! ## Why a journal and not a clone
//!
//! Statement atomicity used to be bought with `DirGraph::fork_transaction()`:
//! a deep clone of the whole graph taken *before* every mutating Cypher
//! statement, discarded on success and swapped back on failure. Correct, but
//! it makes the cost of writing one property proportional to the size of the
//! whole graph — a row-shaped `NodeData` clone deep-copies every
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
//! - **Columnar cell** pre-images ([`UndoEntry::ColumnarCell`]) are recorded
//!   unconditionally, *without* the first-touch dedup above, and that is a
//!   deliberate divergence rather than an oversight. Dedup exists to stop a
//!   whole-entity clone being paid per property; a cell pre-image is one
//!   `Option<Value>`, so the dedup would cost a per-(row, key) hash set to
//!   save a pointer-sized copy. Correctness does not need it either: reverse
//!   replay lands the *earliest* capture last, so a cell written twice in one
//!   statement is restored to its pre-statement value regardless of how many
//!   entries stand between. `replay_order_is_reverse_of_capture` pins the
//!   ordering that argument rests on.

use rustc_hash::FxHashSet;
use std::collections::HashSet;
use std::sync::Arc;

use petgraph::graph::{EdgeIndex, NodeIndex};

use crate::datatypes::Value;
use crate::graph::features::timeseries::NodeTimeseries;
use crate::graph::schema::{
    CompositeIndexKey, CompositeValue, EdgeData, IndexKey, InternedKey, NodeData, TypeSchema,
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

    /// Columnar **cell** pre-images pushed into an undo journal since the last
    /// reset.
    ///
    /// The successor of the `ColumnarHandles` cost model, and the counter that
    /// states the new one: a columnar write must journal one entry per
    /// `(row, key)` it changes, not one per node of the type and not one
    /// whole-store handle. `JOURNAL_NODE_PRE_IMAGES` cannot see these (a
    /// columnar property is not in a `NodeData`) and `COLUMN_STORE_CLONES`
    /// (`storage/column_store.rs`) now reads zero on this path by design — so
    /// without this counter the write path's journal cost would be
    /// unobservable in either direction.
    static JOURNAL_COLUMNAR_CELLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// Columnar **row-append** pre-images pushed into an undo journal since the
    /// last reset.
    ///
    /// The sibling of `JOURNAL_COLUMNAR_CELLS` for the create path, and the
    /// only counter that can see its cost model: an append pre-image holds a
    /// row count and an `Arc<TypeSchema>` clone, so capturing one per created
    /// node instead of one per `(statement, type)` is invisible to the cell
    /// counter (a `CREATE` writes no cells), to `COLUMN_STORE_CLONES` (the
    /// schema `Arc` is not the store) and to `JOURNAL_NODE_PRE_IMAGES`.
    static JOURNAL_COLUMNAR_APPENDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_journal_columnar_appends() {
    JOURNAL_COLUMNAR_APPENDS.set(0);
}

/// Columnar row-append pre-images journalled on this thread since the last
/// reset.
#[cfg(test)]
pub(crate) fn journal_columnar_appends() -> usize {
    JOURNAL_COLUMNAR_APPENDS.get()
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

#[cfg(test)]
pub(crate) fn reset_journal_columnar_cells() {
    JOURNAL_COLUMNAR_CELLS.set(0);
}

/// Columnar cell pre-images journalled on this thread since the last reset.
#[cfg(test)]
pub(crate) fn journal_columnar_cells() -> usize {
    JOURNAL_COLUMNAR_CELLS.get()
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
    /// One cell of a type's master `ColumnStore` is about to be overwritten.
    /// `prior` is the value that cell held before the statement's write; undo
    /// writes it back.
    ///
    /// **O(cells changed), which is the whole point.** The variant this
    /// replaced (`ColumnarHandles`) held an `Arc` of the *whole store*, so the
    /// `Arc::make_mut` at the write site deep-copied every column of the type
    /// to change one cell — a 76-162× tax on every statement against a saved
    /// graph, measured on 0.15.14. A cell pre-image is an `Option<Value>`, and
    /// the master stays uniquely owned, so `make_mut` mutates in place.
    ///
    /// `prior: None` means the cell was absent *or* null before the write, and
    /// undo restores it by writing `Value::Null`. The two are indistinguishable
    /// on every read surface — `ColumnStore::get` returns `None` for both and
    /// `row_properties` skips both — so the restore is observationally exact
    /// even though it is not bit-exact.
    ///
    /// Keyed by `InternedKey`, not by name: capture happens inside the storage
    /// backend, which has no interner. The rollback arm resolves the name when
    /// it needs one.
    ColumnarCell {
        node_type: InternedKey,
        row_id: u32,
        key: InternedKey,
        prior: Option<Value>,
    },
    /// A columnar write introduced a property the type's schema did not have,
    /// so [`ColumnStore::set`] grew the schema and pushed a null-backfilled
    /// column. Undo restores the prior schema `Arc` and truncates the columns
    /// back.
    ///
    /// Captured *before* the paired [`ColumnarCell`](Self::ColumnarCell)
    /// entries, so reverse replay runs it *after* them: the cells restore into
    /// the new column while it still exists, and the column is then dropped.
    ColumnarSchemaGrown {
        node_type: InternedKey,
        prior_schema: Arc<TypeSchema>,
        prior_column_count: usize,
    },
    /// A statement appended rows to a type's master store — one per node it
    /// created, now that construction is columnar. Undo truncates the store
    /// back to `prior_row_count` and drops any column the appended rows'
    /// unseen keys grew the schema by.
    ///
    /// Rows are a stack: only the tail can be appended and only the tail is
    /// removed, so the truncation is exact rather than approximate, and it
    /// restores the *next* row id as well as the row count — a node created
    /// again after a rollback lands on the row its rolled-back predecessor
    /// vacated, which is the columnar half of the petgraph slot-identity
    /// guarantee `NodeAdded` gives.
    ///
    /// Schema growth travels in the same entry rather than a companion
    /// [`ColumnarSchemaGrown`](Self::ColumnarSchemaGrown), because
    /// [`ColumnStore::push_row`] can do both in one call and a single entry
    /// cannot be replayed in the wrong order relative to itself.
    ColumnarRowsAppended {
        node_type: InternedKey,
        prior_row_count: u32,
        prior_schema: Arc<TypeSchema>,
        prior_column_count: usize,
        /// The statement introduced the type's store itself. Undo drops it
        /// rather than leaving an empty one behind — the same distinction
        /// [`BucketAppended`](Self::BucketAppended) draws with
        /// `bucket_was_new`, and for the same reason: an empty-but-present
        /// store is observable (`graph_info`'s `columnar_*` keys, the rollback
        /// fingerprint's master rows).
        store_was_new: bool,
    },
    /// A statement overwrote a row's reserved `__title__` cell. Undo writes the
    /// prior title back; `None` restores `Value::Null`, which is what an absent
    /// title reads as anyway.
    ///
    /// Titles need their own variant because the title column is addressed by
    /// its reserved position rather than by an `InternedKey` slot, so
    /// [`ColumnarCell`](Self::ColumnarCell) cannot name it.
    ColumnarTitle {
        node_type: InternedKey,
        row_id: u32,
        prior: Option<Value>,
    },
    /// A statement tombstoned a row — the columnar half of deleting a node.
    /// Undo clears the flag; the row's values were hidden, never overwritten,
    /// so nothing else has to be restored.
    ColumnarTombstone { node_type: InternedKey, row_id: u32 },
}

/// Which cells of a columnar row a property write is about to change.
///
/// The capture seam under all three columnar write sites — the Cypher master
/// fast path, the five `impl_heap_column_writes!` writers, and `ForkedGraph` —
/// needs the *keys*, not just the row: a cell-grained pre-image cannot be taken
/// without knowing which cells are at risk.
pub(crate) enum ColumnarWrite<'a> {
    /// Exactly one cell.
    Cell(InternedKey),
    /// `replace_node_properties`' shape: every key currently present on the row
    /// is nulled, then these keys are written.
    ReplaceRow(&'a [InternedKey]),
}

/// The pre-images one columnar write needs, read off the store **before** the
/// write and pushed into the journal after the store borrow ends.
///
/// Two-step rather than one because the journal and the store live in sibling
/// fields of the same backend: capture borrows the store, `record` borrows the
/// journal, and nothing borrows both at once.
pub(crate) struct ColumnarPreImages {
    /// `(key, value before the write)`, in capture order.
    cells: Vec<(InternedKey, Option<Value>)>,
    /// Schema and column count as they stood before the write, when the write
    /// introduces at least one property the schema does not have yet. One entry
    /// covers any number of new keys in the same write: replay truncates back
    /// to `column_count` in a single step.
    grown: Option<(Arc<TypeSchema>, usize)>,
}

impl ColumnarPreImages {
    /// Read what `write` is about to overwrite on `row_id` of `store`.
    pub(crate) fn capture(store: &ColumnStore, row_id: u32, write: ColumnarWrite<'_>) -> Self {
        let mut cells = Vec::new();
        let mut grows = false;
        match write {
            ColumnarWrite::Cell(key) => {
                grows = store.slot(key).is_none();
                cells.push((key, store.get(row_id, key)));
            }
            ColumnarWrite::ReplaceRow(keys) => {
                // The writer nulls every present cell first; those keys are by
                // definition already in the schema, so they cannot grow it.
                for (key, value) in store.row_properties(row_id) {
                    cells.push((key, Some(value)));
                }
                for &key in keys {
                    grows |= store.slot(key).is_none();
                    cells.push((key, store.get(row_id, key)));
                }
            }
        }
        Self {
            cells,
            grown: grows.then(|| (store.schema_arc(), store.column_count())),
        }
    }

    /// Push the captured pre-images, schema entry first so reverse replay runs
    /// it last.
    pub(crate) fn record(self, journal: &mut UndoJournal, node_type: InternedKey, row_id: u32) {
        if let Some((prior_schema, prior_column_count)) = self.grown {
            journal.entries.push(UndoEntry::ColumnarSchemaGrown {
                node_type,
                prior_schema,
                prior_column_count,
            });
        }
        for (key, prior) in self.cells {
            #[cfg(test)]
            JOURNAL_COLUMNAR_CELLS.set(JOURNAL_COLUMNAR_CELLS.get() + 1);
            journal.entries.push(UndoEntry::ColumnarCell {
                node_type,
                row_id,
                key,
                prior,
            });
        }
    }
}

/// The pre-image one columnar **row append** needs, read off the store before
/// the append and pushed into the journal after the store borrow ends.
///
/// Same two-step shape as [`ColumnarPreImages`] and for the same reason: the
/// journal and the store are sibling fields of one backend, so capture borrows
/// the store and `record` borrows the journal, never both at once.
pub(crate) struct ColumnarAppendPreImage {
    row_count: u32,
    schema: Arc<TypeSchema>,
    column_count: usize,
}

impl ColumnarAppendPreImage {
    /// Read the store's length and schema as they stand before the append.
    #[inline]
    pub(crate) fn capture(store: &ColumnStore) -> Self {
        Self {
            row_count: store.row_count(),
            schema: store.schema_arc(),
            column_count: store.column_count(),
        }
    }

    /// Push the captured pre-image. `store_was_new` is read before the store is
    /// created, not from the store itself, which is why it arrives here rather
    /// than in [`Self::capture`].
    #[inline]
    pub(crate) fn record(
        self,
        journal: &mut UndoJournal,
        node_type: InternedKey,
        store_was_new: bool,
    ) {
        #[cfg(test)]
        JOURNAL_COLUMNAR_APPENDS.set(JOURNAL_COLUMNAR_APPENDS.get() + 1);
        journal.entries.push(UndoEntry::ColumnarRowsAppended {
            node_type,
            prior_row_count: self.row_count,
            prior_schema: self.schema,
            prior_column_count: self.column_count,
            store_was_new,
        });
    }
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
    /// FxHash, not the std SipHasher, for both: the keys are bare `u32`
    /// indices and these are probed once per touched node/edge per write on
    /// the statement hot path. Membership only — never iterated — so the
    /// hasher cannot reach an ordering anything observes.
    weighed_nodes: FxHashSet<NodeIndex>,
    /// Edge counterpart of `weighed_nodes`.
    weighed_edges: FxHashSet<EdgeIndex>,
    /// Node types whose columnar **append** pre-image this statement has
    /// already captured — the row-append analogue of `weighed_nodes`, and for
    /// the same reason: the first capture is the only one that carries the
    /// pre-statement state.
    appended_types: FxHashSet<InternedKey>,
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

    /// Claim the columnar append pre-image for `node_type`: `true` on the
    /// statement's first row append into that type's store, `false` after.
    ///
    /// **Why first-wins is the whole undo.** `ColumnarRowsAppended` restores
    /// *absolutely* — truncate to `prior_row_count`, restore `prior_schema`,
    /// drop the store when it was new — and reverse replay runs the earliest
    /// capture last, so the end state a statement rolls back to is decided by
    /// the first entry per type and by nothing after it. Every later entry
    /// names an intermediate row count that the first one then overrides. One
    /// per created node was therefore one journal entry (and one schema `Arc`
    /// clone) per node to describe a state already described.
    ///
    /// The dual of `note_node_weight`'s laziness: capture the pre-image on
    /// first touch, skip it thereafter.
    #[inline]
    pub fn claim_columnar_append(&mut self, node_type: InternedKey) -> bool {
        self.appended_types.insert(node_type)
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

    /// A row's reserved title cell is about to be overwritten.
    #[inline]
    pub fn note_columnar_title(
        &mut self,
        node_type: InternedKey,
        row_id: u32,
        prior: Option<Value>,
    ) {
        self.entries.push(UndoEntry::ColumnarTitle {
            node_type,
            row_id,
            prior,
        });
    }

    /// A row of `node_type`'s master store was tombstoned by a node deletion.
    #[inline]
    pub fn note_columnar_tombstone(&mut self, node_type: InternedKey, row_id: u32) {
        self.entries
            .push(UndoEntry::ColumnarTombstone { node_type, row_id });
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

    /// A columnar store with one row and two typed-as-mixed properties.
    fn cell_store(interner: &mut StringInterner) -> (ColumnStore, InternedKey, InternedKey) {
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let schema = Arc::new(crate::graph::schema::TypeSchema::from_keys([a, b]));
        let mut store = ColumnStore::new_mixed(schema);
        store.push_row(&[(a, Value::Int64(1)), (b, Value::Int64(2))]);
        (store, a, b)
    }

    /// Two writes to the same cell in one statement produce two entries, and
    /// reverse replay lands the *earlier* one last — which is why the columnar
    /// capture needs no first-touch dedup (module doc).
    #[test]
    fn columnar_cells_are_not_deduplicated_and_replay_oldest_last() {
        let mut interner = StringInterner::new();
        let (mut store, a, _b) = cell_store(&mut interner);
        let node_type = InternedKey::from_str("T");
        let mut journal = UndoJournal::new();

        ColumnarPreImages::capture(&store, 0, ColumnarWrite::Cell(a)).record(
            &mut journal,
            node_type,
            0,
        );
        store.set(0, a, &Value::Int64(10), None);
        ColumnarPreImages::capture(&store, 0, ColumnarWrite::Cell(a)).record(
            &mut journal,
            node_type,
            0,
        );
        store.set(0, a, &Value::Int64(20), None);

        // Replay them in order; the last write wins, and it is the oldest
        // capture.
        for entry in journal.into_replay_order() {
            match entry {
                UndoEntry::ColumnarCell {
                    row_id, key, prior, ..
                } => {
                    store.set(row_id, key, &prior.unwrap_or(Value::Null), None);
                }
                other => panic!("unexpected entry: {other:?}"),
            }
        }
        assert_eq!(
            store.get(0, a),
            Some(Value::Int64(1)),
            "reverse replay must land the pre-statement capture last"
        );
    }

    /// A write that introduces a new property captures the schema pre-image
    /// *before* the cell, so reverse replay drops the column *after* restoring
    /// into it.
    #[test]
    fn schema_growth_is_captured_before_its_cells() {
        let mut interner = StringInterner::new();
        let (store, _a, _b) = cell_store(&mut interner);
        let fresh = interner.get_or_intern("fresh");
        let node_type = InternedKey::from_str("T");
        let mut journal = UndoJournal::new();

        ColumnarPreImages::capture(&store, 0, ColumnarWrite::Cell(fresh)).record(
            &mut journal,
            node_type,
            0,
        );

        let kinds: Vec<&'static str> = journal
            .into_replay_order()
            .map(|entry| match entry {
                UndoEntry::ColumnarCell { .. } => "cell",
                UndoEntry::ColumnarSchemaGrown {
                    prior_column_count, ..
                } => {
                    assert_eq!(prior_column_count, 2, "the pre-growth column count");
                    "schema"
                }
                other => panic!("unexpected entry: {other:?}"),
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["cell", "schema"],
            "replay must restore the cell while its column still exists, then \
             drop the column"
        );
    }

    /// A write to a key already in the schema captures no schema entry — the
    /// non-vacuity control for the test above.
    #[test]
    fn an_existing_key_captures_no_schema_entry() {
        let mut interner = StringInterner::new();
        let (store, a, _b) = cell_store(&mut interner);
        let mut journal = UndoJournal::new();
        ColumnarPreImages::capture(&store, 0, ColumnarWrite::Cell(a)).record(
            &mut journal,
            InternedKey::from_str("T"),
            0,
        );
        assert_eq!(entries(journal).len(), 1);
    }

    /// `ReplaceRow` captures every cell present on the row plus every key the
    /// write brings, so a rollback can restore both what was overwritten and
    /// what was nulled.
    #[test]
    fn replace_row_captures_present_cells_and_incoming_keys() {
        let mut interner = StringInterner::new();
        let (store, a, b) = cell_store(&mut interner);
        let fresh = interner.get_or_intern("fresh");
        let mut journal = UndoJournal::new();
        ColumnarPreImages::capture(&store, 0, ColumnarWrite::ReplaceRow(&[a, fresh])).record(
            &mut journal,
            InternedKey::from_str("T"),
            0,
        );

        let mut cells: Vec<(InternedKey, Option<Value>)> = journal
            .into_replay_order()
            .filter_map(|entry| match entry {
                UndoEntry::ColumnarCell { key, prior, .. } => Some((key, prior)),
                UndoEntry::ColumnarSchemaGrown { .. } => None,
                other => panic!("unexpected entry: {other:?}"),
            })
            .collect();
        cells.sort_by_key(|(key, _)| key.as_u64());
        let mut expected = vec![
            (a, Some(Value::Int64(1))),
            (b, Some(Value::Int64(2))),
            (a, Some(Value::Int64(1))),
            (fresh, None),
        ];
        expected.sort_by_key(|(key, _)| key.as_u64());
        assert_eq!(cells, expected);
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
