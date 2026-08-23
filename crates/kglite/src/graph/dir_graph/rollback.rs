//! Statement-level rollback: a cheap checkpoint plus undo-journal restore.
//!
//! One mutating Cypher statement is atomic. It can fail *after* its first
//! write (a constraint violation on row 40 of a `CREATE`, a write-scope
//! rejection, an expression error), and the graph must then look exactly as
//! it did before the statement ran.
//!
//! ## Two ways to be atomic
//!
//! A whole-graph clone is unconditionally correct — writing one property to a
//! million-node graph deep-copies a million `NodeData`s first. It stays as the
//! fallback; the journal is the cheap path.
//!
//! | | cost to open | cost to roll back |
//! |---|---|---|
//! | [`StatementCheckpoint::None`] | nothing | n/a — the shape cannot fail after its first write |
//! | [`StatementCheckpoint::Journal`] | O(1) + O(changes) | O(changes) + reindex of affected types |
//! | [`StatementCheckpoint::Clone`] | O(V+E) | O(1) swap |
//!
//! ## How the journal path splits `DirGraph`'s state
//!
//! `DirGraph`'s fields split by *clone cost*:
//!
//! - **O(schema)-sized fields are cloned** into a `shell` and restored
//!   verbatim: the whole schema surface a statement can grow, and also every
//!   field nobody has thought about yet, which is the point — **a field that
//!   is not explicitly parked is automatically restored**, so forgetting to
//!   park a new large field costs speed, never correctness.
//!
//!   Six of them scale with schema width (`node_type_metadata`,
//!   `connection_type_metadata`, `type_schemas`, the two alias maps and
//!   `parent_types`) and are `Arc`-shared copy-on-write: the shell takes a
//!   pointer and the statement's first write to each forks it once. Restore
//!   semantics are identical; a 200-type × 50-column schema costs a refcount
//!   instead of 337 µs of memcpy per statement. Contract, writer rules and
//!   measurement: [`super::schema_cow`].
//! - **O(V+E)-sized fields are parked** (moved aside so `clone()` sees them
//!   empty) and reconstructed from the journal. [`swap_data_scale`] is that
//!   short, closed list; each member has a documented undo story in [`apply`].
//!
//! Two parked fields are *rebuilt* rather than journal-reversed, because their
//! own mutation sites record no inverse: `id_indices` (dropped per touched
//! type, self-healing on the next read) and `unique_indices` (recomputed per
//! touched type by [`DirGraph::rebuild_unique_indices_for_types`]). Both are
//! driven by the [`ReplayFallout`] the replay collects. Parking a field with no
//! undo story of *either* kind is the one way to make this module wrong — see
//! the `unique_indices` note on [`swap_data_scale`].
//!
//! ## Columnar mode: one entry per changed cell, replayed into the live store
//!
//! The per-type master `ColumnStore` map lives on the **storage backend**,
//! which `swap_data_scale` swaps wholesale — so it is journal territory,
//! not shell territory: the shell restore never sees it.
//! [`UndoEntry::ColumnarCell`] carries the prior value of each `(row, key)` a
//! statement overwrote, and its replay writes that value straight back into the
//! live store. [`UndoEntry::ColumnarSchemaGrown`] is its companion for the
//! schema edit a `SET` can make: introducing a property the type's schema
//! lacks, which grows the schema and appends a null-backfilled column. A title
//! is not a schema slot, so it gets its own [`UndoEntry::ColumnarTitle`].
//!
//! What makes that sufficient:
//!
//! - The capture happens **before** the write, at all three columnar write
//!   sites, because after the write the prior value exists nowhere. That
//!   ordering is the entire correctness argument and it cannot be reversed.
//! - Replay is in reverse capture order, so a cell written several times in one
//!   statement is restored by its *earliest* capture — the pre-statement one —
//!   and needs no first-touch dedup. A schema-growth entry captured before its
//!   own cells replays after them, so the cells are restored into a column that
//!   still exists and the column is dropped afterwards.
//! - The journal holds **no handle on the store**: the master stays uniquely
//!   owned through the statement, so `Arc::make_mut` at the write site mutates
//!   one cell in place instead of deep-copying every column of the type. That
//!   direction is asserted at the write site
//!   (`columnar_write::write_column_master`), which is also where the one
//!   legitimate exception lives — a `Forked` backend shares its stores with the
//!   base a reader holds, so its first write per type must copy.
//! - `CREATE` and `DELETE` **do** reach a master store, in every mode: a create
//!   appends a row and a delete tombstones one, both inside the statement
//!   window. Neither is a cell edit, so each has its own entry —
//!   [`UndoEntry::ColumnarRowsAppended`] and [`UndoEntry::ColumnarTombstone`] —
//!   and between them the store's *length* and its *liveness bitmap* are
//!   restored as exactly as its cells are. Rows are append-only, so the
//!   truncation also restores the next row id: a node re-created after a
//!   rollback lands on the vacated row rather than leaking a hole, the columnar
//!   half of the petgraph slot identity below. Every *other* writer of the
//!   store map (`enable_columnar`, `vacuum`, the spill and bulk-batch paths)
//!   runs outside the statement window, so those need no entry — and the
//!   bulk-batch path, which swaps stores wholesale, journals its own append
//!   pre-image before doing so should a caller ever run it under an open
//!   checkpoint.
//! - A columnar `SET` writes into the master, never into a node's weight, so it
//!   produces **no** `NodeWeight` entry for the node it changed; the columnar
//!   entries are the only signal that the type was written at all, which is why
//!   their replay reports it — see [`columnar_type_touched`].
//!
//! Two documented residues, both observational no-ops:
//!
//! - A cell that was **absent** before the statement is restored by writing
//!   `Value::Null`. `ColumnStore::get` answers `None` for absent and null
//!   alike and `row_properties` skips both, so no read surface — including the
//!   `rollback_tests::fingerprint` oracle — can tell the difference.
//! - A rolled-back write whose value did not fit the column's type left the
//!   column **demoted to `Mixed`** on its way in, and the restore puts the
//!   value back without re-narrowing it. Values and reads are identical; only
//!   the column's storage tag differs, in memory, until the next consolidation
//!   re-derives it.
//!
//! ## What "identical" means here
//!
//! A journal rollback restores logical content *and* petgraph slot identity —
//! nodes and edges come back on the exact `NodeIndex`/`EdgeIndex` they
//! vacated (see `storage/undo.rs` for why reverse replay guarantees this),
//! and inverted-index buckets come back in their original order, so scan
//! order is unperturbed. The one thing it does not restore is the backend's
//! high-water mark: `StableGraph` never shrinks its slot vector, so a
//! statement that created and then rolled back a node leaves `node_bound()`
//! one higher with the slot on the free list. That is the same state any
//! `DELETE` leaves behind, it is invisible through every read API, and the
//! next insert reuses the slot.

use std::collections::HashSet;

use petgraph::graph::NodeIndex;

use super::DirGraph;
use crate::datatypes::Value;
use crate::graph::schema::{InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::undo::{BucketId, UndoEntry, UndoJournal};
use crate::graph::storage::{GraphRead, GraphWrite};

/// Move every O(V+E) field between two graphs. O(1) per field.
///
/// **This list is the safety boundary of the journal path.** A field named
/// here is *not* covered by the shell clone and must have an explicit undo
/// story in [`apply`]; a field absent from here is restored verbatim from the
/// shell. When adding a field to `DirGraph`, leave it out of this list unless
/// its clone cost genuinely scales with graph size.
fn swap_data_scale(a: &mut DirGraph, b: &mut DirGraph) {
    std::mem::swap(&mut a.graph, &mut b.graph);
    std::mem::swap(&mut a.type_indices, &mut b.type_indices);
    std::mem::swap(&mut a.id_indices, &mut b.id_indices);
    // The three user-index families are journalled per *bucket edit* — the
    // incremental maintenance a statement's writes drive. Whole-index DDL
    // (`create_index` / `drop_index`, which replace a map wholesale) is not,
    // so a `CREATE INDEX` that failed after installing its map would leave the
    // map behind while the shell restored the `*_index_keys` list without it.
    // That is unreachable rather than handled: on the journal-capable backend
    // index DDL performs no fallible step after it mutates. A new fallible
    // step there needs an undo entry for the whole index, or an explicit veto.
    std::mem::swap(&mut a.property_indices, &mut b.property_indices);
    std::mem::swap(&mut a.composite_indices, &mut b.composite_indices);
    std::mem::swap(&mut a.range_indices, &mut b.range_indices);
    std::mem::swap(&mut a.secondary_label_index, &mut b.secondary_label_index);
    // O(V × dimension) — a 100k × 384 store is 150 MB, so cloning it per
    // statement is out of the question. Its undo story is
    // `UndoEntry::EmbeddingRemoved`, captured where a node deletion prunes the
    // node's vector (`mutation::delete_state::prune_doomed_embeddings`); node
    // deletion is the only writer that reaches this map inside a statement
    // window (ingest runs outside one), so that entry is the whole story.
    std::mem::swap(&mut a.embeddings, &mut b.embeddings);
    std::mem::swap(&mut a.timeseries_store, &mut b.timeseries_store);
    // Parked because its inner map holds one entry per node of a constrained
    // type, so cloning it would put an O(constrained-nodes) copy back on every
    // mutating statement — the cost this module exists to remove.
    //
    // Parking alone would be a *correctness* bug: `commit_unique_claims` /
    // `release_unique_claims` / `evict_unique_claims_for_nodes` record nothing
    // in the journal, and parking keeps the live post-statement map rather
    // than the shell's, so a failed statement's claims would survive rollback
    // — a phantom occupant (permanent spurious `ConstraintViolationError`) or
    // a released claim (a real duplicate admitted). The paired undo story is
    // the per-touched-type rebuild in `StatementCheckpoint::rollback`; do not
    // park this field without it.
    std::mem::swap(&mut a.unique_indices, &mut b.unique_indices);
}

/// Reverse one bucket append: drop the *last* copy of `idx`, which is the one
/// the append pushed.
///
/// `retain`-style removal would be wrong here — it also drops an occurrence
/// that pre-dated the statement, and a bucket can legitimately be appended to
/// twice within one statement.
fn undo_bucket_append(members: Option<&mut Vec<NodeIndex>>, idx: NodeIndex) {
    if let Some(members) = members {
        if let Some(pos) = members.iter().rposition(|member| *member == idx) {
            members.remove(pos);
        }
    }
}

/// Whether the undo journal can reverse everything this graph's mutations can
/// change. Conservative by construction — every `false` arm falls back to the
/// clone checkpoint, so a shape that is merely *unproven* is still safe.
///
/// Only the backend is a gate. Do not re-add a graph-state veto: it is cheap
/// to type and expensive to run — not a local slowdown but a permanent,
/// whole-graph downgrade to an O(V+E) clone per statement, for every shape,
/// for the rest of the session. The three that were tried, and why each is
/// unnecessary:
///
/// - **`column_stores`** guarded the columnar-`SET` master side channel, but
///   fired on every graph that had ever been saved. Covered instead by the
///   unparked shell restore plus the journalled cell pre-images; see the
///   module doc.
/// - **`property_indices` / `range_indices` / `composite_indices`** guarded
///   the absence of position undo for user-index buckets. That undo exists,
///   recorded at the incremental-maintenance choke points in
///   [`crate::graph::dir_graph::indexes`] and
///   [`crate::graph::mutation::maintain`] with the same `BucketAppended` /
///   `BucketRemoved` entries `type_indices` already used. Whole-index DDL is
///   still not journalled — see the note on [`swap_data_scale`].
/// - **`unique_indices`**: routing every constrained graph to
///   `fork_transaction()` is strictly worse than the journal plus the
///   per-touched-type unique rebuild that is that field's undo story.
fn journal_covers(graph: &DirGraph) -> bool {
    // Only a petgraph-backed backend can express an inverse edit: Memory and
    // Mapped, whose `MappedGraph.inner` is the same StableDiGraph. Disk has no
    // petgraph and no NodeIndex identity to restore, and every UndoEntry
    // variant is keyed on one, so it takes the whole-graph checkpoint.
    graph.graph.supports_undo_journal()
}

/// An open rollback checkpoint for exactly one mutating statement.
///
/// Must be closed with [`commit`](Self::commit) or
/// [`rollback`](Self::rollback) on every exit path — closing is what
/// uninstalls the capture journal, so a leaked checkpoint would keep
/// journalling into the next statement.
pub(crate) enum StatementCheckpoint {
    /// Nothing to undo: either the statement is a read, or it was proven to
    /// finish all fallible work before its first write.
    None,
    /// Pre-statement O(schema) state; the O(V+E) state comes back from the
    /// backend's undo journal.
    Journal {
        shell: Box<DirGraph>,
        /// WAL-capture op-buffer length at open time, so a rollback discards
        /// exactly this statement's ops. `None` when the backend captures no
        /// WAL ops.
        recorded_ops: Option<usize>,
    },
    /// Whole-graph clone, for backends and graph shapes outside
    /// [`journal_covers`].
    Clone { snapshot: Box<DirGraph> },
}

impl StatementCheckpoint {
    /// Open a checkpoint for a statement that may fail after its first write.
    pub(crate) fn open(graph: &mut DirGraph) -> Self {
        if !journal_covers(graph) {
            return Self::Clone {
                snapshot: Box::new(graph.fork_transaction()),
            };
        }
        let recorded_ops = graph.graph.recorded_ops_len();
        let shell = Box::new(graph.schema_shell());
        graph.graph.begin_undo();
        Self::Journal {
            shell,
            recorded_ops,
        }
    }

    /// The statement succeeded: drop the checkpoint and stop capturing.
    pub(crate) fn commit(self, graph: &mut DirGraph) {
        if matches!(self, Self::Journal { .. }) {
            // Dropping the journal is the whole commit: the graph already
            // holds the new state.
            graph.graph.take_undo();
        }
    }

    /// The statement failed: restore the pre-statement graph.
    pub(crate) fn rollback(self, graph: &mut DirGraph) {
        match self {
            Self::None => {}
            Self::Clone { snapshot } => *graph = *snapshot,
            Self::Journal {
                shell,
                recorded_ops,
            } => {
                let journal = graph.graph.take_undo();
                let fallout = journal
                    .map(|journal| replay(graph, *journal))
                    .unwrap_or_default();
                // Ops the failed statement buffered for the write-ahead log
                // describe writes that no longer exist.
                if let Some(len) = recorded_ops {
                    graph.graph.truncate_recorded_ops(len);
                }
                graph.restore_schema_shell(*shell);
                // After the shell restore, so the rebuild reads the
                // pre-statement schema (property readers and column metadata)
                // against the restored data. `unique_indices` is parked, so
                // what survives here is the failed statement's occupancy — see
                // the note in `swap_data_scale`.
                graph.rebuild_unique_indices_for_types(&fallout.stale_unique_indices);
                // Edge-derived caches are `Arc`-shared with the shell, so the
                // shell cannot restore them; drop them and let the next read
                // rebuild.
                graph.invalidate_edge_type_counts_cache();
            }
        }
    }
}

impl DirGraph {
    /// Clone only the O(schema) half of this graph: the returned shell has
    /// empty data-scale fields and a faithful copy of everything else.
    ///
    /// The six schema-scale maps in "everything else" are `Arc`-shared, so the
    /// copy, if any, happens at the statement's first writer. See
    /// [`super::schema_cow`] — and do not reach for `Arc::make_mut` on those
    /// fields anywhere else.
    pub(super) fn schema_shell(&mut self) -> DirGraph {
        let mut husk = DirGraph::new();
        swap_data_scale(self, &mut husk);
        let shell = self.clone();
        swap_data_scale(self, &mut husk);
        shell
    }

    /// Adopt `shell`'s O(schema) state while keeping the live O(V+E) state.
    pub(super) fn restore_schema_shell(&mut self, mut shell: DirGraph) {
        // `shell`'s data-scale fields are empty; hand it the live ones, then
        // become it.
        swap_data_scale(&mut shell, self);
        *self = shell;
    }
}

/// The derived per-node-type state a replay perturbed, for the two parked
/// fields that are rebuilt instead of journal-reversed.
#[derive(Default)]
struct ReplayFallout {
    /// Node types whose id-index the replay perturbed. Whole-type
    /// invalidation, not the per-entry `evict_entries` the delete path uses:
    /// a replay restores and removes nodes in the same pass, so the surviving
    /// set is not known per entry, and the read path self-heals via
    /// `lookup_or_build` anyway.
    stale_id_indices: HashSet<String>,
    /// Node types whose unique-occupancy maps must be recomputed from the
    /// restored data. Wider than `stale_id_indices`: a plain property
    /// overwrite leaves identity alone but can add or release a unique claim,
    /// so `NodeWeight` counts here and not there.
    stale_unique_indices: HashSet<String>,
}

impl ReplayFallout {
    /// Record a node type whose identity *and* unique claims were perturbed —
    /// a node appearing or disappearing does both.
    fn node_identity_changed(&mut self, node_type: String) {
        self.stale_unique_indices.insert(node_type.clone());
        self.stale_id_indices.insert(node_type);
    }
}

/// Replay every entry in reverse capture order, reporting what has to be
/// rebuilt afterwards.
fn replay(graph: &mut DirGraph, journal: UndoJournal) -> ReplayFallout {
    let mut fallout = ReplayFallout::default();
    for entry in journal.into_replay_order() {
        apply(graph, entry, &mut fallout);
    }
    for node_type in &fallout.stale_id_indices {
        graph.id_indices.remove(node_type);
    }
    fallout
}

/// Reverse one edit. The journal is already uninstalled, so the `GraphWrite`
/// calls below are not re-captured.
fn apply(graph: &mut DirGraph, entry: UndoEntry, fallout: &mut ReplayFallout) {
    match entry {
        UndoEntry::NodeAdded { idx, node_type } => undo_node_added(graph, idx, node_type, fallout),
        UndoEntry::NodeWeight { idx, prior } => undo_node_weight(graph, idx, prior, fallout),
        UndoEntry::NodeRemoved { idx, prior } => undo_node_removed(graph, idx, prior, fallout),
        UndoEntry::EdgeAdded { idx } => {
            GraphWrite::remove_edge(&mut graph.graph, idx);
        }
        UndoEntry::EdgeWeight { idx, prior } => {
            if let Some(slot) = GraphWrite::edge_weight_mut(&mut graph.graph, idx) {
                *slot = prior;
            }
        }
        UndoEntry::EdgeRemoved {
            idx,
            src,
            tgt,
            prior,
        } => {
            let restored = GraphWrite::add_edge(&mut graph.graph, src, tgt, prior);
            debug_assert_slot_reused(restored.index(), idx.index(), "edge");
        }
        UndoEntry::BucketAppended {
            bucket,
            idx,
            bucket_was_new,
        } => undo_bucket_appended(graph, bucket, idx, bucket_was_new),
        UndoEntry::BucketRemoved { bucket, idx, pos } => {
            undo_bucket_removed(graph, bucket, idx, pos)
        }
        UndoEntry::TimeseriesRemoved { node, prior } => {
            graph.timeseries_store.insert(node, *prior);
        }
        UndoEntry::EmbeddingRemoved {
            store_key,
            node,
            prior,
        } => {
            // Pruning never drops a store, so the key is still present; a
            // missing one would mean some other writer removed the store
            // mid-statement, and re-creating it here from a single vector
            // would invent a dimension.
            if let Some(store) = graph.embeddings.get_mut(&store_key) {
                store.restore_embedding(node, &prior);
            }
        }
        UndoEntry::ColumnarCell {
            node_type,
            row_id,
            key,
            prior,
        } => {
            // `prior: None` restores as `Null` — indistinguishable through
            // every read surface, including the `rollback_tests::fingerprint`
            // oracle; see the module doc.
            edit_column_master(graph, node_type, fallout, |store| {
                store.set(row_id, key, &prior.unwrap_or(Value::Null), None);
            });
        }
        UndoEntry::ColumnarSchemaGrown {
            node_type,
            prior_schema,
            prior_column_count,
        } => {
            // Replayed *after* the cell entries of the same write (captured
            // later, so reverse replay runs them first): each new column is
            // restored to its pre-statement content and only then dropped,
            // which keeps the two entries independent.
            edit_column_master(graph, node_type, fallout, |store| {
                store.restore_schema(prior_schema, prior_column_count);
            });
        }
        UndoEntry::ColumnarRowsAppended {
            node_type,
            prior_row_count,
            prior_schema,
            prior_column_count,
            store_was_new,
        } => undo_columnar_rows_appended(
            graph,
            node_type,
            prior_row_count,
            prior_schema,
            prior_column_count,
            store_was_new,
            fallout,
        ),
        UndoEntry::ColumnarTitle {
            node_type,
            row_id,
            prior,
        } => {
            edit_column_master(graph, node_type, fallout, |store| {
                store.set_title(row_id, &prior.unwrap_or(Value::Null));
            });
        }
        UndoEntry::ColumnarTombstone { node_type, row_id } => {
            edit_column_master(graph, node_type, fallout, |store| {
                store.untombstone(row_id);
            });
        }
    }
}

/// Undo a node insertion: drop the node and invalidate its type's derived
/// per-type structures.
fn undo_node_added(
    graph: &mut DirGraph,
    idx: NodeIndex,
    node_type: InternedKey,
    fallout: &mut ReplayFallout,
) {
    // `type_indices` is reversed by the `BucketAppended` entry the create path
    // recorded; only the derived per-type structures need invalidating here.
    fallout.node_identity_changed(graph.interner.resolve(node_type).to_string());
    // A node created by this statement can only carry edges this statement
    // created, and those replayed first (they were captured later), so it is
    // isolated by now.
    debug_assert!(
        graph
            .graph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .next()
            .is_none()
            && graph
                .graph
                .edges_directed(idx, petgraph::Direction::Incoming)
                .next()
                .is_none(),
        "a rolled-back node must be isolated before removal"
    );
    GraphWrite::remove_node(&mut graph.graph, idx);
}

fn undo_node_weight(
    graph: &mut DirGraph,
    idx: NodeIndex,
    prior: NodeData,
    fallout: &mut ReplayFallout,
) {
    // Both spellings of the type, so a claim under either is recomputed. The
    // primary type is immutable in practice, but reading it costs one interner
    // lookup on a path that only runs when a statement already failed.
    let restored_type = graph.interner.resolve(prior.node_type).to_string();
    if let Some(current) = GraphRead::node_type_of(&graph.graph, idx) {
        let current = graph.interner.resolve(current).to_string();
        if current != restored_type {
            fallout.stale_unique_indices.insert(current);
        }
    }
    fallout.stale_unique_indices.insert(restored_type);
    if let Some(slot) = GraphWrite::node_weight_mut(&mut graph.graph, idx) {
        *slot = prior;
    }
}

/// Re-insert a removed node, which reverse replay lands back on slot `idx`.
fn undo_node_removed(
    graph: &mut DirGraph,
    idx: NodeIndex,
    prior: NodeData,
    fallout: &mut ReplayFallout,
) {
    let type_name = graph.interner.resolve(prior.node_type).to_string();
    let restored = GraphWrite::add_node(&mut graph.graph, prior);
    debug_assert_slot_reused(restored.index(), idx.index(), "node");
    fallout.node_identity_changed(type_name);
}

/// Reverse one inverted-index bucket append across all five bucket families.
fn undo_bucket_appended(
    graph: &mut DirGraph,
    bucket: BucketId,
    idx: NodeIndex,
    bucket_was_new: bool,
) {
    match bucket {
        BucketId::NodeType(name) => {
            // Into this graph's own delta, never into a base a forked
            // reader is holding — see `disk/type_index_layer.rs`. Falls
            // back to the flattening retain when the append is not in the
            // writable tail.
            graph.type_indices.undo_append(&name, idx);
            // Only drop the bucket if this statement introduced it: an
            // empty bucket can legitimately pre-exist (a type whose nodes
            // were all deleted earlier), and dropping that would be its
            // own kind of drift.
            if bucket_was_new {
                graph.type_indices.remove(&name);
            }
        }
        BucketId::SecondaryLabel(label) => {
            if let Some(members) = graph.secondary_label_index.get_mut(&label) {
                members.retain(|member| *member != idx);
                if bucket_was_new || members.is_empty() {
                    graph.secondary_label_index.remove(&label);
                }
            }
        }
        // The three user-index families share one shape: undo the push,
        // then drop the bucket only if this statement created it. An
        // emptied-but-pre-existing bucket is left in place because the
        // maintenance paths leave one there too, and restoring the
        // pre-statement graph means restoring that as well.
        BucketId::PropertyValue { key, value } => {
            if let Some(value_map) = graph.property_indices.get_mut(&key) {
                undo_bucket_append(value_map.get_mut(&value), idx);
                if bucket_was_new {
                    value_map.remove(&value);
                }
            }
        }
        BucketId::RangeValue { key, value } => {
            if let Some(btree) = graph.range_indices.get_mut(&key) {
                undo_bucket_append(btree.get_mut(&value), idx);
                if bucket_was_new {
                    btree.remove(&value);
                }
            }
        }
        BucketId::CompositeTuple { key, value } => {
            if let Some(comp_map) = graph.composite_indices.get_mut(&key) {
                undo_bucket_append(comp_map.get_mut(&value), idx);
                if bucket_was_new {
                    comp_map.remove(&value);
                }
            }
        }
    }
}

/// Re-insert a node into the inverted-index bucket it was removed from, at the
/// position it held.
fn undo_bucket_removed(graph: &mut DirGraph, bucket: BucketId, idx: NodeIndex, pos: usize) {
    match bucket {
        BucketId::NodeType(name) => {
            let members = graph.type_indices.entry_or_default(name);
            let pos = pos.min(members.len());
            members.insert(pos, idx);
        }
        BucketId::SecondaryLabel(label) => {
            let members = graph.secondary_label_index.entry(label).or_default();
            let pos = pos.min(members.len());
            members.insert(pos, idx);
        }
        // A bucket the statement emptied was dropped from its map by the
        // maintenance path, so re-inserting has to recreate it. If the
        // index itself is gone the entry is skipped: nothing to restore
        // into, and a rolled-back statement never removes an index.
        BucketId::PropertyValue { key, value } => {
            if let Some(value_map) = graph.property_indices.get_mut(&key) {
                let members = value_map.entry_or_default(&value);
                let pos = pos.min(members.len());
                members.insert(pos, idx);
            }
        }
        BucketId::RangeValue { key, value } => {
            if let Some(btree) = graph.range_indices.get_mut(&key) {
                let members = btree.entry_or_default(&value);
                let pos = pos.min(members.len());
                members.insert(pos, idx);
            }
        }
        BucketId::CompositeTuple { key, value } => {
            if let Some(comp_map) = graph.composite_indices.get_mut(&key) {
                let members = comp_map.entry_or_default(&value);
                let pos = pos.min(members.len());
                members.insert(pos, idx);
            }
        }
    }
}

/// Run one edit against a type's live master `ColumnStore` and report the type
/// as touched.
///
/// Both lookups fail silently, and both failures mean the same thing — the
/// type has no name or no store to restore into, so there is nothing to write.
/// (Replay runs *before* the shell restore, so what they read is the failed
/// statement's schema, not the pre-statement one.)
fn edit_column_master(
    graph: &mut DirGraph,
    node_type: InternedKey,
    fallout: &mut ReplayFallout,
    edit: impl FnOnce(&mut ColumnStore),
) {
    let Some(type_name) = graph.interner.try_resolve(node_type).map(str::to_string) else {
        return;
    };
    if let Some(store) = GraphWrite::column_store_mut(&mut graph.graph, node_type) {
        edit(std::sync::Arc::make_mut(store));
    }
    columnar_type_touched(fallout, type_name);
}

/// Truncate a statement's row appends away, dropping the store itself when the
/// statement is what introduced it.
///
/// Rows first, then the columns the append's unseen keys grew: `truncate_rows`
/// shortens every column that still exists, and `restore_schema` then drops the
/// ones that should not exist at all. Both are truncations of a stack, so a
/// statement that appended twice replays as two shrinking steps and lands on
/// the pre-statement length exactly.
///
/// Not expressible through [`edit_column_master`]: the `store_was_new` arm
/// removes the store rather than editing it.
fn undo_columnar_rows_appended(
    graph: &mut DirGraph,
    node_type: InternedKey,
    prior_row_count: u32,
    prior_schema: std::sync::Arc<crate::graph::schema::TypeSchema>,
    prior_column_count: usize,
    store_was_new: bool,
    fallout: &mut ReplayFallout,
) {
    let Some(type_name) = graph.interner.try_resolve(node_type).map(str::to_string) else {
        return;
    };
    if store_was_new {
        GraphWrite::take_column_store(&mut graph.graph, node_type);
    } else if let Some(store) = GraphWrite::column_store_mut(&mut graph.graph, node_type) {
        let store = std::sync::Arc::make_mut(store);
        store.truncate_rows(prior_row_count);
        store.restore_schema(prior_schema, prior_column_count);
    }
    columnar_type_touched(fallout, type_name);
}

/// Report a node type whose master column store the replay just wrote.
///
/// A columnar `SET` produces no `NodeWeight` entry (see the module doc), so
/// the columnar undo entries are the *only* signal that a value under a
/// declared unique constraint may have moved. Without the report, a failed
/// columnar `SET` hits exactly the failure mode `swap_data_scale` warns about
/// for the parked `unique_indices`.
fn columnar_type_touched(fallout: &mut ReplayFallout, type_name: String) {
    fallout.stale_unique_indices.insert(type_name);
}

/// Assert that a re-inserted entity landed on the slot it vacated.
///
/// This is the load-bearing invariant of reverse replay (see
/// `storage/undo.rs`): `StableGraph` frees slots onto a LIFO list, so
/// undoing removals in reverse order must reuse them in order. A mismatch
/// means the invariant broke — a dependency change, or a caller that removed
/// entities outside the journal — and the restored graph's derived indexes
/// would silently point at the wrong entities, so say so loudly in debug and
/// at least once in release.
#[inline]
fn debug_assert_slot_reused(restored: usize, expected: usize, kind: &str) {
    if restored != expected {
        debug_assert_eq!(
            restored, expected,
            "rollback re-inserted a {kind} on slot {restored}, expected {expected}: \
             petgraph's LIFO free-list reuse no longer holds"
        );
        eprintln!(
            "[kglite] statement rollback re-inserted a {kind} on slot {restored} \
             instead of {expected}; derived indexes for that {kind} may be stale. \
             Please report this with the failing query."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Value;
    use petgraph::stable_graph::StableDiGraph;
    use std::collections::HashMap;

    /// The invariant every journal rollback rests on: `StableGraph` hands
    /// freed node slots back LIFO, so undoing removals in reverse order
    /// reuses exactly the vacated slots. Pinned here so a petgraph upgrade
    /// that changes the free-list discipline fails loudly instead of
    /// corrupting rollbacks.
    #[test]
    fn petgraph_reuses_freed_node_slots_lifo() {
        let mut g: StableDiGraph<u32, u32> = StableDiGraph::new();
        let indices: Vec<_> = (0..5).map(|i| g.add_node(i)).collect();
        // Remove in ascending order → free list head is the last removed.
        for &idx in &indices[1..4] {
            g.remove_node(idx);
        }
        // Re-add in reverse removal order → each pops its own slot back.
        for &idx in indices[1..4].iter().rev() {
            let restored = g.add_node(99);
            assert_eq!(
                restored, idx,
                "reverse-order re-insertion must reuse the vacated slot"
            );
        }
    }

    /// Same discipline for edges.
    #[test]
    fn petgraph_reuses_freed_edge_slots_lifo() {
        let mut g: StableDiGraph<u32, u32> = StableDiGraph::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let edges: Vec<_> = (0..4).map(|w| g.add_edge(a, b, w)).collect();
        for &e in &edges[..3] {
            g.remove_edge(e);
        }
        for &e in edges[..3].iter().rev() {
            let restored = g.add_edge(a, b, 99);
            assert_eq!(restored, e, "edge slots must be reused in reverse order");
        }
    }

    fn item(graph: &mut DirGraph, id: i64) -> NodeIndex {
        graph.insert_node_routed(
            Value::Int64(id),
            Value::String(format!("item-{id}")),
            "Item",
            HashMap::new(),
        )
    }

    /// The shell clone must copy the schema half and leave the data half
    /// empty — the property the whole design rests on.
    #[test]
    fn schema_shell_copies_schema_and_parks_data() {
        let mut graph = DirGraph::new();
        let idx = item(&mut graph, 1);
        graph
            .type_indices
            .entry_or_default("Item".to_string())
            .push(idx);
        graph.upsert_node_type_metadata("Item", HashMap::new());

        let shell = graph.schema_shell();

        assert_eq!(shell.graph.node_count(), 0, "the backend must be parked");
        assert!(shell.type_indices.is_empty(), "type_indices must be parked");
        assert!(
            shell.node_type_metadata.contains_key("Item"),
            "schema-scale metadata must be cloned"
        );
        assert_eq!(
            graph.graph.node_count(),
            1,
            "the live graph must be handed its data back"
        );
        assert_eq!(graph.type_indices.len(), 1);
    }

    /// User-created indexes no longer veto the journal: their bucket edits are
    /// journalled with positions, so an indexed graph keeps the cheap
    /// checkpoint instead of paying a whole-graph clone per statement for the
    /// rest of the session.
    #[test]
    fn gate_accepts_user_property_indexes() {
        let mut graph = DirGraph::new();
        assert!(journal_covers(&graph));
        graph
            .property_indices
            .insert(("Item".to_string(), "name".to_string()), Default::default());
        graph
            .range_indices
            .insert(("Item".to_string(), "qty".to_string()), Default::default());
        graph.composite_indices.insert(
            ("Item".to_string(), vec!["name".to_string()]),
            Default::default(),
        );
        assert!(journal_covers(&graph));
    }

    /// Undoing an append must remove the copy the append pushed, not every
    /// copy — a bucket can hold a pre-statement occurrence of the same node.
    #[test]
    fn undoing_an_append_leaves_a_pre_existing_occurrence() {
        let mut members = vec![NodeIndex::new(3), NodeIndex::new(7), NodeIndex::new(3)];
        undo_bucket_append(Some(&mut members), NodeIndex::new(3));
        assert_eq!(members, vec![NodeIndex::new(3), NodeIndex::new(7)]);
    }
}
