//! Statement-level rollback: a cheap checkpoint plus undo-journal restore.
//!
//! One mutating Cypher statement is atomic. It can fail *after* its first
//! write (a constraint violation on row 40 of a `CREATE`, a write-scope
//! rejection, an expression error), and the graph must then look exactly as
//! it did before the statement ran.
//!
//! ## Two ways to be atomic
//!
//! The original mechanism was a whole-graph clone taken before every such
//! statement and swapped back on failure. It is unconditionally correct and
//! unconditionally O(V+E) — writing one property to a million-node graph
//! deep-copied a million `NodeData`s first.
//!
//! This module keeps that clone as a fallback and adds the cheap path:
//!
//! | | cost to open | cost to roll back |
//! |---|---|---|
//! | [`StatementCheckpoint::None`] | nothing | n/a — the shape cannot fail after its first write |
//! | [`StatementCheckpoint::Journal`] | O(schema) + O(changes) | O(changes) + reindex of affected types |
//! | [`StatementCheckpoint::Clone`] | O(V+E) | O(1) swap |
//!
//! ## How the journal path splits `DirGraph`'s state
//!
//! `DirGraph` has ~44 fields. The split is by *clone cost*, and the
//! bias is deliberate:
//!
//! - **O(schema)-sized fields are cloned** into a `shell` and restored
//!   verbatim. That is the whole schema surface a statement can grow —
//!   `interner`, `node_type_metadata`, `connection_type_metadata`,
//!   `type_schemas`, `connection_types`, `has_secondary_labels`, the index
//!   *key* lists, aliases, configs, and `version`. It is also every field
//!   nobody has thought about yet, which is the point: **a field that is not
//!   explicitly parked is automatically restored**.
//! - **O(V+E)-sized fields are parked** (moved aside so `clone()` sees them
//!   empty) and reconstructed from the journal. This list is short, closed,
//!   and each member has a documented undo story in [`apply`]:
//!   `graph`, `type_indices`, `id_indices`, `property_indices`,
//!   `composite_indices`, `range_indices`, `secondary_label_index`,
//!   `embeddings`, `timeseries_store`, `unique_indices`.
//!
//! Two parked fields are *rebuilt* rather than journal-reversed, because their
//! own mutation sites record no inverse: `id_indices` (dropped per touched
//! type, self-healing on the next read) and `unique_indices` (recomputed per
//! touched type by [`DirGraph::rebuild_unique_indices_for_types`]). Both are
//! driven by the [`ReplayFallout`] the replay collects. Parking a field with no
//! undo story of *either* kind is the one way to make this module wrong — see
//! the `unique_indices` note on [`swap_data_scale`].
//!
//! Getting that split wrong fails in the safe direction. Forgetting to park a
//! new large field makes the checkpoint *slower*, never wrong; only an
//! explicit addition to [`swap_data_scale`] can introduce a correctness gap,
//! and that is a change a reviewer is looking straight at.
//!
//! ## When the journal is not used
//!
//! [`journal_covers`] is the gate. It is conservative on purpose: any shape
//! whose derived state the journal does not yet reverse keeps the clone, so
//! correctness never depends on the gate being generous.
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

use super::DirGraph;
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
    std::mem::swap(&mut a.property_indices, &mut b.property_indices);
    std::mem::swap(&mut a.composite_indices, &mut b.composite_indices);
    std::mem::swap(&mut a.range_indices, &mut b.range_indices);
    std::mem::swap(&mut a.secondary_label_index, &mut b.secondary_label_index);
    std::mem::swap(&mut a.embeddings, &mut b.embeddings);
    std::mem::swap(&mut a.timeseries_store, &mut b.timeseries_store);
    // Parked because its inner map holds one entry per node of a constrained
    // type, so cloning it would put an O(constrained-nodes) copy back on every
    // mutating statement — the cost this module exists to remove.
    //
    // Parking alone would be a *correctness* bug, not just a cheaper one:
    // `commit_unique_claims` / `release_unique_claims` /
    // `evict_unique_claims_for_nodes` record nothing in the journal, and
    // because parking keeps the live post-statement map (rather than restoring
    // the shell's), a failed statement's claims would survive rollback — a
    // phantom occupant (permanent spurious `ConstraintViolationError`) or a
    // released claim (a real duplicate admitted). The paired undo story is the
    // per-touched-type rebuild in `StatementCheckpoint::rollback`; do not park
    // this field without it.
    std::mem::swap(&mut a.unique_indices, &mut b.unique_indices);
}

/// Whether the undo journal can reverse everything this graph's mutations can
/// change. Conservative by construction — every `false` arm falls back to the
/// clone checkpoint, so a shape that is merely *unproven* is still safe.
fn journal_covers(graph: &DirGraph) -> bool {
    // Only the heap backend can express an inverse petgraph edit.
    graph.graph.supports_undo_journal()
        // Columnar `SET` writes the shared per-type `Arc<ColumnStore>` through
        // a side channel that bypasses `GraphWrite` entirely
        // (`cypher::executor::write`'s columnar-master fast path), so the
        // journal never sees the pre-image.
        && graph.column_stores.is_empty()
        // User-created indexes need per-bucket position undo on the delete
        // path, which the journal does not record yet. Empty in the default
        // graph — `create_index` opts in. Follow-up, tracked in the sprint
        // report.
        && graph.property_indices.is_empty()
        && graph.composite_indices.is_empty()
        && graph.range_indices.is_empty()
    // Deliberately *not* gated on `unique_indices.is_empty()`. It reads as
    // doctrine-consistent but is a pessimisation: it would route every
    // constrained graph to `fork_transaction()`, an O(V+E) clone of everything,
    // which is strictly worse than the journal plus a per-touched-type unique
    // rebuild. The rebuild is that field's undo story.
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
    /// Clone only the O(schema) half of this graph.
    ///
    /// Implemented by parking the O(V+E) fields in a throwaway husk, cloning
    /// what is left, and handing them straight back — so the returned shell
    /// has empty data-scale fields and a faithful copy of everything else.
    fn schema_shell(&mut self) -> DirGraph {
        let mut husk = DirGraph::new();
        swap_data_scale(self, &mut husk);
        let shell = self.clone();
        swap_data_scale(self, &mut husk);
        shell
    }

    /// Adopt `shell`'s O(schema) state while keeping the live O(V+E) state.
    fn restore_schema_shell(&mut self, mut shell: DirGraph) {
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
    /// Node types whose id-index the replay perturbed. `IdIndexStore` has no
    /// per-entry removal, and its read path self-heals via `lookup_or_build`,
    /// so whole-type invalidation is both the cheapest correct move and
    /// exactly what the delete path already does.
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
        UndoEntry::NodeAdded { idx, node_type } => {
            // `type_indices` is reversed by the `BucketAppended` entry the
            // create path recorded; only the derived per-type structures need
            // invalidating here.
            fallout.node_identity_changed(graph.interner.resolve(node_type).to_string());
            // A node created by this statement can only carry edges this
            // statement created, and those replayed first (they were captured
            // later), so it is isolated by now.
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
        UndoEntry::NodeWeight { idx, prior } => {
            // Both spellings of the type, so a claim under either is recomputed.
            // The primary type is immutable in practice, but reading it costs
            // one interner lookup on a path that only runs when a statement
            // already failed.
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
        UndoEntry::NodeRemoved { idx, prior } => {
            let type_name = graph.interner.resolve(prior.node_type).to_string();
            let restored = GraphWrite::add_node(&mut graph.graph, prior);
            debug_assert_slot_reused(restored.index(), idx.index(), "node");
            fallout.node_identity_changed(type_name);
        }
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
        } => match bucket {
            BucketId::NodeType(name) => {
                graph
                    .type_indices
                    .retain_in_type(&name, |member| *member != idx);
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
        },
        UndoEntry::BucketRemoved { bucket, idx, pos } => match bucket {
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
        },
        UndoEntry::TimeseriesRemoved { node, prior } => {
            graph.timeseries_store.insert(node, *prior);
        }
    }
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
    use petgraph::graph::NodeIndex;
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

    /// A graph with user-created property indexes must keep the clone
    /// checkpoint until the journal learns to reverse them.
    #[test]
    fn gate_rejects_user_property_indexes() {
        let mut graph = DirGraph::new();
        assert!(journal_covers(&graph));
        graph
            .property_indices
            .insert(("Item".to_string(), "name".to_string()), HashMap::new());
        assert!(!journal_covers(&graph));
    }
}
