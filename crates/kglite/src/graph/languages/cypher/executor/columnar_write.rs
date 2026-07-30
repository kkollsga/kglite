//! The in-memory columnar master-store write path.
//!
//! Extracted from `write.rs` rather than left in place: `write.rs` reached its
//! 2500-line ceiling, and project doctrine is to split a file that outgrows it
//! rather than raise the allowlist. These three items are the natural seam --
//! they are the only ones that reach the per-type master `Arc<ColumnStore>`,
//! and `execute_set` / `execute_remove` must agree on all of them (a property
//! removed on a node's forked store while the master keeps the value is
//! resurrected by the next clause's handle sweep).

use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::NodeIndex;
use std::sync::Arc;

/// One property write aimed at a node type's master column store.
///
/// `row_id` is `None` for a node whose properties are not `Columnar`, which is
/// one of the fallthrough conditions rather than a caller error — carrying the
/// `Option` here keeps the whole "can this go through the master?" decision in
/// one place.
pub(super) struct ColumnMasterWrite<'a> {
    pub(super) node_idx: NodeIndex,
    pub(super) node_type: &'a str,
    pub(super) property: &'a str,
    pub(super) value: &'a Value,
    pub(super) row_id: Option<u32>,
}

/// Write one property through the in-memory columnar master store, reporting
/// whether it landed there.
///
/// The fast path for `Columnar` storage: route the write through the per-type
/// master `Arc<ColumnStore>` once, instead of through each node's own handle.
/// Every node of the type points at the same allocation, so `Arc::make_mut` on
/// a *node's* handle would clone the whole store on every write — O(N²) for a
/// batch `SET`. Going through the master forks once; the per-node handles are
/// re-pointed in a single sweep at the end of the clause.
///
/// Returns `false` — leaving the caller to fall through to the per-node setter
/// — for disk-backed graphs (which have their own write path), for non-
/// `Columnar` nodes, for a `Columnar` node whose type is absent from
/// `column_stores`, and for `title`/`name`.
///
/// `title`/`name` are excluded deliberately: the fallthrough sets the inline
/// `node.title`, and `enable_columnar` detects that inline override on save
/// (it differs from the stale `__title__` column) and rebuilds, consolidating
/// the fresh title. That single save-side chokepoint covers every title-write
/// path — Cypher `SET`, `add_nodes` update/replace, connection titles —
/// without per-path master writes (petekSuite bug 2).
///
/// The property name is interned into the graph's `StringInterner` *before*
/// `column_stores` is borrowed. The per-node path gets this free via
/// `node.set_property(…, &mut graph.interner)`; the master path once used
/// `InternedKey::from_str()`, which only hashes, leaving `save()` unable to
/// resolve the key back to a string at serialize time. Symptom: every
/// Cypher-`SET` property on a 0.8.39 in-memory Sodir-scale graph survived
/// in-memory but vanished after save+load, with
/// `BUG: InternedKey N not found in StringInterner`.
///
/// Both journals are handled here, and they pull in opposite directions:
///
/// - **WAL**: the write bypasses the recorded `GraphWrite` path, so the one
///   mutated node is captured explicitly. The end-of-clause refresh sweep must
///   *not* be, or a single `SET` would log every node of the type.
/// - **Undo**: the pre-statement master is captured once per type, as
///   [`UndoEntry::ColumnarHandles`], which is what lets the refresh sweep skip
///   per-node capture entirely. The store itself is never copied — the fork
///   already left the pre-statement one pristine, so holding its handle is the
///   whole pre-image. `touched_columnar_types` keeps this to one attempt per
///   type per clause; the journal's own first-touch rule covers a later clause
///   writing the same type again.
pub(super) fn set_via_column_master(
    graph: &mut DirGraph,
    write: ColumnMasterWrite<'_>,
    touched_columnar_types: &mut std::collections::HashSet<String>,
) -> bool {
    // Disk-backed graphs use a separate write path; the master `column_stores`
    // Arc is for the in-memory Columnar mode only.
    let Some(row_id) = write.row_id else {
        return false;
    };
    if graph.graph.is_disk() || write.property == "title" || write.property == "name" {
        return false;
    }
    let key = graph.interner.get_or_intern(write.property);
    if !touched_columnar_types.contains(write.node_type) {
        if let Some(prior) = graph.column_stores.get(write.node_type).map(Arc::clone) {
            if let Some(journal) = graph.graph.undo_journal_mut() {
                journal.note_columnar_fork(write.node_type, || Some(prior));
            }
        }
    }
    let Some(master) = graph.column_stores.get_mut(write.node_type) else {
        return false;
    };
    // Did `make_mut` actually fork, or did it mutate in place?
    //
    // Only a fork leaves the per-node handles stale, and only stale handles
    // need the O(N) end-of-clause sweep. Comparing the allocation address
    // across the call is the exact question — `Arc::strong_count` is not, since
    // it cannot distinguish "nobody else holds this" from "the clone already
    // happened".
    //
    // TODAY THIS IS A PROVABLE NO-OP, and that is the point of landing it
    // alone. Every node of a type holds its own strong handle, so the master's
    // count is `1 (map) + N (nodes)` at the first write of a clause and
    // `make_mut` always forks; the second and later writes in the same clause
    // find the fresh allocation uniquely owned and mutate in place, but the
    // type is already in the set by then. So the set ends up identical either
    // way, which `fork_detection_is_a_no_op_while_nodes_hold_strong_handles`
    // pins.
    //
    // It stops being a no-op the moment nodes stop holding strong handles: then
    // most writes mutate in place, and an unconditional insert would keep
    // paying a sweep that has nothing to re-point.
    let before = Arc::as_ptr(master);
    Arc::make_mut(master).set(row_id, key, write.value, None);
    if !std::ptr::eq(before, Arc::as_ptr(master)) {
        touched_columnar_types.insert(write.node_type.to_string());
    }
    graph.graph.note_recorded_node_upsert(write.node_idx);
    true
}

/// Re-point every node's `Arc<ColumnStore>` handle at the graph master, for
/// each type written through the master during this clause.
///
/// Each node holds its own Arc clone for efficient property reads; after a
/// clause wrote through the graph master those per-node handles are stale and
/// would surface pre-clause values. This sweep is O(N) per touched type and
/// runs once per clause regardless of row count — which is the whole reason
/// the master fast paths accumulate a type set instead of refreshing per row.
///
/// Shared by `execute_set` and `execute_remove`, which must agree: a node
/// whose property was removed on its own forked store while the master kept
/// the value has the value resurrected by the *next* clause's sweep.
pub(super) fn refresh_columnar_node_handles(
    graph: &mut DirGraph,
    touched_columnar_types: std::collections::HashSet<String>,
) {
    for node_type in touched_columnar_types {
        let new_master = match graph.column_stores.get(&node_type) {
            Some(m) => Arc::clone(m),
            None => continue,
        };
        let indices: Vec<NodeIndex> = graph
            .type_indices
            .get(&node_type)
            .map(|s| s.iter().collect())
            .unwrap_or_default();
        for idx in indices {
            // `_silent`: re-pointing per-node Arc handles is internal storage
            // bookkeeping, not a logical mutation — must not be captured by the
            // WAL recorder (the actual write was recorded in the fast path).
            if let Some(node) = GraphWrite::node_weight_mut_silent(&mut graph.graph, idx) {
                if let crate::graph::schema::PropertyStorage::Columnar { store, .. } =
                    &mut node.properties
                {
                    *store = Arc::clone(&new_master);
                }
            }
        }
    }
}
