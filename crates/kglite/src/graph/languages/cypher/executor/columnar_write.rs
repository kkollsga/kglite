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
/// # Journals, and why the capture must come first
///
/// - **WAL**: the write bypasses the recorded `GraphWrite` path, so the one
///   mutated node is captured explicitly. Exactly one node — there is no
///   end-of-clause sweep any more, because no node holds a store handle to
///   re-point (D1 Phase 3).
/// - **Undo**: the pre-statement master is captured once per type as
///   [`UndoEntry::ColumnarHandles`], **before** the write. That ordering is
///   load-bearing and is the whole rollback correctness argument:
///   `Arc::make_mut` below forks *only* when someone else holds a handle, so
///   the journal's clone is what keeps the pre-statement image pristine. With
///   no checkpoint open nobody else holds one, `make_mut` mutates the single
///   row in place, and the statement costs O(1) instead of O(N_type).
///
/// Both directions are asserted rather than argued — see the `debug_assert!`
/// at the write itself and
/// `column_ownership_tests::the_master_is_uniquely_owned_between_statements`.
pub(super) fn set_via_column_master(graph: &mut DirGraph, write: ColumnMasterWrite<'_>) -> bool {
    // Disk-backed graphs use a separate write path; the master `column_stores`
    // Arc is for the in-memory Columnar mode only.
    let Some(row_id) = write.row_id else {
        return false;
    };
    if graph.graph.is_disk() || write.property == "title" || write.property == "name" {
        return false;
    }
    let key = graph.interner.get_or_intern(write.property);
    write_column_master(
        graph,
        write.node_type,
        write.node_idx,
        row_id,
        key,
        write.value,
    )
    .is_some()
}

/// Write one cell of a type's master column store, with the journalling both
/// halves of the engine need.
///
/// The single place a columnar property value is written. Returns the prior
/// value (`None` when the type has no store, which is also the "nothing was
/// written" signal).
///
/// Ordering is the correctness argument, not a detail:
/// 1. clone the master into the undo journal — under an open checkpoint that
///    clone is what makes step 3 fork instead of mutating the pre-statement
///    image;
/// 2. read the prior cell value, for the caller's result and for index
///    maintenance;
/// 3. `Arc::make_mut` and write — in place when nothing else holds a handle,
///    which after D1 Phase 3 is the normal case between statements;
/// 4. note the one mutated node for the WAL, since this bypasses the recorded
///    `GraphWrite` path.
pub(super) fn write_column_master(
    graph: &mut DirGraph,
    node_type: &str,
    node_idx: NodeIndex,
    row_id: u32,
    key: crate::graph::schema::InternedKey,
    value: &crate::datatypes::values::Value,
) -> Option<Option<crate::datatypes::values::Value>> {
    let type_key = crate::graph::schema::InternedKey::from_str(node_type);

    // (1) Pre-image first. If the journal already holds an entry for this type
    // it declines the clone, which is then dropped — returning the count to one
    // so the rest of the statement mutates in place.
    let prior_store = graph.graph.column_store(type_key).map(Arc::clone);
    let captured_now = match (prior_store, graph.graph.undo_journal_mut()) {
        (Some(prior_store), Some(journal)) => {
            journal.note_columnar_fork(type_key, || Some(prior_store))
        }
        _ => false,
    };

    let master = graph.graph.column_store_mut(type_key)?;
    let prior_value = master.get(row_id, key); // (2)
                                               // The rollback invariant, asserted at the exact point it matters.
                                               //
                                               // `captured_now` means *this* call handed the journal the allocation the
                                               // master currently points at — so `make_mut` must fork away from it, or the
                                               // statement is mutating the very image rollback would restore. Silent, and
                                               // exactly the class `rollback.rs::swap_data_scale` warns about.
                                               //
                                               // Later writes in the same statement do *not* fork: the journal declined a
                                               // second clone, so the master is uniquely owned and mutates in place. That
                                               // is correct (the first write already moved it off the pre-image) and is
                                               // why the assertion is keyed on `captured_now` rather than on whether a
                                               // checkpoint is open.
    let before = Arc::as_ptr(master);
    Arc::make_mut(master).set(row_id, key, value, None); // (3)
    debug_assert!(
        !captured_now || !std::ptr::eq(before, Arc::as_ptr(master)),
        "the first columnar write of a statement must fork the master away \
         from the pre-image just handed to the undo journal; mutating it in \
         place leaves rollback nothing pristine to restore"
    );
    graph.graph.note_recorded_node_upsert(node_idx); // (4)
    Some(prior_value)
}
