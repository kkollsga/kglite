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
use crate::graph::storage::undo::{ColumnarPreImages, ColumnarWrite};
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
/// - **Undo**: the cell's prior value is captured, **before** the write, as
///   [`UndoEntry::ColumnarCell`]. That ordering is load-bearing — after the
///   write the prior value is gone — but it no longer forces a copy of
///   anything: the journal holds an `Option<Value>`, never a handle on the
///   store, so the master stays uniquely owned and `Arc::make_mut` below
///   mutates one cell in place whether or not a checkpoint is open.
///
/// The uniquely-owned half is asserted rather than argued — see the
/// `debug_assert!` at the write itself and
/// `dir_graph::rollback_tests::the_master_is_uniquely_owned_between_statements`.
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
/// 1. read the cell's prior value (and, if the write introduces a new
///    property, the pre-growth schema) into the undo journal — after step 3
///    the prior value no longer exists anywhere;
/// 2. read the prior cell value again for the caller's result and for index
///    maintenance — the journal's copy is consumed by the journal;
/// 3. `Arc::make_mut` and write — **in place**, because nothing else holds the
///    master, which is what makes a one-cell write cost O(1);
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

    // (1) Pre-image first, and cell-grained: the journal takes the value this
    // write is about to destroy, never a handle on the store holding it.
    //
    // Gated on the journal actually existing: an unjournalled statement (no
    // open checkpoint) must not pay the store probe for a pre-image nobody
    // will read.
    if graph.graph.undo_journal_mut().is_some() {
        let captured = graph
            .graph
            .column_store(type_key)
            .map(|store| ColumnarPreImages::capture(store, row_id, ColumnarWrite::Cell(key)));
        if let (Some(captured), Some(journal)) = (captured, graph.graph.undo_journal_mut()) {
            captured.record(journal, type_key, row_id);
        }
    }

    // Read before the mutable borrow: a forked backend shares its stores with
    // the base a reader is holding, so its first write per type legitimately
    // copies (`storage/forked.rs`). Everywhere else the copy would be the
    // defect this design removed.
    let forked = graph.graph.is_forked();

    // Column typing for a property the store has no column for yet. Declared
    // metadata wins over the value in hand, because it knows `float64` when the
    // first value that happens to arrive is an integer — and a column typed
    // wrong is a column the next write demotes to `Mixed`, which cannot be
    // spilled. Resolved only on that cold path: the steady-state SET writes an
    // existing column and never pays the lookup.
    let declared_type: Option<String> = {
        let needs_column = graph
            .graph
            .column_store(type_key)
            .is_some_and(|store| store.slot(key).is_none());
        if needs_column {
            graph.interner.try_resolve(key).and_then(|name| {
                graph
                    .node_type_metadata
                    .get(node_type)
                    .and_then(|props| props.get(name))
                    .cloned()
            })
        } else {
            None
        }
    };
    let master = graph.graph.column_store_mut(type_key)?;
    let prior_value = master.get(row_id, key); // (2)

    // The rollback invariant, asserted at the exact point it matters — and it
    // is the *inverse* of the pre-Phase-2 one. The journal used to hold the
    // master's allocation, so the write had to fork away from it; now the
    // journal holds only the cell's prior value, so a fork here would be a
    // silent whole-store copy per statement (O(rows x cols) to write one cell)
    // with nothing gained. On a non-forked backend the master is the backend's
    // alone and the pointer must survive `make_mut` unchanged.
    let before = Arc::as_ptr(master);
    Arc::make_mut(master).set(row_id, key, value, declared_type.as_deref()); // (3)
    debug_assert!(
        forked || std::ptr::eq(before, Arc::as_ptr(master)),
        "a columnar write on a non-forked backend must mutate the master in \
         place; a fork here means something is holding a second handle and \
         every statement is paying a whole-store copy"
    );
    graph.graph.note_recorded_node_upsert(node_idx); // (4)
    Some(prior_value)
}
