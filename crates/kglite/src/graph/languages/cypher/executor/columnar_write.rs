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
use crate::graph::schema::{DirGraph, InternedKey};
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
    /// `node_type`, interned. Resolved by the caller because it is a fact about
    /// the *statement*, not the row: hashing the type name per written row is
    /// what this field removes from a 100k-row `SET`.
    pub(super) type_key: InternedKey,
    pub(super) property: &'a str,
    /// `property`, interned — same reasoning as `type_key`. Interning here also
    /// *registers* the name, which is what lets `save()` resolve the key back
    /// to a string; the caller must therefore hand over a key it obtained from
    /// the graph's `StringInterner`, never a bare `InternedKey::from_str`.
    pub(super) key: InternedKey,
    pub(super) value: &'a Value,
    pub(super) row_id: Option<u32>,
}

/// Whether a master-store write owes its caller the cell's prior value.
///
/// `Skip` is not an optimisation of the *journal* — the undo pre-image is taken
/// either way — it is the read the caller does not consume. `SET` discards it
/// (its index maintenance reads the old value through `node_view`, before the
/// write); `REMOVE` returns it. Reading it regardless cost a `Value` clone per
/// written row, which is an allocation per row for a string property.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PriorCell {
    Skip,
    Read,
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
/// `title`/`name` are still excluded, but no longer because the title lives
/// somewhere else: the fallthrough now writes it through
/// [`GraphWrite::set_node_title`](crate::graph::storage::GraphWrite::set_node_title),
/// which lands in the store's reserved `__title__` column for every title-write
/// path (Cypher `SET`, `add_nodes` update/replace, connection titles). The
/// save-side consolidation chokepoint that used to reconcile an inline override
/// against a stale column — and rebuilt every store to do it (petekSuite bug 2)
/// — has nothing left to reconcile. The exclusion here is a routing detail: a
/// title is not a schema slot, so it cannot go through the cell writer below.
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
    // The keys are the caller's, resolved once per statement instead of per
    // row — so the one thing this path can no longer see for itself is whether
    // they name what is being written. Both are pure functions of the name
    // (FNV-1a), so the check is exact.
    debug_assert_eq!(
        write.key,
        InternedKey::from_str(write.property),
        "the caller's interned key must name the property being written"
    );
    debug_assert_eq!(
        write.type_key,
        InternedKey::from_str(write.node_type),
        "the caller's interned key must name the node type being written"
    );
    write_column_master(
        graph,
        MasterCell {
            node_type: write.node_type,
            type_key: write.type_key,
            node_idx: write.node_idx,
            row_id,
            key: write.key,
            value: write.value,
        },
        PriorCell::Skip,
    )
    .is_some()
}

/// One resolved cell of a master column store — the addressing
/// [`write_column_master`] needs, with every name already interned.
pub(super) struct MasterCell<'a> {
    /// Only the cold path (a property the type has no column for yet) reads
    /// this, to find the declared column type in `node_type_metadata`.
    pub(super) node_type: &'a str,
    pub(super) type_key: InternedKey,
    pub(super) node_idx: NodeIndex,
    pub(super) row_id: u32,
    pub(super) key: InternedKey,
    pub(super) value: &'a Value,
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
/// 2. read the prior cell value again for the caller that asked for one — the
///    journal's copy is consumed by the journal;
/// 3. `Arc::make_mut` and write — **in place**, because nothing else holds the
///    master, which is what makes a one-cell write cost O(1);
/// 4. note the one mutated node for the WAL, since this bypasses the recorded
///    `GraphWrite` path.
///
/// # What is resolved once per statement, and what per row
///
/// The store handle, the type key, the property key and the key's column slot
/// are all facts about the `(type, property)` pair. The row loop used to
/// re-derive all four — a type-name hash, a `column_stores` probe for the
/// declared-type check, a second probe for the write, and two `TypeSchema`
/// lookups — so a 100k-row `SET` paid them 100k times to write one column.
/// What is left per row is the journal capture, one `column_stores` probe, one
/// slot lookup answered through a *shared* borrow, and the cell write. The two
/// `Arc::make_mut` uniqueness checks (store, then column) stay: they are the
/// price of the copy-on-write sharing a fork and a held view rely on, and
/// nothing safe removes them.
pub(super) fn write_column_master(
    graph: &mut DirGraph,
    cell: MasterCell<'_>,
    prior: PriorCell,
) -> Option<Option<crate::datatypes::values::Value>> {
    let MasterCell {
        node_type,
        type_key,
        node_idx,
        row_id,
        key,
        value,
    } = cell;

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

    let master = graph.graph.column_store_mut(type_key)?;
    // The key's column, resolved through the shared borrow — no privatisation,
    // and the answer the write below would otherwise look up again. `None` is
    // the cold path: a property the type has no column for yet, which needs
    // `node_type_metadata` (a different field of `graph`) and therefore cannot
    // run while the store is borrowed.
    let Some(slot) = master.slot(key) else {
        return grow_column_and_write(
            graph, node_type, type_key, node_idx, row_id, key, value, forked, prior,
        );
    };
    let prior_value = match prior {
        // Deliberately `get`, not a slot-addressed read: the full resolution
        // also consults the mmap base and the overflow bag, and a `REMOVE` must
        // report the value a read would have returned.
        PriorCell::Read => master.get(row_id, key), // (2)
        PriorCell::Skip => None,
    };

    // The rollback invariant, asserted at the exact point it matters — and it
    // is the *inverse* of the pre-Phase-2 one. The journal used to hold the
    // master's allocation, so the write had to fork away from it; now the
    // journal holds only the cell's prior value, so a copy here would be a
    // silent whole-store clone per statement (O(rows x cols) to write one cell)
    // with nothing gained.
    //
    // Three holders are legitimate and only these three: a `Forked` backend
    // shares its stores with the base a reader holds; a whole-`DirGraph` clone
    // (`fork_transaction`, the clone checkpoint, a held view) shares them with
    // its twin; and nothing else, because `UndoEntry` has no variant that can
    // hold an `Arc<ColumnStore>` at all — the property this phase's design
    // rests on is enforced by the type, not by this line. What the assert still
    // catches is a *fourth* holder appearing where the store was uniquely
    // owned: the count is read before the write, so a copy taken from a
    // uniquely-owned master fails here loudly. The behavioural gate is the
    // clone counter (`rollback_tests::a_columnar_statement_clones_no_store`).
    let shared = Arc::strong_count(master) > 1;
    let before = Arc::as_ptr(master);
    Arc::make_mut(master).set_at_slot(row_id, slot, value); // (3)
    debug_assert!(
        forked || shared || std::ptr::eq(before, Arc::as_ptr(master)),
        "a columnar write copied a master that nothing else was holding; \
         `Arc::make_mut` on a uniquely-owned handle must mutate in place"
    );
    graph.graph.note_recorded_node_upsert(node_idx); // (4)
    Some(prior_value)
}

/// The cold half of [`write_column_master`]: the type's store has no column for
/// `key` yet, so the write grows the schema.
///
/// Split out because the declared column type lives in `node_type_metadata` —
/// a field of `graph` the store borrow excludes — and because it runs once per
/// `(type, property)` in a statement's lifetime, never in the row loop. The
/// *outer* `None` keeps its one meaning, "the type has no master store".
///
/// A missing column does **not** mean a missing value, which is why this path
/// still honours a `PriorCell::Read`: on a mapped graph the value can live in
/// the store's mmap base or its overflow bag, both of which `get` resolves and
/// neither of which has a dense column until something writes one. Dropping
/// that read would make `REMOVE n.x` report nothing removed — and skip the
/// index eviction — for exactly the properties a `.kgl` load leaves there.
///
/// Declared metadata wins over the value in hand, because it knows `float64`
/// when the first value that happens to arrive is an integer — and a column
/// typed wrong is a column the next write demotes to `Mixed`, which cannot be
/// spilled.
#[allow(clippy::too_many_arguments)]
fn grow_column_and_write(
    graph: &mut DirGraph,
    node_type: &str,
    type_key: InternedKey,
    node_idx: NodeIndex,
    row_id: u32,
    key: InternedKey,
    value: &Value,
    forked: bool,
    prior: PriorCell,
) -> Option<Option<Value>> {
    let declared_type: Option<String> = graph.interner.try_resolve(key).and_then(|name| {
        graph
            .node_type_metadata
            .get(node_type)
            .and_then(|props| props.get(name))
            .cloned()
    });
    let master = graph.graph.column_store_mut(type_key)?;
    let prior_value = match prior {
        PriorCell::Read => master.get(row_id, key),
        PriorCell::Skip => None,
    };
    let shared = Arc::strong_count(master) > 1;
    let before = Arc::as_ptr(master);
    Arc::make_mut(master).set(row_id, key, value, declared_type.as_deref());
    debug_assert!(
        forked || shared || std::ptr::eq(before, Arc::as_ptr(master)),
        "a columnar write copied a master that nothing else was holding; \
         `Arc::make_mut` on a uniquely-owned handle must mutate in place"
    );
    graph.graph.note_recorded_node_upsert(node_idx);
    Some(prior_value)
}
