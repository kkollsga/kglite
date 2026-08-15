//! Unique-constraint claims across a rollback
//!
//! `unique_indices` is parked by `swap_data_scale`, so a journal rollback keeps
//! the *failed statement's* occupancy map while the data underneath is restored.
//! Its undo story is the per-touched-type rebuild in
//! `StatementCheckpoint::rollback`. These tests pin both directions of getting
//! that wrong:
//!
//! - a **phantom claim** — a value the failed statement claimed stays occupied,
//!   so a later legitimate insert is rejected forever;
//! - a **lost claim** — a value the failed statement released stays free, so a
//!   real duplicate is admitted on the next write.
//!
//! The declared tuple is `Item.name`, an explicit non-`id` property, on purpose:
//! `declared_unique_tuples` skips `primary_key == "id"`, so a constraint over
//! `id` leaves `unique_indices` empty and would make these tests vacuous.

use super::*;

/// One constraint's claimed values as `(value, holding slot)`, sorted.
type UniqueClaims = Vec<(String, usize)>;

/// `(node_type, constraint properties, claims)` per declared constraint.
pub(super) type UniqueFingerprint = Vec<(String, Vec<String>, UniqueClaims)>;

/// The whole occupancy map, per declared constraint: constraint tuple → the
/// claimed values and the slot holding each. Slot-level so a claim that comes
/// back pointing at the wrong node is a failure, not just a missing one.
pub(super) fn unique_fingerprint(graph: &DirGraph) -> UniqueFingerprint {
    let mut out: Vec<_> = graph
        .unique_indices
        .iter()
        .map(|((node_type, properties), occupants)| {
            let mut claims: UniqueClaims = occupants
                .iter()
                .map(|(value, idx)| (format!("{value:?}"), idx.index()))
                .collect();
            claims.sort();
            (node_type.clone(), properties.clone(), claims)
        })
        .collect();
    out.sort();
    out
}

/// `seeded()` plus a declared UNIQUE constraint over `Item.name`.
///
/// Asserts the graph still takes the journal path: declaring a unique
/// constraint touches only `unique_indices` / `unique_constraint_keys`, never
/// `property_indices`, so `journal_covers` stays true. If that ever changes,
/// these tests would silently start exercising the clone path and prove
/// nothing — so the precondition is checked, not assumed.
pub(super) fn seeded_with_unique_name() -> DirGraph {
    let mut graph = seeded();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (i:Item) REQUIRE i.name IS UNIQUE",
    );
    assert_eq!(
        graph.unique_indices.len(),
        1,
        "the constraint must be declared and enforcing"
    );
    assert!(
        graph.property_indices.is_empty()
            && graph.composite_indices.is_empty()
            && graph.range_indices.is_empty(),
        "a unique constraint must not create a user index, or these tests \
         would exercise the clone checkpoint instead of the journal"
    );
    graph
}

/// `seeded()` with a UNIQUE constraint over `Item.qty`, then saved.
///
/// `qty` rather than `name` because the columnar fast path deliberately skips
/// `name`/`title` (they fall through to the inline node setter), so a
/// constraint over `name` would exercise the ordinary journalled write and
/// prove nothing about the master side channel. The seeded `qty` values are
/// distinct, so the constraint is satisfiable.
fn seeded_columnar_with_unique_qty() -> DirGraph {
    let mut graph = seeded();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (i:Item) REQUIRE i.qty IS UNIQUE",
    );
    graph.enable_columnar();
    assert_eq!(
        graph.unique_indices.len(),
        1,
        "the constraint must be declared and enforcing"
    );
    assert!(
        graph.column_store_count() > 0,
        "the graph must be saved, or this is the plain unique fixture again"
    );
    graph
}

/// A unique claim moved by a *columnar* `SET` must come back.
///
/// This is the shape with no `NodeWeight` entry behind it: the value goes into
/// the master column store, so the node's weight never changes and the journal
/// sees only `UndoEntry::ColumnarCell` entries. Their replay is what reports
/// the type into `stale_unique_indices`, and that report is the only thing
/// driving the per-type unique rebuild. It has survived two rewrites of the
/// mechanism underneath (the per-node handle sweep, then the per-type store
/// pre-image, now the per-cell value) precisely because this test measures the
/// claim rather than the mechanism: without the report a failed columnar `SET`
/// leaves the claim it took behind and the claim it released free.
#[test]
fn rollback_restores_claims_moved_by_a_columnar_property_overwrite() {
    let mut graph = seeded_columnar_with_unique_qty();
    let before = unique_fingerprint(&graph);
    assert!(
        !before.is_empty() && !before[0].2.is_empty(),
        "the constraint must hold claims, or this test is vacuous"
    );

    let error = expect_failure(
        &mut graph,
        "MATCH (i:Item {id: 1}) SET i.qty = 999 \
         WITH i MATCH (j:Item {id: 2}) SET j.bad = duration({months: 2147483648})",
        None,
    );

    assert_eq!(
        unique_fingerprint(&graph),
        before,
        "a claim moved through the master column store must move back.\
         \nerror: {error}"
    );
    // The observable half: 10 is claimed again (no lost claim) and 999 is free
    // (no phantom occupant).
    expect_failure(&mut graph, "CREATE (:Item {id: 40, qty: 10})", None);
    run(&mut graph, "CREATE (:Item {id: 41, qty: 999})");
}

/// A statement that claims a new value and *then* fails must not leave the
/// claim behind. Without the rebuild, `'zeta'` stays occupied by a node that
/// no longer exists and every later insert of it is rejected forever.
#[test]
fn rollback_releases_a_claim_the_failed_statement_added() {
    let mut graph = seeded_with_unique_name();
    let before = unique_fingerprint(&graph);

    // First `CREATE` claims 'zeta'; the second collides with 'b' (held by the
    // Item seeded with id 2), so the statement fails after its first write.
    let error = expect_failure(
        &mut graph,
        "CREATE (:Item {id: 10, name: 'zeta'}), (:Item {id: 11, name: 'b'})",
        None,
    );

    assert_eq!(
        unique_fingerprint(&graph),
        before,
        "the rolled-back claim must be gone.\nerror: {error}"
    );
    // The observable half: 'zeta' is insertable, which it would not be if the
    // phantom claim survived.
    run(&mut graph, "CREATE (:Item {id: 13, name: 'zeta'})");
}

/// A statement that *releases* a claim by deleting its holder and then fails
/// must put the claim back. Without the rebuild, `'a'` stays free and a real
/// duplicate is admitted on the next write.
#[test]
fn rollback_restores_a_claim_the_failed_statement_released() {
    let mut graph = seeded_with_unique_name();
    let before = unique_fingerprint(&graph);

    let error = expect_failure(
        &mut graph,
        "MATCH (i:Item {id: 1}) DETACH DELETE i CREATE (:Item {id: 20, name: 'b'})",
        None,
    );

    assert_eq!(
        unique_fingerprint(&graph),
        before,
        "the released claim must be restored, pointing at the restored slot.\
         \nerror: {error}"
    );
    // The observable half: 'a' is claimed again, so a duplicate is refused.
    expect_failure(&mut graph, "CREATE (:Item {id: 21, name: 'a'})", None);
}

/// The `NodeWeight`-only shape: a property overwrite moves a claim from the old
/// value to the new one without touching identity, so it is invisible to
/// `stale_id_indices` and needs `stale_unique_indices` to carry it.
#[test]
fn rollback_restores_claims_moved_by_a_property_overwrite() {
    let mut graph = seeded_with_unique_name();
    let before = unique_fingerprint(&graph);

    let error = expect_failure(
        &mut graph,
        "MATCH (i:Item {id: 1}) SET i.name = 'renamed' \
         WITH i MATCH (j:Item {id: 2}) SET j.bad = duration({months: 2147483648})",
        None,
    );

    assert_eq!(
        unique_fingerprint(&graph),
        before,
        "an overwritten claim must move back.\nerror: {error}"
    );
    // 'a' is claimed again (no lost claim) and 'renamed' is free (no phantom).
    expect_failure(&mut graph, "CREATE (:Item {id: 30, name: 'a'})", None);
    run(&mut graph, "CREATE (:Item {id: 31, name: 'renamed'})");
}
