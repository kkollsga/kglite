//! Cell-grained rollback fidelity
//!
//! The arms below exercise the three shapes a cell pre-image has to cover that
//! the whole-store pre-image covered for free: a write that grew the schema, a
//! cell written more than once, and a cell that was absent to begin with. Each
//! one is a way for `UndoEntry::ColumnarCell` / `ColumnarSchemaGrown` to be
//! individually wrong while every pre-existing arm stays green.

use super::*;

/// A failed `SET` that introduced a **new** property must leave the type's
/// column store exactly as wide as it was.
///
/// `ColumnStore::set` grows the schema and pushes a null-backfilled column when
/// it meets an unknown key, and that growth is not a cell edit — restoring
/// every cell would leave the empty column and the widened schema behind. The
/// `fingerprint` oracle cannot see it (`row_properties` skips nulls, so an
/// all-null column is invisible through every read surface), which is why the
/// column count and the schema slot are asserted directly: this is the one
/// residue of a columnar write that is real, persisted at the next `save()`,
/// and unobservable through values.
#[test]
fn rollback_removes_a_column_a_failed_set_introduced() {
    let fresh = crate::graph::schema::InternedKey::from_str("fresh");
    let mut graph = wide_columnar();
    let columns_before = graph.column_store("Item").expect("master").column_count();
    assert!(
        graph
            .column_store("Item")
            .expect("master")
            .slot(fresh)
            .is_none(),
        "precondition: the fixture must not already have the property"
    );
    let before = fingerprint(&mut graph);

    expect_failure(
        &mut graph,
        &format!("MATCH (n:Item {{id: 1}}) SET n.fresh = 7 {FAILS_AFTER_A_COLUMNAR_WRITE}"),
        None,
    );

    let store = graph.column_store("Item").expect("master");
    assert_eq!(
        store.column_count(),
        columns_before,
        "a failed SET left the column it appended behind; the schema-growth \
         undo entry is missing or truncating to the wrong width"
    );
    assert!(
        store.slot(fresh).is_none(),
        "a failed SET left the property in the type schema, so the next write \
         resolves a slot that no longer has a column"
    );
    assert_eq!(item_prop(&graph, 1, "fresh"), None);
    assert_eq!(fingerprint(&mut graph), before);
}

/// The non-vacuity control: the *same* statement, committed, really does grow
/// the store — so the arm above is measuring an undo and not a write that
/// never happened.
#[test]
fn a_committed_set_of_a_new_property_grows_the_store() {
    let fresh = crate::graph::schema::InternedKey::from_str("fresh");
    let mut graph = wide_columnar();
    let columns_before = graph.column_store("Item").expect("master").column_count();

    run(&mut graph, "MATCH (n:Item {id: 1}) SET n.fresh = 7");

    let store = graph.column_store("Item").expect("master");
    assert_eq!(
        store.column_count(),
        columns_before + 1,
        "a SET of an unknown property must append exactly one column"
    );
    assert!(store.slot(fresh).is_some());
    assert_eq!(item_prop(&graph, 1, "fresh"), Some(Value::Int64(7)));
}

/// A cell written **twice** in one failing statement comes back to its
/// pre-statement value, not to the intermediate one.
///
/// Columnar cells are journalled with no first-touch dedup (see
/// `storage/undo.rs`), so correctness here rests entirely on reverse replay
/// landing the earliest capture last. A dedup that kept the *latest* capture —
/// the natural mistake, and the one the weight-capture seam would suggest —
/// restores `111` instead of the seeded value and nothing else in this file
/// notices.
#[test]
fn rollback_restores_a_cell_written_twice_in_one_statement() {
    let mut graph = wide_columnar();
    let before = fingerprint(&mut graph);
    assert_eq!(
        item_prop(&graph, 1, "qty"),
        Some(Value::Int64(1)),
        "precondition: the fixture seeds qty = id"
    );

    expect_failure(
        &mut graph,
        &format!(
            "MATCH (n:Item {{id: 1}}) SET n.qty = 111 SET n.qty = 222 \
             {FAILS_AFTER_A_COLUMNAR_WRITE}"
        ),
        None,
    );

    assert_eq!(
        item_prop(&graph, 1, "qty"),
        Some(Value::Int64(1)),
        "two writes to one cell must roll back to the pre-statement value; \
         reading 111 means only the last capture survived, 222 means none did"
    );
    assert_eq!(fingerprint(&mut graph), before);
}

/// A cell that was **absent** before the statement must read absent again
/// afterwards.
///
/// `prior: None` is restored by writing `Value::Null`, which is not the same
/// bytes but is the same observation: `get` answers `None` for absent and null
/// alike and `row_properties` skips both. This arm is what says the claim is
/// true rather than merely argued — it checks the property read, the
/// `row_properties`-derived master rows inside `fingerprint`, and the
/// node-side view.
///
/// The column is grown by a *committed* statement on a different row first, so
/// the schema-growth entry is not in play and this measures the null restore
/// alone.
#[test]
fn rollback_restores_a_cell_to_absent() {
    let mut graph = wide_columnar();
    run(&mut graph, "MATCH (n:Item {id: 5}) SET n.extra = 'kept'");
    assert_eq!(
        item_prop(&graph, 1, "extra"),
        None,
        "precondition: the column exists but row 1's cell is empty"
    );
    let before = fingerprint(&mut graph);

    expect_failure(
        &mut graph,
        &format!("MATCH (n:Item {{id: 1}}) SET n.extra = 'doomed' {FAILS_AFTER_A_COLUMNAR_WRITE}"),
        None,
    );

    assert_eq!(
        item_prop(&graph, 1, "extra"),
        None,
        "a cell that was absent must read absent after the rollback"
    );
    assert_eq!(
        item_prop(&graph, 5, "extra"),
        Some(Value::String("kept".into())),
        "and the committed value on the other row must be untouched"
    );
    assert_eq!(fingerprint(&mut graph), before);
}

/// Every cell shape above, on the **mapped** backend.
///
/// Mapped is where a columnar write is most likely to go wrong and least
/// likely to be noticed: its columns can be mmap-backed, so the write path has
/// to bring one to heap before mutating it, and the pre-image is read through
/// the same `get` that has an mmap fallback. The whole-store clone used to hide
/// both — it produced a fully heap-resident copy before anything was written.
#[test]
fn mapped_columnar_cells_roll_back() {
    let mut graph = wide_columnar_mapped();
    run(&mut graph, "MATCH (n:Item {id: 5}) SET n.extra = 'kept'");
    let before = fingerprint(&mut graph);
    let columns_before = graph.column_store("Item").expect("master").column_count();

    expect_failure(
        &mut graph,
        &format!(
            "MATCH (n:Item {{id: 1}}) SET n.qty = 111 SET n.qty = 222, \
             n.extra = 'doomed', n.fresh = 7 {FAILS_AFTER_A_COLUMNAR_WRITE}"
        ),
        None,
    );

    assert_eq!(item_prop(&graph, 1, "qty"), Some(Value::Int64(1)));
    assert_eq!(item_prop(&graph, 1, "extra"), None);
    assert_eq!(item_prop(&graph, 1, "fresh"), None);
    assert_eq!(
        graph.column_store("Item").expect("master").column_count(),
        columns_before
    );
    assert_eq!(fingerprint(&mut graph), before);
}

/// The control: a *read* statement on the same fixture clones no store.
///
/// Without a control the assertion above cannot distinguish "the columnar write
/// path clones the master" from "something in every statement clones a store".
/// The control used to be the same statement on a never-consolidated graph —
/// the shape that no longer exists, because construction is columnar and every
/// graph owns master stores from its first node. Its premise is void, so it is
/// replaced rather than reinterpreted: what is still separable is the *write*
/// path from everything else a statement does, and a `MATCH … RETURN` exercises
/// everything else.
#[test]
fn a_read_statement_clones_no_column_store() {
    use crate::graph::storage::column_store::{column_store_clones, reset_column_store_clones};

    let mut graph = wide_columnar();
    assert!(
        graph.column_store_count() > 0,
        "the fixture must own master column stores, or the control is vacuous"
    );

    reset_column_store_clones();
    run(&mut graph, "MATCH (n:Item) WHERE n.qty > 1 RETURN n.qty");
    let cloned = column_store_clones();

    assert_eq!(
        cloned, 0,
        "a read statement cloned {cloned} store(s); the counter must be reading \
         the columnar write path and nothing else"
    );
}

/// An indexed graph rolls back through the journal, not through a whole-graph
/// clone. The fidelity half is covered by the `indexed` arm of every shape
/// above; what this pins is the *cost* half — that a user index no longer
/// downgrades the checkpoint for the rest of the session.
#[test]
fn indexed_graph_rolls_back_without_copying_the_graph() {
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};

    let mut graph = seeded_indexed();
    reset_backend_clone_count();
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item) SET n.name = 'touched', n.bad = duration({months: 2147483648})",
        None,
    );
    assert_eq!(
        backend_clone_nodes(),
        0,
        "an indexed graph must take the journal path, not the clone checkpoint"
    );
}

/// The control every indexed arm in this file depends on: a **committed**
/// multi-row `SET` really does move the nodes between index buckets.
///
/// Without it, the rollback arms below would keep passing if `SET` stopped
/// maintaining indexes altogether — a no-op write is trivially restorable. The
/// arm exists because P6 put the maintenance call behind a statement-scoped
/// "does this type carry an index" answer, and getting that answer wrong is
/// silent: the index keeps serving its stale buckets and `try_index_lookup`
/// consults them with no version check, so an indexed `MATCH` returns the old
/// rows and nothing anywhere reports an error.
#[test]
fn a_committed_set_moves_every_row_between_index_buckets() {
    let mut graph = seeded_indexed();
    assert_eq!(
        index_bucket(&graph, "qty", Value::Int64(10)).len(),
        1,
        "precondition: Item 1 sits in the qty=10 bucket"
    );

    // Every row of the type, so the memo's second and later rows are covered
    // too — a memo that answered only the first row correctly would leave the
    // rest of the bucket stale.
    run(&mut graph, "MATCH (n:Item) SET n.qty = 77");

    assert!(
        index_bucket(&graph, "qty", Value::Int64(10)).is_empty(),
        "the vacated bucket must be empty"
    );
    assert_eq!(
        index_bucket(&graph, "qty", Value::Int64(77)).len(),
        3,
        "all three Items must have joined the new bucket"
    );
    assert_eq!(
        range_bucket(&graph, "qty", Value::Int64(77)).len(),
        3,
        "the range index is maintained by separate code and must move too"
    );
    assert_eq!(
        graph
            .lookup_by_composite_index(
                "Item",
                &["name".to_string(), "qty".to_string()],
                &[Value::String("a".to_string()), Value::Int64(77)],
            )
            .map(|members| members.len()),
        Some(1),
        "and so must the composite index"
    );
}

/// The bucket-order case the position journal exists for.
///
/// `Item` 1 and 3 share a `qty` after the setup write, so that bucket holds
/// two members in a known order. A statement that moves the *first* member out
/// and then fails must put it back at the front — a rollback that merely
/// restored membership would append it, silently reordering the rows an
/// indexed `MATCH` returns.
#[test]
fn rollback_restores_index_bucket_order_not_just_membership() {
    let mut graph = seeded_indexed();
    run(&mut graph, "MATCH (n:Item {id: 3}) SET n.qty = 10");
    let bucket_before = index_bucket(&graph, "qty", Value::Int64(10));
    assert_eq!(
        bucket_before.len(),
        2,
        "the fixture needs a bucket with two members to have an order at all"
    );

    let before = fingerprint(&mut graph);
    let error = expect_failure(
        &mut graph,
        "MATCH (n:Item {id: 1}) SET n.qty = 999 \
         WITH n MATCH (m:Item {id: 2}) SET m.bad = duration({months: 2147483648})",
        None,
    );
    let after = fingerprint(&mut graph);

    assert_eq!(before, after, "statement must roll back.\nerror: {error}");
    assert_eq!(
        index_bucket(&graph, "qty", Value::Int64(10)),
        bucket_before,
        "the evicted member must come back at its original position"
    );
}

/// The range-index counterpart of the bucket-order pin above, run **with a
/// fork outstanding** — the configuration the layered range index introduced.
///
/// `range_indices` is *parked* by `swap_data_scale`: the shell restore does not
/// cover it, so a failed statement's range buckets are put back one inverse
/// edit at a time by `rollback::apply` (`BucketAppended` / `BucketRemoved`).
/// Since the index became a level stack, those inverse edits run against an
/// **overlay** level whenever a reader is holding the base — the writer's
/// `get_mut` / `entry_or_default` materialise the merged bucket into the
/// overlay first. This pins both halves: the writer is restored exactly
/// (position included), and the reader that forced the fork never sees either
/// the failed write or its reversal.
#[test]
fn a_rolled_back_statement_restores_the_parked_range_index_under_a_fork() {
    let mut graph = seeded_indexed();
    run(&mut graph, "MATCH (n:Item {id: 3}) SET n.qty = 10");
    let bucket_before = range_bucket(&graph, "qty", Value::Int64(10));
    assert_eq!(
        bucket_before.len(),
        2,
        "the fixture needs a range bucket with two members to have an order at all"
    );

    // The fork: a held reader, which is what makes the writer's next edit
    // land in a fresh level over a shared base.
    let reader = graph.clone();
    let reader_before = range_bucket(&reader, "qty", Value::Int64(10));

    let before = fingerprint(&mut graph);
    let error = expect_failure(
        &mut graph,
        "MATCH (n:Item {id: 1}) SET n.qty = 999 \
         WITH n MATCH (m:Item {id: 2}) SET m.bad = duration({months: 2147483648})",
        None,
    );
    let after = fingerprint(&mut graph);

    assert_eq!(before, after, "statement must roll back.\nerror: {error}");
    assert_eq!(
        range_bucket(&graph, "qty", Value::Int64(10)),
        bucket_before,
        "the evicted member must come back at its original position in the range bucket"
    );
    assert!(
        range_bucket(&graph, "qty", Value::Int64(999)).is_empty(),
        "the failed statement's new range bucket must be gone"
    );
    assert_eq!(
        range_bucket(&reader, "qty", Value::Int64(10)),
        reader_before,
        "the reader must not have seen the write or its reversal"
    );

    // The index still answers ordered range scans after the round trip.
    let scanned = graph
        .lookup_range(
            "Item",
            "qty",
            std::ops::Bound::Unbounded,
            std::ops::Bound::Unbounded,
        )
        .expect("the range index survives the rollback");
    assert_eq!(
        scanned.len(),
        3,
        "every seeded Item must still be reachable through the range index"
    );
}

/// One `Item` **range**-index bucket's members, in bucket order.
fn range_bucket(graph: &DirGraph, property: &str, value: Value) -> Vec<usize> {
    graph
        .range_indices
        .get(&("Item".to_string(), property.to_string()))
        .and_then(|btree| btree.get(&value))
        .map(|members| members.iter().map(|idx| idx.index()).collect())
        .unwrap_or_default()
}

/// One `Item` property-index bucket's members, in bucket order.
fn index_bucket(graph: &DirGraph, property: &str, value: Value) -> Vec<usize> {
    graph
        .property_indices
        .get(&("Item".to_string(), property.to_string()))
        .and_then(|value_map| value_map.get(&value))
        .map(|members| members.iter().map(|idx| idx.index()).collect())
        .unwrap_or_default()
}
