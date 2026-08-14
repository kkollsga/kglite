//! Row-level undo: the entries always-columnar construction made necessary
//!
//! A `CREATE` appends a row to the type's master store and a `DELETE`
//! tombstones one, both *inside* the statement window — the property the
//! module doc used to argue was impossible in memory mode. The shape tests at
//! the top of this file already compare the master rows through `fingerprint`,
//! so they fail if either undo is missing; what these add is the row *count*
//! and row-*identity* half, which a value comparison cannot see: a rollback
//! that left the store one row longer would still fingerprint clean as long as
//! the extra row belonged to no live node.

use super::*;

/// `Item`'s master store: `(row count, live row count)`.
fn item_rows(graph: &DirGraph) -> (u32, u32) {
    let store = graph.column_store("Item").expect("master store");
    (store.row_count(), store.live_count())
}

/// A rolled-back `CREATE` must leave the store at exactly its pre-statement
/// length — and the next `CREATE` must land on the row the rolled-back one
/// vacated, not on one past it.
///
/// The non-vacuity half is the first assertion: a `CREATE` has to *grow* the
/// store for the rest of the test to mean anything. Before construction became
/// columnar it did not — the node was row-shaped and the store untouched — so
/// this test could not have been written against the old shape at all.
#[test]
fn a_rolled_back_create_restores_the_row_count_and_the_next_row_id() {
    let mut graph = seeded();
    let (rows_before, live_before) = item_rows(&graph);

    run(&mut graph, "CREATE (:Item {id: 90, name: 'probe', qty: 1})");
    assert_eq!(
        item_rows(&graph),
        (rows_before + 1, live_before + 1),
        "a CREATE must append one master row, or the rollback assertions below \
         are vacuous"
    );
    run(&mut graph, "MATCH (n:Item {id: 90}) DELETE n");
    assert_eq!(
        item_rows(&graph),
        (rows_before + 1, live_before),
        "a DELETE must tombstone the row it removed the node for"
    );

    let before = fingerprint(&mut graph);
    let (rows, live) = item_rows(&graph);

    expect_failure(
        &mut graph,
        "CREATE (:Item {id: 100, name: 'a', qty: 1}), \
                (:Item {id: 101, bad: duration({months: 2147483648})})",
        None,
    );

    assert_eq!(
        item_rows(&graph),
        (rows, live),
        "a rolled-back CREATE must truncate the rows it appended"
    );
    assert_eq!(fingerprint(&mut graph), before);

    // Row identity, not just row count: the next create takes the vacated row.
    run(&mut graph, "CREATE (:Item {id: 102, name: 'b', qty: 2})");
    let idx = graph
        .graph
        .node_indices()
        .find(|i| graph.graph.get_node_id(*i) == Some(Value::Int64(102)))
        .expect("the new node");
    let row = match graph.graph.node_weight(idx).map(|n| &n.properties) {
        Some(PropertyStorage::Columnar(row)) => row.row_id(),
        other => panic!("expected a columnar row, got {other:?}"),
    };
    assert_eq!(row, rows, "the re-created node must reuse the vacated row");
    assert_eq!(
        item_prop(&graph, 102, "name"),
        Some(Value::String("b".into())),
        "and read its own values back out of it"
    );
}

/// A `CREATE` of a type the graph has never seen, rolled back, must leave no
/// store behind — an empty one is observable through `graph_info`,
/// `graph_info`, and the saved file.
#[test]
fn a_rolled_back_create_of_a_new_type_leaves_no_store() {
    let mut graph = seeded();
    let before = fingerprint(&mut graph);
    assert!(
        graph.column_store("Fresh").is_none(),
        "precondition: the type must not exist yet"
    );

    expect_failure(
        &mut graph,
        "CREATE (:Fresh {id: 1, name: 'a'}), \
                (:Fresh {id: 2, bad: duration({months: 2147483648})})",
        None,
    );

    assert!(
        graph.column_store("Fresh").is_none(),
        "a rolled-back CREATE must not leave an empty store for a type that \
         never existed"
    );
    assert_eq!(fingerprint(&mut graph), before);
}

/// A rolled-back `DELETE` must un-tombstone the row, and the restored node must
/// read its own values back.
#[test]
fn a_rolled_back_delete_untombstones_the_row() {
    let mut graph = seeded();
    let before = fingerprint(&mut graph);
    let (rows, live) = item_rows(&graph);

    expect_failure(
        &mut graph,
        "MATCH (n:Item {id: 1}) DELETE n WITH n MATCH (m:Item {id: 2}) \
         SET m.qty = duration({months: 2147483648})",
        None,
    );

    assert_eq!(
        item_rows(&graph),
        (rows, live),
        "a rolled-back DELETE must restore the row's liveness"
    );
    assert_eq!(
        item_prop(&graph, 1, "qty"),
        Some(Value::Int64(10)),
        "and the restored node must read its own row"
    );
    assert_eq!(fingerprint(&mut graph), before);
}

/// One statement that creates, sets and deletes before it fails. The three
/// row-level undos have to compose in reverse order without stepping on each
/// other — the append truncates rows a tombstone entry still names, and the
/// re-created node's row is one of them.
#[test]
fn a_failed_statement_that_creates_sets_and_deletes_rolls_all_three_back() {
    let mut graph = seeded();
    let before = fingerprint(&mut graph);
    let (rows, live) = item_rows(&graph);

    expect_failure(
        &mut graph,
        "CREATE (x:Item {id: 200, name: 'new', qty: 1}) \
         WITH x MATCH (a:Item {id: 1}), (b:Item {id: 3}) \
         SET a.qty = 999, x.qty = 5 \
         DELETE b \
         WITH a MATCH (m:Item {id: 2}) \
         SET m.qty = duration({months: 2147483648})",
        None,
    );

    assert_eq!(item_rows(&graph), (rows, live));
    assert_eq!(item_prop(&graph, 1, "qty"), Some(Value::Int64(10)));
    assert_eq!(item_prop(&graph, 3, "qty"), Some(Value::Int64(30)));
    assert_eq!(fingerprint(&mut graph), before);
}

/// The same three-way statement on every fixture the file carries, because a
/// row-level undo is exactly the kind of thing that holds on one backend and
/// not another.
#[test]
fn the_three_way_statement_rolls_back_on_every_fixture() {
    for mut graph in [
        seeded(),
        seeded_columnar(),
        seeded_indexed(),
        seeded_mapped(),
    ] {
        let before = fingerprint(&mut graph);
        expect_failure(
            &mut graph,
            "CREATE (x:Item {id: 200, name: 'new', qty: 1}) \
             WITH x MATCH (a:Item {id: 1}), (b:Item {id: 3}) \
             SET a.qty = 999 \
             DELETE b \
             WITH a MATCH (m:Item {id: 2}) \
             SET m.qty = duration({months: 2147483648})",
            None,
        );
        assert_eq!(fingerprint(&mut graph), before);
    }
}

/// A title write lands in the master's reserved column, not on the node's
/// inline field — and rolls back from there.
///
/// The inline override this replaced was reconciled only by the save-time
/// consolidation pass, so every title write cost the next `save()` a full O(N)
/// rebuild. Both halves are asserted: the store carries the new title (so the
/// write really went through it) and the node's inline field stays on its
/// sentinel (so there is no second copy to reconcile).
#[test]
fn a_title_write_goes_through_the_master_and_rolls_back() {
    let mut graph = seeded();
    let before = fingerprint(&mut graph);

    run(&mut graph, "MATCH (n:Item {id: 1}) SET n.title = 'moved'");

    let idx = graph
        .graph
        .node_indices()
        .find(|i| graph.graph.get_node_id(*i) == Some(Value::Int64(1)))
        .expect("the node");
    let row = match graph.graph.node_weight(idx).map(|n| &n.properties) {
        Some(PropertyStorage::Columnar(row)) => row.row_id(),
        other => panic!("expected a columnar row, got {other:?}"),
    };
    assert_eq!(
        graph.column_store("Item").expect("master").get_title(row),
        Some(Value::String("moved".into())),
        "the title must be written through the master's reserved column"
    );
    assert!(
        matches!(
            graph.graph.node_weight(idx).map(|n| &n.title),
            Some(Value::Null)
        ),
        "and the inline field must stay on its sentinel — a second copy is what \
         forced a rebuild at save"
    );

    let after_write = fingerprint(&mut graph);
    assert_ne!(after_write, before, "the write must be observable");

    expect_failure(
        &mut graph,
        &format!("MATCH (n:Item {{id: 1}}) SET n.title = 'doomed' {FAILS_AFTER_A_COLUMNAR_WRITE}"),
        None,
    );
    assert_eq!(
        fingerprint(&mut graph),
        after_write,
        "a rolled-back title write must restore the prior title"
    );
}
