//! Columnar SET cost: O(changes), not O(type)

use super::*;

/// A one-row columnar `SET` must journal a pre-image for the node it changed,
/// not for every node of the type.
///
/// This is the guard for the cost regression the post-merge benchmark caught:
/// `MATCH (i:Item {id: …}) SET i.priority = …` on a saved 100k-node graph ran
/// ~1.8× slower than the whole-graph clone it replaced. The mechanism is the
/// end-of-batch handle-refresh sweep in `execute_set`, which re-points every
/// node's `Arc<ColumnStore>` at the forked master. That sweep goes through
/// `node_weight_mut_silent` — silent towards the WAL recorder, but until this
/// guard existed it fell through to the *recorded* `node_weight_mut` on
/// `MemoryGraph`, so a single-property write cloned a `NodeData` per node of
/// the type into the journal.
///
/// Why the existing guards cannot see it: `journalled_statements_copy_zero_nodes`
/// reads `BACKEND_CLONE_NODES`, which counts backend clones only — the journal
/// path deliberately clones no backend, so the counter reads zero whether the
/// journal captured one pre-image or two hundred. The cost lives entirely
/// inside the journal, so the counter has to as well.
///
/// **Phase 2 re-point.** The `NodeData` bound above is kept (a columnar SET
/// must still not clone node weights), and the cost oracle it was written for
/// is now stated in the mechanism that carries the cost: one
/// `UndoEntry::ColumnarCell` per `(row, property)` the statement changed. The
/// `ColumnarHandles` entry the old bound coexisted with is gone, so without the
/// cell count this assertion would pass just as happily for a journal that
/// captured the whole store — which is exactly what it did before Phase 2.
#[test]
fn a_columnar_set_journals_one_pre_image_per_changed_node() {
    use crate::graph::storage::undo::{
        journal_columnar_cells, journal_node_pre_images, reset_journal_columnar_cells,
        reset_journal_node_pre_images,
    };

    let mut graph = wide_columnar();
    reset_journal_node_pre_images();
    reset_journal_columnar_cells();
    run(&mut graph, "MATCH (i:Item {id: 7}) SET i.priority = 3");
    let captured = journal_node_pre_images();
    let cells = journal_columnar_cells();

    assert!(
        captured <= 2,
        "a one-row columnar SET captured {captured} node pre-images across \
         {WIDE_ITEMS} nodes of the type; it must be O(nodes changed), not \
         O(nodes of the type) — the handle-refresh sweep is being journalled"
    );
    assert_eq!(
        cells, 1,
        "a one-row, one-property columnar SET must journal exactly one cell \
         pre-image across {WIDE_ITEMS} nodes of the type; {cells} means the \
         capture is sized by something other than the change"
    );
}

/// The cost oracle's second dimension: **properties**, not rows.
///
/// A three-property `SET` on one row journals three cells and nothing else. The
/// arm above cannot see a per-statement or per-type constant (it would read 1
/// either way); this one separates "one entry per changed cell" from "one entry
/// per statement", which is the shape the replaced mechanism had.
#[test]
fn a_columnar_set_journals_one_cell_per_changed_property() {
    use crate::graph::storage::undo::{journal_columnar_cells, reset_journal_columnar_cells};

    let mut graph = wide_columnar();
    reset_journal_columnar_cells();
    run(
        &mut graph,
        "MATCH (i:Item {id: 7}) SET i.qty = 1, i.priority = 3, i.rank = 5",
    );
    assert_eq!(
        journal_columnar_cells(),
        3,
        "three changed cells must journal three pre-images"
    );

    // Two rows x one property is the other axis.
    reset_journal_columnar_cells();
    run(&mut graph, "MATCH (i:Item) WHERE i.id < 2 SET i.qty = 9");
    assert_eq!(
        journal_columnar_cells(),
        2,
        "two changed rows must journal two pre-images, not {WIDE_ITEMS}"
    );
}

/// The same bound on the **mapped** backend, which is where it is easiest to
/// lose and hardest to notice.
///
/// `node_weight_mut_silent` has a trait *default* that forwards to the recorded
/// `node_weight_mut`. `MemoryGraph` overrides it, which is what the arm above
/// pins; `MappedGraph` had no reason to until it gained a journal, and adding
/// the journal without the override re-creates the O(type)-per-write cost
/// exactly. Measured, not assumed: with the override removed this captured
/// **200** pre-images for a one-row `SET`, against 0 with it.
///
/// This arm and `the_mapped_silent_write_path_records_nothing` guard the same
/// override from opposite ends — one at the seam, one through the statement
/// that actually reaches it — because the seam has a second caller
/// (`mutation::batch`'s columnar detach/reattach, gated on
/// `is_mapped() || is_disk()`) that no Cypher statement reaches today and so
/// no end-to-end test can cover.
#[test]
fn a_mapped_columnar_set_journals_one_pre_image_per_changed_node() {
    use crate::graph::storage::undo::{
        journal_columnar_cells, journal_node_pre_images, reset_journal_columnar_cells,
        reset_journal_node_pre_images,
    };

    let mut graph = wide_columnar_mapped();
    reset_journal_node_pre_images();
    reset_journal_columnar_cells();
    run(&mut graph, "MATCH (i:Item {id: 7}) SET i.priority = 3");
    let captured = journal_node_pre_images();
    let cells = journal_columnar_cells();

    assert_eq!(
        cells, 1,
        "a one-row columnar SET on a mapped graph must journal exactly one \
         cell pre-image, not {cells}"
    );
    assert!(
        captured <= 2,
        "a one-row columnar SET on a mapped graph captured {captured} node \
         pre-images across {WIDE_ITEMS} nodes of the type; the mapped \
         handle-refresh sweep is being journalled"
    );
}

/// The same statement on a heap-resident (unmapped) graph, as the control.
///
/// Pins that the bound above is a property of the write path rather than of
/// the mapping or of this fixture's size: if the unmapped path ever started
/// capturing per type, the mapped assertion alone would not say which layer
/// regressed. This used to run against a de-columnarized graph, a shape
/// construction no longer produces.
#[test]
fn an_unmapped_set_journals_one_pre_image_per_changed_node() {
    use crate::graph::storage::undo::{journal_node_pre_images, reset_journal_node_pre_images};

    let mut graph = wide_columnar();
    reset_journal_node_pre_images();
    run(&mut graph, "MATCH (i:Item {id: 7}) SET i.priority = 3");
    let captured = journal_node_pre_images();

    assert!(
        captured <= 2,
        "a one-row SET on an unmapped graph captured {captured} node \
         pre-images across {WIDE_ITEMS} nodes"
    );
}

/// Two columnar writes to the same cell in one statement must both be
/// visible, and the second must win.
///
/// The mechanism has been rewritten twice under this assertion, which is why
/// the assertion is what got kept. Pre-D1-Phase-3: the first write forked away
/// from `1 + N` node handles and registered an end-of-clause re-point sweep.
/// Post-Phase-3, pre-Phase-2: the first write forked away from the undo
/// journal's whole-store pre-image and the second mutated the fork in place.
/// Now: **neither write forks anything** — the journal holds one
/// `UndoEntry::ColumnarCell` per write, both mutate the backend's own store in
/// place, and reverse replay would restore the *first* capture if the statement
/// failed. The observable is unchanged across all three.
#[test]
fn two_columnar_writes_in_one_statement_both_land() {
    let mut graph = wide_columnar();

    // Locate the node up front: `id` is an inline canonical field, not a
    // column-store property, so it cannot be used to read back through the
    // per-node handle. `qty` is columnar and seeded to the node's index.
    let idx = graph
        .graph
        .node_indices()
        .find(|i| {
            graph
                .graph
                .node_view(*i)
                .and_then(|n| n.get_property_value("qty"))
                .map(|v| v == crate::datatypes::Value::Int64(1))
                .unwrap_or(false)
        })
        .expect("fixture seeds qty = node index");

    // Two SET clauses in one statement, same type and same property. Both
    // journal a cell pre-image and both mutate the master IN PLACE.
    let allocation_before = Arc::as_ptr(graph.column_store("Item").expect("master"));
    run(
        &mut graph,
        "MATCH (n:Item {id: 1}) SET n.qty = 111 SET n.qty = 222",
    );
    assert!(
        std::ptr::eq(
            allocation_before,
            Arc::as_ptr(graph.column_store("Item").expect("master"))
        ),
        "neither write may fork the master"
    );

    // Read back through the public route. Both writes must be visible.
    let node = graph.graph.node_view(idx).expect("node still present");
    assert_eq!(
        node.get_property_value("qty"),
        Some(crate::datatypes::Value::Int64(222)),
        "both writes must be visible; reading 1 means the second write landed \
         somewhere the read route does not resolve"
    );

    // And nothing but the backend holds the master.
    let master = graph.column_store("Item").expect("master");
    assert_eq!(
        Arc::strong_count(master),
        1,
        "nothing but the backend may hold the master, or every write pays a \
         whole-store copy"
    );
}

/// **Replaces `every_node_shares_the_master_column_store_handle`.**
///
/// That test pinned the pre-D1 design: `enable_columnar` pointed every node of
/// a type at the master, so its strong count was `1 + nodes-of-type` and every
/// first-write-of-a-statement forked the whole store. D1 Phase 3 deleted the
/// node-held handle, and this is the inverted assertion: *no* node holds one,
/// and the master is uniquely owned.
///
/// Keeping the coverage rather than the assertion is deliberate — the property
/// this file cares about is what the refcount implies for `Arc::make_mut`, and
/// that has flipped from "always copies" to "copies only under a checkpoint".
#[test]
fn no_node_holds_a_column_store_handle() {
    let graph = wide_columnar();
    let master = graph
        .column_store("Item")
        .expect("the fixture installs a master store for Item");

    assert_eq!(
        Arc::strong_count(master),
        1,
        "the backend must be the only owner of the master; a second handle \
         means something re-introduced a replica, and every columnar write \
         would silently go back to copying the whole store"
    );

    // Non-vacuity: the nodes really are columnar, they just carry row ids.
    let columnar = graph
        .graph
        .node_indices()
        .filter(|idx| {
            matches!(
                graph.graph.node_weight(*idx).map(|n| &n.properties),
                Some(PropertyStorage::Columnar(_))
            )
        })
        .count();
    assert_eq!(
        columnar, WIDE_ITEMS,
        "every node of the type must still be columnar, or the refcount above \
         is 1 because the fixture stopped being saved"
    );
}

/// **Replaces `fork_detection_is_a_no_op_while_nodes_hold_strong_handles`.**
///
/// The reference-count invariant the whole programme turns on. Phase 2
/// strengthened it from "uniquely owned *between* statements" to "uniquely
/// owned *always*, on a non-forked backend": the undo journal used to hold the
/// pre-statement store, so mid-statement the count was ≥ 2 and `Arc::make_mut`
/// forked; it now holds cell values only, so nothing but the backend ever owns
/// the master and the write mutates it in place.
///
/// The in-statement half is asserted through **allocation identity**, which is
/// the only way to see mid-statement ownership from outside the statement: an
/// `Arc::make_mut` that forked would leave the backend pointing at a different
/// allocation once the statement returned. The refcount alone cannot say that
/// — it reads 1 either way, because the fork is what dropped the second
/// handle. The `debug_assert!` inside `write_column_master` asserts the same
/// property at the instant it holds.
#[test]
fn the_master_is_uniquely_owned_between_statements() {
    let mut graph = wide_columnar();

    assert_eq!(
        Arc::strong_count(graph.column_store("Item").expect("master")),
        1,
        "precondition: uniquely owned before any statement"
    );
    let allocation_before = Arc::as_ptr(graph.column_store("Item").expect("master"));

    run(&mut graph, "MATCH (n:Item {id: 1}) SET n.qty = 111");

    assert_eq!(
        Arc::strong_count(graph.column_store("Item").expect("master")),
        1,
        "a committed statement must leave nothing else holding the master"
    );
    assert!(
        std::ptr::eq(
            allocation_before,
            Arc::as_ptr(graph.column_store("Item").expect("master"))
        ),
        "the statement replaced the master's allocation, so `Arc::make_mut` \
         forked mid-statement: something held a second handle while the write \
         ran and the write copied {WIDE_ITEMS} rows to change one cell"
    );

    // And the same across a statement that *fails*, where the journal is read
    // rather than dropped. A replay that reinstalled a store would show here.
    let allocation_before = Arc::as_ptr(graph.column_store("Item").expect("master"));
    expect_failure(
        &mut graph,
        "MATCH (n:Item {id: 2}) SET n.qty = 222 \
         WITH n MATCH (m:Item {id: 3}) SET m.qty = duration({months: 2147483648})",
        None,
    );
    assert!(
        std::ptr::eq(
            allocation_before,
            Arc::as_ptr(graph.column_store("Item").expect("master"))
        ),
        "a rolled-back statement must restore cells into the live store, not \
         swap a pre-statement copy back in"
    );
    assert_eq!(
        Arc::strong_count(graph.column_store("Item").expect("master")),
        1,
        "and must leave the master uniquely owned"
    );
    assert_eq!(
        graph
            .graph
            .node_view(
                graph
                    .graph
                    .node_indices()
                    .find(|i| graph.graph.get_node_id(*i)
                        == Some(crate::datatypes::Value::Int64(1)))
                    .expect("node 1")
            )
            .and_then(|n| n.get_property_value("qty")),
        Some(crate::datatypes::Value::Int64(111)),
        "and the write must actually be visible"
    );
}
