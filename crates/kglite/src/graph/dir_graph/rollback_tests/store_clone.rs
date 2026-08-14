//! The cell-grained journal: no store clone per statement

use super::*;

/// **The flipped Phase-1 pin** (was
/// `a_columnar_statement_still_deep_clones_the_master_pinned_defect`).
///
/// Until Phase 2 every mutating statement's first columnar write handed the
/// undo journal an `Arc::clone` of the touched type's master `ColumnStore`,
/// which made the `Arc::make_mut` two lines later deep-copy every column of
/// that type — for a one-cell write. The journal dropped at commit, the
/// refcount returned to one, and the next statement paid again. Measured on
/// 0.15.14: a single-row `SET` cost 4.3 µs on a fresh 50k×12 graph and 328 µs
/// on the same graph after `save()`, a 76× tax scaling with the type's row
/// count. Phase 1 pinned that as `(1, 1)` so it could not vanish unobserved;
/// this is the same test with the number the fix produces.
///
/// The mechanism now: the journal holds the changed cell's prior value
/// (`UndoEntry::ColumnarCell`), never a handle on the store, so the master
/// stays uniquely owned and `make_mut` mutates one cell in place. The
/// `debug_assert!` at the write site asserts the same thing from inside;
/// this asserts it from outside, where a copy performed for some *other*
/// reason would also show up.
///
/// Neither sibling oracle can see this: `BACKEND_CLONE_NODES` counts backend
/// copies (the journal clones no backend) and `JOURNAL_NODE_PRE_IMAGES` counts
/// `NodeData` copies (a columnar property is not in a `NodeData`). Both read
/// zero here, before and after the fix.
#[test]
fn a_columnar_statement_clones_no_store() {
    use crate::graph::storage::column_store::{column_store_clones, reset_column_store_clones};

    let mut graph = wide_columnar();

    reset_column_store_clones();
    run(&mut graph, "MATCH (n:Item {id: 1}) SET n.qty = 111");
    let first = column_store_clones();

    reset_column_store_clones();
    run(&mut graph, "MATCH (n:Item {id: 2}) SET n.qty = 222");
    let second = column_store_clones();

    assert_eq!(
        (first, second),
        (0, 0),
        "a columnar statement must copy no column store: writing one cell of a \
         {WIDE_ITEMS}-row type copied ({first}, {second}) whole stores. A \
         non-zero reading means something is holding a second handle on the \
         master across the write — the O(rows x cols)-per-statement tax this \
         phase removed, back again."
    );

    // Non-vacuity: the writes landed, so the zero is a cheap write and not a
    // skipped one.
    assert_eq!(
        graph
            .graph
            .node_view(
                graph
                    .graph
                    .node_indices()
                    .find(|i| graph.graph.get_node_id(*i) == Some(Value::Int64(2)))
                    .expect("node 2")
            )
            .and_then(|n| n.get_property_value("qty")),
        Some(Value::Int64(222)),
    );
}

/// The same zero on the statement that **fails**, where the journal is not
/// merely opened but actually read.
///
/// The committed arm above proves the capture is cheap; this proves the
/// *restore* is too, and that the two together never fall back to copying the
/// store. A rollback implementation that reinstalled a whole pre-statement
/// store — the mechanism this phase replaced — would show up here and nowhere
/// else.
#[test]
fn a_rolled_back_columnar_statement_clones_no_store() {
    use crate::graph::storage::column_store::{column_store_clones, reset_column_store_clones};

    let mut graph = wide_columnar();

    reset_column_store_clones();
    expect_failure(
        &mut graph,
        "MATCH (n:Item {id: 1}) SET n.qty = 111 \
         WITH n MATCH (m:Item {id: 2}) SET m.qty = duration({months: 2147483648})",
        None,
    );
    let cloned = column_store_clones();

    assert_eq!(
        cloned, 0,
        "a rolled-back columnar statement copied {cloned} whole store(s); \
         capture *and* replay must both be O(cells changed)"
    );
    // Non-vacuity: it really did roll back.
    assert_eq!(
        graph
            .graph
            .node_view(
                graph
                    .graph
                    .node_indices()
                    .find(|i| graph.graph.get_node_id(*i) == Some(Value::Int64(1)))
                    .expect("node 1")
            )
            .and_then(|n| n.get_property_value("qty")),
        Some(Value::Int64(1)),
        "the fixture seeds qty = id, and the failed SET must not have stuck"
    );
}
