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

/// A transaction's first write must privatise **the column it writes**, not
/// the type.
///
/// The two pins above own the *unshared* case: a statement on a graph nobody
/// else holds copies nothing at all. A transaction is the shared case, and it
/// is the one every explicit `begin()`/`commit()` block takes: `working_mut`
/// forks the base, both graphs then point at the same `Arc<ColumnStore>` per
/// type, and the first columnar write has to make its own copy before it can
/// mutate — that copy is the *only* thing standing between the base's readers
/// and the transaction's writes, so it cannot be skipped.
///
/// What it can be is **sized by the change**. Until this pin the copy was
/// whole-store, so a one-cell `SET` inside a transaction deep-copied every
/// column of the type: measured on a 50 k-row `Item` at 24 columns, a
/// transaction's first `SET` cost 406 µs against the same graph's 4.5 µs
/// unshared write, and the per-statement overhead over a 20-statement
/// transaction grew at ~0.5 µs per column of the type (0.15.14, which had no
/// store on a fresh graph to copy, was flat at 0.4–1.0 µs).
///
/// The assertion is two-sided on purpose. **25** — `id`, `name` and
/// `WIDE_SCHEMA_COLUMNS` properties — is the defect. **0** would mean the
/// store was not shared at all, i.e. the fixture stopped exercising a
/// transaction fork and the pin went vacuous; the isolation assertion below
/// would still pass in that state, because a working copy that owns its store
/// outright also isolates. Only 1 is both cheap and real.
#[test]
fn a_transactions_first_write_copies_only_the_column_it_writes() {
    use crate::graph::session::Session;
    use crate::graph::storage::column_store::{column_clones, reset_column_clones};

    let session = Session::new(wide_schema_columnar());
    let mut tx = session.begin();
    let working = tx.working_mut().expect("begin() is read-write");

    reset_column_clones();
    run(working, "MATCH (i:Item {id: 7}) SET i.p0 = 111");
    let first = column_clones();

    // The fork is once per transaction, not once per statement: the working
    // copy owns its column after the first write, so the second writes in
    // place. A per-statement copy here is the same tax wearing a different hat.
    reset_column_clones();
    run(working, "MATCH (i:Item {id: 8}) SET i.p0 = 222");
    let second = column_clones();

    assert_eq!(
        (first, second),
        (1, 0),
        "a transaction's first one-cell write must deep-copy exactly one column \
         of a {WIDE_SCHEMA_COLUMNS}-property type and its later writes none; \
         copied ({first}, {second}). A first reading near {} is the whole-store \
         fork — O(rows x columns) to write one cell. A first reading of 0 means \
         the base is not sharing the store and the pin is vacuous.",
        WIDE_SCHEMA_COLUMNS + 2
    );

    // Non-vacuity, and the reason the copy exists: the writes landed in the
    // working copy and nowhere else.
    assert_eq!(
        item_prop(working, 7, "p0"),
        Some(Value::Int64(111)),
        "the transaction's write must be visible to the transaction"
    );
    assert_eq!(
        item_prop(&session.snapshot(), 7, "p0"),
        Some(Value::Int64(7)),
        "the fixture seeds p0 = id; an uncommitted transaction write that \
         reached the base is a copy-on-write failure, not an optimisation"
    );
}
