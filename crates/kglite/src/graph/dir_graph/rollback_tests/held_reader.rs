//! D2 Phase 2 — rollback while a reader holds the base

use super::*;

/// A failed statement run while a reader is holding the graph must leave
/// **both** graphs exactly as they were.
///
/// This is D2 risk R3, and it is the one unforgivable failure mode in the whole
/// programme: every `UndoEntry` is keyed on a `NodeIndex`/`EdgeIndex` and is
/// reversed through `GraphWrite`. On a `Forked` backend that reversal must land
/// in the *overlay*. If any of it reached the shared base instead, the reader's
/// snapshot would silently acquire a rolled-back write — no error, no crash,
/// and no other test in this file would see it, because every other test owns
/// its graph outright.
///
/// The reader side is the golden snapshot: `fingerprint` is deliberately
/// over-specified (petgraph slot identity, inverted-index bucket *order*, schema
/// metadata, master column rows), so *any* base mutation shows up here, not just
/// one that changes a value a query would return.
#[test]
fn a_rollback_while_a_reader_is_held_touches_neither_graph() {
    use crate::graph::handle::make_dir_graph_mut;
    use std::sync::Arc;

    for (name, build) in [
        ("plain", seeded as fn() -> DirGraph),
        ("columnar", seeded_columnar as fn() -> DirGraph),
        ("indexed", seeded_indexed as fn() -> DirGraph),
    ] {
        let mut writer = Arc::new(build());
        let reader = Arc::clone(&writer);

        // Fingerprinting needs `&mut`, and the reader is shared — so read
        // through a clone of it. The clone is a copy-on-write overlay over the
        // same base, so its content *is* the reader's content.
        let reader_before = fingerprint(&mut (*reader).clone());

        // One `make_dir_graph_mut` for both the fingerprint and the statement:
        // it is what bumps `version`, and `version` is part of the fingerprint
        // *on purpose* (a rolled-back statement must restore it), so calling it
        // twice would fail on a field the statement never touched.
        let writer_before = {
            let graph = make_dir_graph_mut(&mut writer);
            assert!(
                graph.graph.is_forked(),
                "{name}: precondition — a held reader must produce an overlay"
            );
            let before = fingerprint(&mut graph.clone());
            // Fails after its first write: row 1 commits, row 2 violates the
            // write scope.
            expect_failure(
                graph,
                "CREATE (:Item {id: 4000, name: 'first'}), (:Blocked {id: 4001, name: 'second'})",
                Some(&["Item"]),
            );
            before
        };

        assert_eq!(
            fingerprint(&mut (*reader).clone()),
            reader_before,
            "{name}: the reader's graph must be untouched by a write it never \
             asked for — a difference here means the undo journal reversed into \
             the shared base instead of the overlay (D2 R3)"
        );
        assert_eq!(
            fingerprint(&mut (*writer).clone()),
            writer_before,
            "{name}: the writer's failed statement must roll back exactly, \
             overlay or not"
        );
    }
}

/// The forked backend must take the **journal** path, not the clone checkpoint
/// — and the one write it cannot express must cost exactly one copy, not one
/// per statement.
///
/// D2 risk R2: `journal_covers` has exactly one term left,
/// `supports_undo_journal()`. If `Forked` answered `false` there, every
/// statement taken while a view is held would open a
/// `StatementCheckpoint::Clone` — an O(V+E) copy *per statement* instead of the
/// one-off fork this phase removed, i.e. the fix introducing a cliff worse than
/// the defect. The zeros below are what would break.
///
/// The middle assertion is the honest half. An overlay cannot express an
/// adjacency edit (`storage/forked.rs` module doc), so the edge `CREATE`
/// flattens the overlay — one deep copy, the pre-D2 cost. What matters is that
/// it happens **once**: the backend is a plain `Memory` afterwards, so every
/// later statement is back to mutating in place. A per-statement copy here
/// would be the R2 accident wearing a different hat.
#[test]
fn forked_statements_copy_zero_nodes_except_one_flatten() {
    use crate::graph::handle::make_dir_graph_mut;
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};
    use std::sync::Arc;

    /// Overlay-expressible: node adds and weight writes.
    const OVERLAY_QUERIES: &[&str] = &[
        "CREATE (:Item {id: 2000, name: 'x'})",
        "MATCH (n:Item {id: 1}) SET n.qty = 11, n.name = 'renamed'",
        "MATCH (n:Item {id: 2000}) SET n:Featured",
        "MERGE (n:Item {id: 2001}) ON CREATE SET n.name = 'merged'",
    ];
    /// Rewrites existing nodes' petgraph adjacency, so it flattens first.
    const ADJACENCY_QUERY: &str =
        "MATCH (a:Item {id: 1}), (b:Item {id: 3}) CREATE (a)-[:LINKS {weight: 2}]->(b)";

    let mut writer = Arc::new(seeded());
    let reader = Arc::clone(&writer);
    let fixture_nodes = reader.graph.node_count();

    let graph = make_dir_graph_mut(&mut writer);
    assert!(graph.graph.is_forked(), "precondition: the write forked");

    for &query in OVERLAY_QUERIES {
        reset_backend_clone_count();
        run(graph, query);
        assert_eq!(
            backend_clone_nodes(),
            0,
            "an overlay-expressible statement on a forked backend must copy no node: {query}"
        );
        assert!(
            graph.graph.is_forked(),
            "...and must leave the backend forked: {query}"
        );
    }

    reset_backend_clone_count();
    run(graph, ADJACENCY_QUERY);
    assert_eq!(
        backend_clone_nodes(),
        fixture_nodes,
        "the adjacency write flattens the overlay — exactly one copy of the base"
    );
    assert!(
        !graph.graph.is_forked(),
        "flattening must leave a plain backend, so the copy is paid once"
    );

    reset_backend_clone_count();
    run(graph, "MATCH (n:Item {id: 2000}) DETACH DELETE n");
    run(graph, "CREATE (:Item {id: 2002, name: 'after'})");
    assert_eq!(
        backend_clone_nodes(),
        0,
        "after flattening, later statements mutate in place — one copy per fork, \
         not one per statement"
    );

    // The reader is still holding the pre-fork base and must be untouched by
    // any of it.
    assert_eq!(reader.graph.node_count(), fixture_nodes);
}

/// A replace-all write on a forked columnar row must roll back the cells it
/// **nulled**, not just the ones it wrote.
///
/// `GraphWrite::replace_node_properties` is the only write whose pre-image
/// spans keys the caller never mentioned: it clears the row first, so a journal
/// that captured the incoming keys alone would restore the batch's own columns
/// and leave every omitted property permanently null — a rolled-back statement
/// deleting data. Reached through the trait rather than through Cypher because
/// its one caller is `add_nodes` replace-mode (`mutation::batch`), which is not
/// a statement and opens no checkpoint of its own; the checkpoint here is what
/// a caller running it inside one would get.
///
/// The forked fixture is the point: the overlay's arm is a separate
/// implementation from the two heap writers, and until 2026-08-13 it was not
/// replace-all at all, so nothing had ever journalled a `ReplaceRow` on it.
#[test]
fn a_replace_write_on_a_forked_columnar_row_rolls_back_the_cells_it_nulled() {
    use crate::graph::dir_graph::rollback::StatementCheckpoint;
    use crate::graph::handle::make_dir_graph_mut;
    use crate::graph::storage::GraphWrite;
    use std::sync::Arc;

    let mut writer = Arc::new(seeded_columnar());
    let _reader = Arc::clone(&writer);
    let graph = make_dir_graph_mut(&mut writer);
    assert!(
        graph.graph.is_forked(),
        "precondition: a held view must fork the writer"
    );

    let idx = graph
        .lookup_by_id("Item", &Value::Int64(1))
        .expect("the fixture must carry Item id 1");
    // Interned before the fingerprint so the write, not the interning, is what
    // the comparison is about.
    let name = graph.interner.get_or_intern("name");
    let fresh = graph.interner.get_or_intern("fresh");

    let before = fingerprint(&mut graph.clone());

    let checkpoint = StatementCheckpoint::open(graph);
    GraphWrite::replace_node_properties(
        &mut graph.graph,
        idx,
        vec![
            (name, Value::String("replaced".into())),
            (fresh, Value::Int64(42)),
        ],
    );

    // Non-vacuity: the write has to have nulled `qty` and grown the schema,
    // or the rollback below is restoring nothing.
    let mid = fingerprint(&mut graph.clone());
    assert_ne!(before, mid, "the replace must have changed the graph");
    let row = graph
        .graph
        .node_weight(idx)
        .and_then(|node| match &node.properties {
            PropertyStorage::Columnar(row) => Some(row.row_id()),
            _ => None,
        })
        .expect("the fixture node must be a columnar row");
    let qty = graph.interner.get_or_intern("qty");
    let store = graph.column_store("Item").expect("master store");
    assert!(
        store.get(row, qty).is_none(),
        "precondition: the replace must have dropped the property it omitted, \
         or this test cannot see whether the rollback restores it"
    );

    checkpoint.rollback(graph);

    assert_eq!(
        fingerprint(&mut graph.clone()),
        before,
        "a rolled-back replace must restore the whole row — the cells it wrote \
         and the ones it nulled"
    );
}
