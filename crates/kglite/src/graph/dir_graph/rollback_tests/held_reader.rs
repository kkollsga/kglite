//! Rollback while a reader holds the base

use super::unique_claims::{seeded_with_unique_name, unique_fingerprint};
use super::*;

/// A failed statement run while a reader is holding the graph must leave
/// **both** graphs exactly as they were.
///
/// This is the one unforgivable failure mode in the whole held-reader design:
/// every `UndoEntry` is keyed on a `NodeIndex`/`EdgeIndex` and is reversed
/// through `GraphWrite`. On a `Forked` backend that reversal must land in the
/// *overlay*. If any of it reached the shared base instead, the reader's
/// snapshot would silently acquire a rolled-back write — no error, no crash,
/// and no other test in this file would see it, because every other test owns
/// its graph outright.
///
/// The reader side is the golden snapshot: `fingerprint` is deliberately
/// over-specified (petgraph slot identity, inverted-index bucket *order*, schema
/// metadata, master column rows), so *any* base mutation shows up here, not just
/// one that changes a value a query would return.
///
/// The **unique** arm covers what `fingerprint` structurally cannot: unique
/// claims live in `unique_indices`, which the fingerprint does not read at all
/// (it is parked by `swap_data_scale` and restored by a per-type rebuild, not
/// by the journal). Its rollback path is therefore a *different* mechanism, and
/// until 2026-08-15 it had never been run on a forked backend — where the
/// journal's inverse ops replay through the overlay while the rebuild reads the
/// types those ops reported. A phantom or lost claim here rejects or admits
/// writes forever after, and no other assertion in this file can see it.
#[test]
fn a_rollback_while_a_reader_is_held_touches_neither_graph() {
    use crate::graph::handle::make_dir_graph_mut;
    use std::sync::Arc;

    for (name, build) in [
        ("plain", seeded as fn() -> DirGraph),
        ("columnar", seeded_columnar as fn() -> DirGraph),
        ("indexed", seeded_indexed as fn() -> DirGraph),
        ("unique", seeded_with_unique_name as fn() -> DirGraph),
    ] {
        let mut writer = Arc::new(build());
        let reader = Arc::clone(&writer);

        let claims_before = unique_fingerprint(&reader);
        assert_eq!(
            !claims_before.is_empty(),
            name == "unique",
            "{name}: only the unique arm may hold declared constraints, and it must — \
             otherwise the claim assertions below are vacuous"
        );

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
        assert_eq!(
            unique_fingerprint(&writer),
            claims_before,
            "{name}: the failed statement claimed 'first' and must have released it — \
             a claim surviving a rollback rejects that value forever"
        );
        assert_eq!(
            unique_fingerprint(&reader),
            claims_before,
            "{name}: the reader's occupancy map must be untouched by the writer's \
             constraint bookkeeping"
        );
    }
}

/// The **clone fallback**: a graph whose free lists are non-empty cannot be
/// forked, so a write taken while a reader holds it must deep-copy — and be
/// correct.
///
/// `forked::can_fork` is the predicate, and its unit tests cover the predicate
/// alone. Nothing covered what the `DirGraph` above it then *does*: every other
/// held-reader test in this file seeds a graph that has never deleted anything,
/// so they all take the overlay path and the fallback branch of
/// `GraphBackend::clone` was reached by no test at all. It is the branch that
/// has to stay right when the fast one cannot run — a deep copy that shared the
/// base instead (the `g.clone()`-instead-of-`deep_clone()` slip the code's own
/// comment warns about, one character wide) would let every later write mutate
/// the reader's snapshot in place.
///
/// Both directions are pinned: the fallback must fire *and* cost exactly one
/// whole-graph copy, so neither "it forked after a delete" (slot identity
/// broken) nor "it copied per statement" (the cliff since removed) can pass.
#[test]
fn a_write_under_a_held_reader_after_a_delete_takes_the_clone_path() {
    use crate::graph::handle::make_dir_graph_mut;
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};
    use std::sync::Arc;

    let mut writer = Arc::new(seeded());
    // A delete is what puts a slot on petgraph's free list; from here the
    // overlay cannot reproduce append indices, so `can_fork` refuses.
    run(
        Arc::make_mut(&mut writer),
        "MATCH (n:Item {id: 3}) DETACH DELETE n",
    );
    let live_nodes = writer.graph.node_count();

    let reader = Arc::clone(&writer);
    let reader_before = fingerprint(&mut (*reader).clone());

    reset_backend_clone_count();
    let graph = make_dir_graph_mut(&mut writer);
    assert!(
        !graph.graph.is_forked(),
        "a graph with a non-empty free list must NOT fork — the overlay hands out \
         append indices petgraph would reuse from the free list, and the fold-back \
         would then mis-key every DirGraph index recorded against them"
    );
    assert_eq!(
        backend_clone_nodes(),
        live_nodes,
        "the fallback is a genuine deep copy of the base, not an Arc share"
    );

    // Correct, not merely separate: the write lands, the delete stays deleted,
    // and a re-created id is found by the id index rather than the tombstone.
    run(graph, "CREATE (:Item {id: 4, name: 'd', qty: 40})");
    run(graph, "CREATE (:Item {id: 3, name: 'c-again', qty: 33})");
    assert_eq!(item_prop(graph, 4, "qty"), Some(Value::Int64(40)));
    assert_eq!(item_prop(graph, 3, "qty"), Some(Value::Int64(33)));
    assert_eq!(
        graph.graph.node_count(),
        live_nodes + 2,
        "both creates must be live nodes"
    );

    reset_backend_clone_count();
    run(graph, "CREATE (:Item {id: 5, name: 'e', qty: 50})");
    assert_eq!(
        backend_clone_nodes(),
        0,
        "the copy is paid once at the fork point — a per-statement copy here is the \
         cliff the overlay exists to remove, wearing the fallback's hat"
    );

    assert_eq!(
        fingerprint(&mut (*reader).clone()),
        reader_before,
        "the reader's snapshot must be untouched by every one of those writes"
    );
    let mut reader_now = (*reader).clone();
    assert_eq!(
        reader_now.lookup_by_id("Item", &Value::Int64(4)),
        None,
        "none of the writer's three creates may be visible through the reader — the \
         observable half of the deep copy, in the direction a shared backend breaks"
    );
    assert_eq!(
        reader_now.lookup_by_id("Item", &Value::Int64(3)),
        None,
        "and the id the writer re-created must still read as deleted here"
    );
}

/// The forked backend must take the **journal** path, not the clone checkpoint
/// — and the one write it cannot express must cost exactly one copy, not one
/// per statement.
///
/// `journal_covers` has exactly one term left, `supports_undo_journal()`. If
/// `Forked` answered `false` there, every statement taken while a view is held
/// would open a `StatementCheckpoint::Clone` — an O(V+E) copy *per statement*
/// instead of the one-off fork the held-view path performs, i.e. the fix
/// introducing a cliff worse than the defect. The zeros below are what would
/// break.
///
/// The middle assertion is the honest half. An overlay cannot express an
/// adjacency edit (`storage/forked.rs` module doc), so the edge `CREATE`
/// flattens the overlay — one deep copy, the cost every statement used to pay.
/// What matters is that it happens **once**: the backend is a plain `Memory`
/// afterwards, so every later statement is back to mutating in place. A
/// per-statement copy here would be that same accident wearing a different hat.
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

/// An edge-property write taken while a reader holds the base must be visible
/// to every later read of that edge, including the iterating ones.
///
/// `GraphRead::edge_weight` is a point lookup and probes an overlay first, but
/// pattern matching reaches edges through `edges_directed_filtered` /
/// `edges_directed` / `edge_references`, which hand out `&EdgeData` borrowed
/// from whatever store the backend iterates. A backend that answered those from
/// a base while answering the point lookup from a delta returns the pre-write
/// weight to any `WHERE` on a relationship variable — a silent wrong answer,
/// with no error and nothing else in this file able to see it (every other
/// fixture owns its graph outright, so nothing forks). Until 2026-08-23 the
/// forked backend did exactly that: `RETURN r.weight` read 99 while
/// `WHERE r.weight = 99` matched nothing.
///
/// Both arms are their own held-reader episode on purpose. The seam is
/// `GraphBackend::edge_weight_mut`, and both verbs reach it, but only the write
/// that runs *while the backend is still forked* can see whether it does.
///
/// The clone counts are the cost half of the same contract: the write flattens
/// the overlay, and it does so **once** — the backend is a plain `Memory`
/// afterwards, so a second edge write copies nothing. A per-write copy would be
/// the fix introducing a cliff worse than the defect.
///
/// The reader assertion is the third half: the write is the writer's alone, so
/// the held snapshot must still read the pre-write weight.
#[test]
fn an_edge_property_write_under_a_held_reader_reaches_traversal_reads() {
    use crate::graph::handle::make_dir_graph_mut;
    use crate::graph::session::execute::execute_read;
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};
    use std::sync::Arc;

    const PAIR: &str = "MATCH (:Item {id: 1})-[r:LINKS]->(:Item {id: 2})";

    for (verb, predicate) in [
        ("SET r.weight = 99", "r.weight = 99"),
        ("REMOVE r.weight", "r.weight IS NULL"),
    ] {
        let mut writer = Arc::new(seeded());
        let reader = Arc::clone(&writer);
        let fixture_nodes = reader.graph.node_count();

        let graph = make_dir_graph_mut(&mut writer);
        assert!(
            graph.graph.is_forked(),
            "{verb}: precondition — the held reader must have forked the write"
        );

        reset_backend_clone_count();
        run(graph, &format!("{PAIR} {verb}"));
        assert_eq!(
            backend_clone_nodes(),
            fixture_nodes,
            "{verb}: the edge-property write flattens the overlay — one copy of the base"
        );
        assert!(
            !graph.graph.is_forked(),
            "{verb}: flattening must leave a plain backend, so the copy is paid once"
        );

        reset_backend_clone_count();
        run(
            graph,
            "MATCH (:Item {id: 2})-[r:LINKS]->(:Item {id: 3}) SET r.weight = 77",
        );
        assert_eq!(
            backend_clone_nodes(),
            0,
            "{verb}: after flattening, a second edge-property write copies nothing"
        );

        let params = HashMap::new();
        let opts = ExecuteOptions::new(&params);

        // The filter is the shape that silently drops rows — it reads the
        // weight through the traversal iterator rather than by edge index.
        let filtered = execute_read(
            graph,
            &format!("MATCH (:Item)-[r:LINKS]->(:Item) WHERE {predicate} RETURN count(r) AS c"),
            &opts,
        )
        .unwrap_or_else(|e| panic!("{verb}: filter query failed: {e}"))
        .result
        .rows;
        assert_eq!(
            filtered,
            vec![vec![Value::Int64(1)]],
            "{verb}: a WHERE on the edge property must see the write, and match \
             only the edge it touched"
        );

        // ...and the projection, which reads it by edge index.
        let projected = execute_read(graph, &format!("{PAIR} RETURN r.weight AS w"), &opts)
            .unwrap_or_else(|e| panic!("{verb}: projection query failed: {e}"))
            .result
            .rows;
        let expected = if verb.starts_with("SET") {
            Value::Int64(99)
        } else {
            Value::Null
        };
        assert_eq!(
            projected,
            vec![vec![expected]],
            "{verb}: the projection must agree with the filter"
        );

        // The reader keeps the pre-write graph.
        let held = execute_read(&reader, &format!("{PAIR} RETURN r.weight AS w"), &opts)
            .unwrap_or_else(|e| panic!("{verb}: held-reader query failed: {e}"))
            .result
            .rows;
        assert_eq!(
            held,
            vec![vec![Value::Int64(5)]],
            "{verb}: the held snapshot must not observe the writer's edit"
        );
    }
}
