//! The per-row node flush writes in place: a property-`SET` statement copies
//! no column (the cell journal stands in for the checkpoint's share), and a
//! statement the journal does not cover copies each touched column once.
use crate::datatypes::Value;
use crate::graph::schema::DirGraph;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::column_store::{
    column_clones, column_store_clones, reset_column_clones, reset_column_store_clones,
};
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use crate::graph::storage::{GraphRead, GraphWrite};
use std::collections::HashMap;

const ROWS: i64 = 10_000;
const WRITTEN: i64 = 500;

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("{query}: {e}"));
}

/// A disk graph of `ROWS` `T` nodes carrying an int64 column (`x`) and a
/// datetime column (`ts`), which the columnar store holds as `Mixed`.
fn disk_graph(dir: &std::path::Path) -> DirGraph {
    let mut graph = new_dir_graph_in_mode(StorageMode::Disk, Some(dir)).expect("disk graph");
    run(
        &mut graph,
        &format!(
            "UNWIND range(1, {ROWS}) AS i \
             CREATE (:T {{id: i, x: 0, ts: datetime('2019-01-01T00:00:00Z')}})"
        ),
    );
    assert!(graph.graph.is_disk());
    graph
}

fn prop(graph: &DirGraph, id: i64, key: &str) -> Option<Value> {
    let idx = graph
        .graph
        .node_indices()
        .find(|i| graph.graph.get_node_id(*i) == Some(Value::Int64(id)))
        .expect("node exists");
    graph
        .graph
        .node_view(idx)
        .and_then(|n| n.get_property_value(key))
}

/// A statement writing two columns over `WRITTEN` rows used to deep-copy each
/// touched column once per row (the flush cloned the store while the map
/// still held it): 1,000 column copies here, each O(type rows). Written in
/// place, the statement's checkpoint took one copy of each touched column at
/// its first write; the cell journal takes none.
#[test]
fn a_disk_set_statement_copies_no_column() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = disk_graph(dir.path());

    reset_column_clones();
    reset_column_store_clones();
    run(
        &mut graph,
        &format!(
            "UNWIND range(1, {WRITTEN}) AS i MATCH (n:T {{id: i}}) \
             SET n.x = i, n.ts = datetime('2020-01-01T00:00:00Z')"
        ),
    );
    let columns = column_clones();
    let stores = column_store_clones();

    assert_eq!(
        (columns, stores),
        (0, 0),
        "one statement writing 2 columns over {WRITTEN} rows of a {ROWS}-row disk type \
         copied {columns} columns and {stores} whole stores; the cell journal copies none. \
         A reading near {} is the per-row clone-and-replace flush.",
        2 * WRITTEN
    );
    // Non-vacuity: the writes landed.
    assert_eq!(prop(&graph, 1, "x"), Some(Value::Int64(1)));
    assert_eq!(prop(&graph, WRITTEN, "x"), Some(Value::Int64(WRITTEN)));
    assert_eq!(prop(&graph, WRITTEN + 1, "x"), Some(Value::Int64(0)));
}

/// With nothing else holding the store (no statement checkpoint, no fork, no
/// held view), the flush owns it outright and copies nothing.
#[test]
fn an_unshared_disk_flush_copies_no_column() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = disk_graph(dir.path());
    let x = graph.interner.get_or_intern("x");
    let targets: Vec<_> = graph
        .graph
        .node_indices()
        .filter(|i| matches!(graph.graph.get_node_id(*i), Some(Value::Int64(v)) if v <= 50))
        .collect();
    assert_eq!(targets.len(), 50);

    for round in 1..=2i64 {
        reset_column_clones();
        reset_column_store_clones();
        for idx in &targets {
            graph
                .graph
                .set_node_property(*idx, x, Value::Int64(round * 100));
            graph.graph.flush_pending_writes();
        }
        assert_eq!(
            (column_clones(), column_store_clones()),
            (0, 0),
            "round {round}: an unshared disk flush must write in place"
        );
    }
    assert_eq!(prop(&graph, 7, "x"), Some(Value::Int64(200)));
}

/// A holder of the old store (here a backend clone, the same share a statement
/// checkpoint or transaction fork takes) keeps its values: `make_mut` forks
/// the store for the writer and leaves the holder's `Arc` untouched.
#[test]
fn a_retained_disk_snapshot_keeps_its_values() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = disk_graph(dir.path());
    let snapshot = graph.graph.clone();
    run(&mut graph, "MATCH (n:T {id: 3}) SET n.x = 33");

    let idx = graph
        .graph
        .node_indices()
        .find(|i| graph.graph.get_node_id(*i) == Some(Value::Int64(3)))
        .unwrap();
    assert_eq!(prop(&graph, 3, "x"), Some(Value::Int64(33)));
    assert_eq!(
        snapshot
            .node_view(idx)
            .and_then(|n| n.get_property_value("x")),
        Some(Value::Int64(0)),
        "the retained snapshot must not see the later write"
    );
}
