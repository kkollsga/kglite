//! `REMOVE` on an in-memory Columnar node must clear the graph master store.
//!
//! Split into its own file to keep `write.rs` under the source-quality line
//! ceiling.
//!
//! Each node of a columnar type holds its own `Arc<ColumnStore>` clone for
//! cheap property reads, and `graph.column_stores` holds the master. Writing
//! through the node's handle calls `Arc::make_mut`, which **forks** it: the
//! node sees the write and the master does not. `execute_set` avoids this by
//! writing through the master and refreshing the per-node handles in one sweep
//! at the end of the clause; `execute_remove` did not, so a removed property
//! survived in the master and came back the moment any later clause ran that
//! sweep.

use crate::datatypes::Value;
use crate::graph::schema::{DirGraph, InternedKey, PropertyStorage};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphRead;

fn run(graph: &mut DirGraph, query: &str) {
    let params = std::collections::HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("query failed: {query}: {e}"));
}

/// One columnar `Item`, with the fixture's columnar-ness asserted rather than
/// assumed — every test here is vacuous against a `Map`-storage node.
fn columnar_item() -> DirGraph {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (a:Item {id: 1, name: 'a', qty: 10})");
    graph.enable_columnar();
    assert!(
        graph.column_stores.contains_key("Item"),
        "the fixture must own a master column store, or these tests are vacuous"
    );
    let idx = graph
        .lookup_by_id_readonly("Item", &Value::Int64(1))
        .expect("seeded node");
    assert!(
        matches!(
            graph.graph.node_weight(idx).map(|n| &n.properties),
            Some(PropertyStorage::Columnar { .. })
        ),
        "the node must read through a column store, or these tests are vacuous"
    );
    graph
}

fn master_qty(graph: &DirGraph) -> Option<Value> {
    graph
        .column_stores
        .get("Item")
        .and_then(|m| m.get(0, InternedKey::from_str("qty")))
}

fn node_qty(graph: &DirGraph) -> Option<Value> {
    graph
        .lookup_by_id_readonly("Item", &Value::Int64(1))
        .and_then(|i| graph.graph.node_weight(i))
        .and_then(|n| n.get_property("qty"))
        .map(|v| v.into_owned())
}

/// The removal must reach the master, not just the node's forked handle.
#[test]
fn remove_clears_the_master_column_store() {
    let mut graph = columnar_item();
    assert_eq!(master_qty(&graph), Some(Value::Int64(10)));

    run(&mut graph, "MATCH (a:Item {id: 1}) REMOVE a.qty");

    assert_eq!(node_qty(&graph), None, "the node must not see the property");
    assert_eq!(
        master_qty(&graph),
        None,
        "the master must not keep a value the node no longer has — it is what \
         save() persists and what the next handle-refresh sweep broadcasts"
    );
}

/// The user-visible symptom: a removed property comes back.
///
/// No save is involved. `SET a.other = 5` writes through the master and then
/// refreshes every node handle of the type, which re-points the node at a
/// master that still carried `qty`.
#[test]
fn removed_columnar_property_does_not_resurrect_on_the_next_set() {
    let mut graph = columnar_item();
    run(&mut graph, "MATCH (a:Item {id: 1}) REMOVE a.qty");
    assert_eq!(node_qty(&graph), None);

    // Any later master-routed write on the same type triggers the sweep.
    run(&mut graph, "MATCH (a:Item {id: 1}) SET a.other = 5");

    assert_eq!(
        node_qty(&graph),
        None,
        "a removed property must stay removed across a later SET on its type"
    );
    assert_eq!(master_qty(&graph), None);
}

/// REMOVE over several rows must still be correct per node — the batched
/// handle refresh happens once at the end of the clause, so a bug there shows
/// up as one node keeping its value.
#[test]
fn remove_over_many_rows_clears_every_node() {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Item {id: 1, qty: 10}), (:Item {id: 2, qty: 20}), (:Item {id: 3, qty: 30})",
    );
    graph.enable_columnar();
    assert!(graph.column_stores.contains_key("Item"));

    run(&mut graph, "MATCH (a:Item) REMOVE a.qty");

    for id in 1..=3 {
        let value = graph
            .lookup_by_id_readonly("Item", &Value::Int64(id))
            .and_then(|i| graph.graph.node_weight(i))
            .and_then(|n| n.get_property("qty"))
            .map(|v| v.into_owned());
        assert_eq!(value, None, "node {id} kept its removed property");
    }
    // And a later SET does not bring any of them back.
    run(&mut graph, "MATCH (a:Item {id: 1}) SET a.other = 1");
    for id in 1..=3 {
        let value = graph
            .lookup_by_id_readonly("Item", &Value::Int64(id))
            .and_then(|i| graph.graph.node_weight(i))
            .and_then(|n| n.get_property("qty"))
            .map(|v| v.into_owned());
        assert_eq!(value, None, "node {id} resurrected its removed property");
    }
}
