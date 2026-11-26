//! Reads that need only an edge's connection type must not materialise the
//! edge on a disk graph.
//!
//! `GraphEdgeRef::weight()` on the disk backend builds the edge's `EdgeData`
//! (properties included); `connection_type()` reads the CSR endpoint table.
//! Every case below filters, groups or counts by type alone, and used to call
//! `weight()` to do it: a two-hop count on a large disk graph built a record
//! per edge, and `describe()`'s neighbor schema one per incident edge and
//! node. The test-only materialization counter pins that none is built.

use crate::datatypes::Value;
use crate::graph::introspection::schema_overview::{
    compute_all_neighbors_schemas, compute_neighbors_schema,
};
use crate::graph::schema::DirGraph;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::storage::GraphRead;
use std::collections::HashMap;

const NODES: i64 = 40;

/// `NODES` `P` nodes in a `KNOWS` chain whose edges carry a property, moved
/// onto the disk backend.
fn chain() -> DirGraph {
    let mut graph = DirGraph::new();
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    for query in [
        format!("UNWIND range(1, {NODES}) AS i CREATE (:P {{id: i}})"),
        "MATCH (a:P), (b:P) WHERE b.id = a.id + 1 CREATE (a)-[:KNOWS {w: a.id}]->(b)".into(),
        "CREATE (:P {id: 1000})".into(),
    ] {
        execute_mut(&mut graph, &query, &opts).unwrap_or_else(|e| panic!("{query}: {e}"));
    }
    graph.enable_disk_mode().unwrap();
    assert!(graph.graph.is_disk());
    graph
}

fn count(graph: &DirGraph, query: &str) -> Value {
    let params = HashMap::new();
    let out = execute_read(graph, query, &ExecuteOptions::eager(&params))
        .unwrap_or_else(|e| panic!("{query}: {e}"));
    out.result.rows[0][0].clone()
}

#[test]
fn type_only_cypher_reads_materialise_no_edge() {
    let graph = chain();
    let disk = graph.graph.as_disk().expect("disk-backed");
    let cases = [
        ("MATCH (a:P)-[:KNOWS]->(b) RETURN count(*)", NODES - 1),
        (
            "MATCH (a:P)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN count(*)",
            NODES - 2,
        ),
        (
            "MATCH (a:P) WHERE EXISTS { (a)-[:KNOWS]->() } RETURN count(a)",
            NODES - 1,
        ),
        (
            "CALL orphan_node({type: 'P', link_type: 'KNOWS'}) YIELD node RETURN count(node)",
            1,
        ),
        (
            "CALL cardinality_violation({type: 'P', edge: 'KNOWS', max: 0}) \
             YIELD node, count RETURN count(node)",
            NODES - 1,
        ),
    ];
    let mut materialised = Vec::new();
    for (query, expected) in cases {
        disk.reset_materialization_counts();
        assert_eq!(count(&graph, query), Value::Int64(expected), "{query}");
        if disk.edge_materializations() > 0 {
            materialised.push((query, disk.edge_materializations()));
        }
    }
    assert!(materialised.is_empty(), "{materialised:#?}");
}

#[test]
fn neighbor_schemas_materialise_nothing_on_disk() {
    let graph = chain();
    let disk = graph.graph.as_disk().expect("disk-backed");
    disk.reset_materialization_counts();

    let one = compute_neighbors_schema(&graph, "P").unwrap();
    let all = compute_all_neighbors_schemas(&graph);
    assert_eq!(
        (disk.edge_materializations(), disk.node_materializations()),
        (0, 0)
    );

    let summary = |list: &[crate::graph::introspection::NeighborConnection]| {
        list.iter()
            .map(|c| (c.connection_type.clone(), c.other_type.clone(), c.count))
            .collect::<Vec<_>>()
    };
    let expected = vec![("KNOWS".to_string(), "P".to_string(), (NODES - 1) as usize)];
    assert_eq!(summary(&one.outgoing), expected);
    assert_eq!(summary(&one.incoming), expected);
    assert_eq!(summary(&all["P"].outgoing), expected);
    assert_eq!(summary(&all["P"].incoming), expected);
}
