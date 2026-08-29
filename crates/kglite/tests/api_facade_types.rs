//! Compile-fire for the facade paths a Rust downstream needs *by name*.
//!
//! Every item is reached through `kglite::api::…` only — never
//! `kglite::datatypes::…` or the sealed `kglite::graph::…` — and every one is
//! named in a binding or a signature position rather than merely called, so
//! dropping any re-export breaks the build instead of quietly demoting a
//! downstream back to a hand-mirrored copy.

use kglite::api::introspection::{
    compute_neighbors_schema, compute_property_stats, compute_schema, compute_type_connectivity,
    derive_edge_counts_from_triples, graph_scale, ConnectivityTriple, DerivedEdgeStats, GraphScale,
    NeighborConnection, NeighborsSchema, NodeTypeOverview, PropertyStatInfo,
};
use kglite::api::mutation::{add_connections, add_nodes, ColumnData, ColumnType, DataFrame};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{
    DirGraph, Direction, GraphEdgeRef, GraphInfo, GraphRead, NodeIndex, PropMap, Value,
};

/// Two people and one edge between them — enough for every accessor below to
/// have a non-empty answer, so a re-export cannot be "proven" by an empty
/// collection.
fn two_person_graph() -> DirGraph {
    let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create graph");

    let mut nodes = DataFrame::new(Vec::new());
    nodes
        .add_column(
            "id".to_string(),
            ColumnType::String,
            ColumnData::String(vec![Some("a".to_string()), Some("b".to_string())]),
        )
        .expect("id column");
    nodes
        .add_column(
            "age".to_string(),
            ColumnType::Int64,
            ColumnData::Int64(vec![Some(30), Some(41)]),
        )
        .expect("age column");
    // `ColumnData::Map` cells are `PropMap`s, so a Rust caller cannot build a
    // map column without naming that type either.
    let meta: PropMap = PropMap::from_pairs(vec![("rank".to_string(), Value::Int64(1))]);
    nodes
        .add_column(
            "meta".to_string(),
            ColumnType::Map,
            ColumnData::Map(vec![Some(meta), Some(PropMap::new())]),
        )
        .expect("meta column");
    assert_eq!(nodes.row_count(), 2);
    assert_eq!(nodes.column_count(), 3);
    add_nodes(
        &mut graph,
        nodes,
        "Person".to_string(),
        "id".to_string(),
        None,
        None,
    )
    .expect("add nodes");

    let mut edges = DataFrame::new(Vec::new());
    edges
        .add_column(
            "src".to_string(),
            ColumnType::String,
            ColumnData::String(vec![Some("a".to_string())]),
        )
        .expect("src column");
    edges
        .add_column(
            "tgt".to_string(),
            ColumnType::String,
            ColumnData::String(vec![Some("b".to_string())]),
        )
        .expect("tgt column");
    add_connections(
        &mut graph,
        edges,
        "KNOWS".to_string(),
        "Person".to_string(),
        "src".to_string(),
        "Person".to_string(),
        "tgt".to_string(),
        None,
        None,
        None,
    )
    .expect("add connections");

    graph
}

#[test]
fn dataframe_ingest_is_buildable_through_the_facade() {
    let graph = two_person_graph();
    assert_eq!(graph.graph.node_count(), 2);
    assert_eq!(graph.graph.edge_count(), 1);
}

#[test]
fn introspection_result_types_are_nameable_through_the_facade() {
    let graph = two_person_graph();

    let triples: Vec<ConnectivityTriple> = compute_type_connectivity(&graph);
    assert_eq!(triples.len(), 1);
    assert_eq!(triples[0].src, "Person");
    assert_eq!(triples[0].conn, "KNOWS");
    assert_eq!(triples[0].tgt, "Person");
    assert_eq!(triples[0].count, 1);

    let derived: DerivedEdgeStats = derive_edge_counts_from_triples(&triples);
    assert_eq!(derived.counts.get("KNOWS"), Some(&1));
    assert!(derived.endpoints.contains_key("KNOWS"));

    let overview = compute_schema(&graph);
    let (type_name, per_type): &(String, NodeTypeOverview) =
        overview.node_types.first().expect("one node type");
    assert_eq!(type_name, "Person");
    assert_eq!(per_type.count, 2);

    let neighbors: NeighborsSchema = compute_neighbors_schema(&graph, "Person").expect("neighbors");
    let outgoing: &NeighborConnection = neighbors.outgoing.first().expect("one outgoing");
    assert_eq!(outgoing.connection_type, "KNOWS");
    assert_eq!(outgoing.other_type, "Person");

    let props: Vec<PropertyStatInfo> =
        compute_property_stats(&graph, "Person", 10, None).expect("property stats");
    assert!(props.iter().any(|p| p.property_name == "age"));

    let info: GraphInfo = graph.graph_info();
    assert_eq!(info.node_count, 2);
    assert_eq!(info.edge_count, 1);
}

#[test]
fn graph_scale_is_nameable_and_comparable_through_the_facade() {
    let graph = two_person_graph();
    let scale: GraphScale = graph_scale(&graph);
    // Debug/PartialEq are part of the exported contract: a downstream that
    // mirrored the enum matched on it, and `assert_eq!` needs both.
    assert_eq!(scale, GraphScale::Small);
    assert!(matches!(scale, GraphScale::Small));
}

#[test]
fn edge_iterators_yield_a_nameable_reference_type() {
    let graph = two_person_graph();
    let start: NodeIndex = graph
        .graph
        .node_indices()
        .next()
        .expect("at least one node");

    let outgoing: Vec<GraphEdgeRef<'_>> = graph
        .graph
        .edges_directed(start, Direction::Outgoing)
        .collect();
    let incoming: Vec<GraphEdgeRef<'_>> = graph
        .graph
        .edges_directed(start, Direction::Incoming)
        .collect();
    assert_eq!(outgoing.len() + incoming.len(), 1);

    let edge: &GraphEdgeRef<'_> = outgoing.first().or_else(|| incoming.first()).expect("edge");
    assert_ne!(edge.source(), edge.target());
    assert_eq!(
        graph.interner.resolve(edge.connection_type()),
        "KNOWS",
        "connection_type() must not need the EdgeData materialisation"
    );
    assert_eq!(edge.weight().connection_type, edge.connection_type());
}
