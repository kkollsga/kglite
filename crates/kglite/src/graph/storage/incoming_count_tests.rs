//! Direct count contracts: result rows or scan fallback cannot hide a bad fast counter.
use super::backend::GraphBackend;
use super::{GraphRead, GraphWrite, MappedGraph, MemoryGraph};
use crate::datatypes::Value;
use crate::graph::schema::{EdgeData, InternedKey, NodeData, StringInterner};
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::forked::ForkedGraph;
use crate::graph::storage::recording::RecordingGraph;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn key(name: &str) -> InternedKey {
    InternedKey::from_str(name)
}

fn populate(graph: &mut impl GraphWrite) {
    let mut interner = StringInterner::new();
    for (id, kind) in ["N", "N", "M", "N"].into_iter().enumerate() {
        assert_eq!(
            graph.add_node(NodeData::new(
                Value::Int64(id as i64),
                Value::String(format!("node{id}")),
                kind.to_string(),
                HashMap::new(),
                &mut interner,
            )),
            NodeIndex::new(id)
        );
    }
    for (source, target, kind) in [
        (0, 0, "R"),
        (0, 0, "R"),
        (0, 0, "S"),
        (1, 0, "R"),
        (1, 0, "R"),
        (2, 0, "R"),
        (1, 0, "S"),
        (0, 1, "R"),
    ] {
        graph.add_edge(
            NodeIndex::new(source),
            NodeIndex::new(target),
            EdgeData::new(kind.to_string(), HashMap::new(), &mut interner),
        );
    }
}

fn heap(kind: &str) -> GraphBackend {
    if kind == "mapped" {
        let mut graph = MappedGraph::new();
        populate(&mut graph);
        return GraphBackend::Mapped(Arc::new(graph));
    }
    let mut graph = MemoryGraph::new();
    populate(&mut graph);
    let base = Arc::new(graph);
    match kind {
        "memory" => GraphBackend::Memory(base),
        "forked" => GraphBackend::Forked(Box::new(ForkedGraph::new(base))),
        "recording" => {
            GraphBackend::Recording(Box::new(RecordingGraph::new(GraphBackend::Memory(base))))
        }
        _ => panic!("unknown heap fixture"),
    }
}

fn disk(layout: &str, path: &std::path::Path) -> GraphBackend {
    if layout == "unsorted" {
        let mut graph = MemoryGraph::new();
        populate(&mut graph);
        let disk = DiskGraph::from_stable_digraph(graph.inner_mut(), path).unwrap();
        assert!(!disk.csr_sorted_by_type);
        return GraphBackend::Disk(Box::new(disk));
    }
    let mut graph = DiskGraph::new_at_path(path).unwrap();
    populate(&mut graph);
    assert!(graph.has_overflow());
    if layout == "sorted" {
        assert_eq!(graph.compact().unwrap(), 8);
        assert!(graph.csr_sorted_by_type);
        assert!(!graph.has_overflow());
    } else {
        assert_eq!(layout, "overflow");
    }
    GraphBackend::Disk(Box::new(graph))
}

fn assert_counts(graph: &GraphBackend) {
    let anchor = NodeIndex::new(0);
    for (connection, peer_type, expected) in [
        (None, None, 4),
        (Some("R"), None, 3),
        (Some("S"), None, 1),
        (Some("R"), Some("N"), 2),
        (Some("R"), Some("M"), 1),
        (None, Some("N"), 3),
        (None, Some("M"), 1),
        (Some("missing"), None, 0),
        (None, Some("missing"), 0),
    ] {
        assert_eq!(
            graph
                .count_incoming_nonself_edges_filtered(
                    anchor,
                    connection.map(key),
                    peer_type.map(key),
                    None,
                )
                .unwrap(),
            expected,
            "connection={connection:?}, peer={peer_type:?}"
        );
    }
    // Public directed incidence counts still include each loop in each direction.
    assert_eq!(
        graph
            .count_edges_filtered(anchor, Direction::Incoming, None, None, None)
            .unwrap(),
        7
    );
    assert_eq!(
        graph
            .count_edges_filtered(anchor, Direction::Outgoing, None, None, None)
            .unwrap(),
        4
    );
    assert_eq!(
        graph
            .count_incoming_nonself_edges_filtered(NodeIndex::new(3), None, None, None)
            .unwrap(),
        0
    );
}

#[test]
fn incoming_nonself_preserves_heap_mapped_forked_and_recording_contracts() {
    for kind in ["memory", "mapped", "forked", "recording"] {
        let graph = heap(kind);
        assert_counts(&graph);
        let expired = Instant::now() - Duration::from_secs(1);
        assert_eq!(
            graph.count_incoming_nonself_edges_filtered(
                NodeIndex::new(0),
                Some(key("R")),
                None,
                Some(expired),
            ),
            Err("Query timed out".to_string()),
            "{kind}"
        );
    }
}

#[test]
fn incoming_nonself_preserves_sorted_unsorted_overflow_and_recorded_disk_contracts() {
    let directory = tempfile::tempdir().unwrap();
    for layout in ["sorted", "unsorted", "overflow"] {
        let graph = disk(layout, &directory.path().join(layout));
        assert_counts(&graph);
        let expired = Instant::now() - Duration::from_secs(1);
        assert_eq!(
            graph.count_incoming_nonself_edges_filtered(
                NodeIndex::new(0),
                None,
                None,
                Some(expired),
            ),
            Err("Query timed out".to_string()),
            "{layout}"
        );
        let wrapped = GraphBackend::Recording(Box::new(RecordingGraph::new(graph)));
        assert_counts(&wrapped);
    }
}

#[test]
fn incoming_nonself_ignores_removed_csr_edges_and_new_overflow_loops() {
    let directory = tempfile::tempdir().unwrap();
    let mut graph = disk("sorted", directory.path());
    // Remove one existing parallel b->a edge from CSR using its physical identity.
    let removed = graph
        .edges_directed(NodeIndex::new(0), Direction::Incoming)
        .find(|edge| edge.source() == NodeIndex::new(1) && edge.connection_type() == key("R"))
        .unwrap()
        .id();
    assert!(graph.remove_edge(removed).is_some());
    assert_eq!(
        graph
            .count_incoming_nonself_edges_filtered(NodeIndex::new(0), None, None, None)
            .unwrap(),
        3
    );
    assert_eq!(
        graph
            .count_incoming_nonself_edges_filtered(NodeIndex::new(0), Some(key("R")), None, None)
            .unwrap(),
        2
    );
    assert_eq!(
        graph
            .count_edges_filtered(NodeIndex::new(0), Direction::Incoming, None, None, None)
            .unwrap(),
        6
    );
    let mut interner = StringInterner::new();
    let extra_loop = graph.add_edge(
        NodeIndex::new(0),
        NodeIndex::new(0),
        EdgeData::new("R".into(), HashMap::new(), &mut interner),
    );
    let extra_peer = graph.add_edge(
        NodeIndex::new(2),
        NodeIndex::new(0),
        EdgeData::new("R".into(), HashMap::new(), &mut interner),
    );
    assert!(graph.as_disk().unwrap().has_overflow());
    assert_eq!(
        graph
            .count_incoming_nonself_edges_filtered(NodeIndex::new(0), None, None, None)
            .unwrap(),
        4
    );
    assert_eq!(
        graph
            .count_incoming_nonself_edges_filtered(
                NodeIndex::new(0),
                Some(key("R")),
                Some(key("M")),
                None
            )
            .unwrap(),
        2
    );
    assert!(graph.remove_edge(extra_loop).is_some());
    assert!(graph.remove_edge(extra_peer).is_some());
    assert_eq!(
        graph
            .count_incoming_nonself_edges_filtered(NodeIndex::new(0), None, None, None)
            .unwrap(),
        3
    );
}

#[test]
fn disk_find_edge_respects_removed_unique_edges_and_parallel_survivors() {
    let directory = tempfile::tempdir().unwrap();
    for layout in ["sorted", "overflow"] {
        let mut graph = disk(layout, &directory.path().join(layout));
        let anchor = NodeIndex::new(0);
        let unique_peer = NodeIndex::new(2);
        let unique = graph.find_edge(unique_peer, anchor).unwrap();
        assert_eq!(graph.edge_endpoints(unique), Some((unique_peer, anchor)));
        assert!(graph.remove_edge(unique).is_some());
        assert_eq!(graph.find_edge(unique_peer, anchor), None, "{layout}");

        let parallel_peer = NodeIndex::new(1);
        let mut removed = Vec::new();
        for _ in 0..3 {
            let live = graph.find_edge(parallel_peer, anchor).unwrap();
            assert!(
                !removed.contains(&live),
                "returned a removed edge in {layout}"
            );
            assert_eq!(graph.edge_endpoints(live), Some((parallel_peer, anchor)));
            assert!(graph.remove_edge(live).is_some());
            removed.push(live);
        }
        assert_eq!(graph.find_edge(parallel_peer, anchor), None, "{layout}");
    }
}
