use super::*;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use tempfile::TempDir;

fn add_pair(graph: &mut DirGraph) {
    let rows = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        vec![vec![Value::Int64(1)], vec![Value::Int64(2)]],
    )
    .unwrap();
    add_nodes(
        graph,
        rows,
        "Doc".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
}

fn link(source_id: i64, target_id: i64) -> EdgeSpec {
    EdgeSpec {
        source_type: "Doc".to_string(),
        source_id: Value::Int64(source_id),
        target_type: "Doc".to_string(),
        target_id: Value::Int64(target_id),
        edge_type: "LINKS".to_string(),
        properties: HashMap::from([("weight".to_string(), Value::Int64(3))]),
    }
}

#[test]
fn edge_specs_mutate_every_storage_mode_and_invalidate_counts() {
    for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
        let tmp = TempDir::new().unwrap();
        let path = (mode == StorageMode::Disk).then_some(tmp.path());
        let mut graph = new_dir_graph_in_mode(mode, path).unwrap();
        add_pair(&mut graph);
        *graph.edge_type_counts_cache.write().unwrap() =
            Some(HashMap::from([("LINKS".to_string(), 0)]));

        let report = add_edges_from_specs(&mut graph, vec![link(1, 2), link(99, 2)]).unwrap();

        assert_eq!(report.connections_created, 1, "mode={mode:?}");
        assert_eq!(report.skipped_missing_endpoint, 1, "mode={mode:?}");
        assert_eq!(graph.graph.edge_count(), 1, "mode={mode:?}");
        assert!(!graph.has_edge_type_counts_cache(), "mode={mode:?}");
    }
}

#[test]
fn edge_specs_interner_preflight_is_atomic() {
    let mut graph = DirGraph::new();
    add_pair(&mut graph);
    let incoming = "LINKS";
    graph
        .interner
        .try_register(InternedKey::from_str(incoming), "conflicting-existing")
        .unwrap();

    let error = add_edges_from_specs(&mut graph, vec![link(1, 2)]).unwrap_err();

    assert!(error.contains("hash collision"));
    assert_eq!(graph.graph.edge_count(), 0);
    assert!(graph.connection_type_metadata.is_empty());
}
