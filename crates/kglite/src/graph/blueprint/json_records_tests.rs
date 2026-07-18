use super::*;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use crate::graph::storage::GraphRead;
use serde_json::json;
use tempfile::TempDir;

fn endpoint_spec(policy: &str) -> Json {
    json!({
        "on_missing_endpoint": policy,
        "nodes": [{
            "type": "Doc",
            "id_field": "id",
            "records": [{"id": 1}, {"id": 2}]
        }],
        "connections": [{
            "type": "LINKS",
            "source_type": "Doc",
            "source_id_field": "source",
            "target_type": "Doc",
            "target_id_field": "target",
            "records": [
                {"source": 1, "target": 2, "weight": 3},
                {"source": 2, "target": 99, "weight": 4},
                {"source": null, "target": 1, "weight": 5}
            ]
        }]
    })
}

#[test]
fn drop_policy_is_consistent_across_storage_modes() {
    for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
        let tmp = TempDir::new().unwrap();
        let path = (mode == StorageMode::Disk).then_some(tmp.path());
        let mut graph = new_dir_graph_in_mode(mode, path).unwrap();

        let report = from_records(&mut graph, &endpoint_spec("drop")).unwrap();

        assert_eq!(report.nodes_added, 2, "mode={mode:?}");
        assert_eq!(report.edges_added, 1, "mode={mode:?}");
        assert_eq!(report.edges_dropped_missing_endpoint, 2, "mode={mode:?}");
        assert_eq!(graph.graph.node_count(), 2, "mode={mode:?}");
        assert_eq!(graph.graph.edge_count(), 1, "mode={mode:?}");
    }
}

#[test]
fn error_policy_reports_the_first_bad_row_and_is_atomic() {
    for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
        let tmp = TempDir::new().unwrap();
        let path = (mode == StorageMode::Disk).then_some(tmp.path());
        let mut graph = new_dir_graph_in_mode(mode, path).unwrap();
        let before_version = graph.version();

        let error = from_records(&mut graph, &endpoint_spec("error")).unwrap_err();

        assert_eq!(
            error,
            "from_records: connections[0].records[1]: target endpoint Doc(99) does not exist"
        );
        assert_eq!(graph.graph.node_count(), 0, "mode={mode:?}");
        assert_eq!(graph.graph.edge_count(), 0, "mode={mode:?}");
        assert_eq!(graph.version(), before_version, "mode={mode:?}");
    }
}

#[test]
fn error_policy_distinguishes_null_endpoints() {
    let mut spec = endpoint_spec("error");
    spec["connections"][0]["records"][1]["target"] = json!(2);
    let mut graph = DirGraph::new();

    let error = from_records(&mut graph, &spec).unwrap_err();

    assert_eq!(
        error,
        "from_records: connections[0].records[2]: source endpoint id field 'source' is null"
    );
    assert_eq!(graph.graph.node_count(), 0);
}

#[test]
fn default_policy_still_vivifies_missing_non_null_endpoints() {
    let mut spec = endpoint_spec("vivify");
    spec.as_object_mut().unwrap().remove("on_missing_endpoint");
    let mut graph = DirGraph::new();

    let report = from_records(&mut graph, &spec).unwrap();

    assert_eq!(report.edges_added, 2);
    assert_eq!(report.edges_dropped_missing_endpoint, 0);
    assert_eq!(graph.graph.node_count(), 3);
}
