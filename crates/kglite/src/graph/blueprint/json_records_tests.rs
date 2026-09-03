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

#[test]
fn unknown_top_level_key_is_refused_and_names_the_accepted_set() {
    let mut graph = DirGraph::new();

    let error = from_records(
        &mut graph,
        &json!({
            "nodes": [{"type": "Doc", "id_field": "id", "records": [{"id": 1}]}],
            "relationships": [{
                "type": "LINKS",
                "source_type": "Doc",
                "source_id_field": "source",
                "target_type": "Doc",
                "target_id_field": "target",
                "records": [{"source": 1, "target": 1}]
            }]
        }),
    )
    .unwrap_err();

    assert_eq!(
        error,
        "from_records: unknown key 'relationships'. Accepted keys: 'nodes', 'connections', 'on_missing_endpoint'."
    );
    assert_eq!(graph.graph.node_count(), 0);
}

#[test]
fn unknown_node_spec_key_suggests_the_near_miss() {
    let mut graph = DirGraph::new();

    let error = from_records(
        &mut graph,
        &json!({"nodes": [{"type": "Doc", "id_feild": "id", "records": [{"id": 1}]}]}),
    )
    .unwrap_err();

    assert_eq!(
        error,
        "from_records: nodes[0]: unknown key 'id_feild'. Did you mean 'id_field'?"
    );
    assert_eq!(graph.graph.node_count(), 0);
}

#[test]
fn unknown_connection_spec_key_suggests_the_near_miss() {
    let mut graph = DirGraph::new();

    let error = from_records(
        &mut graph,
        &json!({
            "nodes": [{"type": "Doc", "id_field": "id", "records": [{"id": 1}, {"id": 2}]}],
            "connections": [{
                "type": "LINKS",
                "source_typ": "Doc",
                "source_id_field": "source",
                "target_type": "Doc",
                "target_id_field": "target",
                "records": [{"source": 1, "target": 2}]
            }]
        }),
    )
    .unwrap_err();

    assert_eq!(
        error,
        "from_records: connections[0]: unknown key 'source_typ'. Did you mean 'source_type'?"
    );
    assert_eq!(graph.graph.edge_count(), 0);
}

#[test]
fn every_accepted_key_still_builds() {
    let mut graph = DirGraph::new();

    let report = from_records(
        &mut graph,
        &json!({
            "on_missing_endpoint": "error",
            "nodes": [{
                "type": "Doc",
                "id_field": "id",
                "title_field": "name",
                "labels": ["Text"],
                "conflict_handling": "update",
                "records": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
            }],
            "connections": [{
                "type": "LINKS",
                "source_type": "Doc",
                "source_id_field": "source",
                "target_type": "Doc",
                "target_id_field": "target",
                "records": [{"source": 1, "target": 2}]
            }]
        }),
    )
    .unwrap();

    assert_eq!(report.nodes_added, 2);
    assert_eq!(report.edges_added, 1);
}

#[test]
fn labels_survive_a_spec_whose_records_are_empty() {
    // The nodes of a type can all arrive as vivified edge endpoints. The spec
    // still declares that type's labels, so an empty `records` list must not
    // discard them — that is the silent-directive-drop this key exists to end.
    let mut graph = DirGraph::new();

    from_records(
        &mut graph,
        &json!({
            "nodes": [
                {"type": "Doc", "id_field": "id", "labels": ["Text"], "records": []},
                {"type": "Src", "id_field": "id", "records": [{"id": 1}]}
            ],
            "connections": [{
                "type": "CITES",
                "source_type": "Src",
                "source_id_field": "s",
                "target_type": "Doc",
                "target_id_field": "t",
                "records": [{"s": 1, "t": 7}]
            }]
        }),
    )
    .unwrap();

    let doc = graph
        .lookup_by_id_readonly("Doc", &Value::Int64(7))
        .unwrap();
    let labels: Vec<String> = graph
        .node_labels(doc)
        .into_iter()
        .map(|k| graph.interner.resolve(k).to_string())
        .collect();
    assert!(
        labels.iter().any(|l| l == "Text"),
        "vivified Doc(7) carries {labels:?}, not the spec's declared 'Text'"
    );
}
