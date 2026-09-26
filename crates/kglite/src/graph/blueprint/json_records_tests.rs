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

/// Every endpoint policy judges declared relationship constraints alike:
/// `vivify` loads through `add_connections`, `drop` and `error` through
/// `add_edges_from_specs`, and both refuse a record the constraint forbids.
#[test]
fn every_endpoint_policy_enforces_relationship_constraints() {
    let interrupt = crate::graph::algorithms::Interrupt::default();
    for policy in ["vivify", "drop", "error"] {
        let mut spec = endpoint_spec(policy);
        // Keep only the record whose endpoints both exist.
        spec["connections"][0]["records"] = json!([{"source": 1, "target": 2, "weight": 3}]);

        let mut required = DirGraph::new();
        required
            .create_rel_not_null_constraint("LINKS", "since", &interrupt)
            .unwrap();
        let error = from_records(&mut required, &spec)
            .expect_err("a LINKS without `since` violates NOT NULL");
        assert!(error.contains("LINKS.since"), "policy={policy}: {error}");
        assert_eq!(required.graph.edge_count(), 0, "policy={policy}");

        let mut typed = DirGraph::new();
        typed
            .create_rel_property_type_constraint(
                "LINKS",
                "weight",
                crate::graph::property_types::DeclaredType::String,
                &interrupt,
            )
            .unwrap();
        let error =
            from_records(&mut typed, &spec).expect_err("an INTEGER weight violates IS :: STRING");
        assert!(error.contains("LINKS.weight"), "policy={policy}: {error}");
        assert_eq!(typed.graph.edge_count(), 0, "policy={policy}");

        spec["connections"][0]["records"][0]["since"] = json!(2020);
        let mut legal = DirGraph::new();
        legal
            .create_rel_not_null_constraint("LINKS", "since", &interrupt)
            .unwrap();
        let report = from_records(&mut legal, &spec).unwrap();
        assert_eq!(report.edges_added, 1, "policy={policy}");
        assert_eq!(legal.graph.edge_count(), 1, "policy={policy}");
    }
}

/// A refusal is scoped as the CHANGELOG states it: under `vivify` and `drop`
/// each connection spec is gated and written in turn, so a violation in the
/// second leaves the nodes and the first spec's relationships; `error` loads
/// on a transaction fork and writes nothing.
#[test]
fn a_constraint_refuses_from_its_connection_spec_unless_the_policy_is_error() {
    let interrupt = crate::graph::algorithms::Interrupt::default();
    for (policy, edges_left) in [("vivify", 1), ("drop", 1), ("error", 0)] {
        let mut spec = endpoint_spec(policy);
        spec["connections"][0]["records"] = json!([{"source": 1, "target": 2, "weight": 3}]);
        let mut second = spec["connections"][0].clone();
        second["type"] = json!("CITES");
        spec["connections"].as_array_mut().unwrap().push(second);

        let mut graph = DirGraph::new();
        graph
            .create_rel_not_null_constraint("CITES", "since", &interrupt)
            .unwrap();
        let error = from_records(&mut graph, &spec).expect_err("a CITES without `since`");
        assert!(
            error.contains("connections[1]") && error.contains("CITES.since"),
            "policy={policy}: {error}"
        );
        assert_eq!(graph.graph.edge_count(), edges_left, "policy={policy}");
        let nodes_left = if policy == "error" { 0 } else { 2 };
        assert_eq!(graph.graph.node_count(), nodes_left, "policy={policy}");
    }
}
