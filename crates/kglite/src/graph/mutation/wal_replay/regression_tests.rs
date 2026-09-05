use std::sync::Arc;

use super::*;
use crate::datatypes::Value;
use crate::graph::property_types::DeclaredType;
use crate::graph::schema::{InternedKey, NodeSchemaDefinition, SchemaDefinition, SchemaInstall};
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::wal::{MutationOp, WalFrame};

fn node(id: Value, title: Value, props: &[(&str, Value)]) -> MutationOp {
    MutationOp::UpsertNode {
        node_type: "Item".into(),
        id,
        title,
        properties: props
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect(),
    }
}
fn row(id: i64, value: i64) -> MutationOp {
    node(
        Value::Int64(id),
        Value::String(format!("item-{id}")),
        &[
            ("u", Value::Int64(value)),
            ("group", Value::String("x".into())),
        ],
    )
}
fn remove(id: i64) -> MutationOp {
    MutationOp::RemoveNode {
        node_type: "Item".into(),
        id: Value::Int64(id),
    }
}
fn labels(id: i64, labels: &[&str]) -> MutationOp {
    MutationOp::SetNodeLabels {
        node_type: "Item".into(),
        id: Value::Int64(id),
        labels: labels.iter().map(|label| label.to_string()).collect(),
    }
}
fn edge(source: i64, target: i64) -> MutationOp {
    MutationOp::UpsertEdge {
        conn_type: "LINK".into(),
        src_type: "Item".into(),
        src_id: Value::Int64(source),
        tgt_type: "Item".into(),
        tgt_id: Value::Int64(target),
        properties: vec![],
    }
}
fn apply(graph: &mut DirGraph, ops: Vec<MutationOp>) -> Result<u64, String> {
    apply_frames(graph, &[WalFrame { lsn: 1, ops }], 0)
}
fn index(graph: &mut DirGraph, id: i64) -> petgraph::graph::NodeIndex {
    graph.lookup_by_id("Item", &Value::Int64(id)).unwrap()
}
fn value(graph: &mut DirGraph, id: i64, property: &str) -> Option<Value> {
    let idx = index(graph, id);
    graph
        .graph
        .get_node_property(idx, InternedKey::from_str(property))
}
type StoredNodeState = (usize, Value, Value, Vec<(String, Value)>, Vec<String>);

fn stored(graph: &DirGraph) -> Vec<StoredNodeState> {
    let _guard = graph.begin_read_pass();
    graph
        .graph
        .node_indices()
        .map(|idx| {
            let view = graph.graph.node_view(idx).unwrap();
            let mut props: Vec<_> = view
                .properties_cloned(&graph.interner)
                .into_iter()
                .collect();
            props.sort_by(|left, right| left.0.cmp(&right.0));
            (
                idx.index(),
                view.id().into_owned(),
                view.title().into_owned(),
                props,
                graph.secondary_label_names(idx),
            )
        })
        .collect()
}

#[test]
fn exact_typed_identity_title_properties_and_null_roundtrip_on_memory_and_mapped() {
    let values = vec![
        Value::Null,
        Value::Point {
            lat: 59.9,
            lon: 10.7,
        },
        Value::Duration {
            months: 1,
            days: 2,
            seconds: 3,
        },
        Value::Int64(7),
        Value::String("seven".into()),
        Value::List(vec![Value::Int64(1), Value::String("x".into())]),
        Value::Map([("x", Value::Int64(8))].into_iter().collect()),
    ];
    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        let mut graph = new_dir_graph_in_mode(mode, None).unwrap();
        let ops: Vec<_> = values
            .iter()
            .map(|value| node(value.clone(), value.clone(), &[("exact", value.clone())]))
            .collect();
        apply(&mut graph, ops.clone()).unwrap();
        apply(&mut graph, ops).unwrap();
        assert_eq!(graph.graph.node_count(), values.len());
        for expected in &values {
            let idx = graph
                .lookup_by_id("Item", expected)
                .expect("exact identity");
            assert_eq!(graph.graph.get_node_id(idx).as_ref(), Some(expected));
            assert_eq!(graph.graph.get_node_title(idx).as_ref(), Some(expected));
            let actual = graph
                .graph
                .get_node_property(idx, InternedKey::from_str("exact"))
                .unwrap_or(Value::Null);
            assert_eq!(&actual, expected);
        }
    }
}

#[test]
fn point_participates_in_required_type_and_unique_validation_before_publish() {
    let mut graph = DirGraph::new();
    graph.create_not_null_constraint("Item", "loc").unwrap();
    graph
        .create_property_type_constraint("Item", "loc", DeclaredType::Point)
        .unwrap();
    graph.create_unique_constraint("Item", &["loc"]).unwrap();
    let point = Value::Point { lat: 1.0, lon: 2.0 };
    let one = node(Value::Int64(1), Value::Null, &[("loc", point.clone())]);
    apply(&mut graph, vec![one.clone()]).unwrap();
    apply(&mut graph, vec![one]).unwrap();
    let before = stored(&graph);
    assert!(apply(
        &mut graph,
        vec![node(Value::Int64(2), Value::Null, &[("loc", point)])]
    )
    .is_err());
    assert_eq!(stored(&graph), before);
    assert_eq!(
        graph.list_unique_constraints(),
        vec![("Item".into(), vec!["loc".into()])]
    );
}

#[test]
fn simple_composite_and_node_key_cycles_validate_final_occupancy() {
    for kind in 0..3 {
        let mut graph = DirGraph::new();
        apply(&mut graph, vec![row(1, 10), row(2, 20), row(3, 30)]).unwrap();
        match kind {
            0 => {
                graph.create_unique_constraint("Item", &["u"]).unwrap();
            }
            1 => {
                graph
                    .create_unique_constraint("Item", &["u", "group"])
                    .unwrap();
            }
            _ => {
                let mut schema = SchemaDefinition::default();
                schema.node_schemas.insert(
                    "Item".into(),
                    NodeSchemaDefinition {
                        primary_key: Some("u".into()),
                        ..Default::default()
                    },
                );
                graph.set_schema(schema, SchemaInstall::Merge).unwrap();
            }
        }
        apply(&mut graph, vec![row(1, 20), row(2, 30), row(3, 10)]).unwrap();
        assert_eq!(value(&mut graph, 1, "u"), Some(Value::Int64(20)));
        assert_eq!(value(&mut graph, 2, "u"), Some(Value::Int64(30)));
        assert_eq!(value(&mut graph, 3, "u"), Some(Value::Int64(10)));
        assert!(graph.verify_unique_constraints().is_empty());
        apply(&mut graph, vec![remove(2), row(4, 30)]).unwrap();
        assert_eq!(value(&mut graph, 4, "u"), Some(Value::Int64(30)));
        let before = stored(&graph);
        assert!(apply(&mut graph, vec![row(1, 30)]).is_err());
        assert_eq!(stored(&graph), before);
    }
}

#[test]
fn unchanged_legacy_duplicate_occupants_survive_but_new_claimants_do_not() {
    let mut graph = DirGraph::new();
    apply(&mut graph, vec![row(1, 10), row(2, 20), row(3, 30)]).unwrap();
    graph.create_unique_constraint("Item", &["u"]).unwrap();
    let second = index(&mut graph, 2);
    graph
        .graph
        .set_node_property(second, InternedKey::from_str("u"), Value::Int64(10));
    graph.reindex();
    assert_eq!(
        graph.verify_unique_constraints().len(),
        1,
        "fixture has a legacy violation"
    );
    apply(&mut graph, vec![row(1, 10), row(2, 10)]).unwrap();
    assert_eq!(graph.verify_unique_constraints().len(), 1);
    let before = stored(&graph);
    assert!(apply(&mut graph, vec![row(3, 10)]).is_err());
    assert_eq!(stored(&graph), before);
    assert!(
        apply(&mut graph, vec![remove(2), row(2, 10)]).is_err(),
        "new incarnation cannot inherit a legacy exception"
    );
    assert_eq!(stored(&graph), before);
    apply(&mut graph, vec![row(2, 20)]).unwrap();
    assert!(graph.verify_unique_constraints().is_empty());
}

#[test]
fn legacy_missing_and_wrong_typed_values_are_preserved_individually() {
    let mut graph = DirGraph::new();
    apply(&mut graph, vec![row(1, 10), row(2, 20)]).unwrap();
    graph.create_not_null_constraint("Item", "u").unwrap();
    graph.create_not_null_constraint("Item", "group").unwrap();
    graph
        .create_property_type_constraint("Item", "u", DeclaredType::Integer)
        .unwrap();
    let first = index(&mut graph, 1);
    let second = index(&mut graph, 2);
    graph
        .graph
        .set_node_property(first, InternedKey::from_str("u"), Value::Null);
    graph.graph.set_node_property(
        second,
        InternedKey::from_str("u"),
        Value::String("old-invalid".into()),
    );
    let missing = node(
        Value::Int64(1),
        Value::Null,
        &[("group", Value::String("x".into()))],
    );
    let invalid = node(
        Value::Int64(2),
        Value::Null,
        &[
            ("u", Value::String("old-invalid".into())),
            ("group", Value::String("x".into())),
        ],
    );
    apply(&mut graph, vec![missing, invalid]).unwrap();
    let before = stored(&graph);
    assert!(
        apply(&mut graph, vec![node(Value::Int64(1), Value::Null, &[])]).is_err(),
        "a second missing field cannot hide behind the first"
    );
    assert_eq!(stored(&graph), before);
    assert!(apply(
        &mut graph,
        vec![node(
            Value::Int64(2),
            Value::Null,
            &[
                ("u", Value::String("new-invalid".into())),
                ("group", Value::String("x".into()))
            ]
        )]
    )
    .is_err());
    assert_eq!(stored(&graph), before);
}

#[test]
fn reincarnation_removes_old_labels_and_incident_edges_across_frame_boundaries() {
    for separate_frames in [false, true] {
        let mut graph = DirGraph::new();
        apply(
            &mut graph,
            vec![
                row(1, 10),
                row(2, 20),
                labels(1, &["Old"]),
                edge(1, 2),
                edge(1, 1),
            ],
        )
        .unwrap();
        let ops = vec![edge(1, 2), labels(1, &["Stale"]), remove(1), row(1, 30)];
        let frames = if separate_frames {
            ops.into_iter()
                .enumerate()
                .map(|(i, op)| WalFrame {
                    lsn: i as u64 + 1,
                    ops: vec![op],
                })
                .collect()
        } else {
            vec![WalFrame { lsn: 1, ops }]
        };
        apply_frames(&mut graph, &frames, 0).unwrap();
        apply_frames(&mut graph, &frames, 0).unwrap();
        let idx = index(&mut graph, 1);
        assert!(graph.secondary_label_names(idx).is_empty());
        assert_eq!(graph.graph.edge_count(), 0);
        apply(
            &mut graph,
            vec![
                remove(1),
                row(1, 40),
                labels(1, &["New"]),
                edge(1, 2),
                edge(1, 1),
            ],
        )
        .unwrap();
        let idx = index(&mut graph, 1);
        assert_eq!(graph.secondary_label_names(idx), vec!["New"]);
        assert_eq!(graph.graph.edge_count(), 2);
    }
}

#[test]
fn known_deleted_endpoint_is_not_vivified_by_legacy_edge_replay() {
    let mut graph = DirGraph::new();
    apply(&mut graph, vec![row(1, 10), edge(1, 2), remove(2)]).unwrap();
    assert_eq!(graph.graph.node_count(), 1);
    assert_eq!(graph.graph.edge_count(), 0);
    assert!(graph.lookup_by_id("Item", &Value::Int64(2)).is_none());
}

#[test]
fn empty_constrained_type_keeps_its_declaration_and_enforcement() {
    let mut graph = DirGraph::new();
    apply(&mut graph, vec![row(1, 10)]).unwrap();
    graph.create_unique_constraint("Item", &["u"]).unwrap();
    apply(&mut graph, vec![remove(1)]).unwrap();
    assert!(graph.has_unique_constraint("Item", &["u".into()]));
    assert!(apply(&mut graph, vec![row(2, 20), row(3, 20)]).is_err());
    assert_eq!(graph.graph.node_count(), 0);
}

#[test]
fn failed_replay_keeps_cdc_snapshot_metadata_version_and_aliases_unchanged() {
    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        let mut graph = new_dir_graph_in_mode(mode, None).unwrap();
        apply(&mut graph, vec![row(1, 10), row(2, 20)]).unwrap();
        graph.create_unique_constraint("Item", &["u"]).unwrap();
        graph
            .id_field_aliases_mut()
            .insert("Item".into(), "external_id".into());
        graph.graph.wrap_for_capture();
        let idx = index(&mut graph, 1);
        graph
            .graph
            .set_node_title(idx, Value::String("pending CDC".into()));
        let cdc_len = graph.graph.recording_mut().unwrap().ops_len();
        assert!(cdc_len > 0);
        let snapshot = graph.clone();
        let before = stored(&graph);
        let version = graph.version;
        let metadata = graph.node_type_metadata.clone();
        assert!(apply(&mut graph, vec![row(1, 20), row(3, 30)]).is_err());
        assert_eq!(stored(&graph), before);
        assert_eq!(stored(&snapshot), before);
        assert_eq!(graph.version, version);
        assert_eq!(graph.node_type_metadata, metadata);
        assert_eq!(graph.resolve_alias("Item", "external_id"), "id");
        assert_eq!(graph.graph.recording_mut().unwrap().ops_len(), cdc_len);
    }
}

fn files_under(root: &std::path::Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    fn walk(
        root: &std::path::Path,
        path: &std::path::Path,
        out: &mut std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
    ) {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(path).unwrap(),
                );
            }
        }
    }
    let mut files = Default::default();
    walk(root, root, &mut files);
    files
}

#[test]
fn failed_replay_preserves_saved_mapped_and_disk_bytes_and_held_readers() {
    use crate::graph::io::file::{load_file, save_graph};
    for disk in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(if disk { "disk" } else { "mapped.kgl" });
        let mut graph = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
        apply(&mut graph, vec![row(1, 10), row(2, 20)]).unwrap();
        graph.create_unique_constraint("Item", &["u"]).unwrap();
        if disk {
            graph.enable_disk_mode().unwrap();
        }
        let mut graph = Arc::new(graph);
        save_graph(&mut graph, path.to_str().unwrap()).unwrap();
        drop(graph);
        let original = load_file(path.to_str().unwrap()).unwrap();
        let before = stored(&original);
        let files = files_under(dir.path());
        let mut working = (*original).clone();
        let outcome = apply(&mut working, vec![row(1, 20), row(3, 30)]);
        assert!(
            outcome.is_err(),
            "disk={disk}, constraints={:?}, before={before:?}, after={:?}, result={outcome:?}",
            original.list_unique_constraints(),
            stored(&working)
        );
        assert_eq!(stored(&working), before);
        assert_eq!(stored(&original), before);
        drop(working);
        assert_eq!(
            files_under(dir.path()),
            files,
            "failed recovery must not rewrite or leave a workspace beside the saved graph"
        );
        assert_eq!(stored(&load_file(path.to_str().unwrap()).unwrap()), before);
    }
}

#[test]
fn exact_typed_legacy_stub_endpoint_keeps_default_title_and_provisional_marker() {
    let mut graph = DirGraph::new();
    let id = Value::Point { lat: 3.0, lon: 4.0 };
    let op = MutationOp::UpsertEdge {
        conn_type: "LINK".into(),
        src_type: "Item".into(),
        src_id: id.clone(),
        tgt_type: "Item".into(),
        tgt_id: Value::Int64(2),
        properties: vec![],
    };
    apply(&mut graph, vec![op]).unwrap();
    assert_eq!(graph.graph.node_count(), 2);
    let idx = graph.lookup_by_id("Item", &id).unwrap();
    assert_eq!(graph.graph.get_node_title(idx), Some(id));
    assert_eq!(
        graph
            .graph
            .get_node_property(idx, InternedKey::from_str("_provisional")),
        Some(Value::Boolean(true))
    );
}

#[test]
fn float_first_identity_and_existing_float_property_do_not_coerce_integers() {
    let mut graph = DirGraph::new();
    let values = [Value::Float64(7.0), Value::Int64(7), Value::UniqueId(7)];
    let ops: Vec<_> = values
        .iter()
        .map(|id| node(id.clone(), id.clone(), &[("n", id.clone())]))
        .collect();
    apply(&mut graph, ops.clone()).unwrap();
    apply(&mut graph, ops).unwrap();
    let actual: Vec<_> = graph
        .graph
        .node_indices()
        .map(|idx| graph.graph.get_node_id(idx).unwrap())
        .collect();
    assert_eq!(actual, values);
    for idx in graph.graph.node_indices() {
        let id = graph.graph.get_node_id(idx).unwrap();
        assert_eq!(graph.graph.get_node_title(idx), Some(id.clone()));
        assert_eq!(
            graph
                .graph
                .get_node_property(idx, InternedKey::from_str("n")),
            Some(id)
        );
    }
    let mut graph = DirGraph::new();
    apply(
        &mut graph,
        vec![
            node(Value::Int64(1), Value::Null, &[("n", Value::Float64(1.25))]),
            node(Value::Int64(2), Value::Null, &[("n", Value::Float64(2.5))]),
        ],
    )
    .unwrap();
    apply(
        &mut graph,
        vec![node(
            Value::Int64(1),
            Value::Int64(9),
            &[("n", Value::Int64(9))],
        )],
    )
    .unwrap();
    assert_eq!(value(&mut graph, 1, "n"), Some(Value::Int64(9)));
    assert_eq!(value(&mut graph, 2, "n"), Some(Value::Float64(2.5)));
}

#[test]
fn relationship_legacy_invalid_value_is_preserved_but_new_value_or_incarnation_refused() {
    let mut graph = DirGraph::new();
    apply(&mut graph, vec![row(1, 10), row(2, 20)]).unwrap();
    graph
        .create_rel_property_type_constraint(
            "LINK",
            "n",
            DeclaredType::Integer,
            &crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap();
    let edge_with = |value| MutationOp::UpsertEdge {
        conn_type: "LINK".into(),
        src_type: "Item".into(),
        src_id: Value::Int64(1),
        tgt_type: "Item".into(),
        tgt_id: Value::Int64(2),
        properties: vec![("n".into(), value)],
    };
    apply(&mut graph, vec![edge_with(Value::Int64(1))]).unwrap();
    let idx = graph.graph.edge_indices().next().unwrap();
    graph.graph.edge_weight_mut(idx).unwrap().properties =
        vec![(InternedKey::from_str("n"), Value::String("legacy".into()))];
    graph.graph.flush_pending_writes();
    apply(&mut graph, vec![edge_with(Value::String("legacy".into()))]).unwrap();
    assert!(apply(
        &mut graph,
        vec![edge_with(Value::String("different".into()))]
    )
    .is_err());
    let remove = MutationOp::RemoveEdge {
        conn_type: "LINK".into(),
        src_type: "Item".into(),
        src_id: Value::Int64(1),
        tgt_type: "Item".into(),
        tgt_id: Value::Int64(2),
    };
    assert!(apply(
        &mut graph,
        vec![remove, edge_with(Value::String("legacy".into()))]
    )
    .is_err());
    assert_eq!(graph.graph.edge_count(), 1);
    assert_eq!(
        graph.graph.edge_weight(idx).unwrap().properties[0].1,
        Value::String("legacy".into())
    );
}

#[test]
fn new_mixed_numeric_property_on_existing_type_starts_exact() {
    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        let mut graph = new_dir_graph_in_mode(mode, None).unwrap();
        apply(&mut graph, vec![row(1, 10), row(2, 20)]).unwrap();
        let integer = Value::Int64(9_007_199_254_740_993);
        apply(
            &mut graph,
            vec![
                node(
                    Value::Int64(1),
                    Value::Null,
                    &[("fresh", Value::Float64(1.5))],
                ),
                node(Value::Int64(2), Value::Null, &[("fresh", integer.clone())]),
            ],
        )
        .unwrap();
        assert_eq!(value(&mut graph, 1, "fresh"), Some(Value::Float64(1.5)));
        assert_eq!(value(&mut graph, 2, "fresh"), Some(integer));
    }
}

#[test]
fn unique_id_values_into_existing_integer_or_float_columns_keep_the_variant() {
    for old in [Value::Int64(1), Value::Float64(1.5)] {
        let mut graph = DirGraph::new();
        apply(
            &mut graph,
            vec![
                node(Value::Int64(1), Value::Null, &[("n", old.clone())]),
                node(Value::Int64(2), Value::Null, &[("n", old.clone())]),
            ],
        )
        .unwrap();
        apply(
            &mut graph,
            vec![node(
                Value::Int64(1),
                Value::Null,
                &[("n", Value::UniqueId(7))],
            )],
        )
        .unwrap();
        assert_eq!(value(&mut graph, 1, "n"), Some(Value::UniqueId(7)));
        assert_eq!(value(&mut graph, 2, "n"), Some(old));
    }
}

#[test]
fn saved_disk_unique_declaration_is_lazy_visible_and_enforced_on_write() {
    use crate::graph::io::file::{load_file, save_graph};
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("disk");
    let mut graph = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
    apply(&mut graph, vec![row(1, 10), row(2, 20)]).unwrap();
    graph.declare_ddl_unique_constraint("Item", &["u"]).unwrap();
    graph.enable_disk_mode().unwrap();
    let mut graph = Arc::new(graph);
    save_graph(&mut graph, path.to_str().unwrap()).unwrap();
    drop(graph);
    let files = files_under(dir.path());
    let mut loaded = load_file(path.to_str().unwrap()).unwrap();
    let held = Arc::clone(&loaded);
    let before = stored(&held);
    assert_eq!(
        files_under(dir.path()),
        files,
        "read-only load changed files"
    );
    let params = Default::default();
    assert!(loaded
        .lookup_by_id_readonly("Item", &Value::Int64(2))
        .is_some());
    assert!(
        loaded.indexes_deferred(),
        "identity lookup materialized heap indexes"
    );
    let graph = crate::graph::handle::make_dir_graph_mut(&mut loaded);
    let result = execute_mut(
        graph,
        "CREATE (:Item {id: 3, title: 'third', u: 20, group: 'x'})",
        &ExecuteOptions::eager(&params),
    );
    assert!(
        result.is_err(),
        "a saved disk UNIQUE declaration admitted a duplicate"
    );
    assert_eq!(stored(graph), before);
    assert_eq!(
        held.list_unique_constraints(),
        vec![("Item".into(), vec!["u".into()])]
    );
    assert!(held.indexes_deferred());
    assert!(
        held.unique_indices.is_empty(),
        "disk load eagerly rebuilt occupancy"
    );
    assert_eq!(stored(&held), before);
    drop(loaded);
    drop(held);
    assert_eq!(files_under(dir.path()), files);
}
