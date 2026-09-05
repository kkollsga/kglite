use super::*;
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::wal_replay::apply_frames;
use crate::graph::schema::{EdgeData, InternedKey};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::recording::{resolve_ops, wrap_for_durability, RawOp};
use crate::graph::storage::{GraphRead, GraphWrite};
use std::collections::HashMap;
use std::sync::Arc;

fn run(graph: &mut DirGraph, query: &str) -> Result<(), String> {
    let params = HashMap::new();
    execute_mut(graph, query, &ExecuteOptions::eager(&params))
        .map(|_| ())
        .map_err(|error| error.to_string())
}
fn seed() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Item {id: 1}), (:Item {id: 2}), (:Item {id: 3})",
    )
    .unwrap();
    graph
}
fn node(graph: &mut DirGraph, id: i64) -> petgraph::graph::NodeIndex {
    graph.lookup_by_id("Item", &Value::Int64(id)).unwrap()
}
fn add(graph: &mut DirGraph, src: i64, tgt: i64, n: i64) -> petgraph::graph::EdgeIndex {
    let a = node(graph, src);
    let b = node(graph, tgt);
    let kind = graph.interner.get_or_intern("LINK");
    let prop = graph.interner.get_or_intern("n");
    graph.graph.add_edge(
        a,
        b,
        EdgeData::new_interned(kind, vec![(prop, Value::Int64(n))]),
    )
}
fn frame(graph: &mut DirGraph, lsn: u64) -> WalFrame {
    let raw = graph.graph.recording_mut().unwrap().take_ops();
    assert!(
        raw.iter()
            .any(|op| matches!(op, RawOp::WalNode { .. } | RawOp::WalGroup { .. })),
        "actual durable capture must admit v4"
    );
    WalFrame {
        lsn,
        ops: resolve_ops(&raw, &graph.graph, &graph.interner, |idx| {
            graph.secondary_label_names(idx)
        }),
    }
}
fn edges(graph: &DirGraph) -> Vec<(i64, i64, Value)> {
    let _guard = graph.begin_read_pass();
    let mut out: Vec<_> = graph
        .graph
        .edge_indices()
        .map(|idx| {
            let (a, b) = graph.graph.edge_endpoints(idx).unwrap();
            let Value::Int64(a) = graph.graph.get_node_id(a).unwrap() else {
                panic!("source")
            };
            let Value::Int64(b) = graph.graph.get_node_id(b).unwrap() else {
                panic!("target")
            };
            let props = &graph.graph.edge_weight(idx).unwrap().properties;
            (
                a,
                b,
                props
                    .iter()
                    .find(|(k, _)| *k == InternedKey::from_str("n"))
                    .unwrap()
                    .1
                    .clone(),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    out
}

#[test]
fn v4_parallel_maps_update_delete_and_idempotent_replay() {
    let mut live = seed();
    add(&mut live, 1, 2, 10);
    add(&mut live, 1, 2, 10);
    let mut checkpoint = live.clone();
    wrap_for_durability(&mut live).unwrap();
    let added = add(&mut live, 1, 2, 20);
    add(&mut live, 1, 1, 30);
    for value in [21, 22, 23] {
        live.graph.edge_weight_mut(added).unwrap().properties[0].1 = Value::Int64(value);
    }
    let f = frame(&mut live, 1);
    assert_eq!(
        f.ops
            .iter()
            .filter(|op| matches!(op, MutationOp::ReplaceEdgeGroup { .. }))
            .count(),
        2
    );
    assert_eq!(
        edges(&live),
        vec![
            (1, 1, Value::Int64(30)),
            (1, 2, Value::Int64(10)),
            (1, 2, Value::Int64(10)),
            (1, 2, Value::Int64(23))
        ]
    );
    apply_frames(&mut checkpoint, std::slice::from_ref(&f), 0).unwrap();
    apply_frames(&mut checkpoint, &[f], 0).unwrap();
    assert_eq!(edges(&checkpoint), edges(&live));
    live.graph.remove_edge(added);
    let f = frame(&mut live, 2);
    apply_frames(&mut checkpoint, &[f], 1).unwrap();
    assert_eq!(edges(&checkpoint), edges(&live));
}

#[test]
fn v4_slot_reuse_cannot_resolve_an_old_group_as_a_new_edge() {
    let mut live = seed();
    let old = add(&mut live, 1, 2, 1);
    let mut checkpoint = live.clone();
    wrap_for_durability(&mut live).unwrap();
    live.graph.remove_edge(old);
    let reused = add(&mut live, 2, 3, 2);
    assert_eq!(reused, old, "fixture must reuse the physical edge slot");
    live.graph.remove_edge(reused);
    let second = add(&mut live, 3, 1, 3);
    assert_eq!(second, old);
    let f = frame(&mut live, 1);
    assert_eq!(
        f.ops.len(),
        3,
        "old/intermediate/final groups touched exactly once"
    );
    apply_frames(&mut checkpoint, &[f], 0).unwrap();
    assert_eq!(edges(&checkpoint), vec![(3, 1, Value::Int64(3))]);
}

#[test]
fn v4_node_recreation_clears_labels_edges_and_retains_new_parallel_group() {
    let mut live = seed();
    add(&mut live, 1, 2, 1);
    run(&mut live, "MATCH (n:Item {id: 1}) SET n:Old").unwrap();
    let mut checkpoint = live.clone();
    wrap_for_durability(&mut live).unwrap();
    run(&mut live, "MATCH (n:Item {id: 1}) DETACH DELETE n").unwrap();
    run(&mut live, "CREATE (:Item {id: 1})").unwrap();
    add(&mut live, 1, 2, 7);
    add(&mut live, 1, 2, 7);
    let f = frame(&mut live, 1);
    apply_frames(&mut checkpoint, std::slice::from_ref(&f), 0).unwrap();
    apply_frames(&mut checkpoint, &[f], 0).unwrap();
    let idx = node(&mut checkpoint, 1);
    assert!(checkpoint.secondary_label_names(idx).is_empty());
    assert_eq!(
        edges(&checkpoint),
        vec![(1, 2, Value::Int64(7)), (1, 2, Value::Int64(7))]
    );
}

#[test]
fn v4_rollback_truncates_markers_with_the_undone_statement() {
    let mut live = seed();
    let mut checkpoint = live.clone();
    wrap_for_durability(&mut live).unwrap();
    let first = add(&mut live, 1, 2, 1);
    let boundary = live.graph.recording().unwrap().ops_len();
    let rolled = add(&mut live, 2, 3, 2);
    live.graph
        .recording_mut()
        .unwrap()
        .inner_mut()
        .remove_edge(rolled);
    live.graph.recording_mut().unwrap().truncate_ops(boundary);
    assert!(live.graph.edge_weight(first).is_some());
    let f = frame(&mut live, 1);
    assert_eq!(f.ops.len(), 1);
    apply_frames(&mut checkpoint, &[f], 0).unwrap();
    assert_eq!(edges(&checkpoint), vec![(1, 2, Value::Int64(1))]);
}

#[test]
fn v4_group_matching_keeps_unchanged_legacy_invalid_member_only() {
    use crate::graph::property_types::DeclaredType;
    let mut live = seed();
    let a = add(&mut live, 1, 2, 1);
    add(&mut live, 1, 2, 2);
    live.create_rel_property_type_constraint(
        "LINK",
        "n",
        DeclaredType::Integer,
        &crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    live.graph.edge_weight_mut(a).unwrap().properties[0].1 = Value::String("legacy".into());
    let mut checkpoint = live.clone();
    wrap_for_durability(&mut live).unwrap();
    add(&mut live, 1, 2, 3);
    let f = frame(&mut live, 1);
    apply_frames(&mut checkpoint, &[f], 0).unwrap();
    assert_eq!(edges(&checkpoint), edges(&live));
    assert!(run(
        &mut live,
        "MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:LINK {n: 'invalid-new'}]->(b)"
    )
    .is_err());
    let bad = add(&mut live, 1, 2, 4);
    live.graph.edge_weight_mut(bad).unwrap().properties[0].1 = Value::String("unmatched".into());
    let f = frame(&mut live, 2);
    let before = edges(&checkpoint);
    assert!(apply_frames(&mut checkpoint, &[f], 1).is_err());
    assert_eq!(edges(&checkpoint), before);
}

#[test]
fn v4_missing_endpoint_is_not_a_legacy_stub_request() {
    let mut graph = seed();
    let before = graph.graph.node_count();
    let op = MutationOp::ReplaceEdgeGroup {
        conn_type: "LINK".into(),
        src_type: "Item".into(),
        src_id: Value::Int64(1),
        tgt_type: "Item".into(),
        tgt_id: Value::Int64(99),
        edges: vec![vec![]],
    };
    assert!(apply_frames(
        &mut graph,
        &[WalFrame {
            lsn: 1,
            ops: vec![op]
        }],
        0
    )
    .is_err());
    assert_eq!(graph.graph.node_count(), before);
}

#[test]
fn v4_adoption_refuses_duplicate_exact_ids_before_sidecar_or_cdc_change() {
    use crate::graph::durability::open_log;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("checkpoint.kgl");
    let mut graph = seed();
    run(&mut graph, "CREATE (:Item {id: 1})").unwrap();
    graph.graph.wrap_for_capture();
    let before = graph.graph.recording().unwrap().ops_len();
    let mut graph = Arc::new(graph);
    let held = Arc::clone(&graph);
    let error = open_log(&mut graph, &path, DurabilityLevel::Normal)
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate logical node identity"), "{error}");
    assert!(Arc::ptr_eq(&graph, &held));
    assert!(!graph.graph.is_wal_owner());
    assert_eq!(graph.graph.recording().unwrap().ops_len(), before);
    assert!(!wal_path(&path).exists());
    assert_eq!(
        graph.graph.node_count(),
        4,
        "non-durable duplicates remain permitted"
    );
}

#[test]
fn v4_appended_tags_and_legacy_header_upgrade_preserve_exact_payloads() {
    let node = MutationOp::ReplaceNodeState {
        node_type: "Item".into(),
        id: Value::Null,
        title: Value::Point { lat: 1.0, lon: 2.0 },
        properties: vec![],
        labels: vec![],
        reset: true,
    };
    let group = MutationOp::ReplaceEdgeGroup {
        conn_type: "LINK".into(),
        src_type: "Item".into(),
        src_id: Value::Null,
        tgt_type: "Item".into(),
        tgt_id: Value::Null,
        edges: vec![vec![], vec![]],
    };
    for (op, tag) in [(node.clone(), 5), (group.clone(), 6)] {
        let bytes = crate::serde_codec::encode_versioned(
            crate::serde_codec::CodecVersion::PostcardV1,
            &op,
            1024,
        )
        .unwrap();
        assert_eq!(bytes[0], tag, "append-only Postcard tag");
    }
    for version in [2, 3] {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("old.wal");
        let old = WalFrame {
            lsn: 1,
            ops: vec![MutationOp::UpsertNode {
                node_type: "Item".into(),
                id: Value::Null,
                title: Value::Null,
                properties: vec![],
            }],
        };
        let mut bytes = Vec::new();
        write_header_version(&mut bytes, version).unwrap();
        append_frame(&mut bytes, &old).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let mut wal = Wal::open(path.clone(), SyncMode::PageCache).unwrap();
        let new = WalFrame {
            lsn: 2,
            ops: vec![node.clone(), group.clone()],
        };
        wal.append(&new).unwrap();
        drop(wal);
        assert_eq!(std::fs::read(&path).unwrap()[4], 4);
        assert_eq!(recover(&path).unwrap(), vec![old, new]);
    }
}

#[test]
fn v4_public_wrap_admission_is_exact_and_precedes_owner_claim() {
    let mut duplicate = seed();
    run(&mut duplicate, "CREATE (:Item {id: 1})").unwrap();
    assert!(wrap_for_durability(&mut duplicate).is_err());
    assert!(!duplicate.graph.is_wal_owner());
    let mut typed = DirGraph::new();
    let ops = [
        Value::Null,
        Value::Point { lat: 1.0, lon: 2.0 },
        Value::String("1".into()),
        Value::Int64(1),
    ]
    .into_iter()
    .map(|id| MutationOp::UpsertNode {
        node_type: "Typed".into(),
        id,
        title: Value::Null,
        properties: vec![],
    })
    .collect();
    apply_frames(&mut typed, &[WalFrame { lsn: 1, ops }], 0).unwrap();
    wrap_for_durability(&mut typed).unwrap();
    assert!(typed.graph.is_wal_owner());
    assert_eq!(typed.graph.node_count(), 4);
}

#[test]
fn v4_group_snapshot_keeps_exact_nested_values_and_equal_map_count() {
    let mut live = seed();
    let mut checkpoint = live.clone();
    wrap_for_durability(&mut live).unwrap();
    let value = Value::List(vec![
        Value::Point {
            lat: 12.0,
            lon: 13.0,
        },
        Value::Duration {
            months: 1,
            days: 2,
            seconds: 3,
        },
        Value::Null,
    ]);
    for _ in 0..2 {
        let idx = add(&mut live, 1, 1, 0);
        live.graph.edge_weight_mut(idx).unwrap().properties[0].1 = value.clone();
    }
    let f = frame(&mut live, 1);
    let [MutationOp::ReplaceEdgeGroup { edges: members, .. }] = f.ops.as_slice() else {
        panic!("one normalized group")
    };
    assert_eq!(
        members,
        &vec![
            vec![("n".into(), value.clone())],
            vec![("n".into(), value.clone())]
        ]
    );
    apply_frames(&mut checkpoint, &[f], 0).unwrap();
    assert_eq!(
        edges(&checkpoint),
        vec![(1, 1, value.clone()), (1, 1, value)]
    );
}

#[test]
fn v4_endpoint_reset_cannot_reuse_a_legacy_invalid_edge_exception() {
    use crate::graph::property_types::DeclaredType;
    let mut graph = seed();
    let idx = add(&mut graph, 1, 2, 1);
    graph
        .create_rel_property_type_constraint(
            "LINK",
            "n",
            DeclaredType::Integer,
            &crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap();
    graph.graph.edge_weight_mut(idx).unwrap().properties[0].1 = Value::String("legacy".into());
    let before = edges(&graph);
    let ops = vec![
        MutationOp::ReplaceNodeState {
            node_type: "Item".into(),
            id: Value::Int64(1),
            title: Value::Null,
            properties: vec![],
            labels: vec![],
            reset: true,
        },
        MutationOp::ReplaceEdgeGroup {
            conn_type: "LINK".into(),
            src_type: "Item".into(),
            src_id: Value::Int64(1),
            tgt_type: "Item".into(),
            tgt_id: Value::Int64(2),
            edges: vec![vec![("n".into(), Value::String("legacy".into()))]],
        },
    ];
    assert!(apply_frames(&mut graph, &[WalFrame { lsn: 1, ops }], 0).is_err());
    assert_eq!(edges(&graph), before);
}

#[test]
fn v4_session_commit_and_transaction_drop_keep_complete_groups() {
    use crate::graph::session::{CommitOutcome, Session};
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("session.kgl");
    let path_str = path.to_string_lossy().into_owned();
    let session =
        Session::open_durable(Arc::new(seed()), &path_str, DurabilityLevel::Normal).unwrap();
    session.save(&path_str, false).unwrap();
    let held = session.snapshot();
    {
        let mut rolled = session.begin();
        add(rolled.working_mut().unwrap(), 1, 2, 99);
        add(rolled.working_mut().unwrap(), 1, 2, 99);
    }
    let mut tx = session.begin();
    add(tx.working_mut().unwrap(), 1, 2, 7);
    add(tx.working_mut().unwrap(), 1, 2, 7);
    assert!(matches!(
        session.commit(tx, true),
        CommitOutcome::Committed { .. }
    ));
    assert!(edges(&held).is_empty());
    assert_eq!(
        edges(&session.snapshot()),
        vec![(1, 2, Value::Int64(7)), (1, 2, Value::Int64(7))]
    );
    drop(session);
    let checkpoint = crate::graph::io::file::load_file(&path_str).unwrap();
    let reopened = Session::open_durable(checkpoint, &path_str, DurabilityLevel::Normal).unwrap();
    assert_eq!(
        edges(&reopened.snapshot()),
        vec![(1, 2, Value::Int64(7)), (1, 2, Value::Int64(7))]
    );
}

#[test]
fn v4_whole_group_size_failure_writes_no_partial_envelope() {
    let group = MutationOp::ReplaceEdgeGroup {
        conn_type: "LINK".into(),
        src_type: "Item".into(),
        src_id: Value::Int64(1),
        tgt_type: "Item".into(),
        tgt_id: Value::Int64(2),
        edges: vec![vec![("n".into(), Value::String("x".repeat(100)))]; 10],
    };
    let f = WalFrame {
        lsn: 1,
        ops: vec![group],
    };
    let mut written = vec![10, 20];
    let error = append_frame_bounded(
        &mut written,
        &f,
        crate::serde_codec::CodecVersion::PostcardV1,
        128,
    )
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(written, vec![10, 20]);
    let mut complete = Vec::new();
    write_header(&mut complete).unwrap();
    append_frame(&mut complete, &f).unwrap();
    assert_eq!(
        read_frames(&complete[..], complete.len() as u64).unwrap(),
        vec![f]
    );
}

fn legacy_edge(remove: bool) -> MutationOp {
    if remove {
        MutationOp::RemoveEdge {
            conn_type: "LINK".into(),
            src_type: "Item".into(),
            src_id: Value::Int64(1),
            tgt_type: "Item".into(),
            tgt_id: Value::Int64(2),
        }
    } else {
        MutationOp::UpsertEdge {
            conn_type: "LINK".into(),
            src_type: "Item".into(),
            src_id: Value::Int64(1),
            tgt_type: "Item".into(),
            tgt_id: Value::Int64(2),
            properties: vec![("n".into(), Value::Int64(9))],
        }
    }
}
fn files_in(root: &std::path::Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
    fn walk(
        root: &std::path::Path,
        at: &std::path::Path,
        out: &mut std::collections::BTreeMap<PathBuf, Vec<u8>>,
    ) {
        for entry in std::fs::read_dir(at).unwrap() {
            let path = entry.unwrap().path();
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
    let mut out = Default::default();
    walk(root, root, &mut out);
    out
}

#[test]
fn v4_ambiguous_legacy_replay_refuses_without_graph_cdc_or_file_publication() {
    use crate::graph::durability::{open_log, DurableOpenError};
    use crate::graph::io::file::save_graph;
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        for version in [2, 3] {
            for remove in [false, true] {
                let tmp = tempfile::tempdir().unwrap();
                let path = tmp.path().join("legacy.kgl");
                let mut graph = new_dir_graph_in_mode(mode, None).unwrap();
                run(&mut graph, "CREATE(:Item{id:1}),(:Item{id:2})").unwrap();
                add(&mut graph, 1, 2, 1);
                add(&mut graph, 1, 2, 2);
                let mut owner = Arc::new(graph);
                save_graph(&mut owner, path.to_str().unwrap()).unwrap();
                let mut bytes = Vec::new();
                write_header_version(&mut bytes, version).unwrap();
                append_frame(
                    &mut bytes,
                    &WalFrame {
                        lsn: 1,
                        ops: vec![legacy_edge(remove)],
                    },
                )
                .unwrap();
                std::fs::write(wal_path(&path), bytes).unwrap();
                let graph = crate::graph::handle::make_dir_graph_mut(&mut owner);
                graph.graph.wrap_for_capture();
                let idx = node(graph, 1);
                graph
                    .graph
                    .set_node_title(idx, Value::String("unpublished CDC".into()));
                let cdc = graph.graph.recording().unwrap().ops_len();
                assert!(cdc > 0);
                let before_edges = edges(graph);
                let before_version = graph.version;
                let before_files = files_in(tmp.path());
                let held = Arc::clone(&owner);
                let error = open_log(&mut owner, &path, DurabilityLevel::Normal).unwrap_err();
                assert!(
                    matches!(&error,DurableOpenError::Replay(message) if message.contains("ambiguous legacy WAL edge action")),
                    "{error}"
                );
                assert!(Arc::ptr_eq(&owner, &held));
                assert_eq!(edges(&owner), before_edges);
                assert_eq!(edges(&held), before_edges);
                assert_eq!(
                    owner.graph.get_node_title(idx),
                    Some(Value::String("unpublished CDC".into()))
                );
                assert_eq!(owner.graph.recording().unwrap().ops_len(), cdc);
                assert_eq!(owner.version, before_version);
                assert!(!owner.graph.is_wal_owner());
                assert_eq!(files_in(tmp.path()), before_files);
            }
        }
    }
}

#[test]
fn v4_legacy_single_member_remains_valid_and_final_group_disambiguates() {
    for remove in [false, true] {
        let mut graph = seed();
        add(&mut graph, 1, 2, 1);
        let legacy = WalFrame {
            lsn: 1,
            ops: vec![legacy_edge(remove)],
        };
        apply_frames(&mut graph, std::slice::from_ref(&legacy), 0).unwrap();
        assert_eq!(
            edges(&graph),
            if remove {
                vec![]
            } else {
                vec![(1, 2, Value::Int64(9))]
            }
        );
        let mut graph = seed();
        add(&mut graph, 1, 2, 1);
        add(&mut graph, 1, 2, 2);
        let complete = WalFrame {
            lsn: 2,
            ops: vec![MutationOp::ReplaceEdgeGroup {
                conn_type: "LINK".into(),
                src_type: "Item".into(),
                src_id: Value::Int64(1),
                tgt_type: "Item".into(),
                tgt_id: Value::Int64(2),
                edges: vec![vec![("n".into(), Value::Int64(7))]; 2],
            }],
        };
        apply_frames(&mut graph, &[legacy, complete], 0).unwrap();
        assert_eq!(
            edges(&graph),
            vec![(1, 2, Value::Int64(7)), (1, 2, Value::Int64(7))]
        );
    }
}
