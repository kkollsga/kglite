use super::*;
use crate::datatypes::Value;
use crate::graph::schema::{EdgeData, NodeData};
use crate::graph::storage::recording::{resolve_ops, wrap_for_durability};
use crate::graph::storage::GraphWrite;
use crate::graph::wal::WalFrame;
use std::collections::HashMap;
use std::collections::HashSet;

fn graph_with_parallel_edges() -> (DirGraph, EdgeIndex, EdgeIndex, EdgeIndex) {
    let mut graph = DirGraph::new();
    let a = GraphWrite::add_node(
        &mut graph.graph,
        NodeData::new(
            Value::Int64(1),
            Value::String("a".into()),
            "Doc".into(),
            HashMap::new(),
            &mut graph.interner,
        ),
    );
    let b = GraphWrite::add_node(
        &mut graph.graph,
        NodeData::new(
            Value::Int64(2),
            Value::String("b".into()),
            "Doc".into(),
            HashMap::new(),
            &mut graph.interner,
        ),
    );
    let r1 = GraphWrite::add_edge(
        &mut graph.graph,
        a,
        b,
        EdgeData::new("ASSERTS".into(), HashMap::new(), &mut graph.interner),
    );
    let r2 = GraphWrite::add_edge(
        &mut graph.graph,
        a,
        b,
        EdgeData::new("ASSERTS".into(), HashMap::new(), &mut graph.interner),
    );
    let other = GraphWrite::add_edge(
        &mut graph.graph,
        a,
        a,
        EdgeData::new("MENTIONS".into(), HashMap::new(), &mut graph.interner),
    );
    (graph, r1, r2, other)
}

#[test]
fn parallel_edges_keep_distinct_vectors_and_bump_once() {
    let (mut graph, r1, r2, _) = graph_with_parallel_edges();
    let before = graph.version();
    let report = upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0]), (r2, vec![0.0, 1.0])],
        Some("cosine"),
    )
    .unwrap();
    assert_eq!(report.changed, 2);
    assert_eq!(graph.version(), before + 1);
    let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.get(r1), Some(&[1.0, 0.0][..]));
    assert_eq!(store.get(r2), Some(&[0.0, 1.0][..]));
}

#[test]
fn identical_upsert_is_a_true_no_op() {
    let (mut graph, r1, _, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let before = graph.version();
    let report = upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    assert_eq!(report.changed, 0);
    assert_eq!(graph.version(), before);
}

#[test]
fn invalid_batch_is_atomic() {
    let (mut graph, r1, r2, other) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let before = graph.version();
    for bad in [
        vec![(r2, vec![0.0, 1.0]), (other, vec![1.0, 1.0])],
        vec![(r2, vec![0.0, 1.0]), (r2, vec![1.0, 1.0])],
        vec![(r2, vec![f32::NAN, 1.0])],
        vec![(r2, vec![1.0])],
    ] {
        assert!(upsert_edge_embeddings(&mut graph, "ASSERTS", "description", bad, None,).is_err());
        assert_eq!(graph.version(), before);
        let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(r1), Some(&[1.0, 0.0][..]));
        assert_eq!(store.get(r2), None);
    }
}

#[test]
fn removal_validates_every_edge_before_mutating() {
    let (mut graph, r1, r2, other) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0]), (r2, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    let before = graph.version();
    assert!(remove_edge_embeddings(&mut graph, "ASSERTS", "description", &[r1, other],).is_err());
    assert_eq!(graph.version(), before);
    let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert!(store.get(r1).is_some());
    assert!(store.get(r2).is_some());
}

#[test]
fn detach_delete_prunes_incident_edge_vectors() {
    let (mut graph, r1, r2, self_loop) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0]), (r2, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    upsert_edge_embeddings(
        &mut graph,
        "MENTIONS",
        "description",
        vec![(self_loop, vec![0.5, 0.5])],
        None,
    )
    .unwrap();
    let (source, _) = graph.graph.edge_endpoints(r1).unwrap();

    crate::graph::mutation::maintain::detach_delete_nodes(&mut graph, &HashSet::from([source]));

    assert!(graph.edge_embeddings[&edge_store_key("ASSERTS", "description")].is_empty());
    assert!(graph.edge_embeddings[&edge_store_key("MENTIONS", "description")].is_empty());
}

#[test]
fn detach_delete_prunes_edge_vectors_in_mapped_and_disk_modes() {
    use crate::graph::storage::mode::{convert_dir_graph_to_mode, StorageMode};

    for mode in [StorageMode::Mapped, StorageMode::Disk] {
        let (mut graph, r1, r2, _) = graph_with_parallel_edges();
        upsert_edge_embeddings(
            &mut graph,
            "ASSERTS",
            "description",
            vec![(r1, vec![1.0, 0.0]), (r2, vec![0.0, 1.0])],
            None,
        )
        .unwrap();
        let (source, _) = graph.graph.edge_endpoints(r1).unwrap();
        match mode {
            StorageMode::Mapped => convert_dir_graph_to_mode(&mut graph, mode).unwrap(),
            StorageMode::Disk => graph.enable_disk_mode().unwrap(),
            StorageMode::Memory => unreachable!(),
        }

        crate::graph::mutation::maintain::detach_delete_nodes(&mut graph, &HashSet::from([source]));

        assert!(graph.edge_embeddings[&edge_store_key("ASSERTS", "description")].is_empty());
    }
}

#[test]
fn vacuum_remaps_vectors_to_their_surviving_parallel_edges() {
    let (mut graph, r1, r2, other) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r2, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    remove_edge_with_embeddings(&mut graph, r1).unwrap();

    graph.vacuum();

    let asserted: Vec<_> = graph
        .graph
        .edge_indices()
        .filter(|edge| {
            graph
                .graph
                .edge_weight(*edge)
                .unwrap()
                .connection_type_str(&graph.interner)
                == "ASSERTS"
        })
        .collect();
    assert_eq!(asserted.len(), 1);
    let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.get(asserted[0]), Some(&[0.0, 1.0][..]));
    assert_eq!(
        store.get(other),
        None,
        "a different edge never inherits the vector"
    );
}

#[test]
fn reused_edge_slot_never_inherits_the_deleted_vector() {
    let (mut graph, removed, survivor, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(removed, vec![1.0, 0.0]), (survivor, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    let endpoints = graph.graph.edge_endpoints(removed).unwrap();
    remove_edge_with_embeddings(&mut graph, removed).unwrap();
    let replacement = GraphWrite::add_edge(
        &mut graph.graph,
        endpoints.0,
        endpoints.1,
        EdgeData::new("ASSERTS".into(), HashMap::new(), &mut graph.interner),
    );

    assert_eq!(replacement, removed, "the fixture must reuse the slot");
    let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.get(replacement), None);
    assert_eq!(store.get(survivor), Some(&[0.0, 1.0][..]));
}

#[test]
fn disk_compaction_remaps_vectors_after_tombstones_and_overflow() {
    let (mut graph, removed, survivor, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(survivor, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    let endpoints = graph.graph.edge_endpoints(removed).unwrap();
    graph.enable_disk_mode().unwrap();
    remove_edge_with_embeddings(&mut graph, removed).unwrap();
    GraphWrite::add_edge(
        &mut graph.graph,
        endpoints.0,
        endpoints.1,
        EdgeData::new("MENTIONS".into(), HashMap::new(), &mut graph.interner),
    );

    assert_eq!(graph.compact_disk().unwrap(), 1);

    let surviving_assertion = {
        graph
            .graph
            .edge_indices()
            .find(|edge| {
                graph
                    .graph
                    .edge_weight(*edge)
                    .is_some_and(|weight| weight.connection_type_str(&graph.interner) == "ASSERTS")
            })
            .unwrap()
    };
    let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.len(), 1);
    assert_eq!(store.get(surviving_assertion), Some(&[0.0, 1.0][..]));
}

#[test]
fn heap_to_disk_materialization_remaps_vectors_across_sparse_edge_slots() {
    let (mut graph, removed, survivor, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(survivor, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    remove_edge_with_embeddings(&mut graph, removed).unwrap();

    graph.enable_disk_mode().unwrap();

    let surviving_assertion = {
        graph
            .graph
            .edge_indices()
            .find(|edge| {
                graph
                    .graph
                    .edge_weight(*edge)
                    .is_some_and(|weight| weight.connection_type_str(&graph.interner) == "ASSERTS")
            })
            .unwrap()
    };
    assert_eq!(surviving_assertion, EdgeIndex::new(0));
    let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.get(surviving_assertion), Some(&[0.0, 1.0][..]));
    assert_eq!(store.len(), 1);
}

#[test]
fn rollback_restores_the_exact_edge_vector_slot() {
    let (mut graph, r1, r2, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0]), (r2, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    let checkpoint = crate::graph::dir_graph::rollback::StatementCheckpoint::open(&mut graph);
    remove_edge_with_embeddings(&mut graph, r1).unwrap();
    checkpoint.rollback(&mut graph);

    let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.edges().collect::<Vec<_>>(), vec![r1, r2]);
    assert_eq!(store.get(r1), Some(&[1.0, 0.0][..]));
    assert_eq!(store.get(r2), Some(&[0.0, 1.0][..]));
}

#[test]
fn cloned_graphs_own_independent_edge_stores() {
    let (mut graph, r1, _, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let mut child = graph.clone();

    assert_eq!(
        remove_edge_embeddings(&mut child, "ASSERTS", "description", &[r1]).unwrap(),
        1
    );
    assert!(child.edge_embeddings[&edge_store_key("ASSERTS", "description")].is_empty());
    assert_eq!(
        graph.edge_embeddings[&edge_store_key("ASSERTS", "description")].get(r1),
        Some(&[1.0, 0.0][..])
    );
}

#[test]
fn held_frozen_view_keeps_its_edge_vectors_after_writer_mutation() {
    let (mut writer, r1, _, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut writer,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let frozen = writer.clone();

    remove_edge_with_embeddings(&mut writer, r1).unwrap();

    assert!(writer.edge_embeddings[&edge_store_key("ASSERTS", "description")].is_empty());
    assert_eq!(
        frozen.edge_embeddings[&edge_store_key("ASSERTS", "description")].get(r1),
        Some(&[1.0, 0.0][..])
    );
}

#[test]
fn durable_private_writes_capture_group_touches() {
    let (mut graph, r1, _, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    graph.graph.wrap_for_durability();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    assert_eq!(
        remove_edge_embeddings(&mut graph, "ASSERTS", "description", &[r1]).unwrap(),
        1
    );
    let raw = graph.graph.recording_mut().unwrap().take_ops();
    assert!(raw
        .iter()
        .any(|op| matches!(op, crate::graph::storage::recording::RawOp::WalGroup { .. })));
    assert!(graph.edge_embeddings[&edge_store_key("ASSERTS", "description")].is_empty());
}

#[test]
fn durable_group_snapshot_replays_parallel_vectors_properties_and_provenance() {
    let (mut writer, r1, r2, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut writer,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0]), (r2, vec![0.0, 1.0])],
        Some("cosine"),
    )
    .unwrap();
    let store = writer
        .edge_embeddings
        .get_mut(&edge_store_key("ASSERTS", "description"))
        .unwrap();
    store.numeric.model_id = Some("model-a".into());
    store.numeric.set_text_hash(r1.index(), 11);
    store.numeric.set_text_hash(r2.index(), 22);
    let mut checkpoint = writer.clone();

    wrap_for_durability(&mut writer).unwrap();
    let marker = writer.interner.get_or_intern("revision");
    writer.graph.recording_mut().unwrap().note_wal_group(r1);
    writer
        .graph
        .edge_weight_mut(r1)
        .unwrap()
        .properties
        .push((marker, Value::Int64(7)));
    upsert_edge_embeddings(
        &mut writer,
        "ASSERTS",
        "description",
        vec![(r2, vec![0.5, 0.5])],
        Some("cosine"),
    )
    .unwrap();
    let raw = writer.graph.recording_mut().unwrap().take_ops();
    let ops = resolve_ops(&raw, &writer);
    assert!(ops.iter().any(|op| matches!(
        op,
        crate::graph::wal::MutationOp::SetEdgeEmbeddingStore { .. }
    )));
    assert!(ops.iter().any(|op| matches!(
        op,
        crate::graph::wal::MutationOp::PatchEdgeGroupEmbeddings { .. }
    )));

    crate::graph::mutation::wal_replay::apply_frames(
        &mut checkpoint,
        &[WalFrame { lsn: 1, ops }],
        0,
    )
    .unwrap();
    let store = &checkpoint.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(
        store.model_id(),
        None,
        "manual upsert clears aggregate model provenance"
    );
    let mut members: Vec<_> = checkpoint
        .graph
        .edge_indices()
        .filter(|&edge| {
            checkpoint
                .graph
                .edge_weight(edge)
                .is_some_and(|weight| weight.connection_type_str(&checkpoint.interner) == "ASSERTS")
        })
        .collect();
    members.sort_unstable_by_key(|edge| edge.index());
    assert_eq!(store.get(members[0]), Some(&[1.0, 0.0][..]));
    assert_eq!(store.text_hash(members[0]), Some(11));
    assert_eq!(store.get(members[1]), Some(&[0.5, 0.5][..]));
    assert_eq!(store.text_hash(members[1]), None);
    assert!(checkpoint
        .graph
        .edge_weight(members[0])
        .unwrap()
        .properties
        .contains(&(marker, Value::Int64(7))));
}

#[test]
fn replay_replaces_store_dimension_across_all_groups_atomically() {
    use crate::graph::wal::{
        EdgeEmbeddingStoreState, EdgeGroupStoreWalState, EdgeVectorWalState, MutationOp,
    };

    let (mut graph, r1, _, _) = graph_with_parallel_edges();
    let (source, _) = graph.graph.edge_endpoints(r1).unwrap();
    let self_edge = GraphWrite::add_edge(
        &mut graph.graph,
        source,
        source,
        EdgeData::new("ASSERTS".into(), HashMap::new(), &mut graph.interner),
    );
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0; 3]), (self_edge, vec![2.0; 3])],
        None,
    )
    .unwrap();

    let group = |src_id, tgt_id, members: Vec<Option<EdgeVectorWalState>>| {
        MutationOp::ReplaceEdgeGroupEmbeddings {
            conn_type: "ASSERTS".into(),
            src_type: "Doc".into(),
            src_id: Value::Int64(src_id),
            tgt_type: "Doc".into(),
            tgt_id: Value::Int64(tgt_id),
            member_count: members.len(),
            stores: vec![EdgeGroupStoreWalState {
                text_column: "description".into(),
                members,
            }],
        }
    };
    let ops = vec![
        MutationOp::SetEdgeEmbeddingStore {
            conn_type: "ASSERTS".into(),
            text_column: "description".into(),
            state: EdgeEmbeddingStoreState::Present {
                dimension: 4,
                metric: None,
                model_id: Some("model-b".into()),
            },
        },
        MutationOp::ReplaceEdgeGroup {
            conn_type: "ASSERTS".into(),
            src_type: "Doc".into(),
            src_id: Value::Int64(1),
            tgt_type: "Doc".into(),
            tgt_id: Value::Int64(2),
            edges: vec![vec![], vec![]],
        },
        group(
            1,
            2,
            vec![
                Some(EdgeVectorWalState {
                    vector: vec![3.0; 4],
                    text_hash: None,
                }),
                None,
            ],
        ),
        MutationOp::ReplaceEdgeGroup {
            conn_type: "ASSERTS".into(),
            src_type: "Doc".into(),
            src_id: Value::Int64(1),
            tgt_type: "Doc".into(),
            tgt_id: Value::Int64(1),
            edges: vec![vec![]],
        },
        group(
            1,
            1,
            vec![Some(EdgeVectorWalState {
                vector: vec![4.0; 4],
                text_hash: None,
            })],
        ),
    ];
    crate::graph::mutation::wal_replay::apply_frames(&mut graph, &[WalFrame { lsn: 1, ops }], 0)
        .unwrap();

    let store = &graph.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.dimension(), 4);
    assert_eq!(store.model_id(), Some("model-b"));
    assert_eq!(store.len(), 2);
    assert_eq!(store.get(r1), Some(&[3.0; 4][..]));
    assert_eq!(store.get(self_edge), Some(&[4.0; 4][..]));
}

#[test]
fn durable_session_recovers_committed_edge_vectors_and_discards_failed_transaction() {
    let (mut graph, r1, _, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edge-wal.kgl");
    let path_str = path.to_str().unwrap();
    let mut graph = std::sync::Arc::new(graph);
    crate::graph::io::file::save_graph(&mut graph, path_str).unwrap();
    let loaded = crate::graph::io::file::load_file(path_str).unwrap();
    let session = crate::graph::session::Session::open_durable(
        loaded,
        path_str,
        crate::graph::wal::DurabilityLevel::Full,
    )
    .unwrap();
    let mut committed = session.begin();
    upsert_edge_embeddings(
        committed.working_mut().unwrap(),
        "ASSERTS",
        "description",
        vec![(r1, vec![0.5, 0.5])],
        None,
    )
    .unwrap();
    assert!(matches!(
        session.commit(committed, true),
        crate::graph::session::CommitOutcome::Committed { .. }
    ));
    let mut rolled_back = session.begin();
    {
        let working = rolled_back.working_mut().unwrap();
        upsert_edge_embeddings(
            working,
            "ASSERTS",
            "description",
            vec![(r1, vec![0.0, 1.0])],
            None,
        )
        .unwrap();
    }
    session.rollback(rolled_back);
    assert_eq!(
        session.snapshot().edge_embeddings[&edge_store_key("ASSERTS", "description")].get(r1),
        Some(&[0.5, 0.5][..])
    );
    let frames = crate::graph::wal::recover(&crate::graph::wal::wal_path(&path)).unwrap();
    assert!(
        frames
            .iter()
            .flat_map(|frame| &frame.ops)
            .any(|op| matches!(
                op,
                crate::graph::wal::MutationOp::PatchEdgeGroupEmbeddings { patch, .. }
                    if patch.members.iter().any(|member| matches!(
                        member,
                        crate::graph::wal::EdgeGroupMemberPatchWal::Prior { cells, .. }
                            if cells.iter().any(|cell| matches!(
                                cell,
                                crate::graph::wal::EdgeVectorCellPatchWal::Replace(member)
                                    if member.vector == [0.5, 0.5]
                            ))
                    ))
            )),
        "{frames:#?}"
    );
    drop(session);

    let recovered = crate::graph::session::Session::open_durable(
        crate::graph::io::file::load_file(path_str).unwrap(),
        path_str,
        crate::graph::wal::DurabilityLevel::Full,
    )
    .unwrap();
    let snapshot = recovered.snapshot();
    assert_eq!(
        snapshot.edge_embeddings[&edge_store_key("ASSERTS", "description")].get(r1),
        Some(&[0.5, 0.5][..])
    );
}

#[test]
fn malformed_or_identity_incomplete_edge_wal_refuses_atomically() {
    let (mut graph, r1, _, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let before = graph.clone();
    let malformed = WalFrame {
        lsn: 1,
        ops: vec![crate::graph::wal::MutationOp::SetEdgeEmbeddingStore {
            conn_type: "ASSERTS".into(),
            text_column: String::new(),
            state: crate::graph::wal::EdgeEmbeddingStoreState::Absent,
        }],
    };
    assert!(
        crate::graph::mutation::wal_replay::apply_frames(&mut graph, &[malformed], 0)
            .unwrap_err()
            .contains("text column")
    );
    assert_eq!(graph.version(), before.version());
    assert_eq!(
        graph.edge_embeddings[&edge_store_key("ASSERTS", "description")].get(r1),
        before.edge_embeddings[&edge_store_key("ASSERTS", "description")].get(r1)
    );

    let incomplete = WalFrame {
        lsn: 2,
        ops: vec![crate::graph::wal::MutationOp::ReplaceEdgeGroup {
            conn_type: "ASSERTS".into(),
            src_type: "Doc".into(),
            src_id: Value::Int64(1),
            tgt_type: "Doc".into(),
            tgt_id: Value::Int64(2),
            edges: Vec::new(),
        }],
    };
    assert!(
        crate::graph::mutation::wal_replay::apply_frames(&mut graph, &[incomplete], 0)
            .unwrap_err()
            .contains("cannot infer vector identity")
    );
    assert_eq!(
        graph.edge_embeddings[&edge_store_key("ASSERTS", "description")].get(r1),
        Some(&[1.0, 0.0][..])
    );
}

#[test]
#[ignore = "release-only WAL payload measurement"]
fn measure_full_state_edge_wal_matrix() {
    use crate::graph::wal::{
        append_frame, EdgeEmbeddingStoreState, EdgeGroupStoreWalState, EdgeVectorWalState,
        MutationOp,
    };
    use std::hint::black_box;
    use std::time::Instant;

    fn encoded(frame: &WalFrame) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_frame(&mut bytes, frame).unwrap();
        bytes
    }
    fn min_encode_ns(frame: &WalFrame) -> u128 {
        (0..100)
            .map(|_| {
                let start = Instant::now();
                black_box(encoded(black_box(frame)));
                start.elapsed().as_nanos()
            })
            .min()
            .unwrap()
    }

    println!("members,dimension,control_bytes,property_full_bytes,one_change_bytes,amplification,control_min_ns,property_min_ns,one_change_min_ns");
    for members in [1usize, 10, 100] {
        for dimension in [384usize, 1536] {
            let properties: Vec<_> = (0..members)
                .map(|member| vec![("flag".into(), Value::Int64(member as i64))])
                .collect();
            let topology = MutationOp::ReplaceEdgeGroup {
                conn_type: "R".into(),
                src_type: "N".into(),
                src_id: Value::Int64(1),
                tgt_type: "N".into(),
                tgt_id: Value::Int64(2),
                edges: properties,
            };
            let control = WalFrame {
                lsn: 1,
                ops: vec![topology.clone()],
            };
            let full = |changed: bool| WalFrame {
                lsn: 1,
                ops: vec![
                    MutationOp::SetEdgeEmbeddingStore {
                        conn_type: "R".into(),
                        text_column: "text".into(),
                        state: EdgeEmbeddingStoreState::Present {
                            dimension,
                            metric: Some("cosine".into()),
                            model_id: Some("model".into()),
                        },
                    },
                    topology.clone(),
                    MutationOp::ReplaceEdgeGroupEmbeddings {
                        conn_type: "R".into(),
                        src_type: "N".into(),
                        src_id: Value::Int64(1),
                        tgt_type: "N".into(),
                        tgt_id: Value::Int64(2),
                        member_count: members,
                        stores: vec![EdgeGroupStoreWalState {
                            text_column: "text".into(),
                            members: (0..members)
                                .map(|member| {
                                    let value = if changed && member == 0 { 0.5 } else { 0.25 };
                                    Some(EdgeVectorWalState {
                                        vector: vec![value; dimension],
                                        text_hash: Some(member as u64),
                                    })
                                })
                                .collect(),
                        }],
                    },
                ],
            };
            let property = full(false);
            let one_change = full(true);
            let control_bytes = encoded(&control).len();
            let property_bytes = encoded(&property).len();
            let one_change_bytes = encoded(&one_change).len();
            // Full-state frames carry every member's vector: the amplification
            // over the topology-only control is at least the raw f32 payload.
            assert!(
                property_bytes >= control_bytes + members * dimension * std::mem::size_of::<f32>(),
                "members={members} dimension={dimension}: full-state frame ({property_bytes} B) \
                 cannot be smaller than control ({control_bytes} B) plus the vectors"
            );
            assert_eq!(
                property_bytes, one_change_bytes,
                "both full-state frames carry the same member count and width"
            );
            println!(
                "{members},{dimension},{control_bytes},{property_bytes},{one_change_bytes},{:.1},{},{},{}",
                property_bytes as f64 / control_bytes as f64,
                min_encode_ns(&control),
                min_encode_ns(&property),
                min_encode_ns(&one_change),
            );
        }
    }
}

#[test]
fn extend_preserves_target_vector_and_never_copies_source_store() {
    let (mut target, target_edge, target_parallel, target_other) = graph_with_parallel_edges();
    remove_edge_with_embeddings(&mut target, target_parallel).unwrap();
    remove_edge_with_embeddings(&mut target, target_other).unwrap();
    upsert_edge_embeddings(
        &mut target,
        "ASSERTS",
        "description",
        vec![(target_edge, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let (mut source, source_edge, source_parallel, source_other) = graph_with_parallel_edges();
    remove_edge_with_embeddings(&mut source, source_parallel).unwrap();
    remove_edge_with_embeddings(&mut source, source_other).unwrap();
    upsert_edge_embeddings(
        &mut source,
        "ASSERTS",
        "description",
        vec![(source_edge, vec![0.0, 1.0])],
        None,
    )
    .unwrap();

    crate::graph::mutation::extend::extend_graph(&mut target, &source, Some("replace".to_string()))
        .unwrap();

    let store = &target.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.len(), 1);
    assert_eq!(store.get(target_edge), Some(&[1.0, 0.0][..]));
    assert_ne!(store.get(target_edge), Some(&[0.0, 1.0][..]));
}

#[test]
fn a_store_for_a_property_no_relationship_carries_is_refused_until_one_does() {
    let (mut graph, r1, _, _) = graph_with_parallel_edges();
    let error = require_carried_text_property(&graph, "ASSERTS", "description").unwrap_err();
    assert!(
        error.starts_with("Text column 'description' not found on any 'ASSERTS' relationship."),
        "{error}"
    );
    let key = graph.interner.get_or_intern("description");
    graph
        .graph
        .edge_weight_mut(r1)
        .unwrap()
        .properties
        .push((key, Value::String("a claim".into())));
    require_carried_text_property(&graph, "ASSERTS", "description").unwrap();
    // Another type carrying the property does not admit this one.
    assert!(require_carried_text_property(&graph, "MENTIONS", "description").is_err());
}

#[test]
fn an_existing_store_is_accepted_without_a_carrier() {
    let (mut graph, r1, _, _) = graph_with_parallel_edges();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(r1, vec![1.0])],
        None,
    )
    .unwrap();
    require_carried_text_property(&graph, "ASSERTS", "description").unwrap();
}

#[test]
fn a_relationship_is_described_by_type_and_endpoint_ids() {
    let (graph, r1, _, other) = graph_with_parallel_edges();
    assert_eq!(
        describe_relationship(&graph, r1),
        "(Doc id=1)-[:ASSERTS]->(Doc id=2)"
    );
    assert_eq!(
        describe_relationship(&graph, other),
        "(Doc id=1)-[:MENTIONS]->(Doc id=1)"
    );
}
