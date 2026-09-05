use super::*;
use crate::datatypes::Value;
use crate::graph::mutation::wal_replay::apply_frames;
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::wal::MutationOp;

fn frame(id: i64) -> WalFrame {
    WalFrame {
        lsn: id as u64,
        ops: vec![MutationOp::UpsertNode {
            node_type: "Item".into(),
            id: Value::Int64(id),
            title: Value::Int64(id),
            properties: vec![],
        }],
    }
}

#[test]
fn writer_open_failure_does_not_publish_recovery_or_drain_cdc() {
    let mut seed = DirGraph::new();
    apply_frames(&mut seed, &[frame(1)], 0).unwrap();
    seed.graph.wrap_for_capture();
    let idx = seed.lookup_by_id("Item", &Value::Int64(1)).unwrap();
    seed.graph
        .set_node_title(idx, Value::String("pending CDC".into()));
    let cdc_len = seed.graph.recording().unwrap().ops_len();
    assert!(cdc_len > 0);
    let mut graph = Arc::new(seed);
    let held = Arc::clone(&graph);
    let before_version = graph.version;
    let (prepared, lsn) = prepare_replay(&graph, &[frame(2)], 0).unwrap();
    assert_eq!(prepared.as_ref().unwrap().graph.node_count(), 2);
    assert_eq!(graph.graph.node_count(), 1);
    let failure = finish_recovered_open(&mut graph, prepared, lsn, || {
        Err(DurableOpenError::Io(
            "injected append-open/repair IO failure".into(),
        ))
    });
    assert!(matches!(failure, Err(DurableOpenError::Io(_))));
    assert!(Arc::ptr_eq(&graph, &held));
    assert_eq!(graph.graph.node_count(), 1);
    assert_eq!(
        graph.graph.get_node_title(idx),
        Some(Value::String("pending CDC".into()))
    );
    assert_eq!(graph.graph.recording().unwrap().ops_len(), cdc_len);
    assert_eq!(graph.version, before_version);
    assert!(!graph.graph.is_wal_owner());
}

#[test]
fn no_eligible_work_avoids_a_recovery_workspace() {
    let seed = DirGraph::new();
    let (prepared, lsn) = prepare_replay(&seed, &[frame(1)], 1).unwrap();
    assert!(prepared.is_none());
    assert_eq!(lsn, 1);
}
