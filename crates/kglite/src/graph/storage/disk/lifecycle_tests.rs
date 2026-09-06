//! Ownership controls retain private files without retaining publication rights.
use super::{generation::GraphDirectoryLock, graph::DiskGraph};
use crate::graph::{dir_graph::DirGraph, storage::backend::GraphBackend};
use std::sync::Arc;

#[test]
fn ended_reader_lineage_detaches_before_next_write() {
    let root = tempfile::tempdir().unwrap();
    let mut parent = DiskGraph::new_at_path(root.path()).unwrap();
    parent.prepare_mutation().unwrap();
    let old_workspace = parent.active_write_dir().to_path_buf();
    std::fs::write(old_workspace.join("late-value"), "retained").unwrap();
    let mut child = parent.clone();
    child.adopt_writer_lineage(&parent);
    parent.end_writer_authority();
    let new_owner = GraphDirectoryLock::try_acquire(root.path()).unwrap();
    child.prepare_mutation().unwrap();
    assert_ne!(child.logical_root, root.path());
    assert_ne!(child.active_write_dir(), old_workspace);
    drop(parent);
    assert_eq!(
        std::fs::read_to_string(old_workspace.join("late-value")).unwrap(),
        "retained"
    );
    drop(child);
    assert!(!old_workspace.exists());
    drop(new_owner);
}

#[test]
fn recording_disk_detachment_preserves_unread_workspace_files() {
    // Public CDC-on-disk remains refused. Internal wrapper composition must still
    // obey the generic clone/detach helper's resource lifetime contract.
    let root = tempfile::tempdir().unwrap();
    let mut disk = DiskGraph::new_at_path(root.path()).unwrap();
    disk.prepare_mutation().unwrap();
    let old_workspace = disk.active_write_dir().to_path_buf();
    std::fs::write(old_workspace.join("late-value"), "retained").unwrap();
    let mut graph = DirGraph::new();
    graph.graph = GraphBackend::Disk(Box::new(disk));
    graph.graph.wrap_for_capture();
    let detached = graph.detached_persistence_snapshot();
    assert_ne!(detached.graph.as_disk().unwrap().logical_root, root.path());
    graph.end_persistence_authority();
    drop(graph);
    let owner = GraphDirectoryLock::try_acquire(root.path()).unwrap();
    assert_eq!(
        std::fs::read_to_string(old_workspace.join("late-value")).unwrap(),
        "retained"
    );
    drop(detached);
    assert!(!old_workspace.exists());
    drop(owner);
}

#[test]
fn publication_permit_finishes_before_revoke_releases_os_lock() {
    let root = tempfile::tempdir().unwrap();
    let lease = Arc::new(GraphDirectoryLock::try_acquire(root.path()).unwrap());
    let permit = lease.publication_permit().unwrap();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (ended_tx, ended_rx) = std::sync::mpsc::channel();
    let ending = Arc::clone(&lease);
    let thread = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        ending.end();
        ended_tx.send(()).unwrap();
    });
    started_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    assert!(ended_rx
        .recv_timeout(std::time::Duration::from_millis(20))
        .is_err());
    assert!(GraphDirectoryLock::try_acquire(root.path()).is_err());
    drop(permit);
    ended_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    thread.join().unwrap();
    assert!(!lease.is_active());
    assert!(lease.publication_permit().is_err());
    assert!(GraphDirectoryLock::try_acquire(root.path()).is_ok());
}

#[test]
fn direct_compact_after_revocation_uses_a_private_workspace() {
    use crate::datatypes::Value;
    use crate::graph::schema::{EdgeData, NodeData, StringInterner};
    use crate::graph::storage::GraphRead;
    use std::collections::HashMap;
    let root = tempfile::tempdir().unwrap();
    let mut parent = DiskGraph::new_at_path(root.path()).unwrap();
    parent.prepare_mutation().unwrap();
    let mut interner = StringInterner::new();
    let a = parent.add_node(NodeData::new(
        Value::Int64(1),
        Value::Null,
        "N".into(),
        HashMap::new(),
        &mut interner,
    ));
    let b = parent.add_node(NodeData::new(
        Value::Int64(2),
        Value::Null,
        "N".into(),
        HashMap::new(),
        &mut interner,
    ));
    parent.add_edge(
        a,
        b,
        EdgeData::new("R".into(), HashMap::new(), &mut interner),
    );
    let old_workspace = parent.active_write_dir().to_path_buf();
    let mut child = parent.clone();
    child.adopt_writer_lineage(&parent);
    parent.end_writer_authority();
    let _new_owner = GraphDirectoryLock::try_acquire(root.path()).unwrap();
    assert_eq!(child.compact().unwrap(), 1);
    assert_ne!(child.logical_root, root.path());
    assert_ne!(child.active_write_dir(), old_workspace);
    assert_eq!(parent.edge_count, 1);
    assert_eq!(
        child.edge_endpoints(petgraph::graph::EdgeIndex::new(0)),
        Some((a, b))
    );
}
