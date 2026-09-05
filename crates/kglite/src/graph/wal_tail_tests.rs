//! Continuing a recovered WAL must never append behind a discarded frame.
use super::*;

fn node_frame(lsn: u64) -> WalFrame {
    WalFrame {
        lsn,
        ops: vec![MutationOp::UpsertNode {
            node_type: "Item".into(),
            id: Value::Int64(lsn as i64),
            title: Value::String(format!("item-{lsn}")),
            properties: Vec::new(),
        }],
    }
}

fn prefix(version: u8) -> Vec<u8> {
    let mut bytes = WAL_MAGIC.to_vec();
    bytes.push(version);
    append_frame(&mut bytes, &node_frame(1)).unwrap();
    bytes
}

fn encoded_frame(lsn: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    append_frame(&mut bytes, &node_frame(lsn)).unwrap();
    bytes
}

fn assert_continuation(tail: &[u8], version: u8, sync: SyncMode) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.kgl-wal");
    let intact = prefix(version);
    let mut bytes = intact.clone();
    bytes.extend_from_slice(tail);
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(recover(&path).unwrap(), vec![node_frame(1)]);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        bytes,
        "recovery is read-only"
    );

    let mut wal = Wal::open(path.clone(), sync).unwrap();
    let repaired = std::fs::read(&path).unwrap();
    assert_eq!(
        repaired.len(),
        intact.len(),
        "truncate to the last verified frame boundary"
    );
    assert_eq!(&repaired[5..], &intact[5..], "keep every prefix byte");
    assert_eq!(repaired[4], WAL_FORMAT_VERSION);
    wal.append(&node_frame(2)).unwrap();
    drop(wal);
    assert_eq!(recover(&path).unwrap(), vec![node_frame(1), node_frame(2)]);
    let mut wal = Wal::open(path.clone(), sync).unwrap();
    wal.append(&node_frame(3)).unwrap();
    drop(wal);
    assert_eq!(
        recover(&path).unwrap(),
        vec![node_frame(1), node_frame(2), node_frame(3)]
    );
}

#[test]
fn resumed_appends_follow_partial_frame_repair_at_both_sync_modes() {
    let frame = encoded_frame(2);
    for sync in [SyncMode::PageCache, SyncMode::Barrier] {
        for cut in [1, 3, 4, 7, 8, frame.len() - 1] {
            assert_continuation(&frame[..cut], WAL_FORMAT_VERSION, sync);
        }
    }
}

#[test]
fn resumed_appends_follow_checksum_tail_repair_and_legacy_upgrade() {
    let mut frame = encoded_frame(2);
    frame[8] ^= 0xff;
    for version in [MIN_READABLE_WAL_FORMAT_VERSION, WAL_FORMAT_VERSION] {
        for sync in [SyncMode::PageCache, SyncMode::Barrier] {
            assert_continuation(&frame, version, sync);
        }
    }
}

#[test]
fn non_tail_damage_refuses_append_without_touching_file() {
    for later in [encoded_frame(3), vec![1, 2, 3]] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.kgl-wal");
        let mut bytes = prefix(MIN_READABLE_WAL_FORMAT_VERSION);
        let mut bad = encoded_frame(2);
        bad[8] ^= 0xff;
        bytes.extend_from_slice(&bad);
        bytes.extend_from_slice(&later);
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(recover(&path).unwrap(), vec![node_frame(1)]);
        let error = Wal::open(path.clone(), SyncMode::PageCache).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("refusing to append"), "{error}");
        assert_eq!(
            std::fs::read(path).unwrap(),
            bytes,
            "no truncation or legacy header upgrade"
        );
    }
}

#[test]
fn unsupported_version_refuses_before_tail_repair() {
    for version in [1, WAL_FORMAT_VERSION + 1] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.kgl-wal");
        let mut bytes = prefix(version);
        bytes.extend_from_slice(&[1, 2]);
        std::fs::write(&path, &bytes).unwrap();
        assert!(Wal::open(path.clone(), SyncMode::Barrier).is_err());
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }
}

#[test]
fn intact_log_is_not_rewritten_on_append_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.kgl-wal");
    let bytes = prefix(WAL_FORMAT_VERSION);
    std::fs::write(&path, &bytes).unwrap();
    let wal = Wal::open(path.clone(), SyncMode::Barrier).unwrap();
    assert_eq!(std::fs::read(path).unwrap(), bytes);
    drop(wal);
}

#[test]
fn replaced_same_length_file_refuses_recovered_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.kgl-wal");
    let mut bytes = prefix(WAL_FORMAT_VERSION);
    bytes.extend_from_slice(&[1, 2]);
    std::fs::write(&path, &bytes).unwrap();
    let recovered = recover_for_append(&path).unwrap();
    let old = dir.path().join("retained-old-wal");
    std::fs::rename(&path, &old).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let error = Wal::open_recovered(path.clone(), SyncMode::Barrier, recovered).unwrap_err();
    assert!(error.to_string().contains("identity"), "{error}");
    assert_eq!(std::fs::read(path).unwrap(), bytes);
    assert_eq!(std::fs::read(old).unwrap(), bytes);
}

#[test]
fn tail_truncation_error_is_not_reported_as_repaired() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.kgl-wal");
    let intact = prefix(WAL_FORMAT_VERSION);
    let mut bytes = intact.clone();
    bytes.extend_from_slice(&[1, 2]);
    std::fs::write(&path, &bytes).unwrap();
    let read_only = File::open(&path).unwrap();
    let point = ResumePoint {
        version: WAL_FORMAT_VERSION,
        stream_len: bytes.len() as u64,
        valid_bytes: intact.len() as u64,
    };
    assert!(repair_tail(&read_only, point).is_err());
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}

#[test]
fn shared_durable_open_repairs_before_its_next_frame() {
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::durability::open_log;
    use crate::graph::storage::GraphRead;
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.kgl");
    let sidecar = wal_path(&path);
    let mut bytes = prefix(WAL_FORMAT_VERSION);
    bytes.extend_from_slice(&[1, 2]);
    std::fs::write(&sidecar, bytes).unwrap();
    let mut graph = Arc::new(DirGraph::new());
    let (mut wal, next) = open_log(&mut graph, &path, DurabilityLevel::Normal)
        .unwrap()
        .unwrap();
    assert_eq!(next, 2);
    assert_eq!(graph.graph.node_count(), 1);
    assert!(graph.graph.is_wal_owner());
    wal.append(&node_frame(next)).unwrap();
    drop(wal);
    assert_eq!(
        recover(&sidecar).unwrap(),
        vec![node_frame(1), node_frame(2)]
    );
}

#[test]
fn newly_appeared_file_refuses_empty_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.kgl-wal");
    let recovered = recover_for_append(&path).unwrap();
    let bytes = prefix(WAL_FORMAT_VERSION);
    std::fs::write(&path, &bytes).unwrap();
    assert!(Wal::open_recovered(path.clone(), SyncMode::Barrier, recovered).is_err());
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}
