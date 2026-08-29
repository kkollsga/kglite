//! `LoadOptions::storage` — the request that outranks the mode a `.kgl`
//! recorded, and the two directions that have no conversion to make.
//!
//! The deferral half of `LoadOptions` is covered by
//! `file_deferred_index_tests`; this file is the storage half.

use super::*;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::storage::mode::{convert_dir_graph_to_mode, live_storage_mode, StorageMode};
use std::collections::HashMap;

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

fn item_count(graph: &DirGraph) -> i64 {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let result = execute_read(graph, "MATCH (n:Item) RETURN count(n) AS c", &opts)
        .expect("count query failed")
        .result;
    match result.rows[0].first() {
        Some(Value::Int64(n)) => *n,
        other => panic!("unexpected count cell: {other:?}"),
    }
}

/// 20 `:Item` nodes, saved in `mode`, returned as the file's bytes.
fn fixture_bytes(mode: StorageMode) -> Vec<u8> {
    let mut graph = DirGraph::new();
    for i in 0..20u32 {
        run(
            &mut graph,
            &format!("CREATE (:Item {{id: {i}, sku: 'sku-{i}'}})"),
        );
    }
    convert_dir_graph_to_mode(&mut graph, mode).expect("fixture conversion");
    let mut arc = Arc::new(graph);
    prepare_save(&mut arc);
    Arc::make_mut(&mut arc).enable_columnar();
    let mut bytes = Vec::new();
    write_kgl_to(&arc, &mut bytes).expect("fixture write");
    bytes
}

#[test]
fn the_recorded_mode_is_honoured_when_nothing_is_requested() {
    // The control that keeps the two override tests non-vacuous: without a
    // request, each file comes back in the mode it recorded.
    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        let graph = load_kgl_bytes(&fixture_bytes(mode)).unwrap();
        assert_eq!(live_storage_mode(&graph), mode);
        assert_eq!(item_count(&graph), 20);
    }
}

#[test]
fn a_storage_request_overrides_the_recorded_mode_in_both_directions() {
    for (recorded, requested) in [
        (StorageMode::Mapped, StorageMode::Memory),
        (StorageMode::Memory, StorageMode::Mapped),
    ] {
        let bytes = fixture_bytes(recorded);
        let options = LoadOptions::new().with_storage(requested);
        let graph = load_kgl_bytes_with(&bytes, &options).unwrap();
        assert_eq!(
            live_storage_mode(&graph),
            requested,
            "a {recorded:?}-recorded file asked for {requested:?} must come back {requested:?}"
        );
        assert_eq!(item_count(&graph), 20, "the conversion must not touch rows");
    }
}

/// Requesting the mode the file already records is a no-op, not a conversion
/// error — the case a caller that always passes `storage=` hits every time.
#[test]
fn requesting_the_recorded_mode_is_accepted() {
    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        let options = LoadOptions::new().with_storage(mode);
        let graph = load_kgl_bytes_with(&fixture_bytes(mode), &options).unwrap();
        assert_eq!(live_storage_mode(&graph), mode);
    }
}

/// A disk request on a portable file is refused with the mode named and the
/// alternative given — and *before the payload is read*, which is what the
/// corrupted fixture proves: the same bytes fail with a corruption error on
/// every other path.
#[test]
fn a_disk_request_is_refused_before_the_payload_is_decoded() {
    let mut bytes = fixture_bytes(StorageMode::Memory);
    // Flip a byte deep in the section payload; the header and the metadata
    // head, which the mode resolution reads, stay intact.
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;

    let payload_failure = load_kgl_bytes(&bytes)
        .err()
        .expect("the corrupted payload must not load");
    assert!(
        !payload_failure.to_string().contains("disk"),
        "the control must fail on the payload, not on a mode: {payload_failure}"
    );

    let refusal = load_kgl_bytes_with(&bytes, &LoadOptions::new().with_storage(StorageMode::Disk))
        .err()
        .expect("a disk request on a .kgl must be refused");
    assert_eq!(refusal.kind(), std::io::ErrorKind::InvalidInput);
    let message = refusal.to_string();
    assert!(message.contains("disk"), "{message}");
    assert!(message.contains("enable_disk_mode()"), "{message}");
    assert!(message.contains("directory"), "{message}");
}

/// The other disk direction: a disk-graph *directory* has no portable backend
/// to swap to, so a `memory`/`mapped` request on one is refused with the same
/// wording rather than served in a mode the caller did not get.
#[test]
fn a_portable_request_on_a_disk_directory_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("diskgraph");
    let mut graph =
        crate::graph::storage::mode::new_dir_graph_in_mode(StorageMode::Disk, Some(path.as_path()))
            .expect("disk graph creation");
    run(&mut graph, "CREATE (:Item {id: 1, sku: 'sku-1'})");
    let mut arc = Arc::new(graph);
    save_graph(&mut arc, path.to_str().unwrap()).expect("disk save");
    drop(arc);

    // Control: the directory opens fine with no request.
    assert_eq!(
        live_storage_mode(&load_file(path.to_str().unwrap()).unwrap()),
        StorageMode::Disk
    );

    let refusal = load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new().with_storage(StorageMode::Memory),
    )
    .err()
    .expect("a memory request on a disk directory must be refused");
    assert_eq!(refusal.kind(), std::io::ErrorKind::InvalidInput);
    let message = refusal.to_string();
    assert!(message.contains("disk-mode directory"), "{message}");
    assert!(message.contains("memory"), "{message}");

    // Asking for the mode it already is stays legal.
    assert!(load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new().with_storage(StorageMode::Disk),
    )
    .is_ok());
}
