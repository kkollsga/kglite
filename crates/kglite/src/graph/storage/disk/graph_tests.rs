use super::enumerate_segment_dirs;
use super::segment_subdir;
use crate::datatypes::{DataFrame, Value};
use crate::graph::schema::{EdgeData, InternedKey, NodeData, StringInterner};
use crate::graph::storage::backend::GraphBackend;
use crate::graph::storage::disk::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints};
use crate::graph::storage::disk::graph_persist::{concat_segment_csrs, SegmentCsr};
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::DirGraph;
use petgraph::graph::{EdgeIndex, NodeIndex};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tempfile::TempDir;

#[test]
fn pending_edge_failure_is_atomic_and_retryable() {
    use crate::graph::storage::mapped::mmap_vec::{fail_next, FailurePoint};

    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut graph = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    graph.defer_csr = true;
    *graph.pending_edges.get_mut() = MmapOrVec::new();
    let mut edge = || EdgeData::new("LINKS".to_string(), HashMap::new(), &mut interner);

    fail_next(FailurePoint::HeapReserve);
    assert!(graph
        .try_add_pending_edge(NodeIndex::new(0), NodeIndex::new(1), edge())
        .is_err());
    assert_eq!(graph.pending_edges.get_mut().len(), 0);
    assert_eq!(graph.edge_count, 0);
    assert_eq!(graph.next_edge_idx, 0);

    let edge_idx = graph
        .try_add_pending_edge(NodeIndex::new(0), NodeIndex::new(1), edge())
        .unwrap();
    assert_eq!(edge_idx.index(), 0);
    assert_eq!(graph.pending_edges.get_mut().len(), 1);
    assert_eq!(graph.edge_count, 1);
    assert_eq!(graph.next_edge_idx, 1);
}

fn add_docs(graph: &mut DirGraph, ids: &[i64]) {
    let rows = ids
        .iter()
        .map(|id| vec![Value::Int64(*id), Value::String(format!("doc-{id}"))])
        .collect();
    let frame =
        DataFrame::from_cypher_rows(vec!["id".to_string(), "title".to_string()], rows).unwrap();
    crate::graph::mutation::maintain::add_nodes(
        graph,
        frame,
        "Doc".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
}

fn one_doc_frame(id: i64) -> DataFrame {
    DataFrame::from_cypher_rows(
        vec!["id".to_string(), "title".to_string()],
        vec![vec![Value::Int64(id), Value::String(format!("doc-{id}"))]],
    )
    .unwrap()
}

fn snapshot_files(root: &std::path::Path) -> BTreeMap<String, Vec<u8>> {
    fn collect(root: &std::path::Path, dir: &std::path::Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().display().to_string(),
                    std::fs::read(path).unwrap(),
                );
            }
        }
    }

    let mut files = BTreeMap::new();
    collect(root, root, &mut files);
    files
}

fn edge_score(graph: &DirGraph) -> Option<Value> {
    let key = InternedKey::from_str("score");
    // Arena-materializing reads must run under a DiskQueryGuard (the
    // debug assert in materialize_edge enforces the protocol).
    let _guard = graph.graph.begin_query();
    graph
        .graph
        .edge_weight(EdgeIndex::new(0))?
        .properties
        .iter()
        .find_map(|(candidate, value)| (*candidate == key).then(|| value.clone()))
}

#[test]
fn legacy_flat_csr_directory_remains_readable() {
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut graph = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    graph.defer_csr = true;
    let n0 = graph.add_node(seal_test_node(&mut interner, 0, "Doc"));
    let n1 = graph.add_node(seal_test_node(&mut interner, 1, "Doc"));
    graph.add_edge(n0, n1, seal_test_edge(&mut interner, "LINKS"));
    graph.build_csr_from_pending().unwrap();
    graph.save_to_dir(tmp.path(), &interner).unwrap();
    drop(graph);

    // Recreate the pre-segmentation layout: CSR and auxiliary files at the
    // graph root, with the additive layout-version field set to zero.
    let segment = tmp.path().join("seg_000");
    for entry in std::fs::read_dir(&segment).unwrap() {
        let entry = entry.unwrap();
        std::fs::rename(entry.path(), tmp.path().join(entry.file_name())).unwrap();
    }
    std::fs::remove_dir(&segment).unwrap();
    let metadata_path = tmp.path().join("disk_graph_meta.json");
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
    metadata["csr_layout_version"] = serde_json::Value::from(0);
    std::fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    let _ = std::fs::remove_file(tmp.path().join("seg_manifest.json"));

    let mut loaded_interner = StringInterner::new();
    let (loaded, _cache) =
        super::DiskGraph::load_from_dir(tmp.path(), &mut loaded_interner).unwrap();
    assert_eq!(loaded.node_count, 2);
    assert_eq!(loaded.edge_count, 1);
    let start = loaded.out_offsets.get(0) as usize;
    let end = loaded.out_offsets.get(1) as usize;
    assert_eq!(end - start, 1);
    assert_eq!(loaded.out_edges.get(start).peer, 1);
}

#[test]
fn transaction_clone_keeps_published_arrays_mapped() {
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut graph = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    graph.add_node(NodeData::new(
        Value::Int64(1),
        Value::String("doc-1".into()),
        "Doc".into(),
        HashMap::new(),
        &mut interner,
    ));
    graph.save_to_dir(tmp.path(), &interner).unwrap();

    let mut loaded_interner = StringInterner::new();
    let (loaded, _guard) =
        super::DiskGraph::load_from_dir(tmp.path(), &mut loaded_interner).unwrap();
    assert!(loaded.node_slots.is_mapped());
    let fork = loaded.clone();
    assert!(fork.node_slots.is_mapped());
    assert_eq!(fork.out_offsets.is_mapped(), loaded.out_offsets.is_mapped());
    assert_eq!(fork.out_edges.is_mapped(), loaded.out_edges.is_mapped());
    assert_eq!(fork.node_slots.heap_bytes(), 0);
    assert_eq!(fork.out_edges.heap_bytes(), loaded.out_edges.heap_bytes());
}

#[test]
fn transaction_fork_inherits_lease_but_uses_private_workspace() {
    let tmp = TempDir::new().unwrap();
    let mut parent = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    parent.prepare_mutation().unwrap();
    let parent_dir = parent.active_write_dir().to_path_buf();

    let mut child = parent.clone();
    child.adopt_writer_lineage(&parent);
    child.prepare_mutation().unwrap();

    assert!(child.writer_lock.is_some());
    assert_ne!(parent_dir, child.active_write_dir());
    assert!(parent_dir.exists());
    assert!(child.active_write_dir().exists());
}

#[test]
fn independent_copy_uses_lazy_private_root_and_rebases_on_save() {
    let source = TempDir::new().unwrap();
    let destination = TempDir::new().unwrap();
    let source_path = source.path().to_str().unwrap();
    let destination_path = destination.path().to_str().unwrap();

    let mut writer = DirGraph::new();
    add_docs(&mut writer, &[1, 2]);
    writer.enable_disk_mode().unwrap();
    writer.save_disk(source_path).unwrap();
    let frozen_source = snapshot_files(source.path());

    let mut copy = writer.independent_copy();
    let private_root = match &copy.graph {
        GraphBackend::Disk(disk) => disk.independent_root_path().unwrap().to_path_buf(),
        _ => panic!("expected disk backend"),
    };
    assert!(!private_root.exists(), "copy roots must be lazy");

    add_docs(&mut copy, &[3]);
    assert!(private_root.exists());
    assert_eq!(writer.graph.node_count(), 2);
    assert_eq!(copy.graph.node_count(), 3);
    assert_eq!(
        snapshot_files(source.path()),
        frozen_source,
        "copy mutations must not write into the selected source generation"
    );

    copy.save_disk(destination_path).unwrap();
    assert!(
        !private_root.exists(),
        "save-as must clean the scratch root"
    );
    match &copy.graph {
        GraphBackend::Disk(disk) => assert!(disk.independent_root_path().is_none()),
        _ => panic!("expected disk backend"),
    }

    let source_reader = crate::graph::io::file::load_file(source_path).unwrap();
    let copy_reader = crate::graph::io::file::load_file(destination_path).unwrap();
    assert_eq!(source_reader.graph.node_count(), 2);
    assert_eq!(copy_reader.graph.node_count(), 3);
    assert!(source_reader
        .lookup_by_id_readonly("Doc", &Value::Int64(3))
        .is_none());
    assert!(copy_reader
        .lookup_by_id_readonly("Doc", &Value::Int64(3))
        .is_some());

    // The source retains its own writer lease and can continue independently.
    add_docs(&mut writer, &[4]);
    writer.save_disk(source_path).unwrap();
    let newest_source = crate::graph::io::file::load_file(source_path).unwrap();
    let held_copy = crate::graph::io::file::load_file(destination_path).unwrap();
    assert!(newest_source
        .lookup_by_id_readonly("Doc", &Value::Int64(4))
        .is_some());
    assert!(held_copy
        .lookup_by_id_readonly("Doc", &Value::Int64(4))
        .is_none());
}

#[test]
fn dropping_unsaved_independent_copy_cleans_private_root() {
    let source = TempDir::new().unwrap();
    let mut writer = DirGraph::new();
    add_docs(&mut writer, &[1]);
    writer.enable_disk_mode().unwrap();
    writer.save_disk(source.path().to_str().unwrap()).unwrap();

    let mut copy = writer.independent_copy();
    let private_root = match &copy.graph {
        GraphBackend::Disk(disk) => disk.independent_root_path().unwrap().to_path_buf(),
        _ => panic!("expected disk backend"),
    };
    add_docs(&mut copy, &[2]);
    assert!(private_root.exists());
    drop(copy);
    assert!(!private_root.exists());
}

#[test]
fn independent_copy_retains_unsaved_parent_index_files() {
    let source = TempDir::new().unwrap();
    let destination = TempDir::new().unwrap();
    let mut writer = DirGraph::new();
    let frame = DataFrame::from_cypher_rows(
        vec!["id".into(), "title".into(), "tag".into()],
        vec![
            vec![
                Value::Int64(1),
                Value::String("one".into()),
                Value::String("a".into()),
            ],
            vec![
                Value::Int64(2),
                Value::String("two".into()),
                Value::String("b".into()),
            ],
        ],
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut writer,
        frame,
        "Doc".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    writer.enable_disk_mode().unwrap();
    writer.sync_disk_column_stores();
    writer.save_disk(source.path().to_str().unwrap()).unwrap();
    match &mut writer.graph {
        GraphBackend::Disk(disk) => {
            assert_eq!(disk.build_property_index("Doc", "tag").unwrap(), 2);
        }
        _ => panic!("expected disk backend"),
    }

    let mut copy = writer.independent_copy();
    drop(writer);
    copy.save_disk(destination.path().to_str().unwrap())
        .unwrap();
    let reloaded = crate::graph::io::file::load_file(destination.path().to_str().unwrap()).unwrap();
    match &reloaded.graph {
        GraphBackend::Disk(disk) => assert!(disk.has_property_index("Doc", "tag")),
        _ => panic!("expected disk backend"),
    }
}

#[test]
fn arc_copy_on_write_retains_disk_writer_lineage() {
    let source = TempDir::new().unwrap();
    let path = source.path().to_str().unwrap();
    let mut writer = DirGraph::new();
    add_docs(&mut writer, &[1]);
    writer.enable_disk_mode().unwrap();
    writer.save_disk(path).unwrap();

    let mut active = Arc::new(writer);
    let held_snapshot = Arc::clone(&active);
    let active_graph = crate::graph::handle::make_dir_graph_mut(&mut active);
    add_docs(active_graph, &[2]);
    assert_eq!(held_snapshot.graph.node_count(), 1);
    assert_eq!(active.graph.node_count(), 2);
    match &active.graph {
        GraphBackend::Disk(disk) => {
            assert!(disk.writer_lock.is_some());
            assert!(disk.mutation_workspace.is_some());
        }
        _ => panic!("expected disk backend"),
    }

    crate::graph::io::file::save_graph(&mut active, path).unwrap();
    assert_eq!(held_snapshot.graph.node_count(), 1);
    let reloaded = crate::graph::io::file::load_file(path).unwrap();
    assert_eq!(reloaded.graph.node_count(), 2);
}

#[test]
fn generation_publish_keeps_held_reader_on_old_snapshot() {
    let target = TempDir::new().unwrap();
    let path = target.path().to_str().unwrap();
    let mut writer = DirGraph::new();
    add_docs(&mut writer, &[1, 2]);
    writer.enable_disk_mode().unwrap();
    writer.save_disk(path).unwrap();
    let first_current = std::fs::read_to_string(target.path().join("CURRENT")).unwrap();
    let first_snapshot = crate::graph::storage::disk::generation::resolve_snapshot(target.path())
        .unwrap()
        .snapshot_dir;
    let frozen_slots = std::fs::read(first_snapshot.join("seg_000/node_slots.bin")).unwrap();
    let held_reader = crate::graph::io::file::load_file(path).unwrap();
    assert_eq!(held_reader.graph.node_count(), 2);

    add_docs(&mut writer, &[3]);
    assert_eq!(
        std::fs::read(first_snapshot.join("seg_000/node_slots.bin")).unwrap(),
        frozen_slots,
        "creating a mutation overlay must not write the selected generation"
    );
    writer.save_disk(path).unwrap();
    let second_current = std::fs::read_to_string(target.path().join("CURRENT")).unwrap();
    assert_ne!(first_current, second_current);

    let newest = crate::graph::io::file::load_file(path).unwrap();
    assert_eq!(newest.graph.node_count(), 3);
    assert_eq!(
        held_reader.graph.node_count(),
        2,
        "a reader that resolved generation 1 must remain on generation 1"
    );
}

#[test]
fn generation_preserves_edge_property_snapshots_and_rebases_writer() {
    let target = TempDir::new().unwrap();
    let path = target.path().to_str().unwrap();
    let mut writer = DirGraph::new();
    add_docs(&mut writer, &[1, 2]);
    writer.enable_disk_mode().unwrap();
    writer.prepare_disk_mutation().unwrap();
    let source = writer
        .lookup_by_id_readonly("Doc", &Value::Int64(1))
        .unwrap();
    let target_node = writer
        .lookup_by_id_readonly("Doc", &Value::Int64(2))
        .unwrap();
    let edge = EdgeData::new(
        "LINKS".to_string(),
        HashMap::from([("score".to_string(), Value::Int64(1))]),
        &mut writer.interner,
    );
    GraphWrite::add_edge(&mut writer.graph, source, target_node, edge);
    writer.save_disk(path).unwrap();

    let first_snapshot = crate::graph::storage::disk::generation::resolve_snapshot(target.path())
        .unwrap()
        .snapshot_dir;
    let frozen_tree = snapshot_files(&first_snapshot);
    let held_reader = crate::graph::io::file::load_file(path).unwrap();
    assert_eq!(edge_score(&held_reader), Some(Value::Int64(1)));

    writer.prepare_disk_mutation().unwrap();
    let edge = GraphWrite::edge_weight_mut(&mut writer.graph, EdgeIndex::new(0)).unwrap();
    edge.properties
        .iter_mut()
        .find(|(key, _)| *key == InternedKey::from_str("score"))
        .unwrap()
        .1 = Value::Int64(2);
    assert_eq!(snapshot_files(&first_snapshot), frozen_tree);

    writer.save_disk(path).unwrap();
    assert_eq!(snapshot_files(&first_snapshot), frozen_tree);
    assert_eq!(edge_score(&writer), Some(Value::Int64(2)));
    let newest = crate::graph::io::file::load_file(path).unwrap();
    assert_eq!(edge_score(&newest), Some(Value::Int64(2)));
    assert_eq!(edge_score(&held_reader), Some(Value::Int64(1)));
}

#[test]
fn generation_round_trips_node_and_edge_add_delete_overlays() {
    let target = TempDir::new().unwrap();
    let path = target.path().to_str().unwrap();
    let mut writer = DirGraph::new();
    add_docs(&mut writer, &[1, 2, 3]);
    writer.enable_disk_mode().unwrap();
    writer.prepare_disk_mutation().unwrap();
    let nodes: Vec<_> = [1, 2, 3]
        .into_iter()
        .map(|id| {
            writer
                .lookup_by_id_readonly("Doc", &Value::Int64(id))
                .unwrap()
        })
        .collect();
    for pair in nodes.windows(2) {
        let edge = EdgeData::new("LINKS".to_string(), HashMap::new(), &mut writer.interner);
        GraphWrite::add_edge(&mut writer.graph, pair[0], pair[1], edge);
    }
    writer.save_disk(path).unwrap();
    let first_snapshot = crate::graph::storage::disk::generation::resolve_snapshot(target.path())
        .unwrap()
        .snapshot_dir;
    let frozen_tree = snapshot_files(&first_snapshot);
    let held_reader = crate::graph::io::file::load_file(path).unwrap();

    writer.prepare_disk_mutation().unwrap();
    GraphWrite::remove_edge(&mut writer.graph, EdgeIndex::new(0)).unwrap();
    crate::graph::mutation::maintain::detach_delete_nodes(
        &mut writer,
        &std::collections::HashSet::from([nodes[2]]),
    );
    add_docs(&mut writer, &[4]);
    let fourth = writer
        .lookup_by_id_readonly("Doc", &Value::Int64(4))
        .unwrap();
    let edge = EdgeData::new("LINKS".to_string(), HashMap::new(), &mut writer.interner);
    GraphWrite::add_edge(&mut writer.graph, nodes[0], fourth, edge);
    assert_eq!(snapshot_files(&first_snapshot), frozen_tree);

    writer.save_disk(path).unwrap();
    assert_eq!(snapshot_files(&first_snapshot), frozen_tree);
    let newest = crate::graph::io::file::load_file(path).unwrap();
    assert_eq!(newest.graph.node_count(), 3);
    assert_eq!(newest.graph.edge_count(), 1);
    let only_edge = newest.graph.edge_references().next().unwrap();
    assert_eq!((only_edge.source(), only_edge.target()), (nodes[0], fourth));
    assert_eq!(held_reader.graph.node_count(), 3);
    assert_eq!(held_reader.graph.edge_count(), 2);
}

#[test]
fn second_disk_writer_is_rejected_before_mutation() {
    let target = TempDir::new().unwrap();
    let path = target.path().to_str().unwrap();
    let mut first = DirGraph::new();
    add_docs(&mut first, &[1]);
    first.enable_disk_mode().unwrap();
    first.save_disk(path).unwrap();

    let loaded = crate::graph::io::file::load_file(path).unwrap();
    let mut second = match std::sync::Arc::try_unwrap(loaded) {
        Ok(graph) => graph,
        Err(_) => panic!("fresh load unexpectedly had another Arc owner"),
    };
    let error = crate::graph::mutation::maintain::add_nodes(
        &mut second,
        one_doc_frame(2),
        "Doc".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .expect_err("second writer must fail before changing its overlay");
    assert!(error.contains("active writer"), "{error}");
    assert_eq!(second.graph.node_count(), 1);

    drop(first);
    crate::graph::mutation::maintain::add_nodes(
        &mut second,
        one_doc_frame(2),
        "Doc".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    assert_eq!(second.graph.node_count(), 2);
}

#[test]
fn failed_graph_save_keeps_dirty_state_and_withholds_root_metadata() {
    let mut graph = DirGraph::new();
    graph.enable_disk_mode().unwrap();
    let target = TempDir::new().unwrap();
    let blocked_generations = target.path().join("generations");
    std::fs::write(&blocked_generations, b"not a directory").unwrap();

    let error = graph
        .save_disk(target.path().to_str().unwrap())
        .expect_err("a blocked generations directory must fail the save");
    assert!(error.contains("Failed to begin disk generation"));
    assert!(!target.path().join("CURRENT").exists());
    match &graph.graph {
        GraphBackend::Disk(disk) => assert!(disk.persistence_is_dirty()),
        _ => panic!("graph unexpectedly left disk mode"),
    }

    std::fs::remove_file(blocked_generations).unwrap();
    graph.save_disk(target.path().to_str().unwrap()).unwrap();
    let snapshot =
        crate::graph::storage::disk::generation::resolve_snapshot(target.path()).unwrap();
    assert!(snapshot.snapshot_dir.join("metadata.json").exists());
    match &graph.graph {
        GraphBackend::Disk(disk) => assert!(!disk.persistence_is_dirty()),
        _ => panic!("graph unexpectedly left disk mode"),
    }
}

// ------------- fixture helpers -------------

fn seg(
    node_slots: Vec<DiskNodeSlot>,
    out_offsets: Vec<u64>,
    out_edges: Vec<CsrEdge>,
    in_offsets: Vec<u64>,
    in_edges: Vec<CsrEdge>,
    edge_endpoints: Vec<EdgeEndpoints>,
) -> SegmentCsr {
    SegmentCsr {
        node_slots: from_vec(node_slots),
        out_offsets: from_vec(out_offsets),
        out_edges: from_vec(out_edges),
        in_offsets: from_vec(in_offsets),
        in_edges: from_vec(in_edges),
        edge_endpoints: from_vec(edge_endpoints),
        // Phase-5 auxiliary fields default to empty in these
        // CSR-only unit tests. Dedicated phase-5 tests populate
        // them explicitly.
        conn_type_index_types: MmapOrVec::new(),
        conn_type_index_offsets: MmapOrVec::new(),
        conn_type_index_sources: MmapOrVec::new(),
        peer_count_types: MmapOrVec::new(),
        peer_count_offsets: MmapOrVec::new(),
        peer_count_entries: MmapOrVec::new(),
    }
}

fn from_vec<T: crate::graph::storage::mapped::mmap_vec::MmapPod>(v: Vec<T>) -> MmapOrVec<T> {
    let mut m: MmapOrVec<T> = MmapOrVec::with_capacity(v.len());
    for x in v {
        m.push(x);
    }
    m
}

fn slot(node_type: u64, row_id: u32) -> DiskNodeSlot {
    DiskNodeSlot {
        node_type,
        row_id,
        flags: DiskNodeSlot::ALIVE_BIT,
    }
}

#[test]
fn overlapping_query_guards_keep_materializations_alive() {
    let tmp = TempDir::new().unwrap();
    let graph = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    let mut interner = StringInterner::new();

    let first = graph.begin_query();
    graph
        .node_arena
        .lock()
        .unwrap()
        .push(Box::new(NodeData::new(
            Value::Int64(1),
            Value::String("one".into()),
            "Item".into(),
            HashMap::new(),
            &mut interner,
        )));
    graph
        .edge_arena
        .lock()
        .unwrap()
        .push(Box::new(EdgeData::new(
            "LINKS".into(),
            HashMap::new(),
            &mut interner,
        )));

    let second = graph.begin_query();
    assert_eq!(*graph.active_queries.lock().unwrap(), 2);
    assert_eq!(graph.node_arena.lock().unwrap().len(), 1);
    assert_eq!(graph.edge_arena.lock().unwrap().len(), 1);

    // A reset from another execution path must not invalidate either guard.
    graph.reset_arenas();
    assert_eq!(graph.node_arena.lock().unwrap().len(), 1);
    assert_eq!(graph.edge_arena.lock().unwrap().len(), 1);

    drop(first);
    assert_eq!(*graph.active_queries.lock().unwrap(), 1);
    assert_eq!(graph.node_arena.lock().unwrap().len(), 1);
    drop(second);
    assert_eq!(*graph.active_queries.lock().unwrap(), 0);

    // Reclamation is deferred until the next query begins, after the last
    // prior-generation reference is guaranteed to be gone.
    let next = graph.begin_query();
    assert!(graph.node_arena.lock().unwrap().is_empty());
    assert!(graph.edge_arena.lock().unwrap().is_empty());
    drop(next);
}

// ------------- segment_subdir + enumerate (pre-phase-7 cases) -------------

#[test]
fn segment_subdir_zero_pads_three_digits() {
    assert_eq!(segment_subdir(0), "seg_000");
    assert_eq!(segment_subdir(1), "seg_001");
    assert_eq!(segment_subdir(42), "seg_042");
    assert_eq!(segment_subdir(999), "seg_999");
    // Past 999 the name widens; enumerate sorts by parsed u32 so
    // this still round-trips cleanly.
    assert_eq!(segment_subdir(1234), "seg_1234");
}

#[test]
fn enumerate_segment_dirs_returns_sorted_ids() {
    let tmp = TempDir::new().unwrap();
    for id in [5u32, 0, 2, 17] {
        std::fs::create_dir_all(tmp.path().join(segment_subdir(id))).unwrap();
    }
    let got: Vec<u32> = enumerate_segment_dirs(tmp.path())
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(got, vec![0, 2, 5, 17]);
}

#[test]
fn enumerate_segment_dirs_skips_non_matching_entries() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("seg_000")).unwrap();
    std::fs::create_dir_all(tmp.path().join("seg_abc")).unwrap(); // unparsable
    std::fs::create_dir_all(tmp.path().join("not_a_segment")).unwrap();
    // Top-level files must not be mistaken for segments.
    std::fs::write(tmp.path().join("seg_001"), b"not-a-dir").unwrap();
    std::fs::write(tmp.path().join("disk_graph_meta.json"), b"{}").unwrap();

    let got: Vec<u32> = enumerate_segment_dirs(tmp.path())
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(got, vec![0]);
}

#[test]
fn enumerate_segment_dirs_on_missing_dir_returns_empty() {
    let tmp = TempDir::new().unwrap();
    let missing = tmp.path().join("does-not-exist");
    assert!(enumerate_segment_dirs(&missing).is_empty());
}

#[test]
fn enumerate_segment_dirs_empty_root_returns_empty() {
    let tmp = TempDir::new().unwrap();
    assert!(enumerate_segment_dirs(tmp.path()).is_empty());
}

// ------------- concat_segment_csrs (phase 7) -------------

#[test]
fn concat_empty_input_returns_all_empty() {
    let c = concat_segment_csrs(Vec::new()).unwrap();
    assert_eq!(c.node_slots.len(), 0);
    assert_eq!(c.out_offsets.len(), 0);
    assert_eq!(c.out_edges.len(), 0);
    assert_eq!(c.in_offsets.len(), 0);
    assert_eq!(c.in_edges.len(), 0);
    assert_eq!(c.edge_endpoints.len(), 0);
}

#[test]
fn concat_single_segment_returns_it_unchanged() {
    // Node 0 → node 1 (edge_idx 0). One node, one edge for simplicity
    // of comparison: passthrough must not mutate anything.
    let s = seg(
        vec![slot(7, 100), slot(7, 101)],
        vec![0, 1, 1], // out_offsets: node 0 emits edge [0,1), node 1 emits nothing
        vec![CsrEdge {
            peer: 1,
            edge_idx: 0,
        }],
        vec![0, 0, 1], // in_offsets: node 0 receives nothing, node 1 receives [0,1)
        vec![CsrEdge {
            peer: 0,
            edge_idx: 0,
        }],
        vec![EdgeEndpoints {
            source: 0,
            target: 1,
            connection_type: 42,
        }],
    );
    let c = concat_segment_csrs(vec![s]).unwrap();
    assert_eq!(c.node_slots.len(), 2);
    assert_eq!(c.out_offsets.len(), 3);
    assert_eq!(c.out_edges.len(), 1);
    assert_eq!(c.out_edges.get(0).edge_idx, 0); // unchanged
    assert_eq!(c.edge_endpoints.len(), 1);
    assert_eq!(c.edge_endpoints.get(0).source, 0);
}

#[test]
fn concat_two_segments_stitches_offsets_and_shifts_edge_idx() {
    // Segment 0: 2 nodes, 1 intra-segment edge  0 → 1
    let s0 = seg(
        vec![slot(1, 10), slot(1, 11)],
        vec![0, 1, 1],
        vec![CsrEdge {
            peer: 1,
            edge_idx: 0,
        }],
        vec![0, 0, 1],
        vec![CsrEdge {
            peer: 0,
            edge_idx: 0,
        }],
        vec![EdgeEndpoints {
            source: 0,
            target: 1,
            connection_type: 100,
        }],
    );
    // Segment 1: 2 nodes (global ids 2, 3), 2 intra-segment edges
    // 2 → 3 (segment-local edge_idx 0) and 3 → 2 (segment-local 1).
    let s1 = seg(
        vec![slot(2, 20), slot(2, 21)],
        vec![0, 1, 2],
        vec![
            CsrEdge {
                peer: 3,
                edge_idx: 0,
            },
            CsrEdge {
                peer: 2,
                edge_idx: 1,
            },
        ],
        vec![0, 1, 2],
        vec![
            CsrEdge {
                peer: 3,
                edge_idx: 1,
            },
            CsrEdge {
                peer: 2,
                edge_idx: 0,
            },
        ],
        vec![
            EdgeEndpoints {
                source: 2,
                target: 3,
                connection_type: 200,
            },
            EdgeEndpoints {
                source: 3,
                target: 2,
                connection_type: 201,
            },
        ],
    );

    let c = concat_segment_csrs(vec![s0, s1]).unwrap();

    // Shape.
    assert_eq!(c.node_slots.len(), 4);
    assert_eq!(c.out_offsets.len(), 5); // n+1
    assert_eq!(c.in_offsets.len(), 5);
    assert_eq!(c.out_edges.len(), 3);
    assert_eq!(c.in_edges.len(), 3);
    assert_eq!(c.edge_endpoints.len(), 3);

    // Stitched out_offsets: [0, 1, 1, 2, 3] — seg 0 contributes
    // [0,1,1]; seg 1 contributes [+1, +2] atop seg 0's last (=1),
    // so combined ends [..., 2, 3].
    let out_off: Vec<u64> = (0..c.out_offsets.len())
        .map(|i| c.out_offsets.get(i))
        .collect();
    assert_eq!(out_off, vec![0, 1, 1, 2, 3]);

    // Stitched in_offsets: [0, 0, 1, 2, 3] — seg 0 [0,0,1]; seg 1
    // contributes [+1, +2].
    let in_off: Vec<u64> = (0..c.in_offsets.len())
        .map(|i| c.in_offsets.get(i))
        .collect();
    assert_eq!(in_off, vec![0, 0, 1, 2, 3]);

    // out_edges[0] comes from seg 0, edge_idx unchanged (0).
    // out_edges[1..3] come from seg 1, edge_idx shifted by seg 0's
    // edge_endpoints.len() == 1 → (1, 2).
    assert_eq!(c.out_edges.get(0).edge_idx, 0);
    assert_eq!(c.out_edges.get(0).peer, 1);
    assert_eq!(c.out_edges.get(1).edge_idx, 1);
    assert_eq!(c.out_edges.get(1).peer, 3);
    assert_eq!(c.out_edges.get(2).edge_idx, 2);
    assert_eq!(c.out_edges.get(2).peer, 2);

    // in_edges shifts follow the same rule.
    assert_eq!(c.in_edges.get(0).edge_idx, 0); // seg 0, unchanged
    assert_eq!(c.in_edges.get(1).edge_idx, 2); // seg 1, +1
    assert_eq!(c.in_edges.get(2).edge_idx, 1); // seg 1, +1

    // edge_endpoints concat — source/target are global node ids.
    assert_eq!(c.edge_endpoints.get(0).source, 0);
    assert_eq!(c.edge_endpoints.get(0).target, 1);
    assert_eq!(c.edge_endpoints.get(1).source, 2);
    assert_eq!(c.edge_endpoints.get(1).target, 3);
    assert_eq!(c.edge_endpoints.get(2).source, 3);
    assert_eq!(c.edge_endpoints.get(2).target, 2);
}

#[test]
fn concat_three_segments_keeps_offset_chain_consistent() {
    // Three one-node-one-self-edge segments. Verifies that the
    // cumulative shifts carry through multiple iterations.
    let mk_one_node = |global_id: u32, conn: u64| {
        seg(
            vec![slot(1, 0)],
            vec![0, 1],
            vec![CsrEdge {
                peer: global_id,
                edge_idx: 0,
            }],
            vec![0, 1],
            vec![CsrEdge {
                peer: global_id,
                edge_idx: 0,
            }],
            vec![EdgeEndpoints {
                source: global_id,
                target: global_id,
                connection_type: conn,
            }],
        )
    };
    let c = concat_segment_csrs(vec![
        mk_one_node(0, 10),
        mk_one_node(1, 20),
        mk_one_node(2, 30),
    ])
    .unwrap();

    let out_off: Vec<u64> = (0..c.out_offsets.len())
        .map(|i| c.out_offsets.get(i))
        .collect();
    assert_eq!(out_off, vec![0, 1, 2, 3]);

    // Each out_edges entry's edge_idx should point at its own
    // self-loop's endpoint in the combined array — segment K's
    // endpoint lands at index K.
    assert_eq!(c.out_edges.get(0).edge_idx, 0);
    assert_eq!(c.out_edges.get(1).edge_idx, 1);
    assert_eq!(c.out_edges.get(2).edge_idx, 2);

    // The endpoint at edge_idx K should be the self-loop of node K.
    for k in 0..3 {
        assert_eq!(c.edge_endpoints.get(k).source, k as u32);
        assert_eq!(c.edge_endpoints.get(k).target, k as u32);
    }
}

#[test]
fn concat_handles_edgeless_segment() {
    // A segment with nodes but no edges (e.g. a freshly-created
    // empty segment). Offsets must still stitch correctly.
    let s0 = seg(
        vec![slot(1, 0)],
        vec![0, 1],
        vec![CsrEdge {
            peer: 0,
            edge_idx: 0,
        }],
        vec![0, 1],
        vec![CsrEdge {
            peer: 0,
            edge_idx: 0,
        }],
        vec![EdgeEndpoints {
            source: 0,
            target: 0,
            connection_type: 1,
        }],
    );
    let s_empty = seg(
        vec![slot(1, 1)],
        vec![0, 0],
        Vec::new(),
        vec![0, 0],
        Vec::new(),
        Vec::new(),
    );
    let s1 = seg(
        vec![slot(1, 2)],
        vec![0, 1],
        vec![CsrEdge {
            peer: 2,
            edge_idx: 0,
        }],
        vec![0, 1],
        vec![CsrEdge {
            peer: 2,
            edge_idx: 0,
        }],
        vec![EdgeEndpoints {
            source: 2,
            target: 2,
            connection_type: 3,
        }],
    );
    let c = concat_segment_csrs(vec![s0, s_empty, s1]).unwrap();
    let out_off: Vec<u64> = (0..c.out_offsets.len())
        .map(|i| c.out_offsets.get(i))
        .collect();
    // 3 nodes total; middle node contributes no edges.
    assert_eq!(out_off, vec![0, 1, 1, 2]);
    assert_eq!(c.out_edges.len(), 2);
    // Segment 2 (index 2 in the input) had endpoint_base of
    // s0.edge_endpoints.len() + s_empty.edge_endpoints.len() == 1,
    // so its self-loop's edge_idx should now be 1.
    assert_eq!(c.out_edges.get(1).edge_idx, 1);
}

// ------------- seal_to_new_segment round-trip (phase 8) -------------

fn seal_test_node(interner: &mut StringInterner, id: i64, ntype: &str) -> NodeData {
    NodeData::new(
        Value::Int64(id),
        Value::String(format!("n{id}")),
        ntype.to_string(),
        std::collections::HashMap::new(),
        interner,
    )
}

fn seal_test_edge(interner: &mut StringInterner, ct: &str) -> EdgeData {
    EdgeData::new(ct.to_string(), std::collections::HashMap::new(), interner)
}

#[test]
fn seal_rejects_when_nothing_to_seal() {
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    dg.defer_csr = true;
    let _n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
    dg.build_csr_from_pending().unwrap();
    dg.save_to_dir(tmp.path(), &interner).unwrap();
    // save_to_dir set sealed_nodes_bound = node_count, so tail is empty.
    let err = dg.seal_to_new_segment(tmp.path()).unwrap_err();
    assert!(err.to_string().contains("nothing to seal"));
}

#[test]
fn seal_accepts_cross_segment_edges_via_full_range() {
    // Phase 7: cross-segment overflow no longer errors. The new
    // segment writes full-range out_offsets (indexed by global
    // node id) so an edge from a seg_0 source into a tail target
    // — or between two seg_0 sources — is reachable after reload.
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    dg.defer_csr = true;
    let n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
    let n1 = dg.add_node(seal_test_node(&mut interner, 1, "A"));
    dg.add_edge(n0, n1, seal_test_edge(&mut interner, "T"));
    dg.build_csr_from_pending().unwrap();
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    // Add a new node and a cross-segment edge (seg_0's n0 → new n2).
    let n2 = dg.add_node(seal_test_node(&mut interner, 2, "A"));
    dg.add_edge(n0, n2, seal_test_edge(&mut interner, "T"));

    let seg_id = dg.seal_to_new_segment(tmp.path()).unwrap();
    assert_eq!(seg_id, 1);

    // seg_001's out_offsets must cover every global node (4 entries
    // for node_count=3) so concat can locate the n0→n2 edge.
    let out_offsets_size = std::fs::metadata(tmp.path().join("seg_001/out_offsets.bin"))
        .unwrap()
        .len();
    assert_eq!(
        out_offsets_size,
        (3 + 1) * 8,
        "seg_001 must be full-range (4 u64 offsets covering 3 global nodes)"
    );

    // Reload and verify the combined CSR sees BOTH seg_0's n0→n1 and
    // seg_1's n0→n2 as outgoing edges of node 0.
    drop(dg);
    let mut interner2 = StringInterner::new();
    let (reloaded, _tmp_zst) = super::DiskGraph::load_from_dir(tmp.path(), &mut interner2).unwrap();
    let start = reloaded.out_offsets.get(0) as usize;
    let end = reloaded.out_offsets.get(1) as usize;
    let peers: Vec<u32> = (start..end)
        .map(|i| reloaded.out_edges.get(i).peer)
        .collect();
    assert!(peers.contains(&1), "missing seg_0 edge n0→n1");
    assert!(peers.contains(&2), "missing seg_1 cross-segment edge n0→n2");
}

#[test]
fn seal_round_trip_basic_reads() {
    // Build seg_0: 3 nodes of type A with one edge between them.
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    dg.defer_csr = true;
    let n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
    let n1 = dg.add_node(seal_test_node(&mut interner, 1, "A"));
    let _n2 = dg.add_node(seal_test_node(&mut interner, 2, "A"));
    dg.add_edge(n0, n1, seal_test_edge(&mut interner, "T"));
    dg.build_csr_from_pending().unwrap();
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    assert_eq!(dg.node_count, 3);
    assert_eq!(dg.sealed_nodes_bound, 3);

    // Add 2 new nodes of type B + an edge between them.
    // Both endpoints are strictly above the watermark, so the
    // seal constraint holds.
    let n3 = dg.add_node(seal_test_node(&mut interner, 3, "B"));
    let n4 = dg.add_node(seal_test_node(&mut interner, 4, "B"));
    dg.add_edge(n3, n4, seal_test_edge(&mut interner, "U"));

    let pre_edge_count = dg.edge_count;
    let pre_node_count = dg.node_count;
    assert_eq!(pre_node_count, 5);
    assert_eq!(pre_edge_count, 2);

    // Seal the tail to seg_001.
    let seg_id = dg.seal_to_new_segment(tmp.path()).unwrap();
    assert_eq!(seg_id, 1);
    assert_eq!(dg.sealed_nodes_bound, 5);
    assert!(dg.overflow_out.is_empty());
    assert!(dg.overflow_in.is_empty());

    // Verify seg_001/ has the expected files on disk.
    let seg1 = tmp.path().join("seg_001");
    for name in [
        "node_slots.bin",
        "out_offsets.bin",
        "out_edges.bin",
        "in_offsets.bin",
        "in_edges.bin",
        "edge_endpoints.bin",
    ] {
        assert!(seg1.join(name).exists(), "missing {name}");
    }
    // Manifest should have 2 entries now.
    let manifest = super::super::segment_summary::SegmentManifest::load_from(tmp.path()).unwrap();
    assert_eq!(manifest.len(), 2);
    assert_eq!(manifest.segments[1].segment_id, 1);
    assert_eq!(manifest.segments[1].node_id_lo, 3);
    assert_eq!(manifest.segments[1].node_id_hi, 5);
    assert_eq!(manifest.segments[1].edge_count, 1);

    // Drop in-memory graph and reload — this exercises the phase-7
    // concat read path.
    drop(dg);
    let mut interner2 = StringInterner::new();
    let (reloaded, _tmp_zst) = super::DiskGraph::load_from_dir(tmp.path(), &mut interner2).unwrap();

    assert_eq!(reloaded.node_count, pre_node_count);
    assert_eq!(reloaded.edge_count, pre_edge_count);
    // sealed_nodes_bound persists through save/load.
    assert_eq!(reloaded.sealed_nodes_bound, 5);

    // Untyped outgoing edges for node 3 should be exactly 1 (the
    // n3 → n4 edge). This verifies the concat stitched the
    // out_offsets correctly for the sealed tail.
    let n3_idx = 3usize;
    let start = reloaded.out_offsets.get(n3_idx) as usize;
    let end = reloaded.out_offsets.get(n3_idx + 1) as usize;
    assert_eq!(end - start, 1, "expected 1 outgoing edge for seg_1 node 3");
    let e = reloaded.out_edges.get(start);
    assert_eq!(e.peer, 4);
    // Combined edge_idx = segment-local 0 + endpoint_base (= seg_0's 1)
    // = 1. Verifies the phase-7 concat shift lands at the right slot.
    assert_eq!(e.edge_idx, 1);
    // edge_endpoints at that global index should hold the original
    // global node ids.
    let ep = reloaded.edge_endpoints.get(1);
    assert_eq!(ep.source, 3);
    assert_eq!(ep.target, 4);

    // And seg_0's original edge (0 → 1) is still present at
    // combined edge_idx 0.
    let ep0 = reloaded.edge_endpoints.get(0);
    assert_eq!(ep0.source, 0);
    assert_eq!(ep0.target, 1);
    let _ = (n1, n4); // silence unused-warnings if optimised out
}

#[test]
fn seal_round_trip_auxiliary_indexes() {
    // Phase 5: verify that conn_type_index_*, peer_count_*, and
    // edge_properties survive the seal → reload roundtrip for edges
    // in the sealed segment.
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    dg.defer_csr = true;
    // seg_0: 3 nodes of type A with one T edge (0 → 1).
    let n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
    let n1 = dg.add_node(seal_test_node(&mut interner, 1, "A"));
    let _n2 = dg.add_node(seal_test_node(&mut interner, 2, "A"));
    dg.add_edge(n0, n1, seal_test_edge(&mut interner, "T"));
    dg.build_csr_from_pending().unwrap();
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    // Tail: 2 nodes of type B, 2 U-edges — one intra-tail (3→4), one
    // self-loop on 3. Attach a property on the self-loop to verify
    // edge_properties flushing on seal.
    let n3 = dg.add_node(seal_test_node(&mut interner, 3, "B"));
    let n4 = dg.add_node(seal_test_node(&mut interner, 4, "B"));
    dg.add_edge(n3, n4, seal_test_edge(&mut interner, "U"));
    let self_loop = {
        let weight_key = interner.get_or_intern("weight");
        let ed = crate::graph::schema::EdgeData {
            connection_type: interner.get_or_intern("U"),
            properties: vec![(weight_key, Value::Float64(2.5))],
        };
        dg.add_edge(n3, n3, ed)
    };

    dg.seal_to_new_segment(tmp.path()).unwrap();

    // seg_001 has its own auxiliary files now (phase 5).
    let seg1 = tmp.path().join("seg_001");
    for name in [
        "conn_type_index_types.bin",
        "conn_type_index_offsets.bin",
        "conn_type_index_sources.bin",
        "peer_count_types.bin",
        "peer_count_offsets.bin",
        "peer_count_entries.bin",
    ] {
        assert!(seg1.join(name).exists(), "phase-5 missing {name}");
    }

    drop(dg);
    let mut interner2 = StringInterner::new();
    let (reloaded, _tmp_zst) = super::DiskGraph::load_from_dir(tmp.path(), &mut interner2).unwrap();

    // conn_type_index should cover BOTH T (from seg_0) and U (from
    // seg_1). Merge is what `concat_segment_csrs::merge_conn_type_index`
    // produces.
    let t_key = interner2.get_or_intern("T").as_u64();
    let u_key = interner2.get_or_intern("U").as_u64();
    let cti_types: Vec<u64> = (0..reloaded.conn_type_index_types.len())
        .map(|i| reloaded.conn_type_index_types.get(i))
        .collect();
    assert!(
        cti_types.contains(&t_key),
        "T missing from merged conn_type_index"
    );
    assert!(
        cti_types.contains(&u_key),
        "U missing from merged conn_type_index"
    );

    // peer_count histogram for U should report node 4 as a target
    // once (from n3 → n4) and node 3 as a target once (self-loop).
    let u_counts = reloaded
        .lookup_peer_counts(u_key)
        .expect("U histogram present");
    assert_eq!(u_counts.get(&4), Some(&1));
    assert_eq!(u_counts.get(&3), Some(&1));

    // And T's histogram still has the seg_0 entry intact.
    let t_counts = reloaded
        .lookup_peer_counts(t_key)
        .expect("T histogram present");
    assert_eq!(t_counts.get(&1), Some(&1));

    // Edge properties on the self-loop should survive — combined
    // edge_idx matches the original global assignment
    // (seg_0's 1 edge + seg_1's local index), which equals the
    // self_loop edge_index we captured pre-seal.
    let weight_key = interner2.get_or_intern("weight");
    let weight = reloaded
        .edge_properties
        .get(self_loop.index() as u32)
        .expect("self_loop has props");
    let (k, v) = &weight.as_ref()[0];
    assert_eq!(*k, weight_key);
    assert_eq!(*v, Value::Float64(2.5));
}

#[test]
fn save_to_dir_auto_wires_seal_when_tail_is_clean() {
    // Phase 6: a second save after a clean-tail workload should
    // dispatch to `seal_to_new_segment` instead of the traditional
    // compact-and-rewrite path. Verify by checking that seg_001/
    // appears on disk and that the manifest grows to 2 segments.
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    dg.defer_csr = true;
    let n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
    let n1 = dg.add_node(seal_test_node(&mut interner, 1, "A"));
    dg.add_edge(n0, n1, seal_test_edge(&mut interner, "T"));
    dg.build_csr_from_pending().unwrap();
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    // After first save: seg_000 exists, seg_001 does not.
    assert!(tmp.path().join("seg_000").exists());
    assert!(!tmp.path().join("seg_001").exists());
    assert_eq!(dg.sealed_nodes_bound, 2);

    // Tail: 2 new nodes, 1 intra-tail edge.
    let n2 = dg.add_node(seal_test_node(&mut interner, 2, "B"));
    let n3 = dg.add_node(seal_test_node(&mut interner, 3, "B"));
    dg.add_edge(n2, n3, seal_test_edge(&mut interner, "U"));

    // Second save: auto-wire should trigger seal_to_new_segment.
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    assert!(
        tmp.path().join("seg_001").exists(),
        "phase-6 auto-wire should have produced seg_001/"
    );
    assert_eq!(dg.sealed_nodes_bound, 4);

    let manifest = super::super::segment_summary::SegmentManifest::load_from(tmp.path()).unwrap();
    assert_eq!(manifest.len(), 2, "manifest should have 2 segments");

    // Reload and verify the combined view still makes sense.
    drop(dg);
    let mut interner2 = StringInterner::new();
    let (reloaded, _tmp_zst) = super::DiskGraph::load_from_dir(tmp.path(), &mut interner2).unwrap();
    assert_eq!(reloaded.node_count, 4);
    assert_eq!(reloaded.edge_count, 2);
}

#[test]
fn save_to_dir_seals_cross_segment_overflow_as_full_range() {
    // Phase 7: cross-segment overflow is now handled by the full-
    // range seal, not the compact fallback. save_to_dir should
    // produce seg_001/ even when the overflow has an old→new edge,
    // and reload must see both segments' edges combined.
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    dg.defer_csr = true;
    let n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
    let n1 = dg.add_node(seal_test_node(&mut interner, 1, "A"));
    dg.add_edge(n0, n1, seal_test_edge(&mut interner, "T"));
    dg.build_csr_from_pending().unwrap();
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    // Cross-segment edge (old n0 → new n2) + a purely-tail edge.
    let n2 = dg.add_node(seal_test_node(&mut interner, 2, "B"));
    let n3 = dg.add_node(seal_test_node(&mut interner, 3, "B"));
    dg.add_edge(n0, n2, seal_test_edge(&mut interner, "T"));
    dg.add_edge(n2, n3, seal_test_edge(&mut interner, "U"));

    dg.save_to_dir(tmp.path(), &interner).unwrap();
    assert!(
        tmp.path().join("seg_001").exists(),
        "phase-7 save_to_dir should seal cross-segment overflow"
    );

    drop(dg);
    let mut interner2 = StringInterner::new();
    let (reloaded, _tmp_zst) = super::DiskGraph::load_from_dir(tmp.path(), &mut interner2).unwrap();
    assert_eq!(reloaded.node_count, 4);
    assert_eq!(reloaded.edge_count, 3);
}

#[test]
fn conn_type_index_sources_are_global_after_segment_local_seal() {
    // Regression test for the 0.8.11 seal-path bug where the
    // merged `conn_type_index_sources` stored segment-local
    // indices (0..tail_len) instead of global node ids for
    // segment-local segments. Symptom: post-reload,
    // `MATCH (a)-[:T]->(b) RETURN a.id, b.id` returned no rows
    // even though `count(*)` reported the right number — the
    // enumeration looked up `out_offsets[0]` (empty, Person
    // range) instead of `out_offsets[tail_lo]` (the actual
    // TestHuman range).
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    // Stay in the default `defer_csr = false` mode — mirrors the
    // path Python mutations take (ntriples is the only caller that
    // flips defer_csr on, and it always owns a matching
    // build_csr_from_pending after its batch).
    // seg_0: 5 nodes, no edges.
    for i in 0..5 {
        dg.add_node(seal_test_node(&mut interner, i, "A"));
    }
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    // Tail: 3 nodes + 3 intra-tail edges → segment-local seal.
    // Edges go into overflow (defer_csr is false).
    let n5 = dg.add_node(seal_test_node(&mut interner, 5, "B"));
    let n6 = dg.add_node(seal_test_node(&mut interner, 6, "B"));
    let n7 = dg.add_node(seal_test_node(&mut interner, 7, "B"));
    dg.add_edge(n5, n6, seal_test_edge(&mut interner, "T"));
    dg.add_edge(n6, n7, seal_test_edge(&mut interner, "T"));
    dg.add_edge(n7, n5, seal_test_edge(&mut interner, "T"));
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    drop(dg);
    let mut interner2 = StringInterner::new();
    let (reloaded, _tmp_zst) = super::DiskGraph::load_from_dir(tmp.path(), &mut interner2).unwrap();

    // The merged conn_type_index sources must be the GLOBAL ids
    // {5, 6, 7}, not the segment-local {0, 1, 2}.
    let t_key = interner2.get_or_intern("T").as_u64();
    let sources = reloaded
        .sources_for_conn_type(t_key)
        .expect("T index should exist");
    let mut sources = sources;
    sources.sort_unstable();
    assert_eq!(
        sources,
        vec![5u32, 6, 7],
        "segment-local seal's conn_type_index_sources must be shifted \
         by node_lo on merge (regression from 0.8.11 pre-fix)"
    );
}

#[test]
fn compact_rewrite_after_seal_cleans_stale_segs_and_persists_heap_arrays() {
    // Regression test for the 0.8.11 seal-path bugs C and D. Flow
    // that used to fail:
    //   1. Build + save       → seg_000
    //   2. Add nodes + edges  → overflow
    //   3. Save               → seal creates seg_001; reconcile_seg0_csr
    //                           replaces self.{node_slots, …} with
    //                           `MmapOrVec::Heap` copies
    //   4. Add more overflow edges between *existing* nodes (no new
    //                           nodes, so `sealed_nodes_bound ==
    //                           node_count` → falls to compact-rewrite)
    //   5. Save again         → compact-rewrite path
    //
    // Before the fixes, step 5 (a) left seg_001 on disk so reload's
    // `enumerate_segment_dirs` double-counted, and (b) relied on mmap
    // persistence for the core arrays — which was impossible after
    // reconcile made them heap-backed — so the on-disk node_slots /
    // edge_endpoints files stayed at the pre-seal trimmed sizes and
    // reload errored with "File too small".
    let tmp = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
    // seg_0: 5 nodes.
    for i in 0..5 {
        dg.add_node(seal_test_node(&mut interner, i, "A"));
    }
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    // Tail: add 3 nodes + intra-tail edges.
    let n5 = dg.add_node(seal_test_node(&mut interner, 5, "B"));
    let n6 = dg.add_node(seal_test_node(&mut interner, 6, "B"));
    let n7 = dg.add_node(seal_test_node(&mut interner, 7, "B"));
    dg.add_edge(n5, n6, seal_test_edge(&mut interner, "T"));
    dg.add_edge(n6, n7, seal_test_edge(&mut interner, "T"));
    dg.save_to_dir(tmp.path(), &interner).unwrap(); // seal → seg_001

    assert!(
        tmp.path().join("seg_001").exists(),
        "phase-6 seal should have produced seg_001"
    );

    // Now add *edges* between existing nodes only — `sealed_nodes_bound
    // == node_count` so this next save will take the compact-rewrite
    // path, not another seal.
    let n0 = NodeIndex::new(0);
    dg.add_edge(n0, n5, seal_test_edge(&mut interner, "T"));
    dg.save_to_dir(tmp.path(), &interner).unwrap();

    // Fix C: stale seg_001 must be removed.
    assert!(
        !tmp.path().join("seg_001").exists(),
        "compact-rewrite must clean up stale seg_NNN dirs"
    );

    // Fix D: reload must succeed — the heap-backed arrays from
    // reconcile must have been explicitly persisted by save_to_file.
    drop(dg);
    let mut interner2 = StringInterner::new();
    let (reloaded, _tmp_zst) = super::DiskGraph::load_from_dir(tmp.path(), &mut interner2).unwrap();
    assert_eq!(reloaded.node_count, 8, "all 8 nodes must survive");
    assert_eq!(reloaded.edge_count, 3, "3 T-edges must survive");
}
