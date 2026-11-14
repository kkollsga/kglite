//! A mapped open of a CRC-valid v9 file whose topology sections are
//! inconsistent. Each case re-seals the section and metadata checksums, so
//! only the structural checks can object; the expectation is absolute, never
//! a comparison with another load.

use super::container_test_support::patch_section;
use super::frozen_tests::{saved_topo, topo_graph};
use super::index_sections::with_floors;
use super::index_sections_tests::{rows, save};
use super::mappable::SaveLayout;
use super::*;
use crate::graph::introspection::describe::{compute_description, DescribeRequest};
use crate::graph::introspection::DescribeSurface;

const EDGE_RECORD: usize = 12;
const VACANT_SRC: [u8; 4] = u32::MAX.to_le_bytes();

fn load_mapped(path: &std::path::Path) -> io::Result<Arc<DirGraph>> {
    load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new().with_storage(StorageMode::Mapped),
    )
}

/// Rewrite the first live endpoint record (`src`, `dst`, `ty`, pad).
fn patch_first_live_edge(path: &std::path::Path, patch: impl FnOnce(&mut [u8])) {
    patch_section(path, "topo:edges", |bytes| {
        let record = bytes
            .chunks_exact_mut(EDGE_RECORD)
            .find(|r| r[..4] != VACANT_SRC)
            .expect("the fixture has a live edge");
        patch(record);
    });
}

fn patch_meta(path: &std::path::Path, from: &str, to: &str) {
    assert_eq!(from.len(), to.len(), "the metadata keeps its length");
    patch_section(path, "topo:meta", |bytes| {
        let text = std::str::from_utf8(bytes).unwrap().replacen(from, to, 1);
        assert_ne!(text.as_bytes(), bytes, "{from} not found in the metadata");
        bytes.copy_from_slice(text.as_bytes());
    });
}

const BY_TYPE: &str = "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t";

/// Everything a session reads whole-graph, none of which may panic: the edge
/// type consumers, a query naming every edge's type, `describe()`, and a save,
/// whose outcome is returned.
fn exercise(graph: &Arc<DirGraph>, dest: &std::path::Path) -> io::Result<()> {
    graph.get_edge_type_counts();
    graph.get_or_compute_type_connectivity();
    let _ = rows(graph, BY_TYPE);
    let _ = compute_description(graph, &DescribeRequest::new(DescribeSurface::Python));
    let mut copy = Arc::clone(graph);
    with_floors(10, 10, || {
        save_graph_with_layout(&mut copy, dest.to_str().unwrap(), false, SaveLayout::Auto)
    })
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

fn open_ok(path: &std::path::Path) -> Arc<DirGraph> {
    load_mapped(path).unwrap_or_else(|e| panic!("the open must succeed: {e}"))
}

fn resaved(path: &std::path::Path) -> std::path::PathBuf {
    path.with_file_name("resaved.kgl")
}

#[test]
fn an_edge_type_ordinal_past_the_type_list_reads_as_the_invalid_type() {
    let (_tmp, path) = saved_topo();
    let honest = rows(&open_ok(&path), BY_TYPE);
    patch_first_live_edge(&path, |r| r[8..10].copy_from_slice(&9999u16.to_le_bytes()));
    let graph = open_ok(&path);
    let crafted = rows(&graph, BY_TYPE);
    assert_ne!(crafted, honest);
    assert!(crafted.contains("<invalid connection type>"), "{crafted}");
    exercise(&graph, &resaved(&path)).expect("A1 leaves the save to succeed");
}

#[test]
fn an_edge_endpoint_past_the_node_array_does_not_change_a_query() {
    for at in [0, 4] {
        let (_tmp, path) = saved_topo();
        let honest = rows(&open_ok(&path), BY_TYPE);
        patch_first_live_edge(&path, |r| {
            r[at..at + 4].copy_from_slice(&900_000u32.to_le_bytes())
        });
        let graph = open_ok(&path);
        assert_eq!(rows(&graph, BY_TYPE), honest, "field at byte {at}");
        exercise(&graph, &resaved(&path)).expect("A1 leaves the save to succeed");
    }
}

#[test]
fn a_renamed_node_type_in_the_metadata_loads_and_the_save_fails() {
    let (_tmp, path) = saved_topo();
    patch_meta(&path, "\"A\"", "\"Z\"");
    let graph = open_ok(&path);
    let error = exercise(&graph, &resaved(&path)).expect_err("a record of an unnamed type");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
}

#[test]
fn a_record_naming_an_unlisted_type_without_member_lists_is_refused_at_open() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    // Topology sections (edges at the floor) but no node member lists
    // (nodes below theirs): the load rebuilds the type index from the records.
    with_floors(1_000_000, 10, || {
        save(topo_graph(), &path, SaveLayout::Auto)
    });
    patch_section(&path, "topo:nodes", |bytes| {
        let live = bytes.chunks_exact_mut(32).nth(100).unwrap();
        live[..8].copy_from_slice(&0xDEAD_BEEF_1234_5678u64.to_le_bytes());
    });
    let error = load_mapped(&path).err().expect("the open must refuse");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
}
