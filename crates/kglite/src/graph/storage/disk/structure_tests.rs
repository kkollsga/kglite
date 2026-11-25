use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints, TOMBSTONE_EDGE};
use super::graph::DiskGraph;
use crate::graph::schema::StringInterner;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use tempfile::TempDir;

struct Fixture {
    _dir: TempDir,
    graph: DiskGraph,
    interner: StringInterner,
    person: u64,
    knows: u64,
}

fn slot(node_type: u64, alive: bool) -> DiskNodeSlot {
    DiskNodeSlot {
        node_type,
        row_id: 0,
        flags: u32::from(alive),
    }
}

fn edge(source: u32, target: u32, connection_type: u64) -> EdgeEndpoints {
    EdgeEndpoints {
        source,
        target,
        connection_type,
    }
}

/// Three nodes (the middle one a free slot), two edges (one tombstoned), and
/// CSR arrays in which node 2 has no offsets entry (overflow-only).
fn fixture() -> Fixture {
    let dir = TempDir::new().unwrap();
    let mut interner = StringInterner::new();
    let person = interner.get_or_intern("Person").as_u64();
    let knows = interner.get_or_intern("KNOWS").as_u64();
    let mut graph = DiskGraph::new_at_path(dir.path()).unwrap();
    graph.node_slots = MmapOrVec::from_vec(vec![
        slot(person, true),
        slot(0xDEAD, false),
        slot(person, true),
    ]);
    graph.edge_endpoints = MmapOrVec::from_vec(vec![
        edge(0, 2, knows),
        edge(TOMBSTONE_EDGE, TOMBSTONE_EDGE, 0xBEEF),
    ]);
    graph.next_edge_idx = 2;
    let csr = |peer| CsrEdge { peer, edge_idx: 0 };
    graph.out_offsets = MmapOrVec::from_vec(vec![0, 1, 1]);
    graph.out_edges = MmapOrVec::from_vec(vec![csr(2)]);
    graph.in_offsets = MmapOrVec::from_vec(vec![0, 0, 0, 1]);
    graph.in_edges = MmapOrVec::from_vec(vec![csr(0)]);
    Fixture {
        _dir: dir,
        graph,
        interner,
        person,
        knows,
    }
}

fn err(f: &Fixture) -> String {
    f.graph.check_structure(&f.interner).unwrap_err()
}

#[test]
fn valid_graph_passes_with_free_slot_tombstone_and_short_offsets() {
    let f = fixture();
    f.graph.check_structure(&f.interner).unwrap();
}

#[test]
fn empty_graph_passes() {
    let dir = TempDir::new().unwrap();
    let graph = DiskGraph::new_at_path(dir.path()).unwrap();
    graph.check_structure(&StringInterner::new()).unwrap();
}

#[test]
fn alive_slot_with_unknown_type_is_rejected() {
    let mut f = fixture();
    f.graph.node_slots = MmapOrVec::from_vec(vec![slot(f.person, true), slot(0x1234, true)]);
    f.graph.next_edge_idx = 0;
    f.graph.edge_endpoints = MmapOrVec::new();
    assert!(err(&f).starts_with("node_slots: slot 1 "));
}

#[test]
fn live_edge_with_unknown_connection_type_is_rejected() {
    let mut f = fixture();
    f.graph.edge_endpoints = MmapOrVec::from_vec(vec![edge(0, 2, f.knows), edge(0, 2, 0x77)]);
    assert!(err(&f).starts_with("edge_endpoints: edge 1 has unknown connection type"));
}

#[test]
fn edge_endpoint_past_node_range_is_rejected() {
    let mut f = fixture();
    f.graph.edge_endpoints = MmapOrVec::from_vec(vec![edge(0, 3, f.knows), edge(0, 0, f.knows)]);
    assert!(err(&f).contains("edge 0 joins nodes 0 and 3"));
}

#[test]
fn edge_counter_past_the_endpoint_array_is_rejected() {
    let mut f = fixture();
    f.graph.next_edge_idx = 3;
    assert!(err(&f).starts_with("edge_endpoints: next_edge_idx 3"));
}

#[test]
fn offsets_must_start_at_zero_ascend_and_stay_inside_the_edges() {
    let cases: [(&[u64], &str); 3] = [
        (&[1, 1, 1], "first offset is 1"),
        (&[0, 1, 0], "below the previous offset"),
        (&[0, 1, 9], "past the 1 stored edges"),
    ];
    for (offsets, expect) in cases {
        let mut f = fixture();
        f.graph.out_offsets = MmapOrVec::from_vec(offsets.to_vec());
        let message = err(&f);
        assert!(message.starts_with("out_offsets: "), "{message}");
        assert!(message.contains(expect), "{message}");
        let mut f = fixture();
        f.graph.in_offsets = MmapOrVec::from_vec(offsets.to_vec());
        assert!(err(&f).starts_with("in_offsets: "));
    }
}

#[test]
fn deferred_csr_is_not_checked() {
    let mut f = fixture();
    f.graph.out_offsets = MmapOrVec::from_vec(vec![0, 9]);
    f.graph.defer_csr = true;
    f.graph.check_structure(&f.interner).unwrap();
}
