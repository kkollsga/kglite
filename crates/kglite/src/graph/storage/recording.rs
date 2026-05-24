//! Validation backend — [`RecordingGraph`].
//!
//! Phase 6 of the 0.8.0 storage refactor. `RecordingGraph<G>` is a
//! thin wrapper that implements [`GraphRead`] + [`GraphWrite`] on top
//! of any `G: GraphRead`, delegating every call while appending the
//! method name to an interior-mutable `log`. It exists to prove the
//! trait surface is complete enough that a 4th backend drops in
//! without touching the hot paths — the "open/closed" demonstration
//! called out in the Phase 6 report-out.
//!
//! Scope is Rust-only: no Python constructor reaches [`RecordingGraph`].
//! The in-source `#[cfg(test)] mod tests` module below exercises the
//! wrapper against `MemoryGraph`, `MappedGraph`, and `DiskGraph`, and
//! drives the [`crate::graph::schema::GraphBackend::Recording`] enum
//! variant through the existing dispatcher so no code path is dead.
//!
//! Phase 7 relocates this file into `storage/validation/` alongside
//! any other test-only backends.
//!
//! ## Contract
//!
//! - **Reads** (`GraphRead` methods) log the method name to the
//!   wrapper's `Mutex<Vec<&'static str>>` and forward to the inner
//!   backend. Iterator-returning methods log once per call (not per
//!   yielded item). `Mutex` (not `RefCell`) so the type stays `Send`
//!   — PyO3 requires it for anything reachable from the
//!   `KnowledgeGraph` class.
//! - **Writes** (`GraphRead` methods) forward without logging. The
//!   log is a read-path audit hook; write auditing is out of scope
//!   for Phase 6 and would double the API surface for no test gain.
//! - **GAT associated types** forward to the wrapped backend's GATs.
//!   The consumer of `RecordingGraph<G>` gets the same iterator types
//!   as `G`; no boxing overhead beyond the log mutation.
//! - **`is_memory` / `is_mapped` / `is_disk`** forward to the inner
//!   backend so the 26 consumer call sites identified in the Phase 5
//!   audit keep working unchanged when wrapped.

use crate::datatypes::Value;
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Wrapper that logs read invocations on `G` while forwarding every
/// `GraphRead` / `GraphWrite` method to it. See the module docs.
///
/// The log uses a `Mutex` rather than `RefCell` so `RecordingGraph`
/// stays `Send + Sync` — `GraphBackend::Recording` has to live in the
/// `KnowledgeGraph` PyO3 class, which PyO3 requires to be `Send`
/// across `Python::detach` boundaries.
#[derive(Debug, Default)]
pub struct RecordingGraph<G: GraphRead> {
    inner: G,
    log: Mutex<Vec<&'static str>>,
}

impl<G: GraphRead> RecordingGraph<G> {
    /// Wrap `inner` in a fresh-log `RecordingGraph`.
    #[inline]
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            log: Mutex::new(Vec::new()),
        }
    }

    /// Borrow the wrapped backend.
    #[inline]
    pub fn inner(&self) -> &G {
        &self.inner
    }

    /// Snapshot of the read log so far. Test-only audit hook.
    #[inline]
    #[cfg(test)]
    pub fn log(&self) -> Vec<&'static str> {
        self.log.lock().expect("recording log poisoned").clone()
    }

    /// Number of logged reads. Test-only audit hook.
    #[inline]
    #[cfg(test)]
    pub fn log_len(&self) -> usize {
        self.log.lock().expect("recording log poisoned").len()
    }

    /// Drain the log, returning the captured entries. Test-only audit hook.
    #[inline]
    #[cfg(test)]
    pub fn drain_log(&self) -> Vec<&'static str> {
        std::mem::take(&mut *self.log.lock().expect("recording log poisoned"))
    }

    #[inline]
    fn record(&self, method: &'static str) {
        self.log
            .lock()
            .expect("recording log poisoned")
            .push(method);
    }
}

impl<G: GraphRead + Clone> Clone for RecordingGraph<G> {
    #[inline]
    fn clone(&self) -> Self {
        // Cloning starts a fresh log so copies are independently
        // auditable. Matches the intent of `KnowledgeGraph::copy()`.
        Self {
            inner: self.inner.clone(),
            log: Mutex::new(Vec::new()),
        }
    }
}

// `Serialize` forwards to the inner backend verbatim — the log is an
// in-memory audit artefact and intentionally does not persist. For
// `RecordingGraph<GraphBackend>` wrapping a `Disk` variant this lands
// on the existing Disk-serialization error path, which is the correct
// behaviour.
impl<G: GraphRead + serde::Serialize> serde::Serialize for RecordingGraph<G> {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.inner.serialize(ser)
    }
}

impl<'de, G> serde::Deserialize<'de> for RecordingGraph<G>
where
    G: GraphRead + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        G::deserialize(de).map(Self::new)
    }
}

// ─────────────────────────────────────────────────────────────────────
// GraphRead — log every call, forward to `self.inner`.
// ─────────────────────────────────────────────────────────────────────

impl<G: GraphRead> GraphRead for RecordingGraph<G> {
    type NodeIndicesIter<'a>
        = G::NodeIndicesIter<'a>
    where
        Self: 'a;
    type EdgeIndicesIter<'a>
        = G::EdgeIndicesIter<'a>
    where
        Self: 'a;
    type EdgesIter<'a>
        = G::EdgesIter<'a>
    where
        Self: 'a;
    type EdgeReferencesIter<'a>
        = G::EdgeReferencesIter<'a>
    where
        Self: 'a;
    type EdgesConnectingIter<'a>
        = G::EdgesConnectingIter<'a>
    where
        Self: 'a;
    type NeighborsIter<'a>
        = G::NeighborsIter<'a>
    where
        Self: 'a;

    #[inline]
    fn node_count(&self) -> usize {
        self.record("node_count");
        self.inner.node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.record("edge_count");
        self.inner.edge_count()
    }

    #[inline]
    fn node_bound(&self) -> usize {
        self.record("node_bound");
        self.inner.node_bound()
    }

    #[inline]
    fn is_memory(&self) -> bool {
        self.inner.is_memory()
    }

    #[inline]
    fn is_mapped(&self) -> bool {
        self.inner.is_mapped()
    }

    #[inline]
    fn is_disk(&self) -> bool {
        self.inner.is_disk()
    }

    #[inline]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        self.record("node_type_of");
        self.inner.node_type_of(idx)
    }

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        self.record("node_weight");
        self.inner.node_weight(idx)
    }

    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        self.record("get_node_property");
        self.inner.get_node_property(idx, key)
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        self.record("get_node_id");
        self.inner.get_node_id(idx)
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        self.record("get_node_title");
        self.inner.get_node_title(idx)
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        self.record("str_prop_eq");
        self.inner.str_prop_eq(idx, key, target)
    }

    #[inline]
    fn node_indices(&self) -> Self::NodeIndicesIter<'_> {
        self.record("node_indices");
        self.inner.node_indices()
    }

    #[inline]
    fn edge_indices(&self) -> Self::EdgeIndicesIter<'_> {
        self.record("edge_indices");
        self.inner.edge_indices()
    }

    #[inline]
    fn edge_references(&self) -> Self::EdgeReferencesIter<'_> {
        self.record("edge_references");
        self.inner.edge_references()
    }

    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        self.record("edge_weights");
        self.inner.edge_weights()
    }

    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> Self::EdgesIter<'_> {
        self.record("edges_directed");
        self.inner.edges_directed(idx, dir)
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> Self::EdgesIter<'_> {
        self.record("edges");
        self.inner.edges(idx)
    }

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> Self::EdgesIter<'_> {
        self.record("edges_directed_filtered");
        self.inner
            .edges_directed_filtered(idx, dir, conn_type_filter)
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> Self::EdgesConnectingIter<'_> {
        self.record("edges_connecting");
        self.inner.edges_connecting(a, b)
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        self.record("edge_weight");
        self.inner.edge_weight(idx)
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        self.record("find_edge");
        self.inner.find_edge(a, b)
    }

    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.record("edge_endpoints");
        self.inner.edge_endpoints(idx)
    }

    #[inline]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        self.record("edge_endpoint_keys");
        self.inner.edge_endpoint_keys()
    }

    #[inline]
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> Self::NeighborsIter<'_> {
        self.record("neighbors_directed");
        self.inner.neighbors_directed(idx, dir)
    }

    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> Self::NeighborsIter<'_> {
        self.record("neighbors_undirected");
        self.inner.neighbors_undirected(idx)
    }

    #[inline]
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        self.record("sources_for_conn_type_bounded");
        self.inner.sources_for_conn_type_bounded(conn_type, max)
    }

    #[inline]
    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        self.record("lookup_peer_counts");
        self.inner.lookup_peer_counts(conn_type)
    }

    #[inline]
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        self.record("lookup_by_property_eq");
        self.inner.lookup_by_property_eq(node_type, property, value)
    }

    #[inline]
    fn lookup_by_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        self.record("lookup_by_property_prefix");
        self.inner
            .lookup_by_property_prefix(node_type, property, prefix, limit)
    }

    #[inline]
    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        self.record("lookup_by_property_eq_any_type");
        self.inner.lookup_by_property_eq_any_type(property, value)
    }

    #[inline]
    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        self.record("lookup_by_property_prefix_any_type");
        self.inner
            .lookup_by_property_prefix_any_type(property, prefix, limit)
    }

    #[inline]
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        self.record("count_edges_grouped_by_peer");
        self.inner
            .count_edges_grouped_by_peer(conn_type, dir, deadline)
    }

    #[inline]
    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
    ) -> Result<usize, String> {
        self.record("count_edges_filtered");
        self.inner
            .count_edges_filtered(node, dir, conn_type, other_node_type, deadline)
    }

    #[inline]
    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        self.record("iter_peers_filtered");
        self.inner.iter_peers_filtered(node, dir, conn_type)
    }

    #[inline]
    fn reset_arenas(&self) {
        self.record("reset_arenas");
        self.inner.reset_arenas();
    }
}

// ─────────────────────────────────────────────────────────────────────
// GraphWrite — forward without logging. The log is read-side only.
// ─────────────────────────────────────────────────────────────────────

impl<G: GraphWrite> GraphWrite for RecordingGraph<G> {
    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        self.inner.node_weight_mut(idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        self.inner.edge_weight_mut(idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        self.inner.add_node(data)
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        self.inner.remove_node(idx)
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        self.inner.add_edge(a, b, data)
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.inner.remove_edge(idx)
    }

    #[inline]
    fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        self.inner.update_row_id(node_idx, row_id);
    }

    #[inline]
    fn flush_pending_writes(&mut self) {
        self.inner.flush_pending_writes();
    }
}

// ─────────────────────────────────────────────────────────────────────
// In-source parity tests — the Phase 6 "parity matrix run against
// RecordingGraph(MemoryGraph) / RecordingGraph(MappedGraph) /
// RecordingGraph(DiskGraph)" crunch-point.
//
// Exercises the GraphBackend::Recording enum dispatcher end-to-end so
// the new variant is not dead code.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::{EdgeData, GraphBackend, MappedGraph, MemoryGraph, StringInterner};
    use crate::graph::storage::disk::graph::DiskGraph;
    use std::collections::HashMap;
    use tempfile::TempDir;

    // ── fixtures ─────────────────────────────────────────────────────

    fn make_memory_backend(interner: &mut StringInterner) -> GraphBackend {
        let mut g = MemoryGraph::new();
        let a = g.add_node(NodeData::new(
            Value::UniqueId(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            {
                let mut p = HashMap::new();
                p.insert("age".to_string(), Value::Int64(30));
                p
            },
            interner,
        ));
        let b = g.add_node(NodeData::new(
            Value::UniqueId(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            HashMap::new(),
            interner,
        ));
        g.add_edge(
            a,
            b,
            EdgeData::new("KNOWS".to_string(), HashMap::new(), interner),
        );
        GraphBackend::Memory(g)
    }

    fn make_mapped_backend(interner: &mut StringInterner) -> GraphBackend {
        // Mapped backend has identical shape to Memory at this stage;
        // difference is trait-impl identity, which is what we test.
        let mut g = MappedGraph::new();
        let a = g.add_node(NodeData::new(
            Value::UniqueId(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            HashMap::new(),
            interner,
        ));
        let b = g.add_node(NodeData::new(
            Value::UniqueId(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            HashMap::new(),
            interner,
        ));
        g.add_edge(
            a,
            b,
            EdgeData::new("KNOWS".to_string(), HashMap::new(), interner),
        );
        GraphBackend::Mapped(g)
    }

    fn make_disk_backend(dir: &TempDir) -> GraphBackend {
        let dg = DiskGraph::new_at_path(dir.path()).expect("create disk graph");
        GraphBackend::Disk(Box::new(dg))
    }

    // ── helpers ──────────────────────────────────────────────────────

    fn collect_read_surface(g: &impl GraphRead) -> (usize, usize, usize) {
        let nc = g.node_count();
        let ec = g.edge_count();
        let nb = g.node_bound();
        // Iterator methods: exercise them to confirm the GAT associated
        // types line up, then discard.
        let _ = g.node_indices().count();
        let _ = g.edge_indices().count();
        let _ = g.edge_references().count();
        (nc, ec, nb)
    }

    // ── parity: log population + forwarding ──────────────────────────

    #[test]
    fn recording_logs_reads_on_memory() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);

        assert!(rg.log().is_empty());
        let _ = rg.node_count();
        let _ = rg.edge_count();
        let _ = rg.node_weight(NodeIndex::new(0));
        let log = rg.log();
        assert_eq!(log, vec!["node_count", "edge_count", "node_weight"]);
    }

    #[test]
    fn recording_logs_reads_on_mapped() {
        let mut interner = StringInterner::new();
        let backend = make_mapped_backend(&mut interner);
        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);

        let _ = rg.node_indices().count();
        let _ = rg
            .edges_directed(NodeIndex::new(0), Direction::Outgoing)
            .count();
        assert_eq!(rg.log(), vec!["node_indices", "edges_directed"]);
    }

    #[test]
    fn recording_logs_reads_on_disk() {
        let dir = TempDir::new().expect("tempdir");
        let backend = make_disk_backend(&dir);
        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);

        let _ = rg.node_count();
        let _ = rg.edge_count();
        assert_eq!(rg.log(), vec!["node_count", "edge_count"]);
    }

    // ── parity: identity vs unwrapped backend ────────────────────────

    #[test]
    fn recording_trait_parity_memory() {
        let mut a_interner = StringInterner::new();
        let backend_a = make_memory_backend(&mut a_interner);
        let mut b_interner = StringInterner::new();
        let backend_b = make_memory_backend(&mut b_interner);

        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend_b);

        assert_eq!(collect_read_surface(&backend_a), collect_read_surface(&rg));
    }

    #[test]
    fn recording_trait_parity_mapped() {
        let mut a_interner = StringInterner::new();
        let backend_a = make_mapped_backend(&mut a_interner);
        let mut b_interner = StringInterner::new();
        let backend_b = make_mapped_backend(&mut b_interner);

        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend_b);

        assert_eq!(collect_read_surface(&backend_a), collect_read_surface(&rg));
    }

    #[test]
    fn recording_trait_parity_disk() {
        let dir_a = TempDir::new().expect("tempdir");
        let dir_b = TempDir::new().expect("tempdir");
        let backend_a = make_disk_backend(&dir_a);
        let backend_b = make_disk_backend(&dir_b);

        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend_b);

        assert_eq!(collect_read_surface(&backend_a), collect_read_surface(&rg));
    }

    // ── GraphWrite passthrough ────────────────────────────────────────

    #[test]
    fn recording_write_passthrough_memory() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let n0 = backend.node_count();
        let e0 = backend.edge_count();

        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let new_node = NodeData::new(
            Value::UniqueId(3),
            Value::String("Carol".to_string()),
            "Person".to_string(),
            HashMap::new(),
            &mut interner,
        );
        let idx = rg.add_node(new_node);
        rg.add_edge(
            NodeIndex::new(0),
            idx,
            EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut interner),
        );

        assert_eq!(rg.node_count(), n0 + 1);
        assert_eq!(rg.edge_count(), e0 + 1);
    }

    // ── is_* predicates forward through the wrapper ──────────────────

    #[test]
    fn recording_is_predicates_forward() {
        let mut interner = StringInterner::new();

        let mem = RecordingGraph::new(make_memory_backend(&mut interner));
        assert!(mem.is_memory());
        assert!(!mem.is_mapped());
        assert!(!mem.is_disk());

        let mut interner2 = StringInterner::new();
        let mapped = RecordingGraph::new(make_mapped_backend(&mut interner2));
        assert!(!mapped.is_memory());
        assert!(mapped.is_mapped());
        assert!(!mapped.is_disk());

        let dir = TempDir::new().expect("tempdir");
        let disk = RecordingGraph::new(make_disk_backend(&dir));
        assert!(!disk.is_memory());
        assert!(!disk.is_mapped());
        assert!(disk.is_disk());
    }

    // ── GraphBackend::Recording variant drives the dispatcher ────────

    #[test]
    fn enum_variant_dispatches_reads_through_recording_layer() {
        let mut interner = StringInterner::new();
        let inner = make_memory_backend(&mut interner);
        let expected_nc = inner.node_count();
        let expected_ec = inner.edge_count();

        let wrapped = GraphBackend::Recording(Box::new(RecordingGraph::new(inner)));

        // Every trait call goes through:
        //   GraphBackend::Recording dispatcher arm
        //   → RecordingGraph<GraphBackend>::node_count (logs + delegates)
        //     → GraphBackend::Memory dispatcher arm
        //       → MemoryGraph::node_count
        assert_eq!(wrapped.node_count(), expected_nc);
        assert_eq!(wrapped.edge_count(), expected_ec);
        assert!(!wrapped.is_disk());
        assert!(wrapped.is_memory());

        let idx0 = NodeIndex::new(0);
        assert!(wrapped.node_weight(idx0).is_some());
        assert_eq!(
            wrapped.edges_directed(idx0, Direction::Outgoing).count(),
            1,
            "KNOWS edge should appear through the recording layer"
        );

        // Unpack the wrapper and verify it recorded all reads.
        let GraphBackend::Recording(rg) = wrapped else {
            unreachable!()
        };
        let log = rg.log();
        assert!(log.contains(&"node_count"));
        assert!(log.contains(&"edge_count"));
        assert!(log.contains(&"node_weight"));
        assert!(log.contains(&"edges_directed"));
    }

    #[test]
    fn enum_variant_round_trips_every_backend() {
        // Memory
        let mut i1 = StringInterner::new();
        let wrapped_mem =
            GraphBackend::Recording(Box::new(RecordingGraph::new(make_memory_backend(&mut i1))));
        assert!(wrapped_mem.is_memory());
        assert_eq!(wrapped_mem.node_count(), 2);

        // Mapped
        let mut i2 = StringInterner::new();
        let wrapped_mapped =
            GraphBackend::Recording(Box::new(RecordingGraph::new(make_mapped_backend(&mut i2))));
        assert!(wrapped_mapped.is_mapped());
        assert_eq!(wrapped_mapped.node_count(), 2);

        // Disk
        let dir = TempDir::new().expect("tempdir");
        let wrapped_disk =
            GraphBackend::Recording(Box::new(RecordingGraph::new(make_disk_backend(&dir))));
        assert!(wrapped_disk.is_disk());
        assert_eq!(wrapped_disk.node_count(), 0);
    }

    // ── log semantics: drain + clone ─────────────────────────────────

    #[test]
    fn drain_log_empties_the_log() {
        let mut interner = StringInterner::new();
        let rg: RecordingGraph<GraphBackend> =
            RecordingGraph::new(make_memory_backend(&mut interner));
        let _ = rg.node_count();
        let _ = rg.edge_count();
        assert_eq!(rg.log_len(), 2);
        let drained = rg.drain_log();
        assert_eq!(drained, vec!["node_count", "edge_count"]);
        assert_eq!(rg.log_len(), 0);
    }

    #[test]
    fn clone_starts_fresh_log() {
        let mut interner = StringInterner::new();
        let rg: RecordingGraph<GraphBackend> =
            RecordingGraph::new(make_memory_backend(&mut interner));
        let _ = rg.node_count();
        let rg2 = rg.clone();
        assert_eq!(rg.log_len(), 1);
        assert_eq!(rg2.log_len(), 0);
    }

    // ── Edge iterator semantics forward correctly ────────────────────

    #[test]
    fn edge_references_forward_through_recording() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let edges: Vec<_> = rg
            .edge_references()
            .map(|er| (er.source().index(), er.target().index()))
            .collect();
        assert_eq!(edges, vec![(0, 1)]);
    }
}
