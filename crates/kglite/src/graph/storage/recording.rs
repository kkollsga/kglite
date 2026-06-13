//! Write-capture backend — [`RecordingGraph`].
//!
//! Introduced in Phase 6 of the 0.8.0 storage refactor as a read-logging
//! validation wrapper; **repurposed in the Stage 1 durability work** into
//! the production write-capture seam for the WAL. `RecordingGraph<G>`
//! wraps any `G: GraphRead`/`GraphWrite`, forwards every call, and — on
//! the six mutation methods only — buffers a [`RawOp`] describing the
//! change. Reads forward with **zero overhead** (no logging), so a
//! durable graph pays the wrapper cost only on writes.
//!
//! It drives the [`crate::graph::schema::GraphBackend::Recording`] enum
//! variant: a durable graph's backend is
//! `Recording(Box<RecordingGraph<GraphBackend>>)`, so every `GraphWrite`
//! call from the Cypher executor, the fluent/batch mutation paths, and
//! bulk load funnels through this one seam — validated against the call
//! graph (no path mutates the inner `StableDiGraph` around the trait).
//!
//! ## Why raw ops, resolved later
//!
//! The backend stores **interned** node-type / property keys; resolving
//! them to strings needs the `DirGraph`'s `StringInterner`, which the
//! backend does not own. So writes buffer *raw* ops keyed by
//! `NodeIndex`/`EdgeIndex` + `InternedKey`, and [`resolve_ops`] — run at
//! flush, where the interner is in scope — turns them into the
//! string-keyed, identity-keyed [`crate::graph::wal::MutationOp`]s the
//! WAL persists. Upserts are captured as a placeholder index and
//! resolved against the *final* post-batch node/edge state (so an
//! add-then-SET collapses to one upsert; an add-then-remove drops the
//! upsert and keeps the remove). Removes capture their logical
//! `(type, id)` *before* the entry vanishes.
//!
//! ## `Send + Sync` without a `Mutex`
//!
//! All six mutation methods take `&mut self`, and reads no longer record,
//! so the op buffer is a plain `Vec<RawOp>` mutated only through `&mut`.
//! That keeps `RecordingGraph` `Send + Sync` (required for the PyO3
//! `KnowledgeGraph` class) with no lock on the hot path.

use crate::datatypes::Value;
use crate::graph::schema::{EdgeData, InternedKey, NodeData, StringInterner};
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::wal::MutationOp;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use std::collections::HashMap;
use std::time::Instant;

/// A buffered, unresolved mutation. Keyed by petgraph index (for
/// upserts, resolved against the final graph state at flush) or by the
/// pre-removal logical identity (for removes, since the entry is gone by
/// flush time). Turned into a [`MutationOp`] by [`resolve_ops`].
#[derive(Debug, Clone, PartialEq)]
pub enum RawOp {
    /// A node was added or property-mutated. Resolve its full final
    /// state at flush; drop if the node was later removed in the batch.
    UpsertNode(NodeIndex),
    /// A node was removed. Its logical identity, captured before removal.
    RemoveNode { node_type: InternedKey, id: Value },
    /// An edge was added or property-mutated. Resolve at flush.
    UpsertEdge(EdgeIndex),
    /// An edge was removed. Logical identity captured before removal.
    RemoveEdge {
        conn_type: InternedKey,
        src_type: InternedKey,
        src_id: Value,
        tgt_type: InternedKey,
        tgt_id: Value,
    },
}

/// Wrapper that captures write invocations on `G` as [`RawOp`]s while
/// forwarding every `GraphRead`/`GraphWrite` method to it. See the
/// module docs.
#[derive(Debug, Default)]
pub struct RecordingGraph<G: GraphRead> {
    inner: G,
    ops: Vec<RawOp>,
}

impl<G: GraphRead> RecordingGraph<G> {
    /// Wrap `inner` in a fresh-buffer `RecordingGraph`.
    #[inline]
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            ops: Vec::new(),
        }
    }

    /// Borrow the wrapped backend.
    #[inline]
    pub fn inner(&self) -> &G {
        &self.inner
    }

    /// Mutable borrow of the wrapped backend (for mode-switch / teardown
    /// paths that need the raw inner without recording).
    #[inline]
    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.inner
    }

    /// Drain the buffered raw ops, leaving the buffer empty. Called at
    /// each commit/flush before [`resolve_ops`].
    #[inline]
    pub fn take_ops(&mut self) -> Vec<RawOp> {
        std::mem::take(&mut self.ops)
    }

    /// Number of buffered (undrained) raw ops.
    #[inline]
    pub fn ops_len(&self) -> usize {
        self.ops.len()
    }
}

impl<G: GraphRead + Clone> Clone for RecordingGraph<G> {
    #[inline]
    fn clone(&self) -> Self {
        // A clone starts with an empty op buffer. In the CoW transaction
        // model the buffer is always drained after each mutation, so it
        // is empty at clone time anyway; resetting makes that an
        // invariant the clone can rely on rather than a coincidence.
        Self {
            inner: self.inner.clone(),
            ops: Vec::new(),
        }
    }
}

/// Resolve buffered [`RawOp`]s into string-keyed, identity-keyed
/// [`MutationOp`]s, reading final node/edge state from `graph` and
/// resolving interned keys through `interner`. Upserts whose node/edge
/// no longer exists (removed later in the same batch) are dropped — the
/// corresponding remove op already captures the final state.
pub fn resolve_ops(
    raw: &[RawOp],
    graph: &impl GraphRead,
    interner: &StringInterner,
) -> Vec<MutationOp> {
    let mut out = Vec::with_capacity(raw.len());
    for op in raw {
        match op {
            RawOp::UpsertNode(idx) => {
                if let Some(nd) = graph.node_weight(*idx) {
                    out.push(MutationOp::UpsertNode {
                        node_type: nd.node_type_str(interner).to_string(),
                        id: nd.id().into_owned(),
                        title: nd.title().into_owned(),
                        properties: nd.properties_cloned(interner).into_iter().collect(),
                    });
                }
            }
            RawOp::RemoveNode { node_type, id } => {
                out.push(MutationOp::RemoveNode {
                    node_type: interner.resolve(*node_type).to_string(),
                    id: id.clone(),
                });
            }
            RawOp::UpsertEdge(eidx) => {
                if let (Some((a, b)), Some(ed)) =
                    (graph.edge_endpoints(*eidx), graph.edge_weight(*eidx))
                {
                    if let (Some(src), Some(tgt)) = (
                        logical_node(graph, a, interner),
                        logical_node(graph, b, interner),
                    ) {
                        out.push(MutationOp::UpsertEdge {
                            conn_type: ed.connection_type_str(interner).to_string(),
                            src_type: src.0,
                            src_id: src.1,
                            tgt_type: tgt.0,
                            tgt_id: tgt.1,
                            properties: ed.properties_cloned(interner).into_iter().collect(),
                        });
                    }
                }
            }
            RawOp::RemoveEdge {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
            } => {
                out.push(MutationOp::RemoveEdge {
                    conn_type: interner.resolve(*conn_type).to_string(),
                    src_type: interner.resolve(*src_type).to_string(),
                    src_id: src_id.clone(),
                    tgt_type: interner.resolve(*tgt_type).to_string(),
                    tgt_id: tgt_id.clone(),
                });
            }
        }
    }
    out
}

/// Resolve a node index to its logical `(node_type, id)`, or `None` if
/// the node is gone.
fn logical_node(
    graph: &impl GraphRead,
    idx: NodeIndex,
    interner: &StringInterner,
) -> Option<(String, Value)> {
    let nd = graph.node_weight(idx)?;
    Some((nd.node_type_str(interner).to_string(), nd.id().into_owned()))
}

// `Serialize` forwards to the inner backend verbatim — the op buffer is
// transient capture state and intentionally does not persist. For
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
        self.inner.node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    #[inline]
    fn node_bound(&self) -> usize {
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
        self.inner.node_type_of(idx)
    }

    #[inline]
    fn node_labels_of(&self, idx: NodeIndex) -> Vec<InternedKey> {
        self.inner.node_labels_of(idx)
    }

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        self.inner.node_weight(idx)
    }

    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        self.inner.get_node_property(idx, key)
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        self.inner.get_node_id(idx)
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        self.inner.get_node_title(idx)
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        self.inner.str_prop_eq(idx, key, target)
    }

    #[inline]
    fn node_indices(&self) -> Self::NodeIndicesIter<'_> {
        self.inner.node_indices()
    }

    #[inline]
    fn edge_indices(&self) -> Self::EdgeIndicesIter<'_> {
        self.inner.edge_indices()
    }

    #[inline]
    fn edge_references(&self) -> Self::EdgeReferencesIter<'_> {
        self.inner.edge_references()
    }

    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        self.inner.edge_weights()
    }

    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> Self::EdgesIter<'_> {
        self.inner.edges_directed(idx, dir)
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> Self::EdgesIter<'_> {
        self.inner.edges(idx)
    }

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> Self::EdgesIter<'_> {
        self.inner
            .edges_directed_filtered(idx, dir, conn_type_filter)
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> Self::EdgesConnectingIter<'_> {
        self.inner.edges_connecting(a, b)
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        self.inner.edge_weight(idx)
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        self.inner.find_edge(a, b)
    }

    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.inner.edge_endpoints(idx)
    }

    #[inline]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        self.inner.edge_endpoint_keys()
    }

    #[inline]
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> Self::NeighborsIter<'_> {
        self.inner.neighbors_directed(idx, dir)
    }

    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> Self::NeighborsIter<'_> {
        self.inner.neighbors_undirected(idx)
    }

    #[inline]
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        self.inner.sources_for_conn_type_bounded(conn_type, max)
    }

    #[inline]
    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        self.inner.lookup_peer_counts(conn_type)
    }

    #[inline]
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
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
        self.inner
            .lookup_by_property_prefix(node_type, property, prefix, limit)
    }

    #[inline]
    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        self.inner.lookup_by_property_eq_any_type(property, value)
    }

    #[inline]
    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
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
        self.inner.iter_peers_filtered(node, dir, conn_type)
    }

    #[inline]
    fn reset_arenas(&self) {
        self.inner.reset_arenas();
    }
}

// ─────────────────────────────────────────────────────────────────────
// GraphWrite — forward to the inner backend AND buffer a RawOp. This is
// the WAL capture seam; see the module docs.
// ─────────────────────────────────────────────────────────────────────

impl<G: GraphWrite> GraphWrite for RecordingGraph<G> {
    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        // The caller mutates the returned &mut after this returns, so we
        // can't see the change here — record a placeholder and resolve the
        // node's final state at flush. The existence check uses an
        // immutable read that ends before the push (the returned mut borrow
        // would otherwise hold all of `self`). Only record when the node
        // exists (a None borrow changes nothing).
        if self.inner.node_weight(idx).is_some() {
            self.ops.push(RawOp::UpsertNode(idx));
        }
        self.inner.node_weight_mut(idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        if self.inner.edge_weight(idx).is_some() {
            self.ops.push(RawOp::UpsertEdge(idx));
        }
        self.inner.edge_weight_mut(idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        let idx = self.inner.add_node(data);
        self.ops.push(RawOp::UpsertNode(idx));
        idx
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        // Capture the logical identity before the node vanishes.
        let identity = self
            .inner
            .node_type_of(idx)
            .zip(self.inner.get_node_id(idx));
        let removed = self.inner.remove_node(idx);
        if removed.is_some() {
            if let Some((node_type, id)) = identity {
                self.ops.push(RawOp::RemoveNode { node_type, id });
            }
        }
        removed
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        let eidx = self.inner.add_edge(a, b, data);
        self.ops.push(RawOp::UpsertEdge(eidx));
        eidx
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        // Capture conn type + both endpoints' logical identity before the
        // edge vanishes.
        let identity = self.inner.edge_endpoints(idx).and_then(|(a, b)| {
            let conn_type = self.inner.edge_weight(idx)?.connection_type;
            let (src_type, src_id) = (self.inner.node_type_of(a)?, self.inner.get_node_id(a)?);
            let (tgt_type, tgt_id) = (self.inner.node_type_of(b)?, self.inner.get_node_id(b)?);
            Some((conn_type, src_type, src_id, tgt_type, tgt_id))
        });
        let removed = self.inner.remove_edge(idx);
        if removed.is_some() {
            if let Some((conn_type, src_type, src_id, tgt_type, tgt_id)) = identity {
                self.ops.push(RawOp::RemoveEdge {
                    conn_type,
                    src_type,
                    src_id,
                    tgt_type,
                    tgt_id,
                });
            }
        }
        removed
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

    // ── write capture + resolution ───────────────────────────────────

    #[test]
    fn reads_do_not_capture() {
        let mut interner = StringInterner::new();
        let rg: RecordingGraph<GraphBackend> =
            RecordingGraph::new(make_memory_backend(&mut interner));
        let _ = rg.node_count();
        let _ = rg.edge_count();
        let _ = rg.node_weight(NodeIndex::new(0));
        let _ = rg
            .edges_directed(NodeIndex::new(0), Direction::Outgoing)
            .count();
        assert_eq!(rg.ops_len(), 0, "reads must not buffer any ops");
    }

    #[test]
    fn captures_add_node_and_edge_as_upserts() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        let a = rg.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".into()),
            "Person".into(),
            HashMap::from([("age".to_string(), Value::Int64(30))]),
            &mut interner,
        ));
        let b = rg.add_node(NodeData::new(
            Value::Int64(2),
            Value::String("Bob".into()),
            "Person".into(),
            HashMap::new(),
            &mut interner,
        ));
        rg.add_edge(
            a,
            b,
            EdgeData::new("KNOWS".into(), HashMap::new(), &mut interner),
        );

        let raw = rg.take_ops();
        assert_eq!(rg.ops_len(), 0, "take_ops empties the buffer");
        let ops = resolve_ops(&raw, &rg, &interner);
        assert_eq!(
            ops,
            vec![
                MutationOp::UpsertNode {
                    node_type: "Person".into(),
                    id: Value::Int64(1),
                    title: Value::String("Alice".into()),
                    properties: vec![("age".into(), Value::Int64(30))],
                },
                MutationOp::UpsertNode {
                    node_type: "Person".into(),
                    id: Value::Int64(2),
                    title: Value::String("Bob".into()),
                    properties: vec![],
                },
                MutationOp::UpsertEdge {
                    conn_type: "KNOWS".into(),
                    src_type: "Person".into(),
                    src_id: Value::Int64(1),
                    tgt_type: "Person".into(),
                    tgt_id: Value::Int64(2),
                    properties: vec![],
                },
            ]
        );
    }

    #[test]
    fn captures_set_as_node_upsert_with_final_state() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        let a = rg.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".into()),
            "Person".into(),
            HashMap::from([("age".to_string(), Value::Int64(30))]),
            &mut interner,
        ));
        let _ = rg.take_ops(); // drain the add
                               // SET age = 41 via the mutable-borrow path.
        if let Some(nd) = rg.node_weight_mut(a) {
            nd.set_property("age", Value::Int64(41), &mut interner);
        }
        let raw = rg.take_ops();
        let ops = resolve_ops(&raw, &rg, &interner);
        // Resolves to the node's FINAL state (age = 41), not a delta.
        assert_eq!(
            ops,
            vec![MutationOp::UpsertNode {
                node_type: "Person".into(),
                id: Value::Int64(1),
                title: Value::String("Alice".into()),
                properties: vec![("age".into(), Value::Int64(41))],
            }]
        );
    }

    #[test]
    fn captures_remove_node_by_logical_identity() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let removed = rg.remove_node(NodeIndex::new(0));
        assert!(removed.is_some());
        let raw = rg.take_ops();
        let ops = resolve_ops(&raw, &rg, &interner);
        assert_eq!(
            ops,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::UniqueId(1),
            }]
        );
    }

    #[test]
    fn captures_remove_edge_by_logical_identity() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let removed = rg.remove_edge(EdgeIndex::new(0));
        assert!(removed.is_some());
        let raw = rg.take_ops();
        let ops = resolve_ops(&raw, &rg, &interner);
        assert_eq!(
            ops,
            vec![MutationOp::RemoveEdge {
                conn_type: "KNOWS".into(),
                src_type: "Person".into(),
                src_id: Value::UniqueId(1),
                tgt_type: "Person".into(),
                tgt_id: Value::UniqueId(2),
            }]
        );
    }

    #[test]
    fn add_then_remove_in_batch_drops_the_upsert() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        let a = rg.add_node(NodeData::new(
            Value::Int64(7),
            Value::String("Ghost".into()),
            "Person".into(),
            HashMap::new(),
            &mut interner,
        ));
        rg.remove_node(a);
        let raw = rg.take_ops();
        let ops = resolve_ops(&raw, &rg, &interner);
        // The UpsertNode placeholder resolves to None (node gone); only
        // the RemoveNode survives — replay reaches the right final state.
        assert_eq!(
            ops,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::Int64(7),
            }]
        );
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

        // Reads through the enum dispatcher capture nothing.
        let GraphBackend::Recording(rg) = wrapped else {
            unreachable!()
        };
        assert_eq!(
            rg.ops_len(),
            0,
            "reads through the dispatcher must not capture"
        );
    }

    #[test]
    fn enum_variant_captures_writes_through_dispatcher() {
        let mut interner = StringInterner::new();
        let mut wrapped =
            GraphBackend::Recording(Box::new(RecordingGraph::new(GraphBackend::new())));
        // A write through the enum dispatcher reaches the recording layer.
        wrapped.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".into()),
            "Person".into(),
            HashMap::new(),
            &mut interner,
        ));
        let GraphBackend::Recording(rg) = &mut wrapped else {
            unreachable!()
        };
        assert_eq!(
            rg.ops_len(),
            1,
            "add_node through the dispatcher is captured"
        );
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

    // ── buffer semantics: take + clone ───────────────────────────────

    #[test]
    fn take_ops_empties_the_buffer() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        rg.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("A".into()),
            "T".into(),
            HashMap::new(),
            &mut interner,
        ));
        assert_eq!(rg.ops_len(), 1);
        let drained = rg.take_ops();
        assert_eq!(drained.len(), 1);
        assert_eq!(rg.ops_len(), 0);
    }

    #[test]
    fn clone_starts_with_empty_op_buffer() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        rg.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("A".into()),
            "T".into(),
            HashMap::new(),
            &mut interner,
        ));
        let rg2 = rg.clone();
        assert_eq!(rg.ops_len(), 1);
        assert_eq!(rg2.ops_len(), 0, "a clone starts with a fresh op buffer");
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
