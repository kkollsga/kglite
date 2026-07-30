//! Per-backend [`GraphRead`] / [`GraphWrite`] implementations.
//!
//! Phase 5 landed this file. Prior phases routed every trait method
//! through a monolithic `impl GraphRead for GraphBackend` in
//! `schema.rs`; here each backend (`MemoryGraph`, `MappedGraph`,
//! `DiskGraph`) owns its own trait impls so the backends can diverge
//! without re-touching the enum dispatcher. The `impl GraphRead for
//! GraphBackend` that survives in `schema.rs` is now a thin 3-arm
//! dispatcher delegating to the per-backend impls below.
//!
//! Phase 7 relocates these impls into `storage/memory/`,
//! `storage/mapped/`, `storage/disk/` subdirectories. This Phase 5
//! single-file layout keeps the diff cohesive without pre-empting
//! Phase 7's structural reorg.

use crate::datatypes::Value;
use crate::graph::core::iterators::{
    GraphEdgeIndices, GraphEdgeReferences, GraphEdges, GraphEdgesConnecting, GraphNeighbors,
    GraphNodeIndices,
};
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::disk::csr::TOMBSTONE_EDGE;
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::{GraphRead, GraphWrite, MappedGraph, MemoryGraph};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};
use petgraph::Direction;
use std::collections::HashMap;
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// Heap-backed GraphRead / GraphWrite impls
//
// `MemoryGraph` and `MappedGraph` wrap identical `StableDiGraph` today but
// carry distinct type identity for per-backend divergence. The
// `impl_heap_graph_read!` macro emits the shared read body. The write side is
// written out per backend: both carry the statement-scoped undo journal, but
// `MappedGraph` also maintains its own lazy type/property indexes and
// `MemoryGraph` its peer counts, so the two have nothing left in common to
// share.
// ──────────────────────────────────────────────────────────────────────────

macro_rules! impl_heap_graph_read {
    ($ty:ty, is_memory = $is_memory:expr, is_mapped = $is_mapped:expr) => {
        impl GraphRead for $ty {
            type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
            type EdgeIndicesIter<'a> = GraphEdgeIndices<'a>;
            type EdgesIter<'a> = GraphEdges<'a>;
            type EdgeReferencesIter<'a> = GraphEdgeReferences<'a>;
            type EdgesConnectingIter<'a> = GraphEdgesConnecting<'a>;
            type NeighborsIter<'a> = GraphNeighbors<'a>;

            #[inline]
            fn node_count(&self) -> usize {
                self.inner().node_count()
            }

            #[inline]
            fn edge_count(&self) -> usize {
                self.inner().edge_count()
            }

            #[inline]
            fn node_bound(&self) -> usize {
                self.inner().node_bound()
            }

            #[inline]
            fn is_memory(&self) -> bool {
                $is_memory
            }

            #[inline]
            fn is_mapped(&self) -> bool {
                $is_mapped
            }

            #[inline]
            fn is_disk(&self) -> bool {
                false
            }

            #[inline]
            fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
                self.inner().node_weight(idx).map(|nd| nd.node_type)
            }

            // NodeData no longer carries extra_labels — secondary
            // labels live in `DirGraph.secondary_label_index` (the
            // canonical store). Backend `node_labels_of` returns only
            // the primary; callers that need the full label list go
            // through `DirGraph::node_labels`, which has access to the
            // inverted index.

            #[inline]
            fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
                self.inner().node_weight(idx)
            }

            #[inline]
            fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
                self.inner()
                    .node_weight(idx)
                    .and_then(|nd| nd.properties.get_value(key))
            }

            #[inline]
            fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
                self.inner().node_weight(idx).map(|nd| nd.id().into_owned())
            }

            #[inline]
            fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
                self.inner()
                    .node_weight(idx)
                    .map(|nd| nd.title().into_owned())
            }

            #[inline]
            fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
                self.inner()
                    .node_weight(idx)
                    .and_then(|nd| nd.properties.str_prop_eq(key, target))
            }

            #[inline]
            fn node_indices(&self) -> GraphNodeIndices<'_> {
                GraphNodeIndices::InMemory(self.inner().node_indices())
            }

            #[inline]
            fn edge_indices(&self) -> GraphEdgeIndices<'_> {
                GraphEdgeIndices::InMemory(self.inner().edge_indices())
            }

            #[inline]
            fn edge_references(&self) -> GraphEdgeReferences<'_> {
                GraphEdgeReferences::InMemory(self.inner().edge_references())
            }

            #[inline]
            fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
                Box::new(self.inner().edge_weights())
            }

            #[inline]
            fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> GraphEdges<'_> {
                GraphEdges::InMemory(self.inner().edges_directed(idx, dir))
            }

            #[inline]
            fn edges(&self, idx: NodeIndex) -> GraphEdges<'_> {
                GraphEdges::InMemory(self.inner().edges(idx))
            }

            #[inline]
            fn edges_directed_filtered(
                &self,
                idx: NodeIndex,
                dir: Direction,
                _conn_type_filter: Option<InternedKey>,
            ) -> GraphEdges<'_> {
                // Heap backends don't have a pre-filter fast path; callers
                // still post-filter on `connection_type`.
                GraphEdges::InMemory(self.inner().edges_directed(idx, dir))
            }

            #[inline]
            fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> GraphEdgesConnecting<'_> {
                GraphEdgesConnecting::InMemory(self.inner().edges_connecting(a, b))
            }

            #[inline]
            fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
                self.inner().edge_weight(idx)
            }

            #[inline]
            fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
                self.inner().find_edge(a, b)
            }

            #[inline]
            fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
                self.inner().edge_endpoints(idx)
            }

            #[inline(always)]
            fn edge_endpoint_keys<'a>(
                &'a self,
            ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
                Box::new(self.inner().edge_references().map(|er| {
                    let w = er.weight();
                    (er.source(), er.target(), w.connection_type)
                }))
            }

            #[inline]
            fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> GraphNeighbors<'_> {
                GraphNeighbors::InMemory(self.inner().neighbors_directed(idx, dir))
            }

            #[inline]
            fn neighbors_undirected(&self, idx: NodeIndex) -> GraphNeighbors<'_> {
                GraphNeighbors::InMemory(self.inner().neighbors_undirected(idx))
            }

            fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
                let counts = self.ensure_peer_counts(conn_type);
                Some((*counts.by_target).clone())
            }

            fn count_edges_grouped_by_peer(
                &self,
                conn_type: InternedKey,
                dir: Direction,
                _deadline: Option<Instant>,
            ) -> Result<HashMap<u32, i64>, String> {
                let counts = self.ensure_peer_counts(conn_type);
                let selected = match dir {
                    Direction::Outgoing => &counts.by_target,
                    Direction::Incoming => &counts.by_source,
                };
                Ok((**selected).clone())
            }

            fn count_edges_filtered(
                &self,
                node: NodeIndex,
                dir: Direction,
                conn_type: Option<InternedKey>,
                other_node_type: Option<InternedKey>,
                deadline: Option<Instant>,
            ) -> Result<usize, String> {
                let g = self.inner();
                let mut count = 0;
                for (i, edge) in g.edges_directed(node, dir).enumerate() {
                    if i.is_multiple_of(1 << 20) {
                        if let Some(dl) = deadline {
                            if Instant::now() > dl {
                                return Err("Query timed out".to_string());
                            }
                        }
                    }
                    if let Some(ct) = conn_type {
                        if edge.weight().connection_type != ct {
                            continue;
                        }
                    }
                    let other = if dir == Direction::Outgoing {
                        edge.target()
                    } else {
                        edge.source()
                    };
                    if let Some(required_type) = other_node_type {
                        if let Some(nd) = g.node_weight(other) {
                            if nd.node_type != required_type {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                    count += 1;
                }
                Ok(count)
            }

            // iter_peers_filtered / reset_arenas — trait defaults.
        }
    };
}

impl_heap_graph_read!(MemoryGraph, is_memory = true, is_mapped = false);

// ──────────────────────────────────────────────────────────────────────────
// MemoryGraph — GraphWrite, and the statement-scoped undo capture seam.
//
// Written out rather than macro-generated: `MappedGraph` has its own
// hand-written `GraphWrite` (below) and only `MemoryGraph` carries an undo
// journal, so a shared macro had exactly one user and no longer earned the
// indirection.
//
// Every method follows the same shape: capture the inverse of the edit into
// the journal *if one is installed*, then perform the edit. With no journal
// (the steady state, and every read path) the added cost is one `Option`
// discriminant check; the clone-the-pre-image work lives in `#[cold]`
// helpers so it never bloats the inlined hot path.
//
// See `storage/undo.rs` for the replay contract and
// `dir_graph/rollback.rs` for the restore half.
// ──────────────────────────────────────────────────────────────────────────

impl MemoryGraph {
    /// Clone `idx`'s current weight into the journal as its pre-statement
    /// state, unless this statement already captured it.
    #[cold]
    fn capture_node_weight(&mut self, idx: NodeIndex) {
        let inner = &self.inner;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_weight(idx, || inner.node_weight(idx).cloned());
        }
    }

    /// Edge counterpart of [`Self::capture_node_weight`].
    #[cold]
    fn capture_edge_weight(&mut self, idx: EdgeIndex) {
        let inner = &self.inner;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_weight(idx, || inner.edge_weight(idx).cloned());
        }
    }

    /// Any edge incident to `idx`, in either direction.
    fn first_incident_edge(&self, idx: NodeIndex) -> Option<EdgeIndex> {
        self.inner
            .edges_directed(idx, Direction::Outgoing)
            .next()
            .or_else(|| self.inner.edges_directed(idx, Direction::Incoming).next())
            .map(|edge| edge.id())
    }

    /// Detach `idx`'s edges one at a time, through the recorded
    /// [`GraphWrite::remove_edge`], so each gets its own journal entry.
    ///
    /// petgraph's `remove_node` drops incident edges in one internal sweep
    /// whose order we neither observe nor control, which would leave reverse
    /// replay unable to hand each edge back its own free-list slot. Doing the
    /// detach ourselves makes the removal order the recorded order. The end
    /// state is identical either way; the Cypher delete path already detaches
    /// explicitly (`mutation::maintain::detach_delete_nodes`), so in practice
    /// this loop finds nothing to do and exists for every *other* caller of
    /// `remove_node`.
    #[cold]
    fn detach_for_journal(&mut self, idx: NodeIndex) {
        while let Some(edge) = self.first_incident_edge(idx) {
            GraphWrite::remove_edge(self, edge);
        }
    }
}

impl GraphWrite for MemoryGraph {
    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        if self.undo.is_some() {
            self.capture_node_weight(idx);
        }
        self.inner.node_weight_mut(idx)
    }

    /// Bypasses the undo journal as well as the WAL recorder.
    ///
    /// The one caller is the columnar handle-refresh sweep, which re-points
    /// every node of a type at a forked master store. Capturing a `NodeData`
    /// pre-image for each of them would make a one-row `SET` cost one clone
    /// per node *of the type* — the O(V+E)-per-write cost this journal exists
    /// to remove, reintroduced at a smaller constant. The sweep's inverse is
    /// journalled once per type instead, as
    /// [`UndoEntry::ColumnarHandles`](crate::graph::storage::undo::UndoEntry::ColumnarHandles).
    ///
    /// Nothing else may use this method to skip capture: a caller that
    /// changes a node's *content* silently is unrecoverable on rollback.
    #[inline]
    fn node_weight_mut_silent(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        self.inner.node_weight_mut(idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        self.invalidate_peer_counts();
        if self.undo.is_some() {
            self.capture_edge_weight(idx);
        }
        self.inner.edge_weight_mut(idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        let node_type = data.node_type;
        let idx = self.inner.add_node(data);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_added(idx, node_type);
        }
        idx
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        self.invalidate_peer_counts();
        if self.undo.is_some() {
            self.detach_for_journal(idx);
        }
        let removed = self.inner.remove_node(idx)?;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_removed(idx, removed.clone());
        }
        Some(removed)
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        self.invalidate_peer_counts();
        let idx = self.inner.add_edge(a, b, data);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_added(idx);
        }
        idx
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.invalidate_peer_counts();
        let endpoints = self.undo.is_some().then(|| self.inner.edge_endpoints(idx));
        let removed = self.inner.remove_edge(idx)?;
        if let Some(journal) = self.undo.as_deref_mut() {
            if let Some(Some((src, tgt))) = endpoints {
                journal.note_edge_removed(idx, src, tgt, removed.clone());
            }
        }
        Some(removed)
    }

    // update_row_id — trait default no-op (disk-only).
}

// ──────────────────────────────────────────────────────────────────────────
// MappedGraph — hand-written GraphRead impl with a lazy per-conn-type
// index. Delegates most methods to `self.inner` (identical to the macro
// body) and overrides `edges_directed_filtered`,
// `sources_for_conn_type_bounded`, `lookup_peer_counts`, and
// `count_edges_grouped_by_peer` to consult the index. Disk already has
// these structures as persistent mmap; for mapped we rebuild them in
// RAM on first query.
// ──────────────────────────────────────────────────────────────────────────

impl GraphRead for MappedGraph {
    type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = GraphEdgeIndices<'a>;
    type EdgesIter<'a> = GraphEdges<'a>;
    type EdgeReferencesIter<'a> = GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = GraphNeighbors<'a>;

    #[inline]
    fn node_count(&self) -> usize {
        self.inner().node_count()
    }
    #[inline]
    fn edge_count(&self) -> usize {
        self.inner().edge_count()
    }
    #[inline]
    fn node_bound(&self) -> usize {
        self.inner().node_bound()
    }
    #[inline]
    fn is_memory(&self) -> bool {
        false
    }
    #[inline]
    fn is_mapped(&self) -> bool {
        true
    }
    #[inline]
    fn is_disk(&self) -> bool {
        false
    }
    #[inline]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        self.inner().node_weight(idx).map(|nd| nd.node_type)
    }
    // node_labels_of falls back to the default trait impl (returns the
    // primary type only). The full label list comes from
    // `DirGraph::node_labels`, which consults `secondary_label_index`.

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        self.inner().node_weight(idx)
    }
    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        self.inner()
            .node_weight(idx)
            .and_then(|nd| nd.properties.get_value(key))
    }
    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        self.inner().node_weight(idx).map(|nd| nd.id().into_owned())
    }
    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        self.inner()
            .node_weight(idx)
            .map(|nd| nd.title().into_owned())
    }
    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        self.inner()
            .node_weight(idx)
            .and_then(|nd| nd.properties.str_prop_eq(key, target))
    }
    #[inline]
    fn node_indices(&self) -> GraphNodeIndices<'_> {
        GraphNodeIndices::InMemory(self.inner().node_indices())
    }
    #[inline]
    fn edge_indices(&self) -> GraphEdgeIndices<'_> {
        GraphEdgeIndices::InMemory(self.inner().edge_indices())
    }
    #[inline]
    fn edge_references(&self) -> GraphEdgeReferences<'_> {
        GraphEdgeReferences::InMemory(self.inner().edge_references())
    }
    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        Box::new(self.inner().edge_weights())
    }
    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> GraphEdges<'_> {
        GraphEdges::InMemory(self.inner().edges_directed(idx, dir))
    }
    #[inline]
    fn edges(&self, idx: NodeIndex) -> GraphEdges<'_> {
        GraphEdges::InMemory(self.inner().edges(idx))
    }

    // Per-node typed edge iteration stays on the bare petgraph scan.
    // The index's lookup overhead (RwLock read + HashMap + Arc clone +
    // binary_search) is ~100 ns per call, which dominates when the
    // caller only needs to check for a single edge's presence. Bulk
    // queries go through `sources_for_conn_type_bounded` /
    // `lookup_peer_counts` / `count_edges_grouped_by_peer` overrides
    // below, which amortise the index cost over many answered
    // questions. Callers still post-filter on `connection_type`.
    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        _conn_type_filter: Option<InternedKey>,
    ) -> GraphEdges<'_> {
        GraphEdges::InMemory(self.inner().edges_directed(idx, dir))
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> GraphEdgesConnecting<'_> {
        GraphEdgesConnecting::InMemory(self.inner().edges_connecting(a, b))
    }
    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        self.inner().edge_weight(idx)
    }
    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        self.inner().find_edge(a, b)
    }
    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.inner().edge_endpoints(idx)
    }
    #[inline(always)]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        Box::new(self.inner().edge_references().map(|er| {
            let w = er.weight();
            (er.source(), er.target(), w.connection_type)
        }))
    }
    #[inline]
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> GraphNeighbors<'_> {
        GraphNeighbors::InMemory(self.inner().neighbors_directed(idx, dir))
    }
    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> GraphNeighbors<'_> {
        GraphNeighbors::InMemory(self.inner().neighbors_undirected(idx))
    }

    // ─── OVERRIDE: bounded source list comes from the index's out_sources ─
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        let block = self.ensure_type_index(conn_type);
        let slice = match max {
            Some(n) => &block.out_sources[..n.min(block.out_sources.len())],
            None => &block.out_sources[..],
        };
        Some(slice.iter().map(|n| n.index() as u32).collect())
    }

    // ─── OVERRIDE: peer-count histogram lookup ────────────────────────────
    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        let block = self.ensure_type_index(conn_type);
        // Callers use Outgoing semantics (peer = target); the disk helper
        // does the same. Incoming-dir callers go through
        // `count_edges_grouped_by_peer` instead.
        Some(
            block
                .out_peer_counts
                .iter()
                .map(|(n, c)| (n.index() as u32, *c))
                .collect(),
        )
    }

    // ─── OVERRIDE: O(distinct-peers) via the index instead of full scan ──
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        _deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        let block = self.ensure_type_index(conn_type);
        let peer_counts = match dir {
            Direction::Outgoing => &block.out_peer_counts,
            Direction::Incoming => &block.in_peer_counts,
        };
        Ok(peer_counts
            .iter()
            .map(|(n, c)| (n.index() as u32, *c))
            .collect())
    }

    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
    ) -> Result<usize, String> {
        // When conn_type is given, use the index to skip non-matching
        // edges entirely. When absent, fall back to the heap scan used
        // by the macro impl.
        let g = self.inner();
        let mut count = 0usize;
        if let Some(ct) = conn_type {
            let block = self.ensure_type_index(ct);
            let (sources, offsets, edges) = match dir {
                Direction::Outgoing => (&block.out_sources, &block.out_offsets, &block.out_edges),
                Direction::Incoming => (&block.in_sources, &block.in_offsets, &block.in_edges),
            };
            if let Ok(pos) = sources.binary_search_by_key(&node.index(), |n| n.index()) {
                let start = offsets[pos] as usize;
                let end = offsets[pos + 1] as usize;
                for (i, &ei) in edges[start..end].iter().enumerate() {
                    if i.is_multiple_of(1 << 20) {
                        if let Some(dl) = deadline {
                            if Instant::now() > dl {
                                return Err("Query timed out".to_string());
                            }
                        }
                    }
                    if let Some(required_type) = other_node_type {
                        let Some((src, tgt)) = g.edge_endpoints(ei) else {
                            continue;
                        };
                        let other = if dir == Direction::Outgoing { tgt } else { src };
                        if let Some(nd) = g.node_weight(other) {
                            if nd.node_type != required_type {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                    count += 1;
                }
            }
            return Ok(count);
        }
        for (i, edge) in g.edges_directed(node, dir).enumerate() {
            if i.is_multiple_of(1 << 20) {
                if let Some(dl) = deadline {
                    if Instant::now() > dl {
                        return Err("Query timed out".to_string());
                    }
                }
            }
            let other = if dir == Direction::Outgoing {
                edge.target()
            } else {
                edge.source()
            };
            if let Some(required_type) = other_node_type {
                if let Some(nd) = g.node_weight(other) {
                    if nd.node_type != required_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            count += 1;
        }
        Ok(count)
    }

    // ─── OVERRIDE: property-index lookups ─────────────────────────────────
    // Mirrors disk's `PropertyIndex` contract: `Some(vec)` on index hit
    // (possibly empty), `None` if no index exists. Mapped always returns
    // `Some` because the index builds lazily on first hit — same cost
    // model as disk's auto-built `title` global index but triggered by
    // the query rather than by save.
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        let block = self.ensure_property_index(node_type, property);
        // Mirror disk's contract: `None` means "no index for this
        // (type, property)" — the matcher will try the next alias
        // or fall through to a full scan. `Some(vec)` means the
        // index covers this pair (vec may be empty for a miss).
        // Treat "no string values found" as "no index" so the
        // matcher keeps trying aliases (nid→id→qid, etc.).
        if block.keys.is_empty() {
            return None;
        }
        Some(block.lookup_eq(value))
    }

    fn lookup_by_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        let block = self.ensure_property_index(node_type, property);
        if block.keys.is_empty() {
            return None;
        }
        Some(block.lookup_prefix(prefix, limit))
    }

    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        let block = self.ensure_global_property_index(property);
        if block.keys.is_empty() {
            return None;
        }
        Some(block.lookup_eq(value))
    }

    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        let block = self.ensure_global_property_index(property);
        if block.keys.is_empty() {
            return None;
        }
        Some(block.lookup_prefix(prefix, limit))
    }
}

// ──────────────────────────────────────────────────────────────────────────
// MappedGraph GraphWrite — lazy-index invalidation + undo-journal capture
//
// GraphWrite invalidates the type index on every edge mutation and the
// property index on every node-property mutation. `add_node` and
// `remove_node` also clear the property index since the set of
// `(value, node_idx)` pairs has changed.
//
// On top of that, and since 2026-07-30, the same undo-capture seam
// `MemoryGraph` carries above — for the same reason: `inner` is a heap
// `StableDiGraph`, so every `UndoEntry` variant, all keyed on a petgraph
// index, is expressible here. What `StorageMode::Mapped` changes is where
// *properties* live (mmap-spilled column stores), not the node/edge graph, so
// the journal transfers verbatim.
//
// Each method captures the inverse of the edit *if a journal is installed*,
// then performs the edit — never at the cost of the invalidation it already
// owed. With no journal (the steady state, and every read path) the added cost
// is one `Option` discriminant check; the clone-the-pre-image work lives in
// `#[cold]` helpers.
// ──────────────────────────────────────────────────────────────────────────

impl MappedGraph {
    /// Clone `idx`'s current weight into the journal as its pre-statement
    /// state, unless this statement already captured it.
    #[cold]
    fn capture_node_weight(&mut self, idx: NodeIndex) {
        let inner = &self.inner;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_weight(idx, || inner.node_weight(idx).cloned());
        }
    }

    /// Edge counterpart of [`Self::capture_node_weight`].
    #[cold]
    fn capture_edge_weight(&mut self, idx: EdgeIndex) {
        let inner = &self.inner;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_weight(idx, || inner.edge_weight(idx).cloned());
        }
    }

    /// Any edge incident to `idx`, in either direction.
    fn first_incident_edge(&self, idx: NodeIndex) -> Option<EdgeIndex> {
        self.inner
            .edges_directed(idx, Direction::Outgoing)
            .next()
            .or_else(|| self.inner.edges_directed(idx, Direction::Incoming).next())
            .map(|edge| edge.id())
    }

    /// Detach `idx`'s edges one at a time, through the recorded
    /// [`GraphWrite::remove_edge`], so each gets its own journal entry.
    ///
    /// petgraph's `remove_node` drops incident edges in one internal sweep
    /// whose order we neither observe nor control, which would leave reverse
    /// replay unable to hand each edge back its own free-list slot. Doing the
    /// detach ourselves makes the removal order the recorded order. The end
    /// state is identical either way; the Cypher delete path already detaches
    /// explicitly (`mutation::maintain::detach_delete_nodes`), so in practice
    /// this loop finds nothing to do and exists for every *other* caller of
    /// `remove_node`.
    #[cold]
    fn detach_for_journal(&mut self, idx: NodeIndex) {
        while let Some(edge) = self.first_incident_edge(idx) {
            GraphWrite::remove_edge(self, edge);
        }
    }
}

impl GraphWrite for MappedGraph {
    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        // Caller may mutate properties — invalidate property index.
        self.invalidate_property_index();
        if self.undo.is_some() {
            self.capture_node_weight(idx);
        }
        self.inner_mut().node_weight_mut(idx)
    }

    /// Bypasses the undo journal as well as the WAL recorder.
    ///
    /// Two callers, both pure storage bookkeeping and both mapped-relevant:
    /// the columnar handle-refresh sweep in
    /// [`crate::graph::languages::cypher::executor`], and the
    /// detach/reattach pair in [`crate::graph::mutation::batch`], which runs
    /// *only* under `is_mapped() || is_disk()` and touches every existing node
    /// of a type per chunk. Capturing a `NodeData` pre-image for each of them
    /// would make one `SET`, or one bulk `CREATE`, cost a clone per node *of
    /// the type* — the O(V+E)-per-write cost this journal exists to remove,
    /// reintroduced at a smaller constant. Without this override `MappedGraph`
    /// would inherit the trait default, which forwards to the *recorded*
    /// `node_weight_mut`; that is precisely the quadratic amplification commit
    /// 3bf9ef00 removed from the WAL, and no existing guard would see it in
    /// the journal.
    ///
    /// Nothing else may use this method to skip capture: a caller that
    /// changes a node's *content* silently is unrecoverable on rollback.
    #[inline]
    fn node_weight_mut_silent(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        self.invalidate_property_index();
        self.inner_mut().node_weight_mut(idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        // Mutating an edge weight can change its connection_type, which
        // would invalidate the per-conn-type index.
        self.invalidate_type_index();
        if self.undo.is_some() {
            self.capture_edge_weight(idx);
        }
        self.inner_mut().edge_weight_mut(idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        self.invalidate_property_index();
        let node_type = data.node_type;
        let idx = self.inner_mut().add_node(data);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_added(idx, node_type);
        }
        idx
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        // Removing a node removes its incident edges and any property
        // entries pointing at it.
        self.invalidate_type_index();
        self.invalidate_property_index();
        if self.undo.is_some() {
            self.detach_for_journal(idx);
        }
        let removed = self.inner_mut().remove_node(idx)?;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_removed(idx, removed.clone());
        }
        Some(removed)
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        self.invalidate_type_index();
        let idx = self.inner_mut().add_edge(a, b, data);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_added(idx);
        }
        idx
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.invalidate_type_index();
        let endpoints = self.undo.is_some().then(|| self.inner.edge_endpoints(idx));
        let removed = self.inner_mut().remove_edge(idx)?;
        if let Some(journal) = self.undo.as_deref_mut() {
            if let Some(Some((src, tgt))) = endpoints {
                journal.note_edge_removed(idx, src, tgt, removed.clone());
            }
        }
        Some(removed)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Disk-backed GraphRead / GraphWrite impls
//
// Every method here delegates to the corresponding inherent on `DiskGraph`
// (CSR + mmap columns + per-query arenas). Disk-only helpers return
// concrete `Some(...)` values where the trait defaults would return `None`.
// ──────────────────────────────────────────────────────────────────────────

impl GraphRead for DiskGraph {
    type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = GraphEdgeIndices<'a>;
    type EdgesIter<'a> = GraphEdges<'a>;
    type EdgeReferencesIter<'a> = GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = GraphNeighbors<'a>;

    #[inline]
    fn node_count(&self) -> usize {
        DiskGraph::node_count(self)
    }

    #[inline]
    fn edge_count(&self) -> usize {
        DiskGraph::edge_count(self)
    }

    #[inline]
    fn node_bound(&self) -> usize {
        DiskGraph::node_bound(self)
    }

    #[inline]
    fn is_memory(&self) -> bool {
        false
    }

    #[inline]
    fn is_mapped(&self) -> bool {
        false
    }

    #[inline]
    fn is_disk(&self) -> bool {
        true
    }

    #[inline(always)]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        DiskGraph::node_type_of(self, idx)
    }

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        DiskGraph::node_weight(self, idx)
    }

    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        DiskGraph::get_node_property(self, idx, key)
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        DiskGraph::get_node_id(self, idx)
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        DiskGraph::get_node_title(self, idx)
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        // Disk keeps the allocating route — the zero-alloc win is heap/mapped-specific.
        // Equality still works correctly.
        DiskGraph::get_node_property(self, idx, key)
            .map(|v| matches!(v, Value::String(ref s) if s == target))
    }

    #[inline]
    fn node_indices(&self) -> GraphNodeIndices<'_> {
        GraphNodeIndices::Disk(self.node_indices_iter())
    }

    #[inline]
    fn edge_indices(&self) -> GraphEdgeIndices<'_> {
        GraphEdgeIndices::Disk(self.edge_indices_iter())
    }

    #[inline]
    fn edge_references(&self) -> GraphEdgeReferences<'_> {
        GraphEdgeReferences::Disk(self.edge_references_iter())
    }

    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        self.edge_weights_iter()
    }

    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> GraphEdges<'_> {
        GraphEdges::Disk(self.edges_directed_iter(idx, dir))
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> GraphEdges<'_> {
        GraphEdges::Disk(self.edges_directed_iter(idx, Direction::Outgoing))
    }

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> GraphEdges<'_> {
        GraphEdges::Disk(self.edges_directed_filtered_iter(
            idx,
            dir,
            conn_type_filter.map(|k| k.as_u64()),
        ))
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> GraphEdgesConnecting<'_> {
        GraphEdgesConnecting::Disk(self.edges_connecting_iter(a, b))
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        DiskGraph::edge_weight(self, idx)
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        DiskGraph::find_edge(self, a, b)
    }

    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.edge_endpoints_fn(idx)
    }

    #[inline(always)]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        Box::new((0..self.next_edge_idx).filter_map(move |i| {
            let ep = self.edge_endpoint(i as usize);
            if ep.source == TOMBSTONE_EDGE {
                return None;
            }
            Some((
                NodeIndex::new(ep.source as usize),
                NodeIndex::new(ep.target as usize),
                InternedKey::from_u64(ep.connection_type),
            ))
        }))
    }

    #[inline]
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> GraphNeighbors<'_> {
        GraphNeighbors::Disk(self.neighbors_directed_iter(idx, dir))
    }

    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> GraphNeighbors<'_> {
        GraphNeighbors::Disk(self.neighbors_undirected_iter(idx))
    }

    #[inline]
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        DiskGraph::sources_for_conn_type_bounded(self, conn_type.as_u64(), max)
    }

    #[inline]
    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        DiskGraph::lookup_peer_counts(self, conn_type.as_u64())
    }

    #[inline]
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        DiskGraph::lookup_property_eq(self, node_type, property, value)
    }

    #[inline]
    fn lookup_by_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        DiskGraph::lookup_property_prefix(self, node_type, property, prefix, limit)
    }

    #[inline]
    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        DiskGraph::lookup_global_eq(self, property, value)
    }

    #[inline]
    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        DiskGraph::lookup_global_prefix(self, property, prefix, limit)
    }

    #[inline]
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        DiskGraph::count_edges_grouped_by_peer(self, conn_type.as_u64(), dir, deadline)
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
        DiskGraph::count_edges_filtered(
            self,
            node,
            dir,
            conn_type.map(|k| k.as_u64()),
            other_node_type,
            deadline,
        )
    }

    #[inline]
    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        Box::new(
            DiskGraph::iter_peers_filtered(self, node, dir, conn_type)
                .into_iter()
                .map(|(peer, edge_idx)| (peer, EdgeIndex::new(edge_idx as usize))),
        )
    }

    #[inline]
    fn reset_arenas(&self) {
        DiskGraph::reset_arenas(self);
    }
}

impl GraphWrite for DiskGraph {
    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        DiskGraph::node_weight_mut(self, idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        DiskGraph::edge_weight_mut(self, idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        DiskGraph::add_node(self, data)
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        DiskGraph::remove_node(self, idx)
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        DiskGraph::add_edge(self, a, b, data)
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        DiskGraph::remove_edge(self, idx)
    }

    #[inline]
    fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        DiskGraph::update_row_id(self, node_idx, row_id);
    }

    #[inline]
    fn flush_pending_writes(&mut self) {
        // Drains node_mut_cache + edge_mut_cache into column_stores /
        // edge_properties (the steady-state read sources) and resets
        // the materialization arenas. Idempotent — safe to call when
        // the caches are empty.
        DiskGraph::clear_arenas(self);
    }
}
