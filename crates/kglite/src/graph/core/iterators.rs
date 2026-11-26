// src/graph/iterators.rs
//
// Enum iterator wrappers for GraphBackend. These allow the same iterator
// interface to work for both InMemory (petgraph) and Disk backends.
//
// GraphEdgeRef implements petgraph::visit::EdgeRef so callers that import
// that trait can call .source(), .target(), .weight(), .id() unchanged.

use super::csr_iterators::{
    CsrEdgeIndices, CsrEdgeReferences, CsrEdges, CsrEdgesConnecting, CsrNeighbors, CsrNodeIndices,
};
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::disk::csr::{CsrEdge, TOMBSTONE_EDGE};
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::forked_edge_iters::{
    ForkedEdgeRefs, ForkedEdges, ForkedEdgesConnecting, ForkedNeighbors,
};
use crate::graph::storage::forked_edges::OverlayBits;
use crate::graph::storage::forked_slots::ExtraIter;
use crate::graph::storage::frozen::FrozenNodeIndices;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use crate::graph::storage::record_ref::RecordRef;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

/// Binary search for the range of CSR edges matching a specific connection_type.
/// Returns (lo, hi) such that edges[lo..hi] all have the target connection_type.
/// Requires CSR edges to be sorted by connection_type within [start, end).
/// Used by DiskEdges iterator and DiskGraph::count_edges_filtered.
/// Skips tombstoned edges during the search.
pub(crate) fn binary_search_conn_type(
    edges: &MmapOrVec<CsrEdge>,
    graph: &DiskGraph,
    start: usize,
    end: usize,
    target_ct: u64,
) -> (usize, usize) {
    if start >= end {
        return (start, start);
    }

    // Helper: get connection_type for edge at position i
    let ct_at = |i: usize| -> u64 {
        let e = edges.get(i);
        if e.edge_idx == TOMBSTONE_EDGE || !graph.edge_is_alive(e.edge_idx) {
            return u64::MAX; // tombstones sort to end
        }
        graph.edge_endpoint(e.edge_idx as usize).connection_type
    };

    // Lower bound: first edge with ct >= target_ct
    let mut lo_l = start;
    let mut lo_r = end;
    while lo_l < lo_r {
        let mid = lo_l + (lo_r - lo_l) / 2;
        if ct_at(mid) < target_ct {
            lo_l = mid + 1;
        } else {
            lo_r = mid;
        }
    }

    // Upper bound: first edge with ct > target_ct
    let mut hi_l = lo_l;
    let mut hi_r = end;
    while hi_l < hi_r {
        let mid = hi_l + (hi_r - hi_l) / 2;
        if ct_at(mid) <= target_ct {
            hi_l = mid + 1;
        } else {
            hi_r = mid;
        }
    }

    (lo_l, hi_l)
}

// ============================================================================
// GraphEdgeRef — unified edge reference
// ============================================================================

/// Where a [`GraphEdgeRef`]'s full `EdgeData` comes from.
///
/// In-memory / mapped backends already hold a borrowed `&EdgeData`, so they
/// use `Ref`. The disk backend uses `DiskLazy`: it carries only the graph
/// handle and materialises the `EdgeData` (heap alloc + property clone) *on
/// demand* when `weight()` is first called. Traversals that need only the
/// target and the connection type — pattern matching, variable-length paths,
/// `shortestPath` — never trigger that materialisation. The connection type
/// itself is always carried directly (a cheap `u64`-backed `InternedKey` read
/// from the CSR endpoint table), so type filtering costs nothing extra.
#[derive(Clone, Copy)]
enum EdgeWeightSrc<'a> {
    Ref(&'a EdgeData),
    DiskLazy(&'a DiskGraph),
}

/// A lightweight edge reference returned by graph edge iterators.
#[derive(Clone, Copy)]
pub struct GraphEdgeRef<'a> {
    source_: NodeIndex,
    target_: NodeIndex,
    index_: EdgeIndex,
    conn_type_: InternedKey,
    weight_: EdgeWeightSrc<'a>,
}

impl<'a> GraphEdgeRef<'a> {
    #[inline]
    pub fn new(
        source: NodeIndex,
        target: NodeIndex,
        index: EdgeIndex,
        weight: &'a EdgeData,
    ) -> Self {
        GraphEdgeRef {
            source_: source,
            target_: target,
            index_: index,
            conn_type_: weight.connection_type,
            weight_: EdgeWeightSrc::Ref(weight),
        }
    }

    /// Disk constructor: the connection type is read cheaply from the CSR
    /// endpoint table; the full `EdgeData` is materialised lazily in
    /// `weight()` only if a caller actually needs the edge's properties.
    #[inline]
    pub fn new_disk_lazy(
        source: NodeIndex,
        target: NodeIndex,
        index: EdgeIndex,
        conn_type: InternedKey,
        graph: &'a DiskGraph,
    ) -> Self {
        GraphEdgeRef {
            source_: source,
            target_: target,
            index_: index,
            conn_type_: conn_type,
            weight_: EdgeWeightSrc::DiskLazy(graph),
        }
    }

    // Inherent methods matching petgraph's EdgeReference API.
    // These allow callers to use .source()/.target()/.weight()/.id()
    // without importing the EdgeRef trait.

    #[inline]
    pub fn source(&self) -> NodeIndex {
        self.source_
    }

    #[inline]
    pub fn target(&self) -> NodeIndex {
        self.target_
    }

    /// The edge's connection type — always cheap (no materialisation on disk).
    /// Prefer this over `weight().connection_type` in hot traversal paths.
    #[inline]
    pub fn connection_type(&self) -> InternedKey {
        self.conn_type_
    }

    #[inline]
    pub fn weight(&self) -> RecordRef<'a, EdgeData> {
        match self.weight_ {
            EdgeWeightSrc::Ref(w) => RecordRef::Borrowed(w),
            EdgeWeightSrc::DiskLazy(g) => {
                RecordRef::Owned(Box::new(g.materialize_edge(self.index_.index() as u32)))
            }
        }
    }

    #[inline]
    pub fn id(&self) -> EdgeIndex {
        self.index_
    }
}

// ============================================================================
// GraphNodeIndices
// ============================================================================

pub enum GraphNodeIndices<'a> {
    InMemory(petgraph::stable_graph::NodeIndices<'a, NodeData, u32>),
    Csr(CsrNodeIndices<'a>),
    /// A `GraphBackend::Forked` overlay that reused no vacated slot: the
    /// shared base's live indices, then the contiguous run appended past
    /// them. Every appended index is above every base index, so the
    /// concatenation is ascending — the unforked graph's scan order, which
    /// `type_indices` bucket order and the rollback fidelity fingerprints pin.
    Forked {
        base: Box<petgraph::stable_graph::NodeIndices<'a, NodeData, u32>>,
        appended: std::ops::Range<usize>,
    },
    /// A `GraphBackend::Forked` overlay that removed base nodes or took slots
    /// inside the base's range.
    ForkedMerged(
        Box<ForkedMergedIndices<'a, petgraph::stable_graph::NodeIndices<'a, NodeData, u32>>>,
    ),
    /// [`Forked`](Self::Forked) over a frozen base.
    FrozenForked {
        base: Box<FrozenNodeIndices<'a>>,
        appended: std::ops::Range<usize>,
    },
    /// [`ForkedMerged`](Self::ForkedMerged) over a frozen base.
    FrozenForkedMerged(Box<ForkedMergedIndices<'a, FrozenNodeIndices<'a>>>),
}

/// Ascending merge of a base's live indices, minus the base nodes the overlay
/// removed, with every slot the overlay allocated where the base holds no
/// live node (a vacated slot it reused, or a fresh slot past the bound).
pub struct ForkedMergedIndices<'a, B> {
    base: B,
    /// A base index read ahead while a smaller overlay slot was yielded.
    pending: Option<NodeIndex>,
    extra: ExtraIter<'a>,
    /// The smallest overlay slot not yet yielded; `usize::MAX` once exhausted,
    /// so the common step is one comparison.
    next_extra: usize,
    dead: Option<&'a OverlayBits>,
}

impl<'a, B: Iterator<Item = NodeIndex>> ForkedMergedIndices<'a, B> {
    pub(crate) fn new(base: B, mut extra: ExtraIter<'a>, dead: Option<&'a OverlayBits>) -> Self {
        let next_extra = extra.next().map_or(usize::MAX, |idx| idx as usize);
        Self {
            base,
            pending: None,
            extra,
            next_extra,
            dead,
        }
    }

    #[inline]
    fn take_extra(&mut self) -> NodeIndex {
        let idx = self.next_extra;
        self.next_extra = self.extra.next().map_or(usize::MAX, |next| next as usize);
        NodeIndex::new(idx)
    }

    /// An overlay slot was vacant in the base, so the two never collide.
    #[inline]
    fn next_index(&mut self) -> Option<NodeIndex> {
        loop {
            match self.pending.take().or_else(|| self.base.next()) {
                Some(base)
                    if self
                        .dead
                        .is_some_and(|dead| dead.holds(base.index() as u32)) =>
                {
                    continue;
                }
                Some(base) if base.index() < self.next_extra => return Some(base),
                Some(base) => {
                    self.pending = Some(base);
                    return Some(self.take_extra());
                }
                None if self.next_extra != usize::MAX => return Some(self.take_extra()),
                None => return None,
            }
        }
    }

    /// `size_hint` of the merge: removed base nodes make the base's lower bound
    /// too high.
    fn size_hint_merged(&self) -> (usize, Option<usize>) {
        let (blo, bhi) = self.base.size_hint();
        let extra = self.remaining_extra();
        let blo = if self.dead.is_some() { 0 } else { blo };
        (blo + extra, bhi.map(|h| h + extra))
    }

    fn remaining_extra(&self) -> usize {
        self.extra.len() + usize::from(self.next_extra != usize::MAX)
    }
}

/// Iterates alive node slots in the DiskGraph's mmap'd node_slots array.
pub struct DiskNodeIndices<'a> {
    graph: &'a DiskGraph,
    pos: usize,
    len: usize,
}

impl<'a> DiskNodeIndices<'a> {
    pub fn new(graph: &'a DiskGraph) -> Self {
        let len = graph.node_slot_len();
        DiskNodeIndices { graph, pos: 0, len }
    }
}

impl<'a> Iterator for DiskNodeIndices<'a> {
    type Item = NodeIndex;
    fn next(&mut self) -> Option<NodeIndex> {
        while self.pos < self.len {
            let i = self.pos;
            self.pos += 1;
            if self.graph.node_slot(i).is_alive() {
                return Some(NodeIndex::new(i));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.len - self.pos))
    }
}

impl GraphNodeIndices<'_> {
    /// Every arm but the heap one, out of line so the heap arm of `next` stays
    /// small enough to inline into the scan loops.
    #[inline(never)]
    fn next_other(&mut self) -> Option<NodeIndex> {
        match self {
            GraphNodeIndices::InMemory(iter) => iter.next(),
            GraphNodeIndices::Csr(iter) => iter.next(),
            GraphNodeIndices::Forked { base, appended } => {
                base.next().or_else(|| appended.next().map(NodeIndex::new))
            }
            GraphNodeIndices::ForkedMerged(iter) => iter.next_index(),
            GraphNodeIndices::FrozenForked { base, appended } => {
                base.next().or_else(|| appended.next().map(NodeIndex::new))
            }
            GraphNodeIndices::FrozenForkedMerged(iter) => iter.next_index(),
        }
    }
}

impl<'a> Iterator for GraphNodeIndices<'a> {
    type Item = NodeIndex;

    #[inline]
    fn next(&mut self) -> Option<NodeIndex> {
        match self {
            GraphNodeIndices::InMemory(iter) => iter.next(),
            GraphNodeIndices::ForkedMerged(iter) => iter.next_index(),
            other => other.next_other(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            GraphNodeIndices::InMemory(iter) => iter.size_hint(),
            GraphNodeIndices::Csr(iter) => iter.size_hint(),
            GraphNodeIndices::Forked { base, appended } => {
                let (blo, bhi) = base.size_hint();
                let extra = appended.len();
                (blo + extra, bhi.map(|h| h + extra))
            }
            GraphNodeIndices::ForkedMerged(iter) => iter.size_hint_merged(),
            GraphNodeIndices::FrozenForked { base, appended } => {
                let (blo, bhi) = base.size_hint();
                let extra = appended.len();
                (blo + extra, bhi.map(|h| h + extra))
            }
            GraphNodeIndices::FrozenForkedMerged(iter) => iter.size_hint_merged(),
        }
    }
}

// ============================================================================
// GraphEdges — edges from a single node (directed)
// ============================================================================

pub enum GraphEdges<'a> {
    InMemory(petgraph::stable_graph::Edges<'a, EdgeData, petgraph::Directed, u32>),
    Csr(CsrEdges<'a>),
    /// A fork with edge changes: overlay edges, then the base's.
    Forked(Box<ForkedEdges<'a>>),
}

/// Iterates CSR edges for a specific node, materializing EdgeData on the fly.
/// When `conn_type_filter` is set, edges with non-matching connection types
/// are skipped before materialization (avoids an `EdgeData` build + property lookup).
pub struct DiskEdges<'a> {
    graph: &'a DiskGraph,
    direction: Direction,
    source_node: NodeIndex,
    // Lazy CSR edge access — indexes into the mmap'd array on each next() call
    // instead of pre-collecting into a Vec (avoids O(degree) allocation).
    csr_edges: Option<&'a MmapOrVec<CsrEdge>>,
    csr_start: usize,
    csr_end: usize,
    csr_pos: usize,
    // Overflow edges (appended after CSR construction)
    overflow: Option<&'a [CsrEdge]>,
    overflow_pos: usize,
    // Optional connection type pre-filter (u64 for O(1) comparison)
    conn_type_filter: Option<u64>,
}

impl<'a> DiskEdges<'a> {
    pub fn new(
        graph: &'a DiskGraph,
        direction: Direction,
        source_node: NodeIndex,
        edges: &'a MmapOrVec<CsrEdge>,
        start: usize,
        end: usize,
        overflow: Option<&'a Vec<CsrEdge>>,
    ) -> Self {
        DiskEdges {
            graph,
            direction,
            source_node,
            csr_edges: Some(edges),
            csr_start: start,
            csr_end: end,
            csr_pos: start,
            overflow: overflow.map(|v| v.as_slice()),
            overflow_pos: 0,
            conn_type_filter: None,
        }
    }

    /// Set a connection type pre-filter. When the CSR is sorted by connection
    /// type (csr_sorted_by_type=true), narrows the iteration range via binary
    /// search — O(log D) instead of O(D) for high-degree nodes. Falls back to
    /// linear scan when unsorted (backward-compatible with older graphs).
    pub fn with_conn_type_filter(mut self, ct: u64) -> Self {
        self.conn_type_filter = Some(ct);
        // Binary search to narrow range when CSR edges are sorted by type
        if self.graph.csr_sorted_by_type {
            if let Some(edges) = self.csr_edges {
                let (lo, hi) =
                    binary_search_conn_type(edges, self.graph, self.csr_start, self.csr_end, ct);
                self.csr_start = lo;
                self.csr_end = hi;
                self.csr_pos = lo;
                // Clear filter — all edges in [lo, hi) match, no per-edge check needed
                self.conn_type_filter = None;
            }
        }
        self
    }

    #[inline]
    fn make_edge_ref(&self, csr: &CsrEdge) -> GraphEdgeRef<'a> {
        // Lazy: carry only the cheap connection type (a bare u64 read from the
        // endpoint table); defer the EdgeData materialisation (heap alloc +
        // property clone) until a caller calls `weight()`.
        let conn_type = InternedKey::from_u64(
            self.graph
                .edge_endpoint(csr.edge_idx as usize)
                .connection_type,
        );
        let (src, tgt) = match self.direction {
            Direction::Outgoing => (self.source_node, NodeIndex::new(csr.peer as usize)),
            Direction::Incoming => (NodeIndex::new(csr.peer as usize), self.source_node),
        };
        GraphEdgeRef::new_disk_lazy(
            src,
            tgt,
            EdgeIndex::new(csr.edge_idx as usize),
            conn_type,
            self.graph,
        )
    }
}

impl<'a> Iterator for DiskEdges<'a> {
    type Item = GraphEdgeRef<'a>;

    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        // First: iterate CSR edges lazily via index into MmapOrVec
        if let Some(edges) = self.csr_edges {
            while self.csr_pos < self.csr_end {
                let e = edges.get(self.csr_pos);
                self.csr_pos += 1;
                if e.edge_idx != TOMBSTONE_EDGE && self.graph.edge_is_alive(e.edge_idx) {
                    // Pre-filter by connection type before materializing
                    if let Some(ct) = self.conn_type_filter {
                        if self
                            .graph
                            .edge_endpoint(e.edge_idx as usize)
                            .connection_type
                            != ct
                        {
                            continue;
                        }
                    }
                    return Some(self.make_edge_ref(&e));
                }
            }
        }
        // Then: iterate overflow edges
        if let Some(overflow) = self.overflow {
            while self.overflow_pos < overflow.len() {
                let e = &overflow[self.overflow_pos];
                self.overflow_pos += 1;
                if e.edge_idx != TOMBSTONE_EDGE && self.graph.edge_is_alive(e.edge_idx) {
                    if let Some(ct) = self.conn_type_filter {
                        if self
                            .graph
                            .edge_endpoint(e.edge_idx as usize)
                            .connection_type
                            != ct
                        {
                            continue;
                        }
                    }
                    return Some(self.make_edge_ref(e));
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_csr = self.csr_end.saturating_sub(self.csr_pos);
        let remaining_overflow = self
            .overflow
            .map(|o| o.len().saturating_sub(self.overflow_pos))
            .unwrap_or(0);
        (0, Some(remaining_csr + remaining_overflow))
    }
}

impl<'a> GraphEdges<'a> {
    /// Every arm but the heap one, out of line so the in-memory `next` stays small enough to
    /// inline into the traversal loops.
    #[inline(never)]
    fn next_other(&mut self) -> Option<GraphEdgeRef<'a>> {
        match self {
            GraphEdges::InMemory(iter) => iter
                .next()
                .map(|er| GraphEdgeRef::new(er.source(), er.target(), er.id(), er.weight())),
            GraphEdges::Csr(iter) => iter.next(),
            GraphEdges::Forked(iter) => iter.next(),
        }
    }
}

impl<'a> Iterator for GraphEdges<'a> {
    type Item = GraphEdgeRef<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        if let GraphEdges::InMemory(iter) = self {
            return iter
                .next()
                .map(|er| GraphEdgeRef::new(er.source(), er.target(), er.id(), er.weight()));
        }
        self.next_other()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            GraphEdges::InMemory(iter) => iter.size_hint(),
            GraphEdges::Csr(iter) => iter.size_hint(),
            GraphEdges::Forked(_) => (0, None),
        }
    }
}

// ============================================================================
// GraphEdgeReferences — all edges in the graph
// ============================================================================

pub enum GraphEdgeReferences<'a> {
    InMemory(petgraph::stable_graph::EdgeReferences<'a, EdgeData, u32>),
    Csr(CsrEdgeReferences<'a>),
    Forked(Box<ForkedEdgeRefs<'a>>),
}

/// Iterates all edges in the DiskGraph by scanning edge_endpoints.
pub struct DiskEdgeReferences<'a> {
    graph: &'a DiskGraph,
    pos: u32,
    total: u32,
}

impl<'a> DiskEdgeReferences<'a> {
    pub fn new(graph: &'a DiskGraph) -> Self {
        DiskEdgeReferences {
            graph,
            pos: 0,
            total: graph.next_edge_idx,
        }
    }
}

impl<'a> Iterator for DiskEdgeReferences<'a> {
    type Item = GraphEdgeRef<'a>;
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        while self.pos < self.total {
            let i = self.pos;
            self.pos += 1;
            let ep = self.graph.edge_endpoint(i as usize);
            if ep.source != TOMBSTONE_EDGE {
                // Lazy: carry the cheap connection type from the endpoint table;
                // defer EdgeData materialisation until `weight()` is called.
                return Some(GraphEdgeRef::new_disk_lazy(
                    NodeIndex::new(ep.source as usize),
                    NodeIndex::new(ep.target as usize),
                    EdgeIndex::new(i as usize),
                    InternedKey::from_u64(ep.connection_type),
                    self.graph,
                ));
            }
        }
        None
    }
}

impl<'a> Iterator for GraphEdgeReferences<'a> {
    type Item = GraphEdgeRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        match self {
            GraphEdgeReferences::InMemory(iter) => iter
                .next()
                .map(|er| GraphEdgeRef::new(er.source(), er.target(), er.id(), er.weight())),
            GraphEdgeReferences::Csr(iter) => iter.next(),
            GraphEdgeReferences::Forked(iter) => iter.next(),
        }
    }
}

// ============================================================================
// GraphEdgeIndices — edge index iteration
// ============================================================================

pub enum GraphEdgeIndices<'a> {
    InMemory(petgraph::stable_graph::EdgeIndices<'a, EdgeData, u32>),
    Csr(CsrEdgeIndices<'a>),
    Forked(Box<ForkedEdgeRefs<'a>>),
}

/// Iterates valid edge indices by scanning edge_endpoints for non-tombstones.
pub struct DiskEdgeIndices<'a> {
    graph: &'a DiskGraph,
    pos: u32,
    total: u32,
}

impl<'a> DiskEdgeIndices<'a> {
    pub fn new(total: u32, graph: &'a DiskGraph) -> Self {
        DiskEdgeIndices {
            graph,
            pos: 0,
            total,
        }
    }
}

impl<'a> Iterator for DiskEdgeIndices<'a> {
    type Item = EdgeIndex;
    fn next(&mut self) -> Option<EdgeIndex> {
        while self.pos < self.total {
            let i = self.pos;
            self.pos += 1;
            let ep = self.graph.edge_endpoint(i as usize);
            if ep.source != TOMBSTONE_EDGE {
                return Some(EdgeIndex::new(i as usize));
            }
        }
        None
    }
}

impl<'a> Iterator for GraphEdgeIndices<'a> {
    type Item = EdgeIndex;

    #[inline]
    fn next(&mut self) -> Option<EdgeIndex> {
        match self {
            GraphEdgeIndices::InMemory(iter) => iter.next(),
            GraphEdgeIndices::Csr(iter) => iter.next(),
            GraphEdgeIndices::Forked(iter) => iter.next().map(|edge| edge.id()),
        }
    }
}

// ============================================================================
// GraphEdgesConnecting — edges between two specific nodes
// ============================================================================

pub enum GraphEdgesConnecting<'a> {
    InMemory(petgraph::stable_graph::EdgesConnecting<'a, EdgeData, petgraph::Directed, u32>),
    Csr(CsrEdgesConnecting<'a>),
    Forked(Box<ForkedEdgesConnecting<'a>>),
}

/// Iterates edges connecting two specific nodes via the outgoing CSR + overflow.
pub struct DiskEdgesConnecting<'a> {
    inner: DiskEdges<'a>,
    target: u32,
}

impl<'a> DiskEdgesConnecting<'a> {
    pub fn new(graph: &'a DiskGraph, a: NodeIndex, b: NodeIndex) -> Self {
        let inner = graph.edges_directed_iter(a, Direction::Outgoing);
        DiskEdgesConnecting {
            inner,
            target: b.index() as u32,
        }
    }
}

impl<'a> Iterator for DiskEdgesConnecting<'a> {
    type Item = GraphEdgeRef<'a>;
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        self.inner
            .by_ref()
            .find(|edge| edge.target().index() as u32 == self.target)
    }
}

impl<'a> Iterator for GraphEdgesConnecting<'a> {
    type Item = GraphEdgeRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        match self {
            GraphEdgesConnecting::InMemory(iter) => iter
                .next()
                .map(|er| GraphEdgeRef::new(er.source(), er.target(), er.id(), er.weight())),
            GraphEdgesConnecting::Csr(iter) => iter.next(),
            GraphEdgesConnecting::Forked(iter) => iter.next(),
        }
    }
}

// ============================================================================
// GraphNeighbors — neighbor node iteration
// ============================================================================

pub enum GraphNeighbors<'a> {
    InMemory(petgraph::stable_graph::Neighbors<'a, EdgeData, u32>),
    Csr(CsrNeighbors<'a>),
    Forked(Box<ForkedNeighbors<'a>>),
}

/// Iterates neighbor nodes. Uses pre-collected Vec of NodeIndex.
pub struct DiskNeighbors {
    peers: Vec<NodeIndex>,
    pos: usize,
}

impl DiskNeighbors {
    pub fn new(
        edges: &MmapOrVec<CsrEdge>,
        start: usize,
        end: usize,
        overflow: Option<&Vec<CsrEdge>>,
    ) -> Self {
        let mut peers = Vec::with_capacity(end - start);
        for i in start..end {
            let e = edges.get(i);
            if e.edge_idx != TOMBSTONE_EDGE {
                peers.push(NodeIndex::new(e.peer as usize));
            }
        }
        if let Some(list) = overflow {
            for e in list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }
        DiskNeighbors { peers, pos: 0 }
    }

    pub fn new_empty() -> Self {
        DiskNeighbors {
            peers: Vec::new(),
            pos: 0,
        }
    }

    pub fn from_collected(peers: Vec<NodeIndex>) -> Self {
        DiskNeighbors { peers, pos: 0 }
    }
}

impl Iterator for DiskNeighbors {
    type Item = NodeIndex;
    fn next(&mut self) -> Option<NodeIndex> {
        if self.pos < self.peers.len() {
            let idx = self.peers[self.pos];
            self.pos += 1;
            Some(idx)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.peers.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> GraphNeighbors<'a> {
    /// Every arm but the heap one, out of line so the in-memory `next` stays small enough to
    /// inline into the traversal loops.
    #[inline(never)]
    fn next_other(&mut self) -> Option<NodeIndex> {
        match self {
            GraphNeighbors::InMemory(iter) => iter.next(),
            GraphNeighbors::Csr(iter) => iter.next(),
            GraphNeighbors::Forked(iter) => iter.next(),
        }
    }
}

impl<'a> Iterator for GraphNeighbors<'a> {
    type Item = NodeIndex;

    #[inline]
    fn next(&mut self) -> Option<NodeIndex> {
        if let GraphNeighbors::InMemory(iter) = self {
            return iter.next();
        }
        self.next_other()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            GraphNeighbors::InMemory(iter) => iter.size_hint(),
            GraphNeighbors::Csr(iter) => iter.size_hint(),
            GraphNeighbors::Forked(_) => (0, None),
        }
    }
}
