//! `GraphRead` over a [`FrozenGraph`].

use std::collections::HashMap;
use std::time::Instant;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;

use super::iters::{self, FrozenEdgeIndices, FrozenEdgeRefs, FrozenEdges, FrozenNeighbors};
use super::{FrozenGraph, VACANT};
use crate::datatypes::Value;
use crate::graph::core::csr_iterators::{
    CsrEdgeIndices, CsrEdgeReferences, CsrEdges, CsrEdgesConnecting, CsrNeighbors, CsrNodeIndices,
};
use crate::graph::core::iterators::{
    GraphEdgeIndices, GraphEdgeReferences, GraphEdges, GraphEdgesConnecting, GraphNeighbors,
    GraphNodeIndices,
};
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::record_ref::RecordRef;
use crate::graph::storage::GraphRead;

impl FrozenGraph {
    /// Count edges at `node` in `dir`, optionally of one type and with a far
    /// end of one node type. `exclude_self` skips self-loops (the incoming
    /// count the pattern matcher uses, which sees a loop on the outgoing side).
    pub(crate) fn count_edges_filtered_impl(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
        exclude_self: bool,
    ) -> Result<usize, String> {
        let (lo, hi) = match conn_type {
            Some(ct) => match self.ordinal_of(ct) {
                Some(ordinal) => self.typed_run(node.index(), dir, ordinal),
                None => return Ok(0),
            },
            None => self.run(node.index(), dir),
        };
        if !exclude_self && other_node_type.is_none() {
            return Ok(hi.saturating_sub(lo));
        }
        let (edges, _) = self.csr(dir);
        let Some(run) = edges.get(lo..hi) else {
            return Ok(0);
        };
        let mut count = 0usize;
        for (i, entry) in run.iter().enumerate() {
            if i.is_multiple_of(1 << 20) {
                if let Some(limit) = deadline {
                    if Instant::now() > limit {
                        return Err("Query timed out".to_string());
                    }
                }
            }
            let other = NodeIndex::new(entry.peer as usize);
            if exclude_self && other == node {
                continue;
            }
            if let Some(required) = other_node_type {
                if self.node_type_of(other) != Some(required) {
                    continue;
                }
            }
            count += 1;
        }
        Ok(count)
    }

    fn edges_over(
        &self,
        node: NodeIndex,
        dir: Direction,
        filter: Option<InternedKey>,
    ) -> GraphEdges<'_> {
        let edges = match filter {
            None => FrozenEdges::new(self, node, dir),
            Some(key) => match self.ordinal_of(key) {
                Some(ordinal) if !self.may_have_type(node.index(), dir, ordinal) => {
                    FrozenEdges::empty(self, node, dir)
                }
                Some(ordinal) => {
                    let (off, adjacency, tys) = self.dir_arrays(dir);
                    let (lo, hi) = Self::narrow(off, tys, node.index(), ordinal);
                    FrozenEdges::over_slices(self, node, dir, adjacency, tys, lo, hi)
                }
                None => FrozenEdges::empty(self, node, dir),
            },
        };
        GraphEdges::Csr(CsrEdges::Frozen(edges))
    }
}

impl GraphRead for FrozenGraph {
    type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = GraphEdgeIndices<'a>;
    type EdgesIter<'a> = GraphEdges<'a>;
    type EdgeReferencesIter<'a> = GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = GraphNeighbors<'a>;

    #[inline]
    fn node_count(&self) -> usize {
        self.live_node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.live_edge_count()
    }

    #[inline]
    fn node_bound(&self) -> usize {
        self.node_bound_len()
    }

    #[inline]
    fn edge_bound(&self) -> usize {
        self.edge_bound_len()
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
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        self.node_weight(idx).map(|node| node.node_type)
    }

    #[inline(always)]
    fn node_weight(&self, idx: NodeIndex) -> Option<RecordRef<'_, NodeData>> {
        let slot = idx.index();
        if self.node_is_alive(slot) {
            self.node_slice().get(slot).map(RecordRef::borrowed)
        } else {
            None
        }
    }

    // Through `node_view`, which pairs the row id with the store this backend
    // owns, as the mapped backend does.
    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        self.node_view(idx).and_then(|v| v.get_value(key))
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        self.node_view(idx).map(|v| v.id().into_owned())
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        self.node_view(idx).map(|v| v.title().into_owned())
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        self.node_view(idx).and_then(|v| v.str_prop_eq(key, target))
    }

    #[inline]
    fn column_store(&self, type_key: InternedKey) -> Option<&std::sync::Arc<ColumnStore>> {
        self.column_stores.get(&type_key)
    }

    fn column_stores_iter(
        &self,
    ) -> Box<dyn Iterator<Item = (InternedKey, &std::sync::Arc<ColumnStore>)> + '_> {
        Box::new(self.column_stores.iter().map(|(k, v)| (*k, v)))
    }

    #[inline]
    fn node_indices(&self) -> GraphNodeIndices<'_> {
        GraphNodeIndices::Csr(CsrNodeIndices::Frozen(iters::FrozenNodeIndices::new(self)))
    }

    #[inline]
    fn edge_indices(&self) -> GraphEdgeIndices<'_> {
        GraphEdgeIndices::Csr(CsrEdgeIndices::Frozen(FrozenEdgeIndices::new(self)))
    }

    #[inline]
    fn edge_references(&self) -> GraphEdgeReferences<'_> {
        GraphEdgeReferences::Csr(CsrEdgeReferences::Frozen(FrozenEdgeRefs::new(self)))
    }

    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = RecordRef<'a, EdgeData>> + 'a> {
        Box::new(iters::weights(self).map(RecordRef::borrowed))
    }

    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> GraphEdges<'_> {
        GraphEdges::Csr(CsrEdges::Frozen(FrozenEdges::new(self, idx, dir)))
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> GraphEdges<'_> {
        GraphEdges::Csr(CsrEdges::Frozen(FrozenEdges::new(
            self,
            idx,
            Direction::Outgoing,
        )))
    }

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> GraphEdges<'_> {
        self.edges_over(idx, dir, conn_type_filter)
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> GraphEdgesConnecting<'_> {
        GraphEdgesConnecting::Csr(CsrEdgesConnecting::Frozen(
            super::iters::FrozenEdgesConnecting::new(self, a, b),
        ))
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<RecordRef<'_, EdgeData>> {
        let slot = idx.index();
        self.endpoint(slot)
            .map(|ep| RecordRef::borrowed(self.edge_data(slot as u32, ep.ty)))
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        let (lo, hi) = self.run(a.index(), Direction::Outgoing);
        let (edges, _) = self.csr(Direction::Outgoing);
        edges
            .get(lo..hi)?
            .iter()
            .find(|entry| entry.peer as usize == b.index())
            .map(|entry| EdgeIndex::new(entry.edge_idx as usize))
    }

    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.endpoint(idx.index()).map(|ep| {
            (
                NodeIndex::new(ep.src as usize),
                NodeIndex::new(ep.dst as usize),
            )
        })
    }

    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        Box::new(
            self.endpoint_slots()
                .iter()
                .filter(|ep| ep.src != VACANT)
                .map(|ep| {
                    (
                        NodeIndex::new(ep.src as usize),
                        NodeIndex::new(ep.dst as usize),
                        self.type_key(ep.ty),
                    )
                }),
        )
    }

    #[inline]
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> GraphNeighbors<'_> {
        GraphNeighbors::Csr(CsrNeighbors::Frozen(FrozenNeighbors::directed(
            self, idx, dir,
        )))
    }

    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> GraphNeighbors<'_> {
        GraphNeighbors::Csr(CsrNeighbors::Frozen(FrozenNeighbors::undirected(self, idx)))
    }

    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        Some(self.sources_of(conn_type, max))
    }

    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        Some((*self.peer_counts(conn_type, Direction::Outgoing)).clone())
    }

    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        _deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        Ok((*self.peer_counts(conn_type, dir)).clone())
    }

    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
    ) -> Result<usize, String> {
        self.count_edges_filtered_impl(node, dir, conn_type, other_node_type, deadline, false)
    }

    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        let (lo, hi) = match conn_type {
            Some(key) => match self.ordinal_of(InternedKey::from_u64(key)) {
                Some(ordinal) => self.typed_run(node.index(), dir, ordinal),
                None => (0, 0),
            },
            None => self.run(node.index(), dir),
        };
        let (edges, _) = self.csr(dir);
        let run = edges.get(lo..hi).unwrap_or(&[]);
        Box::new(run.iter().map(|entry| {
            (
                NodeIndex::new(entry.peer as usize),
                EdgeIndex::new(entry.edge_idx as usize),
            )
        }))
    }
}
