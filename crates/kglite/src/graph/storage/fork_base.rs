//! The bases a [`ForkedGraph`](super::forked::ForkedGraph) can sit over, and
//! the seam the fold-back writes through.
//!
//! `Memory` and `Mapped` hold the same `StableDiGraph<NodeData, EdgeData>`,
//! column-store map, slot mirror and undo journal; they differ in the lazy
//! indexes `Mapped` caches over the graph. The overlay needs only the shared
//! part, so [`ForkBase`] picks the base and [`FoldTarget`] is what the fold
//! reads and writes. The one behavioural difference is
//! [`FoldTarget::after_fold`]: a fold rewrites the graph without going through
//! the `GraphWrite` hooks that keep those indexes honest.
//!
//! `Frozen` is the third base: a read-only topology served from a `.kgl`'s raw
//! sections. It has no petgraph, so it is never a fold *target*; an overlay over
//! it does not fold back, and a collapse copies it into a heap `MappedGraph`
//! first ([`ForkBase::deep_clone`]).

use std::sync::Arc;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::{EdgeReferences, Edges, StableDiGraph};
use petgraph::visit::{EdgeIndexable, EdgeRef, NodeIndexable};
use petgraph::{Directed, Direction};
use rustc_hash::FxHashMap;
use serde::{Serialize, Serializer};

use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::frozen::{FrozenEdgeRefs, FrozenEdges, FrozenGraph};
use crate::graph::storage::record_ref::RecordRef;
use crate::graph::storage::slot_mirror::SlotMirror;
use crate::graph::storage::undo::UndoJournal;
use crate::graph::storage::{GraphRead, MappedGraph, MemoryGraph};

/// The graph an overlay is folded into: a heap backend that owns the petgraph,
/// the store map, the slot mirror and the journal slot.
pub(crate) trait FoldTarget: Sized {
    fn graph(&self) -> &StableDiGraph<NodeData, EdgeData>;
    fn graph_mut(&mut self) -> &mut StableDiGraph<NodeData, EdgeData>;
    fn slot_mirror(&self) -> &SlotMirror;
    fn slot_mirror_mut(&mut self) -> &mut SlotMirror;
    fn column_stores_mut(&mut self) -> &mut FxHashMap<InternedKey, Arc<ColumnStore>>;
    fn set_undo(&mut self, undo: Option<Box<UndoJournal>>);
    /// Drop whatever derived state a fold left stale. The fold edits the
    /// petgraph and replaces the store map directly, so none of the
    /// invalidation the `GraphWrite` methods do has run.
    fn after_fold(&mut self, touched_edges: bool);
}

impl FoldTarget for MemoryGraph {
    #[inline]
    fn graph(&self) -> &StableDiGraph<NodeData, EdgeData> {
        self.inner()
    }
    #[inline]
    fn graph_mut(&mut self) -> &mut StableDiGraph<NodeData, EdgeData> {
        self.inner_mut()
    }
    #[inline]
    fn slot_mirror(&self) -> &SlotMirror {
        &self.slot_mirror
    }
    #[inline]
    fn slot_mirror_mut(&mut self) -> &mut SlotMirror {
        &mut self.slot_mirror
    }
    #[inline]
    fn column_stores_mut(&mut self) -> &mut FxHashMap<InternedKey, Arc<ColumnStore>> {
        &mut self.column_stores
    }
    #[inline]
    fn set_undo(&mut self, undo: Option<Box<UndoJournal>>) {
        self.undo = undo;
    }
    fn after_fold(&mut self, touched_edges: bool) {
        if touched_edges {
            self.invalidate_peer_counts();
        }
    }
}

impl FoldTarget for MappedGraph {
    #[inline]
    fn graph(&self) -> &StableDiGraph<NodeData, EdgeData> {
        self.inner()
    }
    #[inline]
    fn graph_mut(&mut self) -> &mut StableDiGraph<NodeData, EdgeData> {
        self.inner_mut()
    }
    #[inline]
    fn slot_mirror(&self) -> &SlotMirror {
        &self.slot_mirror
    }
    #[inline]
    fn slot_mirror_mut(&mut self) -> &mut SlotMirror {
        &mut self.slot_mirror
    }
    #[inline]
    fn column_stores_mut(&mut self) -> &mut FxHashMap<InternedKey, Arc<ColumnStore>> {
        &mut self.column_stores
    }
    #[inline]
    fn set_undo(&mut self, undo: Option<Box<UndoJournal>>) {
        self.undo = undo;
    }
    /// Unconditional: the matcher trusts a non-empty cached block verbatim, so
    /// a block built before the fold would answer `MATCH` with the pre-fold
    /// nodes and edges (a wrong result, not a slow one). Node and property
    /// edits and a swapped store map all reach the property indexes, not only
    /// edge edits.
    fn after_fold(&mut self, _touched_edges: bool) {
        self.invalidate_type_index();
        self.invalidate_property_index();
    }
}

/// Whether `base` can be shared behind an overlay rather than deep-copied.
///
/// The overlay allocates by the base's slot-mirror prediction (`forked` module
/// doc), so the mirror must know petgraph's free-list order. It does unless the
/// graph was adopted with holes from a `StableDiGraph` whose free-list order is
/// not observable (`SlotMirror::for_adopted_graph`); a `.kgl` load and a
/// storage-mode conversion know it.
pub(crate) fn can_fork(base: &impl FoldTarget) -> bool {
    base.slot_mirror().is_synced()
}

/// The shared graph an overlay reads through. Never mutated while the overlay
/// exists.
///
/// An enum rather than a type parameter so `GraphBackend::Forked` stays one
/// variant. The base is dispatched with a short branch that is the same on
/// every access of one overlay; the un-forked `Memory` backend, the case the
/// engine is tuned for, never goes through it.
pub(crate) enum ForkBase {
    Memory(Arc<MemoryGraph>),
    Mapped(Arc<MappedGraph>),
    Frozen(Arc<FrozenGraph>),
}

impl From<Arc<MemoryGraph>> for ForkBase {
    fn from(base: Arc<MemoryGraph>) -> Self {
        Self::Memory(base)
    }
}

impl From<Arc<MappedGraph>> for ForkBase {
    fn from(base: Arc<MappedGraph>) -> Self {
        Self::Mapped(base)
    }
}

impl From<Arc<FrozenGraph>> for ForkBase {
    fn from(base: Arc<FrozenGraph>) -> Self {
        Self::Frozen(base)
    }
}

/// Overlay entries a small base is allowed regardless of its size.
pub(crate) const VALVE_FLOOR: usize = 4096;

/// One base edge: `(slot, source, target, weight)`.
pub(crate) type BaseEdge<'a> = (EdgeIndex, NodeIndex, NodeIndex, &'a EdgeData);

/// One node's edges in one direction, in the base's own order.
pub(crate) enum BaseEdges<'a> {
    Heap(Edges<'a, EdgeData, Directed, u32>),
    Frozen(FrozenEdges<'a>),
}

impl<'a> BaseEdges<'a> {
    #[inline]
    pub(crate) fn next_edge(&mut self) -> Option<BaseEdge<'a>> {
        match self {
            Self::Heap(iter) => iter
                .next()
                .map(|er| (er.id(), er.source(), er.target(), er.weight())),
            Self::Frozen(iter) => iter
                .next()
                .map(|er| (er.id(), er.source(), er.target(), er.weight().lent())),
        }
    }
}

/// Every live base edge in ascending slot order.
pub(crate) enum BaseEdgeRefs<'a> {
    Heap(EdgeReferences<'a, EdgeData, u32>),
    Frozen(FrozenEdgeRefs<'a>),
}

impl<'a> BaseEdgeRefs<'a> {
    #[inline]
    pub(crate) fn next_edge(&mut self) -> Option<BaseEdge<'a>> {
        match self {
            Self::Heap(iter) => iter
                .next()
                .map(|er| (er.id(), er.source(), er.target(), er.weight())),
            Self::Frozen(iter) => iter
                .next()
                .map(|er| (er.id(), er.source(), er.target(), er.weight().lent())),
        }
    }
}

impl ForkBase {
    #[inline(always)]
    pub(crate) fn node_count(&self) -> usize {
        match self {
            Self::Memory(g) => g.inner().node_count(),
            Self::Mapped(g) => g.inner().node_count(),
            Self::Frozen(g) => g.live_node_count(),
        }
    }

    #[inline(always)]
    pub(crate) fn edge_count(&self) -> usize {
        match self {
            Self::Memory(g) => g.inner().edge_count(),
            Self::Mapped(g) => g.inner().edge_count(),
            Self::Frozen(g) => g.live_edge_count(),
        }
    }

    /// The base's own `node_bound()`: one past its highest live node.
    #[inline(always)]
    pub(crate) fn node_bound(&self) -> usize {
        match self {
            Self::Memory(g) => g.inner().node_bound(),
            Self::Mapped(g) => g.inner().node_bound(),
            Self::Frozen(g) => g.node_bound_len(),
        }
    }

    /// The base's own `edge_bound()`: one past its highest live edge.
    #[inline(always)]
    pub(crate) fn edge_bound(&self) -> usize {
        match self {
            Self::Memory(g) => EdgeIndexable::edge_bound(g.inner()),
            Self::Mapped(g) => EdgeIndexable::edge_bound(g.inner()),
            Self::Frozen(g) => g.edge_bound_len(),
        }
    }

    /// The node record as a plain borrow (every base lends its records).
    #[inline(always)]
    pub(crate) fn lent_node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        match self {
            Self::Memory(g) => g.inner().node_weight(idx),
            Self::Mapped(g) => g.inner().node_weight(idx),
            Self::Frozen(g) => g.lent_node_weight(idx),
        }
    }

    #[inline(always)]
    pub(crate) fn node_weight(&self, idx: NodeIndex) -> Option<RecordRef<'_, NodeData>> {
        match self {
            Self::Memory(g) => g.inner().node_weight(idx).map(RecordRef::borrowed),
            Self::Mapped(g) => g.inner().node_weight(idx).map(RecordRef::borrowed),
            Self::Frozen(g) => GraphRead::node_weight(&**g, idx),
        }
    }

    #[inline(always)]
    pub(crate) fn edge_weight(&self, idx: EdgeIndex) -> Option<RecordRef<'_, EdgeData>> {
        match self {
            Self::Memory(g) => g.inner().edge_weight(idx).map(RecordRef::borrowed),
            Self::Mapped(g) => g.inner().edge_weight(idx).map(RecordRef::borrowed),
            Self::Frozen(g) => GraphRead::edge_weight(&**g, idx),
        }
    }

    #[inline(always)]
    pub(crate) fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        match self {
            Self::Memory(g) => g.inner().edge_endpoints(idx),
            Self::Mapped(g) => g.inner().edge_endpoints(idx),
            Self::Frozen(g) => GraphRead::edge_endpoints(&**g, idx),
        }
    }

    #[inline]
    pub(crate) fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> BaseEdges<'_> {
        match self {
            Self::Memory(g) => BaseEdges::Heap(g.inner().edges_directed(idx, dir)),
            Self::Mapped(g) => BaseEdges::Heap(g.inner().edges_directed(idx, dir)),
            Self::Frozen(g) => BaseEdges::Frozen(FrozenEdges::new(g, idx, dir)),
        }
    }

    #[inline]
    pub(crate) fn edge_references(&self) -> BaseEdgeRefs<'_> {
        use petgraph::visit::IntoEdgeReferences;
        match self {
            Self::Memory(g) => BaseEdgeRefs::Heap(g.inner().edge_references()),
            Self::Mapped(g) => BaseEdgeRefs::Heap(g.inner().edge_references()),
            Self::Frozen(g) => BaseEdgeRefs::Frozen(FrozenEdgeRefs::new(g)),
        }
    }

    #[inline]
    pub(crate) fn column_stores(&self) -> &FxHashMap<InternedKey, Arc<ColumnStore>> {
        match self {
            Self::Memory(g) => &g.column_stores,
            Self::Mapped(g) => &g.column_stores,
            Self::Frozen(g) => &g.column_stores,
        }
    }

    #[inline]
    pub(crate) fn slot_mirror(&self) -> &SlotMirror {
        match self {
            Self::Memory(g) => &g.slot_mirror,
            Self::Mapped(g) => &g.slot_mirror,
            Self::Frozen(g) => g.slot_mirror(),
        }
    }

    /// Whether the base reports as a mapped backend (`Mapped` and `Frozen`).
    #[inline]
    pub(crate) fn is_mapped(&self) -> bool {
        !matches!(self, Self::Memory(_))
    }

    #[inline]
    pub(crate) fn is_frozen(&self) -> bool {
        matches!(self, Self::Frozen(_))
    }

    /// The frozen topology under the overlay, if that is the base.
    #[inline]
    pub(crate) fn frozen(&self) -> Option<&Arc<FrozenGraph>> {
        match self {
            Self::Frozen(g) => Some(g),
            _ => None,
        }
    }

    /// Another handle on the same base (an `Arc` bump, no copy).
    #[inline]
    pub(crate) fn share(&self) -> Self {
        match self {
            Self::Memory(g) => Self::Memory(Arc::clone(g)),
            Self::Mapped(g) => Self::Mapped(Arc::clone(g)),
            Self::Frozen(g) => Self::Frozen(Arc::clone(g)),
        }
    }

    /// A genuine deep copy of the base. A frozen base becomes a heap `Mapped`
    /// graph (what a whole-graph operation collapses an overlay into).
    pub(crate) fn deep_clone(&self) -> Folded {
        match self {
            Self::Memory(g) => Folded::Memory(g.deep_clone()),
            Self::Mapped(g) => Folded::Mapped(g.deep_clone()),
            Self::Frozen(g) => Folded::Mapped(g.to_mapped()),
        }
    }

    pub(crate) fn count_edges_filtered_impl(
        &self,
        node: petgraph::graph::NodeIndex,
        dir: petgraph::Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
        exclude_self: bool,
    ) -> Result<usize, String> {
        match self {
            Self::Memory(g) => g.count_edges_filtered_impl(
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
                exclude_self,
            ),
            Self::Mapped(g) => g.count_edges_filtered_impl(
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
                exclude_self,
            ),
            Self::Frozen(g) => g.count_edges_filtered_impl(
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
                exclude_self,
            ),
        }
    }
}

/// Run `$e` with `$g` bound to the base as a `&MemoryGraph` / `&MappedGraph` /
/// `&FrozenGraph`. Every arm must produce the same type.
macro_rules! base_read {
    ($base:expr, $g:ident => $e:expr) => {
        match &$base {
            $crate::graph::storage::fork_base::ForkBase::Memory($g) => {
                let $g = &**$g;
                $e
            }
            $crate::graph::storage::fork_base::ForkBase::Mapped($g) => {
                let $g = &**$g;
                $e
            }
            $crate::graph::storage::fork_base::ForkBase::Frozen($g) => {
                let $g = &**$g;
                $e
            }
        }
    };
}
pub(crate) use base_read;

/// A standalone heap graph an overlay was folded or copied into.
pub(crate) enum Folded {
    Memory(MemoryGraph),
    Mapped(MappedGraph),
}

impl Folded {
    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        match self {
            Self::Memory(g) => g.inner().node_count(),
            Self::Mapped(g) => g.inner().node_count(),
        }
    }
}

impl Serialize for Folded {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Memory(g) => g.serialize(serializer),
            Self::Mapped(g) => g.serialize(serializer),
        }
    }
}
