//! `ForkedGraph` — the writer-side copy-on-write overlay over a shared base.
//!
//! ## What this removes
//!
//! Holding any second `Arc<DirGraph>` — a lazy `ResultView`, a `freeze()`, a
//! `Session`, an open `Transaction` — made the next write deep-copy the entire
//! graph. Measured 2026-08-10 at 1M nodes: **36.3 ms** against a 3.0 µs
//! control, of which the backend row was **37.8 ms of a 41.6 ms**
//! `DirGraph::clone` on a plain graph. This module makes that row O(changes).
//!
//! ## The structural fact that forces this shape
//!
//! The copy-on-write has to be **writer-side**. A reader holds
//! `Arc<DirGraph>` and reads it as `&DirGraph` while the writer wants
//! `&mut DirGraph` to the same allocation — that is aliasing UB, and the only
//! escapes are a lock on the read path (over budget in the MATCH loop) or a
//! read guard that makes writes block on a held Python `ResultView` (a
//! deadlock hazard traded for a latency cliff). So the *reader's* graph is
//! left byte-for-byte untouched and the *writer* builds the delta.
//!
//! ## What the overlay covers, and what it deliberately does not
//!
//! | write | forked behaviour |
//! |---|---|
//! | node weight (`SET`, `REMOVE`, labels, title, id) | copied into `nodes` on first touch, O(1) |
//! | edge weight | copied into `edges` on first touch, O(1) |
//! | `add_node` (`CREATE`, `MERGE` insert) | appended to `nodes` at a predicted index, O(1) |
//! | column-store writes | the overlay owns its own map, O(types) `Arc` bumps at fork |
//! | **`add_edge` / `remove_node` / `remove_edge`** | **materialise, then proceed** |
//!
//! The last row is a deliberate scope boundary, not an oversight.
//! `StableDiGraph` threads adjacency through per-node linked lists, so any of
//! those three edits rewrites *existing* nodes' adjacency — which an overlay
//! cannot express without reimplementing `edges_directed`,
//! `edges_directed_filtered`, `edges_connecting`, `neighbors_*` and
//! `edge_references` as base⊕overlay chains behind their GAT iterator types.
//! That is a second body of work with its own correctness surface, and doing it
//! in the same change as the ownership flip would land both untested-in-
//! isolation.
//!
//! **Because no edit reaches base adjacency, the overlay needs no adjacency
//! chaining at all** — every edge and traversal method delegates straight to
//! the base, and only `node_indices` gains a variant. That is the whole reason
//! this module is a few hundred lines rather than a few thousand, and it is why
//! the read path is unchanged for everything except a node-weight probe.
//!
//! `materialise` is exactly today's cost — a base deep clone plus the overlay
//! replayed — so a topology write while a reader is held is no worse than
//! before this module existed, and everything else is O(changes).
//!
//! ## Slot identity, and why forking is *conditional*
//!
//! Rollback guarantees a node or edge comes back on the exact
//! `NodeIndex`/`EdgeIndex` it vacated (`dir_graph/rollback.rs`), and
//! `NodeIndex` is the key of every index structure on `DirGraph`. So the
//! indices the overlay hands out must be the indices the base will produce when
//! the overlay is folded back in. `StableGraph::add_node` reuses free-list
//! slots and offers no index-controlled insertion, so this holds only when the
//! base's node free list is **empty** — then appends are contiguous from
//! `node_bound()` and a sequential replay reproduces them exactly.
//!
//! [`SlotMirror`](super::slot_mirror) is what makes that checkable
//! ([`can_fork`]), and its `debug_assert` inside `note_node_added` is what
//! proves it: every `add_node` in [`ForkedGraph::apply_overlay`] — the
//! fold-back path — runs through the same seam and re-checks the prediction. A
//! base that cannot be forked cheaply falls back to the deep clone, which is
//! slower and never wrong — the same fail-safe direction
//! `rollback::journal_covers` takes.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::NodeIndexable;
use rustc_hash::FxHashMap;

use crate::datatypes::Value;
use crate::graph::core::iterators::GraphNodeIndices;
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::undo::UndoJournal;
use crate::graph::storage::{GraphRead, GraphWrite, MemoryGraph};

/// A writer's copy-on-write view over a base another holder is still reading.
pub struct ForkedGraph {
    /// The reader's graph. **Never mutated while this exists** — that is the
    /// one unforgivable failure mode in this design, because a reader observing
    /// a writer's edit is silent and unrecoverable. `pin_base_generation` is
    /// the golden-snapshot guard for it.
    base: Arc<MemoryGraph>,
    /// Node weights that diverge from the base: copies taken on first write,
    /// plus every node appended since the fork. Keyed by raw node index.
    nodes: FxHashMap<u32, NodeData>,
    /// Edge weights that diverge from the base. No edge is ever *added* here —
    /// `add_edge` materialises first (see the module doc).
    edges: FxHashMap<u32, EdgeData>,
    /// How many nodes have been appended past `base.node_bound()`. Appends are
    /// contiguous by construction (see [`can_fork`]), which is what keeps
    /// `node_indices` globally ascending without a merge.
    appended: u32,
    /// The overlay's own column-store map, seeded with one `Arc` bump per type
    /// at fork time. Complete, so reads never chain into the base for it.
    column_stores: FxHashMap<InternedKey, Arc<ColumnStore>>,
    /// Own, never shared with the base. Sharing would let the writer's
    /// post-mutation counts be observed through the reader's snapshot (D2 R4).
    peer_counts: RwLock<HashMap<u64, Arc<super::MemoryPeerCounts>>>,
    /// Statement-scoped inverse-op buffer. Lives *here*, so every undo entry
    /// reverses through this backend's `GraphWrite` and therefore lands in the
    /// overlay — never in the shared base (D2 R3).
    undo: Option<Box<UndoJournal>>,
    /// Continues the base's mirror. Predictions must stay in step across the
    /// fork or the fold-back would allocate different slots.
    slot_mirror: super::slot_mirror::SlotMirror,
}

/// Whether `base` can be shared behind an overlay rather than deep-copied.
///
/// The condition is exactly the slot-identity precondition from the module
/// doc: the free lists must be provably empty, so appended indices are
/// contiguous from the bounds and a sequential replay reproduces them. A graph
/// that has ever deleted a node or edge and not been vacuumed fails this and
/// keeps the deep clone.
pub(crate) fn can_fork(base: &MemoryGraph) -> bool {
    let node_bound = base.inner().node_bound();
    let edge_bound = petgraph::visit::EdgeIndexable::edge_bound(base.inner());
    base.slot_mirror.predict_next_node(node_bound) == Some(NodeIndex::new(node_bound))
        && base.slot_mirror.predict_next_edge(edge_bound) == Some(EdgeIndex::new(edge_bound))
}

impl ForkedGraph {
    /// Fork `base` — O(types), no node or edge is copied.
    pub(crate) fn new(base: Arc<MemoryGraph>) -> Self {
        let column_stores = base.column_stores.clone();
        let slot_mirror = base.slot_mirror.clone();
        Self {
            base,
            nodes: FxHashMap::default(),
            edges: FxHashMap::default(),
            appended: 0,
            column_stores,
            // Cold on purpose: correct-but-cold beats a cache shared with a
            // reader's snapshot (D2 R4).
            peer_counts: RwLock::new(HashMap::new()),
            undo: None,
            slot_mirror,
        }
    }

    /// The first index this overlay appended, i.e. the base's node bound.
    #[inline]
    fn append_floor(&self) -> usize {
        self.base.inner().node_bound()
    }

    /// How many node weights this overlay holds — the only nodes a clone of it
    /// duplicates, and what the `BACKEND_CLONE_NODES` oracle counts for a fork
    /// of a fork.
    #[cfg(test)]
    #[inline]
    pub(crate) fn overlay_node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Replay the overlay into `target`, which must be a graph in the base's
    /// exact pre-fork state.
    ///
    /// **This is the fold-back path, and the slot-identity proof lives here.**
    /// Appended nodes go back through `GraphWrite::add_node`, whose
    /// `SlotMirror::note_node_added` debug-asserts that the index petgraph
    /// allocated is the index the mirror predicted — the same assertion that
    /// ran when the overlay handed the index out. The explicit `assert_eq!`
    /// below is the release-profile half of the same statement: getting a
    /// different index would silently mis-key every `DirGraph` index that
    /// recorded the overlay's number, which is a data-corruption bug rather
    /// than a crash, so it is worth an unconditional check on a path that runs
    /// once per compaction rather than once per write.
    fn apply_overlay(&mut self, target: &mut MemoryGraph) {
        let floor = target.inner().node_bound();
        // Appended nodes first and in index order, so the sequential
        // reallocation reproduces the overlay's contiguous run.
        for offset in 0..self.appended {
            let idx = floor as u32 + offset;
            let data = self
                .nodes
                .remove(&idx)
                .expect("an appended overlay index must carry its node weight");
            let actual = GraphWrite::add_node(target, data);
            assert_eq!(
                actual.index() as u32,
                idx,
                "fold-back allocated node {} where the overlay handed out {idx}; \
                 slot identity is broken and every index keyed on the overlay's \
                 number is now wrong (see storage/forked.rs)",
                actual.index()
            );
        }
        // Then the copy-on-write weights, which are pure overwrites.
        for (idx, data) in self.nodes.drain() {
            if let Some(slot) = target
                .inner_mut()
                .node_weight_mut(NodeIndex::new(idx as usize))
            {
                *slot = data;
            }
        }
        for (idx, data) in self.edges.drain() {
            if let Some(slot) = target
                .inner_mut()
                .edge_weight_mut(EdgeIndex::new(idx as usize))
            {
                *slot = data;
            }
        }
        target.column_stores = std::mem::take(&mut self.column_stores);
        target.undo = self.undo.take();
        self.appended = 0;
    }

    /// Fold into the base when this writer is the only holder left, and return
    /// the collapsed backend. `Err(self)` when a reader is still outstanding.
    ///
    /// The reader dropping is exactly what makes `Arc::get_mut` succeed, so the
    /// common "hold a view, write, drop the view, write again" pattern
    /// self-heals on the next write with no timer and no bookkeeping.
    pub(crate) fn try_compact(mut self: Box<Self>) -> Result<MemoryGraph, Box<Self>> {
        if Arc::get_mut(&mut self.base).is_none() {
            return Err(self);
        }
        // Sole owner: take the base out and fold into it in place. No node or
        // edge is copied — this is the payoff, and it is why compaction is
        // O(changes) rather than O(V+E).
        let mut owned = Arc::try_unwrap(std::mem::replace(
            &mut self.base,
            Arc::new(MemoryGraph::new()),
        ))
        .unwrap_or_else(|_| unreachable!("get_mut proved unique ownership"));
        self.apply_overlay(&mut owned);
        Ok(owned)
    }

    /// Deep-copy the base and fold into the copy. Used when a write cannot be
    /// expressed in the overlay while a reader is still holding the base — the
    /// pre-D2 cost, paid only on that write.
    pub(crate) fn materialise(&mut self) -> MemoryGraph {
        #[cfg(test)]
        super::backend::note_nodes_copied(self.base.inner().node_count());
        let mut owned = self.base.deep_clone();
        self.apply_overlay(&mut owned);
        owned
    }

    /// A standalone graph equal to what this overlay reads as, without
    /// disturbing it. For `Serialize` and for the petgraph-typed algorithm
    /// path, both of which need one concrete `StableDiGraph`.
    pub(crate) fn to_memory_graph(&self) -> MemoryGraph {
        let mut clone = ForkedGraph {
            base: Arc::clone(&self.base),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            appended: self.appended,
            column_stores: self.column_stores.clone(),
            peer_counts: RwLock::new(HashMap::new()),
            undo: None,
            slot_mirror: self.slot_mirror.clone(),
        };
        let mut owned = self.base.deep_clone();
        clone.apply_overlay(&mut owned);
        owned
    }

    // ── undo journal (mirrors MemoryGraph's, but installed on the overlay) ──

    #[inline]
    pub(crate) fn begin_undo(&mut self) {
        self.undo = Some(Box::new(UndoJournal::new()));
    }

    #[inline]
    pub(crate) fn take_undo(&mut self) -> Option<Box<UndoJournal>> {
        self.undo.take()
    }

    #[inline]
    pub(crate) fn undo_journal_mut(&mut self) -> Option<&mut UndoJournal> {
        self.undo.as_deref_mut()
    }

    #[inline]
    fn invalidate_peer_counts(&mut self) {
        if let Ok(mut cache) = self.peer_counts.write() {
            cache.clear();
        }
    }

    /// The overlay's copy of a base node, taken on first write.
    ///
    /// This is the single point where a base node stops being shared, and the
    /// reason the base is never mutated: every `&mut NodeData` this backend
    /// hands out points into `self.nodes`.
    #[inline]
    fn cow_node(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        let raw = idx.index() as u32;
        if !self.nodes.contains_key(&raw) {
            let base = self.base.inner().node_weight(idx)?.clone();
            self.nodes.insert(raw, base);
        }
        self.nodes.get_mut(&raw)
    }

    #[inline]
    fn cow_edge(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        let raw = idx.index() as u32;
        if !self.edges.contains_key(&raw) {
            let base = self.base.inner().edge_weight(idx)?.clone();
            self.edges.insert(raw, base);
        }
        self.edges.get_mut(&raw)
    }

    // ── undo capture (the overlay's own, so entries reverse into the overlay) ──

    /// Clone `idx`'s current weight into the journal as its pre-statement
    /// state. Reads through [`GraphRead::node_weight`], i.e. overlay-then-base,
    /// so the pre-image is what *this writer* would have read — not what the
    /// base holds, which may already differ if an earlier statement wrote it.
    #[cold]
    fn capture_node_weight(&mut self, idx: NodeIndex) {
        let current = GraphRead::node_weight(self, idx).cloned();
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_weight(idx, || current);
        }
    }

    #[cold]
    fn capture_edge_weight(&mut self, idx: EdgeIndex) {
        let current = GraphRead::edge_weight(self, idx).cloned();
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_weight(idx, || current);
        }
    }

    /// Journal the pre-image a property write is about to overwrite — the
    /// overlay's counterpart of `impl_heap_pre_image_capture!`. Same two
    /// shapes, same reasoning (see that macro): a row-storage node journals a
    /// `NodeData` clone, a columnar one journals the type's store `Arc`.
    #[inline]
    fn capture_property_pre_image(&mut self, idx: NodeIndex) {
        if self.undo.is_none() {
            return;
        }
        let Some(nd) = GraphRead::node_weight(self, idx) else {
            return;
        };
        match nd.properties.columnar_row_id() {
            None => self.capture_node_weight(idx),
            Some(_) => {
                let type_key = nd.node_type;
                let prior = self.column_stores.get(&type_key).map(Arc::clone);
                if let Some(journal) = self.undo.as_deref_mut() {
                    journal.note_columnar_fork(type_key, || prior);
                }
            }
        }
    }

    /// The row id of a columnar node, or `None` for row storage. Read before
    /// any copy-on-write so a columnar property write — which changes the
    /// store, never the node — does not needlessly copy a `NodeData`.
    #[inline]
    fn columnar_row_of(&self, idx: NodeIndex) -> Option<(InternedKey, u32)> {
        let nd = GraphRead::node_weight(self, idx)?;
        nd.properties
            .columnar_row_id()
            .map(|row_id| (nd.node_type, row_id))
    }

    #[inline]
    pub(crate) fn base_stable_digraph(&self) -> &StableDiGraph<NodeData, EdgeData> {
        self.base.inner()
    }
}

impl Clone for ForkedGraph {
    /// Forking a fork keeps the same base and copies only the delta.
    fn clone(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            appended: self.appended,
            column_stores: self.column_stores.clone(),
            peer_counts: RwLock::new(HashMap::new()),
            undo: None,
            slot_mirror: self.slot_mirror.clone(),
        }
    }
}

impl std::fmt::Debug for ForkedGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ForkedGraph {{ base: {} nodes / {} edges, overlay: {} node weights, \
             {} edge weights, {} appended }}",
            self.base.inner().node_count(),
            self.base.inner().edge_count(),
            self.nodes.len(),
            self.edges.len(),
            self.appended
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────
// GraphRead — node state is overlay-then-base; everything edge-shaped is the
// base's, because no edit in this backend can reach base adjacency.
// ──────────────────────────────────────────────────────────────────────────

impl GraphRead for ForkedGraph {
    type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = <MemoryGraph as GraphRead>::EdgeIndicesIter<'a>;
    type EdgesIter<'a> = <MemoryGraph as GraphRead>::EdgesIter<'a>;
    type EdgeReferencesIter<'a> = <MemoryGraph as GraphRead>::EdgeReferencesIter<'a>;
    type EdgesConnectingIter<'a> = <MemoryGraph as GraphRead>::EdgesConnectingIter<'a>;
    type NeighborsIter<'a> = <MemoryGraph as GraphRead>::NeighborsIter<'a>;

    #[inline]
    fn node_count(&self) -> usize {
        self.base.inner().node_count() + self.appended as usize
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.base.inner().edge_count()
    }

    #[inline]
    fn node_bound(&self) -> usize {
        self.append_floor() + self.appended as usize
    }

    #[inline]
    fn is_memory(&self) -> bool {
        true
    }

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        match self.nodes.get(&(idx.index() as u32)) {
            Some(data) => Some(data),
            None => self.base.inner().node_weight(idx),
        }
    }

    #[inline]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        self.node_weight(idx).map(|n| n.node_type)
    }

    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        self.node_view(idx)?.get_value(key)
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        Some(self.node_view(idx)?.id().into_owned())
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        Some(self.node_view(idx)?.title().into_owned())
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        self.node_view(idx)?.str_prop_eq(key, target)
    }

    // Every method below is pure base delegation: nothing this backend can
    // write reaches base adjacency (module doc), so an edge-shaped answer from
    // the base is the whole answer.

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> Self::EdgesIter<'_> {
        GraphRead::edges_directed_filtered(&*self.base, idx, dir, conn_type_filter)
    }

    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        GraphRead::edge_endpoint_keys(&*self.base)
    }

    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: petgraph::Direction,
        deadline: Option<std::time::Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        GraphRead::count_edges_grouped_by_peer(&*self.base, conn_type, dir, deadline)
    }

    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: petgraph::Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
    ) -> Result<usize, String> {
        GraphRead::count_edges_filtered(
            &*self.base,
            node,
            dir,
            conn_type,
            other_node_type,
            deadline,
        )
    }

    #[inline]
    fn column_store(&self, type_key: InternedKey) -> Option<&Arc<ColumnStore>> {
        self.column_stores.get(&type_key)
    }

    fn column_stores_iter(
        &self,
    ) -> Box<dyn Iterator<Item = (InternedKey, &Arc<ColumnStore>)> + '_> {
        Box::new(self.column_stores.iter().map(|(k, v)| (*k, v)))
    }

    /// Base indices then appended ones. Appends are contiguous above every base
    /// index (see [`can_fork`]), so the chain is globally ascending and scan
    /// order — which `type_indices` bucket order and the rollback fidelity
    /// tests both pin — is unchanged.
    #[inline]
    fn node_indices(&self) -> Self::NodeIndicesIter<'_> {
        GraphNodeIndices::Forked {
            base: Box::new(self.base.inner().node_indices()),
            appended: self.append_floor()..self.node_bound(),
        }
    }

    #[inline]
    fn edge_indices(&self) -> Self::EdgeIndicesIter<'_> {
        GraphRead::edge_indices(&*self.base)
    }

    #[inline]
    fn edge_references(&self) -> Self::EdgeReferencesIter<'_> {
        GraphRead::edge_references(&*self.base)
    }

    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        GraphRead::edge_weights(&*self.base)
    }

    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: petgraph::Direction) -> Self::EdgesIter<'_> {
        GraphRead::edges_directed(&*self.base, idx, dir)
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> Self::EdgesIter<'_> {
        GraphRead::edges(&*self.base, idx)
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> Self::EdgesConnectingIter<'_> {
        GraphRead::edges_connecting(&*self.base, a, b)
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        match self.edges.get(&(idx.index() as u32)) {
            Some(data) => Some(data),
            None => self.base.inner().edge_weight(idx),
        }
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        GraphRead::find_edge(&*self.base, a, b)
    }

    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        GraphRead::edge_endpoints(&*self.base, idx)
    }

    #[inline]
    fn neighbors_directed(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
    ) -> Self::NeighborsIter<'_> {
        GraphRead::neighbors_directed(&*self.base, idx, dir)
    }

    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> Self::NeighborsIter<'_> {
        GraphRead::neighbors_undirected(&*self.base, idx)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// GraphWrite — every mutation lands in the overlay. The three that cannot be
// expressed here are intercepted one level up, in `GraphBackend`, which is the
// only place that can replace a `Forked` with a `Memory`.
// ──────────────────────────────────────────────────────────────────────────

impl GraphWrite for ForkedGraph {
    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        if self.undo.is_some() {
            self.capture_node_weight(idx);
        }
        self.cow_node(idx)
    }

    #[inline]
    fn node_weight_mut_silent(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        self.cow_node(idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        self.invalidate_peer_counts();
        if self.undo.is_some() {
            self.capture_edge_weight(idx);
        }
        self.cow_edge(idx)
    }

    #[inline]
    fn install_column_store(&mut self, type_key: InternedKey, store: Arc<ColumnStore>) {
        self.column_stores.insert(type_key, store);
    }

    #[inline]
    fn column_store_mut(&mut self, type_key: InternedKey) -> Option<&mut Arc<ColumnStore>> {
        self.column_stores.get_mut(&type_key)
    }

    #[inline]
    fn take_column_store(&mut self, type_key: InternedKey) -> Option<Arc<ColumnStore>> {
        self.column_stores.remove(&type_key)
    }

    #[inline]
    fn clear_column_stores(&mut self) {
        self.column_stores.clear();
    }

    fn set_node_property(&mut self, idx: NodeIndex, key: InternedKey, value: Value) {
        self.capture_property_pre_image(idx);
        // Columnar: the value lives in the store, the node only holds a row id,
        // so nothing about the node diverges and no `NodeData` is copied.
        //
        // `Arc::make_mut` here sees the base's handle as well as the overlay's,
        // so it copies the store — which is exactly right: the reader's base
        // must keep the store it was forked with. That copy is the same
        // per-statement pre-image cost a non-forked columnar write already pays
        // (D1 Phase 4/5), not a new one this module introduces.
        if let Some((type_key, row_id)) = self.columnar_row_of(idx) {
            if let Some(store) = self.column_stores.get_mut(&type_key) {
                Arc::make_mut(store).set(row_id, key, &value, None);
            }
            return;
        }
        if let Some(nd) = self.cow_node(idx) {
            nd.properties.insert(key, value);
        }
    }

    fn set_node_property_if_absent(&mut self, idx: NodeIndex, key: InternedKey, value: Value) {
        if GraphRead::node_has_property(self, idx, key) {
            return;
        }
        GraphWrite::set_node_property(self, idx, key, value);
    }

    fn remove_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        let previous = GraphRead::get_node_property(self, idx, key);
        self.capture_property_pre_image(idx);
        if let Some((type_key, row_id)) = self.columnar_row_of(idx) {
            if let Some(store) = self.column_stores.get_mut(&type_key) {
                Arc::make_mut(store).set(row_id, key, &Value::Null, None);
            }
            return previous;
        }
        self.cow_node(idx)?.properties.remove(key)
    }

    fn clear_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        GraphWrite::remove_node_property(self, idx, key)
    }

    fn replace_node_properties(&mut self, idx: NodeIndex, pairs: Vec<(InternedKey, Value)>) {
        self.capture_property_pre_image(idx);
        if self.columnar_row_of(idx).is_some() {
            for (key, value) in pairs {
                GraphWrite::set_node_property(self, idx, key, value);
            }
            return;
        }
        if let Some(nd) = self.cow_node(idx) {
            nd.properties.replace_all(pairs);
        }
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        let node_type = data.node_type;
        let bound_before = self.node_bound();
        let idx = NodeIndex::new(bound_before);
        self.nodes.insert(idx.index() as u32, data);
        self.appended += 1;
        // Keeps the mirror in step with the indices this overlay hands out, so
        // the fold-back's own `add_node` predicts the same run.
        self.slot_mirror.note_node_added(bound_before, idx);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_added(idx, node_type);
        }
        idx
    }

    /// Unreachable: `GraphBackend` materialises before dispatching any of the
    /// three adjacency-mutating writes here (see the module doc). The panic is
    /// the assertion that the interception is complete, not a stub — reaching
    /// it would mean a base adjacency edit was about to happen under a live
    /// reader.
    fn remove_node(&mut self, _idx: NodeIndex) -> Option<NodeData> {
        unreachable!("forked backend must be materialised before remove_node")
    }

    fn add_edge(&mut self, _a: NodeIndex, _b: NodeIndex, _data: EdgeData) -> EdgeIndex {
        unreachable!("forked backend must be materialised before add_edge")
    }

    fn remove_edge(&mut self, _idx: EdgeIndex) -> Option<EdgeData> {
        unreachable!("forked backend must be materialised before remove_edge")
    }
}
