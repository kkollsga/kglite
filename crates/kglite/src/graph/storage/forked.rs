//! `ForkedGraph` — the writer-side copy-on-write overlay over a shared base.
//!
//! ## What this removes
//!
//! Holding any second `Arc<DirGraph>` — a lazy `ResultView`, a `freeze()`, a
//! `Session`, an open `Transaction` — made the next write deep-copy the entire
//! graph. Measured 2026-08-10 at 1M nodes: **36.3 ms** against a 3.0 µs
//! control, of which the backend row was **37.8 ms of a 41.6 ms**
//! `DirGraph::clone` on a plain graph. This module makes that row O(changes),
//! for node writes and for adjacency writes alike.
//!
//! The base is a `Memory` or a `Mapped` backend ([`ForkBase`]); both hold the
//! same petgraph, so the overlay is identical over either. Only the fold-back
//! differs: a `Mapped` target also drops its lazy type and property indexes
//! (`FoldTarget::after_fold`).
//!
//! The third base is a `Frozen` topology served from a `.kgl`'s raw sections,
//! which is how a mapped graph takes writes without copying itself. It has no
//! petgraph, so the overlay over it **never folds back**: it persists, its
//! heap is the size of the writes, and a save writes the merged graph out and
//! rebases the graph onto that file (`io/file/rebase.rs`). A whole-graph
//! operation collapses it through [`ForkedGraph::materialise`] into a heap
//! `Mapped` graph; its size never does. Every map in the delta is a shared
//! paged container (`shared_tree.rs`), so a fork of the overlay costs the same
//! at any number of writes.
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
//! ## What the overlay holds
//!
//! | write | forked behaviour |
//! |---|---|
//! | node weight (`SET`, `REMOVE`, labels, title, id) | copied into `nodes` on first touch, O(1) |
//! | `add_node` (`CREATE`, `MERGE` insert) | stored in `nodes` at a predicted slot, O(1) |
//! | `remove_node` | the slot joins `dead` (a base node) or leaves `extra` (an overlay node); incident edges go first |
//! | `add_edge` | stored in the [`EdgeLayer`] at a predicted slot, O(1) |
//! | `remove_edge` | a tombstone (base edge) or an unlink (overlay edge), O(1) |
//! | edge weight (`SET r.p`) | copied into the edge layer on first touch, O(1) |
//! | column-store writes | the overlay owns its own map, O(types) `Arc` bumps at fork |
//!
//! `StableDiGraph` threads adjacency through per-node linked lists with head
//! insertion, so the overlay never touches the base's lists: the edges a node
//! reads are "the overlay's, newest first, then the base's minus the removed"
//! (`forked_edges`), which is exactly the folded graph's order. Reads chain the
//! two behind the shared `Graph*` iterator enums (`forked_edge_iters`); a fork
//! that wrote only nodes hands out the base's own iterators, so the no-overlay
//! read path is unchanged.
//!
//! ## Slot identity, and why forking is *conditional*
//!
//! Rollback guarantees a node or edge comes back on the exact
//! `NodeIndex`/`EdgeIndex` it vacated (`dir_graph/rollback.rs`), and
//! `NodeIndex` is the key of every index structure on `DirGraph`. So the
//! indices the overlay hands out must be the indices the base will produce when
//! the overlay is folded back in. `StableGraph::add_node` and `add_edge` reuse
//! free-list slots (LIFO) and offer no index-controlled insertion, so the
//! overlay allocates exactly as they would: each index is the
//! [`SlotMirror`](super::slot_mirror) prediction — the free-list head while the
//! list is non-empty, then the next slot past every one ever allocated.
//!
//! Removals feed the same free lists, so a fold cannot replay "the appended
//! nodes in allocation order" any more: it replays the **operation log**
//! (`Op`) — every add and remove, in the order the writer issued them —
//! against the base's own `add_node` / `add_edge` / `remove_edge` /
//! `remove_node`, and each add must come back on the slot the overlay handed
//! out. A log that reproduces every slot reproduces both free lists and the
//! adjacency order, because the same operations ran against the same state.
//! Issue #195 was an overlay that appended past `node_bound()` while slots
//! were still listed.
//!
//! That needs a mirror whose free-list order is known, which is what
//! [`can_fork`](super::fork_base::can_fork) checks. The fold simulates the log against a copy of the
//! target's mirror before anything changes (`Overlay::check`), then replays it
//! checking each slot petgraph actually allocates. A slot mismatch while the
//! replay has only added things is undone and the overlay keeps serving; once a
//! removal has run the replay cannot be reversed (re-adding an edge puts it at
//! the head of its lists, not where it was), so a mismatch there can only
//! panic. It cannot happen while the mirror agrees with petgraph, which the
//! debug assertion in `SlotMirror::note_*_added` checks on every insert the
//! test suites perform. A base that cannot be forked falls back to the deep
//! clone, which is slower and never wrong — the same fail-safe direction
//! `rollback::journal_covers` takes.

use std::collections::HashMap;
use std::sync::Arc;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeIndexable, NodeIndexable};
use rustc_hash::FxHashMap;

use crate::datatypes::Value;
use crate::graph::core::iterators::{
    ForkedMergedIndices, GraphEdgeIndices, GraphEdgeReferences, GraphEdges, GraphEdgesConnecting,
    GraphNeighbors, GraphNodeIndices,
};
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::fork_base::{base_read, FoldTarget, Folded, ForkBase, VALVE_FLOOR};
use crate::graph::storage::forked_edge_iters::{
    ForkedEdgeRefs, ForkedEdges, ForkedEdgesConnecting, ForkedNeighbors,
};
use crate::graph::storage::forked_edges::{EdgeLayer, OverlayBits};
use crate::graph::storage::forked_slots::ExtraSlots;
use crate::graph::storage::frozen::FrozenNodeIndices;
use crate::graph::storage::property_storage::PropertyStorage;
use crate::graph::storage::record_ref::RecordRef;
use crate::graph::storage::shared_tree::{OpLog, PagedMap};
use crate::graph::storage::undo::{ColumnarPreImages, ColumnarWrite, OverlayResidue, UndoJournal};
use crate::graph::storage::{GraphRead, GraphWrite, MappedGraph, MemoryGraph};

/// A writer's copy-on-write view over a base another holder is still reading.
pub struct ForkedGraph {
    /// The reader's graph. **Never mutated while this exists** — that is the
    /// one unforgivable failure mode in this design, because a reader observing
    /// a writer's edit is silent and unrecoverable. The over-specified
    /// fingerprint in `dir_graph::rollback_tests::held_reader` is the
    /// golden-snapshot guard for it.
    base: ForkBase,
    /// Everything the fold-back replays into the base.
    delta: Delta,
    /// The overlay's own column-store map, seeded with one `Arc` bump per type
    /// at fork time. Complete, so reads never chain into the base for it.
    column_stores: FxHashMap<InternedKey, Arc<ColumnStore>>,
    /// Statement-scoped inverse-op buffer. Lives *here*, so every undo entry
    /// reverses through this backend's `GraphWrite` and therefore lands in the
    /// overlay — never in the shared base.
    undo: Option<Box<UndoJournal>>,
    /// Continues the base's mirror. Predictions must stay in step across the
    /// fork or the fold-back would allocate different slots.
    slot_mirror: super::slot_mirror::SlotMirror,
}

/// One topology operation, in the order the writer issued it.
///
/// Weights are not logged: a node or edge that is live at the end has its
/// final weight in the delta, and one that is not is replayed with a
/// placeholder.
#[derive(Clone, Copy)]
enum Op {
    AddNode(u32),
    /// The node's incident edges were removed by `RemoveEdge` ops before this.
    RemoveNode(u32),
    AddEdge {
        slot: u32,
        src: u32,
        dst: u32,
    },
    RemoveEdge(u32),
}

/// The overlay's node and edge delta, kept apart from the base so a fold can
/// write into the base the overlay sits on ([`ForkedGraph::fold_in_place`]).
#[derive(Clone, Default)]
struct Delta {
    /// Node weights that diverge from the base: copies taken on first write,
    /// plus every live node the overlay allocated. Keyed by raw node index.
    nodes: OverlayNodes,
    /// Live slots the overlay allocated where the base holds no live node: a
    /// slot the base had vacated, or one past every base slot. Sorted, for
    /// merging into `node_indices`.
    extra: ExtraSlots,
    /// Base nodes the overlay removed and has not re-created.
    dead: OverlayBits,
    dead_count: usize,
    edges: EdgeLayer,
    ops: OpLog<Op>,
}

impl Delta {
    /// An empty delta; over a frozen base its node bitmaps are chunked so a
    /// fork shares them.
    fn new(chunked: bool) -> Self {
        let bits = || {
            if chunked {
                OverlayBits::chunked()
            } else {
                OverlayBits::default()
            }
        };
        Delta {
            nodes: OverlayNodes {
                map: PagedMap::default(),
                present: bits(),
            },
            dead: bits(),
            ..Delta::default()
        }
    }

    /// An empty delta of the same kind.
    fn emptied(&self) -> Self {
        Delta {
            nodes: OverlayNodes {
                map: PagedMap::default(),
                present: self.nodes.present.emptied(),
            },
            dead: self.dead.emptied(),
            ..Delta::default()
        }
    }
}

/// The overlay's node weights, keyed by raw index, with a bitmap in front.
///
/// Every read of a node the overlay does not hold — nearly all of them in a
/// scan — would otherwise pay a hash miss, and `FxHash` misses cost about 2.6×
/// more when the held keys are scattered (reused free-list slots) than when
/// they are one contiguous run (measured 2026-10-05: 2.09 vs 0.80 ms per 1M
/// lookups at 100 keys). One bit test answers those reads instead. The bitmap
/// is allocated on the first insert and grows to the highest index held.
#[derive(Clone, Default)]
struct OverlayNodes {
    map: PagedMap<NodeData>,
    present: OverlayBits,
}

impl OverlayNodes {
    #[inline]
    fn holds(&self, idx: u32) -> bool {
        self.present.holds(idx)
    }

    #[inline]
    fn get(&self, idx: u32) -> Option<&NodeData> {
        if self.holds(idx) {
            self.map.get(idx)
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(&mut self, idx: u32) -> Option<&mut NodeData> {
        self.map.get_mut(idx)
    }

    fn insert(&mut self, idx: u32, data: NodeData) {
        self.present.set(idx);
        self.map.insert(idx, data);
    }

    fn remove(&mut self, idx: u32) -> Option<NodeData> {
        self.present.clear(idx);
        self.map.remove(idx)
    }

    fn drain(&mut self) -> Vec<(u32, NodeData)> {
        self.present = self.present.emptied();
        self.map.drain()
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

/// A [`ForkedGraph`]'s overlay state, borrowed apart from its base so a fold
/// can write into the base the overlay sits on ([`ForkedGraph::fold_in_place`]).
struct Overlay<'a> {
    delta: &'a mut Delta,
    column_stores: &'a mut FxHashMap<InternedKey, Arc<ColumnStore>>,
    undo: &'a mut Option<Box<UndoJournal>>,
}

/// A node that stands in for one the log adds and later removes: the final
/// weights are written after the replay, so only the slot matters here.
fn placeholder_node() -> NodeData {
    NodeData {
        ident: None,
        node_type: InternedKey::default(),
        properties: PropertyStorage::Map(Box::default()),
    }
}

fn placeholder_edge() -> EdgeData {
    EdgeData {
        connection_type: InternedKey::default(),
        properties: thin_vec::ThinVec::new(),
    }
}

impl Overlay<'_> {
    /// Check that folding into `target` reproduces every slot the overlay
    /// handed out, without mutating anything.
    ///
    /// Simulates the operation log against a copy of `target`'s slot mirror:
    /// each add must be predicted onto the slot the overlay chose, and every
    /// live overlay node must carry its weight. A refusal here leaves both the
    /// overlay and `target` untouched (issue #195).
    fn check<T: FoldTarget>(&self, target: &T) -> Result<(), String> {
        for idx in self.delta.extra.iter() {
            if !self.delta.nodes.holds(idx) {
                return Err(format!("appended node {idx} has no weight in the overlay"));
            }
        }
        let mut mirror = target.slot_mirror().clone();
        // What `node_bound()` / `edge_bound()` would read before each add.
        let mut node_bound = target.graph().node_bound();
        let mut edge_bound = EdgeIndexable::edge_bound(target.graph());
        for op in self.delta.ops.iter() {
            match *op {
                Op::AddNode(idx) => match mirror.predict_next_node(node_bound) {
                    Some(predicted) if predicted.index() == idx as usize => {
                        mirror.note_node_added(node_bound, predicted);
                        node_bound = node_bound.max(idx as usize + 1);
                    }
                    other => {
                        return Err(format!(
                            "the overlay handed out node {idx}, but the fold target \
                             would allocate {other:?}"
                        ));
                    }
                },
                Op::AddEdge { slot, .. } => match mirror.predict_next_edge(edge_bound) {
                    Some(predicted) if predicted.index() == slot as usize => {
                        mirror.note_edge_added(edge_bound, predicted);
                        edge_bound = edge_bound.max(slot as usize + 1);
                    }
                    other => {
                        return Err(format!(
                            "the overlay handed out edge {slot}, but the fold target \
                             would allocate {other:?}"
                        ));
                    }
                },
                Op::RemoveEdge(slot) => mirror.note_edge_removed(EdgeIndex::new(slot as usize)),
                Op::RemoveNode(idx) => {
                    mirror.note_node_removed(NodeIndex::new(idx as usize), std::iter::empty());
                }
            }
        }
        Ok(())
    }

    /// Replay the overlay into `target`, which must be a graph in the base's
    /// exact pre-fork state. All or nothing for the failures a refusal can
    /// name: on `Err` neither the overlay nor `target` has changed.
    ///
    /// **This is the fold-back path, and the slot-identity proof lives here.**
    /// [`check`](Self::check) refuses a fold the mirror says would allocate
    /// different slots. [`replay`](Self::replay) then checks each slot petgraph
    /// actually allocates — the mirror itself could disagree with petgraph —
    /// because a different index would silently mis-key every `DirGraph` index
    /// that recorded the overlay's number.
    fn apply<T: FoldTarget>(&mut self, target: &mut T) -> Result<(), String> {
        self.check(target)?;
        self.replay(target)?;
        let touched_edges = self
            .delta
            .ops
            .iter()
            .any(|op| matches!(op, Op::AddEdge { .. } | Op::RemoveEdge(_)))
            || !self.delta.edges.weights.is_empty();
        // The final weights, which are pure overwrites of the placeholders.
        for (idx, data) in self.delta.nodes.drain() {
            if let Some(slot) = target
                .graph_mut()
                .node_weight_mut(NodeIndex::new(idx as usize))
            {
                *slot = data;
            }
        }
        let edges = &mut self.delta.edges;
        for (slot, edge) in edges.added.drain() {
            if let Some(weight) = target
                .graph_mut()
                .edge_weight_mut(EdgeIndex::new(slot as usize))
            {
                *weight = edge.weight;
            }
        }
        for (slot, data) in edges.weights.drain() {
            if let Some(weight) = target
                .graph_mut()
                .edge_weight_mut(EdgeIndex::new(slot as usize))
            {
                *weight = data;
            }
        }
        *self.delta = self.delta.emptied();
        *target.column_stores_mut() = std::mem::take(self.column_stores);
        // The base's stores went with the assignment above, so on compaction
        // each overlay store's columns are its own again and folding its heap
        // tail copies the tail only (`column_store/tail.rs`, "Heap tails").
        for store in target.column_stores_mut().values_mut() {
            if store.has_heap_tail() {
                Arc::make_mut(store).fold_heap_tail();
            }
        }
        target.set_undo(self.undo.take());
        target.after_fold(touched_edges);
        Ok(())
    }

    /// [`apply`](Self::apply) into whichever backend `target` is.
    fn apply_to(&mut self, target: &mut Folded) -> Result<(), String> {
        match target {
            Folded::Memory(g) => self.apply(g),
            Folded::Mapped(g) => self.apply(g),
        }
    }

    /// Run the log against `target`'s own petgraph, so each `add_node` /
    /// `add_edge` pops the slot the overlay took. A slot other than the one
    /// handed out undoes the adds this call made — newest first, since each
    /// removal pushes its slot back on the free-list head and the lists end in
    /// the order they started — and returns an error with the overlay intact.
    fn replay<T: FoldTarget>(&mut self, target: &mut T) -> Result<(), String> {
        for (done, op) in self.delta.ops.iter().enumerate() {
            let mismatch = match *op {
                Op::AddNode(idx) => {
                    let bound_before = target.graph().node_bound();
                    let actual = target.graph_mut().add_node(placeholder_node());
                    if actual.index() as u32 == idx {
                        target
                            .slot_mirror_mut()
                            .note_node_added(bound_before, actual);
                        continue;
                    }
                    target.graph_mut().remove_node(actual);
                    format!(
                        "fold-back allocated node {} where the overlay handed out {idx}; \
                         the slot mirror disagrees with petgraph (see storage/slot_mirror.rs)",
                        actual.index()
                    )
                }
                Op::AddEdge { slot, src, dst } => {
                    let bound_before = EdgeIndexable::edge_bound(target.graph());
                    let actual = target.graph_mut().add_edge(
                        NodeIndex::new(src as usize),
                        NodeIndex::new(dst as usize),
                        placeholder_edge(),
                    );
                    if actual.index() as u32 == slot {
                        target
                            .slot_mirror_mut()
                            .note_edge_added(bound_before, actual);
                        continue;
                    }
                    target.graph_mut().remove_edge(actual);
                    format!(
                        "fold-back allocated edge {} where the overlay handed out {slot}; \
                         the slot mirror disagrees with petgraph (see storage/slot_mirror.rs)",
                        actual.index()
                    )
                }
                Op::RemoveEdge(slot) => {
                    let edge = EdgeIndex::new(slot as usize);
                    let removed = target.graph_mut().remove_edge(edge);
                    debug_assert!(removed.is_some(), "the log removed a dead edge");
                    target.slot_mirror_mut().note_edge_removed(edge);
                    continue;
                }
                Op::RemoveNode(idx) => {
                    let node = NodeIndex::new(idx as usize);
                    let removed = target.graph_mut().remove_node(node);
                    debug_assert!(removed.is_some(), "the log removed a dead node");
                    target
                        .slot_mirror_mut()
                        .note_node_removed(node, std::iter::empty());
                    continue;
                }
            };
            return Err(self.unwind_adds(target, done, mismatch));
        }
        Ok(())
    }

    /// Reverse the first `done` replayed ops, which must all be adds.
    ///
    /// A removal cannot be reversed: re-adding an edge links it at the head of
    /// its lists, not where it was, so the base would read in a different
    /// order than the overlay promised. Reaching one means the slot mirror
    /// disagreed with petgraph *after* removals ran, which `check` rules out
    /// for any mirror that tracks petgraph.
    fn unwind_adds<T: FoldTarget>(&self, target: &mut T, done: usize, reason: String) -> String {
        let ops: Vec<Op> = self.delta.ops.iter().take(done).copied().collect();
        assert!(
            ops.iter()
                .all(|op| matches!(op, Op::AddNode(_) | Op::AddEdge { .. })),
            "{reason}; a removal had already been replayed, so the base cannot be restored"
        );
        for op in ops.iter().rev() {
            match *op {
                Op::AddNode(idx) => {
                    let node = NodeIndex::new(idx as usize);
                    target.graph_mut().remove_node(node);
                    target
                        .slot_mirror_mut()
                        .note_node_removed(node, std::iter::empty());
                }
                Op::AddEdge { slot, .. } => {
                    let edge = EdgeIndex::new(slot as usize);
                    target.graph_mut().remove_edge(edge);
                    target.slot_mirror_mut().note_edge_removed(edge);
                }
                Op::RemoveEdge(_) | Op::RemoveNode(_) => unreachable!("asserted above"),
            }
        }
        reason
    }
}

impl ForkedGraph {
    /// Incoming bindings of `node`, excluding a self-loop's second incidence.
    pub(super) fn count_incoming_nonself_edges_filtered(
        &self,
        node: NodeIndex,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
    ) -> Result<usize, String> {
        self.count_edges(
            node,
            petgraph::Direction::Incoming,
            conn_type,
            other_node_type,
            deadline,
            true,
        )
    }

    /// `MemoryGraph::count_edges_filtered_impl` over this view: the base's
    /// answer while the edge layer is clean, a walk of the chained edges once
    /// it is not (the base knows nothing of an overlay node's type).
    fn count_edges(
        &self,
        node: NodeIndex,
        dir: petgraph::Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
        exclude_self: bool,
    ) -> Result<usize, String> {
        // A peer's type is read from the overlay, so a count that filters on it
        // takes the base's answer only while no edge differs anywhere.
        let untouched = self.delta.edges.is_clean()
            || (other_node_type.is_none() && !self.delta.edges.touches(node));
        if untouched {
            return self.base.count_edges_filtered_impl(
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
                exclude_self,
            );
        }
        let mut count = 0;
        for (i, edge) in GraphRead::edges_directed(self, node, dir).enumerate() {
            if i.is_multiple_of(1 << 20)
                && deadline.is_some_and(|dl| std::time::Instant::now() > dl)
            {
                return Err("Query timed out".to_string());
            }
            if conn_type.is_some_and(|ct| edge.connection_type() != ct) {
                continue;
            }
            let other = if dir == petgraph::Direction::Outgoing {
                edge.target()
            } else {
                edge.source()
            };
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

    /// Every live edge's `(source, target, connection type)`, base-direct
    /// while the edge layer is clean.
    pub(crate) fn for_each_edge_endpoint_key(
        &self,
        mut f: impl FnMut(NodeIndex, NodeIndex, InternedKey),
    ) {
        if self.delta.edges.is_clean() {
            let mut base = self.base.edge_references();
            while let Some((_, source, target, weight)) = base.next_edge() {
                f(source, target, weight.connection_type);
            }
            return;
        }
        for edge in GraphRead::edge_references(self) {
            f(edge.source(), edge.target(), edge.connection_type());
        }
    }

    /// Edges of one connection type with their slot and property slice; the
    /// callback returns `false` to stop.
    pub(crate) fn for_each_edge_of_conn_type(
        &self,
        conn_type: InternedKey,
        mut f: impl FnMut(NodeIndex, NodeIndex, u32, &[(InternedKey, Value)]) -> bool,
    ) {
        if self.delta.edges.is_clean() {
            let mut base = self.base.edge_references();
            while let Some((id, source, target, w)) = base.next_edge() {
                if w.connection_type == conn_type
                    && !f(source, target, id.index() as u32, w.properties.as_slice())
                {
                    return;
                }
            }
            return;
        }
        for edge in GraphRead::edge_references(self) {
            let w = edge.weight();
            if w.connection_type == conn_type
                && !f(
                    edge.source(),
                    edge.target(),
                    edge.id().index() as u32,
                    w.properties.as_slice(),
                )
            {
                return;
            }
        }
    }

    /// Fork `base` — O(types), no node or edge is copied.
    pub(crate) fn new(base: impl Into<ForkBase>) -> Self {
        let base = base.into();
        let base_is_frozen = base.is_frozen();
        let column_stores = base.column_stores().clone();
        let slot_mirror = base.slot_mirror().clone();
        Self {
            base,
            delta: Delta::new(base_is_frozen),
            column_stores,
            undo: None,
            slot_mirror,
        }
    }

    /// How many node weights this overlay holds — the only nodes a clone of it
    /// duplicates, and what the `BACKEND_CLONE_NODES` oracle counts for a fork
    /// of a fork.
    #[cfg(test)]
    #[inline]
    pub(crate) fn overlay_node_count(&self) -> usize {
        self.delta.nodes.len()
    }

    /// Whether copying this overlay for a further fork costs more than
    /// collapsing it once.
    ///
    /// A fork of a fork duplicates the delta, and a reader held continuously
    /// across commits keeps the base shared, so the delta only grows. Past a
    /// sixteenth of the base, one O(graph) collapse
    /// ([`Self::materialise`]) resets it, amortised over the changes that
    /// earned it.
    pub(crate) fn delta_exceeds_clone_cap(&self) -> bool {
        self.delta_exceeds(16)
    }

    /// Whether an adjacency write should collapse the overlay before going on.
    ///
    /// Each edit runs twice if the overlay carries it: into the delta, then
    /// into the base at the fold. A statement that rewrites a large share of the
    /// graph is cheaper on a plain graph, so past a sixty-fourth of the base the
    /// overlay collapses (`GraphBackend::flatten_fork`) and the rest of the
    /// statement runs in place. The cost is then at most what these writes paid
    /// before the overlay held adjacency: one whole-graph copy. The measured
    /// crossover is in `docs/rust/structural-sharing.md`.
    pub(crate) fn delta_exceeds_write_cap(&self) -> bool {
        self.delta_exceeds(64)
    }

    /// The floor keeps a small graph from collapsing on every few edits. An
    /// overlay over a frozen base never collapses on size: it has no fold-back
    /// to pay twice, and its fork, reads and heap stay proportional to what was
    /// written.
    fn delta_exceeds(&self, base_fraction: usize) -> bool {
        if self.base.is_frozen() {
            return false;
        }
        self.delta_len()
            > VALVE_FLOOR.max((self.base.node_count() + self.base.edge_count()) / base_fraction)
    }

    /// Operations, node weights and edge weights the overlay holds: what a
    /// collapse or a clone has to copy.
    pub(crate) fn delta_len(&self) -> usize {
        self.delta.ops.len() + self.delta.nodes.len() + self.delta.edges.weights.len()
    }

    /// The overlay's own state, borrowed apart from its base.
    fn overlay(&mut self) -> Overlay<'_> {
        Overlay {
            delta: &mut self.delta,
            column_stores: &mut self.column_stores,
            undo: &mut self.undo,
        }
    }

    /// Fold into the base in place when this writer is its only holder, and
    /// return the folded graph.
    ///
    /// `None` when a reader is still outstanding, or when the fold would not
    /// reproduce the overlay's indices; the overlay and its base are then
    /// unchanged and keep serving reads and writes. The base is moved out only
    /// after the fold succeeded, so no failure leaves an empty graph behind.
    ///
    /// The reader dropping is exactly what makes `Arc::get_mut` succeed, so the
    /// common "hold a view, write, drop the view, write again" pattern
    /// self-heals on the next write with no timer and no bookkeeping. Sole
    /// owner, the fold writes into the base itself — no node or edge is
    /// copied, which is what makes compaction O(changes), not O(V+E).
    pub(crate) fn fold_in_place(&mut self) -> Option<Folded> {
        let Self {
            base,
            delta,
            column_stores,
            undo,
            ..
        } = self;
        let mut overlay = Overlay {
            delta,
            column_stores,
            undo,
        };
        match base {
            ForkBase::Memory(arc) => {
                let target = Arc::get_mut(arc)?;
                overlay.apply(target).ok()?;
                Some(Folded::Memory(std::mem::replace(
                    target,
                    MemoryGraph::new(),
                )))
            }
            ForkBase::Mapped(arc) => {
                let target = Arc::get_mut(arc)?;
                overlay.apply(target).ok()?;
                Some(Folded::Mapped(std::mem::replace(
                    target,
                    MappedGraph::new(),
                )))
            }
            // A frozen base has no petgraph to fold into; the overlay lives on
            // until a save rebases it or a whole-graph operation collapses it
            // ([`Self::materialise`]).
            ForkBase::Frozen(_) => None,
        }
    }

    /// Deep-copy the base and fold into the copy: the collapse behind
    /// `GraphBackend::flatten_fork`, for the whole-graph operations that need
    /// one concrete `StableDiGraph` and for a delta past its cap
    /// ([`Self::delta_exceeds_write_cap`]).
    ///
    /// `Err` when the fold would not reproduce the overlay's indices; the
    /// overlay is then untouched and the copy is dropped.
    pub(crate) fn materialise(&mut self) -> Result<Folded, String> {
        #[cfg(test)]
        super::backend::note_nodes_copied(self.base.node_count());
        let mut owned = self.base.deep_clone();
        self.overlay().apply_to(&mut owned)?;
        Ok(owned)
    }

    /// A standalone graph equal to what this overlay reads as, without
    /// disturbing it. For the `Serialize` arm in `backend.rs`, which needs one
    /// concrete `StableDiGraph`.
    pub(crate) fn to_folded_graph(&self) -> Result<Folded, String> {
        let mut clone = ForkedGraph {
            base: self.base.share(),
            delta: self.delta.clone(),
            column_stores: self.column_stores.clone(),
            undo: None,
            slot_mirror: self.slot_mirror.clone(),
        };
        let mut owned = self.base.deep_clone();
        clone.overlay().apply_to(&mut owned)?;
        Ok(owned)
    }

    /// Peer counts of `conn_type` over a frozen base with edge writes: the
    /// file's counts, corrected by what the overlay added, removed and
    /// re-typed. O(overlay + peers) where a walk of every edge is O(edges).
    fn peer_counts_over_frozen(
        &self,
        frozen: &crate::graph::storage::frozen::FrozenGraph,
        conn_type: InternedKey,
        dir: petgraph::Direction,
    ) -> HashMap<u32, i64> {
        let mut counts = (*frozen.peer_counts(conn_type, dir)).clone();
        let peer_of = |source: NodeIndex, target: NodeIndex| match dir {
            petgraph::Direction::Outgoing => target,
            petgraph::Direction::Incoming => source,
        };
        let mut bump = |peer: NodeIndex, by: i64| {
            let slot = counts.entry(peer.index() as u32).or_insert(0);
            *slot += by;
            if *slot == 0 {
                counts.remove(&(peer.index() as u32));
            }
        };
        let layer = &self.delta.edges;
        for slot in layer.removed.iter() {
            let slot = EdgeIndex::new(slot as usize);
            if let (Some((source, target)), Some(weight)) =
                (self.base.edge_endpoints(slot), self.base.edge_weight(slot))
            {
                if weight.connection_type == conn_type {
                    bump(peer_of(source, target), -1);
                }
            }
        }
        for (slot, weight) in layer.weights.iter() {
            let slot = EdgeIndex::new(slot as usize);
            if let (Some((source, target)), Some(file)) =
                (self.base.edge_endpoints(slot), self.base.edge_weight(slot))
            {
                let (was, is) = (file.connection_type, weight.connection_type);
                if was != is {
                    let peer = peer_of(source, target);
                    if was == conn_type {
                        bump(peer, -1);
                    }
                    if is == conn_type {
                        bump(peer, 1);
                    }
                }
            }
        }
        for edge in layer.added.values() {
            if edge.weight.connection_type == conn_type {
                let (source, target) = (
                    NodeIndex::new(edge.src as usize),
                    NodeIndex::new(edge.dst as usize),
                );
                bump(peer_of(source, target), 1);
            }
        }
        counts
    }

    /// The frozen base, while the overlay holds no edge write: its recorded
    /// per-connection-type peer counts and sources then describe this view
    /// exactly (a node write moves no edge, and removing a node removes its
    /// edges first).
    #[inline]
    fn untouched_frozen_base(&self) -> Option<&crate::graph::storage::frozen::FrozenGraph> {
        if self.delta.edges.is_clean() {
            self.base.frozen().map(|frozen| &**frozen)
        } else {
            None
        }
    }

    /// The frozen base's cached peer counts for `conn_type`, while it
    /// describes this view; see [`Self::untouched_frozen_base`].
    pub(crate) fn untouched_peer_counts(
        &self,
        conn_type: InternedKey,
        dir: petgraph::Direction,
    ) -> Option<Arc<HashMap<u32, i64>>> {
        Some(self.untouched_frozen_base()?.peer_counts(conn_type, dir))
    }

    /// The frozen topology this overlay sits over, if that is its base.
    #[inline]
    pub(crate) fn frozen_base(&self) -> Option<&Arc<crate::graph::storage::frozen::FrozenGraph>> {
        self.base.frozen()
    }

    /// Whether the base is a `Mapped` backend, which this overlay then reports
    /// as (`GraphRead::is_mapped`).
    #[inline]
    pub(crate) fn base_is_mapped(&self) -> bool {
        self.base.is_mapped()
    }

    #[inline]
    pub(crate) fn begin_undo(&mut self) {
        let mut journal = Box::new(UndoJournal::new());
        journal.note_ops_mark(self.delta.ops.len());
        self.undo = Some(journal);
    }

    /// Give back what a rolled-back statement left in this overlay, once the
    /// journal has restored the logical state (see [`OverlayResidue`]): cut the
    /// operation log to where the statement found it and drop the edge weights
    /// it copied from the base. Does nothing for a statement that removed a node
    /// or an edge, and for an overlay over a heap base, which folds away.
    pub(crate) fn clear_residue(&mut self, residue: OverlayResidue) {
        let Some(mark) = residue.ops_mark else {
            return;
        };
        if residue.removed || self.base.frozen().is_none() {
            return;
        }
        self.delta.ops.truncate(mark);
        for slot in residue.copied_edge_weights {
            self.delta.edges.weights.remove(slot);
        }
    }

    #[inline]
    pub(crate) fn take_undo(&mut self) -> Option<Box<UndoJournal>> {
        self.undo.take()
    }

    #[inline]
    pub(crate) fn undo_journal_mut(&mut self) -> Option<&mut UndoJournal> {
        self.undo.as_deref_mut()
    }

    /// The overlay's copy of a base node, taken on first write.
    ///
    /// This is the single point where a base node stops being shared, and the
    /// reason the base is never mutated: every `&mut NodeData` this backend
    /// hands out points into `self.delta.nodes`.
    #[inline]
    fn cow_node(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        let raw = idx.index() as u32;
        if !self.delta.nodes.holds(raw) {
            if self.delta.dead_count != 0 && self.delta.dead.holds(raw) {
                return None;
            }
            let base = self.base.node_weight(idx)?.clone();
            self.delta.nodes.insert(raw, base);
        }
        self.delta.nodes.get_mut(raw)
    }

    /// Clone `idx`'s current weight into the journal as its pre-statement
    /// state. Reads through [`GraphRead::node_weight`], i.e. overlay-then-base,
    /// so the pre-image is what *this writer* would have read — not what the
    /// base holds, which may already differ if an earlier statement wrote it.
    #[cold]
    fn capture_node_weight(&mut self, idx: NodeIndex) {
        let current = GraphRead::node_weight(self, idx).map(|r| (*r).clone());
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_weight(idx, || current);
        }
    }

    /// Edge counterpart of [`Self::capture_node_weight`]; the clone is lazy, so
    /// a second write to the same edge in a statement copies nothing.
    #[cold]
    fn capture_edge_weight(&mut self, idx: EdgeIndex) {
        let Self {
            base, delta, undo, ..
        } = self;
        if let Some(journal) = undo.as_deref_mut() {
            journal.note_edge_weight(idx, || delta.edges.weight(base, idx).map(|r| (*r).clone()));
        }
    }

    /// Journal the pre-image a property write is about to overwrite — the
    /// overlay's counterpart of `impl_heap_pre_image_capture!`. Same two
    /// shapes, same reasoning (see that macro): a row-storage node journals a
    /// `NodeData` clone, a columnar one journals the prior value of each cell
    /// the write names.
    ///
    /// The store read is `self.column_stores`, the overlay's own map, which is
    /// what the write will mutate — so the pre-image is what *this writer*
    /// would read back, not what the shared base holds.
    #[inline]
    fn capture_property_pre_image(&mut self, idx: NodeIndex, write: ColumnarWrite<'_>) {
        if self.undo.is_none() {
            return;
        }
        let Some(nd) = GraphRead::node_weight(self, idx) else {
            return;
        };
        let columnar = nd
            .properties
            .columnar_row_id()
            .map(|row_id| (nd.node_type, row_id));
        match columnar {
            None => self.capture_node_weight(idx),
            Some((type_key, row_id)) => {
                let captured = self
                    .column_stores
                    .get(&type_key)
                    .map(|store| ColumnarPreImages::capture(store, row_id, write));
                if let (Some(captured), Some(journal)) = (captured, self.undo.as_deref_mut()) {
                    captured.record(journal, type_key, row_id);
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

    /// Take a live node's weight out of the view. A base node becomes `dead`;
    /// an overlay node leaves `extra`.
    fn take_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        let raw = idx.index() as u32;
        let own = self.delta.nodes.remove(raw);
        match self.base.node_weight(idx) {
            Some(base) => {
                if self.delta.dead_count != 0 && self.delta.dead.holds(raw) {
                    return None;
                }
                self.delta.dead.set(raw);
                self.delta.dead_count += 1;
                Some(own.unwrap_or_else(|| base.clone()))
            }
            None => {
                self.delta.extra.remove(raw);
                own
            }
        }
    }

    /// Hide the master row a just-removed node owned, and journal the flip —
    /// the columnar half of a node deletion (`impl_heap_pre_image_capture!`).
    fn tombstone_removed_row(&mut self, removed: &NodeData) {
        let Some(row_id) = removed.properties.columnar_row_id() else {
            return;
        };
        let type_key = removed.node_type;
        let Some(store) = self.column_stores.get_mut(&type_key) else {
            return;
        };
        Arc::make_mut(store).tombstone(row_id);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_columnar_tombstone(type_key, row_id);
        }
    }

    /// Remove an edge from this view, journalling and logging it. The weight
    /// comes back only when `want_weight` asks for it: detaching a node's edges
    /// discards every one, and a base edge's weight can only be *copied* out of
    /// the base, so an unwanted copy is a wasted allocation per edge.
    fn unlink_edge(&mut self, idx: EdgeIndex, want_weight: bool) -> Option<EdgeData> {
        let raw = idx.index() as u32;
        let need_weight = want_weight || self.undo.is_some();
        let (src, dst, weight) = match self.delta.edges.take_added(raw) {
            Some(edge) => (edge.src, edge.dst, need_weight.then_some(edge.weight)),
            None => {
                if self.delta.edges.removed.contains(raw) {
                    return None;
                }
                let (src, dst) = self.base.edge_endpoints(idx)?;
                let base_weight = self.base.edge_weight(idx)?;
                let own = self.delta.edges.weights.remove(raw);
                let weight = need_weight.then(|| own.unwrap_or_else(|| base_weight.clone()));
                self.delta
                    .edges
                    .tombstone(raw, src.index() as u32, dst.index() as u32);
                (src.index() as u32, dst.index() as u32, weight)
            }
        };
        self.slot_mirror.note_edge_removed(idx);
        self.delta.ops.push(Op::RemoveEdge(raw));
        match (self.undo.as_deref_mut(), weight) {
            (Some(journal), Some(weight)) => {
                let (src, dst) = (NodeIndex::new(src as usize), NodeIndex::new(dst as usize));
                if want_weight {
                    journal.note_edge_removed(idx, src, dst, weight.clone());
                    Some(weight)
                } else {
                    journal.note_edge_removed(idx, src, dst, weight);
                    None
                }
            }
            (_, weight) => weight,
        }
    }

    /// The edges `remove_node` detaches, in the order the folded graph's
    /// petgraph would free them: outgoing head-first, then incoming, with a
    /// self-loop counted once.
    fn incident_edges(&self, idx: NodeIndex) -> Vec<EdgeIndex> {
        let mut edges: Vec<EdgeIndex> =
            GraphRead::edges_directed(self, idx, petgraph::Direction::Outgoing)
                .map(|edge| edge.id())
                .collect();
        edges.extend(
            GraphRead::edges_directed(self, idx, petgraph::Direction::Incoming)
                .filter(|edge| edge.source() != idx)
                .map(|edge| edge.id()),
        );
        edges
    }
}

impl Clone for ForkedGraph {
    /// Forking a fork keeps the same base and copies only the delta.
    fn clone(&self) -> Self {
        Self {
            base: self.base.share(),
            delta: self.delta.clone(),
            column_stores: self.column_stores.clone(),
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
             {} nodes allocated, {} edges added, {} removed }}",
            self.base.node_count(),
            self.base.edge_count(),
            self.delta.nodes.len(),
            self.delta.extra.len(),
            self.delta.edges.added.len(),
            self.delta.edges.removed.len()
        )
    }
}

/// The node scan of an overlay over a petgraph base.
fn heap_node_indices<'a>(
    base: &'a petgraph::stable_graph::StableDiGraph<NodeData, EdgeData>,
    delta: &'a Delta,
    appended: Option<std::ops::Range<u32>>,
    dead: Option<&'a OverlayBits>,
) -> GraphNodeIndices<'a> {
    let nodes = base.node_indices();
    match appended {
        Some(run) => GraphNodeIndices::Forked {
            base: Box::new(nodes),
            appended: run.start as usize..run.end as usize,
        },
        None => GraphNodeIndices::ForkedMerged(Box::new(ForkedMergedIndices::new(
            nodes,
            delta.extra.iter(),
            dead,
        ))),
    }
}

// Node weights and edges are overlay-then-base: a slot the overlay holds an
// opinion about is answered from the delta, every other from the base.
impl GraphRead for ForkedGraph {
    type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = GraphEdgeIndices<'a>;
    type EdgesIter<'a> = GraphEdges<'a>;
    type EdgeReferencesIter<'a> = GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = GraphNeighbors<'a>;

    #[inline]
    fn node_count(&self) -> usize {
        self.base.node_count() - self.delta.dead_count + self.delta.extra.len()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        (self.base.edge_count() as isize + self.delta.edges.count_delta()) as usize
    }

    /// One past the highest live slot, as petgraph's own `node_bound()` reads
    /// (`StableGraph` trims to the last live node). A removed trailing base
    /// node walks the bound down over the dead and vacant slots below it.
    fn node_bound(&self) -> usize {
        let base = &self.base;
        let mut bound = base.node_bound();
        if self.delta.dead_count != 0 {
            while bound > 0
                && (self.delta.dead.holds(bound as u32 - 1)
                    || base.node_weight(NodeIndex::new(bound - 1)).is_none())
            {
                bound -= 1;
            }
        }
        bound.max(self.delta.extra.last().map_or(0, |top| top as usize + 1))
    }

    /// One past the highest live edge slot; see [`Self::node_bound`].
    fn edge_bound(&self) -> usize {
        let base = &self.base;
        let mut bound = base.edge_bound();
        let removed = &self.delta.edges.removed;
        while !removed.is_empty()
            && bound > 0
            && (removed.contains(bound as u32 - 1)
                || base.edge_weight(EdgeIndex::new(bound - 1)).is_none())
        {
            bound -= 1;
        }
        bound.max(self.delta.edges.added_bound())
    }

    #[inline]
    fn is_memory(&self) -> bool {
        !self.base.is_mapped()
    }

    #[inline]
    fn is_mapped(&self) -> bool {
        self.base.is_mapped()
    }

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<RecordRef<'_, NodeData>> {
        let raw = idx.index() as u32;
        match self.delta.nodes.get(raw) {
            Some(data) => Some(RecordRef::Borrowed(data)),
            None if self.delta.dead_count != 0 && self.delta.dead.holds(raw) => None,
            None => self.base.node_weight(idx),
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

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> Self::EdgesIter<'_> {
        if self.delta.edges.is_clean() || !self.delta.edges.touches(idx) {
            return base_read!(self.base, g => GraphRead::edges_directed_filtered(g, idx, dir, conn_type_filter));
        }
        GraphRead::edges_directed(self, idx, dir)
    }

    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        if self.delta.edges.is_clean() {
            return base_read!(self.base, g => GraphRead::edge_endpoint_keys(g));
        }
        Box::new(
            GraphRead::edge_references(self)
                .map(|edge| (edge.source(), edge.target(), edge.connection_type())),
        )
    }

    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: petgraph::Direction,
        deadline: Option<std::time::Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        if self.delta.edges.is_clean() {
            return base_read!(self.base, g => GraphRead::count_edges_grouped_by_peer(g, conn_type, dir, deadline));
        }
        if let Some(frozen) = self.base.frozen() {
            return Ok(self.peer_counts_over_frozen(frozen, conn_type, dir));
        }
        let mut counts: HashMap<u32, i64> = HashMap::new();
        for (i, edge) in GraphRead::edge_references(self).enumerate() {
            if i.is_multiple_of(1 << 20)
                && deadline.is_some_and(|dl| std::time::Instant::now() > dl)
            {
                return Err("Query timed out".to_string());
            }
            if edge.connection_type() != conn_type {
                continue;
            }
            let peer = match dir {
                petgraph::Direction::Outgoing => edge.target(),
                petgraph::Direction::Incoming => edge.source(),
            };
            *counts.entry(peer.index() as u32).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: petgraph::Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
    ) -> Result<usize, String> {
        self.count_edges(node, dir, conn_type, other_node_type, deadline, false)
    }

    /// The file's recorded sources, while no edge differs from the file's. A
    /// heap base keeps no such record, so a fork over one answers `None` and
    /// the caller scans.
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        self.untouched_frozen_base()?
            .sources_for_conn_type_bounded(conn_type, max)
    }

    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        self.untouched_frozen_base()?.lookup_peer_counts(conn_type)
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

    /// Every live index in ascending order — the unforked graph's scan order,
    /// which `type_indices` bucket order and the rollback fidelity tests both
    /// pin. With no node removed and the overlay's slots one run past the base,
    /// a chain suffices; otherwise they are merged in.
    #[inline]
    fn node_indices(&self) -> Self::NodeIndicesIter<'_> {
        let delta = &self.delta;
        let appended = if delta.dead_count == 0 {
            delta
                .extra
                .as_single_run()
                .filter(|run| run.is_empty() || run.start as usize >= self.base.node_bound())
        } else {
            None
        };
        let dead = (delta.dead_count != 0).then_some(&delta.dead);
        match &self.base {
            ForkBase::Memory(g) => heap_node_indices(g.inner(), delta, appended, dead),
            ForkBase::Mapped(g) => heap_node_indices(g.inner(), delta, appended, dead),
            ForkBase::Frozen(g) => {
                let base = FrozenNodeIndices::new(g);
                match appended {
                    Some(run) => GraphNodeIndices::FrozenForked {
                        base: Box::new(base),
                        appended: run.start as usize..run.end as usize,
                    },
                    None => GraphNodeIndices::FrozenForkedMerged(Box::new(
                        ForkedMergedIndices::new(base, delta.extra.iter(), dead),
                    )),
                }
            }
        }
    }

    #[inline]
    fn edge_indices(&self) -> Self::EdgeIndicesIter<'_> {
        if self.delta.edges.is_clean() {
            return base_read!(self.base, g => GraphRead::edge_indices(g));
        }
        GraphEdgeIndices::Forked(Box::new(ForkedEdgeRefs::new(
            &self.delta.edges,
            self.base.edge_references(),
        )))
    }

    #[inline]
    fn edge_references(&self) -> Self::EdgeReferencesIter<'_> {
        if self.delta.edges.is_clean() {
            return base_read!(self.base, g => GraphRead::edge_references(g));
        }
        GraphEdgeReferences::Forked(Box::new(ForkedEdgeRefs::new(
            &self.delta.edges,
            self.base.edge_references(),
        )))
    }

    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = RecordRef<'a, EdgeData>> + 'a> {
        if self.delta.edges.is_clean() {
            return base_read!(self.base, g => GraphRead::edge_weights(g));
        }
        Box::new(GraphRead::edge_references(self).map(|edge| edge.weight()))
    }

    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: petgraph::Direction) -> Self::EdgesIter<'_> {
        if self.delta.edges.is_clean() || !self.delta.edges.touches(idx) {
            return base_read!(self.base, g => GraphRead::edges_directed(g, idx, dir));
        }
        GraphEdges::Forked(Box::new(ForkedEdges::new(
            &self.delta.edges,
            idx,
            dir,
            self.base.edges_directed(idx, dir),
        )))
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> Self::EdgesIter<'_> {
        GraphRead::edges_directed(self, idx, petgraph::Direction::Outgoing)
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> Self::EdgesConnectingIter<'_> {
        if self.delta.edges.is_clean() || !self.delta.edges.touches(a) {
            return base_read!(self.base, g => GraphRead::edges_connecting(g, a, b));
        }
        GraphEdgesConnecting::Forked(Box::new(ForkedEdgesConnecting::new(
            ForkedEdges::new(
                &self.delta.edges,
                a,
                petgraph::Direction::Outgoing,
                self.base.edges_directed(a, petgraph::Direction::Outgoing),
            ),
            b,
        )))
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<RecordRef<'_, EdgeData>> {
        self.delta.edges.weight(&self.base, idx)
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        if self.delta.edges.is_clean() || !self.delta.edges.touches(a) {
            return base_read!(self.base, g => GraphRead::find_edge(g, a, b));
        }
        GraphRead::edges_connecting(self, a, b)
            .next()
            .map(|edge| edge.id())
    }

    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.delta.edges.endpoints(&self.base, idx)
    }

    #[inline]
    fn neighbors_directed(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
    ) -> Self::NeighborsIter<'_> {
        if self.delta.edges.is_clean() || !self.delta.edges.touches(idx) {
            return base_read!(self.base, g => GraphRead::neighbors_directed(g, idx, dir));
        }
        let walk = |dir| {
            ForkedEdges::new(
                &self.delta.edges,
                idx,
                dir,
                self.base.edges_directed(idx, dir),
            )
        };
        let (out, inn) = match dir {
            petgraph::Direction::Outgoing => (Some(walk(dir)), None),
            petgraph::Direction::Incoming => (None, Some(walk(dir))),
        };
        GraphNeighbors::Forked(Box::new(ForkedNeighbors::new(out, inn, None)))
    }

    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> Self::NeighborsIter<'_> {
        if self.delta.edges.is_clean() || !self.delta.edges.touches(idx) {
            return base_read!(self.base, g => GraphRead::neighbors_undirected(g, idx));
        }
        let walk = |dir| {
            ForkedEdges::new(
                &self.delta.edges,
                idx,
                dir,
                self.base.edges_directed(idx, dir),
            )
        };
        GraphNeighbors::Forked(Box::new(ForkedNeighbors::new(
            Some(walk(petgraph::Direction::Outgoing)),
            Some(walk(petgraph::Direction::Incoming)),
            Some(idx),
        )))
    }
}

// Every mutation lands in the overlay.
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

    /// An edge weight is copied into the edge layer on first write, so the
    /// reader's base keeps the weight it was forked with, and every iterating
    /// read serves the copy (`forked_edge_iters`).
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        let raw = idx.index() as u32;
        if self.undo.is_some() {
            self.capture_edge_weight(idx);
        }
        let edges = &mut self.delta.edges;
        if edges.added.contains_key(raw) {
            return edges.added.get_mut(raw).map(|edge| &mut edge.weight);
        }
        if edges.removed.contains(raw) {
            return None;
        }
        let base = self.base.edge_weight(idx)?;
        let (src, dst) = self.base.edge_endpoints(idx)?;
        if let Some(journal) = self.undo.as_deref_mut() {
            if !edges.weights.contains_key(raw) {
                journal.note_copied_edge_weight(raw);
            }
        }
        Some(edges.weight_mut(raw, (src.index() as u32, dst.index() as u32), &base))
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
        self.capture_property_pre_image(idx, ColumnarWrite::Cell(key));
        // Columnar: the value lives in the store, the node only holds a row id,
        // so nothing about the node diverges and no `NodeData` is copied.
        //
        // `Arc::make_mut` here sees the base's handle as well as the overlay's,
        // so the *first* write per type copies the store — which is exactly
        // right: the reader's base must keep the store it was forked with. It
        // is once per fork, not once per statement: the copy the overlay now
        // owns is uniquely held, so every later write mutates it in place. This
        // is the one place the master legitimately forks, and it is why the
        // write-site assertion in `columnar_write.rs` exempts a forked backend.
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

    /// Titles follow properties: on a columnar node they live in the store's
    /// reserved column, so the overlay writes them there — copying the store
    /// once per type on the first write, exactly as `set_node_property` does,
    /// so the base a reader is holding keeps its own titles.
    fn set_node_title(&mut self, idx: NodeIndex, value: Value) {
        if let Some((type_key, row_id)) = self.columnar_row_of(idx) {
            if self.undo.is_some() {
                let prior = self
                    .column_stores
                    .get(&type_key)
                    .and_then(|store| store.get_title(row_id));
                if let Some(journal) = self.undo.as_deref_mut() {
                    journal.note_columnar_title(type_key, row_id, prior);
                }
            }
            if let Some(store) = self.column_stores.get_mut(&type_key) {
                if Arc::make_mut(store).set_title(row_id, &value) {
                    return;
                }
            }
        }
        if let Some(nd) = self.cow_node(idx) {
            nd.set_title(value);
        }
    }

    fn remove_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        let previous = GraphRead::get_node_property(self, idx, key);
        self.capture_property_pre_image(idx, ColumnarWrite::Cell(key));
        if let Some((type_key, row_id)) = self.columnar_row_of(idx) {
            if previous.is_some() {
                if let Some(store) = self.column_stores.get_mut(&type_key) {
                    Arc::make_mut(store).set(row_id, key, &Value::Null, None);
                }
            }
            return previous;
        }
        self.cow_node(idx)?.properties.remove(key)
    }

    fn clear_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        GraphWrite::remove_node_property(self, idx, key)
    }

    /// Replace-**all**: every cell present on the row is nulled, then `pairs`
    /// are written — arm for arm what `impl_heap_column_writes!` does, so a
    /// held view cannot turn a caller's replace into an update.
    fn replace_node_properties(&mut self, idx: NodeIndex, pairs: Vec<(InternedKey, Value)>) {
        // Whole-row shape: the pre-image has to span the nulled cells as well
        // as the incoming keys, which is exactly `ReplaceRow`.
        let written: Vec<InternedKey> = pairs.iter().map(|(k, _)| *k).collect();
        self.capture_property_pre_image(idx, ColumnarWrite::ReplaceRow(&written));
        if let Some((type_key, row_id)) = self.columnar_row_of(idx) {
            let Some(store) = self.column_stores.get_mut(&type_key) else {
                return;
            };
            let store = Arc::make_mut(store);
            let existing: Vec<_> = store
                .row_properties(row_id)
                .into_iter()
                .map(|(k, _)| k)
                .collect();
            for key in existing {
                store.set(row_id, key, &Value::Null, None);
            }
            for (key, value) in pairs {
                store.set(row_id, key, &value, None);
            }
            return;
        }
        if let Some(nd) = self.cow_node(idx) {
            nd.properties.replace_all(pairs);
        }
    }

    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        let node_type = data.node_type;
        let bound_before = GraphRead::node_bound(self);
        // The slot petgraph would hand out (module doc): `can_fork` admits only
        // a synced mirror, so there is always a prediction.
        let idx = self
            .slot_mirror
            .predict_next_node(bound_before)
            .expect("can_fork admits only a base whose slot mirror is synced");
        let raw = idx.index() as u32;
        if self.delta.dead_count != 0 && self.delta.dead.holds(raw) {
            // A base slot the overlay removed and is now reusing.
            self.delta.dead.clear(raw);
            self.delta.dead_count -= 1;
        } else {
            self.delta.extra.insert(raw);
        }
        self.delta.nodes.insert(raw, data);
        self.slot_mirror.note_node_added(bound_before, idx);
        self.delta.ops.push(Op::AddNode(raw));
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_added(idx, node_type);
        }
        idx
    }

    /// Detaches the node's edges one at a time through [`Self::unlink_edge`]
    /// (so each is logged, journalled and freed in the order petgraph would
    /// free it), then frees the node's slot.
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        GraphRead::node_weight(self, idx)?;
        for edge in self.incident_edges(idx) {
            self.unlink_edge(edge, false);
        }
        let removed = self.take_node(idx)?;
        self.slot_mirror.note_node_removed(idx, std::iter::empty());
        self.delta.ops.push(Op::RemoveNode(idx.index() as u32));
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_removed(idx, removed.clone());
        }
        self.tombstone_removed_row(&removed);
        Some(removed)
    }

    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        for endpoint in [a, b] {
            assert!(
                GraphRead::node_weight(self, endpoint).is_some(),
                "StableGraph::add_edge: node index {} is not a node in the graph",
                endpoint.index()
            );
        }
        // The bound only matters when no slot has been vacated, so the exact
        // `edge_bound` (a scan of the overlay's edges) is not needed here.
        let bound_before = self.delta.edges.append_bound(self.base.edge_bound());
        let idx = self
            .slot_mirror
            .predict_next_edge(bound_before)
            .expect("can_fork admits only a base whose slot mirror is synced");
        let slot = idx.index() as u32;
        let (src, dst) = (a.index() as u32, b.index() as u32);
        self.delta.edges.insert_added(slot, src, dst, data);
        self.slot_mirror.note_edge_added(bound_before, idx);
        self.delta.ops.push(Op::AddEdge { slot, src, dst });
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_added(idx);
        }
        idx
    }

    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.unlink_edge(idx, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::storage::interner::StringInterner;

    fn node(i: i64, interner: &mut StringInterner) -> NodeData {
        NodeData::new(
            Value::Int64(i),
            Value::String(format!("n{i}")),
            "Item".to_string(),
            HashMap::new(),
            interner,
        )
    }

    fn edge(interner: &mut StringInterner) -> EdgeData {
        EdgeData::new("LINKS".to_string(), HashMap::new(), interner)
    }

    fn graph_of(n: i64, interner: &mut StringInterner) -> MemoryGraph {
        let mut graph = MemoryGraph::new();
        for i in 0..n {
            GraphWrite::add_node(&mut graph, node(i, interner));
        }
        graph
    }

    /// A fold whose slots would not line up is refused before either side
    /// changes. Without the up-front check the replay allocated the listed
    /// slot, asserted, and left the target half-folded (issue #195).
    #[test]
    fn a_fold_that_would_misplace_a_node_changes_nothing() {
        let mut interner = StringInterner::new();
        let mut forked = ForkedGraph::new(Arc::new(graph_of(3, &mut interner)));
        let appended = GraphWrite::add_node(&mut forked, node(3, &mut interner));
        assert_eq!(appended, NodeIndex::new(3));

        // Same bound (3), but slot 1 heads the free list, so `add_node` would
        // reuse it instead of appending at 3.
        let mut target = graph_of(4, &mut interner);
        GraphWrite::remove_node(&mut target, NodeIndex::new(3));
        GraphWrite::remove_node(&mut target, NodeIndex::new(1));
        assert_eq!(target.inner().node_bound(), 3);

        let error = forked
            .overlay()
            .apply(&mut target)
            .expect_err("the fold must refuse a target that reuses a listed slot");
        assert!(error.contains("node 3"), "{error}");
        assert_eq!(
            target.inner().node_count(),
            2,
            "the target must be untouched"
        );
        assert_eq!(
            forked.delta.extra.len(),
            1,
            "the overlay must keep its append"
        );
        assert!(
            GraphRead::node_weight(&forked, appended).is_some(),
            "the overlay must still serve the appended node"
        );
    }

    /// A base whose mirror disagrees with petgraph: node 1 left the petgraph
    /// free list's head without the mirror hearing of it, so the overlay and
    /// the fold check both expect slot 4 while `add_node` reuses slot 1.
    fn forked_over_a_misled_mirror(interner: &mut StringInterner) -> (ForkedGraph, NodeIndex) {
        let mut base = graph_of(4, interner);
        base.inner_mut().remove_node(NodeIndex::new(1));
        let mut forked = ForkedGraph::new(Arc::new(base));
        let appended = GraphWrite::add_node(&mut forked, node(9, interner));
        assert_eq!(appended, NodeIndex::new(4));
        (forked, appended)
    }

    /// The fold finds the disagreement only when petgraph allocates. It must
    /// undo its appends and leave the overlay serving, not panic half-folded.
    #[test]
    fn a_fold_that_petgraph_misplaces_is_undone() {
        let mut interner = StringInterner::new();
        let (mut forked, appended) = forked_over_a_misled_mirror(&mut interner);
        assert!(
            forked.fold_in_place().is_none(),
            "the fold must refuse a misplaced slot"
        );
        assert_eq!(
            forked.delta.extra.len(),
            1,
            "the overlay must keep its append"
        );
        assert!(GraphRead::node_weight(&forked, appended).is_some());
        assert_eq!(forked.base.node_count(), 3, "the base is unchanged");
    }

    /// `GraphBackend::try_compact` on the same overlay keeps a forked backend
    /// that reads exactly as before — never the empty placeholder a panic
    /// mid-fold used to leave.
    #[test]
    fn try_compact_keeps_the_graph_when_the_fold_is_refused() {
        use crate::graph::storage::backend::GraphBackend;
        let mut interner = StringInterner::new();
        let (forked, appended) = forked_over_a_misled_mirror(&mut interner);
        let mut backend = GraphBackend::Forked(Box::new(forked));
        let before: Vec<_> = backend.node_indices().collect();
        let compacted =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| backend.try_compact()));
        assert!(compacted.is_ok(), "the refused fold panicked");
        assert!(backend.is_forked(), "the overlay must keep serving");
        assert_eq!(backend.node_indices().collect::<Vec<_>>(), before);
        assert_eq!(backend.node_count(), 4);
        assert!(backend.node_weight(appended).is_some());
    }

    /// A fork of a fork copies the delta, so a delta that only grows must be
    /// collapsed at some size or every commit under a held reader pays for all
    /// the commits before it.
    #[test]
    fn a_large_delta_collapses_instead_of_being_copied_again() {
        use crate::graph::storage::backend::GraphBackend;
        let mut interner = StringInterner::new();
        let mut forked = ForkedGraph::new(Arc::new(graph_of(10, &mut interner)));
        let first = GraphWrite::add_node(&mut forked, node(10, &mut interner));
        let backend = GraphBackend::Forked(Box::new(forked));
        assert!(
            backend.clone().is_forked(),
            "a small delta is copied, not collapsed"
        );

        let GraphBackend::Forked(mut forked) = backend else {
            unreachable!("built as Forked")
        };
        for i in 11..5000 {
            let a = GraphWrite::add_node(&mut *forked, node(i, &mut interner));
            GraphWrite::add_edge(&mut *forked, first, a, edge(&mut interner));
        }
        let before: Vec<_> = GraphRead::node_indices(&*forked).collect();
        let edges = GraphRead::edge_count(&*forked);
        let backend = GraphBackend::Forked(forked);
        let copy = backend.clone();
        assert!(
            !copy.is_forked(),
            "a large delta must collapse on the next fork"
        );
        assert_eq!(copy.node_indices().collect::<Vec<_>>(), before);
        assert_eq!(copy.edge_count(), edges);
    }

    /// A statement that rewrites a large share of the graph pays for two
    /// replays if the overlay carries it, so it collapses once mid-statement:
    /// the journal must come through the collapse, and what is left is a plain
    /// graph that reads as the edits dictate.
    #[test]
    fn a_large_adjacency_delta_collapses_mid_statement_and_keeps_the_journal() {
        use crate::graph::storage::backend::GraphBackend;
        let mut interner = StringInterner::new();
        let base = graph_of(10, &mut interner);
        let mut backend = GraphBackend::Forked(Box::new(ForkedGraph::new(Arc::new(base))));
        let hub = NodeIndex::new(0);
        backend.begin_undo();
        let mut edges = Vec::new();
        for _ in 0..5000 {
            edges.push(GraphWrite::add_edge(
                &mut backend,
                hub,
                NodeIndex::new(1),
                edge(&mut interner),
            ));
        }
        assert!(
            !backend.is_forked(),
            "the cap must have collapsed the overlay"
        );
        assert_eq!(backend.edge_count(), 5000);
        assert_eq!(
            backend
                .edges_directed(hub, petgraph::Direction::Outgoing)
                .count(),
            5000
        );
        let journal = backend
            .take_undo()
            .expect("the journal survives the collapse");
        let entries = journal.into_replay_order().count();
        assert_eq!(
            entries, 5000,
            "one EdgeAdded per edge, none lost or doubled"
        );
        for edge in edges.into_iter().rev() {
            GraphWrite::remove_edge(&mut backend, edge);
        }
        assert_eq!(backend.edge_count(), 0);
    }
}
