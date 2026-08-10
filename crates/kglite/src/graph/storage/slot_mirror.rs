//! A mirror of petgraph's node and edge free lists, so a slot can be
//! *predicted* before it is allocated.
//!
//! ## Why this exists
//!
//! Statement rollback guarantees that a node or edge comes back on **the exact
//! `NodeIndex`/`EdgeIndex` it vacated** (`dir_graph/rollback.rs`, "What
//! `identical` means here"), and `NodeIndex` is the key of every index
//! structure on `DirGraph`. D2 Phase 2 puts a *writer-side overlay* in front of
//! the shared base graph: the overlay must hand out an index at write time and
//! then reproduce that same index when the overlay is folded back into the base
//! at compaction. `StableGraph::add_node` reuses free-list slots and offers no
//! index-controlled insertion, so an overlay that invents indices from
//! `node_bound()..` would silently disagree with the base — mis-keying every
//! index that recorded the overlay's number. That is the deepest risk in the
//! programme (`docs/rust/structural-sharing.md`), because a gap produces wrong
//! data rather than a crash.
//!
//! This module is Phase 1's answer, landed a phase early **on purpose**: it
//! costs a `Vec` push/pop per node/edge add/remove today and buys a debug
//! assertion that validates the prediction against real petgraph behaviour on
//! every insert the whole test suite performs — 1500 Rust tests and the entire
//! Python suite — *before* anything depends on it being right.
//!
//! ## How petgraph behaves, and why a `Vec` mirrors it
//!
//! `StableGraph` threads its free slots through the node/edge arrays as a
//! singly-linked list with a head pointer (`free_node` / `free_edge`).
//! `remove_node`/`remove_edge` push the vacated slot onto the **front**;
//! `add_node`/`add_edge` pop the **front**. That is a LIFO stack, so a
//! `Vec` used as a stack reproduces the order exactly, and the next index is
//! `last()` when the stack is non-empty and `bound()` when it is empty.
//!
//! `remove_node` additionally frees every incident edge first, walking the
//! outgoing adjacency list to exhaustion and then the incoming one, removing
//! the *head* each time. So the freed-edge order is the iteration order of
//! `edges_directed(Outgoing)` followed by `edges_directed(Incoming)` — with
//! **self-loops counted once**, because a self-loop appears in both lists but
//! `remove_edge` unlinks it from both. [`SlotMirror::note_node_removed`] takes
//! the edges in that order and is documented at its call sites accordingly.
//!
//! ## The unknown-free-list case, and why it fails safe
//!
//! A `StableDiGraph` this process did not build — one adopted by
//! `from_graph`, or restored by serde — has a free list whose *order* is not
//! observable through petgraph's public API. Guessing it would be the one way
//! to make this module wrong in the direction that matters.
//!
//! So the mirror carries [`SlotMirror::synced`] and adopts an external graph
//! only when it provably has **no holes** (`node_count == node_bound` and
//! `edge_count == edge_bound`), which is exactly the case where the free lists
//! are provably empty. Otherwise it marks itself unsynced,
//! [`SlotMirror::predict_next_node`] returns `None`, and a caller that needs a
//! prediction must fall back. Unsynced means "slower", never "wrong" — the same
//! fail-safe direction `rollback::journal_covers` uses.

use petgraph::graph::{EdgeIndex, NodeIndex};

/// Mirror of the two petgraph free lists, plus whether it can be trusted.
#[derive(Debug, Clone)]
pub(crate) struct SlotMirror {
    /// Vacated node slots, LIFO — `last()` is what `add_node` will reuse.
    free_nodes: Vec<u32>,
    /// Vacated edge slots, LIFO — `last()` is what `add_edge` will reuse.
    free_edges: Vec<u32>,
    /// `false` once the mirror has adopted a graph whose free-list order it
    /// could not know. Predictions are refused rather than guessed.
    synced: bool,
}

/// The default mirror is the mirror of a *default* (empty) graph, so it is
/// synced. Spelled out rather than derived: `#[derive(Default)]` would produce
/// `synced: false`, which would silently disable prediction for every backend
/// built through `Default` — a gate that cannot go green rather than one that
/// cannot go red, but wrong either way.
impl Default for SlotMirror {
    #[inline]
    fn default() -> Self {
        Self::for_empty_graph()
    }
}

impl SlotMirror {
    /// A mirror for a graph this process is about to build from empty.
    ///
    /// Trustworthy by construction: an empty `StableDiGraph` has empty free
    /// lists, so an empty mirror is not an assumption but a fact.
    #[inline]
    pub(crate) fn for_empty_graph() -> Self {
        Self {
            free_nodes: Vec::new(),
            free_edges: Vec::new(),
            synced: true,
        }
    }

    /// A mirror for an adopted graph. Trustworthy **only** if the graph has no
    /// holes, because that is the only externally observable state whose free
    /// lists are known (empty).
    ///
    /// `node_count == node_bound` is the check, not `node_count > 0`: a
    /// compacted graph — a fresh load, a `vacuum` result, a bulk build — passes,
    /// and a graph carrying tombstones from earlier deletions does not.
    #[inline]
    pub(crate) fn for_adopted_graph(
        node_count: usize,
        node_bound: usize,
        edge_count: usize,
        edge_bound: usize,
    ) -> Self {
        Self {
            free_nodes: Vec::new(),
            free_edges: Vec::new(),
            synced: node_count == node_bound && edge_count == edge_bound,
        }
    }

    /// The index `add_node` will return next, or `None` when unsynced.
    ///
    /// `bound` is the caller's `node_bound()`; the mirror deliberately does not
    /// hold its own copy, so it cannot drift out of step with the graph it
    /// mirrors.
    #[inline]
    pub(crate) fn predict_next_node(&self, bound: usize) -> Option<NodeIndex> {
        if !self.synced {
            return None;
        }
        Some(match self.free_nodes.last() {
            Some(&slot) => NodeIndex::new(slot as usize),
            None => NodeIndex::new(bound),
        })
    }

    /// The index `add_edge` will return next, or `None` when unsynced.
    #[inline]
    pub(crate) fn predict_next_edge(&self, bound: usize) -> Option<EdgeIndex> {
        if !self.synced {
            return None;
        }
        Some(match self.free_edges.last() {
            Some(&slot) => EdgeIndex::new(slot as usize),
            None => EdgeIndex::new(bound),
        })
    }

    /// Record that `add_node` allocated `actual`, and **check the prediction
    /// that would have been served a moment earlier**.
    ///
    /// `bound_before` is `node_bound()` sampled before the insert, which is
    /// what makes this the full assertion rather than half of one: it validates
    /// the empty-free-list branch (predict == bound) as well as the reuse
    /// branch, so both halves of [`predict_next_node`](Self::predict_next_node)
    /// are exercised by every insert the test suites perform. In release the
    /// `debug_assert` and its argument vanish and this is a single `Vec::pop`.
    ///
    /// The unconditional `pop` is correct in both branches: when the free list
    /// is non-empty its head is exactly the slot petgraph just reused, and when
    /// it is empty `Vec::pop` is a no-op.
    #[inline]
    pub(crate) fn note_node_added(&mut self, bound_before: usize, actual: NodeIndex) {
        if !self.synced {
            return;
        }
        debug_assert_eq!(
            self.predict_next_node(bound_before),
            Some(actual),
            "slot mirror mispredicted a node slot; the free-list mirror has \
             drifted from petgraph (see storage/slot_mirror.rs)"
        );
        self.free_nodes.pop();
    }

    /// Record that `add_edge` allocated `actual`. See
    /// [`note_node_added`](Self::note_node_added) for why `bound_before` is
    /// taken and why the `pop` is unconditional.
    #[inline]
    pub(crate) fn note_edge_added(&mut self, bound_before: usize, actual: EdgeIndex) {
        if !self.synced {
            return;
        }
        debug_assert_eq!(
            self.predict_next_edge(bound_before),
            Some(actual),
            "slot mirror mispredicted an edge slot; the free-list mirror has \
             drifted from petgraph (see storage/slot_mirror.rs)"
        );
        self.free_edges.pop();
    }

    /// Record a removed node **and the edges petgraph frees with it**.
    ///
    /// `freed_edges` must arrive in petgraph's own removal order: outgoing
    /// adjacency order first, then incoming, with self-loops appearing once.
    /// See the module doc; the call sites in `storage/impls.rs` build it.
    #[inline]
    pub(crate) fn note_node_removed(
        &mut self,
        idx: NodeIndex,
        freed_edges: impl Iterator<Item = EdgeIndex>,
    ) {
        if !self.synced {
            return;
        }
        for edge in freed_edges {
            self.free_edges.push(edge.index() as u32);
        }
        self.free_nodes.push(idx.index() as u32);
    }

    /// Record a directly removed edge.
    #[inline]
    pub(crate) fn note_edge_removed(&mut self, idx: EdgeIndex) {
        if !self.synced {
            return;
        }
        self.free_edges.push(idx.index() as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::schema::{EdgeData, NodeData};
    use crate::graph::storage::interner::StringInterner;
    use petgraph::stable_graph::StableDiGraph;
    use petgraph::visit::{EdgeIndexable, NodeIndexable};
    use std::collections::HashMap;

    fn node(i: u64, interner: &mut StringInterner) -> NodeData {
        NodeData::new(
            Value::Int64(i as i64),
            Value::String(format!("n{i}")),
            "Item".to_string(),
            HashMap::new(),
            interner,
        )
    }

    fn edge(interner: &mut StringInterner) -> EdgeData {
        EdgeData::new("LINKS".to_string(), HashMap::new(), interner)
    }

    /// The mirror must agree with petgraph over a removal/insertion sequence
    /// that actually exercises slot reuse in both arrays.
    ///
    /// This is the test that would have caught a LIFO/FIFO mix-up, and it
    /// asserts the *prediction* before each insert rather than only after —
    /// `note_*_added`'s own `debug_assert` is the continuous guard, and this is
    /// the deliberate one.
    #[test]
    fn the_mirror_predicts_every_slot_petgraph_actually_allocates() {
        let mut interner = StringInterner::new();
        let mut g: StableDiGraph<NodeData, EdgeData> = StableDiGraph::new();
        let mut mirror = SlotMirror::for_empty_graph();

        let mut nodes = Vec::new();
        for i in 0..6 {
            let bound_before = g.node_bound();
            let predicted = mirror.predict_next_node(bound_before).expect("synced");
            let actual = g.add_node(node(i, &mut interner));
            assert_eq!(predicted, actual, "node insert {i}");
            mirror.note_node_added(bound_before, actual);
            nodes.push(actual);
        }

        let mut edges = Vec::new();
        for pair in [(0, 1), (1, 2), (2, 3), (0, 3), (3, 3)] {
            let bound_before = g.edge_bound();
            let predicted = mirror.predict_next_edge(bound_before).expect("synced");
            let actual = g.add_edge(nodes[pair.0], nodes[pair.1], edge(&mut interner));
            assert_eq!(predicted, actual, "edge insert {pair:?}");
            mirror.note_edge_added(bound_before, actual);
            edges.push(actual);
        }

        // A plain edge removal, then reuse.
        g.remove_edge(edges[1]).expect("edge present");
        mirror.note_edge_removed(edges[1]);
        let bound_before = g.edge_bound();
        let predicted = mirror.predict_next_edge(bound_before).expect("synced");
        let actual = g.add_edge(nodes[4], nodes[5], edge(&mut interner));
        assert_eq!(predicted, actual, "edge slot must be reused LIFO");
        mirror.note_edge_added(bound_before, actual);

        // A node removal that also frees incident edges, including a self-loop
        // (node 3 carries `(2,3)`, `(0,3)` incoming and `(3,3)` both ways).
        let victim = nodes[3];
        let freed = crate::graph::storage::impls::freed_edges_for_removal(&g, victim);
        g.remove_node(victim).expect("node present");
        mirror.note_node_removed(victim, freed.into_iter());

        // Every freed edge slot must now come back in petgraph's order.
        for i in 0..3 {
            let bound_before = g.edge_bound();
            let predicted = mirror.predict_next_edge(bound_before).expect("synced");
            let actual = g.add_edge(nodes[0], nodes[1], edge(&mut interner));
            assert_eq!(predicted, actual, "freed edge slot {i} must match petgraph");
            mirror.note_edge_added(bound_before, actual);
        }

        let bound_before = g.node_bound();
        let predicted = mirror.predict_next_node(bound_before).expect("synced");
        let actual = g.add_node(node(99, &mut interner));
        assert_eq!(predicted, actual, "freed node slot must be reused");
        mirror.note_node_added(bound_before, actual);
    }

    /// A graph with holes cannot have its free-list order known, so the mirror
    /// must refuse to predict rather than guess.
    #[test]
    fn an_adopted_graph_with_holes_refuses_to_predict() {
        let compact = SlotMirror::for_adopted_graph(10, 10, 4, 4);
        assert_eq!(
            compact.predict_next_node(10),
            Some(NodeIndex::new(10)),
            "a hole-free graph has provably empty free lists, so it may predict"
        );

        let holed = SlotMirror::for_adopted_graph(9, 10, 4, 4);
        assert_eq!(
            holed.predict_next_node(10),
            None,
            "a node hole means an unknown free-list order: refuse, never guess"
        );
        assert_eq!(holed.predict_next_edge(4), None);

        let edge_holed = SlotMirror::for_adopted_graph(10, 10, 3, 4);
        assert_eq!(
            edge_holed.predict_next_node(10),
            None,
            "an edge hole counts too"
        );
    }
}
