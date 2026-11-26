//! The adjacency half of the fork overlay: edges added, removed or re-weighted
//! behind a base that a reader still holds.
//!
//! ## Shape
//!
//! `StableDiGraph` threads adjacency through per-node linked lists with head
//! insertion, so a node's edges read newest-first and a removal only unlinks.
//! The folded graph's read order is therefore exactly "edges added since the
//! fork, newest first, then the base's own list minus the removed ones". That
//! sentence is the whole read contract; every iterator in
//! [`forked_edge_iters`](super::forked_edge_iters) implements it, and the
//! randomised model test pins it against a graph that never forks.
//!
//! | state | holds |
//! |---|---|
//! | `added` | edges the overlay allocated: slot -> endpoints + weight |
//! | `out_head` / `in_head` | per node, the newest `added` edge; older ones follow the chain through the edges themselves |
//! | `removed` | base slots that are not live in this view |
//! | `weights` | base edges whose weight was written (copy-on-write) |
//! | `touched` | a superset gate: one bit test answers "may the base be wrong about this slot?" |
//!
//! A slot can sit in `removed` and in `added` at once: the overlay freed a base
//! edge and then reused its slot. Base reads consult `removed`; overlay reads
//! consult `added`; neither sees the other's entry.
//!
//! Slot identity (the issue #195 class) is not decided here: the slot an edge
//! takes comes from the fork's `SlotMirror`, and the fold replays the operation
//! log against the base and checks every slot petgraph hands back
//! (`forked.rs`, "Slot identity").

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;

use crate::graph::schema::EdgeData;
use crate::graph::storage::fork_base::ForkBase;
use crate::graph::storage::record_ref::RecordRef;
use crate::graph::storage::shared_tree::{PagedBits, PagedMap, PagedSet};

/// Words per chunk of a chunked [`OverlayBits`]: 32,768 bits, 4 KiB.
const CHUNK_WORDS: usize = 512;

/// A set of small integers as a bitmap, grown on demand.
///
/// One bit test answers reads the overlay holds nothing for, which is nearly
/// all of them; a hash probe costs about 2.6x more when the held keys are
/// scattered (measured 2026-10-05, see `forked.rs` `OverlayNodes`). Flat rather
/// than paged for an overlay over a heap graph, because a node scan tests it
/// once per node, where a second dependent load measured +12 % on a scan under
/// a held overlay; and a flat probe that can fall through to a second layout
/// must pay nothing when it lands in the words (an enum of the two layouts
/// measured +6 % on the same scan).
///
/// An overlay over a frozen base is the other layout: its bitmap spans every
/// node the writes ever reached (3 KB per 24,000 nodes) and each transaction
/// forks it, so it is held in `Arc`ed chunks (`words` stays empty) that a fork
/// shares and a write copies one of.
#[derive(Clone, Default)]
pub(crate) struct OverlayBits {
    words: Vec<u64>,
    chunks: Vec<std::sync::Arc<[u64; CHUNK_WORDS]>>,
    chunked: bool,
}

#[inline(never)]
fn chunked_holds(chunks: &[std::sync::Arc<[u64; CHUNK_WORDS]>], idx: u32) -> bool {
    chunks
        .get((idx >> 15) as usize)
        .is_some_and(|chunk| (chunk[((idx >> 6) as usize) % CHUNK_WORDS] >> (idx & 63)) & 1 == 1)
}

impl OverlayBits {
    /// An empty set that forks in O(chunks).
    pub(crate) fn chunked() -> Self {
        Self {
            chunked: true,
            ..Self::default()
        }
    }

    /// An empty set of the same layout.
    pub(crate) fn emptied(&self) -> Self {
        Self {
            chunked: self.chunked,
            ..Self::default()
        }
    }

    #[inline]
    pub(crate) fn holds(&self, idx: u32) -> bool {
        match self.words.get((idx >> 6) as usize) {
            Some(word) => (word >> (idx & 63)) & 1 == 1,
            None => !self.chunks.is_empty() && chunked_holds(&self.chunks, idx),
        }
    }

    pub(crate) fn set(&mut self, idx: u32) {
        if self.chunked {
            let chunk = (idx >> 15) as usize;
            if chunk >= self.chunks.len() {
                self.chunks
                    .resize_with(chunk + 1, || std::sync::Arc::new([0; CHUNK_WORDS]));
            }
            std::sync::Arc::make_mut(&mut self.chunks[chunk])
                [((idx >> 6) as usize) % CHUNK_WORDS] |= 1 << (idx & 63);
            return;
        }
        let word = (idx >> 6) as usize;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << (idx & 63);
    }

    pub(crate) fn clear(&mut self, idx: u32) {
        if self.chunked {
            if let Some(chunk) = self.chunks.get_mut((idx >> 15) as usize) {
                std::sync::Arc::make_mut(chunk)[((idx >> 6) as usize) % CHUNK_WORDS] &=
                    !(1 << (idx & 63));
            }
        } else if let Some(word) = self.words.get_mut((idx >> 6) as usize) {
            *word &= !(1 << (idx & 63));
        }
    }
}

/// End of an overlay adjacency chain.
const NIL: u32 = u32::MAX;

/// An edge the overlay allocated, linked into the chain of the edges added at
/// its source and at its target (each newest first).
#[derive(Clone)]
pub(crate) struct OverlayEdge {
    pub(crate) src: u32,
    pub(crate) dst: u32,
    pub(crate) weight: EdgeData,
    next_out: u32,
    next_in: u32,
}

/// One node's overlay-added edges in one direction, newest first.
pub(crate) struct AddedChain<'a> {
    added: &'a PagedMap<OverlayEdge>,
    next: u32,
    outgoing: bool,
}

impl<'a> Iterator for AddedChain<'a> {
    type Item = (u32, &'a OverlayEdge);

    #[inline]
    fn next(&mut self) -> Option<(u32, &'a OverlayEdge)> {
        if self.next == NIL {
            return None;
        }
        let slot = self.next;
        let edge = self
            .added
            .get(slot)
            .expect("a chained overlay edge is always in the added map");
        self.next = if self.outgoing {
            edge.next_out
        } else {
            edge.next_in
        };
        Some((slot, edge))
    }
}

/// The overlay's edge delta. Empty (`is_clean`) for a fork that only wrote
/// nodes, and then every read takes the base's own iterators.
#[derive(Clone, Default)]
pub(crate) struct EdgeLayer {
    pub(crate) added: PagedMap<OverlayEdge>,
    /// Per node, the newest added edge at it; the rest follow the chain in
    /// [`OverlayEdge`]. Chains instead of a `Vec` per node, so cloning an
    /// overlay (which a transaction does) allocates nothing per node.
    out_head: PagedMap<u32>,
    in_head: PagedMap<u32>,
    pub(crate) removed: PagedSet,
    pub(crate) weights: PagedMap<EdgeData>,
    pub(crate) touched: PagedBits,
    /// Nodes an overlay edge, a removed base edge or a re-weighted base edge is
    /// incident to. A node outside it reads exactly as the base reads it, so
    /// its edges come straight from the base's own iterator.
    dirty_nodes: PagedBits,
    /// One past the highest slot the overlay ever allocated; with the base's
    /// bound it is the slot an append takes when no slot has been vacated.
    alloc_bound: usize,
}

impl EdgeLayer {
    #[inline]
    pub(crate) fn is_clean(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.weights.is_empty()
    }

    /// Whether `node` has any edge the overlay changed or added. `false` is a
    /// proof that the base's edges at `node` are this view's edges.
    #[inline]
    pub(crate) fn touches(&self, node: NodeIndex) -> bool {
        self.dirty_nodes.holds(node.index() as u32)
    }

    /// Edges live in this view minus the base's: `added - removed`.
    #[inline]
    pub(crate) fn count_delta(&self) -> isize {
        self.added.len() as isize - self.removed.len() as isize
    }

    /// Where an `add_edge` appends when the free list is empty: past every slot
    /// the overlay ever took, and past the base's.
    #[inline]
    pub(crate) fn append_bound(&self, base_bound: usize) -> usize {
        base_bound.max(self.alloc_bound)
    }

    /// One past the highest overlay-added slot, or 0.
    pub(crate) fn added_bound(&self) -> usize {
        self.added.max_key().map_or(0, |slot| slot as usize + 1)
    }

    /// The overlay-added edges incident to `node` in `dir`, newest first.
    #[inline]
    pub(crate) fn chain(&self, node: NodeIndex, dir: Direction) -> AddedChain<'_> {
        let heads = match dir {
            Direction::Outgoing => &self.out_head,
            Direction::Incoming => &self.in_head,
        };
        AddedChain {
            added: &self.added,
            next: heads.get(node.index() as u32).copied().unwrap_or(NIL),
            outgoing: dir == Direction::Outgoing,
        }
    }

    /// Register an edge the overlay allocated at `slot`.
    pub(crate) fn insert_added(&mut self, slot: u32, src: u32, dst: u32, weight: EdgeData) {
        let next_out = self.out_head.insert(src, slot).unwrap_or(NIL);
        let next_in = self.in_head.insert(dst, slot).unwrap_or(NIL);
        self.added.insert(
            slot,
            OverlayEdge {
                src,
                dst,
                weight,
                next_out,
                next_in,
            },
        );
        self.touched.set(slot);
        self.dirty_nodes.set(src);
        self.dirty_nodes.set(dst);
        self.alloc_bound = self.alloc_bound.max(slot as usize + 1);
    }

    /// Drop an overlay-added edge, unlinking it from both chains. Rollback and
    /// a tail delete remove what was added last, which heads its chains.
    pub(crate) fn take_added(&mut self, slot: u32) -> Option<OverlayEdge> {
        let edge = self.added.remove(slot)?;
        self.unlink(slot, edge.src, edge.next_out, true);
        self.unlink(slot, edge.dst, edge.next_in, false);
        Some(edge)
    }

    /// Remove `slot` from `node`'s chain, whose entry after it is `after`.
    fn unlink(&mut self, slot: u32, node: u32, after: u32, outgoing: bool) {
        let heads = if outgoing {
            &mut self.out_head
        } else {
            &mut self.in_head
        };
        let Some(&head) = heads.get(node) else {
            return;
        };
        if head == slot {
            if after == NIL {
                heads.remove(node);
            } else {
                heads.insert(node, after);
            }
            return;
        }
        let mut at = head;
        while let Some(edge) = self.added.get_mut(at) {
            let link = if outgoing {
                &mut edge.next_out
            } else {
                &mut edge.next_in
            };
            if *link == slot {
                *link = after;
                return;
            }
            at = *link;
        }
    }

    /// Mark a live base edge as no longer live in this view.
    pub(crate) fn tombstone(&mut self, slot: u32, src: u32, dst: u32) {
        self.removed.insert(slot);
        self.weights.remove(slot);
        self.touched.set(slot);
        self.dirty_nodes.set(src);
        self.dirty_nodes.set(dst);
    }

    /// The copy-on-write weight of a base edge, copied on first touch.
    pub(crate) fn weight_mut<'a>(
        &'a mut self,
        slot: u32,
        (src, dst): (u32, u32),
        base_weight: &EdgeData,
    ) -> &'a mut EdgeData {
        self.touched.set(slot);
        self.dirty_nodes.set(src);
        self.dirty_nodes.set(dst);
        self.weights
            .get_or_insert_with(slot, || base_weight.clone())
    }

    /// The weight of `slot` as this view reads it.
    #[inline]
    pub(crate) fn weight<'a>(
        &'a self,
        base: &'a ForkBase,
        slot: EdgeIndex,
    ) -> Option<RecordRef<'a, EdgeData>> {
        let raw = slot.index() as u32;
        if !self.touched.holds(raw) {
            return base.edge_weight(slot);
        }
        if let Some(edge) = self.added.get(raw) {
            return Some(RecordRef::Borrowed(&edge.weight));
        }
        if self.removed.contains(raw) {
            return None;
        }
        self.weights
            .get(raw)
            .map(RecordRef::Borrowed)
            .or_else(|| base.edge_weight(slot))
    }

    /// The endpoints of `slot` as this view reads them.
    #[inline]
    pub(crate) fn endpoints(
        &self,
        base: &ForkBase,
        slot: EdgeIndex,
    ) -> Option<(NodeIndex, NodeIndex)> {
        let raw = slot.index() as u32;
        if !self.touched.holds(raw) {
            return base.edge_endpoints(slot);
        }
        if let Some(edge) = self.added.get(raw) {
            return Some((
                NodeIndex::new(edge.src as usize),
                NodeIndex::new(edge.dst as usize),
            ));
        }
        if self.removed.contains(raw) {
            return None;
        }
        base.edge_endpoints(slot)
    }

    /// Overlay-added slots in ascending order, for the whole-graph scans.
    pub(crate) fn added_sorted(&self) -> Vec<u32> {
        let mut slots: Vec<u32> = self.added.keys().collect();
        slots.sort_unstable();
        slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::InternedKey;

    /// Both kinds of node bitmap hold exactly what was set and not cleared
    /// across chunk boundaries, and a clone of a chunked one keeps what it had.
    #[test]
    fn overlay_bits_of_both_kinds_agree_with_a_model_and_chunked_clones_are_isolated() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        for kind in 0..2 {
            let mut bits = if kind == 0 {
                OverlayBits::default()
            } else {
                OverlayBits::chunked()
            };
            let mut model = std::collections::BTreeSet::new();
            let mut snapshots = Vec::new();
            for step in 0..4_000u64 {
                let idx = next(300_000) as u32;
                match next(6) {
                    0..=2 => {
                        bits.set(idx);
                        model.insert(idx);
                    }
                    3 => {
                        bits.clear(idx);
                        model.remove(&idx);
                    }
                    4 if step % 7 == 0 && snapshots.len() < 5 => {
                        snapshots.push((bits.clone(), model.clone()));
                    }
                    _ => {}
                }
                assert_eq!(
                    bits.holds(idx),
                    model.contains(&idx),
                    "kind {kind} idx {idx}"
                );
            }
            for probe in (0..300_000u32).step_by(7).chain(model.iter().copied()) {
                assert_eq!(
                    bits.holds(probe),
                    model.contains(&probe),
                    "kind {kind} probe {probe}"
                );
            }
            for (snap, expected) in &snapshots {
                for probe in (0..300_000u32).step_by(11).chain(expected.iter().copied()) {
                    assert_eq!(
                        snap.holds(probe),
                        expected.contains(&probe),
                        "snapshot {probe}"
                    );
                }
            }
            assert!(!bits.emptied().holds(*model.iter().next().unwrap_or(&0)));
        }
    }

    /// Overlay edges chain newest first at both ends, and an edge leaves the
    /// chain whether it is the head, the middle or the tail.
    #[test]
    fn added_edges_chain_newest_first_and_unlink_from_anywhere() {
        let weight = || EdgeData::new_interned(InternedKey::from_u64(1), Vec::new());
        let out_of = |layer: &EdgeLayer, node: usize| -> Vec<u32> {
            layer
                .chain(NodeIndex::new(node), Direction::Outgoing)
                .map(|(slot, _)| slot)
                .collect()
        };
        let into = |layer: &EdgeLayer, node: usize| -> Vec<u32> {
            layer
                .chain(NodeIndex::new(node), Direction::Incoming)
                .map(|(slot, _)| slot)
                .collect()
        };
        let mut layer = EdgeLayer::default();
        // Slots 10..15 leave node 1; 12 and 14 arrive at node 2, the rest at 3;
        // 15 is a self-loop at node 1.
        for (slot, src, dst) in [
            (10, 1, 3),
            (11, 1, 3),
            (12, 1, 2),
            (13, 1, 3),
            (14, 1, 2),
            (15, 1, 1),
        ] {
            layer.insert_added(slot, src, dst, weight());
        }
        assert_eq!(out_of(&layer, 1), [15, 14, 13, 12, 11, 10]);
        assert_eq!(into(&layer, 3), [13, 11, 10]);
        assert_eq!(
            into(&layer, 1),
            [15],
            "a self-loop is on both of its node's chains"
        );
        for taken in [13, 15, 10, 14, 11, 12] {
            let expected_out: Vec<u32> = out_of(&layer, 1)
                .into_iter()
                .filter(|&s| s != taken)
                .collect();
            let expected_in3: Vec<u32> = into(&layer, 3)
                .into_iter()
                .filter(|&s| s != taken)
                .collect();
            let expected_in2: Vec<u32> = into(&layer, 2)
                .into_iter()
                .filter(|&s| s != taken)
                .collect();
            assert!(layer.take_added(taken).is_some());
            assert!(layer.take_added(taken).is_none(), "an edge leaves once");
            assert_eq!(out_of(&layer, 1), expected_out, "after taking {taken}");
            assert_eq!(into(&layer, 3), expected_in3, "after taking {taken}");
            assert_eq!(into(&layer, 2), expected_in2, "after taking {taken}");
        }
        assert!(layer.added.is_empty() && out_of(&layer, 1).is_empty());
        assert!(
            layer.out_head.is_empty() && layer.in_head.is_empty(),
            "emptied chains leave no head"
        );
    }
}
