//! [`NodeRemap`] — the old→new node-index mapping a [`vacuum`] hands back.
//!
//! [`vacuum`]: super::DirGraph::vacuum
//!
//! # Why this is not a `HashMap`
//!
//! It was one, and the rebuild built *both*: a dense `Vec<u32>` for its own
//! edge pass (endpoint remapping is two probes per edge and SipHash was ~22% of
//! a fired vacuum at 1M), and a `HashMap<NodeIndex, NodeIndex>` carrying the
//! same information for the caller — an O(V) hash insert per live node plus
//! ~30 MB of transient allocation at 1M nodes, on top of the vector that
//! already knew the answer.
//!
//! The graph is index-addressed, so the vector *is* the natural map. This type
//! is that vector with the map's interface: `get` is one bounds-checked load,
//! `len` is a counter the rebuild keeps as it goes, and the only consumer that
//! needs the pairs (`remap_embedding_slots`) iterates them.

use petgraph::graph::NodeIndex;

/// The old→new node index mapping produced by a `vacuum` rebuild.
///
/// Empty when nothing was remapped (a no-op vacuum, or a backend that does not
/// compact in place).
#[derive(Debug, Clone, Default)]
pub struct NodeRemap {
    /// `dense[old_raw]` = new raw index, or [`Self::VACANT`] for a slot that
    /// held no live node.
    dense: Vec<u32>,
    /// How many slots are occupied — the number of nodes actually carried over.
    live: usize,
}

impl NodeRemap {
    /// Marks an old slot that held no live node.
    const VACANT: u32 = u32::MAX;

    /// A mapping over `bound` old slots, all vacant.
    pub(super) fn with_bound(bound: usize) -> Self {
        NodeRemap {
            dense: vec![Self::VACANT; bound],
            live: 0,
        }
    }

    /// Record that `old` was carried over to `new`.
    pub(super) fn set(&mut self, old: usize, new: NodeIndex) {
        self.dense[old] = new.index() as u32;
        self.live += 1;
    }

    /// The new index for `old`, or `None` when that node did not survive.
    #[inline]
    pub fn get(&self, old: NodeIndex) -> Option<NodeIndex> {
        match self.dense.get(old.index()).copied() {
            Some(Self::VACANT) | None => None,
            Some(new) => Some(NodeIndex::new(new as usize)),
        }
    }

    /// The new *raw* index for an old raw index, for the rebuild's own edge
    /// pass — no `NodeIndex` round-trip, no `Option`.
    #[inline]
    pub(super) fn raw(&self, old_raw: usize) -> u32 {
        self.dense[old_raw]
    }

    /// Whether `raw` reported a slot that held no live node.
    #[inline]
    pub(super) fn is_vacant(raw: u32) -> bool {
        raw == Self::VACANT
    }

    /// How many nodes were remapped.
    #[inline]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether nothing was remapped.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Whether this mapping describes a rebuild at all — i.e. whether the old
    /// indices a caller is holding have been invalidated.
    ///
    /// **This is the question a holder of node indices must ask, and it is not
    /// [`Self::is_empty`].** A `NodeRemap::default()` is what every *no-op*
    /// vacuum returns — the disk backend, which compacts by publishing a fresh
    /// generation rather than rebuilding in place, and the columnar-only
    /// reclaim that drops dead rows without touching a single node slot. There
    /// the old indices are still exactly right and a holder must keep them.
    ///
    /// `is_empty` is also true for a rebuild whose survivors numbered *zero*
    /// (every node deleted), and there the opposite is required: every old
    /// index is genuinely gone. The two cases differ in whether the mapping
    /// covers any slots, which is what this reads.
    #[inline]
    pub fn describes_rebuild(&self) -> bool {
        !self.dense.is_empty()
    }

    /// Every `(old, new)` pair, in ascending old-index order.
    pub fn iter(&self) -> impl Iterator<Item = (NodeIndex, NodeIndex)> + '_ {
        self.dense
            .iter()
            .enumerate()
            .filter(|(_, &new)| new != Self::VACANT)
            .map(|(old, &new)| (NodeIndex::new(old), NodeIndex::new(new as usize)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vacant_slots_are_absent_and_uncounted() {
        let mut remap = NodeRemap::with_bound(4);
        remap.set(0, NodeIndex::new(0));
        remap.set(3, NodeIndex::new(1));

        assert_eq!(remap.len(), 2);
        assert!(!remap.is_empty());
        assert_eq!(remap.get(NodeIndex::new(0)), Some(NodeIndex::new(0)));
        assert_eq!(remap.get(NodeIndex::new(1)), None);
        assert_eq!(remap.get(NodeIndex::new(3)), Some(NodeIndex::new(1)));
        // Out of range reads as absent, exactly like a missing map entry.
        assert_eq!(remap.get(NodeIndex::new(9)), None);
        assert_eq!(
            remap.iter().collect::<Vec<_>>(),
            vec![
                (NodeIndex::new(0), NodeIndex::new(0)),
                (NodeIndex::new(3), NodeIndex::new(1)),
            ]
        );
    }

    #[test]
    fn an_untouched_mapping_is_empty() {
        assert!(NodeRemap::default().is_empty());
        assert!(NodeRemap::with_bound(8).is_empty());
        assert_eq!(NodeRemap::with_bound(8).len(), 0);
    }

    #[test]
    fn a_no_op_vacuum_is_distinguishable_from_a_rebuild_with_no_survivors() {
        // The disk / columnar-only shape: no mapping at all, old indices stand.
        assert!(!NodeRemap::default().describes_rebuild());

        // A rebuild that carried nobody over: `is_empty` agrees with the
        // no-op above, `describes_rebuild` does not — and a holder acting on
        // `is_empty` would keep indices into a graph that no longer has them.
        let wiped = NodeRemap::with_bound(8);
        assert!(wiped.is_empty());
        assert!(wiped.describes_rebuild());

        let mut kept = NodeRemap::with_bound(2);
        kept.set(1, NodeIndex::new(0));
        assert!(kept.describes_rebuild());
    }
}
