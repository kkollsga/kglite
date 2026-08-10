//! One node type's index bucket, held as a **stack of shared, immutable
//! levels** so a fork copies pointers instead of a million `NodeIndex`es.
//!
//! ## Why
//!
//! After D2 Phase 2 removed the backend row and Phase 3 removed `id_indices`,
//! `type_indices` was the **only O(V) field left in `DirGraph::clone`** on a
//! plain graph — ~80 µs of a 128 µs held-view round at 1M nodes. The bucket is
//! a `Vec<NodeIndex>` with one entry per node of the type, so the derived clone
//! copied it whole on every fork.
//!
//! ## Why append-only levels, and not per-bucket copy-on-write
//!
//! The realistic shape is *one* type holding nearly every node, so a
//! copy-on-write that materialises the bucket on first touch would copy the
//! whole 1M-entry `Vec` on the first `CREATE` and win nothing. What a `CREATE`
//! actually does to this structure is **append**, and an append needs no access
//! to what came before.
//!
//! So the bucket is a `Vec<Arc<Vec<NodeIndex>>>`: levels in order, the merged
//! content is their concatenation, and the **last level is the writable tail**.
//! A push writes into the tail when this graph uniquely owns it, and starts a
//! fresh level when it does not. That makes every operation this design cares
//! about O(1):
//!
//! | | cost |
//! |---|---|
//! | fork (`Clone`) | O(depth) `Arc` clones — no element is copied |
//! | append (`CREATE`) | one `Arc::get_mut` probe + a `Vec::push` |
//! | read, unlayered | **identical to before** — one level is handed out as a plain slice |
//! | read, layered | one branch per level boundary |
//! | compaction | O(delta): the base level is *moved*, never copied |
//!
//! ## Why the tail is an `Arc` too, rather than a plain owned `Vec`
//!
//! Because `Clone` only has `&self`. `IdIndexStore` solves the same problem by
//! taking a write lock and converting the *parent* in place
//! (`disk/id_index_layer.rs`), which this store cannot do: its reads hand out
//! borrowed slices, so it has no lock to borrow through. Keeping every level —
//! including the one being written — behind an `Arc` means a fork is a pointer
//! copy from `&self` alone, and the writer discovers the fork lazily, at its
//! next push, through `Arc::get_mut`.
//!
//! This is the trap D2 Phase 3 §B.2 recorded, avoided by construction: a
//! `share()` that has to merge before it can hand back one immutable value is
//! O(N) *on every fork that follows a write*, which is the founding defect's
//! own shape.
//!
//! ## Removals flatten; that is deliberate
//!
//! A delete, a `retain`, or a positional rollback insert needs a single mutable
//! `Vec`, so [`TypeBucket::to_mut`] merges the levels first — O(N) when the
//! bucket is genuinely shared, free when it is not. This mirrors the Phase 2
//! boundary where `remove_node` flattens the backend overlay: the cost is paid
//! **once per fork**, not per statement, and it keeps the delete path's
//! semantics byte-identical to the pre-layer code.
//!
//! ## Rollback
//!
//! Unlike `id_indices`, this field *does* carry undo entries
//! (`BucketId::NodeType`), and their reversal must land in this graph's own
//! delta rather than in a base another graph is reading:
//!
//! - [`TypeBucket::undo_append`] reverses `BucketAppended` by removing the
//!   entry from the writable tail. A statement's appends are all in the tail —
//!   no fork can interleave with a statement, because a fork needs `&DirGraph`
//!   while the writer holds `&mut DirGraph` — so this is O(tail). It falls back
//!   to a flatten-and-retain if it ever does not hold, which is slower and
//!   still correct.
//! - `BucketRemoved` goes through `entry_or_default`, i.e. [`to_mut`], which
//!   flattens. The delete that recorded the entry had already flattened the
//!   bucket for the same reason, so in practice the entry is already owned and
//!   the positional insert lands in exactly the `Vec` the position was measured
//!   against.
//!
//! [`to_mut`]: TypeBucket::to_mut

use std::sync::Arc;

use petgraph::graph::NodeIndex;

/// How many levels may stack before a push flattens them.
///
/// Every level is one extra branch on a full-bucket scan and one retained
/// allocation, so an unbounded stack would trade the fork cost for a read cost
/// and a leak. A stack only grows while a reader is *continuously* held — any
/// write with nothing shared compacts it back to one level — so in practice it
/// stays at 1.
///
/// The value matches `id_index_layer::MAX_CHAIN_DEPTH` and for the same
/// measured reason: the flatten is O(N_type), so its amortised cost is
/// `|bucket| / K` per fork, and K = 8 put a spike of ~5x the median into one
/// round in eight (D2 Phase 3 residual profile §B). K = 32 keeps the worst case
/// near 2x the median without retaining a deep stack of deltas.
const MAX_LAYER_DEPTH: usize = 32;

/// One node type's member list.
///
/// Empty `levels` is the empty bucket. One level is the steady state and reads
/// exactly like the plain `Vec` this replaced; more than one means a fork is
/// outstanding.
#[derive(Debug, Clone, Default)]
pub struct TypeBucket {
    levels: Vec<Arc<Vec<NodeIndex>>>,
}

impl From<Vec<NodeIndex>> for TypeBucket {
    #[inline]
    fn from(members: Vec<NodeIndex>) -> Self {
        Self {
            levels: vec![Arc::new(members)],
        }
    }
}

impl TypeBucket {
    /// The levels, in merge order. Empty for an empty bucket.
    #[inline]
    pub fn levels(&self) -> &[Arc<Vec<NodeIndex>>] {
        &self.levels
    }

    /// Append one member. **The hot path** — every `CREATE` lands here.
    ///
    /// Writes into the tail level when this graph owns it outright, and starts
    /// a fresh level when a fork is holding it. Both are O(1); neither can
    /// touch memory another graph is reading.
    #[inline]
    pub fn push(&mut self, idx: NodeIndex) {
        if let Some(tail) = self.levels.last_mut() {
            if let Some(tail) = Arc::get_mut(tail) {
                tail.push(idx);
                return;
            }
        }
        if self.levels.len() >= MAX_LAYER_DEPTH {
            // Bounded: reaching here needs MAX_LAYER_DEPTH consecutive
            // fork-then-write rounds with a reader held throughout.
            self.flatten();
            // `flatten` leaves exactly one, uniquely-owned level.
            Arc::get_mut(&mut self.levels[0])
                .expect("flatten produces a uniquely owned level")
                .push(idx);
            return;
        }
        self.levels.push(Arc::new(vec![idx]));
    }

    /// The bucket as one mutable `Vec`, merging the levels if it is layered.
    ///
    /// O(1) when this graph already owns a single level — the steady state and
    /// every unforked delete. O(N) otherwise, which is the deliberate boundary
    /// described in the module doc: removals flatten rather than grow a masking
    /// structure that every read would have to consult.
    pub fn to_mut(&mut self) -> &mut Vec<NodeIndex> {
        let owned_single = self.levels.len() == 1 && Arc::get_mut(&mut self.levels[0]).is_some();
        if !owned_single {
            self.flatten();
        }
        if self.levels.is_empty() {
            self.levels.push(Arc::new(Vec::new()));
        }
        Arc::get_mut(&mut self.levels[0]).expect("a flattened bucket owns its single level")
    }

    /// Merge every level into one uniquely-owned level.
    ///
    /// The leading level is **moved** out of its `Arc` when nothing else holds
    /// it, so the common shape (one big base plus small deltas) merges in
    /// O(delta) rather than O(N). Written this way on purpose: the obvious
    /// spelling — collect everything into a fresh `Vec` — is the O(N)
    /// compaction trap recorded in D2 Phase 3 §B.1, which turns every write
    /// after a dropped reader into a full copy.
    fn flatten(&mut self) {
        if self.levels.len() <= 1 {
            if self.levels.len() == 1 && Arc::get_mut(&mut self.levels[0]).is_none() {
                let copy = self.levels[0].as_ref().clone();
                self.levels[0] = Arc::new(copy);
            }
            return;
        }
        let mut levels = std::mem::take(&mut self.levels).into_iter();
        let first = levels.next().expect("length checked above");
        let mut merged = Arc::try_unwrap(first).unwrap_or_else(|shared| shared.as_ref().clone());
        for level in levels {
            match Arc::try_unwrap(level) {
                Ok(owned) => merged.extend(owned),
                Err(shared) => merged.extend_from_slice(shared.as_ref()),
            }
        }
        self.levels.push(Arc::new(merged));
    }

    /// Fold the levels back into one when this graph is their last holder.
    ///
    /// Called at write entry, so "hold a view, write, drop the view, write
    /// again" returns to the flat representation on the very next write. A
    /// still-shared leading level is left alone and only the owned suffix
    /// merges, which bounds the depth without copying the base.
    pub fn try_compact(&mut self) {
        if self.levels.len() <= 1 {
            return;
        }
        // The first level this graph owns outright; everything from there on
        // can be merged without touching memory a reader holds.
        let first_owned = self
            .levels
            .iter_mut()
            .position(|level| Arc::get_mut(level).is_some());
        let Some(first_owned) = first_owned else {
            return;
        };
        if first_owned + 1 >= self.levels.len() {
            return;
        }
        let tail = self.levels.split_off(first_owned);
        let mut tail = tail.into_iter();
        let first = tail.next().expect("split_off yields at least one level");
        let mut merged = Arc::try_unwrap(first).expect("position proved sole ownership");
        for level in tail {
            match Arc::try_unwrap(level) {
                Ok(owned) => merged.extend(owned),
                Err(shared) => merged.extend_from_slice(shared.as_ref()),
            }
        }
        self.levels.push(Arc::new(merged));
    }

    /// Reverse one journalled append: drop `idx` from this graph's own delta.
    ///
    /// Returns `false` when `idx` is not in the writable tail, which tells the
    /// caller to fall back to the flatten-and-retain path. See the module doc
    /// for why the fast path is the one that fires.
    ///
    /// Removes *every* occurrence in the tail rather than the last one, which
    /// is what the pre-layer `retain_in_type` reversal did. The two agree here
    /// because a `CREATE`'s append carries a freshly allocated `NodeIndex`: the
    /// slot was free a moment earlier, so no pre-statement occurrence of it can
    /// exist to be spared.
    pub fn undo_append(&mut self, idx: NodeIndex) -> bool {
        let Some(last) = self.levels.len().checked_sub(1) else {
            return false;
        };
        let Some(tail) = Arc::get_mut(&mut self.levels[last]) else {
            return false;
        };
        if !tail.contains(&idx) {
            return false;
        }
        tail.retain(|member| *member != idx);
        let tail_emptied = tail.is_empty();
        debug_assert!(
            !self.levels[..last].iter().any(|level| level.contains(&idx)),
            "a journalled append was also present in a shared level: a fork \
             interleaved with a statement, which the undo re-pointing assumes \
             cannot happen (storage/disk/type_index_layer.rs)"
        );
        if last > 0 && tail_emptied {
            self.levels.pop();
        }
        true
    }

    #[cfg(test)]
    pub(crate) fn depth(&self) -> usize {
        self.levels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bucket(members: &[usize]) -> TypeBucket {
        TypeBucket::from(
            members
                .iter()
                .map(|i| NodeIndex::new(*i))
                .collect::<Vec<_>>(),
        )
    }

    fn members(bucket: &TypeBucket) -> Vec<usize> {
        bucket
            .levels()
            .iter()
            .flat_map(|level| level.iter())
            .map(|idx| idx.index())
            .collect()
    }

    /// A fork copies no element, and each side's appends are invisible to the
    /// other — the property the whole layer exists for.
    #[test]
    fn a_fork_shares_the_members_and_isolates_the_appends() {
        let mut parent = bucket(&[1, 2, 3]);
        let mut child = parent.clone();

        assert!(
            Arc::ptr_eq(&parent.levels()[0], &child.levels()[0]),
            "the fork must share the base allocation, not copy it"
        );

        child.push(NodeIndex::new(4));
        parent.push(NodeIndex::new(5));

        assert_eq!(members(&child), vec![1, 2, 3, 4]);
        assert_eq!(members(&parent), vec![1, 2, 3, 5]);
        assert!(
            Arc::ptr_eq(&parent.levels()[0], &child.levels()[0]),
            "neither side may have copied the shared base to write"
        );
        assert_eq!(members(&child).len(), 4);
        assert_eq!(members(&parent).len(), 4);
    }

    /// With nothing shared, a push stays in the single level — no growth, and
    /// therefore no read-path cost in the steady state.
    #[test]
    fn an_unshared_bucket_never_grows_a_level() {
        let mut solo = bucket(&[1]);
        for i in 2..50 {
            solo.push(NodeIndex::new(i));
        }
        assert_eq!(solo.depth(), 1);
        assert_eq!(members(&solo).len(), 49);
    }

    /// Compaction folds the delta back once the reader drops, and it must not
    /// fire while the reader is alive.
    #[test]
    fn compaction_waits_for_the_reader_and_then_folds() {
        let mut writer = bucket(&[1, 2]);
        let reader = writer.clone();
        writer.push(NodeIndex::new(3));
        assert_eq!(writer.depth(), 2);

        writer.try_compact();
        assert_eq!(
            writer.depth(),
            2,
            "a live reader must block the fold — the base is its data"
        );
        assert_eq!(members(&reader), vec![1, 2]);

        drop(reader);
        writer.try_compact();
        assert_eq!(writer.depth(), 1, "the last holder must collapse");
        assert_eq!(members(&writer), vec![1, 2, 3]);
    }

    /// Compaction must move the base rather than copy it — the D2 Phase 3 §B.1
    /// trap, where a fold spelled as "materialise into a fresh Vec" turned every
    /// post-reader write into a full copy.
    #[test]
    fn compaction_moves_the_base_allocation_instead_of_copying_it() {
        // Spare capacity so that appending during the fold cannot reallocate
        // for an ordinary `Vec` growth reason — any pointer change then means
        // the fold rebuilt the base.
        let mut base = Vec::with_capacity(16);
        base.extend([1, 2, 3].map(NodeIndex::new));
        let mut writer = TypeBucket::from(base);

        let reader = writer.clone();
        writer.push(NodeIndex::new(4));
        assert_eq!(writer.depth(), 2, "a live reader forces a new level");

        drop(reader);
        let base_ptr = writer.levels()[0].as_ptr();
        writer.try_compact();

        assert_eq!(writer.depth(), 1);
        assert_eq!(
            writer.levels()[0].as_ptr(),
            base_ptr,
            "the fold must move the base allocation, not rebuild it — the O(N) \
             compaction trap from D2 Phase 3 §B.1"
        );
        assert_eq!(members(&writer), vec![1, 2, 3, 4]);
    }

    /// A reader held across many fork-then-write rounds must not stack levels
    /// without bound, and must still read correctly at the cap.
    #[test]
    fn a_never_compacted_stack_stays_bounded_and_correct() {
        let mut writer = bucket(&[0]);
        let mut readers = Vec::new();
        for round in 1..=(MAX_LAYER_DEPTH * 3) {
            readers.push(writer.clone());
            writer.push(NodeIndex::new(round));
        }

        assert!(
            writer.depth() <= MAX_LAYER_DEPTH,
            "level stack {} exceeded the cap",
            writer.depth()
        );
        assert_eq!(
            members(&writer),
            (0..=MAX_LAYER_DEPTH * 3).collect::<Vec<_>>(),
            "every append must survive the flattens, in order"
        );
        assert_eq!(
            members(&readers[0]),
            vec![0],
            "the oldest reader sees its own snapshot"
        );

        drop(readers);
        writer.try_compact();
        assert_eq!(writer.depth(), 1, "nothing shared: the stack collapses");
    }

    /// `to_mut` is the removal path. It must merge without disturbing a reader,
    /// and it must be free when nothing is shared.
    #[test]
    fn to_mut_merges_for_a_removal_and_leaves_the_reader_intact() {
        let mut writer = bucket(&[1, 2, 3]);
        let reader = writer.clone();
        writer.push(NodeIndex::new(4));

        writer.to_mut().retain(|idx| idx.index() != 2);

        assert_eq!(members(&writer), vec![1, 3, 4]);
        assert_eq!(members(&reader), vec![1, 2, 3]);

        // Unshared now: `to_mut` hands back the same allocation.
        drop(reader);
        let ptr = writer.levels()[0].as_ptr();
        writer.to_mut().push(NodeIndex::new(9));
        assert_eq!(writer.levels()[0].as_ptr(), ptr);
    }

    /// The undo re-pointing: reversing an append edits this graph's tail, never
    /// the shared base a reader is holding.
    #[test]
    fn undo_append_edits_the_delta_and_never_the_shared_base() {
        let mut writer = bucket(&[1, 2]);
        let reader = writer.clone();
        writer.push(NodeIndex::new(3));

        assert!(writer.undo_append(NodeIndex::new(3)));
        assert_eq!(members(&writer), vec![1, 2]);
        assert_eq!(members(&reader), vec![1, 2]);
        assert!(
            Arc::ptr_eq(&writer.levels()[0], &reader.levels()[0]),
            "the reversal must not have copied or edited the base"
        );

        // An entry that is not in the tail is refused, so the caller falls back
        // to the flattening path rather than silently dropping the reversal.
        assert!(!writer.undo_append(NodeIndex::new(1)));
        assert_eq!(members(&writer), vec![1, 2]);
    }
}
