//! One user-created **range** index's value → members map, held as a stack of
//! shared, immutable levels so a fork copies pointers instead of a B-tree.
//!
//! ## Why a second layer type
//!
//! This is [`super::index_layer::LayeredIndex`]'s mechanism — the same level
//! stack, the same `None`-is-a-tombstone rule, the same
//! materialise-the-merged-bucket contract with the undo journal, the same
//! depth cap — with **ordered** levels. It cannot be the same type: a range
//! index answers `lookup_range`, whose whole value is that the keys come out
//! sorted, and `LayeredIndex`'s levels are `HashMap`s. `K: Ord` and `K: Hash`
//! are different bounds over different level maps, and the fast paths that
//! make either type worth having (the single-level arm borrowing the
//! underlying map's own iterator) are spelled in terms of the concrete map.
//! The precedent is the module this one mirrors: `index_layer` is itself
//! `disk::type_index_layer`'s mechanism with a map payload instead of a `Vec`.
//!
//! ## What it fixes
//!
//! `range_indices` was a plain `HashMap<IndexKey, BTreeMap<Value, Vec<..>>>`,
//! so every copy-on-write fork of a graph carrying a range index deep-copied
//! the whole B-tree — one `Value` key and one `Vec` allocation per distinct
//! value. Held-view first write at 100k nodes measured **0.889 ms** with a
//! range index against **0.048 ms** with an equality index on the same
//! property and 0.046 ms with no index at all (P4, 2026-08-13): the entire
//! 16-19x gap was this copy.
//!
//! ## Ordered iteration under layering
//!
//! One level — every unforked graph — is the plain `BTreeMap`, and
//! [`LayeredRangeIndex::range`] hands back its own `Range` iterator, so the
//! read path is byte for byte what it was. While a fork is outstanding the
//! range is merged across levels through a `BTreeMap` keyed by reference:
//! newest level wins per key (a tombstone included), and the merge is
//! re-sorted by construction. The merge is bounded by the keys *inside the
//! requested range*, not by the index.

use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::Arc;

use petgraph::graph::NodeIndex;

use super::index_layer::MAX_LAYER_DEPTH;

/// One level's edits. `None` is a tombstone — the key is absent from here down.
type Level<K> = BTreeMap<K, Option<Vec<NodeIndex>>>;

/// A range index's ordered `value -> members` map.
///
/// Zero levels is the empty index; one level reads exactly like the plain
/// `BTreeMap` this replaced.
#[derive(Debug, Clone)]
pub struct LayeredRangeIndex<K: Ord + Clone> {
    levels: Vec<Arc<Level<K>>>,
}

impl<K: Ord + Clone> Default for LayeredRangeIndex<K> {
    fn default() -> Self {
        Self { levels: Vec::new() }
    }
}

impl<K: Ord + Clone> From<BTreeMap<K, Vec<NodeIndex>>> for LayeredRangeIndex<K> {
    fn from(map: BTreeMap<K, Vec<NodeIndex>>) -> Self {
        Self {
            levels: vec![Arc::new(
                map.into_iter().map(|(k, v)| (k, Some(v))).collect(),
            )],
        }
    }
}

impl<K: Ord + Clone> FromIterator<(K, Vec<NodeIndex>)> for LayeredRangeIndex<K> {
    fn from_iter<I: IntoIterator<Item = (K, Vec<NodeIndex>)>>(iter: I) -> Self {
        Self::from(iter.into_iter().collect::<BTreeMap<K, Vec<NodeIndex>>>())
    }
}

impl<K: Ord + Clone> LayeredRangeIndex<K> {
    /// This value's members, or `None` when the index does not hold the value.
    pub fn get(&self, key: &K) -> Option<&Vec<NodeIndex>> {
        match self.levels.as_slice() {
            [] => None,
            // The steady state, spelled out so it compiles to the single
            // B-tree probe this replaced.
            [only] => only.get(key)?.as_ref(),
            levels => {
                for level in levels.iter().rev() {
                    if let Some(entry) = level.get(key) {
                        return entry.as_ref();
                    }
                }
                None
            }
        }
    }

    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Distinct live values.
    pub fn len(&self) -> usize {
        match self.levels.as_slice() {
            [] => 0,
            [only] => only.len(),
            _ => self.iter().count(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every live `(value, members)` pair, **in value order**.
    pub fn iter(&self) -> RangeIter<'_, K> {
        self.range((Bound::Unbounded, Bound::Unbounded))
    }

    /// The live `(value, members)` pairs whose value falls in `bounds`, in
    /// value order — the read path `DirGraph::lookup_range` serves
    /// `WHERE n.prop > x` from.
    ///
    /// With one level this *is* `BTreeMap::range`. With a fork outstanding the
    /// levels are merged newest-first into a reference-keyed `BTreeMap`, which
    /// both masks shadowed keys and restores the ordering.
    pub fn range<'a>(&'a self, bounds: (Bound<&'a K>, Bound<&'a K>)) -> RangeIter<'a, K> {
        match self.levels.as_slice() {
            [] => RangeIter::Empty,
            [only] => RangeIter::Flat(only.range(bounds)),
            levels => {
                let mut merged: BTreeMap<&K, Option<&Vec<NodeIndex>>> = BTreeMap::new();
                for level in levels.iter().rev() {
                    for (key, entry) in level.range(bounds) {
                        // First writer wins: levels are walked newest-first, so
                        // the newest mention of a key decides — tombstone
                        // included.
                        merged.entry(key).or_insert(entry.as_ref());
                    }
                }
                RangeIter::Merged(
                    merged
                        .into_iter()
                        .filter_map(|(key, members)| Some((key, members?)))
                        .collect::<Vec<_>>()
                        .into_iter(),
                )
            }
        }
    }

    /// Mutable access to this value's members, materialising the bucket into
    /// this graph's own level first.
    ///
    /// The materialised bucket is a **full copy of the merged bucket**, which
    /// is what lets the journal's positional and last-occurrence reversals run
    /// unchanged against it (`index_layer`'s module doc has the argument).
    pub fn get_mut(&mut self, key: &K) -> Option<&mut Vec<NodeIndex>> {
        if self.is_flat() {
            return Arc::get_mut(&mut self.levels[0])
                .expect("is_flat proved sole ownership")
                .get_mut(key)
                .and_then(|entry| entry.as_mut());
        }
        self.get(key)?;
        Some(self.materialize(key))
    }

    /// Mutable access to this value's members, creating an empty bucket when
    /// the value is absent — `BTreeMap::entry(..).or_default()`.
    pub fn entry_or_default(&mut self, key: &K) -> &mut Vec<NodeIndex> {
        if self.is_flat() {
            return Arc::get_mut(&mut self.levels[0])
                .expect("is_flat proved sole ownership")
                .entry(key.clone())
                .or_insert_with(|| Some(Vec::new()))
                .get_or_insert_with(Vec::new);
        }
        self.materialize(key)
    }

    /// Drop this value. Leaves a tombstone when a lower level still holds it,
    /// so the base a forked reader is using stays untouched.
    pub fn remove(&mut self, key: &K) -> Option<Vec<NodeIndex>> {
        if self.is_flat() {
            // Nothing below to mask, so drop the entry outright and keep the
            // invariant that an unshared index carries no tombstones — which is
            // what lets `len` and the flat `range` arm skip the filtering.
            return Arc::get_mut(&mut self.levels[0])
                .expect("is_flat proved sole ownership")
                .remove(key)
                .flatten();
        }
        let prior = self.get(key).cloned();
        prior.as_ref()?;
        self.writable_tail().insert(key.clone(), None);
        prior
    }

    /// Run `predicate` over every live bucket's members, then drop the buckets
    /// it emptied — the delete sweep (`mutation::maintain`).
    ///
    /// Prunes emptied buckets, unlike [`super::index_layer::LayeredIndex::
    /// retain_members`], because the plain-`BTreeMap` sweep this replaces did:
    /// it followed its `values_mut` pass with `retain(|_, v| !v.is_empty())`.
    ///
    /// Flattens first — a `&mut` into every bucket cannot be served out of
    /// shared levels — so it is O(index) when a fork is outstanding and free
    /// otherwise.
    pub fn retain_members_pruning_empty<F: FnMut(&NodeIndex) -> bool + Copy>(
        &mut self,
        predicate: F,
    ) {
        self.flatten();
        let Some(level) = self.levels.first_mut() else {
            return;
        };
        let level = Arc::get_mut(level).expect("a flattened index owns its single level");
        for members in level.values_mut().flatten() {
            members.retain(predicate);
        }
        level.retain(|_, entry| entry.as_ref().is_some_and(|members| !members.is_empty()));
    }

    /// Fold the level stack back into one, for the levels this graph is the
    /// last holder of. Called at write entry (`graph::handle`) beside the
    /// other layered stores' compaction.
    pub fn try_compact(&mut self) {
        if self.levels.len() <= 1 {
            return;
        }
        let Some(first_owned) = self
            .levels
            .iter_mut()
            .position(|level| Arc::get_mut(level).is_some())
        else {
            return;
        };
        if first_owned + 1 >= self.levels.len() {
            return;
        }
        // Whether the fold reaches the bottom decides what happens to the
        // tombstones it merges (below).
        let folds_to_one = first_owned == 0;
        let mut tail = self.levels.split_off(first_owned).into_iter();
        let first = tail.next().expect("split_off yields at least one level");
        // The leading owned level is **moved** out of its `Arc`, never copied —
        // collecting into a fresh map is the O(N) compaction trap.
        let mut merged = Arc::try_unwrap(first)
            .unwrap_or_else(|_| unreachable!("position proved sole ownership"));
        for level in tail {
            match Arc::try_unwrap(level) {
                Ok(owned) => merged.extend(owned),
                Err(shared) => {
                    merged.extend(shared.iter().map(|(k, v)| (k.clone(), v.clone())));
                }
            }
        }
        if folds_to_one {
            // Nothing is left below for a tombstone to mask, so drop them and
            // keep the invariant `remove` states: a single level carries no
            // tombstones, which is what lets `len` and the flat `range` arm
            // skip the filtering. A *partial* fold must keep them — they are
            // the only thing masking the shared levels underneath.
            merged.retain(|_, entry| entry.is_some());
        }
        self.levels.push(Arc::new(merged));
    }

    // ── internals ────────────────────────────────────────────────────────

    /// Exactly one level, owned outright: the steady state, in which this type
    /// is a plain `BTreeMap` with a pointer in front of it.
    #[inline]
    fn is_flat(&mut self) -> bool {
        self.levels.len() == 1 && Arc::get_mut(&mut self.levels[0]).is_some()
    }

    /// A level this graph may write into, starting a new one when the current
    /// tail is shared with a fork.
    fn writable_tail(&mut self) -> &mut Level<K> {
        let tail_is_ours = self
            .levels
            .last_mut()
            .is_some_and(|tail| Arc::get_mut(tail).is_some());
        if !tail_is_ours {
            if self.levels.len() >= MAX_LAYER_DEPTH {
                // Bounded: reaching here needs MAX_LAYER_DEPTH consecutive
                // fork-then-write rounds with a reader held throughout.
                self.flatten();
            } else {
                self.levels.push(Arc::new(Level::default()));
            }
        }
        Arc::get_mut(self.levels.last_mut().expect("just ensured a level"))
            .expect("the tail is owned by construction")
    }

    /// Copy this value's merged bucket into the writable tail and hand back a
    /// `&mut` to it. Creates an empty bucket when the value is absent.
    fn materialize(&mut self, key: &K) -> &mut Vec<NodeIndex> {
        let tail_has = self
            .levels
            .last_mut()
            .and_then(Arc::get_mut)
            .is_some_and(|tail| tail.contains_key(key));
        // Only reach below when this graph has not already taken the bucket.
        let merged = if tail_has {
            None
        } else {
            self.get(key).cloned()
        };
        let tail = self.writable_tail();
        // `entry` rather than a blind insert: a bucket this graph already
        // materialised must not be reset to the base's version, and a tombstone
        // this graph wrote must start the new bucket empty rather than
        // resurrecting what it masked.
        tail.entry(key.clone())
            .or_insert_with(|| Some(merged.unwrap_or_default()))
            .get_or_insert_with(Vec::new)
    }

    /// Merge every level into one uniquely-owned level, dropping tombstones.
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
                Err(shared) => merged.extend(shared.iter().map(|(k, v)| (k.clone(), v.clone()))),
            }
        }
        merged.retain(|_, entry| entry.is_some());
        self.levels.push(Arc::new(merged));
    }

    #[cfg(test)]
    pub(crate) fn depth(&self) -> usize {
        self.levels.len()
    }
}

/// Ordered iterator over a [`LayeredRangeIndex`]'s live `(value, members)`
/// pairs within a range.
///
/// An enum rather than a boxed trait object so the single-level case — every
/// unforked graph — walks the B-tree directly.
pub enum RangeIter<'a, K> {
    Empty,
    Flat(std::collections::btree_map::Range<'a, K, Option<Vec<NodeIndex>>>),
    Merged(std::vec::IntoIter<(&'a K, &'a Vec<NodeIndex>)>),
}

impl<'a, K> Iterator for RangeIter<'a, K> {
    type Item = (&'a K, &'a Vec<NodeIndex>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RangeIter::Empty => None,
            // A single level carries no tombstones — `remove` drops the entry
            // outright when there is nothing below to mask, and a fold that
            // reaches the bottom drops the ones it merged. Filtered anyway:
            // the level type is the same `Option` map at every depth.
            RangeIter::Flat(it) => it.find_map(|(key, entry)| Some((key, entry.as_ref()?))),
            RangeIter::Merged(it) => it.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(i: usize) -> NodeIndex {
        NodeIndex::new(i)
    }

    fn index(pairs: &[(i64, &[usize])]) -> LayeredRangeIndex<i64> {
        pairs
            .iter()
            .map(|(key, members)| (*key, members.iter().copied().map(idx).collect()))
            .collect()
    }

    fn listed(index: &LayeredRangeIndex<i64>) -> Vec<(i64, Vec<usize>)> {
        index
            .iter()
            .map(|(k, v)| (*k, v.iter().map(|i| i.index()).collect()))
            .collect()
    }

    /// Iteration is ordered — the property the whole type exists for — and it
    /// stays ordered once a fork has layered an overlay on top.
    #[test]
    fn iteration_is_ordered_across_levels() {
        let mut writer = index(&[(10, &[1]), (30, &[3])]);
        let reader = writer.clone();

        writer.entry_or_default(&20).push(idx(2));
        writer.entry_or_default(&5).push(idx(0));
        writer.remove(&30);

        assert!(writer.depth() > 1, "the fork must have layered the writer");
        assert_eq!(
            listed(&writer),
            vec![(5, vec![0]), (10, vec![1]), (20, vec![2])],
            "the merged view must come out in value order, tombstone applied"
        );
        assert_eq!(listed(&reader), vec![(10, vec![1]), (30, vec![3])]);
    }

    /// A bounded range must return exactly the in-range live values, in order,
    /// with both bound kinds honoured on every level.
    #[test]
    fn a_bounded_range_masks_and_orders_across_levels() {
        let mut writer = index(&[(1, &[1]), (5, &[5]), (9, &[9])]);
        let _reader = writer.clone();
        writer.entry_or_default(&4).push(idx(4));
        writer.entry_or_default(&7).push(idx(7));
        writer.remove(&5);

        let scan: Vec<i64> = writer
            .range((Bound::Included(&2), Bound::Excluded(&9)))
            .map(|(value, _)| *value)
            .collect();
        assert_eq!(
            scan,
            vec![4, 7],
            "5 is tombstoned, 1 and 9 are out of range"
        );

        let unbounded: Vec<i64> = writer
            .range((Bound::Unbounded, Bound::Unbounded))
            .map(|(value, _)| *value)
            .collect();
        assert_eq!(unbounded, vec![1, 4, 7, 9]);
    }

    /// A fork copies no bucket, and each side's edits are invisible to the
    /// other.
    #[test]
    fn a_fork_shares_the_buckets_and_isolates_the_edits() {
        let mut writer = index(&[(1, &[1]), (2, &[2, 3])]);
        let reader = writer.clone();
        let shared = reader.get(&1).expect("bucket present").as_ptr();
        assert_eq!(
            writer.get(&1).expect("bucket present").as_ptr(),
            shared,
            "the fork must share the allocation, not copy it"
        );

        writer.entry_or_default(&1).push(idx(9));
        writer.entry_or_default(&3).push(idx(4));
        writer.remove(&2);

        assert_eq!(listed(&writer), vec![(1, vec![1, 9]), (3, vec![4])]);
        assert_eq!(listed(&reader), vec![(1, vec![1]), (2, vec![2, 3])]);
        assert_eq!(writer.len(), 2);
        assert_eq!(reader.len(), 2);
        assert!(!writer.contains_key(&2));
        assert!(reader.contains_key(&2));
    }

    /// The two journal contracts, in the shape `rollback::apply` reverses them:
    /// an append is undone by dropping the *last* occurrence (sparing a
    /// pre-statement one), an eviction by re-inserting at the recorded
    /// position. Both must hold against a bucket that started in a shared level.
    #[test]
    fn the_journal_reversals_hold_against_a_shared_bucket() {
        let mut writer = index(&[(7, &[3, 7, 3])]);
        let reader = writer.clone();

        writer.entry_or_default(&7).push(idx(3));
        let members = writer.get_mut(&7).expect("bucket present");
        let pos = members
            .iter()
            .rposition(|member| *member == idx(3))
            .expect("the appended member");
        members.remove(pos);
        assert_eq!(listed(&writer), vec![(7, vec![3, 7, 3])]);

        // ...and the positional restore.
        let pos = writer
            .get(&7)
            .unwrap()
            .iter()
            .position(|member| *member == idx(7))
            .expect("member present");
        writer.get_mut(&7).unwrap().retain(|m| *m != idx(7));
        let members = writer.entry_or_default(&7);
        let pos = pos.min(members.len());
        members.insert(pos, idx(7));

        assert_eq!(listed(&writer), vec![(7, vec![3, 7, 3])]);
        assert_eq!(listed(&reader), vec![(7, vec![3, 7, 3])]);
    }

    /// A removed value must read as absent without the base losing it, and a
    /// re-insert after a tombstone must start empty.
    #[test]
    fn a_tombstone_masks_the_base_and_can_be_written_over() {
        let mut writer = index(&[(1, &[1])]);
        let reader = writer.clone();

        writer.remove(&1);
        assert_eq!(writer.get(&1), None);
        assert!(writer.is_empty());

        writer.entry_or_default(&1).push(idx(5));
        assert_eq!(listed(&writer), vec![(1, vec![5])]);
        assert_eq!(listed(&reader), vec![(1, vec![1])]);
    }

    /// With nothing shared, mutations stay in the single level.
    #[test]
    fn an_unshared_index_never_grows_a_level() {
        let mut solo = index(&[(0, &[0])]);
        for i in 1..50 {
            solo.entry_or_default(&(i as i64)).push(idx(i));
        }
        assert_eq!(solo.depth(), 1);
        assert_eq!(solo.len(), 50);
    }

    /// Compaction folds once the reader drops, and declines while it lives.
    #[test]
    fn compaction_waits_for_the_reader_and_then_folds() {
        let mut writer = index(&[(1, &[1])]);
        let reader = writer.clone();
        writer.entry_or_default(&2).push(idx(2));
        assert_eq!(writer.depth(), 2);

        writer.try_compact();
        assert_eq!(writer.depth(), 2, "a live reader must block the fold");
        assert_eq!(listed(&reader), vec![(1, vec![1])]);

        drop(reader);
        writer.try_compact();
        assert_eq!(writer.depth(), 1);
        assert_eq!(listed(&writer), vec![(1, vec![1]), (2, vec![2])]);
    }

    /// A reader held across many fork-then-write rounds must not stack levels
    /// without bound, and must still answer — in order — at the cap.
    #[test]
    fn a_never_compacted_stack_stays_bounded_and_correct() {
        let mut writer = index(&[(0, &[0])]);
        let mut readers = Vec::new();
        for round in 1..=(MAX_LAYER_DEPTH * 3) {
            readers.push(writer.clone());
            writer.entry_or_default(&(round as i64)).push(idx(round));
        }

        assert!(
            writer.depth() <= MAX_LAYER_DEPTH,
            "level stack {} exceeded the cap",
            writer.depth()
        );
        let values: Vec<i64> = writer.iter().map(|(value, _)| *value).collect();
        let expected: Vec<i64> = (0..=(MAX_LAYER_DEPTH * 3) as i64).collect();
        assert_eq!(
            values, expected,
            "the capped stack must stay ordered + complete"
        );
        assert_eq!(listed(&readers[0]), vec![(0, vec![0])]);

        drop(readers);
        writer.try_compact();
        assert_eq!(writer.depth(), 1);
        assert_eq!(writer.len(), MAX_LAYER_DEPTH * 3 + 1);
    }

    /// **The tombstone-free single level.** `len`'s single-level arm is the
    /// B-tree's own `len` and the flat `range` arm skips the merge, so a fold
    /// that reaches the bottom level must not leave a tombstone in it — `len`
    /// would then disagree with `iter`.
    #[test]
    fn a_full_fold_drops_the_tombstones_it_merged() {
        let mut writer = index(&[(1, &[1]), (2, &[2])]);
        let reader = writer.clone();

        // The base still holds 1, so the removal can only tombstone it.
        writer.remove(&1);
        assert_eq!(writer.depth(), 2);

        drop(reader);
        writer.try_compact();

        assert_eq!(writer.depth(), 1);
        assert_eq!(writer.iter().count(), 1);
        assert_eq!(writer.len(), 1, "len counted a folded tombstone");
        assert!(!writer.is_empty());
        assert_eq!(listed(&writer), vec![(2, vec![2])]);
    }

    /// The complement: a **partial** fold still has shared levels under it, and
    /// its tombstones are the only thing masking their live entries.
    #[test]
    fn a_partial_fold_keeps_the_tombstones_masking_a_shared_level() {
        let mut writer = index(&[(1, &[1]), (2, &[2])]);
        let base_reader = writer.clone();

        writer.remove(&1);
        let mid_reader = writer.clone();
        writer.remove(&2);
        assert_eq!(writer.depth(), 3);

        // Only the two overlay levels are the writer's; the base is still read.
        drop(mid_reader);
        writer.try_compact();

        assert_eq!(writer.depth(), 2, "the shared base must survive the fold");
        assert_eq!(writer.get(&1), None, "the fold resurrected a removed value");
        assert_eq!(writer.get(&2), None);
        assert!(writer.is_empty());
        assert_eq!(listed(&base_reader), vec![(1, vec![1]), (2, vec![2])]);
    }

    /// The delete sweep prunes the buckets it empties — what the plain-map
    /// `values_mut` + `retain(!is_empty())` pair did — and leaves the reader's
    /// buckets alone.
    #[test]
    fn the_delete_sweep_prunes_emptied_buckets_without_disturbing_the_reader() {
        let mut writer = index(&[(1, &[1, 2]), (2, &[2])]);
        let reader = writer.clone();
        writer.entry_or_default(&3).push(idx(2));

        writer.retain_members_pruning_empty(|member| *member != idx(2));

        assert_eq!(
            listed(&writer),
            vec![(1, vec![1])],
            "buckets emptied by the sweep are dropped, as the B-tree sweep dropped them"
        );
        assert_eq!(listed(&reader), vec![(1, vec![1, 2]), (2, vec![2])]);
    }
}
