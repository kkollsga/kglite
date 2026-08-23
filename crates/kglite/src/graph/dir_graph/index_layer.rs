//! One user-created index's value → members map, held as a **stack of shared,
//! immutable levels** so a fork copies pointers instead of a million buckets.
//!
//! ## Why
//!
//! After the backend, `id_indices` and `type_indices` were layered, these two
//! families are what is left — and on an indexed graph they are nearly all of
//! it. Measured at 1M nodes: `composite_indices` **88.9 ms**, `property_indices`
//! **48.0 ms**, together **96.9%** of a 141.5 ms `DirGraph::clone`. The cost is
//! not the postings but the *keys*: one `Value` (usually a heap `String`) and
//! one `Vec<NodeIndex>` allocated per distinct value, and a `CompositeValue` is
//! a `Vec<Value>` — a `Vec` allocation plus a `String` per component, per
//! distinct tuple.
//!
//! ## The shape, and why this one
//!
//! Levels oldest-first, **the last level is the writable tail**, and a lookup
//! walks from the tail down and takes the first level that mentions the key.
//! `None` is a tombstone: the key is *absent*, without editing the base that
//! still holds it.
//!
//! This is [`crate::graph::storage::disk::type_index_layer`]'s mechanism with a
//! map payload rather than a `Vec` one, and it is the right fit here for a
//! reason specific to these two families: **the delta is bucket-granular
//! copy-on-write, and a materialised delta bucket is a full copy of the merged
//! bucket.** That is what keeps the journal's two contracts intact without
//! restating either of them:
//!
//! - `BucketRemoved` re-inserts a member at a recorded **position**. The
//!   position was measured against the merged bucket, and the delta bucket *is*
//!   the merged bucket, so `insert(pos, idx)` lands where it did before.
//! - `BucketAppended` reverses by dropping the **last** occurrence, sparing a
//!   pre-statement occurrence of the same index (`rollback::undo_bucket_append`,
//!   and the test that pins it). A delta that recorded only "what this graph
//!   appended" could not see the earlier occurrence and would drop the wrong
//!   one — or, worse, reach into the shared base to find it.
//!
//! The bucket copy is bounded by the bucket, not the index: a high-cardinality
//! index copies one or two postings per touched value; a low-cardinality one
//! copies that value's posting list once per fork.
//!
//! ## The fork is `&self`-only, on purpose
//!
//! `DirGraph::clone` reaches these fields as `&self`, and unlike `IdIndexStore`
//! there is no `RwLock` to split through — `get` hands out a borrowed
//! `&Vec<NodeIndex>`. Keeping every level behind an `Arc`, including the one
//! being written, makes `Clone` a pointer copy from `&self` alone; the writer
//! discovers the fork lazily at its next mutation through `Arc::get_mut`. It is
//! also what avoids the merge-on-share trap by construction: there is no
//! `share()` that must merge a non-empty delta before it can hand back one
//! immutable value.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use petgraph::graph::NodeIndex;

/// One level's edits. `None` is a tombstone — the key is absent from here down.
type Level<K> = HashMap<K, Option<Vec<NodeIndex>>>;

/// How many levels may stack before a mutation flattens them.
///
/// Shared with [`super::range_index_layer`], which is the same mechanism over
/// ordered levels and wants the same tuning decision, not a second one.
///
/// Matches `type_index_layer::MAX_LAYER_DEPTH` and
/// `id_index_layer::MAX_CHAIN_DEPTH` for the same measured reason: the flatten
/// is O(index), so its amortised cost is `|index| / K` per fork, and K = 8 put a
/// spike of ~5x the median into one round in eight. A stack only grows while a
/// reader is continuously held; any mutation with nothing shared folds it back
/// to one level.
pub(crate) const MAX_LAYER_DEPTH: usize = 32;

/// A user index's `value -> members` map.
///
/// Zero levels is the empty index; one level reads exactly like the plain
/// `HashMap` this replaced.
#[derive(Debug, Clone)]
pub struct LayeredIndex<K: Eq + Hash + Clone> {
    levels: Vec<Arc<Level<K>>>,
}

impl<K: Eq + Hash + Clone> Default for LayeredIndex<K> {
    fn default() -> Self {
        Self { levels: Vec::new() }
    }
}

impl<K: Eq + Hash + Clone> From<HashMap<K, Vec<NodeIndex>>> for LayeredIndex<K> {
    fn from(map: HashMap<K, Vec<NodeIndex>>) -> Self {
        Self {
            levels: vec![Arc::new(
                map.into_iter().map(|(k, v)| (k, Some(v))).collect(),
            )],
        }
    }
}

impl<K: Eq + Hash + Clone> FromIterator<(K, Vec<NodeIndex>)> for LayeredIndex<K> {
    fn from_iter<I: IntoIterator<Item = (K, Vec<NodeIndex>)>>(iter: I) -> Self {
        Self::from(iter.into_iter().collect::<HashMap<K, Vec<NodeIndex>>>())
    }
}

impl<K: Eq + Hash + Clone> LayeredIndex<K> {
    /// This value's members, or `None` when the index does not hold the value.
    ///
    /// **The read path** — `lookup_by_index` / `lookup_by_composite_index` and
    /// therefore the matcher's `try_index_lookup`. One `HashMap` probe in the
    /// steady state, one per level while a fork is outstanding, and the first
    /// level that mentions the key decides (a tombstone included).
    pub fn get(&self, key: &K) -> Option<&Vec<NodeIndex>> {
        match self.levels.as_slice() {
            [] => None,
            // The steady state, spelled out so it compiles to the single map
            // probe this replaced.
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

    /// Every live `(value, members)` pair, in no particular order.
    ///
    /// Cold paths only: eviction journalling on delete, index statistics, and
    /// the composite occupancy scan. With one level this is the plain map
    /// iterator; a layered index pays a seen-set to mask shadowed keys.
    pub fn iter(&self) -> IndexIter<'_, K> {
        match self.levels.as_slice() {
            [] => IndexIter::Empty,
            // The steady state borrows the map's own iterator — no allocation,
            // which matters because the composite `SET` path scans every bucket
            // of its index on every write.
            [only] => IndexIter::Flat(only.iter()),
            levels => {
                let mut seen: HashSet<&K> = HashSet::new();
                let mut out: Vec<(&K, &Vec<NodeIndex>)> = Vec::new();
                for level in levels.iter().rev() {
                    for (key, entry) in level.iter() {
                        if !seen.insert(key) {
                            continue;
                        }
                        if let Some(members) = entry {
                            out.push((key, members));
                        }
                    }
                }
                IndexIter::Merged(out.into_iter())
            }
        }
    }

    /// Mutable access to this value's members, materialising the bucket into
    /// this graph's own level first.
    ///
    /// The materialised bucket is a **full copy of the merged bucket**, which is
    /// what lets the journal's positional and last-occurrence reversals run
    /// unchanged against it (module doc).
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
    /// the value is absent — `HashMap::entry(..).or_default()`.
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
            // what lets `len` and `iter` take their single-level fast paths.
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

    /// Drop every value.
    ///
    /// Tombstones rather than truncates when a fork is outstanding, so the
    /// levels a reader is holding keep their content.
    pub fn clear(&mut self) {
        if self.is_flat() {
            Arc::get_mut(&mut self.levels[0])
                .expect("is_flat proved sole ownership")
                .clear();
            return;
        }
        let keys: Vec<K> = self.iter().map(|(key, _)| key.clone()).collect();
        let tail = self.writable_tail();
        for key in keys {
            tail.insert(key, None);
        }
    }

    /// Run `predicate` over every live bucket's members, in place.
    ///
    /// The delete sweep. Flattens first — a `&mut` into every bucket cannot be
    /// served out of shared levels — so it is O(index) when a fork is
    /// outstanding and free otherwise. Empty buckets are left in place, exactly
    /// as the plain-`HashMap` sweep this replaced did.
    pub fn retain_members<F: FnMut(&NodeIndex) -> bool + Copy>(&mut self, predicate: F) {
        self.flatten();
        let Some(level) = self.levels.first_mut() else {
            return;
        };
        let level = Arc::get_mut(level).expect("a flattened index owns its single level");
        for members in level.values_mut().flatten() {
            members.retain(predicate);
        }
    }

    /// Fold the level stack back into one, for the levels this graph is the
    /// last holder of.
    ///
    /// Called at write entry alongside the backend's, `id_indices`' and
    /// `type_indices`' compaction. A still-shared leading level is left alone
    /// and only the owned suffix merges, which bounds the depth without copying
    /// the base.
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
        // The leading owned level is **moved** out of its `Arc`, never copied:
        // spelling this as "collect everything into a fresh map" is the O(N)
        // compaction trap, which turns every write after a dropped reader
        // into a full index copy.
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
            // tombstones, which is what lets `len` and `iter` take their
            // single-level fast paths. A *partial* fold must keep them — they
            // are the only thing masking the shared levels underneath.
            merged.retain(|_, entry| entry.is_some());
        }
        self.levels.push(Arc::new(merged));
    }

    /// The whole index as one plain map — the merged, tombstone-free content.
    ///
    /// Cold path: `create_index` replaces the whole structure rather than
    /// merging into it, so this exists for callers that genuinely need an owned
    /// `HashMap` (statistics, tests).
    pub fn to_map(&self) -> HashMap<K, Vec<NodeIndex>> {
        self.iter()
            .map(|(key, members)| (key.clone(), members.clone()))
            .collect()
    }

    // ── internals ────────────────────────────────────────────────────────

    /// Exactly one level, owned outright: the steady state, in which this type
    /// is a plain `HashMap` with a pointer in front of it.
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
    /// A lone level is only made unique: it carries none to drop.
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

/// Iterator over a [`LayeredIndex`]'s live `(value, members)` pairs.
///
/// An enum rather than a boxed trait object so the single-level case — every
/// unforked graph — iterates the underlying map directly.
pub enum IndexIter<'a, K> {
    Empty,
    Flat(std::collections::hash_map::Iter<'a, K, Option<Vec<NodeIndex>>>),
    Merged(std::vec::IntoIter<(&'a K, &'a Vec<NodeIndex>)>),
}

impl<'a, K> Iterator for IndexIter<'a, K> {
    type Item = (&'a K, &'a Vec<NodeIndex>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IndexIter::Empty => None,
            // A single level carries no tombstones — `remove` drops the entry
            // outright when there is nothing below to mask, and a fold that
            // reaches the bottom drops the ones it merged. Filtered anyway:
            // the level type is the same `Option` map at every depth.
            IndexIter::Flat(it) => it.find_map(|(key, entry)| Some((key, entry.as_ref()?))),
            IndexIter::Merged(it) => it.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(i: usize) -> NodeIndex {
        NodeIndex::new(i)
    }

    fn index(pairs: &[(&str, &[usize])]) -> LayeredIndex<String> {
        pairs
            .iter()
            .map(|(key, members)| {
                (
                    (*key).to_string(),
                    members.iter().copied().map(idx).collect(),
                )
            })
            .collect()
    }

    fn sorted(index: &LayeredIndex<String>) -> Vec<(String, Vec<usize>)> {
        let mut out: Vec<(String, Vec<usize>)> = index
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|i| i.index()).collect()))
            .collect();
        out.sort();
        out
    }

    // ── the two journal contracts, written before the mechanism ──────────

    /// **Gate 1a.** Reversing an append must drop the occurrence the append
    /// pushed and spare a pre-statement one — `rollback::undo_bucket_append`'s
    /// contract, now that the pre-statement occurrence can live in a level a
    /// forked reader is holding.
    ///
    /// A delta that recorded only "what this graph appended" would either drop
    /// the base's occurrence (wrong member gone) or have to reach into the base
    /// to find it (reader corruption). Materialising the whole merged bucket is
    /// what makes the existing reversal correct unchanged.
    #[test]
    fn reversing_an_append_spares_a_pre_statement_occurrence_in_a_shared_level() {
        // The bucket already holds node 3 twice over — the shape
        // `rollback_tests::undoing_an_append_leaves_a_pre_existing_occurrence`
        // pins for the unlayered map.
        let mut writer = index(&[("v", &[3, 7, 3])]);
        let reader = writer.clone();

        // The statement appends node 3 again...
        writer.entry_or_default(&"v".to_string()).push(idx(3));
        assert_eq!(sorted(&writer), vec![("v".into(), vec![3, 7, 3, 3])]);

        // ...and is then reversed exactly as `rollback::undo_bucket_append`
        // does it: drop the *last* occurrence.
        let members = writer.get_mut(&"v".to_string()).expect("bucket present");
        let pos = members
            .iter()
            .rposition(|member| *member == idx(3))
            .expect("the appended member");
        members.remove(pos);

        assert_eq!(
            sorted(&writer),
            vec![("v".into(), vec![3, 7, 3])],
            "both pre-statement occurrences must survive, in order"
        );
        assert_eq!(
            sorted(&reader),
            vec![("v".into(), vec![3, 7, 3])],
            "the reader's bucket must be untouched by either edit"
        );
    }

    /// **Gate 1b.** Reversing an eviction must put the member back at the
    /// *position* it vacated, in a bucket that started life in a shared level.
    ///
    /// Bucket order is the row order an un-`ORDER BY`'d indexed `MATCH`
    /// returns, so a reversal that appends instead of inserting is a silent
    /// reordering.
    #[test]
    fn reversing_an_eviction_restores_the_recorded_position_in_a_shared_bucket() {
        let mut writer = index(&[("v", &[10, 11, 12, 13])]);
        let reader = writer.clone();

        // Evict the *middle* member, recording its position first — the
        // `note_property_eviction` / `undo` pair.
        let key = "v".to_string();
        let pos = writer
            .get(&key)
            .unwrap()
            .iter()
            .position(|member| *member == idx(12))
            .expect("member present");
        assert_eq!(pos, 2);
        writer
            .get_mut(&key)
            .unwrap()
            .retain(|member| *member != idx(12));
        assert_eq!(sorted(&writer), vec![("v".into(), vec![10, 11, 13])]);

        // Roll back: `BucketRemoved`'s arm, verbatim.
        let members = writer.entry_or_default(&key);
        let pos = pos.min(members.len());
        members.insert(pos, idx(12));

        assert_eq!(
            sorted(&writer),
            vec![("v".into(), vec![10, 11, 12, 13])],
            "the member must return to its own position, not to the end"
        );
        assert_eq!(sorted(&reader), vec![("v".into(), vec![10, 11, 12, 13])]);
    }

    // ── the layering itself ──────────────────────────────────────────────

    /// A fork copies no bucket, and each side's edits are invisible to the
    /// other.
    #[test]
    fn a_fork_shares_the_buckets_and_isolates_the_edits() {
        let mut writer = index(&[("a", &[1]), ("b", &[2, 3])]);
        let reader = writer.clone();

        writer.entry_or_default(&"a".to_string()).push(idx(9));
        writer.entry_or_default(&"c".to_string()).push(idx(4));
        writer.remove(&"b".to_string());

        assert_eq!(
            sorted(&writer),
            vec![("a".into(), vec![1, 9]), ("c".into(), vec![4])]
        );
        assert_eq!(
            sorted(&reader),
            vec![("a".into(), vec![1]), ("b".into(), vec![2, 3])],
            "the reader must see its own snapshot, tombstone and all"
        );
        assert_eq!(writer.len(), 2);
        assert_eq!(reader.len(), 2);
        assert!(!writer.contains_key(&"b".to_string()));
        assert!(reader.contains_key(&"b".to_string()));
    }

    /// A removed value must read as absent without the base losing it, and a
    /// re-insert after a tombstone must resurrect cleanly.
    #[test]
    fn a_tombstone_masks_the_base_and_can_be_written_over() {
        let mut writer = index(&[("a", &[1])]);
        let reader = writer.clone();

        writer.remove(&"a".to_string());
        assert_eq!(writer.get(&"a".to_string()), None);
        assert_eq!(writer.len(), 0);
        assert!(writer.is_empty());

        writer.entry_or_default(&"a".to_string()).push(idx(5));
        assert_eq!(
            sorted(&writer),
            vec![("a".into(), vec![5])],
            "a write over a tombstone must not resurrect the base's members"
        );
        assert_eq!(sorted(&reader), vec![("a".into(), vec![1])]);
    }

    /// Removing a value that was never there must not leave a tombstone that a
    /// later fold would carry as a real entry.
    #[test]
    fn removing_an_absent_value_is_a_no_op() {
        let mut writer = index(&[("a", &[1])]);
        assert_eq!(writer.remove(&"zz".to_string()), None);
        assert_eq!(writer.len(), 1);
        assert_eq!(sorted(&writer), vec![("a".into(), vec![1])]);
    }

    /// With nothing shared, mutations stay in the single level — no growth, and
    /// therefore no read-path cost in the steady state.
    #[test]
    fn an_unshared_index_never_grows_a_level() {
        let mut solo = index(&[("a", &[1])]);
        for i in 0..50 {
            solo.entry_or_default(&format!("k{i}")).push(idx(i));
        }
        assert_eq!(solo.depth(), 1);
        assert_eq!(solo.len(), 51);
    }

    /// Compaction folds once the reader drops, and declines while it lives.
    #[test]
    fn compaction_waits_for_the_reader_and_then_folds() {
        let mut writer = index(&[("a", &[1])]);
        let reader = writer.clone();
        writer.entry_or_default(&"b".to_string()).push(idx(2));
        assert_eq!(writer.depth(), 2);

        writer.try_compact();
        assert_eq!(writer.depth(), 2, "a live reader must block the fold");
        assert_eq!(sorted(&reader), vec![("a".into(), vec![1])]);

        drop(reader);
        writer.try_compact();
        assert_eq!(writer.depth(), 1);
        assert_eq!(
            sorted(&writer),
            vec![("a".into(), vec![1]), ("b".into(), vec![2])]
        );
    }

    /// A reader held across many fork-then-write rounds must not stack levels
    /// without bound, and must still answer correctly at the cap.
    #[test]
    fn a_never_compacted_stack_stays_bounded_and_correct() {
        let mut writer = index(&[("a", &[0])]);
        let mut readers = Vec::new();
        for round in 1..=(MAX_LAYER_DEPTH * 3) {
            readers.push(writer.clone());
            writer
                .entry_or_default(&format!("k{round}"))
                .push(idx(round));
        }

        assert!(
            writer.depth() <= MAX_LAYER_DEPTH,
            "level stack {} exceeded the cap",
            writer.depth()
        );
        assert_eq!(writer.len(), MAX_LAYER_DEPTH * 3 + 1);
        for round in 1..=(MAX_LAYER_DEPTH * 3) {
            assert_eq!(
                writer.get(&format!("k{round}")).map(Vec::as_slice),
                Some([idx(round)].as_slice()),
                "value k{round} lost in the stack"
            );
        }
        assert_eq!(sorted(&readers[0]), vec![("a".into(), vec![0])]);

        drop(readers);
        writer.try_compact();
        assert_eq!(writer.depth(), 1);
        assert_eq!(writer.len(), MAX_LAYER_DEPTH * 3 + 1);
    }

    /// **The tombstone-free single level.** `remove` keeps that invariant when
    /// it is the only level, and a fold that reaches the bottom must keep it
    /// too — `len`'s single-level arm is the map's own `len`, so a tombstone
    /// folded into that level makes `len` disagree with `iter`, and the index
    /// statistics (`unique_values`, the `avg_entries` divisor, the worst-bucket
    /// rebuild heuristic) all read `len`.
    #[test]
    fn a_full_fold_drops_the_tombstones_it_merged() {
        let mut writer = index(&[("a", &[1]), ("b", &[2])]);
        let reader = writer.clone();

        // The base still holds "a", so the removal can only tombstone it.
        writer.remove(&"a".to_string());
        assert_eq!(writer.depth(), 2);

        drop(reader);
        writer.try_compact();

        assert_eq!(writer.depth(), 1);
        assert_eq!(writer.iter().count(), 1);
        assert_eq!(writer.len(), 1, "len counted a folded tombstone");
        assert!(!writer.is_empty());
        assert_eq!(sorted(&writer), vec![("b".into(), vec![2])]);
    }

    /// The complement: a **partial** fold still has shared levels under it, and
    /// its tombstones are the only thing masking their live entries. Dropping
    /// them there resurrects removed values.
    #[test]
    fn a_partial_fold_keeps_the_tombstones_masking_a_shared_level() {
        let mut writer = index(&[("a", &[1]), ("b", &[2])]);
        let base_reader = writer.clone();

        writer.remove(&"a".to_string());
        let mid_reader = writer.clone();
        writer.remove(&"b".to_string());
        assert_eq!(writer.depth(), 3);

        // Only the two overlay levels are the writer's; the base is still read.
        drop(mid_reader);
        writer.try_compact();

        assert_eq!(writer.depth(), 2, "the shared base must survive the fold");
        assert_eq!(
            writer.get(&"a".to_string()),
            None,
            "the fold resurrected a removed value"
        );
        assert_eq!(writer.get(&"b".to_string()), None);
        assert!(writer.is_empty());
        assert_eq!(
            sorted(&base_reader),
            vec![("a".into(), vec![1]), ("b".into(), vec![2])]
        );
    }

    /// The delete sweep flattens, and must leave the reader's buckets alone.
    #[test]
    fn the_delete_sweep_flattens_without_disturbing_the_reader() {
        let mut writer = index(&[("a", &[1, 2]), ("b", &[2])]);
        let reader = writer.clone();
        writer.entry_or_default(&"c".to_string()).push(idx(2));

        writer.retain_members(|member| *member != idx(2));

        assert_eq!(
            sorted(&writer),
            vec![
                ("a".into(), vec![1]),
                ("b".into(), Vec::new()),
                ("c".into(), Vec::new())
            ],
            "emptied buckets stay, exactly as the plain-map sweep left them"
        );
        assert_eq!(
            sorted(&reader),
            vec![("a".into(), vec![1, 2]), ("b".into(), vec![2])]
        );
    }
}
