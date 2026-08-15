//! One type's id index, either owned outright or **layered over a base a
//! forked graph is still reading**.
//!
//! ## Why
//!
//! `id_indices` is a `HashMap` with one entry per node of every materialised
//! type, so `DirGraph::clone` deep-copies it on every fork. Measured 2026-08-10
//! at 1M nodes: **3.7 ms**, which after D2 Phase 2 removed the backend row is
//! **90% of everything left** in a plain graph's fork. This type is what makes
//! that O(changes).
//!
//! ## The split has to happen inside `Clone`
//!
//! Every fork reaches this field as `&self` — derived `DirGraph::clone` →
//! `IdIndexStore::clone`. By the time write entry holds a `&mut DirGraph` the
//! deep copy has already happened, so there is no `&mut` moment between "a
//! reader exists" and "the map was copied". [`TypeEntry::share`] is therefore
//! called from `Clone` through the store's existing `RwLock`: it converts the
//! *parent* in place from `Owned` to `Layered` and hands the child the same
//! `Arc`. Both graphs then read identical content; only the representation
//! changed.
//!
//! ## Tombstones instead of a second key set
//!
//! [`TypeIdIndex`] is an enum over two key types (`Integer(HashMap<u32, _>)`
//! and `General(HashMap<Value, _>)`), so a `removed` set would have to be
//! written twice and kept in step with the demotion rule in
//! `TypeIdIndex::insert`. Instead a deletion is recorded *in the delta* as
//! `NodeIndex::end()` — petgraph's own sentinel, which no live node can hold.
//! One map, one key discipline, and the demotion logic is reused rather than
//! mirrored.
//!
//! That two-variant split turns out **not** to be the obstacle it looked like:
//! `TypeIdIndex::get` takes `&Value` and does its own `UniqueId`/`Int64`/
//! `Float64` coercion per map, so a `Layered` entry can chain a `General` delta
//! over an `Integer` base and every lookup still resolves under whichever
//! spelling the value was stored as.
//!
//! ## Rollback needs nothing here
//!
//! Unlike the user index families, `id_indices` has **no undo entries**: a
//! failed statement drops the touched type's index and the next read rebuilds
//! it from the graph (`dir_graph/rollback.rs`, and the `id_indices` note on
//! `swap_data_scale`). Dropping a `Layered` entry releases the shared base and
//! rebuilds from the restored graph, which is the same self-healing path as
//! before — so this layer adds no rollback surface at all.

use std::sync::Arc;

use petgraph::graph::NodeIndex;

use crate::datatypes::Value;
use crate::graph::schema::TypeIdIndex;

/// A deleted id, recorded in the delta so it masks the shared base.
///
/// `NodeIndex::end()` is petgraph's own "no node" sentinel (`usize::MAX`); a
/// live graph can never hand it out, so it cannot collide with a real mapping.
#[inline]
fn tombstone() -> NodeIndex {
    NodeIndex::end()
}

/// One node type's id index.
#[derive(Debug, Clone)]
pub enum TypeEntry {
    /// Uniquely owned by this graph — the steady state, and byte-for-byte what
    /// this field held before D2 Phase 3.
    Owned(TypeIdIndex),
    /// A base shared with at least one other graph, plus this graph's delta.
    /// Reads chain delta → base; writes only ever touch `delta`.
    ///
    /// `base` is a `TypeEntry`, not a `TypeIdIndex`, and that recursion is the
    /// point: it lets [`share`](TypeEntry::share) wrap *whatever this entry
    /// already is* in one `Arc` — O(1) — instead of merging a non-empty delta
    /// into its base first. Merging is O(N_type), and it fires on every fork
    /// that follows a write, which is exactly the founding defect's shape (a
    /// read-then-write loop that re-takes a view each iteration). Measured at
    /// 1M before the recursion: the held-view first write stayed at 4.1 ms
    /// because every round paid that merge.
    ///
    /// `depth` bounds the chain — see [`MAX_CHAIN_DEPTH`].
    Layered {
        base: Arc<TypeEntry>,
        delta: TypeIdIndex,
        depth: u16,
    },
}

/// How many `Layered` levels may stack before [`TypeEntry::share`] flattens.
///
/// Every level is one extra `HashMap` probe on a miss and one retained delta,
/// so an unbounded chain would trade the fork cost for a read cost and a leak.
/// A chain only grows while a reader is continuously held — any write with no
/// live reader compacts the whole thing — so in practice it stays at 1.
///
/// **Tuned from measurement, 2026-08-10.** The flatten is O(N_type), so its
/// amortised cost is `|index| / MAX_CHAIN_DEPTH` per fork; at 1M that is ~3.7 ms
/// spread over K rounds. The value started at 8, which put a ~5 ms spike in
/// roughly one round in eight and dominated the *mean* of the held-view cell
/// (min 126 µs, median 139 µs, **mean 608 µs**, max 5 021 µs — two spikes in
/// twenty rounds). At 32 the harness cell reads min 114 / median 118 /
/// **mean 129** / max 289 µs: the tail is gone and the mean is the real
/// per-round cost.
///
/// 128 measured identically in-window (mean 129 µs) and would amortise the
/// flatten 4x further, but every extra level is a probe on a read miss and a
/// retained delta, and 32 already moves the worst case from ~5x the median to
/// ~2x. Raising it further is tuning against one benchmark's hold window rather
/// than against a mechanism.
pub(crate) const MAX_CHAIN_DEPTH: u16 = 32;

/// The measured value, pinned as a value rather than as a symbol.
///
/// Every test below is written in terms of `MAX_CHAIN_DEPTH`, which makes them
/// *any*-value tests: set it to 1 or to 4096 and they all still pass, because
/// the expectations move with it. The number itself is the finding — the D2
/// Phase 3 profile above rejects 8 (a ~5x-median spike in one round in eight)
/// and rejects raising it past 32 (128 measured identically and only deepens
/// the read-miss probe) — so the number gets a check of its own. Changing it is
/// legitimate; changing it *without a new measurement* is what this stops.
const _: () = assert!(
    MAX_CHAIN_DEPTH == 32,
    "MAX_CHAIN_DEPTH is a measured value (D2 Phase 3 residual profile §B), not a \
     free parameter — re-measure the held-view cell's mean before moving it, and \
     update the doc comment with the new numbers"
);

/// The three layered-index caps are **one** tuning decision, taken once and
/// applied to three mechanisms with the same amortisation curve. Each of the
/// other two documents itself as "matches `id_index_layer::MAX_CHAIN_DEPTH`",
/// which is a claim no compiler was checking: any one of them could be edited
/// alone and the other two would go on advertising a value they no longer
/// shared. This is that claim, enforced.
const _: () = assert!(
    MAX_CHAIN_DEPTH as usize == super::type_index_layer::MAX_LAYER_DEPTH
        && MAX_CHAIN_DEPTH as usize == crate::graph::dir_graph::index_layer::MAX_LAYER_DEPTH,
    "the layered-index depth caps have drifted apart: id_index_layer::MAX_CHAIN_DEPTH, \
     type_index_layer::MAX_LAYER_DEPTH and dir_graph::index_layer::MAX_LAYER_DEPTH \
     (shared with range_index_layer) each document themselves as matching the others, \
     so they move together or the doc comments are lying"
);

impl Default for TypeEntry {
    fn default() -> Self {
        TypeEntry::Owned(TypeIdIndex::default())
    }
}

impl From<TypeIdIndex> for TypeEntry {
    #[inline]
    fn from(index: TypeIdIndex) -> Self {
        TypeEntry::Owned(index)
    }
}

impl TypeEntry {
    #[inline]
    pub fn get(&self, id: &Value) -> Option<NodeIndex> {
        match self {
            TypeEntry::Owned(index) => index.get(id),
            TypeEntry::Layered { base, delta, .. } => match delta.get(id) {
                Some(idx) if idx == tombstone() => None,
                Some(idx) => Some(idx),
                None => base.get(id),
            },
        }
    }

    #[inline]
    pub fn insert(&mut self, id: Value, idx: NodeIndex) {
        match self {
            TypeEntry::Owned(index) => index.insert(id, idx),
            TypeEntry::Layered { delta, .. } => delta.insert(id, idx),
        }
    }

    /// Drop `id`, but only when it currently resolves to `idx` — same contract
    /// as [`TypeIdIndex::remove_matching`], so a re-pointed id is left intact.
    pub fn remove_matching(&mut self, id: &Value, idx: NodeIndex) -> bool {
        match self {
            TypeEntry::Owned(index) => index.remove_matching(id, idx),
            TypeEntry::Layered { base, delta, .. } => {
                if delta.get(id) == Some(tombstone()) {
                    return false;
                }
                let resolved = delta.get(id).or_else(|| base.get(id));
                if resolved != Some(idx) {
                    return false;
                }
                // Tombstone rather than remove: the mapping may live in the
                // shared base, which this graph must not touch.
                delta.insert(id.clone(), tombstone());
                true
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TypeEntry::Owned(index) => index.len(),
            TypeEntry::Layered { base, delta, .. } => {
                let mut live = base.len() as i64;
                for (id, idx) in delta.iter() {
                    let in_base = base.get(&id).is_some();
                    if idx == tombstone() {
                        if in_base {
                            live -= 1;
                        }
                    } else if !in_base {
                        live += 1;
                    }
                }
                live.max(0) as usize
            }
        }
    }

    /// The merged view as one owned index. Cold path — `save`, N-Triples
    /// export, and the `add_nodes` conflict-check fast path.
    pub fn materialize(&self) -> TypeIdIndex {
        match self {
            TypeEntry::Owned(index) => index.clone(),
            TypeEntry::Layered { base, delta, .. } => {
                let mut merged = base.materialize();
                for (id, idx) in delta.iter() {
                    if idx == tombstone() {
                        if let Some(current) = merged.get(&id) {
                            merged.remove_matching(&id, current);
                        }
                    } else {
                        merged.insert(id, idx);
                    }
                }
                merged
            }
        }
    }

    /// Convert this entry into a shared base and return a handle to it, so a
    /// clone can start from the same allocation.
    ///
    /// **Called from `IdIndexStore::clone` through the store's `RwLock`** — the
    /// only place the split can happen (see the module doc). `Owned` is wrapped
    /// in place; an already-`Layered` entry with an empty delta re-shares its
    /// existing base for free, and one with writes merges them into a fresh
    /// base first. That merge is the only O(N_type) path here, and it needs a
    /// *second* fork after writes to reach — the common
    /// fork-write-drop-fork sequence compacts in between.
    pub fn share(&mut self) -> Arc<TypeEntry> {
        if self.depth() >= MAX_CHAIN_DEPTH {
            // Flatten first: this is the only O(N_type) path, and reaching it
            // needs MAX_CHAIN_DEPTH consecutive fork-after-write rounds with a
            // reader held throughout.
            *self = TypeEntry::Owned(self.materialize());
        }
        let taken = std::mem::replace(self, TypeEntry::Owned(TypeIdIndex::default()));
        let depth = taken.depth().saturating_add(1);
        let base = Arc::new(taken);
        *self = TypeEntry::Layered {
            base: Arc::clone(&base),
            delta: TypeIdIndex::default(),
            depth,
        };
        base
    }

    #[inline]
    fn depth(&self) -> u16 {
        match self {
            TypeEntry::Owned(_) => 0,
            TypeEntry::Layered { depth, .. } => *depth,
        }
    }

    /// A child's view over an already-shared base.
    #[inline]
    pub fn layered_over(base: Arc<TypeEntry>) -> Self {
        let depth = base.depth().saturating_add(1);
        TypeEntry::Layered {
            base,
            delta: TypeIdIndex::default(),
            depth,
        }
    }

    /// Fold the delta back into the base when this graph is its last holder,
    /// returning to the flat `Owned` representation.
    ///
    /// The fold is a plain map overwrite — unlike the topology overlay there is
    /// no slot to predict, because the delta already recorded the real
    /// `NodeIndex` values the graph handed out. What it must not do is mutate a
    /// base another graph is reading, which is exactly what `Arc::get_mut`
    /// gates.
    pub fn try_compact(&mut self) {
        let TypeEntry::Layered { base, .. } = self else {
            return;
        };
        if Arc::get_mut(base).is_none() {
            return;
        }
        // Sole owner: **take the base out of the `Arc` and apply the delta into
        // it in place**, recursing so a whole chain collapses at once. The
        // obvious spelling — `*self = Owned(self.materialize())` — is O(N_type),
        // because `materialize` clones the base before merging; that turned
        // compaction into a full index copy on every write after a dropped view
        // and showed up as a +289% regression on the harness's `dropped_view`
        // control at 1M. Compaction must be O(delta) or it is just the deep
        // clone this phase removed, moved one write later.
        let TypeEntry::Layered { base, delta, .. } =
            std::mem::replace(self, TypeEntry::Owned(TypeIdIndex::default()))
        else {
            unreachable!("just matched Layered")
        };
        let mut inner =
            Arc::try_unwrap(base).unwrap_or_else(|_| unreachable!("get_mut proved uniqueness"));
        inner.try_compact();
        let mut owned = match inner {
            TypeEntry::Owned(index) => index,
            // A level below is still shared, so it cannot be unwrapped; fall
            // back to a merged copy rather than mutating what it shares.
            layered => layered.materialize(),
        };
        for (id, idx) in delta.iter() {
            if idx == tombstone() {
                if let Some(current) = owned.get(&id) {
                    owned.remove_matching(&id, current);
                }
            } else {
                owned.insert(id, idx);
            }
        }
        *self = TypeEntry::Owned(owned);
    }

    #[cfg(test)]
    pub fn is_layered(&self) -> bool {
        matches!(self, TypeEntry::Layered { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(pairs: &[(u32, usize)]) -> TypeEntry {
        let mut index = TypeIdIndex::default();
        for (id, node) in pairs {
            index.insert(Value::UniqueId(*id), NodeIndex::new(*node));
        }
        TypeEntry::Owned(index)
    }

    /// A layered entry must answer exactly as the owned index it was split
    /// from, for every operation — that equivalence is the whole contract, and
    /// a divergence here is a wrong `MATCH (n {id: …})` answer.
    #[test]
    fn layering_preserves_every_answer_the_owned_index_gives() {
        let mut parent = owned(&[(1, 10), (2, 20), (3, 30)]);
        let base = parent.share();
        let child = TypeEntry::layered_over(base);

        assert!(parent.is_layered() && child.is_layered());
        for id in 1..=3u32 {
            let v = Value::UniqueId(id);
            assert_eq!(child.get(&v), Some(NodeIndex::new(id as usize * 10)));
            assert_eq!(parent.get(&v), child.get(&v));
        }
        assert_eq!(child.get(&Value::UniqueId(9)), None);
        assert_eq!(child.len(), 3);

        // Int64 must still coerce onto an Integer base through the delta layer.
        assert_eq!(child.get(&Value::Int64(2)), Some(NodeIndex::new(20)));
    }

    /// Writes land in the delta and are invisible to the other holder — the
    /// property the fork exists for.
    #[test]
    fn a_write_through_one_holder_is_invisible_to_the_other() {
        let mut parent = owned(&[(1, 10), (2, 20)]);
        let mut child = TypeEntry::layered_over(parent.share());

        child.insert(Value::UniqueId(3), NodeIndex::new(30));
        assert_eq!(child.get(&Value::UniqueId(3)), Some(NodeIndex::new(30)));
        assert_eq!(
            parent.get(&Value::UniqueId(3)),
            None,
            "the other holder must not see a delta write"
        );
        assert_eq!(child.len(), 3);
        assert_eq!(parent.len(), 2);

        // An overwrite of a base entry, likewise.
        child.insert(Value::UniqueId(1), NodeIndex::new(99));
        assert_eq!(child.get(&Value::UniqueId(1)), Some(NodeIndex::new(99)));
        assert_eq!(parent.get(&Value::UniqueId(1)), Some(NodeIndex::new(10)));
        assert_eq!(child.len(), 3, "an overwrite is not a new entry");
    }

    /// Deletion masks the shared base without touching it, and keeps
    /// `remove_matching`'s re-pointed-id guard.
    #[test]
    fn deletion_tombstones_rather_than_editing_the_shared_base() {
        let mut parent = owned(&[(1, 10), (2, 20)]);
        let mut child = TypeEntry::layered_over(parent.share());

        assert!(child.remove_matching(&Value::UniqueId(1), NodeIndex::new(10)));
        assert_eq!(child.get(&Value::UniqueId(1)), None);
        assert_eq!(child.len(), 1);
        assert_eq!(
            parent.get(&Value::UniqueId(1)),
            Some(NodeIndex::new(10)),
            "the other holder keeps the entry"
        );

        // Already gone: a second removal is a no-op, not a double-decrement.
        assert!(!child.remove_matching(&Value::UniqueId(1), NodeIndex::new(10)));
        assert_eq!(child.len(), 1);

        // Re-pointed id: the live mapping must survive.
        assert!(!child.remove_matching(&Value::UniqueId(2), NodeIndex::new(77)));
        assert_eq!(child.get(&Value::UniqueId(2)), Some(NodeIndex::new(20)));
    }

    /// Compaction returns the flat representation and preserves every answer,
    /// including the tombstoned ones.
    #[test]
    fn compaction_folds_the_delta_and_preserves_every_answer() {
        let mut child = {
            let mut parent = owned(&[(1, 10), (2, 20), (3, 30)]);
            TypeEntry::layered_over(parent.share())
            // `parent` drops here, so the child is the last holder.
        };
        child.insert(Value::UniqueId(4), NodeIndex::new(40));
        child.remove_matching(&Value::UniqueId(2), NodeIndex::new(20));

        let before: Vec<_> = (1..=4u32)
            .map(|id| child.get(&Value::UniqueId(id)))
            .collect();
        let len_before = child.len();

        child.try_compact();

        assert!(
            !child.is_layered(),
            "the last holder must collapse to Owned"
        );
        let after: Vec<_> = (1..=4u32)
            .map(|id| child.get(&Value::UniqueId(id)))
            .collect();
        assert_eq!(after, before, "compaction must not change a single answer");
        assert_eq!(child.len(), len_before);
        assert_eq!(
            after,
            vec![
                Some(NodeIndex::new(10)),
                None,
                Some(NodeIndex::new(30)),
                Some(NodeIndex::new(40))
            ]
        );
    }

    /// While another holder is alive, compaction must decline — folding would
    /// mutate a base that holder is reading.
    #[test]
    fn compaction_declines_while_another_holder_is_alive() {
        let mut parent = owned(&[(1, 10)]);
        let mut child = TypeEntry::layered_over(parent.share());
        child.insert(Value::UniqueId(2), NodeIndex::new(20));

        child.try_compact();
        assert!(child.is_layered(), "a live co-holder must block the fold");
        assert_eq!(parent.get(&Value::UniqueId(2)), None);

        drop(parent);
        child.try_compact();
        assert!(!child.is_layered());
        assert_eq!(child.get(&Value::UniqueId(2)), Some(NodeIndex::new(20)));
    }

    /// A `General`-keyed delta over an `Integer` base: the case the two-variant
    /// enum makes possible, and the reason the delta is a `TypeIdIndex` rather
    /// than a hand-rolled map.
    #[test]
    fn a_general_delta_over_an_integer_base_resolves_both_ways() {
        let mut parent = owned(&[(1, 10), (2, 20)]);
        let mut child = TypeEntry::layered_over(parent.share());

        // A non-UniqueId key demotes the delta to General; the base stays
        // Integer, and lookups must still resolve through both.
        child.insert(Value::String("abc".to_string()), NodeIndex::new(30));

        assert_eq!(
            child.get(&Value::String("abc".to_string())),
            Some(NodeIndex::new(30))
        );
        assert_eq!(child.get(&Value::UniqueId(1)), Some(NodeIndex::new(10)));
        assert_eq!(child.get(&Value::Int64(2)), Some(NodeIndex::new(20)));
        assert_eq!(child.len(), 3);

        let merged = child.materialize();
        assert_eq!(merged.get(&Value::UniqueId(1)), Some(NodeIndex::new(10)));
        assert_eq!(
            merged.get(&Value::String("abc".to_string())),
            Some(NodeIndex::new(30))
        );
    }

    /// A reader held across many fork-after-write rounds must not build an
    /// unbounded chain — and must still answer correctly at the cap.
    ///
    /// This is the pathological shape the recursion trades against: a loop that
    /// re-takes a view every iteration, so compaction never fires. `share` is
    /// O(1) until `MAX_CHAIN_DEPTH`, then flattens once.
    #[test]
    fn a_never_compacted_chain_stays_bounded_and_correct() {
        let mut writer = owned(&[(0, 0)]);
        let mut readers = Vec::new();
        for round in 1..=(MAX_CHAIN_DEPTH as u32 * 3) {
            // Each round: fork (a reader appears), then write.
            readers.push(TypeEntry::layered_over(writer.share()));
            writer.insert(Value::UniqueId(round), NodeIndex::new(round as usize));
        }

        assert!(
            writer.depth() <= MAX_CHAIN_DEPTH,
            "chain depth {} exceeded the cap",
            writer.depth()
        );
        // Every id ever written still resolves, through however many layers.
        for round in 1..=(MAX_CHAIN_DEPTH as u32 * 3) {
            assert_eq!(
                writer.get(&Value::UniqueId(round)),
                Some(NodeIndex::new(round as usize)),
                "id {round} lost in the chain"
            );
        }
        assert_eq!(writer.get(&Value::UniqueId(0)), Some(NodeIndex::new(0)));
        assert_eq!(writer.len(), MAX_CHAIN_DEPTH as usize * 3 + 1);

        // The oldest reader still sees only what existed when it forked.
        assert_eq!(readers[0].get(&Value::UniqueId(1)), None);
        assert_eq!(readers[0].get(&Value::UniqueId(0)), Some(NodeIndex::new(0)));

        // Dropping every reader lets one write collapse the whole chain.
        drop(readers);
        writer.try_compact();
        assert!(
            !writer.is_layered(),
            "a chain must collapse once nothing shares it"
        );
        assert_eq!(writer.len(), MAX_CHAIN_DEPTH as usize * 3 + 1);
    }

    /// The cap is **32**, and the flatten fires on the 33rd fork — asserted in
    /// literals, not in terms of the constant.
    ///
    /// `a_never_compacted_chain_stays_bounded_and_correct` above is written
    /// entirely in `MAX_CHAIN_DEPTH`, so it passes at any value: at 1 it would
    /// flatten every round (the O(N_type) cost the tuning removed), at 4096 it
    /// would retain 4096 deltas and probe all of them on a read miss, and the
    /// suite would stay green through either. The two static assertions beside
    /// the constant pin the *number*; this pins the **behaviour** the number
    /// buys, which is where a change to the comparison itself (`>=` vs `>`, or
    /// a saturating increment that stops climbing) would show up instead.
    #[test]
    fn the_chain_flattens_on_the_thirty_third_uncompacted_fork() {
        let mut writer = owned(&[(0, 0)]);
        let mut readers = Vec::new();
        let mut depths = Vec::new();
        for round in 1..=34u32 {
            readers.push(TypeEntry::layered_over(writer.share()));
            writer.insert(Value::UniqueId(round), NodeIndex::new(round as usize));
            depths.push(writer.depth());
        }

        // Rounds 1..=32 climb one level each; round 33 is the flatten, which
        // resets the chain to a single fresh level; round 34 climbs from there.
        let expected: Vec<u16> = (1..=32).chain([1, 2]).collect();
        assert_eq!(
            depths, expected,
            "the chain must climb to exactly 32 and wrap on the 33rd fork"
        );

        // The flatten is lossless: every id written before it still resolves,
        // and the oldest reader still sees only its own snapshot.
        for round in 1..=34u32 {
            assert_eq!(
                writer.get(&Value::UniqueId(round)),
                Some(NodeIndex::new(round as usize)),
                "id {round} lost across the flatten"
            );
        }
        assert_eq!(writer.get(&Value::UniqueId(0)), Some(NodeIndex::new(0)));
        assert_eq!(readers[0].get(&Value::UniqueId(1)), None);
    }
}
