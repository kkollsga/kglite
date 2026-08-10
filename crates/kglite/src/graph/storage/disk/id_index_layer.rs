//! One type's id index, either owned outright or **layered over a base a
//! forked graph is still reading**.
//!
//! ## Why
//!
//! `id_indices` is a `HashMap` with one entry per node of every materialised
//! type, so `DirGraph::clone` deep-copies it on every fork. Measured 2026-08-10
//! at 1M nodes: **3.7 ms**, which after D2 Phase 2 removed the backend row is
//! **90% of everything left** in a plain graph's fork
//! (`dev-docs/bench/results/2026-08-10-d2-phase2-forked.md` §D.3). This type is
//! what makes that O(changes).
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
/// live reader compacts the whole thing — so in practice it stays at 1. The cap
/// makes the pathological case (a loop that re-takes a view every iteration,
/// forever) amortised O(N/K) rather than unbounded, and it is deliberately
/// small: the flatten it triggers is the same merge this recursion exists to
/// avoid, so the win is in how rarely it runs, not in the constant.
const MAX_CHAIN_DEPTH: u16 = 8;

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
}
