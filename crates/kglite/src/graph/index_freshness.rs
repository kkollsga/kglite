//! Incremental catch-up bookkeeping, shared by every index kind.
//!
//! An index built over a live graph goes out of date the moment the graph
//! moves. The three ways to handle that are: maintain the index on every write
//! (a tax on ingest, paid by every graph whether it has an index or not),
//! rebuild on demand (O(corpus) for a one-node change), or **track what
//! changed and re-read only that**. This module is the third.
//!
//! # What is tracked, and what it costs
//!
//! [`IndexFreshness`] holds one number and one set per index instance:
//!
//! * a **high-watermark** — the node-slot bound the last build or refresh
//!   covered. Every slot handed out after it is new by construction, so node
//!   creation needs no per-row *accounting* to be noticed: the gap between the
//!   watermark and the graph's current slot bound *is* the creation delta.
//!   That is why bulk ingest is untouched by an index existing, and why a
//!   creation funnel that forgets to call in here is still caught — the gap is
//!   read from the graph, not accumulated from notifications.
//! * a **dirty set** — slots *below* the watermark that need re-reading. Two
//!   things land here, and only two: a write to the indexed property of an
//!   already-covered node, and a creation into a **recycled slot**.
//!
//! The recycled slot is the one the watermark cannot see. `StableDiGraph` hands
//! a freed `NodeIndex` to the next node created, so a creation can land *below*
//! the watermark and look, to the gap arithmetic alone, like a node that was
//! already indexed. The check that catches it is one `u32` comparison at the
//! creation site — [`IndexFreshness::note_created`].
//!
//! # Over-approximation is the safe direction
//!
//! Every producer here may mark more than strictly changed; none may mark less.
//! A redundant re-read costs one tokenization and yields the same index; a
//! missed one is a wrong answer that no later operation notices. So a bulk
//! update path that cannot say *which* fields it wrote marks the node dirty
//! regardless, a rolled-back statement leaves its slots dirty, and
//! [`IndexFreshness::delta_size`] is an upper bound rather than an exact count.
//!
//! The one place that bound is tightened is a creation of a type the index does
//! not cover, landing exactly at the watermark: the watermark steps over it, so
//! bulk-loading an unrelated node type does not make every other index look
//! stale by a million documents. It stays sound because a *later* node taking
//! that slot is, by then, a below-watermark creation — the recycled-slot case
//! the dirty set already exists for.
//!
//! # Threshold policy
//!
//! [`IndexFreshness::delta_size`] is O(1) — two relaxed atomic loads and a
//! subtraction — so a query can ask "is this worth refreshing inline?" for
//! free. Under the limit, the caller refreshes and serves fresh results; over
//! it, the caller serves what it has and says so, rather than hiding a
//! corpus-sized rebuild inside someone's query.
//!
//! # Concurrency and lock order
//!
//! Refresh happens at **query** entry, where the graph is `&DirGraph` — the
//! read path cannot take `&mut`, so the state here is behind interior
//! mutability and every mutator takes `&self`. Writers are exclusive by Rust's
//! own rules (`&mut DirGraph`), so the only real contention is several readers
//! discovering staleness at once.
//!
//! **The lock order is: the index's own lock first, this module's second, and
//! never the reverse.** A refresher takes the index's write lock, *then* calls
//! [`IndexFreshness::take_delta`]. That ordering is what makes the
//! double-check correct: a second reader either blocks on the index lock and
//! then finds an empty delta, or sees a zero delta and blocks on the index lock
//! anyway — in both cases it reads the refreshed index, never a half-applied
//! one. Taking this module's lock first would let a reader clear the delta and
//! release, and a second reader would then read the *unrefreshed* index while
//! believing it fresh.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

use rustc_hash::FxHashSet;

/// Documents an index will fold in inline at query entry before it declines and
/// serves stale results instead.
///
/// A round number rather than a measured one: the point is to bound the pause a
/// query can inherit, and one thousand documents is small enough that the
/// refresh disappears into the query it rides on. Callers that know their
/// corpus can override it per index.
pub const DEFAULT_AUTO_REFRESH_LIMIT: usize = 1000;

/// Change tracking for one index instance, independent of what the index holds.
///
/// See the module docs for the watermark/dirty-set split and the lock order.
/// Nothing here knows about postings, vectors, or properties: a kind attaches
/// this, notifies it from the write path, and asks [`Self::take_delta`] for the
/// slots to re-read.
#[derive(Debug)]
pub struct IndexFreshness {
    /// Slot bound the last build/refresh covered.
    watermark: AtomicU32,
    /// `dirty.len()`, mirrored so [`Self::delta_size`] needs no lock.
    dirty_len: AtomicUsize,
    /// Inline-refresh ceiling in documents.
    limit: AtomicUsize,
    /// Slots below the watermark that need re-reading.
    dirty: Mutex<FxHashSet<u32>>,
}

impl IndexFreshness {
    /// Freshness state for an index that has just covered every slot below
    /// `node_bound`.
    pub fn covering(node_bound: u32, limit: Option<usize>) -> Self {
        Self {
            watermark: AtomicU32::new(node_bound),
            dirty_len: AtomicUsize::new(0),
            limit: AtomicUsize::new(limit.unwrap_or(DEFAULT_AUTO_REFRESH_LIMIT)),
            dirty: Mutex::new(FxHashSet::default()),
        }
    }

    fn dirty_set(&self) -> std::sync::MutexGuard<'_, FxHashSet<u32>> {
        // A panic inside a critical section here would have to come from the
        // allocator; the state it guards stays coherent either way, so a
        // poisoned lock is recovered rather than propagated into a query.
        self.dirty.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The slot bound the index currently covers.
    pub fn watermark(&self) -> u32 {
        self.watermark.load(Ordering::Relaxed)
    }

    /// The inline-refresh ceiling.
    pub fn limit(&self) -> usize {
        self.limit.load(Ordering::Relaxed)
    }

    /// Documents the next refresh would re-read, given the graph's current slot
    /// bound. O(1), lock-free, and an **upper bound** — see the
    /// over-approximation note in the module docs.
    pub fn delta_size(&self, node_bound: u32) -> usize {
        let gap = node_bound.saturating_sub(self.watermark()) as usize;
        gap + self.dirty_len.load(Ordering::Relaxed)
    }

    /// Whether anything has changed since the last build or refresh.
    pub fn is_stale(&self, node_bound: u32) -> bool {
        self.delta_size(node_bound) > 0
    }

    /// Whether the outstanding delta is small enough to fold in inline.
    ///
    /// `false` for a clean index too — there is nothing to fold in — so this
    /// answers "refresh now?", not "is the delta small?".
    pub fn within_limit(&self, node_bound: u32) -> bool {
        let delta = self.delta_size(node_bound);
        delta > 0 && delta <= self.limit()
    }

    /// A node was created at `slot`. `covered` says whether this index would
    /// hold a document for it — a text index covering `Person` passes `false`
    /// for a `Company`.
    ///
    /// Three cases, all O(1): a covered creation above the watermark needs
    /// nothing (the gap has it); a creation below the watermark is a recycled
    /// `NodeIndex` and goes into the dirty set; a *foreign* creation exactly at
    /// the watermark steps the watermark over it, so an unrelated bulk load
    /// does not inflate this index's delta.
    #[inline]
    pub fn note_created(&self, slot: u32, covered: bool) {
        let watermark = self.watermark.load(Ordering::Relaxed);
        if slot < watermark {
            if covered {
                self.note_changed(slot);
            }
            // A foreign node in a recycled slot needs nothing: whatever
            // document used to live there was pruned when its node was
            // deleted, which is what freed the slot.
            return;
        }
        if !covered && slot == watermark {
            self.watermark.store(slot + 1, Ordering::Relaxed);
        }
    }

    /// A covered node's indexed content changed at `slot`.
    ///
    /// A no-op above the watermark: those slots are already in the gap the next
    /// refresh walks.
    #[inline]
    pub fn note_changed(&self, slot: u32) {
        if slot >= self.watermark.load(Ordering::Relaxed) {
            return;
        }
        let mut dirty = self.dirty_set();
        dirty.insert(slot);
        self.dirty_len.store(dirty.len(), Ordering::Relaxed);
    }

    /// Claim the outstanding delta and mark the index current up to
    /// `node_bound`. `None` when there is nothing to do.
    ///
    /// **The caller must already hold the index's own write lock** — that is
    /// the whole double-check (module docs, "Concurrency and lock order").
    pub fn take_delta(&self, node_bound: u32) -> Option<FreshnessDelta> {
        let mut dirty = self.dirty_set();
        let from = self.watermark.load(Ordering::Relaxed);
        if dirty.is_empty() && from >= node_bound {
            return None;
        }
        let taken = std::mem::take(&mut *dirty);
        self.dirty_len.store(0, Ordering::Relaxed);
        self.watermark
            .store(node_bound.max(from), Ordering::Relaxed);
        Some(FreshnessDelta {
            from,
            to: node_bound.max(from),
            dirty: taken,
        })
    }
}

impl Clone for IndexFreshness {
    /// Deep, never shared. A fork's index is its own (`independent_copy`), so
    /// its freshness must be too — a dirty set shared with the parent would let
    /// one graph's refresh silence the other's.
    fn clone(&self) -> Self {
        let dirty = self.dirty_set().clone();
        Self {
            watermark: AtomicU32::new(self.watermark()),
            dirty_len: AtomicUsize::new(dirty.len()),
            limit: AtomicUsize::new(self.limit()),
            dirty: Mutex::new(dirty),
        }
    }
}

/// The slots one refresh has to re-read, claimed atomically from an
/// [`IndexFreshness`].
#[derive(Debug)]
pub struct FreshnessDelta {
    from: u32,
    to: u32,
    dirty: FxHashSet<u32>,
}

impl FreshnessDelta {
    /// Every slot to re-read, each exactly once.
    ///
    /// A dirty slot is below the watermark by construction
    /// ([`IndexFreshness::note_changed`]), so the two sources cannot overlap
    /// and no deduplication is needed — asserted in debug.
    pub fn slots(&self) -> impl Iterator<Item = u32> + '_ {
        debug_assert!(
            self.dirty.iter().all(|slot| *slot < self.from),
            "a dirty slot at or above the old watermark would be walked twice"
        );
        self.dirty.iter().copied().chain(self.from..self.to)
    }
}

/// Where the write path tells the indexes something moved.
///
/// One funnel, so a write site names *changes*, not index kinds — and so
/// adding a kind (the vector adoption in P10c) is an arm here rather than an
/// edit to every `CREATE` and `SET` in the executor.
///
/// **The global gate is the design promise.** Every function starts with one
/// predictable branch — "does this graph have any index that tracks freshness?"
/// — and a graph with none pays that branch and nothing else, per row. Keep it
/// that way: work that a graph without an index can observe does not belong
/// below this line.
pub(crate) mod write_hooks {
    use petgraph::graph::NodeIndex;

    use crate::graph::dir_graph::DirGraph;

    // Counts hook invocations that got past the global gate. The
    // zero-cost-when-unindexed promise is a *structural* claim about a branch,
    // not a timing claim, so it is tested structurally. Thread-local because
    // the test harness runs tests in parallel threads and a shared counter
    // would make every such assertion a race.
    #[cfg(test)]
    thread_local! {
        static WORK_PAST_GATE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Hook invocations on this thread that found an index to notify.
    #[cfg(test)]
    pub(crate) fn work_past_gate() -> usize {
        WORK_PAST_GATE.with(std::cell::Cell::get)
    }

    #[inline]
    fn note_work() {
        #[cfg(test)]
        WORK_PAST_GATE.with(|count| count.set(count.get() + 1));
    }

    /// Whether any freshness-tracking index exists on this graph — the global
    /// gate, and the only thing an unindexed graph's write path evaluates.
    ///
    /// `pub(crate)` because a caller that would have to *derive* the hook's
    /// arguments (resolve a node type, allocate its name) has to be able to ask
    /// first; deriving them and then discovering the gate is closed is exactly
    /// the per-row cost this design exists to avoid.
    #[inline]
    pub(crate) fn any_tracked_index(graph: &DirGraph) -> bool {
        // P10c adds `|| !graph.embeddings.is_empty()` here, and its own arm
        // below each gate.
        !graph.text_indexes.is_empty()
    }

    /// A node of `node_type` was created at `node`.
    #[inline]
    pub(crate) fn note_node_created(graph: &DirGraph, node: NodeIndex, node_type: &str) {
        if !any_tracked_index(graph) {
            return;
        }
        note_work();
        crate::graph::text_indexes::note_node_created(graph, node, node_type);
    }

    /// A covered node's property was written.
    ///
    /// `field` is the **alias-resolved** field the write landed in, so an index
    /// that records its own resolved field can tell "the indexed property
    /// moved" from "some other property moved" without re-resolving per row.
    /// `None` means the caller wrote a set of fields it did not decompose —
    /// every index on the type treats that as a change (over-approximation, per
    /// the module docs).
    #[inline]
    pub(crate) fn note_property_written(
        graph: &DirGraph,
        node: NodeIndex,
        node_type: &str,
        field: Option<&str>,
    ) {
        if !any_tracked_index(graph) {
            return;
        }
        note_work();
        crate::graph::text_indexes::note_property_written(graph, node, node_type, field);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_creation_above_the_watermark_needs_no_dirty_entry() {
        let freshness = IndexFreshness::covering(10, None);
        freshness.note_created(10, true);
        freshness.note_created(11, true);

        assert_eq!(freshness.delta_size(12), 2, "the gap is the delta");
        assert!(
            freshness.dirty_set().is_empty(),
            "the watermark gap already covers them"
        );
        let delta = freshness.take_delta(12).expect("outstanding");
        let mut slots: Vec<u32> = delta.slots().collect();
        slots.sort_unstable();
        assert_eq!(slots, vec![10, 11]);
        assert_eq!(freshness.delta_size(12), 0);
        assert_eq!(freshness.watermark(), 12);
    }

    /// The gap is read from the graph, not accumulated: a creation funnel that
    /// never calls in here is still caught.
    #[test]
    fn an_unnotified_creation_is_still_in_the_delta() {
        let freshness = IndexFreshness::covering(4, None);
        assert_eq!(freshness.delta_size(6), 2);
        assert!(freshness.is_stale(6));
    }

    /// The hole the dirty set exists for: petgraph hands a freed slot back out,
    /// and the gap arithmetic alone would read that creation as "already
    /// indexed".
    #[test]
    fn a_creation_into_a_recycled_slot_goes_into_the_dirty_set() {
        let freshness = IndexFreshness::covering(10, None);
        freshness.note_created(4, true);

        assert_eq!(freshness.delta_size(10), 1);
        let delta = freshness.take_delta(10).expect("outstanding");
        assert_eq!(delta.slots().collect::<Vec<_>>(), vec![4]);
    }

    /// Bulk-loading a type this index does not cover must not make it look
    /// stale by a million documents.
    #[test]
    fn a_foreign_creation_at_the_watermark_steps_the_watermark_over_it() {
        let freshness = IndexFreshness::covering(10, None);
        for slot in 10..20 {
            freshness.note_created(slot, false);
        }

        assert_eq!(freshness.watermark(), 20);
        assert_eq!(freshness.delta_size(20), 0, "none of them is a document");
        assert!(!freshness.is_stale(20));
    }

    /// …and a node of the covered type later taking one of those slots is a
    /// below-watermark creation, which the dirty set does see.
    #[test]
    fn a_covered_node_reusing_a_stepped_over_slot_is_still_caught() {
        let freshness = IndexFreshness::covering(10, None);
        freshness.note_created(10, false);
        assert_eq!(freshness.watermark(), 11);

        freshness.note_created(10, true);

        assert_eq!(freshness.delta_size(11), 1);
        let delta = freshness.take_delta(11).expect("outstanding");
        assert_eq!(delta.slots().collect::<Vec<_>>(), vec![10]);
    }

    #[test]
    fn a_change_above_the_watermark_is_not_counted_twice() {
        let freshness = IndexFreshness::covering(10, None);
        freshness.note_changed(10);
        freshness.note_changed(10);

        assert_eq!(freshness.delta_size(11), 1, "the gap, not a dirty entry");
        assert!(freshness.dirty_set().is_empty());
    }

    #[test]
    fn a_repeated_change_to_one_slot_counts_once() {
        let freshness = IndexFreshness::covering(10, None);
        freshness.note_changed(3);
        freshness.note_changed(3);

        assert_eq!(freshness.delta_size(10), 1);
    }

    #[test]
    fn the_limit_gates_inline_refresh_without_hiding_staleness() {
        let freshness = IndexFreshness::covering(0, Some(2));
        assert!(!freshness.is_stale(0));
        assert!(
            !freshness.within_limit(0),
            "a clean index refreshes nothing"
        );

        assert!(freshness.is_stale(2));
        assert!(freshness.within_limit(2));

        assert!(freshness.is_stale(3), "over the limit is still stale");
        assert!(!freshness.within_limit(3));
        assert_eq!(freshness.delta_size(3), 3);
    }

    #[test]
    fn take_delta_is_empty_when_nothing_moved() {
        let freshness = IndexFreshness::covering(7, None);
        assert!(freshness.take_delta(7).is_none());
    }

    #[test]
    fn a_clone_shares_no_state_with_its_source() {
        let freshness = IndexFreshness::covering(5, Some(9));
        freshness.note_changed(2);
        let copy = freshness.clone();

        assert_eq!(copy.delta_size(5), 1);
        assert_eq!(copy.limit(), 9);
        assert_eq!(copy.watermark(), 5);

        copy.note_changed(3);
        assert_eq!(copy.delta_size(5), 2);
        assert_eq!(freshness.delta_size(5), 1, "the source must not move");
    }
}
