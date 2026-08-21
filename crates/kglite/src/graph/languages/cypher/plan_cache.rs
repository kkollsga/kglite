//! Per-process cache for **optimized** CypherQuery plans.
//!
//! The sibling [`super::parse_cache`] caches the parsed AST; this caches the
//! post-optimizer plan, which its own comment flagged as the bigger win:
//! "parse + plan accounts for ~80% of small-query cost." The optimizer
//! (`planner::optimize_with_disabled`) re-runs on every call because its
//! output depends on graph state (schema + cardinality). This cache lets a
//! repeated query on an **unchanged graph** skip parse + validate + optimize
//! entirely — the common pattern for a served, read-heavy graph (bolt/mcp)
//! and for any hot read loop.
//!
//! ## Soundness — the key is `(graph_id, version, query)`
//!
//! - `version` changes on **every** mutation (see `DirGraph::bump_version`,
//!   wired into `execute_mut`, the bulk-ingest fns, and `make_dir_graph_mut`),
//!   so a cache hit means the graph is byte-for-byte the same state it was
//!   when the plan was computed → the cached plan is *identical* to
//!   re-optimizing. A mutation bumps `version` → the old key never hits again.
//! - `graph_id` is process-unique and never reused, so two different graphs
//!   that happen to share a `version` (e.g. both freshly loaded at version 0)
//!   can never collide on each other's plans.
//!
//! **`version` alone is not enough, and that is what `graph_id` is really
//! for.** Two transactions forked from one base bump in lockstep, so they hold
//! *different* graphs at the same version — the key is only unambiguous
//! because `fork_transaction` mints a fresh `graph_id` for a working copy.
//! Before it did, a sibling could be served a plan carrying the other fork's
//! resolved anchor `NodeIndex` and return a wrong count; the case is pinned in
//! `session::plan_cache_cost_tests::a_sibling_fork_is_never_served_another_forks_plan`.
//! Any future clone that becomes an independently mutable lineage owes itself
//! a new id for the same reason.
//!
//! ## Reads only
//!
//! `session::execute::prepare` inserts a plan only for a **non-mutating**
//! statement. A mutation bumps `version` right after its plan would be stored,
//! so the entry is unreachable to that writer forever — a serial writer
//! measured 0 hits in 600 identical writes while filling this whole cache with
//! its own dead entries and evicting other graphs' live read plans. The
//! *lookup* is not skipped (classification does not exist yet at that point in
//! `prepare`; see the comment there), it is simply a guaranteed miss.
//!
//! ## What a hit carries besides the plan
//!
//! The non-fatal schema warnings (`schema_check::collect_unknown_pattern_warnings`
//! — unknown label / relationship type, with a "did you mean?") ride on the
//! entry. They are a pure function of `(query, graph schema)`, and the key
//! already pins the graph state, so a hit hands back exactly what a miss would
//! have computed. Re-deriving them at lookup time is not an option: it needs
//! the parsed AST, and skipping the parse is the whole point of this cache.
//! Not carrying them at all is worse — the *second* run of a typo'd query
//! would silently lose its warning, which is the shape this cache had before
//! `QueryDiagnostics.warnings` was populated.
//!
//! Only **param-less, codec-free, no-disabled-passes, non-`text_score`**
//! queries are cached (see `session::execute::prepare`): with those excluded,
//! the optimized plan is a pure function of `(query, graph state)`, and
//! parameter binding happens later at execute time. text_score queries inject
//! per-call embedding params, so they're never inserted (the insert is gated
//! on the post-prepare param map staying empty) and therefore never hit.

use super::CypherQuery;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock, RwLock};

/// Maximum cached plans. A served graph cycles through a small working set of
/// queries at a stable version; 512 comfortably covers it. (Larger than the
/// parse cache because each graph-version generation gets its own entries; old
/// generations age out via FIFO as the working set re-populates post-mutation.)
pub(crate) const CACHE_CAPACITY: usize = 512;

/// `(graph_id, version, lazy_eligible, query_hash)`. `lazy_eligible` is part of
/// the key because the cached plan is stored **post lazy-marking** (so a hit is
/// a pure `Arc` clone with no per-call mutation); the wheel runs
/// `lazy_eligible=true`, the bolt/mcp servers `false`, so each gets its own
/// variant. See the module docs for `graph_id` / `version`.
type PlanKey = (u64, u64, bool, u64);

/// What a lookup hands back: the ready-to-execute plan plus the schema
/// warnings computed for it (see the module docs). Both are behind `Arc`, so a
/// hit is two refcount bumps and no clone of either payload.
#[derive(Clone)]
pub struct CachedPlan {
    pub plan: Arc<CypherQuery>,
    pub warnings: Arc<[String]>,
}

struct PlanCache {
    /// Plans are stored behind `Arc` so a cache hit is a refcount bump, not a
    /// deep AST clone — execute borrows the plan read-only, so sharing is safe.
    map: HashMap<PlanKey, CachedPlan>,
    /// Insertion order — front = oldest, for FIFO eviction at capacity.
    order: VecDeque<PlanKey>,
}

impl PlanCache {
    fn new() -> Self {
        Self {
            map: HashMap::with_capacity(CACHE_CAPACITY),
            order: VecDeque::with_capacity(CACHE_CAPACITY),
        }
    }
}

static CACHE: OnceLock<RwLock<PlanCache>> = OnceLock::new();

fn cache() -> &'static RwLock<PlanCache> {
    CACHE.get_or_init(|| RwLock::new(PlanCache::new()))
}

fn hash_query(query: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    query.hash(&mut hasher);
    hasher.finish()
}

/// Look up a cached, ready-to-execute plan for `query` against the graph
/// identified by `(graph_id, version)` at the given `lazy_eligible` mode.
/// Returns an `Arc` clone on hit (no AST copy), `None` on miss.
pub fn get(graph_id: u64, version: u64, lazy: bool, query: &str) -> Option<CachedPlan> {
    let key = (graph_id, version, lazy, hash_query(query));
    let guard = cache().read().expect("plan_cache RwLock poisoned");
    let hit = guard.map.get(&key).cloned();
    #[cfg(test)]
    instrumentation::record_lookup(hit.is_some());
    hit
}

/// Cache `plan` (the optimized AST, already lazy-marked for `lazy`) plus the
/// schema `warnings` computed for it, for `query` against `(graph_id,
/// version)`. FIFO-evicts the oldest entry at capacity.
pub fn insert(
    graph_id: u64,
    version: u64,
    lazy: bool,
    query: &str,
    plan: Arc<CypherQuery>,
    warnings: Arc<[String]>,
) {
    let key = (graph_id, version, lazy, hash_query(query));
    let mut guard = cache().write().expect("plan_cache RwLock poisoned");
    if guard.map.contains_key(&key) {
        return; // benign race: another thread inserted the same key.
    }
    if guard.map.len() >= CACHE_CAPACITY {
        if let Some(oldest) = guard.order.pop_front() {
            guard.map.remove(&oldest);
            #[cfg(test)]
            instrumentation::record_eviction();
        }
    }
    guard.order.push_back(key);
    guard.map.insert(key, CachedPlan { plan, warnings });
    #[cfg(test)]
    instrumentation::record_insertion();
}

/// Test-only event counters, split by the kind of statement whose `prepare()`
/// caused the event.
///
/// **Why the caller kind cannot simply be passed in.** `prepare()` looks the
/// plan up *before* anything has parsed the query, so at lookup time nobody
/// knows whether the statement mutates; `is_mutation_query` runs on the
/// prepared plan, one line later in `execute_read` / `execute_mut`. So events
/// are buffered per in-flight `prepare()` and attributed retroactively by
/// [`classify_pending`] once the caller knows. A `prepare()` that ends in an
/// error never classifies, and its buffered events are folded into
/// [`CallerStats::unclassified`] by the next [`begin_prepare`] rather than
/// silently landing in the next statement's bucket.
///
/// **Why thread-local and not global.** The cache is process-wide, but `cargo
/// test` runs cases on separate threads: a global counter would interleave
/// unrelated tests and force every counter-reading case onto one lock. Every
/// event is recorded on the thread that caused it, so per-thread totals are
/// exactly "what this test did". Cross-thread interference in the *map* is a
/// non-issue for the same reason the cache is sound at all — `graph_id` is
/// process-unique, so no other test's entries share a key.
#[cfg(test)]
pub mod instrumentation {
    use std::cell::Cell;

    #[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CacheStats {
        /// Calls to [`super::get`].
        pub lookups: u64,
        /// Subset of `lookups` that returned a plan.
        pub hits: u64,
        /// Entries actually added by [`super::insert`] (a benign-race duplicate
        /// key returns early and is not counted).
        pub insertions: u64,
        /// FIFO evictions forced by those insertions.
        pub evictions: u64,
    }

    const EMPTY: CacheStats = CacheStats {
        lookups: 0,
        hits: 0,
        insertions: 0,
        evictions: 0,
    };

    impl CacheStats {
        fn add(self, other: CacheStats) -> CacheStats {
            CacheStats {
                lookups: self.lookups + other.lookups,
                hits: self.hits + other.hits,
                insertions: self.insertions + other.insertions,
                evictions: self.evictions + other.evictions,
            }
        }
    }

    #[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CallerStats {
        /// Events caused by a statement that turned out to be a read.
        pub read: CacheStats,
        /// Events caused by a statement that turned out to be a mutation.
        pub mutation: CacheStats,
        /// Events from a `prepare()` that never reached classification — an
        /// error before `is_mutation_query`, or a direct `get`/`insert` call
        /// from a `plan_cache` unit test.
        pub unclassified: CacheStats,
    }

    thread_local! {
        /// Events of the `prepare()` currently in flight on this thread.
        static PENDING: Cell<CacheStats> = const { Cell::new(EMPTY) };
        static TOTALS: Cell<CallerStats> = const {
            Cell::new(CallerStats { read: EMPTY, mutation: EMPTY, unclassified: EMPTY })
        };
    }

    fn bump(f: impl FnOnce(&mut CacheStats)) {
        PENDING.with(|pending| {
            let mut stats = pending.get();
            f(&mut stats);
            pending.set(stats);
        });
    }

    pub(super) fn record_lookup(hit: bool) {
        bump(|stats| {
            stats.lookups += 1;
            stats.hits += u64::from(hit);
        });
    }

    pub(super) fn record_insertion() {
        bump(|stats| stats.insertions += 1);
    }

    pub(super) fn record_eviction() {
        bump(|stats| stats.evictions += 1);
    }

    fn take_pending() -> CacheStats {
        PENDING.with(|pending| pending.replace(EMPTY))
    }

    /// Open a fresh attribution window. Any events still buffered belong to a
    /// `prepare()` that errored out before classifying, so they are banked as
    /// `unclassified` instead of contaminating this statement.
    pub fn begin_prepare() {
        let leftover = take_pending();
        if leftover != EMPTY {
            TOTALS.with(|totals| {
                let mut all = totals.get();
                all.unclassified = all.unclassified.add(leftover);
                totals.set(all);
            });
        }
    }

    /// Attribute the in-flight `prepare()`'s events now that the caller has
    /// classified the statement.
    pub fn classify_pending(is_mutation: bool) {
        let pending = take_pending();
        TOTALS.with(|totals| {
            let mut all = totals.get();
            if is_mutation {
                all.mutation = all.mutation.add(pending);
            } else {
                all.read = all.read.add(pending);
            }
            totals.set(all);
        });
    }

    /// Zero this thread's counters, including any unclassified in-flight events.
    pub fn reset() {
        take_pending();
        TOTALS.with(|totals| totals.set(CallerStats::default()));
    }

    /// This thread's classified totals since the last [`reset`].
    pub fn totals() -> CallerStats {
        TOTALS.with(|totals| totals.get())
    }
}

/// Serializes every test whose assertion depends on cache *contents*. The
/// cache is a process-wide singleton, so one test's [`clear_for_tests`] (or a
/// capacity test's 600 inserts) would otherwise evict another's entry between
/// its insert and its assert. Shared with `session::plan_cache_cost_tests`.
#[cfg(test)]
pub static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub fn clear_for_tests() {
    let mut guard = cache().write().expect("plan_cache RwLock poisoned");
    guard.map.clear();
    guard.order.clear();
}

#[cfg(test)]
pub fn entry_count_for_tests() -> usize {
    cache()
        .read()
        .expect("plan_cache RwLock poisoned")
        .map
        .len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::languages::cypher::parser::parse_cypher;

    fn plan(q: &str) -> Arc<CypherQuery> {
        Arc::new(parse_cypher(q).expect("parse"))
    }

    fn no_warnings() -> Arc<[String]> {
        Vec::new().into()
    }

    #[test]
    fn miss_then_hit_same_key() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        let q = "MATCH (n:T) RETURN n";
        assert!(get(1, 0, false, q).is_none(), "cold miss");
        insert(
            1,
            0,
            false,
            q,
            plan(q),
            vec!["typo'd label".to_string()].into(),
        );
        let hit = get(1, 0, false, q).expect("warm hit");
        assert_eq!(
            &*hit.warnings,
            ["typo'd label".to_string()],
            "a hit carries the warnings its miss computed"
        );
    }

    #[test]
    fn version_graph_id_and_lazy_partition_the_key() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        let q = "MATCH (n:T) RETURN n";
        insert(7, 3, false, q, plan(q), no_warnings());
        // Same query, different version / graph / lazy-mode → must miss.
        assert!(get(7, 4, false, q).is_none(), "version change invalidates");
        assert!(
            get(8, 3, false, q).is_none(),
            "different graph never collides"
        );
        assert!(get(7, 3, true, q).is_none(), "lazy mode is part of the key");
        assert!(get(7, 3, false, q).is_some(), "exact key hits");
    }

    #[test]
    fn evicts_at_capacity() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        for i in 0..(CACHE_CAPACITY as u64 + 5) {
            insert(
                1,
                i,
                false,
                "MATCH (n:T) RETURN n",
                plan("MATCH (n:T) RETURN n"),
                no_warnings(),
            );
        }
        assert_eq!(entry_count_for_tests(), CACHE_CAPACITY);
    }
}
