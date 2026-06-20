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
//! Only **param-less, codec-free, no-disabled-passes, non-`text_score`**
//! queries are cached (see `session::execute::prepare`): with those excluded,
//! the optimized plan is a pure function of `(query, graph state)`, and
//! parameter binding happens later at execute time. text_score queries inject
//! per-call embedding params, so they're never inserted (the insert is gated
//! on the post-prepare param map staying empty) and therefore never hit.

use super::CypherQuery;
use std::collections::{HashMap, VecDeque};
use std::sync::{OnceLock, RwLock};

/// Maximum cached plans. A served graph cycles through a small working set of
/// queries at a stable version; 512 comfortably covers it. (Larger than the
/// parse cache because each graph-version generation gets its own entries; old
/// generations age out via FIFO as the working set re-populates post-mutation.)
const CACHE_CAPACITY: usize = 512;

/// `(graph_id, version, query_hash)` — see the module docs for why all three.
type PlanKey = (u64, u64, u64);

struct PlanCache {
    map: HashMap<PlanKey, CypherQuery>,
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

/// Look up a cached optimized plan for `query` against the graph identified by
/// `(graph_id, version)`. Returns a cloned plan on hit (the caller mutates it
/// further — `mark_lazy_eligibility` — so it must own it), `None` on miss.
pub fn get(graph_id: u64, version: u64, query: &str) -> Option<CypherQuery> {
    let key = (graph_id, version, hash_query(query));
    let guard = cache().read().expect("plan_cache RwLock poisoned");
    guard.map.get(&key).cloned()
}

/// Cache `plan` (the optimized AST, before lazy-marking) for `query` against
/// `(graph_id, version)`. FIFO-evicts the oldest entry at capacity.
pub fn insert(graph_id: u64, version: u64, query: &str, plan: &CypherQuery) {
    let key = (graph_id, version, hash_query(query));
    let mut guard = cache().write().expect("plan_cache RwLock poisoned");
    if guard.map.contains_key(&key) {
        return; // benign race: another thread inserted the same key.
    }
    if guard.map.len() >= CACHE_CAPACITY {
        if let Some(oldest) = guard.order.pop_front() {
            guard.map.remove(&oldest);
        }
    }
    guard.order.push_back(key);
    guard.map.insert(key, plan.clone());
}

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
    use std::sync::Mutex;

    // The cache is a process-wide singleton; serialize the cases so one
    // test's `clear_for_tests()` can't wipe another's entries mid-assert.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn plan(q: &str) -> CypherQuery {
        parse_cypher(q).expect("parse")
    }

    #[test]
    fn miss_then_hit_same_key() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        let q = "MATCH (n:T) RETURN n";
        assert!(get(1, 0, q).is_none(), "cold miss");
        insert(1, 0, q, &plan(q));
        assert!(get(1, 0, q).is_some(), "warm hit");
    }

    #[test]
    fn version_and_graph_id_partition_the_key() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        let q = "MATCH (n:T) RETURN n";
        insert(7, 3, q, &plan(q));
        // Same query, different version (mutation) or graph → must miss.
        assert!(get(7, 4, q).is_none(), "version change invalidates");
        assert!(get(8, 3, q).is_none(), "different graph never collides");
        assert!(get(7, 3, q).is_some(), "exact key hits");
    }

    #[test]
    fn evicts_at_capacity() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        for i in 0..(CACHE_CAPACITY as u64 + 5) {
            insert(1, i, "MATCH (n:T) RETURN n", &plan("MATCH (n:T) RETURN n"));
        }
        assert_eq!(entry_count_for_tests(), CACHE_CAPACITY);
    }
}
