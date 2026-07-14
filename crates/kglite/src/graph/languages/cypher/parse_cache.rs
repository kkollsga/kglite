//! Per-process LRU cache for parsed CypherQuery ASTs.
//!
//! Phase A.3 / 0.9.53 — Issue #2 fix.
//!
//! Pre-cache, every `cypher()` call re-parsed the input string from scratch.
//! The audit showed parse + plan accounts for ~80% of small-query cost
//! (1.1µs out of 1.4µs for a typical `MATCH (n {id: X}) RETURN n.prop`).
//! Bolt sessions with hot query loops (parameterized queries reissued many
//! times) pay this on every call.
//!
//! This module caches the **parsed AST only**. The planner / optimizer
//! still re-runs per call because plan output depends on graph state
//! (schema, cardinality estimates) which changes whenever the graph
//! mutates. Caching post-optimization would need a schema-version key
//! and cache-invalidation on every mutation — deferred to a future pass.
//!
//! Concurrency: the cache is a `RwLock<HashMap>`. Reads (cache hits) take
//! a read lock; misses upgrade to a write lock for insertion. Eviction
//! is FIFO at a capacity bound — simpler than true LRU and good enough
//! for the access patterns we observe (parameterized queries cycle
//! through a small fixed set).

use super::CypherQuery;
use crate::error::KgError;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Maximum cached ASTs. Bolt agents typically cycle through a small fixed
/// set of parameterized queries; 256 entries comfortably covers a session's
/// hot working set without unbounded memory growth.
const CACHE_CAPACITY: usize = 256;

/// Insertion-ordered list to drive FIFO eviction; we evict whatever was
/// inserted first when at capacity.
struct ParseCache {
    map: HashMap<u64, CypherQuery>,
    /// Insertion order — front = oldest. Cap mirrors `map.len()`.
    order: std::collections::VecDeque<u64>,
}

impl ParseCache {
    fn new() -> Self {
        Self {
            map: HashMap::with_capacity(CACHE_CAPACITY),
            order: std::collections::VecDeque::with_capacity(CACHE_CAPACITY),
        }
    }
}

static CACHE: OnceLock<RwLock<ParseCache>> = OnceLock::new();

fn cache() -> &'static RwLock<ParseCache> {
    CACHE.get_or_init(|| RwLock::new(ParseCache::new()))
}

fn hash_query(query: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    query.hash(&mut hasher);
    hasher.finish()
}

/// Cached parse. Returns a cloned AST on hit, or runs the real parser
/// and caches the result on miss.
///
/// The AST clone on hit is intentional — callers mutate the AST through
/// the optimizer (`cypher::optimize`) and we must not share an AST through
/// the cache. `CypherQuery`'s `Clone` impl is shallow-heap (a `Vec<Clause>`
/// plus a handful of metadata strings); cache HIT measured at ~700 ns end-to-
/// end for typical queries vs ~1.4 µs uncached.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn parse_cypher_cached(query: &str) -> Result<CypherQuery, KgError> {
    let key = hash_query(query);

    // Fast path: read lock, hit, clone, done.
    {
        let guard = cache().read().expect("parse_cache RwLock poisoned");
        if let Some(ast) = guard.map.get(&key) {
            return Ok(ast.clone());
        }
    }

    // Miss path: run the real parser (slow), then insert.
    let parsed = super::parser::parse_cypher(query)?;

    // Cache the parse result. Concurrent insertions of the same key are
    // benign (last-write-wins, both ASTs are equivalent).
    let mut guard = cache().write().expect("parse_cache RwLock poisoned");
    if guard.map.len() >= CACHE_CAPACITY && !guard.map.contains_key(&key) {
        // Evict the oldest entry.
        if let Some(oldest) = guard.order.pop_front() {
            guard.map.remove(&oldest);
        }
    }
    if !guard.map.contains_key(&key) {
        guard.order.push_back(key);
    }
    guard.map.insert(key, parsed.clone());

    Ok(parsed)
}

/// Drop all cached parses. Test-only — production has no reason to
/// invalidate the cache because parse output is purely a function of
/// the input text.
#[cfg(test)]
pub fn clear_for_tests() {
    let mut guard = cache().write().expect("parse_cache RwLock poisoned");
    guard.map.clear();
    guard.order.clear();
}

/// Current cache occupancy. Test-only.
#[cfg(test)]
pub fn entry_count_for_tests() -> usize {
    cache()
        .read()
        .expect("parse_cache RwLock poisoned")
        .map
        .len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The parse cache is a process-wide singleton; cargo test runs the
    /// `mod tests` cases in parallel by default and they interfere with
    /// each other (one test's `clear_for_tests()` wipes another's
    /// just-inserted entries before the assertion fires). Serialize via
    /// a test-only Mutex so each test gets exclusive access to the
    /// cache for its `clear → populate → assert` cycle. The lock is
    /// also held during the cache mutations so a failure-poisoned
    /// guard from one test surfaces as a "previous test poisoned the
    /// lock" rather than as a spooky count-mismatch in the next.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn cache_hit_returns_equivalent_ast() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        let q = "MATCH (n:Person) RETURN n.name";
        let first = parse_cypher_cached(q).unwrap();
        let second = parse_cypher_cached(q).unwrap();
        // ASTs are independently owned — mutation of one doesn't affect
        // the cached entry or other consumers.
        assert_eq!(first.clauses.len(), second.clauses.len());
        assert_eq!(entry_count_for_tests(), 1);
    }

    #[test]
    fn cache_evicts_at_capacity() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        // Insert CACHE_CAPACITY + 5 unique queries.
        for i in 0..(CACHE_CAPACITY + 5) {
            let q = format!("MATCH (n:T{}) RETURN n", i);
            parse_cypher_cached(&q).unwrap();
        }
        assert_eq!(entry_count_for_tests(), CACHE_CAPACITY);
    }

    #[test]
    fn parse_errors_are_not_cached() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_tests();
        let q = "MATCH NOT VALID CYPHER";
        let r1 = parse_cypher_cached(q);
        let r2 = parse_cypher_cached(q);
        assert!(r1.is_err());
        assert!(r2.is_err());
        // Cache should not have any entry for the failing query.
        assert_eq!(entry_count_for_tests(), 0);
    }
}
