//! Pattern-compilation cache for Cypher `text_match_regex(...)` queries.
//!
//! Regex compilation is non-trivial (microseconds to milliseconds for
//! complex patterns); caching across query executions matters when the
//! same pattern is used in a hot loop ("filter every row by this
//! regex"). The Cypher executor compiles each pattern at most once
//! per process via this cache.
//!
//! Cache shape: `pattern_string → Arc<Regex>`. Read lock for the hit
//! path (free clone of the Arc), write lock only on miss. No
//! eviction: practical Cypher queries use a small set of patterns
//! (each query has a handful at most), and a `Regex` is ~1 KB. We
//! grow until process exit. If a deployment turns out to thrash the
//! cache (>10k distinct patterns), revisit; for now unbounded is
//! correct.
//!
//! Returns the regex `Error` on invalid patterns so the Cypher
//! executor can surface them as `KgError::CypherExecution`.

use regex::Regex;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

static CACHE: LazyLock<RwLock<HashMap<String, Arc<Regex>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Look up `pattern` in the cache; compile + insert on miss.
/// Returns the shared `Arc<Regex>` for `is_match` / `find` / etc.
pub fn get_or_compile(pattern: &str) -> Result<Arc<Regex>, regex::Error> {
    // Fast path: cache hit. Cheap Arc::clone.
    if let Some(cached) = CACHE.read().unwrap().get(pattern) {
        return Ok(cached.clone());
    }
    // Miss: compile under write lock. Re-check inside the write lock
    // in case a concurrent thread populated the entry between our
    // read-release and write-acquire.
    let mut guard = CACHE.write().unwrap();
    if let Some(cached) = guard.get(pattern) {
        return Ok(cached.clone());
    }
    let compiled = Arc::new(Regex::new(pattern)?);
    guard.insert(pattern.to_string(), compiled.clone());
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_and_caches() {
        let re1 = get_or_compile(r"^\d+$").unwrap();
        let re2 = get_or_compile(r"^\d+$").unwrap();
        // Same pattern → same Arc (cache hit on second call).
        assert!(Arc::ptr_eq(&re1, &re2));
        assert!(re1.is_match("12345"));
        assert!(!re1.is_match("abc"));
    }

    #[test]
    fn invalid_pattern_errors_cleanly() {
        let r = get_or_compile(r"(?P<bad");
        assert!(r.is_err());
    }

    #[test]
    fn flags_inline_work() {
        let re = get_or_compile(r"(?i)hello").unwrap();
        assert!(re.is_match("HELLO"));
        assert!(re.is_match("Hello"));
        assert!(re.is_match("hello"));
    }
}
