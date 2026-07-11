//! Bounded pattern-compilation cache for Cypher regex functions.
//!
//! Compilation happens outside the lock so an expensive or invalid pattern
//! cannot stall unrelated cache hits. The cache uses FIFO eviction: repeated
//! query patterns still hit the cache, while adversarial streams of unique
//! patterns cannot grow process memory without bound.

use regex::{Regex, RegexBuilder};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

const CACHE_CAPACITY: usize = 128;
const REGEX_SIZE_LIMIT: usize = 2 * 1024 * 1024;

#[derive(Default)]
struct CacheEntries {
    values: HashMap<String, Arc<Regex>>,
    insertion_order: VecDeque<String>,
}

struct RegexCache {
    capacity: usize,
    entries: RwLock<CacheEntries>,
}

impl RegexCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: RwLock::new(CacheEntries::default()),
        }
    }

    fn read(&self) -> RwLockReadGuard<'_, CacheEntries> {
        self.entries
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, CacheEntries> {
        self.entries
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn get_or_compile(&self, pattern: &str) -> Result<Arc<Regex>, regex::Error> {
        if let Some(cached) = self.read().values.get(pattern) {
            return Ok(Arc::clone(cached));
        }

        // Intentionally outside the write lock. Concurrent misses may compile
        // the same pattern twice, but only one value is published.
        // Bound each compiled automaton as well as the entry count. A count-only
        // cache could otherwise retain hundreds of multi-megabyte programs.
        let compiled = Arc::new(
            RegexBuilder::new(pattern)
                .size_limit(REGEX_SIZE_LIMIT)
                .build()?,
        );
        if self.capacity == 0 {
            return Ok(compiled);
        }

        let mut entries = self.write();
        if let Some(cached) = entries.values.get(pattern) {
            return Ok(Arc::clone(cached));
        }
        while entries.values.len() >= self.capacity {
            if let Some(oldest) = entries.insertion_order.pop_front() {
                entries.values.remove(&oldest);
            } else if let Some(oldest) = entries.values.keys().next().cloned() {
                // A recovered poisoned lock may expose partially-updated
                // bookkeeping. Preserve the hard capacity bound regardless.
                entries.values.remove(&oldest);
            } else {
                break;
            }
        }
        entries.insertion_order.push_back(pattern.to_owned());
        entries
            .values
            .insert(pattern.to_owned(), Arc::clone(&compiled));
        Ok(compiled)
    }
}

static CACHE: LazyLock<RegexCache> = LazyLock::new(|| RegexCache::new(CACHE_CAPACITY));

/// Look up `pattern` in the process-wide cache; compile and insert on miss.
pub fn get_or_compile(pattern: &str) -> Result<Arc<Regex>, regex::Error> {
    CACHE.get_or_compile(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_and_caches() {
        let cache = RegexCache::new(4);
        let re1 = cache.get_or_compile(r"^\d+$").unwrap();
        let re2 = cache.get_or_compile(r"^\d+$").unwrap();
        assert!(Arc::ptr_eq(&re1, &re2));
        assert!(re1.is_match("12345"));
        assert!(!re1.is_match("abc"));
    }

    #[test]
    fn evicts_at_capacity_and_recompiles() {
        let cache = RegexCache::new(2);
        let first = cache.get_or_compile("first").unwrap();
        cache.get_or_compile("second").unwrap();
        cache.get_or_compile("third").unwrap();

        let entries = cache.read();
        assert_eq!(entries.values.len(), 2);
        assert!(!entries.values.contains_key("first"));
        drop(entries);

        let recompiled = cache.get_or_compile("first").unwrap();
        assert!(!Arc::ptr_eq(&first, &recompiled));
    }

    #[test]
    fn zero_capacity_compiles_without_storing() {
        let cache = RegexCache::new(0);
        let first = cache.get_or_compile("x").unwrap();
        let second = cache.get_or_compile("x").unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(cache.read().values.is_empty());
    }

    #[test]
    fn invalid_pattern_errors_without_consuming_capacity() {
        let cache = RegexCache::new(1);
        assert!(cache.get_or_compile(r"(?P<bad").is_err());
        assert!(cache.read().values.is_empty());
    }

    #[test]
    fn inline_flags_work() {
        let cache = RegexCache::new(1);
        let re = cache.get_or_compile(r"(?i)hello").unwrap();
        assert!(re.is_match("HELLO"));
        assert!(re.is_match("Hello"));
    }

    #[test]
    fn concurrent_misses_publish_one_cached_value() {
        let cache = Arc::new(RegexCache::new(4));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || cache.get_or_compile("concurrent").unwrap())
            })
            .collect();
        let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        let cached = cache.get_or_compile("concurrent").unwrap();
        assert!(values.iter().any(|value| Arc::ptr_eq(value, &cached)));
        assert_eq!(cache.read().values.len(), 1);
    }

    #[test]
    fn poisoned_lock_is_recovered() {
        let cache = Arc::new(RegexCache::new(2));
        let poisoner = Arc::clone(&cache);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.entries.write().unwrap();
            panic!("poison cache lock");
        })
        .join();

        assert!(cache
            .get_or_compile("after-poison")
            .unwrap()
            .is_match("after-poison"));
    }

    #[test]
    fn inconsistent_eviction_order_cannot_break_capacity() {
        let cache = RegexCache::new(1);
        cache.get_or_compile("first").unwrap();
        cache.write().insertion_order.clear();
        cache.get_or_compile("second").unwrap();
        assert_eq!(cache.read().values.len(), 1);
    }
}
