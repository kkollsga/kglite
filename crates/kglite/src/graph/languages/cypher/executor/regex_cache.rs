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

// ---------------------------------------------------------------------------
// Compile-failure messages, and recognising one again
// ---------------------------------------------------------------------------
//
// A pattern that does not compile is an error in the *query text*, not a
// property of the row being tested: it is wrong for every row and it can never
// become right. The unfused WHERE path has always raised it. The fused paths
// deliberately swallow predicate errors — a predicate that merely does not
// evaluate for a row (an unbound OPTIONAL MATCH binding, an aggregate
// reference in HAVING) drops the row rather than failing the query — so they
// need a way to tell the one class apart from the other before swallowing.
//
// Both messages are built here and recognised here, over shared constants, so
// the recogniser cannot drift away from the wording it recognises.

/// Opening words of the `=~` compile-failure message. Pinned by
/// `tests/test_regex_operator.py` and `tests/test_error_types.py`.
const OPERATOR_PREFIX: &str = "Invalid regular expression ";

/// Middle of a regex *function*'s compile-failure message.
const FUNCTION_INFIX: &str = "() invalid pattern: ";

/// Message for a `=~` pattern that failed to compile.
pub(super) fn operator_compile_error(pattern: &str, err: &regex::Error) -> String {
    format!("{OPERATOR_PREFIX}'{pattern}': {err}")
}

/// Message for a regex scalar function's pattern that failed to compile.
/// `function` is the bare name, without parentheses.
pub(super) fn function_compile_error(function: &str, err: &regex::Error) -> String {
    format!("{function}{FUNCTION_INFIX}{err}")
}

/// True when `message` reports a regex the user supplied and the engine could
/// not compile — i.e. one of the two messages above.
///
/// The fused execution paths call this before swallowing a predicate error:
/// a flagged message propagates (matching the unfused path), anything else
/// keeps the historical "this row does not match" behaviour.
pub(super) fn is_compile_error(message: &str) -> bool {
    message.starts_with(OPERATOR_PREFIX) || message.contains(FUNCTION_INFIX)
}

static CACHE: LazyLock<RegexCache> = LazyLock::new(|| RegexCache::new(CACHE_CAPACITY));

// ---------------------------------------------------------------------------
// Anchoring — the `=~` operator's full-string rule
// ---------------------------------------------------------------------------
//
// openCypher's `=~` succeeds only when the pattern matches the *entire*
// subject; `text_match_regex()` is this project's documented *search*
// function and must keep its unanchored behaviour. The two therefore differ
// in the pattern they compile, not in the cache they use: the anchored form
// is a distinct pattern string and gets its own entry in [`CACHE`], so the
// shared cache stays coherent and one caller can never serve the other.

/// Wrap `pattern` so a match must span the whole subject.
///
/// The group is non-capturing and encloses the entire pattern, so a top-level
/// alternation anchors as a unit: `cat|dog` becomes `^(?:cat|dog)$`, never
/// `^cat|dog$` — which would mean "starts with cat, or ends with dog".
///
/// Mirrored deliberately across the crate boundary by the fluent `{'=~': …}`
/// operator (`kglite-py/src/datatypes/py_in.rs`); the two must build the same
/// pattern.
fn anchor(pattern: &str) -> String {
    format!("^(?:{pattern})$")
}

/// Compile the anchored form of `pattern`, reporting a failure against the
/// pattern the *user* wrote.
///
/// `regex::Error` embeds the offending source text, and the user never wrote
/// the `^(?:…)$` wrapper — a bare `[` must not be reported as `^(?:[)$`.
/// Every pattern the anchored form rejects is rejected bare as well, so the
/// re-compile below supplies the message; if the wrapper alone is at fault
/// (reachable only through the compiled-size limit), its own error stands.
fn compile_anchored(pattern: &str) -> Result<Arc<Regex>, regex::Error> {
    CACHE.get_or_compile(&anchor(pattern)).map_err(|wrapped| {
        RegexBuilder::new(pattern)
            .size_limit(REGEX_SIZE_LIMIT)
            .build()
            .err()
            .unwrap_or(wrapped)
    })
}

/// Per-thread front cache in front of [`CACHE`].
///
/// A `=~` predicate resolves its pattern **once per row**, and the shared
/// cache answers a hit under an `RwLock` read — one atomic read-modify-write
/// on a single cache line. That is cheap on one thread and catastrophic on
/// ten: with the parallel runtime enabled, an 800k-row `=~` scan measured
/// **6.3x slower than sequential** (49 ms → 305 ms, release, 10-core M4)
/// purely from threads queuing on that line, while the identical query
/// written with `CONTAINS` — same per-row work, no shared lookup — gained
/// 6.5x. The front cache removes the shared access from the steady state
/// entirely: a repeated pattern is a short slice scan of `&str` comparisons
/// and an `Arc` clone, with no atomics beyond the refcount.
///
/// Safe to key per thread because the entries are immutable compiled
/// programs: which thread compiled one cannot change what it matches. Bounded
/// like its parent so a stream of unique patterns cannot grow per-thread
/// memory; the shared cache remains the backing store and the eviction
/// authority.
///
/// Entries are the **anchored** programs, keyed by the pattern the user
/// wrote — keying by the anchored form would build a string on every row,
/// which is the allocation this cache exists to avoid. `=~` is the only
/// per-row caller; an unanchored per-row caller would need its own cache,
/// not this one.
const LOCAL_CAPACITY: usize = 8;

thread_local! {
    static LOCAL_ANCHORED: std::cell::RefCell<Vec<(String, Regex)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Populate the per-thread front cache for `pattern` if it is not already
/// there.
fn ensure_local(pattern: &str) -> Result<(), regex::Error> {
    if LOCAL_ANCHORED.with(|local| local.borrow().iter().any(|(cached, _)| cached == pattern)) {
        return Ok(());
    }
    // Cloned, deliberately: a `Regex` carries an internal pool of scratch
    // space, and threads that share one `Regex` serialise on it. Cloning gives
    // each thread its own pool over the same automaton, which is what turns
    // the parallel `=~` scan from a regression into a win — see the module
    // note above. The clone happens once per thread per pattern; the shared
    // cache still does the compiling.
    let compiled = compile_anchored(pattern)?;
    LOCAL_ANCHORED.with(|local| {
        let mut local = local.borrow_mut();
        if local.len() >= LOCAL_CAPACITY {
            local.remove(0);
        }
        local.push((pattern.to_owned(), (*compiled).clone()));
    });
    Ok(())
}

/// Run `f` against the compiled **anchored** form of `pattern`, borrowing it
/// from the per-thread cache.
///
/// The subject must match `pattern` in full — this is `=~`'s entry point and
/// its semantics; see [`anchor`]. Callers that want a search compile through
/// [`get_or_compile`] instead.
///
/// This is the per-row entry point, and the borrow is the point of it.
/// [`get_or_compile`] hands back an `Arc`, and cloning one is an atomic
/// increment on a refcount every thread shares — which is the same contended
/// cache line the front cache was added to escape, just moved. Removing the
/// clone took the parallel `=~` scan from 0.48x to parity-and-above; keeping
/// it capped the win at half of sequential.
///
/// `f` must not re-enter this module: the thread-local is borrowed across the
/// call, so a nested lookup would panic. Every caller passes a leaf operation
/// (`is_match`, `replace_all`), which is why the borrow is safe to hold.
pub fn with_compiled_anchored<R>(
    pattern: &str,
    f: impl FnOnce(&Regex) -> R,
) -> Result<R, regex::Error> {
    ensure_local(pattern)?;
    Ok(LOCAL_ANCHORED.with(|local| {
        let local = local.borrow();
        let compiled = local
            .iter()
            .find(|(cached, _)| cached == pattern)
            .map(|(_, compiled)| compiled)
            .expect("invariant: ensure_local just inserted this pattern");
        f(compiled)
    }))
}

/// Look up `pattern` verbatim in the process-wide cache; compile and insert
/// on miss. Unanchored: this is the *search* entry point, used by
/// `text_match_regex()` and friends.
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
    fn compile_error_messages_are_recognised() {
        // Bound, not inlined: `clippy::invalid_regex` rejects a literal bad
        // pattern at `Regex::new`, and this test needs one.
        let bad = String::from("[");
        let err = Regex::new(&bad).expect_err("'[' must not compile");
        let operator = operator_compile_error(&bad, &err);
        let function = function_compile_error("text_match_regex", &err);
        assert!(operator.starts_with("Invalid regular expression '['"));
        assert!(function.starts_with("text_match_regex() invalid pattern: "));
        assert!(is_compile_error(&operator));
        assert!(is_compile_error(&function));
    }

    #[test]
    fn other_evaluation_errors_are_not_compile_errors() {
        // The messages the fused paths must keep swallowing.
        assert!(!is_compile_error("Missing parameter: $min"));
        assert!(!is_compile_error(
            "Cannot evaluate aggregate function in this context"
        ));
        assert!(!is_compile_error("Variable 'x' not bound"));
        assert!(!is_compile_error(""));
    }

    #[test]
    fn anchored_matches_the_whole_subject_only() {
        // The `=~` contract: a pattern that only occurs *inside* the subject
        // does not match. `'inactive' =~ 'active'` was true before 0.16.6.
        assert!(with_compiled_anchored("active", |re| re.is_match("active")).unwrap());
        assert!(!with_compiled_anchored("active", |re| re.is_match("inactive")).unwrap());
        assert!(!with_compiled_anchored("b", |re| re.is_match("abc")).unwrap());
        // An explicitly anchored pattern is unaffected by the wrapper.
        assert!(with_compiled_anchored("^A.*", |re| re.is_match("Alice")).unwrap());
    }

    #[test]
    fn anchored_alternation_binds_as_a_unit() {
        // `^cat|dog$` would mean "starts with cat, or ends with dog"; the
        // non-capturing group is what makes `^(?:cat|dog)$` correct.
        assert!(with_compiled_anchored("cat|dog", |re| re.is_match("cat")).unwrap());
        assert!(with_compiled_anchored("cat|dog", |re| re.is_match("dog")).unwrap());
        assert!(!with_compiled_anchored("cat|dog", |re| re.is_match("catx")).unwrap());
        assert!(!with_compiled_anchored("cat|dog", |re| re.is_match("xdog")).unwrap());
    }

    #[test]
    fn anchored_keeps_inline_flags() {
        assert!(with_compiled_anchored("(?i)active", |re| re.is_match("ACTIVE")).unwrap());
        assert!(!with_compiled_anchored("active", |re| re.is_match("ACTIVE")).unwrap());
    }

    #[test]
    fn anchored_compile_error_names_the_users_pattern() {
        // The wrapper is ours; the error text must not show it.
        let bad = String::from("[");
        let err = with_compiled_anchored(&bad, |_| ()).expect_err("'[' must not compile");
        let message = operator_compile_error(&bad, &err);
        assert!(
            message.starts_with("Invalid regular expression '['"),
            "{message}"
        );
        assert!(!message.contains("^(?:"), "{message}");
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
