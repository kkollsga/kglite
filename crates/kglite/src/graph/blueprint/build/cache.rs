//! Whole-table cache shared by the serial build phases.

use super::super::input::InputRegistry;
use super::super::table::RawCsv;
use std::collections::HashMap;

/// Cache of whole-read tables keyed by input name. Populated in parallel at
/// the start of the build (see `parse_in_parallel`) so serial phases that read
/// the same input (node load + FK edges) never block on I/O. Junction edges
/// bypass it entirely — see `load_junction_edges`.
///
/// A failed read is cached as the failure. Every phase reads a given input
/// through here, so re-attempting it would parse a broken input once per
/// consumer and report the same error several times — and for an input that
/// is expensive to read, the retry costs as much as the successful path.
#[derive(Default)]
pub(super) struct CsvCache {
    inner: std::sync::Mutex<HashMap<String, Result<std::sync::Arc<RawCsv>, String>>>,
}

impl CsvCache {
    pub(super) fn get(
        &self,
        registry: &InputRegistry,
        name: &str,
    ) -> Result<std::sync::Arc<RawCsv>, String> {
        {
            let guard = self.inner.lock().unwrap();
            if let Some(hit) = guard.get(name) {
                return hit.clone();
            }
        }
        let result = registry
            .get(name)
            .and_then(|source| source.read_all())
            .map(std::sync::Arc::new);
        self.inner
            .lock()
            .unwrap()
            .insert(name.to_string(), result.clone());
        result
    }
}

/// Read all given inputs in parallel, populating the cache.
///
/// Both outcomes are stored: the phase that consumes an input gets the same
/// `Err` this pass produced, with the same message, and reports it against
/// the spec that owns the input.
pub(super) fn parse_in_parallel(names: &[String], registry: &InputRegistry, cache: &CsvCache) {
    use rayon::prelude::*;
    names.par_iter().for_each(|name| {
        let _ = cache.get(registry, name);
    });
}

#[cfg(test)]
mod cache_tests {
    use super::super::super::input::csv::CsvFile;
    use super::super::super::input::test_double::CountingSource;
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    /// A read that fails is reported from the cache, not re-attempted: the
    /// pre-pass and the serial phase together must open the input once.
    #[test]
    fn a_failed_read_is_cached_and_not_retried() {
        let (counting, opens) = CountingSource::new(Box::new(CsvFile::new(
            PathBuf::from("/nonexistent/definitely-not-here.csv"),
            "missing.csv".to_string(),
        )));
        let mut registry = InputRegistry::default();
        registry.insert("missing.csv", Box::new(counting));

        let cache = CsvCache::default();
        parse_in_parallel(&["missing.csv".to_string()], &registry, &cache);
        let err = cache
            .get(&registry, "missing.csv")
            .err()
            .expect("a file that is not there fails to read");

        assert!(err.starts_with("CSV open missing.csv: "), "{err}");
        assert_eq!(
            opens.load(Ordering::SeqCst),
            1,
            "the pre-pass read is the only one"
        );
    }
}
