//! Whole-table cache shared by the serial build phases.

use super::super::input::InputRegistry;
use super::super::table::RawCsv;
use std::collections::HashMap;

/// Cache of whole-read tables keyed by input name. Populated in parallel at
/// the start of the build (see `parse_in_parallel`) so serial phases that read
/// the same input (node load + FK edges) never block on I/O. Junction edges
/// bypass it entirely — see `load_junction_edges`.
#[derive(Default)]
pub(super) struct CsvCache {
    inner: std::sync::Mutex<HashMap<String, std::sync::Arc<RawCsv>>>,
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
                return Ok(hit.clone());
            }
        }
        let raw = registry.get(name)?.read_all()?;
        let arc = std::sync::Arc::new(raw);
        self.inner
            .lock()
            .unwrap()
            .insert(name.to_string(), arc.clone());
        Ok(arc)
    }

    fn insert(&self, name: &str, raw: RawCsv) {
        self.inner
            .lock()
            .unwrap()
            .insert(name.to_string(), std::sync::Arc::new(raw));
    }
}

/// Read all given inputs in parallel, populating the cache. Failures
/// are silently skipped — the caller will see the `Err` again when it tries
/// to look up that name serially (and can emit a targeted error then).
pub(super) fn parse_in_parallel(names: &[String], registry: &InputRegistry, cache: &CsvCache) {
    use rayon::prelude::*;
    names.par_iter().for_each(|name| {
        if let Ok(raw) = registry.get(name).and_then(|s| s.read_all()) {
            cache.insert(name, raw);
        }
    });
}
