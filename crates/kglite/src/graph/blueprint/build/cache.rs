//! Whole-file CSV cache shared by the serial build phases.

use super::super::table::{read_csv_raw, RawCsv};
use std::collections::HashMap;
use std::path::Path;

/// Cache of raw CSVs keyed by relative path. Populated in parallel at the
/// start of the build (see `parse_in_parallel`) so serial phases that read
/// the same CSV (node load + FK edges) never block on disk. Junction edges
/// bypass it entirely — see `load_junction_edges`.
#[derive(Default)]
pub(super) struct CsvCache {
    inner: std::sync::Mutex<HashMap<String, std::sync::Arc<RawCsv>>>,
}

impl CsvCache {
    pub(super) fn get(&self, root: &Path, rel: &str) -> Result<std::sync::Arc<RawCsv>, String> {
        {
            let guard = self.inner.lock().unwrap();
            if let Some(hit) = guard.get(rel) {
                return Ok(hit.clone());
            }
        }
        let full = root.join(rel);
        let raw = read_csv_raw(&full)?;
        let arc = std::sync::Arc::new(raw);
        self.inner
            .lock()
            .unwrap()
            .insert(rel.to_string(), arc.clone());
        Ok(arc)
    }

    fn insert(&self, rel: &str, raw: RawCsv) {
        self.inner
            .lock()
            .unwrap()
            .insert(rel.to_string(), std::sync::Arc::new(raw));
    }
}

/// Parse all given CSV paths in parallel, populating the cache. Failures
/// are silently skipped — the caller will see the `Err` again when it tries
/// to look up that path serially (and can emit a targeted error then).
pub(super) fn parse_in_parallel(paths: &[String], root: &Path, cache: &CsvCache) {
    use rayon::prelude::*;
    paths.par_iter().for_each(|rel| {
        let full = root.join(rel);
        if let Ok(raw) = read_csv_raw(&full) {
            cache.insert(rel, raw);
        }
    });
}
