//! `sodir_index.json` — the per-dataset fetch manifest — and the
//! two-tier cooldown decision logic.
//!
//! Two independent cooldowns gate refetching (ported from the Python
//! `wrapper.py`):
//!
//! - **index cooldown** — once `index_cooldown_days` elapse since the
//!   last full *sweep*, every dataset gets a cheap `count` probe.
//! - **dataset cooldown** — a dataset whose `fetched_at_iso` is older
//!   than `dataset_cooldown_days` is hard-refetched regardless.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::datasets::sodir::error::{Result, SodirError};
use crate::datasets::sodir::layout::Workdir;

/// One dataset's entry in `sodir_index.json`. `layer_id`, `base_url`
/// and `fetch_duration_secs` are absent for user-supplied datasets
/// (no REST catalog entry), so they serialise as omitted, not null.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    pub csv_path: String,
    pub row_count: u64,
    pub fetched_at_iso: String,
    pub count_checked_at_iso: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fetch_duration_secs: Option<f64>,
}

impl DatasetEntry {
    /// Build an entry for a freshly fetched catalog dataset.
    pub fn fetched(
        kind: &str,
        layer_id: u32,
        base_url: &str,
        stem: &str,
        row_count: u64,
        duration_secs: f64,
        now: &str,
    ) -> Self {
        Self {
            kind: kind.to_string(),
            layer_id: Some(layer_id),
            base_url: Some(base_url.to_string()),
            csv_path: format!("csv/{stem}.csv"),
            row_count,
            fetched_at_iso: now.to_string(),
            count_checked_at_iso: now.to_string(),
            fetch_duration_secs: Some((duration_secs * 100.0).round() / 100.0),
        }
    }

    /// Build an entry for a user-supplied CSV (no REST catalog entry).
    pub fn user_supplied(stem: &str, row_count: u64, now: &str) -> Self {
        Self {
            kind: "user_supplied".to_string(),
            layer_id: None,
            base_url: None,
            csv_path: format!("csv/{stem}.csv"),
            row_count,
            fetched_at_iso: now.to_string(),
            count_checked_at_iso: now.to_string(),
            fetch_duration_secs: None,
        }
    }
}

/// The whole `sodir_index.json` document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SodirIndex {
    pub schema_version: u32,
    pub endpoint: String,
    pub last_full_check_iso: Option<String>,
    /// `BTreeMap` so the on-disk JSON has deterministically ordered
    /// dataset keys.
    pub datasets: BTreeMap<String, DatasetEntry>,
}

impl Default for SodirIndex {
    fn default() -> Self {
        Self {
            schema_version: 1,
            endpoint: "https://factmaps.sodir.no/api/rest/services/DataService".to_string(),
            last_full_check_iso: None,
            datasets: BTreeMap::new(),
        }
    }
}

/// Load `sodir_index.json`, or a fresh empty index if absent.
pub fn load(workdir: &Workdir) -> Result<SodirIndex> {
    let path = workdir.index_file();
    if !path.is_file() {
        return Ok(SodirIndex::default());
    }
    let text = std::fs::read_to_string(&path)?;
    serde_json::from_str(&text).map_err(|e| SodirError::Decode(format!("sodir_index.json: {e}")))
}

/// Write `sodir_index.json`. Small (~150 entries); written after each
/// dataset completes so a Ctrl-C never loses progress.
pub fn save(workdir: &Workdir, index: &SodirIndex) -> Result<()> {
    let text = serde_json::to_string_pretty(index)
        .map_err(|e| SodirError::Decode(format!("serialize index: {e}")))?;
    std::fs::write(workdir.index_file(), text)?;
    Ok(())
}

/// Current UTC time as an RFC-3339 string.
pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Age of an RFC-3339 timestamp in days. Unparseable → `INFINITY`
/// (treated as "very stale" so the caller refetches).
pub fn age_days_iso(iso: &str) -> f64 {
    match chrono::DateTime::parse_from_rfc3339(iso) {
        Ok(dt) => {
            let delta = chrono::Utc::now().signed_duration_since(dt.with_timezone(&chrono::Utc));
            delta.num_seconds() as f64 / 86_400.0
        }
        Err(_) => f64::INFINITY,
    }
}

/// Age of a file's mtime in days, or `None` if it doesn't exist.
/// Drives the disk-mode "reopen → load, don't rebuild" short-circuit.
pub fn file_mtime_age_days(path: &Path) -> Option<f64> {
    let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
    let age = std::time::SystemTime::now().duration_since(mtime).ok()?;
    Some(age.as_secs_f64() / 86_400.0)
}

/// Header-excluded line count of a CSV — used to size user-supplied
/// datasets that have no REST `count` probe.
pub fn quick_row_count(csv_path: &Path) -> u64 {
    match std::fs::read_to_string(csv_path) {
        Ok(content) => (content.lines().count().saturating_sub(1)) as u64,
        Err(_) => 0,
    }
}

/// True if a full cooldown sweep is due — no prior sweep, or the last
/// one is older than `index_cooldown_days`.
pub fn sweep_due(last_full_check: Option<&str>, index_cooldown_days: i64) -> bool {
    match last_full_check {
        None => true,
        Some(iso) => age_days_iso(iso) >= index_cooldown_days as f64,
    }
}

/// What to do with one dataset this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Keep the existing CSV untouched.
    Skip,
    /// Cheap `count` probe; upgrades to `Fetch` if the count drifted.
    Probe,
    /// Download the dataset fresh.
    Fetch,
}

/// Decide what to do with one dataset, given its index entry and the
/// on-disk CSV. Precedence (ported from `_decide_action`): no entry or
/// missing/corrupt CSV → Fetch; past the hard dataset cooldown →
/// Fetch; sweep due → Probe; else Skip.
pub fn decide_action(
    entry: Option<&DatasetEntry>,
    csv_path: &Path,
    sweep_due: bool,
    dataset_cooldown_days: i64,
) -> Action {
    let Some(entry) = entry else {
        return Action::Fetch;
    };
    if !csv_path.is_file() {
        return Action::Fetch;
    }
    // A CSV under 5 bytes can't even hold a header — treat as corrupt.
    if std::fs::metadata(csv_path)
        .map(|m| m.len() < 5)
        .unwrap_or(true)
    {
        return Action::Fetch;
    }
    if age_days_iso(&entry.fetched_at_iso) >= dataset_cooldown_days as f64 {
        return Action::Fetch;
    }
    if sweep_due {
        return Action::Probe;
    }
    Action::Skip
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(fetched_at: &str) -> DatasetEntry {
        DatasetEntry {
            kind: "layer".to_string(),
            layer_id: Some(7100),
            base_url: Some("http://x".to_string()),
            csv_path: "csv/field.csv".to_string(),
            row_count: 87,
            fetched_at_iso: fetched_at.to_string(),
            count_checked_at_iso: fetched_at.to_string(),
            fetch_duration_secs: Some(1.0),
        }
    }

    #[test]
    fn index_json_roundtrips() {
        let mut idx = SodirIndex::default();
        idx.datasets.insert("field".to_string(), entry(&now_iso()));
        let json = serde_json::to_string(&idx).unwrap();
        let back: SodirIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, 1);
        assert!(back.datasets.contains_key("field"));
    }

    #[test]
    fn user_supplied_entry_omits_catalog_fields() {
        let e = DatasetEntry::user_supplied("custom", 10, &now_iso());
        let json = serde_json::to_string(&e).unwrap();
        assert!(!json.contains("layer_id"));
        assert!(!json.contains("base_url"));
        assert!(!json.contains("fetch_duration_secs"));
    }

    #[test]
    fn decide_action_truth_table() {
        let tmp = tempfile::tempdir().unwrap();
        let csv = tmp.path().join("field.csv");
        std::fs::write(&csv, "a,b\n1,2\n").unwrap();
        let fresh = now_iso();

        // No entry → Fetch.
        assert_eq!(decide_action(None, &csv, false, 30), Action::Fetch);
        // Missing CSV → Fetch.
        let missing = tmp.path().join("nope.csv");
        assert_eq!(
            decide_action(Some(&entry(&fresh)), &missing, false, 30),
            Action::Fetch
        );
        // Fresh entry, no sweep → Skip.
        assert_eq!(
            decide_action(Some(&entry(&fresh)), &csv, false, 30),
            Action::Skip
        );
        // Fresh entry, sweep due → Probe.
        assert_eq!(
            decide_action(Some(&entry(&fresh)), &csv, true, 30),
            Action::Probe
        );
        // Stale entry → Fetch even without a sweep.
        let stale = "2000-01-01T00:00:00+00:00";
        assert_eq!(
            decide_action(Some(&entry(stale)), &csv, false, 30),
            Action::Fetch
        );
    }

    #[test]
    fn corrupt_tiny_csv_forces_fetch() {
        let tmp = tempfile::tempdir().unwrap();
        let csv = tmp.path().join("field.csv");
        std::fs::write(&csv, "\n").unwrap(); // < 5 bytes
        assert_eq!(
            decide_action(Some(&entry(&now_iso())), &csv, false, 30),
            Action::Fetch
        );
    }

    #[test]
    fn sweep_due_logic() {
        assert!(sweep_due(None, 14));
        assert!(sweep_due(Some("2000-01-01T00:00:00+00:00"), 14));
        assert!(!sweep_due(Some(&now_iso()), 14));
    }
}
