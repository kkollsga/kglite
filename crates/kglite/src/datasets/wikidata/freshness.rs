//! Cache-freshness decision tree for Wikidata's disk-mode flow.
//!
//! Wikidata builds a graph from a multi-GB `.nt.bz2` dump that
//! takes minutes to load. Each `open()` call needs to decide:
//! "is the cached disk graph still fresh enough to reuse, or should
//! we re-fetch the dump and rebuild?"
//!
//! That decision is the same in every binding (Python, future Go,
//! JS, JVM): four input signals (graph age, source-mtime stamp,
//! remote-mtime probe, force-rebuild flag) collapse to one of a
//! small set of outcomes. Lifted out of the Python wrapper
//! (`kglite/datasets/wikidata.py::open`) so every binding shares
//! the same rules without re-implementing the comparisons.
//!
//! The binding still decides what to *do* with the outcome — print
//! progress, hit its process-local cache, etc. — those are
//! binding-specific ergonomics. The decision itself is core.

use chrono::{DateTime, Utc};
use std::path::Path;

/// What `open()` should do based on the freshness check. The
/// reason fields are human-readable so bindings can use them in
/// verbose prints without re-deriving the comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheDecision {
    /// Force-rebuild flag was set, or no cached graph exists yet.
    /// The binding should run the full fetch + build path.
    Build { reason: &'static str },
    /// Cached graph is fresh — `age_days < cooldown_days` — no
    /// remote probe needed. Load from cache.
    Load { reason: String },
    /// Cooldown elapsed and remote dump is newer than what the
    /// cached graph was built from. Rebuild.
    Rebuild { reason: String },
}

/// Inputs to the freshness check. Each one is computed by the
/// binding (or one of the existing core helpers); this struct
/// is just the carrier.
pub struct FreshnessInputs<'a> {
    /// Has the user explicitly asked to skip caches?
    pub force_rebuild: bool,
    /// `disk_graph_meta.json` path under the graph dir. If missing
    /// the graph isn't built yet → `Build`.
    pub graph_meta_path: &'a Path,
    /// `wikidata_source.json` path under the graph dir. May be
    /// missing on graphs built before source-meta stamping landed.
    pub source_meta_path: &'a Path,
    /// Cooldown window: graphs younger than this skip the remote
    /// probe entirely.
    pub cooldown_days: i64,
    /// `Last-Modified` header from a HEAD against the remote dump.
    /// `None` if the probe failed (the binding falls back to the
    /// cached graph as a graceful-degradation choice).
    pub remote_mtime: Option<DateTime<Utc>>,
}

/// Run the four-outcome decision tree:
///
/// 1. `force_rebuild = true` → `Build("force_rebuild")`
/// 2. `graph_meta_path` missing → `Build("no_cache")`
/// 3. `graph_age_days < cooldown_days` → `Load("within_cooldown")`
/// 4. `remote_mtime = None` → `Load("remote_unreachable")` (degrade gracefully)
/// 5. `embedded_mtime` missing or `remote_mtime > embedded_mtime` → `Rebuild("remote_newer")`
/// 6. Otherwise → `Load("remote_unchanged")`
pub fn decide(inputs: FreshnessInputs<'_>) -> CacheDecision {
    if inputs.force_rebuild {
        return CacheDecision::Build {
            reason: "force_rebuild",
        };
    }
    if !inputs.graph_meta_path.exists() {
        return CacheDecision::Build { reason: "no_cache" };
    }
    let graph_age = age_days(file_mtime_utc(inputs.graph_meta_path));
    if graph_age < inputs.cooldown_days as f64 {
        return CacheDecision::Load {
            reason: format!(
                "within_cooldown ({graph_age:.1}d < {}d)",
                inputs.cooldown_days
            ),
        };
    }
    let embedded_mtime = read_remote_mtime_from_source_meta(inputs.source_meta_path);
    let Some(remote_mtime) = inputs.remote_mtime else {
        return CacheDecision::Load {
            reason: format!(
                "remote_unreachable (cache built from {})",
                embedded_mtime.map_or_else(|| "unknown".to_string(), |m| m.to_rfc3339()),
            ),
        };
    };
    match embedded_mtime {
        Some(emb) if remote_mtime <= emb => CacheDecision::Load {
            reason: "remote_unchanged".to_string(),
        },
        Some(emb) => CacheDecision::Rebuild {
            reason: format!(
                "remote_newer (remote={} > embedded={})",
                remote_mtime.to_rfc3339(),
                emb.to_rfc3339()
            ),
        },
        None => CacheDecision::Rebuild {
            reason: "embedded_mtime_missing".to_string(),
        },
    }
}

/// Read `remote_last_modified_iso` (or fall back to
/// `source_mtime_iso`) from a `wikidata_source.json` payload.
/// Returns `None` if the file is missing, malformed, or has no
/// usable timestamp. Public so bindings can use it independently
/// of [`decide`].
pub fn read_remote_mtime_from_source_meta(path: &Path) -> Option<DateTime<Utc>> {
    let bytes = std::fs::read(path).ok()?;
    let data: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let iso = data
        .get("remote_last_modified_iso")
        .or_else(|| data.get("source_mtime_iso"))
        .and_then(|v| v.as_str())?;
    DateTime::parse_from_rfc3339(iso)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

/// File mtime in UTC, or `None` if the file is missing / unreadable.
pub fn file_mtime_utc(path: &Path) -> Option<DateTime<Utc>> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    Some(mtime.into())
}

/// Days between `when` and now (UTC). `None` → `+∞` (treat as
/// arbitrarily old) so the cooldown check trivially fails.
pub fn age_days(when: Option<DateTime<Utc>>) -> f64 {
    let Some(when) = when else {
        return f64::INFINITY;
    };
    (Utc::now() - when).num_milliseconds() as f64 / (86_400_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn make_meta(dir: &std::path::Path, name: &str, payload: Option<&str>) -> std::path::PathBuf {
        let p = dir.join(name);
        if let Some(body) = payload {
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(body.as_bytes()).unwrap();
        }
        p
    }

    #[test]
    fn force_rebuild_short_circuits() {
        let tmp = tempdir().unwrap();
        let graph = make_meta(tmp.path(), "graph.json", Some("{}"));
        let source = tmp.path().join("source.json");
        let d = decide(FreshnessInputs {
            force_rebuild: true,
            graph_meta_path: &graph,
            source_meta_path: &source,
            cooldown_days: 31,
            remote_mtime: None,
        });
        assert!(matches!(
            d,
            CacheDecision::Build {
                reason: "force_rebuild"
            }
        ));
    }

    #[test]
    fn no_cache_returns_build() {
        let tmp = tempdir().unwrap();
        let graph = tmp.path().join("missing.json"); // does NOT exist
        let source = tmp.path().join("source.json");
        let d = decide(FreshnessInputs {
            force_rebuild: false,
            graph_meta_path: &graph,
            source_meta_path: &source,
            cooldown_days: 31,
            remote_mtime: None,
        });
        assert!(matches!(d, CacheDecision::Build { reason: "no_cache" }));
    }

    #[test]
    fn within_cooldown_loads_without_remote_probe() {
        let tmp = tempdir().unwrap();
        let graph = make_meta(tmp.path(), "graph.json", Some("{}"));
        let source = tmp.path().join("source.json");
        let d = decide(FreshnessInputs {
            force_rebuild: false,
            graph_meta_path: &graph,
            source_meta_path: &source,
            cooldown_days: 31_000, // ridiculous cooldown → always within
            remote_mtime: None,
        });
        assert!(matches!(d, CacheDecision::Load { .. }));
    }

    #[test]
    fn cooldown_elapsed_remote_unreachable_loads_cache() {
        let tmp = tempdir().unwrap();
        let graph = make_meta(tmp.path(), "graph.json", Some("{}"));
        let source = make_meta(
            tmp.path(),
            "source.json",
            Some(r#"{"source_mtime_iso":"2024-01-01T00:00:00+00:00"}"#),
        );
        let d = decide(FreshnessInputs {
            force_rebuild: false,
            graph_meta_path: &graph,
            source_meta_path: &source,
            cooldown_days: 0,   // elapsed instantly
            remote_mtime: None, // unreachable
        });
        match d {
            CacheDecision::Load { reason } => assert!(reason.starts_with("remote_unreachable")),
            other => panic!("expected Load, got {other:?}"),
        }
    }

    #[test]
    fn cooldown_elapsed_remote_unchanged_loads_cache() {
        let tmp = tempdir().unwrap();
        let graph = make_meta(tmp.path(), "graph.json", Some("{}"));
        let source = make_meta(
            tmp.path(),
            "source.json",
            Some(r#"{"remote_last_modified_iso":"2030-01-01T00:00:00+00:00"}"#),
        );
        let remote_mtime = Some(
            DateTime::parse_from_rfc3339("2025-01-01T00:00:00+00:00")
                .unwrap()
                .with_timezone(&Utc),
        );
        let d = decide(FreshnessInputs {
            force_rebuild: false,
            graph_meta_path: &graph,
            source_meta_path: &source,
            cooldown_days: 0,
            remote_mtime,
        });
        assert!(matches!(d, CacheDecision::Load { .. }));
    }

    #[test]
    fn cooldown_elapsed_remote_newer_rebuilds() {
        let tmp = tempdir().unwrap();
        let graph = make_meta(tmp.path(), "graph.json", Some("{}"));
        let source = make_meta(
            tmp.path(),
            "source.json",
            Some(r#"{"remote_last_modified_iso":"2020-01-01T00:00:00+00:00"}"#),
        );
        let remote_mtime = Some(
            DateTime::parse_from_rfc3339("2030-01-01T00:00:00+00:00")
                .unwrap()
                .with_timezone(&Utc),
        );
        let d = decide(FreshnessInputs {
            force_rebuild: false,
            graph_meta_path: &graph,
            source_meta_path: &source,
            cooldown_days: 0,
            remote_mtime,
        });
        assert!(matches!(d, CacheDecision::Rebuild { .. }));
    }
}
