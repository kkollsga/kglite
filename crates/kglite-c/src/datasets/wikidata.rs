//! Wikidata C ABI — resumable RDF dump fetch + cache-freshness
//! decision tree.
//!
//! Three entry points cover the typical binding workflow:
//!
//! 1. `kglite_datasets_wikidata_ensure_dump` — pulls the
//!    `latest-truthy.nt.bz2` to `<workdir>/cache/`, resumable, with
//!    a cooldown window before re-fetching.
//! 2. `kglite_datasets_wikidata_remote_last_modified` — sync HEAD
//!    probe returning the dump's `Last-Modified` header as an ISO
//!    8601 string (or null if the probe failed).
//! 3. `kglite_datasets_wikidata_decide_cache` — pure-CPU
//!    freshness decision tree (Build / Load / Rebuild) that every
//!    binding's `open()` flow asks the same questions through.
//!
//! The dump-to-graph build step (parse the .nt.bz2 → DirGraph) is
//! a separate concern — the wheel uses
//! `KnowledgeGraph::load_ntriples`; future bindings can reach it
//! via the standard graph-load path once the file is on disk.

use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::datasets::wikidata::{
    decide, ensure_dump, remote_last_modified, CacheDecision, FreshnessInputs, Workdir,
};
use std::ffi::{c_char, CStr};
use std::path::Path;

/// Ensure the Wikidata `latest-truthy.nt.bz2` dump is present
/// under `<workdir>/cache/`. Resumable: a partially-downloaded
/// file gets continued via HTTP `Range` requests. Cooldown:
/// fully-present files within `cooldown_days` of their mtime
/// skip the re-fetch entirely.
///
/// # Arguments
///
/// - `workdir_path` (in, borrowed): root for the workdir layout.
/// - `cooldown_days` (in): how stale the cached dump can be before
///   re-fetching. Wheel default: 7.
/// - `verbose` (in): non-zero turns on the engine's progress
///   logging.
/// - `out_dump_path` (out, owned): JSON-encoded path string to the
///   downloaded dump (e.g. `"\"<workdir>/cache/latest-truthy.nt.bz2\""`).
///   We JSON-encode the path because filesystem paths can contain
///   characters that need escaping in some downstream consumers.
/// - `out_remote_mtime_iso` (out, owned, may be null): if the
///   server returned a `Last-Modified` header, the ISO 8601
///   timestamp string. Set to null if the probe failed.
/// - `out_error_msg` (out, owned, may be null).
///
/// # Safety
///
/// `workdir_path` must be null-terminated UTF-8. The out-pointers
/// must be valid writable slots.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_wikidata_ensure_dump(
    workdir_path: *const c_char,
    cooldown_days: i64,
    verbose: u8,
    out_dump_path: *mut *const c_char,
    out_remote_mtime_iso: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    if workdir_path.is_null() || out_dump_path.is_null() {
        return KgliteStatusCode::NullPointer;
    }
    let workdir_str = match unsafe { CStr::from_ptr(workdir_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return KgliteStatusCode::InvalidUtf8,
    };
    let workdir = Workdir::new(workdir_str);

    match ensure_dump(&workdir, cooldown_days, verbose != 0) {
        Ok((dump_path, remote_mtime)) => {
            let path_str = dump_path.to_string_lossy().to_string();
            unsafe {
                *out_dump_path = alloc_c_string(&path_str);
            }
            if !out_remote_mtime_iso.is_null() {
                unsafe {
                    *out_remote_mtime_iso = match remote_mtime {
                        Some(dt) => alloc_c_string(&dt.to_rfc3339()),
                        None => std::ptr::null(),
                    };
                }
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = std::ptr::null();
                }
            }
            KgliteStatusCode::Ok
        }
        Err(err) => {
            unsafe {
                *out_dump_path = std::ptr::null();
            }
            if !out_remote_mtime_iso.is_null() {
                unsafe {
                    *out_remote_mtime_iso = std::ptr::null();
                }
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = alloc_c_string(&err.to_string());
                }
            }
            KgliteStatusCode::Internal
        }
    }
}

/// Sync HEAD-request probe for the dump's `Last-Modified` header.
/// Returns the timestamp as RFC 3339 / ISO 8601 string, or null if
/// the probe failed (network down, server returned no header,
/// etc.).
///
/// # Arguments
///
/// - `out_iso` (out, owned, may be null): ISO 8601 string on
///   success; null on probe failure. Caller frees via
///   [`kglite_free_string`](crate::kglite_free_string).
///
/// # Safety
///
/// `out_iso` must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_wikidata_remote_last_modified(
    out_iso: *mut *const c_char,
) -> KgliteStatusCode {
    if out_iso.is_null() {
        return KgliteStatusCode::NullPointer;
    }
    match remote_last_modified() {
        Some(dt) => unsafe {
            *out_iso = alloc_c_string(&dt.to_rfc3339());
        },
        None => unsafe {
            *out_iso = std::ptr::null();
        },
    }
    KgliteStatusCode::Ok
}

/// Run the cache-freshness decision tree. Pure-CPU; the only I/O
/// is the file-mtime stat that the engine's `decide()` performs
/// internally on the two meta-paths. Returns a JSON object:
///
/// ```json
/// {"decision": "build" | "load" | "rebuild", "reason": "..." }
/// ```
///
/// # Arguments
///
/// - `force_rebuild` (in): non-zero → always `Build("force_rebuild")`.
/// - `graph_meta_path` (in, borrowed): path to
///   `<graph_dir>/disk_graph_meta.json`. Missing file →
///   `Build("no_cache")`.
/// - `source_meta_path` (in, borrowed): path to
///   `<graph_dir>/wikidata_source.json`. May be missing on graphs
///   built before source-meta stamping landed.
/// - `cooldown_days` (in): graphs younger than this skip the
///   remote probe.
/// - `remote_mtime_iso` (in, borrowed, may be null): RFC 3339 /
///   ISO 8601 timestamp from a prior call to
///   [`kglite_datasets_wikidata_remote_last_modified`]. Null →
///   probe was skipped or failed.
/// - `out_decision_json` (out, owned).
/// - `out_error_msg` (out, owned, may be null).
///
/// # Safety
///
/// All input strings must be null-terminated UTF-8.
/// `out_decision_json` must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_wikidata_decide_cache(
    force_rebuild: u8,
    graph_meta_path: *const c_char,
    source_meta_path: *const c_char,
    cooldown_days: i64,
    remote_mtime_iso: *const c_char,
    out_decision_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    if graph_meta_path.is_null() || source_meta_path.is_null() || out_decision_json.is_null() {
        return KgliteStatusCode::NullPointer;
    }
    let graph_meta_str = match unsafe { CStr::from_ptr(graph_meta_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return KgliteStatusCode::InvalidUtf8,
    };
    let source_meta_str = match unsafe { CStr::from_ptr(source_meta_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return KgliteStatusCode::InvalidUtf8,
    };

    // Optional remote mtime — parse the RFC 3339 string if given.
    let remote_mtime = if remote_mtime_iso.is_null() {
        None
    } else {
        let s = match unsafe { CStr::from_ptr(remote_mtime_iso) }.to_str() {
            Ok(s) => s,
            Err(_) => return KgliteStatusCode::InvalidUtf8,
        };
        match chrono::DateTime::parse_from_rfc3339(s) {
            Ok(dt) => Some(dt.with_timezone(&chrono::Utc)),
            Err(_) => return KgliteStatusCode::InvalidArgument,
        }
    };

    let inputs = FreshnessInputs {
        force_rebuild: force_rebuild != 0,
        graph_meta_path: Path::new(graph_meta_str),
        source_meta_path: Path::new(source_meta_str),
        cooldown_days,
        remote_mtime,
    };
    let decision = decide(inputs);
    let json = serialize_decision(&decision);
    unsafe {
        *out_decision_json = alloc_c_string(&json);
    }
    if !out_error_msg.is_null() {
        unsafe {
            *out_error_msg = std::ptr::null();
        }
    }
    KgliteStatusCode::Ok
}

fn serialize_decision(decision: &CacheDecision) -> String {
    match decision {
        CacheDecision::Build { reason } => serde_json::json!({
            "decision": "build",
            "reason": reason,
        })
        .to_string(),
        CacheDecision::Load { reason } => serde_json::json!({
            "decision": "load",
            "reason": reason,
        })
        .to_string(),
        CacheDecision::Rebuild { reason } => serde_json::json!({
            "decision": "rebuild",
            "reason": reason,
        })
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn decide_cache_force_rebuild_returns_build() {
        let graph_meta = CString::new("/tmp/does_not_exist/disk_graph_meta.json").unwrap();
        let source_meta = CString::new("/tmp/does_not_exist/wikidata_source.json").unwrap();
        let mut out: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_wikidata_decide_cache(
                /*force_rebuild=*/ 1,
                graph_meta.as_ptr(),
                source_meta.as_ptr(),
                7,
                std::ptr::null(),
                &mut out as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        let s = unsafe { CStr::from_ptr(out).to_str().unwrap() };
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert_eq!(parsed["decision"], "build");
        assert_eq!(parsed["reason"], "force_rebuild");
        unsafe { crate::kglite_free_string(out) };
    }

    #[test]
    fn decide_cache_missing_meta_returns_build_no_cache() {
        let graph_meta = CString::new("/tmp/kglite_c_wikidata_test_missing").unwrap();
        let source_meta = CString::new("/tmp/kglite_c_wikidata_test_missing_src").unwrap();
        let mut out: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_wikidata_decide_cache(
                0,
                graph_meta.as_ptr(),
                source_meta.as_ptr(),
                7,
                std::ptr::null(),
                &mut out as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        let s = unsafe { CStr::from_ptr(out).to_str().unwrap() };
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert_eq!(parsed["decision"], "build");
        assert_eq!(parsed["reason"], "no_cache");
        unsafe { crate::kglite_free_string(out) };
    }

    #[test]
    fn decide_cache_bad_remote_mtime_returns_invalid_argument() {
        let graph_meta = CString::new("/tmp/anywhere").unwrap();
        let source_meta = CString::new("/tmp/anywhere").unwrap();
        let bad_iso = CString::new("not-a-timestamp").unwrap();
        let mut out: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_wikidata_decide_cache(
                0,
                graph_meta.as_ptr(),
                source_meta.as_ptr(),
                7,
                bad_iso.as_ptr(),
                &mut out as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::InvalidArgument);
    }
}
