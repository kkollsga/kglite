//! Dataset C ABI — synchronous wrappers over kglite's blocking
//! fetchers. Each dataset's API uses kglite's existing
//! `*_blocking` companion functions (Phase 5), so the C side
//! doesn't drag in tokio.
//!
//! Feature-gated: each dataset is enabled by the matching Cargo
//! feature on this crate (`sec`, `sodir`, `wikidata`). A consumer
//! that doesn't enable the feature simply won't see those C
//! functions in `kglite.h`.
//!
//! Pattern (mirrors the wheel's `_sodir_internal::refresh`):
//!   1. Convert C inputs → Rust args (CStr → str, parse JSON arrays,
//!      etc.)
//!   2. Build the dataset's workdir/client from primitive args
//!   3. Call the existing `_blocking` entry point
//!   4. On Ok: serialize the report into JSON string per the
//!      negative-space convention (wire format is a per-binding
//!      concern)
//!   5. On Err: set the error message string + return a status code
//!
//! Future H.3 sub-iterations add the SEC + Wikidata entry points
//! following the same pattern.

#[cfg(feature = "sodir")]
mod sodir_ffi {
    use crate::status::KgliteStatusCode;
    use crate::strings::alloc_c_string;
    use kglite::api::datasets::sodir::{fetch_all_blocking, FetchAllReport, Workdir};
    use std::ffi::{c_char, CStr};

    /// Fetch all Sodir (Norwegian Continental Shelf) datasets the
    /// caller asks for. Sync wrapper around the engine's
    /// [`fetch_all_blocking`] entry point — spins up a single-thread
    /// tokio runtime per call, fetches missing/stale CSVs from the
    /// ArcGIS FactMaps REST API, applies preprocessing (FK fixups),
    /// returns a report.
    ///
    /// # Arguments
    ///
    /// - `workdir_path` (in, borrowed): directory under which CSVs
    ///   land. Layout: `<root>/csv/<dataset>.csv`,
    ///   `<root>/index.json`. Created on first use; subsequent
    ///   calls reuse the layout.
    /// - `datasets_json` (in, borrowed): JSON array of dataset
    ///   stem names the caller wants — e.g.
    ///   `["field", "wellbore_exploration", "production_profile"]`.
    ///   Must not be null. Pass `"[]"` for no datasets.
    /// - `index_cooldown_days` (in): how long the workdir's
    ///   `index.json` is trusted before re-probing the catalog.
    ///   Wheel default: 7.
    /// - `dataset_cooldown_days` (in): how long an already-fetched
    ///   CSV is trusted before re-fetching. Wheel default: 30.
    /// - `concurrency` (in): max parallel HTTP fetches. Wheel
    ///   default: 10.
    /// - `out_report_json` (out, owned): on success, set to a
    ///   JSON object string with the report fields. Caller must
    ///   free via [`kglite_free_string`](crate::kglite_free_string).
    ///   Shape:
    ///   ```json
    ///   {
    ///     "refresh": {
    ///       "fetched": ["..."],
    ///       "unchanged": ["..."],
    ///       "user_supplied": ["..."],
    ///       "cached": ["..."],
    ///       "unfetchable": ["..."],
    ///       "errors": [["stem", "message"], ...]
    ///     },
    ///     "preprocess": {
    ///       "petreg_licence_pk": null | <int>,
    ///       "seismic_progress_fk": null | <int>,
    ///       "chrono_parent_fk": null | <int>,
    ///       "announced_block_fk": null | <int>
    ///     }
    ///   }
    ///   ```
    /// - `out_error_msg` (out, owned, may be null): on failure,
    ///   set to an owned error message string. Caller must free
    ///   via [`kglite_free_string`](crate::kglite_free_string).
    ///
    /// # Errors
    ///
    /// - `KGLITE_STATUS_CODE_NULL_POINTER` — required pointer is null
    /// - `KGLITE_STATUS_CODE_INVALID_UTF8` — input string isn't valid UTF-8
    /// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — `datasets_json` isn't a
    ///   JSON array of strings
    /// - `KGLITE_STATUS_CODE_FILE_IO` — workdir creation or CSV write failed
    /// - `KGLITE_STATUS_CODE_INTERNAL` — REST API call failed, tokio
    ///   runtime build failed, or other engine-level error
    ///
    /// # Safety
    ///
    /// `workdir_path` and `datasets_json` must be null-terminated
    /// UTF-8 strings. `out_report_json` must be a valid writable
    /// pointer to a `*const c_char` slot.
    #[no_mangle]
    pub unsafe extern "C" fn kglite_datasets_sodir_fetch_all(
        workdir_path: *const c_char,
        datasets_json: *const c_char,
        index_cooldown_days: i64,
        dataset_cooldown_days: i64,
        concurrency: usize,
        out_report_json: *mut *const c_char,
        out_error_msg: *mut *const c_char,
    ) -> KgliteStatusCode {
        if workdir_path.is_null() || datasets_json.is_null() || out_report_json.is_null() {
            return KgliteStatusCode::NullPointer;
        }
        let workdir_str = match unsafe { CStr::from_ptr(workdir_path) }.to_str() {
            Ok(s) => s,
            Err(_) => return KgliteStatusCode::InvalidUtf8,
        };
        let datasets_str = match unsafe { CStr::from_ptr(datasets_json) }.to_str() {
            Ok(s) => s,
            Err(_) => return KgliteStatusCode::InvalidUtf8,
        };
        let datasets: Vec<String> = match serde_json::from_str(datasets_str) {
            Ok(v) => v,
            Err(_) => return KgliteStatusCode::InvalidArgument,
        };

        let workdir = Workdir::new(workdir_str);
        match fetch_all_blocking(
            &workdir,
            &datasets,
            index_cooldown_days,
            dataset_cooldown_days,
            concurrency,
        ) {
            Ok(report) => {
                let json = serialize_fetch_all_report(&report);
                unsafe {
                    *out_report_json = alloc_c_string(&json);
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
                    *out_report_json = std::ptr::null();
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

    /// Hand-build the report JSON. We don't add `Serialize` derives
    /// to the core report types because wire format is a
    /// per-binding concern per CLAUDE.md's negative-space table —
    /// the wheel builds a PyDict here; bolt-server would build a
    /// BoltMap; the C ABI builds a JSON string. Each binding owns
    /// its own shape.
    fn serialize_fetch_all_report(report: &FetchAllReport) -> String {
        let refresh = &report.refresh;
        let preprocess = &report.preprocess;

        let errors_json = serde_json::Value::Array(
            refresh
                .errors
                .iter()
                .map(|(stem, msg)| {
                    serde_json::Value::Array(vec![
                        serde_json::Value::String(stem.clone()),
                        serde_json::Value::String(msg.clone()),
                    ])
                })
                .collect(),
        );

        let value = serde_json::json!({
            "refresh": {
                "fetched": refresh.fetched,
                "unchanged": refresh.unchanged,
                "user_supplied": refresh.user_supplied,
                "cached": refresh.cached,
                "unfetchable": refresh.unfetchable,
                "errors": errors_json,
            },
            "preprocess": {
                "petreg_licence_pk": preprocess.petreg_licence_pk,
                "seismic_progress_fk": preprocess.seismic_progress_fk,
                "chrono_parent_fk": preprocess.chrono_parent_fk,
                "announced_block_fk": preprocess.announced_block_fk,
            }
        });

        // to_string() is infallible for the value-tree above (every
        // sub-value is well-formed) but match for clippy.
        serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn serialize_default_report() {
            let report = FetchAllReport::default();
            let json = serialize_fetch_all_report(&report);
            // Round-trip via serde to assert structure.
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(parsed["refresh"].is_object());
            assert!(parsed["preprocess"].is_object());
            assert_eq!(parsed["refresh"]["fetched"], serde_json::json!([]));
            assert_eq!(
                parsed["preprocess"]["petreg_licence_pk"],
                serde_json::Value::Null
            );
        }

        #[test]
        fn null_pointer_returns_null_pointer_status() {
            let mut out_report: *const c_char = std::ptr::null();
            let mut out_err: *const c_char = std::ptr::null();
            let rc = unsafe {
                kglite_datasets_sodir_fetch_all(
                    std::ptr::null(),
                    std::ptr::null(),
                    7,
                    30,
                    10,
                    &mut out_report as *mut _,
                    &mut out_err as *mut _,
                )
            };
            assert_eq!(rc, KgliteStatusCode::NullPointer);
        }

        #[test]
        fn bad_datasets_json_returns_invalid_argument() {
            use std::ffi::CString;
            let workdir = CString::new("/tmp/kglite_c_sodir_bad_json").unwrap();
            let bad_json = CString::new("not-json").unwrap();
            let mut out_report: *const c_char = std::ptr::null();
            let mut out_err: *const c_char = std::ptr::null();
            let rc = unsafe {
                kglite_datasets_sodir_fetch_all(
                    workdir.as_ptr(),
                    bad_json.as_ptr(),
                    7,
                    30,
                    10,
                    &mut out_report as *mut _,
                    &mut out_err as *mut _,
                )
            };
            assert_eq!(rc, KgliteStatusCode::InvalidArgument);
        }
    }
}

#[cfg(feature = "sodir")]
pub use sodir_ffi::*;
