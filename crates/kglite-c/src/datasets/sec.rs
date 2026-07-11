//! SEC EDGAR C ABI — fetcher entry points + pure-CPU helpers
//! over `kglite::api::datasets::sec`. Mirrors the wheel's
//! `_sec_internal` surface.
//!
//! ## Surface summary
//!
//! - `KgliteSecClient` opaque handle wrapping a `SecClient` (the
//!   rate-limited HTTP client SEC's policy requires).
//! - `kglite_datasets_sec_client_new` / `_free` — factory + Drop.
//! - Blocking fetchers (each takes the client + a workdir path):
//!   `fetch_quarterly_master_idx`, `fetch_submissions_bulk`,
//!   `fetch_company_tickers`, `fetch_company_facts`.
//! - Pure-CPU helpers (no client / I/O needed):
//!   `resolve_fetch_buckets`, `parse_tickers_json`.
//! - `kglite_datasets_sec_run_all` — extract pipeline over the
//!   workdir's raw/ tier; takes a JSON-shaped slice spec.
//!
//! Future iterations can extend with the per-filing fetchers
//! (`fetch_form4_filing`, `fetch_13f_info_table`,
//! `fetch_exhibit21_attachment`, etc.) — same pattern.

use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::datasets::sec::{
    fetch_company_facts, fetch_company_tickers, fetch_quarterly_master_idx, fetch_submissions_bulk,
    parse_tickers_json, resolve_fetch_buckets, run_all, ExtractReport, SecClient, SecFormBucket,
    SliceSpec, Workdir, YearRange,
};
use std::ffi::{c_char, CStr};

// ───────────────────────── client handle ────────────────────────────

/// Opaque handle for a SEC HTTP client (rate-limited, user-agent
/// validating). See [`KgliteGraph`](crate::KgliteGraph) for the
/// rationale on the empty `#[repr(C)]` facade pattern.
#[repr(C)]
pub struct KgliteSecClient {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Private state backing a [`KgliteSecClient`] handle.
pub(crate) struct SecClientState {
    pub(crate) inner: SecClient,
}

impl SecClientState {
    fn into_handle(client: SecClient) -> *mut KgliteSecClient {
        let boxed = Box::new(SecClientState { inner: client });
        Box::into_raw(boxed).cast::<KgliteSecClient>()
    }

    unsafe fn from_handle<'a>(handle: *const KgliteSecClient) -> &'a SecClientState {
        unsafe { &*handle.cast::<SecClientState>() }
    }

    unsafe fn free_handle(handle: *mut KgliteSecClient) {
        if handle.is_null() {
            return;
        }
        let _ = unsafe { Box::from_raw(handle.cast::<SecClientState>()) };
    }
}

/// Construct a SEC HTTP client. The `user_agent` is mandatory and
/// must be non-empty — SEC's fair-access policy requires a
/// descriptive identifier with contact info (e.g.
/// `"Acme Corp research@example.com"`).
///
/// # Arguments
///
/// - `user_agent` (in, borrowed): UTF-8 string, non-empty after trim.
/// - `out_client` (out, owned): on success, set to a client handle.
///   Caller must free via [`kglite_datasets_sec_client_free`].
/// - `out_error_msg` (out, owned, may be null): on failure, set to
///   an error message string.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — required pointer is null
/// - `KGLITE_STATUS_CODE_INVALID_UTF8` — `user_agent` isn't valid UTF-8
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — `user_agent` is empty
///   after trim
///
/// # Safety
///
/// `user_agent` must be null-terminated UTF-8.
/// `out_client` must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_client_new(
    user_agent: *const c_char,
    out_client: *mut *mut KgliteSecClient,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_client, std::ptr::null_mut()),
        || {
            if user_agent.is_null() || out_client.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let ua = match unsafe { CStr::from_ptr(user_agent) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            match SecClient::new(ua) {
                Ok(client) => {
                    unsafe {
                        *out_client = SecClientState::into_handle(client);
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
                        *out_client = std::ptr::null_mut();
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    KgliteStatusCode::InvalidArgument
                }
            }
        },
    )
}

/// Free a SEC client handle. Idempotent on null.
///
/// # Safety
///
/// `client` must be either null or a valid pointer previously
/// returned by [`kglite_datasets_sec_client_new`].
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_client_free(client: *mut KgliteSecClient) {
    crate::ffi::void_boundary(|| unsafe { SecClientState::free_handle(client) });
}

// ───────────────────────── fetchers ─────────────────────────────────

/// Fetch the quarterly `master.idx` files covering a year range,
/// landing them in `<workdir>/raw/master_idx/`. Returns counts of
/// files written and files skipped (already present).
///
/// # Arguments
///
/// - `client` (in, borrowed): SEC HTTP client.
/// - `workdir_path` (in, borrowed): root for the layout.
/// - `year_start`, `year_end` (in): inclusive year range. EDGAR's
///   earliest quarter is 1993 Q3; quarters before that are skipped.
/// - `current_year`, `current_quarter` (in): the "now" reference
///   so the fetcher knows to skip future quarters. Callers
///   typically pass the system clock's year + (month/3 + 1).
/// - `out_pair_json` (out, owned): on success, set to a 2-element
///   JSON array `[fetched_count, skipped_count]`.
/// - `out_error_msg` (out, owned, may be null).
///
/// # Safety
///
/// `client` and the string args must be valid. `out_pair_json`
/// must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_fetch_quarterly_master_idx(
    client: *const KgliteSecClient,
    workdir_path: *const c_char,
    year_start: u16,
    year_end: u16,
    current_year: u16,
    current_quarter: u8,
    out_pair_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_pair_json, std::ptr::null()),
        || {
            if client.is_null() || workdir_path.is_null() || out_pair_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let workdir_str = match unsafe { CStr::from_ptr(workdir_path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            if year_start > year_end {
                return KgliteStatusCode::InvalidArgument;
            }
            let client_state = unsafe { SecClientState::from_handle(client) };
            let workdir = Workdir::new(workdir_str);
            let range = YearRange::new(year_start, year_end);

            match fetch_quarterly_master_idx(
                &client_state.inner,
                &workdir,
                range,
                current_year,
                current_quarter,
            ) {
                Ok((fetched, skipped)) => {
                    let json = format!("[{},{}]", fetched, skipped);
                    unsafe {
                        *out_pair_json = alloc_c_string(&json);
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
                        *out_pair_json = std::ptr::null();
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    KgliteStatusCode::Internal
                }
            }
        },
    )
}

/// Fetch the bulk-download `submissions.zip` (all companies' filing
/// metadata in one archive). Lands at
/// `<workdir>/raw/bulk/submissions.zip`. Returns `true` if a fresh
/// download landed (mtime older than `staleness_hours`, or
/// `force_refetch`), `false` if the existing file was reused.
///
/// # Arguments
///
/// - `client` (in, borrowed).
/// - `workdir_path` (in, borrowed).
/// - `staleness_hours` (in): how stale the cached zip can be before
///   re-fetching. SEC publishes nightly so 24 is a reasonable default.
/// - `force_refetch` (in): non-zero forces re-download regardless of
///   staleness.
/// - `out_fetched` (out): set to 1 if downloaded fresh, 0 if reused.
/// - `out_error_msg` (out, owned, may be null).
///
/// # Safety
///
/// `client` and `workdir_path` must be valid. `out_fetched` must be
/// a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_fetch_submissions_bulk(
    client: *const KgliteSecClient,
    workdir_path: *const c_char,
    staleness_hours: u64,
    force_refetch: u8,
    out_fetched: *mut u8,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_fetched, 0),
        || {
            if client.is_null() || workdir_path.is_null() || out_fetched.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let workdir_str = match unsafe { CStr::from_ptr(workdir_path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let client_state = unsafe { SecClientState::from_handle(client) };
            let workdir = Workdir::new(workdir_str);

            match fetch_submissions_bulk(
                &client_state.inner,
                &workdir,
                staleness_hours,
                force_refetch != 0,
            ) {
                Ok(fetched) => {
                    unsafe {
                        *out_fetched = u8::from(fetched);
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
                        *out_fetched = 0;
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    KgliteStatusCode::Internal
                }
            }
        },
    )
}

/// Fetch the `company_tickers.json` mapping (TICKER → CIK).
/// Lands at `<workdir>/raw/company_tickers.json`. Returns `true`
/// if a fresh download landed.
///
/// # Safety
///
/// Same shape as [`kglite_datasets_sec_fetch_submissions_bulk`].
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_fetch_company_tickers(
    client: *const KgliteSecClient,
    workdir_path: *const c_char,
    force_refetch: u8,
    out_fetched: *mut u8,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_fetched, 0),
        || {
            if client.is_null() || workdir_path.is_null() || out_fetched.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let workdir_str = match unsafe { CStr::from_ptr(workdir_path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let client_state = unsafe { SecClientState::from_handle(client) };
            let workdir = Workdir::new(workdir_str);

            match fetch_company_tickers(&client_state.inner, &workdir, force_refetch != 0) {
                Ok(fetched) => {
                    unsafe {
                        *out_fetched = u8::from(fetched);
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
                        *out_fetched = 0;
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    KgliteStatusCode::Internal
                }
            }
        },
    )
}

/// Fetch the XBRL `companyfacts/CIK<cik>.json` file (a single
/// company's full XBRL fact history). Lands at
/// `<workdir>/raw/company_facts/CIK<cik>.json`. Returns `true` if
/// a fresh download landed.
///
/// # Arguments
///
/// - `client`, `workdir_path`: see other fetchers.
/// - `cik` (in): the company's CIK as an integer (no zero-padding).
/// - `force_refetch` (in): non-zero forces re-download.
///
/// # Safety
///
/// Same shape as the other fetchers.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_fetch_company_facts(
    client: *const KgliteSecClient,
    workdir_path: *const c_char,
    cik: u64,
    force_refetch: u8,
    out_fetched: *mut u8,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_fetched, 0),
        || {
            if client.is_null() || workdir_path.is_null() || out_fetched.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let workdir_str = match unsafe { CStr::from_ptr(workdir_path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let client_state = unsafe { SecClientState::from_handle(client) };
            let workdir = Workdir::new(workdir_str);

            match fetch_company_facts(&client_state.inner, &workdir, cik, force_refetch != 0) {
                Ok(fetched) => {
                    unsafe {
                        *out_fetched = u8::from(fetched);
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
                        *out_fetched = 0;
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    KgliteStatusCode::Internal
                }
            }
        },
    )
}

// ───────────────────────── pure helpers ─────────────────────────────

/// Resolve a list of user-supplied form-type strings into the
/// per-filing-fetcher buckets needed to cover them, plus a list of
/// unrecognized form types the caller should warn about.
///
/// Pure-CPU — no I/O, no client. Mirrors
/// `kglite::api::datasets::sec::resolve_fetch_buckets`.
///
/// # Arguments
///
/// - `form_types_json` (in, borrowed): JSON array of form-type strings
///   (e.g. `["10-K", "4", "13F-HR"]`), or the literal `"null"`
///   for "use the lean default set".
/// - `out_active_json` (out, owned): JSON array of bucket name
///   strings, e.g. `["form4", "13f"]`.
/// - `out_unmatched_json` (out, owned): JSON array of strings that
///   didn't match any bucket.
/// - `out_error_msg` (out, owned, may be null).
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — `form_types_json` isn't
///   a JSON array of strings (or null).
///
/// # Safety
///
/// All input strings must be null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_resolve_fetch_buckets(
    form_types_json: *const c_char,
    out_active_json: *mut *const c_char,
    out_unmatched_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {
            crate::ffi::init_out(out_active_json, std::ptr::null());
            crate::ffi::init_out(out_unmatched_json, std::ptr::null());
        },
        || {
            if form_types_json.is_null()
                || out_active_json.is_null()
                || out_unmatched_json.is_null()
            {
                return KgliteStatusCode::NullPointer;
            }
            let json_str = match unsafe { CStr::from_ptr(form_types_json) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            // Accept either null (default lean buckets) or an array of strings.
            let owned_strings: Option<Vec<String>> = if json_str.trim() == "null" {
                None
            } else {
                match serde_json::from_str::<Vec<String>>(json_str) {
                    Ok(v) => Some(v),
                    Err(_) => return KgliteStatusCode::InvalidArgument,
                }
            };

            let form_types_slice: Option<Vec<&str>> = owned_strings
                .as_ref()
                .map(|v| v.iter().map(String::as_str).collect());

            let (active, unmatched) = resolve_fetch_buckets(form_types_slice.as_deref());
            let active_names: Vec<&str> = active.iter().map(bucket_str).collect();
            let active_json =
                serde_json::to_string(&active_names).unwrap_or_else(|_| "[]".to_string());
            let unmatched_json =
                serde_json::to_string(&unmatched).unwrap_or_else(|_| "[]".to_string());

            unsafe {
                *out_active_json = alloc_c_string(&active_json);
                *out_unmatched_json = alloc_c_string(&unmatched_json);
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = std::ptr::null();
                }
            }
            KgliteStatusCode::Ok
        },
    )
}

/// Bucket → wire-name. Stable strings the caller can pattern-match
/// against from any language. Matches the wheel's `_FORM_BUCKETS`
/// names.
fn bucket_str(bucket: &SecFormBucket) -> &'static str {
    match bucket {
        SecFormBucket::Form3 => "form3",
        SecFormBucket::Form4 => "form4",
        SecFormBucket::Form5 => "form5",
        SecFormBucket::Form144 => "form144",
        SecFormBucket::Form13f => "13f",
        SecFormBucket::Form8k => "8k",
        SecFormBucket::Sc13d => "sc13d",
        SecFormBucket::Sc13g => "sc13g",
        SecFormBucket::Def14a => "def14a",
        SecFormBucket::Form10k => "form10k",
    }
}

/// Parse SEC's `company_tickers.json` shape into a TICKER → CIK
/// map. Pure-CPU, no I/O. Returns the map as JSON object string.
///
/// # Arguments
///
/// - `tickers_json` (in, borrowed): the raw JSON from SEC's published
///   `company_tickers.json`.
/// - `out_map_json` (out, owned): JSON object `{"AAPL": 320193, ...}`.
/// - `out_error_msg` (out, owned, may be null).
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — `tickers_json` isn't
///   valid JSON.
///
/// # Safety
///
/// `tickers_json` must be null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_parse_tickers_json(
    tickers_json: *const c_char,
    out_map_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_map_json, std::ptr::null()),
        || {
            if tickers_json.is_null() || out_map_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let json_str = match unsafe { CStr::from_ptr(tickers_json) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            match parse_tickers_json(json_str) {
                Ok(map) => {
                    // Serialize as JSON object — HashMap<String, u64> goes
                    // direct via serde.
                    let out_json = serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string());
                    unsafe {
                        *out_map_json = alloc_c_string(&out_json);
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
                        *out_map_json = std::ptr::null();
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    KgliteStatusCode::InvalidArgument
                }
            }
        },
    )
}

// ───────────────────────── extract pipeline ─────────────────────────

/// Run the SEC extract pipeline — reads `<workdir>/raw/` (the
/// downloaded artifacts) and produces `<workdir>/processed/`
/// CSVs (`company.csv`, `filing_index.csv`, `form4_transaction.csv`,
/// `holding.csv`, etc.).
///
/// # Arguments
///
/// - `workdir_path` (in, borrowed).
/// - `slice_json` (in, borrowed, may be null): JSON object with
///   optional filters:
///   ```json
///   {
///     "cik_list":   [320193, 789019],
///     "form_types": ["10-K", "10-Q"],
///     "year_range": [2020, 2024]
///   }
///   ```
///   Any missing / null field means "no restriction on that axis".
///   Pass null or `"{}"` for fully unrestricted.
/// - `force` (in): non-zero re-runs even when the
///   `<workdir>/processed/holding.csv` sentinel says we already
///   extracted.
/// - `out_report_json` (out, owned): JSON object with extract
///   stats. Caller frees via `kglite_free_string`.
/// - `out_error_msg` (out, owned, may be null).
///
/// # Safety
///
/// `workdir_path` and (if non-null) `slice_json` must be
/// null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn kglite_datasets_sec_run_all(
    workdir_path: *const c_char,
    slice_json: *const c_char,
    force: u8,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_report_json, std::ptr::null()),
        || {
            if workdir_path.is_null() || out_report_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let workdir_str = match unsafe { CStr::from_ptr(workdir_path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let slice = match parse_slice_json(slice_json) {
                Ok(s) => s,
                Err(rc) => return rc,
            };

            let workdir = Workdir::new(workdir_str);
            match run_all(&workdir, &slice, force != 0) {
                Ok(report) => {
                    let json = serialize_extract_report(&report);
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
        },
    )
}

/// Parse the slice-spec JSON into a `SliceSpec`. Null / empty /
/// `"{}"` produces an unrestricted spec; missing fields default to
/// no-restriction on that axis.
fn parse_slice_json(slice_json: *const c_char) -> Result<SliceSpec, KgliteStatusCode> {
    if slice_json.is_null() {
        return Ok(SliceSpec::default());
    }
    let json_str = match unsafe { CStr::from_ptr(slice_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return Err(KgliteStatusCode::InvalidUtf8),
    };
    if json_str.trim().is_empty() {
        return Ok(SliceSpec::default());
    }
    #[derive(serde::Deserialize)]
    struct SliceJson {
        cik_list: Option<Vec<u64>>,
        form_types: Option<Vec<String>>,
        year_range: Option<(u16, u16)>,
    }
    let parsed: SliceJson = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Err(KgliteStatusCode::InvalidArgument),
    };
    let mut spec = SliceSpec::default();
    if let Some(ciks) = parsed.cik_list {
        spec = spec.with_cik_list(ciks);
    }
    if let Some(forms) = parsed.form_types {
        spec = spec.with_form_types(forms);
    }
    if let Some((start, end)) = parsed.year_range {
        if start > end {
            return Err(KgliteStatusCode::InvalidArgument);
        }
        spec = spec.with_year_range(start, end);
    }
    Ok(spec)
}

/// Serialize the engine's `ExtractReport` into a stable JSON shape
/// the C side can parse. Hand-built rather than `Serialize`-derived
/// on the core type per the negative-space convention.
fn serialize_extract_report(report: &ExtractReport) -> String {
    // ExtractReport is { extracted_at, identity_counts,
    // identity_ms, total_ms, form3, form4, form5, ... } — but the
    // exact field set is engine-internal. The wheel's `_sec_internal`
    // serializes it field-by-field too. For v1 we use serde_json's
    // Debug-shaped reflection: turn the report into a Value via the
    // Debug repr.
    //
    // We use a stable subset of fields: extracted_at + identity_ms
    // + total_ms + the per-form `rows_emitted` counts where we can
    // reach them. This matches what most binding consumers care
    // about (success + timing); detailed per-form breakdowns can
    // be added incrementally.
    serde_json::json!({
        "extracted_at": report.extracted_at,
        "identity_ms": report.identity_ms,
        "total_ms": report.total_ms,
        // Form-by-form counts: ExtractReport has form3/4/5/13f/etc.
        // fields, each with `rows_emitted` + `duration_ms`. We
        // serialize them generically via the Debug representation.
        "debug": format!("{:?}", report),
    })
    .to_string()
}

// ───────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn client_factory_rejects_empty_user_agent() {
        let ua = CString::new("").unwrap();
        let mut client: *mut KgliteSecClient = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_sec_client_new(ua.as_ptr(), &mut client as *mut _, &mut err as *mut _)
        };
        assert_eq!(rc, KgliteStatusCode::InvalidArgument);
        assert!(client.is_null());
        assert!(!err.is_null());
        unsafe { crate::kglite_free_string(err) };
    }

    #[test]
    fn client_factory_accepts_valid_ua() {
        let ua = CString::new("kglite-c test test@example.com").unwrap();
        let mut client: *mut KgliteSecClient = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_sec_client_new(ua.as_ptr(), &mut client as *mut _, &mut err as *mut _)
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        assert!(!client.is_null());
        unsafe { kglite_datasets_sec_client_free(client) };
    }

    #[test]
    fn resolve_fetch_buckets_with_null_returns_defaults() {
        let null_json = CString::new("null").unwrap();
        let mut active: *const c_char = std::ptr::null();
        let mut unmatched: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_sec_resolve_fetch_buckets(
                null_json.as_ptr(),
                &mut active as *mut _,
                &mut unmatched as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        let active_str = unsafe { CStr::from_ptr(active).to_str().unwrap() };
        let parsed: Vec<String> = serde_json::from_str(active_str).unwrap();
        // Lean defaults — at least one bucket is in there.
        assert!(!parsed.is_empty());
        unsafe { crate::kglite_free_string(active) };
        unsafe { crate::kglite_free_string(unmatched) };
    }

    #[test]
    fn resolve_fetch_buckets_with_known_forms() {
        let forms = CString::new(r#"["10-K", "4", "ZZZ-UNKNOWN"]"#).unwrap();
        let mut active: *const c_char = std::ptr::null();
        let mut unmatched: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_sec_resolve_fetch_buckets(
                forms.as_ptr(),
                &mut active as *mut _,
                &mut unmatched as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        let unmatched_str = unsafe { CStr::from_ptr(unmatched).to_str().unwrap() };
        assert!(unmatched_str.contains("ZZZ-UNKNOWN"));
        unsafe { crate::kglite_free_string(active) };
        unsafe { crate::kglite_free_string(unmatched) };
    }

    #[test]
    fn parse_tickers_handles_simple_object() {
        let raw = CString::new(
            r#"{"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."},
                "1":{"cik_str":789019,"ticker":"MSFT","title":"Microsoft Corp"}}"#,
        )
        .unwrap();
        let mut out: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_sec_parse_tickers_json(
                raw.as_ptr(),
                &mut out as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        let s = unsafe { CStr::from_ptr(out).to_str().unwrap() };
        let map: std::collections::HashMap<String, u64> = serde_json::from_str(s).unwrap();
        assert_eq!(map.get("AAPL"), Some(&320193u64));
        assert_eq!(map.get("MSFT"), Some(&789019u64));
        unsafe { crate::kglite_free_string(out) };
    }

    #[test]
    fn parse_tickers_with_bad_json_returns_invalid_argument() {
        let bad = CString::new("not-json").unwrap();
        let mut out: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_datasets_sec_parse_tickers_json(
                bad.as_ptr(),
                &mut out as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::InvalidArgument);
        assert!(out.is_null());
        if !err.is_null() {
            unsafe { crate::kglite_free_string(err) };
        }
    }

    #[test]
    fn parse_slice_json_empty_is_unrestricted() {
        let null_json = CString::new("{}").unwrap();
        let spec = parse_slice_json(null_json.as_ptr()).unwrap();
        assert!(spec.cik_list.is_none());
        assert!(spec.form_types.is_none());
        assert!(spec.year_range.is_none());
    }

    #[test]
    fn parse_slice_json_with_all_filters() {
        let s =
            CString::new(r#"{"cik_list":[320193],"form_types":["10-K"],"year_range":[2020,2024]}"#)
                .unwrap();
        let spec = parse_slice_json(s.as_ptr()).unwrap();
        assert_eq!(spec.year_range, Some((2020, 2024)));
        assert!(spec.cik_list.as_ref().unwrap().contains(&320193));
        assert!(spec.form_types.as_ref().unwrap().contains("10-K"));
    }

    #[test]
    fn parse_slice_json_invalid_year_range_rejected() {
        let s = CString::new(r#"{"year_range":[2024,2020]}"#).unwrap();
        let err = parse_slice_json(s.as_ptr()).unwrap_err();
        assert_eq!(err, KgliteStatusCode::InvalidArgument);
    }
}
