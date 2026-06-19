//! End-to-end smoke tests through the C ABI surface.
//!
//! These tests go through the same `#[no_mangle] extern "C"`
//! entry points a Go / JS / JVM binding would call, just from
//! Rust. The unit tests in `src/*.rs` exercise individual
//! functions in isolation; this file exercises the full
//! load → session → execute_read → result-accessors → free
//! pipeline so we catch ABI-boundary regressions (handle move
//! semantics, ownership transfer, JSON shape, etc.).

use kglite_c::{
    kglite_abi_version, kglite_cypher_result_columns_json, kglite_cypher_result_free,
    kglite_cypher_result_row_count, kglite_cypher_result_rows_json, kglite_free_string,
    kglite_graph_new, kglite_load_file, kglite_session_execute_mut, kglite_session_execute_read,
    kglite_session_free, kglite_session_new, KgliteCypherResult, KgliteGraph, KgliteSession,
    KgliteStatusCode,
};

#[cfg(feature = "fastembed")]
use kglite_c::{
    kglite_embedder_fastembed_new, kglite_embedder_free, kglite_session_set_embedder,
    KgliteEmbedder,
};
use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;

fn fixture_path() -> CString {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/fixtures/timeseries_graph.kgl");
    CString::new(path.to_str().unwrap()).unwrap()
}

#[test]
fn abi_version_is_aligned_with_crate() {
    let v = kglite_abi_version();
    // Derived from the crate version at compile time — assert it matches
    // rather than hard-coding numbers that silently go stale.
    assert_eq!(
        format!("{}.{}.{}", v.major, v.minor, v.patch),
        format!(
            "{}.{}.{}",
            env!("CARGO_PKG_VERSION_MAJOR"),
            env!("CARGO_PKG_VERSION_MINOR"),
            env!("CARGO_PKG_VERSION_PATCH"),
        )
    );
}

#[test]
fn end_to_end_load_query_free() {
    // 1. Load
    let path = fixture_path();
    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    let mut err_msg: *const c_char = std::ptr::null();
    let rc =
        unsafe { kglite_load_file(path.as_ptr(), &mut graph as *mut _, &mut err_msg as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok, "load failed");
    assert!(!graph.is_null());
    assert!(err_msg.is_null());

    // 2. Wrap in session (moves graph ownership)
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    let rc = unsafe { kglite_session_new(graph, &mut session as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!session.is_null());
    // graph pointer is now invalid — don't free it.

    // 3. Run a Cypher query
    let query = CString::new("MATCH (n) RETURN count(n) AS n").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err_msg: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read(
            session,
            query.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err_msg as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok, "execute_read failed");
    assert!(!result.is_null());
    assert!(err_msg.is_null());

    // 4. Get columns JSON
    let cols_ptr = unsafe { kglite_cypher_result_columns_json(result) };
    assert!(!cols_ptr.is_null());
    let cols_str = unsafe { CStr::from_ptr(cols_ptr).to_str().unwrap() };
    assert_eq!(cols_str, r#"["n"]"#);
    unsafe { kglite_free_string(cols_ptr) };

    // 5. Get rows JSON
    let rows_ptr = unsafe { kglite_cypher_result_rows_json(result) };
    assert!(!rows_ptr.is_null());
    let rows_str = unsafe { CStr::from_ptr(rows_ptr).to_str().unwrap() };
    // Should look like [{"n":<integer>}]
    assert!(rows_str.starts_with("[{\"n\":"));
    assert!(rows_str.ends_with("}]"));
    unsafe { kglite_free_string(rows_ptr) };

    // 6. Row count is 1
    let row_count = unsafe { kglite_cypher_result_row_count(result) };
    assert_eq!(row_count, 1);

    // 7. Teardown
    unsafe { kglite_cypher_result_free(result) };
    unsafe { kglite_session_free(session) };
}

#[test]
fn cypher_syntax_error_returns_error_message() {
    // Load fixture
    let path = fixture_path();
    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    let mut err_msg: *const c_char = std::ptr::null();
    unsafe { kglite_load_file(path.as_ptr(), &mut graph as *mut _, &mut err_msg as *mut _) };
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(graph, &mut session as *mut _) };

    // Bad query — unbalanced bracket forces the parser to fail.
    let query = CString::new("MATCH (n RETURN n").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err_msg: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read(
            session,
            query.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err_msg as *mut _,
        )
    };
    assert_ne!(rc, KgliteStatusCode::Ok);
    assert_eq!(rc, KgliteStatusCode::CypherSyntax);
    assert!(result.is_null());
    assert!(!err_msg.is_null());
    // The message should mention the parse failure
    let msg = unsafe { CStr::from_ptr(err_msg).to_str().unwrap() };
    assert!(!msg.is_empty());
    unsafe { kglite_free_string(err_msg) };

    unsafe { kglite_session_free(session) };
}

#[test]
fn params_json_round_trip() {
    let path = fixture_path();
    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    let mut err_msg: *const c_char = std::ptr::null();
    unsafe { kglite_load_file(path.as_ptr(), &mut graph as *mut _, &mut err_msg as *mut _) };
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(graph, &mut session as *mut _) };

    let query = CString::new("RETURN $x AS x, $y AS y").unwrap();
    let params = CString::new(r#"{"x": 42, "y": "hello"}"#).unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err_msg: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read(
            session,
            query.as_ptr(),
            params.as_ptr(),
            &mut result as *mut _,
            &mut err_msg as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!result.is_null());

    let rows_ptr = unsafe { kglite_cypher_result_rows_json(result) };
    let rows_str = unsafe { CStr::from_ptr(rows_ptr).to_str().unwrap() };
    // Natural untagged JSON for scalar params.
    assert_eq!(rows_str, r#"[{"x":42,"y":"hello"}]"#);
    unsafe { kglite_free_string(rows_ptr) };
    unsafe { kglite_cypher_result_free(result) };
    unsafe { kglite_session_free(session) };
}

#[test]
fn create_empty_graph_then_mutate_and_read() {
    // The hole this closes: build a graph from scratch at the C boundary
    // (no pre-built `.kgl` file), mutate it, and read it back — the path a
    // fresh binding needs for "hello, query a graph".
    let graph = kglite_graph_new();
    assert!(!graph.is_null());

    let mut session: *mut KgliteSession = std::ptr::null_mut();
    let rc = unsafe { kglite_session_new(graph, &mut session as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    // graph ownership moved into the session — don't free it.

    // Mutate: create two nodes via execute_mut (auto-commits).
    let create = CString::new("CREATE (:T {id: 1}), (:T {id: 2})").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_mut(
            session,
            create.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok, "execute_mut failed");
    assert!(err.is_null());
    unsafe { kglite_cypher_result_free(result) };

    // Read it back — both created nodes must be present. Assert via the
    // row-count accessor (encoding-independent) so this test stays green
    // regardless of how scalar values are JSON-encoded in the rows blob.
    let q = CString::new("MATCH (n:T) RETURN n.id AS id ORDER BY id").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read(
            session,
            q.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    let row_count = unsafe { kglite_cypher_result_row_count(result) };
    assert_eq!(row_count, 2, "both created nodes should be returned");
    let rows_ptr = unsafe { kglite_cypher_result_rows_json(result) };
    let rows = unsafe { CStr::from_ptr(rows_ptr).to_str().unwrap() };
    // Natural untagged JSON — bare numbers, not `{"Int64":1}`.
    assert_eq!(rows, r#"[{"id":1},{"id":2}]"#);
    unsafe { kglite_free_string(rows_ptr) };
    unsafe { kglite_cypher_result_free(result) };
    unsafe { kglite_session_free(session) };
}

// ───────────────────────── embedder ─────────────────────────────────

#[cfg(feature = "fastembed")]
#[test]
fn fastembed_factory_rejects_unknown_model() {
    let model = CString::new("definitely-not-a-real-model").unwrap();
    let mut embedder: *mut KgliteEmbedder = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_embedder_fastembed_new(model.as_ptr(), &mut embedder as *mut _, &mut err as *mut _)
    };
    assert_eq!(rc, KgliteStatusCode::InvalidArgument);
    assert!(embedder.is_null());
    assert!(!err.is_null());
    unsafe { kglite_free_string(err) };
}

#[cfg(feature = "fastembed")]
#[test]
fn set_embedder_with_null_args_returns_null_pointer() {
    let rc = unsafe { kglite_session_set_embedder(std::ptr::null_mut(), std::ptr::null()) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
}

#[cfg(feature = "fastembed")]
#[test]
fn embedder_free_is_null_safe() {
    unsafe { kglite_embedder_free(std::ptr::null_mut()) };
}

// ───────────────────────── Sodir dataset ────────────────────────────

#[cfg(feature = "sodir")]
#[test]
fn sodir_fetch_with_bad_json_returns_invalid_argument() {
    use kglite_c::kglite_datasets_sodir_fetch_all;
    let workdir = CString::new("/tmp/kglite_c_sodir_integration_bad").unwrap();
    let bad = CString::new("not-a-json-array").unwrap();
    let mut out_report: *const c_char = std::ptr::null();
    let mut out_err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_datasets_sodir_fetch_all(
            workdir.as_ptr(),
            bad.as_ptr(),
            7,
            30,
            10,
            &mut out_report as *mut _,
            &mut out_err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::InvalidArgument);
    assert!(out_report.is_null());
}

#[cfg(feature = "sodir")]
#[test]
fn sodir_fetch_empty_datasets_succeeds_with_empty_report() {
    use kglite_c::kglite_datasets_sodir_fetch_all;
    use std::fs;

    // Empty datasets array → no fetches → succeeds with default report.
    let workdir_path = std::env::temp_dir().join("kglite_c_sodir_empty");
    let _ = fs::remove_dir_all(&workdir_path); // start clean
    let workdir = CString::new(workdir_path.to_str().unwrap()).unwrap();
    let datasets = CString::new("[]").unwrap();
    let mut out_report: *const c_char = std::ptr::null();
    let mut out_err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_datasets_sodir_fetch_all(
            workdir.as_ptr(),
            datasets.as_ptr(),
            7,
            30,
            10,
            &mut out_report as *mut _,
            &mut out_err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!out_report.is_null());
    let report_str = unsafe { CStr::from_ptr(out_report).to_str().unwrap() };
    // Report is a JSON object with refresh + preprocess.
    let parsed: serde_json::Value = serde_json::from_str(report_str).unwrap();
    assert!(parsed["refresh"].is_object());
    assert!(parsed["preprocess"].is_object());
    unsafe { kglite_free_string(out_report) };
    let _ = fs::remove_dir_all(&workdir_path);
}
