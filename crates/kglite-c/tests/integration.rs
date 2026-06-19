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
    kglite_abi_version, kglite_create_edges_batch, kglite_cypher_result_columns_json,
    kglite_cypher_result_free, kglite_cypher_result_row_count, kglite_cypher_result_rows_json,
    kglite_compute_schema_json, kglite_free_bytes, kglite_free_string, kglite_graph_from_bytes,
    kglite_graph_free, kglite_graph_new, kglite_graph_to_bytes, kglite_load_file,
    kglite_save_graph_durable,
    kglite_session_execute_mut, kglite_session_execute_mut_batch, kglite_session_execute_read,
    kglite_session_execute_read_batch, kglite_session_execute_read_opts, kglite_session_free,
    kglite_session_new, KgliteCypherResult, KgliteGraph, KgliteSession, KgliteStatusCode,
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

#[test]
fn execute_batch_read_and_mut() {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    let rc = unsafe { kglite_session_new(graph, &mut session as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);

    // Two creates in one atomic transaction.
    let muts =
        CString::new(r#"[{"query":"CREATE (:T {id: 1})"},{"query":"CREATE (:T {id: 2})"}]"#)
            .unwrap();
    let mut out: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_mut_batch(
            session,
            muts.as_ptr(),
            &mut out as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok, "mut batch failed");
    assert!(!out.is_null());
    let parsed: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(out).to_str().unwrap() }).unwrap();
    assert_eq!(parsed.as_array().unwrap().len(), 2, "one result per query");
    unsafe { kglite_free_string(out) };

    // Two reads against a single snapshot.
    let reads = CString::new(
        r#"[{"query":"MATCH (n:T) RETURN count(n) AS c"},{"query":"MATCH (n:T) RETURN n.id AS id ORDER BY id"}]"#,
    )
    .unwrap();
    let mut out: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read_batch(
            session,
            reads.as_ptr(),
            &mut out as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    let parsed: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(out).to_str().unwrap() }).unwrap();
    let arr = parsed.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    // count = 2 as a natural number; second query returns two id rows.
    assert_eq!(arr[0]["rows"][0]["c"], serde_json::json!(2));
    assert_eq!(arr[1]["rows"].as_array().unwrap().len(), 2);
    unsafe { kglite_free_string(out) };

    unsafe { kglite_session_free(session) };
}

#[test]
fn execute_mut_batch_is_atomic_on_failure() {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(graph, &mut session as *mut _) };

    // First query valid, second a syntax error → the whole batch rolls back.
    let muts =
        CString::new(r#"[{"query":"CREATE (:Z {id: 1})"},{"query":"MATCH (n RETURN n"}]"#).unwrap();
    let mut out: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_mut_batch(
            session,
            muts.as_ptr(),
            &mut out as *mut _,
            &mut err as *mut _,
        )
    };
    assert_ne!(rc, KgliteStatusCode::Ok);
    assert!(out.is_null());
    assert!(!err.is_null());
    unsafe { kglite_free_string(err) };

    // The valid first CREATE must NOT have landed.
    let q = CString::new("MATCH (n:Z) RETURN count(n) AS c").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    unsafe {
        kglite_session_execute_read(
            session,
            q.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    let rows_ptr = unsafe { kglite_cypher_result_rows_json(result) };
    let rows = unsafe { CStr::from_ptr(rows_ptr).to_str().unwrap() };
    assert_eq!(rows, r#"[{"c":0}]"#, "first create should have rolled back");
    unsafe { kglite_free_string(rows_ptr) };
    unsafe { kglite_cypher_result_free(result) };
    unsafe { kglite_session_free(session) };
}

#[test]
fn create_edges_batch_by_id() {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(graph, &mut session as *mut _) };

    // Seed nodes via Cypher.
    let create =
        CString::new("CREATE (:Person {id: 1}), (:Person {id: 2}), (:Company {id: 10})").unwrap();
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
    assert_eq!(rc, KgliteStatusCode::Ok);
    unsafe { kglite_cypher_result_free(result) };

    // Bulk-add edges by stable id + type. The third edge's source (99)
    // doesn't exist → it should be skipped, not error the batch.
    let edges = CString::new(
        r#"[
          {"src_id":1,"src_type":"Person","dst_id":2,"dst_type":"Person","type":"KNOWS"},
          {"src_id":1,"src_type":"Person","dst_id":10,"dst_type":"Company","type":"WORKS_AT","props":{"since":2020}},
          {"src_id":99,"src_type":"Person","dst_id":2,"dst_type":"Person","type":"KNOWS"}
        ]"#,
    )
    .unwrap();
    let mut out: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_create_edges_batch(
            session,
            edges.as_ptr(),
            &mut out as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok, "create_edges_batch failed");
    assert!(!out.is_null());
    let report: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(out).to_str().unwrap() }).unwrap();
    assert_eq!(report["connections_created"], serde_json::json!(2));
    assert_eq!(report["skipped_missing_endpoint"], serde_json::json!(1));
    unsafe { kglite_free_string(out) };

    // Verify two edges actually landed.
    let q = CString::new("MATCH ()-[r]->() RETURN count(r) AS c").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    unsafe {
        kglite_session_execute_read(
            session,
            q.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    let rows_ptr = unsafe { kglite_cypher_result_rows_json(result) };
    let rows = unsafe { CStr::from_ptr(rows_ptr).to_str().unwrap() };
    assert_eq!(rows, r#"[{"c":2}]"#);
    unsafe { kglite_free_string(rows_ptr) };
    unsafe { kglite_cypher_result_free(result) };
    unsafe { kglite_session_free(session) };
}

#[test]
fn execute_read_opts_caps_rows() {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(graph, &mut session as *mut _) };

    let create = CString::new("CREATE (:T {id: 1}), (:T {id: 2}), (:T {id: 3})").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    unsafe {
        kglite_session_execute_mut(
            session,
            create.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    unsafe { kglite_cypher_result_free(result) };

    // max_rows is a safety guard: a 3-row query with max_rows=2 ERRORS
    // (it does not truncate).
    let q = CString::new("MATCH (n:T) RETURN n.id AS id").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read_opts(
            session,
            q.as_ptr(),
            std::ptr::null(),
            0,
            2,
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    assert_ne!(rc, KgliteStatusCode::Ok, "exceeding max_rows must error");
    assert!(result.is_null());
    assert!(!err.is_null());
    unsafe { kglite_free_string(err) };

    // A limit at/above the row count succeeds and returns all rows.
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read_opts(
            session,
            q.as_ptr(),
            std::ptr::null(),
            0,
            5,
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert_eq!(unsafe { kglite_cypher_result_row_count(result) }, 3);
    unsafe { kglite_cypher_result_free(result) };
    unsafe { kglite_session_free(session) };
}

#[test]
fn graph_bytes_round_trip() {
    // Load fixture → serialize to bytes → free original → load from bytes
    // → query: the round-tripped graph must hold the same nodes.
    let path = fixture_path();
    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    unsafe { kglite_load_file(path.as_ptr(), &mut graph as *mut _, &mut err as *mut _) };

    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut len: usize = 0;
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_graph_to_bytes(graph, &mut buf as *mut _, &mut len as *mut _, &mut err as *mut _)
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!buf.is_null() && len > 0);
    unsafe { kglite_graph_free(graph) }; // original no longer needed

    let mut graph2: *mut KgliteGraph = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc =
        unsafe { kglite_graph_from_bytes(buf, len, &mut graph2 as *mut _, &mut err as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!graph2.is_null());
    unsafe { kglite_free_bytes(buf, len) };

    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(graph2, &mut session as *mut _) };
    let q = CString::new("MATCH (n) RETURN count(n) AS n").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    unsafe {
        kglite_session_execute_read(
            session,
            q.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    let rows_ptr = unsafe { kglite_cypher_result_rows_json(result) };
    let rows = unsafe { CStr::from_ptr(rows_ptr).to_str().unwrap() };
    let parsed: serde_json::Value = serde_json::from_str(rows).unwrap();
    assert!(parsed[0]["n"].as_u64().unwrap() > 0, "round-tripped graph has nodes");
    unsafe { kglite_free_string(rows_ptr) };
    unsafe { kglite_cypher_result_free(result) };
    unsafe { kglite_session_free(session) };
}

#[test]
fn save_graph_durable_round_trips() {
    let path = fixture_path();
    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    unsafe { kglite_load_file(path.as_ptr(), &mut graph as *mut _, &mut err as *mut _) };

    let tmp = std::env::temp_dir().join("kglite_c_durable.kgl");
    let _ = std::fs::remove_file(&tmp);
    let tmp_c = CString::new(tmp.to_str().unwrap()).unwrap();
    let mut err: *const c_char = std::ptr::null();
    // fsync = 1 → durable.
    let rc = unsafe { kglite_save_graph_durable(graph, tmp_c.as_ptr(), 1, &mut err as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    unsafe { kglite_graph_free(graph) };

    // Reloads cleanly.
    let mut graph2: *mut KgliteGraph = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc =
        unsafe { kglite_load_file(tmp_c.as_ptr(), &mut graph2 as *mut _, &mut err as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!graph2.is_null());
    unsafe { kglite_graph_free(graph2) };
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn compute_schema_json_describes_graph() {
    let path = fixture_path();
    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    unsafe { kglite_load_file(path.as_ptr(), &mut graph as *mut _, &mut err as *mut _) };

    let mut out: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc =
        unsafe { kglite_compute_schema_json(graph, &mut out as *mut _, &mut err as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!out.is_null());
    let parsed: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(out).to_str().unwrap() }).unwrap();
    assert!(parsed["node_count"].as_u64().unwrap() > 0);
    assert!(!parsed["node_types"].as_array().unwrap().is_empty());
    unsafe { kglite_free_string(out) };
    unsafe { kglite_graph_free(graph) };
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
