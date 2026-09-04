//! End-to-end smoke tests through the C ABI surface.
//!
//! These tests go through the same `#[no_mangle] extern "C"`
//! entry points a Go / JS / JVM binding would call, just from
//! Rust. The unit tests in `src/*.rs` exercise individual
//! functions in isolation; this file exercises the full
//! load → session → execute_read → result-accessors → free
//! pipeline so we catch ABI-boundary regressions (handle move
//! semantics, ownership transfer, JSON shape, etc.).

use kglite::api::io::save_graph;
use kglite::api::session::{execute_mut, ExecuteOptions};
use kglite::api::DirGraph;
use kglite_c::{
    kglite_abi_version, kglite_blueprint_build, kglite_compute_schema_json,
    kglite_create_edges_batch, kglite_cypher_result_columns_json, kglite_cypher_result_free,
    kglite_cypher_result_row_count, kglite_cypher_result_rows_json, kglite_free_bytes,
    kglite_free_string, kglite_graph_free, kglite_graph_from_bytes, kglite_graph_new,
    kglite_graph_to_bytes, kglite_load_file, kglite_open_or_create_graph_in_mode,
    kglite_save_graph_durable, kglite_session_add_embeddings, kglite_session_build_vector_index,
    kglite_session_execute_mut, kglite_session_execute_mut_batch, kglite_session_execute_mut_opts,
    kglite_session_execute_read, kglite_session_execute_read_batch,
    kglite_session_execute_read_opts, kglite_session_free, kglite_session_list_embeddings,
    kglite_session_new, kglite_session_save, kglite_session_set_embeddings,
    kglite_writer_lease_acquire, kglite_writer_lease_acquire_ex, kglite_writer_lease_free,
    KgliteCypherResult, KgliteGraph, KgliteSession, KgliteStatusCode, KgliteWriterLease,
};

#[cfg(feature = "fastembed")]
use kglite_c::{
    kglite_embedder_fastembed_new, kglite_embedder_free, kglite_session_set_embedder,
    KgliteEmbedder,
};
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::{Arc, OnceLock};

fn fixture_path() -> CString {
    static FIXTURE: OnceLock<CString> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let path = std::env::temp_dir().join(format!(
                "kglite-c-current-format-fixture-{}.kgl",
                std::process::id()
            ));
            let path_string = path.to_string_lossy().into_owned();
            let mut graph = DirGraph::new();
            let params = HashMap::new();
            execute_mut(
                &mut graph,
                "CREATE (:Person {id: 1, title: 'Fixture'})",
                &ExecuteOptions::eager(&params),
            )
            .unwrap();
            let mut graph = Arc::new(graph);
            save_graph(&mut graph, &path_string).unwrap();
            CString::new(path_string).unwrap()
        })
        .clone()
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
fn concurrent_auto_commit_mutations_compose() {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_session_new(graph, &mut session) },
        KgliteStatusCode::Ok
    );
    let seed = CString::new("CREATE (:Counter {id: 1, n: 0})").unwrap();
    let mut result = std::ptr::null_mut();
    let mut error = std::ptr::null();
    assert_eq!(
        unsafe {
            kglite_session_execute_mut(
                session,
                seed.as_ptr(),
                std::ptr::null(),
                &mut result,
                &mut error,
            )
        },
        KgliteStatusCode::Ok
    );
    unsafe { kglite_cypher_result_free(result) };

    let raw_session = session as usize;
    let workers: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(move || {
                let session = raw_session as *mut KgliteSession;
                let query = CString::new("MATCH (n:Counter {id: 1}) SET n.n = n.n + 1").unwrap();
                for _ in 0..50 {
                    let mut result = std::ptr::null_mut();
                    let mut error = std::ptr::null();
                    let status = unsafe {
                        kglite_session_execute_mut(
                            session,
                            query.as_ptr(),
                            std::ptr::null(),
                            &mut result,
                            &mut error,
                        )
                    };
                    assert_eq!(status, KgliteStatusCode::Ok);
                    assert!(error.is_null());
                    unsafe { kglite_cypher_result_free(result) };
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }

    let query = CString::new("MATCH (n:Counter {id: 1}) RETURN n.n AS n").unwrap();
    let mut result = std::ptr::null_mut();
    let mut error = std::ptr::null();
    assert_eq!(
        unsafe {
            kglite_session_execute_read(
                session,
                query.as_ptr(),
                std::ptr::null(),
                &mut result,
                &mut error,
            )
        },
        KgliteStatusCode::Ok
    );
    let rows = unsafe { kglite_cypher_result_rows_json(result) };
    assert_eq!(
        unsafe { CStr::from_ptr(rows).to_str().unwrap() },
        r#"[{"n":200}]"#
    );
    unsafe {
        kglite_free_string(rows);
        kglite_cypher_result_free(result);
        kglite_session_free(session);
    }
}

#[test]
fn execute_batch_read_and_mut() {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    let rc = unsafe { kglite_session_new(graph, &mut session as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);

    // Two creates in one atomic transaction.
    let muts = CString::new(r#"[{"query":"CREATE (:T {id: 1})"},{"query":"CREATE (:T {id: 2})"}]"#)
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

    // max_work_units is a work budget, not a row cap: a 3-row query run with
    // max_work_units=2 ERRORS (it does not truncate).
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
    assert_ne!(
        rc,
        KgliteStatusCode::Ok,
        "exceeding max_work_units must error"
    );
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
fn execute_mut_opts_caps_rows_and_rolls_back_statement() {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(graph, &mut session as *mut _) };

    let seed = CString::new("CREATE (:T {id: 'seed', flag: false})").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    assert_eq!(
        unsafe {
            kglite_session_execute_mut(
                session,
                seed.as_ptr(),
                std::ptr::null(),
                &mut result,
                &mut err,
            )
        },
        KgliteStatusCode::Ok
    );
    unsafe { kglite_cypher_result_free(result) };

    let mutation = CString::new(
        "MATCH (n:T {id: 'seed'}) SET n.flag = true \
         WITH [1,2,3] AS xs UNWIND xs AS x RETURN x",
    )
    .unwrap();
    let mut result = std::ptr::dangling_mut::<KgliteCypherResult>();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_mut_opts(
            session,
            mutation.as_ptr(),
            std::ptr::null(),
            0,
            2,
            &mut result,
            &mut err,
        )
    };
    assert_ne!(rc, KgliteStatusCode::Ok);
    assert!(result.is_null());
    assert!(!err.is_null());
    unsafe { kglite_free_string(err) };

    let verify = CString::new("MATCH (n:T {id: 'seed'}) RETURN n.flag AS flag").unwrap();
    let mut result = std::ptr::null_mut();
    let mut err = std::ptr::null();
    assert_eq!(
        unsafe {
            kglite_session_execute_read(
                session,
                verify.as_ptr(),
                std::ptr::null(),
                &mut result,
                &mut err,
            )
        },
        KgliteStatusCode::Ok
    );
    let rows_ptr = unsafe { kglite_cypher_result_rows_json(result) };
    assert_eq!(
        unsafe { CStr::from_ptr(rows_ptr).to_str().unwrap() },
        r#"[{"flag":false}]"#
    );
    unsafe { kglite_free_string(rows_ptr) };
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
        kglite_graph_to_bytes(
            graph,
            &mut buf as *mut _,
            &mut len as *mut _,
            &mut err as *mut _,
        )
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
    assert!(
        parsed[0]["n"].as_u64().unwrap() > 0,
        "round-tripped graph has nodes"
    );
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
    let rc = unsafe { kglite_load_file(tmp_c.as_ptr(), &mut graph2 as *mut _, &mut err as *mut _) };
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
    let rc = unsafe { kglite_compute_schema_json(graph, &mut out as *mut _, &mut err as *mut _) };
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

/// The embedder's write cycle end to end: take the lease, open-or-create in an
/// explicit mode, mutate, save, release. The lease has to cover the whole
/// read-modify-save interval — this asserts it is still refusing a second
/// writer at the moment the save lands, and grants one immediately after the
/// handle is freed.
#[test]
fn writer_lease_covers_the_open_mutate_save_interval() {
    let dir = std::env::temp_dir().join(format!("kglite_c_lease_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("owned.kgl");
    let path_c = CString::new(path.to_str().unwrap()).unwrap();

    let mut lease: *mut KgliteWriterLease = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc =
        unsafe { kglite_writer_lease_acquire(path_c.as_ptr(), 0, &mut lease, &mut err as *mut _) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!lease.is_null());

    let mode = CString::new("memory").unwrap();
    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    let mut converted: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_open_or_create_graph_in_mode(
            path_c.as_ptr(),
            mode.as_ptr(),
            &mut graph as *mut _,
            &mut converted as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!graph.is_null() && converted.is_null());

    // Mutate through a session — which takes ownership of the graph handle,
    // so the checkpoint below has to come from the session.
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_session_new(graph, &mut session as *mut _) },
        KgliteStatusCode::Ok
    );
    let create = CString::new("CREATE (:Owned {id: 1, title: 'kept'})").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    assert_eq!(
        unsafe {
            kglite_session_execute_mut(
                session,
                create.as_ptr(),
                std::ptr::null(),
                &mut result,
                &mut err,
            )
        },
        KgliteStatusCode::Ok
    );
    unsafe { kglite_cypher_result_free(result) };

    // A second writer is still refused at save time — the interval the lease
    // exists to cover.
    let mut contender: *mut KgliteWriterLease = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_writer_lease_acquire(path_c.as_ptr(), 0, &mut contender, &mut err as *mut _)
    };
    assert_eq!(rc, KgliteStatusCode::WriterLeaseHeld);
    assert!(contender.is_null() && !err.is_null());
    unsafe { kglite_free_string(err) };

    let mut err: *const c_char = std::ptr::null();
    assert_eq!(
        unsafe { kglite_session_save(session, path_c.as_ptr(), 1, &mut err) },
        KgliteStatusCode::Ok
    );
    unsafe { kglite_session_free(session) };
    unsafe { kglite_writer_lease_free(lease) };

    // Released — the next writer gets it, and reopening honours what was saved.
    let mut next: *mut KgliteWriterLease = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    assert_eq!(
        unsafe { kglite_writer_lease_acquire(path_c.as_ptr(), 0, &mut next, &mut err as *mut _) },
        KgliteStatusCode::Ok
    );
    unsafe { kglite_writer_lease_free(next) };

    let mut reopened: *mut KgliteGraph = std::ptr::null_mut();
    let mut converted: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    assert_eq!(
        unsafe {
            kglite_open_or_create_graph_in_mode(
                path_c.as_ptr(),
                std::ptr::null(),
                &mut reopened as *mut _,
                &mut converted as *mut _,
                &mut err as *mut _,
            )
        },
        KgliteStatusCode::Ok
    );
    assert!(converted.is_null());

    // The mutation made through the session is in the file.
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(reopened, &mut session as *mut _) };
    let query = CString::new("MATCH (n:Owned) RETURN n.title AS title").unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    assert_eq!(
        unsafe {
            kglite_session_execute_read(
                session,
                query.as_ptr(),
                std::ptr::null(),
                &mut result,
                &mut err,
            )
        },
        KgliteStatusCode::Ok
    );
    let rows = unsafe { kglite_cypher_result_rows_json(result) };
    assert_eq!(
        unsafe { CStr::from_ptr(rows).to_str().unwrap() },
        r#"[{"title":"kept"}]"#,
        "the session's checkpoint must contain the mutation"
    );
    unsafe { kglite_free_string(rows) };
    unsafe { kglite_cypher_result_free(result) };
    unsafe { kglite_session_free(session) };
    let _ = std::fs::remove_dir_all(&dir);
}

/// Regression: a failing `kglite_blueprint_build` must null BOTH out-params
/// (graph + report), so a caller that frees the report on error doesn't free
/// an uninitialized/wild pointer (segfault / heap corruption).
#[test]
fn blueprint_build_error_clears_out_report_json() {
    let bad_path = CString::new("/nonexistent/does-not-exist.yaml").unwrap();
    let dir = CString::new("/tmp").unwrap();
    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    // Sentinel non-null: proves the callee actively clears the slot.
    let mut report: *const c_char = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut err_msg: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_blueprint_build(
            bad_path.as_ptr(),
            dir.as_ptr(),
            &mut graph as *mut _,
            &mut report as *mut _,
            &mut err_msg as *mut _,
        )
    };
    assert_ne!(rc, KgliteStatusCode::Ok, "bad blueprint path must fail");
    assert!(graph.is_null(), "out_graph must be null on error");
    assert!(
        report.is_null(),
        "out_report_json must be nulled on error (else freeing it is UB)"
    );
    if !err_msg.is_null() {
        unsafe { kglite_free_string(err_msg) };
    }
}

#[test]
fn fallible_exports_clear_all_outputs_before_validation() {
    let sentinel_ptr = std::ptr::NonNull::<u8>::dangling().as_ptr();
    let sentinel_cstr: *const c_char = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut error = sentinel_cstr;

    let mut graph = sentinel_ptr.cast::<KgliteGraph>();
    let rc = unsafe {
        kglite_c::kglite_graph_new_in_mode(
            std::ptr::null(),
            std::ptr::null(),
            &mut graph,
            &mut error,
        )
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(graph.is_null() && error.is_null());

    graph = sentinel_ptr.cast();
    error = sentinel_cstr;
    let rc = unsafe { kglite_load_file(std::ptr::null(), &mut graph, &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(graph.is_null() && error.is_null());

    let mut mode = sentinel_cstr;
    error = sentinel_cstr;
    let rc =
        unsafe { kglite_c::kglite_graph_storage_mode(std::ptr::null_mut(), &mut mode, &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(mode.is_null() && error.is_null());

    let mut rdf_graph: *mut KgliteGraph = sentinel_ptr.cast();
    let mut rdf_stats = sentinel_cstr;
    error = sentinel_cstr;
    #[cfg(feature = "rdf")]
    {
        let rc = unsafe {
            kglite_c::kglite_load_rdf(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                -1,
                &mut rdf_graph,
                &mut rdf_stats,
                &mut error,
            )
        };
        assert_eq!(rc, KgliteStatusCode::NullPointer);
        assert!(rdf_graph.is_null() && rdf_stats.is_null() && error.is_null());
    }
    #[cfg(not(feature = "rdf"))]
    let _ = (&mut rdf_graph, &mut rdf_stats, &mut error);

    let mut session = sentinel_ptr.cast::<KgliteSession>();
    let rc = unsafe { kglite_session_new(std::ptr::null_mut(), &mut session) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(session.is_null());

    macro_rules! assert_query_output_reset {
        ($call:expr) => {{
            let mut result = sentinel_ptr.cast::<KgliteCypherResult>();
            error = sentinel_cstr;
            let rc = unsafe { $call(&mut result, &mut error) };
            assert_eq!(rc, KgliteStatusCode::NullPointer);
            assert!(result.is_null() && error.is_null());
        }};
    }
    assert_query_output_reset!(|result, err| kglite_session_execute_read(
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        result,
        err,
    ));
    assert_query_output_reset!(|result, err| kglite_session_execute_read_opts(
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        0,
        0,
        result,
        err,
    ));
    assert_query_output_reset!(|result, err| kglite_session_execute_mut(
        std::ptr::null_mut(),
        std::ptr::null(),
        std::ptr::null(),
        result,
        err,
    ));
    assert_query_output_reset!(|result, err| kglite_session_execute_mut_opts(
        std::ptr::null_mut(),
        std::ptr::null(),
        std::ptr::null(),
        0,
        0,
        result,
        err,
    ));

    let mut json = sentinel_cstr;
    error = sentinel_cstr;
    let rc = unsafe {
        kglite_session_execute_read_batch(std::ptr::null(), std::ptr::null(), &mut json, &mut error)
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(json.is_null() && error.is_null());

    json = sentinel_cstr;
    error = sentinel_cstr;
    let rc = unsafe {
        kglite_c::kglite_graphgen_to_dir(1, 1, 1, 0, 1.0, std::ptr::null(), &mut json, &mut error)
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(json.is_null() && error.is_null());

    graph = sentinel_ptr.cast();
    json = sentinel_cstr;
    error = sentinel_cstr;
    let rc = unsafe {
        kglite_blueprint_build(
            std::ptr::null(),
            std::ptr::null(),
            &mut graph,
            &mut json,
            &mut error,
        )
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(graph.is_null() && json.is_null() && error.is_null());

    for call in [kglite_session_execute_mut_batch, kglite_create_edges_batch] {
        json = sentinel_cstr;
        error = sentinel_cstr;
        let rc = unsafe {
            call(
                std::ptr::null_mut(),
                std::ptr::null(),
                &mut json,
                &mut error,
            )
        };
        assert_eq!(rc, KgliteStatusCode::NullPointer);
        assert!(json.is_null() && error.is_null());
    }

    let mut bytes = sentinel_ptr;
    let mut len = usize::MAX;
    error = sentinel_cstr;
    let rc =
        unsafe { kglite_graph_to_bytes(std::ptr::null_mut(), &mut bytes, &mut len, &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(bytes.is_null() && len == 0 && error.is_null());

    graph = sentinel_ptr.cast();
    error = sentinel_cstr;
    let rc = unsafe { kglite_graph_from_bytes(std::ptr::null(), 0, &mut graph, &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(graph.is_null() && error.is_null());

    graph = sentinel_ptr.cast();
    json = sentinel_cstr;
    error = sentinel_cstr;
    let rc = unsafe {
        kglite_open_or_create_graph_in_mode(
            std::ptr::null(),
            std::ptr::null(),
            &mut graph,
            &mut json,
            &mut error,
        )
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(graph.is_null() && json.is_null() && error.is_null());

    let mut lease = sentinel_ptr.cast::<KgliteWriterLease>();
    error = sentinel_cstr;
    let rc = unsafe { kglite_writer_lease_acquire(std::ptr::null(), 0, &mut lease, &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(lease.is_null() && error.is_null());

    error = sentinel_cstr;
    let rc = unsafe { kglite_session_save(std::ptr::null_mut(), std::ptr::null(), 1, &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(error.is_null());

    json = sentinel_cstr;
    error = sentinel_cstr;
    let rc = unsafe { kglite_compute_schema_json(std::ptr::null_mut(), &mut json, &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(json.is_null() && error.is_null());

    error = sentinel_cstr;
    let rc = unsafe {
        kglite_c::kglite_define_schema(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            &mut error,
        )
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(error.is_null());

    error = sentinel_cstr;
    let rc =
        unsafe { kglite_c::kglite_save_graph(std::ptr::null_mut(), std::ptr::null(), &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(error.is_null());

    error = sentinel_cstr;
    let rc =
        unsafe { kglite_save_graph_durable(std::ptr::null_mut(), std::ptr::null(), 0, &mut error) };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(error.is_null());

    #[cfg(feature = "fastembed")]
    {
        let mut embedder = sentinel_ptr.cast::<KgliteEmbedder>();
        error = sentinel_cstr;
        let rc =
            unsafe { kglite_embedder_fastembed_new(std::ptr::null(), &mut embedder, &mut error) };
        assert_eq!(rc, KgliteStatusCode::NullPointer);
        assert!(embedder.is_null() && error.is_null());
    }

    // Validation failures after all required pointers are accepted must keep
    // the same deterministic output contract.
    let owned_graph = kglite_graph_new();
    let mut owned_session = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_session_new(owned_graph, &mut owned_session) },
        KgliteStatusCode::Ok
    );
    let invalid_utf8 = [0xff_u8, 0];
    let valid_query = CString::new("RETURN 1 AS n").unwrap();
    let malformed_json = CString::new("[").unwrap();
    let invalid_query = CString::new("THIS IS NOT CYPHER").unwrap();
    for (query, params, expected) in [
        (
            invalid_utf8.as_ptr().cast(),
            std::ptr::null(),
            KgliteStatusCode::InvalidUtf8,
        ),
        (
            valid_query.as_ptr(),
            malformed_json.as_ptr(),
            KgliteStatusCode::InvalidArgument,
        ),
        (
            invalid_query.as_ptr(),
            std::ptr::null(),
            KgliteStatusCode::CypherSyntax,
        ),
    ] {
        let mut result = sentinel_ptr.cast::<KgliteCypherResult>();
        error = sentinel_cstr;
        let rc = unsafe {
            kglite_session_execute_read(owned_session, query, params, &mut result, &mut error)
        };
        assert_eq!(rc, expected);
        assert!(result.is_null());
        if expected == KgliteStatusCode::CypherSyntax {
            assert!(!error.is_null());
            unsafe { kglite_free_string(error) };
        } else {
            assert!(error.is_null());
        }
    }
    unsafe { kglite_session_free(owned_session) };
}

// ── embedding ingest (kglite_session_{set,add}_embeddings,
//    build_vector_index, list_embeddings) ─────────────────────────────

/// Build a session over a fresh in-memory graph and seed `Note` nodes with
/// a `body` text column (the ingest primitive requires the source column to
/// exist on at least one node of the type).
fn seed_notes(create: &str) -> *mut KgliteSession {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    unsafe { kglite_session_new(graph, &mut session as *mut _) };
    let create = CString::new(create).unwrap();
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
    assert_eq!(rc, KgliteStatusCode::Ok, "seed failed");
    unsafe { kglite_cypher_result_free(result) };
    session
}

/// Call `kglite_session_set_embeddings` with packed floats, returning
/// (status, report-json, error-msg).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn set_embeddings(
    session: *mut KgliteSession,
    node_type: &str,
    text_column: &str,
    ids_json: &str,
    vectors: &[f32],
    dim: usize,
    count: usize,
    metric: Option<&str>,
) -> (KgliteStatusCode, Option<serde_json::Value>, Option<String>) {
    ingest_call(
        session,
        node_type,
        text_column,
        ids_json,
        vectors,
        dim,
        count,
        metric,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn ingest_call(
    session: *mut KgliteSession,
    node_type: &str,
    text_column: &str,
    ids_json: &str,
    vectors: &[f32],
    dim: usize,
    count: usize,
    metric: Option<&str>,
    add: bool,
) -> (KgliteStatusCode, Option<serde_json::Value>, Option<String>) {
    let nt = CString::new(node_type).unwrap();
    let tc = CString::new(text_column).unwrap();
    let ids = CString::new(ids_json).unwrap();
    let metric_c = metric.map(|m| CString::new(m).unwrap());
    let metric_ptr = metric_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let mut report: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let func = if add {
        kglite_session_add_embeddings
    } else {
        kglite_session_set_embeddings
    };
    let status = unsafe {
        func(
            session,
            nt.as_ptr(),
            tc.as_ptr(),
            ids.as_ptr(),
            vectors.as_ptr(),
            dim,
            count,
            metric_ptr,
            &mut report as *mut _,
            &mut err as *mut _,
        )
    };
    let report_json = (!report.is_null()).then(|| {
        let s = unsafe { CStr::from_ptr(report) }.to_str().unwrap();
        let v = serde_json::from_str(s).unwrap();
        unsafe { kglite_free_string(report) };
        v
    });
    let err_msg = (!err.is_null()).then(|| {
        let s = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
        unsafe { kglite_free_string(err) };
        s
    });
    (status, report_json, err_msg)
}

/// Run a read query with JSON params and return the parsed rows array.
fn query_rows(session: *mut KgliteSession, query: &str, params: &str) -> serde_json::Value {
    let q = CString::new(query).unwrap();
    let p = CString::new(params).unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read(
            session,
            q.as_ptr(),
            p.as_ptr(),
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok, "query failed: {query}");
    let rows_ptr = unsafe { kglite_cypher_result_rows_json(result) };
    let rows: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(rows_ptr) }.to_str().unwrap()).unwrap();
    unsafe { kglite_free_string(rows_ptr) };
    unsafe { kglite_cypher_result_free(result) };
    rows
}

/// End-to-end: packed-float ingest → build index → query with a raw query
/// vector through `vector_score`, asserting the ranking. The ordering
/// assertion is the readback that catches a packed-float offset bug —
/// mis-slicing the buffer scrambles which id owns which vector and flips the
/// order. This is the assertion §7.2 requires: never merely `status == OK`.
#[test]
fn set_embeddings_then_vector_score_ranks_by_packed_vectors() {
    let session = seed_notes(
        "CREATE (:Note {id: 1, body: 'a'}), (:Note {id: 2, body: 'b'}), (:Note {id: 3, body: 'c'})",
    );

    // id 1 -> [1,0], id 2 -> [0,1], id 3 -> [0.9, 0.1]. Row-major, aligned
    // with ids_json order.
    let vectors: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.9, 0.1];
    let (status, report, err) = set_embeddings(
        session,
        "Note",
        "body",
        "[1, 2, 3]",
        &vectors,
        2,
        3,
        Some("cosine"),
    );
    assert_eq!(status, KgliteStatusCode::Ok, "err: {err:?}");
    let report = report.expect("report json");
    assert_eq!(report["embeddings_stored"], 3);
    assert_eq!(report["dimension"], 2);
    assert_eq!(report["skipped"], 0);
    assert_eq!(report["store_created"], true);

    // The store must be readable right after the call (the §7.2 assertion) —
    // vector_score names the STORE ('body_emb'), the query vector rides $q.
    let rows = query_rows(
        session,
        "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s ORDER BY s DESC",
        r#"{"q": [1.0, 0.0]}"#,
    );
    let ids: Vec<i64> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids, vec![1, 3, 2], "ranking must follow the packed vectors");
    // Cosine of the query against id 1's own vector is exactly 1.0.
    assert!((rows[0]["s"].as_f64().unwrap() - 1.0).abs() < 1e-6);

    // Build the index, then the same query still ranks correctly (the build
    // must not corrupt the store).
    let nt = CString::new("Note").unwrap();
    let tc = CString::new("body").unwrap();
    let mut ireport: *const c_char = std::ptr::null();
    let mut ierr: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_build_vector_index(
            session,
            nt.as_ptr(),
            tc.as_ptr(),
            0,
            0,
            0,
            std::ptr::null(),
            &mut ireport as *mut _,
            &mut ierr as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok, "build_vector_index failed");
    let ireport_json: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(ireport) }.to_str().unwrap()).unwrap();
    assert_eq!(ireport_json["indexed"], 3);
    assert_eq!(ireport_json["metric"], "cosine");
    unsafe { kglite_free_string(ireport) };

    let rows = query_rows(
        session,
        "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s ORDER BY s DESC",
        r#"{"q": [1.0, 0.0]}"#,
    );
    let ids: Vec<i64> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids, vec![1, 3, 2], "ranking must survive index build");

    // list_embeddings reports the store.
    let mut lreport: *const c_char = std::ptr::null();
    let mut lerr: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_list_embeddings(session, &mut lreport as *mut _, &mut lerr as *mut _)
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    let list: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(lreport) }.to_str().unwrap()).unwrap();
    unsafe { kglite_free_string(lreport) };
    let arr = list.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["node_type"], "Note");
    assert_eq!(arr[0]["text_column"], "body");
    assert_eq!(arr[0]["dimension"], 2);
    assert_eq!(arr[0]["count"], 3);
    assert_eq!(arr[0]["metric"], "cosine");

    unsafe { kglite_session_free(session) };
}

/// `add_embeddings` upserts across batches and reports `store_created`.
#[test]
fn add_embeddings_upserts_across_batches() {
    let session = seed_notes("CREATE (:Note {id: 1, body: 'a'}), (:Note {id: 2, body: 'b'})");

    let (status, report, err) = ingest_call(
        session,
        "Note",
        "body",
        "[1]",
        &[1.0, 0.0],
        2,
        1,
        Some("cosine"),
        true,
    );
    assert_eq!(status, KgliteStatusCode::Ok, "err: {err:?}");
    assert_eq!(report.unwrap()["store_created"], true);

    // Second batch into the same store: creates nothing, extends it.
    let (status, report, _) = ingest_call(
        session,
        "Note",
        "body",
        "[2]",
        &[0.0, 1.0],
        2,
        1,
        None,
        true,
    );
    assert_eq!(status, KgliteStatusCode::Ok);
    let report = report.unwrap();
    assert_eq!(report["store_created"], false);
    assert_eq!(report["embeddings_stored"], 2);

    unsafe { kglite_session_free(session) };
}

/// An unresolvable id is skipped and counted, never fatal.
#[test]
fn set_embeddings_skips_unknown_ids() {
    let session = seed_notes("CREATE (:Note {id: 1, body: 'a'})");
    let (status, report, err) = set_embeddings(
        session,
        "Note",
        "body",
        "[1, 999]",
        &[1.0, 0.0, 0.0, 1.0],
        2,
        2,
        None,
    );
    assert_eq!(status, KgliteStatusCode::Ok, "err: {err:?}");
    let report = report.unwrap();
    assert_eq!(report["embeddings_stored"], 1);
    assert_eq!(report["skipped"], 1);
    unsafe { kglite_session_free(session) };
}

/// A null required argument is rejected with `NullPointer`, and the report
/// slot is reset to null before validation.
#[test]
fn set_embeddings_rejects_null_arguments() {
    let session = seed_notes("CREATE (:Note {id: 1, body: 'a'})");
    let tc = CString::new("body").unwrap();
    let ids = CString::new("[1]").unwrap();
    let vectors = [1.0f32, 0.0];
    let mut report: *const c_char = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut err: *const c_char = std::ptr::null();
    // node_type is null.
    let status = unsafe {
        kglite_session_set_embeddings(
            session,
            std::ptr::null(),
            tc.as_ptr(),
            ids.as_ptr(),
            vectors.as_ptr(),
            2,
            1,
            std::ptr::null(),
            &mut report as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(status, KgliteStatusCode::NullPointer);
    assert!(report.is_null(), "report slot must reset before validation");
    unsafe { kglite_session_free(session) };
}

/// `ids_json` whose length disagrees with `count` is `InvalidArgument`, not a
/// silent truncation or over-read of the packed buffer.
#[test]
fn set_embeddings_rejects_id_count_mismatch() {
    let session = seed_notes("CREATE (:Note {id: 1, body: 'a'}), (:Note {id: 2, body: 'b'})");
    // count says 2, ids_json has 1.
    let (status, report, err) = set_embeddings(
        session,
        "Note",
        "body",
        "[1]",
        &[1.0, 0.0, 0.0, 1.0],
        2,
        2,
        None,
    );
    assert_eq!(status, KgliteStatusCode::InvalidArgument);
    assert!(report.is_none());
    assert!(
        err.is_none(),
        "a boundary-shape rejection has no engine message"
    );
    unsafe { kglite_session_free(session) };
}

/// `count == 0` is the empty batch: a zero report, no store created, and a
/// null `vectors` pointer is tolerated (never dereferenced).
#[test]
fn set_embeddings_empty_batch_is_a_noop() {
    let session = seed_notes("CREATE (:Note {id: 1, body: 'a'})");
    let nt = CString::new("Note").unwrap();
    let tc = CString::new("body").unwrap();
    let ids = CString::new("[]").unwrap();
    let mut report: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let status = unsafe {
        kglite_session_set_embeddings(
            session,
            nt.as_ptr(),
            tc.as_ptr(),
            ids.as_ptr(),
            std::ptr::null(), // vectors null, allowed when count == 0
            0,
            0,
            std::ptr::null(),
            &mut report as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(status, KgliteStatusCode::Ok, "err: {err:?}");
    let report_json: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(report) }.to_str().unwrap()).unwrap();
    unsafe { kglite_free_string(report) };
    assert_eq!(report_json["embeddings_stored"], 0);
    assert_eq!(report_json["store_created"], false);

    // No store exists.
    let mut lreport: *const c_char = std::ptr::null();
    let mut lerr: *const c_char = std::ptr::null();
    unsafe { kglite_session_list_embeddings(session, &mut lreport as *mut _, &mut lerr as *mut _) };
    let list: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(lreport) }.to_str().unwrap()).unwrap();
    unsafe { kglite_free_string(lreport) };
    assert_eq!(list.as_array().unwrap().len(), 0);
    unsafe { kglite_session_free(session) };
}

/// Building an index for a store that does not exist is a clean
/// `InvalidArgument` with an explanatory message, not a panic.
#[test]
fn build_vector_index_without_a_store_errors() {
    let session = seed_notes("CREATE (:Note {id: 1, body: 'a'})");
    let nt = CString::new("Note").unwrap();
    let tc = CString::new("body").unwrap();
    let mut report: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_build_vector_index(
            session,
            nt.as_ptr(),
            tc.as_ptr(),
            0,
            0,
            0,
            std::ptr::null(),
            &mut report as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::InvalidArgument);
    assert!(report.is_null());
    assert!(!err.is_null(), "a rejected build must explain itself");
    unsafe { kglite_free_string(err) };
    unsafe { kglite_session_free(session) };
}

// ── text index (kglite_session_build_text_index) ─────────────────────

/// Call the symbol, returning (status, report JSON, error message).
fn build_text_index(
    session: *mut KgliteSession,
    node_type: &str,
    property: &str,
) -> (KgliteStatusCode, Option<serde_json::Value>, Option<String>) {
    let nt = CString::new(node_type).unwrap();
    let prop = CString::new(property).unwrap();
    let mut report: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_c::kglite_session_build_text_index(
            session,
            nt.as_ptr(),
            prop.as_ptr(),
            &mut report as *mut _,
            &mut err as *mut _,
        )
    };
    let parsed = (!report.is_null()).then(|| {
        let json: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(report) }.to_str().unwrap()).unwrap();
        unsafe { kglite_free_string(report) };
        json
    });
    let message = (!err.is_null()).then(|| {
        let text = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
        unsafe { kglite_free_string(err) };
        text
    });
    (rc, parsed, message)
}

/// The build reports what it indexed and the index becomes visible to
/// `SHOW INDEXES` — the only query surface it has in this release.
#[test]
fn build_text_index_reports_and_registers() {
    let session = seed_notes(
        "CREATE (:Note {id: 1, body: 'the quick brown fox'}) \
         CREATE (:Note {id: 2, body: 'a quick brown marmoset'}) \
         CREATE (:Note {id: 3, body: 42})",
    );

    let (rc, report, err) = build_text_index(session, "Note", "body");

    assert_eq!(rc, KgliteStatusCode::Ok, "build_text_index failed: {err:?}");
    let report = report.expect("a successful build reports");
    assert_eq!(report["indexed"], 2);
    assert_eq!(report["skipped"], 1, "the numeric body is not text");
    assert_eq!(report["terms"], 6);

    let rows = query_rows(session, "SHOW INDEXES", "{}");
    let text: Vec<&serde_json::Value> = rows
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["type"] == "FULLTEXT")
        .collect();
    assert_eq!(text.len(), 1);
    assert_eq!(text[0]["name"], "Note.body");

    unsafe { kglite_session_free(session) };
}

/// A misspelled property is a clean `InvalidArgument` with an explanatory
/// message, not a panic and not an empty index that silently never matches.
#[test]
fn build_text_index_on_an_unindexable_property_errors() {
    let session = seed_notes("CREATE (:Note {id: 1, body: 'a'})");

    let (rc, report, err) = build_text_index(session, "Note", "bdoy");

    assert_eq!(rc, KgliteStatusCode::InvalidArgument);
    assert!(report.is_none());
    let err = err.expect("a rejected build must explain itself");
    assert!(err.contains("bdoy"), "{err}");

    unsafe { kglite_session_free(session) };
}

/// A null required argument is refused before anything is read.
#[test]
fn build_text_index_null_arguments_are_refused() {
    let session = seed_notes("CREATE (:Note {id: 1, body: 'a'})");
    let nt = CString::new("Note").unwrap();
    let mut report: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_c::kglite_session_build_text_index(
            session,
            nt.as_ptr(),
            std::ptr::null(),
            &mut report as *mut _,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
    assert!(report.is_null());
    unsafe { kglite_session_free(session) };
}

// ── declarative schema (kglite_define_schema) ────────────────────────

/// Call `kglite_define_schema`, returning (status, error-msg).
fn define_schema(
    session: *mut KgliteSession,
    schema_json: &str,
    mode: Option<&str>,
) -> (KgliteStatusCode, Option<String>) {
    let json = CString::new(schema_json).unwrap();
    let mode_c = mode.map(|m| CString::new(m).unwrap());
    let mode_ptr = mode_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let mut err: *const c_char = std::ptr::null();
    let status =
        unsafe { kglite_c::kglite_define_schema(session, json.as_ptr(), mode_ptr, &mut err) };
    let err_msg = (!err.is_null()).then(|| {
        let s = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
        unsafe { kglite_free_string(err) };
        s
    });
    (status, err_msg)
}

/// Run a mutation for its status + message rather than asserting success.
fn try_mutate(session: *mut KgliteSession, query: &str) -> (KgliteStatusCode, Option<String>) {
    let q = CString::new(query).unwrap();
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut err: *const c_char = std::ptr::null();
    let status = unsafe {
        kglite_session_execute_mut(
            session,
            q.as_ptr(),
            std::ptr::null(),
            &mut result as *mut _,
            &mut err as *mut _,
        )
    };
    if !result.is_null() {
        unsafe { kglite_cypher_result_free(result) };
    }
    let err_msg = (!err.is_null()).then(|| {
        let s = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
        unsafe { kglite_free_string(err) };
        s
    });
    (status, err_msg)
}

fn empty_session() -> *mut KgliteSession {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_session_new(graph, &mut session as *mut _) },
        KgliteStatusCode::Ok
    );
    session
}

/// The round trip that matters: a schema declared through the C ABI must come
/// back out of the C ABI *and be enforced*. Reading it back from
/// `CALL db.constraints()` proves it was recorded; the rejected duplicate
/// proves it was installed, which a JSON echo alone would not.
#[test]
fn define_schema_installs_constraints_readable_through_the_abi() {
    let session = empty_session();
    let (rc, err) = define_schema(
        session,
        r#"{"nodes": {"User": {"required": ["id", "email"],
                               "types": {"email": "string"},
                               "primary_key": "email",
                               "unique": [["handle"]]}},
            "connections": {"KNOWS": {"source": "User", "target": "User"}}}"#,
        None,
    );
    assert_eq!(rc, KgliteStatusCode::Ok, "define_schema failed: {err:?}");
    assert!(err.is_none());

    // Recorded: both declared constraints come back through the same ABI.
    let rows = query_rows(
        session,
        "CALL db.constraints() YIELD name, type, properties RETURN name, type, properties \
         ORDER BY name",
        "{}",
    );
    let names: Vec<&str> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("email")) && names.iter().any(|n| n.contains("handle")),
        "declared constraints must be readable back: {names:?}"
    );

    // Installed: the primary key is enforced on the write path.
    let (rc, _) = try_mutate(
        session,
        "CREATE (:User {id: 1, email: 'a@x.com', handle: 'a'})",
    );
    assert_eq!(rc, KgliteStatusCode::Ok);
    let (rc, err) = try_mutate(
        session,
        "CREATE (:User {id: 2, email: 'a@x.com', handle: 'b'})",
    );
    assert_ne!(
        rc,
        KgliteStatusCode::Ok,
        "a duplicate primary key must be rejected"
    );
    assert!(
        err.unwrap_or_default().to_lowercase().contains("email"),
        "the rejection must name the offending property"
    );
    unsafe { kglite_session_free(session) };
}

/// Merge scopes the withdrawal to the types the call names; replace does not.
/// The default (a null `mode`) is merge — the mode whose mistake is a rejected
/// write rather than an admitted duplicate.
#[test]
fn merge_keeps_undeclared_types_enforced_and_replace_withdraws_them() {
    for (mode, user_still_enforced) in [
        (None, true),
        (Some("merge"), true),
        (Some("replace"), false),
    ] {
        let session = empty_session();
        assert_eq!(
            define_schema(
                session,
                r#"{"nodes": {"User": {"primary_key": "email"}}}"#,
                None
            )
            .0,
            KgliteStatusCode::Ok
        );
        assert_eq!(
            try_mutate(session, "CREATE (:User {id: 1, email: 'a@x.com'})").0,
            KgliteStatusCode::Ok
        );
        // A second declaration that never mentions User.
        assert_eq!(
            define_schema(
                session,
                r#"{"nodes": {"Task": {"primary_key": "ref"}}}"#,
                mode
            )
            .0,
            KgliteStatusCode::Ok,
            "mode {mode:?}"
        );
        let rejected = try_mutate(session, "CREATE (:User {id: 2, email: 'a@x.com'})").0
            != KgliteStatusCode::Ok;
        assert_eq!(
            rejected, user_still_enforced,
            "mode {mode:?}: User.email enforcement should be {user_still_enforced}"
        );
        unsafe { kglite_session_free(session) };
    }
}

/// Every refusal arrives as a status *and* a message. A silent
/// `INVALID_ARGUMENT` would leave a binding unable to say what was wrong.
#[test]
fn define_schema_rejections_carry_a_message() {
    let session = empty_session();
    for (json, mode, needle) in [
        (r#"{"nodes":"#, None, "could not be parsed"),
        (r#"[]"#, None, "must be a dictionary"),
        (
            r#"{"nodes": {"P": {"layer": "other"}}}"#,
            None,
            "'managed' or 'runtime'",
        ),
        (
            r#"{"connections": {"R": {"target": "B"}}}"#,
            None,
            "missing required 'source' field",
        ),
        (
            r#"{"nodes": {}}"#,
            Some("Replace"),
            "unknown schema install mode",
        ),
    ] {
        let (rc, err) = define_schema(session, json, mode);
        assert_eq!(rc, KgliteStatusCode::InvalidArgument, "{json} / {mode:?}");
        let message = err.unwrap_or_default();
        assert!(
            message.contains(needle),
            "message for {json} must contain {needle:?}, got {message:?}"
        );
    }
    unsafe { kglite_session_free(session) };
}

/// A declaration that existing data already violates leaves the graph exactly
/// as it was — the promise `set_schema` documents, checked through the ABI.
#[test]
fn a_violated_declaration_changes_nothing() {
    let session = empty_session();
    assert_eq!(
        try_mutate(
            session,
            "CREATE (:User {id: 1, email: 'dup@x.com'}), (:User {id: 2, email: 'dup@x.com'})"
        )
        .0,
        KgliteStatusCode::Ok
    );
    let (rc, err) = define_schema(
        session,
        r#"{"nodes": {"User": {"primary_key": "email"}}}"#,
        None,
    );
    assert_ne!(rc, KgliteStatusCode::Ok, "existing duplicates must refuse");
    assert!(err.is_some(), "a refusal must explain itself");
    // Nothing installed: a third duplicate is still accepted.
    assert_eq!(
        try_mutate(session, "CREATE (:User {id: 3, email: 'dup@x.com'})").0,
        KgliteStatusCode::Ok,
        "a refused declaration must not have installed a constraint"
    );
    unsafe { kglite_session_free(session) };
}

/// A schema installed through C survives a save/load, so the C-side
/// declaration is a real graph property rather than session state.
#[test]
fn a_schema_defined_through_c_survives_a_save() {
    let session = empty_session();
    assert_eq!(
        define_schema(
            session,
            r#"{"nodes": {"User": {"primary_key": "email"}}}"#,
            None
        )
        .0,
        KgliteStatusCode::Ok
    );
    let path = std::env::temp_dir().join(format!(
        "kglite-c-define-schema-{}-{:?}.kgl",
        std::process::id(),
        std::thread::current().id()
    ));
    let path_c = CString::new(path.to_string_lossy().into_owned()).unwrap();
    let mut err: *const c_char = std::ptr::null();
    assert_eq!(
        unsafe { kglite_session_save(session, path_c.as_ptr(), 1, &mut err) },
        KgliteStatusCode::Ok
    );
    unsafe { kglite_session_free(session) };

    let mut graph: *mut KgliteGraph = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_load_file(path_c.as_ptr(), &mut graph, &mut err) },
        KgliteStatusCode::Ok
    );
    let mut reloaded: *mut KgliteSession = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_session_new(graph, &mut reloaded) },
        KgliteStatusCode::Ok
    );
    assert_eq!(
        try_mutate(reloaded, "CREATE (:User {id: 1, email: 'a@x.com'})").0,
        KgliteStatusCode::Ok
    );
    assert_ne!(
        try_mutate(reloaded, "CREATE (:User {id: 2, email: 'a@x.com'})").0,
        KgliteStatusCode::Ok,
        "the saved schema must still be enforced after a reload"
    );
    unsafe { kglite_session_free(reloaded) };
    let _ = std::fs::remove_file(&path);
}

/// `RETURN n` / `RETURN r` / `RETURN p` must reach a C consumer as JSON
/// objects. Until 0.16.1 the shared `kglite_value_to_json` had a
/// `Debug`-string fall-through, so the rows blob carried
/// `"Node(NodeValue { id: 7, labels: [\"T\"], .. })"` — the Rust type's
/// pretty-printer, on the wire of a language-agnostic ABI. The shape here
/// mirrors the Python binding's `py_out::value_to_py` so a JSON consumer
/// and a Python consumer of the same query read the same field names.
#[test]
fn graph_entities_cross_the_abi_as_json_objects() {
    let session = empty_session();
    let (rc, err) = try_mutate(
        session,
        "CREATE (a:T {id: 1, name: 'Ada'})-[:KNOWS {weight: 3}]->(b:T {id: 2, name: 'Bo'})",
    );
    assert_eq!(rc, KgliteStatusCode::Ok, "seed failed: {err:?}");

    let rows = query_rows(session, "MATCH (n:T) WHERE n.id = 1 RETURN n", "{}");
    let node = &rows[0]["n"];
    assert!(
        !rows.to_string().contains("NodeValue"),
        "the Debug rendering leaked across the ABI: {rows}"
    );
    assert!(node.is_object(), "a node must be a JSON object, got {node}");
    assert_eq!(node["labels"], serde_json::json!(["T"]));
    assert_eq!(node["properties"]["name"], serde_json::json!("Ada"));
    assert!(node["id"].is_number());

    let rows = query_rows(session, "MATCH (:T)-[r:KNOWS]->(:T) RETURN r", "{}");
    let rel = &rows[0]["r"];
    assert_eq!(rel["type"], serde_json::json!("KNOWS"));
    assert_eq!(rel["properties"]["weight"], serde_json::json!(3));
    assert!(rel["start"].is_number() && rel["end"].is_number());

    let rows = query_rows(session, "MATCH p = (:T)-[:KNOWS]->(:T) RETURN p", "{}");
    let path = &rows[0]["p"];
    assert_eq!(path["nodes"].as_array().map(Vec::len), Some(2));
    assert_eq!(path["relationships"].as_array().map(Vec::len), Some(1));
    assert_eq!(path["relationships"][0]["type"], serde_json::json!("KNOWS"));

    unsafe { kglite_session_free(session) };
}

/// The `_ex` acquisition hands a refused caller the holder as data. Before
/// 0.16.1 the only way across the ABI was the prose in `out_error_msg`, so
/// every binding that wanted the pid — the Java wrapper's `holder()`, whose
/// javadoc promises "pid, and when the lease was taken" — had to regex a
/// sentence written for humans.
#[test]
fn lease_acquire_ex_reports_the_holder_as_json() {
    let dir = std::env::temp_dir().join(format!("kglite_c_lease_ex_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("held.kgl");
    let path_c = CString::new(path.to_str().unwrap()).unwrap();

    // Success clears the holder slot: there is no holder to report.
    let mut lease: *mut KgliteWriterLease = std::ptr::null_mut();
    let mut holder: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_writer_lease_acquire_ex(
            path_c.as_ptr(),
            0,
            &mut lease,
            &mut holder as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(!lease.is_null() && holder.is_null() && err.is_null());

    // The refusal carries pid / since / self as fields, not prose.
    let mut contender: *mut KgliteWriterLease = std::ptr::null_mut();
    let mut holder: *const c_char = std::ptr::null();
    let mut err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_writer_lease_acquire_ex(
            path_c.as_ptr(),
            0,
            &mut contender,
            &mut holder as *mut _,
            &mut err as *mut _,
        )
    };
    assert_eq!(rc, KgliteStatusCode::WriterLeaseHeld);
    assert!(contender.is_null());
    assert!(!holder.is_null(), "a held lease must report its holder");
    let parsed: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(holder) }.to_str().unwrap()).unwrap();
    assert_eq!(
        parsed["pid"],
        serde_json::json!(std::process::id()),
        "the holder is this test process: {parsed}"
    );
    assert!(
        parsed["since"].as_str().is_some_and(|s| s.contains('T')),
        "since must be an RFC-3339 timestamp: {parsed}"
    );
    assert_eq!(parsed["self"], serde_json::json!(true));
    // The prose is still there, and still the same string the plain symbol
    // would have produced.
    // Owned before the free below — `message` outlives the C string.
    let message = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
    assert_eq!(parsed["message"], serde_json::json!(message));
    unsafe { kglite_free_string(holder) };
    unsafe { kglite_free_string(err) };

    // The plain symbol is unchanged — same code, same message.
    let mut contender: *mut KgliteWriterLease = std::ptr::null_mut();
    let mut plain_err: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_writer_lease_acquire(path_c.as_ptr(), 0, &mut contender, &mut plain_err as *mut _)
    };
    assert_eq!(rc, KgliteStatusCode::WriterLeaseHeld);
    assert_eq!(
        unsafe { CStr::from_ptr(plain_err) }.to_str().unwrap(),
        message
    );
    unsafe { kglite_free_string(plain_err) };

    unsafe { kglite_writer_lease_free(lease) };
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn retrieval_diagnostics_survive_c_handles_and_batches() {
    let session = seed_notes("CREATE (:Note {id:1, body:'a'}), (:Note {id:2, body:'b'})");
    let (rc, _, _) = set_embeddings(
        session,
        "Note",
        "body",
        "[1,2]",
        &[1., 0., 0., 1.],
        2,
        2,
        None,
    );
    assert_eq!(rc, KgliteStatusCode::Ok);
    let query = "MATCH (n:Note) RETURN vector_score(n,'body_emb',[1.0,0.0],{exact:true}) AS s ORDER BY s DESC LIMIT 1";
    let query_c = CString::new(query).unwrap();
    let mut result = std::ptr::null_mut();
    let mut error = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read(
            session,
            query_c.as_ptr(),
            std::ptr::null(),
            &mut result,
            &mut error,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    let json = unsafe { kglite_c::kglite_cypher_result_diagnostics_json(result) };
    unsafe { kglite_cypher_result_free(result) };
    let diagnostics: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(json).to_str().unwrap() }).unwrap();
    assert_eq!(diagnostics["retrieval"][0]["actual_mode"], "exact");
    assert_eq!(
        diagnostics["retrieval"][0]["fallback_reason"],
        "forced_exact"
    );
    unsafe { kglite_free_string(json) };
    let batch = CString::new(
        serde_json::json!([{ "query": query }, { "query": "RETURN 1 AS n" }]).to_string(),
    )
    .unwrap();
    let mut output = std::ptr::null();
    let rc = unsafe {
        kglite_session_execute_read_batch(session, batch.as_ptr(), &mut output, &mut error)
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    let batch: serde_json::Value =
        serde_json::from_str(unsafe { CStr::from_ptr(output).to_str().unwrap() }).unwrap();
    assert_eq!(
        batch[0]["diagnostics"]["retrieval"],
        diagnostics["retrieval"]
    );
    assert_eq!(batch[1]["diagnostics"]["retrieval"], serde_json::json!([]));
    unsafe {
        kglite_free_string(output);
        kglite_session_free(session);
    }
}
