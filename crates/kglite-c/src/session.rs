//! `KgliteSession` opaque handle — session creation +
//! execute_read / execute_mut.
//!
//! The Session owns the graph after [`kglite_session_new`] — the
//! Arc moves in and the caller should NOT free the graph handle
//! afterwards.

use crate::graph::{GraphState, KgliteGraph};
use crate::result::{result_to_json_object, KgliteCypherResult, ResultState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::mutation::{add_edges_from_specs, EdgeSpec};
use kglite::api::param::{json_object_to_value_map, json_value_to_kglite_value};
use kglite::api::session::{execute_mut, execute_read, ExecuteOptions, Session};
use kglite::api::{Embedder, Value};
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Opaque handle for a session. See [`KgliteGraph`](crate::KgliteGraph)
/// for the rationale on the empty `#[repr(C)]` facade pattern.
#[repr(C)]
pub struct KgliteSession {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Private state backing a [`KgliteSession`] handle.
pub(crate) struct SessionState {
    pub(crate) inner: Session,
    /// Optional embedder attached to this session. When set, every
    /// execute_read / execute_mut call passes the embedder into
    /// `ExecuteOptions` so `text_score()` and friends work.
    /// Attached via
    /// [`kglite_session_set_embedder`](crate::kglite_session_set_embedder).
    ///
    /// Behind a `Mutex` because the ABI documents the session as
    /// cross-thread safe: `set_embedder` may race with concurrent
    /// execute calls cloning the field, and a bare `&mut` write
    /// through the handle would alias those `&` reads (UB). The lock
    /// is held only for the clone/store — never across a query.
    pub(crate) embedder: Mutex<Option<Arc<dyn Embedder>>>,
}

impl SessionState {
    fn into_handle(session: Session) -> *mut KgliteSession {
        let boxed = Box::new(SessionState {
            inner: session,
            embedder: Mutex::new(None),
        });
        Box::into_raw(boxed).cast::<KgliteSession>()
    }

    pub(crate) unsafe fn from_handle<'a>(handle: *const KgliteSession) -> &'a SessionState {
        unsafe { &*handle.cast::<SessionState>() }
    }

    /// Replace the session's embedder. Interior mutability (see the
    /// `embedder` field doc) — callers hold only `&SessionState`.
    pub(crate) fn set_embedder(&self, embedder: Arc<dyn Embedder>) {
        *self.embedder.lock().unwrap_or_else(PoisonError::into_inner) = Some(embedder);
    }

    unsafe fn free_handle(handle: *mut KgliteSession) {
        if handle.is_null() {
            return;
        }
        let _ = unsafe { Box::from_raw(handle.cast::<SessionState>()) };
    }
}

/// Create a new session from a graph handle. The session takes
/// ownership of the graph — the caller MUST NOT call
/// [`kglite_graph_free`](crate::kglite_graph_free) on the handle
/// after this call. Free the session via
/// [`kglite_session_free`] when done.
///
/// # Arguments
///
/// - `graph` (in, MOVED): graph handle. After this call, the
///   pointer is no longer valid for any other use.
/// - `out_session` (out, owned): set to the session handle on
///   success; caller must free via [`kglite_session_free`].
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `graph` or `out_session` is null
///
/// # Safety
///
/// `graph` must be a valid `*mut KgliteGraph` previously returned
/// by [`kglite_load_file`](crate::kglite_load_file) and not yet
/// freed or moved into another session. `out_session` must be a
/// valid writable pointer to a `*mut KgliteSession` slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_new(
    graph: *mut KgliteGraph,
    out_session: *mut *mut KgliteSession,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        std::ptr::null_mut(),
        || crate::ffi::init_out(out_session, std::ptr::null_mut()),
        || {
            if graph.is_null() || out_session.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            // Safety: caller's contract — graph is a valid handle, not
            // yet freed. We MOVE the Arc out by reconstructing the Box
            // behind the opaque facade.
            let graph_state = unsafe { Box::from_raw(graph.cast::<GraphState>()) };
            let session = Session::from_arc(graph_state.inner);
            unsafe { *out_session = SessionState::into_handle(session) };
            KgliteStatusCode::Ok
        },
    )
}

/// Run a read-only Cypher query.
///
/// # Arguments
///
/// - `session` (in, borrowed): the session.
/// - `query` (in, borrowed): UTF-8 Cypher query, null-terminated.
/// - `params_json` (in, borrowed, may be null): JSON object of
///   parameter bindings. Pass null or `"{}"` for no params.
/// - `out_result` (out, owned): on success, set to the result
///   handle; caller must free via [`kglite_cypher_result_free`].
/// - `out_error_msg` (out, owned, may be null): on failure, set
///   to the error message; caller must free via
///   [`kglite_free_string`](crate::kglite_free_string).
///
/// # Errors
///
/// Any `KgErrorCode` variant — Cypher syntax / type mismatch /
/// timeout / execution error / node-not-found / argument
/// validation. The error message describes the specific failure.
///
/// # Safety
///
/// `session` must be valid. `query` and (if non-null) `params_json`
/// must be null-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_execute_read(
    session: *const KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_result, std::ptr::null_mut()),
        || {
            if session.is_null() || query.is_null() || out_result.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let query_str = match unsafe { CStr::from_ptr(query) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let params = match parse_params_json(params_json) {
                Ok(p) => p,
                Err(rc) => return rc,
            };

            let session_state = unsafe { SessionState::from_handle(session) };
            let snapshot = session_state.inner.snapshot();
            let opts = session_state.make_opts(&params);

            match execute_read(&snapshot, query_str, &opts) {
                Ok(outcome) => {
                    unsafe {
                        *out_result = ResultState::into_handle(outcome.result);
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
                        *out_result = std::ptr::null_mut();
                    }
                    let code = KgliteStatusCode::from_kg_error_code(err.code());
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    code
                }
            }
        },
    )
}

/// Run a read-only Cypher query with execution options. Same as
/// [`kglite_session_execute_read`], plus:
///
/// - `timeout_ms`: past this wall-clock budget the query returns
///   `CypherTimeout`. `0` = no deadline.
/// - `max_rows`: reject the query (error) if it would produce more than
///   this many rows — a safety guard against runaway results, not a
///   silent truncation; add a `LIMIT` clause to bound output. `0` = no
///   limit.
///
/// # Safety
///
/// Same as [`kglite_session_execute_read`].
#[no_mangle]
pub unsafe extern "C" fn kglite_session_execute_read_opts(
    session: *const KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    timeout_ms: u64,
    max_rows: u64,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_result, std::ptr::null_mut()),
        || {
            if session.is_null() || query.is_null() || out_result.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let query_str = match unsafe { CStr::from_ptr(query) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let params = match parse_params_json(params_json) {
                Ok(p) => p,
                Err(rc) => return rc,
            };

            let session_state = unsafe { SessionState::from_handle(session) };
            let snapshot = session_state.inner.snapshot();
            let mut opts = session_state.make_opts(&params);
            if timeout_ms > 0 {
                opts.deadline = Some(Instant::now() + Duration::from_millis(timeout_ms));
            }
            if max_rows > 0 {
                opts.max_rows = Some(max_rows as usize);
            }

            match execute_read(&snapshot, query_str, &opts) {
                Ok(outcome) => {
                    unsafe {
                        *out_result = ResultState::into_handle(outcome.result);
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
                        *out_result = std::ptr::null_mut();
                    }
                    let code = KgliteStatusCode::from_kg_error_code(err.code());
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    code
                }
            }
        },
    )
}

/// Run a mutating Cypher query. Same shape as
/// [`kglite_session_execute_read`] but accepts CREATE / SET /
/// DELETE / REMOVE / MERGE statements. The session's underlying
/// graph is auto-committed after a successful execute (no
/// explicit begin/commit in v1 — explicit transactions land in
/// a future ABI version once a binding needs them).
///
/// # Safety
///
/// Same as [`kglite_session_execute_read`] except `session` is
/// declared as `*mut` (the call mutates the session's interior
/// graph via commit-swap).
#[no_mangle]
pub unsafe extern "C" fn kglite_session_execute_mut(
    session: *mut KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    unsafe { execute_mut_impl(session, query, params_json, 0, 0, out_result, out_error_msg) }
}

/// Run a mutating query with the same timeout and row/collection budget
/// semantics as [`kglite_session_execute_read_opts`]. A budget failure rolls
/// back the complete statement. `0` disables the corresponding option.
///
/// # Safety
///
/// Same as [`kglite_session_execute_mut`].
#[no_mangle]
pub unsafe extern "C" fn kglite_session_execute_mut_opts(
    session: *mut KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    timeout_ms: u64,
    max_rows: u64,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    unsafe {
        execute_mut_impl(
            session,
            query,
            params_json,
            timeout_ms,
            max_rows,
            out_result,
            out_error_msg,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn execute_mut_impl(
    session: *mut KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    timeout_ms: u64,
    max_rows: u64,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_result, std::ptr::null_mut()),
        || {
            if session.is_null() || query.is_null() || out_result.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let query_str = match unsafe { CStr::from_ptr(query) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let params = match parse_params_json(params_json) {
                Ok(p) => p,
                Err(rc) => return rc,
            };

            // `execute_mut` takes `*mut` for the C ABI but the SessionState
            // mutex makes the actual interior mutation thread-safe — we
            // borrow `&SessionState` here and rely on Session's internal
            // Mutex for the commit-swap.
            let session_state = unsafe { SessionState::from_handle(session) };
            let mut opts = session_state.make_opts(&params);
            if timeout_ms > 0 {
                opts.deadline = Some(Instant::now() + Duration::from_millis(timeout_ms));
            }
            if max_rows > 0 {
                opts.max_rows = Some(max_rows as usize);
            }

            // Hold the core Session write guard across execution. This serializes the
            // complete mutation (preventing last-writer-loses races) and reaches the
            // unique-owner path without the old redundant working-copy clone.
            let mut working = session_state.inner.write();
            let exec_result = execute_mut(&mut working, query_str, &opts);

            match exec_result {
                Ok(outcome) => {
                    unsafe {
                        *out_result = ResultState::into_handle(outcome.result);
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = std::ptr::null();
                        }
                    }
                    KgliteStatusCode::Ok
                }
                Err(err) => {
                    // tx drops without commit — no mutation reaches the
                    // session's stored Arc.
                    unsafe {
                        *out_result = std::ptr::null_mut();
                    }
                    let code = KgliteStatusCode::from_kg_error_code(err.code());
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    code
                }
            }
        },
    )
}

/// Run several read-only Cypher queries against a single consistent
/// snapshot, in one lock acquisition.
///
/// `queries_json` is a JSON array of objects, each `{"query": "...",
/// "params": {...}}` (the `params` key is optional). Every query sees
/// the same snapshot, taken once up front — cheaper and more consistent
/// than N separate [`kglite_session_execute_read`] calls when a binding
/// issues many small reads.
///
/// On success `out_results_json` is set to an owned JSON string: an
/// array of `{"columns": [...], "rows": [{...}]}` objects, one per input
/// query in order, with the same natural-value encoding as
/// [`kglite_cypher_result_rows_json`]. Free it with
/// [`kglite_free_string`](crate::kglite_free_string).
///
/// The batch aborts on the first failing query: `out_results_json` is
/// set to null and the status code / `out_error_msg` describe that
/// query's failure.
///
/// # Safety
///
/// `session` must be valid; `queries_json` a null-terminated UTF-8 JSON
/// array; `out_results_json` a valid writable `*const c_char` slot;
/// `out_error_msg` null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_execute_read_batch(
    session: *const KgliteSession,
    queries_json: *const c_char,
    out_results_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_results_json, std::ptr::null()),
        || {
            if session.is_null() || queries_json.is_null() || out_results_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let queries = match parse_batch_queries(queries_json) {
                Ok(q) => q,
                Err(rc) => return rc,
            };
            let session_state = unsafe { SessionState::from_handle(session) };
            let snapshot = session_state.inner.snapshot();
            let mut results = Vec::with_capacity(queries.len());
            for (query, params) in &queries {
                let opts = session_state.make_opts(params);
                match execute_read(&snapshot, query, &opts) {
                    Ok(outcome) => results.push(result_to_json_object(&outcome.result)),
                    Err(err) => {
                        unsafe {
                            *out_results_json = std::ptr::null();
                        }
                        let code = KgliteStatusCode::from_kg_error_code(err.code());
                        if !out_error_msg.is_null() {
                            unsafe {
                                *out_error_msg = alloc_c_string(&err.to_string());
                            }
                        }
                        return code;
                    }
                }
            }
            let json = serde_json::Value::Array(results).to_string();
            unsafe {
                *out_results_json = alloc_c_string(&json);
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

/// Run several mutating Cypher queries in a single transaction — one
/// `begin`, N executes (each sees the previous query's writes), a single
/// `commit`. The batch is **atomic**: if any query fails, the
/// transaction is dropped uncommitted and none of the batch's mutations
/// reach the graph.
///
/// `queries_json` / `out_results_json` have the same shape as
/// [`kglite_session_execute_read_batch`]. On failure `out_results_json`
/// is null and the status / `out_error_msg` describe the failing query.
///
/// # Safety
///
/// Same as [`kglite_session_execute_read_batch`] except `session` is
/// `*mut` (the call mutates the session's interior graph via
/// commit-swap).
#[no_mangle]
pub unsafe extern "C" fn kglite_session_execute_mut_batch(
    session: *mut KgliteSession,
    queries_json: *const c_char,
    out_results_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_results_json, std::ptr::null()),
        || {
            if session.is_null() || queries_json.is_null() || out_results_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let queries = match parse_batch_queries(queries_json) {
                Ok(q) => q,
                Err(rc) => return rc,
            };
            let session_state = unsafe { SessionState::from_handle(session) };
            let transaction: Result<Vec<serde_json::Value>, Box<kglite::api::KgError>> =
                session_state.inner.transact(|working| {
                    let mut results = Vec::with_capacity(queries.len());
                    for (query, params) in &queries {
                        let opts = session_state.make_opts(params);
                        let outcome = execute_mut(working, query, &opts).map_err(Box::new)?;
                        results.push(result_to_json_object(&outcome.result));
                    }
                    Ok(results)
                });
            let results = match transaction {
                Ok(results) => results,
                Err(err) => {
                    unsafe {
                        *out_results_json = std::ptr::null();
                    }
                    let code = KgliteStatusCode::from_kg_error_code(err.code());
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&err.to_string());
                        }
                    }
                    return code;
                }
            };
            let json = serde_json::Value::Array(results).to_string();
            unsafe {
                *out_results_json = alloc_c_string(&json);
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

/// Bulk-create edges addressed by **stable node id + type**, bypassing
/// Cypher — the fast ingest path for bindings loading many edges.
///
/// `edges_json` is a JSON array of objects:
/// `{"src_id": <id>, "src_type": "Person", "dst_id": <id>,
///   "dst_type": "Company", "type": "WORKS_AT", "props": {...}}`
/// (`props` optional). `src_id`/`dst_id` are the nodes' stable ids (the
/// same value `n.id` returns), not internal indices. Runs in one
/// transaction: the whole batch commits together, or — on error — none
/// of it lands. Endpoints must already exist; an edge whose source or
/// target id isn't found for its declared type is skipped and counted.
///
/// On success `out_report_json` is set to an owned JSON object
/// `{"connections_created": N, "skipped_missing_endpoint": M}`; free it
/// with [`kglite_free_string`](crate::kglite_free_string).
///
/// This wraps the shared core primitive
/// [`add_edges_from_specs`](kglite::api::mutation::add_edges_from_specs) —
/// the same engine the Python `add_connections` DataFrame path uses.
///
/// # Safety
///
/// `session` must be valid; `edges_json` a null-terminated UTF-8 JSON
/// array; `out_report_json` a valid writable `*const c_char` slot;
/// `out_error_msg` null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_create_edges_batch(
    session: *mut KgliteSession,
    edges_json: *const c_char,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_report_json, std::ptr::null()),
        || {
            if session.is_null() || edges_json.is_null() || out_report_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let specs = match parse_edge_specs(edges_json) {
                Ok(s) => s,
                Err(rc) => return rc,
            };
            let session_state = unsafe { SessionState::from_handle(session) };
            // Route through `Session::transact` so the whole batch runs
            // under the Session write lock: serialized with concurrent
            // execute_mut writers (no last-writer-wins Arc-swap losing
            // their commits) and atomic — an error drops the fork with
            // no partial writes.
            let transaction: Result<_, String> = session_state
                .inner
                .transact(|working| add_edges_from_specs(working, specs));
            match transaction {
                Ok(report) => {
                    let json = serde_json::json!({
                        "connections_created": report.connections_created,
                        "skipped_missing_endpoint": report.skipped_missing_endpoint,
                    })
                    .to_string();
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
                Err(msg) => {
                    // transact dropped the fork → none of the batch's
                    // edges land.
                    unsafe {
                        *out_report_json = std::ptr::null();
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&msg);
                        }
                    }
                    KgliteStatusCode::Internal
                }
            }
        },
    )
}

/// Free a session handle. Idempotent on null (no-op).
///
/// # Safety
///
/// `session` must be either null or a valid pointer previously
/// returned by [`kglite_session_new`] and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_free(session: *mut KgliteSession) {
    crate::ffi::void_boundary(|| unsafe { SessionState::free_handle(session) });
}

impl SessionState {
    /// Build the per-call [`ExecuteOptions`] for this session — eager
    /// defaults plus the session's embedder. Centralized so the read / mut /
    /// batch paths can't drift on per-call option defaults.
    fn make_opts<'a>(&self, params: &'a HashMap<String, Value>) -> ExecuteOptions<'a> {
        let mut opts = ExecuteOptions::eager(params);
        opts.embedder = self
            .embedder
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        opts
    }
}

/// Parse a JSON-string params argument into a HashMap. Null /
/// empty / "{}" → empty map. Any other shape (array, scalar,
/// nested object value) maps via
/// [`json_value_to_kglite_value`](kglite::api::param::json_value_to_kglite_value).
fn parse_params_json(
    params_json: *const c_char,
) -> Result<HashMap<String, Value>, KgliteStatusCode> {
    if params_json.is_null() {
        return Ok(HashMap::new());
    }
    let s = match unsafe { CStr::from_ptr(params_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return Err(KgliteStatusCode::InvalidUtf8),
    };
    if s.is_empty() {
        return Ok(HashMap::new());
    }
    let parsed: serde_json::Value = match serde_json::from_str(s) {
        Ok(v) => v,
        Err(_) => return Err(KgliteStatusCode::InvalidArgument),
    };
    match parsed {
        serde_json::Value::Object(obj) => Ok(json_object_to_value_map(&obj)),
        serde_json::Value::Null => Ok(HashMap::new()),
        _ => Err(KgliteStatusCode::InvalidArgument),
    }
}

/// Read an optional JSON-object field (`params` / `props`) off a batch
/// entry and build its `Value` map: absent / null → empty map; an object →
/// the converted map; any other shape → `InvalidArgument`. Shared by the
/// batch-query and edge-spec parsers so the two stay byte-identical.
fn optional_object_map(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<HashMap<String, Value>, KgliteStatusCode> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(HashMap::new()),
        Some(serde_json::Value::Object(o)) => Ok(json_object_to_value_map(o)),
        Some(_) => Err(KgliteStatusCode::InvalidArgument),
    }
}

/// One parsed batch entry: a query string and its parameter map.
type BatchQuery = (String, HashMap<String, Value>);

/// Parse a batch `queries_json` argument into `(query, params)` pairs.
/// Expects a JSON array of objects, each `{"query": "...", "params":
/// {...}}` (the `params` key is optional). Any other shape →
/// `InvalidArgument`. Assumes `queries_json` is non-null (callers check).
fn parse_batch_queries(queries_json: *const c_char) -> Result<Vec<BatchQuery>, KgliteStatusCode> {
    let s = match unsafe { CStr::from_ptr(queries_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return Err(KgliteStatusCode::InvalidUtf8),
    };
    let parsed: serde_json::Value = match serde_json::from_str(s) {
        Ok(v) => v,
        Err(_) => return Err(KgliteStatusCode::InvalidArgument),
    };
    let arr = match parsed.as_array() {
        Some(a) => a,
        None => return Err(KgliteStatusCode::InvalidArgument),
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = match item.as_object() {
            Some(o) => o,
            None => return Err(KgliteStatusCode::InvalidArgument),
        };
        let query = match obj.get("query").and_then(|v| v.as_str()) {
            Some(q) => q.to_string(),
            None => return Err(KgliteStatusCode::InvalidArgument),
        };
        let params = optional_object_map(obj, "params")?;
        out.push((query, params));
    }
    Ok(out)
}

/// Parse an `edges_json` argument into `EdgeSpec`s. Expects a JSON array
/// of objects with `src_id`, `src_type`, `dst_id`, `dst_type`, `type`
/// (the edge type) and optional `props`. Any other shape →
/// `InvalidArgument`. Assumes `edges_json` is non-null (callers check).
fn parse_edge_specs(edges_json: *const c_char) -> Result<Vec<EdgeSpec>, KgliteStatusCode> {
    let s = match unsafe { CStr::from_ptr(edges_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return Err(KgliteStatusCode::InvalidUtf8),
    };
    let parsed: serde_json::Value = match serde_json::from_str(s) {
        Ok(v) => v,
        Err(_) => return Err(KgliteStatusCode::InvalidArgument),
    };
    let arr = match parsed.as_array() {
        Some(a) => a,
        None => return Err(KgliteStatusCode::InvalidArgument),
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = match item.as_object() {
            Some(o) => o,
            None => return Err(KgliteStatusCode::InvalidArgument),
        };
        let req_str = |key: &str| -> Result<String, KgliteStatusCode> {
            obj.get(key)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .ok_or(KgliteStatusCode::InvalidArgument)
        };
        let req_id = |key: &str| -> Result<Value, KgliteStatusCode> {
            obj.get(key)
                .map(json_value_to_kglite_value)
                .ok_or(KgliteStatusCode::InvalidArgument)
        };
        let properties = optional_object_map(obj, "props")?;
        out.push(EdgeSpec {
            source_type: req_str("src_type")?,
            source_id: req_id("src_id")?,
            target_type: req_str("dst_type")?,
            target_id: req_id("dst_id")?,
            edge_type: req_str("type")?,
            properties,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn parse_params_empty_string_is_empty_map() {
        let s = CString::new("").unwrap();
        let m = parse_params_json(s.as_ptr()).unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn parse_params_object_round_trips() {
        let s = CString::new(r#"{"x": 42, "y": "hello"}"#).unwrap();
        let m = parse_params_json(s.as_ptr()).unwrap();
        assert_eq!(m.get("x"), Some(&Value::Int64(42)));
        assert_eq!(m.get("y"), Some(&Value::String("hello".to_string())));
    }

    #[test]
    fn parse_params_null_pointer_is_empty_map() {
        let m = parse_params_json(std::ptr::null()).unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn parse_params_array_is_invalid_argument() {
        let s = CString::new("[1, 2, 3]").unwrap();
        let err = parse_params_json(s.as_ptr()).unwrap_err();
        assert_eq!(err, KgliteStatusCode::InvalidArgument);
    }

    // ── kglite_create_edges_batch ────────────────────────────────────

    /// Build a session handle around a fresh in-memory graph. Callers
    /// free it via `kglite_session_free`.
    fn new_test_session() -> *mut KgliteSession {
        use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
        let graph = new_dir_graph_in_mode(StorageMode::Memory, None).expect("memory graph");
        SessionState::into_handle(Session::new(graph))
    }

    /// Run one mutating Cypher statement through the C ABI, asserting
    /// success.
    fn exec_mut(session: *mut KgliteSession, query: &str) {
        let q = CString::new(query).unwrap();
        let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let status = unsafe {
            kglite_session_execute_mut(session, q.as_ptr(), std::ptr::null(), &mut result, &mut err)
        };
        assert_eq!(status, KgliteStatusCode::Ok, "query failed: {query}");
        unsafe { crate::kglite_cypher_result_free(result) };
    }

    /// Count query helper: run `query` (must RETURN a single Int64
    /// column named anything) and return the first cell.
    fn count(session: *const KgliteSession, query: &str) -> i64 {
        let state = unsafe { SessionState::from_handle(session) };
        let snapshot = state.inner.snapshot();
        let params = HashMap::new();
        let opts = state.make_opts(&params);
        let outcome = execute_read(&snapshot, query, &opts).expect("count query");
        match outcome.result.rows.first().and_then(|r| r.first()) {
            Some(Value::Int64(n)) => *n,
            other => panic!("expected Int64 count, got {other:?}"),
        }
    }

    /// Call kglite_create_edges_batch and return (status, report-json,
    /// error-msg).
    fn edges_batch(
        session: *mut KgliteSession,
        edges_json: &str,
    ) -> (KgliteStatusCode, Option<serde_json::Value>, Option<String>) {
        let edges_c = CString::new(edges_json).unwrap();
        let mut report: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let status =
            unsafe { kglite_create_edges_batch(session, edges_c.as_ptr(), &mut report, &mut err) };
        let report_json = (!report.is_null()).then(|| {
            let s = unsafe { CStr::from_ptr(report) }.to_str().unwrap();
            let v = serde_json::from_str(s).unwrap();
            unsafe { crate::kglite_free_string(report) };
            v
        });
        let err_msg = (!err.is_null()).then(|| {
            let s = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
            unsafe { crate::kglite_free_string(err) };
            s
        });
        (status, report_json, err_msg)
    }

    #[test]
    fn create_edges_batch_lands_atomically_and_reports_errors() {
        let session = new_test_session();
        exec_mut(session, "CREATE (:Src {id: 1})");
        exec_mut(session, "CREATE (:Dst {id: 2})");

        // One valid edge + one missing endpoint: the valid edge lands,
        // the other is counted as skipped — one commit.
        let (status, report, err) = edges_batch(
            session,
            r#"[
                {"src_id": 1, "src_type": "Src", "dst_id": 2, "dst_type": "Dst", "type": "REL"},
                {"src_id": 99, "src_type": "Src", "dst_id": 2, "dst_type": "Dst", "type": "REL"}
            ]"#,
        );
        assert_eq!(status, KgliteStatusCode::Ok, "err: {err:?}");
        let report = report.expect("report json");
        assert_eq!(report["connections_created"], 1);
        assert_eq!(report["skipped_missing_endpoint"], 1);
        assert_eq!(
            count(session, "MATCH (:Src)-[r:REL]->(:Dst) RETURN count(r) AS c"),
            1
        );

        // Error path: an invalid spec (empty node type) must surface the
        // engine's error through the ABI error slot — not be discarded —
        // and land none of the batch.
        let (status, report, err) = edges_batch(
            session,
            r#"[
                {"src_id": 1, "src_type": "Src", "dst_id": 2, "dst_type": "Dst", "type": "REL2"},
                {"src_id": 1, "src_type": "", "dst_id": 2, "dst_type": "Dst", "type": "REL2"}
            ]"#,
        );
        assert_eq!(status, KgliteStatusCode::Internal);
        assert!(report.is_none(), "failed batch must not produce a report");
        assert!(
            err.is_some_and(|m| !m.is_empty()),
            "failed batch must report its error message"
        );
        assert_eq!(
            count(session, "MATCH ()-[r:REL2]->() RETURN count(r) AS c"),
            0,
            "failed batch must be atomic — no partial edges"
        );

        unsafe { kglite_session_free(session) };
    }

    #[test]
    fn create_edges_batch_serializes_with_concurrent_execute_mut() {
        // Regression test for the lost-update bug: create_edges_batch used
        // begin()+commit(check_occ=false), so its last-writer-wins Arc swap
        // silently discarded any execute_mut commit that landed between its
        // begin and commit. Routed through Session::transact, both writers
        // serialize on the Session lock and every committed write survives.
        const N: usize = 30;
        let session = new_test_session();
        exec_mut(session, "CREATE (:Src {id: 0})");
        for i in 0..N {
            exec_mut(session, &format!("CREATE (:Dst {{id: {i}}})"));
        }

        let addr = session as usize;
        let writer = std::thread::spawn(move || {
            let session = addr as *mut KgliteSession;
            for i in 0..N {
                exec_mut(session, &format!("CREATE (:P {{id: {i}}})"));
            }
        });

        for i in 0..N {
            let edges = format!(
                r#"[{{"src_id": 0, "src_type": "Src", "dst_id": {i}, "dst_type": "Dst", "type": "REL"}}]"#
            );
            let (status, report, err) = edges_batch(session, &edges);
            assert_eq!(status, KgliteStatusCode::Ok, "err: {err:?}");
            assert_eq!(report.expect("report")["connections_created"], 1);
        }
        writer.join().expect("writer thread panicked");

        assert_eq!(
            count(session, "MATCH (n:P) RETURN count(n) AS c"),
            N as i64,
            "no execute_mut commit may be lost to a concurrent edge batch"
        );
        assert_eq!(
            count(session, "MATCH ()-[r:REL]->() RETURN count(r) AS c"),
            N as i64,
            "no edge batch may be lost to a concurrent execute_mut"
        );

        unsafe { kglite_session_free(session) };
    }
}
