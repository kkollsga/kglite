//! `KgliteSession` opaque handle — session creation +
//! execute_read / execute_mut.
//!
//! The Session owns the graph after [`kglite_session_new`] — the
//! Arc moves in and the caller should NOT free the graph handle
//! afterwards.

use crate::graph::{GraphState, KgliteGraph};
use crate::result::{KgliteCypherResult, ResultState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::param::json_value_to_kglite_value;
use kglite::api::session::{execute_mut, execute_read, ExecuteOptions, Session};
use kglite::api::{Embedder, Value};
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::sync::Arc;

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
    pub(crate) embedder: Option<Arc<dyn Embedder>>,
}

impl SessionState {
    fn into_handle(session: Session) -> *mut KgliteSession {
        let boxed = Box::new(SessionState {
            inner: session,
            embedder: None,
        });
        Box::into_raw(boxed).cast::<KgliteSession>()
    }

    pub(crate) unsafe fn from_handle<'a>(handle: *const KgliteSession) -> &'a SessionState {
        unsafe { &*handle.cast::<SessionState>() }
    }

    pub(crate) unsafe fn from_handle_mut<'a>(handle: *mut KgliteSession) -> &'a mut SessionState {
        unsafe { &mut *handle.cast::<SessionState>() }
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
    if graph.is_null() || out_session.is_null() {
        return KgliteStatusCode::NullPointer;
    }
    // Safety: caller's contract — graph is a valid handle, not
    // yet freed. We MOVE the Arc out by reconstructing the Box
    // behind the opaque facade.
    let graph_state = unsafe { Box::from_raw(graph.cast::<GraphState>()) };
    let session = Session::from_arc(graph_state.inner);
    unsafe {
        *out_session = SessionState::into_handle(session);
    }
    KgliteStatusCode::Ok
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
    let mut opts = ExecuteOptions::eager(&params);
    opts.embedder = session_state.embedder.clone();

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
    let mut opts = ExecuteOptions::eager(&params);
    opts.embedder = session_state.embedder.clone();

    // Mirror the bolt-server execute_in_tx pattern: begin →
    // working_mut → execute_mut → commit. The Transaction's
    // working_mut lazily clones the snapshot's DirGraph for
    // mutation; commit atomically swaps it back via the Session
    // mutex.
    let mut tx = session_state.inner.begin();
    let exec_result = {
        let working = match tx.working_mut() {
            Ok(w) => w,
            Err(err) => {
                let code = KgliteStatusCode::from_kg_error_code(err.code());
                if !out_error_msg.is_null() {
                    unsafe {
                        *out_error_msg = alloc_c_string(&err.to_string());
                    }
                }
                unsafe {
                    *out_result = std::ptr::null_mut();
                }
                return code;
            }
        };
        execute_mut(working, query_str, &opts)
    };

    match exec_result {
        Ok(outcome) => {
            // Auto-commit. `check_occ = false` matches bolt-server's
            // current default — no inter-session OCC checking at
            // the C ABI surface in v1. Explicit OCC lands when a
            // binding actually needs it.
            let _ = session_state.inner.commit(tx, /*check_occ=*/ false);
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
}

/// Free a session handle. Idempotent on null (no-op).
///
/// # Safety
///
/// `session` must be either null or a valid pointer previously
/// returned by [`kglite_session_new`] and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_free(session: *mut KgliteSession) {
    unsafe { SessionState::free_handle(session) };
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
        serde_json::Value::Object(obj) => Ok(obj
            .into_iter()
            .map(|(k, v)| (k, json_value_to_kglite_value(&v)))
            .collect()),
        serde_json::Value::Null => Ok(HashMap::new()),
        _ => Err(KgliteStatusCode::InvalidArgument),
    }
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
}
