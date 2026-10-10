//! `KgliteSession` opaque handle — session creation +
//! execute_read / execute_mut.
//!
//! The Session owns the graph after a *successful*
//! [`kglite_session_new`] — the Arc moves in and the caller must not
//! free the graph handle afterwards. A failed call moves nothing and
//! leaves the handle the caller's to free.

use crate::cancel::KgliteCancelToken;
use crate::graph::{GraphState, KgliteGraph};
use crate::result::{result_to_json_object, KgliteCypherResult, ResultState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::mutation::{add_edges_from_specs, EdgeSpec};
use kglite::api::param::{
    json_object_to_query_value_map, json_object_to_value_map, json_text_to_query_value_map,
    json_value_to_kglite_value, validate_json_query_numbers_at, JsonQueryTextError,
};
use kglite::api::session::{execute_mut, execute_read, BackupOptions, ExecuteOptions, Session};
use kglite::api::{Embedder, Value};
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

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
    /// Whether JSON results render typed values as tagged objects instead of
    /// their natural JSON; set by [`kglite_session_set_result_encoding`]. Read
    /// when a result is created, so a result keeps the encoding in force at
    /// its own execution.
    pub(crate) tagged_results: AtomicBool,
    /// Serializes this handle's writers against each other: batches,
    /// relationship batches, ontology changes, explicit-transaction commits,
    /// non-durable statements and durable statements outside the group-commit
    /// queue hold it exclusively. A durable write runs on a fork and commits
    /// optimistically, so two such writers racing on one session could
    /// otherwise lose a commit to `TransactionConflict` (the Java and C
    /// contracts promise that writes serialize).
    ///
    /// A durable auto-commit statement that core's group-commit queue takes
    /// (`Session::auto_commit_is_grouped`) holds it *shared*: the queue
    /// serializes those against each other and lets concurrent ones share one
    /// log barrier, which an exclusive gate would forbid. Shared holders still
    /// exclude every exclusive holder, in both directions, so a transaction's
    /// begin-to-commit window never overlaps a queued statement's publish.
    /// Lock order: `write_gate` -> session graph -> durable log; released
    /// before the inline checkpoint so readers and the next writer do not wait
    /// on a log fold.
    pub(crate) write_gate: RwLock<()>,
    /// Path, read-only flag and writer lease of a session opened through
    /// [`kglite_open_session`](crate::kglite_open_session); inert for a
    /// session wrapped around a graph handle. Declared last so the session
    /// (and its write-ahead log) drops before the lease is released.
    pub(crate) life: crate::lifecycle::Lifecycle,
}

impl SessionState {
    fn into_handle(session: Session) -> *mut KgliteSession {
        Self::with_lifecycle(session, crate::lifecycle::Lifecycle::plain()).into_handle_boxed()
    }

    pub(crate) fn with_lifecycle(session: Session, life: crate::lifecycle::Lifecycle) -> Self {
        SessionState {
            inner: session,
            embedder: Mutex::new(None),
            tagged_results: AtomicBool::new(false),
            write_gate: RwLock::new(()),
            life,
        }
    }

    /// Hold [`Self::write_gate`] for one write. A panic inside a write that
    /// held it leaves no state the gate protects, so poisoning is ignored.
    pub(crate) fn write_gate(&self) -> std::sync::RwLockWriteGuard<'_, ()> {
        self.write_gate.write().unwrap_or_else(|p| p.into_inner())
    }

    /// Hold [`Self::write_gate`] shared, for a durable auto-commit statement
    /// that core's group-commit queue serializes.
    pub(crate) fn shared_write_gate(&self) -> std::sync::RwLockReadGuard<'_, ()> {
        self.write_gate.read().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn into_handle_boxed(self) -> *mut KgliteSession {
        Box::into_raw(Box::new(self)).cast::<KgliteSession>()
    }

    pub(crate) unsafe fn from_handle<'a>(handle: *const KgliteSession) -> &'a SessionState {
        unsafe { &*handle.cast::<SessionState>() }
    }

    /// Replace the session's embedder. Interior mutability (see the
    /// `embedder` field doc) — callers hold only `&SessionState`.
    pub(crate) fn tagged_results(&self) -> bool {
        self.tagged_results.load(Ordering::Relaxed)
    }

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
/// - `graph` (in, MOVED on success): graph handle. After a successful
///   call the pointer is no longer valid for any other use.
/// - `out_session` (out, owned): set to the session handle on
///   success; caller must free via [`kglite_session_free`].
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `graph` or `out_session` is null
///
/// **The graph handle is consumed only on `Ok`; on any error the caller
/// retains ownership and must still free it** with
/// [`kglite_graph_free`](crate::kglite_graph_free). The move happens after
/// argument validation, so a rejected call leaves the handle exactly as it
/// was — a binding that treats "moved" as unconditional leaks the graph on
/// every failed session open. Unlike the rest of the fallible surface this
/// takes no `out_error_msg`: the sole failure is a null argument, which has
/// no message beyond the code, and the parameter list is frozen for this ABI
/// major version.
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
///   parameter bindings. Pass null or `"{}"` for no params. Integer tokens at
///   any nesting depth must fit signed 64-bit; decimal/exponent tokens must fit
///   a finite 64-bit float.
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
/// validation. An unrepresentable numeric parameter returns
/// `KGLITE_STATUS_CODE_INVALID_ARGUMENT`; when `out_error_msg` is non-null, its
/// owned message identifies the nested parameter path. The caller frees that
/// message with [`kglite_free_string`](crate::kglite_free_string).
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
                Err(error) => return report_query_param_error(error, out_error_msg),
            };

            let session_state = unsafe { SessionState::from_handle(session) };
            let snapshot = session_state.inner.snapshot();
            let opts = session_state.make_opts(&params);

            match execute_read(&snapshot, query_str, &opts) {
                Ok(outcome) => {
                    unsafe {
                        *out_result = ResultState::into_handle(
                            outcome.result,
                            session_state.tagged_results(),
                        );
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
                    let code = KgliteStatusCode::from_kg_error(&err);
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
/// - `max_work_units`: work budget for the query, **not** a result-row cap.
///   It is charged against intermediate rows, retained collection items and
///   scan work — every quantity the executor holds or walks on the way to an
///   answer — so the count can far exceed the rows returned. Exceeding it
///   fails the query with an error; nothing is ever silently truncated to it.
///   Add a `LIMIT` clause to bound output. `0` = no explicit budget.
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
    max_work_units: u64,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    let limits = RunLimits {
        timeout_ms,
        max_work_units,
        row_limit: None,
        cancel: None,
    };
    unsafe {
        run_read(
            session,
            query,
            params_json,
            limits,
            out_result,
            out_error_msg,
        )
    }
}

/// Per-call budgets shared by the `_opts` and `_ex` execute symbols. `0` in
/// `timeout_ms` / `max_work_units` disables that budget; `row_limit` is a
/// retention cap, where `Some(0)` is legal and keeps no rows.
#[derive(Clone, Default)]
pub(crate) struct RunLimits {
    pub(crate) timeout_ms: u64,
    pub(crate) max_work_units: u64,
    pub(crate) row_limit: Option<u64>,
    /// A clone taken when the options were read, so the handle behind the
    /// caller's pointer can be freed while the call runs.
    pub(crate) cancel: Option<kglite::api::session::CancelToken>,
}

impl RunLimits {
    pub(crate) fn apply(&self, opts: &mut ExecuteOptions<'_>) {
        if self.timeout_ms > 0 {
            opts.set_timeout_ms(Some(self.timeout_ms));
        }
        if self.max_work_units > 0 {
            opts.max_work_units = Some(self.max_work_units as usize);
        }
        if let Some(limit) = self.row_limit {
            opts.row_limit = Some(limit as usize);
        }
        if let Some(token) = &self.cancel {
            opts.cancel = Some(token.clone());
        }
    }
}

unsafe fn run_read(
    session: *const KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    limits: RunLimits,
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
                Err(error) => return report_query_param_error(error, out_error_msg),
            };

            let session_state = unsafe { SessionState::from_handle(session) };
            let snapshot = session_state.inner.snapshot();
            let mut opts = session_state.make_opts(&params);
            limits.apply(&mut opts);

            let outcome = execute_read(&snapshot, query_str, &opts);
            finish_query(session_state, outcome, out_result, out_error_msg)
        },
    )
}

/// Publish one query outcome to the caller's out-slots.
pub(crate) fn finish_query(
    session_state: &SessionState,
    outcome: Result<kglite::api::session::ExecuteOutcome, kglite::api::KgError>,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    match outcome {
        Ok(outcome) => {
            unsafe {
                *out_result =
                    ResultState::into_handle(outcome.result, session_state.tagged_results());
            }
            KgliteStatusCode::Ok
        }
        Err(err) => {
            let code = KgliteStatusCode::from_kg_error(&err);
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
/// graph is auto-committed after a successful execute; use
/// [`kglite_session_begin`](crate::kglite_session_begin) for a multi-statement
/// transaction.
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

/// Run a mutating query with the same timeout and work-budget semantics as
/// [`kglite_session_execute_read_opts`] — `max_work_units` is a work budget
/// that errors when exceeded, not a result-row cap. A budget failure rolls
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
    max_work_units: u64,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    unsafe {
        execute_mut_impl(
            session,
            query,
            params_json,
            timeout_ms,
            max_work_units,
            out_result,
            out_error_msg,
        )
    }
}

/// Execution options for [`kglite_session_execute_read_ex`] and
/// [`kglite_session_execute_mut_ex`]. A versioned struct: the caller sets
/// `struct_size` to `sizeof(KgliteExecuteOptions)` as it was compiled, and the
/// library reads only that many bytes, treating every field beyond them as
/// zero. A field appended in a later release is therefore an additive change —
/// an older caller's struct keeps meaning exactly what it did. Zero-initialise
/// the struct, then set `struct_size` and the fields you want.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KgliteExecuteOptions {
    /// `sizeof(KgliteExecuteOptions)` in the caller's build. Must cover at
    /// least this field.
    pub struct_size: usize,
    /// Wall-clock budget in milliseconds; past it the query fails with
    /// `KGLITE_STATUS_CODE_CYPHER_TIMEOUT`. `0` = no deadline.
    pub timeout_ms: u64,
    /// Work budget, as in [`kglite_session_execute_read_opts`]: exceeding it
    /// fails the query. `0` = no explicit budget.
    pub max_work_units: u64,
    /// Result-row retention cap, honoured only when `flags` bit 0 (value `1`)
    /// is set. The query still runs to completion; only
    /// the rows kept stop at the cap, and the truncation is reported in the
    /// result's diagnostics JSON (`row_limit`, `total_rows`, a `warnings`
    /// entry). `0` with the flag set keeps no rows and reports the total.
    pub row_limit: u64,
    /// Bit set. Bit 0 (`1`): apply `row_limit`.
    pub flags: u32,
    /// Reserved; set to zero.
    pub reserved: u32,
    /// Cancellation token from [`kglite_cancel_token_new`]; null for none.
    /// Read only when `struct_size` covers this field. The call takes its own
    /// reference before it starts, so the token may be freed while the call
    /// runs. Cancelling it makes the call return `KGLITE_STATUS_CODE_CANCELLED`.
    pub cancel: *const KgliteCancelToken,
}

/// `KgliteExecuteOptions.flags` bit 0: apply `row_limit`. Not exported to the
/// header (cbindgen is configured for no constants); the field doc names the
/// value.
const EXECUTE_ROW_LIMIT: u32 = 1;

pub(crate) fn read_execute_options(
    options: *const KgliteExecuteOptions,
) -> Result<RunLimits, String> {
    if options.is_null() {
        return Ok(RunLimits::default());
    }
    let declared = unsafe { std::ptr::addr_of!((*options).struct_size).read_unaligned() };
    if declared < std::mem::size_of::<usize>() {
        return Err("KgliteExecuteOptions.struct_size is smaller than its first field".to_string());
    }
    // SAFETY: zero is a valid bit pattern for every field.
    let mut local: KgliteExecuteOptions = unsafe { std::mem::zeroed() };
    let take = declared.min(std::mem::size_of::<KgliteExecuteOptions>());
    unsafe {
        std::ptr::copy_nonoverlapping(
            options.cast::<u8>(),
            std::ptr::addr_of_mut!(local).cast::<u8>(),
            take,
        );
    }
    let cancel_end = std::mem::offset_of!(KgliteExecuteOptions, cancel)
        + std::mem::size_of::<*const KgliteCancelToken>();
    let cancel = (declared >= cancel_end && !local.cancel.is_null())
        .then(|| unsafe { (*local.cancel).clone_token() });
    Ok(RunLimits {
        cancel,
        timeout_ms: local.timeout_ms,
        max_work_units: local.max_work_units,
        row_limit: (local.flags & EXECUTE_ROW_LIMIT != 0).then_some(local.row_limit),
    })
}

/// [`kglite_session_execute_read`] with a [`KgliteExecuteOptions`] block:
/// timeout, work budget and a result-row cap that truncates with a report
/// rather than failing. `options` may be null (no limits). A block whose
/// `struct_size` is smaller than its first field is
/// `KGLITE_STATUS_CODE_INVALID_ARGUMENT`.
///
/// # Safety
///
/// As [`kglite_session_execute_read`]; `options` null or a valid pointer to at
/// least `options->struct_size` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_execute_read_ex(
    session: *const KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    options: *const KgliteExecuteOptions,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    match read_execute_options(options) {
        Ok(limits) => unsafe {
            run_read(
                session,
                query,
                params_json,
                limits,
                out_result,
                out_error_msg,
            )
        },
        Err(message) => unsafe { reject_options(out_result, out_error_msg, &message) },
    }
}

/// [`kglite_session_execute_mut`] with a [`KgliteExecuteOptions`] block. The
/// row cap bounds only the rows the trailing `RETURN` reports; every write
/// still happens. On a session opened by
/// [`kglite_open_session`](crate::kglite_open_session) the statement is
/// write-ahead logged at the session's durability level, and a read-only or
/// closed session refuses it (`KGLITE_STATUS_CODE_READ_ONLY` for the former).
///
/// # Safety
///
/// As [`kglite_session_execute_mut`]; `options` as for
/// [`kglite_session_execute_read_ex`].
#[no_mangle]
pub unsafe extern "C" fn kglite_session_execute_mut_ex(
    session: *mut KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    options: *const KgliteExecuteOptions,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    match read_execute_options(options) {
        Ok(limits) => unsafe {
            run_mut(
                session,
                query,
                params_json,
                limits,
                out_result,
                out_error_msg,
            )
        },
        Err(message) => unsafe { reject_options(out_result, out_error_msg, &message) },
    }
}

pub(crate) unsafe fn reject_options(
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
    message: &str,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_result, std::ptr::null_mut()),
        || {
            crate::ffi::init_out(out_error_msg, alloc_c_string(message));
            KgliteStatusCode::InvalidArgument
        },
    )
}

// The arity is the published C ABI's: this is the shared body of
// kglite_session_execute_mut and _mut_opts, so its parameters are exactly the
// wider exported signature and cannot be grouped into a struct without
// changing what those symbols take.
#[allow(clippy::too_many_arguments)]
unsafe fn execute_mut_impl(
    session: *mut KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    timeout_ms: u64,
    max_work_units: u64,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    let limits = RunLimits {
        timeout_ms,
        max_work_units,
        row_limit: None,
        cancel: None,
    };
    unsafe {
        run_mut(
            session,
            query,
            params_json,
            limits,
            out_result,
            out_error_msg,
        )
    }
}

/// Attempts a durable session's auto-committed statement makes while its
/// commit loses an optimistic race against another writer.
const DURABLE_WRITE_ATTEMPTS: u32 = 3;

unsafe fn run_mut(
    session: *mut KgliteSession,
    query: *const c_char,
    params_json: *const c_char,
    limits: RunLimits,
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
                Err(error) => return report_query_param_error(error, out_error_msg),
            };

            // The ABI signature is `*mut`, but the handle is only borrowed
            // shared here: the mutation is serialized by the Session's own
            // lock, taken below.
            let session_state = unsafe { SessionState::from_handle(session) };
            if let Err(refusal) = session_state.guard_write() {
                return crate::lifecycle::refuse(out_error_msg, refusal);
            }
            let mut opts = session_state.make_opts(&params);
            limits.apply(&mut opts);

            let outcome = if session_state.inner.durability().is_some() {
                // A durable session's mutations must reach its write-ahead
                // log, which only `begin`/`commit` writes; the direct write
                // guard below would latch the log as diverged.
                let outcome = {
                    // Statements core serializes itself share the gate (and a
                    // log barrier); the rest commit optimistically and must
                    // not overlap another writer.
                    let grouped = session_state.inner.auto_commit_is_grouped(query_str, &opts);
                    let _shared = grouped.then(|| session_state.shared_write_gate());
                    let _exclusive = (!grouped).then(|| session_state.write_gate());
                    session_state.inner.execute_auto_commit(
                        query_str,
                        &opts,
                        DURABLE_WRITE_ATTEMPTS,
                    )
                };
                if outcome.is_ok() {
                    crate::lifecycle::auto_checkpoint(&session_state.inner);
                }
                outcome
            } else {
                // Hold the core Session write guard across execution. This
                // serializes the complete mutation (preventing last-writer-loses
                // races) and reaches the unique-owner path without the old
                // redundant working-copy clone. `execute_mut` rolls its own
                // statement checkpoint back on error, so the graph under the
                // guard is unmutated on failure.
                let _gate = session_state.write_gate();
                let mut working = session_state.inner.write();
                execute_mut(&mut working, query_str, &opts)
            };
            finish_query(session_state, outcome, out_result, out_error_msg)
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
/// issues many small reads. Each `params` object follows the exact numeric
/// admission and owned-error-message contract of
/// [`kglite_session_execute_read`]; its message also identifies the batch
/// entry.
///
/// On success `out_results_json` is set to an owned JSON string: an
/// array of `{"columns": [...], "rows": [{...}], "diagnostics": {...}}` objects, one per input
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
                Err(error) => return report_query_param_error(error, out_error_msg),
            };
            let session_state = unsafe { SessionState::from_handle(session) };
            let snapshot = session_state.inner.snapshot();
            let mut results = Vec::with_capacity(queries.len());
            for (query, params) in &queries {
                let opts = session_state.make_opts(params);
                match execute_read(&snapshot, query, &opts) {
                    Ok(outcome) => results.push(result_to_json_object(
                        &outcome.result,
                        session_state.tagged_results(),
                    )),
                    Err(err) => {
                        unsafe {
                            *out_results_json = std::ptr::null();
                        }
                        let code = KgliteStatusCode::from_kg_error(&err);
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
                Err(error) => return report_query_param_error(error, out_error_msg),
            };
            let session_state = unsafe { SessionState::from_handle(session) };
            if let Err(refusal) = session_state.guard_write() {
                return crate::lifecycle::refuse(out_error_msg, refusal);
            }
            let run_all = |working: &mut kglite::api::DirGraph| {
                let mut results = Vec::with_capacity(queries.len());
                for (query, params) in &queries {
                    let opts = session_state.make_opts(params);
                    let outcome = execute_mut(working, query, &opts).map_err(Box::new)?;
                    results.push(result_to_json_object(
                        &outcome.result,
                        session_state.tagged_results(),
                    ));
                }
                Ok(results)
            };
            let transaction: Result<Vec<serde_json::Value>, Box<kglite::api::KgError>> =
                if session_state.inner.durability().is_some() {
                    crate::lifecycle::durable_transaction(session_state, run_all, Box::new)
                } else {
                    let _gate = session_state.write_gate();
                    session_state.inner.transact(run_all)
                };
            let results = match transaction {
                Ok(results) => results,
                Err(err) => {
                    unsafe {
                        *out_results_json = std::ptr::null();
                    }
                    let code = KgliteStatusCode::from_kg_error(&err);
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
/// A batch that would leave a relationship violating a declared
/// relationship constraint (`IS NOT NULL`, `IS :: <type>`) is refused
/// whole with `KgliteStatusCode::ConstraintViolation`.
///
/// On success `out_report_json` is set to an owned JSON object
/// `{"connections_created": N, "connections_updated": U,
/// "skipped_missing_endpoint": M, "warnings": [...]}`, where
/// `connections_updated` counts specs that met an existing edge of the same
/// type between the same endpoints and merged their properties into it, and
/// `warnings` lists advisories about the edges written (such as edges whose
/// validity interval is empty under a `half_open` declaration; empty when
/// there are none); free it with
/// [`kglite_free_string`](crate::kglite_free_string).
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
            if let Err(refusal) = session_state.guard_write() {
                return crate::lifecycle::refuse(out_error_msg, refusal);
            }
            // Route through `Session::transact` so the whole batch runs
            // under the Session write lock: serialized with concurrent
            // execute_mut writers (no last-writer-wins Arc-swap losing
            // their commits) and atomic — an error drops the fork with
            // no partial writes.
            // A refusal by a declared constraint is taken off the fork before
            // `transact` drops it, so it surfaces as `ConstraintViolation`.
            let add_edges = |working: &mut kglite::api::DirGraph| {
                add_edges_from_specs(working, specs).map_err(|message| {
                    let typed = working.take_constraint_error(&message).map(Box::new);
                    (message, typed)
                })
            };
            let transaction: Result<_, (String, Option<Box<kglite::api::KgError>>)> =
                if session_state.inner.durability().is_some() {
                    // Logged: a durable session's edges must reach its log.
                    crate::lifecycle::durable_transaction(session_state, add_edges, |e| {
                        (e.to_string(), Some(Box::new(e)))
                    })
                } else {
                    let _gate = session_state.write_gate();
                    session_state.inner.transact(add_edges)
                };
            match transaction {
                Ok(report) => {
                    let json = serde_json::json!({
                        "connections_created": report.connections_created,
                        "connections_updated": report.connections_updated,
                        "skipped_missing_endpoint": report.skipped_missing_endpoint,
                        "warnings": report.warnings,
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
                Err((message, typed)) => {
                    unsafe {
                        *out_report_json = std::ptr::null();
                    }
                    let (code, message) = match typed {
                        Some(err) => (KgliteStatusCode::from_kg_error(&err), err.to_string()),
                        None => (KgliteStatusCode::Internal, message),
                    };
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&message);
                        }
                    }
                    code
                }
            }
        },
    )
}

/// Checkpoint a session's graph to `path` — the save half of the
/// open / mutate / save cycle.
///
/// [`kglite_session_new`] takes ownership of the graph handle, so a graph
/// mutated through
/// [`kglite_session_execute_mut`](crate::kglite_session_execute_mut) can only
/// be persisted from the session that now holds it; this is that call.
/// [`kglite_save_graph`](crate::kglite_save_graph) remains the entry point
/// for a graph handle that has *not* been moved into a session.
///
/// **The lease contract: a caller that saves must hold the writer lease
/// across the whole open / mutate / save interval, not merely at this call.**
/// Take it with
/// [`kglite_writer_lease_acquire`](crate::kglite_writer_lease_acquire) before
/// opening, free it after saving. Two processes that both open one path, both
/// mutate, and both save each write a complete snapshot and the later one
/// wins outright and silently — locking only at save time is already too late
/// to notice. Read-only sessions take no lease.
///
/// The save writes through the session's own graph, so it never copies the
/// graph to checkpoint it, and it is serialized against concurrent
/// `execute_mut` calls on the same session: the file is a consistent
/// point-in-time image. Readers holding a result from before the save are
/// unaffected.
///
/// `fsync` != 0 is the durable default: atomic temp+rename plus a file and
/// parent-directory flush, so the checkpoint survives power loss. `fsync` ==
/// 0 is the fast, **non-durable** opt-out — still never a torn file, but the
/// bytes may not survive an OS crash. The storage mode is written from the
/// graph being saved, so reopening with
/// [`kglite_open_or_create_graph_in_mode`](crate::kglite_open_or_create_graph_in_mode)
/// and a null mode brings the graph back in the mode it was saved in.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `session` or `path` is null
/// - `KGLITE_STATUS_CODE_INVALID_UTF8` — `path` isn't valid UTF-8
/// - `KGLITE_STATUS_CODE_FILE_IO` — the write failed
///
/// # Safety
///
/// `session` must be a valid handle from [`kglite_session_new`], not yet
/// freed; `path` a null-terminated UTF-8 string; `out_error_msg` null or a
/// valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_save(
    session: *mut KgliteSession,
    path: *const c_char,
    fsync: u8,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {},
        || {
            if session.is_null() || path.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let session_state = unsafe { SessionState::from_handle(session) };
            // `Session::save` reaches the session's own Arc under its lock —
            // the only no-copy route, since saving a `snapshot()` would find a
            // second strong reference and deep-clone the whole graph.
            match session_state.inner.save(path_str, fsync != 0) {
                Ok(()) => KgliteStatusCode::Ok,
                Err(message) => {
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&message);
                        }
                    }
                    KgliteStatusCode::FileIo
                }
            }
        },
    )
}

/// Write a consistent single-file `.kgl` backup of a session's published graph
/// to `dest` while writers keep committing.
///
/// Unlike [`kglite_session_save`](crate::kglite_session_save), a backup is an
/// independent copy: it takes no writer lease, creates no `-wal` sidecar, and
/// does not touch the session's checkpoint. Memory and mapped graphs are
/// supported; a disk-mode graph is refused. An existing `dest` is replaced
/// atomically.
///
/// `live_path` (nullable) is the file the graph was opened from, when the
/// caller has one. A session cannot know it, so pass it to have a backup over
/// the live file refused; a durable session also detects its own log sidecar.
///
/// On success `out_report_json` is an owned JSON object: `path`, `bytes`,
/// `nodes`, `relationships`, `graph_version`, `lsn` (null for a session
/// without a write-ahead log), `lock_hold_ms`, `elapsed_ms`,
/// `prepared_copy` (true when the snapshot needed a private prepared copy
/// first). Free it with
/// [`kglite_free_string`](crate::kglite_free_string). On failure it is null.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `session`, `dest` or `out_report_json` is null
/// - `KGLITE_STATUS_CODE_INVALID_UTF8` — `dest` or `live_path` isn't valid UTF-8
/// - `KGLITE_STATUS_CODE_FILE_IO` — the write failed, or the backup was
///   refused (destination aliases the live checkpoint, disk-mode graph);
///   the message says which
///
/// # Safety
///
/// `session` must be a valid handle from [`kglite_session_new`], not yet
/// freed; `dest` a null-terminated UTF-8 string; `live_path` null or a
/// null-terminated UTF-8 string; `out_report_json` a valid writable slot;
/// `out_error_msg` null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_backup(
    session: *const KgliteSession,
    dest: *const c_char,
    live_path: *const c_char,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_report_json, std::ptr::null()),
        || {
            if session.is_null() || dest.is_null() || out_report_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let Ok(dest_str) = unsafe { CStr::from_ptr(dest) }.to_str() else {
                return KgliteStatusCode::InvalidUtf8;
            };
            let live = if live_path.is_null() {
                None
            } else {
                match unsafe { CStr::from_ptr(live_path) }.to_str() {
                    Ok(s) => Some(std::path::PathBuf::from(s)),
                    Err(_) => return KgliteStatusCode::InvalidUtf8,
                }
            };
            let session_state = unsafe { SessionState::from_handle(session) };
            let opts = BackupOptions { live_path: live };
            match session_state
                .inner
                .backup(std::path::Path::new(dest_str), &opts)
            {
                Ok(report) => {
                    let json = serde_json::json!({
                        "path": report.path.to_string_lossy(),
                        "bytes": report.bytes,
                        "nodes": report.nodes,
                        "relationships": report.relationships,
                        "graph_version": report.graph_version,
                        "lsn": report.lsn,
                        "lock_hold_ms": report.lock_hold.as_secs_f64() * 1000.0,
                        "elapsed_ms": report.elapsed.as_secs_f64() * 1000.0,
                        "prepared_copy": report.prepared_copy,
                    })
                    .to_string();
                    unsafe { *out_report_json = alloc_c_string(&json) };
                    KgliteStatusCode::Ok
                }
                Err(error) => {
                    if !out_error_msg.is_null() {
                        unsafe { *out_error_msg = alloc_c_string(&error.to_string()) };
                    }
                    KgliteStatusCode::FileIo
                }
            }
        },
    )
}

/// How a session's JSON results spell values JSON has no type for.
/// Pass one of these to [`kglite_session_set_result_encoding`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KgliteResultEncoding {
    /// Natural JSON, the default: a date is a string, a point is
    /// `{"latitude", "longitude"}`, and NaN or an infinity is `null`.
    Natural = 0,
    /// Typed values as one-key tagged objects, the same tags a query
    /// parameter accepts: `{"$date": "2020-01-01"}`,
    /// `{"$datetime": "2020-01-02T03:04:05.250"}`,
    /// `{"$duration": {"months": 0, "days": 1, "seconds": 0}}`,
    /// `{"$point": {"lat": 60.1, "lon": 5.2}}` and
    /// `{"$float": "NaN" | "inf" | "-inf"}`. A map whose only key is a tag name
    /// is wrapped as `{"$map": {...}}` so it is not read as that tag.
    Tagged = 1,
}

/// Choose how this session's JSON results spell typed values.
///
/// With [`KgliteResultEncoding::Natural`] (the default for a new session)
/// results are the JSON every earlier release produced. With
/// [`KgliteResultEncoding::Tagged`] a date, datetime, duration, point and every
/// non-finite float render as the tagged objects documented on that enum, at
/// any depth, including node and relationship properties. They are the tags a
/// query parameter accepts, so a result cell read back and bound again is
/// unchanged. The setting governs
/// [`kglite_cypher_result_rows_json`](crate::kglite_cypher_result_rows_json)
/// of results produced afterwards and the rows of the batch-execute results; a
/// result already returned keeps the encoding it was created with. Ids,
/// strings, integers, finite floats (including `-0.0`) and booleans are
/// unaffected.
///
/// Returns `KGLITE_STATUS_CODE_INVALID_ARGUMENT` for a value that is not a
/// `KgliteResultEncoding`, and `KGLITE_STATUS_CODE_NULL_POINTER` for a null
/// `session`.
///
/// # Safety
///
/// `session` must be a valid session pointer not yet freed.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_set_result_encoding(
    session: *const KgliteSession,
    encoding: u32,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        std::ptr::null_mut(),
        || {},
        || {
            if session.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let tagged = match encoding {
                0 => false,
                1 => true,
                _ => return KgliteStatusCode::InvalidArgument,
            };
            let state = unsafe { SessionState::from_handle(session) };
            state.tagged_results.store(tagged, Ordering::Relaxed);
            KgliteStatusCode::Ok
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
    /// defaults with the streaming aggregate pipeline on (its rows are
    /// materialized too), plus the session's embedder. Centralized so the
    /// read / mut / batch paths can't drift on per-call option defaults.
    pub(crate) fn make_opts<'a>(&self, params: &'a HashMap<String, Value>) -> ExecuteOptions<'a> {
        let mut opts = ExecuteOptions::eager(params);
        opts.streaming = true;
        opts.embedder = self
            .embedder
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        opts
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct QueryParamDecodeError {
    code: KgliteStatusCode,
    message: String,
}

impl QueryParamDecodeError {
    fn new(code: KgliteStatusCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

pub(crate) fn report_query_param_error(
    error: QueryParamDecodeError,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    if !out_error_msg.is_null() {
        unsafe {
            *out_error_msg = alloc_c_string(&error.message);
        }
    }
    error.code
}

/// Parse a JSON-string params argument into a HashMap. Null / empty /
/// `null` / `{}` → empty map; a JSON object → its converted map. Any other
/// top-level shape (array, scalar) → `InvalidArgument`.
pub(crate) fn parse_params_json(
    params_json: *const c_char,
) -> Result<HashMap<String, Value>, QueryParamDecodeError> {
    if params_json.is_null() {
        return Ok(HashMap::new());
    }
    let s = match unsafe { CStr::from_ptr(params_json) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            return Err(QueryParamDecodeError::new(
                KgliteStatusCode::InvalidUtf8,
                "params_json is not valid UTF-8",
            ));
        }
    };
    if s.is_empty() {
        return Ok(HashMap::new());
    }
    match json_text_to_query_value_map(s) {
        Ok(params) => Ok(params),
        Err(JsonQueryTextError::TopLevelNull) => Ok(HashMap::new()),
        Err(JsonQueryTextError::Syntax(message)) => Err(QueryParamDecodeError::new(
            KgliteStatusCode::InvalidArgument,
            format!("params_json is not valid JSON: {message}"),
        )),
        Err(JsonQueryTextError::TopLevelNotAnObject) => Err(QueryParamDecodeError::new(
            KgliteStatusCode::InvalidArgument,
            "params_json must be a JSON object or null",
        )),
        Err(error @ JsonQueryTextError::Parameter(_)) => Err(QueryParamDecodeError::new(
            KgliteStatusCode::InvalidArgument,
            error.to_string(),
        )),
    }
}

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

fn optional_query_params_map(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<HashMap<String, Value>, QueryParamDecodeError> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(HashMap::new()),
        Some(serde_json::Value::Object(params)) => {
            json_object_to_query_value_map(params).map_err(|error| {
                QueryParamDecodeError::new(KgliteStatusCode::InvalidArgument, error.to_string())
            })
        }
        Some(_) => Err(QueryParamDecodeError::new(
            KgliteStatusCode::InvalidArgument,
            format!("{key} must be a JSON object or null"),
        )),
    }
}

type BatchQuery = (String, HashMap<String, Value>);

/// Parse a batch `queries_json` argument into `(query, params)` pairs — the
/// shape published on [`kglite_session_execute_read_batch`]. Any other shape →
/// `InvalidArgument`. Assumes `queries_json` is non-null (callers check).
fn parse_batch_queries(
    queries_json: *const c_char,
) -> Result<Vec<BatchQuery>, QueryParamDecodeError> {
    let s = match unsafe { CStr::from_ptr(queries_json) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            return Err(QueryParamDecodeError::new(
                KgliteStatusCode::InvalidUtf8,
                "queries_json is not valid UTF-8",
            ));
        }
    };
    validate_json_query_numbers_at(s, &["[]", "params"]).map_err(|error| {
        QueryParamDecodeError::new(KgliteStatusCode::InvalidArgument, error.to_string())
    })?;
    let parsed: serde_json::Value = match serde_json::from_str(s) {
        Ok(v) => v,
        Err(error) => {
            return Err(QueryParamDecodeError::new(
                KgliteStatusCode::InvalidArgument,
                format!("queries_json is not valid JSON: {error}"),
            ));
        }
    };
    let arr = match parsed.as_array() {
        Some(a) => a,
        None => {
            return Err(QueryParamDecodeError::new(
                KgliteStatusCode::InvalidArgument,
                "queries_json must be a JSON array",
            ));
        }
    };
    let mut out = Vec::with_capacity(arr.len());
    for (index, item) in arr.iter().enumerate() {
        let obj = match item.as_object() {
            Some(o) => o,
            None => {
                return Err(QueryParamDecodeError::new(
                    KgliteStatusCode::InvalidArgument,
                    format!("queries_json[{index}] must be a JSON object"),
                ));
            }
        };
        let query = match obj.get("query").and_then(|v| v.as_str()) {
            Some(q) => q.to_string(),
            None => {
                return Err(QueryParamDecodeError::new(
                    KgliteStatusCode::InvalidArgument,
                    format!("queries_json[{index}].query must be a string"),
                ));
            }
        };
        let params = optional_query_params_map(obj, "params").map_err(|error| {
            QueryParamDecodeError::new(
                error.code,
                format!("queries_json[{index}].params: {}", error.message),
            )
        })?;
        out.push((query, params));
    }
    Ok(out)
}

/// Parse an `edges_json` argument into `EdgeSpec`s — the shape published on
/// [`kglite_create_edges_batch`]. Any other shape → `InvalidArgument`.
/// Assumes `edges_json` is non-null (callers check).
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
        assert_eq!(err.code, KgliteStatusCode::InvalidArgument);
    }

    /// The published `params_json` shapes, which the shared text entry
    /// reports as separate variants: a JSON `null` document means "no
    /// parameters", any other non-object is refused by name, and a syntax
    /// error keeps serde_json's own diagnostic.
    #[test]
    fn parse_params_document_null_is_empty_and_other_shapes_keep_their_message() {
        let null = CString::new("null").unwrap();
        assert!(parse_params_json(null.as_ptr()).unwrap().is_empty());
        for raw in ["[1, 2, 3]", "42", r#""text""#, "true"] {
            let s = CString::new(raw).unwrap();
            let err = parse_params_json(s.as_ptr()).unwrap_err();
            assert_eq!(err.code, KgliteStatusCode::InvalidArgument);
            assert_eq!(err.message, "params_json must be a JSON object or null");
        }
        let broken = CString::new("{").unwrap();
        let err = parse_params_json(broken.as_ptr()).unwrap_err();
        assert_eq!(err.code, KgliteStatusCode::InvalidArgument);
        assert!(
            err.message.starts_with("params_json is not valid JSON: "),
            "{}",
            err.message
        );
    }

    /// The documented ownership-on-failure contract: a rejected
    /// `kglite_session_new` must NOT have consumed the graph, so the caller's
    /// handle is still live and still theirs to free. If the move ever crept
    /// above the argument validation this frees an already-dropped Box, which
    /// the sanitizer/leak jobs report — and a binding following the header
    /// would leak on every failed open.
    #[test]
    fn session_new_failure_leaves_the_graph_handle_with_the_caller() {
        let graph = crate::kglite_graph_new();
        assert!(!graph.is_null());
        let rc = unsafe { kglite_session_new(graph, std::ptr::null_mut()) };
        assert_eq!(rc, KgliteStatusCode::NullPointer);
        // Still ours: usable, then freeable exactly once.
        let mut mode: *const c_char = std::ptr::null();
        let probe = unsafe {
            crate::kglite_graph_storage_mode(graph, &mut mode as *mut _, std::ptr::null_mut())
        };
        assert_eq!(probe, KgliteStatusCode::Ok, "the handle must still be live");
        unsafe { crate::kglite_free_string(mode) };
        unsafe { crate::kglite_graph_free(graph) };
    }

    /// Build a session handle around a fresh in-memory graph. Callers
    /// free it via `kglite_session_free`.
    fn new_test_session() -> *mut KgliteSession {
        use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
        let graph = new_dir_graph_in_mode(StorageMode::Memory, None).expect("memory graph");
        SessionState::into_handle(Session::new(graph))
    }

    /// Run one mutating Cypher statement through the C ABI, asserting success.
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

        // The same endpoints again merge into that edge: an update, not a creation.
        let (status, report, err) = edges_batch(
            session,
            r#"[{"src_id": 1, "src_type": "Src", "dst_id": 2, "dst_type": "Dst", "type": "REL", "props": {"w": 2}}]"#,
        );
        assert_eq!(status, KgliteStatusCode::Ok, "err: {err:?}");
        let report = report.expect("report json");
        assert_eq!(report["connections_created"], 0);
        assert_eq!(report["connections_updated"], 1);
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
    fn session_save_rejects_null_arguments() {
        let mut error: *const c_char = std::ptr::NonNull::<c_char>::dangling().as_ptr();
        let status =
            unsafe { kglite_session_save(std::ptr::null_mut(), std::ptr::null(), 1, &mut error) };
        assert_eq!(status, KgliteStatusCode::NullPointer);
        assert!(
            error.is_null(),
            "the error slot must be reset before validation"
        );
    }

    /// A failed checkpoint has to reach the caller as an error with a message,
    /// not a silent success — the one outcome that would let a binding report
    /// "saved" for data that is not on disk.
    #[test]
    fn session_save_surfaces_a_write_failure() {
        let session = new_test_session();
        exec_mut(session, "CREATE (:T {id: 1})");
        // A path whose parent directory does not exist: the write cannot
        // succeed, and the engine's message says why.
        let path = CString::new("/nonexistent-kglite-c-dir/inner/graph.kgl").unwrap();
        let mut error: *const c_char = std::ptr::null();
        let status = unsafe { kglite_session_save(session, path.as_ptr(), 0, &mut error) };
        assert_eq!(status, KgliteStatusCode::FileIo);
        assert!(!error.is_null(), "a failed save must explain itself");
        unsafe { crate::kglite_free_string(error) };
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

    #[test]
    fn query_json_rejects_unrepresentable_numbers_without_changing_edge_ingestion() {
        for raw in [
            r#"{"value":1267650600228229401496703205376}"#,
            r#"{"nested":{"value":-1267650600228229401496703205376}}"#,
            r#"{"value":1e400}"#,
        ] {
            let params = CString::new(raw).unwrap();
            assert_eq!(
                parse_params_json(params.as_ptr()).unwrap_err().code,
                KgliteStatusCode::InvalidArgument
            );
        }

        let batch = CString::new(
            r#"[{"query":"RETURN $value","params":{"value":1267650600228229401496703205376}}]"#,
        )
        .unwrap();
        assert_eq!(
            parse_batch_queries(batch.as_ptr()).unwrap_err().code,
            KgliteStatusCode::InvalidArgument
        );

        let edge: serde_json::Value =
            serde_json::from_str(r#"{"props":{"value":1267650600228229401496703205376}}"#).unwrap();
        let props = optional_object_map(edge.as_object().unwrap(), "props").unwrap();
        assert!(matches!(props["value"], Value::Float64(_)));
    }

    fn assert_query_param_path(
        invoke: impl FnOnce(*mut *mut KgliteCypherResult, *mut *const c_char) -> KgliteStatusCode,
    ) {
        let mut result = std::ptr::dangling_mut();
        let mut error = std::ptr::null();
        let status = invoke(&mut result, &mut error);
        assert_eq!(status, KgliteStatusCode::InvalidArgument);
        assert!(result.is_null());
        assert!(!error.is_null(), "parameter refusal must explain itself");
        let message = unsafe { CStr::from_ptr(error) }
            .to_str()
            .unwrap()
            .to_string();
        assert!(message.contains("$.outer[0].value"), "{message}");
        unsafe { crate::kglite_free_string(error) };
    }

    #[test]
    fn every_c_query_entry_reports_and_owns_strict_parameter_errors() {
        let session = new_test_session();
        let query = CString::new("RETURN $outer").unwrap();
        let mutating_query = CString::new("CREATE (:Rejected {id: $outer})").unwrap();
        let params_json = r#"{"outer":[{"value":1267650600228229401496703205376}]}"#;
        let params = CString::new(params_json).unwrap();

        assert_query_param_path(|result, error| unsafe {
            kglite_session_execute_read(session, query.as_ptr(), params.as_ptr(), result, error)
        });
        assert_query_param_path(|result, error| unsafe {
            kglite_session_execute_read_opts(
                session,
                query.as_ptr(),
                params.as_ptr(),
                0,
                0,
                result,
                error,
            )
        });
        assert_query_param_path(|result, error| unsafe {
            kglite_session_execute_mut(
                session,
                mutating_query.as_ptr(),
                params.as_ptr(),
                result,
                error,
            )
        });
        assert_query_param_path(|result, error| unsafe {
            kglite_session_execute_mut_opts(
                session,
                mutating_query.as_ptr(),
                params.as_ptr(),
                0,
                0,
                result,
                error,
            )
        });

        let batch = CString::new(format!(
            r#"[{{"query":"RETURN $outer","params":{}}}]"#,
            params_json
        ))
        .unwrap();
        let mut output = std::ptr::null();
        let mut error = std::ptr::null();
        let status = unsafe {
            kglite_session_execute_read_batch(session, batch.as_ptr(), &mut output, &mut error)
        };
        assert_eq!(status, KgliteStatusCode::InvalidArgument);
        assert!(output.is_null());
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }
            .to_str()
            .unwrap()
            .to_string();
        assert!(message.contains("$[0].outer[0].value"), "{message}");
        unsafe { crate::kglite_free_string(error) };

        let mut output = std::ptr::null();
        let mut error = std::ptr::null();
        let status = unsafe {
            kglite_session_execute_mut_batch(session, batch.as_ptr(), &mut output, &mut error)
        };
        assert_eq!(status, KgliteStatusCode::InvalidArgument);
        assert!(output.is_null());
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }
            .to_str()
            .unwrap()
            .to_string();
        assert!(message.contains("$[0].outer[0].value"), "{message}");
        unsafe { crate::kglite_free_string(error) };

        let mut result = std::ptr::dangling_mut();
        let status = unsafe {
            kglite_session_execute_read(
                session,
                query.as_ptr(),
                params.as_ptr(),
                &mut result,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, KgliteStatusCode::InvalidArgument);
        assert!(result.is_null());

        assert_eq!(
            count(session, "MATCH (n:Rejected) RETURN count(n) AS c"),
            0,
            "failed admission must happen before either write entry mutates"
        );
        let valid_query = CString::new("CREATE (:Accepted {id: $id})").unwrap();
        let valid_params = CString::new(r#"{"id":1}"#).unwrap();
        let mut valid_result = std::ptr::null_mut();
        let mut valid_error = std::ptr::null();
        let status = unsafe {
            kglite_session_execute_mut(
                session,
                valid_query.as_ptr(),
                valid_params.as_ptr(),
                &mut valid_result,
                &mut valid_error,
            )
        };
        assert_eq!(status, KgliteStatusCode::Ok);
        assert!(!valid_result.is_null());
        assert!(valid_error.is_null());
        unsafe { crate::kglite_cypher_result_free(valid_result) };
        assert_eq!(
            count(session, "MATCH (n:Accepted) RETURN count(n) AS c"),
            1,
            "representable query parameters still reach the write path"
        );

        unsafe { kglite_session_free(session) };
    }

    #[test]
    fn query_params_preserve_a_serde_private_number_marker_object() {
        let session = new_test_session();
        let query = CString::new("RETURN $value AS value").unwrap();
        let params = CString::new(r#"{"value":{"$serde_json::private::Number":"123"}}"#).unwrap();
        let mut result = std::ptr::null_mut();
        let mut error = std::ptr::null();
        let status = unsafe {
            kglite_session_execute_read(
                session,
                query.as_ptr(),
                params.as_ptr(),
                &mut result,
                &mut error,
            )
        };
        assert_eq!(status, KgliteStatusCode::Ok);
        assert!(error.is_null());
        let rows = unsafe { crate::kglite_cypher_result_rows_json(result) };
        let text = unsafe { CStr::from_ptr(rows) }.to_str().unwrap();
        assert!(text.contains(r#"$serde_json::private::Number"#), "{text}");
        unsafe {
            crate::kglite_free_string(rows);
            crate::kglite_cypher_result_free(result);
            kglite_session_free(session);
        }
    }

    #[test]
    fn strict_parameter_error_escapes_nul_in_object_key_for_c_message() {
        let session = new_test_session();
        let query = CString::new("RETURN $value").unwrap();
        let params = CString::new(r#"{"a\u0000b":1267650600228229401496703205376}"#).unwrap();
        let mut result = std::ptr::dangling_mut();
        let mut error = std::ptr::null();
        let status = unsafe {
            kglite_session_execute_read(
                session,
                query.as_ptr(),
                params.as_ptr(),
                &mut result,
                &mut error,
            )
        };
        assert_eq!(status, KgliteStatusCode::InvalidArgument);
        assert!(result.is_null());
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(message.contains(r#"$["a\u0000b"]"#), "{message}");
        unsafe { crate::kglite_free_string(error) };
        unsafe { kglite_session_free(session) };
    }
}
