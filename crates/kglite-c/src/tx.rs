//! `KgliteTx`: an explicit transaction on a session. `kglite_session_begin`
//! takes the snapshot, `kglite_tx_execute` runs statements against the
//! transaction's own view (a statement sees the earlier statements' rows
//! before anything commits), and `kglite_tx_commit` publishes the lot
//! atomically under the same optimistic check, and write-ahead-log append on
//! a durable session, that Bolt's explicit COMMIT uses.

use crate::result::KgliteCypherResult;
use crate::session::{
    finish_query, parse_params_json, read_execute_options, reject_options,
    report_query_param_error, KgliteExecuteOptions, KgliteSession, SessionState,
};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::cypher::parse_with_mutation_check;
use kglite::api::session::{execute_mut, execute_read, CommitOutcome, Transaction};
use std::ffi::{c_char, CStr};
use std::sync::{Mutex, PoisonError};

/// Opaque handle for an explicit transaction. See
/// [`KgliteGraph`](crate::KgliteGraph) for the empty-`#[repr(C)]` facade.
#[repr(C)]
pub struct KgliteTx {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

struct TxState {
    /// The session this transaction was begun on. The handle contract (the
    /// transaction is freed or finished before its session is freed) is what
    /// keeps this pointer valid.
    session: *const SessionState,
    read_only: bool,
    /// `None` once committed or rolled back. The mutex serializes a misuse
    /// that shares one handle across threads; it is not a license to.
    inner: Mutex<Option<Transaction>>,
}

impl TxState {
    unsafe fn from_handle<'a>(handle: *const KgliteTx) -> &'a TxState {
        unsafe { &*handle.cast::<TxState>() }
    }
}

fn finished() -> KgliteStatusCode {
    KgliteStatusCode::InvalidArgument
}

fn report(out_error_msg: *mut *const c_char, message: &str) {
    if !out_error_msg.is_null() {
        unsafe { *out_error_msg = alloc_c_string(message) };
    }
}

/// Begin an explicit transaction on `session`.
///
/// A read-write transaction (`read_only` false) is refused with
/// `KGLITE_STATUS_CODE_READ_ONLY` on a session opened read-only, and with
/// `KGLITE_STATUS_CODE_INVALID_ARGUMENT` on a closed one. A read-only
/// transaction reads one fixed snapshot and refuses every write with
/// `KGLITE_STATUS_CODE_READ_ONLY`.
///
/// # Arguments
///
/// - `session` (in): a session handle that outlives the transaction.
/// - `read_only` (in): whether the transaction may write.
/// - `out_tx` (out, owned): the transaction handle; free it with
///   [`kglite_tx_free`]. Null on error.
/// - `out_error_msg` (out, owned, nullable): error message; free with
///   [`kglite_free_string`](crate::kglite_free_string).
///
/// # Thread safety
///
/// A transaction is single-threaded: do not call its functions concurrently.
/// The session stays usable from other threads, and other transactions
/// begun on it run independently.
///
/// # Safety
///
/// `session` a valid live session handle; `out_tx` a valid writable slot;
/// `out_error_msg` null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_begin(
    session: *mut KgliteSession,
    read_only: bool,
    out_tx: *mut *mut KgliteTx,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_tx, std::ptr::null_mut()),
        || {
            if session.is_null() || out_tx.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let state = unsafe { SessionState::from_handle(session) };
            if read_only {
                if state.life.is_closed() {
                    return crate::lifecycle::refuse(
                        out_error_msg,
                        (
                            KgliteStatusCode::InvalidArgument,
                            "this session is closed".to_string(),
                        ),
                    );
                }
            } else if let Err(refusal) = state.guard_write() {
                return crate::lifecycle::refuse(out_error_msg, refusal);
            }
            let tx = if read_only {
                state.inner.begin_read()
            } else {
                state.inner.begin()
            };
            let boxed = Box::new(TxState {
                session: std::ptr::from_ref(state),
                read_only,
                inner: Mutex::new(Some(tx)),
            });
            unsafe { *out_tx = Box::into_raw(boxed).cast::<KgliteTx>() };
            KgliteStatusCode::Ok
        },
    )
}

/// Run one statement inside `tx`.
///
/// Reads and writes both see this transaction's earlier writes and nothing
/// that other writers committed since [`kglite_session_begin`]. The writes
/// stay private until [`kglite_tx_commit`]. A failed statement is rolled
/// back on its own; the transaction stays open. `options` is as for
/// [`kglite_session_execute_read_ex`](crate::kglite_session_execute_read_ex);
/// null means no limits.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_READ_ONLY`: a write in a read-only transaction.
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT`: the transaction was already
///   committed or rolled back.
///
/// # Safety
///
/// `tx` a live handle from [`kglite_session_begin`], not used concurrently;
/// `query` a NUL-terminated UTF-8 string; `params_json` null or one;
/// `options` null or readable for `options->struct_size` bytes; the out
/// slots valid and writable (`out_error_msg` may be null).
#[no_mangle]
pub unsafe extern "C" fn kglite_tx_execute(
    tx: *mut KgliteTx,
    query: *const c_char,
    params_json: *const c_char,
    options: *const KgliteExecuteOptions,
    out_result: *mut *mut KgliteCypherResult,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    let limits = match read_execute_options(options) {
        Ok(limits) => limits,
        Err(message) => return unsafe { reject_options(out_result, out_error_msg, &message) },
    };
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_result, std::ptr::null_mut()),
        || {
            if tx.is_null() || query.is_null() || out_result.is_null() {
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
            let tx_state = unsafe { TxState::from_handle(tx) };
            let session = unsafe { &*tx_state.session };
            let mut guard = tx_state
                .inner
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let Some(inner) = guard.as_mut() else {
                report(
                    out_error_msg,
                    "the transaction is already committed or rolled back",
                );
                return finished();
            };
            let is_write = match parse_with_mutation_check(query_str) {
                Ok((parsed, is_mutation)) => is_mutation && !parsed.explain,
                Err(error) => {
                    let code = KgliteStatusCode::from_kg_error(&error);
                    report(out_error_msg, &error.to_string());
                    return code;
                }
            };
            let mut opts = session.make_opts(&params);
            limits.apply(&mut opts);
            let outcome = if is_write {
                match inner.working_mut() {
                    Ok(working) => execute_mut(working, query_str, &opts),
                    Err(error) => Err(error),
                }
            } else {
                match inner.current() {
                    Some(graph) => execute_read(graph, query_str, &opts),
                    None => {
                        report(
                            out_error_msg,
                            "the transaction is already committed or rolled back",
                        );
                        return finished();
                    }
                }
            };
            finish_query(session, outcome, out_result, out_error_msg)
        },
    )
}

/// Commit `tx`, publishing its writes atomically, then finish it.
///
/// If another writer committed since [`kglite_session_begin`], nothing is
/// applied and the status is `KGLITE_STATUS_CODE_TRANSACTION_CONFLICT`:
/// begin a new transaction and redo the work. On a durable session the commit
/// is logged before it is published; a log failure is
/// `KGLITE_STATUS_CODE_DURABILITY_FAILED` and nothing is applied. In every
/// case the transaction is finished afterwards; only
/// [`kglite_tx_free`] remains to be called. Committing a read-only or
/// write-free transaction succeeds and changes nothing.
///
/// # Safety
///
/// `tx` a live handle, not used concurrently; `out_error_msg` null or a valid
/// writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_tx_commit(
    tx: *mut KgliteTx,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {},
        || {
            if tx.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let tx_state = unsafe { TxState::from_handle(tx) };
            let session = unsafe { &*tx_state.session };
            let taken = tx_state
                .inner
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            let Some(inner) = taken else {
                report(
                    out_error_msg,
                    "the transaction is already committed or rolled back",
                );
                return finished();
            };
            if !tx_state.read_only && inner.has_writes() {
                if let Err(refusal) = session.guard_write() {
                    return crate::lifecycle::refuse(out_error_msg, refusal);
                }
            }
            let outcome = {
                let _gate = session.write_gate();
                session.inner.commit(inner, true)
            };
            match outcome {
                CommitOutcome::NoWritesNoOp | CommitOutcome::Committed { .. } => {
                    crate::lifecycle::auto_checkpoint(&session.inner);
                    KgliteStatusCode::Ok
                }
                CommitOutcome::ConflictDetected {
                    current_version,
                    base_version,
                } => {
                    report(
                        out_error_msg,
                        &format!(
                            "transaction conflict: the graph advanced from version \
                             {base_version} to {current_version} since begin; nothing was \
                             applied, retry the transaction"
                        ),
                    );
                    KgliteStatusCode::TransactionConflict
                }
                CommitOutcome::DurabilityFailed { error } => {
                    report(
                        out_error_msg,
                        &format!(
                            "the commit was not applied: the write-ahead log rejected it: {error}"
                        ),
                    );
                    KgliteStatusCode::DurabilityFailed
                }
                CommitOutcome::OntologyViolated { error } => {
                    let code = KgliteStatusCode::from_kg_error(&error);
                    report(out_error_msg, &error.to_string());
                    code
                }
                other => {
                    report(
                        out_error_msg,
                        &format!("the commit was not applied ({other:?})"),
                    );
                    KgliteStatusCode::Internal
                }
            }
        },
    )
}

/// Discard `tx` and its writes, then finish it. Rolling back a finished
/// transaction is a no-op that returns `KGLITE_STATUS_CODE_OK`.
///
/// # Safety
///
/// `tx` a live handle, not used concurrently.
#[no_mangle]
pub unsafe extern "C" fn kglite_tx_rollback(tx: *mut KgliteTx) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        std::ptr::null_mut(),
        || {},
        || {
            if tx.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let tx_state = unsafe { TxState::from_handle(tx) };
            drop(
                tx_state
                    .inner
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take(),
            );
            KgliteStatusCode::Ok
        },
    )
}

/// Free a transaction handle. A transaction still open is rolled back; free
/// never commits. Null is a no-op. Free every transaction before freeing or
/// closing its session.
///
/// # Safety
///
/// `tx` null or a handle from [`kglite_session_begin`] not yet freed, not
/// used concurrently.
#[no_mangle]
pub unsafe extern "C" fn kglite_tx_free(tx: *mut KgliteTx) {
    crate::ffi::void_boundary(|| {
        if !tx.is_null() {
            drop(unsafe { Box::from_raw(tx.cast::<TxState>()) });
        }
    });
}
