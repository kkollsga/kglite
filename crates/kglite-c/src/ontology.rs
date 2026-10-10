//! Ontology declaration through the C ABI.
//!
//! [`kglite_session_define_ontology`] parses the one ontology dialect
//! (`kglite::api::ontology_from_json`, the grammar Python's
//! `define_ontology` also reaches) and installs it through
//! `DirGraph::define_ontology`, so the declare-over-data verification and the
//! operator lock are the engine's. [`kglite_session_clear_ontology`] removes
//! it. Both run in a session transaction. On a durable session
//! (`kglite_open_session`) the commit is logged; otherwise persistence is
//! `kglite_session_save`.

use crate::session::{KgliteSession, SessionState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::session::CommitOutcome;
use kglite::api::{ontology_from_json, DefineOntologyError, KgError};
use std::ffi::{c_char, CStr};

fn emit(
    out_error_msg: *mut *const c_char,
    message: &str,
    code: KgliteStatusCode,
) -> KgliteStatusCode {
    crate::ffi::init_out(out_error_msg, alloc_c_string(message));
    code
}

fn commit_status(outcome: CommitOutcome, out_error_msg: *mut *const c_char) -> KgliteStatusCode {
    match outcome {
        CommitOutcome::Committed { .. } | CommitOutcome::NoWritesNoOp => KgliteStatusCode::Ok,
        CommitOutcome::ConflictDetected { .. } => emit(
            out_error_msg,
            "the graph changed while the ontology was being declared; retry",
            KgliteStatusCode::from_kg_error_code(kglite::api::KgErrorCode::TransactionConflict),
        ),
        CommitOutcome::DurabilityFailed { error } => emit(
            out_error_msg,
            &error,
            KgliteStatusCode::from_kg_error_code(kglite::api::KgErrorCode::DurabilityFailed),
        ),
        CommitOutcome::OntologyViolated { error } => emit(
            out_error_msg,
            &error.to_string(),
            KgliteStatusCode::from_kg_error(&error),
        ),
        other => emit(
            out_error_msg,
            &format!("the ontology change was not committed ({other:?})"),
            KgliteStatusCode::Internal,
        ),
    }
}

/// Declare the session graph's ontology from a JSON document.
///
/// `ontology_json` uses the same dialect as the Python wheel's
/// `define_ontology` (`classes`, `relationships`, `closed_labels`,
/// `enforcement`, `version`), parsed by the same core function. Stored data is
/// checked against the declaration first. An `error`-level rule that stored
/// data already breaks refuses the declaration: nothing changes and the
/// previous ontology stays.
///
/// On success `out_warnings_json` is an owned JSON array of strings (the
/// `warn`-level findings; empty when there are none). On an
/// `KGLITE_STATUS_CODE_ONTOLOGY_VIOLATION` refusal it is an owned JSON array
/// of report objects `{rule, entity, entity_type, property, count}`,
/// `out_error_msg` carries the readable report, and
/// [`kglite_last_error_details_json`](crate::details::kglite_last_error_details_json)
/// returns the headline fields with the same report. On any other failure it is
/// null. `out_warnings_json` may be null when the caller wants neither. Free
/// both with [`kglite_free_string`](crate::kglite_free_string).
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `session` or `ontology_json` is null.
/// - `KGLITE_STATUS_CODE_INVALID_UTF8` — `ontology_json` is not valid UTF-8.
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — the JSON did not parse, is not in
///   the dialect, or the ontology is locked by the operator.
/// - `KGLITE_STATUS_CODE_ONTOLOGY_VIOLATION` — stored data breaks an
///   `error`-level rule of the declaration.
///
/// On a durable session from [`kglite_open_session`](crate::kglite_open_session)
/// the declaration is write-ahead logged. On a session without a log it is not
/// durable until [`kglite_session_save`](crate::kglite_session_save).
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new); `ontology_json` a
/// null-terminated UTF-8 string; `out_warnings_json` and `out_error_msg` null
/// or valid writable slots.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_define_ontology(
    session: *const KgliteSession,
    ontology_json: *const c_char,
    out_warnings_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_warnings_json, std::ptr::null()),
        || {
            if session.is_null() || ontology_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let Ok(json) = unsafe { CStr::from_ptr(ontology_json) }.to_str() else {
                return KgliteStatusCode::InvalidUtf8;
            };
            let store = match ontology_from_json(json) {
                Ok(store) => store,
                Err(message) => {
                    return emit(out_error_msg, &message, KgliteStatusCode::InvalidArgument)
                }
            };
            let state = unsafe { SessionState::from_handle(session) };
            if let Err(refusal) = state.guard_write() {
                return crate::lifecycle::refuse(out_error_msg, refusal);
            }
            let _gate = state.write_gate();
            let mut tx = state.inner.begin();
            let working = match tx.working_mut() {
                Ok(working) => working,
                Err(e) => {
                    return emit(
                        out_error_msg,
                        &e.to_string(),
                        KgliteStatusCode::from_kg_error(&e),
                    )
                }
            };
            let warnings = match working.define_ontology(store) {
                Ok(warnings) => warnings,
                Err(DefineOntologyError::Invalid(message)) => {
                    return emit(out_error_msg, &message, KgliteStatusCode::InvalidArgument)
                }
                Err(DefineOntologyError::Refused(refused)) => {
                    crate::ffi::init_out(
                        out_warnings_json,
                        alloc_c_string(&crate::details::report_json(&refused.entries).to_string()),
                    );
                    let error = KgError::from(refused);
                    return emit(
                        out_error_msg,
                        &error.to_string(),
                        KgliteStatusCode::from_kg_error(&error),
                    );
                }
            };
            let status = commit_status(state.inner.commit(tx, true), out_error_msg);
            if status == KgliteStatusCode::Ok {
                crate::ffi::init_out(
                    out_warnings_json,
                    alloc_c_string(&serde_json::json!(warnings).to_string()),
                );
            }
            status
        },
    )
}

/// Remove the session graph's declared ontology.
///
/// A no-op success when none is declared. Refused with
/// `KGLITE_STATUS_CODE_INVALID_ARGUMENT` when the operator locked the
/// ontology.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `session` is null.
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — the ontology is locked.
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new); `out_error_msg` null or
/// a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_clear_ontology(
    session: *const KgliteSession,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {},
        || {
            if session.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let state = unsafe { SessionState::from_handle(session) };
            if let Err(refusal) = state.guard_write() {
                return crate::lifecycle::refuse(out_error_msg, refusal);
            }
            let _gate = state.write_gate();
            let mut tx = state.inner.begin();
            let working = match tx.working_mut() {
                Ok(working) => working,
                Err(e) => {
                    return emit(
                        out_error_msg,
                        &e.to_string(),
                        KgliteStatusCode::from_kg_error(&e),
                    )
                }
            };
            if let Err(message) = working.clear_ontology() {
                return emit(out_error_msg, &message, KgliteStatusCode::InvalidArgument);
            }
            commit_status(state.inner.commit(tx, true), out_error_msg)
        },
    )
}
