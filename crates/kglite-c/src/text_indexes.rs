//! Lexical (BM25) text-index construction through the C ABI.
//!
//! One `#[no_mangle] extern "C"` symbol on the session, wrapping
//! [`kglite::api::text_indexes::build_text_index`] so every non-Rust binding
//! can build the same index the Python wrapper does. There is no query symbol
//! and there will not be one: ranking is a Cypher function, so a consumer that
//! can call `kglite_session_query` can already search.
//!
//! **Receiver.** The build validates before it installs anything and writes a
//! single map entry, so it runs under
//! [`Session::write`](kglite::api::session::Session::write) — the same session
//! lock `execute_mut` takes — rather than paying `transact()`'s deep copy of
//! the whole graph for a call that has written nothing when it fails. Same
//! reasoning as the embedding ingest next door.

use crate::ffi::required_str;
use crate::session::{KgliteSession, SessionState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::text_indexes::build_text_index;
use std::ffi::c_char;

/// Build a BM25 lexical index over `node_type`'s `property`, for keyword
/// ranking in Cypher.
///
/// Wraps [`kglite::api::text_indexes::build_text_index`]. Opt-in and explicit.
/// The index does not follow later writes eagerly: it records them and folds
/// them in at query time while the outstanding delta stays under the default
/// auto-refresh limit, and once past it a rebuild — this call again — is the
/// route back. `SHOW INDEXES` reports the delta. Deleting a node prunes its
/// document immediately (a freed node slot is reused, and an orphaned document
/// would be inherited by its next owner); `vacuum` renumbers every node and
/// therefore drops text indexes wholesale. An empty string indexes as an empty
/// document; a property that is absent or holds a non-string is skipped and
/// counted in the report. The property is read through the same alias
/// resolution a `MATCH` filter uses.
///
/// The auto-refresh limit is not a parameter here: this signature is published
/// and additive-only within the ABI major, so the C route takes the default and
/// a future symbol carries the override if one is ever asked for.
///
/// On success `out_report_json` is an owned JSON object
/// `{"indexed": N, "skipped": S, "terms": T}`; free it with
/// [`kglite_free_string`](crate::kglite_free_string).
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `session`, `node_type`, `property`, or
///   `out_report_json` is null.
/// - `KGLITE_ERR_INVALID_UTF8` — a string argument is not valid UTF-8.
/// - `KGLITE_ERR_INVALID_ARGUMENT` — the node type is unknown, the graph is
///   disk-backed (the index is heap-resident, so disk mode refuses), or the
///   type has nodes and none of them carries a string for `property`; the
///   message explains which.
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new). `node_type` and
/// `property` must be null-terminated UTF-8 strings. `out_report_json` must be
/// a valid writable slot; `out_error_msg` null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_build_text_index(
    session: *mut KgliteSession,
    node_type: *const c_char,
    property: *const c_char,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_report_json, std::ptr::null()),
        || {
            if session.is_null()
                || node_type.is_null()
                || property.is_null()
                || out_report_json.is_null()
            {
                return KgliteStatusCode::NullPointer;
            }
            let node_type = match unsafe { required_str(node_type) } {
                Ok(s) => s,
                Err(rc) => return rc,
            };
            let property = match unsafe { required_str(property) } {
                Ok(s) => s,
                Err(rc) => return rc,
            };

            let session_state = unsafe { SessionState::from_handle(session) };
            let mut working = session_state.inner.write();
            match build_text_index(&mut working, node_type, property, None) {
                Ok(report) => {
                    let json = serde_json::json!({
                        "indexed": report.indexed,
                        "skipped": report.skipped,
                        "terms": report.terms,
                    })
                    .to_string();
                    unsafe {
                        *out_report_json = alloc_c_string(&json);
                    }
                    KgliteStatusCode::Ok
                }
                Err(msg) => {
                    unsafe {
                        *out_report_json = std::ptr::null();
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&msg);
                        }
                    }
                    KgliteStatusCode::InvalidArgument
                }
            }
        },
    )
}
