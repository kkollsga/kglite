//! Declarative schema installation through the C ABI.
//!
//! One `#[no_mangle] extern "C"` symbol, [`kglite_define_schema`], wrapping
//! [`kglite::api::schema_from_json`] + `DirGraph::set_schema` — the C-side
//! counterpart of the Python wheel's `define_schema`.
//!
//! **One dialect.** The schema document is parsed by the *same* core function
//! the Python wrapper calls; Python converts its dict to a `Value` and this
//! symbol converts JSON to the same `Value`, so both reach one grammar with one
//! set of messages. That matters more here than anywhere else in the crate: a
//! published C signature never changes within an ABI major, so a
//! C-ABI-specific schema shape would be a fork that could not be closed. The
//! shape is documented on [`kglite_define_schema`] and in
//! `kglite::api::schema_from_json`.
//!
//! **Receiver.** `set_schema` is all-or-nothing by construction — a
//! declaration that existing data already violates leaves neither the schema
//! nor the indexes changed — so it runs under
//! [`Session::write`](kglite::api::session::Session::write), the same session
//! lock `execute_mut` takes, exactly as the embedding ingest does and for the
//! same reason: `transact()`'s rollback would buy nothing and its deep copy
//! would cost the whole graph.

use crate::session::{KgliteSession, SessionState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::{schema_from_json, SchemaInstall};
use std::ffi::{c_char, CStr};

/// Resolve the `mode` argument. Null is [`SchemaInstall::Merge`] — the default
/// everywhere, and the mode whose mistake is a rejected write rather than an
/// admitted duplicate. An unrecognised spelling is rejected rather than
/// defaulted, so a typo cannot silently withdraw every constraint the call did
/// not name.
fn parse_mode(mode: *const c_char) -> Result<SchemaInstall, String> {
    if mode.is_null() {
        return Ok(SchemaInstall::Merge);
    }
    let s = unsafe { CStr::from_ptr(mode) }
        .to_str()
        .map_err(|_| "mode is not valid UTF-8".to_string())?;
    match s {
        "merge" => Ok(SchemaInstall::Merge),
        "replace" => Ok(SchemaInstall::Replace),
        other => Err(format!(
            "unknown schema install mode '{other}'; expected \"merge\" or \"replace\""
        )),
    }
}

/// Install a declarative schema on the session's graph.
///
/// `schema_json` is a JSON object in the one schema dialect kglite speaks —
/// the same one the Python wheel's `define_schema` takes, parsed by the same
/// core function:
///
/// ```json
/// {
///   "nodes": {
///     "Person": {
///       "required":       ["id", "name"],
///       "optional":       ["email"],
///       "types":          {"id": "integer", "name": "string"},
///       "primary_key":    "id",
///       "unique":         [["email"]],
///       "layer":          "managed",
///       "auto_timestamp": true
///     }
///   },
///   "connections": {
///     "KNOWS": {
///       "source":              "Person",
///       "target":              "Person",
///       "cardinality":         "many-to-many",
///       "required_properties": ["since"],
///       "property_types":      {"since": "integer"}
///     }
///   }
/// }
/// ```
///
/// `source` and `target` are the only mandatory keys anywhere; an absent
/// `nodes` or `connections` section is a no-op, so a document may declare
/// either half alone.
///
/// `mode` (may be null) is `"merge"` or `"replace"`; null means `"merge"`.
/// **Merge** scopes the declaration to the types the document names — every
/// other type keeps its constraints. **Replace** makes this document the whole
/// schema, so every type it does not name *stops being enforced*. The Python
/// wrapper emits a warning naming each constraint a replace withdraws; C has no
/// warning channel, so pass `"replace"` only when withdrawing is the intent.
///
/// Installing a schema installs the UNIQUE / NODE KEY constraints it declares,
/// so the call fails when data already in the graph violates one. In that case
/// **nothing changes** — neither the schema nor the indexes — and the caller
/// can fix the data and retry.
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `session` or `schema_json` is null.
/// - `KGLITE_ERR_INVALID_UTF8` — `schema_json` is not valid UTF-8.
/// - `KGLITE_ERR_INVALID_ARGUMENT` — the JSON did not parse, the document is
///   not in the dialect above, or `mode` is not `"merge"` / `"replace"`; the
///   message says which.
/// - A constraint status (`KGLITE_ERR_CONSTRAINT_VIOLATION` and friends) when
///   existing data violates a declared constraint; the message names it.
///
/// **The schema is not durable until saved.** Call
/// [`kglite_session_save`](crate::kglite_session_save) to persist it.
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new). `schema_json` must be a
/// null-terminated UTF-8 string; `mode` null or the same. `out_error_msg` must
/// be null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_define_schema(
    session: *mut KgliteSession,
    schema_json: *const c_char,
    mode: *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {},
        || {
            if session.is_null() || schema_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let json = match unsafe { CStr::from_ptr(schema_json) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let emit = |message: String, code: KgliteStatusCode| {
                if !out_error_msg.is_null() {
                    unsafe {
                        *out_error_msg = alloc_c_string(&message);
                    }
                }
                code
            };
            let mode = match parse_mode(mode) {
                Ok(m) => m,
                Err(message) => return emit(message, KgliteStatusCode::InvalidArgument),
            };
            // The shared grammar: a JSON document and a Python dict become the
            // identical `Value` before either is parsed, so the two bindings
            // cannot drift apart on what a schema means.
            let schema = match schema_from_json(json) {
                Ok(s) => s,
                Err(e) => return emit(e.message, KgliteStatusCode::InvalidArgument),
            };

            let session_state = unsafe { SessionState::from_handle(session) };
            let mut working = session_state.inner.write();
            match working.set_schema(schema, mode) {
                Ok(()) => KgliteStatusCode::Ok,
                Err(e) => {
                    let code = KgliteStatusCode::from_kg_error_code(e.code());
                    emit(e.to_string(), code)
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_arguments_are_rejected_before_anything_is_read() {
        let mut error: *const c_char = std::ptr::NonNull::<c_char>::dangling().as_ptr();
        let rc = unsafe {
            kglite_define_schema(
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                &mut error,
            )
        };
        assert_eq!(rc, KgliteStatusCode::NullPointer);
        assert!(
            error.is_null(),
            "the error slot must be reset before validation"
        );
    }

    #[test]
    fn mode_accepts_only_the_two_spellings() {
        assert_eq!(parse_mode(std::ptr::null()).unwrap(), SchemaInstall::Merge);
        let merge = std::ffi::CString::new("merge").unwrap();
        assert_eq!(parse_mode(merge.as_ptr()).unwrap(), SchemaInstall::Merge);
        let replace = std::ffi::CString::new("replace").unwrap();
        assert_eq!(
            parse_mode(replace.as_ptr()).unwrap(),
            SchemaInstall::Replace
        );
        // A typo is refused, not defaulted — defaulting "Replace" to "merge"
        // would silently keep constraints the caller meant to withdraw, and
        // defaulting "mrege" to replace would silently withdraw every one.
        let typo = std::ffi::CString::new("Replace").unwrap();
        let message = parse_mode(typo.as_ptr()).unwrap_err();
        assert!(message.contains("unknown schema install mode"), "{message}");
    }
}
