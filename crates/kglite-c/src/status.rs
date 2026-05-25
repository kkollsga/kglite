//! Status-code surface: `KgliteStatusCode` enum + 1:1 mapping to
//! `kglite::api::KgErrorCode` + the three canonical accessors
//! (`name`, `neo4j_status`, `http_status`).
//!
//! The mapping is fixed in declaration order and the discriminants
//! are stable for the lifetime of the ABI major version. Adding a
//! new `KgErrorCode` variant in core appends a new status code
//! here; removing one would require an ABI-major bump.

use crate::strings::alloc_c_string;
use kglite::api::KgErrorCode;
use std::ffi::c_char;

/// C-ABI-side error code. Variants 1-16 map 1:1 to
/// [`kglite::api::KgErrorCode`]; variants 100+ are C-ABI-specific
/// (invalid UTF-8 at the boundary, null pointer, OOM — conditions
/// that don't have a corresponding `KgErrorCode` because they
/// can't arise from inside the engine).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KgliteStatusCode {
    Ok = 0,

    // 1-16: mirror KgErrorCode, same order as declared there.
    CypherSyntax = 1,
    CypherTimeout = 2,
    CypherExecution = 3,
    CypherTypeMismatch = 4,
    Schema = 5,
    Validation = 6,
    Expr = 7,
    NodeNotFound = 8,
    ConnectionNotFound = 9,
    PropertyNotFound = 10,
    FileNotFound = 11,
    FileFormat = 12,
    FileIo = 13,
    InvalidArgument = 14,
    MissingArgument = 15,
    Internal = 16,

    // 100+: C-ABI-only errors.
    /// A string argument failed UTF-8 validation. The C-side
    /// caller passed a `*const c_char` whose bytes didn't decode
    /// as UTF-8 — typically a corrupted buffer or a non-UTF-8
    /// locale string. kglite is UTF-8 throughout.
    InvalidUtf8 = 100,
    /// A required pointer argument was null. The function
    /// can't proceed; check your call site.
    NullPointer = 101,
}

impl KgliteStatusCode {
    /// Map a core `KgErrorCode` variant to its C-ABI counterpart.
    /// Inline-callable from anywhere in the crate (used by every
    /// fallible wrapper that catches a `KgError` from the engine).
    pub(crate) fn from_kg_error_code(code: KgErrorCode) -> Self {
        match code {
            KgErrorCode::CypherSyntax => Self::CypherSyntax,
            KgErrorCode::CypherTimeout => Self::CypherTimeout,
            KgErrorCode::CypherExecution => Self::CypherExecution,
            KgErrorCode::CypherTypeMismatch => Self::CypherTypeMismatch,
            KgErrorCode::Schema => Self::Schema,
            KgErrorCode::Validation => Self::Validation,
            KgErrorCode::Expr => Self::Expr,
            KgErrorCode::NodeNotFound => Self::NodeNotFound,
            KgErrorCode::ConnectionNotFound => Self::ConnectionNotFound,
            KgErrorCode::PropertyNotFound => Self::PropertyNotFound,
            KgErrorCode::FileNotFound => Self::FileNotFound,
            KgErrorCode::FileFormat => Self::FileFormat,
            KgErrorCode::FileIo => Self::FileIo,
            KgErrorCode::InvalidArgument => Self::InvalidArgument,
            KgErrorCode::MissingArgument => Self::MissingArgument,
            KgErrorCode::Internal => Self::Internal,
        }
    }

    /// Reverse: C-ABI code → `KgErrorCode` so the helper accessors
    /// can delegate. Returns `None` for `Ok` and the C-ABI-only
    /// codes (`InvalidUtf8`, `NullPointer`) which have no
    /// `KgErrorCode` counterpart.
    pub(crate) fn to_kg_error_code(self) -> Option<KgErrorCode> {
        Some(match self {
            Self::Ok | Self::InvalidUtf8 | Self::NullPointer => return None,
            Self::CypherSyntax => KgErrorCode::CypherSyntax,
            Self::CypherTimeout => KgErrorCode::CypherTimeout,
            Self::CypherExecution => KgErrorCode::CypherExecution,
            Self::CypherTypeMismatch => KgErrorCode::CypherTypeMismatch,
            Self::Schema => KgErrorCode::Schema,
            Self::Validation => KgErrorCode::Validation,
            Self::Expr => KgErrorCode::Expr,
            Self::NodeNotFound => KgErrorCode::NodeNotFound,
            Self::ConnectionNotFound => KgErrorCode::ConnectionNotFound,
            Self::PropertyNotFound => KgErrorCode::PropertyNotFound,
            Self::FileNotFound => KgErrorCode::FileNotFound,
            Self::FileFormat => KgErrorCode::FileFormat,
            Self::FileIo => KgErrorCode::FileIo,
            Self::InvalidArgument => KgErrorCode::InvalidArgument,
            Self::MissingArgument => KgErrorCode::MissingArgument,
            Self::Internal => KgErrorCode::Internal,
        })
    }
}

/// Return the canonical human-readable name of a status code (e.g.
/// `"CypherSyntax"`, `"NodeNotFound"`, `"InvalidUtf8"`).
///
/// The returned string is OWNED by the caller and must be freed
/// via [`kglite_free_string`](crate::kglite_free_string). Returns
/// null on `Ok` (no error to name).
#[no_mangle]
pub extern "C" fn kglite_status_code_name(code: KgliteStatusCode) -> *const c_char {
    let s = match code {
        KgliteStatusCode::Ok => return std::ptr::null(),
        KgliteStatusCode::InvalidUtf8 => "InvalidUtf8",
        KgliteStatusCode::NullPointer => "NullPointer",
        other => match other.to_kg_error_code() {
            Some(kg) => kg.as_str(),
            None => return std::ptr::null(),
        },
    };
    alloc_c_string(s)
}

/// Return the Neo4j wire status code for a status code (e.g.
/// `"Neo.ClientError.Statement.SyntaxError"`). Useful for bindings
/// implementing the Neo4j Bolt wire protocol or compatible HTTP
/// APIs.
///
/// The returned string is OWNED by the caller and must be freed
/// via [`kglite_free_string`](crate::kglite_free_string). Returns
/// null on `Ok` or on C-ABI-only error codes that have no Neo4j
/// counterpart (`InvalidUtf8`, `NullPointer`).
#[no_mangle]
pub extern "C" fn kglite_status_code_neo4j_status(code: KgliteStatusCode) -> *const c_char {
    match code.to_kg_error_code() {
        Some(kg) => alloc_c_string(kg.neo4j_status_code()),
        None => std::ptr::null(),
    }
}

/// Return the HTTP status code mapping for a status code (e.g.
/// 400 for `CypherSyntax`, 404 for `NodeNotFound`, 500 for
/// `Internal`). Useful for REST/gRPC bindings.
///
/// Returns 0 for `Ok` and 500 for C-ABI-only codes (`InvalidUtf8`
/// = 400 / bad request from caller, `NullPointer` = 400).
#[no_mangle]
pub extern "C" fn kglite_status_code_http_status(code: KgliteStatusCode) -> u16 {
    match code {
        KgliteStatusCode::Ok => 0,
        KgliteStatusCode::InvalidUtf8 | KgliteStatusCode::NullPointer => 400,
        other => match other.to_kg_error_code() {
            Some(kg) => kg.http_status_code(),
            None => 500,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kg_error_code_round_trips() {
        // Exhaustive check — every KgErrorCode maps to a
        // KgliteStatusCode and back.
        for code in [
            KgErrorCode::CypherSyntax,
            KgErrorCode::CypherTimeout,
            KgErrorCode::CypherExecution,
            KgErrorCode::CypherTypeMismatch,
            KgErrorCode::Schema,
            KgErrorCode::Validation,
            KgErrorCode::Expr,
            KgErrorCode::NodeNotFound,
            KgErrorCode::ConnectionNotFound,
            KgErrorCode::PropertyNotFound,
            KgErrorCode::FileNotFound,
            KgErrorCode::FileFormat,
            KgErrorCode::FileIo,
            KgErrorCode::InvalidArgument,
            KgErrorCode::MissingArgument,
            KgErrorCode::Internal,
        ] {
            let c = KgliteStatusCode::from_kg_error_code(code);
            let back = c.to_kg_error_code();
            assert_eq!(back, Some(code), "round-trip failed for {code:?}");
        }
    }

    #[test]
    fn http_status_helpers_match_core() {
        // Sanity: an arbitrary code matches what core says.
        assert_eq!(
            kglite_status_code_http_status(KgliteStatusCode::CypherSyntax),
            400
        );
        assert_eq!(
            kglite_status_code_http_status(KgliteStatusCode::NodeNotFound),
            404
        );
        assert_eq!(
            kglite_status_code_http_status(KgliteStatusCode::Internal),
            500
        );
        assert_eq!(kglite_status_code_http_status(KgliteStatusCode::Ok), 0);
    }
}
