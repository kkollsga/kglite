//! Status-code surface: `KgliteStatusCode` enum + 1:1 mapping to
//! `kglite::api::KgErrorCode` + the canonical accessors (`name`,
//! `name_static`, `neo4j_status`, `http_status`).
//!
//! The mapping is fixed in declaration order and the discriminants
//! are stable for the lifetime of the ABI major version. Adding a
//! new `KgErrorCode` variant in core appends a new status code
//! here; removing one would require an ABI-major bump.

use crate::strings::alloc_c_string;
use kglite::api::KgErrorCode;
use std::ffi::{c_char, CStr};

/// C-ABI-side error code. Variants 1-17 map by meaning to
/// [`kglite::api::KgErrorCode`]; variants 100+ are C-ABI-specific
/// (invalid UTF-8 at the boundary, null pointer, OOM — conditions
/// that don't have a corresponding `KgErrorCode` because they
/// can't arise from inside the engine).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KgliteStatusCode {
    Ok = 0,

    // 1-16 preserve the original ABI order; Cancelled was appended as 17.
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
    /// The query was cooperatively cancelled (a binding flipped the
    /// cancel flag). Appended here (not renumbered into core's
    /// declaration position) to keep the existing discriminants stable
    /// across this ABI major version.
    Cancelled = 17,
    /// A write violated a declared integrity constraint (UNIQUE / NOT
    /// NULL / NODE KEY). The write was rejected before touching
    /// storage, so the graph is unchanged. Appended to keep the
    /// existing discriminants stable across this ABI major version.
    ConstraintViolation = 18,
    /// Declaring a constraint failed because the stored data already
    /// violates it. Deduplicate the node type, then re-declare.
    ConstraintCreationFailed = 19,
    /// An optimistic-concurrency commit lost its race: the graph advanced
    /// between `begin()` and `commit()`, so nothing was applied. Retry the
    /// whole transaction. Appended to keep the existing discriminants
    /// stable across this ABI major version.
    TransactionConflict = 20,

    // 100+: C-ABI-only errors.
    /// A string argument failed UTF-8 validation. The C-side
    /// caller passed a `*const c_char` whose bytes didn't decode
    /// as UTF-8 — typically a corrupted buffer or a non-UTF-8
    /// locale string. kglite is UTF-8 throughout.
    InvalidUtf8 = 100,
    /// A required pointer argument was null. The function
    /// can't proceed; check your call site.
    NullPointer = 101,
    /// Another process (or another un-freed handle in this one) holds the
    /// cross-process writer lease for that graph path, so
    /// [`kglite_writer_lease_acquire`](crate::kglite_writer_lease_acquire)
    /// gave up. The error message names the holder. Retriable as-is, which
    /// is why it is its own code rather than a `FileIo`: "wait and try
    /// again" and "the disk is broken" call for opposite reactions, and a
    /// binding cannot tell them apart by string-matching a message.
    ///
    /// Appended (not renumbered) to keep the existing discriminants stable
    /// across this ABI major version.
    WriterLeaseHeld = 102,
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
            KgErrorCode::Cancelled => Self::Cancelled,
            KgErrorCode::ConstraintViolation => Self::ConstraintViolation,
            KgErrorCode::ConstraintCreationFailed => Self::ConstraintCreationFailed,
            KgErrorCode::TransactionConflict => Self::TransactionConflict,
        }
    }

    /// Reverse: C-ABI code → `KgErrorCode` so the helper accessors
    /// can delegate. Returns `None` for `Ok` and the C-ABI-only
    /// codes (`InvalidUtf8`, `NullPointer`, `WriterLeaseHeld`) which
    /// have no `KgErrorCode` counterpart.
    pub(crate) fn to_kg_error_code(self) -> Option<KgErrorCode> {
        Some(match self {
            Self::Ok | Self::InvalidUtf8 | Self::NullPointer | Self::WriterLeaseHeld => {
                return None
            }
            Self::CypherSyntax => KgErrorCode::CypherSyntax,
            Self::CypherTimeout => KgErrorCode::CypherTimeout,
            Self::CypherExecution => KgErrorCode::CypherExecution,
            Self::CypherTypeMismatch => KgErrorCode::CypherTypeMismatch,
            Self::Schema => KgErrorCode::Schema,
            Self::ConstraintViolation => KgErrorCode::ConstraintViolation,
            Self::ConstraintCreationFailed => KgErrorCode::ConstraintCreationFailed,
            Self::TransactionConflict => KgErrorCode::TransactionConflict,
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
            Self::Cancelled => KgErrorCode::Cancelled,
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
    crate::ffi::value_boundary(std::ptr::null(), || {
        let s = match code {
            KgliteStatusCode::Ok => return std::ptr::null(),
            KgliteStatusCode::InvalidUtf8 => "InvalidUtf8",
            KgliteStatusCode::NullPointer => "NullPointer",
            KgliteStatusCode::WriterLeaseHeld => "WriterLeaseHeld",
            other => match other.to_kg_error_code() {
                Some(kg) => kg.as_str(),
                None => return std::ptr::null(),
            },
        };
        alloc_c_string(s)
    })
}

/// Return the canonical name of a status code as a **static** string — the
/// allocation-free companion to [`kglite_status_code_name`].
///
/// Same text (`"CypherSyntax"`, `"NodeNotFound"`, `"InvalidUtf8"`, …) and the
/// same null on `Ok`, but the pointer is a `'static` constant in the library's
/// own read-only data rather than a fresh heap copy.
///
/// **Do NOT free the returned pointer.** Handing it to
/// [`kglite_free_string`](crate::kglite_free_string) is undefined behaviour —
/// it is not allocator-owned memory. That is the whole difference between the
/// two functions, so pick one per call site and do not mix them:
/// [`kglite_status_code_name`] for a caller that would rather free one uniform
/// kind of string, this one for a binding that names a code on every error
/// (the common case — the name goes straight into the exception it raises),
/// which then skips the allocate/copy/free round trip entirely.
///
/// Added rather than changing [`kglite_status_code_name`] in place: that
/// function shipped with owned-string semantics and this ABI is additive-only
/// within a major version, so silently flipping who frees it would turn a
/// correct caller into a double-free.
///
/// The table is exhaustive over `KgliteStatusCode` by construction (no
/// wildcard arm), so a newly added status code is a compile error here rather
/// than a silently unnamed one.
#[no_mangle]
pub extern "C" fn kglite_status_code_name_static(code: KgliteStatusCode) -> *const c_char {
    crate::ffi::value_boundary(std::ptr::null(), || {
        static_name(code).map_or(std::ptr::null(), CStr::as_ptr)
    })
}

/// The `'static`, nul-terminated name table behind
/// [`kglite_status_code_name_static`]. Deliberately written out instead of
/// nul-terminating [`KgErrorCode::as_str`] at runtime — that would need an
/// allocation or a lazy leak, which is exactly what this exists to avoid. The
/// duplication is gated by `static_names_match_the_owned_ones`, which compares
/// every code's static name against the owned one core produces.
fn static_name(code: KgliteStatusCode) -> Option<&'static CStr> {
    Some(match code {
        KgliteStatusCode::Ok => return None,
        KgliteStatusCode::CypherSyntax => c"CypherSyntax",
        KgliteStatusCode::CypherTimeout => c"CypherTimeout",
        KgliteStatusCode::CypherExecution => c"CypherExecution",
        KgliteStatusCode::CypherTypeMismatch => c"CypherTypeMismatch",
        KgliteStatusCode::Schema => c"Schema",
        KgliteStatusCode::Validation => c"Validation",
        KgliteStatusCode::Expr => c"Expr",
        KgliteStatusCode::NodeNotFound => c"NodeNotFound",
        KgliteStatusCode::ConnectionNotFound => c"ConnectionNotFound",
        KgliteStatusCode::PropertyNotFound => c"PropertyNotFound",
        KgliteStatusCode::FileNotFound => c"FileNotFound",
        KgliteStatusCode::FileFormat => c"FileFormat",
        KgliteStatusCode::FileIo => c"FileIo",
        KgliteStatusCode::InvalidArgument => c"InvalidArgument",
        KgliteStatusCode::MissingArgument => c"MissingArgument",
        KgliteStatusCode::Internal => c"Internal",
        KgliteStatusCode::Cancelled => c"Cancelled",
        KgliteStatusCode::ConstraintViolation => c"ConstraintViolation",
        KgliteStatusCode::ConstraintCreationFailed => c"ConstraintCreationFailed",
        KgliteStatusCode::TransactionConflict => c"TransactionConflict",
        KgliteStatusCode::InvalidUtf8 => c"InvalidUtf8",
        KgliteStatusCode::NullPointer => c"NullPointer",
        KgliteStatusCode::WriterLeaseHeld => c"WriterLeaseHeld",
    })
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
    crate::ffi::value_boundary(std::ptr::null(), || match code.to_kg_error_code() {
        Some(kg) => alloc_c_string(kg.neo4j_status_code()),
        None => std::ptr::null(),
    })
}

/// Return the HTTP status code mapping for a status code (e.g.
/// 400 for `CypherSyntax`, 404 for `NodeNotFound`, 500 for
/// `Internal`). Useful for REST/gRPC bindings.
///
/// Returns 0 for `Ok` and 500 for C-ABI-only codes (`InvalidUtf8`
/// = 400 / bad request from caller, `NullPointer` = 400,
/// `WriterLeaseHeld` = 409 / conflict, retriable as-is — the same
/// mapping core gives `TransactionConflict`, the other lost-race code).
#[no_mangle]
pub extern "C" fn kglite_status_code_http_status(code: KgliteStatusCode) -> u16 {
    crate::ffi::value_boundary(500, || match code {
        KgliteStatusCode::Ok => 0,
        KgliteStatusCode::InvalidUtf8 | KgliteStatusCode::NullPointer => 400,
        KgliteStatusCode::WriterLeaseHeld => 409,
        other => match other.to_kg_error_code() {
            Some(kg) => kg.http_status_code(),
            None => 500,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `KgliteStatusCode`, for the accessor sweeps below. The compile
    /// gate on completeness is `static_name`'s wildcard-free match; this list
    /// only has to keep up for the *equality* check to stay total.
    const ALL_CODES: &[KgliteStatusCode] = &[
        KgliteStatusCode::Ok,
        KgliteStatusCode::CypherSyntax,
        KgliteStatusCode::CypherTimeout,
        KgliteStatusCode::CypherExecution,
        KgliteStatusCode::CypherTypeMismatch,
        KgliteStatusCode::Schema,
        KgliteStatusCode::Validation,
        KgliteStatusCode::Expr,
        KgliteStatusCode::NodeNotFound,
        KgliteStatusCode::ConnectionNotFound,
        KgliteStatusCode::PropertyNotFound,
        KgliteStatusCode::FileNotFound,
        KgliteStatusCode::FileFormat,
        KgliteStatusCode::FileIo,
        KgliteStatusCode::InvalidArgument,
        KgliteStatusCode::MissingArgument,
        KgliteStatusCode::Internal,
        KgliteStatusCode::Cancelled,
        KgliteStatusCode::ConstraintViolation,
        KgliteStatusCode::ConstraintCreationFailed,
        KgliteStatusCode::TransactionConflict,
        KgliteStatusCode::InvalidUtf8,
        KgliteStatusCode::NullPointer,
        KgliteStatusCode::WriterLeaseHeld,
    ];

    /// The static table is a hand-written duplicate of the names core hands
    /// out; this is the gate that keeps it from drifting. A binding that
    /// switched to the static form and got a stale name would report the
    /// wrong error kind with no other signal.
    #[test]
    fn static_names_match_the_owned_ones() {
        for &code in ALL_CODES {
            let owned = kglite_status_code_name(code);
            let stat = kglite_status_code_name_static(code);
            if owned.is_null() {
                assert!(stat.is_null(), "{code:?}: owned named nothing, static did");
                continue;
            }
            assert!(
                !stat.is_null(),
                "{code:?}: static must name what owned names"
            );
            let owned_text = unsafe { CStr::from_ptr(owned) }
                .to_str()
                .unwrap()
                .to_string();
            let static_text = unsafe { CStr::from_ptr(stat) }.to_str().unwrap();
            assert_eq!(static_text, owned_text, "{code:?}");
            unsafe { crate::kglite_free_string(owned) };
        }
        // Only `Ok` is nameless — otherwise the sweep above could pass vacuously
        // by having every code return null from both.
        assert!(kglite_status_code_name_static(KgliteStatusCode::Ok).is_null());
        assert_eq!(
            ALL_CODES
                .iter()
                .filter(|&&c| kglite_status_code_name_static(c).is_null())
                .count(),
            1
        );
    }

    /// The pointer is a constant, not a copy: two calls hand back the *same*
    /// address. A caller that freed it would be freeing library rodata, which
    /// is why the header says not to.
    #[test]
    fn static_names_are_the_same_pointer_every_call() {
        let first = kglite_status_code_name_static(KgliteStatusCode::NodeNotFound);
        let second = kglite_status_code_name_static(KgliteStatusCode::NodeNotFound);
        assert!(!first.is_null());
        assert_eq!(first, second);
        // The owned form, by contrast, must hand back a fresh allocation.
        let a = kglite_status_code_name(KgliteStatusCode::NodeNotFound);
        let b = kglite_status_code_name(KgliteStatusCode::NodeNotFound);
        assert_ne!(a, b);
        assert_ne!(a, first, "the owned copy must not alias the static table");
        unsafe { crate::kglite_free_string(a) };
        unsafe { crate::kglite_free_string(b) };
    }

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

    /// The boundary-only codes have no `KgErrorCode` to delegate to, so each
    /// accessor answers for them directly — and must keep answering, since a
    /// binding routes on exactly these three.
    #[test]
    fn boundary_only_codes_have_their_own_accessors() {
        for (code, name, http) in [
            (KgliteStatusCode::InvalidUtf8, "InvalidUtf8", 400),
            (KgliteStatusCode::NullPointer, "NullPointer", 400),
            // 409, not 500: a held lease is a retriable conflict, and a
            // binding that saw a 5xx here would surface an outage instead of
            // a wait-and-retry.
            (KgliteStatusCode::WriterLeaseHeld, "WriterLeaseHeld", 409),
        ] {
            let named = kglite_status_code_name(code);
            assert!(!named.is_null(), "{name} must name itself");
            let text = unsafe { std::ffi::CStr::from_ptr(named) }.to_str().unwrap();
            assert_eq!(text, name);
            unsafe { crate::kglite_free_string(named) };
            assert_eq!(kglite_status_code_http_status(code), http, "{name}");
            // No Neo4j wire code exists for a boundary-only failure.
            assert!(kglite_status_code_neo4j_status(code).is_null(), "{name}");
        }
    }
}
