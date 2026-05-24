//! Map kglite's typed [`KgError`] / [`KgErrorCode`] onto Bolt FAILURE
//! status codes (`Neo.{Class}.{Category}.{Title}` strings).
//!
//! Phase A.2 added the typed error hierarchy on the Python boundary
//! (`kglite.CypherSyntaxError`, `kglite.CypherTimeoutError`, etc.);
//! Phase C.6 (this module) wires the same hierarchy to the Bolt wire
//! so the neo4j Python driver raises the matching driver-side
//! exception class (`CypherSyntaxError` instead of generic
//! `ClientError`).
//!
//! ## Mapping table
//!
//! | `KgErrorCode`             | Neo4j status code                                  | Driver class       |
//! |---------------------------|----------------------------------------------------|--------------------|
//! | `CypherSyntax`            | `Neo.ClientError.Statement.SyntaxError`            | CypherSyntaxError  |
//! | `CypherTimeout`           | `Neo.ClientError.Transaction.TransactionTimedOut`  | ClientError        |
//! | `CypherTypeMismatch`      | `Neo.ClientError.Statement.TypeError`              | ClientError        |
//! | `CypherExecution`         | `Neo.DatabaseError.Statement.ExecutionFailed`      | DatabaseError      |
//! | `Schema`                  | `Neo.ClientError.Schema.ConstraintValidationFailed`| ClientError        |
//! | `Validation` / `Expr`     | `Neo.ClientError.Statement.ArgumentError`          | ClientError        |
//! | `NodeNotFound` / `ConnectionNotFound` / `PropertyNotFound` | `Neo.ClientError.Statement.EntityNotFound` | ClientError |
//! | `InvalidArgument`         | `Neo.ClientError.Statement.ArgumentError`          | ClientError        |
//! | `MissingArgument`         | `Neo.ClientError.Statement.ParameterMissing`       | ClientError        |
//! | `FileNotFound` / `FileFormat` / `FileIo` | `Neo.DatabaseError.General.UnknownError` (server-side I/O — surface as DB error) | DatabaseError |
//! | `Internal`                | `Neo.DatabaseError.General.UnknownError`           | DatabaseError      |
//!
//! Codes that don't have an exact Neo4j equivalent reuse the closest
//! ClientError-class fallback (matches what most Neo4j servers do for
//! their own unmapped extensions).

use boltr::error::BoltError;
use kglite::api::{KgError, KgErrorCode};

/// Map a [`KgError`] to a [`BoltError::Query`] with the right
/// `Neo.{Class}.{Category}.{Title}` code. boltr's
/// `BoltError::to_failure_metadata` passes the code+message through
/// to the wire FAILURE response, where the driver routes by class
/// prefix (ClientError vs DatabaseError vs TransientError).
pub fn kg_to_bolt(err: KgError) -> BoltError {
    let code = neo4j_status_code(err.code());
    BoltError::Query {
        code: code.into(),
        message: err.to_string(),
    }
}

/// Canonical Neo4j status code for a kglite error code.
fn neo4j_status_code(code: KgErrorCode) -> &'static str {
    match code {
        KgErrorCode::CypherSyntax => "Neo.ClientError.Statement.SyntaxError",
        KgErrorCode::CypherTimeout => "Neo.ClientError.Transaction.TransactionTimedOut",
        KgErrorCode::CypherTypeMismatch => "Neo.ClientError.Statement.TypeError",
        KgErrorCode::CypherExecution => "Neo.DatabaseError.Statement.ExecutionFailed",
        KgErrorCode::Schema => "Neo.ClientError.Schema.ConstraintValidationFailed",
        KgErrorCode::Validation | KgErrorCode::Expr => "Neo.ClientError.Statement.ArgumentError",
        KgErrorCode::NodeNotFound
        | KgErrorCode::ConnectionNotFound
        | KgErrorCode::PropertyNotFound => "Neo.ClientError.Statement.EntityNotFound",
        KgErrorCode::InvalidArgument => "Neo.ClientError.Statement.ArgumentError",
        KgErrorCode::MissingArgument => "Neo.ClientError.Statement.ParameterMissing",
        KgErrorCode::FileNotFound
        | KgErrorCode::FileFormat
        | KgErrorCode::FileIo
        | KgErrorCode::Internal => "Neo.DatabaseError.General.UnknownError",
    }
}

/// Substring-based heuristic that maps a `String` error from the
/// kglite Cypher executor (rewrite_text_score, CypherExecutor::execute,
/// execute_mutable) onto a typed `BoltError::Query` with the right
/// Neo4j status code.
///
/// Not every kglite error type travels as a typed `KgError` today —
/// the executor returns `String`. This helper bridges the gap until
/// kglite's executor is refactored to return `KgError` natively (a
/// separate plan). Each heuristic is conservative: lowercase
/// substring match against well-known phrases. On no match, falls
/// back to `BoltError::Backend` (Neo.DatabaseError.General.UnknownError)
/// which is the previous behavior — so this is a strict refinement.
pub fn string_to_bolt(error_msg: String) -> BoltError {
    let lower = error_msg.to_lowercase();
    let code = if lower.contains("timeout") || lower.contains("timed out") {
        Some("Neo.ClientError.Transaction.TransactionTimedOut")
    } else if lower.contains("type mismatch")
        || lower.contains("type error")
        || lower.contains("typeerror")
    {
        Some("Neo.ClientError.Statement.TypeError")
    } else if lower.contains("unknown parameter")
        || lower.contains("missing parameter")
        || lower.contains("parameter not found")
        || lower.contains("expected parameter")
    {
        Some("Neo.ClientError.Statement.ParameterMissing")
    } else if lower.contains("syntax error") {
        // Some executor errors echo "syntax error" even though they
        // come from non-parse phases. Map to SyntaxError anyway.
        Some("Neo.ClientError.Statement.SyntaxError")
    } else if lower.contains("constraint") {
        Some("Neo.ClientError.Schema.ConstraintValidationFailed")
    } else if lower.contains("unknown function") || lower.contains("undefined function") {
        Some("Neo.ClientError.Procedure.ProcedureNotFound")
    } else {
        None
    };
    match code {
        Some(c) => BoltError::Query {
            code: c.into(),
            message: error_msg,
        },
        None => BoltError::Backend(error_msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_error_maps_to_neo_clienterror_statement_syntaxerror() {
        let err = KgError::CypherSyntax {
            message: "unexpected token 'NOT'".into(),
            line: Some(1),
            col: Some(7),
        };
        let bolt = kg_to_bolt(err);
        match bolt {
            BoltError::Query { code, .. } => {
                assert_eq!(code, "Neo.ClientError.Statement.SyntaxError");
            }
            other => panic!("expected Query, got {other:?}"),
        }
    }

    #[test]
    fn string_to_bolt_matches_known_phrases() {
        let cases = &[
            (
                "Query timed out after 1000 ms",
                "Neo.ClientError.Transaction.TransactionTimedOut",
            ),
            (
                "type mismatch: expected Integer, got String",
                "Neo.ClientError.Statement.TypeError",
            ),
            ("TypeError on n.age", "Neo.ClientError.Statement.TypeError"),
            (
                "missing parameter $x",
                "Neo.ClientError.Statement.ParameterMissing",
            ),
            (
                "unknown parameter referenced: $y",
                "Neo.ClientError.Statement.ParameterMissing",
            ),
            (
                "syntax error at offset 42",
                "Neo.ClientError.Statement.SyntaxError",
            ),
            (
                "constraint violation: Person.id must be unique",
                "Neo.ClientError.Schema.ConstraintValidationFailed",
            ),
            (
                "Unknown function: foo()",
                "Neo.ClientError.Procedure.ProcedureNotFound",
            ),
        ];
        for (msg, expected_code) in cases {
            let bolt = string_to_bolt((*msg).to_string());
            match bolt {
                BoltError::Query { code, .. } => {
                    assert_eq!(code, *expected_code, "for msg {msg:?}")
                }
                other => panic!("expected Query for {msg:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn string_to_bolt_unknown_falls_back_to_backend() {
        let bolt = string_to_bolt("some unanticipated executor message".to_string());
        match bolt {
            BoltError::Backend(_) => {} // expected
            other => panic!("expected Backend fallback, got {other:?}"),
        }
    }

    #[test]
    fn every_code_has_a_neo4j_string_starting_with_neo_dot() {
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
            let s = neo4j_status_code(code);
            assert!(
                s.starts_with("Neo."),
                "code {:?} mapped to non-Neo.* string: {}",
                code,
                s
            );
            // All Neo4j codes have exactly 4 dotted segments.
            assert_eq!(
                s.split('.').count(),
                4,
                "code {:?} mapped to wrong-shaped string: {} (want 4 dotted segments)",
                code,
                s
            );
        }
    }
}
