//! Typed error taxonomy for KGLite.
//!
//! Phase A.2 of bolt_implementation.md — replaces the prior
//! "everything is a `String` then wrapped as `PyValueError` /
//! `PyRuntimeError`" pattern with a structured [`KgError`] enum + a
//! [`KgErrorCode`] classification. Existing per-module error types
//! (`SchemaError`, `ValidationError`, `ExprError`) are preserved and
//! bridged in via `From` impls — no taxonomy duplication.
//!
//! ## Why
//!
//! - Python consumers can `except kglite.CypherSyntaxError:` instead of
//!   grep'ing message strings.
//! - Cypher parser line/col survives the boundary instead of being
//!   embedded in the formatted string.
//! - Phase C.6 (Bolt FAILURE-code mapping) needs typed codes; landing
//!   them now means the whole engine surface is uniformly classified.
//! - MCP server error responses gain structured codes; agents can
//!   react programmatically.
//!
//! ## Hierarchy
//!
//! Every kglite-raised exception is a subclass of `kglite.KgError`.
//! The Python class chain is defined in [`crate::error_py`] via PyO3's
//! `create_exception!` macro. The Rust [`KgError`] enum + the
//! `From<KgError> for PyErr` impl at the boundary pick the most
//! specific subclass for each variant.
//!
//! Cypher: `CypherSyntaxError`, `CypherTimeoutError`,
//! `CypherExecutionError`, `CypherTypeMismatchError` — all subclass
//! `CypherError` which subclasses `KgError`.
//!
//! Schema/Validation: `SchemaError`, `ValidationError` — subclass
//! `KgError` directly.
//!
//! Resource access: `NodeNotFoundError`, `ConnectionNotFoundError`,
//! `PropertyNotFoundError` — subclass `KgError`.
//!
//! File/IO: `FileError`, `FileFormatError` — subclass `KgError`.
//!
//! Argument validation: `ArgumentError`, `MissingArgumentError` —
//! subclass `KgError`.
//!
//! Internal: `InternalError` — subclass `KgError`. Reserved for
//! invariants that should never trip (e.g. node-binding lookup that
//! was guaranteed by upstream pattern match).

use std::fmt;
use std::path::PathBuf;

use crate::graph::blueprint::expr::ExprError;
use crate::graph::languages::cypher::planner::schema_check::SchemaError;
use crate::graph::schema::ValidationError;

// ─── KgErrorCode ─────────────────────────────────────────────────────────────

/// Canonical classification of every error KGLite raises.
///
/// Maps 1-to-1 with the [`KgError`] enum variants but is `Copy + Eq +
/// Hash` so it can be used in match dispatch tables (e.g. Phase C.6's
/// Bolt FAILURE-code lookup).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KgErrorCode {
    // Cypher pipeline
    CypherSyntax,
    CypherTimeout,
    CypherExecution,
    CypherTypeMismatch,

    // Schema / validation
    Schema,
    Validation,
    Expr,

    // Resource / access
    NodeNotFound,
    ConnectionNotFound,
    PropertyNotFound,

    // File / I/O
    FileNotFound,
    FileFormat,
    FileIo,

    // Argument validation
    InvalidArgument,
    MissingArgument,

    // Internal / should-never-happen
    Internal,
}

impl KgErrorCode {
    /// Stable string representation. Used by the MCP server and Bolt
    /// server for typed wire reporting. PascalCase variant name.
    pub fn as_str(&self) -> &'static str {
        match self {
            KgErrorCode::CypherSyntax => "CypherSyntax",
            KgErrorCode::CypherTimeout => "CypherTimeout",
            KgErrorCode::CypherExecution => "CypherExecution",
            KgErrorCode::CypherTypeMismatch => "CypherTypeMismatch",
            KgErrorCode::Schema => "Schema",
            KgErrorCode::Validation => "Validation",
            KgErrorCode::Expr => "Expr",
            KgErrorCode::NodeNotFound => "NodeNotFound",
            KgErrorCode::ConnectionNotFound => "ConnectionNotFound",
            KgErrorCode::PropertyNotFound => "PropertyNotFound",
            KgErrorCode::FileNotFound => "FileNotFound",
            KgErrorCode::FileFormat => "FileFormat",
            KgErrorCode::FileIo => "FileIo",
            KgErrorCode::InvalidArgument => "InvalidArgument",
            KgErrorCode::MissingArgument => "MissingArgument",
            KgErrorCode::Internal => "Internal",
        }
    }

    /// Canonical Neo4j Bolt status code for this error code, of the
    /// shape `Neo.{Class}.{Category}.{Title}`. The Bolt protocol
    /// wraps these in a `FAILURE` response and drivers route by the
    /// class prefix (`ClientError` vs `DatabaseError` vs
    /// `TransientError`).
    ///
    /// Lifted from `kglite-bolt-server` so any future Neo4j-wire-
    /// compatible binding (Go driver, Java driver alternative, custom
    /// proxy) gets the canonical mapping without re-deriving the
    /// table. Bindings still own the wrapping in their own error
    /// type — only the code string is shared.
    pub fn neo4j_status_code(&self) -> &'static str {
        match self {
            KgErrorCode::CypherSyntax => "Neo.ClientError.Statement.SyntaxError",
            KgErrorCode::CypherTimeout => "Neo.ClientError.Transaction.TransactionTimedOut",
            KgErrorCode::CypherTypeMismatch => "Neo.ClientError.Statement.TypeError",
            KgErrorCode::CypherExecution => "Neo.DatabaseError.Statement.ExecutionFailed",
            KgErrorCode::Schema => "Neo.ClientError.Schema.ConstraintValidationFailed",
            KgErrorCode::Validation | KgErrorCode::Expr => {
                "Neo.ClientError.Statement.ArgumentError"
            }
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
}

impl fmt::Display for KgErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── KgError ─────────────────────────────────────────────────────────────────

/// The canonical error type for KGLite. Every fallible operation
/// reachable from the public API returns `Result<T, KgError>`
/// (directly or via `?` from a `From`-convertible source type).
///
/// At the PyO3 boundary, a `From<KgError> for PyErr` impl (in
/// [`crate::error_py`]) picks the most specific Python exception
/// subclass based on the variant.
#[derive(Debug)]
pub enum KgError {
    // ── Cypher pipeline ──────────────────────────────────────────────
    /// Cypher syntax error from the tokenizer or parser. Carries the
    /// line and column (1-indexed) where parsing failed. Both are
    /// `Option` because some parser-internal errors aren't pinned to
    /// a specific position (e.g. "expected end of input").
    CypherSyntax {
        message: String,
        line: Option<usize>,
        col: Option<usize>,
    },

    /// Cypher query exceeded its `timeout_ms` budget. Both elapsed and
    /// limit reported so the agent can decide whether to retry with a
    /// longer budget or rewrite the query.
    CypherTimeout { elapsed_ms: u64, limit_ms: u64 },

    /// Cypher executor failure (mutation conflict, predicate panic,
    /// missing aggregate context, etc.). Optional position points at
    /// the AST node when known.
    CypherExecution {
        message: String,
        position: Option<(usize, usize)>,
    },

    /// Type mismatch in Cypher evaluation (e.g. arithmetic on a String,
    /// IN over a non-list). Distinct from `CypherExecution` so consumers
    /// can react with a type-coercion retry vs a bail.
    CypherTypeMismatch {
        expected: String,
        found: String,
        context: String,
    },

    // ── Schema / validation ──────────────────────────────────────────
    /// Schema check failure (unknown property, type mismatch at
    /// pattern literal). Bridged from
    /// [`SchemaError`](crate::graph::languages::cypher::planner::schema_check::SchemaError)
    /// via `From`.
    Schema {
        kind: SchemaErrorKindRepr,
        message: String,
    },

    /// Structural validation failure (missing required field, wrong
    /// connection endpoint, etc.). Wraps the existing 6-variant
    /// [`ValidationError`] enum verbatim.
    Validation(ValidationError),

    /// Blueprint expression evaluation failure. Wraps the existing
    /// 7-variant [`ExprError`] enum verbatim.
    Expr(ExprError),

    // ── Resource / access ────────────────────────────────────────────
    /// A node identified by `(node_type, id)` doesn't exist in the
    /// graph. Used by mutation and traversal paths that expect a node.
    NodeNotFound { node_type: String, id: String },

    /// A connection type isn't declared in the schema.
    ConnectionNotFound { connection_type: String },

    /// A property is missing from a node or relationship.
    PropertyNotFound { node_type: String, property: String },

    // ── File / I/O ───────────────────────────────────────────────────
    /// A file the user named doesn't exist on disk.
    FileNotFound(PathBuf),

    /// A file exists but its contents are malformed (bad `.kgl` header,
    /// truncated blueprint JSON, etc.). The v3→v4 hard-break message
    /// from Phase A.1 / C5 surfaces here too.
    FileFormat { path: PathBuf, message: String },

    /// Generic I/O failure (permission denied, mid-read EOF, mmap
    /// failure). Carries the original [`std::io::Error`] for
    /// downstream inspection.
    FileIo(std::io::Error),

    // ── Argument validation ──────────────────────────────────────────
    /// A user-supplied argument violated a precondition with full
    /// structured context — argument name, what was expected, what
    /// was found. Used when the call site can naturally populate all
    /// three; agents can react programmatically on the structured fields.
    InvalidArgument {
        argument: String,
        expected: String,
        found: String,
    },

    /// A user-supplied argument violated a precondition; free-form
    /// message form for sites where the existing message is already
    /// good and forcing a structured shape would lose information.
    /// Maps to the same `kglite.ArgumentError` Python class as
    /// `InvalidArgument`.
    Argument(String),

    /// A required argument wasn't passed.
    MissingArgument(String),

    // ── Internal ─────────────────────────────────────────────────────
    /// An invariant was violated. Reserved for "should never happen"
    /// — e.g. a node-binding lookup whose existence was guaranteed by
    /// an upstream pattern match. Used when replacing `unwrap()`s in
    /// Phase A.4 (folded into A.2 / C4): if the unwrap would have
    /// panicked, this returns the typed error instead. The `location`
    /// is a `'static str` pointing at the source site (e.g.
    /// `"match_clause.rs::evaluate_pattern node_var lookup"`).
    Internal {
        message: String,
        location: &'static str,
    },
}

/// Wire-stable repr of [`SchemaErrorKind`](crate::graph::languages::cypher::planner::schema_check::SchemaErrorKind).
///
/// We don't re-export `SchemaErrorKind` because that crate path is
/// nested deep in the cypher tree; the repr lives next to `KgError`
/// for ergonomic match-on-variant in downstream callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaErrorKindRepr {
    UnknownProperty,
}

impl From<crate::graph::languages::cypher::planner::schema_check::SchemaErrorKind>
    for SchemaErrorKindRepr
{
    fn from(
        value: crate::graph::languages::cypher::planner::schema_check::SchemaErrorKind,
    ) -> Self {
        use crate::graph::languages::cypher::planner::schema_check::SchemaErrorKind;
        match value {
            SchemaErrorKind::UnknownProperty => SchemaErrorKindRepr::UnknownProperty,
        }
    }
}

// ─── Accessors ───────────────────────────────────────────────────────────────

impl KgError {
    /// Canonical [`KgErrorCode`] for this error. Drives the
    /// `From<KgError> for PyErr` boundary mapping and the future Bolt
    /// `Neo.ClientError.*` lookup.
    pub fn code(&self) -> KgErrorCode {
        match self {
            KgError::CypherSyntax { .. } => KgErrorCode::CypherSyntax,
            KgError::CypherTimeout { .. } => KgErrorCode::CypherTimeout,
            KgError::CypherExecution { .. } => KgErrorCode::CypherExecution,
            KgError::CypherTypeMismatch { .. } => KgErrorCode::CypherTypeMismatch,
            KgError::Schema { .. } => KgErrorCode::Schema,
            KgError::Validation(_) => KgErrorCode::Validation,
            KgError::Expr(_) => KgErrorCode::Expr,
            KgError::NodeNotFound { .. } => KgErrorCode::NodeNotFound,
            KgError::ConnectionNotFound { .. } => KgErrorCode::ConnectionNotFound,
            KgError::PropertyNotFound { .. } => KgErrorCode::PropertyNotFound,
            KgError::FileNotFound(_) => KgErrorCode::FileNotFound,
            KgError::FileFormat { .. } => KgErrorCode::FileFormat,
            KgError::FileIo(_) => KgErrorCode::FileIo,
            KgError::InvalidArgument { .. } | KgError::Argument(_) => KgErrorCode::InvalidArgument,
            KgError::MissingArgument(_) => KgErrorCode::MissingArgument,
            KgError::Internal { .. } => KgErrorCode::Internal,
        }
    }

    /// Source position (1-indexed line and column) when the error has
    /// one. Currently set by `CypherSyntax` (always when the tokenizer/
    /// parser knows it) and optionally by `CypherExecution`. Returns
    /// `None` for everything else.
    pub fn position(&self) -> Option<(usize, usize)> {
        match self {
            KgError::CypherSyntax {
                line: Some(l),
                col: Some(c),
                ..
            } => Some((*l, *c)),
            KgError::CypherExecution {
                position: Some(p), ..
            } => Some(*p),
            _ => None,
        }
    }
}

// ─── Display + std::error::Error ─────────────────────────────────────────────

impl fmt::Display for KgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KgError::CypherSyntax { message, line, col } => match (line, col) {
                (Some(l), Some(c)) => write!(
                    f,
                    "Cypher syntax error at line {}, col {}: {}",
                    l, c, message
                ),
                _ => write!(f, "Cypher syntax error: {}", message),
            },
            KgError::CypherTimeout {
                elapsed_ms,
                limit_ms,
            } => write!(
                f,
                "Cypher query exceeded timeout: elapsed {}ms, limit {}ms",
                elapsed_ms, limit_ms
            ),
            KgError::CypherExecution { message, position } => match position {
                Some((l, c)) => write!(
                    f,
                    "Cypher execution error at line {}, col {}: {}",
                    l, c, message
                ),
                None => write!(f, "Cypher execution error: {}", message),
            },
            KgError::CypherTypeMismatch {
                expected,
                found,
                context,
            } => write!(
                f,
                "Cypher type mismatch in {}: expected {}, found {}",
                context, expected, found
            ),
            KgError::Schema { message, .. } => write!(f, "Schema error: {}", message),
            KgError::Validation(v) => write!(f, "Validation error: {}", v),
            KgError::Expr(e) => write!(f, "Expression error: {}", e),
            KgError::NodeNotFound { node_type, id } => {
                write!(f, "Node not found: {} with id {:?}", node_type, id)
            }
            KgError::ConnectionNotFound { connection_type } => {
                write!(f, "Connection type not found: {}", connection_type)
            }
            KgError::PropertyNotFound {
                node_type,
                property,
            } => write!(f, "Property '{}' not found on {}", property, node_type),
            KgError::FileNotFound(path) => write!(f, "File not found: {}", path.display()),
            KgError::FileFormat { path, message } => {
                write!(f, "File format error ({}): {}", path.display(), message)
            }
            KgError::FileIo(e) => write!(f, "File I/O error: {}", e),
            KgError::InvalidArgument {
                argument,
                expected,
                found,
            } => write!(
                f,
                "Invalid argument '{}': expected {}, found {}",
                argument, expected, found
            ),
            KgError::Argument(message) => write!(f, "Invalid argument: {}", message),
            KgError::MissingArgument(name) => write!(f, "Missing required argument: {}", name),
            KgError::Internal { message, location } => {
                write!(f, "Internal error at {}: {}", location, message)
            }
        }
    }
}

impl std::error::Error for KgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KgError::FileIo(e) => Some(e),
            _ => None,
        }
    }
}

// ─── From impls ──────────────────────────────────────────────────────────────

impl From<SchemaError> for KgError {
    fn from(e: SchemaError) -> Self {
        KgError::Schema {
            kind: e.kind.into(),
            message: e.message,
        }
    }
}

impl From<ValidationError> for KgError {
    fn from(e: ValidationError) -> Self {
        KgError::Validation(e)
    }
}

impl From<ExprError> for KgError {
    fn from(e: ExprError) -> Self {
        KgError::Expr(e)
    }
}

impl From<std::io::Error> for KgError {
    fn from(e: std::io::Error) -> Self {
        match e.kind() {
            std::io::ErrorKind::NotFound => {
                // The path isn't recoverable from io::Error, but the
                // caller usually has it; the From impl preserves the
                // I/O error for downstream inspection via FileIo.
                KgError::FileIo(e)
            }
            _ => KgError::FileIo(e),
        }
    }
}

/// `Result` alias for KGLite operations. Use throughout the crate.
///
/// `#[allow(dead_code)]` until the C2+ commits start using it (the
/// foundation lands first; consumers migrate after).
#[allow(dead_code)]
pub type KgResult<T> = std::result::Result<T, KgError>;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_round_trip() {
        let e = KgError::CypherSyntax {
            message: "expected RETURN".to_string(),
            line: Some(3),
            col: Some(12),
        };
        assert_eq!(e.code(), KgErrorCode::CypherSyntax);
        assert_eq!(e.position(), Some((3, 12)));
    }

    #[test]
    fn display_includes_position() {
        let e = KgError::CypherSyntax {
            message: "expected RETURN".to_string(),
            line: Some(3),
            col: Some(12),
        };
        let s = format!("{}", e);
        assert!(s.contains("line 3"));
        assert!(s.contains("col 12"));
        assert!(s.contains("expected RETURN"));
    }

    #[test]
    fn display_without_position() {
        let e = KgError::CypherExecution {
            message: "div by zero".to_string(),
            position: None,
        };
        let s = format!("{}", e);
        assert!(s.contains("div by zero"));
        assert!(!s.contains("line"));
    }

    #[test]
    fn kgerror_code_as_str_stable() {
        // Wire-stable codes — any change here is a Bolt protocol breaking change.
        assert_eq!(KgErrorCode::CypherSyntax.as_str(), "CypherSyntax");
        assert_eq!(KgErrorCode::NodeNotFound.as_str(), "NodeNotFound");
        assert_eq!(KgErrorCode::FileFormat.as_str(), "FileFormat");
    }

    #[test]
    fn from_schema_error_preserves_kind_and_message() {
        use crate::graph::languages::cypher::planner::schema_check::SchemaErrorKind;
        let se = SchemaError {
            kind: SchemaErrorKind::UnknownProperty,
            message: "no such property 'foo'".to_string(),
        };
        let kg: KgError = se.into();
        assert_eq!(kg.code(), KgErrorCode::Schema);
        match kg {
            KgError::Schema { kind, message } => {
                assert_eq!(kind, SchemaErrorKindRepr::UnknownProperty);
                assert_eq!(message, "no such property 'foo'");
            }
            _ => panic!("expected Schema variant"),
        }
    }

    #[test]
    fn from_io_error_classifies_as_file_io() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let kg: KgError = io.into();
        assert_eq!(kg.code(), KgErrorCode::FileIo);
    }
}
