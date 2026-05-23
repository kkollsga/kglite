//! Python-side machinery for the [`crate::error::KgError`] taxonomy.
//!
//! Phase A.2 of bolt_implementation.md — defines the typed Python
//! exception classes (`kglite.CypherSyntaxError`, `kglite.SchemaError`,
//! etc.) via PyO3's `create_exception!` macro, and provides the
//! [`From<KgError> for PyErr`] impl that picks the most specific
//! subclass for each variant at the PyO3 boundary.
//!
//! ## Hierarchy
//!
//! All kglite-raised exceptions descend from `kglite.KgError`, which
//! itself descends from `Exception`:
//!
//! ```text
//! Exception
//! └── kglite.KgError                          (base)
//!     ├── kglite.CypherError                   (Cypher pipeline base)
//!     │   ├── kglite.CypherSyntaxError
//!     │   ├── kglite.CypherTimeoutError
//!     │   ├── kglite.CypherExecutionError
//!     │   └── kglite.CypherTypeMismatchError
//!     ├── kglite.SchemaError
//!     ├── kglite.ValidationError
//!     ├── kglite.ExprError
//!     ├── kglite.NodeNotFoundError
//!     ├── kglite.ConnectionNotFoundError
//!     ├── kglite.PropertyNotFoundError
//!     ├── kglite.FileError                     (FileNotFound)
//!     ├── kglite.FileFormatError
//!     ├── kglite.FileIoError
//!     ├── kglite.ArgumentError
//!     ├── kglite.MissingArgumentError
//!     └── kglite.InternalError
//! ```
//!
//! ## Backward compatibility (pre-A.2 vs A.2)
//!
//! Before A.2, all kglite errors surfaced as `ValueError` /
//! `RuntimeError` / `KeyError` / etc. Existing user code that did
//! `except ValueError:` will NOT automatically catch the new typed
//! exceptions — A.2 is a deliberate break for the sake of clean
//! typing.
//!
//! PyO3's `create_exception!` macro is single-inheritance; combining
//! `kglite.KgError` as a base AND `PyValueError` as an additional
//! base would require Python-level multiple inheritance which PyO3
//! doesn't support cleanly. The user-decided trade-off in
//! `bolt_implementation.md` is consistency-first: every kglite error
//! is now `isinstance(e, kglite.KgError)`.
//!
//! Migration path: change `except ValueError as e:` to
//! `except kglite.CypherSyntaxError as e:` (or
//! `except kglite.KgError as e:` for the catch-all). The CHANGELOG
//! `[Unreleased]` entry names this break loudly.

use pyo3::prelude::*;
use pyo3::types::PyModule;

// Alias Rust types on import — every `create_exception!` macro call
// below generates a Python-side struct (e.g. `KgError`, `SchemaError`)
// in this module, colliding with the Rust enum / pyo3-public types of
// the same names if imported unaliased. The `Rust*` prefix keeps the
// From-impl machinery distinct from the user-facing Python classes.
use crate::error::KgError as RustKgError;

// ─── Exception class declarations (single-inheritance chain) ─────────────────
//
// `create_exception!(module, ClassName, BaseClass, docstring)`. The
// third argument must be a single class. KgError extends PyException
// (Exception); every kglite typed exception extends KgError (or a
// kglite mid-tier like CypherError).

pyo3::create_exception!(
    kglite,
    KgError,
    pyo3::exceptions::PyException,
    "Base class for every kglite-raised exception. Catch this to handle any kglite error."
);

// ── Cypher pipeline ──────────────────────────────────────────────────

pyo3::create_exception!(
    kglite,
    CypherError,
    KgError,
    "Base for all Cypher-related errors (syntax, timeout, execution, type)."
);

pyo3::create_exception!(
    kglite,
    CypherSyntaxError,
    CypherError,
    "Cypher parser / tokenizer rejected the query. Has `.line` and `.col` attributes when known."
);

pyo3::create_exception!(
    kglite,
    CypherTimeoutError,
    CypherError,
    "Cypher query exceeded its `timeout_ms`."
);

pyo3::create_exception!(
    kglite,
    CypherExecutionError,
    CypherError,
    "Cypher executor failure during query evaluation."
);

pyo3::create_exception!(
    kglite,
    CypherTypeMismatchError,
    CypherError,
    "Cypher value-type mismatch in an expression (e.g. arithmetic on a String)."
);

// ── Schema / validation ──────────────────────────────────────────────

pyo3::create_exception!(
    kglite,
    SchemaError,
    KgError,
    "Schema validation failure (unknown property, type mismatch at pattern literal)."
);

pyo3::create_exception!(
    kglite,
    ValidationError,
    KgError,
    "Structural validation failure (missing required field, wrong connection endpoint, etc.)."
);

pyo3::create_exception!(
    kglite,
    ExprError,
    KgError,
    "Blueprint expression evaluation failure."
);

// ── Resource / access ────────────────────────────────────────────────

pyo3::create_exception!(
    kglite,
    NodeNotFoundError,
    KgError,
    "A node identified by `(node_type, id)` doesn't exist."
);

pyo3::create_exception!(
    kglite,
    ConnectionNotFoundError,
    KgError,
    "A connection type isn't declared in the schema."
);

pyo3::create_exception!(
    kglite,
    PropertyNotFoundError,
    KgError,
    "A property is missing from a node or relationship."
);

// ── File / I/O ───────────────────────────────────────────────────────

pyo3::create_exception!(
    kglite,
    FileError,
    KgError,
    "A file the user named doesn't exist on disk."
);

pyo3::create_exception!(
    kglite,
    FileFormatError,
    KgError,
    "A file's contents are malformed (bad .kgl header, truncated blueprint, etc.)."
);

pyo3::create_exception!(
    kglite,
    FileIoError,
    KgError,
    "Generic I/O failure (permission denied, mid-read EOF, mmap failure)."
);

// ── Argument validation ──────────────────────────────────────────────

pyo3::create_exception!(
    kglite,
    ArgumentError,
    KgError,
    "A user-supplied argument violated a precondition."
);

pyo3::create_exception!(
    kglite,
    MissingArgumentError,
    KgError,
    "A required argument wasn't passed."
);

// ── Internal ─────────────────────────────────────────────────────────

pyo3::create_exception!(
    kglite,
    InternalError,
    KgError,
    "Invariant violation — kglite-internal bug. Reports the source location."
);

// ─── PyErr boundary ──────────────────────────────────────────────────────────

/// Convert a Rust [`RustKgError`] into a Python [`PyErr`], picking
/// the most specific subclass for the variant.
///
/// This is the canonical conversion at the PyO3 boundary — every
/// `?` that flows a `Result<T, KgError>` into a `PyResult<T>` goes
/// through this impl.
impl From<RustKgError> for PyErr {
    fn from(e: RustKgError) -> Self {
        let message = e.to_string();
        match e {
            RustKgError::CypherSyntax { .. } => CypherSyntaxError::new_err(message),
            RustKgError::CypherTimeout { .. } => CypherTimeoutError::new_err(message),
            RustKgError::CypherExecution { .. } => CypherExecutionError::new_err(message),
            RustKgError::CypherTypeMismatch { .. } => CypherTypeMismatchError::new_err(message),
            RustKgError::Schema { .. } => SchemaError::new_err(message),
            RustKgError::Validation(_) => ValidationError::new_err(message),
            RustKgError::Expr(_) => ExprError::new_err(message),
            RustKgError::NodeNotFound { .. } => NodeNotFoundError::new_err(message),
            RustKgError::ConnectionNotFound { .. } => ConnectionNotFoundError::new_err(message),
            RustKgError::PropertyNotFound { .. } => PropertyNotFoundError::new_err(message),
            RustKgError::FileNotFound(_) => FileError::new_err(message),
            RustKgError::FileFormat { .. } => FileFormatError::new_err(message),
            RustKgError::FileIo(_) => FileIoError::new_err(message),
            RustKgError::InvalidArgument { .. } | RustKgError::Argument(_) => {
                ArgumentError::new_err(message)
            }
            RustKgError::MissingArgument(_) => MissingArgumentError::new_err(message),
            RustKgError::Internal { .. } => InternalError::new_err(message),
        }
    }
}

// ─── Module registration ─────────────────────────────────────────────────────

/// Register every typed exception class on the `kglite` Python module.
/// Called from `#[pymodule] fn kglite(...)` in `src/lib.rs`.
pub(crate) fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("KgError", py.get_type::<KgError>())?;

    // Cypher pipeline
    m.add("CypherError", py.get_type::<CypherError>())?;
    m.add("CypherSyntaxError", py.get_type::<CypherSyntaxError>())?;
    m.add("CypherTimeoutError", py.get_type::<CypherTimeoutError>())?;
    m.add(
        "CypherExecutionError",
        py.get_type::<CypherExecutionError>(),
    )?;
    m.add(
        "CypherTypeMismatchError",
        py.get_type::<CypherTypeMismatchError>(),
    )?;

    // Schema / validation
    m.add("SchemaError", py.get_type::<SchemaError>())?;
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    m.add("ExprError", py.get_type::<ExprError>())?;

    // Resource / access
    m.add("NodeNotFoundError", py.get_type::<NodeNotFoundError>())?;
    m.add(
        "ConnectionNotFoundError",
        py.get_type::<ConnectionNotFoundError>(),
    )?;
    m.add(
        "PropertyNotFoundError",
        py.get_type::<PropertyNotFoundError>(),
    )?;

    // File / I/O
    m.add("FileError", py.get_type::<FileError>())?;
    m.add("FileFormatError", py.get_type::<FileFormatError>())?;
    m.add("FileIoError", py.get_type::<FileIoError>())?;

    // Argument validation
    m.add("ArgumentError", py.get_type::<ArgumentError>())?;
    m.add(
        "MissingArgumentError",
        py.get_type::<MissingArgumentError>(),
    )?;

    // Internal
    m.add("InternalError", py.get_type::<InternalError>())?;

    Ok(())
}
