//! Python-side machinery for the [`crate::error::KgError`] taxonomy.
//!
//! Phase A.2 of docs/history/bolt-implementation.md — defines the typed Python
//! exception classes (`kglite.CypherSyntaxError`, `kglite.SchemaError`,
//! etc.) via PyO3's `create_exception!` macro, and provides the
//! [`From<KgError> for PyErr`] impl that picks the most specific
//! subclass for each variant at the PyO3 boundary.
//!
//! ## Hierarchy
//!
//! Typed engine errors descend from `kglite.KgError`, which itself descends
//! from `Exception`. Wrapper operations that implement conventional Python
//! protocols may raise built-in exceptions directly.
//!
//! Every instance carries a stable `.code` string (the
//! [`KgErrorCode`](crate::error::KgErrorCode) name, e.g. `"ConstraintViolation"`)
//! so applications branch on a classifier rather than on message prose.
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
//!     ├── kglite.ConstraintError            (declared-integrity base)
//!     │   ├── kglite.ConstraintViolationError
//!     │   └── kglite.ConstraintCreationError
//!     ├── kglite.TransactionConflictError
//!     ├── kglite.NodeNotFoundError
//!     ├── kglite.ConnectionNotFoundError
//!     ├── kglite.PropertyNotFoundError
//!     ├── kglite.FileError                     (FileNotFound)
//!     ├── kglite.FileFormatError
//!     ├── kglite.FileIoError
//!     ├── kglite.ArgumentError
//!     ├── kglite.MissingArgumentError
//!     ├── kglite.InternerCollisionError
//!     └── kglite.InternalError
//! ```
//!
//! ## Built-in exception boundary
//!
//! PyO3's `create_exception!` macro is single-inheritance; combining
//! `kglite.KgError` as a base AND `PyValueError` as an additional
//! base would require Python-level multiple inheritance which PyO3
//! doesn't support cleanly. Engine failures use the typed hierarchy; Python
//! lookup, argument-shape, filesystem, and object-lifecycle conventions keep
//! their documented built-in exception families.

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
    "Cypher parser / tokenizer rejected the query. Always has `.line` and `.col` attributes (1-indexed); both are `None` when the parser couldn't pin a position."
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
    "Cypher executor failure during query evaluation. Has `.line` and `.col` attributes when the failure is pinned to a source position."
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

pyo3::create_exception!(
    kglite,
    ConstraintError,
    KgError,
    "Base class for declared-integrity-constraint failures (UNIQUE / NOT NULL / NODE KEY). Catch this to handle any constraint problem."
);

pyo3::create_exception!(
    kglite,
    ConstraintViolationError,
    ConstraintError,
    "A write violated a declared constraint — a UNIQUE duplicate, or a NOT NULL / NODE KEY property left absent. The write was rejected before touching storage, so the graph is unchanged."
);

pyo3::create_exception!(
    kglite,
    ConstraintCreationError,
    ConstraintError,
    "Declaring a constraint failed because the stored data already violates it. Deduplicate the node type, then re-declare."
);

// ── Concurrency ──────────────────────────────────────────────────────

pyo3::create_exception!(
    kglite,
    TransactionConflictError,
    KgError,
    "An optimistic-concurrency commit lost its race — the graph advanced between `begin()` and `commit()`, so nothing was applied. Retry the whole transaction against a fresh `begin()`; `kglite.retry_on_conflict` does this for you."
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

pyo3::create_exception!(
    kglite,
    InternerCollisionError,
    KgError,
    "Two distinct names collided on the persisted interner key; the operation was rejected unchanged."
);

// ─── PyErr boundary ──────────────────────────────────────────────────────────

/// Convert a Rust [`RustKgError`] into a Python [`PyErr`], picking
/// the most specific subclass for the variant.
///
/// This is the canonical conversion at the PyO3 boundary. Post-Phase
/// G.3a the `impl From<KgError> for PyErr` form is orphan-rule
/// blocked (KgError lives in the kglite engine, PyErr in pyo3, neither
/// local to this crate), so callers explicitly route through this
/// function via `Err(kg_to_pyerr(KgError::Foo(...)))` or
/// `.map_err(kg_to_pyerr)?`.
pub fn kg_to_pyerr(e: RustKgError) -> PyErr {
    let message = e.to_string();
    // Stable classifier, captured before `e` is consumed by the match. Every
    // `kglite.*` exception instance carries it as `.code`, so an application
    // can branch on a wire-stable string (`"ConstraintViolation"`) instead of
    // the message prose — the promise `docs/python/guides/primary-store.md`
    // makes. `Cancelled` is excluded: it maps to the builtin
    // `KeyboardInterrupt`, which is deliberately outside the KgError family.
    let code = e.code().as_str();
    let is_cancelled = matches!(e, RustKgError::Cancelled);
    let err = kg_to_pyerr_class(e, message);
    if is_cancelled {
        return err;
    }
    with_code_attr(err, code)
}

/// Pick the most specific Python exception class for `e` and construct it with
/// `message`. Split from [`kg_to_pyerr`] so the `.code` decoration applies
/// uniformly to every arm rather than being repeated 20 times.
fn kg_to_pyerr_class(e: RustKgError, message: String) -> PyErr {
    match e {
        RustKgError::CypherSyntax { line, col, .. } => {
            // `.line` / `.col` are always present on CypherSyntaxError —
            // `None` when the parser couldn't pin a position.
            with_position_attrs(CypherSyntaxError::new_err(message), line, col)
        }
        RustKgError::CypherTimeout { .. } => CypherTimeoutError::new_err(message),
        RustKgError::CypherExecution { position, .. } => {
            let (line, col) = match position {
                Some((l, c)) => (Some(l), Some(c)),
                None => (None, None),
            };
            with_position_attrs(CypherExecutionError::new_err(message), line, col)
        }
        RustKgError::CypherTypeMismatch { .. } => CypherTypeMismatchError::new_err(message),
        // Cooperative cancellation (the wheel's Ctrl-C handler flipped the
        // cancel flag mid-query) surfaces as the builtin KeyboardInterrupt,
        // not a kglite.* error class — it's an interrupt, not a query fault.
        RustKgError::Cancelled => {
            pyo3::exceptions::PyKeyboardInterrupt::new_err("Query interrupted")
        }
        RustKgError::Schema { .. } => SchemaError::new_err(message),
        RustKgError::Validation(_) => ValidationError::new_err(message),
        RustKgError::ConstraintViolation { .. } => ConstraintViolationError::new_err(message),
        RustKgError::ConstraintCreationFailed { .. } => ConstraintCreationError::new_err(message),
        RustKgError::TransactionConflict { .. } => TransactionConflictError::new_err(message),
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
        RustKgError::InternerCollision(_) => InternerCollisionError::new_err(message),
        RustKgError::Internal { .. } => InternalError::new_err(message),
    }
}

// Post-G.3a: `impl From<RustKgError> for PyErr` would violate
// Rust's orphan rule (both types foreign to this crate — KgError
// lives in the kglite engine, PyErr in pyo3). All call sites use
// `kg_to_pyerr(...)` directly.

/// Set `.line` / `.col` attributes (1-indexed source position) on the
/// exception *value*. `PyErr::new_err` is lazy, so `err.value(py)`
/// normalizes the exception first; attribute assignment on an exception
/// instance can't reasonably fail, but any failure is swallowed rather
/// than masking the original error.
/// Set the stable `.code` classifier on the exception *value*. Same
/// normalize-then-setattr shape as [`with_position_attrs`]; a failure here
/// would mask the real error, so it is swallowed.
fn with_code_attr(err: PyErr, code: &'static str) -> PyErr {
    Python::attach(|py| {
        let value = err.value(py);
        let _ = value.setattr("code", code);
    });
    err
}

fn with_position_attrs(err: PyErr, line: Option<usize>, col: Option<usize>) -> PyErr {
    Python::attach(|py| {
        let value = err.value(py);
        let _ = value.setattr("line", line);
        let _ = value.setattr("col", col);
    });
    err
}

// ─── Module registration ─────────────────────────────────────────────────────

/// Register every typed exception class on the `kglite` Python module.
/// Called from `#[pymodule] fn kglite(...)` in `src/lib.rs`.
pub(crate) fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("KgError", py.get_type::<KgError>())?;

    // Cypher pipeline
    m.add("CypherError", py.get_type::<CypherError>())?;
    m.add("CypherSyntaxError", py.get_type::<CypherSyntaxError>())?;
    // Class-level `.line` / `.col` defaults (None) for the two
    // position-carrying classes: instances raised with a known position
    // shadow these with instance attributes (see `with_position_attrs`),
    // and the attributes stay readable on any instance either way.
    for cls in [
        py.get_type::<CypherSyntaxError>(),
        py.get_type::<CypherExecutionError>(),
    ] {
        cls.setattr("line", py.None())?;
        cls.setattr("col", py.None())?;
    }
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
    m.add("ConstraintError", py.get_type::<ConstraintError>())?;
    m.add(
        "ConstraintViolationError",
        py.get_type::<ConstraintViolationError>(),
    )?;
    m.add(
        "ConstraintCreationError",
        py.get_type::<ConstraintCreationError>(),
    )?;

    // Concurrency
    m.add(
        "TransactionConflictError",
        py.get_type::<TransactionConflictError>(),
    )?;

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
    m.add(
        "InternerCollisionError",
        py.get_type::<InternerCollisionError>(),
    )?;
    m.add("InternalError", py.get_type::<InternalError>())?;

    register_class_codes(py)?;

    Ok(())
}

/// Publish the stable [`KgErrorCode`](crate::error::KgErrorCode) string as a
/// **class-level** `.code` on each concrete exception class, so callers can
/// compare against `kglite.ConstraintViolationError.code` without an instance
/// and `.code` is readable even on an exception constructed by hand.
///
/// Instances raised by the engine shadow these with the code of the actual
/// `KgError` variant (see `with_code_attr`); the two always agree because both
/// come from `KgError::code()`. The three abstract bases — `KgError`,
/// `CypherError`, `ConstraintError` — cover several codes, so they get `None`
/// rather than an arbitrary pick.
fn register_class_codes(py: Python<'_>) -> PyResult<()> {
    use crate::error::KgErrorCode as C;

    py.get_type::<KgError>().setattr("code", py.None())?;
    py.get_type::<CypherError>().setattr("code", py.None())?;
    py.get_type::<ConstraintError>()
        .setattr("code", py.None())?;

    py.get_type::<CypherSyntaxError>()
        .setattr("code", C::CypherSyntax.as_str())?;
    py.get_type::<CypherTimeoutError>()
        .setattr("code", C::CypherTimeout.as_str())?;
    py.get_type::<CypherExecutionError>()
        .setattr("code", C::CypherExecution.as_str())?;
    py.get_type::<CypherTypeMismatchError>()
        .setattr("code", C::CypherTypeMismatch.as_str())?;
    py.get_type::<SchemaError>()
        .setattr("code", C::Schema.as_str())?;
    py.get_type::<ValidationError>()
        .setattr("code", C::Validation.as_str())?;
    py.get_type::<ExprError>()
        .setattr("code", C::Expr.as_str())?;
    py.get_type::<ConstraintViolationError>()
        .setattr("code", C::ConstraintViolation.as_str())?;
    py.get_type::<ConstraintCreationError>()
        .setattr("code", C::ConstraintCreationFailed.as_str())?;
    py.get_type::<TransactionConflictError>()
        .setattr("code", C::TransactionConflict.as_str())?;
    py.get_type::<NodeNotFoundError>()
        .setattr("code", C::NodeNotFound.as_str())?;
    py.get_type::<ConnectionNotFoundError>()
        .setattr("code", C::ConnectionNotFound.as_str())?;
    py.get_type::<PropertyNotFoundError>()
        .setattr("code", C::PropertyNotFound.as_str())?;
    py.get_type::<FileError>()
        .setattr("code", C::FileNotFound.as_str())?;
    py.get_type::<FileFormatError>()
        .setattr("code", C::FileFormat.as_str())?;
    py.get_type::<FileIoError>()
        .setattr("code", C::FileIo.as_str())?;
    py.get_type::<ArgumentError>()
        .setattr("code", C::InvalidArgument.as_str())?;
    py.get_type::<MissingArgumentError>()
        .setattr("code", C::MissingArgument.as_str())?;
    // Both collapse to `Internal` in `KgError::code()`.
    py.get_type::<InternerCollisionError>()
        .setattr("code", C::Internal.as_str())?;
    py.get_type::<InternalError>()
        .setattr("code", C::Internal.as_str())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interner_collision_maps_to_dedicated_python_error() {
        Python::initialize();
        Python::attach(|py| {
            let collision = kglite_core::api::InternerCollision {
                key: 7,
                existing: "first".into(),
                conflicting: "second".into(),
            };
            let error = kg_to_pyerr(RustKgError::InternerCollision(collision));
            assert!(error.is_instance_of::<InternerCollisionError>(py));
        });
    }
}
