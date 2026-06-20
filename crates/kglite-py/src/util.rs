//! Cross-cutting helpers for the PyO3 boundary.
//!
//! [`EnterKg`] is the single chokepoint for "release the GIL, run a
//! blocking chunk of engine work, map any [`KgError`] back to the typed
//! Python exception". Before this existed, every `#[pymethods]` body
//! open-coded `py.detach(move || ...).map_err(crate::error_py::kg_to_pyerr)`,
//! which made it easy to forget the GIL release or the error mapping and
//! left no single place to add cancellation (Phase 2 hangs the SIGINT /
//! cancel-flag plumbing off this trait).
//!
//! Mirrors the role polars' `EnterPolarsExt::enter_polars`
//! (`crates/polars-python/src/utils.rs`) plays in that binding.

use pyo3::marker::Ungil;
use pyo3::prelude::*;

use crate::error::KgError;
use crate::error_py::kg_to_pyerr;

/// GIL-release + error-mapping helper, implemented on [`Python`].
///
/// Wrap any blocking engine call so the GIL is released for its duration
/// (letting other Python threads run, and — Phase 2 — letting a pending
/// `KeyboardInterrupt` be observed) and so a returned [`KgError`] is
/// converted to the most specific `kglite.*` Python exception.
pub(crate) trait EnterKg {
    /// Release the GIL, run `f`, and map `Err(e)` through
    /// [`kg_to_pyerr`]. The fallible counterpart used by every Cypher /
    /// load path.
    fn enter_kg<T, E, F>(self, f: F) -> PyResult<T>
    where
        F: Ungil + Send + FnOnce() -> Result<T, E>,
        T: Ungil + Send,
        E: Ungil + Send + Into<KgError>;
}

impl EnterKg for Python<'_> {
    #[inline]
    fn enter_kg<T, E, F>(self, f: F) -> PyResult<T>
    where
        F: Ungil + Send + FnOnce() -> Result<T, E>,
        T: Ungil + Send,
        E: Ungil + Send + Into<KgError>,
    {
        self.detach(f).map_err(|e| kg_to_pyerr(e.into()))
    }
}
