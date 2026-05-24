//! [`PyEmbedderAdapter`] — wraps a user-provided Python embedder
//! class so the Cypher engine and `embed_texts` / `search_text`
//! pymethods can dispatch through the generic [`Embedder`] trait.
//!
//! This is the bridge that keeps the Python API
//! (`g.set_embedder(my_python_class)`) working unchanged after the
//! 0.9.18 embedder-trait refactor. The user's class must implement
//! the existing `EmbeddingModel` Protocol documented in
//! `kglite/__init__.pyi`: `dimension: int`, `embed(texts) ->
//! list[list[float]]`, optional `load()` / `unload()`.

use pyo3::prelude::*;

use super::Embedder;

/// Holds a `Py<PyAny>` pointing at the user's embedder instance.
///
/// Acquires the GIL on every trait call to invoke the Python methods.
/// This is unavoidable given the API contract — the user's `embed()`
/// implementation runs Python code (typically wrapping
/// `sentence-transformers`). For binaries that don't want libpython
/// in the link, use [`super::fastembed::FastEmbedAdapter`] instead;
/// it has no PyO3 dependency.
pub struct PyEmbedderAdapter {
    instance: Py<PyAny>,
    dimension: usize,
}

impl PyEmbedderAdapter {
    /// Build an adapter around the user's embedder instance.
    /// Eagerly reads `instance.dimension` at construction so subsequent
    /// `dimension()` calls don't need the GIL.
    pub fn new(py: Python<'_>, instance: Py<PyAny>) -> PyResult<Self> {
        let dimension: usize = instance.bind(py).getattr("dimension")?.extract()?;
        Ok(Self {
            instance,
            dimension,
        })
    }

    /// Borrow the underlying Python object. Used by call sites that
    /// need GIL-attached access for reasons beyond the trait surface
    /// (e.g. legacy `try_load_embedder` shape during the transition).
    #[allow(dead_code)] // Reserved for downstream callers; kept on the API for symmetry.
    pub fn instance(&self) -> &Py<PyAny> {
        &self.instance
    }
}

impl Embedder for PyEmbedderAdapter {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Python::attach(|py| -> PyResult<Vec<Vec<f32>>> {
            let py_texts = pyo3::types::PyList::new(py, texts)?;
            let result = self.instance.call_method1(py, "embed", (py_texts,))?;
            result.extract(py)
        })
        .map_err(|e| format!("embedder.embed() failed: {e}"))
    }

    fn load(&self) -> Result<(), String> {
        Python::attach(|py| -> PyResult<()> {
            let bound = self.instance.bind(py);
            if bound.hasattr("load")? {
                bound.call_method0("load")?;
            }
            Ok(())
        })
        .map_err(|e| format!("embedder.load() failed: {e}"))
    }

    fn unload(&self) {
        let _ = Python::attach(|py| -> PyResult<()> {
            let bound = self.instance.bind(py);
            if bound.hasattr("unload").unwrap_or(false) {
                let _ = bound.call_method0("unload");
            }
            Ok(())
        });
    }
}
