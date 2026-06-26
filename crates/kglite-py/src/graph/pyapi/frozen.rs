//! `FrozenGraph` — an immutable, concurrently-readable snapshot.
//!
//! `KnowledgeGraph.freeze()` returns one of these. It shares the source
//! graph's `Arc<DirGraph>` (an O(1) clone — no deep copy), exposes *only*
//! read methods, and never takes an exclusive borrow. Because it has no
//! mutating method, no `borrow_mut` can ever fire — so any number of
//! threads can call `cypher()` on the *same* frozen handle concurrently
//! without tripping the single-owner borrow guard that a live
//! `KnowledgeGraph` enforces.
//!
//! Copy-on-write makes the snapshot stable: if the source graph is later
//! mutated, `Arc::make_mut` clones it, leaving this frozen view pointing
//! at the original bytes. That is the "build → freeze → share → swap"
//! model — build a fresh graph cheaply, `freeze()` it, hand it to readers,
//! and atomically swap in a new frozen snapshot when the data changes.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::sync::Arc;

use crate::datatypes::py_in;
use crate::graph::languages::cypher;
use crate::graph::pyapi::result_view::ResultView;
use crate::graph::{resolve_noderefs, DirGraph};
use crate::util::EnterKg;
use kglite_core::api::session::{execute_read, ExecuteOptions};
use kglite_core::api::GraphRead;

/// Immutable, `Send`-able read snapshot of a graph. See module docs.
#[pyclass(frozen)]
pub struct FrozenGraph {
    pub(crate) inner: Arc<DirGraph>,
    pub(crate) embedder: Option<Arc<dyn crate::graph::embedder::Embedder>>,
}

impl FrozenGraph {
    /// Construct from a shared graph snapshot + optional embedder. O(1) —
    /// the `Arc` is cloned by the caller (`KnowledgeGraph::freeze`).
    pub(crate) fn new(
        inner: Arc<DirGraph>,
        embedder: Option<Arc<dyn crate::graph::embedder::Embedder>>,
    ) -> Self {
        FrozenGraph { inner, embedder }
    }
}

#[pymethods]
impl FrozenGraph {
    /// Run a **read-only** Cypher query against the snapshot.
    ///
    /// Identical semantics to `KnowledgeGraph.cypher` for reads —
    /// `MATCH` / `WHERE` / `RETURN` / aggregations, and semantic search via
    /// `text_score()` / `vector_score()`. A mutation query
    /// (`CREATE` / `SET` / `DELETE` / `REMOVE` / `MERGE`) is rejected: a
    /// frozen snapshot is immutable — mutate the source `KnowledgeGraph`,
    /// then take a fresh `freeze()`.
    ///
    /// Safe to call from many threads on the same `FrozenGraph` at once.
    #[pyo3(signature = (query, to_df=false, params=None, timeout_ms=None, max_rows=None))]
    fn cypher(
        &self,
        py: Python<'_>,
        query: &str,
        to_df: bool,
        params: Option<&Bound<'_, PyDict>>,
        timeout_ms: Option<u64>,
        max_rows: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        // Reject mutations up front with a frozen-specific message (clearer
        // than execute_read's generic "use execute_mut").
        let pre_parsed = cypher::parse_cypher(query).map_err(crate::error_py::kg_to_pyerr)?;
        if cypher::is_mutation_query(&pre_parsed) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "FrozenGraph is an immutable snapshot — CREATE/SET/DELETE/REMOVE/MERGE are \
                 not allowed. Mutate the source KnowledgeGraph, then take a fresh freeze().",
            ));
        }

        // Decode params (PyDict → HashMap) under the GIL, before detaching.
        let param_map = if let Some(params_dict) = params {
            let mut map = std::collections::HashMap::new();
            for (key, val) in params_dict.iter() {
                let key_str: String = key.extract()?;
                map.insert(key_str, py_in::py_value_to_value(&val)?);
            }
            map
        } else {
            std::collections::HashMap::new()
        };
        // timeout_ms == 0 is the documented "no deadline" escape hatch.
        let deadline = match timeout_ms {
            Some(0) | None => None,
            Some(ms) => Some(std::time::Instant::now() + std::time::Duration::from_millis(ms)),
        };

        let inner = Arc::clone(&self.inner);
        let embedder = self.embedder.clone();
        let query_owned = query.to_string();
        // GIL-free execution — the whole point of a frozen snapshot is that
        // many readers run in parallel against the shared, immutable graph.
        let result = py.enter_kg(
            move |cancel| -> Result<cypher::CypherResult, crate::error::KgError> {
                let opts = ExecuteOptions {
                    params: &param_map,
                    deadline,
                    max_rows,
                    lazy_eligible: false,
                    disabled_passes: None,
                    embedder,
                    value_codecs: None,
                    cancel,
                    // FrozenGraph is read-only — write-scope + provenance never apply.
                    write_scope: None,
                    git_sha: None,
                    modified_by: None,
                };
                let outcome = execute_read(&inner, &query_owned, &opts)?;
                let mut result = outcome.result;
                resolve_noderefs(&inner.graph, &mut result.rows);
                Ok(result)
            },
        )?;

        if pre_parsed.output_format == cypher::OutputFormat::Csv {
            return result.to_csv().into_py_any(py);
        }
        if to_df {
            let preprocessed = cypher::py_convert::preprocess_values_owned(result.rows);
            cypher::py_convert::preprocessed_result_to_dataframe(py, &result.columns, &preprocessed)
        } else {
            let view = ResultView::from_cypher_result(result);
            Py::new(py, view).map(|v| v.into_any())
        }
    }

    /// Number of nodes in the snapshot (excludes the internal schema node).
    fn node_count(&self) -> usize {
        // Mirror KnowledgeGraph.node_count: total live nodes minus the
        // reserved schema node when present.
        self.inner.graph.node_count()
    }

    /// Node type names present in the snapshot.
    #[getter]
    fn node_types(&self) -> Vec<String> {
        self.inner.get_node_types()
    }

    fn __repr__(&self) -> String {
        format!(
            "FrozenGraph(nodes={}, types={})",
            self.inner.graph.node_count(),
            self.inner.get_node_types().len()
        )
    }
}
