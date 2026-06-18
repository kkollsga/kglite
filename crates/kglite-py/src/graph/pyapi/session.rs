//! `Session` — a thread-safe, shareable concurrency handle over a graph.
//!
//! ## Why this exists
//!
//! A live `KnowledgeGraph` is **single-owner**: it is a PyO3 `#[pyclass]`
//! guarded by a `RefCell`-style borrow, and it carries per-caller cursor
//! state (the fluent `selection`). Sharing one across a thread pool and
//! mutating it concurrently trips the borrow guard. That is correct for an
//! ergonomic builder handle, but wrong for a server that fans many agent /
//! request threads at one graph.
//!
//! `Session` is the answer. It wraps the engine's
//! [`kglite_core::graph::session::Session`] — a `Mutex<Arc<DirGraph>>` — and
//! exposes **only** `&self` methods. Synchronisation lives in the Session's
//! mutex, not in PyO3's borrow guard, so:
//!
//! - **Reads** take a momentary snapshot (`Arc::clone`), drop the lock, and
//!   run GIL-free. Any number of threads read the same `Session` in parallel,
//!   lock-free during execution.
//! - **Writes** (see Phase 2 — `execute`) serialise through the mutex:
//!   snapshot → copy-on-write working copy → atomic Arc swap. One writer at a
//!   time; readers never block.
//!
//! ## Relationship to `KnowledgeGraph`
//!
//! `kg.session()` seeds a `Session` from the graph's **current** state. The
//! `Session` is then an **independent owner** — it shares the underlying
//! `Arc<DirGraph>` at creation, but once either side mutates, copy-on-write
//! forks them and they no longer track each other. The intended model is
//! "build / load with a `KnowledgeGraph`, then `.session()` and serve every
//! thread through the `Session`" — mirroring build → freeze → share → swap,
//! but with a mutable shared owner. Don't keep mutating the original
//! `KnowledgeGraph` after handing out a `Session`; treat the `Session` as the
//! live store.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::sync::Arc;

use crate::datatypes::py_in;
use crate::graph::languages::cypher;
use crate::graph::pyapi::frozen::FrozenGraph;
use crate::graph::pyapi::result_view::ResultView;
use crate::graph::session::{execute_read, ExecuteOptions, Session as CoreSession};
use crate::graph::storage::GraphRead;
use crate::graph::{resolve_noderefs, DirGraph};

/// Thread-safe, shareable handle over a graph. See module docs.
///
/// Build or load a graph with a `KnowledgeGraph`, call `.session()`, then
/// share the `Session` across threads: concurrent `cypher()` reads run
/// lock-free; `execute()` writes serialise behind the Session's lock.
#[pyclass]
pub struct Session {
    pub(crate) inner: CoreSession,
    pub(crate) embedder: Option<Arc<dyn crate::graph::embedder::Embedder>>,
}

impl Session {
    /// Construct from a shared graph snapshot + optional embedder. The core
    /// `Session` is `Mutex<Arc<DirGraph>>`; `from_arc` wraps the caller's
    /// existing `Arc` (O(1) — no deep copy).
    pub(crate) fn from_arc(
        inner: Arc<DirGraph>,
        embedder: Option<Arc<dyn crate::graph::embedder::Embedder>>,
    ) -> Self {
        Session {
            inner: CoreSession::from_arc(inner),
            embedder,
        }
    }
}

#[pymethods]
impl Session {
    /// Run a **read-only** Cypher query against a momentary snapshot.
    ///
    /// Takes a snapshot (`Arc::clone`), releases the Session lock, and runs
    /// the query GIL-free — so many threads can call `cypher()` on the same
    /// `Session` at once without blocking each other. Each call sees the
    /// graph as of the moment the snapshot was taken.
    ///
    /// Read semantics are identical to `KnowledgeGraph.cypher` /
    /// `FrozenGraph.cypher`. A mutation query
    /// (`CREATE` / `SET` / `DELETE` / `REMOVE` / `MERGE`) is rejected — use
    /// `Session.execute()` for writes.
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
        let pre_parsed = cypher::parse_cypher(query).map_err(crate::error_py::kg_to_pyerr)?;
        if cypher::is_mutation_query(&pre_parsed) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Session.cypher() is read-only — CREATE/SET/DELETE/REMOVE/MERGE are not \
                 allowed here. Use Session.execute() for serialized writes.",
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

        // Momentary lock acquire to clone the current snapshot, then release.
        let inner = self.inner.snapshot();
        let embedder = self.embedder.clone();
        let query_owned = query.to_string();
        let result = py
            .detach(
                move || -> Result<cypher::result::CypherResult, crate::error::KgError> {
                    let opts = ExecuteOptions {
                        params: &param_map,
                        deadline,
                        max_rows,
                        lazy_eligible: false,
                        disabled_passes: None,
                        embedder,
                        value_codecs: None,
                    };
                    let outcome = execute_read(&inner, &query_owned, &opts)?;
                    let mut result = outcome.result;
                    resolve_noderefs(&inner.graph, &mut result.rows);
                    Ok(result)
                },
            )
            .map_err(crate::error_py::kg_to_pyerr)?;

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

    /// Take an immutable, concurrently-readable snapshot of the current state.
    ///
    /// Returns a `FrozenGraph` — an O(1) `Arc` clone — that stays stable even
    /// if the `Session` is later written to (copy-on-write forks the writer).
    /// Use this to hold a consistent multi-query view, or to hand a fixed
    /// read snapshot to a pool of readers.
    fn snapshot(&self) -> FrozenGraph {
        FrozenGraph::new(self.inner.snapshot(), self.embedder.clone())
    }

    /// Monotonic version of the current graph. Bumped by each committed
    /// write. Useful for cheap "did anything change?" checks.
    fn version(&self) -> u64 {
        self.inner.version()
    }

    /// Number of nodes in the current snapshot.
    fn node_count(&self) -> usize {
        self.inner.snapshot().graph.node_count()
    }

    /// Node type names present in the current snapshot.
    #[getter]
    fn node_types(&self) -> Vec<String> {
        self.inner.snapshot().get_node_types()
    }

    fn __repr__(&self) -> String {
        let snap = self.inner.snapshot();
        format!(
            "Session(nodes={}, types={}, version={})",
            snap.graph.node_count(),
            snap.get_node_types().len(),
            self.inner.version(),
        )
    }
}
