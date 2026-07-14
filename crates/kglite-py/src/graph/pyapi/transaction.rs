//! `Transaction` `#[pyclass]` + its `#[pymethods]`.
//!
//! Moved out of `graph::mod.rs` in Phase 8.

use crate::datatypes::py_in;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher;
use crate::graph::KnowledgeGraph;
use kglite_core::api::session::Transaction as CoreTransaction;
use kglite_core::api::CowSelection;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{Bound, IntoPyObjectExt};
use std::collections::HashMap;
use std::sync::Arc;

/// Mutable working copy during a transaction.
///
/// Created by `graph.begin()`, provides a separate `DirGraph` that can be
/// modified without affecting the original. Call `commit()` to apply changes
/// back, or let it drop to discard.
///
/// ## Isolation semantics
///
/// - **Snapshot isolation**: `begin()` takes an `Arc` snapshot (O(1)) and
///   defers its backend-specific working fork until the first mutation.
///   Memory/mapped modes clone then; disk mode remaps immutable bases and
///   copies only overlays. Read-only cycles pay no fork cost.
/// - **Write isolation**: the first mutation swaps the transaction from
///   snapshot-only mode into its isolated backend working fork.
///   Subsequent mutations all land on the working copy without touching
///   the original graph.
/// - **Commit**: `commit()` of a no-write transaction is a no-op (no
///   version bump, no Arc swap). `commit()` of a tx that did mutate
///   replaces the owner's `Arc<DirGraph>` with the working copy via an
///   atomic pointer swap.
/// - **No concurrent-transaction guarantees**: if two transactions are
///   created from the same graph, each gets an independent snapshot.
///   The first commit wins; the second raises a `Transaction conflict`
///   error via optimistic concurrency control (version check).
/// - **No read-snapshot across transactions**: reads on the original graph
///   while a transaction is open will see the pre-transaction state. After
///   commit, they see the post-transaction state.
///
/// ## State transitions
///
/// The snapshot/working/CoW/OCC state machine is delegated to the engine's
/// [`CoreTransaction`] (`kglite_core::graph::session::Transaction`) — the same
/// type the bolt-server drives — so deferred forking
/// materialisation, and version tracking live in one place (Phase E). This
/// wrapper adds only the binding-specific concerns: the owning
/// `KnowledgeGraph` (so `commit()` swaps *its* `Arc`), the optional
/// transaction-level deadline, and the Python result marshalling.
///   - **Deferred** (initial): `CoreTransaction::current()` reads the Arc
///     snapshot; no clone cost.
///   - **Materialized** (after first mutation): `CoreTransaction::working_mut()`
///     materialises the working copy (in place if uniquely held, else a deep
///     clone). All reads + writes run against it.
///
/// `inner` is `None` after `commit()` / `rollback()` — any further use errors.
#[pyclass(module = "kglite")]
pub struct Transaction {
    /// Back-reference to the owning KnowledgeGraph (for commit).
    pub(crate) owner: Py<KnowledgeGraph>,
    /// The engine transaction holding the snapshot/working/CoW/OCC state.
    /// `None` once the transaction has been committed or rolled back.
    pub(crate) inner: Option<CoreTransaction>,
    /// Optional transaction-level deadline — all operations fail after this instant.
    pub(crate) deadline: Option<std::time::Instant>,
}

#[pymethods]
impl Transaction {
    /// Execute a Cypher query within this transaction.
    ///
    /// Mutations are applied to the transaction's working copy, not the original graph.
    /// Read queries also operate on the working copy (seeing uncommitted changes).
    ///
    /// Args:
    ///     query: A Cypher query string.
    ///     params: Optional dict of query parameters.
    ///     to_df: If True, return a pandas DataFrame instead of list of dicts.
    ///
    /// Returns:
    ///     Query results (same format as KnowledgeGraph.cypher).
    /// Whether this is a read-only transaction.
    #[getter]
    fn is_read_only(&self) -> bool {
        self.inner
            .as_ref()
            .is_some_and(CoreTransaction::is_read_only)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, params=None, to_df=false, timeout_ms=None, max_rows=None, write_scope=None))]
    fn cypher(
        &mut self,
        py: Python<'_>,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
        to_df: bool,
        timeout_ms: Option<u64>,
        max_rows: Option<usize>,
        write_scope: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let write_scope_set: Option<std::collections::HashSet<String>> =
            write_scope.map(|v| v.into_iter().collect());
        // Check transaction-level deadline first
        if let Some(tx_deadline) = self.deadline {
            if std::time::Instant::now() >= tx_deadline {
                // Phase A.3 / 0.9.53 — typed exception (was PyTimeoutError).
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::CypherTimeout {
                        elapsed_ms: 0,
                        limit_ms: 0,
                    },
                ));
            }
        }

        // Merge per-query timeout with transaction deadline (use the earlier one).
        // timeout_ms == 0 is the documented escape hatch: "no per-query deadline"
        // (the transaction-level deadline still applies if set).
        let effective_timeout_ms = match timeout_ms {
            Some(0) => None,
            Some(ms) => Some(ms),
            None => {
                // Fall through to the graph's backend-aware default.
                let graph = self.inner.as_ref().and_then(CoreTransaction::current);
                graph.and_then(super::kg_core::backend_default_timeout_ms)
            }
        };
        let query_deadline = effective_timeout_ms
            .map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        let deadline = match (self.deadline, query_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        // Convert params
        let param_map: HashMap<String, Value> = match params {
            Some(d) => {
                let mut map = HashMap::new();
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    let val = py_in::py_value_to_value(&v)?;
                    map.insert(key, val);
                }
                map
            }
            None => HashMap::new(),
        };

        // Phase E (completed) — both the execution pipeline and the
        // snapshot/working state machine now live in core: the engine
        // `CoreTransaction` holds the snapshot/working copy and
        // `session::execute_*` runs parse+validate+optimize+execute. This
        // wrapper only routes and marshals.
        //
        // Decision routing:
        //   - is_mutation + read_only → reject (begin()-appropriate message)
        //   - is_mutation + RW → tx.working_mut() (materialize), execute_mut
        //   - read → execute_read against tx.current() (working or snapshot)
        //
        // The pre-parse below is on the cached parser (~700ns hit)
        // so session::execute's own parse inside is free.
        let pre_parsed = cypher::parse_cypher(query).map_err(crate::error_py::kg_to_pyerr)?;
        let is_mut = cypher::is_mutation_query(&pre_parsed);

        // The engine transaction owns the snapshot/working state; `None` means
        // the tx was already committed or rolled back.
        let tx = self.inner.as_mut().ok_or_else(|| -> PyErr {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                "Transaction already committed or rolled back".to_string(),
            ))
        })?;

        // Reject mutations on a read-only tx with a begin()-appropriate message
        // (core's working_mut() message references Session::begin, the wrong
        // entry point for a wheel `graph.begin_read()` caller).
        if is_mut && tx.is_read_only() {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "Read-only transaction does not support mutations \
                 (CREATE, SET, DELETE, REMOVE, MERGE). Use begin() for read-write."
                        .to_string(),
                ),
            ));
        }

        let output_csv = pre_parsed.output_format == cypher::OutputFormat::Csv;
        let opts = kglite_core::api::session::ExecuteOptions {
            params: &param_map,
            deadline,
            max_rows,
            // Transactions historically went through the eager path
            // (mark_lazy off, streaming off) — no lazy materializer
            // is wired through the tx ResultView. Preserve that.
            lazy_eligible: false,
            disabled_passes: None,
            embedder: None,
            value_codecs: None,
            // Cancellation is NOT wired on the transaction path: `working_mut`
            // mutates in place when the graph is uniquely held, so an aborted
            // mutation isn't reliably rolled back (a Ctrl-C could leave partial
            // state). For interruptible + atomic mutations use `Session.execute`
            // (separate working copy + atomic-swap commit). The deadline applies.
            cancel: None,
            write_scope: write_scope_set.as_ref(),
            git_sha: None,
            modified_by: None,
        };

        let result = if is_mut {
            // `working_mut` materialises the backend-specific working copy on
            // first mutation and rejects writes on a read-only tx — both in core.
            let working = tx.working_mut().map_err(crate::error_py::kg_to_pyerr)?;
            kglite_core::api::session::execute_mut(working, query, &opts)
                .map_err(crate::error_py::kg_to_pyerr)?
                .result
        } else {
            // `current()` returns the working copy if materialised, else the
            // snapshot — so reads see uncommitted changes.
            let graph = tx.current().ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                    "Transaction already committed or rolled back".to_string(),
                ))
            })?;
            kglite_core::api::session::execute_read(graph, query, &opts)
                .map_err(crate::error_py::kg_to_pyerr)?
                .result
        };

        if pre_parsed.explain {
            let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
            return Py::new(py, view).map(|v| v.into_any());
        }
        if output_csv {
            result.to_csv().into_py_any(py)
        } else if to_df {
            let preprocessed = cypher::py_convert::preprocess_values_owned(result.rows);
            cypher::py_convert::preprocessed_result_to_dataframe(py, &result.columns, &preprocessed)
        } else {
            let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
            Py::new(py, view).map(|v| v.into_any())
        }
    }

    /// Commit the transaction — apply all changes to the original graph.
    ///
    /// For read-only transactions, this is a no-op.
    /// For a read-write transaction that performed no mutations (deferred
    /// state never materialized), this is also a no-op — no version bump,
    /// no Arc swap, no OCC check needed.
    /// After commit, the transaction cannot be used again.
    fn commit(&mut self) -> PyResult<()> {
        let tx = self.inner.take().ok_or_else(|| -> PyErr {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                "Transaction already committed or rolled back".to_string(),
            ))
        })?;

        // `take_working()` yields the working copy (Some only if a mutation
        // materialised it) plus the version captured at begin(). No working
        // copy → read-only or deferred-never-materialised → no-op commit.
        let (working, base_version) = tx.take_working();
        let Some(mut working) = working else {
            return Ok(());
        };

        // Optimistic concurrency control: the owner graph must not have moved
        // since begin(). (The OCC check stays here because the commit target
        // is the owner KnowledgeGraph's Arc, not a core Session's.)
        let current_version = Python::attach(|py| self.owner.borrow(py).inner.version);
        if current_version != base_version {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "Transaction conflict: graph was modified since begin(). \
                     Retry the transaction."
                        .to_string(),
                ),
            ));
        }

        Python::attach(|py| {
            let mut kg = self.owner.borrow_mut(py);
            working.set_version(current_version + 1);
            kg.inner = Arc::new(working);
            kg.cursor.selection = CowSelection::new();
        });
        Ok(())
    }

    /// Roll back the transaction — discard all changes.
    ///
    /// After rollback, the transaction cannot be used again.
    fn rollback(&mut self) -> PyResult<()> {
        // Dropping the engine transaction discards its working copy / snapshot.
        if self.inner.take().is_none() {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "Transaction already committed or rolled back".to_string(),
                ),
            ));
        }
        Ok(())
    }

    /// Context manager entry — returns self.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit — commits on success, rolls back on exception.
    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_val: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        // A transaction is active while it still holds engine state.
        if self.inner.is_none() {
            // Already committed or rolled back
            return Ok(false);
        }

        if exc_type.is_some() {
            // Exception occurred — rollback (drop the engine transaction).
            self.inner = None;
        } else {
            // No exception — commit
            self.commit()?;
        }

        // Return false = don't suppress exception
        Ok(false)
    }
}
