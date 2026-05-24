//! `Transaction` `#[pyclass]` + its `#[pymethods]`.
//!
//! Moved out of `graph::mod.rs` in Phase 8.

use crate::datatypes::py_in;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher;
use crate::graph::schema::{CowSelection, DirGraph};
use crate::graph::KnowledgeGraph;
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
///   defers the deep clone until the first mutation lands. Read-only-then-
///   commit cycles pay zero clone cost. Read-write transactions that
///   actually mutate clone on demand. Either way the transaction sees a
///   frozen view of the graph at the moment `begin()` was called.
/// - **Write isolation**: the first mutation triggers
///   `Arc::try_unwrap` (cheap if no other ref) or a deep clone, swapping
///   the transaction from "snapshot-only" mode into "working-copy" mode.
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
/// A read-write transaction has two states:
///   - **Deferred** (initial): `snapshot=Some, working=None`. Reads
///     run against the Arc snapshot.
///   - **Materialized** (after first mutation): `snapshot=None,
///     working=Some`. All reads + writes run against the working copy.
///
/// Read-only transactions stay in the snapshot state for their lifetime.
#[pyclass]
pub struct Transaction {
    /// Back-reference to the owning KnowledgeGraph (for commit)
    pub(crate) owner: Py<KnowledgeGraph>,
    /// Mutable working copy — `Some` only after the first mutation lands
    /// (read-write transactions only).
    pub(crate) working: Option<DirGraph>,
    /// Whether commit() was called
    pub(crate) committed: bool,
    /// Read-only transactions hold an Arc snapshot for their lifetime
    pub(crate) read_only: bool,
    /// Arc snapshot:
    ///   - read-only tx: held for the tx's full lifetime
    ///   - read-write tx in deferred state (before first mutation): held
    ///     until materialization swaps it into `working`
    pub(crate) snapshot: Option<Arc<DirGraph>>,
    /// Graph version at `begin()` time — used for optimistic concurrency control
    pub(crate) base_version: u64,
    /// Optional transaction-level deadline — all operations fail after this instant
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
        self.read_only
    }

    #[pyo3(signature = (query, params=None, to_df=false, timeout_ms=None))]
    fn cypher(
        &mut self,
        py: Python<'_>,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
        to_df: bool,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
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
                let graph = self.snapshot.as_deref().or(self.working.as_ref());
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

        // Phase E.2 — delegate pipeline orchestration to
        // kglite::api::session. The pyapi Transaction keeps its
        // own snapshot/working storage (so commit() can swap the
        // owner KG's Arc directly), but parse+validate+optimize+
        // execute now flows through the canonical session helpers.
        //
        // Decision routing:
        //   - is_mutation + read_only → reject
        //   - is_mutation + RW → materialize working, execute_mut
        //   - read → execute_read against current graph view
        //
        // The pre-parse below is on the cached parser (~700ns hit)
        // so session::execute's own parse inside is free.
        let pre_parsed = cypher::parse_cypher(query)?;
        let is_mut = cypher::is_mutation_query(&pre_parsed);

        if is_mut && self.read_only {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "Read-only transaction does not support mutations \
                 (CREATE, SET, DELETE, REMOVE, MERGE). Use begin() for read-write."
                        .to_string(),
                ),
            ));
        }

        let output_csv = pre_parsed.output_format == cypher::OutputFormat::Csv;
        let opts = crate::graph::session::ExecuteOptions {
            params: &param_map,
            deadline,
            max_rows: None,
            // Transactions historically went through the eager path
            // (mark_lazy off, streaming off) — no lazy materializer
            // is wired through the tx ResultView. Preserve that.
            lazy_eligible: false,
            disabled_passes: None,
            embedder: None,
        };

        let result = if is_mut {
            // Materialize the working copy on first mutation. Arc::try_unwrap
            // skips the clone when this transaction holds the only reference
            // (true when no Python-side ResultView / other tx holds the graph).
            if self.working.is_none() {
                let snap = self.snapshot.take().ok_or_else(|| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                        "Transaction already committed or rolled back".to_string(),
                    ))
                })?;
                let working = Arc::try_unwrap(snap).unwrap_or_else(|arc| (*arc).clone());
                self.working = Some(working);
            }
            let working = self
                .working
                .as_mut()
                .expect("invariant: materialized above");
            crate::graph::session::execute_mut(working, query, &opts)?.result
        } else {
            let graph: &DirGraph = self
                .working
                .as_ref()
                .map(|g| g as &DirGraph)
                .or(self.snapshot.as_deref())
                .ok_or_else(|| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                        "Transaction already committed or rolled back".to_string(),
                    ))
                })?;
            crate::graph::session::execute_read(graph, query, &opts)?.result
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
        if self.read_only {
            // Read-only: just release the snapshot
            if self.snapshot.is_none() && !self.committed {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(
                        "Transaction already committed or rolled back".to_string(),
                    ),
                ));
            }
            self.snapshot = None;
            self.committed = true;
            return Ok(());
        }

        // Read-write transaction. Two sub-cases:
        //   - working.is_some(): the deferred clone got materialized by a
        //     mutation. Run OCC check + Arc swap.
        //   - working.is_none() && snapshot.is_some(): deferred state
        //     never materialized — no writes happened. No-op commit.
        //   - both None: already committed/rolled back; error.
        if let Some(working) = self.working.take() {
            // Optimistic concurrency control: check version hasn't changed.
            let current_version = Python::attach(|py| {
                let kg = self.owner.borrow(py);
                kg.inner.version
            });
            if current_version != self.base_version {
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
                let mut working = working;
                working.version = current_version + 1;
                kg.inner = Arc::new(working);
                kg.selection = CowSelection::new();
            });

            self.committed = true;
            Ok(())
        } else if self.snapshot.is_some() {
            // Deferred state never materialized — no writes, no-op commit.
            self.snapshot = None;
            self.committed = true;
            Ok(())
        } else {
            Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "Transaction already committed or rolled back".to_string(),
                ),
            ))
        }
    }

    /// Roll back the transaction — discard all changes.
    ///
    /// After rollback, the transaction cannot be used again.
    fn rollback(&mut self) -> PyResult<()> {
        if self.read_only {
            if self.snapshot.is_none() {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(
                        "Transaction already committed or rolled back".to_string(),
                    ),
                ));
            }
            self.snapshot = None;
            return Ok(());
        }
        // Read-write: discard whichever container holds state. Both empty
        // = already committed/rolled back.
        if self.working.is_none() && self.snapshot.is_none() {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "Transaction already committed or rolled back".to_string(),
                ),
            ));
        }
        self.working = None;
        self.snapshot = None;
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
        // A transaction is active if either container holds state.
        // After Issue #1 (deferred clone), a read-write tx in the deferred
        // state has snapshot=Some / working=None — still active.
        let is_active = self.snapshot.is_some() || self.working.is_some();

        if !is_active {
            // Already committed or rolled back
            return Ok(false);
        }

        if exc_type.is_some() {
            // Exception occurred — rollback
            self.working = None;
            self.snapshot = None;
        } else {
            // No exception — commit
            self.commit()?;
        }

        // Return false = don't suppress exception
        Ok(false)
    }
}
