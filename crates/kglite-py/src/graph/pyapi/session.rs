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
//! exposes **only** `&self` methods. Synchronisation lives in the Session, not
//! in PyO3's borrow guard, so:
//!
//! - **Reads** (`cypher`, `snapshot`) take a momentary snapshot
//!   (`Arc::clone`), drop the lock, and run GIL-free. Any number of threads
//!   read the same `Session` in parallel, lock-free during execution.
//! - **Writes** (`execute`) serialise behind a writer lock held across the
//!   whole mutation. The core Session mutates its Arc in place when no reader
//!   snapshot is alive and copy-on-write forks once otherwise. The writer lock makes concurrent writes
//!   *compose* — writer B's `begin()` snapshots writer A's committed state, so
//!   B builds on A's changes rather than racing and silently overwriting them
//!   (the lost-update failure mode of a naive shared mutable handle). Readers
//!   that already hold a snapshot never block on the writer; a new reader may
//!   briefly wait while a unique-owner write holds the core graph mutex.
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
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::datatypes::py_in;
use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::languages::cypher;
use crate::graph::pyapi::frozen::FrozenGraph;
use crate::graph::pyapi::result_view::ResultView;
use crate::graph::{resolve_noderefs, DirGraph};
use crate::util::EnterKg;
use kglite_core::api::session::{
    execute_mut, execute_read, ExecuteOptions, Session as CoreSession,
};
use kglite_core::api::GraphRead;

/// Thread-safe, shareable handle over a graph. See module docs.
///
/// Build or load a graph with a `KnowledgeGraph`, call `.session()`, then
/// share the `Session` across threads: concurrent `cypher()` reads run
/// lock-free; `execute()` writes serialise behind the Session's writer lock.
#[pyclass(module = "kglite", frozen)]
pub struct Session {
    pub(crate) inner: CoreSession,
    pub(crate) embedder: Option<Arc<dyn crate::graph::embedder::Embedder>>,
    /// Serialises writers. Held across the whole `begin → mutate → commit` so
    /// concurrent `execute()` calls compose (each sees prior commits) instead
    /// of racing into a lost update. Readers never touch it.
    pub(crate) write_lock: Mutex<()>,
}

/// Decode an optional Python params dict into a native param map under the
/// GIL (must happen before any `py.detach`).
fn decode_params(params: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, Value>> {
    let mut map = HashMap::new();
    if let Some(params_dict) = params {
        for (key, val) in params_dict.iter() {
            let key_str: String = key.extract()?;
            map.insert(key_str, py_in::py_value_to_value(&val)?);
        }
    }
    Ok(map)
}

/// `timeout_ms == 0` is the documented "no deadline" escape hatch.
fn deadline_from(timeout_ms: Option<u64>) -> Option<std::time::Instant> {
    match timeout_ms {
        Some(0) | None => None,
        Some(ms) => Some(std::time::Instant::now() + std::time::Duration::from_millis(ms)),
    }
}

/// Decoded per-call query options shared by the read and write paths.
struct QueryOpts {
    to_df: bool,
    deadline: Option<std::time::Instant>,
    max_rows: Option<usize>,
    output_csv: bool,
    /// Role-scoped write whitelist; only consulted on the write path.
    write_scope: Option<std::collections::HashSet<String>>,
}

impl QueryOpts {
    fn from_parts(
        to_df: bool,
        timeout_ms: Option<u64>,
        max_rows: Option<usize>,
        csv: bool,
        write_scope: Option<std::collections::HashSet<String>>,
    ) -> Self {
        QueryOpts {
            to_df,
            deadline: deadline_from(timeout_ms),
            max_rows,
            output_csv: csv,
            write_scope,
        }
    }
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
            write_lock: Mutex::new(()),
        }
    }

    /// Read path: snapshot → GIL-free `execute_read` → marshal. Shared by
    /// `cypher` and `execute`'s non-mutation fast path (so a read passed to
    /// `execute` never materialises a working copy).
    // The detached closure preserves the engine's structured KgError until PyErr conversion.
    #[allow(clippy::result_large_err)]
    fn run_read(
        &self,
        py: Python<'_>,
        query: &str,
        param_map: HashMap<String, Value>,
        qopts: QueryOpts,
    ) -> PyResult<Py<PyAny>> {
        let inner = self.inner.snapshot();
        let embedder = self.embedder.clone();
        let query_owned = query.to_string();
        let deadline = qopts.deadline;
        let max_rows = qopts.max_rows;
        let result = py.enter_kg(move |cancel| -> Result<cypher::CypherResult, KgError> {
            let opts = ExecuteOptions {
                params: &param_map,
                deadline,
                max_rows,
                lazy_eligible: false,
                disabled_passes: None,
                embedder,
                value_codecs: None,
                cancel,
                write_scope: None,
                git_sha: None,
                modified_by: None,
            };
            let outcome = execute_read(&inner, &query_owned, &opts)?;
            let mut result = outcome.result;
            resolve_noderefs(&inner.graph, &mut result.rows);
            Ok(result)
        })?;
        marshal_result(py, result, qopts.to_df, qopts.output_csv)
    }

    /// Write path: take the writer lock, then hold the core Session's mutable
    /// guard for the complete GIL-free mutation. This avoids the old
    /// begin-created Arc clone that made the unique-owner path unreachable.
    // The detached closure preserves the engine's structured KgError until PyErr conversion.
    #[allow(clippy::result_large_err)]
    fn run_write(
        &self,
        py: Python<'_>,
        query: &str,
        param_map: HashMap<String, Value>,
        qopts: QueryOpts,
    ) -> PyResult<Py<PyAny>> {
        let core = &self.inner;
        let write_lock = &self.write_lock;
        let query_owned = query.to_string();
        let deadline = qopts.deadline;
        let max_rows = qopts.max_rows;
        let write_scope = qopts.write_scope;
        // Mutations don't take an embedder snapshot (matches the live
        // KnowledgeGraph mutation path — text_score in a write is atypical and
        // would force a GIL re-acquire inside the detached block).
        let result = py.enter_kg(move |cancel| -> Result<cypher::CypherResult, KgError> {
            // Acquire the writer lock *with the GIL released* (we are
            // already inside py.detach). Locking before the detach
            // would deadlock: a waiting writer would hold the GIL while
            // blocking on the lock, and the lock-holder needs the GIL
            // back to return. Poison-recover — the graph swaps
            // atomically, so a prior writer's panic doesn't cascade.
            let _wguard = write_lock.lock().unwrap_or_else(|p| p.into_inner());
            let mut graph = core.write();
            let opts = ExecuteOptions {
                params: &param_map,
                deadline,
                max_rows,
                lazy_eligible: false,
                disabled_passes: None,
                embedder: None,
                value_codecs: None,
                cancel,
                write_scope: write_scope.as_ref(),
                git_sha: None,
                modified_by: None,
            };
            let outcome = execute_mut(&mut graph, &query_owned, &opts)?;
            let mut result = outcome.result;
            // Resolve NodeRefs against the working graph before commit
            // consumes the transaction.
            resolve_noderefs(&graph.graph, &mut result.rows);
            Ok(result)
        })?;
        marshal_result(py, result, qopts.to_df, qopts.output_csv)
    }
}

/// Marshal a `CypherResult` into the Python return shape (CSV string / pandas
/// DataFrame / `ResultView`).
fn marshal_result(
    py: Python<'_>,
    result: cypher::CypherResult,
    to_df: bool,
    output_csv: bool,
) -> PyResult<Py<PyAny>> {
    if output_csv {
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
        let param_map = decode_params(params)?;
        let output_csv = pre_parsed.output_format == cypher::OutputFormat::Csv;
        let qopts = QueryOpts::from_parts(to_df, timeout_ms, max_rows, output_csv, None);
        self.run_read(py, query, param_map, qopts)
    }

    /// Run a Cypher **write** against the shared graph, serialized.
    ///
    /// Mutations (`CREATE` / `SET` / `DELETE` / `REMOVE` / `MERGE`) take the
    /// Session's writer lock for the duration of the mutation,
    /// so concurrent `execute()` calls run one at a time and each sees the
    /// prior writer's committed changes — no lost updates. The commit is an
    /// Readers already holding snapshots keep seeing the pre-write graph. A
    /// reader arriving during a unique-owner write may briefly wait for the
    /// core graph mutex.
    ///
    /// A read-only query passed to `execute()` is fast-pathed to the read
    /// path (no working-copy materialisation), so it is always safe to route
    /// mixed traffic through `execute()`.
    ///
    /// Returns the query result (rows for `... RETURN`, otherwise mutation
    /// stats), same shape as `KnowledgeGraph.cypher`.
    #[pyo3(signature = (query, to_df=false, params=None, timeout_ms=None, max_rows=None, write_scope=None))]
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        py: Python<'_>,
        query: &str,
        to_df: bool,
        params: Option<&Bound<'_, PyDict>>,
        timeout_ms: Option<u64>,
        max_rows: Option<usize>,
        write_scope: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let pre_parsed = cypher::parse_cypher(query).map_err(crate::error_py::kg_to_pyerr)?;
        let param_map = decode_params(params)?;
        let output_csv = pre_parsed.output_format == cypher::OutputFormat::Csv;
        let scope_set = write_scope.map(|v| v.into_iter().collect());
        let qopts = QueryOpts::from_parts(to_df, timeout_ms, max_rows, output_csv, scope_set);
        if cypher::is_mutation_query(&pre_parsed) {
            self.run_write(py, query, param_map, qopts)
        } else {
            self.run_read(py, query, param_map, qopts)
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

    /// Spawn a per-thread **query cursor**: a `KnowledgeGraph` bound to a
    /// snapshot of this session's current state, with a fresh fluent cursor.
    ///
    /// Where `snapshot()` hands out a read-only `FrozenGraph` (just `cypher()`),
    /// `cursor()` hands out the **full fluent surface** — `select` / `where` /
    /// `sort` / `traverse` / `to_df` / `collect` / `cypher` / … — as an
    /// independent single-owner handle. Each call returns its own handle, so N
    /// threads can each take a cursor off the same shared `Session` and run
    /// fluent chains in parallel, lock-free, with no single-owner borrow
    /// conflict.
    ///
    /// The cursor is bound to the snapshot at call time: it observes the graph
    /// as of now, and any mutation on the cursor is isolated via copy-on-write
    /// (it does not write back to the `Session`). To pick up later session
    /// writes, take a fresh `cursor()`.
    fn cursor(&self) -> crate::graph::KnowledgeGraph {
        let mut kg = crate::graph::KnowledgeGraph::from_arc(self.inner.snapshot());
        if let Some(e) = &self.embedder {
            kg.set_embedder_native(e.clone());
        }
        kg
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
