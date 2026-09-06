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
//!   whole mutation. Without an embedder, the core Session mutates its Arc in
//!   place when uniquely owned. Callback-capable writes use a transaction fork
//!   outside the core graph lock so callbacks can read committed state. Concurrent writes
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

use super::query_defaults::QueryDefaults;
use crate::datatypes::py_in;
use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::languages::cypher;
use crate::graph::pyapi::frozen::FrozenGraph;
use crate::graph::pyapi::result_view::ResultView;
use crate::graph::DirGraph;
use crate::util::EnterKg;
use kglite_core::api::session::{
    execute_mut, execute_read, CommitOutcome, CsvImportPolicy, ExecuteOptions,
    Session as CoreSession,
};
use kglite_core::api::GraphRead;

/// Thread-safe, shareable handle over a graph. See module docs.
///
/// Build or load a graph with a `KnowledgeGraph`, call `.session()`, then
/// share the `Session` across threads: concurrent `cypher()` reads run
/// lock-free; `execute()` writes serialise behind the Session's writer lock.
/// The source embedder binding is captured at Session creation.
#[pyclass(module = "kglite", frozen)]
pub struct Session {
    defaults: QueryDefaults,
    source_authority: Option<super::lifecycle::SourceAuthority>,
    pub(crate) inner: CoreSession,
    pub(crate) embedder: Option<Arc<dyn crate::graph::embedder::Embedder>>,
    /// Serialises writers. Held across the whole `begin → mutate → commit` so
    /// concurrent `execute()` calls compose (each sees prior commits) instead
    /// of racing into a lost update. Readers never touch it.
    pub(crate) write_lock: Mutex<()>,
}

thread_local! {
    static CALLBACK_WRITES: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

/// A callback may read committed state but cannot wait on its own writer lock.
struct CallbackWriteGuard(usize);

impl CallbackWriteGuard {
    fn enter(session: &Session) -> PyResult<Self> {
        let key = std::ptr::from_ref(session) as usize;
        if CALLBACK_WRITES.with(|active| active.borrow_mut().insert(key)) {
            Ok(Self(key))
        } else {
            Err(crate::error_py::kg_to_pyerr(KgError::Argument(
                "An embedding callback cannot re-enter writes on the same Session; \
                 read its committed snapshot instead."
                    .to_string(),
            )))
        }
    }
}

impl Drop for CallbackWriteGuard {
    fn drop(&mut self) {
        CALLBACK_WRITES.with(|active| active.borrow_mut().remove(&self.0));
    }
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

/// Decoded per-call query options shared by the read and write paths.
struct QueryOpts {
    to_df: bool,
    deadline: Option<std::time::Instant>,
    max_work_units: Option<usize>,
    row_limit: Option<usize>,
    output_csv: bool,
    /// Role-scoped write whitelist; only consulted on the write path.
    write_scope: Option<std::collections::HashSet<String>>,
    git_sha: Option<String>,
    modified_by: Option<String>,
}

impl QueryOpts {
    // The Python boundary mirrors the public query-option surface.
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        defaults: QueryDefaults,
        to_df: bool,
        timeout_ms: Option<u64>,
        max_work_units: Option<usize>,
        row_limit: Option<usize>,
        csv: bool,
        write_scope: Option<std::collections::HashSet<String>>,
        git_sha: Option<String>,
        modified_by: Option<String>,
    ) -> Self {
        let effective = defaults.resolve(timeout_ms, max_work_units, row_limit);
        QueryOpts {
            to_df,
            deadline: effective.deadline,
            max_work_units: effective.max_work_units,
            row_limit: effective.row_limit,
            output_csv: csv,
            write_scope,
            git_sha,
            modified_by,
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
        Self::with_defaults(inner, embedder, QueryDefaults::default(), None)
    }

    pub(crate) fn with_defaults(
        inner: Arc<DirGraph>,
        embedder: Option<Arc<dyn crate::graph::embedder::Embedder>>,
        defaults: QueryDefaults,
        source_authority: Option<super::lifecycle::SourceAuthority>,
    ) -> Self {
        Session {
            defaults,
            source_authority,
            inner: CoreSession::from_arc(inner),
            embedder,
            write_lock: Mutex::new(()),
        }
    }

    fn read_snapshot(&self, py: Python<'_>) -> Arc<DirGraph> {
        // A writer or its Python callback may need the GIL before releasing
        // the core graph mutex, so no attached caller may block on it.
        py.detach(|| self.inner.snapshot())
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
        let inner = self.read_snapshot(py);
        let embedder = self.embedder.clone();
        let query_owned = query.to_string();
        let deadline = qopts.deadline;
        let max_work_units = qopts.max_work_units;
        let row_limit = qopts.row_limit;
        let result = py.enter_kg(move |cancel| -> Result<cypher::CypherResult, KgError> {
            let opts = ExecuteOptions {
                params: &param_map,
                deadline,
                max_work_units,
                row_limit,
                lazy_eligible: false,
                parallel: false,
                disabled_passes: None,
                embedder,
                value_codecs: None,
                cancel,
                write_scope: None,
                git_sha: None,
                modified_by: None,
                csv_import: CsvImportPolicy::LocalFilesystem,
            };
            let outcome = execute_read(&inner, &query_owned, &opts)?;
            Ok(outcome.result)
        })?;
        marshal_result(py, result, qopts.to_df, qopts.output_csv)
    }

    /// Serialize writes; callback-capable execution releases the core mutex
    /// while operating on a transaction fork. Ordinary writes retain the
    /// unique-owner path without a begin-created Arc clone.
    // The detached closure preserves the engine's structured KgError until PyErr conversion.
    #[allow(clippy::result_large_err)]
    fn run_write(
        &self,
        py: Python<'_>,
        query: &str,
        param_map: HashMap<String, Value>,
        qopts: QueryOpts,
    ) -> PyResult<Py<PyAny>> {
        let _callback_write = self
            .embedder
            .as_ref()
            .map(|_| CallbackWriteGuard::enter(self))
            .transpose()?;
        let core = &self.inner;
        let source_authority = self.source_authority.clone();
        let write_lock = &self.write_lock;
        let query_owned = query.to_string();
        let deadline = qopts.deadline;
        let max_work_units = qopts.max_work_units;
        let row_limit = qopts.row_limit;
        let write_scope = qopts.write_scope;
        let git_sha = qopts.git_sha;
        let modified_by = qopts.modified_by;
        let embedder = self.embedder.clone();
        let result = py.enter_kg(move |cancel| -> Result<cypher::CypherResult, KgError> {
            // Acquire the writer lock *with the GIL released* (we are
            // already inside py.detach). Locking before the detach
            // would deadlock: a waiting writer would hold the GIL while
            // blocking on the lock, and the lock-holder needs the GIL
            // back to return. Poison-recover — the graph swaps
            // atomically, so a prior writer's panic doesn't cascade.
            let _wguard = write_lock.lock().unwrap_or_else(|p| p.into_inner());
            let mut graph = core.write();
            // Independent Session data cannot publish into the source's CDC
            // stream. It also has no WAL handle to commit captured writes.
            if graph.cdc_enabled() {
                return Err(KgError::Argument(
                    "A Session cannot execute write queries while change-data capture (CDC) \
                     is enabled: its independent data shares the source graph's change stream. \
                     Run the mutation on the graph itself or in its transaction. \
                     Sessions remain available for reads."
                        .to_string(),
                ));
            }
            if graph.owns_wal_capture()
                && source_authority
                    .as_ref()
                    .is_some_and(|source| source.ended())
            {
                *graph = graph.detached_persistence_snapshot();
            }
            if graph.owns_wal_capture() {
                return Err(KgError::Argument(
                    "A Session cannot execute write queries against a graph opened with \
                     durable=True: session writes are not recorded in the write-ahead log, \
                     so they would be lost on a crash. Run the mutation on the graph itself \
                     (g.cypher(...)) or in a transaction (with g.begin() as tx: ...), both \
                     of which are logged. Sessions remain available for reads."
                        .to_string(),
                ));
            }
            let opts = ExecuteOptions {
                params: &param_map,
                deadline,
                max_work_units,
                row_limit,
                lazy_eligible: false,
                parallel: false,
                disabled_passes: None,
                embedder,
                value_codecs: None,
                cancel,
                write_scope: write_scope.as_ref(),
                git_sha: git_sha.as_deref(),
                modified_by: modified_by.as_deref(),
                csv_import: CsvImportPolicy::LocalFilesystem,
            };
            if opts.embedder.is_none() {
                return Ok(execute_mut(&mut graph, &query_owned, &opts)?.result);
            }
            // Python callbacks see the committed graph. Retain only the
            // writer lock while preparing/executing the isolated statement.
            drop(graph);
            let mut tx = core.begin();
            let base_version = tx.base_version();
            let working = tx.working_mut()?;
            let result = execute_mut(working, &query_owned, &opts)?.result;
            if working.version() == base_version {
                return Ok(result);
            }
            match core.commit(tx, true) {
                CommitOutcome::Committed { .. } | CommitOutcome::NoWritesNoOp => Ok(result),
                CommitOutcome::ConflictDetected {
                    current_version,
                    base_version,
                } => Err(KgError::TransactionConflict {
                    current_version,
                    base_version,
                }),
                CommitOutcome::DurabilityFailed { error } => {
                    Err(KgError::FileIo(std::io::Error::other(error)))
                }
                other => Err(KgError::Internal {
                    message: format!("Unexpected Session commit outcome: {other:?}"),
                    location: "Session::run_write",
                }),
            }
        })?;
        drop(_callback_write);
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
    // Both Session paths land here, and before the output shape is chosen: a
    // CSV string and a DataFrame cannot carry diagnostics.
    crate::warning_policy::announce(py, result.diagnostics.as_ref())?;
    if output_csv {
        return result.to_csv().into_py_any(py);
    }
    if to_df {
        cypher::py_convert::rows_to_dataframe(py, &result.columns, &result.rows)
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
    #[pyo3(signature = (query, to_df=false, params=None, timeout_ms=None, max_work_units=None, row_limit=None))]
    // The Python boundary mirrors the public query-option surface.
    #[allow(clippy::too_many_arguments)]
    fn cypher(
        &self,
        py: Python<'_>,
        query: &str,
        to_df: bool,
        params: Option<&Bound<'_, PyDict>>,
        timeout_ms: Option<u64>,
        max_work_units: Option<usize>,
        row_limit: Option<usize>,
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
        let qopts = QueryOpts::from_parts(
            self.defaults,
            to_df,
            timeout_ms,
            max_work_units,
            row_limit,
            output_csv,
            None,
            None,
            None,
        );
        self.run_read(py, query, param_map, qopts)
    }

    /// Run a Cypher **write** against the shared graph, serialized.
    ///
    /// Mutations (`CREATE` / `MERGE` / `SET` / `REMOVE` / `DELETE` /
    /// `DETACH DELETE` / `FOREACH`, and schema DDL) take the Session's writer
    /// lock for the duration of the mutation, so concurrent `execute()` calls
    /// run one at a time and each sees the prior writer's committed changes —
    /// no lost updates. Readers already holding snapshots keep seeing the
    /// pre-write graph. A reader arriving during a unique-owner write may
    /// briefly wait for the core graph mutex.
    ///
    /// A read-only query passed to `execute()` is fast-pathed to the read
    /// path (no working-copy materialisation), so it is always safe to route
    /// mixed traffic through `execute()`.
    /// Callback-bearing writes use an isolated working copy: callbacks may
    /// read committed Session state, while same-Session writes are refused.
    ///
    /// Args:
    ///     query: A Cypher query string (read or write).
    ///     to_df: If True, return a pandas DataFrame.
    ///     params: Optional dict of query parameters.
    ///     timeout_ms: Per-call deadline in milliseconds.
    ///     max_work_units: Work budget for the query, not a result-row cap;
    ///         exceeding it is an error.
    ///     row_limit: Cap on the result rows kept. The query — writes
    ///         included — still runs in full; only retention stops at the
    ///         cap, and truncation warns and reports the exact pre-truncation
    ///         `total_rows` in `diagnostics`.
    ///     write_scope: Role-scoped write whitelist restricting this
    ///         statement's mutations — every node write is judged by the
    ///         node's *stored* type (a pattern label cannot widen it), and a
    ///         relationship write needs at least one endpoint's stored type in
    ///         the list. `None` (default) = unrestricted; `[]` denies every
    ///         mutation. See `KnowledgeGraph.cypher` for the exact perimeter.
    ///     git_sha, modified_by: Freshness provenance stamped alongside
    ///         `updated_at` on types that declare `auto_timestamp`.
    ///
    /// Returns the query result (rows for `... RETURN`, otherwise mutation
    /// stats), same shape as `KnowledgeGraph.cypher`.
    #[pyo3(signature = (query, to_df=false, params=None, timeout_ms=None, max_work_units=None, row_limit=None, write_scope=None, git_sha=None, modified_by=None))]
    // Python boundary mirrors the public query option surface.
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        py: Python<'_>,
        query: &str,
        to_df: bool,
        params: Option<&Bound<'_, PyDict>>,
        timeout_ms: Option<u64>,
        max_work_units: Option<usize>,
        row_limit: Option<usize>,
        write_scope: Option<Vec<String>>,
        git_sha: Option<String>,
        modified_by: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let pre_parsed = cypher::parse_cypher(query).map_err(crate::error_py::kg_to_pyerr)?;
        let param_map = decode_params(params)?;
        let output_csv = pre_parsed.output_format == cypher::OutputFormat::Csv;
        let scope_set = write_scope.map(|v| v.into_iter().collect());
        let qopts = QueryOpts::from_parts(
            self.defaults,
            to_df,
            timeout_ms,
            max_work_units,
            row_limit,
            output_csv,
            scope_set,
            git_sha,
            modified_by,
        );
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
    fn snapshot(&self, py: Python<'_>) -> FrozenGraph {
        FrozenGraph::with_defaults(self.read_snapshot(py), self.embedder.clone(), self.defaults)
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
    fn cursor(&self, py: Python<'_>) -> crate::graph::KnowledgeGraph {
        let mut kg = crate::graph::KnowledgeGraph::from_arc(self.read_snapshot(py));
        self.defaults.apply_to(&mut kg);
        kg.lifecycle.orphaned_from_cdc = kg.inner.cdc_enabled();
        if kg.inner.owns_wal_capture() {
            if self
                .source_authority
                .as_ref()
                .is_some_and(|source| source.ended())
            {
                kg.inner = Arc::new(kg.inner.detached_persistence_snapshot());
            } else {
                kg.lifecycle.orphaned_from_durable = true;
            }
        }
        if let Some(e) = &self.embedder {
            kg.set_embedder_native(e.clone());
        }
        kg
    }

    /// Monotonic version of the current graph. Bumped by each committed
    /// write. Useful for cheap "did anything change?" checks.
    fn version(&self, py: Python<'_>) -> u64 {
        self.read_snapshot(py).version()
    }

    /// Number of nodes in the current snapshot.
    fn node_count(&self, py: Python<'_>) -> usize {
        self.read_snapshot(py).graph.node_count()
    }

    /// Node type names present in the current snapshot.
    #[getter]
    fn node_types(&self, py: Python<'_>) -> Vec<String> {
        self.read_snapshot(py).get_node_types()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let snap = self.read_snapshot(py);
        format!(
            "Session(nodes={}, types={}, version={})",
            snap.graph.node_count(),
            snap.get_node_types().len(),
            snap.version(),
        )
    }
}
