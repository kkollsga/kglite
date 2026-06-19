//! KnowledgeGraph #[pymethods]: caches + schema + cypher + transactions.
//!
//! Part of the Phase 9 split of the kg_methods.rs monolith (5,419 lines
//! single pymethods block). PyO3 merges multiple `#[pymethods] impl`
//! blocks at class-registration time, so the split is purely structural —
//! no runtime impact.

use crate::datatypes::{py_in, py_out};
use crate::graph::introspection::{self, reporting::OperationReport};
use crate::graph::io;
use crate::graph::io::ntriples::{Cancelled, ProgressEvent, ProgressSink, ProgressValue};
use crate::graph::languages::cypher;
use crate::graph::pyapi::transaction::Transaction;
use crate::graph::schema::{
    self, ConnectionSchemaDefinition, NodeSchemaDefinition, SchemaDefinition,
};
use crate::graph::{get_graph_mut, resolve_noderefs, KnowledgeGraph};
use kglite_core::api::GraphRead;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::{Bound, IntoPyObjectExt};
use std::collections::HashMap;
use std::sync::Arc;

/// Adapter: routes pure-Rust [`ProgressEvent`]s into a Python callable.
/// Errors raised by the callback are swallowed — a broken UI must not
/// kill a multi-hour build. Lives in the pyapi layer so the loader
/// itself never touches PyO3 types.
struct PyProgressSink {
    callback: Py<PyAny>,
}

impl ProgressSink for PyProgressSink {
    fn emit(&self, event: ProgressEvent<'_>) -> Result<(), Cancelled> {
        Python::attach(|py| {
            // Briefly reacquiring the GIL inside `py.detach()` lets
            // Python observe a pending SIGINT (Ctrl+C). check_signals
            // returns Err if the user pressed Ctrl+C; we surface that
            // up as `Cancelled` so the loader can stop cleanly.
            if py.check_signals().is_err() {
                return Err(Cancelled);
            }
            let d = PyDict::new(py);
            let _ = match event {
                ProgressEvent::Start {
                    phase,
                    label,
                    total,
                    unit,
                } => {
                    let _ = d.set_item("kind", "start");
                    let _ = d.set_item("phase", phase);
                    let _ = d.set_item("label", label);
                    let _ = d.set_item("unit", unit);
                    match total {
                        Some(t) => d.set_item("total", t),
                        None => d.set_item("total", py.None()),
                    }
                }
                ProgressEvent::Update {
                    phase,
                    current,
                    fields,
                } => {
                    let _ = d.set_item("kind", "update");
                    let _ = d.set_item("phase", phase);
                    let _ = d.set_item("current", current);
                    set_fields(&d, fields)
                }
                ProgressEvent::Complete {
                    phase,
                    elapsed_s,
                    fields,
                } => {
                    let _ = d.set_item("kind", "complete");
                    let _ = d.set_item("phase", phase);
                    let _ = d.set_item("elapsed_s", elapsed_s);
                    set_fields(&d, fields)
                }
            };
            let _ = self.callback.call1(py, (d,));
            Ok(())
        })
    }
}

/// Format an integer with comma thousands separators ("1234567" → "1,234,567").
/// Used by `KnowledgeGraph::__repr__` — keeps large-graph summaries legible
/// without pulling a dep just for `num-format`.
/// Clear, actionable error for a cross-thread borrow conflict on a shared
/// `KnowledgeGraph`. PyO3's `#[pyclass]` is `RefCell`-guarded: while one
/// thread mutates the graph (`add_nodes` / `embed_texts` / a `CREATE`
/// query / `save`) it holds the exclusive borrow, and a second thread
/// touching the *same* instance hits the guard. The raw symptom is either
/// a cryptic `RuntimeError: Already borrowed` or — at the hand-written
/// `slf.borrow()` sites — a panic. We turn both into this one message.
///
/// Kept in the Python wrapper (not core `KgError`): the borrow guard is a
/// PyO3/GIL concern, so a non-Python binding never raises it. Raised as a
/// `RuntimeError` so existing `except RuntimeError` handlers still catch
/// it (`PyBorrowError` is itself a `RuntimeError`).
fn concurrent_access_pyerr() -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "KnowledgeGraph accessed concurrently from another thread while it was being \
         used elsewhere. A KnowledgeGraph is single-owner: it is not safe to share one \
         instance across threads while any thread mutates it (add_nodes, add_connections, \
         embed_texts, a CREATE/SET/DELETE query, or save). For concurrent access, reach \
         for a Session instead of sharing the KnowledgeGraph: `g.session()` returns a \
         thread-safe handle whose `cypher()` reads run lock-free and `execute()` writes \
         serialize; `g.freeze()` gives a lock-free read-only snapshot; \
         `g.session().cursor()` hands each thread its own fluent handle. (Or give each \
         worker its own `copy()`.) See docs/concepts/concurrency.md.",
    )
}

fn fmt_with_commas(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

fn set_fields(d: &Bound<'_, PyDict>, fields: &[(&str, ProgressValue<'_>)]) -> PyResult<()> {
    for (k, v) in fields {
        match v {
            ProgressValue::U64(n) => d.set_item(*k, *n)?,
            ProgressValue::F64(n) => d.set_item(*k, *n)?,
            ProgressValue::Str(s) => d.set_item(*k, *s)?,
        }
    }
    Ok(())
}

#[pymethods]
impl KnowledgeGraph {
    // ================================================================
    // Schema Introspection
    // ================================================================

    /// Force recomputation of internal caches (edge type counts,
    /// type connectivity, connection endpoint types).
    ///
    /// Performs a single O(E) pass to compute type connectivity triples,
    /// then derives edge type counts and connection endpoint types from
    /// the triples (no additional edge scans).
    ///
    /// Call once after bulk mutations to warm the cache before ``save()``
    /// or ``describe()``. The caches are persisted by ``save()`` and
    /// restored by ``load()``, so this only needs to be called once
    /// after building or mutating a graph.
    fn rebuild_caches(&mut self) {
        // Order matters on large disk graphs: `compute_type_connectivity`
        // runs first so its single sequential sweep of `edge_endpoints.bin`
        // also warms the page cache for the histogram builder below. On
        // Wikidata (~13.8 GB edge_endpoints) this avoids a second cold
        // read and cuts `rebuild_caches` by ~100 s on loaded-from-cold-disk
        // graphs. See `src/graph/storage/disk/builder.rs::build_peer_count_histogram`
        // — it deliberately no longer evicts pages after finishing.

        // Single O(E) pass: compute type connectivity triples
        // Uses edge_endpoint_keys() — mmap reads only, no heap allocation per edge.
        let triples = introspection::compute_type_connectivity(&self.inner);

        // On disk graphs, also rebuild the per-(conn_type, peer) edge-count
        // histogram so the Cypher planner's fast path for unanchored
        // aggregate queries works. Legacy disk graphs built before v0.7.13
        // won't have the histogram files — this call creates them.
        {
            let graph = get_graph_mut(&mut self.inner);
            if let schema::GraphBackend::Disk(ref mut dg) = graph.graph {
                dg.rebuild_peer_count_histogram();
            }
        }

        // Derive edge type counts + endpoint types from triples (no extra scan)
        let derived = introspection::derive_edge_counts_from_triples(&triples);

        // Populate edge type counts cache
        *self.inner.edge_type_counts_cache.write().unwrap() = Some(derived.counts);

        // Backfill connection_type_metadata with discovered endpoint types
        let graph = get_graph_mut(&mut self.inner);
        for (conn_type, (src_types, tgt_types)) in derived.endpoints {
            let info = graph.connection_type_metadata.entry(conn_type).or_default();
            if info.source_types.is_empty() {
                info.source_types = src_types;
            }
            if info.target_types.is_empty() {
                info.target_types = tgt_types;
            }
        }

        // Store type connectivity triples
        self.inner.set_type_connectivity(triples);
    }

    /// Total counts of nodes and edges in the graph, pandas-style.
    ///
    /// Returns ``(node_count, edge_count)``. O(1) via the storage
    /// backend — does not materialise per-type breakdowns. For full
    /// per-type structure use ``schema()`` or ``describe()``.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        use kglite_core::api::GraphRead;
        (self.inner.graph.node_count(), self.inner.graph.edge_count())
    }

    /// Human-readable summary used by ``print(graph)`` and the REPL.
    /// Format: ``KnowledgeGraph(N nodes, M edges)`` with thousands
    /// separators for legibility on large graphs.
    fn __repr__(&self) -> String {
        use kglite_core::api::GraphRead;
        format!(
            "KnowledgeGraph({} nodes, {} edges)",
            fmt_with_commas(self.inner.graph.node_count()),
            fmt_with_commas(self.inner.graph.edge_count()),
        )
    }

    /// Return a full schema overview of the graph.
    fn schema(&self) -> PyResult<Py<PyAny>> {
        let overview = introspection::compute_schema(&self.inner);
        Python::attach(|py| {
            let result = PyDict::new(py);

            // node_types
            let node_types_dict = PyDict::new(py);
            for (nt, info) in &overview.node_types {
                let type_dict = PyDict::new(py);
                type_dict.set_item("count", info.count)?;
                let props_dict = PyDict::new(py);
                for (k, v) in &info.properties {
                    props_dict.set_item(k.as_str(), v.as_str())?;
                }
                type_dict.set_item("properties", props_dict)?;
                node_types_dict.set_item(nt.as_str(), type_dict)?;
            }
            result.set_item("node_types", node_types_dict)?;

            // connection_types
            let conn_dict = PyDict::new(py);
            for ct in &overview.connection_types {
                let ct_dict = PyDict::new(py);
                ct_dict.set_item("count", ct.count)?;
                ct_dict.set_item("source_types", &ct.source_types)?;
                ct_dict.set_item("target_types", &ct.target_types)?;
                conn_dict.set_item(ct.connection_type.as_str(), ct_dict)?;
            }
            result.set_item("connection_types", conn_dict)?;

            result.set_item("indexes", &overview.indexes)?;
            result.set_item("node_count", overview.node_count)?;
            result.set_item("edge_count", overview.edge_count)?;

            Ok(result.into())
        })
    }

    /// Return all connection types with counts and endpoint type sets.
    #[pyo3(name = "connection_types")]
    fn connection_types_info(&self) -> PyResult<Py<PyAny>> {
        let stats = introspection::compute_connection_type_stats(&self.inner);
        Python::attach(|py| {
            let result_list = PyList::empty(py);
            for ct in &stats {
                let ct_dict = PyDict::new(py);
                ct_dict.set_item("type", ct.connection_type.as_str())?;
                ct_dict.set_item("count", ct.count)?;
                ct_dict.set_item("source_types", &ct.source_types)?;
                ct_dict.set_item("target_types", &ct.target_types)?;
                result_list.append(ct_dict)?;
            }
            Ok(result_list.into())
        })
    }

    /// Return property statistics for a node type.
    #[pyo3(signature = (node_type, max_values=20))]
    fn properties(&self, node_type: &str, max_values: usize) -> PyResult<Py<PyAny>> {
        // Sample large types for faster response; exact stats for small types
        let count = self
            .inner
            .type_indices
            .get(node_type)
            .map(|v| v.len())
            .unwrap_or(0);
        let sample = if count > 1000 { Some(500) } else { None };
        let stats =
            introspection::compute_property_stats(&self.inner, node_type, max_values, sample)
                .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        Python::attach(|py| {
            let result = PyDict::new(py);
            for prop in &stats {
                let prop_dict = PyDict::new(py);
                prop_dict.set_item("type", prop.type_string.as_str())?;
                prop_dict.set_item("non_null", prop.non_null)?;
                prop_dict.set_item("unique", prop.unique)?;
                if let Some(ref vals) = prop.values {
                    let py_vals = PyList::empty(py);
                    for v in vals {
                        py_vals.append(py_out::value_to_py(py, v)?)?;
                    }
                    prop_dict.set_item("values", py_vals)?;
                }
                result.set_item(prop.property_name.as_str(), prop_dict)?;
            }
            Ok(result.into())
        })
    }

    /// Return connection topology for a node type (outgoing and incoming).
    fn neighbors_schema(&self, node_type: &str) -> PyResult<Py<PyAny>> {
        let ns = introspection::compute_neighbors_schema(&self.inner, node_type)
            .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        Python::attach(|py| {
            let result = PyDict::new(py);

            let out_list = PyList::empty(py);
            for nc in &ns.outgoing {
                let d = PyDict::new(py);
                d.set_item("connection_type", nc.connection_type.as_str())?;
                d.set_item("target_type", nc.other_type.as_str())?;
                d.set_item("count", nc.count)?;
                out_list.append(d)?;
            }
            result.set_item("outgoing", out_list)?;

            let in_list = PyList::empty(py);
            for nc in &ns.incoming {
                let d = PyDict::new(py);
                d.set_item("connection_type", nc.connection_type.as_str())?;
                d.set_item("source_type", nc.other_type.as_str())?;
                d.set_item("count", nc.count)?;
                in_list.append(d)?;
            }
            result.set_item("incoming", in_list)?;

            Ok(result.into())
        })
    }

    /// Return a quick sample of nodes.
    ///
    /// Can be called as:
    ///   - ``sample("Person")`` — sample 5 nodes of the given type
    ///   - ``sample("Person", 10)`` — sample 10 nodes of the given type
    ///   - ``sample(3)`` — sample 3 nodes from the current selection
    ///   - ``sample()`` — sample 5 nodes from the current selection
    #[pyo3(signature = (node_type_or_n=None, n=None))]
    fn sample(
        &self,
        node_type_or_n: Option<&Bound<'_, PyAny>>,
        n: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let default_n = 5usize;

        // Parse first arg: could be str (node_type) or int (n)
        let (node_type, count) = match node_type_or_n {
            Some(arg) => {
                if let Ok(s) = arg.extract::<String>() {
                    (Some(s), n.unwrap_or(default_n))
                } else if let Ok(i) = arg.extract::<usize>() {
                    (None, i)
                } else {
                    return Err(crate::error_py::kg_to_pyerr(
                        crate::error::KgError::Argument(
                            "sample() first argument must be a node type (str) or count (int)"
                                .to_string(),
                        ),
                    ));
                }
            }
            None => (None, n.unwrap_or(default_n)),
        };

        if let Some(nt) = node_type {
            let type_indices = self.inner.type_indices.get(&nt).ok_or_else(|| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                    (format!("Node type '{}' not found", nt)).to_string(),
                ))
            })?;
            let indices: Vec<_> = type_indices.iter().take(count).collect();
            let view = crate::graph::pyapi::result_view::ResultView::from_nodes_with_graph(
                &self.inner,
                &indices,
            );
            return Python::attach(|py| Py::new(py, view).map(|v| v.into_any()));
        }

        // Selection-based: sample from current selection
        let level_count = self.cursor.selection.get_level_count();
        if level_count == 0 {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "sample() requires either a selection or a node_type argument".to_string(),
                ),
            ));
        }
        let last = level_count - 1;
        let level = self
            .cursor
            .selection
            .get_level(last)
            .ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                    "Empty selection".to_string(),
                ))
            })?;
        let all_indices = level.get_all_nodes();
        let indices: Vec<_> = all_indices.into_iter().take(count).collect();
        let view = crate::graph::pyapi::result_view::ResultView::from_nodes_with_graph(
            &self.inner,
            &indices,
        );
        Python::attach(|py| Py::new(py, view).map(|v| v.into_any()))
    }

    /// Return a unified list of all indexes (single-property and composite).
    fn indexes(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let result_list = PyList::empty(py);

            for (node_type, property) in self.inner.property_indices.keys() {
                let d = PyDict::new(py);
                d.set_item("node_type", node_type.as_str())?;
                d.set_item("property", property.as_str())?;
                d.set_item("type", "equality")?;
                result_list.append(d)?;
            }

            for (node_type, properties) in self.inner.composite_indices.keys() {
                let d = PyDict::new(py);
                d.set_item("node_type", node_type.as_str())?;
                d.set_item("properties", properties)?;
                d.set_item("type", "composite")?;
                result_list.append(d)?;
            }

            Ok(result_list.into())
        })
    }

    fn clear(&mut self) -> PyResult<()> {
        self.cursor.selection.clear();
        Ok(())
    }

    /// Load an N-Triples file (supports .bz2, .gz, plain) into the graph.
    /// Designed for Wikidata truthy dumps but works with any N-Triples file.
    #[pyo3(signature = (path, *, predicates=None, languages=None, node_types=None, predicate_labels=None, max_entities=None, max_triples=None, verbose=false, progress=None))]
    #[allow(clippy::too_many_arguments)]
    fn load_ntriples(
        &mut self,
        py: Python<'_>,
        path: &str,
        predicates: Option<Vec<String>>,
        languages: Option<Vec<String>>,
        node_types: Option<&Bound<'_, PyDict>>,
        predicate_labels: Option<&Bound<'_, PyDict>>,
        max_entities: Option<usize>,
        max_triples: Option<u64>,
        verbose: bool,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        use std::collections::HashSet;

        if let Some(ref cb) = progress {
            if !cb.bind(py).is_callable() {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "progress must be callable",
                ));
            }
        }

        let progress_sink: Option<Box<dyn ProgressSink>> =
            progress.map(|cb| Box::new(PyProgressSink { callback: cb }) as Box<dyn ProgressSink>);

        let config = crate::graph::io::ntriples::NTriplesConfig {
            predicates: predicates.map(|v| v.into_iter().collect::<HashSet<_>>()),
            languages: languages.map(|v| v.into_iter().collect::<HashSet<_>>()),
            node_types: node_types
                .map(|d| {
                    d.iter()
                        .filter_map(|(k, v)| {
                            Some((k.extract::<String>().ok()?, v.extract::<String>().ok()?))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            predicate_labels: predicate_labels
                .map(|d| {
                    d.iter()
                        .filter_map(|(k, v)| {
                            Some((k.extract::<String>().ok()?, v.extract::<String>().ok()?))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            max_entities,
            max_triples,
            verbose,
            auto_type: true,
            progress: progress_sink,
        };

        let graph = Arc::make_mut(&mut self.inner);
        // Release the GIL during the multi-minute load so Python
        // heartbeat threads (download/build progress monitors) can run.
        // Cancellation: when the progress sink notices a pending SIGINT
        // (Python::check_signals), the loader returns the literal
        // `<cancelled>` sentinel — translate that into the KeyboardInterrupt
        // the user actually expects, instead of a generic RuntimeError.
        let stats = py
            .detach(|| crate::graph::io::ntriples::load_ntriples(graph, path, &config))
            .map_err(|e| {
                if e == "<cancelled>" {
                    PyErr::new::<pyo3::exceptions::PyKeyboardInterrupt, _>(
                        "load_ntriples cancelled",
                    )
                } else {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)
                }
            })?;

        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("entities", stats.entities_created)?;
            dict.set_item("edges", stats.edges_created)?;
            dict.set_item("edges_skipped", stats.edges_skipped)?;
            dict.set_item("triples_scanned", stats.triples_scanned)?;
            dict.set_item("seconds", stats.seconds)?;
            Ok(dict.into())
        })
    }

    #[pyo3(signature = (path=None, *, fsync=true))]
    fn save(&mut self, py: Python<'_>, path: Option<&str>, fsync: bool) -> PyResult<()> {
        // Resolve the target: explicit path wins; otherwise fall back to the
        // origin file this graph was opened from (`kglite.open`/`load`). A
        // graph built in memory with no origin and no explicit path has
        // nowhere to go — guide the user rather than panicking.
        let effective: String = match path {
            Some(p) => p.to_string(),
            None => self
                .lifecycle
                .source_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "save() needs a path: this graph was not opened from a file. \
                         Pass an explicit path (g.save('graph.kgl')) or open it with \
                         kglite.open('graph.kgl') so save() remembers the target.",
                    )
                })?,
        };
        // Remember the target so a later bare save() / auto-save-on-close
        // writes back to the same file ("save as" updates the home path).
        self.lifecycle.source_path = Some(std::path::PathBuf::from(&effective));
        let path: &str = &effective;

        // Disk mode: save as directory (the folder IS the graph)
        if self.inner.graph.is_disk() {
            let graph = Arc::make_mut(&mut self.inner);
            return graph
                .save_disk(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()));
        }

        // Prep phase (quick): stamp metadata, snapshot index keys
        io::file::prepare_save(&mut self.inner);

        // Consolidate ALL node properties into column stores (v3 requires columnar).
        // Always rebuild: after load+add, some nodes may have Compact storage;
        // after load+update, COW clones diverge from graph.column_stores.
        // enable_columnar() handles all cases (fresh, mixed, and mapped mode).
        {
            let graph = Arc::make_mut(&mut self.inner);
            graph.enable_columnar();
        }

        // Heavy phase: serialize, compress, write — release GIL for other Python threads.
        // The write is atomic (temp + rename) and, with fsync=True (default),
        // durable (file + directory fsync) — a crash mid-save can't tear the .kgl.
        let inner = self.inner.clone();
        let path_owned = path.to_string();
        py.detach(move || io::file::write_kgl_with(&inner, &path_owned, fsync))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        // Durable checkpoint: the .kgl now holds the full current state, so
        // discard the capture buffer (those ops are folded in) and truncate
        // the WAL. Order matters — the .kgl write above succeeded before we
        // truncate, and replay is idempotent, so a crash between the two
        // only costs a harmless re-apply on the next open.
        if self.lifecycle.durable.is_some() {
            if let kglite_core::graph::schema::GraphBackend::Recording(rg) =
                &mut Arc::make_mut(&mut self.inner).graph
            {
                let _ = rg.take_ops();
            }
            if let Some(ds) = self.lifecycle.durable.as_mut() {
                ds.wal
                    .reset()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                ds.next_lsn = 1;
            }
        }
        Ok(())
    }

    /// Serialize the in-memory graph to a `.kgl` byte buffer (the same
    /// bytes `save()` writes to disk) and return them as `bytes`.
    ///
    /// Lets a caller own the write — hand the bytes to object storage, a
    /// pipe, a checksum, or a custom atomic-write routine — instead of
    /// being limited to a filesystem path. Round-trips through
    /// `kglite.from_bytes(data)`.
    ///
    /// In-memory / mapped graphs only: a `disk`-mode graph is a directory,
    /// not a single byte stream, so this raises for disk graphs (use
    /// `save(dir)` there).
    // `&mut self` (not the `to_*`-convention `&self`): like `save()`, this
    // consolidates node properties into column stores before serializing,
    // which mutates the graph in place (it stays columnar afterwards).
    #[allow(clippy::wrong_self_convention)]
    fn to_bytes(&mut self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        if self.inner.graph.is_disk() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "to_bytes() is not supported for disk-mode graphs (a disk graph is a \
                 directory, not a single byte stream). Use save('dir/') instead.",
            ));
        }
        // Same prep as save(): stamp metadata + consolidate to columnar.
        io::file::prepare_save(&mut self.inner);
        {
            let graph = Arc::make_mut(&mut self.inner);
            graph.enable_columnar();
        }
        // Serialize off the GIL into an owned buffer.
        let inner = self.inner.clone();
        let bytes = py
            .detach(move || -> std::io::Result<Vec<u8>> {
                let mut buf: Vec<u8> = Vec::new();
                io::file::write_kgl_to(&inner, &mut buf)?;
                Ok(buf)
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    /// Take an immutable, concurrently-readable snapshot of the graph.
    ///
    /// Returns a `FrozenGraph` that shares this graph's data (an O(1)
    /// clone — no deep copy) and exposes only read methods. Unlike a live
    /// `KnowledgeGraph` — which is single-owner and raises if a second
    /// thread touches it mid-mutation — a `FrozenGraph` has no mutating
    /// method, so any number of threads can run `cypher()` against the same
    /// snapshot in parallel, lock-free.
    ///
    /// The snapshot is stable: mutating the source graph afterwards
    /// copy-on-writes a fresh copy, leaving the frozen view on the original
    /// data. This is the "build → freeze → share → swap" model — build a
    /// graph cheaply, freeze it, serve concurrent readers, and swap in a new
    /// `freeze()` when the data changes.
    fn freeze(&self) -> crate::graph::pyapi::frozen::FrozenGraph {
        crate::graph::pyapi::frozen::FrozenGraph::new(
            std::sync::Arc::clone(&self.inner),
            self.embedder.clone(),
        )
    }

    /// Seed a thread-safe, shareable `Session` from this graph's current
    /// state.
    ///
    /// Unlike a live `KnowledgeGraph` — which is single-owner and trips a
    /// borrow guard when shared across threads mid-mutation — a `Session`
    /// has only `&self` methods, with synchronisation in an internal lock.
    /// Concurrent `cypher()` reads run lock-free; `execute()` writes
    /// serialise behind the lock with copy-on-write + atomic swap.
    ///
    /// The `Session` is an **independent owner**: it shares this graph's
    /// `Arc<DirGraph>` at creation (O(1), no copy), but once either side
    /// mutates, copy-on-write forks them. The intended model is "build /
    /// load with a `KnowledgeGraph`, then `.session()` and serve every thread
    /// through the `Session`" — don't keep mutating the original graph after.
    fn session(&self) -> crate::graph::pyapi::session::Session {
        crate::graph::pyapi::session::Session::from_arc(
            std::sync::Arc::clone(&self.inner),
            self.embedder.clone(),
        )
    }

    /// Persist the graph to its remembered origin path and release nothing
    /// else (the graph stays usable). No-op if the graph has no associated
    /// path (built in memory, never opened/saved to a file) — there is
    /// nowhere to write, and silently doing nothing is friendlier than
    /// raising on a best-effort cleanup call. Pair with `save(path)` if you
    /// need an explicit target.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.lifecycle.source_path.is_some() {
            self.save(py, None, true)?;
        }
        Ok(())
    }

    /// Context-manager entry: `with kglite.open(path) as g:` binds the graph
    /// itself, so the body queries/mutates `g` directly.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context-manager exit. On a **clean** exit (no exception) of a graph
    /// that remembers an origin path, snapshot to that path — the
    /// auto-save-on-close that gives `open()` its embedded-database feel. On
    /// an exception, the save is skipped so the on-disk file keeps its last
    /// good state. Never suppresses the exception (returns `False`).
    ///
    /// This is a *clean-exit* checkpoint, not crash safety: a hard crash
    /// (`kill -9`, power loss) mid-block writes nothing. Durable-on-commit is
    /// a separate capability.
    #[pyo3(signature = (exc_type, _exc_value, _traceback))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: &Bound<'_, pyo3::PyAny>,
        _exc_value: &Bound<'_, pyo3::PyAny>,
        _traceback: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<bool> {
        if exc_type.is_none() && self.lifecycle.source_path.is_some() {
            self.save(py, None, true)?;
        }
        Ok(false)
    }

    /// Compact a disk-mode graph: merge overflow edges back into CSR arrays.
    /// Returns the number of overflow edges that were merged.
    /// Overflow edges accumulate when edges are added after the initial CSR build
    /// (e.g., after loading a graph and adding new connections).
    /// Compaction rebuilds the CSR to include all overflow edges, restoring
    /// optimal query performance.
    /// No-op if there are no overflow edges or the graph is not in disk mode.
    fn compact(&mut self) -> PyResult<usize> {
        if !self.inner.graph.is_disk() {
            return Ok(0);
        }
        let graph = Arc::make_mut(&mut self.inner);
        graph
            .compact_disk()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Set a default query timeout (milliseconds) applied to all cypher() calls.
    ///
    /// - `None` (default): fall through to the built-in default (180_000 ms / 3 min).
    /// - `0`: disable the deadline for every query unless a per-call
    ///   `timeout_ms` overrides.
    /// - Any positive value: use it as the default.
    ///
    /// Per-query `timeout_ms` always overrides this setting.
    #[pyo3(signature = (timeout_ms=None))]
    fn set_default_timeout(&mut self, timeout_ms: Option<u64>) {
        self.default_timeout_ms = timeout_ms;
    }

    /// Get the current default query timeout in milliseconds, or None.
    fn get_default_timeout(&self) -> Option<u64> {
        self.default_timeout_ms
    }

    /// Set a default max rows limit applied to all cypher() calls.
    /// Queries producing more intermediate rows than this will error.
    /// Pass None to disable (default). Per-query max_rows overrides this.
    #[pyo3(signature = (max_rows=None))]
    fn set_default_max_rows(&mut self, max_rows: Option<usize>) {
        self.default_max_rows = max_rows;
    }

    /// Get the current default max rows limit, or None.
    fn get_default_max_rows(&self) -> Option<usize> {
        self.default_max_rows
    }

    /// Get the most recent operation report as a Python dictionary
    fn last_report(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            if let Some(report) = self.cursor.reports.get_last_report() {
                match report {
                    OperationReport::NodeOperation(node_report) => {
                        let report_dict = PyDict::new(py);
                        report_dict.set_item("operation", &node_report.operation_type)?;
                        report_dict.set_item("timestamp", node_report.timestamp.to_rfc3339())?;
                        report_dict.set_item("nodes_created", node_report.nodes_created)?;
                        report_dict.set_item("nodes_updated", node_report.nodes_updated)?;
                        report_dict.set_item("nodes_skipped", node_report.nodes_skipped)?;
                        report_dict
                            .set_item("processing_time_ms", node_report.processing_time_ms)?;

                        // Add errors array if there are any
                        if !node_report.errors.is_empty() {
                            report_dict.set_item("errors", &node_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }

                        Ok(report_dict.into())
                    }
                    OperationReport::ConnectionOperation(conn_report) => {
                        let report_dict = PyDict::new(py);
                        report_dict.set_item("operation", &conn_report.operation_type)?;
                        report_dict.set_item("timestamp", conn_report.timestamp.to_rfc3339())?;
                        report_dict
                            .set_item("connections_created", conn_report.connections_created)?;
                        report_dict
                            .set_item("connections_skipped", conn_report.connections_skipped)?;
                        report_dict.set_item(
                            "property_fields_tracked",
                            conn_report.property_fields_tracked,
                        )?;
                        report_dict
                            .set_item("processing_time_ms", conn_report.processing_time_ms)?;

                        // Add errors array if there are any
                        if !conn_report.errors.is_empty() {
                            report_dict.set_item("errors", &conn_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }

                        Ok(report_dict.into())
                    }
                    OperationReport::CalculationOperation(calc_report) => {
                        let report_dict = PyDict::new(py);
                        report_dict.set_item("operation", &calc_report.operation_type)?;
                        report_dict.set_item("timestamp", calc_report.timestamp.to_rfc3339())?;
                        report_dict.set_item("expression", &calc_report.expression)?;
                        report_dict.set_item("nodes_processed", calc_report.nodes_processed)?;
                        report_dict.set_item("nodes_updated", calc_report.nodes_updated)?;
                        report_dict.set_item("nodes_with_errors", calc_report.nodes_with_errors)?;
                        report_dict
                            .set_item("processing_time_ms", calc_report.processing_time_ms)?;
                        report_dict.set_item("is_aggregation", calc_report.is_aggregation)?;

                        // Add errors array if there are any
                        if !calc_report.errors.is_empty() {
                            report_dict.set_item("errors", &calc_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }

                        Ok(report_dict.into())
                    }
                }
            } else {
                let empty_dict = PyDict::new(py);
                Ok(empty_dict.into())
            }
        })
    }

    /// Get the last operation index (a sequential ID of operations performed)
    fn operation_index(&self) -> usize {
        self.cursor.reports.get_last_operation_index()
    }

    /// Get all report history as a list of dictionaries
    fn report_history(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            // Create an empty list with PyList::empty
            let report_list = PyList::empty(py);

            for report in self.cursor.reports.get_all_reports() {
                let report_dict = match report {
                    OperationReport::NodeOperation(node_report) => {
                        let dict = PyDict::new(py);
                        dict.set_item("operation", &node_report.operation_type)?;
                        dict.set_item("timestamp", node_report.timestamp.to_rfc3339())?;
                        dict.set_item("nodes_created", node_report.nodes_created)?;
                        dict.set_item("nodes_updated", node_report.nodes_updated)?;
                        dict.set_item("nodes_skipped", node_report.nodes_skipped)?;
                        dict.set_item("processing_time_ms", node_report.processing_time_ms)?;

                        // Add errors array if there are any
                        if !node_report.errors.is_empty() {
                            dict.set_item("errors", &node_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }

                        dict
                    }
                    OperationReport::ConnectionOperation(conn_report) => {
                        let dict = PyDict::new(py);
                        dict.set_item("operation", &conn_report.operation_type)?;
                        dict.set_item("timestamp", conn_report.timestamp.to_rfc3339())?;
                        dict.set_item("connections_created", conn_report.connections_created)?;
                        dict.set_item("connections_skipped", conn_report.connections_skipped)?;
                        dict.set_item(
                            "property_fields_tracked",
                            conn_report.property_fields_tracked,
                        )?;
                        dict.set_item("processing_time_ms", conn_report.processing_time_ms)?;

                        // Add errors array if there are any
                        if !conn_report.errors.is_empty() {
                            dict.set_item("errors", &conn_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }

                        dict
                    }
                    OperationReport::CalculationOperation(calc_report) => {
                        let dict = PyDict::new(py);
                        dict.set_item("operation", &calc_report.operation_type)?;
                        dict.set_item("timestamp", calc_report.timestamp.to_rfc3339())?;
                        dict.set_item("expression", &calc_report.expression)?;
                        dict.set_item("nodes_processed", calc_report.nodes_processed)?;
                        dict.set_item("nodes_updated", calc_report.nodes_updated)?;
                        dict.set_item("nodes_with_errors", calc_report.nodes_with_errors)?;
                        dict.set_item("processing_time_ms", calc_report.processing_time_ms)?;
                        dict.set_item("is_aggregation", calc_report.is_aggregation)?;

                        // Add errors array if there are any
                        if !calc_report.errors.is_empty() {
                            dict.set_item("errors", &calc_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }

                        dict
                    }
                };
                report_list.append(report_dict)?;
            }
            Ok(report_list.into())
        })
    }

    /// Perform union of two selections - combines all nodes from both selections
    /// Returns a new KnowledgeGraph with the union of both selections
    fn union(&self, other: &Self) -> PyResult<Self> {
        self.derive_with(|_inner, cursor| {
            crate::graph::mutation::set_ops::union_selections(
                &mut cursor.selection,
                &other.cursor.selection,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
        })
    }

    /// Perform intersection of two selections - keeps only nodes present in both
    /// Returns a new KnowledgeGraph with only nodes that exist in both selections
    fn intersection(&self, other: &Self) -> PyResult<Self> {
        self.derive_with(|_inner, cursor| {
            crate::graph::mutation::set_ops::intersection_selections(
                &mut cursor.selection,
                &other.cursor.selection,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
        })
    }

    /// Perform difference of two selections - keeps nodes in self but not in other
    /// Returns a new KnowledgeGraph with nodes from self that are not in other
    fn difference(&self, other: &Self) -> PyResult<Self> {
        self.derive_with(|_inner, cursor| {
            crate::graph::mutation::set_ops::difference_selections(
                &mut cursor.selection,
                &other.cursor.selection,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
        })
    }

    /// Perform symmetric difference of two selections - keeps nodes in either but not both
    /// Returns a new KnowledgeGraph with nodes that are in exactly one of the selections
    fn symmetric_difference(&self, other: &Self) -> PyResult<Self> {
        self.derive_with(|_inner, cursor| {
            crate::graph::mutation::set_ops::symmetric_difference_selections(
                &mut cursor.selection,
                &other.cursor.selection,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
        })
    }

    // ========================================================================
    // Schema Definition & Validation Methods
    // ========================================================================

    /// Define the expected schema for the graph
    ///
    /// Args:
    ///     schema_dict: A dictionary defining the schema with the following structure:
    ///         {
    ///             'nodes': {
    ///                 'NodeType': {
    ///                     'required': ['field1', 'field2'],  # Required fields
    ///                     'optional': ['field3'],            # Optional fields (for documentation)
    ///                     'types': {'field1': 'string', 'field2': 'integer'}  # Field types
    ///                 }
    ///             },
    ///             'connections': {
    ///                 'CONNECTION_TYPE': {
    ///                     'source': 'SourceNodeType',
    ///                     'target': 'TargetNodeType',
    ///                     'cardinality': 'one-to-many',  # Optional
    ///                     'required_properties': ['prop1'],  # Optional
    ///                     'property_types': {'prop1': 'float'}  # Optional
    ///                 }
    ///             }
    ///         }
    ///
    /// Returns:
    ///     Self with schema defined
    fn define_schema(&mut self, schema_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut schema = SchemaDefinition::new();

        // Parse node schemas
        if let Some(nodes_dict) = schema_dict.get_item("nodes")? {
            if let Ok(nodes) = nodes_dict.cast::<PyDict>() {
                for (node_type_key, node_schema_val) in nodes.iter() {
                    let node_type: String = node_type_key.extract()?;
                    let node_schema_dict = node_schema_val.cast::<PyDict>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                            "Schema for node type '{}' must be a dictionary",
                            node_type
                        ))
                    })?;

                    let mut node_schema = NodeSchemaDefinition::default();

                    // Parse required fields
                    if let Some(required) = node_schema_dict.get_item("required")? {
                        node_schema.required_fields = required.extract::<Vec<String>>()?;
                    }

                    // Parse optional fields
                    if let Some(optional) = node_schema_dict.get_item("optional")? {
                        node_schema.optional_fields = optional.extract::<Vec<String>>()?;
                    }

                    // Parse field types
                    if let Some(types) = node_schema_dict.get_item("types")? {
                        let types_dict = types.cast::<PyDict>().map_err(|_| {
                            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "types must be a dictionary",
                            )
                        })?;
                        for (field, type_val) in types_dict.iter() {
                            node_schema
                                .field_types
                                .insert(field.extract::<String>()?, type_val.extract::<String>()?);
                        }
                    }

                    schema.add_node_schema(node_type, node_schema);
                }
            }
        }

        // Parse connection schemas
        if let Some(connections_dict) = schema_dict.get_item("connections")? {
            if let Ok(connections) = connections_dict.cast::<PyDict>() {
                for (conn_type_key, conn_schema_val) in connections.iter() {
                    let conn_type: String = conn_type_key.extract()?;
                    let conn_schema_dict = conn_schema_val.cast::<PyDict>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                            "Schema for connection type '{}' must be a dictionary",
                            conn_type
                        ))
                    })?;

                    let source_type: String = conn_schema_dict
                        .get_item("source")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                                "Connection '{}' missing required 'source' field",
                                conn_type
                            ))
                        })?
                        .extract()?;

                    let target_type: String = conn_schema_dict
                        .get_item("target")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                                "Connection '{}' missing required 'target' field",
                                conn_type
                            ))
                        })?
                        .extract()?;

                    let mut conn_schema = ConnectionSchemaDefinition {
                        source_type,
                        target_type,
                        cardinality: None,
                        required_properties: Vec::new(),
                        property_types: HashMap::new(),
                    };

                    // Parse optional cardinality
                    if let Some(cardinality) = conn_schema_dict.get_item("cardinality")? {
                        conn_schema.cardinality = Some(cardinality.extract::<String>()?);
                    }

                    // Parse required_properties
                    if let Some(required_props) =
                        conn_schema_dict.get_item("required_properties")?
                    {
                        conn_schema.required_properties =
                            required_props.extract::<Vec<String>>()?;
                    }

                    // Parse property_types
                    if let Some(prop_types) = conn_schema_dict.get_item("property_types")? {
                        let types_dict = prop_types.cast::<PyDict>().map_err(|_| {
                            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "property_types must be a dictionary",
                            )
                        })?;
                        for (field, type_val) in types_dict.iter() {
                            conn_schema
                                .property_types
                                .insert(field.extract::<String>()?, type_val.extract::<String>()?);
                        }
                    }

                    schema.add_connection_schema(conn_type, conn_schema);
                }
            }
        }

        get_graph_mut(&mut self.inner).set_schema(schema);

        Ok(self.clone())
    }

    /// Validate the graph against the defined schema
    ///
    /// Args:
    ///     strict: If True, reports node/connection types that exist in the graph
    ///             but are not defined in the schema. Default is False.
    ///
    /// Returns:
    ///     A list of validation error dictionaries. Empty list means validation passed.
    ///     Each error dict contains:
    ///         - 'error_type': Type of error (e.g., 'missing_required_field', 'type_mismatch')
    ///         - 'message': Human-readable error message
    ///         - Additional fields depending on error type
    #[pyo3(signature = (strict=None))]
    fn validate_schema(&self, py: Python<'_>, strict: Option<bool>) -> PyResult<Py<PyAny>> {
        let schema = self.inner.get_schema().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No schema defined. Call define_schema() first.",
            )
        })?;

        let errors = crate::graph::mutation::validation::validate_graph(
            &self.inner,
            schema,
            strict.unwrap_or(false),
        );

        // Convert errors to Python list of dicts
        let result = PyList::empty(py);
        for error in errors {
            let error_dict = PyDict::new(py);

            match &error {
                schema::ValidationError::MissingRequiredField {
                    node_type,
                    node_title,
                    field,
                } => {
                    error_dict.set_item("error_type", "missing_required_field")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("node_title", node_title)?;
                    error_dict.set_item("field", field)?;
                }
                schema::ValidationError::TypeMismatch {
                    node_type,
                    node_title,
                    field,
                    expected_type,
                    actual_type,
                } => {
                    error_dict.set_item("error_type", "type_mismatch")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("node_title", node_title)?;
                    error_dict.set_item("field", field)?;
                    error_dict.set_item("expected_type", expected_type)?;
                    error_dict.set_item("actual_type", actual_type)?;
                }
                schema::ValidationError::InvalidConnectionEndpoint {
                    connection_type,
                    expected_source,
                    expected_target,
                    actual_source,
                    actual_target,
                } => {
                    error_dict.set_item("error_type", "invalid_connection_endpoint")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("expected_source", expected_source)?;
                    error_dict.set_item("expected_target", expected_target)?;
                    error_dict.set_item("actual_source", actual_source)?;
                    error_dict.set_item("actual_target", actual_target)?;
                }
                schema::ValidationError::MissingConnectionProperty {
                    connection_type,
                    source_title,
                    target_title,
                    property,
                } => {
                    error_dict.set_item("error_type", "missing_connection_property")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("source_title", source_title)?;
                    error_dict.set_item("target_title", target_title)?;
                    error_dict.set_item("property", property)?;
                }
                schema::ValidationError::UndefinedNodeType { node_type, count } => {
                    error_dict.set_item("error_type", "undefined_node_type")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("count", count)?;
                }
                schema::ValidationError::UndefinedConnectionType {
                    connection_type,
                    count,
                } => {
                    error_dict.set_item("error_type", "undefined_connection_type")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("count", count)?;
                }
            }

            error_dict.set_item("message", error.to_string())?;
            result.append(error_dict)?;
        }

        Ok(result.into())
    }

    /// Check if a schema has been defined for this graph
    fn has_schema(&self) -> bool {
        self.inner.get_schema().is_some()
    }

    /// Clear the schema definition from the graph
    fn clear_schema(&mut self) -> PyResult<Self> {
        get_graph_mut(&mut self.inner).clear_schema();
        Ok(self.clone())
    }

    /// Get the current schema definition as a dictionary
    fn schema_definition(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let schema = match self.inner.get_schema() {
            Some(s) => s,
            None => return Ok(py.None()),
        };

        let result = PyDict::new(py);

        // Convert node schemas
        let nodes_dict = PyDict::new(py);
        for (node_type, node_schema) in &schema.node_schemas {
            let schema_dict = PyDict::new(py);
            schema_dict.set_item("required", &node_schema.required_fields)?;
            schema_dict.set_item("optional", &node_schema.optional_fields)?;

            let types_dict = PyDict::new(py);
            for (field, field_type) in &node_schema.field_types {
                types_dict.set_item(field, field_type)?;
            }
            schema_dict.set_item("types", types_dict)?;

            nodes_dict.set_item(node_type, schema_dict)?;
        }
        result.set_item("nodes", nodes_dict)?;

        // Convert connection schemas
        let connections_dict = PyDict::new(py);
        for (conn_type, conn_schema) in &schema.connection_schemas {
            let schema_dict = PyDict::new(py);
            schema_dict.set_item("source", &conn_schema.source_type)?;
            schema_dict.set_item("target", &conn_schema.target_type)?;

            if let Some(cardinality) = &conn_schema.cardinality {
                schema_dict.set_item("cardinality", cardinality)?;
            }

            if !conn_schema.required_properties.is_empty() {
                schema_dict.set_item("required_properties", &conn_schema.required_properties)?;
            }

            if !conn_schema.property_types.is_empty() {
                let types_dict = PyDict::new(py);
                for (prop, prop_type) in &conn_schema.property_types {
                    types_dict.set_item(prop, prop_type)?;
                }
                schema_dict.set_item("property_types", types_dict)?;
            }

            connections_dict.set_item(conn_type, schema_dict)?;
        }
        result.set_item("connections", connections_dict)?;

        Ok(result.into())
    }

    // ========================================================================
    // Pattern Matching Methods
    // ========================================================================

    /// Match a Cypher-like pattern against the graph.
    ///
    /// Supports patterns like:
    /// - Simple node: `(p:Person)`
    /// - Single hop: `(p:Person)-[:KNOWS]->(f:Person)`
    /// - Multi-hop: `(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)`
    /// - Property filters: `(p:Person {name: "Alice"})`
    /// - Edge filters: `(a)-[:KNOWS {since: 2020}]->(b)`
    /// - Bidirectional: `(a)-[:KNOWS]-(b)` (matches both directions)
    /// - Incoming: `(a)<-[:KNOWS]-(b)` (matches edges from b to a)
    ///
    /// Syntax:
    /// - Node: `(variable:Type {property: value})`
    /// - Edge: `-[:TYPE {property: value}]->` or `<-[:TYPE]-` or `-[:TYPE]-`
    /// - Variable and type are optional: `()`, `(:Type)`, `(var)`
    ///
    /// Args:
    ///     pattern: The Cypher-like pattern string
    ///     max_matches: Maximum number of matches to return (default: unlimited)
    ///
    /// Returns:
    ///     A list of match dictionaries. Each match contains bindings for
    ///     named variables in the pattern. Node bindings have 'type', 'title',
    ///     'id', and 'properties'. Edge bindings have 'source', 'target',
    ///     'connection_type', and 'properties'.
    ///
    /// Example:
    ///     ```python
    ///     # Find all plays with their prospects
    ///     matches = graph.match_pattern('(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)')
    ///     for m in matches:
    ///         print(f"Play: {m['p']['title']}, Prospect: {m['pr']['title']}")
    ///
    /// ```text
    /// # Find discoveries from specific prospects
    /// matches = graph.match_pattern(
    ///     '(pr:Prospect {status: "Active"})-[:BECAME_DISCOVERY]->(d:Discovery)'
    /// )
    ///
    /// # Limit results
    /// top_10 = graph.match_pattern('(p:Person)-[:KNOWS]->(f:Person)', max_matches=10)
    /// ```
    /// ```
    #[pyo3(signature = (pattern, max_matches=None))]
    fn match_pattern(
        &self,
        py: Python<'_>,
        pattern: &str,
        max_matches: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        // Parse the pattern
        let parsed = crate::graph::core::pattern_matching::parse_pattern(pattern).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Pattern syntax error: {}", e))
        })?;

        // Execute the pattern
        let executor =
            crate::graph::core::pattern_matching::PatternExecutor::new(&self.inner, max_matches);
        let matches = executor.execute(&parsed).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Pattern execution error: {}",
                e
            ))
        })?;

        // Convert matches to Python
        py_out::pattern_matches_to_pylist(py, &matches, &self.inner.interner)
    }

    /// Execute a Cypher query against the graph.
    ///
    /// Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT, SKIP, WITH,
    /// OPTIONAL MATCH, UNWIND, UNION, and aggregation functions
    /// (count, sum, avg, min, max, collect, std).
    ///
    /// The MATCH clause uses the same pattern syntax as match_pattern().
    /// WHERE supports AND/OR/NOT, comparisons (=, <>, <, <=, >, >=),
    /// IS NULL, IS NOT NULL, IN, STARTS WITH, ENDS WITH, CONTAINS.
    /// RETURN supports property access (n.prop), aliases (AS), aggregation,
    /// and DISTINCT.
    ///
    /// Args:
    ///     query: The Cypher query string
    ///     timeout_ms: Deadline in milliseconds. If omitted, uses
    ///         `set_default_timeout()` when set, otherwise the built-in
    ///         default of 180_000 ms (3 min). Pass `0` to disable the
    ///         deadline entirely for this call.
    ///     max_rows: Cap on intermediate result rows; queries producing
    ///         more return an error. Defaults to `set_default_max_rows()`.
    ///
    /// Returns:
    ///     A dict with 'columns' (list of column names) and 'rows'
    ///     (list of row dicts mapping column name to value).
    ///
    /// Example:
    ///     ```python
    ///     result = graph.cypher('''
    ///         MATCH (p:Person)-[:KNOWS]->(f:Person)
    ///         WHERE p.age > 25
    ///         RETURN p.name AS person, count(f) AS friends
    ///         ORDER BY friends DESC
    ///         LIMIT 10
    ///     ''')
    ///     for row in result:
    ///         print(f"{row['person']}: {row['friends']} friends")
    ///     ```
    #[pyo3(signature = (query, *, to_df=false, params=None, timeout_ms=None, max_rows=None, streaming=true, disable_optimizer=false, disabled_passes=None))]
    #[allow(clippy::too_many_arguments)]
    fn cypher(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        query: &str,
        to_df: bool,
        params: Option<&Bound<'_, PyDict>>,
        timeout_ms: Option<u64>,
        max_rows: Option<usize>,
        streaming: bool,
        disable_optimizer: bool,
        disabled_passes: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let self_ref = slf.try_borrow().map_err(|_| concurrent_access_pyerr())?;
        let effective_timeout = timeout_ms
            .or(self_ref.default_timeout_ms)
            .or_else(|| backend_default_timeout_ms(&self_ref.inner));
        let effective_max_rows = max_rows.or(self_ref.default_max_rows);
        drop(self_ref);
        // timeout_ms == 0 is the documented escape hatch for "no deadline".
        let deadline = match effective_timeout {
            Some(0) | None => None,
            Some(ms) => Some(std::time::Instant::now() + std::time::Duration::from_millis(ms)),
        };

        // Decode params (PyDict → HashMap<String, Value>).
        let param_map = if let Some(params_dict) = params {
            let mut map = std::collections::HashMap::new();
            for (key, val) in params_dict.iter() {
                let key_str: String = key.extract()?;
                let value = py_in::py_value_to_value(&val)?;
                map.insert(key_str, value);
            }
            map
        } else {
            std::collections::HashMap::new()
        };

        // Build the planner's disabled-passes set. Hot path: when
        // both kwargs are at defaults, use the static empty-set
        // reference and skip the HashSet allocation.
        let disabled_owned: Option<std::collections::HashSet<String>> =
            if disable_optimizer || disabled_passes.is_some() {
                Some(build_disabled_passes(disable_optimizer, disabled_passes)?)
            } else {
                None
            };

        // Phase E.2: pre-parse to decide whether this is a mutation
        // (routes to execute_mut against &mut DirGraph) or a read
        // (routes to execute_read against &DirGraph via Arc snapshot).
        // The parser is cached so this second parse inside
        // session::execute is a hit, ~0 µs overhead.
        let pre_parsed = cypher::parse_cypher(query).map_err(crate::error_py::kg_to_pyerr)?;
        let is_mutation = cypher::is_mutation_query(&pre_parsed);

        // Read-only graph guard: reject mutations on a read-only kg.
        // (Separate from per-transaction read_only — this is a
        // graph-wide flag set via kg.read_only(True).)
        if is_mutation {
            let this = slf.try_borrow().map_err(|_| concurrent_access_pyerr())?;
            if this.inner.read_only {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::CypherExecution {
                        message: "Graph is in read-only mode — CREATE, SET, DELETE, REMOVE, and \
                                  MERGE are disabled. Use kg.read_only(False) to re-enable \
                                  mutations."
                            .to_string(),
                        position: None,
                    },
                ));
            }
        }

        let query_started = std::time::Instant::now();

        if is_mutation {
            // Mutation path: needs exclusive borrow of the graph.
            let mut this = slf
                .try_borrow_mut()
                .map_err(|_| concurrent_access_pyerr())?;
            let graph = get_graph_mut(&mut this.inner);

            // Embedder snapshot for text_score() — borrow ends before
            // session::execute_mut so the embed call inside can grab
            // the GIL again if needed.
            let embedder_for_opts: Option<std::sync::Arc<dyn crate::graph::embedder::Embedder>> =
                None; // mutations don't typically use text_score; skip the embedder snapshot

            let opts = crate::graph::session::ExecuteOptions {
                params: &param_map,
                deadline,
                max_rows: effective_max_rows,
                lazy_eligible: streaming,
                disabled_passes: disabled_owned.as_ref(),
                embedder: embedder_for_opts,
                // value_codecs are an MCP-manifest feature; the Python API
                // doesn't configure them (the engine path uses native types).
                value_codecs: None,
            };

            let outcome = crate::graph::session::execute_mut(graph, query, &opts)
                .map_err(crate::error_py::kg_to_pyerr)?;
            let mut result = outcome.result;
            let output_csv = outcome.output_format == cypher::OutputFormat::Csv;

            if outcome.explain {
                let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
                return Py::new(py, view).map(|v| v.into_any());
            }

            // Auto-vacuum + last-mutation stats — pyapi-specific
            // post-mutation bookkeeping.
            if let Some(ref stats) = result.stats {
                if (stats.nodes_deleted > 0 || stats.relationships_deleted > 0)
                    && graph.check_auto_vacuum()
                {
                    this.cursor.selection = schema::CowSelection::new();
                }
                this.cursor.last_mutation_stats = Some(stats.clone());
            }

            // Durability: append + fsync this mutation's WAL frame before
            // returning. No-op for non-durable graphs. (The `graph` borrow
            // above has ended; flush_wal re-borrows self.inner.)
            this.flush_wal()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            // Resolve NodeRef values to node titles before Python conversion.
            resolve_noderefs(&this.inner.graph, &mut result.rows);

            return if output_csv {
                result.to_csv().into_py_any(py)
            } else if to_df {
                let preprocessed = cypher::py_convert::preprocess_values_owned(result.rows);
                cypher::py_convert::preprocessed_result_to_dataframe(
                    py,
                    &result.columns,
                    &preprocessed,
                )
            } else {
                let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
                Py::new(py, view).map(|v| v.into_any())
            };
        }

        // Read path: clone Arc for the shared snapshot, release the
        // pyclass borrow, then route through session::execute_read
        // inside py.detach so the GIL is free during parse / optimize
        // / execute / resolve_noderefs.
        let inner = {
            let this = slf.try_borrow().map_err(|_| concurrent_access_pyerr())?;
            this.inner.clone()
        };
        let embedder_for_opts: Option<std::sync::Arc<dyn crate::graph::embedder::Embedder>> = {
            let this = slf.try_borrow().map_err(|_| concurrent_access_pyerr())?;
            this.embedder.clone()
        };
        let result = {
            let opts = crate::graph::session::ExecuteOptions {
                params: &param_map,
                deadline,
                max_rows: effective_max_rows,
                lazy_eligible: streaming,
                disabled_passes: disabled_owned.as_ref(),
                embedder: embedder_for_opts,
                // value_codecs are an MCP-manifest feature; the Python API
                // doesn't configure them (the engine path uses native types).
                value_codecs: None,
            };
            let inner_for_detach = std::sync::Arc::clone(&inner);
            py.detach(move || -> Result<crate::graph::languages::cypher::result::CypherResult, crate::error::KgError> {
                let outcome = crate::graph::session::execute_read(&inner_for_detach, query, &opts)?;
                let mut result = outcome.result;
                resolve_noderefs(&inner_for_detach.graph, &mut result.rows);
                Ok(result)
            }).map_err(crate::error_py::kg_to_pyerr)?
        };
        let elapsed_ms = query_started.elapsed().as_millis() as u64;
        // EXPLAIN: session::execute_read renders the plan into
        // result.rows — wrap in ResultView and return.
        // (Detect via lack of regular execution markers: explain
        // results have specific column names; simpler to re-parse
        // the flag from the cache.)
        let output_csv = {
            // Use the cache-hit pre_parsed AST for output_format /
            // explain detection without re-parsing inside py.detach.
            pre_parsed.output_format == cypher::OutputFormat::Csv
        };
        if pre_parsed.explain {
            let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
            return Py::new(py, view).map(|v| v.into_any());
        }

        // `Some(0)` is the documented "disable deadline" escape hatch.
        // Report it as "no deadline" (None) in diagnostics.
        let reported_timeout_ms = match effective_timeout {
            Some(0) | None => None,
            other => other,
        };
        let diagnostics = cypher::result::QueryDiagnostics {
            elapsed_ms,
            timed_out: false,
            timeout_ms: reported_timeout_ms,
            // Surface schema "did you mean?" warnings structurally so agent /
            // programmatic callers (who can't see the stderr emission) get the
            // signal too — e.g. a typo'd label that returned zero rows.
            warnings: cypher::collect_unknown_pattern_warnings(&pre_parsed, &inner),
        };
        {
            let columns = result.columns;
            let stats = result.stats;
            let profile = result.profile;
            // resolve_noderefs already happened inside the py.detach block
            // above (Phase A.3 / 0.9.53 Issue #3 partial fix).
            let rows = result.rows;
            if output_csv {
                // CSV consumes every cell — if the executor handed us a
                // lazy descriptor (RETURN was flagged `lazy_eligible`),
                // materialise it through ResultView first so to_csv()
                // sees the actual values rather than the empty rows
                // placeholder.
                if let Some(lazy_desc) = result.lazy {
                    let lazy_result = cypher::CypherResult {
                        columns: columns.clone(),
                        rows: Vec::new(),
                        stats: None,
                        profile: None,
                        diagnostics: None,
                        lazy: Some(lazy_desc),
                    };
                    let view =
                        crate::graph::pyapi::result_view::ResultView::from_cypher_result_with_graph(
                            lazy_result,
                            std::sync::Arc::clone(&inner),
                        );
                    let materialised = view
                        .materialise_all()
                        .into_iter()
                        .map(|row| {
                            row.into_iter()
                                // Phase A.1 / C7a — ParsedJson variant deleted;
                                // only Plain remains.
                                .map(|pv| match pv {
                                    cypher::py_convert::PreProcessedValue::Plain(v) => v,
                                })
                                .collect()
                        })
                        .collect();
                    let csv_result = cypher::CypherResult {
                        columns,
                        rows: materialised,
                        stats,
                        profile,
                        diagnostics: Some(diagnostics),
                        lazy: None,
                    };
                    return csv_result.to_csv().into_py_any(py);
                }
                let csv_result = cypher::CypherResult {
                    columns,
                    rows,
                    stats,
                    profile,
                    diagnostics: Some(diagnostics),
                    lazy: None,
                };
                csv_result.to_csv().into_py_any(py)
            } else if let Some(lazy_desc) = result.lazy {
                // Lazy path: planner flagged the terminal RETURN as eligible
                // and the executor skipped per-row property evaluation.
                // Hand the pending rows + return items to ResultView, which
                // materialises cells on Python access.
                let lazy_result = cypher::CypherResult {
                    columns,
                    rows: Vec::new(),
                    stats,
                    profile,
                    diagnostics: Some(diagnostics),
                    lazy: Some(lazy_desc),
                };
                let view =
                    crate::graph::pyapi::result_view::ResultView::from_cypher_result_with_graph(
                        lazy_result,
                        std::sync::Arc::clone(&inner),
                    );
                if to_df {
                    // DataFrame consumes every row — let the view materialise
                    // its lazy rows via to_df_inner (which handles both
                    // backings). For now, route through to_list-style
                    // materialisation by triggering the lazy resolver per
                    // row and rebuilding the eager form.
                    let preprocessed = view.materialise_all();
                    let cols = view.columns_owned();
                    cypher::py_convert::preprocessed_result_to_dataframe(py, &cols, &preprocessed)
                } else {
                    Py::new(py, view).map(|v| v.into_any())
                }
            } else {
                let preprocessed = cypher::py_convert::preprocess_values_owned(rows);
                if to_df {
                    cypher::py_convert::preprocessed_result_to_dataframe(
                        py,
                        &columns,
                        &preprocessed,
                    )
                } else {
                    let view = crate::graph::pyapi::result_view::ResultView::from_preprocessed(
                        columns,
                        preprocessed,
                        stats,
                        profile,
                        Some(diagnostics),
                    );
                    Py::new(py, view).map(|v| v.into_any())
                }
            }
        }
    }

    /// Mutation statistics from the last Cypher mutation query (CREATE/SET/DELETE/REMOVE/MERGE).
    ///
    /// Returns None if no mutation has been executed yet.
    #[getter]
    fn last_mutation_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.cursor.last_mutation_stats {
            Some(stats) => {
                let dict = PyDict::new(py);
                dict.set_item("nodes_created", stats.nodes_created)?;
                dict.set_item("relationships_created", stats.relationships_created)?;
                dict.set_item("properties_set", stats.properties_set)?;
                dict.set_item("nodes_deleted", stats.nodes_deleted)?;
                dict.set_item("relationships_deleted", stats.relationships_deleted)?;
                dict.set_item("properties_removed", stats.properties_removed)?;
                Ok(dict.into())
            }
            None => Ok(py.None()),
        }
    }

    // ========================================================================
    // Transaction Support
    // ========================================================================

    /// Begin a transaction — returns a Transaction object with a working copy of the graph.
    ///
    /// Creates a snapshot of the current graph state. All mutations within the
    /// transaction are isolated until ``commit()`` is called. If the transaction
    /// is rolled back (or dropped without committing), no changes are applied.
    ///
    /// **Note:** the snapshot is a full deep-clone of the graph, so creating a
    /// transaction on a very large graph has a one-time memory cost proportional
    /// to graph size. Embeddings are *not* cloned (they live outside `DirGraph`).
    ///
    /// Can also be used as a context manager:
    ///
    /// Example:
    ///     ```python
    ///     with graph.begin() as tx:
    ///         tx.cypher("CREATE (n:Person {name: 'Alice', age: 30})")
    ///         tx.cypher("CREATE (n:Person {name: 'Bob', age: 25})")
    ///         # auto-commits on success, auto-rollbacks on exception
    ///     ```
    #[pyo3(signature = (timeout_ms=None))]
    fn begin(slf: Py<Self>, timeout_ms: Option<u64>) -> PyResult<Transaction> {
        // Phase A.3 / 0.9.53 — Issue #1 fix: deferred clone.
        //
        // Previously this call deep-cloned the entire DirGraph up front,
        // costing O(graph_size) per begin(). For a 100k-node graph that's
        // ~3 ms; for 1M-node Bolt deployments that's tens of ms PER
        // SESSION. Now we take an Arc snapshot (O(1)) and defer the clone
        // until the first mutation actually lands. Read-only-then-commit
        // transactions pay zero clone cost; mutating transactions pay
        // the clone only when needed (and skip it entirely if
        // Arc::try_unwrap succeeds in the materialization path).
        let core_tx = Python::attach(|py| {
            let kg = slf.borrow(py);
            // Seed a core Transaction from the KG's current Arc. The throwaway
            // Session is dropped immediately; the Transaction owns its snapshot
            // Arc + base version. The CoW/OCC state machine now lives in core.
            crate::graph::session::Session::from_arc(Arc::clone(&kg.inner)).begin()
        });
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        Ok(Transaction {
            owner: slf,
            inner: Some(core_tx),
            deadline,
        })
    }

    /// Begin a read-only transaction — O(1) cost, zero memory overhead.
    ///
    /// Returns a Transaction backed by an Arc reference to the current graph
    /// state. Mutations (CREATE, SET, DELETE, REMOVE, MERGE) are rejected.
    ///
    /// Ideal for concurrent read-heavy workloads (e.g. MCP server agents)
    /// where you want a consistent snapshot without the cost of a full clone.
    ///
    /// Can also be used as a context manager:
    ///
    /// Example:
    ///     ```python
    ///     with graph.begin_read() as tx:
    ///         result = tx.cypher("MATCH (n:Person) RETURN n.name")
    ///         # auto-closes on exit (no commit needed)
    ///     ```
    #[pyo3(signature = (timeout_ms=None))]
    fn begin_read(slf: Py<Self>, timeout_ms: Option<u64>) -> PyResult<Transaction> {
        let core_tx = Python::attach(|py| {
            let kg = slf.borrow(py);
            crate::graph::session::Session::from_arc(Arc::clone(&kg.inner)).begin_read()
        });
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        Ok(Transaction {
            owner: slf,
            inner: Some(core_tx),
            deadline,
        })
    }
}

/// Resolve the `disable_optimizer` / `disabled_passes` kwargs into the
/// set passed to `optimize_with_disabled`. `disable_optimizer=True`
/// expands to all registered pass names; `disabled_passes` adds named
/// passes on top, validated against the registry so typos surface as a
/// `ValueError` instead of a silent no-op.
fn build_disabled_passes(
    disable_optimizer: bool,
    disabled_passes: Option<Vec<String>>,
) -> PyResult<std::collections::HashSet<String>> {
    let mut set: std::collections::HashSet<String> = std::collections::HashSet::new();
    if disable_optimizer {
        set.extend(cypher::planner::all_pass_names());
    }
    if let Some(names) = disabled_passes {
        for name in names {
            if !cypher::planner::is_known_pass(&name) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown optimizer pass: {:?}. See `kglite.cypher_pass_names()` for valid names.",
                    name
                )));
            }
            set.insert(name);
        }
    }
    Ok(set)
}

/// Backend-aware default Cypher timeout (milliseconds).
///
/// Default Cypher query deadline applied when no per-call `timeout_ms` and
/// no `set_default_timeout()` are set. A 3-minute ceiling is loose enough
/// that legitimate cold queries on large mapped/disk graphs complete, while
/// still guaranteeing that pathological scans (e.g. unanchored patterns on
/// a 100M+ node graph) error out instead of wedging the host process or an
/// MCP server. Users override per-call with `timeout_ms=N` (or `0` to
/// disable), or globally via `set_default_timeout(ms)`.
pub(crate) fn backend_default_timeout_ms(_graph: &schema::DirGraph) -> Option<u64> {
    Some(180_000)
}
