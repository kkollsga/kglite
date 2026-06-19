// src/graph/mod.rs
//
// kglite-py's graph module — re-exports the engine's `graph::*`
// subtree from the sibling `kglite` crate (imported here as
// `kglite_core` via a `package = "kglite"` dep alias in
// Cargo.toml; the alias dodges the extern-crate collision with
// this crate's own `[lib] name = "kglite_py"`). Adds the PyO3
// wrapper concerns (KnowledgeGraph #[pyclass], pyapi/ submodule,
// pyo3 param-extract helpers) that only the wrapper crate needs.
//
// Every engine subtree (algorithms, blueprint, core, dir_graph,
// explore, features, introspection, io, mutation, schema,
// session, storage) lives in `kglite_core::graph` — the glob
// re-export below keeps every `crate::graph::X::Y` path in
// pyapi/ resolving unchanged.
pub use kglite_core::graph::*;

// Mixed subtrees (have both engine and pyo3 parts) — local
// module declarations shadow the re-exported ones from the
// kglite engine crate.
pub mod embedder;
pub mod languages;
pub mod pyapi;

pub use pyapi::transaction::Transaction;

use crate::datatypes::py_out;
use crate::datatypes::values::{FilterCondition, Value};
use kglite_core::api::{DirGraph, GraphRead, OperationReport, OperationReports};
// `MutationStats` is not yet in `api::cypher` (Piece 2 lift candidate);
// `CowSelection`/`PlanStep` are the fluent cursor types (Piece 3 decision).
use kglite_core::api::{CowSelection, PlanStep};
use kglite_core::graph::languages::cypher;
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// Shadow the engine's re-exports of these types with the local
// pub(crate) versions that the pyapi closures construct directly.
// (Same name, same underlying enum — just promotes the visibility
// for local use without depending on the engine's pub visibility.)
pub(crate) type EmbeddingColumnData = Vec<(String, Vec<(Value, Vec<f32>)>)>;

/// Extract `ConnectionDetail` from a Python `bool | list[str] | None` parameter.
pub(crate) fn extract_detail_param(
    obj: Option<&Bound<'_, PyAny>>,
    param_name: &str,
) -> PyResult<introspection::ConnectionDetail> {
    let Some(obj) = obj else {
        return Ok(introspection::ConnectionDetail::Off);
    };
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(if b {
            introspection::ConnectionDetail::Overview
        } else {
            introspection::ConnectionDetail::Off
        });
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let topics: Vec<String> = list
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(introspection::ConnectionDetail::Topics(topics));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{} must be bool or list of strings",
        param_name
    )))
}

/// Extract `CypherDetail` from a Python `bool | list[str] | None` parameter.
pub(crate) fn extract_cypher_param(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<introspection::CypherDetail> {
    let Some(obj) = obj else {
        return Ok(introspection::CypherDetail::Off);
    };
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(if b {
            introspection::CypherDetail::Overview
        } else {
            introspection::CypherDetail::Off
        });
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let topics: Vec<String> = list
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(introspection::CypherDetail::Topics(topics));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "cypher must be bool or list of strings",
    ))
}

/// Extract `FluentDetail` from a Python `bool | list[str] | None` parameter.
pub(crate) fn extract_fluent_param(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<introspection::FluentDetail> {
    let Some(obj) = obj else {
        return Ok(introspection::FluentDetail::Off);
    };
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(if b {
            introspection::FluentDetail::Overview
        } else {
            introspection::FluentDetail::Off
        });
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let topics: Vec<String> = list
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(introspection::FluentDetail::Topics(topics));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "fluent must be bool or list of strings",
    ))
}

/// Resolve any `Value::NodeRef` in Cypher result rows to node titles.
/// Called just before Python conversion so that NodeRef (an internal
/// representation used to preserve node identity through collect/WITH)
/// is never exposed to Python.
/// Thin delegate to `kglite_core::graph::session::resolve_noderefs` —
/// see that function for the lookup semantics. Kept as a crate-local
/// re-export so existing wheel callers (`graph/pyapi/*.rs`) don't have
/// to change their imports; the engine logic lives in core.
pub(crate) use kglite_core::api::session::resolve_noderefs;

/// Per-caller query **cursor** state — the part of a `KnowledgeGraph` that is
/// specific to one chain of fluent calls, separable from the shared graph
/// storage and the graph's lifecycle. Carries the fluent `selection`, the
/// `date()` temporal context, the last cypher-mutation stats, and the
/// accumulated operation reports.
///
/// Cloned onto every derived view (O(1) — `selection`/`reports` are Arc-backed)
/// and reset by `copy()`. Grouping these into one struct is the first step of
/// the god-object decomposition (see `roadmap.md`): it is the seam the future
/// public `Cursor` type is built on. No behaviour change — purely a field
/// grouping.
#[derive(Clone)]
pub(crate) struct CursorState {
    /// Cow wrapper for copy-on-write selection semantics.
    pub(crate) selection: CowSelection,
    pub(crate) reports: OperationReports,
    pub(crate) last_mutation_stats: Option<cypher::result::MutationStats>,
    /// Temporal context for auto-filtering temporal nodes/connections.
    /// Set via `date()`. Default = Today (resolve at query time).
    pub(crate) temporal_context: TemporalContext,
}

impl CursorState {
    /// A fresh cursor: empty selection, no reports, no stats, default temporal
    /// context. Used by `from_arc`, `copy()`, and other fresh-graph sites.
    pub(crate) fn new() -> Self {
        CursorState {
            selection: CowSelection::new(),
            reports: OperationReports::new(),
            last_mutation_stats: None,
            temporal_context: TemporalContext::default(),
        }
    }
}

/// Main knowledge graph type exposed to Python via PyO3.
///
/// Wraps a `DirGraph` behind an `Arc` for cheap cloning (read-heavy workloads).
/// All read methods take `&self`; mutations use `Arc::make_mut` for copy-on-write.
/// Supports Cypher queries, property filtering, traversals, graph algorithms,
/// and code entity exploration methods (`find`, `source`, `context`, `toc`).
#[pyclass(skip_from_py_object)]
pub struct KnowledgeGraph {
    pub(crate) inner: Arc<DirGraph>,
    /// Per-caller fluent cursor state — see [`CursorState`].
    pub(crate) cursor: CursorState,
    /// Registered embedding model (not serialized — re-set after load).
    /// Backend-agnostic via [`embedder::Embedder`] trait: Python
    /// embedders flow through [`embedder::py_adapter::PyEmbedderAdapter`];
    /// Rust-native embedders (e.g. fastembed-rs) implement the trait
    /// directly. Switched from `Option<Py<PyAny>>` in 0.9.18 so
    /// downstream Rust binaries (kglite-mcp-server) don't inherit a
    /// libpython dep transitively.
    pub(crate) embedder: Option<Arc<dyn embedder::Embedder>>,
    /// Default per-query timeout in milliseconds. Applied to cypher() when
    /// timeout_ms is not explicitly passed. None = no timeout (default).
    pub(crate) default_timeout_ms: Option<u64>,
    /// Default maximum result rows. Applied to cypher() when max_rows is not
    /// explicitly passed. Queries exceeding this limit return an error.
    /// None = no limit (default).
    pub(crate) default_max_rows: Option<usize>,
    /// Graph lifecycle / identity — save target + durability session. See
    /// [`GraphLifecycle`].
    pub(crate) lifecycle: GraphLifecycle,
}

/// Graph **lifecycle / identity** state — the save target plus the durability
/// session. These are the fields that genuinely cannot be shared or cloned
/// freely: `durable` owns an OS `File` handle (the WAL) so it is `None` on
/// every clone/derived view, and `source_path` is the graph's save identity
/// (preserved by a true `Clone`, reset on `copy()` / derived views).
///
/// Grouped out of the `KnowledgeGraph` god-object alongside [`CursorState`]
/// (per-query) and the shared `DirGraph` (storage). This is the surface a
/// future core-`Session` lifecycle lift would target (see `roadmap.md`).
pub(crate) struct GraphLifecycle {
    /// Path this graph was opened from / last associated with, if any. Set by
    /// `kglite.open(path)` / `kglite.load(path)`; lets `save()` default to the
    /// origin file and powers the context-manager auto-save-on-close lifecycle.
    /// `None` for in-memory graphs and derived views.
    pub(crate) source_path: Option<std::path::PathBuf>,
    /// Durability state for a graph opened with `durable=True`: the
    /// session-scoped WAL handle plus the next log-sequence number. `Some`
    /// only on the primary durable graph; `None` everywhere else (a `File`
    /// handle isn't shareable). When `Some`, the backend is wrapped in
    /// `GraphBackend::Recording` so mutations are captured for the WAL.
    pub(crate) durable: Option<DurableState>,
}

impl GraphLifecycle {
    /// A detached lifecycle: no save target, not durable. Used by in-memory
    /// constructors, `copy()`, and every derived view.
    pub(crate) fn detached() -> Self {
        GraphLifecycle {
            source_path: None,
            durable: None,
        }
    }
}

/// Session-scoped durability state held by a durable [`KnowledgeGraph`].
/// Lives on the binding (not the CoW-cloned `DirGraph`) because it owns a
/// `File` handle. See `flush_wal`.
pub(crate) struct DurableState {
    pub(crate) wal: kglite_core::graph::wal::Wal,
    /// Monotonic log-sequence number stamped on the next WAL frame.
    pub(crate) next_lsn: u64,
}

impl std::fmt::Debug for DurableState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableState")
            .field("wal", &self.wal.path())
            .field("next_lsn", &self.next_lsn)
            .finish()
    }
}

// `TemporalContext`, `SourceLocation`, and `SourceLookup` live
// in the kglite engine crate and reach this module via the
// `pub use kglite_core::graph::*;` re-export at the top of this
// file (`kglite_core` is the local dep alias for the engine —
// see Cargo.toml). The previous local definitions were
// duplicates that confused type inference (function signatures
// referenced the engine version while local construction sites
// used the duplicate).

// (formerly `fn value_to_string`; consolidated 0.9.53 into
// `crate::datatypes::values::raw_string`)

impl KnowledgeGraph {
    /// Wrap an `Arc<DirGraph>` in a `KnowledgeGraph` with default
    /// binding-ergonomic state (no embedder, default temporal context,
    /// no timeout, no row cap). Used by the pyapi `load()` pyfunction,
    /// the `code_tree.build()` / `code_tree.repo_tree()` pyfunctions,
    /// and the sibling Rust crates (kglite-bolt-server,
    /// kglite-mcp-server) that consume `kglite_core::api::load_file ->
    /// Arc<DirGraph>` and need to wrap it into a `KnowledgeGraph`
    /// alongside their own state. Phase G.3-pre decoupled the
    /// engine-side constructors from the binding-side wrapper.
    pub fn from_arc(inner: Arc<DirGraph>) -> Self {
        KnowledgeGraph {
            inner,
            cursor: CursorState::new(),
            embedder: None,
            default_timeout_ms: None,
            default_max_rows: None,
            lifecycle: crate::graph::GraphLifecycle::detached(),
        }
    }

    /// Derive a new handle that shares this graph's storage, lifecycle, and
    /// embedder, carrying a cursor produced by mutating a clone of the current
    /// cursor. **The single choke point** for every fluent "narrow the
    /// selection" operation: `f` reads the (shared, immutable) graph and
    /// mutates the freshly-cloned cursor, then the derived handle is returned.
    ///
    /// This replaces the copy-pasted `let mut new_kg = self.clone(); …mutate
    /// new_kg.cursor…; Ok(new_kg)` body in the fluent methods, and is the seam
    /// the future public `Cursor` type is built on (see `roadmap.md`).
    pub(crate) fn derive_with<F>(&self, f: F) -> PyResult<Self>
    where
        F: FnOnce(&Arc<DirGraph>, &mut CursorState) -> PyResult<()>,
    {
        let mut new_kg = self.clone();
        f(&self.inner, &mut new_kg.cursor)?;
        Ok(new_kg)
    }

    /// Bind an embedder implementing the [`embedder::Embedder`] trait.
    /// The pure-Rust counterpart of the `set_embedder` pymethod —
    /// used by `kglite-mcp-server` and other Rust consumers that
    /// don't have a `Py<PyAny>` to hand. The pymethod wraps user
    /// Python objects in `PyEmbedderAdapter` and ultimately stores
    /// the same `Arc<dyn Embedder>` here.
    pub fn set_embedder_native(&mut self, embedder: Arc<dyn embedder::Embedder>) {
        self.embedder = Some(embedder);
    }

    /// Access the active backend, if any. Returns `None` until
    /// `set_embedder` / `set_embedder_native` has been called.
    pub fn embedder(&self) -> Option<&Arc<dyn embedder::Embedder>> {
        self.embedder.as_ref()
    }

    /// Wrap a `DirGraph`'s backend in the `Recording` write-capture layer
    /// so mutations are buffered for the WAL. Idempotent. Used by
    /// `kglite.open(..., durable=True)` after load / create.
    pub(crate) fn wrap_backend_for_durability(dir: &mut DirGraph) {
        use kglite_core::graph::schema::GraphBackend;
        use kglite_core::graph::storage::recording::RecordingGraph;
        if matches!(dir.graph, GraphBackend::Recording(_)) {
            return;
        }
        let inner = std::mem::replace(&mut dir.graph, GraphBackend::new());
        dir.graph = GraphBackend::Recording(Box::new(RecordingGraph::new(inner)));
    }

    /// Drain the capture buffer, resolve it to logical ops, and append a
    /// durably-`fsync`'d WAL frame. No-op for a non-durable graph or when
    /// no ops are pending. Called after each mutation on a durable graph;
    /// the `fsync` inside makes the committed mutation crash-safe before
    /// control returns to the caller.
    pub(crate) fn flush_wal(&mut self) -> std::io::Result<()> {
        if self.lifecycle.durable.is_none() {
            return Ok(());
        }
        // Drain + resolve in a scope so the `self.inner` borrow ends before
        // we touch the `self.lifecycle.durable` field (disjoint, but keep it clean).
        let ops = {
            let dir = get_graph_mut(&mut self.inner);
            let raw = match &mut dir.graph {
                kglite_core::graph::schema::GraphBackend::Recording(rg) => rg.take_ops(),
                // Not wrapped — durable state without a recording backend
                // shouldn't happen, but treat it as nothing to flush.
                _ => return Ok(()),
            };
            if raw.is_empty() {
                return Ok(());
            }
            kglite_core::graph::storage::recording::resolve_ops(&raw, &dir.graph, &dir.interner)
        };
        let ds = self
            .lifecycle
            .durable
            .as_mut()
            .expect("durable checked Some above; not cleared in between");
        let lsn = ds.next_lsn;
        ds.next_lsn += 1;
        ds.wal
            .append(&kglite_core::graph::wal::WalFrame { lsn, ops })
    }
}

impl Clone for KnowledgeGraph {
    fn clone(&self) -> Self {
        KnowledgeGraph {
            inner: Arc::clone(&self.inner),
            cursor: self.cursor.clone(), // selection/reports Arc-backed — O(1)
            embedder: self.embedder.as_ref().map(Arc::clone),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
            // A true Clone preserves the save identity (source_path) but never
            // the durable session (the WAL File handle isn't shareable).
            lifecycle: GraphLifecycle {
                source_path: self.lifecycle.source_path.clone(),
                durable: None,
            },
        }
    }
}

/// Error message shown when embed_texts/search_text is called without set_embedder().
const EMBEDDER_SKELETON_MSG: &str = "\
No embedding model registered. Call g.set_embedder(model) first.

Your model must implement:

    class MyEmbedder:
        dimension: int  # vector dimensionality (e.g. 384)

        def embed(self, texts: list[str]) -> list[list[float]]:
            # Return one vector per input text
            ...

Example with sentence-transformers:

    from sentence_transformers import SentenceTransformer

    class Embedder:
        def __init__(self, model_name=\"all-MiniLM-L6-v2\"):
            self._model = SentenceTransformer(model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()

        def embed(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(texts).tolist()

    g.set_embedder(Embedder())";

impl KnowledgeGraph {
    pub(crate) fn add_report(&mut self, report: OperationReport) -> usize {
        self.cursor.reports.add_report(report)
    }

    /// Convert a ConnectionOperationReport to a Python dict and emit a warning
    /// if any rows were skipped.
    pub(crate) fn connection_report_to_py(
        result: &kglite_core::api::ConnectionOperationReport,
        connection_type: &str,
    ) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let report_dict = PyDict::new(py);
            report_dict.set_item("operation", &result.operation_type)?;
            report_dict.set_item("timestamp", result.timestamp.to_rfc3339())?;
            report_dict.set_item("connections_created", result.connections_created)?;
            report_dict.set_item("connections_skipped", result.connections_skipped)?;
            report_dict.set_item("stubs_vivified", result.stubs_vivified)?;
            report_dict.set_item("property_fields_tracked", result.property_fields_tracked)?;
            report_dict.set_item("processing_time_ms", result.processing_time_ms)?;

            let has_errors = !result.errors.is_empty() || result.connections_skipped > 0;
            if !result.errors.is_empty() {
                report_dict.set_item("errors", &result.errors)?;
            }
            report_dict.set_item("has_errors", has_errors)?;

            // Emit a warning whenever the report flags skips or errors —
            // silent skips on bulk edge loads were a recurring footgun.
            if has_errors {
                let total = result.connections_created + result.connections_skipped;
                let detail = if result.errors.is_empty() {
                    String::new()
                } else {
                    format!(" {}", result.errors.join("; "))
                };
                let msg = if result.connections_skipped > 0 {
                    format!(
                        "add_connections('{}'): {} of {} rows skipped.{}",
                        connection_type, result.connections_skipped, total, detail
                    )
                } else {
                    format!(
                        "add_connections('{}'): completed with errors.{}",
                        connection_type, detail
                    )
                };
                let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
                let _ = PyErr::warn(
                    py,
                    py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                    cmsg.as_c_str(),
                    1,
                );
            }

            // Vivification is not an error — but the caller implicitly
            // created stub nodes, so surface it the same way.
            if result.stubs_vivified > 0 {
                let msg = format!(
                    "add_connections('{}'): {} stub node(s) vivified for missing endpoints — \
                     call purge_provisional() to drop any left unpromoted.",
                    connection_type, result.stubs_vivified
                );
                let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
                let _ = PyErr::warn(
                    py,
                    py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                    cmsg.as_c_str(),
                    1,
                );
            }

            Ok(report_dict.into())
        })
    }

    /// Thin delegate to `kglite_core::graph::handle::discover_property_keys_from_data`.
    /// Engine logic lifted to core in 0.10.1.
    pub(crate) fn discover_property_keys_from_data(
        nodes: &[(&str, &kglite_core::api::NodeData)],
        interner: &kglite_core::api::StringInterner,
    ) -> Vec<String> {
        kglite_core::api::discover_property_keys_from_data(nodes, interner)
    }

    /// Thin delegate to `kglite_core::api::infer_selection_node_type`.
    /// Engine logic lifted to core in 0.10.1.
    pub(crate) fn infer_selection_node_type(&self) -> Option<String> {
        kglite_core::api::infer_selection_node_type(&self.cursor.selection, &self.inner)
    }

    /// Get the registered embedder or return a helpful error with a skeleton.
    /// Returns an `Arc<dyn Embedder>` — call sites can downcast or just
    /// use the trait surface (`embed`, `dimension`, `load`, `unload`).
    pub(crate) fn get_embedder_or_error(&self) -> PyResult<Arc<dyn embedder::Embedder>> {
        match &self.embedder {
            Some(model) => Ok(Arc::clone(model)),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                EMBEDDER_SKELETON_MSG,
            )),
        }
    }

    /// Resolve a name (or qualified_name) to a single code entity NodeIndex.
    /// Thin delegate to the pure-Rust core impl at
    /// `kglite_core::graph::handle::resolve_code_entity` — see that
    /// function for the lookup-order semantics. Kept as a method on
    /// the wheel's `KnowledgeGraph` so the existing internal callers
    /// (`source_one`, `source_location`, `kg_fluent::find_one`) don't
    /// each have to construct the `&self.inner` borrow.
    pub(crate) fn resolve_code_entity(
        &self,
        name: &str,
        node_type: Option<&str>,
    ) -> (
        Option<NodeIndex>,
        Vec<(NodeIndex, kglite_core::api::NodeInfo)>,
    ) {
        kglite_core::api::resolve_code_entity(&self.inner, name, node_type)
    }

    /// Build a source-location dict for a single name.
    pub(crate) fn source_one(
        &self,
        py: Python,
        name: &str,
        node_type: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let (resolved, matches) = self.resolve_code_entity(name, node_type);

        let target_idx = match resolved {
            Some(idx) => idx,
            None => {
                let dict = PyDict::new(py);
                dict.set_item("name", name)?;
                if matches.is_empty() {
                    dict.set_item("error", format!("Node not found: {}", name))?;
                } else {
                    dict.set_item("ambiguous", true)?;
                    let match_list = PyList::empty(py);
                    for (_, info) in &matches {
                        let d = py_out::nodeinfo_to_pydict(py, info)?;
                        match_list.append(d)?;
                    }
                    dict.set_item("matches", match_list)?;
                }
                return Ok(dict.into());
            }
        };

        let node = self
            .inner
            .get_node(target_idx)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Node disappeared"))?;

        let dict = PyDict::new(py);
        dict.set_item("type", node.get_node_type_ref(&self.inner.interner))?;
        dict.set_item("name", py_out::value_to_py(py, &node.title())?)?;
        dict.set_item("qualified_name", py_out::value_to_py(py, &node.id())?)?;

        if let Some(v) = node.get_field_ref("file_path") {
            dict.set_item("file_path", py_out::value_to_py(py, &v)?)?;
        }
        if let Some(v) = node.get_field_ref("line_number") {
            dict.set_item("line_number", py_out::value_to_py(py, &v)?)?;
        }
        if let Some(v) = node.get_field_ref("end_line") {
            dict.set_item("end_line", py_out::value_to_py(py, &v)?)?;
        }
        if let (Some(Value::Int64(start)), Some(Value::Int64(end))) = (
            node.get_field_ref("line_number").as_deref(),
            node.get_field_ref("end_line").as_deref(),
        ) {
            dict.set_item("line_count", end - start + 1)?;
        }
        if let Some(v) = node.get_field_ref("signature") {
            dict.set_item("signature", py_out::value_to_py(py, &v)?)?;
        }

        Ok(dict.into())
    }

    /// Pure-Rust counterpart of `source_one` for `kglite::api` consumers
    /// (notably the kglite-mcp-server `read_code_source` tool). Returns
    /// an enum so callers can format ambiguous / not-found cases their
    /// own way without unpacking a PyDict.
    ///
    /// Mirrors the data shape `source_one` populates but with Rust types
    /// (Strings + i64) — see [`SourceLocation`] / [`SourceLookup`].
    ///
    /// Thin delegate to the pure-Rust core impl at
    /// `kglite_core::graph::handle::source_location`. The wheel
    /// crate keeps this method for back-compat with Python callers
    /// via `#[pymethods]`; the engine logic lives in `kglite`.
    pub fn source_location(&self, name: &str, node_type: Option<&str>) -> SourceLookup {
        kglite_core::api::source_location(&self.inner, name, node_type)
    }

    // `field_contains_ci` and `field_starts_with_ci` lifted to
    // `NodeData` methods in core (0.10.1). Call sites in pyapi/
    // use `node.field_contains_ci(...)` directly now.
}

/// Parse spatial column_types entries and produce a SpatialConfig + cleaned column_types dict.
///
/// Recognizes: `location.lat`, `location.lon`, `geometry`, `point.<name>.lat`,
/// `point.<name>.lon`, `shape.<name>`. These are replaced with natural storage
/// types (`float` / `str`) in the returned dict so `pandas_to_dataframe` can handle them.
///
/// Returns `(Some(config), cleaned_dict)` if any spatial entries were found,
/// or `(None, original_dict)` if none were found.
pub(crate) fn parse_spatial_column_types(
    py: Python<'_>,
    column_types: &Bound<'_, PyDict>,
) -> PyResult<(Option<kglite_core::api::SpatialConfig>, Py<PyDict>)> {
    // PyO3-only boundary: extract dict → Vec<(String, String)>,
    // delegate to core, repack cleaned pairs into a fresh PyDict.
    // Engine logic in `kglite::api::parse_spatial_column_types_from_pairs`
    // (lifted in 0.10.1).
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (key, value) in column_types.iter() {
        let col_name: String = key.extract()?;
        let type_str: String = value.extract()?;
        pairs.push((col_name, type_str));
    }
    let (config, cleaned_pairs) = kglite_core::api::parse_spatial_column_types_from_pairs(pairs)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let cleaned = PyDict::new(py);
    for (col_name, type_str) in cleaned_pairs {
        cleaned.set_item(col_name, type_str)?;
    }
    Ok((config, cleaned.unbind()))
}

/// Parse temporal column_types entries and produce a TemporalConfig + cleaned column_types dict.
///
/// Recognizes: `validFrom`, `validTo`. These are replaced with `datetime` in the
/// returned dict so `pandas_to_dataframe` can handle them as date columns.
///
/// Returns `(Some(config), cleaned_dict)` if both validFrom and validTo were found,
/// or `(None, original_dict)` if neither or only one was found.
pub(crate) fn parse_temporal_column_types(
    py: Python<'_>,
    column_types: &Bound<'_, PyDict>,
) -> PyResult<(Option<kglite_core::api::TemporalConfig>, Py<PyDict>)> {
    // PyO3-only boundary; engine logic in
    // `kglite::api::parse_temporal_column_types_from_pairs` (0.10.1).
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (key, value) in column_types.iter() {
        let col_name: String = key.extract()?;
        let type_str: String = value.extract()?;
        pairs.push((col_name, type_str));
    }
    let (config, cleaned_pairs) = kglite_core::api::parse_temporal_column_types_from_pairs(pairs)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let cleaned = PyDict::new(py);
    for (col_name, type_str) in cleaned_pairs {
        cleaned.set_item(col_name, type_str)?;
    }
    Ok((config, cleaned.unbind()))
}

// ─── Inline timeseries parsing ──────────────────────────────────────────────

// `TimeSpec` and `InlineTimeseriesConfig` moved to
// `kglite_core::graph::features::timeseries` in 0.10.1. Re-exported
// under the wheel's previous paths so the local `parse_inline_timeseries`
// + downstream `pyapi/*.rs` callers compile unchanged.
pub(crate) use kglite_core::api::{InlineTimeseriesConfig, TimeSpec};

/// Parse the `timeseries` PyDict parameter from `add_nodes`.
///
/// Expected keys:
/// - `time` (required): column name (string) or dict mapping `year`, `month`, `day`, `hour`, `minute` to column names
/// - `channels` (required): list of column names for timeseries data
/// - `resolution` (optional): "year", "month", "day", "hour", "minute" — auto-detected if omitted
/// - `units` (optional): dict mapping channel name to unit string
pub(crate) fn parse_inline_timeseries(
    ts_dict: &Bound<'_, PyDict>,
) -> PyResult<InlineTimeseriesConfig> {
    // Parse 'time' key (required)
    let time_val = ts_dict
        .get_item("time")?
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "timeseries dict requires a 'time' key (column name or dict of year/month/day/hour/minute)",
            )
        })?;

    // PyO3-only step: extract the heterogeneous `time` value into either
    // a String (Variant A) or a HashMap (Variant B). Engine logic in
    // core's `InlineTimeseriesConfig::from_components` validates the
    // shape and assembles the config (lifted in 0.10.1).
    let (time_col, time_components) = if let Ok(col_name) = time_val.extract::<String>() {
        (Some(col_name), None)
    } else if let Ok(dict) = time_val.cast::<PyDict>() {
        let mut map: HashMap<String, String> = HashMap::new();
        for (key, val) in dict.iter() {
            let k: String = key.extract()?;
            let v: String = val.extract()?;
            map.insert(k, v);
        }
        (None, Some(map))
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "timeseries 'time' must be a column name (str) or dict of {year/month/day/hour/minute: col_name}",
        ));
    };

    let channels: Vec<String> = ts_dict
        .get_item("channels")?
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "timeseries dict requires a 'channels' key (list of column names)",
            )
        })?
        .extract()?;

    let resolution: Option<String> = ts_dict
        .get_item("resolution")?
        .map(|v| v.extract())
        .transpose()?;

    let units: HashMap<String, String> = ts_dict
        .get_item("units")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();

    InlineTimeseriesConfig::from_components(time_col, time_components, channels, resolution, units)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// Helper function to get a mutable DirGraph from Arc.
/// Uses Arc::make_mut which clones only if there are other references,
/// otherwise gives a mutable reference in place. Callers mutate the graph
/// through the returned reference — no extraction/replacement needed.
///
/// WARNING: If other Arc references exist (e.g., a ResultView still in Python
/// scope, or a cloned KnowledgeGraph), this will deep-clone the entire DirGraph
/// including all nodes, edges, and indices. In read-heavy workloads this is fine,
/// but be aware that a lingering reference can cause unexpected memory spikes on mutation.
/// Thin delegate to `kglite_core::graph::dir_graph::make_dir_graph_mut` —
/// renamed in the lift; this `use` alias keeps the wheel's existing
/// `get_graph_mut` callers compiling unchanged.
pub(crate) use kglite_core::api::make_dir_graph_mut as get_graph_mut;

/// Lightweight centrality result conversion: returns {title: score} dict.
/// Creates ONE Python dict instead of N dicts — returns {title: score} format.
/// ~3-4x faster PyO3 serialization for large graphs.
pub(crate) fn centrality_results_to_py_dict(
    py: Python<'_>,
    graph: &DirGraph,
    results: Vec<kglite_core::api::algorithms::CentralityResult>,
    top_k: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let limit = top_k.unwrap_or(results.len());
    let scores_dict = PyDict::new(py);

    for result in results.into_iter().take(limit) {
        if let Some(node) = graph.get_node(result.node_idx) {
            let id_py = py_out::value_to_py(py, &node.id())?;
            scores_dict.set_item(id_py, result.score)?;
        }
    }

    Ok(scores_dict.into())
}

/// Convert centrality results to a pandas DataFrame with columns:
/// type, title, id, score — sorted by score descending.
pub(crate) fn centrality_results_to_dataframe(
    py: Python<'_>,
    graph: &DirGraph,
    results: Vec<kglite_core::api::algorithms::CentralityResult>,
    top_k: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let limit = top_k.unwrap_or(results.len());

    let mut types: Vec<&str> = Vec::with_capacity(limit);
    let mut titles: Vec<String> = Vec::with_capacity(limit);
    let mut ids: Vec<Py<PyAny>> = Vec::with_capacity(limit);
    let mut scores: Vec<f64> = Vec::with_capacity(limit);

    for result in results.into_iter().take(limit) {
        if let Some(node) = graph.get_node(result.node_idx) {
            types.push(node.node_type_str(&graph.interner));
            let node_title = node.title();
            let title_str = match &*node_title {
                Value::String(s) => s.clone(),
                _ => String::new(),
            };
            titles.push(title_str);
            ids.push(py_out::value_to_py(py, &node.id())?);
            scores.push(result.score);
        }
    }

    let pd = py.import("pandas")?;
    let data = PyDict::new(py);
    data.set_item("type", PyList::new(py, &types)?)?;
    data.set_item("title", PyList::new(py, &titles)?)?;
    data.set_item("id", PyList::new(py, &ids)?)?;
    data.set_item("score", PyList::new(py, &scores)?)?;

    let df = pd.call_method1("DataFrame", (data,))?;
    Ok(df.unbind())
}

/// Helper to convert community detection results to Python dict.
/// Accesses node data directly and uses interned keys for faster dict construction.
pub(crate) fn community_results_to_py(
    py: Python<'_>,
    graph: &DirGraph,
    result: kglite_core::api::algorithms::CommunityResult,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    // Pre-intern keys
    let key_type = pyo3::intern!(py, "type");
    let key_title = pyo3::intern!(py, "title");
    let key_id = pyo3::intern!(py, "id");

    // Group nodes by community
    let communities = PyDict::new(py);
    let mut grouped: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for a in &result.assignments {
        grouped.entry(a.community_id).or_default().push(a.node_idx);
    }

    for (comm_id, members) in &grouped {
        let member_list = PyList::empty(py);
        for &node_idx in members {
            if let Some(node) = graph.get_node(node_idx) {
                let node_dict = PyDict::new(py);
                node_dict.set_item(key_type, node.node_type_str(&graph.interner))?;
                let node_title = node.title();
                let title_str = match &*node_title {
                    Value::String(s) => s.as_str(),
                    _ => "",
                };
                node_dict.set_item(key_title, title_str)?;
                node_dict.set_item(key_id, py_out::value_to_py(py, &node.id())?)?;
                member_list.append(node_dict)?;
            }
        }
        communities.set_item(comm_id, member_list)?;
    }

    dict.set_item("communities", communities)?;
    dict.set_item("modularity", result.modularity)?;
    dict.set_item("num_communities", result.num_communities)?;

    Ok(dict.into())
}

/// Parse the `method` parameter of `traverse()` — accepts a string or dict.
///
/// String shorthand: `method='contains'` → MethodConfig with defaults.
/// Dict form: `method={'type': 'distance', 'max_m': 5000, 'resolve': 'centroid'}`
pub(crate) fn parse_method_param(
    val: &Bound<'_, PyAny>,
) -> PyResult<kglite_core::api::fluent::MethodConfig> {
    use kglite_core::api::fluent::MethodConfig;

    // Try string first
    if let Ok(s) = val.extract::<String>() {
        return Ok(MethodConfig::from_string(s));
    }

    // Try dict
    let dict = val.cast::<PyDict>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "method= must be a string (e.g. 'contains') or a dict (e.g. {'type': 'distance', 'max_m': 5000})"
        )
    })?;

    // PyO3-only boundary: extract each dict field, then hand off to
    // `MethodConfig::from_components` (lifted to core in 0.10.1) which
    // validates the resolve string and assembles the struct.
    let method_type: String = dict
        .get_item("type")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "method dict must contain 'type' key (e.g. {'type': 'contains'})",
            )
        })?
        .extract()?;
    let resolve_str: Option<String> = dict.get_item("resolve")?.map(|v| v.extract()).transpose()?;
    let max_distance_m: Option<f64> = dict.get_item("max_m")?.map(|v| v.extract()).transpose()?;
    let geometry_field: Option<String> = dict
        .get_item("geometry")?
        .map(|v| v.extract())
        .transpose()?;
    let property: Option<String> = dict
        .get_item("property")?
        .map(|v| v.extract())
        .transpose()?;
    let threshold: Option<f64> = dict
        .get_item("threshold")?
        .map(|v| v.extract())
        .transpose()?;
    let metric: Option<String> = dict.get_item("metric")?.map(|v| v.extract()).transpose()?;
    let algorithm: Option<String> = dict
        .get_item("algorithm")?
        .map(|v| v.extract())
        .transpose()?;
    let features: Option<Vec<String>> = dict
        .get_item("features")?
        .map(|v| v.extract())
        .transpose()?;
    let k: Option<usize> = dict.get_item("k")?.map(|v| v.extract()).transpose()?;
    let eps: Option<f64> = dict.get_item("eps")?.map(|v| v.extract()).transpose()?;
    let min_samples: Option<usize> = dict
        .get_item("min_samples")?
        .map(|v| v.extract())
        .transpose()?;

    MethodConfig::from_components(
        method_type,
        resolve_str,
        max_distance_m,
        geometry_field,
        property,
        threshold,
        metric,
        algorithm,
        features,
        k,
        eps,
        min_samples,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Shared comparison traversal logic used by `compare()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_inner(
    inner: &Arc<DirGraph>,
    selection: &mut CowSelection,
    target_type: Option<&str>,
    config: &kglite_core::api::fluent::MethodConfig,
    conditions: Option<&HashMap<String, FilterCondition>>,
    sort_fields: Option<&Vec<(String, bool)>>,
    limit: Option<usize>,
    estimated: usize,
) -> PyResult<usize> {
    kglite_core::api::fluent::make_comparison_traversal(
        inner,
        selection,
        target_type,
        config,
        conditions,
        sort_fields,
        limit,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let actual = selection
        .get_level(selection.get_level_count().saturating_sub(1))
        .map(|l| l.node_count())
        .unwrap_or(0);
    selection.add_plan_step(
        PlanStep::new(
            "COMPARE",
            Some(target_type.unwrap_or(&config.method_type)),
            estimated,
        )
        .with_actual_rows(actual),
    );
    Ok(actual)
}
