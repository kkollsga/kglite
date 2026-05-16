// src/graph/mod.rs
use crate::datatypes::py_out;
use crate::datatypes::values::{FilterCondition, Value};
use crate::graph::introspection::reporting::{OperationReport, OperationReports};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub mod algorithms;
pub mod blueprint;
pub mod core;
pub mod dir_graph;
pub mod embedder;
pub mod explore;
pub mod features;
pub mod introspection;
pub mod io;
pub mod languages;
pub mod mutation;
pub mod schema;
pub mod storage;

pub mod pyapi;

pub use pyapi::transaction::Transaction;

use languages::cypher;
use schema::{CowSelection, DirGraph, PlanStep};

/// Embedding column data extracted from a DataFrame: `[(column_name, [(node_id, embedding)])]`
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
pub(crate) fn resolve_noderefs(graph: &schema::GraphBackend, rows: &mut [Vec<Value>]) {
    for row in rows.iter_mut() {
        for val in row.iter_mut() {
            if let Value::NodeRef(idx) = val {
                let node_idx = petgraph::graph::NodeIndex::new(*idx as usize);
                if let Some(node) = graph.node_weight(node_idx) {
                    *val = node.title().into_owned();
                } else {
                    *val = Value::Null;
                }
            }
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
    pub(crate) selection: CowSelection, // Using Cow wrapper for copy-on-write semantics
    pub(crate) reports: OperationReports,
    pub(crate) last_mutation_stats: Option<cypher::result::MutationStats>,
    /// Registered embedding model (not serialized — re-set after load).
    /// Backend-agnostic via [`embedder::Embedder`] trait: Python
    /// embedders flow through [`embedder::py_adapter::PyEmbedderAdapter`];
    /// Rust-native embedders (e.g. fastembed-rs) implement the trait
    /// directly. Switched from `Option<Py<PyAny>>` in 0.9.18 so
    /// downstream Rust binaries (kglite-mcp-server) don't inherit a
    /// libpython dep transitively.
    pub(crate) embedder: Option<Arc<dyn embedder::Embedder>>,
    /// Temporal context for auto-filtering temporal nodes/connections.
    /// Set via `date()` method. Default = Today (resolve at query time).
    pub(crate) temporal_context: TemporalContext,
    /// Default per-query timeout in milliseconds. Applied to cypher() when
    /// timeout_ms is not explicitly passed. None = no timeout (default).
    pub(crate) default_timeout_ms: Option<u64>,
    /// Default maximum result rows. Applied to cypher() when max_rows is not
    /// explicitly passed. Queries exceeding this limit return an error.
    /// None = no limit (default).
    pub(crate) default_max_rows: Option<usize>,
}

/// Temporal context for automatic date filtering on select/traverse/collect.
/// Set via the `date()` method. Carried through clone (fluent API chaining).
#[derive(Clone, Debug, Default)]
pub(crate) enum TemporalContext {
    /// Use today's date (default). Resolved at query time.
    #[default]
    Today,
    /// Point-in-time: valid_from <= date AND (valid_to IS NULL OR valid_to >= date).
    At(chrono::NaiveDate),
    /// Range overlap: valid_from <= end AND (valid_to IS NULL OR valid_to >= start).
    During(chrono::NaiveDate, chrono::NaiveDate),
    /// No temporal filtering — show everything regardless of validity dates.
    All,
}

impl TemporalContext {
    fn is_all(&self) -> bool {
        matches!(self, TemporalContext::All)
    }
}

/// Resolved code-entity location returned by [`KnowledgeGraph::source_location`].
/// All optional fields mirror what `code_tree` stores on the node — graphs
/// built from non-code-tree sources may have fewer populated.
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub type_name: String,
    pub name: String,
    pub qualified_name: String,
    pub file_path: Option<String>,
    pub line_number: Option<i64>,
    pub end_line: Option<i64>,
    pub signature: Option<String>,
}

/// Outcome of a [`KnowledgeGraph::source_location`] lookup.
#[derive(Debug, Clone)]
pub enum SourceLookup {
    Found(SourceLocation),
    /// Multiple code entities matched the given (name, node_type). The
    /// payload lists each match's qualified_name so the caller can ask
    /// the agent to disambiguate.
    Ambiguous(Vec<String>),
    NotFound,
}

/// Render a `Value` into a `String` for the pure-Rust source-location
/// API. Mirrors `py_out::value_to_py`'s coercion for the field types
/// `code_tree` actually emits (String / Int64 / UniqueId).
fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Int64(n) => n.to_string(),
        Value::UniqueId(u) => u.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Null => String::new(),
        other => format!("{:?}", other),
    }
}

impl KnowledgeGraph {
    /// Create a fresh in-memory KnowledgeGraph without going through PyO3.
    /// Used by internal Rust modules (e.g. code_tree) that need to build a
    /// graph from native data without holding the GIL.
    pub(crate) fn new_empty() -> Self {
        KnowledgeGraph {
            inner: Arc::new(DirGraph::new()),
            selection: CowSelection::new(),
            reports: OperationReports::new(),
            last_mutation_stats: None,
            embedder: None,
            temporal_context: TemporalContext::default(),
            default_timeout_ms: None,
            default_max_rows: None,
        }
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
}

impl Clone for KnowledgeGraph {
    fn clone(&self) -> Self {
        KnowledgeGraph {
            inner: Arc::clone(&self.inner),
            selection: self.selection.clone(), // Arc clone - O(1), shares data
            reports: self.reports.clone(),
            last_mutation_stats: self.last_mutation_stats.clone(),
            embedder: self.embedder.as_ref().map(Arc::clone),
            temporal_context: self.temporal_context.clone(),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
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
        self.reports.add_report(report)
    }

    /// Convert a ConnectionOperationReport to a Python dict and emit a warning
    /// if any rows were skipped.
    pub(crate) fn connection_report_to_py(
        result: &crate::graph::introspection::reporting::ConnectionOperationReport,
        connection_type: &str,
    ) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let report_dict = PyDict::new(py);
            report_dict.set_item("operation", &result.operation_type)?;
            report_dict.set_item("timestamp", result.timestamp.to_rfc3339())?;
            report_dict.set_item("connections_created", result.connections_created)?;
            report_dict.set_item("connections_skipped", result.connections_skipped)?;
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

            Ok(report_dict.into())
        })
    }

    /// Discover property keys by scanning node data (fallback for to_df).
    pub(crate) fn discover_property_keys_from_data(
        nodes: &[(&str, &schema::NodeData)],
        interner: &schema::StringInterner,
    ) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut keys = Vec::new();
        for (_, node) in nodes {
            for key in node.property_keys(interner) {
                if seen.insert(key.to_string()) {
                    keys.push(key.to_string());
                }
            }
        }
        keys.sort();
        keys
    }

    /// Infer the node type of the current (latest level) selection by sampling
    /// the first node. Returns None if the selection is empty.
    pub(crate) fn infer_selection_node_type(&self) -> Option<String> {
        let level_idx = self.selection.get_level_count().saturating_sub(1);
        let level = self.selection.get_level(level_idx)?;
        let first_idx = level.iter_node_indices().next()?;
        self.inner
            .graph
            .node_weight(first_idx)
            .map(|n| n.node_type_str(&self.inner.interner).to_string())
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

    /// Code entity node types used by find/context/source.
    const CODE_TYPES: &[&str] = &[
        "Function",
        "Struct",
        "Class",
        "Enum",
        "Trait",
        "Protocol",
        "Interface",
        "Module",
        "Constant",
    ];

    /// Resolve a name (or qualified_name) to a single code entity NodeIndex.
    ///
    /// Returns:
    /// - `Ok(Ok(idx))` — uniquely resolved
    /// - `Ok(Err(matches))` — ambiguous (>1) or not found (0)
    pub(crate) fn resolve_code_entity(
        &self,
        name: &str,
        node_type: Option<&str>,
    ) -> (Option<NodeIndex>, Vec<(NodeIndex, schema::NodeInfo)>) {
        let name_val = Value::String(name.to_string());
        let types_to_search: Vec<&str> = match node_type {
            Some(nt) => vec![nt],
            None => Self::CODE_TYPES.to_vec(),
        };

        // Try qualified_name (stored as "id") exact match first
        for nt in &types_to_search {
            if let Some(indices) = self.inner.type_indices.get(nt) {
                for idx in indices.iter() {
                    if let Some(node) = self.inner.get_node(idx) {
                        if *node.id() == name_val {
                            return (Some(idx), Vec::new());
                        }
                    }
                }
            }
        }

        // Try qualified_name suffix match (e.g. "CypherExecutor::execute_single_clause"
        // matches "crate::graph::languages::cypher::executor::CypherExecutor::execute_single_clause")
        if name.contains("::") {
            let suffix = format!("::{}", name);
            let mut matches: Vec<(NodeIndex, schema::NodeInfo)> = Vec::new();
            for nt in &types_to_search {
                if let Some(indices) = self.inner.type_indices.get(nt) {
                    for idx in indices.iter() {
                        if let Some(node) = self.inner.get_node(idx) {
                            if let Value::String(qn) = &*node.id() {
                                if qn.ends_with(&suffix) {
                                    matches.push((idx, node.to_node_info(&self.inner.interner)));
                                }
                            }
                        }
                    }
                }
            }
            if matches.len() == 1 {
                return (Some(matches[0].0), matches);
            } else if !matches.is_empty() {
                return (None, matches);
            }
        }

        // Fall back to name/title search
        let mut matches: Vec<(NodeIndex, schema::NodeInfo)> = Vec::new();
        for nt in &types_to_search {
            if let Some(indices) = self.inner.type_indices.get(nt) {
                for idx in indices.iter() {
                    if let Some(node) = self.inner.get_node(idx) {
                        let name_match = node
                            .get_field_ref("name")
                            .map(|v| *v == name_val)
                            .unwrap_or(false)
                            || node
                                .get_field_ref("title")
                                .map(|v| *v == name_val)
                                .unwrap_or(false);
                        if name_match {
                            matches.push((idx, node.to_node_info(&self.inner.interner)));
                        }
                    }
                }
            }
        }

        if matches.len() == 1 {
            (Some(matches[0].0), matches)
        } else {
            (None, matches)
        }
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
    pub fn source_location(&self, name: &str, node_type: Option<&str>) -> SourceLookup {
        let (resolved, matches) = self.resolve_code_entity(name, node_type);

        if let Some(target_idx) = resolved {
            let node = match self.inner.get_node(target_idx) {
                Some(n) => n,
                None => return SourceLookup::NotFound,
            };
            let type_name = node.get_node_type_ref(&self.inner.interner).to_string();
            let entity_name = value_to_string(&node.title());
            let qname = value_to_string(&node.id());
            let file_path = node
                .get_field_ref("file_path")
                .as_deref()
                .map(value_to_string);
            let line_number = node
                .get_field_ref("line_number")
                .as_deref()
                .and_then(|v| match v {
                    Value::Int64(n) => Some(*n),
                    _ => None,
                });
            let end_line = node
                .get_field_ref("end_line")
                .as_deref()
                .and_then(|v| match v {
                    Value::Int64(n) => Some(*n),
                    _ => None,
                });
            let signature = node
                .get_field_ref("signature")
                .as_deref()
                .map(value_to_string);
            SourceLookup::Found(SourceLocation {
                type_name,
                name: entity_name,
                qualified_name: qname,
                file_path,
                line_number,
                end_line,
                signature,
            })
        } else if matches.is_empty() {
            SourceLookup::NotFound
        } else {
            let qnames: Vec<String> = matches
                .iter()
                .map(|(_, info)| value_to_string(&info.id))
                .collect();
            SourceLookup::Ambiguous(qnames)
        }
    }

    /// Check if a node's field value contains the given lowercase string (case-insensitive).
    pub(crate) fn field_contains_ci(
        node: &schema::NodeData,
        field: &str,
        needle_lower: &str,
    ) -> bool {
        node.get_field_ref(field)
            .and_then(|v| match &*v {
                Value::String(s) => Some(s.to_lowercase().contains(needle_lower)),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Check if a node's field value starts with the given lowercase string (case-insensitive).
    pub(crate) fn field_starts_with_ci(
        node: &schema::NodeData,
        field: &str,
        prefix_lower: &str,
    ) -> bool {
        node.get_field_ref(field)
            .and_then(|v| match &*v {
                Value::String(s) => Some(s.to_lowercase().starts_with(prefix_lower)),
                _ => None,
            })
            .unwrap_or(false)
    }
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
) -> PyResult<(Option<schema::SpatialConfig>, Py<PyDict>)> {
    let cleaned = PyDict::new(py);
    let mut config = schema::SpatialConfig::default();
    let mut has_spatial = false;

    // Track partial location/point entries (need both lat and lon)
    let mut location_lat: Option<String> = None;
    let mut location_lon: Option<String> = None;
    let mut point_lats: HashMap<String, String> = HashMap::new();
    let mut point_lons: HashMap<String, String> = HashMap::new();

    for (key, value) in column_types.iter() {
        let col_name: String = key.extract()?;
        let type_str: String = value.extract()?;
        let type_lower = type_str.to_lowercase();

        match type_lower.as_str() {
            "location.lat" => {
                location_lat = Some(col_name.clone());
                cleaned.set_item(&col_name, "float")?;
                has_spatial = true;
            }
            "location.lon" => {
                location_lon = Some(col_name.clone());
                cleaned.set_item(&col_name, "float")?;
                has_spatial = true;
            }
            "geometry" => {
                config.geometry = Some(col_name.clone());
                cleaned.set_item(&col_name, "str")?;
                has_spatial = true;
            }
            _ if type_lower.starts_with("point.") => {
                // point.<name>.lat or point.<name>.lon
                let parts: Vec<&str> = type_lower.splitn(3, '.').collect();
                if parts.len() == 3 {
                    let name = parts[1].to_string();
                    match parts[2] {
                        "lat" => {
                            point_lats.insert(name, col_name.clone());
                        }
                        "lon" => {
                            point_lons.insert(name, col_name.clone());
                        }
                        _ => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Invalid spatial type '{}' for column '{}'. \
                                     Expected 'point.<name>.lat' or 'point.<name>.lon'.",
                                type_str, col_name
                            )));
                        }
                    }
                    cleaned.set_item(&col_name, "float")?;
                    has_spatial = true;
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid spatial type '{}' for column '{}'. \
                         Expected 'point.<name>.lat' or 'point.<name>.lon'.",
                        type_str, col_name
                    )));
                }
            }
            _ if type_lower.starts_with("shape.") => {
                // shape.<name>
                let parts: Vec<&str> = type_lower.splitn(2, '.').collect();
                if parts.len() == 2 {
                    let name = parts[1].to_string();
                    config.shapes.insert(name, col_name.clone());
                    cleaned.set_item(&col_name, "str")?;
                    has_spatial = true;
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid spatial type '{}' for column '{}'.",
                        type_str, col_name
                    )));
                }
            }
            _ => {
                // Non-spatial type — pass through unchanged
                cleaned.set_item(&col_name, &type_str)?;
            }
        }
    }

    if !has_spatial {
        return Ok((None, column_types.clone().unbind()));
    }

    // Assemble location
    match (location_lat, location_lon) {
        (Some(lat), Some(lon)) => config.location = Some((lat, lon)),
        (Some(_), None) | (None, Some(_)) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Incomplete location: both 'location.lat' and 'location.lon' must be specified.",
            ));
        }
        (None, None) => {}
    }

    // Assemble named points
    let all_point_names: HashSet<&String> = point_lats.keys().chain(point_lons.keys()).collect();
    for name in all_point_names {
        match (point_lats.get(name), point_lons.get(name)) {
            (Some(lat), Some(lon)) => {
                config
                    .points
                    .insert(name.clone(), (lat.clone(), lon.clone()));
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Incomplete point '{}': both 'point.{}.lat' and 'point.{}.lon' must be specified.",
                    name, name, name
                )));
            }
        }
    }

    Ok((Some(config), cleaned.unbind()))
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
) -> PyResult<(Option<schema::TemporalConfig>, Py<PyDict>)> {
    let cleaned = PyDict::new(py);
    let mut valid_from_col: Option<String> = None;
    let mut valid_to_col: Option<String> = None;

    for (key, value) in column_types.iter() {
        let col_name: String = key.extract()?;
        let type_str: String = value.extract()?;
        let type_lower = type_str.to_lowercase();

        match type_lower.as_str() {
            "validfrom" => {
                valid_from_col = Some(col_name.clone());
                cleaned.set_item(&col_name, "datetime")?;
            }
            "validto" => {
                valid_to_col = Some(col_name.clone());
                cleaned.set_item(&col_name, "datetime")?;
            }
            _ => {
                cleaned.set_item(&col_name, &type_str)?;
            }
        }
    }

    match (valid_from_col, valid_to_col) {
        (Some(from), Some(to)) => Ok((
            Some(schema::TemporalConfig {
                valid_from: from,
                valid_to: to,
            }),
            cleaned.unbind(),
        )),
        (Some(_), None) | (None, Some(_)) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Incomplete temporal config: both 'validFrom' and 'validTo' column types must be specified.",
        )),
        (None, None) => Ok((None, column_types.clone().unbind())),
    }
}

// ─── Inline timeseries parsing ──────────────────────────────────────────────

/// How the time column(s) are specified in the `timeseries` dict.
pub(crate) enum TimeSpec {
    /// Single column with date strings: "2020-01", "2020-01-15 10:30"
    StringColumn(String),
    /// Separate columns ordered by depth: [year_col, month_col, ...]
    SeparateColumns(Vec<String>),
}

/// Parsed inline timeseries configuration from the `timeseries` dict.
pub(crate) struct InlineTimeseriesConfig {
    pub(crate) time: TimeSpec,
    pub(crate) channels: Vec<String>,
    pub(crate) resolution: Option<String>,
    pub(crate) units: HashMap<String, String>,
}

impl InlineTimeseriesConfig {
    /// All column names used by the timeseries config (for exclusion from node properties).
    pub(crate) fn all_columns(&self) -> Vec<String> {
        let mut cols = self.channels.clone();
        match &self.time {
            TimeSpec::StringColumn(c) => cols.push(c.clone()),
            TimeSpec::SeparateColumns(cs) => cols.extend(cs.iter().cloned()),
        }
        cols
    }
}

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

    let time = if let Ok(col_name) = time_val.extract::<String>() {
        TimeSpec::StringColumn(col_name)
    } else if let Ok(dict) = time_val.cast::<PyDict>() {
        // Map semantic keys to column names, ordered by depth
        let semantic_order = ["year", "month", "day", "hour", "minute"];
        let mut ordered_cols = Vec::new();
        let mut found_gap = false;

        for &key in &semantic_order {
            if let Some(val) = dict.get_item(key)? {
                if found_gap {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "timeseries time dict has '{}' but is missing a higher-level component",
                        key
                    )));
                }
                let col: String = val.extract()?;
                ordered_cols.push(col);
            } else {
                found_gap = true;
            }
        }

        if ordered_cols.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "timeseries time dict must contain at least 'year'",
            ));
        }

        TimeSpec::SeparateColumns(ordered_cols)
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "timeseries 'time' must be a column name (str) or dict of {year/month/day/hour/minute: col_name}",
        ));
    };

    // Parse 'channels' key (required)
    let channels_val = ts_dict.get_item("channels")?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "timeseries dict requires a 'channels' key (list of column names)",
        )
    })?;
    let channels: Vec<String> = channels_val.extract()?;
    if channels.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "timeseries 'channels' must not be empty",
        ));
    }

    // Parse 'resolution' key (optional)
    let resolution = if let Some(val) = ts_dict.get_item("resolution")? {
        let r: String = val.extract()?;
        crate::graph::features::timeseries::validate_resolution(&r)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Some(r)
    } else {
        None
    };

    // Parse 'units' key (optional)
    let units = if let Some(val) = ts_dict.get_item("units")? {
        val.extract::<HashMap<String, String>>()?
    } else {
        HashMap::new()
    };

    Ok(InlineTimeseriesConfig {
        time,
        channels,
        resolution,
        units,
    })
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
pub(crate) fn get_graph_mut(arc: &mut Arc<DirGraph>) -> &mut DirGraph {
    let graph = Arc::make_mut(arc);
    graph.version += 1;
    graph
}

/// Lightweight centrality result conversion: returns {title: score} dict.
/// Creates ONE Python dict instead of N dicts — returns {title: score} format.
/// ~3-4x faster PyO3 serialization for large graphs.
pub(crate) fn centrality_results_to_py_dict(
    py: Python<'_>,
    graph: &DirGraph,
    results: Vec<crate::graph::algorithms::graph_algorithms::CentralityResult>,
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
    results: Vec<crate::graph::algorithms::graph_algorithms::CentralityResult>,
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
    result: crate::graph::algorithms::graph_algorithms::CommunityResult,
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
) -> PyResult<crate::graph::core::traversal::MethodConfig> {
    use crate::graph::core::traversal::MethodConfig;

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

    let method_type: String = dict
        .get_item("type")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "method dict must contain 'type' key (e.g. {'type': 'contains'})",
            )
        })?
        .extract()?;

    let resolve = if let Some(v) = dict.get_item("resolve")? {
        let s: String = v.extract()?;
        Some(MethodConfig::parse_resolve(&s).map_err(pyo3::exceptions::PyValueError::new_err)?)
    } else {
        None
    };

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

    Ok(MethodConfig {
        method_type,
        resolve,
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
    })
}

/// Shared comparison traversal logic used by `compare()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_inner(
    inner: &Arc<DirGraph>,
    selection: &mut CowSelection,
    target_type: Option<&str>,
    config: &crate::graph::core::traversal::MethodConfig,
    conditions: Option<&HashMap<String, FilterCondition>>,
    sort_fields: Option<&Vec<(String, bool)>>,
    limit: Option<usize>,
    estimated: usize,
) -> PyResult<usize> {
    crate::graph::core::traversal::make_comparison_traversal(
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
