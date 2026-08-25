use crate::datatypes::{py_in, py_out};
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;
use std::sync::Arc;

use crate::graph::{get_graph_mut, KnowledgeGraph, NodeKeyGuard, NodeKeyKind};
use kglite_core::api::io as file;
use kglite_core::api::GraphRead;

/// One-arg `embeddings(text_column)` keys the *selection* by bare id, and a
/// selection can span node types. The two-arg form is the type-namespaced
/// way to read both stores.
const SELECTION_ID_KEY: NodeKeyGuard<'static> = NodeKeyGuard {
    surface: "embeddings()",
    kind: NodeKeyKind::Id,
    recipe: "call the two-arg form embeddings(node_type, text_column) once \
             per type, which keys a single type's id namespace",
};

#[pymethods]
impl KnowledgeGraph {
    /// Store embeddings for nodes of the given type.
    ///
    /// **Replaces** any existing store for ``(node_type, "{text_column}_emb")``.
    /// For incremental ingest where multiple batches must coexist, use
    /// ``add_embeddings()`` instead (it upserts without clobbering — no
    /// read-merge-write needed at the call site).
    ///
    /// Args:
    ///     node_type: The node type (e.g. 'Article')
    ///     text_column: Source column name (e.g. 'summary'). Stored as '{text_column}_emb'.
    ///     embeddings: Dict mapping node IDs to embedding vectors (list of floats)
    ///
    /// Returns:
    ///     dict: {'embeddings_stored': int, 'dimension': int, 'skipped': int}
    #[pyo3(signature = (node_type, text_column, embeddings, metric=None))]
    fn set_embeddings(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        embeddings: &Bound<'_, PyDict>,
        metric: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let entries = marshal_embedding_batch(embeddings)?;
        let g = get_graph_mut(&mut self.inner);
        let report = kglite_core::api::embeddings::set_embeddings(
            g,
            node_type,
            text_column,
            metric,
            entries,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let result = PyDict::new(py);
        result.set_item("embeddings_stored", report.embeddings_stored)?;
        result.set_item("dimension", report.dimension)?;
        result.set_item("skipped", report.skipped)?;
        Ok(result.into())
    }

    /// Add or update embeddings for nodes of the given type without
    /// discarding the existing store.
    ///
    /// Differs from ``set_embeddings`` (which replaces the store) by
    /// upserting entries into an existing ``(node_type, "{text_column}_emb")``
    /// store. If no store exists yet, behaves like ``set_embeddings`` —
    /// the first call creates one; subsequent calls extend it.
    ///
    /// Use this for incremental ingest workflows where multiple
    /// ``add_nodes`` + embedding batches need to coexist without a
    /// read-merge-write cycle through the user's process.
    ///
    /// Args:
    ///     node_type: The node type (e.g. 'Article')
    ///     text_column: Source column name (e.g. 'summary'). Stored as '{text_column}_emb'.
    ///     embeddings: Dict mapping node IDs to embedding vectors (list of floats).
    ///
    /// Returns:
    ///     dict: {'embeddings_stored': int, 'dimension': int, 'skipped': int, 'store_created': bool}
    #[pyo3(signature = (node_type, text_column, embeddings, metric=None))]
    fn add_embeddings(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        embeddings: &Bound<'_, PyDict>,
        metric: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let entries = marshal_embedding_batch(embeddings)?;
        let g = get_graph_mut(&mut self.inner);
        let report = kglite_core::api::embeddings::add_embeddings(
            g,
            node_type,
            text_column,
            metric,
            entries,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let result = PyDict::new(py);
        result.set_item("embeddings_stored", report.embeddings_stored)?;
        result.set_item("dimension", report.dimension)?;
        result.set_item("skipped", report.skipped)?;
        result.set_item("store_created", report.store_created)?;
        Ok(result.into())
    }

    /// Vector similarity search within the current selection.
    ///
    /// Args:
    ///     text_column: Source column name (e.g. 'summary'). Resolves to '{text_column}_emb'.
    ///     query_vector: The query embedding vector (list of floats)
    ///     top_k: Number of results to return (default 10)
    ///     metric: Distance metric - 'cosine', 'dot_product', 'euclidean', or 'poincare'.
    ///            If omitted, uses the unique metric stored by the selected
    ///            embedding stores, or cosine when none is stored. Selections
    ///            spanning different stored metrics must pass this explicitly.
    ///     to_df: If True, return a pandas DataFrame instead of list of dicts
    ///
    ///     returning: Optional list of fields to project onto each hit. When
    ///            omitted (default), a hit carries ``id``, ``title``, ``type``,
    ///            ``score``, and **all** node properties — so no follow-up join
    ///            is needed to recover them. When given, a hit carries only
    ///            ``id`` + ``score`` plus the named fields (each a property or a
    ///            structural field like ``title``/``type``) — trim the payload
    ///            for ranking-heavy or wide-node workloads.
    ///
    /// Returns:
    ///     List of dicts. By default each has ``id``, ``title``, ``type``,
    ///     ``score``, and all node properties (``score`` always present, every
    ///     metric; properties read live so a hit is identical before/after
    ///     save/reload). With ``returning=[...]`` each has ``id`` + ``score`` +
    ///     the requested fields only.
    ///
    /// Raises:
    ///     ValueError: if **no** selected node type has an embedding store for
    ///         ``text_column`` — a wrong column or an un-embedded type, which
    ///         used to come back as a silent ``[]``. A selection where *some*
    ///         type has the store is a partial result, not an error.
    #[pyo3(signature = (text_column, query_vector, top_k=10, metric=None, to_df=false, returning=None, exact=false))]
    #[allow(clippy::too_many_arguments)]
    fn vector_search(
        &self,
        py: Python<'_>,
        text_column: &str,
        query_vector: Vec<f32>,
        top_k: Option<usize>,
        metric: Option<&str>,
        to_df: Option<bool>,
        returning: Option<Vec<String>>,
        exact: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _arena_guard = self.inner.begin_read_pass(); // disk arena guard (no-op on memory/mapped)
        let top_k = top_k.unwrap_or(10);
        let exact = exact.unwrap_or(false);
        let embedding_property = kglite_core::api::embeddings::store_name(text_column);
        // `id` and `score` are always kept — identity + rank.
        let keep: Option<std::collections::HashSet<String>> =
            returning.map(|v| v.into_iter().collect());
        let want =
            |k: &str| k == "id" || k == "score" || keep.as_ref().is_none_or(|set| set.contains(k));

        let metric = metric
            .map(|name| {
                kglite_core::api::algorithms::DistanceMetric::from_name(name).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown metric '{}'. Use 'cosine', 'dot_product', 'euclidean', or 'poincare'.",
                        name
                    ))
                })
            })
            .transpose()?;
        let inner = self.inner.clone();
        let selection = self.cursor.selection.clone();
        let results = py
            .detach(|| {
                let options = kglite_core::api::algorithms::VectorSearchOptions::default()
                    .with_top_k(top_k)
                    .with_exact(exact);
                let options = match metric {
                    Some(metric) => options.with_metric(metric),
                    None => options.with_stored_metric(),
                };
                kglite_core::api::algorithms::vector_search(
                    &inner,
                    &selection,
                    &embedding_property,
                    &query_vector,
                    &options,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        if to_df.unwrap_or(false) {
            let pandas = py.import("pandas")?;
            let records: Vec<Py<PyAny>> = results
                .iter()
                .filter_map(|r| self.inner.graph.node_view(r.node_idx).map(|node| (r, node)))
                .map(|(r, node)| -> PyResult<Py<PyAny>> {
                    let dict = PyDict::new(py);
                    dict.set_item("id", py_out::value_to_py(py, &node.id())?)?;
                    if want("title") {
                        dict.set_item("title", py_out::value_to_py(py, &node.title())?)?;
                    }
                    if want("type") {
                        dict.set_item("type", node.node_type_str(&self.inner.interner))?;
                    }
                    dict.set_item("score", r.score)?;
                    // properties_cloned reads PropertyStorage::Columnar (the
                    // durable shape); property_iter yields nothing for it.
                    for (k, v) in node.properties_cloned(&self.inner.interner) {
                        if want(&k) {
                            dict.set_item(k, py_out::value_to_py(py, &v)?)?;
                        }
                    }
                    Ok(dict.into())
                })
                .collect::<PyResult<_>>()?;
            let py_list = PyList::new(py, &records)?;
            let df = pandas.call_method1("DataFrame", (py_list,))?;
            return df.into_py_any(py);
        }

        let py_list = PyList::empty(py);
        for r in &results {
            if let Some(node) = self.inner.graph.node_view(r.node_idx) {
                let dict = PyDict::new(py);
                dict.set_item("id", py_out::value_to_py(py, &node.id())?)?;
                if want("title") {
                    dict.set_item("title", py_out::value_to_py(py, &node.title())?)?;
                }
                if want("type") {
                    dict.set_item("type", node.node_type_str(&self.inner.interner))?;
                }
                dict.set_item("score", r.score)?;
                for (k, v) in node.properties_cloned(&self.inner.interner) {
                    if want(&k) {
                        dict.set_item(k, py_out::value_to_py(py, &v)?)?;
                    }
                }
                py_list.append(dict)?;
            }
        }

        py_list.into_py_any(py)
    }

    /// The vector dimension of the `(node_type, text_column)` embedding store,
    /// or ``None`` if no store exists for it.
    ///
    /// A cheap, direct way to detect an embedder/model change without
    /// bookkeeping: compare it against your model's dimension before
    /// `embed_texts`/`add_embeddings` (which reject a mismatch). `text_column`
    /// is the source column name (stored as ``{text_column}_emb``).
    fn embedding_dim(&self, node_type: &str, text_column: &str) -> Option<usize> {
        let key = kglite_core::api::embeddings::store_key(node_type, text_column);
        self.inner.embeddings.get(&key).map(|s| s.dimension)
    }

    /// Provenance for the `(node_type, text_column)` embedding store, or
    /// ``None`` if no store exists.
    ///
    /// Returns a dict with ``dimension``, ``count`` (vectors stored),
    /// ``model`` (the embedder id stamped at `embed_texts` time, or ``None``
    /// for vectors supplied directly), ``metric``, and ``hashed`` (how many
    /// vectors carry a source-text hash for `embed_texts(mode='changed')`
    /// change-detection). Lets a caller detect a model swap or a partially-
    /// hashed store without external bookkeeping.
    fn embedding_info(
        &self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
    ) -> PyResult<Py<PyAny>> {
        let key = kglite_core::api::embeddings::store_key(node_type, text_column);
        match self.inner.embeddings.get(&key) {
            None => Ok(py.None()),
            Some(store) => {
                let d = PyDict::new(py);
                d.set_item("node_type", node_type)?;
                d.set_item("text_column", text_column)?;
                d.set_item("dimension", store.dimension)?;
                d.set_item("count", store.len())?;
                d.set_item("model", store.model_id.clone())?;
                // Report the *effective* metric: a store created by `embed_texts`
                // (or imported pre-provenance) carries no explicit metric, but
                // search falls back to cosine — report what search uses, not `None`.
                d.set_item("metric", store.metric.as_deref().unwrap_or("cosine"))?;
                d.set_item("hashed", store.text_hashes.len())?;
                d.into_py_any(py)
            }
        }
    }

    /// Copy every embedding store from `other` into this graph, matching
    /// vectors by node id.
    ///
    /// The one-call answer to the "rebuild a fresh graph from a source of
    /// truth on each load, keep the vectors" workflow: build the new graph,
    /// then `new.copy_embeddings_from(old)`. Vectors land on the new nodes that
    /// share an id, carrying each store's dimension, metric, model id, and
    /// per-node text hashes — so a following `embed_texts(mode='changed')`
    /// re-embeds only genuinely-new/changed text. Vectors whose id has no
    /// matching node here are skipped (counted). Replaces the manual
    /// `embeddings()` → `add_embeddings()` → `embed_texts()` carry.
    ///
    /// Returns a dict with ``stores_copied``, ``vectors_copied``, and
    /// ``vectors_skipped``.
    fn copy_embeddings_from(
        &mut self,
        py: Python<'_>,
        other: &Bound<'_, KnowledgeGraph>,
    ) -> PyResult<Py<PyAny>> {
        // Mirror extend()'s safe shape: clone the source Arc first (so a
        // self-copy doesn't double-borrow), then mutate self.
        let src_arc = match other.try_borrow() {
            Ok(o) => Arc::clone(&o.inner),
            Err(_) => Arc::clone(&self.inner),
        };
        let g = crate::graph::get_graph_mut(&mut self.inner);
        let (stores, vectors, skipped) = g.copy_embeddings_from(&src_arc);
        let d = PyDict::new(py);
        d.set_item("stores_copied", stores)?;
        d.set_item("vectors_copied", vectors)?;
        d.set_item("vectors_skipped", skipped)?;
        d.into_py_any(py)
    }

    /// List all embedding stores in the graph.
    ///
    /// Returns:
    ///     List of dicts with 'node_type', 'text_column', 'store_name',
    ///     'dimension', 'count', 'metric'. ``text_column`` is what this API
    ///     takes ('summary'); ``store_name`` is what Cypher's ``vector_score``
    ///     takes ('summary_emb').
    fn list_embeddings(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let py_list = PyList::empty(py);
        for info in kglite_core::api::embeddings::list_embeddings(&self.inner) {
            let dict = PyDict::new(py);
            dict.set_item("node_type", info.node_type)?;
            dict.set_item("text_column", info.text_column)?;
            dict.set_item("store_name", info.store_name)?;
            dict.set_item("dimension", info.dimension)?;
            dict.set_item("count", info.count)?;
            dict.set_item("metric", info.metric)?;
            py_list.append(dict)?;
        }
        py_list.into_py_any(py)
    }

    /// Diagnose embedding coverage per (node_type, text_column).
    ///
    /// Surfaces three states the silent-drop case maps to:
    ///
    /// - ``"embedded"``: an embedding store exists and at least one node
    ///   has the underlying property.
    /// - ``"embeddable"``: nodes have a string-typed property but no
    ///   embedding store has been created or restored.
    /// - ``"store_orphan"``: an embedding store exists but no node in
    ///   the current graph has the underlying property — the symptom
    ///   ``import_embeddings()`` warns about when keys mismatch.
    ///
    /// Each row also carries a ``length_stats`` dict so callers can
    /// filter on string-length distribution + cardinality before
    /// committing to embed a column. ISO timestamps, status enums, and
    /// fully-unique identifiers are surfaced with the same status but
    /// distinguishable by their ``length_stats``:
    ///
    /// - ``mean_length`` / ``max_length``: average and max byte length of
    ///   non-null values. Sub-20-byte means usually indicate flags,
    ///   timestamps, or short codes (poor embedding candidates).
    /// - ``distinct_count``: number of unique values seen.
    /// - ``distinct_ratio``: ``distinct_count / value_count``. A ratio
    ///   of 1.0 means every value is unique (likely an identifier).
    ///
    /// Args:
    ///     node_type: Optional. When set, only that node type is scanned.
    ///         When ``None``, every type in the graph is scanned (may be
    ///         expensive on graphs with millions of nodes — pass a type
    ///         to scope the scan).
    ///
    /// Returns:
    ///     List of dicts with: ``node_type``, ``text_column``,
    ///     ``embedding_key`` (= ``f"{text_column}_emb"``),
    ///     ``nodes_with_property``, ``nodes_embedded``,
    ///     ``dimension`` (or ``None``), ``metric`` (or ``None``),
    ///     ``status``, and ``length_stats``.
    #[pyo3(signature = (node_type=None))]
    fn embedding_diagnostics(
        &self,
        py: Python<'_>,
        node_type: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let _arena_guard = self.inner.begin_read_pass(); // disk arena guard (no-op on memory/mapped)
        use crate::datatypes::values::Value;
        use std::collections::HashSet;

        // Validate the filter type up front so unknown types fail loudly
        // instead of silently returning an empty list.
        if let Some(t) = node_type {
            if !self.inner.type_indices.contains_key(t) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Node type '{}' does not exist in the graph",
                    t
                )));
            }
        }

        #[derive(Default)]
        struct Stats<'a> {
            nodes_with_property: usize,
            total_length: usize,
            max_length: usize,
            distinct: HashSet<String>,
            store: Option<&'a kglite_core::api::storage::EmbeddingStore>,
        }
        let mut by_key: std::collections::BTreeMap<(String, String), Stats<'_>> =
            std::collections::BTreeMap::new();

        let types_to_scan: Vec<String> = match node_type {
            Some(t) => vec![t.to_string()],
            None => self.inner.type_indices.keys().map(String::from).collect(),
        };

        // First pass: count string-typed properties per node type. Skips
        // builtin columns (id / title / type) — those are handled below
        // when an embedding store keys against them.
        //
        // Use `properties_cloned()`, not `property_iter()`: the latter yields
        // *nothing* for `PropertyStorage::Columnar` — the durable shape every
        // node's properties land in — which produced `nodes_with_property=0`
        // for every columnarised graph and flipped a healthy steady-state
        // graph's status to `store_orphan`.
        for type_name in &types_to_scan {
            let type_indices = match self.inner.type_indices.get(type_name) {
                Some(ix) => ix,
                None => continue,
            };
            for nidx in type_indices.iter() {
                let node = match self.inner.graph.node_view(nidx) {
                    Some(n) => n,
                    None => continue,
                };
                for (key, value) in node.properties_cloned(&self.inner.interner) {
                    if let Value::String(s) = value {
                        let entry = by_key.entry((type_name.clone(), key)).or_default();
                        let len = s.len();
                        entry.nodes_with_property += 1;
                        entry.total_length += len;
                        if len > entry.max_length {
                            entry.max_length = len;
                        }
                        entry.distinct.insert(s);
                    }
                }
            }
        }

        // Second pass: attach embedding store info, and add entries for
        // stores whose underlying column had no corresponding string
        // property (e.g. builtin columns like `title`, or actual orphans
        // after an import_embeddings silent-drop).
        for ((store_type, store_name), store) in &self.inner.embeddings {
            if let Some(t) = node_type {
                if store_type != t {
                    continue;
                }
            }
            let text_column = store_name
                .strip_suffix("_emb")
                .unwrap_or(store_name.as_str())
                .to_string();
            let entry = by_key
                .entry((store_type.clone(), text_column.clone()))
                .or_default();
            entry.store = Some(store);
            // Treat builtin columns as universally present so we don't
            // mis-flag a `title_emb` store as a store_orphan.
            if matches!(text_column.as_str(), "id" | "title" | "type")
                && entry.nodes_with_property == 0
            {
                if let Some(type_indices) = self.inner.type_indices.get(store_type) {
                    entry.nodes_with_property = type_indices.len();
                }
            }
        }

        let py_list = PyList::empty(py);
        for ((type_name, text_column), stats) in by_key {
            // Drop entries that ended up with no signal at all (no
            // property, no store) — they happen when a non-string slot
            // shows up via the schema scan path.
            if stats.nodes_with_property == 0 && stats.store.is_none() {
                continue;
            }
            let dict = PyDict::new(py);
            dict.set_item("node_type", &type_name)?;
            dict.set_item("text_column", &text_column)?;
            dict.set_item(
                "embedding_key",
                kglite_core::api::embeddings::store_name(&text_column),
            )?;
            dict.set_item("nodes_with_property", stats.nodes_with_property)?;
            let nodes_embedded = stats.store.map(|s| s.len()).unwrap_or(0);
            dict.set_item("nodes_embedded", nodes_embedded)?;
            let status = if stats.store.is_none() {
                "embeddable"
            } else if stats.nodes_with_property == 0 {
                "store_orphan"
            } else {
                "embedded"
            };
            dict.set_item("status", status)?;
            match stats.store {
                Some(s) => {
                    dict.set_item("dimension", s.dimension)?;
                    dict.set_item(
                        "metric",
                        s.metric.clone().unwrap_or_else(|| "cosine".to_string()),
                    )?;
                }
                None => {
                    dict.set_item("dimension", py.None())?;
                    dict.set_item("metric", py.None())?;
                }
            }

            let length_stats = PyDict::new(py);
            let distinct_count = stats.distinct.len();
            let mean_length = if stats.nodes_with_property > 0 {
                stats.total_length as f64 / stats.nodes_with_property as f64
            } else {
                0.0
            };
            let distinct_ratio = if stats.nodes_with_property > 0 {
                distinct_count as f64 / stats.nodes_with_property as f64
            } else {
                0.0
            };
            length_stats.set_item("mean_length", mean_length)?;
            length_stats.set_item("max_length", stats.max_length)?;
            length_stats.set_item("distinct_count", distinct_count)?;
            length_stats.set_item("distinct_ratio", distinct_ratio)?;
            dict.set_item("length_stats", length_stats)?;

            py_list.append(dict)?;
        }

        py_list.into_py_any(py)
    }

    /// Remove an embedding store.
    ///
    /// Args:
    ///     node_type: The node type
    ///     text_column: Source column name (e.g. 'summary')
    fn remove_embeddings(&mut self, node_type: &str, text_column: &str) -> PyResult<()> {
        let g = get_graph_mut(&mut self.inner);
        let key = kglite_core::api::embeddings::store_key(node_type, text_column);
        g.embeddings.remove(&key);
        Ok(())
    }

    /// Export embeddings to a standalone .kgle file.
    ///
    /// Exported embeddings are keyed by node ID, so they survive graph rebuilds.
    ///
    /// Args:
    ///     path: File path to write (typically ending in .kgle)
    ///     node_types: Optional filter. Can be:
    ///         - None: export all embeddings
    ///         - list[str]: export all embedding stores for these node types
    ///         - dict[str, list[str]]: export specific (node_type -> [text_columns]) pairs.
    ///           An empty list means all properties for that type.
    ///
    /// Returns:
    ///     Dict with 'stores' (int) and 'embeddings' (int) counts.
    #[pyo3(signature = (path, node_types=None))]
    fn export_embeddings(
        &self,
        py: Python<'_>,
        path: &str,
        node_types: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let filter = match &node_types {
            None => None,
            Some(obj) => {
                if let Ok(list) = obj.cast::<PyList>() {
                    let types: Vec<String> = list.extract()?;
                    Some(file::EmbeddingExportFilter::Types(types))
                } else if let Ok(dict) = obj.cast::<PyDict>() {
                    let mut map: HashMap<String, Vec<String>> = HashMap::new();
                    for (k, v) in dict.iter() {
                        let key: String = k.extract()?;
                        let vals: Vec<String> = v.extract()?;
                        map.insert(key, vals);
                    }
                    Some(file::EmbeddingExportFilter::TypeProperties(map))
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "node_types must be a list of strings or a dict of {str: list[str]}",
                    ));
                }
            }
        };

        let inner = self.inner.clone();
        let path_owned = path.to_string();
        let stats = py
            .detach(move || file::export_embeddings_to_file(&inner, &path_owned, filter.as_ref()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let result = PyDict::new(py);
        result.set_item("stores", stats.stores)?;
        result.set_item("embeddings", stats.embeddings)?;
        result.into_py_any(py)
    }

    /// Import embeddings from a .kgle file.
    ///
    /// Matches embeddings to nodes by (node_type, node_id). Embeddings whose
    /// node ID doesn't exist in the current graph are skipped. When all
    /// embeddings (or all stores) are skipped — a strong signal that the
    /// .kgle file was exported from a graph with different IDs or types —
    /// a ``UserWarning`` is emitted so the silent-drop case becomes visible.
    ///
    /// Args:
    ///     path: Path to a .kgle file previously created by export_embeddings.
    ///
    /// Returns:
    ///     Dict with 'stores' (int), 'imported' (int), 'skipped' (int), and
    ///     'dropped_stores' (int) counts. ``dropped_stores`` is the number
    ///     of per-type stores that contained entries but had zero matches.
    fn import_embeddings(&mut self, py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
        let g = get_graph_mut(&mut self.inner);
        let stats = file::import_embeddings_from_file(g, path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        // Surface the silent-drop cases as a UserWarning: visible by default,
        // still suppressible via the standard `warnings` module.
        if stats.imported == 0 && stats.skipped > 0 {
            let msg = format!(
                "import_embeddings('{}'): imported 0 embeddings, skipped {} — \
                 no node IDs in the file match the current graph. The file \
                 may have been exported from a different graph, or the node \
                 ID/type schema has changed since export.",
                path, stats.skipped
            );
            let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
            let _ = PyErr::warn(
                py,
                py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                cmsg.as_c_str(),
                1,
            );
        } else if stats.dropped_stores > 0 {
            let msg = format!(
                "import_embeddings('{}'): {} embedding store(s) had zero \
                 matches and were dropped (imported={}, skipped={}, \
                 stores_kept={}). Some types in the file don't exist in \
                 the current graph, or their node IDs don't match.",
                path, stats.dropped_stores, stats.imported, stats.skipped, stats.stores
            );
            let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
            let _ = PyErr::warn(
                py,
                py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                cmsg.as_c_str(),
                1,
            );
        }

        let result = PyDict::new(py);
        result.set_item("stores", stats.stores)?;
        result.set_item("imported", stats.imported)?;
        result.set_item("skipped", stats.skipped)?;
        result.set_item("dropped_stores", stats.dropped_stores)?;
        result.into_py_any(py)
    }

    /// Retrieve embeddings for nodes.
    ///
    /// Can be called in two ways:
    ///   - ``embeddings(node_type, text_column)`` — returns all embeddings of that type
    ///   - ``embeddings(text_column)`` — returns embeddings for the current selection
    ///
    /// Args:
    ///     text_column: Source column name (e.g. 'summary'). Resolves to '{text_column}_emb'.
    ///
    /// Returns:
    ///     Dict mapping node IDs to embedding vectors (list of floats).
    ///
    /// Raises:
    ///     ArgumentError: The one-arg form's selection spans two node types
    ///         sharing an id. Ids are unique per type only; call the two-arg
    ///         form once per type instead.
    #[pyo3(signature = (node_type_or_text_column, text_column=None))]
    fn embeddings(
        &self,
        py: Python<'_>,
        node_type_or_text_column: &str,
        text_column: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let _arena_guard = self.inner.begin_read_pass(); // disk arena guard (no-op on memory/mapped)
        let result = PyDict::new(py);

        if let Some(col) = text_column {
            let key = kglite_core::api::embeddings::store_key(node_type_or_text_column, col);
            let store = match self.inner.embeddings.get(&key) {
                Some(s) => s,
                None => return result.into_py_any(py),
            };

            for (&node_index, &_slot) in &store.node_to_slot {
                if let Some(embedding) = store.get_embedding(node_index) {
                    if let Some(node) = self.inner.graph.node_view(NodeIndex::new(node_index)) {
                        let py_id = py_out::value_to_py(py, &node.id())?;
                        let py_vec = PyList::new(py, embedding)?;
                        result.set_item(py_id, py_vec)?;
                    }
                }
            }

            return result.into_py_any(py);
        }

        // One-arg form: embeddings(text_column) — selection-based. A selection
        // that was never narrowed means the whole graph (the never-selected
        // rule get_nodes() applies); one a query emptied stays empty.
        let col = node_type_or_text_column;

        let selection = &self.cursor.selection;
        let level = selection
            .get_level(selection.get_level_count().saturating_sub(1))
            .filter(|level| level.node_count() > 0);
        let nodes: Vec<NodeIndex> = match level {
            Some(level) => level.get_all_nodes(),
            None if selection.never_selected() => {
                GraphRead::node_indices(&self.inner.graph).collect()
            }
            None => Vec::new(),
        };

        for node_idx in &nodes {
            let node = match self.inner.graph.node_view(*node_idx) {
                Some(n) => n,
                None => continue,
            };

            let key = kglite_core::api::embeddings::store_key(
                node.node_type_str(&self.inner.interner),
                col,
            );
            let store = match self.inner.embeddings.get(&key) {
                Some(s) => s,
                None => continue,
            };

            if let Some(embedding) = store.get_embedding(node_idx.index()) {
                let py_id = py_out::value_to_py(py, &node.id())?;
                let py_vec = PyList::new(py, embedding)?;
                // The selection can span types, and ids are only unique
                // within one — two colliding nodes would silently leave one
                // vector out of the dict.
                SELECTION_ID_KEY.insert(&result, py_id.bind(py), py_vec)?;
            }
        }

        result.into_py_any(py)
    }

    /// Retrieve a single node's embedding vector.
    ///
    /// Args:
    ///     node_type: The node type (e.g. 'Article').
    ///     text_column: Source column name (e.g. 'summary').
    ///     node_id: The node ID to look up.
    ///
    /// Returns:
    ///     The embedding vector as a list of floats, or None if not found.
    fn embedding(
        &self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        node_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let id = py_in::py_value_to_value(node_id)?;

        let node_idx = match self.inner.lookup_by_id_readonly(node_type, &id) {
            Some(idx) => idx,
            None => return Ok(py.None()),
        };

        let key = kglite_core::api::embeddings::store_key(node_type, text_column);
        let store = match self.inner.embeddings.get(&key) {
            Some(s) => s,
            None => return Ok(py.None()),
        };

        match store.get_embedding(node_idx.index()) {
            Some(embedding) => {
                let py_vec = PyList::new(py, embedding)?;
                py_vec.into_py_any(py)
            }
            None => Ok(py.None()),
        }
    }

    /// Register or unbind an embedding model on the graph.
    ///
    /// Pass a model object to register; pass ``None`` to unbind the
    /// currently-registered embedder.
    ///
    /// The model must have:
    /// - ``dimension: int`` — the embedding vector size
    /// - ``embed(texts: list[str]) -> list[list[float]]`` — batch embedding method
    ///
    /// After registering, ``embed_texts()`` and ``search_text()`` use the
    /// registered model automatically.  The model is **not** serialized —
    /// call ``set_embedder()`` again after ``load()``.
    #[pyo3(signature = (model,))]
    fn set_embedder(&mut self, py: Python<'_>, model: Option<Py<PyAny>>) -> PyResult<()> {
        let Some(model) = model else {
            self.embedder = None;
            return Ok(());
        };
        let bound = model.bind(py);
        bound.getattr("dimension").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                "model must have a 'dimension' attribute (int)",
            )
        })?;
        bound.getattr("embed").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyAttributeError, _>("model must have an 'embed' method")
        })?;
        let adapter = crate::graph::embedder::py_adapter::PyEmbedderAdapter::new(py, model)?;
        self.embedder = Some(Arc::new(adapter));
        Ok(())
    }

    /// Embed a text column for all nodes of a given type.
    ///
    /// Uses the model registered via ``set_embedder()``.  Reads each node's
    /// ``text_column`` property, calls ``model.embed()`` in batches, and stores
    /// the resulting vectors as ``{text_column}_emb``.  Nodes with missing or
    /// non-string text values are skipped.
    ///
    /// Args:
    ///     node_type: The node type to embed (e.g. ``'Article'``).
    ///     text_column: The column holding the text to embed. Resolves as
    ///         ``set_embeddings`` resolves it — a stored property, an identity
    ///         alias (a ``title_field='name'`` type embeds its titles under
    ///         ``'name'``), the canonical ``id``/``title``, or a structural
    ///         alias. A column that resolves to none of those raises.
    ///     batch_size: Number of texts per ``model.embed()`` call (default 256).
    ///     show_progress: Show a tqdm progress bar (default ``True``).
    ///         Requires ``tqdm`` to be installed; silently falls back to no
    ///         progress bar if it is not available.
    ///     mode: Which nodes to embed —
    ///         ``'missing'`` (default): only nodes without an embedding yet;
    ///         ``'changed'``: nodes missing an embedding *or* whose text changed
    ///         since the last embed (detected via a stored per-node content
    ///         hash) — the incremental re-embed;
    ///         ``'all'``: re-embed every node, rebuilding the store fresh.
    ///
    /// Returns:
    ///     Dict with ``embedded``, ``skipped``, ``skipped_existing``,
    ///     ``reembedded_changed``, and ``dimension``.
    ///
    /// Raises:
    ///     ValueError: if ``node_type`` does not exist in the graph (the same
    ///         complaint ``set_embeddings`` makes — raised before the model is
    ///         loaded), if ``text_column`` resolves to no readable column, or
    ///         if ``mode`` is not one of the three names.
    #[pyo3(signature = (node_type, text_column, batch_size=256, show_progress=true, mode=None))]
    fn embed_texts(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        batch_size: Option<usize>,
        show_progress: Option<bool>,
        mode: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let _arena_guard = self.inner.begin_read_pass(); // disk arena guard (no-op on memory/mapped)
        let model = self.get_embedder_or_error()?;
        let embedding_property = kglite_core::api::embeddings::store_name(text_column);
        let batch_size = batch_size.unwrap_or(256);
        let mode = match mode.unwrap_or("missing") {
            "missing" => "missing",
            "changed" => "changed",
            "all" => "all",
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "embed_texts(mode={other:?}): unknown mode. Use 'missing' (default), \
                     'changed' (re-embed nodes whose text changed), or 'all'."
                )));
            }
        };
        let rebuild_store = mode == "all";

        // Resolve the source column through the *same* predicate the ingest
        // guard uses (`set_embeddings`' `resolve_source_column`), and read the
        // text through the field it returns. Reading `get_property` directly
        // was the bug: it excludes `id`/`title` by contract, so `('Person',
        // 'name')` on a `title_field='name'` type embedded nothing and
        // reported it as `skipped`.
        // A type with no nodes stays the `{'embedded': 0}` no-op it has always
        // been; a type the graph has never *seen* is a mistake and gets
        // `set_embeddings`' complaint, before the model is loaded.
        if !self.inner.type_indices.contains_key(node_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node type '{}' does not exist in the graph",
                node_type
            )));
        }
        let node_indices: Vec<NodeIndex> = self
            .inner
            .type_indices
            .get(node_type)
            .map(|v| v.to_vec())
            .unwrap_or_default();
        let source_field = if node_indices.is_empty() {
            text_column.to_string()
        } else {
            kglite_core::api::embeddings::resolve_source_column(&self.inner, node_type, text_column)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?
                .to_string()
        };
        let source_key = kglite_core::api::InternedKey::from_str(&source_field);

        model
            .load()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        let dimension: usize = model.dimension();

        let emb_key = (node_type.to_string(), embedding_property.clone());
        let existing_store = if rebuild_store {
            None
        } else {
            self.inner.embeddings.get(&emb_key)
        };

        // Reject an incremental embed at a dimension the store doesn't hold:
        // mixing dimensions silently corrupts similarity search.
        if let Some(s) = existing_store {
            if s.dimension != dimension {
                model.unload();
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "embed_texts(): the model produces {dimension}-d vectors but the existing \
                     '{node_type}.{text_column}_emb' store is {}-d — embedding the rest would mix \
                     dimensions and corrupt search. Re-embed the whole column with mode='all' to \
                     rebuild at the new dimension, or remove_embeddings('{node_type}', '{text_column}') first.",
                    s.dimension
                )));
            }
        }

        let candidates = collect_embed_candidates(
            &self.inner.graph,
            &node_indices,
            node_type,
            &source_field,
            source_key,
            existing_store,
            mode == "changed",
        );

        if candidates.texts.is_empty() {
            model.unload();
            return candidates.report(py, 0, dimension);
        }

        let mut store = match existing_store {
            Some(s) => s.clone(),
            None => kglite_core::api::storage::EmbeddingStore::new(dimension),
        };
        store.data.reserve(candidates.texts.len() * dimension);

        let progress_bar = open_progress_bar(
            py,
            show_progress.unwrap_or(true),
            candidates.texts.len(),
            format!("Embedding {}.{}", node_type, text_column),
        );
        let outcome = embed_in_batches(
            py,
            model.as_ref(),
            &mut store,
            &candidates.texts,
            batch_size,
            dimension,
            progress_bar.as_ref(),
        );

        if let Some(ref bar) = progress_bar {
            let _ = bar.call_method0("close");
        }

        model.unload();
        outcome?;

        // Stamp the model identity onto the store (provenance) when the
        // embedder names its model — leaves a prior id intact otherwise.
        if let Some(mid) = model.model_id() {
            store.model_id = Some(mid);
        }

        let embedded = candidates.texts.len();
        let g = get_graph_mut(&mut self.inner);
        g.embeddings.insert(emb_key, store);
        candidates.report(py, embedded, dimension)
    }

    /// Search embeddings using a text query.
    ///
    /// Uses the model registered via ``set_embedder()`` to embed the query,
    /// then performs vector search within the current selection.  The user
    /// refers to the text column name (e.g. ``"summary"``); the graph
    /// resolves it to ``"summary_emb"`` internally.
    ///
    /// Args:
    ///     text_column: Text column whose embeddings to search (e.g. ``'summary'``).
    ///     query: The text query to search for.
    ///     top_k: Number of results to return (default 10).
    ///     metric: Distance metric. Omitted uses the same selection-aware stored
    ///         metric resolution as ``vector_search``.
    ///     to_df: If True, return a pandas DataFrame.
    ///
    /// Returns:
    ///     Same format as ``vector_search()`` — list of dicts or DataFrame.
    #[pyo3(signature = (text_column, query, top_k=10, metric=None, to_df=false, returning=None, exact=false))]
    #[allow(clippy::too_many_arguments)]
    fn search_text(
        &self,
        py: Python<'_>,
        text_column: &str,
        query: &str,
        top_k: Option<usize>,
        metric: Option<&str>,
        to_df: Option<bool>,
        returning: Option<Vec<String>>,
        exact: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let model = self.get_embedder_or_error()?;

        model
            .load()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        // Unload regardless of success or failure — hence the `?` after it.
        let texts = vec![query.to_string()];
        let embed_result = py.detach(|| model.embed(&texts));
        model.unload();
        let embeddings = embed_result.map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        if embeddings.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "model.embed() returned an empty list",
            ));
        }

        let query_vector = embeddings.into_iter().next().unwrap();

        self.vector_search(
            py,
            text_column,
            query_vector,
            top_k,
            metric,
            to_df,
            returning,
            exact,
        )
    }

    /// Build an HNSW approximate-nearest-neighbour index over an embedding store
    /// so subsequent vector searches scale sub-linearly on large stores.
    ///
    /// Opt-in (like ``create_index``): without it, search is an exact brute-force
    /// scan. Once built, ``vector_search`` / ``search_text`` auto-use the index
    /// for queries covering most of a large store; pass ``exact=True`` to force
    /// an exact scan.
    ///
    /// Later vector writes (``add_embeddings`` / ``embed_texts`` /
    /// ``set_embeddings``) do **not** drop the index: they are recorded, and the
    /// next vector query folds them in — while the outstanding delta stays at or
    /// under ``auto_refresh_limit``. A larger delta is served by the exact scan,
    /// which is correct and slower, until you rebuild. Catch-up only ever
    /// indexes vectors that exist; a node with no embedding is reported by
    /// ``SHOW INDEXES`` as ``unembedded`` and is never embedded by a query.
    /// Deleting an embedded node, ``compact()`` and a rolled-back delete still
    /// drop the index outright — each of them moves the slot layout the index
    /// addresses — so rebuild after those.
    ///
    /// The selection does **not** have to be that one node type: as long as
    /// only one type carries ``text_column``, a whole-graph search (or any
    /// selection spanning other types) still uses the index. When two or more
    /// types carry the same column, only a selection of a single one of them
    /// does — a selection spanning both is ranked by exact scan so neither
    /// type's rows can be dropped.
    ///
    /// Args:
    ///     node_type: The node type (e.g. ``'Article'``).
    ///     text_column: Source column name (e.g. ``'summary'``; the store is
    ///         ``'{text_column}_emb'``).
    ///     m: Max neighbours per node on upper layers (default 16). Higher →
    ///         better recall + larger index.
    ///     ef_construction: Build-time search width (default 200). Higher →
    ///         better graph, slower build.
    ///     ef_search: Default query-time search width (default 64). Higher →
    ///         better recall, slower query.
    ///     metric: Distance metric to index for — ``'cosine'`` (default),
    ///         ``'dot_product'``, or ``'euclidean'``. ``'poincare'`` is not
    ///         supported (it stays on the exact path). If omitted, uses the
    ///         store's metric, else ``'cosine'``.
    ///     auto_refresh_limit: How many outstanding vectors a query will fold
    ///         into the index inline before it serves the exact scan instead
    ///         (default 1000). Omit on a rebuild to keep the current value.
    ///
    /// Returns:
    ///     dict: ``{'indexed': int, 'metric': str, 'm': int}`` — vectors indexed.
    ///
    /// Raises:
    ///     ValueError: if the store doesn't exist or the metric is unsupported.
    #[pyo3(signature = (node_type, text_column, m=None, ef_construction=None, ef_search=None, metric=None, auto_refresh_limit=None))]
    #[allow(clippy::too_many_arguments)]
    fn build_vector_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        metric: Option<&str>,
        auto_refresh_limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let g = get_graph_mut(&mut self.inner);
        // Build off the GIL — pure CPU over the contiguous vector buffer.
        let report = py
            .detach(|| {
                kglite_core::api::embeddings::build_vector_index(
                    g,
                    node_type,
                    text_column,
                    m,
                    ef_construction,
                    ef_search,
                    metric,
                    auto_refresh_limit,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let result = PyDict::new(py);
        result.set_item("indexed", report.indexed)?;
        result.set_item("metric", report.metric)?;
        result.set_item("m", report.m)?;
        Ok(result.into())
    }

    /// Drop the HNSW index for an embedding store (search reverts to exact
    /// brute-force). The vectors are untouched. No-op if no index exists.
    /// Returns ``True`` if one was dropped.
    #[pyo3(signature = (node_type, text_column))]
    fn drop_vector_index(&mut self, node_type: &str, text_column: &str) -> PyResult<bool> {
        Ok(kglite_core::api::embeddings::drop_vector_index(
            get_graph_mut(&mut self.inner),
            node_type,
            text_column,
        ))
    }

    /// Whether an HNSW index is currently built over an embedding store.
    #[pyo3(signature = (node_type, text_column))]
    fn has_vector_index(&self, node_type: &str, text_column: &str) -> PyResult<bool> {
        Ok(kglite_core::api::embeddings::has_vector_index(
            &self.inner,
            node_type,
            text_column,
        ))
    }

    /// Fold every outstanding vector into the HNSW index now, instead of
    /// waiting for a query to do it.
    ///
    /// Returns the number of vectors folded in — ``0`` when the index is
    /// already current, when none is built (catch-up never builds one), or on
    /// a read-only graph. Queries do this on their own while the delta stays
    /// under ``auto_refresh_limit``; call it explicitly to pay the cost at a
    /// moment of your choosing, or to bring an over-limit delta back in one
    /// step without a rebuild.
    #[pyo3(signature = (node_type, text_column))]
    fn refresh_vector_index(&self, node_type: &str, text_column: &str) -> PyResult<usize> {
        Ok(
            kglite_core::api::embeddings::refresh_vector_index(&self.inner, node_type, text_column)
                .unwrap_or(0),
        )
    }
}

/// Marshal a `{id: [floats]}` dict into the `(id, vector)` pairs the engine
/// primitive consumes. Purely a boundary conversion — every validation rule
/// (node type, source column, id resolution, dimension) lives in
/// `kglite::api::embeddings`.
fn marshal_embedding_batch(
    embeddings: &Bound<'_, PyDict>,
) -> PyResult<Vec<(kglite_core::api::Value, Vec<f32>)>> {
    let mut entries = Vec::with_capacity(embeddings.len());
    for (key, value) in embeddings.iter() {
        entries.push((
            py_in::py_value_to_value(&key)?,
            value.extract::<Vec<f32>>()?,
        ));
    }
    Ok(entries)
}

/// What one `embed_texts` pass decided about a node type's nodes: the texts
/// to send to the model, plus the three counters its report returns.
struct EmbedCandidates {
    /// `(node_index, text, text_hash)` for every node that needs embedding.
    texts: Vec<(NodeIndex, String, u64)>,
    /// Nodes whose source field held no non-empty string.
    skipped: usize,
    /// Nodes left alone because their embedding is already current.
    skipped_existing: usize,
    /// Nodes that *had* an embedding and are being re-embedded ('changed').
    reembedded_changed: usize,
}

impl EmbedCandidates {
    /// `embed_texts`' return dict — one mint for the nothing-to-do early
    /// return (`embedded = 0`) and for the completed pass alike.
    fn report(&self, py: Python<'_>, embedded: usize, dimension: usize) -> PyResult<Py<PyAny>> {
        let result = PyDict::new(py);
        result.set_item("embedded", embedded)?;
        result.set_item("skipped", self.skipped)?;
        result.set_item("skipped_existing", self.skipped_existing)?;
        result.set_item("reembedded_changed", self.reembedded_changed)?;
        result.set_item("dimension", dimension)?;
        Ok(result.into())
    }
}

/// Split a node type's nodes into "needs embedding" and the skip counters.
///
/// Each node's text is read through the alias-resolved matcher field, so
/// `source_field`/`source_key` must be what `resolve_source_column` returned —
/// the same predicate the ingest guard applies.
///
/// `changed_mode` selects nodes that are missing an embedding *or* whose
/// stored text hash is stale; otherwise a node that already has a vector in
/// `existing_store` is left alone. `mode='all'` passes `existing_store =
/// None`, which makes every node carrying text a candidate.
fn collect_embed_candidates(
    graph: &impl GraphRead,
    node_indices: &[NodeIndex],
    node_type: &str,
    source_field: &str,
    source_key: kglite_core::api::InternedKey,
    existing_store: Option<&kglite_core::api::storage::EmbeddingStore>,
    changed_mode: bool,
) -> EmbedCandidates {
    let mut found = EmbedCandidates {
        texts: Vec::new(),
        skipped: 0,
        skipped_existing: 0,
        reembedded_changed: 0,
    };
    for &node_idx in node_indices {
        let Some(node) = graph.node_view(node_idx) else {
            continue;
        };
        match node
            .resolved_field(node_type, source_field, source_key)
            .as_deref()
        {
            Some(crate::datatypes::values::Value::String(s)) if !s.is_empty() => {
                let hash = kglite_core::api::storage::EmbeddingStore::text_hash(s);
                let has_emb = existing_store
                    .map(|st| st.get_embedding(node_idx.index()).is_some())
                    .unwrap_or(false);
                if changed_mode {
                    let stale = existing_store
                        .map(|st| st.is_stale(node_idx.index(), hash))
                        .unwrap_or(true);
                    if stale {
                        if has_emb {
                            found.reembedded_changed += 1;
                        }
                        found.texts.push((node_idx, s.clone(), hash));
                    } else {
                        found.skipped_existing += 1;
                    }
                } else if has_emb {
                    found.skipped_existing += 1;
                } else {
                    found.texts.push((node_idx, s.clone(), hash));
                }
            }
            _ => {
                found.skipped += 1;
            }
        }
    }
    found
}

/// The optional tqdm bar `embed_texts` drives. `None` when the caller opted
/// out, and `None` rather than an error when tqdm is not installed — the
/// documented silent fallback.
fn open_progress_bar<'py>(
    py: Python<'py>,
    show_progress: bool,
    total: usize,
    desc: String,
) -> Option<Bound<'py, PyAny>> {
    if !show_progress {
        return None;
    }
    py.import("tqdm.auto")
        .or_else(|_| py.import("tqdm"))
        .ok()
        .and_then(|tqdm_mod| {
            let kwargs = PyDict::new(py);
            let _ = kwargs.set_item("total", total);
            let _ = kwargs.set_item("desc", desc);
            let _ = kwargs.set_item("unit", "text");
            tqdm_mod.call_method("tqdm", (), Some(&kwargs)).ok()
        })
}

/// Embed the selected texts in `batch_size` chunks, writing each vector and
/// its text hash into `store` and ticking the progress bar per batch.
///
/// Teardown on failure belongs to the caller: it closes the bar and unloads
/// the model on every path, so the three error exits here just return.
fn embed_in_batches(
    py: Python<'_>,
    model: &dyn kglite_core::api::Embedder,
    store: &mut kglite_core::api::storage::EmbeddingStore,
    node_texts: &[(NodeIndex, String, u64)],
    batch_size: usize,
    dimension: usize,
    progress_bar: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    for batch in node_texts.chunks(batch_size) {
        let texts: Vec<String> = batch.iter().map(|(_, t, _)| t.clone()).collect();

        // Release the GIL while embedding — PyEmbedderAdapter
        // reacquires inside, fastembed never needs it.
        let embeddings = py
            .detach(|| model.embed(&texts))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        if embeddings.len() != batch.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "model.embed() returned {} vectors for {} texts",
                embeddings.len(),
                batch.len()
            )));
        }

        for (i, vec) in embeddings.iter().enumerate() {
            if vec.len() != dimension {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "model.embed() returned vector of dimension {} (expected {})",
                    vec.len(),
                    dimension
                )));
            }
            store.set_embedding(batch[i].0.index(), vec);
            store.set_text_hash(batch[i].0.index(), batch[i].2);
        }

        if let Some(bar) = progress_bar {
            let _ = bar.call_method1("update", (batch.len(),));
        }
    }
    Ok(())
}
