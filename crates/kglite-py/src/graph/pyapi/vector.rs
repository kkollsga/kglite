// Embedding / Vector Search #[pymethods] — extracted from mod.rs

use crate::datatypes::{py_in, py_out};
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;
use std::sync::Arc;

use crate::graph::KnowledgeGraph;
use kglite_core::api::io as file;
use kglite_core::api::GraphRead;

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Embedding / Vector Search Methods
    // ========================================================================

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
        let g = Arc::make_mut(&mut self.inner);
        let embedding_property = format!("{}_emb", text_column);

        // Validate node type exists
        if !g.type_indices.contains_key(node_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node type '{}' does not exist in the graph",
                node_type
            )));
        }

        // Validate source column exists (skip for empty dicts)
        if !embeddings.is_empty() {
            let is_builtin = matches!(text_column, "id" | "title" | "type");
            if !is_builtin {
                let has_property = g
                    .type_indices
                    .get(node_type)
                    .map(|indices| {
                        indices.iter().any(|idx| {
                            g.graph
                                .node_weight(idx)
                                .map(|n| n.has_property(text_column))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
                if !has_property {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Source column '{}' not found on any '{}' node. \
                         set_embeddings() expects the text column name \
                         (e.g. 'summary'), not the embedding store name.",
                        text_column, node_type
                    )));
                }
            }
        }

        // Build ID index for this node type if not already built
        g.build_id_index(node_type);

        let mut dimension: Option<usize> = None;
        let mut entries: Vec<(NodeIndex, Vec<f32>)> = Vec::new();
        let mut skipped = 0usize;

        for (key, value) in embeddings.iter() {
            // Convert key to Value for ID lookup
            let id = py_in::py_value_to_value(&key)?;

            // Look up node by ID
            let node_idx = match g.lookup_by_id(node_type, &id) {
                Some(idx) => idx,
                None => {
                    skipped += 1;
                    continue;
                }
            };

            // Convert embedding to Vec<f32>
            let vec: Vec<f32> = value.extract()?;

            // Validate/set dimension
            match dimension {
                None => dimension = Some(vec.len()),
                Some(d) => {
                    if vec.len() != d {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Inconsistent embedding dimensions: expected {} but got {}",
                            d,
                            vec.len()
                        )));
                    }
                }
            }

            entries.push((node_idx, vec));
        }

        let dim = match dimension {
            Some(d) => d,
            None => {
                let result = PyDict::new(py);
                result.set_item("embeddings_stored", 0)?;
                result.set_item("dimension", 0)?;
                result.set_item("skipped", skipped)?;
                return Ok(result.into());
            }
        };

        // Create or replace the EmbeddingStore
        let mut store = match metric {
            Some(m) => kglite_core::api::storage::EmbeddingStore::with_metric(dim, m),
            None => kglite_core::api::storage::EmbeddingStore::new(dim),
        };
        store.data.reserve(entries.len() * dim);
        for (node_idx, vec) in &entries {
            store.set_embedding(node_idx.index(), vec);
        }

        let stored = store.len();
        g.embeddings
            .insert((node_type.to_string(), embedding_property), store);

        let result = PyDict::new(py);
        result.set_item("embeddings_stored", stored)?;
        result.set_item("dimension", dim)?;
        result.set_item("skipped", skipped)?;
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
        let g = Arc::make_mut(&mut self.inner);
        let embedding_property = format!("{}_emb", text_column);

        if !g.type_indices.contains_key(node_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node type '{}' does not exist in the graph",
                node_type
            )));
        }

        g.build_id_index(node_type);

        let store_key = (node_type.to_string(), embedding_property);
        let store_existed = g.embeddings.contains_key(&store_key);

        // Snapshot the existing dimension/metric, if any, so we can
        // validate incoming vectors against the live store's shape.
        let existing_dim = g.embeddings.get(&store_key).map(|s| s.dimension);

        let mut entries: Vec<(NodeIndex, Vec<f32>)> = Vec::new();
        let mut skipped = 0usize;
        let mut dim_seen: Option<usize> = existing_dim;

        for (key, value) in embeddings.iter() {
            let id = py_in::py_value_to_value(&key)?;
            let node_idx = match g.lookup_by_id(node_type, &id) {
                Some(idx) => idx,
                None => {
                    skipped += 1;
                    continue;
                }
            };
            let vec: Vec<f32> = value.extract()?;
            match dim_seen {
                None => dim_seen = Some(vec.len()),
                Some(d) => {
                    if vec.len() != d {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Inconsistent embedding dimension: store has {} but got {}",
                            d,
                            vec.len()
                        )));
                    }
                }
            }
            entries.push((node_idx, vec));
        }

        let dim = match dim_seen {
            Some(d) => d,
            None => {
                let result = PyDict::new(py);
                result.set_item("embeddings_stored", 0)?;
                result.set_item("dimension", 0)?;
                result.set_item("skipped", skipped)?;
                result.set_item("store_created", false)?;
                return Ok(result.into());
            }
        };

        let store = g
            .embeddings
            .entry(store_key.clone())
            .or_insert_with(|| match metric {
                Some(m) => kglite_core::api::storage::EmbeddingStore::with_metric(dim, m),
                None => kglite_core::api::storage::EmbeddingStore::new(dim),
            });
        for (node_idx, vec) in &entries {
            store.set_embedding(node_idx.index(), vec);
        }

        let result = PyDict::new(py);
        result.set_item("embeddings_stored", store.len())?;
        result.set_item("dimension", dim)?;
        result.set_item("skipped", skipped)?;
        result.set_item("store_created", !store_existed)?;
        Ok(result.into())
    }

    /// Vector similarity search within the current selection.
    ///
    /// Args:
    ///     text_column: Source column name (e.g. 'summary'). Resolves to '{text_column}_emb'.
    ///     query_vector: The query embedding vector (list of floats)
    ///     top_k: Number of results to return (default 10)
    ///     metric: Distance metric - 'cosine' (default), 'dot_product', 'euclidean', or 'poincare'.
    ///            If omitted, uses the metric stored with set_embeddings(), or 'cosine'.
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
    #[pyo3(signature = (text_column, query_vector, top_k=None, metric=None, to_df=None, returning=None, exact=None))]
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
        let top_k = top_k.unwrap_or(10);
        let exact = exact.unwrap_or(false);
        let embedding_property = format!("{}_emb", text_column);
        // Projection set: None = include everything (default). Some = include
        // only these fields; `id` and `score` are always kept (identity + rank).
        let keep: Option<std::collections::HashSet<String>> =
            returning.map(|v| v.into_iter().collect());
        let want =
            |k: &str| k == "id" || k == "score" || keep.as_ref().is_none_or(|set| set.contains(k));

        // Resolve metric: explicit arg > stored metric > cosine default
        let effective_metric = match metric {
            Some(m) => m.to_string(),
            None => {
                // Look up stored metric from any embedding store matching this property
                self.inner
                    .embeddings
                    .iter()
                    .find(|((_, pn), _)| pn == &embedding_property)
                    .and_then(|(_, store)| store.metric.clone())
                    .unwrap_or_else(|| "cosine".to_string())
            }
        };
        let metric = match effective_metric.as_str() {
            "cosine" => kglite_core::api::algorithms::DistanceMetric::Cosine,
            "dot_product" => kglite_core::api::algorithms::DistanceMetric::DotProduct,
            "euclidean" => kglite_core::api::algorithms::DistanceMetric::Euclidean,
            "poincare" => kglite_core::api::algorithms::DistanceMetric::Poincare,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown metric '{}'. Use 'cosine', 'dot_product', 'euclidean', or 'poincare'.",
                    other
                )));
            }
        };
        // Release GIL during heavy vector similarity computation
        let inner = self.inner.clone();
        let selection = self.cursor.selection.clone();
        let results = py
            .detach(|| {
                kglite_core::api::algorithms::vector_search(
                    &inner,
                    &selection,
                    &embedding_property,
                    &query_vector,
                    top_k,
                    metric,
                    exact,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        if to_df.unwrap_or(false) {
            // Build DataFrame via pandas
            let pandas = py.import("pandas")?;
            let records: Vec<Py<PyAny>> = results
                .iter()
                .filter_map(|r| {
                    self.inner.graph.node_weight(r.node_idx).map(|node| {
                        let dict = PyDict::new(py);
                        let _ = dict.set_item("id", py_out::value_to_py(py, &node.id()).ok());
                        if want("title") {
                            let _ =
                                dict.set_item("title", py_out::value_to_py(py, &node.title()).ok());
                        }
                        if want("type") {
                            let _ = dict.set_item("type", node.node_type_str(&self.inner.interner));
                        }
                        let _ = dict.set_item("score", r.score);
                        // properties_cloned reads from PropertyStorage::Columnar
                        // (the post-reload variant); property_iter yields
                        // nothing for that variant.
                        for (k, v) in node.properties_cloned(&self.inner.interner) {
                            if want(&k) {
                                let _ = dict.set_item(k, py_out::value_to_py(py, &v).ok());
                            }
                        }
                        dict.into()
                    })
                })
                .collect();
            let py_list = PyList::new(py, &records)?;
            let df = pandas.call_method1("DataFrame", (py_list,))?;
            return df.into_py_any(py);
        }

        // Return as list of dicts
        let py_list = PyList::empty(py);
        for r in &results {
            if let Some(node) = self.inner.graph.node_weight(r.node_idx) {
                let dict = PyDict::new(py);
                dict.set_item("id", py_out::value_to_py(py, &node.id())?)?;
                if want("title") {
                    dict.set_item("title", py_out::value_to_py(py, &node.title())?)?;
                }
                if want("type") {
                    dict.set_item("type", node.node_type_str(&self.inner.interner))?;
                }
                dict.set_item("score", r.score)?;
                // properties_cloned reads from PropertyStorage::Columnar
                // (the post-reload variant); property_iter yields nothing
                // for that variant.
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
        let key = (node_type.to_string(), format!("{text_column}_emb"));
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
        let key = (node_type.to_string(), format!("{text_column}_emb"));
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
                // search falls back to cosine — so report what search actually
                // uses rather than a bare `None` (operator note: the metric blank
                // was confusing even though ranking was correct).
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
    ///     List of dicts with 'node_type', 'text_column', 'dimension', 'count', 'metric'.
    fn list_embeddings(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let py_list = PyList::empty(py);
        for ((node_type, store_name), store) in &self.inner.embeddings {
            let text_column = store_name
                .strip_suffix("_emb")
                .unwrap_or(store_name.as_str());
            let dict = PyDict::new(py);
            dict.set_item("node_type", node_type)?;
            dict.set_item("text_column", text_column)?;
            dict.set_item("dimension", store.dimension)?;
            dict.set_item("count", store.len())?;
            dict.set_item("metric", store.metric.as_deref().unwrap_or("cosine"))?;
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
        // **Important**: use `properties_cloned()` (or `iter_owned()`)
        // rather than `property_iter()` — the latter yields *nothing* for
        // `PropertyStorage::Columnar` (the variant nodes use after a
        // save+reload cycle), which produced `nodes_with_property=0` for
        // every columnarised graph and flipped the status to
        // `store_orphan` on a healthy steady-state graph.
        for type_name in &types_to_scan {
            let type_indices = match self.inner.type_indices.get(type_name) {
                Some(ix) => ix,
                None => continue,
            };
            for nidx in type_indices.iter() {
                let node = match self.inner.graph.node_weight(nidx) {
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
            dict.set_item("embedding_key", format!("{}_emb", text_column))?;
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

            // length_stats: the heuristic data callers need to filter
            // out short-string columns (timestamps, enums) and
            // fully-unique columns (identifiers) before declaring a
            // candidate worth embedding.
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
        let g = Arc::make_mut(&mut self.inner);
        let key = (node_type.to_string(), format!("{}_emb", text_column));
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
        let g = Arc::make_mut(&mut self.inner);
        let stats = file::import_embeddings_from_file(g, path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        // Surface silent-drop cases: if the file contained entries but none
        // matched nodes in the current graph, the user almost always wants
        // to know — they're importing into the wrong graph or against an
        // ID schema that has drifted (e.g. code_tree qualified-name format
        // changes between releases). Emit a UserWarning that's visible by
        // default but still suppressible via the standard `warnings` module.
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
    #[pyo3(signature = (node_type_or_text_column, text_column=None))]
    fn embeddings(
        &self,
        py: Python<'_>,
        node_type_or_text_column: &str,
        text_column: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let result = PyDict::new(py);

        // Two-arg form: embeddings(node_type, text_column)
        if let Some(col) = text_column {
            let key = (node_type_or_text_column.to_string(), format!("{}_emb", col));
            let store = match self.inner.embeddings.get(&key) {
                Some(s) => s,
                None => return result.into_py_any(py),
            };

            for (&node_index, &_slot) in &store.node_to_slot {
                if let Some(embedding) = store.get_embedding(node_index) {
                    if let Some(node) = self.inner.graph.node_weight(NodeIndex::new(node_index)) {
                        let py_id = py_out::value_to_py(py, &node.id())?;
                        let py_vec = PyList::new(py, embedding)?;
                        result.set_item(py_id, py_vec)?;
                    }
                }
            }

            return result.into_py_any(py);
        }

        // One-arg form: embeddings(text_column) — selection-based
        let col = node_type_or_text_column;

        let level_count = self.cursor.selection.get_level_count();
        if level_count == 0 {
            return result.into_py_any(py);
        }

        let nodes: Vec<NodeIndex> = self
            .cursor
            .selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default();

        for node_idx in &nodes {
            let node = match self.inner.graph.node_weight(*node_idx) {
                Some(n) => n,
                None => continue,
            };

            let key = (
                node.node_type_str(&self.inner.interner).to_string(),
                format!("{}_emb", col),
            );
            let store = match self.inner.embeddings.get(&key) {
                Some(s) => s,
                None => continue,
            };

            if let Some(embedding) = store.get_embedding(node_idx.index()) {
                let py_id = py_out::value_to_py(py, &node.id())?;
                let py_vec = PyList::new(py, embedding)?;
                result.set_item(py_id, py_vec)?;
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

        let key = (node_type.to_string(), format!("{}_emb", text_column));
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

    // ========================================================================
    // Text-Level Embedding API
    // ========================================================================

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
    ///     text_column: The node property containing text to embed.
    ///     batch_size: Number of texts per ``model.embed()`` call (default 256).
    ///     show_progress: Show a tqdm progress bar (default ``True``).
    ///         Requires ``tqdm`` to be installed; silently falls back to no
    ///         progress bar if it is not available.
    ///     replace: Legacy alias for ``mode``. ``True`` → ``mode='all'``,
    ///         ``False`` → ``mode='missing'``. Ignored if ``mode`` is given.
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
    #[pyo3(signature = (node_type, text_column, batch_size=None, show_progress=None, replace=None, mode=None))]
    #[allow(clippy::too_many_arguments)]
    fn embed_texts(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        batch_size: Option<usize>,
        show_progress: Option<bool>,
        replace: Option<bool>,
        mode: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let model = self.get_embedder_or_error()?;
        let embedding_property = format!("{}_emb", text_column);
        let batch_size = batch_size.unwrap_or(256);
        // Resolve the embed mode. `mode` wins; else fall back to the legacy
        // `replace` bool (True→all, False→missing).
        let mode = match mode {
            Some(m) => match m {
                "missing" | "changed" | "all" => m.to_string(),
                other => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "embed_texts(mode={other:?}): unknown mode. Use 'missing' (default), \
                         'changed' (re-embed nodes whose text changed), or 'all'."
                    )));
                }
            },
            None => {
                if replace.unwrap_or(false) {
                    "all".to_string()
                } else {
                    "missing".to_string()
                }
            }
        };
        let replace = mode == "all";

        // Load model if it has a load() lifecycle method
        model
            .load()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        let dimension: usize = model.dimension();

        // Collect (node_index, text, text_hash) for nodes that need embedding
        let mut node_texts: Vec<(NodeIndex, String, u64)> = Vec::new();
        let mut skipped = 0usize;
        let mut skipped_existing = 0usize;
        let mut reembedded_changed = 0usize;

        let emb_key = (node_type.to_string(), embedding_property.clone());
        let existing_store = if replace {
            None
        } else {
            self.inner.embeddings.get(&emb_key)
        };

        // B4/B5 (operator 2026-06-17): an upsert (replace=False) into a store
        // whose dimension differs from the current model's would silently mix
        // dimensions and corrupt similarity search. Reject it with a clear
        // recipe instead. (replace=True is fine — it rebuilds a fresh store at
        // the new dimension below, so a model swap is deterministic that way.)
        if let Some(s) = existing_store {
            if s.dimension != dimension {
                model.unload();
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "embed_texts(): the model produces {dimension}-d vectors but the existing \
                     '{node_type}.{text_column}_emb' store is {}-d — embedding the rest would mix \
                     dimensions and corrupt search. Re-embed the whole column with replace=True to \
                     rebuild at the new dimension, or remove_embeddings('{node_type}', '{text_column}') first.",
                    s.dimension
                )));
            }
        }

        let node_indices: Vec<NodeIndex> = self
            .inner
            .type_indices
            .get(node_type)
            .map(|v| v.to_vec())
            .unwrap_or_default();

        let changed_mode = mode == "changed";
        for &node_idx in &node_indices {
            if let Some(node) = self.inner.graph.node_weight(node_idx) {
                match node.get_property(text_column).as_deref() {
                    Some(crate::datatypes::values::Value::String(s)) if !s.is_empty() => {
                        let hash = kglite_core::api::storage::EmbeddingStore::text_hash(s);
                        let has_emb = existing_store
                            .map(|st| st.get_embedding(node_idx.index()).is_some())
                            .unwrap_or(false);
                        if changed_mode {
                            // Re-embed nodes that are missing OR whose text changed.
                            let stale = existing_store
                                .map(|st| st.is_stale(node_idx.index(), hash))
                                .unwrap_or(true);
                            if stale {
                                if has_emb {
                                    reembedded_changed += 1;
                                }
                                node_texts.push((node_idx, s.clone(), hash));
                            } else {
                                skipped_existing += 1;
                            }
                        } else if has_emb {
                            // 'missing' mode: skip nodes that already have one.
                            skipped_existing += 1;
                        } else {
                            node_texts.push((node_idx, s.clone(), hash));
                        }
                    }
                    _ => {
                        skipped += 1;
                    }
                }
            }
        }

        if node_texts.is_empty() {
            model.unload();
            let result = PyDict::new(py);
            result.set_item("embedded", 0)?;
            result.set_item("skipped", skipped)?;
            result.set_item("skipped_existing", skipped_existing)?;
            result.set_item("reembedded_changed", reembedded_changed)?;
            result.set_item("dimension", dimension)?;
            return Ok(result.into());
        }

        // Clone existing store or create new — we'll merge new embeddings into it
        let mut store = match existing_store {
            Some(s) => s.clone(),
            None => kglite_core::api::storage::EmbeddingStore::new(dimension),
        };
        store.data.reserve(node_texts.len() * dimension);

        // Try to create a tqdm progress bar (if tqdm is installed and show_progress != false)
        let progress_bar = if show_progress.unwrap_or(true) {
            py.import("tqdm.auto")
                .or_else(|_| py.import("tqdm"))
                .ok()
                .and_then(|tqdm_mod| {
                    let kwargs = PyDict::new(py);
                    let _ = kwargs.set_item("total", node_texts.len());
                    let _ =
                        kwargs.set_item("desc", format!("Embedding {}.{}", node_type, text_column));
                    let _ = kwargs.set_item("unit", "text");
                    tqdm_mod.call_method("tqdm", (), Some(&kwargs)).ok()
                })
        } else {
            None
        };

        for batch in node_texts.chunks(batch_size) {
            let texts: Vec<String> = batch.iter().map(|(_, t, _)| t.clone()).collect();

            // Release the GIL while embedding — PyEmbedderAdapter
            // reacquires inside, fastembed never needs it.
            let embeddings = match py.detach(|| model.embed(&texts)) {
                Ok(v) => v,
                Err(e) => {
                    if let Some(ref bar) = progress_bar {
                        let _ = bar.call_method0("close");
                    }
                    model.unload();
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e));
                }
            };

            if embeddings.len() != batch.len() {
                if let Some(ref bar) = progress_bar {
                    let _ = bar.call_method0("close");
                }
                model.unload();
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "model.embed() returned {} vectors for {} texts",
                    embeddings.len(),
                    batch.len()
                )));
            }

            for (i, vec) in embeddings.iter().enumerate() {
                if vec.len() != dimension {
                    if let Some(ref bar) = progress_bar {
                        let _ = bar.call_method0("close");
                    }
                    model.unload();
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "model.embed() returned vector of dimension {} (expected {})",
                        vec.len(),
                        dimension
                    )));
                }
                store.set_embedding(batch[i].0.index(), vec);
                store.set_text_hash(batch[i].0.index(), batch[i].2);
            }

            // Update progress bar
            if let Some(ref bar) = progress_bar {
                let _ = bar.call_method1("update", (batch.len(),));
            }
        }

        // Close progress bar
        if let Some(ref bar) = progress_bar {
            let _ = bar.call_method0("close");
        }

        // Unload model after embedding is complete
        model.unload();

        // Stamp the model identity onto the store (provenance) when the
        // embedder names its model — leaves a prior id intact otherwise.
        if let Some(mid) = model.model_id() {
            store.model_id = Some(mid);
        }

        let embedded = node_texts.len();
        let g = Arc::make_mut(&mut self.inner);
        g.embeddings.insert(emb_key, store);

        let result = PyDict::new(py);
        result.set_item("embedded", embedded)?;
        result.set_item("skipped", skipped)?;
        result.set_item("skipped_existing", skipped_existing)?;
        result.set_item("reembedded_changed", reembedded_changed)?;
        result.set_item("dimension", dimension)?;
        Ok(result.into())
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
    ///     metric: Distance metric (default ``'cosine'``).
    ///     to_df: If True, return a pandas DataFrame.
    ///
    /// Returns:
    ///     Same format as ``vector()`` — list of dicts or DataFrame.
    #[pyo3(signature = (text_column, query, top_k=None, metric=None, to_df=None, returning=None, exact=None))]
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

        // Load model if it has a load() lifecycle method
        model
            .load()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        // Embed the query text, then unload regardless of success/failure.
        // Release the GIL while the embedder runs.
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

        // Delegate to existing vector_search
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
    /// for whole-corpus queries on large stores; pass ``exact=True`` to force an
    /// exact scan. The index is dropped automatically whenever the store's
    /// vectors change (``add_embeddings`` / ``embed_texts`` / ``compact``) —
    /// rebuild it afterwards.
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
    ///
    /// Returns:
    ///     dict: ``{'indexed': int, 'metric': str, 'm': int}`` — vectors indexed.
    ///
    /// Raises:
    ///     ValueError: if the store doesn't exist or the metric is unsupported.
    #[pyo3(signature = (node_type, text_column, m=None, ef_construction=None, ef_search=None, metric=None))]
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
    ) -> PyResult<Py<PyAny>> {
        use kglite_core::api::algorithms::DistanceMetric;
        use kglite_core::api::algorithms::HnswParams;

        let embedding_property = format!("{}_emb", text_column);
        let key = (node_type.to_string(), embedding_property);

        // Resolve metric: explicit arg > stored metric > cosine.
        let metric_name = match metric {
            Some(m) => m.to_string(),
            None => self
                .inner
                .embeddings
                .get(&key)
                .and_then(|s| s.metric.clone())
                .unwrap_or_else(|| "cosine".to_string()),
        };
        let dmetric = match metric_name.as_str() {
            "cosine" => DistanceMetric::Cosine,
            "dot_product" => DistanceMetric::DotProduct,
            "euclidean" => DistanceMetric::Euclidean,
            "poincare" => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "build_vector_index: the 'poincare' metric is not supported by HNSW; \
                     Poincaré search stays on the exact (brute-force) path.",
                ));
            }
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown metric '{}'. Use 'cosine', 'dot_product', or 'euclidean'.",
                    other
                )));
            }
        };

        let params = HnswParams {
            m: m.unwrap_or(16).max(2),
            ef_construction: ef_construction.unwrap_or(200).max(1),
            ef_search: ef_search.unwrap_or(64).max(1),
        };

        let g = Arc::make_mut(&mut self.inner);
        let store = g.embeddings.get_mut(&key).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "No embedding store '{}.{}_emb' to index. Call set_embeddings()/embed_texts() first.",
                node_type, text_column
            ))
        })?;
        let indexed = store.len();

        // Build off the GIL — pure CPU over the contiguous vector buffer.
        // A deterministic seed keeps builds reproducible.
        let seed = 0x9E37_79B9_7F4A_7C15 ^ (indexed as u64);
        let build = py.detach(|| store.build_index(dmetric, params, seed));
        build.map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let result = PyDict::new(py);
        result.set_item("indexed", indexed)?;
        result.set_item("metric", metric_name)?;
        result.set_item("m", params.m)?;
        Ok(result.into())
    }

    /// Drop the HNSW index for an embedding store (search reverts to exact
    /// brute-force). No-op if no index exists. Returns ``True`` if one was
    /// dropped.
    #[pyo3(signature = (node_type, text_column))]
    fn drop_vector_index(&mut self, node_type: &str, text_column: &str) -> PyResult<bool> {
        let key = (node_type.to_string(), format!("{}_emb", text_column));
        let g = Arc::make_mut(&mut self.inner);
        match g.embeddings.get_mut(&key) {
            Some(store) => {
                let had = store.has_index();
                store.invalidate_index();
                Ok(had)
            }
            None => Ok(false),
        }
    }

    /// Whether an HNSW index is currently built over an embedding store.
    #[pyo3(signature = (node_type, text_column))]
    fn has_vector_index(&self, node_type: &str, text_column: &str) -> PyResult<bool> {
        let key = (node_type.to_string(), format!("{}_emb", text_column));
        Ok(self
            .inner
            .embeddings
            .get(&key)
            .map(|s| s.has_index())
            .unwrap_or(false))
    }
}
