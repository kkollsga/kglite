use crate::datatypes::{py_in, py_out};
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;
use std::sync::Arc;

use crate::graph::{KnowledgeGraph, NodeKeyGuard, NodeKeyKind};
use kglite_core::api::embeddings::{EmbedError, EmbedHooks, EmbedMode};
use kglite_core::api::index_names::{IndexClass, NamedIndex};
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
    /// Replace the embedding store of a node type with vectors keyed by node id.
    #[pyo3(signature = (node_type, text_column, embeddings, metric=None, *, precision=None))]
    pub(super) fn set_node_embeddings(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        embeddings: &Bound<'_, PyDict>,
        metric: Option<&str>,
        precision: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        this.check_durable_owner()?;
        let precision = super::quantization::parse_precision(precision)?;
        super::quantization::refuse_poincare(precision, metric)?;
        let entries = marshal_embedding_batch(embeddings)?;
        let report = this.logged_write(|g| {
            kglite_core::api::embeddings::set_embeddings(g, node_type, text_column, metric, entries)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
        })?;
        let quantized = this.quantize_written(
            kglite_core::api::embeddings::EmbeddingEntity::Node,
            node_type,
            text_column,
            precision,
        )?;

        let result = PyDict::new(py);
        result.set_item("embeddings_stored", report.embeddings_stored)?;
        result.set_item("dimension", report.dimension)?;
        result.set_item("skipped", report.skipped)?;
        if precision.is_some() {
            result.set_item(
                "quantization",
                super::quantization::quantization_value(py, quantized.as_ref())?,
            )?;
        }
        // The replaced store's buffer is freed; hand its pages back.
        crate::allocator::release_after_seam(py);
        Ok(result.into())
    }

    /// Upsert node vectors keyed by id into a node type's embedding store, creating it if absent.
    #[pyo3(signature = (node_type, text_column, embeddings, metric=None))]
    pub(super) fn add_node_embeddings(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        embeddings: &Bound<'_, PyDict>,
        metric: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        this.check_durable_owner()?;
        let entries = marshal_embedding_batch(embeddings)?;
        let report = this.logged_write(|g| {
            kglite_core::api::embeddings::add_embeddings(g, node_type, text_column, metric, entries)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
        })?;

        let result = PyDict::new(py);
        result.set_item("embeddings_stored", report.embeddings_stored)?;
        result.set_item("dimension", report.dimension)?;
        result.set_item("skipped", report.skipped)?;
        result.set_item("store_created", report.store_created)?;
        Ok(result.into())
    }

    /// Rank the current selection's node vectors in a text column's store against a query vector.
    #[pyo3(signature = (text_column, query_vector, top_k=10, metric=None, to_df=false, returning=None, exact=false))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn node_vector_search(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        text_column: &str,
        query_vector: Vec<f32>,
        top_k: Option<usize>,
        metric: Option<&str>,
        to_df: Option<bool>,
        returning: Option<Vec<String>>,
        exact: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

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
        let inner = this.graph().clone();
        let selection = this.cursor.selection.clone();
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
            let records: Vec<Py<PyAny>> = results
                .iter()
                .filter_map(|r| {
                    this.graph()
                        .graph
                        .node_view(r.node_idx)
                        .map(|node| (r, node))
                })
                .map(|(r, node)| -> PyResult<Py<PyAny>> {
                    let dict = PyDict::new(py);
                    dict.set_item("id", py_out::value_to_py(py, &node.id())?)?;
                    if want("title") {
                        dict.set_item("title", py_out::value_to_py(py, &node.title())?)?;
                    }
                    if want("type") {
                        dict.set_item("type", node.node_type_str(&this.graph().interner))?;
                    }
                    dict.set_item("score", r.score)?;
                    // properties_cloned reads PropertyStorage::Columnar (the
                    // durable shape); property_iter yields nothing for it.
                    for (k, v) in node.properties_cloned(&this.graph().interner) {
                        if want(&k) {
                            dict.set_item(k, py_out::value_to_py(py, &v)?)?;
                        }
                    }
                    Ok(dict.into())
                })
                .collect::<PyResult<_>>()?;
            let py_list = PyList::new(py, &records)?;
            return crate::datatypes::pandas_out::dataframe(py, py_list.as_any(), None, None, None);
        }

        let py_list = PyList::empty(py);
        for r in &results {
            if let Some(node) = this.graph().graph.node_view(r.node_idx) {
                let dict = PyDict::new(py);
                dict.set_item("id", py_out::value_to_py(py, &node.id())?)?;
                if want("title") {
                    dict.set_item("title", py_out::value_to_py(py, &node.title())?)?;
                }
                if want("type") {
                    dict.set_item("type", node.node_type_str(&this.graph().interner))?;
                }
                dict.set_item("score", r.score)?;
                for (k, v) in node.properties_cloned(&this.graph().interner) {
                    if want(&k) {
                        dict.set_item(k, py_out::value_to_py(py, &v)?)?;
                    }
                }
                py_list.append(dict)?;
            }
        }

        py_list.into_py_any(py)
    }

    /// The vector dimension of a node type's embedding store for a text column, or None when there is no such store.
    pub(super) fn node_embedding_dim(
        slf: &Bound<'_, Self>,
        node_type: &str,
        text_column: &str,
    ) -> PyResult<Option<usize>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok({
            let key = kglite_core::api::embeddings::store_key(node_type, text_column);
            this.graph().embeddings.get(&key).map(|s| s.dimension)
        })
    }

    /// Provenance for one node or relationship embedding store, or None.
    #[pyo3(signature = (node_type, text_column, *, entity="node"))]
    fn embedding_info(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        entity: &str,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        use kglite_core::api::embeddings::{embedding_info, EmbeddingEntity};
        let entity =
            EmbeddingEntity::parse(entity).map_err(crate::error_py::ArgumentError::new_err)?;
        let Some(info) = embedding_info(this.graph(), entity, node_type, text_column) else {
            return Ok(py.None());
        };
        let d = PyDict::new(py);
        d.set_item(type_key(info.entity), info.type_name)?;
        d.set_item("text_column", info.text_column)?;
        d.set_item("dimension", info.dimension)?;
        d.set_item("count", info.count)?;
        d.set_item("model", info.model)?;
        d.set_item("metric", info.metric)?;
        d.set_item("hashed", info.hashed)?;
        d.set_item(
            "quantization",
            super::quantization::quantization_value(py, info.quantization.as_ref())?,
        )?;
        d.into_py_any(py)
    }

    /// Copy every node and relationship embedding store from `other` into this graph.
    #[pyo3(signature = (other, *, relationship_keys=None))]
    fn copy_embeddings_from(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, KnowledgeGraph>,
        relationship_keys: Option<HashMap<String, String>>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        // Mirror extend()'s safe shape: clone the source Arc first (so a
        // self-copy doesn't double-borrow), then mutate self.
        let src_arc = match other.try_borrow() {
            Ok(o) => Arc::clone(o.graph()),
            Err(_) => Arc::clone(this.graph()),
        };
        this.check_durable_owner()?;
        let keys = relationship_keys.unwrap_or_default();
        let report = this.logged_write(|g| {
            g.copy_embeddings_with_relationships_from(&src_arc, &keys)
                .map_err(crate::error_py::ArgumentError::new_err)
        })?;
        crate::allocator::release_after_seam(py);
        let d = PyDict::new(py);
        d.set_item("stores_copied", report.stores_copied)?;
        d.set_item("vectors_copied", report.vectors_copied)?;
        d.set_item("vectors_skipped", report.vectors_skipped)?;
        d.set_item("relationship_stores_copied", report.relationships.stores)?;
        d.set_item("relationship_vectors_copied", report.relationships.carried)?;
        d.set_item("relationship_vectors_skipped", report.relationships.skipped)?;
        d.into_py_any(py)
    }

    /// List every embedding store: node stores, then relationship stores.
    fn list_embeddings(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        use kglite_core::api::embeddings::{list_embeddings, list_relationship_embeddings};
        let py_list = PyList::empty(py);
        for info in list_embeddings(this.graph()) {
            let dict = PyDict::new(py);
            dict.set_item("entity", "node")?;
            dict.set_item("node_type", info.node_type)?;
            dict.set_item("text_column", info.text_column)?;
            dict.set_item("store_name", info.store_name)?;
            dict.set_item("dimension", info.dimension)?;
            dict.set_item("count", info.count)?;
            dict.set_item("metric", info.metric)?;
            dict.set_item("storage", info.storage)?;
            dict.set_item(
                "quantization",
                super::quantization::quantization_value(py, info.quantization.as_ref())?,
            )?;
            py_list.append(dict)?;
        }
        for info in list_relationship_embeddings(this.graph()) {
            let dict = PyDict::new(py);
            dict.set_item("entity", "relationship")?;
            dict.set_item("relationship_type", info.relationship_type)?;
            dict.set_item("text_column", info.text_column)?;
            dict.set_item("store_name", info.store_name)?;
            dict.set_item("dimension", info.dimension)?;
            dict.set_item("count", info.count)?;
            dict.set_item("metric", info.metric)?;
            dict.set_item("storage", info.storage)?;
            dict.set_item(
                "quantization",
                super::quantization::quantization_value(py, info.quantization.as_ref())?,
            )?;
            py_list.append(dict)?;
        }
        py_list.into_py_any(py)
    }

    /// Diagnose embedding coverage per (type, text column) for nodes and relationships.
    #[pyo3(signature = (node_type=None, *, relationship_type=None))]
    fn embedding_diagnostics(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type: Option<&str>,
        relationship_type: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        use kglite_core::api::embeddings::{embedding_diagnostics, EmbeddingEntity};
        let rows = embedding_diagnostics(this.graph(), node_type, relationship_type)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let py_list = PyList::empty(py);
        for row in rows {
            let (with_key, embedded_key) = match row.entity {
                EmbeddingEntity::Node => ("nodes_with_property", "nodes_embedded"),
                EmbeddingEntity::Relationship => {
                    ("relationships_with_property", "relationships_embedded")
                }
            };
            let dict = PyDict::new(py);
            dict.set_item("entity", row.entity.as_str())?;
            dict.set_item(type_key(row.entity), row.type_name)?;
            dict.set_item("text_column", row.text_column)?;
            dict.set_item("embedding_key", row.embedding_key)?;
            dict.set_item(with_key, row.with_property)?;
            dict.set_item(embedded_key, row.embedded)?;
            dict.set_item("status", row.status.as_str())?;
            dict.set_item("dimension", row.dimension)?;
            dict.set_item("metric", row.metric)?;
            dict.set_item("storage", row.storage)?;
            dict.set_item(
                "quantization",
                super::quantization::quantization_value(py, row.quantization.as_ref())?,
            )?;
            let length_stats = PyDict::new(py);
            length_stats.set_item("mean_length", row.length_stats.mean_length)?;
            length_stats.set_item("max_length", row.length_stats.max_length)?;
            length_stats.set_item("distinct_count", row.length_stats.distinct_count)?;
            length_stats.set_item("distinct_ratio", row.length_stats.distinct_ratio)?;
            dict.set_item("length_stats", length_stats)?;
            py_list.append(dict)?;
        }
        py_list.into_py_any(py)
    }

    /// Remove a node type's embedding store for a text column; refuses a store that does not exist.
    pub(super) fn remove_node_embeddings(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
    ) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        this.logged_write(|g| {
            kglite_core::api::embeddings::remove_embeddings(g, node_type, text_column)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
        })?;
        crate::allocator::release_after_seam(py);
        Ok(())
    }

    /// Export node and relationship embeddings to a standalone .kgle file.
    #[pyo3(signature = (path, node_types=None, *, relationship_keys=None))]
    fn export_embeddings(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        path: &str,
        node_types: Option<Bound<'_, PyAny>>,
        relationship_keys: Option<HashMap<String, String>>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

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

        let inner = this.graph().clone();
        let path_owned = path.to_string();
        let keys = relationship_keys.unwrap_or_default();
        let stats = py
            .detach(move || {
                file::export_embeddings_to_file(&inner, &path_owned, filter.as_ref(), &keys)
            })
            .map_err(embedding_file_error)?;

        let result = PyDict::new(py);
        result.set_item("stores", stats.stores)?;
        result.set_item("embeddings", stats.embeddings)?;
        result.set_item("relationship_stores", stats.relationship_stores)?;
        result.set_item("relationship_embeddings", stats.relationship_embeddings)?;
        result.into_py_any(py)
    }

    /// Import node and relationship embeddings from a .kgle file.
    #[pyo3(signature = (path, *, relationship_keys=None, precision=None))]
    fn import_embeddings(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        path: &str,
        relationship_keys: Option<HashMap<String, String>>,
        precision: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        this.check_durable_owner()?;
        let precision = super::quantization::parse_precision(precision)?;
        let keys = relationship_keys.unwrap_or_default();
        let stats = this.logged_write(|g| {
            file::import_embeddings_from_file(g, path, &keys).map_err(embedding_file_error)
        })?;
        // The import's decode buffers are dead; their pages are not reusable.
        crate::allocator::release_after_seam(py);
        // Every store the import installed gets the requested copy; a store
        // that cannot hold one (poincare metric) fails the call after the
        // vectors are in, and the message says so.
        let mut quantized = 0usize;
        if precision.is_some() {
            use kglite_core::api::embeddings::EmbeddingEntity;
            for (type_name, column) in &stats.installed {
                this.quantize_written(EmbeddingEntity::Node, type_name, column, precision)?;
                quantized += 1;
            }
            for (type_name, column) in &stats.relationships.installed {
                this.quantize_written(EmbeddingEntity::Relationship, type_name, column, precision)?;
                quantized += 1;
            }
        }

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

        let relationships = &stats.relationships;
        if relationships.carried == 0 && relationships.skipped > 0 {
            let msg = format!(
                "import_embeddings('{}'): imported 0 relationship embeddings, skipped {} — \
                 no relationship in the file connects nodes of this graph by the same \
                 type and endpoint ids.",
                path, relationships.skipped
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
        result.set_item("relationship_stores", relationships.stores)?;
        result.set_item("relationship_imported", relationships.carried)?;
        result.set_item("relationship_skipped", relationships.skipped)?;
        result.set_item("relationship_dropped_stores", relationships.dropped_stores)?;
        if precision.is_some() {
            result.set_item("quantized_stores", quantized)?;
        }
        result.into_py_any(py)
    }

    /// Read node vectors by id: a whole node type's store, or the current selection's.
    #[pyo3(signature = (node_type_or_text_column, text_column=None))]
    pub(super) fn node_embeddings(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type_or_text_column: &str,
        text_column: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let result = PyDict::new(py);

        if let Some(col) = text_column {
            let rows = kglite_core::api::embeddings::node_embeddings(
                this.graph(),
                node_type_or_text_column,
                col,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            for (id, vector) in rows {
                result.set_item(py_out::value_to_py(py, &id)?, PyList::new(py, vector)?)?;
            }
            return result.into_py_any(py);
        }

        // One-arg form: embeddings(text_column) — selection-based. A selection
        // that was never narrowed means the whole graph (the never-selected
        // rule get_nodes() applies); one a query emptied stays empty.
        let col = node_type_or_text_column;

        let selection = &this.cursor.selection;
        let level = selection
            .get_level(selection.get_level_count().saturating_sub(1))
            .filter(|level| level.node_count() > 0);
        let nodes: Vec<NodeIndex> = match level {
            Some(level) => level.get_all_nodes(),
            None if selection.never_selected() => {
                GraphRead::node_indices(&this.graph().graph).collect()
            }
            None => Vec::new(),
        };

        for node_idx in &nodes {
            let node = match this.graph().graph.node_view(*node_idx) {
                Some(n) => n,
                None => continue,
            };

            let key = kglite_core::api::embeddings::store_key(
                node.node_type_str(&this.graph().interner),
                col,
            );
            let store = match this.graph().embeddings.get(&key) {
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

    /// One node's stored vector by id, or None when the node or its vector is absent; refuses a store that does not exist.
    pub(super) fn node_embedding(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        node_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let id = py_in::py_value_to_value(node_id)?;
        let vector =
            kglite_core::api::embeddings::node_embedding(this.graph(), node_type, text_column, &id)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        match vector {
            Some(vector) => PyList::new(py, vector)?.into_py_any(py),
            None => Ok(py.None()),
        }
    }

    /// Every vector in a relationship store, addressed by endpoint ids, as rows for an edge list and edge-feature matrix.
    #[pyo3(signature = (relationship_type, text_column, *, relationship_keys=None))]
    pub(super) fn relationship_embeddings(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        relationship_type: &str,
        text_column: &str,
        relationship_keys: Option<HashMap<String, String>>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let keys = relationship_keys.unwrap_or_default();
        let rows = kglite_core::api::embeddings::relationship_embeddings(
            this.graph(),
            relationship_type,
            text_column,
            &keys,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let out = PyList::empty(py);
        for row in rows {
            let item = PyDict::new(py);
            item.set_item("source", py_out::value_to_py(py, &row.source_id)?)?;
            item.set_item("target", py_out::value_to_py(py, &row.target_id)?)?;
            item.set_item("source_type", row.source_type)?;
            item.set_item("target_type", row.target_type)?;
            match &row.key {
                Some(key) => item.set_item("key", py_out::value_to_py(py, key)?)?,
                None => item.set_item("key", py.None())?,
            }
            item.set_item("vector", PyList::new(py, &row.vector)?)?;
            out.append(item)?;
        }
        out.into_py_any(py)
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
    fn set_embedder(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        model: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let Some(model) = model else {
            this.embedder = None;
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
        this.embedder = Some(Arc::new(adapter));
        Ok(())
    }

    /// Embed a text column for every node of a type with the registered model.
    #[pyo3(signature = (node_type, text_column, batch_size=256, show_progress=true, mode=None, *, metric=None, precision=None))]
    // One Rust argument per Python keyword, as embed_relationship_texts has.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn embed_node_texts(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        batch_size: Option<usize>,
        show_progress: Option<bool>,
        mode: Option<&str>,
        metric: Option<&str>,
        precision: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        // Refuse derived durable/CDC handles before loading or invoking the
        // model. `commit_wal()` retains the same guard after mutation as
        // defense in depth, but it cannot undo an already-installed store.
        this.check_durable_owner()?;
        let precision = super::quantization::parse_precision(precision)?;
        super::quantization::refuse_poincare(precision, metric)?;
        let (model, _) = this.embedder_or_idle(
            kglite_core::api::embeddings::EmbeddingEntity::Node,
            node_type,
            text_column,
        );
        let mode = match mode.unwrap_or("missing") {
            "missing" => EmbedMode::Missing,
            "changed" => EmbedMode::Changed,
            "all" => EmbedMode::All,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "embed_texts(mode={other:?}): unknown mode. Use 'missing' (default), \
                     'changed' (re-embed nodes whose text changed), or 'all'."
                )));
            }
        };
        // A type with no nodes stays the `{'embedded': 0}` no-op it has always
        // been; a type the graph has never *seen* is a mistake and gets
        // `set_embeddings`' complaint, before the model is loaded.
        if !this.graph().type_indices.contains_key(node_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node type '{}' does not exist in the graph",
                node_type
            )));
        }

        // Resolve tqdm here, before core takes over: an optional module has to
        // be looked up from the method body, not from a hook core drives.
        let progress_factory = resolve_progress_factory(py, show_progress.unwrap_or(true));
        // The bar is opened by `on_start`, which core calls with the count
        // only when there is something to embed — so a no-op pass draws no bar.
        let progress_bar: std::cell::RefCell<Option<Bound<'_, PyAny>>> =
            std::cell::RefCell::new(None);
        let open_bar = |total: usize| {
            *progress_bar.borrow_mut() = open_progress_bar(
                progress_factory.as_ref(),
                total,
                format!("Embedding {}.{}", node_type, text_column),
            );
        };
        let tick = |done: usize| {
            if let Some(bar) = progress_bar.borrow().as_ref() {
                let _ = bar.call_method1("update", (done,));
            }
        };
        // Release the GIL while embedding — PyEmbedderAdapter reacquires
        // inside, fastembed never needs it.
        let embed_batch = |texts: &[String]| py.detach(|| model.embed(texts));
        let hooks = EmbedHooks {
            batch_size: batch_size.unwrap_or(256),
            // A pass with nothing to embed leaves the model unloaded; the
            // reported dimension comes from the store or the registered model.
            load_when_idle: false,
            embed_batch: Some(&embed_batch),
            on_start: Some(&open_bar),
            on_batch: Some(&tick),
        };
        let outcome = this.logged_write_lazy(|inner| {
            let outcome = kglite_core::api::embeddings::embed_property(
                inner,
                node_type,
                text_column,
                mode,
                model.as_ref(),
                &hooks,
                metric,
            );
            if let Some(bar) = progress_bar.borrow().as_ref() {
                let _ = bar.call_method0("close");
            }
            outcome.map_err(|error| embed_error(error, node_type, text_column))
        })?;
        let quantized = this.quantize_written(
            kglite_core::api::embeddings::EmbeddingEntity::Node,
            node_type,
            text_column,
            precision,
        )?;
        let result = PyDict::new(py);
        result.set_item("embedded", outcome.embedded)?;
        result.set_item("skipped", outcome.skipped)?;
        result.set_item("skipped_existing", outcome.skipped_existing)?;
        result.set_item("reembedded_changed", outcome.reembedded_changed)?;
        let dimension = if outcome.dimension == 0 {
            model.dimension()
        } else {
            outcome.dimension
        };
        result.set_item("dimension", dimension)?;
        if precision.is_some() {
            result.set_item(
                "quantization",
                super::quantization::quantization_value(py, quantized.as_ref())?,
            )?;
        }
        crate::allocator::release_after_seam(py);
        Ok(result.into())
    }

    /// Embed a query with the registered model and rank the current selection's node vectors against it.
    #[pyo3(signature = (text_column, query, top_k=10, metric=None, to_df=false, returning=None, exact=false))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn node_search_text(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        text_column: &str,
        query: &str,
        top_k: Option<usize>,
        metric: Option<&str>,
        to_df: Option<bool>,
        returning: Option<Vec<String>>,
        exact: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let query_vector = {
            let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
            guard.embed_query(py, query)?
        };
        Self::node_vector_search(
            slf,
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

    /// Build an HNSW index over a node embedding store so vector search scales on large stores.
    #[pyo3(signature = (node_type, text_column, m=None, ef_construction=None, ef_search=None, metric=None, auto_refresh_limit=None, *, name=None))]
    // One Rust argument per HNSW knob the Python signature exposes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_node_vector_index(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        metric: Option<&str>,
        auto_refresh_limit: Option<usize>,
        name: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let named = NamedIndex::node(IndexClass::Vector, node_type, &[text_column.to_string()]);
        // Build off the GIL — pure CPU over the contiguous vector buffer.
        let report = this.logged_write(|g| {
            claim_name(g, name, &named)?;
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
            register_name(g, name, named);
            Ok(report)
        })?;

        let result = PyDict::new(py);
        result.set_item("indexed", report.indexed)?;
        result.set_item("metric", report.metric)?;
        result.set_item("m", report.m)?;
        Ok(result.into())
    }

    /// Drop the HNSW index over a node embedding store (the vectors stay); True if one was dropped.
    #[pyo3(signature = (node_type, text_column))]
    pub(super) fn drop_node_vector_index(
        slf: &Bound<'_, Self>,
        node_type: &str,
        text_column: &str,
    ) -> PyResult<bool> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        this.logged_write(|g| {
            Ok(kglite_core::api::embeddings::drop_vector_index(
                g,
                node_type,
                text_column,
            ))
        })
    }

    /// Whether an HNSW index is currently built over a node embedding store.
    #[pyo3(signature = (node_type, text_column))]
    pub(super) fn has_node_vector_index(
        slf: &Bound<'_, Self>,
        node_type: &str,
        text_column: &str,
    ) -> PyResult<bool> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        Ok(kglite_core::api::embeddings::has_vector_index(
            this.graph(),
            node_type,
            text_column,
        ))
    }

    /// Fold every outstanding vector into a node store's HNSW index now; refuses when no index is built.
    #[pyo3(signature = (node_type, text_column))]
    pub(super) fn refresh_node_vector_index(
        slf: &Bound<'_, Self>,
        node_type: &str,
        text_column: &str,
    ) -> PyResult<usize> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        kglite_core::api::embeddings::refresh_vector_index(this.graph(), node_type, text_column)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }
}

impl KnowledgeGraph {
    /// One query string through the registered model, checked against the
    /// model's declared dimension — the query side of both `search_text` twins.
    pub(super) fn embed_query(&self, py: Python<'_>, query: &str) -> PyResult<Vec<f32>> {
        let model = self.get_embedder_or_error()?;
        model
            .load()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        // After `load()`: a model that builds on first use declares no width
        // before then.
        let model_dimension = model.dimension();

        // Unload regardless of success or failure — hence the `?` after it.
        let texts = vec![query.to_string()];
        let embed_result = py.detach(|| model.embed(&texts));
        model.unload();
        let embeddings = embed_result.map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        if embeddings.len() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "search_text: model.embed() returned {} vectors for 1 texts",
                embeddings.len()
            )));
        }

        let query_vector = embeddings.into_iter().next().unwrap();
        if query_vector.len() != model_dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "search_text: model.embed() returned vector width {}, expected registered model dimension {model_dimension}",
                query_vector.len()
            )));
        }
        Ok(query_vector)
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
    let mut rows = py_in::F32Rows::default();
    for (key, value) in embeddings.iter() {
        entries.push((py_in::py_value_to_value(&key)?, rows.extract(&value)?));
    }
    Ok(entries)
}

/// One core [`EmbedError`] as the exception class `embed_texts` has always
/// raised for it, with the Python-side remedy spelled out where there is one.
/// Core states the fact; naming `mode='all'` and `remove_embeddings()` is this
/// binding's job, because they are this binding's spellings.
fn embed_error(error: EmbedError, node_type: &str, text_column: &str) -> PyErr {
    match error {
        EmbedError::Dimension { store, model } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "embed_texts(): the model produces {model}-d vectors but the existing \
                 '{node_type}.{text_column}_emb' store is {store}-d — embedding the rest would mix \
                 dimensions and corrupt search. Re-embed the whole column with mode='all' to \
                 rebuild at the new dimension, or remove_embeddings('{node_type}', \
                 '{text_column}') first."
            ))
        }
        EmbedError::Column(message) | EmbedError::Output(message) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
        }
        EmbedError::Model(message) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(message),
    }
}

/// The `tqdm` callable `embed_texts` opens its bar with, resolved once before
/// the pass starts. `None` when the caller opted out, and `None` rather than
/// an error when tqdm is not installed — the documented silent fallback.
///
/// Resolved through `importlib.import_module`, not `py.import`: the latter is
/// CPython's `PyImport_Import`, which calls `__import__` with an *empty*
/// fromlist, and that path discards the resolved submodule and imports the
/// top-level package instead. So `py.import("tqdm.auto")` fails with "No
/// module named 'tqdm'" whenever the parent is unimportable, even though
/// `sys.modules["tqdm.auto"]` already holds the target. `import_module`
/// honours that `sys.modules` entry for the full dotted name.
pub(super) fn resolve_progress_factory<'py>(
    py: Python<'py>,
    show_progress: bool,
) -> Option<Bound<'py, PyAny>> {
    if !show_progress {
        return None;
    }
    let import_module = py.import("importlib").ok()?.getattr("import_module").ok()?;
    ["tqdm.auto", "tqdm"]
        .iter()
        .find_map(|name| import_module.call1((*name,)).ok())
        .and_then(|tqdm_mod| tqdm_mod.getattr("tqdm").ok())
}

/// One tqdm bar from the factory `resolve_progress_factory` returned, sized to
/// the texts this pass will embed. `None` keeps the silent fallback: no
/// factory, or a factory that refuses the call, simply draws no bar.
pub(super) fn open_progress_bar<'py>(
    factory: Option<&Bound<'py, PyAny>>,
    total: usize,
    desc: String,
) -> Option<Bound<'py, PyAny>> {
    let factory = factory?;
    let kwargs = PyDict::new(factory.py());
    kwargs.set_item("total", total).ok()?;
    kwargs.set_item("desc", desc).ok()?;
    kwargs.set_item("unit", "text").ok()?;
    factory.call((), Some(&kwargs)).ok()
}

/// The row key that names a store's type: `node_type` or `relationship_type`.
fn type_key(entity: kglite_core::api::embeddings::EmbeddingEntity) -> &'static str {
    match entity {
        kglite_core::api::embeddings::EmbeddingEntity::Node => "node_type",
        kglite_core::api::embeddings::EmbeddingEntity::Relationship => "relationship_type",
    }
}

/// An ambiguous relationship carry is the caller's to resolve (it names a
/// key), so it is an `ArgumentError`; anything else about the file stays an
/// `IOError`.
fn embedding_file_error(error: std::io::Error) -> PyErr {
    if error.kind() == std::io::ErrorKind::InvalidInput {
        crate::error_py::ArgumentError::new_err(error.to_string())
    } else {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(error.to_string())
    }
}

/// Claim `name` (when given) for the index `named` describes, before the build:
/// a name already taken refuses the call with nothing built.
pub(super) fn claim_name(
    graph: &kglite_core::api::DirGraph,
    name: Option<&str>,
    named: &NamedIndex,
) -> PyResult<()> {
    match name {
        Some(name) => kglite_core::api::index_names::check_index_name(graph, name, named)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>),
        None => Ok(()),
    }
}

/// Record the name [`claim_name`] accepted, once the build has succeeded.
pub(super) fn register_name(
    graph: &mut kglite_core::api::DirGraph,
    name: Option<&str>,
    named: NamedIndex,
) {
    if let Some(name) = name {
        kglite_core::api::index_names::register_index_name(graph, name, named);
    }
}
