// Index Management #[pymethods] — extracted from mod.rs

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::{get_graph_mut, KnowledgeGraph};

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Index Management Methods
    // ========================================================================

    /// Create an index on a property for a specific node type.
    ///
    /// Indexes dramatically speed up equality filters on the indexed property.
    /// Once created, the index is automatically used by where() operations.
    ///
    /// Args:
    ///     node_type: The type of nodes to index
    ///     property: The property name to index
    ///
    /// Returns:
    ///     Dictionary with 'unique_values' count and success status
    ///
    /// Example:
    ///     ```python
    ///     # Create an index for faster lookups
    ///     graph.create_index('Prospect', 'geoprovince')
    ///
    /// ```text
    /// # Now this filter will use the index (O(1) instead of O(n))
    /// graph.select('Prospect').where({'geoprovince': 'North Sea'})
    /// ```
    /// ```
    fn create_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        property: &str,
    ) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);
        // In-memory backends use the existing HashMap-based property_indices.
        // On the Disk backend that HashMap would silently OOM on large types
        // (~13M rows × String × Vec = multiple GB of heap rebuilt every load);
        // route there to the persistent mmap-backed PropertyIndex instead.
        let mut persistent_disk = false;
        let mut disk_count = 0usize;
        if let crate::graph::storage::backend::GraphBackend::Disk(dg) = &graph.graph {
            match dg.build_property_index(node_type, property) {
                Ok(n) => {
                    persistent_disk = true;
                    disk_count = n;
                }
                Err(e) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to build persistent property index for {}.{}: {}",
                        node_type, property, e
                    )));
                }
            }
        }
        let unique_values = if persistent_disk {
            // Skip the in-memory HashMap build on disk graphs — the
            // persistent index takes its place. Return disk_count (nodes
            // indexed) as the `unique_values` field for API parity.
            disk_count
        } else {
            graph.create_index(node_type, property)
        };

        let result_dict = PyDict::new(py);
        result_dict.set_item("node_type", node_type)?;
        result_dict.set_item("property", property)?;
        result_dict.set_item("unique_values", unique_values)?;
        result_dict.set_item("persistent", persistent_disk)?;
        result_dict.set_item("created", true)?;

        Ok(result_dict.into())
    }

    /// Drop (remove) an index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     property: The property name
    ///
    /// Returns:
    ///     True if index existed and was removed, False otherwise
    fn drop_index(&mut self, node_type: &str, property: &str) -> PyResult<bool> {
        let removed = get_graph_mut(&mut self.inner).drop_index(node_type, property);
        Ok(removed)
    }

    /// Build a cross-type global index on `property`. Unlike
    /// ``create_index`` (keyed by ``(node_type, property)``), this
    /// indexes EVERY node with a non-empty string value at that
    /// property, regardless of type.
    ///
    /// Enables two agent-friendly patterns:
    ///     * ``MATCH (n {label: 'Norway'})`` — untyped lookup, routes
    ///       through the global index in O(log N).
    ///     * ``graph.search('Norway')`` — returns the top-k nodes by
    ///       that property across all types.
    ///
    /// Disk-backed graphs only. On memory/mapped graphs this is a
    /// no-op that returns 0 — per-type ``create_index`` already covers
    /// the use case at in-memory scale.
    ///
    /// Args:
    ///     property: The property name to index (e.g. 'label', 'title', 'name').
    ///
    /// Returns:
    ///     Dict with ``property``, ``unique_values`` (node count indexed),
    ///     and ``created``.
    fn create_global_index(&mut self, py: Python<'_>, property: &str) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);
        let count = match &graph.graph {
            crate::graph::storage::backend::GraphBackend::Disk(dg) => {
                dg.build_global_property_index(property).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to build global property index for '{}': {}",
                        property, e
                    ))
                })?
            }
            _ => 0,
        };
        let result = PyDict::new(py);
        result.set_item("property", property)?;
        result.set_item("unique_values", count)?;
        result.set_item("created", true)?;
        Ok(result.into())
    }

    /// Search for nodes matching ``text`` on a property (default ``title``).
    ///
    /// Uses the cross-type global index when one has been built — see
    /// ``create_global_index(property)``. Alias-aware: a miss on
    /// ``title`` also tries ``label`` and ``name`` (and ``id``/``nid``/
    /// ``qid`` for the id family), so an index built under one alias
    /// is still hit when queried with another. Tries exact match
    /// first; if none, falls back to prefix match.
    ///
    /// Returns the top ``limit`` results as dicts with ``id`` (node
    /// index), ``type``, ``title``, and ``id_value``.
    ///
    /// Returns an empty list if no global index exists for any alias
    /// of ``property``.
    ///
    /// Example::
    ///
    /// ```text
    /// graph.create_global_index('label')   # or 'title'
    /// hits = graph.search('Norway')
    /// # [{'id': 12345, 'type': 'country', 'title': 'Norway',
    /// #   'id_value': 'Q20'}, ...]
    /// ```
    #[pyo3(signature = (text, *, property="title", limit=10))]
    fn search(
        &self,
        py: Python<'_>,
        text: &str,
        property: &str,
        limit: usize,
    ) -> PyResult<Py<PyAny>> {
        use crate::graph::storage::GraphRead;
        let backend = &self.inner.graph;

        // Alias-aware lookup. Mirrors the matcher's cross-type fast
        // path so `g.search(...)` and `MATCH (n {title: ...})` resolve
        // through the same candidate list.
        let candidates: Vec<&str> = match property {
            "title" => vec!["title", "label", "name"],
            "label" => vec!["label", "title", "name"],
            "name" => vec!["name", "title", "label"],
            "id" => vec!["id", "nid", "qid"],
            "nid" => vec!["nid", "id", "qid"],
            "qid" => vec!["qid", "id", "nid"],
            other => vec![other],
        };

        // Exact match across every candidate name, then prefix
        // fallback on the same names.
        let mut hits: Vec<petgraph::graph::NodeIndex> = Vec::new();
        for name in &candidates {
            if let Some(v) = backend.lookup_by_property_eq_any_type(name, text) {
                if !v.is_empty() {
                    hits = v;
                    break;
                }
            }
        }
        if hits.is_empty() {
            for name in &candidates {
                if let Some(v) = backend.lookup_by_property_prefix_any_type(name, text, limit) {
                    if !v.is_empty() {
                        hits = v;
                        break;
                    }
                }
            }
        }
        hits.truncate(limit);

        let result_list = pyo3::types::PyList::empty(py);
        for idx in hits {
            let Some(node) = backend.node_weight(idx) else {
                continue;
            };
            let dict = PyDict::new(py);
            dict.set_item("id", idx.index())?;
            dict.set_item("type", node.node_type_str(&self.inner.interner))?;
            let title = node.title();
            match title.as_ref() {
                crate::datatypes::values::Value::String(s) => dict.set_item("title", s.as_str())?,
                crate::datatypes::values::Value::Null => dict.set_item("title", py.None())?,
                other => dict.set_item("title", format!("{:?}", other))?,
            }
            let node_id = node.id();
            match node_id.as_ref() {
                crate::datatypes::values::Value::String(s) => {
                    dict.set_item("id_value", s.as_str())?
                }
                crate::datatypes::values::Value::Int64(n) => dict.set_item("id_value", *n)?,
                crate::datatypes::values::Value::Null => dict.set_item("id_value", py.None())?,
                other => dict.set_item("id_value", format!("{:?}", other))?,
            }
            result_list.append(dict)?;
        }
        Ok(result_list.into_any().unbind())
    }

    /// List all existing indexes.
    ///
    /// Returns:
    ///     List of dictionaries with 'type' and 'property' keys
    ///
    /// Example:
    ///     ```python
    ///     indexes = graph.list_indexes()
    ///     for idx in indexes:
    ///         print(f"{idx['type']}.{idx['property']}")
    ///     ```
    fn list_indexes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let indexes = self.inner.list_indexes();

        let result_list = pyo3::types::PyList::empty(py);
        for (node_type, property) in indexes {
            let idx_dict = PyDict::new(py);
            idx_dict.set_item("node_type", node_type)?;
            idx_dict.set_item("property", property)?;
            result_list.append(idx_dict)?;
        }

        Ok(result_list.into())
    }

    /// Check if an index exists.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     property: The property name
    ///
    /// Returns:
    ///     True if index exists, False otherwise
    fn has_index(&self, node_type: &str, property: &str) -> bool {
        self.inner.has_index(node_type, property)
    }

    /// Get statistics about an index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     property: The property name
    ///
    /// Returns:
    ///     Dictionary with index statistics, or None if index doesn't exist
    ///
    /// Example:
    ///     ```python
    ///     stats = graph.index_stats('Prospect', 'geoprovince')
    ///     print(f"Unique values: {stats['unique_values']}")
    ///     print(f"Total entries: {stats['total_entries']}")
    ///     ```
    fn index_stats(&self, py: Python<'_>, node_type: &str, property: &str) -> PyResult<Py<PyAny>> {
        match self.inner.get_index_stats(node_type, property) {
            Some(stats) => {
                let result_dict = PyDict::new(py);
                result_dict.set_item("node_type", node_type)?;
                result_dict.set_item("property", property)?;
                result_dict.set_item("unique_values", stats.unique_values)?;
                result_dict.set_item("total_entries", stats.total_entries)?;
                result_dict.set_item("avg_entries_per_value", stats.avg_entries_per_value)?;
                Ok(result_dict.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Create a range index (B-Tree) on a property for a specific node type.
    ///
    /// Range indexes enable efficient range queries (>, >=, <, <=, BETWEEN)
    /// using ``where()`` with comparison conditions.
    ///
    /// Args:
    ///     node_type: The type of nodes to index.
    ///     property: The property name to index.
    ///
    /// Returns:
    ///     dict with keys: ``type``, ``property``, ``unique_values``, ``created``
    ///
    /// Example:
    ///     ```python
    ///     graph.create_range_index('Person', 'age')
    ///     # Now range queries on age use the B-Tree index:
    ///     result = graph.select('Person').where({'age': {'>': 25}}).collect()
    ///     ```
    fn create_range_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        property: &str,
    ) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);
        let unique_values = graph.create_range_index(node_type, property);

        let result_dict = PyDict::new(py);
        result_dict.set_item("node_type", node_type)?;
        result_dict.set_item("property", property)?;
        result_dict.set_item("unique_values", unique_values)?;
        result_dict.set_item("created", true)?;

        Ok(result_dict.into())
    }

    /// Drop a range index.
    ///
    /// Args:
    ///     node_type: The type of nodes.
    ///     property: The property name.
    ///
    /// Returns:
    ///     True if index existed and was removed, False otherwise.
    fn drop_range_index(&mut self, node_type: &str, property: &str) -> PyResult<bool> {
        let removed = get_graph_mut(&mut self.inner).drop_range_index(node_type, property);
        Ok(removed)
    }

    /// Rebuild all indexes.
    ///
    /// Call this after batch updates to ensure indexes are current.
    ///
    /// Returns:
    ///     Number of indexes rebuilt
    fn rebuild_indexes(&mut self) -> PyResult<usize> {
        let graph = get_graph_mut(&mut self.inner);

        let index_keys: Vec<_> = graph.property_indices.keys().cloned().collect();

        for (node_type, property) in &index_keys {
            graph.create_index(node_type, property);
        }

        Ok(index_keys.len())
    }

    // ========================================================================
    // Composite Index Methods
    // ========================================================================

    /// Create a composite index on multiple properties for efficient multi-field queries.
    ///
    /// Composite indexes are useful when you frequently filter on the same combination
    /// of fields together. They provide O(1) lookup for exact matches on all indexed fields.
    ///
    /// Args:
    ///     node_type: The type of nodes to index
    ///     properties: A list of property names to include in the composite index
    ///
    /// Returns:
    ///     Number of unique value combinations indexed
    ///
    /// Example:
    ///     ```python
    ///     # Create an index for queries filtering on both 'geoprovince' and 'status'
    ///     graph.create_composite_index('Prospect', ['geoprovince', 'status'])
    ///
    /// ```text
    /// # Now this filter is very fast:
    /// graph.select('Prospect').where({
    ///     'geoprovince': 'N3',
    ///     'status': 'Active'
    /// })
    /// ```
    /// ```
    fn create_composite_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        properties: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);

        let props_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        let unique_values = graph.create_composite_index(node_type, &props_refs);
        let result_dict = PyDict::new(py);
        result_dict.set_item("node_type", node_type)?;
        result_dict.set_item("properties", properties)?;
        result_dict.set_item("unique_combinations", unique_values)?;

        Ok(result_dict.into())
    }

    /// Drop a composite index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     properties: The list of property names in the composite index
    ///
    /// Returns:
    ///     True if index existed and was dropped, False otherwise
    fn drop_composite_index(&mut self, node_type: &str, properties: Vec<String>) -> PyResult<bool> {
        let removed = get_graph_mut(&mut self.inner).drop_composite_index(node_type, &properties);
        Ok(removed)
    }

    /// List all composite indexes in the graph.
    ///
    /// Returns:
    ///     A list of dicts with 'type' and 'properties' keys
    fn list_composite_indexes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let indexes = self.inner.list_composite_indexes();

        let result_list = pyo3::types::PyList::empty(py);
        for (node_type, properties) in indexes {
            let idx_dict = PyDict::new(py);
            idx_dict.set_item("node_type", node_type)?;
            idx_dict.set_item("properties", properties)?;
            result_list.append(idx_dict)?;
        }

        Ok(result_list.into())
    }

    /// Check if a composite index exists.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     properties: The list of property names in the composite index
    ///
    /// Returns:
    ///     True if index exists, False otherwise
    fn has_composite_index(&self, node_type: &str, properties: Vec<String>) -> bool {
        self.inner.has_composite_index(node_type, &properties)
    }

    /// Get statistics about a composite index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     properties: The list of property names in the composite index
    ///
    /// Returns:
    ///     Dictionary with index statistics, or None if index doesn't exist
    fn composite_index_stats(
        &self,
        py: Python<'_>,
        node_type: &str,
        properties: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        match self.inner.get_composite_index_stats(node_type, &properties) {
            Some(stats) => {
                let result_dict = PyDict::new(py);
                result_dict.set_item("node_type", node_type)?;
                result_dict.set_item("properties", properties)?;
                result_dict.set_item("unique_combinations", stats.unique_values)?;
                result_dict.set_item("total_entries", stats.total_entries)?;
                result_dict.set_item("avg_entries_per_combination", stats.avg_entries_per_value)?;
                Ok(result_dict.into())
            }
            None => Ok(py.None()),
        }
    }
}
