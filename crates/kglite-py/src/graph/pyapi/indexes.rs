use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::{get_graph_mut, KnowledgeGraph};

#[pymethods]
impl KnowledgeGraph {
    /// Create an equality index on a node type's property and report what it serves.
    fn create_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        property: &str,
    ) -> PyResult<Py<PyAny>> {
        self.check_durable_owner()?;
        let graph = get_graph_mut(&mut self.inner);
        // The checked build Cypher `CREATE INDEX` shares: backend routing
        // (persistent mmap index on disk, where a heap HashMap would OOM on a
        // large type), the refusals, and whether queries will read the index.
        let built = graph
            .create_property_index_checked(node_type, property)
            .map_err(|e| match e {
                kglite_core::api::PropertyIndexError::Refused(message) => {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
                }
                kglite_core::api::PropertyIndexError::Build(message) => {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to build persistent property index for {}.{}: {}",
                        node_type, property, message
                    ))
                }
            })?;

        let result_dict = PyDict::new(py);
        result_dict.set_item("node_type", node_type)?;
        result_dict.set_item("property", property)?;
        result_dict.set_item("unique_values", built.unique_values)?;
        result_dict.set_item("persistent", built.persistent)?;
        result_dict.set_item("created", built.created)?;
        result_dict.set_item("serves_lookups", built.serves_lookups)?;
        result_dict.set_item("not_serving", built.not_serving)?;
        result_dict.set_item("node_type_known", built.node_type_known)?;
        self.commit_wal()?;

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
        self.check_durable_owner()?;
        let removed = get_graph_mut(&mut self.inner)
            .drop_index(node_type, property)
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
        self.commit_wal()?;
        Ok(removed)
    }

    /// Build a cross-type global index on `property`. Unlike
    /// ``create_index`` (keyed by ``(node_type, property)``), this
    /// indexes EVERY node with a non-empty string value at that
    /// property, regardless of type.
    ///
    /// Enables two agent-friendly patterns:
    ///     * ``MATCH (n {title: 'Norway'})`` — untyped lookup, routes
    ///       through the global index in O(log N).
    ///     * ``graph.search('Norway')`` — returns the top-k nodes by
    ///       that property across all types.
    ///
    /// A bundle on a structurally resolved name (``name``, ``type``,
    /// ``node_type``, ``label``) holds stored values only, so Cypher declines
    /// it and scans; ``search()`` still reads it.
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
        let count = match &mut graph.graph {
            kglite_core::api::storage::GraphBackend::Disk(dg) => {
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
    /// Uses the cross-type global index when one is built and current, and
    /// scans otherwise. Alias-aware: a miss on ``title`` also tries ``label``
    /// and ``name`` (and ``id``/``nid``/``qid`` for the id family). Tries
    /// exact match first, then prefix.
    ///
    /// Returns the top ``limit`` results as dicts with ``id`` (node
    /// index), ``type``, ``title``, and ``id_value``.
    ///
    /// Example:
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
        use kglite_core::api::GraphRead;
        let backend = &self.inner.graph;
        // Direct GraphRead traversal — hold the disk arena guard while
        // borrowed node weights live (arena protocol; no-op in memory/mapped).
        let _arena_guard = self.inner.begin_read_pass();

        // Index-or-scan lives in the core so every binding resolves the same
        // candidate set the matcher does, and so a declining disk bundle
        // cannot turn a search into a silent empty list.
        let hits = self.inner.search_by_property(text, property, limit);

        let result_list = pyo3::types::PyList::empty(py);
        for idx in hits {
            let Some(node) = backend.node_view(idx) else {
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

    /// List the equality indexes, in-memory and persistent disk-backed.
    ///
    /// Example:
    ///     ```python
    ///     indexes = graph.list_indexes()
    ///     for idx in indexes:
    ///         print(f"{idx['node_type']}.{idx['property']}")
    ///     ```
    fn list_indexes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // The state-aware lister, so a `defer_index_rebuild` load's declared
        // indexes are listed rather than silently absent. It is a *listing* —
        // `has_index` stays on the built stores (see
        // `DirGraph::list_indexes_with_state`).
        let indexes = self.inner.list_indexes_with_state();

        let result_list = pyo3::types::PyList::empty(py);
        for (node_type, property, state) in indexes {
            let idx_dict = PyDict::new(py);
            // `state` is Neo4j's word for "was it built" — `ONLINE` for every
            // index this listing can see. Whether a query then *reads* it is a
            // separate question, and one an index on a type-string name
            // (`label`, `type`, `node_type`) answers no to; a deferred one answers no because it has not
            // been built yet.
            let serves = self.inner.index_serves_lookups(&node_type, &property);
            idx_dict.set_item("node_type", node_type)?;
            idx_dict.set_item("property", property)?;
            idx_dict.set_item("state", state.as_str())?;
            idx_dict.set_item("serves_lookups", serves)?;
            idx_dict.set_item("persistent", false)?;
            result_list.append(idx_dict)?;
        }
        for (node_type, property) in self.inner.list_persistent_indexes() {
            let idx_dict = PyDict::new(py);
            let serves = self.inner.index_serves_lookups(&node_type, &property);
            idx_dict.set_item("node_type", node_type)?;
            idx_dict.set_item("property", property)?;
            idx_dict.set_item("state", "ONLINE")?;
            idx_dict.set_item("serves_lookups", serves)?;
            idx_dict.set_item("persistent", true)?;
            result_list.append(idx_dict)?;
        }

        Ok(result_list.into())
    }

    /// Check if an equality index exists, in-memory or persistent disk-backed.
    fn has_index(&self, node_type: &str, property: &str) -> bool {
        self.inner.has_any_index(node_type, property)
    }

    /// Get statistics about an index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     property: The property name
    ///
    /// Returns:
    ///     Dictionary with index statistics, or None when no in-memory
    ///     equality index exists — a disk-backed persistent index reports None
    ///
    /// Example:
    ///     ```python
    ///     stats = graph.index_stats('Proposal', 'geoprovince')
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
    ///     dict with keys: ``node_type``, ``property``, ``unique_values``,
    ///     ``created``
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
        self.check_durable_owner()?;
        let graph = get_graph_mut(&mut self.inner);
        graph
            .reject_secondary_only_index_type(node_type)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let unique_values = graph.declare_range_index(node_type, property);
        self.commit_wal()?;

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
        self.check_durable_owner()?;
        let removed = get_graph_mut(&mut self.inner).drop_range_index(node_type, property);
        self.commit_wal()?;
        Ok(removed)
    }

    /// Rebuild the in-memory equality indexes. Range, composite and
    /// disk-backed persistent indexes are left untouched.
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

    /// Create a composite index on multiple properties for efficient multi-field queries.
    ///
    /// Composite indexes are useful when you frequently filter on the same combination
    /// of fields together. They provide O(1) lookup for exact matches on all indexed fields.
    ///
    /// The order of ``properties`` is not significant: the index is stored under
    /// its property names sorted, and ``list_composite_indexes()`` and
    /// ``SHOW INDEXES`` report that spelling.
    ///
    /// Args:
    ///     node_type: The type of nodes to index
    ///     properties: A list of property names to include in the composite index
    ///
    /// Returns:
    ///     Dict with ``node_type``, ``properties``, and
    ///     ``unique_combinations`` (count of indexed combinations)
    ///
    /// Example:
    ///     ```python
    ///     # Create an index for queries filtering on both 'geoprovince' and 'status'
    ///     graph.create_composite_index('Proposal', ['geoprovince', 'status'])
    ///
    ///     # Now this filter is very fast:
    ///     graph.select('Proposal').where({
    ///         'geoprovince': 'N3',
    ///         'status': 'Active'
    ///     })
    ///     ```
    fn create_composite_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        properties: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        self.check_durable_owner()?;
        let graph = get_graph_mut(&mut self.inner);
        graph
            .reject_secondary_only_index_type(node_type)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let props_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        let unique_values = graph.declare_composite_index(node_type, &props_refs);
        self.commit_wal()?;
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
        self.check_durable_owner()?;
        let removed = get_graph_mut(&mut self.inner).drop_composite_index(node_type, &properties);
        self.commit_wal()?;
        Ok(removed)
    }

    /// List all composite indexes in the graph.
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'properties' and 'state' keys
    fn list_composite_indexes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let indexes = self.inner.list_composite_indexes_with_state();

        let result_list = pyo3::types::PyList::empty(py);
        for (node_type, properties, state) in indexes {
            let idx_dict = PyDict::new(py);
            idx_dict.set_item("node_type", node_type)?;
            idx_dict.set_item("properties", properties)?;
            idx_dict.set_item("state", state.as_str())?;
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

    /// Build a BM25 lexical index over a node type's text or string/null-list property.
    #[pyo3(signature = (node_type, property, auto_refresh_limit = None))]
    fn build_text_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        property: &str,
        auto_refresh_limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);
        // Built off the GIL — tokenizing a corpus is pure CPU over graph memory.
        let report = py
            .detach(|| {
                kglite_core::api::text_indexes::build_text_index(
                    graph,
                    node_type,
                    property,
                    auto_refresh_limit,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let result = PyDict::new(py);
        result.set_item("indexed", report.indexed)?;
        result.set_item("skipped", report.skipped)?;
        result.set_item("terms", report.terms)?;
        Ok(result.into())
    }

    /// Drop the BM25 text index for a property; True if one existed.
    #[pyo3(signature = (node_type, property))]
    fn drop_text_index(&mut self, node_type: &str, property: &str) -> bool {
        kglite_core::api::text_indexes::drop_text_index(
            get_graph_mut(&mut self.inner),
            node_type,
            property,
        )
    }

    /// Whether a BM25 text index is built over a property.
    #[pyo3(signature = (node_type, property))]
    fn has_text_index(&self, node_type: &str, property: &str) -> bool {
        kglite_core::api::text_indexes::has_text_index(&self.inner, node_type, property)
    }
}
