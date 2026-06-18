// Graph Algorithms #[pymethods] — extracted from mod.rs

use crate::datatypes::values::Value;
use crate::datatypes::{py_in, py_out};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use crate::graph::introspection::reporting::OperationReports;
use crate::graph::schema::{CowSelection, PlanStep};
use crate::graph::storage::lookups;
use crate::graph::{
    centrality_results_to_dataframe, centrality_results_to_py_dict, community_results_to_py,
    KnowledgeGraph, TemporalContext,
};

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Graph Algorithms: Path Finding & Connectivity
    // ========================================================================

    /// Find the shortest path between two nodes.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     A dictionary with:
    ///         - 'path': List of node info dicts along the path
    ///         - 'connections': List of connection types between nodes
    ///         - 'length': Number of hops in the path
    ///     Returns None if no path exists.
    #[pyo3(signature = (source_type, source_id, target_type, target_id, connection_types=None, via_types=None, weight_property=None, timeout_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn shortest_path(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        connection_types: Option<Vec<String>>,
        via_types: Option<Vec<String>>,
        weight_property: Option<String>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(|e: String| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup
            .check_uid(&source_value)
            .ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                )))
            })?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string()).map_err(
                |e: String| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)),
            )?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value).ok_or_else(|| {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                "Target node with id {:?} not found in type '{}'",
                target_value, target_type
            )))
        })?;

        // Find shortest path — Dijkstra when weight_property is set,
        // BFS otherwise.
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        if let Some(prop) = weight_property {
            let result = crate::graph::algorithms::graph_algorithms::shortest_path_weighted(
                &self.inner,
                source_idx,
                target_idx,
                &prop,
                connection_types.as_deref(),
                via_types.as_deref(),
                deadline,
            );
            return Ok(match result {
                Some(wp) => {
                    let dict = PyDict::new(py);
                    let path_list = PyList::empty(py);
                    for &node_idx in &wp.path {
                        if let Some(info) =
                            crate::graph::algorithms::graph_algorithms::get_node_info(
                                &self.inner,
                                node_idx,
                            )
                        {
                            let node_dict = PyDict::new(py);
                            node_dict.set_item("type", &info.node_type)?;
                            node_dict.set_item("title", &info.title)?;
                            node_dict.set_item("id", py_out::value_to_py(py, &info.id)?)?;
                            path_list.append(node_dict)?;
                        }
                    }
                    dict.set_item("path", path_list)?;
                    let connections =
                        crate::graph::algorithms::graph_algorithms::get_path_connections(
                            &self.inner,
                            &wp.path,
                        );
                    let conn_list = PyList::empty(py);
                    for conn in connections {
                        match conn {
                            Some(c) => conn_list.append(&c)?,
                            None => conn_list.append(py.None())?,
                        }
                    }
                    dict.set_item("connections", conn_list)?;
                    dict.set_item("length", wp.path.len().saturating_sub(1))?;
                    dict.set_item("weight", wp.weight)?;
                    dict.into()
                }
                None => py.None(),
            });
        }

        let result = crate::graph::algorithms::graph_algorithms::shortest_path(
            &self.inner,
            source_idx,
            target_idx,
            connection_types.as_deref(),
            via_types.as_deref(),
            deadline,
        );

        match result {
            Some(path_result) => {
                let result_dict = PyDict::new(py);

                // Build path info list
                let path_list = PyList::empty(py);
                for &node_idx in &path_result.path {
                    if let Some(info) = crate::graph::algorithms::graph_algorithms::get_node_info(
                        &self.inner,
                        node_idx,
                    ) {
                        let node_dict = PyDict::new(py);
                        node_dict.set_item("type", &info.node_type)?;
                        node_dict.set_item("title", &info.title)?;
                        node_dict.set_item("id", py_out::value_to_py(py, &info.id)?)?;
                        path_list.append(node_dict)?;
                    }
                }
                result_dict.set_item("path", path_list)?;

                // Build connections list
                let connections = crate::graph::algorithms::graph_algorithms::get_path_connections(
                    &self.inner,
                    &path_result.path,
                );
                let conn_list = PyList::empty(py);
                for conn in connections {
                    match conn {
                        Some(c) => conn_list.append(&c)?,
                        None => conn_list.append(py.None())?,
                    }
                }
                result_dict.set_item("connections", conn_list)?;
                result_dict.set_item("length", path_result.cost)?;

                Ok(result_dict.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Get just the length (hop count) of the shortest path between two nodes.
    ///
    /// This is a lightweight version of shortest_path() that avoids materializing
    /// node data, making it much faster when you only need the distance.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     Hop count (int) when ``weight_property`` is None; total weight
    ///     (float) otherwise. ``None`` if no path exists.
    #[pyo3(signature = (source_type, source_id, target_type, target_id, weight_property=None))]
    fn shortest_path_length(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        weight_property: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        // Use O(1) direct lookup from id_indices (populated during add_nodes)
        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = self
            .inner
            .lookup_by_id_normalized(source_type, &source_value)
            .ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                )))
            })?;

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = self
            .inner
            .lookup_by_id_normalized(target_type, &target_value)
            .ok_or_else(|| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Target node with id {:?} not found in type '{}'",
                    target_value, target_type
                )))
            })?;

        if let Some(prop) = weight_property {
            return Ok(
                match crate::graph::algorithms::graph_algorithms::shortest_path_cost_weighted(
                    &self.inner,
                    source_idx,
                    target_idx,
                    &prop,
                    None,
                    None,
                    None,
                ) {
                    Some(w) => w.into_pyobject(py)?.unbind().into_any(),
                    None => py.None(),
                },
            );
        }

        // Find shortest path cost only (no path reconstruction — faster)
        Ok(
            match crate::graph::algorithms::graph_algorithms::shortest_path_cost(
                &self.inner,
                source_idx,
                target_idx,
            ) {
                Some(c) => c.into_pyobject(py)?.unbind().into_any(),
                None => py.None(),
            },
        )
    }

    /// Batch shortest path lengths — computes distances for multiple pairs at once.
    ///
    /// Much faster than calling shortest_path_length in a loop because it:
    /// 1. Builds the adjacency list once (amortized across all pairs)
    /// 2. Reuses the visited-tracking allocation between queries
    ///
    /// Args:
    ///     node_type: The node type for all source/target nodes
    ///     pairs: List of (source_id, target_id) tuples
    ///
    /// Returns:
    ///     List of distances (None where no path exists), same order as input pairs.
    fn shortest_path_lengths_batch(
        &self,
        py: Python<'_>,
        node_type: &str,
        pairs: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
    ) -> PyResult<Py<PyAny>> {
        // Resolve all node indices up front
        let mut index_pairs = Vec::with_capacity(pairs.len());
        for (src_py, tgt_py) in &pairs {
            let src_val = py_in::py_value_to_value(src_py)?;
            let src_idx = self
                .inner
                .lookup_by_id_normalized(node_type, &src_val)
                .ok_or_else(|| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                        "Node with id {:?} not found in type '{}'",
                        src_val, node_type
                    )))
                })?;

            let tgt_val = py_in::py_value_to_value(tgt_py)?;
            let tgt_idx = self
                .inner
                .lookup_by_id_normalized(node_type, &tgt_val)
                .ok_or_else(|| {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                        "Node with id {:?} not found in type '{}'",
                        tgt_val, node_type
                    )))
                })?;

            index_pairs.push((src_idx, tgt_idx));
        }

        let results = crate::graph::algorithms::graph_algorithms::shortest_path_cost_batch(
            &self.inner,
            &index_pairs,
        );

        let result_list = PyList::empty(py);
        for result in results {
            match result {
                Some(cost) => result_list.append(cost)?,
                None => result_list.append(py.None())?,
            }
        }

        Ok(result_list.into())
    }

    /// Get just the node IDs along the shortest path between two nodes.
    ///
    /// This is a lightweight version of shortest_path() that returns only the
    /// node IDs without full node info, making it faster when you don't need
    /// titles, types, or connection info.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     A list of node IDs along the path, or None if no path exists.
    #[pyo3(signature = (source_type, source_id, target_type, target_id, connection_types=None, via_types=None, timeout_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn shortest_path_ids(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        connection_types: Option<Vec<String>>,
        via_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        // Use O(1) direct lookup from id_indices (populated during add_nodes)
        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = self
            .inner
            .lookup_by_id_normalized(source_type, &source_value)
            .ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                )))
            })?;

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = self
            .inner
            .lookup_by_id_normalized(target_type, &target_value)
            .ok_or_else(|| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Target node with id {:?} not found in type '{}'",
                    target_value, target_type
                )))
            })?;

        // Find shortest path
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        match crate::graph::algorithms::graph_algorithms::shortest_path(
            &self.inner,
            source_idx,
            target_idx,
            connection_types.as_deref(),
            via_types.as_deref(),
            deadline,
        ) {
            Some(path_result) => {
                // Extract just the IDs - no PyDict creation per node
                let ids: Vec<Py<PyAny>> = path_result
                    .path
                    .iter()
                    .filter_map(|&idx| {
                        self.inner
                            .get_node(idx)
                            .and_then(|node| node.get_field_ref("id"))
                            .map(|id| py_out::value_to_py(py, &id).unwrap_or_else(|_| py.None()))
                    })
                    .collect();
                Ok(PyList::new(py, ids)?.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Get just the raw graph indices along the shortest path between two nodes.
    ///
    /// This is the fastest path query - returns only integer indices without
    /// any node data lookup. Use this when you need maximum performance and
    /// will look up node data separately if needed.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     A list of integer node indices along the path, or None if no path exists.
    #[pyo3(signature = (source_type, source_id, target_type, target_id, connection_types=None, via_types=None, timeout_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn shortest_path_indices(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        connection_types: Option<Vec<String>>,
        via_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        // Use O(1) direct lookup from id_indices (populated during add_nodes)
        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = self
            .inner
            .lookup_by_id_normalized(source_type, &source_value)
            .ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                )))
            })?;

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = self
            .inner
            .lookup_by_id_normalized(target_type, &target_value)
            .ok_or_else(|| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Target node with id {:?} not found in type '{}'",
                    target_value, target_type
                )))
            })?;

        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        // Find shortest path and return raw indices
        match crate::graph::algorithms::graph_algorithms::shortest_path(
            &self.inner,
            source_idx,
            target_idx,
            connection_types.as_deref(),
            via_types.as_deref(),
            deadline,
        ) {
            Some(path_result) => {
                let indices: Vec<usize> = path_result.path.iter().map(|idx| idx.index()).collect();
                Ok(PyList::new(py, indices)?.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Find all paths between two nodes up to a maximum number of hops.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///     max_hops: Maximum path length to search (default: 5)
    ///     max_results: Stop after finding this many paths (default: unlimited).
    ///                  Use this to prevent OOM on dense graphs.
    ///
    /// Returns:
    ///     A list of path dictionaries, each with 'path', 'connections', and 'length'
    #[pyo3(signature = (source_type, source_id, target_type, target_id, max_hops=None, max_results=None, connection_types=None, via_types=None, timeout_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn all_paths(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        max_hops: Option<usize>,
        max_results: Option<usize>,
        connection_types: Option<Vec<String>>,
        via_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let max_hops = max_hops.unwrap_or(5);

        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(|e: String| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup
            .check_uid(&source_value)
            .ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                )))
            })?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string()).map_err(
                |e: String| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)),
            )?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value).ok_or_else(|| {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                "Target node with id {:?} not found in type '{}'",
                target_value, target_type
            )))
        })?;

        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        // Find all paths
        let paths = crate::graph::algorithms::graph_algorithms::all_paths(
            &self.inner,
            source_idx,
            target_idx,
            max_hops,
            max_results,
            connection_types.as_deref(),
            via_types.as_deref(),
            deadline,
        );

        // Convert to Python output
        let result_list = PyList::empty(py);
        for path in paths {
            let path_dict = PyDict::new(py);

            // Build path info list
            let path_list = PyList::empty(py);
            for &node_idx in &path {
                if let Some(info) =
                    crate::graph::algorithms::graph_algorithms::get_node_info(&self.inner, node_idx)
                {
                    let node_dict = PyDict::new(py);
                    node_dict.set_item("type", &info.node_type)?;
                    node_dict.set_item("title", &info.title)?;
                    node_dict.set_item("id", py_out::value_to_py(py, &info.id)?)?;
                    path_list.append(node_dict)?;
                }
            }
            path_dict.set_item("path", path_list)?;

            // Build connections list
            let connections = crate::graph::algorithms::graph_algorithms::get_path_connections(
                &self.inner,
                &path,
            );
            let conn_list = PyList::empty(py);
            for conn in connections {
                match conn {
                    Some(c) => conn_list.append(&c)?,
                    None => conn_list.append(py.None())?,
                }
            }
            path_dict.set_item("connections", conn_list)?;
            path_dict.set_item("length", path.len().saturating_sub(1))?;

            result_list.append(path_dict)?;
        }

        Ok(result_list.into())
    }

    /// Find all connected components in the graph.
    ///
    /// Args:
    ///     weak: If True (default), find weakly connected components (treating graph as undirected).
    ///           If False, find strongly connected components (respecting edge direction).
    ///
    /// Returns:
    ///     A list of components, each component is a list of node info dicts.
    ///     Components are sorted by size (largest first).
    #[pyo3(signature = (weak=None, titles_only=None))]
    fn connected_components(
        &self,
        py: Python<'_>,
        weak: Option<bool>,
        titles_only: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let weak = weak.unwrap_or(true);

        let components = if weak {
            crate::graph::algorithms::graph_algorithms::weakly_connected_components(
                &self.inner,
                None,
            )
            .map_err(|e: String| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?
        } else {
            crate::graph::algorithms::graph_algorithms::connected_components(&self.inner)
        };

        if titles_only.unwrap_or(false) {
            // Lightweight: return [[title1, title2, ...], [...], ...]
            // Creates only string objects — no PyDicts per node
            let result_list = PyList::empty(py);
            for component in components {
                let comp_list = PyList::empty(py);
                for &node_idx in &component {
                    if let Some(node) = self.inner.get_node(node_idx) {
                        let node_title = node.title();
                        let title_str = match &*node_title {
                            Value::String(s) => s.as_str(),
                            _ => "",
                        };
                        comp_list.append(title_str)?;
                    }
                }
                result_list.append(comp_list)?;
            }
            Ok(result_list.into())
        } else {
            // Full: return [[{type, title, id}, ...], [...], ...]
            let key_type = pyo3::intern!(py, "type");
            let key_title = pyo3::intern!(py, "title");
            let key_id = pyo3::intern!(py, "id");

            let result_list = PyList::empty(py);
            for component in components {
                let comp_list = PyList::empty(py);
                for &node_idx in &component {
                    if let Some(node) = self.inner.get_node(node_idx) {
                        let node_dict = PyDict::new(py);
                        node_dict.set_item(key_type, node.node_type_str(&self.inner.interner))?;
                        let node_title = node.title();
                        let title_str = match &*node_title {
                            Value::String(s) => s.as_str(),
                            _ => "",
                        };
                        node_dict.set_item(key_title, title_str)?;
                        node_dict.set_item(key_id, py_out::value_to_py(py, &node.id())?)?;
                        comp_list.append(node_dict)?;
                    }
                }
                result_list.append(comp_list)?;
            }
            Ok(result_list.into())
        }
    }

    /// Check if two nodes are connected (directly or indirectly).
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     True if the nodes are connected, False otherwise
    fn are_connected(
        &self,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(|e: String| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup
            .check_uid(&source_value)
            .ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                )))
            })?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string()).map_err(
                |e: String| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)),
            )?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value).ok_or_else(|| {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                "Target node with id {:?} not found in type '{}'",
                target_value, target_type
            )))
        })?;

        Ok(crate::graph::algorithms::graph_algorithms::are_connected(
            &self.inner,
            source_idx,
            target_idx,
        ))
    }

    /// Get the degree (number of connections) for nodes in the current selection.
    ///
    /// Returns:
    ///     A dictionary mapping node titles to their degree counts
    fn degrees(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result_dict = PyDict::new(py);

        let level_count = self.cursor.selection.get_level_count();
        if level_count == 0 {
            return Ok(result_dict.into());
        }

        let level = self
            .cursor
            .selection
            .get_level(level_count - 1)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No selection level")
            })?;

        for node_idx in level.iter_node_indices() {
            if let Some(info) =
                crate::graph::algorithms::graph_algorithms::get_node_info(&self.inner, node_idx)
            {
                let degree =
                    crate::graph::algorithms::graph_algorithms::node_degree(&self.inner, node_idx);
                result_dict.set_item(&info.title, degree)?;
            }
        }

        Ok(result_dict.into())
    }

    // ========================================================================
    // Centrality Algorithms
    // ========================================================================

    /// Calculate betweenness centrality for nodes in the graph.
    ///
    /// Betweenness centrality measures how often a node lies on the shortest path
    /// between other pairs of nodes. Higher values indicate nodes that are more
    /// important as "bridges" in the network.
    ///
    /// Args:
    ///     normalized: If True, normalize scores to [0, 1] range (default: True)
    ///     sample_size: Optional number of source nodes to sample for faster computation
    ///                  on large graphs. If None, uses all nodes.
    ///     top_k: Return only the top K nodes by centrality (default: all)
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'title', 'id', and 'score' keys,
    ///     sorted by score descending.
    ///
    /// Example:
    ///     ```python
    ///     # Find the most central nodes (bridges)
    ///     central_nodes = graph.betweenness_centrality(top_k=10)
    ///     for node in central_nodes:
    ///         print(f"{node['title']}: {node['score']:.4f}")
    ///     ```
    #[pyo3(signature = (normalized=None, sample_size=None, connection_types=None, top_k=None, as_dict=None, timeout_ms=None, to_df=None))]
    #[allow(clippy::too_many_arguments)]
    fn betweenness_centrality(
        &self,
        py: Python<'_>,
        normalized: Option<bool>,
        sample_size: Option<usize>,
        connection_types: Option<Vec<String>>,
        top_k: Option<usize>,
        as_dict: Option<bool>,
        timeout_ms: Option<u64>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let normalized = normalized.unwrap_or(true);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let inner = Arc::clone(&self.inner);
        let results = py
            .detach(move || {
                crate::graph::algorithms::graph_algorithms::betweenness_centrality(
                    &inner,
                    normalized,
                    sample_size,
                    connection_types.as_deref(),
                    None,
                    deadline,
                )
            })
            .map_err(|e: String| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

        if to_df.unwrap_or(false) {
            centrality_results_to_dataframe(py, &self.inner, results, top_k)
        } else if as_dict.unwrap_or(false) {
            centrality_results_to_py_dict(py, &self.inner, results, top_k)
        } else {
            {
                let view = crate::graph::pyapi::result_view::ResultView::from_centrality(
                    &self.inner,
                    results,
                    top_k,
                );
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    /// Calculate PageRank centrality for nodes in the graph.
    ///
    /// PageRank measures the importance of nodes based on the structure of
    /// incoming links. Originally developed by Google for ranking web pages.
    ///
    /// Args:
    ///     damping_factor: Probability of following a link (default: 0.85)
    ///     max_iterations: Maximum number of iterations (default: 100)
    ///     tolerance: Convergence threshold (default: 1e-6)
    ///     top_k: Return only the top K nodes by centrality (default: all)
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'title', 'id', and 'score' keys,
    ///     sorted by score descending.
    ///
    /// Example:
    ///     ```python
    ///     # Find the most important nodes by PageRank
    ///     important_nodes = graph.pagerank(top_k=10)
    ///     for node in important_nodes:
    ///         print(f"{node['title']}: {node['score']:.6f}")
    ///     ```
    #[pyo3(signature = (damping_factor=None, max_iterations=None, tolerance=None, connection_types=None, top_k=None, as_dict=None, timeout_ms=None, to_df=None))]
    #[allow(clippy::too_many_arguments)]
    fn pagerank(
        &self,
        py: Python<'_>,
        damping_factor: Option<f64>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
        connection_types: Option<Vec<String>>,
        top_k: Option<usize>,
        as_dict: Option<bool>,
        timeout_ms: Option<u64>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let damping = damping_factor.unwrap_or(0.85);
        let max_iter = max_iterations.unwrap_or(100);
        let tol = tolerance.unwrap_or(1e-6);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let inner = Arc::clone(&self.inner);
        let results = py
            .detach(move || {
                crate::graph::algorithms::graph_algorithms::pagerank(
                    &inner,
                    damping,
                    max_iter,
                    tol,
                    connection_types.as_deref(),
                    None,
                    deadline,
                )
            })
            .map_err(|e: String| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

        if to_df.unwrap_or(false) {
            centrality_results_to_dataframe(py, &self.inner, results, top_k)
        } else if as_dict.unwrap_or(false) {
            centrality_results_to_py_dict(py, &self.inner, results, top_k)
        } else {
            {
                let view = crate::graph::pyapi::result_view::ResultView::from_centrality(
                    &self.inner,
                    results,
                    top_k,
                );
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    /// Calculate degree centrality for nodes in the graph.
    ///
    /// Degree centrality simply counts the number of connections each node has.
    /// This is the simplest centrality measure but often effective.
    ///
    /// Args:
    ///     normalized: If True, normalize by (n-1) for values in [0, 1] (default: True)
    ///     top_k: Return only the top K nodes by centrality (default: all)
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'title', 'id', and 'score' keys,
    ///     sorted by score descending.
    ///
    /// Example:
    ///     ```python
    ///     # Find the most connected nodes
    ///     connected_nodes = graph.degree_centrality(top_k=10)
    ///     ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (normalized=None, connection_types=None, top_k=None, as_dict=None, timeout_ms=None, to_df=None))]
    fn degree_centrality(
        &self,
        py: Python<'_>,
        normalized: Option<bool>,
        connection_types: Option<Vec<String>>,
        top_k: Option<usize>,
        as_dict: Option<bool>,
        timeout_ms: Option<u64>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let normalized = normalized.unwrap_or(true);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let inner = Arc::clone(&self.inner);
        let results = py
            .detach(move || {
                crate::graph::algorithms::graph_algorithms::degree_centrality(
                    &inner,
                    normalized,
                    connection_types.as_deref(),
                    None,
                    deadline,
                )
            })
            .map_err(|e: String| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

        if to_df.unwrap_or(false) {
            centrality_results_to_dataframe(py, &self.inner, results, top_k)
        } else if as_dict.unwrap_or(false) {
            centrality_results_to_py_dict(py, &self.inner, results, top_k)
        } else {
            {
                let view = crate::graph::pyapi::result_view::ResultView::from_centrality(
                    &self.inner,
                    results,
                    top_k,
                );
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    /// Calculate closeness centrality for nodes in the graph.
    ///
    /// Closeness centrality measures how close a node is to all other nodes,
    /// based on the sum of shortest path distances. Higher values mean the
    /// node can reach other nodes more quickly.
    ///
    /// Args:
    ///     normalized: If True, adjust for disconnected components (default: True)
    ///     sample_size: Optional number of source nodes to sample for faster computation
    ///                  on large graphs. If None, uses all nodes.
    ///     top_k: Return only the top K nodes by centrality (default: all)
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'title', 'id', and 'score' keys,
    ///     sorted by score descending.
    ///
    /// Example:
    ///     ```python
    ///     # Find nodes that are "closest" to all others
    ///     close_nodes = graph.closeness_centrality(top_k=10)
    ///     # Approximate with sampling for large graphs
    ///     close_nodes = graph.closeness_centrality(sample_size=100, top_k=10)
    ///     ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (normalized=None, sample_size=None, connection_types=None, top_k=None, as_dict=None, timeout_ms=None, to_df=None))]
    fn closeness_centrality(
        &self,
        py: Python<'_>,
        normalized: Option<bool>,
        sample_size: Option<usize>,
        connection_types: Option<Vec<String>>,
        top_k: Option<usize>,
        as_dict: Option<bool>,
        timeout_ms: Option<u64>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let normalized = normalized.unwrap_or(true);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let inner = Arc::clone(&self.inner);
        let results = py
            .detach(move || {
                crate::graph::algorithms::graph_algorithms::closeness_centrality(
                    &inner,
                    normalized,
                    sample_size,
                    connection_types.as_deref(),
                    None,
                    deadline,
                )
            })
            .map_err(|e: String| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

        if to_df.unwrap_or(false) {
            centrality_results_to_dataframe(py, &self.inner, results, top_k)
        } else if as_dict.unwrap_or(false) {
            centrality_results_to_py_dict(py, &self.inner, results, top_k)
        } else {
            {
                let view = crate::graph::pyapi::result_view::ResultView::from_centrality(
                    &self.inner,
                    results,
                    top_k,
                );
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    // ========================================================================
    // Community Detection
    // ========================================================================

    /// Detect communities using the Louvain modularity optimization algorithm.
    ///
    /// Args:
    ///     weight_property: Edge property to use as weight (default: all edges weight 1.0)
    ///     resolution: Resolution parameter (default: 1.0). Higher values produce more communities.
    ///
    /// Returns:
    ///     dict with 'communities' (dict mapping community_id -> list of node dicts),
    ///     'modularity' (float), and 'num_communities' (int)
    #[pyo3(signature = (weight_property=None, resolution=None, connection_types=None, timeout_ms=None))]
    fn louvain_communities(
        &self,
        py: Python<'_>,
        weight_property: Option<String>,
        resolution: Option<f64>,
        connection_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let res = resolution.unwrap_or(1.0);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        let result = crate::graph::algorithms::graph_algorithms::louvain_communities(
            &self.inner,
            weight_property.as_deref(),
            res,
            connection_types.as_deref(),
            None,
            deadline,
        )
        .map_err(|e: String| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?;
        community_results_to_py(py, &self.inner, result)
    }

    /// Detect communities using label propagation.
    ///
    /// Each node adopts the most frequent label among its neighbors until convergence.
    ///
    /// Args:
    ///     max_iterations: Maximum iterations before stopping (default: 100)
    ///
    /// Returns:
    ///     dict with 'communities' (dict mapping community_id -> list of node dicts),
    ///     'modularity' (float), and 'num_communities' (int)
    #[pyo3(signature = (max_iterations=None, connection_types=None, timeout_ms=None))]
    fn label_propagation(
        &self,
        py: Python<'_>,
        max_iterations: Option<usize>,
        connection_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let max_iter = max_iterations.unwrap_or(100);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        let result = crate::graph::algorithms::graph_algorithms::label_propagation(
            &self.inner,
            max_iter,
            connection_types.as_deref(),
            None,
            deadline,
        )
        .map_err(|e: String| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?;
        community_results_to_py(py, &self.inner, result)
    }

    // ========================================================================
    // Subgraph Extraction Methods
    // ========================================================================

    /// Expand the current selection by N hops.
    ///
    /// This performs a breadth-first expansion from all currently selected nodes,
    /// including all nodes within the specified number of hops. The expansion
    /// considers edges in both directions (undirected).
    ///
    /// Args:
    ///     hops: Number of hops to expand (default: 1)
    ///
    /// Returns:
    ///     A new KnowledgeGraph with the expanded selection
    ///
    /// Example:
    ///     ```python
    ///     # Start with a single field and expand to include connected nodes
    ///     expanded = graph.select('Field').where({'name': 'EKOFISK'}).expand(hops=2)
    ///     ```
    #[pyo3(signature = (hops=None))]
    fn expand(&self, hops: Option<usize>) -> PyResult<Self> {
        let hops = hops.unwrap_or(1);
        self.derive_with(|inner, cursor| {
            // node_count of the last selection level (avoids allocation).
            let last_count = |sel: &crate::graph::schema::CowSelection| {
                sel.get_level(sel.get_level_count().saturating_sub(1))
                    .map(|l| l.node_count())
                    .unwrap_or(0)
            };
            let estimated = last_count(&cursor.selection);
            crate::graph::mutation::subgraph::expand_selection(inner, &mut cursor.selection, hops)
                .map_err(|e: String| {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?;
            let actual = last_count(&cursor.selection);
            cursor
                .selection
                .add_plan_step(PlanStep::new("EXPAND", None, estimated).with_actual_rows(actual));
            Ok(())
        })
    }

    /// Extract the currently selected nodes into a new independent subgraph.
    ///
    /// Creates a new KnowledgeGraph containing only the nodes in the current
    /// selection and all edges that connect those nodes. The new graph is
    /// completely independent from the original.
    ///
    /// Returns:
    ///     A new KnowledgeGraph containing only the selected nodes and their
    ///     connecting edges
    ///
    /// Example:
    ///     ```python
    ///     # Extract a subgraph of a specific region
    ///     subgraph = (
    ///         graph.select('Field')
    ///         .where({'region': 'North Sea'})
    ///         .expand(hops=2)
    ///         .to_subgraph()
    ///     )
    ///     # Save the subgraph for later use
    ///     subgraph.save('north_sea_region.kgl')
    ///     ```
    fn to_subgraph(&self) -> PyResult<Self> {
        let extracted =
            crate::graph::mutation::subgraph::extract_subgraph(&self.inner, &self.cursor.selection)
                .map_err(|e: String| {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?;

        Ok(KnowledgeGraph {
            inner: Arc::new(extracted),
            cursor: crate::graph::CursorState::new(),
            embedder: None,
            default_timeout_ms: None,
            default_max_rows: None,
            lifecycle: crate::graph::GraphLifecycle::detached(),
        })
    }

    /// Save the current selection as an independent subgraph file.
    ///
    /// Equivalent to `kg.to_subgraph().save(path)` but as a single call
    /// that does not bind the intermediate in-memory graph to a local
    /// variable. Output is a v3 binary file that reloads via
    /// ``kglite.load(path)`` (or ``load(path, storage='disk')`` for disk
    /// mode). All edges between selected nodes are included; node and
    /// edge properties round-trip byte-for-byte.
    ///
    /// Args:
    ///     path: Destination path for the subgraph file.
    ///
    /// Example:
    ///     ```python
    ///     # Articles + their authors, written to disk for later use.
    ///     (
    ///         kg.select('Article')
    ///           .expand(hops=1, type='AUTHORED_BY')
    ///           .save_subset('articles_with_authors.kgl')
    ///     )
    ///     ```
    fn save_subset(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let inner = self.inner.clone();
        let selection = self.cursor.selection.clone();
        let path_owned = path.to_string();
        py.detach(move || {
            crate::graph::mutation::subgraph_streaming::save_subset(
                &inner,
                &selection,
                std::path::Path::new(&path_owned),
                None,
            )
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
    }

    /// [DEBUG] Save a subgraph filtered by edge type, bypassing the
    /// fluent selection chain.
    ///
    /// Equivalent in spirit to `expand(hops=1, edge_types=...)` followed
    /// by `save_subset` — but `expand` does not currently take an
    /// edge-type filter, so this method drives Pass A directly to build
    /// the kept-nodes set, then routes the resulting node set through
    /// the existing in-memory save path.
    ///
    /// Disk-backed sources only. The kept set is exactly the union of
    /// endpoints of every edge whose connection type matches one of
    /// `edge_types` — useful for "all (a)-[:E]->(b) and just those
    /// nodes" filters where `expand` would over-pull.
    ///
    /// Returns scan + save stats as a dict so callers can time the
    /// component steps without re-loading the source.
    #[pyo3(signature = (path, edge_types))]
    fn _save_subset_filtered_by_edge_type(
        &self,
        py: Python<'_>,
        path: &str,
        edge_types: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        use crate::graph::mutation::subgraph_streaming::{
            pass_a_scan, save_subset_streaming_disk, SubsetSpec,
        };
        use crate::graph::storage::backend::GraphBackend;
        use std::collections::HashMap;

        let edge_type_keys_u64: Vec<u64> = edge_types
            .iter()
            .map(|s| crate::graph::storage::interner::InternedKey::from_str(s).as_u64())
            .collect();
        let spec = SubsetSpec {
            edge_types: Some(
                edge_types
                    .iter()
                    .map(|s| crate::graph::storage::interner::InternedKey::from_str(s))
                    .collect(),
            ),
        };

        let inner = self.inner.clone();
        let path_owned = path.to_string();
        let edge_filter = edge_type_keys_u64.clone();
        let (scan_secs, kept_node_count, kept_edge_count, group_secs, save_secs) =
            py.detach(move || -> PyResult<(f64, u64, u64, f64, f64)> {
                let disk = match &inner.graph {
                    GraphBackend::Disk(dg) => dg.as_ref(),
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "_save_subset_filtered_by_edge_type requires a disk-backed source.",
                        ));
                    }
                };

                // Pass A: sequential edge scan → kept_nodes bitset.
                let scan_t0 = std::time::Instant::now();
                let pass_a = pass_a_scan(disk, &spec);
                let scan_secs = scan_t0.elapsed().as_secs_f64();

                // Group kept nodes by source type, in source-id order.
                // type_indices slices are already sorted, so the resulting
                // Vec per type is sorted too — required by the streaming
                // pipeline for sequential ColumnStore reads.
                let group_t0 = std::time::Instant::now();
                let mut kept_per_type: HashMap<String, Vec<u32>> = HashMap::new();
                for (type_name, type_nodes) in inner.type_indices.iter() {
                    let mut kept: Vec<u32> = Vec::new();
                    for nid in type_nodes.iter() {
                        let id = nid.index();
                        if pass_a.kept_nodes.get(id) {
                            kept.push(id as u32);
                        }
                    }
                    if !kept.is_empty() {
                        kept_per_type.insert(type_name.to_string(), kept);
                    }
                }
                let group_secs = group_t0.elapsed().as_secs_f64();

                // Streaming materialize + save: dest is disk-mode from
                // the start, no in-memory petgraph copy.
                let save_t0 = std::time::Instant::now();
                save_subset_streaming_disk(
                    &inner,
                    &kept_per_type,
                    Some(&edge_filter),
                    std::path::Path::new(&path_owned),
                )
                .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
                let save_secs = save_t0.elapsed().as_secs_f64();

                Ok((
                    scan_secs,
                    pass_a.stats.kept_node_count,
                    pass_a.stats.kept_edge_count,
                    group_secs,
                    save_secs,
                ))
            })?;

        let dict = PyDict::new(py);
        dict.set_item("pass_a_scan_secs", scan_secs)?;
        dict.set_item("group_per_type_secs", group_secs)?;
        dict.set_item("save_secs", save_secs)?;
        dict.set_item("kept_node_count", kept_node_count)?;
        dict.set_item("kept_edge_count", kept_edge_count)?;
        Ok(dict.into())
    }

    /// Variant of `_save_subset_filtered_by_edge_type` that produces
    /// the *induced* subgraph: the kept-node set is still computed from
    /// edges whose connection type matches one of `edge_types` (Pass A),
    /// but the output keeps **every** edge between any two kept nodes,
    /// not just the filter edge type itself.
    ///
    /// Useful for "all relations between articles and their authors":
    /// `_save_subset_induced_by_edge_type(out, ["P50"])` keeps all P50
    /// author edges *and* article→article citations (P2860), authorship-
    /// shaped edges of other types (P2093 stated-author, P98 editor),
    /// human→human relations between coauthors, etc. — every edge in
    /// the source whose source and target are both endpoints of some
    /// P50 edge.
    ///
    /// Output is bigger than the simple filter (often 2–5× more edges)
    /// but matches the "subgraph induced by these node-set" intuition.
    /// Disk-backed sources only.
    #[pyo3(signature = (path, edge_types))]
    fn _save_subset_induced_by_edge_type(
        &self,
        py: Python<'_>,
        path: &str,
        edge_types: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        use crate::graph::mutation::subgraph_streaming::{
            pass_a_scan, save_subset_streaming_disk, SubsetSpec,
        };
        use crate::graph::storage::backend::GraphBackend;
        use std::collections::HashMap;

        let spec = SubsetSpec {
            edge_types: Some(
                edge_types
                    .iter()
                    .map(|s| crate::graph::storage::interner::InternedKey::from_str(s))
                    .collect(),
            ),
        };

        let inner = self.inner.clone();
        let path_owned = path.to_string();
        let (scan_secs, kept_node_count, group_secs, save_secs, kept_edge_count_written) = py
            .detach(move || -> PyResult<(f64, u64, f64, f64, u64)> {
                let disk = match &inner.graph {
                    GraphBackend::Disk(dg) => dg.as_ref(),
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "_save_subset_induced_by_edge_type requires a disk-backed source.",
                        ));
                    }
                };

                let scan_t0 = std::time::Instant::now();
                let pass_a = pass_a_scan(disk, &spec);
                let scan_secs = scan_t0.elapsed().as_secs_f64();

                let group_t0 = std::time::Instant::now();
                let mut kept_per_type: HashMap<String, Vec<u32>> = HashMap::new();
                for (type_name, type_nodes) in inner.type_indices.iter() {
                    let mut kept: Vec<u32> = Vec::new();
                    for nid in type_nodes.iter() {
                        let id = nid.index();
                        if pass_a.kept_nodes.get(id) {
                            kept.push(id as u32);
                        }
                    }
                    if !kept.is_empty() {
                        kept_per_type.insert(type_name.to_string(), kept);
                    }
                }
                let group_secs = group_t0.elapsed().as_secs_f64();

                // Edge filter is None — keeps every edge whose endpoints
                // are both in the kept-node set (the rank index drops
                // edges whose endpoints are out of scope automatically).
                let save_t0 = std::time::Instant::now();
                save_subset_streaming_disk(
                    &inner,
                    &kept_per_type,
                    None,
                    std::path::Path::new(&path_owned),
                )
                .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
                let save_secs = save_t0.elapsed().as_secs_f64();

                // We don't know the written edge count without re-scanning;
                // report 0 here. Stats consumers can ask the loaded graph.
                Ok((
                    scan_secs,
                    pass_a.stats.kept_node_count,
                    group_secs,
                    save_secs,
                    0,
                ))
            })?;

        let dict = PyDict::new(py);
        dict.set_item("pass_a_scan_secs", scan_secs)?;
        dict.set_item("group_per_type_secs", group_secs)?;
        dict.set_item("save_secs", save_secs)?;
        dict.set_item("kept_node_count", kept_node_count)?;
        // kept_edge_count for induced mode is "all edges among kept nodes" —
        // not known without re-scanning. Caller can query the output graph.
        let _ = kept_edge_count_written;
        Ok(dict.into())
    }

    /// Get statistics about the subgraph that would be extracted.
    ///
    /// Returns information about what would be included in a subgraph extraction
    /// without actually creating the subgraph. Useful for understanding the
    /// scope of an extraction before committing to it.
    ///
    /// Returns:
    ///     A dictionary with:
    ///         - 'node_count': Total number of nodes
    ///         - 'edge_count': Total number of edges
    ///         - 'node_types': Dict of node type -> count
    ///         - 'connection_types': Dict of connection type -> count
    fn subgraph_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = crate::graph::mutation::subgraph::get_subgraph_stats(
            &self.inner,
            &self.cursor.selection,
        )
        .map_err(|e: String| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?;

        let result_dict = PyDict::new(py);
        result_dict.set_item("node_count", stats.node_count)?;
        result_dict.set_item("edge_count", stats.edge_count)?;

        let node_types_dict = PyDict::new(py);
        for (node_type, count) in &stats.node_types {
            node_types_dict.set_item(node_type, count)?;
        }
        result_dict.set_item("node_types", node_types_dict)?;

        let conn_types_dict = PyDict::new(py);
        for (conn_type, count) in &stats.connection_types {
            conn_types_dict.set_item(conn_type, count)?;
        }
        result_dict.set_item("connection_types", conn_types_dict)?;

        Ok(result_dict.into())
    }

    /// [DEBUG] Phase 2 spike for `save_subset`: drives Pass A of the
    /// streaming disk-to-disk subgraph filter and returns scan stats.
    ///
    /// Walks the source disk graph's `edge_endpoints.bin` sequentially,
    /// filters by `edge_types`, and builds the kept-nodes bitset. No
    /// output files are written. The leading underscore marks this as an
    /// unstable API surface; the public `save_subset` ships in a later
    /// phase.
    ///
    /// Errors when the source graph is not disk-backed — Memory and
    /// Mapped graphs already have a fast `to_subgraph().save()` path and
    /// the streaming pipeline is gated to disk only.
    ///
    /// Returns:
    ///     {
    ///         'kept_node_count':       u64,
    ///         'kept_edge_count':       u64,
    ///         'total_edge_count':      u64,
    ///         'scan_duration_secs':    f64,
    ///     }
    #[pyo3(signature = (edge_types=None, kept_edges_out=None))]
    fn _scan_edges_filtered(
        &self,
        py: Python<'_>,
        edge_types: Option<Vec<String>>,
        kept_edges_out: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        use crate::graph::mutation::subgraph_streaming::{
            pass_a_scan, pass_a_scan_to_file, RankIndex, SubsetSpec,
        };
        use crate::graph::storage::backend::GraphBackend;

        let disk = match &self.inner.graph {
            GraphBackend::Disk(dg) => dg.as_ref(),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "_scan_edges_filtered requires a disk-backed graph (storage='disk').",
                ));
            }
        };

        let edge_type_keys = edge_types.map(|names| {
            names
                .into_iter()
                .map(|s| crate::graph::storage::interner::InternedKey::from_str(&s))
                .collect::<Vec<_>>()
        });
        let spec = SubsetSpec {
            edge_types: edge_type_keys,
        };

        let dict = PyDict::new(py);
        match kept_edges_out {
            None => {
                let result = pass_a_scan(disk, &spec);
                dict.set_item("kept_node_count", result.stats.kept_node_count)?;
                dict.set_item("kept_edge_count", result.stats.kept_edge_count)?;
                dict.set_item("total_edge_count", result.stats.total_edge_count)?;
                dict.set_item("scan_duration_secs", result.stats.scan_duration_secs)?;
            }
            Some(path_str) => {
                let path = std::path::Path::new(path_str);
                let result = pass_a_scan_to_file(disk, &spec, path).map_err(|e: String| {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?;

                // Build the rank index from the bitset we got back —
                // exercises the Phase 3 primitive end-to-end on real
                // disk data, and surfaces kept_count for differential
                // checks against `count_ones`.
                let rank = RankIndex::from_bitset(result.kept_nodes);

                dict.set_item("kept_node_count", result.stats.kept_node_count)?;
                dict.set_item("kept_edge_count", result.stats.kept_edge_count)?;
                dict.set_item("total_edge_count", result.stats.total_edge_count)?;
                dict.set_item("scan_duration_secs", result.stats.scan_duration_secs)?;
                dict.set_item("kept_edges_path", result.kept_edges_path.to_string_lossy())?;
                dict.set_item("kept_edge_records", result.kept_edge_records)?;
                dict.set_item("rank_kept_count", rank.kept_count() as u64)?;
            }
        }
        Ok(dict.into())
    }
}
