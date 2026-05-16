//! KnowledgeGraph #[pymethods]: maintenance + introspection.
//!
//! Part of the Phase 9 split of the kg_methods.rs monolith (5,419 lines
//! single pymethods block). PyO3 merges multiple `#[pymethods] impl`
//! blocks at class-registration time, so the split is purely structural —
//! no runtime impact.

use crate::datatypes::values::Value;
use crate::datatypes::{py_in, py_out};
use crate::graph::core::calculations::StatResult;
use crate::graph::introspection::{
    self,
    reporting::{OperationReport, OperationReports},
};
use crate::graph::schema::{CowSelection, PlanStep};
use crate::graph::storage::GraphRead;
use crate::graph::{
    compare_inner, extract_cypher_param, extract_detail_param, extract_fluent_param, get_graph_mut,
    parse_method_param, KnowledgeGraph, TemporalContext,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{Bound, IntoPyObjectExt};
use std::collections::HashMap;
use std::sync::Arc;

#[pymethods]
impl KnowledgeGraph {
    /// Build ID indices for specified node types for faster node() lookups.
    ///
    /// Call this after loading a graph if you plan to do many ID lookups.
    /// Indices are built lazily anyway, but this pre-builds them.
    ///
    /// Args:
    ///     node_types: List of node types to index. If None, indexes all types.
    ///
    /// Example:
    ///     ```python
    ///     graph.build_id_indices(["User", "Product"])
    ///     ```
    #[pyo3(signature = (node_types=None))]
    fn build_id_indices(&mut self, node_types: Option<Vec<String>>) {
        let graph = Arc::make_mut(&mut self.inner);

        match node_types {
            Some(types) => {
                for node_type in types {
                    graph.build_id_index(&node_type);
                }
            }
            None => {
                // Build for all existing types
                let types: Vec<String> = graph.type_indices.keys().map(|s| s.to_string()).collect();
                for node_type in types {
                    graph.build_id_index(&node_type);
                }
            }
        }
    }

    /// Rebuild all indexes from the current graph state.
    ///
    /// Reconstructs type_indices, property_indices, and composite_indices by
    /// scanning all live nodes. Clears lazy caches (id_indices, connection_types)
    /// so they rebuild on next access.
    ///
    /// Use after bulk mutations (especially Cypher DELETE/REMOVE) to ensure
    /// index consistency.
    ///
    /// Example:
    ///     ```python
    ///     graph.reindex()
    ///     ```
    fn reindex(&mut self) {
        let graph = Arc::make_mut(&mut self.inner);
        graph.reindex();
    }

    /// Convert node properties to columnar storage.
    ///
    /// Properties are moved from per-node storage into per-type column stores,
    /// reducing memory usage for homogeneous typed columns (int64, float64, etc.).
    /// Automatically compacts properties first if not already compacted.
    ///
    /// Example:
    ///     ```python
    ///     graph.enable_columnar()
    ///     assert graph.is_columnar()
    ///     ```
    fn enable_columnar(&mut self) {
        let graph = Arc::make_mut(&mut self.inner);
        graph.enable_columnar();
    }

    /// Convert the graph to disk-backed storage mode.
    ///
    /// Enables columnar storage first (if not already), then builds
    /// CSR (Compressed Sparse Row) edge arrays on disk. Nodes stay
    /// in memory (~40 bytes each), edges are mmap'd from disk.
    ///
    /// This reduces memory usage to ~10% of the in-memory graph for
    /// edge-heavy graphs. Best called after all data is loaded.
    fn enable_disk_mode(&mut self) -> PyResult<()> {
        let graph = Arc::make_mut(&mut self.inner);
        graph
            .enable_disk_mode()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Convert columnar properties back to compact per-node storage.
    ///
    /// This is the inverse of enable_columnar(). Useful before saving
    /// or when columnar storage is no longer needed.
    fn disable_columnar(&mut self) {
        let graph = Arc::make_mut(&mut self.inner);
        graph.disable_columnar();
    }

    /// Move mmap-backed columnar data back to heap memory.
    ///
    /// Useful after deleting nodes when you want data back in RAM for
    /// faster access. Internally rebuilds columnar stores from scratch
    /// (disable_columnar + enable_columnar) with the memory limit
    /// temporarily suspended to prevent re-spilling.
    ///
    /// No-op if the graph is not in columnar mode.
    ///
    /// Example:
    ///     ```python
    ///     graph.unspill()
    ///     info = graph.graph_info()
    ///     assert not info['columnar_is_mapped']
    ///     ```
    fn unspill(&mut self) {
        let graph = Arc::make_mut(&mut self.inner);
        if !graph.is_columnar() {
            return;
        }
        let saved_limit = graph.memory_limit.take();
        graph.disable_columnar();
        graph.enable_columnar();
        graph.memory_limit = saved_limit;
    }

    /// Returns True if any nodes use columnar property storage.
    #[getter]
    fn is_columnar(&self) -> bool {
        self.inner.is_columnar()
    }

    /// Compact the graph by removing tombstones left by node/edge deletions.
    ///
    /// With StableDiGraph, deletions leave holes in the internal storage.
    /// Over time, this wastes memory and degrades iteration performance.
    /// vacuum() rebuilds the graph with contiguous indices, then rebuilds all indexes.
    ///
    /// **Important**: This resets the current selection since node indices change.
    /// Call this between query chains, not in the middle of one.
    ///
    /// Returns:
    ///     dict: Statistics about the compaction:
    ///         - 'nodes_remapped': Number of nodes that were remapped
    ///         - 'tombstones_removed': Number of tombstone slots reclaimed
    ///
    /// Example:
    ///     ```python
    ///     info = graph.graph_info()
    ///     if info['fragmentation_ratio'] > 0.3:
    ///         result = graph.vacuum()
    ///         print(f"Reclaimed {result['tombstones_removed']} slots")
    ///     ```
    fn vacuum(&mut self) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);

        let tombstones_before = graph.graph.node_bound() - graph.graph.node_count();
        let was_columnar = graph.is_columnar();
        let old_to_new = graph.vacuum();
        let nodes_remapped = old_to_new.len();

        // Reset selection — indices have changed
        if nodes_remapped > 0 {
            self.selection = CowSelection::new();
        }

        Python::attach(|py| {
            let result = PyDict::new(py);
            result.set_item("nodes_remapped", nodes_remapped)?;
            result.set_item("tombstones_removed", tombstones_before)?;
            result.set_item("columnar_rebuilt", was_columnar)?;
            Ok(result.into())
        })
    }

    /// Get diagnostic information about graph storage health.
    ///
    /// Returns a dictionary with storage metrics useful for deciding when
    /// to call vacuum() or reindex().
    ///
    /// Returns:
    ///     dict: Graph health metrics:
    ///         - 'node_count': Number of live nodes
    ///         - 'node_capacity': Upper bound of node indices (includes tombstones)
    ///         - 'node_tombstones': Number of wasted slots from deletions
    ///         - 'edge_count': Number of live edges
    ///         - 'fragmentation_ratio': Ratio of wasted storage (0.0 = clean)
    ///         - 'type_count': Number of distinct node types
    ///         - 'property_index_count': Number of single-property indexes
    ///         - 'composite_index_count': Number of composite indexes
    ///
    /// Example:
    ///     ```python
    ///     info = graph.graph_info()
    ///     if info['fragmentation_ratio'] > 0.3:
    ///         graph.vacuum()
    ///     ```
    fn graph_info(&self) -> PyResult<Py<PyAny>> {
        let info = self.inner.graph_info();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("node_count", info.node_count)?;
            dict.set_item("node_capacity", info.node_capacity)?;
            dict.set_item("node_tombstones", info.node_tombstones)?;
            dict.set_item("edge_count", info.edge_count)?;
            dict.set_item("fragmentation_ratio", info.fragmentation_ratio)?;
            dict.set_item("type_count", info.type_count)?;
            dict.set_item("property_index_count", info.property_index_count)?;
            dict.set_item("composite_index_count", info.composite_index_count)?;
            dict.set_item("format_version", self.inner.save_metadata.format_version)?;
            dict.set_item("library_version", &self.inner.save_metadata.library_version)?;
            // Columnar memory info
            let heap_bytes: usize = self
                .inner
                .column_stores
                .values()
                .map(|s| s.heap_bytes())
                .sum();
            let is_mapped = self.inner.column_stores.values().any(|s| s.is_mapped());
            dict.set_item("columnar_heap_bytes", heap_bytes)?;
            dict.set_item("columnar_is_mapped", is_mapped)?;
            dict.set_item("memory_limit", self.inner.memory_limit)?;
            dict.set_item("columnar_total_rows", info.columnar_total_rows)?;
            dict.set_item("columnar_live_rows", info.columnar_live_rows)?;
            Ok(dict.into())
        })
    }

    /// Configure automatic vacuum after DELETE operations.
    ///
    /// When enabled, the graph automatically compacts itself after Cypher DELETE
    /// operations if the fragmentation ratio exceeds the threshold and there are
    /// more than 100 tombstones.
    ///
    /// Args:
    ///     threshold: A float between 0.0 and 1.0, or None to disable.
    ///         Default is 0.3 (30% fragmentation triggers vacuum).
    ///         Set to None to disable auto-vacuum entirely.
    ///
    /// Example:
    ///     ```python
    ///     graph.set_auto_vacuum(0.2)   # more aggressive — vacuum at 20% fragmentation
    ///     graph.set_auto_vacuum(None)  # disable auto-vacuum
    ///     graph.set_auto_vacuum(0.3)   # restore default
    ///     ```
    #[pyo3(signature = (threshold))]
    fn set_auto_vacuum(&mut self, threshold: Option<f64>) -> PyResult<()> {
        if let Some(t) = threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "threshold must be between 0.0 and 1.0, or None to disable",
                ));
            }
        }
        let graph = get_graph_mut(&mut self.inner);
        graph.auto_vacuum_threshold = threshold;
        Ok(())
    }

    /// Configure automatic memory-pressure spill for columnar storage.
    ///
    /// When a memory limit is set, enable_columnar() will automatically
    /// spill the largest column stores to temporary files on disk when
    /// the total heap usage exceeds the limit.
    ///
    /// Args:
    ///     limit_bytes: Maximum heap bytes for columnar data, or None to disable.
    ///     spill_dir: Directory for spill files. Defaults to system temp dir.
    ///
    /// Example:
    ///     ```python
    ///     graph.set_memory_limit(500_000_000)  # 500 MB limit
    ///     graph.enable_columnar()  # auto-spills if over limit
    ///     graph.set_memory_limit(None)  # disable limit
    ///     ```
    #[pyo3(signature = (limit_bytes, spill_dir=None))]
    fn set_memory_limit(
        &mut self,
        limit_bytes: Option<usize>,
        spill_dir: Option<String>,
    ) -> PyResult<()> {
        let graph = get_graph_mut(&mut self.inner);
        graph.memory_limit = limit_bytes;
        graph.spill_dir = spill_dir.map(std::path::PathBuf::from);
        Ok(())
    }

    /// Set or query read-only mode for the Cypher layer.
    ///
    /// When enabled, all Cypher mutation queries (CREATE, SET, DELETE, REMOVE,
    /// MERGE) are rejected with an error, and `describe()` omits mutation
    /// documentation.  Read-only queries (MATCH, RETURN, CALL, etc.) are
    /// unaffected.
    ///
    /// Args:
    ///     enabled: If True, enable read-only mode. If False, disable it.
    ///              If omitted, return the current state without changing it.
    ///
    /// Returns:
    ///     The current read-only state (after applying the change, if any).
    ///
    /// Example::
    ///
    /// ```text
    /// graph.read_only(True)   # lock the graph
    /// graph.read_only()       # -> True
    /// graph.read_only(False)  # unlock
    /// ```
    #[pyo3(signature = (enabled=None))]
    fn read_only(&mut self, enabled: Option<bool>) -> bool {
        if let Some(v) = enabled {
            let graph = get_graph_mut(&mut self.inner);
            graph.read_only = v;
        }
        self.inner.read_only
    }

    /// Lock the schema: future Cypher mutations (CREATE, SET, MERGE) must
    /// conform to the currently known node types, connection types, and
    /// property types.
    ///
    /// Returns:
    ///     Self for method chaining.
    ///
    /// Example::
    ///
    /// ```text
    /// graph.lock_schema()
    /// graph.cypher("CREATE (p:Typo {name: 'x'})")  # raises RuntimeError
    /// ```
    fn lock_schema(&mut self) -> Self {
        let graph = get_graph_mut(&mut self.inner);
        graph.schema_locked = true;
        self.clone()
    }

    /// Unlock the schema: allow any Cypher mutations without schema validation.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn unlock_schema(&mut self) -> Self {
        let graph = get_graph_mut(&mut self.inner);
        graph.schema_locked = false;
        self.clone()
    }

    /// Whether the schema is currently locked.
    #[getter]
    fn schema_locked(&self) -> bool {
        self.inner.schema_locked
    }

    /// Returns a dict of {node_type: count} using the type index (O(type_count)).
    fn node_type_counts(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            for (node_type, indices) in self.inner.type_indices.iter() {
                dict.set_item(node_type, indices.len())?;
            }
            Ok(dict.into())
        })
    }

    #[pyo3(signature = (indices=None, parent_info=None, include_node_properties=None,
                        flatten_single_parent=true))]
    fn connections(
        &self,
        indices: Option<Vec<usize>>,
        parent_info: Option<bool>,
        include_node_properties: Option<bool>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let connections = crate::graph::core::data_retrieval::get_connections(
            &self.inner,
            &self.selection,
            None,
            indices.as_deref(),
            include_node_properties.unwrap_or(true),
        );
        Python::attach(|py| {
            py_out::level_connections_to_pydict(
                py,
                &connections,
                parent_info,
                flatten_single_parent,
            )
        })
    }

    #[pyo3(signature = (limit=None, indices=None, flatten_single_parent=None))]
    fn titles(
        &self,
        limit: Option<usize>,
        indices: Option<Vec<usize>>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let values = crate::graph::core::data_retrieval::get_property_values(
            &self.inner,
            &self.selection,
            None,
            &["title"],
            indices.as_deref(),
            limit,
        );
        Python::attach(|py| {
            py_out::level_single_values_to_pydict(py, &values, flatten_single_parent)
        })
    }

    /// Returns a string representation of the query execution plan.
    ///
    /// Shows each operation in the query chain with estimated and actual row counts.
    /// Example output: "TYPE_FILTER Prospect (6775 nodes) -> TRAVERSE HAS_ESTIMATE (10954 nodes)"
    fn explain(&self) -> PyResult<String> {
        let plan = self.selection.get_execution_plan();
        if plan.is_empty() {
            return Ok("No query operations recorded".to_string());
        }

        let steps: Vec<String> = plan
            .iter()
            .map(|step| {
                let type_info = step
                    .node_type
                    .as_ref()
                    .map(|t| format!(" {}", t))
                    .unwrap_or_default();
                let rows = step.actual_rows.unwrap_or(step.estimated_rows);
                format!("{}{} ({} nodes)", step.operation, type_info, rows)
            })
            .collect();

        Ok(steps.join(" -> "))
    }

    #[pyo3(signature = (properties, limit=None, indices=None, flatten_single_parent=None))]
    fn get_properties(
        &self,
        properties: Vec<String>,
        limit: Option<usize>,
        indices: Option<Vec<usize>>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let property_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        let values = crate::graph::core::data_retrieval::get_property_values(
            &self.inner,
            &self.selection,
            None,
            &property_refs,
            indices.as_deref(),
            limit,
        );
        Python::attach(|py| py_out::level_values_to_pydict(py, &values, flatten_single_parent))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (property, group_by_parent=None, level_index=None, indices=None, store_as=None, max_length=None, keep_selection=None))]
    fn unique_values(
        &mut self,
        property: String,
        group_by_parent: Option<bool>,
        level_index: Option<usize>,
        indices: Option<Vec<usize>>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let values = crate::graph::core::data_retrieval::get_unique_values(
            &self.inner,
            &self.selection,
            &property,
            level_index,
            group_by_parent.unwrap_or(true),
            indices.as_deref(),
        );

        if let Some(target_property) = store_as {
            let nodes = crate::graph::core::data_retrieval::format_unique_values_for_storage(
                &values, max_length,
            );

            let graph = get_graph_mut(&mut self.inner);

            crate::graph::mutation::maintain::update_node_properties(
                graph,
                &nodes,
                target_property,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            if !keep_selection.unwrap_or(false) {
                self.selection.clear();
            }

            Python::attach(|py| Ok(Py::new(py, self.clone())?.into_any()))
        } else {
            Python::attach(|py| py_out::level_unique_values_to_pydict(py, &values))
        }
    }

    /// Traverse connections to discover related nodes.
    ///
    /// Two modes:
    ///
    /// - **Edge mode** (default): follow graph edges of a given type.
    /// - **Comparison mode** (``method=``): spatial, semantic, or clustering.
    ///
    /// Args:
    ///     connection_type (str): Edge type to follow (e.g. ``'HAS_LICENSEE'``).
    ///         In comparison mode, this is the target node type instead.
    ///     direction (str): ``'outgoing'``, ``'incoming'``, or ``None`` (both).
    ///     target_type (str | list[str]): Filter targets to specific node type(s).
    ///         Useful when a connection type connects to multiple node types.
    ///     where (dict): Property conditions for target nodes — same operators
    ///         as ``.where()`` (``'>'``, ``'contains'``, ``'in'``, etc.).
    ///     where_connection (dict): Property conditions for edge properties.
    ///     sort_target: Sort targets per source. Field name or ``[(field, asc)]``.
    ///     limit (int): Max target nodes per source.
    ///     at (str): Temporal point-in-time filter (``'2005'``).
    ///     during (tuple[str,str]): Temporal range filter (``('2000','2010')``).
    ///     temporal (bool): Override temporal filtering (``False`` = off).
    ///     method: Comparison method — string or dict with settings.
    ///     filter_target (dict): Deprecated alias for ``where``.
    ///     filter_connection (dict): Deprecated alias for ``where_connection``.
    ///
    /// Returns:
    ///     New KnowledgeGraph with traversal results selected.
    ///
    /// Examples::
    ///
    /// ```text
    /// g.select('Field').traverse('HAS_LICENSEE')
    /// g.select('Field').traverse('OF_FIELD', direction='incoming',
    ///     target_type='ProductionProfile')
    /// g.select('Field').traverse('HAS_LICENSEE',
    ///     where={'title': 'Equinor Energy AS'})
    /// g.select('Field').traverse('HAS_LICENSEE', at='2005')
    /// ```
    #[pyo3(signature = (connection_type, level_index=None, direction=None, filter_target=None, filter_connection=None, sort_target=None, limit=None, new_level=None, at=None, during=None, temporal=None, target_type=None, r#where=None, where_connection=None))]
    #[allow(clippy::too_many_arguments)]
    fn traverse(
        &mut self,
        connection_type: String,
        level_index: Option<usize>,
        direction: Option<String>,
        filter_target: Option<&Bound<'_, PyDict>>,
        filter_connection: Option<&Bound<'_, PyDict>>,
        sort_target: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        new_level: Option<bool>,
        at: Option<&str>,
        during: Option<(String, String)>,
        temporal: Option<bool>,
        target_type: Option<&Bound<'_, PyAny>>,
        r#where: Option<&Bound<'_, PyDict>>,
        where_connection: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();

        // Estimate based on current selection (source nodes) - use node_count() to avoid allocation
        let estimated = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        // Resolve where → filter_target alias (error if both provided)
        let effective_filter_target =
            match (filter_target, r#where) {
                (Some(_), Some(_)) => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot use both 'filter_target' and 'where' — they are aliases. Use 'where'.",
                )),
                (Some(ft), None) => Some(ft),
                (None, Some(w)) => Some(w),
                (None, None) => None,
            };

        // Resolve where_connection → filter_connection alias
        let effective_filter_connection = match (filter_connection, where_connection) {
            (Some(_), Some(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot use both 'filter_connection' and 'where_connection' — they are aliases. Use 'where_connection'.",
                ))
            }
            (Some(fc), None) => Some(fc),
            (None, Some(wc)) => Some(wc),
            (None, None) => None,
        };

        // Parse target_type: str → vec![str], list[str] → vec[str]
        let target_types: Option<Vec<String>> = if let Some(tt) = target_type {
            if let Ok(s) = tt.extract::<String>() {
                Some(vec![s])
            } else if let Ok(list) = tt.extract::<Vec<String>>() {
                if list.is_empty() {
                    None
                } else {
                    Some(list)
                }
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "target_type must be a string or list of strings",
                ));
            }
        } else {
            None
        };

        let conditions = if let Some(cond) = effective_filter_target {
            Some(py_in::pydict_to_filter_conditions(cond)?)
        } else {
            None
        };

        let conn_conditions = if let Some(cond) = effective_filter_connection {
            Some(py_in::pydict_to_filter_conditions(cond)?)
        } else {
            None
        };

        let sort_fields = if let Some(spec) = sort_target {
            Some(py_in::parse_sort_fields(spec, None)?)
        } else {
            None
        };

        // Build temporal filter for edge-based traversal
        // Priority: temporal=False > at > during > config+temporal_context
        let temporal_filter = if temporal == Some(false) {
            None
        } else if let Some(at_str) = at {
            let (date, _) = crate::graph::features::timeseries::parse_date_query(at_str)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            self.inner
                .temporal_edge_configs
                .get(&connection_type)
                .map(|configs| {
                    crate::graph::core::traversal::TemporalEdgeFilter::At(configs.clone(), date)
                })
        } else if let Some((start_str, end_str)) = &during {
            let (start, _) = crate::graph::features::timeseries::parse_date_query(start_str)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let (end, _) = crate::graph::features::timeseries::parse_date_query(end_str)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            self.inner
                .temporal_edge_configs
                .get(&connection_type)
                .map(|configs| {
                    crate::graph::core::traversal::TemporalEdgeFilter::During(
                        configs.clone(),
                        start,
                        end,
                    )
                })
        } else {
            // Auto: use config + temporal_context
            match &self.temporal_context {
                TemporalContext::All => None,
                TemporalContext::Today => self
                    .inner
                    .temporal_edge_configs
                    .get(&connection_type)
                    .map(|configs| {
                        let today = chrono::Local::now().date_naive();
                        crate::graph::core::traversal::TemporalEdgeFilter::At(
                            configs.clone(),
                            today,
                        )
                    }),
                TemporalContext::At(d) => self
                    .inner
                    .temporal_edge_configs
                    .get(&connection_type)
                    .map(|configs| {
                        crate::graph::core::traversal::TemporalEdgeFilter::At(configs.clone(), *d)
                    }),
                TemporalContext::During(start, end) => self
                    .inner
                    .temporal_edge_configs
                    .get(&connection_type)
                    .map(|configs| {
                        crate::graph::core::traversal::TemporalEdgeFilter::During(
                            configs.clone(),
                            *start,
                            *end,
                        )
                    }),
            }
        };

        crate::graph::core::traversal::make_traversal(
            &self.inner,
            &mut new_kg.selection,
            connection_type.clone(),
            level_index,
            direction,
            conditions.as_ref(),
            conn_conditions.as_ref(),
            sort_fields.as_ref(),
            limit,
            new_level,
            temporal_filter.as_ref(),
            target_types.as_deref(),
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let actual = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("TRAVERSE", Some(&connection_type), estimated).with_actual_rows(actual),
        );

        Ok(new_kg)
    }

    /// Compare selected nodes against a target type using spatial, semantic,
    /// or clustering methods.
    ///
    /// Examples::
    ///
    /// ```text
    /// g.select('Structure').compare('Well', 'contains')
    /// g.select('Well').compare('Well', {'type': 'distance', 'max_m': 5000})
    /// g.select('Well').compare('Well', {'type': 'text_score', 'property': 'name'})
    /// g.select('Well').compare('Well', {'type': 'cluster', 'k': 5})
    /// ```
    #[pyo3(signature = (target_type, method, *, filter=None, sort=None, limit=None, level_index=None, new_level=None))]
    #[allow(clippy::too_many_arguments)]
    fn compare(
        &mut self,
        target_type: &Bound<'_, PyAny>,
        method: &Bound<'_, PyAny>,
        filter: Option<&Bound<'_, PyDict>>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        level_index: Option<usize>,
        new_level: Option<bool>,
    ) -> PyResult<Self> {
        let _ = (level_index, new_level); // accepted but not yet used
        let mut new_kg = self.clone();

        let estimated = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        // Parse target_type: str → Some(str), list[str] → first element
        let resolved_target: Option<String> = if let Ok(s) = target_type.extract::<String>() {
            Some(s)
        } else if let Ok(list) = target_type.extract::<Vec<String>>() {
            list.into_iter().next()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "target_type must be a string or list of strings",
            ));
        };

        let config = parse_method_param(method)?;

        let conditions = if let Some(cond) = filter {
            Some(py_in::pydict_to_filter_conditions(cond)?)
        } else {
            None
        };

        let sort_fields = if let Some(spec) = sort {
            Some(py_in::parse_sort_fields(spec, None)?)
        } else {
            None
        };

        compare_inner(
            &self.inner,
            &mut new_kg.selection,
            resolved_target.as_deref(),
            &config,
            conditions.as_ref(),
            sort_fields.as_ref(),
            limit,
            estimated,
        )?;

        Ok(new_kg)
    }

    /// Create derived edges from the selection chain into a new
    /// connection type.
    ///
    /// **Important — fluent-chain mutation semantics:** The Cypher
    /// engine (`g.cypher("CREATE ...")`) mutates the graph in place,
    /// but `create_connections` follows the fluent pattern: each
    /// chain step (`select`, `traverse`, ...) clones the graph's
    /// `Arc<DirGraph>` handle, and `create_connections` writes to
    /// that handle. Discarding the return value loses the writes.
    /// Always **assign** the result back to keep mutations:
    ///
    /// ```python
    /// # WRONG — discards the mutated graph:
    /// g.select("Person").traverse("WORKS_AT").create_connections("PERSON_AT")
    ///
    /// # RIGHT — assign to retain mutations:
    /// g = g.select("Person").traverse("WORKS_AT").create_connections("PERSON_AT")
    /// ```
    ///
    /// Or for many cases, the simpler form is to use Cypher with
    /// `add_connections(query=...)`:
    ///
    /// ```python
    /// rows = g.cypher("MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
    ///                 "RETURN DISTINCT p.id AS pid, c.id AS cid").to_df()
    /// g.add_connections(data=rows, connection_type="PERSON_AT", ...)
    /// ```
    #[pyo3(signature = (connection_type, keep_selection=None, conflict_handling=None, properties=None, source_type=None, target_type=None))]
    #[allow(clippy::too_many_arguments)]
    fn create_connections(
        &mut self,
        py: Python<'_>,
        connection_type: String,
        keep_selection: Option<bool>,
        conflict_handling: Option<String>,
        properties: Option<&Bound<'_, PyDict>>,
        source_type: Option<String>,
        target_type: Option<String>,
    ) -> PyResult<Self> {
        // Convert properties PyDict → HashMap<String, Vec<String>>
        let copy_properties = if let Some(dict) = properties {
            let mut map = HashMap::new();
            for (key, value) in dict.iter() {
                let type_name: String = key.extract()?;
                let prop_names: Vec<String> = value.extract()?;
                map.insert(type_name, prop_names);
            }
            Some(map)
        } else {
            None
        };

        // Detect the "chain temp" case before we mutate. When the inner
        // Arc is shared (refcount > 1), the upcoming `Arc::make_mut`
        // will clone, and the mutation lands on the clone rather than
        // any other handle. Users who don't capture the return value
        // (`g.select(...).create_connections(...)` without assigning
        // back) silently lose the mutation. Emit a Python `UserWarning`
        // so the failure mode is at least visible in stderr.
        if Arc::strong_count(&self.inner) > 1 {
            let warning_module = py.import("warnings")?;
            warning_module.call_method1(
                "warn",
                (
                    format!(
                        "create_connections('{}') was called on a chained graph view \
                         (Arc refcount={}). The mutation lands on a temporary clone — \
                         either capture the return value (`g = g.select(...).create_connections('{}')`) \
                         or use the equivalent `add_connections(data=cypher_result, ...)` form. \
                         The original graph variable will NOT see these edges.",
                        connection_type,
                        Arc::strong_count(&self.inner),
                        connection_type,
                    ),
                    py.import("builtins")?.getattr("UserWarning")?,
                    2u32, // stacklevel: blame the user's call site, not this fn
                ),
            )?;
        }

        let graph = get_graph_mut(&mut self.inner);

        let result = crate::graph::mutation::maintain::create_connections(
            graph,
            &self.selection,
            connection_type,
            conflict_handling,
            copy_properties,
            source_type,
            target_type,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            selection: if keep_selection.unwrap_or(false) {
                self.selection.clone()
            } else {
                CowSelection::new()
            },
            reports: self.reports.clone(), // Copy over existing reports
            last_mutation_stats: None,
            embedder: self.embedder.as_ref().map(Arc::clone),
            temporal_context: self.temporal_context.clone(),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
        };

        // Store the report in the new graph
        new_kg.add_report(OperationReport::ConnectionOperation(result));

        // Just return the new KnowledgeGraph
        Ok(new_kg)
    }

    /// Enrich selected (leaf) nodes by copying, renaming, aggregating, or computing
    /// properties from ancestor nodes in the traversal hierarchy.
    ///
    /// The `properties` dict maps source node type → property spec:
    ///   - `{'B': ['prop_a', 'prop_b']}` — copy listed properties as-is
    ///   - `{'B': []}` — copy all properties from B
    ///   - `{'B': {'new_name': 'old_name'}}` — copy with rename
    ///   - `{'B': {'avg_depth': 'mean(depth)'}}` — aggregate (count, sum, mean, min, max, std, collect)
    ///   - `{'B': {'dist': 'distance'}}` — spatial compute (distance, area, perimeter, centroid_lat, centroid_lon)
    #[pyo3(signature = (properties, keep_selection=None))]
    fn add_properties(
        &mut self,
        properties: &Bound<'_, PyDict>,
        keep_selection: Option<bool>,
    ) -> PyResult<Self> {
        use crate::graph::mutation::maintain::{
            add_properties as core_add_properties, PropertySpec,
        };

        // Convert PyDict → HashMap<String, PropertySpec>
        let mut spec_map: HashMap<String, PropertySpec> = HashMap::new();
        for (key, value) in properties.iter() {
            let source_type: String = key.extract()?;

            // Try as list first
            if let Ok(list) = value.extract::<Vec<String>>() {
                if list.is_empty() {
                    spec_map.insert(source_type, PropertySpec::CopyAll);
                } else {
                    spec_map.insert(source_type, PropertySpec::CopyList(list));
                }
            } else if let Ok(dict) = value.cast::<PyDict>() {
                // It's a dict: {target_name: source_expr}
                let mut rename_map: HashMap<String, String> = HashMap::new();
                for (dk, dv) in dict.iter() {
                    let target_name: String = dk.extract()?;
                    let source_expr: String = dv.extract()?;
                    rename_map.insert(target_name, source_expr);
                }
                spec_map.insert(source_type, PropertySpec::RenameMap(rename_map));
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Value for type '{}' must be a list (copy) or dict (rename/aggregate). Got: {:?}",
                    source_type,
                    value.get_type().name()?
                )));
            }
        }

        let graph = get_graph_mut(&mut self.inner);
        let result = core_add_properties(graph, &self.selection, spec_map)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            selection: if keep_selection.unwrap_or(true) {
                self.selection.clone()
            } else {
                CowSelection::new()
            },
            reports: self.reports.clone(),
            last_mutation_stats: None,
            embedder: self.embedder.as_ref().map(Arc::clone),
            temporal_context: self.temporal_context.clone(),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
        };

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("ADD_PROPERTIES", None, result.nodes_updated)
                .with_actual_rows(result.properties_set),
        );

        Ok(new_kg)
    }

    #[pyo3(signature = (property=None, r#where=None, sort=None, limit=None, store_as=None, max_length=None, keep_selection=None))]
    #[allow(clippy::too_many_arguments)]
    fn collect_children(
        &mut self,
        property: Option<&str>,
        r#where: Option<&Bound<'_, PyDict>>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let property_name = property.unwrap_or("title");

        // Apply filtering and sorting if needed
        let mut filtered_kg = self.clone();

        if let Some(where_dict) = r#where {
            let conditions = py_in::pydict_to_filter_conditions(where_dict)?;
            let sort_fields = match sort {
                Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
                None => None,
            };

            crate::graph::core::filtering::filter_nodes(
                &self.inner,
                &mut filtered_kg.selection,
                conditions,
                sort_fields,
                limit,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        } else if let Some(spec) = sort {
            let sort_fields = py_in::parse_sort_fields(spec, None)?;

            crate::graph::core::filtering::sort_nodes(
                &self.inner,
                &mut filtered_kg.selection,
                sort_fields,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            if let Some(max) = limit {
                crate::graph::core::filtering::limit_nodes_per_group(
                    &self.inner,
                    &mut filtered_kg.selection,
                    max,
                )
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            }
        } else if let Some(max) = limit {
            crate::graph::core::filtering::limit_nodes_per_group(
                &self.inner,
                &mut filtered_kg.selection,
                max,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }

        // Generate the property lists with titles already included
        let property_groups = crate::graph::core::traversal::get_children_properties(
            &filtered_kg.inner,
            &filtered_kg.selection,
            property_name,
        );

        // If store_as is not provided, return the properties as a dictionary
        if store_as.is_none() {
            // Format for dictionary display
            let dict_pairs =
                crate::graph::core::traversal::format_for_dictionary(&property_groups, max_length);

            return Python::attach(|py| py_out::string_pairs_to_pydict(py, &dict_pairs));
        }

        // Format for storage
        let nodes = crate::graph::core::traversal::format_for_storage(&property_groups, max_length);

        let graph = get_graph_mut(&mut self.inner);

        let result = crate::graph::mutation::maintain::update_node_properties(
            graph,
            &nodes,
            store_as.unwrap(),
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            selection: if keep_selection.unwrap_or(false) {
                self.selection.clone()
            } else {
                CowSelection::new()
            },
            reports: self.reports.clone(),
            last_mutation_stats: None,
            embedder: self.embedder.as_ref().map(Arc::clone),
            temporal_context: self.temporal_context.clone(),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
        };

        // Store the report
        new_kg.add_report(OperationReport::NodeOperation(result));

        // Return the updated graph (no report in return value)
        Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
    }

    #[pyo3(signature = (property, level_index=None, group_by=None))]
    fn statistics(
        &self,
        property: &str,
        level_index: Option<usize>,
        group_by: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        // group_by: compute statistics grouped by a property value
        if let Some(group_prop) = group_by {
            let nodes = crate::graph::core::statistics::collect_selected_nodes(
                &self.selection,
                level_index,
            );
            let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
            for idx in nodes {
                if let Some(node) = self.inner.get_node(idx) {
                    let nt = node.node_type_str(&self.inner.interner);
                    let resolved_group = self.inner.resolve_alias(nt, group_prop);
                    let key = match node.get_field_ref(resolved_group).as_deref() {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Int64(i)) => i.to_string(),
                        Some(v) => format!("{:?}", v),
                        None => "null".to_string(),
                    };
                    let resolved_prop = self.inner.resolve_alias(nt, property);
                    if let Some(val) = node.get_field_ref(resolved_prop) {
                        let num = match &*val {
                            Value::Int64(i) => Some(*i as f64),
                            Value::Float64(f) => Some(*f),
                            Value::UniqueId(u) => Some(*u as f64),
                            _ => None,
                        };
                        if let Some(n) = num {
                            groups.entry(key).or_default().push(n);
                        } else {
                            groups.entry(key).or_default(); // ensure group exists
                        }
                    } else {
                        groups.entry(key).or_default();
                    }
                }
            }
            return Python::attach(|py| {
                let result = PyDict::new(py);
                for (key, values) in &groups {
                    let stats = PyDict::new(py);
                    let count = values.len();
                    stats.set_item("count", count)?;
                    if count > 0 {
                        let sum: f64 = values.iter().sum();
                        let mean = sum / count as f64;
                        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        stats.set_item("sum", sum)?;
                        stats.set_item("mean", mean)?;
                        stats.set_item("min", min)?;
                        stats.set_item("max", max)?;
                        if count > 1 {
                            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                / (count - 1) as f64;
                            stats.set_item("std", variance.sqrt())?;
                        }
                    }
                    result.set_item(key, stats)?;
                }
                Ok(result.into_any().unbind())
            });
        }

        let pairs =
            crate::graph::core::statistics::get_parent_child_pairs(&self.selection, level_index);
        let stats =
            crate::graph::core::statistics::calculate_property_stats(&self.inner, &pairs, property);
        py_out::convert_stats_for_python(stats)
    }

    #[pyo3(signature = (expression, level_index=None, store_as=None, keep_selection=None, aggregate_connections=None))]
    fn calculate(
        &mut self,
        expression: &str,
        level_index: Option<usize>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
        aggregate_connections: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // If we're storing results, we'll need a mutable graph
        if let Some(target_property) = store_as {
            let graph = get_graph_mut(&mut self.inner);

            let process_result = crate::graph::core::calculations::process_equation(
                graph,
                &self.selection,
                expression,
                level_index,
                Some(target_property),
                aggregate_connections,
            );

            match process_result {
                Ok(crate::graph::core::calculations::EvaluationResult::Stored(report)) => {
                    let mut new_kg = KnowledgeGraph {
                        inner: self.inner.clone(),
                        selection: if keep_selection.unwrap_or(false) {
                            self.selection.clone()
                        } else {
                            CowSelection::new()
                        },
                        reports: self.reports.clone(), // Copy existing reports
                        last_mutation_stats: None,
                        embedder: self.embedder.as_ref().map(Arc::clone),
                        temporal_context: self.temporal_context.clone(),
                        default_timeout_ms: self.default_timeout_ms,
                        default_max_rows: self.default_max_rows,
                    };

                    // Store the calculation report
                    new_kg.add_report(OperationReport::CalculationOperation(report));

                    Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
                }
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unexpected result type when storing calculation result",
                )),
                Err(e) => {
                    let error_msg = format!("Error evaluating expression '{}': {}", expression, e);
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error_msg))
                }
            }
        } else {
            // Just computing without storing - no need to modify graph
            let process_result = crate::graph::core::calculations::process_equation(
                &mut (*self.inner).clone(), // Create a temporary clone just for calculation
                &self.selection,
                expression,
                level_index,
                None,
                aggregate_connections,
            );

            // Handle regular errors with descriptive messages
            match process_result {
                Ok(crate::graph::core::calculations::EvaluationResult::Computed(results)) => {
                    // Check for errors
                    let error_count = results.iter().filter(|r| r.error_msg.is_some()).count();
                    if error_count == results.len() && !results.is_empty() {
                        if let Some(first_error) = results.iter().find(|r| r.error_msg.is_some()) {
                            if let Some(error_text) = &first_error.error_msg {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    format!(
                                        "Error in calculation '{}': {}",
                                        expression, error_text
                                    ),
                                ));
                            }
                        }
                    }

                    // Filter out results with errors
                    let valid_results: Vec<StatResult> = results
                        .into_iter()
                        .filter(|r| r.error_msg.is_none())
                        .collect();

                    if valid_results.is_empty() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "No valid results found for expression '{}'",
                            expression
                        )));
                    }

                    py_out::convert_computation_results_for_python(valid_results)
                }
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unexpected result type when computing",
                )),
                Err(e) => {
                    let error_msg = format!("Error evaluating expression '{}': {}", expression, e);
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error_msg))
                }
            }
        }
    }

    #[pyo3(signature = (level_index=None, group_by_parent=None, store_as=None, keep_selection=None, group_by=None))]
    fn count(
        &mut self,
        level_index: Option<usize>,
        group_by_parent: Option<bool>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
        group_by: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        // group_by property: count nodes grouped by a property value
        if let Some(property) = group_by {
            let nodes = crate::graph::core::statistics::collect_selected_nodes(
                &self.selection,
                level_index,
            );
            let mut groups: HashMap<String, usize> = HashMap::new();
            for idx in nodes {
                if let Some(node) = self.inner.get_node(idx) {
                    let resolved = self
                        .inner
                        .resolve_alias(node.node_type_str(&self.inner.interner), property);
                    let key = match node.get_field_ref(resolved).as_deref() {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Int64(i)) => i.to_string(),
                        Some(Value::Float64(f)) => format!("{}", f),
                        Some(Value::Boolean(b)) => b.to_string(),
                        Some(Value::UniqueId(u)) => u.to_string(),
                        Some(Value::DateTime(d)) => d.to_string(),
                        Some(Value::Point { lat, lon }) => format!("({}, {})", lat, lon),
                        Some(Value::Duration {
                            months,
                            days,
                            seconds,
                        }) => format!("duration(M={},D={},S={})", months, days, seconds),
                        Some(Value::NodeRef(idx)) => format!("node#{}", idx),
                        Some(Value::Null) | None => "null".to_string(),
                    };
                    *groups.entry(key).or_insert(0) += 1;
                }
            }
            return Python::attach(|py| {
                let dict = PyDict::new(py);
                for (k, v) in &groups {
                    dict.set_item(k, v)?;
                }
                Ok(dict.into_any().unbind())
            });
        }

        // Default to grouping by parent if we have a nested structure
        let has_multiple_levels = self.selection.get_level_count() > 1;
        // Use the provided group_by_parent if given, otherwise default based on structure
        let use_grouping = group_by_parent.unwrap_or(has_multiple_levels);

        if let Some(target_property) = store_as {
            let graph = get_graph_mut(&mut self.inner);

            let result = match crate::graph::core::calculations::store_count_results(
                graph,
                &self.selection,
                level_index,
                use_grouping,
                target_property,
            ) {
                Ok(report) => report,
                Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
            };

            let mut new_kg = KnowledgeGraph {
                inner: self.inner.clone(),
                selection: if keep_selection.unwrap_or(false) {
                    self.selection.clone()
                } else {
                    CowSelection::new()
                },
                reports: self.reports.clone(), // Copy existing reports
                last_mutation_stats: None,
                embedder: self.embedder.as_ref().map(Arc::clone),
                temporal_context: self.temporal_context.clone(),
                default_timeout_ms: self.default_timeout_ms,
                default_max_rows: self.default_max_rows,
            };

            // Add the report
            new_kg.add_report(OperationReport::CalculationOperation(result));

            Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
        } else if use_grouping {
            // Return counts grouped by parent
            let counts = crate::graph::core::calculations::count_nodes_by_parent(
                &self.inner,
                &self.selection,
                level_index,
            );
            py_out::convert_computation_results_for_python(counts)
        } else {
            // Simple flat count
            let count = crate::graph::core::calculations::count_nodes_in_level(
                &self.selection,
                level_index,
            );
            Python::attach(|py| count.into_py_any(py))
        }
    }

    fn schema_text(&self) -> PyResult<String> {
        let schema_string = introspection::debugging::get_schema_string(&self.inner);
        Ok(schema_string)
    }

    /// Mark a node type as a supporting (child) type of a parent core type.
    ///
    /// Supporting types are hidden from the `describe()` inventory and instead
    /// appear in the `<supporting>` section when the parent type is inspected.
    /// Their capabilities (timeseries, spatial, etc.) bubble up to the parent.
    #[pyo3(signature = (node_type, parent_type))]
    fn set_parent_type(&mut self, node_type: String, parent_type: String) -> PyResult<()> {
        if !self.inner.type_indices.contains_key(&node_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node type '{}' not found",
                node_type
            )));
        }
        if !self.inner.type_indices.contains_key(&parent_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Parent type '{}' not found",
                parent_type
            )));
        }
        let graph = get_graph_mut(&mut self.inner);
        graph.parent_types.insert(node_type, parent_type);
        Ok(())
    }

    /// Return an XML description of this graph for AI agents (progressive disclosure).
    ///
    /// Five independent axes:
    /// - `types` → Node type detail (None=inventory, list=focused)
    /// - `type_search` → Search types by name with neighborhood fan-out
    /// - `connections` → Connection type docs (True=overview, list=deep-dive)
    /// - `cypher` → Cypher language reference (True=compact, list=detailed topics)
    /// - `fluent` → Fluent API reference (True=compact, list=detailed topics)
    ///
    /// `max_pairs` bounds the `(src_type, tgt_type)` breakdown rendered for each
    /// `describe(connections=['T'])` deep-dive. Defaults to 50. Raise it to drill
    /// into wide fan-out connection types (e.g. Wikidata `P31` has 191k distinct
    /// pairs); the head-by-count distribution is emitted first regardless.
    ///
    /// `sample_truncate` controls truncation of long string values in the
    /// `vals=`, sample-node, and edge-attribute fields. Defaults to 40
    /// (current behavior). Pass `None` to disable truncation entirely —
    /// useful when you want full titles in an LLM prompt and have
    /// context-window budget for it.
    ///
    /// When `type_search`, `connections`, `cypher`, or `fluent` is set, only those tracks are returned.
    #[pyo3(signature = (types=None, type_search=None, connections=None, cypher=None, fluent=None, max_pairs=None, sample_truncate=Some(40)))]
    #[allow(clippy::too_many_arguments)]
    fn describe(
        &self,
        types: Option<Vec<String>>,
        type_search: Option<String>,
        connections: Option<&Bound<'_, PyAny>>,
        cypher: Option<&Bound<'_, PyAny>>,
        fluent: Option<&Bound<'_, PyAny>>,
        max_pairs: Option<usize>,
        sample_truncate: Option<usize>,
    ) -> PyResult<String> {
        let conn_detail = extract_detail_param(connections, "connections")?;
        let cypher_detail = extract_cypher_param(cypher)?;
        let fluent_detail = extract_fluent_param(fluent)?;
        introspection::compute_description(
            &self.inner,
            types.as_deref(),
            &conn_detail,
            &cypher_detail,
            &fluent_detail,
            type_search.as_deref(),
            max_pairs,
            sample_truncate,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// One-call codebase exploration: lexical search over Function/Class/
    /// Interface names + docstrings + signatures, 2-hop traversal along
    /// CALLS/USES_TYPE/HAS_METHOD/DEFINES/REFERENCES_FN, and grouped
    /// source slices for the entry points. Returns a markdown string.
    ///
    /// Designed for the "how does X work in this codebase" question that
    /// would otherwise turn into a chain of grep+read calls. Composes
    /// FTS + traversal + source slicing into one Rust-side call.
    ///
    /// - `query` — free-text topic; matched against Function/Class names,
    ///   signatures, and docstrings.
    /// - `max_entities` — top N entry points after lexical ranking (default 10).
    /// - `max_depth` — hops for the neighborhood traversal (default 2).
    /// - `include_source` — whether to include grouped source slices for
    ///   the entry points (default True). Set False for a faster, smaller
    ///   response when you just want the entity list.
    /// - `source_roots` — list of filesystem roots to resolve `file_path`
    ///   properties against. Files matched literally are tried first;
    ///   roots are searched in order. Default: cwd only.
    #[pyo3(signature = (
        query,
        max_entities=10,
        max_depth=2,
        include_source=true,
        source_roots=None,
    ))]
    fn explore(
        &self,
        query: &str,
        max_entities: usize,
        max_depth: usize,
        include_source: bool,
        source_roots: Option<Vec<String>>,
    ) -> PyResult<String> {
        let roots: Vec<std::path::PathBuf> = source_roots
            .unwrap_or_default()
            .into_iter()
            .map(std::path::PathBuf::from)
            .collect();
        let opts = crate::graph::explore::ExploreOptions {
            max_entities,
            max_depth,
            include_source,
            ..Default::default()
        };
        Ok(crate::graph::explore::explore_markdown(
            self, query, &opts, &roots,
        ))
    }

    /// File a bug report to `reported_bugs.md`.
    ///
    /// Appends a timestamped, version-tagged report to the top of the file
    /// (creating it if needed). All inputs are sanitised against code injection.
    ///
    /// - `query` — The Cypher query that triggered the bug.
    /// - `result` — The actual result you got.
    /// - `expected` — The result you expected.
    /// - `description` — Free-text explanation.
    /// - `path` — Optional file path (default: `reported_bugs.md` in cwd).
    #[pyo3(signature = (query, result, expected, description, path=None))]
    fn bug_report(
        &self,
        query: &str,
        result: &str,
        expected: &str,
        description: &str,
        path: Option<&str>,
    ) -> PyResult<String> {
        crate::graph::introspection::bug_report::write_bug_report(
            query,
            result,
            expected,
            description,
            path,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
    }

    /// Return a self-contained XML quickstart for setting up a KGLite MCP server.
    ///
    /// Includes: server code template, core/optional tool descriptions,
    /// and Claude Desktop / Claude Code registration config.
    #[staticmethod]
    fn explain_mcp() -> String {
        introspection::mcp_quickstart()
    }

    fn selection(&self) -> PyResult<String> {
        Ok(introspection::debugging::get_selection_string(
            &self.inner,
            &self.selection,
        ))
    }

    // ================================================================
    // Copy / Clone
    // ================================================================

    /// Create an independent deep copy of this graph.
    ///
    /// Returns a new ``KnowledgeGraph`` that shares no mutable state with
    /// the original. Useful for running mutations without affecting the
    /// source graph.
    fn copy(&self) -> Self {
        KnowledgeGraph {
            inner: Arc::new((*self.inner).clone()),
            selection: CowSelection::new(),
            reports: OperationReports::new(),
            last_mutation_stats: None,
            embedder: self.embedder.as_ref().map(Arc::clone),
            temporal_context: self.temporal_context.clone(),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
        }
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy()
    }
}
