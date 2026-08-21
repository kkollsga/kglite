//! KnowledgeGraph #[pymethods]: maintenance + introspection.
//!
//! Part of the Phase 9 split of the kg_methods.rs monolith (5,419 lines
//! single pymethods block). PyO3 merges multiple `#[pymethods] impl`
//! blocks at class-registration time, so the split is purely structural —
//! no runtime impact.

use crate::datatypes::values::Value;
use crate::datatypes::{py_in, py_out};
use crate::graph::pyapi::kg_core::file_io_err;
use crate::graph::{
    compare_inner, extract_cypher_param, extract_detail_param, extract_fluent_param, get_graph_mut,
    parse_method_param, KnowledgeGraph, TemporalContext,
};
use kglite_core::api::fluent::StatResult;
use kglite_core::api::introspection;
use kglite_core::api::io;
use kglite_core::api::mutation::{OperationReport, OperationReports};
use kglite_core::api::{CowSelection, PlanStep};
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
    fn build_id_indices(&self, py: Python<'_>, node_types: Option<Vec<String>>) {
        // Pre-warm through the IdIndexStore's interior mutability
        // (`ensure_id_index`, the same store the self-healing read path
        // uses) — a cache warm is a read, so no `&mut` / `Arc::make_mut`,
        // which would deep-copy the whole graph when any other handle
        // (fluent clone, frozen view, session) shares the Arc. The index
        // scans are pure Rust; release the GIL for their duration.
        let inner = &self.inner;
        py.detach(|| match node_types {
            Some(types) => {
                for node_type in types {
                    inner.ensure_id_index(&node_type);
                }
            }
            None => {
                // Build for all existing types
                for node_type in inner.type_indices.keys() {
                    inner.ensure_id_index(node_type);
                }
            }
        });
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
        let graph = get_graph_mut(&mut self.inner);
        graph.reindex();
    }

    /// Materialize the in-memory graph as a disk-backed one.
    ///
    /// Builds CSR (Compressed Sparse Row) edge arrays in files and switches
    /// the graph onto the disk backend. Nodes stay in memory (~40 bytes each);
    /// edges are read through the mapping.
    ///
    /// Args:
    ///     path: Directory to materialize the graph into and publish it at.
    ///         The conversion writes there (never through the system temp
    ///         directory), the directory is left in the published, mapped
    ///         state a fresh `kglite.open(path)` reads, and this graph's
    ///         save target becomes `path` — a later bare `save()` writes
    ///         back to it, exactly as `save(path)` rebinds it.
    ///         Omit it only for a throwaway conversion (see below).
    ///
    /// **This does not shrink the process.** It is a conversion, and the
    /// conversion itself adds the on-disk edge structures on top of what is
    /// already resident — expect resident memory to go *up*, not down, for
    /// the lifetime of this process. The in-memory structures it replaces are
    /// freed, but the allocator keeps the pages; call
    /// `kglite.trim_memory()` afterwards to return them to the OS.
    ///
    /// **Where the small footprint actually comes from** is the directory,
    /// reopened in a fresh process: `graph.enable_disk_mode("g.kgl")`, then
    /// `kglite.open("g.kgl")` elsewhere, starts at roughly a tenth of the
    /// in-memory graph's resident size (measured 56 MB vs 492 MB on the same
    /// graph) because the edges are paged in on demand instead of built.
    ///
    /// To run at that footprint from the start — never paying the in-memory
    /// peak at all — build into disk storage directly:
    /// `KnowledgeGraph(storage="disk", path="g.kgl")`.
    ///
    /// **Without `path`** the CSR is materialized into a scratch directory in
    /// the system temp location and deleted when the graph is dropped: a
    /// process-scoped conversion that persists nothing and, on a machine whose
    /// temp directory is small or RAM-backed, writes the whole edge structure
    /// somewhere the caller did not choose. That form warns for exactly that
    /// reason; pass `path` to convert where you mean to.
    ///
    /// `graph_info()` reports the result: `storage_mode` becomes `'disk'` and
    /// `edges_mapped` becomes True. (`columnar_is_mapped` is about property-
    /// column spilling and stays False here — that is disk mode's normal
    /// shape, not a failed conversion.)
    ///
    /// All query methods (Cypher, fluent API, algorithms) work identically
    /// afterwards.
    #[pyo3(signature = (path=None))]
    fn enable_disk_mode(&mut self, py: Python<'_>, path: Option<&str>) -> PyResult<()> {
        // Already converted: the second call cannot "convert" anything, and its
        // core message ("Already in disk mode") arrived through the save
        // dispatch as a *file I/O* error, which it is not. Name the state and
        // the operation the caller actually wants.
        if kglite_core::api::storage::live_storage_mode(&self.inner)
            == kglite_core::api::storage::StorageMode::Disk
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "this graph is already disk-backed — there is nothing to convert. Use \
                 save(path) to publish it into another directory, or save() to publish a new \
                 generation into the one it already has.",
            ));
        }
        // A durable graph's backend is a `RecordingGraph` wrapper, and the
        // conversion cannot unwrap one without silently dropping the capture
        // layer — core refuses it. Say so here in the caller's vocabulary
        // (`kglite.open()` attaches a log by default, so this is the shape a
        // user actually meets) instead of surfacing the internal wrapper name.
        // This also subsumes the diverged-log check `save()` runs: a diverged
        // log implies a durable owner, which is already refused here.
        if self.lifecycle.durable.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "a durable graph cannot be converted to disk mode: disk mode keeps no \
                 write-ahead log, so the conversion would drop the capture layer along with \
                 anything it has buffered. Open the graph with durable=False (or \
                 kglite.load(path)) and convert that handle, or build into disk storage \
                 directly with KnowledgeGraph(storage='disk', path=...).",
            ));
        }
        let Some(path) = path else {
            // Naming the directory is the whole point of the warning: the
            // pathless conversion is legitimate (scratch that dies with the
            // process) but it silently picks the filesystem, and a conversion
            // larger than a small or RAM-backed `/tmp` failed with no hint of
            // where the bytes had gone.
            let message = format!(
                "enable_disk_mode() without a path materializes the edge structures into a \
                 scratch directory under {} and deletes them when this graph is dropped — they \
                 do not survive the process, a reboot, or a temp-directory sweep, and a graph \
                 larger than that filesystem will fail there rather than where you meant to \
                 write. Pass a directory — enable_disk_mode(path='graph.kgl') — to convert into \
                 it and publish it, so kglite.open('graph.kgl') reopens it later.",
                std::env::temp_dir().display()
            );
            let message = std::ffi::CString::new(message).unwrap_or_default();
            PyErr::warn(
                py,
                py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                message.as_c_str(),
                1,
            )?;
            let graph = get_graph_mut(&mut self.inner);
            return graph
                .enable_disk_mode()
                .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>);
        };
        // The path form publishes, so it takes the guard `save()` takes for a
        // handle *derived* from a durable graph (the durable owner itself is
        // already refused above): such a handle holds a fork no log describes,
        // and publishing it is the same hazard as saving it.
        self.check_durable_owner()?;
        let inner = &mut self.inner;
        py.detach(move || io::materialize_disk_graph(inner, path))
            .map_err(|error| match error {
                io::SaveError::Refused(message) => {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
                }
                io::SaveError::Io(message) => file_io_err(std::io::Error::other(message)),
            })?;
        // Same "save as" rebinding `save(path)` does: the directory is now this
        // graph's home, so a later bare `save()` writes back to it.
        self.lifecycle.source_path = Some(std::path::PathBuf::from(path));
        Ok(())
    }

    /// Move mmap-backed property columns back to heap memory.
    ///
    /// Useful after a spill (see `set_memory_limit()`), or after deleting
    /// nodes when you want the data back in RAM for faster access. Rebuilds
    /// every column from the live nodes with the memory limit temporarily
    /// suspended to prevent re-spilling, so rows left behind by deleted nodes
    /// are reclaimed in the same pass. The limit is restored afterwards.
    ///
    /// Example:
    ///     ```python
    ///     graph.unspill()
    ///     info = graph.graph_info()
    ///     assert not info['columnar_is_mapped']
    ///     ```
    fn unspill(&mut self) {
        // Nothing to move, and nothing to fork the graph for: a graph with no
        // column stores at all (one assembled through the direct `add_node`
        // route) has no mmap-backed column to bring home, and rebuilding
        // would be a shape change rather than an unspill.
        if self.inner.column_store_count() == 0 {
            return;
        }
        let graph = get_graph_mut(&mut self.inner);
        graph.rebuild_columns_to_heap();
    }

    /// Compact the graph by removing tombstones left by node/edge deletions.
    ///
    /// With StableDiGraph, deletions leave holes in the internal storage.
    /// Over time, this wastes memory and degrades iteration performance.
    /// vacuum() rebuilds the graph with contiguous indices, then rebuilds all indexes.
    ///
    /// The current selection is **carried through** the compaction: nodes that
    /// survived keep their place in it at their new indices, and nodes the
    /// deletes took are dropped from it. A group whose parent node was deleted
    /// is dropped whole.
    ///
    /// Returns:
    ///     dict: Statistics about the compaction:
    ///         - 'nodes_remapped': Number of nodes carried into the compacted
    ///           graph
    ///         - 'tombstones_removed': Number of free node slots reclaimed
    ///         - 'edge_tombstones_removed': Number of free edge slots
    ///           reclaimed — a relationship-only delete workload leaves these
    ///           and no node tombstones at all
    ///         - 'columnar_rebuilt': True when the pass actually dropped
    ///           property-column rows left behind by deleted nodes (False for
    ///           a vacuum that found nothing to reclaim)
    ///
    /// Example:
    ///     ```python
    ///     info = graph.graph_info()
    ///     if info['fragmentation_ratio'] > 0.3:
    ///         result = graph.vacuum()
    ///         print(f"Reclaimed {result['tombstones_removed']} slots")
    ///     ```
    fn vacuum(&mut self) -> PyResult<Py<PyAny>> {
        // Before the rebuild, not after: captured ops are keyed by
        // `NodeIndex` and a vacuum remaps every index, so ops resolved
        // afterwards would describe the wrong nodes.
        self.commit_wal()?;
        let graph = get_graph_mut(&mut self.inner);

        let info_before = graph.graph_info();
        let tombstones_before = info_before.node_tombstones;
        let edge_tombstones_before = info_before.edge_tombstones;
        // "Were the columnar stores rebuilt?" used to be answered by "is this
        // graph columnar at all?", which was a fair proxy only while a graph
        // could be non-columnar. Every graph owns stores from its first node
        // now, so the proxy would answer `true` for a vacuum that reclaimed
        // nothing. Read the row count instead: a rebuild is exactly what drops
        // the rows deleted nodes left behind.
        let rows_before = info_before.columnar_total_rows;
        let old_to_new = graph.vacuum();
        let info_after = graph.graph_info();
        let columnar_rebuilt = info_after.columnar_total_rows != rows_before;
        let nodes_remapped = old_to_new.len();

        // Carry the selection through, rather than dropping it: the indices
        // moved, but the *set of nodes the caller chose* did not, minus
        // whatever was deleted. The documented reset was a limitation of not
        // having the mapping to hand, never a contract.
        self.cursor.selection.remap_indices(&old_to_new);

        Python::attach(|py| {
            let result = PyDict::new(py);
            result.set_item("nodes_remapped", nodes_remapped)?;
            result.set_item("tombstones_removed", tombstones_before)?;
            result.set_item(
                "edge_tombstones_removed",
                edge_tombstones_before.saturating_sub(info_after.edge_tombstones),
            )?;
            result.set_item("columnar_rebuilt", columnar_rebuilt)?;
            Ok(result.into())
        })
    }

    /// Delete every node still marked provisional — a stub auto-created
    /// to satisfy an edge to a missing node, never promoted by a real
    /// node row — together with its incident edges.
    ///
    /// Resets the current selection (node indices change). Call between
    /// query chains, not mid-chain.
    ///
    /// Returns:
    ///     dict with keys:
    ///         - ``nodes_purged``: provisional stub nodes deleted
    ///         - ``edges_removed``: incident edges removed with them
    fn purge_provisional(&mut self) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);
        let (nodes_purged, edges_removed) =
            kglite_core::api::mutation::purge_provisional_nodes(graph);
        if nodes_purged > 0 {
            self.cursor.selection = CowSelection::new();
        }
        self.commit_wal()?;
        Python::attach(|py| {
            let result = PyDict::new(py);
            result.set_item("nodes_purged", nodes_purged)?;
            result.set_item("edges_removed", edges_removed)?;
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
    ///         - 'edge_capacity': Upper bound of edge indices (includes slots
    ///           freed by relationship deletes)
    ///         - 'edge_tombstones': Wasted edge slots — the only garbage a
    ///           relationship-only delete workload produces
    ///         - 'fragmentation_ratio': Ratio of wasted *node* storage
    ///           (0.0 = clean)
    ///         - 'auto_vacuum_threshold': The configured threshold, or None
    ///           when auto-vacuum is disabled
    ///         - 'auto_vacuums_run': How many times auto-vacuum has fired on
    ///           this graph object
    ///         - 'type_count': Number of distinct node types
    ///         - 'property_index_count': Number of single-property indexes
    ///         - 'composite_index_count': Number of composite indexes
    ///         - 'storage_mode': 'memory', 'mapped' or 'disk' — the backend the
    ///           graph is actually running on, which for an opened graph is the
    ///           mode its checkpoint recorded
    ///         - 'columnar_is_mapped': whether any *property column* is
    ///           file-backed rather than heap-resident. Reports column
    ///           spilling (`set_memory_limit`, or the 'mapped' storage mode),
    ///           **not** disk-mode health — a disk graph reports False here
    ///           unless its columns also spilled
    ///         - 'edges_mapped': whether the edge CSR arrays are memory-mapped
    ///           from files. True on a disk graph whose CSR is materialized;
    ///           always False on the memory and mapped backends, which have no
    ///           CSR
    ///         - 'edge_property_overlay_rows': edges whose properties are held
    ///           in the disk backend's heap mutation overlay rather than the
    ///           mmap-backed base. 0 on every non-disk backend
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
            dict.set_item("edge_capacity", info.edge_capacity)?;
            dict.set_item("edge_tombstones", info.edge_tombstones)?;
            dict.set_item("fragmentation_ratio", info.fragmentation_ratio)?;
            dict.set_item("type_count", info.type_count)?;
            dict.set_item("property_index_count", info.property_index_count)?;
            dict.set_item("composite_index_count", info.composite_index_count)?;
            dict.set_item("format_version", self.inner.save_metadata.format_version)?;
            dict.set_item("library_version", &self.inner.save_metadata.library_version)?;
            dict.set_item("user_schema_version", self.inner.user_schema_version)?;
            // Which backend this graph is actually on. Load-bearing since a
            // checkpoint records its mode: this is how a caller confirms the
            // reopen (or a `storage=` conversion) landed where they expected.
            dict.set_item(
                "storage_mode",
                kglite_core::api::storage::live_storage_mode(&self.inner).as_str(),
            )?;
            // Columnar memory info — from `graph_info()`, not from storage:
            // the column stores belong to the backend (D1 Phase 3) and are not
            // reachable from a binding.
            dict.set_item("columnar_heap_bytes", info.columnar_heap_bytes)?;
            dict.set_item("columnar_is_mapped", info.columnar_is_mapped)?;
            // Edge-storage observability (E1). `columnar_is_mapped` answers
            // "did the memory limit spill the property columns"; these two
            // answer "what shape are the edges in", which is what a caller
            // checking a disk-mode conversion actually wants.
            dict.set_item("edges_mapped", info.edges_mapped)?;
            dict.set_item(
                "edge_property_overlay_rows",
                info.edge_property_overlay_rows,
            )?;
            dict.set_item("memory_limit", self.inner.memory_limit)?;
            dict.set_item("columnar_total_rows", info.columnar_total_rows)?;
            dict.set_item("columnar_live_rows", info.columnar_live_rows)?;
            // Auto-vacuum's own state. `set_auto_vacuum` was write-only until
            // now, so "is it on, and at what threshold?" had no answer, and
            // "did it ever fire?" was only inferable from tombstone counts.
            dict.set_item("auto_vacuum_threshold", self.inner.auto_vacuum_threshold)?;
            dict.set_item("auto_vacuums_run", self.inner.auto_vacuums_run)?;
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

    /// Configure automatic memory-pressure spill for the property columns.
    ///
    /// While a limit is set, the graph spills its largest column stores to
    /// temporary files on disk whenever their total heap usage would exceed
    /// it — checked after each mutating statement and by the consolidation
    /// pass `save()` and `vacuum()` run. `unspill()` brings them back to the
    /// heap; the limit itself survives that and is re-applied by the next
    /// write, so pass `None` first to keep them resident.
    ///
    /// The limit governs **property columns only** — it does not bound nodes,
    /// edges or indexes, and `enable_disk_mode()` is not one of its
    /// checkpoints. `graph_info()['columnar_is_mapped']` is how you see
    /// whether a spill has happened.
    ///
    /// Args:
    ///     limit_bytes: Maximum heap bytes for column data, or None to disable.
    ///     spill_dir: Directory for spill files. Defaults to system temp dir.
    ///
    /// Example:
    ///     ```python
    ///     graph.set_memory_limit(500_000_000)  # 500 MB limit
    ///     graph.set_memory_limit(None)         # disable limit
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
    ///     This same graph (not a copy), so the call can be chained.
    ///
    /// Example::
    ///
    /// ```text
    /// graph.lock_schema()
    /// graph.cypher("CREATE (p:Typo {name: 'x'})")  # raises RuntimeError
    /// ```
    fn lock_schema(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        get_graph_mut(&mut slf.inner).schema_locked = true;
        slf
    }

    /// Unlock the schema: allow any Cypher mutations without schema validation.
    ///
    /// Returns:
    ///     This same graph (not a copy), so the call can be chained.
    fn unlock_schema(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        get_graph_mut(&mut slf.inner).schema_locked = false;
        slf
    }

    /// Whether the schema is currently locked.
    #[getter]
    fn schema_locked(&self) -> bool {
        self.inner.schema_locked
    }

    /// Your own data-model revision, persisted with the graph.
    ///
    /// This is *your* number, not kglite's: the engine stores and returns it
    /// but never interprets it. It exists so a migration script can ask how far
    /// this graph has been migrated. `0` means unversioned, which is also what
    /// a graph saved before this field existed reports.
    ///
    /// Distinct from `graph_info()['format_version']`, which is the `.kgl`
    /// on-disk layout version and belongs to the engine.
    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.user_schema_version
    }

    /// Stamp your data-model revision on the graph. Persisted on the next
    /// `save()`.
    ///
    /// Args:
    ///     version: The revision number. `0` marks the graph unversioned.
    ///
    /// Returns:
    ///     This same graph (not a copy), so the call can be chained.
    ///
    /// Example::
    ///
    /// ```text
    /// graph.cypher("MATCH (p:Person) SET p.email = 'unknown'")
    /// graph.set_schema_version(1).save("graph.kgl")
    /// ```
    #[pyo3(signature = (version))]
    fn set_schema_version(mut slf: PyRefMut<'_, Self>, version: u32) -> PyRefMut<'_, Self> {
        get_graph_mut(&mut slf.inner).user_schema_version = version;
        slf
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
        let connections = kglite_core::api::fluent::get_connections(
            &self.inner,
            &self.cursor.selection,
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
        let values = kglite_core::api::fluent::get_property_values(
            &self.inner,
            &self.cursor.selection,
            None,
            &["title"],
            indices.as_deref(),
            limit,
        );
        Python::attach(|py| {
            py_out::level_single_values_to_pydict(py, &values, flatten_single_parent)
        })
    }

    /// Returns a string representation of the fluent chain's execution plan.
    ///
    /// Fluent methods return a new handle, so the plan lives on the object
    /// they return — call `explain()` on the chain result, not on the graph:
    /// `g.select("Person").where({"city": "Oslo"}).explain()`. Each recorded
    /// operation (`SELECT`, `WHERE`, `TRAVERSE`, `EXPAND`, `VALID_AT`,
    /// `VALID_DURING`, the spatial predicates) is shown with the node count it
    /// produced.
    ///
    /// Example output: "SELECT Person (500 nodes) -> WHERE (42 nodes)"
    ///
    /// This reports fluent chains only. For Cypher, prefix the query with
    /// `EXPLAIN` (plan without executing) or `PROFILE` (execute and return
    /// per-clause statistics on `result.profile`).
    fn explain(&self) -> PyResult<String> {
        let plan = self.cursor.selection.get_execution_plan();
        if plan.is_empty() {
            return Ok(
                "No query operations recorded — explain() reports the fluent chain it is \
                       called on: graph.select(...).where(...).explain(). For Cypher, prefix the \
                       query with EXPLAIN or PROFILE."
                    .to_string(),
            );
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
        let values = kglite_core::api::fluent::get_property_values(
            &self.inner,
            &self.cursor.selection,
            None,
            &property_refs,
            indices.as_deref(),
            limit,
        );
        Python::attach(|py| py_out::level_values_to_pydict(py, &values, flatten_single_parent))
    }

    // Keyword-rich Python surface: each arg is an optional pyo3 kwarg; splitting
    // into a params struct would not change the Python call shape.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (property, group_by_parent=None, level_index=None, indices=None, store_as=None, max_length=None, keep_selection=None))]
    fn unique_values(
        mut slf: PyRefMut<'_, Self>,
        property: String,
        group_by_parent: Option<bool>,
        level_index: Option<usize>,
        indices: Option<Vec<usize>>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let values = kglite_core::api::fluent::get_unique_values(
            &slf.inner,
            &slf.cursor.selection,
            &property,
            level_index,
            group_by_parent.unwrap_or(true),
            indices.as_deref(),
        );

        if let Some(target_property) = store_as {
            let nodes =
                kglite_core::api::fluent::format_unique_values_for_storage(&values, max_length);

            let graph = get_graph_mut(&mut slf.inner);

            kglite_core::api::mutation::update_node_properties(graph, &nodes, target_property)
                .map_err(|e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?;

            if !keep_selection.unwrap_or(false) {
                slf.cursor.selection.clear();
            }

            // `store_as` writes node properties, which the write-ahead log can
            // express — so it has to reach the log like every other logged
            // mutation. It never did: the property landed in memory, the frame
            // was never appended, and a crash before the next checkpoint lost
            // it while every other write around it survived.
            slf.commit_wal()?;

            // The same handle back, not a copy of it: a copy would share the
            // storage but not the write-ahead log, so `g = g.unique_values(...,
            // store_as=...)` on a durable graph used to hand back a handle
            // whose every later write was unlogged.
            Ok(slf.into_pyobject(py)?.into_any().unbind())
        } else {
            py_out::level_unique_values_to_pydict(py, &values)
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
    #[pyo3(signature = (connection_type, level_index=None, direction=None, sort_target=None, limit=None, new_level=None, at=None, during=None, temporal=None, target_type=None, r#where=None, where_connection=None))]
    #[allow(clippy::too_many_arguments)]
    fn traverse(
        &self,
        py: Python<'_>,
        connection_type: String,
        level_index: Option<usize>,
        direction: Option<String>,
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
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

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
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(
                        "target_type must be a string or list of strings".to_string(),
                    ),
                ));
            }
        } else {
            None
        };

        let conditions = if let Some(cond) = r#where {
            Some(py_in::pydict_to_filter_conditions(cond)?)
        } else {
            None
        };

        let conn_conditions = if let Some(cond) = where_connection {
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
            let (date, _) = kglite_core::api::timeseries::parse_date_query(at_str).map_err(
                |e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                },
            )?;
            self.inner
                .temporal_edge_configs
                .get(&connection_type)
                .map(|configs| {
                    kglite_core::api::fluent::TemporalEdgeFilter::At(configs.clone(), date)
                })
        } else if let Some((start_str, end_str)) = &during {
            let (start, _) = kglite_core::api::timeseries::parse_date_query(start_str).map_err(
                |e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                },
            )?;
            let (end, _) = kglite_core::api::timeseries::parse_date_query(end_str).map_err(
                |e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                },
            )?;
            self.inner
                .temporal_edge_configs
                .get(&connection_type)
                .map(|configs| {
                    kglite_core::api::fluent::TemporalEdgeFilter::During(
                        configs.clone(),
                        start,
                        end,
                    )
                })
        } else {
            // Auto: use config + temporal_context
            match &self.cursor.temporal_context {
                TemporalContext::All => None,
                TemporalContext::Today => self
                    .inner
                    .temporal_edge_configs
                    .get(&connection_type)
                    .map(|configs| {
                        let today = chrono::Local::now().date_naive();
                        kglite_core::api::fluent::TemporalEdgeFilter::At(configs.clone(), today)
                    }),
                TemporalContext::At(d) => self
                    .inner
                    .temporal_edge_configs
                    .get(&connection_type)
                    .map(|configs| {
                        kglite_core::api::fluent::TemporalEdgeFilter::At(configs.clone(), *d)
                    }),
                TemporalContext::During(start, end) => self
                    .inner
                    .temporal_edge_configs
                    .get(&connection_type)
                    .map(|configs| {
                        kglite_core::api::fluent::TemporalEdgeFilter::During(
                            configs.clone(),
                            *start,
                            *end,
                        )
                    }),
            }
        };

        // All inputs are converted to pure Rust by now — run the traversal
        // itself off-GIL so other Python threads keep making progress
        // during a large multi-level expansion.
        {
            let inner = &self.inner;
            let selection = &mut new_kg.cursor.selection;
            py.detach(|| {
                kglite_core::api::fluent::make_traversal(
                    inner,
                    selection,
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
            })
            .map_err(|e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;
        }

        let actual = new_kg
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg.cursor.selection.add_plan_step(
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
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        // Parse target_type: str → Some(str), list[str] → first element
        let resolved_target: Option<String> = if let Ok(s) = target_type.extract::<String>() {
            Some(s)
        } else if let Ok(list) = target_type.extract::<Vec<String>>() {
            list.into_iter().next()
        } else {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "target_type must be a string or list of strings".to_string(),
                ),
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
            &mut new_kg.cursor.selection,
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

        let result = kglite_core::api::mutation::create_connections(
            graph,
            &self.cursor.selection,
            connection_type,
            conflict_handling,
            copy_properties,
            source_type,
            target_type,
        )
        .map_err(|e: String| -> PyErr {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
        })?;
        self.commit_wal()?;

        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            cursor: crate::graph::CursorState {
                selection: if keep_selection.unwrap_or(false) {
                    self.cursor.selection.clone()
                } else {
                    CowSelection::new()
                },
                reports: self.cursor.reports.clone(), // Copy over existing reports
                last_mutation_stats: None,
                temporal_context: self.cursor.temporal_context.clone(),
            },
            embedder: self.embedder.as_ref().map(Arc::clone),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
            lifecycle: crate::graph::GraphLifecycle::detached_from(&self.lifecycle),
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
        use kglite_core::api::mutation::{add_properties as core_add_properties, PropertySpec};

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
                return Err(crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Value for type '{}' must be a list (copy) or dict (rename/aggregate). Got: {:?}",
                    source_type,
                    value.get_type().name()?
                ))));
            }
        }

        let graph = get_graph_mut(&mut self.inner);
        let result = core_add_properties(graph, &self.cursor.selection, spec_map).map_err(
            |e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            },
        )?;
        self.commit_wal()?;

        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            cursor: crate::graph::CursorState {
                selection: if keep_selection.unwrap_or(true) {
                    self.cursor.selection.clone()
                } else {
                    CowSelection::new()
                },
                reports: self.cursor.reports.clone(),
                last_mutation_stats: None,
                temporal_context: self.cursor.temporal_context.clone(),
            },
            embedder: self.embedder.as_ref().map(Arc::clone),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
            lifecycle: crate::graph::GraphLifecycle::detached_from(&self.lifecycle),
        };

        // Record plan step
        new_kg.cursor.selection.add_plan_step(
            PlanStep::new("ADD_PROPERTIES", None, result.nodes_updated)
                .with_actual_rows(result.properties_set),
        );

        Ok(new_kg)
    }

    #[pyo3(signature = (property=None, r#where=None, sort=None, limit=None, store_as=None, max_length=None, keep_selection=None))]
    #[allow(clippy::too_many_arguments)]
    fn collect_children(
        &mut self,
        py: Python<'_>,
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

            kglite_core::api::fluent::filter_nodes(
                &self.inner,
                &mut filtered_kg.cursor.selection,
                conditions,
                sort_fields,
                limit,
            )
            .map_err(|e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;
        } else if let Some(spec) = sort {
            let sort_fields = py_in::parse_sort_fields(spec, None)?;

            kglite_core::api::fluent::sort_nodes(
                &self.inner,
                &mut filtered_kg.cursor.selection,
                sort_fields,
            )
            .map_err(|e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

            if let Some(max) = limit {
                kglite_core::api::fluent::limit_nodes_per_group(
                    &self.inner,
                    &mut filtered_kg.cursor.selection,
                    max,
                )
                .map_err(|e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?;
            }
        } else if let Some(max) = limit {
            kglite_core::api::fluent::limit_nodes_per_group(
                &self.inner,
                &mut filtered_kg.cursor.selection,
                max,
            )
            .map_err(|e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;
        }

        // Generate the property lists with titles already included.
        // Pure-Rust scan over the selection — run off-GIL.
        let property_groups = {
            let inner = &filtered_kg.inner;
            let selection = &filtered_kg.cursor.selection;
            py.detach(|| {
                kglite_core::api::fluent::get_children_properties(inner, selection, property_name)
            })
        };

        // If store_as is not provided, return the properties as a dictionary
        if store_as.is_none() {
            // Format for dictionary display
            let dict_pairs =
                kglite_core::api::fluent::format_for_dictionary(&property_groups, max_length);

            return Python::attach(|py| py_out::string_pairs_to_pydict(py, &dict_pairs));
        }

        // Format for storage
        let nodes = kglite_core::api::fluent::format_for_storage(&property_groups, max_length);

        let graph = get_graph_mut(&mut self.inner);

        let result =
            kglite_core::api::mutation::update_node_properties(graph, &nodes, store_as.unwrap())
                .map_err(|e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?;

        // The `store_as` write is node properties, which the write-ahead log
        // can express — so it has to reach the log like every other logged
        // mutation. It never did.
        self.commit_wal()?;

        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            cursor: crate::graph::CursorState {
                selection: if keep_selection.unwrap_or(false) {
                    self.cursor.selection.clone()
                } else {
                    CowSelection::new()
                },
                reports: self.cursor.reports.clone(),
                last_mutation_stats: None,
                temporal_context: self.cursor.temporal_context.clone(),
            },
            embedder: self.embedder.as_ref().map(Arc::clone),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
            lifecycle: crate::graph::GraphLifecycle::detached_from(&self.lifecycle),
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
            let groups = kglite_core::api::fluent::calculate_grouped_property_stats(
                &self.inner,
                &self.cursor.selection,
                property,
                group_prop,
                level_index,
            );
            return Python::attach(|py| {
                let result = PyDict::new(py);
                for (key, group) in &groups {
                    let stats = PyDict::new(py);
                    stats.set_item("count", group.count)?;
                    if let Some(value) = group.sum {
                        stats.set_item("sum", value)?;
                    }
                    if let Some(value) = group.mean {
                        stats.set_item("mean", value)?;
                    }
                    if let Some(value) = group.min {
                        stats.set_item("min", value)?;
                    }
                    if let Some(value) = group.max {
                        stats.set_item("max", value)?;
                    }
                    if let Some(value) = group.std {
                        stats.set_item("std", value)?;
                    }
                    result.set_item(key, stats)?;
                }
                Ok(result.into_any().unbind())
            });
        }

        let pairs =
            kglite_core::api::fluent::get_parent_child_pairs(&self.cursor.selection, level_index);
        let stats =
            kglite_core::api::fluent::calculate_property_stats(&self.inner, &pairs, property);
        py_out::convert_stats_for_python(stats)
    }

    #[pyo3(signature = (expression, level_index=None, store_as=None, keep_selection=None, aggregate_connections=None))]
    fn calculate(
        &mut self,
        py: Python<'_>,
        expression: &str,
        level_index: Option<usize>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
        aggregate_connections: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // If we're storing results, we'll need a mutable graph
        if let Some(target_property) = store_as {
            let graph = get_graph_mut(&mut self.inner);

            // Pure-Rust evaluation + store — run off-GIL.
            let selection = &self.cursor.selection;
            let process_result = py.detach(|| {
                kglite_core::api::fluent::process_equation(
                    &mut *graph,
                    selection,
                    expression,
                    level_index,
                    Some(target_property),
                    aggregate_connections,
                )
            });

            match process_result {
                Ok(kglite_core::api::fluent::EvaluationResult::Stored(report)) => {
                    // Same as `collect_children`: the stored calculation is a
                    // node-property write and belongs in the log.
                    self.commit_wal()?;
                    let mut new_kg = KnowledgeGraph {
                        inner: self.inner.clone(),
                        cursor: crate::graph::CursorState {
                            selection: if keep_selection.unwrap_or(false) {
                                self.cursor.selection.clone()
                            } else {
                                CowSelection::new()
                            },
                            reports: self.cursor.reports.clone(), // Copy existing reports
                            last_mutation_stats: None,
                            temporal_context: self.cursor.temporal_context.clone(),
                        },
                        embedder: self.embedder.as_ref().map(Arc::clone),
                        default_timeout_ms: self.default_timeout_ms,
                        default_max_rows: self.default_max_rows,
                        lifecycle: crate::graph::GraphLifecycle::detached_from(&self.lifecycle),
                    };

                    // Store the calculation report
                    new_kg.add_report(OperationReport::CalculationOperation(report));

                    Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
                }
                Ok(_) => Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(
                        "Unexpected result type when storing calculation result".to_string(),
                    ),
                )),
                Err(e) => {
                    let error_msg = format!("Error evaluating expression '{}': {}", expression, e);
                    Err(crate::error_py::kg_to_pyerr(
                        crate::error::KgError::Argument(error_msg),
                    ))
                }
            }
        } else {
            // Just computing without storing - no need to modify graph.
            // The temporary whole-graph clone + evaluation are pure Rust —
            // run off-GIL. (That `process_equation` demands `&mut DirGraph`
            // even when nothing is stored — forcing this deep clone — is a
            // pre-existing core-signature wart, out of scope here.)
            let inner = &self.inner;
            let selection = &self.cursor.selection;
            let process_result = py.detach(|| {
                kglite_core::api::fluent::process_equation(
                    &mut (**inner).clone(), // Create a temporary clone just for calculation
                    selection,
                    expression,
                    level_index,
                    None,
                    aggregate_connections,
                )
            });

            // Handle regular errors with descriptive messages
            match process_result {
                Ok(kglite_core::api::fluent::EvaluationResult::Computed(results)) => {
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
                        return Err(crate::error_py::kg_to_pyerr(
                            crate::error::KgError::Argument(format!(
                                "No valid results found for expression '{}'",
                                expression
                            )),
                        ));
                    }

                    py_out::convert_computation_results_for_python(valid_results)
                }
                Ok(_) => Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(
                        "Unexpected result type when computing".to_string(),
                    ),
                )),
                Err(e) => {
                    let error_msg = format!("Error evaluating expression '{}': {}", expression, e);
                    Err(crate::error_py::kg_to_pyerr(
                        crate::error::KgError::Argument(error_msg),
                    ))
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
        let _arena_guard = self.inner.begin_read_pass(); // disk arena guard (no-op on memory/mapped)
                                                         // group_by property: count nodes grouped by a property value
        if let Some(property) = group_by {
            let nodes = kglite_core::api::fluent::collect_selected_nodes(
                &self.cursor.selection,
                level_index,
            );
            let mut groups: HashMap<String, usize> = HashMap::new();
            for idx in nodes {
                if let Some(node) = self.inner.node_view(idx) {
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
                        // Phase A.1 — collection / graph-entity
                        // property values delegate to format_value
                        // for the group-by key.
                        Some(other) => crate::datatypes::values::format_value(other),
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
        let has_multiple_levels = self.cursor.selection.get_level_count() > 1;
        // Use the provided group_by_parent if given, otherwise default based on structure
        let use_grouping = group_by_parent.unwrap_or(has_multiple_levels);

        if let Some(target_property) = store_as {
            let graph = get_graph_mut(&mut self.inner);

            let result = match kglite_core::api::fluent::store_count_results(
                graph,
                &self.cursor.selection,
                level_index,
                use_grouping,
                target_property,
            ) {
                Ok(report) => report,
                Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
            };

            // Same as `collect_children`: the stored count is a node-property
            // write and belongs in the log.
            self.commit_wal()?;

            let mut new_kg = KnowledgeGraph {
                inner: self.inner.clone(),
                cursor: crate::graph::CursorState {
                    selection: if keep_selection.unwrap_or(false) {
                        self.cursor.selection.clone()
                    } else {
                        CowSelection::new()
                    },
                    reports: self.cursor.reports.clone(), // Copy existing reports
                    last_mutation_stats: None,
                    temporal_context: self.cursor.temporal_context.clone(),
                },
                embedder: self.embedder.as_ref().map(Arc::clone),
                default_timeout_ms: self.default_timeout_ms,
                default_max_rows: self.default_max_rows,
                lifecycle: crate::graph::GraphLifecycle::detached_from(&self.lifecycle),
            };

            // Add the report
            new_kg.add_report(OperationReport::CalculationOperation(result));

            Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
        } else if use_grouping {
            // Return counts grouped by parent
            let counts = kglite_core::api::fluent::count_nodes_by_parent(
                &self.inner,
                &self.cursor.selection,
                level_index,
            );
            py_out::convert_computation_results_for_python(counts)
        } else {
            // Simple flat count
            let count =
                kglite_core::api::fluent::count_nodes_in_level(&self.cursor.selection, level_index);
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
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(format!("Node type '{}' not found", node_type)),
            ));
        }
        if !self.inner.type_indices.contains_key(&parent_type) {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(format!("Parent type '{}' not found", parent_type)),
            ));
        }
        let graph = get_graph_mut(&mut self.inner);
        graph.parent_types_mut().insert(node_type, parent_type);
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
    /// There is no ``limit`` kwarg. Older tutorials sometimes show
    /// ``describe(limit=N)`` — that's not a supported signature; use
    /// ``sample_truncate`` for string-value clipping, or filter
    /// ``types=[...]`` to narrow the inventory.
    ///
    /// When `type_search`, `connections`, `cypher`, or `fluent` is set, only those tracks are returned.
    #[pyo3(signature = (types=None, type_search=None, connections=None, cypher=None, fluent=None, max_pairs=None, sample_truncate=40))]
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
        .map_err(|e: String| -> PyErr {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
        })
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
        let opts = kglite_core::api::ExploreOptions {
            max_entities,
            max_depth,
            include_source,
            ..Default::default()
        };
        Ok(kglite_core::api::explore_markdown(
            &self.inner,
            query,
            &opts,
            &roots,
        ))
    }

    /// Return the label-pair edge-count cardinality cache —
    /// `[(src_type, edge_type, tgt_type, count), ...]` triples backing
    /// the planner's selectivity-aware cost model (0.9.35).
    ///
    /// First call computes O(E); subsequent calls are O(triples)
    /// against the cached snapshot. Edge mutations (Cypher CREATE /
    /// DELETE, Python `add_connections`) invalidate the cache.
    ///
    /// Each row is a 4-tuple: `(src_type, edge_type, tgt_type, count)`.
    fn label_pair_counts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let triples = self.inner.get_or_compute_type_connectivity();
        let out: Vec<(String, String, String, u64)> = triples
            .into_iter()
            .map(|t| (t.src, t.conn, t.tgt, t.count as u64))
            .collect();
        Ok(out.into_pyobject(py)?.into_any().unbind())
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
        kglite_core::api::introspection::write_bug_report(
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
            &self.cursor.selection,
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
            inner: Arc::new(self.inner.independent_copy()),
            // copy() resets the selection/reports/stats but preserves the
            // temporal context (the as-of date carries to the independent copy).
            cursor: crate::graph::CursorState {
                selection: CowSelection::new(),
                reports: OperationReports::new(),
                last_mutation_stats: None,
                temporal_context: self.cursor.temporal_context.clone(),
            },
            embedder: self.embedder.as_ref().map(Arc::clone),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
            lifecycle: crate::graph::GraphLifecycle::detached(),
        }
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy()
    }
}
