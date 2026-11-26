//! KnowledgeGraph #[pymethods]: maintenance, introspection, and the
//! traverse / compare / derive-property half of the fluent chain (the
//! selection and filter half is in `kg_fluent.rs`).
//!
//! PyO3 merges multiple `#[pymethods] impl` blocks at class-registration
//! time, so splitting them across files is purely structural — no runtime
//! impact.

use super::borrow::borrow_mut_waiting;
use crate::datatypes::values::Value;
use crate::datatypes::{py_in, py_out};
use crate::graph::pyapi::kg_core::file_io_err;
use crate::graph::{
    compare_inner, extract_cypher_param, extract_detail_param, extract_fluent_param,
    parse_method_param, KnowledgeGraph,
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
    fn build_id_indices(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        node_types: Option<Vec<String>>,
    ) -> PyResult<()> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        // Warm through the IdIndexStore's interior mutability (the same store
        // the self-healing read path uses): a cache warm is a read, so no
        // `&mut` / `Arc::make_mut` deep-copying the whole graph when another
        // handle (fluent clone, frozen view, session) shares the Arc. The
        // scans are pure Rust; release the GIL for their duration.
        let inner = this.graph();
        py.detach(|| match node_types {
            Some(types) => {
                for node_type in types {
                    inner.ensure_id_index(&node_type);
                }
            }
            None => {
                for node_type in inner.type_indices.keys() {
                    inner.ensure_id_index(node_type);
                }
            }
        });
        Ok(())
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
    fn reindex(slf: &Bound<'_, Self>) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        this.unlogged_write(|graph| graph.reindex());
        Ok(())
    }

    /// Materialize the in-memory graph as a disk-backed one.
    ///
    /// Builds CSR (Compressed Sparse Row) edge arrays in files and switches
    /// the graph onto the disk backend. Node slots become a mmap'd array of
    /// 16 bytes per node; edges are read through the mapping.
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
    fn enable_disk_mode(slf: &Bound<'_, Self>, py: Python<'_>, path: Option<&str>) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        // Already converted: the second call cannot "convert" anything, and the
        // core message ("Already in disk mode") arrived through the save
        // dispatch as a *file I/O* error, which it is not.
        if kglite_core::api::storage::live_storage_mode(this.graph())
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
        // layer — core refuses it, in the caller's vocabulary rather than the
        // internal wrapper name. This also subsumes the diverged-log check
        // `save()` runs: a diverged log implies a durable owner.
        if this.lifecycle.durable.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "a durable graph cannot be converted to disk mode: disk mode keeps no \
                 write-ahead log, so the conversion would drop the capture layer along with \
                 anything it has buffered. Open the graph with durable=False (or \
                 kglite.load(path)) and convert that handle, or build into disk storage \
                 directly with KnowledgeGraph(storage='disk', path=...).",
            ));
        }
        let Some(path) = path else {
            // The pathless conversion is legitimate (scratch that dies with
            // the process) but it silently picks the filesystem: one larger
            // than a small or RAM-backed `/tmp` failed with no hint of where
            // the bytes had gone. Hence naming the directory in the warning.
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
            return this
                .unlogged_write(|graph| graph.enable_disk_mode())
                .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>);
        };
        // The path form publishes, so it takes the guard `save()` takes for a
        // handle *derived* from a durable graph (the durable owner itself is
        // already refused above): such a handle holds a fork no log describes,
        // and publishing it is the same hazard as saving it.
        this.check_durable_owner()?;
        this.unlogged_write_lazy(|inner| {
            py.detach(move || io::materialize_disk_graph(inner, path))
        })
        .map_err(|error| match error {
            io::SaveError::Refused(message) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
            }
            io::SaveError::Io(message) => file_io_err(std::io::Error::other(message)),
        })?;
        // Same "save as" rebinding `save(path)` does: the directory is now this
        // graph's home, so a later bare `save()` writes back to it.
        this.lifecycle.source_path = Some(std::path::PathBuf::from(path));
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
    fn unspill(slf: &Bound<'_, Self>) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;
        // Nothing to move, and nothing to fork the graph for: a graph with no
        // column stores at all (one assembled through the direct `add_node`
        // route) has no mmap-backed column to bring home, and rebuilding
        // would be a shape change rather than an unspill.
        if this.graph().column_store_count() != 0 {
            this.unlogged_write(|graph| graph.rebuild_columns_to_heap());
        }
        Ok(())
    }

    /// Compact the graph by removing tombstones left by node/edge deletions.
    ///
    /// With StableDiGraph, deletions leave holes in the internal storage.
    /// Over time, this wastes memory and degrades iteration performance.
    /// vacuum() rebuilds the graph with contiguous indices; a compaction that
    /// reclaims tombstones drops every vector and text index, node and
    /// relationship, so rebuild the ones you need afterwards.
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
    fn vacuum(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let (
            tombstones_before,
            edge_tombstones_before,
            info_after,
            columnar_rebuilt,
            nodes_remapped,
        ) = this.renumbering_write(|graph, cursor| {
            let info_before = graph.graph_info();
            let tombstones_before = info_before.node_tombstones;
            let edge_tombstones_before = info_before.edge_tombstones;
            // Read the row count, not "is this graph columnar at all": every graph
            // owns stores from its first node now, so that proxy would report
            // `true` for a vacuum that reclaimed nothing.
            let rows_before = info_before.columnar_total_rows;
            let old_to_new = graph.vacuum();
            let info_after = graph.graph_info();
            let columnar_rebuilt = info_after.columnar_total_rows != rows_before;
            let nodes_remapped = old_to_new.len();

            // The indices moved, but the *set of nodes the caller chose* did not,
            // minus whatever was deleted. The documented reset was a limitation of
            // not having the mapping to hand, never a contract.
            cursor.selection.remap_indices(&old_to_new);
            (
                tombstones_before,
                edge_tombstones_before,
                info_after,
                columnar_rebuilt,
                nodes_remapped,
            )
        })?;

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
    fn purge_provisional(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let (nodes_purged, edges_removed) = this.logged_write_with_cursor(|graph, cursor| {
            let (nodes_purged, edges_removed) =
                kglite_core::api::mutation::purge_provisional_nodes(graph)
                    .map_err(|message| super::kg_mutation::bulk_write_err(graph, message))?;
            if nodes_purged > 0 {
                cursor.selection = CowSelection::new();
            }
            Ok((nodes_purged, edges_removed))
        })?;
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
    ///         - 'topology_mapped_bytes': bytes of the topology read in place
    ///           from the file mapping after a 'mapped' load of a v9 file; 0 for
    ///           a heap topology
    ///         - 'topology_heap_bytes': heap bytes of the topology (the node
    ///           and edge arrays of a heap graph; only what a load copied for a
    ///           mapped one); 0 on a disk graph
    ///         - 'topology_overlay_entries': entries of the write overlay over
    ///           a mapped topology; 0 before the first write
    ///
    /// Example:
    ///     ```python
    ///     info = graph.graph_info()
    ///     if info['fragmentation_ratio'] > 0.3:
    ///         graph.vacuum()
    ///     ```
    fn graph_info(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let info = this.graph().graph_info();
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
            dict.set_item("format_version", this.graph().save_metadata.format_version)?;
            dict.set_item(
                "library_version",
                &this.graph().save_metadata.library_version,
            )?;
            dict.set_item("user_schema_version", this.graph().user_schema_version)?;
            dict.set_item(
                "storage_mode",
                kglite_core::api::storage::live_storage_mode(this.graph()).as_str(),
            )?;
            dict.set_item("columnar_heap_bytes", info.columnar_heap_bytes)?;
            dict.set_item("columnar_is_mapped", info.columnar_is_mapped)?;
            dict.set_item("edges_mapped", info.edges_mapped)?;
            dict.set_item(
                "edge_property_overlay_rows",
                info.edge_property_overlay_rows,
            )?;
            dict.set_item("topology_mapped_bytes", info.topology_mapped_bytes)?;
            dict.set_item("topology_heap_bytes", info.topology_heap_bytes)?;
            dict.set_item("topology_overlay_entries", info.topology_overlay_entries)?;
            dict.set_item("topology_status", info.topology_status)?;
            dict.set_item("topology_heap_reason", info.topology_heap_reason)?;
            let fallbacks = pyo3::types::PyList::empty(py);
            for fallback in &info.mapped_fallbacks {
                let row = PyDict::new(py);
                row.set_item("part", &fallback.part)?;
                row.set_item("reason", &fallback.reason)?;
                fallbacks.append(row)?;
            }
            dict.set_item("mapped_fallbacks", fallbacks)?;
            dict.set_item("write_budget_bytes", info.write_budget_bytes)?;
            dict.set_item("write_budget_used_bytes", info.write_budget_used_bytes)?;
            dict.set_item("write_budget_state", info.write_budget_state.as_str())?;
            dict.set_item("memory_limit", this.graph().memory_limit)?;
            dict.set_item("columnar_total_rows", info.columnar_total_rows)?;
            dict.set_item("columnar_live_rows", info.columnar_live_rows)?;
            dict.set_item("auto_vacuum_threshold", this.graph().auto_vacuum_threshold)?;
            dict.set_item("auto_vacuums_run", this.graph().auto_vacuums_run)?;
            let default_dict = PyDict::new(py);
            default_dict.set_item("effective", this.graph().valid_time_default.to_string())?;
            default_dict.set_item("stored", this.graph().stored_valid_time_default.to_string())?;
            dict.set_item("valid_time_default", default_dict)?;
            if let Some(build) = &this.graph().build_info {
                dict.set_item("build", build_info_dict(py, build)?)?;
            }
            if !this.graph().advisories.is_empty() {
                let list = pyo3::types::PyList::empty(py);
                for a in &this.graph().advisories {
                    let entry = PyDict::new(py);
                    entry.set_item("code", &a.code)?;
                    entry.set_item("writer", &a.writer)?;
                    entry.set_item("message", &a.message)?;
                    entry.set_item("affected", &a.affected)?;
                    list.append(entry)?;
                }
                dict.set_item("advisories", list)?;
            }
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
    fn set_auto_vacuum(slf: &Bound<'_, Self>, threshold: Option<f64>) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        if let Some(t) = threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "threshold must be between 0.0 and 1.0, or None to disable",
                ));
            }
        }
        this.unlogged_write(|graph| graph.auto_vacuum_threshold = threshold);
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
        slf: &Bound<'_, Self>,
        limit_bytes: Option<usize>,
        spill_dir: Option<String>,
    ) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        this.unlogged_write(|graph| {
            graph.memory_limit = limit_bytes;
            graph.spill_dir = spill_dir.map(std::path::PathBuf::from);
        });
        Ok(())
    }

    /// Set or query read-only mode for the Cypher layer.
    ///
    /// When enabled, all Cypher mutation queries (CREATE, SET, DELETE, REMOVE,
    /// MERGE) are rejected with an error, and `describe()` announces the
    /// restriction in a `<read-only>` element (the Cypher reference it renders
    /// is unchanged).  Read-only queries (MATCH, RETURN, CALL, etc.) are
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
    fn read_only(slf: &Bound<'_, Self>, enabled: Option<bool>) -> PyResult<bool> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;
        Ok({
            if let Some(v) = enabled {
                this.unlogged_write(|graph| graph.read_only = v);
            }
            this.graph().read_only
        })
    }

    /// Lock the schema: future Cypher mutations (CREATE, SET, MERGE) must
    /// conform to the currently known node types, connection types, and
    /// property types.
    ///
    /// Reads are validated too: an unknown node label, and a property name no
    /// node of the type carries, raise `SchemaError` instead of returning zero
    /// rows or a column of nulls. The full set of cases the read check stays
    /// silent on is documented on the `.pyi` stub, which is the API-doc source
    /// of truth.
    ///
    /// Returns:
    ///     This same graph (not a copy), so the call can be chained.
    ///
    /// Example::
    ///
    /// ```text
    /// graph.lock_schema()
    /// graph.cypher("CREATE (p:Typo {name: 'x'})")  # raises RuntimeError
    /// graph.cypher("MATCH (p:Person) RETURN p.agee")  # raises SchemaError
    /// ```
    fn lock_schema<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let mut guard = borrow_mut_waiting(slf)?;
        guard.unlogged_write(|g| g.schema_locked = true);
        Ok(slf.clone())
    }

    /// Unlock the schema: allow any Cypher mutations without schema validation,
    /// and return reads to reporting unknown labels / absent properties as
    /// non-fatal warnings.
    ///
    /// Returns:
    ///     This same graph (not a copy), so the call can be chained.
    fn unlock_schema<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let mut guard = borrow_mut_waiting(slf)?;
        guard.unlogged_write(|g| g.schema_locked = false);
        Ok(slf.clone())
    }

    /// Whether the schema is currently locked.
    #[getter]
    fn schema_locked(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok(this.graph().schema_locked)
    }

    /// The directory this graph was built from, or None.
    #[getter]
    fn source_root(slf: &Bound<'_, Self>) -> PyResult<Option<String>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok(this.graph().source_root.clone())
    }

    /// What okf.fingerprint() said about source_root at build time, or None.
    #[getter]
    fn source_fingerprint(slf: &Bound<'_, Self>) -> PyResult<Option<u64>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok(this.graph().source_fingerprint)
    }

    /// The dialect okf.build() read source_root with, or None.
    #[getter]
    fn source_dialect(slf: &Bound<'_, Self>) -> PyResult<Option<String>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok(this.graph().source_dialect.clone())
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
    fn schema_version(slf: &Bound<'_, Self>) -> PyResult<u32> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok(this.graph().user_schema_version)
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
    fn set_schema_version<'py>(slf: &Bound<'py, Self>, version: u32) -> PyResult<Bound<'py, Self>> {
        let mut guard = borrow_mut_waiting(slf)?;
        guard.logged_write(|g| {
            g.set_user_schema_version(version);
            Ok(())
        })?;
        Ok(slf.clone())
    }

    /// Returns a dict of {node_type: count} using the type index (O(type_count)).
    fn node_type_counts(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        Python::attach(|py| {
            let dict = PyDict::new(py);
            for (node_type, indices) in this.graph().type_indices.iter() {
                // A listing, so it hides what `node_types()` hides.
                if kglite_core::api::is_system_label(node_type) {
                    continue;
                }
                dict.set_item(node_type, indices.len())?;
            }
            Ok(dict.into())
        })
    }

    /// The relationships of the selected nodes, grouped by parent.
    #[pyo3(signature = (indices=None, parent_info=None, include_node_properties=None,
                        flatten_single_parent=true))]
    pub(super) fn relationships(
        slf: &Bound<'_, Self>,
        indices: Option<Vec<usize>>,
        parent_info: Option<bool>,
        include_node_properties: Option<bool>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let argument = |e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e));
        let valid_time = kglite_core::api::fluent::FluentFilter::for_walk(
            this.graph(),
            &this.cursor.temporal_context,
            None,
            "relationships()",
        )
        .map_err(argument)?;
        let connections = kglite_core::api::fluent::get_connections(
            this.graph(),
            &this.cursor.selection,
            None,
            indices.as_deref(),
            include_node_properties.unwrap_or(true),
            Some(&valid_time),
        )
        .map_err(argument)?;
        Python::attach(|py| {
            py_out::level_connections_to_pydict(
                py,
                &this.graph().graph,
                &connections,
                parent_info,
                flatten_single_parent,
            )
        })
    }

    /// Get titles of selected nodes.
    #[pyo3(signature = (limit=None, indices=None, flatten_single_parent=None))]
    fn titles(
        slf: &Bound<'_, Self>,
        limit: Option<usize>,
        indices: Option<Vec<usize>>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let values = kglite_core::api::fluent::get_property_values(
            this.graph(),
            &this.cursor.selection,
            None,
            &["title"],
            indices.as_deref(),
            limit,
        );
        Python::attach(|py| {
            py_out::level_single_values_to_pydict(
                py,
                &this.graph().graph,
                &values,
                flatten_single_parent,
            )
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
    fn explain(slf: &Bound<'_, Self>) -> PyResult<String> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let plan = this.cursor.selection.get_execution_plan();
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

    /// Get specific properties for selected nodes.
    #[pyo3(signature = (properties, limit=None, indices=None, flatten_single_parent=None))]
    fn get_properties(
        slf: &Bound<'_, Self>,
        properties: Vec<String>,
        limit: Option<usize>,
        indices: Option<Vec<usize>>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let property_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        let values = kglite_core::api::fluent::get_property_values(
            this.graph(),
            &this.cursor.selection,
            None,
            &property_refs,
            indices.as_deref(),
            limit,
        );
        Python::attach(|py| {
            py_out::level_values_to_pydict(py, &this.graph().graph, &values, flatten_single_parent)
        })
    }

    /// Get unique values of a property, optionally storing results.
    // Keyword-rich Python surface: each arg is an optional pyo3 kwarg; splitting
    // into a params struct would not change the Python call shape.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (property, group_by_parent=None, level_index=None, indices=None, store_as=None, max_length=None, keep_selection=None))]
    fn unique_values<'py>(
        slf: &Bound<'py, Self>,
        property: String,
        group_by_parent: Option<bool>,
        level_index: Option<usize>,
        indices: Option<Vec<usize>>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = borrow_mut_waiting(slf)?;
        let py = slf.py();
        let values = kglite_core::api::fluent::get_unique_values(
            guard.graph(),
            &guard.cursor.selection,
            &property,
            level_index,
            group_by_parent.unwrap_or(true),
            indices.as_deref(),
        );

        if let Some(target_property) = store_as {
            let nodes =
                kglite_core::api::fluent::format_unique_values_for_storage(&values, max_length);

            // Logged first, so a warning promoted to an error cannot leave a
            // write the log never saw.
            let written = guard.logged_write(|graph| {
                kglite_core::api::mutation::update_node_properties(graph, &nodes, target_property)
                    .map_err(|m| crate::graph::store_as_refused(graph, m))
            })?;
            crate::graph::warn_all(py, &written.warnings)?;

            if !keep_selection.unwrap_or(false) {
                guard.cursor.selection.clear();
            }

            // The same handle back, not a copy of it: a copy would share the
            // storage but not the write-ahead log, so `g = g.unique_values(...,
            // store_as=...)` on a durable graph used to hand back a handle
            // whose every later write was unlogged.
            Ok(slf.clone().into_any().unbind())
        } else {
            py_out::level_unique_values_to_pydict(py, &guard.graph().graph, &values)
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
    ///     connection_type (str): Edge type to follow (e.g. ``'HAS_HOLDER'``).
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
    /// g.select('Project').traverse('HAS_HOLDER')
    /// g.select('Project').traverse('OF_PROJECT', direction='incoming',
    ///     target_type='ProductionProfile')
    /// g.select('Project').traverse('HAS_HOLDER',
    ///     where={'title': 'Northwind Energy AS'})
    /// g.select('Project').traverse('HAS_HOLDER', at='2005')
    /// ```
    #[pyo3(signature = (connection_type, level_index=None, direction=None, sort_target=None, limit=None, new_level=None, at=None, during=None, temporal=None, target_type=None, r#where=None, where_connection=None))]
    #[allow(clippy::too_many_arguments)]
    fn traverse(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        connection_type: String,
        level_index: Option<usize>,
        direction: Option<String>,
        sort_target: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        new_level: Option<bool>,
        at: Option<&Bound<'_, PyAny>>,
        during: Option<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
        temporal: Option<bool>,
        target_type: Option<&Bound<'_, PyAny>>,
        r#where: Option<&Bound<'_, PyDict>>,
        where_connection: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let mut new_kg = this.clone();

        // node_count() on the level rather than materializing it: no allocation.
        let estimated = new_kg
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        let target_types = crate::graph::string_or_list(target_type, "target_type")?;

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

        let at = at
            .map(|value| crate::datatypes::py_in::query_date(value, "at"))
            .transpose()?
            .map(|(date, _)| date);
        let during = match &during {
            Some((start, end)) => {
                let (start, _) = crate::datatypes::py_in::query_date(start, "during")?;
                let (end, precision) = crate::datatypes::py_in::query_date(end, "during")?;
                Some((
                    start,
                    kglite_core::api::timeseries::expand_end(end, precision),
                ))
            }
            None => None,
        };
        let temporal_filter = kglite_core::api::fluent::FluentFilter::for_traverse(
            this.graph(),
            &this.cursor.temporal_context,
            at,
            during,
            temporal,
            &connection_type,
            target_types.as_deref(),
        )
        .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?;

        // All inputs are pure Rust by now — run the traversal off-GIL.
        {
            let inner = this.graph();
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
                    Some(&temporal_filter),
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
    /// Exactly one target type is supported. A single-element list is
    /// accepted for symmetry with `traverse()`, but a list of two or more
    /// raises `ArgumentError` — call `compare()` once per type instead.
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
        slf: &Bound<'_, Self>,
        target_type: &Bound<'_, PyAny>,
        method: &Bound<'_, PyAny>,
        filter: Option<&Bound<'_, PyDict>>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        level_index: Option<usize>,
        new_level: Option<bool>,
    ) -> PyResult<Self> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let _ = (level_index, new_level); // accepted but not yet used
        let mut new_kg = this.clone();

        let estimated = new_kg
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        // The comparison traversal is single-target: every method in
        // `make_comparison_traversal` dispatches on one type name. A
        // multi-element list is refused rather than silently truncated.
        let resolved_target: Option<String> =
            match crate::graph::string_or_list(Some(target_type), "target_type")? {
                Some(types) if types.len() > 1 => {
                    return Err(crate::error_py::kg_to_pyerr(
                        crate::error::KgError::Argument(format!(
                            "compare() supports one target_type; got {}. Pass a single type, \
                             or call compare() once per type.",
                            types.len()
                        )),
                    ));
                }
                Some(types) => types.into_iter().next(),
                None => None,
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

        let valid_time = match resolved_target.as_deref() {
            Some(target) => kglite_core::api::fluent::FluentFilter::for_label(
                this.graph(),
                &this.cursor.temporal_context,
                target,
                "compare()",
            )
            .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?,
            None => kglite_core::api::fluent::FluentFilter::default(),
        };
        compare_inner(
            this.graph(),
            &mut new_kg.cursor.selection,
            resolved_target.as_deref(),
            &config,
            conditions.as_ref(),
            sort_fields.as_ref(),
            limit,
            estimated,
            &valid_time,
        )?;

        Ok(new_kg)
    }

    /// Materialise direct relationships from the selection chain; assign the returned graph to keep them.
    #[pyo3(signature = (connection_type, keep_selection=None, conflict_handling=None, properties=None, source_type=None, target_type=None))]
    // One Rust argument per Python keyword.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_relationships(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        connection_type: String,
        keep_selection: Option<bool>,
        conflict_handling: Option<String>,
        properties: Option<&Bound<'_, PyDict>>,
        source_type: Option<String>,
        target_type: Option<String>,
    ) -> PyResult<Self> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

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

        // Detect the "chain temp" case before we mutate. With the inner Arc
        // shared (refcount > 1), the upcoming `Arc::make_mut` clones and the
        // mutation lands on that clone, so a caller who does not capture the
        // return value (`g.select(...).create_relationships(...)` without
        // assigning back) silently loses it.
        if Arc::strong_count(this.graph()) > 1 {
            let warning_module = py.import("warnings")?;
            warning_module.call_method1(
                "warn",
                (
                    format!(
                        "create_relationships('{}') was called on a chained graph view \
                         (Arc refcount={}). The mutation lands on a temporary clone — \
                         either capture the return value (`g = g.select(...).create_relationships('{}')`) \
                         or use the equivalent `add_relationships(data=cypher_result, ...)` form. \
                         The original graph variable will NOT see these edges.",
                        connection_type,
                        Arc::strong_count(this.graph()),
                        connection_type,
                    ),
                    py.import("builtins")?.getattr("UserWarning")?,
                    2u32, // stacklevel: blame the user's call site, not this fn
                ),
            )?;
        }

        let result = this.logged_write_with_cursor(|graph, cursor| {
            kglite_core::api::mutation::create_connections(
                graph,
                &cursor.selection,
                connection_type,
                conflict_handling,
                copy_properties,
                source_type,
                target_type,
            )
            .map_err(|message| super::kg_mutation::bulk_write_err(graph, message))
        })?;
        crate::graph::warn_all(py, &result.warnings)?;

        let mut new_kg = this.detached_view(keep_selection.unwrap_or(false));

        new_kg.add_report(OperationReport::ConnectionOperation(result));

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
        slf: &Bound<'_, Self>,
        properties: &Bound<'_, PyDict>,
        keep_selection: Option<bool>,
    ) -> PyResult<Self> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        use kglite_core::api::mutation::{add_properties as core_add_properties, PropertySpec};

        let mut spec_map: HashMap<String, PropertySpec> = HashMap::new();
        for (key, value) in properties.iter() {
            let source_type: String = key.extract()?;

            if let Ok(list) = value.extract::<Vec<String>>() {
                if list.is_empty() {
                    spec_map.insert(source_type, PropertySpec::CopyAll);
                } else {
                    spec_map.insert(source_type, PropertySpec::CopyList(list));
                }
            } else if let Ok(dict) = value.cast::<PyDict>() {
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

        let result = this.logged_write_with_cursor(|graph, cursor| {
            core_add_properties(graph, &cursor.selection, spec_map)
                .map_err(|e: String| super::kg_mutation::bulk_write_err(graph, e))
        })?;
        Python::attach(|py| crate::graph::warn_all(py, &result.warnings))?;

        let mut new_kg = this.detached_view(keep_selection.unwrap_or(true));

        new_kg.cursor.selection.add_plan_step(
            PlanStep::new("ADD_PROPERTIES", None, result.nodes_updated)
                .with_actual_rows(result.properties_set),
        );

        Ok(new_kg)
    }

    /// Collect child-node property values into comma-separated lists.
    // Keyword-rich Python surface: each arg is an optional pyo3 kwarg; splitting
    // into a params struct would not change the Python call shape.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (property=None, r#where=None, sort=None, limit=None, store_as=None, max_length=None, keep_selection=None))]
    fn collect_children(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        property: Option<&str>,
        r#where: Option<&Bound<'_, PyDict>>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let property_name = property.unwrap_or("title");

        let mut filtered_kg = this.clone();

        if let Some(where_dict) = r#where {
            let conditions = py_in::pydict_to_filter_conditions(where_dict)?;
            let sort_fields = match sort {
                Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
                None => None,
            };

            kglite_core::api::fluent::filter_nodes(
                this.graph(),
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
                this.graph(),
                &mut filtered_kg.cursor.selection,
                sort_fields,
            )
            .map_err(|e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;

            if let Some(max) = limit {
                kglite_core::api::fluent::limit_nodes_per_group(
                    this.graph(),
                    &mut filtered_kg.cursor.selection,
                    max,
                )
                .map_err(|e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?;
            }
        } else if let Some(max) = limit {
            kglite_core::api::fluent::limit_nodes_per_group(
                this.graph(),
                &mut filtered_kg.cursor.selection,
                max,
            )
            .map_err(|e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;
        }

        // Pure-Rust scan over the selection (titles included) — run off-GIL.
        let property_groups = {
            let inner = filtered_kg.graph();
            let selection = &filtered_kg.cursor.selection;
            py.detach(|| {
                kglite_core::api::fluent::get_children_properties(inner, selection, property_name)
            })
        };

        if store_as.is_none() {
            let dict_pairs =
                kglite_core::api::fluent::format_for_dictionary(&property_groups, max_length);

            return Python::attach(|py| py_out::string_pairs_to_pydict(py, &dict_pairs));
        }

        let nodes = kglite_core::api::fluent::format_for_storage(&property_groups, max_length);

        let result = this.logged_write(|graph| {
            kglite_core::api::mutation::update_node_properties(graph, &nodes, store_as.unwrap())
                .map_err(|m| crate::graph::store_as_refused(graph, m))
        })?;
        Python::attach(|py| crate::graph::warn_all(py, &result.warnings))?;

        let mut new_kg = this.detached_view(keep_selection.unwrap_or(false));

        new_kg.add_report(OperationReport::NodeOperation(result));

        Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
    }

    /// Compute descriptive statistics for a numeric property.
    #[pyo3(signature = (property, level_index=None, group_by=None))]
    fn statistics(
        slf: &Bound<'_, Self>,
        property: &str,
        level_index: Option<usize>,
        group_by: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        if let Some(group_prop) = group_by {
            let groups = kglite_core::api::fluent::calculate_grouped_property_stats(
                this.graph(),
                &this.cursor.selection,
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
            kglite_core::api::fluent::get_parent_child_pairs(&this.cursor.selection, level_index);
        let stats =
            kglite_core::api::fluent::calculate_property_stats(this.graph(), &pairs, property);
        py_out::convert_stats_for_python(stats)
    }

    /// Evaluate a mathematical expression on selected nodes.
    #[pyo3(signature = (expression, level_index=None, store_as=None, keep_selection=None, aggregate_connections=None))]
    fn calculate(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        expression: &str,
        level_index: Option<usize>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
        aggregate_connections: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        if let Some(target_property) = store_as {
            let report = this.logged_write_with_cursor(|graph, cursor| {
                // Pure-Rust evaluation + store — run off-GIL.
                let selection = &cursor.selection;
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
                    Ok(kglite_core::api::fluent::EvaluationResult::Stored(report)) => Ok(report),
                    Ok(_) => Err(crate::graph::store_as_refused(
                        graph,
                        "Unexpected result type when storing calculation result".to_string(),
                    )),
                    // A constraint or ontology refusal parked on the graph keeps
                    // its typed class; the expression prefix would break the
                    // identity that recovers it.
                    Err(e) => Err(match graph.take_constraint_error(&e) {
                        Some(typed) => crate::error_py::kg_to_pyerr(typed),
                        None => crate::graph::store_as_refused(
                            graph,
                            format!("Error evaluating expression '{expression}': {e}"),
                        ),
                    }),
                }
            })?;
            Python::attach(|py| crate::graph::warn_all(py, &report.warnings))?;
            let mut new_kg = this.detached_view(keep_selection.unwrap_or(false));

            new_kg.add_report(OperationReport::CalculationOperation(report));

            Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
        } else {
            // The temporary whole-graph clone + evaluation are pure Rust —
            // run off-GIL. (That `process_equation` demands `&mut DirGraph`
            // even when nothing is stored — forcing this deep clone — is a
            // pre-existing core-signature wart.)
            let inner = this.graph();
            let selection = &this.cursor.selection;
            let process_result = py.detach(|| {
                kglite_core::api::fluent::process_equation(
                    &mut (**inner).clone(),
                    selection,
                    expression,
                    level_index,
                    None,
                    aggregate_connections,
                )
            });

            match process_result {
                Ok(kglite_core::api::fluent::EvaluationResult::Computed(results)) => {
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

    /// Count nodes, optionally grouped by parent or by a property.
    #[pyo3(signature = (level_index=None, group_by_parent=None, store_as=None, keep_selection=None, group_by=None))]
    fn count(
        slf: &Bound<'_, Self>,
        level_index: Option<usize>,
        group_by_parent: Option<bool>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
        group_by: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        if let Some(property) = group_by {
            let nodes = kglite_core::api::fluent::collect_selected_nodes(
                &this.cursor.selection,
                level_index,
            );
            let mut groups: HashMap<String, usize> = HashMap::new();
            for idx in nodes {
                if let Some(node) = this.graph().node_view(idx) {
                    let resolved = this
                        .graph()
                        .resolve_alias(node.node_type_str(&this.graph().interner), property);
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
                            nanos: 0,
                        }) => format!("duration(M={},D={},S={})", months, days, seconds),
                        Some(Value::NodeRef(idx)) => format!("node#{}", idx),
                        Some(Value::Null) | None => "null".to_string(),
                        // Everything left is a collection or graph-entity value.
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

        let has_multiple_levels = this.cursor.selection.get_level_count() > 1;
        let use_grouping = group_by_parent.unwrap_or(has_multiple_levels);

        if let Some(target_property) = store_as {
            let result = this.logged_write_with_cursor(|graph, cursor| {
                kglite_core::api::fluent::store_count_results(
                    graph,
                    &cursor.selection,
                    level_index,
                    use_grouping,
                    target_property,
                )
                .map_err(|m| crate::graph::store_as_refused(graph, m))
            })?;
            Python::attach(|py| crate::graph::warn_all(py, &result.warnings))?;

            let mut new_kg = this.detached_view(keep_selection.unwrap_or(false));

            new_kg.add_report(OperationReport::CalculationOperation(result));

            Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
        } else if use_grouping {
            let counts = kglite_core::api::fluent::count_nodes_by_parent(
                this.graph(),
                &this.cursor.selection,
                level_index,
            );
            py_out::convert_computation_results_for_python(counts)
        } else {
            let count =
                kglite_core::api::fluent::count_nodes_in_level(&this.cursor.selection, level_index);
            Python::attach(|py| count.into_py_any(py))
        }
    }

    /// Return a text summary of the graph schema (node types, connections).
    fn schema_text(slf: &Bound<'_, Self>) -> PyResult<String> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let schema_string = introspection::debugging::get_schema_string(this.graph());
        Ok(schema_string)
    }

    /// Mark a node type as a supporting (child) type of a parent core type.
    ///
    /// Supporting types are hidden from the `describe()` inventory and instead
    /// appear in the `<supporting>` section when the parent type is inspected.
    /// Their capabilities (timeseries, spatial, etc.) bubble up to the parent.
    #[pyo3(signature = (node_type, parent_type))]
    fn set_parent_type(
        slf: &Bound<'_, Self>,
        node_type: String,
        parent_type: String,
    ) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        if !this.graph().type_indices.contains_key(&node_type) {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(format!("Node type '{}' not found", node_type)),
            ));
        }
        if !this.graph().type_indices.contains_key(&parent_type) {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(format!("Parent type '{}' not found", parent_type)),
            ));
        }
        this.logged_write(|g| {
            g.set_parent_type(&node_type, Some(&parent_type));
            Ok(())
        })
    }

    /// Return an XML description of this graph for AI agents (progressive disclosure).
    /// For structured data use `schema()` and `graph_info()`.
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
        slf: &Bound<'_, Self>,
        types: Option<Vec<String>>,
        type_search: Option<String>,
        connections: Option<&Bound<'_, PyAny>>,
        cypher: Option<&Bound<'_, PyAny>>,
        fluent: Option<&Bound<'_, PyAny>>,
        max_pairs: Option<usize>,
        sample_truncate: Option<usize>,
    ) -> PyResult<String> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let conn_detail = extract_detail_param(connections, "connections")?;
        let cypher_detail = extract_cypher_param(cypher)?;
        let fluent_detail = extract_fluent_param(fluent)?;
        introspection::compute_description(
            this.graph(),
            &introspection::DescribeRequest {
                types: types.as_deref(),
                connections: &conn_detail,
                cypher: &cypher_detail,
                fluent: &fluent_detail,
                type_search: type_search.as_deref(),
                max_pairs,
                sample_truncate,
                ..introspection::DescribeRequest::new(introspection::DescribeSurface::Python)
            },
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
        slf: &Bound<'_, Self>,
        query: &str,
        max_entities: usize,
        max_depth: usize,
        include_source: bool,
        source_roots: Option<Vec<String>>,
    ) -> PyResult<String> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

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
            this.graph(),
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
    /// DELETE, Python `add_relationships`) invalidate the cache.
    ///
    /// Each row is a 4-tuple: `(src_type, edge_type, tgt_type, count)`.
    fn label_pair_counts(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let triples = this.graph().get_or_compute_type_connectivity();
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
        _slf: &Bound<'_, Self>,
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

    /// Return a text summary of the current selection state.
    fn selection(slf: &Bound<'_, Self>) -> PyResult<String> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        Ok(introspection::debugging::get_selection_string(
            this.graph(),
            &this.cursor.selection,
        ))
    }

    /// Create an independent deep copy of this graph.
    ///
    /// Returns a new ``KnowledgeGraph`` that shares no mutable state with
    /// the original. Useful for running mutations without affecting the
    /// source graph.
    fn copy(slf: &Bound<'_, Self>) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let copy = this
            .graph()
            .try_independent_copy()
            .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::FileIo(e)))?;
        Ok(this.derive_handle(
            Arc::new(copy),
            // Deliberate: the as-of date carries to the copy, the rest does not.
            crate::graph::CursorState {
                selection: CowSelection::new(),
                reports: OperationReports::new(),
                last_mutation_stats: None,
                temporal_context: this.cursor.temporal_context.clone(),
            },
            this.embedder.as_ref().map(Arc::clone),
            crate::graph::GraphLifecycle::detached(),
        ))
    }

    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Self> {
        Self::copy(slf)
    }

    fn __deepcopy__(slf: &Bound<'_, Self>, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::copy(slf)
    }
}

/// `graph_info()['build']`: the counts per group and the recorded advisories
/// of the blueprint build that produced the graph.
fn build_info_dict<'py>(
    py: Python<'py>,
    build: &kglite_core::api::blueprint::BuildInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let summary = PyDict::new(py);
    for (group, count) in &build.summary {
        summary.set_item(group.as_str(), count)?;
    }
    let diagnostics = pyo3::types::PyList::empty(py);
    for d in &build.diagnostics {
        let entry = PyDict::new(py);
        entry.set_item("group", d.group.as_str())?;
        entry.set_item("kind", &d.kind)?;
        entry.set_item("message", &d.message)?;
        diagnostics.append(entry)?;
    }
    let out = PyDict::new(py);
    out.set_item("summary", summary)?;
    out.set_item("diagnostics", diagnostics)?;
    Ok(out)
}
