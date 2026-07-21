// NetworkX interop — wrapper-side only (#[pymethods] on KnowledgeGraph).
//
// Boundary principle (CLAUDE.md): networkx is a Python library, so all
// the marshalling lives here in kglite-py, NOT in the kglite core crate.
// A Go/JVM binding wouldn't touch any of this.
//
// `to_networkx()` iterates the internal graph directly (the same node /
// edge walk the d3/graphml exporters use) and builds an
// `nx.MultiDiGraph`. The reverse direction (`from_networkx`) is pure
// Python in `kglite/networkx_interop.py` — it bulk-loads via the
// DataFrame fast paths (`add_nodes` / `add_connections`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::datatypes::py_out::value_to_py;
use crate::graph::KnowledgeGraph;
use kglite_core::api::GraphRead;

#[pymethods]
impl KnowledgeGraph {
    /// Convert the graph to a :class:`networkx.MultiDiGraph`.
    ///
    /// KGLite is a directed multigraph with typed nodes and typed edges,
    /// so ``MultiDiGraph`` is the lossless target. Each node's ``id`` is
    /// used as the networkx node key; ``node_type``, ``title`` and every
    /// property are attached as node attributes. Each edge's
    /// ``connection_type`` is used as the first networkx edge key for a node
    /// pair; additional same-type parallel edges receive a collision-safe
    /// composite key. The type is always stored as the ``connection_type``
    /// edge attribute alongside every edge property.
    ///
    /// Requires the ``networkx`` package: ``pip install networkx``.
    ///
    /// Returns:
    ///     A ``networkx.MultiDiGraph`` mirroring the full graph.
    ///
    /// Note:
    ///     v1 always exports the full graph (selections are ignored).
    ///     A future revision may honour the active selection.
    ///
    /// Example:
    ///     ```python
    ///     import networkx as nx
    ///     nxg = graph.to_networkx()
    ///     scores = nx.pagerank(nxg)
    ///     ```
    fn to_networkx(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let nx = py.import("networkx").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>(
                "The 'networkx' package is required for to_networkx(). \
                 Install with: pip install networkx",
            )
        })?;

        let nxg = nx.getattr("MultiDiGraph")?.call0()?;
        let add_node = nxg.getattr("add_node")?;
        let add_edge = nxg.getattr("add_edge")?;

        let graph = &self.inner;
        let interner = &graph.interner;
        // Direct GraphRead traversal — hold the disk arena guard while
        // borrowed node/edge weights live (arena protocol; no-op in
        // memory/mapped).
        let _arena_guard = graph.begin_read_pass();

        // Cache node-index -> Python id key once. Reused for every edge
        // endpoint so the conversion stays O(n + e), not O(n + e·k).
        let mut id_by_index: std::collections::HashMap<usize, Py<PyAny>> =
            std::collections::HashMap::with_capacity(graph.graph.node_count());

        // Build nodes. Node key = id (the canonical per-mode integer/string).
        for idx in graph.graph.node_indices() {
            let Some(node) = graph.graph.node_weight(idx) else {
                continue;
            };
            let key = value_to_py(py, &node.id())?;
            let attrs = PyDict::new(py);
            attrs.set_item("node_type", node.node_type_str(interner))?;
            attrs.set_item("title", value_to_py(py, &node.title())?)?;
            // properties_cloned covers both row-backed and post-reload
            // columnar property storage; property_iter is empty for the latter.
            for (k, v) in node.properties_cloned(interner) {
                attrs.set_item(k, value_to_py(py, &v)?)?;
            }
            add_node.call((key.clone_ref(py),), Some(&attrs))?;
            id_by_index.insert(idx.index(), key);
        }

        // Build edges in a single global pass. Keep the readable connection
        // type key for the first edge, then disambiguate legal same-type
        // parallel edges with their stable edge index.
        for edge in graph.graph.edge_references() {
            let (Some(src), Some(tgt)) = (
                id_by_index.get(&edge.source().index()),
                id_by_index.get(&edge.target().index()),
            ) else {
                continue;
            };
            let ctype = edge.weight().connection_type_str(interner);
            let attrs = PyDict::new(py);
            attrs.set_item("connection_type", ctype)?;
            for (k, v) in edge.weight().property_iter(interner) {
                attrs.set_item(k, value_to_py(py, v)?)?;
            }
            let key_in_use: bool = nxg
                .call_method1("has_edge", (src.clone_ref(py), tgt.clone_ref(py), ctype))?
                .extract()?;
            if key_in_use {
                add_edge.call(
                    (
                        src.clone_ref(py),
                        tgt.clone_ref(py),
                        (ctype, edge.id().index()),
                    ),
                    Some(&attrs),
                )?;
            } else {
                add_edge.call((src.clone_ref(py), tgt.clone_ref(py), ctype), Some(&attrs))?;
            }
        }

        Ok(nxg.into())
    }
}
