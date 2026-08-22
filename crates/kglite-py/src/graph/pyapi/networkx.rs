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
use pyo3::types::{PyDict, PyString, PyTuple};

use crate::datatypes::py_out::value_to_py;
use crate::graph::{KnowledgeGraph, NodeKeyGuard, NodeKeyKind};
use kglite_core::api::GraphRead;

/// Under `node_key="id"` node keys are bare ids, so two types sharing an id
/// would merge into one NetworkX node (attrs overwritten, both nodes' edges
/// rewired onto the survivor). Refuse instead — and name two recipes the
/// whole-graph export can actually honour.
const ID_KEY: NodeKeyGuard<'static> = NodeKeyGuard {
    surface: "to_networkx()",
    kind: NodeKeyKind::Id,
    recipe: "give the colliding node types disjoint ids, or export with \
             node_key='type_id' to key nodes by (node_type, id) - the export \
             always covers the whole graph, so narrowing the selection cannot \
             avoid this",
};

/// How `node_key` maps a node onto its NetworkX key.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeKeyMode {
    /// Bare `id`. Graph-unique only when no two types share an id, so the
    /// `ID_KEY` guard is installed and a collision refuses the export.
    Id,
    /// The `(node_type, id)` 2-tuple. Ids are unique *within* a type, so this
    /// key is graph-unique by construction — no guard, no collision.
    TypeId,
}

impl NodeKeyMode {
    /// Reject an unknown value by name, listing what is valid (doctrine:
    /// never silently fall back to a default the caller did not ask for).
    fn parse(node_key: &str) -> PyResult<Self> {
        match node_key {
            "id" => Ok(Self::Id),
            "type_id" => Ok(Self::TypeId),
            other => Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(format!(
                    "to_networkx() got node_key='{other}'; valid values are \
                     'id' (bare node id - refuses a cross-type id collision) \
                     and 'type_id' (a (node_type, id) tuple key - collision-free)."
                )),
            )),
        }
    }
}

#[pymethods]
impl KnowledgeGraph {
    /// Convert the graph to a :class:`networkx.MultiDiGraph`.
    ///
    /// KGLite is a directed multigraph with typed nodes and typed edges,
    /// so ``MultiDiGraph`` is the lossless target. ``node_key`` selects the
    /// networkx node key: ``"id"`` (default) uses the bare node id, and
    /// ``"type_id"`` uses the ``(node_type, id)`` 2-tuple. ``node_type``,
    /// ``title`` and every property are attached as node attributes — the
    /// two identity attributes always win over a same-named property. Each
    /// edge's ``connection_type`` is used as the first networkx edge key for
    /// a node pair; additional same-type parallel edges receive a
    /// collision-safe composite key. The type is always stored as the
    /// ``connection_type`` edge attribute alongside every edge property.
    ///
    /// Requires the ``networkx`` package: ``pip install networkx``.
    ///
    /// Args:
    ///     node_key: ``"id"`` (default) or ``"type_id"``. Ids are unique
    ///         within a type but reused across types, so ``"type_id"`` is
    ///         the collision-free choice for a multi-type graph.
    ///
    /// Returns:
    ///     A ``networkx.MultiDiGraph`` mirroring the full graph.
    ///
    /// Raises:
    ///     ArgumentError: ``node_key`` is neither ``"id"`` nor ``"type_id"``;
    ///         or ``node_key="id"`` and two nodes of different types share an
    ///         id. Ids are unique per type, not across types, so a bare-id
    ///         node key would merge them into one networkx node. Give the
    ///         colliding types disjoint ids, or export with
    ///         ``node_key="type_id"`` — the export is whole-graph, so
    ///         narrowing the selection cannot avoid it.
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
    #[pyo3(signature = (*, node_key = "id"))]
    fn to_networkx(&self, py: Python<'_>, node_key: &str) -> PyResult<Py<PyAny>> {
        let key_mode = NodeKeyMode::parse(node_key)?;
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
        // Keys already handed to `add_node`. A NetworkX graph has no
        // insert-or-fail, so the guard owns a dict of its own; `add_node`
        // would silently overwrite.
        let seen_keys = PyDict::new(py);

        // Build nodes. Node key = the id (canonical per-mode integer/string),
        // or the (node_type, id) tuple under `node_key="type_id"`.
        for idx in graph.graph.node_indices() {
            let Some(node) = graph.graph.node_view(idx) else {
                continue;
            };
            let node_type = node.node_type_str(interner);
            let id = value_to_py(py, &node.id())?;
            let key: Py<PyAny> = match key_mode {
                NodeKeyMode::Id => id,
                NodeKeyMode::TypeId => {
                    PyTuple::new(py, [PyString::new(py, node_type).into_any().unbind(), id])?
                        .into_any()
                        .unbind()
                }
            };
            let attrs = PyDict::new(py);
            // Properties first: the two identity attributes below overwrite a
            // property that happens to share their name, rather than being
            // silently shadowed by it (the importer reads `node_type` back).
            // properties_cloned covers both row-backed and post-reload
            // columnar property storage; property_iter is empty for the latter.
            for (k, v) in node.properties_cloned(interner) {
                attrs.set_item(k, value_to_py(py, &v)?)?;
            }
            attrs.set_item("node_type", node_type)?;
            attrs.set_item("title", value_to_py(py, &node.title())?)?;
            // Tuple keys are graph-unique by construction (ids are unique
            // within a type), so only the bare-id mode needs the guard.
            if key_mode == NodeKeyMode::Id {
                ID_KEY.insert(&seen_keys, key.bind(py), py.None())?;
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
