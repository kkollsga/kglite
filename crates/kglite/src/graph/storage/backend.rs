//! GraphBackend enum + per-backend dispatcher + GraphRead / GraphWrite impls.
//!
//! The `GraphBackend` enum is the runtime variant of all storage backends
//! (Memory / Mapped / Disk / Recording). Its trait impls forward to the
//! inner backend via enum match. Boundary file — one of 7 whitelisted
//! `GraphBackend::[A-Z]` match sites.

use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::recording::RecordingGraph;
use crate::graph::storage::{GraphRead, GraphWrite, MappedGraph, MemoryGraph};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::graph::storage::disk::graph::DiskGraph;

// ============================================================================
// Graph Backend Abstraction
// ============================================================================

/// Graph storage backend. Four variants — heap-resident memory,
/// mmap-columnar-spilled mapped, CSR-on-disk, and a Phase 6 validation
/// wrapper that logs reads. Phase 5 promoted `MappedGraph` from a type
/// alias to a distinct struct so each backend owns its own
/// [`GraphRead`] / [`GraphWrite`] impl in
/// [`crate::graph::storage::impls`]; Phase 6 added
/// [`RecordingGraph`] as a live test of the trait surface. This
/// enum is now a 4-arm dumb dispatcher.
///
/// The `Recording` variant wraps any other `GraphBackend` — including,
/// in principle, another `Recording` — via
/// `Box<RecordingGraph<GraphBackend>>`. Its `is_memory` / `is_mapped`
/// / `is_disk` predicates forward to the inner backend so consumers
/// that switch on "what's the underlying storage" keep working
/// unchanged when wrapped.
#[allow(clippy::large_enum_variant)]
pub enum GraphBackend {
    Memory(MemoryGraph),
    Mapped(MappedGraph),
    Disk(Box<DiskGraph>),
    // Phase 6 validation wrapper. Constructed only from Rust
    // `#[cfg(test)]` modules in `storage/recording.rs`; no Python
    // constructor reaches this arm. The variant exists so the enum
    // dispatcher exercises a 4th backend through the same match arms
    // as the production three, proving open/closed at the trait
    // surface. `dead_code` in release builds sees no constructor,
    // hence the allow.
    #[allow(dead_code)]
    Recording(Box<RecordingGraph<GraphBackend>>),
}

impl GraphBackend {
    #[inline]
    pub fn new() -> Self {
        GraphBackend::Memory(MemoryGraph::new())
    }

    /// Borrow the inner heap `StableDiGraph` for petgraph algorithms
    /// (e.g. `kosaraju_scc`) that require concrete petgraph types.
    /// Disk panics — callers must gate on [`GraphRead::is_disk`].
    /// `Recording` forwards to the wrapped backend.
    #[inline]
    pub fn as_stable_digraph(&self) -> &StableDiGraph<NodeData, EdgeData> {
        match self {
            GraphBackend::Memory(g) => g.inner(),
            GraphBackend::Mapped(g) => g.inner(),
            GraphBackend::Disk(_) => unimplemented!("Disk backend: as_stable_digraph"),
            GraphBackend::Recording(rg) => rg.inner().as_stable_digraph(),
        }
    }

    /// Closure-based hot-path iteration over all live edges, yielding
    /// `(source, target, connection_type)` per edge.
    ///
    /// Avoids the `Box<dyn Iterator>` + virtual `.next()` dispatch that
    /// [`GraphRead::edge_endpoint_keys`] requires. Monomorphises per
    /// backend at the call site, so the compiler fully inlines the hot
    /// loop. Use this from any code path that walks every edge on a
    /// large graph (e.g. `compute_type_connectivity`, cache rebuild,
    /// bulk index builds). 863M-edge benchmarks show ~40–90 s savings
    /// per sweep vs the boxed-iterator path on Wikidata-scale graphs.
    ///
    /// The `Recording` variant forwards to its wrapped backend without
    /// recording — it logs only the trait-path methods.
    #[inline(always)]
    pub fn for_each_edge_endpoint_key<F>(&self, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, InternedKey),
    {
        use petgraph::visit::{EdgeRef, IntoEdgeReferences};
        match self {
            GraphBackend::Memory(g) => {
                for er in g.inner().edge_references() {
                    let w = er.weight();
                    f(er.source(), er.target(), w.connection_type);
                }
            }
            GraphBackend::Mapped(g) => {
                for er in g.inner().edge_references() {
                    let w = er.weight();
                    f(er.source(), er.target(), w.connection_type);
                }
            }
            GraphBackend::Disk(g) => {
                let dg = g.as_ref();
                for i in 0..dg.next_edge_idx {
                    let ep = dg.edge_endpoints.get(i as usize);
                    if ep.source == crate::graph::storage::disk::csr::TOMBSTONE_EDGE {
                        continue;
                    }
                    f(
                        NodeIndex::new(ep.source as usize),
                        NodeIndex::new(ep.target as usize),
                        InternedKey::from_u64(ep.connection_type),
                    );
                }
            }
            GraphBackend::Recording(rg) => {
                rg.inner().for_each_edge_endpoint_key(f);
            }
        }
    }

    /// Iterate only edges whose connection type matches `conn_type`,
    /// yielding `(src, tgt, edge_idx, properties)` per match.
    ///
    /// The callback returns `true` to continue or `false` to stop — so
    /// callers collecting a bounded prefix (sample edges, first match)
    /// don't pay for the rest of the matches.
    ///
    /// Avoids the disk backend's per-edge `Box<EdgeData>` arena push by
    /// reading `edge_endpoints` + `edge_properties` directly. On
    /// Memory/Mapped the petgraph iterator already hands out `&EdgeData`
    /// references into the resident storage, so there is no arena cost.
    ///
    /// On the disk backend, complexity is O(matching edges) thanks to
    /// the persisted `conn_type_index_*` inverted index — not O(all
    /// edges) as a filtered `edge_references()` sweep would be.
    ///
    /// `properties` is the empty slice when the edge has no custom
    /// properties (common case for topology-heavy graphs).
    #[inline(always)]
    pub fn for_each_edge_of_conn_type<F>(&self, conn_type: InternedKey, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, u32, &[(InternedKey, Value)]) -> bool,
    {
        use petgraph::visit::{EdgeRef, IntoEdgeReferences};
        let ct_u64 = conn_type.as_u64();
        match self {
            GraphBackend::Memory(g) => {
                for er in g.inner().edge_references() {
                    let w = er.weight();
                    if w.connection_type == conn_type
                        && !f(
                            er.source(),
                            er.target(),
                            er.id().index() as u32,
                            w.properties.as_slice(),
                        )
                    {
                        return;
                    }
                }
            }
            GraphBackend::Mapped(g) => {
                for er in g.inner().edge_references() {
                    let w = er.weight();
                    if w.connection_type == conn_type
                        && !f(
                            er.source(),
                            er.target(),
                            er.id().index() as u32,
                            w.properties.as_slice(),
                        )
                    {
                        return;
                    }
                }
            }
            GraphBackend::Disk(g) => {
                let dg = g.as_ref();
                dg.for_each_edge_of_conn_type(ct_u64, |src, tgt, edge_idx| {
                    // edge_properties_at returns Cow; bind to extend its
                    // lifetime across the callback, then deref to a slice.
                    let props_cow = dg.edge_properties_at(edge_idx);
                    let props: &[(
                        crate::graph::schema::InternedKey,
                        crate::datatypes::values::Value,
                    )] = props_cow.as_deref().unwrap_or(&[]);
                    f(src, tgt, edge_idx, props)
                });
            }
            GraphBackend::Recording(rg) => {
                rg.inner().for_each_edge_of_conn_type(conn_type, f);
            }
        }
    }
}

// -- Index traits --

impl std::ops::Index<NodeIndex> for GraphBackend {
    type Output = NodeData;
    #[inline]
    fn index(&self, index: NodeIndex) -> &NodeData {
        match self {
            GraphBackend::Memory(g) => &g.inner()[index],
            GraphBackend::Mapped(g) => &g.inner()[index],
            GraphBackend::Disk(dg) => &dg[index],
            GraphBackend::Recording(rg) => &rg.inner()[index],
        }
    }
}

impl std::ops::Index<EdgeIndex> for GraphBackend {
    type Output = EdgeData;
    #[inline]
    fn index(&self, index: EdgeIndex) -> &EdgeData {
        match self {
            GraphBackend::Memory(g) => &g.inner()[index],
            GraphBackend::Mapped(g) => &g.inner()[index],
            GraphBackend::Disk(dg) => &dg[index],
            GraphBackend::Recording(rg) => &rg.inner()[index],
        }
    }
}

// -- Clone --

impl Clone for GraphBackend {
    fn clone(&self) -> Self {
        match self {
            GraphBackend::Memory(g) => GraphBackend::Memory(g.clone()),
            GraphBackend::Mapped(g) => GraphBackend::Mapped(g.clone()),
            GraphBackend::Disk(dg) => GraphBackend::Disk(dg.clone()),
            GraphBackend::Recording(rg) => GraphBackend::Recording(Box::new((**rg).clone())),
        }
    }
}

// -- Serialize / Deserialize --
// Delegates to StableDiGraph so the binary format is identical to before.

impl Serialize for GraphBackend {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            GraphBackend::Memory(g) => g.serialize(serializer),
            GraphBackend::Mapped(g) => g.serialize(serializer),
            GraphBackend::Disk(_) => Err(serde::ser::Error::custom(
                "Disk backend does not support serialization",
            )),
            // Validation wrapper is transparent — serialize as the
            // wrapped backend. Recursively hits the Disk error arm
            // if the wrapped backend is Disk.
            GraphBackend::Recording(rg) => rg.inner().serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for GraphBackend {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let g = StableDiGraph::<NodeData, EdgeData>::deserialize(deserializer)?;
        Ok(GraphBackend::Memory(MemoryGraph(g)))
    }
}

// -- Debug --

impl std::fmt::Debug for GraphBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphBackend::Memory(g) => write!(
                f,
                "Memory({} nodes, {} edges)",
                g.node_count(),
                g.edge_count()
            ),
            GraphBackend::Mapped(g) => write!(
                f,
                "Mapped({} nodes, {} edges)",
                g.node_count(),
                g.edge_count()
            ),
            GraphBackend::Disk(_) => write!(f, "Disk(placeholder)"),
            GraphBackend::Recording(rg) => write!(f, "Recording({:?})", rg.inner()),
        }
    }
}

// ============================================================================
// GraphRead / GraphWrite dispatcher impls
//
// Phase 5 shrank these to dumb 3-arm dispatchers. The real impls live
// on each backend in `src/graph/storage/impls.rs`. The inherent
// `impl GraphBackend` method blocks that used to host these bodies are
// deleted — every caller either uses the trait (most of the codebase)
// or goes through the per-backend impl directly.
// ============================================================================

use crate::datatypes::values::Value;
use std::collections::HashMap;

impl GraphRead for GraphBackend {
    type NodeIndicesIter<'a> = crate::graph::core::iterators::GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = crate::graph::core::iterators::GraphEdgeIndices<'a>;
    type EdgesIter<'a> = crate::graph::core::iterators::GraphEdges<'a>;
    type EdgeReferencesIter<'a> = crate::graph::core::iterators::GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = crate::graph::core::iterators::GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = crate::graph::core::iterators::GraphNeighbors<'a>;

    #[inline]
    fn node_count(&self) -> usize {
        match self {
            Self::Memory(g) => GraphRead::node_count(g),
            Self::Mapped(g) => GraphRead::node_count(g),
            Self::Disk(g) => GraphRead::node_count(g.as_ref()),
            Self::Recording(rg) => GraphRead::node_count(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_count(&self) -> usize {
        match self {
            Self::Memory(g) => GraphRead::edge_count(g),
            Self::Mapped(g) => GraphRead::edge_count(g),
            Self::Disk(g) => GraphRead::edge_count(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_count(rg.as_ref()),
        }
    }

    #[inline]
    fn node_bound(&self) -> usize {
        match self {
            Self::Memory(g) => GraphRead::node_bound(g),
            Self::Mapped(g) => GraphRead::node_bound(g),
            Self::Disk(g) => GraphRead::node_bound(g.as_ref()),
            Self::Recording(rg) => GraphRead::node_bound(rg.as_ref()),
        }
    }

    #[inline]
    fn is_memory(&self) -> bool {
        match self {
            Self::Memory(_) => true,
            Self::Recording(rg) => GraphRead::is_memory(rg.as_ref()),
            _ => false,
        }
    }

    #[inline]
    fn is_mapped(&self) -> bool {
        match self {
            Self::Mapped(_) => true,
            Self::Recording(rg) => GraphRead::is_mapped(rg.as_ref()),
            _ => false,
        }
    }

    #[inline]
    fn is_disk(&self) -> bool {
        match self {
            Self::Disk(_) => true,
            Self::Recording(rg) => GraphRead::is_disk(rg.as_ref()),
            _ => false,
        }
    }

    #[inline(always)]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        match self {
            Self::Memory(g) => GraphRead::node_type_of(g, idx),
            Self::Mapped(g) => GraphRead::node_type_of(g, idx),
            Self::Disk(g) => GraphRead::node_type_of(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::node_type_of(rg.as_ref(), idx),
        }
    }

    #[inline(always)]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        match self {
            Self::Memory(g) => GraphRead::node_weight(g, idx),
            Self::Mapped(g) => GraphRead::node_weight(g, idx),
            Self::Disk(g) => GraphRead::node_weight(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::node_weight(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphRead::get_node_property(g, idx, key),
            Self::Mapped(g) => GraphRead::get_node_property(g, idx, key),
            Self::Disk(g) => GraphRead::get_node_property(g.as_ref(), idx, key),
            Self::Recording(rg) => GraphRead::get_node_property(rg.as_ref(), idx, key),
        }
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphRead::get_node_id(g, idx),
            Self::Mapped(g) => GraphRead::get_node_id(g, idx),
            Self::Disk(g) => GraphRead::get_node_id(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::get_node_id(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphRead::get_node_title(g, idx),
            Self::Mapped(g) => GraphRead::get_node_title(g, idx),
            Self::Disk(g) => GraphRead::get_node_title(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::get_node_title(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        match self {
            Self::Memory(g) => GraphRead::str_prop_eq(g, idx, key, target),
            Self::Mapped(g) => GraphRead::str_prop_eq(g, idx, key, target),
            Self::Disk(g) => GraphRead::str_prop_eq(g.as_ref(), idx, key, target),
            Self::Recording(rg) => GraphRead::str_prop_eq(rg.as_ref(), idx, key, target),
        }
    }

    #[inline]
    fn node_indices(&self) -> crate::graph::core::iterators::GraphNodeIndices<'_> {
        match self {
            Self::Memory(g) => GraphRead::node_indices(g),
            Self::Mapped(g) => GraphRead::node_indices(g),
            Self::Disk(g) => GraphRead::node_indices(g.as_ref()),
            Self::Recording(rg) => GraphRead::node_indices(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_indices(&self) -> crate::graph::core::iterators::GraphEdgeIndices<'_> {
        match self {
            Self::Memory(g) => GraphRead::edge_indices(g),
            Self::Mapped(g) => GraphRead::edge_indices(g),
            Self::Disk(g) => GraphRead::edge_indices(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_indices(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_references(&self) -> crate::graph::core::iterators::GraphEdgeReferences<'_> {
        match self {
            Self::Memory(g) => GraphRead::edge_references(g),
            Self::Mapped(g) => GraphRead::edge_references(g),
            Self::Disk(g) => GraphRead::edge_references(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_references(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        match self {
            Self::Memory(g) => GraphRead::edge_weights(g),
            Self::Mapped(g) => GraphRead::edge_weights(g),
            Self::Disk(g) => GraphRead::edge_weights(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_weights(rg.as_ref()),
        }
    }

    #[inline]
    fn edges_directed(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
    ) -> crate::graph::core::iterators::GraphEdges<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges_directed(g, idx, dir),
            Self::Mapped(g) => GraphRead::edges_directed(g, idx, dir),
            Self::Disk(g) => GraphRead::edges_directed(g.as_ref(), idx, dir),
            Self::Recording(rg) => GraphRead::edges_directed(rg.as_ref(), idx, dir),
        }
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> crate::graph::core::iterators::GraphEdges<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges(g, idx),
            Self::Mapped(g) => GraphRead::edges(g, idx),
            Self::Disk(g) => GraphRead::edges(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::edges(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> crate::graph::core::iterators::GraphEdges<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges_directed_filtered(g, idx, dir, conn_type_filter),
            Self::Mapped(g) => GraphRead::edges_directed_filtered(g, idx, dir, conn_type_filter),
            Self::Disk(g) => {
                GraphRead::edges_directed_filtered(g.as_ref(), idx, dir, conn_type_filter)
            }
            Self::Recording(rg) => {
                GraphRead::edges_directed_filtered(rg.as_ref(), idx, dir, conn_type_filter)
            }
        }
    }

    #[inline]
    fn edges_connecting(
        &self,
        a: NodeIndex,
        b: NodeIndex,
    ) -> crate::graph::core::iterators::GraphEdgesConnecting<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges_connecting(g, a, b),
            Self::Mapped(g) => GraphRead::edges_connecting(g, a, b),
            Self::Disk(g) => GraphRead::edges_connecting(g.as_ref(), a, b),
            Self::Recording(rg) => GraphRead::edges_connecting(rg.as_ref(), a, b),
        }
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        match self {
            Self::Memory(g) => GraphRead::edge_weight(g, idx),
            Self::Mapped(g) => GraphRead::edge_weight(g, idx),
            Self::Disk(g) => GraphRead::edge_weight(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::edge_weight(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        match self {
            Self::Memory(g) => GraphRead::find_edge(g, a, b),
            Self::Mapped(g) => GraphRead::find_edge(g, a, b),
            Self::Disk(g) => GraphRead::find_edge(g.as_ref(), a, b),
            Self::Recording(rg) => GraphRead::find_edge(rg.as_ref(), a, b),
        }
    }

    #[inline(always)]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        match self {
            Self::Memory(g) => GraphRead::edge_endpoints(g, idx),
            Self::Mapped(g) => GraphRead::edge_endpoints(g, idx),
            Self::Disk(g) => GraphRead::edge_endpoints(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::edge_endpoints(rg.as_ref(), idx),
        }
    }

    #[inline(always)]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        match self {
            Self::Memory(g) => GraphRead::edge_endpoint_keys(g),
            Self::Mapped(g) => GraphRead::edge_endpoint_keys(g),
            Self::Disk(g) => GraphRead::edge_endpoint_keys(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_endpoint_keys(rg.as_ref()),
        }
    }

    #[inline]
    fn neighbors_directed(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
    ) -> crate::graph::core::iterators::GraphNeighbors<'_> {
        match self {
            Self::Memory(g) => GraphRead::neighbors_directed(g, idx, dir),
            Self::Mapped(g) => GraphRead::neighbors_directed(g, idx, dir),
            Self::Disk(g) => GraphRead::neighbors_directed(g.as_ref(), idx, dir),
            Self::Recording(rg) => GraphRead::neighbors_directed(rg.as_ref(), idx, dir),
        }
    }

    #[inline]
    fn neighbors_undirected(
        &self,
        idx: NodeIndex,
    ) -> crate::graph::core::iterators::GraphNeighbors<'_> {
        match self {
            Self::Memory(g) => GraphRead::neighbors_undirected(g, idx),
            Self::Mapped(g) => GraphRead::neighbors_undirected(g, idx),
            Self::Disk(g) => GraphRead::neighbors_undirected(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::neighbors_undirected(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        match self {
            Self::Memory(g) => GraphRead::sources_for_conn_type_bounded(g, conn_type, max),
            Self::Mapped(g) => GraphRead::sources_for_conn_type_bounded(g, conn_type, max),
            Self::Disk(g) => GraphRead::sources_for_conn_type_bounded(g.as_ref(), conn_type, max),
            Self::Recording(rg) => {
                GraphRead::sources_for_conn_type_bounded(rg.as_ref(), conn_type, max)
            }
        }
    }

    #[inline]
    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        match self {
            Self::Memory(g) => GraphRead::lookup_peer_counts(g, conn_type),
            Self::Mapped(g) => GraphRead::lookup_peer_counts(g, conn_type),
            Self::Disk(g) => GraphRead::lookup_peer_counts(g.as_ref(), conn_type),
            Self::Recording(rg) => GraphRead::lookup_peer_counts(rg.as_ref(), conn_type),
        }
    }

    #[inline]
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        match self {
            Self::Memory(g) => GraphRead::lookup_by_property_eq(g, node_type, property, value),
            Self::Mapped(g) => GraphRead::lookup_by_property_eq(g, node_type, property, value),
            Self::Disk(g) => {
                GraphRead::lookup_by_property_eq(g.as_ref(), node_type, property, value)
            }
            Self::Recording(rg) => {
                GraphRead::lookup_by_property_eq(rg.as_ref(), node_type, property, value)
            }
        }
    }

    #[inline]
    fn lookup_by_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        match self {
            Self::Memory(g) => {
                GraphRead::lookup_by_property_prefix(g, node_type, property, prefix, limit)
            }
            Self::Mapped(g) => {
                GraphRead::lookup_by_property_prefix(g, node_type, property, prefix, limit)
            }
            Self::Disk(g) => {
                GraphRead::lookup_by_property_prefix(g.as_ref(), node_type, property, prefix, limit)
            }
            Self::Recording(rg) => GraphRead::lookup_by_property_prefix(
                rg.as_ref(),
                node_type,
                property,
                prefix,
                limit,
            ),
        }
    }

    #[inline]
    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        match self {
            Self::Memory(g) => GraphRead::lookup_by_property_eq_any_type(g, property, value),
            Self::Mapped(g) => GraphRead::lookup_by_property_eq_any_type(g, property, value),
            Self::Disk(g) => GraphRead::lookup_by_property_eq_any_type(g.as_ref(), property, value),
            Self::Recording(rg) => {
                GraphRead::lookup_by_property_eq_any_type(rg.as_ref(), property, value)
            }
        }
    }

    #[inline]
    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        match self {
            Self::Memory(g) => {
                GraphRead::lookup_by_property_prefix_any_type(g, property, prefix, limit)
            }
            Self::Mapped(g) => {
                GraphRead::lookup_by_property_prefix_any_type(g, property, prefix, limit)
            }
            Self::Disk(g) => {
                GraphRead::lookup_by_property_prefix_any_type(g.as_ref(), property, prefix, limit)
            }
            Self::Recording(rg) => {
                GraphRead::lookup_by_property_prefix_any_type(rg.as_ref(), property, prefix, limit)
            }
        }
    }

    #[inline]
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: petgraph::Direction,
        deadline: Option<std::time::Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        match self {
            Self::Memory(g) => GraphRead::count_edges_grouped_by_peer(g, conn_type, dir, deadline),
            Self::Mapped(g) => GraphRead::count_edges_grouped_by_peer(g, conn_type, dir, deadline),
            Self::Disk(g) => {
                GraphRead::count_edges_grouped_by_peer(g.as_ref(), conn_type, dir, deadline)
            }
            Self::Recording(rg) => {
                GraphRead::count_edges_grouped_by_peer(rg.as_ref(), conn_type, dir, deadline)
            }
        }
    }

    #[inline]
    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: petgraph::Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
    ) -> Result<usize, String> {
        match self {
            Self::Memory(g) => {
                GraphRead::count_edges_filtered(g, node, dir, conn_type, other_node_type, deadline)
            }
            Self::Mapped(g) => {
                GraphRead::count_edges_filtered(g, node, dir, conn_type, other_node_type, deadline)
            }
            Self::Disk(g) => GraphRead::count_edges_filtered(
                g.as_ref(),
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
            ),
            Self::Recording(rg) => GraphRead::count_edges_filtered(
                rg.as_ref(),
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
            ),
        }
    }

    #[inline]
    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: petgraph::Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        match self {
            Self::Memory(g) => GraphRead::iter_peers_filtered(g, node, dir, conn_type),
            Self::Mapped(g) => GraphRead::iter_peers_filtered(g, node, dir, conn_type),
            Self::Disk(g) => GraphRead::iter_peers_filtered(g.as_ref(), node, dir, conn_type),
            Self::Recording(rg) => {
                GraphRead::iter_peers_filtered(rg.as_ref(), node, dir, conn_type)
            }
        }
    }

    #[inline]
    fn reset_arenas(&self) {
        match self {
            Self::Disk(g) => GraphRead::reset_arenas(g.as_ref()),
            Self::Recording(rg) => GraphRead::reset_arenas(rg.as_ref()),
            _ => {}
        }
    }
}

impl GraphWrite for GraphBackend {
    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        match self {
            Self::Memory(g) => GraphWrite::node_weight_mut(g, idx),
            Self::Mapped(g) => GraphWrite::node_weight_mut(g, idx),
            Self::Disk(g) => GraphWrite::node_weight_mut(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::node_weight_mut(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        match self {
            Self::Memory(g) => GraphWrite::edge_weight_mut(g, idx),
            Self::Mapped(g) => GraphWrite::edge_weight_mut(g, idx),
            Self::Disk(g) => GraphWrite::edge_weight_mut(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::edge_weight_mut(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        match self {
            Self::Memory(g) => GraphWrite::add_node(g, data),
            Self::Mapped(g) => GraphWrite::add_node(g, data),
            Self::Disk(g) => GraphWrite::add_node(g.as_mut(), data),
            Self::Recording(rg) => GraphWrite::add_node(rg.as_mut(), data),
        }
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        match self {
            Self::Memory(g) => GraphWrite::remove_node(g, idx),
            Self::Mapped(g) => GraphWrite::remove_node(g, idx),
            Self::Disk(g) => GraphWrite::remove_node(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::remove_node(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        match self {
            Self::Memory(g) => GraphWrite::add_edge(g, a, b, data),
            Self::Mapped(g) => GraphWrite::add_edge(g, a, b, data),
            Self::Disk(g) => GraphWrite::add_edge(g.as_mut(), a, b, data),
            Self::Recording(rg) => GraphWrite::add_edge(rg.as_mut(), a, b, data),
        }
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        match self {
            Self::Memory(g) => GraphWrite::remove_edge(g, idx),
            Self::Mapped(g) => GraphWrite::remove_edge(g, idx),
            Self::Disk(g) => GraphWrite::remove_edge(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::remove_edge(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        match self {
            Self::Disk(g) => GraphWrite::update_row_id(g.as_mut(), node_idx, row_id),
            Self::Recording(rg) => GraphWrite::update_row_id(rg.as_mut(), node_idx, row_id),
            _ => {}
        }
    }

    #[inline]
    fn flush_pending_writes(&mut self) {
        match self {
            Self::Memory(g) => GraphWrite::flush_pending_writes(g),
            Self::Mapped(g) => GraphWrite::flush_pending_writes(g),
            Self::Disk(g) => GraphWrite::flush_pending_writes(g.as_mut()),
            Self::Recording(rg) => GraphWrite::flush_pending_writes(rg.as_mut()),
        }
    }
}
