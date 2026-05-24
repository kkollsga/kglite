//! Storage-backend types and `GraphRead` / `GraphWrite` traits.
//!
//! Anchor for the 0.8.0 storage-architecture refactor.
//! Every backend implements [`GraphRead`] / [`GraphWrite`] directly;
//! the [`crate::graph::schema::GraphBackend`] enum is a dumb dispatcher.
//! Per-backend trait impls live in [`crate::graph::storage::impls`].
//!
//! - [`MemoryGraph`] — heap-resident, petgraph `StableDiGraph`.
//! - [`MappedGraph`] — mmap-columnar-spill variant (Phase 5 promoted
//!   this from a type alias to a distinct struct so its trait impls
//!   can diverge from memory's once the column ownership differs).
//! - [`crate::graph::storage::disk::graph::DiskGraph`] — CSR + mmap
//!   columns.
//!
//! Rule for new storage operations: add the method to [`GraphRead`] or
//! [`GraphWrite`] first, implement per-backend, and let the
//! `GraphBackend` dispatcher route to them — never the other way.

pub mod backend;
pub mod column_store;
pub mod disk;
pub mod interner;
pub mod lookups;
pub mod mapped;
pub mod mapped_graph_impl;
pub mod memory;
pub mod type_build_meta;

use crate::datatypes::Value;
use crate::graph::core::iterators::GraphEdgeRef;
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use petgraph::Direction;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, RwLock};
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// GraphRead — unified read interface over storage backends
// ──────────────────────────────────────────────────────────────────────────

/// Read-side interface shared by every storage backend.
///
/// Phase 0.3 seeded this trait with counts + single-property reads.
/// Phase 1 expanded it to cover iteration, neighbour lookup, backend-kind
/// predicates, and disk-only helpers. Phase 3 converted iterator-returning
/// methods to GATs (associated types with lifetime parameters) and promoted
/// the remaining inherent edge accessors (`edges`, `edge_references`,
/// `edge_weight`, `edge_indices`, `find_edge`, `edges_connecting`,
/// `edge_weights`) onto the trait.
///
/// Implemented today for [`crate::graph::schema::GraphBackend`];
/// per-backend impls (`MemoryGraph`, `DiskGraph`) will land in Phase 5
/// alongside the columnar cleanup that lets them diverge meaningfully.
///
/// ### GATs and object-safety
///
/// The iterator methods use generic associated types (e.g.
/// [`GraphRead::EdgesIter`]). This makes the trait **non-object-safe**:
/// `&dyn GraphRead` does not compile. All consumers take `&impl GraphRead`
/// (monomorphised) instead. Two methods that need type erasure for
/// backend-specific fast paths (`iter_peers_filtered`, `edge_endpoint_keys`)
/// return `Box<dyn Iterator<…> + 'a>` explicitly and stay non-GAT; they
/// would otherwise require a second associated type per method.
///
/// ### Disk-only helpers
///
/// Methods such as [`GraphRead::sources_for_conn_type_bounded`],
/// [`GraphRead::lookup_peer_counts`], and [`GraphRead::iter_peers_filtered`]
/// have meaningful implementations only on the disk backend (they read
/// from persistent indexes built at `.kgl` load time). Memory and mapped
/// backends return `None` / fall back via `edges_directed`. The
/// `Option` / fallback contract is preserved from the pre-refactor
/// inherent methods so callers do not need to change their handling.
pub trait GraphRead {
    // ─────────────── generic associated types ───────────────

    /// Iterator over all live node indices.
    type NodeIndicesIter<'a>: Iterator<Item = NodeIndex>
    where
        Self: 'a;

    /// Iterator over all live edge indices.
    type EdgeIndicesIter<'a>: Iterator<Item = EdgeIndex>
    where
        Self: 'a;

    /// Iterator over edges incident to a node (directed).
    type EdgesIter<'a>: Iterator<Item = GraphEdgeRef<'a>>
    where
        Self: 'a;

    /// Iterator over all edges in the graph (yielded as `GraphEdgeRef`).
    type EdgeReferencesIter<'a>: Iterator<Item = GraphEdgeRef<'a>>
    where
        Self: 'a;

    /// Iterator over edges connecting a given pair of nodes.
    type EdgesConnectingIter<'a>: Iterator<Item = GraphEdgeRef<'a>>
    where
        Self: 'a;

    /// Iterator over neighbour node indices.
    type NeighborsIter<'a>: Iterator<Item = NodeIndex>
    where
        Self: 'a;
    // ─────────────── counts / backend identity ───────────────

    /// Total live node count across all types.
    fn node_count(&self) -> usize;

    /// Total live edge count.
    fn edge_count(&self) -> usize;

    /// Upper bound on node indices (petgraph `node_bound`). May exceed
    /// [`GraphRead::node_count`] when nodes have been removed from a
    /// `StableDiGraph` without vacuuming.
    fn node_bound(&self) -> usize;

    /// `true` for heap-resident [`GraphBackend::Memory`]. Used by
    /// `recording.rs` tests to verify backend identity.
    #[allow(dead_code)]
    fn is_memory(&self) -> bool;

    /// `true` for the mmap-Columnar [`GraphBackend::Mapped`] variant.
    fn is_mapped(&self) -> bool {
        false
    }

    /// `true` for disk-backed [`GraphBackend::Disk`] (CSR + mmap columns).
    fn is_disk(&self) -> bool {
        false
    }

    // ─────────────── per-node reads ───────────────

    /// Node type key for a given index. `None` if the node has been removed.
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey>;

    /// Borrow the full NodeData. **Escape hatch** — prefer granular reads
    /// ([`GraphRead::get_node_property`], [`GraphRead::get_node_id`], etc.)
    /// in hot loops. On the disk backend, materialises NodeData through
    /// the per-query arena, which is cheap per-call but accumulates if
    /// called many times without [`GraphRead::reset_arenas`].
    ///
    /// Named `node_weight` for consistency with petgraph's `StableDiGraph`
    /// primitive, which is the heap-backed implementation of this method.
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData>;

    /// Read a single property without full NodeData materialisation.
    /// Used by the hot WHERE-scan path. Returns `None` if the property
    /// is missing or set to `Value::Null`.
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value>;

    /// Read the node id (handles mapped-mode sentinel values).
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value>;

    /// Read the node title (handles mapped-mode sentinel values).
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value>;

    /// Zero-allocation string-equality check for a property against `target`.
    /// Skips the `Value::String(owned)` materialisation that `get_node_property`
    /// would do on mapped graphs. Used by the Cypher executor to short-circuit
    /// `WHERE n.strProp = 'literal'` scans.
    ///
    /// Returns:
    /// - `None` — property is missing or null for this row
    /// - `Some(true)` — stored value equals `target` byte-for-byte
    /// - `Some(false)` — stored value is present but differs
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool>;

    // ─────────────── iteration ───────────────

    /// Iterator over all live node indices.
    fn node_indices(&self) -> Self::NodeIndicesIter<'_>;

    /// Iterator over all live edge indices.
    fn edge_indices(&self) -> Self::EdgeIndicesIter<'_>;

    /// Iterator over every live edge in the graph, yielding
    /// [`GraphEdgeRef`] with materialised `EdgeData`.
    fn edge_references(&self) -> Self::EdgeReferencesIter<'_>;

    /// Iterator over every live edge's weight (EdgeData). Boxed because
    /// petgraph's underlying `edge_weights` returns an opaque
    /// `impl Iterator` that can't be named as a GAT associated type.
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a>;

    // ─────────────── per-node edges / neighbours ───────────────

    /// Directed edges incident to `idx` (yielded as [`GraphEdgeRef`]).
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> Self::EdgesIter<'_>;

    /// Default-direction edges (outgoing) incident to `idx` — matches
    /// petgraph's `StableDiGraph::edges`.
    fn edges(&self, idx: NodeIndex) -> Self::EdgesIter<'_>;

    /// Like [`GraphRead::edges_directed`] but the disk backend can
    /// pre-filter by connection type, skipping EdgeData materialisation
    /// for non-matching edges. Memory/mapped callers still post-filter.
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> Self::EdgesIter<'_>;

    /// Iterator over edges directly connecting `a` → `b`.
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> Self::EdgesConnectingIter<'_>;

    /// Borrow a single edge's weight.
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData>;

    /// First edge index from `a` to `b`, if one exists.
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex>;

    /// `(source, target)` endpoints for an edge, without materialising
    /// EdgeData. `None` if the edge has been removed.
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)>;

    /// Iterate edge endpoint metadata without materialising EdgeData.
    /// Yields `(source, target, connection_type)` for every live edge.
    /// On the disk backend this reads mmap'd `edge_endpoints` directly
    /// (zero heap allocation per edge).
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a>;

    /// Neighbours reached via an edge in `dir`.
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> Self::NeighborsIter<'_>;

    /// Neighbours reached via an edge in either direction.
    fn neighbors_undirected(&self, idx: NodeIndex) -> Self::NeighborsIter<'_>;

    // ─────────────── disk-only helpers (Option / fallback contract) ─────

    /// Source nodes with outgoing edges of a given connection type,
    /// read from the disk inverted index. `None` on memory/mapped or on
    /// older disk graphs without this index.
    ///
    /// `max` caps the number of sources returned to avoid eager
    /// allocations when the pattern executor will truncate downstream.
    fn sources_for_conn_type_bounded(
        &self,
        _conn_type: InternedKey,
        _max: Option<usize>,
    ) -> Option<Vec<u32>> {
        None
    }

    /// Per-peer edge count for a connection type, read from the
    /// histogram cache on the disk backend. `None` on memory/mapped or
    /// on older disk graphs (caller falls back to
    /// [`GraphRead::count_edges_grouped_by_peer`]).
    fn lookup_peer_counts(&self, _conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        None
    }

    /// Exact-match lookup on a persistent string property index.
    ///
    /// Returns `Some(Vec)` (possibly empty) when an index for
    /// `(node_type, property)` exists; returns `None` when no index
    /// exists — the caller falls back to a scan. Default `None` for
    /// backends without persistent indexes; the disk backend overrides
    /// to consult its mmap'd `PropertyIndex`.
    fn lookup_by_property_eq(
        &self,
        _node_type: &str,
        _property: &str,
        _value: &str,
    ) -> Option<Vec<NodeIndex>> {
        None
    }

    /// Prefix lookup (STARTS WITH) on a persistent string property
    /// index. Same `None`/`Some` semantics as
    /// [`GraphRead::lookup_by_property_eq`].
    fn lookup_by_property_prefix(
        &self,
        _node_type: &str,
        _property: &str,
        _prefix: &str,
        _limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        None
    }

    /// Exact-match lookup across every node type using a cross-type
    /// global index. Returns `Some(Vec)` (possibly empty) when a
    /// global index for `property` exists; `None` otherwise (caller
    /// falls back to scan or per-type iteration).
    fn lookup_by_property_eq_any_type(
        &self,
        _property: &str,
        _value: &str,
    ) -> Option<Vec<NodeIndex>> {
        None
    }

    /// Prefix lookup (STARTS WITH) across every node type. Same
    /// `None`/`Some` semantics as [`GraphRead::lookup_by_property_eq_any_type`].
    fn lookup_by_property_prefix_any_type(
        &self,
        _property: &str,
        _prefix: &str,
        _limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        None
    }

    /// Count edges of a connection type grouped by peer node, via a full
    /// scan. Every backend implements this — disk uses sequential CSR
    /// I/O; memory/mapped iterate petgraph edges.
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String>;

    /// Count edges from/to `node` matching optional connection-type and
    /// peer-node-type filters. On disk uses sorted-CSR binary search
    /// (O(log D + matching)); on memory/mapped iterates without
    /// allocation.
    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
    ) -> Result<usize, String>;

    /// Peer-iteration fast path used by the Cypher edge-no-variable
    /// optimisation. Yields `(peer, edge_idx)` pairs **without**
    /// materialising EdgeData — on disk this halves I/O on Wikidata-scale
    /// graphs.
    ///
    /// Default implementation falls back to [`GraphRead::edges_directed`]
    /// + post-filter. The disk backend overrides with a direct CSR walk.
    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        let iter = self.edges_directed(node, dir).filter_map(move |er| {
            if let Some(want) = conn_type {
                if er.weight().connection_type.as_u64() != want {
                    return None;
                }
            }
            let peer = match dir {
                Direction::Outgoing => er.target(),
                Direction::Incoming => er.source(),
            };
            Some((peer, er.id()))
        });
        Box::new(iter)
    }

    /// Reset per-query materialisation arenas. No-op on memory/mapped;
    /// frees NodeData / EdgeData allocated during the previous query on
    /// the disk backend. Called between Cypher queries to cap memory.
    fn reset_arenas(&self) {}
}

// ──────────────────────────────────────────────────────────────────────────
// GraphWrite — unified mutation interface over storage backends
// ──────────────────────────────────────────────────────────────────────────

/// Write-side interface shared by every storage backend.
///
/// Phase 2 of the 0.8.0 refactor. Pulls together the
/// mutation methods that were inherent on
/// [`crate::graph::schema::GraphBackend`] so write-path files can
/// dispatch through the trait instead of matching on the backend
/// variant.
///
/// Transaction bookkeeping (OCC `version`, `read_only`,
/// `schema_locked`) lives on [`crate::graph::schema::DirGraph`], not
/// on this trait — no backend has its own OCC state, and validation
/// against the schema metadata sits architecturally above storage.
/// Documented decision: keep transactions on DirGraph.
///
/// Dispatch guidance: `&mut impl GraphWrite` everywhere. Because
/// `GraphWrite: GraphRead` and `GraphRead` is non-object-safe (GAT
/// iterators — see [`GraphRead`] docs), `&mut dyn GraphWrite` also
/// does not compile. All mutation consumers take `&mut impl GraphWrite`.
pub trait GraphWrite: GraphRead {
    /// Mutable borrow of the full NodeData. Escape hatch for property
    /// mutation — prefer higher-level helpers (`NodeData::set_property`,
    /// `NodeData::remove_property`) where available.
    ///
    /// **Disk backend staging contract (0.9.0 Cluster 6):** on disk,
    /// `node_weight_mut` does NOT mutate the live store directly. It
    /// stages writes in an internal `node_mut_cache` to dodge the
    /// `Arc<ColumnStore>` share-clone storm; the cache is drained
    /// into `column_stores` by the next call to
    /// [`GraphWrite::flush_pending_writes`] (or any subsequent
    /// `&mut self` op via `clear_arenas`).
    ///
    /// **Callers MUST call `flush_pending_writes()` before any
    /// subsequent `&self` read of the same node**, or the read will
    /// return the pre-write value from `column_stores`. The Cypher
    /// executor (`execute_mutable`) does this automatically after
    /// every SET/REMOVE/MERGE clause; new code paths that mutate
    /// through this method must replicate that pattern. A debug-only
    /// assertion in `DiskGraph::node_weight` warns if a staged write
    /// is shadowed by a read.
    ///
    /// Memory + Mapped backends mutate `StableDiGraph` in place — no
    /// flush needed.
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData>;

    /// Mutable borrow of the full EdgeData.
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData>;

    /// Insert a new node, returning its assigned index.
    fn add_node(&mut self, data: NodeData) -> NodeIndex;

    /// Remove a node, returning its NodeData if present. On the disk
    /// backend this writes a tombstone; on memory/mapped the
    /// StableDiGraph entry is removed in-place.
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData>;

    /// Insert a directed edge from `a` to `b`.
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex;

    /// Remove an edge, returning its EdgeData if present.
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData>;

    /// Disk-only: after a columnar-properties row is materialised for a
    /// newly-added node, persist the per-type `row_id` back to the
    /// disk slot so subsequent reads find the correct columnar row.
    /// No-op on memory/mapped (their slot storage carries no separate
    /// row_id field). Invariant: callers must invoke this only after
    /// they have already assigned `PropertyStorage::Columnar { row_id }`
    /// to the node's `NodeData`; otherwise disk reads will drift.
    fn update_row_id(&mut self, _node_idx: NodeIndex, _row_id: u32) {}

    /// Flush any pending mutation state into the steady-state stores so
    /// subsequent `&self` reads observe the writes.
    ///
    /// Memory/mapped backends mutate their `StableDiGraph` in place via
    /// `node_weight_mut` / `edge_weight_mut`, so reads see writes
    /// immediately — default no-op.
    ///
    /// Disk stages `node_weight_mut` / `edge_weight_mut` writes in
    /// `node_mut_cache` / `edge_mut_cache` to dodge `Arc<ColumnStore>`
    /// share-clone storms; those caches are otherwise drained lazily on
    /// the next `&mut self` op (e.g. on save). Without an explicit
    /// flush at end of a mutation query, a subsequent read goes through
    /// `node_weight` which reads `column_stores` directly and misses
    /// the staged writes — Cypher SET appears to silently no-op until
    /// the next `add_node`/`save`. Override on disk routes through
    /// `clear_arenas` (which already does the clone-apply-replace
    /// flush + arena reset).
    fn flush_pending_writes(&mut self) {}
}

// ──────────────────────────────────────────────────────────────────────────
// Newtype backends
// ──────────────────────────────────────────────────────────────────────────

/// Heap-resident in-memory graph backend. Wraps `StableDiGraph` and
/// `Deref`s to it so existing petgraph call sites compile unchanged.
#[derive(Clone, Debug, Default)]
pub struct MemoryGraph(pub(crate) StableDiGraph<NodeData, EdgeData>);

/// Memory-mapped in-memory graph backend — Phase 5 promoted this to a
/// distinct struct (previously a type alias for [`MemoryGraph`]) so
/// per-backend trait impls can diverge. 0.8.15 added a lazy per-
/// connection-type index to accelerate typed edge traversals and
/// aggregations — `MappedGraph` builds per-type inverted indexes on
/// first use, mirroring the `conn_type_index_*` / `peer_count_*`
/// structures `DiskGraph` persists on save but materialising them
/// in-memory from `StableDiGraph::edge_references()`.
#[derive(Debug, Default)]
pub struct MappedGraph {
    pub(crate) inner: StableDiGraph<NodeData, EdgeData>,
    /// Lazy per-conn-type index. Populated on first typed-edge query;
    /// cleared on any edge mutation. Each entry is `Arc` so the outer
    /// `RwLock` can be released before callers iterate the block.
    pub(crate) type_index: RwLock<HashMap<u64, Arc<MappedTypeIndex>>>,
    /// Lazy per-(node_type, property) string-value index. Mirrors the
    /// disk `PropertyIndex` and gives `MATCH (n:Type {prop: val})` a
    /// binary-search path instead of a full scan. Built on first
    /// `lookup_by_property_eq` / `_prefix` hit; cleared on any node
    /// mutation.
    pub(crate) property_index: RwLock<HashMap<(String, String), Arc<MappedPropertyIndex>>>,
    /// Lazy cross-type property index keyed by property name only.
    /// Backs `lookup_by_property_eq_any_type` / `_prefix_any_type`
    /// (used by untyped patterns like `MATCH (n {title: 'X'})`).
    pub(crate) global_property_index: RwLock<HashMap<String, Arc<MappedPropertyIndex>>>,
}

/// Per-conn-type edge index for `MappedGraph`. CSR-style layout
/// mirrors the disk backend's `conn_type_index_*` arrays but holds
/// `NodeIndex` / `EdgeIndex` directly (no id→index lookup needed —
/// `StableDiGraph`'s indices *are* the heap identity).
#[derive(Debug, Default)]
pub struct MappedTypeIndex {
    /// Distinct source nodes with ≥ 1 outgoing edge of this conn_type,
    /// sorted ascending. Binary-searchable in `edges_directed_filtered`.
    pub out_sources: Vec<NodeIndex>,
    /// CSR offsets into `out_edges`. Length = `out_sources.len() + 1`.
    pub out_offsets: Vec<u32>,
    /// Flat edge list. `out_edges[out_offsets[i]..out_offsets[i+1]]`
    /// are the outgoing edges from `out_sources[i]` of this conn_type.
    pub out_edges: Vec<EdgeIndex>,
    /// Same three, but for incoming edges keyed by target.
    pub in_sources: Vec<NodeIndex>,
    pub in_offsets: Vec<u32>,
    pub in_edges: Vec<EdgeIndex>,
    /// target → count of outgoing edges of this conn_type landing there.
    /// Powers `count_edges_grouped_by_peer(conn, Outgoing)`.
    pub out_peer_counts: HashMap<NodeIndex, i64>,
    /// source → count of outgoing edges of this conn_type from there.
    /// Powers `count_edges_grouped_by_peer(conn, Incoming)` (peer is
    /// the source per the trait's `dir=Incoming` semantics).
    pub in_peer_counts: HashMap<NodeIndex, i64>,
}

/// Sorted in-memory property index for `MappedGraph`. Mirrors the
/// parallel-array layout of disk's `PropertyIndex` (`keys` + `nodes`
/// sorted by `(key, node_idx)`) so equality and prefix lookups reduce
/// to the same binary-search + linear-scan primitives.
#[derive(Debug, Default)]
pub struct MappedPropertyIndex {
    /// Property string values, sorted lexicographically (ties broken
    /// by `nodes[i]`). Duplicates are adjacent, as in the disk layout.
    pub keys: Vec<String>,
    /// Parallel to `keys`. `nodes[i]` is the `NodeIndex` whose
    /// property value was `keys[i]`.
    pub nodes: Vec<NodeIndex>,
}

impl MappedPropertyIndex {
    /// Binary-search lower bound: index of first key >= `target`.
    fn lower_bound(&self, target: &str) -> usize {
        let mut lo = 0usize;
        let mut hi = self.keys.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.keys[mid].as_str() < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    pub fn lookup_eq(&self, value: &str) -> Vec<NodeIndex> {
        let start = self.lower_bound(value);
        let mut out = Vec::new();
        let mut i = start;
        while i < self.keys.len() && self.keys[i] == value {
            out.push(self.nodes[i]);
            i += 1;
        }
        out
    }

    pub fn lookup_prefix(&self, prefix: &str, limit: usize) -> Vec<NodeIndex> {
        if limit == 0 {
            return Vec::new();
        }
        let start = self.lower_bound(prefix);
        let mut out = Vec::with_capacity(limit.min(16));
        let mut i = start;
        while i < self.keys.len() && out.len() < limit {
            if !self.keys[i].starts_with(prefix) {
                break;
            }
            out.push(self.nodes[i]);
            i += 1;
        }
        out
    }
}

impl Clone for MappedGraph {
    fn clone(&self) -> Self {
        // All lazy indexes are derived state; drop them on clone and
        // let the clone rebuild on demand. Avoids `RwLock` clone
        // plumbing.
        Self {
            inner: self.inner.clone(),
            type_index: RwLock::new(HashMap::new()),
            property_index: RwLock::new(HashMap::new()),
            global_property_index: RwLock::new(HashMap::new()),
        }
    }
}

impl MemoryGraph {
    #[inline]
    pub fn new() -> Self {
        Self(StableDiGraph::new())
    }

    /// Borrow the inner `StableDiGraph`. Shared with [`MappedGraph`]
    /// for match arms that need the heap backend's petgraph view.
    #[inline]
    pub fn inner(&self) -> &StableDiGraph<NodeData, EdgeData> {
        &self.0
    }

    /// Mutable borrow of the inner `StableDiGraph`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut StableDiGraph<NodeData, EdgeData> {
        &mut self.0
    }
}

pub(super) fn flatten_to_csr(
    mut map: HashMap<NodeIndex, Vec<EdgeIndex>>,
) -> (Vec<NodeIndex>, Vec<u32>, Vec<EdgeIndex>) {
    let mut sources: Vec<NodeIndex> = map.keys().copied().collect();
    sources.sort_by_key(|n| n.index());
    let mut offsets: Vec<u32> = Vec::with_capacity(sources.len() + 1);
    let total: usize = map.values().map(|v| v.len()).sum();
    let mut flat: Vec<EdgeIndex> = Vec::with_capacity(total);
    offsets.push(0);
    for src in &sources {
        if let Some(edges) = map.remove(src) {
            flat.extend(edges);
        }
        offsets.push(flat.len() as u32);
    }
    (sources, offsets, flat)
}

// Read-only Deref for MemoryGraph / MappedGraph stays — petgraph's
// inherent read methods (`node_weight`, `edge_references`, etc.) are
// the same shape as the GraphRead trait methods, and trait dispatch
// is enforced explicitly elsewhere via UFCS or `use Trait`.
//
// DerefMut is REMOVED (0.9.0 Cluster 6 / D2 hygiene). Without it,
// callers that need a mutable petgraph view must go through
// `inner_mut()`, and any mutation that requires lazy-index
// invalidation must route through the GraphWrite trait. This kills
// the silent footgun: pre-fix, `g.add_node(data)` on `&mut MappedGraph`
// auto-deref'd to petgraph, bypassing
// `MappedGraph::invalidate_property_index()`. Post-fix, the same call
// site fails to compile and forces the author to choose explicitly.
impl Deref for MemoryGraph {
    type Target = StableDiGraph<NodeData, EdgeData>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for MappedGraph {
    type Target = StableDiGraph<NodeData, EdgeData>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// Serialize as the inner StableDiGraph so the on-disk binary format
// is unchanged between this refactor and pre-refactor code.
impl serde::Serialize for MemoryGraph {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(ser)
    }
}

impl<'de> serde::Deserialize<'de> for MemoryGraph {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        StableDiGraph::deserialize(de).map(MemoryGraph)
    }
}

impl serde::Serialize for MappedGraph {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.inner.serialize(ser)
    }
}

impl<'de> serde::Deserialize<'de> for MappedGraph {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        StableDiGraph::deserialize(de).map(|inner| MappedGraph {
            inner,
            type_index: RwLock::new(HashMap::new()),
            property_index: RwLock::new(HashMap::new()),
            global_property_index: RwLock::new(HashMap::new()),
        })
    }
}

pub mod impls;
pub mod recording;

// Phase-6 recording backend — re-exported so downstream consumers (and
// the Phase-6 parity test) can construct it without reaching into
// `storage::recording::`. DO NOT REMOVE despite unused-import warnings;
// `tests/test_phase6_parity.py::test_recording_graph_symbol_exported`
// asserts this exact line survives.
#[allow(unused_imports)]
pub use recording::RecordingGraph;
