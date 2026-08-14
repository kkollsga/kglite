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
pub(crate) mod forked;
pub mod interner;
pub mod lookups;
pub mod mapped;
pub mod mapped_graph_impl;
pub mod memory;
mod memory_graph_impl;
pub mod mode;
pub mod node_view;
pub mod overflow;
pub(crate) mod packed_codec;
pub mod property_storage;
pub(crate) mod slot_mirror;
pub mod type_build_meta;
pub mod undo;

use crate::datatypes::Value;
use crate::graph::core::iterators::GraphEdgeRef;
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
pub use crate::graph::storage::column_store::ColumnStore;
pub use crate::graph::storage::node_view::NodeView;
use crate::graph::storage::slot_mirror::SlotMirror;
use crate::graph::storage::undo::UndoJournal;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use petgraph::Direction;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, RwLock};
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// StrField — a borrowed read for string-only tests
// ──────────────────────────────────────────────────────────────────────────

/// One field read in the *only* form a string predicate needs.
///
/// Every columnar string read through [`ColumnStore::get`] materialises a
/// `Value::String`, one heap allocation per row — the whole cost of a
/// `CONTAINS` / `STARTS WITH` / `ENDS WITH` / `=` scan once construction
/// became columnar. `StrField` borrows out of the column instead, and keeps
/// the two non-string outcomes distinct because the resolution order in
/// [`NodeView::resolved_field`] depends on them: an *absent* field falls
/// through to the structural soft alias, a field holding a **non-string**
/// value does not.
///
/// `Cow` rather than `&str` for the one route that cannot borrow: the
/// overflow bag decodes a blob per read.
#[derive(Debug, Clone, PartialEq)]
pub enum StrField<'a> {
    /// The field holds a string.
    Str(std::borrow::Cow<'a, str>),
    /// The field holds a value that is not a string. No string test can pass,
    /// and no fallback applies — the field *resolved*.
    NotString,
    /// The field is absent or null for this row.
    Absent,
}

impl StrField<'_> {
    /// Apply a string test, answering `false` for every non-string outcome —
    /// which is what every string matcher does with a non-string value.
    #[inline]
    pub fn is(&self, test: impl FnOnce(&str) -> bool) -> bool {
        match self {
            StrField::Str(s) => test(s),
            StrField::NotString | StrField::Absent => false,
        }
    }
}

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
/// Implemented for [`crate::graph::schema::GraphBackend`] and directly for
/// the mapped and disk backends where their storage-specific iterators matter.
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

    /// All labels for a node that *this backend* can see, which is the
    /// primary type alone: secondary labels are not backend state at all —
    /// they live in `DirGraph::secondary_label_index`, one layer up, and
    /// `NodeData` carries none. No backend overrides this today, and one
    /// that did would still be missing the secondaries.
    ///
    /// **Callers wanting a node's real label set want
    /// `DirGraph::node_labels`**, which consults that index and returns
    /// `[primary, ...secondaries sorted by name]`. Consumers that only need
    /// the primary type should keep using `node_type_of` (no allocation).
    fn node_labels_of(&self, idx: NodeIndex) -> Vec<InternedKey> {
        match self.node_type_of(idx) {
            Some(key) => vec![key],
            None => Vec::new(),
        }
    }

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
    /// Equality is the engine's, not `str`'s: every implementation answers
    /// [`crate::graph::core::filtering::str_values_equal`], so a stored
    /// `'["Oslo"]'` equals `'Oslo'` here exactly as it does in `values_equal`,
    /// `IN` and the compiled scan predicates. A plain `==` here made a bare
    /// `n.tag = 'Oslo'` the one route that disagreed with the other seven.
    ///
    /// Returns:
    /// - `None` — property is missing or null for this row
    /// - `Some(true)` — stored value equals `target`
    /// - `Some(false)` — stored value is present but differs
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool>;

    // ─────────────── authoritative node views ───────────────
    //
    // These are *the* route for reading a node's properties. Reaching into
    // `NodeData` / `PropertyStorage` directly reads one replica of a columnar
    // type's store rather than the store the backend owns — see
    // `storage/node_view.rs`.
    //
    // Every method below is complete for columnar storage; the removed
    // `NodeData::property_iter` yielded nothing there.

    /// A borrowed read handle for one node, with its column store resolved
    /// once. Prefer this to [`GraphRead::node_weight`] whenever more than one
    /// property of the same node is read.
    ///
    /// On the disk backend the returned view borrows per-query arena memory;
    /// it must not outlive the `begin_query()` guard.
    #[inline]
    fn node_view(&self, idx: NodeIndex) -> Option<NodeView<'_>> {
        let data = self.node_weight(idx)?;
        // The node carries a row id; the store is the backend's, keyed by the
        // node's type. This is the single resolution point for columnar reads.
        let store = data.properties.columnar_row_id().and_then(|row_id| {
            self.column_store(data.node_type)
                .map(|store| (&**store, row_id))
        });
        Some(NodeView::new(data, store))
    }

    /// Every present property of a node as `(interned key, owned value)`.
    /// Empty when the node does not exist.
    #[inline]
    fn node_row_properties(&self, idx: NodeIndex) -> Vec<(InternedKey, Value)> {
        self.node_view(idx)
            .map(|v| v.property_pairs())
            .unwrap_or_default()
    }

    /// Every present property key of a node. Empty when the node does not
    /// exist.
    #[inline]
    fn node_property_keys(&self, idx: NodeIndex) -> Vec<InternedKey> {
        self.node_row_properties(idx)
            .into_iter()
            .map(|(k, _)| k)
            .collect()
    }

    /// `true` when the node has the property present and non-`Null`.
    #[inline]
    fn node_has_property(&self, idx: NodeIndex, key: InternedKey) -> bool {
        self.node_view(idx).is_some_and(|v| v.contains(key))
    }

    /// Number of present properties on a node; `0` when it does not exist.
    #[inline]
    fn node_property_count(&self, idx: NodeIndex) -> usize {
        self.node_view(idx).map_or(0, |v| v.property_count())
    }

    // ─────────────── column-store ownership (read side) ───────────────
    //
    // The backend is the sole owner of a columnar type's `ColumnStore`
    // (D1 Phase 3). A node carries only its `row_id`; the store is resolved
    // here, keyed by the node's type.

    /// The column store this backend owns for `type_key`, if the type is
    /// columnar.
    fn column_store(&self, type_key: InternedKey) -> Option<&Arc<ColumnStore>>;

    /// Every `(type_key, store)` this backend owns.
    fn column_stores_iter(&self)
        -> Box<dyn Iterator<Item = (InternedKey, &Arc<ColumnStore>)> + '_>;

    /// `true` when this backend owns at least one column store — i.e. the
    /// graph has been through `enable_columnar` (which `save()` calls).
    fn has_column_stores(&self) -> bool {
        self.column_stores_iter().next().is_some()
    }

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
    /// Mutable borrow of the full NodeData. Escape hatch for the record's
    /// own fields — for property mutation use `set_node_property` /
    /// `remove_node_property` on this trait, which route by storage
    /// variant (the removed `NodeData` mutators wrote only the node's
    /// replica, which columnar storage ignores).
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

    /// Like [`node_weight_mut`](Self::node_weight_mut) but **not** captured
    /// by a write-recording wrapper (the WAL `RecordingGraph`). For internal
    /// storage bookkeeping that must not surface as a logical mutation —
    /// notably the columnar-`SET` per-node `Arc<ColumnStore>` handle refresh,
    /// which touches every node of a type to re-point its handle after the
    /// master store was mutated. Recording those as user mutations would log
    /// the whole type per `SET` (O(N) WAL frames). Default = the recorded
    /// `node_weight_mut`; only the recording wrapper overrides it to bypass.
    fn node_weight_mut_silent(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        self.node_weight_mut(idx)
    }

    /// Mutable borrow of the full EdgeData.
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData>;

    // ─────────────── column-store ownership (write side) ───────────────

    /// Install (or replace) the store for `type_key`.
    fn install_column_store(&mut self, type_key: InternedKey, store: Arc<ColumnStore>);

    /// Mutable access to a type's store, for the copy-on-write master write.
    fn column_store_mut(&mut self, type_key: InternedKey) -> Option<&mut Arc<ColumnStore>>;

    /// Remove and return a type's store.
    fn take_column_store(&mut self, type_key: InternedKey) -> Option<Arc<ColumnStore>>;

    /// Drop every store (used by the rebuild half of `enable_columnar` and by
    /// the mid-build page-cache reclaim in the N-Triples loader).
    fn clear_column_stores(&mut self);

    // ─────────────── node property writes ───────────────
    //
    // A columnar node has no store handle of its own, so a property write
    // cannot be expressed on `&mut NodeData` alone: it needs the backend's
    // store *and* the node's `row_id` at once. These five are the only way to
    // write a node property, and they resolve the right route per storage
    // variant.

    /// Insert or update one property.
    fn set_node_property(&mut self, idx: NodeIndex, key: InternedKey, value: Value);

    /// Insert only when the key is absent or `Null` (Preserve conflict mode).
    fn set_node_property_if_absent(&mut self, idx: NodeIndex, key: InternedKey, value: Value);

    /// Remove a property, returning the prior value.
    fn remove_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value>;

    /// Mark a property cleared — writes `Null` rather than dropping the key, so
    /// a disk flush propagates the removal. Returns the prior value.
    fn clear_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value>;

    /// Replace the whole property set (Replace conflict mode).
    fn replace_node_properties(&mut self, idx: NodeIndex, pairs: Vec<(InternedKey, Value)>);

    /// Write a node's title.
    ///
    /// The sixth member of the family above, and it exists for the same reason
    /// they do: a columnar node's title lives in its store's reserved
    /// `__title__` column, not in the inline `NodeData.title` field, so a title
    /// write needs the backend's store and the node's `row_id` at once.
    ///
    /// It used to be written inline unconditionally, with `enable_columnar`
    /// detecting the divergence at `save()` time and rebuilding every store to
    /// consolidate it. That single save-side chokepoint was cheap only while a
    /// per-path master write was expensive; it is not any more, and paying an
    /// O(N) rebuild on the next save for one title write is the opposite of a
    /// bargain. The default below keeps the inline write for backends with no
    /// per-type store to write through.
    fn set_node_title(&mut self, idx: NodeIndex, value: Value) {
        if let Some(node) = self.node_weight_mut(idx) {
            node.title = value;
        }
    }

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
#[derive(Debug, Default)]
pub struct MemoryGraph {
    pub(crate) inner: StableDiGraph<NodeData, EdgeData>,
    /// **The column stores this backend owns**, keyed by node-type
    /// `InternedKey`.
    ///
    /// `FxHashMap`, not `HashMap`: this is probed once per `node_view`, i.e.
    /// once per columnar node per property access on every scan, and the key is
    /// an already-hashed `u64`. SipHash over 8 bytes measured ~14 ns per probe
    /// there — a +22% regression on `columnar_cypher_where` — against ~1 ns for
    /// FxHash. Same reasoning as the 0.9.x `FxHash` conversions elsewhere in the
    /// engine.
    pub(crate) column_stores: FxHashMap<InternedKey, Arc<ColumnStore>>,

    /// Lazy per-connection-type peer counts used by grouped Cypher
    /// aggregations. Derived state: empty on clone/load and invalidated by
    /// every edge mutation.
    pub(crate) peer_counts: RwLock<HashMap<u64, Arc<MemoryPeerCounts>>>,
    /// Statement-scoped inverse-op buffer. `Some` only while a mutating
    /// Cypher statement holds a rollback checkpoint; `None` is the steady
    /// state, so reads pay nothing and writes pay one discriminant check.
    /// See [`crate::graph::storage::undo`].
    pub(crate) undo: Option<Box<UndoJournal>>,
    /// Mirror of petgraph's node/edge free lists, so D2 Phase 2's overlay can
    /// *predict* the slot `add_node`/`add_edge` will hand out before the base
    /// graph is available to ask. See [`crate::graph::storage::slot_mirror`].
    pub(crate) slot_mirror: SlotMirror,
}

#[derive(Debug, Default)]
pub(crate) struct MemoryPeerCounts {
    pub(crate) by_target: Arc<HashMap<u32, i64>>,
    pub(crate) by_source: Arc<HashMap<u32, i64>>,
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
        &self.inner
    }
}

// Serialize as the inner StableDiGraph so the on-disk binary format
// is unchanged between this refactor and pre-refactor code.
impl serde::Serialize for MemoryGraph {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.inner.serialize(ser)
    }
}

impl<'de> serde::Deserialize<'de> for MemoryGraph {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        StableDiGraph::deserialize(de).map(MemoryGraph::from_graph)
    }
}

pub mod impls;
pub mod recording;

// The mapped backend's own types live with the rest of the mapped backend
// (`storage/mapped/`); re-exported here so `storage::MappedGraph` keeps
// resolving for every existing call site.
pub use mapped::{MappedGraph, MappedPropertyIndex, MappedTypeIndex};

// Phase-6 recording backend — re-exported so downstream consumers can
// construct it without reaching into `storage::recording::`. DO NOT REMOVE
// despite unused-import warnings; the centralized source-quality gate asserts
// this exact line survives.
#[cfg(test)]
#[path = "column_ownership_tests.rs"]
mod column_ownership_tests;

#[allow(unused_imports)]
pub use recording::RecordingGraph;
