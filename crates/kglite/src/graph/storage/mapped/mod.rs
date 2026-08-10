//! mmap-backed columnar storage backend.
//!
//! Owns the `MappedGraph` backend type and its lazy per-connection-type /
//! per-property indexes, plus the mmap primitives (`mmap_vec`) and the
//! mmap-backed column store the backend spills into.
//!
//! Split out of `storage/mod.rs` when that file passed its 800-line module
//! cap: the mapped backend's own types belong with the rest of the mapped
//! backend, and `storage/mod.rs` keeps only the traits and the memory
//! backend. `MappedGraph` is re-exported from `storage` so every existing
//! `crate::graph::storage::MappedGraph` path still resolves.

pub mod column_store;
pub mod mmap_vec;

use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::undo::UndoJournal;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

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
    /// **The column stores this backend owns** — see `MemoryGraph`'s field of
    /// the same name. Mapped is the mode where this matters most: its nodes
    /// carry *no* row copy of their properties at all, so the store map is
    /// their only property storage.
    pub(crate) column_stores: HashMap<InternedKey, Arc<ColumnStore>>,
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
    /// Statement-scoped inverse-op buffer. `Some` only while a mutating
    /// Cypher statement holds a rollback checkpoint; `None` is the steady
    /// state, so reads pay nothing and writes pay one discriminant check.
    /// See [`crate::graph::storage::undo`].
    ///
    /// Mapped journals for exactly the same reason memory does: `inner` is a
    /// heap `StableDiGraph`, so every `UndoEntry` variant — all of which are
    /// keyed on a petgraph `NodeIndex`/`EdgeIndex` — is expressible here. The
    /// mmap spilling that `StorageMode::Mapped` turns on lives in the
    /// *columnar property store*, not in the node/edge graph.
    pub(crate) undo: Option<Box<UndoJournal>>,
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

impl Deref for MappedGraph {
    type Target = StableDiGraph<NodeData, EdgeData>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
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
            column_stores: HashMap::new(),
            type_index: RwLock::new(HashMap::new()),
            property_index: RwLock::new(HashMap::new()),
            global_property_index: RwLock::new(HashMap::new()),
            undo: None,
        })
    }
}
