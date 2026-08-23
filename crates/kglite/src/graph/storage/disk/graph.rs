// Disk-backed graph storage using CSR (Compressed Sparse Row) format.
//
// Memory budget: ~10% of the equivalent petgraph in-memory graph, for a
// graph *opened* from a disk directory — the edges are paged in on demand.
// For 100M nodes + 1B edges: ~5-6 GB RAM + OS page cache. Converting a
// resident in-memory graph with `enable_disk_mode()` does not reach that
// budget in the converting process: it builds these structures on top of
// what is already there.

use crate::datatypes::values::Value;
use crate::graph::core::iterators::{
    DiskEdgeIndices, DiskEdgeReferences, DiskEdges, DiskEdgesConnecting, DiskNeighbors,
    DiskNodeIndices,
};
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints, PendingEdge, TOMBSTONE_EDGE};
use super::edge_properties::EdgePropertyStore;
use super::property_index;

/// CSR + column binaries live in a per-segment subdirectory of the graph
/// root. Top-level files (disk_graph_meta.json, seg_manifest.json,
/// interner.json, metadata.json) stay at the graph root. Legacy graphs gated
/// by `DiskGraphMeta::csr_layout_version` == 0 use the flat layout
/// (everything at the root) — see `load_from_dir`.
pub(crate) fn segment_subdir(id: u32) -> String {
    // Past 999 `{:03}` widens naturally, which breaks lexicographic ordering;
    // callers order by the u32 parsed in `enumerate_segment_dirs` instead.
    format!("seg_{id:03}")
}

/// Discover every `seg_NNN/` subdirectory under `root`, sorted ascending by
/// the numeric id parsed from the name. Non-matching directory entries and
/// unparsable `seg_*` names are skipped silently.
///
/// Drives the CSR-load enumeration; at save time the next free id is
/// `last().map(|(id, _)| id + 1).unwrap_or(0)`.
pub(crate) fn enumerate_segment_dirs(root: &Path) -> Vec<(u32, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<(u32, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            if !e.file_type().ok()?.is_dir() {
                return None;
            }
            let name = e.file_name();
            let s = name.to_str()?;
            let id_str = s.strip_prefix("seg_")?;
            let id: u32 = id_str.parse().ok()?;
            Some((id, e.path()))
        })
        .collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

/// Current CSR-layout version emitted by every save. 0 = legacy flat,
/// 1 = segmented (seg_NNN/ subdirs). Loading tolerates both via the
/// version field in `DiskGraphMeta` (serde-defaulted to 0 for
/// pre-phase-4 graphs).
pub(crate) const CURRENT_CSR_LAYOUT_VERSION: u8 = 1;

/// Truly disk-backed graph. All data lives on disk via mmap.
///
/// - Nodes: `MmapOrVec<DiskNodeSlot>` (16 bytes/node, mmap'd)
///   Actual node data (id, title, properties) in ColumnStore columns (mmap'd).
///   `node_weight()` materializes NodeData into an arena on access.
/// - Edges: CSR arrays (`out_offsets`, `out_edges`, etc.) — mmap'd
/// - Edge properties: mmap'd columnar base + heap overlay for mutations
/// - Arenas: query-lifetime parking for materialized NodeData/EdgeData refs
pub struct DiskGraph {
    // ── Node storage (mmap'd on disk) ──
    pub(crate) node_slots: MmapOrVec<DiskNodeSlot>,
    /// Copy-on-write changes layered over an immutable published snapshot.
    pub(super) node_slot_updates: HashMap<u32, DiskNodeSlot>,
    pub(super) appended_node_slots: Vec<DiskNodeSlot>,
    pub(super) node_count: usize,
    pub(super) free_node_slots: Vec<u32>,

    // ── Node/edge materialization arenas + their reclamation epochs ──
    //
    // Disk reads have nothing in memory to borrow, so `node_weight` /
    // `materialize_edge` build a record and park it here. Thread-safety and
    // the epoch protocol governing when a record may be dropped are the
    // SAFETY block below and [`super::query_arena`] — read both before
    // touching either materialization path.
    pub(super) arenas: std::sync::Arc<super::query_arena::QueryArenas>,

    // ── Column stores for node properties (Arc refs, data mmap'd) ──
    /// `FxHashMap` for the same reason as the heap backends' field: probed
    /// once per `node_view` on every scan, over an already-hashed `u64` key.
    pub(crate) column_stores:
        rustc_hash::FxHashMap<InternedKey, Arc<crate::graph::storage::column_store::ColumnStore>>,

    // ── Edge CSR (mmap'd) ──
    pub(super) out_offsets: MmapOrVec<u64>,
    pub(super) out_edges: MmapOrVec<CsrEdge>,
    pub(super) in_offsets: MmapOrVec<u64>,
    pub(super) in_edges: MmapOrVec<CsrEdge>,

    pub(crate) edge_endpoints: MmapOrVec<EdgeEndpoints>,
    /// Endpoints appended after the immutable base snapshot and logical
    /// removals of base/overflow edges. CSR arrays themselves stay frozen.
    pub(super) appended_edge_endpoints: Vec<EdgeEndpoints>,
    pub(super) removed_edges: HashSet<u32>,
    pub(crate) edge_count: usize,
    pub(crate) next_edge_idx: u32,

    // ── Edge properties: mmap'd columnar base + heap overlay that grows with
    // mutation count, not graph size. See edge_properties.rs.
    pub(super) edge_properties: EdgePropertyStore,

    /// Cache for edge_weight_mut: stores materialized EdgeData that may be modified.
    /// Flushed to edge_properties on next clear_arenas call.
    pub(super) edge_mut_cache: HashMap<u32, EdgeData>,
    /// Cache for `node_weight_mut`: stages Cypher-SET-style exact-row writes
    /// as `PropertyStorage::Map` until `clear_arenas` drains it — see
    /// `flush_node_mut_cache` for why the flush must replace whole `Arc`s
    /// rather than mutate through them.
    pub(super) node_mut_cache: HashMap<u32, NodeData>,

    // File-backed (MmapOrVec) to avoid ~14 GB heap allocation at Wikidata scale.
    // Interior mutability: see item 2 of the SAFETY block below.
    pub(crate) pending_edges: UnsafeCell<MmapOrVec<PendingEdge>>,

    // ── Mutation overflow (for incremental edges after CSR) ──
    pub(super) overflow_out: HashMap<u32, Vec<CsrEdge>>,
    pub(super) overflow_in: HashMap<u32, Vec<CsrEdge>>,
    pub(super) free_edge_slots: Vec<u32>,

    pub(crate) data_dir: PathBuf,
    /// User-visible graph root and retained cross-process writer lease.
    pub(crate) logical_root: PathBuf,
    pub(crate) writer_lock: Option<Arc<super::generation::GraphDirectoryLock>>,
    pub(super) mutation_workspace: Option<Arc<super::generation::MutationWorkspace>>,
    pub(super) parent_workspaces: Vec<Arc<super::generation::MutationWorkspace>>,
    /// Cleanup owner for an explicit copy's lazily-created private writer
    /// root. Generic clones retain this lineage; only `independent_copy`
    /// replaces it with a fresh root.
    pub(super) independent_root: Option<Arc<super::generation::IndependentGraphRoot>>,
    // ── CSR edges are sorted by (node, connection_type) — enables binary search
    pub(crate) csr_sorted_by_type: bool,
    // ── Defer CSR build: edges accumulate in pending_edges without
    // intermediate CSR rebuilds, and the CSR is built at the next
    // `DirGraph::ensure_disk_edges_built` (save and write-statement flush
    // points). Set true during construction from add_nodes/add_connections,
    // cleared after CSR build.
    pub(crate) defer_csr: bool,
    // ── Edge type counts computed during CSR build (raw InternedKey u64 → count).
    // Converted to String keys by the caller using the interner.
    pub(crate) edge_type_counts_raw: Option<HashMap<u64, usize>>,
    // ── Connection-type inverted index: maps conn_type → list of source node IDs
    // that have at least one outgoing edge of that type. Built during CSR merge sort.
    // conn_type_index_offsets[i] = start position in conn_type_index_sources for type i.
    // conn_type_index_types: list of connection type u64s (ordered).
    pub(crate) conn_type_index_types: MmapOrVec<u64>,
    pub(crate) conn_type_index_offsets: MmapOrVec<u64>,
    pub(crate) conn_type_index_sources: MmapOrVec<u32>,
    // ── Per-(conn_type, peer) edge-count histogram.
    // Built alongside conn_type_index at CSR time; answers unanchored-aggregate
    // queries (`MATCH (a)-[:T]->(b) RETURN b, count(a) ...`) in O(distinct-peers)
    // instead of O(|edge_endpoints|). 3-array CSR layout mirrors
    // conn_type_index. `peer_count_entries` is flat (peer_u32, count_u32) pairs
    // sorted by peer within each type's slice — stored as u32 pairs to avoid
    // alignment fuss (length is always 2× the pair count).
    pub(crate) peer_count_types: MmapOrVec<u64>,
    pub(crate) peer_count_offsets: MmapOrVec<u64>, // in units of pairs, not u32s
    pub(crate) peer_count_entries: MmapOrVec<u32>, // [peer0, count0, peer1, count1, …]
    // ── Tombstone tracking: set true when any node/edge is removed. Lets
    // count_edges_filtered short-circuit the per-edge tombstone check when
    // no removals have happened (fresh builds, reloaded read-only graphs).
    pub(crate) has_tombstones: bool,
    // ── Persistent property indexes (lazy-loaded).
    //
    // Populated by `build_property_index(type, prop)` (the `create_index`
    // path — writes 4 files to `data_dir`), or on the first
    // `lookup_property_eq` miss, which scans `data_dir` for a
    // `property_index_{type}_{prop}_meta.bin` and mmaps it.
    //
    // The `None` sentinel records "we checked and no index exists" so
    // repeat misses don't stat the filesystem. `Arc` so concurrent reads
    // of the same index don't hold the outer RwLock.
    pub(crate) property_indexes: PropertyIndexCache,
    /// Exact typed indexes removed in the current writer workspace. Save
    /// excludes their base-generation bundles instead of mutating the
    /// immutable snapshot in place.
    pub(super) removed_property_indexes: HashSet<(String, String)>,
    // ── Persistent cross-type global property indexes (lazy-loaded).
    //
    // Keyed by property name only. Built by `build_global_property_index(prop)`
    // — scans every alive `DiskNodeSlot` and collects one
    // `(string_value, NodeIndex)` entry per node where `prop` resolves
    // (via column slot, title alias, or id alias). Powers untyped
    // patterns like `MATCH (n {label: 'X'})`.
    pub(crate) global_indexes: GlobalIndexCache,
    // ── Segment manifest.
    //
    // Persisted at `seg_manifest.json` alongside the CSR files. Legacy
    // graphs that lack the file load as an empty manifest — the planner
    // treats that as "pre-segmented, don't prune". Fresh saves always write
    // a one-segment manifest describing the whole graph; multi-segment
    // writes and reads update this summary as generations seal.
    pub(crate) segment_manifest: super::segment_summary::SegmentManifest,
    // ── Sealed-nodes watermark.
    //
    // Node ids in `[0, sealed_nodes_bound)` are accounted for in a prior
    // sealed segment's `node_slots`; ids in `[sealed_nodes_bound,
    // node_count)` are in the active (still-mutable) tail, not yet sealed
    // into any segment. `seal_to_new_segment` flushes the tail into a new
    // `seg_NNN/` and advances this watermark.
    //
    // Zero on freshly-built / pre-phase-8 graphs; the `DiskGraphMeta`
    // serde `default` keeps old `.kgl` directories loadable.
    pub(crate) sealed_nodes_bound: u32,
}

/// Lazy-loaded cache of persistent property indexes, keyed by
/// `(node_type, property)`. `None` records "checked and absent".
type PropertyIndexCache =
    std::sync::RwLock<HashMap<(String, String), Option<Arc<property_index::PropertyIndex>>>>;

/// Lazy-loaded cache of persistent cross-type global indexes, keyed
/// by property name. `None` records "checked and absent".
type GlobalIndexCache =
    std::sync::RwLock<HashMap<String, Option<Arc<property_index::PropertyIndex>>>>;

use std::sync::Arc;

// SAFETY — DiskGraph interior-mutability model:
//
// 1. `arenas: Arc<QueryArenas>` — the node/edge materialization arenas,
//    thread-safe for Rayon parallel queries. Each record is boxed (stable heap
//    pointer that survives the arena's own growth) and pushed under a Mutex,
//    because the Cypher executor's projection phase runs
//    `evaluate_expression` under `par_iter_mut` (return_clause.rs) and any
//    spatial / non-fast-path `resolve_property` branch reaches `node_weight`
//    through that parallel context. Pre-0.9.3 this was
//    `UnsafeCell<Vec<NodeData>>` and races were silent: a sibling Rayon task's
//    `arena.push` realloc invalidated references already returned to other
//    tasks, surfacing as either wrong-row reads on disk-mode aggregations
//    (Bug A in the 0.9.2 disk regression — ~13% NEAREST_AFEX_HUB edges
//    silently lost) or use-after-free segfaults with `BUG: InternedKey N not
//    found in StringInterner` on stderr (Bug B in the same report).
//
//    The *lifetime* side lives in `super::query_arena`: every materializing
//    read runs under a `DiskQueryGuard`, records are stamped with an epoch
//    above every live query's id, and a record is dropped only once every
//    query that could hold it has finished. `reset_arenas` reclaims only while
//    nothing is reading; `clear_arenas` takes `&mut self`, which the borrow
//    checker already orders after any outstanding materialization borrow.
//
// 2. `pending_edges: UnsafeCell<MmapOrVec<…>>` — every *mutation* goes
//    through `get_mut()` in a `&mut self` context (`try_add_pending_edge`,
//    `build_csr_from_pending`, `compact`, the bootstrap/builder paths), so
//    the borrow checker already excludes a concurrent writer. The one
//    `&self` reader is `Clone`, which dereferences the cell to copy the
//    buffer; its own SAFETY note carries that argument.
unsafe impl Send for DiskGraph {}
unsafe impl Sync for DiskGraph {}

pub use super::query_arena::DiskQueryGuard;

/// Debug-only check that a materializing read is running under an
/// active [`DiskQueryGuard`]. The arena SAFETY argument (see the block
/// comment above and `super::query_arena`) rests on a live guard keeping the
/// record reachable — a materialization with no guard at all is a protocol
/// violation that could become a use-after-free the moment any query starts or
/// ends. Compiles to nothing in release builds.
#[inline(always)]
fn debug_assert_arena_guard_active(graph: &DiskGraph, who: &str) {
    #[cfg(debug_assertions)]
    {
        debug_assert!(
            graph.arenas.active_count() > 0,
            "DiskGraph::{who} materialized into the arena without an active \
             DiskQueryGuard (no query is open); wrap the read in begin_query() \
             — see the arena SAFETY protocol comment in disk/graph.rs"
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (graph, who);
    }
}

impl std::fmt::Debug for DiskGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DiskGraph({} nodes, {} edges, dir={:?})",
            self.node_count,
            self.edge_count,
            self.data_dir.display()
        )
    }
}

include!("graph/bootstrap.rs");

impl DiskGraph {
    /// Backs `GraphRead::column_stores_iter` for the disk backend.
    pub fn column_stores_iter(
        &self,
    ) -> impl Iterator<
        Item = (
            &InternedKey,
            &Arc<crate::graph::storage::column_store::ColumnStore>,
        ),
    > {
        self.column_stores.iter()
    }

    /// `true` when the out-edge CSR array is memory-mapped from a file
    /// rather than heap-resident. A freshly constructed disk graph whose CSR
    /// has not been built yet (edges still in `pending_edges`/overflow)
    /// reports `false`.
    pub fn csr_is_mapped(&self) -> bool {
        self.out_edges.is_mapped()
    }

    /// Rows in the edge-property heap overlay — see
    /// [`EdgePropertyStore::overlay_len`].
    pub fn edge_property_overlay_len(&self) -> usize {
        self.edge_properties.overlay_len()
    }

    /// O(1) node type lookup from mmap'd node_slots — no materialization.
    /// Returns None if the node is dead or out of bounds.
    #[inline]
    pub fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        let i = idx.index();
        if i >= self.node_slot_len() {
            return None;
        }
        let slot = self.node_slot(i);
        if !slot.is_alive() {
            return None;
        }
        Some(InternedKey::from_u64(slot.node_type))
    }

    /// O(1) property read from ColumnStore — no NodeData materialization.
    /// Returns None if the node is dead, out of bounds, or the property doesn't exist.
    #[inline]
    pub fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        let i = idx.index();
        if i >= self.node_slot_len() {
            return None;
        }
        let slot = self.node_slot(i);
        if !slot.is_alive() {
            return None;
        }
        let type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&type_key)?;
        // id and title are stored as __id__/__title__ (separate from schema),
        // so store.get() won't find them by their alias names ("title", "id").
        if let Some(val) = store.get(slot.row_id, key) {
            return Some(val);
        }
        if key == InternedKey::from_str("title") {
            return store.get_title(slot.row_id);
        }
        if key == InternedKey::from_str("id") {
            return store.get_id(slot.row_id);
        }
        None
    }

    /// O(1) id value read from ColumnStore — no NodeData materialization.
    #[inline]
    pub fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        let i = idx.index();
        if i >= self.node_slot_len() {
            return None;
        }
        let slot = self.node_slot(i);
        if !slot.is_alive() {
            return None;
        }
        let type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&type_key)?;
        store.get_id(slot.row_id)
    }

    /// O(1) title value read from ColumnStore — no NodeData materialization.
    #[inline]
    pub fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        let i = idx.index();
        if i >= self.node_slot_len() {
            return None;
        }
        let slot = self.node_slot(i);
        if !slot.is_alive() {
            return None;
        }
        let type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&type_key)?;
        store.get_title(slot.row_id)
    }

    #[inline]
    pub fn node_slot(&self, i: usize) -> DiskNodeSlot {
        if i < self.node_slots.len() {
            self.node_slot_updates
                .get(&(i as u32))
                .copied()
                .unwrap_or_else(|| self.node_slots.get(i))
        } else if i < self.node_slot_len() {
            self.appended_node_slots[i - self.node_slots.len()]
        } else {
            DiskNodeSlot::default()
        }
    }

    /// Materialize a NodeData from disk slot + ColumnStore into the arena.
    ///
    /// Every call parks a record that survives until the calling query ends.
    /// Callers that finish with the record inside their own stack frame — the
    /// scans and filters, which are the bulk of the traffic — should use
    /// [`Self::owned_node_data`] instead and leave the arena alone.
    #[inline]
    pub fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        let node_data = self.materialize_node_data(idx)?;
        // The record is parked in the query arena; the epoch protocol in
        // `super::query_arena` keeps its heap pointer alive until every query
        // that could reach it has finished. The `&self` borrow alone would NOT
        // be enough — `begin_query`/`reset_arenas` reclaim through `&self`.
        debug_assert_arena_guard_active(self, "node_weight");
        let ptr = self.arenas.push_node(node_data);
        // SAFETY: the guard asserted above plus that protocol keep the pointer
        // alive for at least the returned reference's lifetime.
        unsafe { Some(&*ptr) }
    }

    /// Materialize a node into the *caller's* frame.
    ///
    /// The arena-free counterpart of [`Self::node_weight`]: the record dies
    /// where the caller drops it, so a scan over a million nodes holds one at
    /// a time instead of a million, and never touches the arena mutex.
    /// Nothing is parked, so no [`DiskQueryGuard`] is required for the
    /// record's sake (callers still hold one for whatever else the read does).
    #[inline]
    pub(crate) fn owned_node_data(&self, idx: NodeIndex) -> Option<NodeData> {
        self.materialize_node_data(idx)
    }

    /// Build a `NodeData` for `idx` from the node slot + its type's
    /// ColumnStore. Pure: the caller decides where the record lives.
    #[inline]
    fn materialize_node_data(&self, idx: NodeIndex) -> Option<NodeData> {
        let i = idx.index();
        if i >= self.node_slot_len() {
            return None;
        }
        let slot = self.node_slot(i);
        if !slot.is_alive() {
            return None;
        }

        // Preventative invariant (0.9.0 Cluster 6): `node_weight_mut` stages
        // writes in `node_mut_cache`, so a *Map-typed* entry for this index on
        // the read path means a staged write is about to be silently shadowed
        // by the column_stores read — a missed flush_pending_writes call, i.e.
        // a new code path that needs an explicit flush.
        //
        // Only Map entries count (0.9.26). `batch.rs::flush_chunk` leaves
        // `PropertyStorage::Columnar { row_id, .. }` scratch in the cache after
        // persisting via its own full-Arc replacement; firing on those was a
        // false positive that made the warning noisy in normal test runs.
        #[cfg(debug_assertions)]
        if let Some(staged) = self.node_mut_cache.get(&(i as u32)) {
            use crate::graph::schema::PropertyStorage;
            if matches!(staged.properties, PropertyStorage::Map(_))
                && !matches!(staged.properties, PropertyStorage::Map(ref m) if m.is_empty())
            {
                eprintln!(
                    "BUG: DiskGraph::node_weight({}) called while node_mut_cache holds a \
                     staged Map-typed write for that index. Missing flush_pending_writes() \
                     call. See 0.9.0 readiness Cluster 6 / node_weight_mut docs.",
                    i
                );
            }
        }

        let node_type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&node_type_key);

        Some(if let Some(store) = store {
            let id = store.get_id(slot.row_id).unwrap_or(Value::Null);
            let title = store.get_title(slot.row_id).unwrap_or(Value::Null);
            NodeData {
                id,
                title,
                node_type: node_type_key,
                properties: crate::graph::schema::PropertyStorage::Columnar(
                    crate::graph::storage::property_storage::ColumnarRow::new(slot.row_id),
                ),
            }
        } else {
            NodeData {
                id: Value::Null,
                title: Value::Null,
                node_type: node_type_key,
                properties: crate::graph::schema::PropertyStorage::Map(HashMap::new()),
            }
        })
    }

    pub fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        let i = idx.index();
        if i >= self.node_slot_len() {
            return None;
        }
        let slot = self.node_slot(i);
        if !slot.is_alive() {
            return None;
        }

        let node_type_key = InternedKey::from_u64(slot.node_type);
        let key = i as u32;

        // Stage exact-row mutations as `PropertyStorage::Map`, not `Columnar`:
        // a `Columnar` variant routes `node.set_property(k, v)` through
        // `Arc::make_mut(store)`, which clones the store when DirGraph still
        // holds another Arc and lands the mutation on a detached copy.
        //
        // Reseed path: `batch.rs::flush_chunk` (and similar bulk paths)
        // transiently assigns `PropertyStorage::Columnar{...}`; a stale one
        // still in the cache is replaced with Map, since batch already
        // persisted it via full-Arc replacement.
        let needs_reseed = match self.node_mut_cache.get(&key) {
            None => true,
            Some(nd) => !matches!(nd.properties, crate::graph::schema::PropertyStorage::Map(_)),
        };
        if needs_reseed {
            let store = self.column_stores.get(&node_type_key);
            let (id_val, title_val) = if let Some(s) = store {
                (
                    s.get_id(slot.row_id).unwrap_or(Value::Null),
                    s.get_title(slot.row_id).unwrap_or(Value::Null),
                )
            } else {
                (Value::Null, Value::Null)
            };
            self.node_mut_cache.insert(
                key,
                NodeData {
                    id: id_val,
                    title: title_val,
                    node_type: node_type_key,
                    properties: crate::graph::schema::PropertyStorage::Map(HashMap::new()),
                },
            );
        }
        Some(self.node_mut_cache.get_mut(&key).unwrap())
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    #[inline]
    pub(crate) fn node_slot_len(&self) -> usize {
        self.node_slots.len() + self.appended_node_slots.len()
    }

    #[inline]
    pub(crate) fn set_node_slot(&mut self, index: usize, slot: DiskNodeSlot) {
        if index < self.node_slots.len() {
            self.node_slot_updates.insert(index as u32, slot);
        } else {
            self.appended_node_slots[index - self.node_slots.len()] = slot;
        }
    }

    #[inline]
    pub(crate) fn edge_endpoint_len(&self) -> usize {
        self.edge_endpoints.len() + self.appended_edge_endpoints.len()
    }

    #[inline]
    pub(crate) fn edge_endpoint(&self, index: usize) -> EdgeEndpoints {
        if self.removed_edges.contains(&(index as u32)) {
            return EdgeEndpoints {
                source: TOMBSTONE_EDGE,
                target: TOMBSTONE_EDGE,
                connection_type: 0,
            };
        }
        if index < self.edge_endpoints.len() {
            self.edge_endpoints.get(index)
        } else {
            self.appended_edge_endpoints[index - self.edge_endpoints.len()]
        }
    }

    #[inline]
    pub(crate) fn edge_is_alive(&self, index: u32) -> bool {
        (index as usize) < self.edge_endpoint_len()
            && self.edge_endpoint(index as usize).source != TOMBSTONE_EDGE
    }

    #[inline]
    pub fn node_bound(&self) -> usize {
        self.node_slot_len()
    }

    pub fn add_node(&mut self, data: NodeData) -> NodeIndex {
        self.clear_arenas();

        let row_id = match &data.properties {
            crate::graph::schema::PropertyStorage::Columnar(row) => row.row_id(),
            _ => self.node_slot_len() as u32,
        };

        let slot = DiskNodeSlot {
            node_type: data.node_type.as_u64(),
            row_id,
            flags: DiskNodeSlot::ALIVE_BIT,
        };

        if let Some(recycled) = self.free_node_slots.pop() {
            let idx = recycled as usize;
            self.set_node_slot(idx, slot);
            self.node_count += 1;
            NodeIndex::new(idx)
        } else {
            let idx = self.node_slot_len();
            self.appended_node_slots.push(slot);
            self.node_count += 1;
            NodeIndex::new(idx)
        }
    }

    pub fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        self.clear_arenas();
        let i = idx.index();
        if i >= self.node_slot_len() {
            return None;
        }
        let slot = self.node_slot(i);
        if !slot.is_alive() {
            return None;
        }

        let node_type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&node_type_key).cloned();
        let (id_val, title_val) = if let Some(ref s) = store {
            (
                s.get_id(slot.row_id).unwrap_or(Value::Null),
                s.get_title(slot.row_id).unwrap_or(Value::Null),
            )
        } else {
            (Value::Null, Value::Null)
        };
        let data = NodeData {
            id: id_val,
            title: title_val,
            node_type: node_type_key,
            properties: if store.is_some() {
                crate::graph::schema::PropertyStorage::Columnar(
                    crate::graph::storage::property_storage::ColumnarRow::new(slot.row_id),
                )
            } else {
                crate::graph::schema::PropertyStorage::Map(HashMap::new())
            },
        };

        let mut dead_slot = slot;
        dead_slot.flags = 0;
        self.set_node_slot(i, dead_slot);
        self.node_count -= 1;
        self.free_node_slots.push(i as u32);
        self.has_tombstones = true;

        self.tombstone_edges_for_node(i);

        Some(data)
    }

    /// Repoint a node slot at its per-type ColumnStore row. Bulk paths assign
    /// the row while building the store and call this to persist the mapping.
    pub fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        let i = node_idx.index();
        if i < self.node_slot_len() {
            let mut slot = self.node_slot(i);
            slot.row_id = row_id;
            self.set_node_slot(i, slot);
        }
    }

    pub fn node_indices_iter(&self) -> DiskNodeIndices<'_> {
        DiskNodeIndices::new(self)
    }

    /// Materialize an EdgeData into the arena. Reads conn_type from
    /// EdgeEndpoints (O(1) lookup) and properties from `edge_properties`.
    #[inline]
    pub(crate) fn materialize_edge(&self, edge_idx: u32) -> &EdgeData {
        let ep = self.edge_endpoint(edge_idx as usize);
        let ct = InternedKey::from_u64(ep.connection_type);
        let props = if self.edge_properties.is_empty() {
            Vec::new()
        } else {
            self.edge_properties
                .get(edge_idx)
                .map(|cow| cow.into_owned())
                .unwrap_or_default()
        };
        debug_assert_arena_guard_active(self, "materialize_edge");
        // Same lifetime protocol as `node_weight`: the arena's stable heap
        // pointer is protected by the epoch bookkeeping in
        // `super::query_arena`, not by the `&self` borrow alone.
        let ptr = self.arenas.push_edge(EdgeData {
            connection_type: ct,
            properties: props,
        });
        // SAFETY: the guard asserted above keeps the record alive for the
        // returned reference's lifetime.
        unsafe { &*ptr }
    }

    /// Count edges of a specific type without materializing EdgeData.
    /// With sorted CSR, uses binary search to find the exact range, then counts
    /// peers matching the optional node type filter. Zero allocations.
    pub fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
    ) -> Result<usize, String> {
        self.ensure_csr();
        let idx = node.index();
        let (offsets, edges) = match dir {
            Direction::Outgoing => (&self.out_offsets, &self.out_edges),
            Direction::Incoming => (&self.in_offsets, &self.in_edges),
        };
        // Empty CSR range when offsets don't cover `idx + 1` (overflow-only node);
        // fall through to the overflow count below rather than returning early.
        let (mut start, mut end) = if idx + 1 < offsets.len() {
            (offsets.get(idx) as usize, offsets.get(idx + 1) as usize)
        } else {
            (0, 0)
        };

        if let Some(ct) = conn_type {
            if self.csr_sorted_by_type {
                let (lo, hi) = crate::graph::core::iterators::binary_search_conn_type(
                    edges, self, start, end, ct,
                );
                start = lo;
                end = hi;
            }
        }

        // Fast path: no tombstones and no peer-type filter → the answer is
        // literally the range length + overflow size, no scan required. This
        // turns Q5-class "count all P31 incoming" queries from 40 M loop
        // iterations (20+ s on USB SSD) into O(log D) binary search + two
        // integer subtractions.
        let can_shortcut = !self.has_tombstones
            && other_node_type.is_none()
            && (conn_type.is_none() || self.csr_sorted_by_type);
        if can_shortcut {
            let overflow = match dir {
                Direction::Outgoing => self.overflow_out.get(&(idx as u32)),
                Direction::Incoming => self.overflow_in.get(&(idx as u32)),
            };
            let mut overflow_count = 0usize;
            if let Some(list) = overflow {
                for e in list {
                    if let Some(ct) = conn_type {
                        if self.edge_endpoint(e.edge_idx as usize).connection_type != ct {
                            continue;
                        }
                    }
                    overflow_count += 1;
                }
            }
            return Ok(end.saturating_sub(start) + overflow_count);
        }

        let mut count = 0usize;
        for i in start..end {
            // Deadline check every 1 M edges — enough for Q5-scale hub fan-in
            // (~40 M P31 incoming) to terminate at ~20 s rather than 100 s.
            if (i - start).is_multiple_of(1 << 20) {
                if let Some(dl) = deadline {
                    if std::time::Instant::now() > dl {
                        return Err("Query timed out".to_string());
                    }
                }
            }
            let e = edges.get(i);
            if e.edge_idx == TOMBSTONE_EDGE {
                continue;
            }
            if let Some(ct) = conn_type {
                if !self.csr_sorted_by_type
                    && self.edge_endpoint(e.edge_idx as usize).connection_type != ct
                {
                    continue;
                }
            }
            if let Some(required_type) = other_node_type {
                let peer_idx = NodeIndex::new(e.peer as usize);
                if let Some(nt) = self.node_type_of(peer_idx) {
                    if nt != required_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            count += 1;
        }

        let overflow = match dir {
            Direction::Outgoing => self.overflow_out.get(&(idx as u32)),
            Direction::Incoming => self.overflow_in.get(&(idx as u32)),
        };
        if let Some(list) = overflow {
            for e in list {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                if let Some(ct) = conn_type {
                    if self.edge_endpoint(e.edge_idx as usize).connection_type != ct {
                        continue;
                    }
                }
                if let Some(required_type) = other_node_type {
                    let peer_idx = NodeIndex::new(e.peer as usize);
                    if let Some(nt) = self.node_type_of(peer_idx) {
                        if nt != required_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                count += 1;
            }
        }
        Ok(count)
    }

    /// Peers reachable over edges of a specific type, without materializing
    /// EdgeData. When the CSR is sorted by type the matching slice is found by
    /// binary search and nothing reads edge_endpoints.bin (13 GB at Wikidata
    /// scale); with a type filter the unsorted-CSR and overflow paths still
    /// probe it per edge to recover the connection type.
    pub fn iter_peers_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Vec<(NodeIndex, u32)> {
        self.ensure_csr();
        let idx = node.index();
        let (offsets, edges) = match dir {
            Direction::Outgoing => (&self.out_offsets, &self.out_edges),
            Direction::Incoming => (&self.in_offsets, &self.in_edges),
        };
        // If the CSR offset table doesn't cover `idx + 1` (fresh in-memory disk
        // graph where edges live in overflow, or a node appended after the last
        // CSR build), use an empty CSR range and fall through to the overflow
        // scan below — mirroring `edges_directed_filtered_iter`. Returning early
        // here drops the node's overflow-resident edges entirely.
        let (mut start, mut end) = if idx + 1 < offsets.len() {
            (offsets.get(idx) as usize, offsets.get(idx + 1) as usize)
        } else {
            (0, 0)
        };

        if let Some(ct) = conn_type {
            if self.csr_sorted_by_type {
                let (lo, hi) = crate::graph::core::iterators::binary_search_conn_type(
                    edges, self, start, end, ct,
                );
                start = lo;
                end = hi;
            }
        }

        let mut result = Vec::with_capacity(end - start);
        for i in start..end {
            let e = edges.get(i);
            if e.edge_idx == TOMBSTONE_EDGE {
                continue;
            }
            if let Some(ct) = conn_type {
                if !self.csr_sorted_by_type
                    && self.edge_endpoint(e.edge_idx as usize).connection_type != ct
                {
                    continue;
                }
            }
            result.push((NodeIndex::new(e.peer as usize), e.edge_idx));
        }

        let overflow = match dir {
            Direction::Outgoing => self.overflow_out.get(&(idx as u32)),
            Direction::Incoming => self.overflow_in.get(&(idx as u32)),
        };
        if let Some(list) = overflow {
            for e in list {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                if let Some(ct) = conn_type {
                    if self.edge_endpoint(e.edge_idx as usize).connection_type != ct {
                        continue;
                    }
                }
                result.push((NodeIndex::new(e.peer as usize), e.edge_idx));
            }
        }
        result
    }

    /// Count all edges of a connection type, grouped by peer (target for
    /// outgoing, source for incoming). Single sequential scan of
    /// edge_endpoints — O(E) total, no random access.
    pub fn count_edges_grouped_by_peer(
        &self,
        conn_type: u64,
        dir: Direction,
        deadline: Option<std::time::Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        self.ensure_csr();
        let mut counts: HashMap<u32, i64> = HashMap::new();

        // Advise kernel: sequential read of edge_endpoints (13 GB).
        // MADV_SEQUENTIAL enables aggressive readahead and avoids polluting
        // the page cache with pages we won't revisit.
        self.edge_endpoints.advise_sequential();

        // Deadline check every 1M entries keeps the per-check cost <0.001%
        // while bounding wall-clock overshoot to ~0.3s.
        let total = self.next_edge_idx as usize;
        for i in 0..total {
            if i.is_multiple_of(1 << 20) {
                if let Some(dl) = deadline {
                    if std::time::Instant::now() > dl {
                        self.edge_endpoints.advise_dontneed();
                        return Err("Query timed out".to_string());
                    }
                }
            }
            let ep = self.edge_endpoint(i);
            if ep.source == TOMBSTONE_EDGE {
                continue;
            }
            if ep.connection_type != conn_type {
                continue;
            }
            let peer = match dir {
                Direction::Outgoing => ep.target,
                Direction::Incoming => ep.source,
            };
            *counts.entry(peer).or_insert(0) += 1;
        }

        self.edge_endpoints.advise_dontneed();

        Ok(counts)
    }

    /// Source nodes that have outgoing edges of the given connection type.
    /// Returns None if the inverted index is not built (older graph format).
    #[allow(dead_code)] // Test-only.
    pub fn sources_for_conn_type(&self, conn_type: u64) -> Option<Vec<u32>> {
        self.sources_for_conn_type_bounded(conn_type, None)
    }

    /// Bounded variant of `sources_for_conn_type`. When `max.is_some()` the
    /// function stops copying after the requested number of source node IDs
    /// — avoids eagerly materialising ~100 M u32s (400 MB) into a heap Vec
    /// when the caller will immediately truncate it to a few thousand.
    ///
    /// Overflow sources are always fully collected (tiny by definition).
    pub fn sources_for_conn_type_bounded(
        &self,
        conn_type: u64,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        if self.conn_type_index_types.is_empty() && self.overflow_out.is_empty() {
            return None;
        }

        let mut sources = Vec::new();
        if !self.conn_type_index_types.is_empty() {
            let num_types = self.conn_type_index_types.len();
            let mut lo = 0usize;
            let mut hi = num_types;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let mid_type = self.conn_type_index_types.get(mid);
                if mid_type < conn_type {
                    lo = mid + 1;
                } else if mid_type > conn_type {
                    hi = mid;
                } else {
                    let start = self.conn_type_index_offsets.get(mid) as usize;
                    let end = self.conn_type_index_offsets.get(mid + 1) as usize;
                    let take_end = match max {
                        Some(m) => start + (end - start).min(m),
                        None => end,
                    };
                    sources.reserve(take_end - start);
                    for i in start..take_end {
                        sources.push(self.conn_type_index_sources.get(i));
                    }
                    break;
                }
            }
        }

        // Supplement with overflow sources. `max` is not applied here: overflow
        // is almost always small, and bounding it would need a second dedup
        // pass to keep the contract honest.
        if !self.overflow_out.is_empty() {
            for (&node_id, edges) in &self.overflow_out {
                for e in edges {
                    if e.edge_idx != TOMBSTONE_EDGE {
                        let ep = self.edge_endpoint(e.edge_idx as usize);
                        if ep.connection_type == conn_type {
                            sources.push(node_id);
                            break; // One matching edge is enough
                        }
                    }
                }
            }
            // Deduplicate (a node may appear in both CSR index and overflow)
            sources.sort_unstable();
            sources.dedup();
        }

        Some(sources)
    }

    /// Iterate only the edges matching `conn_type`, yielding `(src, tgt, edge_idx)`
    /// per match. Never calls `materialize_edge` — nothing is parked in the
    /// query arenas.
    ///
    /// Path: the persisted inverted index (`conn_type_index_*`) gives the sources
    /// with at least one outgoing edge of that type; each source's outgoing CSR
    /// slice is then filtered by `conn_type` (binary-search when the CSR is
    /// sorted by type, linear fallback otherwise). Overflow-out entries cover
    /// sources added after the last CSR build.
    ///
    /// The callback returns `true` to continue, `false` to stop iteration —
    /// lets callers collect a bounded prefix (e.g. two sample edges) without
    /// scanning every match.
    ///
    /// O(matching edges) when `csr_sorted_by_type`, not O(all edges); backs the
    /// introspection fast path (`describe(connections=['T'])`).
    pub fn for_each_edge_of_conn_type<F>(&self, conn_type: u64, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, u32) -> bool,
    {
        self.ensure_csr();

        if !self.conn_type_index_types.is_empty() {
            let num_types = self.conn_type_index_types.len();
            let mut lo = 0usize;
            let mut hi = num_types;
            let mut range: Option<(usize, usize)> = None;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let mid_type = self.conn_type_index_types.get(mid);
                if mid_type < conn_type {
                    lo = mid + 1;
                } else if mid_type > conn_type {
                    hi = mid;
                } else {
                    let s = self.conn_type_index_offsets.get(mid) as usize;
                    let e = self.conn_type_index_offsets.get(mid + 1) as usize;
                    range = Some((s, e));
                    break;
                }
            }

            if let Some((src_start, src_end)) = range {
                let out_offsets_len = self.out_offsets.len().saturating_sub(1);
                for i in src_start..src_end {
                    let src_u32 = self.conn_type_index_sources.get(i);
                    let src_idx = src_u32 as usize;
                    if src_idx >= out_offsets_len {
                        continue;
                    }
                    let csr_start = self.out_offsets.get(src_idx) as usize;
                    let csr_end = self.out_offsets.get(src_idx + 1) as usize;

                    if self.csr_sorted_by_type {
                        let (lo_p, hi_p) = crate::graph::core::iterators::binary_search_conn_type(
                            &self.out_edges,
                            self,
                            csr_start,
                            csr_end,
                            conn_type,
                        );
                        for p in lo_p..hi_p {
                            let e = self.out_edges.get(p);
                            if e.edge_idx == TOMBSTONE_EDGE {
                                continue;
                            }
                            if !f(
                                NodeIndex::new(src_u32 as usize),
                                NodeIndex::new(e.peer as usize),
                                e.edge_idx,
                            ) {
                                return;
                            }
                        }
                    } else {
                        for p in csr_start..csr_end {
                            let e = self.out_edges.get(p);
                            if e.edge_idx == TOMBSTONE_EDGE {
                                continue;
                            }
                            let ep = self.edge_endpoint(e.edge_idx as usize);
                            if ep.connection_type == conn_type
                                && !f(
                                    NodeIndex::new(src_u32 as usize),
                                    NodeIndex::new(e.peer as usize),
                                    e.edge_idx,
                                )
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }

        // Overflow sources — edges appended after the last CSR build. Typically tiny.
        for (&src_u32, edges) in &self.overflow_out {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ep = self.edge_endpoint(e.edge_idx as usize);
                if ep.connection_type == conn_type
                    && !f(
                        NodeIndex::new(src_u32 as usize),
                        NodeIndex::new(e.peer as usize),
                        e.edge_idx,
                    )
                {
                    return;
                }
            }
        }
    }

    /// Borrow an edge's property slice without materializing `EdgeData`.
    /// Returns `None` when the edge has no custom properties (common case).
    /// Safe to call in hot loops — nothing is parked in the query arenas.
    ///
    /// The returned `Cow` is `Borrowed` for overlay hits (zero copy) and
    /// `Owned` for columnar-base hits (one binary payload decode). Callers
    /// that need `&[(InternedKey, Value)]` can use `.as_deref()`.
    #[inline]
    pub fn edge_properties_at(&self, edge_idx: u32) -> Option<Cow<'_, [(InternedKey, Value)]>> {
        self.edge_properties.get(edge_idx)
    }

    /// Edge-centric sweep: scan `edge_endpoints` linearly, invoking `f` for
    /// every match. Return `false` from the callback to stop early.
    ///
    /// Contrast with [`Self::for_each_edge_of_conn_type`], which walks
    /// the source-centric `conn_type_index` and binary-searches each
    /// source's CSR slice. The binary search reads
    /// `edge_endpoints[edge_idx].connection_type` per comparison; at
    /// Wikidata-1B scale (247 MB endpoints, ~11 M matching sources ×
    /// log D comparisons) those random reads miss the system-level
    /// cache on every iteration, blowing the aggregation out to
    /// ~4.5 s. The linear form touches the same array in address order,
    /// so the kernel-prefetched sequential read completes in under the
    /// 247 MB / ~50 GB/s memory-bandwidth bound.
    ///
    /// Trade-off: O(|all edges|) regardless of how selective `conn_type`
    /// is. Prefer the source-centric form when the matching source set
    /// is small relative to total edges; prefer this when the matches
    /// cover a meaningful fraction (≥ a few percent) and/or the graph
    /// is too large to keep `edge_endpoints` in cache.
    pub fn scan_edges_of_conn_type_linear<F>(&self, conn_type: u64, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, u32) -> bool,
    {
        let n = self.next_edge_idx as usize;
        for edge_idx in 0..n {
            let ep = self.edge_endpoint(edge_idx);
            if ep.source == TOMBSTONE_EDGE {
                continue;
            }
            if ep.connection_type != conn_type {
                continue;
            }
            if !f(
                NodeIndex::new(ep.source as usize),
                NodeIndex::new(ep.target as usize),
                edge_idx as u32,
            ) {
                return;
            }
        }
    }

    /// Warm hot mmap regions into page cache after load. Non-blocking — the
    /// kernel reads the pages asynchronously.
    pub fn prefetch_hot_regions(&self) {
        // Offsets only (948 MB each at Wikidata scale, needed by every
        // traversal). node_slots (2 GB) costs more load latency than it saves
        // and is left to page in on demand.
        self.out_offsets.advise_willneed();
        self.in_offsets.advise_willneed();
    }

    #[inline]
    fn ensure_csr(&self) {
        // No-op check — pending edges should be empty after build_csr_from_pending.
        // If not empty, queries may miss some edges (but won't crash).
    }

    /// Clear all materialization arenas. Called before any &mut self operation.
    #[inline]
    pub(crate) fn clear_arenas(&mut self) {
        for (edge_idx, edge_data) in self.edge_mut_cache.drain() {
            if edge_data.properties.is_empty() {
                self.edge_properties.remove(edge_idx);
            } else {
                self.edge_properties.insert(edge_idx, edge_data.properties);
            }
        }
        self.flush_node_mut_cache();
        self.arenas.clear_all();
    }

    /// Drain `node_mut_cache` and apply the staged writes to
    /// `self.column_stores`, cloning each affected store once and replacing
    /// the whole `Arc`. Dead slots (tombstoned via `remove_node`) flush a
    /// `store.tombstone(row_id)` instead. Node analogue of
    /// `batch.rs::flush_chunk`'s deferred-columnar pass — the only pattern
    /// proven to survive the Arc sharing between DirGraph and DiskGraph.
    fn flush_node_mut_cache(&mut self) {
        if self.node_mut_cache.is_empty() {
            return;
        }
        use crate::graph::schema::PropertyStorage;
        let drained: Vec<(u32, NodeData)> = self.node_mut_cache.drain().collect();
        let mut by_type: HashMap<InternedKey, Vec<(u32, NodeData)>> = HashMap::new();
        for (i, nd) in drained {
            if (i as usize) >= self.node_slot_len() {
                continue;
            }
            let slot = self.node_slot(i as usize);
            let type_key = InternedKey::from_u64(slot.node_type);
            by_type.entry(type_key).or_default().push((i, nd));
        }
        for (type_key, updates) in by_type {
            let Some(current_arc) = self.column_stores.get(&type_key) else {
                continue;
            };
            // Skip the clone + Arc-replace unless something would actually be
            // written. `batch.rs::flush_chunk` leaves `PropertyStorage::
            // Columnar` scratch in the cache (batch persists via its own
            // full-Arc replacement); those yield nothing to flush.
            let any_writes_needed = updates.iter().any(|(i, nd)| {
                let slot = self.node_slot(*i as usize);
                if !slot.is_alive() {
                    return true;
                }
                if let PropertyStorage::Map(map) = &nd.properties {
                    if !map.is_empty() {
                        return true;
                    }
                }
                // Only a title differing from the store counts — see the
                // `Str::set` offset-corruption note at the write below.
                if !matches!(nd.title, Value::Null) {
                    let current = current_arc.get_title(slot.row_id);
                    return match (current, &nd.title) {
                        (Some(a), b) => a != *b,
                        (None, _) => true,
                    };
                }
                false
            });
            if !any_writes_needed {
                continue;
            }
            // One explicit deep clone — the clone has refcount 1 so
            // `ColumnStore::set` / `set_title` / `tombstone` operate
            // in place with no further Arc work.
            let mut new_store: crate::graph::storage::column_store::ColumnStore =
                (**current_arc).clone();
            for (i, nd) in updates {
                let slot = self.node_slot(i as usize);
                let row_id = slot.row_id;
                if !slot.is_alive() {
                    // Tombstoned by `remove_node` — mark the row dead
                    // in the ColumnStore so reloads skip it.
                    new_store.tombstone(row_id);
                    continue;
                }
                // Title is in its own column, written only when the cached
                // value differs from the stored one: `TypedColumn::Str::set`
                // updates just offsets[idx]/offsets[idx+1] instead of shifting
                // the tail, so a same-row overwrite corrupts every following
                // row's title on reload.
                if !matches!(nd.title, Value::Null) {
                    let current = new_store.get_title(row_id);
                    let differs = match (&current, &nd.title) {
                        (Some(a), b) => a != b,
                        (None, _) => true,
                    };
                    if differs {
                        let _ = new_store.set_title(row_id, &nd.title);
                    }
                }
                if let PropertyStorage::Map(map) = &nd.properties {
                    for (key, value) in map {
                        let _ = new_store.set(row_id, *key, value, None);
                    }
                }
            }
            // Replace the Arc wholesale. This map is the only owner, so every
            // later reader sees the flushed store without a mirror step.
            self.column_stores
                .insert(type_key, std::sync::Arc::new(new_store));
        }
    }

    /// Enter a read query that may materialize disk-backed nodes or edges.
    ///
    /// Overlapping and nested queries each take their own guard; a query's
    /// materializations are released when *its* guard drops, unless an older
    /// query is still running (epoch protocol in [`super::query_arena`]). No
    /// query can invalidate references held by another.
    pub(crate) fn begin_query(&self) -> DiskQueryGuard {
        self.arenas.begin()
    }

    #[cfg(test)]
    pub(crate) fn node_arena_len(&self) -> usize {
        self.arenas.node_len()
    }

    #[cfg(test)]
    pub(crate) fn edge_arena_len(&self) -> usize {
        self.arenas.edge_len()
    }

    #[cfg(test)]
    pub(crate) fn active_query_count(&self) -> usize {
        self.arenas.active_count()
    }

    /// Reclaim materialization arenas when no guarded read query is active.
    /// Mutation execution calls this while holding exclusive graph ownership;
    /// the active-count check also makes accidental concurrent resets safe.
    pub fn reset_arenas(&self) {
        self.arenas.reclaim_if_idle();
    }

    pub fn edges_directed_iter(&self, a: NodeIndex, dir: Direction) -> DiskEdges<'_> {
        self.edges_directed_filtered_iter(a, dir, None)
    }

    pub fn edges_directed_filtered_iter(
        &self,
        a: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<u64>,
    ) -> DiskEdges<'_> {
        self.ensure_csr();
        let node = a.index();
        let (offsets, edges) = match dir {
            Direction::Outgoing => (&self.out_offsets, &self.out_edges),
            Direction::Incoming => (&self.in_offsets, &self.in_edges),
        };
        let overflow = match dir {
            Direction::Outgoing => self.overflow_out.get(&(node as u32)),
            Direction::Incoming => self.overflow_in.get(&(node as u32)),
        };

        // If the CSR offset table hasn't been built yet (fresh disk graph
        // pre-first-build, or a node added after build_csr_from_pending),
        // the overflow path may still carry edges. Fall through with an
        // empty CSR range instead of skipping the iterator entirely.
        let (start, end) = if node < offsets.len().saturating_sub(1) {
            (offsets.get(node) as usize, offsets.get(node + 1) as usize)
        } else {
            (0, 0)
        };

        let iter = DiskEdges::new(self, dir, a, edges, start, end, overflow);
        if let Some(ct) = conn_type_filter {
            iter.with_conn_type_filter(ct)
        } else {
            iter
        }
    }

    pub fn edge_references_iter(&self) -> DiskEdgeReferences<'_> {
        self.ensure_csr();
        DiskEdgeReferences::new(self)
    }

    pub fn edge_indices_iter(&self) -> DiskEdgeIndices<'_> {
        self.ensure_csr();
        DiskEdgeIndices::new(self.next_edge_idx, self)
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        self.ensure_csr();
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoint(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }

        Some(self.materialize_edge(ei as u32))
    }

    pub fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoint(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }
        // Store in a dedicated cache, not the arena: the arena is append-only
        // and shared with the read-only `edge_weight`, so flushing writes back
        // out of it would need fragile offset tracking.
        let ct = InternedKey::from_u64(ep.connection_type);
        let props = self
            .edge_properties
            .get(ei as u32)
            .map(|cow| cow.into_owned())
            .unwrap_or_default();
        self.edge_mut_cache.entry(ei as u32).or_insert(EdgeData {
            connection_type: ct,
            properties: props,
        });
        Some(self.edge_mut_cache.get_mut(&(ei as u32)).unwrap())
    }

    pub fn edge_endpoints_fn(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.ensure_csr();
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoint(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }
        Some((
            NodeIndex::new(ep.source as usize),
            NodeIndex::new(ep.target as usize),
        ))
    }

    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        if self.defer_csr {
            return self
                .try_add_pending_edge(a, b, data)
                .expect("deferred bulk callers must use try_add_pending_edge");
        }
        self.clear_arenas();
        let edge_idx = self.next_edge_idx;
        self.next_edge_idx += 1;

        let ct = data.connection_type;

        if !data.properties.is_empty() {
            self.edge_properties.insert(edge_idx, data.properties);
        }

        let src = a.index() as u32;
        let tgt = b.index() as u32;
        let ct_u64 = ct.as_u64();

        // Post-CSR mode: go directly to overflow + edge_endpoints.
        self.appended_edge_endpoints.push(EdgeEndpoints {
            source: src,
            target: tgt,
            connection_type: ct_u64,
        });
        self.overflow_out.entry(src).or_default().push(CsrEdge {
            peer: tgt,
            edge_idx,
        });
        self.overflow_in.entry(tgt).or_default().push(CsrEdge {
            peer: src,
            edge_idx,
        });

        self.edge_count += 1;
        EdgeIndex::new(edge_idx as usize)
    }

    /// Fallible bulk-build edge append. The mmap write happens before any
    /// counter or property change, so callers can retry an allocation/I/O
    /// failure without observing a partial edge.
    pub(crate) fn try_add_pending_edge(
        &mut self,
        a: NodeIndex,
        b: NodeIndex,
        data: EdgeData,
    ) -> std::io::Result<EdgeIndex> {
        let edge_idx = self.next_edge_idx;
        let pending = PendingEdge {
            source: a.index() as u32,
            target: b.index() as u32,
            connection_type: data.connection_type.as_u64(),
        };
        self.pending_edges.get_mut().try_push(pending)?;

        if !data.properties.is_empty() {
            self.edge_properties.insert(edge_idx, data.properties);
        }
        self.next_edge_idx += 1;
        self.edge_count += 1;
        Ok(EdgeIndex::new(edge_idx as usize))
    }

    pub fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.clear_arenas();
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoint(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }

        let ct = InternedKey::from_u64(ep.connection_type);
        let props = self.edge_properties.take(ei as u32).unwrap_or_default();
        let result = EdgeData {
            connection_type: ct,
            properties: props,
        };

        let src = ep.source as usize;
        let tgt = ep.target as usize;
        let ei32 = ei as u32;

        if let Some(list) = self.overflow_out.get_mut(&(src as u32)) {
            list.retain(|e| e.edge_idx != ei32);
        }
        if let Some(list) = self.overflow_in.get_mut(&(tgt as u32)) {
            list.retain(|e| e.edge_idx != ei32);
        }

        // Logical tombstone only: the published endpoint/CSR files are an
        // immutable reader snapshot. Iterators consult this overlay.
        self.removed_edges.insert(ei32);

        self.edge_count -= 1;
        self.free_edge_slots.push(ei32);
        self.has_tombstones = true;
        self.csr_sorted_by_type = false;
        Some(result)
    }

    pub fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        self.ensure_csr();
        let src = a.index();
        let tgt = b.index() as u32;

        if src < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(src) as usize;
            let end = self.out_offsets.get(src + 1) as usize;
            for i in start..end {
                let e = self.out_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE && e.peer == tgt {
                    return Some(EdgeIndex::new(e.edge_idx as usize));
                }
            }
        }

        if let Some(list) = self.overflow_out.get(&(src as u32)) {
            for e in list {
                if e.edge_idx != TOMBSTONE_EDGE && e.peer == tgt {
                    return Some(EdgeIndex::new(e.edge_idx as usize));
                }
            }
        }

        None
    }

    pub fn edges_connecting_iter(&self, a: NodeIndex, b: NodeIndex) -> DiskEdgesConnecting<'_> {
        self.ensure_csr();
        DiskEdgesConnecting::new(self, a, b)
    }

    pub fn edge_weights_iter(&self) -> Box<dyn Iterator<Item = &EdgeData> + '_> {
        self.ensure_csr();
        Box::new((0..self.next_edge_idx).filter_map(move |i| {
            let ep = self.edge_endpoint(i as usize);
            if ep.source == TOMBSTONE_EDGE {
                return None;
            }

            Some(self.materialize_edge(i))
        }))
    }

    pub fn neighbors_directed_iter(&self, a: NodeIndex, dir: Direction) -> DiskNeighbors {
        self.ensure_csr();
        let node = a.index();
        let (offsets, edges) = match dir {
            Direction::Outgoing => (&self.out_offsets, &self.out_edges),
            Direction::Incoming => (&self.in_offsets, &self.in_edges),
        };

        let overflow = match dir {
            Direction::Outgoing => self.overflow_out.get(&(node as u32)),
            Direction::Incoming => self.overflow_in.get(&(node as u32)),
        };

        // Empty CSR range when offsets don't cover `node + 1` (overflow-only
        // node); still build the iterator so overflow edges are yielded.
        let (start, end) = if node + 1 < offsets.len() {
            (offsets.get(node) as usize, offsets.get(node + 1) as usize)
        } else {
            (0, 0)
        };

        DiskNeighbors::new(edges, start, end, overflow)
    }

    pub fn neighbors_undirected_iter(&self, a: NodeIndex) -> DiskNeighbors {
        self.ensure_csr();
        let node = a.index();
        let mut peers = Vec::new();

        if node < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(node) as usize;
            let end = self.out_offsets.get(node + 1) as usize;
            for i in start..end {
                let e = self.out_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }
        if let Some(list) = self.overflow_out.get(&(node as u32)) {
            for e in list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }

        if node < self.in_offsets.len().saturating_sub(1) {
            let start = self.in_offsets.get(node) as usize;
            let end = self.in_offsets.get(node + 1) as usize;
            for i in start..end {
                let e = self.in_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }
        if let Some(list) = self.overflow_in.get(&(node as u32)) {
            for e in list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }

        DiskNeighbors::from_collected(peers)
    }

    /// True if any overflow edges are present (edges added after the initial
    /// CSR build). Save uses it to decide whether to `compact` first, so the
    /// derived indexes it rebuilds cover every live edge; the per-batch
    /// `ensure_disk_edges_built` deliberately skips that O(E) merge.
    pub fn has_overflow(&self) -> bool {
        self.overflow_out.values().any(|v| !v.is_empty())
            || self.overflow_in.values().any(|v| !v.is_empty())
    }

    pub fn build_csr_from_pending(&mut self) -> std::io::Result<()> {
        let node_bound = self.node_slot_len();
        let pending = self.pending_edges.get_mut();
        if pending.is_empty() {
            return Ok(());
        }

        let edge_count = pending.len();
        let verbose = std::env::var("KGLITE_BUILD_DEBUG").is_ok();
        let use_merge_sort = std::env::var("KGLITE_CSR_ALGO").is_ok_and(|v| v == "merge_sort");
        if use_merge_sort {
            self.build_csr_merge_sort(node_bound, edge_count, verbose)?;
        } else {
            self.build_csr_partitioned(node_bound, edge_count, verbose)?;
        }
        // Subsequent add_edge calls now route to overflow.
        self.defer_csr = false;
        Ok(())
    }

    /// Merge overflow edges back into the CSR arrays via full rebuild.
    /// Collects all live edges (CSR + overflow, excluding tombstones),
    /// writes to pending_edges, clears overflow, and rebuilds CSR.
    /// Returns the number of overflow edges that were merged.
    pub fn compact(&mut self) -> std::io::Result<usize> {
        let overflow_count: usize = self.overflow_out.values().map(|v| v.len()).sum();
        if overflow_count == 0 {
            return Ok(0);
        }

        let verbose = std::env::var("KGLITE_BUILD_DEBUG").is_ok();
        if verbose {
            eprintln!(
                "Compacting: {} CSR edges + {} overflow edges",
                self.edge_count.saturating_sub(overflow_count),
                overflow_count
            );
        }

        let node_bound = self.node_slot_len();

        // Collect live edges from edge_endpoints, which covers both CSR and
        // post-CSR overflow edges.
        let mut live_count = 0usize;
        let total_endpoints = self.next_edge_idx as usize;

        let pending_path = self.active_write_dir().join("_compact_pending.bin");
        let mut new_pending: MmapOrVec<PendingEdge> =
            MmapOrVec::mapped(&pending_path, total_endpoints)?;

        // Edge index remapping: old_idx → new_idx
        // Needed because compaction produces a dense edge array.
        let mut idx_remap: Vec<u32> = vec![TOMBSTONE_EDGE; total_endpoints];

        for (old_idx, remap_slot) in idx_remap.iter_mut().enumerate().take(total_endpoints) {
            let ep = self.edge_endpoint(old_idx);
            if ep.source != TOMBSTONE_EDGE
                && (ep.source as usize) < node_bound
                && (ep.target as usize) < node_bound
            {
                *remap_slot = live_count as u32;
                new_pending.try_push(PendingEdge {
                    source: ep.source,
                    target: ep.target,
                    connection_type: ep.connection_type,
                })?;
                live_count += 1;
            }
        }

        // `mem::take` gives us ownership of the old store (base mmaps
        // stay live until we drop it at end of scope); we iterate every
        // potentially-populated slot and re-insert survivors into the
        // fresh store. Properties of tombstoned edges are discarded.
        let old_props = std::mem::take(&mut self.edge_properties);
        let upper = old_props.upper_bound();
        for old_idx in 0..upper {
            if let Some(cow) = old_props.get(old_idx) {
                let new_idx = idx_remap[old_idx as usize];
                if new_idx != TOMBSTONE_EDGE {
                    self.edge_properties.insert(new_idx, cow.into_owned());
                }
            }
        }
        drop(old_props);

        self.overflow_out.clear();
        self.overflow_in.clear();
        self.free_edge_slots.clear();

        self.edge_count = live_count;
        self.next_edge_idx = live_count as u32;

        let old_pending_path = self
            .pending_edges
            .get_mut()
            .file_path()
            .map(|p| p.to_path_buf());
        // The assignment drops the previous buffer's mapping, so the old
        // backing file is unmapped before it is deleted.
        *self.pending_edges.get_mut() = new_pending;
        if let Some(path) = old_pending_path {
            super::remove_scratch_file(&path)?;
        }

        self.build_csr_from_pending()?;

        // Clean up the compaction scratch file. `build_csr_from_pending`
        // normally clears `pending_edges` (releasing this mapping), but it
        // returns early when the buffer is empty — so release it explicitly
        // rather than relying on that. Deleting a still-mapped file is an
        // error on Windows, and the removal is checked so it stays visible.
        if self.pending_edges.get_mut().file_path() == Some(pending_path.as_path()) {
            *self.pending_edges.get_mut() = MmapOrVec::new();
        }
        super::remove_scratch_file(&pending_path)?;

        if verbose {
            eprintln!(
                "Compaction done: {} live edges (removed {} tombstoned)",
                live_count,
                total_endpoints - live_count
            );
        }

        Ok(overflow_count)
    }

    pub fn lookup_peer_counts(&self, conn_type: u64) -> Option<HashMap<u32, i64>> {
        if self.peer_count_types.is_empty() {
            return None;
        }
        let n = self.peer_count_types.len();
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let t = self.peer_count_types.get(mid);
            match t.cmp(&conn_type) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let start = self.peer_count_offsets.get(mid) as usize;
                    let end = self.peer_count_offsets.get(mid + 1) as usize;
                    let mut out: HashMap<u32, i64> = HashMap::with_capacity(end - start);
                    for i in start..end {
                        let peer = self.peer_count_entries.get(i * 2);
                        let count = self.peer_count_entries.get(i * 2 + 1);
                        out.insert(peer, count as i64);
                    }
                    return Some(out);
                }
            }
        }
        // Type not found in histogram. Return None so the caller falls back
        // to the sequential `count_edges_grouped_by_peer` scan — the
        // histogram may be stale (e.g. built pre-overflow-merge) and we
        // prefer a slow but correct answer over a fast but empty one.
        None
    }

    fn tombstone_edges_for_node(&mut self, node: usize) {
        let mut incident = HashSet::new();
        if node < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(node) as usize;
            let end = self.out_offsets.get(node + 1) as usize;
            for i in start..end {
                let edge = self.out_edges.get(i);
                if edge.edge_idx != TOMBSTONE_EDGE && self.edge_is_alive(edge.edge_idx) {
                    incident.insert(edge.edge_idx);
                }
            }
        }
        if node < self.in_offsets.len().saturating_sub(1) {
            let start = self.in_offsets.get(node) as usize;
            let end = self.in_offsets.get(node + 1) as usize;
            for i in start..end {
                let edge = self.in_edges.get(i);
                if edge.edge_idx != TOMBSTONE_EDGE && self.edge_is_alive(edge.edge_idx) {
                    incident.insert(edge.edge_idx);
                }
            }
        }
        if let Some(list) = self.overflow_out.get(&(node as u32)) {
            for edge in list {
                if self.edge_is_alive(edge.edge_idx) {
                    incident.insert(edge.edge_idx);
                }
            }
        }
        if let Some(list) = self.overflow_in.get(&(node as u32)) {
            for edge in list {
                if self.edge_is_alive(edge.edge_idx) {
                    incident.insert(edge.edge_idx);
                }
            }
        }

        for edge_idx in incident {
            let endpoint = self.edge_endpoint(edge_idx as usize);
            if let Some(list) = self.overflow_out.get_mut(&endpoint.source) {
                list.retain(|edge| edge.edge_idx != edge_idx);
            }
            if let Some(list) = self.overflow_in.get_mut(&endpoint.target) {
                list.retain(|edge| edge.edge_idx != edge_idx);
            }
            self.removed_edges.insert(edge_idx);
            self.edge_properties.remove(edge_idx);
            self.edge_count -= 1;
            self.free_edge_slots.push(edge_idx);
        }
        self.csr_sorted_by_type = false;
    }
}

impl Clone for DiskGraph {
    fn clone(&self) -> Self {
        // Published disk arrays are immutable; a transaction remaps their
        // files and copies only mutation-sized overlays. This is O(number of
        // changed rows), not O(nodes + edges), and keeps reader snapshots on
        // the prior generation. Heap-backed arrays still clone normally.
        fn snapshot<T: crate::graph::storage::mapped::mmap_vec::MmapPod>(
            name: &str,
            value: &MmapOrVec<T>,
        ) -> MmapOrVec<T> {
            value
                .clone_snapshot()
                .unwrap_or_else(|error| panic!("failed to clone disk {name} snapshot: {error}"))
        }

        DiskGraph {
            node_slots: snapshot("node slots", &self.node_slots),
            node_slot_updates: self.node_slot_updates.clone(),
            appended_node_slots: self.appended_node_slots.clone(),
            node_count: self.node_count,
            free_node_slots: self.free_node_slots.clone(),
            arenas: super::query_arena::QueryArenas::new(0),
            column_stores: self.column_stores.clone(),
            out_offsets: snapshot("out offsets", &self.out_offsets),
            out_edges: snapshot("out edges", &self.out_edges),
            in_offsets: snapshot("in offsets", &self.in_offsets),
            in_edges: snapshot("in edges", &self.in_edges),
            edge_endpoints: snapshot("edge endpoints", &self.edge_endpoints),
            appended_edge_endpoints: self.appended_edge_endpoints.clone(),
            removed_edges: self.removed_edges.clone(),
            edge_count: self.edge_count,
            next_edge_idx: self.next_edge_idx,
            edge_properties: self.edge_properties.fork_overlay(),
            edge_mut_cache: HashMap::new(),
            node_mut_cache: HashMap::new(),
            // SAFETY: cloning takes `&self`; every mutation of pending_edges is
            // gated by `&mut self`, so no writer can overlap this read.
            pending_edges: UnsafeCell::new(unsafe { &*self.pending_edges.get() }.clone()),
            overflow_out: self.overflow_out.clone(),
            overflow_in: self.overflow_in.clone(),
            free_edge_slots: self.free_edge_slots.clone(),
            data_dir: self.data_dir.clone(),
            logical_root: self.logical_root.clone(),
            writer_lock: None,
            mutation_workspace: None,
            parent_workspaces: self.parent_workspaces.clone(),
            independent_root: self.independent_root.clone(),
            csr_sorted_by_type: self.csr_sorted_by_type,
            defer_csr: self.defer_csr,
            edge_type_counts_raw: self.edge_type_counts_raw.clone(),
            conn_type_index_types: snapshot(
                "connection type index types",
                &self.conn_type_index_types,
            ),
            conn_type_index_offsets: snapshot(
                "connection type index offsets",
                &self.conn_type_index_offsets,
            ),
            conn_type_index_sources: snapshot(
                "connection type index sources",
                &self.conn_type_index_sources,
            ),
            peer_count_types: snapshot("peer count types", &self.peer_count_types),
            peer_count_offsets: snapshot("peer count offsets", &self.peer_count_offsets),
            peer_count_entries: snapshot("peer count entries", &self.peer_count_entries),
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            has_tombstones: self.has_tombstones,
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            removed_property_indexes: self.removed_property_indexes.clone(),
            segment_manifest: self.segment_manifest.clone(),
            sealed_nodes_bound: self.sealed_nodes_bound,
        }
    }
}

impl DiskGraph {
    /// Detach a user-requested copy from the source graph's writer lease.
    /// Immutable mapped arrays remain shared, while the first subsequent
    /// write materialises a private root and workspace.
    pub(crate) fn detach_for_independent_copy(&mut self, parent: &Self) {
        self.parent_workspaces = parent.parent_workspaces.clone();
        if let Some(workspace) = &parent.mutation_workspace {
            self.parent_workspaces.push(Arc::clone(workspace));
        }
        let ancestors = parent.independent_root.iter().cloned().collect();
        let root = Arc::new(super::generation::IndependentGraphRoot::new(ancestors));
        self.logical_root = root.path().to_path_buf();
        self.writer_lock = None;
        self.mutation_workspace = None;
        self.independent_root = Some(root);
    }

    /// Whether a bulk build may write straight into this graph's own
    /// directory — true only while `data_dir` is a segment of the graph root
    /// itself (`root/seg_000`, or the root for the pre-segment layout).
    ///
    /// A handle that has published a generation — an explicit `save()`, or a
    /// graph opened from a saved directory — has `data_dir` *inside that
    /// immutable snapshot* instead, and so does a detached copy, whose base
    /// snapshot belongs to the graph it was copied from.
    fn builds_in_its_own_directory(&self) -> bool {
        self.data_dir == self.logical_root
            || self.data_dir.parent() == Some(self.logical_root.as_path())
    }

    /// Stage a bulk build in a mutation workspace unless it can safely write
    /// where it lies.
    ///
    /// A build writes straight into [`Self::active_write_dir`]. For a freshly
    /// created directory that is the graph's own segment — fine, and the
    /// reason a fresh disk build is reloadable with no `save()`. For a handle
    /// sitting on a *published generation* it is a snapshot readers can
    /// already see: the rebuild's sidecars land beside the snapshot's own, the
    /// snapshot's `interner.bin.zst` shadows the rebuilt `interner.json`, and
    /// the reload fails outright with an unresolved type key. Published
    /// generations are immutable, so the build stages into a workspace and
    /// `finalize_disk_graph` publishes it as a new generation instead.
    pub(crate) fn prepare_bulk_load_workspace(&mut self) -> std::io::Result<()> {
        if !self.builds_in_its_own_directory() {
            self.prepare_mutation()?;
        }
        Ok(())
    }

    /// The graph root a finished bulk build must be published into as a new
    /// generation, or `None` when it may be published where it lies.
    ///
    /// A build that ran in a mutation workspace is not reachable from the
    /// graph root at all — the workspace is removed when the handle drops — so
    /// leaving it there loses the entire build silently. Republishing it as a
    /// generation is the only way it survives, and it is what any other write
    /// staged in a workspace does at `save()`.
    ///
    /// A detached copy is the exception: its workspace *is* its graph
    /// directory (a private root under the temp dir), it has no generations,
    /// and its own `save()` to a real path is the publication.
    pub(crate) fn bulk_build_generation_root(&self) -> Option<&Path> {
        if self.independent_root.is_some() {
            return None;
        }
        self.mutation_workspace
            .as_ref()
            .map(|_| self.logical_root.as_path())
    }

    #[cfg(test)]
    pub(crate) fn independent_root_path(&self) -> Option<&std::path::Path> {
        self.independent_root.as_deref().map(|root| root.path())
    }

    /// Adopt the retained writer lease from a shared-identity parent.
    /// Generic clones intentionally lack writer authority unless a controlled
    /// transaction or Arc copy-on-write path calls this method.
    pub(crate) fn adopt_writer_lineage(&mut self, parent: &Self) {
        self.writer_lock = parent.writer_lock.clone();
        self.parent_workspaces = parent.parent_workspaces.clone();
        if let Some(workspace) = &parent.mutation_workspace {
            self.parent_workspaces.push(Arc::clone(workspace));
        }
        self.mutation_workspace = None;
        self.independent_root = parent.independent_root.clone();
    }
}

impl std::ops::Index<NodeIndex> for DiskGraph {
    type Output = NodeData;
    #[inline]
    fn index(&self, index: NodeIndex) -> &NodeData {
        self.node_weight(index).expect("DiskGraph: node not found")
    }
}

impl std::ops::Index<EdgeIndex> for DiskGraph {
    type Output = EdgeData;
    #[inline]
    fn index(&self, index: EdgeIndex) -> &EdgeData {
        self.edge_weight(index).expect("DiskGraph: edge not found")
    }
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod tests;
