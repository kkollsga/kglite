// src/graph/disk_graph.rs
//
// Disk-backed graph storage using CSR (Compressed Sparse Row) format.
// Nodes are stored in memory (~40 bytes each in Columnar mode).
// Edges are stored in mmap'd CSR arrays (8 bytes per edge per direction).
// Edge properties are stored sparsely (only for edges that have them).
//
// Memory budget: ~10% of equivalent petgraph InMemory graph.
// For 100M nodes + 1B edges: ~5-6 GB RAM + OS page cache.

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
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints, TOMBSTONE_EDGE};
use super::edge_properties::EdgePropertyStore;
use super::property_index;

/// PR1 phase 4: CSR + column binaries live in a per-segment subdirectory
/// of the graph root. Top-level files (disk_graph_meta.json,
/// seg_manifest.json, interner.json, metadata.json) stay at the graph root.
/// Legacy graphs gated by `DiskGraphMeta::csr_layout_version` == 0 use the
/// flat layout (everything at the root) — see `load_from_dir`.
///
/// PR1 phase 6 parameterised the directory name on segment id so future
/// saves can emit `seg_001/`, `seg_002/`, … next to the original
/// `seg_000/`. Call sites format via [`segment_subdir`]; directories are
/// discovered via [`enumerate_segment_dirs`].
pub(crate) fn segment_subdir(id: u32) -> String {
    // Three-digit zero-padding is enough for Wikidata-scale graphs
    // (plan targets ~200 segments); overflow past 999 is handled by
    // `{:03}` naturally widening without changing lexicographic ordering
    // (seg_999 < seg_1000 still sorts as seg_1000 first, but we use the
    // parsed u32 for ordering — see `enumerate_segment_dirs`).
    format!("seg_{id:03}")
}

/// Discover every `seg_NNN/` subdirectory under `root`, sorted
/// ascending by the numeric id parsed from the name. Returns
/// `(segment_id, path)` pairs. Non-matching directory entries and
/// unparsable `seg_*` names are skipped silently.
///
/// Used at load time to drive the CSR-load enumeration; at save time
/// the next free id is `last().map(|(id, _)| id + 1).unwrap_or(0)`.
/// PR1 phase 6 infrastructure — today every graph has exactly one
/// segment directory (`seg_000`), so the returned Vec has length 1
/// for non-legacy graphs. Phase 7+ will produce additional segments.
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

// ============================================================================
// DiskGraph
// ============================================================================

/// Truly disk-backed graph. All data lives on disk via mmap.
///
/// - Nodes: `MmapOrVec<DiskNodeSlot>` (16 bytes/node, mmap'd)
///   Actual node data (id, title, properties) in ColumnStore columns (mmap'd).
///   `node_weight()` materializes NodeData into an arena on access.
/// - Edges: CSR arrays (`out_offsets`, `out_edges`, etc.) — mmap'd
/// - Edge properties: sparse HashMap (loaded to heap)
/// - Arenas: append-only caches for materialized NodeData/EdgeData refs
pub struct DiskGraph {
    // ── Node storage (mmap'd on disk) ──
    pub(crate) node_slots: MmapOrVec<DiskNodeSlot>,
    pub(super) node_count: usize,
    pub(super) free_node_slots: Vec<u32>,

    // ── Node materialization arena ──
    //
    // `Mutex<Vec<Box<NodeData>>>` mirrors `edge_arena` below — the Box
    // gives stable heap pointers that survive Vec growth, and the Mutex
    // serialises pushes from concurrent `node_weight` callers (the
    // Cypher executor's `return_clause::project_row` runs under
    // `par_iter_mut`, and any call to `evaluate_expression` from there
    // can reach `node_weight` via `resolve_property` /
    // `build_node_spatial_data`). Pre-0.9.3 this was
    // `UnsafeCell<Vec<NodeData>>` which raced under the same parallel
    // path: a concurrent `arena.push` realloc invalidated references
    // returned by sibling Rayon tasks, leaking into either silent
    // wrong-row reads (Bug A in the disk-mode regression report —
    // ~13% NEAREST_AFEX_HUB edges lost) or use-after-free segfaults
    // with `BUG: InternedKey N not found in StringInterner` lines on
    // stderr (Bug B in the same report).
    // clippy::vec_box: the Box is load-bearing — gives stable heap
    // addresses that survive Vec growth, so `&NodeData` references
    // returned by `node_weight` stay valid across concurrent pushes
    // from sibling Rayon tasks. Removing the Box would put the
    // race back. Same exemption applies to `edge_arena` below.
    #[allow(clippy::vec_box)]
    pub(super) node_arena: std::sync::Mutex<Vec<Box<NodeData>>>,

    // ── Column stores for node properties (Arc refs, data mmap'd) ──
    pub(crate) column_stores:
        HashMap<InternedKey, Arc<crate::graph::storage::column_store::ColumnStore>>,

    // ── Edge CSR (mmap'd) ──
    pub(super) out_offsets: MmapOrVec<u64>,
    pub(super) out_edges: MmapOrVec<CsrEdge>,
    pub(super) in_offsets: MmapOrVec<u64>,
    pub(super) in_edges: MmapOrVec<CsrEdge>,

    // ── Edge metadata ──
    pub(crate) edge_endpoints: MmapOrVec<EdgeEndpoints>,
    pub(crate) edge_count: usize,
    pub(crate) next_edge_idx: u32,

    // ── Edge properties (columnar base + mutation overlay) ──
    // PR2: heap-only HashMap replaced by disk-backed columnar store.
    // Overlay grows with mutations, base is mmap'd. See edge_properties.rs.
    pub(super) edge_properties: EdgePropertyStore,

    // ── Edge materialization arena (Mutex + Box for Rayon thread safety) ──
    // Box gives stable heap pointers that survive Vec reallocation.
    #[allow(clippy::vec_box)]
    pub(super) edge_arena: std::sync::Mutex<Vec<Box<EdgeData>>>,
    /// Cache for edge_weight_mut: stores materialized EdgeData that may be modified.
    /// Flushed to edge_properties on next clear_arenas call.
    pub(super) edge_mut_cache: HashMap<u32, EdgeData>,
    /// Cache for `node_weight_mut`: stages Cypher-SET-style exact-row
    /// writes as `PropertyStorage::Map`. On `clear_arenas`, drains are
    /// grouped by type, each type's ColumnStore is cloned once, all
    /// staged writes are applied to the clone, and the result is
    /// inserted back into `self.column_stores` as a fresh `Arc`. This
    /// mirrors the full-`Arc` replacement pattern that
    /// `batch.rs::flush_chunk` uses for `add_nodes` — avoiding
    /// `Arc::make_mut` on an `Arc` that DirGraph still holds (which
    /// would clone + diverge, losing writes).
    pub(super) node_mut_cache: HashMap<u32, NodeData>,

    // ── Pending edges (UnsafeCell reserved for a planned &self CSR-build path) ──
    // File-backed (MmapOrVec) to avoid ~14 GB heap allocation at Wikidata scale.
    // Today every access goes through `get_mut()` in `&mut self` methods
    // (add_edge with defer_csr, build_csr_from_pending, compact) — see the
    // struct-level SAFETY block for the full accounting of interior-mutability
    // fields.
    pub(crate) pending_edges: UnsafeCell<MmapOrVec<(u32, u32, u64)>>,

    // ── Mutation overflow (for incremental edges after CSR) ──
    pub(super) overflow_out: HashMap<u32, Vec<CsrEdge>>,
    pub(super) overflow_in: HashMap<u32, Vec<CsrEdge>>,
    pub(super) free_edge_slots: Vec<u32>,

    // ── Storage directory (the graph lives here) ──
    pub(crate) data_dir: PathBuf,
    // ── Dirty flag: flushed on Drop or next query ──
    pub(super) metadata_dirty: bool,
    // ── CSR edges are sorted by (node, connection_type) — enables binary search
    pub(crate) csr_sorted_by_type: bool,
    // ── Defer CSR build: when true, ensure_csr() is a no-op. Edges accumulate
    // in pending_edges without intermediate CSR rebuilds. The CSR is built once
    // at save time via ensure_disk_edges_built(). Set true during construction
    // from add_nodes/add_connections, cleared after CSR build.
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
    // Populated in two ways:
    //   1. `build_property_index(type, prop)` — user calls `create_index`
    //      on a disk graph; writes 4 files to `data_dir` and caches the
    //      handle.
    //   2. `lookup_property_eq(type, prop, value)` on first miss — scans
    //      `data_dir` for a `property_index_{type}_{prop}_meta.bin`; if
    //      present, mmaps it and caches.
    //
    // The `None` sentinel records "we checked and no index exists" so
    // repeat misses don't stat the filesystem. `Arc` so concurrent reads
    // of the same index don't hold the outer RwLock.
    pub(crate) property_indexes: PropertyIndexCache,
    // ── Persistent cross-type global property indexes (lazy-loaded).
    //
    // Keyed by property name only. Built by `build_global_property_index(prop)`
    // — scans every alive `DiskNodeSlot` and collects one
    // `(string_value, NodeIndex)` entry per node where `prop` resolves
    // (via column slot, title alias, or id alias). Powers untyped
    // patterns like `MATCH (n {label: 'X'})` where the agent doesn't
    // know the node type.
    pub(crate) global_indexes: GlobalIndexCache,
    // ── Segment manifest (PR1 phases 2/5, single-segment view today).
    //
    // Persisted at `seg_manifest.json` alongside the CSR files. Legacy
    // graphs that lack the file load as an empty manifest — the planner
    // treats that as "pre-segmented, don't prune" in subsequent phases.
    // Fresh saves always write a one-segment manifest describing the
    // whole graph; PR1 phase 5 populates `indexed_prop_ranges` for each
    // `PropertyIndex` discovered in the segment directory, using
    // `StringBloomPlaceholder` until the bloom-filter variant lands.
    // Phase 6+ will split into multiple segments once multi-segment
    // writes and reads are in place.
    pub(crate) segment_manifest: super::segment_summary::SegmentManifest,
    // ── Sealed-nodes watermark (PR1 phase 8).
    //
    // Node ids in `[0, sealed_nodes_bound)` are accounted for in a prior
    // sealed segment's `node_slots`. Node ids in
    // `[sealed_nodes_bound, node_count)` are either in the active
    // (still-mutable) tail — not yet sealed into any segment — or were
    // just added via `add_node`. `seal_to_new_segment` flushes the tail
    // into a new `seg_NNN/` and advances this watermark.
    //
    // Zero on freshly-built / pre-phase-8 graphs; the `DiskGraphMeta`
    // serde `default` keeps old `.kgl` directories loadable.
    pub(crate) sealed_nodes_bound: u32,
    // ── Temp dir for CSR mmap files (cleaned up on Drop) ──
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
// Two arena-like fields share the same thread-safety pattern:
//
// 1. `node_arena: Mutex<Vec<Box<NodeData>>>` — thread-safe for Rayon
//    parallel queries. `Box` gives stable heap pointers that survive
//    Vec reallocation; the Mutex serialises pushes from concurrent
//    `node_weight` callers. The Cypher executor's projection phase
//    runs `evaluate_expression` under `par_iter_mut`
//    (return_clause.rs), and any spatial / non-fast-path
//    `resolve_property` branch reaches `node_weight` through that
//    parallel context. Pre-0.9.3 this was `UnsafeCell<Vec<NodeData>>`
//    and races were silent: a sibling Rayon task's `arena.push`
//    realloc invalidated references already returned to other tasks,
//    surfacing as either wrong-row reads on disk-mode aggregations
//    (Bug A in the 0.9.2 disk regression — ~13% NEAREST_AFEX_HUB
//    edges silently lost) or use-after-free segfaults with
//    `BUG: InternedKey N not found in StringInterner` on stderr
//    (Bug B in the same report). Mirrors the long-standing pattern
//    used by `edge_arena` immediately below.
//
// 2. `edge_arena: Mutex<Vec<Box<EdgeData>>>` — thread-safe for Rayon
//    parallel queries (same pattern, predates 0.9.3).
//
// 3. `pending_edges: UnsafeCell<MmapOrVec<…>>` — only accessed via
//    `get_mut()` in `&mut self` contexts (`add_edge` with `defer_csr`,
//    `build_csr_from_pending`, `compact`). `UnsafeCell` is retained here
//    for a planned future auto-CSR-build-from-`&self` path; today no
//    `&self` access exists, so the current soundness argument is just
//    Rust's standard borrow checker.
unsafe impl Send for DiskGraph {}
unsafe impl Sync for DiskGraph {}

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

impl DiskGraph {
    // ====================================================================
    // Construction
    // ====================================================================

    /// Create an empty DiskGraph at the given directory path.
    /// All data is written directly to disk via mmap.
    ///
    /// Fresh graphs use the segmented layout (PR1 phase 4): CSR / column
    /// binaries live at `root/seg_000/*.bin`, top-level `disk_graph_meta.json`
    /// and `seg_manifest.json` stay at `root/`. Legacy .kgl directories with
    /// the flat (pre-phase-4) layout continue to load — see `load_from_dir`.
    pub fn new_at_path(root_dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(root_dir)?;
        let data_dir = root_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&data_dir)?;
        let data_dir = data_dir.as_path();

        Ok(DiskGraph {
            node_slots: MmapOrVec::mapped(&data_dir.join("node_slots.bin"), 1024)?,
            node_count: 0,
            free_node_slots: Vec::new(),
            node_arena: std::sync::Mutex::new(Vec::with_capacity(256)),
            column_stores: HashMap::new(),
            out_offsets: MmapOrVec::mapped(&data_dir.join("out_offsets.bin"), 1025)?,
            out_edges: MmapOrVec::new(),
            in_offsets: MmapOrVec::mapped(&data_dir.join("in_offsets.bin"), 1025)?,
            in_edges: MmapOrVec::new(),
            edge_endpoints: MmapOrVec::new(),
            edge_count: 0,
            next_edge_idx: 0,
            edge_properties: EdgePropertyStore::new(),
            edge_arena: std::sync::Mutex::new(Vec::with_capacity(256)),
            edge_mut_cache: HashMap::new(),
            node_mut_cache: HashMap::new(),
            pending_edges: UnsafeCell::new(
                MmapOrVec::mapped(&data_dir.join("_pending_edges.bin"), 1 << 20)
                    .unwrap_or_else(|_| MmapOrVec::new()),
            ),
            overflow_out: HashMap::new(),
            overflow_in: HashMap::new(),
            free_edge_slots: Vec::new(),
            data_dir: data_dir.to_path_buf(),
            metadata_dirty: false,
            csr_sorted_by_type: false,
            // Phase 5: `defer_csr = false` by default so one-off Cypher
            // CREATE / MERGE inserts route directly to overflow_out /
            // overflow_in + edge_endpoints, where `edges_directed` reads
            // them immediately. Bulk loaders that want to batch edges in
            // `pending_edges` and rebuild the CSR at the end (ntriples)
            // flip this to `true` on the freshly-constructed DiskGraph.
            // Previously the default-`true` path silently dropped
            // Cypher-created edges from subsequent MATCH queries — the
            // pending buffer was written but `edges_directed_filtered_iter`
            // only reads CSR + overflow, not pending.
            defer_csr: false,
            edge_type_counts_raw: None,
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
            has_tombstones: false,
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            segment_manifest: super::segment_summary::SegmentManifest::new(),
            // Freshly-created graph has no sealed segments yet; the
            // first save seals everything up to node_count into seg_000
            // and advances this watermark accordingly.
            sealed_nodes_bound: 0,
        })
    }

    /// Build a DiskGraph from a petgraph StableDiGraph.
    /// Converts nodes to DiskNodeSlots on disk, builds CSR arrays.
    ///
    /// `root_dir` is the graph root; CSR binaries land in `root_dir/seg_000/`
    /// per the PR1 phase-4 segment layout.
    pub fn from_stable_digraph(
        graph: &mut petgraph::stable_graph::StableDiGraph<NodeData, EdgeData>,
        root_dir: &Path,
    ) -> std::io::Result<Self> {
        use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};

        std::fs::create_dir_all(root_dir)?;
        let data_dir_buf = root_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&data_dir_buf)?;
        let data_dir = data_dir_buf.as_path();

        let node_bound = graph.node_bound();
        let edge_count = graph.edge_count();

        // ── Build node slots on disk ──
        let mut node_slots = MmapOrVec::mapped(&data_dir.join("node_slots.bin"), node_bound)?;
        let mut node_count = 0usize;
        for i in 0..node_bound {
            let idx = NodeIndex::new(i);
            if let Some(node) = graph.node_weight(idx) {
                let row_id = match &node.properties {
                    crate::graph::schema::PropertyStorage::Columnar { row_id, .. } => *row_id,
                    _ => i as u32,
                };
                node_slots.push(DiskNodeSlot {
                    node_type: node.node_type.as_u64(),
                    row_id,
                    flags: DiskNodeSlot::ALIVE_BIT,
                });
                node_count += 1;
            } else {
                node_slots.push(DiskNodeSlot::default()); // dead slot
            }
        }

        // ── Count outgoing/incoming edges per node ──
        let mut out_counts = vec![0u64; node_bound];
        let mut in_counts = vec![0u64; node_bound];
        for edge in graph.edge_references() {
            let s = edge.source().index();
            let t = edge.target().index();
            out_counts[s] += 1;
            in_counts[t] += 1;
        }

        // ── Build offset arrays (prefix sums) ──
        let mut out_offsets = MmapOrVec::mapped(&data_dir.join("out_offsets.bin"), node_bound + 1)?;
        let mut in_offsets = MmapOrVec::mapped(&data_dir.join("in_offsets.bin"), node_bound + 1)?;

        let mut out_acc = 0u64;
        let mut in_acc = 0u64;
        for i in 0..node_bound {
            out_offsets.push(out_acc);
            in_offsets.push(in_acc);
            out_acc += out_counts[i];
            in_acc += in_counts[i];
        }
        out_offsets.push(out_acc);
        in_offsets.push(in_acc);

        // ── Build CSR edge arrays ──
        let mut out_edges = MmapOrVec::mapped(&data_dir.join("out_edges.bin"), edge_count)?;
        let mut in_edges = MmapOrVec::mapped(&data_dir.join("in_edges.bin"), edge_count)?;
        let mut edge_endpoints_vec =
            MmapOrVec::mapped(&data_dir.join("edge_endpoints.bin"), edge_count)?;
        let mut edge_properties: HashMap<u32, Vec<(InternedKey, Value)>> = HashMap::new();

        // Initialize edge arrays with enough space
        for _ in 0..edge_count {
            out_edges.push(CsrEdge::default());
            in_edges.push(CsrEdge::default());
            edge_endpoints_vec.push(EdgeEndpoints::default());
        }

        // Fill positions: use write cursors per node
        let mut out_cursors = vec![0u64; node_bound];
        let mut in_cursors = vec![0u64; node_bound];

        let mut edge_idx = 0u32;
        for edge in graph.edge_references() {
            let s = edge.source().index();
            let t = edge.target().index();
            let ct = edge.weight().connection_type;

            let csr_out = CsrEdge {
                peer: t as u32,
                edge_idx,
            };
            let out_pos = out_offsets.get(s) + out_cursors[s];
            out_edges.set(out_pos as usize, csr_out);
            out_cursors[s] += 1;

            let csr_in = CsrEdge {
                peer: s as u32,
                edge_idx,
            };
            let in_pos = in_offsets.get(t) + in_cursors[t];
            in_edges.set(in_pos as usize, csr_in);
            in_cursors[t] += 1;

            edge_endpoints_vec.set(
                edge_idx as usize,
                EdgeEndpoints {
                    source: s as u32,
                    target: t as u32,
                    connection_type: ct.as_u64(),
                },
            );

            if !edge.weight().properties.is_empty() {
                edge_properties.insert(edge_idx, edge.weight().properties.clone());
            }

            edge_idx += 1;
        }

        Ok(DiskGraph {
            node_slots,
            node_count,
            free_node_slots: Vec::new(),
            node_arena: std::sync::Mutex::new(Vec::with_capacity(1024)),
            column_stores: HashMap::new(), // filled by caller via set_column_stores()
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints: edge_endpoints_vec,
            edge_count,
            next_edge_idx: edge_idx,
            edge_properties: EdgePropertyStore::from_overlay(edge_properties),
            edge_arena: std::sync::Mutex::new(Vec::with_capacity(1024)),
            edge_mut_cache: HashMap::new(),
            node_mut_cache: HashMap::new(),
            pending_edges: UnsafeCell::new(MmapOrVec::new()),
            overflow_out: HashMap::new(),
            overflow_in: HashMap::new(),
            free_edge_slots: Vec::new(),
            data_dir: data_dir.to_path_buf(),
            metadata_dirty: false,
            csr_sorted_by_type: false,
            // Phase 5: `defer_csr = false` by default so one-off Cypher
            // CREATE / MERGE inserts route directly to overflow_out /
            // overflow_in + edge_endpoints, where `edges_directed` reads
            // them immediately. Bulk loaders that want to batch edges in
            // `pending_edges` and rebuild the CSR at the end (ntriples)
            // flip this to `true` on the freshly-constructed DiskGraph.
            // Previously the default-`true` path silently dropped
            // Cypher-created edges from subsequent MATCH queries — the
            // pending buffer was written but `edges_directed_filtered_iter`
            // only reads CSR + overflow, not pending.
            defer_csr: false,
            edge_type_counts_raw: None,
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
            has_tombstones: false,
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            segment_manifest: super::segment_summary::SegmentManifest::new(),
            // Fresh build from a petgraph: no sealed segments yet.
            // First save seals the whole graph into seg_000.
            sealed_nodes_bound: 0,
        })
    }

    // ====================================================================
    // Node methods
    // ====================================================================

    /// Set column store references. Called by DirGraph after columnar setup or load.
    pub fn set_column_stores(
        &mut self,
        stores: HashMap<InternedKey, Arc<crate::graph::storage::column_store::ColumnStore>>,
    ) {
        self.column_stores = stores;
    }

    /// Iterate `(type_key, store_arc)` pairs — read-only borrow.
    /// Used by `DirGraph::sync_column_stores_from_disk` to mirror
    /// mutations applied in `clear_arenas` back into DirGraph's side of
    /// the Arc pair so the sidecar writer sees the post-mutation state.
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

    /// O(1) node type lookup from mmap'd node_slots — no materialization.
    /// Returns None if the node is dead or out of bounds.
    #[inline]
    pub fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
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
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }
        let type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&type_key)?;
        // Try schema column first, then fall back to special id/title columns.
        // The id and title are stored as __id__/__title__ (separate from schema),
        // so store.get() won't find them by their alias names (e.g., "title", "id").
        if let Some(val) = store.get(slot.row_id, key) {
            return Some(val);
        }
        // Fallback: check if this is an id/title alias
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
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
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
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }
        let type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&type_key)?;
        store.get_title(slot.row_id)
    }

    /// Get a DiskNodeSlot by index (for rebuild_type_indices without arena).
    #[inline]
    pub fn node_slot(&self, i: usize) -> DiskNodeSlot {
        if i < self.node_slots.len() {
            self.node_slots.get(i)
        } else {
            DiskNodeSlot::default()
        }
    }

    // ====================================================================
    // Node methods
    // ====================================================================

    /// Materialize a NodeData from disk slot + ColumnStore into the arena.
    #[inline]
    pub fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }

        // 0.9.0 Cluster 6 — preventative invariant. node_weight_mut
        // stages writes in node_mut_cache; if a *Map-typed* entry
        // exists for this index here on the read path, the staged
        // write is about to be silently shadowed by the column_stores
        // read (i.e. a missed flush_pending_writes call). 0.8.41's
        // execute_mutable post-write flush should keep this empty
        // outside of mid-mutation evaluation; firing here means a
        // new code path needs an explicit flush.
        //
        // 0.9.26: filter the check to `PropertyStorage::Map` entries.
        // `batch.rs::flush_chunk` (the `add_nodes` path) leaves
        // `PropertyStorage::Columnar { row_id, .. }` scratch in the
        // cache as a transient post-write artifact — that's "already
        // persisted via full-Arc replacement, safe to discard" (see
        // the reseed-path comment further down in this file). Those
        // entries are not a missed-flush concern; firing the
        // assertion on them was a false positive that made the
        // warning noisy during normal test runs.
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

        let node_data = if let Some(store) = store {
            let id = store.get_id(slot.row_id).unwrap_or(Value::Null);
            let title = store.get_title(slot.row_id).unwrap_or(Value::Null);
            NodeData {
                id,
                title,
                node_type: node_type_key,
                properties: crate::graph::schema::PropertyStorage::Columnar {
                    store: Arc::clone(store),
                    row_id: slot.row_id,
                },
            }
        } else {
            NodeData {
                id: Value::Null,
                title: Value::Null,
                node_type: node_type_key,
                properties: crate::graph::schema::PropertyStorage::Map(HashMap::new()),
            }
        };

        // `Box::new` puts NodeData on the heap; the pointer stays valid
        // even when the arena Vec reallocates on push. The Mutex
        // serialises pushes from concurrent `node_weight` callers (the
        // Cypher executor runs projection under `par_iter_mut` and
        // reaches this through `resolve_property` /
        // `build_node_spatial_data`). Mirrors the Box pattern already
        // used by `edge_arena`.
        let boxed = Box::new(node_data);
        let ptr: *const NodeData = &*boxed;
        self.node_arena.lock().unwrap().push(boxed);
        // SAFETY: `boxed` lives in the arena until `clear_arenas`
        // (`&mut self`) or `reset_arenas` (called between top-level
        // queries with no live materialization refs). Our `&self`
        // borrow keeps both at bay, so the heap pointer is valid for
        // the returned `&NodeData`'s lifetime.
        unsafe { Some(&*ptr) }
    }

    pub fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }

        let node_type_key = InternedKey::from_u64(slot.node_type);
        let key = i as u32;

        // Stage exact-row mutations in `node_mut_cache` with
        // `PropertyStorage::Map`, not `Columnar`. A `Columnar` variant
        // would route `node.set_property(k, v)` through
        // `Arc::make_mut(store)`; that clones the store when DirGraph
        // still holds another Arc, and the mutation then lands on a
        // detached copy. Using Map captures every write in the node's
        // own state; `clear_arenas` groups by type, clones each
        // ColumnStore once, applies all pending writes, and replaces
        // the Arc — mirroring `batch.rs::flush_chunk`.
        //
        // Reseed path: `batch.rs::flush_chunk` (and a few similar
        // bulk paths) assigns `node.properties = PropertyStorage::
        // Columnar{...}` as a transient step. If the cache still holds
        // that stale assignment next time we reach here, replace it
        // with Map — batch already persisted via full-Arc replacement,
        // so the stale Columnar scratch is safe to discard.
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
    pub fn node_bound(&self) -> usize {
        self.node_slots.len()
    }

    pub fn add_node(&mut self, data: NodeData) -> NodeIndex {
        self.clear_arenas();
        self.metadata_dirty = true;

        // Extract row_id from property storage if columnar, else use slot index
        let row_id = match &data.properties {
            crate::graph::schema::PropertyStorage::Columnar { row_id, .. } => *row_id,
            _ => self.node_slots.len() as u32,
        };

        let slot = DiskNodeSlot {
            node_type: data.node_type.as_u64(),
            row_id,
            flags: DiskNodeSlot::ALIVE_BIT,
        };

        if let Some(recycled) = self.free_node_slots.pop() {
            let idx = recycled as usize;
            self.node_slots.set(idx, slot);
            self.node_count += 1;
            NodeIndex::new(idx)
        } else {
            let idx = self.node_slots.len();
            self.node_slots.push(slot);
            // Extend CSR offset arrays
            let last_out = if !self.out_offsets.is_empty() {
                self.out_offsets.get(self.out_offsets.len() - 1)
            } else {
                0
            };
            self.out_offsets.push(last_out);
            let last_in = if !self.in_offsets.is_empty() {
                self.in_offsets.get(self.in_offsets.len() - 1)
            } else {
                0
            };
            self.in_offsets.push(last_in);
            self.node_count += 1;
            NodeIndex::new(idx)
        }
    }

    pub fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        self.metadata_dirty = true;
        self.clear_arenas();
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }

        // Materialize the NodeData before removing
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
            properties: if let Some(s) = store {
                crate::graph::schema::PropertyStorage::Columnar {
                    store: s,
                    row_id: slot.row_id,
                }
            } else {
                crate::graph::schema::PropertyStorage::Map(HashMap::new())
            },
        };

        // Mark slot as dead
        let mut dead_slot = slot;
        dead_slot.flags = 0;
        self.node_slots.set(i, dead_slot);
        self.node_count -= 1;
        self.free_node_slots.push(i as u32);
        self.has_tombstones = true;

        // Tombstone all incident edges
        self.tombstone_edges_for_node(i);

        Some(data)
    }

    /// Update the row_id in a node's DiskNodeSlot.
    /// Used after BuildColumnStore conversion to fix per-type row_id mapping.
    pub fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        let i = node_idx.index();
        if i < self.node_slots.len() {
            let mut slot = self.node_slots.get(i);
            slot.row_id = row_id;
            self.node_slots.set(i, slot);
        }
    }

    pub fn node_indices_iter(&self) -> DiskNodeIndices<'_> {
        DiskNodeIndices::new(&self.node_slots)
    }

    // ====================================================================
    // Edge methods
    // ====================================================================

    /// Materialize an EdgeData into the arena. Reads conn_type from EdgeEndpoints
    /// (O(1) lookup) and properties from edge_properties HashMap.
    #[inline]
    pub(crate) fn materialize_edge(&self, edge_idx: u32) -> &EdgeData {
        let ep = self.edge_endpoints.get(edge_idx as usize);
        let ct = InternedKey::from_u64(ep.connection_type);
        let props = if self.edge_properties.is_empty() {
            Vec::new()
        } else {
            self.edge_properties
                .get(edge_idx)
                .map(|cow| cow.into_owned())
                .unwrap_or_default()
        };
        let boxed = Box::new(EdgeData {
            connection_type: ct,
            properties: props,
        });
        // SAFETY: Box allocates on the heap — the pointer is stable even when
        // the Vec grows. The arena is never cleared while &self borrows are alive
        // (clear_arenas requires &mut self).
        let ptr = &*boxed as *const EdgeData;
        let mut arena = self.edge_arena.lock().unwrap();
        arena.push(boxed);
        // SAFETY: `boxed: Box<EdgeData>` gives a stable heap pointer; the
        // arena keeps it alive until `clear_arenas` (`&mut self`); our
        // `&self` borrow prevents that. See block comment above.
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
        if idx >= offsets.len().saturating_sub(1) {
            return Ok(0);
        }
        let mut start = offsets.get(idx) as usize;
        let mut end = offsets.get(idx + 1) as usize;

        // Narrow range via binary search when CSR is sorted by type
        if let Some(ct) = conn_type {
            if self.csr_sorted_by_type {
                let (lo, hi) = crate::graph::core::iterators::binary_search_conn_type(
                    edges,
                    &self.edge_endpoints,
                    start,
                    end,
                    ct,
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
                        if self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct {
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
            // Check connection type (only needed if CSR is NOT sorted)
            if let Some(ct) = conn_type {
                if !self.csr_sorted_by_type
                    && self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct
                {
                    continue;
                }
            }
            // Check peer node type (O(1) mmap read, no materialization)
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

        // Count overflow edges too
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
                    if self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct {
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

    /// Iterate peer node indices for edges of a specific type, without materializing
    /// EdgeData. Yields (peer_idx, edge_idx) pairs. With sorted CSR, uses binary
    /// search. Completely avoids reading edge_endpoints.bin (13 GB) — only touches
    /// out_edges.bin/in_edges.bin + node_slots.bin.
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
        if idx >= offsets.len().saturating_sub(1) {
            return Vec::new();
        }
        let mut start = offsets.get(idx) as usize;
        let mut end = offsets.get(idx + 1) as usize;

        // Narrow range via binary search when CSR is sorted
        if let Some(ct) = conn_type {
            if self.csr_sorted_by_type {
                let (lo, hi) = crate::graph::core::iterators::binary_search_conn_type(
                    edges,
                    &self.edge_endpoints,
                    start,
                    end,
                    ct,
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
            // When CSR is NOT sorted, must check type via edge_endpoints
            if let Some(ct) = conn_type {
                if !self.csr_sorted_by_type
                    && self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct
                {
                    continue;
                }
            }
            result.push((NodeIndex::new(e.peer as usize), e.edge_idx));
        }

        // Include overflow edges
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
                    if self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct {
                        continue;
                    }
                }
                result.push((NodeIndex::new(e.peer as usize), e.edge_idx));
            }
        }
        result
    }

    /// Advise the kernel to prefetch hot mmap regions into page cache.
    /// Called after load to warm offset arrays and node_slots, reducing
    /// cold-cache penalty on first queries. Non-blocking — the kernel
    /// reads pages asynchronously in the background.
    /// Count all edges of a connection type, grouped by peer (target for outgoing,
    /// source for incoming). Returns a HashMap<peer_node_idx, count>.
    /// Uses a single sequential scan of edge_endpoints — O(E) total, purely sequential
    /// I/O (no random access). For outgoing grouping: counts by target. For incoming: by source.
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

        // Sequential scan of edge_endpoints — each entry is (source, target, conn_type).
        // 16 bytes per edge, purely sequential. Deadline check every 1M entries
        // keeps the per-check cost <0.001% while bounding wall-clock overshoot to ~0.3s.
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
            let ep = self.edge_endpoints.get(i);
            if ep.source == TOMBSTONE_EDGE {
                continue;
            }
            if ep.connection_type != conn_type {
                continue;
            }
            let peer = match dir {
                Direction::Outgoing => ep.target, // group by target
                Direction::Incoming => ep.source, // group by source
            };
            *counts.entry(peer).or_insert(0) += 1;
        }

        // Release page cache pages after scan to reduce memory pressure.
        self.edge_endpoints.advise_dontneed();

        Ok(counts)
    }

    /// Look up source nodes that have outgoing edges of the given connection type.
    /// Returns an iterator-like slice of source node IDs from the inverted index.
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

        // Read from persisted inverted index (binary search)
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

        // Supplement with overflow sources — check each overflow node for matching edges.
        // Overflow is almost always small; we don't apply `max` here because it'd require
        // a second dedup pass and complicate the contract. Callers that use `max` on cold
        // reads will rarely hit overflow anyway.
        if !self.overflow_out.is_empty() {
            for (&node_id, edges) in &self.overflow_out {
                for e in edges {
                    if e.edge_idx != TOMBSTONE_EDGE {
                        let ep = self.edge_endpoints.get(e.edge_idx as usize);
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
    /// per match. Never calls `materialize_edge` — no growth of `edge_arena`.
    ///
    /// Path: persisted inverted index (`conn_type_index_*`) gives the sources with
    /// at least one outgoing edge of that type. Each source's outgoing CSR slice
    /// is then filtered by `conn_type` (binary-search when the CSR is sorted by
    /// type, linear fallback otherwise). Overflow-out entries are visited for
    /// sources added after the last CSR build.
    ///
    /// The callback returns `true` to continue, `false` to stop iteration —
    /// lets callers collect a bounded prefix (e.g. two sample edges) without
    /// scanning every match.
    ///
    /// Complexity is O(matching edges) when `csr_sorted_by_type`, not O(all edges).
    /// Designed for the introspection fast path (`describe(connections=['T'])`)
    /// which previously did three full `edge_references()` sweeps per topic.
    pub fn for_each_edge_of_conn_type<F>(&self, conn_type: u64, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, u32) -> bool,
    {
        self.ensure_csr();

        // CSR-indexed sources via the inverted index.
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
                            &self.edge_endpoints,
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
                            let ep = self.edge_endpoints.get(e.edge_idx as usize);
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
                let ep = self.edge_endpoints.get(e.edge_idx as usize);
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
    /// Safe to call in hot loops — does not push into `edge_arena`.
    ///
    /// The returned `Cow` is `Borrowed` for overlay hits (zero copy) and
    /// `Owned` for columnar-base hits (one bincode deserialize). Callers
    /// that need `&[(InternedKey, Value)]` can use `.as_deref()`.
    #[inline]
    pub fn edge_properties_at(&self, edge_idx: u32) -> Option<Cow<'_, [(InternedKey, Value)]>> {
        self.edge_properties.get(edge_idx)
    }

    /// Edge-centric sweep: scan `edge_endpoints` linearly and invoke `f`
    /// for every edge whose `connection_type` matches. Return `false`
    /// from the callback to stop early.
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
            let ep = self.edge_endpoints.get(edge_idx);
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

    pub fn prefetch_hot_regions(&self) {
        // Prefetch out_offsets + in_offsets (948 MB each — always needed for traversal).
        // Skip node_slots (2 GB) — prefetching it adds too much load latency.
        // The kernel will page in node_slots on demand during queries.
        self.out_offsets.advise_willneed();
        self.in_offsets.advise_willneed();
    }

    /// Auto-build CSR from pending edges if needed. Called from &self query methods.
    /// Check if pending edges need to be built into CSR.
    /// Panics with a helpful message if called with unbuilt edges.
    #[inline]
    fn ensure_csr(&self) {
        // No-op check — pending edges should be empty after build_csr_from_pending.
        // If not empty, queries may miss some edges (but won't crash).
    }

    /// Clear all materialization arenas. Called before any &mut self operation.
    #[inline]
    pub(crate) fn clear_arenas(&mut self) {
        // Flush modified edge properties from edge_weight_mut cache
        for (edge_idx, edge_data) in self.edge_mut_cache.drain() {
            if edge_data.properties.is_empty() {
                self.edge_properties.remove(edge_idx);
            } else {
                self.edge_properties.insert(edge_idx, edge_data.properties);
            }
        }
        // Flush staged node writes via clone-apply-replace on the
        // affected ColumnStores. Group by type_key so each store is
        // cloned at most once per flush, regardless of how many rows
        // were mutated.
        self.flush_node_mut_cache();
        self.node_arena.lock().unwrap().clear();
        self.edge_arena.lock().unwrap().clear();
    }

    /// Drain `node_mut_cache` and apply the staged writes to
    /// `self.column_stores` using full-`Arc` replacement:
    ///
    ///   let mut new_store = (**current_arc).clone();  // one deep clone
    ///   new_store.set_title(row_id, &title);
    ///   new_store.set(row_id, key, &value, None);     // per staged mutation
    ///   self.column_stores.insert(type_key, Arc::new(new_store));
    ///
    /// Dead slots (tombstoned via `remove_node`) flush a
    /// `store.tombstone(row_id)` instead. This is the node analogue of
    /// `batch.rs::flush_chunk`'s deferred-columnar pass — the only
    /// pattern that's proven to survive the Arc sharing between
    /// DirGraph and DiskGraph.
    fn flush_node_mut_cache(&mut self) {
        if self.node_mut_cache.is_empty() {
            return;
        }
        use crate::graph::schema::PropertyStorage;
        // Group drained entries by type_key.
        let drained: Vec<(u32, NodeData)> = self.node_mut_cache.drain().collect();
        let mut by_type: HashMap<InternedKey, Vec<(u32, NodeData)>> = HashMap::new();
        for (i, nd) in drained {
            if (i as usize) >= self.node_slots.len() {
                continue;
            }
            let slot = self.node_slots.get(i as usize);
            let type_key = InternedKey::from_u64(slot.node_type);
            by_type.entry(type_key).or_default().push((i, nd));
        }
        for (type_key, updates) in by_type {
            let Some(current_arc) = self.column_stores.get(&type_key) else {
                continue;
            };
            // Check whether any entry would actually write anything
            // before paying the clone + Arc-replace cost. Entries from
            // `batch.rs::flush_chunk` leave behind `PropertyStorage::
            // Columnar` scratch in the cache (batch persists via its
            // own full-Arc replacement); those yield nothing to flush.
            // Dead slots still need a tombstone so they contribute
            // work. Entries with Map-form properties or a non-null
            // title also contribute.
            let any_writes_needed = updates.iter().any(|(i, nd)| {
                let slot = self.node_slots.get(*i as usize);
                if !slot.is_alive() {
                    return true;
                }
                if let PropertyStorage::Map(map) = &nd.properties {
                    if !map.is_empty() {
                        return true;
                    }
                }
                // Title write only counts when it differs from the
                // current store — otherwise `Str::set`'s offset update
                // would corrupt neighbouring rows and we'd skip it
                // anyway below.
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
                let slot = self.node_slots.get(i as usize);
                let row_id = slot.row_id;
                if !slot.is_alive() {
                    // Tombstoned by `remove_node` — mark the row dead
                    // in the ColumnStore so reloads skip it.
                    new_store.tombstone(row_id);
                    continue;
                }
                // Title is in its own column. Only write through the
                // cached value when it actually differs from what's
                // already in the store — `TypedColumn::Str::set`
                // corrupts offsets on same-row overwrite (it only
                // updates offsets[idx]/offsets[idx+1] rather than
                // shifting the tail), so a naive always-write would
                // break every other row's title on reload. The
                // deep-clone's title column still has the original
                // string; skip the write when cached title == stored
                // title to avoid corrupting the column.
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
            // Replace the Arc wholesale. DirGraph's Arc is re-synced
            // from DiskGraph's post-save via
            // `DirGraph::sync_column_stores_from_disk`.
            self.column_stores
                .insert(type_key, std::sync::Arc::new(new_store));
        }
    }

    /// Reset materialization arenas between queries to prevent unbounded growth.
    /// Only call when no references from prior `node_weight()` /
    /// `materialize_edge()` calls are alive — i.e. between top-level queries.
    pub fn reset_arenas(&self) {
        // The doc contract (no live materialization refs) is satisfied
        // by KGLite's single-threaded query loop calling this between
        // top-level queries. The two Mutex'd arenas are independent —
        // edge_arena uses the same pattern.
        self.node_arena.lock().unwrap().clear();
        self.edge_arena.lock().unwrap().clear();
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
        DiskEdgeIndices::new(self.next_edge_idx, &self.edge_endpoints)
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
        let ep = self.edge_endpoints.get(ei);
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
        let ep = self.edge_endpoints.get(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }
        self.metadata_dirty = true;
        // Store in dedicated cache (not the arena) so we can flush correctly.
        // The arena is append-only and shared with edge_weight (read-only),
        // making offset tracking fragile. The cache is keyed by edge_idx.
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
        let ep = self.edge_endpoints.get(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }
        Some((
            NodeIndex::new(ep.source as usize),
            NodeIndex::new(ep.target as usize),
        ))
    }

    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        self.clear_arenas();
        self.metadata_dirty = true;
        let edge_idx = self.next_edge_idx;
        self.next_edge_idx += 1;

        let ct = data.connection_type;

        if !data.properties.is_empty() {
            self.edge_properties.insert(edge_idx, data.properties);
        }

        let src = a.index() as u32;
        let tgt = b.index() as u32;
        let ct_u64 = ct.as_u64();

        if self.defer_csr {
            // Bulk-build mode: accumulate in pending_edges; caller is
            // responsible for `build_csr_from_pending()` at batch end.
            // Set explicitly by bulk loaders (ntriples); off by default so
            // individual Cypher mutations stay visible via overflow.
            self.pending_edges.get_mut().push((src, tgt, ct_u64));
        } else {
            // Post-CSR mode: go directly to overflow + edge_endpoints.
            // This makes new edges immediately visible to queries via
            // the DiskEdges iterator which merges CSR + overflow.
            self.edge_endpoints.push(EdgeEndpoints {
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
        }

        self.edge_count += 1;
        EdgeIndex::new(edge_idx as usize)
    }

    pub fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.clear_arenas();
        self.metadata_dirty = true;
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoints.get(ei);
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

        // Tombstone in outgoing CSR
        Self::tombstone_in_array(&self.out_offsets, &mut self.out_edges, src, ei32);
        // Tombstone in incoming CSR
        Self::tombstone_in_array(&self.in_offsets, &mut self.in_edges, tgt, ei32);

        // Tombstone in overflow lists
        if let Some(list) = self.overflow_out.get_mut(&(src as u32)) {
            list.retain(|e| e.edge_idx != ei32);
        }
        if let Some(list) = self.overflow_in.get_mut(&(tgt as u32)) {
            list.retain(|e| e.edge_idx != ei32);
        }

        // Tombstone in endpoints
        self.edge_endpoints.set(
            ei,
            EdgeEndpoints {
                source: TOMBSTONE_EDGE,
                target: TOMBSTONE_EDGE,
                connection_type: 0,
            },
        );

        self.edge_count -= 1;
        self.free_edge_slots.push(ei32);
        self.has_tombstones = true;
        Some(result)
    }

    pub fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        self.ensure_csr();
        let src = a.index();
        let tgt = b.index() as u32;

        // Search CSR outgoing edges from a
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

        // Search overflow
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
            let ep = self.edge_endpoints.get(i as usize);
            if ep.source == TOMBSTONE_EDGE {
                return None;
            }

            Some(self.materialize_edge(i))
        }))
    }

    // ====================================================================
    // Neighbor methods
    // ====================================================================

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

        if node >= offsets.len().saturating_sub(1) {
            return DiskNeighbors::new_empty();
        }

        let start = offsets.get(node) as usize;
        let end = offsets.get(node + 1) as usize;

        DiskNeighbors::new(edges, start, end, overflow)
    }

    pub fn neighbors_undirected_iter(&self, a: NodeIndex) -> DiskNeighbors {
        self.ensure_csr();
        // Collect both outgoing and incoming neighbors
        let node = a.index();
        let mut peers = Vec::new();

        // Outgoing
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

        // Incoming
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

    // ====================================================================
    // CSR construction from pending edges
    // ====================================================================

    /// Build CSR arrays from the pending edges log. Called lazily on first
    /// query, or explicitly on save. Uses external merge sort — all I/O is
    /// sequential, designed for larger-than-RAM graphs.
    /// True if any overflow edges are present (edges added after the initial
    /// CSR build). Used by `ensure_disk_edges_built` to decide whether to
    /// merge overflow back into CSR so downstream indexes stay consistent.
    pub fn has_overflow(&self) -> bool {
        self.overflow_out.values().any(|v| !v.is_empty())
            || self.overflow_in.values().any(|v| !v.is_empty())
    }

    pub fn build_csr_from_pending(&mut self) {
        let pending = self.pending_edges.get_mut();
        if pending.is_empty() {
            return;
        }

        let node_bound = self.node_slots.len();
        let edge_count = pending.len();
        let verbose = std::env::var("KGLITE_BUILD_DEBUG").is_ok();
        let use_merge_sort = std::env::var("KGLITE_CSR_ALGO").is_ok_and(|v| v == "merge_sort");
        if use_merge_sort {
            self.build_csr_merge_sort(node_bound, edge_count, verbose);
        } else {
            self.build_csr_partitioned(node_bound, edge_count, verbose);
        }
        // After CSR is built, subsequent add_edge calls should route to overflow
        self.defer_csr = false;
    }

    /// Merge overflow edges back into the CSR arrays via full rebuild.
    /// Collects all live edges (CSR + overflow, excluding tombstones),
    /// writes to pending_edges, clears overflow, and rebuilds CSR.
    /// Returns the number of overflow edges that were merged.
    pub fn compact(&mut self) -> usize {
        let overflow_count: usize = self.overflow_out.values().map(|v| v.len()).sum();
        if overflow_count == 0 {
            return 0;
        }

        let verbose = std::env::var("KGLITE_BUILD_DEBUG").is_ok();
        if verbose {
            eprintln!(
                "Compacting: {} CSR edges + {} overflow edges",
                self.edge_count.saturating_sub(overflow_count),
                overflow_count
            );
        }

        let node_bound = self.node_slots.len();

        // Collect all live edges into a fresh pending_edges buffer.
        // Source: edge_endpoints (covers both CSR and post-CSR overflow edges).
        // Skip tombstoned entries.
        let mut live_count = 0usize;
        let total_endpoints = self.next_edge_idx as usize;

        let pending_path = self.data_dir.join("_compact_pending.bin");
        let mut new_pending: MmapOrVec<(u32, u32, u64)> =
            MmapOrVec::mapped(&pending_path, total_endpoints)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(total_endpoints));

        // Edge index remapping: old_idx → new_idx
        // Needed because compaction produces a dense edge array.
        let mut idx_remap: Vec<u32> = vec![TOMBSTONE_EDGE; total_endpoints];

        for (old_idx, remap_slot) in idx_remap.iter_mut().enumerate().take(total_endpoints) {
            let ep = self.edge_endpoints.get(old_idx);
            if ep.source != TOMBSTONE_EDGE
                && (ep.source as usize) < node_bound
                && (ep.target as usize) < node_bound
            {
                *remap_slot = live_count as u32;
                new_pending.push((ep.source, ep.target, ep.connection_type));
                live_count += 1;
            }
        }

        // Remap edge properties to new indices.
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

        // Clear overflow and free slots
        self.overflow_out.clear();
        self.overflow_in.clear();
        self.free_edge_slots.clear();

        // Reset edge tracking
        self.edge_count = live_count;
        self.next_edge_idx = live_count as u32;

        // Replace pending_edges and rebuild CSR
        let old_pending_path = self
            .pending_edges
            .get_mut()
            .file_path()
            .map(|p| p.to_path_buf());
        *self.pending_edges.get_mut() = new_pending;
        if let Some(path) = old_pending_path {
            let _ = std::fs::remove_file(path);
        }

        self.build_csr_from_pending();

        // Clean up compact temp file
        let _ = std::fs::remove_file(&pending_path);

        if verbose {
            eprintln!(
                "Compaction done: {} live edges (removed {} tombstoned)",
                live_count,
                total_endpoints - live_count
            );
        }

        overflow_count
    }

    /// [DEV] External merge sort variant — zero random reads.
    /// Sorts pending data into chunks, merges sequentially. All I/O is sequential.
    /// Use `KGLITE_CSR_ALGO=merge_sort` to select.
    pub fn lookup_peer_counts(&self, conn_type: u64) -> Option<HashMap<u32, i64>> {
        if self.peer_count_types.is_empty() {
            return None;
        }
        let n = self.peer_count_types.len();
        // Binary search the types array.
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

    // ====================================================================
    // Internal helpers
    // ====================================================================

    fn tombstone_edges_for_node(&mut self, node: usize) {
        // Tombstone outgoing CSR edges
        if node < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(node) as usize;
            let end = self.out_offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = self.out_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE {
                    let ei = e.edge_idx;
                    e.edge_idx = TOMBSTONE_EDGE;
                    self.out_edges.set(i, e);
                    // Tombstone the corresponding incoming edge
                    self.tombstone_in_edge_for(e.peer as usize, ei);
                    // Tombstone endpoints
                    self.edge_endpoints.set(
                        ei as usize,
                        EdgeEndpoints {
                            source: TOMBSTONE_EDGE,
                            target: TOMBSTONE_EDGE,
                            connection_type: 0,
                        },
                    );
                    self.edge_properties.remove(ei);
                    self.edge_count -= 1;
                    self.free_edge_slots.push(ei);
                }
            }
        }

        // Tombstone incoming CSR edges
        if node < self.in_offsets.len().saturating_sub(1) {
            let start = self.in_offsets.get(node) as usize;
            let end = self.in_offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = self.in_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE {
                    let ei = e.edge_idx;
                    e.edge_idx = TOMBSTONE_EDGE;
                    self.in_edges.set(i, e);
                    self.tombstone_out_edge_for(e.peer as usize, ei);
                    self.edge_endpoints.set(
                        ei as usize,
                        EdgeEndpoints {
                            source: TOMBSTONE_EDGE,
                            target: TOMBSTONE_EDGE,
                            connection_type: 0,
                        },
                    );
                    self.edge_properties.remove(ei);
                    self.edge_count -= 1;
                    self.free_edge_slots.push(ei);
                }
            }
        }

        // Tombstone overflow edges
        if let Some(list) = self.overflow_out.remove(&(node as u32)) {
            for e in &list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    self.tombstone_in_edge_for(e.peer as usize, e.edge_idx);
                    self.edge_endpoints.set(
                        e.edge_idx as usize,
                        EdgeEndpoints {
                            source: TOMBSTONE_EDGE,
                            target: TOMBSTONE_EDGE,
                            connection_type: 0,
                        },
                    );
                    self.edge_properties.remove(e.edge_idx);
                    self.edge_count -= 1;
                    self.free_edge_slots.push(e.edge_idx);
                }
            }
        }
        if let Some(list) = self.overflow_in.remove(&(node as u32)) {
            for e in &list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    self.tombstone_out_edge_for(e.peer as usize, e.edge_idx);
                    self.edge_endpoints.set(
                        e.edge_idx as usize,
                        EdgeEndpoints {
                            source: TOMBSTONE_EDGE,
                            target: TOMBSTONE_EDGE,
                            connection_type: 0,
                        },
                    );
                    self.edge_properties.remove(e.edge_idx);
                    self.edge_count -= 1;
                    self.free_edge_slots.push(e.edge_idx);
                }
            }
        }
    }

    fn tombstone_in_edge_for(&mut self, node: usize, edge_idx: u32) {
        if node < self.in_offsets.len().saturating_sub(1) {
            let start = self.in_offsets.get(node) as usize;
            let end = self.in_offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = self.in_edges.get(i);
                if e.edge_idx == edge_idx {
                    e.edge_idx = TOMBSTONE_EDGE;
                    self.in_edges.set(i, e);
                    return;
                }
            }
        }
        if let Some(list) = self.overflow_in.get_mut(&(node as u32)) {
            list.retain(|e| e.edge_idx != edge_idx);
        }
    }

    fn tombstone_out_edge_for(&mut self, node: usize, edge_idx: u32) {
        if node < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(node) as usize;
            let end = self.out_offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = self.out_edges.get(i);
                if e.edge_idx == edge_idx {
                    e.edge_idx = TOMBSTONE_EDGE;
                    self.out_edges.set(i, e);
                    return;
                }
            }
        }
        if let Some(list) = self.overflow_out.get_mut(&(node as u32)) {
            list.retain(|e| e.edge_idx != edge_idx);
        }
    }

    fn tombstone_in_array(
        offsets: &MmapOrVec<u64>,
        edges: &mut MmapOrVec<CsrEdge>,
        node: usize,
        edge_idx: u32,
    ) {
        if node < offsets.len().saturating_sub(1) {
            let start = offsets.get(node) as usize;
            let end = offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = edges.get(i);
                if e.edge_idx == edge_idx {
                    e.edge_idx = TOMBSTONE_EDGE;
                    edges.set(i, e);
                    return;
                }
            }
        }
    }
}

impl Clone for DiskGraph {
    fn clone(&self) -> Self {
        // Clone into heap copies (clone is expensive for disk graphs)
        let mut node_slots = MmapOrVec::with_capacity(self.node_slots.len());
        for i in 0..self.node_slots.len() {
            node_slots.push(self.node_slots.get(i));
        }
        let mut out_offsets = MmapOrVec::with_capacity(self.out_offsets.len());
        for i in 0..self.out_offsets.len() {
            out_offsets.push(self.out_offsets.get(i));
        }
        let mut out_edges = MmapOrVec::with_capacity(self.out_edges.len());
        for i in 0..self.out_edges.len() {
            out_edges.push(self.out_edges.get(i));
        }
        let mut in_offsets = MmapOrVec::with_capacity(self.in_offsets.len());
        for i in 0..self.in_offsets.len() {
            in_offsets.push(self.in_offsets.get(i));
        }
        let mut in_edges = MmapOrVec::with_capacity(self.in_edges.len());
        for i in 0..self.in_edges.len() {
            in_edges.push(self.in_edges.get(i));
        }
        let mut edge_endpoints = MmapOrVec::with_capacity(self.edge_endpoints.len());
        for i in 0..self.edge_endpoints.len() {
            edge_endpoints.push(self.edge_endpoints.get(i));
        }

        DiskGraph {
            node_slots,
            node_count: self.node_count,
            free_node_slots: self.free_node_slots.clone(),
            node_arena: std::sync::Mutex::new(Vec::new()),
            column_stores: self.column_stores.clone(),
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints,
            edge_count: self.edge_count,
            next_edge_idx: self.next_edge_idx,
            edge_properties: self.edge_properties.deep_clone(),
            edge_arena: std::sync::Mutex::new(Vec::new()),
            edge_mut_cache: HashMap::new(),
            node_mut_cache: HashMap::new(),
            pending_edges: UnsafeCell::new(MmapOrVec::new()),
            overflow_out: self.overflow_out.clone(),
            overflow_in: self.overflow_in.clone(),
            free_edge_slots: self.free_edge_slots.clone(),
            data_dir: self.data_dir.clone(),
            metadata_dirty: false,
            csr_sorted_by_type: self.csr_sorted_by_type,
            defer_csr: false,
            edge_type_counts_raw: None,
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            has_tombstones: self.has_tombstones,
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            segment_manifest: self.segment_manifest.clone(),
            sealed_nodes_bound: self.sealed_nodes_bound,
        }
    }
}

impl Drop for DiskGraph {
    fn drop(&mut self) {
        // Flush metadata only if mutations happened since last write
        if self.metadata_dirty {
            let _ = self.write_metadata();
        }
    }
}

// ============================================================================
// Persistent property indexes
// ============================================================================

// Index implementations
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

// ============================================================================
// Disk persistence — the directory IS the saved graph
// ============================================================================

/// Metadata stored alongside the binary files in the disk graph directory.
#[cfg(test)]
#[path = "graph_tests.rs"]
mod tests;
