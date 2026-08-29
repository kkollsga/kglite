//! Disk-to-disk streaming subgraph filter.
//!
//! Gated to disk-backed sources: in-memory and mapped graphs route through
//! the non-streaming `save_subset` path per CLAUDE.md ("in-memory wins every
//! time"). All I/O here is sequential — no per-node random edge lookups.

use crate::graph::schema::{CowSelection, DirGraph, InternedKey};
use crate::graph::storage::disk::csr::{PendingEdge, TOMBSTONE_EDGE};
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use crate::graph::storage::property_storage::ColumnarRow;
use std::path::Path;

/// Edge-type filter for the disk Pass A scans ([`pass_a_scan`],
/// [`pass_a_scan_to_file`]). Node-selection-driven saves do not take one —
/// the selection carries the filter there.
#[derive(Clone, Debug, Default)]
pub struct SubsetSpec {
    /// Restrict to edges of these types. `None` means all edge types.
    pub edge_types: Option<Vec<InternedKey>>,
}

/// Stats reported back from a Pass A scan.
#[derive(Clone, Debug, Default)]
pub struct ScanStats {
    pub kept_node_count: u64,
    pub kept_edge_count: u64,
    pub total_edge_count: u64,
    pub scan_duration_secs: f64,
}

/// Compact bitset over node ids. `Vec<u64>` blocks; one bit per source
/// node id. [`RankIndex`] adds a popcount prefix array for O(1)
/// old→new id translation.
#[derive(Clone, Debug)]
pub struct Bitset {
    blocks: Vec<u64>,
    len: usize,
}

impl Bitset {
    pub fn with_len(len: usize) -> Self {
        let n_blocks = len.div_ceil(64);
        Self {
            blocks: vec![0u64; n_blocks],
            len,
        }
    }

    /// Out-of-range writes are silently ignored: Pass A keeps ids within
    /// `node_bound`, but disk graphs surface tombstone-adjacent ids.
    #[inline]
    pub fn set(&mut self, i: usize) {
        if i < self.len {
            self.blocks[i / 64] |= 1u64 << (i % 64);
        }
    }

    /// Read bit `i`. Out-of-range reads return false.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        (self.blocks[i / 64] >> (i % 64)) & 1 == 1
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn count_ones(&self) -> u64 {
        self.blocks.iter().map(|b| b.count_ones() as u64).sum()
    }

    /// Raw block view — lets [`RankIndex`] build prefixes without re-scanning.
    #[inline]
    pub(crate) fn blocks(&self) -> &[u64] {
        &self.blocks
    }
}

// ── Rank-1 over the kept-nodes bitset ──────────────────────────────────────
//
// A popcount-prefix array makes old→new translation O(1) and entirely in
// RAM: 15 MB bitset + 15 MB prefix = 30 MB at Wikidata scale (120M nodes),
// against 480 MB for a dense remap. The new id space is contiguous
// `[0..kept_count)` ordered by source id — matching the sequential walk of
// `node_slots.bin` in Pass B.

/// O(1) translator from source node ids to dense destination ids. Built
/// once after Pass A; consumed by node materialization and edge translation.
#[derive(Debug, Clone)]
pub struct RankIndex {
    bitset: Bitset,
    /// `block_prefix[k]` = popcount of `bitset.blocks()[0..k]`: the rank of
    /// bit `k * 64` exclusive of itself. Length = `blocks().len() + 1`.
    block_prefix: Vec<u32>,
    kept_count: u32,
}

impl RankIndex {
    /// Single linear pass over the bitset's blocks; O(n_blocks).
    pub fn from_bitset(bitset: Bitset) -> Self {
        let blocks = bitset.blocks();
        let mut block_prefix: Vec<u32> = Vec::with_capacity(blocks.len() + 1);
        let mut acc: u32 = 0;
        block_prefix.push(0);
        for blk in blocks {
            acc = acc.saturating_add(blk.count_ones());
            block_prefix.push(acc);
        }
        Self {
            bitset,
            block_prefix,
            kept_count: acc,
        }
    }

    /// Total kept nodes — also the size of the destination id space.
    #[inline]
    pub fn kept_count(&self) -> u32 {
        self.kept_count
    }

    #[inline]
    pub fn contains(&self, old_id: u32) -> bool {
        self.bitset.get(old_id as usize)
    }

    /// Map a source node id to its destination id in O(1). Returns `None`
    /// when `old_id` is not in the kept set or is out of range.
    #[inline]
    pub fn old_to_new(&self, old_id: u32) -> Option<u32> {
        let len = self.bitset.len();
        let i = old_id as usize;
        if i >= len {
            return None;
        }
        let block_idx = i / 64;
        let bit = i % 64;
        let block = self.bitset.blocks()[block_idx];
        if (block >> bit) & 1 == 0 {
            return None;
        }
        // Mask of bits before `bit` in the same block.
        let mask: u64 = if bit == 0 { 0 } else { (1u64 << bit) - 1 };
        let within_block = (block & mask).count_ones();
        Some(self.block_prefix[block_idx] + within_block)
    }
}

pub struct PassAResult {
    /// Source node ids that survive the filter: the endpoints of every kept
    /// edge. The writer consumes this to build the rank-1 index.
    pub kept_nodes: Bitset,
    pub stats: ScanStats,
}

/// Pass A: sequential scan of `edge_endpoints.bin` with an edge-type
/// filter, building a kept-nodes bitset.
///
/// Memory: the `Bitset` is `n_nodes / 8` bytes (~15 MB at 120M Wikidata
/// nodes); nothing else scales with graph size.
///
/// I/O: one sequential mmap'd read of `edge_endpoints` (16 B per edge).
/// Tombstoned edges (TOMBSTONE_EDGE in `source`) are skipped without
/// touching property storage.
pub fn pass_a_scan(source: &DiskGraph, spec: &SubsetSpec) -> PassAResult {
    let n_nodes = source.node_slot_len();
    let mut kept_nodes = Bitset::with_len(n_nodes);

    // Key the filter on the raw u64 hash so it matches
    // `EdgeEndpoints.connection_type` without re-interning per edge.
    let edge_type_set: Option<std::collections::HashSet<u64>> = spec
        .edge_types
        .as_ref()
        .map(|v| v.iter().map(|k| k.as_u64()).collect());

    source.edge_endpoints.advise_sequential();

    let scan_start = std::time::Instant::now();
    let n_edges = source.next_edge_idx as usize;
    let mut kept_edge_count: u64 = 0;
    for edge_idx in 0..n_edges {
        let ep = source.edge_endpoint(edge_idx);
        if ep.source == TOMBSTONE_EDGE {
            continue;
        }
        if let Some(ref types) = edge_type_set {
            if !types.contains(&ep.connection_type) {
                continue;
            }
        }
        kept_nodes.set(ep.source as usize);
        kept_nodes.set(ep.target as usize);
        kept_edge_count += 1;
    }

    let kept_node_count = kept_nodes.count_ones();

    // Drop the source's edge_endpoints page cache now the sweep is done: on
    // Wikidata that is 9 GB of mmap pages that would dominate RSS for the
    // rest of the pipeline. `advise_sequential` is only a hint and macOS
    // does not honor it aggressively; DONTNEED forces eviction.
    source.edge_endpoints.advise_dontneed();

    PassAResult {
        kept_nodes,
        stats: ScanStats {
            kept_node_count,
            kept_edge_count,
            total_edge_count: n_edges as u64,
            scan_duration_secs: scan_start.elapsed().as_secs_f64(),
        },
    }
}

// ── Pass A with file output ────────────────────────────────────────────────
//
// The temp record shape matches the existing CSR builder's input
// (`csr_build::build_csr_files` consumes `&MmapOrVec<PendingEdge>`), so a
// consumer drives the merge sort over this file by translating `(src, tgt)`
// through the rank index in the iterator.
//
// Edge property bytes are NOT inlined — that would need either (a) a sidecar
// file keyed by source edge_idx or (b) a fourth column in the temp record.
// Keeping the builder-compatible shape instead lets properties travel
// independently.
//
// Memory: bitset only (~15 MB at 120M nodes); appended records go straight
// to the mmap'd file. I/O: one sequential read of `edge_endpoints` plus one
// sequential write to `kept_edges_path`.

pub struct PassAFileResult {
    pub kept_nodes: Bitset,
    pub stats: ScanStats,
    pub kept_edges_path: std::path::PathBuf,
    pub kept_edge_records: u64,
}

/// Pass A with file output. Identical semantics to [`pass_a_scan`] but also
/// appends `(src, tgt, conn_type)` per kept edge to `kept_edges_path`. The
/// file is sized for the total edge count up front (a safe upper bound on
/// kept edges); the consumer reads the actual count from `kept_edge_records`.
pub fn pass_a_scan_to_file(
    source: &DiskGraph,
    spec: &SubsetSpec,
    kept_edges_path: &Path,
) -> Result<PassAFileResult, String> {
    let n_nodes = source.node_slot_len();
    let n_edges = source.next_edge_idx as usize;
    let mut kept_nodes = Bitset::with_len(n_nodes);

    let edge_type_set: Option<std::collections::HashSet<u64>> = spec
        .edge_types
        .as_ref()
        .map(|v| v.iter().map(|k| k.as_u64()).collect());

    if let Some(parent) = kept_edges_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "save_subset: failed to create temp dir {}: {}",
                    parent.display(),
                    e
                )
            })?;
        }
    }

    // No heap fallback: `with_capacity(n_edges)` would allocate 16 B × the
    // source's *total* edge count — 2 GB at Wikidata scale — in the one path
    // whose stated memory bound is "bitset only". A path we cannot map is a
    // failed scan, not a quietly more expensive one.
    let mut kept_edges: MmapOrVec<PendingEdge> = MmapOrVec::mapped(kept_edges_path, n_edges)
        .map_err(|error| {
            format!(
                "save_subset: failed to create the kept edge buffer at {}: {}",
                kept_edges_path.display(),
                error
            )
        })?;

    source.edge_endpoints.advise_sequential();
    let scan_start = std::time::Instant::now();
    let mut kept_edge_count: u64 = 0;
    for edge_idx in 0..n_edges {
        let ep = source.edge_endpoint(edge_idx);
        if ep.source == TOMBSTONE_EDGE {
            continue;
        }
        if let Some(ref types) = edge_type_set {
            if !types.contains(&ep.connection_type) {
                continue;
            }
        }
        kept_nodes.set(ep.source as usize);
        kept_nodes.set(ep.target as usize);
        kept_edges
            .try_push(PendingEdge {
                source: ep.source,
                target: ep.target,
                connection_type: ep.connection_type,
            })
            .map_err(|error| format!("save_subset: append kept edge: {error}"))?;
        kept_edge_count += 1;
    }

    let kept_node_count = kept_nodes.count_ones();
    let scan_duration_secs = scan_start.elapsed().as_secs_f64();

    Ok(PassAFileResult {
        kept_nodes,
        stats: ScanStats {
            kept_node_count,
            kept_edge_count,
            total_edge_count: n_edges as u64,
            scan_duration_secs,
        },
        kept_edges_path: kept_edges_path.to_path_buf(),
        kept_edge_records: kept_edge_count,
    })
}

// ── Streaming disk-to-disk pipeline ───────────────────────────────────────
//
// Eliminates the in-memory petgraph step that drove the in-memory baseline
// to 7.1 GB peak RSS on Wikidata Articles+P50+Authors: the destination is a
// disk-mode `DirGraph` from the start, so column rows, pending edges and CSR
// input all land in file-backed storage. The numbered steps in
// `save_subset_streaming_disk` below carry the detail.
//
// Memory at Wikidata Articles+P50+Authors scale (17.4M kept nodes): ~70 MB of
// sorted per-type kept-id vectors, ~30 MB rank index, and file-backed pending
// edges at ~16 B each. Every phase is sequential: one pass over source
// `edge_endpoints.bin`, one source column store at a time (kept ids in
// source-id order = monotone row_ids), appends to the dest writers, then the
// existing external merge sort for the CSR.

/// Save a filtered subgraph to disk.
///
/// `selection` defines which nodes are kept — typically from the fluent
/// chain `kg.select(...).expand(...)`. All edges between kept nodes are
/// included.
///
/// The output reloads into either portable mode via `kglite.open(path,
/// storage=...)` or `kglite.load(path, storage=...)`; with no argument both
/// restore the mode the checkpoint recorded. Disk is not among them — a `.kgl`
/// is a file, and a disk graph is a directory.
pub fn save_subset(
    source: &DirGraph,
    selection: &CowSelection,
    out_path: &Path,
) -> Result<(), String> {
    use crate::graph::mutation::subgraph::extract_subgraph;

    // 1. Materialize the filtered subgraph in-memory. `extract_subgraph`
    //    reads through `GraphRead`, so it works for every source mode; for
    //    disk sources the extracted graph holds `Arc` references into the
    //    source's column stores rather than deep-cloning property data.
    let mut extracted = extract_subgraph(source, selection)?;

    // 2. Consolidate properties into self-contained column stores so the
    //    output is independent of the source's stores. Both save paths need it.
    extracted.enable_columnar();

    let path_str = out_path.to_str().ok_or_else(|| {
        format!(
            "save_subset: out_path is not valid UTF-8: {}",
            out_path.display()
        )
    })?;

    // 3. Choose the serializer by size: the portable `.kgl` path
    //    (`write_kgl`) assembles one compressed in-memory payload, while
    //    Wikidata-class extracts (~17 M nodes / 35 M edges) need the bounded
    //    disk-directory path. `kglite.load(path)` auto-detects file vs
    //    directory. The threshold is empirical — the single-file format is
    //    comfortable below ~1 M nodes.
    const SINGLE_FILE_NODE_THRESHOLD: u64 = 1_000_000;

    use crate::graph::storage::GraphRead;
    let node_count = u64::try_from(extracted.graph.node_count()).unwrap_or(u64::MAX);
    if node_count <= SINGLE_FILE_NODE_THRESHOLD {
        let mut arc = std::sync::Arc::new(extracted);
        crate::graph::io::file::prepare_save(&mut arc);
        crate::graph::io::file::write_kgl(&arc, path_str)
            .map_err(|e| format!("save_subset: write_kgl failed: {}", e))
    } else {
        // `enable_disk_mode_at` builds the CSR inside `path_str` and publishes
        // it there, so a large subset never transits the system temp directory
        // on its way to the caller's destination.
        extracted.enable_disk_mode_at(path_str)
    }
}

/// Copy the graph-level metadata a subset needs to be self-contained on
/// reload: the interner and type schemas the rows are encoded against, the
/// alias/tier maps `describe()` and property resolution read, and the caller's
/// user-schema version.
///
/// The version carries because a subset of a graph at user-schema version N
/// is still at version N — only which rows came along changed — so a
/// migration runner pointed at the subset must not re-run migrations `1..=N`.
fn clone_subset_metadata(dest: &mut DirGraph, source: &DirGraph) {
    dest.interner = source.interner.clone();
    dest.type_schemas = source.type_schemas.clone();
    dest.node_type_metadata = source.node_type_metadata.clone();
    dest.connection_type_metadata = source.connection_type_metadata.clone();
    dest.id_field_aliases = source.id_field_aliases.clone();
    dest.title_field_aliases = source.title_field_aliases.clone();
    dest.parent_types = source.parent_types.clone();
    dest.user_schema_version = source.user_schema_version;
}

/// Streaming disk-to-disk subgraph filter.
///
/// `kept_per_type` maps each kept node type to its sorted source node ids
/// (ascending). The caller must guarantee sortedness — Pass A's bitset
/// intersected with `type_indices` already produces sorted output.
///
/// `edge_filter` keeps only edges whose connection type's interned u64
/// hash is in the set. `None` keeps every edge between kept nodes.
///
/// Output: a self-contained disk-mode graph at `out_path`. The caller is
/// responsible for ensuring `out_path` is empty / does not exist —
/// `DiskGraph::new_at_path` creates the directory layout.
pub fn save_subset_streaming_disk(
    source: &DirGraph,
    kept_per_type: &std::collections::HashMap<String, Vec<u32>>,
    edge_filter: Option<&[u64]>,
    out_path: &Path,
) -> Result<(), String> {
    use crate::datatypes::values::Value;
    use crate::graph::schema::{EdgeData, NodeData, PropertyStorage};
    use crate::graph::storage::backend::GraphBackend;
    use crate::graph::storage::column_store::ColumnStore;
    use crate::graph::storage::disk::graph::DiskGraph;
    use crate::graph::storage::interner::InternedKey;
    use petgraph::graph::NodeIndex;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    // Stage timers — opt-in via `KGLITE_STREAMING_TIMING=1` so production
    // stderr stays clean.
    let timing_enabled = std::env::var("KGLITE_STREAMING_TIMING")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    let log_phase = |label: &str, t0: Instant| {
        if timing_enabled {
            eprintln!(
                "save_subset_streaming_disk: {} = {:.3}s",
                label,
                t0.elapsed().as_secs_f64()
            );
        }
    };
    let phase_start = Instant::now();

    let path_str = out_path.to_str().ok_or_else(|| {
        format!(
            "save_subset_streaming_disk: out_path is not valid UTF-8: {}",
            out_path.display()
        )
    })?;

    // 1. Create destination as a disk-mode DirGraph.
    std::fs::create_dir_all(out_path).map_err(|e| {
        format!(
            "save_subset_streaming_disk: create_dir_all({}): {}",
            out_path.display(),
            e
        )
    })?;
    let dest_disk = DiskGraph::new_at_path(out_path)
        .map_err(|e| format!("save_subset_streaming_disk: DiskGraph::new_at_path: {}", e))?;
    let mut dest = DirGraph::from_graph(GraphBackend::Disk(Box::new(dest_disk)));

    clone_subset_metadata(&mut dest, source);

    // Bulk-loader contract: defer CSR build until save_disk so add_edge
    // appends to the file-backed pending_edges instead of going through
    // the slow per-edge overflow path.
    if let GraphBackend::Disk(ref mut dg) = dest.graph {
        dg.defer_csr = true;
    }

    // 2. Build a global rank index over the kept node set: O(1) source→dest
    //    id translation in RAM, new ids assigned in source-id order across
    //    all types. Sized to the source node bound like Pass A's bitset, so
    //    peak together stays at ~30 MB on Wikidata.
    use crate::graph::storage::GraphRead;
    let n_source_nodes = source.graph.node_bound();
    let mut bitset = Bitset::with_len(n_source_nodes);
    for sorted_ids in kept_per_type.values() {
        for &id in sorted_ids {
            bitset.set(id as usize);
        }
    }
    let rank = RankIndex::from_bitset(bitset);

    // 3. For each kept type T, build dest's ColumnStore by walking source
    //    nodes of type T in source-id order — one source store at a time
    //    keeps the OS page cache warm.
    //
    //    A per-type `TypeWriter` streams rows straight to the dest's final
    //    column files: no heap-backed chunk buffer and no merge step, so peak
    //    heap is bounded by `(open_buf_writers × BUF_SIZE) + Mixed-column
    //    heap`. `subgraph_streaming_writer.rs` documents the writer protocol.
    //
    //    A row_id within a store is just the push position; the separate
    //    `old_id → dest NodeIndex` mapping comes from the rank index, so the
    //    edge phase builds the right `NodeIndex`.
    use crate::graph::mutation::subgraph_streaming_writer::TypeWriter;

    let scratch_root = out_path.join(".tmp_streaming");
    std::fs::create_dir_all(&scratch_root).map_err(|e| {
        format!(
            "save_subset_streaming_disk: create_dir_all({}): {}",
            scratch_root.display(),
            e
        )
    })?;

    // One writer per kept type, all open at once: the source walk visits
    // types in arbitrary (source-id-determined) order, so without every
    // writer alive we would either reorder source reads — defeating the
    // sequential pattern — or reopen files per push. macOS
    // `kern.maxfilesperproc` is typically 61440 (verified via sysctl);
    // 4500 types × ~5 files = ~22500 fds, and Linux defaults are as generous.
    let mut writers: HashMap<String, TypeWriter> = HashMap::new();
    for type_name in kept_per_type.keys() {
        let schema = if let Some(src_store) = source.column_store(type_name) {
            Arc::clone(src_store.schema())
        } else if let Some(s) = source.type_schemas.get(type_name) {
            Arc::clone(s)
        } else {
            continue; // type with no schema anywhere — nothing to push
        };
        let meta = source
            .node_type_metadata
            .get(type_name)
            .cloned()
            .unwrap_or_default();
        let writer_dir = scratch_root.join(sanitize_type_name(type_name));

        // Match the source's id/title column types: Wikidata mixes `string`
        // (Q-codes) and `uniqueid` ids across types, so no single variant can
        // be hard-coded. Default to "mixed" when the source has no column
        // store entry — TypeWriter's Mixed buffer handles anything.
        let id_type = source
            .column_store(type_name)
            .and_then(|s| s.id_type_str())
            .unwrap_or("mixed");
        let title_type = source
            .column_store(type_name)
            .and_then(|s| s.title_type_str())
            .unwrap_or("mixed");

        let writer = TypeWriter::new(
            schema,
            meta,
            writer_dir,
            &source.interner,
            id_type,
            title_type,
        )
        .map_err(|e| {
            format!(
                "save_subset_streaming_disk: TypeWriter::new {}: {}",
                type_name, e
            )
        })?;
        writers.insert(type_name.clone(), writer);
    }
    log_phase("setup (rank + writer creation)", phase_start);
    let phase_node_walk = Instant::now();

    // 4. Single-pass node walk — bypasses `DiskGraph::node_weight`, which
    //    would allocate every read into the source's `node_arena`.
    let source_disk_for_nodes: Option<&DiskGraph> = match &source.graph {
        GraphBackend::Disk(dg) => Some(dg.as_ref()),
        _ => None,
    };

    // Sub-phase timers on the same KGLITE_STREAMING_TIMING flag. A clock read
    // is ~50 ns on macOS and five sub-phases are timed per row, so Wikidata's
    // 17 M rows pay a few seconds — small against the 446 s node walk, zero
    // when unset.
    let mut t_lookups = std::time::Duration::ZERO;
    let mut t_read_id_title = std::time::Duration::ZERO;
    let mut t_read_props = std::time::Duration::ZERO;
    let mut t_push_row = std::time::Duration::ZERO;
    let mut t_add_node = std::time::Duration::ZERO;
    let mut row_counter: u64 = 0;

    // Periodic per-million-row breakdown when timing is on: a perf change
    // shows in ~30 s instead of after the full ten-minute walk.
    const PROGRESS_EVERY: u64 = 1_000_000;
    let mut last_progress_row: u64 = 0;
    let mut last_progress_at = Instant::now();
    let mut last_t_lookups = std::time::Duration::ZERO;
    let mut last_t_read_id_title = std::time::Duration::ZERO;
    let mut last_t_read_props = std::time::Duration::ZERO;
    let mut last_t_push_row = std::time::Duration::ZERO;
    let mut last_t_add_node = std::time::Duration::ZERO;

    if let Some(sdg) = source_disk_for_nodes {
        for old_id in 0..n_source_nodes as u32 {
            if !rank.contains(old_id) {
                continue;
            }
            let slot = sdg.node_slot(old_id as usize);
            if !slot.is_alive() {
                continue;
            }
            let t0 = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let type_key = InternedKey::from_u64(slot.node_type);
            let type_name = source.interner.resolve(type_key);

            let src_store = match source.column_store(type_name) {
                Some(s) => s.as_ref(),
                None => continue,
            };
            let writer = match writers.get_mut(type_name) {
                Some(w) => w,
                None => continue,
            };
            if let Some(t) = t0 {
                t_lookups += t.elapsed();
            }

            // Borrowed-read fast path: id/title come back as
            // `BorrowedValue<'_>` / `&str` slices into the source's mmap, and
            // the streaming visitor forwards each property's borrowed bytes
            // straight into dest's BufWriters. This kills the
            // `Value::String(s.to_string())` clone × ~30 properties × 17 M
            // rows = ~510 M heap allocations that dominated the node walk.
            let t1 = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let id_borrowed = src_store
                .id_borrowed(slot.row_id)
                .unwrap_or(crate::datatypes::values::BorrowedValue::Null);
            let title_borrowed = match src_store.title_borrowed(slot.row_id) {
                Some(s) => crate::datatypes::values::BorrowedValue::String(s),
                None => crate::datatypes::values::BorrowedValue::Null,
            };
            if let Some(t) = t1 {
                t_read_id_title += t.elapsed();
            }

            let t2 = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let dest_row_id = writer
                .push_row_borrowed(id_borrowed, title_borrowed, |row| {
                    let t1b = if timing_enabled {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    let r = src_store.try_for_each_property_borrowed(slot.row_id, |key, bv| {
                        row.push_property(key, bv)
                    });
                    if let Some(t) = t1b {
                        t_read_props += t.elapsed();
                    }
                    r
                })
                .map_err(|e| format!("save_subset_streaming_disk: push_row: {}", e))?;
            if let Some(t) = t2 {
                t_push_row += t.elapsed();
            }

            let t3 = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let new_node_data = NodeData {
                // `add_node` reads only `node_type` and `properties.row_id`
                // on the disk path, so `id`/`title` can be Null — later reads
                // resolve through `dest.column_stores[type]`, which sees what
                // `push_row_borrowed` wrote.
                id: Value::Null,
                title: Value::Null,
                node_type: type_key,
                properties: PropertyStorage::Columnar(ColumnarRow::new(dest_row_id)),
            };
            crate::graph::storage::GraphWrite::add_node(&mut dest.graph, new_node_data);
            if let Some(t) = t3 {
                t_add_node += t.elapsed();
            }
            row_counter += 1;

            if timing_enabled && row_counter - last_progress_row >= PROGRESS_EVERY {
                let now = Instant::now();
                let dt = now.duration_since(last_progress_at).as_secs_f64();
                let drows = (row_counter - last_progress_row) as f64;
                let d_lookups = (t_lookups - last_t_lookups).as_secs_f64();
                let d_idtitle = (t_read_id_title - last_t_read_id_title).as_secs_f64();
                let d_props = (t_read_props - last_t_read_props).as_secs_f64();
                let d_push = (t_push_row - last_t_push_row).as_secs_f64();
                let d_addn = (t_add_node - last_t_add_node).as_secs_f64();
                eprintln!(
                    "save_subset_streaming_disk:   progress: rows={} (+{:.0}M) wall={:.2}s \
                     ({:.1}us/row) | dlookups={:.2}s did+t={:.2}s dprops={:.2}s \
                     dpush={:.2}s daddn={:.2}s",
                    row_counter,
                    drows / 1_000_000.0,
                    dt,
                    dt * 1e6 / drows,
                    d_lookups,
                    d_idtitle,
                    d_props,
                    d_push,
                    d_addn,
                );
                last_progress_row = row_counter;
                last_progress_at = now;
                last_t_lookups = t_lookups;
                last_t_read_id_title = t_read_id_title;
                last_t_read_props = t_read_props;
                last_t_push_row = t_push_row;
                last_t_add_node = t_add_node;
            }
        }
    } else if !writers.is_empty() {
        // Streaming targets disk sources only; in-memory / mapped route
        // through `extract_subgraph` + `write_kgl` in the public `save_subset`.
        return Err(
            "save_subset_streaming_disk currently requires a disk-backed source".to_string(),
        );
    }
    log_phase("node walk (push rows + add_node)", phase_node_walk);
    if timing_enabled {
        let t_read_source = t_read_id_title + t_read_props;
        eprintln!(
            "save_subset_streaming_disk:   node walk sub-phases ({} rows): \
             lookups={:.3}s, read_source={:.3}s (id+title={:.3}s, props={:.3}s), \
             push_row={:.3}s, add_node={:.3}s",
            row_counter,
            t_lookups.as_secs_f64(),
            t_read_source.as_secs_f64(),
            t_read_id_title.as_secs_f64(),
            t_read_props.as_secs_f64(),
            t_push_row.as_secs_f64(),
            t_add_node.as_secs_f64(),
        );
    }
    let phase_finalize = Instant::now();

    // 5. Finalize each per-type writer: flush BufWriters, mmap the closed
    //    files, build TypedColumns, install Arc<ColumnStore>. No merge step —
    //    the writers wrote the canonical final files directly.
    let mut arc_dest_stores: HashMap<String, Arc<ColumnStore>> = HashMap::new();
    for (type_name, writer) in writers.into_iter() {
        let store = writer
            .finalize(&source.interner)
            .map_err(|e| format!("save_subset_streaming_disk: finalize {}: {}", type_name, e))?;
        arc_dest_stores.insert(type_name, store);
    }
    dest.clear_column_stores();
    for (type_name, store) in arc_dest_stores {
        dest.install_column_store(&type_name, store);
    }
    log_phase("writer finalize (close + mmap per type)", phase_finalize);
    let phase_edge_walk = Instant::now();

    // 6. Walk source edges in edge_idx order. For each edge passing the
    //    filter and with both endpoints in the kept set, translate via
    //    the rank index and append to dest's pending_edges.
    let edge_filter_set: Option<std::collections::HashSet<u64>> =
        edge_filter.map(|v| v.iter().copied().collect());

    let source_disk = match &source.graph {
        GraphBackend::Disk(dg) => Some(dg.as_ref()),
        _ => None,
    };

    if let Some(sdg) = source_disk {
        // Disk source: sequential read of edge_endpoints.bin with lockstep
        // `edge_properties_at` lookups (source's prop heap in edge_idx order).
        sdg.edge_endpoints.advise_sequential();
        // Re-evict source node_slots: the node pass touched ~17 M slots whose
        // pages would otherwise pile on top of the edge phase's page cache.
        // Both madvise (region hint) and fadvise (fd-level hint) are issued
        // for best-effort eviction across Linux and macOS. Source column
        // stores are mmap'd via MmapColumnStore, which has no advise API yet
        // — their pages stay resident, the next bottleneck here.
        sdg.node_slots.advise_dontneed();
        sdg.node_slots.fadvise_dontneed();
        let n_edges = sdg.next_edge_idx as usize;
        for edge_idx in 0..n_edges {
            let ep = sdg.edge_endpoint(edge_idx);
            if ep.source == crate::graph::storage::disk::csr::TOMBSTONE_EDGE {
                continue;
            }
            if let Some(ref types) = edge_filter_set {
                if !types.contains(&ep.connection_type) {
                    continue;
                }
            }
            let new_src = match rank.old_to_new(ep.source) {
                Some(x) => NodeIndex::new(x as usize),
                None => continue,
            };
            let new_tgt = match rank.old_to_new(ep.target) {
                Some(x) => NodeIndex::new(x as usize),
                None => continue,
            };
            let conn_type = InternedKey::from_u64(ep.connection_type);
            let props = sdg
                .edge_properties_at(edge_idx as u32)
                .map(|cow| cow.into_owned())
                .unwrap_or_default();
            let edge_data = EdgeData::new_interned(conn_type, props);
            let GraphBackend::Disk(ref mut dest_disk) = dest.graph else {
                unreachable!("streaming subset destination is always disk-backed")
            };
            dest_disk
                .try_add_pending_edge(new_src, new_tgt, edge_data)
                .map_err(|error| format!("save_subset_streaming_disk: append edge: {error}"))?;
        }
    } else {
        // Memory source: per-edge lookup via `edge_references()` — there is
        // no generic edge_idx → properties path on the backend-agnostic
        // surface. Plain `Memory` only; every other backend errors out below.
        use petgraph::visit::IntoEdgeReferences;
        let backend = source.graph.plain_memory_digraph();
        if let Some(g) = backend {
            for er in g.edge_references() {
                use petgraph::visit::EdgeRef;
                let w = er.weight();
                if let Some(ref types) = edge_filter_set {
                    if !types.contains(&w.connection_type.as_u64()) {
                        continue;
                    }
                }
                let src = er.source().index() as u32;
                let tgt = er.target().index() as u32;
                let new_src = match rank.old_to_new(src) {
                    Some(x) => NodeIndex::new(x as usize),
                    None => continue,
                };
                let new_tgt = match rank.old_to_new(tgt) {
                    Some(x) => NodeIndex::new(x as usize),
                    None => continue,
                };
                let edge_data = EdgeData::new_interned(w.connection_type, w.properties.clone());
                let GraphBackend::Disk(ref mut dest_disk) = dest.graph else {
                    unreachable!("streaming subset destination is always disk-backed")
                };
                dest_disk
                    .try_add_pending_edge(new_src, new_tgt, edge_data)
                    .map_err(|error| format!("save_subset_streaming_disk: append edge: {error}"))?;
            }
        } else {
            return Err(
                "save_subset_streaming_disk: mapped + recording sources not yet supported"
                    .to_string(),
            );
        }
    }

    log_phase("edge walk (translate + add_edge)", phase_edge_walk);
    let phase_save = Instant::now();

    // 7. Rebuild type_indices from the freshly-added nodes: the streaming
    //    add_node path bypasses the bulk loader's index maintenance, so
    //    dest.type_indices is empty until we walk node_weights here. The
    //    saved `type_indices.bin` is what `MATCH (n:Type)` hits after reload
    //    — without this rebuild the subset reloads with correct node_count
    //    and edges, but every typed Cypher query returns 0.
    dest.rebuild_type_indices();

    // 8. Save: triggers build_csr_from_pending (the external merge sort).
    let save_result = dest.save_disk(path_str);
    log_phase("save_disk (CSR build + sidecars)", phase_save);
    log_phase("TOTAL", phase_start);

    // 9. Drop dest before cleaning the scratch dir: its column_stores hold
    //    Arc handles to the scratch mmaps, and the kernel only releases the
    //    files once those Arcs are gone.
    drop(dest);
    let _ = std::fs::remove_dir_all(&scratch_root);

    save_result
}

/// File-system-safe, **collision-free** slug for a node type name.
/// Wikidata type names carry spaces, accents, CJK and much more, and the
/// obvious sanitization (non-ASCII → `_`) collapses distinct types onto
/// identical paths: 10 distinct single-char types (`ग`, `झ`, `色`, `藪`, ...)
/// all become `_`, and `établissement public` / `Établissement public`,
/// `C♯` / `C♭`, `梅林` / `連合` each collide pairwise. Two colliding types
/// racing through the writer's `OpenOptions::truncate(true).open(...)`
/// overwrite each other's files — observed as a `slice index starts at N but
/// ends at 0` panic far downstream in `rebuild_type_indices`, when the second
/// type to finalize truncates the first's mmap-backed offsets file.
///
/// Appending the InternedKey u64 hash as a hex suffix keeps each path unique
/// even when the readable prefix collapses.
fn sanitize_type_name(name: &str) -> String {
    use std::fmt::Write as _;
    let mut prefix: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let hash = InternedKey::from_str(name).as_u64();
    let _ = write!(prefix, "_{:016x}", hash);
    prefix
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `kept_edges_path` is caller-supplied by every [`pass_a_scan_to_file`]
    /// caller, so an unusable path is reachable. The scan's contract is "memory: bitset
    /// only (~15 MB at 120M nodes)"; falling back to a heap `Vec<PendingEdge>`
    /// would silently allocate 16 B × the source's *total* edge count, turning a
    /// bad path into an OOM at Wikidata scale instead of a returned error.
    #[test]
    fn an_unusable_kept_edges_path_is_an_error_not_a_heap_fallback() {
        use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

        let root = tempfile::tempdir().unwrap();
        let graph =
            new_dir_graph_in_mode(StorageMode::Disk, Some(root.path())).expect("disk graph");
        let crate::graph::schema::GraphBackend::Disk(ref disk) = graph.graph else {
            panic!("expected a disk backend");
        };

        // An existing directory: its parent is creatable, but the file is not.
        let occupied = root.path().join("kept_edges_dir");
        std::fs::create_dir_all(&occupied).unwrap();

        let error = match pass_a_scan_to_file(disk, &SubsetSpec::default(), &occupied) {
            Ok(_) => panic!("an unusable kept-edges path must fail the scan"),
            Err(error) => error,
        };
        assert!(
            error.contains("kept edge buffer"),
            "unexpected error text: {error}"
        );
    }

    #[test]
    fn bitset_set_count_across_block_boundaries() {
        let mut bs = Bitset::with_len(200);
        bs.set(0);
        bs.set(63);
        bs.set(64);
        bs.set(199);
        // Setting the same bit twice does not double-count.
        bs.set(64);
        assert_eq!(bs.count_ones(), 4);
    }

    #[test]
    fn bitset_out_of_range_writes_are_ignored() {
        let mut bs = Bitset::with_len(100);
        bs.set(99);
        bs.set(500); // ignored, not panic
        assert_eq!(bs.count_ones(), 1);
    }

    /// Distinct Wikidata-observed type names that share an ASCII-prefix
    /// shape (after non-alnum→`_` mapping) MUST sanitize to distinct paths —
    /// see `sanitize_type_name` for the truncation panic this pins.
    #[test]
    fn sanitize_type_name_is_collision_free() {
        let groups = vec![
            vec!["établissement public", "Établissement public"],
            vec!["ग", "झ", "ज", "च", "त", "छ", "色", "ञ", "ट", "藪"],
            vec!["C♯", "C♭"],
            vec!["梅林", "連合"],
        ];
        for group in groups {
            let mut sanitized: Vec<String> = group.iter().map(|n| sanitize_type_name(n)).collect();
            sanitized.sort();
            sanitized.dedup();
            assert_eq!(
                sanitized.len(),
                group.len(),
                "sanitize_type_name collapsed distinct types {:?}",
                group
            );
        }
    }

    /// A miniature 'Wikidata-shape' disk graph: Articles, Authors and Venues
    /// joined by two edge types, so an edge-type filter has to discriminate.
    /// AUTHORED_BY covers 5 edges over 4 Articles + 3 Authors; PUBLISHED_IN
    /// covers 4 edges over the same 4 Articles + 2 Venues.
    fn articles_and_authors_on_disk(root: &Path) -> DirGraph {
        use crate::graph::session::execute::{execute_mut, ExecuteOptions};
        use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

        let mut graph = new_dir_graph_in_mode(StorageMode::Disk, Some(root)).expect("disk graph");
        let params = std::collections::HashMap::new();
        let mut run = |query: &str| {
            let opts = ExecuteOptions::eager(&params);
            execute_mut(&mut graph, query, &opts)
                .unwrap_or_else(|e| panic!("query failed: {query}: {e}"));
        };

        run("CREATE (:Article {id: 'a1'}), (:Article {id: 'a2'}), \
             (:Article {id: 'a3'}), (:Article {id: 'a4'})");
        run("CREATE (:Author {id: 'p1'}), (:Author {id: 'p2'}), (:Author {id: 'p3'})");
        run("CREATE (:Venue {id: 'v1'}), (:Venue {id: 'v2'})");
        for (article, author) in [
            ("a1", "p1"),
            ("a1", "p2"),
            ("a2", "p2"),
            ("a3", "p3"),
            ("a4", "p3"),
        ] {
            run(&format!(
                "MATCH (a:Article {{id: '{article}'}}), (p:Author {{id: '{author}'}}) \
                 CREATE (a)-[:AUTHORED_BY]->(p)"
            ));
        }
        for (article, venue) in [("a1", "v1"), ("a2", "v1"), ("a3", "v2"), ("a4", "v2")] {
            run(&format!(
                "MATCH (a:Article {{id: '{article}'}}), (v:Venue {{id: '{venue}'}}) \
                 CREATE (a)-[:PUBLISHED_IN]->(v)"
            ));
        }
        graph
    }

    fn spec_for(edge_types: &[&str]) -> SubsetSpec {
        SubsetSpec {
            edge_types: Some(
                edge_types
                    .iter()
                    .map(|s| InternedKey::from_str(s))
                    .collect(),
            ),
        }
    }

    fn disk_of(graph: &DirGraph) -> &DiskGraph {
        let crate::graph::schema::GraphBackend::Disk(ref disk) = graph.graph else {
            panic!("expected a disk backend");
        };
        disk
    }

    /// The kept set is exactly the endpoints of the matching edges — the
    /// property `save_subset_streaming_disk` builds its rank index from, and
    /// the one no unit test on `Bitset`/`RankIndex` alone can reach: it needs
    /// a real `DiskGraph`'s `edge_endpoints.bin`.
    #[test]
    fn pass_a_keeps_exactly_the_endpoints_of_the_filtered_edge_types() {
        let root = tempfile::tempdir().unwrap();
        let graph = articles_and_authors_on_disk(root.path());
        let disk = disk_of(&graph);

        let authored = pass_a_scan(disk, &spec_for(&["AUTHORED_BY"]));
        assert_eq!(authored.stats.total_edge_count, 9);
        assert_eq!(authored.stats.kept_edge_count, 5);
        assert_eq!(authored.stats.kept_node_count, 7, "4 Articles + 3 Authors");

        let published = pass_a_scan(disk, &spec_for(&["PUBLISHED_IN"]));
        assert_eq!(published.stats.kept_edge_count, 4);
        assert_eq!(published.stats.kept_node_count, 6, "4 Articles + 2 Venues");

        let both = pass_a_scan(disk, &spec_for(&["AUTHORED_BY", "PUBLISHED_IN"]));
        assert_eq!(both.stats.kept_edge_count, 9);
        assert_eq!(both.stats.kept_node_count, 9);

        // No filter is not the same code path as "every type named": it skips
        // the per-edge set lookup entirely, so it gets its own assertion.
        let unfiltered = pass_a_scan(disk, &SubsetSpec::default());
        assert_eq!(unfiltered.stats.kept_edge_count, 9);
        assert_eq!(unfiltered.stats.kept_node_count, 9);

        // An unknown type must keep nothing rather than degrade to "keep all".
        let unknown = pass_a_scan(disk, &spec_for(&["NOT_A_REAL_EDGE_TYPE"]));
        assert_eq!(unknown.stats.kept_edge_count, 0);
        assert_eq!(unknown.stats.kept_node_count, 0);
    }

    /// The file variant must agree with the in-memory one on every stat, and
    /// the `RankIndex` built from its bitset must agree with `count_ones` —
    /// a disagreement means the writer would renumber nodes it did not keep.
    #[test]
    fn pass_a_to_file_agrees_with_the_in_memory_scan_and_its_rank_index() {
        let root = tempfile::tempdir().unwrap();
        let graph = articles_and_authors_on_disk(root.path());
        let disk = disk_of(&graph);
        let out = root.path().join("kept_edges.tmp");

        let spec = spec_for(&["AUTHORED_BY"]);
        let filed = pass_a_scan_to_file(disk, &spec, &out).expect("scan to file");

        assert_eq!(filed.stats.kept_edge_count, 5);
        assert_eq!(filed.stats.kept_node_count, 7);
        assert_eq!(filed.kept_edge_records, 5);
        assert!(out.exists());
        assert!(std::fs::metadata(&out).unwrap().len() >= 5 * 16);

        let rank = RankIndex::from_bitset(filed.kept_nodes);
        assert_eq!(rank.kept_count() as u64, filed.stats.kept_node_count);
    }

    /// Brute-force rank-1 oracle for the popcount-prefix implementation:
    /// O(n_bits) bit-by-bit count, trivially correct.
    fn brute_force_rank(bs: &Bitset, old_id: u32) -> Option<u32> {
        if !bs.get(old_id as usize) {
            return None;
        }
        let mut rank: u32 = 0;
        for i in 0..(old_id as usize) {
            if bs.get(i) {
                rank += 1;
            }
        }
        Some(rank)
    }

    #[test]
    fn rank_index_dense_pattern() {
        // Every 3rd id kept in [0..200). Crosses three 64-bit blocks
        // and ensures within-block + cross-block popcounts both fire.
        let mut bs = Bitset::with_len(200);
        for i in (0..200).step_by(3) {
            bs.set(i);
        }
        let idx = RankIndex::from_bitset(bs.clone());

        // Total kept = ceil(200 / 3) = 67.
        assert_eq!(idx.kept_count(), 67);

        assert_eq!(idx.old_to_new(0), Some(0));
        assert_eq!(idx.old_to_new(3), Some(1));
        assert_eq!(idx.old_to_new(6), Some(2));
        assert_eq!(idx.old_to_new(63), Some(21)); // 63/3 = 21
        assert_eq!(idx.old_to_new(66), Some(22));
        assert_eq!(idx.old_to_new(198), Some(66));

        assert_eq!(idx.old_to_new(1), None);
        assert_eq!(idx.old_to_new(64), None);

        for i in 0..200u32 {
            assert_eq!(idx.old_to_new(i), brute_force_rank(&bs, i));
        }
    }

    #[test]
    fn rank_index_sparse_pattern() {
        // One bit set per 64-bit block. Tests block-prefix advancement
        // when `within_block` is always 0.
        let mut bs = Bitset::with_len(256);
        bs.set(0);
        bs.set(64);
        bs.set(128);
        bs.set(192);
        let idx = RankIndex::from_bitset(bs.clone());

        assert_eq!(idx.kept_count(), 4);
        assert_eq!(idx.old_to_new(0), Some(0));
        assert_eq!(idx.old_to_new(64), Some(1));
        assert_eq!(idx.old_to_new(128), Some(2));
        assert_eq!(idx.old_to_new(192), Some(3));

        assert_eq!(idx.old_to_new(63), None);
        assert_eq!(idx.old_to_new(65), None);
        assert_eq!(idx.old_to_new(255), None);
    }

    #[test]
    fn rank_index_full_pattern() {
        let mut bs = Bitset::with_len(130);
        for i in 0..130 {
            bs.set(i);
        }
        let idx = RankIndex::from_bitset(bs);

        assert_eq!(idx.kept_count(), 130);
        for i in 0..130u32 {
            assert_eq!(idx.old_to_new(i), Some(i));
        }
    }

    #[test]
    fn rank_index_empty_pattern() {
        let bs = Bitset::with_len(200);
        let idx = RankIndex::from_bitset(bs);

        assert_eq!(idx.kept_count(), 0);
        for i in 0..200u32 {
            assert_eq!(idx.old_to_new(i), None);
        }
    }

    #[test]
    fn rank_index_out_of_range_returns_none() {
        let mut bs = Bitset::with_len(100);
        bs.set(99);
        let idx = RankIndex::from_bitset(bs);

        assert_eq!(idx.old_to_new(99), Some(0));
        assert_eq!(idx.old_to_new(100), None);
        assert_eq!(idx.old_to_new(u32::MAX), None);
    }

    #[test]
    fn rank_index_zero_length_bitset() {
        let bs = Bitset::with_len(0);
        let idx = RankIndex::from_bitset(bs);
        assert_eq!(idx.kept_count(), 0);
        assert_eq!(idx.old_to_new(0), None);
    }

    #[test]
    fn rank_index_single_bit_each_block_boundary() {
        // Stress the bit==0 branch (mask = 0, no shift).
        let mut bs = Bitset::with_len(256);
        for i in (0..256).step_by(64) {
            bs.set(i);
        }
        let idx = RankIndex::from_bitset(bs);

        assert_eq!(idx.old_to_new(0), Some(0));
        assert_eq!(idx.old_to_new(64), Some(1));
        assert_eq!(idx.old_to_new(128), Some(2));
        assert_eq!(idx.old_to_new(192), Some(3));
    }

    #[test]
    fn rank_index_pseudo_random_differential() {
        // Deterministic pseudorandom bitset, differential against brute force.
        let mut bs = Bitset::with_len(1024);
        let mut state: u32 = 0xDEAD_BEEF;
        for i in 0..1024 {
            // Cheap LCG; not a real RNG but sufficient for varied bits.
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            if state & 0b11 != 0 {
                // ~75% set rate
                bs.set(i);
            }
        }
        let idx = RankIndex::from_bitset(bs.clone());
        for i in 0..1024u32 {
            assert_eq!(idx.old_to_new(i), brute_force_rank(&bs, i));
        }
    }

    #[test]
    fn rank_index_contains_matches_bitset() {
        let mut bs = Bitset::with_len(100);
        bs.set(5);
        bs.set(50);
        bs.set(99);
        let idx = RankIndex::from_bitset(bs);

        assert!(idx.contains(5));
        assert!(idx.contains(50));
        assert!(idx.contains(99));
        assert!(!idx.contains(0));
        assert!(!idx.contains(49));
        assert!(!idx.contains(100));
    }
}
