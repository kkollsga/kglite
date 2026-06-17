//! Disk-to-disk streaming subgraph filter.
//!
//! Phase 2 of `save_subset`: Pass A — sequential edge scan that builds a
//! kept-nodes bitset. No output files yet; this is the spike validating
//! the I/O shape on real Wikidata before output plumbing exists.
//!
//! Subsequent phases extend this module:
//! - Phase 3: rank-1 over the bitset + per-type node materialization.
//! - Phase 4: edge translation + property byte-range copy + CSR build.
//! - Phase 5: sidecars + atomic finalize.
//!
//! The streaming pipeline is gated to disk-backed sources. In-memory and
//! mapped graphs route through the existing `to_subgraph().save()` path
//! per CLAUDE.md ("in-memory wins every time"). All I/O here is
//! sequential — no per-node random edge lookups.

use crate::graph::schema::{CowSelection, DirGraph, InternedKey};
use crate::graph::storage::disk::csr::TOMBSTONE_EDGE;
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use std::path::Path;

/// Canonical streaming-filter spec. The fluent chain
/// (`select().expand().save_subset()`) lowers into this; future Cypher
/// integration will lower into the same struct.
#[derive(Clone, Debug, Default)]
pub struct SubsetSpec {
    /// Restrict to edges of these types. `None` means all edge types.
    /// In Phase 2 this is the only filter knob.
    pub edge_types: Option<Vec<InternedKey>>,
    // Phase 3+ will add: seed_node_ids, seed_node_types, expand_hops.
}

/// Stats reported back from a Pass A scan. Used for sanity-checking on
/// real Wikidata before committing to write the output graph.
#[derive(Clone, Debug, Default)]
pub struct ScanStats {
    pub kept_node_count: u64,
    pub kept_edge_count: u64,
    pub total_edge_count: u64,
    pub scan_duration_secs: f64,
}

/// Compact bitset over node ids. `Vec<u64>` blocks; one bit per source
/// node id. Phase 3 will extend this with rank-1 (popcount prefix array)
/// for O(1) old→new id translation.
#[derive(Clone, Debug)]
pub struct Bitset {
    blocks: Vec<u64>,
    len: usize,
}

impl Bitset {
    /// Allocate a bitset covering `[0..len)`. All bits start cleared.
    pub fn with_len(len: usize) -> Self {
        let n_blocks = len.div_ceil(64);
        Self {
            blocks: vec![0u64; n_blocks],
            len,
        }
    }

    /// Set bit `i` to 1. Out-of-range writes are silently ignored — the
    /// caller (Pass A) is expected to keep ids within `node_bound` but
    /// disk graphs occasionally surface tombstone-adjacent ids.
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

    /// Number of bits the bitset covers.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Total set bits. Used to size the output graph in Phase 3+.
    pub fn count_ones(&self) -> u64 {
        self.blocks.iter().map(|b| b.count_ones() as u64).sum()
    }

    /// Raw block view — used by [`RankIndex`] to compute block-prefix
    /// popcounts without re-scanning the bitset.
    #[inline]
    pub(crate) fn blocks(&self) -> &[u64] {
        &self.blocks
    }
}

// ── Rank-1 over the kept-nodes bitset ──────────────────────────────────────
//
// Phase 3: build a popcount-prefix array so old→new node id translation
// is O(1) and entirely in RAM. No 480 MB dense remap.
//
// `block_prefix[k]` holds `popcount(bitset[0..k*64])`. For a query
// `old_to_new(old_id)`:
//   block = old_id / 64
//   bit   = old_id % 64
//   mask  = (1 << bit) - 1
//   rank  = block_prefix[block] + popcount(bitset_block[block] & mask)
//
// At Wikidata scale (120M nodes): 15 MB bitset + 15 MB prefix = 30 MB
// total. Two `count_ones` calls per query; both compile to `popcnt` on
// x86_64 / equivalents on ARM. No disk access.
//
// The new id space is contiguous `[0..kept_count)` ordered by source id —
// matches the natural sequential walk of `node_slots.bin` in Pass B.

/// O(1) translator from source node ids to dense destination ids. Built
/// once after Pass A; consumed by node materialization (Phase 4) and
/// edge translation (Phase 4). All RAM, zero disk reads.
///
/// `dead_code` is allowed at the impl level because Phase 3 ships the
/// primitive in isolation — Phase 4 is the first consumer and lights up
/// every method.
#[derive(Debug, Clone)]
pub struct RankIndex {
    bitset: Bitset,
    /// `block_prefix[k]` = popcount of `bitset.blocks()[0..k]`, i.e. the
    /// rank of bit `k * 64` exclusive of itself. Length = `bitset.blocks().len() + 1`
    /// so a 1-block bitset has prefixes `[0, popcount(block 0)]`.
    block_prefix: Vec<u32>,
    /// Total kept count = `block_prefix.last()`.
    kept_count: u32,
}

impl RankIndex {
    /// Build the rank-1 index from a kept-nodes bitset. Single linear
    /// pass over the `Vec<u64>` blocks; cost is O(n_blocks).
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

    /// True if source id `old_id` is in the kept set.
    #[inline]
    pub fn contains(&self, old_id: u32) -> bool {
        self.bitset.get(old_id as usize)
    }

    /// Map a source node id to its destination id. Returns `None` when
    /// `old_id` is not in the kept set or is out of range.
    ///
    /// Cost: O(1) — two array accesses + one `popcount`.
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
        // Bit must be set for `old_id` to be in the kept set.
        if (block >> bit) & 1 == 0 {
            return None;
        }
        // Mask of bits before `bit` in the same block.
        let mask: u64 = if bit == 0 { 0 } else { (1u64 << bit) - 1 };
        let within_block = (block & mask).count_ones();
        Some(self.block_prefix[block_idx] + within_block)
    }
}

/// Result of [`pass_a_scan`].
pub struct PassAResult {
    /// Source node ids that survive the filter. Endpoints of every kept
    /// edge, plus any explicit seed nodes (Phase 3+). Phase 2 spike does
    /// not surface the bitset to callers; Phase 3 consumes it to build
    /// the rank-1 index.
    #[allow(dead_code)]
    pub kept_nodes: Bitset,
    pub stats: ScanStats,
}

/// Pass A: sequential scan of `edge_endpoints.bin` with an edge-type
/// filter, building a kept-nodes bitset.
///
/// Memory: `Bitset` is `n_nodes / 8` bytes (~15 MB at 120M Wikidata
/// nodes). No other allocations scale with graph size.
///
/// I/O: one sequential read of `edge_endpoints` (16 B per edge). Disk
/// reads are mmap'd — the kernel prefetches sequential access. Tombstoned
/// edges (TOMBSTONE_EDGE marker in `source`) are skipped without
/// touching property storage.
///
/// Phase 4 will extend this loop to also write `kept_edges.tmp` +
/// `edge_prop_offsets.bin` + `edge_prop_heap.bin` for the destination
/// graph, all in lockstep with the same sequential scan.
pub fn pass_a_scan(source: &DiskGraph, spec: &SubsetSpec) -> PassAResult {
    let n_nodes = source.node_slots.len();
    let mut kept_nodes = Bitset::with_len(n_nodes);

    // Convert edge-type filter to a hash set keyed on the raw u64 hash —
    // matches `EdgeEndpoints.connection_type` directly without
    // re-interning per edge. Empty filter = "keep all".
    let edge_type_set: Option<std::collections::HashSet<u64>> = spec
        .edge_types
        .as_ref()
        .map(|v| v.iter().map(|k| k.as_u64()).collect());

    // Hint the kernel: we're about to scan endpoints front-to-back.
    source.edge_endpoints.advise_sequential();

    let scan_start = std::time::Instant::now();
    let n_edges = source.next_edge_idx as usize;
    let mut kept_edge_count: u64 = 0;
    for edge_idx in 0..n_edges {
        let ep = source.edge_endpoints.get(edge_idx);
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

    // Drop the source's edge_endpoints page cache now that we're done
    // sweeping it sequentially. On Wikidata that's 9 GB of mmap pages
    // dirty enough to dominate RSS for the rest of the pipeline if left
    // resident — `advise_sequential` is just a hint and macOS doesn't
    // honor it aggressively. DONTNEED forces eviction.
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

// ── Pass A with file output — Phase 4 ──────────────────────────────────────
//
// Same sequential scan as `pass_a_scan` but also spills the kept edges to
// a `MmapOrVec<(u32, u32, u64)>` at `kept_edges_path`. The on-disk record
// shape matches the existing CSR builder's input
// (`csr_build::build_csr_files` consumes `&MmapOrVec<(u32, u32, u64)>`),
// so a later phase can drive the merge sort over this file by translating
// `(src, tgt)` via the rank index in the iterator.
//
// Edge property bytes are NOT inlined yet — that's a Phase 5 extension
// that requires either (a) a sidecar file keyed by source edge_idx or
// (b) a fourth column in the temp record. For Phase 4 we keep the temp
// file shape compatible with the existing builder; properties travel
// independently in subsequent work.
//
// Memory: bitset (~15 MB at 120M nodes). No heap buffer scales with kept
// edge count; appended records go straight to the mmap'd file.
//
// I/O: one sequential read of `edge_endpoints` + one sequential write to
// `kept_edges_path`.

/// Result of [`pass_a_scan_to_file`]. The temp file path is returned so
/// the caller (Phase 5+) can hand it to `csr_build::build_csr_files`
/// after wrapping it with a rank-translating iterator.
pub struct PassAFileResult {
    pub kept_nodes: Bitset,
    pub stats: ScanStats,
    pub kept_edges_path: std::path::PathBuf,
    pub kept_edge_records: u64,
}

/// Pass A with file output. Identical semantics to [`pass_a_scan`] but
/// also appends `(src, tgt, conn_type)` for each kept edge to a file at
/// `kept_edges_path` via `MmapOrVec::mapped`. The file is sized for the
/// total edge count up front (a safe upper bound on kept edges); Phase 5
/// reads the actual count from `kept_edge_records`.
pub fn pass_a_scan_to_file(
    source: &DiskGraph,
    spec: &SubsetSpec,
    kept_edges_path: &Path,
) -> Result<PassAFileResult, String> {
    let n_nodes = source.node_slots.len();
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

    let mut kept_edges: MmapOrVec<(u32, u32, u64)> = MmapOrVec::mapped(kept_edges_path, n_edges)
        .unwrap_or_else(|_| MmapOrVec::with_capacity(n_edges));

    source.edge_endpoints.advise_sequential();
    let scan_start = std::time::Instant::now();
    let mut kept_edge_count: u64 = 0;
    for edge_idx in 0..n_edges {
        let ep = source.edge_endpoints.get(edge_idx);
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
        kept_edges.push((ep.source, ep.target, ep.connection_type));
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
// to 7.1 GB peak RSS on Wikidata Articles+P50+Authors. The streaming path:
//
// 1. Determine kept node ids per type (sorted by source id).
// 2. Create the destination as a disk-mode `DirGraph` from the start —
//    its `pending_edges` are file-backed via `MmapOrVec`, no in-memory
//    petgraph copy.
// 3. For each kept type T, build the destination's `ColumnStore` for T
//    by walking source's nodes of type T in sorted source-id order.
//    Source row reads stay sequential within a single `ColumnStore` at
//    a time, so the OS page cache stays warm — kills the random-I/O
//    pattern that came from `enable_columnar`'s interleaved reads
//    across multiple type stores.
// 4. Walk source `node_slots` in id order; for each kept old_idx, add a
//    single `NodeData::Columnar { store: arc(dest_store), row_id }` to
//    the dest's disk graph. `DiskGraph::add_node` only reads `row_id`,
//    so the dest's auto-assigned `NodeIndex` matches the global
//    `RankIndex.old_to_new(old_idx)`.
// 5. Walk source `edge_endpoints` in edge_idx order, applying the
//    edge-type filter. For each kept edge, translate src/tgt via the
//    rank index and call `dest.graph.add_edge`. With `defer_csr=true`
//    the dest disk graph appends to its file-backed `pending_edges`
//    (sequential write, bounded heap).
// 6. `dest.save_disk(out_path)` triggers `build_csr_from_pending` (the
//    Phase 1 merge-sort builder, all sequential I/O).
//
// Memory budget at Wikidata Articles+P50+Authors scale (17.4M kept):
//  - kept_per_type indices: ~70 MB (sorted u32 per kept type)
//  - rank index: ~30 MB
//  - dest column stores during push: bounded by max-type heap.
//  - pending_edges: file-backed, ~16 B per kept edge, mmap.
//
// Sequential I/O strategy:
//  - Pass A scan: 1× sequential read of source `edge_endpoints.bin`.
//  - Per-type column-store push: sequential reads of one source store
//    at a time (kept ids in source-id order = monotone row_ids).
//  - Dest column-store writes: sequential append (heap during push,
//    flushed once at save_disk).
//  - Pending-edges write: sequential append to file-backed mmap.
//  - CSR build: external merge sort (existing).

/// Save a filtered subgraph to disk.
///
/// `selection` defines which nodes are kept (typically built via the
/// fluent chain: `kg.select(...).expand(...)` returns a selection that
/// gets handed here). All edges between kept nodes are included.
///
/// Output is a v3 binary file (single `.kgl` file for in-memory/mapped
/// sources, directory for disk sources via `save_disk`). The file can be
/// reloaded into any storage mode via `kglite.load(path, storage=...)`.
///
/// `_spec` is currently unused — the selection carries the filter. The
/// parameter exists so v2 (Cypher integration) can lower into the same
/// entry point without an API change.
pub fn save_subset(
    source: &DirGraph,
    selection: &CowSelection,
    out_path: &Path,
    _spec: Option<&SubsetSpec>,
) -> Result<(), String> {
    use crate::graph::mutation::subgraph::extract_subgraph;

    // 1. Materialize the filtered subgraph in-memory. `extract_subgraph`
    //    works for every source storage mode — its reads go through the
    //    `GraphRead` trait which is implemented for memory/mapped/disk.
    //    For disk sources, `node_weight` returns NodeData with
    //    `PropertyStorage::Columnar { store: Arc<source_store>, row_id }`,
    //    so the extracted graph holds Arc references into the source's
    //    column stores. No deep clone of property data.
    let mut extracted = extract_subgraph(source, selection)?;

    // 2. Consolidate properties into self-contained column stores so the
    //    output is independent of the source's stores. Required by both
    //    save paths below.
    extracted.enable_columnar();

    let path_str = out_path.to_str().ok_or_else(|| {
        format!(
            "save_subset: out_path is not valid UTF-8: {}",
            out_path.display()
        )
    })?;

    // 3. Choose the right serializer based on graph size. The single-file
    //    v3 format (`write_kgl`) bincode-serializes the whole
    //    DirGraph in one go and trips bincode's size limit at scale —
    //    Wikidata-class extracts (~17 M nodes / 35 M edges) hit it.
    //    Above the threshold, convert to disk mode and write the
    //    directory format which scales without per-blob limits. Reload
    //    works for both: `kglite.load(path)` auto-detects file vs dir.
    //
    //    Threshold picked empirically: the v3 single-file format is
    //    comfortable below ~1 M nodes; everything above gets the
    //    directory treatment for safety.
    const SINGLE_FILE_NODE_THRESHOLD: u64 = 1_000_000;

    use crate::graph::storage::GraphRead;
    let node_count = u64::try_from(extracted.graph.node_count()).unwrap_or(u64::MAX);
    if node_count <= SINGLE_FILE_NODE_THRESHOLD {
        // Single .kgl file path — small subgraphs.
        let mut arc = std::sync::Arc::new(extracted);
        crate::graph::io::file::prepare_save(&mut arc);
        crate::graph::io::file::write_kgl(&arc, path_str)
            .map_err(|e| format!("save_subset: write_kgl failed: {}", e))
    } else {
        // Directory format — large subgraphs. enable_disk_mode builds CSR
        // files in a temp dir; save_disk renames them into `path_str`.
        extracted.enable_disk_mode()?;
        extracted.save_disk(path_str)
    }
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

    // Stage timers — opt-in via `KGLITE_STREAMING_TIMING=1` so we
    // don't pollute production stderr. The 4 phases (writer setup,
    // node walk, edge walk, finalize+save_disk) are the granularity at
    // which we attack performance regressions in this pipeline.
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

    // Clone source metadata so the dest is fully self-contained on reload.
    dest.interner = source.interner.clone();
    dest.type_schemas = source.type_schemas.clone();
    dest.node_type_metadata = source.node_type_metadata.clone();
    dest.connection_type_metadata = source.connection_type_metadata.clone();
    dest.id_field_aliases = source.id_field_aliases.clone();
    dest.title_field_aliases = source.title_field_aliases.clone();
    dest.parent_types = source.parent_types.clone();

    // Bulk-loader contract: defer CSR build until save_disk so add_edge
    // appends to the file-backed pending_edges instead of going through
    // the slow per-edge overflow path.
    if let GraphBackend::Disk(ref mut dg) = dest.graph {
        dg.defer_csr = true;
    }

    // 2. Build a global rank index over the kept node set so we can
    //    translate source node ids to dest ids in O(1) RAM. New ids are
    //    assigned in source-id order across all types.
    //
    //    The bitset is sized to the source node bound; Pass A also uses
    //    this size, so peak together stays at ~30 MB on Wikidata.
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
    //    nodes of type T in source-id order. Sequential reads of one
    //    source store at a time = warm OS page cache.
    //
    //    NodeData with PropertyStorage::Columnar carries `row_id` directly
    //    into DiskGraph::add_node — that's what binds the new node_slot
    //    to the column-store row we just pushed. The store Arc is dropped
    //    by add_node, so the per-type Arc is installed in dest.column_stores
    //    at the end of this loop.
    //
    //    Per-type "row_id within store" is the push position (0, 1, 2,
    //    …); per-source-id mapping `(old_idx → dest_row_id)` is recorded
    //    via the rank index so the edge phase below builds the right
    //    `NodeIndex`.
    //
    // v3: per-type `TypeWriter` streams rows directly to the dest's
    // final column files. No heap-backed chunk buffer, no merge step
    // — each push appends to the pre-opened `BufWriter`s. peak heap is
    // bounded by `(open_buf_writers × BUF_SIZE) + Mixed-column heap`.
    // See `subgraph_streaming_writer.rs` for the writer protocol.
    use crate::graph::mutation::subgraph_streaming_writer::TypeWriter;

    let scratch_root = out_path.join(".tmp_streaming");
    std::fs::create_dir_all(&scratch_root).map_err(|e| {
        format!(
            "save_subset_streaming_disk: create_dir_all({}): {}",
            scratch_root.display(),
            e
        )
    })?;

    // One writer per kept type. Open all up front: they hold the file
    // handles for the type's column files. We need every writer alive
    // simultaneously because the source walk visits types in arbitrary
    // (source-id-determined) order — without all writers open, we'd
    // either reorder source reads (defeating sequential-read patterns)
    // or have to re-open files per push. macOS `kern.maxfilesperproc`
    // is typically 61440 (verified via sysctl); 4500 types × ~5 files
    // each = ~22500 fds, well under the limit. Linux defaults are at
    // least as generous.
    let mut writers: HashMap<String, TypeWriter> = HashMap::new();
    for type_name in kept_per_type.keys() {
        let schema = if let Some(src_store) = source.column_stores.get(type_name) {
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

        // Match source's id/title column types — Wikidata mixes
        // `string` (Q-codes) and `uniqueid` ids across types, so we
        // can't hard-code one variant. `id_type_str` / `title_type_str`
        // return the source's stored type tag (mmap-backed or
        // in-memory). Default to "mixed" when the source has no
        // column store entry — TypeWriter's Mixed buffer handles
        // anything.
        let id_type = source
            .column_stores
            .get(type_name)
            .and_then(|s| s.id_type_str())
            .unwrap_or("mixed");
        let title_type = source
            .column_stores
            .get(type_name)
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

    // 4. Single-pass node walk — bypasses `DiskGraph::node_weight` which
    //    would allocate every read into source's `node_arena`. Direct
    //    disk access pattern:
    //      sdg.node_slots[old_id] → (type, src_row_id)
    //      source.column_stores[type] → get_id / get_title / row_properties
    //      writers[type].push_row(id, title, props)
    //      dest.graph.add_node
    let source_disk_for_nodes: Option<&DiskGraph> = match &source.graph {
        GraphBackend::Disk(dg) => Some(dg.as_ref()),
        _ => None,
    };

    // Placeholder Arc handed to DiskGraph::add_node — it reads only
    // `row_id` from the Columnar variant and drops the Arc. Real per-
    // type Arcs are produced by writer.finalize() and installed into
    // dest.column_stores after the loop.
    let placeholder_arc: Option<Arc<ColumnStore>> = if writers.is_empty() {
        None
    } else {
        Some(Arc::new(ColumnStore::new(
            Arc::new(crate::graph::schema::TypeSchema::new()),
            &HashMap::new(),
            &source.interner,
        )))
    };

    // Sub-phase timers, gated on the same KGLITE_STREAMING_TIMING flag.
    // Each `Instant::now()` is ~50 ns on macOS so 4 calls × 17 M
    // iterations = ~3-4 s of measurement overhead on Wikidata when
    // enabled, which is a small fraction of the 446 s node walk.
    // Off when the env var is unset — zero overhead.
    let mut t_lookups = std::time::Duration::ZERO;
    let mut t_read_id_title = std::time::Duration::ZERO;
    let mut t_read_props = std::time::Duration::ZERO;
    let mut t_push_row = std::time::Duration::ZERO;
    let mut t_add_node = std::time::Duration::ZERO;
    let mut row_counter: u64 = 0;

    // Periodic progress: when timing is on, print per-million-row
    // breakdown every PROGRESS_EVERY rows. Lets us assess a perf
    // change in ~30 seconds of bench time instead of 10 minutes —
    // if the per-million numbers don't move, the fix isn't working.
    const PROGRESS_EVERY: u64 = 1_000_000;
    let mut last_progress_row: u64 = 0;
    let mut last_progress_at = Instant::now();
    let mut last_t_lookups = std::time::Duration::ZERO;
    let mut last_t_read_id_title = std::time::Duration::ZERO;
    let mut last_t_read_props = std::time::Duration::ZERO;
    let mut last_t_push_row = std::time::Duration::ZERO;
    let mut last_t_add_node = std::time::Duration::ZERO;

    if let (Some(sdg), Some(placeholder_arc)) = (source_disk_for_nodes, placeholder_arc) {
        for old_id in 0..n_source_nodes as u32 {
            if !rank.contains(old_id) {
                continue;
            }
            let slot = sdg.node_slots.get(old_id as usize);
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

            let src_store = match source.column_stores.get(type_name) {
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

            // Borrowed-read fast path: get_id / get_title return
            // `BorrowedValue<'_>` / `&str` slices into the source's
            // mmap with no heap allocation. The streaming visitor
            // forwards each property's borrowed bytes straight into
            // dest's BufWriters via `push_row_borrowed`. This kills
            // the `Value::String(s.to_string())` clone × ~30
            // properties × 17 M rows = ~510 M heap allocations that
            // dominated the v3 node walk wall time.
            //
            // `DiskGraph::add_node` reads only `data.node_type` and
            // `data.properties.row_id`; the `id` and `title` fields
            // are ignored on the disk path (verified — see the
            // `add_node` impl). So we can pass `Value::Null` for
            // both and skip the clone for NodeData. Reads later go
            // through `dest.column_stores[type].get_id/title` which
            // sees what `push_row_borrowed` wrote.
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
                // add_node ignores `id` and `title` on the disk path
                // (only reads node_type + properties.row_id). Skip
                // the heap allocation entirely.
                id: Value::Null,
                title: Value::Null,
                node_type: type_key,
                properties: PropertyStorage::Columnar {
                    store: Arc::clone(&placeholder_arc),
                    row_id: dest_row_id,
                },
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
        // Streaming primitives only target disk sources. In-memory /
        // mapped sources route through extract_subgraph + write_kgl
        // earlier in the public save_subset.
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

    // 5. Finalize each per-type writer: flush BufWriters, mmap the
    //    closed files, build TypedColumns, install Arc<ColumnStore>.
    //    No merge step — the writers wrote directly to canonical
    //    final files, so this is just a close+mmap.
    let mut arc_dest_stores: HashMap<String, Arc<ColumnStore>> = HashMap::new();
    for (type_name, writer) in writers.into_iter() {
        let store = writer
            .finalize(&source.interner)
            .map_err(|e| format!("save_subset_streaming_disk: finalize {}: {}", type_name, e))?;
        arc_dest_stores.insert(type_name, store);
    }
    dest.column_stores = arc_dest_stores;
    dest.sync_disk_column_stores();
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
        // Disk source: sequential read of edge_endpoints.bin + lockstep
        // edge_properties_at lookups (which read source's prop heap in
        // edge_idx order = sequential).
        sdg.edge_endpoints.advise_sequential();
        // Re-evict source node_slots now that the node materialization
        // pass above touched ~17 M source slots — without an explicit
        // drop, those pages stay resident and pile on top of edge-pass
        // page cache. Apply both madvise (region hint) and fadvise (fd-
        // level page-cache hint) for best-effort eviction across Linux
        // and macOS. Source column stores are mmap'd via
        // MmapColumnStore which doesn't yet have an advise API; their
        // pages remain resident through the edge phase — that's the
        // next bottleneck once these levers are in place.
        sdg.node_slots.advise_dontneed();
        sdg.node_slots.fadvise_dontneed();
        let n_edges = sdg.next_edge_idx as usize;
        for edge_idx in 0..n_edges {
            let ep = sdg.edge_endpoints.get(edge_idx);
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
            crate::graph::storage::GraphWrite::add_edge(
                &mut dest.graph,
                new_src,
                new_tgt,
                edge_data,
            );
        }
    } else {
        // Memory / mapped source: walk via for_each_edge_endpoint_key to
        // get the same sequential shape on a backend-agnostic surface.
        // Properties go through `for_each_edge_of_conn_type` per kept
        // edge type — but we don't have the edge_idx → properties path
        // generically here, so fall back to per-edge lookup via
        // `edge_references()`. Less optimal but correct.
        use petgraph::visit::IntoEdgeReferences;
        let backend = match &source.graph {
            GraphBackend::Memory(g) => Some(g.inner()),
            GraphBackend::Mapped(_) | GraphBackend::Recording(_) | GraphBackend::Disk(_) => None,
        };
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
                crate::graph::storage::GraphWrite::add_edge(
                    &mut dest.graph,
                    new_src,
                    new_tgt,
                    edge_data,
                );
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

    // 7. Rebuild type_indices from the freshly-added nodes. The streaming
    //    add_node path bypasses the bulk loader's index maintenance, so
    //    dest.type_indices is empty until we walk node_weights here. The
    //    saved `type_indices.bin` is what `MATCH (n:Type)` queries hit
    //    after reload — without this rebuild, the subset reloads with
    //    correct node_count and edges but every typed Cypher query
    //    returns 0.
    dest.rebuild_type_indices();

    // 8. Save: triggers build_csr_from_pending (the Phase 1 merge sort).
    let save_result = dest.save_disk(path_str);
    log_phase("save_disk (CSR build + sidecars)", phase_save);
    log_phase("TOTAL", phase_start);

    // 9. Drop dest before cleaning the scratch dir so the mmap files
    //    aren't held open. dest's column_stores carry Arc handles to the
    //    scratch mmaps; once dest is gone, the Arcs drop and the kernel
    //    releases the file handles.
    drop(dest);
    let _ = std::fs::remove_dir_all(&scratch_root);

    save_result
}

/// File-system-safe, **collision-free** slug for a node type name.
/// Wikidata's type names include spaces, commas, accents, CJK, and
/// many other non-ASCII characters; the obvious sanitization (replace
/// non-ASCII with `_`) collapses distinct types onto identical paths.
/// On real Wikidata: 10 distinct single-char types (`ग`, `झ`, `色`,
/// `藪`, ...) all sanitize to `_`, the pair `établissement public` /
/// `Établissement public` both collide on `_tablissement_public`,
/// `C♯` and `C♭` collide on `C_`, and `梅林` / `連合` both collide on
/// `__`. Two such types racing through the writer's `OpenOptions::
/// truncate(true).open(...)` would overwrite each other's files —
/// observed as a `slice index starts at N but ends at 0` panic far
/// downstream during `rebuild_type_indices` (the second-to-finalize
/// type's mmap-backed offsets file gets truncated out from under the
/// first type's `Arc<ColumnStore>`).
///
/// Fix: append the InternedKey u64 hash as a hex suffix. Even when
/// the readable prefix collapses, the hash makes each type's path
/// globally unique.
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
    /// shape (after non-alnum→`_` mapping) MUST sanitize to distinct
    /// paths. Without per-type uniqueness, the writers' files race and
    /// the second-to-finalize type truncates the first's mmap-backed
    /// files out from under it.
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

    // Note: `pass_a_scan` requires a real `DiskGraph`. End-to-end tests
    // live in `tests/test_subgraph_streaming.py` (Phase 2) where building
    // a disk graph is straightforward via the public Python API.

    // ── Phase 3: RankIndex ────────────────────────────────────────────

    /// Helper: brute-force rank-1 to validate the popcount-prefix
    /// implementation against. Iterates the bitset bit-by-bit, counting
    /// kept bits before `old_id`. O(n_bits) but trivially correct.
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

        // Spot-check the first few mappings.
        assert_eq!(idx.old_to_new(0), Some(0));
        assert_eq!(idx.old_to_new(3), Some(1));
        assert_eq!(idx.old_to_new(6), Some(2));
        assert_eq!(idx.old_to_new(63), Some(21)); // 63/3 = 21
        assert_eq!(idx.old_to_new(66), Some(22));
        assert_eq!(idx.old_to_new(198), Some(66));

        // Not-in-set: 1, 2, 4, 5, ...
        assert_eq!(idx.old_to_new(1), None);
        assert_eq!(idx.old_to_new(64), None);

        // Differential against brute force across the whole range.
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

        // Bits before/after each set bit must report None.
        assert_eq!(idx.old_to_new(63), None);
        assert_eq!(idx.old_to_new(65), None);
        assert_eq!(idx.old_to_new(255), None);
    }

    #[test]
    fn rank_index_full_pattern() {
        // All bits set. old_to_new(i) == i for every i.
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
        // Empty bitset — every query returns None, kept_count = 0.
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
        // Pseudorandom-but-deterministic bitset; differential against
        // brute-force rank across a whole 1024-bit range.
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
