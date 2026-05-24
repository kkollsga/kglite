//! CSR construction primitives — extracted from `builder.rs` so the
//! merge-sort pipeline can be driven from any `MmapOrVec<(u32, u32, u64)>`
//! source (the existing `pending_edges` build path, or the streaming
//! subgraph filter that ships in a later phase).
//!
//! All I/O is sequential: degree-count + offsets + two external merge
//! sorts (out by source, in by target). Same external-sort design that
//! drove the Wikidata build from 26 h back down to ~2 h.
//!
//! The caller decides what to do with the artifacts — atomic-swap into a
//! `DiskGraph` or finalize as a fresh on-disk graph.
//!
//! Design notes:
//! - Reads `pending` three times (degree count + out merge sort + in merge
//!   sort). All three are sequential; mmap'd `MmapOrVec<(u32, u32, u64)>`
//!   means the OS handles paging.
//! - Writes `edge_endpoints.bin`, `out_offsets.bin`, `in_offsets.bin`,
//!   `out_edges.bin`, `in_edges.bin` into `build_dir`.
//! - Uses `tmp_dir` for sort spill chunks (`chunk_<label>_<n>.bin`); chunk
//!   files are removed at the end of each merge phase.

use super::csr::{CsrEdge, EdgeEndpoints, MergeSortEntry};
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use std::collections::HashMap;
use std::path::Path;

/// Programmatic chunking config for the external merge sort. The existing
/// build path reads its values from env vars (`KGLITE_CSR_CHUNK_MB`,
/// `KGLITE_CSR_FORCE_CHUNKS`); the streaming filter path will pass values
/// directly so the small-RAM target isn't dependent on the user setting
/// env vars.
#[derive(Clone, Debug, Default)]
pub(super) struct BuilderConfig {
    /// Override chunk size in MB. `None` → 12 GB default (preserves the
    /// existing path's behaviour when env var is unset).
    pub(super) chunk_mb_override: Option<usize>,
    /// Force a specific number of chunks (testing). `None` → auto.
    pub(super) force_chunks: Option<usize>,
}

impl BuilderConfig {
    /// Read overrides from `KGLITE_CSR_CHUNK_MB` / `KGLITE_CSR_FORCE_CHUNKS`.
    /// Used by the existing `build_csr_merge_sort` path so behaviour with
    /// unset env vars matches the pre-refactor binary exactly.
    pub(super) fn from_env() -> Self {
        let chunk_mb_override = std::env::var("KGLITE_CSR_CHUNK_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &usize| v > 0);
        let force_chunks = std::env::var("KGLITE_CSR_FORCE_CHUNKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &usize| v > 0);
        Self {
            chunk_mb_override,
            force_chunks,
        }
    }
}

/// CSR artifacts produced by [`build_csr_files`]. `edge_type_counts` is
/// used by the existing path to populate `DiskGraph::edge_type_counts_raw`;
/// the streaming-filter path consumes it for the new graph's metadata.
pub(super) struct CsrArtifacts {
    pub(super) edge_endpoints: MmapOrVec<EdgeEndpoints>,
    pub(super) out_offsets: MmapOrVec<u64>,
    pub(super) in_offsets: MmapOrVec<u64>,
    pub(super) out_edges: MmapOrVec<CsrEdge>,
    pub(super) in_edges: MmapOrVec<CsrEdge>,
    pub(super) edge_type_counts: HashMap<u64, usize>,
}

/// Build CSR files from a pending edges source via external merge sort.
///
/// Output (all sequential writes, all mmap-backed):
/// - `build_dir/edge_endpoints.bin` — `[EdgeEndpoints; edge_count]`
/// - `build_dir/out_offsets.bin` — `[u64; node_bound + 1]`
/// - `build_dir/in_offsets.bin` — `[u64; node_bound + 1]`
/// - `build_dir/out_edges.bin` — `[CsrEdge; edge_count]` sorted by `(src, conn_type)`
/// - `build_dir/in_edges.bin` — `[CsrEdge; edge_count]` sorted by `(tgt, conn_type)`
///
/// `tmp_dir` holds spill chunks during the merge phase. Both directories
/// must already exist; this function does not create them.
pub(super) fn build_csr_files(
    pending: &MmapOrVec<(u32, u32, u64)>,
    edge_count: usize,
    node_bound: usize,
    build_dir: &Path,
    tmp_dir: &Path,
    config: &BuilderConfig,
    verbose: bool,
) -> CsrArtifacts {
    // ── Step 1: edge_endpoints + degree counts (single sequential scan) ──
    let step = std::time::Instant::now();
    let mut edge_endpoints = MmapOrVec::mapped(&build_dir.join("edge_endpoints.bin"), edge_count)
        .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));
    let mut out_counts = vec![0u64; node_bound];
    let mut in_counts = vec![0u64; node_bound];
    let mut edge_type_counts: HashMap<u64, usize> = HashMap::new();
    for i in 0..edge_count {
        let (src, tgt, ct) = pending.get(i);
        edge_endpoints.push(EdgeEndpoints {
            source: src,
            target: tgt,
            connection_type: ct,
        });
        if (src as usize) < node_bound {
            out_counts[src as usize] += 1;
        }
        if (tgt as usize) < node_bound {
            in_counts[tgt as usize] += 1;
        }
        *edge_type_counts.entry(ct).or_insert(0) += 1;
    }
    if verbose {
        eprintln!(
            "    CSR step 1/4: endpoints + degrees ({:.1}s)",
            step.elapsed().as_secs_f64()
        );
    }

    // ── Step 2: prefix-sum offsets (mmap-backed, ~2 GB heap savings at scale) ──
    let step = std::time::Instant::now();
    let mut out_offsets: MmapOrVec<u64> =
        MmapOrVec::mapped(&build_dir.join("out_offsets.bin"), node_bound + 1)
            .unwrap_or_else(|_| MmapOrVec::with_capacity(node_bound + 1));
    let mut in_offsets: MmapOrVec<u64> =
        MmapOrVec::mapped(&build_dir.join("in_offsets.bin"), node_bound + 1)
            .unwrap_or_else(|_| MmapOrVec::with_capacity(node_bound + 1));
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
    drop(out_counts);
    drop(in_counts);
    if verbose {
        eprintln!(
            "    CSR step 2/4: build offsets ({:.1}s)",
            step.elapsed().as_secs_f64()
        );
    }

    // ── Step 3: out_edges (merge-sort by source) ──
    let step = std::time::Instant::now();
    let out_edges = merge_sort_build(
        pending, edge_count, true, tmp_dir, build_dir, "out", config, verbose,
    );
    if verbose {
        eprintln!(
            "    CSR step 3/4: out_edges merge sort ({:.1}s)",
            step.elapsed().as_secs_f64()
        );
    }

    // ── Step 4: in_edges (merge-sort by target) ──
    let step = std::time::Instant::now();
    let in_edges = merge_sort_build(
        pending, edge_count, false, tmp_dir, build_dir, "in", config, verbose,
    );
    if verbose {
        eprintln!(
            "    CSR step 4/4: in_edges merge sort ({:.1}s)",
            step.elapsed().as_secs_f64()
        );
    }

    CsrArtifacts {
        edge_endpoints,
        out_offsets,
        in_offsets,
        out_edges,
        in_edges,
        edge_type_counts,
    }
}

/// External merge sort: read `pending` in chunks, sort each chunk in
/// memory, spill to a chunk file, then k-way merge into the output. All
/// I/O sequential. Reads `pending` once; writes `output_dir/<label>_edges.bin`
/// once. Spill chunks under `chunk_dir/chunk_<label>_<n>.bin` are removed
/// at the end.
#[allow(clippy::too_many_arguments)]
fn merge_sort_build(
    pending: &MmapOrVec<(u32, u32, u64)>,
    edge_count: usize,
    by_source: bool,
    chunk_dir: &Path,
    output_dir: &Path,
    label: &str,
    config: &BuilderConfig,
    verbose: bool,
) -> MmapOrVec<CsrEdge> {
    // Resolve chunking. `force_chunks` wins; otherwise `chunk_mb_override`
    // determines max bytes per chunk; otherwise 12 GB (matches the existing
    // env-var default so behaviour-when-unset is byte-equal).
    let force_chunks = config.force_chunks.unwrap_or(0);
    let chunk_mb = config.chunk_mb_override.unwrap_or(0);
    let (chunk_size, num_chunks) = if force_chunks > 0 {
        let cs = edge_count.div_ceil(force_chunks);
        (cs, force_chunks.min(edge_count.div_ceil(cs)))
    } else {
        let max_bytes = if chunk_mb > 0 {
            chunk_mb << 20
        } else {
            12 << 30 // 12 GB default
        };
        let max_entries = max_bytes / std::mem::size_of::<MergeSortEntry>();
        let cs = max_entries.min(edge_count);
        (cs, edge_count.div_ceil(cs))
    };

    // ── Single-chunk fast path: sort in memory, write directly to output ──
    if num_chunks == 1 {
        let step = std::time::Instant::now();
        let mut entries: Vec<MergeSortEntry> = Vec::with_capacity(edge_count);
        for i in 0..edge_count {
            let (src, tgt, ct) = pending.get(i);
            let (key, peer) = if by_source { (src, tgt) } else { (tgt, src) };
            entries.push(MergeSortEntry {
                key,
                conn_type: ct,
                peer,
                orig_idx: i as u32,
            });
        }
        // Sort by (node, connection_type) so edges are grouped by type within
        // each node's CSR range — enables binary search for type filtering.
        entries.sort_unstable_by_key(|e| (e.key, e.conn_type));

        let mut output =
            MmapOrVec::mapped(&output_dir.join(format!("{}_edges.bin", label)), edge_count)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));
        for entry in &entries {
            output.push(CsrEdge {
                peer: entry.peer,
                edge_idx: entry.orig_idx,
            });
        }
        drop(entries);
        if verbose {
            eprintln!(
                "      {label} single-chunk sort+write: {:.1}s",
                step.elapsed().as_secs_f64()
            );
        }
        return output;
    }

    // ── Multi-chunk path: external merge sort ──
    // Phase A: build sorted chunks (sequential read + sort + sequential write).
    let step = std::time::Instant::now();
    let mut chunk_mmaps: Vec<MmapOrVec<MergeSortEntry>> = Vec::new();
    let mut chunk_lens: Vec<usize> = Vec::new();

    for c in 0..num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(edge_count);
        let len = end - start;

        let mut chunk: Vec<MergeSortEntry> = Vec::with_capacity(len);
        for i in start..end {
            let (src, tgt, ct) = pending.get(i);
            let (key, peer) = if by_source { (src, tgt) } else { (tgt, src) };
            chunk.push(MergeSortEntry {
                key,
                conn_type: ct,
                peer,
                orig_idx: i as u32,
            });
        }
        chunk.sort_unstable_by_key(|e| (e.key, e.conn_type));

        let path = chunk_dir.join(format!("chunk_{}_{}.bin", label, c));
        let mut mmap: MmapOrVec<MergeSortEntry> =
            MmapOrVec::mapped(&path, len).unwrap_or_else(|_| MmapOrVec::with_capacity(len));
        for entry in &chunk {
            mmap.push(*entry);
        }
        chunk_mmaps.push(mmap);
        chunk_lens.push(len);
        drop(chunk);
    }
    if verbose {
        eprintln!(
            "      {label} sort {num_chunks} chunks: {:.1}s",
            step.elapsed().as_secs_f64()
        );
    }

    // Phase B: K-way merge using a binary heap (O(E log K) instead of O(E×K)).
    let merge_start = std::time::Instant::now();
    let mut positions: Vec<usize> = vec![0; num_chunks];
    let mut output =
        MmapOrVec::mapped(&output_dir.join(format!("{}_edges.bin", label)), edge_count)
            .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));

    use std::cmp::Reverse;
    let mut heap: std::collections::BinaryHeap<Reverse<(u32, u64, usize)>> =
        std::collections::BinaryHeap::with_capacity(num_chunks);
    for c in 0..num_chunks {
        if positions[c] < chunk_lens[c] {
            let entry = chunk_mmaps[c].get(positions[c]);
            heap.push(Reverse((entry.key, entry.conn_type, c)));
        }
    }

    for _ in 0..edge_count {
        let Reverse((_key, _ct, best_chunk)) = heap.pop().unwrap();
        let entry = chunk_mmaps[best_chunk].get(positions[best_chunk]);
        positions[best_chunk] += 1;
        output.push(CsrEdge {
            peer: entry.peer,
            edge_idx: entry.orig_idx,
        });
        if positions[best_chunk] < chunk_lens[best_chunk] {
            let next = chunk_mmaps[best_chunk].get(positions[best_chunk]);
            heap.push(Reverse((next.key, next.conn_type, best_chunk)));
        }
    }

    // Cleanup chunk files.
    for c in 0..num_chunks {
        let path = chunk_dir.join(format!("chunk_{}_{}.bin", label, c));
        let _ = std::fs::remove_file(path);
    }
    drop(chunk_mmaps);

    if verbose {
        eprintln!(
            "      {label} merge: {:.1}s",
            merge_start.elapsed().as_secs_f64()
        );
    }
    output
}
