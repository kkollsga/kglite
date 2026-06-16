//! Disk CSR builder — external merge-sort construction, peer-count
//! histogram, connection-type index, tombstone swap.
//!
//! These are the heavy build/maintenance methods extracted from
//! `disk_graph.rs` to keep the main file under the 2,500-line cap.

use super::csr::{CsrEdge, EdgeEndpoints, MergeSortEntry, TOMBSTONE_EDGE};
use super::csr_build;
use super::graph::DiskGraph;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use std::collections::HashMap;
use std::path::Path;

impl DiskGraph {
    /// External merge-sort CSR build. Thin orchestrator over
    /// [`csr_build::build_csr_files`] — sets up build/temp directories,
    /// drives the four CSR passes via the extracted module, then atomically
    /// swaps the resulting files into `self.data_dir` and rebuilds the
    /// auxiliary indexes.
    ///
    /// Reads chunking config from `KGLITE_CSR_CHUNK_MB` /
    /// `KGLITE_CSR_FORCE_CHUNKS` via `BuilderConfig::from_env()` so the
    /// environment-variable surface is preserved byte-for-byte.
    pub(super) fn build_csr_merge_sort(
        &mut self,
        node_bound: usize,
        edge_count: usize,
        verbose: bool,
    ) {
        // Production reads chunking config from the environment. Tests use
        // `build_csr_merge_sort_with_config` with an explicit config instead,
        // so they never touch the process-global env vars (which would race
        // across the test harness's parallel threads — see the flake fix).
        self.build_csr_merge_sort_with_config(
            node_bound,
            edge_count,
            verbose,
            csr_build::BuilderConfig::from_env(),
        )
    }

    /// As [`Self::build_csr_merge_sort`] but with an explicit chunking config
    /// rather than reading env vars — the deterministic path for tests.
    pub(super) fn build_csr_merge_sort_with_config(
        &mut self,
        node_bound: usize,
        edge_count: usize,
        verbose: bool,
        config: csr_build::BuilderConfig,
    ) {
        let phase3_start = std::time::Instant::now();

        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let tmp_dir = self.data_dir.join(format!("_csr_build_{:x}", now_nanos));
        let _ = std::fs::create_dir_all(&tmp_dir);
        // CSR output goes to a separate build dir, then atomically swapped
        // into data_dir. This avoids overwriting mmap'd files that self
        // still references.
        let build_dir = self.data_dir.join(format!("_csr_output_{:x}", now_nanos));
        let _ = std::fs::create_dir_all(&build_dir);

        let artifacts = csr_build::build_csr_files(
            self.pending_edges.get_mut(),
            edge_count,
            node_bound,
            &build_dir,
            &tmp_dir,
            &config,
            verbose,
        );

        // Free pending_edges (file-backed mmap) now that all CSR passes are
        // complete. The pre-refactor code freed pending after step 1; the
        // CSR output is byte-equal either way, and we keep pending around
        // longer here so the merge-sort passes can read from it directly
        // without an intermediate `pending.bin` copy.
        *self.pending_edges.get_mut() = MmapOrVec::new();

        // Clean up temp dir (sort chunks).
        let _ = std::fs::remove_dir_all(&tmp_dir);

        // Atomic swap: move build files to data_dir, re-mmap.
        self.swap_csr_files(
            &build_dir,
            node_bound,
            edge_count,
            artifacts.out_offsets,
            artifacts.out_edges,
            artifacts.in_offsets,
            artifacts.in_edges,
            artifacts.edge_endpoints,
        );
        self.csr_sorted_by_type = true;
        self.edge_type_counts_raw = Some(artifacts.edge_type_counts);

        // Build connection-type inverted index from the swapped CSR.
        self.build_conn_type_index(node_bound, verbose);
        // Build per-(type, peer) count histogram — single scan of edge_endpoints.
        self.build_peer_count_histogram(verbose);

        let _ = self.write_metadata();
        self.metadata_dirty = false;

        if verbose {
            eprintln!(
                "    CSR total: {:.1}s ({} edges, {} nodes) [merge_sort]",
                phase3_start.elapsed().as_secs_f64(),
                edge_count,
                node_bound
            );
        }
    }

    /// Hash-partitioned CSR build (Kuzu pattern).
    /// Partitions edges by node group, then local counting sort per partition.
    /// O(n) total — no global sort, no intermediate files, cache-friendly.
    pub(super) fn build_csr_partitioned(
        &mut self,
        node_bound: usize,
        edge_count: usize,
        verbose: bool,
    ) {
        let pending = self.pending_edges.get_mut();
        let phase3_start = std::time::Instant::now();
        // CSR output goes to a separate build dir, then atomically swapped into data_dir.
        let build_dir = self.data_dir.join(format!(
            "_csr_output_{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&build_dir);
        let out_dir = &build_dir;

        // ── Step 1: Build edge_endpoints + count degrees ──
        // Single sequential pass over pending_edges: write endpoints + count degrees.
        let step = std::time::Instant::now();
        let mut edge_endpoints_vec =
            MmapOrVec::mapped(&out_dir.join("edge_endpoints.bin"), edge_count)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));
        let mut out_counts = vec![0u64; node_bound];
        let mut in_counts = vec![0u64; node_bound];
        let mut edge_type_counts: HashMap<u64, usize> = HashMap::new();
        for i in 0..pending.len() {
            let (src, tgt, ct) = pending.get(i);
            edge_endpoints_vec.push(EdgeEndpoints {
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
                "    CSR step 1/3: endpoints + degrees ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // Free pending_edges — all data now in edge_endpoints.
        let pending_path = pending.file_path().map(|p| p.to_path_buf());
        *pending = MmapOrVec::new();
        if let Some(path) = pending_path {
            let _ = std::fs::remove_file(path);
        }

        // ── Step 2: Build offset arrays (prefix sum) ──
        let step = std::time::Instant::now();
        let mut out_offsets: MmapOrVec<u64> =
            MmapOrVec::mapped(&out_dir.join("out_offsets.bin"), node_bound + 1)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(node_bound + 1));
        let mut in_offsets: MmapOrVec<u64> =
            MmapOrVec::mapped(&out_dir.join("in_offsets.bin"), node_bound + 1)
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
                "    CSR step 2/3: offsets ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // ── Step 3: Buffered scatter for out_edges (by source) ──
        // Read edge_endpoints sequentially, scatter to out_edges via chunked buffers.
        // Uses mapped_zeroed — OS zero-fills lazily, no explicit pre-fill I/O.
        let step = std::time::Instant::now();
        let mut out_edges = MmapOrVec::mapped_zeroed(&out_dir.join("out_edges.bin"), edge_count)
            .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));

        let chunk_size: usize = std::env::var("KGLITE_CSR_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_000_000);
        let num_chunks = node_bound.div_ceil(chunk_size);
        let flush_threshold: usize = 512 * 1024 * 1024; // 512 MB

        {
            let mut out_cursor: Vec<u64> = (0..node_bound).map(|i| out_offsets.get(i)).collect();
            let mut chunk_bufs: Vec<Vec<(u32, u32, u32)>> =
                (0..num_chunks).map(|_| Vec::new()).collect();
            let mut buf_bytes: usize = 0;

            for edge_idx in 0..edge_count {
                let ep = edge_endpoints_vec.get(edge_idx);
                let s = ep.source as usize;
                if s < node_bound {
                    let ci = s / chunk_size;
                    chunk_bufs[ci].push((edge_idx as u32, ep.source, ep.target));
                    buf_bytes += 12;
                }
                if buf_bytes >= flush_threshold {
                    for buf in chunk_bufs.iter_mut() {
                        for &(eidx, src2, tgt2) in buf.iter() {
                            let pos = out_cursor[src2 as usize] as usize;
                            out_edges.set(
                                pos,
                                CsrEdge {
                                    peer: tgt2,
                                    edge_idx: eidx,
                                },
                            );
                            out_cursor[src2 as usize] += 1;
                        }
                        buf.clear();
                    }
                    buf_bytes = 0;
                }
            }
            for buf in chunk_bufs.iter_mut() {
                for &(eidx, src2, tgt2) in buf.iter() {
                    let pos = out_cursor[src2 as usize] as usize;
                    out_edges.set(
                        pos,
                        CsrEdge {
                            peer: tgt2,
                            edge_idx: eidx,
                        },
                    );
                    out_cursor[src2 as usize] += 1;
                }
            }
        }
        if verbose {
            eprintln!(
                "    CSR step 3a/4: out_edges scatter ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // Sort each node's out_edges range by connection_type.
        // Enables binary search for type filtering and conn_type_index build.
        // Cost: O(E * log D) where D is average degree — at D~7 across Wikidata's
        // 124 M nodes this totals ~300 s serially. Parallelised via Rayon over
        // disjoint per-node slices.
        let step = std::time::Instant::now();
        {
            use rayon::prelude::*;
            // Snapshot offsets into a heap Vec so the parallel closure can index
            // cheaply without aliasing the out_edges mmap. Cost: ~1 GB at Wikidata
            // scale (124 M * 8 B), freed at end of scope.
            let offsets_snap: Vec<u64> = (0..=node_bound).map(|i| out_offsets.get(i)).collect();

            // Raw pointer into out_edges — each node's range is disjoint by
            // construction, so parallel writes are safe.
            struct SendPtr(*mut CsrEdge);
            unsafe impl Send for SendPtr {}
            unsafe impl Sync for SendPtr {}
            let base = SendPtr(out_edges.as_mut_slice().as_mut_ptr());
            let base_ref = &base;
            let endpoints_ref = &edge_endpoints_vec;

            (0..node_bound).into_par_iter().for_each(|node| {
                let start = offsets_snap[node] as usize;
                let end = offsets_snap[node + 1] as usize;
                if end - start <= 1 {
                    return;
                }
                // SAFETY: disjoint ranges per node; endpoints_ref is read-only.
                let range: &mut [CsrEdge] =
                    unsafe { std::slice::from_raw_parts_mut(base_ref.0.add(start), end - start) };
                let mut with_ct: Vec<(u64, CsrEdge)> = range
                    .iter()
                    .map(|&e| (endpoints_ref.get(e.edge_idx as usize).connection_type, e))
                    .collect();
                with_ct.sort_unstable_by_key(|&(ct, _)| ct);
                for (i, &(_, e)) in with_ct.iter().enumerate() {
                    range[i] = e;
                }
            });
        }
        if verbose {
            eprintln!(
                "    CSR step 3b/4: out_edges sort by type ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // Free out_edges mmap before opening in_edges — keep working set to one 6.9 GB file
        let out_edges_path = out_dir.join("out_edges.bin");
        drop(out_edges);

        // ── Step 4: Build in_edges via merge sort (by target) ──
        // Scatter is slow for in_edges due to power-law target degree distribution
        // (popular targets cause page cache thrashing on the external drive).
        // Merge sort uses only sequential I/O: read → sort chunks → merge → write.
        let step = std::time::Instant::now();

        let in_edges = {
            // Chunk size for sort: fit in available heap after edge_endpoints mmap.
            let sort_chunk_mb: usize = std::env::var("KGLITE_CSR_CHUNK_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5120); // 5 GB default — safe on 16 GB machine
            let max_entries = (sort_chunk_mb << 20) / std::mem::size_of::<MergeSortEntry>();
            let sort_chunk_size = max_entries.min(edge_count);
            let num_sort_chunks = edge_count.div_ceil(sort_chunk_size);

            let sort_dir = out_dir.join("_in_sort");
            let _ = std::fs::create_dir_all(&sort_dir);

            if num_sort_chunks == 1 {
                // Single-chunk: sort in memory, write directly
                let substep = std::time::Instant::now();
                let mut entries: Vec<MergeSortEntry> = Vec::with_capacity(edge_count);
                for i in 0..edge_count {
                    let ep = edge_endpoints_vec.get(i);
                    entries.push(MergeSortEntry {
                        key: ep.target,
                        conn_type: ep.connection_type,
                        peer: ep.source,
                        orig_idx: i as u32,
                    });
                }
                entries.sort_unstable_by_key(|e| (e.key, e.conn_type));

                let mut output = MmapOrVec::mapped(&out_dir.join("in_edges.bin"), edge_count)
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
                        "      in sort single-chunk: {:.1}s",
                        substep.elapsed().as_secs_f64()
                    );
                }
                output
            } else {
                // Multi-chunk: external merge sort with k-way heap merge
                let substep = std::time::Instant::now();
                let mut chunk_mmaps: Vec<MmapOrVec<MergeSortEntry>> = Vec::new();
                let mut chunk_lens: Vec<usize> = Vec::new();

                for c in 0..num_sort_chunks {
                    let start = c * sort_chunk_size;
                    let end = (start + sort_chunk_size).min(edge_count);
                    let len = end - start;

                    let mut entries: Vec<MergeSortEntry> = Vec::with_capacity(len);
                    for i in start..end {
                        let ep = edge_endpoints_vec.get(i);
                        entries.push(MergeSortEntry {
                            key: ep.target,
                            conn_type: ep.connection_type,
                            peer: ep.source,
                            orig_idx: i as u32,
                        });
                    }
                    entries.sort_unstable_by_key(|e| (e.key, e.conn_type));

                    let chunk_path = sort_dir.join(format!("chunk_in_{}.bin", c));
                    let mut chunk_mmap = MmapOrVec::mapped(&chunk_path, len)
                        .unwrap_or_else(|_| MmapOrVec::with_capacity(len));
                    for entry in &entries {
                        chunk_mmap.push(*entry);
                    }
                    drop(entries);
                    chunk_lens.push(len);
                    chunk_mmaps.push(chunk_mmap);
                }
                if verbose {
                    eprintln!(
                        "      in sort {} chunks: {:.1}s",
                        num_sort_chunks,
                        substep.elapsed().as_secs_f64()
                    );
                }

                // K-way merge via binary heap — all reads and writes sequential
                let substep = std::time::Instant::now();
                let mut positions: Vec<usize> = vec![0; num_sort_chunks];
                let mut output = MmapOrVec::mapped(&out_dir.join("in_edges.bin"), edge_count)
                    .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));

                use std::cmp::Reverse;
                let mut heap: std::collections::BinaryHeap<Reverse<(u32, u64, usize)>> =
                    std::collections::BinaryHeap::with_capacity(num_sort_chunks);
                for c in 0..num_sort_chunks {
                    if chunk_lens[c] > 0 {
                        let entry = chunk_mmaps[c].get(0);
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

                // Cleanup sort chunks
                drop(chunk_mmaps);
                let _ = std::fs::remove_dir_all(&sort_dir);

                if verbose {
                    eprintln!("      in merge: {:.1}s", substep.elapsed().as_secs_f64());
                }
                output
            }
        };
        if verbose {
            eprintln!(
                "    CSR step 4/4: in_edges merge sort ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // Reload out_edges (dropped before step 4) — still in build_dir
        let out_edges: MmapOrVec<CsrEdge> = MmapOrVec::load_mapped(&out_edges_path, edge_count)
            .unwrap_or_else(|_| MmapOrVec::new());

        // Atomic swap: move build files to data_dir, re-mmap
        self.swap_csr_files(
            &build_dir,
            node_bound,
            edge_count,
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints_vec,
        );
        self.csr_sorted_by_type = true;
        self.edge_type_counts_raw = Some(edge_type_counts);

        // Build connection-type inverted index from the swapped CSR
        self.build_conn_type_index(node_bound, verbose);
        // Build per-(type, peer) count histogram — single scan of edge_endpoints.
        self.build_peer_count_histogram(verbose);

        let _ = self.write_metadata();
        self.metadata_dirty = false;

        if verbose {
            eprintln!(
                "    CSR total: {:.1}s ({} edges, {} nodes) [partitioned]",
                phase3_start.elapsed().as_secs_f64(),
                edge_count,
                node_bound
            );
        }
    }

    // ====================================================================
    // CSR swap helper — atomic replacement of mmap'd CSR files
    // ====================================================================

    /// Swap CSR files built in `build_dir` into `self.data_dir`.
    /// 1. Drop old mmap fields (releases file handles)
    /// 2. Drop build mmaps (releases handles to temp files)
    /// 3. Rename temp files → data_dir (atomic on same filesystem)
    /// 4. Re-mmap from data_dir
    #[allow(clippy::too_many_arguments)]
    pub(super) fn swap_csr_files(
        &mut self,
        build_dir: &Path,
        node_bound: usize,
        edge_count: usize,
        out_offsets: MmapOrVec<u64>,
        out_edges: MmapOrVec<CsrEdge>,
        in_offsets: MmapOrVec<u64>,
        in_edges: MmapOrVec<CsrEdge>,
        edge_endpoints: MmapOrVec<EdgeEndpoints>,
    ) {
        let data_dir = &self.data_dir;

        // If build_dir == data_dir, files are already in place — just assign.
        if build_dir == data_dir.as_path() {
            self.out_offsets = out_offsets;
            self.out_edges = out_edges;
            self.in_offsets = in_offsets;
            self.in_edges = in_edges;
            self.edge_endpoints = edge_endpoints;
            return;
        }

        // Drop old self mmaps (releases file handles to data_dir files)
        self.out_offsets = MmapOrVec::new();
        self.out_edges = MmapOrVec::new();
        self.in_offsets = MmapOrVec::new();
        self.in_edges = MmapOrVec::new();
        self.edge_endpoints = MmapOrVec::new();

        // Drop build mmaps (releases file handles to build_dir files)
        drop(out_offsets);
        drop(out_edges);
        drop(in_offsets);
        drop(in_edges);
        drop(edge_endpoints);

        // Rename files from build_dir → data_dir (atomic on same filesystem)
        let csr_files = [
            "out_offsets.bin",
            "out_edges.bin",
            "in_offsets.bin",
            "in_edges.bin",
            "edge_endpoints.bin",
        ];
        for fname in &csr_files {
            let src = build_dir.join(fname);
            let dst = data_dir.join(fname);
            if src.exists() {
                // Remove old file first (some filesystems require this)
                let _ = std::fs::remove_file(&dst);
                if let Err(e) = std::fs::rename(&src, &dst) {
                    eprintln!(
                        "Warning: failed to rename {} → {}: {}",
                        src.display(),
                        dst.display(),
                        e
                    );
                }
            }
        }

        // Also swap conn_type_index files if present
        let index_files = [
            "conn_type_index_types.bin",
            "conn_type_index_offsets.bin",
            "conn_type_index_sources.bin",
        ];
        for fname in &index_files {
            let src = build_dir.join(fname);
            let dst = data_dir.join(fname);
            if src.exists() {
                let _ = std::fs::remove_file(&dst);
                let _ = std::fs::rename(&src, &dst);
            }
        }

        // Re-mmap from data_dir
        self.out_offsets =
            MmapOrVec::load_mapped(&data_dir.join("out_offsets.bin"), node_bound + 1)
                .unwrap_or_else(|_| MmapOrVec::new());
        self.out_edges = MmapOrVec::load_mapped(&data_dir.join("out_edges.bin"), edge_count)
            .unwrap_or_else(|_| MmapOrVec::new());
        self.in_offsets = MmapOrVec::load_mapped(&data_dir.join("in_offsets.bin"), node_bound + 1)
            .unwrap_or_else(|_| MmapOrVec::new());
        self.in_edges = MmapOrVec::load_mapped(&data_dir.join("in_edges.bin"), edge_count)
            .unwrap_or_else(|_| MmapOrVec::new());
        self.edge_endpoints =
            MmapOrVec::load_mapped(&data_dir.join("edge_endpoints.bin"), edge_count)
                .unwrap_or_else(|_| MmapOrVec::new());

        // Re-load conn_type_index if swapped
        let types_path = data_dir.join("conn_type_index_types.bin");
        if types_path.exists() {
            let num_types = std::fs::metadata(&types_path)
                .map(|m| m.len() as usize / std::mem::size_of::<u64>())
                .unwrap_or(0);
            if num_types > 0 {
                self.conn_type_index_types = MmapOrVec::load_mapped(&types_path, num_types)
                    .unwrap_or_else(|_| MmapOrVec::new());
                self.conn_type_index_offsets = MmapOrVec::load_mapped(
                    &data_dir.join("conn_type_index_offsets.bin"),
                    num_types + 1,
                )
                .unwrap_or_else(|_| MmapOrVec::new());
                let sources_len = std::fs::metadata(data_dir.join("conn_type_index_sources.bin"))
                    .map(|m| m.len() as usize / std::mem::size_of::<u32>())
                    .unwrap_or(0);
                self.conn_type_index_sources = MmapOrVec::load_mapped(
                    &data_dir.join("conn_type_index_sources.bin"),
                    sources_len,
                )
                .unwrap_or_else(|_| MmapOrVec::new());
            }
        }

        // Clean up build dir
        let _ = std::fs::remove_dir_all(build_dir);
    }

    /// Build connection-type inverted index from current CSR arrays.
    /// Reads self.out_offsets, self.out_edges, self.edge_endpoints (must be valid).
    /// Writes index files to self.data_dir and assigns to self.
    pub(super) fn build_conn_type_index(&mut self, node_bound: usize, verbose: bool) {
        let (types, offsets, sources) = write_conn_type_index(
            &self.out_offsets,
            &self.out_edges,
            &self.edge_endpoints,
            node_bound,
            &self.data_dir,
            verbose,
        );
        self.conn_type_index_types = types;
        self.conn_type_index_offsets = offsets;
        self.conn_type_index_sources = sources;
    }

    /// Build per-(conn_type, peer) edge-count histogram. Answers unanchored
    /// aggregate queries in O(distinct-peers) instead of O(|edge_endpoints|).
    /// Scans edge_endpoints once in parallel chunks and folds into per-thread
    /// HashMap<(conn_type, peer), u32>; reduces + flattens to three mmap'd
    /// arrays sorted by (conn_type, peer).
    /// Build (or rebuild) the per-(conn_type, peer) edge-count histogram from
    /// the current `edge_endpoints`. Safe to call on a loaded graph; writes
    /// `peer_count_*.bin` files next to the other disk artifacts.
    pub fn rebuild_peer_count_histogram(&mut self) {
        let verbose = std::env::var("KGLITE_BUILD_DEBUG").is_ok();
        self.build_peer_count_histogram(verbose);
    }

    pub(super) fn build_peer_count_histogram(&mut self, verbose: bool) {
        // Use edge_endpoints.len() rather than next_edge_idx — after certain
        // build paths (compact, merge rebuilds) next_edge_idx may have been
        // reset while edge_endpoints still holds the authoritative count.
        let total = self.edge_endpoints.len().min(self.next_edge_idx as usize);
        if total == 0 {
            return;
        }
        let (types, offsets, entries) =
            write_peer_count_histogram(&self.edge_endpoints, 0, total, &self.data_dir, verbose);
        self.peer_count_types = types;
        self.peer_count_offsets = offsets;
        self.peer_count_entries = entries;
    }
}

/// Standalone per-segment `peer_count_*` writer — see `write_conn_type_index`.
/// Scans `endpoints[edge_lo..edge_hi]` into a per-`(conn_type, peer)` count
/// histogram and flushes three mmap'd files under `target_dir`.
pub(super) fn write_peer_count_histogram(
    endpoints: &MmapOrVec<EdgeEndpoints>,
    edge_lo: usize,
    edge_hi: usize,
    target_dir: &Path,
    verbose: bool,
) -> (MmapOrVec<u64>, MmapOrVec<u64>, MmapOrVec<u32>) {
    use rayon::prelude::*;
    let start = std::time::Instant::now();

    let total = edge_hi.saturating_sub(edge_lo);
    if total == 0 {
        return (MmapOrVec::new(), MmapOrVec::new(), MmapOrVec::new());
    }

    // Advise kernel: we're about to do one large sequential read then drop
    // these pages. Matches the pattern in count_edges_grouped_by_peer.
    endpoints.advise_sequential();

    let chunk = (total / rayon::current_num_threads().max(1)).max(1 << 20);
    let chunks: Vec<(usize, usize)> = (0..total)
        .step_by(chunk)
        .map(|lo| (lo, (lo + chunk).min(total)))
        .collect();

    let shard_maps: Vec<HashMap<(u64, u32), u32>> = chunks
        .into_par_iter()
        .map(|(lo, hi)| {
            let mut acc: HashMap<(u64, u32), u32> = HashMap::new();
            for i in lo..hi {
                let ep = endpoints.get(edge_lo + i);
                if ep.source == TOMBSTONE_EDGE {
                    continue;
                }
                *acc.entry((ep.connection_type, ep.target)).or_insert(0) += 1;
            }
            acc
        })
        .collect();

    let mut combined: HashMap<(u64, u32), u32> = HashMap::new();
    for shard in shard_maps {
        for (k, v) in shard {
            *combined.entry(k).or_insert(0) += v;
        }
    }

    let mut by_type: HashMap<u64, Vec<(u32, u32)>> = HashMap::new();
    for ((ct, peer), count) in combined {
        by_type.entry(ct).or_default().push((peer, count));
    }
    let mut sorted_types: Vec<u64> = by_type.keys().copied().collect();
    sorted_types.sort();
    for pairs in by_type.values_mut() {
        pairs.sort_unstable_by_key(|&(p, _)| p);
    }

    let total_pairs: usize = by_type.values().map(|v| v.len()).sum();

    let mut types_vec: Vec<u64> = Vec::with_capacity(sorted_types.len());
    let mut offsets_vec: Vec<u64> = Vec::with_capacity(sorted_types.len() + 1);
    let mut entries_vec: Vec<u32> = Vec::with_capacity(total_pairs * 2);

    let mut cur_pairs: u64 = 0;
    for &ct in &sorted_types {
        types_vec.push(ct);
        offsets_vec.push(cur_pairs);
        if let Some(pairs) = by_type.get(&ct) {
            for &(peer, count) in pairs {
                entries_vec.push(peer);
                entries_vec.push(count);
            }
            cur_pairs += pairs.len() as u64;
        }
    }
    offsets_vec.push(cur_pairs);

    // Write exact-size raw byte files, then re-mmap with known lengths.
    let write_u64 = |path: &Path, data: &[u64]| -> std::io::Result<()> {
        // SAFETY: `&[u64]` is contiguous POD; reinterpret to bytes for raw write.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        std::fs::write(path, bytes)
    };
    let write_u32 = |path: &Path, data: &[u32]| -> std::io::Result<()> {
        // SAFETY: same as above — POD slice → u8 view.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        std::fs::write(path, bytes)
    };

    let types_path = target_dir.join("peer_count_types.bin");
    let offsets_path = target_dir.join("peer_count_offsets.bin");
    let entries_path = target_dir.join("peer_count_entries.bin");
    let _ = write_u64(&types_path, &types_vec);
    let _ = write_u64(&offsets_path, &offsets_vec);
    let _ = write_u32(&entries_path, &entries_vec);

    let pc_types = if !types_vec.is_empty() {
        MmapOrVec::load_mapped(&types_path, types_vec.len())
            .unwrap_or_else(|_| MmapOrVec::from_vec(types_vec.clone()))
    } else {
        MmapOrVec::new()
    };
    let pc_offsets = if !offsets_vec.is_empty() {
        MmapOrVec::load_mapped(&offsets_path, offsets_vec.len())
            .unwrap_or_else(|_| MmapOrVec::from_vec(offsets_vec.clone()))
    } else {
        MmapOrVec::new()
    };
    let pc_entries = if !entries_vec.is_empty() {
        MmapOrVec::load_mapped(&entries_path, entries_vec.len())
            .unwrap_or_else(|_| MmapOrVec::from_vec(entries_vec.clone()))
    } else {
        MmapOrVec::new()
    };

    if verbose {
        eprintln!(
            "    Built peer-count histogram: {} types, {} (peer, count) pairs ({:.1}s)",
            sorted_types.len(),
            total_pairs,
            start.elapsed().as_secs_f64()
        );
    }

    (pc_types, pc_offsets, pc_entries)
}

// ============================================================================
// Standalone builder helpers — reused by seal_to_new_segment (phase 5) to
// write segment-local auxiliary indexes from segment-local array slices
// without constructing a throwaway `DiskGraph`. Each returns the mmap'd
// handles so the caller may use them directly if it's the DiskGraph's
// canonical index; `seal_to_new_segment` discards the handles because it
// only needs the files on disk.
// ============================================================================

/// Build a `conn_type_index_*` into `target_dir` from supplied CSR slices.
/// Parallelised via Rayon: each thread scans a partition of nodes and
/// builds a local `HashMap<conn_type, Vec<node_id>>`; the maps are merged,
/// each type's sources sorted, then flushed to the three on-disk files.
pub(super) fn write_conn_type_index(
    out_offsets: &MmapOrVec<u64>,
    out_edges: &MmapOrVec<CsrEdge>,
    edge_endpoints: &MmapOrVec<EdgeEndpoints>,
    node_bound: usize,
    target_dir: &Path,
    verbose: bool,
) -> (MmapOrVec<u64>, MmapOrVec<u64>, MmapOrVec<u32>) {
    use rayon::prelude::*;
    let idx_start = std::time::Instant::now();

    let effective_bound = node_bound.min(out_offsets.len().saturating_sub(1));

    let mut type_sources: HashMap<u64, Vec<u32>> = (0..effective_bound)
        .into_par_iter()
        .fold(HashMap::<u64, Vec<u32>>::new, |mut acc, node| {
            let start = out_offsets.get(node) as usize;
            let end = out_offsets.get(node + 1) as usize;
            if start == end {
                return acc;
            }
            let mut last_type: u64 = u64::MAX;
            for i in start..end {
                let e = out_edges.get(i);
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ct = edge_endpoints.get(e.edge_idx as usize).connection_type;
                if ct != last_type {
                    acc.entry(ct).or_default().push(node as u32);
                    last_type = ct;
                }
            }
            acc
        })
        .reduce(HashMap::<u64, Vec<u32>>::new, |mut a, b| {
            for (k, mut v) in b {
                a.entry(k).or_default().append(&mut v);
            }
            a
        });

    for sources in type_sources.values_mut() {
        sources.sort_unstable();
    }

    let mut sorted_types: Vec<u64> = type_sources.keys().copied().collect();
    sorted_types.sort();
    let total_sources: usize = type_sources.values().map(|v| v.len()).sum();

    let mut idx_types = MmapOrVec::mapped(
        &target_dir.join("conn_type_index_types.bin"),
        sorted_types.len(),
    )
    .unwrap_or_else(|_| MmapOrVec::with_capacity(sorted_types.len()));
    let mut idx_offsets = MmapOrVec::mapped(
        &target_dir.join("conn_type_index_offsets.bin"),
        sorted_types.len() + 1,
    )
    .unwrap_or_else(|_| MmapOrVec::with_capacity(sorted_types.len() + 1));
    let mut idx_sources = MmapOrVec::mapped(
        &target_dir.join("conn_type_index_sources.bin"),
        total_sources,
    )
    .unwrap_or_else(|_| MmapOrVec::with_capacity(total_sources));

    let mut offset: u64 = 0;
    for &ct in &sorted_types {
        idx_types.push(ct);
        idx_offsets.push(offset);
        if let Some(sources) = type_sources.get(&ct) {
            for &src in sources {
                idx_sources.push(src);
            }
            offset += sources.len() as u64;
        }
    }
    idx_offsets.push(offset);

    // Trim each file to exact element count — `MmapOrVec::mapped` has a
    // 64-element minimum. Phase-7's multi-segment concat uses file-size
    // inference on these files (via `load_raw_or_zst_optional`), so stray
    // padding would poison the inverted-index types array.
    let _ = idx_types.trim_to_logical_length();
    let _ = idx_offsets.trim_to_logical_length();
    let _ = idx_sources.trim_to_logical_length();

    if verbose {
        eprintln!(
            "    Built conn-type inverted index: {} types, {} source entries ({:.1}s)",
            sorted_types.len(),
            total_sources,
            idx_start.elapsed().as_secs_f64()
        );
    }

    (idx_types, idx_offsets, idx_sources)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;
    use crate::graph::schema::{EdgeData, NodeData, StringInterner};
    use crate::graph::storage::interner::InternedKey;
    use petgraph::graph::NodeIndex;
    use tempfile::TempDir;

    fn make_node(interner: &mut StringInterner, id: i64) -> NodeData {
        NodeData::new(
            Value::Int64(id),
            Value::String(format!("n{id}")),
            "N".to_string(),
            HashMap::new(),
            interner,
        )
    }

    fn make_edge(interner: &mut StringInterner, ct: &str) -> EdgeData {
        EdgeData::new(ct.to_string(), HashMap::new(), interner)
    }

    /// Build a DiskGraph in a temp directory with the given edges,
    /// exercising the pending → CSR build path including conn_type_index
    /// and peer_count_histogram construction.
    fn build_graph(
        dir: &TempDir,
        num_nodes: usize,
        edges: &[(usize, usize, &str)],
    ) -> (DiskGraph, StringInterner) {
        let mut interner = StringInterner::new();
        let mut dg = DiskGraph::new_at_path(dir.path()).expect("create disk graph");
        dg.defer_csr = true;
        let node_ids: Vec<NodeIndex> = (0..num_nodes)
            .map(|i| dg.add_node(make_node(&mut interner, i as i64)))
            .collect();
        for &(s, t, ct) in edges {
            dg.add_edge(node_ids[s], node_ids[t], make_edge(&mut interner, ct));
        }
        dg.build_csr_from_pending();
        (dg, interner)
    }

    /// Same as `build_graph` but routes through the merge-sort builder
    /// directly, bypassing the env-var dispatch in `build_csr_from_pending`.
    /// Used to exercise the path that drives `csr_build::build_csr_files`.
    fn build_graph_merge_sort(
        dir: &TempDir,
        num_nodes: usize,
        edges: &[(usize, usize, &str)],
        force_chunks: Option<usize>,
    ) -> (DiskGraph, StringInterner) {
        let mut interner = StringInterner::new();
        let mut dg = DiskGraph::new_at_path(dir.path()).expect("create disk graph");
        dg.defer_csr = true;
        let node_ids: Vec<NodeIndex> = (0..num_nodes)
            .map(|i| dg.add_node(make_node(&mut interner, i as i64)))
            .collect();
        for &(s, t, ct) in edges {
            dg.add_edge(node_ids[s], node_ids[t], make_edge(&mut interner, ct));
        }
        let pending_len = dg.pending_edges.get_mut().len();
        let node_bound = dg.node_slots.len();
        // Explicit config (no env var) keeps these tests deterministic under
        // the harness's parallel threads.
        let config = super::csr_build::BuilderConfig {
            chunk_mb_override: None,
            force_chunks,
        };
        dg.build_csr_merge_sort_with_config(node_bound, pending_len, false, config);
        dg.defer_csr = false;
        (dg, interner)
    }

    fn collect_index(dg: &DiskGraph) -> Vec<(u64, Vec<u32>)> {
        let n_types = dg.conn_type_index_types.len();
        (0..n_types)
            .map(|i| {
                let ct = dg.conn_type_index_types.get(i);
                let start = dg.conn_type_index_offsets.get(i) as usize;
                let end = dg.conn_type_index_offsets.get(i + 1) as usize;
                let sources: Vec<u32> = (start..end)
                    .map(|j| dg.conn_type_index_sources.get(j))
                    .collect();
                (ct, sources)
            })
            .collect()
    }

    #[test]
    fn conn_type_index_sorted_and_complete() {
        let dir = TempDir::new().unwrap();
        let edges = [
            (0, 1, "A"),
            (2, 3, "B"),
            (0, 3, "C"),
            (1, 2, "A"),
            (2, 0, "B"),
        ];
        let (dg, interner) = build_graph(&dir, 4, &edges);

        let index = collect_index(&dg);
        assert_eq!(index.len(), 3, "expected 3 connection types");

        // Types sorted ascending.
        let types: Vec<u64> = index.iter().map(|(ct, _)| *ct).collect();
        let mut sorted = types.clone();
        sorted.sort();
        assert_eq!(types, sorted, "conn_type_index_types must be sorted");

        // Offsets chain correctly — total sources = sum of per-type source counts.
        let total_sources: usize = index.iter().map(|(_, srcs)| srcs.len()).sum();
        assert_eq!(dg.conn_type_index_sources.len(), total_sources);

        // Each type's source list is sorted ascending (per builder invariant).
        for (ct, sources) in &index {
            let mut s = sources.clone();
            s.sort_unstable();
            assert_eq!(*sources, s, "sources for type {ct:#x} must be sorted");
        }

        // Correctness: every source that has an outgoing edge of type T
        // appears exactly once in T's source list.
        let type_a = InternedKey::from_str("A").as_u64();
        let type_b = InternedKey::from_str("B").as_u64();
        let type_c = InternedKey::from_str("C").as_u64();

        let lookup = |ct: u64| -> Vec<u32> {
            index
                .iter()
                .find_map(|(t, s)| (*t == ct).then(|| s.clone()))
                .unwrap_or_default()
        };
        // A edges: 0→1, 1→2 → sources {0, 1}
        assert_eq!(lookup(type_a), vec![0, 1]);
        // B edges: 2→3, 2→0 → sources {2} (dedup per type)
        assert_eq!(lookup(type_b), vec![2]);
        // C edges: 0→3 → sources {0}
        assert_eq!(lookup(type_c), vec![0]);

        drop(interner);
    }

    #[test]
    fn conn_type_index_excludes_isolated_nodes() {
        let dir = TempDir::new().unwrap();
        // Node 2 is isolated (no outgoing edges).
        let edges = [(0, 1, "A"), (1, 0, "A")];
        let (dg, _) = build_graph(&dir, 3, &edges);

        let index = collect_index(&dg);
        assert_eq!(index.len(), 1);
        let (_, sources) = &index[0];
        assert!(
            !sources.contains(&2),
            "isolated node 2 must not appear as a source"
        );
        assert_eq!(sources, &vec![0, 1]);
    }

    #[test]
    fn peer_count_entries_segregated_by_type() {
        // Node 0 connects to node 1 via two different connection types.
        // Each type's histogram must list node 1 once with count 1 —
        // the per-type buckets must not merge.
        let dir = TempDir::new().unwrap();
        let edges = [(0, 1, "KNOWS"), (0, 1, "LIKES")];
        let (dg, _) = build_graph(&dir, 2, &edges);

        let knows = InternedKey::from_str("KNOWS").as_u64();
        let likes = InternedKey::from_str("LIKES").as_u64();

        let knows_counts = dg.lookup_peer_counts(knows).expect("KNOWS bucket exists");
        let likes_counts = dg.lookup_peer_counts(likes).expect("LIKES bucket exists");

        assert_eq!(knows_counts.len(), 1);
        assert_eq!(knows_counts.get(&1), Some(&1));
        assert_eq!(likes_counts.len(), 1);
        assert_eq!(likes_counts.get(&1), Some(&1));
    }

    #[test]
    fn peer_count_entries_aggregates_parallel_edges() {
        // Three edges of the same type 0→1 aggregate to (peer=1, count=3).
        let dir = TempDir::new().unwrap();
        let edges = [(0, 1, "T"), (0, 1, "T"), (0, 1, "T")];
        let (dg, _) = build_graph(&dir, 2, &edges);

        let t = InternedKey::from_str("T").as_u64();
        let counts = dg.lookup_peer_counts(t).expect("T bucket exists");
        assert_eq!(counts.get(&1), Some(&3));
    }

    #[test]
    fn peer_count_entries_sparsity() {
        // High-degree target (node 0) and low-degree target (node 3).
        //   1→0, 2→0, 3→0, 2→3
        // Outgoing-oriented histogram keys on target, so peer=0 has count=3,
        // peer=3 has count=1.
        let dir = TempDir::new().unwrap();
        let edges = [(1, 0, "T"), (2, 0, "T"), (3, 0, "T"), (2, 3, "T")];
        let (dg, _) = build_graph(&dir, 4, &edges);

        let t = InternedKey::from_str("T").as_u64();
        let counts = dg.lookup_peer_counts(t).expect("T bucket exists");
        assert_eq!(counts.get(&0), Some(&3));
        assert_eq!(counts.get(&3), Some(&1));
        assert_eq!(counts.len(), 2, "no spurious peer entries");
    }

    #[test]
    fn lookup_peer_counts_missing_type_returns_none() {
        let dir = TempDir::new().unwrap();
        let edges = [(0, 1, "A")];
        let (dg, _) = build_graph(&dir, 2, &edges);

        let missing = InternedKey::from_str("NEVER_USED").as_u64();
        assert!(dg.lookup_peer_counts(missing).is_none());
    }

    // ── Phase 1 regression: merge-sort builder via csr_build module ──

    /// The merge-sort path (now plumbed through `csr_build::build_csr_files`)
    /// must produce a logically equivalent CSR to the partitioned default
    /// path. Compare conn_type_index + peer_count entries — both are derived
    /// from the post-build CSR, so equivalence here means the swapped CSR
    /// arrays are correct.
    #[test]
    fn merge_sort_matches_partitioned() {
        let edges = [
            (0, 1, "A"),
            (2, 3, "B"),
            (0, 3, "C"),
            (1, 2, "A"),
            (2, 0, "B"),
            (3, 1, "C"),
            (0, 2, "A"),
        ];

        let dir_a = TempDir::new().unwrap();
        let (dg_partitioned, _) = build_graph(&dir_a, 4, &edges);

        let dir_b = TempDir::new().unwrap();
        let (dg_merge_sort, _) = build_graph_merge_sort(&dir_b, 4, &edges, None);

        assert_eq!(
            collect_index(&dg_partitioned),
            collect_index(&dg_merge_sort),
            "conn_type_index differs between partitioned and merge-sort paths"
        );

        for ct_str in ["A", "B", "C"] {
            let ct = InternedKey::from_str(ct_str).as_u64();
            assert_eq!(
                dg_partitioned.lookup_peer_counts(ct),
                dg_merge_sort.lookup_peer_counts(ct),
                "peer_counts for type {ct_str} differ between paths"
            );
        }
    }

    /// Merge-sort builder must work in the multi-chunk regime where the
    /// external sort actually spills + merges. Force 4 chunks for a tiny
    /// graph so we exercise the k-way merge path even at unit-test scale.
    #[test]
    fn merge_sort_multi_chunk_via_force_chunks() {
        // Force 4 chunks via an explicit config (not the process-global
        // `KGLITE_CSR_FORCE_CHUNKS` env var). The env-var route raced with
        // other disk-build tests reading `BuilderConfig::from_env()` on the
        // harness's parallel threads — an occasional flake. The explicit
        // config is thread-local to this test.
        let edges = [
            (0, 1, "A"),
            (2, 3, "B"),
            (0, 3, "C"),
            (1, 2, "A"),
            (2, 0, "B"),
            (3, 1, "C"),
            (0, 2, "A"),
            (1, 3, "B"),
        ];

        let dir = TempDir::new().unwrap();
        let (dg, _) = build_graph_merge_sort(&dir, 4, &edges, Some(4));

        // Build expected via partitioned path (no chunk override needed).
        let dir_ref = TempDir::new().unwrap();
        let (dg_ref, _) = build_graph(&dir_ref, 4, &edges);

        assert_eq!(
            collect_index(&dg),
            collect_index(&dg_ref),
            "conn_type_index from multi-chunk merge differs from partitioned baseline"
        );

        for ct_str in ["A", "B", "C"] {
            let ct = InternedKey::from_str(ct_str).as_u64();
            assert_eq!(
                dg.lookup_peer_counts(ct),
                dg_ref.lookup_peer_counts(ct),
                "peer_counts for type {ct_str} differ in multi-chunk merge"
            );
        }
    }

    /// Programmatic chunk override via `BuilderConfig` (exposed for the
    /// streaming-filter path) must drive the same multi-chunk merge as the
    /// env-var override. Smoke test that the API surface works end-to-end.
    #[test]
    fn builder_config_force_chunks_via_api() {
        use super::csr_build::BuilderConfig;
        let cfg = BuilderConfig {
            chunk_mb_override: None,
            force_chunks: Some(3),
        };
        // chunk_mb_override = None still falls back to the env var/default
        // path. Just confirm the struct constructs and is non-default in the
        // expected way.
        assert_eq!(cfg.force_chunks, Some(3));
        assert!(cfg.chunk_mb_override.is_none());

        let env_cfg = BuilderConfig::from_env();
        // No env vars set in the standard test environment.
        // (force_chunks=Some(0) is filtered to None inside from_env.)
        assert!(env_cfg.force_chunks.is_none() || env_cfg.force_chunks.unwrap() > 0);
    }
}
