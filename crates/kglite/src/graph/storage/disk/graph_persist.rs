//! On-disk persistence for `DiskGraph`: metadata schema, save/load
//! pipelines, multi-segment manifest building, and segment CSR
//! reconciliation.
//!
//! Split out of `graph.rs` to keep the core graph file under the
//! 2,500-line cap. Lives in a sibling `impl DiskGraph {}` block.

use crate::graph::schema::InternedKey;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use serde::{Deserialize, Serialize};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::csr::TOMBSTONE_EDGE;
use super::edge_properties::{EdgePropertyStore, EdgePropertyStoreMeta};
use super::graph::{enumerate_segment_dirs, segment_subdir, DiskGraph, CURRENT_CSR_LAYOUT_VERSION};
use super::property_index;

/// Metadata stored alongside the binary files in the disk graph directory.
#[derive(Serialize, Deserialize)]
struct DiskGraphMeta {
    node_count: usize,
    node_slots_len: usize,
    edge_count: usize,
    next_edge_idx: u32,
    out_offsets_len: usize,
    out_edges_len: usize,
    in_offsets_len: usize,
    in_edges_len: usize,
    edge_endpoints_len: usize,
    free_node_slots: Vec<u32>,
    free_edge_slots: Vec<u32>,
    /// CSR edges sorted by (node, connection_type) — enables binary search.
    /// Added in v0.7.8; older graphs default to false.
    #[serde(default)]
    csr_sorted_by_type: bool,
    /// True if any node or edge has been removed since construction.
    /// Enables `count_edges_filtered` to short-circuit the per-edge
    /// tombstone check on fresh / read-only graphs. Default false is
    /// correct for legacy graphs only if they never saw a removal; since
    /// we can't retroactively prove that, treat unknown (missing field)
    /// as `true` (conservative) via a custom default.
    #[serde(default = "default_has_tombstones")]
    has_tombstones: bool,
    /// Edge property storage format. 0 = legacy bincode+zstd HashMap
    /// (edge_properties.bin.zst). 1 = columnar mmap base + overlay
    /// (edge_prop_offsets.bin + edge_prop_heap.bin). Added in PR2 of
    /// the disk-graph-improvement-plan; defaults to 0 for backward
    /// compat with older .kgl directories.
    #[serde(default)]
    edge_properties_format: u8,
    /// Lengths needed to mmap the columnar edge-property files. Zero
    /// for format=0 graphs or graphs that don't have any edge properties.
    #[serde(default)]
    edge_properties_meta: EdgePropertyStoreMeta,
    /// CSR-layout version. 0 = legacy flat (all files at graph root).
    /// 1 = segmented (CSR / columns / per-segment indexes live under
    /// `seg_000/`). Added in PR1 phase 4; defaults to 0 so pre-phase-4
    /// .kgl directories still load.
    #[serde(default)]
    csr_layout_version: u8,
    /// Boundary past which nodes are in the still-mutable tail (not yet
    /// sealed into any segment). `seal_to_new_segment` flushes
    /// `node_slots[sealed_nodes_bound..node_count]` into a new
    /// `seg_NNN/` and advances this. Added in PR1 phase 8; zero for
    /// pre-phase-8 graphs via serde default — their `seg_000` accounts
    /// for everything below `node_count`, so `seal_to_new_segment`
    /// will treat subsequent adds as the next-segment tail on first
    /// call (after saving to bump the watermark).
    #[serde(default)]
    sealed_nodes_bound: u32,
}

fn default_has_tombstones() -> bool {
    // Conservative: older graphs that lack this field get the slow path
    // (tombstone-aware). Fresh builds emit `false` explicitly.
    true
}

impl DiskGraph {
    /// Write metadata JSON to the graph directory.
    /// Called automatically after CSR build and after mutations. Reads
    /// the edge-property file metadata from the current data_dir so the
    /// JSON reflects whatever was last persisted there; mutations since
    /// then live in the overlay until the next explicit `save_to_dir`.
    pub(crate) fn write_metadata(&self) -> std::io::Result<()> {
        let edge_props_meta = EdgePropertyStore::meta_for(&self.data_dir);
        self.write_metadata_to(&self.data_dir, edge_props_meta)
    }

    fn write_metadata_to(
        &self,
        dir: &Path,
        edge_props_meta: EdgePropertyStoreMeta,
    ) -> std::io::Result<()> {
        let meta = DiskGraphMeta {
            node_count: self.node_count,
            node_slots_len: self.node_slots.len(),
            edge_count: self.edge_count,
            next_edge_idx: self.next_edge_idx,
            out_offsets_len: self.out_offsets.len(),
            out_edges_len: self.out_edges.len(),
            in_offsets_len: self.in_offsets.len(),
            in_edges_len: self.in_edges.len(),
            edge_endpoints_len: self.edge_endpoints.len(),
            free_node_slots: self.free_node_slots.clone(),
            free_edge_slots: self.free_edge_slots.clone(),
            csr_sorted_by_type: self.csr_sorted_by_type,
            has_tombstones: self.has_tombstones,
            // PR2: format=1 = columnar. Fresh graphs always emit the new
            // format; legacy format=0 is only ever loaded, never written.
            edge_properties_format: 1,
            edge_properties_meta: edge_props_meta,
            // PR1 phase 4: fresh saves always emit the segmented layout.
            csr_layout_version: CURRENT_CSR_LAYOUT_VERSION,
            // PR1 phase 8: persist the watermark so reloads know which
            // nodes already live in sealed segments vs which are tail.
            sealed_nodes_bound: self.sealed_nodes_bound,
        };
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        std::fs::write(dir.join("disk_graph_meta.json"), json)
    }

    /// Build a single-segment summary covering the whole graph. PR1 phase 2
    /// — subsequent phases split the graph into multiple segments and this
    /// helper becomes one of several per-segment builders.
    ///
    /// conn_types are read from the conn_type_index (built alongside the
    /// CSR), so the summary only reflects types that made it into the
    /// index — typically all of them after a save-time compact. Node type
    /// counts come from `column_stores` (one entry per node type); the
    /// count is the stored row count minus any tombstone slack (which we
    /// don't tally precisely here — the planner treats `has_node_type` as
    /// a lower-bound predicate).
    ///
    /// PR1 phase 5 additionally populates `indexed_prop_ranges` from the
    /// segment's on-disk `PropertyIndex` files (which live in `data_dir`,
    /// which under phase-4 layout is `seg_000/`). Today only string
    /// indexes exist, so every entry uses `PropRange::StringBloomPlaceholder`
    /// — a conservative placeholder that never prunes, but registers the
    /// `(type_hash, prop_hash)` pair so phase 6+ can upgrade to real
    /// bloom filters without changing the manifest schema.
    fn build_single_segment_manifest(&self) -> super::segment_summary::SegmentManifest {
        use super::segment_summary::{PropRange, SegmentManifest, SegmentSummary};
        use std::collections::HashSet;
        let mut summary = SegmentSummary::new(0, 0);
        summary.node_id_hi = self.node_count as u32;
        summary.edge_count = self.edge_count as u64;

        // Connection types: iterate the persisted inverted-index u64 list.
        for i in 0..self.conn_type_index_types.len() {
            summary.conn_types.insert(self.conn_type_index_types.get(i));
        }
        // Also include overflow edge conn_types that may not yet be in
        // the persisted index (post-CSR mutations).
        for edges in self.overflow_out.values() {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ct = self.edge_endpoints.get(e.edge_idx as usize).connection_type;
                summary.conn_types.insert(ct);
            }
        }

        // Node type counts: one column_store per node type.
        // `row_count` includes tombstoned rows — conservative upper
        // bound is fine for planner pruning (the planner uses these
        // only as "any rows of this type?" predicates).
        for (type_key, store) in &self.column_stores {
            summary
                .node_type_counts
                .insert(type_key.as_u64(), store.row_count());
        }

        // PR1 phase 5: record every (type, prop) index present in the
        // segment. Prefer the in-memory cache — its keys hold the
        // *original* type/prop strings the user passed in, so hashes
        // round-trip cleanly through `InternedKey::from_str`. Fall back
        // to a disk scan for indexes that were persisted earlier and
        // haven't been queried this session; scanned names are sanitised
        // filenames, which are identity-equal to originals for the only
        // shape we ever emit (`[A-Za-z0-9_-]`).
        let mut seen: HashSet<(u64, u64)> = HashSet::new();
        if let Ok(cache) = self.property_indexes.read() {
            for ((ty, prop), slot) in cache.iter() {
                if slot.is_none() {
                    continue;
                }
                let t_hash = InternedKey::from_str(ty).as_u64();
                let p_hash = InternedKey::from_str(prop).as_u64();
                if seen.insert((t_hash, p_hash)) {
                    summary.indexed_prop_ranges.push((
                        t_hash,
                        p_hash,
                        PropRange::StringBloomPlaceholder,
                    ));
                }
            }
        }
        for (t_hash, p_hash) in property_index::scan_segment_hashes(&self.data_dir) {
            if seen.insert((t_hash, p_hash)) {
                summary.indexed_prop_ranges.push((
                    t_hash,
                    p_hash,
                    PropRange::StringBloomPlaceholder,
                ));
            }
        }

        let mut manifest = SegmentManifest::new();
        manifest.append(summary);
        manifest
    }

    /// Save disk graph state. For disk mode, binary arrays are already on disk
    /// via mmap — this only flushes metadata + any in-memory overflow/properties.
    ///
    /// Takes `&mut self` because the edge-property store may need to drop
    /// its base mmap before overwriting the files (when target_dir equals
    /// the current data_dir).
    pub fn save_to_dir(
        &mut self,
        target_dir: &Path,
        _interner: &crate::graph::schema::StringInterner,
    ) -> std::io::Result<()> {
        // Drain mutation caches before writing:
        //   - `edge_mut_cache` → `edge_properties` (existing behavior,
        //     edge props have been persisting correctly because
        //     `edge_properties` is owned by DiskGraph).
        //   - `node_mut_cache` → `self.column_stores` via the
        //     clone-apply-replace flush. The caller
        //     (`DirGraph::save_disk`) mirrors the post-flush Arcs back
        //     into its own side immediately after.
        self.clear_arenas();
        std::fs::create_dir_all(target_dir)?;

        // Phase 6 + 7 auto-wire. `seal_to_new_segment` now handles
        // both clean-tail (segment-local) and cross-segment
        // (full-range) overflow — the `overflow_is_clean` gate from
        // phase 6 is gone. Fall through to compact-and-rewrite when:
        //   - no tail (nothing new to seal)
        //   - target_dir differs from the graph's root (save-as)
        //   - segment_manifest has never been persisted (initial save)
        let have_prior_save = !self.segment_manifest.is_empty();
        let same_dir = target_dir == self.data_dir.parent().unwrap_or(target_dir);
        if have_prior_save && same_dir && self.sealed_nodes_bound < self.node_count as u32 {
            let _seg_id = self.seal_to_new_segment(target_dir)?;
            return Ok(());
        }

        // PR1 phase 4: CSR binaries live under a per-segment subdirectory.
        // Fresh graphs already have self.data_dir pointing at their own
        // seg_000/; save-as to a different path creates a matching
        // subdir. Phase 6 parameterised the subdir name on segment id —
        // today every save still targets id 0 until phase 7 splits
        // writes across segments.
        let csr_target = target_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&csr_target)?;

        // The compact-rewrite path consolidates the full in-memory
        // graph into seg_000. If a prior seal produced seg_NNN > 0
        // dirs, they carried a subset of the now-consolidated state;
        // leaving them on disk causes the next reload's
        // `enumerate_segment_dirs` to pick them up and concat against
        // the fresh seg_000 — double-counting nodes and edges. Remove
        // every seg_NNN for N > 0 before rewriting seg_000.
        if csr_target == self.data_dir {
            for (seg_id, seg_path) in enumerate_segment_dirs(target_dir) {
                if seg_id > 0 {
                    let _ = std::fs::remove_dir_all(&seg_path);
                }
            }
        }

        // Always persist the core CSR arrays, regardless of mmap vs
        // heap backing. Previously this path skipped writes when
        // `csr_target == self.data_dir` and relied on mmap file
        // persistence. After a prior seal (`reconcile_seg0_csr`)
        // these arrays are heap-backed, so mmap persistence doesn't
        // apply — the on-disk file stays at the pre-seal trimmed size
        // while the in-memory Vec carries the full state. On reload
        // the meta → file-length mismatch fails loudly. `save_to_file`
        // handles both backings: Heap writes bytes; Mapped-same-path
        // truncates to logical length.
        self.node_slots
            .save_to_file(&csr_target.join("node_slots.bin"))?;
        self.out_offsets
            .save_to_file(&csr_target.join("out_offsets.bin"))?;
        self.out_edges
            .save_to_file(&csr_target.join("out_edges.bin"))?;
        self.in_offsets
            .save_to_file(&csr_target.join("in_offsets.bin"))?;
        self.in_edges
            .save_to_file(&csr_target.join("in_edges.bin"))?;
        self.edge_endpoints
            .save_to_file(&csr_target.join("edge_endpoints.bin"))?;

        // Save overflow edges (bincode + zstd)
        if !self.overflow_out.is_empty() || !self.overflow_in.is_empty() {
            let overflow = (&self.overflow_out, &self.overflow_in);
            let bytes = bincode::serialize(&overflow).map_err(std::io::Error::other)?;
            let compressed =
                zstd::encode_all(bytes.as_slice(), 3).map_err(std::io::Error::other)?;
            std::fs::write(target_dir.join("overflow_edges.bin.zst"), compressed)?;
        }

        // Save edge properties (columnar: edge_prop_offsets.bin + edge_prop_heap.bin).
        // Always write even when empty so format=1 + zero-length files are
        // self-consistent with the metadata. No interner/guard needed —
        // the columnar format stores raw u64 hashes directly. Phase-4
        // layout puts these alongside the CSR in `csr_target`.
        let upper = self.next_edge_idx;
        self.edge_properties.save_to(&csr_target, upper)?;
        let edge_props_meta = EdgePropertyStore::meta_for(&csr_target);

        // Trim the conn_type_index mmap'd files to their logical length.
        // `MmapOrVec::mapped(path, initial_cap)` has a 64-element minimum,
        // so a 1-type index leaves 512 bytes on disk with stale zeros that
        // the loader can't distinguish from real u64 type hashes. Without
        // this trim, `[r:TYPE]` typed-edge queries return 0 rows after
        // reload (pre-existing bug on v0.8.10).
        for field in [
            &self.conn_type_index_types as &MmapOrVec<u64>,
            &self.conn_type_index_offsets,
        ] {
            if let Some(path) = field.file_path().map(PathBuf::from) {
                let _ = field.save_to_file(&path);
            }
        }
        if let Some(path) = self.conn_type_index_sources.file_path().map(PathBuf::from) {
            let _ = self.conn_type_index_sources.save_to_file(&path);
        }

        // PR1 phase 8: trim the core CSR mmap files to their logical
        // length when writing in-place (csr_target == self.data_dir).
        // The not-in-place branch above already writes exact-sized
        // files via `save_to_file(&different_path)`. Without this trim
        // the multi-segment load path would misread the padding as
        // real CSR data — the single-segment path uses `meta.*_len`
        // and is unaffected.
        //
        // `trim_to_logical_length` truncates the file AND remaps, so
        // subsequent `push`es on the same MmapOrVec see the new size
        // as the starting capacity and extend cleanly. A naive
        // `save_to_file(&same_path)` set_len without remap leaves the
        // mmap spanning past the new EOF and SIGBUSes on the next
        // push — caught in the 0.8.11 ingest benchmark.
        if csr_target == self.data_dir {
            let _ = self.node_slots.trim_to_logical_length();
            let _ = self.out_offsets.trim_to_logical_length();
            let _ = self.out_edges.trim_to_logical_length();
            let _ = self.in_offsets.trim_to_logical_length();
            let _ = self.in_edges.trim_to_logical_length();
            let _ = self.edge_endpoints.trim_to_logical_length();
        }

        // PR1 phase 2: compute and persist a single-segment manifest.
        // Subsequent phases split saves into multiple segments; today the
        // manifest describes the whole graph as one segment so the planner
        // hook can be wired up without changing read-path behaviour.
        let manifest = self.build_single_segment_manifest();
        manifest.save_to(target_dir)?;
        self.segment_manifest = manifest;

        // Save metadata to target_dir (not data_dir)
        self.write_metadata_to(target_dir, edge_props_meta)?;

        // Also save optional persisted-cache files if present and csr_target differs from data_dir
        if csr_target != self.data_dir {
            for fname in [
                "conn_type_index_types.bin",
                "conn_type_index_offsets.bin",
                "conn_type_index_sources.bin",
                "peer_count_types.bin",
                "peer_count_offsets.bin",
                "peer_count_entries.bin",
            ] {
                let src = self.data_dir.join(fname);
                if src.exists() {
                    let _ = std::fs::copy(&src, csr_target.join(fname));
                }
            }
        }

        // PR1 phase 8: after a full save, everything up to node_count
        // is accounted for in the (single-segment) on-disk state. Bump
        // the sealed watermark so any subsequent `seal_to_new_segment`
        // correctly treats post-save adds as the new tail.
        self.sealed_nodes_bound = self.node_count as u32;

        Ok(())
    }

    /// Seal the still-mutable tail of the graph — nodes in
    /// `[sealed_nodes_bound, node_count)` plus overflow edges — into a
    /// fresh `seg_NNN/` directory under `root`. Advances
    /// `sealed_nodes_bound` to `node_count`, clears consumed overflow,
    /// appends a [`SegmentSummary`] to the on-disk manifest, and
    /// rewrites `disk_graph_meta.json`.
    ///
    /// ## Two output modes (phase 7)
    ///
    /// The new segment is written in one of two modes depending on
    /// whether the overflow contains cross-segment edges:
    ///
    /// - **Segment-local** (phase 8 default): all overflow edges have
    ///   both source AND target in `[tail_lo, tail_hi)`. The new
    ///   segment's `out_offsets` / `in_offsets` have length
    ///   `tail_len + 1` and index by the segment's node_slots
    ///   positions (0..tail_len). This is what clean-tail workloads
    ///   (phase 6's `save_to_dir` auto-wire) produce.
    ///
    /// - **Full-range** (phase 7): at least one overflow edge has an
    ///   endpoint below `tail_lo`. The segment's `out_offsets` /
    ///   `in_offsets` have length `node_count + 1` and index by
    ///   global node id — nodes without edges in this seal get
    ///   zero-length ranges. Allows a single segment to carry edges
    ///   whose source / target is in any prior segment's node range.
    ///
    /// `concat_segment_csrs` at load time distinguishes the two modes
    /// by comparing `out_offsets.len()` to `node_slots.len() + 1`. In
    /// full-range mode it unions per-node contributions across
    /// segments; in segment-local mode it preserves the phase-7
    /// "each node's edges live in exactly one segment" invariant.
    ///
    /// ## Auxiliary indexes (phase 5)
    ///
    /// Every seal — segment-local or full-range — writes its own
    /// `conn_type_index_*`, `peer_count_*`, and flushes the
    /// `edge_properties` overlay. Reload merges all three across
    /// segments, so typed-edge matches, peer aggregates, and
    /// `edge_weight()` all work correctly on sealed edges.
    pub fn seal_to_new_segment(&mut self, root: &Path) -> std::io::Result<u32> {
        use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints};

        let tail_lo = self.sealed_nodes_bound;
        let tail_hi = self.node_count as u32;
        if tail_hi <= tail_lo {
            return Err(std::io::Error::other(
                "seal_to_new_segment: nothing to seal — node_count <= sealed_nodes_bound",
            ));
        }
        let tail_len = (tail_hi - tail_lo) as usize;

        // Classify overflow: segment-local (all endpoints in tail) or
        // cross-segment (at least one edge has an endpoint below
        // tail_lo). Also catches tombstoned entries so they're dropped
        // silently rather than written.
        let mut has_cross_segment = false;
        for edges in self.overflow_out.values() {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ep = self.edge_endpoints.get(e.edge_idx as usize);
                if ep.source < tail_lo || ep.target < tail_lo {
                    has_cross_segment = true;
                    break;
                }
            }
            if has_cross_segment {
                break;
            }
        }

        // Next segment id = max(existing) + 1, or 0 if the dir is
        // empty (shouldn't happen in practice — first save creates
        // seg_000 before any seal).
        let existing = enumerate_segment_dirs(root);
        let next_id = existing
            .iter()
            .map(|(id, _)| *id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let seg_dir = root.join(segment_subdir(next_id));
        std::fs::create_dir_all(&seg_dir)?;

        // Collect overflow edges. Each entry records GLOBAL source /
        // target so full-range mode can index by global id; segment-
        // local mode subtracts `tail_lo` on write.
        struct SealEdge {
            src_global: u32,
            tgt_global: u32,
            conn_type: u64,
        }
        let mut seal_edges: Vec<SealEdge> = Vec::new();
        for (&src_global, edges) in &self.overflow_out {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ep = self.edge_endpoints.get(e.edge_idx as usize);
                seal_edges.push(SealEdge {
                    src_global,
                    tgt_global: ep.target,
                    conn_type: ep.connection_type,
                });
            }
        }
        // Sort by (source, conn_type) — identical to pre-phase-7
        // behaviour for segment-local; full-range uses the same order
        // so offsets can be built in one sweep.
        seal_edges.sort_by_key(|e| (e.src_global, e.conn_type));
        let n_edges = seal_edges.len();

        // In full-range mode the offset arrays span every global node.
        // In segment-local mode they span only the tail.
        let offsets_len = if has_cross_segment {
            self.node_count + 1
        } else {
            tail_len + 1
        };

        // ─── node_slots — unchanged: tail only. ───
        let mut node_slots: MmapOrVec<DiskNodeSlot> = MmapOrVec::with_capacity(tail_len);
        for i in 0..tail_len {
            node_slots.push(self.node_slots.get(tail_lo as usize + i));
        }

        // ─── edge_endpoints: global source/target, segment-local
        //     edge_idx 0..n_edges (same as before). ───
        let mut edge_endpoints: MmapOrVec<EdgeEndpoints> = MmapOrVec::with_capacity(n_edges);
        for e in &seal_edges {
            edge_endpoints.push(EdgeEndpoints {
                source: e.src_global,
                target: e.tgt_global,
                connection_type: e.conn_type,
            });
        }

        // ─── out_offsets / out_edges: CSR keyed by (segment-local
        //     or global) source. For segment-local mode the offset
        //     index is `src - tail_lo`; for full-range it's `src`
        //     directly. Phase 7's concat uses `offsets_len` vs
        //     `node_slots.len() + 1` to distinguish modes. ───
        let offset_key = |s: u32| -> u32 {
            if has_cross_segment {
                s
            } else {
                s - tail_lo
            }
        };

        let mut out_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(offsets_len);
        let mut out_edges: MmapOrVec<CsrEdge> = MmapOrVec::with_capacity(n_edges);
        let mut cursor = 0usize;
        for k in 0..(offsets_len - 1) as u32 {
            out_offsets.push(cursor as u64);
            while cursor < n_edges && offset_key(seal_edges[cursor].src_global) == k {
                let e = &seal_edges[cursor];
                out_edges.push(CsrEdge {
                    peer: e.tgt_global,
                    edge_idx: cursor as u32,
                });
                cursor += 1;
            }
        }
        out_offsets.push(cursor as u64);

        // ─── in_offsets / in_edges: mirror keyed by target. ───
        let mut by_target: Vec<(u32, u32)> = seal_edges
            .iter()
            .enumerate()
            .map(|(orig_idx, e)| (e.tgt_global, orig_idx as u32))
            .collect();
        by_target.sort_by_key(|(t, _)| *t);

        let mut in_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(offsets_len);
        let mut in_edges: MmapOrVec<CsrEdge> = MmapOrVec::with_capacity(n_edges);
        let mut tcursor = 0usize;
        for k in 0..(offsets_len - 1) as u32 {
            in_offsets.push(tcursor as u64);
            while tcursor < n_edges && offset_key(by_target[tcursor].0) == k {
                let (_, orig_idx) = by_target[tcursor];
                let src_peer = seal_edges[orig_idx as usize].src_global;
                in_edges.push(CsrEdge {
                    peer: src_peer,
                    edge_idx: orig_idx,
                });
                tcursor += 1;
            }
        }
        in_offsets.push(tcursor as u64);

        // ─── Persist core CSR to disk. ───
        node_slots.save_to_file(&seg_dir.join("node_slots.bin"))?;
        out_offsets.save_to_file(&seg_dir.join("out_offsets.bin"))?;
        out_edges.save_to_file(&seg_dir.join("out_edges.bin"))?;
        in_offsets.save_to_file(&seg_dir.join("in_offsets.bin"))?;
        in_edges.save_to_file(&seg_dir.join("in_edges.bin"))?;
        edge_endpoints.save_to_file(&seg_dir.join("edge_endpoints.bin"))?;

        // ─── Persist auxiliary indexes (phase 5). ───
        //
        // Without these, typed-edge matches, peer-count aggregates,
        // and `edge_weight()` all return incomplete results on the
        // sealed edges — phase 8's seal_to_new_segment commit doc
        // documented this as a deferred limitation. Now we write the
        // per-segment `conn_type_index_*`, `peer_count_*`, and flush
        // the global `edge_properties` overlay so the sealed edges'
        // properties survive into the segment's store.
        //
        // Segment-local inputs: the just-built node_slots / out_offsets
        // / out_edges / edge_endpoints vectors above. All are
        // `MmapOrVec::Heap` — no file handles — so the builders don't
        // race with anything mmap'd under `self.data_dir`.
        // conn_type_index is keyed by offset-array index (segment-local
        // in segment-local mode, global in full-range mode). The
        // builder's `node_bound` argument must match the offsets
        // indexing so it walks the full offset range.
        let _ = super::builder::write_conn_type_index(
            &out_offsets,
            &out_edges,
            &edge_endpoints,
            offsets_len - 1,
            &seg_dir,
            false,
        );
        let _ = super::builder::write_peer_count_histogram(
            &edge_endpoints,
            0,
            n_edges,
            &seg_dir,
            false,
        );

        // Flush the edge_properties overlay to seg_0's base store. The
        // overlay currently holds props for the sealed edges (keyed by
        // their original global edge_idx). `save_to` absorbs the
        // overlay into seg_0's edge_prop_* files, which after phase 7
        // concat cover every segment's edges (since concat preserves
        // global edge_idx). Sealed edges' properties survive reload.
        let upper = self.next_edge_idx;
        self.edge_properties.save_to(&self.data_dir, upper)?;

        // ─── Manifest bookkeeping. ───
        use super::segment_summary::SegmentSummary;
        let mut summary = SegmentSummary::new(next_id, tail_lo);
        summary.node_id_hi = tail_hi;
        summary.edge_count = n_edges as u64;
        // Connection types touched by this segment's edges.
        for e in &seal_edges {
            summary.conn_types.insert(e.conn_type);
        }
        // Node-type counts from the tail's slots.
        for i in 0..tail_len {
            let ns = self.node_slots.get(tail_lo as usize + i);
            if !ns.is_alive() {
                continue;
            }
            *summary.node_type_counts.entry(ns.node_type).or_insert(0) += 1;
        }
        // (indexed_prop_ranges stays empty for the new segment — phase
        // 5's cache+scan populates for seg 0's indexes only, and
        // per-segment indexes are phase 9.)

        self.segment_manifest.append(summary);
        self.segment_manifest.save_to(root)?;

        // ─── Reconcile seg_0's on-disk files with the new layout. ───
        //
        // self.{node_slots, out_offsets, in_offsets, edge_endpoints}
        // all grew during the post-save adds — their files are at
        // seg_0/... and now span past seg_0's logical extent (the
        // tail entries belong in seg_NNN/, which was just written).
        // Truncate each seg_0 file to its pre-tail size and swap
        // self's backing to heap-owned copies that still hold the
        // combined view for in-memory queries. On reload, seg_0 reads
        // cleanly via file-size inference, then concat stitches seg_NNN.
        //
        // `out_edges` / `in_edges` were NOT pushed during the overflow
        // adds (add_edge's post-CSR path writes overflow_out/in +
        // edge_endpoints only), so their files stay at seg_0's size —
        // no reconcile needed.
        let sealed_edge_count = n_edges;
        let seg0_next_edge_idx = self.next_edge_idx as usize - sealed_edge_count;
        reconcile_seg0_csr::<DiskNodeSlot>(&mut self.node_slots, tail_lo as usize)?;
        reconcile_seg0_csr::<u64>(&mut self.out_offsets, tail_lo as usize + 1)?;
        reconcile_seg0_csr::<u64>(&mut self.in_offsets, tail_lo as usize + 1)?;
        reconcile_seg0_csr::<EdgeEndpoints>(&mut self.edge_endpoints, seg0_next_edge_idx)?;

        // ─── Clear consumed overflow + advance watermark. ───
        //
        // All validated overflow edges belong to this seal (their
        // source and target are both >= tail_lo). Drop them in-memory;
        // the persisted CSR in seg_NNN/ is now the source of truth.
        self.overflow_out.clear();
        self.overflow_in.clear();
        self.sealed_nodes_bound = tail_hi;

        // Persist the updated metadata (watermark, manifest presence)
        // at the root. `write_metadata_to` reads edge-property meta
        // from `self.data_dir` — seg 0's subdir — which is the right
        // behaviour here since seal_to_new_segment doesn't rewrite
        // edge_properties.
        let edge_props_meta = EdgePropertyStore::meta_for(&self.data_dir);
        self.write_metadata_to(root, edge_props_meta)?;

        Ok(next_id)
    }

    /// Load a disk graph from a directory.
    /// Raw .bin files are mmap'd directly from the graph dir (no temp dir needed).
    /// Also supports legacy .bin.zst files (decompressed to temp dir).
    /// Returns `(DiskGraph, temp_dir)` — temp_dir may be empty if no decompression needed.
    ///
    /// `interner` is only mutated when loading a legacy format=0 graph
    /// whose `edge_properties.bin.zst` stores InternedKey as strings; the
    /// new columnar format stores raw u64 hashes and never touches it.
    pub fn load_from_dir(
        dir: &Path,
        interner: &mut crate::graph::schema::StringInterner,
    ) -> std::io::Result<(Self, PathBuf)> {
        use crate::graph::io::load_timing::{log_stage, stage_timer};

        let t = stage_timer();
        let meta_str = std::fs::read_to_string(dir.join("disk_graph_meta.json"))?;
        let meta: DiskGraphMeta = serde_json::from_str(&meta_str).map_err(std::io::Error::other)?;
        log_stage("dg.meta_parse", t);

        // PR1 phase 4: CSR binaries live under seg_NNN/ when the graph
        // was written with csr_layout_version >= 1. Legacy .kgl directories
        // (version=0, the serde default) keep the flat layout.
        //
        // Phase 6 added `enumerate_segment_dirs`; phase 7 wires the
        // multi-segment concat read path through it. For csr_layout
        // version >= 1 the CSR dir choice (single-segment) or the
        // concat'd combined arrays (multi-segment) come out of this
        // block. Writes are still single-segment today (phase 8 will
        // seal overflow into additional segments), so the N>1 branch
        // is only exercised by unit tests on `concat_segment_csrs`
        // until then.
        //
        // Auxiliary per-segment data (conn_type_index_*, peer_count_*,
        // edge_properties, column_stores, per-(type,prop) property
        // indexes) is still loaded from segment 0 only in the N>1
        // branch — that's a documented limitation on `SegmentCsr`
        // pending phase 8. No code path produces an N>1 graph today,
        // so no running workload sees it.
        // Temp dir for legacy .zst decompression (only created if needed).
        // Lives inside graph dir so no external temp space required.
        let temp_dir = dir.join("_zst_cache");

        let t = stage_timer();
        let (csr_dir, segment_csr): (PathBuf, SegmentCsr) = if meta.csr_layout_version >= 1 {
            let segs = enumerate_segment_dirs(dir);
            match segs.len() {
                0 => {
                    return Err(std::io::Error::other(format!(
                        "csr_layout_version={} but no seg_NNN/ directory found under {}",
                        meta.csr_layout_version,
                        dir.display()
                    )));
                }
                1 => {
                    // Single-segment: stay on the direct mmap path using
                    // the graph-level `meta.*_len` values. No allocation,
                    // zero overhead vs pre-phase-7 load.
                    let seg_dir = segs.into_iter().next().unwrap().1;
                    let csr = SegmentCsr {
                        node_slots: load_raw_or_zst(
                            &seg_dir.join("node_slots"),
                            meta.node_slots_len,
                            &temp_dir,
                        )?,
                        out_offsets: load_raw_or_zst(
                            &seg_dir.join("out_offsets"),
                            meta.out_offsets_len,
                            &temp_dir,
                        )?,
                        out_edges: load_raw_or_zst(
                            &seg_dir.join("out_edges"),
                            meta.out_edges_len,
                            &temp_dir,
                        )?,
                        in_offsets: load_raw_or_zst(
                            &seg_dir.join("in_offsets"),
                            meta.in_offsets_len,
                            &temp_dir,
                        )?,
                        in_edges: load_raw_or_zst(
                            &seg_dir.join("in_edges"),
                            meta.in_edges_len,
                            &temp_dir,
                        )?,
                        edge_endpoints: load_raw_or_zst(
                            &seg_dir.join("edge_endpoints"),
                            meta.edge_endpoints_len,
                            &temp_dir,
                        )?,
                        conn_type_index_types: load_raw_or_zst_optional(
                            &seg_dir.join("conn_type_index_types"),
                        ),
                        conn_type_index_offsets: load_raw_or_zst_optional(
                            &seg_dir.join("conn_type_index_offsets"),
                        ),
                        conn_type_index_sources: load_raw_or_zst_optional(
                            &seg_dir.join("conn_type_index_sources"),
                        ),
                        peer_count_types: load_raw_or_zst_optional(
                            &seg_dir.join("peer_count_types"),
                        ),
                        peer_count_offsets: load_raw_or_zst_optional(
                            &seg_dir.join("peer_count_offsets"),
                        ),
                        peer_count_entries: load_raw_or_zst_optional(
                            &seg_dir.join("peer_count_entries"),
                        ),
                    };
                    (seg_dir, csr)
                }
                _ => {
                    // Multi-segment: load each segment via the file-size-
                    // inferring loader, then concat. The first segment's
                    // path doubles as `data_dir` — that's where the
                    // auxiliary-indexes limitation points.
                    let mut loaded = Vec::with_capacity(segs.len());
                    let first_dir = segs[0].1.clone();
                    for (_, sdir) in &segs {
                        loaded.push(SegmentCsr::load_from(sdir, &temp_dir)?);
                    }
                    let csr = concat_segment_csrs(loaded);
                    (first_dir, csr)
                }
            }
        } else {
            // Legacy flat layout: load from root as one segment, using
            // meta's *_len values. Same code as phase 6 once unwrapped.
            let csr = SegmentCsr {
                node_slots: load_raw_or_zst(
                    &dir.join("node_slots"),
                    meta.node_slots_len,
                    &temp_dir,
                )?,
                out_offsets: load_raw_or_zst(
                    &dir.join("out_offsets"),
                    meta.out_offsets_len,
                    &temp_dir,
                )?,
                out_edges: load_raw_or_zst(&dir.join("out_edges"), meta.out_edges_len, &temp_dir)?,
                in_offsets: load_raw_or_zst(
                    &dir.join("in_offsets"),
                    meta.in_offsets_len,
                    &temp_dir,
                )?,
                in_edges: load_raw_or_zst(&dir.join("in_edges"), meta.in_edges_len, &temp_dir)?,
                edge_endpoints: load_raw_or_zst(
                    &dir.join("edge_endpoints"),
                    meta.edge_endpoints_len,
                    &temp_dir,
                )?,
                conn_type_index_types: load_raw_or_zst_optional(&dir.join("conn_type_index_types")),
                conn_type_index_offsets: load_raw_or_zst_optional(
                    &dir.join("conn_type_index_offsets"),
                ),
                conn_type_index_sources: load_raw_or_zst_optional(
                    &dir.join("conn_type_index_sources"),
                ),
                peer_count_types: load_raw_or_zst_optional(&dir.join("peer_count_types")),
                peer_count_offsets: load_raw_or_zst_optional(&dir.join("peer_count_offsets")),
                peer_count_entries: load_raw_or_zst_optional(&dir.join("peer_count_entries")),
            };
            (dir.to_path_buf(), csr)
        };
        log_stage("dg.segment_csr", t);

        let SegmentCsr {
            node_slots,
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints,
            conn_type_index_types,
            conn_type_index_offsets,
            conn_type_index_sources,
            peer_count_types,
            peer_count_offsets,
            peer_count_entries,
        } = segment_csr;

        // Load edge properties — columnar (format=1) or legacy (format=0).
        // In the segmented layout the files live alongside the CSR.
        let t = stage_timer();
        let edge_properties = EdgePropertyStore::load_from(
            &csr_dir,
            meta.edge_properties_format,
            meta.edge_properties_meta,
            interner,
        )?;
        log_stage("dg.edge_properties", t);

        // Load overflow edges (kept at the graph root; orthogonal to segments)
        let t = stage_timer();
        let (overflow_out, overflow_in) = if dir.join("overflow_edges.bin.zst").exists() {
            let compressed = std::fs::read(dir.join("overflow_edges.bin.zst"))?;
            let bytes = zstd::decode_all(compressed.as_slice()).map_err(std::io::Error::other)?;
            bincode::deserialize(&bytes).map_err(std::io::Error::other)?
        } else {
            (HashMap::new(), HashMap::new())
        };
        log_stage("dg.overflow_edges", t);

        let t = stage_timer();
        let segment_manifest =
            super::segment_summary::SegmentManifest::load_from(dir).unwrap_or_default();
        log_stage("dg.segment_manifest", t);

        // PR1 phase 8: serde-defaulted to 0 on pre-phase-8 graphs.
        // Their `seg_000` already accounts for every node, so bump the
        // watermark to `node_count` here. Without this, a re-save calls
        // `seal_to_new_segment` with `tail_lo=0` and `tail_hi=node_count`,
        // which writes a fresh empty `seg_001` AND truncates seg_000's
        // `out_offsets.bin` / `in_offsets.bin` via `reconcile_seg0_csr`
        // — corrupting the graph. Fresh phase-8+ graphs persist the
        // correct watermark, so the bump is a no-op for them.
        let sealed_nodes_bound = if meta.sealed_nodes_bound == 0
            && !segment_manifest.is_empty()
            && meta.node_count > 0
        {
            meta.node_count as u32
        } else {
            meta.sealed_nodes_bound
        };

        Ok((
            DiskGraph {
                node_slots,
                node_count: meta.node_count,
                free_node_slots: meta.free_node_slots,
                node_arena: std::sync::Mutex::new(Vec::with_capacity(1024)),
                column_stores: HashMap::new(),
                out_offsets,
                out_edges,
                in_offsets,
                in_edges,
                edge_endpoints,
                edge_count: meta.edge_count,
                next_edge_idx: meta.next_edge_idx,
                edge_properties,
                edge_arena: std::sync::Mutex::new(Vec::with_capacity(1024)),
                edge_mut_cache: HashMap::new(),
                node_mut_cache: HashMap::new(),
                pending_edges: UnsafeCell::new(MmapOrVec::new()),
                overflow_out,
                overflow_in,
                free_edge_slots: meta.free_edge_slots,
                data_dir: csr_dir.clone(),
                metadata_dirty: false,
                csr_sorted_by_type: meta.csr_sorted_by_type,
                defer_csr: false,
                edge_type_counts_raw: None,
                // Auxiliary indexes — single-segment graphs read them
                // from `csr_dir` directly via `SegmentCsr::load_from`
                // (or the manual constructor in the N==1 branch above);
                // multi-segment graphs get the merged view from
                // `concat_segment_csrs`.
                conn_type_index_types,
                conn_type_index_offsets,
                conn_type_index_sources,
                peer_count_types,
                peer_count_offsets,
                peer_count_entries,
                has_tombstones: meta.has_tombstones,
                property_indexes: std::sync::RwLock::new(HashMap::new()),
                global_indexes: std::sync::RwLock::new(HashMap::new()),
                // Legacy .kgl directories have no seg_manifest.json;
                // load_from returns an empty manifest which subsequent
                // PR1 phases treat as "pre-segmented, don't prune".
                segment_manifest,
                sealed_nodes_bound,
            },
            temp_dir,
        ))
    }
}

// ============================================================================
// Compression helpers
// ============================================================================

/// Write a MmapOrVec as a zstd-compressed file.
/// Load a binary array: try raw `.bin` first (direct mmap, no temp dir),
/// fall back to `.bin.zst` (decompress to temp dir, then mmap).
fn load_raw_or_zst<T: Copy + Default + 'static>(
    base_path: &Path,
    len: usize,
    temp_dir: &Path,
) -> std::io::Result<MmapOrVec<T>> {
    let raw_path = base_path.with_extension("bin");
    if raw_path.exists() && len > 0 {
        return MmapOrVec::load_mapped(&raw_path, len);
    }
    let zst_path = base_path.with_extension("bin.zst");
    if zst_path.exists() && len > 0 {
        std::fs::create_dir_all(temp_dir)?;
        return load_compressed(&zst_path, len, temp_dir);
    }
    Ok(MmapOrVec::new())
}

/// Load a raw .bin file if it exists, otherwise return empty MmapOrVec.
/// Used for optional supplementary files (e.g., connection-type inverted index).
fn load_raw_or_zst_optional<T: Copy + Default + 'static>(base_path: &Path) -> MmapOrVec<T> {
    let raw_path = base_path.with_extension("bin");
    if raw_path.exists() {
        let file_len = std::fs::metadata(&raw_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let elem_size = std::mem::size_of::<T>();
        if file_len > 0 && elem_size > 0 {
            let len = file_len / elem_size;
            return MmapOrVec::load_mapped(&raw_path, len).unwrap_or_else(|_| MmapOrVec::new());
        }
    }
    MmapOrVec::new()
}

/// Load a zstd-compressed file, decompress to temp file, and mmap it.
/// Used only for loading legacy .bin.zst files from older graph format.
fn load_compressed<T: Copy + Default + 'static>(
    path: &Path,
    len: usize,
    temp_dir: &Path,
) -> std::io::Result<MmapOrVec<T>> {
    if !path.exists() || len == 0 {
        return Ok(MmapOrVec::new());
    }
    let compressed = std::fs::read(path)?;
    let raw = zstd::decode_all(compressed.as_slice())?;

    // Write decompressed data to temp file and mmap
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("data")
        .trim_end_matches(".zst");
    let temp_path = temp_dir.join(file_name);
    std::fs::write(&temp_path, &raw)?;
    MmapOrVec::load_mapped(&temp_path, len)
}

// ============================================================================
// Multi-segment CSR (PR1 phase 7)
// ============================================================================

/// One segment's core CSR arrays, loaded from its subdirectory. Used
/// only when a graph spans multiple `seg_NNN/` dirs — single-segment
/// graphs continue on the direct mmap path in [`DiskGraph::load_from_dir`]
/// for zero-overhead compatibility with every existing `.kgl` directory.
///
/// The arrays bundled here are the CSR backbone:
///
///   - `node_slots`: one `DiskNodeSlot` per node in this segment (the
///     segment owns a disjoint node-id range reported in its
///     `SegmentSummary`).
///   - `out_offsets` / `in_offsets`: CSR offsets into `out_edges` /
///     `in_edges`, indexed by local node position inside the segment.
///     Length = `node_slots.len() + 1`.
///   - `out_edges` / `in_edges`: `CsrEdge` arrays. Each entry's
///     `edge_idx` is **segment-local** (`0..edge_endpoints.len()`),
///     so concat shifts them onto combined edge_endpoints.
///   - `edge_endpoints`: `EdgeEndpoints` array, one per edge recorded
///     in this segment. `source` / `target` store *global* node ids
///     (unchanged by concat).
///
/// The auxiliary inverted indexes (`conn_type_index_*`, `peer_count_*`)
/// and the `edge_properties` / `column_stores` / per-type property
/// indexes are **not** bundled here — they remain loaded from segment 0
/// only. That's a known limitation pending the phase 8 multi-segment
/// write path, which will exercise it end-to-end. No current code
/// produces multi-segment graphs, so no existing workload sees the
/// limitation today.
pub(crate) struct SegmentCsr {
    pub(crate) node_slots: MmapOrVec<super::csr::DiskNodeSlot>,
    pub(crate) out_offsets: MmapOrVec<u64>,
    pub(crate) out_edges: MmapOrVec<super::csr::CsrEdge>,
    pub(crate) in_offsets: MmapOrVec<u64>,
    pub(crate) in_edges: MmapOrVec<super::csr::CsrEdge>,
    pub(crate) edge_endpoints: MmapOrVec<super::csr::EdgeEndpoints>,
    // Phase 5: per-segment auxiliary indexes. Each segment carries its
    // own inverted indexes; `concat_segment_csrs` merges them at load
    // time.
    pub(crate) conn_type_index_types: MmapOrVec<u64>,
    pub(crate) conn_type_index_offsets: MmapOrVec<u64>,
    pub(crate) conn_type_index_sources: MmapOrVec<u32>,
    pub(crate) peer_count_types: MmapOrVec<u64>,
    pub(crate) peer_count_offsets: MmapOrVec<u64>,
    pub(crate) peer_count_entries: MmapOrVec<u32>,
}

impl SegmentCsr {
    /// Load the core CSR arrays from `csr_dir`, inferring each array's
    /// length from the file size (matches `load_raw_or_zst_optional`).
    /// Legacy `.bin.zst` fallback uses `temp_dir` for the decompressed
    /// staging files.
    pub(crate) fn load_from(csr_dir: &Path, temp_dir: &Path) -> std::io::Result<Self> {
        Ok(SegmentCsr {
            node_slots: load_with_inferred_len(&csr_dir.join("node_slots"), temp_dir)?,
            out_offsets: load_with_inferred_len(&csr_dir.join("out_offsets"), temp_dir)?,
            out_edges: load_with_inferred_len(&csr_dir.join("out_edges"), temp_dir)?,
            in_offsets: load_with_inferred_len(&csr_dir.join("in_offsets"), temp_dir)?,
            in_edges: load_with_inferred_len(&csr_dir.join("in_edges"), temp_dir)?,
            edge_endpoints: load_with_inferred_len(&csr_dir.join("edge_endpoints"), temp_dir)?,
            // Auxiliary files are optional — older segments (seg_000
            // written before phase 5 wired auxiliary writes into seal)
            // may lack them. `load_raw_or_zst_optional` already returns
            // an empty MmapOrVec when the file is absent.
            conn_type_index_types: load_raw_or_zst_optional(&csr_dir.join("conn_type_index_types")),
            conn_type_index_offsets: load_raw_or_zst_optional(
                &csr_dir.join("conn_type_index_offsets"),
            ),
            conn_type_index_sources: load_raw_or_zst_optional(
                &csr_dir.join("conn_type_index_sources"),
            ),
            peer_count_types: load_raw_or_zst_optional(&csr_dir.join("peer_count_types")),
            peer_count_offsets: load_raw_or_zst_optional(&csr_dir.join("peer_count_offsets")),
            peer_count_entries: load_raw_or_zst_optional(&csr_dir.join("peer_count_entries")),
        })
    }
}

/// Post-seal cleanup helper — see `DiskGraph::seal_to_new_segment`.
///
/// `field` is one of seg_0's CSR mmap-backed arrays that grew past its
/// seg_0 logical size during the post-save add batch (e.g.,
/// `self.node_slots` got pushes from `add_node`, `self.edge_endpoints`
/// got pushes from `add_edge`). We need three things at once:
///
///  1. The on-disk file trimmed to exactly `seg0_len` elements — so
///     the next reload reads seg_0 with the right element count.
///  2. The in-memory data to keep all current entries (seg_0 + tail)
///     so queries between seal and drop still see the combined graph.
///  3. The file handle released, so `set_len` doesn't race an
///     existing mmap.
///
/// Simplest way to get all three: snapshot the current contents into
/// a Vec, replace `field` with a fresh heap-backed MmapOrVec holding
/// that Vec, then reopen the file briefly just to `set_len`.
fn reconcile_seg0_csr<T: Copy + Default + 'static>(
    field: &mut MmapOrVec<T>,
    seg0_len: usize,
) -> std::io::Result<()> {
    let all = field.to_vec();
    let path = field.file_path().map(PathBuf::from);
    // Replace before truncate so the old mmap is dropped (releases the
    // file) before we `set_len` on the path.
    *field = MmapOrVec::from_vec(all);
    if let Some(p) = path {
        let f = std::fs::OpenOptions::new().write(true).open(&p)?;
        f.set_len((seg0_len * std::mem::size_of::<T>()) as u64)?;
    }
    Ok(())
}

/// Like [`load_raw_or_zst`] but derives the element count from the file
/// size on disk rather than from a pre-known length. Used in the multi-
/// segment load path, where `DiskGraphMeta`'s `*_len` fields describe
/// the *graph-level* concat total, not any one segment.
fn load_with_inferred_len<T: Copy + Default + 'static>(
    base_path: &Path,
    temp_dir: &Path,
) -> std::io::Result<MmapOrVec<T>> {
    let elem = std::mem::size_of::<T>();
    let raw_path = base_path.with_extension("bin");
    if raw_path.exists() && elem > 0 {
        let bytes = std::fs::metadata(&raw_path)?.len() as usize;
        let len = bytes / elem;
        if len > 0 {
            return MmapOrVec::load_mapped(&raw_path, len);
        }
    }
    let zst_path = base_path.with_extension("bin.zst");
    if zst_path.exists() && elem > 0 {
        // Legacy path: zstd stream doesn't carry the element count in
        // metadata, so we decompress to a temp file and infer from its
        // size. Matches `load_raw_or_zst`'s `load_compressed` flow,
        // minus the advance length check.
        std::fs::create_dir_all(temp_dir)?;
        let compressed = std::fs::read(&zst_path)?;
        let raw = zstd::decode_all(compressed.as_slice())?;
        let file_name = zst_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("data")
            .trim_end_matches(".zst");
        let temp_path = temp_dir.join(file_name);
        std::fs::write(&temp_path, &raw)?;
        let len = raw.len() / elem;
        if len > 0 {
            return MmapOrVec::load_mapped(&temp_path, len);
        }
    }
    Ok(MmapOrVec::new())
}

/// Combine per-segment CSR arrays into a single unified CSR by
/// concatenating node_slots / edge_endpoints, stitching offsets, and
/// shifting each segment's `edge_idx` values onto the combined
/// `edge_endpoints` numbering.
///
/// Single-segment input is returned as-is — the `Vec<SegmentCsr>` is
/// popped once and no allocation happens, so the N==1 case pays
/// essentially nothing beyond the function call.
///
/// The returned `SegmentCsr` is always `MmapOrVec::Vec`-backed — it
/// does not touch the filesystem. A graph-level page cache concat-to-
/// disk would be a future win; for now the in-memory combined CSR is
/// what the read path sees.
///
/// Assumptions on inputs (documented so the phase-8 writer obeys them):
///   1. Segments are provided in manifest order (ascending
///      `segment_id`), covering contiguous disjoint node-id ranges
///      `[0, n_0) + [n_0, n_0 + n_1) + ...`. The caller
///      ([`DiskGraph::load_from_dir`]) preserves `enumerate_segment_dirs`
///      ordering.
///   2. Each segment's `out_offsets` / `in_offsets` is segment-local:
///      `out_offsets[0] == 0`, `out_offsets[last] == out_edges.len()`.
///   3. Each segment's `CsrEdge::edge_idx` values are segment-local
///      (`0..edge_endpoints.len()`).
///   4. `EdgeEndpoints::{source, target}` hold *global* node ids and
///      are never rewritten by concat.
///
/// Violations produce a garbage combined CSR; the assumptions are
/// phase 7's contract with phase 8's writer.
pub(crate) fn concat_segment_csrs(mut segments: Vec<SegmentCsr>) -> SegmentCsr {
    use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints};
    match segments.len() {
        0 => SegmentCsr {
            node_slots: MmapOrVec::new(),
            out_offsets: MmapOrVec::new(),
            out_edges: MmapOrVec::new(),
            in_offsets: MmapOrVec::new(),
            in_edges: MmapOrVec::new(),
            edge_endpoints: MmapOrVec::new(),
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
        },
        1 => segments.pop().unwrap(),
        _ => {
            // Pre-size everything.
            let total_nodes: usize = segments.iter().map(|s| s.node_slots.len()).sum();
            let total_out_edges: usize = segments.iter().map(|s| s.out_edges.len()).sum();
            let total_in_edges: usize = segments.iter().map(|s| s.in_edges.len()).sum();
            let total_endpoints: usize = segments.iter().map(|s| s.edge_endpoints.len()).sum();

            let mut node_slots: MmapOrVec<DiskNodeSlot> = MmapOrVec::with_capacity(total_nodes);
            let mut edge_endpoints: MmapOrVec<EdgeEndpoints> =
                MmapOrVec::with_capacity(total_endpoints);
            let mut out_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(total_nodes + 1);
            let mut out_edges: MmapOrVec<CsrEdge> = MmapOrVec::with_capacity(total_out_edges);
            let mut in_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(total_nodes + 1);
            let mut in_edges: MmapOrVec<CsrEdge> = MmapOrVec::with_capacity(total_in_edges);

            // Per-segment metadata for the per-node walk below.
            //   node_lo[k]..node_hi[k]   : combined-index range owned by segment k
            //   endpoint_base[k]         : edge_idx shift for segment k's CsrEdges
            //   is_full_range[k]         : out_offsets covers all nodes (phase 7
            //                              cross-segment seal) vs just its own
            let mut node_lo: Vec<usize> = Vec::with_capacity(segments.len());
            let mut node_hi: Vec<usize> = Vec::with_capacity(segments.len());
            let mut endpoint_base: Vec<u32> = Vec::with_capacity(segments.len());
            let mut is_full: Vec<bool> = Vec::with_capacity(segments.len());
            let mut node_cursor = 0usize;
            let mut ep_cursor: u32 = 0;
            for seg in &segments {
                node_lo.push(node_cursor);
                node_cursor += seg.node_slots.len();
                node_hi.push(node_cursor);
                endpoint_base.push(ep_cursor);
                ep_cursor += seg.edge_endpoints.len() as u32;
                // Full-range segments have an offset entry per GLOBAL
                // node (produced by phase 7's cross-segment seal). The
                // total node count isn't known inside the writer at
                // seal time, but it's always ≥ node_slots.len(); so
                // `out_offsets.len() > node_slots.len() + 1` uniquely
                // signals full-range here.
                is_full.push(seg.out_offsets.len() > seg.node_slots.len() + 1);
            }

            // Node slots + edge endpoints: straight concat (unchanged from
            // phase 7). Segment ownership of nodes is the same; only the
            // edge-index mapping differs under full-range.
            for seg in &segments {
                for i in 0..seg.node_slots.len() {
                    node_slots.push(seg.node_slots.get(i));
                }
                for i in 0..seg.edge_endpoints.len() {
                    edge_endpoints.push(seg.edge_endpoints.get(i));
                }
            }

            // Walk every combined node id and UNION each segment's
            // out_edges / in_edges contributions for that node. Per
            // node, a segment contributes when:
            //   - segment-local & node id in [node_lo, node_hi)  → key = gid - node_lo
            //   - full-range                                       → key = gid  (cap: offsets len)
            out_offsets.push(0);
            in_offsets.push(0);
            for gid in 0..total_nodes {
                for (k, seg) in segments.iter().enumerate() {
                    let key: Option<usize> = if is_full[k] {
                        if gid + 1 < seg.out_offsets.len() {
                            Some(gid)
                        } else {
                            None
                        }
                    } else if gid >= node_lo[k] && gid < node_hi[k] {
                        Some(gid - node_lo[k])
                    } else {
                        None
                    };
                    if let Some(key) = key {
                        let start = seg.out_offsets.get(key) as usize;
                        let end = seg.out_offsets.get(key + 1) as usize;
                        for i in start..end {
                            let mut e = seg.out_edges.get(i);
                            e.edge_idx = e.edge_idx.wrapping_add(endpoint_base[k]);
                            out_edges.push(e);
                        }
                    }
                }
                out_offsets.push(out_edges.len() as u64);

                for (k, seg) in segments.iter().enumerate() {
                    let key: Option<usize> = if is_full[k] {
                        if gid + 1 < seg.in_offsets.len() {
                            Some(gid)
                        } else {
                            None
                        }
                    } else if gid >= node_lo[k] && gid < node_hi[k] {
                        Some(gid - node_lo[k])
                    } else {
                        None
                    };
                    if let Some(key) = key {
                        let start = seg.in_offsets.get(key) as usize;
                        let end = seg.in_offsets.get(key + 1) as usize;
                        for i in start..end {
                            let mut e = seg.in_edges.get(i);
                            e.edge_idx = e.edge_idx.wrapping_add(endpoint_base[k]);
                            in_edges.push(e);
                        }
                    }
                }
                in_offsets.push(in_edges.len() as u64);
            }

            // ─── Phase 5: auxiliary index merge ───────────────────────────
            //
            // conn_type_index — per-type source-list union across segments.
            // Each segment's source list is globally sorted (segments own
            // disjoint node ranges); concatenating segments in manifest
            // order preserves ascending global order per type.
            let (cti_types, cti_offsets, cti_sources) = merge_conn_type_index(&segments);

            // peer_count — sum counts per (type, peer) across segments.
            let (pc_types, pc_offsets, pc_entries) = merge_peer_count_histogram(&segments);

            SegmentCsr {
                node_slots,
                out_offsets,
                out_edges,
                in_offsets,
                in_edges,
                edge_endpoints,
                conn_type_index_types: cti_types,
                conn_type_index_offsets: cti_offsets,
                conn_type_index_sources: cti_sources,
                peer_count_types: pc_types,
                peer_count_offsets: pc_offsets,
                peer_count_entries: pc_entries,
            }
        }
    }
}

/// Merge per-segment `conn_type_index_*` into a combined index.
/// For each connection type, the combined sources list is the
/// concatenation of per-segment sources lists (already globally sorted
/// because segments own disjoint node ranges and per-segment lists are
/// locally sorted). Types are unioned and sorted ascending.
fn merge_conn_type_index(
    segments: &[SegmentCsr],
) -> (MmapOrVec<u64>, MmapOrVec<u64>, MmapOrVec<u32>) {
    use std::collections::BTreeMap;
    // Per-segment source-id shift: segment-local seals write
    // `conn_type_index_sources` using offset-array indices
    // (0..tail_len), since the writer walks `out_offsets` which is
    // indexed locally. Add `node_lo` for those segments to recover the
    // global node id. Full-range seals already store global ids so
    // their shift is 0.
    let mut node_lo: Vec<u32> = Vec::with_capacity(segments.len());
    let mut is_full: Vec<bool> = Vec::with_capacity(segments.len());
    let mut cursor: u32 = 0;
    for seg in segments {
        node_lo.push(cursor);
        cursor += seg.node_slots.len() as u32;
        is_full.push(seg.out_offsets.len() > seg.node_slots.len() + 1);
    }

    // Walk each segment and accumulate: type → Vec<segment_index>.
    let mut type_to_segs: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (si, seg) in segments.iter().enumerate() {
        for i in 0..seg.conn_type_index_types.len() {
            let t = seg.conn_type_index_types.get(i);
            type_to_segs.entry(t).or_default().push(si);
        }
    }
    let total_sources: usize = segments
        .iter()
        .map(|s| s.conn_type_index_sources.len())
        .sum();
    let mut out_types: MmapOrVec<u64> = MmapOrVec::with_capacity(type_to_segs.len());
    let mut out_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(type_to_segs.len() + 1);
    let mut out_sources: MmapOrVec<u32> = MmapOrVec::with_capacity(total_sources);
    let mut cur_off: u64 = 0;
    for (t, seg_idxs) in &type_to_segs {
        out_types.push(*t);
        out_offsets.push(cur_off);
        for &si in seg_idxs {
            let seg = &segments[si];
            let shift = if is_full[si] { 0 } else { node_lo[si] };
            // Find the type's [start, end) slice inside this segment.
            let n = seg.conn_type_index_types.len();
            // Linear scan — typical segment has ≤ hundreds of types,
            // so a BTreeMap-per-segment isn't worth the setup cost.
            for j in 0..n {
                if seg.conn_type_index_types.get(j) == *t {
                    let start = seg.conn_type_index_offsets.get(j) as usize;
                    let end = seg.conn_type_index_offsets.get(j + 1) as usize;
                    for k in start..end {
                        out_sources.push(seg.conn_type_index_sources.get(k) + shift);
                    }
                    cur_off += (end - start) as u64;
                    break;
                }
            }
        }
    }
    out_offsets.push(cur_off);
    (out_types, out_offsets, out_sources)
}

/// Merge per-segment `peer_count_*` histograms by summing counts for
/// every `(conn_type, peer)` pair that appears in any segment.
fn merge_peer_count_histogram(
    segments: &[SegmentCsr],
) -> (MmapOrVec<u64>, MmapOrVec<u64>, MmapOrVec<u32>) {
    use std::collections::BTreeMap;
    // type → peer → summed count.
    let mut by_type: BTreeMap<u64, BTreeMap<u32, u64>> = BTreeMap::new();
    for seg in segments {
        let n = seg.peer_count_types.len();
        for i in 0..n {
            let t = seg.peer_count_types.get(i);
            let start = seg.peer_count_offsets.get(i) as usize;
            let end = seg.peer_count_offsets.get(i + 1) as usize;
            let type_bucket = by_type.entry(t).or_default();
            // Entries are flat (peer, count) pairs.
            let mut k = start;
            while k < end {
                let peer = seg.peer_count_entries.get(k * 2);
                let count = seg.peer_count_entries.get(k * 2 + 1) as u64;
                *type_bucket.entry(peer).or_insert(0) += count;
                k += 1;
            }
        }
    }
    let mut out_types: MmapOrVec<u64> = MmapOrVec::with_capacity(by_type.len());
    let mut out_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(by_type.len() + 1);
    let mut out_entries: MmapOrVec<u32> = MmapOrVec::new();
    let mut cur_pairs: u64 = 0;
    for (t, peers) in &by_type {
        out_types.push(*t);
        out_offsets.push(cur_pairs);
        for (peer, count) in peers {
            out_entries.push(*peer);
            // u64 count saturates to u32 for the on-disk format; sums
            // across segments in practice fit because per-segment
            // counts are u32 and at most `edge_count`.
            out_entries.push((*count).min(u32::MAX as u64) as u32);
        }
        cur_pairs += peers.len() as u64;
    }
    out_offsets.push(cur_pairs);
    (out_types, out_offsets, out_entries)
}
