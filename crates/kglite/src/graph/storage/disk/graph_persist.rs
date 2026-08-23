//! On-disk persistence for `DiskGraph`: metadata schema, save/load
//! pipelines, multi-segment manifest building, and segment CSR
//! reconciliation.

use crate::graph::schema::InternedKey;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use serde::{Deserialize, Serialize};
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::csr::{CsrEdge, TOMBSTONE_EDGE};
use super::edge_properties::{EdgePropertyStore, EdgePropertyStoreMeta};
use super::graph::{enumerate_segment_dirs, segment_subdir, DiskGraph, CURRENT_CSR_LAYOUT_VERSION};
use super::property_index;
use super::segment_summary::SegmentManifest;

/// Metadata stored alongside the binary files in the disk graph directory.
#[derive(Serialize, Deserialize)]
struct DiskGraphMeta {
    /// Codec for Serde-backed payloads owned by this disk snapshot.
    /// Missing means a retired pre-0.14 snapshot and is rejected on load.
    #[serde(default)]
    serde_codec_version: u8,
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
    /// tombstone check on fresh / read-only graphs. Legacy graphs missing
    /// the field cannot prove they never saw a removal, so they default
    /// to `true`.
    #[serde(default = "default_has_tombstones")]
    has_tombstones: bool,
    /// Edge property storage format. 2 = Postcard columnar mmap base +
    /// overlay (edge_prop_offsets.bin + edge_prop_heap.bin). Serde-defaults
    /// to 0, which `validate_disk_format` reads as a retired pre-0.14
    /// snapshot and rejects.
    #[serde(default)]
    edge_properties_format: u8,
    /// Lengths needed to mmap the columnar edge-property files. Zero for
    /// graphs that don't have any edge properties.
    #[serde(default)]
    edge_properties_meta: EdgePropertyStoreMeta,
    /// CSR-layout version. 0 = legacy flat (all files at graph root).
    /// 1 = segmented (CSR / columns / per-segment indexes live under
    /// `seg_000/`). Defaults to 0 so legacy flat .kgl directories still load.
    #[serde(default)]
    csr_layout_version: u8,
    /// Boundary past which nodes are in the still-mutable tail (not yet
    /// sealed into any segment). `seal_to_new_segment` flushes
    /// `node_slots[sealed_nodes_bound..node_count]` into a new `seg_NNN/`
    /// and advances this. Serde-defaults to zero for older graphs whose
    /// `seg_000` already accounts for everything below `node_count`;
    /// `load_from_dir` bumps those to `node_count` on load.
    #[serde(default)]
    sealed_nodes_bound: u32,
}

fn default_has_tombstones() -> bool {
    true
}

const MAX_DISK_CODEC_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn validate_disk_format(meta: &DiskGraphMeta) -> std::io::Result<crate::serde_codec::CodecVersion> {
    if meta.serde_codec_version != crate::serde_codec::CURRENT_CODEC.tag()
        || meta.edge_properties_format < 2
    {
        return Err(crate::graph::io::file::pre_014_bincode_error(
            "disk graph snapshot",
        ));
    }
    crate::serde_codec::CodecVersion::from_tag(meta.serde_codec_version)
        .map_err(std::io::Error::other)
}

type OverflowEdges = (HashMap<u32, Vec<CsrEdge>>, HashMap<u32, Vec<CsrEdge>>);

/// Persistence shape selected for a target directory.
///
/// Sealing mutates and structurally extends the graph's current root. A
/// generation stage is a different, initially empty directory, so it must
/// receive a complete compact snapshot instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SaveDisposition {
    Seal,
    Rewrite,
}

fn load_overflow_edges(
    dir: &Path,
    codec: crate::serde_codec::CodecVersion,
) -> std::io::Result<OverflowEdges> {
    let path = dir.join("overflow_edges.bin.zst");
    if !path.exists() {
        return Ok((HashMap::new(), HashMap::new()));
    }
    let compressed = std::fs::read(path)?;
    let bytes = zstd::decode_all(compressed.as_slice()).map_err(std::io::Error::other)?;
    crate::serde_codec::decode_exact_with(
        codec,
        &bytes,
        bytes.capacity() as u64,
        crate::serde_codec::DecodeLimits::new(MAX_DISK_CODEC_BYTES, MAX_DISK_CODEC_BYTES),
    )
    .map_err(std::io::Error::other)
}

impl DiskGraph {
    fn save_logical_node_slots(&mut self, path: &Path) -> std::io::Result<()> {
        let logical_len = self.node_slot_len();
        let replace_mapped = self.node_slots.file_path() == Some(path);
        let output = if replace_mapped {
            path.with_extension(format!("bin.stage-{}", std::process::id()))
        } else {
            path.to_path_buf()
        };
        let file = std::fs::File::create(&output)?;
        let mut writer = BufWriter::new(file);
        let mut chunk = Vec::with_capacity(8192);
        for start in (0..self.node_slot_len()).step_by(8192) {
            chunk.clear();
            let end = (start + 8192).min(self.node_slot_len());
            chunk.extend((start..end).map(|index| self.node_slot(index)));
            // SAFETY: DiskNodeSlot is Copy + repr(C), and the persisted disk
            // format is the exact contiguous in-memory representation used by
            // MmapOrVec's existing writer.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr() as *const u8,
                    std::mem::size_of_val(chunk.as_slice()),
                )
            };
            writer.write_all(bytes)?;
        }
        writer.flush()?;
        let file = writer
            .into_inner()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        if replace_mapped {
            // Atomic-publish doctrine (io/file.rs::write_kgl_with): fsync
            // the staged bytes, rename OVER the live target (no
            // remove-first window where NO file exists), then fsync the
            // parent dir so the rename is durable. A crash at any point
            // leaves either the old file or the new one.
            file.sync_all()?;
            drop(file);
            // Drop the live mapping before replacing the file under it.
            self.node_slots = MmapOrVec::new();
            std::fs::rename(&output, path)?;
            if let Some(parent) = path.parent() {
                super::generation::sync_directory(parent)?;
            }
            self.node_slots = MmapOrVec::load_mapped(path, logical_len)?;
            self.node_slot_updates.clear();
            self.appended_node_slots.clear();
        }
        Ok(())
    }

    /// Merge heap-side node slots (`appended_node_slots` +
    /// `node_slot_updates`) into the mapped `node_slots.bin` backing file.
    ///
    /// Bulk loaders (ntriples) add nodes through `add_node`, which appends
    /// to the heap-side overlay without extending the mmap'd file. A build
    /// that finalises *without* an explicit `save()` used to leave
    /// `node_slots.bin` at its initial 1024-slot allocation while
    /// `disk_graph_meta.json` already claimed the full logical length —
    /// making the directory unloadable ("File too small") past 1024 nodes,
    /// and silently loading zeroed (dead) slots below it. Loaders call this
    /// during finalisation so the on-disk file matches the logical state.
    pub(crate) fn flush_node_slots(&mut self) -> std::io::Result<()> {
        if self.appended_node_slots.is_empty() && self.node_slot_updates.is_empty() {
            return Ok(());
        }
        let path = self.active_write_dir().join("node_slots.bin");
        self.save_logical_node_slots(&path)?;
        // `save_logical_node_slots` only swaps in the merged file (and
        // clears the overlay) when node_slots was already mapped at `path`.
        // A heap-resident node_slots wrote the file above but keeps its
        // overlay; reload so the in-memory view matches the published file.
        if self.node_slots.file_path() != Some(path.as_path()) {
            let logical_len = self.node_slot_len();
            self.node_slots = MmapOrVec::load_mapped(&path, logical_len)?;
            self.node_slot_updates.clear();
            self.appended_node_slots.clear();
        }
        Ok(())
    }

    fn save_logical_edge_endpoints(&mut self, path: &Path) -> std::io::Result<()> {
        let logical_len = self.edge_endpoint_len();
        let replace_mapped = self.edge_endpoints.file_path() == Some(path);
        let output = if replace_mapped {
            path.with_extension(format!("bin.stage-{}", std::process::id()))
        } else {
            path.to_path_buf()
        };
        let file = std::fs::File::create(&output)?;
        let mut writer = BufWriter::new(file);
        let mut chunk = Vec::with_capacity(8192);
        for start in (0..self.edge_endpoint_len()).step_by(8192) {
            chunk.clear();
            let end = (start + 8192).min(self.edge_endpoint_len());
            chunk.extend((start..end).map(|index| self.edge_endpoint(index)));
            // SAFETY: EdgeEndpoints is Copy + repr(C), matching MmapOrVec's
            // existing raw persisted representation.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr() as *const u8,
                    std::mem::size_of_val(chunk.as_slice()),
                )
            };
            writer.write_all(bytes)?;
        }
        writer.flush()?;
        let file = writer
            .into_inner()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        if replace_mapped {
            // Same atomic-publish shape as `save_logical_node_slots`
            // above: fsync stage, rename over the target (no remove),
            // fsync parent dir.
            file.sync_all()?;
            drop(file);
            self.edge_endpoints = MmapOrVec::new();
            std::fs::rename(&output, path)?;
            if let Some(parent) = path.parent() {
                super::generation::sync_directory(parent)?;
            }
            self.edge_endpoints = MmapOrVec::load_mapped(path, logical_len)?;
            self.appended_edge_endpoints.clear();
            self.removed_edges.clear();
        }
        Ok(())
    }

    fn copy_persisted_indexes(&self, target: &Path) -> std::io::Result<()> {
        let excluded_names: HashSet<_> = self
            .removed_property_indexes
            .iter()
            .flat_map(|(node_type, property)| {
                property_index::removal_paths(target, node_type, property)
                    .into_iter()
                    .filter_map(|path| path.file_name().map(|name| name.to_owned()))
            })
            .collect();
        let mut sources = vec![self.data_dir.as_path()];
        sources.extend(
            self.parent_workspaces
                .iter()
                .map(|workspace| workspace.segment_dir()),
        );
        if let Some(workspace) = &self.mutation_workspace {
            sources.push(workspace.segment_dir());
        }
        for source in sources {
            if !source.is_dir() || source == target {
                continue;
            }
            for entry in std::fs::read_dir(source)? {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    continue;
                }
                let name = entry.file_name();
                if excluded_names.contains(&name) {
                    continue;
                }
                let name_str = name.to_string_lossy();
                let keep = name_str.starts_with("conn_type_index_")
                    || name_str.starts_with("peer_count_")
                    || name_str.starts_with("property_index_")
                    || name_str.starts_with("global_index_");
                if keep {
                    std::fs::copy(entry.path(), target.join(name))?;
                }
            }
        }
        property_index::validate_v2_bundles(target)
    }

    /// Write metadata JSON to the graph directory, called at the end of a
    /// CSR build. Reads the edge-property file metadata from the current
    /// data_dir so the JSON reflects whatever was last persisted there;
    /// mutations since then live in the overlay until the next explicit
    /// `save_to_dir`.
    pub(crate) fn write_metadata(&self) -> std::io::Result<()> {
        let segment_dir = self.active_write_dir();
        let root = segment_dir.parent().unwrap_or(segment_dir);
        let edge_props_meta = EdgePropertyStore::meta_for(segment_dir);
        self.write_metadata_to(root, edge_props_meta)
    }

    fn write_metadata_to(
        &self,
        dir: &Path,
        edge_props_meta: EdgePropertyStoreMeta,
    ) -> std::io::Result<()> {
        let meta = DiskGraphMeta {
            serde_codec_version: crate::serde_codec::CURRENT_CODEC.tag(),
            node_count: self.node_count,
            node_slots_len: self.node_slot_len(),
            edge_count: self.edge_count,
            next_edge_idx: self.next_edge_idx,
            out_offsets_len: self.out_offsets.len(),
            out_edges_len: self.out_edges.len(),
            in_offsets_len: self.in_offsets.len(),
            in_edges_len: self.in_edges.len(),
            edge_endpoints_len: self.edge_endpoint_len(),
            free_node_slots: self.free_node_slots.clone(),
            free_edge_slots: self.free_edge_slots.clone(),
            csr_sorted_by_type: self.csr_sorted_by_type,
            has_tombstones: self.has_tombstones,
            // Fresh graphs use Postcard columnar slots; anything below 2 is
            // a pre-0.14 snapshot and is rejected on load.
            edge_properties_format: 2,
            edge_properties_meta: edge_props_meta,
            // Fresh saves always emit the segmented layout.
            csr_layout_version: CURRENT_CSR_LAYOUT_VERSION,
            // Persist the watermark so reloads know which nodes already
            // live in sealed segments vs which are tail.
            sealed_nodes_bound: self.sealed_nodes_bound,
        };
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        std::fs::write(dir.join("disk_graph_meta.json"), json)
    }

    /// Build a manifest holding one summary that covers the whole graph.
    /// Used by the compact-rewrite save path, which consolidates every
    /// node into `seg_000`; `seal_to_new_segment` appends its own
    /// per-segment summaries instead.
    ///
    /// conn_types are read from the conn_type_index (built alongside the
    /// CSR), so the summary only reflects types that made it into the
    /// index — typically all of them after a save-time compact.
    ///
    /// `indexed_prop_ranges` comes from the segment's on-disk
    /// `PropertyIndex` files. Only string indexes exist, so every entry
    /// uses `PropRange::StringBloomPlaceholder` — it never prunes, but
    /// registers the `(type_hash, prop_hash)` pair so real bloom filters
    /// can replace it without a manifest schema change.
    fn build_single_segment_manifest(
        &self,
        index_dir: &Path,
    ) -> std::io::Result<super::segment_summary::SegmentManifest> {
        use super::segment_summary::{PropRange, SegmentManifest, SegmentSummary};
        use std::collections::HashSet;
        let mut summary = SegmentSummary::new(0, 0);
        summary.node_id_hi = self.node_count as u32;
        summary.edge_count = self.edge_count as u64;

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
                let ct = self.edge_endpoint(e.edge_idx as usize).connection_type;
                summary.conn_types.insert(ct);
            }
        }

        // `row_count` includes tombstoned rows — a conservative upper
        // bound is fine because the planner uses these only as "any rows
        // of this type?" predicates.
        for (type_key, store) in &self.column_stores {
            summary
                .node_type_counts
                .insert(type_key.as_u64(), store.row_count());
        }

        // Record every (type, prop) index present in the segment.
        // Prefer the in-memory cache — its keys hold the
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
        for (t_hash, p_hash) in property_index::scan_segment_hashes(index_dir)? {
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
        Ok(manifest)
    }

    /// Decide whether `target_dir` can be extended with an incremental sealed
    /// segment or needs a complete rewritten snapshot.
    ///
    /// A seal is valid only against the graph's selected data root: the
    /// implementation reuses its existing segment files and updates their
    /// metadata in place. Save-as targets and immutable generation stages are
    /// distinct directories and therefore always require a rewrite.
    pub(crate) fn save_disposition(&self, target_dir: &Path) -> SaveDisposition {
        let have_prior_save = !self.segment_manifest.is_empty();
        let current_root = self.data_dir.parent().unwrap_or(target_dir);
        let have_unsealed_tail = self.sealed_nodes_bound < self.node_count as u32;
        if have_prior_save && target_dir == current_root && have_unsealed_tail {
            SaveDisposition::Seal
        } else {
            SaveDisposition::Rewrite
        }
    }

    /// Save disk graph state into `target_dir`, either by sealing the
    /// unsealed tail into a new segment or by writing a complete compact
    /// snapshot — see `save_disposition`.
    ///
    /// Takes `&mut self` because the edge-property store may need to drop
    /// its base mmap before overwriting the files (when target_dir equals
    /// the current data_dir).
    pub fn save_to_dir(
        &mut self,
        target_dir: &Path,
        _interner: &crate::graph::schema::StringInterner,
    ) -> std::io::Result<()> {
        // Drain mutation caches: `edge_mut_cache` → `edge_properties`, and
        // `node_mut_cache` → `self.column_stores` via clone-apply-replace.
        // The caller (`DirGraph::save_disk`) mirrors the post-flush Arcs
        // back into its own side immediately after.
        self.clear_arenas();
        std::fs::create_dir_all(target_dir)?;

        // The orchestration layer uses this exact decision before calling us
        // so overflow compaction and the eventual write shape cannot drift.
        if self.save_disposition(target_dir) == SaveDisposition::Seal {
            let _seg_id = self.seal_to_new_segment(target_dir)?;
            return Ok(());
        }

        // CSR binaries live under a per-segment subdirectory; save-as to a
        // different path creates a matching subdir. This compact-rewrite
        // path always consolidates into id 0 — the seal path above is what
        // writes higher ids.
        let csr_target = target_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&csr_target)?;

        // Any seg_NNN > 0 left by a prior seal carries a subset of the
        // now-consolidated state; leaving it on disk makes the next
        // reload's `enumerate_segment_dirs` concat it against the fresh
        // seg_000 — double-counting nodes and edges.
        if csr_target == self.data_dir {
            for (seg_id, seg_path) in enumerate_segment_dirs(target_dir) {
                if seg_id > 0 {
                    std::fs::remove_dir_all(&seg_path)?;
                }
            }
        }

        // Immutable generations inherit base indexes, then overlay rebuilt
        // workspace indexes so the latter win without touching the snapshot.
        self.copy_persisted_indexes(&csr_target)?;

        // Always persist the core CSR arrays, regardless of mmap vs heap
        // backing: after a prior seal (`reconcile_seg0_csr`) they are
        // heap-backed, so relying on mmap persistence would leave the
        // on-disk file at its pre-seal trimmed size while the in-memory
        // Vec carries the full state — the reload then fails loudly on the
        // meta → file-length mismatch. `save_to_file` handles both
        // backings: Heap writes bytes; Mapped-same-path truncates to
        // logical length.
        self.save_logical_node_slots(&csr_target.join("node_slots.bin"))?;
        self.out_offsets
            .save_to_file(&csr_target.join("out_offsets.bin"))?;
        self.out_edges
            .save_to_file(&csr_target.join("out_edges.bin"))?;
        self.in_offsets
            .save_to_file(&csr_target.join("in_offsets.bin"))?;
        self.in_edges
            .save_to_file(&csr_target.join("in_edges.bin"))?;
        self.save_logical_edge_endpoints(&csr_target.join("edge_endpoints.bin"))?;

        if !self.overflow_out.is_empty() || !self.overflow_in.is_empty() {
            let overflow = (&self.overflow_out, &self.overflow_in);
            let bytes = crate::serde_codec::encode_versioned(
                crate::serde_codec::CURRENT_CODEC,
                &overflow,
                MAX_DISK_CODEC_BYTES,
            )
            .map_err(std::io::Error::other)?;
            let compressed =
                zstd::encode_all(bytes.as_slice(), 3).map_err(std::io::Error::other)?;
            std::fs::write(target_dir.join("overflow_edges.bin.zst"), compressed)?;
        }

        // Save edge properties (columnar: edge_prop_offsets.bin + edge_prop_heap.bin).
        // Always write even when empty so format=2 + zero-length files are
        // self-consistent with the metadata. No interner/guard needed —
        // the columnar format stores raw u64 hashes directly. The
        // segment layout puts these alongside the CSR in `csr_target`.
        let upper = self.next_edge_idx;
        self.edge_properties.save_to(&csr_target, upper)?;
        let edge_props_meta = EdgePropertyStore::meta_for(&csr_target);

        // Trim the conn_type_index mmap'd files to their logical length.
        // `MmapOrVec::mapped(path, initial_cap)` has a 64-element minimum,
        // so a 1-type index leaves 512 bytes on disk with stale zeros that
        // the loader can't distinguish from real u64 type hashes. Without
        // this trim, `[r:TYPE]` typed-edge queries return 0 rows after
        // reload (pre-existing bug on v0.8.10).
        // Trimming a buffer to its own backing file is a resize of a mapped
        // file, so go through `trim_to_logical_length`, which releases and
        // re-establishes the mapping in the order each platform requires.
        for field in [
            &mut self.conn_type_index_types as &mut MmapOrVec<u64>,
            &mut self.conn_type_index_offsets,
        ] {
            if field.file_path().is_some() {
                field.trim_to_logical_length()?;
            }
        }
        if self.conn_type_index_sources.file_path().is_some() {
            self.conn_type_index_sources.trim_to_logical_length()?;
        }

        // Trim the core CSR mmap files to their logical length when
        // writing in place (csr_target == self.data_dir).
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
            self.out_offsets.trim_to_logical_length()?;
            self.out_edges.trim_to_logical_length()?;
            self.in_offsets.trim_to_logical_length()?;
            self.in_edges.trim_to_logical_length()?;
        }

        let manifest = self.build_single_segment_manifest(&csr_target)?;
        manifest.save_to(target_dir)?;
        self.segment_manifest = manifest;

        self.write_metadata_to(target_dir, edge_props_meta)?;

        // After a full save everything up to node_count is accounted for
        // in the single-segment on-disk state, so bump the watermark: a
        // subsequent `seal_to_new_segment` must treat only post-save adds
        // as the new tail.
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
    /// ## Two output modes
    ///
    /// The new segment is written in one of two modes depending on
    /// whether the overflow contains cross-segment edges:
    ///
    /// - **Segment-local** (the clean-tail case): all overflow edges have
    ///   both source AND target in `[tail_lo, tail_hi)`. The new
    ///   segment's `out_offsets` / `in_offsets` have length
    ///   `tail_len + 1` and index by the segment's node_slots
    ///   positions (0..tail_len).
    ///
    /// - **Full-range**: at least one overflow edge has an endpoint
    ///   below `tail_lo`. The segment's `out_offsets` / `in_offsets`
    ///   have length `node_count + 1` and index by global node id —
    ///   nodes without edges in this seal get zero-length ranges. Lets a
    ///   single segment carry edges whose source / target is in any
    ///   prior segment's node range.
    ///
    /// `concat_segment_csrs` at load time distinguishes the two modes
    /// by comparing `out_offsets.len()` to `node_slots.len() + 1`. In
    /// full-range mode it unions per-node contributions across segments;
    /// in segment-local mode it preserves the "each node's edges live in
    /// exactly one segment" invariant.
    ///
    /// ## Auxiliary indexes
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
                let ep = self.edge_endpoint(e.edge_idx as usize);
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
                let ep = self.edge_endpoint(e.edge_idx as usize);
                seal_edges.push(SealEdge {
                    src_global,
                    tgt_global: ep.target,
                    conn_type: ep.connection_type,
                });
            }
        }
        // Sort by (source, conn_type) — both modes use this order so the
        // offsets can be built in one sweep.
        seal_edges.sort_by_key(|e| (e.src_global, e.conn_type));
        let n_edges = seal_edges.len();

        // In full-range mode the offset arrays span every global node.
        // In segment-local mode they span only the tail.
        let offsets_len = if has_cross_segment {
            self.node_count + 1
        } else {
            tail_len + 1
        };

        // ─── node_slots: tail only. ───
        let mut node_slots: MmapOrVec<DiskNodeSlot> = MmapOrVec::with_capacity(tail_len);
        for i in 0..tail_len {
            node_slots.try_push(self.node_slot(tail_lo as usize + i))?;
        }

        // ─── edge_endpoints: global source/target, segment-local
        //     edge_idx 0..n_edges. ───
        let mut edge_endpoints: MmapOrVec<EdgeEndpoints> = MmapOrVec::with_capacity(n_edges);
        for e in &seal_edges {
            edge_endpoints.try_push(EdgeEndpoints {
                source: e.src_global,
                target: e.tgt_global,
                connection_type: e.conn_type,
            })?;
        }

        // ─── out_offsets / out_edges: CSR keyed by (segment-local
        //     or global) source. For segment-local mode the offset
        //     index is `src - tail_lo`; for full-range it's `src`
        //     directly. The concat uses `offsets_len` vs
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
            out_offsets.try_push(cursor as u64)?;
            while cursor < n_edges && offset_key(seal_edges[cursor].src_global) == k {
                let e = &seal_edges[cursor];
                out_edges.try_push(CsrEdge {
                    peer: e.tgt_global,
                    edge_idx: cursor as u32,
                })?;
                cursor += 1;
            }
        }
        out_offsets.try_push(cursor as u64)?;

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
            in_offsets.try_push(tcursor as u64)?;
            while tcursor < n_edges && offset_key(by_target[tcursor].0) == k {
                let (_, orig_idx) = by_target[tcursor];
                let src_peer = seal_edges[orig_idx as usize].src_global;
                in_edges.try_push(CsrEdge {
                    peer: src_peer,
                    edge_idx: orig_idx,
                })?;
                tcursor += 1;
            }
        }
        in_offsets.try_push(tcursor as u64)?;

        node_slots.save_to_file(&seg_dir.join("node_slots.bin"))?;
        out_offsets.save_to_file(&seg_dir.join("out_offsets.bin"))?;
        out_edges.save_to_file(&seg_dir.join("out_edges.bin"))?;
        in_offsets.save_to_file(&seg_dir.join("in_offsets.bin"))?;
        in_edges.save_to_file(&seg_dir.join("in_edges.bin"))?;
        edge_endpoints.save_to_file(&seg_dir.join("edge_endpoints.bin"))?;

        // The just-built input vectors are all `MmapOrVec::Heap` — no file
        // handles — so the builders can't race anything mmap'd under
        // `self.data_dir`.
        // conn_type_index is keyed by offset-array index (segment-local
        // in segment-local mode, global in full-range mode). The
        // builder's `node_bound` argument must match the offsets
        // indexing so it walks the full offset range.
        super::builder::write_conn_type_index(
            &out_offsets,
            &out_edges,
            &edge_endpoints,
            offsets_len - 1,
            &seg_dir,
            false,
        )?;
        super::builder::write_peer_count_histogram(&edge_endpoints, 0, n_edges, &seg_dir, false)?;

        // Flush the edge_properties overlay to seg_0's base store. The
        // overlay currently holds props for the sealed edges (keyed by
        // their original global edge_idx). `save_to` absorbs the overlay
        // into seg_0's edge_prop_* files, which cover every segment's
        // edges because concat preserves global edge_idx — so sealed
        // edges' properties survive reload.
        let upper = self.next_edge_idx;
        self.edge_properties.save_to(&self.data_dir, upper)?;

        use super::segment_summary::SegmentSummary;
        let mut summary = SegmentSummary::new(next_id, tail_lo);
        summary.node_id_hi = tail_hi;
        summary.edge_count = n_edges as u64;
        for e in &seal_edges {
            summary.conn_types.insert(e.conn_type);
        }
        for i in 0..tail_len {
            let ns = self.node_slot(tail_lo as usize + i);
            if !ns.is_alive() {
                continue;
            }
            *summary.node_type_counts.entry(ns.node_type).or_insert(0) += 1;
        }
        // indexed_prop_ranges stays empty: the cache+scan populates seg 0's
        // indexes only, and per-segment property indexes do not exist.

        self.segment_manifest.append(summary);
        self.segment_manifest.save_to(root)?;

        // Reconcile seg_0's on-disk files with the new layout:
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

        // Both modes wrote the entire overflow map into this seal, so drop
        // it in-memory; the persisted CSR in seg_NNN/ is now the source of
        // truth.
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

    /// Load a disk graph from a directory. Raw .bin files are mmap'd
    /// directly from the graph dir; legacy .bin.zst files are decompressed
    /// into a `_zst_cache` staging dir first. Returns `(DiskGraph,
    /// temp_dir)` — that staging path is always returned, but only created
    /// if something actually needed decompressing.
    ///
    /// `interner` is threaded through to the edge-property loader but never
    /// mutated: format-2 columnar payloads store raw u64 hashes, and the
    /// pre-0.14 formats that stored InternedKey as strings are rejected.
    pub fn load_from_dir(
        dir: &Path,
        interner: &mut crate::graph::schema::StringInterner,
    ) -> std::io::Result<(Self, PathBuf)> {
        use crate::graph::io::load_timing::{log_stage, stage_timer};

        let t = stage_timer();
        let meta_str = std::fs::read_to_string(dir.join("disk_graph_meta.json"))?;
        let meta: DiskGraphMeta = serde_json::from_str(&meta_str).map_err(std::io::Error::other)?;
        let serde_codec = validate_disk_format(&meta)?;
        log_stage("dg.meta_parse", t);

        // CSR binaries live under seg_NNN/ when the graph was written
        // with csr_layout_version >= 1. Legacy .kgl directories
        // (version=0, the serde default) keep the flat layout.
        //
        // Multi-segment graphs are produced by ordinary saves:
        // `save_to_dir` seals a clean tail into a new seg_NNN whenever a
        // prior save exists (see `save_disposition`).
        //
        // Auxiliary per-segment data is handled unevenly in the N>1 branch
        // — see the limitation documented on `SegmentCsr`.

        // Staging dir for legacy .zst decompression, inside the graph dir so
        // no external temp space is required.
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
                    // and none of the concat work the multi-segment
                    // branch below does.
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
                    let csr = concat_segment_csrs(loaded)?;
                    (first_dir, csr)
                }
            }
        } else {
            // Legacy flat layout: load from root as one segment, using
            // meta's *_len values.
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

        // Postcard columnar edge properties (format 2). In the segmented
        // layout these files live alongside the CSR.
        let t = stage_timer();
        let edge_properties = EdgePropertyStore::load_from(
            &csr_dir,
            meta.edge_properties_format,
            meta.edge_properties_meta,
            interner,
        )?;
        log_stage("dg.edge_properties", t);

        // Overflow edges live at the graph root, orthogonal to segments.
        let t = stage_timer();
        let (overflow_out, overflow_in) = load_overflow_edges(dir, serde_codec)?;
        log_stage("dg.overflow_edges", t);

        let t = stage_timer();
        let segment_manifest = SegmentManifest::load_from(dir).unwrap_or_default();
        log_stage("dg.segment_manifest", t);

        // Older graphs predate the persisted watermark and serde-default
        // it to 0, but their `seg_000` already accounts for every node.
        // Without the bump, a re-save calls `seal_to_new_segment` with
        // `tail_lo=0` and `tail_hi=node_count`, writing a fresh empty
        // `seg_001` AND truncating seg_000's `out_offsets.bin` /
        // `in_offsets.bin` via `reconcile_seg0_csr` — corrupting the
        // graph. Graphs that persist the watermark are unaffected.
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
                node_slot_updates: HashMap::new(),
                appended_node_slots: Vec::new(),
                node_count: meta.node_count,
                free_node_slots: meta.free_node_slots,
                arenas: crate::graph::storage::disk::query_arena::QueryArenas::new(1024),
                column_stores: rustc_hash::FxHashMap::default(),
                out_offsets,
                out_edges,
                in_offsets,
                in_edges,
                edge_endpoints,
                appended_edge_endpoints: Vec::new(),
                removed_edges: std::collections::HashSet::new(),
                edge_count: meta.edge_count,
                next_edge_idx: meta.next_edge_idx,
                edge_properties,
                edge_mut_cache: HashMap::new(),
                node_mut_cache: HashMap::new(),
                pending_edges: UnsafeCell::new(MmapOrVec::new()),
                overflow_out,
                overflow_in,
                free_edge_slots: meta.free_edge_slots,
                data_dir: csr_dir.clone(),
                logical_root: dir.to_path_buf(),
                writer_lock: None,
                mutation_workspace: None,
                parent_workspaces: Vec::new(),
                independent_root: None,
                csr_sorted_by_type: meta.csr_sorted_by_type,
                defer_csr: false,
                edge_type_counts_raw: None,
                conn_type_index_types,
                conn_type_index_offsets,
                conn_type_index_sources,
                peer_count_types,
                peer_count_offsets,
                peer_count_entries,
                has_tombstones: meta.has_tombstones,
                property_indexes: std::sync::RwLock::new(HashMap::new()),
                removed_property_indexes: HashSet::new(),
                global_indexes: std::sync::RwLock::new(HashMap::new()),
                // Legacy .kgl directories have no seg_manifest.json; the
                // resulting empty manifest means "pre-segmented, don't prune".
                segment_manifest,
                sealed_nodes_bound,
            },
            temp_dir,
        ))
    }
}

/// Load a binary array: try raw `.bin` first (direct mmap, no temp dir),
/// fall back to `.bin.zst` (decompress to temp dir, then mmap).
fn load_raw_or_zst<T: crate::graph::storage::mapped::mmap_vec::MmapPod>(
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
fn load_raw_or_zst_optional<T: crate::graph::storage::mapped::mmap_vec::MmapPod>(
    base_path: &Path,
) -> MmapOrVec<T> {
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
fn load_compressed<T: crate::graph::storage::mapped::mmap_vec::MmapPod>(
    path: &Path,
    len: usize,
    temp_dir: &Path,
) -> std::io::Result<MmapOrVec<T>> {
    if !path.exists() || len == 0 {
        return Ok(MmapOrVec::new());
    }
    let compressed = std::fs::read(path)?;
    let raw = zstd::decode_all(compressed.as_slice())?;

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
// Multi-segment CSR
// ============================================================================

/// One segment's core CSR arrays, loaded from its subdirectory. Used
/// only when a graph spans multiple `seg_NNN/` dirs — single-segment
/// graphs stay on the direct mmap path in [`DiskGraph::load_from_dir`].
///
/// Indexing conventions concat relies on:
///
///   - `node_slots`: one entry per node in this segment; the segment owns
///     a disjoint node-id range reported in its `SegmentSummary`.
///   - `out_offsets` / `in_offsets`: indexed by local node position for a
///     segment-local seal, by global node id for a full-range one (see
///     [`DiskGraph::seal_to_new_segment`]).
///   - `out_edges` / `in_edges`: each entry's `edge_idx` is
///     **segment-local** (`0..edge_endpoints.len()`), so concat shifts
///     them onto the combined edge_endpoints.
///   - `edge_endpoints`: `source` / `target` store *global* node ids and
///     are never rewritten by concat.
///
/// The auxiliary inverted indexes (`conn_type_index_*`, `peer_count_*`)
/// are bundled per segment and merged by `concat_segment_csrs`.
/// `edge_properties` / `column_stores` / per-(type,prop) property indexes
/// are **not** bundled here — they load from segment 0 only. Sealing
/// flushes the new segment's edge properties back into seg_0's store and
/// concat preserves global `edge_idx`, so edge properties survive;
/// per-segment property indexes remain a known limitation.
pub(crate) struct SegmentCsr {
    pub(crate) node_slots: MmapOrVec<super::csr::DiskNodeSlot>,
    pub(crate) out_offsets: MmapOrVec<u64>,
    pub(crate) out_edges: MmapOrVec<super::csr::CsrEdge>,
    pub(crate) in_offsets: MmapOrVec<u64>,
    pub(crate) in_edges: MmapOrVec<super::csr::CsrEdge>,
    pub(crate) edge_endpoints: MmapOrVec<super::csr::EdgeEndpoints>,
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
            // Auxiliary files are optional: segments written before seal
            // emitted auxiliary indexes lack them, and
            // `load_raw_or_zst_optional` yields an empty MmapOrVec then.
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
fn reconcile_seg0_csr<T: crate::graph::storage::mapped::mmap_vec::MmapPod>(
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
fn load_with_inferred_len<T: crate::graph::storage::mapped::mmap_vec::MmapPod>(
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
        // Legacy path: the zstd stream carries no element count, so
        // decompress to a temp file and infer the count from its size.
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
/// Single-segment input is returned as-is, without allocating. For N > 1
/// the combined arrays are heap-backed; nothing is written to disk, so
/// the read path sees an in-memory combined CSR only.
///
/// Beyond the per-segment indexing conventions documented on
/// [`SegmentCsr`], this assumes segments arrive in manifest order
/// (ascending `segment_id`) covering contiguous disjoint node-id ranges
/// `[0, n_0) + [n_0, n_0 + n_1) + ...` — the caller
/// ([`DiskGraph::load_from_dir`]) preserves `enumerate_segment_dirs`
/// ordering.
///
/// Violations produce a garbage combined CSR; `seal_to_new_segment` is
/// the writer that must honour them.
pub(crate) fn concat_segment_csrs(mut segments: Vec<SegmentCsr>) -> std::io::Result<SegmentCsr> {
    use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints};
    let combined = match segments.len() {
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
            //   is_full[k]               : out_offsets covers all global
            //                              nodes vs just this segment's
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
                // node. The writer doesn't know the total node count at
                // seal time, but it is always >= node_slots.len(), so
                // `out_offsets.len() > node_slots.len() + 1` uniquely
                // signals full-range.
                is_full.push(seg.out_offsets.len() > seg.node_slots.len() + 1);
            }

            // Node slots + edge endpoints: straight concat.
            for seg in &segments {
                for i in 0..seg.node_slots.len() {
                    node_slots.try_push(seg.node_slots.get(i))?;
                }
                for i in 0..seg.edge_endpoints.len() {
                    edge_endpoints.try_push(seg.edge_endpoints.get(i))?;
                }
            }

            // Walk every combined node id and UNION each segment's
            // out_edges / in_edges contributions for that node. Per
            // node, a segment contributes when:
            //   - segment-local & node id in [node_lo, node_hi)  → key = gid - node_lo
            //   - full-range                                       → key = gid  (cap: offsets len)
            out_offsets.try_push(0)?;
            in_offsets.try_push(0)?;
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
                            out_edges.try_push(e)?;
                        }
                    }
                }
                out_offsets.try_push(out_edges.len() as u64)?;

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
                            in_edges.try_push(e)?;
                        }
                    }
                }
                in_offsets.try_push(in_edges.len() as u64)?;
            }

            let (cti_types, cti_offsets, cti_sources) = merge_conn_type_index(&segments)?;
            let (pc_types, pc_offsets, pc_entries) = merge_peer_count_histogram(&segments)?;

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
    };
    Ok(combined)
}

/// Merge per-segment `conn_type_index_*` into a combined index.
/// For each connection type, the combined sources list is the
/// concatenation of per-segment sources lists (already globally sorted
/// because segments own disjoint node ranges and per-segment lists are
/// locally sorted). Types are unioned and sorted ascending.
fn merge_conn_type_index(
    segments: &[SegmentCsr],
) -> std::io::Result<(MmapOrVec<u64>, MmapOrVec<u64>, MmapOrVec<u32>)> {
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
        out_types.try_push(*t)?;
        out_offsets.try_push(cur_off)?;
        for &si in seg_idxs {
            let seg = &segments[si];
            let shift = if is_full[si] { 0 } else { node_lo[si] };
            let n = seg.conn_type_index_types.len();
            // Linear scan — typical segment has ≤ hundreds of types,
            // so a BTreeMap-per-segment isn't worth the setup cost.
            for j in 0..n {
                if seg.conn_type_index_types.get(j) == *t {
                    let start = seg.conn_type_index_offsets.get(j) as usize;
                    let end = seg.conn_type_index_offsets.get(j + 1) as usize;
                    for k in start..end {
                        out_sources.try_push(seg.conn_type_index_sources.get(k) + shift)?;
                    }
                    cur_off += (end - start) as u64;
                    break;
                }
            }
        }
    }
    out_offsets.try_push(cur_off)?;
    Ok((out_types, out_offsets, out_sources))
}

/// Merge per-segment `peer_count_*` histograms by summing counts for
/// every `(conn_type, peer)` pair that appears in any segment.
fn merge_peer_count_histogram(
    segments: &[SegmentCsr],
) -> std::io::Result<(MmapOrVec<u64>, MmapOrVec<u64>, MmapOrVec<u32>)> {
    use std::collections::BTreeMap;
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
        out_types.try_push(*t)?;
        out_offsets.try_push(cur_pairs)?;
        for (peer, count) in peers {
            out_entries.try_push(*peer)?;
            // u64 count saturates to u32 for the on-disk format; sums
            // across segments in practice fit because per-segment
            // counts are u32 and at most `edge_count`.
            out_entries.try_push((*count).min(u32::MAX as u64) as u32)?;
        }
        cur_pairs += peers.len() as u64;
    }
    out_offsets.try_push(cur_pairs)?;
    Ok((out_types, out_offsets, out_entries))
}
