//! Disk-mode lifecycle and persistence orchestration.

use super::*;
use crate::graph::storage::packed_codec::IntColumnEncoding;
use std::sync::atomic::{AtomicU64, Ordering};

static DISK_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn write_compressed_disk_serde<T: serde::Serialize + ?Sized>(
    dir: &std::path::Path,
    filename: &str,
    value: &T,
    label: &str,
) -> Result<(), String> {
    let bytes = crate::graph::io::file::encode_disk_serde(value)
        .map_err(|e| format!("{label} serialization failed: {e}"))?;
    let compressed = zstd::encode_all(bytes.as_slice(), 3)
        .map_err(|e| format!("{label} compression failed: {e}"))?;
    std::fs::write(dir.join(filename), compressed)
        .map_err(|e| format!("Failed to write {label}: {e}"))
}

impl DirGraph {
    /// Convert the graph to disk-backed storage mode.
    /// Enables columnar storage first, then builds CSR edge arrays on disk.
    /// Nodes stay in memory (~40 bytes each), edges are mmap'd.
    pub fn enable_disk_mode(&mut self) -> Result<(), String> {
        // Ensure columnar storage for compact node representation
        if self.column_store_count() == 0 {
            self.enable_columnar();
        }

        // Create a temp directory for CSR files
        let sequence = DISK_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let data_dir = std::env::temp_dir().join(format!(
            "kglite_disk_{}_{:x}_{sequence:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));

        // The heap backend owns the stores; the new `DiskGraph` must inherit
        // them, or every columnar property read returns Null after the switch.
        // Before D1 Phase 3 a DirGraph-level copy survived the swap and
        // an explicit mirror step pushed it in — there is no such copy now.
        let carried_stores: HashMap<InternedKey, Arc<ColumnStore>> = self
            .graph
            .column_stores_iter()
            .map(|(k, v)| (k, Arc::clone(v)))
            .collect();

        // A forked backend has no single `StableDiGraph` to hand over, so
        // collapse it first. Free when the reader has already dropped.
        self.graph.flatten_fork();

        // Extract the StableDiGraph and build DiskGraph
        let disk_graph = match &mut self.graph {
            GraphBackend::Memory(g) => {
                crate::graph::storage::disk::graph::DiskGraph::from_stable_digraph(
                    crate::graph::storage::backend::unique_heap_backend(g).inner_mut(),
                    &data_dir,
                )
            }
            GraphBackend::Mapped(g) => {
                crate::graph::storage::disk::graph::DiskGraph::from_stable_digraph(
                    crate::graph::storage::backend::unique_heap_backend(g).inner_mut(),
                    &data_dir,
                )
            }
            GraphBackend::Forked(_) => unreachable!("flatten_fork collapsed the overlay above"),
            GraphBackend::Disk(_) => return Err("Already in disk mode".to_string()),
            GraphBackend::Recording(_) => {
                return Err(
                    "enable_disk_mode not supported while wrapped in RecordingGraph".to_string(),
                )
            }
        }
        .map_err(|e| format!("Failed to create DiskGraph: {}", e))?;

        // Register temp dir for cleanup
        if let Ok(mut dirs) = self.temp_dirs.lock() {
            dirs.push(data_dir);
        }

        self.graph = GraphBackend::Disk(Box::new(disk_graph));
        for (type_key, store) in carried_stores {
            GraphWrite::install_column_store(&mut self.graph, type_key, store);
        }
        Ok(())
    }

    /// Acquire the retained disk-writer lease before creating a mutation
    /// overlay. Memory and mapped backends are unaffected.
    pub(crate) fn prepare_disk_mutation(&mut self) -> std::io::Result<()> {
        if let GraphBackend::Disk(disk) = &mut self.graph {
            disk.prepare_mutation()?;
        }
        Ok(())
    }

    /// Build CSR from pending edges if in disk mode. No-op otherwise.
    /// Called after add_connections, before queries, and before save.
    pub fn ensure_disk_edges_built(&mut self) -> Result<(), String> {
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            dg.build_csr_from_pending()
                .map_err(|e| format!("disk CSR build failed: {e}"))?;
            // Don't compact here — overflow-merge is O(E), so calling it
            // after every add_connections batch would make multi-batch
            // builds quadratic. Queries still see overflow edges via the
            // merged DiskEdges iterator. Aggregate caches (conn_type_index
            // / peer_count_histogram) are refreshed at save time by
            // `save_disk` when overflow is present.
        }
        Ok(())
    }

    /// Compact a disk-mode graph: merge overflow edges back into CSR arrays.
    /// Returns the number of overflow edges that were merged.
    /// No-op if there are no overflow edges.
    pub fn compact_disk(&mut self) -> Result<usize, String> {
        self.prepare_disk_mutation()
            .map_err(|e| format!("disk mutation lease failed: {e}"))?;
        match &mut self.graph {
            GraphBackend::Disk(ref mut dg) => dg.compact().map_err(|e| e.to_string()),
            _ => Err("compact requires disk mode".to_string()),
        }
    }

    /// Save a disk-mode graph to a directory. The directory IS the graph.
    /// Persists CSR files, node data, edge properties, column stores, and metadata.
    pub fn save_disk(&mut self, path: &str) -> Result<(), String> {
        let root = std::path::PathBuf::from(path);
        let writer_lock = match &mut self.graph {
            GraphBackend::Disk(disk)
                if disk
                    .writer_lock
                    .as_ref()
                    .is_some_and(|lock| lock.root == root) =>
            {
                disk.writer_lock.as_ref().unwrap().clone()
            }
            GraphBackend::Disk(_) => std::sync::Arc::new(
                crate::graph::storage::disk::generation::GraphDirectoryLock::try_acquire(&root)
                    .map_err(|e| format!("Failed to acquire disk writer lock: {e}"))?,
            ),
            _ => return Err("save_disk requires disk mode".to_string()),
        };
        if let GraphBackend::Disk(disk) = &mut self.graph {
            disk.writer_lock = Some(writer_lock.clone());
            disk.prepare_mutation()
                .map_err(|e| format!("Failed to prepare disk workspace: {e}"))?;
            disk.begin_persist();
        }
        let generation = crate::graph::storage::disk::generation::GenerationTxn::begin(&root)
            .map_err(|e| format!("Failed to begin disk generation: {e}"))?;
        self.write_disk_snapshot(generation.stage_dir())?;
        let published = generation
            .publish()
            .map_err(|e| format!("Failed to publish disk generation: {e}"))?;
        if let GraphBackend::Disk(disk) = &mut self.graph {
            disk.finish_generation(root, published, writer_lock)
                .map_err(|e| format!("Failed to activate published disk generation: {e}"))?;
        }
        Ok(())
    }

    fn write_disk_snapshot(&mut self, dir: &std::path::Path) -> Result<(), String> {
        self.consolidate_disk_for_save(dir)?;

        // save_to_dir needs &mut access so the edge-property store can
        // drop its base mmap before overwriting (PR2).
        let dg = match &mut self.graph {
            GraphBackend::Disk(dg) => dg,
            _ => return Err("save_disk requires disk mode".to_string()),
        };
        dg.begin_persist();

        // Save DiskGraph files (CSR, nodes, edge properties, metadata).
        // `save_to_dir` runs `clear_arenas` internally, which drains
        // `node_mut_cache` via the clone-apply-replace flush, updating
        // each mutated type's Arc in `DiskGraph.column_stores`.
        dg.save_to_dir(dir, &self.interner)
            .map_err(|e| format!("DiskGraph save failed: {}", e))?;
        // No mirror to refresh: `DiskGraph` *is* the owner of the column
        // stores (D1 Phase 3), so the sidecar writer below reads the same
        // `Arc`s `save_to_dir` just flushed into. The pre-D1 shape kept a
        // second copy on `DirGraph`, and a stale one is exactly how Cypher
        // `SET` / `DETACH DELETE` corrections once failed to reach disk.

        // Save DirGraph metadata. 0.8.13 stripped `type_connectivity`;
        // 0.8.28 strips the two heavy HashMap fields
        // (`node_type_metadata`, `connection_type_metadata`) into
        // dedicated binary sidecars. The remaining metadata.json is
        // small (under a few hundred KB even on Wikidata-scale) and
        // parses in milliseconds.
        crate::graph::io::file::write_node_type_metadata_bin(dir, self)?;
        crate::graph::io::file::write_connection_type_metadata_bin(dir, self)?;
        // Secondary labels — disk's columnar layout has no slot for
        // NodeData.extra_labels, so we persist the inverted index as
        // a sidecar. Skipped when the graph has no secondaries
        // (single-label disk graphs pay zero extra bytes).
        crate::graph::io::file::write_secondary_labels_bin(dir, self)?;
        let mut meta = crate::graph::io::file::build_disk_metadata(self);
        crate::graph::io::file::strip_type_connectivity(&mut meta);
        crate::graph::io::file::strip_heavy_metadata(&mut meta);
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| format!("Metadata serialization failed: {}", e))?;
        // Emit the packed binary `type_connectivity.bin.zst` at the
        // graph root; no-op when the cache is empty.
        crate::graph::io::file::write_type_connectivity_bin(dir, self)?;

        // The framed interner sidecar stores `Vec<String>`; hashes are
        // re-derived on load. Unframed binary data is rejected, while the
        // older JSON representation remains a read-only data fallback.
        crate::graph::io::file::write_interner_bin(dir, self)?;

        self.write_unified_column_file(dir)?;
        self.write_column_sidecars(dir)?;

        // 0.8.13: type_indices uses a flat CSR binary keyed by interner
        // hashes. 0.8.28+: id_indices uses an mmap-resident raw `.bin`
        // layout — load reads via memory-mapped binary search, no eager
        // HashMap rebuild. The loader can still read the earlier flat-CSR
        // sidecars when the mmap files are absent; pre-0.14 bincode caches
        // are ignored and rebuilt.
        crate::graph::storage::disk::type_index::write_type_indices_bin(
            dir,
            &self.type_indices,
            &self.interner,
        )?;
        crate::graph::storage::disk::id_index::write_id_indices_bin(
            dir,
            &self.id_indices,
            &self.interner,
        )?;

        // Save embeddings if any (matches write_kgl behavior for in-memory saves).
        // BTreeMap view for byte-determinism — same rationale as write_kgl.
        if !self.embeddings.is_empty() {
            let ordered: std::collections::BTreeMap<_, _> = self.embeddings.iter().collect();
            write_compressed_disk_serde(dir, "embeddings.bin.zst", &ordered, "embeddings")?;
        }

        // Save timeseries_store if any (BTreeMap view for byte-determinism).
        if !self.timeseries_store.is_empty() {
            let ordered: std::collections::BTreeMap<_, _> = self.timeseries_store.iter().collect();
            write_compressed_disk_serde(dir, "timeseries.bin.zst", &ordered, "timeseries")?;
        }

        // Root metadata is the graph-level completion marker. Publish it only
        // after every required sidecar has been written successfully.
        std::fs::write(dir.join("metadata.json"), meta_json)
            .map_err(|e| format!("Failed to publish metadata: {}", e))?;

        Ok(())
    }

    /// Bring a disk graph's CSR, overflow edges and global indexes to their
    /// final saved state before anything is written.
    fn consolidate_disk_for_save(&mut self, dir: &std::path::Path) -> Result<(), String> {
        // Build CSR from pending edges if not yet built.
        self.ensure_disk_edges_built()?;
        // Merge overflow edges back so conn_type_index and
        // peer_count_histogram reflect every live edge. Skipped during
        // builds; done here as a one-shot so users only pay the cost at
        // save time, not per add_connections batch.
        //
        // The seal path consumes `overflow_out` / `overflow_in` directly,
        // while a rewrite must compact first so every derived index is rebuilt
        // from the complete CSR. Use the lower layer's target-aware decision:
        // generation stages are distinct directories and always rewrite.
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            dg.begin_persist();
            let disposition = dg.save_disposition(dir);
            if disposition == crate::graph::storage::disk::graph_persist::SaveDisposition::Rewrite
                && dg.has_overflow()
            {
                dg.compact()
                    .map_err(|e| format!("disk compaction failed: {e}"))?;
            }
            // Auto-build the cross-type global title index so that
            // `MATCH (n {title: 'X'})` and `g.search(text)` are O(log N)
            // out of the box on every saved disk graph. Runs after
            // CSR / overflow consolidation so it sees the final node
            // set. Tied to `save_disk` rather than `build_csr_*` so
            // node-only graphs (no edges) still get the index built.
            dg.build_global_property_index("title")
                .map_err(|e| format!("title index build failed: {e}"))?;
            // Likewise index `nid` — the string id form for prefixed-id
            // datasets (Wikidata `"Q42"`). Since 0.11.0 `{nid: 'Q42'}` is a
            // plain string-property lookup (not the integer id-index), so the
            // index keeps it O(log N) instead of a 124M-row scan. No-op when
            // no type has a `nid` column.
            dg.build_global_property_index("nid")
                .map_err(|e| format!("nid index build failed: {e}"))?;
        }
        Ok(())
    }

    /// Write the unified `columns.bin` mega-file for a graph that has none.
    ///
    /// Mode 3 (0.9.15): a fresh save — streaming carve, `save_subset`, or the
    /// mutation persist of an in-memory build — emits the same layout the
    /// ntriples builder produces, so the saved graph reloads through the mmap
    /// fast path. Without it a saved `DiskGraph` fell back to per-type zstd
    /// sidecars and took ~70 s to load on a 17 M-node Wikidata carve against
    /// ~150 ms for the full graph.
    ///
    /// A pre-existing `columns.bin` (root or `seg_000/`, the pre- and
    /// post-phase-4 layouts) means the ntriples builder already wrote one;
    /// [`Self::write_column_sidecars`] covers the types added since.
    fn write_unified_column_file(&self, dir: &std::path::Path) -> Result<(), String> {
        let preexisting_columns_bin =
            dir.join("seg_000/columns.bin").exists() || dir.join("columns.bin").exists();
        if !preexisting_columns_bin && self.column_store_count() > 0 {
            let stores: HashMap<String, Arc<crate::graph::storage::column_store::ColumnStore>> =
                self.column_stores_by_name()
                    .into_iter()
                    .map(|(name, store)| (name.to_string(), Arc::clone(store)))
                    .collect();
            crate::graph::io::unified_columns::write_unified_columns(dir, &stores, &self.interner)
                .map_err(|e| format!("unified columns write failed: {}", e))?;
        }
        Ok(())
    }

    /// Which node types the graph's `columns.bin` already covers.
    ///
    /// Read from whichever `columns_meta` sidecar exists — the binary form and
    /// the older JSON one, at the segmented (`seg_000/`) or the legacy flat
    /// root. An absent sidecar means no type is covered.
    fn types_in_columns_bin(
        dir: &std::path::Path,
    ) -> Result<std::collections::HashSet<String>, String> {
        let columns_meta_path = [
            "seg_000/columns_meta.bin.zst",
            "seg_000/columns_meta.json",
            "columns_meta.bin.zst",
            "columns_meta.json",
        ]
        .into_iter()
        .map(|name| dir.join(name))
        .find(|path| path.exists());
        let Some(meta_path) = columns_meta_path else {
            return Ok(std::collections::HashSet::new());
        };
        use crate::graph::io::ntriples::ColumnTypeMeta;
        let metas: Vec<ColumnTypeMeta> = if meta_path.extension().and_then(|s| s.to_str())
            == Some("zst")
        {
            let compressed = std::fs::read(&meta_path)
                .map_err(|e| format!("read {}: {}", meta_path.display(), e))?;
            let bytes = zstd::decode_all(compressed.as_slice())
                .map_err(|e| format!("decompress columns_meta: {}", e))?;
            crate::graph::io::file::decode_disk_serde(&bytes, bytes.capacity() as u64)
                .map_err(|e| format!("parse columns_meta.bin: {}", e))?
        } else {
            let json = std::fs::read_to_string(&meta_path)
                .map_err(|e| format!("read {}: {}", meta_path.display(), e))?;
            serde_json::from_str(&json).map_err(|e| format!("parse columns_meta.json: {}", e))?
        };
        Ok(metas.into_iter().map(|tm| tm.type_name).collect())
    }

    /// Write a per-type `columns.zst` sidecar for every type the unified
    /// `columns.bin` does not cover.
    ///
    /// Types added post-build via `add_nodes` / `add_node` are absent from an
    /// ntriples-built `columns.bin` and were silently dropped on save before
    /// this existed. Covered types keep the fast mmap path.
    fn write_column_sidecars(&self, dir: &std::path::Path) -> Result<(), String> {
        let types_in_columns_bin = Self::types_in_columns_bin(dir)?;
        let columns_dir = dir.join("columns");
        let mut sidecars_written = 0usize;
        for (type_name, store) in self.column_stores_by_name() {
            if types_in_columns_bin.contains(type_name) {
                continue; // covered by the fast mmap path on reload
            }
            if sidecars_written == 0 {
                std::fs::create_dir_all(&columns_dir)
                    .map_err(|e| format!("Failed to create columns dir: {}", e))?;
            }
            let type_dir = columns_dir.join(type_name);
            std::fs::create_dir_all(&type_dir)
                .map_err(|e| format!("Failed to create type dir: {}", e))?;
            let packed = store
                .write_packed_with_codec(
                    &self.interner,
                    crate::serde_codec::CodecVersion::PostcardV1,
                    // `KGLCOLv2` is its own container; `.kgl` v6 encodings do not apply.
                    IntColumnEncoding::Raw,
                )
                .map_err(|e| format!("Column pack failed: {}", e))?;
            // Prefix with a magic tag + the ColumnStore's row_count so
            // `load_column_sidecars` can pass the correct row count to
            // `ColumnStore::load_packed`. Pre-fix the loader derived
            // row_count from `type_indices[type].len()`, which counts
            // only *live* rows — after a DETACH DELETE that leaves
            // tombstoned rows in the store, the mismatch caused
            // `load_packed` to read column blobs at the wrong offsets
            // and produce garbage titles / null ages on reload.
            //
            // Format:
            //   magic: 8 bytes b"KGLCOLv2"
            //   row_count: u32 LE
            //   packed: existing `write_packed` output
            //
            // Old-format sidecars (no magic tag) stay loadable via a
            // fallback in the load path.
            let mut framed: Vec<u8> = Vec::with_capacity(12 + packed.len());
            framed.extend_from_slice(b"KGLCOLv2");
            framed.extend_from_slice(&store.row_count().to_le_bytes());
            framed.extend_from_slice(&packed);
            let compressed = zstd::encode_all(framed.as_slice(), 3)
                .map_err(|e| format!("Column compression failed: {}", e))?;
            std::fs::write(type_dir.join("columns.zst"), compressed)
                .map_err(|e| format!("Failed to write columns: {}", e))?;
            sidecars_written += 1;
        }
        Ok(())
    }
}
