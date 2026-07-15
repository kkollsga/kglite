//! Disk-mode lifecycle and persistence orchestration.

use super::*;

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
        if !self.is_columnar() {
            self.enable_columnar();
        }

        // Create a temp directory for CSR files
        let data_dir = std::env::temp_dir().join(format!(
            "kglite_disk_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));

        // Extract the StableDiGraph and build DiskGraph
        let disk_graph = match &mut self.graph {
            GraphBackend::Memory(g) => {
                crate::graph::storage::disk::graph::DiskGraph::from_stable_digraph(
                    g.inner_mut(),
                    &data_dir,
                )
            }
            GraphBackend::Mapped(g) => {
                crate::graph::storage::disk::graph::DiskGraph::from_stable_digraph(
                    g.inner_mut(),
                    &data_dir,
                )
            }
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
        Ok(())
    }

    /// Sync column store references from DirGraph to DiskGraph.
    /// Called after enable_columnar(), add_nodes(), and load.
    pub fn sync_disk_column_stores(&mut self) {
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            let mut stores = HashMap::new();
            for (type_name, store) in &self.column_stores {
                let key = InternedKey::from_str(type_name);
                stores.insert(key, Arc::clone(store));
            }
            dg.set_column_stores(stores);
        }
    }

    /// Acquire the retained disk-writer lease before creating a mutation
    /// overlay. Memory and mapped backends are unaffected.
    pub(crate) fn prepare_disk_mutation(&mut self) -> std::io::Result<()> {
        if let GraphBackend::Disk(disk) = &mut self.graph {
            disk.prepare_mutation()?;
        }
        Ok(())
    }

    /// Mirror DiskGraph's column_stores back into DirGraph's
    /// `self.column_stores`. Called after mutations that flushed
    /// through `DiskGraph::flush_node_mut_cache` so the sidecar writer
    /// in `save_disk` and any other DirGraph-side reader sees the
    /// post-flush state rather than the pre-flush (stale) Arcs.
    pub fn sync_column_stores_from_disk(&mut self) {
        if let GraphBackend::Disk(ref dg) = self.graph {
            let pairs: Vec<(
                String,
                Arc<crate::graph::storage::column_store::ColumnStore>,
            )> = dg
                .column_stores_iter()
                .map(|(k, v)| (self.interner.resolve(*k).to_string(), Arc::clone(v)))
                .collect();
            for (name, arc) in pairs {
                self.column_stores.insert(name, arc);
            }
        }
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
        // Build CSR from pending edges if not yet built.
        self.ensure_disk_edges_built()?;
        // Merge overflow edges back so conn_type_index and
        // peer_count_histogram reflect every live edge. Skipped during
        // builds; done here as a one-shot so users only pay the cost at
        // save time, not per add_connections batch.
        //
        // Gate: the phase-6 seal path in `save_to_dir` consumes
        // `overflow_out` / `overflow_in` directly. Running `compact()`
        // first moves those edges into the CSR (clearing overflow),
        // which causes seal to write an empty segment and lose the
        // new edges on reload. Only compact when we're taking the
        // compact-rewrite path (no prior save, or no tail above the
        // sealed watermark).
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            dg.begin_persist();
            let will_seal =
                !dg.segment_manifest.is_empty() && dg.sealed_nodes_bound < dg.node_count() as u32;
            if !will_seal && dg.has_overflow() {
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
        // Mirror the post-flush Arcs back into `self.column_stores` so
        // the per-type sidecar writer below sees the mutated stores
        // rather than the pre-flush (stale) Arcs. Pre-fix, mutations
        // landed in DiskGraph's Arcs but the sidecar writer read
        // DirGraph's Arcs — Cypher SET and DETACH DELETE property
        // corrections never reached disk.
        self.sync_column_stores_from_disk();

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

        // 0.8.13: interner switches from JSON (hash → original) to
        // bincode `Vec<String>` of originals. The hash is re-derived
        // deterministically on load via `get_or_intern`. Loader falls
        // back to `interner.json` for graphs saved by 0.8.12 and
        // earlier.
        crate::graph::io::file::write_interner_bin(dir, self)?;

        // Save column stores (per type, sidecar format). Two modes:
        //
        // 1. **No `columns.bin`** — DirGraph → disk saves that never
        //    went through the N-Triples streaming builder. Write every
        //    column store as a sidecar.
        //
        // 2. **`columns.bin` exists** — N-Triples-built disk graphs.
        //    The single-file v3 layout covers every type that was
        //    present at ingest time, and reloading it via the mmap fast
        //    path (`file.rs:580`) is dramatically cheaper than walking
        //    per-type sidecars on a 88k-type wiki graph. BUT: types
        //    added post-build via `add_nodes` / `add_node` are not in
        //    `columns.bin` and were silently dropped on save before
        //    this fix. Emit sidecars *only* for those types — keeps the
        //    fast path for initial types, makes mutation persistence
        //    correct for new ones.
        //
        // 0.8.12 phase-1: PR1 phase 4 moved `columns.bin` under
        // `seg_000/`, so the presence check covers both the root (legacy
        // flat layout) and `seg_000/` (post-phase-4 segmented layout).
        // Mode-3 (new in 0.9.15): no preexisting `columns.bin` AND
        // we have in-memory `column_stores` (typical fresh save:
        // streaming carve, save_subset, mutation persist of an
        // in-memory build). Emit the unified mega-file format that
        // the loader's mmap fast path consumes — same layout the
        // ntriples builder produces — so the saved graph loads with
        // the same speed as a freshly-built one. Without this, a
        // saved DiskGraph fell into the per-type zstd sidecar path
        // and took ~70 s to load on a 17 M-node Wikidata carve vs.
        // ~150 ms for the full graph.
        let preexisting_columns_bin =
            dir.join("seg_000/columns.bin").exists() || dir.join("columns.bin").exists();
        if !preexisting_columns_bin && !self.column_stores.is_empty() {
            crate::graph::io::unified_columns::write_unified_columns(
                dir,
                &self.column_stores,
                &self.interner,
            )
            .map_err(|e| format!("unified columns write failed: {}", e))?;
        }

        let columns_meta_path = {
            let seg0_bin = dir.join("seg_000/columns_meta.bin.zst");
            let seg0_json = dir.join("seg_000/columns_meta.json");
            let root_bin = dir.join("columns_meta.bin.zst");
            let root_json = dir.join("columns_meta.json");
            if seg0_bin.exists() {
                Some(seg0_bin)
            } else if seg0_json.exists() {
                Some(seg0_json)
            } else if root_bin.exists() {
                Some(root_bin)
            } else if root_json.exists() {
                Some(root_json)
            } else {
                None
            }
        };
        let types_in_columns_bin: std::collections::HashSet<String> =
            if let Some(meta_path) = &columns_meta_path {
                use crate::graph::io::ntriples::ColumnTypeMeta;
                let metas: Vec<ColumnTypeMeta> =
                    if meta_path.extension().and_then(|s| s.to_str()) == Some("zst") {
                        let compressed = std::fs::read(meta_path)
                            .map_err(|e| format!("read {}: {}", meta_path.display(), e))?;
                        let bytes = zstd::decode_all(compressed.as_slice())
                            .map_err(|e| format!("decompress columns_meta: {}", e))?;
                        crate::graph::io::file::decode_disk_serde(&bytes, bytes.capacity() as u64)
                            .map_err(|e| format!("parse columns_meta.bin: {}", e))?
                    } else {
                        let json = std::fs::read_to_string(meta_path)
                            .map_err(|e| format!("read {}: {}", meta_path.display(), e))?;
                        serde_json::from_str(&json)
                            .map_err(|e| format!("parse columns_meta.json: {}", e))?
                    };
                metas.into_iter().map(|tm| tm.type_name).collect()
            } else {
                std::collections::HashSet::new()
            };

        let columns_dir = dir.join("columns");
        let mut sidecars_written = 0usize;
        for (type_name, store) in &self.column_stores {
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

        // 0.8.13: type_indices uses a flat CSR binary keyed by interner
        // hashes. 0.8.28+: id_indices uses an mmap-resident raw `.bin`
        // layout — load reads via memory-mapped binary search, no eager
        // HashMap rebuild. Backward-compat loaders fall through to the
        // old bincode/zstd paths when the new file is absent.
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

        // Save embeddings if any (matches write_kgl behavior for in-memory saves)
        if !self.embeddings.is_empty() {
            write_compressed_disk_serde(dir, "embeddings.bin.zst", &self.embeddings, "embeddings")?;
        }

        // Save timeseries_store if any
        if !self.timeseries_store.is_empty() {
            write_compressed_disk_serde(
                dir,
                "timeseries.bin.zst",
                &self.timeseries_store,
                "timeseries",
            )?;
        }

        // Root metadata is the graph-level completion marker. Publish it only
        // after every required sidecar has been written successfully.
        std::fs::write(dir.join("metadata.json"), meta_json)
            .map_err(|e| format!("Failed to publish metadata: {}", e))?;

        Ok(())
    }
}
