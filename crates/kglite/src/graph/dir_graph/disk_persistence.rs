//! Disk-mode lifecycle and persistence orchestration.

use super::*;
use crate::graph::storage::packed_codec::IntColumnEncoding;
use std::sync::atomic::{AtomicU64, Ordering};

static DISK_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A unique scratch-directory name for a disk conversion: `prefix` + pid +
/// wall-clock nanos + a process-local sequence, so two conversions in the same
/// process (and in the same nanosecond) cannot collide.
fn scratch_dir_name(prefix: &str) -> String {
    let sequence = DISK_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!(
        "{prefix}{}_{:x}_{sequence:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

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

/// An empty `ColumnStore` shaped like `source`: same schema, same column
/// types, no rows.
///
/// The column types come off the source store rather than the type's declared
/// metadata, so a column that was demoted (a heterogeneous property that
/// became `Mixed`) is reproduced as it actually is instead of being re-derived
/// into a shape its own values do not fit. An mmap-backed source carries its
/// schema in the mapping rather than in `schema()`, and yields an empty store
/// that regrows its columns from the first row of each key — the same path a
/// fresh ingest takes.
fn empty_store_like(source: &ColumnStore, interner: &StringInterner) -> ColumnStore {
    let schema = source.schema_arc();
    let mut column_types: HashMap<String, String> = HashMap::with_capacity(schema.len());
    for (slot, key) in schema.iter() {
        if let (Some(name), Some(type_str)) = (
            interner.try_resolve(key),
            source.column_type_str(slot as usize),
        ) {
            column_types.insert(name.to_string(), type_str.to_string());
        }
    }
    ColumnStore::new(schema, &column_types, interner)
}

/// Append `source`'s row `row_id` to `destination`, returning its new row id.
///
/// `scratch` is the caller's reusable property buffer — this runs once per live
/// node, and a fresh `Vec` per row was the allocation the in-memory rebuild
/// pass had to hoist out for exactly the same reason.
fn copy_row(
    source: &ColumnStore,
    row_id: u32,
    destination: &mut ColumnStore,
    scratch: &mut Vec<(InternedKey, Value)>,
) -> u32 {
    debug_assert!(
        !source.is_tombstoned(row_id),
        "a live node's row must not be tombstoned — copying one would silently \
         write an empty row for a node that still exists"
    );
    // id/title live in the store's reserved columns on a disk graph (the
    // node's inline fields hold the `Null` sentinel), so they are copied
    // through the same reserved columns — and only when the source has them,
    // or the copy would invent an all-null id column for a store that has
    // none.
    if source.has_id_title_columns() {
        destination.push_id(&source.get_id(row_id).unwrap_or(Value::Null));
        destination.push_title(&source.get_title(row_id).unwrap_or(Value::Null));
    }
    source.row_properties_into(row_id, scratch);
    destination.push_row(scratch)
}

impl DirGraph {
    /// Convert the graph to disk-backed storage mode, materializing the CSR
    /// into a **scratch directory under the system temp location**.
    ///
    /// Enables columnar storage first, then builds CSR edge arrays on disk.
    /// Nodes stay in memory (~40 bytes each), edges are mmap'd.
    ///
    /// The scratch directory is process-scoped: it is removed when this graph
    /// drops, so nothing here survives the process. A caller that wants the
    /// converted graph to *land* somewhere — on a chosen filesystem, in the
    /// published directory layout a later `kglite.open(dir)` reads — calls
    /// [`Self::enable_disk_mode_at`] instead, which materializes inside the
    /// destination and publishes it. Bindings surface that as
    /// `enable_disk_mode(path=...)`.
    pub fn enable_disk_mode(&mut self) -> Result<(), String> {
        let scratch = std::env::temp_dir().join(scratch_dir_name("kglite_disk_"));
        self.convert_to_disk_in(scratch)
    }

    /// Convert to disk-backed storage **and publish the result at `path`**,
    /// leaving the live handle on the published, mapped generation — the same
    /// end state `kglite.open(path)` reaches in a fresh process, and the state
    /// the directory's small resident footprint belongs to.
    ///
    /// Two things distinguish this from `enable_disk_mode()` +
    /// [`Self::save_disk`], and both are the reason it exists as one call:
    ///
    /// * the CSR is materialized **inside `path`**, not in the system temp
    ///   directory, so a conversion of a graph too large for `/tmp` (or for a
    ///   RAM-backed `tmpfs`) writes where the caller pointed it; and
    /// * the scratch directory is removed as soon as the publish rebases every
    ///   mapping onto the new generation, instead of lingering until drop —
    ///   so peak disk is one copy plus the staged generation, not two.
    ///
    /// A failed publish leaves the graph converted onto its scratch directory,
    /// which is registered for cleanup at drop exactly like the pathless form.
    pub(crate) fn enable_disk_mode_at(&mut self, path: &str) -> Result<(), String> {
        // Dot-prefixed and inside the destination, matching `MutationWorkspace`
        // (`.working-…`) and the generation stages (`.stage-…`): a graph
        // directory's own scratch is hidden, named by pid, and swept by the
        // same drop-time cleanup.
        let scratch = std::path::PathBuf::from(path).join(scratch_dir_name(".converting-"));
        self.convert_to_disk_in(scratch.clone())?;
        self.save_disk(path)?;
        // `finish_generation` re-mapped every array onto the published
        // generation (pinned by `save_rebases_every_mapping_onto_the_published_
        // generation`), so nothing reads through the scratch any more. Best
        // effort: if the removal fails the registration stays and drop retries.
        if std::fs::remove_dir_all(&scratch).is_ok() {
            if let Ok(mut dirs) = self.temp_dirs.lock() {
                dirs.retain(|dir| dir != &scratch);
            }
        }
        Ok(())
    }

    /// The conversion itself: columnar nodes, CSR + edge properties built into
    /// `data_dir`, backend switched to [`GraphBackend::Disk`]. Shared by both
    /// entry points above, which differ only in where `data_dir` lives and what
    /// happens to it afterwards.
    fn convert_to_disk_in(&mut self, data_dir: std::path::PathBuf) -> Result<(), String> {
        // Ensure columnar storage for compact node representation
        if self.column_store_count() == 0 {
            self.enable_columnar();
        }

        // Registered before the build, not after: `from_stable_digraph` creates
        // the directory as its first act and can fail with files already in it
        // (ENOSPC on a large conversion is the realistic case), and a failure
        // that never registered the path leaked it for the life of the process.
        if let Ok(mut dirs) = self.temp_dirs.lock() {
            dirs.push(data_dir.clone());
        }

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

    /// Merge a disk-mode graph's overflow edges back into its CSR arrays.
    /// Returns the number of overflow edges that were merged.
    /// No-op if there are no overflow edges.
    ///
    /// **Edges only.** This does not touch columnar rows: dead rows left by
    /// `DELETE` are dropped by [`Self::save_disk`], which rewrites the columns
    /// without them (see [`Self::drop_dead_column_rows`]). Node slots freed by
    /// a delete are not reclaimed by either — a disk graph's node capacity only
    /// shrinks when the directory is rebuilt from a fresh ingest.
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

    /// Write the graph's complete on-disk form into `dir` — the payload half
    /// of a generation publish, with no pointer swap and no rebase of the live
    /// handle. `pub(crate)` because disk-mode *creation* stages its initial
    /// empty generation through it too (`storage::mode`).
    pub(crate) fn write_disk_snapshot(&mut self, dir: &std::path::Path) -> Result<(), String> {
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
        let mut rewriting = false;
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            dg.begin_persist();
            rewriting = dg.save_disposition(dir)
                == crate::graph::storage::disk::graph_persist::SaveDisposition::Rewrite;
            if rewriting && dg.has_overflow() {
                dg.compact()
                    .map_err(|e| format!("disk compaction failed: {e}"))?;
            }
        }
        // Drop columnar rows no live node points at, so the columns this save
        // writes carry only live data. Rewrite only: a seal extends the
        // graph's current root and leaves the already-published columns in
        // place, so renumbering rows there would leave every slot naming a row
        // the published file does not have.
        if rewriting {
            self.drop_dead_column_rows();
        }
        if let GraphBackend::Disk(ref mut dg) = self.graph {
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

    /// Drop every columnar row no live node points at, renumbering the rows
    /// that survive and the node slots that name them. Returns the number of
    /// rows dropped.
    ///
    /// # Why a save needs this
    ///
    /// `DELETE` on a disk graph tombstones the node's slot and leaves its row
    /// in the type's `ColumnStore` — the store is append-only under mutation,
    /// and `vacuum()` cannot reclaim the row because a disk graph's node
    /// numbering is frozen mmap. So the garbage accumulated across the
    /// process's whole lifetime *and then got written*: measured before this
    /// existed, a 20 k-node graph with half its nodes deleted wrote its
    /// columns at 2.00x the size of the same graph built from the survivors,
    /// and a reload censused every dead row back in — so the next save wrote
    /// them again.
    ///
    /// # Why renumbering rows here is safe
    ///
    /// A disk graph binds a node to its row **explicitly**, through
    /// `DiskNodeSlot::row_id`, and the slots are written by the same save that
    /// writes the columns (`save_logical_node_slots`). Row ids are therefore
    /// private to the pair of artifacts and can be reassigned as long as both
    /// are rewritten together — which is exactly the rewrite disposition this
    /// runs under. Nothing else is a row coordinate: `type_indices` /
    /// `id_indices` and the global property indexes key on `NodeIndex`, and
    /// the property indexes are (re)built after this returns.
    ///
    /// **Node slots are not renumbered**, only their `row_id` field — the
    /// petgraph/CSR node numbering is untouched, so every held selection,
    /// edge endpoint and index entry keeps meaning what it meant.
    fn drop_dead_column_rows(&mut self) -> usize {
        // Staged node writes still name *old* row ids, so they must land
        // before anything is renumbered. `clear_arenas` is what `save_to_dir`
        // would call moments later anyway; calling it here only moves it
        // ahead of the renumbering.
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            dg.clear_arenas();
        }

        // Live rows per type, from the slots themselves — the census on
        // `DirGraph` counts `type_indices`, which is a different structure and
        // must not be the one that decides what gets copied.
        let slot_count = match &self.graph {
            GraphBackend::Disk(dg) => dg.node_slot_len(),
            _ => return 0,
        };
        let mut live_rows: HashMap<InternedKey, usize> = HashMap::new();
        if let GraphBackend::Disk(dg) = &self.graph {
            for index in 0..slot_count {
                let slot = dg.node_slot(index);
                if slot.is_alive() {
                    *live_rows
                        .entry(InternedKey::from_u64(slot.node_type))
                        .or_insert(0) += 1;
                }
            }
        }

        // Only types that actually carry garbage are rebuilt: a clean type
        // keeps its store as-is, mmap base included.
        let mut dropped = 0usize;
        let mut destinations: HashMap<InternedKey, (Arc<ColumnStore>, ColumnStore)> =
            HashMap::new();
        for (type_key, store) in self.graph.column_stores_iter() {
            let live = live_rows.get(&type_key).copied().unwrap_or(0);
            let total = store.row_count() as usize;
            if total <= live {
                continue;
            }
            dropped += total - live;
            let empty = empty_store_like(store, &self.interner);
            destinations.insert(type_key, (Arc::clone(store), empty));
        }
        if destinations.is_empty() {
            return 0;
        }

        // One pass over the slots, copying each live row into its type's
        // replacement store and repointing the slot at the new row. Slot order
        // is ascending node order, so the rewritten rows keep the locality the
        // original build gave them.
        //
        // **Cost.** Nothing here builds a whole-graph structure of its own, but
        // two allocations are sized by the *compacted types*: the replacement
        // stores (the live data itself, heap-resident even where the store they
        // replace was mmap-backed) and the node-slot overlay, which takes one
        // entry per renumbered slot because a published generation's
        // `node_slots.bin` may be mapped by other readers and must not be
        // written through. A rewriting save already materialises every column
        // as bytes to write `columns.bin`, so this is a constant-factor
        // increase on a path that was already sized by the graph — not a new
        // order of growth. It is also skipped entirely for a type with no dead
        // rows, which is every type of a graph that has not deleted anything.
        let mut pairs: Vec<(InternedKey, Value)> = Vec::new();
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            for index in 0..slot_count {
                let slot = dg.node_slot(index);
                if !slot.is_alive() {
                    continue;
                }
                let type_key = InternedKey::from_u64(slot.node_type);
                let Some((source, destination)) = destinations.get_mut(&type_key) else {
                    continue;
                };
                let row_id = copy_row(source, slot.row_id, destination, &mut pairs);
                if row_id != slot.row_id {
                    let mut moved = slot;
                    moved.row_id = row_id;
                    dg.set_node_slot(index, moved);
                }
            }
        }

        for (type_key, (_, store)) in destinations {
            GraphWrite::install_column_store(&mut self.graph, type_key, Arc::new(store));
        }
        // The replacements are heap-resident even where the store they replace
        // was mmap-backed, so the memory limit has to be re-enforced against
        // them — same obligation the in-memory rebuild discharges at the end of
        // `rebuild_column_stores`. No-op when no limit is set.
        self.maybe_spill_columns();
        dropped
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
