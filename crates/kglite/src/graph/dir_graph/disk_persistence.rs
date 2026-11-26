//! Disk-mode lifecycle and persistence orchestration.

use super::*;
use crate::graph::io::columns_meta::ColumnsMeta;
use crate::graph::io::file::SaveError;
use crate::graph::storage::packed_codec::IntColumnEncoding;
use std::sync::atomic::{AtomicU64, Ordering};

static DISK_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Column stores by type name, as one save writes them.
type SaveStores = HashMap<String, Arc<ColumnStore>>;

/// The column metadata of the stage a save is assembling: read from a stage an
/// ntriples build left, or returned by the column-file writer, then given its
/// `sidecars` and published once.
struct StageColumns {
    meta: ColumnsMeta,
    /// Where the metadata is published; its `files` resolve against it.
    seg0: std::path::PathBuf,
    /// Whether `publish` has anything to write.
    dirty: bool,
    /// Whether this save wrote the column files the metadata names, which is
    /// what lets the post-publish remap use it instead of reading it back.
    fresh: bool,
}

impl StageColumns {
    fn written(meta: ColumnsMeta, seg0: std::path::PathBuf) -> Self {
        let dirty = !meta.types.is_empty();
        Self {
            meta,
            seg0,
            dirty,
            fresh: true,
        }
    }

    fn existing(meta: ColumnsMeta, seg0: &std::path::Path) -> Self {
        Self {
            meta,
            seg0: seg0.to_path_buf(),
            dirty: false,
            fresh: false,
        }
    }

    /// The node types whose bytes already live in the stage's column files.
    fn covered_types(&self) -> std::collections::HashSet<String> {
        self.meta
            .types
            .iter()
            .map(|ty| ty.type_name.clone())
            .collect()
    }

    /// Record which zstd sidecar directory holds each type the column files do
    /// not, so a reader learns a sidecar's type from the metadata and never from
    /// a directory name.
    fn record_sidecars(&mut self, sidecars: std::collections::BTreeMap<String, String>) {
        if !sidecars.is_empty() {
            self.meta.sidecars = sidecars;
            self.dirty = true;
        }
    }

    /// Write the metadata (both encodings) if the save produced or changed it,
    /// and hand it back when the post-publish remap can use it.
    fn publish(self) -> Result<Option<ColumnsMeta>, String> {
        if self.dirty {
            std::fs::create_dir_all(&self.seg0)
                .and_then(|()| crate::graph::io::columns_meta::publish(&self.seg0, &self.meta))
                .map_err(|e| format!("Failed to write column metadata: {}", e))?;
        }
        Ok((self.fresh && !self.meta.types.is_empty()).then_some(self.meta))
    }
}

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
/// `scratch` is the caller's reusable property buffer: this runs once per live
/// node, so a fresh `Vec` per row would allocate once per node.
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
    /// Node slots become a 16-byte-per-node mmap'd array alongside the mmap'd
    /// CSR and edge properties.
    ///
    /// The scratch directory is process-scoped: it is removed when this graph
    /// drops, so nothing here survives the process. To *land* the converted
    /// graph somewhere instead, see [`Self::enable_disk_mode_at`] — surfaced by
    /// the bindings as `enable_disk_mode(path=...)`.
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
        // There is no DirGraph-level mirror to push them in afterwards.
        let carried_stores: HashMap<InternedKey, Arc<ColumnStore>> = self
            .graph
            .column_stores_iter()
            .map(|(k, v)| (k, Arc::clone(v)))
            .collect();

        // A forked backend has no single `StableDiGraph` to hand over, so
        // collapse it first. Free when the reader has already dropped.
        self.graph.flatten_fork();
        // A frozen topology is read-only: the heap graph the conversion packs is
        // what promotion builds.
        self.graph.promote_frozen();

        // `from_stable_digraph` packs live heap edges densely in the same
        // order as petgraph's edge iterator. Preserve sidecar association
        // when deleted heap slots left holes before this conversion.
        let edge_remap = if self.edge_embeddings.is_empty() {
            None
        } else {
            let mut remap =
                crate::graph::edge_embeddings::EdgeRemap::with_bound(self.graph.edge_bound());
            for (new, old) in self.graph.edge_indices().enumerate() {
                remap.set(old, petgraph::graph::EdgeIndex::new(new));
            }
            Some(remap)
        };

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
            GraphBackend::Frozen(_) => unreachable!("promote_frozen replaced it above"),
            GraphBackend::Disk(_) => return Err("Already in disk mode".to_string()),
            GraphBackend::Recording(_) => {
                return Err(
                    "enable_disk_mode not supported while wrapped in RecordingGraph".to_string(),
                )
            }
        }
        .map_err(|e| format!("Failed to create DiskGraph: {}", e))?;

        self.graph = GraphBackend::Disk(Box::new(disk_graph));
        if let Some(remap) = edge_remap {
            crate::graph::edge_embeddings::remap_edge_embeddings(self, &remap);
        }
        for (type_key, store) in carried_stores {
            GraphWrite::install_column_store(&mut self.graph, type_key, store);
        }
        Ok(())
    }

    /// Prepare a direct mutation by materializing deferred index declarations.
    /// Disk graphs acquire their writer lease and workspace first, so a refused
    /// writer leaves the deferred state untouched. The lease lasts until
    /// [`Self::save_disk`] rebases the handle onto the published generation.
    /// Already-materialized memory and mapped graphs only check the flag.
    pub(crate) fn prepare_mutation(&mut self) -> std::io::Result<()> {
        if let GraphBackend::Disk(disk) = &mut self.graph {
            disk.prepare_mutation()?;
        }
        self.materialize_indexes();
        Ok(())
    }

    /// Build CSR from pending edges if in disk mode. No-op otherwise.
    /// Called after `add_connections` and Cypher edge creation, so queries see
    /// them, and by the save path.
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
        self.prepare_mutation()
            .map_err(|e| format!("disk mutation lease failed: {e}"))?;
        let (count, remap) = match &mut self.graph {
            GraphBackend::Disk(ref mut dg) => {
                dg.compact_with_edge_remap().map_err(|e| e.to_string())?
            }
            _ => return Err("compact requires disk mode".to_string()),
        };
        if let Some(remap) = remap {
            let remap = crate::graph::edge_embeddings::EdgeRemap::from_raw(remap);
            crate::graph::edge_embeddings::remap_edge_embeddings(self, &remap);
        }
        Ok(count)
    }

    /// Fold a disk graph's overflow edges into its CSR before a write when the
    /// bag is large enough that copying it on every write costs more than one
    /// fold. A graph built in-process holds its whole edge set as overflow until
    /// `save()` or `compact()`, so without this its first write — and every one
    /// after — pays O(edges). The threshold bounds the fold's amortized cost at
    /// four edges' worth of work per overflow edge added since the last fold.
    /// A failed fold leaves the graph as it was; the write proceeds on the slow
    /// path. No-op outside disk mode.
    pub(crate) fn compact_disk_if_overflow_large(&mut self) {
        const MIN_OVERFLOW: usize = 100_000;
        let Some(disk) = self.graph.as_disk() else {
            return;
        };
        let overflow = disk.overflow_edge_count();
        if overflow < MIN_OVERFLOW.max(disk.edge_count / 4) {
            return;
        }
        let _ = self.compact_disk();
    }

    /// Save a disk-mode graph to a directory. The directory IS the graph.
    /// Persists CSR files, node data, edge properties, column stores, and metadata.
    ///
    /// Refused with [`SaveError::Refused`] when another writer published a
    /// generation since this handle loaded or last published one: the save
    /// would build the new generation from this handle's older state and
    /// silently drop the other writer's work. Nothing is written.
    pub fn save_disk(&mut self, path: &str) -> Result<(), SaveError> {
        let root = std::path::PathBuf::from(path);
        if let Some(disk) = self.graph.as_disk_mut() {
            disk.detach_ended_lineage();
        }
        let writer_lock = match &mut self.graph {
            GraphBackend::Disk(disk)
                if disk
                    .writer_lock
                    .as_ref()
                    .is_some_and(|lock| lock.root == root && lock.is_active()) =>
            {
                disk.writer_lock.as_ref().unwrap().clone()
            }
            // Not `try_acquire`: after a publish this handle holds no lease,
            // and a lineage sibling may still hold the one it published under
            // — `take_lease` re-joins that rather than deadlocking on it.
            GraphBackend::Disk(disk) => disk
                .take_lease(&root)
                .map_err(|e| format!("Failed to acquire disk writer lock: {e}"))?,
            _ => return Err("save_disk requires disk mode".to_string().into()),
        };
        if let GraphBackend::Disk(disk) = &mut self.graph {
            disk.writer_lock = Some(writer_lock.clone());
            disk.prepare_mutation()
                .map_err(|e| format!("Failed to prepare disk workspace: {e}"))?;
        }
        let publication = writer_lock
            .publication_permit()
            .map_err(|e| format!("Failed to retain disk publication authority: {e}"))?;
        if let GraphBackend::Disk(disk) = &self.graph {
            disk.ensure_current_generation_unchanged(&root)?;
        }
        let mut clock = crate::graph::io::load_timing::StageClock::start();
        let generation = crate::graph::storage::disk::generation::GenerationTxn::begin(&root)
            .map_err(|e| format!("Failed to begin disk generation: {e}"))?;
        clock.mark("save_begin");
        let columns =
            self.write_disk_snapshot(generation.stage_dir(), generation.previous_snapshot())?;
        clock.mark("save_snapshot_total");
        let published = generation
            .publish()
            .map_err(|e| format!("Failed to publish disk generation: {e}"))?;
        clock.mark("save_publish");
        // The save is durable from here on; this only shrinks the handle's
        // footprint and never fails the call (see `disk_rebase`). It runs before
        // `finish_generation` clears the mutation workspace: an append parks
        // file-backed columns there, and a store still mapping them would keep
        // the directory alive on platforms that refuse to remove mapped files.
        self.rebase_after_publish(&published, columns);
        clock.mark("save_rebase");
        if let GraphBackend::Disk(disk) = &mut self.graph {
            disk.finish_generation(root.clone(), published)
                .map_err(|e| format!("Failed to activate published disk generation: {e}"))?;
        }
        clock.mark("save_finish_generation");
        // Older generations than the kept window go now that this handle maps
        // the new one. Last, and never an error: a generation something still
        // maps, or the platform will not delete, waits for a later save.
        crate::graph::storage::disk::generation::prune_generations(
            &root,
            crate::graph::storage::disk::generation::keep_previous_generations(),
        );
        clock.mark("save_prune_generations");
        // The lease was held across the whole snapshot-write / publish /
        // rebase sequence by this local `Arc`; `finish_generation` cleared the
        // graph's own reference, so dropping it here is what actually releases
        // the directory — unless a live transaction fork cloned the lease, in
        // which case the lock outlives this save by design.
        drop(publication);
        drop(writer_lock);
        Ok(())
    }

    /// Write the graph's complete on-disk form into `dir` — the payload half
    /// of a generation publish, with no pointer swap and no rebase of the live
    /// handle. `pub(crate)` because disk-mode *creation* stages its initial
    /// empty generation through it too (`storage::mode`).
    ///
    /// `previous` is the published generation this one replaces; a node type
    /// whose column file nothing has changed is linked from it rather than
    /// rewritten (see `io::column_link`).
    ///
    /// Returns the column metadata the stage's `seg_000/` now holds when this
    /// call wrote it, so the post-publish remap need not read it back.
    pub(crate) fn write_disk_snapshot(
        &mut self,
        dir: &std::path::Path,
        previous: Option<&std::path::Path>,
    ) -> Result<Option<ColumnsMeta>, String> {
        let mut clock = crate::graph::io::load_timing::StageClock::start();
        self.consolidate_disk_for_save(dir)?;
        clock.mark("save_consolidate");

        let edge_embeddings_required =
            crate::graph::edge_embeddings::has_persisted_edge_embeddings(self);

        // save_to_dir needs &mut access so the edge-property store can
        // drop its base mmap before overwriting.
        let dg = match &mut self.graph {
            GraphBackend::Disk(dg) => dg,
            _ => return Err("save_disk requires disk mode".to_string()),
        };

        // `save_to_dir` runs `flush_mut_caches` internally, which drains
        // `node_mut_cache` into each mutated type's store in
        // `DiskGraph.column_stores`.
        dg.save_to_dir_with_edge_embeddings(dir, &self.interner, edge_embeddings_required)
            .map_err(|e| format!("DiskGraph save failed: {}", e))?;
        clock.mark("save_csr_and_edges");
        // No mirror to refresh: `DiskGraph` *is* the owner of the column
        // stores, so the sidecar writer below reads the same `Arc`s
        // `save_to_dir` just flushed into. An earlier shape kept a
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
        clock.mark("save_type_metadata");
        // Secondary labels — disk's columnar layout has no slot for
        // NodeData.extra_labels, so we persist the inverted index as
        // a sidecar. Skipped when the graph has no secondaries
        // (single-label disk graphs pay zero extra bytes).
        crate::graph::io::file::write_secondary_labels_bin(dir, self)?;
        // Snapshot live declarations; a deferred graph retains its loaded lists.
        self.populate_index_keys();
        let mut meta = crate::graph::io::file::build_disk_metadata(self);
        crate::graph::io::file::strip_type_connectivity(&mut meta);
        crate::graph::io::file::strip_heavy_metadata(&mut meta);
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| format!("Metadata serialization failed: {}", e))?;
        // No-op when the type-connectivity cache is empty.
        crate::graph::io::file::write_type_connectivity_bin(dir, self)?;

        // The framed interner sidecar stores `Vec<String>`; hashes are
        // re-derived on load. Unframed binary data is rejected, while the
        // older JSON representation remains a read-only data fallback.
        crate::graph::io::file::write_interner_bin(dir, self)?;
        clock.mark("save_interner_and_metadata");

        let stores = self.column_stores_for_save();
        clock.mark("save_columns_prepare");
        let mut columns = self.write_unified_column_file(dir, &stores, previous)?;
        clock.mark("save_columns_files");
        self.write_column_sidecars(dir, &stores, &mut columns)?;
        drop(stores);
        let columns = columns.publish()?;
        clock.mark("save_column_sidecars");

        // 0.8.13: type_indices uses a flat CSR binary keyed by interner
        // hashes. 0.8.28+: id_indices uses an mmap-resident raw `.bin`
        // layout — load reads via memory-mapped binary search, no eager
        // HashMap rebuild. The loader can still read the earlier flat-CSR
        // sidecars when the mmap files are absent; pre-0.14 bincode caches
        // are ignored and rebuilt.
        // An index nothing has written over is the previous generation's file
        // as it stands, so the stage links it; `id_indices.bin` is derived from
        // both stores (a type whose ids repeat is left out by comparing them),
        // so it is carried only while both are unchanged.
        let types_unchanged = self.type_indices.unchanged_origin();
        let types_linked = crate::graph::io::column_link::carry_unchanged_index(
            previous,
            "type_indices.bin",
            types_unchanged,
            dir,
        );
        if !types_linked {
            crate::graph::storage::disk::type_index::write_type_indices_bin(
                dir,
                &self.type_indices,
                &self.interner,
            )?;
        }
        clock.mark("save_type_indices");
        let ids_linked = types_unchanged.is_some()
            && crate::graph::io::column_link::carry_unchanged_index(
                previous,
                "id_indices.bin",
                self.id_indices.unchanged_origin(),
                dir,
            );
        if !ids_linked {
            crate::graph::storage::disk::id_index::write_id_indices_bin(
                dir,
                &self.id_indices,
                &self.type_indices,
                &self.interner,
            )?;
        }
        clock.mark("save_id_indices");

        // BTreeMap view for byte-determinism — same rationale as write_kgl.
        if !self.embeddings.is_empty() {
            // A mapped store streams from its mapping: refuse a corrupt one
            // here rather than copying it into a fresh, valid-looking sidecar.
            for store in self.embeddings.values() {
                store.verify_storage().map_err(|e| e.to_string())?;
            }
            let ordered: std::collections::BTreeMap<_, _> = self.embeddings.iter().collect();
            write_compressed_disk_serde(dir, "embeddings.bin.zst", &ordered, "embeddings")?;
        }

        if edge_embeddings_required {
            for store in self.edge_embeddings.values() {
                store
                    .numeric()
                    .verify_storage()
                    .map_err(|e| e.to_string())?;
            }
            let ordered =
                crate::graph::edge_embeddings::persisted_edge_embedding_stores(self, |_| false);
            write_compressed_disk_serde(
                dir,
                "edge_embeddings.bin.zst",
                &ordered,
                "edge embeddings",
            )?;
        }

        if !self.timeseries_store.is_empty() {
            let ordered: std::collections::BTreeMap<_, _> = self.timeseries_store.iter().collect();
            write_compressed_disk_serde(dir, "timeseries.bin.zst", &ordered, "timeseries")?;
        }

        // Root metadata is the graph-level completion marker. Publish it only
        // after every required sidecar has been written successfully.
        std::fs::write(dir.join("metadata.json"), meta_json)
            .map_err(|e| format!("Failed to publish metadata: {}", e))?;

        Ok(columns)
    }

    /// Bring a disk graph's CSR, overflow edges and global indexes to their
    /// final saved state before anything is written.
    fn consolidate_disk_for_save(&mut self, dir: &std::path::Path) -> Result<(), String> {
        let mut clock = crate::graph::io::load_timing::StageClock::start();
        self.ensure_disk_edges_built()?;
        clock.mark("save_consolidate_edges_built");
        // Merge overflow edges back so conn_type_index and
        // peer_count_histogram reflect every live edge — the one-shot the
        // per-batch build path defers (see `ensure_disk_edges_built`).
        //
        // The seal path consumes `overflow_out` / `overflow_in` directly,
        // while a rewrite must compact first so every derived index is rebuilt
        // from the complete CSR. Use the lower layer's target-aware decision:
        // generation stages are distinct directories and always rewrite.
        let mut rewriting = false;
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            rewriting = dg.save_disposition(dir)
                == crate::graph::storage::disk::graph_persist::SaveDisposition::Rewrite;
            if rewriting && dg.has_overflow() {
                let (_, remap) = dg
                    .compact_with_edge_remap()
                    .map_err(|e| format!("disk compaction failed: {e}"))?;
                if let Some(remap) = remap {
                    let remap = crate::graph::edge_embeddings::EdgeRemap::from_raw(remap);
                    crate::graph::edge_embeddings::remap_edge_embeddings(self, &remap);
                }
            }
        }
        // Drop columnar rows no live node points at, so the columns this save
        // writes carry only live data. Rewrite only: a seal extends the
        // graph's current root and leaves the already-published columns in
        // place, so renumbering rows there would leave every slot naming a row
        // the published file does not have.
        clock.mark("save_consolidate_compact");
        if rewriting {
            self.drop_dead_column_rows();
        }
        clock.mark("save_consolidate_drop_dead_rows");
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            // The cross-type global title index makes
            // `MATCH (n {title: 'X'})` and `g.search(text)` O(log N) out of the
            // box on every saved disk graph. Runs after CSR / overflow
            // consolidation so it sees the final node set. Tied to `save_disk`
            // rather than `build_csr_*` so node-only graphs (no edges) still
            // get the index built. One that already exists is left for the
            // refresh below to rebuild only if the graph has moved under it.
            dg.build_global_property_index_if_absent("title")
                .map_err(|e| format!("title index build failed: {e}"))?;
            clock.mark("save_consolidate_title_index");
            // Likewise index `nid` — the string id form for prefixed-id
            // datasets (Wikidata `"Q42"`). Since 0.11.0 `{nid: 'Q42'}` is a
            // plain string-property lookup (not the integer id-index), so the
            // index keeps it O(log N) instead of a 124M-row scan. No-op when
            // no type has a `nid` column.
            dg.build_global_property_index_if_absent("nid")
                .map_err(|e| format!("nid index build failed: {e}"))?;
            clock.mark("save_consolidate_nid_index");
            // Every bundle the next generation will carry — `title` and `nid`
            // included — rebuilt if the graph has moved under it.
            // `copy_persisted_indexes` copies bundles into the new generation
            // verbatim, and a stale one that crosses a generation boundary
            // re-arms the defect in the next process — where nothing remembers
            // it was stale. Fresh bundles are skipped, so an unmutated save
            // scans no node for any of them.
            dg.refresh_persistent_indexes(false)
                .map_err(|e| format!("property index rebuild failed: {e}"))?;
            clock.mark("save_consolidate_refresh_indexes");
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
        // before anything is renumbered. `flush_mut_caches` is what `save_to_dir`
        // would call moments later anyway; calling it here only moves it
        // ahead of the renumbering.
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            dg.flush_mut_caches();
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

        // Slot order is ascending node order, so the rewritten rows keep the
        // locality the original build gave them.
        //
        // **Cost.** Two allocations are sized by the *compacted types*: the
        // replacement stores, and the node-slot overlay, which takes one entry
        // per renumbered slot because a published generation's
        // `node_slots.bin` may be mapped by other readers and must not be
        // written through. A rewriting save already writes every column's bytes
        // into its type file, so this is a constant factor on a path already
        // sized by the graph, not a new order of growth.
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

    /// Write one column file per node type for a graph whose stage has none, and
    /// return the stage's column metadata, not yet published.
    ///
    /// Since 0.9.15 a fresh save — streaming carve, `save_subset`, or the
    /// mutation persist of an in-memory build — emits the region layout the
    /// ntriples builder produces, so the saved graph reloads through the mmap
    /// fast path. Without it a saved `DiskGraph` fell back to per-type zstd
    /// sidecars and took ~70 s to load on a 17 M-node Wikidata carve against
    /// ~150 ms for the full graph.
    ///
    /// A stage that already holds column metadata (or a legacy `columns.bin`)
    /// means the ntriples builder wrote one; that metadata is read, and
    /// [`Self::write_column_sidecars`] covers the types added since.
    fn write_unified_column_file(
        &self,
        dir: &std::path::Path,
        stores: &SaveStores,
        previous: Option<&std::path::Path>,
    ) -> Result<StageColumns, String> {
        let located = crate::graph::io::columns_meta::locate(dir);
        let has_shared_file =
            dir.join("seg_000/columns.bin").exists() || dir.join("columns.bin").exists();
        if located.is_some() || has_shared_file || stores.is_empty() {
            return match located {
                Some(path) => {
                    let meta = crate::graph::io::columns_meta::read(&path)
                        .map_err(|e| format!("read {}: {}", path.display(), e))?;
                    Ok(StageColumns::existing(meta, path.parent().unwrap_or(dir)))
                }
                None => Ok(StageColumns::existing(
                    ColumnsMeta::default(),
                    &dir.join("seg_000"),
                )),
            };
        }
        let previous = previous.and_then(crate::graph::io::column_link::PreviousColumns::open);
        let meta = crate::graph::io::unified_columns::write_unified_columns(
            dir,
            stores,
            previous.as_ref(),
        )
        .map_err(|e| format!("unified columns write failed: {}", e))?;
        Ok(StageColumns::written(meta, dir.join("seg_000")))
    }

    /// The column stores a save writes, by type name: the live stores, except
    /// that an mmap-backed store with local changes the column regions cannot
    /// hold is flattened into an owned typed store (one that is nothing but its
    /// base, or that base with same-kind `SET` columns and a same-kind tail, is
    /// left alone and written from its mapping), and a heap store holding a
    /// `Mixed` column whose values all have one kind is
    /// re-typed (a graph an earlier build saved through the all-`Mixed`
    /// sidecar writer heals on its next save). A column holding values of
    /// several kinds stays `Mixed`, and so in a sidecar: typing it would
    /// convert values (see `ColumnStore::flattened_owned`).
    ///
    /// Written as is, an mmap-backed store with local changes can only go to a
    /// sidecar, and one whose every column is `Mixed` (id and title included);
    /// the next load decodes that onto the heap and every later save keeps it
    /// there — so a reopened graph left its mmap-served column files on its
    /// first re-save, write or not. The copies are save-scoped: the live stores
    /// keep serving from the mapping they have until the post-publish re-point.
    pub(super) fn column_stores_for_save(&self) -> SaveStores {
        self.column_stores_by_name()
            .into_iter()
            .map(|(name, store)| {
                let empty = HashMap::new();
                let meta = self.node_type_metadata.get(name).unwrap_or(&empty);
                // A tail of sealed segments is merged into one before the
                // region plan looks at it; the plan reads a single tail.
                let merged = match store.with_consolidated_tail() {
                    std::borrow::Cow::Borrowed(_) => Arc::clone(store),
                    std::borrow::Cow::Owned(whole) => Arc::new(whole),
                };
                let store = &merged;
                // A store that is nothing but an mmap base re-emits its regions
                // directly (`write_unified_columns`), so it is not flattened;
                // neither is one that is that base with `SET` values over it
                // and a tail, all of the same column kinds, whose regions are
                // the base's with the overlay written over them.
                let flatten = (store.has_mmap_base()
                    && store.pure_mmap_store().is_none()
                    && store.region_parts().is_none())
                    || store.has_retypable_mixed_column(meta);
                let store = if flatten {
                    Arc::new(store.flattened_owned(meta, &self.interner))
                } else if store.has_heap_tail() {
                    Arc::new(store.with_heap_tail_folded().into_owned())
                } else {
                    Arc::clone(store)
                };
                (name.to_string(), store)
            })
            .collect()
    }

    /// Write a per-type `columns.zst` sidecar for every type the unified
    /// column files do not cover.
    ///
    /// Types added post-build via `add_nodes` / `add_node` are absent from an
    /// ntriples-built `columns.bin` and were silently dropped on save before
    /// this existed. Covered types keep the fast mmap path.
    ///
    /// A sidecar's directory is named by the type's interned key and recorded in
    /// the stage's column metadata (`sidecars`); the type name is data and never
    /// reaches the filesystem.
    fn write_column_sidecars(
        &self,
        dir: &std::path::Path,
        stores: &SaveStores,
        columns: &mut StageColumns,
    ) -> Result<(), String> {
        let covered = columns.covered_types();
        let mut type_names: Vec<&String> = stores
            .keys()
            .filter(|name| !covered.contains(name.as_str()))
            .collect();
        type_names.sort();
        let mut used = std::collections::HashSet::new();
        let mut recorded = std::collections::BTreeMap::new();
        for type_name in type_names {
            let store = &stores[type_name];
            let relative = crate::graph::io::columns_meta::sidecar_dir_name(type_name, &mut used);
            let type_dir = dir.join(&relative);
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
            recorded.insert(type_name.clone(), relative);
        }
        columns.record_sidecars(recorded);
        Ok(())
    }
}

#[cfg(test)]
#[path = "disk_deferred_tests.rs"]
mod disk_deferred_tests;

#[cfg(test)]
#[path = "disk_save_shape_tests.rs"]
mod disk_save_shape_tests;
