//! `load_ntriples` entry point + columnar build pipeline.
//!
//! Streaming single-pass load: parse → accumulate → flush → build columns.
//! Supports mem/mapped/disk storage modes; disk mode uses an overflow
//! edge buffer and mmap-backed column builders.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, InternedKey, NodeData, PropertyStorage};
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use crate::graph::storage::type_build_meta::TypeBuildMeta;
use crate::graph::storage::{GraphRead, GraphWrite};
use flate2::read::GzDecoder;
use std::collections::HashMap;
#[cfg(test)]
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use super::column_builder::ColumnTypeMeta;
use super::parser::{
    extract_lang_text, language_matches, parse_line, parse_qcode_number, typed_literal_to_value,
    CompactNTripleEdge, EdgeBuffer, EntityAccumulator, Object, Predicate, Subject,
};
use super::writer::{create_edges_from_buffer, create_edges_with_qnum_map};
use super::{
    Cancelled, NTriplesConfig, NTriplesStats, ProgressEvent, ProgressSink, ProgressValue as PV,
};

macro_rules! eplog {
    ($($arg:tt)*) => {
        eprintln!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), format_args!($($arg)*))
    };
}

/// Sentinel string returned from `load_ntriples` when the configured
/// `ProgressSink` requests cancellation. The pyapi layer maps this to
/// `PyKeyboardInterrupt` so users see the right exception type.
const CANCELLED_TOKEN: &str = "<cancelled>";
const READER_BATCH_SIZE: usize = 200_000;
const READER_TARGET_BATCH_BYTES: usize = 16 * 1024 * 1024;

type ReaderBatch = Result<super::parser::LineBuffer, String>;

fn spawn_reader(
    reader: Box<dyn Read + Send>,
) -> (
    std::sync::mpsc::Receiver<ReaderBatch>,
    std::thread::JoinHandle<Result<(), String>>,
) {
    let (tx, rx) = std::sync::mpsc::sync_channel::<ReaderBatch>(32);
    let handle = std::thread::spawn(move || {
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, reader);
        let mut raw = Vec::with_capacity(512);
        let mut batch =
            super::parser::LineBuffer::with_capacity(READER_BATCH_SIZE, READER_TARGET_BATCH_BYTES);
        let prefix: &[u8] = b"<http://www.wikidata.org/entity/Q";

        loop {
            raw.clear();
            let bytes_read = match reader.read_until(b'\n', &mut raw) {
                Ok(bytes_read) => bytes_read,
                Err(error) => {
                    let message = format!("N-Triples reader error: {error}");
                    if !batch.is_empty() {
                        let next = super::parser::LineBuffer::with_capacity(
                            READER_BATCH_SIZE,
                            READER_TARGET_BATCH_BYTES,
                        );
                        let full = std::mem::replace(&mut batch, next);
                        if tx.send(Ok(full)).is_err() {
                            return Ok(());
                        }
                    }
                    if tx.send(Err(message.clone())).is_err() {
                        return Ok(());
                    }
                    return Err(message);
                }
            };
            if bytes_read == 0 {
                if !batch.is_empty() && tx.send(Ok(batch)).is_err() {
                    return Ok(());
                }
                return Ok(());
            }

            // Byte-level fast reject stays ahead of UTF-8 validation. Only
            // accepted entity lines enter the batch and pay validation cost.
            if !raw.starts_with(prefix) {
                continue;
            }

            batch.push_line(&raw);
            if batch.offsets.len() >= READER_BATCH_SIZE
                || batch.data.len() >= READER_TARGET_BATCH_BYTES
            {
                let next = super::parser::LineBuffer::with_capacity(
                    READER_BATCH_SIZE,
                    READER_TARGET_BATCH_BYTES,
                );
                let full = std::mem::replace(&mut batch, next);
                if tx.send(Ok(full)).is_err() {
                    return Ok(());
                }
            }
        }
    });
    (rx, handle)
}

fn join_reader(handle: std::thread::JoinHandle<Result<(), String>>) -> Result<(), String> {
    match handle.join() {
        Ok(result) => result,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            Err(format!("N-Triples reader thread panicked: {detail}"))
        }
    }
}

#[inline]
fn validated_line(bytes: &[u8]) -> Result<&str, std::str::Utf8Error> {
    if bytes.is_ascii() {
        // SAFETY: every ASCII byte is a one-byte UTF-8 code point. `is_ascii`
        // is the validation step and is substantially cheaper on the common
        // Wikidata URI/numeric fast path than the general UTF-8 state machine.
        Ok(unsafe { std::str::from_utf8_unchecked(bytes) })
    } else {
        std::str::from_utf8(bytes)
    }
}

/// Forward an event to the configured sink, if any. The sink may
/// request cancellation by returning `Err(Cancelled)`; the loader
/// surfaces that as `Err("<cancelled>")` so it can short-circuit at
/// the next safe point.
#[inline]
fn emit(sink: Option<&dyn ProgressSink>, event: ProgressEvent<'_>) -> Result<(), String> {
    if let Some(s) = sink {
        s.emit(event)
            .map_err(|Cancelled| CANCELLED_TOKEN.to_string())?;
    }
    Ok(())
}

/// Rewrite every node's `node_type` key according to `rename_map`, in one
/// sequential pass — O(nodes), not O(renames × nodes). Used after Q-code type
/// resolution promotes placeholder types to their resolved names.
///
/// The `Recording` write-capture wrapper must be recursed through, not treated
/// as impossible: it stopped being test-only when `durable=True` shipped, and
/// an `unreachable!()` there panicked `load_ntriples` on a durable graph.
///
/// The rename deliberately reaches the wrapped backend directly, so it is not
/// captured as a logical mutation: the nodes' final types are what the load's
/// own captured writes already resolve to at flush.
fn apply_type_renames(
    backend: &mut crate::graph::schema::GraphBackend,
    rename_map: &HashMap<u64, u64>,
) {
    use crate::graph::schema::GraphBackend;
    match backend {
        GraphBackend::Disk(dg) => {
            for i in 0..dg.node_slot_len() {
                let slot = dg.node_slot(i);
                if slot.is_alive() {
                    if let Some(&new_type) = rename_map.get(&slot.node_type) {
                        let mut new_slot = slot;
                        new_slot.node_type = new_type;
                        dg.set_node_slot(i, new_slot);
                    }
                }
            }
        }
        GraphBackend::Memory(g) => retype_heap_nodes(
            crate::graph::storage::backend::unique_heap_backend(g),
            rename_map,
        ),
        GraphBackend::Mapped(g) => retype_heap_nodes(
            crate::graph::storage::backend::unique_heap_backend(g),
            rename_map,
        ),
        GraphBackend::Recording(rg) => apply_type_renames(rg.inner_mut(), rename_map),
        // Renaming every node's type would copy-on-write the whole overlay, so
        // flatten first and rename in place — the same total work, without
        // leaving a fully-diverged overlay behind.
        GraphBackend::Forked(_) => {
            backend.flatten_fork();
            apply_type_renames(backend, rename_map);
        }
    }
}

/// The heap half of [`apply_type_renames`], shared by the memory and mapped
/// backends.
fn retype_heap_nodes(g: &mut impl GraphWrite, rename_map: &HashMap<u64, u64>) {
    for i in 0..g.node_bound() {
        let idx = petgraph::graph::NodeIndex::new(i);
        if let Some(node) = g.node_weight_mut(idx) {
            if let Some(&new_key) = rename_map.get(&node.node_type.as_u64()) {
                node.node_type = InternedKey::from_u64(new_key);
            }
        }
    }
}

fn open_ntriples_reader(path: &Path, display_path: &str) -> Result<Box<dyn Read + Send>, String> {
    if display_path.ends_with(".bz2") {
        // The parallel reader falls back to MultiBzDecoder for a single stream.
        return super::parallel_bz2::open(path)
            .map_err(|error| format!("Cannot open {display_path}: {error}"));
    }
    let file = File::open(path).map_err(|error| format!("Cannot open {display_path}: {error}"))?;
    if display_path.ends_with(".gz") {
        Ok(Box::new(GzDecoder::new(BufReader::new(file))))
    } else if display_path.ends_with(".zst") || display_path.ends_with(".zstd") {
        Ok(Box::new(
            zstd::Decoder::new(BufReader::new(file))
                .map_err(|error| format!("zstd decoder error: {error}"))?,
        ))
    } else {
        Ok(Box::new(file))
    }
}

fn finalize_disk_graph(
    graph: &mut DirGraph,
    config: &NTriplesConfig,
    sink: Option<&dyn ProgressSink>,
) -> Result<(), String> {
    let finalising_start = Instant::now();
    if config.verbose && graph.graph.is_disk() {
        eplog!("[Finalising] Building auxiliary indexes + saving metadata");
    }
    if graph.graph.is_disk() {
        emit(
            sink,
            ProgressEvent::Start {
                phase: "finalising",
                label: "Finalising: Auxiliary indexes + metadata",
                total: None,
                unit: "step",
            },
        )?;
    }

    // Warm edge_type_counts_cache from CSR build data (avoids 14 GB rescan on first query)
    if graph.graph.is_disk() {
        if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
            if let Some(raw_counts) = dg.edge_type_counts_raw.take() {
                let string_counts: HashMap<String, usize> = raw_counts
                    .into_iter()
                    .map(|(key_u64, count)| {
                        let key = InternedKey::from_u64(key_u64);
                        let name = graph.interner.resolve(key).to_string();
                        (name, count)
                    })
                    .collect();
                if build_debug() {
                    eplog!(
                        "  Cached {} edge type counts from CSR build",
                        string_counts.len()
                    );
                }
                *graph.edge_type_counts_cache.write().unwrap() = Some(Arc::new(string_counts));
            }
        }
    }

    // Rebuild type_indices from DiskNodeSlots (dropped before Phase 2 to save 1 GB)
    if graph.graph.is_disk() {
        let rebuild_start = Instant::now();
        if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
            for i in 0..dg.node_slot_len() {
                let slot = dg.node_slot(i);
                if slot.is_alive() {
                    let type_key = InternedKey::from_u64(slot.node_type);
                    let type_name = graph.interner.resolve(type_key).to_string();
                    graph
                        .type_indices
                        .entry_or_default(type_name)
                        .push(petgraph::graph::NodeIndex::new(i));
                }
            }
        }
        if build_debug() {
            eplog!(
                "  Rebuilt {} type indices ({})",
                graph.type_indices.len(),
                fmt_dur(rebuild_start.elapsed().as_secs_f64()),
            );
        }
    }

    if graph.graph.is_disk() {
        let data_dir = if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
            dg.active_write_dir().to_path_buf()
        } else {
            std::path::PathBuf::new()
        };
        let mmap_path = data_dir.join("columns.bin");
        let meta_path = data_dir.join("columns_meta.json");
        if mmap_path.exists() && meta_path.exists() {
            let reload_start = Instant::now();
            let reload_result: Result<(), String> = (|| {
                let meta_json = std::fs::read_to_string(&meta_path)
                    .map_err(|e| format!("read columns_meta.json: {}", e))?;
                let columns_meta: Vec<ColumnTypeMeta> = serde_json::from_str(&meta_json)
                    .map_err(|e| format!("parse columns_meta.json: {}", e))?;

                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&mmap_path)
                    .map_err(|e| format!("open columns.bin: {}", e))?;
                // SAFETY: this DiskGraph exclusively owns the active build
                // workspace, and GraphDirectoryLock serializes external
                // writers. No other writer can truncate or replace columns.bin
                // while this mapping is live.
                let mmap = unsafe {
                    memmap2::MmapMut::map_mut(&file)
                        .map_err(|e| format!("mmap columns.bin: {}", e))?
                };
                let mmap_arc = Arc::new(mmap);

                for type_meta in &columns_meta {
                    let mmap_store = type_meta.to_mmap_store(Arc::clone(&mmap_arc));
                    let store = crate::graph::storage::column_store::ColumnStore::from_mmap_store(
                        Arc::new(mmap_store),
                    );
                    graph.install_column_store(&type_meta.type_name, Arc::new(store));
                }
                Ok(())
            })();

            if let Err(e) = reload_result {
                eplog!("  Warning: failed to reload column stores: {}", e);
            }

            if build_debug() {
                eplog!(
                    "  Reloaded {} column stores from mmap ({})",
                    graph.column_store_count(),
                    fmt_dur(reload_start.elapsed().as_secs_f64()),
                );
            }
        }
    }

    // Build id_indices for all types so WHERE id(n) = X is O(1).
    // Uses column stores directly — no node materialization, no arena growth.
    if graph.graph.is_disk() {
        let id_start = Instant::now();
        let type_names: Vec<String> = graph.type_indices.keys().map(|s| s.to_string()).collect();
        for type_name in &type_names {
            graph.build_id_index(type_name);
        }
        if build_debug() {
            eplog!(
                "  Built {} id indices ({})",
                type_names.len(),
                fmt_dur(id_start.elapsed().as_secs_f64()),
            );
        }
    }

    // Save interner + metadata to disk so load() works.
    if graph.graph.is_disk() {
        // Merge the heap-side node-slot overlay into node_slots.bin.
        // `add_node` appends to `appended_node_slots` without extending
        // the mmap'd file; without this flush the directory publishes a
        // metadata node_slots_len the file can't back, and a reload
        // fails ("File too small") past 1024 nodes — or silently loads
        // zeroed dead slots below that.
        if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
            let flush_step = Instant::now();
            dg.flush_node_slots()
                .map_err(|e| format!("Failed to flush node slots: {e}"))?;
            if build_debug() {
                eplog!(
                    "  node_slots.bin flush: {}",
                    fmt_dur(flush_step.elapsed().as_secs_f64())
                );
            }
        }
        build_type_connectivity_cache(graph);

        // Where the build lands. A build staged in a mutation workspace is
        // unreachable from the graph root and is removed with the workspace,
        // so it is published as a new generation; a build that ran in the
        // graph's own directory is published where it lies.
        let generation_root = match &graph.graph {
            crate::graph::schema::GraphBackend::Disk(dg) => dg
                .bulk_build_generation_root()
                .map(std::path::Path::to_path_buf),
            _ => None,
        };
        match generation_root {
            Some(root) => publish_build_as_generation(graph, &root)?,
            None => publish_build_in_place(graph)?,
        }

        let finalising_elapsed = finalising_start.elapsed().as_secs_f64();
        if config.verbose {
            eplog!("[Finalising] Complete in {}", fmt_dur(finalising_elapsed));
        }
        emit(
            sink,
            ProgressEvent::Complete {
                phase: "finalising",
                elapsed_s: finalising_elapsed,
                fields: &[],
            },
        )?;
    }

    Ok(())
}

/// Publish the finished build as the graph at `root_dir`.
///
/// Root metadata goes last: readers must never observe a new completion marker
/// before the sidecars it names exist.
///
/// A flat root build and a `CURRENT` pointer are mutually exclusive ways of
/// naming the same directory's graph, and `resolve_snapshot` prefers the
/// pointer. Disk-mode *creation* publishes an empty generation so a crash
/// before the first `save()` still leaves a loadable path
/// (`storage::mode::publish_initial_generation`), and that pointer would
/// otherwise shadow the build just written to `root_dir` — a reload would
/// silently return the empty graph in the one case documented as needing no
/// `save()`.
///
/// So dropping the pointer is the completing half of the flat publish, a single
/// atomic unlink: until it happens a reader sees the previously published
/// generation, after it the build. It finds nothing to remove when the build
/// was staged into a mutation workspace or a generation, because `CURRENT`
/// only ever lives in the graph root. Published generations keep their bytes;
/// the next `save()` numbers past them.
fn publish_flat_root(root_dir: &Path, metadata_json: &str) -> Result<(), String> {
    std::fs::write(root_dir.join("metadata.json"), metadata_json)
        .map_err(|e| format!("Failed to publish graph metadata: {e}"))?;
    match std::fs::remove_file(root_dir.join("CURRENT")) {
        Ok(()) => {
            // Durability of the unlink, not of its contents: the
            // reader-visible switch is the directory entry.
            if let Ok(handle) = std::fs::File::open(root_dir) {
                let _ = handle.sync_all();
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!(
            "Failed to retire the previous generation pointer: {e}"
        )),
    }
}

/// Build the type-connectivity cache the build's own counts can answer.
///
/// Makes `describe(types=[...])` instant instead of a 10K-node scan. Runs
/// before publication because both routes persist it.
fn build_type_connectivity_cache(graph: &mut DirGraph) {
    {
        let mut triples = Vec::new();
        for (conn_type, info) in graph.connection_type_metadata.iter() {
            let edge_count = graph
                .edge_type_counts_cache
                .read()
                .unwrap()
                .as_ref()
                .and_then(|counts| counts.get(conn_type).copied())
                .unwrap_or(0);
            for src in &info.source_types {
                for tgt in &info.target_types {
                    triples.push(crate::graph::schema::ConnectivityTriple {
                        src: src.clone(),
                        conn: conn_type.clone(),
                        tgt: tgt.clone(),
                        count: edge_count,
                    });
                }
            }
        }
        if !triples.is_empty() {
            *graph.type_connectivity_cache.write().unwrap() = Some(triples);
            if build_debug() {
                eplog!(
                    "  Built type connectivity cache ({} triples)",
                    graph
                        .type_connectivity_cache
                        .read()
                        .unwrap()
                        .as_ref()
                        .map(|t| t.len())
                        .unwrap_or(0),
                );
            }
        }
    }
}

/// Publish a build that ran in the graph's own directory: write the
/// `DirGraph`-level sidecars at the root and hand the directory over.
fn publish_build_in_place(graph: &DirGraph) -> Result<(), String> {
    let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph else {
        return Ok(());
    };
    let data_dir = dg.active_write_dir().to_path_buf();
    // DirGraph-level sidecars (interner, metadata, id/type indexes) belong at
    // the graph ROOT next to disk_graph_meta.json — that's where
    // `load_disk_dir` reads them (only columns.bin has a seg_000 fallback).
    // `data_dir` is the segment dir (`root/seg_000/`); writing the sidecars
    // there made a fresh ntriples disk build reload with an empty interner and
    // no type indexes, so every typed MATCH returned zero rows until an
    // explicit save() rewrote them at the root.
    let root_dir = data_dir
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| data_dir.clone());

    let save_step = Instant::now();
    let interner_map: HashMap<String, String> = graph
        .interner
        .iter()
        .map(|(k, v)| (k.as_u64().to_string(), v.to_string()))
        .collect();
    let json = serde_json::to_string(&interner_map)
        .map_err(|e| format!("Failed to serialize interner: {e}"))?;
    std::fs::write(root_dir.join("interner.json"), json)
        .map_err(|e| format!("Failed to write interner: {e}"))?;
    if build_debug() {
        eplog!(
            "  interner.json ({} entries): {}",
            interner_map.len(),
            fmt_dur(save_step.elapsed().as_secs_f64())
        );
    }

    // Save DirGraph metadata. 0.8.28+ emits the two heavy HashMap
    // fields as separate binary sidecars and strips them from the
    // JSON; the remaining JSON is tiny and parses in ms.
    let save_step = Instant::now();
    crate::graph::io::file::write_node_type_metadata_bin(&root_dir, graph)
        .map_err(|e| format!("Failed to write node-type metadata: {e}"))?;
    crate::graph::io::file::write_connection_type_metadata_bin(&root_dir, graph)
        .map_err(|e| format!("Failed to write connection-type metadata: {e}"))?;
    let mut meta = crate::graph::io::file::build_disk_metadata(graph);
    crate::graph::io::file::strip_heavy_metadata(&mut meta);
    let json = serde_json::to_string_pretty(&meta)
        .map_err(|e| format!("Failed to serialize graph metadata: {e}"))?;
    if build_debug() {
        eplog!("  metadata: {}", fmt_dur(save_step.elapsed().as_secs_f64()));
    }

    // Save id_indices as raw mmap-friendly `.bin` (0.8.28+).
    let save_step = Instant::now();
    if !graph.id_indices.is_empty() {
        crate::graph::storage::disk::id_index::write_id_indices_bin(
            &root_dir,
            &graph.id_indices,
            &graph.interner,
        )
        .map_err(|e| format!("Failed to write id indexes: {e}"))?;
    }
    if build_debug() {
        eplog!(
            "  id_indices.bin ({} types): {}",
            graph.id_indices.len(),
            fmt_dur(save_step.elapsed().as_secs_f64())
        );
    }

    // Save type_indices as raw mmap-friendly `.bin` (0.8.28+).
    let save_step = Instant::now();
    if !graph.type_indices.is_empty() {
        crate::graph::storage::disk::type_index::write_type_indices_bin(
            &root_dir,
            &graph.type_indices,
            &graph.interner,
        )
        .map_err(|e| format!("Failed to write type indexes: {e}"))?;
    }
    if build_debug() {
        eplog!(
            "  type_indices.bin ({} types): {}",
            graph.type_indices.len(),
            fmt_dur(save_step.elapsed().as_secs_f64())
        );
    }

    publish_flat_root(&root_dir, &json)?;
    Ok(())
}

/// Publish a build staged in a mutation workspace as a new immutable
/// generation of `root`.
///
/// The staged build is the whole graph, but it lives in a directory that is
/// removed when the handle drops and that no reader resolves. `save_disk` is
/// the publication every other workspace-staged write already uses: it writes
/// the snapshot into a generation stage, swaps `CURRENT` atomically, and
/// rebases this handle onto the generation it just published. The generations
/// the build superseded keep their bytes — that immutability is the reason
/// the build was staged rather than written over one of them.
fn publish_build_as_generation(graph: &mut DirGraph, root: &Path) -> Result<(), String> {
    let publish_step = Instant::now();
    let root_str = root.to_str().ok_or_else(|| {
        format!(
            "Cannot publish the disk build: graph root is not valid UTF-8: {}",
            root.display()
        )
    })?;
    graph.save_disk(root_str)?;
    if build_debug() {
        eplog!(
            "  published build as a new generation: {}",
            fmt_dur(publish_step.elapsed().as_secs_f64())
        );
    }
    Ok(())
}

fn build_columns(
    graph: &mut DirGraph,
    config: &NTriplesConfig,
    sink: Option<&dyn ProgressSink>,
    mut prop_log: Option<crate::graph::storage::memory::property_log::PropertyLogWriter>,
    type_meta: HashMap<String, TypeBuildMeta>,
    type_rename_map: &HashMap<String, String>,
) -> Result<(), String> {
    // Phase 1b: Convert to columnar storage.
    // Disk and mapped modes pre-allocate columns from metadata, then
    // direct-write from the property log.
    if let Some(log_writer) = prop_log.take() {
        let phase1b_total = log_writer.count();
        let phase1b_label = format!(
            "Phase 1b: Building columnar storage ({} entities, {} types)",
            format_count(phase1b_total),
            type_meta.len(),
        );
        if config.verbose {
            eplog!("[Phase 1b] {}", phase1b_label);
        }
        emit(
            sink,
            ProgressEvent::Start {
                phase: "phase1b",
                label: &phase1b_label,
                total: Some(phase1b_total),
                unit: "ent",
            },
        )?;
        let conv_start = Instant::now();
        let log_path = log_writer
            .finish()
            .map_err(|e| format!("Failed to finish property log: {}", e))?;
        let build_result = super::column_builder::build_columns_direct(
            graph,
            &log_path,
            &type_meta,
            type_rename_map,
            build_debug(),
            sink,
        );
        let _ = std::fs::remove_file(&log_path);
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
        match build_result {
            Ok(()) => {}
            Err(super::column_builder::BuildColumnsError::Cancelled) => {
                return Err(CANCELLED_TOKEN.to_string());
            }
            Err(super::column_builder::BuildColumnsError::Io(error)) => {
                return Err(format!("Failed to build columns: {error}"));
            }
        }
        let phase1b_elapsed = conv_start.elapsed().as_secs_f64();
        if config.verbose {
            eplog!("[Phase 1b] Complete in {}", fmt_dur(phase1b_elapsed));
        }
        emit(
            sink,
            ProgressEvent::Complete {
                phase: "phase1b",
                elapsed_s: phase1b_elapsed,
                fields: &[("entities", PV::U64(phase1b_total))],
            },
        )?;
    } else {
        // Memory mode only; mapped now takes the `Some(log_writer)` arm above,
        // since the per-entity `enable_columnar()` loop was the 7× bottleneck
        // vs disk builds. One bulk O(N) pass at the end of the build — the same
        // pass the first `save()` would otherwise run, moved here so a loaded
        // graph comes out in the shape every other construction path produces
        // instead of changing write regime on its first save.
        debug_assert!(
            !graph.graph.is_mapped(),
            "mapped load_ntriples must populate prop_log and use build_columns_direct"
        );
        graph.enable_columnar();
    }

    // Free everything not needed for Phase 2+3 to maximize page cache.
    // Phase 2 only needs qnum_to_idx + edge_buffer. Phase 3 only needs pending_edges.
    if graph.graph.is_disk() {
        // One map to clear, not two: the backend owns it, so there is no
        // DirGraph copy left behind to resurrect the pages.
        let dropped_stores = graph.column_store_count();
        graph.clear_column_stores();
        drop(type_meta);
        // type_indices: 1 GB — not needed for Phase 2/3. Rebuild from node_slots after.
        let type_indices_count = graph.type_indices.len();
        graph.type_indices.clear();
        if build_debug() {
            eplog!(
                "  Freed {} column stores + {} type indices before Phase 2",
                dropped_stores,
                type_indices_count,
            );
        }
    }

    Ok(())
}

fn build_edges(
    graph: &mut DirGraph,
    config: &NTriplesConfig,
    sink: Option<&dyn ProgressSink>,
    stats: &mut NTriplesStats,
    edge_buffer: EdgeBuffer,
    mut qnum_to_idx: Option<MmapOrVec<u32>>,
) -> Result<(), String> {
    let phase2_total = edge_buffer.len() as u64;
    if config.verbose {
        eplog!("[Phase 2] Creating edges");
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
    emit(
        sink,
        ProgressEvent::Start {
            phase: "phase2",
            label: "Phase 2: Creating edges",
            total: Some(phase2_total),
            unit: "edge",
        },
    )?;

    // Phase 2: Create edges from buffer
    let edge_start = Instant::now();
    if let Some(ref qt) = qnum_to_idx {
        // Fast path: use pre-built qnum_to_idx from Phase 1 (disk mode)
        create_edges_with_qnum_map(graph, &edge_buffer, stats, qt, sink)?;
    } else {
        create_edges_from_buffer(graph, &edge_buffer, stats, sink)?;
    }

    let phase2_elapsed = edge_start.elapsed().as_secs_f64();
    if config.verbose {
        eplog!(
            "[Phase 2] Complete: {} edges created ({} skipped) in {}",
            format_count(stats.edges_created),
            format_count(stats.edges_skipped),
            fmt_dur(phase2_elapsed),
        );
    }
    emit(
        sink,
        ProgressEvent::Complete {
            phase: "phase2",
            elapsed_s: phase2_elapsed,
            fields: &[
                ("edges_created", PV::U64(stats.edges_created)),
                ("edges_skipped", PV::U64(stats.edges_skipped)),
            ],
        },
    )?;

    // Free qnum_to_idx after Phase 2
    if let Some(qt) = qnum_to_idx.take() {
        let qt_path = qt.file_path().map(|p| p.to_path_buf());
        drop(qt);
        if let Some(path) = qt_path {
            let _ = std::fs::remove_file(path);
        }
    }

    // Free edge_buffer before Phase 3.
    let edge_file_path = match &edge_buffer {
        EdgeBuffer::Compact(buf) => buf.file_path().map(|p| p.to_path_buf()),
        _ => None,
    };
    drop(edge_buffer);
    if let Some(path) = edge_file_path {
        let _ = std::fs::remove_file(&path);
    }

    Ok(())
}

fn build_disk_csr(
    graph: &mut DirGraph,
    config: &NTriplesConfig,
    sink: Option<&dyn ProgressSink>,
) -> Result<(), String> {
    // Phase 3: Build CSR from pending edges (disk mode)
    if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
        if config.verbose {
            eplog!("[Phase 3] Building CSR edge index");
        }
        emit(
            sink,
            ProgressEvent::Start {
                phase: "phase3",
                label: "Phase 3: Building CSR edge index",
                total: None,
                unit: "step",
            },
        )?;
        let csr_start = Instant::now();
        dg.build_csr_from_pending()
            .map_err(|e| format!("Failed to build disk CSR: {e}"))?;
        let phase3_elapsed = csr_start.elapsed().as_secs_f64();
        if config.verbose {
            eplog!("[Phase 3] Complete in {}", fmt_dur(phase3_elapsed));
        }
        emit(
            sink,
            ProgressEvent::Complete {
                phase: "phase3",
                elapsed_s: phase3_elapsed,
                fields: &[],
            },
        )?;
    }

    Ok(())
}

fn resolve_type_labels(
    graph: &mut DirGraph,
    config: &NTriplesConfig,
    mut label_writer: Option<super::label_spill::LabelSpillWriter>,
    type_meta: &mut HashMap<String, TypeBuildMeta>,
) -> Result<HashMap<String, String>, String> {
    // Post-Phase-1: resolve the raw Q-codes Phase 1 left as type names
    // (e.g. "Q5") against the label journal — one read, and only for the
    // Q-codes that actually became type names. See `initialize_spills` for
    // why the journal replaced an in-memory cache.
    let mut type_rename_map: HashMap<String, String> = HashMap::new();

    // Flush + close the label journal before reading it back.
    let label_journal_path = if let Some(writer) = label_writer.take() {
        let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
        });
        let path = spill_dir.join("labels.bin");
        let journal_size = writer
            .finish()
            .map_err(|e| format!("Failed to finish label journal: {e}"))?;
        if build_debug() {
            eplog!(
                "  Label journal: {} bytes on disk",
                format_count(journal_size)
            );
        }
        Some(path)
    } else {
        None
    };

    if config.auto_type {
        let wanted: std::collections::HashSet<u32> = graph
            .type_indices
            .keys()
            .filter_map(parse_qcode_number)
            .collect();

        // One forward scan; skips unwanted records without allocating.
        let label_lookup: HashMap<u32, String> = if let Some(ref path) = label_journal_path {
            super::label_spill::read_labels_for(path, &wanted).unwrap_or_else(|e| {
                eplog!("  WARN: failed to read label journal: {}", e);
                HashMap::new()
            })
        } else {
            HashMap::new()
        };

        let mut renames: Vec<(String, String)> = Vec::new();
        for type_name in graph.type_indices.keys() {
            if let Some(qnum) = parse_qcode_number(type_name) {
                if let Some(label) = label_lookup.get(&qnum) {
                    if label != type_name {
                        renames.push((type_name.to_string(), label.clone()));
                    }
                }
            }
        }
        if !renames.is_empty() {
            let rename_start = std::time::Instant::now();
            if build_debug() {
                eplog!(
                    "  Resolving {} Q-code type names to labels...",
                    renames.len()
                );
            }
            graph
                .interner
                .validate_names(
                    renames
                        .iter()
                        .flat_map(|(old, new)| [old.as_str(), new.as_str()]),
                )
                .map_err(|e| e.to_string())?;
            let old_key_to_new_key: Vec<(InternedKey, InternedKey)> = renames
                .iter()
                .map(|(old, new)| {
                    let old_key = graph.interner.get_or_intern(old);
                    let new_key = graph.interner.get_or_intern(new);
                    (old_key, new_key)
                })
                .collect();
            for (old_name, new_name) in &renames {
                if let Some(indices) = graph.type_indices.remove(old_name) {
                    graph
                        .type_indices
                        .entry_or_default(new_name.clone())
                        .extend(indices);
                }
                if let Some(old_meta) = graph.node_type_metadata_mut().remove(old_name) {
                    let entry = graph
                        .node_type_metadata_mut()
                        .entry(new_name.clone())
                        .or_default();
                    for (k, v) in old_meta {
                        entry.entry(k).or_insert(v);
                    }
                }
                if let Some(old_schema) = graph.type_schemas_mut().remove(old_name) {
                    if let Some(existing) = graph.type_schemas.get(new_name) {
                        let merged = existing.merge(&old_schema);
                        graph
                            .type_schemas_mut()
                            .insert(new_name.clone(), Arc::new(merged));
                    } else {
                        graph
                            .type_schemas_mut()
                            .insert(new_name.clone(), old_schema);
                    }
                }
                if let Some(old_build) = type_meta.remove(old_name) {
                    let entry = type_meta
                        .entry(new_name.clone())
                        .or_insert_with(TypeBuildMeta::new);
                    entry.merge_from(&old_build);
                }
            }
            // Build type_rename_map for Phase 1b (property log entries use old names)
            for (old_name, new_name) in &renames {
                type_rename_map.insert(old_name.clone(), new_name.clone());
            }

            let rename_map: HashMap<u64, u64> = old_key_to_new_key
                .iter()
                .map(|(old, new)| (old.as_u64(), new.as_u64()))
                .collect();
            apply_type_renames(&mut graph.graph, &rename_map);
            if build_debug() {
                eplog!(
                    "  Resolved {} Q-code types ({})",
                    renames.len(),
                    fmt_dur(rename_start.elapsed().as_secs_f64())
                );
            }
        }
    }

    Ok(type_rename_map)
}

struct LoadSpills {
    prop_log: Option<crate::graph::storage::memory::property_log::PropertyLogWriter>,
    edge_buffer: EdgeBuffer,
    label_writer: Option<super::label_spill::LabelSpillWriter>,
    type_meta: HashMap<String, TypeBuildMeta>,
    qnum_to_idx: Option<MmapOrVec<u32>>,
}

fn initialize_spills(graph: &mut DirGraph, config: &NTriplesConfig) -> Result<LoadSpills, String> {
    let use_streaming_build = graph.graph.is_disk() || graph.graph.is_mapped();
    let use_compact = use_streaming_build;

    // Property log for streaming builds: serialize properties during Phase 1,
    // replay in Phase 1b. Used by both disk and mapped modes.
    let prop_log: Option<crate::graph::storage::memory::property_log::PropertyLogWriter> =
        if use_streaming_build {
            let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
                std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
            });
            // Clean up stale spill dirs from previous killed builds.
            //
            // Safe for concurrent runs: only delete directories whose
            // contents haven't been modified in the last hour. A running
            // build writes to its `properties.log.zst` continuously, so
            // active spill dirs always look "fresh". Without the age gate a
            // parallel `load_ntriples` would wipe the *other* run's log.
            const STALE_AFTER_SECS: u64 = 3600;
            if let Some(parent) = spill_dir.parent() {
                if let Ok(entries) = std::fs::read_dir(parent) {
                    let now = std::time::SystemTime::now();
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let name = name.to_string_lossy();
                        if !name.starts_with("kglite_build_") {
                            continue;
                        }
                        if entry.path() == spill_dir {
                            continue;
                        }
                        let is_stale = entry
                            .metadata()
                            .and_then(|m| m.modified())
                            .ok()
                            .and_then(|m| now.duration_since(m).ok())
                            .map(|d| d.as_secs() > STALE_AFTER_SECS)
                            .unwrap_or(false);
                        if is_stale {
                            let _ = std::fs::remove_dir_all(entry.path());
                        }
                    }
                }
            }
            // Clean up stale pending_edges from previous killed builds.
            //
            // Never remove the file the live graph is currently mapping: it is
            // not stale by definition, and unlinking it silently detaches the
            // active buffer from its path on POSIX while failing outright on
            // Windows, which refuses to delete a file with a mapped view.
            if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
                let stale = dg.active_write_dir().join("_pending_edges.bin");
                let mapped_live = dg.pending_edges.get_mut().file_path() == Some(stale.as_path());
                if stale.exists() && !mapped_live {
                    let _ = std::fs::remove_file(&stale);
                }
            }
            let log_path = spill_dir.join("properties.log.zst");
            if build_debug() {
                eplog!("  Property log: {}", log_path.display());
            }
            Some(
                crate::graph::storage::memory::property_log::PropertyLogWriter::new(&log_path, 1)
                    .map_err(|e| format!("Failed to create property log: {}", e))?,
            )
        } else {
            None
        };
    let edge_buffer = if use_compact {
        if use_streaming_build {
            // File-backed edge buffer: avoids holding ~14 GB in RAM during Phase 1b
            let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
                std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
            });
            std::fs::create_dir_all(&spill_dir)
                .map_err(|e| format!("Failed to create spill dir: {}", e))?;
            let edge_path = spill_dir.join("edges.bin");
            EdgeBuffer::Compact(
                MmapOrVec::mapped(&edge_path, 1 << 20)
                    .map_err(|e| format!("Failed to create edge buffer file: {}", e))?,
            )
        } else {
            EdgeBuffer::Compact(MmapOrVec::new())
        }
    } else {
        EdgeBuffer::Strings(Vec::new())
    };

    // Label journal for auto-typing. As a `HashMap<u32, String>` this grew to
    // ~10 GB of heap at Wikidata scale (124M entities), pushing 16 GB machines
    // into swap and collapsing the Phase 1 rate from 1.8M to 450K triples/s.
    // Now a buffered sequential write to `{spill_dir}/labels.bin` — zero heap
    // growth during Phase 1; the post-Phase-1 rename pass reads it once and
    // keeps only the ~88K labels that appear as type names.
    let label_writer: Option<super::label_spill::LabelSpillWriter> = if config.auto_type {
        let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
        });
        std::fs::create_dir_all(&spill_dir)
            .map_err(|e| format!("Failed to create label spill directory: {e}"))?;
        Some(
            super::label_spill::LabelSpillWriter::new(&spill_dir.join("labels.bin"))
                .map_err(|e| format!("Failed to create label journal: {}", e))?,
        )
    } else {
        None
    };

    // Per-type build metadata for Phase 1b pre-allocation.
    let type_meta: HashMap<String, TypeBuildMeta> = HashMap::new();

    // For disk mode: build qnum_to_idx during Phase 1 (instead of from id_indices in Phase 2).
    // This lets us skip id_indices entirely, saving ~11 GB at full Wikidata scale.
    // Pre-allocate for 150M Q-numbers (~600 MB mmap, lazily paged).
    let qnum_to_idx: Option<MmapOrVec<u32>> = if graph.graph.is_disk() {
        let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
        });
        std::fs::create_dir_all(&spill_dir)
            .map_err(|e| format!("Failed to create id spill directory: {e}"))?;
        Some(
            MmapOrVec::mapped_prefilled(&spill_dir.join("qnum_to_idx.bin"), 150_000_000)
                .unwrap_or_else(|_| MmapOrVec::from_vec(vec![0u32; 150_000_000])),
        )
    } else {
        None
    };

    Ok(LoadSpills {
        prop_log,
        edge_buffer,
        label_writer,
        type_meta,
        qnum_to_idx,
    })
}

struct Phase1Ingest<'a> {
    graph: &'a mut DirGraph,
    path: &'a str,
    path_obj: &'a Path,
    config: &'a NTriplesConfig,
    stats: &'a mut NTriplesStats,
    edge_buffer: &'a mut EdgeBuffer,
    prop_log: &'a mut Option<crate::graph::storage::memory::property_log::PropertyLogWriter>,
    label_writer: &'a mut Option<super::label_spill::LabelSpillWriter>,
    type_meta: &'a mut HashMap<String, TypeBuildMeta>,
    qnum_to_idx: &'a mut Option<MmapOrVec<u32>>,
    rx: std::sync::mpsc::Receiver<ReaderBatch>,
    reader_handle: std::thread::JoinHandle<Result<(), String>>,
    started: Instant,
}

fn ingest_phase1(context: Phase1Ingest<'_>) -> Result<(), String> {
    let Phase1Ingest {
        graph,
        path,
        path_obj,
        config,
        stats,
        edge_buffer,
        prop_log,
        label_writer,
        type_meta,
        qnum_to_idx,
        rx,
        reader_handle,
        started: start,
    } = context;
    let mapped = false;
    let mut current: Option<EntityAccumulator> = None;
    let mut entity_limit_reached = false;
    // Reusable scratch buffer for `flush_entity`'s property-log
    // `Vec<(InternedKey, Value)>`. Hoisted out of `flush_entity` so the
    // alloc + grow cost is paid once, not per entity (~2% of loader CPU
    // under samply).
    let mut scratch_props: Vec<(InternedKey, Value)> = Vec::with_capacity(64);
    // Cheap bucket counter (decrement in the hot loop); every 5M triples we
    // check wall time and only log a `[Phase 1]` line if PROGRESS_INTERVAL_SECS
    // have elapsed since the last one, so the terminal isn't spammed.
    let mut progress_countdown: u64 = 5_000_000;
    let mut last_progress_log = Instant::now();
    const PROGRESS_BUCKET: u64 = 5_000_000;
    const PROGRESS_INTERVAL_SECS: f64 = 60.0;
    let include_labels = true;

    if config.verbose {
        eplog!("[Phase 1] Streaming and parsing N-triples ({})", path);
    }
    let sink = config.progress.as_deref();
    // The bar tracks triples — the loop's natural unit. When the
    // caller has set `max_triples` we use that as the bar's total so
    // tqdm shows ETA; otherwise the bar runs unbounded.
    let phase1_label = format!(
        "Phase 1: Streaming N-triples ({})",
        path_obj
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
    );
    emit(
        sink,
        ProgressEvent::Start {
            phase: "phase1",
            label: &phase1_label,
            total: config.max_triples,
            unit: "tri",
        },
    )?;

    let mut reader_error = None;
    'outer: while let Ok(message) = rx.recv() {
        let batch = match message {
            Ok(batch) => batch,
            Err(error) => {
                reader_error = Some(error);
                break;
            }
        };
        let n_lines = batch.offsets.len();
        for i in 0..n_lines {
            // Slice into the batch's contiguous buffer — pointer math,
            // no per-line heap dereference.
            let line = match validated_line(batch.line(i)) {
                Ok(line) => line,
                Err(error) => {
                    reader_error = Some(format!(
                        "N-Triples input contains invalid UTF-8 in an accepted entity line: {error}"
                    ));
                    break 'outer;
                }
            };
            if entity_limit_reached && current.is_none() {
                break 'outer;
            }
            // `max_triples` short-circuit: hard-stop after N triples
            // scanned, regardless of entity progress. Unlike the
            // `max_entities` check above we don't gate on
            // `current.is_none()` — `current` is essentially always
            // `Some` during steady-state Wikidata reading (subject-sorted
            // dump = always mid-entity), so the gate would never trip.
            // We trade losing one in-progress entity for a deterministic
            // triple cap; that's the right call for a perf benchmark.
            if let Some(cap) = config.max_triples {
                if stats.triples_scanned >= cap {
                    break 'outer;
                }
            }

            stats.triples_scanned += 1;

            progress_countdown -= 1;
            if progress_countdown == 0 {
                progress_countdown = PROGRESS_BUCKET;
                let buf_len = edge_buffer.len() as u64;
                // Callback fires on every bucket so tqdm stays live;
                // the sink may also signal cancellation here (Ctrl+C).
                emit(
                    sink,
                    ProgressEvent::Update {
                        phase: "phase1",
                        current: stats.triples_scanned,
                        fields: &[
                            ("entities", PV::U64(stats.entities_created)),
                            ("edges_buffered", PV::U64(buf_len)),
                        ],
                    },
                )?;
                if config.verbose
                    && last_progress_log.elapsed().as_secs_f64() >= PROGRESS_INTERVAL_SECS
                {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = stats.triples_scanned as f64 / elapsed;
                    eplog!(
                        "[Phase 1] {} triples, {} entities, {} edges buffered — {:.0}k triples/s",
                        format_count(stats.triples_scanned),
                        format_count(stats.entities_created),
                        format_count(buf_len),
                        rate / 1000.0,
                    );
                    last_progress_log = Instant::now();
                }
            }

            // No prefix check here: the reader thread already dropped
            // non-entity lines before they reached the channel.
            let (subject, predicate, object) = match parse_line(line) {
                Some(parsed) => parsed,
                None => continue,
            };

            let subj_id = match subject {
                Subject::Entity(id) => id,
                Subject::Other => continue,
            };

            if current.as_ref().is_some_and(|c| c.id != subj_id) {
                if let Some(acc) = current.take() {
                    flush_entity(
                        graph,
                        acc,
                        config,
                        edge_buffer,
                        stats,
                        mapped,
                        prop_log,
                        label_writer,
                        type_meta,
                        qnum_to_idx,
                        &mut scratch_props,
                    )?;
                }
                if entity_limit_reached {
                    break;
                }
            }

            if current.is_none() {
                if let Some(max) = config.max_entities {
                    if stats.entities_created >= max as u64 {
                        entity_limit_reached = true;
                        continue;
                    }
                }
                current = Some(EntityAccumulator::new(subj_id.to_string()));
            }

            let acc = current.as_mut().unwrap();

            match predicate {
                Predicate::Label => {
                    if include_labels {
                        if let Some(text) = extract_lang_text(&object, &config.languages) {
                            acc.label = Some(text);
                        }
                    }
                }
                Predicate::Description => {
                    if let Some(text) = extract_lang_text(&object, &config.languages) {
                        acc.description = Some(text);
                    }
                }
                Predicate::AltLabel => {}
                Predicate::Type => {}
                Predicate::WikidataDirect(pcode) => {
                    if let Some(ref allowed) = config.predicates {
                        if !allowed.contains(pcode) {
                            continue;
                        }
                    }

                    let pred_label = config
                        .predicate_labels
                        .get(pcode)
                        .cloned()
                        .unwrap_or_else(|| pcode.to_string());

                    match &object {
                        Object::Entity(target_qcode) => {
                            if pcode == "P31" && acc.type_qcode.is_none() {
                                acc.type_qcode = Some(target_qcode.to_string());
                            }
                            acc.outgoing_edges
                                .push((pred_label, target_qcode.to_string()));
                        }
                        Object::Literal(text) => {
                            acc.properties
                                .insert(pred_label, Value::String(text.clone()));
                        }
                        Object::LangLiteral(text, lang) => {
                            if language_matches(lang, &config.languages) {
                                acc.properties
                                    .insert(pred_label, Value::String(text.clone()));
                            }
                        }
                        Object::TypedLiteral(text, type_uri) => {
                            acc.properties
                                .insert(pred_label, typed_literal_to_value(text, type_uri));
                        }
                        Object::Other => {}
                    }
                }
                Predicate::Other => {}
            }
        } // end for line in batch
    } // end for batch in rx
      // CRITICAL: drop `rx` before joining so the reader thread's
      // `tx.send` returns Err (channel closed) on its next batch and
      // exits. Without this, an early `break 'outer` (e.g. from the
      // `max_triples` cap) leaves the reader thread blocked on a full
      // bounded channel that nobody is draining → `join()` deadlocks.
    drop(rx);
    let join_result = join_reader(reader_handle);
    if let Some(error) = reader_error {
        return Err(error);
    }
    join_result?;

    if let Some(acc) = current.take() {
        flush_entity(
            graph,
            acc,
            config,
            edge_buffer,
            stats,
            mapped,
            prop_log,
            label_writer,
            type_meta,
            qnum_to_idx,
            &mut scratch_props,
        )?;
    }

    Ok(())
}

/// Open the input, staging the disk build in a mutation workspace unless it
/// can write where it lies.
///
/// Fresh disk builders intentionally finalise in place so their directory is
/// reloadable without save(). A detached copy's base snapshot belongs to its
/// source, and a handle sitting on a published generation must not rewrite
/// that immutable snapshot — both stage instead
/// (`DiskGraph::prepare_bulk_load_workspace`).
fn open_reader_for_load(
    graph: &mut DirGraph,
    path: &Path,
    display_path: &str,
) -> Result<Box<dyn Read + Send>, String> {
    let reader = open_ntriples_reader(path, display_path)?;
    if let crate::graph::schema::GraphBackend::Disk(ref mut disk) = graph.graph {
        disk.prepare_bulk_load_workspace()
            .map_err(|error| format!("Failed to prepare the disk build workspace: {error}"))?;
    }
    Ok(reader)
}

pub fn load_ntriples(
    graph: &mut DirGraph,
    path: &str,
    config: &NTriplesConfig,
) -> Result<NTriplesStats, String> {
    let start = Instant::now();
    let path_obj = Path::new(path);

    let reader = open_reader_for_load(graph, path_obj, path)?;
    // Reader thread: decompresses + reads lines via channel (hides I/O latency).
    //
    // Each batch packs lines into a single `LineBuffer` (contiguous bytes
    // + offset table) instead of `Vec<String>` with 200k separately
    // heap-allocated `String`s. Cache-friendly iteration on the loader
    // side, no per-line allocation in the reader, and the channel
    // transports one buffer at a time. Channel capacity 32 × ~16 MB =
    // ~512 MB ceiling on in-flight bytes — well within memory budget,
    // smooths out short loader stalls.
    let (rx, reader_handle) = spawn_reader(reader);

    let mut stats = NTriplesStats {
        triples_scanned: 0,
        entities_created: 0,
        edges_created: 0,
        edges_skipped: 0,
        seconds: 0.0,
    };

    // Phase 1: Parse and ingest. Disk and mapped modes stream properties into
    // a compressed log (~100 ns/entity) plus a packed `EdgeBuffer::Compact`,
    // which Phase 1b replays into ColumnStores in bulk; memory mode inserts
    // `PropertyStorage::Map` rows and converts at the end of Phase 1b.
    let LoadSpills {
        mut prop_log,
        mut edge_buffer,
        mut label_writer,
        mut type_meta,
        mut qnum_to_idx,
    } = initialize_spills(graph, config)?;

    ingest_phase1(Phase1Ingest {
        graph,
        path,
        path_obj,
        config,
        stats: &mut stats,
        edge_buffer: &mut edge_buffer,
        prop_log: &mut prop_log,
        label_writer: &mut label_writer,
        type_meta: &mut type_meta,
        qnum_to_idx: &mut qnum_to_idx,
        rx,
        reader_handle,
        started: start,
    })?;
    let sink = config.progress.as_deref();

    let type_rename_map = resolve_type_labels(graph, config, label_writer, &mut type_meta)?;

    let phase1_elapsed = start.elapsed().as_secs_f64();
    let phase1_buf_len = edge_buffer.len() as u64;
    let phase1_num_types = type_meta.len() as u64;
    let phase1_total_cols: u64 = type_meta.values().map(|m| m.columns.len() as u64).sum();
    if config.verbose {
        eplog!(
            "[Phase 1] Complete: {} entities, {} types ({} columns), {} edges buffered in {}",
            format_count(stats.entities_created),
            format_count(phase1_num_types),
            format_count(phase1_total_cols),
            format_count(phase1_buf_len),
            fmt_dur(phase1_elapsed),
        );
    }
    emit(
        sink,
        ProgressEvent::Complete {
            phase: "phase1",
            elapsed_s: phase1_elapsed,
            fields: &[
                ("entities", PV::U64(stats.entities_created)),
                ("edges_buffered", PV::U64(phase1_buf_len)),
                ("types", PV::U64(phase1_num_types)),
                ("columns", PV::U64(phase1_total_cols)),
                ("triples_scanned", PV::U64(stats.triples_scanned)),
            ],
        },
    )?;

    build_columns(graph, config, sink, prop_log, type_meta, &type_rename_map)?;

    build_edges(graph, config, sink, &mut stats, edge_buffer, qnum_to_idx)?;

    build_disk_csr(graph, config, sink)?;

    finalize_disk_graph(graph, config, sink)?;

    stats.seconds = start.elapsed().as_secs_f64();
    if config.verbose {
        eplog!("[Build] Total elapsed: {}", fmt_dur(stats.seconds));
    }
    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
fn flush_entity(
    graph: &mut DirGraph,
    acc: EntityAccumulator,
    config: &NTriplesConfig,
    edge_buffer: &mut EdgeBuffer,
    stats: &mut NTriplesStats,
    mapped: bool,
    prop_log: &mut Option<crate::graph::storage::memory::property_log::PropertyLogWriter>,
    label_writer: &mut Option<super::label_spill::LabelSpillWriter>,
    type_meta: &mut HashMap<String, TypeBuildMeta>,
    qnum_to_idx: &mut Option<MmapOrVec<u32>>,
    // Reusable property-log buffer, hoisted to the caller (see `ingest_phase1`).
    scratch_props: &mut Vec<(InternedKey, Value)>,
) -> Result<(), String> {
    // Disk/mapped mode requires a `u32` Q-number. A `Value::String` id
    // would flip `id_is_string=true` on the type's columnar metadata,
    // and `flush_type_entries` would then leave the *other* (UniqueId)
    // rows' offsets at zero — which makes `MmapColumnStore::read_str`
    // panic on reload (`offsets[row-1] > offsets[row]`). Wikidata's
    // truthy dump has a small number of non-parseable Q-codes (e.g.,
    // entries with trailing suffixes that slip past the entity-prefix
    // filter). They're not legitimate data — drop them.
    let use_compact_ids = mapped || graph.graph.is_disk();
    if use_compact_ids && parse_qcode_number(&acc.id).is_none() {
        return Ok(());
    }

    let title = acc.label.unwrap_or_else(|| acc.id.clone());

    // Append to the label journal for post-Phase-1 type resolution: a
    // sequential write, so zero heap pressure during the streaming phase.
    if let Some(ref mut w) = label_writer {
        if let Some(qnum) = parse_qcode_number(&acc.id) {
            w.append(qnum, &title)
                .map_err(|e| format!("Failed to append label journal: {e}"))?;
        }
    }

    // With auto_type on, Phase 1 keeps the raw P31 Q-code as the type name —
    // the post-Phase-1 rename pass resolves it against the label journal,
    // avoiding an in-loop lookup on a 124M-entry map.
    let node_type = if let Some(ref tq) = acc.type_qcode {
        if let Some(mapped_name) = config.node_types.get(tq) {
            mapped_name.clone()
        } else if config.auto_type {
            tq.clone()
        } else {
            "Entity".to_string()
        }
    } else {
        "Entity".to_string()
    };

    let mut names = vec![node_type.as_str(), "nid", "description", "P31"];
    names.extend(acc.properties.keys().map(String::as_str));
    names.extend(
        acc.outgoing_edges
            .iter()
            .map(|(predicate, _)| predicate.as_str()),
    );
    graph
        .interner
        .validate_names(names)
        .map_err(|e| e.to_string())?;

    let mut properties = acc.properties;
    properties.insert("nid".to_string(), Value::String(acc.id.clone()));
    if let Some(desc) = acc.description {
        properties.insert("description".to_string(), Value::String(desc));
    }
    // Store P31 Q-code as a property (preserves type info when defaulting to "Entity")
    if let Some(ref tq) = acc.type_qcode {
        properties.insert("P31".to_string(), Value::String(tq.clone()));
    }

    // ID representation is mode-INDEPENDENT (cross-mode parity, 0.11.0): a
    // parseable Q-code is stored as a compact `UniqueId` in every mode, so
    // `n.id` is the same integer and `{id: N}` matches identically in memory,
    // mapped, and disk. The human-readable string form (`"Q42"`) lives in the
    // `nid` property (stored above) — query `{nid: 'Q42'}`, not `{id: 'Q42'}`.
    // Non-parseable ids fall back to `String` (memory keeps them via the
    // General id-index; disk/mapped already dropped them at the guard above).
    let id_value = parse_qcode_number(&acc.id)
        .map(Value::UniqueId)
        .unwrap_or_else(|| Value::String(acc.id.clone()));
    let title_value = Value::String(title);

    let mut node_data = NodeData::new(
        id_value.clone(),
        title_value,
        node_type.clone(),
        properties,
        &mut graph.interner,
    );

    // Mapped mode: push properties into ColumnStore.
    if mapped {
        let interned_props = node_data
            .properties
            .drain_to_interned_pairs(&graph.interner);
        let keys: Vec<_> = interned_props.iter().map(|(k, _)| *k).collect();
        graph.ensure_type_schema_keys(&node_type, &keys);
        let store = graph.ensure_column_store_for_push(&node_type);
        store.push_id(&node_data.id);
        store.push_title(&node_data.title);
        store.push_row(&interned_props);
        node_data.properties = PropertyStorage::Map(HashMap::new());
        node_data.id = Value::Null;
        node_data.title = Value::Null;
    }

    // Properties must reach the log BEFORE they are cleared for `add_node`
    // (which discards them anyway on disk).
    let saved_id = node_data.id.clone();
    let saved_title = node_data.title.clone();
    if prop_log.is_some() {
        scratch_props.clear();
        scratch_props.extend(
            node_data
                .properties
                .drain_to_interned_pairs(&graph.interner),
        );
        node_data.properties = PropertyStorage::Map(HashMap::new());
        node_data.id = Value::Null;
        node_data.title = Value::Null;
    }

    let node_idx = GraphWrite::add_node(&mut graph.graph, node_data);

    // Write to property log after add_node so we have the node_idx
    if let Some(ref mut log) = prop_log {
        let node_type_key = graph.interner.get_or_intern(&node_type);
        log.write_entity(
            node_type_key,
            node_idx,
            &saved_id,
            &saved_title,
            scratch_props,
        )
        .map_err(|e| format!("Property log write failed: {e}"))?;

        type_meta
            .entry(node_type.clone())
            .or_insert_with(TypeBuildMeta::new)
            .record_entity(&saved_id, &saved_title, scratch_props);
    }

    graph
        .type_indices
        .entry_or_default(node_type.clone())
        .push(node_idx);

    // Disk mode writes qnum_to_idx directly and keeps no id_indices at all
    // (see `initialize_spills`); every other mode fills id_indices.
    if let Some(ref mut qt) = qnum_to_idx {
        if let Some(qnum) = parse_qcode_number(&acc.id) {
            if (qnum as usize) < qt.len() {
                qt.set(qnum as usize, node_idx.index() as u32 + 1); // +1: 0 = not present
            }
        }
    } else {
        graph
            .id_indices
            .entry_or_default(node_type)
            .insert(id_value, node_idx);
    }

    stats.entities_created += 1;

    if mapped && stats.entities_created.is_multiple_of(100_000) {
        graph.maybe_spill_columns();
    }

    match edge_buffer {
        EdgeBuffer::Compact(buf) => {
            if let Some(src_num) = parse_qcode_number(&acc.id) {
                for (pred_label, target_qcode) in acc.outgoing_edges {
                    if let Some(tgt_num) = parse_qcode_number(&target_qcode) {
                        let pred_key = graph.interner.get_or_intern(&pred_label);
                        buf.try_push(CompactNTripleEdge {
                            source_qnum: src_num,
                            target_qnum: tgt_num,
                            predicate: pred_key,
                        })
                        .map_err(|error| format!("append compact N-Triples edge: {error}"))?;
                    }
                }
            }
        }
        EdgeBuffer::Strings(buf) => {
            for (pred_label, target_qcode) in acc.outgoing_edges {
                buf.push((acc.id.clone(), target_qcode, pred_label));
            }
        }
    }
    Ok(())
}

pub(super) fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

/// Compact duration formatter for phase headers — `1h32m04s` / `47m18s` /
/// `12.3s`. Uses the same shape as `examples/wikidata_disk.py::_fmt_dur` so
/// users see identical wording from Rust and Python.
pub(super) fn fmt_dur(secs: f64) -> String {
    if secs < 60.0 {
        return format!("{:.1}s", secs);
    }
    let total = secs as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{}h{:02}m{:02}s", h, m, s)
    } else {
        format!("{}m{:02}s", m, s)
    }
}

/// Dev-grade verbosity. `verbose=True` keeps the high-level `[Phase N]`
/// gate messages clean for end users; setting `KGLITE_BUILD_DEBUG=1` adds
/// the per-sub-step timings (interner save, CSR step 1/4, peer-count
/// histogram timings, …) used while diagnosing build performance.
fn build_debug() -> bool {
    std::env::var("KGLITE_BUILD_DEBUG").is_ok()
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
