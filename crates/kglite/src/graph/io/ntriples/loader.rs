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
    EdgeBuffer, EntityAccumulator, Object, Predicate, Subject,
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

pub fn load_ntriples(
    graph: &mut DirGraph,
    path: &str,
    config: &NTriplesConfig,
) -> Result<NTriplesStats, String> {
    let start = Instant::now();
    let path_obj = Path::new(path);

    // Open decompression reader
    let reader: Box<dyn Read + Send> = if path.ends_with(".bz2") {
        // Multistream bz2 (pbzip2 / Wikidata dumps) decompresses
        // in parallel; single-stream files fall through to
        // `MultiBzDecoder` inside `parallel_bz2::open`.
        super::parallel_bz2::open(path_obj).map_err(|e| format!("Cannot open {}: {}", path, e))?
    } else {
        let file = File::open(path_obj).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        if path.ends_with(".gz") {
            Box::new(GzDecoder::new(BufReader::new(file)))
        } else if path.ends_with(".zst") || path.ends_with(".zstd") {
            Box::new(
                zstd::Decoder::new(BufReader::new(file))
                    .map_err(|e| format!("zstd decoder error: {}", e))?,
            )
        } else {
            Box::new(file)
        }
    };
    // Reader thread: decompresses + reads lines via channel (hides I/O latency).
    //
    // Each batch packs lines into a single `LineBuffer` (contiguous bytes
    // + offset table) instead of `Vec<String>` with 200k separately
    // heap-allocated `String`s. Cache-friendly iteration on the loader
    // side, no per-line allocation in the reader, and the channel
    // transports one buffer at a time. Channel capacity 32 × ~16 MB =
    // ~512 MB ceiling on in-flight bytes — well within memory budget,
    // smooths out short loader stalls.
    const BATCH_SIZE: usize = 200_000;
    const TARGET_BATCH_BYTES: usize = 16 * 1024 * 1024;
    let (tx, rx) = std::sync::mpsc::sync_channel::<super::parser::LineBuffer>(32);
    let reader_handle = std::thread::spawn(move || {
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, reader);
        let mut raw = Vec::with_capacity(512);
        let mut batch = super::parser::LineBuffer::with_capacity(BATCH_SIZE, TARGET_BATCH_BYTES);
        let prefix: &[u8] = b"<http://www.wikidata.org/entity/Q";

        loop {
            raw.clear();
            if reader.read_until(b'\n', &mut raw).unwrap_or(0) == 0 {
                if !batch.is_empty() {
                    let _ = tx.send(batch);
                }
                break;
            }

            // Fast reject at byte level — skip non-entity lines.
            if !raw.starts_with(prefix) {
                continue;
            }

            // Append raw bytes to the current batch; no `String` allocation.
            batch.push_line(&raw);
            if batch.offsets.len() >= BATCH_SIZE || batch.data.len() >= TARGET_BATCH_BYTES {
                let next = super::parser::LineBuffer::with_capacity(BATCH_SIZE, TARGET_BATCH_BYTES);
                let full = std::mem::replace(&mut batch, next);
                if tx.send(full).is_err() {
                    break;
                }
            }
        }
    });

    let mut stats = NTriplesStats {
        triples_scanned: 0,
        entities_created: 0,
        edges_created: 0,
        edges_skipped: 0,
        seconds: 0.0,
    };

    // Phase 1: Parse and ingest.
    // For Disk mode: serialize properties to a compressed log file (fast, ~100 ns/entity).
    // Phase 1b replays the log to build ColumnStores in bulk.
    // For other modes: use fast non-mapped insertion (HashMap properties, then Phase 1b).
    // final_mode was graph.storage_mode; now read from graph.graph variant
    let mapped = false;
    let mut current: Option<EntityAccumulator> = None;
    // Mapped mode reuses the disk path's Phase 1 streaming: a spill-backed
    // property log + packed `EdgeBuffer::Compact` in place of the slow
    // per-entity `PropertyStorage::Map` + `EdgeBuffer::Strings` path.
    // Phase 1b then routes through `build_columns_direct`, writing a single
    // `columns.bin` instead of the per-entity `enable_columnar()` loop.
    let use_streaming_build = graph.graph.is_disk() || graph.graph.is_mapped();
    let use_compact = use_streaming_build;

    // Property log for streaming builds: serialize properties during Phase 1,
    // replay in Phase 1b. Used by both disk and mapped modes.
    let mut prop_log: Option<crate::graph::storage::memory::property_log::PropertyLogWriter> =
        if use_streaming_build {
            let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
                std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
            });
            // Clean up stale spill dirs from previous killed builds.
            //
            // Safe for concurrent runs: only delete directories whose
            // contents haven't been modified in the last hour. A running
            // build writes to its `properties.log.zst` continuously, so
            // active spill dirs always look "fresh" and won't be touched.
            // This matters because a parallel `load_ntriples` call (e.g.
            // `api_benchmark.py` spawning multiple loaders, or a long
            // Wikidata rebuild running alongside a small test) would
            // otherwise wipe out the log file of the *other* run.
            const STALE_AFTER_SECS: u64 = 3600; // 1 hour
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
            // Clean up stale pending_edges from previous killed builds
            if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
                let stale = dg.data_dir.join("_pending_edges.bin");
                if stale.exists() {
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
    let mut edge_buffer = if use_compact {
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

    // Label journal for auto-typing. Previously a `HashMap<u32, String>`
    // that grew to ~10 GB of heap at Wikidata scale (124M entities),
    // pushing 16 GB machines into swap and collapsing the Phase 1
    // rate from 1.8M to 450K triples/s. Now a buffered sequential
    // write to `{spill_dir}/labels.bin` — zero heap growth during
    // Phase 1. The post-Phase-1 rename pass reads the journal once,
    // keeping only the ~88K labels that actually appear as type names.
    // In-Phase-1 `get` is gone entirely (it was best-effort anyway —
    // misses always fell through to post-Phase-1 rename).
    let mut label_writer: Option<super::label_spill::LabelSpillWriter> = if config.auto_type {
        let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
        });
        let _ = std::fs::create_dir_all(&spill_dir);
        Some(
            super::label_spill::LabelSpillWriter::new(&spill_dir.join("labels.bin"))
                .map_err(|e| format!("Failed to create label journal: {}", e))?,
        )
    } else {
        None
    };

    // Per-type build metadata for Phase 1b pre-allocation.
    let mut type_meta: HashMap<String, TypeBuildMeta> = HashMap::new();

    // For disk mode: build qnum_to_idx during Phase 1 (instead of from id_indices in Phase 2).
    // This lets us skip id_indices entirely, saving ~11 GB at full Wikidata scale.
    // Pre-allocate for 150M Q-numbers (~600 MB mmap, lazily paged).
    let mut qnum_to_idx: Option<MmapOrVec<u32>> = if graph.graph.is_disk() {
        let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
        });
        let _ = std::fs::create_dir_all(&spill_dir);
        // 150M covers all Wikidata Q-numbers. OS allocates pages lazily.
        Some(
            MmapOrVec::mapped_prefilled(&spill_dir.join("qnum_to_idx.bin"), 150_000_000)
                .unwrap_or_else(|_| MmapOrVec::from_vec(vec![0u32; 150_000_000])),
        )
    } else {
        None
    };

    let mut entity_limit_reached = false;
    // Reusable scratch buffer for `flush_entity`'s property-log
    // `Vec<(InternedKey, Value)>`. Hoisted out of `flush_entity` so the
    // alloc + grow cost is paid once, not per entity (~2% of loader CPU
    // under samply).
    let mut scratch_props: Vec<(InternedKey, Value)> = Vec::with_capacity(64);
    // Cheap bucket counter (decrement in the hot loop); every 5M triples we
    // check wall time and only log a `[Phase 1]` progress line if at least
    // PROGRESS_INTERVAL_SECS have elapsed since the last line.
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

    'outer: for batch in &rx {
        let n_lines = batch.offsets.len();
        for i in 0..n_lines {
            // Slice into the batch's contiguous buffer — pointer math,
            // no per-line heap dereference. SAFETY: bytes originated
            // from a UTF-8 input stream; `parse_line` re-validates the
            // sub-slices it actually returns.
            let line = unsafe { std::str::from_utf8_unchecked(batch.line(i)) };
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
                // eplog stays on the 60s gate so terminal output isn't spammed.
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

            // Note: the reader thread already filters out non-entity
            // lines via `starts_with(ENTITY_PREFIX)` before they reach
            // the channel, so no redundant prefix check is needed here.
            let (subject, predicate, object) = match parse_line(line) {
                Some(parsed) => parsed,
                None => continue,
            };

            let subj_id = match subject {
                Subject::Entity(id) => id,
                Subject::Other => continue,
            };

            // Subject changed → flush previous entity
            if current.as_ref().is_some_and(|c| c.id != subj_id) {
                if let Some(acc) = current.take() {
                    flush_entity(
                        graph,
                        acc,
                        config,
                        &mut edge_buffer,
                        &mut stats,
                        mapped,
                        &mut prop_log,
                        &mut label_writer,
                        &mut type_meta,
                        &mut qnum_to_idx,
                        &mut scratch_props,
                    );
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

            // Process triple based on predicate type
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
    let _ = reader_handle.join();

    // Flush last entity
    if let Some(acc) = current.take() {
        flush_entity(
            graph,
            acc,
            config,
            &mut edge_buffer,
            &mut stats,
            mapped,
            &mut prop_log,
            &mut label_writer,
            &mut type_meta,
            &mut qnum_to_idx,
            &mut scratch_props,
        );
    }

    // Post-Phase-1: resolve Q-code type names using the label journal.
    // During Phase 1 every entity's type stayed as its raw Q-code (e.g.
    // "Q5") because the old HashMap cache was removed to avoid the ~10
    // GB heap spike. Now we read the journal ONCE and pull labels only
    // for the small set of Q-codes that actually became type names —
    // typically tens of thousands on Wikidata, not the 124M entries
    // the in-memory cache held.
    let mut type_rename_map: HashMap<String, String> = HashMap::new();

    // Flush + close the label journal before reading it back.
    let label_journal_path = if let Some(writer) = label_writer.take() {
        let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
        });
        let path = spill_dir.join("labels.bin");
        let journal_size = writer.finish().unwrap_or(0);
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
        // Collect the Q-numbers that actually need label resolution —
        // only those that appear as type names in `type_indices`.
        let wanted: std::collections::HashSet<u32> = graph
            .type_indices
            .keys()
            .filter_map(parse_qcode_number)
            .collect();

        // Pull those labels (and only those) from the journal. One
        // forward scan; skips unwanted records without allocating.
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
            let old_key_to_new_key: Vec<(InternedKey, InternedKey)> = renames
                .iter()
                .map(|(old, new)| {
                    let old_key = graph.interner.get_or_intern(old);
                    let new_key = graph.interner.get_or_intern(new);
                    (old_key, new_key)
                })
                .collect();
            for (old_name, new_name) in &renames {
                // Merge type_indices: if target name already exists, append indices
                if let Some(indices) = graph.type_indices.remove(old_name) {
                    graph
                        .type_indices
                        .entry_or_default(new_name.clone())
                        .extend(indices);
                }
                // Merge node_type_metadata: keep the richer entry (more property keys)
                if let Some(old_meta) = graph.node_type_metadata.remove(old_name) {
                    let entry = graph
                        .node_type_metadata
                        .entry(new_name.clone())
                        .or_default();
                    for (k, v) in old_meta {
                        entry.entry(k).or_insert(v);
                    }
                }
                // Merge type_schemas: union property keys
                if let Some(old_schema) = graph.type_schemas.remove(old_name) {
                    if let Some(existing) = graph.type_schemas.get(new_name) {
                        let merged = existing.merge(&old_schema);
                        graph
                            .type_schemas
                            .insert(new_name.clone(), Arc::new(merged));
                    } else {
                        graph.type_schemas.insert(new_name.clone(), old_schema);
                    }
                }
                // Merge type_build_meta: combine row counts and column info
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

            // Update node_type InternedKey on affected nodes.
            // Build lookup map first, then ONE sequential pass over all nodes.
            // This is O(nodes) not O(renames × nodes).
            let rename_map: HashMap<u64, u64> = old_key_to_new_key
                .iter()
                .map(|(old, new)| (old.as_u64(), new.as_u64()))
                .collect();
            match &mut graph.graph {
                crate::graph::schema::GraphBackend::Disk(ref mut dg) => {
                    let n = dg.node_slots.len();
                    for i in 0..n {
                        let slot = dg.node_slots.get(i);
                        if slot.is_alive() {
                            if let Some(&new_type) = rename_map.get(&slot.node_type) {
                                let mut new_slot = slot;
                                new_slot.node_type = new_type;
                                dg.node_slots.set(i, new_slot);
                            }
                        }
                    }
                }
                crate::graph::schema::GraphBackend::Memory(ref mut g) => {
                    for i in 0..g.node_bound() {
                        let idx = petgraph::graph::NodeIndex::new(i);
                        if let Some(node) = g.node_weight_mut(idx) {
                            if let Some(&new_key) = rename_map.get(&node.node_type.as_u64()) {
                                node.node_type = InternedKey::from_u64(new_key);
                            }
                        }
                    }
                }
                crate::graph::schema::GraphBackend::Mapped(ref mut g) => {
                    for i in 0..g.node_bound() {
                        let idx = petgraph::graph::NodeIndex::new(i);
                        if let Some(node) = g.node_weight_mut(idx) {
                            if let Some(&new_key) = rename_map.get(&node.node_type.as_u64()) {
                                node.node_type = InternedKey::from_u64(new_key);
                            }
                        }
                    }
                }
                // RecordingGraph is a Phase 6 validation wrapper only
                // constructed in Rust tests; the ntriples loader never
                // sees it in practice.
                crate::graph::schema::GraphBackend::Recording(_) => {
                    unreachable!("ntriples loader does not run on a Recording-wrapped graph");
                }
            }
            if build_debug() {
                eplog!(
                    "  Resolved {} Q-code types ({})",
                    renames.len(),
                    fmt_dur(rename_start.elapsed().as_secs_f64())
                );
            }
        }
    }

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

    // Phase 1b: Convert to columnar storage.
    // For Disk mode: pre-allocate columns from metadata, then direct-write from log.
    // For Mapped mode: bulk convert from HashMap properties.
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
        super::column_builder::build_columns_direct(
            graph,
            &log_path,
            &type_meta,
            &type_rename_map,
            build_debug(),
            sink,
        )
        .map_err(|e| format!("Failed to build columns: {}", e))?;
        let _ = std::fs::remove_file(&log_path);
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::remove_dir(parent);
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
        // Mapped mode now goes through the `Some(log_writer)` arm above
        // (the disk path). The prior `enable_columnar()` per-entity loop
        // was the 7× bottleneck vs disk builds and has been retired for
        // N-Triples builds. Memory mode lands here and intentionally
        // does nothing — its columnar conversion is triggered on demand
        // elsewhere.
        debug_assert!(
            !graph.graph.is_mapped(),
            "mapped load_ntriples must populate prop_log and use build_columns_direct"
        );
    }

    // Free everything not needed for Phase 2+3 to maximize page cache.
    // Phase 2 only needs qnum_to_idx + edge_buffer. Phase 3 only needs pending_edges.
    if graph.graph.is_disk() {
        let dropped_stores = graph.column_stores.len();
        graph.column_stores.clear();
        graph.sync_disk_column_stores();
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
        create_edges_with_qnum_map(graph, &edge_buffer, &mut stats, qt, sink)?;
    } else {
        create_edges_from_buffer(graph, &edge_buffer, &mut stats, sink)?;
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
        dg.build_csr_from_pending();
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
                *graph.edge_type_counts_cache.write().unwrap() = Some(string_counts);
            }
        }
    }

    // Rebuild type_indices from DiskNodeSlots (dropped before Phase 2 to save 1 GB)
    if graph.graph.is_disk() {
        let rebuild_start = Instant::now();
        if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
            for i in 0..dg.node_slots.len() {
                let slot = dg.node_slots.get(i);
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

    // Reload column stores by re-opening the mmap file + reading saved metadata.
    if graph.graph.is_disk() {
        let data_dir = if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
            dg.data_dir.clone()
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
                // SAFETY: columns.bin was just written by a prior save pass;
                // we re-open + mmap the existing file read-write. No other
                // handle is writing to it concurrently.
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
                    graph
                        .column_stores
                        .insert(type_meta.type_name.clone(), Arc::new(store));
                }
                Ok(())
            })();

            if let Err(e) = reload_result {
                eplog!("  Warning: failed to reload column stores: {}", e);
            }

            if build_debug() {
                eplog!(
                    "  Reloaded {} column stores from mmap ({})",
                    graph.column_stores.len(),
                    fmt_dur(reload_start.elapsed().as_secs_f64()),
                );
            }
        }
        graph.sync_disk_column_stores();
    }

    // Build id_indices for all types so WHERE id(n) = X is O(1).
    // Uses column stores directly — no node materialization, no arena growth.
    if graph.graph.is_disk() {
        let id_start = Instant::now();
        let type_names: Vec<String> = graph.type_indices.keys().map(|s| s.to_string()).collect();
        for type_name in &type_names {
            graph.build_id_index_from_columns(type_name);
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
        if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
            let data_dir = dg.data_dir.clone();

            // Build type_connectivity_cache from connection_type_metadata + edge_type_counts.
            // This makes describe(types=['human']) instant instead of scanning 10K nodes.
            {
                let mut triples = Vec::new();
                for (conn_type, info) in &graph.connection_type_metadata {
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

            // Save interner
            let save_step = Instant::now();
            let interner_map: HashMap<String, String> = graph
                .interner
                .iter()
                .map(|(k, v)| (k.as_u64().to_string(), v.to_string()))
                .collect();
            if let Ok(json) = serde_json::to_string(&interner_map) {
                let _ = std::fs::write(data_dir.join("interner.json"), json);
            }
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
            let _ = crate::graph::io::file::write_node_type_metadata_bin(&data_dir, graph);
            let _ = crate::graph::io::file::write_connection_type_metadata_bin(&data_dir, graph);
            let mut meta = crate::graph::io::file::build_disk_metadata(graph);
            crate::graph::io::file::strip_heavy_metadata(&mut meta);
            if let Ok(json) = serde_json::to_string_pretty(&meta) {
                let _ = std::fs::write(data_dir.join("metadata.json"), json);
            }
            if build_debug() {
                eplog!("  metadata: {}", fmt_dur(save_step.elapsed().as_secs_f64()));
            }

            // Save id_indices as raw mmap-friendly `.bin` (0.8.28+).
            let save_step = Instant::now();
            if !graph.id_indices.is_empty() {
                let _ = crate::graph::storage::disk::id_index::write_id_indices_bin(
                    &data_dir,
                    &graph.id_indices,
                    &graph.interner,
                );
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
                let _ = crate::graph::storage::disk::type_index::write_type_indices_bin(
                    &data_dir,
                    &graph.type_indices,
                    &graph.interner,
                );
            }
            if build_debug() {
                eplog!(
                    "  type_indices.bin ({} types): {}",
                    graph.type_indices.len(),
                    fmt_dur(save_step.elapsed().as_secs_f64())
                );
            }
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

    stats.seconds = start.elapsed().as_secs_f64();
    if config.verbose {
        eplog!("[Build] Total elapsed: {}", fmt_dur(stats.seconds));
    }
    Ok(stats)
}

/// Flush an accumulated entity into the graph as a node.
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
    // Reusable buffer for the property-log key/value pairs. Hoisted out
    // of this function so the alloc cost is paid once instead of per
    // entity (showed up at ~2% of loader CPU under samply).
    scratch_props: &mut Vec<(InternedKey, Value)>,
) {
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
        return;
    }

    let title = acc.label.unwrap_or_else(|| acc.id.clone());

    // Append to the label journal for post-Phase-1 type resolution.
    // Sequential write only — zero heap pressure during the streaming
    // phase. The previous HashMap-based cache grew to ~10 GB at
    // Wikidata scale and caused swap thrash on 16 GB machines.
    if let Some(ref mut w) = label_writer {
        if let Some(qnum) = parse_qcode_number(&acc.id) {
            let _ = w.append(qnum, &title);
        }
    }

    // Determine node type from P31 value. During Phase 1 we always use
    // the raw Q-code when auto_type is on — the post-Phase-1 rename
    // pass resolves it to the human-readable label using the journal.
    // This avoids the old in-loop HashMap lookup on a 124M-entry map.
    let node_type = if let Some(ref tq) = acc.type_qcode {
        if let Some(mapped_name) = config.node_types.get(tq) {
            mapped_name.clone()
        } else if config.auto_type {
            // Raw Q-code; post-Phase-1 rename will resolve to label.
            tq.clone()
        } else {
            "Entity".to_string()
        }
    } else {
        "Entity".to_string()
    };

    let mut properties = acc.properties;
    // Store nid as a queryable string property (e.g., "Q42") so Cypher {nid: 'Q42'} works
    properties.insert("nid".to_string(), Value::String(acc.id.clone()));
    if let Some(desc) = acc.description {
        properties.insert("description".to_string(), Value::String(desc));
    }
    // Store P31 Q-code as a property (preserves type info when defaulting to "Entity")
    if let Some(ref tq) = acc.type_qcode {
        properties.insert("P31".to_string(), Value::String(tq.clone()));
    }

    // Choose ID representation: compact u32 for disk only, String for default/mapped
    let use_compact_ids = graph.graph.is_disk();
    let id_value = if mapped || use_compact_ids {
        parse_qcode_number(&acc.id)
            .map(Value::UniqueId)
            .unwrap_or_else(|| Value::String(acc.id.clone()))
    } else {
        Value::String(acc.id.clone())
    };
    let title_value = Value::String(title);

    let mut node_data = NodeData::new(
        id_value.clone(),
        title_value,
        node_type.clone(),
        properties,
        &mut graph.interner,
    );

    // Mapped mode: push properties into ColumnStore (existing path).
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

    // For disk mode: serialize properties to the log file BEFORE clearing them.
    // We need node_data properties to still be present for the log write,
    // then clear them before add_node (DiskGraph discards them anyway).
    let saved_id = node_data.id.clone();
    let saved_title = node_data.title.clone();
    if prop_log.is_some() {
        // Reuse `scratch_props` instead of allocating a fresh Vec per
        // flush. `clear()` preserves capacity so subsequent entities
        // skip the alloc + grow.
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
        .expect("Property log write failed");

        // Collect per-type metadata for Phase 1b pre-allocation
        type_meta
            .entry(node_type.clone())
            .or_insert_with(TypeBuildMeta::new)
            .record_entity(&saved_id, &saved_title, scratch_props);
    }

    // Update type_indices
    graph
        .type_indices
        .entry_or_default(node_type.clone())
        .push(node_idx);

    // For disk mode: write directly to qnum_to_idx mmap (skip id_indices to save ~11 GB RAM).
    // For other modes: use id_indices HashMap as before.
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

    // Periodic spill: every 100K entities, check if columns should be spilled to disk
    if mapped && stats.entities_created.is_multiple_of(100_000) {
        graph.maybe_spill_columns();
    }

    // Buffer outgoing edges
    match edge_buffer {
        EdgeBuffer::Compact(buf) => {
            if let Some(src_num) = parse_qcode_number(&acc.id) {
                for (pred_label, target_qcode) in acc.outgoing_edges {
                    if let Some(tgt_num) = parse_qcode_number(&target_qcode) {
                        let pred_key = graph.interner.get_or_intern(&pred_label);
                        buf.push((src_num, tgt_num, pred_key));
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
}

/// Create edges from the buffer. Looks up source/target by Q-code across all types.
/// Fast edge creation using pre-built qnum_to_idx from Phase 1 (disk mode).
/// Avoids rebuilding the lookup table from id_indices (saves ~11 GB RAM at full scale).
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
#[allow(clippy::approx_constant)]
mod tests {
    use super::super::parser::{XSD_BOOLEAN, XSD_DECIMAL, XSD_DOUBLE};
    use super::*;

    #[test]
    fn test_parse_entity_triple() {
        let line = r#"<http://www.wikidata.org/entity/Q42> <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> ."#;
        let (subj, pred, obj) = parse_line(line).unwrap();
        assert!(matches!(subj, Subject::Entity("Q42")));
        assert!(matches!(pred, Predicate::WikidataDirect("P31")));
        assert!(matches!(obj, Object::Entity("Q5")));
    }

    #[test]
    fn test_parse_literal_triple() {
        let line = r#"<http://www.wikidata.org/entity/Q42> <http://www.w3.org/2000/01/rdf-schema#label> "Douglas Adams"@en ."#;
        let (subj, pred, obj) = parse_line(line).unwrap();
        assert!(matches!(subj, Subject::Entity("Q42")));
        assert!(matches!(pred, Predicate::Label));
        assert!(matches!(obj, Object::LangLiteral(ref t, "en") if t == "Douglas Adams"));
    }

    #[test]
    fn test_parse_typed_literal() {
        let line = r#"<http://www.wikidata.org/entity/Q31> <http://www.wikidata.org/prop/direct/P1082> "+11825551"^^<http://www.w3.org/2001/XMLSchema#decimal> ."#;
        let (_, pred, obj) = parse_line(line).unwrap();
        assert!(matches!(pred, Predicate::WikidataDirect("P1082")));
        assert!(matches!(obj, Object::TypedLiteral(ref t, _) if t == "+11825551"));
    }

    #[test]
    fn test_parse_escaped_string() {
        let line = r#"<http://www.wikidata.org/entity/Q31> <http://www.wikidata.org/prop/direct/P1448> "K\u00F6nigreich Belgien"@de ."#;
        let (_, _, obj) = parse_line(line).unwrap();
        assert!(matches!(obj, Object::LangLiteral(ref t, "de") if t == "Königreich Belgien"));
    }

    #[test]
    fn test_typed_literal_to_value() {
        assert_eq!(
            typed_literal_to_value("+11825551", XSD_DECIMAL),
            Value::Int64(11825551)
        );
        assert_eq!(
            typed_literal_to_value("3.14", XSD_DOUBLE),
            Value::Float64(3.14)
        );
        assert_eq!(
            typed_literal_to_value("true", XSD_BOOLEAN),
            Value::Boolean(true)
        );
    }

    #[test]
    fn test_language_filter() {
        let filter = Some(HashSet::from(["en".to_string()]));
        assert!(language_matches("en", &filter));
        assert!(!language_matches("de", &filter));
        assert!(language_matches("de", &None));
    }

    #[test]
    fn test_parse_qcode_number() {
        assert_eq!(parse_qcode_number("Q42"), Some(42));
        assert_eq!(parse_qcode_number("Q0"), Some(0));
        assert_eq!(parse_qcode_number("Q130000000"), Some(130_000_000));
        assert_eq!(parse_qcode_number("P31"), None); // not a Q-code
        assert_eq!(parse_qcode_number("Q"), None); // no number
        assert_eq!(parse_qcode_number(""), None); // empty
        assert_eq!(parse_qcode_number("Q-1"), None); // negative
    }

    #[test]
    fn test_edge_buffer_compact_size() {
        // Verify compact edge buffer entry is much smaller than string-based
        assert_eq!(
            std::mem::size_of::<(u32, u32, InternedKey)>(),
            16 // 4 + 4 + 8
        );
        // String tuple is at least 72 bytes on stack (3 × 24 for String)
        assert!(std::mem::size_of::<(String, String, String)>() >= 72);
    }
}
