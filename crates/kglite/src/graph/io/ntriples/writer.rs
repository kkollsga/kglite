//! Edge-creation dispatch — writes buffered (source_id, target_id, predicate)
//! triples into the graph, using one of three paths depending on mode.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, EdgeData, InternedKey};
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use crate::graph::storage::{GraphRead, GraphWrite};
use std::collections::{HashMap, HashSet};

use super::parser::{parse_qcode_number, CompactNTripleEdge, EdgeBuffer};
use super::{Cancelled, NTriplesStats, ProgressEvent, ProgressSink, ProgressValue as PV};

/// Edges between Phase 2 progress callbacks. Roughly ~200 ms apart at
/// in-memory edge-creation rates (~5 M edges/s), which gives Ctrl+C a
/// responsive cancel point and tqdm a smoothly-moving bar.
const PHASE2_TICK: usize = 1_000_000;

/// Helper used inside the hot loops below: emit an `update` event for
/// Phase 2 and propagate `Cancelled` (Ctrl+C) up as a string error.
#[inline]
fn phase2_tick(
    sink: Option<&dyn ProgressSink>,
    current: u64,
    edges_created: u64,
    edges_skipped: u64,
) -> Result<(), String> {
    if let Some(s) = sink {
        s.emit(ProgressEvent::Update {
            phase: "phase2",
            current,
            fields: &[
                ("edges_created", PV::U64(edges_created)),
                ("edges_skipped", PV::U64(edges_skipped)),
            ],
        })
        .map_err(|Cancelled| "<cancelled>".to_string())?;
    }
    Ok(())
}

pub(super) fn create_edges_with_qnum_map(
    graph: &mut DirGraph,
    edge_buffer: &EdgeBuffer,
    stats: &mut NTriplesStats,
    qnum_to_idx: &MmapOrVec<u32>,
    sink: Option<&dyn ProgressSink>,
) -> Result<(), String> {
    let buf = match edge_buffer {
        EdgeBuffer::Compact(b) => b,
        EdgeBuffer::Strings(_) => return Ok(()), // shouldn't happen for disk mode
    };

    let qnum_len = qnum_to_idx.len();
    let buf_len = buf.len();
    let mut conn_types_seen: HashSet<InternedKey> = HashSet::new();

    let lookup = |qnum: u32| -> Option<u32> {
        if (qnum as usize) >= qnum_len {
            return None;
        }
        let v = qnum_to_idx.get(qnum as usize);
        if v == 0 {
            None
        } else {
            Some(v - 1)
        }
    };

    if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
        for i in 0..buf_len {
            let edge = buf.get(i);
            let (src_num, tgt_num, pred_key) = (edge.source_qnum, edge.target_qnum, edge.predicate);
            if let (Some(src_idx), Some(tgt_idx)) = (lookup(src_num), lookup(tgt_num)) {
                dg.try_add_pending_edge(
                    petgraph::graph::NodeIndex::new(src_idx as usize),
                    petgraph::graph::NodeIndex::new(tgt_idx as usize),
                    EdgeData::new_interned(pred_key, Vec::new()),
                )
                .map_err(|error| format!("append pending N-Triples edge: {error}"))?;
                conn_types_seen.insert(pred_key);
                stats.edges_created += 1;
            } else {
                stats.edges_skipped += 1;
            }
            if (i + 1) % PHASE2_TICK == 0 {
                phase2_tick(
                    sink,
                    (i + 1) as u64,
                    stats.edges_created,
                    stats.edges_skipped,
                )?;
            }
        }
    }

    // Register connection type names (no O(types²) metadata loop).
    for conn_key in &conn_types_seen {
        let conn_name = graph.interner.resolve(*conn_key).to_string();
        graph.connection_type_metadata.entry(conn_name).or_default();
    }
    graph.invalidate_edge_type_counts_cache();
    Ok(())
}

pub(super) fn create_edges_from_buffer(
    graph: &mut DirGraph,
    edge_buffer: &EdgeBuffer,
    stats: &mut NTriplesStats,
    sink: Option<&dyn ProgressSink>,
) -> Result<(), String> {
    match edge_buffer {
        EdgeBuffer::Compact(buf) => create_edges_compact(graph, buf, stats, sink),
        EdgeBuffer::Strings(buf) => create_edges_strings(graph, buf, stats, sink),
    }
}

/// Compact path: edges stored as [`CompactNTripleEdge`].
/// Uses dense Vec lookup (not HashMap) for cache-friendly O(1) access.
/// Streams edges directly — no intermediate allocation.
pub(super) fn create_edges_compact(
    graph: &mut DirGraph,
    buf: &MmapOrVec<CompactNTripleEdge>,
    stats: &mut NTriplesStats,
    sink: Option<&dyn ProgressSink>,
) -> Result<(), String> {
    // Build dense Vec lookup: Q-number → NodeIndex.
    // Much faster than HashMap for Wikidata's dense Q-number space.
    let mut max_qnum: u32 = 0;
    for id_map in graph.id_indices.values() {
        for (id_val, _) in id_map.iter() {
            let n = match id_val {
                Value::UniqueId(n) => n,
                Value::String(s) => {
                    if let Some(n) = parse_qcode_number(s.as_str()) {
                        n
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            if n > max_qnum {
                max_qnum = n;
            }
        }
    }

    // File-backed dense lookup: qnum → (node_index + 1). Zero = not present.
    // Uses mapped_prefilled which is zero-initialized by OS (lazy pages), so only
    // pages we write to consume I/O. No 0xFF fill needed.
    let qnum_count = max_qnum as usize + 1;
    let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
    });
    let _ = std::fs::create_dir_all(&spill_dir);
    let mut qnum_to_idx: MmapOrVec<u32> =
        MmapOrVec::mapped_prefilled(&spill_dir.join("qnum_to_idx.bin"), qnum_count)
            .unwrap_or_else(|_| MmapOrVec::from_vec(vec![0u32; qnum_count]));
    // Store node_index + 1 so 0 remains the "not present" sentinel
    for id_map in graph.id_indices.values() {
        for (id_val, node_idx) in id_map.iter() {
            let n = match id_val {
                Value::UniqueId(n) => n,
                Value::String(s) => {
                    if let Some(n) = parse_qcode_number(s.as_str()) {
                        n
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            qnum_to_idx.set(n as usize, node_idx.index() as u32 + 1);
        }
    }

    // Track unique connection types (for metadata, computed once — not per edge)
    let mut conn_types_seen: HashSet<InternedKey> = HashSet::new();

    // Stream edges — use direct pending_edges push for disk mode (bypass add_edge overhead)
    let buf_len = buf.len();
    let qnum_len = qnum_to_idx.len();
    let is_disk = GraphRead::is_disk(&graph.graph);

    // Lookup helper: qnum_to_idx stores node_index+1, 0 = not present
    let lookup = |qnum: u32| -> Option<u32> {
        if (qnum as usize) >= qnum_len {
            return None;
        }
        let v = qnum_to_idx.get(qnum as usize);
        if v == 0 {
            None
        } else {
            Some(v - 1)
        }
    };

    if is_disk {
        // Fast path for disk mode: push directly to pending_edges.
        // Keep this loop LEAN — no random I/O per edge.
        if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
            for i in 0..buf_len {
                let edge = buf.get(i);
                let (src_num, tgt_num, pred_key) =
                    (edge.source_qnum, edge.target_qnum, edge.predicate);
                if let (Some(src_idx), Some(tgt_idx)) = (lookup(src_num), lookup(tgt_num)) {
                    dg.try_add_pending_edge(
                        petgraph::graph::NodeIndex::new(src_idx as usize),
                        petgraph::graph::NodeIndex::new(tgt_idx as usize),
                        EdgeData::new_interned(pred_key, Vec::new()),
                    )
                    .map_err(|error| format!("append pending N-Triples edge: {error}"))?;
                    conn_types_seen.insert(pred_key);
                    stats.edges_created += 1;
                } else {
                    stats.edges_skipped += 1;
                }
                if (i + 1) % PHASE2_TICK == 0 {
                    phase2_tick(
                        sink,
                        (i + 1) as u64,
                        stats.edges_created,
                        stats.edges_skipped,
                    )?;
                }
            }
        }
    } else {
        // Standard path for petgraph: per-edge add_edge
        for i in 0..buf_len {
            let edge = buf.get(i);
            let (src_num, tgt_num, pred_key) = (edge.source_qnum, edge.target_qnum, edge.predicate);
            if let (Some(src_idx), Some(tgt_idx)) = (lookup(src_num), lookup(tgt_num)) {
                let src = petgraph::graph::NodeIndex::new(src_idx as usize);
                let tgt = petgraph::graph::NodeIndex::new(tgt_idx as usize);
                let edge_data = EdgeData {
                    connection_type: pred_key,
                    properties: Vec::new(),
                };
                GraphWrite::add_edge(&mut graph.graph, src, tgt, edge_data);
                conn_types_seen.insert(pred_key);
                stats.edges_created += 1;
            } else {
                stats.edges_skipped += 1;
            }
            if (i + 1) % PHASE2_TICK == 0 {
                phase2_tick(
                    sink,
                    (i + 1) as u64,
                    stats.edges_created,
                    stats.edges_skipped,
                )?;
            }
        }
    }

    // Register connection type names (no O(types²) metadata loop).
    for conn_key in &conn_types_seen {
        let conn_name = graph.interner.resolve(*conn_key).to_string();
        graph.connection_type_metadata.entry(conn_name).or_default();
    }

    graph.invalidate_edge_type_counts_cache();

    // Clean up qnum_to_idx temp file
    let qnum_path = qnum_to_idx.file_path().map(|p| p.to_path_buf());
    drop(qnum_to_idx);
    if let Some(path) = qnum_path {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

/// String path: edges stored as (String, String, String).
pub(super) fn create_edges_strings(
    graph: &mut DirGraph,
    buf: &[(String, String, String)],
    stats: &mut NTriplesStats,
    sink: Option<&dyn ProgressSink>,
) -> Result<(), String> {
    // Build Q-code → NodeIndex lookup. Since 0.11.0 a parseable Q-code id is
    // stored as `Value::UniqueId(42)` (cross-mode parity), so map the edge
    // buffer's `"Q42"` strings via `parse_qcode_number` → u32. Non-parseable
    // ids remain `Value::String` and are matched verbatim.
    let mut qnum_to_idx: HashMap<u32, petgraph::graph::NodeIndex> = HashMap::new();
    let mut qstr_to_idx: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();
    for id_map in graph.id_indices.values() {
        for (id_val, node_idx) in id_map.iter() {
            match id_val {
                Value::UniqueId(n) => {
                    qnum_to_idx.insert(n, node_idx);
                }
                Value::String(s) => {
                    qstr_to_idx.insert(s, node_idx);
                }
                _ => {}
            }
        }
    }
    let lookup = |qcode: &str| -> Option<petgraph::graph::NodeIndex> {
        parse_qcode_number(qcode)
            .and_then(|n| qnum_to_idx.get(&n).copied())
            .or_else(|| qstr_to_idx.get(qcode).copied())
    };

    let mut conn_type_pairs: HashMap<String, (HashSet<String>, HashSet<String>)> = HashMap::new();

    for (i, (source_qcode, target_qcode, pred_label)) in buf.iter().enumerate() {
        let source_idx = lookup(source_qcode.as_str());
        let target_idx = lookup(target_qcode.as_str());

        match (source_idx, target_idx) {
            (Some(src), Some(tgt)) => {
                let edge_data =
                    EdgeData::new(pred_label.clone(), HashMap::new(), &mut graph.interner);

                let src_type = GraphRead::node_weight(&graph.graph, src)
                    .unwrap()
                    .node_type_str(&graph.interner)
                    .to_string();
                let tgt_type = GraphRead::node_weight(&graph.graph, tgt)
                    .unwrap()
                    .node_type_str(&graph.interner)
                    .to_string();
                let entry = conn_type_pairs
                    .entry(pred_label.clone())
                    .or_insert_with(|| (HashSet::new(), HashSet::new()));
                entry.0.insert(src_type);
                entry.1.insert(tgt_type);

                GraphWrite::add_edge(&mut graph.graph, src, tgt, edge_data);
                stats.edges_created += 1;
            }
            _ => {
                stats.edges_skipped += 1;
            }
        }
        if (i + 1) % PHASE2_TICK == 0 {
            phase2_tick(
                sink,
                (i + 1) as u64,
                stats.edges_created,
                stats.edges_skipped,
            )?;
        }
    }

    for (conn_type, (source_types, target_types)) in conn_type_pairs {
        for src_type in &source_types {
            for tgt_type in &target_types {
                graph.upsert_connection_type_metadata(
                    &conn_type,
                    src_type,
                    tgt_type,
                    HashMap::new(),
                );
            }
        }
    }

    graph.invalidate_edge_type_counts_cache();
    Ok(())
}
