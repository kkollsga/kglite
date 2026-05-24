//! Type connectivity triples — derived statistics about which types
//! connect to which, via which connection type.

use crate::graph::schema::{ConnectivityTriple, DirGraph, GraphBackend, InternedKey};
use crate::graph::storage::GraphRead;
use std::collections::{HashMap, HashSet};

use super::{NeighborConnection, NeighborsSchema};

// ── Type connectivity ──────────────────────────────────────────────────────

type CountMap = HashMap<(InternedKey, InternedKey, InternedKey), usize>;

/// Compute type connectivity triples via a single O(E) pass.
///
/// Uses InternedKey-based aggregation (no string allocation during scan)
/// and sequential iteration for cache-friendly I/O on disk graphs.
/// Resolves keys to strings only at the end.
///
/// **Hot path (861M+ edges on Wikidata).** Disk-backed graphs use a
/// Rayon shard-and-merge parallel scan over `edge_endpoints` directly
/// (same pattern as `DiskGraph::build_peer_count_histogram`). This gives
/// an 8–10× wall-clock win on multi-core machines for billion-edge graphs.
/// Memory and Mapped modes keep the single-threaded closure path —
/// they're either small enough that overhead dominates, or their edge
/// iteration isn't trivially parallelisable through petgraph.
pub fn compute_type_connectivity(graph: &DirGraph) -> Vec<ConnectivityTriple> {
    let backend = &graph.graph;

    // Aggregate with InternedKey tuples — no string allocation per edge.
    let counts: CountMap = match backend {
        GraphBackend::Disk(dg) => compute_disk_parallel(dg),
        _ => compute_serial(backend),
    };

    // Resolve to strings once at the end
    let mut triples: Vec<ConnectivityTriple> = counts
        .into_iter()
        .map(|((sk, ck, tk), count)| ConnectivityTriple {
            src: graph.interner.resolve(sk).to_string(),
            conn: graph.interner.resolve(ck).to_string(),
            tgt: graph.interner.resolve(tk).to_string(),
            count,
        })
        .collect();
    triples.sort_by_key(|t| std::cmp::Reverse(t.count));
    triples
}

/// Disk-backend fast path: shard the edge range, scan each chunk in
/// parallel with per-shard HashMaps, merge serially at the end.
/// Mirrors `DiskGraph::build_peer_count_histogram` (same chunking +
/// `advise_sequential` pattern). On Wikidata-scale graphs this cuts
/// `rebuild_caches` from 200+ s to tens of seconds.
fn compute_disk_parallel(dg: &crate::graph::storage::disk::graph::DiskGraph) -> CountMap {
    use crate::graph::storage::disk::csr::TOMBSTONE_EDGE;
    use petgraph::graph::NodeIndex;
    use rayon::prelude::*;

    let total = (dg.next_edge_idx as usize).min(dg.edge_endpoints.len());
    if total == 0 {
        return HashMap::new();
    }

    // Prefetch: long sequential scan followed by a drop.
    dg.edge_endpoints.advise_sequential();

    // Chunk size matches the histogram builder — at least 1M edges per
    // shard so per-thread bookkeeping stays amortised, and at most
    // `total / n_threads` so all cores get work.
    let chunk = (total / rayon::current_num_threads().max(1)).max(1 << 20);
    let ranges: Vec<(usize, usize)> = (0..total)
        .step_by(chunk)
        .map(|lo| (lo, (lo + chunk).min(total)))
        .collect();

    let shard_maps: Vec<CountMap> = ranges
        .into_par_iter()
        .map(|(lo, hi)| {
            let mut acc: CountMap = HashMap::new();
            for i in lo..hi {
                let ep = dg.edge_endpoints.get(i);
                if ep.source == TOMBSTONE_EDGE {
                    continue;
                }
                let src = NodeIndex::new(ep.source as usize);
                let tgt = NodeIndex::new(ep.target as usize);
                if let (Some(sk), Some(tk)) = (dg.node_type_of(src), dg.node_type_of(tgt)) {
                    let conn = InternedKey::from_u64(ep.connection_type);
                    *acc.entry((sk, conn, tk)).or_insert(0) += 1;
                }
            }
            acc
        })
        .collect();

    dg.edge_endpoints.advise_dontneed();

    // Merge shards serially — hot keys cluster fast, and parallel merge
    // would need a mutex that defeats the per-shard-isolation win.
    let mut combined: CountMap = HashMap::new();
    for shard in shard_maps {
        for (k, v) in shard {
            *combined.entry(k).or_insert(0) += v;
        }
    }
    combined
}

/// Single-threaded fallback for Memory / Mapped / Recording backends.
/// petgraph's `edge_references` isn't trivially Rayon-parallel and
/// these backends operate at scales (thousands to low-millions of
/// edges) where threading overhead dominates.
fn compute_serial(backend: &GraphBackend) -> CountMap {
    let mut counts: CountMap = HashMap::new();
    backend.for_each_edge_endpoint_key(|src_idx, tgt_idx, conn_key| {
        let src_key = backend.node_type_of(src_idx);
        let tgt_key = backend.node_type_of(tgt_idx);
        if let (Some(sk), Some(tk)) = (src_key, tgt_key) {
            *counts.entry((sk, conn_key, tk)).or_insert(0) += 1;
        }
    });
    counts
}

/// Build NeighborsSchema for a specific type from pre-computed triples.
/// O(triples) linear scan — use `TypeConnectivityIndex` for O(1) lookups.
pub fn neighbors_from_triples(triples: &[ConnectivityTriple], node_type: &str) -> NeighborsSchema {
    let mut outgoing: Vec<NeighborConnection> = Vec::new();
    let mut incoming: Vec<NeighborConnection> = Vec::new();

    for t in triples {
        if t.src == node_type {
            outgoing.push(NeighborConnection {
                connection_type: t.conn.clone(),
                other_type: t.tgt.clone(),
                count: t.count,
            });
        }
        if t.tgt == node_type {
            incoming.push(NeighborConnection {
                connection_type: t.conn.clone(),
                other_type: t.src.clone(),
                count: t.count,
            });
        }
    }

    outgoing.sort_by_key(|o| std::cmp::Reverse(o.count));
    incoming.sort_by_key(|i| std::cmp::Reverse(i.count));

    NeighborsSchema { outgoing, incoming }
}

/// Pre-indexed type connectivity for O(1) neighbor lookups.
/// Built once from triples, used for all describe operations in a session.
pub struct TypeConnectivityIndex {
    /// type_name → (outgoing, incoming) neighbor connections, sorted by count desc.
    index: HashMap<String, NeighborsSchema>,
}

impl TypeConnectivityIndex {
    /// Build index from flat triples. O(triples) one-time cost.
    pub fn from_triples(triples: &[ConnectivityTriple]) -> Self {
        let mut out_map: HashMap<String, Vec<NeighborConnection>> = HashMap::new();
        let mut in_map: HashMap<String, Vec<NeighborConnection>> = HashMap::new();

        for t in triples {
            out_map
                .entry(t.src.clone())
                .or_default()
                .push(NeighborConnection {
                    connection_type: t.conn.clone(),
                    other_type: t.tgt.clone(),
                    count: t.count,
                });
            in_map
                .entry(t.tgt.clone())
                .or_default()
                .push(NeighborConnection {
                    connection_type: t.conn.clone(),
                    other_type: t.src.clone(),
                    count: t.count,
                });
        }

        // Merge into NeighborsSchema per type, sort by count desc.
        // Collect all type names first (owned) to avoid borrow conflicts.
        let all_types: HashSet<String> = out_map.keys().chain(in_map.keys()).cloned().collect();

        let mut index = HashMap::with_capacity(all_types.len());
        for nt in all_types {
            let mut outgoing = out_map.remove(&nt).unwrap_or_default();
            outgoing.sort_by_key(|o| std::cmp::Reverse(o.count));
            let mut incoming = in_map.remove(&nt).unwrap_or_default();
            incoming.sort_by_key(|i| std::cmp::Reverse(i.count));
            index.insert(nt, NeighborsSchema { outgoing, incoming });
        }

        TypeConnectivityIndex { index }
    }

    /// O(1) lookup of neighbors for a type.
    pub fn get(&self, node_type: &str) -> NeighborsSchema {
        self.index
            .get(node_type)
            .cloned()
            .unwrap_or(NeighborsSchema {
                outgoing: Vec::new(),
                incoming: Vec::new(),
            })
    }
}

/// Derived edge statistics from type connectivity triples.
pub struct DerivedEdgeStats {
    /// Edge type → count.
    pub counts: HashMap<String, usize>,
    /// Edge type → (source_types, target_types).
    pub endpoints: HashMap<String, (HashSet<String>, HashSet<String>)>,
}

/// Derive edge type counts + endpoint types from type connectivity triples.
/// Avoids separate O(E) scans for these derived data.
pub fn derive_edge_counts_from_triples(triples: &[ConnectivityTriple]) -> DerivedEdgeStats {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut endpoints: HashMap<String, (HashSet<String>, HashSet<String>)> = HashMap::new();

    for t in triples {
        *counts.entry(t.conn.clone()).or_insert(0) += t.count;
        let entry = endpoints
            .entry(t.conn.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()));
        entry.0.insert(t.src.clone());
        entry.1.insert(t.tgt.clone());
    }

    DerivedEdgeStats { counts, endpoints }
}
