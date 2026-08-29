//! Type connectivity triples — derived statistics about which types
//! connect to which, via which connection type.

use crate::graph::schema::{ConnectivityTriple, DirGraph, GraphBackend, InternedKey};
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::GraphRead;
use std::collections::{HashMap, HashSet};

use super::{NeighborConnection, NeighborsSchema};

type CountMap = HashMap<(InternedKey, InternedKey, InternedKey), usize>;

/// Compute type connectivity triples in a single O(E) pass, aggregating on
/// `InternedKey` tuples (no string allocation during the scan) and resolving
/// to strings only at the end.
///
/// **Hot path (861M+ edges on Wikidata).** Disk-backed graphs take a Rayon
/// shard-and-merge scan over `edge_endpoints` — an 8–10× wall-clock win on
/// multi-core machines for billion-edge graphs. Memory and Mapped keep the
/// single-threaded path (see [`compute_serial`]).
pub fn compute_type_connectivity(graph: &DirGraph) -> Vec<ConnectivityTriple> {
    let backend = &graph.graph;

    let counts: CountMap = match disk_scan_target(backend) {
        Some(dg) => compute_disk_parallel(dg),
        None => compute_serial(backend),
    };

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

/// Impose the canonical persisted order on connectivity triples.
///
/// Every writer of the cache — the `.kgl` metadata field and the disk-mode
/// `type_connectivity.bin.zst` sidecar — sorts through here. The triples reach
/// the cache in `HashMap` iteration order, which is reseeded per process, and
/// every reader treats the list as a set keyed by `(src, conn, tgt)`, so the
/// order is free to fix — and fixing it is what makes two saves of one graph
/// produce identical bytes.
///
/// Deliberately not applied in the getter: a Wikidata-scale graph carries
/// millions of triples and the planner reads them per query.
pub fn sort_connectivity_triples(triples: &mut [ConnectivityTriple]) {
    triples.sort_unstable_by(|a, b| {
        a.src
            .cmp(&b.src)
            .then_with(|| a.conn.cmp(&b.conn))
            .then_with(|| a.tgt.cmp(&b.tgt))
    });
}

/// The disk graph [`compute_type_connectivity`] will Rayon-scan, or `None`
/// for the serial path. Named because the routing decision is the only thing
/// a test can observe here — both paths produce identical counts, so nothing
/// in the result distinguishes them.
///
/// Goes through `GraphBackend::as_disk`, which looks *through* the
/// write-capture wrapper that `durable=True` and `cdc::enable` install: a bare
/// `GraphBackend::Disk` match answers `None` for every such graph and silently
/// forfeits the parallel scan on exactly the billion-edge graphs that need it.
fn disk_scan_target(backend: &GraphBackend) -> Option<&DiskGraph> {
    backend.as_disk()
}

/// Disk-backend fast path: shard the edge range, scan each chunk in
/// parallel with per-shard HashMaps, merge serially at the end.
/// Mirrors `DiskGraph::build_peer_count_histogram` (same chunking +
/// `advise_sequential` pattern). On Wikidata-scale graphs this cuts
/// `rebuild_caches` from 200+ s to tens of seconds.
fn compute_disk_parallel(dg: &DiskGraph) -> CountMap {
    use crate::graph::storage::disk::csr::TOMBSTONE_EDGE;
    use petgraph::graph::NodeIndex;
    use rayon::prelude::*;

    let total = (dg.next_edge_idx as usize).min(dg.edge_endpoint_len());
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
                let ep = dg.edge_endpoint(i);
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

/// Single-threaded fallback for every non-disk backend — including a
/// `Recording` wrapper over one; a `Recording` over a *disk* graph takes the
/// parallel path (see [`disk_scan_target`]).
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

pub struct DerivedEdgeStats {
    /// Edge type → count.
    pub counts: HashMap<String, usize>,
    /// Edge type → (source_types, target_types).
    pub endpoints: HashMap<String, (HashSet<String>, HashSet<String>)>,
}

/// Derive edge type counts + endpoint types from connectivity triples, instead
/// of a separate O(E) scan each.
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

/// Routing regression for the write-capture wrapper: a disk graph opened
/// `durable=True` (or with `cdc::enable`) is still a disk graph, and must
/// still take the Rayon scan. The counts are identical either way — the
/// forfeit is silent, which is why the routing seam is asserted directly.
#[cfg(test)]
mod capture_wrapped_routing_tests {
    use super::*;
    use crate::datatypes::{DataFrame, Value};
    use tempfile::TempDir;

    /// Two `Person`s pointing at one `City`, saved as a disk graph.
    fn disk_graph(dir: &TempDir) -> DirGraph {
        let people = DataFrame::from_cypher_rows(
            vec!["id".into(), "title".into()],
            vec![
                vec![Value::Int64(1), Value::String("p1".into())],
                vec![Value::Int64(2), Value::String("p2".into())],
            ],
        )
        .unwrap();
        let cities = DataFrame::from_cypher_rows(
            vec!["id".into(), "title".into()],
            vec![vec![Value::Int64(10), Value::String("Oslo".into())]],
        )
        .unwrap();
        let visits = DataFrame::from_cypher_rows(
            vec!["src".into(), "tgt".into()],
            vec![
                vec![Value::Int64(1), Value::Int64(10)],
                vec![Value::Int64(2), Value::Int64(10)],
            ],
        )
        .unwrap();

        let mut graph = DirGraph::new();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            people,
            "Person".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            cities,
            "City".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        crate::graph::mutation::maintain::add_connections(
            &mut graph,
            visits,
            "VISITED".to_string(),
            "Person".to_string(),
            "src".to_string(),
            "City".to_string(),
            "tgt".to_string(),
            None,
            None,
            None,
        )
        .unwrap();
        graph.enable_disk_mode().unwrap();
        graph.save_disk(dir.path().to_str().unwrap()).unwrap();
        graph
    }

    /// `ConnectivityTriple` is a plain serde record with no `PartialEq`.
    fn triples_of(graph: &DirGraph) -> Vec<(String, String, String, usize)> {
        compute_type_connectivity(graph)
            .into_iter()
            .map(|t| (t.src, t.conn, t.tgt, t.count))
            .collect()
    }

    #[test]
    fn a_capture_wrapped_disk_graph_still_routes_to_the_parallel_scan() {
        let dir = TempDir::new().unwrap();
        let mut graph = disk_graph(&dir);
        assert!(
            disk_scan_target(&graph.graph).is_some(),
            "fixture must be a disk graph before wrapping"
        );
        let bare = triples_of(&graph);

        graph.graph.wrap_for_durability();
        assert!(
            matches!(&graph.graph, GraphBackend::Recording(_)),
            "the wrap must have taken effect, or this test asserts nothing"
        );
        assert!(
            disk_scan_target(&graph.graph).is_some(),
            "a durability-wrapped disk graph must still reach compute_disk_parallel"
        );

        let wrapped = triples_of(&graph);
        assert_eq!(bare, wrapped, "both scans must agree on the triples");
        assert_eq!(
            wrapped,
            vec![(
                "Person".to_string(),
                "VISITED".to_string(),
                "City".to_string(),
                2
            )]
        );
    }
}
