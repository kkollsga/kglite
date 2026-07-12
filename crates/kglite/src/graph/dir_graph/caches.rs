//! Lazily-computed, mutation-invalidated caches and derived statistics on
//! [`DirGraph`]: edge-type counts, type connectivity, and per-`(type,
//! property)` distinct-value counts (NDV) for the planner's selectivity
//! estimator. Split out of `dir_graph/mod.rs` to stay under the god-file
//! ceiling; a child module, so it retains access to `DirGraph`'s private
//! fields.

use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead; // edge_endpoint_keys()
use std::collections::HashMap;

impl DirGraph {
    /// Compute edge counts grouped by connection type. Lazily cached.
    pub fn get_edge_type_counts(&self) -> HashMap<String, usize> {
        // Fast path: return cached result
        {
            let read = self.edge_type_counts_cache.read().unwrap();
            if let Some(ref cached) = *read {
                return cached.clone();
            }
        }
        // Slow path: compute O(E) and cache.
        // Uses edge_endpoint_keys() (mmap reads, zero heap per edge) instead of
        // edge_weights() (which materializes EdgeData → OOM on extreme-scale disk graphs).
        let mut counts: HashMap<InternedKey, usize> = HashMap::new();
        for (_src, _tgt, conn_key) in self.graph.edge_endpoint_keys() {
            *counts.entry(conn_key).or_insert(0) += 1;
        }
        // Resolve to strings
        let string_counts: HashMap<String, usize> = counts
            .into_iter()
            .map(|(k, v)| (self.interner.resolve(k).to_string(), v))
            .collect();
        let mut write = self.edge_type_counts_cache.write().unwrap();
        *write = Some(string_counts.clone());
        string_counts
    }

    /// Invalidate edge caches (call after edge mutations).
    pub(crate) fn invalidate_edge_type_counts_cache(&self) {
        *self.edge_type_counts_cache.write().unwrap() = None;
        *self.type_connectivity_cache.write().unwrap() = None;
    }

    /// Distinct-value count (NDV) for `(node_type, property)`, lazily computed
    /// and cached per graph `version`. The planner uses it to estimate
    /// non-indexed equality selectivity as `type_count / ndv` instead of a
    /// flat heuristic (so a boolean ≈ `count/2`, an enum ≈ `count/k`, a
    /// high-cardinality field ≈ `count/N`). Returns `None` when the type is
    /// absent or larger than `MAX_SCAN` (caller falls back to the heuristic);
    /// at that scale a real property index is the right tool and gives exact
    /// selectivity anyway. Plan-time read path only — never the write hot path.
    pub fn property_ndv(&self, node_type: &str, property: &str) -> Option<usize> {
        const MAX_SCAN: usize = 200_000;
        let nodes = self.type_indices.get(node_type)?;
        if nodes.is_empty() || nodes.len() > MAX_SCAN {
            return None;
        }
        let key = (node_type.to_string(), property.to_string());
        // Fast path: cache hit at the current graph version.
        {
            let read = self.property_ndv_cache.read().unwrap();
            if read.0 == self.version {
                if let Some(&ndv) = read.1.get(&key) {
                    return Some(ndv);
                }
            }
        }
        // Slow path: count distinct values across the type's nodes (O(type)).
        // Arena guard: get_node -> node_weight materializes on the disk
        // backend (protocol in disk/graph.rs); no-op on memory/mapped.
        let _arena_guard = self.graph.begin_query();
        let mut seen: std::collections::HashSet<Value> = std::collections::HashSet::new();
        for idx in nodes.iter() {
            if let Some(node) = self.get_node(idx) {
                if let Some(val) = node.get_property(property) {
                    seen.insert(val.into_owned());
                }
            }
        }
        let ndv = seen.len().max(1);
        let mut write = self.property_ndv_cache.write().unwrap();
        // Drop a stale-version map before inserting (auto-invalidation).
        if write.0 != self.version {
            write.1.clear();
            write.0 = self.version;
        }
        write.1.insert(key, ndv);
        Some(ndv)
    }

    /// Check if edge type count cache is populated (avoids O(E) scan).
    pub fn has_edge_type_counts_cache(&self) -> bool {
        self.edge_type_counts_cache.read().unwrap().is_some()
    }

    /// Check if type connectivity cache is populated.
    pub fn has_type_connectivity_cache(&self) -> bool {
        self.type_connectivity_cache.read().unwrap().is_some()
    }
}
