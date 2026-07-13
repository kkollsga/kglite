//! Lazy derived indexes for the default in-memory backend.

use super::{MemoryGraph, MemoryPeerCounts};
use crate::graph::schema::InternedKey;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

impl MemoryGraph {
    /// Drop derived edge counts after any mutation that can change edge type,
    /// identity, or endpoints.
    pub(crate) fn invalidate_peer_counts(&mut self) {
        if let Ok(mut cache) = self.peer_counts.write() {
            cache.clear();
        }
    }

    /// Fetch or build source/target counts for one relationship type.
    pub(crate) fn ensure_peer_counts(&self, conn_type: InternedKey) -> Arc<MemoryPeerCounts> {
        self.ensure_peer_counts_with_deadline(conn_type, None)
            .expect("peer-count build without a deadline cannot time out")
    }

    /// Deadline-aware form used by query execution. A completed cache lookup
    /// is constant-time; only the initial edge scan needs periodic checks.
    pub(crate) fn ensure_peer_counts_with_deadline(
        &self,
        conn_type: InternedKey,
        deadline: Option<Instant>,
    ) -> Result<Arc<MemoryPeerCounts>, String> {
        let key = conn_type.as_u64();
        if let Ok(cache) = self.peer_counts.read() {
            if let Some(counts) = cache.get(&key) {
                return Ok(Arc::clone(counts));
            }
        }

        let mut by_target = HashMap::new();
        let mut by_source = HashMap::new();
        for (edge_idx, edge) in self.inner.edge_references().enumerate() {
            if edge_idx.is_multiple_of(1 << 20) && deadline.is_some_and(|dl| Instant::now() > dl) {
                return Err("Query timed out".to_string());
            }
            if edge.weight().connection_type != conn_type {
                continue;
            }
            *by_target.entry(edge.target().index() as u32).or_insert(0) += 1;
            *by_source.entry(edge.source().index() as u32).or_insert(0) += 1;
        }
        let built = Arc::new(MemoryPeerCounts {
            by_target: Arc::new(by_target),
            by_source: Arc::new(by_source),
        });
        let mut cache = match self.peer_counts.write() {
            Ok(cache) => cache,
            Err(_) => return Ok(built),
        };
        Ok(Arc::clone(cache.entry(key).or_insert(built)))
    }
}
