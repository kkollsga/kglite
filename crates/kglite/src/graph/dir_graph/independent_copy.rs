//! Explicit independent-copy semantics for [`DirGraph`].
//!
//! Generic `Clone` preserves graph identity because it backs snapshots,
//! transactions, and copy-on-write views. User-requested copies need a
//! different contract: independent identity and independently mutable caches.

use super::{next_graph_id, DirGraph};
use std::sync::{Arc, RwLock};

fn copy_cache<T: Clone>(cache: &Arc<RwLock<T>>) -> Arc<RwLock<T>> {
    Arc::new(RwLock::new(
        cache
            .read()
            .expect("DirGraph cache RwLock poisoned")
            .clone(),
    ))
}

impl DirGraph {
    /// Copy this graph into an independent runtime lineage.
    ///
    /// Unlike [`Clone`], this mints a new process identity and gives every
    /// state-derived cache its own lock and value. Immutable backing resources
    /// remain shared through their existing copy-on-write ownership. This is
    /// the core primitive for binding-level explicit copy operations; snapshots
    /// and transactions must continue to use `Clone` so they preserve lineage.
    pub fn independent_copy(&self) -> Self {
        let mut copy = self.clone();
        copy.graph_id = next_graph_id();
        // A change stream is addressed by `(epoch, seq)`, so an independent
        // lineage needs an independent epoch: the copy's events describe
        // *its* writes, and a cursor from the original must be refused rather
        // than resolved against them. Capacity carries over; the ring does not
        // (the copy has published nothing).
        copy.cdc = self.cdc.as_ref().map(|handle| {
            let capacity = handle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .capacity();
            std::sync::Arc::new(std::sync::Mutex::new(crate::graph::cdc::CdcLog::new(
                capacity,
            )))
        });
        copy.wkt_cache = copy_cache(&self.wkt_cache);
        // The two edge-derived caches need nothing here: they are
        // `ForkPrivateCache`, so `self.clone()` above already gave the copy its
        // own empty one. Re-wrapping them was this method's half of the D2 R6
        // workaround; the hazard is now closed at the type level for every
        // clone, not just for the explicit-copy path.
        copy.property_ndv_cache = copy_cache(&self.property_ndv_cache);
        copy.graph.detach_independent_copy(&self.graph);
        copy.active_write_scope = None;
        copy.active_git_sha = None;
        copy.active_modified_by = None;
        copy.pending_constraint_violation = None;
        copy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// `independent_copy` must hand back a graph whose caches nothing else can
    /// write through.
    ///
    /// Split by mechanism since D2 Phase 3, because the two families now get
    /// there differently and it is worth saying which is which:
    ///
    /// - `wkt_cache` / `property_ndv_cache` are `Arc`-shared by ordinary
    ///   `Clone` **on purpose** (pure-function and version-tagged respectively —
    ///   see `caches::ForkPrivateCache`), so this method still has to re-wrap
    ///   them, and it deep-copies their contents.
    /// - the two edge-derived caches are `ForkPrivateCache`, so `Clone` already
    ///   gave the copy its own empty one. They arrive **cold**, not copied, and
    ///   that is the change: a warm copy was the old behaviour, an independent
    ///   one is the contract.
    #[test]
    fn independent_copy_mints_identity_and_owns_semantic_caches() {
        let mut graph = DirGraph::new();
        graph.version = 7;
        graph.active_write_scope = Some(HashSet::from(["Item".to_string()]));
        graph.active_git_sha = Some("abc".to_string());
        graph.active_modified_by = Some("test".to_string());
        *graph.edge_type_counts_cache.write().unwrap() = Some(std::sync::Arc::new(HashMap::from(
            [("LINKS".to_string(), 3usize)],
        )));

        let copy = graph.independent_copy();

        assert_ne!(copy.graph_id(), graph.graph_id());
        assert_eq!(copy.version(), graph.version());
        assert!(!Arc::ptr_eq(&copy.wkt_cache, &graph.wkt_cache));
        assert!(!Arc::ptr_eq(
            &copy.property_ndv_cache,
            &graph.property_ndv_cache
        ));

        // Cold, not copied — and writing through one cannot reach the other,
        // which is the property that matters and the one R6 broke.
        assert!(
            copy.edge_type_counts_cache.read().unwrap().is_none(),
            "a fork-private cache is reborn empty"
        );
        assert!(copy.type_connectivity_cache.read().unwrap().is_none());
        *copy.edge_type_counts_cache.write().unwrap() =
            Some(std::sync::Arc::new(HashMap::from([(
                "LINKS".to_string(),
                99usize,
            )])));
        assert_eq!(
            graph.edge_type_counts_cache.read().unwrap().as_ref(),
            Some(&std::sync::Arc::new(HashMap::from([(
                "LINKS".to_string(),
                3usize
            )]))),
            "the original must keep its own entry"
        );

        assert!(copy.active_write_scope.is_none());
        assert!(copy.active_git_sha.is_none());
        assert!(copy.active_modified_by.is_none());
    }
}
