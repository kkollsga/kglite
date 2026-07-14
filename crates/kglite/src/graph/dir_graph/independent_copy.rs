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
        copy.wkt_cache = copy_cache(&self.wkt_cache);
        copy.edge_type_counts_cache = copy_cache(&self.edge_type_counts_cache);
        copy.type_connectivity_cache = copy_cache(&self.type_connectivity_cache);
        copy.property_ndv_cache = copy_cache(&self.property_ndv_cache);
        copy.active_write_scope = None;
        copy.active_git_sha = None;
        copy.active_modified_by = None;
        copy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn independent_copy_mints_identity_and_owns_semantic_caches() {
        let mut graph = DirGraph::new();
        graph.version = 7;
        graph.active_write_scope = Some(HashSet::from(["Item".to_string()]));
        graph.active_git_sha = Some("abc".to_string());
        graph.active_modified_by = Some("test".to_string());
        *graph.edge_type_counts_cache.write().unwrap() =
            Some(HashMap::from([("LINKS".to_string(), 3)]));

        let copy = graph.independent_copy();

        assert_ne!(copy.graph_id(), graph.graph_id());
        assert_eq!(copy.version(), graph.version());
        assert!(!Arc::ptr_eq(&copy.wkt_cache, &graph.wkt_cache));
        assert!(!Arc::ptr_eq(
            &copy.edge_type_counts_cache,
            &graph.edge_type_counts_cache
        ));
        assert!(!Arc::ptr_eq(
            &copy.type_connectivity_cache,
            &graph.type_connectivity_cache
        ));
        assert!(!Arc::ptr_eq(
            &copy.property_ndv_cache,
            &graph.property_ndv_cache
        ));
        assert_eq!(
            *copy.edge_type_counts_cache.read().unwrap(),
            *graph.edge_type_counts_cache.read().unwrap()
        );
        *copy.edge_type_counts_cache.write().unwrap() = None;
        assert!(graph.edge_type_counts_cache.read().unwrap().is_some());
        assert!(copy.active_write_scope.is_none());
        assert!(copy.active_git_sha.is_none());
        assert!(copy.active_modified_by.is_none());
    }
}
