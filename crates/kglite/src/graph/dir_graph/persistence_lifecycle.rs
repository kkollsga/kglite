//! Ending a persistence owner preserves graph data and runtime observation identity.
use super::DirGraph;

impl DirGraph {
    /// Whether this graph's capture layer still claims a write-ahead log owner.
    /// Bindings without that log must refuse writes until ownership is detached.
    pub fn owns_wal_capture(&self) -> bool {
        self.graph.is_wal_owner()
    }

    /// Prepare a detached snapshot with the same data and CDC stream, but independent
    /// plan identity and mutable caches. This performs no filesystem writes and does not end the
    /// original owner's authority; call `end_persistence_authority` only after
    /// preparation succeeds and the binding is ready to publish its new state.
    pub fn detached_persistence_snapshot(&self) -> Self {
        let mut snapshot = self.independent_data_copy();
        snapshot.graph.release_wal_ownership();
        if !snapshot.cdc_enabled() {
            snapshot.graph.unwrap_capture_if_unowned();
        }
        snapshot
    }

    /// End the disk writer lineage while retained snapshots keep their backing
    /// files alive. New mutations on an ended lineage detach to a private root.
    /// Callers must not hold a Python GIL or acquire a graph lock from this call.
    pub fn end_persistence_authority(&self) {
        if let Some(disk) = self.graph.as_disk() {
            disk.end_writer_authority();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::cdc::{self, CdcEnrichment};
    use crate::graph::session::{execute_mut, ExecuteOptions};
    use std::sync::Arc;

    #[test]
    fn detachment_preserves_data_and_cdc_identity_without_wal_ownership() {
        let mut graph = DirGraph::new();
        cdc::enable(&mut graph, Some(16), CdcEnrichment::Off).unwrap();
        graph.graph.wrap_for_durability();
        execute_mut(
            &mut graph,
            "CREATE (:N {id:1,late:'retained'})",
            &ExecuteOptions::new(&std::collections::HashMap::new()),
        )
        .unwrap();
        cdc::drain_at_commit(&mut graph);
        let detached = graph.detached_persistence_snapshot();
        assert_ne!(detached.graph_id, graph.graph_id);
        assert_eq!(detached.version, graph.version);
        assert!(Arc::ptr_eq(
            detached.cdc.as_ref().unwrap(),
            graph.cdc.as_ref().unwrap()
        ));
        assert!(graph.owns_wal_capture());
        assert!(!detached.owns_wal_capture());
        assert!(detached.graph.is_recording());
        let result = crate::graph::session::execute_read(
            &detached,
            "MATCH(n:N) RETURN n.late AS late",
            &ExecuteOptions::new(&std::collections::HashMap::new()),
        )
        .unwrap();
        assert_eq!(
            result.result.rows,
            vec![vec![crate::datatypes::Value::String("retained".into())]]
        );
        assert_eq!(cdc::read(&detached, 0, None, &[]).unwrap().len(), 1);
    }
    #[test]
    fn detached_and_original_never_share_a_baked_anchor_plan() {
        use crate::graph::languages::cypher::plan_cache;
        let _guard = plan_cache::TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        plan_cache::clear_for_tests();
        let params = std::collections::HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let mut original = DirGraph::new();
        let mut detached = original.detached_persistence_snapshot();
        execute_mut(&mut original, "CREATE (:T {id:5})", &opts).unwrap();
        execute_mut(
            &mut detached,
            "CREATE (b:T {id:77})-[:E]->(a:T {id:88})",
            &opts,
        )
        .unwrap();
        assert_eq!(original.version, detached.version);
        let query = "MATCH ({id:5})-[:E]->(x) RETURN count(x) AS c";
        let first = crate::graph::session::execute_read(&original, query, &opts).unwrap();
        let second = crate::graph::session::execute_read(&detached, query, &opts).unwrap();
        assert_eq!(
            first.result.rows,
            vec![vec![crate::datatypes::Value::Int64(0)]]
        );
        assert_eq!(
            second.result.rows,
            vec![vec![crate::datatypes::Value::Int64(0)]]
        );
    }

    #[test]
    fn core_session_cdc_is_shared_so_wrapper_requires_write_ownership() {
        let mut original = DirGraph::new();
        cdc::enable(&mut original, Some(16), CdcEnrichment::Off).unwrap();
        let original = Arc::new(original);
        let session = crate::graph::session::Session::from_arc(Arc::clone(&original));
        let mut working = session.write();
        execute_mut(
            &mut working,
            "CREATE (:N {id:1})",
            &ExecuteOptions::new(&std::collections::HashMap::new()),
        )
        .unwrap();
        cdc::drain_at_commit(&mut working);
        // Core's writer guard shares capture lineage. A Python independent
        // Session cannot expose this as source-graph CDC write authority.
        use crate::graph::storage::GraphRead;
        assert_eq!(original.graph.node_count(), 0);
        assert_eq!(cdc::read(&original, 0, None, &[]).unwrap().len(), 1);
    }
}
