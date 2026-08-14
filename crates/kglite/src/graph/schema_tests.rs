//! Unit tests extracted from schema.rs to keep the production module focused.

use super::*;

#[cfg(test)]
mod type_id_index_tests {
    use super::*;
    use petgraph::graph::NodeIndex;

    fn integer_index() -> TypeIdIndex {
        let mut m = rustc_hash::FxHashMap::default();
        m.insert(1u32, NodeIndex::new(10));
        m.insert(42u32, NodeIndex::new(20));
        TypeIdIndex::Integer(m)
    }

    #[test]
    fn numeric_coercions_retained() {
        let idx = integer_index();
        // UniqueId / Int64 / Float all hit the integer key.
        assert_eq!(idx.get(&Value::UniqueId(42)), Some(NodeIndex::new(20)));
        assert_eq!(idx.get(&Value::Int64(42)), Some(NodeIndex::new(20)));
        assert_eq!(idx.get(&Value::Float64(42.0)), Some(NodeIndex::new(20)));
        // Non-integral float and out-of-range miss.
        assert_eq!(idx.get(&Value::Float64(42.5)), None);
        assert_eq!(idx.get(&Value::Int64(-1)), None);
    }

    #[test]
    fn string_no_longer_coerces_to_int() {
        // Regression lock (0.10.10): a `String` id must NOT be prefix-stripped
        // into the integer index. `{id:'a1'}` / `{id:'Q1'}` must NOT resolve to
        // `UniqueId(1)` — that was the wrong-node false-positive bug.
        let idx = integer_index();
        assert_eq!(idx.get(&Value::String("a1".into())), None);
        assert_eq!(idx.get(&Value::String("x1".into())), None);
        assert_eq!(idx.get(&Value::String("Q1".into())), None);
        assert_eq!(idx.get(&Value::String("1".into())), None);
    }
}

#[cfg(test)]
mod connection_type_compatibility_tests {
    use super::*;

    #[test]
    fn legacy_singular_endpoint_fields_remain_readable() {
        let info: ConnectionTypeInfo = serde_json::from_str(
            r#"{
                "source_type": "Person",
                "target_type": "Company",
                "property_types": {"since": "Int64"}
            }"#,
        )
        .unwrap();

        assert_eq!(info.source_types, HashSet::from(["Person".to_string()]));
        assert_eq!(info.target_types, HashSet::from(["Company".to_string()]));
        assert_eq!(
            info.property_types.get("since").map(String::as_str),
            Some("Int64")
        );
    }
}

#[cfg(test)]
mod maintenance_tests {
    use super::*;
    use crate::graph::storage::{GraphRead, GraphWrite};

    /// Helper: create a DirGraph with N Person nodes and edges between consecutive pairs.
    ///
    /// Built through **`add_nodes`**, the real ingest funnel, rather than by
    /// hand-assembling `NodeData` and pushing into `type_indices`. The hand-built
    /// form bypassed every construction path a user can reach, and since 0.16.0
    /// made construction columnar everywhere, that is the difference between a
    /// fixture whose nodes carry inline `id`/`title` values and one whose nodes
    /// carry the `Value::Null` sentinel with their identity in the type's
    /// `ColumnStore` — i.e. between a fixture that hides the sentinel class and
    /// one that exercises it. Tests here therefore read values through
    /// `node_view`, never off `node_weight` (codingest's 0.16.0 report, ask 4b).
    fn make_test_graph(num_nodes: usize, num_edges: bool) -> DirGraph {
        use crate::datatypes::DataFrame;
        use crate::graph::mutation::maintain::add_nodes;

        let mut g = DirGraph::new();
        if num_nodes > 0 {
            let rows: Vec<Vec<Value>> = (0..num_nodes)
                .map(|i| {
                    vec![
                        Value::UniqueId(i as u32),
                        Value::String(format!("Person_{i}")),
                        Value::Int64(20 + i as i64),
                    ]
                })
                .collect();
            let df = DataFrame::from_cypher_rows(
                vec!["id".to_string(), "title".to_string(), "age".to_string()],
                rows,
            )
            .expect("frame");
            add_nodes(
                &mut g,
                df,
                "Person".to_string(),
                "id".to_string(),
                Some("title".to_string()),
                None,
            )
            .expect("add_nodes");
        }
        if num_edges {
            for i in 0..(num_nodes.saturating_sub(1)) {
                let src = NodeIndex::new(i);
                let tgt = NodeIndex::new(i + 1);
                g.graph.add_edge(
                    src,
                    tgt,
                    EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut g.interner),
                );
            }
        }
        g
    }

    #[test]
    fn test_graph_info_clean() {
        let g = make_test_graph(5, true);
        let info = g.graph_info();
        assert_eq!(info.node_count, 5);
        assert_eq!(info.node_capacity, 5);
        assert_eq!(info.node_tombstones, 0);
        assert_eq!(info.edge_count, 4);
        assert_eq!(info.fragmentation_ratio, 0.0);
        assert_eq!(info.type_count, 1);
    }

    #[test]
    fn test_graph_info_after_deletion() {
        let mut g = make_test_graph(5, false);
        // Delete node 2 — leaves a tombstone
        g.graph.remove_node(NodeIndex::new(2));
        let info = g.graph_info();
        assert_eq!(info.node_count, 4);
        assert_eq!(info.node_capacity, 5); // Still 5 slots
        assert_eq!(info.node_tombstones, 1);
        assert!(info.fragmentation_ratio > 0.19 && info.fragmentation_ratio < 0.21);
    }

    #[test]
    fn test_graph_info_empty() {
        let g = DirGraph::new();
        let info = g.graph_info();
        assert_eq!(info.node_count, 0);
        assert_eq!(info.node_capacity, 0);
        assert_eq!(info.fragmentation_ratio, 0.0);
    }

    #[test]
    fn test_reindex_rebuilds_type_indices() {
        let mut g = make_test_graph(5, false);

        // Manually corrupt type_indices (simulate drift)
        g.type_indices.clear();
        assert!(g.type_indices.is_empty());

        g.reindex();

        // type_indices should be rebuilt
        assert_eq!(g.type_indices.len(), 1);
        assert_eq!(g.type_indices.get("Person").unwrap().len(), 5);
    }

    #[test]
    fn test_reindex_rebuilds_property_indices() {
        let mut g = make_test_graph(5, false);

        // Create a property index
        g.create_index("Person", "age");
        assert!(g.has_index("Person", "age"));

        // Manually corrupt the property index
        g.property_indices
            .get_mut(&("Person".to_string(), "age".to_string()))
            .unwrap()
            .clear();

        g.reindex();

        // Property index should be rebuilt with correct data
        let stats = g.get_index_stats("Person", "age").unwrap();
        assert_eq!(stats.unique_values, 5); // ages 20..24
        assert_eq!(stats.total_entries, 5);
    }

    #[test]
    fn test_reindex_rebuilds_composite_indices() {
        let mut g = make_test_graph(5, false);
        g.create_composite_index("Person", &["age"]);
        assert!(g.has_composite_index("Person", &["age".to_string()]));

        // Corrupt composite index
        g.composite_indices.values_mut().for_each(|v| v.clear());

        g.reindex();

        let stats = g
            .get_composite_index_stats("Person", &["age".to_string()])
            .unwrap();
        assert_eq!(stats.unique_values, 5);
    }

    #[test]
    fn test_reindex_clears_id_indices() {
        let mut g = make_test_graph(3, false);
        g.build_id_index("Person");
        assert!(g.id_indices.contains_key("Person"));

        g.reindex();

        // id_indices should be cleared (lazy rebuild on next access)
        assert!(g.id_indices.is_empty());
    }

    #[test]
    fn test_reindex_after_deletion() {
        let mut g = make_test_graph(5, false);
        // Delete node 2
        g.graph.remove_node(NodeIndex::new(2));
        // type_indices still has the stale entry
        assert_eq!(g.type_indices.get("Person").unwrap().len(), 5);

        g.reindex();

        // Now type_indices should reflect only 4 live nodes
        assert_eq!(g.type_indices.get("Person").unwrap().len(), 4);
        // And none of them should be index 2
        assert!(!g
            .type_indices
            .get("Person")
            .unwrap()
            .contains(&NodeIndex::new(2)));
    }

    #[test]
    fn test_vacuum_noop_when_clean() {
        let mut g = make_test_graph(5, true);
        let mapping = g.vacuum();
        assert!(mapping.is_empty()); // No remapping needed
        assert_eq!(g.graph.node_count(), 5);
        assert_eq!(g.graph_info().node_tombstones, 0);
    }

    /// A vacuum must not change *which backend* the graph has.
    ///
    /// It used to assign `GraphBackend::Memory(..)` unconditionally, which
    /// silently downgraded a mapped graph to heap storage — the user asked for
    /// mmap-backed columns and quietly stopped getting them after the first
    /// auto-vacuum.
    #[test]
    fn test_vacuum_preserves_the_mapped_backend() {
        use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
        let mut g = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
        for i in 0..5 {
            let data = NodeData::new(
                Value::Int64(i),
                Value::String(format!("n{i}")),
                "Person".to_string(),
                HashMap::new(),
                &mut g.interner,
            );
            g.graph.add_node(data);
        }
        g.graph.remove_node(NodeIndex::new(2));
        assert!(g.graph.is_mapped());

        let mapping = g.vacuum();

        assert_eq!(mapping.len(), 4, "the rebuild must actually have happened");
        assert!(g.graph.is_mapped(), "vacuum must not downgrade the backend");
        assert_eq!(g.graph.node_count(), 4);
        assert_eq!(g.graph_info().node_tombstones, 0);
    }

    /// The severe half of the same bug: a vacuum dropped the `Recording`
    /// wrapper, so a durable graph stopped write-ahead logging for the rest of
    /// the session — silently, with no error and no way for the caller to
    /// notice until a crash lost everything since the last checkpoint.
    #[test]
    fn test_vacuum_preserves_the_write_capture_wrapper() {
        use crate::graph::storage::recording::RecordingGraph;
        let mut g = make_test_graph(5, true);
        let inner = std::mem::replace(&mut g.graph, GraphBackend::new());
        g.graph = GraphBackend::Recording(Box::new(RecordingGraph::new(inner)));
        g.graph.remove_node(NodeIndex::new(2));

        let mapping = g.vacuum();
        assert_eq!(mapping.len(), 4, "the rebuild must actually have happened");
        assert!(
            matches!(g.graph, GraphBackend::Recording(_)),
            "vacuum must not drop the WAL capture wrapper"
        );

        // And the wrapper must still be capturing afterwards.
        let before = g.graph.recorded_ops_len();
        let data = NodeData::new(
            Value::Int64(99),
            Value::String("after".to_string()),
            "Person".to_string(),
            HashMap::new(),
            &mut g.interner,
        );
        g.graph.add_node(data);
        assert!(
            g.graph.recorded_ops_len() > before,
            "writes after a vacuum must still be captured"
        );
    }

    /// Disk keeps its data in frozen CSR mmap, not a `StableDiGraph`. A
    /// petgraph-style rebuild there would both lose the disk root and
    /// materialise the entire graph on the heap — the one thing the backend
    /// exists to avoid. Disk reclaims space by publishing a new generation.
    #[test]
    fn test_vacuum_is_a_noop_on_disk() {
        use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
        let dir = tempfile::tempdir().unwrap();
        let mut g = new_dir_graph_in_mode(StorageMode::Disk, Some(dir.path())).unwrap();
        assert!(g.graph.is_disk());

        assert!(g.vacuum().is_empty());
        assert!(g.graph.is_disk(), "vacuum must not convert disk to heap");
    }

    #[test]
    fn test_vacuum_compacts_after_deletion() {
        let mut g = make_test_graph(5, true);
        // Delete middle node (creates tombstone)
        g.graph.remove_node(NodeIndex::new(2));
        assert_eq!(g.graph.node_count(), 4);
        assert_eq!(g.graph_info().node_tombstones, 1);

        let mapping = g.vacuum();

        // After vacuum: no tombstones, indices are contiguous
        assert_eq!(g.graph.node_count(), 4);
        assert_eq!(g.graph_info().node_tombstones, 0);
        assert_eq!(g.graph_info().node_capacity, 4);

        // Mapping should have 4 entries (one for each surviving node)
        assert_eq!(mapping.len(), 4);
    }

    #[test]
    fn test_vacuum_preserves_node_data() {
        let mut g = make_test_graph(3, false);
        g.graph.remove_node(NodeIndex::new(1)); // Delete Person_1

        let mapping = g.vacuum();

        // Verify all surviving nodes are present with correct data.
        //
        // Read through `node_view`, not `node_weight`: an ingested node's
        // inline `title` is the `Value::Null` sentinel and its real title lives
        // in the type's `ColumnStore`. Off `node_weight` this loop matches no
        // `Value::String` at all and collects an empty vector — the failure
        // mode is a test that stops observing anything rather than one that
        // observes something wrong.
        let mut titles: Vec<String> = Vec::new();
        for idx in g.graph.node_indices() {
            if let Some(node) = g.graph.node_view(idx) {
                if let Value::String(s) = &*node.title() {
                    titles.push(s.clone());
                }
            }
        }
        titles.sort();
        assert_eq!(titles, vec!["Person_0", "Person_2"]);
        assert_eq!(mapping.len(), 2);
    }

    #[test]
    fn test_vacuum_preserves_edges() {
        let mut g = make_test_graph(4, true);
        // Edges: 0→1, 1→2, 2→3
        // Delete node 0 (and its edge to 1)
        g.graph.remove_node(NodeIndex::new(0));
        // Remaining edges should be 1→2, 2→3

        let _mapping = g.vacuum();

        assert_eq!(g.graph.edge_count(), 2);
        assert_eq!(g.graph.node_count(), 3);
    }

    /// `test_vacuum_preserves_edges` counts edges, which a rebuild that
    /// remaps endpoints *wrongly* still satisfies: reversing every `(src,
    /// tgt)` pair leaves the count untouched and the whole Rust suite green.
    /// Direction is the part of an edge a vacuum can silently corrupt, so
    /// assert the endpoints themselves, by node title so the assertion does
    /// not encode the compaction's index arithmetic.
    #[test]
    fn test_vacuum_preserves_edge_direction() {
        let mut g = make_test_graph(4, true); // 0→1, 1→2, 2→3
        g.graph.remove_node(NodeIndex::new(0)); // leaves 1→2, 2→3

        g.vacuum();

        // `node_view`, not `node_weight` — see `test_vacuum_preserves_node_data`.
        let title = |idx: NodeIndex| match &*g.graph.node_view(idx).unwrap().title() {
            Value::String(s) => s.clone(),
            other => panic!("unexpected title {other:?}"),
        };
        let mut pairs: Vec<(String, String)> = g
            .graph
            .edge_indices()
            .map(|e| {
                let (src, tgt) = g.graph.edge_endpoints(e).unwrap();
                (title(src), title(tgt))
            })
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("Person_1".to_string(), "Person_2".to_string()),
                ("Person_2".to_string(), "Person_3".to_string()),
            ]
        );
    }

    #[test]
    fn test_vacuum_rebuilds_type_indices() {
        let mut g = make_test_graph(5, false);
        g.graph.remove_node(NodeIndex::new(2));

        g.vacuum();

        // type_indices should point to valid, contiguous indices
        assert_eq!(g.type_indices.get("Person").unwrap().len(), 4);
        for idx in g.type_indices.get("Person").unwrap().iter() {
            assert!(g.graph.node_weight(idx).is_some());
        }
    }

    #[test]
    fn test_vacuum_rebuilds_property_indices() {
        let mut g = make_test_graph(5, false);
        g.create_index("Person", "age");
        g.graph.remove_node(NodeIndex::new(2));

        g.vacuum();

        // Property index should still exist with correct entries
        assert!(g.has_index("Person", "age"));
        let stats = g.get_index_stats("Person", "age").unwrap();
        assert_eq!(stats.total_entries, 4); // 5 - 1 deleted
    }

    #[test]
    fn test_vacuum_heavy_fragmentation() {
        let mut g = make_test_graph(100, false);
        // Delete every other node — 50% fragmentation
        for i in (0..100).step_by(2) {
            g.graph.remove_node(NodeIndex::new(i));
        }
        assert_eq!(g.graph.node_count(), 50);
        let info = g.graph_info();
        assert!(info.fragmentation_ratio > 0.49);

        let mapping = g.vacuum();

        assert_eq!(mapping.len(), 50);
        assert_eq!(g.graph.node_count(), 50);
        assert_eq!(g.graph_info().node_tombstones, 0);
        assert_eq!(g.graph_info().fragmentation_ratio, 0.0);
    }

    // ========================================================================
    // Incremental Index Update Tests
    // ========================================================================

    #[test]
    fn test_update_property_indices_for_add() {
        let mut g = DirGraph::new();
        // Add a node and create an index
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        g.create_index("Person", "city");

        // Add a second node and call the helper
        let mut props2 = HashMap::new();
        props2.insert("city".to_string(), Value::String("Bergen".to_string()));
        let n1 = g.graph.add_node(NodeData::new(
            Value::Int64(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            props2,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n1);
        g.update_property_indices_for_add("Person", n1);

        // Verify index was updated
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert_eq!(oslo.unwrap().len(), 1);
        let bergen = g.lookup_by_index("Person", "city", &Value::String("Bergen".to_string()));
        let bergen = bergen.unwrap();
        assert_eq!(bergen.len(), 1);
        assert_eq!(bergen[0], n1);
    }

    #[test]
    fn test_update_property_indices_for_set() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        g.create_index("Person", "city");

        // Simulate SET n.city = 'Bergen'
        let old_val = Value::String("Oslo".to_string());
        let new_val = Value::String("Bergen".to_string());
        // Actually change the property on the node
        let city_key = g.interner.get_or_intern("city");
        GraphWrite::set_node_property(&mut g.graph, n0, city_key, new_val.clone());
        g.update_property_indices_for_set("Person", n0, "city", Some(&old_val), &new_val);

        // Verify: Oslo bucket should be empty, Bergen should have the node
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert!(oslo.is_none() || oslo.unwrap().is_empty());
        let bergen = g.lookup_by_index("Person", "city", &Value::String("Bergen".to_string()));
        assert_eq!(bergen.unwrap(), vec![n0]);
    }

    #[test]
    fn test_update_property_indices_for_remove() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        g.create_index("Person", "city");

        // Simulate REMOVE n.city
        let old_val = Value::String("Oslo".to_string());
        let city_key = g.interner.get_or_intern("city");
        GraphWrite::remove_node_property(&mut g.graph, n0, city_key);
        g.update_property_indices_for_remove("Person", n0, "city", &old_val);

        // Verify: Oslo bucket should be empty
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert!(oslo.is_none() || oslo.unwrap().is_empty());
    }

    #[test]
    fn test_update_composite_index_on_property_change() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        props.insert("age".to_string(), Value::Int64(30));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        g.create_composite_index("Person", &["city", "age"]);

        // Verify initial state
        let key = (
            "Person".to_string(),
            vec!["city".to_string(), "age".to_string()],
        );
        assert!(g.composite_indices.get(&key).unwrap().len() == 1);

        // Change city to Bergen
        let old_val = Value::String("Oslo".to_string());
        let new_val = Value::String("Bergen".to_string());
        let city_key = g.interner.get_or_intern("city");
        GraphWrite::set_node_property(&mut g.graph, n0, city_key, new_val.clone());
        g.update_property_indices_for_set("Person", n0, "city", Some(&old_val), &new_val);

        // Verify: old composite value gone, new one present
        let comp_map = g.composite_indices.get(&key).unwrap();
        let old_comp = CompositeValue(vec![Value::String("Oslo".to_string()), Value::Int64(30)]);
        let new_comp = CompositeValue(vec![Value::String("Bergen".to_string()), Value::Int64(30)]);
        assert!(!comp_map.contains_key(&old_comp) || comp_map.get(&old_comp).unwrap().is_empty());
        assert_eq!(comp_map.get(&new_comp).unwrap(), &vec![n0]);
    }

    /// A type with no index costs **nothing** to maintain — not "no crash", no
    /// work at all.
    ///
    /// The three updaters used to run in full on such a type: a value read-back
    /// through `property_reader` (which resolves the alias and interns the
    /// resolved name), a `String` for the resolved field, a key-set `Vec`, and
    /// a hash probe per index family — all to edit maps that hold no key for
    /// the type. On a 100k-row `SET` that was ~23% of the statement, 1.8× the
    /// cell write it accompanied.
    ///
    /// The assertion is a call counter rather than a timing because the work is
    /// invisible to every other oracle: it clones no backend, journals nothing
    /// (`note_*` returns early with no index to name) and forks no schema map,
    /// so `BACKEND_CLONE_NODES`, `JOURNAL_NODE_PRE_IMAGES` and
    /// `SCHEMA_MAP_FORKS` all read identically whether it ran or not.
    #[test]
    fn test_no_update_when_no_index_exists() {
        use crate::graph::dir_graph::indexes::{
            index_maintenance_passes, reset_index_maintenance_passes,
        };

        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        // No index created — these must be no-ops, and must not even look.
        reset_index_maintenance_passes();
        g.update_property_indices_for_add("Person", n0);
        g.update_property_indices_for_set(
            "Person",
            n0,
            "city",
            Some(&Value::String("Oslo".to_string())),
            &Value::String("Bergen".to_string()),
        );
        g.update_property_indices_for_remove(
            "Person",
            n0,
            "city",
            &Value::String("Oslo".to_string()),
        );
        assert_eq!(
            index_maintenance_passes(),
            0,
            "a type with no index must skip incremental maintenance outright"
        );
        assert!(g.property_indices.is_empty());
    }

    /// The counter above is not vacuous, and the gate is keyed on the **type**,
    /// not on the graph: a type that *does* carry an index still pays full
    /// maintenance while an index-free type in the same graph pays none.
    ///
    /// Without this arm, "0 passes" would also be the reading for a gate that
    /// disabled index maintenance altogether — which is the way this
    /// optimisation breaks (silently, with a stale index that an indexed
    /// `MATCH` then reads as truth).
    #[test]
    fn index_maintenance_gate_is_per_type() {
        use crate::graph::dir_graph::indexes::{
            index_maintenance_passes, reset_index_maintenance_passes,
        };

        let mut g = DirGraph::new();
        let mut mk = |id: i64, city: &str, node_type: &str| {
            let mut props = HashMap::new();
            props.insert("city".to_string(), Value::String(city.to_string()));
            let idx = g.graph.add_node(NodeData::new(
                Value::Int64(id),
                Value::String(format!("n{id}")),
                node_type.to_string(),
                props,
                &mut g.interner,
            ));
            g.type_indices
                .entry_or_default(node_type.to_string())
                .push(idx);
            idx
        };
        let person = mk(1, "Oslo", "Person");
        let ghost = mk(2, "Oslo", "Ghost");
        g.create_index("Person", "city");

        reset_index_maintenance_passes();
        let old = Value::String("Oslo".to_string());
        let new = Value::String("Bergen".to_string());
        let city = g.interner.get_or_intern("city");
        GraphWrite::set_node_property(&mut g.graph, person, city, new.clone());
        g.update_property_indices_for_set("Person", person, "city", Some(&old), &new);
        assert_eq!(
            index_maintenance_passes(),
            1,
            "an indexed type must still run maintenance"
        );

        // …and the index really moved, which is what the pass was for.
        assert_eq!(
            g.lookup_by_index("Person", "city", &new),
            Some(vec![person])
        );

        // The same write on the index-free type in the same graph: no pass.
        reset_index_maintenance_passes();
        GraphWrite::set_node_property(&mut g.graph, ghost, city, new.clone());
        g.update_property_indices_for_set("Ghost", ghost, "city", Some(&old), &new);
        assert_eq!(
            index_maintenance_passes(),
            0,
            "an index-free type must skip maintenance even when the graph has \
             indexes on other types"
        );
    }

    // ─── Columnar storage tests ──────────────────────────────────────────

    #[test]
    fn test_enable_columnar_preserves_properties() {
        let mut g = make_test_graph(5, false);
        // Add metadata so columnar knows types
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata_mut()
            .insert("Person".to_string(), meta);
        g.rebuild_type_schemas();

        // Snapshot properties before
        let before: Vec<(Value, Value, i64)> = g
            .type_indices
            .get("Person")
            .unwrap()
            .iter()
            .map(|idx| {
                let n = g.graph.node_view(idx).unwrap();
                let age = n
                    .get_property("age")
                    .map(|c| match c.as_ref() {
                        Value::Int64(v) => *v,
                        _ => panic!("expected Int64"),
                    })
                    .unwrap();
                (n.id().into_owned(), n.title().into_owned(), age)
            })
            .collect();

        g.enable_columnar();
        assert!(g.column_store_count() > 0);

        // Verify properties match
        let after: Vec<(Value, Value, i64)> = g
            .type_indices
            .get("Person")
            .unwrap()
            .iter()
            .map(|idx| {
                let n = g.graph.node_view(idx).unwrap();
                let age = n
                    .get_property("age")
                    .map(|c| match c.as_ref() {
                        Value::Int64(v) => *v,
                        _ => panic!("expected Int64"),
                    })
                    .unwrap();
                (n.id().into_owned(), n.title().into_owned(), age)
            })
            .collect();

        assert_eq!(before, after);
    }

    #[test]
    fn test_columnar_set_property() {
        let mut g = make_test_graph(2, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata_mut()
            .insert("Person".to_string(), meta);
        g.rebuild_type_schemas();
        g.enable_columnar();

        let idx = g.type_indices.get("Person").unwrap().get(0).unwrap();

        // Update existing property — through the backend, which owns the store.
        let age_key = g.interner.get_or_intern("age");
        GraphWrite::set_node_property(&mut g.graph, idx, age_key, Value::Int64(99));
        assert_eq!(
            g.graph
                .node_view(idx)
                .unwrap()
                .get_property("age")
                .map(|c| c.into_owned()),
            Some(Value::Int64(99))
        );
    }

    #[test]
    fn test_columnar_property_count_and_keys() {
        let mut g = make_test_graph(2, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata_mut()
            .insert("Person".to_string(), meta);
        g.rebuild_type_schemas();
        g.enable_columnar();

        let idx = g.type_indices.get("Person").unwrap().get(0).unwrap();
        let node = g.graph.node_view(idx).unwrap();

        assert_eq!(node.property_count(), 1); // just "age"
        let keys: Vec<&str> = node.property_keys(&g.interner);
        assert_eq!(keys, vec!["age"]);
    }

    /// A columnar graph round-trips its properties through the **`.kgl` save
    /// path**, which is the only production serializer that meets a columnar
    /// node.
    ///
    /// Replaces `test_columnar_serialize_roundtrip`, which serialized the
    /// backend directly and expected properties in the topology stream. D1
    /// Phase 3 made that impossible: the store belongs to the backend, and
    /// `PropertyStorage` — which is what serde sees — carries only a row id.
    /// `write_kgl_to` therefore writes columns in their own sections and sets
    /// `StripPropertiesGuard` while serializing topology; the `Serialize` impl
    /// `debug_assert!`s that guard is set, so a new save path that forgets it
    /// fails loudly instead of writing empty property maps.
    #[test]
    fn columnar_properties_round_trip_through_the_kgl_save_path() {
        let mut g = make_test_graph(3, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata_mut()
            .insert("Person".to_string(), meta);
        g.rebuild_type_schemas();
        g.enable_columnar();
        assert!(
            g.column_store_count() > 0,
            "fixture must be columnar, or this is vacuous"
        );

        let mut buf: Vec<u8> = Vec::new();
        crate::graph::io::file::write_kgl_to(&g, &mut buf).unwrap();
        let loaded = crate::graph::io::file::load_kgl_bytes(&buf).unwrap();

        let node0 = loaded.graph.node_view(NodeIndex::new(0)).unwrap();
        assert!(
            node0.get_property("age").is_some(),
            "columnar properties must survive the save path"
        );
        assert_eq!(
            node0.get_property("age").map(|c| c.into_owned()),
            g.graph
                .node_view(NodeIndex::new(0))
                .unwrap()
                .get_property("age")
                .map(|c| c.into_owned()),
            "and must come back with the same value"
        );
    }

    /// Vacuum a **columnar** graph — the combination the `test_vacuum_*` family
    /// above never reaches, because every one of them builds row storage.
    ///
    /// Three things this release changed meet here and nothing else pins their
    /// junction:
    ///
    /// * `GraphBackend::replace_heap_graph` has to *carry* the per-type column
    ///   stores across the swap. Dropping them leaves every node holding a
    ///   `PropertyStorage::Columnar(row_id)` with no store behind it, and
    ///   `NodeView` reads a storeless columnar node as an **empty property
    ///   set** — silently, with no error anywhere.
    /// * the compaction reassigns node indices, so `type_indices` and the row
    ///   ids must still agree afterwards.
    ///
    /// Asserted on the reserved `__id__`/`__title__` columns as well as an
    /// ordinary property, because a consolidated node stores `Null` inline and
    /// gets its identity back from the store — the failure mode is nodes that
    /// survive the vacuum with the right *count* and no identity.
    #[test]
    fn test_vacuum_preserves_columnar_properties_and_identity() {
        let mut g = make_test_graph(6, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata_mut()
            .insert("Person".to_string(), meta);
        g.rebuild_type_schemas();
        g.enable_columnar();
        assert!(
            g.column_store_count() > 0,
            "fixture must be columnar, or this test is vacuous"
        );

        // Tombstone two nodes so the vacuum has something to compact.
        g.graph.remove_node(NodeIndex::new(1));
        g.graph.remove_node(NodeIndex::new(4));

        let expected: Vec<(Value, Value, Value)> = [0usize, 2, 3, 5]
            .iter()
            .map(|i| {
                (
                    Value::UniqueId(*i as u32),
                    Value::String(format!("Person_{i}")),
                    Value::Int64(20 + *i as i64),
                )
            })
            .collect();

        // Reads through `type_indices`, because that is the route every
        // per-type sweep takes. Stale entries are tolerated (a raw
        // `remove_node` leaves them until the vacuum cleans them) but a *live*
        // node that reads wrong is not.
        let observe = |g: &DirGraph| -> Vec<(Value, Value, Value)> {
            let mut seen: Vec<(Value, Value, Value)> = g
                .type_indices
                .get("Person")
                .expect("Person bucket")
                .iter()
                .filter_map(|idx| {
                    let node = g.graph.node_view(idx)?;
                    Some((
                        node.id().into_owned(),
                        node.title().into_owned(),
                        node.get_property("age")
                            .map(|c| c.into_owned())
                            .unwrap_or(Value::Null),
                    ))
                })
                .collect();
            seen.sort_by(|a, b| format!("{:?}", a.1).cmp(&format!("{:?}", b.1)));
            seen
        };

        assert_eq!(
            observe(&g),
            expected,
            "precondition: readable before vacuum"
        );

        g.vacuum();

        assert!(
            g.column_store_count() > 0,
            "the vacuum must carry the column stores across the heap swap; a \
             storeless columnar node reads as an empty property set, silently"
        );
        assert_eq!(g.graph.node_count(), 4);
        assert_eq!(
            observe(&g),
            expected,
            "every surviving node must keep its id, title and properties across \
             the compaction's index reassignment"
        );
    }
}

#[cfg(test)]
mod embedding_store_tests {
    use super::*;

    #[test]
    fn text_hash_is_deterministic_and_distinguishing() {
        assert_eq!(
            EmbeddingStore::text_hash("hello"),
            EmbeddingStore::text_hash("hello"),
            "same text must hash identically (cross-process stable)"
        );
        assert_ne!(
            EmbeddingStore::text_hash("hello"),
            EmbeddingStore::text_hash("world"),
        );
        assert_ne!(
            EmbeddingStore::text_hash("hello"),
            EmbeddingStore::text_hash("Hello"),
        );
    }

    #[test]
    fn is_stale_covers_missing_changed_and_unhashed() {
        let mut store = EmbeddingStore::new(2);
        let h = EmbeddingStore::text_hash("v1");

        // No embedding yet → stale.
        assert!(store.is_stale(7, h));

        // Embedding present + matching hash → not stale.
        store.set_embedding(7, &[1.0, 2.0]);
        store.set_text_hash(7, h);
        assert!(!store.is_stale(7, h));

        // Text changed (different hash) → stale.
        assert!(store.is_stale(7, EmbeddingStore::text_hash("v2")));

        // Embedding present but no recorded hash (e.g. add_embeddings) → stale,
        // so mode='changed' will (re)hash it on the next pass.
        store.set_embedding(9, &[3.0, 4.0]);
        assert!(store.is_stale(9, EmbeddingStore::text_hash("anything")));
    }

    #[test]
    fn new_store_has_empty_provenance() {
        let store = EmbeddingStore::new(4);
        assert_eq!(store.model_id, None);
        assert!(store.text_hashes.is_empty());
    }

    #[test]
    fn serialized_embedding_shape_requires_exact_data_cardinality() {
        let mut store = EmbeddingStore::new(2);
        store.set_embedding(7, &[1.0, 2.0]);
        assert_eq!(store.validate_shape(), Ok(()));
        store.data.pop();
        assert!(store.validate_shape().is_err());
    }

    #[test]
    fn serialized_embedding_shape_requires_node_slot_bijection() {
        let mut store = EmbeddingStore::new(2);
        store.set_embedding(7, &[1.0, 2.0]);
        store.node_to_slot.insert(7, 1);
        assert!(store.validate_shape().is_err());
    }

    #[test]
    fn malformed_embedding_store_cannot_build_an_index() {
        use crate::graph::algorithms::hnsw::HnswParams;
        use crate::graph::algorithms::vector::DistanceMetric;

        let mut store = EmbeddingStore::new(2);
        store.set_embedding(7, &[1.0, 2.0]);
        store.data.pop();
        let before = format!("{store:?}");
        assert!(store
            .build_index(DistanceMetric::Cosine, HnswParams::default(), 1)
            .is_err());
        assert_eq!(format!("{store:?}"), before);
    }

    #[test]
    fn zero_dimension_store_is_valid_but_cannot_build_an_index() {
        use crate::graph::algorithms::hnsw::HnswParams;
        use crate::graph::algorithms::vector::DistanceMetric;

        let mut store = EmbeddingStore::new(0);
        store.set_embedding(7, &[]);
        assert_eq!(store.validate_shape(), Ok(()));
        let before = format!("{store:?}");
        let error = store
            .build_index(DistanceMetric::Cosine, HnswParams::default(), 1)
            .unwrap_err();
        assert!(error.contains("non-zero embedding dimension"));
        assert_eq!(format!("{store:?}"), before);
    }

    #[test]
    fn invalid_hnsw_parameters_do_not_mutate_embedding_store() {
        use crate::graph::algorithms::hnsw::HnswParams;
        use crate::graph::algorithms::vector::DistanceMetric;

        let mut valid = EmbeddingStore::new(2);
        valid.set_embedding(7, &[1.0, 2.0]);
        let invalid = [
            HnswParams {
                m: 1,
                ..HnswParams::default()
            },
            HnswParams {
                ef_construction: 0,
                ..HnswParams::default()
            },
            HnswParams {
                ef_search: 0,
                ..HnswParams::default()
            },
            HnswParams {
                m: usize::MAX,
                ..HnswParams::default()
            },
        ];
        for params in invalid {
            let mut store = valid.clone();
            let before = format!("{store:?}");
            assert!(store
                .build_index(DistanceMetric::Cosine, params, 1)
                .is_err());
            assert_eq!(format!("{store:?}"), before);
        }
    }
}

/// Incremental index maintenance must file a node under the value an index
/// *rebuild* would file it under — including when the index is registered under
/// an id/title **alias** spelling, where the value the matcher's scan consults
/// is the node's id/title and not the verbatim property.
///
/// Before the fix, `update_property_indices_for_add` / `_for_set` read the node
/// by the user-facing key (`NodeData::get_property`) while `create_index` /
/// `refresh_indexes_for_type` read through the alias-resolving `read_indexed`.
/// A Cypher `CREATE` into such a type filed the node under a bucket value the
/// scan can never produce: indexed `MATCH` returned rows the same query without
/// the index did not, and the two spellings of the same predicate (inline map
/// vs `WHERE`) disagreed with each other.
#[cfg(test)]
mod alias_index_maintenance_tests {
    use super::*;
    use crate::datatypes::DataFrame;
    use crate::graph::mutation::maintain::add_nodes;
    use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
    use petgraph::graph::NodeIndex;

    /// `Term` carries both alias kinds: `term_id` is its id field and
    /// `term_name` its title field, so `add_nodes` hoists both columns off the
    /// property map. `city` is an ordinary property, for the non-alias arms.
    fn aliased_graph() -> DirGraph {
        let rows: Vec<Vec<Value>> = (1..=3)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("term-{i}")),
                    Value::String("Oslo".to_string()),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(
            vec![
                "term_id".to_string(),
                "term_name".to_string(),
                "city".to_string(),
            ],
            rows,
        )
        .expect("frame");
        let mut graph = DirGraph::new();
        add_nodes(
            &mut graph,
            df,
            "Term".to_string(),
            "term_id".to_string(),
            Some("term_name".to_string()),
            None,
        )
        .expect("add_nodes");
        graph
    }

    fn run(graph: &mut DirGraph, statement: &str) {
        let params = HashMap::new();
        execute_mut(graph, statement, &ExecuteOptions::eager(&params))
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }

    fn count(graph: &DirGraph, query: &str) -> usize {
        let params = HashMap::new();
        let result = execute_read(graph, query, &ExecuteOptions::eager(&params))
            .unwrap_or_else(|error| panic!("{query}: {error}"));
        result.result.rows.len()
    }

    /// A graph carrying `index_on`, mutated by `mutations`.
    fn mutated(index_on: &str, mutations: &[&str]) -> DirGraph {
        let mut graph = aliased_graph();
        graph.create_index("Term", index_on);
        for statement in mutations {
            run(&mut graph, statement);
        }
        graph
    }

    /// The divergence oracle: ask `query` on the mutated graph with the index
    /// installed, then drop the index — same data, same query — and ask again.
    fn assert_index_agrees_with_scan(index_on: &str, mutations: &[&str], query: &str) {
        let mut graph = mutated(index_on, mutations);
        let with_index = count(&graph, query);
        graph.drop_index("Term", index_on).expect("drop");
        let scanned = count(&graph, query);
        assert_eq!(
            with_index, scanned,
            "indexed and unindexed answers diverge after {mutations:?} for: {query}"
        );
    }

    /// One property index's buckets in comparable form: (value, members) rows.
    type BucketRows = Vec<(Value, Vec<NodeIndex>)>;

    /// Every property-index bucket, in a comparable, order-normalised form.
    fn index_snapshot(graph: &DirGraph) -> Vec<(IndexKey, BucketRows)> {
        let mut snapshot: Vec<(IndexKey, BucketRows)> = graph
            .property_indices
            .iter()
            .map(|(key, index)| {
                let mut buckets: Vec<(Value, Vec<NodeIndex>)> = index
                    .iter()
                    .map(|(value, members)| {
                        let mut members = members.clone();
                        members.sort();
                        (value.clone(), members)
                    })
                    .collect();
                buckets.sort();
                (key.clone(), buckets)
            })
            .collect();
        snapshot.sort();
        snapshot
    }

    const CREATE_X: &str = "CREATE (:Term {term_id: 99, term_name: 'created-x'})";
    const CREATE_COLLIDING: &str = "CREATE (:Term {term_id: 98, term_name: 'term-1'})";

    #[test]
    fn cypher_create_on_a_title_aliased_index_agrees_with_the_scan() {
        // The phantom: the CREATE's verbatim `term_name` becomes a bucket the
        // scan can never produce, so the indexed MATCH returns a row the
        // unindexed one does not.
        assert_index_agrees_with_scan(
            "term_name",
            &[CREATE_X],
            "MATCH (t:Term {term_name: 'created-x'}) RETURN t",
        );
        assert_index_agrees_with_scan(
            "term_name",
            &[CREATE_X],
            "MATCH (t:Term) WHERE t.term_name = 'created-x' RETURN t",
        );
    }

    #[test]
    fn a_cypher_create_does_not_poison_an_existing_bucket() {
        // The poisoned bucket: the CREATE's verbatim value equals an existing
        // node's title, so the index reports two members for a value only one
        // node really carries.
        assert_index_agrees_with_scan(
            "term_name",
            &[CREATE_COLLIDING],
            "MATCH (t:Term {term_name: 'term-1'}) RETURN t",
        );
        assert_index_agrees_with_scan(
            "term_name",
            &[CREATE_COLLIDING],
            "MATCH (t:Term) WHERE t.term_name = 'term-1' RETURN t",
        );
    }

    #[test]
    fn the_two_spellings_of_one_predicate_agree_on_an_indexed_graph() {
        // Inline map vs WHERE: only the first consults the index, so a poisoned
        // index makes the same predicate answer two different ways.
        let graph = mutated("term_name", &[CREATE_X, CREATE_COLLIDING]);
        for value in ["created-x", "term-1", "term-2"] {
            let inline = count(
                &graph,
                &format!("MATCH (t:Term {{term_name: '{value}'}}) RETURN t"),
            );
            let filtered = count(
                &graph,
                &format!("MATCH (t:Term) WHERE t.term_name = '{value}' RETURN t"),
            );
            assert_eq!(
                inline, filtered,
                "inline-map and WHERE spellings disagree for term_name = '{value}'"
            );
        }
    }

    #[test]
    fn cypher_set_on_a_title_aliased_index_agrees_with_the_scan() {
        let set = "MATCH (t:Term {term_id: 1}) SET t.term_name = 'renamed'";
        for query in [
            "MATCH (t:Term {term_name: 'renamed'}) RETURN t",
            "MATCH (t:Term) WHERE t.term_name = 'renamed' RETURN t",
            "MATCH (t:Term {term_name: 'term-1'}) RETURN t",
            "MATCH (t:Term) WHERE t.term_name = 'term-1' RETURN t",
        ] {
            assert_index_agrees_with_scan("term_name", &[CREATE_X, set], query);
        }
    }

    #[test]
    fn cypher_set_on_the_title_keeps_an_alias_spelled_index_in_step() {
        // The index is registered under the alias spelling but holds titles, so
        // a write to `title` moves the node between its buckets.
        let set = "MATCH (t:Term {term_id: 1}) SET t.title = 'retitled'";
        for query in [
            "MATCH (t:Term {term_name: 'retitled'}) RETURN t",
            "MATCH (t:Term) WHERE t.term_name = 'retitled' RETURN t",
            "MATCH (t:Term {term_name: 'term-1'}) RETURN t",
        ] {
            assert_index_agrees_with_scan("term_name", &[set], query);
        }
    }

    #[test]
    fn cypher_set_on_the_name_synonym_moves_the_node_between_buckets() {
        // `name` is Cypher's own spelling of the title and the SET executor
        // writes the title through it, so — unlike an arbitrary alias spelling —
        // this write *does* move the field the index holds. A type that names
        // its title column `name` gets both readings of the same word.
        let rows: Vec<Vec<Value>> = (1..=3)
            .map(|i| vec![Value::Int64(i), Value::String(format!("person-{i}"))])
            .collect();
        let df =
            DataFrame::from_cypher_rows(vec!["person_id".to_string(), "name".to_string()], rows)
                .expect("frame");
        let mut graph = DirGraph::new();
        add_nodes(
            &mut graph,
            df,
            "Person".to_string(),
            "person_id".to_string(),
            Some("name".to_string()),
            None,
        )
        .expect("add_nodes");
        graph.create_index("Person", "name");
        run(
            &mut graph,
            "MATCH (p:Person {name: 'person-1'}) SET p.name = 'renamed'",
        );

        assert_eq!(
            count(&graph, "MATCH (p:Person {name: 'renamed'}) RETURN p"),
            1,
            "the index lost the node its SET renamed"
        );
        assert_eq!(
            count(&graph, "MATCH (p:Person {name: 'person-1'}) RETURN p"),
            0,
            "the index still answers with the pre-SET value"
        );
    }

    #[test]
    fn cypher_remove_on_a_title_aliased_index_agrees_with_the_scan() {
        let remove = "MATCH (t:Term {term_id: 1}) REMOVE t.term_name";
        for query in [
            "MATCH (t:Term {term_name: 'term-1'}) RETURN t",
            "MATCH (t:Term) WHERE t.term_name = 'term-1' RETURN t",
        ] {
            assert_index_agrees_with_scan("term_name", &[remove], query);
        }
    }

    #[test]
    fn incremental_maintenance_reproduces_a_rebuilt_index() {
        // The general contract, of which every case above is an instance:
        // whatever the writes were, the incrementally maintained index must
        // equal the one `refresh_indexes_for_type` builds from live state.
        let mutations = [
            CREATE_X,
            CREATE_COLLIDING,
            "MATCH (t:Term {term_id: 1}) SET t.term_name = 'renamed'",
            "MATCH (t:Term {term_id: 2}) SET t.title = 'retitled'",
            "MATCH (t:Term {term_id: 3}) SET t.city = 'Bergen'",
            "MATCH (t:Term {term_id: 99}) REMOVE t.city",
        ];
        let mut graph = aliased_graph();
        graph.create_index("Term", "term_name");
        graph.create_index("Term", "term_id");
        graph.create_index("Term", "city");
        for statement in mutations {
            run(&mut graph, statement);
        }

        let incremental = index_snapshot(&graph);
        graph.refresh_indexes_for_type("Term");
        assert_eq!(
            incremental,
            index_snapshot(&graph),
            "incremental index maintenance diverged from a rebuild"
        );
    }

    #[test]
    fn a_created_node_joins_a_doubly_indexed_property_once() {
        // A property carrying both a hash and a range index appears twice in
        // the key iteration; the node must still land in each bucket once, or
        // an indexed MATCH returns it twice.
        let mut graph = aliased_graph();
        graph.create_index("Term", "city");
        graph.create_range_index("Term", "city");
        run(&mut graph, "CREATE (:Term {term_id: 99, city: 'Bergen'})");

        let bucket = graph
            .lookup_by_index("Term", "city", &Value::String("Bergen".to_string()))
            .expect("bucket");
        assert_eq!(bucket.len(), 1, "hash-index bucket holds the node twice");
        let range = graph
            .lookup_range(
                "Term",
                "city",
                std::ops::Bound::Included(&Value::String("Bergen".to_string())),
                std::ops::Bound::Included(&Value::String("Bergen".to_string())),
            )
            .expect("range bucket");
        assert_eq!(range.len(), 1, "range-index bucket holds the node twice");
        assert_eq!(
            count(&graph, "MATCH (t:Term {city: 'Bergen'}) RETURN t"),
            1,
            "indexed MATCH returned the node more than once"
        );
    }

    /// Every node the index still points at, across all buckets.
    fn indexed_members(graph: &DirGraph, property: &str) -> Vec<NodeIndex> {
        graph
            .property_indices
            .get(&("Term".to_string(), property.to_string()))
            .expect("index")
            .iter()
            .flat_map(|(_, members)| members.iter().copied())
            .collect()
    }

    #[test]
    fn deleting_every_node_empties_the_buckets() {
        // The add/remove symmetry: whichever bucket a write filed a node under,
        // the delete has to reclaim it. Deletion evicts by *membership*, so it
        // is value-agnostic — this pins that it stays that way.
        let mut graph = mutated("term_name", &[CREATE_X]);
        run(&mut graph, "MATCH (t:Term) DELETE t");

        assert!(
            indexed_members(&graph, "term_name").is_empty(),
            "deleted nodes left entries behind: {:?}",
            indexed_members(&graph, "term_name")
        );
        assert_eq!(count(&graph, "MATCH (t:Term) RETURN t"), 0);
    }

    #[test]
    fn deleting_one_node_leaves_the_survivors_indexed() {
        let mut graph = mutated("term_name", &[CREATE_X]);
        run(&mut graph, "MATCH (t:Term {term_id: 1}) DELETE t");

        let members = indexed_members(&graph, "term_name");
        assert!(
            members.iter().all(|idx| graph.get_node(*idx).is_some()),
            "the index points at a deleted node: {members:?}"
        );
        assert_index_agrees_with_scan(
            "term_name",
            &[CREATE_X, "MATCH (t:Term {term_id: 1}) DELETE t"],
            "MATCH (t:Term {term_name: 'term-1'}) RETURN t",
        );
    }
}

#[cfg(test)]
mod soft_alias_names_tests {
    use super::*;

    /// [`SOFT_ALIAS_NAMES`] and [`soft_alias_fallback`] are two views of one
    /// fact, and a projection that completes from the *list* while resolution
    /// reads the *match* would silently stop recovering a name that fell out of
    /// sync. Both directions are checked: every listed name classifies, and
    /// nothing outside the list does (over the property names an engine
    /// actually sees — the identity fields, and the structural names' near
    /// misses).
    #[test]
    fn soft_alias_names_match_the_classifier() {
        for name in SOFT_ALIAS_NAMES {
            assert!(
                soft_alias_fallback(name).is_some(),
                "{name} is listed as a soft alias but does not classify as one"
            );
        }
        for name in [
            "id",
            "title",
            "names",
            "Name",
            "labels",
            "nodetype",
            "node_types",
            "",
        ] {
            assert!(
                soft_alias_fallback(name).is_none(),
                "{name} classifies as a soft alias but is not listed in SOFT_ALIAS_NAMES"
            );
        }
    }
}
