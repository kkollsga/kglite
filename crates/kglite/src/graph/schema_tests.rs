//! Unit tests extracted from schema.rs to keep the production module focused.

use super::*;

#[cfg(test)]
mod type_id_index_tests {
    use super::*;
    use petgraph::graph::NodeIndex;

    fn integer_index() -> TypeIdIndex {
        let mut m = HashMap::new();
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

    /// Helper: create a DirGraph with N Person nodes and edges between consecutive pairs
    fn make_test_graph(num_nodes: usize, num_edges: bool) -> DirGraph {
        let mut g = DirGraph::new();
        for i in 0..num_nodes {
            let mut props = HashMap::new();
            props.insert("age".to_string(), Value::Int64(20 + i as i64));
            let node = NodeData::new(
                Value::UniqueId(i as u32),
                Value::String(format!("Person_{}", i)),
                "Person".to_string(),
                props,
                &mut g.interner,
            );
            let idx = g.graph.add_node(node);
            g.type_indices
                .entry_or_default("Person".to_string())
                .push(idx);
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

        // Verify all surviving nodes are present with correct data
        let mut titles: Vec<String> = Vec::new();
        for idx in g.graph.node_indices() {
            if let Some(node) = g.graph.node_weight(idx) {
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
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.set_property("city", new_val.clone(), &mut g.interner);
        }
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
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.remove_property("city");
        }
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
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.set_property("city", new_val.clone(), &mut g.interner);
        }
        g.update_property_indices_for_set("Person", n0, "city", Some(&old_val), &new_val);

        // Verify: old composite value gone, new one present
        let comp_map = g.composite_indices.get(&key).unwrap();
        let old_comp = CompositeValue(vec![Value::String("Oslo".to_string()), Value::Int64(30)]);
        let new_comp = CompositeValue(vec![Value::String("Bergen".to_string()), Value::Int64(30)]);
        assert!(!comp_map.contains_key(&old_comp) || comp_map.get(&old_comp).unwrap().is_empty());
        assert_eq!(comp_map.get(&new_comp).unwrap(), &vec![n0]);
    }

    #[test]
    fn test_no_update_when_no_index_exists() {
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
        // No index created — these should be no-ops without crash
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
        assert!(g.property_indices.is_empty());
    }

    // ─── Columnar storage tests ──────────────────────────────────────────

    #[test]
    fn test_enable_columnar_preserves_properties() {
        let mut g = make_test_graph(5, false);
        // Add metadata so columnar knows types
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();

        // Snapshot properties before
        let before: Vec<(Value, Value, i64)> = g
            .type_indices
            .get("Person")
            .unwrap()
            .iter()
            .map(|idx| {
                let n = g.graph.node_weight(idx).unwrap();
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
        assert!(g.is_columnar());

        // Verify properties match
        let after: Vec<(Value, Value, i64)> = g
            .type_indices
            .get("Person")
            .unwrap()
            .iter()
            .map(|idx| {
                let n = g.graph.node_weight(idx).unwrap();
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
    fn test_columnar_roundtrip_via_disable() {
        let mut g = make_test_graph(3, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();

        // Enable columnar, then disable back to Compact
        g.enable_columnar();
        assert!(g.is_columnar());
        g.disable_columnar();
        assert!(!g.is_columnar());

        // Verify properties still work
        let idx = g.type_indices.get("Person").unwrap().get(0).unwrap();
        let node = g.graph.node_weight(idx).unwrap();
        assert!(matches!(node.properties, PropertyStorage::Compact { .. }));
        assert!(node.get_property("age").is_some());
    }

    #[test]
    fn test_columnar_set_property() {
        let mut g = make_test_graph(2, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();
        g.enable_columnar();

        let idx = g.type_indices.get("Person").unwrap().get(0).unwrap();
        let node = g.graph.node_weight_mut(idx).unwrap();

        // Update existing property
        node.set_property("age", Value::Int64(99), &mut g.interner);
        assert_eq!(
            node.get_property("age").map(|c| c.into_owned()),
            Some(Value::Int64(99))
        );
    }

    #[test]
    fn test_columnar_property_count_and_keys() {
        let mut g = make_test_graph(2, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();
        g.enable_columnar();

        let idx = g.type_indices.get("Person").unwrap().get(0).unwrap();
        let node = g.graph.node_weight(idx).unwrap();

        assert_eq!(node.property_count(), 1); // just "age"
        let keys: Vec<&str> = node.property_keys(&g.interner).collect();
        assert_eq!(keys, vec!["age"]);
    }

    #[test]
    fn test_columnar_serialize_roundtrip() {
        let mut g = make_test_graph(3, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();
        g.enable_columnar();

        // Serialize (Columnar should produce same output as Compact)
        let serialized = {
            let _guard = SerdeSerializeGuard::new(&g.interner);
            crate::serde_codec::encode(&g.graph).unwrap()
        };

        // Deserialize into a new graph — will come back as Map
        let graph2: GraphBackend = {
            let _guard = SerdeDeserializeGuard::new(&mut g.interner);
            crate::serde_codec::decode(&serialized).unwrap()
        };
        let node0 = graph2.node_weight(NodeIndex::new(0)).unwrap();

        // Properties should survive the round-trip
        assert!(node0.get_property("age").is_some());
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
}
