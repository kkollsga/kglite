//! DirGraph regression tests extracted from mod.rs.

use super::*;

#[cfg(test)]
mod multi_label_tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::schema::NodeData;
    use crate::graph::storage::GraphWrite;

    fn add_node(graph: &mut DirGraph, id: &str, node_type: &str) -> NodeIndex {
        let nd = NodeData::new(
            Value::String(id.to_string()),
            Value::String(id.to_string()),
            node_type.to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = GraphWrite::add_node(&mut graph.graph, nd);
        graph
            .type_indices
            .entry_or_default(node_type.to_string())
            .push(idx);
        idx
    }

    #[test]
    fn add_node_label_idempotent_and_no_op_on_primary() {
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");
        let person = g.interner.get_or_intern("Person");

        assert!(g.add_node_label(idx, reviewer));
        assert!(g.has_secondary_labels);
        assert_eq!(g.secondary_label_index[&reviewer], vec![idx]);

        // Idempotent — second add is a no-op.
        assert!(!g.add_node_label(idx, reviewer));
        assert_eq!(g.secondary_label_index[&reviewer], vec![idx]);

        // Primary type is a no-op too.
        assert!(!g.add_node_label(idx, person));

        let labels = g.node_labels(idx);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], person);
        assert_eq!(labels[1], reviewer);
    }

    #[test]
    fn bucket_invariant_survives_arbitrary_sequences() {
        // Sorted + deduped must hold through shuffled adds, interleaved
        // removes, and re-adds — the invariant every binary-search consumer
        // (node_has_label, secondary_labels, the fused paths) relies on.
        let mut g = DirGraph::new();
        let idxs: Vec<NodeIndex> = (0..20)
            .map(|i| add_node(&mut g, &format!("n{i}"), "Person"))
            .collect();
        let vip = g.interner.get_or_intern("Vip");
        for idx in idxs.iter().rev() {
            assert!(g.add_node_label(*idx, vip));
        }
        let sorted = |b: &[NodeIndex]| b.windows(2).all(|w| w[0] < w[1]);
        assert!(sorted(&g.secondary_label_index[&vip]));
        assert_eq!(g.secondary_label_index[&vip].len(), 20);

        for idx in idxs.iter().step_by(3) {
            assert!(g.remove_node_label(*idx, vip).unwrap());
        }
        assert!(sorted(&g.secondary_label_index[&vip]));
        for idx in idxs.iter().step_by(3) {
            assert!(g.add_node_label(*idx, vip));
        }
        assert!(sorted(&g.secondary_label_index[&vip]));
        assert_eq!(g.secondary_label_index[&vip].len(), 20);

        // Idempotent re-adds change nothing.
        for idx in &idxs {
            assert!(!g.add_node_label(*idx, vip));
        }
        assert_eq!(g.secondary_label_index[&vip].len(), 20);
        for idx in &idxs {
            assert!(g.node_has_label(*idx, vip));
        }
    }

    #[test]
    fn create_index_refused_on_secondary_only_label() {
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");
        g.add_node_label(idx, reviewer);

        // Secondary-only label: the index would never be consulted — refuse.
        let err = g
            .create_property_index_routed("Reviewer", "name")
            .unwrap_err();
        assert!(err.contains("secondary label"), "got: {err}");
        assert!(g.reject_secondary_only_index_type("Reviewer").is_err());

        // Primary type stays indexable, secondary label or not.
        assert!(g.create_property_index_routed("Person", "name").is_ok());

        // A label unknown to the graph stays allowed (pre-declaration).
        assert!(g.reject_secondary_only_index_type("Unknown").is_ok());

        // A label that is BOTH a primary type and a secondary label serves
        // the primary side of the union — allowed.
        let other = add_node(&mut g, "n2", "Reviewer");
        let _ = other;
        assert!(g.reject_secondary_only_index_type("Reviewer").is_ok());
    }

    #[test]
    fn remove_node_label_errors_on_primary() {
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let person = g.interner.get_or_intern("Person");

        let err = g.remove_node_label(idx, person).unwrap_err();
        assert!(err.contains("primary label"));
    }

    #[test]
    fn remove_node_label_clears_index_when_last_node_drops_it() {
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person");
        let b = add_node(&mut g, "b", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");

        g.add_node_label(a, reviewer);
        g.add_node_label(b, reviewer);
        assert_eq!(g.secondary_label_index[&reviewer].len(), 2);

        assert!(g.remove_node_label(a, reviewer).unwrap());
        assert_eq!(g.secondary_label_index[&reviewer], vec![b]);
        assert!(g.has_secondary_labels);

        assert!(g.remove_node_label(b, reviewer).unwrap());
        assert!(!g.secondary_label_index.contains_key(&reviewer));
        // No labels left anywhere, fast-skip resets.
        assert!(!g.has_secondary_labels);
    }

    #[test]
    fn rebuild_does_not_clobber_secondary_index() {
        // After 0.10.5's perf fix, NodeData no longer carries
        // extra_labels — `secondary_label_index` is the canonical
        // store. `rebuild_type_indices` rebuilds only type_indices
        // and leaves the secondary index intact (it's repopulated by
        // the load path via the disk sidecar / .kgl section).
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");
        g.add_node_label(idx, reviewer);

        let before = g.secondary_label_index.clone();
        let before_flag = g.has_secondary_labels;

        g.rebuild_type_indices();

        // Secondary index is untouched.
        assert_eq!(g.secondary_label_index, before);
        assert_eq!(g.has_secondary_labels, before_flag);
        // Primary type_indices is rebuilt correctly.
        assert_eq!(
            g.type_indices.get("Person").map(|s| s.iter().collect()),
            Some(vec![idx])
        );
    }

    #[test]
    fn dir_graph_node_labels_returns_primary_plus_extras() {
        // The canonical path for "all labels of node X" is
        // `DirGraph::node_labels` (which scans `secondary_label_index`).
        // Backend trait `node_labels_of` returns only the primary
        // type and is no longer the authoritative source.
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");
        let person = g.interner.get_or_intern("Person");
        g.add_node_label(idx, reviewer);

        let labels = g.node_labels(idx);
        assert_eq!(labels, vec![person, reviewer]);
    }

    #[test]
    fn nodes_with_label_single_label_fast_path() {
        // With no secondary labels anywhere, nodes_with_label must
        // return exactly type_indices[label] — the byte-identical
        // result every primary-only call site produced pre-multi-label.
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person");
        let b = add_node(&mut g, "b", "Person");
        add_node(&mut g, "w", "Widget");

        assert!(!g.has_secondary_labels);
        assert_eq!(g.nodes_with_label("Person"), vec![a, b]);
        assert_eq!(g.nodes_with_label("Widget").len(), 1);
        assert!(g.nodes_with_label("Absent").is_empty());
    }

    #[test]
    fn nodes_with_label_unions_primary_and_secondary() {
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person"); // primary Person, + VIP
        let b = add_node(&mut g, "b", "Person"); // primary Person only
        let w = add_node(&mut g, "w", "Widget"); // primary Widget, + VIP
        let vip = g.interner.get_or_intern("VIP");
        g.add_node_label(a, vip);
        g.add_node_label(w, vip);

        // Primary lookups still include their primary-typed nodes.
        let persons = g.nodes_with_label("Person");
        assert_eq!(persons, vec![a, b]);

        // :VIP is a secondary-only label — union pulls from both buckets.
        let mut vips = g.nodes_with_label("VIP");
        vips.sort();
        let mut expected = vec![a, w];
        expected.sort();
        assert_eq!(vips, expected);
    }

    #[test]
    fn node_has_label_primary_secondary_and_absent() {
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person");
        let person = g.interner.get_or_intern("Person");
        let vip = g.interner.get_or_intern("VIP");
        let ghost = g.interner.get_or_intern("Ghost");
        g.add_node_label(a, vip);

        assert!(g.node_has_label(a, person)); // primary
        assert!(g.node_has_label(a, vip)); // secondary
        assert!(!g.node_has_label(a, ghost)); // absent
    }

    #[test]
    fn detach_delete_evicts_secondary_label_index() {
        use std::collections::HashSet;
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person");
        let b = add_node(&mut g, "b", "Person");
        let vip = g.interner.get_or_intern("VIP");
        g.add_node_label(a, vip);
        g.add_node_label(b, vip);
        assert_eq!(g.secondary_label_index[&vip].len(), 2);

        let to_del: HashSet<NodeIndex> = [a].into_iter().collect();
        crate::graph::mutation::maintain::detach_delete_nodes(&mut g, &to_del);

        // `a` evicted from the secondary index; `b` survives. Without the
        // eviction the StableDiGraph would keep `a` live in the bucket and
        // `nodes_with_label` / counts would over-report.
        assert_eq!(g.secondary_label_index.get(&vip).map(|v| v.len()), Some(1));
        assert!(g.has_secondary_labels);
        assert_eq!(g.nodes_with_label("VIP"), vec![b]);
    }
}

#[cfg(test)]
mod bulk_index_freshness_tests {
    use super::*;
    use crate::datatypes::values::{DataFrame, Value};
    use crate::graph::mutation::maintain::add_nodes;

    fn people(rows: Vec<(&str, &str)>) -> DataFrame {
        DataFrame::from_cypher_rows(
            vec!["id".to_string(), "city".to_string()],
            rows.into_iter()
                .map(|(id, city)| {
                    vec![
                        Value::String(id.to_string()),
                        Value::String(city.to_string()),
                    ]
                })
                .collect(),
        )
        .expect("dataframe")
    }

    #[test]
    fn add_nodes_keeps_property_index_fresh() {
        let mut g = DirGraph::new();
        add_nodes(
            &mut g,
            people(vec![("p1", "Oslo")]),
            "Person".to_string(),
            "id".to_string(),
            None,
            None,
        )
        .expect("first load");

        assert_eq!(g.create_index("Person", "city"), 1);

        add_nodes(
            &mut g,
            people(vec![("p2", "Oslo"), ("p3", "Bergen")]),
            "Person".to_string(),
            "id".to_string(),
            None,
            None,
        )
        .expect("second load");

        let oslo = g
            .lookup_by_index("Person", "city", &Value::String("Oslo".to_string()))
            .unwrap_or_default();
        assert_eq!(oslo.len(), 2, "bulk load left the property index stale");
    }

    #[test]
    fn add_nodes_keeps_range_and_composite_indexes_fresh() {
        let mut g = DirGraph::new();
        add_nodes(
            &mut g,
            people(vec![("p1", "Oslo")]),
            "Person".to_string(),
            "id".to_string(),
            None,
            None,
        )
        .expect("first load");

        g.create_range_index("Person", "city");
        g.create_composite_index("Person", &["city"]);

        add_nodes(
            &mut g,
            people(vec![("p2", "Oslo")]),
            "Person".to_string(),
            "id".to_string(),
            None,
            None,
        )
        .expect("second load");

        let oslo = Value::String("Oslo".to_string());
        let ranged = g
            .lookup_range(
                "Person",
                "city",
                std::ops::Bound::Included(&oslo),
                std::ops::Bound::Included(&oslo),
            )
            .unwrap_or_default();
        assert_eq!(ranged.len(), 2, "bulk load left the range index stale");

        let composite = g
            .lookup_by_composite_index("Person", &["city".to_string()], &[oslo])
            .unwrap_or_default();
        assert_eq!(
            composite.len(),
            2,
            "bulk load left the composite index stale"
        );
    }
}

#[cfg(test)]
mod constraint_snapshot_tests {
    use super::*;

    /// `populate_index_keys` snapshots the declared UNIQUE constraints out of a
    /// `HashMap`, whose iteration order is reseeded per process. Left unsorted,
    /// two saves of the same graph produce different bytes — and because the
    /// order only varies *between* processes, no single-process test catches it.
    /// Asserting the snapshot is sorted pins the invariant directly.
    #[test]
    fn populate_index_keys_snapshots_unique_constraints_sorted() {
        let mut graph = DirGraph::new();
        // Declared out of order, and across two node types, so an unsorted
        // snapshot has plenty of room to disagree with a sorted one.
        for (node_type, properties) in [
            ("Person", vec!["email"]),
            ("Order", vec!["ref"]),
            ("Person", vec!["city", "street"]),
            ("Person", vec!["ssn"]),
            ("Order", vec!["customer", "seq"]),
        ] {
            graph
                .create_unique_constraint(node_type, &properties)
                .expect("empty graph cannot violate a constraint");
        }

        graph.populate_index_keys();

        // Spelled out rather than compared against `sorted(snapshot)`: a
        // self-referential assertion can pass by luck when the HashMap happens
        // to hand back an already-ordered set.
        let expected: Vec<(String, Vec<String>)> = [
            ("Order", vec!["customer", "seq"]),
            ("Order", vec!["ref"]),
            ("Person", vec!["city", "street"]),
            ("Person", vec!["email"]),
            ("Person", vec!["ssn"]),
        ]
        .into_iter()
        .map(|(t, props)| {
            (
                t.to_string(),
                props.into_iter().map(str::to_string).collect(),
            )
        })
        .collect();
        assert_eq!(
            graph.unique_constraint_keys, expected,
            "unique_constraint_keys must be persisted in a deterministic order"
        );
    }
}

/// Index freshness after the two *property-overwrite* paths.
///
/// Sibling of `bulk_index_freshness_tests` above, which covers the bulk
/// *append*. The failure mode here is strictly worse: appending behind a stale
/// index hides rows, whereas overwriting a value behind one makes
/// `MATCH (n:T {prop: <old value>})` return a node that no longer holds the old
/// value. A wrong answer, not a missing one.
#[cfg(test)]
mod overwrite_index_freshness_tests {
    use super::*;
    use crate::datatypes::values::{DataFrame, Value};
    use crate::graph::mutation::maintain::{add_nodes, update_node_properties};
    use crate::graph::storage::GraphWrite;

    fn people(rows: Vec<(&str, &str)>) -> DataFrame {
        DataFrame::from_cypher_rows(
            vec!["id".to_string(), "city".to_string()],
            rows.into_iter()
                .map(|(id, city)| {
                    vec![
                        Value::String(id.to_string()),
                        Value::String(city.to_string()),
                    ]
                })
                .collect(),
        )
        .expect("dataframe")
    }

    /// `update_node_properties` writes through the batch path, which skips the
    /// per-write index maintenance the Cypher SET path performs. Before the fix
    /// the equality index kept the pre-update value, so a lookup for the *old*
    /// value still returned the node.
    #[test]
    fn update_node_properties_keeps_the_property_index_fresh() {
        let mut g = DirGraph::new();
        add_nodes(
            &mut g,
            people(vec![("p1", "Oslo")]),
            "Person".to_string(),
            "id".to_string(),
            None,
            None,
        )
        .expect("load");

        assert_eq!(g.create_index("Person", "city"), 1);
        let node = g
            .type_indices
            .get("Person")
            .and_then(|nodes| nodes.iter().next())
            .expect("the loaded Person node");

        update_node_properties(
            &mut g,
            &[(Some(node), Value::String("Bergen".to_string()))],
            "city",
        )
        .expect("update");

        let stale = g
            .lookup_by_index("Person", "city", &Value::String("Oslo".to_string()))
            .unwrap_or_default();
        assert!(
            stale.is_empty(),
            "the index still resolves the overwritten value 'Oslo' to {stale:?} — \
             MATCH (n:Person {{city: 'Oslo'}}) would return a node whose city is 'Bergen'"
        );

        let fresh = g
            .lookup_by_index("Person", "city", &Value::String("Bergen".to_string()))
            .unwrap_or_default();
        assert_eq!(fresh, vec![node], "the new value is not indexed");
    }

    /// The range and composite structures share the same refresh, so they must
    /// forget the overwritten value too.
    #[test]
    fn update_node_properties_keeps_range_and_composite_indexes_fresh() {
        let mut g = DirGraph::new();
        add_nodes(
            &mut g,
            people(vec![("p1", "Oslo")]),
            "Person".to_string(),
            "id".to_string(),
            None,
            None,
        )
        .expect("load");

        g.create_range_index("Person", "city");
        g.create_composite_index("Person", &["city"]);
        let node = g
            .type_indices
            .get("Person")
            .and_then(|nodes| nodes.iter().next())
            .expect("the loaded Person node");

        update_node_properties(
            &mut g,
            &[(Some(node), Value::String("Bergen".to_string()))],
            "city",
        )
        .expect("update");

        let oslo = Value::String("Oslo".to_string());
        let ranged = g
            .lookup_range(
                "Person",
                "city",
                std::ops::Bound::Included(&oslo),
                std::ops::Bound::Included(&oslo),
            )
            .unwrap_or_default();
        assert!(
            ranged.is_empty(),
            "the range index still resolves the overwritten value: {ranged:?}"
        );

        let composite = g
            .lookup_by_composite_index("Person", &["city".to_string()], &[oslo])
            .unwrap_or_default();
        assert!(
            composite.is_empty(),
            "the composite index still resolves the overwritten value: {composite:?}"
        );
    }

    /// Bulk-update validation is intentionally reused when batch actions are
    /// assembled. Duplicate live rows must still count as duplicate updates,
    /// while missing/absent rows retain the existing report and error shape.
    #[test]
    fn update_node_properties_reuses_validation_without_changing_report_semantics() {
        let mut g = DirGraph::new();
        add_nodes(
            &mut g,
            people(vec![("p1", "Oslo"), ("p2", "Trondheim")]),
            "Person".to_string(),
            "id".to_string(),
            None,
            None,
        )
        .expect("load");

        let loaded: Vec<NodeIndex> = g
            .type_indices
            .get("Person")
            .expect("the loaded Person nodes")
            .iter()
            .collect();
        let [node, dead] = loaded.as_slice() else {
            panic!("expected exactly two loaded Person nodes");
        };
        let (node, dead) = (*node, *dead);
        GraphWrite::remove_node(&mut g.graph, dead).expect("remove the second node");
        let missing = NodeIndex::new(node.index() + 10_000);
        let report = update_node_properties(
            &mut g,
            &[
                (Some(node), Value::Int64(7)),
                (Some(node), Value::Int64(7)),
                (Some(dead), Value::Int64(7)),
                (Some(missing), Value::Int64(7)),
                (None, Value::Int64(7)),
            ],
            "city",
        )
        .expect("valid rows still update when other rows are absent");

        assert_eq!(
            report.nodes_updated, 2,
            "duplicate live rows remain updates"
        );
        assert_eq!(
            report.nodes_skipped, 6,
            "dead, missing, and absent rows retain validation + assembly skip accounting"
        );
        assert_eq!(report.errors.len(), 5);
        for invalid in [dead, missing] {
            assert!(report
                .errors
                .iter()
                .any(|error| error == &format!("Node index {:?} not found in graph", invalid)));
            assert!(report
                .errors
                .iter()
                .any(|error| error == &format!("Node index {:?} is out of bounds", invalid)));
        }
        assert!(report
            .errors
            .iter()
            .any(|error| error.contains("Type mismatch")));
        assert_eq!(
            g.node_view(node)
                .and_then(|data| data.get_property("city"))
                .map(|value| value.into_owned()),
            Some(Value::Int64(7))
        );
    }
}

/// The range index must fork by pointer, not by copy — the same structural
/// property `property_indices` gets from [`super::index_layer::LayeredIndex`].
///
/// Held-view first write at 100k measured 0.889 ms with a range index against
/// 0.048 ms with an equality index (P4, 2026-08-13): the whole gap was the
/// plain `BTreeMap` being deep-cloned on the copy-on-write fork.
#[cfg(test)]
mod range_index_fork_tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::session::{execute_mut, ExecuteOptions};

    fn run(graph: &mut DirGraph, query: &str) {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("`{query}` failed: {e}"));
    }

    fn indexed_graph() -> DirGraph {
        let mut graph = DirGraph::new();
        run(
            &mut graph,
            "UNWIND range(0, 999) AS i CREATE (:Item {id: i, qty: i % 97})",
        );
        graph.create_range_index("Item", "qty");
        graph
    }

    fn bucket_ptr(graph: &DirGraph, value: i64) -> *const petgraph::graph::NodeIndex {
        graph
            .range_indices
            .get(&("Item".to_string(), "qty".to_string()))
            .expect("range index present")
            .get(&Value::Int64(value))
            .expect("bucket present")
            .as_ptr()
    }

    /// A fork shares the range index's buckets outright.
    ///
    /// `as_ptr` on the members `Vec` is the observation: a deep-cloned
    /// `BTreeMap` reallocates every bucket, a shared immutable level hands both
    /// sides the same allocation.
    #[test]
    fn a_fork_shares_the_range_index_buckets() {
        let graph = indexed_graph();
        let fork = graph.clone();

        for value in [0i64, 13, 96] {
            assert_eq!(
                bucket_ptr(&graph, value),
                bucket_ptr(&fork, value),
                "bucket {value} was copied by the fork instead of shared"
            );
        }
    }

    /// ...and the writer's edits stay invisible to the reader that forced the
    /// fork, which is what makes the sharing safe.
    #[test]
    fn a_write_after_the_fork_leaves_the_readers_buckets_alone() {
        let mut writer = indexed_graph();
        let reader = writer.clone();

        run(&mut writer, "MATCH (n:Item {id: 5}) SET n.qty = 500");

        let key = ("Item".to_string(), "qty".to_string());
        let reader_bucket = reader.range_indices[&key]
            .get(&Value::Int64(5))
            .expect("the reader keeps its pre-write bucket");
        assert!(
            reader_bucket.len() > 1,
            "the reader's bucket must still hold every pre-write member"
        );
        assert!(
            reader.range_indices[&key].get(&Value::Int64(500)).is_none(),
            "the reader must not see the writer's new bucket"
        );
        assert!(
            writer.range_indices[&key]
                .get(&Value::Int64(500))
                .is_some_and(|members| members.len() == 1),
            "the writer's new bucket must exist on its side"
        );

        // Ordered iteration is what a range index is for: the merged view must
        // still come out sorted, tombstones and overlays included.
        let values: Vec<i64> = writer.range_indices[&key]
            .iter()
            .filter_map(|(value, _)| match value {
                Value::Int64(n) => Some(*n),
                _ => None,
            })
            .collect();
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_eq!(values, sorted, "the merged iteration must stay ordered");
        assert_eq!(values.last(), Some(&500));
    }
}

/// `ensure_column_store_for_push` used to rebuild the whole store whenever the
/// registered `TypeSchema` had grown past the store's own — every existing row
/// re-pushed into a fresh store, per newly-seen key. With `ColumnStore::push_row`
/// appending its own columns that rebuild is not merely an optimisation
/// opportunity, it is wrong work: it is O(rows x cols) on a path whose contract
/// is one row.
#[cfg(test)]
mod ensure_column_store_for_push_tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::storage::column_store::{
        column_store_row_pushes, reset_column_store_row_pushes,
    };

    fn push(graph: &mut DirGraph, node_type: &str, pairs: &[(&str, Value)]) -> u32 {
        let interned: Vec<(InternedKey, Value)> = pairs
            .iter()
            .map(|(k, v)| (graph.interner.get_or_intern(k), v.clone()))
            .collect();
        let keys: Vec<InternedKey> = interned.iter().map(|(k, _)| *k).collect();
        graph.ensure_type_schema_keys(node_type, &keys);
        let store = graph.ensure_column_store_for_push(node_type);
        store.push_row(&interned)
    }

    #[test]
    fn a_widening_key_set_never_rebuilds_the_store() {
        let mut g = DirGraph::new();
        for i in 0..50i64 {
            push(&mut g, "Item", &[("p0", Value::Int64(i))]);
        }

        reset_column_store_row_pushes();
        // Three statements, each introducing a property the type has never
        // carried. Before the append path this cost 51 + 52 + 53 row pushes.
        push(
            &mut g,
            "Item",
            &[("p0", Value::Int64(50)), ("p1", Value::Int64(1))],
        );
        push(
            &mut g,
            "Item",
            &[("p0", Value::Int64(51)), ("p2", Value::Int64(2))],
        );
        push(
            &mut g,
            "Item",
            &[("p0", Value::Int64(52)), ("p3", Value::Int64(3))],
        );
        assert_eq!(
            column_store_row_pushes(),
            3,
            "growing a type's schema rebuilt its ColumnStore row by row"
        );

        // ... and nothing was lost or shifted by the growth.
        let store = g.column_store("Item").expect("store");
        let p0 = InternedKey::from_str("p0");
        assert_eq!(store.row_count(), 53);
        for i in 0..53u32 {
            assert_eq!(store.get(i, p0), Some(Value::Int64(i as i64)));
        }
        assert_eq!(
            store.get(50, InternedKey::from_str("p1")),
            Some(Value::Int64(1))
        );
        assert_eq!(
            store.get(51, InternedKey::from_str("p2")),
            Some(Value::Int64(2))
        );
        assert_eq!(
            store.get(52, InternedKey::from_str("p3")),
            Some(Value::Int64(3))
        );
        // Rows that predate a column read as absent.
        assert_eq!(store.get(0, InternedKey::from_str("p1")), None);
    }

    #[test]
    fn a_rebuild_would_have_resurrected_tombstoned_rows() {
        // The rebuild loop re-pushed `0..row_count` and never carried the
        // tombstone bitmap across, so a row deleted before a schema growth came
        // back as a live row. `materialize_for_append`, the other migrate-style
        // copy in the store, does re-tombstone — the two disagreed.
        let mut g = DirGraph::new();
        for i in 0..8i64 {
            push(&mut g, "Item", &[("p0", Value::Int64(i))]);
        }
        Arc::make_mut(g.column_store_mut("Item").expect("store")).tombstone(3);
        assert_eq!(g.column_store("Item").expect("store").live_count(), 7);

        push(
            &mut g,
            "Item",
            &[("p0", Value::Int64(8)), ("fresh", Value::Int64(1))],
        );

        let store = g.column_store("Item").expect("store");
        assert_eq!(
            store.live_count(),
            8,
            "a schema growth resurrected a tombstoned row"
        );
        assert_eq!(store.get(3, InternedKey::from_str("p0")), None);
    }
}

/// Auto-vacuum's trigger, and the kind of garbage it could not see.
///
/// `node_bound - node_count` counts *free petgraph slots*, which a later
/// create takes back. A columnar row does not work that way: a delete leaves
/// its row behind and a create appends a new one, so replacement churn grows
/// the store without ever moving the node-slot reading off zero. Measured on
/// the disk backend, which is already always-columnar: 1,500 delete/create
/// pairs over a 2,000-node type left 3,500 rows for 2,000 live nodes at
/// `fragmentation_ratio` 0.000. Under the always-columnar flip that becomes
/// every graph's steady state.
#[cfg(test)]
mod auto_vacuum_trigger_tests {
    use super::*;
    use crate::datatypes::{DataFrame, Value};

    fn columnar_items(n: i64) -> DirGraph {
        let mut g = DirGraph::new();
        let rows: Vec<Vec<Value>> = (1..=n)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("t{i}")),
                    Value::Int64(i * 10),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(
            vec!["id".to_string(), "title".to_string(), "c0".to_string()],
            rows,
        )
        .unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut g,
            df,
            "Item".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        g.enable_columnar();
        g
    }

    /// Rows in the store that no live node points at, with the petgraph slot
    /// count reading clean — the churn residue, reproduced directly.
    fn orphan_rows(g: &mut DirGraph, count: usize) {
        let key = g.interner.get_or_intern("c0");
        let store = g.ensure_column_store_for_push("Item");
        for i in 0..count {
            store.push_id(&Value::Int64(1_000_000 + i as i64));
            store.push_title(&Value::String(format!("dead{i}")));
            store.push_row(&[(key, Value::Int64(-1))]);
        }
    }

    #[test]
    fn columnar_garbage_triggers_a_vacuum_that_reclaims_it() {
        let mut g = columnar_items(200);
        g.auto_vacuum_threshold = Some(0.3);
        orphan_rows(&mut g, 150);

        let (total, live) = g.columnar_row_census();
        assert_eq!((total, live), (350, 200), "fixture drift");
        assert_eq!(
            g.graph.node_bound() - g.graph.node_count(),
            0,
            "precondition: the node-slot reading must be clean, or this test is \
             measuring the old trigger"
        );

        let remap = g.check_auto_vacuum().expect(
            "43% of the type's rows are garbage and auto-vacuum did not fire: \
             the trigger is reading free petgraph slots, which replacement \
             churn returns to zero",
        );
        // Node slots were clean, so the columnar-only arm ran: no petgraph
        // rebuild, no index movement, and a holder of node indices must keep
        // them.
        assert!(
            !remap.describes_rebuild(),
            "a columnar-only reclaim moved no node index; reporting a rebuild \
             would cost every index holder its state for nothing"
        );
        let (total, live) = g.columnar_row_census();
        assert_eq!(
            (total, live),
            (200, 200),
            "the vacuum fired but reclaimed no columnar rows"
        );
        // The live data is untouched by the reclamation.
        assert_eq!(g.graph.node_count(), 200);
    }

    #[test]
    fn a_clean_store_does_not_trigger_a_vacuum() {
        // Non-vacuity's other half: the trigger must still say no.
        let mut g = columnar_items(200);
        g.auto_vacuum_threshold = Some(0.3);
        assert!(g.check_auto_vacuum().is_none());

        // ... and garbage under the small-graph floor is not worth a rebuild.
        orphan_rows(&mut g, 40);
        assert!(g.check_auto_vacuum().is_none());

        // ... nor is garbage below the ratio threshold.
        let mut g = columnar_items(2000);
        g.auto_vacuum_threshold = Some(0.3);
        orphan_rows(&mut g, 300);
        assert!(g.check_auto_vacuum().is_none());
        assert_eq!(g.columnar_row_census(), (2300, 2000));
    }

    /// The mapping a fired vacuum hands back has to name the surviving nodes,
    /// or the caller that follows it lands on the wrong ones.
    #[test]
    fn a_fired_vacuum_returns_the_mapping_its_caller_needs() {
        let mut g = columnar_items(400);
        g.auto_vacuum_threshold = Some(0.3);

        // Delete the first 200 nodes: node slots 0..200 go free, 200..400 stay.
        let doomed: Vec<NodeIndex> = (0..200).map(NodeIndex::new).collect();
        for idx in &doomed {
            g.graph.remove_node(*idx);
        }
        assert_eq!(g.graph.node_bound() - g.graph.node_count(), 200);

        let remap = g
            .check_auto_vacuum()
            .expect("50% of node slots are free at threshold 0.3");
        assert!(remap.describes_rebuild());
        assert_eq!(remap.len(), 200, "every survivor must be in the mapping");

        // The survivors compact to 0..200 in ascending old-index order.
        for (offset, old_raw) in (200..400).enumerate() {
            assert_eq!(
                remap.get(NodeIndex::new(old_raw)),
                Some(NodeIndex::new(offset)),
            );
        }
        // The dead are absent, not silently aliased onto a live node.
        for idx in &doomed {
            assert_eq!(remap.get(*idx), None);
        }
        assert_eq!(g.auto_vacuums_run, 1);
    }

    /// The counter only moves when a vacuum actually fires.
    #[test]
    fn the_run_counter_counts_fired_vacuums_only() {
        let mut g = columnar_items(200);
        g.auto_vacuum_threshold = Some(0.3);
        assert_eq!(g.auto_vacuums_run, 0);

        // Below the floor: no fire, no count.
        orphan_rows(&mut g, 40);
        assert!(g.check_auto_vacuum().is_none());
        assert_eq!(g.auto_vacuums_run, 0);

        orphan_rows(&mut g, 110);
        assert!(g.check_auto_vacuum().is_some());
        assert_eq!(g.auto_vacuums_run, 1);

        // Disabled: never fires, whatever the garbage.
        orphan_rows(&mut g, 500);
        g.auto_vacuum_threshold = None;
        assert!(g.check_auto_vacuum().is_none());
        assert_eq!(g.auto_vacuums_run, 1);
    }
}

/// Edge slots are the third garbage population, and the one nothing could see.
///
/// A relationship-only delete workload (`MATCH ()-[r]->() DELETE r`) leaves
/// every node alive and every columnar row referenced, so both of the other
/// two readings are clean. Measured before this: 500 of 1,000 edges deleted
/// reported `fragmentation_ratio` 0.000, could never trigger an auto-vacuum,
/// and got a no-op out of an explicit `vacuum()` as well.
#[cfg(test)]
mod edge_fragmentation_tests {
    use super::*;
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::schema::EdgeData;
    use std::collections::HashMap;

    /// `n` nodes in a ring of `n` `KNOWS` edges.
    fn ring(n: usize) -> DirGraph {
        let mut g = DirGraph::new();
        let rows: Vec<Vec<Value>> = (0..n as i64)
            .map(|i| vec![Value::Int64(i), Value::String(format!("t{i}"))])
            .collect();
        let df =
            DataFrame::from_cypher_rows(vec!["id".to_string(), "title".to_string()], rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut g,
            df,
            "Item".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        for i in 0..n {
            let data = EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut g.interner);
            g.graph
                .add_edge(NodeIndex::new(i), NodeIndex::new((i + 1) % n), data);
        }
        g
    }

    /// Delete the first `k` edge slots, leaving the tail live so the bound
    /// stays put — the shape a `WHERE`-filtered `DELETE r` produces.
    fn delete_leading_edges(g: &mut DirGraph, k: usize) {
        for i in 0..k {
            g.graph.remove_edge(EdgeIndex::new(i));
        }
    }

    #[test]
    fn edge_only_garbage_is_visible_to_graph_info() {
        let mut g = ring(1000);
        assert_eq!(g.graph_info().edge_tombstones, 0);

        delete_leading_edges(&mut g, 500);
        let info = g.graph_info();
        assert_eq!(info.edge_count, 500);
        assert_eq!(info.edge_capacity, 1000);
        assert_eq!(info.edge_tombstones, 500);
        // The node-shaped reading is clean, which is exactly why this needed
        // its own number rather than a wider `fragmentation_ratio`.
        assert_eq!(info.node_tombstones, 0);
        assert_eq!(info.fragmentation_ratio, 0.0);
    }

    #[test]
    fn edge_only_garbage_triggers_a_vacuum_that_reclaims_it() {
        let mut g = ring(1000);
        g.auto_vacuum_threshold = Some(0.3);
        delete_leading_edges(&mut g, 500);

        assert!(
            g.check_auto_vacuum().is_some(),
            "half the edge slots are free and auto-vacuum did not fire: the \
             trigger is blind to relationship-only deletes"
        );
        let info = g.graph_info();
        assert_eq!(info.edge_count, 500);
        assert_eq!(info.edge_tombstones, 0, "the vacuum reclaimed no edge slot");
        assert_eq!(info.node_count, 1000, "live data must survive untouched");
    }

    #[test]
    fn an_explicit_vacuum_reclaims_edge_slots_on_a_node_clean_graph() {
        // The measured no-op, inverted: `vacuum()` used to return early on
        // `node_count == node_bound` and leave every free edge slot in place.
        let mut g = ring(1000);
        delete_leading_edges(&mut g, 500);
        assert_eq!(g.graph.node_bound(), g.graph.node_count(), "precondition");

        let remap = g.vacuum();
        assert_eq!(g.graph_info().edge_tombstones, 0);
        assert_eq!(g.graph_info().edge_count, 500);
        // Node indices did not move — the mapping is the identity — but a
        // rebuild did happen, so the mapping covers every slot.
        assert!(remap.describes_rebuild());
        assert_eq!(remap.get(NodeIndex::new(7)), Some(NodeIndex::new(7)));
    }

    #[test]
    fn a_clean_edge_set_does_not_trigger_a_vacuum() {
        let mut g = ring(1000);
        g.auto_vacuum_threshold = Some(0.3);
        assert!(g.check_auto_vacuum().is_none());

        // Under the small-graph floor.
        delete_leading_edges(&mut g, 100);
        assert!(g.check_auto_vacuum().is_none());
        assert_eq!(g.graph_info().edge_tombstones, 100);

        // Above the floor but under the ratio.
        let mut g = ring(1000);
        g.auto_vacuum_threshold = Some(0.5);
        delete_leading_edges(&mut g, 400);
        assert!(g.check_auto_vacuum().is_none());
        assert_eq!(g.graph_info().edge_tombstones, 400);
    }
}

/// `CurrentSelection::remap_indices` — the half of the vacuum fix that keeps a
/// held selection describing the same set of nodes.
#[cfg(test)]
mod selection_remap_tests {
    use super::*;
    use crate::graph::schema::{CurrentSelection, SelectionLevel};

    fn remap_of(pairs: &[(usize, usize)], bound: usize) -> NodeRemap {
        let mut remap = NodeRemap::with_bound(bound);
        for (old, new) in pairs {
            remap.set(*old, NodeIndex::new(*new));
        }
        remap
    }

    fn root_selection(indices: &[usize]) -> CurrentSelection {
        let mut sel = CurrentSelection::new();
        let level = sel.get_level_mut(0).unwrap();
        level.add_selection(None, indices.iter().copied().map(NodeIndex::new).collect());
        sel
    }

    fn sorted(level: &SelectionLevel) -> Vec<usize> {
        let mut v: Vec<usize> = level.iter_node_indices().map(|i| i.index()).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn survivors_follow_the_compaction_and_the_dead_drop_out() {
        let mut sel = root_selection(&[3, 5, 7]);
        // 5 did not survive; 3 and 7 moved down.
        sel.remap_indices(&remap_of(&[(3, 0), (7, 1)], 8));

        assert_eq!(sorted(sel.get_level(0).unwrap()), vec![0, 1]);
        assert_eq!(sel.current_node_count(), 2);
    }

    #[test]
    fn a_group_whose_parent_died_is_dropped_whole() {
        let mut sel = CurrentSelection::new();
        let level = sel.get_level_mut(0).unwrap();
        level.add_selection(Some(NodeIndex::new(1)), vec![NodeIndex::new(4)]);
        level.add_selection(Some(NodeIndex::new(2)), vec![NodeIndex::new(5)]);

        // Parent 2 is gone; its child 5 survived the vacuum but is no longer
        // reachable through the traversal that put it here.
        sel.remap_indices(&remap_of(&[(1, 0), (4, 1), (5, 2)], 8));

        let level = sel.get_level(0).unwrap();
        let groups: Vec<_> = level.iter_groups().collect();
        assert_eq!(
            groups.len(),
            1,
            "the orphaned group must not be re-parented"
        );
        let (parent, children) = groups[0];
        assert_eq!(*parent, Some(NodeIndex::new(0)));
        assert_eq!(children, &vec![NodeIndex::new(1)]);
    }

    #[test]
    fn a_group_that_loses_every_child_stays_empty_rather_than_vanishing() {
        let mut sel = CurrentSelection::new();
        let level = sel.get_level_mut(0).unwrap();
        level.add_selection(
            Some(NodeIndex::new(1)),
            vec![NodeIndex::new(4), NodeIndex::new(5)],
        );

        sel.remap_indices(&remap_of(&[(1, 0)], 8));

        let level = sel.get_level(0).unwrap();
        assert_eq!(level.iter_groups().count(), 1);
        assert_eq!(level.node_count(), 0);
    }

    #[test]
    fn every_level_is_remapped_not_just_the_last() {
        let mut sel = root_selection(&[3]);
        sel.add_level();
        sel.get_level_mut(1)
            .unwrap()
            .add_selection(Some(NodeIndex::new(3)), vec![NodeIndex::new(7)]);

        sel.remap_indices(&remap_of(&[(3, 0), (7, 1)], 8));

        assert_eq!(sorted(sel.get_level(0).unwrap()), vec![0]);
        assert_eq!(sorted(sel.get_level(1).unwrap()), vec![1]);
    }

    #[test]
    fn a_mapping_that_describes_no_rebuild_leaves_the_selection_alone() {
        // The disk-backend and columnar-only shape. Indices did not move, so
        // treating "nothing in the mapping" as "everything died" would empty a
        // perfectly valid selection.
        let mut sel = root_selection(&[3, 5, 7]);
        sel.remap_indices(&NodeRemap::default());
        assert_eq!(sorted(sel.get_level(0).unwrap()), vec![3, 5, 7]);
    }

    #[test]
    fn a_rebuild_with_no_survivors_empties_the_selection() {
        // The other side of the same discrimination: here the mapping is also
        // `is_empty()`, but every index the caller holds is genuinely gone.
        let mut sel = root_selection(&[3, 5, 7]);
        sel.remap_indices(&NodeRemap::with_bound(8));
        assert_eq!(sel.current_node_count(), 0);
    }
}

/// A disk graph wrapped for write capture (`durable=` / `cdc::enable`, whose
/// disk refusals are the *caller's* — see `durability::open_log`) is still a
/// disk graph. Code that matches `GraphBackend::Disk` directly misses it while
/// the guard in front of it (`GraphRead::is_disk`, `create_property_index_routed`)
/// does not: `CREATE INDEX` falls to the in-memory hash builder the routed
/// entry point exists to avoid, `DROP INDEX` reports success while the
/// persistent index stays on disk, and the bulk loader aborts on `unreachable!`.
#[cfg(test)]
mod capture_wrapped_backend_routing_tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::storage::backend::GraphBackend;
    use tempfile::TempDir;

    /// Built through `add_nodes` so the type carries the column metadata the
    /// disk property-index builder reads, and saved as a disk graph. Each test
    /// calls `wrap_for_capture` itself, at the point that makes its assertion
    /// non-vacuous.
    fn disk_graph(dir: &TempDir) -> DirGraph {
        let frame = crate::datatypes::DataFrame::from_cypher_rows(
            vec!["id".into(), "title".into(), "city".into()],
            vec![
                vec![
                    Value::Int64(1),
                    Value::String("p1".into()),
                    Value::String("Oslo".into()),
                ],
                vec![
                    Value::Int64(2),
                    Value::String("p2".into()),
                    Value::String("Bergen".into()),
                ],
            ],
        )
        .unwrap();
        let mut graph = DirGraph::new();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            frame,
            "Person".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        graph.enable_disk_mode().unwrap();
        graph.save_disk(dir.path().to_str().unwrap()).unwrap();
        graph
    }

    fn wrap(graph: &mut DirGraph) {
        graph.graph.wrap_for_capture();
        assert!(
            matches!(&graph.graph, GraphBackend::Recording(_)),
            "the fixture must be capture-wrapped"
        );
    }

    #[test]
    fn create_index_routes_to_the_persistent_builder_under_capture() {
        let dir = TempDir::new().unwrap();
        let mut graph = disk_graph(&dir);
        wrap(&mut graph);
        let (count, persistent) = graph
            .create_property_index_routed("Person", "city")
            .unwrap();
        assert!(
            persistent,
            "a capture-wrapped disk graph must still build the mmap index"
        );
        assert_eq!(count, 2);
        assert!(graph.has_persistent_property_index("Person", "city"));
        assert!(
            !graph.has_index("Person", "city"),
            "no in-memory hash index may be built over a disk graph"
        );
    }

    /// The bulk loader gates its columnar-write path on `GraphRead::is_disk`,
    /// which *does* look through the wrapper — so the disk arm behind it must
    /// too, or a wrapped disk graph reaches an `unreachable!` and aborts the
    /// load.
    #[test]
    fn a_bulk_update_on_a_captured_disk_graph_writes_the_columnar_row() {
        let dir = TempDir::new().unwrap();
        let mut graph = disk_graph(&dir);
        wrap(&mut graph);
        let update = crate::datatypes::DataFrame::from_cypher_rows(
            vec!["id".into(), "title".into(), "city".into()],
            vec![vec![
                Value::Int64(1),
                Value::String("p1".into()),
                Value::String("Trondheim".into()),
            ]],
        )
        .unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            update,
            "Person".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            Some("update".to_string()),
        )
        .unwrap();
        let rows = graph
            .type_indices
            .get("Person")
            .map_or(0, |nodes| nodes.iter().count());
        assert_eq!(rows, 2, "an update must not add a row");
        let reader = graph.property_reader("Person", "city");
        let cities: Vec<Option<Value>> = graph
            .type_indices
            .get("Person")
            .unwrap()
            .iter()
            .map(|idx| graph.read_indexed(&reader, idx))
            .collect();
        assert!(
            cities.contains(&Some(Value::String("Trondheim".into()))),
            "the updated columnar row must be readable: {cities:?}"
        );
    }

    #[test]
    fn drop_index_removes_the_persistent_index_under_capture() {
        let dir = TempDir::new().unwrap();
        let mut graph = disk_graph(&dir);
        // Built *before* wrapping, so this test pins the drop side alone: the
        // index it removes is unambiguously the persistent one.
        assert!(
            graph
                .create_property_index_routed("Person", "city")
                .unwrap()
                .1
        );
        wrap(&mut graph);
        assert!(graph.drop_index("Person", "city").unwrap());
        assert!(
            !graph.has_persistent_property_index("Person", "city"),
            "DROP INDEX reported success, so the index must be gone"
        );
    }
}
