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
}
