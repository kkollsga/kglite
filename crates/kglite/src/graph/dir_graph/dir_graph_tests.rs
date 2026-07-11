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
