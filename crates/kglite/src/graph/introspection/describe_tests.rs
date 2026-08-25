//! Describe-output regression tests extracted from describe.rs.

use super::*;

#[cfg(test)]
mod declared_type_annotation_tests {
    use super::*;
    use crate::datatypes::values::Value;
    use crate::graph::property_types::DeclaredType;
    use crate::graph::schema::NodeData;
    use crate::graph::storage::GraphWrite;
    use std::collections::HashMap;

    fn person_graph() -> DirGraph {
        let mut graph = DirGraph::new();
        for (id, age) in [(1u32, 30i64), (2, 25)] {
            let node = NodeData::new(
                Value::UniqueId(id),
                Value::String(format!("p{id}")),
                "Person".to_string(),
                HashMap::from([("age".to_string(), Value::Int64(age))]),
                &mut graph.interner,
            );
            let idx = graph.graph.add_node(node);
            graph
                .type_indices
                .entry_or_default("Person".to_string())
                .push(idx);
        }
        graph
    }

    fn describe(graph: &DirGraph) -> String {
        compute_description(
            graph,
            None,
            &ConnectionDetail::Off,
            &CypherDetail::Off,
            &FluentDetail::Off,
            None,
            None,
            None,
        )
        .unwrap()
    }

    /// An agent planning a write needs to know the value will be rejected
    /// *before* attempting it, which is the whole reason `describe()` annotates
    /// declared constraints at all.
    #[test]
    fn describe_annotates_a_declared_property_type() {
        let mut graph = person_graph();
        assert!(
            !describe(&graph).contains("declared_type"),
            "an unconstrained property must carry no annotation"
        );

        graph
            .create_property_type_constraint("Person", "age", DeclaredType::Integer)
            .unwrap();
        let described = describe(&graph);
        assert!(
            described.contains("declared_type=\"INTEGER\""),
            "got: {described}"
        );
    }

    /// The same two facts on the connection side. An agent planning
    /// `CREATE (a)-[:KNOWS {since: …}]->(b)` needs the edge annotation for the
    /// same reason it needs the node one — and in the same vocabulary, or it
    /// has to learn two.
    #[test]
    fn describe_annotates_a_declared_relationship_constraint() {
        use crate::graph::algorithms::Interrupt;
        use crate::graph::languages::cypher::executor::write::execute_mutable;
        use crate::graph::languages::cypher::parser::parse_cypher;

        let mut graph = DirGraph::new();
        let parsed = parse_cypher(
            "CREATE (a:Person {person_id: 1})-[:KNOWS {since: 2020}]->(b:Person {person_id: 2})",
        )
        .unwrap();
        execute_mutable(&mut graph, &parsed, HashMap::new(), Interrupt::default()).unwrap();

        let with_connections = |graph: &DirGraph| {
            compute_description(
                graph,
                None,
                &ConnectionDetail::Topics(vec!["KNOWS".to_string()]),
                &CypherDetail::Off,
                &FluentDetail::Off,
                None,
                None,
                None,
            )
            .unwrap()
        };
        assert!(
            !with_connections(&graph).contains("constraint="),
            "an unconstrained edge property must carry no annotation"
        );

        graph
            .create_rel_not_null_constraint("KNOWS", "since", &Interrupt::default())
            .unwrap();
        graph
            .create_rel_property_type_constraint(
                "KNOWS",
                "since",
                DeclaredType::Integer,
                &Interrupt::default(),
            )
            .unwrap();
        let described = with_connections(&graph);
        assert!(
            described.contains("constraint=\"not_null\""),
            "got: {described}"
        );
        assert!(
            described.contains("declared_type=\"INTEGER\""),
            "got: {described}"
        );
    }

    /// The two facts are orthogonal — a property can be unique *and* typed — so
    /// the annotation must not replace the constraint one.
    #[test]
    fn a_typed_property_keeps_its_uniqueness_annotation() {
        let mut graph = person_graph();
        graph.create_unique_constraint("Person", &["age"]).unwrap();
        graph
            .create_property_type_constraint("Person", "age", DeclaredType::Integer)
            .unwrap();

        let described = describe(&graph);
        assert!(
            described.contains("constraint=\"unique\""),
            "got: {described}"
        );
        assert!(
            described.contains("declared_type=\"INTEGER\""),
            "got: {described}"
        );
    }
}

#[cfg(test)]
mod mcp_quickstart_tests {
    use super::mcp_quickstart;

    #[test]
    fn names_only_current_install_and_extension_contracts() {
        let quickstart = mcp_quickstart();
        for expected in [
            "pip install kglite",
            "cargo install kglite-mcp-server",
            "trust.allow_embedder: true",
            "library: sentence-transformers",
            "--features fastembed",
        ] {
            assert!(quickstart.contains(expected), "missing {expected:?}");
        }
        for retired in ["kglite[mcp]", "--embedder", "--trust-tools", "python:"] {
            assert!(
                !quickstart.contains(retired),
                "retired contract returned: {retired:?}"
            );
        }
    }
}

#[cfg(test)]
mod focused_detail_error_tests {
    use super::*;
    use crate::graph::session::{execute_mut, ExecuteOptions};
    use std::collections::HashMap;

    fn vessel_graph() -> DirGraph {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let mut graph = DirGraph::new();
        execute_mut(&mut graph, "CREATE (:Vessel {id: 1})", &opts).expect("seed");
        graph
    }

    /// `graph_overview(types=['vessel'])` on a graph of `Vessel`s named the
    /// available types and left the reader to spot the case difference. It is
    /// the same near-miss the MATCH warnings hint at, so it gets the same hint.
    #[test]
    fn a_near_miss_type_name_is_suggested() {
        let graph = vessel_graph();
        let error = build_focused_detail(&graph, &["vessel".to_string()], None)
            .expect_err("unknown type must error");
        assert!(error.contains("Did you mean 'Vessel'?"), "{error}");
        assert!(error.contains("Available: Vessel"), "{error}");
    }

    /// A name nothing is close to gets the list and no invented suggestion —
    /// `did_you_mean`'s bar is "genuinely close, or silent".
    #[test]
    fn a_far_type_name_gets_no_invented_suggestion() {
        let graph = vessel_graph();
        let error = build_focused_detail(&graph, &["Xyzzy".to_string()], None)
            .expect_err("unknown type must error");
        assert!(!error.contains("Did you mean"), "{error}");
    }
}

#[cfg(test)]
mod index_annotation_tests {
    use super::*;
    use crate::datatypes::values::Value;
    use crate::graph::storage::backend::GraphBackend;
    use tempfile::TempDir;

    /// Built through `add_nodes` so the type carries column metadata: the disk
    /// property index reads the `Str` column, and `describe()`'s `type_string`
    /// comes from that metadata.
    fn city_graph() -> DirGraph {
        let frame = crate::datatypes::DataFrame::from_cypher_rows(
            vec!["id".into(), "title".into(), "city".into(), "pop".into()],
            vec![
                vec![
                    Value::Int64(1),
                    Value::String("p1".into()),
                    Value::String("Oslo".into()),
                    Value::Int64(700),
                ],
                vec![
                    Value::Int64(2),
                    Value::String("p2".into()),
                    Value::String("Bergen".into()),
                    Value::Int64(280),
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
        graph
    }

    fn describe(graph: &DirGraph) -> String {
        compute_description(
            graph,
            None,
            &ConnectionDetail::Off,
            &CypherDetail::Off,
            &FluentDetail::Off,
            None,
            None,
            None,
        )
        .unwrap()
    }

    fn attr_of(described: &str, property: &str) -> String {
        let needle = format!("name=\"{property}\"");
        let line = described
            .lines()
            .find(|l| l.contains(&needle))
            .unwrap_or_else(|| panic!("no <prop {needle}> in: {described}"));
        line.trim().to_string()
    }

    /// The in-memory index is a value→members hash, and the memory backends
    /// inherit `lookup_by_property_prefix`'s `None` default — a `STARTS WITH`
    /// here full-scans, so advertising `prefix` points an agent at a path the
    /// engine does not have.
    #[test]
    fn a_memory_hash_index_advertises_equality_only() {
        let mut graph = city_graph();
        graph.create_index("Person", "city");
        let line = attr_of(&describe(&graph), "city");
        assert!(line.contains("indexed=\"eq\""), "got: {line}");
    }

    /// A range index serves ordered predicates, not equality — folding it into
    /// `eq` would promise an O(log N) point lookup that does not exist.
    #[test]
    fn a_range_index_is_reported_under_its_own_name() {
        let mut graph = city_graph();
        graph.create_range_index("Person", "pop");
        let line = attr_of(&describe(&graph), "pop");
        assert!(line.contains("indexed=\"range\""), "got: {line}");
    }

    /// Both structures on one property — what `CREATE RANGE INDEX` builds.
    #[test]
    fn equality_and_range_on_one_property_report_both() {
        let mut graph = city_graph();
        graph.create_index("Person", "pop");
        graph.create_range_index("Person", "pop");
        let line = attr_of(&describe(&graph), "pop");
        assert!(line.contains("indexed=\"eq,range\""), "got: {line}");
    }

    /// The disk index *is* a sorted key array, so prefix is real there — the
    /// one place `eq,prefix` is true.
    #[test]
    fn a_disk_string_index_advertises_prefix() {
        let dir = TempDir::new().unwrap();
        let mut graph = city_graph();
        graph.enable_disk_mode().unwrap();
        graph.save_disk(dir.path().to_str().unwrap()).unwrap();
        match &mut graph.graph {
            GraphBackend::Disk(disk) => {
                assert_eq!(disk.build_property_index("Person", "city").unwrap(), 2);
            }
            _ => panic!("expected disk backend"),
        }
        let line = attr_of(&describe(&graph), "city");
        assert!(line.contains("indexed=\"eq,prefix\""), "got: {line}");
    }

    /// `durable=True` and `cdc::enable` wrap the backend, and the index is
    /// still there underneath: a graph that loses its `indexed=` annotations
    /// the moment capture is switched on tells an agent to stop using an index
    /// it still has.
    #[test]
    fn capture_wrapping_does_not_hide_the_disk_index() {
        let dir = TempDir::new().unwrap();
        let mut graph = city_graph();
        graph.enable_disk_mode().unwrap();
        graph.save_disk(dir.path().to_str().unwrap()).unwrap();
        match &mut graph.graph {
            GraphBackend::Disk(disk) => {
                assert_eq!(disk.build_property_index("Person", "city").unwrap(), 2);
            }
            _ => panic!("expected disk backend"),
        }
        graph.graph.wrap_for_capture();
        assert!(
            graph.has_any_index("Person", "city"),
            "a wrapped disk graph still has its persistent index"
        );
        let line = attr_of(&describe(&graph), "city");
        assert!(line.contains("indexed=\"eq,prefix\""), "got: {line}");
    }
}

/// `describe(connections=['LINKS'])` samples edges through
/// `for_each_edge_of_conn_type`. A disk graph converted by `enable_disk_mode`
/// has its edges in the CSR with no `conn_type_index_*`, and an empty sweep
/// there reports a connection type that exists with no endpoints and no
/// samples — a wrong answer an agent plans against.
#[cfg(test)]
mod disk_connection_sampling_tests {
    use super::*;
    use crate::datatypes::{DataFrame, Value};

    fn linked_docs() -> DirGraph {
        let nodes = DataFrame::from_cypher_rows(
            vec!["id".into(), "title".into()],
            vec![
                vec![Value::Int64(1), Value::String("a".into())],
                vec![Value::Int64(2), Value::String("b".into())],
                vec![Value::Int64(3), Value::String("c".into())],
            ],
        )
        .unwrap();
        let links = DataFrame::from_cypher_rows(
            vec!["src".into(), "tgt".into()],
            vec![
                vec![Value::Int64(1), Value::Int64(3)],
                vec![Value::Int64(2), Value::Int64(3)],
            ],
        )
        .unwrap();
        let mut graph = DirGraph::new();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            nodes,
            "Doc".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        crate::graph::mutation::maintain::add_connections(
            &mut graph,
            links,
            "LINKS".to_string(),
            "Doc".to_string(),
            "src".to_string(),
            "Doc".to_string(),
            "tgt".to_string(),
            None,
            None,
            None,
        )
        .unwrap();
        graph
    }

    #[test]
    fn a_converted_disk_graph_still_samples_its_connections() {
        let mut graph = linked_docs();
        graph.enable_disk_mode().unwrap();
        assert!(
            graph
                .graph
                .as_disk()
                .expect("disk mode")
                .conn_type_index_types
                .is_empty(),
            "the conversion builds no conn-type index, or this test asserts nothing"
        );

        let acc = accumulate_connection_topic(&graph, InternedKey::from_str("LINKS"), "LINKS", 5);
        assert!(
            !acc.samples.is_empty(),
            "an index-less disk graph must still yield sample edges"
        );
        assert_eq!(
            acc.pair_counts.get(&("Doc".to_string(), "Doc".to_string())),
            Some(&2),
            "both Doc→Doc edges must be counted, got {:?}",
            acc.pair_counts
        );
    }
}
