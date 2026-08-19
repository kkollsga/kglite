//! Record goldens on the Rust side of the projection boundary.
//!
//! # Why this file exists (Part N safety net, phase N1)
//!
//! `tests/test_node_record_golden.py` pins the same records *after*
//! `py_out::value_to_py` has turned them into Python dicts. That is the shape
//! users see, but it cannot distinguish "the materialisation changed" from
//! "the conversion changed" — and Part N's amended N2 rewrites **both** sides
//! of that boundary (`NodeValue::properties` becomes an `Arc`'d sorted flat
//! map; `py_out` gains a Path double-clone fix).
//!
//! So the record is pinned twice, once per side. A red line here localises the
//! change to `collect_node_properties` / `materialize_node_value`; a red line
//! only in the Python golden localises it to `py_out`.
//!
//! The expectations are **checked-in literals**, never re-derived from the
//! pipeline under test.
//!
//! **N2 must pass against these records unchanged.**

use crate::datatypes::values::{NodeValue, RelValue, Value};
use crate::datatypes::DataFrame;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_read, ExecuteOptions};
use std::collections::HashMap;

/// The `node_projection_graph` shape at readable size: `pid` -> id,
/// `name` -> title, plus `age` and `city`, with one null `city`.
/// Same column contract as `tests/benchmarks/test_bench_core.py`'s fixture and
/// as `tests/test_node_record_golden.py`'s, so the three goldens describe one
/// graph.
const N_NODES: i64 = 6;

fn projection_graph() -> DirGraph {
    let columns = vec![
        "pid".to_string(),
        "name".to_string(),
        "age".to_string(),
        "city".to_string(),
    ];
    let rows: Vec<Vec<Value>> = (0..N_NODES)
        .map(|i| {
            vec![
                Value::Int64(i),
                Value::String(format!("P{i}")),
                Value::Int64(20 + i),
                if i == N_NODES - 1 {
                    // The null row — the omission rule must drop the key.
                    Value::Null
                } else {
                    Value::String(format!("city_{}", i % 3))
                },
            ]
        })
        .collect();

    let mut g = DirGraph::new();
    let df = DataFrame::from_cypher_rows(columns, rows).unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut g,
        df,
        "Person".to_string(),
        "pid".to_string(),
        Some("name".to_string()),
        None,
    )
    .unwrap();

    let edge_rows: Vec<Vec<Value>> = (0..N_NODES - 1)
        .map(|i| vec![Value::Int64(i), Value::Int64(i + 1), Value::Int64(2020 + i)])
        .collect();
    let edge_df = DataFrame::from_cypher_rows(
        vec!["s".to_string(), "d".to_string(), "since".to_string()],
        edge_rows,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_connections(
        &mut g,
        edge_df,
        "KNOWS".to_string(),
        "Person".to_string(),
        "s".to_string(),
        "Person".to_string(),
        "d".to_string(),
        None,
        None,
        None,
    )
    .unwrap();
    g
}

/// Run a read query the way a non-lazy binding (Bolt / MCP / CLI) does, so the
/// rows are materialised rather than deferred behind a lazy descriptor.
fn rows_of(graph: &DirGraph, query: &str) -> Vec<Vec<(String, Value)>> {
    let params: HashMap<String, Value> = HashMap::new();
    // Non-lazy execution, the way the Bolt / MCP / CLI bindings drive it, so
    // every row is materialised instead of deferred behind a lazy descriptor.
    let opts = ExecuteOptions {
        params: &params,
        deadline: None,
        max_rows: None,
        lazy_eligible: false,
        disabled_passes: None,
        embedder: None,
        value_codecs: None,
        cancel: None,
        write_scope: None,
        git_sha: None,
        modified_by: None,
        csv_import: crate::graph::languages::cypher::executor::load_csv::CsvImportPolicy::default(),
        parallel: false,
    };
    let outcome = execute_read(graph, query, &opts).expect("query executes");
    outcome
        .result
        .rows
        .iter()
        .map(|row| {
            outcome
                .result
                .columns
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), row[i].clone()))
                .collect()
        })
        .collect()
}

fn single_node(graph: &DirGraph, query: &str) -> NodeValue {
    let rows = rows_of(graph, query);
    assert_eq!(rows.len(), 1, "expected exactly one row from `{query}`");
    match &rows[0][0].1 {
        Value::Node(n) => (**n).clone(),
        other => panic!("expected a Value::Node, got {other:?}"),
    }
}

fn s(v: &str) -> Value {
    Value::String(v.to_string())
}

/// The complete property map of node `pid = 0`, as a checked-in literal.
///
/// Each entry is a distinct rule of `collect_node_properties`:
///   `age`, `city` — ordinary stored properties
///   `id`          — the `id` virtual (canonical identity)
///   `title`       — the `title` virtual
///   `type`        — the `type` soft alias (structural type string)
///   `pid`         — the ID COLUMN ALIAS re-surfaced under its df name
///   `name`        — the TITLE COLUMN ALIAS re-surfaced under its df name
fn expected_node_0_properties() -> Vec<(&'static str, Value)> {
    vec![
        ("age", Value::Int64(20)),
        ("city", s("city_0")),
        ("id", Value::Int64(0)),
        ("name", s("P0")),
        ("pid", Value::Int64(0)),
        ("title", s("P0")),
        ("type", s("Person")),
    ]
}

/// Assert a `NodeValue`'s property map equals the literal exactly — key set,
/// key ORDER and values. Not `assert_eq!(map, other_map)` alone, because key
/// order is half of what this pins and a map compare would still pass if the
/// container stopped being ordered.
fn assert_properties_exactly(node: &NodeValue, expected: &[(&str, Value)], what: &str) {
    let got_keys: Vec<&str> = node.properties.keys().collect();
    let want_keys: Vec<&str> = expected.iter().map(|(k, _)| *k).collect();
    assert_eq!(
        got_keys, want_keys,
        "{what}: property key set/ORDER changed. N2 must pass against this \
         record unchanged."
    );
    for (k, want) in expected {
        let got = node
            .properties
            .get(k)
            .unwrap_or_else(|| panic!("{what}: key `{k}` vanished"));
        assert_eq!(got, want, "{what}: value for key `{k}` changed");
    }
}

#[test]
fn return_node_record_golden() {
    let g = projection_graph();
    let node = single_node(&g, "MATCH (n:Person) WHERE n.pid = 0 RETURN n");

    // ---- CHECKED-IN EXPECTATION -------------------------------------------
    assert_eq!(
        node.labels,
        vec!["Person".to_string()],
        "label set changed for a node with no secondary labels"
    );
    assert_properties_exactly(&node, &expected_node_0_properties(), "RETURN n");
    // ------------------------------------------------------------------------

    // `id` is the petgraph index, not the user's `pid`. They coincide here
    // (creation order == pid order), which is why the record carries BOTH and
    // why an alias regression is easy to miss without a literal.
    assert_eq!(node.id, 0, "NodeValue.id is the petgraph node index");
}

#[test]
fn return_node_omits_null_properties_golden() {
    let g = projection_graph();
    let node = single_node(&g, "MATCH (n:Person) WHERE n.pid = 5 RETURN n");

    // ---- CHECKED-IN EXPECTATION (city absent, everything else present) -----
    let expected = vec![
        ("age", Value::Int64(25)),
        ("id", Value::Int64(5)),
        ("name", s("P5")),
        ("pid", Value::Int64(5)),
        ("title", s("P5")),
        ("type", s("Person")),
    ];
    // ------------------------------------------------------------------------

    assert!(
        !node.properties.contains_key("city"),
        "a null property must be OMITTED from the materialised node, not \
         stored as Value::Null"
    );
    assert_properties_exactly(&node, &expected, "RETURN n (null row)");
}

#[test]
fn return_node_excludes_reserved_provenance_keys_golden() {
    let g = projection_graph();
    let rows = rows_of(&g, "MATCH (n:Person) RETURN n");
    assert_eq!(rows.len(), N_NODES as usize);
    for row in &rows {
        let Value::Node(node) = &row[0].1 else {
            panic!("expected Value::Node");
        };
        for key in node.properties.keys() {
            assert!(
                !crate::graph::schema::is_reserved_provenance_key(key),
                "reserved provenance key `{key}` leaked into a materialised node"
            );
        }
    }
}

/// `keys(n)` and `properties(n)` share one collection pass so they cannot
/// drift. Asserted here on the Rust side over every node of the fixture — the
/// Python golden covers the wider type corpus.
#[test]
fn keys_equals_keys_of_properties_golden() {
    let g = projection_graph();
    let rows = rows_of(
        &g,
        "MATCH (n:Person) RETURN keys(n) AS k, properties(n) AS p",
    );
    assert_eq!(rows.len(), N_NODES as usize);
    for row in &rows {
        let Value::List(keys) = &row[0].1 else {
            panic!("keys(n) must return a List, got {:?}", row[0].1);
        };
        let Value::Map(props) = &row[1].1 else {
            panic!("properties(n) must return a Map, got {:?}", row[1].1);
        };
        let key_names: Vec<&str> = keys
            .iter()
            .map(|v| match v {
                Value::String(s) => s.as_str(),
                other => panic!("keys(n) yielded a non-string: {other:?}"),
            })
            .collect();
        let prop_names: Vec<&str> = props.keys().collect();
        assert_eq!(
            key_names, prop_names,
            "keys(n) != keys(properties(n)) — the shared collection pass has \
             forked between its names-only and value sinks"
        );
    }
}

/// The relationship record, pinned the same way.
#[test]
fn return_relationship_record_golden() {
    let g = projection_graph();
    let rows = rows_of(
        &g,
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.pid = 0 AND b.pid = 1 RETURN r",
    );
    assert_eq!(rows.len(), 1);
    let Value::Relationship(rel) = &rows[0][0].1 else {
        panic!("expected a Value::Relationship, got {:?}", rows[0][0].1);
    };

    // ---- CHECKED-IN EXPECTATION -------------------------------------------
    assert_eq!(rel.rel_type, "KNOWS");
    assert_eq!(rel.start_id, 0);
    assert_eq!(rel.end_id, 1);
    let got: Vec<(&str, &Value)> = rel.properties.iter().collect();
    assert_eq!(
        got,
        vec![("since", &Value::Int64(2020))],
        "relationship property map changed"
    );
    // ------------------------------------------------------------------------
}

/// The path record: nodes and rels, each identical to its standalone form.
#[test]
fn return_path_record_golden() {
    let g = projection_graph();
    let rows = rows_of(
        &g,
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person) WHERE a.pid = 0 AND b.pid = 1 RETURN p",
    );
    assert_eq!(rows.len(), 1);
    let Value::Path(path) = &rows[0][0].1 else {
        panic!("expected a Value::Path, got {:?}", rows[0][0].1);
    };

    // ---- CHECKED-IN EXPECTATION -------------------------------------------
    assert_eq!(path.nodes.len(), 2, "a one-hop path carries k+1 = 2 nodes");
    assert_eq!(path.rels.len(), 1, "a one-hop path carries k = 1 rel");
    // ------------------------------------------------------------------------

    // The node nested in the path must be identical to the standalone one —
    // same materialisation, no path-specific shortcut.
    let standalone = single_node(&g, "MATCH (n:Person) WHERE n.pid = 0 RETURN n");
    assert_eq!(
        path.nodes[0], standalone,
        "the node nested in a path differs from the same node returned alone"
    );
    assert_properties_exactly(
        &path.nodes[0],
        &expected_node_0_properties(),
        "path.nodes[0]",
    );

    let expected_rel = RelValue {
        id: path.rels[0].id,
        start_id: 0,
        end_id: 1,
        rel_type: "KNOWS".to_string(),
        properties: [("since".to_string(), Value::Int64(2020))]
            .into_iter()
            .collect(),
    };
    assert_eq!(
        path.rels[0], expected_rel,
        "the relationship nested in a path changed shape"
    );
}

/// Mixed-type projection — the retirement record's unused pre-mortem scenario.
/// `RETURN n, id(n)` produces the node record and a scalar derived from it via
/// a different code path; they must agree, and the record must not degrade.
#[test]
fn mixed_type_projection_golden() {
    let g = projection_graph();
    let rows = rows_of(&g, "MATCH (n:Person) RETURN n, id(n) AS nid ORDER BY id(n)");
    assert_eq!(rows.len(), N_NODES as usize);
    for row in &rows {
        assert_eq!(
            row.iter().map(|(c, _)| c.as_str()).collect::<Vec<_>>(),
            vec!["n", "nid"],
            "result column ORDER changed"
        );
        let Value::Node(node) = &row[0].1 else {
            panic!("the node column degraded to {:?}", row[0].1);
        };
        let nid = match &row[1].1 {
            Value::Int64(v) => *v as u32,
            Value::UniqueId(v) => *v,
            other => panic!("id(n) returned {other:?}"),
        };
        assert_eq!(
            node.id, nid,
            "the materialised record's `id` disagrees with id(n) — two code \
             paths for the same identity have diverged"
        );
    }
}

/// Coverage guard for the `keys(n) == keys(properties(n))` corpus.
///
/// # What this pins, and why it is not what the plan assumed
///
/// [`collect_node_properties`] has a branch gated on
/// `NodeView::properties_are_columnar()` (the columnar-completion pass). The
/// Part N brief carried a "C2 lesson" premise: that a Cypher-`CREATE`d node on
/// **mapped** mode is Map-stored rather than columnar, so a corpus built only
/// from bulk `add_nodes` would silently test one branch twice.
///
/// **That premise no longer reproduces on this branch (0.16.4).** Measured
/// here across both backends: bulk-loaded and Cypher-`CREATE`d nodes are
/// *both* columnar, on Memory and on Mapped alike — consistent with the
/// shipped shape-convergence program ("one columnar shape").
///
/// So this test pins the fact rather than the stale assumption: **no node
/// reachable from either construction path is non-columnar.** If that ever
/// stops being true — a Map-stored branch returns — this goes red, and the
/// `keys(n)` corpus in `tests/test_node_record_golden.py` must grow a case
/// covering it before the branch ships. That is the coverage guarantee the
/// brief was reaching for, stated against measured behaviour.
#[test]
fn keys_invariant_holds_across_both_property_storage_shapes() {
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
    use crate::graph::storage::GraphRead;

    let params: HashMap<String, Value> = HashMap::new();
    let opts = ExecuteOptions {
        params: &params,
        deadline: None,
        max_rows: None,
        lazy_eligible: false,
        disabled_passes: None,
        embedder: None,
        value_codecs: None,
        cancel: None,
        write_scope: None,
        git_sha: None,
        modified_by: None,
        csv_import: crate::graph::languages::cypher::executor::load_csv::CsvImportPolicy::default(),
        parallel: false,
    };

    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        let mut g = new_dir_graph_in_mode(mode, None).expect("create graph");

        // Path 1: bulk `add_nodes` (aliased id/title, columnar).
        let columns = vec!["pid".to_string(), "name".to_string(), "age".to_string()];
        let rows: Vec<Vec<Value>> = (0..3i64)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("P{i}")),
                    Value::Int64(20 + i),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(columns, rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut g,
            df,
            "Person".to_string(),
            "pid".to_string(),
            Some("name".to_string()),
            None,
        )
        .unwrap();

        // Path 2: Cypher CREATE (no aliases) — the other construction route.
        crate::graph::session::execute_mut(
            &mut g,
            "CREATE (:Gadget {gid: 1, label: 'g1', weight: 2.5})",
            &opts,
        )
        .expect("CREATE executes");

        let mut non_columnar = Vec::new();
        for idx in g.graph.node_indices() {
            if let Some(view) = g.graph.node_view(idx) {
                if !view.properties_are_columnar() {
                    non_columnar.push(idx.index());
                }
            }
        }
        assert!(
            non_columnar.is_empty(),
            "mode={mode:?}: nodes {non_columnar:?} are NOT columnar. The \
             Map-stored branch of `collect_node_properties` is live again — \
             extend the keys(n) corpus in tests/test_node_record_golden.py to \
             cover it before relying on that branch."
        );

        // The invariant itself, on both construction paths.
        for label in ["Person", "Gadget"] {
            let rows = rows_of(
                &g,
                &format!("MATCH (n:{label}) RETURN keys(n) AS k, properties(n) AS p"),
            );
            assert!(!rows.is_empty(), "mode={mode:?}: no {label} nodes");
            for row in &rows {
                let Value::List(keys) = &row[0].1 else {
                    panic!("keys(n) must return a List");
                };
                let Value::Map(props) = &row[1].1 else {
                    panic!("properties(n) must return a Map");
                };
                let key_names: Vec<&str> = keys
                    .iter()
                    .map(|v| match v {
                        Value::String(s) => s.as_str(),
                        other => panic!("keys(n) yielded a non-string: {other:?}"),
                    })
                    .collect();
                assert_eq!(
                    key_names,
                    props.keys().collect::<Vec<_>>(),
                    "mode={mode:?}: keys(n) != keys(properties(n)) for a {label} node"
                );
            }
        }
    }
}
