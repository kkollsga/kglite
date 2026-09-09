//! Executor unit tests.
//!
//! Split into a module directory when the flat file reached the 2500-line
//! source-quality ceiling. Shared helpers and the imports every submodule
//! needs live here; each submodule pulls them in with `use super::*`.
//!
//! - [`exists_witness`] — which `EXISTS { … }` subqueries may stop at one match
//! - [`expressions`] — comparison, arithmetic, coercion, CASE, parameters
//! - [`mutations`] — CREATE / SET / DELETE / REMOVE / MERGE and index upkeep
//! - [`identifiers`] — quoted-identifier escaping (the injection class)
//! - [`lists`] — list parsing, slicing, sizing, and quantifier predicates
//! - [`score_fuse`] — the retrieval-fusion scalar: absent lanes, weights, fold
//! - [`semantics`] — absolute goldens for expression semantics (duplicate
//!   result columns, `datetime()` time/zone, integer overflow and div-by-zero)
//! - [`strings`] — string functions and procedure list arguments
//! - [`text_bm25`] — the BM25 scalar's null/zero split, freshness policy, cache
//! - [`vector_score`] — the embedding-store scalar's per-query argument cache
//! - [`vectors`] — `dot` / `cosine` / `norm` over list-valued data
//! - [`deadline_rows`] — deadline/cancel polling inside the sequential MATCH
//!   row loops (match-to-row, comma-pattern join, driving-row join)
//! - [`parallel`] — deadline/cancel polling inside the rayon-parallel regions
#![allow(clippy::approx_constant)]

use super::helpers::*;
use super::write::execute_mutable;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::schema::{EdgeData, NodeData};
use crate::graph::storage::GraphWrite;
// The tests parse real Cypher, and `parser` is a sibling of `executor` rather
// than a child, so it is imported by absolute path — depth-independent, unlike
// the `super::super::parser` the flat file used before the split.
use crate::graph::languages::cypher::parser;

mod cypher25_clauses;
mod deadline_rows;
mod exists_witness;
mod expressions;
mod identifiers;
mod lists;
mod mutations;
mod parallel;
mod score_fuse;
mod self_loops;
mod semantics;
mod strings;
mod text_bm25;
mod vector_score;
mod vectors;

/// Test helper: unwraps evaluate_comparison Result for use in assert!()
pub(super) fn cmp(left: &Value, op: &ComparisonOp, right: &Value) -> bool {
    evaluate_comparison(left, op, right).unwrap()
}

pub(super) fn projected_rows(name: &str, count: usize) -> Vec<ResultRow> {
    (0..count)
        .map(|i| {
            let mut row = ResultRow::new();
            row.projected
                .insert(name.to_string(), Value::Int64(i as i64));
            row
        })
        .collect()
}

/// Helper: build a small test graph with 2 Person nodes and 1 KNOWS edge
pub(super) fn build_test_graph() -> DirGraph {
    let mut graph = DirGraph::new();
    let alice = NodeData::new(
        Value::UniqueId(1),
        Value::String("Alice".to_string()),
        "Person".to_string(),
        HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
            ("age".to_string(), Value::Int64(30)),
        ]),
        &mut graph.interner,
    );
    let bob = NodeData::new(
        Value::UniqueId(2),
        Value::String("Bob".to_string()),
        "Person".to_string(),
        HashMap::from([
            ("name".to_string(), Value::String("Bob".to_string())),
            ("age".to_string(), Value::Int64(25)),
        ]),
        &mut graph.interner,
    );
    let alice_idx = graph.graph.add_node(alice);
    let bob_idx = graph.graph.add_node(bob);
    graph
        .type_indices
        .entry_or_default("Person".to_string())
        .push(alice_idx);
    graph
        .type_indices
        .entry_or_default("Person".to_string())
        .push(bob_idx);

    let edge = EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(alice_idx, bob_idx, edge);
    graph.register_connection_type("KNOWS".to_string());

    graph
}

// Interrupt plumbing is executor-wide rather than per-clause, so it stays at
// the module root next to the helpers it shares.
#[test]
fn periodic_interrupt_reaches_range_unwind_and_single_group_aggregate() {
    let graph = DirGraph::new();
    let mut params = HashMap::new();
    params.insert(
        "items".to_string(),
        Value::List((0..8193).map(Value::Int64).collect()),
    );
    let executor = CypherExecutor::with_params(&graph, &params, None);

    CypherExecutor::interrupt_after_periodic_polls(1);
    let range_args = [
        Expression::Literal(Value::Int64(0)),
        Expression::Literal(Value::Int64(8192)),
    ];
    assert!(executor
        .test_eval_collection_fn("range", &range_args, &ResultRow::new())
        .unwrap_err()
        .contains("test hook"));

    CypherExecutor::interrupt_after_periodic_polls(1);
    let unwind =
        super::super::parser::parse_cypher("UNWIND $items AS item RETURN count(item) AS n")
            .unwrap();
    assert!(executor.execute(&unwind).unwrap_err().contains("test hook"));

    let aggregate_query = super::super::parser::parse_cypher("RETURN collect(x) AS xs").unwrap();
    let Clause::Return(return_clause) = &aggregate_query.clauses[0] else {
        panic!("expected RETURN clause");
    };
    let rows = projected_rows("x", 8193);
    let refs: Vec<&ResultRow> = rows.iter().collect();
    CypherExecutor::interrupt_after_periodic_polls(1);
    assert!(executor
        .evaluate_aggregate_with_rows(&return_clause.items[0].expression, &refs)
        .unwrap_err()
        .contains("test hook"));
}

#[test]
fn periodic_interrupt_reaches_union_and_single_row_subquery_join_loops() {
    let graph = DirGraph::new();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);

    let union_query =
        super::super::parser::parse_cypher("RETURN 1 AS x UNION RETURN 1 AS x").unwrap();
    let Clause::Union(union_clause) = &union_query.clauses[1] else {
        panic!("expected UNION clause");
    };
    let left = ResultSet {
        rows: projected_rows("x", 8193),
        columns: vec!["x".to_string()],
        lazy_return_items: None,
    };
    CypherExecutor::interrupt_after_periodic_polls(1);
    assert!(executor
        .execute_union(union_clause, left)
        .unwrap_err()
        .contains("test hook"));

    let call_query =
        super::super::parser::parse_cypher("CALL { RETURN 1 AS inner_value } RETURN inner_value")
            .unwrap();
    let Clause::CallSubquery { import, body } = &call_query.clauses[0] else {
        panic!("expected CALL subquery clause");
    };
    let outer = ResultSet {
        rows: projected_rows("outer_value", 8193),
        columns: vec!["outer_value".to_string()],
        lazy_return_items: None,
    };
    CypherExecutor::interrupt_after_periodic_polls(1);
    assert!(executor
        .execute_call_subquery(import, body, outer, &std::collections::HashSet::new())
        .unwrap_err()
        .contains("test hook"));
}

#[test]
fn call_subquery_set_arms_share_outer_seed_without_sharing_arm_results() {
    let graph = DirGraph::new();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);

    let mut modern = parser::parse_cypher(
        "UNWIND [1, 2] AS x CALL (x) { RETURN x AS n UNION ALL RETURN x + 10 AS n } \
         RETURN x, n ORDER BY x, n",
    )
    .unwrap();
    crate::graph::languages::cypher::planner::optimize(&mut modern, &graph, &params);
    let result = executor.execute(&modern).unwrap();
    assert_eq!(result.columns, vec!["x", "n"]);
    assert_eq!(
        result.rows,
        vec![
            vec![Value::Int64(1), Value::Int64(1)],
            vec![Value::Int64(1), Value::Int64(11)],
            vec![Value::Int64(2), Value::Int64(2)],
            vec![Value::Int64(2), Value::Int64(12)],
        ]
    );

    let mut legacy = parser::parse_cypher(
        "WITH 1 AS x, 2 AS y CALL { WITH x RETURN x AS n UNION ALL WITH y RETURN y AS n } \
         RETURN n ORDER BY n",
    )
    .unwrap();
    crate::graph::languages::cypher::planner::optimize(&mut legacy, &graph, &params);
    let result = executor.execute(&legacy).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![Value::Int64(1)], vec![Value::Int64(2)]]
    );

    for (operator, expected) in [
        ("UNION", vec![1]),
        ("UNION ALL", vec![1, 1]),
        ("INTERSECT", vec![1]),
        ("EXCEPT", Vec::new()),
    ] {
        let query = parser::parse_cypher(&format!(
            "CALL () {{ RETURN 1 AS n {operator} RETURN 1 AS n }} RETURN n"
        ))
        .unwrap();
        let result = executor.execute(&query).unwrap();
        let values: Vec<i64> = result
            .rows
            .iter()
            .map(|row| match row.as_slice() {
                [Value::Int64(value)] => *value,
                other => panic!("unexpected set row: {other:?}"),
            })
            .collect();
        assert_eq!(values, expected, "{operator}");
    }
}

#[test]
fn empty_outer_call_subquery_keeps_its_static_output_schema() {
    let graph = DirGraph::new();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);

    let cases = [
        ("MATCH (n:Missing) CALL (n) { RETURN n AS z }", vec!["z"]),
        ("MATCH (n:Missing) CALL () { RETURN 1 AS z }", vec!["z"]),
        ("MATCH (n:Missing) CALL { RETURN 1 AS z }", vec!["z"]),
        (
            "MATCH (n:Missing) WITH n AS kept CALL (kept) { RETURN kept AS z }",
            vec!["kept", "z"],
        ),
        (
            "MATCH (n:Missing) CALL () { RETURN 1 AS b, 2 AS a }",
            vec!["b", "a"],
        ),
    ];
    for (source, expected) in cases {
        let query = parser::parse_cypher(source).unwrap();
        let result = executor.execute(&query).unwrap();
        assert!(result.rows.is_empty(), "{source}");
        assert_eq!(result.columns, expected, "{source}");
    }

    for operator in ["UNION", "UNION ALL", "INTERSECT", "EXCEPT"] {
        let source = format!(
            "MATCH (:Missing) CALL () {{ RETURN 1 AS b, 2 AS a {operator} RETURN 3 AS b, 4 AS a }}"
        );
        let query = parser::parse_cypher(&source).unwrap();
        let result = executor.execute(&query).unwrap();
        assert!(result.rows.is_empty(), "{operator}");
        assert_eq!(result.columns, vec!["b", "a"], "{operator}");
    }

    for (source, expected) in [
        (
            "WITH 1 AS x CALL () { MATCH (:Missing) RETURN 1 AS z }",
            vec!["x", "z"],
        ),
        (
            "WITH 1 AS seed CALL () { MATCH (n:Missing) RETURN * }",
            vec!["seed", "n"],
        ),
        (
            "WITH 1 AS seed CALL () { WITH 2 AS a FILTER false RETURN * }",
            vec!["seed", "a"],
        ),
        (
            "WITH 1 AS seed CALL () { UNWIND [] AS a RETURN * }",
            vec!["seed", "a"],
        ),
        (
            "WITH 1 AS seed FILTER false CALL () { WITH 2 AS b, 3 AS a RETURN * \
             UNION ALL WITH 4 AS b, 5 AS a RETURN * }",
            vec!["seed", "b", "a"],
        ),
    ] {
        let query = parser::parse_cypher(source).unwrap();
        let result = executor.execute(&query).unwrap();
        assert!(result.rows.is_empty(), "{source}");
        assert_eq!(result.columns, expected, "{source}");
    }

    for (source, collision) in [
        (
            "WITH 1 AS a FILTER false CALL () { WITH 2 AS a RETURN * }",
            "`a`",
        ),
        (
            "WITH 1 AS x FILTER false CALL (x) { WITH 2 AS a RETURN * }",
            "`x`",
        ),
    ] {
        let query = parser::parse_cypher(source).unwrap();
        let error = executor.execute(&query).unwrap_err();
        assert!(error.contains(collision), "{source}: {error}");
    }

    let error = parser::parse_cypher(
        "WITH 1 AS seed FILTER false CALL () { WITH 2 AS a RETURN * \
         UNION ALL WITH 2 AS b RETURN * }",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("same return column names"), "{error}");
}

#[test]
fn subquery_return_star_schema_is_independent_of_rows_and_binding_families() {
    let empty = DirGraph::new();
    let populated = build_test_graph();
    let params = HashMap::new();
    let path_query = "CALL () { MATCH p=(a)-[r:KNOWS]->(b) RETURN * }";

    let empty_result = CypherExecutor::with_params(&empty, &params, None)
        .execute(&parser::parse_cypher(path_query).unwrap())
        .unwrap();
    assert_eq!(empty_result.columns, vec!["a", "r", "b", "p"]);
    assert!(empty_result.rows.is_empty());

    let populated_executor = CypherExecutor::with_params(&populated, &params, None);
    let path_result = populated_executor
        .execute(&parser::parse_cypher(path_query).unwrap())
        .unwrap();
    assert_eq!(path_result.columns, vec!["a", "r", "b", "p"]);
    assert_eq!(path_result.rows.len(), 1);
    assert!(matches!(path_result.rows[0][0], Value::Node(_)));
    assert!(matches!(path_result.rows[0][1], Value::Relationship(_)));
    assert!(matches!(path_result.rows[0][2], Value::Node(_)));
    assert!(matches!(path_result.rows[0][3], Value::Path(_)));

    let filtered = populated_executor
        .execute(
            &parser::parse_cypher("CALL () { MATCH p=(a)-[r:KNOWS]->(b) FILTER false RETURN * }")
                .unwrap(),
        )
        .unwrap();
    assert_eq!(filtered.columns, vec!["a", "r", "b", "p"]);
    assert!(filtered.rows.is_empty());

    for graph in [&empty, &populated] {
        let result = CypherExecutor::with_params(graph, &params, None)
            .execute(
                &parser::parse_cypher("CALL () { MATCH (a)-[r:KNOWS]->(b) RETURN * }").unwrap(),
            )
            .unwrap();
        assert_eq!(result.columns, vec!["a", "r", "b"]);
    }

    for (yield_items, expected) in [
        ("node AS n, score AS s", vec!["n", "s"]),
        ("score AS s, node AS n", vec!["s", "n"]),
    ] {
        let source = format!("CALL () {{ CALL pagerank() YIELD {yield_items} RETURN * }}");
        for graph in [&empty, &populated] {
            let result = CypherExecutor::with_params(graph, &params, None)
                .execute(&parser::parse_cypher(&source).unwrap())
                .unwrap();
            assert_eq!(result.columns, expected, "{source}");
        }
    }

    let nested = populated_executor
        .execute(
            &parser::parse_cypher(
                "CALL () { CALL () { MATCH p=(a)-[r:KNOWS]->(b) RETURN * } RETURN * }",
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(nested.columns, vec!["a", "r", "b", "p"]);
    assert!(matches!(nested.rows[0][3], Value::Path(_)));

    let nested_explicit = populated_executor
        .execute(
            &parser::parse_cypher(
                "CALL () { CALL () { WITH 2 AS b, 3 AS a RETURN b, a } RETURN * }",
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(nested_explicit.columns, vec!["b", "a"]);
}

#[test]
fn subquery_return_star_set_arms_preserve_paths_and_distinct() {
    let populated = build_test_graph();
    let params = HashMap::new();

    for (operator, populated_rows) in [
        ("UNION ALL", 2),
        ("UNION", 1),
        ("INTERSECT", 1),
        ("EXCEPT", 0),
    ] {
        for (left, right) in [
            ("RETURN a, r, b, p", "RETURN *"),
            ("RETURN *", "RETURN a, r, b, p"),
            ("RETURN *", "RETURN *"),
        ] {
            let source = format!(
                "CALL () {{ MATCH p=(a)-[r:KNOWS]->(b) {left} \
                 {operator} MATCH p=(a)-[r:KNOWS]->(b) {right} }}"
            );
            for (graph, expected_rows) in [(&DirGraph::new(), 0), (&populated, populated_rows)] {
                let result = CypherExecutor::with_params(graph, &params, None)
                    .execute(&parser::parse_cypher(&source).unwrap())
                    .unwrap();
                assert_eq!(result.columns, vec!["a", "r", "b", "p"], "{source}");
                assert_eq!(result.rows.len(), expected_rows, "{source}");
                assert!(
                    result
                        .rows
                        .iter()
                        .all(|row| matches!(row[3], Value::Path(_))),
                    "{source}"
                );
            }
        }
    }

    let mut parallel = build_test_graph();
    let duplicate = EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut parallel.interner);
    parallel.graph.add_edge(
        petgraph::graph::NodeIndex::new(0),
        petgraph::graph::NodeIndex::new(1),
        duplicate,
    );
    let distinct = CypherExecutor::with_params(&parallel, &params, None)
        .execute(
            &parser::parse_cypher("CALL () { MATCH p=(a)-[:KNOWS]->(b) RETURN DISTINCT * }")
                .unwrap(),
        )
        .unwrap();
    assert_eq!(distinct.columns, vec!["a", "b", "p"]);
    assert_eq!(distinct.rows.len(), 2);
    assert!(distinct
        .rows
        .iter()
        .all(|row| matches!(row[2], Value::Path(_))));
}

#[test]
fn subquery_return_star_scope_and_set_order_are_static() {
    let empty = DirGraph::new();
    let populated = build_test_graph();
    let params = HashMap::new();

    for (items, expected) in [("[1]", 1), ("[]", 0)] {
        let source = format!("CALL () {{ WITH {items} AS xs UNWIND xs AS x RETURN * }}");
        let result = CypherExecutor::with_params(&empty, &params, None)
            .execute(&parser::parse_cypher(&source).unwrap())
            .unwrap();
        assert_eq!(result.columns, vec!["xs", "x"]);
        assert_eq!(result.rows.len(), expected);
    }

    for (left, right) in [
        ("RETURN *", "RETURN n, s"),
        ("RETURN n, s", "RETURN *"),
        ("RETURN *", "RETURN *"),
    ] {
        let source = format!(
            "CALL () {{ CALL pagerank() YIELD node AS n, score AS s {left} \
             UNION ALL CALL pagerank() YIELD node AS n, score AS s {right} }}"
        );
        for (graph, expected_rows) in [(&empty, 0), (&populated, 4)] {
            let result = CypherExecutor::with_params(graph, &params, None)
                .execute(&parser::parse_cypher(&source).unwrap())
                .unwrap();
            assert_eq!(result.columns, vec!["n", "s"], "{source}");
            assert_eq!(result.rows.len(), expected_rows, "{source}");
        }
    }

    let graph_stats = CypherExecutor::with_params(&empty, &params, None)
        .execute(
            &parser::parse_cypher(
                "CALL () { CALL db.graph_stats() YIELD edge_count AS e, node_count AS n \
                 RETURN * }",
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(graph_stats.columns, vec!["e", "n"]);

    let error = parser::parse_cypher(
        "CALL () { CALL pagerank() YIELD node AS n, score AS s RETURN * \
         UNION ALL CALL pagerank() YIELD score AS s, node AS n RETURN * }",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("same return column names"), "{error}");

    for variable in ["a", "r", "p"] {
        for filter in ["", "FILTER false"] {
            let source = format!(
                "MATCH p=(a)-[r:KNOWS]->(b) {filter} \
                 CALL ({variable}) {{ WITH 1 AS local RETURN * }}"
            );
            let error = CypherExecutor::with_params(&populated, &params, None)
                .execute(&parser::parse_cypher(&source).unwrap())
                .unwrap_err();
            assert!(
                error.contains(&format!("`{variable}`")),
                "{source}: {error}"
            );
        }
    }

    for filter in ["", "FILTER false"] {
        let source = format!("WITH 1 AS x {filter} CALL (x) {{ WITH 2 AS local RETURN * }}");
        let error = CypherExecutor::with_params(&empty, &params, None)
            .execute(&parser::parse_cypher(&source).unwrap())
            .unwrap_err();
        assert!(error.contains("`x`"), "{source}: {error}");
    }

    let legacy = CypherExecutor::with_params(&populated, &params, None)
        .execute(
            &parser::parse_cypher(
                "MATCH p=(a)-[:KNOWS]->(b) CALL { WITH p WITH 1 AS local RETURN * }",
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(legacy.columns, vec!["local"]);

    let legacy_collision = CypherExecutor::with_params(&populated, &params, None)
        .execute(
            &parser::parse_cypher("MATCH p=(a)-[:KNOWS]->(b) CALL { WITH p RETURN * }").unwrap(),
        )
        .unwrap_err();
    assert!(legacy_collision.contains("`p`"), "{legacy_collision}");
}

#[test]
fn periodic_interrupt_reaches_seeded_call_subquery_right_arm() {
    let graph = DirGraph::new();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);
    let query = parser::parse_cypher(
        "CALL () { RETURN 0 AS x UNION ALL UNWIND range(1, 8193) AS x RETURN x } RETURN x",
    )
    .unwrap();

    CypherExecutor::interrupt_after_periodic_polls(1);
    assert!(executor.execute(&query).unwrap_err().contains("test hook"));
}

#[test]
fn call_subquery_set_decides_null_anchor_kind_per_arm() {
    let graph = build_test_graph();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);
    let mut query = parser::parse_cypher(
        "OPTIONAL MATCH (x:Nope) CALL (x) { RETURN coalesce(x, 'fallback') AS value \
         UNION ALL MATCH (x)-[:KNOWS]->(f) RETURN count(f) AS value } RETURN value",
    )
    .unwrap();
    crate::graph::languages::cypher::planner::optimize(&mut query, &graph, &params);

    let result = executor.execute(&query).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![Value::String("fallback".to_string())],
            vec![Value::Int64(0)],
        ]
    );
}
