//! Executor unit tests.
//!
//! Split into a module directory when the flat file reached the 2500-line
//! source-quality ceiling. Shared helpers and the imports every submodule
//! needs live here; each submodule pulls them in with `use super::*`.
//!
//! - [`expressions`] — comparison, arithmetic, coercion, CASE, parameters
//! - [`mutations`] — CREATE / SET / DELETE / REMOVE / MERGE and index upkeep
//! - [`lists`] — list parsing, slicing, sizing, and quantifier predicates
//! - [`strings`] — string functions and procedure list arguments
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

mod expressions;
mod lists;
mod mutations;
mod strings;

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
fn periodic_interrupt_reaches_union_and_subquery_join_inner_loops() {
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
