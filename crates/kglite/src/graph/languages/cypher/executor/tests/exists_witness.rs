//! The `EXISTS { … }` witness cap — which subqueries may stop at one match.
//!
//! `Predicate::Exists` is the single evaluation site for every EXISTS spelling
//! (bare pattern predicate, `NOT EXISTS`, a projected `RETURN EXISTS {…}`,
//! `CASE WHEN EXISTS`), so the cap decision is one function and these are its
//! goldens. Non-vacuity: turning the `Some(1)` return into `None` turns
//! [`a_plain_single_pattern_subquery_stops_at_one_witness`] red.

use super::*;
use crate::graph::core::pattern_matching::parse_pattern;
use crate::graph::languages::cypher::result::EdgeBinding;
use petgraph::graph::{EdgeIndex, NodeIndex};

fn cap(pattern: &str, row: &ResultRow) -> Option<usize> {
    let parsed = parse_pattern(pattern).unwrap_or_else(|e| panic!("{pattern}: {e}"));
    CypherExecutor::exists_witness_cap(std::slice::from_ref(&parsed), &None, row)
}

#[test]
fn a_plain_single_pattern_subquery_stops_at_one_witness() {
    let mut row = ResultRow::new();
    row.node_bindings.insert("a".to_string(), NodeIndex::new(3));

    // Nothing about these can reject a returned match, so one settles it.
    assert_eq!(cap("(a)-[:R]->(b)", &row), Some(1));
    assert_eq!(cap("(a)-[:R*1..3]->(:Person)", &row), Some(1));
    assert_eq!(cap("(x:Person)-[:R]->(y:Person)", &row), Some(1));
}

#[test]
fn a_binding_the_executor_cannot_push_down_forbids_the_cap() {
    // A node variable carried only as a projected VALUE (`UNWIND collect(n)
    // AS n`, a folded `WITH n`) is enforced by `bindings_compatible` AFTER
    // the executor runs — a cap of one has no second candidate to survive.
    let mut projected_only = ResultRow::new();
    projected_only
        .projected
        .insert("b".to_string(), Value::NodeRef(7));
    assert_eq!(cap("(a)-[:R]->(b)", &projected_only), None);

    // Same for a relationship variable already bound on the row.
    let mut bound_edge = ResultRow::new();
    bound_edge.edge_bindings.insert(
        "r".to_string(),
        EdgeBinding {
            source: NodeIndex::new(0),
            target: NodeIndex::new(1),
            edge_index: EdgeIndex::new(0),
        },
    );
    assert_eq!(cap("(a)-[r:R]->(b)", &bound_edge), None);

    // A node binding IS pushed down as a pre-binding, so it does not forbid it.
    let mut bound_node = ResultRow::new();
    bound_node
        .node_bindings
        .insert("b".to_string(), NodeIndex::new(7));
    assert_eq!(cap("(a)-[:R]->(b)", &bound_node), Some(1));
}

#[test]
fn a_join_or_an_inner_where_forbids_the_cap() {
    let row = ResultRow::new();
    let first = parse_pattern("(a)-[:R]->(b)").unwrap();
    let second = parse_pattern("(b)-[:S]->(c)").unwrap();

    // The arm joins the patterns: the first match of pattern one may be
    // incompatible with every match of pattern two.
    assert_eq!(
        CypherExecutor::exists_witness_cap(&[first.clone(), second], &None, &row),
        None
    );

    // The inner WHERE runs after the join, so the first witness may fail it
    // while a later one passes.
    let inner = Some(Box::new(Predicate::IsNotNull(Expression::PropertyAccess {
        variable: "b".to_string(),
        property: "name".to_string(),
    })));
    assert_eq!(
        CypherExecutor::exists_witness_cap(std::slice::from_ref(&first), &inner, &row),
        None
    );
}
