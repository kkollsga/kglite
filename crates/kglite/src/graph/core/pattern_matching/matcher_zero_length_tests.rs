//! Zero-length variable-length paths, and the undirected self-loop guard.
//!
//! Split out of `matcher.rs` to keep that file under the source-quality line
//! ceiling, matching `matcher_id_lookup_tests.rs`.
//!
//! Two contracts, both about relationship expansion:
//!
//! * A `*0..` segment always yields the zero-length path binding both
//!   endpoints to the same node. The path has **no relationship**, so nothing
//!   the segment says about relationship types applies to it — an unknown
//!   type must not suppress it. `expand_from_node`'s unknown-type early exit
//!   used to run ahead of the variable-length dispatch and returned no rows
//!   at all.
//! * `shortestPath` carries its own copy of the same contract: with `*0..`
//!   and both endpoints on the same node the answer is the zero-length path,
//!   not "no path". Its endpoint loop skipped `source == target` outright.
//! * An undirected self-loop is **one** relationship binding despite being
//!   yielded by both incident directions. That guard is a `continue` in three
//!   separate expansion loops — exactly the shape that regrows when a fourth
//!   expansion site is added — so it is pinned here even though it is
//!   currently correct (T0-7, refuted on 0.17.0).

use super::*;
use crate::datatypes::Value;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

/// The single projected column of every row, in row order.
fn column(graph: &DirGraph, query: &str) -> Vec<Value> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let outcome = execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("read query failed: {query}: {e}"));
    assert!(
        outcome.result.lazy.is_none(),
        "eager options must materialize rows for {query}"
    );
    outcome
        .result
        .rows
        .iter()
        .map(|row| {
            row.first()
                .unwrap_or_else(|| panic!("no projected column for {query}"))
                .clone()
        })
        .collect()
}

/// `(a:P {id:1})-[:K]->(b:P {id:2})` — one relationship, of one type.
fn two_nodes_one_edge() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (a:P {id: 1}), (b:P {id: 2}), (a)-[:K]->(b)",
    );
    graph
}

#[test]
fn a_zero_length_path_survives_an_unknown_relationship_type() {
    let graph = two_nodes_one_edge();
    for pattern in [
        "(a:P {id: 1})-[:ZZZ*0..2]->(b)",
        "(a:P {id: 1})-[:ZZZ*0]->(b)",
        "(a:P {id: 1})-[:ZZZ*0..0]->(b)",
        "(a:P {id: 1})-[:ZZZ*0..2]-(b)",
        "(a:P {id: 1})<-[:ZZZ*0..2]-(b)",
        "(a:P {id: 1})-[:ZZZ|YYY*0..2]->(b)",
    ] {
        assert_eq!(
            column(&graph, &format!("MATCH {pattern} RETURN b.id")),
            vec![Value::Int64(1)],
            "{pattern} must yield the zero-length path binding b to a"
        );
    }
}

#[test]
fn an_unknown_type_still_yields_nothing_from_one_hop_up() {
    let graph = two_nodes_one_edge();
    for pattern in [
        "(a:P {id: 1})-[:ZZZ*1..2]->(b)",
        "(a:P {id: 1})-[:ZZZ*1]->(b)",
        "(a:P {id: 1})-[:ZZZ*2..3]-(b)",
        "(a:P {id: 1})-[:ZZZ]->(b)",
    ] {
        assert!(
            column(&graph, &format!("MATCH {pattern} RETURN b.id")).is_empty(),
            "{pattern} must yield nothing: the type holds no relationship"
        );
    }
}

#[test]
fn a_known_type_zero_length_path_still_reaches_every_hop() {
    let graph = two_nodes_one_edge();
    assert_eq!(
        column(&graph, "MATCH (a:P {id: 1})-[:K*0..2]->(b) RETURN b.id"),
        vec![Value::Int64(1), Value::Int64(2)]
    );
    assert_eq!(
        column(&graph, "MATCH (a:P {id: 1})-[*0..2]->(b) RETURN b.id"),
        vec![Value::Int64(1), Value::Int64(2)]
    );
}

#[test]
fn a_zero_length_path_of_an_unknown_type_has_no_relationships() {
    let graph = two_nodes_one_edge();
    // `r` binds a path value rather than a relationship list here (so
    // `size(r)` is null for *every* hop count and type — unrelated to the
    // zero-length rule); `relationships(p)` is the list.
    assert_eq!(
        column(
            &graph,
            "MATCH p = (a:P {id: 1})-[r:ZZZ*0..2]->(b) RETURN size(relationships(p))"
        ),
        vec![Value::Int64(0)]
    );
    assert_eq!(
        column(
            &graph,
            "MATCH p = (a:P {id: 1})-[:ZZZ*0..2]->(b) RETURN length(p)"
        ),
        vec![Value::Int64(0)]
    );
}

#[test]
fn the_zero_length_path_still_honours_the_target_pattern() {
    let mut graph = two_nodes_one_edge();
    run(&mut graph, "CREATE (:Q {id: 9})");
    // The zero-length path binds the source to itself, so a target label the
    // source does not carry must still reject it.
    assert!(column(&graph, "MATCH (a:P {id: 1})-[:ZZZ*0..2]->(b:Q) RETURN b.id").is_empty());
    assert_eq!(
        column(&graph, "MATCH (a:P {id: 1})-[:ZZZ*0..2]->(b:P) RETURN b.id"),
        vec![Value::Int64(1)]
    );
}

#[test]
fn an_undirected_self_loop_binds_one_relationship() {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (a:P {id: 1}), (a)-[:K {k: 0}]->(a)");
    // Every construction of the same undirected traversal — labelled or not,
    // typed or not, bound relationship or anonymous, fixed or variable
    // length — sees the loop once, not once per incident direction.
    for pattern in ["(a:P)-[r:K]-(b:P)", "(a)-[r:K]-(b)", "(a:P)-[r]-(b:P)"] {
        assert_eq!(
            column(&graph, &format!("MATCH {pattern} RETURN r.k")),
            vec![Value::Int64(0)],
            "{pattern} must bind the loop exactly once"
        );
    }
    for pattern in [
        "(a:P)-[:K]-(b:P)",
        "p = (a:P)-[:K]-(b:P)",
        "(a:P)-[:K*1..1]-(b:P)",
        // A var-length relationship variable binds the *list*, so it is
        // counted rather than read for a property.
        "(a:P)-[r:K*1..1]-(b:P)",
    ] {
        assert_eq!(
            column(&graph, &format!("MATCH {pattern} RETURN count(*)")),
            vec![Value::Int64(1)],
            "{pattern} must produce exactly one row"
        );
    }
}

#[test]
fn an_undirected_self_loop_beside_an_ordinary_edge_keeps_both_orientations() {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (a:P {id: 1}), (b:P {id: 2}), (a)-[:K {k: 0}]->(a), (a)-[:K {k: 1}]->(b)",
    );
    // The loop once; the ordinary edge in both orientations.
    assert_eq!(
        column(
            &graph,
            "MATCH (a:P)-[r:K]-(b:P) RETURN a.id * 100 + r.k * 10 + b.id \
             ORDER BY a.id, r.k, b.id"
        ),
        vec![Value::Int64(101), Value::Int64(112), Value::Int64(211)]
    );
}

#[test]
fn shortest_path_answers_the_zero_length_path_between_a_node_and_itself() {
    let graph = two_nodes_one_edge();
    for pattern in [
        "shortestPath((a)-[:K*0..]-(a))",
        "shortestPath((a)-[*0..]-(a))",
        "shortestPath((a)-[:ZZZ*0..]-(a))",
        "shortestPath((a)-[:K*0..3]->(a))",
    ] {
        assert_eq!(
            column(
                &graph,
                &format!("MATCH (a:P {{id: 1}}) MATCH p = {pattern} RETURN length(p)")
            ),
            vec![Value::Int64(0)],
            "{pattern} must answer the zero-length path"
        );
    }
}

#[test]
fn shortest_path_still_refuses_a_min_one_walk_back_to_the_same_node() {
    let graph = two_nodes_one_edge();
    for pattern in [
        "shortestPath((a)-[:K*1..]-(a))",
        "shortestPath((a)-[*]-(a))",
        "shortestPath((a)-[:K*2..3]-(a))",
    ] {
        assert!(
            column(
                &graph,
                &format!("MATCH (a:P {{id: 1}}) MATCH p = {pattern} RETURN length(p)")
            )
            .is_empty(),
            "{pattern} must answer no path: a trail cannot return to its start here"
        );
    }
}

#[test]
fn shortest_path_between_distinct_nodes_is_unchanged() {
    let graph = two_nodes_one_edge();
    assert_eq!(
        column(
            &graph,
            "MATCH (a:P {id: 1}), (c:P {id: 2}) MATCH p = shortestPath((a)-[:K*0..]-(c)) \
             RETURN length(p)"
        ),
        vec![Value::Int64(1)]
    );
}
