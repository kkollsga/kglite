//! Pattern comprehension `[(a)-->(b) WHERE … | expr]`: one element per match of
//! the pattern, correlated with the enclosing row.

use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use std::collections::HashMap;

/// a -K{w:1}-> b, a -K{w:5}-> c, b -K{w:2}-> c, and an isolated d.
fn graph() -> DirGraph {
    let mut graph = DirGraph::new();
    let params = HashMap::new();
    execute_mut(
        &mut graph,
        "CREATE (a:P {name: 'a'})-[:K {w: 1}]->(b:P {name: 'b'}), \
         (a)-[:K {w: 5}]->(c:P {name: 'c'}), (b)-[:K {w: 2}]->(c), (:P {name: 'd'})",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    graph
}

fn run(graph: &DirGraph, query: &str) -> Result<Vec<Vec<Value>>, String> {
    let params = HashMap::new();
    execute_read(graph, query, &ExecuteOptions::eager(&params))
        .map(|outcome| outcome.result.rows)
        .map_err(|error| error.to_string())
}

fn text(value: &str) -> Value {
    Value::String(value.into())
}

/// The list in the first column of every row, sorted — match order is not
/// part of the contract.
fn sorted_lists(rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
    rows.into_iter()
        .map(|row| match row.into_iter().next() {
            Some(Value::List(mut items)) => {
                items.sort_by_key(|item| format!("{item:?}"));
                items
            }
            other => panic!("expected a list, got {other:?}"),
        })
        .collect()
}

#[test]
fn projects_one_element_per_correlated_match() {
    let graph = graph();
    for (query, expected) in [
        (
            "MATCH (n:P) RETURN [(n)-->(m) | m.name] AS r ORDER BY n.name",
            vec![vec![text("b"), text("c")], vec![text("c")], vec![], vec![]],
        ),
        (
            "MATCH (n:P) RETURN [(n)-[r:K]->(m) WHERE r.w > 1 | m.name + r.w] AS r ORDER BY n.name",
            vec![vec![text("c5")], vec![text("c2")], vec![], vec![]],
        ),
        (
            "MATCH (n:P) RETURN [(n)<--(m) | m.name] AS r ORDER BY n.name",
            vec![vec![], vec![text("a")], vec![text("a"), text("b")], vec![]],
        ),
        (
            "MATCH (n:P {name: 'a'}) RETURN [(n)-->(m:P {name: 'b'}) | m.name] AS r",
            vec![vec![text("b")]],
        ),
        (
            "MATCH (n:P {name: 'a'}) RETURN [(n)-[:K|X]->(m) | m.name] AS r",
            vec![vec![text("b"), text("c")]],
        ),
        // Uncorrelated: every match in the graph.
        (
            "RETURN [(x:P)-->(y) | y.name] AS r",
            vec![vec![text("b"), text("c"), text("c")]],
        ),
        // An outer projected value is visible to the projection.
        (
            "MATCH (n:P {name: 'a'}) WITH n, 10 AS k RETURN [(n)-->(m) | k + size(m.name)] AS r",
            vec![vec![Value::Int64(11), Value::Int64(11)]],
        ),
        // Nested, and relationship uniqueness within the pattern.
        (
            "MATCH (n:P {name: 'b'}) RETURN [(n)-->(m) | [(m)<--(x) | x.name]] AS r",
            vec![vec![Value::List(vec![text("a"), text("b")])]],
        ),
        (
            "MATCH (n:P {name: 'b'}) RETURN [(n)-[r1]-(x)-[r2]-(n) | x.name] AS r",
            vec![vec![]],
        ),
    ] {
        let rows = run(&graph, query).unwrap_or_else(|error| panic!("{query}: {error}"));
        let mut actual = sorted_lists(rows);
        for list in actual.iter_mut() {
            if let [Value::List(inner)] = list.as_mut_slice() {
                inner.sort_by_key(|item| format!("{item:?}"));
            }
        }
        assert_eq!(actual, expected, "{query}");
    }
}

#[test]
fn works_in_filters_aggregates_and_with() {
    let graph = graph();
    for (query, expected) in [
        (
            "MATCH (n:P) WHERE size([(n)-->(m) | m]) > 1 RETURN n.name AS r",
            vec![vec![text("a")]],
        ),
        (
            "MATCH (n:P) RETURN n.name AS n, size([(n)--() | 1]) AS deg ORDER BY n",
            vec![
                vec![text("a"), Value::Int64(2)],
                vec![text("b"), Value::Int64(2)],
                vec![text("c"), Value::Int64(2)],
                vec![text("d"), Value::Int64(0)],
            ],
        ),
        (
            "MATCH (n:P) WITH size([(n)-->() | 1]) AS out RETURN out, count(*) AS c ORDER BY out",
            vec![
                vec![Value::Int64(0), Value::Int64(2)],
                vec![Value::Int64(1), Value::Int64(1)],
                vec![Value::Int64(2), Value::Int64(1)],
            ],
        ),
        (
            "MATCH (q:P {name: 'd'}) OPTIONAL MATCH (q)-->(n) RETURN [(n)-->(m) | m.name] AS r",
            vec![vec![Value::List(vec![])]],
        ),
    ] {
        assert_eq!(
            run(&graph, query).unwrap_or_else(|error| panic!("{query}: {error}")),
            expected,
            "{query}"
        );
    }
}

#[test]
fn a_bracketed_non_comprehension_stays_a_list() {
    let graph = graph();
    for (query, expected) in [
        (
            "RETURN [(1), 2] AS r",
            vec![Value::Int64(1), Value::Int64(2)],
        ),
        ("RETURN [(1) - (2)] AS r", vec![Value::Int64(-1)]),
        (
            "WITH 1 AS x RETURN [x = (1)] AS r",
            vec![Value::Boolean(true)],
        ),
        (
            "MATCH (n:P {name: 'a'}) RETURN [(n)-->(), 1] AS r",
            vec![Value::Boolean(true), Value::Int64(1)],
        ),
    ] {
        assert_eq!(
            run(&graph, query).unwrap(),
            vec![vec![Value::List(expected)]],
            "{query}"
        );
    }
}

#[test]
fn pattern_variables_stay_inside_and_malformed_forms_are_errors() {
    let graph = graph();
    for (query, needle) in [
        ("MATCH (n:P) WITH [(n)-->(m) | m.name] AS r RETURN m", "m"),
        (
            "MATCH (n:P) RETURN [(n)-->(m) WHERE m.name = 'c'] AS r",
            "projection",
        ),
        ("MATCH (n:P) WITH [p = (n)-->(m) | 1] AS r RETURN p", "p"),
        ("MATCH (n:P) RETURN [(n)-->(m) | z] AS r", "z"),
    ] {
        let error = run(&graph, query).expect_err(query);
        assert!(error.contains(needle), "{query}: {error}");
    }
}

#[test]
fn a_named_path_binds_each_match() {
    let graph = graph();
    for (query, expected) in [
        (
            "MATCH (n:P {name: 'a'}) RETURN [p = (n)-->(m) | length(p)] AS r",
            vec![Value::Int64(1), Value::Int64(1)],
        ),
        (
            "MATCH (n:P {name: 'a'}) RETURN [p = (n)-->()-->(m) | [x IN nodes(p) | x.name]] AS r",
            vec![Value::List(vec![text("a"), text("b"), text("c")])],
        ),
        (
            "MATCH (n:P {name: 'a'}) RETURN [p = (n)-[:K*1..2]->(m) WHERE length(p) = 2 | size(relationships(p))] AS r",
            vec![Value::Int64(2)],
        ),
        (
            "MATCH (n:P {name: 'a'}) RETURN [p = (n)-[r:K]->(m) WHERE r.w > 1 | [startNode(relationships(p)[0]).name, m.name]] AS r",
            vec![Value::List(vec![text("a"), text("c")])],
        ),
        (
            "MATCH (n:P {name: 'b'}) RETURN [p = (n)-->()<--(x) | x.name] AS r",
            vec![text("a")],
        ),
    ] {
        let rows = run(&graph, query).unwrap_or_else(|error| panic!("{query}: {error}"));
        assert_eq!(rows, vec![vec![Value::List(expected)]], "{query}");
    }
    // The bound value is a path: its nodes and relationships in order.
    let rows = run(
        &graph,
        "MATCH (n:P {name: 'a'}) RETURN [p = (n)-[:K {w: 5}]->(m) | p] AS r",
    )
    .unwrap();
    let Value::List(paths) = &rows[0][0] else {
        panic!("{rows:?}");
    };
    assert!(
        matches!(paths.as_slice(), [Value::Path(path)] if path.nodes.len() == 2 && path.rels.len() == 1),
        "{rows:?}"
    );
}
