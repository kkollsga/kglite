//! A comparison or boolean expression is a value wherever an expression is:
//! inside a map, list, call argument, map projection, list comprehension
//! projection, `reduce`, `IN` list, `SET`, `ORDER BY` and `UNWIND`, without
//! parentheses.

use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use std::collections::HashMap;

fn graph() -> DirGraph {
    let mut graph = DirGraph::new();
    let params = HashMap::new();
    execute_mut(
        &mut graph,
        "CREATE (:N {id: 1, x: 3})",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    graph
}

fn rows(graph: &mut DirGraph, query: &str) -> Vec<Vec<Value>> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let result = if query.contains("SET") || query.contains("CREATE") {
        execute_mut(graph, query, &opts).map(|outcome| outcome.result.rows)
    } else {
        execute_read(graph, query, &opts).map(|outcome| outcome.result.rows)
    };
    result.unwrap_or_else(|error| panic!("{query}: {error}"))
}

fn b(value: bool) -> Value {
    Value::Boolean(value)
}

fn list(items: Vec<Value>) -> Value {
    Value::List(items)
}

fn map(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

/// Each query's result, and the same query with the expression parenthesised:
/// the two must agree.
#[test]
fn an_unparenthesised_predicate_is_a_value_everywhere() {
    let cases: Vec<(&str, &str, Vec<Vec<Value>>)> = vec![
        (
            "UNWIND [1, 2] AS i RETURN {b: i % 2 = 0} AS m",
            "UNWIND [1, 2] AS i RETURN {b: (i % 2 = 0)} AS m",
            vec![
                vec![map(vec![("b", b(false))])],
                vec![map(vec![("b", b(true))])],
            ],
        ),
        (
            "RETURN {b: 1 = 1 AND NOT 2 < 1, c: {d: 1 IN [1]}} AS m",
            "RETURN {b: (1 = 1 AND NOT 2 < 1), c: {d: (1 IN [1])}} AS m",
            vec![vec![map(vec![
                ("b", b(true)),
                ("c", map(vec![("d", b(true))])),
            ])]],
        ),
        (
            "RETURN [1 = 1, true AND false, 2 > 1 OR false] AS l",
            "RETURN [(1 = 1), (true AND false), (2 > 1 OR false)] AS l",
            vec![vec![list(vec![b(true), b(false), b(true)])]],
        ),
        (
            "RETURN toString(1 = 1) AS s, coalesce(null, 1 < 2) AS c",
            "RETURN toString((1 = 1)) AS s, coalesce(null, (1 < 2)) AS c",
            vec![vec![Value::String("true".into()), b(true)]],
        ),
        (
            "MATCH (n:N) RETURN n {.id, big: n.x > 2} AS m",
            "MATCH (n:N) RETURN n {.id, big: (n.x > 2)} AS m",
            vec![vec![map(vec![("id", Value::Int64(1)), ("big", b(true))])]],
        ),
        (
            "RETURN [x IN [1, 2] | x > 1] AS l",
            "RETURN [x IN [1, 2] | (x > 1)] AS l",
            vec![vec![list(vec![b(false), b(true)])]],
        ),
        (
            "RETURN reduce(acc = 1 = 1, x IN [1, 2] | acc AND x > 0) AS r",
            "RETURN reduce(acc = (1 = 1), x IN [1, 2] | (acc AND x > 0)) AS r",
            vec![vec![b(true)]],
        ),
        (
            "RETURN true IN [1 = 1] AS r",
            "RETURN true IN [(1 = 1)] AS r",
            vec![vec![b(true)]],
        ),
        (
            "UNWIND [1 = 1, 1 = 2] AS v RETURN v",
            "UNWIND [(1 = 1), (1 = 2)] AS v RETURN v",
            vec![vec![b(true)], vec![b(false)]],
        ),
        (
            "UNWIND [1, 2, 3] AS i RETURN i ORDER BY i = 2 DESC, i",
            "UNWIND [1, 2, 3] AS i RETURN i ORDER BY (i = 2) DESC, i",
            vec![
                vec![Value::Int64(2)],
                vec![Value::Int64(1)],
                vec![Value::Int64(3)],
            ],
        ),
        (
            "MATCH (n:N) SET n.flag = n.x > 2 RETURN n.flag AS f",
            "MATCH (n:N) SET n.flag = (n.x > 2) RETURN n.flag AS f",
            vec![vec![b(true)]],
        ),
        (
            "CREATE (m:M {flag: 1 = 1}) RETURN m.flag AS f",
            "CREATE (m:M {flag: (1 = 1)}) RETURN m.flag AS f",
            vec![vec![b(true)]],
        ),
    ];
    for (bare, parenthesised, expected) in cases {
        let mut g = graph();
        assert_eq!(rows(&mut g, parenthesised), expected, "{parenthesised}");
        let mut g = graph();
        assert_eq!(rows(&mut g, bare), expected, "{bare}");
    }
}

/// A pattern predicate closes at the `]` of the list holding it.
#[test]
fn a_pattern_predicate_ends_at_a_closing_bracket() {
    let mut graph = graph();
    execute_mut(
        &mut graph,
        "MATCH (n:N) CREATE (n)-[:R]->(:M {id: 2})",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap();
    for (query, expected) in [
        ("MATCH (n:N) RETURN [(n)-->()] AS r", list(vec![b(true)])),
        (
            "MATCH (n:N) RETURN [1, (n)-[:R]->(:M)] AS r",
            list(vec![Value::Int64(1), b(true)]),
        ),
        ("MATCH (n:N) RETURN [(n)<--()] AS r", list(vec![b(false)])),
        ("MATCH (n:N) RETURN [(n:N)] AS r", list(vec![b(true)])),
    ] {
        assert_eq!(rows(&mut graph, query), vec![vec![expected]], "{query}");
    }
}
