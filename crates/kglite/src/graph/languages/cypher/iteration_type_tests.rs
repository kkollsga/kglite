//! A list comprehension, a list quantifier and `reduce` iterate a list; over
//! null they are null, and over any other value they are a type error rather
//! than an empty iteration. `UNWIND` of a non-list is that one value.

use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use std::collections::HashMap;

fn graph() -> DirGraph {
    let mut graph = DirGraph::new();
    let params = HashMap::new();
    execute_mut(
        &mut graph,
        "CREATE (:N {id: 1, s: 'abc', legacy: '[1, 2]'})",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    graph
}

fn run(graph: &DirGraph, query: &str) -> Result<Vec<Vec<Value>>, String> {
    let params: HashMap<String, Value> = [("s".to_string(), Value::String("abc".into()))].into();
    execute_read(graph, query, &ExecuteOptions::eager(&params))
        .map(|outcome| outcome.result.rows)
        .map_err(|error| error.to_string())
}

#[test]
fn iterating_a_non_list_is_a_type_error() {
    let graph = graph();
    for query in [
        "RETURN [x IN 'abc' | x] AS r",
        "RETURN [x IN 5 | x] AS r",
        "RETURN [x IN 5 WHERE x > 1] AS r",
        "RETURN [x IN {a: 1} | x] AS r",
        "RETURN [x IN true | x] AS r",
        "RETURN [x IN $s | x] AS r",
        "MATCH (n:N) RETURN [x IN n.s | x] AS r",
        "MATCH (n:N) RETURN [x IN n | x] AS r",
        "RETURN any(x IN 'abc' WHERE x = 'a') AS r",
        "RETURN all(x IN 5 WHERE x > 1) AS r",
        "RETURN none(x IN 5 WHERE x > 1) AS r",
        "RETURN single(x IN 5 WHERE x > 1) AS r",
        "RETURN reduce(a = 0, x IN 5 | a + x) AS r",
        "RETURN reduce(a = 0, x IN 'abc' | a + 1) AS r",
        // In a filter too, on the fused and the unfused path alike.
        "MATCH (n:N) WHERE all(x IN n.s WHERE x > 1) RETURN n.id AS id",
        "MATCH (n:N) WHERE size([x IN n.s | x]) = 0 RETURN n.id AS id",
        "MATCH (n:N) WITH n WHERE any(x IN n.s WHERE x = 'a') RETURN n.id AS id",
        "MATCH (n:N) WHERE n.id = 1 AND none(x IN n.s WHERE x = 'a') RETURN count(n) AS c",
    ] {
        let error = run(&graph, query).expect_err(query);
        assert!(error.contains("expects a list"), "{query}: {error}");
    }
}

#[test]
fn lists_null_and_unwind_keep_their_meaning() {
    let graph = graph();
    for (query, expected) in [
        (
            "RETURN [x IN [1, 2] | x * 10] AS r",
            Value::List(vec![Value::Int64(10), Value::Int64(20)]),
        ),
        ("RETURN [x IN null | x] AS r", Value::Null),
        ("RETURN all(x IN null WHERE x > 1) AS r", Value::Null),
        ("RETURN reduce(a = 0, x IN null | a + x) AS r", Value::Null),
        (
            "RETURN reduce(a = 0, x IN [1, 2] | a + x) AS r",
            Value::Int64(3),
        ),
        // A list stored as its text form still iterates as the list.
        (
            "MATCH (n:N) RETURN [x IN n.legacy | x] AS r",
            Value::List(vec![Value::Int64(1), Value::Int64(2)]),
        ),
        ("UNWIND 5 AS x RETURN x", Value::Int64(5)),
        ("UNWIND 'abc' AS x RETURN x", Value::String("abc".into())),
    ] {
        assert_eq!(run(&graph, query).unwrap(), vec![vec![expected]], "{query}");
    }
}

#[test]
fn list_operators_over_a_non_list_are_a_type_error() {
    let graph = graph();
    for query in [
        "RETURN 'a' IN 'abc' AS r",
        "RETURN 1 IN 5 AS r",
        "RETURN 1 IN {a: 1} AS r",
        "RETURN 1 IN true AS r",
        "RETURN null IN 'abc' AS r",
        "RETURN 'a' IN $s AS r",
        "MATCH (n:N) WHERE 'a' IN n.s RETURN n.id AS id",
        "MATCH (n:N) WITH n WHERE 'a' IN n.s RETURN n.id AS id",
        "MATCH (n:N) WHERE n.id IN $s RETURN n.id AS id",
        "MATCH (n:N) WHERE n.id IN 5 RETURN n.id AS id",
        "MATCH (n:N) WHERE 1 IN n.id RETURN count(n) AS c",
        "RETURN head('abc') AS r",
        "RETURN last(5) AS r",
        "RETURN size(5) AS r",
        "RETURN length(true) AS r",
        "MATCH (n:N) RETURN size(n) AS r",
        "RETURN 'abc'[0] AS r",
        "RETURN 5[0] AS r",
        "RETURN {a: 1}[0] AS r",
        "MATCH (n:N) RETURN n.s[0] AS r",
        "RETURN 'abc'[0..2] AS r",
        "RETURN 5[..1] AS r",
        "MATCH (n:N) RETURN count(n)[0] AS r",
        "MATCH (n:N) RETURN count(n)[0..1] AS r",
    ] {
        let error = run(&graph, query).expect_err(query);
        assert!(error.contains("expects a list"), "{query}: {error}");
    }
}

#[test]
fn list_operators_keep_their_meaning_over_lists_and_null() {
    let graph = graph();
    let ints = |items: &[i64]| Value::List(items.iter().copied().map(Value::Int64).collect());
    for (query, expected) in [
        ("RETURN 1 IN null AS r", Value::Null),
        ("RETURN 2 IN [1, 2] AS r", Value::Boolean(true)),
        ("RETURN 3 IN [1, null] AS r", Value::Null),
        (
            "MATCH (n:N) RETURN 2 IN n.legacy AS r",
            Value::Boolean(true),
        ),
        ("RETURN head([1, 2]) AS r", Value::Int64(1)),
        ("RETURN last([1, 2]) AS r", Value::Int64(2)),
        ("RETURN head([]) AS r", Value::Null),
        ("RETURN last(null) AS r", Value::Null),
        ("MATCH (n:N) RETURN last(n.legacy) AS r", Value::Int64(2)),
        ("RETURN size('abc') AS r", Value::Int64(3)),
        ("RETURN size([1, 2]) AS r", Value::Int64(2)),
        ("RETURN size(null) AS r", Value::Null),
        ("RETURN length(null) AS r", Value::Null),
        ("RETURN [1, 2][5] AS r", Value::Null),
        ("RETURN [1, 2][-1] AS r", Value::Int64(2)),
        ("RETURN null[0] AS r", Value::Null),
        ("RETURN null[0..2] AS r", Value::Null),
        ("RETURN [1, 2, 3][1..] AS r", ints(&[2, 3])),
        ("MATCH (n:N) RETURN n.legacy[1] AS r", Value::Int64(2)),
        ("MATCH (n:N) RETURN n.legacy[0..1] AS r", ints(&[1])),
        ("RETURN {a: 1}['a'] AS r", Value::Int64(1)),
        ("MATCH (n:N) RETURN collect(n.id)[0] AS r", Value::Int64(1)),
        ("MATCH (n:N) RETURN collect(n.id)[0..1] AS r", ints(&[1])),
    ] {
        assert_eq!(run(&graph, query).unwrap(), vec![vec![expected]], "{query}");
    }
}
