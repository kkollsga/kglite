//! String functions, and the procedure arguments that arrive as JSON list
//! literals.

use super::*;

// ========================================================================
// String function tests
// ========================================================================

/// Helper: create a graph with one node and run a Cypher RETURN expression
fn eval_string_fn(query: &str) -> Value {
    let mut graph = DirGraph::new();
    let setup =
        parser::parse_cypher("CREATE (n:Item {name: 'hello world', path: 'src/graph/mod.rs'})")
            .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher(query).unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1, "Expected 1 row for query: {}", query);
    result.rows[0].first().cloned().unwrap_or(Value::Null)
}

#[test]
fn test_split_function() {
    let val = eval_string_fn("MATCH (n:Item) RETURN split(n.path, '/')");
    assert_eq!(
        val,
        Value::List(vec![
            Value::String("src".to_string()),
            Value::String("graph".to_string()),
            Value::String("mod.rs".to_string()),
        ])
    );
}

#[test]
fn test_split_function_single_char() {
    let val = eval_string_fn("MATCH (n:Item) RETURN split(n.name, ' ')");
    assert_eq!(
        val,
        Value::List(vec![
            Value::String("hello".to_string()),
            Value::String("world".to_string()),
        ])
    );
}

#[test]
fn test_replace_function() {
    let val = eval_string_fn("MATCH (n:Item) RETURN replace(n.path, '/', '.')");
    assert_eq!(val, Value::String("src.graph.mod.rs".to_string()));
}

#[test]
fn test_substring_two_args() {
    let val = eval_string_fn("MATCH (n:Item) RETURN substring(n.name, 6)");
    assert_eq!(val, Value::String("world".to_string()));
}

#[test]
fn test_substring_three_args() {
    let val = eval_string_fn("MATCH (n:Item) RETURN substring(n.name, 0, 5)");
    assert_eq!(val, Value::String("hello".to_string()));
}

#[test]
fn test_left_function() {
    let val = eval_string_fn("MATCH (n:Item) RETURN left(n.name, 5)");
    assert_eq!(val, Value::String("hello".to_string()));
}

#[test]
fn test_right_function() {
    let val = eval_string_fn("MATCH (n:Item) RETURN right(n.name, 5)");
    assert_eq!(val, Value::String("world".to_string()));
}

#[test]
fn test_trim_function() {
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher("MATCH (n:Item) RETURN trim(n.val)").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("hello".to_string()))
    );
}

#[test]
fn test_ltrim_function() {
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher("MATCH (n:Item) RETURN ltrim(n.val)").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("hello  ".to_string()))
    );
}

#[test]
fn test_rtrim_function() {
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher("MATCH (n:Item) RETURN rtrim(n.val)").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("  hello".to_string()))
    );
}

#[test]
fn test_reverse_function() {
    let val = eval_string_fn("MATCH (n:Item) RETURN reverse(n.name)");
    assert_eq!(val, Value::String("dlrow olleh".to_string()));
}

#[test]
fn test_string_functions_auto_coerce() {
    // String functions on non-string values should auto-coerce to string
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher("CREATE (n:Item {num: 42})").unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    // split(42, '/') → ["42"] as a native list (coerced to "42", no '/' found)
    let q = parser::parse_cypher("MATCH (n:Item) RETURN split(n.num, '/')").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![Value::String("42".to_string())])),
    );

    // substring(42, 0) → "42"
    let q = parser::parse_cypher("MATCH (n:Item) RETURN substring(n.num, 0)").unwrap();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("42".to_string())),
    );

    // reverse(42) → "24"
    let q = parser::parse_cypher("MATCH (n:Item) RETURN reverse(n.num)").unwrap();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("24".to_string())),
    );

    // Null input should still return Null
    let q = parser::parse_cypher("MATCH (n:Item) RETURN substring(n.missing, 0)").unwrap();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Null),);
}

#[test]
fn test_call_param_string_list_parses_json_array() {
    // List literals like ['CALLS'] are serialized as JSON strings "[\"CALLS\"]"
    // call_param_string_list must parse them back into Vec<String>
    let mut params = HashMap::new();

    // Single string value (existing behavior)
    params.insert("types".to_string(), Value::String("CALLS".to_string()));
    assert_eq!(
        call_param_string_list(&params, "types"),
        Some(vec!["CALLS".to_string()])
    );

    // JSON array string from list literal (the bug fix)
    params.insert(
        "types".to_string(),
        Value::String("[\"CALLS\"]".to_string()),
    );
    assert_eq!(
        call_param_string_list(&params, "types"),
        Some(vec!["CALLS".to_string()])
    );

    // Multiple items in list
    params.insert(
        "types".to_string(),
        Value::String("[\"CALLS\", \"IMPORTS\"]".to_string()),
    );
    assert_eq!(
        call_param_string_list(&params, "types"),
        Some(vec!["CALLS".to_string(), "IMPORTS".to_string()])
    );

    // Missing key
    assert_eq!(call_param_string_list(&params, "missing"), None);
}

#[test]
fn test_pagerank_connection_types_list_syntax() {
    // Regression: pagerank({connection_types: ['CALLS']}) must produce
    // the same results as pagerank({connection_types: 'CALLS'})
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher(
        "CREATE (a:Fn {title: 'A'}), (b:Fn {title: 'B'}), (c:Fn {title: 'C'}), \
         (a)-[:CALLS]->(b), (b)-[:CALLS]->(c), (a)-[:IMPORTS]->(c)",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    // String syntax
    let q1 = parser::parse_cypher(
        "CALL pagerank({connection_types: 'CALLS'}) YIELD node, score RETURN node.title, score ORDER BY score DESC",
    )
    .unwrap();
    let r1 = CypherExecutor::with_params(&graph, &HashMap::new(), None)
        .execute(&q1)
        .unwrap();

    // List syntax (was broken — gave uniform 1/N scores)
    let q2 = parser::parse_cypher(
        "CALL pagerank({connection_types: ['CALLS']}) YIELD node, score RETURN node.title, score ORDER BY score DESC",
    )
    .unwrap();
    let r2 = CypherExecutor::with_params(&graph, &HashMap::new(), None)
        .execute(&q2)
        .unwrap();

    assert_eq!(r1.rows.len(), r2.rows.len());
    // Scores must match between string and list syntax
    for (row1, row2) in r1.rows.iter().zip(r2.rows.iter()) {
        assert_eq!(row1.first(), row2.first(), "Node names should match");
        assert_eq!(row1.get(1), row2.get(1), "Scores should match");
    }

    // Verify non-uniform: node C receives links, so its score should differ from A
    let score_first = match r1.rows[0].get(1) {
        Some(Value::Float64(f)) => *f,
        _ => panic!("Expected float score"),
    };
    let score_last = match r1.rows[2].get(1) {
        Some(Value::Float64(f)) => *f,
        _ => panic!("Expected float score"),
    };
    assert!(
        (score_first - score_last).abs() > 0.01,
        "Scores should be non-uniform when filtering by connection type"
    );
}
