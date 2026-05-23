//! Executor unit tests.
#![allow(clippy::approx_constant)]
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::schema::{EdgeData, NodeData};
use crate::graph::storage::GraphWrite;

/// Test helper: unwraps evaluate_comparison Result for use in assert!()
fn cmp(left: &Value, op: &ComparisonOp, right: &Value) -> bool {
    evaluate_comparison(left, op, right, None).unwrap()
}

// ========================================================================
// evaluate_comparison
// ========================================================================

#[test]
fn test_comparison_equals() {
    assert!(cmp(
        &Value::Int64(5),
        &ComparisonOp::Equals,
        &Value::Int64(5)
    ));
    assert!(!cmp(
        &Value::Int64(5),
        &ComparisonOp::Equals,
        &Value::Int64(6)
    ));
}

#[test]
fn test_comparison_not_equals() {
    assert!(cmp(
        &Value::Int64(5),
        &ComparisonOp::NotEquals,
        &Value::Int64(6)
    ));
    assert!(!cmp(
        &Value::Int64(5),
        &ComparisonOp::NotEquals,
        &Value::Int64(5)
    ));
}

#[test]
fn test_comparison_less_than() {
    assert!(cmp(
        &Value::Int64(3),
        &ComparisonOp::LessThan,
        &Value::Int64(5)
    ));
    assert!(!cmp(
        &Value::Int64(5),
        &ComparisonOp::LessThan,
        &Value::Int64(5)
    ));
}

#[test]
fn test_comparison_less_than_eq() {
    assert!(cmp(
        &Value::Int64(5),
        &ComparisonOp::LessThanEq,
        &Value::Int64(5)
    ));
    assert!(cmp(
        &Value::Int64(3),
        &ComparisonOp::LessThanEq,
        &Value::Int64(5)
    ));
    assert!(!cmp(
        &Value::Int64(6),
        &ComparisonOp::LessThanEq,
        &Value::Int64(5)
    ));
}

#[test]
fn test_comparison_greater_than() {
    assert!(cmp(
        &Value::Int64(7),
        &ComparisonOp::GreaterThan,
        &Value::Int64(5)
    ));
    assert!(!cmp(
        &Value::Int64(5),
        &ComparisonOp::GreaterThan,
        &Value::Int64(5)
    ));
}

#[test]
fn test_comparison_greater_than_eq() {
    assert!(cmp(
        &Value::Int64(5),
        &ComparisonOp::GreaterThanEq,
        &Value::Int64(5)
    ));
    assert!(cmp(
        &Value::Int64(7),
        &ComparisonOp::GreaterThanEq,
        &Value::Int64(5)
    ));
}

#[test]
fn test_comparison_cross_type() {
    // Int64 vs Float64
    assert!(cmp(
        &Value::Int64(5),
        &ComparisonOp::Equals,
        &Value::Float64(5.0)
    ));
    assert!(cmp(
        &Value::Int64(3),
        &ComparisonOp::LessThan,
        &Value::Float64(3.5)
    ));
}

// ========================================================================
// arithmetic helpers
// ========================================================================

#[test]
fn test_arithmetic_add_integers() {
    assert_eq!(
        arithmetic_add(&Value::Int64(3), &Value::Int64(4)),
        Value::Int64(7)
    );
}

#[test]
fn test_arithmetic_add_floats() {
    let result = arithmetic_add(&Value::Float64(1.5), &Value::Float64(2.5));
    assert_eq!(result, Value::Float64(4.0));
}

#[test]
fn test_arithmetic_add_string_concatenation() {
    let result = arithmetic_add(
        &Value::String("hello".to_string()),
        &Value::String(" world".to_string()),
    );
    assert_eq!(result, Value::String("hello world".to_string()));
}

#[test]
fn test_arithmetic_add_mixed_numeric() {
    let result = arithmetic_add(&Value::Int64(3), &Value::Float64(1.5));
    assert_eq!(result, Value::Float64(4.5));
}

#[test]
fn test_arithmetic_sub() {
    assert_eq!(
        arithmetic_sub(&Value::Int64(10), &Value::Int64(3)),
        Value::Int64(7)
    );
    assert_eq!(
        arithmetic_sub(&Value::Float64(5.0), &Value::Float64(2.0)),
        Value::Float64(3.0)
    );
}

#[test]
fn test_arithmetic_mul() {
    assert_eq!(
        arithmetic_mul(&Value::Int64(3), &Value::Int64(4)),
        Value::Int64(12)
    );
}

#[test]
fn test_arithmetic_div() {
    // int / int → int (truncated), per 0.9.0 §5. Float promotion only
    // when at least one operand is a float.
    assert_eq!(
        arithmetic_div(&Value::Int64(10), &Value::Int64(4)),
        Value::Int64(2)
    );
    assert_eq!(
        arithmetic_div(&Value::Int64(10), &Value::Float64(4.0)),
        Value::Float64(2.5)
    );
}

#[test]
fn test_arithmetic_div_by_zero() {
    assert_eq!(
        arithmetic_div(&Value::Int64(10), &Value::Int64(0)),
        Value::Null
    );
    assert_eq!(
        arithmetic_div(&Value::Float64(10.0), &Value::Float64(0.0)),
        Value::Null
    );
}

#[test]
fn test_arithmetic_negate() {
    assert_eq!(arithmetic_negate(&Value::Int64(5)), Value::Int64(-5));
    assert_eq!(
        arithmetic_negate(&Value::Float64(3.14)),
        Value::Float64(-3.14)
    );
    assert_eq!(
        arithmetic_negate(&Value::String("x".to_string())),
        Value::Null
    );
}

#[test]
fn test_arithmetic_incompatible_returns_null() {
    assert_eq!(
        arithmetic_add(&Value::Boolean(true), &Value::Boolean(false)),
        Value::Null
    );
    assert_eq!(
        arithmetic_sub(&Value::String("a".to_string()), &Value::Int64(1)),
        Value::Null
    );
}

// ========================================================================
// value_to_f64
// ========================================================================

#[test]
fn test_value_to_f64_conversions() {
    assert_eq!(value_to_f64(&Value::Int64(42)), Some(42.0));
    assert_eq!(value_to_f64(&Value::Float64(3.14)), Some(3.14));
    assert_eq!(value_to_f64(&Value::UniqueId(7)), Some(7.0));
    assert_eq!(value_to_f64(&Value::String("x".to_string())), None);
    assert_eq!(value_to_f64(&Value::Null), None);
    assert_eq!(value_to_f64(&Value::Boolean(true)), None);
}

// ========================================================================
// to_integer / to_float
// ========================================================================

#[test]
fn test_to_integer() {
    assert_eq!(to_integer(&Value::Int64(42)), Value::Int64(42));
    assert_eq!(to_integer(&Value::Float64(3.7)), Value::Int64(3));
    assert_eq!(to_integer(&Value::UniqueId(5)), Value::Int64(5));
    assert_eq!(
        to_integer(&Value::String("123".to_string())),
        Value::Int64(123)
    );
    assert_eq!(to_integer(&Value::String("abc".to_string())), Value::Null);
    assert_eq!(to_integer(&Value::Boolean(true)), Value::Int64(1));
    assert_eq!(to_integer(&Value::Boolean(false)), Value::Int64(0));
    assert_eq!(to_integer(&Value::Null), Value::Null);
}

#[test]
fn test_to_float() {
    assert_eq!(to_float(&Value::Float64(3.14)), Value::Float64(3.14));
    assert_eq!(to_float(&Value::Int64(42)), Value::Float64(42.0));
    assert_eq!(to_float(&Value::UniqueId(5)), Value::Float64(5.0));
    assert_eq!(
        to_float(&Value::String("2.5".to_string())),
        Value::Float64(2.5)
    );
    assert_eq!(to_float(&Value::String("abc".to_string())), Value::Null);
}

// ========================================================================
// format_value_compact
// ========================================================================

#[test]
fn test_format_value_compact() {
    assert_eq!(format_value_compact(&Value::UniqueId(42)), "42");
    assert_eq!(format_value_compact(&Value::Int64(-5)), "-5");
    assert_eq!(format_value_compact(&Value::Float64(3.0)), "3.0");
    assert_eq!(format_value_compact(&Value::Float64(3.14)), "3.14");
    assert_eq!(format_value_compact(&Value::String("hi".to_string())), "hi");
    assert_eq!(format_value_compact(&Value::Boolean(true)), "true");
    assert_eq!(format_value_compact(&Value::Null), "null");
}

// ========================================================================
// parse_value_string
// ========================================================================

#[test]
fn test_parse_value_string() {
    assert_eq!(parse_value_string("null"), Value::Null);
    assert_eq!(parse_value_string("true"), Value::Boolean(true));
    assert_eq!(parse_value_string("false"), Value::Boolean(false));
    assert_eq!(parse_value_string("42"), Value::Int64(42));
    assert_eq!(parse_value_string("3.14"), Value::Float64(3.14));
    assert_eq!(
        parse_value_string("\"hello\""),
        Value::String("hello".to_string())
    );
    assert_eq!(
        parse_value_string("'world'"),
        Value::String("world".to_string())
    );
    assert_eq!(
        parse_value_string("unquoted"),
        Value::String("unquoted".to_string())
    );
}

// ========================================================================
// is_aggregate_expression
// ========================================================================

#[test]
fn test_is_aggregate_expression() {
    let agg = Expression::FunctionCall {
        name: "count".to_string(),
        args: vec![Expression::Star],
        distinct: false,
    };
    assert!(is_aggregate_expression(&agg));

    let non_agg = Expression::FunctionCall {
        name: "toUpper".to_string(),
        args: vec![Expression::Variable("x".to_string())],
        distinct: false,
    };
    assert!(!is_aggregate_expression(&non_agg));
}

#[test]
fn test_is_aggregate_in_arithmetic() {
    let expr = Expression::Add(
        Box::new(Expression::FunctionCall {
            name: "sum".to_string(),
            args: vec![Expression::Variable("x".to_string())],
            distinct: false,
        }),
        Box::new(Expression::Literal(Value::Int64(1))),
    );
    assert!(is_aggregate_expression(&expr));
}

#[test]
fn test_is_aggregate_literal_false() {
    assert!(!is_aggregate_expression(&Expression::Literal(
        Value::Int64(1)
    )));
    assert!(!is_aggregate_expression(&Expression::Variable(
        "x".to_string()
    )));
}

// ========================================================================
// CASE expression evaluation
// ========================================================================

#[test]
fn test_case_simple_form_evaluation() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let row = ResultRow::new();

    // CASE 'Oslo' WHEN 'Oslo' THEN 'capital' ELSE 'other' END
    let expr = Expression::Case {
        operand: Some(Box::new(Expression::Literal(Value::String(
            "Oslo".to_string(),
        )))),
        when_clauses: vec![(
            CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
            Expression::Literal(Value::String("capital".to_string())),
        )],
        else_expr: Some(Box::new(Expression::Literal(Value::String(
            "other".to_string(),
        )))),
    };

    let result = executor.evaluate_expression(&expr, &row).unwrap();
    assert_eq!(result, Value::String("capital".to_string()));
}

#[test]
fn test_case_simple_form_else() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let row = ResultRow::new();

    // CASE 'Bergen' WHEN 'Oslo' THEN 'capital' ELSE 'other' END
    let expr = Expression::Case {
        operand: Some(Box::new(Expression::Literal(Value::String(
            "Bergen".to_string(),
        )))),
        when_clauses: vec![(
            CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
            Expression::Literal(Value::String("capital".to_string())),
        )],
        else_expr: Some(Box::new(Expression::Literal(Value::String(
            "other".to_string(),
        )))),
    };

    let result = executor.evaluate_expression(&expr, &row).unwrap();
    assert_eq!(result, Value::String("other".to_string()));
}

#[test]
fn test_case_no_else_returns_null() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let row = ResultRow::new();

    // CASE 'Bergen' WHEN 'Oslo' THEN 'capital' END → null
    let expr = Expression::Case {
        operand: Some(Box::new(Expression::Literal(Value::String(
            "Bergen".to_string(),
        )))),
        when_clauses: vec![(
            CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
            Expression::Literal(Value::String("capital".to_string())),
        )],
        else_expr: None,
    };

    let result = executor.evaluate_expression(&expr, &row).unwrap();
    assert_eq!(result, Value::Null);
}

#[test]
fn test_case_generic_form_evaluation() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let mut row = ResultRow::new();
    row.projected.insert("val".to_string(), Value::Int64(25));

    // CASE WHEN val > 18 THEN 'adult' ELSE 'minor' END
    let expr = Expression::Case {
        operand: None,
        when_clauses: vec![(
            CaseCondition::Predicate(Predicate::Comparison {
                left: Expression::Variable("val".to_string()),
                operator: ComparisonOp::GreaterThan,
                right: Expression::Literal(Value::Int64(18)),
            }),
            Expression::Literal(Value::String("adult".to_string())),
        )],
        else_expr: Some(Box::new(Expression::Literal(Value::String(
            "minor".to_string(),
        )))),
    };

    let result = executor.evaluate_expression(&expr, &row).unwrap();
    assert_eq!(result, Value::String("adult".to_string()));
}

// ========================================================================
// Parameter evaluation
// ========================================================================

#[test]
fn test_parameter_resolution() {
    let graph = DirGraph::new();
    let params = HashMap::from([
        ("name".to_string(), Value::String("Alice".to_string())),
        ("age".to_string(), Value::Int64(30)),
    ]);
    let executor = CypherExecutor::with_params(&graph, &params, None);
    let row = ResultRow::new();

    let result = executor
        .evaluate_expression(&Expression::Parameter("name".to_string()), &row)
        .unwrap();
    assert_eq!(result, Value::String("Alice".to_string()));

    let result = executor
        .evaluate_expression(&Expression::Parameter("age".to_string()), &row)
        .unwrap();
    assert_eq!(result, Value::Int64(30));
}

#[test]
fn test_parameter_missing_error() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let row = ResultRow::new();

    let result = executor.evaluate_expression(&Expression::Parameter("missing".to_string()), &row);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Missing parameter"));
}

#[test]
fn test_expression_to_string_case() {
    let expr = Expression::Case {
        operand: None,
        when_clauses: vec![],
        else_expr: None,
    };
    assert_eq!(expression_to_string(&expr), "CASE");
}

#[test]
fn test_expression_to_string_parameter() {
    let expr = Expression::Parameter("foo".to_string());
    assert_eq!(expression_to_string(&expr), "$foo");
}

// ========================================================================
// CREATE / SET mutation tests
// ========================================================================

/// Helper: build a small test graph with 2 Person nodes and 1 KNOWS edge
fn build_test_graph() -> DirGraph {
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

#[test]
fn test_create_single_node() {
    let mut graph = DirGraph::new();
    let query =
        super::super::parser::parse_cypher("CREATE (n:Person {name: 'Alice', age: 30})").unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert!(result.stats.is_some());
    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_created, 1);
    assert_eq!(stats.relationships_created, 0);

    // Verify node was created (no SchemaNodes — metadata stored in HashMap)
    assert_eq!(graph.graph.node_count(), 1);
    let node = graph
        .graph
        .node_weight(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("name").as_deref(),
        Some(&Value::String("Alice".to_string()))
    );
}

#[test]
fn test_create_node_with_properties() {
    let mut graph = DirGraph::new();
    let query =
        super::super::parser::parse_cypher("CREATE (n:Product {name: 'Laptop', price: 999})")
            .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
    let node = graph
        .graph
        .node_weight(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("price").as_deref(),
        Some(&Value::Int64(999))
    );
    assert_eq!(node.get_node_type_ref(&graph.interner), "Product");
}

#[test]
fn test_create_edge_between_matched() {
    let mut graph = build_test_graph();
    let query = super::super::parser::parse_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:FRIENDS]->(b)",
    )
    .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_created, 0);
    assert_eq!(stats.relationships_created, 1);

    // Verify edge was created (graph should now have 2 edges: KNOWS + FRIENDS)
    assert_eq!(graph.graph.edge_count(), 2);
}

#[test]
fn test_create_path() {
    let mut graph = DirGraph::new();
    let query = super::super::parser::parse_cypher(
        "CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})",
    )
    .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_created, 2);
    assert_eq!(stats.relationships_created, 1);
    // 2 Person nodes (no SchemaNodes — metadata stored in HashMap)
    assert_eq!(graph.graph.node_count(), 2);
    assert_eq!(graph.graph.edge_count(), 1);
}

#[test]
fn test_create_with_params() {
    let mut graph = DirGraph::new();
    let query =
        super::super::parser::parse_cypher("CREATE (n:Person {name: $name, age: $age})").unwrap();
    let params = HashMap::from([
        ("name".to_string(), Value::String("Charlie".to_string())),
        ("age".to_string(), Value::Int64(35)),
    ]);
    let result = execute_mutable(&mut graph, &query, params, None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
    let node = graph
        .graph
        .node_weight(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("name").as_deref(),
        Some(&Value::String("Charlie".to_string()))
    );
}

#[test]
fn test_create_return() {
    let mut graph = DirGraph::new();
    let query = super::super::parser::parse_cypher(
        "CREATE (n:Person {name: 'Test'}) RETURN n.name AS name",
    )
    .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.columns, vec!["name"]);
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Test".to_string()));
}

#[test]
fn test_set_property() {
    let mut graph = build_test_graph();
    let query =
        super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.age = 31")
            .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.properties_set, 1);

    // Verify property was updated
    let node = graph
        .graph
        .node_weight(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("age").as_deref(),
        Some(&Value::Int64(31))
    );
}

#[test]
fn test_set_title() {
    let mut graph = build_test_graph();
    let query = super::super::parser::parse_cypher(
        "MATCH (n:Person {name: 'Alice'}) SET n.name = 'Alicia'",
    )
    .unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    // title is accessed via "name" or "title"
    let node = graph
        .graph
        .node_weight(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("name").as_deref(),
        Some(&Value::String("Alicia".to_string()))
    );
}

#[test]
fn test_set_id_error() {
    let mut graph = build_test_graph();
    let query =
        super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.id = 999")
            .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("immutable"));
}

#[test]
fn test_set_expression() {
    let mut graph = build_test_graph();
    // Alice has age 30, add 1
    let query = super::super::parser::parse_cypher(
        "MATCH (n:Person {name: 'Alice'}) SET n.age = n.age + 1",
    )
    .unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let node = graph
        .graph
        .node_weight(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("age").as_deref(),
        Some(&Value::Int64(31))
    );
}

#[test]
fn test_is_mutation_query() {
    let read_query = super::super::parser::parse_cypher("MATCH (n:Person) RETURN n").unwrap();
    assert!(!is_mutation_query(&read_query));

    let create_query = super::super::parser::parse_cypher("CREATE (n:Person {name: 'A'})").unwrap();
    assert!(is_mutation_query(&create_query));

    let set_query = super::super::parser::parse_cypher("MATCH (n:Person) SET n.age = 30").unwrap();
    assert!(is_mutation_query(&set_query));

    let delete_query = super::super::parser::parse_cypher("MATCH (n:Person) DELETE n").unwrap();
    assert!(is_mutation_query(&delete_query));

    let merge_query = super::super::parser::parse_cypher("MERGE (n:Person {name: 'A'})").unwrap();
    assert!(is_mutation_query(&merge_query));

    let remove_query = super::super::parser::parse_cypher("MATCH (n:Person) REMOVE n.age").unwrap();
    assert!(is_mutation_query(&remove_query));
}

// ==================================================================
// DELETE Tests
// ==================================================================

#[test]
fn test_detach_delete_node() {
    let mut graph = build_test_graph();
    assert_eq!(graph.graph.node_count(), 2);
    assert_eq!(graph.graph.edge_count(), 1);

    let query =
        super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")
            .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_deleted, 1);
    assert_eq!(stats.relationships_deleted, 1);
    assert_eq!(graph.graph.node_count(), 1);
    assert_eq!(graph.graph.edge_count(), 0);
}

#[test]
fn test_delete_node_with_edges_error() {
    let mut graph = build_test_graph();
    let query =
        super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DELETE n").unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("DETACH DELETE"));
}

#[test]
fn test_delete_relationship() {
    let mut graph = build_test_graph();
    assert_eq!(graph.graph.edge_count(), 1);

    let query =
        super::super::parser::parse_cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) DELETE r")
            .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.relationships_deleted, 1);
    assert_eq!(graph.graph.edge_count(), 0);
    assert_eq!(graph.graph.node_count(), 2);
}

#[test]
fn test_delete_node_no_edges() {
    let mut graph = DirGraph::new();
    let node = NodeData::new(
        Value::UniqueId(1),
        Value::String("Solo".to_string()),
        "Person".to_string(),
        HashMap::from([("name".to_string(), Value::String("Solo".to_string()))]),
        &mut graph.interner,
    );
    let idx = graph.graph.add_node(node);
    graph
        .type_indices
        .entry_or_default("Person".to_string())
        .push(idx);

    let query =
        super::super::parser::parse_cypher("MATCH (n:Person {name: 'Solo'}) DELETE n").unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.unwrap().nodes_deleted, 1);
    assert_eq!(graph.graph.node_count(), 0);
}

#[test]
fn test_detach_delete_updates_type_indices() {
    let mut graph = build_test_graph();
    let query =
        super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")
            .unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let person_indices = graph.type_indices.get("Person").unwrap();
    assert_eq!(person_indices.len(), 1);
}

// ==================================================================
// REMOVE Tests
// ==================================================================

#[test]
fn test_remove_property() {
    let mut graph = build_test_graph();
    let query = super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.age")
        .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().properties_removed, 1);

    let node = graph
        .graph
        .node_weight(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(node.get_field_ref("age").as_deref(), None);
}

#[test]
fn test_remove_nonexistent_property() {
    let mut graph = build_test_graph();
    let query =
        super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.nonexistent")
            .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();
    assert_eq!(result.stats.as_ref().unwrap().properties_removed, 0);
}

#[test]
fn test_remove_label_error() {
    let mut graph = build_test_graph();
    let query =
        super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n:Person")
            .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not supported"));
}

// ==================================================================
// MERGE Tests
// ==================================================================

#[test]
fn test_merge_creates_when_not_found() {
    let mut graph = DirGraph::new();
    let query = super::super::parser::parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
    // 1 Person node (no SchemaNodes — metadata stored in HashMap)
    assert_eq!(graph.graph.node_count(), 1);
}

#[test]
fn test_merge_matches_when_found() {
    let mut graph = build_test_graph();
    let initial_count = graph.graph.node_count();
    let query = super::super::parser::parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 0);
    // No new nodes — MERGE matched existing; schema may or may not exist already
    assert_eq!(graph.graph.node_count(), initial_count);
}

#[test]
fn test_merge_on_create_set() {
    let mut graph = DirGraph::new();
    let query = super::super::parser::parse_cypher(
        "MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30",
    )
    .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
    assert_eq!(result.stats.as_ref().unwrap().properties_set, 1);
}

#[test]
fn test_merge_on_match_set() {
    let mut graph = build_test_graph();
    let query = super::super::parser::parse_cypher(
        "MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1",
    )
    .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 0);
    assert_eq!(result.stats.as_ref().unwrap().properties_set, 1);

    let node = graph
        .graph
        .node_weight(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("visits").as_deref(),
        Some(&Value::Int64(1))
    );
}

#[test]
fn test_merge_relationship_matches() {
    let mut graph = build_test_graph();
    let query = super::super::parser::parse_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) MERGE (a)-[r:KNOWS]->(b)",
    )
    .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 0);
    assert_eq!(graph.graph.edge_count(), 1);
}

#[test]
fn test_merge_creates_relationship() {
    let mut graph = build_test_graph();
    let query = super::super::parser::parse_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) MERGE (a)-[r:FRIENDS]->(b)",
    )
    .unwrap();
    let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 1);
    assert_eq!(graph.graph.edge_count(), 2);
}

// ========================================================================
// Index auto-maintenance integration tests
// ========================================================================

#[test]
fn test_create_updates_property_index() {
    let mut graph = build_test_graph();
    graph.create_index("Person", "age");

    // CREATE a new Person — should appear in the age index
    let query =
        super::super::parser::parse_cypher("CREATE (p:Person {name: 'Charlie', age: 40})").unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let found = graph.lookup_by_index("Person", "age", &Value::Int64(40));
    assert!(found.is_some());
    assert_eq!(found.unwrap().len(), 1);
}

#[test]
fn test_set_updates_property_index() {
    let mut graph = build_test_graph();
    graph.create_index("Person", "age");

    // SET Alice.age from 30 to 99
    let query =
        super::super::parser::parse_cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 99")
            .unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    // Old value should be gone
    let old = graph.lookup_by_index("Person", "age", &Value::Int64(30));
    assert!(old.is_none() || old.unwrap().is_empty());

    // New value should be present
    let new = graph.lookup_by_index("Person", "age", &Value::Int64(99));
    assert!(new.is_some());
    assert_eq!(new.unwrap().len(), 1);
}

#[test]
fn test_remove_updates_property_index() {
    let mut graph = build_test_graph();
    graph.create_index("Person", "age");

    // REMOVE Alice.age — should disappear from index
    let query = super::super::parser::parse_cypher("MATCH (p:Person {name: 'Alice'}) REMOVE p.age")
        .unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let found = graph.lookup_by_index("Person", "age", &Value::Int64(30));
    assert!(found.is_none() || found.unwrap().is_empty());
}

#[test]
fn test_create_creates_type_metadata() {
    let mut graph = DirGraph::new();
    let query =
        super::super::parser::parse_cypher("CREATE (p:Animal {name: 'Rex', species: 'Dog'})")
            .unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    // Type metadata for "Animal" should exist
    let metadata = graph.get_node_type_metadata("Animal");
    assert!(
        metadata.is_some(),
        "Type metadata for Animal should exist after CREATE"
    );
    let props = metadata.unwrap();
    assert!(props.contains_key("name"), "metadata should contain 'name'");
    assert!(
        props.contains_key("species"),
        "metadata should contain 'species'"
    );
}

#[test]
fn test_merge_updates_indices() {
    let mut graph = build_test_graph();
    graph.create_index("Person", "age");

    // MERGE create path — new node should appear in index
    let query = super::super::parser::parse_cypher(
        "MERGE (p:Person {name: 'Dave'}) ON CREATE SET p.age = 50",
    )
    .unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let found = graph.lookup_by_index("Person", "age", &Value::Int64(50));
    assert!(found.is_some());
    assert_eq!(found.unwrap().len(), 1);

    // MERGE match path with SET — index should update
    let query2 = super::super::parser::parse_cypher(
        "MERGE (p:Person {name: 'Alice'}) ON MATCH SET p.age = 31",
    )
    .unwrap();
    execute_mutable(&mut graph, &query2, HashMap::new(), None).unwrap();

    // Old Alice age gone
    let old = graph.lookup_by_index("Person", "age", &Value::Int64(30));
    assert!(old.is_none() || old.unwrap().is_empty());

    // New Alice age present
    let new = graph.lookup_by_index("Person", "age", &Value::Int64(31));
    assert!(new.is_some());
    assert_eq!(new.unwrap().len(), 1);
}

#[test]
fn test_self_loop_pattern_same_variable() {
    // Build graph manually: Alice -KNOWS-> Bob, Alice -KNOWS-> Alice (self-loop)
    let mut graph = build_test_graph(); // Alice -> Bob via KNOWS
                                        // Add self-loop: Alice -> Alice
    let alice_idx = graph.type_indices.get("Person").unwrap().get(0).unwrap();
    let self_edge = EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(alice_idx, alice_idx, self_edge);

    // MATCH (p)-[:KNOWS]->(p) should only return the self-loop (Alice->Alice)
    let read_query =
        super::super::parser::parse_cypher("MATCH (p:Person)-[:KNOWS]->(p) RETURN p.name").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&read_query).unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("Alice".to_string()))
    );
}

#[test]
fn test_edge_variable_in_expression() {
    // Edge variables should resolve to connection_type, not Null
    let graph = build_test_graph(); // Alice -KNOWS-> Bob
    let query = super::super::parser::parse_cypher(
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r, count(r) AS cnt",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&query).unwrap();

    assert!(!result.rows.is_empty());
    // count(r) should be non-zero (was 0 before fix)
    let cnt_col = result.columns.iter().position(|c| c == "cnt").unwrap();
    assert_eq!(result.rows[0].get(cnt_col), Some(&Value::Int64(1)));
}

#[test]
fn test_path_variable_count() {
    // Path variables should be countable (non-null)
    let mut graph = DirGraph::new();
    let query = super::super::parser::parse_cypher(
        "CREATE (a:Node {name: 'A'}), (b:Node {name: 'B'}), (c:Node {name: 'C'}), \
         (a)-[:LINK]->(b), (b)-[:LINK]->(c)",
    )
    .unwrap();
    execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

    let read_query = super::super::parser::parse_cypher(
        "MATCH path = (a:Node)-[:LINK*1..2]->(b:Node) RETURN count(path) AS cnt",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&read_query).unwrap();

    assert_eq!(result.rows.len(), 1);
    let cnt_col = result.columns.iter().position(|c| c == "cnt").unwrap();
    // Should be > 0 (A->B, B->C, A->B->C = 3 paths)
    match result.rows[0].get(cnt_col) {
        Some(Value::Int64(n)) => assert!(*n > 0, "count(path) should be > 0, got {}", n),
        other => panic!("Expected Int64, got {:?}", other),
    }
}

// ========================================================================
// parse_list_value + split_top_level_commas tests
// ========================================================================

#[test]
fn test_parse_list_value_simple_ints() {
    let val = Value::String("[1, 2, 3]".to_string());
    let items = parse_list_value(&val);
    assert_eq!(items.len(), 3);
    assert_eq!(items[0], Value::Int64(1));
    assert_eq!(items[1], Value::Int64(2));
    assert_eq!(items[2], Value::Int64(3));
}

#[test]
fn test_parse_list_value_strings() {
    let val = Value::String(r#"["hello", "world"]"#.to_string());
    let items = parse_list_value(&val);
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], Value::String("hello".to_string()));
    assert_eq!(items[1], Value::String("world".to_string()));
}

#[test]
fn test_parse_list_value_empty() {
    let val = Value::String("[]".to_string());
    let items = parse_list_value(&val);
    assert!(items.is_empty());
}

#[test]
fn test_parse_list_value_json_objects() {
    // This is the critical test — JSON objects must not be split on inner commas
    let val =
        Value::String(r#"[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]"#.to_string());
    let items = parse_list_value(&val);
    assert_eq!(items.len(), 2);
    // Each item should be a complete JSON object string
    match &items[0] {
        Value::String(s) => assert!(s.contains("Alice"), "first item: {}", s),
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_parse_list_value_booleans() {
    let val = Value::String("[true, false, null]".to_string());
    let items = parse_list_value(&val);
    assert_eq!(items.len(), 3);
    assert_eq!(items[0], Value::Boolean(true));
    assert_eq!(items[1], Value::Boolean(false));
    assert_eq!(items[2], Value::Null);
}

#[test]
fn test_parse_list_value_non_list() {
    let val = Value::String("not a list".to_string());
    let items = parse_list_value(&val);
    assert!(items.is_empty());
}

#[test]
fn test_parse_list_value_non_string() {
    let val = Value::Int64(42);
    let items = parse_list_value(&val);
    assert!(items.is_empty());
}

#[test]
fn test_split_top_level_commas_simple() {
    let items = split_top_level_commas("a, b, c");
    assert_eq!(items, vec!["a", " b", " c"]);
}

#[test]
fn test_split_top_level_commas_nested_braces() {
    let items = split_top_level_commas(r#"{"a": 1, "b": 2}, {"c": 3}"#);
    assert_eq!(items.len(), 2);
    assert!(items[0].contains("\"a\": 1"));
    assert!(items[1].contains("\"c\": 3"));
}

#[test]
fn test_split_top_level_commas_nested_brackets() {
    let items = split_top_level_commas("[1, 2], [3, 4]");
    assert_eq!(items.len(), 2);
}

#[test]
fn test_split_top_level_commas_quoted_strings() {
    let items = split_top_level_commas(r#""hello, world", "foo""#);
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].trim(), r#""hello, world""#);
}

// ========================================================================
// String function tests
// ========================================================================

/// Helper: create a graph with one node and run a Cypher RETURN expression
fn eval_string_fn(query: &str) -> Value {
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (n:Item {name: 'hello world', path: 'src/graph/mod.rs'})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher(query).unwrap();
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
        Value::String(r#"["src", "graph", "mod.rs"]"#.to_string())
    );
}

#[test]
fn test_split_function_single_char() {
    let val = eval_string_fn("MATCH (n:Item) RETURN split(n.name, ' ')");
    assert_eq!(val, Value::String(r#"["hello", "world"]"#.to_string()));
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
    let setup = super::super::parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN trim(n.val)").unwrap();
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
    let setup = super::super::parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN ltrim(n.val)").unwrap();
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
    let setup = super::super::parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN rtrim(n.val)").unwrap();
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
    let setup = super::super::parser::parse_cypher("CREATE (n:Item {num: 42})").unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    // split(42, '/') → ["42"] (coerced to "42", no '/' found)
    let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN split(n.num, '/')").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("[\"42\"]".to_string())),
    );

    // substring(42, 0) → "42"
    let q =
        super::super::parser::parse_cypher("MATCH (n:Item) RETURN substring(n.num, 0)").unwrap();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("42".to_string())),
    );

    // reverse(42) → "24"
    let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN reverse(n.num)").unwrap();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("24".to_string())),
    );

    // Null input should still return Null
    let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN substring(n.missing, 0)")
        .unwrap();
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
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Fn {title: 'A'}), (b:Fn {title: 'B'}), (c:Fn {title: 'C'}), \
         (a)-[:CALLS]->(b), (b)-[:CALLS]->(c), (a)-[:IMPORTS]->(c)",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    // String syntax
    let q1 = super::super::parser::parse_cypher(
        "CALL pagerank({connection_types: 'CALLS'}) YIELD node, score RETURN node.title, score ORDER BY score DESC",
    )
    .unwrap();
    let r1 = CypherExecutor::with_params(&graph, &HashMap::new(), None)
        .execute(&q1)
        .unwrap();

    // List syntax (was broken — gave uniform 1/N scores)
    let q2 = super::super::parser::parse_cypher(
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

#[test]
fn test_list_slice_basic() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    // Phase A.1 / C4 — slice now returns Value::List, not JSON string.

    // [start..end]
    let q = super::super::parser::parse_cypher("RETURN [1,2,3,4,5][1..3]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![Value::Int64(2), Value::Int64(3)]))
    );

    // [..end]
    let q = super::super::parser::parse_cypher("RETURN [1,2,3][..2]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![Value::Int64(1), Value::Int64(2)]))
    );

    // [start..]
    let q = super::super::parser::parse_cypher("RETURN [1,2,3][1..]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![Value::Int64(2), Value::Int64(3)]))
    );
}

#[test]
fn test_list_slice_edge_cases() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    // Phase A.1 / C4 — slice returns native Value::List.

    // Out of bounds — clamps to available
    let q = super::super::parser::parse_cypher("RETURN [1,2,3][..100]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]))
    );

    // Empty slice (start >= end)
    let q = super::super::parser::parse_cypher("RETURN [1,2,3][3..1]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::List(Vec::new())));

    // Negative index in slice
    let q = super::super::parser::parse_cypher("RETURN [1,2,3,4,5][-3..]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![Value::Int64(3), Value::Int64(4), Value::Int64(5)]))
    );
}

#[test]
fn test_list_index_still_works() {
    // Verify plain indexing is unbroken
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    let q = super::super::parser::parse_cypher("RETURN [10,20,30][0]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(10)));

    let q = super::super::parser::parse_cypher("RETURN [10,20,30][-1]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(30)));
}

#[test]
fn test_list_slice_with_collect() {
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), \
         (c:Item {name: 'C'}), (d:Item {name: 'D'})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher(
        "MATCH (n:Item) WITH collect(n.name) AS names RETURN names[..2]",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();

    // Should return a list with exactly 2 elements
    let val = result.rows[0].first().unwrap();
    let items = parse_list_value(val);
    assert_eq!(items.len(), 2);
}

#[test]
fn test_size_on_list() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    // size() on a list literal should return element count, not string length
    let q = super::super::parser::parse_cypher("RETURN size([1,2,3])").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(3)));

    // size() on a plain string should return character count
    let q = super::super::parser::parse_cypher("RETURN size('hello')").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(5)));

    // size() on empty list
    let q = super::super::parser::parse_cypher("RETURN size([])").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(0)));
}

#[test]
fn test_length_on_list() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    // length() on a list should return element count
    let q = super::super::parser::parse_cypher("RETURN length([10,20,30,40])").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(4)));
}

#[test]
fn test_size_on_collect_result() {
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher(
        "MATCH (n:Item) WITH collect(n.name) AS names RETURN size(names)",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(3)));
}

#[test]
fn test_aggregate_with_slice() {
    // collect(...)[0..N] in RETURN with aggregation
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Item {cat: 'X', name: 'A'}), (b:Item {cat: 'X', name: 'B'}), \
         (c:Item {cat: 'X', name: 'C'}), (d:Item {cat: 'Y', name: 'D'})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher(
        "MATCH (n:Item) \
         RETURN n.cat AS cat, count(n) AS cnt, collect(n.name)[..2] AS sample \
         ORDER BY cat",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();

    assert_eq!(result.rows.len(), 2);
    // Group X has 3 items, sliced to 2
    let x_row = &result.rows[0];
    assert_eq!(x_row.first(), Some(&Value::String("X".into())));
    assert_eq!(x_row.get(1), Some(&Value::Int64(3)));
    let sample = parse_list_value(x_row.get(2).unwrap());
    assert_eq!(sample.len(), 2);

    // Group Y has 1 item, sliced to at most 2
    let y_row = &result.rows[1];
    assert_eq!(y_row.first(), Some(&Value::String("Y".into())));
    assert_eq!(y_row.get(1), Some(&Value::Int64(1)));
    let sample_y = parse_list_value(y_row.get(2).unwrap());
    assert_eq!(sample_y.len(), 1);
}

#[test]
fn test_aggregate_arithmetic() {
    // count(*) + 1 in RETURN with aggregation
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN count(n) + 1 AS cnt_plus")
        .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    // count(n)=3, 3+1=4.0 (float because add_values promotes)
    let val = result.rows[0].first().unwrap();
    match val {
        Value::Int64(i) => assert_eq!(*i, 4),
        Value::Float64(f) => assert!((f - 4.0).abs() < 0.001),
        _ => panic!("Expected numeric, got {:?}", val),
    }
}

#[test]
fn test_size_of_collect_in_return() {
    // size(collect(...)) in RETURN — non-aggregate wrapping aggregate
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    // No grouping — all rows aggregated
    let q =
        super::super::parser::parse_cypher("MATCH (n:Item) RETURN size(collect(n.name)) AS cnt")
            .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(3)));
}

#[test]
fn test_size_of_collect_grouped() {
    // size(collect(...)) with grouping
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Item {cat: 'X', name: 'A'}), (b:Item {cat: 'X', name: 'B'}), \
         (c:Item {cat: 'X', name: 'C'}), (d:Item {cat: 'Y', name: 'D'})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher(
        "MATCH (n:Item) \
         RETURN n.cat AS cat, size(collect(n.name)) AS cnt \
         ORDER BY cat",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.rows[0].get(1), Some(&Value::Int64(3))); // X: 3
    assert_eq!(result.rows[1].get(1), Some(&Value::Int64(1))); // Y: 1
}

// ========================================================================
// List Quantifier Predicate Tests
// ========================================================================

#[test]
fn test_list_predicate_any() {
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [1, 2, 3, 4, 5] AS nums \
         RETURN any(x IN nums WHERE x > 3) AS result",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(true)));
}

#[test]
fn test_list_predicate_any_false() {
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [1, 2, 3] AS nums \
         RETURN any(x IN nums WHERE x > 10) AS result",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(false)));
}

#[test]
fn test_list_predicate_all() {
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [2, 4, 6] AS nums \
         RETURN all(x IN nums WHERE x > 0) AS result",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(true)));
}

#[test]
fn test_list_predicate_all_false() {
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [2, 4, 6] AS nums \
         RETURN all(x IN nums WHERE x > 3) AS result",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(false)));
}

#[test]
fn test_list_predicate_none() {
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [1, 2, 3] AS nums \
         RETURN none(x IN nums WHERE x > 10) AS result",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(true)));
}

#[test]
fn test_list_predicate_none_false() {
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [1, 2, 3] AS nums \
         RETURN none(x IN nums WHERE x > 2) AS result",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(false)));
}

#[test]
fn test_list_predicate_single() {
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [1, 2, 3] AS nums \
         RETURN single(x IN nums WHERE x > 2) AS result",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(true)));
}

#[test]
fn test_list_predicate_single_false_multiple() {
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [1, 2, 3] AS nums \
         RETURN single(x IN nums WHERE x > 1) AS result",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(false)));
}

#[test]
fn test_list_predicate_in_where_clause() {
    // The user's actual use case: any(w IN list WHERE w.prop IS NOT NULL)
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Well {name: 'W1', depth: 100}), \
         (b:Well {name: 'W2'}), \
         (c:Well {name: 'W3', depth: 300})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    let q = super::super::parser::parse_cypher(
        "MATCH (w:Well) \
         WITH collect(w.depth) AS depths \
         WHERE any(d IN depths WHERE d IS NOT NULL) \
         RETURN size(depths) AS count",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    // any(d IN depths WHERE d IS NOT NULL) should be true (W1 and W3 have depth)
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn test_list_predicate_with_is_not_null() {
    // Matches the user's real use case: any(w IN values WHERE w IS NOT NULL)
    let graph = DirGraph::new();
    let q = super::super::parser::parse_cypher(
        "WITH [1, null, 3, null, 5] AS values \
         RETURN any(v IN values WHERE v IS NOT NULL) AS has_value, \
                all(v IN values WHERE v IS NOT NULL) AS all_present, \
                none(v IN values WHERE v IS NOT NULL) AS none_present",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(true))); // any: true
    assert_eq!(result.rows[0].get(1), Some(&Value::Boolean(false))); // all: false
    assert_eq!(result.rows[0].get(2), Some(&Value::Boolean(false))); // none: false
}

#[test]
fn test_list_predicate_collected_nodes_property_access() {
    // User's exact pattern: collect nodes, then any(w IN wells WHERE w.prop IS NOT NULL)
    let mut graph = DirGraph::new();
    let setup = super::super::parser::parse_cypher(
        "CREATE (a:Well {name: 'W1', formation: 'Sandstone'}), \
         (b:Well {name: 'W2'}), \
         (c:Well {name: 'W3', formation: 'Limestone'})",
    )
    .unwrap();
    execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

    // any() with collected node property access
    let q = super::super::parser::parse_cypher(
        "MATCH (w:Well) \
         WITH collect(w) AS wells \
         RETURN any(x IN wells WHERE x.formation IS NOT NULL) AS has_formation",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].first(), Some(&Value::Boolean(true)));

    // all() — should be false (W2 has no formation)
    let q2 = super::super::parser::parse_cypher(
        "MATCH (w:Well) \
         WITH collect(w) AS wells \
         RETURN all(x IN wells WHERE x.formation IS NOT NULL) AS all_have",
    )
    .unwrap();
    let executor2 = CypherExecutor::with_params(&graph, &no_params, None);
    let result2 = executor2.execute(&q2).unwrap();
    assert_eq!(result2.rows.len(), 1);
    assert_eq!(result2.rows[0].first(), Some(&Value::Boolean(false)));
}
