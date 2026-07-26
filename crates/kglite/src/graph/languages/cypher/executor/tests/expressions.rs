//! Expression evaluation: comparison, arithmetic, coercion, aggregate
//! detection, CASE, and parameter resolution.

use super::*;

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
fn expression_only_window_function_has_an_exact_boundary_error() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let expr = Expression::WindowFunction {
        name: "row_number".to_string(),
        partition_by: Vec::new(),
        order_by: Vec::new(),
    };

    assert_eq!(
        executor.evaluate_expression(&expr, &ResultRow::new()),
        Err("Window function must appear in RETURN/WITH clause".to_string())
    );
}

#[test]
fn count_subquery_expression_propagates_deadline_and_cancellation() {
    static CANCELLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

    let graph = build_test_graph();
    let no_params = HashMap::new();
    let query = parser::parse_cypher("RETURN COUNT { (n) } AS c").unwrap();
    let Clause::Return(return_clause) = &query.clauses[0] else {
        panic!("expected RETURN clause");
    };
    let expr = &return_clause.items[0].expression;

    let timed_out = CypherExecutor::with_params(
        &graph,
        &no_params,
        Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
    )
    .evaluate_expression(expr, &ResultRow::new())
    .unwrap_err();
    assert_eq!(
        timed_out,
        "Query timed out during node scan. Hint: add an index on a predicate property \
         (create_index), anchor with MATCH (n {id: ...}), or raise timeout_ms."
    );

    let cancelled = CypherExecutor::with_params(&graph, &no_params, None)
        .with_cancel(Some(&CANCELLED))
        .evaluate_expression(expr, &ResultRow::new())
        .unwrap_err();
    assert_eq!(cancelled, "Query cancelled");

    let over_budget = CypherExecutor::with_params(&graph, &no_params, None)
        .with_max_rows(Some(1))
        .evaluate_expression(expr, &ResultRow::new())
        .unwrap_err();
    assert!(over_budget.contains("max_rows limit of 1"), "{over_budget}");
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
