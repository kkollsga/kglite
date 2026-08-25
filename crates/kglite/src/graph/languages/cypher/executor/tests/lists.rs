//! List values: parsing, slicing, indexing, sizing, and the
//! any/all/none/single quantifier predicates.

use super::*;

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

#[test]
fn test_list_slice_basic() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    // Slice returns Value::List, not a JSON string.

    // [start..end]
    let q = parser::parse_cypher("RETURN [1,2,3,4,5][1..3]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![Value::Int64(2), Value::Int64(3)]))
    );

    // [..end]
    let q = parser::parse_cypher("RETURN [1,2,3][..2]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![Value::Int64(1), Value::Int64(2)]))
    );

    // [start..]
    let q = parser::parse_cypher("RETURN [1,2,3][1..]").unwrap();
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

    // Slice returns a native Value::List.

    // Out of bounds — clamps to available
    let q = parser::parse_cypher("RETURN [1,2,3][..100]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![
            Value::Int64(1),
            Value::Int64(2),
            Value::Int64(3)
        ]))
    );

    // Empty slice (start >= end)
    let q = parser::parse_cypher("RETURN [1,2,3][3..1]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::List(Vec::new())));

    // Negative index in slice
    let q = parser::parse_cypher("RETURN [1,2,3,4,5][-3..]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::List(vec![
            Value::Int64(3),
            Value::Int64(4),
            Value::Int64(5)
        ]))
    );
}

#[test]
fn test_list_index_still_works() {
    // Verify plain indexing is unbroken
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    let q = parser::parse_cypher("RETURN [10,20,30][0]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(10)));

    let q = parser::parse_cypher("RETURN [10,20,30][-1]").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(30)));
}

#[test]
fn test_node_list_property_index_borrow_path() {
    // The borrow fast-path in evaluate_index_access indexes a stored list
    // property off a node binding without cloning the whole list. It must
    // stay bit-identical to the owned path for positive, negative, and
    // out-of-range indices, and inside a reduce loop (the vector-scoring
    // shape that motivated the change).
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher("CREATE (n:V {id: 1, arr: [10, 20, 30]})").unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let q = parser::parse_cypher(
        "MATCH (n:V) RETURN n.arr[0] AS a, n.arr[-1] AS b, n.arr[5] AS d, \
         reduce(s = 0, i IN range(0, 2) | s + n.arr[i]) AS total",
    )
    .unwrap();
    let result = executor.execute(&q).unwrap();
    let row = &result.rows[0];
    assert_eq!(row[0], Value::Int64(10)); // first element
    assert_eq!(row[1], Value::Int64(30)); // negative index from the end
    assert_eq!(row[2], Value::Null); // out of range
    assert_eq!(row[3], Value::Int64(60)); // 10 + 20 + 30 via per-element subscript
}

#[test]
fn test_projected_and_parameter_list_index_borrow_path() {
    // The other two borrowable container shapes: a variable projected into
    // the row (WITH … AS q) and a query parameter. Both must index without
    // cloning the whole list and match the owned path.
    let graph = DirGraph::new();
    let mut params = HashMap::new();
    params.insert(
        "p".to_string(),
        Value::List(vec![Value::Int64(7), Value::Int64(8), Value::Int64(9)]),
    );
    let executor = CypherExecutor::with_params(&graph, &params, None);
    let q = parser::parse_cypher(
        "WITH [100, 200, 300] AS q RETURN q[1] AS from_projected, $p[-1] AS from_param",
    )
    .unwrap();
    let result = executor.execute(&q).unwrap();
    let row = &result.rows[0];
    assert_eq!(row[0], Value::Int64(200));
    assert_eq!(row[1], Value::Int64(9));
}

#[test]
fn test_list_slice_with_collect() {
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher(
        "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), \
         (c:Item {name: 'C'}), (d:Item {name: 'D'})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher("MATCH (n:Item) WITH collect(n.name) AS names RETURN names[..2]")
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
    let q = parser::parse_cypher("RETURN size([1,2,3])").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(3)));

    // size() on a plain string should return character count
    let q = parser::parse_cypher("RETURN size('hello')").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(5)));

    // size() on empty list
    let q = parser::parse_cypher("RETURN size([])").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(0)));
}

/// size()/length() count **characters**, not UTF-8 bytes. The ASCII-only
/// assertions above could not tell the two apart, and the byte reading
/// disagreed with the char-indexed substring()/left()/right().
#[test]
fn size_and_length_count_characters_not_utf8_bytes() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    let eval = |query: &str| {
        let q = parser::parse_cypher(query).unwrap();
        executor.execute(&q).unwrap().rows[0].first().cloned()
    };

    // 'ø' is 2 UTF-8 bytes, each CJK char is 3, the emoji is 4.
    assert_eq!(eval("RETURN size('Tromsø')"), Some(Value::Int64(6)));
    assert_eq!(eval("RETURN length('Tromsø')"), Some(Value::Int64(6)));
    assert_eq!(eval("RETURN size('日本語')"), Some(Value::Int64(3)));
    assert_eq!(eval("RETURN length('日本語')"), Some(Value::Int64(3)));
    // One emoji is one char here: 🎉 is a single scalar value.
    assert_eq!(eval("RETURN size('héllo🎉')"), Some(Value::Int64(6)));

    // The point of the fix: size() now composes with the char-indexed
    // substring(), which byte counting made land mid-character.
    assert_eq!(
        eval("RETURN substring('Tromsø', size('Tromsø') - 1)"),
        Some(Value::String("ø".to_string()))
    );
}

/// A string that *looks* like a JSON list is still a string, and is measured
/// in characters like every other — matching Neo4j's `size(STRING)`. It used
/// to report the element count of the list it parsed to, which made
/// `size('[redacted]')` 1 and `size('[]')` 0: neither the characters nor the
/// bytes of anything the caller wrote, and no error to say so. Only the
/// argument's *type* decides; the rest of the legacy collect-as-JSON family
/// (`UNWIND`, indexing, `head`/`last`/`reverse`, `IN`) still coerces this
/// shape and is tracked separately.
#[test]
fn size_of_a_bracketed_string_counts_characters() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    let eval = |query: &str| {
        let q = parser::parse_cypher(query).unwrap();
        executor.execute(&q).unwrap().rows[0].first().cloned()
    };

    assert_eq!(eval("RETURN size('[1,2,3]')"), Some(Value::Int64(7)));
    assert_eq!(eval("RETURN length('[1,2,3]')"), Some(Value::Int64(7)));
    // Ordinary text that merely starts with '[' — the silent wrong answer.
    assert_eq!(eval("RETURN size('[redacted]')"), Some(Value::Int64(10)));
    assert_eq!(eval("RETURN size('[]')"), Some(Value::Int64(2)));
    // A real list still reports its element count: the type decides.
    assert_eq!(eval("RETURN size([1,2,3])"), Some(Value::Int64(3)));
}

#[test]
fn test_length_on_list() {
    let graph = DirGraph::new();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    // length() on a list should return element count
    let q = parser::parse_cypher("RETURN length([10,20,30,40])").unwrap();
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(4)));
}

#[test]
fn test_size_on_collect_result() {
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher(
        "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher("MATCH (n:Item) WITH collect(n.name) AS names RETURN size(names)")
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
    let setup = parser::parse_cypher(
        "CREATE (a:Item {cat: 'X', name: 'A'}), (b:Item {cat: 'X', name: 'B'}), \
         (c:Item {cat: 'X', name: 'C'}), (d:Item {cat: 'Y', name: 'D'})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher(
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
    let setup = parser::parse_cypher(
        "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher("MATCH (n:Item) RETURN count(n) + 1 AS cnt_plus").unwrap();
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
    let setup = parser::parse_cypher(
        "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    // No grouping — all rows aggregated
    let q = parser::parse_cypher("MATCH (n:Item) RETURN size(collect(n.name)) AS cnt").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&q).unwrap();
    assert_eq!(result.rows[0].first(), Some(&Value::Int64(3)));
}

#[test]
fn test_size_of_collect_grouped() {
    // size(collect(...)) with grouping
    let mut graph = DirGraph::new();
    let setup = parser::parse_cypher(
        "CREATE (a:Item {cat: 'X', name: 'A'}), (b:Item {cat: 'X', name: 'B'}), \
         (c:Item {cat: 'X', name: 'C'}), (d:Item {cat: 'Y', name: 'D'})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let setup = parser::parse_cypher(
        "CREATE (a:Well {name: 'W1', depth: 100}), \
         (b:Well {name: 'W2'}), \
         (c:Well {name: 'W3', depth: 300})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let q = parser::parse_cypher(
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
    let q = parser::parse_cypher(
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
    let setup = parser::parse_cypher(
        "CREATE (a:Well {name: 'W1', formation: 'Sandstone'}), \
         (b:Well {name: 'W2'}), \
         (c:Well {name: 'W3', formation: 'Limestone'})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &setup,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    // any() with collected node property access
    let q = parser::parse_cypher(
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
    let q2 = parser::parse_cypher(
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
