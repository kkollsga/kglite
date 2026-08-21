//! **Absolute goldens for expression semantics that used to fail silently.**
//!
//! Each group here pins a case where the engine answered with a wrong value or
//! a `null` instead of reporting that it could not answer. The pre-fix result
//! is written next to every assertion, so the file doubles as the record of
//! what changed — and each of these was observed red against the shipped
//! 0.15.9 extension before the fix was written (R1: a verification must be able
//! to fail, and these were seen failing).
//!
//! The differential corpus is deliberately *not* the gate for any of this:
//! these are executor/parser semantics with one right answer, not planner
//! rewrites whose only contract is "same answer with the pass on and off".

use super::*;

/// Run a read query for its error, or panic if it unexpectedly succeeded.
fn query_error(graph: &DirGraph, query: &str) -> String {
    let parsed = match parser::parse_cypher(query) {
        Ok(parsed) => parsed,
        // Parse-time rejections (duplicate result columns) arrive here.
        Err(e) => return e.to_string(),
    };
    let no_params = HashMap::new();
    match CypherExecutor::with_params(graph, &no_params, None).execute(&parsed) {
        Ok(result) => panic!(
            "query unexpectedly succeeded: {query}\n  rows: {:?}",
            result.rows
        ),
        Err(e) => e,
    }
}

/// Run a read query and return its single row's single cell.
fn one_cell(graph: &DirGraph, query: &str) -> Value {
    let parsed = parser::parse_cypher(query)
        .unwrap_or_else(|e| panic!("query failed to parse: {query}\n  error: {e}"));
    let no_params = HashMap::new();
    let result = CypherExecutor::with_params(graph, &no_params, None)
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("query failed: {query}\n  error: {e}"));
    assert_eq!(result.rows.len(), 1, "expected one row from: {query}");
    assert_eq!(result.rows[0].len(), 1, "expected one column from: {query}");
    result.rows[0][0].clone()
}

// ========================================================================
// Duplicate result columns
// ========================================================================
//
// Pre-fix, every case below "succeeded" and lost data:
//
//   RETURN 1 AS x, 2 AS x                  -> {x: 2, x: null}
//   MATCH … RETURN n.a AS x, n.b AS x      -> {x: <n.b>}   (n.a gone)
//   MATCH … RETURN count(n) AS c, count(n) AS c -> {c: null}
//   MATCH … WITH n.a AS x, n.b AS x …      -> x is n.b     (n.a gone)
//
// The projection is one name-keyed map per row, so two items sharing a name
// are never two columns. Neo4j rejects the same shape.

#[test]
fn duplicate_return_aliases_are_rejected() {
    let graph = DirGraph::new();
    for query in [
        "RETURN 1 AS x, 2 AS x",
        "RETURN 1 AS x, 2 AS y, 3 AS x",
        "MATCH (n:Person) RETURN n.a AS x, n.b AS x",
        "MATCH (n:Person) RETURN DISTINCT n.a AS x, n.b AS x",
        "MATCH (n:Person) RETURN count(n) AS c, count(n) AS c",
        "MATCH (n:Person) WITH n.a AS x, n.b AS x RETURN x",
        "MATCH (n:Person) RETURN n.a, n.a",
        // The backtick-quoted spelling is the same name.
        "RETURN 1 AS `x`, 2 AS x",
    ] {
        let error = query_error(&graph, query);
        assert!(
            error.contains("Multiple result columns with the same name are not supported"),
            "expected a duplicate-column rejection for `{query}`, got: {error}"
        );
    }
}

#[test]
fn distinct_column_names_still_project() {
    // Non-vacuity: the check rejects collisions, not projections. Column names
    // are case-sensitive, so `x` and `X` are two columns.
    let graph = DirGraph::new();
    let parsed = parser::parse_cypher("RETURN 1 AS x, 2 AS X, 3 AS y").unwrap();
    let no_params = HashMap::new();
    let result = CypherExecutor::with_params(&graph, &no_params, None)
        .execute(&parsed)
        .unwrap();
    assert_eq!(result.columns, vec!["x", "X", "y"]);
    assert_eq!(
        result.rows[0],
        vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]
    );
}

// ========================================================================
// datetime() time-of-day and zone
// ========================================================================
//
// Pre-fix results, all silent:
//
//   datetime('2024-01-15T10:30:00Z')      -> 2024-01-15T00:00:00
//   datetime('2024-01-15T10:30:00+02:00') -> 2024-01-15T00:00:00
//   datetime('2024-01-15T10:30:00.500')   -> 2024-01-15T00:00:00
//   datetime('2024-01-15T10:30')          -> 2024-01-15T00:00:00
//   localdatetime('2024-01-15T10:30:00Z') -> 2024-01-15T00:00:00
//
// The bare-date fallback split *any* input on 'T' and re-parsed the date half,
// so every form the exact `%Y-%m-%dT%H:%M:%S` parse missed answered midnight.

fn timestamp(y: i32, m: u32, d: u32, hh: u32, mm: u32, ss: u32) -> Value {
    Value::Timestamp(
        chrono::NaiveDate::from_ymd_opt(y, m, d)
            .unwrap()
            .and_hms_opt(hh, mm, ss)
            .unwrap(),
    )
}

#[test]
fn datetime_preserves_time_and_normalises_the_zone_to_utc() {
    let graph = DirGraph::new();
    let cases: &[(&str, Value)] = &[
        // Zone-less: unchanged, and the case that always worked.
        (
            "RETURN datetime('2024-01-15T10:30:00') AS d",
            timestamp(2024, 1, 15, 10, 30, 0),
        ),
        // `Z` is UTC — the reading is already UTC.
        (
            "RETURN datetime('2024-01-15T10:30:00Z') AS d",
            timestamp(2024, 1, 15, 10, 30, 0),
        ),
        // A positive offset is *applied*, not dropped: 10:30+02:00 is 08:30 UTC.
        (
            "RETURN datetime('2024-01-15T10:30:00+02:00') AS d",
            timestamp(2024, 1, 15, 8, 30, 0),
        ),
        // A negative offset likewise, across the date boundary.
        (
            "RETURN datetime('2024-01-15T01:30:00-05:00') AS d",
            timestamp(2024, 1, 15, 6, 30, 0),
        ),
        // Sub-second precision truncates (Value::Timestamp is second-precision)
        // without taking the time with it.
        (
            "RETURN datetime('2024-01-15T10:30:00.500Z') AS d",
            timestamp(2024, 1, 15, 10, 30, 0),
        ),
        // Minute precision.
        (
            "RETURN datetime('2024-01-15T10:30') AS d",
            timestamp(2024, 1, 15, 10, 30, 0),
        ),
        // A bare date still means midnight.
        (
            "RETURN datetime('2024-01-15') AS d",
            timestamp(2024, 1, 15, 0, 0, 0),
        ),
        // Unparseable input keeps the documented Null contract — but a string
        // that *has* a time part and does not parse is now Null rather than a
        // silently invented midnight.
        ("RETURN datetime('not-a-date') AS d", Value::Null),
        ("RETURN datetime('2024-01-15T25:99:99') AS d", Value::Null),
        // A year wider than four digits is representable in
        // `NaiveDateTime`, so it parses rather than being refused for
        // chrono's unsigned-`%Y` rule.
        (
            "RETURN datetime('10000-01-01T00:00:00') AS d",
            timestamp(10000, 1, 1, 0, 0, 0),
        ),
    ];
    for (query, expected) in cases {
        assert_eq!(one_cell(&graph, query), *expected, "for: {query}");
    }
}

#[test]
fn localdatetime_keeps_the_wall_clock_reading() {
    // `localdatetime` names a zone-less local reading, so an offset-bearing
    // input keeps its wall clock and loses only the zone label — the deliberate
    // difference from `datetime()`, which normalises. Neither drops the time.
    let graph = DirGraph::new();
    assert_eq!(
        one_cell(
            &graph,
            "RETURN localdatetime('2024-01-15T10:30:00+02:00') AS d"
        ),
        timestamp(2024, 1, 15, 10, 30, 0)
    );
    assert_eq!(
        one_cell(&graph, "RETURN localdatetime('2024-01-15T10:30:00Z') AS d"),
        timestamp(2024, 1, 15, 10, 30, 0)
    );
    assert_eq!(
        one_cell(&graph, "RETURN localdatetime('2024-01-15') AS d"),
        timestamp(2024, 1, 15, 0, 0, 0)
    );
}

// ========================================================================
// Integer overflow and division by zero
// ========================================================================
//
// Pre-fix results, all silent:
//
//   RETURN 9223372036854775807 + 1  -> -9223372036854775808
//   RETURN 9223372036854775807 * 2  -> -2
//   RETURN 1 / 0                    -> null
//   RETURN 1 % 0                    -> null

#[test]
fn integer_overflow_and_division_by_zero_are_query_errors() {
    let graph = DirGraph::new();
    let cases: &[(&str, &str)] = &[
        (
            "RETURN 9223372036854775807 + 1 AS n",
            "Integer overflow in addition",
        ),
        (
            "RETURN (-9223372036854775807 - 1) - 1 AS n",
            "Integer overflow in subtraction",
        ),
        (
            "RETURN 9223372036854775807 * 2 AS n",
            "Integer overflow in multiplication",
        ),
        (
            "RETURN (-9223372036854775807 - 1) / -1 AS n",
            "Integer overflow in division",
        ),
        (
            "RETURN (-9223372036854775807 - 1) % -1 AS n",
            "Integer overflow in modulo",
        ),
        ("RETURN 1 / 0 AS n", "Integer division by zero"),
        ("RETURN 1 % 0 AS n", "Integer modulo by zero"),
    ];
    for (query, expected) in cases {
        let error = query_error(&graph, query);
        assert!(
            error.contains(expected),
            "expected `{expected}` for `{query}`, got: {error}"
        );
    }
}

#[test]
fn ordinary_integer_arithmetic_is_unchanged() {
    // Non-vacuity for the group above: the checks reject overflow, not
    // arithmetic. Includes the truncate-toward-zero rounding and the
    // date-bucketing shape the integer-division semantics exist for.
    let graph = DirGraph::new();
    for (query, expected) in [
        (
            "RETURN 9223372036854775806 + 1 AS n",
            Value::Int64(i64::MAX),
        ),
        ("RETURN -7 / 2 AS n", Value::Int64(-3)),
        ("RETURN 1967 / 10 * 10 AS n", Value::Int64(1960)),
        ("RETURN 7 % 3 AS n", Value::Int64(1)),
        // Float division by zero keeps its Null (no wire format carries Inf).
        ("RETURN 1.0 / 0 AS n", Value::Null),
    ] {
        assert_eq!(one_cell(&graph, query), expected, "for: {query}");
    }
}

// ========================================================================
// Fused node-scan aggregates over a mixed-type property
// ========================================================================
//
// Pre-fix, the fused operator's inline accumulator counted every non-null
// value but summed only the numeric ones, then divided one by the other:
//
//   n.v ∈ {10, 20, 'hello'}   MATCH (n:S) RETURN avg(n.v)  -> 10.0  (30/3)
//
// `sum(n.v)` on the same input already counted only numerics, so the two
// disagreed — 30 summed over "3 values". The unfused path (materialized
// aggregation) answered 15.0 throughout, which is why the differential
// corpus is the other half of this fix's gate.

/// A graph of `:S` nodes whose `v` property holds `values` positionally.
fn build_mixed_property_graph(values: Vec<Value>) -> DirGraph {
    let mut graph = DirGraph::new();
    for (i, value) in values.into_iter().enumerate() {
        let node = NodeData::new(
            Value::UniqueId(i as u32 + 1),
            Value::String(format!("s{i}")),
            "S".to_string(),
            HashMap::from([("v".to_string(), value)]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("S".to_string())
            .push(idx);
    }
    graph
}

/// Run `query` twice — through the optimizer (the fused operator) and
/// unoptimized (the materialized path) — and return both row sets.
fn optimized_and_unoptimized(graph: &DirGraph, query: &str) -> (Vec<Value>, Vec<Value>) {
    let params = HashMap::new();
    let mut optimized = parser::parse_cypher(query).unwrap();
    crate::graph::languages::cypher::planner::optimize(&mut optimized, graph, &params);
    assert!(
        optimized
            .clauses
            .iter()
            .any(|c| matches!(c, Clause::FusedNodeScanAggregate { .. })),
        "non-vacuity: `{query}` did not fuse, so it would not exercise the \
         inline accumulator at all"
    );
    let unoptimized = parser::parse_cypher(query).unwrap();

    let run = |q: &CypherQuery| -> Vec<Value> {
        let result = CypherExecutor::with_params(graph, &params, None)
            .execute(q)
            .unwrap_or_else(|e| panic!("query failed: {query}\n  error: {e}"));
        assert_eq!(result.rows.len(), 1, "expected one row from: {query}");
        (0..result.rows[0].len())
            .map(|i| result.rows[0][i].clone())
            .collect()
    };
    (run(&optimized), run(&unoptimized))
}

#[test]
fn fused_avg_divides_by_the_numeric_count_not_the_non_null_count() {
    let graph = build_mixed_property_graph(vec![
        Value::Int64(10),
        Value::Int64(20),
        Value::String("hello".to_string()),
    ]);

    let (fused, materialized) = optimized_and_unoptimized(
        &graph,
        "MATCH (n:S) RETURN avg(n.v) AS a, sum(n.v) AS s, count(n.v) AS c",
    );

    // avg = 30/2, not 30/3 (the pre-fix answer was Float64(10.0)).
    assert_eq!(fused[0], Value::Float64(15.0), "avg over [10, 20, 'hello']");
    // sum and count are unchanged: sum sees numbers, count sees non-nulls.
    assert_eq!(fused[1], Value::Int64(30), "sum over [10, 20, 'hello']");
    assert_eq!(fused[2], Value::Int64(3), "count over [10, 20, 'hello']");
    assert_eq!(fused, materialized, "fused vs materialized aggregation");
}

#[test]
fn fused_avg_and_sum_over_zero_numeric_values_match_the_unfused_path() {
    let graph = build_mixed_property_graph(vec![
        Value::String("a".to_string()),
        Value::String("b".to_string()),
    ]);

    let (fused, materialized) =
        optimized_and_unoptimized(&graph, "MATCH (n:S) RETURN avg(n.v) AS a, sum(n.v) AS s");

    // No numeric input at all: avg is null and sum is 0 — the same answers
    // `collect_numeric_values` produces when it comes back empty.
    assert_eq!(fused[0], Value::Null, "avg over ['a', 'b']");
    assert_eq!(fused[1], Value::Int64(0), "sum over ['a', 'b']");
    assert_eq!(fused, materialized, "fused vs materialized aggregation");
}

#[test]
fn fused_sum_keeps_the_unfused_paths_numeric_type_on_mixed_columns() {
    // `sum()`'s Int64-vs-Float64 choice is the unfused (streaming) path's
    // rule: every numeric input must be an Int64 and the total must be whole.
    // Deriving it from `min()` instead made a single string cell flip the
    // type, because a string sorts below every number in the cross-type order.
    for values in [
        vec![
            Value::Int64(10),
            Value::Int64(20),
            Value::String("x".into()),
        ],
        vec![Value::String("x".into()), Value::Int64(10)],
        vec![Value::Null, Value::Int64(1), Value::Int64(2)],
        vec![Value::Int64(1), Value::Float64(2.5)],
        vec![Value::Float64(1.5), Value::Int64(2)],
        vec![Value::Int64(1), Value::Int64(2)],
    ] {
        let graph = build_mixed_property_graph(values.clone());
        let (fused, materialized) = optimized_and_unoptimized(
            &graph,
            "MATCH (n:S) RETURN sum(n.v) AS s, avg(n.v) AS a, count(n.v) AS c",
        );
        assert_eq!(fused, materialized, "fused vs materialized over {values:?}");
    }
}

// ========================================================================
// Uncompilable regexes on the fused scan
// ========================================================================
//
// The fused scan drops a row whose WHERE predicate cannot be evaluated
// rather than failing the query — that is how an unbound binding behaves,
// and it stays. A pattern that does not compile is not that case: it is
// wrong for every row, and the unfused path has always raised it. Swallowing
// it answered `WHERE n.v =~ '['` with a silent empty result, and turned a
// lookaround pattern (valid in Neo4j, unsupported by the `regex` crate) into
// "no matches" rather than "unsupported".

#[test]
fn fused_scan_raises_on_a_pattern_that_does_not_compile() {
    let graph = build_mixed_property_graph(vec![
        Value::String("Alice".into()),
        Value::String("Bob".into()),
    ]);
    let params = HashMap::new();

    for pattern in ["[", "A(?=l)ice", r"(a)\\1"] {
        let query = format!("MATCH (n:S) WHERE n.v =~ '{pattern}' RETURN count(*) AS c");
        let mut optimized = parser::parse_cypher(&query).unwrap();
        crate::graph::languages::cypher::planner::optimize(&mut optimized, &graph, &params);
        assert!(
            optimized
                .clauses
                .iter()
                .any(|c| matches!(c, Clause::FusedNodeScanAggregate { .. })),
            "non-vacuity: `{query}` did not fuse, so it would not exercise the \
             fused scan's predicate at all"
        );

        let err = CypherExecutor::with_params(&graph, &params, None)
            .execute(&optimized)
            .expect_err(&format!("`{query}` must not answer with rows"));
        assert!(
            err.contains("Invalid regular expression"),
            "fused scan error for `{pattern}`: {err}"
        );

        // The unfused path's answer, for the same query.
        let unoptimized = parser::parse_cypher(&query).unwrap();
        let unfused_err = CypherExecutor::with_params(&graph, &params, None)
            .execute(&unoptimized)
            .expect_err(&format!("`{query}` must not answer with rows unfused"));
        assert_eq!(err, unfused_err, "fused vs unfused error for `{pattern}`");
    }
}

// ========================================================================
// Unbound `$parameter`s on the fused scan
// ========================================================================
//
// Same shape as the uncompilable regex above, one class wider. A parameter
// the caller never bound is missing for every row and no row can supply it,
// and the unfused path has always raised it — but the fused scan swallowed it
// with the "this predicate does not evaluate for this row" errors it drops by
// design, so `WHERE v.flag = $flag RETURN count(v)` answered `0` and no error.
// A zero count is the worst possible answer here: the caller reads it as "the
// graph has none of those" off their own mistake.

#[test]
fn fused_scan_raises_on_an_unbound_parameter() {
    let graph = build_mixed_property_graph(vec![
        Value::String("Alice".into()),
        Value::String("Bob".into()),
    ]);
    let params = HashMap::new();

    let query = "MATCH (n:S) WHERE n.v = $missing RETURN count(*) AS c";
    let mut optimized = parser::parse_cypher(query).unwrap();
    crate::graph::languages::cypher::planner::optimize(&mut optimized, &graph, &params);
    assert!(
        optimized
            .clauses
            .iter()
            .any(|c| matches!(c, Clause::FusedNodeScanAggregate { .. })),
        "non-vacuity: `{query}` did not fuse, so it would not exercise the \
         fused scan's predicate at all"
    );

    let err = CypherExecutor::with_params(&graph, &params, None)
        .execute(&optimized)
        .expect_err("an unbound parameter must not answer with a count");
    assert!(err.contains("Missing parameter: $missing"), "fused: {err}");

    // The unfused path's answer, for the same query.
    let unoptimized = parser::parse_cypher(query).unwrap();
    let unfused_err = CypherExecutor::with_params(&graph, &params, None)
        .execute(&unoptimized)
        .expect_err("an unbound parameter must not answer with a count unfused");
    assert_eq!(err, unfused_err, "fused vs unfused error for `{query}`");
}

#[test]
fn fused_scan_still_counts_with_the_parameter_bound() {
    // The golden the error must not have cost.
    let graph = build_mixed_property_graph(vec![
        Value::String("Alice".into()),
        Value::String("Bob".into()),
        Value::String("Alice".into()),
    ]);
    let params = HashMap::from([("bound".to_string(), Value::String("Alice".into()))]);

    let query = "MATCH (n:S) WHERE n.v = $bound RETURN count(*) AS c";
    let mut optimized = parser::parse_cypher(query).unwrap();
    crate::graph::languages::cypher::planner::optimize(&mut optimized, &graph, &params);
    let result = CypherExecutor::with_params(&graph, &params, None)
        .execute(&optimized)
        .expect("a bound parameter still answers");
    assert_eq!(result.rows[0][0], Value::Int64(2));
}

#[test]
fn fused_scan_still_drops_rows_whose_predicate_cannot_be_evaluated() {
    // A valid pattern against a non-string cell: the comparison yields
    // "no match" for that row, never an error, and the scan keeps counting.
    let graph = build_mixed_property_graph(vec![
        Value::String("Alice".into()),
        Value::Int64(7),
        Value::Null,
    ]);

    let (fused, materialized) = optimized_and_unoptimized(
        &graph,
        "MATCH (n:S) WHERE n.v =~ '^A.*' RETURN count(*) AS c",
    );
    assert_eq!(fused[0], Value::Int64(1), "only the string cell matches");
    assert_eq!(fused, materialized, "fused vs materialized filter");
}

/// Run `query` unoptimized — what `disable_optimizer=True` does — and return
/// its one row. Without the fusion pass there is no `FusedNodeScanAggregate`
/// clause, so an aggregate the MATCH clause's inline accumulator declines
/// falls through to the materialized executor.
fn materialized_aggregation_row(graph: &DirGraph, query: &str) -> Vec<Value> {
    let params = HashMap::new();
    let parsed = parser::parse_cypher(query).unwrap();
    let result = CypherExecutor::with_params(graph, &params, None)
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("query failed: {query}\n  error: {e}"));
    assert_eq!(result.rows.len(), 1, "expected one row from: {query}");
    (0..result.rows[0].len())
        .map(|i| result.rows[0][i].clone())
        .collect()
}

#[test]
fn materialized_sum_keeps_the_streaming_paths_numeric_type() {
    // `sum()`'s Int64-vs-Float64 choice is one rule across all three
    // aggregation paths: integer iff every numeric input was an `Int64` and
    // the total is whole. The materialized executor instead probed the
    // group's *first* row, so a leading string or null — which says nothing
    // about the numerics behind it — flipped the type. Pre-fix, the first two
    // cases below answered `Float64(10.0)` and `Float64(3.0)` here while the
    // streaming and fused paths answered `Int64`.
    //
    // Two shapes, because the materialized executor decides the type twice.
    //   * `sum` beside `median`: no grouping key, and `median` is one of the
    //     aggregates both the streaming recognizer and the fused scan refuse,
    //     so the whole RETURN falls to `evaluate_aggregate_with_rows`.
    //   * a literal grouping key: grouping routes through
    //     `try_fused_numeric_aggregation`, and the literal keeps the MATCH
    //     clause's inline accumulator from absorbing the scan.
    for (values, expected) in [
        (
            vec![Value::String("x".into()), Value::Int64(10)],
            Value::Int64(10),
        ),
        (
            vec![Value::Null, Value::Int64(1), Value::Int64(2)],
            Value::Int64(3),
        ),
        (
            vec![Value::Float64(1.5), Value::Int64(2)],
            Value::Float64(3.5),
        ),
        (
            vec![Value::Int64(2), Value::Float64(1.5)],
            Value::Float64(3.5),
        ),
        (vec![Value::Int64(1), Value::Int64(2)], Value::Int64(3)),
        (
            vec![Value::String("x".into()), Value::String("y".into())],
            Value::Int64(0),
        ),
    ] {
        let graph = build_mixed_property_graph(values.clone());

        let unkeyed = "MATCH (n:S) RETURN sum(n.v) AS s, median(n.v) AS m";
        // The streaming recognizer refuses `median`, so this shape cannot
        // reach `stream::aggregate` either — the materialized executor is the
        // only thing left that can answer it.
        assert!(
            crate::graph::languages::cypher::executor::stream::aggregate::try_compile_specs(
                match &parser::parse_cypher(unkeyed).unwrap().clauses[1] {
                    Clause::Return(rc) => rc,
                    other => panic!("expected a RETURN clause, got {other:?}"),
                }
            )
            .is_err(),
            "non-vacuity: the streaming path accepted `{unkeyed}`"
        );
        assert_eq!(
            materialized_aggregation_row(&graph, unkeyed)[0],
            expected,
            "unkeyed sum over {values:?}"
        );

        let keyed = "MATCH (n:S) RETURN 1 AS k, sum(n.v) AS s";
        assert_eq!(
            materialized_aggregation_row(&graph, keyed)[1],
            expected,
            "grouped sum over {values:?}"
        );

        // DISTINCT bails out of `try_fused_numeric_aggregation` back to
        // `evaluate_aggregate_with_rows` — the same rule has to hold there.
        let distinct = "MATCH (n:S) RETURN 1 AS k, sum(DISTINCT n.v) AS s";
        assert_eq!(
            materialized_aggregation_row(&graph, distinct)[1],
            expected,
            "grouped sum(DISTINCT) over {values:?}"
        );
    }
}

#[test]
fn materialized_distinct_aggregates_dedup_on_the_value() {
    // A DISTINCT aggregate dedups on the `Value` — the same key
    // `RETURN DISTINCT`, `WITH DISTINCT` and `count(DISTINCT …)` use. The
    // materialized executor had two private keys instead, and both lost data
    // no other path lost:
    //
    //   * the numeric collector keyed on the `f64` **bit pattern**, so
    //     `Int64(1)` and `Float64(1.0)` were one value here and two
    //     everywhere else — `sum(DISTINCT …)` over `[1, 1.0, 2]` answered
    //     `Int64(3)` on this path and `Float64(4.0)` on the streaming one;
    //   * `collect(DISTINCT …)` keyed on `format_value_compact`, which renders
    //     `Int64(1)` and `String("1")` both as `"1"` — so the list came back
    //     one element short of the `count(DISTINCT …)` computed beside it.
    //
    // Every query below is refused by the streaming recognizer (`median` and
    // `collect` are both outside its reach), so nothing but
    // `evaluate_aggregate_with_rows` can answer them — asserted, not assumed.
    for (values, query, expected) in [
        (
            vec![Value::Int64(1), Value::Float64(1.0), Value::Int64(2)],
            "MATCH (n:S) RETURN sum(DISTINCT n.v) AS s, median(n.v) AS m",
            Value::Float64(4.0),
        ),
        (
            vec![Value::Int64(1), Value::Int64(1), Value::Int64(2)],
            "MATCH (n:S) RETURN sum(DISTINCT n.v) AS s, median(n.v) AS m",
            Value::Int64(3),
        ),
        // `0.0` and `-0.0` are one value under `Value`'s `Eq`/`Hash`; the bit
        // pattern said two, so this summed to `1.5` over three values.
        (
            vec![
                Value::Float64(0.0),
                Value::Float64(-0.0),
                Value::Float64(1.5),
            ],
            "MATCH (n:S) RETURN avg(DISTINCT n.v) AS a, median(n.v) AS m",
            Value::Float64(0.75),
        ),
        (
            vec![Value::Int64(1), Value::String("1".into())],
            "MATCH (n:S) RETURN collect(DISTINCT n.v) AS l",
            Value::List(vec![Value::Int64(1), Value::String("1".into())]),
        ),
    ] {
        let graph = build_mixed_property_graph(values.clone());
        assert!(
            crate::graph::languages::cypher::executor::stream::aggregate::try_compile_specs(
                match &parser::parse_cypher(query).unwrap().clauses[1] {
                    Clause::Return(rc) => rc,
                    other => panic!("expected a RETURN clause, got {other:?}"),
                }
            )
            .is_err(),
            "non-vacuity: the streaming path accepted `{query}`"
        );
        assert_eq!(
            materialized_aggregation_row(&graph, query)[0],
            expected,
            "distinct aggregate over {values:?}"
        );
    }
}
