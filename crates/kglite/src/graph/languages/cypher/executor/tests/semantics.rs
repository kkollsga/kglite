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
