//! Interrupt coverage for the executor's rayon-parallel regions.
//!
//! Before this landed, four of the five query-path parallel sites polled
//! neither the deadline nor the cancel flag: a large projection ran to
//! completion no matter what the caller asked for. Each test here drives the
//! region above its row threshold with an interruption already pending, and
//! each is paired with a **below-threshold** call that must still succeed —
//! the sequential branch does not poll, so the `Err` above is only meaningful
//! if the `Ok` below holds. Without that pairing a test like this passes for
//! the wrong reason the day some upstream clause starts polling.
//!
//! The `TEST_PERIODIC_POLLS_BEFORE_INTERRUPT` hook is deliberately unused: it
//! is a `thread_local!`, so it is invisible to rayon workers. These drive the
//! real `&'static AtomicBool` cancel flag and the real deadline instead.

use super::*;

static CANCELLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// `count` rows all bound to the same `Person`, enough to take (or miss) the
/// parallel branch depending on `count` vs `RAYON_THRESHOLD`.
fn bound_rows(graph: &DirGraph, count: usize) -> ResultSet {
    let idx = graph
        .type_indices
        .get("Person")
        .expect("Person index")
        .get(0)
        .expect("at least one Person");
    let mut set = ResultSet::new();
    set.rows = (0..count)
        .map(|_| {
            let mut row = ResultRow::new();
            row.node_bindings.insert("n".to_string(), idx);
            row
        })
        .collect();
    set
}

/// `count` already-projected rows, for the `finalize_result` cell-materialise
/// region (which reads `projected`, not bindings).
fn projected_rows(count: usize) -> ResultSet {
    let mut set = ResultSet::new();
    set.columns = vec!["x".to_string()];
    set.rows = (0..count)
        .map(|i| {
            let mut row = ResultRow::new();
            row.projected
                .insert("x".to_string(), Value::Int64(i as i64));
            row
        })
        .collect();
    set
}

fn return_clause_of(query: &CypherQuery) -> &ReturnClause {
    match query.clauses.last().expect("clauses") {
        Clause::Return(rc) => rc,
        other => panic!("expected RETURN clause, got {other:?}"),
    }
}

#[test]
fn parallel_projection_observes_the_cancel_flag() {
    let graph = build_test_graph();
    let params = HashMap::new();
    let query = parser::parse_cypher("MATCH (n:Person) RETURN n.name AS name").unwrap();
    let clause = return_clause_of(&query);
    let executor = CypherExecutor::with_params(&graph, &params, None).with_cancel(Some(&CANCELLED));

    let cancelled = executor
        .execute_return_projection(clause, bound_rows(&graph, RAYON_THRESHOLD * 4))
        .unwrap_err();
    assert_eq!(cancelled, "Query cancelled");

    // Non-vacuity meter: identical call, one row under the threshold, takes
    // the sequential branch — which does not poll. If this ever starts
    // failing, the assertion above stopped proving anything about the
    // parallel branch and this file needs a new probe.
    assert!(
        executor
            .execute_return_projection(clause, bound_rows(&graph, RAYON_THRESHOLD - 1))
            .is_ok(),
        "below-threshold projection must still take the unpolled sequential branch"
    );
}

#[test]
fn parallel_projection_observes_the_deadline() {
    let graph = build_test_graph();
    let params = HashMap::new();
    let query = parser::parse_cypher("MATCH (n:Person) RETURN n.name AS name").unwrap();
    let clause = return_clause_of(&query);
    let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let executor = CypherExecutor::with_params(&graph, &params, Some(past));

    let timed_out = executor
        .execute_return_projection(clause, bound_rows(&graph, RAYON_THRESHOLD * 4))
        .unwrap_err();
    assert!(
        timed_out.starts_with("Query timed out."),
        "unexpected error: {timed_out}"
    );

    assert!(
        executor
            .execute_return_projection(clause, bound_rows(&graph, RAYON_THRESHOLD - 1))
            .is_ok(),
        "below-threshold projection must still take the unpolled sequential branch"
    );
}

#[test]
fn parallel_window_projection_observes_the_cancel_flag() {
    let graph = build_test_graph();
    let params = HashMap::new();
    let query = parser::parse_cypher(
        "MATCH (n:Person) RETURN n.name AS name, row_number() OVER (ORDER BY n.name) AS rn",
    )
    .unwrap();
    let clause = return_clause_of(&query);
    let executor = CypherExecutor::with_params(&graph, &params, None).with_cancel(Some(&CANCELLED));

    let cancelled = executor
        .execute_return_with_windows(clause, bound_rows(&graph, RAYON_THRESHOLD * 4))
        .unwrap_err();
    assert_eq!(cancelled, "Query cancelled");

    assert!(
        executor
            .execute_return_with_windows(clause, bound_rows(&graph, RAYON_THRESHOLD - 1))
            .is_ok(),
        "below-threshold window projection must still take the sequential branch"
    );
}

#[test]
fn parallel_result_materialisation_observes_the_cancel_flag() {
    let graph = build_test_graph();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None).with_cancel(Some(&CANCELLED));

    let cancelled = executor
        .finalize_result(projected_rows(RAYON_THRESHOLD * 4))
        .unwrap_err();
    assert_eq!(cancelled, "Query cancelled");

    assert!(
        executor
            .finalize_result(projected_rows(RAYON_THRESHOLD - 1))
            .is_ok(),
        "below-threshold materialisation must still take the sequential branch"
    );
}
