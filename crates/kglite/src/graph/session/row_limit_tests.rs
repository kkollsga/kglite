//! `ExecuteOptions::row_limit` — a *retention* cap, not a work budget.
//!
//! The pairing with `max_work_units` is the whole point and the whole risk:
//! the two knobs read alike and behave oppositely (budget errors, cap
//! truncates), so these tests pin the difference and the interactions where a
//! naive implementation quietly returns the wrong rows rather than fewer of
//! the right ones — UNION arms capped before the set operation, an ORDER BY
//! whose top-N is not the top-N, an `EXCEPT` that stops excluding.
//!
//! The truncation signal is tested as hard as the truncation: a cap that fires
//! silently is the failure mode this feature exists to prevent, so every
//! assertion on rows has a companion assertion on `total_rows`.

use std::collections::HashMap;

use super::execute::{execute_mut, execute_read, ExecuteOptions, ExecuteOutcome};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::languages::cypher::result::QueryDiagnostics;

fn empty_params() -> HashMap<String, Value> {
    HashMap::new()
}

/// `n` `Item` nodes with `seq` running 1..=n, so ordering assertions can name
/// exactly which rows must survive a cap.
fn seeded(n: i64) -> DirGraph {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = DirGraph::new();
    for seq in 1..=n {
        execute_mut(
            &mut graph,
            &format!("CREATE (:Item {{id: {seq}, seq: {seq}}})"),
            &opts,
        )
        .expect("seed write");
    }
    graph
}

fn read_capped(graph: &DirGraph, query: &str, row_limit: Option<usize>) -> ExecuteOutcome {
    let params = empty_params();
    let mut opts = ExecuteOptions::eager(&params);
    opts.row_limit = row_limit;
    execute_read(graph, query, &opts).expect("read")
}

fn diagnostics(outcome: &ExecuteOutcome) -> &QueryDiagnostics {
    outcome
        .result
        .diagnostics
        .as_ref()
        .expect("every execution carries diagnostics")
}

/// The `seq` column of every retained row, in order.
fn seqs(outcome: &ExecuteOutcome) -> Vec<i64> {
    outcome
        .result
        .rows
        .iter()
        .map(|row| match row.first() {
            Some(Value::Int64(n)) => *n,
            other => panic!("expected an integer seq, got {other:?}"),
        })
        .collect()
}

#[test]
fn cap_truncates_and_reports_the_exact_pre_truncation_total() {
    let graph = seeded(50);
    let outcome = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq ORDER BY n.seq",
        Some(5),
    );

    assert_eq!(seqs(&outcome), vec![1, 2, 3, 4, 5]);
    let d = diagnostics(&outcome);
    assert_eq!(d.row_limit, Some(5));
    assert_eq!(
        d.total_rows,
        Some(50),
        "the total must be the exact pre-truncation count, not the retained count"
    );
}

/// A cap that does not bite must leave *no* truncation signal — otherwise a
/// caller cannot tell a full result from a cut one.
#[test]
fn under_cap_result_is_untouched_and_carries_no_truncation_signal() {
    let graph = seeded(5);
    let outcome = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq ORDER BY n.seq",
        Some(500),
    );

    assert_eq!(seqs(&outcome), vec![1, 2, 3, 4, 5]);
    let d = diagnostics(&outcome);
    assert_eq!(d.row_limit, Some(500), "the cap in force is still echoed");
    assert_eq!(
        d.total_rows, None,
        "nothing was dropped, so nothing to report"
    );
    assert!(
        !d.warnings.iter().any(|w| w.contains("row_limit")),
        "an untruncated result must not be warned about: {:?}",
        d.warnings
    );
}

/// Exactly at the cap is not over it.
#[test]
fn cap_equal_to_row_count_does_not_truncate() {
    let graph = seeded(5);
    let outcome = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq ORDER BY n.seq",
        Some(5),
    );

    assert_eq!(seqs(&outcome), vec![1, 2, 3, 4, 5]);
    assert_eq!(diagnostics(&outcome).total_rows, None);
}

/// No cap configured means the diagnostics say so, so `row_limit: None` and
/// `row_limit: Some(huge)` stay distinguishable.
#[test]
fn no_cap_leaves_both_fields_none() {
    let graph = seeded(5);
    let outcome = read_capped(&graph, "MATCH (n:Item) RETURN n.seq", None);
    let d = diagnostics(&outcome);
    assert_eq!(d.row_limit, None);
    assert_eq!(d.total_rows, None);
}

/// `Some(0)` is legal: retain nothing, still count everything.
#[test]
fn cap_of_zero_retains_nothing_and_still_counts() {
    let graph = seeded(7);
    let outcome = read_capped(&graph, "MATCH (n:Item) RETURN n.seq", Some(0));

    assert!(outcome.result.rows.is_empty());
    assert_eq!(
        outcome.result.columns,
        vec!["n.seq".to_string()],
        "a RETURN's columns survive a zero cap — only rows are capped"
    );
    let d = diagnostics(&outcome);
    assert_eq!(d.row_limit, Some(0));
    assert_eq!(d.total_rows, Some(7));
}

/// The cap runs after ORDER BY, so the retained rows are the genuine top-N of
/// the sorted answer — not N arbitrary rows that happen to be first in scan
/// order. Descending, so a truncation applied *before* the sort would return
/// the wrong five values rather than merely a differently-ordered five.
#[test]
fn cap_applies_after_order_by_so_it_keeps_the_real_top_n() {
    let graph = seeded(50);
    let outcome = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq ORDER BY n.seq DESC",
        Some(5),
    );

    assert_eq!(seqs(&outcome), vec![50, 49, 48, 47, 46]);
    assert_eq!(diagnostics(&outcome).total_rows, Some(50));
}

/// An explicit `LIMIT` is a clause and runs first, so the effective cap is the
/// smaller of the two — and the reported total is what the query *returned*
/// (post-LIMIT), which is the number a "showing X of Y" banner needs.
#[test]
fn query_limit_and_row_limit_take_the_minimum() {
    let graph = seeded(50);

    let cap_wins = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq ORDER BY n.seq LIMIT 20",
        Some(5),
    );
    assert_eq!(seqs(&cap_wins), vec![1, 2, 3, 4, 5]);
    assert_eq!(
        diagnostics(&cap_wins).total_rows,
        Some(20),
        "the total is the rows the query produced, which LIMIT had already bounded"
    );

    let limit_wins = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq ORDER BY n.seq LIMIT 3",
        Some(5),
    );
    assert_eq!(seqs(&limit_wins), vec![1, 2, 3]);
    assert_eq!(diagnostics(&limit_wins).total_rows, None);
}

/// Aggregation output rows are result rows like any other.
#[test]
fn cap_applies_to_aggregation_output_rows() {
    let graph = seeded(9);
    let outcome = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq AS seq, count(*) AS c ORDER BY seq",
        Some(4),
    );

    assert_eq!(outcome.result.rows.len(), 4);
    let d = diagnostics(&outcome);
    assert_eq!(
        d.total_rows,
        Some(9),
        "nine groups were computed; four were kept"
    );
}

/// A whole-set aggregate is one row and must never be capped away by a cap
/// that was sized for a row listing... unless the caller asked for zero.
#[test]
fn scalar_aggregate_survives_a_generous_cap() {
    let graph = seeded(1000);
    let outcome = read_capped(&graph, "MATCH (n:Item) RETURN count(*) AS c", Some(5));

    assert_eq!(outcome.result.rows.len(), 1);
    assert_eq!(outcome.result.rows[0][0], Value::Int64(1000));
    assert_eq!(diagnostics(&outcome).total_rows, None);
}

/// The regression this feature could most easily introduce: capping a UNION
/// arm instead of the union. `EXCEPT` makes it visible as a *wrong answer* —
/// a truncated right side stops excluding the rows it should exclude.
#[test]
fn set_operations_see_their_full_arms() {
    let graph = seeded(20);
    let outcome = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq AS seq \
         EXCEPT \
         MATCH (n:Item) WHERE n.seq > 3 RETURN n.seq AS seq",
        Some(2),
    );

    // 20 rows minus the 17 the right arm excludes = 3; capped to 2.
    assert_eq!(
        diagnostics(&outcome).total_rows,
        Some(3),
        "the right arm must be evaluated in full before the cap applies"
    );
    let mut kept = seqs(&outcome);
    kept.sort_unstable();
    assert_eq!(kept.len(), 2);
    assert!(
        kept.iter().all(|seq| (1..=3).contains(seq)),
        "kept rows must come from the correct difference, got {kept:?}"
    );
}

/// UNION ALL: the cap belongs to the combined result, and the total counts
/// both arms.
#[test]
fn union_all_counts_both_arms_before_capping() {
    let graph = seeded(10);
    let outcome = read_capped(
        &graph,
        "MATCH (n:Item) RETURN n.seq AS seq \
         UNION ALL \
         MATCH (n:Item) RETURN n.seq AS seq",
        Some(3),
    );

    assert_eq!(outcome.result.rows.len(), 3);
    assert_eq!(diagnostics(&outcome).total_rows, Some(20));
}

/// A `CALL {}` body is an input, not the result: its rows must not be capped
/// out from under the outer query.
#[test]
fn subquery_bodies_are_not_capped() {
    let graph = seeded(30);
    let outcome = read_capped(
        &graph,
        "CALL { MATCH (n:Item) RETURN n.seq AS seq } RETURN count(seq) AS c",
        Some(2),
    );

    assert_eq!(outcome.result.rows.len(), 1);
    assert_eq!(
        outcome.result.rows[0][0],
        Value::Int64(30),
        "the subquery must see all 30 rows even though the cap is 2"
    );
}

/// The lazy/streaming path returns rows through a descriptor rather than
/// `result.rows`; the cap must reach it, and the count must stay exact.
#[test]
fn lazy_path_is_capped_with_an_exact_total() {
    let graph = seeded(40);
    let params = empty_params();
    let mut opts = ExecuteOptions::eager(&params);
    opts.lazy_eligible = true;
    opts.row_limit = Some(6);
    let outcome = execute_read(&graph, "MATCH (n:Item) RETURN n.seq", &opts).expect("read");

    let retained = match outcome.result.lazy.as_ref() {
        Some(descriptor) => descriptor.len(),
        // The planner is free to decline lazy eligibility; either way the cap
        // must hold, so assert against whichever carrier holds the rows.
        None => outcome.result.rows.len(),
    };
    assert_eq!(retained, 6);
    let d = outcome
        .result
        .diagnostics
        .as_ref()
        .expect("diagnostics")
        .clone();
    assert_eq!(d.row_limit, Some(6));
    assert_eq!(d.total_rows, Some(40));
}

/// Truncation is announced through the ordinary query-warning channel, so
/// every binding's existing warning surface carries it with no new wiring.
/// The message must name both numbers a "showing X of Y" banner needs.
#[test]
fn truncation_raises_a_query_warning_naming_both_counts() {
    let graph = seeded(50);
    let outcome = read_capped(&graph, "MATCH (n:Item) RETURN n.seq", Some(5));

    let warning = diagnostics(&outcome)
        .warnings
        .iter()
        .find(|w| w.contains("row_limit"))
        .expect("a truncated result must warn");
    assert!(
        warning.contains('5') && warning.contains("50"),
        "the warning must carry the cap and the total: {warning}"
    );
}

/// A mutation's trailing RETURN is capped like any other result — but the
/// writes all happen, and `MutationStats` still counts every one of them.
#[test]
fn mutation_return_is_capped_while_every_write_still_lands() {
    let mut graph = seeded(12);
    let params = empty_params();
    let mut opts = ExecuteOptions::eager(&params);
    opts.row_limit = Some(3);
    let outcome = execute_mut(
        &mut graph,
        "MATCH (n:Item) SET n.touched = true RETURN n.seq",
        &opts,
    )
    .expect("mutation");

    assert!(outcome.is_mutation);
    assert_eq!(outcome.result.rows.len(), 3);
    let d = outcome
        .result
        .diagnostics
        .as_ref()
        .expect("diagnostics")
        .clone();
    assert_eq!(d.total_rows, Some(12));
    assert_eq!(
        outcome
            .result
            .stats
            .as_ref()
            .expect("mutation stats")
            .properties_set,
        12,
        "the cap reports fewer rows; it must never write fewer properties"
    );

    // ...and the graph really did take all twelve writes.
    let check = read_capped(
        &graph,
        "MATCH (n:Item) WHERE n.touched = true RETURN count(*) AS c",
        None,
    );
    assert_eq!(check.result.rows[0][0], Value::Int64(12));
}

/// `row_limit` and `max_work_units` are orthogonal: a generous cap does not
/// rescue a query from an exhausted work budget, and the budget's failure mode
/// stays an error rather than becoming a truncation.
#[test]
fn a_cap_does_not_convert_a_budget_overrun_into_a_truncation() {
    let graph = seeded(50);
    let params = empty_params();
    let mut opts = ExecuteOptions::eager(&params);
    opts.max_work_units = Some(2);
    opts.row_limit = Some(1);
    let error = match execute_read(&graph, "MATCH (n:Item) RETURN n.seq", &opts) {
        Err(error) => error,
        Ok(_) => panic!("the work budget must still fail the query"),
    };
    assert!(
        error.to_string().contains("max_work_units"),
        "unexpected error: {error}"
    );
}

/// EXPLAIN renders a plan, not data. Capping it would make a diagnostic tool
/// lie about the plan it is showing.
#[test]
fn explain_is_exempt_from_the_cap() {
    let graph = seeded(50);
    let uncapped = read_capped(&graph, "EXPLAIN MATCH (n:Item) RETURN n.seq", None);
    let capped = read_capped(&graph, "EXPLAIN MATCH (n:Item) RETURN n.seq", Some(1));

    assert!(capped.explain);
    assert_eq!(capped.result.rows.len(), uncapped.result.rows.len());
    assert_eq!(diagnostics(&capped).total_rows, None);
}
