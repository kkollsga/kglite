//! Interrupt coverage for the executor's **sequential** MATCH row loops.
//!
//! `match_execution.rs` polled nothing at all: the pattern matcher stops on a
//! deadline, and every row loop downstream of it — the match-to-row
//! conversion, the comma-pattern join, the subsequent-MATCH driving join —
//! then ran to completion no matter how long ago the deadline passed. A
//! downstream 3-hop query with 1.9M intermediate rows reported this as a 30 s
//! deadline that ran past 120 s and took 7.29 GB before the process was
//! OOM-killed.
//!
//! The hook tests below are timing-free: `interrupt_after_periodic_polls` is a
//! thread-local that fires on the Nth poll, so each one asserts that a named
//! loop *reaches* a poll, and is red on the unfixed tree for the only possible
//! reason — the loop never polled. The wall-clock test then asserts the thing
//! the downstream report is actually about: that the abort happens early in
//! the loop rather than after it.

use super::*;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup failed: {query}: {e}"));
}

/// `hubs` `B` nodes, each the target of `fan` `A`s and `fan` `C`s — so the
/// three-element pattern below has `hubs * fan * fan` matches while the graph
/// itself stays at `hubs * fan * 2` nodes and as many edges.
fn fan_in_graph(hubs: i64, fan: i64) -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        &format!("UNWIND range(1, {hubs}) AS b CREATE (:B {{bid: b}})"),
    );
    run(
        &mut graph,
        &format!("MATCH (x:B) UNWIND range(1, {fan}) AS i CREATE (a:A {{aid: i}})-[:R1]->(x)"),
    );
    run(
        &mut graph,
        &format!("MATCH (x:B) UNWIND range(1, {fan}) AS i CREATE (c:C {{cid: i}})-[:R2]->(x)"),
    );
    graph
}

fn read(graph: &DirGraph, query: &str) -> Result<usize, String> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_read(graph, query, &opts)
        .map(|outcome| outcome.result.rows.len())
        .map_err(|e| e.to_string())
}

/// Every hook test needs the same non-vacuity meter: the identical query with
/// no hook armed must succeed, or the `Err` above proves nothing.
fn assert_hook_fired_and_query_is_otherwise_fine(graph: &DirGraph, query: &str, err: String) {
    assert!(
        err.contains("test hook"),
        "expected the periodic-poll hook to fire, got: {err}"
    );
    assert!(
        read(graph, query).is_ok(),
        "non-vacuity: {query} must succeed with no hook armed"
    );
}

/// The match-to-row conversion in `first_pattern_rows` — the loop the
/// downstream OOM was spending its time in.
#[test]
fn first_match_row_loop_polls_the_interrupt() {
    let graph = fan_in_graph(2, 4);
    let query = "MATCH (n0:A)-[r1:R1]->(n1:B)<-[r2:R2]-(n2:C) RETURN n0.aid";

    CypherExecutor::interrupt_after_periodic_polls(0);
    let err = read(&graph, query).expect_err("the row loop must reach a poll");
    assert_hook_fired_and_query_is_otherwise_fine(&graph, query, err);
}

/// The comma-pattern join inside one MATCH clause. One poll is spent by the
/// first pattern's row loop above (the graph is far under the poll interval,
/// so it polls exactly once), leaving the next poll to the join.
#[test]
fn comma_pattern_join_polls_the_interrupt() {
    let graph = fan_in_graph(2, 4);
    let query = "MATCH (a:A), (b:B) RETURN a.aid, b.bid";

    CypherExecutor::interrupt_after_periodic_polls(1);
    let err = read(&graph, query).expect_err("the comma-pattern join must reach a poll");
    assert_hook_fired_and_query_is_otherwise_fine(&graph, query, err);
}

/// The subsequent-MATCH join: every row of the incoming set drives its own
/// expansion.
#[test]
fn subsequent_match_join_polls_the_interrupt() {
    let graph = fan_in_graph(2, 4);
    let query = "MATCH (a:A) WITH a MATCH (a)-[:R1]->(b:B) RETURN b.bid";

    CypherExecutor::interrupt_after_periodic_polls(1);
    let err = read(&graph, query).expect_err("the driving-row join must reach a poll");
    assert_hook_fired_and_query_is_otherwise_fine(&graph, query, err);
}

/// The report itself, in miniature: a deadline set a quarter of the way into a
/// query whose time is dominated by the row loop must abort near the deadline,
/// not after the loop finishes.
///
/// Self-calibrating rather than absolute — the budget is a fraction of this
/// machine's own uncapped runtime, so a slow machine moves both numbers
/// together. The unfixed tree fails it by running the whole loop out
/// (measured: a 500 ms deadline detected 1.52 s late on a 2.1 s query).
#[test]
fn a_deadline_inside_the_row_loop_aborts_without_finishing_it() {
    let graph = fan_in_graph(60, 100);
    // The WHERE is fused into the MATCH, so its per-row evaluation happens
    // inside the row loop — which is what makes that loop, rather than the
    // matcher, the dominant cost.
    let query = "MATCH (n0:A)-[r1:R1]->(n1:B)<-[r2:R2]-(n2:C) \
                 WHERE n0.aid + n1.bid + n2.cid > -1 AND toString(n0.aid) <> 'zzz' \
                 RETURN n0.aid";
    let params = HashMap::new();

    let started = std::time::Instant::now();
    let rows = read(&graph, query).expect("uncapped run succeeds");
    let uncapped = started.elapsed();
    assert_eq!(
        rows, 600_000,
        "the fixture must produce the row count it sizes for"
    );

    let mut opts = ExecuteOptions::eager(&params);
    opts.deadline = Some(std::time::Instant::now() + uncapped / 4);
    let started = std::time::Instant::now();
    let err = match execute_read(&graph, query, &opts) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the deadline must fire"),
    };
    let elapsed = started.elapsed();

    assert!(err.contains("timed out"), "unexpected error: {err}");
    assert!(
        elapsed < uncapped / 2,
        "deadline at {:?} of a {uncapped:?} query aborted only after {elapsed:?} — \
         the row loop ran past it",
        uncapped / 4
    );
}
