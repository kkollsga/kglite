//! Coverage for [`PatternExecutor::set_match_ceiling`] — the absolute bound a
//! materializing caller puts on the matcher's in-flight buffers.
//!
//! The ceiling exists because the Cypher executor's row backstop only ever saw
//! the *finished* match vector: a variable-length expansion could hold
//! gigabytes of `PatternMatch` before a single check ran, and a deep enough one
//! never reached the check at all. So the tests below assert on the producer,
//! at a ceiling small enough to reach in a unit test, and each one names the
//! buffer it is bounding:
//!
//! * the per-source trail buffer inside `expand_var_length`,
//! * the per-hop `new_matches` buffer of the sequential expansion,
//! * the cross-worker total of the parallel expansion.
//!
//! Each has a matching "no ceiling" case, because a ceiling that fires
//! unconditionally would pass every assertion above while breaking every query.

use super::*;
use crate::graph::core::pattern_matching::parser::parse_pattern;
use crate::graph::languages::cypher::executor::budget::{
    ExecutionBudget, MatchCeiling, MAX_UNBOUNDED_ROWS,
};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

/// Six `P` nodes, every pair joined by an `R` edge. Small, but every extra hop
/// multiplies the trail count, so a variable-length pattern over it produces
/// thousands of matches from a handful of nodes.
fn clique() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "UNWIND range(1, 6) AS i CREATE (:P {id: i, name: 'n' + toString(i)})",
    );
    run(
        &mut graph,
        "MATCH (a:P), (b:P) WHERE a.id < b.id CREATE (a)-[:R]->(b)",
    );
    graph
}

/// `matches.len()` for `pattern`, run under `ceiling`.
fn execute(
    graph: &DirGraph,
    pattern: &str,
    ceiling: Option<MatchCeiling>,
) -> Result<usize, String> {
    let params = HashMap::new();
    let parsed = parse_pattern(pattern).expect("pattern parses");
    PatternExecutor::new_lightweight_with_params(graph, None, &params)
        .set_match_ceiling(ceiling)
        .execute(&parsed)
        .map(|matches| matches.len())
}

fn ceiling(max: usize) -> Option<MatchCeiling> {
    Some(MatchCeiling::new(max, "MATCH expansion"))
}

/// A ceiling breach must read like the row backstop's own message: the count
/// reached, the ceiling, the operator, and both ways out.
fn assert_quantified(err: &str, held_at_least: usize, max: usize) {
    assert!(err.contains("MATCH expansion"), "{err}");
    assert!(err.contains(&max.to_string()), "{err}");
    assert!(err.contains("max_rows"), "{err}");
    assert!(err.contains("LIMIT"), "{err}");
    let count: usize = err
        .split_whitespace()
        .nth(2)
        .and_then(|word| word.parse().ok())
        .unwrap_or_else(|| panic!("no count in message: {err}"));
    assert!(
        count > max && count >= held_at_least,
        "reported {count}, ceiling {max}: {err}"
    );
}

#[test]
fn trail_expansion_stops_at_the_ceiling() {
    let graph = clique();
    // `*1..5` over K6 is thousands of trails from six start nodes — the shape
    // whose only previous bound was the number of trails the graph admits.
    let uncapped = execute(&graph, "(a:P)-[:R*1..5]-(b:P)", None).expect("no ceiling, no error");
    assert!(uncapped > 2000, "fixture is too small to bound: {uncapped}");

    let err = execute(&graph, "(a:P)-[:R*1..5]-(b:P)", ceiling(64))
        .expect_err("the ceiling must stop the trail expansion");
    assert_quantified(&err, 65, 64);
}

#[test]
fn fixed_hop_expansion_stops_at_the_ceiling() {
    let graph = clique();
    let uncapped = execute(&graph, "(a:P)-[:R]-(b:P)-[:R]-(c:P)", None).expect("no ceiling");
    assert!(uncapped > 100, "fixture too small: {uncapped}");

    let err = execute(&graph, "(a:P)-[:R]-(b:P)-[:R]-(c:P)", ceiling(16))
        .expect_err("the ceiling must stop the hop expansion");
    assert_quantified(&err, 17, 16);
}

#[test]
fn a_ceiling_above_the_match_count_changes_nothing() {
    let graph = clique();
    for pattern in ["(a:P)-[:R*1..5]-(b:P)", "(a:P)-[:R]-(b:P)-[:R]-(c:P)"] {
        let uncapped = execute(&graph, pattern, None).expect("no ceiling");
        let capped = execute(&graph, pattern, ceiling(MAX_UNBOUNDED_ROWS))
            .unwrap_or_else(|e| panic!("{pattern}: ceiling fired below its bound: {e}"));
        assert_eq!(uncapped, capped, "{pattern}");
    }
}

#[test]
fn the_node_scan_is_exempt_from_the_ceiling() {
    // A node-only pattern is bounded by the graph's node count, which the
    // caller already holds — the same reason `Charge::Work` is exempt. It must
    // answer even when the ceiling is below the node count.
    let graph = clique();
    assert_eq!(execute(&graph, "(a:P)", ceiling(2)), Ok(6));
}

/// 9000 `P` nodes, each joined to all ten `Q` nodes: enough seeds to clear
/// `EXPANSION_RAYON_THRESHOLD` (so the hop fans out across rayon workers), and
/// enough matches per seed that one worker accumulates a publish block.
fn wide_parallel_fanout() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "UNWIND range(1, 9000) AS i CREATE (:P {id: i, name: 'p'})",
    );
    run(
        &mut graph,
        "UNWIND range(1, 40) AS i CREATE (:Q {id: i, name: 'q'})",
    );
    run(&mut graph, "MATCH (p:P), (q:Q) CREATE (p)-[:R]->(q)");
    const { assert!(EXPANSION_RAYON_THRESHOLD < 9000) };
    graph
}

#[test]
fn the_parallel_expansion_stops_at_the_ceiling() {
    // Above `EXPANSION_RAYON_THRESHOLD` seeds with no `max_matches`, the hop
    // runs across rayon workers and no single worker can see the total — the
    // one buffer the sequential `new_matches.len()` check cannot reach.
    let graph = wide_parallel_fanout();
    assert_eq!(execute(&graph, "(a:P)-[:R]->(b:Q)", None), Ok(360_000));

    let err = execute(&graph, "(a:P)-[:R]->(b:Q)", ceiling(100))
        .expect_err("the ceiling must stop the parallel expansion");
    assert_quantified(&err, 101, 100);
}

#[test]
fn the_parallel_expansion_reports_the_ceiling_before_it_finishes() {
    // The point of publishing per block rather than only after `collect` is
    // that the region stops while it is still filling. A ceiling above one
    // publish stride but below the full fan-out separates the two: an
    // in-flight breach names a count far short of the 90 000 the completed
    // buffer would have reported.
    let graph = wide_parallel_fanout();
    let err = execute(&graph, "(a:P)-[:R]->(b:Q)", ceiling(20_000))
        .expect_err("the ceiling must stop the parallel expansion");
    assert_quantified(&err, 20_001, 20_000);
    assert!(
        !err.contains("360000"),
        "the breach was only detected after the whole buffer existed: {err}"
    );
}

#[test]
fn only_an_unbounded_budget_hands_down_a_ceiling() {
    // An explicit `max_rows` already bounds the producer through
    // `budget_probe_limit`; re-imposing it here would reject the intermediate
    // hops' deliberate overcommit.
    let unbounded = ExecutionBudget::new(None)
        .match_ceiling("MATCH expansion")
        .expect("the default path must carry the backstop");
    assert!(unbounded.check(MAX_UNBOUNDED_ROWS).is_ok());
    assert!(unbounded.check(MAX_UNBOUNDED_ROWS + 1).is_err());

    assert!(ExecutionBudget::new(Some(10))
        .match_ceiling("MATCH expansion")
        .is_none());
}
