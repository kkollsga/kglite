//! An unbound `$parameter` in a pattern's inline property map is an error.
//!
//! `MATCH (v:Vessel {flag: $flag})` with `$flag` unbound used to return zero
//! rows and no error, while the same predicate written `WHERE v.flag = $flag`
//! raised `Missing parameter: $flag`. Zero rows is the worst possible answer
//! here: the caller reads it as "the graph has no NO-flagged vessels" when the
//! graph has 91 of them. See `cypher::dynamic_labels` for why the check lives
//! in that pass and not in the matcher.
//!
//! The trap this module exists for is the **plan cache**, which is why these
//! tests are here and not only in the pass's own unit tests. `cacheable`
//! requires `opts.params.is_empty()`, and a query carrying `$flag` with no
//! params satisfies it — so if an entry for such a text could ever exist, the
//! *second* call would be served from the cache, skip the parse, skip this
//! pass, and silently revert to the old empty-result behaviour.

use std::collections::HashMap;

use super::execute::{execute_mut, execute_read, ExecuteOptions};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::languages::cypher::plan_cache;
use crate::graph::languages::cypher::plan_cache::instrumentation;

fn empty_params() -> HashMap<String, Value> {
    HashMap::new()
}

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// Three `NO`-flagged vessels and two `SE`-flagged ones — the eval's shape in
/// miniature, so a "matches nothing" regression is visible as a count.
fn fleet() -> DirGraph {
    let p = empty_params();
    let opts = ExecuteOptions::eager(&p);
    let mut graph = DirGraph::new();
    execute_mut(
        &mut graph,
        "CREATE (:Vessel {id: 1, flag: 'NO'}), (:Vessel {id: 2, flag: 'NO'}), \
         (:Vessel {id: 3, flag: 'NO'}), (:Vessel {id: 4, flag: 'SE'}), \
         (:Vessel {id: 5, flag: 'SE'})",
        &opts,
    )
    .expect("seed");
    execute_mut(
        &mut graph,
        "MATCH (a:Vessel {id: 1}), (b:Vessel {id: 4}) CREATE (a)-[:SISTER {since: 1990}]->(b)",
        &opts,
    )
    .expect("seed edge");
    graph
}

fn read_err(graph: &DirGraph, query: &str, p: &HashMap<String, Value>) -> String {
    let opts = ExecuteOptions::eager(p);
    execute_read(graph, query, &opts)
        .err()
        .unwrap_or_else(|| panic!("{query} must not succeed"))
        .to_string()
}

fn row_count(graph: &DirGraph, query: &str, p: &HashMap<String, Value>) -> i64 {
    let opts = ExecuteOptions::eager(p);
    let outcome = execute_read(graph, query, &opts).expect("read");
    match outcome.result.rows[0].first().expect("one column") {
        Value::Int64(n) => *n,
        other => panic!("expected a count, got {other:?}"),
    }
}

#[test]
fn an_unbound_inline_map_parameter_raises() {
    let graph = fleet();
    let err = read_err(
        &graph,
        "MATCH (v:Vessel {flag: $flag}) RETURN count(v) AS c",
        &empty_params(),
    );
    assert!(err.contains("Missing parameter: $flag"), "{err}");
}

/// The two spellings of one predicate now answer identically — that agreement
/// is the fix, not the error text on its own.
///
/// Both projections are asserted. The aggregate one is the case this pass did
/// *not* settle: the fused scan-aggregate path swallowed every non-regex
/// evaluation error from a `WHERE` (its `unwrap_or(false)` is what OPTIONAL
/// MATCH's unbound bindings rely on), so `WHERE v.flag = $flag RETURN
/// count(v)` answered zero while the inline map raised — the inline map was
/// briefly the *stricter* of the two spellings. The fused filters now
/// recognise the missing-parameter class alongside the regex-compile one, so
/// the four combinations agree; see `executor::helpers::is_user_input_error`.
#[test]
fn the_inline_map_and_where_spellings_report_the_same_thing() {
    let graph = fleet();
    for projection in ["RETURN v.id AS id", "RETURN count(v) AS c"] {
        let inline = read_err(
            &graph,
            &format!("MATCH (v:Vessel {{flag: $flag}}) {projection}"),
            &empty_params(),
        );
        let where_clause = read_err(
            &graph,
            &format!("MATCH (v:Vessel) WHERE v.flag = $flag {projection}"),
            &empty_params(),
        );
        assert_eq!(inline, where_clause, "{projection}");
    }
}

/// The golden the error must not have cost: a *bound* parameter still matches
/// exactly the rows it always did.
#[test]
fn a_bound_inline_map_parameter_still_matches() {
    let graph = fleet();
    let bound = params(&[("flag", Value::String("NO".into()))]);
    assert_eq!(
        row_count(
            &graph,
            "MATCH (v:Vessel {flag: $flag}) RETURN count(v) AS c",
            &bound
        ),
        3
    );
    assert_eq!(
        row_count(
            &graph,
            "MATCH (v:Vessel) WHERE v.flag = $flag RETURN count(v) AS c",
            &bound
        ),
        3
    );
    // Mixed: one parameter in the map, one in the WHERE.
    let both = params(&[
        ("flag", Value::String("NO".into())),
        ("floor", Value::Int64(2)),
    ]);
    assert_eq!(
        row_count(
            &graph,
            "MATCH (v:Vessel {flag: $flag}) WHERE v.id >= $floor RETURN count(v) AS c",
            &both
        ),
        2
    );
    // ...and the mixed shape still raises when only the map's is unbound.
    let err = read_err(
        &graph,
        "MATCH (v:Vessel {flag: $flag}) WHERE v.id >= $floor RETURN count(v) AS c",
        &params(&[("floor", Value::Int64(2))]),
    );
    assert!(err.contains("Missing parameter: $flag"), "{err}");
}

/// **The cache trap.** Running the same unbound text twice must raise twice.
/// The counters are the proof: the first call errors *before* `prepare`
/// reaches `plan_cache::insert`, so no entry is left behind for the second
/// call to be served from — asserted rather than assumed, since a passing
/// pair of errors would look identical if the cache were simply never
/// consulted.
#[test]
fn the_same_unbound_query_raises_on_every_call() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let graph = fleet();
    let query = "MATCH (v:Vessel {flag: $flag}) RETURN count(v) AS c";

    instrumentation::reset();
    let first = read_err(&graph, query, &empty_params());
    let second = read_err(&graph, query, &empty_params());
    // A `prepare` that errors never reaches `classify_pending`, so its events
    // stay pending until the *next* `begin_prepare` banks them as
    // `unclassified`. One more statement flushes the second call's.
    let _ = row_count(
        &graph,
        "MATCH (v:Vessel {flag: 'SE'}) RETURN count(v) AS c",
        &empty_params(),
    );
    let totals = instrumentation::totals();

    assert!(first.contains("Missing parameter: $flag"), "{first}");
    assert_eq!(first, second, "the second call must not go quiet");
    // Both erroring calls looked the cache up and both missed; neither
    // inserted.
    let stats = totals.unclassified;
    assert_eq!(
        stats.lookups, 2,
        "both calls must consult the cache, else this proves nothing: {stats:?}"
    );
    assert_eq!(stats.hits, 0, "{stats:?}");
    assert_eq!(
        stats.insertions, 0,
        "an erroring statement must leave nothing cached: {stats:?}"
    );
}

/// The other half of the cache argument: a *clean* query still caches, so the
/// pass costs the hot path nothing. If this went to zero hits the fix would
/// have moved the check in front of the lookup by accident.
#[test]
fn a_clean_query_still_hits_the_plan_cache() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let graph = fleet();
    let query = "MATCH (v:Vessel {flag: 'NO'}) RETURN count(v) AS c";

    instrumentation::reset();
    assert_eq!(row_count(&graph, query, &empty_params()), 3);
    assert_eq!(row_count(&graph, query, &empty_params()), 3);
    let stats = instrumentation::totals().read;
    assert_eq!(stats.hits, 1, "{stats:?}");
}

/// Every read-pattern position the check covers, exercised end-to-end rather
/// than against the pass in isolation.
#[test]
fn every_read_pattern_position_raises_end_to_end() {
    let graph = fleet();
    for query in [
        "MATCH (v:Vessel {flag: $flag}) RETURN v",
        "MATCH (v:Vessel) OPTIONAL MATCH (w:Vessel {flag: $flag}) RETURN v, w",
        "MATCH (a:Vessel)-[:SISTER {since: $flag}]->(b) RETURN a",
        "MATCH (a:Vessel) WHERE EXISTS { MATCH (a)-[:SISTER]->(:Vessel {flag: $flag}) } RETURN a",
        "MATCH (a:Vessel) RETURN COUNT { (a)-[:SISTER]->(:Vessel {flag: $flag}) } AS n",
        "CALL { MATCH (v:Vessel {flag: $flag}) RETURN v } RETURN v",
        "MATCH (v:Vessel {flag: $flag}) RETURN v UNION MATCH (v:Vessel) RETURN v",
    ] {
        let err = read_err(&graph, query, &empty_params());
        assert!(err.contains("Missing parameter: $flag"), "{query}: {err}");
    }
}

/// Write patterns reach the same message by the evaluator's route, so the
/// contract "an unbound parameter always raises" holds across all four
/// clauses the eval named.
#[test]
fn write_clauses_raise_through_the_evaluator() {
    let p = empty_params();
    let opts = ExecuteOptions::eager(&p);
    for query in [
        "CREATE (v:Vessel {flag: $flag})",
        "MERGE (v:Vessel {flag: $flag})",
    ] {
        let mut graph = fleet();
        let err = execute_mut(&mut graph, query, &opts)
            .err()
            .unwrap_or_else(|| panic!("{query} must not succeed"))
            .to_string();
        assert!(err.contains("Missing parameter: $flag"), "{query}: {err}");
    }
}

/// A pattern that is unbound in both positions reports the label, because the
/// two checks share one walk and the label bind runs first. Pinned so the
/// message an agent sees is a property of the code, not of iteration order.
#[test]
fn an_unbound_label_wins_over_an_unbound_map_parameter() {
    let graph = fleet();
    let err = read_err(
        &graph,
        "MATCH (v:$label {flag: $flag}) RETURN v",
        &empty_params(),
    );
    assert!(
        err.contains("$label") && err.contains("label or relationship type"),
        "{err}"
    );
}
