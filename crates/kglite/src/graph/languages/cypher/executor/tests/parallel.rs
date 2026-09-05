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

/// Exactly the projection fan-out gate, so these tests keep straddling it if
/// the measured constant moves.
fn gate_rows() -> usize {
    crate::graph::parallel::PROJECTION_MIN_ROWS
}

/// `count` rows all bound to the same `Person`, enough to take (or miss) the
/// parallel branch depending on `count` vs the projection gate.
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
        .execute_return_projection(clause, bound_rows(&graph, gate_rows()))
        .unwrap_err();
    assert_eq!(cancelled, "Query cancelled");

    // Non-vacuity meter: identical call, one row under the threshold, takes
    // the sequential branch — which does not poll. If this ever starts
    // failing, the assertion above stopped proving anything about the
    // parallel branch and this file needs a new probe.
    assert!(
        executor
            .execute_return_projection(clause, bound_rows(&graph, gate_rows() - 1))
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
        .execute_return_projection(clause, bound_rows(&graph, gate_rows()))
        .unwrap_err();
    assert!(
        timed_out.starts_with("Query timed out."),
        "unexpected error: {timed_out}"
    );

    assert!(
        executor
            .execute_return_projection(clause, bound_rows(&graph, gate_rows() - 1))
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
        .execute_return_with_windows(clause, bound_rows(&graph, gate_rows()))
        .unwrap_err();
    assert_eq!(cancelled, "Query cancelled");

    assert!(
        executor
            .execute_return_with_windows(clause, bound_rows(&graph, gate_rows() - 1))
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
        .finalize_result(projected_rows(gate_rows()))
        .unwrap_err();
    assert_eq!(cancelled, "Query cancelled");

    assert!(
        executor
            .finalize_result(projected_rows(gate_rows() - 1))
            .is_ok(),
        "below-threshold materialisation must still take the sequential branch"
    );
}

// ── Parallel fused node-scan aggregate (Q2) ─────────────────────────────────

use crate::graph::parallel::{
    parallel_scans, PARALLEL_MIN_ROWS_COMPILED, PARALLEL_MIN_ROWS_INTERPRETED,
};

/// [`parallel_scans`] is a process-global counter and `cargo test` runs these
/// in parallel threads, so a bare before/after read is a race — one test's
/// fan-out lands inside another's window. Every test that *reads* the meter
/// holds this, and so does every test that could move it.
static METER: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn meter_guard() -> std::sync::MutexGuard<'static, ()> {
    METER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `n` `Item` nodes with a numeric `value` and a low-cardinality `cat`.
/// Built through the storage API rather than `CREATE`, so a 200k-node fixture
/// costs a fraction of a second in a debug build.
fn scan_graph(n: usize) -> DirGraph {
    let mut graph = DirGraph::new();
    for i in 0..n {
        let node = NodeData::new(
            Value::UniqueId(i as u32),
            Value::String(format!("Item_{i}")),
            "Item".to_string(),
            HashMap::from([
                ("value".to_string(), Value::Int64((i % 1000) as i64)),
                ("cat".to_string(), Value::String(format!("cat_{}", i % 7))),
            ]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Item".to_string())
            .push(idx);
    }
    graph
}

fn run(graph: &DirGraph, query: &str, parallel: bool) -> CypherResult {
    let params: HashMap<String, Value> = HashMap::new();
    let mut parsed = parser::parse_cypher(query).expect("query parses");
    crate::graph::languages::cypher::planner::optimize(&mut parsed, graph, &params);
    CypherExecutor::with_params(graph, &params, None)
        .with_parallel(parallel)
        .execute(&parsed)
        .expect("query executes")
}

/// Every aggregate this operator serves must merge to the identical answer,
/// and the grouped shapes must emit groups in the identical order — the
/// partitioned scan folds partials in candidate order precisely so that
/// first-seen order survives.
#[test]
fn parallel_scan_aggregate_matches_serial() {
    let _meter = meter_guard();
    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED + 137);
    // `toUpper(...)` does not compile to a column route, which puts these on
    // the interpreted side of the runtime gate — the lower row threshold, so
    // the fixture stays small enough for a debug test.
    let queries = [
        "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(*) AS n",
        "MATCH (n:Item) RETURN toUpper(n.cat) AS c, sum(n.value) AS s, avg(n.value) AS a",
        "MATCH (n:Item) RETURN toUpper(n.cat) AS c, min(n.value) AS lo, max(n.value) AS hi",
        "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(DISTINCT n.value) AS d",
        "MATCH (n:Item) WHERE n.value > 500 RETURN toUpper(n.cat) AS c, count(*) AS n",
        "MATCH (n:Item) RETURN count(*) AS n, toUpper(n.cat) AS c",
    ];
    let before = parallel_scans();
    for query in queries {
        let serial = run(&graph, query, false);
        let parallel = run(&graph, query, true);
        assert_eq!(serial.columns, parallel.columns, "columns differ: {query}");
        assert_eq!(
            serial.rows, parallel.rows,
            "parallel diverged from serial (values or group order): {query}"
        );
        assert!(!serial.rows.is_empty(), "vacuous fixture for {query}");
    }
    // Non-vacuity: the comparison above is worthless if nothing fanned out.
    assert!(
        parallel_scans() > before,
        "no query fanned out — the equality assertions compared two serial runs"
    );
}

/// The gate, not the flag, decides. A query that opts in but sits below the
/// row threshold must stay on the sequential scan.
#[test]
fn scan_aggregate_stays_serial_below_the_gate() {
    let _meter = meter_guard();
    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED - 1);
    let query = "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(*) AS n";
    let before = parallel_scans();
    let opted_in = run(&graph, query, true);
    assert_eq!(
        parallel_scans(),
        before,
        "a below-gate query fanned out — the runtime gate is not being consulted"
    );
    assert_eq!(opted_in.rows, run(&graph, query, false).rows);
}

/// Opting out must never fan out, however many rows there are.
#[test]
fn scan_aggregate_stays_serial_without_opt_in() {
    let _meter = meter_guard();
    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED + 137);
    let query = "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(*) AS n";
    let before = parallel_scans();
    run(&graph, query, false);
    assert_eq!(
        parallel_scans(),
        before,
        "parallel=false fanned out — the opt-in is not being honoured"
    );
}

/// A flag that is already set when the query starts is caught by candidate
/// discovery, *before* the scan loop — so this test says nothing about the
/// scan loop itself. It is here to pin that opting in to the parallel runtime
/// does not lose the cancellation the serial path already honoured.
/// [`parallel_scan_aggregate_is_interruptible_mid_scan`] is the one that
/// exercises the scan loop's own poll.
#[test]
fn parallel_scan_aggregate_honours_a_pre_tripped_flag() {
    let _meter = meter_guard();
    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED + 137);
    let params: HashMap<String, Value> = HashMap::new();
    let query = "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(*) AS n";
    let mut parsed = parser::parse_cypher(query).expect("query parses");
    crate::graph::languages::cypher::planner::optimize(&mut parsed, &graph, &params);

    let cancelled = CypherExecutor::with_params(&graph, &params, None)
        .with_parallel(true)
        .with_cancel(Some(&CANCELLED))
        .execute(&parsed)
        .unwrap_err();
    assert_eq!(cancelled, "Query cancelled");

    let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let timed_out = CypherExecutor::with_params(&graph, &params, Some(past))
        .with_parallel(true)
        .execute(&parsed)
        .unwrap_err();
    assert!(
        timed_out.starts_with("Query timed out."),
        "unexpected error: {timed_out}"
    );
}

/// The scan loop's own interrupt poll.
///
/// Cancellation has to arrive *during* the scan: candidate discovery polls the
/// same flag, so a pre-tripped one never reaches the loop. Rather than sleep
/// for a guessed fraction of the run — which is a flake waiting for a busy
/// machine — the flipper spins on the `PARALLEL_SCANS` meter, which the
/// executor bumps immediately before it fans out and therefore strictly
/// *after* candidate discovery has returned. The flag is set with the whole
/// scan still ahead, and every partition polls at its first row.
///
/// Red-first: with `ParallelInterrupt::check` removed from `scan_partition`
/// this returns rows instead of `Err`.
#[test]
fn parallel_scan_aggregate_is_interruptible_mid_scan() {
    let _meter = meter_guard();
    static MID_SCAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    MID_SCAN.store(false, std::sync::atomic::Ordering::Relaxed);

    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED * 10);
    let params: HashMap<String, Value> = HashMap::new();
    let query = "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(*) AS n";
    let mut parsed = parser::parse_cypher(query).expect("query parses");
    crate::graph::languages::cypher::planner::optimize(&mut parsed, &graph, &params);

    let before = parallel_scans();
    let flipper = std::thread::spawn(move || {
        let give_up = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while parallel_scans() == before && std::time::Instant::now() < give_up {
            std::hint::spin_loop();
        }
        MID_SCAN.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let outcome = CypherExecutor::with_params(&graph, &params, None)
        .with_parallel(true)
        .with_cancel(Some(&MID_SCAN))
        .execute(&parsed);
    flipper.join().expect("flipper thread");

    assert!(
        parallel_scans() > before,
        "the query never fanned out — this test is not measuring the parallel scan"
    );
    assert_eq!(
        outcome.err(),
        Some("Query cancelled".to_string()),
        "the parallel scan ran to completion through a cancellation raised mid-scan"
    );
}

/// The servers get their `ExecuteOptions` from `eager(..)` and never touch
/// this field (verified by grep at the time of writing), so the *default* is
/// what keeps Bolt and MCP sequential. Pin it: a default flipped to `true`
/// would silently hand every connected client's query the whole machine.
#[test]
fn parallel_is_off_in_the_default_execute_options() {
    let params: HashMap<String, Value> = HashMap::new();
    assert!(!crate::graph::session::ExecuteOptions::eager(&params).parallel);
    assert!(!crate::graph::session::ExecuteOptions::new(&params).parallel);
    // And the executor's own default matches, for callers that build one
    // directly rather than going through a session.
    let graph = build_test_graph();
    assert!(!CypherExecutor::with_params(&graph, &params, None).parallel);
}

// ── Parallel candidate scan + filter (Q3) ───────────────────────────────────

use crate::graph::parallel::parallel_candidate_scans;

/// The same `Item` shape as [`scan_graph`], but bulk-loaded so the properties
/// land in a `ColumnStore` — which is what lets `ColumnFilter::compile` accept
/// the matcher and the scan take the column route. `scan_graph` adds nodes one
/// at a time and stays on row storage.
/// `n` `Item` nodes plus one `LINKS` edge per node, so a two-element pattern
/// produces `n` rows — enough to put the *materialized* grouping path (rather
/// than the fused node-scan operator) above its row gate.
fn linked_graph(n: usize) -> DirGraph {
    let mut graph = scan_graph(n);
    let indices: Vec<petgraph::graph::NodeIndex> =
        graph.type_indices.get("Item").expect("Item index").to_vec();
    for (i, &src) in indices.iter().enumerate() {
        let dst = indices[(i * 7 + 13) % indices.len()];
        let edge = crate::graph::schema::EdgeData::new(
            "LINKS".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        graph.graph.add_edge(src, dst, edge);
    }
    graph.register_connection_type("LINKS".to_string());
    graph
}

fn columnar_scan_graph(n: usize) -> DirGraph {
    use crate::datatypes::DataFrame;
    let mut graph = DirGraph::new();
    let columns: Vec<String> = ["nid", "name", "value", "cat"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows: Vec<Vec<Value>> = (0..n)
        .map(|i| {
            vec![
                Value::Int64(i as i64),
                Value::String(format!("Item_{i}")),
                Value::Int64((i % 1000) as i64),
                Value::String(format!("cat_{}", i % 7)),
            ]
        })
        .collect();
    let df = DataFrame::from_cypher_rows(columns, rows).unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        df,
        "Item".to_string(),
        "nid".to_string(),
        Some("name".to_string()),
        None,
    )
    .unwrap();
    graph
}

/// Scan+filter shapes. Every one routes through `filter_node_candidates`: an
/// inline property map on the node pattern is what a `WHERE` on a scanned
/// property is rewritten to.
const SCAN_FILTER_QUERIES: &[&str] = &[
    "MATCH (n:Item {cat: 'cat_3'}) RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.cat = 'cat_3' RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.value > 500 RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.name STARTS WITH 'Item_1' RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.name CONTAINS '99' RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.cat IN ['cat_1', 'cat_5'] RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.value >= 100 AND n.value < 200 RETURN n.name AS nm",
];

/// **Row order is the assertion, not row content.** Bucket order of an
/// un-`ORDER BY`'d MATCH is a documented, test-gated invariant, and a
/// partitioned scan is exactly the change that could break it while every
/// set-comparison stayed green. These compare `Vec`s, in order.
#[test]
fn parallel_candidate_scan_matches_serial_in_order() {
    let _meter = meter_guard();
    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED + 137);
    let before = parallel_candidate_scans();
    for query in SCAN_FILTER_QUERIES {
        let serial = run(&graph, query, false);
        let parallel = run(&graph, query, true);
        assert_eq!(serial.columns, parallel.columns, "columns differ: {query}");
        assert_eq!(
            serial.rows, parallel.rows,
            "parallel candidate scan diverged from serial (values or ORDER): {query}"
        );
        assert!(!serial.rows.is_empty(), "vacuous fixture for {query}");
    }
    assert!(
        parallel_candidate_scans() > before,
        "no query fanned out its candidate scan — the order assertions above \
         compared two serial runs"
    );
}

/// Secondary labels union a second candidate bucket into the scan, and the
/// union's order is part of the same invariant.
#[test]
fn parallel_candidate_scan_preserves_multi_label_order() {
    let _meter = meter_guard();
    let mut graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED + 137);
    // Give every seventh node a secondary label, so the scan walks the
    // primary type index and then the secondary bucket.
    let tagged: Vec<petgraph::graph::NodeIndex> = graph
        .type_indices
        .get("Item")
        .expect("Item index")
        .to_vec()
        .into_iter()
        .step_by(7)
        .collect();
    graph.secondary_label_index.insert(
        crate::graph::schema::InternedKey::from_str("Tagged"),
        tagged,
    );
    graph.has_secondary_labels = true;

    let before = parallel_candidate_scans();
    for query in [
        "MATCH (n:Item:Tagged) RETURN n.name AS nm",
        "MATCH (n:Item) WHERE n.value > 100 RETURN n.name AS nm",
    ] {
        assert_eq!(
            run(&graph, query, false).rows,
            run(&graph, query, true).rows,
            "multi-label scan order diverged: {query}"
        );
    }
    assert!(parallel_candidate_scans() > before, "nothing fanned out");
}

/// The gate, not the flag, decides — and opting out never fans out.
#[test]
fn candidate_scan_respects_the_gate_and_the_opt_in() {
    let _meter = meter_guard();
    let query = SCAN_FILTER_QUERIES[0];

    let small = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED - 1);
    let before = parallel_candidate_scans();
    run(&small, query, true);
    assert_eq!(
        parallel_candidate_scans(),
        before,
        "a below-gate candidate scan fanned out"
    );

    let large = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED + 137);
    let before = parallel_candidate_scans();
    run(&large, query, false);
    assert_eq!(
        parallel_candidate_scans(),
        before,
        "parallel=false fanned out its candidate scan"
    );
}

/// The compiled-filter meter must survive the fan-out.
///
/// `ROWS_FILTERED` is thread-local and the scan now runs on rayon workers, so
/// without the delta fold in `filter_candidates_parallel` this reads zero and
/// every differential that depends on it compares the row route with itself.
#[test]
fn compiled_filter_meter_sees_rows_answered_on_workers() {
    let _meter = meter_guard();
    // Columnar storage means the matcher compiles, which puts this scan on
    // the *compiled* side of the gate — the higher row threshold.
    let graph = columnar_scan_graph(PARALLEL_MIN_ROWS_COMPILED + 137);
    let query = "MATCH (n:Item {cat: 'cat_3'}) RETURN n.name AS nm";

    // Serial first: establishes that this query reaches a compiled filter at
    // all, so a zero reading from the parallel run means the fold is broken
    // rather than that the shape never compiles.
    crate::graph::core::pattern_matching::column_filter::reset_rows_filtered();
    let serial = run(&graph, query, false);
    let serial_rows = crate::graph::core::pattern_matching::column_filter::rows_filtered();
    assert!(
        serial_rows > 0,
        "the serial run never reached a compiled filter — pick another shape"
    );

    let before = parallel_candidate_scans();
    crate::graph::core::pattern_matching::column_filter::reset_rows_filtered();
    let parallel = run(&graph, query, true);
    let parallel_rows = crate::graph::core::pattern_matching::column_filter::rows_filtered();
    assert!(
        parallel_candidate_scans() > before,
        "the query did not fan out — this test is not measuring the worker fold"
    );
    assert_eq!(
        parallel_rows, serial_rows,
        "the compiled-filter meter lost rows answered on worker threads"
    );
    assert_eq!(serial.rows, parallel.rows);
}

/// The candidate scan's own interrupt poll.
///
/// Driven through `PatternExecutor::find_matching_nodes_pub` rather than a
/// whole Cypher query, and deliberately so: routed through `execute`, a
/// cancellation raised during the scan is *also* seen by the clause pipeline
/// afterwards, so the test passed with the scan's poll deleted — it was
/// measuring a later checkpoint. Calling the scan directly leaves its own poll
/// as the only thing that can return this error.
///
/// The flipper spins on the candidate-scan meter, bumped immediately before
/// the fan-out, so the flag is set with the whole scan still ahead.
///
/// Red-first: with `ParallelInterrupt::check` removed from
/// `filter_candidate_partition` this returns nodes instead of `Err`.
#[test]
fn parallel_candidate_scan_is_interruptible_mid_scan() {
    use crate::graph::core::pattern_matching::{NodePattern, PatternExecutor, PropertyMatcher};

    let _meter = meter_guard();
    static MID_SCAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    MID_SCAN.store(false, std::sync::atomic::Ordering::Relaxed);

    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED * 10);
    let pattern = NodePattern {
        variable: Some("n".to_string()),
        node_type: Some("Item".to_string()),
        extra_labels: Vec::new(),
        alt_labels: None,
        properties: Some(HashMap::from([(
            "cat".to_string(),
            PropertyMatcher::Equals(Value::String("cat_3".to_string())),
        )])),
        label_params: Vec::new(),
    };

    let before = parallel_candidate_scans();
    let flipper = std::thread::spawn(move || {
        let give_up = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while parallel_candidate_scans() == before && std::time::Instant::now() < give_up {
            std::hint::spin_loop();
        }
        MID_SCAN.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let outcome = PatternExecutor::new(&graph, None)
        .set_parallel(true)
        .set_cancel(Some(&MID_SCAN))
        .find_matching_nodes_pub(&pattern);
    flipper.join().expect("flipper thread");

    assert!(
        parallel_candidate_scans() > before,
        "the scan never fanned out"
    );
    assert_eq!(
        outcome.err(),
        Some("Query cancelled".to_string()),
        "the parallel candidate scan ran to completion through a mid-scan cancellation"
    );
}

// ── Parallel grouped aggregation (Q4) ───────────────────────────────────────

use crate::graph::parallel::parallel_aggregations;

/// Aggregation shapes routed through the *materialized* grouping path (a
/// multi-element pattern keeps them off the fused node-scan operator).
const AGG_QUERIES: &[&str] = &[
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, count(*) AS n",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, sum(a.value) AS s, avg(a.value) AS av",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, min(a.value) AS lo, max(a.value) AS hi",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, count(DISTINCT a.value) AS d",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, collect(a.value) AS vals",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, collect(DISTINCT a.value) AS vals",
    // Evaluated group key: one surrogate group per *value*, thousands of rows
    // each, so the row-index list inside a group is observable through
    // `collect`.
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN toUpper(b.cat) AS c, collect(a.value) AS v",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN toUpper(b.cat) AS c, count(*) AS n, min(a.value) AS lo",
];

/// Every aggregate the materialized grouping path serves, including the ones
/// the streaming pipeline declines (`collect`, `std`, `median`, `mode`,
/// `percentile_*`) and therefore hands here.
const AGG_QUERIES_EXTRA: &[&str] = &[
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, std(a.value) AS sd, variance(a.value) AS vr",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, median(a.value) AS md, mode(a.value) AS mo",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, percentile_cont(a.value, 0.9) AS p90, percentile_disc(a.value, 0.5) AS p50",
    "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN b.cat AS c, count(*) AS n ORDER BY n DESC, c ASC",
    "MATCH (a:Item)-[:LINKS]->(b:Item) WITH b.cat AS c, count(*) AS n WHERE n > 1 RETURN c, n",
];

/// parallel == serial, with **exact row order** and exact values.
///
/// `collect` is the order-sensitive one: it concatenates its group's values in
/// row order, and sequential grouping keeps each group's row indices in
/// that order while indexed parallel evaluation preserves group order. A set
/// comparison would not see a `collect` that came back permuted.
#[test]
fn parallel_aggregation_matches_serial_in_order() {
    let _meter = meter_guard();
    let graph = linked_graph(PARALLEL_MIN_ROWS_COMPILED * 2);
    let before = parallel_aggregations();
    for query in AGG_QUERIES.iter().chain(AGG_QUERIES_EXTRA) {
        let serial = run(&graph, query, false);
        let parallel = run(&graph, query, true);
        assert_eq!(serial.columns, parallel.columns, "columns differ: {query}");
        assert_eq!(
            serial.rows, parallel.rows,
            "parallel aggregation diverged from serial (values or group ORDER): {query}"
        );
        assert!(!serial.rows.is_empty(), "vacuous fixture for {query}");
    }
    assert!(
        parallel_aggregations() > before,
        "nothing fanned out — the assertions above compared two serial runs"
    );
}

/// Exact carried binding and list-order oracles on serial and parallel group
/// evaluation. The fixture has `value`, not `nid`; sorting an absent property
/// would make the carried-first-row obligation vacuous.
#[test]
fn parallel_aggregation_carries_the_global_first_row() {
    let _meter = meter_guard();
    let n = PARALLEL_MIN_ROWS_COMPILED * 2;
    let graph = linked_graph(n);
    let mut expected: Vec<(usize, Vec<Value>)> = Vec::new();
    for i in 0..n {
        let dst = (i * 7 + 13) % n;
        let key = Value::String(format!("CAT_{}", dst % 7));
        if let Some((_, row)) = expected.iter_mut().find(|(_, row)| row[0] == key) {
            let Value::List(values) = &mut row[1] else {
                unreachable!()
            };
            values.push(Value::Int64((i % 1000) as i64));
        } else {
            expected.push((
                dst % 1000,
                vec![key, Value::List(vec![Value::Int64((i % 1000) as i64)])],
            ));
        }
    }
    expected.sort_by_key(|(carried, _)| *carried);
    for mixed in [false, true] {
        let extra = if mixed { ",median(a.value) AS m" } else { "" };
        let query = format!("MATCH(a:Item)-[:LINKS]->(b:Item) RETURN toUpper(b.cat) AS c,collect(a.value) AS v{extra} ORDER BY b.value ASC");
        let serial = run(&graph, &query, false);
        let before = parallel_aggregations();
        let opted_in = run(&graph, &query, true);
        assert_eq!(parallel_aggregations() - before, 1);
        assert_eq!(serial.rows, opted_in.rows);
        let observed: Vec<Vec<Value>> = opted_in.rows.iter().map(|r| r[..2].to_vec()).collect();
        assert_eq!(
            observed,
            expected
                .iter()
                .map(|(_, row)| row.clone())
                .collect::<Vec<_>>()
        );
        assert!(expected.len() > 1);
    }
}

/// The gate and the opt-in, both directions.
#[test]
fn aggregation_respects_the_gate_and_the_opt_in() {
    let _meter = meter_guard();
    let query = AGG_QUERIES[1];

    let small = linked_graph(PARALLEL_MIN_ROWS_INTERPRETED - 1);
    let before = parallel_aggregations();
    run(&small, query, true);
    assert_eq!(
        parallel_aggregations(),
        before,
        "a below-gate aggregation fanned out"
    );

    let large = linked_graph(PARALLEL_MIN_ROWS_COMPILED * 2);
    let before = parallel_aggregations();
    run(&large, query, false);
    assert_eq!(
        parallel_aggregations(),
        before,
        "parallel=false fanned out evaluation across groups"
    );
}

/// A single-group aggregation has one unit of work, so fanning out across
/// groups would be pure hand-off cost: the gate requires at least two.
#[test]
fn single_group_aggregation_stays_serial() {
    let _meter = meter_guard();
    let graph = linked_graph(PARALLEL_MIN_ROWS_COMPILED * 2);
    let query = "MATCH (a:Item)-[:LINKS]->(b:Item) RETURN 1 AS one, collect(a.value) AS v";
    let before = parallel_aggregations();
    let opted_in = run(&graph, query, true);
    assert_eq!(
        parallel_aggregations(),
        before,
        "a one-group aggregation fanned out across groups"
    );
    assert_eq!(opted_in.rows, run(&graph, query, false).rows);
}

// ── ORDER BY sort-key precompute + regex cache (Q5) ─────────────────────────

use crate::graph::parallel::parallel_sort_keys;

/// The sort-key precompute is positional, so a parallel map is order-safe;
/// the sort that consumes it is stable and stays sequential. Ties must
/// therefore keep input order in both modes — which is what a low-cardinality
/// sort key checks.
#[test]
fn parallel_sort_keys_match_serial_in_order() {
    let _meter = meter_guard();
    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED * 4);
    let before = parallel_sort_keys();
    for query in [
        // `cat` has 7 values across 20k rows: almost every comparison is a
        // tie, so an unstable sort would shuffle visibly.
        "MATCH (n:Item) RETURN n.name AS nm, n.cat AS c ORDER BY n.cat ASC",
        "MATCH (n:Item) RETURN n.name AS nm ORDER BY n.value DESC, n.name ASC",
        "MATCH (n:Item) RETURN n.name AS nm ORDER BY toUpper(n.cat) ASC, n.nid ASC",
    ] {
        let serial = run(&graph, query, false);
        let parallel = run(&graph, query, true);
        assert_eq!(
            serial.rows, parallel.rows,
            "parallel sort keys changed the sorted order (stability?): {query}"
        );
        assert!(!serial.rows.is_empty(), "vacuous fixture for {query}");
    }
    assert!(
        parallel_sort_keys() > before,
        "no ORDER BY fanned out its sort-key precompute"
    );
}

#[test]
fn sort_keys_respect_the_gate_and_the_opt_in() {
    let _meter = meter_guard();
    let query = "MATCH (n:Item) RETURN n.name AS nm ORDER BY n.value DESC";

    let small = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED - 1);
    let before = parallel_sort_keys();
    run(&small, query, true);
    assert_eq!(
        parallel_sort_keys(),
        before,
        "a below-gate ORDER BY fanned out"
    );

    let large = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED * 4);
    let before = parallel_sort_keys();
    run(&large, query, false);
    assert_eq!(
        parallel_sort_keys(),
        before,
        "parallel=false fanned out its sort keys"
    );
}

/// The per-thread regex cache must answer identically to the shared one, and
/// an invalid pattern must still surface its error rather than being cached.
#[test]
fn regex_predicate_matches_across_modes() {
    let _meter = meter_guard();
    let graph = scan_graph(PARALLEL_MIN_ROWS_INTERPRETED * 4);
    for query in [
        "MATCH (n:Item) WHERE n.name =~ '.*_1[0-9][0-9]$' RETURN count(*) AS n",
        "MATCH (n:Item) WHERE n.cat =~ 'cat_[135]' RETURN n.name AS nm",
        "MATCH (n:Item) WHERE NOT n.cat =~ 'cat_[135]' RETURN count(*) AS n",
    ] {
        let serial = run(&graph, query, false);
        let parallel = run(&graph, query, true);
        assert_eq!(serial.rows, parallel.rows, "regex diverged: {query}");
        assert!(!serial.rows.is_empty(), "vacuous fixture for {query}");
    }
}

fn exact_sum_graph(cancel: bool) -> DirGraph {
    use crate::datatypes::DataFrame;
    let n = PARALLEL_MIN_ROWS_COMPILED + 1;
    let columns = ["nid", "name", "value", "group"]
        .map(str::to_string)
        .to_vec();
    let rows = (0..n)
        .map(|i| {
            let value = if i == 0 {
                i64::MAX
            } else if i == 1 {
                1
            } else if cancel && i == n - 1 {
                -i64::MAX
            } else {
                0
            };
            vec![
                Value::Int64(i as i64),
                Value::String(format!("N{i}")),
                Value::Int64(value),
                Value::Int64(i64::from(i == 2)),
            ]
        })
        .collect();
    let frame = DataFrame::from_cypher_rows(columns, rows).unwrap();
    let mut graph = DirGraph::new();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        frame,
        "Item".to_string(),
        "nid".to_string(),
        Some("name".to_string()),
        None,
    )
    .unwrap();
    graph
}

#[test]
fn exact_sum_parallel_partition_overflow_cancels_before_finalization() {
    let _meter = meter_guard();
    let graph = exact_sum_graph(true);
    let query = "MATCH(n:Item) RETURN sum(n.value) AS s";
    let before = parallel_scans();
    assert_eq!(run(&graph, query, false).rows, vec![vec![Value::Int64(1)]]);
    assert_eq!(parallel_scans(), before, "serial control must not fan out");
    assert_eq!(run(&graph, query, true).rows, vec![vec![Value::Int64(1)]]);
    assert!(
        parallel_scans() > before,
        "the exact aggregate must actually fan out"
    );
    assert_eq!(
        run(
            &graph,
            "MATCH(n:Item) RETURN n.group AS g,sum(n.value) AS s ORDER BY g",
            true
        )
        .rows,
        vec![
            vec![Value::Int64(0), Value::Int64(1)],
            vec![Value::Int64(1), Value::Int64(0)]
        ]
    );
}

#[test]
fn exact_sum_parallel_final_overflow_returns_no_partial_result() {
    let _meter = meter_guard();
    let graph = exact_sum_graph(false);
    let params = HashMap::new();
    let mut query = parser::parse_cypher("MATCH(n:Item) RETURN sum(n.value) AS s").unwrap();
    crate::graph::languages::cypher::planner::optimize(&mut query, &graph, &params);
    for parallel in [false, true] {
        let before = parallel_scans();
        let result = CypherExecutor::with_params(&graph, &params, None)
            .with_parallel(parallel)
            .execute(&query);
        assert!(result.unwrap_err().contains("Integer overflow in sum"));
        if parallel {
            assert!(parallel_scans() > before);
        }
    }
}
