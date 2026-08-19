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

use crate::graph::parallel::{parallel_scans, PARALLEL_MIN_ROWS_INTERPRETED};

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
