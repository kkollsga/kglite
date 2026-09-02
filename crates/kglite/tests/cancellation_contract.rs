//! The cancellation contract, as a corpus of query shapes.
//!
//! **Every read query observes its deadline — and its cancel flag, which is
//! the same check — within a bounded wall-clock latency, whatever its shape.**
//!
//! Unobserved cancellation has bitten this engine three times, each time in a
//! loop nobody had thought to check: the 2026-06 interruptible-Cypher work,
//! the `Interrupt` plumbing through the graph algorithms, and the 2026-08
//! report of a 3-hop query that ran 120 s past a 30 s deadline and reached
//! 7.29 GB before the serving process was OOM-killed. Each was fixed where it
//! was found, by an agent who went looking for that one loop. This file exists
//! so the *next* unchecked loop fails a test instead of a downstream server:
//! each shape is a claim that one family of plans is bounded, and an operator
//! added without a poll turns its shape red.
//!
//! Each shape asserts only on the **cancelled** path — a short deadline must
//! produce the timeout error within `MAX_ABORT`, teardown included. The
//! uncancelled runtimes recorded in the comments are notes, not assertions,
//! because they are a property of the machine; what they establish is that
//! every shape runs far longer than `MAX_ABORT` when nothing stops it, so a
//! missing poll cannot pass. Regenerate them with
//!
//! ```text
//! cargo test -p kglite --test cancellation_contract -- --ignored --nocapture
//! ```
//!
//! Adding a shape: put the query in the `CORPUS` table, give it a
//! `deadline_shape!` test, and run the measurement above. A shape whose
//! uncancelled runtime is not comfortably above `MAX_ABORT` proves nothing —
//! grow the fixture or the query until it is.
//!
//! **What this corpus can and cannot reach.** It aborts each plan a fixed
//! 100 ms in, so it tests whichever phase the plan is still in at 100 ms —
//! for most shapes, the matcher or the first row loop. A poll in a *later*
//! phase is unreachable from here by construction: the path-binding pass, for
//! one, starts only once matching and joining have both finished, by which
//! time any deadline short enough to be a bounded-vs-unbounded discriminator
//! has already fired upstream. Those sites are pinned instead by the
//! poll-counting hook tests in
//! `graph::languages::cypher::executor::tests::deadline_rows`, which can put
//! the abort in an exact loop, and by
//! `executor::tests::expressions::count_subquery_expression_propagates_deadline_and_cancellation`
//! for the unanchored-scan poll. The division is deliberate: the hook tests
//! pin individual poll *sites*, this file pins whole plan *families*, and only
//! this file notices a family that has grown a new unpolled operator.
//!
//! Verified by deleting one poll at a time and re-running (2026-08-30):
//! removing the `first_pattern_rows` poll reddens shape 1, removing the
//! comma-join fan-out poll reddens shape 3, and removing the cancel-flag check
//! from the variable-length matcher reddened the var-length interrupt shape —
//! which is how that bug was found.

use kglite::api::session::{execute_mut, execute_read, ExecuteOptions};
use kglite::api::{DirGraph, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// How long a query gets before its deadline fires.
const DEADLINE: Duration = Duration::from_millis(100);

/// The wall-clock ceiling a cancelled run must fit inside, measured from the
/// call and including teardown — dropping the partial row set is part of the
/// latency the caller sees, and Phase 1 found teardown rather than detection
/// dominates the residual overshoot.
///
/// Derived from measurement, from both ends. Every shape in the corpus aborts
/// 7–33 ms past its 100 ms deadline on this machine (worst: 133 ms, the
/// triple-regex scan), so the ceiling carries ~7× slack for a loaded CI
/// runner — and the slack grows rather than shrinks on a slower machine, since
/// the residual is one poll interval of work, not a fixed number of
/// milliseconds. From the other end, the slowest thing it has to catch is an
/// *unpolled* shape, and the fastest shape here runs 2.24 s unbounded. This is
/// a bounded-vs-unbounded discriminator, not a latency budget.
const MAX_ABORT: Duration = Duration::from_millis(1000);

fn no_params() -> HashMap<String, Value> {
    HashMap::new()
}

fn run_mut(graph: &mut DirGraph, query: &str) {
    let params = no_params();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts)
        .unwrap_or_else(|e| panic!("fixture step failed: {query}: {e}"));
}

/// The shared corpus fixture, behind a mutex that doubles as the timing lock:
/// cargo runs these tests on parallel threads, and an abort timed while
/// seventeen sibling shapes hammer the same allocator measures the harness.
static FIXTURE: OnceLock<Mutex<DirGraph>> = OnceLock::new();

fn fixture() -> MutexGuard<'static, DirGraph> {
    FIXTURE
        .get_or_init(|| Mutex::new(build_fixture()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One graph for every shape, sized for *cardinality* rather than size: the
/// fan-in structure yields `HUBS * FAN * FAN` = 1,014,000 three-element matches
/// out of 15,660 nodes, which is what lets a shape run for seconds while the
/// fixture builds in ~3 s (measured; the `:Doc` rows are most of it).
///
/// - `:B` — `HUBS` hubs, chained `(b)-[:HUBLINK]->(b+1)`. The chain is what
///   makes the component's diameter large enough for an all-pairs algorithm to
///   be expensive; without it every BFS terminates two hops out and shape 14
///   finishes in 130 ms, too fast to prove anything.
/// - `:A` / `:C` — `HUBS * FAN` each, one edge apiece into a hub, carrying a
///   `tag` string so a text predicate has a million values to evaluate.
/// - `:P` — a mid-sized standalone label, the right-hand side of the cartesian.
/// - `:K` — a 14-clique: variable-length paths over it explode combinatorially.
/// - `:Doc` — `DOCS` string-carrying nodes, the type-scan universe.
fn build_fixture() -> DirGraph {
    const HUBS: i64 = 60;
    const FAN: i64 = 130;
    const PEERS: i64 = 400;
    const CLIQUE: i64 = 14;
    const DOCS: i64 = 500_000;

    let mut graph = DirGraph::new();
    run_mut(
        &mut graph,
        &format!("UNWIND range(1, {HUBS}) AS b CREATE (:B {{bid: b}})"),
    );
    run_mut(
        &mut graph,
        &format!(
            "MATCH (x:B) UNWIND range(1, {FAN}) AS i \
             CREATE (a:A {{aid: i, tag: 'tag-alpha-' + toString(i)}})-[:R1]->(x)"
        ),
    );
    run_mut(
        &mut graph,
        &format!(
            "MATCH (x:B) UNWIND range(1, {FAN}) AS i \
             CREATE (c:C {{cid: i, tag: 'tag-beta-' + toString(i)}})-[:R2]->(x)"
        ),
    );
    run_mut(
        &mut graph,
        &format!("UNWIND range(1, {PEERS}) AS p CREATE (:P {{pid: p}})"),
    );
    run_mut(
        &mut graph,
        "MATCH (a:B), (b:B) WHERE b.bid = a.bid + 1 CREATE (a)-[:HUBLINK]->(b)",
    );
    run_mut(
        &mut graph,
        &format!("UNWIND range(1, {CLIQUE}) AS k CREATE (:K {{kid: k}})"),
    );
    run_mut(
        &mut graph,
        "MATCH (a:K), (b:K) WHERE a.kid < b.kid CREATE (a)-[:KE]->(b)",
    );
    run_mut(
        &mut graph,
        &format!(
            "UNWIND range(1, {DOCS}) AS i CREATE (:Doc {{did: i, \
             body: 'lorem ipsum dolor sit amet magna aliqua ' + toString(i)}})"
        ),
    );
    graph
}

/// The contract, applied to one shape: a deadline `DEADLINE` from now must
/// produce the timeout error, and the call must return within `MAX_ABORT`
/// including teardown.
fn assert_deadline_is_observed(graph: &DirGraph, query: &str) {
    let params = no_params();
    let mut opts = ExecuteOptions::eager(&params);
    opts.deadline = Some(Instant::now() + DEADLINE);

    let started = Instant::now();
    let outcome = execute_read(graph, query, &opts);
    let elapsed = started.elapsed();

    match outcome {
        Ok(result) => panic!(
            "the deadline was never observed: {query}\n\
             returned {} rows in {elapsed:?} — this shape ran to completion past a \
             {DEADLINE:?} deadline, so some loop on its plan polls nothing",
            result.result.rows.len()
        ),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("timed out"),
                "expected a timeout for {query}, got: {message}"
            );
        }
    }
    assert!(
        elapsed < MAX_ABORT,
        "{query}\naborted only after {elapsed:?} — a {DEADLINE:?} deadline must be \
         observed within {MAX_ABORT:?}, teardown included"
    );
    // Visible under `--nocapture`; the source of the recorded abort latencies.
    println!("deadline abort in {elapsed:?}: {query}");
}

/// The same contract driven by the cooperative-cancellation flag instead of the
/// clock — the Ctrl-C path.
///
/// The two share every check site: `CypherExecutor::check_deadline` and
/// `Interrupt::exceeded` each test the clock *and* the flag, and nothing polls
/// one without the other. So parity needs a sample rather than a second full
/// corpus, and the sample is chosen by *carrier* — one shape per distinct
/// interrupt object on the read path — not by plan shape.
///
/// `ExecuteOptions::cancel` takes a `&'static AtomicBool` because its only
/// production setter is a signal handler, which cannot capture state; each call
/// here leaks one, and the corpus flips three.
fn assert_interrupt_is_observed(graph: &DirGraph, query: &str) {
    let flag: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
    let params = no_params();
    let mut opts = ExecuteOptions::eager(&params);
    opts.cancel = Some(flag);

    let raiser = std::thread::spawn(move || {
        std::thread::sleep(DEADLINE);
        flag.store(true, Ordering::Relaxed);
    });

    let started = Instant::now();
    let outcome = execute_read(graph, query, &opts);
    let elapsed = started.elapsed();
    raiser.join().expect("the interrupt thread must finish");

    match outcome {
        Ok(result) => panic!(
            "the interrupt was never observed: {query}\nreturned {} rows in {elapsed:?}",
            result.result.rows.len()
        ),
        Err(e) => {
            let message = e.to_string().to_lowercase();
            assert!(
                message.contains("cancel") || message.contains("interrupt"),
                "expected a cancellation for {query}, got: {message}"
            );
        }
    }
    assert!(
        elapsed < MAX_ABORT,
        "{query}\nobserved its interrupt only after {elapsed:?}, ceiling {MAX_ABORT:?}"
    );
    println!("interrupt observed in {elapsed:?}: {query}");
}

// ── The corpus ────────────────────────────────────────────────────────────
//
// Each constant is one plan family, and each carries the pair of numbers that
// makes its test meaningful: how long it runs with nothing stopping it, and
// how long it took to stop. Both measured 2026-08-30, debug profile, on the
// development machine under ordinary load.

/// 1. Full type scan with a `WHERE`, over 500k nodes. The scan itself is cheap
///    and the three regexes are not, and they fuse into the MATCH — so this
///    shape spends its 100 ms in the match-to-row conversion, and deleting that
///    loop's poll is what reddens it (verified). The scan loop's own poll is
///    pinned by a unit test; see the module docs.
///    Uncancelled 2.24 s / 500,000 rows · abort 133 ms.
const SCAN_WITH_WHERE: &str = "MATCH (d:Doc) WHERE d.did > 0 AND d.body =~ '.*aliqua [0-9]+' \
     AND d.body =~ '(?i).*dolor.*' AND d.body =~ '.*ipsum.*' RETURN d.did";

/// 2. Multi-hop join with high intermediate cardinality — the plan family the
///    downstream OOM report was about, where a 30 s deadline overran by 90 s
///    because everything past the matcher polled nothing. At 100 ms the abort
///    lands in the matcher's expansion, so what this shape guards is that the
///    family is bounded at all; the row loop past it is where `deadline_rows`
///    puts its own aborts.
///    Uncancelled 2.35 s / 1,014,000 rows · abort 116 ms.
const MULTI_HOP_JOIN: &str = "MATCH (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) \
     WHERE a.aid + b.bid + c.cid > -1 AND toString(a.aid) <> 'zzz' RETURN a.aid";

/// 3. Comma-pattern cartesian product inside one MATCH clause.
///    Uncancelled 4.15 s / 3,120,000 rows · abort 115 ms.
const COMMA_CARTESIAN: &str = "MATCH (a:A), (p:P) WHERE a.aid + p.pid > -1 RETURN a.aid, p.pid";

/// 4. Variable-length path over the clique. Unbounded this one does not even
///    finish: it hits the 10,000,000-row safety ceiling at 4.66 s, which is
///    itself the point — the deadline has to win the race to that ceiling, and
///    does, by a factor of forty.
///    Abort 113 ms.
const VAR_LENGTH_PATH: &str = "MATCH (a:K)-[*1..8]-(b:K) RETURN a.kid, b.kid";

/// 5. ORDER BY over a large materialized set.
///    Uncancelled 3.12 s / 1,014,000 rows · abort 116 ms.
const ORDER_BY_LARGE: &str =
    "MATCH (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) RETURN a.aid, c.cid ORDER BY a.aid, c.cid";

/// 6. Grouped aggregation folding a million rows into 16,900 groups.
///    Uncancelled 2.71 s · abort 116 ms.
const GROUPED_AGGREGATION: &str =
    "MATCH (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) RETURN a.aid, c.cid, count(*) AS n";

/// 7. UNWIND of a large list crossed with a MATCH fan-out.
///    Uncancelled 3.12 s / 3,120,000 rows · abort 113 ms.
const UNWIND_FANOUT: &str =
    "UNWIND range(1, 400) AS i MATCH (a:A)-[:R1]->(b:B) RETURN i, a.aid, b.bid";

/// 8. WITH-chained multi-stage pipeline — the subsequent-MATCH driving join,
///    where each incoming row expands on its own.
///    Uncancelled 3.72 s / 1,014,000 rows · abort 109 ms.
const WITH_PIPELINE: &str = "MATCH (a:A)-[:R1]->(b:B) WITH a, b \
     MATCH (c:C)-[:R2]->(b) WITH a, b, c WHERE c.cid > 0 RETURN a.aid, c.cid";

/// 9. A CALL subquery whose body is shape 2, correlated to the outer row: the
///    long loop is one nesting level below the clause the executor is running.
///    Uncancelled 2.40 s / 1,014,000 rows · abort 115 ms.
const CALL_SUBQUERY: &str = "MATCH (b:B) CALL { WITH b \
     MATCH (a:A)-[:R1]->(b)<-[:R2]-(c:C) WHERE toString(a.aid) <> 'zzz' \
     RETURN a.aid AS x, c.cid AS y } RETURN x, y";

/// 10. UNION of two long arms.
///     Uncancelled 3.58 s · abort 115 ms.
const UNION_LONG_ARMS: &str = "MATCH (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) \
     WHERE toString(a.aid) <> 'zzz' RETURN a.aid AS x \
     UNION MATCH (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) \
     WHERE toString(c.cid) <> 'zzz' RETURN c.cid AS x";

/// 11. EXCEPT with a cheap left arm and an expensive right one. The right arm
///     contributes no rows to the answer and still has to be bounded — a set
///     operation cannot be interruptible only where it returns data.
///     Uncancelled 2.68 s · abort 116 ms.
const EXCEPT_LONG_RIGHT: &str = "MATCH (a:A) RETURN a.aid AS x \
     EXCEPT MATCH (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) \
     WHERE a.aid + b.bid + c.cid > -1 AND toString(c.cid) <> 'zzz' RETURN c.cid AS x";

/// 12. Path functions over bound paths. The propagation pass itself runs after
///     matching and joining are finished and so is out of this corpus's reach
///     (module docs); what this shape covers is that binding a path variable
///     does not route the plan around the polls the same query without `p`
///     goes through.
///     Uncancelled 2.80 s / 1,014,000 rows · abort 117 ms.
const PATH_FUNCTIONS: &str = "MATCH p = (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) \
     RETURN size(nodes(p)) + size(relationships(p)) AS n";

/// 13. Seven text predicates evaluated over a million rows, all of them true
///     so none short-circuits the rest away. Deliberately driven by cardinality
///     rather than by node count: per-row scan overhead swamps `CONTAINS`, so a
///     scan-shaped version of this needs ~1.8M nodes to run as long as the join
///     does with 15,660.
///     Uncancelled 4.73 s / 1,014,000 rows · abort 107 ms.
const TEXT_SCAN: &str = "MATCH (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) \
     WHERE a.tag CONTAINS 'alpha' AND c.tag CONTAINS 'beta' AND a.tag STARTS WITH 'tag' \
     AND c.tag STARTS WITH 'tag' AND a.tag CONTAINS 'g-al' AND c.tag CONTAINS 'g-be' \
     AND a.tag ENDS WITH a.tag RETURN a.aid";

/// 14. A graph algorithm reached through CALL. Different carrier: the
///     algorithms take an `Interrupt` and poll `Interrupt::exceeded` on their
///     own strides, entirely outside the executor's row loops.
///     Uncancelled 4.01 s · abort 120 ms.
const ALGORITHM_CALL: &str =
    "CALL betweenness({node_type: ['A', 'B', 'C']}) YIELD node, score RETURN node, score";

/// 15. DISTINCT over a million rows, none of which dedup away.
///     Uncancelled 2.33 s / 1,014,000 rows · abort 115 ms.
const DISTINCT_LARGE: &str = "MATCH (a:A)-[:R1]->(b:B)<-[:R2]-(c:C) \
     RETURN DISTINCT a.aid, b.bid, c.cid, toString(a.aid) + toString(c.cid)";

/// The table the measurement test walks. Every entry has a `deadline_shape!`
/// test below; a shape measured but not asserted is the gap this file exists
/// to close.
const CORPUS: &[(&str, &str)] = &[
    ("1  scan + WHERE", SCAN_WITH_WHERE),
    ("2  multi-hop join", MULTI_HOP_JOIN),
    ("3  comma cartesian", COMMA_CARTESIAN),
    ("4  var-length path", VAR_LENGTH_PATH),
    ("5  ORDER BY", ORDER_BY_LARGE),
    ("6  grouped aggregation", GROUPED_AGGREGATION),
    ("7  UNWIND fan-out", UNWIND_FANOUT),
    ("8  WITH pipeline", WITH_PIPELINE),
    ("9  CALL subquery", CALL_SUBQUERY),
    ("10 UNION", UNION_LONG_ARMS),
    ("11 EXCEPT", EXCEPT_LONG_RIGHT),
    ("12 path functions", PATH_FUNCTIONS),
    ("13 text predicates", TEXT_SCAN),
    ("14 algorithm CALL", ALGORITHM_CALL),
    ("15 DISTINCT", DISTINCT_LARGE),
];

macro_rules! deadline_shape {
    ($name:ident, $query:ident) => {
        #[test]
        fn $name() {
            let graph = fixture();
            assert_deadline_is_observed(&graph, $query);
        }
    };
}

deadline_shape!(scan_with_where_observes_its_deadline, SCAN_WITH_WHERE);
deadline_shape!(multi_hop_join_observes_its_deadline, MULTI_HOP_JOIN);
deadline_shape!(comma_cartesian_observes_its_deadline, COMMA_CARTESIAN);
deadline_shape!(var_length_path_observes_its_deadline, VAR_LENGTH_PATH);
deadline_shape!(order_by_observes_its_deadline, ORDER_BY_LARGE);
deadline_shape!(
    grouped_aggregation_observes_its_deadline,
    GROUPED_AGGREGATION
);
deadline_shape!(unwind_fanout_observes_its_deadline, UNWIND_FANOUT);
deadline_shape!(with_pipeline_observes_its_deadline, WITH_PIPELINE);
deadline_shape!(call_subquery_observes_its_deadline, CALL_SUBQUERY);
deadline_shape!(union_observes_its_deadline, UNION_LONG_ARMS);
deadline_shape!(except_observes_its_deadline, EXCEPT_LONG_RIGHT);
deadline_shape!(path_functions_observe_their_deadline, PATH_FUNCTIONS);
deadline_shape!(text_predicates_observe_their_deadline, TEXT_SCAN);
deadline_shape!(algorithm_call_observes_its_deadline, ALGORITHM_CALL);
deadline_shape!(distinct_observes_its_deadline, DISTINCT_LARGE);

// ── Interrupt parity ──────────────────────────────────────────────────────
//
// One shape per interrupt carrier on the read path, per the sampling argument
// on `assert_interrupt_is_observed`.

/// The executor's own row loops (`CypherExecutor::check_interrupt_periodic`).
#[test]
fn multi_hop_join_observes_an_interrupt() {
    let graph = fixture();
    assert_interrupt_is_observed(&graph, MULTI_HOP_JOIN);
}

/// The pattern matcher's checkpoints (`PatternMatcher::interrupt_reason`),
/// which a variable-length expansion spends all its time inside.
#[test]
fn var_length_path_observes_an_interrupt() {
    let graph = fixture();
    assert_interrupt_is_observed(&graph, VAR_LENGTH_PATH);
}

/// The algorithms' `Interrupt`, which neither of the above reaches.
#[test]
fn algorithm_call_observes_an_interrupt() {
    let graph = fixture();
    assert_interrupt_is_observed(&graph, ALGORITHM_CALL);
}

// ── Mutations ─────────────────────────────────────────────────────────────

/// A long mutation observes its deadline too, and the statement it abandons
/// leaves nothing behind.
///
/// Worth stating explicitly, because a neighbouring decision reads like an
/// exemption and is not one. The bindings deliberately do **not** wire the
/// cancel flag on the live-KG and Transaction mutation paths — both mutate a
/// singly-owned graph in place, so a Ctrl-C could strand partial state (the
/// `cancel: None` sites in `kglite-py`'s `kg_core.rs` and `transaction.rs` say
/// so, and add "the deadline still bounds this path"). That is a decision about
/// *one carrier on two binding paths*, not an atomicity exemption from the
/// deadline: `execute_mut` polls `write.rs::check_interrupt_periodic` through
/// its row loops and unwinds the statement through its rollback checkpoint.
///
/// Measured: this 400,000-node `CREATE` runs 1.79 s uncancelled and returns
/// 117–429 ms after a 100 ms deadline, leaving zero nodes; the spread is the
/// rollback, whose cost depends on how many rows the statement got through
/// before the poll fired. A `SET` over 400,000 nodes runs 843 ms uncancelled
/// and returns after 135 ms with the graph intact.
///
/// The wording is pinned, and pinned to the *read* path's: both routes report
/// through `executor::check_interrupt`, so a timed-out mutation says "timed
/// out" and a cancelled one says "cancelled". Before that, the mutation engine
/// ran its own poller reporting a flat "Query interrupted" for either, and a
/// caller could not tell a mutation timeout from any other mutation failure.
#[test]
fn a_mutation_observes_its_deadline_and_rolls_back() {
    let mut graph = DirGraph::new();
    let params = no_params();
    let mut opts = ExecuteOptions::eager(&params);
    opts.deadline = Some(Instant::now() + DEADLINE);

    let started = Instant::now();
    let outcome = execute_mut(
        &mut graph,
        "UNWIND range(1, 400000) AS i CREATE (:M {v: i})",
        &opts,
    );
    let elapsed = started.elapsed();

    let message = match outcome {
        Ok(_) => {
            panic!("the mutation ran to completion past a {DEADLINE:?} deadline in {elapsed:?}")
        }
        Err(e) => e.to_string().to_lowercase(),
    };
    assert!(
        message.contains("timed out"),
        "expected a timeout, got: {message}"
    );

    let read_params = no_params();
    let read_opts = ExecuteOptions::eager(&read_params);
    let rows = execute_read(&graph, "MATCH (n:M) RETURN n.v", &read_opts)
        .expect("the graph is readable after the abandoned mutation")
        .result
        .rows
        .len();
    assert_eq!(
        rows, 0,
        "the abandoned CREATE left {rows} nodes behind — the statement did not roll back"
    );
    println!("mutation abort in {elapsed:?}, {rows} nodes retained");
}

/// The mutation twin of [`assert_interrupt_is_observed`]: the same statement
/// stopped by the cancel flag instead of the clock reports a *cancellation*,
/// and still rolls back.
///
/// This is the pair that catches a regression to a collapsed poller: one
/// carrier reporting the other's wording passes neither of these two tests.
#[test]
fn a_mutation_observes_its_cancel_flag_and_rolls_back() {
    let flag: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
    let mut graph = DirGraph::new();
    let params = no_params();
    let mut opts = ExecuteOptions::eager(&params);
    opts.cancel = Some(flag);

    let raiser = std::thread::spawn(move || {
        std::thread::sleep(DEADLINE);
        flag.store(true, Ordering::Relaxed);
    });

    let started = Instant::now();
    let outcome = execute_mut(
        &mut graph,
        "UNWIND range(1, 400000) AS i CREATE (:M {v: i})",
        &opts,
    );
    let elapsed = started.elapsed();
    raiser.join().expect("the interrupt thread must finish");

    let message = match outcome {
        Ok(_) => panic!("the mutation ran to completion past its cancel flag in {elapsed:?}"),
        Err(e) => e.to_string().to_lowercase(),
    };
    assert!(
        message.contains("cancel"),
        "expected a cancellation, got: {message}"
    );

    let read_params = no_params();
    let read_opts = ExecuteOptions::eager(&read_params);
    let rows = execute_read(&graph, "MATCH (n:M) RETURN n.v", &read_opts)
        .expect("the graph is readable after the abandoned mutation")
        .result
        .rows
        .len();
    assert_eq!(
        rows, 0,
        "the cancelled CREATE left {rows} nodes behind — the statement did not roll back"
    );
    println!("mutation cancel in {elapsed:?}, {rows} nodes retained");
}

#[test]
#[ignore = "measurement, not an assertion: run with --ignored --nocapture to refresh the notes"]
fn measure_uncancelled_runtimes() {
    let build_started = Instant::now();
    let graph = fixture();
    println!("\nfixture build {:?}", build_started.elapsed());
    for (name, query) in CORPUS {
        let params = no_params();
        let opts = ExecuteOptions::eager(&params);
        let started = Instant::now();
        let outcome = execute_read(&graph, query, &opts);
        let elapsed = started.elapsed();
        match outcome {
            Ok(r) => println!("{name:26} {elapsed:>12.3?}  {} rows", r.result.rows.len()),
            Err(e) => println!("{name:26} {elapsed:>12.3?}  ERR {e}"),
        }
    }
}
