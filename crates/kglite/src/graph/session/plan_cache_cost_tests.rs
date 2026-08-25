//! The optimized plan cache is **reads only**, and this is where that policy is
//! pinned from both sides.
//!
//! `prepare()` (`session/execute.rs`) inserts a plan only when
//! `is_mutation_query` says the statement does not write. Before that term
//! existed, every write inserted an entry whose key carried the graph version
//! that the same write was about to bump — measured at 600 identical serial
//! writes producing 600 insertions, **0 hits**, 88 evictions and a shared
//! 512-entry cache left entirely full of entries only the writer could reach.
//!
//! **What the bypass gives up, deliberately.** Two shapes did hit and no longer
//! can, and both are pinned below rather than left to be rediscovered:
//!
//! - two transactions forked from the same base version used to share
//!   `(graph_id, version)`, so the second one's identical write hit the first
//!   one's entry. That sharing has since been removed outright as a
//!   *correctness* fix — `fork_transaction` now mints a fresh `graph_id`
//!   (`session/transaction.rs`), because a shared key also served sibling
//!   forks each other's **read** plans, and one plan shape bakes a physical
//!   `NodeIndex` (see `a_sibling_fork_is_never_served_another_forks_plan`);
//! - a mutation that fails *after* `prepare()` (name collision, constraint,
//!   write scope) never reaches `bump_version`, so a retry used to hit.
//!
//! Both are same-version *replays*. The trade is a hit in that narrow window
//! against an insert every serial write pays — and, more importantly, against
//! a write loop evicting other graphs' live read plans out of a process-global
//! 512-entry cache (`a_write_burst_evicts_no_other_graphs_read_plan`).
//!
//! The read cases are not decoration: they are what stops the mutation
//! assertions from passing vacuously in a world where nothing is cached at all.
//!
//! The counters are thread-local and `#[cfg(test)]`-only
//! (`plan_cache::instrumentation`); `TEST_LOCK` serializes the cases that
//! depend on cache contents.

use std::collections::HashMap;

use super::execute::{execute_mut, execute_read, ExecuteOptions};
use super::transaction::Session;
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::languages::cypher::plan_cache;
use crate::graph::languages::cypher::plan_cache::instrumentation::{self, CacheStats};

/// A mutation whose text never changes — the shape a cache would most want to
/// serve, and the one the version in the key makes it useless for.
const WRITE: &str = "CREATE (:Item {id: 1})";
const READ: &str = "MATCH (n:Item) RETURN n.id";

fn empty_params() -> HashMap<String, Value> {
    HashMap::new()
}

/// `(lookups, hits, insertions)` — the three counters a case can assert
/// deterministically.
///
/// Evictions are deliberately projected out here: the cache is process-wide, so
/// whether any given insert also evicts depends on what ran before it in the
/// same process. `a_write_burst_evicts_no_other_graphs_read_plan` asserts the
/// eviction counter directly, where it is the point.
fn events(stats: CacheStats) -> (u64, u64, u64) {
    (stats.lookups, stats.hits, stats.insertions)
}

/// A seeded in-memory graph. Every case builds its own, so `graph_id` — which
/// is process-unique and part of the cache key — guarantees a cold start
/// without clearing the shared map.
fn seeded() -> DirGraph {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = DirGraph::new();
    execute_mut(&mut graph, WRITE, &opts).expect("seed write");
    graph
}

#[test]
fn repeated_read_hits_on_the_second_run() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let graph = seeded();
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);

    instrumentation::reset();
    execute_read(&graph, READ, &opts).expect("cold read");
    execute_read(&graph, READ, &opts).expect("warm read");

    let stats = instrumentation::totals();
    assert_eq!(
        events(stats.read),
        (2, 1, 1),
        "an unchanged graph must serve the second identical read from cache"
    );
    assert_eq!(
        events(stats.mutation),
        (0, 0, 0),
        "no mutation ran in this case"
    );
}

#[test]
fn serial_mutations_perform_no_plan_cache_insertion() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let mut graph = DirGraph::new();
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);

    const WRITES: u64 = 8;
    instrumentation::reset();
    for _ in 0..WRITES {
        execute_mut(&mut graph, WRITE, &opts).expect("write");
    }

    let stats = instrumentation::totals();
    assert_eq!(
        events(stats.mutation),
        (WRITES, 0, 0),
        "a write must insert nothing: its own bump_version would move the key \
         before the entry could ever be read back"
    );
    assert_eq!(events(stats.read), (0, 0, 0), "no read ran in this case");
}

#[test]
fn a_mutation_between_two_reads_invalidates_the_read_plan() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let mut graph = seeded();
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);

    execute_read(&graph, READ, &opts).expect("populate the read plan");

    // Control phase. Without an intervening write this read hits; if that term
    // ever stops holding, the invalidation phase below becomes vacuous — a
    // "miss" would prove nothing about the write.
    instrumentation::reset();
    execute_read(&graph, READ, &opts).expect("warm read");
    assert_eq!(
        events(instrumentation::totals().read),
        (1, 1, 0),
        "control: an unchanged graph serves this read from cache"
    );

    // Invalidation phase.
    execute_mut(&mut graph, WRITE, &opts).expect("write");
    instrumentation::reset();
    execute_read(&graph, READ, &opts).expect("post-write read");
    let stats = instrumentation::totals();
    assert_eq!(
        events(stats.read),
        (1, 0, 1),
        "the same read must miss after a write and re-plan against the new \
         version"
    );
    assert_eq!(events(stats.mutation), (0, 0, 0));
}

#[test]
fn transactions_forked_from_one_base_version_no_longer_reuse_a_mutation_plan() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let session = Session::new(seeded());
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);

    // Both transactions snapshot the same version, and `fork_transaction` used
    // to clone `graph_id` alongside it — so their working copies shared one
    // cache key. Before the mutation bypass, the second write hit the first's
    // entry: (2 lookups, 1 hit, 1 insertion), measured 2026-08-09.
    //
    // Two independent reasons now keep that from happening, and this case
    // survives as the pin on the first: the mutation bypass (nothing is
    // inserted) and, since the sibling-fork correctness fix, distinct
    // `graph_id`s (nothing would match even if it were).
    //
    // ⚠ THE TRADE, STATED SO IT IS NOT RE-LITIGATED AS A REGRESSION. This is
    // the *only* shape in which a cached mutation plan was ever reused — a
    // same-version replay: two forks of one base, or a retry of a write that
    // errored before `bump_version`. Every other write, including every write
    // a serial writer makes, paid an insert that nothing could read back. The
    // narrow reuse window was given up for that, and for the larger effect the
    // insert had on *other* graphs: a 512-entry process-global FIFO that a
    // single write loop saturates. If a workload ever appears that replays
    // identical writes across many same-version forks, this comment is where
    // to start — the answer is not to re-cache every write.
    let mut first = session.begin();
    let mut second = session.begin();

    instrumentation::reset();
    execute_mut(first.working_mut().expect("tx1 working"), WRITE, &opts).expect("tx1 write");
    execute_mut(second.working_mut().expect("tx2 working"), WRITE, &opts).expect("tx2 write");

    let stats = instrumentation::totals();
    assert_eq!(
        events(stats.mutation),
        (2, 0, 0),
        "a fork's write must insert nothing, independently of the fact that \
         sibling forks no longer share a key at all"
    );
}

#[test]
fn a_mutation_that_errors_before_the_version_bump_no_longer_lets_a_retry_hit() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let mut graph = DirGraph::new();
    graph
        .interner
        .try_register(
            crate::graph::schema::InternedKey::from_str("CollisionType"),
            "conflicting-existing",
        )
        .expect("register the colliding name");
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let colliding = "CREATE (:CollisionType {id: 1})";

    instrumentation::reset();
    // The name check runs *after* `prepare`, and `bump_version` is never
    // reached — so before the bypass this retry hit the failed attempt's own
    // entry: (2 lookups, 1 hit, 1 insertion), measured 2026-08-09. The second
    // half of the same-version-replay trade documented above.
    assert!(execute_mut(&mut graph, colliding, &opts).is_err());
    assert!(execute_mut(&mut graph, colliding, &opts).is_err());
    assert_eq!(
        graph.version(),
        0,
        "a failed write must not bump the version"
    );

    let stats = instrumentation::totals();
    assert_eq!(
        events(stats.mutation),
        (2, 0, 0),
        "an unbumped version no longer helps: a failed write cached nothing"
    );
}

/// The composition invariant — the durable win, and the one that does not
/// depend on a timing measurement to be believed.
///
/// The cache is process-global and holds 512 entries, so before the bypass a
/// single write loop did not merely waste its own effort: its 600 dead entries
/// pushed out every *other* graph's live read plan (measured: 88 evictions, a
/// cache left 512/512 mutation-keyed).
///
/// A **second graph** is what makes that observable. A same-graph read would be
/// invalidated by the writer's own `bump_version` regardless of eviction, so it
/// could never tell the two apart. Across graphs the reader's `graph_id`
/// differs and its version never moves — its plan can only disappear by being
/// evicted.
#[test]
fn a_write_burst_evicts_no_other_graphs_read_plan() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    plan_cache::clear_for_tests();
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);

    // The reader: one warm plan, then never written to again.
    let reader = seeded();
    execute_read(&reader, READ, &opts).expect("warm the reader's plan");

    // The writer: comfortably past the 512-entry FIFO cap, so pre-bypass this
    // loop would have cycled the entire cache through its own entries.
    let mut writer = DirGraph::new();
    const BURST: u64 = 600;
    instrumentation::reset();
    for _ in 0..BURST {
        execute_mut(&mut writer, WRITE, &opts).expect("write");
    }

    let stats = instrumentation::totals();
    assert_eq!(
        events(stats.mutation),
        (BURST, 0, 0),
        "a burst of {BURST} writes must leave no mutation-keyed entry behind"
    );
    assert_eq!(
        stats.mutation.evictions, 0,
        "a write that inserts nothing cannot evict anything"
    );
    assert!(
        plan_cache::entry_count_for_tests() < plan_cache::CACHE_CAPACITY,
        "the cache must not be saturated by a writer; {BURST} writes left {} of \
         {} entries resident",
        plan_cache::entry_count_for_tests(),
        plan_cache::CACHE_CAPACITY
    );

    // The decisive assertion. Pre-bypass this read missed — its plan had been
    // evicted by an unrelated graph's writes.
    instrumentation::reset();
    execute_read(&reader, READ, &opts).expect("read after the burst");
    assert_eq!(
        events(instrumentation::totals().read),
        (1, 1, 0),
        "{BURST} writes on an unrelated graph must not cost this reader its \
         cached plan"
    );
}

/// Two sibling working copies must never be served each other's plans.
///
/// The plan cache's soundness argument is that a hit means the graph is the
/// same state the plan was computed against, and `(graph_id, version)` is how
/// it decides that. `fork_transaction` used to clone **both**, so two
/// transactions forked from one base and each writing once arrived at an
/// identical key holding *different* graphs — and the second one's read was
/// served the first one's plan.
///
/// For almost every plan that is invisible: the passes make cost and ordering
/// decisions, so a plan computed against a sibling's data is at worst
/// mis-ordered. `fuse_anchored_edge_count` is the exception, and it is why this
/// test asserts a **row value** rather than a cache counter: it resolves the
/// literal `{id: VAL}` anchor to a physical `NodeIndex` and bakes that u32 into
/// `Clause::FusedCountAnchoredEdges` (`ast.rs`), which is an identity, not an
/// estimate. Reused across lineages it counts a *different node's* edges.
///
/// The case below is the minimal reproduction, measured before the fix:
/// `tx2` holds no node with `id: 5` at all, so the only correct answer is 0 —
/// and it returned **1**, the outgoing `:E` degree of whatever sat at the index
/// `tx1` had baked.
#[test]
fn a_sibling_fork_is_never_served_another_forks_plan() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    plan_cache::clear_for_tests();
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);

    let session = Session::new(DirGraph::new());
    let mut first = session.begin();
    let mut second = session.begin();

    // One write each, so both working copies sit one bump above the same base.
    execute_mut(
        first.working_mut().expect("tx1 working"),
        "CREATE (:T {id: 5})",
        &opts,
    )
    .expect("tx1 write");
    // `tx2` has no `id: 5`, and the node at tx1's anchor index here *does* have
    // an outgoing `:E` — which is what turns a shared key into a wrong number.
    execute_mut(
        second.working_mut().expect("tx2 working"),
        "CREATE (b:T {id: 77})-[:E]->(a:T {id: 88})",
        &opts,
    )
    .expect("tx2 write");

    const ANCHORED_COUNT: &str = "MATCH ({id: 5})-[:E]->(x) RETURN count(x) AS c";
    let tx1 = execute_read(first.current().expect("tx1 current"), ANCHORED_COUNT, &opts)
        .expect("tx1 read");
    let tx2 = execute_read(
        second.current().expect("tx2 current"),
        ANCHORED_COUNT,
        &opts,
    )
    .expect("tx2 read");

    assert_eq!(
        tx1.result.rows,
        vec![vec![crate::datatypes::Value::Int64(0)]],
        "tx1's own answer: it has an id-5 node, with no outgoing :E edges"
    );
    assert_eq!(
        tx2.result.rows,
        vec![vec![crate::datatypes::Value::Int64(0)]],
        "tx2 holds no node with id 5, so its only correct answer is 0; a 1 here \
         means it was served tx1's plan with tx1's anchor NodeIndex baked in"
    );
}

/// `text_bm25` stays cacheable, unlike `text_score`.
///
/// `text_score` is excluded structurally: its plan-time rewrite injects
/// embedding vectors as parameters, and only a param-less statement is cached.
/// `text_bm25` has no plan-time rewrite — the index is read at execution — so
/// nothing about it should force a query out of the cache. Read like that, this
/// is an absence, and an absence is exactly the claim that goes stale silently;
/// measuring the hit is what keeps it honest.
#[test]
fn a_text_bm25_read_is_cached_like_any_other() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = DirGraph::new();
    execute_mut(
        &mut graph,
        "CREATE (:Item {id: 1, body: 'a quick fox'})",
        &opts,
    )
    .expect("seed");
    crate::graph::text_indexes::build_text_index(&mut graph, "Item", "body", None)
        .expect("build the index");
    const BM25: &str = "MATCH (n:Item) RETURN text_bm25(n, 'body', 'fox') AS s";

    instrumentation::reset();
    execute_read(&graph, BM25, &opts).expect("cold read");
    execute_read(&graph, BM25, &opts).expect("warm read");

    assert_eq!(
        events(instrumentation::totals().read),
        (2, 1, 1),
        "text_bm25 carries no plan-time rewrite, so its plan is cacheable"
    );
}
