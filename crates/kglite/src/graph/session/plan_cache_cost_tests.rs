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
//! - two transactions forked from the same base version share `(graph_id,
//!   version)` (`session/transaction.rs`, `fork_transaction` clones both), so
//!   the second one's identical write used to hit the first one's entry;
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

    // Both transactions snapshot the same version, and `fork_transaction`
    // clones `graph_id` and `version` — so their working copies share one cache
    // key. Before the mutation bypass, the second write hit the first's entry:
    // (2 lookups, 1 hit, 1 insertion), measured 2026-08-09.
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
        "same-base-version forks share a key, but nothing is cached to share"
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
