//! What the optimized plan cache costs a *writer*, measured rather than argued.
//!
//! `prepare()` (`session/execute.rs`) looks a plan up and inserts one with no
//! `is_mutation` term, and the key carries the graph version
//! (`cypher/plan_cache.rs`), which every successful mutation bumps
//! (`execute.rs`, `bump_version`). For a serial writer that combination means
//! each write pays an `RwLock` write, a FIFO push and a retained `Arc` for an
//! entry the next write can never reach.
//!
//! **"Never" is a measurement, not a deduction, and these cases pin both
//! halves of it.** Two shapes really do hit:
//!
//! - two transactions forked from the same base version share `(graph_id,
//!   version)` (`session/transaction.rs`, `fork_transaction` clones both), so
//!   the second one's identical write hits the first one's entry;
//! - a mutation that fails *after* `prepare()` (name-collision, constraint,
//!   scope) never reaches `bump_version`, so a retry hits.
//!
//! Those two are documented here as facts about the current code, so that
//! whatever the plan-cache policy becomes, the change to them is deliberate and
//! visible in a diff rather than discovered later.
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

/// A mutation whose text never changes, which is what makes the waste visible:
/// an identical string is exactly the case the cache is *supposed* to serve,
/// and the only reason it cannot is the version in the key.
const WRITE: &str = "CREATE (:Item {id: 1})";
const READ: &str = "MATCH (n:Item) RETURN n.id";

fn empty_params() -> HashMap<String, Value> {
    HashMap::new()
}

/// `(lookups, hits, insertions)` — the three counters a case can assert
/// deterministically.
///
/// Evictions are deliberately projected out here. The cache is process-wide
/// and a single write loop saturates it (see the capacity case below), so
/// whether any given insert also evicts depends on what ran before it in the
/// same process. `a_write_loop_fills_the_512_entry_cache_with_entries_only_it_can_reach`
/// asserts the eviction counter directly, where the fill is the point.
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
fn serial_mutations_never_hit_and_insert_one_dead_entry_each() {
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
        (WRITES, 0, WRITES),
        "a serial writer repeating one statement looks up and inserts once per \
         write and can never hit: its own bump_version moved the key"
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
fn transactions_forked_from_one_base_version_do_reuse_a_mutation_plan() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let session = Session::new(seeded());
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);

    // Both transactions snapshot the same version, and `fork_transaction`
    // clones `graph_id` and `version` — so their working copies share the
    // cache key the first write inserts under.
    let mut first = session.begin();
    let mut second = session.begin();

    instrumentation::reset();
    execute_mut(first.working_mut().expect("tx1 working"), WRITE, &opts).expect("tx1 write");
    execute_mut(second.working_mut().expect("tx2 working"), WRITE, &opts).expect("tx2 write");

    let stats = instrumentation::totals();
    assert_eq!(
        events(stats.mutation),
        (2, 1, 1),
        "same-base-version forks share the key, so the second identical write \
         hits — this is the one shape where a cached mutation plan is reused"
    );
}

#[test]
fn a_mutation_that_errors_before_the_version_bump_lets_a_retry_hit() {
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
    // The name check runs *after* `prepare` — so the plan is cached, then the
    // statement fails, and `bump_version` is never reached.
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
        (2, 1, 1),
        "an unbumped version means the retry finds its own plan"
    );
}

#[test]
fn a_write_loop_fills_the_512_entry_cache_with_entries_only_it_can_reach() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    plan_cache::clear_for_tests();
    let mut graph = DirGraph::new();
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);

    // Comfortably past the 512-entry FIFO cap, so the steady state — not the
    // fill — is what is asserted.
    const WRITES: u64 = 600;
    instrumentation::reset();
    for _ in 0..WRITES {
        execute_mut(&mut graph, WRITE, &opts).expect("write");
    }

    let stats = instrumentation::totals();
    assert_eq!(stats.mutation.insertions, WRITES);
    assert_eq!(stats.mutation.hits, 0);
    assert_eq!(
        stats.read.insertions, 0,
        "every entry this loop added is mutation-keyed"
    );
    let capacity = plan_cache::CACHE_CAPACITY as u64;
    assert_eq!(
        plan_cache::entry_count_for_tests(),
        plan_cache::CACHE_CAPACITY,
        "one write loop saturates the whole shared cache"
    );
    // `>=` rather than `==`: another test thread may hold entries this loop
    // then evicts. The floor is what matters — a saturated cache evicts a
    // previously-cached plan for every further write.
    assert!(
        stats.mutation.evictions >= WRITES - capacity,
        "expected at least {} evictions, saw {}",
        WRITES - capacity,
        stats.mutation.evictions
    );
}
