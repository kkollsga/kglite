//! Journal-specific invariants

use super::*;

#[test]
fn rollback_reuses_the_vacated_slots() {
    let mut graph = seeded();
    let slots_before: Vec<usize> = graph
        .graph
        .node_indices()
        .map(|idx| idx.index())
        .collect::<Vec<_>>();
    expect_failure(
        &mut graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
    let slots_after: Vec<usize> = graph.graph.node_indices().map(|idx| idx.index()).collect();
    assert_eq!(
        slots_before, slots_after,
        "restored nodes must land on the slots they vacated"
    );
}

#[test]
fn successful_statement_leaves_no_journal_installed() {
    let mut graph = seeded();
    run(&mut graph, "CREATE (:Item {id: 1000})");
    assert!(
        graph.graph.take_undo().is_none(),
        "a committed statement must uninstall its journal"
    );
}

#[test]
fn failed_statement_leaves_no_journal_installed() {
    let mut graph = seeded();
    expect_failure(
        &mut graph,
        "CREATE (:Item {id: 1001}), (:Blocked {id: 1002})",
        Some(&["Item"]),
    );
    assert!(
        graph.graph.take_undo().is_none(),
        "a rolled-back statement must uninstall its journal"
    );
}

#[test]
fn a_second_statement_after_a_rollback_still_commits() {
    let mut graph = seeded();
    expect_failure(
        &mut graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
    // The graph must be fully usable afterwards — including the id index the
    // rollback invalidated and the slots it handed back.
    run(&mut graph, "CREATE (:Item {id: 1100, name: 'after'})");
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let out = execute_mut(&mut graph, "MATCH (n:Item) RETURN count(n) AS c", &opts)
        .expect("read after rollback");
    let rows = out.result.rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(
        format!("{:?}", rows[0][0]),
        format!("{:?}", Value::Int64(4))
    );
}

/// One statement of each mutating shape, run in an order where each leaves the
/// graph as the next expects it.
const ZERO_COPY_QUERIES: &[&str] = &[
    "CREATE (:Item {id: 2000, name: 'x'})",
    "MATCH (n:Item {id: 1}) SET n.qty = 11, n.name = 'renamed'",
    "MATCH (n:Item {id: 2000}) SET n:Featured",
    "MATCH (a:Item {id: 1}), (b:Item {id: 3}) CREATE (a)-[:LINKS {weight: 2}]->(b)",
    "MATCH (n:Item {id: 2000}) DETACH DELETE n",
    "MERGE (n:Item {id: 2001}) ON CREATE SET n.name = 'merged'",
];

/// Assert no statement copies a node, i.e. every one of them took the journal
/// path rather than the clone checkpoint.
///
/// The counter is `BACKEND_CLONE_NODES`, bumped by `impl Clone for
/// GraphBackend` by the number of nodes copied. That is what makes it a real
/// oracle for *which path ran*: the clone checkpoint forks the whole graph and
/// registers the fixture's node count, while the journal checkpoint clones a
/// `DirGraph` whose backend was deliberately emptied first and registers zero.
/// It counts nodes only — a `HashMap` clone bumps nothing — so it says nothing
/// about how much *else* a statement copied.
fn assert_statements_copy_zero_nodes(graph: &mut DirGraph, fixture: &str) {
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};

    for &query in ZERO_COPY_QUERIES {
        reset_backend_clone_count();
        run(graph, query);
        assert_eq!(
            backend_clone_nodes(),
            0,
            "statement must not copy any node on the {fixture} fixture: {query}"
        );
    }
}

/// The point of the whole exercise: a mutating statement on the journal path
/// must not copy a single node. This is the regression guard for the
/// O(V+E)-per-write cost the sprint removed — a benchmark would only notice
/// the reintroduction once the graph is large.
///
/// Nodes copied, not clones performed: the checkpoint does clone a `DirGraph`
/// whose backend has been deliberately emptied first (the schema shell), and
/// that O(1) clone is the design, not a regression.
#[test]
fn journalled_statements_copy_zero_nodes() {
    assert_statements_copy_zero_nodes(&mut seeded(), "plain");
}

/// The same guard on a graph that has been saved.
///
/// This arm is the one that would have caught the columnar veto. A fresh
/// `seeded()` graph has no column stores and no user indexes, so it cannot
/// take the clone path no matter what the gate says — which made the guard
/// above structurally blind to a gate that sends every *real* application
/// graph down the expensive path.
#[test]
fn journalled_statements_copy_zero_nodes_on_a_saved_graph() {
    assert_statements_copy_zero_nodes(&mut seeded_columnar(), "columnar");
}

/// The same guard on a graph with user-created indexes — the other half of
/// the blind spot.
#[test]
fn journalled_statements_copy_zero_nodes_on_an_indexed_graph() {
    assert_statements_copy_zero_nodes(&mut seeded_indexed(), "indexed");
}

/// The mapped backend now takes the **journal** path too.
///
/// This arm was the inverse until 2026-07-30: it asserted `copied > 0`,
/// pinning the whole-graph clone every mapped statement used to open, with a
/// note to invert rather than delete it when Mapped gained a journal. This is
/// that inversion. `MappedGraph.inner` is the same heap `StableDiGraph` as
/// `MemoryGraph.inner`, so the same capture seam applies; only `Disk`, which
/// has no petgraph and no `NodeIndex` identity for an `UndoEntry` to name,
/// still returns `false` from `supports_undo_journal`.
///
/// Keeping the arm rather than folding it into the list above is deliberate:
/// it is still the only place the *cost* of a mapped statement is measured,
/// and a regression that quietly re-vetoes Mapped would otherwise be invisible
/// again.
#[test]
fn mapped_statements_copy_zero_nodes() {
    let mut graph = seeded_mapped();
    assert!(
        graph.graph.node_count() > 0,
        "fixture must have nodes or the counter proves nothing"
    );
    assert_statements_copy_zero_nodes(&mut graph, "mapped");
}

/// Mapped rollback must be *correct*, whichever path it takes.
///
/// Separate from the cost arm above on purpose. The clone path and the journal
/// path produce identical observable state by construction, which is exactly
/// why the Python parity oracle cannot tell them apart — it compares outputs.
/// So the cost is pinned by a counter and correctness is pinned by the
/// fingerprint, and neither stands in for the other.
///
/// This is the arm that kept passing unchanged when Mapped switched to the
/// journal. If it ever fails, the journal restored less than the clone used
/// to, and the fingerprint says which field.
#[test]
fn mapped_rolls_back_completely() {
    let mut graph = seeded_mapped();
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
}

/// The mapped **silent** write path must record nothing in the undo journal.
///
/// [`GraphWrite::node_weight_mut_silent`] has a *trait default* that forwards
/// to the recorded `node_weight_mut`. `MemoryGraph` overrides it to bypass
/// capture; `MappedGraph` relying on the default was harmless only while
/// Mapped had no journal to capture into. The moment it got one, the default
/// makes every silent write clone a `NodeData` pre-image. The sweeps that used
/// to justify this — the columnar handle refresh and `add_nodes`'
/// detach/reattach dance — are gone with D1 Phase 3, but the override still
/// has to hold: `BatchProcessor::reattach_columnar_stores` remains a silent
/// per-node write, it runs only under `is_mapped() || is_disk()`, and no
/// memory-backed test exercises it. That is exactly the O(type)-per-write
/// amplification commit 3bf9ef00 removed from the WAL, re-created inside the
/// journal.
///
/// Nothing existing catches it. The WAL payload guard
/// (`tests/benchmarks/test_bench_wal_payload.py`) measures WAL *bytes*, which
/// this override does not change; every `BACKEND_CLONE_NODES` arm counts
/// backend clones, and the journal path clones no backend by construction. So
/// the assertion has to read the journal itself.
///
/// The recorded arm is the non-vacuity control: without it a test that
/// silently failed to install a journal, or looked at the wrong node, would
/// pass by reading zero from an empty buffer.
#[test]
fn the_mapped_silent_write_path_records_nothing() {
    use crate::graph::storage::GraphWrite;

    let mut graph = seeded_mapped();
    let idx = graph
        .graph
        .node_indices()
        .next()
        .expect("the fixture must have a node");

    // Control: the recorded seam does capture, so the journal is really live.
    graph.graph.begin_undo();
    GraphWrite::node_weight_mut(&mut graph.graph, idx).expect("node is live");
    let recorded = graph
        .graph
        .take_undo()
        .expect("begin_undo must install a journal on a mapped graph")
        .into_replay_order()
        .count();
    assert_eq!(
        recorded, 1,
        "the recorded seam must capture a pre-image, or the silent arm below \
         is comparing against an empty journal for the wrong reason"
    );

    // The claim: the silent seam captures nothing.
    graph.graph.begin_undo();
    GraphWrite::node_weight_mut_silent(&mut graph.graph, idx).expect("node is live");
    let silent = graph
        .graph
        .take_undo()
        .expect("begin_undo must install a journal on a mapped graph")
        .into_replay_order()
        .count();
    assert_eq!(
        silent, 0,
        "the mapped silent write path journalled {silent} entries; it must \
         journal none, or the columnar detach/reattach and handle-refresh \
         sweeps cost one pre-image per node of the type per chunk"
    );
}

/// Rollback is allowed to be the expensive direction, but it must still not
/// reach for a whole-graph copy on the journal path.
fn assert_rollback_copies_zero_nodes(graph: &mut DirGraph, fixture: &str) {
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};

    reset_backend_clone_count();
    expect_failure(
        graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
    assert_eq!(
        backend_clone_nodes(),
        0,
        "rollback must not copy any node on the {fixture} fixture"
    );
}

#[test]
fn journalled_rollback_copies_zero_nodes() {
    assert_rollback_copies_zero_nodes(&mut seeded(), "plain");
}

#[test]
fn journalled_rollback_copies_zero_nodes_on_a_saved_graph() {
    assert_rollback_copies_zero_nodes(&mut seeded_columnar(), "columnar");
}

#[test]
fn journalled_rollback_copies_zero_nodes_on_an_indexed_graph() {
    assert_rollback_copies_zero_nodes(&mut seeded_indexed(), "indexed");
}

/// The rollback half of the mapped switch. `mapped_rolls_back_completely`
/// pins that the restore is faithful; this pins that it got there without a
/// whole-graph copy.
#[test]
fn journalled_rollback_copies_zero_nodes_on_a_mapped_graph() {
    assert_rollback_copies_zero_nodes(&mut seeded_mapped(), "mapped");
}

/// Pins that the `columnar` arms exercise the master side channel rather than
/// quietly falling through to the per-node setter.
///
/// `execute_set`'s columnar fast path only fires for an in-memory `Columnar`
/// node writing a property that is neither `title` nor `name`. If that stopped
/// holding for this fixture — a storage-mode change, a different fallthrough
/// condition — every columnar rollback arm above would still pass, while
/// testing the write path that was never at risk. The veto this file's change
/// removed was aimed squarely at the master write, so "the master write
/// happens here" is a precondition of the whole exercise, not a detail.
#[test]
fn the_columnar_fixture_writes_through_the_master_store() {
    let mut graph = seeded_columnar();
    let before = fingerprint(&mut graph);
    run(&mut graph, "MATCH (n:Item {id: 1}) SET n.qty = 12345");
    let after = fingerprint(&mut graph);

    assert_ne!(
        before.column_masters, after.column_masters,
        "a successful columnar SET must land in the master store; if it does \
         not, the columnar arms are exercising the per-node fallback"
    );
    assert_eq!(
        before.columnar_rows, after.columnar_rows,
        "a columnar SET must not move any node to a different row — it writes \
         a cell of the store the backend owns, and the node's row identity is \
         exactly what must stay put"
    );
}
