//! CDC core tests.
//!
//! The load-bearing ones are the phantom-event arms: a change that was rolled
//! back must not reach the stream. They are written against the same harnesses
//! the durability work uses (`Recording(Forked)`, a held reader forcing a
//! copy-on-write fork) because those are the shapes where a capture buffer and
//! a commit disagree about what happened.

use super::*;
use crate::datatypes::Value;
use crate::graph::cdc::{self, CdcEnrichment};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::session::Session;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use crate::graph::storage::GraphRead;
use crate::graph::wal::{DurabilityLevel, WAL_FORMAT_VERSION};
use std::collections::HashMap;
use std::sync::Arc;

// ── harness ──────────────────────────────────────────────────────────

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("query failed: {query}: {e}"));
}

fn expect_failure(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    if execute_mut(graph, query, &opts).is_ok() {
        panic!("expected {query} to fail mid-statement");
    }
}

/// A mid-statement failure: the create and the set land, then the duration
/// overflow aborts the statement and rollback undoes them. The shape is
/// borrowed from `rollback_tests::row_undo`.
const FAILS_AFTER_WRITING: &str = "CREATE (x:Item {id: 200, name: 'new'}) \
     WITH x MATCH (a:Item {id: 1}) SET a.name = 'clobbered' \
     WITH a MATCH (m:Item {id: 1}) SET m.qty = duration({months: 2147483648})";

/// The commit boundary for a bare `DirGraph`, which is what an autocommit
/// binding calls after each statement. Named so the tests read as commits.
fn commit(graph: &mut DirGraph) {
    cdc::drain_at_commit(graph);
}

fn events(graph: &DirGraph) -> Vec<CdcEvent> {
    cdc::read(graph, 0, None).expect("capture must be enabled")
}

fn node_id(event: &CdcEvent) -> Value {
    match &event.change {
        CdcChange::Node { id, .. } => id.clone(),
        other => panic!("expected a node event, got {other:?}"),
    }
}

fn node_property(event: &CdcEvent, key: &str) -> Option<Value> {
    match &event.change {
        CdcChange::Node { after, .. } => after.as_ref().and_then(|state| {
            state
                .properties
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.clone())
        }),
        other => panic!("expected a node event, got {other:?}"),
    }
}

/// An enabled graph holding two `Item` nodes and one edge, with the stream
/// already drained so each test starts from an empty log.
fn seeded() -> DirGraph {
    let mut graph = DirGraph::new();
    cdc::enable(&mut graph, None, CdcEnrichment::Off).expect("enable on a plain in-memory graph");
    run(
        &mut graph,
        "CREATE (:Item {id: 1, name: 'one', qty: 10}), (:Item {id: 2, name: 'two', qty: 20})",
    );
    run(
        &mut graph,
        "MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:LINKS {weight: 1}]->(b)",
    );
    commit(&mut graph);
    let handle = graph.cdc_log().expect("enabled").clone();
    handle
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .reconfigure(super::DEFAULT_CAPACITY, CdcEnrichment::Off);
    // Start each test from a known-empty ring without re-minting the epoch:
    // read everything, then assert the tests below only see what they cause.
    let drained = events(&graph).len();
    assert!(drained > 0, "the fixture itself must publish");
    graph
}

/// Cursor position after the fixture, so a test reads only its own events.
fn cursor(graph: &DirGraph) -> u64 {
    cdc::status(graph).expect("enabled").current
}

fn since(graph: &DirGraph, from: u64) -> Vec<CdcEvent> {
    cdc::read(graph, from, None).expect("enabled")
}

// ── capture → event correctness ──────────────────────────────────────

#[test]
fn create_update_and_delete_publish_one_event_each() {
    let mut graph = seeded();
    let from = cursor(&graph);

    run(&mut graph, "CREATE (:Item {id: 3, name: 'three'})");
    commit(&mut graph);
    run(&mut graph, "MATCH (i:Item {id: 3}) SET i.name = 'renamed'");
    commit(&mut graph);
    run(&mut graph, "MATCH (i:Item {id: 3}) DELETE i");
    commit(&mut graph);

    let published = since(&graph, from);
    assert_eq!(
        published
            .iter()
            .map(|event| (event.kind, event.change.element()))
            .collect::<Vec<_>>(),
        vec![
            (CdcEventKind::Create, "node"),
            (CdcEventKind::Update, "node"),
            (CdcEventKind::Delete, "node"),
        ],
        "each committed statement publishes exactly one event for the node it touched"
    );
    assert_eq!(node_id(&published[0]), Value::Int64(3));
    assert_eq!(
        node_property(&published[0], "name"),
        Some(Value::String("three".into())),
        "a create carries the after-state"
    );
    assert_eq!(
        node_property(&published[1], "name"),
        Some(Value::String("renamed".into())),
        "an update carries the state after the write, not before it"
    );
    assert!(
        matches!(
            &published[2].change,
            CdcChange::Node { after: None, id, .. } if *id == Value::Int64(3)
        ),
        "a delete carries identity only — v1 keeps no before-image"
    );
}

#[test]
fn edge_create_update_and_delete_publish_edge_events() {
    let mut graph = seeded();
    let from = cursor(&graph);

    run(
        &mut graph,
        "MATCH (a:Item {id: 2}), (b:Item {id: 1}) CREATE (a)-[:LINKS {weight: 5}]->(b)",
    );
    commit(&mut graph);
    run(
        &mut graph,
        "MATCH (:Item {id: 2})-[r:LINKS]->(:Item {id: 1}) SET r.weight = 9",
    );
    commit(&mut graph);
    run(
        &mut graph,
        "MATCH (:Item {id: 2})-[r:LINKS]->(:Item {id: 1}) DELETE r",
    );
    commit(&mut graph);

    let published = since(&graph, from);
    assert_eq!(
        published
            .iter()
            .map(|event| (event.kind, event.change.element()))
            .collect::<Vec<_>>(),
        vec![
            (CdcEventKind::Create, "edge"),
            (CdcEventKind::Update, "edge"),
            (CdcEventKind::Delete, "edge"),
        ]
    );
    match &published[0].change {
        CdcChange::Edge {
            conn_type,
            src_id,
            tgt_id,
            after,
            ..
        } => {
            assert_eq!(conn_type, "LINKS");
            assert_eq!(*src_id, Value::Int64(2));
            assert_eq!(*tgt_id, Value::Int64(1));
            assert_eq!(
                after.as_ref().map(|state| state.properties.clone()),
                Some(vec![("weight".to_string(), Value::Int64(5))]),
                "an edge event names both endpoints logically and carries its properties"
            );
        }
        other => panic!("expected an edge event, got {other:?}"),
    }
    assert!(matches!(
        &published[2].change,
        CdcChange::Edge { after: None, .. }
    ));
}

/// Pinned semantics: a create followed by writes to the same entity **in one
/// commit** is one `create` carrying the final state, not a create plus an
/// update. A consumer mirroring state must not be told an entity was updated
/// before it was told it exists.
#[test]
fn create_then_set_in_one_statement_collapses_to_one_create() {
    let mut graph = seeded();
    let from = cursor(&graph);

    run(
        &mut graph,
        "CREATE (i:Item {id: 7, name: 'seven'}) SET i.name = 'final', i.qty = 3",
    );
    commit(&mut graph);

    let published = since(&graph, from);
    assert_eq!(
        published.len(),
        1,
        "one entity changed in one commit, so one event: {published:?}"
    );
    assert_eq!(published[0].kind, CdcEventKind::Create);
    assert_eq!(
        node_property(&published[0], "name"),
        Some(Value::String("final".into())),
        "the collapsed event carries the state the commit left behind"
    );
}

#[test]
fn repeated_updates_in_one_commit_collapse_to_one_update() {
    let mut graph = seeded();
    let from = cursor(&graph);

    run(&mut graph, "MATCH (i:Item {id: 1}) SET i.name = 'a'");
    run(&mut graph, "MATCH (i:Item {id: 1}) SET i.name = 'b'");
    commit(&mut graph);

    let published = since(&graph, from);
    assert_eq!(published.len(), 1, "{published:?}");
    assert_eq!(published[0].kind, CdcEventKind::Update);
    assert_eq!(
        node_property(&published[0], "name"),
        Some(Value::String("b".into()))
    );
}

#[test]
fn secondary_labels_publish_an_update_carrying_the_label_set() {
    let mut graph = seeded();
    let from = cursor(&graph);

    run(&mut graph, "MATCH (i:Item {id: 1}) SET i:Featured");
    commit(&mut graph);

    let published = since(&graph, from);
    assert_eq!(published.len(), 1, "{published:?}");
    assert_eq!(published[0].kind, CdcEventKind::Update);
    match &published[0].change {
        CdcChange::Node { after, .. } => assert_eq!(
            after.as_ref().map(|state| state.labels.clone()),
            Some(vec!["Featured".to_string()]),
            "a label change is an update whose after-state names the labels"
        ),
        other => panic!("expected a node event, got {other:?}"),
    }
}

/// An entity created and removed inside one commit publishes the delete only:
/// the upsert resolves against final state and finds nothing.
#[test]
fn create_then_delete_in_one_commit_publishes_only_the_delete() {
    let mut graph = seeded();
    let from = cursor(&graph);

    run(
        &mut graph,
        "CREATE (i:Item {id: 42, name: 'ephemeral'}) WITH i MATCH (d:Item {id: 42}) DELETE d",
    );
    commit(&mut graph);

    let published = since(&graph, from);
    assert_eq!(
        published.iter().map(|event| event.kind).collect::<Vec<_>>(),
        vec![CdcEventKind::Delete],
        "{published:?}"
    );
}

// ── the no-phantom invariant ─────────────────────────────────────────

/// A statement that fails after writing must contribute nothing to the
/// stream. The rollback truncates the ops it buffered, and the drain — which
/// runs only at the commit boundary — never sees them.
#[test]
fn a_rolled_back_statement_publishes_nothing() {
    let mut graph = seeded();
    let from = cursor(&graph);

    expect_failure(&mut graph, FAILS_AFTER_WRITING);
    commit(&mut graph);

    assert_eq!(
        since(&graph, from),
        Vec::new(),
        "a rolled-back statement's writes are gone from the graph; publishing them \
         would tell a consumer about data that never existed"
    );
    // Non-vacuity: the same statement shape, succeeding, does publish.
    run(&mut graph, "CREATE (x:Item {id: 201, name: 'new'})");
    commit(&mut graph);
    assert_eq!(since(&graph, from).len(), 1);
}

/// The same invariant one rung up: a failed statement inside an otherwise
/// good commit contributes nothing, while its neighbours in the same commit
/// still publish.
#[test]
fn a_failed_statement_does_not_poison_the_commit_around_it() {
    let mut graph = seeded();
    let from = cursor(&graph);

    run(&mut graph, "CREATE (:Item {id: 8, name: 'eight'})");
    expect_failure(&mut graph, FAILS_AFTER_WRITING);
    run(&mut graph, "CREATE (:Item {id: 9, name: 'nine'})");
    commit(&mut graph);

    let published = since(&graph, from);
    let ids: Vec<Value> = published.iter().map(node_id).collect();
    assert_eq!(
        ids,
        vec![Value::Int64(8), Value::Int64(9)],
        "only the two committed creates may appear: {published:?}"
    );
}

/// A rolled-back **transaction** publishes nothing: its working copy is
/// dropped with its capture buffer undrained, so no boundary ever publishes
/// it.
#[test]
fn a_rolled_back_transaction_publishes_nothing() {
    let session = Session::new(seeded());
    let from = cursor(&session.snapshot());

    let mut tx = session.begin();
    run(
        tx.working_mut().expect("writable tx"),
        "CREATE (:Item {id: 99, name: 'doomed'})",
    );
    session.rollback(tx);

    let graph = session.snapshot();
    assert_eq!(
        since(&graph, from),
        Vec::new(),
        "a transaction that was never committed must not appear in the stream"
    );
    assert_eq!(
        graph.graph.node_count(),
        2,
        "non-vacuity: the rollback really did discard the write"
    );

    let mut tx = session.begin();
    run(
        tx.working_mut().expect("writable tx"),
        "CREATE (:Item {id: 99, name: 'kept'})",
    );
    session.commit(tx, false);
    assert_eq!(
        since(&session.snapshot(), from).len(),
        1,
        "non-vacuity: the committed twin of that transaction does publish"
    );
}

/// A held reader forces the writer to fork copy-on-write. The fork shares the
/// log, so the commit lands **once**, in the log both handles see.
#[test]
fn a_commit_taken_over_a_held_reader_lands_once_in_the_shared_log() {
    let session = Session::new(seeded());
    let reader = session.snapshot();
    let from = cursor(&reader);

    let mut tx = session.begin();
    {
        let working = tx.working_mut().expect("writable tx");
        run(
            working,
            "MATCH (i:Item {id: 1}) SET i.name = 'written under a reader'",
        );
        assert!(
            working
                .graph
                .recording()
                .is_some_and(|recording| recording.inner().is_forked()),
            "precondition: the write must be taken on a Recording(Forked) backend — \
             the composition this arm exists to cover"
        );
    }
    session.commit(tx, false);

    let writer = session.snapshot();
    assert!(
        !Arc::ptr_eq(&reader, &writer),
        "precondition: the commit must have published a different graph, or the \
         copy-on-write shape under test never happened"
    );
    assert_eq!(
        since(&writer, from).len(),
        1,
        "exactly one event, not one per holder"
    );
    assert_eq!(
        since(&reader, from).len(),
        1,
        "and the reader's handle addresses the same log — a fork shares it"
    );
    assert_eq!(
        cdc::status(&reader).map(|status| status.epoch),
        cdc::status(&writer).map(|status| status.epoch),
        "one epoch across the fork"
    );
}

/// The phantom invariant on the `Recording(Forked)` composition: a statement
/// that fails while a reader holds the graph must publish nothing, even though
/// its writes went through an overlay backend the reader shares a base with.
#[test]
fn a_rolled_back_statement_under_a_held_reader_publishes_nothing() {
    let session = Session::new(seeded());
    let reader = session.snapshot();
    let from = cursor(&reader);

    let mut tx = session.begin();
    {
        let working = tx.working_mut().expect("writable tx");
        run(working, "CREATE (:Item {id: 50, name: 'kept'})");
        assert!(
            working
                .graph
                .recording()
                .is_some_and(|recording| recording.inner().is_forked()),
            "precondition: Recording(Forked)"
        );
        expect_failure(working, FAILS_AFTER_WRITING);
    }
    session.commit(tx, false);

    let published = since(&session.snapshot(), from);
    assert_eq!(
        published.iter().map(node_id).collect::<Vec<_>>(),
        vec![Value::Int64(50)],
        "only the statement that survived may publish: {published:?}"
    );
    assert_eq!(
        since(&reader, from).len(),
        1,
        "and the reader addresses the same log"
    );
}

// ── retention ────────────────────────────────────────────────────────

#[test]
fn eviction_bounds_the_ring_and_advances_the_earliest_watermark() {
    let mut graph = DirGraph::new();
    cdc::enable(&mut graph, Some(4), CdcEnrichment::Off).expect("enable");

    for id in 0..40 {
        run(&mut graph, &format!("CREATE (:Item {{id: {id}}})"));
        commit(&mut graph);
        let status = cdc::status(&graph).expect("enabled");
        assert!(
            status.buffered <= 4,
            "the ring must stay within its capacity at every step, saw {status:?}"
        );
    }

    let status = cdc::status(&graph).expect("enabled");
    assert_eq!(status.buffered, 4);
    assert_eq!(status.current, 40, "every commit still consumed a sequence");
    assert_eq!(
        status.earliest, 37,
        "the earliest readable position advances as events are evicted"
    );
    let retained = events(&graph);
    assert_eq!(retained.first().map(|event| event.seq), Some(37));
    assert_eq!(
        cdc::read(&graph, 0, None).map(|read| read.len()),
        Some(4),
        "a cursor older than the watermark reads what survives; B2 turns that \
         into a typed 'cursor too old' refusal"
    );
}

#[test]
fn a_single_commit_larger_than_the_ring_is_still_bounded() {
    let mut graph = DirGraph::new();
    cdc::enable(&mut graph, Some(8), CdcEnrichment::Off).expect("enable");

    let creates = (0..500)
        .map(|id| format!("(:Item {{id: {id}}})"))
        .collect::<Vec<_>>()
        .join(", ");
    run(&mut graph, &format!("CREATE {creates}"));
    commit(&mut graph);

    let status = cdc::status(&graph).expect("enabled");
    assert_eq!(status.buffered, 8, "{status:?}");
    assert_eq!(status.current, 500);
    assert_eq!(status.earliest, 493);
}

#[test]
fn re_enabling_resizes_in_place_and_keeps_the_epoch() {
    let mut graph = DirGraph::new();
    let first = cdc::enable(&mut graph, Some(100), CdcEnrichment::Off).expect("enable");
    for id in 0..10 {
        run(&mut graph, &format!("CREATE (:Item {{id: {id}}})"));
    }
    commit(&mut graph);

    let resized = cdc::enable(&mut graph, Some(3), CdcEnrichment::Off).expect("re-enable");
    assert_eq!(
        resized.epoch, first.epoch,
        "a live consumer's cursors must survive a capacity change"
    );
    assert_eq!(resized.capacity, 3);
    assert_eq!(resized.buffered, 3, "the shrink evicts from the front");
    assert_eq!(resized.earliest, 8);
}

// ── lifecycle ────────────────────────────────────────────────────────

#[test]
fn enable_wraps_a_plain_graph_without_claiming_the_write_ahead_log() {
    let mut graph = DirGraph::new();
    assert!(!graph.graph.is_recording());

    let status =
        cdc::enable(&mut graph, None, CdcEnrichment::Off).expect("enable on a plain graph");
    assert_eq!(status.capacity, super::DEFAULT_CAPACITY);
    assert_eq!(status.current, 0);
    assert!(
        graph.graph.is_recording(),
        "enable installs the capture seam the events are derived from"
    );
    assert!(
        !graph.graph.is_wal_owner(),
        "…without presenting itself as a durable owner, or a later durable open \
         would be refused and the duplicate-id rule would silently tighten"
    );
}

/// The duplicate-id refusal belongs to the write-ahead log, not to the capture
/// wrapper — enabling CDC must not change what a write is allowed to do.
#[test]
fn enable_does_not_impose_the_durable_duplicate_id_refusal() {
    let mut graph = DirGraph::new();
    cdc::enable(&mut graph, None, CdcEnrichment::Off).expect("enable");
    run(&mut graph, "CREATE (:Item {id: 1, name: 'first'})");
    run(&mut graph, "CREATE (:Item {id: 1, name: 'second'})");
    commit(&mut graph);
    assert_eq!(graph.graph.node_count(), 2);
}

#[test]
fn disable_drops_the_log_and_stops_publishing() {
    let mut graph = seeded();
    assert!(cdc::disable(&mut graph), "was enabled");
    assert!(!cdc::disable(&mut graph), "already off");
    assert!(cdc::status(&graph).is_none());
    assert!(cdc::read(&graph, 0, None).is_none());

    assert!(
        !graph.graph.is_recording(),
        "disable takes the capture layer off a graph no write-ahead log owns — \
         leaving it would keep charging the write path for a stream nobody reads"
    );

    run(&mut graph, "CREATE (:Item {id: 5})");
    commit(&mut graph);
    assert!(cdc::status(&graph).is_none());

    let restarted = cdc::enable(&mut graph, None, CdcEnrichment::Off).expect("re-enable");
    assert_eq!(
        restarted.current, 0,
        "a restart is a new epoch and a new sequence: nothing from before it is \
         addressable"
    );
}

/// The other half of the disable rule: a durable graph keeps its wrapper,
/// because that wrapper is the write-ahead log's seam.
#[test]
fn disable_leaves_a_durable_graphs_capture_layer_alone() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("graph.kgl");
    let session = Session::open_durable(
        Arc::new(DirGraph::new()),
        &path.to_string_lossy(),
        DurabilityLevel::Full,
    )
    .expect("durable open");

    let mut tx = session.begin();
    {
        let working = tx.working_mut().expect("writable tx");
        cdc::enable(working, None, CdcEnrichment::Off).expect("enable");
        assert!(cdc::disable(working), "was enabled");
        assert!(
            working.graph.is_wal_owner(),
            "disabling the stream must not disarm the log"
        );
    }
    run(
        tx.working_mut().expect("writable tx"),
        "CREATE (:Item {id: 1})",
    );
    session.commit(tx, false);

    let graph = session.snapshot();
    assert!(graph.graph.is_wal_owner());
    assert!(cdc::status(&graph).is_none());
    let wal = crate::graph::wal::wal_path(&path);
    assert!(
        std::fs::metadata(&wal).map(|meta| meta.len()).unwrap_or(0) > 0,
        "and the commit after it must still reach the log"
    );
}

#[test]
fn disk_mode_refuses_enable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut graph = new_dir_graph_in_mode(StorageMode::Disk, Some(dir.path())).expect("disk graph");
    let error = cdc::enable(&mut graph, None, CdcEnrichment::Off).expect_err("disk must refuse");
    let message = error.to_string();
    assert!(
        message.contains("storage='disk'") && message.contains("generation"),
        "the refusal must name the mode and why its change boundary is different: {message}"
    );
    assert!(!graph.cdc_enabled());
}

/// Mapped mode serves, and its **columnar** property write publishes too.
/// That write reaches the capture seam through `note_recorded_node_upsert`
/// rather than through a recorded `GraphWrite` borrow, so it is its own arm:
/// a stream that silently dropped every mapped `SET` would otherwise look
/// healthy.
#[test]
fn mapped_mode_serves_including_the_columnar_write_path() {
    let mut graph = new_dir_graph_in_mode(StorageMode::Mapped, None).expect("mapped graph");
    cdc::enable(&mut graph, None, CdcEnrichment::Off).expect("mapped must serve");

    run(&mut graph, "CREATE (:Item {id: 1, name: 'one'})");
    commit(&mut graph);
    run(&mut graph, "MATCH (i:Item {id: 1}) SET i.name = 'renamed'");
    commit(&mut graph);
    run(&mut graph, "MATCH (i:Item {id: 1}) DELETE i");
    commit(&mut graph);

    let published = events(&graph);
    assert_eq!(
        published.iter().map(|event| event.kind).collect::<Vec<_>>(),
        vec![
            CdcEventKind::Create,
            CdcEventKind::Update,
            CdcEventKind::Delete
        ],
        "{published:?}"
    );
    assert_eq!(
        node_property(&published[1], "name"),
        Some(Value::String("renamed".into()))
    );
}

/// The bulk loader is the other write path into the capture seam, and it
/// creates nodes through `GraphWrite::add_node` like Cypher does — so a load
/// publishes creates, one per row, not updates.
#[test]
fn a_bulk_load_publishes_one_create_per_row() {
    use crate::datatypes::DataFrame;
    use crate::graph::mutation::maintain::add_nodes;

    let mut graph = DirGraph::new();
    cdc::enable(&mut graph, None, CdcEnrichment::Off).expect("enable");

    let columns = vec!["id".to_string(), "name".to_string()];
    let rows: Vec<Vec<Value>> = (0..25)
        .map(|i| vec![Value::Int64(i), Value::String(format!("row-{i}"))])
        .collect();
    let frame = DataFrame::from_cypher_rows(columns, rows).expect("dataframe");
    add_nodes(
        &mut graph,
        frame,
        "Person".to_string(),
        "id".to_string(),
        Some("name".to_string()),
        None,
    )
    .expect("bulk load");
    commit(&mut graph);

    let published = events(&graph);
    assert_eq!(published.len(), 25, "one event per row, not per write");
    assert!(
        published
            .iter()
            .all(|event| event.kind == CdcEventKind::Create),
        "a loaded row is a created node: {published:?}"
    );
    assert_eq!(node_id(&published[0]), Value::Int64(0));
}

#[test]
fn capacity_zero_and_oversize_are_refused() {
    let mut graph = DirGraph::new();
    assert!(cdc::enable(&mut graph, Some(0), CdcEnrichment::Off).is_err());
    assert!(cdc::enable(
        &mut graph,
        Some(super::MAX_CAPACITY + 1),
        CdcEnrichment::Off
    )
    .is_err());
    assert!(!graph.cdc_enabled(), "a refused enable installs nothing");
    assert!(cdc::enable(&mut graph, Some(super::MAX_CAPACITY), CdcEnrichment::Off).is_ok());
}

#[test]
fn independent_copy_re_mints_the_epoch_and_starts_empty() {
    let graph = seeded();
    let copy = graph.independent_copy();

    let original = cdc::status(&graph).expect("enabled");
    let copied = cdc::status(&copy).expect("an independent copy keeps capture on");
    assert_ne!(
        copied.epoch, original.epoch,
        "an independent lineage needs an identity of its own, like graph_id — a \
         cursor from the original must be refusable, not resolvable here"
    );
    assert_eq!(copied.buffered, 0, "the copy has published nothing");
    assert_eq!(copied.current, 0);
    assert_eq!(copied.capacity, original.capacity);
}

#[test]
fn a_clone_shares_the_log_but_an_independent_copy_does_not() {
    let mut graph = seeded();
    let from = cursor(&graph);
    let mut copy = graph.independent_copy();

    run(&mut copy, "CREATE (:Item {id: 300})");
    commit(&mut copy);

    assert_eq!(
        since(&graph, from),
        Vec::new(),
        "a write to the copy must not appear in the original's stream"
    );
    assert_eq!(since(&copy, 0).len(), 1);

    let clone = graph.clone();
    run(&mut graph, "CREATE (:Item {id: 301})");
    commit(&mut graph);
    assert_eq!(
        since(&clone, from).len(),
        1,
        "a plain clone shares the log — that is what makes a copy-on-write commit \
         visible exactly once"
    );
}

// ── durability composition ───────────────────────────────────────────

/// The marker's reason for existing: capture installed for CDC must not read
/// as a durable owner, so a durable open after `enable` still works.
#[test]
fn a_cdc_enabled_graph_can_still_be_opened_durably() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("graph.kgl");
    let mut graph = DirGraph::new();
    cdc::enable(&mut graph, None, CdcEnrichment::Off).expect("enable");

    let session = Session::open_durable(
        Arc::new(graph),
        &path.to_string_lossy(),
        DurabilityLevel::Full,
    )
    .expect("a CDC-enabled graph is not a second durable owner");
    assert_eq!(session.durability(), Some(DurabilityLevel::Full));
    assert!(session.snapshot().graph.is_wal_owner());
}

/// A durable session publishes the same commit to both consumers: one WAL
/// frame, one set of CDC events, derived from one drained buffer.
#[test]
fn a_durable_commit_publishes_to_the_log_and_the_stream() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("graph.kgl");
    let session = Session::open_durable(
        Arc::new(DirGraph::new()),
        &path.to_string_lossy(),
        DurabilityLevel::Full,
    )
    .expect("durable open");

    // Enabling on the working copy is exactly how B2's `CALL db.cdc.enable`
    // will run: a mutation statement against a transaction's graph.
    let mut tx = session.begin();
    cdc::enable(
        tx.working_mut().expect("writable tx"),
        Some(16),
        CdcEnrichment::Off,
    )
    .expect("enable");
    run(
        tx.working_mut().expect("writable tx"),
        "CREATE (:Item {id: 1, name: 'durable'})",
    );
    session.commit(tx, false);

    let graph = session.snapshot();
    assert_eq!(
        events(&graph)
            .iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>(),
        vec![CdcEventKind::Create],
        "the durable drain must publish, not consume, the capture buffer"
    );
    let wal = crate::graph::wal::wal_path(&path);
    let size = std::fs::metadata(&wal).map(|meta| meta.len()).unwrap_or(0);
    assert!(
        size > 0,
        "and the write-ahead log must still hold the frame for that commit"
    );
}

/// A durable commit whose write-ahead append fails is not a commit: the
/// `Arc` swap is skipped and the caller is told so. The stream must agree —
/// publishing before the append would put a change nobody made into it.
///
/// Mutation-checked: moving `publish_drained` back above the append in
/// `log_working_commit` turns this red.
#[test]
fn a_durable_commit_that_could_not_be_logged_publishes_nothing() {
    use crate::graph::session::CommitOutcome;

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("graph.kgl");
    let session = Session::open_durable(
        Arc::new(DirGraph::new()),
        &path.to_string_lossy(),
        DurabilityLevel::Full,
    )
    .expect("durable open");

    let mut tx = session.begin();
    cdc::enable(
        tx.working_mut().expect("writable tx"),
        None,
        CdcEnrichment::Off,
    )
    .expect("enable");
    session.commit(tx, false);
    let from = cursor(&session.snapshot());

    session.set_fail_append(true);
    let mut tx = session.begin();
    run(
        tx.working_mut().expect("writable tx"),
        "CREATE (:Item {id: 1, name: 'unlogged'})",
    );
    assert!(matches!(
        session.commit(tx, false),
        CommitOutcome::DurabilityFailed { .. }
    ));
    assert_eq!(
        since(&session.snapshot(), from),
        Vec::new(),
        "a commit the caller was told failed must not appear in the stream"
    );

    // Non-vacuity: with the fault cleared, the same write commits and publishes.
    session.set_fail_append(false);
    let mut tx = session.begin();
    run(
        tx.working_mut().expect("writable tx"),
        "CREATE (:Item {id: 1, name: 'logged'})",
    );
    assert!(matches!(
        session.commit(tx, false),
        CommitOutcome::Committed { .. }
    ));
    assert_eq!(since(&session.snapshot(), from).len(), 1);
}

/// CDC changes no on-disk format. The WAL's create/update distinction lives in
/// the in-memory capture buffer precisely so this constant does not move.
#[test]
fn cdc_does_not_move_the_wal_format_version() {
    assert_eq!(
        WAL_FORMAT_VERSION, 3,
        "CDC derives create-vs-update from an in-memory capture marker; if this \
         constant moved, a `MutationOp` gained a field it did not need"
    );
}
