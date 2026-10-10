//! `Session::execute_auto_commit` running in place on an unshared graph.
//!
//! Two promises are pinned here. A statement that runs in place and fails for
//! any reason leaves the graph exactly as it was (a full fingerprint, not a
//! count). And a graph anyone else can see is never written: a snapshot, a
//! cursor or a backup image holds an `Arc` clone, so the statement forks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::execute::{execute_mut_held, ExecuteOutcome};
use super::{execute_read, CancelToken, ExecuteOptions, Session};
use crate::datatypes::Value;
use crate::error::{KgError, KgErrorCode};
use crate::graph::dir_graph::rollback::StatementCheckpoint;
use crate::graph::dir_graph::rollback_tests::digest;
use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use crate::graph::wal::DurabilityLevel;

fn write(session: &Session, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::new(&params);
    session
        .execute_auto_commit(query, &opts, 1)
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn count_of(graph: &DirGraph, query: &str) -> i64 {
    let params = HashMap::new();
    let opts = ExecuteOptions::new(&params);
    match execute_read(graph, query, &opts).unwrap().result.rows[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("not a count: {other:?}"),
    }
}

fn nodes(graph: &DirGraph) -> i64 {
    count_of(graph, "MATCH (n) RETURN count(n)")
}

fn digest_of(session: &Session) -> String {
    let mut graph = session.snapshot().try_clone().unwrap();
    digest(&mut graph)
}

/// A graph with an index, a uniqueness constraint, a type with edges and some
/// deleted slots, so a rollback has indexes, free lists and columns to restore.
fn seeded(mode: StorageMode) -> Session {
    let dir = tempfile::tempdir().unwrap();
    let graph = new_dir_graph_in_mode(mode, Some(dir.path())).unwrap();
    let session = Session::new(graph);
    // Schema statements fork by design; the data below runs in place.
    write(&session, "CREATE INDEX FOR (n:Item) ON (n.tag)");
    write(
        &session,
        "CREATE CONSTRAINT FOR (n:Item) REQUIRE n.code IS UNIQUE",
    );
    write(
        &session,
        "UNWIND range(1, 30) AS i CREATE (:Item {id: i, code: i * 10, tag: 't' + toString(i % 4), n: i})",
    );
    write(
        &session,
        "MATCH (a:Item), (b:Item) WHERE b.id = a.id + 1 CREATE (a)-[:NEXT {w: a.id}]->(b)",
    );
    write(
        &session,
        "MATCH (n:Item) WHERE n.id IN [3, 9, 17] DETACH DELETE n",
    );
    std::mem::forget(dir);
    session
}

/// The statement's own runner, then a pause with the lock held and the writes
/// already in the published graph.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
fn run_then_linger(
    graph: &mut DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
    held: &mut StatementCheckpoint,
    keep_undo: bool,
) -> Result<ExecuteOutcome, KgError> {
    let outcome = execute_mut_held(graph, query, opts, held, keep_undo);
    MID_STATEMENT.with(|signal| signal.borrow().as_ref().unwrap().send(()).unwrap());
    std::thread::sleep(std::time::Duration::from_millis(300));
    outcome
}

/// The statement's own runner, then a panic with its writes in place.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
fn run_then_panic(
    graph: &mut DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
    held: &mut StatementCheckpoint,
    keep_undo: bool,
) -> Result<ExecuteOutcome, KgError> {
    execute_mut_held(graph, query, opts, held, keep_undo).unwrap();
    panic!("injected panic after the statement's writes");
}

thread_local! {
    /// Where `run_then_linger` announces that the statement's writes are in.
    static MID_STATEMENT: std::cell::RefCell<Option<std::sync::mpsc::Sender<()>>> =
        const { std::cell::RefCell::new(None) };
}

const IN_MEMORY_MODES: [StorageMode; 2] = [StorageMode::Memory, StorageMode::Mapped];

#[test]
fn an_unshared_graph_is_written_in_place_and_a_shared_one_forks() {
    let session = Session::new(DirGraph::new());
    write(&session, "CREATE (:A {id: 1})");
    assert_eq!(
        (session.in_place_commit_count(), session.shared_fork_count()),
        (1, 0)
    );

    let reader = session.snapshot();
    let before = nodes(&reader);
    write(&session, "CREATE (:A {id: 2})");
    assert_eq!(
        (session.in_place_commit_count(), session.shared_fork_count()),
        (1, 1),
        "a held snapshot makes the write fork"
    );
    assert_eq!(nodes(&reader), before, "the reader never sees the write");
    assert_eq!(nodes(&session.snapshot()), before + 1);

    drop(reader);
    write(&session, "CREATE (:A {id: 3})");
    assert_eq!(session.in_place_commit_count(), 2, "unshared again");
}

#[test]
fn a_snapshot_taken_before_an_in_place_write_never_sees_it() {
    for query in [
        "CREATE (:A {id: 9})",
        "MATCH (n:Item) WHERE n.id = 1 SET n.n = 999",
        "MATCH (a:Item {id: 1})-[r:NEXT]->() DELETE r",
        "MATCH (n:Item {id: 1}) DETACH DELETE n",
        "MATCH (a:Item {id: 1}), (b:Item {id: 5}) CREATE (a)-[:NEXT {w: 0}]->(b)",
    ] {
        let session = seeded(StorageMode::Memory);
        let before = digest_of(&session);
        let reader = session.snapshot();
        write(&session, query);
        let mut after = reader.try_clone().unwrap();
        assert_eq!(digest(&mut after), before, "{query}");
        assert_ne!(digest_of(&session), before, "{query} wrote nothing");
    }
}

#[test]
fn a_cursor_and_an_open_transaction_make_the_write_fork() {
    let session = Session::new(DirGraph::new());
    write(&session, "UNWIND range(1, 5) AS i CREATE (:A {id: i})");
    let base = session.in_place_commit_count();

    let tx = session.begin();
    write(&session, "CREATE (:A {id: 100})");
    assert_eq!(
        session.in_place_commit_count(),
        base,
        "open tx holds the Arc"
    );
    session.rollback(tx);

    write(&session, "UNWIND range(1, 5000) AS i CREATE (:B {id: i})");
    let base = session.in_place_commit_count();
    let params = HashMap::new();
    let opts = ExecuteOptions::new(&params);
    // A streamed cursor: its worker blocks on a two-slot channel holding the
    // snapshot. (A materialised one copies its rows and drops the snapshot.)
    let mut cursor = session
        .execute_read_cursor("MATCH (n:B) RETURN n.id AS id", &opts)
        .unwrap();
    assert!(cursor.streamed());
    let mut seen = cursor.next_batch(2).unwrap().len();
    write(&session, "MATCH (n:B) DETACH DELETE n");
    assert_eq!(
        session.in_place_commit_count(),
        base,
        "the cursor's snapshot holds the Arc"
    );
    loop {
        let batch = cursor.next_batch(500).unwrap();
        if batch.is_empty() {
            break;
        }
        seen += batch.len();
    }
    assert_eq!(seen, 5000, "the cursor still streams the pre-write graph");
}

/// Each failure arrives after the statement's first write, so the in-place
/// graph is dirty when it is refused.
fn assert_rolled_back(
    session: &Session,
    query: &str,
    opts: &ExecuteOptions<'_>,
    code: KgErrorCode,
) {
    let before = digest_of(session);
    let version = session.version();
    let in_place = session.in_place_commit_count();
    let err = session
        .execute_auto_commit(query, opts, 1)
        .err()
        .unwrap_or_else(|| panic!("{query} should fail"));
    assert_eq!(err.code(), code, "{query}: {err}");
    assert_eq!(digest_of(session), before, "{query}");
    assert_eq!(session.version(), version, "{query}");
    assert_eq!(
        session.in_place_commit_count(),
        in_place,
        "a refused statement is not a commit"
    );
}

#[test]
fn a_failed_in_place_statement_restores_the_exact_graph() {
    for mode in IN_MEMORY_MODES {
        let session = seeded(mode);
        let params = HashMap::new();
        let opts = ExecuteOptions::new(&params);
        // Expression error after the first row's writes.
        assert_rolled_back(
            &session,
            "UNWIND [1, 2] AS i CREATE (:Item {id: 100 + i, tag: 'x'})-[:NEXT]->\
             (:Item {id: 200 + i, tag: duration({months: 2147483648})})",
            &opts,
            KgErrorCode::CypherExecution,
        );
        // Uniqueness refusal on the second row, after the first row wrote.
        assert_rolled_back(
            &session,
            "UNWIND [300, 1] AS i CREATE (:Item {id: 3000 + i, code: i * 10, tag: 'u'})",
            &opts,
            KgErrorCode::ConstraintViolation,
        );
        // A set + delete + edge mix that dies on its last clause.
        assert_rolled_back(
            &session,
            "MATCH (n:Item) WHERE n.id < 6 SET n.n = -1 \
             WITH n MATCH (n)-[r:NEXT]->() DELETE r \
             WITH n CREATE (:Item {id: 77, code: 10})",
            &opts,
            KgErrorCode::ConstraintViolation,
        );
        write(&session, "CREATE (:Item {id: 400, tag: 'ok'})");
        assert!(session.in_place_commit_count() > 0, "{mode:?} ran in place");
    }
}

#[test]
fn an_expired_deadline_and_a_cancellation_roll_back() {
    let session = seeded(StorageMode::Memory);
    let params = HashMap::new();
    let mut opts = ExecuteOptions::new(&params);
    opts.deadline = Some(std::time::Instant::now());
    assert_rolled_back(
        &session,
        "UNWIND range(1, 20000) AS i CREATE (:Item {id: 1000 + i, tag: 'd'})",
        &opts,
        KgErrorCode::CypherTimeout,
    );

    // Cancel while the statement is writing.
    let token = CancelToken::new();
    let mut opts = ExecuteOptions::new(&params);
    opts.cancel = Some(token.clone());
    let before = digest_of(&session);
    let canceller = {
        let token = token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            token.cancel();
        })
    };
    let err = session
        .execute_auto_commit(
            "UNWIND range(1, 3000000) AS i CREATE (:Item {id: 5000 + i, tag: 'c'})",
            &opts,
            1,
        )
        .err()
        .expect("cancelled mid-statement");
    canceller.join().unwrap();
    assert_eq!(err.code(), KgErrorCode::Cancelled, "{err}");
    assert_eq!(digest_of(&session), before);
}

#[test]
fn a_refused_log_append_restores_the_graph_in_place() {
    for query in [
        "CREATE (:Item {id: 700, tag: 'w'})",
        "UNWIND range(1, 10) AS i CREATE (:Item {id: 800 + i, tag: 'w'})",
        "MATCH (n:Item) WHERE n.id < 5 SET n.n = 0",
        "MATCH (n:Item {id: 2}) DETACH DELETE n",
        "MATCH (a:Item {id: 1}), (b:Item {id: 5}) CREATE (a)-[:NEXT {w: 0}]->(b)",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = Session::open_durable(
            Arc::new(DirGraph::new()),
            &path.to_string_lossy(),
            DurabilityLevel::Normal,
        )
        .unwrap();
        write(&session, "CREATE INDEX FOR (n:Item) ON (n.tag)");
        write(
            &session,
            "UNWIND range(1, 12) AS i CREATE (:Item {id: i, tag: 't' + toString(i % 3), n: i})",
        );
        write(
            &session,
            "MATCH (a:Item), (b:Item) WHERE b.id = a.id + 1 CREATE (a)-[:NEXT {w: a.id}]->(b)",
        );
        let before = digest_of(&session);
        let version = session.version();
        let in_place = session.in_place_commit_count();
        assert!(in_place >= 2, "durable sessions at normal run in place");

        session.set_fail_append(true);
        let params = HashMap::new();
        let opts = ExecuteOptions::new(&params);
        let err = session
            .execute_auto_commit(query, &opts, 1)
            .err()
            .expect("the log refuses");
        assert_eq!(err.code(), KgErrorCode::DurabilityFailed, "{query}");
        assert_eq!(digest_of(&session), before, "{query}");
        assert_eq!(session.version(), version, "{query}");
        assert_eq!(session.in_place_commit_count(), in_place, "{query}");

        // The log is healthy again: the same statement now lands.
        session.set_fail_append(false);
        write(&session, query);
        assert_eq!(session.version(), version + 1, "{query}");
    }
}

#[test]
fn an_in_place_durable_statement_survives_a_crash_and_a_refused_one_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");
    let p = path.to_string_lossy().into_owned();
    let session =
        Session::open_durable(Arc::new(DirGraph::new()), &p, DurabilityLevel::Normal).unwrap();
    write(&session, "CREATE (:N {id: 1})");
    write(&session, "CREATE (:N {id: 2})-[:R]->(:N {id: 3})");
    assert_eq!(session.in_place_commit_count(), 2);
    session.set_fail_append(true);
    let params = HashMap::new();
    let opts = ExecuteOptions::new(&params);
    session
        .execute_auto_commit("CREATE (:N {id: 99})", &opts, 1)
        .err()
        .expect("refused");
    session.set_fail_append(false);
    write(&session, "MATCH (n:N {id: 1}) SET n.seen = true");
    drop(session); // no checkpoint: the process is gone

    let recovered = Session::open_durable(Arc::new(DirGraph::new()), &p, DurabilityLevel::Full)
        .expect("recover from the log");
    let graph = recovered.snapshot();
    assert_eq!(count_of(&graph, "MATCH (n:N) RETURN count(n)"), 3);
    assert_eq!(count_of(&graph, "MATCH (n:N {id: 99}) RETURN count(n)"), 0);
    assert_eq!(count_of(&graph, "MATCH ()-[r:R]->() RETURN count(r)"), 1);
    assert_eq!(
        count_of(
            &graph,
            "MATCH (n:N {id: 1}) WHERE n.seen = true RETURN count(n)"
        ),
        1
    );
}

#[test]
fn a_panic_during_an_in_place_statement_rolls_the_graph_back() {
    let session = seeded(StorageMode::Memory);
    let before = digest_of(&session);
    let params = HashMap::new();
    let opts = ExecuteOptions::new(&params);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        session.in_place_with(
            "UNWIND range(1, 5) AS i CREATE (:Item {id: 900 + i, tag: 'p'})",
            &opts,
            run_then_panic,
        )
    }));
    assert!(caught.is_err());
    assert_eq!(
        digest_of(&session),
        before,
        "nothing of the statement stays"
    );
    // The poisoned lock is recovered, and the session keeps working in place.
    write(&session, "CREATE (:Item {id: 1000, tag: 'after'})");
    assert_eq!(
        count_of(
            &session.snapshot(),
            "MATCH (n:Item {id: 1000}) RETURN count(n)"
        ),
        1
    );
}

#[test]
fn statements_that_are_not_plain_data_writes_fork() {
    let session = seeded(StorageMode::Memory);
    let in_place = session.in_place_commit_count();
    write(&session, "CREATE INDEX FOR (n:Item) ON (n.n)");
    assert_eq!(session.in_place_commit_count(), in_place);
    write(&session, "CREATE (:Z {c: 1})");
    assert_eq!(session.in_place_commit_count(), in_place + 1);
}

#[test]
fn a_disk_graph_forks() {
    let dir = tempfile::tempdir().unwrap();
    let graph = new_dir_graph_in_mode(StorageMode::Disk, Some(dir.path())).unwrap();
    let session = Session::new(graph);
    write(&session, "CREATE (:A {id: 1})");
    write(&session, "MATCH (a:A) SET a.v = 2");
    assert_eq!(session.in_place_commit_count(), 0);
    assert_eq!(nodes(&session.snapshot()), 1);
}

#[test]
fn a_reader_arriving_mid_statement_waits_for_it_and_sees_all_of_it() {
    let session = Arc::new(Session::new(DirGraph::new()));
    let (mid_tx, mid_rx) = std::sync::mpsc::channel();
    let writer = {
        let session = Arc::clone(&session);
        std::thread::spawn(move || {
            MID_STATEMENT.with(|signal| *signal.borrow_mut() = Some(mid_tx));
            let params = HashMap::new();
            let opts = ExecuteOptions::new(&params);
            let outcome = session.in_place_with(
                "UNWIND range(1, 100) AS i CREATE (:Bulk {id: i})",
                &opts,
                run_then_linger,
            );
            assert!(outcome.expect("ran in place").is_ok());
            Instant::now()
        })
    };
    mid_rx.recv().unwrap();
    let reader = {
        let session = Arc::clone(&session);
        std::thread::spawn(move || {
            let seen = count_of(&session.snapshot(), "MATCH (n:Bulk) RETURN count(n)");
            (seen, Instant::now())
        })
    };
    writer.join().unwrap();
    let (seen, _) = reader.join().unwrap();
    // The reader started mid-statement, so seeing all 100 rows proves it waited
    // for the statement to finish. A timestamp comparison is no proof: the
    // writer can only stamp its finish after releasing the lock, and the reader
    // may take the lock and stamp first.
    assert_eq!(seen, 100, "the reader saw the whole statement, not part");
}

#[test]
fn concurrent_readers_see_all_or_nothing_of_each_write() {
    const ROWS: i64 = 60_000;
    let session = Arc::new(Session::new(DirGraph::new()));
    let done = Arc::new(AtomicBool::new(false));
    let writer = {
        let session = Arc::clone(&session);
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            write(
                &session,
                &format!("UNWIND range(1, {ROWS}) AS i CREATE (:Bulk {{id: i}})"),
            );
            done.store(true, Ordering::SeqCst);
        })
    };
    let mut seen = Vec::new();
    while !done.load(Ordering::SeqCst) {
        seen.push(count_of(
            &session.snapshot(),
            "MATCH (n:Bulk) RETURN count(n)",
        ));
    }
    writer.join().unwrap();
    seen.push(count_of(
        &session.snapshot(),
        "MATCH (n:Bulk) RETURN count(n)",
    ));
    assert!(
        seen.iter().all(|&n| n == 0 || n == ROWS),
        "a reader saw a partial statement: {:?}",
        seen.iter()
            .filter(|&&n| n != 0 && n != ROWS)
            .collect::<Vec<_>>()
    );
    assert_eq!(*seen.last().unwrap(), ROWS);
    // Whichever path the race picked, the write ran exactly once.
    assert_eq!(
        session.in_place_commit_count() + session.shared_fork_count(),
        1
    );
}

#[test]
fn the_statement_parks_its_checkpoint_instead_of_closing_it() {
    let mut graph = DirGraph::new();
    let params = HashMap::new();
    let opts = ExecuteOptions::new(&params);
    let mut held = StatementCheckpoint::None;
    execute_mut_held(&mut graph, "CREATE (:A {id: 1})", &opts, &mut held, true).unwrap();
    assert!(!matches!(held, StatementCheckpoint::None));
    std::mem::replace(&mut held, StatementCheckpoint::None).rollback(&mut graph);
    assert_eq!(nodes(&graph), 0);
}
