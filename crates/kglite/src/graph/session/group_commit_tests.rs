//! Concurrent durable auto-commit statements at `full` share log barriers.
//!
//! The first writer is parked inside its barrier ([`ParkHook`]) so a known
//! number of others queue behind it; releasing it lets them run as one batch.
//! Barrier counts come from the log itself, so "fewer barriers than commits"
//! is observed, not inferred from timing.

use super::execute::{execute_read, ExecuteOptions};
use super::group_commit::MAX_BATCH;
use super::Session;
use crate::datatypes::Value;
use crate::error::KgError;
use crate::graph::cdc::{self, CdcChange, CdcEnrichment};
use crate::graph::dir_graph::DirGraph;
use crate::graph::ontology::ontology_from_json;
use crate::graph::wal::{AppendFault, DurabilityLevel, ParkHook};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(10);

/// Releases a parked commit when dropped, so a failing assertion cannot leave
/// the writer blocked forever.
struct ReleaseOnDrop(Arc<ParkHook>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

type Landed = Result<(), Box<KgError>>;

fn open(path: &Path, cdc_on: bool, ontology: Option<&str>) -> Session {
    let p = path.to_string_lossy().into_owned();
    let graph = if path.exists() {
        crate::graph::io::file::load_file(&p).unwrap()
    } else {
        let mut graph = DirGraph::new();
        if cdc_on {
            cdc::enable(&mut graph, Some(4096), CdcEnrichment::Off).unwrap();
        }
        if let Some(json) = ontology {
            graph
                .define_ontology(ontology_from_json(json).unwrap())
                .expect("declaration accepted");
        }
        Arc::new(graph)
    };
    Session::open_durable(graph, &p, DurabilityLevel::Full).unwrap()
}

fn write(session: &Session, query: &str) -> Landed {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    session
        .execute_auto_commit(query, &opts, 1)
        .map(|_| ())
        .map_err(Box::new)
}

fn spawn_write(session: &Arc<Session>, query: String) -> JoinHandle<Landed> {
    let session = Arc::clone(session);
    thread::spawn(move || write(&session, &query))
}

fn column(graph: &DirGraph, query: &str) -> Vec<i64> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let mut values: Vec<i64> = execute_read(graph, query, &opts)
        .unwrap()
        .result
        .rows
        .iter()
        .map(|row| match &row[0] {
            Value::Int64(v) => *v,
            other => panic!("unexpected value {other:?}"),
        })
        .collect();
    values.sort_unstable();
    values
}

fn ids(graph: &DirGraph) -> Vec<i64> {
    column(graph, "MATCH (n:N) RETURN n.id AS id")
}

/// Wait until `count` writers are parked behind the running batch.
fn queued(session: &Session, count: usize) {
    let deadline = Instant::now() + WAIT;
    while session.queued_writers() < count {
        assert!(
            Instant::now() < deadline,
            "only {} of {count} writers queued",
            session.queued_writers()
        );
        thread::sleep(Duration::from_millis(2));
    }
}

/// Park one writer in its barrier, queue `queries` behind it, release, and
/// return every writer's result (the parked one first).
fn parked_batch(session: &Arc<Session>, first: &str, queries: Vec<String>) -> Vec<Landed> {
    let hook = session.park_next_commit();
    let _release = ReleaseOnDrop(Arc::clone(&hook));
    let leader = spawn_write(session, first.to_string());
    assert!(hook.wait_entered(WAIT), "the first write never parked");
    let followers: Vec<_> = queries
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let handle = spawn_write(session, q.clone());
            queued(session, i + 1);
            handle
        })
        .collect();
    hook.release();
    let mut results = vec![leader.join().unwrap()];
    results.extend(followers.into_iter().map(|h| h.join().unwrap()));
    results
}

#[test]
fn writers_queued_behind_a_flush_share_one_barrier() {
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(open(&dir.path().join("g.kgl"), false, None));
    write(&session, "CREATE (:N {id: 0})").unwrap();
    let before = session.wal_barrier_count();

    let queries = (2..10)
        .map(|i| format!("CREATE (:N {{id: {i}}})"))
        .collect();
    let results = parked_batch(&session, "CREATE (:N {id: 1})", queries);

    assert!(results.iter().all(Result::is_ok), "{results:?}");
    assert_eq!(ids(&session.snapshot()), (0..10).collect::<Vec<_>>());
    assert_eq!(
        session.wal_barrier_count() - before,
        2,
        "the parked write's barrier plus one for the eight queued behind it"
    );
}

#[test]
fn a_batch_is_bounded_and_served_first_come_first_served() {
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(open(&dir.path().join("g.kgl"), true, None));
    write(&session, "CREATE (:N {id: 0})").unwrap();
    let before = session.wal_barrier_count();

    let waiting = MAX_BATCH * 2 + 20;
    let queries = (1..=waiting)
        .map(|i| format!("CREATE (:N {{id: {i}}})"))
        .collect();
    let results = parked_batch(&session, "CREATE (:N {id: -1})", queries);

    assert!(results.iter().all(Result::is_ok), "{results:?}");
    // The parked write, then the queue in batches of at most MAX_BATCH.
    assert_eq!(
        session.wal_barrier_count() - before,
        1 + waiting.div_ceil(MAX_BATCH) as u64
    );
    let snapshot = session.snapshot();
    assert_eq!(ids(&snapshot).len(), waiting + 2);
    let order: Vec<i64> = cdc::read(&snapshot, 0, None, &[])
        .unwrap()
        .iter()
        .filter_map(|event| match &event.change {
            CdcChange::Node {
                id: Value::Int64(id),
                ..
            } => Some(*id),
            _ => None,
        })
        .collect();
    let expected: Vec<i64> = std::iter::once(0)
        .chain(std::iter::once(-1))
        .chain(1..=waiting as i64)
        .collect();
    assert_eq!(order, expected, "writers commit in the order they queued");
}

const REQUIRED_EDGE: &str = r#"{"classes": {"Person": {}, "Company": {}},
    "relationships": {"WORKS_AT": {"domain": "Person", "range": "Company",
        "required": true, "enforcement": "error"}}}"#;

#[test]
fn a_failing_statement_amid_a_batch_fails_alone_and_the_rest_are_durable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");
    let session = Arc::new(open(&path, false, Some(REQUIRED_EDGE)));
    let good = |i: i64| format!("CREATE (:Person {{id: {i}}})-[:WORKS_AT]->(:Company {{id: {i}}})");

    let results = parked_batch(
        &session,
        &good(1),
        vec![
            good(2),
            "CREATE (:Person {id: 3})".to_string(),
            good(4),
            "UNWIND [5, 6] AS i CREATE (:Person {id: i})".to_string(),
            good(7),
        ],
    );

    let verdicts: Vec<bool> = results.iter().map(Result::is_ok).collect();
    assert_eq!(
        verdicts,
        [true, true, false, true, false, true],
        "{results:?}"
    );
    for refused in [&results[2], &results[4]] {
        assert!(
            matches!(refused, Err(e) if matches!(**e, KgError::OntologyViolation { .. })),
            "a refused statement returns its own ontology error: {refused:?}"
        );
    }
    let people = |graph: &DirGraph| column(graph, "MATCH (p:Person) RETURN p.id");
    assert_eq!(people(&session.snapshot()), [1, 2, 4, 7]);

    drop(session);
    let reopened = open(&path, false, Some(REQUIRED_EDGE));
    assert_eq!(
        people(&reopened.snapshot()),
        [1, 2, 4, 7],
        "a crash-image reopen holds exactly the statements that succeeded"
    );
}

#[test]
fn change_events_carry_each_statements_own_after_state_in_commit_order() {
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(open(&dir.path().join("g.kgl"), true, None));
    write(&session, "CREATE (:N {id: 1, v: 0})").unwrap();

    let bump = "MATCH (n:N {id: 1}) SET n.v = n.v + 1".to_string();
    let results = parked_batch(&session, &bump, vec![bump.clone(); 5]);
    assert!(results.iter().all(Result::is_ok), "{results:?}");

    let events = cdc::read(&session.snapshot(), 0, None, &[]).unwrap();
    let seen: Vec<i64> = events
        .iter()
        .map(|event| match &event.change {
            CdcChange::Node {
                after: Some(after), ..
            } => after
                .properties
                .iter()
                .find_map(|(key, value)| match (key.as_str(), value) {
                    ("v", Value::Int64(v)) => Some(*v),
                    _ => None,
                })
                .expect("v is set"),
            other => panic!("unexpected change {other:?}"),
        })
        .collect();
    assert_eq!(
        seen,
        [0, 1, 2, 3, 4, 5, 6],
        "each event shows its own statement's value, once, in commit order"
    );
    let seqs: Vec<u64> = events.iter().map(|event| event.seq).collect();
    assert!(seqs.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn a_failed_barrier_fails_every_member_of_its_batch_and_the_log_recovers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");
    let session = Arc::new(open(&path, true, None));
    write(&session, "CREATE (:N {id: 0})").unwrap();
    let version = session.version();

    let hook = session.park_next_commit();
    let _release = ReleaseOnDrop(Arc::clone(&hook));
    let leader = spawn_write(&session, "CREATE (:N {id: 1})".to_string());
    assert!(hook.wait_entered(WAIT));
    let followers: Vec<_> = (2..5)
        .map(|i| {
            let handle = spawn_write(&session, format!("CREATE (:N {{id: {i}}})"));
            queued(&session, i - 1);
            handle
        })
        .collect();
    // Frames staged from here on carry the failing barrier.
    session.set_wal_fault(Some(AppendFault::SyncError));
    let lsn_before_batch = session.next_lsn();
    hook.release();

    leader.join().unwrap().unwrap();
    for follower in followers {
        assert!(
            matches!(
                follower.join().unwrap(),
                Err(e) if matches!(*e, KgError::DurabilityFailed { .. })
            ),
            "a member of a batch whose barrier failed is not acknowledged"
        );
    }
    assert_eq!(ids(&session.snapshot()), [0, 1]);
    assert_eq!(session.version(), version + 1);
    assert_eq!(
        session.next_lsn(),
        lsn_before_batch,
        "the batch's LSNs are given back"
    );
    assert_eq!(
        cdc::read(&session.snapshot(), 0, None, &[]).unwrap().len(),
        2,
        "a refused batch publishes no change event"
    );

    session.set_wal_fault(None);
    write(&session, "CREATE (:N {id: 9})").unwrap();
    drop(session);
    assert_eq!(
        ids(&open(&path, false, None).snapshot()),
        [0, 1, 9],
        "replay holds exactly the acknowledged commits"
    );
}

#[test]
fn checkpoints_racing_a_queued_batch_lose_nothing() {
    type Op = fn(&Session, &Path);
    let ops: [(&str, Op); 3] = [
        ("save", |s, p| s.save(&p.to_string_lossy(), true).unwrap()),
        ("checkpoint_online", |s, _| {
            s.checkpoint_online().unwrap();
        }),
        ("backup", |s, p| {
            s.backup(&p.with_file_name("backup.kgl"), &Default::default())
                .unwrap();
        }),
    ];
    for (name, op) in ops {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = Arc::new(open(&path, false, None));
        let hook = session.park_next_commit();
        let _release = ReleaseOnDrop(Arc::clone(&hook));
        let leader = spawn_write(&session, "CREATE (:N {id: 1})".to_string());
        assert!(hook.wait_entered(WAIT), "{name}: the write never parked");
        let followers: Vec<_> = (2..5)
            .map(|i| {
                let handle = spawn_write(&session, format!("CREATE (:N {{id: {i}}})"));
                queued(&session, i - 1);
                handle
            })
            .collect();
        let racer = Arc::clone(&session);
        let racer_path = path.clone();
        let racing = thread::spawn(move || op(&racer, &racer_path));
        thread::sleep(Duration::from_millis(200));
        hook.release();
        leader.join().unwrap().unwrap();
        for follower in followers {
            follower.join().unwrap().unwrap();
        }
        racing.join().unwrap();

        drop(session);
        assert_eq!(
            ids(&open(&path, false, None).snapshot()),
            [1, 2, 3, 4],
            "{name}: a crash after the race must lose nothing and invent nothing"
        );
    }
}
