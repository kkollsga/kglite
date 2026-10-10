//! A durable commit's barrier must not stall readers, and nothing may observe
//! a commit before it is durable.
//!
//! Each test parks one commit inside its barrier ([`ParkHook`]) and acts from
//! other threads while it is held there. Waits for "this must not block" use
//! `recv_timeout`, never a sleep; the one place a sleep stands in is giving a
//! thread that is *expected* to block time to reach its lock.

use super::execute::{execute_mut, execute_read, ExecuteOptions};
use super::{CommitOutcome, Session};
use crate::graph::cdc::{self, CdcEnrichment};
use crate::graph::dir_graph::DirGraph;
use crate::graph::wal::{AppendFault, DurabilityLevel, ParkHook};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(10);

/// Releases a parked commit when dropped, so a failing assertion cannot leave
/// the committing thread blocked forever.
struct ReleaseOnDrop(Arc<ParkHook>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn open(path: &Path, level: DurabilityLevel) -> Session {
    open_graph(path, level, None)
}

fn open_graph(path: &Path, level: DurabilityLevel, cdc_on: Option<()>) -> Session {
    let p = path.to_string_lossy().into_owned();
    let graph = if path.exists() {
        crate::graph::io::file::load_file(&p).unwrap()
    } else {
        let mut graph = DirGraph::new();
        if cdc_on.is_some() {
            cdc::enable(&mut graph, Some(64), CdcEnrichment::Off).unwrap();
        }
        Arc::new(graph)
    };
    Session::open_durable(graph, &p, level).unwrap()
}

fn commit(session: &Session, query: &str) -> CommitOutcome {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let mut tx = session.begin();
    execute_mut(tx.working_mut().unwrap(), query, &opts).unwrap();
    session.commit(tx, true)
}

fn ids(graph: &DirGraph) -> Vec<i64> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let mut ids: Vec<i64> = execute_read(graph, "MATCH (n:N) RETURN n.id AS id", &opts)
        .unwrap()
        .result
        .rows
        .iter()
        .map(|row| match &row[0] {
            crate::datatypes::Value::Int64(v) => *v,
            other => panic!("unexpected id {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

fn spawn_commit(session: &Arc<Session>, query: &'static str) -> JoinHandle<CommitOutcome> {
    let session = Arc::clone(session);
    thread::spawn(move || commit(&session, query))
}

/// Run `f` on its own thread and wait for its answer.
fn within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(5)).ok()
}

/// Reopen the way a crash does: checkpoint if any, then replay the sidecar.
fn reopen(path: &Path) -> Session {
    open(path, DurabilityLevel::Full)
}

#[test]
fn a_reader_is_not_blocked_by_a_commit_parked_in_its_barrier() {
    for level in [DurabilityLevel::Full, DurabilityLevel::Normal] {
        let dir = tempfile::tempdir().unwrap();
        let session = Arc::new(open(&dir.path().join("g.kgl"), level));
        assert!(matches!(
            commit(&session, "CREATE (:N {id: 1})"),
            CommitOutcome::Committed { .. }
        ));
        let hook = session.park_next_commit();
        let _release = ReleaseOnDrop(Arc::clone(&hook));
        let writer = spawn_commit(&session, "CREATE (:N {id: 2})");
        assert!(
            hook.wait_entered(WAIT),
            "{level:?}: the commit never parked"
        );

        let reader = Arc::clone(&session);
        let seen = within(move || ids(&reader.snapshot()));
        assert_eq!(
            seen,
            Some(vec![1]),
            "{level:?}: a reader must return promptly and must not see the parked write"
        );

        hook.release();
        assert!(matches!(
            writer.join().unwrap(),
            CommitOutcome::Committed { .. }
        ));
        assert_eq!(ids(&session.snapshot()), vec![1, 2], "{level:?}");
    }
}

#[test]
fn change_events_follow_the_commit_they_describe() {
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(open_graph(
        &dir.path().join("g.kgl"),
        DurabilityLevel::Full,
        Some(()),
    ));
    let hook = session.park_next_commit();
    let _release = ReleaseOnDrop(Arc::clone(&hook));
    let writer = spawn_commit(&session, "CREATE (:N {id: 1})");
    assert!(hook.wait_entered(WAIT));

    let reader = Arc::clone(&session);
    let parked = within(move || {
        let snapshot = reader.snapshot();
        (
            ids(&snapshot),
            cdc::read(&snapshot, 0, None, &[]).unwrap().len(),
        )
    });
    assert_eq!(
        parked,
        Some((vec![], 0)),
        "no event before the commit is durable"
    );

    hook.release();
    writer.join().unwrap();
    let snapshot = session.snapshot();
    assert_eq!(ids(&snapshot), vec![1]);
    assert_eq!(cdc::read(&snapshot, 0, None, &[]).unwrap().len(), 1);
}

#[test]
fn a_failed_barrier_leaves_the_graph_unpublished_and_the_session_usable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");
    let session = open_graph(&path, DurabilityLevel::Full, Some(()));
    assert!(matches!(
        commit(&session, "CREATE (:N {id: 1})"),
        CommitOutcome::Committed { .. }
    ));
    let version = session.version();
    let lsn = session.next_lsn();

    session.set_wal_fault(Some(AppendFault::SyncError));
    assert!(matches!(
        commit(&session, "CREATE (:N {id: 2})"),
        CommitOutcome::DurabilityFailed { .. }
    ));
    assert_eq!(ids(&session.snapshot()), vec![1]);
    assert_eq!(session.version(), version);
    assert_eq!(session.next_lsn(), lsn, "a refused frame consumes no LSN");
    assert_eq!(
        cdc::read(&session.snapshot(), 0, None, &[]).unwrap().len(),
        1,
        "a refused commit publishes no change event"
    );

    session.set_wal_fault(None);
    assert!(matches!(
        commit(&session, "CREATE (:N {id: 3})"),
        CommitOutcome::Committed { .. }
    ));
    assert_eq!(ids(&session.snapshot()), vec![1, 3]);
    drop(session);
    assert_eq!(
        ids(&reopen(&path).snapshot()),
        vec![1, 3],
        "replay holds exactly the acknowledged commits"
    );
}

/// A crash image taken while a commit is between "frame written" and "swap":
/// the frame replays, and the live graph never showed it.
#[test]
fn a_crash_between_the_frame_and_the_swap_replays_the_frame() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");
    let session = Arc::new(open(&path, DurabilityLevel::Full));
    commit(&session, "CREATE (:N {id: 1})");
    let live = session.snapshot();
    let hook = session.park_next_commit();
    let _release = ReleaseOnDrop(Arc::clone(&hook));
    let writer = spawn_commit(&session, "CREATE (:N {id: 2})");
    assert!(hook.wait_entered(WAIT));

    let image = tempfile::tempdir().unwrap();
    let copy = image.path().join("g.kgl");
    let wal = crate::graph::wal::wal_path(&path);
    std::fs::copy(&wal, crate::graph::wal::wal_path(&copy)).unwrap();
    assert_eq!(
        ids(&live),
        vec![1],
        "the pre-swap graph never showed the frame"
    );
    assert_eq!(ids(&reopen(&copy).snapshot()), vec![1, 2]);

    hook.release();
    writer.join().unwrap();
}

/// What a checkpoint-shaped operation fixes while a commit is parked must not
/// pair a graph without the frame with an LSN that includes it.
#[test]
fn checkpoints_racing_a_parked_commit_lose_nothing() {
    type Op = fn(&Session, &Path, u64);
    let ops: [(&str, Op); 3] = [
        ("save", |s, p, _| {
            s.save(&p.to_string_lossy(), true).unwrap()
        }),
        ("checkpoint_online", |s, _, _| {
            s.checkpoint_online().unwrap();
        }),
        ("backup", |s, p, first| {
            let dest = p.with_file_name("backup.kgl");
            let report = s.backup(&dest, &Default::default()).unwrap();
            let loaded = crate::graph::io::file::load_file(&dest.to_string_lossy()).unwrap();
            let lsn = report.lsn.unwrap();
            // Node `i` was committed at LSN `lsn_of_1 + i - 1`; the backup holds
            // exactly the commits its stamp claims.
            assert_eq!(
                ids(&loaded).len() as u64,
                lsn - first + 1,
                "backup stamped lsn {lsn} must contain exactly the commits up to it"
            );
        }),
    ];
    for (name, op) in ops {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = Arc::new(open(&path, DurabilityLevel::Full));
        let first = session.next_lsn().unwrap();
        commit(&session, "CREATE (:N {id: 1})");
        let hook = session.park_next_commit();
        let _release = ReleaseOnDrop(Arc::clone(&hook));
        let writer = spawn_commit(&session, "CREATE (:N {id: 2})");
        assert!(hook.wait_entered(WAIT), "{name}: the commit never parked");

        let racer = Arc::clone(&session);
        let racer_path = path.clone();
        let racing = thread::spawn(move || op(&racer, &racer_path, first));
        // The racer is expected to wait for the commit; give it time to reach
        // whatever it takes first.
        thread::sleep(Duration::from_millis(300));
        hook.release();
        writer.join().unwrap();
        racing.join().unwrap();

        let session = Arc::try_unwrap(session).ok().expect("all threads joined");
        drop(session);
        assert_eq!(
            ids(&reopen(&path).snapshot()),
            vec![1, 2],
            "{name}: a crash after the race must lose nothing and invent nothing"
        );
    }
}

/// A committer that began before another's commit published conflicts on the
/// version, even though it reached the gate while that commit was mid-barrier.
#[test]
fn a_committer_queued_behind_a_parked_commit_sees_its_version() {
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(open(&dir.path().join("g.kgl"), DurabilityLevel::Full));
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let mut late = session.begin();
    execute_mut(late.working_mut().unwrap(), "CREATE (:N {id: 9})", &opts).unwrap();

    let hook = session.park_next_commit();
    let _release = ReleaseOnDrop(Arc::clone(&hook));
    let writer = spawn_commit(&session, "CREATE (:N {id: 1})");
    assert!(hook.wait_entered(WAIT));
    let queued = {
        let session = Arc::clone(&session);
        thread::spawn(move || session.commit(late, true))
    };
    thread::sleep(Duration::from_millis(200));
    hook.release();
    assert!(matches!(
        writer.join().unwrap(),
        CommitOutcome::Committed { .. }
    ));
    assert!(matches!(
        queued.join().unwrap(),
        CommitOutcome::ConflictDetected { .. }
    ));
    assert_eq!(ids(&session.snapshot()), vec![1]);
}
