//! `Session::write_logged`, `apply_unlogged`, `retire`, `attach_log` and the
//! LSN accessors.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::{execute_mut, execute_read, CommitOutcome, ExecuteOptions, Session, WriteError};
use crate::error::KgError;
use crate::graph::dir_graph::DirGraph;
use crate::graph::durability::DurableOpenError;
use crate::graph::wal::{recover, wal_path, DurabilityLevel};

fn run(graph: &mut DirGraph, query: &str) -> Result<(), String> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn ids(session: &Session) -> Vec<i64> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let out = execute_read(
        &session.snapshot(),
        "MATCH (n:N) RETURN n.id AS id ORDER BY id",
        &opts,
    )
    .unwrap();
    out.result
        .rows
        .iter()
        .map(|row| match &row[0] {
            crate::datatypes::Value::Int64(n) => *n,
            other => panic!("id was {other:?}"),
        })
        .collect()
}

fn durable(path: &Path, level: DurabilityLevel) -> Session {
    Session::attach_log(Arc::new(DirGraph::new()), &path.to_string_lossy(), level)
        .unwrap_or_else(|e| panic!("attach failed: {e}"))
}

fn reopen(path: &Path, level: DurabilityLevel) -> Session {
    let p = path.to_string_lossy().into_owned();
    let graph = if path.exists() {
        crate::graph::io::file::load_file(&p).unwrap()
    } else {
        Arc::new(DirGraph::new())
    };
    Session::attach_log(graph, &p, level).unwrap_or_else(|e| panic!("reopen failed: {e}"))
}

fn create(session: &Session, id: i64) -> Result<super::WriteOutcome<()>, WriteError<String>> {
    session.write_logged(|g| run(g, &format!("CREATE (:N {{id: {id}}})")))
}

#[test]
fn a_logged_write_publishes_logs_and_replays_after_a_crash() {
    for level in [DurabilityLevel::Full, DurabilityLevel::Normal] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = durable(&path, level);
        let before = session.version();

        let first = create(&session, 1).unwrap();
        assert_eq!(first.lsn, Some(1));
        assert_eq!(first.version, before + 1);
        let second = create(&session, 2).unwrap();
        assert_eq!(second.lsn, Some(2));
        assert_eq!(session.next_lsn(), Some(3));
        assert_eq!(session.last_lsn(), Some(2));
        assert_eq!(ids(&session), vec![1, 2]);
        drop(session); // crash-shaped: no checkpoint was ever written

        assert!(!path.exists());
        let recovered = reopen(&path, level);
        assert_eq!(ids(&recovered), vec![1, 2], "{level:?}: both frames replay");
        assert_eq!(recovered.next_lsn(), Some(3), "{level:?}: LSNs continue");
    }
}

#[test]
fn a_closure_that_fails_publishes_nothing_and_burns_no_lsn() {
    let dir = tempfile::tempdir().unwrap();
    let session = durable(&dir.path().join("g.kgl"), DurabilityLevel::Full);
    create(&session, 1).unwrap();
    let snapshot = session.snapshot();
    let (version, next) = (session.version(), session.next_lsn());

    let err = session
        .write_logged(|g| {
            run(g, "CREATE (:N {id: 2})")?;
            Err::<(), _>("boom".to_string())
        })
        .unwrap_err();
    assert!(
        matches!(err, WriteError::Closure(ref m) if m == "boom"),
        "{err:?}"
    );
    assert!(Arc::ptr_eq(&snapshot, &session.snapshot()));
    assert_eq!((session.version(), session.next_lsn()), (version, next));
    assert_eq!(ids(&session), vec![1]);

    // Non-vacuity: the same closure body without the error does publish.
    create(&session, 2).unwrap();
    assert_eq!(ids(&session), vec![1, 2]);
}

#[test]
fn a_failed_append_blocks_the_publish_and_the_session_stays_usable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");
    let session = durable(&path, DurabilityLevel::Full);
    create(&session, 1).unwrap();
    let snapshot = session.snapshot();
    let (version, next) = (session.version(), session.next_lsn());

    session.set_fail_append(true);
    let err = create(&session, 2).unwrap_err();
    assert!(
        matches!(err, WriteError::Durability(ref m) if m.contains("injected")),
        "{err:?}"
    );
    assert!(Arc::ptr_eq(&snapshot, &session.snapshot()));
    assert_eq!((session.version(), session.next_lsn()), (version, next));
    assert_eq!(ids(&session), vec![1]);

    // D2: no poison. Once the fault clears the same write lands on the LSN the
    // failed one never took, and the log replays it.
    session.set_fail_append(false);
    assert_eq!(create(&session, 2).unwrap().lsn, next);
    drop(session);
    assert_eq!(ids(&reopen(&path, DurabilityLevel::Full)), vec![1, 2]);
}

#[test]
fn a_logged_write_and_an_auto_commit_statement_write_the_same_frame() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let logged = durable(&a.path().join("g.kgl"), DurabilityLevel::Full);
    let statement = durable(&b.path().join("g.kgl"), DurabilityLevel::Full);
    let query = "CREATE (:N {id: 7, name: 'x'})-[:R {w: 1}]->(:M {id: 8})";

    logged.write_logged(|g| run(g, query)).unwrap();
    let params = HashMap::new();
    statement
        .execute_auto_commit(query, &ExecuteOptions::eager(&params), 1)
        .unwrap();
    drop((logged, statement));

    let frames_a = recover(&wal_path(&a.path().join("g.kgl"))).unwrap();
    let frames_b = recover(&wal_path(&b.path().join("g.kgl"))).unwrap();
    assert_eq!(frames_a.len(), 1);
    assert_eq!(frames_a, frames_b);
    assert!(
        !frames_a[0].ops.is_empty(),
        "a vacuous comparison proves nothing"
    );
}

#[test]
fn a_non_durable_session_runs_closure_writes_without_a_log() {
    let session = Session::new(DirGraph::new());
    let out = create(&session, 1).unwrap();
    assert_eq!((out.lsn, session.next_lsn()), (None, None));
    assert_eq!(ids(&session), vec![1]);
    assert!(session.apply_unlogged(|_| Ok::<_, ()>(())).is_ok());
}

#[test]
fn a_closure_that_changes_nothing_logged_is_published_without_a_frame() {
    let dir = tempfile::tempdir().unwrap();
    let session = durable(&dir.path().join("g.kgl"), DurabilityLevel::Full);
    create(&session, 1).unwrap();
    let (version, next) = (session.version(), session.next_lsn());

    let out = session
        .write_logged(|g| {
            g.schema_locked = true;
            Ok::<_, ()>(())
        })
        .unwrap();
    assert_eq!((out.lsn, out.version), (None, version));
    assert_eq!((session.version(), session.next_lsn()), (version, next));
    assert!(
        session.snapshot().schema_locked,
        "the write was published, not dropped"
    );
}

#[test]
fn apply_unlogged_publishes_unlogged_state_and_refuses_captured_ops() {
    let dir = tempfile::tempdir().unwrap();
    let session = durable(&dir.path().join("g.kgl"), DurabilityLevel::Full);
    create(&session, 1).unwrap();
    let (version, next) = (session.version(), session.next_lsn());

    let out = session
        .apply_unlogged(|g| {
            g.schema_locked = true;
            Ok::<_, ()>(5)
        })
        .unwrap();
    assert_eq!((out.value, out.lsn), (5, None));
    assert!(session.snapshot().schema_locked);
    assert_eq!((session.version(), session.next_lsn()), (version, next));

    // The mis-filed mutator: it changes logged state, so it must not pass.
    let snapshot = session.snapshot();
    let err = session
        .apply_unlogged(|g| run(g, "CREATE (:N {id: 2})"))
        .unwrap_err();
    assert!(
        matches!(err, WriteError::CapturedOps(n) if n > 0),
        "{err:?}"
    );
    assert!(Arc::ptr_eq(&snapshot, &session.snapshot()));
    assert_eq!(ids(&session), vec![1]);
    assert_eq!(session.next_lsn(), next, "a refused write takes no LSN");
}

#[test]
fn a_retired_session_refuses_every_way_to_publish() {
    for level in [DurabilityLevel::Full, DurabilityLevel::Normal] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = durable(&path, level);
        create(&session, 1).unwrap();
        let held = session.begin();
        let mut open_tx = session.begin();
        run(open_tx.working_mut().unwrap(), "CREATE (:N {id: 9})").unwrap();
        let snapshot = session.snapshot();
        let (version, next) = (session.version(), session.next_lsn());
        let wal_len = std::fs::metadata(wal_path(&path)).unwrap().len();

        session.retire();
        session.retire(); // idempotent
        assert!(session.is_retired());

        assert!(matches!(create(&session, 2), Err(WriteError::Retired)));
        assert!(matches!(
            session.apply_unlogged(|_| Ok::<_, ()>(())),
            Err(WriteError::Retired)
        ));
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        assert!(matches!(
            session.execute_auto_commit("CREATE (:N {id: 3})", &opts, 1),
            Err(KgError::DurabilityFailed { .. })
        ));
        assert!(matches!(
            session.commit(open_tx, true),
            CommitOutcome::DurabilityFailed { .. }
        ));
        assert!(session.save(&path.to_string_lossy(), true).is_err());
        assert!(session.checkpoint_online().is_err());
        assert!(!session.needs_checkpoint());
        assert!(session.check_direct_write_allowed().is_err());
        session.rollback(held);

        // Nothing moved, and reads still work.
        assert!(Arc::ptr_eq(&snapshot, &session.snapshot()), "{level:?}");
        assert_eq!((session.version(), session.next_lsn()), (version, next));
        assert_eq!(
            std::fs::metadata(wal_path(&path)).unwrap().len(),
            wal_len,
            "{level:?}: a retired session appends nothing"
        );
        assert_eq!(ids(&session), vec![1]);
    }
}

#[test]
fn attach_log_keeps_the_failure_category() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");

    // Refused: the graph is already wrapped by a log owner.
    let owner = durable(&path, DurabilityLevel::Full);
    let wrapped = owner.snapshot();
    let other = dir.path().join("other.kgl");
    match Session::attach_log(wrapped, &other.to_string_lossy(), DurabilityLevel::Full) {
        Err(DurableOpenError::Refused(_)) => {}
        Err(other) => panic!("expected Refused, got {other:?}"),
        Ok(_) => panic!("a second owner must be refused"),
    }

    // Io: the sidecar path cannot be read as a log.
    let blocked = dir.path().join("blocked.kgl");
    std::fs::create_dir(wal_path(&blocked)).unwrap();
    match Session::attach_log(
        Arc::new(DirGraph::new()),
        &blocked.to_string_lossy(),
        DurabilityLevel::Full,
    ) {
        Err(DurableOpenError::Io(_)) => {}
        Err(other) => panic!("expected Io, got {other:?}"),
        Ok(_) => panic!("an unreadable sidecar must not attach"),
    }

    // `open_durable` is the same attach with the category flattened.
    let flattened = Session::open_durable(
        Arc::new(DirGraph::new()),
        &blocked.to_string_lossy(),
        DurabilityLevel::Full,
    );
    assert!(flattened.is_err());
}
