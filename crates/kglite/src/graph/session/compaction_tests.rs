use super::{execute_mut, execute_read, CommitOutcome, ExecuteOptions, Session};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};
use crate::graph::storage::GraphRead;
use std::collections::HashMap;
use std::sync::Arc;

fn execute(graph: &mut DirGraph, query: &str) {
    execute_mut(graph, query, &ExecuteOptions::eager(&HashMap::new())).unwrap();
}

fn seed() -> Session {
    let mut graph = DirGraph::new();
    execute(&mut graph, "UNWIND range(0, 7) AS i CREATE (:Item {id:i, title:toString(i), city:'old', region:'r', score:i})");
    Session::new(graph)
}

fn commit(session: &Session, query: &str) -> CommitOutcome {
    let mut tx = session.begin();
    execute(tx.working_mut().unwrap(), query);
    session.commit(tx, true)
}

fn append(session: &Session, id: usize) {
    assert!(matches!(
        commit(
            session,
            &format!(
                "CREATE (:Item {{id:{id}, title:'{id}', city:'new', region:'r', score:{id}}})"
            )
        ),
        CommitOutcome::Committed { .. }
    ));
}

fn ids(graph: &DirGraph, query: &str) -> Vec<String> {
    execute_read(graph, query, &ExecuteOptions::eager(&HashMap::new()))
        .unwrap()
        .result
        .rows
        .into_iter()
        .map(|row| {
            let Value::String(id) = &row[0] else {
                panic!("expected string ID")
            };
            id.clone()
        })
        .collect()
}

fn exact_slots(graph: &DirGraph, count: usize) {
    let actual: Vec<_> = graph
        .graph
        .node_indices()
        .map(|index| {
            let view = graph.graph.node_view(index).unwrap();
            (
                index.index(),
                view.id().to_string(),
                view.title().into_owned(),
            )
        })
        .collect();
    let expected: Vec<_> = (0..count)
        .map(|id| (id, id.to_string(), Value::String(id.to_string())))
        .collect();
    assert_eq!(
        actual, expected,
        "compaction must preserve physical coordinates and identity values"
    );
}

#[test]
fn serial_publication_releases_the_old_owner_before_compacting() {
    let session = seed();
    for id in 8..16 {
        append(&session, id);
        let snapshot = session.snapshot();
        exact_slots(&snapshot, id + 1);
        assert!(
            !snapshot.graph.is_forked(),
            "unheld successful publication must fold its overlay"
        );
    }
}

#[test]
fn serial_publication_does_not_copy_accumulated_node_weights() {
    let session = seed();
    reset_backend_clone_count();
    for id in 8..16 {
        append(&session, id);
    }
    exact_slots(&session.snapshot(), 16);
    assert_eq!(
        backend_clone_nodes(),
        0,
        "serial commits must not repeatedly copy prior committed weights"
    );
}

#[test]
fn held_reader_stays_exact_and_next_publication_after_drop_compacts() {
    let session = seed();
    let holder = session.snapshot();
    for id in 8..12 {
        append(&session, id);
    }
    exact_slots(&holder, 8);
    assert!(session.snapshot().graph.is_forked());
    drop(holder);
    append(&session, 12);
    exact_slots(&session.snapshot(), 13);
    assert!(!session.snapshot().graph.is_forked());
    reset_backend_clone_count();
    append(&session, 13);
    assert_eq!(backend_clone_nodes(), 0);
}

#[test]
fn fresh_reader_at_every_commit_can_keep_the_overlay_shared() {
    let session = seed();
    for id in 8..12 {
        let holder = session.snapshot();
        append(&session, id);
        exact_slots(&holder, id);
        exact_slots(&session.snapshot(), id + 1);
        assert!(session.snapshot().graph.is_forked());
        drop(holder);
    }
    append(&session, 12);
    exact_slots(&session.snapshot(), 13);
}

#[test]
fn publication_preserves_column_and_all_user_index_queries() {
    let session = seed();
    for query in [
        "CREATE INDEX FOR (n:Item) ON (n.city)",
        "CREATE INDEX FOR (n:Item) ON (n.city, n.region)",
        "CREATE RANGE INDEX FOR (n:Item) ON (n.score)",
    ] {
        assert!(matches!(
            commit(&session, query),
            CommitOutcome::Committed { .. }
        ));
    }
    let old = session.snapshot();
    for id in 8..13 {
        append(&session, id);
    }
    assert!(matches!(
        commit(
            &session,
            "MATCH (n:Item {id:2}) SET n.city='moved', n.score=99"
        ),
        CommitOutcome::Committed { .. }
    ));
    assert_eq!(
        ids(
            &old,
            "MATCH (n:Item) WHERE n.city='moved' RETURN toString(n.id)"
        ),
        Vec::<String>::new()
    );
    drop(old);
    append(&session, 13);
    let graph = session.snapshot();
    exact_slots(&graph, 14);
    for query in [
        "MATCH (n:Item) WHERE n.city='moved' RETURN toString(n.id)",
        "MATCH (n:Item) WHERE n.city='moved' AND n.region='r' RETURN toString(n.id)",
        "MATCH (n:Item {id:2}) RETURN toString(n.id)",
    ] {
        assert_eq!(ids(&graph, query), vec!["2"]);
    }
    assert_eq!(
        ids(
            &graph,
            "MATCH (n:Item) WHERE n.score>=10 RETURN toString(n.id) ORDER BY n.id"
        ),
        vec!["2", "10", "11", "12", "13"]
    );
}

#[test]
fn conflict_rollback_and_no_write_do_not_change_publication() {
    let session = seed();
    let mut loser = session.begin();
    execute(
        loser.working_mut().unwrap(),
        "CREATE (:Item {id:99,title:'99'})",
    );
    append(&session, 8);
    let before = session.snapshot();
    let version = session.version();
    assert!(matches!(
        session.commit(loser, true),
        CommitOutcome::ConflictDetected { .. }
    ));
    assert!(Arc::ptr_eq(&before, &session.snapshot()));
    let mut rollback = session.begin();
    execute(
        rollback.working_mut().unwrap(),
        "CREATE (:Item {id:99,title:'99'})",
    );
    session.rollback(rollback);
    assert!(matches!(
        session.commit(session.begin(), true),
        CommitOutcome::NoWritesNoOp
    ));
    assert!(Arc::ptr_eq(&before, &session.snapshot()));
    assert_eq!(session.version(), version);
    exact_slots(&session.snapshot(), 9);
}

#[test]
fn durable_append_failure_preserves_published_identity_then_recovery_is_exact() {
    use crate::graph::io::file::{load_file, save_graph_with};
    use crate::graph::wal::DurabilityLevel;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.kgl");
    let path = path.to_str().unwrap();
    let seed = seed();
    let mut graph = seed.snapshot();
    drop(seed);
    save_graph_with(&mut graph, path, false).unwrap();
    let session = Session::open_durable(graph, path, DurabilityLevel::Normal).unwrap();
    append(&session, 8);
    let before = session.snapshot();
    session.set_fail_append(true);
    assert!(matches!(
        commit(&session, "CREATE (:Item {id:99,title:'99'})"),
        CommitOutcome::DurabilityFailed { .. }
    ));
    assert!(Arc::ptr_eq(&before, &session.snapshot()));
    exact_slots(&before, 9);
    drop(before);
    session.set_fail_append(false);
    append(&session, 9);
    session.sync().unwrap();
    drop(session);
    let recovered =
        Session::open_durable(load_file(path).unwrap(), path, DurabilityLevel::Normal).unwrap();
    // Logical WAL replay folds net entities through a map and does not promise
    // original physical slots. Publication above does; recovery promises values.
    let rows = execute_read(&recovered.snapshot(),
        "MATCH (n:Item) RETURN toString(n.id), n.title, n.city, n.region, toString(n.score) ORDER BY n.id",
        &ExecuteOptions::eager(&HashMap::new())).unwrap().result.rows;
    let expected: Vec<Vec<Value>> = (0..10)
        .map(|id| {
            [
                id.to_string(),
                id.to_string(),
                if id < 8 { "old" } else { "new" }.into(),
                "r".into(),
                id.to_string(),
            ]
            .into_iter()
            .map(Value::String)
            .collect()
        })
        .collect();
    assert_eq!(rows, expected);
}
