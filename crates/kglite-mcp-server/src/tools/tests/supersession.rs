//! Supersession tests: what a newer activation keeps, requeues or discards.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use kglite::api::session::ExecuteOutcome;
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::Value;

use super::*;

fn assert_compatible_supersession_requeues(fail_lazy: bool) {
    use std::sync::Barrier;

    let requests = Arc::new(Mutex::new(Vec::new()));
    let rebuild_started = Arc::new(Barrier::new(2));
    let release_rebuild = Arc::new(Barrier::new(2));
    let hooks = blocking_lazy_hooks(
        fail_lazy,
        requests.clone(),
        rebuild_started.clone(),
        release_rebuild.clone(),
    );
    let state =
        GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace)).with_workspace_graph(Some(hooks));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    let changed = root.join("changed.rs");
    let newer = root.join("newer.rs");
    state
        .build_workspace_graph(&root, None)
        .expect("install generation one");
    state.tag_workspace_graph_dirty(std::slice::from_ref(&changed));

    let rebuilding = state.clone();
    let thread = std::thread::spawn(move || rebuilding.ensure_workspace_graph_fresh());
    rebuild_started.wait();
    for generation in 2..=3 {
        let prepared = state
            .prepare_workspace_graph(&root, None, WorkspaceGraphChanges::Full)
            .unwrap_or_else(|error| panic!("prepare generation {generation}: {error}"));
        state.commit_workspace_graph(prepared);
    }
    state.tag_workspace_graph_dirty(std::slice::from_ref(&newer));
    release_rebuild.wait();
    thread.join().expect("superseded lazy thread");

    let active = state.active_workspace_target().expect("latest target");
    {
        let pending = read_lock(&state.pending_rebuild);
        let pending = pending.as_ref().expect("consumed A requeued");
        assert_eq!(pending.target, active, "A binds to latest generation");
        assert_eq!(
            pending.changed_paths,
            BTreeSet::from([changed.clone(), newer.clone()]),
            "consumed A unions with newer B"
        );
        assert!(pending.ready);
    }
    state.ensure_workspace_graph_fresh();
    assert!(state.workspace_rebuild_failure().is_none());
    assert_eq!(
        mutex_lock(&requests).last(),
        Some(&(root, WorkspaceGraphChanges::Changed(vec![changed, newer]),))
    );
}

#[test]
fn superseded_lazy_success_requeues_across_two_compatible_commits() {
    assert_compatible_supersession_requeues(false);
}

#[test]
fn superseded_lazy_failure_requeues_across_two_compatible_commits() {
    assert_compatible_supersession_requeues(true);
}

fn assert_incompatible_supersession_discards(fail_lazy: bool, change_revisions: bool) {
    use std::sync::Barrier;

    let requests = Arc::new(Mutex::new(Vec::new()));
    let rebuild_started = Arc::new(Barrier::new(2));
    let release_rebuild = Arc::new(Barrier::new(2));
    let hooks = blocking_lazy_hooks(
        fail_lazy,
        requests,
        rebuild_started.clone(),
        release_rebuild.clone(),
    );
    let state =
        GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace)).with_workspace_graph(Some(hooks));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root_a = workspace.path().join("a");
    let root_b = workspace.path().join("b");
    std::fs::create_dir_all(&root_a).expect("root A");
    std::fs::create_dir_all(&root_b).expect("root B");
    state
        .build_workspace_graph(&root_a, None)
        .expect("install source A");
    state.tag_workspace_graph_dirty(&[root_a.join("changed.rs")]);

    let rebuilding = state.clone();
    let thread = std::thread::spawn(move || rebuilding.ensure_workspace_graph_fresh());
    rebuild_started.wait();
    let prepared = if change_revisions {
        state.prepare_workspace_graph(
            &root_a,
            Some(&["HEAD".to_string()]),
            WorkspaceGraphChanges::Full,
        )
    } else {
        state.prepare_workspace_graph(&root_b, None, WorkspaceGraphChanges::Full)
    }
    .expect("prepare incompatible source");
    state.commit_workspace_graph(prepared);
    release_rebuild.wait();
    thread.join().expect("superseded lazy thread");

    assert!(read_lock(&state.pending_rebuild).is_none());
    assert!(state.workspace_rebuild_failure().is_none());
}

#[test]
fn superseded_lazy_success_discards_across_changed_root() {
    assert_incompatible_supersession_discards(false, false);
}

#[test]
fn superseded_lazy_failure_discards_across_changed_revisions() {
    assert_incompatible_supersession_discards(true, true);
}

#[test]
fn callback_binding_survives_two_compatible_commits() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(recording_hooks(requests)));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    let changed = root.join("changed.rs");
    state
        .build_workspace_graph(&root, None)
        .expect("install generation one");
    let observed_one = state.active_workspace_target().expect("generation one");

    let generation_two = state
        .prepare_workspace_graph(&root, None, WorkspaceGraphChanges::Full)
        .expect("prepare generation two");
    state.commit_workspace_graph(generation_two);

    let generation_three = state
        .prepare_workspace_graph(&root, None, WorkspaceGraphChanges::Full)
        .expect("prepare generation three");
    state.commit_workspace_graph(generation_three);
    assert_eq!(
        state.bind_workspace_graph_changes(&observed_one, std::slice::from_ref(&changed)),
        WorkspaceGraphEnqueue::Enqueued
    );

    let active = state.active_workspace_target().expect("generation three");
    let pending = read_lock(&state.pending_rebuild);
    let pending = pending
        .as_ref()
        .expect("compatible path binds without a finite retry race");
    assert_eq!(pending.target, active);
    assert_eq!(pending.changed_paths, BTreeSet::from([changed]));
}

#[test]
fn changed_revisions_discard_old_pending_hints() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(recording_hooks(requests)));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    let changed = root.join("changed.rs");
    state
        .build_workspace_graph(&root, None)
        .expect("install working-tree target");
    let observed = state.active_workspace_target().expect("old receipt");
    state.tag_workspace_graph_dirty(std::slice::from_ref(&changed));

    let prepared = state
        .prepare_workspace_graph(
            &root,
            Some(&["HEAD".to_string()]),
            WorkspaceGraphChanges::Full,
        )
        .expect("prepare revision target");
    state.commit_workspace_graph(prepared);

    assert!(
        read_lock(&state.pending_rebuild).is_none(),
        "working-tree hints do not cross into a revision snapshot"
    );
    assert_eq!(
        state.bind_workspace_graph_changes(&observed, &[changed]),
        WorkspaceGraphEnqueue::IncompatibleSource,
        "the old revision receipt is rejected"
    );
}

#[test]
fn different_root_activation_discards_old_pending_hints() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(recording_hooks(requests)));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root_a = workspace.path().join("a");
    let root_b = workspace.path().join("b");
    std::fs::create_dir_all(&root_a).expect("root A");
    std::fs::create_dir_all(&root_b).expect("root B");
    state
        .build_workspace_graph(&root_a, None)
        .expect("install target A");
    state.tag_workspace_graph_dirty(&[root_a.join("changed.rs")]);

    let prepared = state
        .prepare_workspace_graph(&root_b, None, WorkspaceGraphChanges::Full)
        .expect("prepare target B");
    state.commit_workspace_graph(prepared);

    assert!(
        read_lock(&state.pending_rebuild).is_none(),
        "target A hints do not cross into target B"
    );
}

#[test]
fn event_during_rebuild_retargets_only_newer_paths() {
    use std::sync::{atomic::AtomicUsize, atomic::Ordering, Barrier};

    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let rebuild_started = Arc::new(Barrier::new(2));
    let release_rebuild = Arc::new(Barrier::new(2));
    let hooks = Arc::new(WorkspaceGraphHooks {
        build: Box::new({
            let calls = calls.clone();
            let requests = requests.clone();
            let rebuild_started = rebuild_started.clone();
            let release_rebuild = release_rebuild.clone();
            move |request| {
                let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                mutex_lock(&requests).push(request.changes().clone());
                if call == 2 {
                    rebuild_started.wait();
                    release_rebuild.wait();
                }
                empty_workspace_result(&request)
            }
        }),
        is_relevant: Box::new(|_| true),
    });
    let state =
        GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace)).with_workspace_graph(Some(hooks));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let first = workspace.path().join("first.rs");
    let newer = workspace.path().join("newer.rs");
    state
        .build_workspace_graph(workspace.path(), None)
        .expect("initial build");
    state.tag_workspace_graph_dirty(std::slice::from_ref(&first));

    let rebuilding = state.clone();
    let thread = std::thread::spawn(move || rebuilding.ensure_workspace_graph_fresh());
    rebuild_started.wait();
    state.tag_workspace_graph_dirty(std::slice::from_ref(&newer));
    release_rebuild.wait();
    thread.join().expect("successful rebuild thread");

    let active_target = state.active_workspace_target().expect("installed target");
    let pending_guard = read_lock(&state.pending_rebuild);
    let pending = pending_guard.as_ref().expect("newer event remains pending");
    assert_eq!(
        pending.target, active_target,
        "event retargeted to generation two"
    );
    assert_eq!(pending.changed_paths, BTreeSet::from([newer.clone()]));
    drop(pending_guard);

    state.ensure_workspace_graph_fresh();
    let requests = mutex_lock(&requests);
    assert_eq!(
        requests.as_slice(),
        [
            WorkspaceGraphChanges::Full,
            WorkspaceGraphChanges::Changed(vec![first]),
            WorkspaceGraphChanges::Changed(vec![newer]),
        ]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn concurrent_freshness_call_waits_for_installed_generation() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Barrier,
    };

    fn generation(outcome: ExecuteOutcome) -> i64 {
        match outcome.result.rows.first().and_then(|row| row.first()) {
            Some(Value::Int64(value)) => *value,
            value => panic!("expected integer generation marker, got {value:?}"),
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let rebuild_started = Arc::new(Barrier::new(2));
    let release_rebuild = Arc::new(Barrier::new(2));
    let hooks = Arc::new(WorkspaceGraphHooks {
        build: Box::new({
            let calls = calls.clone();
            let rebuild_started = rebuild_started.clone();
            let release_rebuild = release_rebuild.clone();
            move |_| {
                let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                if call == 2 {
                    rebuild_started.wait();
                    release_rebuild.wait();
                }
                let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None)
                    .map_err(|error| error.to_string())?;
                let params = HashMap::new();
                let options = kglite::api::session::ExecuteOptions::eager(&params);
                kglite::api::session::execute_mut(
                    &mut graph,
                    &format!("CREATE (:Generation {{value: {call}}})"),
                    &options,
                )
                .map_err(|error| error.to_string())?;
                Ok(WorkspaceGraphResult::new(Arc::new(graph)))
            }
        }),
        is_relevant: Box::new(|_| true),
    });
    let state =
        GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace)).with_workspace_graph(Some(hooks));
    state
        .build_workspace_graph(Path::new("/workspace/current"), None)
        .expect("install generation one");
    state.tag_workspace_graph_dirty(&[PathBuf::from("/workspace/current/changed.rs")]);

    let first_state = state.clone();
    let first = std::thread::spawn(move || {
        first_state
            .execute_cypher_read(
                "MATCH (g:Generation) RETURN g.value AS generation",
                HashMap::new(),
            )
            .expect("first caller queries rebuilt graph")
    });
    rebuild_started.wait();

    let (second_result_tx, second_result_rx) = mpsc::channel();
    let second_state = state.clone();
    let second = std::thread::spawn(move || {
        let outcome = second_state
            .execute_cypher_read(
                "MATCH (g:Generation) RETURN g.value AS generation",
                HashMap::new(),
            )
            .expect("second caller queries rebuilt graph");
        second_result_tx
            .send(generation(outcome))
            .expect("report second result");
    });

    // Observe the second caller inside the gate's waiter set. This is not
    // a scheduler/sleep assertion: release happens only after the caller
    // is known to be blocked behind the in-flight rebuild.
    state.rebuild_gate.wait_for_waiter();
    assert!(
        matches!(second_result_rx.try_recv(), Err(mpsc::TryRecvError::Empty)),
        "second caller must not execute against generation one"
    );
    release_rebuild.wait();

    assert_eq!(
        generation(first.join().expect("first freshness caller joins")),
        2
    );
    second.join().expect("second freshness caller joins");
    assert_eq!(
        second_result_rx.recv().expect("second generation result"),
        2,
        "second caller sees the generation installed by the first caller"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2, "only one rebuild occurs");
}

#[test]
fn workspace_rebuild_preserves_multi_revision_target() {
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    std::fs::write(
        workspace.path().join("m.py"),
        "def changed():\n    return 1\n",
    )
    .unwrap();
    let state = GraphState::new(Some(WorkspaceGraphMode::Workspace))
        .with_workspace_graph(Some(test_hooks()));
    let revisions = vec!["base".to_string(), "head".to_string()];
    state
        .build_workspace_graph(workspace.path(), Some(&revisions))
        .expect("initial revision-set build");

    state.tag_workspace_graph_dirty(&[workspace.path().join("m.py")]);
    state.ensure_workspace_graph_fresh();

    assert_eq!(state.active_workspace_revisions(), Some(revisions));
    assert!(read_lock(&state.pending_rebuild).is_none());
}

#[test]
fn workspace_rebuild_cannot_overwrite_newer_activation() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let rebuild_started = Arc::new(Barrier::new(2));
    let release_rebuild = Arc::new(Barrier::new(2));
    let hooks = Arc::new(WorkspaceGraphHooks {
        build: Box::new({
            let calls = calls.clone();
            let requests = requests.clone();
            let rebuild_started = rebuild_started.clone();
            let release_rebuild = release_rebuild.clone();
            move |request| {
                let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                mutex_lock(&requests)
                    .push((request.root().to_path_buf(), request.changes().clone()));
                if call == 2 {
                    rebuild_started.wait();
                    release_rebuild.wait();
                }
                let graph = new_dir_graph_in_mode(StorageMode::Memory, None)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())?;
                Ok(match request.revisions() {
                    Some(revisions) => {
                        WorkspaceGraphResult::with_revisions(graph, revisions.to_vec())
                    }
                    None => WorkspaceGraphResult::new(graph),
                })
            }
        }),
        is_relevant: Box::new(|_| true),
    });
    let state =
        GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace)).with_workspace_graph(Some(hooks));
    let root_a = PathBuf::from("/workspace/a");
    let root_b = PathBuf::from("/workspace/b");
    let first_change = root_a.join("changed.rs");
    let newer_change = root_b.join("newer.rs");
    state
        .build_workspace_graph(&root_a, None)
        .expect("install initial graph");
    state.tag_workspace_graph_dirty(std::slice::from_ref(&first_change));

    let rebuilding = state.clone();
    let rebuild_thread = std::thread::spawn(move || rebuilding.ensure_workspace_graph_fresh());
    rebuild_started.wait();

    let newer = state
        .prepare_workspace_graph(&root_b, None, WorkspaceGraphChanges::Full)
        .expect("prepare newer activation");
    state.commit_workspace_graph(newer);
    state.tag_workspace_graph_dirty(std::slice::from_ref(&newer_change));
    release_rebuild.wait();
    rebuild_thread.join().expect("rebuild thread");

    assert_eq!(state.active_workspace_root(), Some(root_b.clone()));
    {
        let pending = read_lock(&state.pending_rebuild);
        let pending = pending.as_ref().expect("new-generation event remains");
        assert_eq!(pending.target.root, root_b);
        assert_eq!(
            pending.changed_paths,
            BTreeSet::from([newer_change.clone()])
        );
    }
    state.ensure_workspace_graph_fresh();
    assert!(read_lock(&state.pending_rebuild).is_none());
    assert!(state.rebuild_error_note().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(
        mutex_lock(&requests).as_slice(),
        [
            (root_a.clone(), WorkspaceGraphChanges::Full),
            (root_a, WorkspaceGraphChanges::Changed(vec![first_change]),),
            (root_b.clone(), WorkspaceGraphChanges::Full),
            (root_b, WorkspaceGraphChanges::Changed(vec![newer_change]),),
        ]
    );
}
