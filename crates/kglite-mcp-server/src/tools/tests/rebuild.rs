//! Lazy workspace-rebuild failure, back-off and change-binding tests.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use super::*;

#[test]
fn save_does_not_deep_copy_the_active_graph() {
    // `run_save` / `save_as` must save through the active graph's OWN
    // Arc so `prepare_save`'s `Arc::make_mut` sees refcount 1. The old
    // `kg.dir().clone()` route deep-copied the entire graph on every
    // save (and threw the columnar consolidation away with the clone).
    // Pin the fix by asserting the DirGraph allocation is pointer-
    // identical across the save.
    let mut active = fresh_active();
    let path = std::env::temp_dir().join(format!(
        "kglite-mcp-save-noclone-{}.kgl",
        std::process::id()
    ));
    active.source_path = Some(path.clone());
    // The real write sequence: ownership takes its pristine snapshot on the
    // first write and the mutation forks the live graph away from it, so the
    // snapshot `publish` still holds during the save is the *old* allocation
    // and the save runs at refcount 1. Pins that a held snapshot never makes
    // a save fork.
    let mut ownership = kglite::api::io::WriteOwnership::new(
        path.clone(),
        kglite::api::io::GraphFileIdentity::capture(&path).expect("capture a missing path"),
        active.kg.dir(),
        None,
        true,
    );
    ownership
        .begin_write(active.kg.dir_mut())
        .expect("first write on an uncontended path");
    active.ownership = Some(ownership);
    {
        // Put something in the graph so the save isn't trivially empty.
        let dir = kglite::api::make_dir_graph_mut(active.kg.dir_mut());
        let opts_params = std::collections::HashMap::new();
        let opts = kglite::api::session::ExecuteOptions::eager(&opts_params);
        kglite::api::session::execute_mut(
            dir,
            "CREATE (a:Thing {id: 1, name: 'a'})-[:REL]->(b:Thing {id: 2, name: 'b'})",
            &opts,
        )
        .expect("seed mutation");
    }

    let before = Arc::as_ptr(active.kg.dir());
    let msg = run_save(&mut active, false, true).expect("save must succeed");
    assert!(msg.starts_with("Saved"), "save must succeed: {msg}");
    assert_eq!(
        Arc::as_ptr(active.kg.dir()),
        before,
        "save must not deep-copy the active graph (refcount must be 1 \
         at prepare_save's Arc::make_mut)"
    );

    // save_as: same invariant, and the save target must rebind.
    let path2 = std::env::temp_dir().join(format!(
        "kglite-mcp-saveas-noclone-{}.kgl",
        std::process::id()
    ));
    let state = GraphState::new(None);
    *write_lock(&state.inner) = Some(active);
    let msg = state.save_as(&path2).expect("save_as succeeds");
    assert!(msg.starts_with("Saved"), "{msg}");
    {
        let guard = read_lock(&state.inner);
        let active = guard.as_ref().expect("still active");
        assert_eq!(
            Arc::as_ptr(active.kg.dir()),
            before,
            "save_as must not deep-copy the active graph"
        );
        assert_eq!(active.source_path.as_deref(), Some(path2.as_path()));
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&path2);
}

#[test]
fn failed_rebuild_restores_marker_then_backs_off_after_cap() {
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    std::fs::write(root.join("m.py"), "def stale():\n    return 1\n").unwrap();
    state
        .build_workspace_graph(&root, None)
        .expect("install the graph that later becomes stale");
    let target = state.active_workspace_target().expect("workspace target");
    std::fs::remove_dir_all(&root).expect("make the current target fail rebuilding");

    // A failed rebuild must restore the dirty marker (so the next tool
    // call retries) and record the error — up to the hot-fail cap.
    state.tag_workspace_graph_dirty(&[root.join("m.py")]);
    for failure in 1..MAX_CONSECUTIVE_REBUILD_FAILURES {
        state.ensure_workspace_graph_fresh();
        assert_eq!(
            read_lock(&state.pending_rebuild)
                .as_ref()
                .map(|pending| &pending.target),
            Some(&target),
            "failure {failure} must restore the marker for a retry"
        );
        let note = state.rebuild_error_note().expect("error recorded");
        assert!(note.contains("STALE"), "note flags staleness: {note}");
        let failure = state
            .workspace_rebuild_failure()
            .expect("typed failure snapshot recorded");
        assert_eq!(failure.reason.code(), "rebuild_failed");
        assert_eq!(failure.retry_limit, MAX_CONSECUTIVE_REBUILD_FAILURES);
    }

    // Failure #cap: retain the marker as dormant — the stale graph keeps
    // being served with the error surfaced, without automatic retries.
    state.ensure_workspace_graph_fresh();
    {
        let pending = read_lock(&state.pending_rebuild);
        assert!(
            !pending
                .as_ref()
                .expect("capped paths remain retained")
                .ready,
            "after {MAX_CONSECUTIVE_REBUILD_FAILURES} consecutive failures \
             the marker must be dormant (no hot-fail loop)"
        );
    }
    let note = state.rebuild_error_note().expect("error still surfaced");
    assert!(
        note.contains(&format!("{MAX_CONSECUTIVE_REBUILD_FAILURES} consecutive")),
        "note reports the failure count: {note}"
    );
    let failure = state
        .workspace_rebuild_failure()
        .expect("hot-fail snapshot recorded");
    assert_eq!(failure.reason, WorkspaceRebuildFailureReason::HotFail);
    assert_eq!(failure.reason.code(), "hot_fail");
    // A dormant marker makes further tool calls no-ops (no retry storm).
    state.ensure_workspace_graph_fresh();

    // A fresh FS event re-tags → exactly one more retry; still failing,
    // so the marker becomes dormant again.
    state.tag_workspace_graph_dirty(&[root.join("m.py")]);
    state.ensure_workspace_graph_fresh();
    assert!(
        !read_lock(&state.pending_rebuild)
            .as_ref()
            .expect("failed retry remains retained")
            .ready
    );
    assert!(state.rebuild_error_note().is_some());

    // A successful rebuild clears the error and resets the counter.
    std::fs::create_dir_all(&root).expect("restore workspace directory");
    std::fs::write(root.join("m.py"), "def ok():\n    return 1\n").unwrap();
    state.tag_workspace_graph_dirty(&[root.join("m.py")]);
    state.ensure_workspace_graph_fresh();
    assert!(
        state.rebuild_error_note().is_none(),
        "successful rebuild must clear the recorded failure"
    );
    assert!(read_lock(&state.pending_rebuild).is_none());
}

#[test]
fn failed_rebuild_unions_consumed_and_newer_changes() {
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
                    return Err("injected rebuild failure".into());
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
    let target = state.active_workspace_target().expect("active target");
    {
        let mut status = write_lock(&state.rebuild_status);
        status.failed_target = Some(target);
        status.consecutive_failures = MAX_CONSECUTIVE_REBUILD_FAILURES - 1;
    }
    state.tag_workspace_graph_dirty(std::slice::from_ref(&first));

    let rebuilding = state.clone();
    let thread = std::thread::spawn(move || rebuilding.ensure_workspace_graph_fresh());
    rebuild_started.wait();
    state.tag_workspace_graph_dirty(std::slice::from_ref(&newer));
    release_rebuild.wait();
    thread.join().expect("failed rebuild thread");

    {
        let pending = read_lock(&state.pending_rebuild);
        assert_eq!(
            pending
                .as_ref()
                .expect("fresh event remains a retry marker")
                .changed_paths,
            BTreeSet::from([first.clone(), newer.clone()]),
            "cap failure completes an already-present fresh event"
        );
    }
    assert_eq!(
        state
            .workspace_rebuild_failure()
            .expect("cap failure recorded")
            .reason,
        WorkspaceRebuildFailureReason::HotFail
    );
    state.ensure_workspace_graph_fresh();
    let requests = mutex_lock(&requests);
    assert_eq!(
        requests.as_slice(),
        [
            WorkspaceGraphChanges::Full,
            WorkspaceGraphChanges::Changed(vec![first.clone()]),
            WorkspaceGraphChanges::Changed(vec![first, newer]),
        ]
    );
    assert!(state.rebuild_error_note().is_none(), "retry succeeded");
}

#[test]
fn hot_fail_dormant_paths_join_next_fresh_event() {
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    let first = root.join("first.rs");
    let newer = root.join("newer.rs");
    std::fs::write(&first, "fn first() {}\n").expect("seed workspace");
    state
        .build_workspace_graph(&root, None)
        .expect("install initial graph");
    let target = state.active_workspace_target().expect("active target");
    {
        let mut status = write_lock(&state.rebuild_status);
        status.failed_target = Some(target);
        status.consecutive_failures = MAX_CONSECUTIVE_REBUILD_FAILURES - 1;
    }
    std::fs::remove_dir_all(&root).expect("make rebuild fail");

    state.tag_workspace_graph_dirty(std::slice::from_ref(&first));
    state.ensure_workspace_graph_fresh();
    state.tag_workspace_graph_dirty(std::slice::from_ref(&newer));

    assert_eq!(
        read_lock(&state.pending_rebuild)
            .as_ref()
            .expect("fresh event re-arms dormant work")
            .changed_paths,
        BTreeSet::from([first, newer]),
        "a capped batch remains dormant and joins the next genuine event"
    );
}

#[test]
fn stale_filtered_paths_never_bind_to_a_new_target() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(recording_hooks(requests)));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root_a = workspace.path().join("a");
    let root_b = workspace.path().join("b");
    std::fs::create_dir_all(&root_a).expect("root A");
    std::fs::create_dir_all(&root_b).expect("root B");
    let accepted_under_a = root_a.join("changed.rs");
    state
        .build_workspace_graph(&root_a, None)
        .expect("install target A");
    let observed_a = state.active_workspace_target().expect("target A receipt");

    // Model a callback that accepted the path against A, then lost the
    // race to activation before it could enqueue the accepted batch.
    let prepared_b = state
        .prepare_workspace_graph(&root_b, None, WorkspaceGraphChanges::Full)
        .expect("prepare target B");
    state.commit_workspace_graph(prepared_b);
    let result =
        state.bind_workspace_graph_changes(&observed_a, std::slice::from_ref(&accepted_under_a));

    assert_eq!(result, WorkspaceGraphEnqueue::IncompatibleSource);
    assert!(
        read_lock(&state.pending_rebuild).is_none(),
        "a path filtered against A must never be rebound to B"
    );
}

#[test]
fn same_target_activation_preserves_post_scan_event() {
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    let changed = root.join("changed.rs");
    state
        .build_workspace_graph(&root, None)
        .expect("install initial graph");

    // The full scan has completed, but publication has not. An event in
    // this window is not represented by the prepared artifact.
    let prepared = state
        .prepare_workspace_graph(&root, None, WorkspaceGraphChanges::Full)
        .expect("prepare same-target activation");
    state.tag_workspace_graph_dirty(std::slice::from_ref(&changed));
    state.commit_workspace_graph(prepared);

    let active = state.active_workspace_target().expect("installed target");
    let pending = read_lock(&state.pending_rebuild);
    let pending = pending.as_ref().expect("post-scan event survives commit");
    assert_eq!(pending.target, active, "event retargeted to new generation");
    assert_eq!(pending.changed_paths, BTreeSet::from([changed]));
    assert!(pending.ready, "armed event stays armed across publication");
}

#[test]
fn same_source_activation_rearms_dormant_paths_for_one_retry_series() {
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    let changed = root.join("changed.rs");
    std::fs::write(&changed, "fn changed() {}\n").expect("seed workspace");
    state
        .build_workspace_graph(&root, None)
        .expect("install initial graph");
    let target = state.active_workspace_target().expect("active target");
    {
        let mut status = write_lock(&state.rebuild_status);
        status.failed_target = Some(target);
        status.consecutive_failures = MAX_CONSECUTIVE_REBUILD_FAILURES - 1;
    }
    std::fs::remove_dir_all(&root).expect("make rebuild fail");
    state.tag_workspace_graph_dirty(std::slice::from_ref(&changed));
    state.ensure_workspace_graph_fresh();
    assert!(
        !read_lock(&state.pending_rebuild)
            .as_ref()
            .expect("capped work retained")
            .ready,
        "capped work is dormant"
    );

    std::fs::create_dir_all(&root).expect("restore workspace");
    let prepared = state
        .prepare_workspace_graph(&root, None, WorkspaceGraphChanges::Full)
        .expect("prepare same-source activation");
    state.commit_workspace_graph(prepared);
    let active = state.active_workspace_target().expect("new target");
    {
        let pending = read_lock(&state.pending_rebuild);
        let pending = pending.as_ref().expect("dormant hint survives");
        assert_eq!(pending.target, active);
        assert_eq!(pending.changed_paths, BTreeSet::from([changed]));
        assert!(pending.ready, "full commit re-arms dormant work");
    }
    state.ensure_workspace_graph_fresh();
    assert!(
        read_lock(&state.pending_rebuild).is_none(),
        "fresh activation permits one successful retry"
    );
}

#[test]
fn post_prepare_dormant_change_is_armed_by_compatible_activation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let hooks = Arc::new(WorkspaceGraphHooks {
        build: Box::new({
            let calls = calls.clone();
            let requests = requests.clone();
            move |request| {
                let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                mutex_lock(&requests).push(request.changes().clone());
                if call == 3 {
                    return Err("injected capped rebuild failure".into());
                }
                empty_workspace_result(&request)
            }
        }),
        is_relevant: Box::new(|_| true),
    });
    let state =
        GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace)).with_workspace_graph(Some(hooks));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    let changed = root.join("changed.rs");
    state
        .build_workspace_graph(&root, None)
        .expect("install initial graph");

    // The full scan finishes before A arrives and becomes dormant.
    let prepared = state
        .prepare_workspace_graph(&root, None, WorkspaceGraphChanges::Full)
        .expect("prepare full activation before A");
    let target = state.active_workspace_target().expect("old target");
    {
        let mut status = write_lock(&state.rebuild_status);
        status.failed_target = Some(target);
        status.consecutive_failures = MAX_CONSECUTIVE_REBUILD_FAILURES - 1;
    }
    state.tag_workspace_graph_dirty(std::slice::from_ref(&changed));
    state.ensure_workspace_graph_fresh();
    assert!(
        !read_lock(&state.pending_rebuild)
            .as_ref()
            .expect("A retained at cap")
            .ready,
        "A is dormant before activation"
    );

    state.commit_workspace_graph(prepared);
    {
        let active = state.active_workspace_target().expect("new target");
        let pending = read_lock(&state.pending_rebuild);
        let pending = pending.as_ref().expect("A survives activation");
        assert_eq!(pending.target, active);
        assert!(pending.ready, "post-scan dormant A is re-armed");
    }
    state.ensure_workspace_graph_fresh();

    assert!(read_lock(&state.pending_rebuild).is_none());
    assert!(state.workspace_rebuild_failure().is_none());
    assert_eq!(
        mutex_lock(&requests).as_slice(),
        [
            WorkspaceGraphChanges::Full,
            WorkspaceGraphChanges::Full,
            WorkspaceGraphChanges::Changed(vec![changed.clone()]),
            WorkspaceGraphChanges::Changed(vec![changed]),
        ]
    );
}
