//! Workspace request/change/hook surface tests.

use std::path::Path;
use std::sync::{Arc, Mutex};

use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};

use super::*;

#[test]
fn workspace_requests_distinguish_full_from_changed_paths() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(recording_hooks(requests.clone())));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path();

    state
        .build_workspace_graph(root, None)
        .expect("plain build");
    state
        .build_workspace_graph(root, Some(&["HEAD".into()]))
        .expect("revision build");
    state.tag_workspace_graph_dirty(&[root.join("changed.rs")]);
    state.ensure_workspace_graph_fresh();

    let requests = mutex_lock(&requests);
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].1, WorkspaceGraphChanges::Full);
    assert_eq!(requests[1].1, WorkspaceGraphChanges::Full);
    assert_eq!(
        requests[2].1,
        WorkspaceGraphChanges::Changed(vec![root.join("changed.rs")])
    );
}

#[test]
fn workspace_change_batches_coalesce_deterministically_and_keep_deleted_paths() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(recording_hooks(requests.clone())));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path();
    state
        .build_workspace_graph(root, None)
        .expect("initial build");

    let a = root.join("a.py");
    let deleted = root.join("deleted.rs");
    let z = root.join("z.rs");
    assert!(!deleted.exists(), "fixture path represents a deletion");
    state.tag_workspace_graph_dirty(&[z.clone(), a.clone(), z.clone()]);
    state.tag_workspace_graph_dirty(&[deleted.clone(), a.clone()]);
    state.ensure_workspace_graph_fresh();
    state.ensure_workspace_graph_fresh();

    let requests = mutex_lock(&requests);
    assert_eq!(requests.len(), 2, "coalesced batches cause one rebuild");
    assert_eq!(
        requests[1].1,
        WorkspaceGraphChanges::Changed(vec![a, deleted, z])
    );
}

#[test]
fn workspace_graph_hooks_unify_builds_and_own_relevance() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let build_calls = Arc::new(AtomicUsize::new(0));
    let calls = build_calls.clone();
    let hooks = WorkspaceGraphHooks {
        build: Box::new(move |request| {
            calls.fetch_add(1, Ordering::SeqCst);
            new_dir_graph_in_mode(StorageMode::Memory, None)
                .map(Arc::new)
                .map(|graph| match request.revisions() {
                    Some(revisions) => {
                        WorkspaceGraphResult::with_revisions(graph, revisions.to_vec())
                    }
                    None => WorkspaceGraphResult::new(graph),
                })
                .map_err(|e| e.to_string())
        }),
        is_relevant: Box::new(|change| change.path().extension().is_some_and(|e| e == "zig")),
    };
    let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(Arc::new(hooks)));

    // Watch predicate comes from the hook, not language_for_path
    // (in-tree has no zig parser; hook says only zig is code).
    assert!(gs.is_graph_relevant(Path::new("a.zig")));
    assert!(!gs.is_graph_relevant(Path::new("a.rs")));

    // build goes through the hook and swaps in the returned graph.
    gs.build_workspace_graph(Path::new("/nonexistent-is-fine-for-hook"), None)
        .expect("hook build");
    assert_eq!(build_calls.load(Ordering::SeqCst), 1);
    assert!(gs.schema().is_some(), "hook-built graph became active");

    // revs path records the hook's canonical rev list.
    gs.build_workspace_graph(
        Path::new("/nonexistent-is-fine-for-hook"),
        Some(&["a".into(), "b".into()]),
    )
    .expect("revision-set hook build");
}

#[test]
fn without_hooks_nothing_is_relevant_and_builds_refuse() {
    // The in-tree builder is gone: a hook-less state can't rebuild, so
    // no path is graph-relevant and build requests refuse with a
    // pointer at codingest.
    let gs = GraphState::new(Some(WorkspaceGraphMode::Workspace));
    assert!(!gs.is_graph_relevant(Path::new("a.rs")));
    assert!(!gs.is_graph_relevant(Path::new("README.md")));
    let err = gs
        .build_workspace_graph(Path::new("/tmp"), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("codingest"), "refusal names codingest: {err}");
    let err = gs
        .build_workspace_graph(Path::new("/tmp"), Some(&["r1".into()]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("codingest"), "refusal names codingest: {err}");
    // The producer, not KGLite, chooses that markdown is relevant in
    // clone-backed workspace mode.
    let gs_docs = GraphState::new(Some(WorkspaceGraphMode::Workspace))
        .with_workspace_graph(Some(test_hooks()));
    assert!(gs_docs.is_graph_relevant(Path::new("README.md")));
}
