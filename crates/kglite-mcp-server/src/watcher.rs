//! Filesystem-watch wiring: which changed paths a mode accepts, how they
//! are enqueued against the active workspace target, and watcher spawn.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use mcp_methods::server::{maybe_watch, watch};

use crate::tools::{absolute_lexical_path, GraphState, WorkspaceGraphTarget};
use crate::*;

/// Build the exact non-blocking callback used by workspace watch modes.
/// Filtering happens before paths enter GraphState's target-bound pending set.
pub(crate) fn accepted_watch_paths(
    graph_state: &GraphState,
    target: &WorkspaceGraphTarget,
    paths: &[PathBuf],
) -> Vec<PathBuf> {
    paths
        .iter()
        .filter_map(|path| absolute_lexical_path(path))
        .filter(|path| path.starts_with(target.root()) && graph_state.is_graph_relevant(path))
        .collect()
}

pub(crate) fn enqueue_watch_paths(
    graph_state: &GraphState,
    observed: WorkspaceGraphTarget,
    paths: &[PathBuf],
) {
    let accepted = accepted_watch_paths(graph_state, &observed, paths);
    if accepted.is_empty() {
        return;
    }
    let _ = graph_state.bind_workspace_graph_changes(&observed, &accepted);
}

pub(crate) fn mode_change_handler(
    mode: &Mode,
    graph_state: &GraphState,
) -> Option<watch::ChangeHandler> {
    match mode {
        Mode::Watch { .. } => {
            let gs = graph_state.clone();
            Some(Arc::new(move |paths| {
                let Some(target) = gs.workspace_target_receipt() else {
                    return;
                };
                enqueue_watch_paths(&gs, target, paths);
            }))
        }
        Mode::LocalWorkspace { watch: true, .. } => {
            // Hand mcp-methods the wide `workspace.root` to monitor —
            // FSEvents/inotify only emit events for files inside the
            // subtree, so watching wide is cheap. Filtering happens
            // in the callback.
            //
            // Operator inbox 2026-05-25: pre-fix this captured
            // `workspace.root` and rebuilt the entire wide tree on
            // every event (build storm on any `cargo build` /
            // editor save anywhere in the sandbox). Fix: read the
            // committed graph identity, skip when nothing changed under the
            // active root, and rebuild against that root only.
            let gs = graph_state.clone();
            Some(Arc::new(move |paths| {
                let Some(target) = gs.workspace_target_receipt() else {
                    // No `set_root_dir` yet; nothing to rebuild.
                    return;
                };
                // Tag for rebuild; the actual rebuild fires on the
                // next MCP tool call (ensure_workspace_graph_fresh).
                enqueue_watch_paths(&gs, target, paths);
            }))
        }
        _ => None,
    }
}

/// Watch handler: rebuild on every debounced change batch. Both explicit
/// `--watch DIR` and watched local-workspace mode use the same tested callback
/// construction. Returns the handle kept alive for the server lifetime.
pub(crate) fn resolved_mode_watch_root(mode: &Mode) -> Result<Option<PathBuf>> {
    let root = match mode {
        Mode::Watch { dir } => dir,
        Mode::LocalWorkspace {
            root, watch: true, ..
        } => root,
        _ => return Ok(None),
    };
    root.canonicalize()
        .with_context(|| format!("failed to canonicalize watch root {}", root.display()))
        .map(Some)
}

pub(crate) fn spawn_mode_watcher(
    mode: &Mode,
    graph_state: &GraphState,
) -> Result<Option<watch::WatchHandle>> {
    let handler = mode_change_handler(mode, graph_state);
    let root = resolved_mode_watch_root(mode)?;
    maybe_watch(root.as_deref(), handler)
}

#[cfg(test)]
mod watcher_change_tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::tools::GraphState;

    use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
    use std::sync::Mutex;

    type CapturedRequest = (PathBuf, WorkspaceGraphChanges);

    fn recording_state(mode: WorkspaceGraphMode) -> (GraphState, Arc<Mutex<Vec<CapturedRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let hooks = WorkspaceGraphHooks {
            build: Box::new({
                let requests = requests.clone();
                move |request| {
                    requests
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push((request.root().to_path_buf(), request.changes().clone()));
                    let graph = new_dir_graph_in_mode(StorageMode::Memory, None)
                        .map(Arc::new)
                        .map_err(|error| error.to_string())?;
                    Ok(match request.revisions() {
                        Some(revisions) => {
                            WorkspaceGraphResult::with_revisions(graph, revisions.to_vec())
                        }
                        None => WorkspaceGraphResult::new(graph),
                    })
                }
            }),
            is_relevant: Box::new(|change| {
                change
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "rs" || extension == "py")
            }),
        };
        (
            GraphState::new(Some(mode)).with_workspace_graph(Some(Arc::new(hooks))),
            requests,
        )
    }

    #[test]
    fn watch_mode_filters_outside_root_and_irrelevant_paths() {
        let workspace = tempfile::tempdir().expect("watch root");
        let root = workspace.path().to_path_buf();
        let outside = root
            .parent()
            .expect("tempdir parent")
            .join("outside-phase2.rs");
        let accepted = root.join("src/accepted.rs");
        let irrelevant = root.join("notes.txt");
        let (state, requests) = recording_state(WorkspaceGraphMode::Watch);
        state
            .build_workspace_graph(&root, None)
            .expect("initial watch graph");
        let handler =
            mode_change_handler(&Mode::Watch { dir: root.clone() }, &state).expect("watch handler");

        handler(&[outside.clone(), irrelevant.clone()]);
        state.ensure_workspace_graph_fresh();
        assert_eq!(
            requests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            1,
            "empty filtered batch does not enqueue"
        );
        handler(&[outside, irrelevant, accepted.clone()]);
        state.ensure_workspace_graph_fresh();

        let requests = requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1],
            (root, WorkspaceGraphChanges::Changed(vec![accepted]),)
        );
    }

    #[test]
    fn local_watch_filters_outside_active_root_and_irrelevant_paths() {
        let workspace = tempfile::tempdir().expect("wide root");
        let wide = workspace.path().to_path_buf();
        let active = wide.join("active");
        std::fs::create_dir(&active).expect("active root");
        let accepted = active.join("accepted.py");
        let irrelevant = active.join("notes.txt");
        let sibling = wide.join("sibling.py");
        let (state, requests) = recording_state(WorkspaceGraphMode::LocalWorkspace);
        state
            .build_workspace_graph(&active, None)
            .expect("initial local graph");
        let handler = mode_change_handler(
            &Mode::LocalWorkspace {
                root: wide,
                watch: true,
            },
            &state,
        )
        .expect("local watch handler");

        handler(&[sibling.clone(), irrelevant.clone()]);
        state.ensure_workspace_graph_fresh();
        assert_eq!(
            requests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            1,
            "empty filtered batch does not enqueue"
        );
        handler(&[sibling, irrelevant, accepted.clone()]);
        state.ensure_workspace_graph_fresh();

        let requests = requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1],
            (active, WorkspaceGraphChanges::Changed(vec![accepted]),)
        );
    }

    #[test]
    fn watch_mode_normalizes_relative_deleted_paths() {
        let workspace = tempfile::Builder::new()
            .prefix("kglite-relative-watch-")
            .tempdir_in(".")
            .expect("relative watch root");
        let cwd = std::env::current_dir().expect("current directory");
        let relative_root = workspace
            .path()
            .strip_prefix(&cwd)
            .expect("tempdir is beneath cwd")
            .to_path_buf();
        assert!(relative_root.is_relative(), "fixture root stays relative");
        let absolute_root = relative_root.canonicalize().expect("absolute root");
        let relative_deleted = relative_root.join("deleted.rs");
        let absolute_deleted = absolute_root.join("deleted.rs");
        assert!(!relative_deleted.exists(), "fixture models a deletion");
        let (state, requests) = recording_state(WorkspaceGraphMode::Watch);
        state
            .build_workspace_graph(&absolute_root, None)
            .expect("initial watch graph");
        let handler = mode_change_handler(&Mode::Watch { dir: relative_root }, &state)
            .expect("watch handler");

        handler(std::slice::from_ref(&relative_deleted));
        state.ensure_workspace_graph_fresh();

        let requests = requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(requests.len(), 2, "relative event triggers one rebuild");
        assert_eq!(
            requests[1],
            (
                absolute_root,
                WorkspaceGraphChanges::Changed(vec![absolute_deleted]),
            )
        );
    }

    #[cfg(unix)]
    #[test]
    fn watch_mode_accepts_deleted_paths_beneath_a_symlink_root() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().expect("workspace tempdir");
        let real = workspace.path().join("real");
        let link = workspace.path().join("link");
        std::fs::create_dir(&real).expect("real root");
        symlink(&real, &link).expect("symlink root");
        let canonical_root = real.canonicalize().expect("canonical real root");
        let deleted_through_link = link.join("deleted.rs");
        let expected = canonical_root.join("deleted.rs");
        let (state, requests) = recording_state(WorkspaceGraphMode::Watch);
        state
            .build_workspace_graph(&canonical_root, None)
            .expect("initial graph uses canonical root");
        let mode = Mode::Watch { dir: link.clone() };
        let watched_root = resolved_mode_watch_root(&mode)
            .expect("resolve watch root")
            .expect("watch mode has root");
        assert_eq!(watched_root, canonical_root);
        assert_eq!(
            resolved_mode_watch_root(&Mode::LocalWorkspace {
                root: link,
                watch: true,
            })
            .expect("resolve local-wide watch root"),
            Some(canonical_root.clone()),
            "local-wide watcher also receives the canonical directory"
        );
        let handler = mode_change_handler(&mode, &state).expect("watch handler");

        let emitted_deleted = watched_root.join(
            deleted_through_link
                .file_name()
                .expect("deleted path has suffix"),
        );
        handler(std::slice::from_ref(&emitted_deleted));
        state.ensure_workspace_graph_fresh();

        let requests = requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(requests.len(), 2, "symlink-root event triggers rebuild");
        assert_eq!(
            requests[1],
            (
                canonical_root,
                WorkspaceGraphChanges::Changed(vec![expected]),
            )
        );
    }

    #[test]
    fn plain_watch_root_remains_canonical() {
        let workspace = tempfile::tempdir().expect("watch root");
        let root = workspace.path().canonicalize().expect("canonical root");
        assert_eq!(
            resolved_mode_watch_root(&Mode::Watch { dir: root.clone() }).expect("resolve root"),
            Some(root)
        );
    }

    #[test]
    fn stale_callback_binds_across_a_compatible_activation() {
        let workspace = tempfile::tempdir().expect("watch root");
        let root = workspace.path().to_path_buf();
        let changed = root.join("changed.rs");
        let (state, requests) = recording_state(WorkspaceGraphMode::Watch);
        state
            .build_workspace_graph(&root, None)
            .expect("install generation one");
        let observed = state.workspace_target_receipt().expect("old receipt");

        let prepared = state
            .prepare_workspace_graph(&root, None, WorkspaceGraphChanges::Full)
            .expect("prepare compatible activation");
        state.commit_workspace_graph(prepared);
        enqueue_watch_paths(&state, observed, std::slice::from_ref(&changed));
        state.ensure_workspace_graph_fresh();

        let requests = requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(requests.len(), 3);
        assert_eq!(
            requests[2],
            (root, WorkspaceGraphChanges::Changed(vec![changed]))
        );
    }

    #[test]
    fn stale_callback_does_not_bind_across_changed_revisions() {
        let workspace = tempfile::tempdir().expect("watch root");
        let root = workspace.path().to_path_buf();
        let changed = root.join("changed.rs");
        let (state, requests) = recording_state(WorkspaceGraphMode::Watch);
        state
            .build_workspace_graph(&root, None)
            .expect("install working-tree target");
        let observed = state.workspace_target_receipt().expect("old receipt");

        let prepared = state
            .prepare_workspace_graph(
                &root,
                Some(&["HEAD".to_string()]),
                WorkspaceGraphChanges::Full,
            )
            .expect("prepare revision target");
        state.commit_workspace_graph(prepared);
        enqueue_watch_paths(&state, observed, std::slice::from_ref(&changed));
        state.ensure_workspace_graph_fresh();

        assert_eq!(
            requests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            2,
            "a working-tree event cannot follow a changed revision snapshot"
        );
    }
}
