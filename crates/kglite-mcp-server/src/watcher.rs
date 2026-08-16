//! Filesystem-watch wiring: which changed paths a mode accepts, how they
//! are enqueued against the active workspace target, and watcher spawn.

use std::path::{Path, PathBuf};
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

/// What an armed `extensions.graph_watch` watcher observes in `--graph` mode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphWatchTarget {
    /// The canonicalized `.kgl` file. The callback marks a reload for this
    /// path and nothing else.
    pub(crate) file: PathBuf,
    /// Its parent directory — what the OS watcher is actually pointed at.
    /// mcp-methods' watch API is directory-only, and a file-level watch would
    /// miss the republish anyway: kglite (like most producers) writes a
    /// sibling temp file and renames it over the target, so the inode the
    /// watch was registered on is the one that gets replaced.
    pub(crate) dir: PathBuf,
}

/// Resolve the watch target for `extensions.graph_watch`, or `None` (with a
/// boot warning explaining why) when this deployment cannot be watched.
///
/// Declining is deliberate in two cases:
/// - **Not `--graph` mode.** Every other mode either has its own producer-driven
///   freshness lifecycle or no graph file at all; the key is ignored, loudly.
/// - **A disk-graph directory.** Those are a directory of retained mmaps behind
///   a `CURRENT` pointer, not one file republished atomically, so "the file
///   changed" has no single-event meaning. `reload_graph` still covers them.
pub(crate) fn resolve_graph_watch_target(mode: &Mode, enabled: bool) -> Option<GraphWatchTarget> {
    if !enabled {
        return None;
    }
    let Mode::Graph { path } = mode else {
        tracing::warn!(
            "extensions.graph_watch applies to --graph mode only — ignored for this server"
        );
        return None;
    };
    if path.is_dir() {
        tracing::warn!(
            path = %path.display(),
            "extensions.graph_watch supports single-file graphs only; the served graph is a \
             disk-graph directory — no watcher started (use the reload_graph tool)"
        );
        return None;
    }
    let file = match path.canonicalize() {
        Ok(file) => file,
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "extensions.graph_watch: cannot resolve the served graph file — no watcher started"
            );
            return None;
        }
    };
    let Some(dir) = file.parent().map(Path::to_path_buf) else {
        tracing::warn!(
            path = %file.display(),
            "extensions.graph_watch: the served graph file has no parent directory to watch"
        );
        return None;
    };
    Some(GraphWatchTarget { file, dir })
}

/// The `--graph` watch callback: mark the state for reload when *this* file
/// changed, and do nothing else.
///
/// The exact-path filter is what makes an atomic republish cost one reload.
/// kglite's own save writes `<name>.kgl.tmp.<pid>.<nonce>` beside the target
/// and renames it over, and the writer lease churns `<name>.kgl.lock`; every
/// one of those is a sibling in the watched directory, and all of them coalesce
/// to nothing here. The reload itself runs later, on the next tool call — a
/// watch callback runs on the debouncer thread and must not block.
pub(crate) fn graph_change_handler(
    target: &GraphWatchTarget,
    graph_state: &GraphState,
) -> watch::ChangeHandler {
    let gs = graph_state.clone();
    let file = target.file.clone();
    Arc::new(move |paths: &[PathBuf]| {
        let touched = paths
            .iter()
            .filter_map(|path| absolute_lexical_path(path))
            .any(|path| path == file);
        if !touched {
            return;
        }
        gs.mark_graph_reload_pending();
    })
}

pub(crate) fn spawn_mode_watcher(
    mode: &Mode,
    graph_state: &GraphState,
    graph_watch: bool,
) -> Result<Option<watch::WatchHandle>> {
    if let Some(target) = resolve_graph_watch_target(mode, graph_watch) {
        // Deliberately NOT `maybe_watch`: its default `WatchConfig` drops any
        // path containing `/build/`, `/dist/`, `/target/`, `/node_modules/`,
        // `/.venv/` or `/.git/`, and any `.tmp` extension. Those defaults are
        // right for source trees and wrong for a data artifact — a `.kgl`
        // generated into a build directory is an ordinary deployment, and it
        // would be silently invisible. The callback already narrows to one
        // exact path, so an unfiltered config is both correct and cheap.
        let handler = graph_change_handler(&target, graph_state);
        tracing::info!(
            file = %target.file.display(),
            dir = %target.dir.display(),
            "extensions.graph_watch: watching the served graph file for external rewrites"
        );
        return watch::watch_with_config(
            &target.dir,
            Some(handler),
            None,
            watch::WatchConfig::unfiltered(),
        )
        .map(Some);
    }
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

    /// The nine tool call sites now call the mode-dispatching
    /// [`GraphState::ensure_graph_fresh`] rather than the workspace method
    /// directly. A workspace mode must still reach the producer rebuild
    /// through it — every other test in this module is the regression net for
    /// the rest of that path, this one pins the dispatch itself.
    #[test]
    fn ensure_graph_fresh_dispatches_workspace_modes_to_the_rebuild_path() {
        let workspace = tempfile::tempdir().expect("watch root");
        let root = workspace.path().to_path_buf();
        let changed = root.join("changed.rs");
        let (state, requests) = recording_state(WorkspaceGraphMode::Watch);
        state
            .build_workspace_graph(&root, None)
            .expect("initial watch graph");
        let handler =
            mode_change_handler(&Mode::Watch { dir: root.clone() }, &state).expect("watch handler");

        handler(std::slice::from_ref(&changed));
        state.ensure_graph_fresh();

        let requests = requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            requests.len(),
            2,
            "ensure_graph_fresh must reach the workspace producer"
        );
        assert_eq!(
            requests[1],
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

/// `--graph` mode's opt-in watcher (`extensions.graph_watch: true`): which
/// events mark the served file for reload, and what the lazy re-read does with
/// that mark on the next tool call.
#[cfg(test)]
mod graph_watch_tests {
    use super::*;

    use kglite::api::session::{execute_mut, ExecuteOptions};
    use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};

    use crate::tools::{GraphState, MAX_CONSECUTIVE_REBUILD_FAILURES};

    /// Write a `.kgl` holding `nodes` nodes — the artifact an external producer
    /// publishes. Built through the engine rather than a fixture blob so the
    /// bytes are whatever the current format writes.
    fn seed_kgl(path: &Path, nodes: u64) {
        let mut dir = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create graph");
        let params = std::collections::HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        for i in 0..nodes {
            execute_mut(&mut dir, &format!("CREATE (:N {{id:'{i}'}})"), &opts).expect("seed node");
        }
        let mut graph = Arc::new(dir);
        kglite::api::io::save_graph(&mut graph, path.to_str().expect("utf-8 fixture path"))
            .expect("save seeded graph");
    }

    /// A `--graph` server's state: no workspace mode, one file open.
    fn serving(path: &Path) -> GraphState {
        let state = GraphState::new(None);
        state.open_or_create(path, None).expect("serve the graph");
        state
    }

    fn nodes(state: &GraphState) -> u64 {
        state.schema().expect("a graph is active").0
    }

    fn generation(state: &GraphState) -> u64 {
        state.generation().expect("a graph is active")
    }

    #[test]
    fn only_the_served_file_marks_a_reload() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let served = tmp.path().join("served.kgl");
        seed_kgl(&served, 2);
        let state = serving(&served);
        let canonical = served.canonicalize().expect("canonical served path");

        let target = resolve_graph_watch_target(
            &Mode::Graph {
                path: served.clone(),
            },
            true,
        )
        .expect("a regular-file graph is watchable");
        assert_eq!(target.file, canonical);
        assert_eq!(
            target.dir,
            canonical.parent().expect("served file has a parent"),
            "mcp-methods watches directories; the parent is the registration target"
        );
        let handler = graph_change_handler(&target, &state);
        let dir = target.dir.clone();

        // Exactly the sibling churn an atomic republish produces: kglite writes
        // `<name>.kgl.tmp.<pid>.<nonce>` and renames it over, and the writer
        // lease touches `<name>.kgl.lock`. None of it is the served file.
        handler(&[
            dir.join("served.kgl.tmp.44213.9f3ab1"),
            dir.join("served.kgl.lock"),
            dir.join("unrelated.kgl"),
            dir.join("notes.txt"),
        ]);
        assert!(
            !state.graph_reload_pending(),
            "sibling temp/lock churn and unrelated files must not mark a reload"
        );

        // The same batch with the real target in it marks exactly once.
        handler(&[
            dir.join("served.kgl.tmp.44213.9f3ab1"),
            canonical.clone(),
            dir.join("notes.txt"),
        ]);
        assert!(
            state.graph_reload_pending(),
            "an event on the served file must mark the reload"
        );
    }

    #[test]
    fn a_marked_reload_serves_the_rewritten_file_on_the_next_call() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let served = tmp.path().join("served.kgl");
        let rebuilt = tmp.path().join("rebuilt.kgl");
        seed_kgl(&served, 2);
        seed_kgl(&rebuilt, 5);
        let state = serving(&served);
        let canonical = served.canonicalize().expect("canonical served path");
        let before = generation(&state);
        let handler = graph_change_handler(
            &resolve_graph_watch_target(
                &Mode::Graph {
                    path: served.clone(),
                },
                true,
            )
            .expect("watchable"),
            &state,
        );

        // An external producer republishes the path (rename-over, as kglite's
        // own save does), and the watcher reports it.
        std::fs::rename(&rebuilt, &served).expect("republish");
        handler(std::slice::from_ref(&canonical));
        assert_eq!(
            nodes(&state),
            2,
            "the callback must not load anything itself — the reload is lazy"
        );
        assert_eq!(generation(&state), before);

        state.ensure_graph_fresh();
        assert_eq!(nodes(&state), 5, "the next tool call must serve new bytes");
        assert_eq!(
            generation(&state),
            before + 1,
            "a completed reload installs a graph and bumps the generation"
        );
        assert!(!state.graph_reload_pending(), "the mark was consumed");

        state.ensure_graph_fresh();
        assert_eq!(
            generation(&state),
            before + 1,
            "a consumed mark must not reload again on every later call"
        );
    }

    #[test]
    fn repeated_reload_failures_keep_the_old_graph_then_go_dormant() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let served = tmp.path().join("served.kgl");
        let rebuilt = tmp.path().join("rebuilt.kgl");
        seed_kgl(&served, 3);
        let state = serving(&served);
        let canonical = served.canonicalize().expect("canonical served path");
        let before = generation(&state);
        let handler = graph_change_handler(
            &resolve_graph_watch_target(
                &Mode::Graph {
                    path: served.clone(),
                },
                true,
            )
            .expect("watchable"),
            &state,
        );

        // A producer writing non-atomically (or a truncated copy) leaves bytes
        // that cannot be opened.
        std::fs::write(&served, b"not a kgl file").expect("torn republish");
        handler(std::slice::from_ref(&canonical));

        for expected in 1..=MAX_CONSECUTIVE_REBUILD_FAILURES {
            state.ensure_graph_fresh();
            assert_eq!(
                state.graph_reload_failures(),
                expected,
                "each failed reload counts once"
            );
            assert_eq!(
                nodes(&state),
                3,
                "a failed reload must leave the previous graph serving"
            );
            assert_eq!(
                generation(&state),
                before,
                "a failed reload installs nothing"
            );
        }
        assert!(
            !state.graph_reload_pending(),
            "the failure cap stops the per-call retry"
        );
        let note = state
            .rebuild_error_note()
            .expect("the stale graph must be flagged on tool output");
        assert!(
            note.contains("graph reload failed")
                && note.contains("3 consecutive failure(s)")
                && note.contains("dormant"),
            "the note must name the failure and the dormancy: {note}"
        );

        // A further event while dormant must not attempt another load — asserted
        // through the failure counter, which a fourth attempt would increment.
        handler(std::slice::from_ref(&canonical));
        assert!(
            !state.graph_reload_pending(),
            "events are ignored while dormant"
        );
        state.ensure_graph_fresh();
        assert_eq!(
            state.graph_reload_failures(),
            MAX_CONSECUTIVE_REBUILD_FAILURES,
            "no load may be attempted while dormant"
        );

        // The documented escape hatch: what the `reload_graph` tool calls.
        seed_kgl(&rebuilt, 7);
        std::fs::rename(&rebuilt, &served).expect("republish readable bytes");
        state
            .open_or_create(&served, None)
            .expect("a manual reload of readable bytes succeeds");
        assert_eq!(nodes(&state), 7);
        assert_eq!(
            state.graph_reload_failures(),
            0,
            "a successful open resets the counter"
        );
        assert!(
            state.rebuild_error_note().is_none(),
            "and clears the stale-graph warning"
        );
        handler(std::slice::from_ref(&canonical));
        assert!(
            state.graph_reload_pending(),
            "dormancy must be lifted by the successful reload"
        );
    }

    #[test]
    fn graph_watch_declines_disk_directories_other_modes_and_the_default() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let served = tmp.path().join("served.kgl");
        seed_kgl(&served, 1);
        let disk_graph = tmp.path().join("disk-graph");
        std::fs::create_dir(&disk_graph).expect("disk-graph dir");

        assert!(
            resolve_graph_watch_target(
                &Mode::Graph {
                    path: served.clone()
                },
                false
            )
            .is_none(),
            "the watcher is opt-in: absent/false manifest key arms nothing"
        );
        assert!(
            resolve_graph_watch_target(&Mode::Graph { path: disk_graph }, true).is_none(),
            "a disk-graph directory is not a single republished file"
        );
        assert!(
            resolve_graph_watch_target(
                &Mode::Watch {
                    dir: tmp.path().to_path_buf()
                },
                true
            )
            .is_none(),
            "other modes own their own freshness lifecycle"
        );
        assert!(
            resolve_graph_watch_target(&Mode::Graph { path: served }, true).is_some(),
            "the control arm: a regular-file graph in --graph mode IS watchable"
        );
    }
}
