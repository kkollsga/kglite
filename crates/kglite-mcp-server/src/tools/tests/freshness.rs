//! Per-call freshness for a `--graph` server: the stat before every tool call,
//! what it declines to do while this server holds unsaved changes, and how a
//! failed re-read is retried.

use std::path::Path;
use std::sync::Arc;

use kglite::api::session::{execute_mut, ExecuteOptions};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};

use super::*;

/// Write a `.kgl` holding `nodes` nodes — the artifact an external producer
/// publishes. Built through the engine rather than a fixture blob so the bytes
/// are whatever the current format writes.
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

/// The headline property: a clean server never answers from a snapshot older
/// than the file was at the time of the call, with no watcher and no manifest
/// key involved.
#[test]
fn an_external_rewrite_is_served_by_the_next_call() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let served = tmp.path().join("served.kgl");
    let rebuilt = tmp.path().join("rebuilt.kgl");
    seed_kgl(&served, 2);
    seed_kgl(&rebuilt, 5);
    let state = serving(&served);
    let before = generation(&state);

    // An external producer republishes the path (rename-over, as kglite's own
    // save does).
    std::fs::rename(&rebuilt, &served).expect("republish");

    state.ensure_graph_fresh();
    assert_eq!(nodes(&state), 5, "the next tool call must serve new bytes");
    assert_eq!(
        generation(&state),
        before + 1,
        "a completed reload installs a graph and bumps the generation"
    );

    state.ensure_graph_fresh();
    assert_eq!(
        generation(&state),
        before + 1,
        "an unchanged file must not reload again on every later call"
    );
}

/// A server holding unsaved changes must not have them replaced by a peer's
/// republish — and must say so, because `save_graph` will now refuse.
#[test]
fn a_dirty_server_does_not_auto_reload_and_warns() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let served = tmp.path().join("dirty.kgl");
    let rebuilt = tmp.path().join("rebuilt.kgl");
    seed_kgl(&served, 2);
    seed_kgl(&rebuilt, 5);
    let state = serving(&served);
    let before = generation(&state);
    state
        .with_active_mut(|active| write(active, "CREATE (:N {id:'local'})", None))
        .expect("a graph is active")
        .expect("the write applies");
    assert!(state.is_dirty(), "the fixture must model an unsaved change");

    std::fs::rename(&rebuilt, &served).expect("republish under a dirty server");
    state.ensure_graph_fresh();

    assert_eq!(
        generation(&state),
        before,
        "a reload while dirty would silently discard this server's unsaved work"
    );
    assert_eq!(nodes(&state), 3, "the local mutation is still served");
    let note = state
        .rebuild_error_note()
        .expect("the divergence must reach tool output");
    assert!(
        note.contains("unsaved changes")
            && note.contains("save_graph_as")
            && note.contains("discard_unsaved=true"),
        "the note must name both ways out: {note}"
    );
}

/// This server's own save must not look like somebody else's rewrite: the
/// publish recaptures the identity, so the next call stats equal and serves
/// the graph it already has.
#[test]
fn a_save_does_not_make_the_next_call_reload_itself() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let served = tmp.path().join("self_write.kgl");
    let rebuilt = tmp.path().join("rebuilt.kgl");
    seed_kgl(&served, 2);
    seed_kgl(&rebuilt, 5);
    let state = serving(&served);
    state
        .with_active_mut(|active| write(active, "CREATE (:N {id:'local'})", None))
        .expect("a graph is active")
        .expect("the write applies");
    state
        .with_active_mut(run_save)
        .expect("a graph is active")
        .expect("the save lands");
    let after_save = generation(&state);

    state.ensure_graph_fresh();
    assert_eq!(
        generation(&state),
        after_save,
        "a server must not re-read the file it just wrote"
    );
    assert_eq!(nodes(&state), 3, "and must keep serving what it saved");
    assert!(!state.is_dirty(), "a published graph is clean");

    // The control arm: the same server DOES re-read somebody else's rewrite,
    // so the suppression above is about authorship and not about the stat
    // being asleep.
    std::fs::rename(&rebuilt, &served).expect("republish");
    state.ensure_graph_fresh();
    assert_eq!(generation(&state), after_save + 1);
    assert_eq!(nodes(&state), 5);
}

/// A torn republish must not cost every later tool call a doomed load, and a
/// producer republishing torn bytes under a *new* identity each time must not
/// buy one full failed load per republish either.
#[test]
fn a_failed_reload_retries_per_identity_behind_a_time_backstop() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let served = tmp.path().join("torn.kgl");
    seed_kgl(&served, 3);
    let state = serving(&served);
    let before = generation(&state);

    // A producer writing non-atomically (or a truncated copy) leaves bytes
    // that cannot be opened.
    std::fs::write(&served, b"not a kgl file").expect("torn republish");
    state.ensure_graph_fresh();
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
    let note = state
        .rebuild_error_note()
        .expect("the stale graph must be flagged on tool output");
    assert!(
        note.contains("graph reload failed"),
        "the note must name the failure: {note}"
    );
    let first_attempt = state
        .graph_reload_failed_at()
        .expect("a failed re-read records when it failed");

    // Same bytes, next tool call: no second attempt. A re-attempt would
    // re-stamp `failed_at`, so this reads the attempt rather than the outcome.
    state.ensure_graph_fresh();
    assert_eq!(
        state.graph_reload_failed_at(),
        Some(first_attempt),
        "the same failing bytes must not be re-loaded on every tool call"
    );

    // *Different* torn bytes — a producer republishing a truncated file in a
    // loop. New identity, so the per-identity rule alone would allow a full
    // failed load per republish; the time backstop is what stops it.
    std::fs::write(&served, b"not a kgl file either").expect("second torn republish");
    state.ensure_graph_fresh();
    assert_eq!(
        state.graph_reload_failed_at(),
        Some(first_attempt),
        "new bytes inside the retry interval must still wait"
    );

    // Past the interval, new bytes are tried again — and fail again.
    state.backdate_graph_reload_failure(GRAPH_RELOAD_RETRY_INTERVAL);
    state.ensure_graph_fresh();
    let second_attempt = state
        .graph_reload_failed_at()
        .expect("the retry records its own failure");
    assert!(
        second_attempt > first_attempt,
        "changed bytes past the interval must be retried"
    );
    assert_eq!(nodes(&state), 3, "and the old graph is still serving");

    // Past the interval but with the *same* bytes: still no retry. Both halves
    // are required, so a permanently broken file costs one load, not one per
    // interval — `reload_graph` is the way back.
    state.backdate_graph_reload_failure(GRAPH_RELOAD_RETRY_INTERVAL);
    let backdated = state.graph_reload_failed_at();
    state.ensure_graph_fresh();
    assert_eq!(
        state.graph_reload_failed_at(),
        backdated,
        "unchanged bytes are never retried automatically, however long it has been"
    );

    // Readable bytes past the interval recover on their own.
    let rebuilt = tmp.path().join("rebuilt.kgl");
    seed_kgl(&rebuilt, 7);
    std::fs::rename(&rebuilt, &served).expect("republish readable bytes");
    state.ensure_graph_fresh();
    assert_eq!(nodes(&state), 7, "a readable republish is served again");
    assert!(
        state.rebuild_error_note().is_none(),
        "and the stale-graph warning is cleared"
    );
}

/// A file written by a *newer* kglite is the one failure where retrying is
/// pointless and the file is not at fault, so it gets its own advice: restart
/// this server on a build that can read it.
#[test]
fn a_newer_container_says_restart_rather_than_rebuild() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let served = tmp.path().join("from_the_future.kgl");
    seed_kgl(&served, 2);
    let state = serving(&served);

    // The header a later kglite would write: `RGF` plus a container byte this
    // build has no reader for. Only the version byte is forged — the loader
    // refuses on it before anything else in the file is parsed.
    let mut bytes = std::fs::read(&served).expect("read the seeded file");
    bytes[3] += 1;
    let future = bytes[3];
    std::fs::write(&served, &bytes).expect("republish from the future");

    state.ensure_graph_fresh();
    assert_eq!(nodes(&state), 2, "the loaded snapshot keeps serving");
    let note = state
        .rebuild_error_note()
        .expect("an unreadable newer file must be surfaced");
    assert!(
        note.contains(&format!("container v{future}"))
            && note.contains(env!("CARGO_PKG_VERSION"))
            && note.contains("restart"),
        "the note must name both versions and the way out: {note}"
    );
    assert!(
        !note.contains("rebuild"),
        "nothing is wrong with the file — telling an operator to rebuild it \
         would destroy the newer graph: {note}"
    );
}

/// The explicit tool ignores the backstop: an agent asking for a re-read has
/// said the file is ready and is willing to wait for the answer.
#[test]
fn the_reload_tool_retries_bytes_the_backstop_is_holding() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let served = tmp.path().join("explicit.kgl");
    let rebuilt = tmp.path().join("rebuilt.kgl");
    seed_kgl(&served, 2);
    seed_kgl(&rebuilt, 4);
    let state = serving(&served);

    std::fs::write(&served, b"not a kgl file").expect("torn republish");
    state.ensure_graph_fresh();
    assert!(state.graph_reload_failed_at().is_some());

    // Within the retry interval, and the automatic path is holding — the tool
    // still re-reads, because `reload_graph` calls the open path directly.
    std::fs::rename(&rebuilt, &served).expect("republish readable bytes");
    state.ensure_graph_fresh();
    assert_eq!(
        nodes(&state),
        2,
        "the backstop is holding the automatic path"
    );
    state
        .open_or_create(&served, None)
        .expect("the explicit reload_graph path re-reads regardless");
    assert_eq!(nodes(&state), 4);
    assert!(state.rebuild_error_note().is_none());
}

/// Eligibility is decided once, at the open, and only for the shapes the
/// comparison is meaningful for: a graph some peer republishes atomically at a
/// path this server can stat.
#[test]
fn freshness_is_armed_only_for_an_atomically_republished_graph() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let served = tmp.path().join("armed.kgl");
    seed_kgl(&served, 1);
    let state = serving(&served);
    assert_eq!(
        state.with_active(|active| active.freshness_path.clone()),
        Some(Some(served)),
        "a regular-file --graph server stats its file"
    );

    // A path being *created* has nothing to compare against — the identity is
    // decided before the open, when the directory is not there yet.
    let disk = tmp.path().join("disk-graph");
    let disk_state = GraphState::new(None);
    disk_state
        .create_in_mode(&disk, StorageMode::Disk)
        .expect("create a disk graph");
    assert_eq!(
        disk_state.with_active(|active| active.freshness_path.clone()),
        Some(None),
        "a path this open creates has no prior identity to stat"
    );
    // Re-opened, it is a CURRENT-bearing directory: a peer's publish stages a
    // new generation and swings the pointer, and `GraphFileIdentity::capture`
    // folds the pointer's bytes in, so the comparison is exact.
    disk_state
        .open_or_create(&disk, None)
        .expect("re-open the created disk graph");
    assert_eq!(
        disk_state.with_active(|active| active.freshness_path.clone()),
        Some(Some(disk)),
        "a disk-graph directory with a CURRENT pointer is stat-refreshed"
    );

    // A workspace graph refreshes from its producer; it has no file at all.
    let workspace = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let root = tmp.path().join("ws");
    std::fs::create_dir(&root).expect("workspace root");
    std::fs::write(root.join("m.py"), "def f():\n    return 1\n").expect("source file");
    workspace
        .build_workspace_graph(&root, None)
        .expect("workspace build");
    assert_eq!(
        workspace.with_active(|active| active.freshness_path.clone()),
        Some(None),
        "a producer-backed graph is not stat-refreshed"
    );
}
