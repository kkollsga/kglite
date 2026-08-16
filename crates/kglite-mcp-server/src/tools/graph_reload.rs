//! `--graph` mode's lazy reload lifecycle: the pending flag a filesystem
//! watcher sets, the single-flight re-read that consumes it on the next tool
//! call, and its failure bookkeeping.
//!
//! Workspace modes rebuild their graph from an injected producer
//! (`state_workspace`). A `--graph` server has no producer — its graph *is* a
//! file some other process rewrites — so `extensions.graph_watch: true` arms a
//! watcher whose callback does nothing but mark this state dirty. The re-read
//! happens here, on the next graph tool call, so no MCP request ever blocks on
//! work it did not ask for and N events between two calls cost one reload.

use std::time::SystemTime;

use crate::tools::*;

/// Watcher-driven reload bookkeeping for `--graph` mode. Cleared by every
/// successful graph open (see [`GraphState::open_or_create`]).
#[derive(Default)]
pub(crate) struct GraphReloadStatus {
    /// Set by the watch callback, consumed by
    /// [`GraphState::ensure_graph_fresh`].
    pub(crate) pending: bool,
    /// Human-readable description of the last failed reload, surfaced next to
    /// tool output until the next success.
    pub(crate) last_error: Option<String>,
    /// When that failure happened (for age display).
    pub(crate) failed_at: Option<SystemTime>,
    /// Consecutive failed reloads with no intervening success. At
    /// [`MAX_CONSECUTIVE_REBUILD_FAILURES`] the watcher-driven reload goes
    /// dormant — see [`GraphReloadStatus::dormant`].
    pub(crate) consecutive_failures: u32,
}

impl GraphReloadStatus {
    /// Whether watcher events are being ignored because the file keeps failing
    /// to open. Same cap as the workspace rebuild path
    /// ([`MAX_CONSECUTIVE_REBUILD_FAILURES`]) and for the same reason: a
    /// permanently broken source must not make every tool call pay for a
    /// doomed load. Unlike the workspace path a fresh event does *not* re-arm
    /// it — a `--graph` producer that republishes a torn file typically keeps
    /// republishing it — so recovery is the explicit `reload_graph` tool
    /// (any successful open clears the counter).
    fn dormant(&self) -> bool {
        self.consecutive_failures >= MAX_CONSECUTIVE_REBUILD_FAILURES
    }
}

impl GraphState {
    /// Bring the active graph up to date before a tool reads it — the single
    /// mode-dispatching freshness entry point every graph tool calls.
    ///
    /// Workspace/watch modes take the producer rebuild path unchanged;
    /// everything else takes the `--graph` reload path, which is a plain
    /// no-op unless a watcher marked the served file rewritten. Modes are
    /// mutually exclusive by construction (`workspace_mode` is `Some` exactly
    /// for the producer-backed modes), so this is a dispatch, not a fallthrough
    /// running both.
    pub(crate) fn ensure_graph_fresh(&self) {
        if self.workspace_mode.is_some() {
            self.ensure_workspace_graph_fresh();
            return;
        }
        self.ensure_reloaded_graph_fresh();
    }

    /// Mark the served graph file as externally rewritten. Called from the
    /// watch callback, so it must stay non-blocking: one short lock, no I/O.
    /// Returns whether the mark was taken (`false` while dormant).
    pub(crate) fn mark_graph_reload_pending(&self) -> bool {
        let mut status = write_lock(&self.graph_reload);
        if status.dormant() {
            tracing::debug!(
                failures = status.consecutive_failures,
                "graph file changed but reloads are dormant after repeated failures; \
                 call reload_graph to retry"
            );
            return false;
        }
        status.pending = true;
        true
    }

    /// Re-read the served file if a watcher marked it changed since the last
    /// tool call.
    fn ensure_reloaded_graph_fresh(&self) {
        // Every tool call passes through here and the overwhelmingly common
        // case is "no watcher configured, nothing pending". Check that under a
        // read lock first so the ordinary path never contends on the
        // single-flight gate.
        if !read_lock(&self.graph_reload).pending {
            return;
        }
        // Single-flight: two callers that both observed the dirty flag must not
        // both load the file. The loser waits here and finds the flag consumed.
        let _reload_owner = self.rebuild_gate.enter();
        {
            let mut status = write_lock(&self.graph_reload);
            if !status.pending {
                return;
            }
            status.pending = false;
        }
        let Some(path) = self.source_path() else {
            // No graph, or one with no backing file: nothing to re-read.
            return;
        };
        tracing::info!(
            path = %path.display(),
            "re-reading the served graph (filesystem changed)"
        );
        // `open_or_create(path, None)`: no storage mode is requested, so a
        // reload never re-runs the boot `--storage` conversion, exactly as the
        // `reload_graph` tool does. The load runs off-lock and every failure
        // returns *before* the write lock, so the active graph survives it. A
        // success clears the failure bookkeeping from inside `open_or_create`.
        if let Err(error) = self.open_or_create(&path, None) {
            self.record_graph_reload_failure(&error);
        }
    }

    /// Record a failed reload and decide whether to retry on the next call.
    fn record_graph_reload_failure(&self, error: &anyhow::Error) {
        let mut status = write_lock(&self.graph_reload);
        status.consecutive_failures = status.consecutive_failures.saturating_add(1);
        status.last_error = Some(error.to_string());
        status.failed_at = Some(SystemTime::now());
        if status.dormant() {
            // Clear explicitly rather than relying on it already being false: a
            // watcher event landing *during* the failed load could have re-set
            // it, and dormancy means dormant.
            status.pending = false;
            tracing::warn!(
                failures = status.consecutive_failures,
                error = %error,
                "graph reload keeps failing — still serving the previously loaded graph; \
                 watcher-driven reloads are dormant until a reload_graph succeeds"
            );
        } else {
            status.pending = true;
            tracing::warn!(error = %error, "graph reload failed; retrying on the next tool call");
        }
    }

    /// Forget any recorded reload failure. Called from every successful graph
    /// open, so `reload_graph` / `load_graph` / `create_graph` all lift
    /// dormancy. Deliberately does *not* touch `pending`: an event that landed
    /// while the load was in flight describes bytes this load may not have
    /// seen.
    pub(crate) fn clear_graph_reload_failures(&self) {
        let mut status = write_lock(&self.graph_reload);
        status.last_error = None;
        status.failed_at = None;
        status.consecutive_failures = 0;
    }

    /// A one-line warning describing the last failed reload of the served
    /// graph file, or `None` when the last reload succeeded (the common case).
    /// The `--graph` counterpart of the workspace rebuild note, surfaced
    /// through the same [`GraphState::rebuild_error_note`] channel.
    pub(crate) fn graph_reload_error_note(&self) -> Option<String> {
        let status = read_lock(&self.graph_reload);
        let message = status.last_error.as_ref()?;
        let age = humanize_age(status.failed_at?);
        let recovery = if status.dormant() {
            " Watcher-driven reloads are dormant; call reload_graph once the file is readable."
        } else {
            ""
        };
        Some(format!(
            "WARNING: graph reload failed {age} ago ({} consecutive failure(s)) — the \
             active graph is STALE relative to the file on disk. Error: {message}.{recovery}",
            status.consecutive_failures
        ))
    }

    #[cfg(test)]
    pub(crate) fn graph_reload_pending(&self) -> bool {
        read_lock(&self.graph_reload).pending
    }

    #[cfg(test)]
    pub(crate) fn graph_reload_failures(&self) -> u32 {
        read_lock(&self.graph_reload).consecutive_failures
    }
}
