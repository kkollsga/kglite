//! `--graph` mode's per-call freshness: the `stat` every tool call makes
//! against the served file, the single-flight re-read it triggers when
//! somebody else has republished it, and the bookkeeping that keeps a
//! failing file from costing every later call a doomed load.
//!
//! Workspace modes rebuild their graph from an injected producer
//! (`state_workspace`). A `--graph` server has no producer — its graph *is* a
//! file some other process rewrites — so freshness here is one `stat` compared
//! against the identity the graph was loaded (or last published) at. It is
//! unconditional, not an opt-in: the property an agent needs is that a clean
//! server never answers from a snapshot older than the file was at the time of
//! the call. The cost is that each peer's save costs this server one re-read
//! on its next call, which is the point rather than a side effect.
//!
//! Two things this deliberately does *not* do. It never re-reads while this
//! server holds unsaved changes — that would discard them silently, so the
//! divergence is surfaced as a warning instead and the agent chooses. And it
//! never re-reads a producer-backed graph or a legacy flat disk directory (one
//! with no `CURRENT` pointer, whose identity degenerates to the root inode and
//! so carries no change signal to compare): eligibility is decided once, at the
//! open (`ActiveGraph::freshness_path`). A `CURRENT`-bearing directory *is*
//! re-read — its publish stages a new generation and swings the pointer, never
//! rewriting the generation this server has mapped, so it is stale-and-
//! reloadable exactly as a republished file is.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use kglite::api::io::GraphFileIdentity;

use crate::tools::*;

/// Shortest interval between two failed re-reads of the served file.
///
/// The per-identity rule alone is not a backstop: a producer republishing torn
/// bytes writes *new* bytes each time, so every republish would buy one full
/// failed load — on a 133 MB graph, inside an agent's tool call. Retrying only
/// when the file both changed again *and* this much time has passed bounds
/// that at one doomed load per interval. The explicit `reload_graph` tool
/// bypasses it entirely: an agent asking for a re-read has said the file is
/// ready, and is willing to wait for the answer.
pub(crate) const GRAPH_RELOAD_RETRY_INTERVAL: Duration = Duration::from_secs(5);

/// Freshness bookkeeping for `--graph` mode. Cleared by every successful graph
/// open (see [`GraphState::open_or_create`]).
#[derive(Default)]
pub(crate) struct GraphReloadStatus {
    /// Human-readable description of the last failed re-read, surfaced next to
    /// tool output until the next success.
    pub(crate) last_error: Option<String>,
    /// When that failure happened (for age display, and for the retry
    /// backstop).
    pub(crate) failed_at: Option<SystemTime>,
    /// The file identity that failed to load. A retry needs bytes that are not
    /// these — re-reading the same failing file on every tool call is the
    /// behaviour this replaces.
    pub(crate) failed_identity: Option<GraphFileIdentity>,
    /// When the served file was first seen to have diverged from this server's
    /// snapshot *while it held unsaved changes*. Not a failure — the re-read
    /// was declined on purpose — but the agent has to know, because
    /// `save_graph` will refuse from here on.
    pub(crate) diverged_since: Option<SystemTime>,
}

/// What the read-lock precheck needs in order to decide, without holding any
/// lock across the `stat` or the load.
struct FreshnessProbe {
    path: PathBuf,
    /// The identity the active graph is in step with — the ownership's
    /// `synced` when this server may write the file, else the identity it was
    /// opened at.
    synced: GraphFileIdentity,
    dirty: bool,
}

impl GraphState {
    /// Bring the active graph up to date before a tool reads it — the single
    /// mode-dispatching freshness entry point every graph tool calls.
    ///
    /// Workspace/watch modes take the producer rebuild path unchanged;
    /// everything else takes the `--graph` stat path. Modes are mutually
    /// exclusive by construction (`workspace_mode` is `Some` exactly for the
    /// producer-backed modes), so this is a dispatch, not a fallthrough
    /// running both.
    pub(crate) fn ensure_graph_fresh(&self) {
        if self.workspace_mode.is_some() {
            self.ensure_workspace_graph_fresh();
            return;
        }
        self.ensure_reloaded_graph_fresh();
    }

    /// Read out everything the freshness decision needs under one short read
    /// lock. `None` for a state with no graph, or one whose graph is not
    /// republished atomically at a path it can stat.
    fn freshness_probe(&self) -> Option<FreshnessProbe> {
        let guard = read_lock(&self.inner);
        let active = guard.as_ref()?;
        Some(FreshnessProbe {
            path: active.freshness_path.clone()?,
            synced: active.synced_identity()?.clone(),
            dirty: active.is_dirty(),
        })
    }

    /// Re-read the served file when the filesystem says it is not the one this
    /// graph came from.
    fn ensure_reloaded_graph_fresh(&self) {
        let Some(probe) = self.freshness_probe() else {
            return;
        };
        let Some(current) = self.capture_served_identity(&probe.path) else {
            return;
        };
        if current == probe.synced {
            // The overwhelmingly common case, and the one the self-written
            // file lands in: `publish` recaptured the identity, so a server
            // never re-reads what it just saved.
            self.clear_divergence_note();
            return;
        }
        if probe.dirty {
            self.note_divergence_while_dirty();
            return;
        }
        if !self.reload_is_due(&current) {
            return;
        }
        // Single-flight: two concurrent tool calls that both saw the change
        // must not both load the file. The loser waits here and re-probes.
        let _reload_owner = self.rebuild_gate.enter();
        let Some(probe) = self.freshness_probe() else {
            return;
        };
        let Some(current) = self.capture_served_identity(&probe.path) else {
            return;
        };
        // Everything is re-decided inside the gate: the winner may have
        // installed exactly these bytes, and a write may have landed between
        // the precheck and the gate.
        if current == probe.synced || probe.dirty || !self.reload_is_due(&current) {
            return;
        }
        tracing::info!(
            path = %probe.path.display(),
            "re-reading the served graph (the file on disk changed)"
        );
        // `open_or_create(path, None)`: no storage mode is requested, so a
        // reload never re-runs the boot `--storage` conversion, exactly as the
        // `reload_graph` tool does. The load runs off-lock and every failure
        // returns *before* the write lock, so the active graph survives it. A
        // success clears the failure bookkeeping from inside `open_or_create`.
        if let Err(error) = self.open_or_create(&probe.path, None) {
            self.record_graph_reload_failure(&error, current);
        }
    }

    /// `stat` the served path, or `None` when it cannot be read at all.
    ///
    /// A path that has been *deleted* captures successfully (as an all-`None`
    /// identity), so it compares unequal and takes the reload path, where the
    /// open fails and the previous graph keeps serving with a warning. Only a
    /// genuine I/O failure — an unreadable directory, a dead mount — lands
    /// here, and the right answer for it is to serve what we have.
    fn capture_served_identity(&self, path: &std::path::Path) -> Option<GraphFileIdentity> {
        match GraphFileIdentity::capture(path) {
            Ok(identity) => Some(identity),
            Err(error) => {
                tracing::debug!(
                    path = %path.display(),
                    %error,
                    "cannot stat the served graph file; serving the loaded snapshot"
                );
                None
            }
        }
    }

    /// Whether a re-read of `current` may be attempted now.
    ///
    /// Both halves of the backstop: the same failing bytes are never retried
    /// automatically (the explicit `reload_graph` tool is the way back), and
    /// even *new* bytes wait out [`GRAPH_RELOAD_RETRY_INTERVAL`] since the last
    /// failure.
    fn reload_is_due(&self, current: &GraphFileIdentity) -> bool {
        let status = read_lock(&self.graph_reload);
        let (Some(failed_identity), Some(failed_at)) =
            (status.failed_identity.as_ref(), status.failed_at)
        else {
            return true;
        };
        if failed_identity == current {
            return false;
        }
        SystemTime::now()
            .duration_since(failed_at)
            .is_ok_and(|elapsed| elapsed >= GRAPH_RELOAD_RETRY_INTERVAL)
    }

    /// Record a failed re-read, and the bytes it failed on.
    fn record_graph_reload_failure(&self, error: &anyhow::Error, identity: GraphFileIdentity) {
        let mut status = write_lock(&self.graph_reload);
        status.last_error = Some(error.to_string());
        status.failed_at = Some(SystemTime::now());
        status.failed_identity = Some(identity);
        tracing::warn!(
            error = %error,
            "re-reading the served graph failed; still serving the previously loaded graph"
        );
    }

    /// Remember that the file moved under a dirty server, once — the age in
    /// the warning is the age of the *divergence*, not of the last tool call.
    fn note_divergence_while_dirty(&self) {
        {
            let status = read_lock(&self.graph_reload);
            if status.diverged_since.is_some() {
                return;
            }
        }
        let mut status = write_lock(&self.graph_reload);
        let first = status.diverged_since.get_or_insert_with(SystemTime::now);
        tracing::warn!(
            since = %iso8601(*first),
            "the served graph file changed on disk while this server has unsaved changes; \
             not re-reading (save_graph_as keeps them, reload_graph(discard_unsaved=true) drops them)"
        );
    }

    /// Retract the divergence warning once the file and this server agree
    /// again — which is what a `save_graph` here, or a peer restoring the file,
    /// produces.
    fn clear_divergence_note(&self) {
        if read_lock(&self.graph_reload).diverged_since.is_none() {
            return;
        }
        write_lock(&self.graph_reload).diverged_since = None;
    }

    /// Forget any recorded reload failure. Called from every successful graph
    /// open, so `reload_graph` / `load_graph` / `create_graph` all clear the
    /// retry backstop and the divergence warning together: a graph installed
    /// from this path *is* the file, whatever the previous disagreement was.
    pub(crate) fn clear_graph_reload_failures(&self) {
        let mut status = write_lock(&self.graph_reload);
        status.last_error = None;
        status.failed_at = None;
        status.failed_identity = None;
        status.diverged_since = None;
    }

    /// A one-line warning describing why the served graph is not the file on
    /// disk, or `None` when it is (the common case). The `--graph` counterpart
    /// of the workspace rebuild note, surfaced through the same
    /// [`GraphState::rebuild_error_note`] channel.
    pub(crate) fn graph_reload_error_note(&self) -> Option<String> {
        // Copied out and the lock dropped before anything else is taken: this
        // is the one place that would otherwise hold `graph_reload` while
        // reading the active-graph slot, which is the reverse of the order
        // every other path takes them in.
        let (last_error, failed_at, diverged_since) = {
            let status = read_lock(&self.graph_reload);
            (
                status.last_error.clone(),
                status.failed_at,
                status.diverged_since,
            )
        };
        if let (Some(message), Some(failed_at)) = (last_error, failed_at) {
            return Some(self.reload_failure_note(&message, failed_at));
        }
        let since = diverged_since?;
        Some(format!(
            "WARNING: the served graph file changed on disk at {} ({} ago) while this server \
             has unsaved changes, so it was NOT re-read — save_graph will refuse. \
             save_graph_as to another path keeps your work; \
             reload_graph(discard_unsaved=true) drops it and serves the file on disk.",
            iso8601(since),
            humanize_age(since)
        ))
    }

    /// Render one failed re-read for an agent.
    ///
    /// The newer-container case gets its own text because the advice is the
    /// opposite of the usual one: nothing about the file is wrong and no
    /// retry will ever succeed — the *binary* is old, and only an operator
    /// restarting this server on a newer kglite can serve it.
    fn reload_failure_note(&self, message: &str, failed_at: SystemTime) -> String {
        let age = humanize_age(failed_at);
        if let Some(container) = newer_container_version(message) {
            let serving = self
                .with_active(|active| iso8601(active.built_at))
                .unwrap_or_else(|| "load".to_string());
            return format!(
                "WARNING: the graph file on disk was written by a newer kglite (container \
                 v{container}); this server runs {} — restart it to serve the new file. \
                 Serving the snapshot loaded at {serving} (the re-read failed {age} ago).",
                env!("CARGO_PKG_VERSION")
            );
        }
        format!(
            "WARNING: graph reload failed {age} ago — the active graph is STALE relative to \
             the file on disk. Error: {message}."
        )
    }

    #[cfg(test)]
    pub(crate) fn graph_reload_failed_at(&self) -> Option<SystemTime> {
        read_lock(&self.graph_reload).failed_at
    }

    /// Age the recorded failure by `by`, so a test can reach the far side of
    /// [`GRAPH_RELOAD_RETRY_INTERVAL`] without sleeping through it.
    #[cfg(test)]
    pub(crate) fn backdate_graph_reload_failure(&self, by: Duration) {
        let mut status = write_lock(&self.graph_reload);
        if let Some(failed_at) = status.failed_at {
            status.failed_at = Some(failed_at - by);
        }
    }
}

/// The container version named by the core's "this library only supports up to
/// version N" refusal, or `None` for any other failure.
///
/// Matched on the message because that refusal is a `SaveError`/`io::Error`
/// string by the time it reaches this crate; the phrase is
/// `crates/kglite/src/graph/io/magic.rs`'s, and the fallback text below covers
/// it correctly if that phrasing ever moves.
fn newer_container_version(message: &str) -> Option<u32> {
    let rest = message.split("File uses .kgl container version ").nth(1)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}
