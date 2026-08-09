//! Lazy workspace-graph rebuild machinery: failure bookkeeping, the
//! single-owner rebuild gate, and the pending-rebuild slot the watcher tags.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::SystemTime;

use crate::tools::*;

/// Hot-fail guard for the lazy workspace-graph rebuild: after this many
/// consecutive failures for the same target, [`GraphState::ensure_workspace_graph_fresh`]
/// stops restoring the dirty marker (no more per-tool-call retries) and
/// keeps serving the stale graph — with the failure surfaced in tool
/// output — until a new FS event re-tags the target.
pub(crate) const MAX_CONSECUTIVE_REBUILD_FAILURES: u32 = 3;

/// Bookkeeping for lazy workspace-graph rebuild failures. Reset to default on
/// the next successful build.
#[derive(Default)]
pub(crate) struct RebuildStatus {
    /// Human-readable description of the last failed rebuild.
    pub(crate) last_error: Option<String>,
    /// When that failure happened (for age display).
    pub(crate) failed_at: Option<SystemTime>,
    /// Consecutive failures for `failed_target` with no intervening
    /// success.
    pub(crate) consecutive_failures: u32,
    /// The target whose rebuilds keep failing.
    pub(crate) failed_target: Option<WorkspaceGraphTarget>,
}

/// Machine-readable reason that an ensured workspace graph remains stale.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkspaceRebuildFailureReason {
    /// The failed target remains queued for a retry on the next ensured read.
    RebuildFailed,
    /// The consecutive-failure cap was reached; retries resume only after a
    /// new relevant filesystem event re-tags the workspace.
    HotFail,
}

impl WorkspaceRebuildFailureReason {
    /// Stable detail value for structured MCP errors.
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::RebuildFailed => "rebuild_failed",
            Self::HotFail => "hot_fail",
        }
    }
}

/// Typed snapshot of a failed rebuild for the currently installed workspace
/// generation. Structured routes consume this after freshness handling; the
/// legacy warning renderer uses the same snapshot rather than parsing prose.
#[derive(Clone, Debug)]
pub(crate) struct WorkspaceRebuildFailureSnapshot {
    pub(crate) reason: WorkspaceRebuildFailureReason,
    pub(crate) message: String,
    pub(crate) failed_at: SystemTime,
    pub(crate) consecutive_failures: u32,
    pub(crate) retry_limit: u32,
}

#[derive(Default)]
pub(crate) struct WorkspaceRebuildGate {
    pub(crate) state: Mutex<WorkspaceRebuildGateState>,
    pub(crate) changed: Condvar,
}

#[derive(Default)]
pub(crate) struct WorkspaceRebuildGateState {
    pub(crate) in_flight: bool,
    pub(crate) waiters: usize,
}

impl WorkspaceRebuildGate {
    pub(crate) fn enter(&self) -> WorkspaceRebuildGuard<'_> {
        let mut state = mutex_lock(&self.state);
        if state.in_flight {
            state.waiters += 1;
            self.changed.notify_all();
            while state.in_flight {
                state = self
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            state.waiters -= 1;
        }
        state.in_flight = true;
        drop(state);
        WorkspaceRebuildGuard { gate: self }
    }

    #[cfg(test)]
    pub(crate) fn wait_for_waiter(&self) {
        let state = mutex_lock(&self.state);
        let (state, timeout) = self
            .changed
            .wait_timeout_while(state, std::time::Duration::from_secs(5), |state| {
                state.waiters == 0
            })
            .unwrap_or_else(PoisonError::into_inner);
        assert!(
            !timeout.timed_out() && state.waiters > 0,
            "second freshness caller never waited behind the in-flight rebuild"
        );
    }
}

pub(crate) struct WorkspaceRebuildGuard<'a> {
    pub(crate) gate: &'a WorkspaceRebuildGate,
}

impl Drop for WorkspaceRebuildGuard<'_> {
    fn drop(&mut self) {
        let mut state = mutex_lock(&self.gate.state);
        debug_assert!(state.in_flight, "rebuild gate released without an owner");
        state.in_flight = false;
        drop(state);
        self.gate.changed.notify_all();
    }
}

/// Exact installed workspace product a watcher event observed. The generation
/// prevents a slow rebuild prepared for an older activation from overwriting a
/// newer graph, even when the root/revision labels later repeat.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WorkspaceGraphTarget {
    pub(crate) root: PathBuf,
    pub(crate) revisions: Option<Vec<String>>,
    pub(crate) generation: u64,
}

impl WorkspaceGraphTarget {
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn same_source(&self, other: &Self) -> bool {
        self.root == other.root && self.revisions == other.revisions
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkspaceGraphEnqueue {
    Enqueued,
    Empty,
    IncompatibleSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkspaceRebuildCommit {
    Installed,
    RequeuedCompatible,
    DiscardedIncompatible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkspaceRebuildFailureDisposition {
    Current(u32),
    RequeuedCompatible,
    DiscardedIncompatible,
}

/// One target-bound set of watcher paths waiting for a lazy rebuild.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PendingWorkspaceRebuild {
    pub(crate) target: WorkspaceGraphTarget,
    pub(crate) changed_paths: BTreeSet<PathBuf>,
    /// Dormant work retains evidence after the hot-fail cap without causing
    /// another rebuild until a genuine filesystem event re-arms it.
    pub(crate) ready: bool,
}

/// Make a path absolute and collapse lexical `.` / `..` components without
/// touching the filesystem. Unlike canonicalization this preserves deleted
/// watcher paths.
pub(crate) fn absolute_lexical_path(path: &Path) -> Option<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
        }
    }
    Some(normalized)
}

pub(crate) fn merge_ready_pending(
    pending: &mut Option<PendingWorkspaceRebuild>,
    target: &WorkspaceGraphTarget,
    changed_paths: impl IntoIterator<Item = PathBuf>,
) {
    match pending.as_mut() {
        Some(existing) if existing.target.same_source(target) => {
            existing.target = target.clone();
            existing.changed_paths.extend(changed_paths);
            existing.ready = true;
        }
        _ => {
            *pending = Some(PendingWorkspaceRebuild {
                target: target.clone(),
                changed_paths: changed_paths.into_iter().collect(),
                ready: true,
            });
        }
    }
}
