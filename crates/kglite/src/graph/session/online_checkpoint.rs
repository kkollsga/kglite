//! Online checkpoint: fold the log into the `.kgl` without stalling writers.
//!
//! [`Session::save`] holds the graph mutex for the whole serialize, so every
//! committer waits out the write. [`Session::checkpoint_online`] instead:
//!
//! 1. **Snapshot** under the commit gate and both session locks for an `Arc`
//!    clone: the published graph, the LSN of the last frame inside it, the log
//!    offset just past that frame and the log's epoch (a commit appends and
//!    publishes under the gate, so the four agree; a commit mid-barrier makes
//!    this step wait for it).
//! 2. **Write** the snapshot as the live `.kgl`, stamped with that LSN, by temp +
//!    fsync + rename + directory fsync, with no session lock held.
//! 3. **Trim** under the gate and both locks again: drop the frames up to the recorded
//!    offset and keep the frames committed during step 2
//!    ([`Wal::trim_through`](crate::graph::wal::Wal::trim_through)).
//!
//! **Crash safety.** Before the rename the old checkpoint and the whole log are
//! intact. Between the rename and the trim the new checkpoint is stamped
//! `checkpoint_lsn = L` and the log still holds every frame: replay skips frames
//! at or below `L` and applies the rest, exactly as it does after a crash
//! inside [`Session::save`]. The trim is itself a temp + rename, so the log is
//! always either the old whole one or the new tail. The log is never cut before
//! the checkpoint that replaces it is durable.
//!
//! **Exclusion.** `save` and this share [`Session::checkpoint_gate`]-ordered
//! access: a `save` that reset the log between steps 1 and 3 would make the
//! recorded offset stale (the epoch check refuses the trim) and the older
//! stamped file would overwrite a newer checkpoint, so they never overlap.
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::transaction::Session;
use crate::graph::io::file::SaveError;
use crate::graph::storage::GraphRead;

/// The log size, in bytes, past which a session's automatic-checkpoint policy
/// asks for a checkpoint when the caller sets none.
///
/// Measured on macOS with the edge workload (40-row upserts, about 2 KB of log
/// per commit): restart replay costs 7-9 MiB above an idle server for a 6 MB or
/// a 56 MB log that updates a bounded set of nodes, and up to 5x the log when
/// the commits keep creating and deleting new nodes, because the replay plan
/// keeps one entry per distinct node the log names. 16 MiB therefore caps a
/// restart near 80 MiB in the worst case, at a checkpoint roughly every 8,000
/// commits. A graph large enough to make a rewrite expensive makes the log's
/// own size floor (the `.kgl` size) the binding one.
pub const DEFAULT_AUTO_CHECKPOINT_WAL_BYTES: u64 = 16 * 1024 * 1024;

/// What an online checkpoint wrote.
#[derive(Debug, Clone)]
pub struct OnlineCheckpointReport {
    /// The live checkpoint file that was replaced.
    pub path: PathBuf,
    /// Size of the published file.
    pub bytes: u64,
    /// The `checkpoint_lsn` stamped into it.
    pub lsn: u64,
    /// Version of the snapshot written.
    pub graph_version: u64,
    /// Log frame bytes before the checkpoint and after the trim.
    pub wal_bytes_before: u64,
    pub wal_bytes_after: u64,
    /// How long the locks were held to fix the snapshot, and again to trim.
    /// These are the only intervals a committer can wait on this checkpoint.
    pub snapshot_hold: Duration,
    pub trim_hold: Duration,
    /// Whole call.
    pub elapsed: Duration,
}

impl Session {
    /// Set the log size that makes [`Self::needs_checkpoint`] true; `None`
    /// disables the policy. A no-op on a session without a log.
    pub fn set_auto_checkpoint_wal_bytes(&self, bytes: Option<u64>) {
        if let Some(ds) = self
            .durable
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_mut()
        {
            ds.set_auto_checkpoint(bytes);
        }
    }

    /// Whether the log has outgrown the automatic-checkpoint bound. One lock and
    /// a counter read: cheap enough to call after every commit. The caller
    /// decides where [`Self::checkpoint_online`] then runs.
    pub fn needs_checkpoint(&self) -> bool {
        !self.is_retired()
            && self
                .durable
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .is_some_and(|ds| ds.needs_checkpoint())
    }

    /// [`Self::checkpoint_online`] when [`Self::needs_checkpoint`] holds and no
    /// other checkpoint is running; `Ok(None)` otherwise. A failure backs the
    /// policy off by one threshold of log growth.
    pub fn maybe_checkpoint_online(&self) -> Result<Option<OnlineCheckpointReport>, String> {
        if !self.needs_checkpoint() {
            return Ok(None);
        }
        let Ok(gate) = self.checkpoint_gate.try_lock() else {
            return Ok(None);
        };
        let report = self.checkpoint_online_gated();
        drop(gate);
        report.map(Some)
    }

    /// Fold the log into the live checkpoint without holding the session locks
    /// across the write; see the module docs for the order and its crash
    /// safety. Waits for a running `save` or checkpoint. Refused on a session
    /// without a log, and on one a direct write has diverged from its log (take
    /// [`Self::save`], which folds those mutations in).
    pub fn checkpoint_online(&self) -> Result<OnlineCheckpointReport, String> {
        let _gate = self
            .checkpoint_gate
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        self.checkpoint_online_gated()
    }

    fn checkpoint_online_gated(&self) -> Result<OnlineCheckpointReport, String> {
        self.ensure_not_retired()?;
        let started = Instant::now();
        let (snapshot, point, snapshot_hold) = {
            let _commit = self.lock_commit_gate();
            let graph = self.graph.lock().unwrap_or_else(|p| p.into_inner());
            let mut slot = self.durable.lock().unwrap_or_else(|p| p.into_inner());
            let held = Instant::now();
            let Some(ds) = slot.as_mut() else {
                return Err(
                    "checkpoint_online() needs a session opened with a write-ahead \
                            log (Session::open_durable at level 'full' or 'normal')"
                        .to_string(),
                );
            };
            let point = ds.begin_online()?;
            (std::sync::Arc::clone(&graph), point, held.elapsed())
        };
        #[cfg(test)]
        hooks::run(hooks::Stage::AfterSnapshot);

        let written = self.write_online(&snapshot, &point);
        #[cfg(test)]
        hooks::run(hooks::Stage::BeforeTrim);

        let graph_version = snapshot.version();
        drop(snapshot);
        let (trimmed, trim_hold) = {
            let _commit = self.lock_commit_gate();
            let _graph = self.graph.lock().unwrap_or_else(|p| p.into_inner());
            let mut slot = self.durable.lock().unwrap_or_else(|p| p.into_inner());
            let held = Instant::now();
            let ds = slot.as_mut().ok_or("session lost its log mid-checkpoint")?;
            let outcome = ds.finish_online(&point, written.as_ref().map(|b| *b).map_err(|_| ()));
            (outcome, held.elapsed())
        };
        let bytes = written?;
        let wal_bytes_after = trimmed?;
        Ok(OnlineCheckpointReport {
            path: point.checkpoint,
            bytes,
            lsn: point.lsn,
            graph_version,
            wal_bytes_before: point.wal_bytes,
            wal_bytes_after,
            snapshot_hold,
            trim_hold,
            elapsed: started.elapsed(),
        })
    }

    /// Barrier the frames the snapshot contains, then publish the stamped file.
    fn write_online(
        &self,
        snapshot: &std::sync::Arc<crate::graph::dir_graph::DirGraph>,
        point: &super::durable::OnlinePoint,
    ) -> Result<u64, String> {
        if snapshot.graph.is_disk() {
            return Err("online checkpoint cannot write a disk-mode graph".to_string());
        }
        if let Some(barrier) = &point.barrier {
            crate::graph::durable_io::trace::record(|| "wal sync_data".to_string());
            barrier.sync_data().map_err(|e| e.to_string())?;
        }
        let dest = point.checkpoint.to_str().ok_or_else(|| {
            SaveError::Refused("checkpoint path is not valid UTF-8".to_string()).to_string()
        })?;
        super::backup::write_stamped(snapshot, dest, point.lsn).map_err(|e| e.to_string())
    }
}

/// Test seams for the two windows a crash or a concurrent commit can land in.
#[cfg(test)]
pub(super) mod hooks {
    use std::cell::RefCell;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(in crate::graph::session) enum Stage {
        AfterSnapshot,
        BeforeTrim,
    }

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static HOOKS: RefCell<Vec<(Stage, Hook)>> = const { RefCell::new(Vec::new()) };
    }

    pub(in crate::graph::session) fn set(stage: Stage, hook: impl FnOnce() + 'static) {
        HOOKS.with(|h| h.borrow_mut().push((stage, Box::new(hook))));
    }

    pub(super) fn run(stage: Stage) {
        let hook = HOOKS.with(|h| {
            let mut h = h.borrow_mut();
            h.iter()
                .position(|(s, _)| *s == stage)
                .map(|i| h.remove(i).1)
        });
        if let Some(hook) = hook {
            hook();
        }
    }
}
