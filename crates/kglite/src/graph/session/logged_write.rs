//! Closure-shaped writes on a [`Session`]: [`Session::write_logged`] for
//! mutations the write-ahead log must describe, [`Session::apply_unlogged`]
//! for the ones it deliberately must not, and [`Session::retire`] to end a
//! session's right to publish anything.
//!
//! [`Session::write`] and [`Session::transact`] take a `&mut DirGraph` closure
//! too, but on a durable session they latch the session as diverged because
//! their captured ops never reach a frame. These two verbs are the logged and
//! the explicitly-unlogged halves of that shape:
//!
//! - `write_logged` forks the published graph, runs the closure on the fork,
//!   and publishes it through the same stage -> barrier -> swap path a
//!   transaction commit takes. A closure that errors, a refused ontology
//!   verdict, a failed append or a failed barrier publishes nothing and
//!   consumes no LSN.
//! - `apply_unlogged` publishes the same way but **refuses** a closure that
//!   left captured ops behind: a mutator filed as "checkpoint-only" that in
//!   fact changes logged state would otherwise return success and be lost by
//!   the next crash.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::transaction::{CommitOutcome, Session};
use crate::error::KgError;
use crate::graph::dir_graph::DirGraph;

/// Why a closure write did not publish. Nothing is visible in any variant.
#[derive(Debug)]
#[non_exhaustive]
pub enum WriteError<E> {
    /// The closure returned this error; its working copy was dropped.
    Closure(E),
    /// The session was [retired](Session::retire).
    Retired,
    /// The working copy could not be forked (a disk graph out of descriptors
    /// or address space).
    Fork(String),
    /// The log append or barrier failed. The frame was cut back and its LSN
    /// given back, so the session is exactly as it was and stays usable.
    Durability(String),
    /// The transaction-end ontology verdict refused the closure's end state.
    Ontology(Box<KgError>),
    /// [`Session::apply_unlogged`] only: the closure captured this many ops
    /// that no log frame would describe.
    CapturedOps(usize),
}

/// What a published closure write did.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteOutcome<T> {
    /// The closure's return value.
    pub value: T,
    /// Graph version after the write (unchanged when the closure mutated nothing).
    pub version: u64,
    /// LSN of the frame the write logged; `None` when it logged nothing
    /// (a non-durable session, or a closure that captured no ops).
    pub lsn: Option<u64>,
    /// `warn`-level ontology findings of the end-of-write verdict.
    pub warnings: Vec<String>,
}

pub(super) const RETIRED_MSG: &str =
    "this session was retired (its graph was closed), so it no longer accepts writes, \
     checkpoints or commits";

impl Session {
    /// Run `f` on a working copy and publish it after its captured ops are
    /// durable at the session's level.
    ///
    /// The commit gate is held from the fork to the swap, so writers
    /// serialize and the closure sees the latest published graph; readers
    /// keep taking snapshots of the previous graph throughout, including during
    /// the barrier. A closure that captured no ops and moved no version is
    /// still published (so a mutator filed under the wrong verb cannot lose its
    /// write) but logs no frame and leaves the version alone.
    ///
    /// The closure runs under the gate: it must not call back into this
    /// session, and foreign callbacks (an embedder, a progress hook) belong
    /// before the call, with their results passed in.
    pub fn write_logged<T, E>(
        &self,
        f: impl FnOnce(&mut DirGraph) -> Result<T, E>,
    ) -> Result<WriteOutcome<T>, WriteError<E>> {
        self.closure_write(f, false)
    }

    /// Run `f` on a working copy and publish it **without** a log frame, for
    /// state the log does not describe by design (schema, configuration, text
    /// indexes, column maintenance).
    ///
    /// Errors with [`WriteError::CapturedOps`] — publishing nothing — when `f`
    /// left captured ops. A session captures ops when it is durable or feeds
    /// change data capture; on a session that does neither there is nothing to
    /// observe and the write publishes. The ops are the evidence that the
    /// mutator changes logged state and belongs in [`Self::write_logged`].
    pub fn apply_unlogged<T, E>(
        &self,
        f: impl FnOnce(&mut DirGraph) -> Result<T, E>,
    ) -> Result<WriteOutcome<T>, WriteError<E>> {
        self.closure_write(f, true)
    }

    fn closure_write<T, E>(
        &self,
        f: impl FnOnce(&mut DirGraph) -> Result<T, E>,
        unlogged: bool,
    ) -> Result<WriteOutcome<T>, WriteError<E>> {
        let _commit = self.lock_commit_gate();
        self.ensure_not_retired().map_err(|_| WriteError::Retired)?;
        let base = self.snapshot();
        let base_version = base.version();
        let mut working = base
            .try_fork_transaction()
            .map_err(|e| WriteError::Fork(e.to_string()))?;
        drop(base);
        // The whole closure is one transaction: its "must exist" verdict is
        // judged once at the end, not per statement inside it.
        working.ontology_tx_deferred = true;
        let value = f(&mut working).map_err(WriteError::Closure)?;
        let warnings = working
            .judge_transaction_end()
            .map_err(|e| WriteError::Ontology(Box::new(e)))?;
        let captured = working
            .graph
            .recording_mut()
            .map_or(0, |recording| recording.ops_len());
        if unlogged && captured > 0 {
            return Err(WriteError::CapturedOps(captured));
        }
        if working.version() == base_version && captured == 0 {
            // Nothing logged and no version moved: the closure changed state
            // the log does not describe (or nothing). Still published -- a
            // closure that returned Ok must not be dropped, since a mutator
            // filed under the wrong verb would otherwise lose its write
            // silently -- but without a frame or a version bump.
            *self.graph.lock().unwrap_or_else(|p| p.into_inner()) = Arc::new(working);
            return Ok(WriteOutcome {
                value,
                version: base_version,
                lsn: None,
                warnings,
            });
        }
        working.compact_columns_if_fragmented();
        let (outcome, lsn) = self.publish_gated(working, base_version, false);
        match outcome {
            CommitOutcome::Committed { new_version } => Ok(WriteOutcome {
                value,
                version: new_version,
                lsn,
                warnings,
            }),
            CommitOutcome::DurabilityFailed { error } => Err(WriteError::Durability(error)),
            // `publish_gated` with `check_occ == false` reaches neither.
            other => unreachable!("an unchecked publish returned {other:?}"),
        }
    }

    /// End this session's right to publish: every later commit, auto-commit
    /// statement, closure write, save and checkpoint is refused. Reads and snapshots keep working.
    ///
    /// Takes the checkpoint and commit gates, so a commit or checkpoint
    /// already in flight finishes first and nothing publishes after this
    /// returns. Idempotent. A binding calls it after its final save and before
    /// releasing the writer lease, so a retained `Session` or `Transaction`
    /// cannot append to a log another process now owns.
    pub fn retire(&self) {
        let _checkpoint = self
            .checkpoint_gate
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let _commit = self.lock_commit_gate();
        self.retired.store(true, Ordering::Release);
    }

    /// Whether [`Self::retire`] has been called.
    pub fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    pub(super) fn ensure_not_retired(&self) -> Result<(), String> {
        if self.is_retired() {
            Err(RETIRED_MSG.to_string())
        } else {
            Ok(())
        }
    }
}
