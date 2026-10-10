//! Group commit: concurrent durable auto-commit statements at `full` share one
//! log barrier.
//!
//! A barrier costs milliseconds and is the same whether it covers one frame or
//! sixty-four, so writers that arrive while another is flushing are collected
//! and flushed together. Contract: a statement returns success only after its
//! own frame is durable, and no reader sees any statement of the batch before
//! the whole batch is — the batch publishes once, after its barrier.
//!
//! **A baton, not a closure queue.** Each writer runs its own statement on its
//! own thread: a statement borrows its caller's [`ExecuteOptions`], so running
//! it on another thread would need that borrow's lifetime erased. The first
//! writer to find no leader leads; the rest park in a FIFO. The leader takes
//! the commit gate, forks the published graph once, runs its statement, then
//! hands the fork to the front of the FIFO in turn (up to [`MAX_BATCH`]
//! statements) and waits for each to finish. Every statement runs with its own
//! undo journal, so one that fails rolls back alone and returns its own error;
//! one that succeeds appends its frame (no barrier) and builds its change
//! events against the fork as that statement left it. The leader then takes
//! one barrier on the last frame, swaps the fork in, publishes the events in
//! order, releases the gate and wakes everyone with the batch's verdict.
//! Leadership passes to the FIFO's front, so a writer waits for at most the
//! running batch and the one it joins.
//!
//! **Failure.** A barrier that fails or unwinds cuts the log back to the
//! batch's first frame and drops the fork: every statement that had succeeded
//! returns [`KgError::DurabilityFailed`] and none was ever visible. A panic in
//! a statement rolls that statement back, releases the baton, and resumes on
//! its own thread; a leader that unwinds wakes its members with an error.
//!
//! **Locks.** `checkpoint_gate` -> `commit_gate` -> `graph` -> `durable` is
//! unchanged. The queue mutex, the parkers and `Batch::data` are leaves: none
//! is held while taking any of those, and the only thread that waits on
//! `Batch::data` is the one whose turn it is.

use std::collections::VecDeque;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use super::execute::{execute_mut_held, ExecuteOptions, ExecuteOutcome};
use super::transaction::Session;
use crate::error::KgError;
use crate::graph::cdc::{self, PendingEvent};
use crate::graph::dir_graph::rollback::StatementCheckpoint;
use crate::graph::dir_graph::DirGraph;
use crate::graph::wal::{DurabilityLevel, StagedFrame};

/// Statements one barrier may cover. Bounds how long the first member of a
/// batch waits for the others' statements to run, and the log written ahead of
/// one barrier.
pub(super) const MAX_BATCH: usize = 256;

/// The session's writer queue.
#[derive(Default)]
pub(super) struct GroupCommit {
    queue: Mutex<Queue>,
}

#[derive(Default)]
struct Queue {
    /// A writer holds (or is about to take) the commit gate to lead a batch.
    leader_active: bool,
    /// Writers waiting to join the leader's batch or, once it ends, to lead the
    /// next one. First come, first served.
    waiting: VecDeque<Arc<Parker>>,
}

enum Msg {
    /// You lead the next batch.
    Lead,
    /// Run your statement on this batch's fork now.
    Turn(Arc<Batch>),
    /// The writer you handed the fork to is finished with it.
    TurnDone,
    /// The batch's verdict: the barrier was taken, or why it was not.
    Flushed(Result<(), String>),
}

/// One writer's mailbox. Messages arrive one at a time, so a single slot is
/// enough.
#[derive(Default)]
struct Parker {
    slot: Mutex<Option<Msg>>,
    ready: Condvar,
}

impl Parker {
    fn post(&self, msg: Msg) {
        *lock(&self.slot) = Some(msg);
        self.ready.notify_one();
    }

    fn wait(&self) -> Msg {
        let mut slot = lock(&self.slot);
        loop {
            if let Some(msg) = slot.take() {
                return msg;
            }
            slot = self.ready.wait(slot).unwrap_or_else(|p| p.into_inner());
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|p| p.into_inner())
}

/// What the batch's statements share. The fork is taken out for the publish.
struct Batch {
    leader: Arc<Parker>,
    data: Mutex<BatchData>,
}

struct BatchData {
    fork: Option<DirGraph>,
    /// Frames staged by the statements that succeeded, in log order.
    frames: Vec<(u64, StagedFrame)>,
    /// Change events per succeeded statement, in commit order.
    events: Vec<Vec<PendingEvent>>,
    /// Statements that succeeded (including ones that wrote nothing).
    committed: u64,
}

/// Passes leadership on when a batch ends, however it ends.
struct Handoff<'a>(&'a Session);

impl Drop for Handoff<'_> {
    fn drop(&mut self) {
        let next = {
            let mut queue = lock(&self.0.group.queue);
            let next = queue.waiting.pop_front();
            if next.is_none() {
                queue.leader_active = false;
            }
            next
        };
        if let Some(next) = next {
            next.post(Msg::Lead);
        }
    }
}

/// The writers admitted to a batch, who are owed its verdict. Dropping it with
/// the verdict unsent (an unwinding leader) tells them the batch failed.
#[derive(Default)]
struct Roster(Vec<Arc<Parker>>);

impl Roster {
    fn announce(&mut self, verdict: &Result<(), String>) {
        for member in self.0.drain(..) {
            member.post(Msg::Flushed(verdict.clone()));
        }
    }
}

impl Drop for Roster {
    fn drop(&mut self) {
        self.announce(&Err(
            "the group commit's leader unwound before its barrier completed".to_string(),
        ));
    }
}

/// Tells the leader a writer is done with the fork, including by unwinding.
struct TurnDone(Arc<Parker>);

impl Drop for TurnDone {
    fn drop(&mut self) {
        self.0.post(Msg::TurnDone);
    }
}

impl Session {
    /// Writers parked behind the running batch. Test-only.
    #[cfg(test)]
    pub(super) fn queued_writers(&self) -> usize {
        lock(&self.group.queue).waiting.len()
    }

    /// Whether [`Self::execute_auto_commit`] routes `query` through the
    /// group-commit queue: a plain data write on a session durable at `full`,
    /// the only level with a barrier to share. A binding that serializes its
    /// own writers ahead of the session needs this to know which statements the
    /// session serializes itself — those may run concurrently, so they can
    /// share a barrier — while every other statement still commits
    /// optimistically and must not overlap another writer.
    pub fn auto_commit_is_grouped(&self, query: &str, opts: &ExecuteOptions<'_>) -> bool {
        self.durability() == Some(DurabilityLevel::Full)
            && super::auto_commit::is_plain_data_write(query, opts)
    }

    /// Run `query` through the group-commit queue, or `None` when it does not
    /// qualify (see [`Self::auto_commit_is_grouped`]) and the caller takes the
    /// ordinary fork path.
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    pub(super) fn execute_grouped(
        &self,
        query: &str,
        opts: &ExecuteOptions<'_>,
    ) -> Option<Result<ExecuteOutcome, KgError>> {
        if !self.auto_commit_is_grouped(query, opts) {
            return None;
        }
        let me = Arc::new(Parker::default());
        let first = {
            let mut queue = lock(&self.group.queue);
            if queue.leader_active {
                queue.waiting.push_back(Arc::clone(&me));
                None
            } else {
                queue.leader_active = true;
                Some(Msg::Lead)
            }
        };
        match first.unwrap_or_else(|| me.wait()) {
            Msg::Lead => Some(self.lead_batch(&me, query, opts)),
            Msg::Turn(batch) => Some(self.follow_batch(&me, &batch, query, opts)),
            Msg::TurnDone | Msg::Flushed(_) => {
                unreachable!("a queued writer is woken with Lead or Turn")
            }
        }
    }

    /// Lead one batch: fork, run this writer's statement, admit the queue's
    /// front in turn, take one barrier, publish.
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    fn lead_batch(
        &self,
        me: &Arc<Parker>,
        query: &str,
        opts: &ExecuteOptions<'_>,
    ) -> Result<ExecuteOutcome, KgError> {
        // Declared first so it drops last, after the gate is released.
        let _handoff = Handoff(self);
        let gate = self.lock_commit_gate();
        let fork = {
            let base = self.snapshot();
            base.try_fork_transaction().map_err(KgError::FileIo)?
        };
        let start_version = fork.version();
        let batch = Arc::new(Batch {
            leader: Arc::clone(me),
            data: Mutex::new(BatchData {
                fork: Some(fork),
                frames: Vec::new(),
                events: Vec::new(),
                committed: 0,
            }),
        });
        let mut roster = Roster::default();
        let mine = self.run_in_batch(&batch, query, opts);
        while roster.0.len() + 1 < MAX_BATCH {
            let Some(next) = lock(&self.group.queue).waiting.pop_front() else {
                break;
            };
            next.post(Msg::Turn(Arc::clone(&batch)));
            roster.0.push(next);
            match me.wait() {
                Msg::TurnDone => {}
                _ => unreachable!("a leader is only ever told a turn is done"),
            }
        }

        let (fork, frames, events, committed) = {
            let mut data = lock(&batch.data);
            (
                data.fork.take(),
                std::mem::take(&mut data.frames),
                std::mem::take(&mut data.events),
                data.committed,
            )
        };
        let verdict = match fork {
            Some(fork) if committed > 0 => self
                .flush_group(frames)
                .map(|()| self.publish_batch(fork, start_version + committed, events)),
            _ => Ok(()),
        };
        drop(gate);
        roster.announce(&verdict);
        match (mine, verdict) {
            (Ok(_), Err(message)) => Err(KgError::DurabilityFailed { message }),
            (mine, _) => mine,
        }
    }

    /// Run this writer's statement on the leader's fork when its turn comes,
    /// then wait for the batch's verdict.
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    fn follow_batch(
        &self,
        me: &Arc<Parker>,
        batch: &Arc<Batch>,
        query: &str,
        opts: &ExecuteOptions<'_>,
    ) -> Result<ExecuteOutcome, KgError> {
        let mine = {
            let _done = TurnDone(Arc::clone(&batch.leader));
            self.run_in_batch(batch, query, opts)
        };
        let outcome = mine?;
        match me.wait() {
            Msg::Flushed(Ok(())) => Ok(outcome),
            Msg::Flushed(Err(message)) => Err(KgError::DurabilityFailed { message }),
            _ => unreachable!("a writer past its turn is only told the batch's verdict"),
        }
    }

    /// Run one statement on the batch's fork under its own undo journal, stage
    /// its frame without a barrier, and build its change events at its own
    /// post-state. A statement that fails leaves the fork as it found it.
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    fn run_in_batch(
        &self,
        batch: &Batch,
        query: &str,
        opts: &ExecuteOptions<'_>,
    ) -> Result<ExecuteOutcome, KgError> {
        let mut guard = lock(&batch.data);
        let data = &mut *guard;
        let graph = data
            .fork
            .as_mut()
            .expect("the fork stays in the batch until the leader publishes");
        let mut held = StatementCheckpoint::None;
        let ran = catch_unwind(AssertUnwindSafe(|| {
            execute_mut_held(graph, query, opts, &mut held, true)
        }));
        let outcome = match ran {
            Ok(Ok(outcome)) => outcome,
            // `mut_statement` rolled itself back before returning the error.
            Ok(Err(error)) => return Err(error),
            Err(panic) => {
                std::mem::replace(&mut held, StatementCheckpoint::None).rollback(graph);
                drop(guard);
                resume_unwind(panic);
            }
        };
        let mut staged = match self.stage_working_commit(graph) {
            Ok(staged) => staged,
            Err(message) => {
                std::mem::replace(&mut held, StatementCheckpoint::None).rollback(graph);
                return Err(KgError::DurabilityFailed { message });
            }
        };
        if let Some(frame) = staged.take_frame() {
            data.frames.push(frame);
        }
        data.events.push(cdc::pending_events(graph, staged.raw()));
        held.commit(graph);
        data.committed += 1;
        Ok(outcome)
    }

    /// Swap the batch's fork in as the published graph and release its change
    /// events, in commit order. Called with the commit gate held, after the
    /// batch's barrier.
    fn publish_batch(&self, mut fork: DirGraph, version: u64, events: Vec<Vec<PendingEvent>>) {
        fork.set_version(version);
        fork.maybe_spill_columns();
        fork.compact_columns_if_fragmented();
        let mut guard = lock_graph(self);
        *guard = Arc::new(fork);
        for statement in events {
            cdc::publish_pending(&guard, statement);
        }
        if let Some(published) = Arc::get_mut(&mut guard) {
            crate::graph::handle::compact_dir_graph(published);
        }
    }
}

fn lock_graph(session: &Session) -> MutexGuard<'_, Arc<DirGraph>> {
    session.graph.lock().unwrap_or_else(|p| p.into_inner())
}
