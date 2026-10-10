//! Auto-commit write and checkpoint-if-changed: the two `Session` verbs every
//! serving binding writes identically on top of [`Session::begin`] /
//! [`Session::commit`] / [`Session::save`].

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::execute::{execute_mut, execute_mut_held, ExecuteOptions, ExecuteOutcome};
use super::transaction::{CommitOutcome, Session};
use crate::error::KgError;
use crate::graph::dir_graph::rollback::StatementCheckpoint;
use crate::graph::languages::cypher;
use crate::graph::languages::cypher::ast::Clause;

/// What [`Session::checkpoint_if_changed`] did, and at which graph version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointOutcome {
    /// The graph was saved; the value is the version it was saved at.
    Written(u64),
    /// Unchanged since the last successful checkpoint at this version.
    Skipped(u64),
}

impl Session {
    /// Run one mutating statement as a one-shot transaction, making up to
    /// `attempts` tries (minimum one) while the commit loses an optimistic
    /// race.
    ///
    /// **In place when nothing else holds the graph** (never at `full`
    /// durability, which always forks so readers are not held off while the
    /// write's log frame is flushed). Under the session lock,
    /// a published graph with no other owner (no reader snapshot, cursor,
    /// backup or checkpoint image, open transaction or held result) is
    /// mutated directly instead of forked: no whole-graph copy, no free of the
    /// replaced graph. The statement's undo journal is kept until the commit
    /// is final, so a statement error, an ontology refusal, a cancellation or
    /// a failed log append restores the exact prior graph. Because the lock is
    /// held for the whole statement, **a reader arriving mid-statement waits
    /// for it** and then sees all of it or none of it; the fork path never
    /// blocked readers. The in-place path cannot lose an optimistic race, so
    /// it ignores `attempts`. See [`Self::execute_auto_commit_in_place`] for
    /// the conditions; anything else takes the fork path described below.
    ///
    /// A lost race published nothing, so re-running on a fresh `begin()` cannot
    /// double-apply. Errors: whatever the statement raised, a
    /// [`KgError::TransactionConflict`] once the attempts are spent, and a
    /// [`KgError::DurabilityFailed`] when the write-ahead log rejected the
    /// frame (the commit was not published, and is never retried — the log is
    /// not going to answer differently).
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    pub fn execute_auto_commit(
        &self,
        query: &str,
        opts: &ExecuteOptions<'_>,
        attempts: u32,
    ) -> Result<ExecuteOutcome, KgError> {
        if let Some(outcome) = self.execute_auto_commit_in_place(query, opts) {
            return outcome;
        }
        self.execute_auto_commit_observed(query, opts, attempts, &mut |_| {})
    }

    /// Statements run in place so far (test observability).
    #[cfg(test)]
    pub(crate) fn in_place_commit_count(&self) -> u64 {
        self.in_place_commits.load(Ordering::Relaxed)
    }

    /// Statements that wanted the in-place path but found the graph shared
    /// (test observability).
    #[cfg(test)]
    pub(crate) fn shared_fork_count(&self) -> u64 {
        self.forked_commits.load(Ordering::Relaxed)
    }

    /// Run `query` directly on the published graph when that is provably
    /// invisible to everyone else, or `None` to leave it to the fork path.
    ///
    /// In place requires all of:
    /// - the statement is a plain data write: only `MATCH`/`WHERE`/`WITH`/
    ///   `RETURN`-family clauses and `CREATE`/`MERGE`/`SET`/`REMOVE`/`DELETE`/
    ///   `FOREACH`. Schema DDL, procedure calls, `UNION`, `LOAD CSV` and
    ///   subqueries fork: their undo is either not journalled (index DDL) or
    ///   unbounded in lock time, and a procedure may call back into the session;
    /// - no embedder: its model callbacks run foreign code (a Python
    ///   embedder needs the GIL that a waiting reader may hold), which must
    ///   not run under the session lock;
    /// - a backend whose undo journal reverses every edit (memory and mapped,
    ///   also under the durable capture wrapper). Disk graphs fork: their
    ///   fork is a remap of immutable bases, and the in-place alternative is a
    ///   whole-graph clone checkpoint per statement;
    /// - the session is not durable at `full` (see the comment in the body);
    /// - `Arc::get_mut` succeeds on the published graph while the lock is
    ///   held, i.e. no snapshot exists. A snapshot taken earlier owns a clone
    ///   of the `Arc`, so it can never observe the write — such a statement
    ///   forks. A snapshot requested during the statement waits on the lock.
    ///
    /// The journal outlives the statement until the log append (durable
    /// sessions) has succeeded, and a panic unwinding out of the statement
    /// rolls it back before resuming, so a failed or aborted statement never
    /// leaves a half-written graph published.
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    pub(crate) fn execute_auto_commit_in_place(
        &self,
        query: &str,
        opts: &ExecuteOptions<'_>,
    ) -> Option<Result<ExecuteOutcome, KgError>> {
        self.in_place_with(query, opts, execute_mut_held)
    }

    /// [`Self::execute_auto_commit_in_place`] with the statement runner
    /// injected, so a test can make it panic after the statement's writes.
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    pub(super) fn in_place_with(
        &self,
        query: &str,
        opts: &ExecuteOptions<'_>,
        run: impl FnOnce(
            &mut crate::graph::dir_graph::DirGraph,
            &str,
            &ExecuteOptions<'_>,
            &mut StatementCheckpoint,
            bool,
        ) -> Result<ExecuteOutcome, KgError>,
    ) -> Option<Result<ExecuteOutcome, KgError>> {
        if opts.embedder.is_some() {
            return None;
        }
        let parsed = cypher::parse_cypher(query).ok()?;
        if parsed.explain
            || parsed.profile
            || !cypher::is_mutation_query(&parsed)
            || !parsed.clauses.iter().all(is_plain_data_clause)
        {
            return None;
        }
        // At `full` the statement's graph is the one readers receive, so it
        // would have to hold them off until the flush finishes (one fsync per
        // write, ~4 ms on macOS) to keep an unflushed write invisible. The fork
        // path flushes with readers unblocked, and costs a few hundred
        // microseconds more per write on a 550,000-node graph, so `full`
        // always forks. `normal` has no flush to wait for.
        if self.durability() == Some(crate::graph::wal::DurabilityLevel::Full) {
            return None;
        }
        // The gate keeps a fork-path committer's check-to-swap window free of
        // this statement's in-place mutation and version bump.
        let _commit = self.lock_commit_gate();
        let mut guard = self.graph.lock().unwrap_or_else(|p| p.into_inner());
        let Some(graph) = Arc::get_mut(&mut guard) else {
            self.forked_commits.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        if !graph.graph.supports_undo_journal() {
            return None;
        }
        let durable = self
            .durable
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some();
        let start_version = graph.version();
        let mut held = StatementCheckpoint::None;
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(graph, query, opts, &mut held, durable)
        }));
        let outcome = match ran {
            Ok(Ok(outcome)) => outcome,
            // `mut_statement` rolled itself back before returning the error.
            Ok(Err(error)) => return Some(Err(error)),
            Err(panic) => {
                std::mem::replace(&mut held, StatementCheckpoint::None).rollback(graph);
                std::panic::resume_unwind(panic);
            }
        };
        // The write-ahead frame is appended before the write is final; a
        // refusal rolls the statement back through the journal it kept.
        if let Err(message) = self.log_working_commit(graph) {
            std::mem::replace(&mut held, StatementCheckpoint::None).rollback(graph);
            return Some(Err(KgError::DurabilityFailed { message }));
        }
        held.commit(graph);
        graph.set_version(start_version + 1);
        graph.maybe_spill_columns();
        graph.compact_columns_if_fragmented();
        crate::graph::handle::compact_dir_graph(graph);
        self.in_place_commits.fetch_add(1, Ordering::Relaxed);
        Some(Ok(outcome))
    }

    /// [`Self::execute_auto_commit`] with `between` called after each
    /// execution and before its commit, with the 1-based attempt number. A test
    /// seam: it lets a competing commit land inside the race window.
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    pub fn execute_auto_commit_observed(
        &self,
        query: &str,
        opts: &ExecuteOptions<'_>,
        attempts: u32,
        between: &mut dyn FnMut(u32),
    ) -> Result<ExecuteOutcome, KgError> {
        let mut attempt = 1;
        loop {
            let mut tx = self.begin();
            let working = tx.working_mut()?;
            let outcome = execute_mut(working, query, opts)?;
            between(attempt);
            match self.commit(tx, true) {
                CommitOutcome::Committed { .. } | CommitOutcome::NoWritesNoOp => {
                    return Ok(outcome)
                }
                CommitOutcome::ConflictDetected {
                    current_version,
                    base_version,
                } => {
                    if attempt >= attempts {
                        return Err(KgError::TransactionConflict {
                            base_version,
                            current_version,
                        });
                    }
                    attempt += 1;
                }
                // Exhaustive on purpose: an outcome added later must be decided
                // here, not fall into a catch-all that could read as success.
                CommitOutcome::DurabilityFailed { error } => {
                    return Err(KgError::DurabilityFailed { message: error });
                }
                CommitOutcome::OntologyViolated { error } => return Err(*error),
            }
        }
    }

    /// Save the session to `path` unless it is unchanged since the last
    /// successful checkpoint recorded in `last_version`.
    ///
    /// **First call always writes.** `last_version` starts `None` for the
    /// process, and the file on disk may predate it entirely (a stale `.kgl`,
    /// or a graph mutated and never checkpointed by a previous run), so no
    /// version comparison can be trusted until this process has written one.
    ///
    /// **Version read before the save, never after.** A commit landing between
    /// the read and the save's lock acquisition makes the recorded version one
    /// behind what reached disk, so the next call re-saves: a redundant write.
    /// Recording afterwards fails the other way: that commit would be recorded
    /// as saved when it was not, and the next call would skip it.
    ///
    /// A failed save leaves `last_version` untouched, so a retry still writes.
    /// Holding `last_version` as `&mut` is what serializes two checkpoints of
    /// one session; a caller sharing it across threads keeps it in a `Mutex`
    /// and holds the guard across this call.
    pub fn checkpoint_if_changed(
        &self,
        path: &std::path::Path,
        last_version: &mut Option<u64>,
    ) -> Result<CheckpointOutcome, String> {
        let version = self.version();
        if *last_version == Some(version) {
            return Ok(CheckpointOutcome::Skipped(version));
        }
        self.save(&path.to_string_lossy(), true)?;
        *last_version = Some(version);
        Ok(CheckpointOutcome::Written(version))
    }
}

/// Whether `clause` is a data read or write the in-place path may run: no
/// schema command, procedure, `UNION`, `LOAD CSV` or subquery.
fn is_plain_data_clause(clause: &Clause) -> bool {
    match clause {
        Clause::Match(_)
        | Clause::OptionalMatch(_)
        | Clause::Where(_)
        | Clause::Filter(_)
        | Clause::Return(_)
        | Clause::Finish
        | Clause::With(_)
        | Clause::OrderBy(_)
        | Clause::Skip(_)
        | Clause::Limit(_)
        | Clause::Unwind(_)
        | Clause::Create(_)
        | Clause::Set(_)
        | Clause::Delete(_)
        | Clause::Remove(_)
        | Clause::Merge(_) => true,
        Clause::Foreach { body, .. } => body.iter().all(is_plain_data_clause),
        _ => false,
    }
}
