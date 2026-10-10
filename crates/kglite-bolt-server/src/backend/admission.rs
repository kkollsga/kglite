//! Commit publication and writer-slot admission for [`KgliteBackend`]: the
//! outcome-to-wire mapping shared by `COMMIT` and one-shot schema statements,
//! and the slot acquire / idle-reclaim / reaped-handle error helpers.

use super::writer_slot::WaitTimedOut;
use super::*;

impl KgliteBackend {
    /// Discard the transaction `handle` because it sat idle while writers
    /// queued behind it. Dropping its `TxState` releases the slot.
    pub(super) fn reclaim_idle_writer(&self, handle: &str) {
        let removed = {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.remove(handle)
        };
        if removed.is_some() {
            self.reaped.record(handle);
            tracing::warn!(
                tx = %handle,
                "rolled back an idle write transaction: writers were waiting and it had \
                 no activity past --writer-idle-timeout"
            );
        }
    }

    /// The error for a message on a transaction that no longer exists:
    /// specific when an idle reclaim discarded it, otherwise `unknown`.
    pub(super) fn missing_tx_error(&self, handle: &str, otherwise: String) -> BoltError {
        if self.reaped.contains(handle) {
            BoltError::Query {
                code: "Neo.ClientError.Transaction.TransactionTimedOut".into(),
                message: format!(
                    "transaction {handle} was rolled back: it was idle past \
                     --writer-idle-timeout while other writers were waiting for the \
                     writer slot. Nothing it wrote was applied."
                ),
            }
        } else {
            BoltError::Transaction(otherwise)
        }
    }

    /// Commit `tx` through the session — OCC check, durable log append, publish —
    /// and map each outcome to what the wire must say. Shared by an explicit
    /// `COMMIT` and the one-shot transaction an auto-commit schema statement
    /// runs in, so the two cannot disagree about a conflict or a log failure.
    pub(super) fn publish(
        &self,
        tx: kglite::api::session::Transaction,
        session_id: &str,
        tx_id: &str,
    ) -> Result<(), BoltError> {
        let (outcome, ontology_warnings) =
            self.session.commit_reporting(tx, /* check_occ = */ true);
        // A COMMIT's SUCCESS has no place for them, so the operator reads
        // the `warn`-level end-of-transaction findings in the log.
        for warning in &ontology_warnings {
            tracing::warn!(session_id = %session_id, tx = %tx_id, "{warning}");
        }
        match outcome {
            kglite::api::session::CommitOutcome::NoWritesNoOp => {
                tracing::debug!(
                    session_id = %session_id,
                    tx = %tx_id,
                    "commit (no-op; no mutations)"
                );
            }
            kglite::api::session::CommitOutcome::Committed { new_version } => {
                tracing::debug!(
                    session_id = %session_id,
                    tx = %tx_id,
                    new_version,
                    "commit (with mutations)"
                );
            }
            // `--durability full`/`normal`: the frame could not be appended, so
            // the engine did not publish the commit (append-then-publish — see
            // `Session::commit`). The client must see FAILURE, because the
            // alternative is a driver that returns success for a write the
            // server deliberately discarded. `Backend` rather than `Query`: the
            // statement was fine and re-running it may well work, but nothing
            // about it can be fixed client-side, and the Neo4j taxonomy has no
            // retriable class that means "the server's disk answered no".
            kglite::api::session::CommitOutcome::DurabilityFailed { ref error } => {
                tracing::error!(
                    session_id = %session_id,
                    tx = %tx_id,
                    error = %error,
                    "commit rejected: the write could not be logged, so it was not applied"
                );
                return Err(BoltError::Backend(format!(
                    "commit was NOT applied — the write-ahead log rejected it and the \
                     server does not acknowledge writes it cannot log: {error}"
                )));
            }
            // A rule that demands something be present failed on the
            // transaction's end state. Nothing was published; the typed
            // violation reaches the client with its structured prefix, as a
            // statement-level refusal does.
            kglite::api::session::CommitOutcome::OntologyViolated { error } => {
                tracing::debug!(
                    session_id = %session_id,
                    tx = %tx_id,
                    "commit refused: the transaction breaks a declared ontology rule"
                );
                return Err(crate::error_map::kg_to_bolt(*error));
            }
            kglite::api::session::CommitOutcome::ConflictDetected {
                current_version,
                base_version,
            } => {
                tracing::debug!(
                    session_id = %session_id,
                    tx = %tx_id,
                    current_version,
                    base_version,
                    "commit conflict — another writer committed first"
                );
                // `BoltError::Query` rather than `BoltError::Transaction`:
                // boltr maps the latter to
                // `Neo.ClientError.Transaction.TransactionStartFailed` — wrong
                // twice over, since the transaction started fine and the
                // `ClientError` class tells every Neo4j driver the failure is
                // *not* retriable. A lost OCC race is the textbook retriable
                // failure, so the code must sit in the `TransientError` class:
                // `session.execute_write` then re-runs the unit of work on a
                // fresh transaction (fresh base version) without the caller
                // writing a retry loop at all. `Query` lets us set the code
                // directly; the string itself comes from the shared taxonomy
                // so this site and the embedded/pyo3 path cannot drift apart.
                return Err(BoltError::Query {
                    code: kglite::api::KgErrorCode::TransactionConflict
                        .neo4j_status_code()
                        .into(),
                    message: format!(
                        "Transaction conflict: graph was modified by another committer \
                         since this transaction's BEGIN (base version {base_version}, \
                         current version {current_version}). Retry the transaction."
                    ),
                });
            }
            // `CommitOutcome` is `#[non_exhaustive]`: an outcome this build
            // does not recognise reaches the error path, never the success
            // path. Fail closed — the engine only ever adds outcomes that mean
            // "not published".
            ref other => {
                tracing::error!(
                    session_id = %session_id,
                    tx = %tx_id,
                    outcome = ?other,
                    "commit returned an outcome this build does not recognise"
                );
                return Err(BoltError::Backend(
                    "commit returned an unrecognised outcome; the transaction was not applied"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Wait for the writer slot for transaction `handle`, mapping a wait
    /// timeout to the retriable `LockAcquisitionTimeout` failure.
    pub(super) async fn acquire_writer_slot(
        &self,
        handle: &str,
    ) -> Result<WriterPermit, BoltError> {
        let started = Instant::now();
        let permit = self
            .writer
            .acquire(handle, |holder| self.reclaim_idle_writer(holder))
            .await
            .map_err(|timed_out| wait_timeout_error(handle, &timed_out))?;
        tracing::debug!(
            tx = %handle,
            waited_ms = started.elapsed().as_millis() as u64,
            "acquired the writer slot"
        );
        Ok(permit)
    }
}

/// The retriable failure for a write that gave up waiting for the writer slot.
pub(super) fn wait_timeout_error(handle: &str, timed_out: &WaitTimedOut) -> BoltError {
    tracing::warn!(
        tx = %handle,
        waited_ms = timed_out.waited.as_millis() as u64,
        "write transaction gave up waiting for the writer slot"
    );
    BoltError::Query {
        code: "Neo.TransientError.Transaction.LockAcquisitionTimeout".into(),
        message: format!(
            "Could not begin a write transaction: another write transaction held \
             the writer slot for {:.1}s (--writer-wait-timeout). Retry the \
             transaction.",
            timed_out.waited.as_secs_f64()
        ),
    }
}
