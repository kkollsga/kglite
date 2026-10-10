//! Auto-commit execution for [`KgliteBackend`]: reads run on a snapshot, and
//! every mutation (data or schema) runs as a one-shot transaction that commits
//! before `execute()` returns.
//!
//! Neo4j commits an auto-commit transaction when its result stream is fully
//! consumed. boltr hands the backend no hook after streaming, and the engine
//! materialises a statement's rows before the commit anyway, so the commit
//! happens here, at RUN. A write that fails, conflicts or cannot be logged
//! sends no rows and applies nothing. The one observable difference: a RESET
//! or disconnect between RUN and PULL does not undo the write. A RESET or
//! disconnect *during* the statement cancels it before anything is published.

use super::*;

/// Internal attempts at an auto-commit write that loses an optimistic commit
/// race. Drivers never retry `session.run`, so the server does.
pub(super) const OPTIMISTIC_ATTEMPTS: u32 = 3;

/// Whether `query` writes data or schema, as the executor would run it.
/// `EXPLAIN` of a write only describes a plan, so it is a read: running it
/// against a working copy would publish an unchanged graph under a new version.
pub(super) fn is_write_statement(query: &str) -> Result<bool, BoltError> {
    let (parsed, is_mutation) = cypher::parse_with_mutation_check(query).map_err(kg_to_bolt)?;
    Ok(is_mutation && !parsed.explain)
}

/// The refusal of a write in a read-mode session or transaction, with the
/// code and wording Neo4j uses.
pub(super) fn access_mode_error(scope: &str) -> BoltError {
    BoltError::Query {
        code: "Neo.ClientError.Statement.AccessMode".into(),
        message: format!(
            "Writing in read access mode not allowed: this {scope} was opened with mode \"r\" \
             (read access). Use a write session or transaction."
        ),
    }
}

impl KgliteBackend {
    /// Run one auto-commit statement. `read_mode` is the RUN's `mode: "r"`.
    pub(super) async fn execute_auto_commit(
        &self,
        query: &str,
        parameters: &HashMap<String, BoltValue>,
        meta: &TxMeta,
        read_mode: bool,
        cancel: &kglite::api::session::CancelToken,
    ) -> Result<ResultStream, BoltError> {
        // The executor's parse cache makes the engine's own parse free.
        if !is_write_statement(query)? {
            return off_async_worker(|| {
                let kg_params = decode_params(parameters)?;
                let started = Instant::now();
                let snapshot = self.session.snapshot();
                let mut opts = self.execute_opts(&kg_params, meta);
                opts.cancel = Some(cancel.clone());
                let outcome = kglite::api::session::execute_read(&snapshot, query, &opts)
                    .map_err(kg_to_bolt)?;
                finish_stream(outcome.result, "r", outcome.explain, started)
            });
        }
        if self.readonly {
            return Err(read_only_refusal(
                "server is read-only — mutations rejected (--readonly flag)",
            ));
        }
        if read_mode {
            return Err(access_mode_error("session"));
        }

        let kg_params = decode_params(parameters)?;
        let started = Instant::now();
        let mut opts = self.execute_opts(&kg_params, meta);
        opts.cancel = Some(cancel.clone());
        // Queue mode holds the slot to the end of the function, past the
        // publish; optimistic mode takes none and retries a lost race. A
        // statement the engine's group-commit queue takes holds the slot
        // shared, so concurrent ones reach the queue and share a log barrier
        // while an open write transaction still excludes them.
        let (_slot, attempts): (Option<Box<dyn Send>>, u32) =
            if self.writer.config().mode == WriteConcurrency::Queue {
                if self.session.auto_commit_is_grouped(query, &opts) {
                    (Some(Box::new(self.auto_commit_shared_slot().await?)), 1)
                } else {
                    (Some(Box::new(self.auto_commit_slot().await?)), 1)
                }
            } else {
                (None, OPTIMISTIC_ATTEMPTS)
            };
        let result = off_async_worker(|| {
            self.session
                .execute_auto_commit(query, &opts, attempts)
                .map(|outcome| outcome.result)
                .map_err(kg_to_bolt_logged)
        })?;
        // Neo4j's summary types: `s` schema, `rw` a write that returns rows,
        // `w` a write that does not.
        let type_str = if is_schema_ddl(query) {
            "s"
        } else if result.columns.is_empty() {
            "w"
        } else {
            "rw"
        };
        finish_stream(result, type_str, false, started)
    }

    /// Wait for a shared permit on the writer slot, for a grouped statement.
    async fn auto_commit_shared_slot(&self) -> Result<SharedPermit, BoltError> {
        let started = Instant::now();
        let permit = self
            .writer
            .acquire_shared(|holder| self.reclaim_idle_writer(holder))
            .await
            .map_err(|timed_out| wait_timeout_error("auto-commit", &timed_out))?;
        tracing::debug!(
            waited_ms = started.elapsed().as_millis() as u64,
            "acquired a shared writer permit"
        );
        Ok(permit)
    }

    /// Wait for the writer slot as an auto-commit statement. The returned guard
    /// keeps the slot's idle reclaim from reaping a running statement.
    async fn auto_commit_slot(&self) -> Result<(WriterPermit, impl Sized), BoltError> {
        let id = self.tx_counter.fetch_add(1, Ordering::Relaxed);
        let permit = self
            .acquire_writer_slot(&format!("auto-commit-{id}"))
            .await?;
        let running = permit.activity().begin_query();
        Ok((permit, running))
    }
}

/// [`kg_to_bolt`], logging the one failure an operator must see: a write the
/// log rejected, which the engine did not publish and the server does not
/// acknowledge.
fn kg_to_bolt_logged(error: kglite::api::KgError) -> BoltError {
    if error.code() == kglite::api::KgErrorCode::DurabilityFailed {
        tracing::error!(
            error = %error,
            "auto-commit rejected: the write could not be logged, so it was not applied"
        );
    }
    kg_to_bolt(error)
}
