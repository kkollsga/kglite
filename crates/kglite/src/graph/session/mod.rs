//! `kglite::api::session` — canonical query + transaction surface.
//!
//! Pure-Rust, no PyO3, no async, no transport. Bindings (pyapi,
//! bolt-server, mcp-server, future Go/TS) wrap this module's types
//! and free functions. The Cypher pipeline orchestration + the
//! snapshot/working CoW transaction mechanics live here exactly
//! once.
//!
//! **Why this module exists.** Before Phase E, the same pipeline
//! (parse → validate → rewrite_text_score → optimize → mark_lazy →
//! mutation gate → execute) was duplicated three times — once in
//! `src/graph/pyapi/kg_core.rs::cypher`, once in
//! `crates/kglite-mcp-server/src/tools.rs::cypher_query`, and once
//! in `crates/kglite-bolt-server/src/backend.rs`. The CoW
//! transaction state was duplicated twice (pyapi/transaction.rs +
//! bolt-server backend). That drift cost the team twice in real
//! bugs: `validate_schema` was missing from two consumers; the
//! bolt-server's incorrect `mark_lazy_eligibility` call returned
//! 0 rows for any non-ORDER-BY RETURN until the robustness pass
//! surfaced it.
//!
//! See [`bolt_implementation.md`](../../../bolt_implementation.md)
//! Phase E for the full rationale.
//!
//! ## Surface
//!
//! - [`Session`] — shared graph state with commit-swap semantics
//!   (`Arc<DirGraph>` behind a `Mutex` for atomic swap). Bindings
//!   wrap a Session inside their own concurrency model.
//! - [`Transaction`] — snapshot/working CoW state, built via
//!   [`Session::begin`] and finalized via [`Session::commit`] or
//!   [`Session::rollback`].
//! - [`execute_read`] / [`execute_mut`] — pure-Rust pipeline
//!   orchestration. Bindings call these for every Cypher query
//!   (auto-commit reads use `execute_read` against a snapshot;
//!   in-transaction queries use the helpers on `Session` that
//!   route reads vs writes against `Transaction::current()` vs
//!   `Transaction::working_mut()`).
//! - [`ExecuteOptions`] — single struct for all per-query knobs
//!   (params, deadline, max_rows, lazy_eligible flag, disabled
//!   planner passes, optional embedder reference).
//! - [`ExecuteOutcome`] — wraps `CypherResult` with `is_mutation`,
//!   `output_format`, `explain` flags that callers need for
//!   serialization decisions.
//! - [`CommitOutcome`] — `NoWritesNoOp` / `Committed` /
//!   `ConflictDetected` so the binding maps to its own error type
//!   (PyErr / BoltError / etc.).

pub use self::execute::{execute_mut, execute_read, ExecuteOptions, ExecuteOutcome};
pub use self::transaction::{CommitOutcome, Session, Transaction};

pub(crate) mod execute;
pub(crate) mod transaction;
