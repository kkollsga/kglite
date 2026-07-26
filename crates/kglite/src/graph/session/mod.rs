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
//! See [`docs/history/bolt-implementation.md`](../../../../../docs/history/bolt-implementation.md)
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

use crate::datatypes::Value;
use crate::graph::schema::GraphBackend;
// `node_weight` is on the GraphRead trait; the wheel's import path
// did `pub use kglite_core::graph::*` glob which brought it in.
use crate::graph::storage::GraphRead;

/// Stack size a thread must have to run [`execute_read`] / [`execute_mut`]
/// safely — servers that dispatch queries onto their own threads should
/// configure their pool with this value.
///
/// The pipeline recurses once per level of expression/predicate nesting, in
/// the parser, in ~25 planner walkers, in the executor's predicate evaluator,
/// and again in the drop glue of the boxed AST. The parser caps nesting so
/// that recursion is bounded (a "simplify the query" parse error past the
/// budget), but the *bound* still has to fit in the thread's stack, and a
/// runtime default worker stack (tokio: 2 MiB) is not comfortably larger than
/// the deepest permitted tree in a debug build. A thread with less than this
/// cannot merely fail a query — a Rust stack overflow aborts the **process**,
/// so one client's deep query would take down every other session sharing it.
///
/// 8 MiB matches the main-thread default that the CLI and the Python wheel
/// already get for free, so every frontend has the same headroom.
pub const QUERY_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Resolve any `Value::NodeRef` entries in Cypher result rows to the
/// referenced node's `title` value. Called by bindings just before
/// emitting rows to their consumer (`PyDict`/`PyList` for the wheel,
/// `RecordMessage` for bolt-server, JSON for mcp-server). `NodeRef`
/// is an internal sentinel used by `collect()` / `WITH` to preserve
/// node identity through the planner — it should never appear in
/// output.
///
/// Lifted from the wheel crate in 0.10.1 so every binding can call
/// the same post-execute cleanup instead of re-implementing it.
pub fn resolve_noderefs(graph: &GraphBackend, rows: &mut [Vec<Value>]) {
    // Arena guard: node_weight materializes on the disk backend (arena
    // protocol in disk/graph.rs, enforced by a debug assert); no-op on
    // memory/mapped backends.
    let _arena_guard = graph.begin_query();
    for row in rows.iter_mut() {
        for val in row.iter_mut() {
            if let Value::NodeRef(idx) = val {
                let node_idx = petgraph::graph::NodeIndex::new(*idx as usize);
                if let Some(node) = graph.node_weight(node_idx) {
                    *val = node.title().into_owned();
                } else {
                    *val = Value::Null;
                }
            }
        }
    }
}
