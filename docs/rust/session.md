# `kglite::api::session`

This is the binding-shared, synchronous execution and transaction core. Rust
embedders and Rust-side wrappers call it directly; non-Rust bindings reach the
subset exposed by `kglite-c`.

## Canonical pipeline

```rust
use kglite::api::session::{execute_read, ExecuteOptions};
use std::collections::HashMap;

let params = HashMap::new();
let opts = ExecuteOptions::eager(&params);
let outcome = execute_read(&graph, "MATCH (n) RETURN count(n)", &opts)?;
```

`execute_read`/`execute_mut` own parse → schema validation → optimization →
execution. `ExecuteOptions` carries parameters, deadline, max-row/work budget,
lazy eligibility, disabled optimizer passes, embedder/value codecs,
cancellation, write scope, and write provenance. Start with
`ExecuteOptions::eager(&params)` and override only the options the binding
actually exposes.

The core runs to completion on the calling thread. Bindings own async runtimes,
transport, logging, iteration/chunking, error presentation, embedder lifecycle,
and value conversion.

## Session and transaction

`Session` owns the current `Arc<DirGraph>`. A snapshot briefly locks the owner
only to clone the Arc, then reads a stable immutable graph without holding the
lock. Successful writers publish with an atomic Arc swap.

```rust
let session = Session::new(graph);
let snapshot = session.snapshot();
let mut tx = session.begin();
let working = tx.working_mut()?; // lazy backend-specific fork on first write
// execute_mut(working, query, &opts)?;

match session.commit(tx, true) {
    CommitOutcome::NoWritesNoOp => {}
    CommitOutcome::Committed { .. } => {}
    CommitOutcome::ConflictDetected { .. } => { /* retry or surface */ }
}
```

`begin()` is O(1); the working fork materializes only at first mutation.
`begin_read()` never permits `working_mut`. Reads inside a write transaction
route to the working graph after the first mutation. Readers holding a prior
snapshot continue seeing it after commit; new snapshots see the committed graph.

Pass `check_occ=true` in production so a transaction based on a stale version
returns `ConflictDetected`. Last-writer-wins is not a safe default.

## Binding models

- Python exposes direct `KnowledgeGraph`, explicit `Session`,
  `open_session`, `begin`, and `begin_read`; it is not GIL-dependent at the
  core boundary.
- Bolt shares `Arc<Session>` across connections and keeps per-Bolt-session
  transaction state.
- MCP uses the same session pipeline for graph tools and optional writable
  lifecycle operations.
- C ABI v1 exposes sessions and atomic mutation batches but no explicit
  begin/commit handle. Do not call nonexistent `kglite_session_begin` APIs.

## Cancellation and rollback

Deadlines/max-row budgets return typed errors. Cancellation is a
binding-provided flag with the lifetime required by `ExecuteOptions`; do not
pass a short-lived request-local reference where the API requires a static
flag. Direct `execute_mut` mutates its graph in place. For rollback on failure,
execute against a transaction working fork or use the `Session` writer path and
publish only on success.

## Lazy results

Bindings without a lazy materializer must use eager options. If
`lazy_eligible=true`, the outcome may contain a lazy descriptor instead of
materialized rows; the wrapper must implement the descriptor contract before
exposing that mode.

## Source map

- `graph/session/execute.rs` — options/outcome and canonical pipeline.
- `graph/session/transaction.rs` — `Session`, `Transaction`, OCC, snapshots.
- `api::session` — supported public re-exports.
- `crates/kglite-py/src/graph/pyapi/` — Python wrapper.
- `crates/kglite-bolt-server/src/backend.rs` — Bolt wrapper.
- `crates/kglite-mcp-server/src/tools.rs` — MCP wrapper.
- [C ABI](c-abi.md) — non-Rust handle/status contract.
