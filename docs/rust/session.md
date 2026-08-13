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

`CommitOutcome` is `#[non_exhaustive]` — match a catch-all arm, not only the
variants that existed when you wrote the binding.

## Durable sessions

`Session::open_durable(graph, checkpoint_path, level)` builds a session that
appends every commit to the path's write-ahead sidecar. It performs the whole
open ordering, so a binding does not re-derive it: recover the sidecar, replay
the frames the loaded checkpoint does not already contain, *then* wrap the
backend for write capture, then open the log for append.

```rust
use kglite::api::durable::DurabilityLevel;

let session = Session::open_durable(graph, "/data/app.kgl", DurabilityLevel::Normal)?;
match session.commit(tx, true) {
    CommitOutcome::DurabilityFailed { error } => { /* IO/backend error, not retriable */ }
    _ => {}
}
session.sync()?;                 // on-demand barrier: what makes `Normal` usable
session.save("/data/app.kgl", true)?;  // checkpoint: folds the log in and truncates it
```

Contract points a binding has to honour:

- **The frame is appended between the OCC check and the publish.** A log that
  cannot be written blocks the commit rather than reporting success over an
  unlogged write; `CommitOutcome::DurabilityFailed { error }` says so, and the
  graph, its version and its readers are untouched. Surface it as an
  IO/backend error, not as a retriable conflict — re-running hits the same
  wall.
- **`save` is the checkpoint**: flush the log, stamp `checkpoint_lsn`, write
  the file, truncate the log. Its signature is unchanged, and it forces
  `fsync` on a durable session because it destroys the log that would
  otherwise still describe those commits.
- **`write()` / `transact` are not logged paths** and are unsupported on a
  durable session. Taking one anyway latches the session: every later
  durability operation fails loudly until a checkpoint folds the direct write
  in. Callers with an error channel check
  `Session::check_direct_write_allowed` first; bindings should route mutations
  through `begin`/`commit` instead.
- **One durable owner per path.** A durable `Session` and a durable
  `KnowledgeGraph` over the same path are not a supported pair — the split
  checkpoint/next-LSN state is what the replay gate reads.
- **Refusals are explicit**, not silent degradation: disk-mode graphs at any
  logging level, a path another durable owner already wrapped, and — the
  data-safety one — level `Off` over a sidecar holding commits the checkpoint
  does not contain. Recovery on open is unconditional; a non-durable open of
  such a path is an error rather than a graph quietly missing acknowledged
  writes.

`Session::durability()` reports the level (`None` when the session logs
nothing). Non-durable sessions are unaffected in behaviour and in cost.
Levels, per-level loss windows and the measured cost of each are in
[Bolt server → Durability](../operators/bolt-server.md#durability), whose
`--durability` flag is this API's only server-side consumer.

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
- `graph/session/durable.rs` — `open_durable`, `sync`, the checkpoint order.
- `api::session` — supported public re-exports.
- `crates/kglite-py/src/graph/pyapi/` — Python wrapper.
- `crates/kglite-bolt-server/src/backend.rs` — Bolt wrapper.
- `crates/kglite-mcp-server/src/tools/` — MCP wrapper.
- [C ABI](c-abi.md) — non-Rust handle/status contract.
