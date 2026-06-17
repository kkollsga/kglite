# `kglite::api::session` — the binding-shared core

This document is written for **binding implementers**: anyone wiring
kglite into a new language runtime (Go via cgo, TypeScript via napi,
Java via JNI, …) or a new transport (a custom RPC layer, an
embedded HTTP service, etc.). If you only use kglite from Python or
through the bundled Bolt / MCP servers, you don't need to read this —
those bindings already wrap the session module for you.

## What the module is

`kglite::api::session` is the single source of truth for the
canonical Cypher pipeline and the snapshot/working CoW transaction
mechanics. Every kglite binding (the PyO3 surface, the
`kglite-bolt-server`, the `kglite-mcp-server`) wraps the same two
free functions and the same two types:

```rust
// Free functions — the pipeline orchestration:
pub fn execute_read(graph: &DirGraph,    query: &str, opts: &ExecuteOptions) -> Result<ExecuteOutcome, KgError>;
pub fn execute_mut (graph: &mut DirGraph, query: &str, opts: &ExecuteOptions) -> Result<ExecuteOutcome, KgError>;

// Types — the transaction state machine:
pub struct Session { … }       // shared Arc<DirGraph> with commit-swap semantics
pub struct Transaction { … }   // snapshot + lazy working CoW + base_version
pub enum   CommitOutcome { NoWritesNoOp, Committed{…}, ConflictDetected{…} }
```

It is **pure Rust** — no PyO3, no async runtime, no transport. The
binding chooses its own concurrency (Python's GIL, bolt-server's
per-tx `Mutex`, your hypothetical Go binding's `sync.Mutex`), its
own error mapping (PyErr, BoltError::Query, a Go `error`), and its
own value serialization (`PreProcessedValue`, `BoltValue`, MCP
JSON). The session module just runs the same code path beneath
them.

## Why it exists

Prior to Phase E, the same Cypher pipeline orchestration lived in
**three places** (pyapi, mcp-server, bolt-server) and the CoW
transaction code in **two** (pyapi, bolt-server). This duplication
silently cost correctness twice in a single sprint:

1. `validate_schema` was present in pyapi but missing from
   mcp-server and bolt-server — a query referencing an unknown
   label would error in Python but succeed-with-empty-result in
   Bolt and MCP.
2. `mark_lazy_eligibility` was wrongly included in bolt-server,
   producing 0-row `RETURN x` responses for any non-`ORDER BY`
   query (the lazy descriptor was populated but `result.rows` was
   empty).

A Go binding or a future async TypeScript binding would have
multiplied the drift. Phase E extracts the pipeline once, so future
bindings cannot accidentally diverge.

## Surface area

### `Session`

```rust
let session = Session::new(DirGraph::new());

let v0     = session.version();    // OCC base
let snap   = session.snapshot();   // cheap Arc clone; readers hold this
let tx     = session.begin();      // read-write
let read_tx = session.begin_read(); // read-only (working_mut rejected)
```

The outer `Mutex<Arc<DirGraph>>` is brief-acquire only. A
`snapshot()` call grabs the lock just long enough to `Arc::clone`
the inner — readers then hold a stable view via their Arc handle,
unaffected by subsequent commits.

### `Transaction`

```rust
let mut tx = session.begin();
let g_pre = tx.current().unwrap();       // &DirGraph (the snapshot view)

// First mutation materializes the working copy. Either Arc::try_unwrap
// (free — tx held the sole Arc ref) or a deep clone fallback.
let g_post = tx.working_mut()?;           // &mut DirGraph
// ... mutating Cypher executes against g_post ...

// Subsequent reads route through working automatically:
let _ = tx.current();                     // now &working, not &snapshot

assert_eq!(tx.has_writes(), true);
```

Read-only transactions reject `working_mut` with
`KgError::Argument`. Bindings surface the rejection as their typed
"read-only operation" error.

### `Session::commit`

```rust
match session.commit(tx, /* check_occ = */ true) {
    CommitOutcome::NoWritesNoOp => { /* nothing to do */ }
    CommitOutcome::Committed { new_version } => { /* readers see new graph on next snapshot() */ }
    CommitOutcome::ConflictDetected { current_version, base_version } => {
        // Another writer committed between this tx's begin() and commit().
        // Working copy is dropped; binding surfaces a typed conflict error.
    }
}
```

OCC is **opt-in** per call. Pass `true` to enforce the version
check; pass `false` for last-writer-wins semantics (some test
fixtures use this — production bindings should always pass `true`).
Bolt-server passes `true`.

### `ExecuteOptions`

```rust
let opts = ExecuteOptions {
    params: &params,                    // borrowed &HashMap<String, Value>
    deadline: Some(Instant::now() + Duration::from_secs(30)),
    max_rows: Some(10_000),
    lazy_eligible: false,                // true only if the binding has a lazy materializer (pyapi's ResultView does; bolt-server doesn't)
    disabled_passes: None,               // or Some(set) for user-toggle
    embedder: None,                      // Arc<dyn Embedder> required for text_score()
};
```

`lazy_eligible` matters: when `true`, the executor returns a
`CypherResult` whose `lazy` field may be `Some(LazyResultDescriptor)`,
and the binding must materialize it row-by-row on demand. Bindings
that don't have a lazy materializer must pass `false` — otherwise
they'll see empty `result.rows` for any `RETURN x` query without
`ORDER BY`. (This was the bolt-server bug fixed in C.6.)

## Snapshot isolation guarantees

| Scenario | Visibility |
|---|---|
| Reader inside tx_a sees its own pending writes | ✅ — `tx.current()` routes through `working` after the first `working_mut` |
| Outside reader (auto-commit) sees tx_a's pending writes | ❌ — they hold a snapshot Arc that doesn't include the in-flight `working` copy |
| Reader holding `session.snapshot()` from before tx_a.commit() | ❌ — they see the pre-commit graph; their Arc still points at the old inner |
| Reader who calls `session.snapshot()` after tx_a.commit() | ✅ — fresh Arc clones the post-commit graph |
| tx_b's reads after tx_a commits | tx_b's snapshot is fixed at tx_b's begin(); doesn't refresh mid-transaction |

The snapshot semantics are MVCC-style: each transaction sees a
stable view of the graph from its begin() moment, and commits are
atomic Arc swaps.

## Concurrency models

The session module is `Send + Sync`. Bindings layer their own
concurrency over it:

- **Python (pyapi)**: a single `KnowledgeGraph` wraps an
  `Arc<DirGraph>` directly. The GIL serializes pyapi calls; reads
  release the GIL via `py.detach()` for parallel readers. Sessions
  are implicit (one per `KnowledgeGraph`).
- **bolt-server**: an `Arc<Session>` shared across all connected
  Bolt clients. Each Bolt session owns a `Mutex<TxState>`; the
  outer Mutex is per-tx, not per-server, so concurrent writers in
  different Bolt sessions don't block each other on the open-tx
  path. They only contend on the brief `commit-swap` in
  `Session::commit`. OCC turns conflicting commits into
  `ClientError("Transaction conflict")`.
- **mcp-server**: stdio MCP is single-threaded per connection; no
  Mutex needed beyond what `Session` provides internally.

A new Go binding would typically use `sync.Mutex` around a
`*Session` handle; a new napi binding the same in async JS. The
session module imposes no preference.

## Sketch: wrapping from a new binding

A minimal wrapper looks like this (Go/cgo flavor; sketch only):

```go
// kglite-go: pseudo-cgo bindings to kglite::api::session
type Session struct { handle *C.KgliteSession }
type Tx       struct { handle *C.KgliteTransaction }

func (s *Session) Run(query string, params map[string]any) (Rows, error) {
    opts := buildExecuteOptions(params)
    res := C.kglite_session_execute_read(s.handle, cstr(query), opts)
    if res.err != nil {
        return nil, mapError(res.err)
    }
    return rowsFromOutcome(res), nil
}

func (s *Session) Begin() *Tx { return &Tx{handle: C.kglite_session_begin(s.handle)} }

func (s *Session) Commit(tx *Tx) error {
    out := C.kglite_session_commit(s.handle, tx.handle, /*check_occ=*/1)
    switch out.kind {
    case C.CommitOutcomeKind_NoWritesNoOp:    return nil
    case C.CommitOutcomeKind_Committed:        return nil
    case C.CommitOutcomeKind_ConflictDetected: return ErrConflict
    }
}
```

The hard part — the pipeline + CoW + OCC — is shared via the C ABI
exposure of session::*. Each binding only owns the marshalling
layer.

## What's NOT in this module

| Out of scope | Why |
|---|---|
| Async / Future / Tokio | The binding chooses. bolt-server uses tokio; pyapi is sync. The session pipeline itself is synchronous. |
| Lazy materializer | Lives in pyapi (`result_view.rs::materialise_lazy_row`). Bindings without a lazy path pass `lazy_eligible=false`. A future commit may lift it into session for shared use. |
| Streaming PULL n | Bolt's record-by-record PULL is implemented in bolt-server's `ResultStream`. Session returns a fully-materialized `CypherResult`; the binding handles chunking. |
| Routing / clustering / TLS | Transport concerns owned by the binding. |
| Value serialization | Each binding has its own value type (`PreProcessedValue`, `BoltValue`, MCP JSON). Session emits `kglite::Value`; the binding converts. |
| Embedder construction | Bindings construct (or load) the `Arc<dyn Embedder>` and pass it via `ExecuteOptions`. Session doesn't manage embedder lifecycles. |

## Where to read the code

- `src/graph/session/mod.rs` — module-level rationale + re-exports.
- `src/graph/session/execute.rs` — `ExecuteOptions`, `ExecuteOutcome`,
  `execute_read`, `execute_mut` (the canonical pipeline).
- `src/graph/session/transaction.rs` — `Session`, `Transaction`,
  `CommitOutcome` plus ~15 unit tests pinning the contract.
- `src/lib.rs::api::session` — public re-exports.
- `src/graph/pyapi/kg_core.rs::cypher` — Python wrapper (≈80 lines after Phase E).
- `crates/kglite-bolt-server/src/backend.rs` — Bolt wrapper (≈350 lines).
- `crates/kglite-mcp-server/src/tools.rs::run_cypher_inner` — MCP wrapper.
