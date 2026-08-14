# C ABI

`kglite-c` is the supported boundary for non-Rust bindings (cgo, napi, JNI,
P/Invoke, Swift, and similar FFIs). Rust embedders should call
`kglite::api::*` directly. The generated
[`kglite.h`](https://github.com/kkollsga/kglite/blob/main/crates/kglite-c/include/kglite.h)
is the exact symbol/signature authority; this page explains ownership and use.

## Build and versioning

```bash
cargo build -p kglite-c --release
```

The workspace version is lockstep across the engine, Python wrapper, C ABI,
servers, and CLI. `kglite_abi_version()` derives major/minor/patch from that
package version. Header drift is CI-gated; regenerate through the crate build,
never edit `kglite.h` by hand.

Precompiled C ABI libraries are not currently attached to releases. Build the
library from the matching workspace/crate source and package it for the target
platform alongside your binding.

## Status and error ownership

Every fallible call returns `KgliteStatusCode`:

```c
KgliteGraph *graph = NULL;
const char *error = NULL;
KgliteStatusCode status = kglite_load_file("graph.kgl", &graph, &error);
if (status != KGLITE_STATUS_CODE_OK) {
    fprintf(stderr, "%s\n", error ? error : "unknown kglite error");
    kglite_free_string(error);
    return 1;
}
```

Engine codes are `KGLITE_STATUS_CODE_CYPHER_SYNTAX` through
`KGLITE_STATUS_CODE_CANCELLED` (1–17). Boundary-only failures use 100+ such as
`INVALID_UTF8` and `NULL_POINTER`. Output handles/messages are reset before
validation, and any returned error string is Rust-owned until freed with
`kglite_free_string`.

Name a code with `kglite_status_code_name_static`, which returns a `'static`
pointer into the library's own data: no allocation, and **never free it**. A
binding that renders the name on every error — the usual shape, since the name
goes into the exception it raises — should prefer it. `kglite_status_code_name`
returns the same text as an owned copy that *must* be freed with
`kglite_free_string`; it predates the static form and stays for callers that
would rather free one uniform kind of string. Pick one per call site; freeing
the static pointer is undefined behaviour. Both return null for
`KGLITE_STATUS_CODE_OK`.

## Opaque handles

`KgliteGraph`, `KgliteSession`, `KgliteCypherResult`, `KgliteEmbedder`, and
`KgliteWriterLease` are opaque. Create/load them only through exported
constructors and release them with their matching `*_free` function. Null-safe
free functions simplify error paths. Never copy/dereference the structs or free
Rust memory with the host allocator.

## Lifecycle and persistence

The header exposes:

- graph creation by storage mode, `.kgl`/RDF loading, graph generation, and
  blueprint construction;
- `kglite_open_or_create_graph_in_mode`, which opens or creates a path in an
  explicit mode (null mode = honour what the checkpoint recorded) and reports
  any conversion through `out_converted_from`;
- `kglite_graph_storage_mode`, which reports the mode a graph handle is running
  in *right now* as an owned `"memory"` / `"mapped"` / `"disk"` string. It
  borrows the handle, so call it before `kglite_session_new` consumes it.
  `out_converted_from` above answers "what was it before?" and is null whenever
  nothing changed; this answers "what is it now?" unconditionally — the question
  after a creation, after an unspecified-mode open, or when asserting the mode
  you asked for is the mode you got;
- `kglite_writer_lease_acquire` / `kglite_writer_lease_free`. **Any caller that
  may save to a path must hold the lease across the whole read-modify-save
  interval; readers take none.** Two processes that both open, mutate, and save
  one path each publish a complete snapshot and the later one silently wins, so
  locking at save time is already too late. `timeout_ms = 0` fails fast, and a
  refusal (`KGLITE_STATUS_CODE_WRITER_LEASE_HELD`) names the holding process.
  The full write cycle is: acquire the lease → `kglite_open_or_create_graph_in_mode`
  → `kglite_session_new` → `kglite_session_execute_mut` → `kglite_session_save`
  → free the session → free the lease;
- atomic/durable save, byte serialization, and schema JSON;
- `kglite_session_save`, the checkpoint for a graph that has been moved into a
  session. `kglite_session_new` takes ownership of the graph handle, so a graph
  mutated through `kglite_session_execute_mut` is persisted from the session
  rather than from the (now-consumed) graph handle; `kglite_save_graph` stays
  the entry point for a graph that was never moved into one. The save writes
  through the session's own graph — it never copies the graph to checkpoint it
  — and is serialized against concurrent mutations on that session. The graph
  handle is consumed **only on `KGLITE_STATUS_CODE_OK`**: a failed
  `kglite_session_new` leaves ownership with the caller, who must still
  `kglite_graph_free` it;
- session construction plus read/mutation execution with timeout/row budgets;
- read and mutation batches, including atomic edge batches;
- JSON result metadata/rows, memory statistics, and embedder binding.

`.kgl` is the cross-binding handoff format. The current writer emits RGF
v6/Postcard and the current reader accepts v6 and v5. RGF v4/bincode and
older containers are rejected with a clear migration/rebuild message; convert
them with kglite 0.13.4 before crossing the C boundary. A v6 file cannot be
read by kglite 0.15.14 or earlier, so a prebuilt consumer must be rebuilt
against this engine before it is handed one.

## Sessions and transactions

Use `kglite_session_execute_read[_opts]` for reads and
`kglite_session_execute_mut[_opts]` for auto-committed mutations. Mutation
batches commit atomically. ABI v1 does **not** expose explicit begin/commit
transaction handles; do not invent wrapper calls such as
`kglite_session_begin`. A future ABI revision should add them only with a real
consumer and an ownership/error contract.

## Result access

Results remain owned by `KgliteCypherResult` until
`kglite_cypher_result_free`. Column and row helpers return JSON strings for
portable decoding in the host language. Copy/parse data before freeing the
result, and free every independently returned string with
`kglite_free_string`.

## Binding checklist

1. Validate UTF-8 and nullability before calls.
2. Map all status codes, including `CANCELLED`; preserve the message/code.
3. Wrap opaque handles in deterministic finalizers plus explicit close/free.
4. Keep async/runtime/logging/iteration style in the binding; the core is sync.
5. Test null outputs, double-free-safe cleanup paths, malformed JSON/UTF-8,
   timeout/budget failures, and concurrent session use.
6. Compile against the generated header and run the C-ABI integration/header
   drift checks for every release.

See [Implementing a binding](implementing-a-binding.md) for the architectural
boundary and [Session abstraction](session.md) for the native Rust pipeline.
