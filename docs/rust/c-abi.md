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

## Opaque handles

`KgliteGraph`, `KgliteSession`, `KgliteCypherResult`, and `KgliteEmbedder` are
opaque. Create/load them only through exported constructors and release them
with their matching `*_free` function. Null-safe free functions simplify error
paths. Never copy/dereference the structs or free Rust memory with the host
allocator.

## Lifecycle and persistence

The header exposes:

- graph creation by storage mode, `.kgl`/RDF loading, graph generation, and
  blueprint construction;
- atomic/durable save, byte serialization, and schema JSON;
- session construction plus read/mutation execution with timeout/row budgets;
- read and mutation batches, including atomic edge batches;
- JSON result metadata/rows, memory statistics, and embedder binding.

`.kgl` is the cross-binding handoff format. The current writer emits RGF
v5/Postcard; supported v4 files remain readable and v3 is refused with a clear
rebuild message.

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
