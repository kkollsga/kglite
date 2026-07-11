# kglite-c — C ABI for kglite

Stable `extern "C"` surface over the [kglite](https://crates.io/crates/kglite)
knowledge graph engine. Non-Rust bindings (Go via cgo, JavaScript via
napi, JVM via JNI, .NET via P/Invoke) consume a single C header
(`include/kglite.h`) rather than re-implementing wrappers in their
host language.

This crate is glue. The engine itself (Cypher pipeline, transaction
model, storage backends, dataset loaders) lives in the sibling
`kglite` crate. `kglite-c` exposes a curated subset of
`kglite::api::*` via `#[no_mangle] extern "C"` functions plus
cbindgen-generated header.

## Status: Phase H.2 (skeleton)

This is the initial skeleton — top-12 entry points (lifecycle /
session / Cypher / result accessors / error introspection / ABI
version). See `docs/rust/c-abi.md` for the design conventions and
the full Phase H roadmap (H.3 datasets + embedder, H.4 Go PoC
consumer, H.5 release coordination).

## Use from C

```c
#include <stdio.h>
#include "kglite.h"

int main(void) {
    KgliteGraph* graph = NULL;
    const char* err = NULL;
    KgliteStatusCode rc = kglite_load_file("graph.kgl", &graph, &err);
    if (rc != KGLITE_OK) {
        fprintf(stderr, "load failed: %s\n", err);
        kglite_free_string(err);
        return 1;
    }

    KgliteSession* session = NULL;
    kglite_session_new(graph, &session);
    // session takes ownership of graph; do NOT call kglite_graph_free

    KgliteCypherResult* result = NULL;
    rc = kglite_session_execute_read(
        session,
        "MATCH (n) RETURN count(n)",
        NULL,                       // no params
        &result,
        &err
    );
    if (rc != KGLITE_OK) {
        fprintf(stderr, "query failed: %s\n", err);
        kglite_free_string(err);
        kglite_session_free(session);
        return rc;
    }

    const char* cols = kglite_cypher_result_columns_json(result);
    const char* rows = kglite_cypher_result_rows_json(result);
    printf("columns: %s\n", cols);
    printf("rows: %s\n", rows);
    kglite_free_string(cols);
    kglite_free_string(rows);

    kglite_cypher_result_free(result);
    kglite_session_free(session);
    return 0;
}
```

## Use from Go (sketch — full Phase H.4 PoC pending)

```go
// #cgo LDFLAGS: -lkglite_c
// #include <kglite.h>
import "C"

func main() {
    var graph *C.KgliteGraph
    var errMsg *C.char
    rc := C.kglite_load_file(C.CString("graph.kgl"), &graph, &errMsg)
    // ...
}
```

## Memory ownership

Every function documents who owns what. The rules:

- Arguments by `*const c_char` / `*const T` — borrowed for the call.
- Arguments by `*mut T` (opaque handle) — borrowed for the call,
  caller still owns.
- Return values by `*mut T` — caller OWNS, must free via
  `kglite_<type>_free`.
- Return values by `*const c_char` — caller OWNS, must free via
  `kglite_free_string`.
- Return values by-value primitives — no ownership concern.

## Error handling

errno-style: every fallible function returns `KgliteStatusCode`
(`KGLITE_OK == 0` on success), with out-params for both the result
handle and an optional error message string. The error message,
when present, is owned and must be freed via `kglite_free_string`.

`KgliteStatusCode` variants 1-16 map 1:1 to `kglite::api::KgErrorCode`
variants. Bindings can pull the canonical human-readable name,
the Neo4j `Neo.ClientError.*` status code, or the HTTP status code
via:

```c
const char* kglite_status_code_name(KgliteStatusCode);
const char* kglite_status_code_neo4j_status(KgliteStatusCode);
uint16_t    kglite_status_code_http_status(KgliteStatusCode);
```

## Sync only

The C ABI is fully synchronous. Bindings own their own async/threading
model — Go uses goroutines, JS uses worker threads, JVM uses thread
pools, each wrapping the sync C calls. Async dataset fetchers
(`kglite::api::datasets::*`) are exposed via their `*_blocking`
companions in Phase H.3.

Fallible calls initialize every non-null output slot before validation. Rust
panics caused by valid calls are contained at the boundary and reported as
`KGLITE_STATUS_CODE_INTERNAL`; invalid or dangling caller pointers remain
outside the ABI contract.

## Versioning

`kglite-c` versions track `kglite`'s minor version. The
`kglite_abi_version()` function returns the runtime ABI version
for binding-author sanity checks.

## License

MIT (matches `kglite`).
