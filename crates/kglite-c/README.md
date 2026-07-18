# kglite-c — C ABI for KGLite

`kglite-c` is the supported C boundary for non-Rust bindings such as cgo,
napi, JNI, P/Invoke, and Swift FFI. The pure-Rust engine owns Cypher, storage,
sessions, and persistence; this crate exposes a curated synchronous ABI through
the generated [`include/kglite.h`](include/kglite.h).

The generated header is the exact symbol, signature, status, and ownership
authority. Header drift is checked in CI. Precompiled C ABI libraries are not
currently attached to releases, so build and package the matching source for
each target platform:

```bash
cargo build -p kglite-c --release
```

## Minimal C example

```c
#include <stdio.h>
#include "kglite.h"

int main(void) {
    KgliteGraph *graph = NULL;
    const char *error = NULL;
    KgliteStatusCode status = kglite_load_file("graph.kgl", &graph, &error);
    if (status != KGLITE_STATUS_CODE_OK) {
        fprintf(stderr, "%s\n", error ? error : "load failed");
        kglite_free_string(error);
        return 1;
    }

    KgliteSession *session = NULL;
    status = kglite_session_new(graph, &session); /* moves graph ownership */
    if (status != KGLITE_STATUS_CODE_OK) {
        kglite_graph_free(graph);
        return 1;
    }

    KgliteCypherResult *result = NULL;
    status = kglite_session_execute_read(
        session,
        "MATCH (n) RETURN count(n) AS count",
        NULL,
        &result,
        &error
    );
    if (status != KGLITE_STATUS_CODE_OK) {
        fprintf(stderr, "%s\n", error ? error : "query failed");
        kglite_free_string(error);
        kglite_session_free(session);
        return 1;
    }

    const char *rows = kglite_cypher_result_rows_json(result);
    printf("%s\n", rows);
    kglite_free_string(rows);
    kglite_cypher_result_free(result);
    kglite_session_free(session);
    return 0;
}
```

## Ownership

- Input pointers are borrowed for the duration of the call.
- A successful `kglite_session_new` takes ownership of its graph handle.
- Opaque result/session/graph/embedder handles use their matching `*_free`
  function.
- Every returned Rust-owned string uses `kglite_free_string` exactly once.
- Null-safe free calls are allowed; double-free and foreign-allocator pointers
  are not.

Fallible calls reset non-null output slots before validation. Valid calls that
panic inside Rust are contained and reported as `KGLITE_STATUS_CODE_INTERNAL`;
dangling or invalid caller pointers remain outside the ABI contract.

## Errors and runtime model

`KgliteStatusCode` contains engine status codes 1–17 plus boundary-only 100+
codes for conditions such as invalid UTF-8 and null pointers. Preserve the code
and message separately; do not classify failures by parsing text. The ABI is
synchronous, so each binding owns async scheduling, logging, display, and
teardown conventions.

## Versioning

`kglite-c` follows the workspace version. `kglite_abi_version()` reports the
linked runtime version, and the `.kgl` reader/writer follows the engine's saved
format lifecycle. See the [C ABI guide](../../docs/rust/c-abi.md) and
[binding guide](../../docs/rust/implementing-a-binding.md).

## License

MIT, matching the KGLite workspace.
