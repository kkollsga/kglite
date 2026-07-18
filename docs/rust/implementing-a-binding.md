# Implementing a binding

Choose the boundary by implementation language:

- A Rust-side wrapper (PyO3, Bolt, MCP, a Rust service) calls
  `kglite::api::*` directly.
- A non-Rust wrapper (cgo, napi, JNI, P/Invoke, Swift FFI) calls the supported
  [`kglite-c` ABI](c-abi.md).

Precompiled C ABI libraries are not currently attached to releases; non-Rust
bindings must build and package the matching `kglite-c` library for each target.

Do not bind internal `graph::*`/`datatypes::*` modules or recreate the query
pipeline in the wrapper.

## Boundary principle

Core owns synchronous, binding-independent behavior: graph lifecycle, storage
configuration, typed errors, the session/transaction state machine, the
canonical Cypher pipeline, and embedder registration. Wrappers own runtime
idioms: async scheduling, wire/value conversion, display/error formatting,
iteration/chunking, logging, teardown, authentication, and transport.

Cypher is the first choice for query-expressible features. Add direct API only
for lifecycle/configuration/registration work Cypher cannot express. Read the
full [boundary principle](boundary-principle.md) before adding a surface.

## Rust-side wrapper

```rust
use kglite::api::{io::load_file, session::{execute_read, ExecuteOptions}};
use std::collections::HashMap;

let graph = load_file("graph.kgl")?;
let params = HashMap::new();
let opts = ExecuteOptions::eager(&params);
let outcome = execute_read(&graph, "MATCH (n) RETURN count(n)", &opts)?;
```

Use `api::session::{execute_read, execute_mut}` so parsing, schema validation,
optimizer passes, budgets, cancellation, write scope, and provenance remain
identical across wrappers. Use `Session`/`Transaction` when failed mutations
must not publish.

`ExecuteOptions::cancel` has the lifetime required by the public signature; do
not hand it an ordinary request-local atomic. A wrapper can enforce its own
request deadline or own a cancellation flag with a sufficiently long lifetime.

RDF ingestion is feature-gated and lives under
`kglite::api::io::{load_rdf, load_ntriples}`. Code-graph construction belongs
to codingest; link its own documentation rather than promising an unverified
function signature here.

## Non-Rust wrapper

Include the generated header and use only exported functions:

```c
#include "kglite.h"

KgliteGraph *graph = NULL;
const char *error = NULL;
KgliteStatusCode status = kglite_load_file("graph.kgl", &graph, &error);
if (status != KGLITE_STATUS_CODE_OK) {
    /* copy/map error, then */
    kglite_free_string(error);
}
```

Wrap opaque handles with deterministic `close`/`free` plus a defensive
finalizer. Every returned Rust string uses `kglite_free_string`; every graph,
session, result, and embedder uses its matching free function. Decode result
JSON before freeing its owning result handle.

ABI v1 supports sessions, read/mutation options, and atomic mutation batches.
It does not expose explicit transaction begin/commit calls. Model explicit
transactions only after the ABI grows a real handle contract; never invent
`kglite_session_begin` in wrapper code.

## Values and errors

Rust wrappers map all `Value` variants explicitly, including Timestamp, List,
Map, Node, Relationship, and Path. Non-Rust wrappers decode the JSON result
shape supplied by `kglite-c`.

Preserve `KgErrorCode`/`KgliteStatusCode` and the human message separately.
Map all 17 engine codes, including cancellation, plus the C-boundary codes for
invalid UTF-8/null pointers. Do not infer categories from message text.

## Storage and persistence

Expose memory/mapped/disk choice without changing semantics. The current `.kgl`
writer emits RGF v5 with an explicit Postcard codec; readers accept supported
v4 files and reject v3 with a rebuild message. A binding must not serialize
internal structs independently or promise compatibility broader than the core
reader.

For cross-package handoff, the declared consumer version floor must read the
format written by the linked engine. Verify the exact artifact in a clean
consumer package, not only from the workspace source tree.

## Surface checklist

1. Lifecycle: create/open/load/save/bytes/free and storage configuration.
2. Query: canonical session pipeline, parameters, budgets, typed results.
3. Transactions/concurrency: snapshots, serialized shared writes, OCC mapping.
4. Errors: stable codes, messages, null/UTF-8/allocation failures.
5. Optional capabilities: embedder, RDF, blueprint, schema/introspection.
6. Documentation: copy-paste install and first-query example for the target.

Avoid mirroring every Python convenience method. Cypher exposes algorithms,
procedures, spatial/temporal/text functions, and most per-query features to all
bindings without a larger ABI.

## Verification

- Compile a downstream consumer against the packaged crate/header/library.
- Exercise create/query/mutate/save/reload and every owned-result cleanup path.
- Test malformed UTF-8/null pointers for FFI wrappers.
- Test concurrent reads, serialized writes, conflicts, deadlines, row budgets,
  cancellation, and read-only rejection.
- Run storage parity and format round trips for memory/mapped/disk.
- Diff the Rust public API or generated C header in CI; update the baseline only
  for an intentional reviewed change.

See [API reference](api-reference.md), [Session](session.md),
[Embedding](embedding.md), and [C ABI](c-abi.md).
