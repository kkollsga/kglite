# Implementing a binding

This document is the **deep-dive companion** to
[`embedding.md`](embedding.md) and [`session.md`](session.md). Those
two cover the broad shape of the engine and the canonical Cypher
pipeline. This document covers the concerns that surface specifically
when you sit down to write a new wrapper — error mapping, embedder
implementation, dataset wrapping patterns, the bridge-layer choice.

If you only want to embed kglite from a Rust binary, start with
[`embedding.md`](embedding.md) and stop there. If you want to
publish a Go, JavaScript, JVM, or other-language binding that other
people will depend on — read this one.

## Reference implementations

KGLite ships **three working bindings**. They are the canonical
worked examples; every pattern in this guide is grounded in one of
them.

| Binding | Path | Audience | Bridge style |
|---|---|---|---|
| **`kglite-py`** | `crates/kglite-py/` | `pip install kglite` users; Jupyter; the bundled MCP server | PyO3 (Rust → CPython C ABI) |
| **`kglite-bolt-server`** | `crates/kglite-bolt-server/` | Anything that talks the Neo4j Bolt protocol — `cypher-shell`, the official drivers, Neo4j Browser | Network protocol (no in-process binding; you connect over TCP) |
| **`kglite-mcp-server`** | `crates/kglite-mcp-server/` | LLM agents speaking the Model Context Protocol | Network protocol (stdio or TCP) |

When in doubt about an unfamiliar concern, the right move is to grep
the existing binding crates for how they handled it. They are the
deployed answer.

## The bridge-layer choice

Your binding needs to get values across the language boundary. KGLite
the engine is a Rust library — it doesn't know your language exists.
You have three options:

### Option 1 — Rust-to-Rust direct (no bridge)

The simplest case. Your "binding" is another Rust crate (a CLI tool,
a custom server, a worker process). You depend on `kglite` directly,
call `kglite::api::*` functions, get back native Rust types. No FFI,
no marshalling, no glue.

```toml
[dependencies]
kglite = "0.11"
```

Everything in this guide still applies, but the FFI sections are
informational rather than required.

The bolt-server crate is the canonical worked example of this style.
It depends on `kglite` directly, wraps `Session` in its own
connection-state struct, and never crosses a language boundary.

### Option 2 — Language-specific FFI (the common case)

You're writing a Go binding (cgo), a Node binding (napi-rs), a JVM
binding (jni-rs), a .NET binding (com-rs), etc. Each language
has a runtime-specific Rust crate that handles the lowest-level
marshalling.

Your binding becomes a Rust crate that:

1. Depends on `kglite` (the engine) **and** on the language-specific
   FFI crate (`napi`, `jni`, `pyo3`, etc.).
2. Wraps `kglite::api::*` functions with the FFI macros for the
   target runtime.
3. Builds as a `cdylib` (or whatever your runtime wants) and is
   loaded into the host process.

The pyapi crate (`crates/kglite-py/`) is the canonical worked example.
It's a 5k-line crate that wraps everything you'd want from Python:
`KnowledgeGraph` PyClass, `Selection` PyClass, NumPy/Pandas conversion,
async/GIL handling, the works. A Go binding aiming for similar
completeness would be similar size.

For the smaller-bridge approach (just the engine, no ergonomics) the
crate is more like 1k lines — see the `cgo sketch` in
[embedding.md](embedding.md#non-rust-bindings-via-the-c-abi).

### Option 3 — Use the `kglite-c` crate (the canonical non-Rust path)

`crates/kglite-c/` exposes `kglite::api::*` through a stable C ABI.
Every language with FFI (Go, JavaScript, JVM, .NET, …) can bind to
kglite through a single C header without writing Rust.

The conventions are fixed in [`c-abi.md`](c-abi.md); the short
version:

- `kglite_` prefix on every function.
- Opaque-handle pattern: `KgliteGraph*` / `KgliteSession*` /
  `KgliteCypherResult*` / `KgliteEmbedder*` / `KgliteSecClient*`.
- Errno-style errors: every fallible function returns
  `KgliteStatusCode` (0 = OK) with out-parameters for both the
  result handle and an owned error-message string. 16 status
  variants map 1:1 to `KgErrorCode`; three are C-ABI-specific
  (`KGLITE_STATUS_CODE_INVALID_UTF8`, `_NULL_POINTER`, `_OUT_OF_MEMORY`).
- Memory: caller frees every `*mut T` handle via the type's
  `kglite_<type>_free`; every owned out-string via the single
  `kglite_free_string`.
- Sync-only: bindings own their async/threading. Async dataset
  fetchers expose `*_blocking` companions.
- JSON-at-boundary: parameters in, rows out, dataset reports out —
  all as JSON strings. Caller parses with their language's stdlib
  JSON facility.

#### Setup

```toml
# Cargo.toml of an in-Rust consumer
[dependencies]
kglite-c = "0.13"     # cdylib + staticlib + rlib
```

For non-Rust consumers, link against `libkglite_c.{so,dylib,dll}`
and include the header that ships at
`crates/kglite-c/include/kglite.h`. Build the platform library from
the `kglite-c` source crate with `cargo build --release -p kglite-c`.
Precompiled C ABI libraries are not currently attached to releases; the
generated header is committed in the repository and included with the source
crate.

#### Worked example — cgo (Go)

```go
package kglite

/*
#cgo LDFLAGS: -lkglite_c
#include <stdlib.h>
#include "kglite.h"
*/
import "C"

import (
    "encoding/json"
    "errors"
    "fmt"
    "unsafe"
)

type Graph struct {
    handle *C.KgliteGraph
}

func LoadFile(path string) (*Graph, error) {
    cpath := C.CString(path)
    defer C.free(unsafe.Pointer(cpath))

    var graph *C.KgliteGraph
    var errMsg *C.char
    rc := C.kglite_load_file(cpath, &graph, &errMsg)
    if rc != C.KGLITE_STATUS_CODE_OK {
        defer C.kglite_free_string(errMsg)
        return nil, fmt.Errorf("load_file: %s", C.GoString(errMsg))
    }
    return &Graph{handle: graph}, nil
}

func (g *Graph) Close() {
    C.kglite_graph_free(g.handle)
    g.handle = nil
}

type Session struct {
    handle *C.KgliteSession
}

func (g *Graph) NewSession() (*Session, error) {
    var sess *C.KgliteSession
    rc := C.kglite_session_new(g.handle, &sess)
    if rc != C.KGLITE_STATUS_CODE_OK {
        return nil, errors.New("session_new failed")
    }
    // Graph ownership moved into the session.
    g.handle = nil
    return &Session{handle: sess}, nil
}

func (s *Session) Cypher(query string, params map[string]any) ([]map[string]any, error) {
    cquery := C.CString(query)
    defer C.free(unsafe.Pointer(cquery))

    var cparams *C.char
    if params != nil {
        paramJSON, _ := json.Marshal(params)
        cparams = C.CString(string(paramJSON))
        defer C.free(unsafe.Pointer(cparams))
    }

    var result *C.KgliteCypherResult
    var errMsg *C.char
    rc := C.kglite_session_execute_read(s.handle, cquery, cparams, &result, &errMsg)
    if rc != C.KGLITE_STATUS_CODE_OK {
        defer C.kglite_free_string(errMsg)
        return nil, fmt.Errorf("execute_read: %s", C.GoString(errMsg))
    }
    defer C.kglite_cypher_result_free(result)

    rowsJSON := C.kglite_cypher_result_rows_json(result)
    defer C.kglite_free_string(rowsJSON)

    var rows []map[string]any
    if err := json.Unmarshal([]byte(C.GoString(rowsJSON)), &rows); err != nil {
        return nil, err
    }
    return rows, nil
}
```

#### Worked example — napi-rs (Node.js)

```rust
// crates/kglite-js/src/lib.rs (in a hypothetical kglite-js crate)
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ffi::CString;

#[napi]
pub struct Graph {
    inner: *mut kglite_c::KgliteGraph,
}

#[napi]
impl Graph {
    #[napi(factory)]
    pub fn load_file(path: String) -> Result<Self> {
        let c_path = CString::new(path).map_err(|e| Error::from_reason(e.to_string()))?;
        let mut graph: *mut kglite_c::KgliteGraph = std::ptr::null_mut();
        let mut err_msg: *const std::ffi::c_char = std::ptr::null();
        let rc = unsafe {
            kglite_c::kglite_load_file(
                c_path.as_ptr(),
                &mut graph as *mut _,
                &mut err_msg as *mut _,
            )
        };
        if rc != kglite_c::KgliteStatusCode::Ok {
            let msg = unsafe { std::ffi::CStr::from_ptr(err_msg) }.to_string_lossy().to_string();
            unsafe { kglite_c::kglite_free_string(err_msg) };
            return Err(Error::from_reason(msg));
        }
        Ok(Self { inner: graph })
    }

    // ... session / cypher methods follow the same shape ...
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { kglite_c::kglite_graph_free(self.inner) };
        }
    }
}
```

A pure-JS consumer (no Rust at all) can link against
`libkglite_c.so` via `ffi-napi` or `koffi`, calling the C
functions directly with the same shapes.

#### Worked example — JNI (JVM)

```java
// Java / Kotlin / Scala — through a JNI shim crate. The shim
// is ~500 LOC of jni-rs glue similar to the cgo shape above.

public class Graph implements AutoCloseable {
    private long handle;  // opaque *mut KgliteGraph

    public static Graph loadFile(String path) {
        return new Graph(KgliteJni.loadFile(path));
    }

    public Session newSession() {
        return new Session(KgliteJni.sessionNew(handle));
        // handle is moved into session; do not close graph after this.
    }

    @Override
    public void close() {
        if (handle != 0) {
            KgliteJni.graphFree(handle);
            handle = 0;
        }
    }
}
```

The JNI native crate would call `kglite_c::kglite_*` functions
through `jni-rs` bindings, mapping the JVM `long` handles to
`*mut KgliteX` round-trip casts. Cypher result rows come back as
JSON; the Java side parses with Jackson / Gson / `JsonParser`.

#### What kglite-c hands you (v1 surface)

- **Lifecycle**: `load_file`, `save_graph`, `graph_free`.
- **Session**: `session_new`, `session_execute_read`,
  `session_execute_mut`, `session_free`. Plus `session_set_embedder`.
- **Result**: `cypher_result_columns_json`, `cypher_result_rows_json`,
  `cypher_result_row_count`, `cypher_result_free`.
- **Error introspection**: `status_code_name`,
  `status_code_neo4j_status`, `status_code_http_status`.
- **Datasets**: per-loader fetchers + extract pipelines for SEC
  EDGAR, Sodir, Wikidata (each feature-gated behind a `KGLITE_FEATURE_*`
  preprocessor define).
- **Embedder**: `embedder_fastembed_new` (feature-gated),
  `embedder_free`, `session_set_embedder`.
- **ABI version**: `kglite_abi_version()` for startup checks.

That's ~30 C functions total. Phase H added the surface in three
sub-phases (H.2 skeleton, H.3 Sodir + embedder, H.3a SEC +
Wikidata); future iterations can extend with per-filing fetchers
or user-supplied embedder callbacks as bindings ask.

#### Alternatives if kglite-c doesn't fit

- **Use the existing PyO3 wrapper indirectly** (your binding calls
  Python which calls kglite — slow and weird, but works today).
- **Use the network protocols.** If your binding is HTTP/RPC-shaped,
  the Bolt server or MCP server are already canonical wire formats —
  your "binding" becomes a Bolt/MCP client in your target language,
  zero compiled code required.

## Error mapping

Every kglite API call returns `Result<T, KgError>`. `KgError` is a
typed enum with 17 variants; each variant has a stable
`KgErrorCode` discriminant. Your binding maps these to its target
language's idiomatic error types. The file-I/O variants are worth
surfacing distinctly: `FileNotFound`, `FileFormat` (corrupt /
truncated / wrong-format `.kgl` — what `load_file` / `load_kgl_bytes`
return on a bad file), and `FileIo` (permission, mid-read) — so a
consumer can tell "rebuild from source" from "create new".

The table below is the recommended mapping. The "Recoverable?"
column is from the agent's POV — should the binding's caller
retry? rewrite? give up?

| `KgErrorCode` | When it fires | Recoverable? | Suggested language idiom |
|---|---|---|---|
| `CypherSyntax` | Tokenizer / parser rejected the query string | **No** — the query is malformed | Type/usage error (`TypeError`, `IllegalArgumentException`, `SyntaxError`) |
| `CypherTimeout` | Query exceeded its `timeout_ms` budget | **Maybe** — retry with longer budget or rewrite | Timeout error (`TimeoutError`, `DeadlineExceeded`) |
| `Cancelled` | Caller flipped `ExecuteOptions.cancel` mid-run (e.g. Ctrl-C, client disconnect) | **No** — the caller asked to stop | Interrupt / cancel idiom (`KeyboardInterrupt`, cancelled-context, abort status) |
| `CypherExecution` | Mutation conflict, predicate panic, etc. | **Sometimes** — context-dependent | Runtime error |
| `CypherTypeMismatch` | A param was the wrong type for an operator | **No** — fix the call site | Type error |
| `Schema` | Query references unknown label/property/index | **No** — schema mismatch | Validation error |
| `Validation` | Bulk-mutation row failed validation (FK, unique, type) | **No** for that row — others may succeed | Validation error per row |
| `Expr` | Blueprint expression failed to evaluate | **No** — fix the expression | Validation error |
| `NodeNotFound` | Lookup by ID/name returned nothing | **No** — caller's domain logic | KeyError / NotFoundError |
| `ConnectionNotFound` | Same, for edges | **No** | KeyError / NotFoundError |
| `PropertyNotFound` | Property doesn't exist on the node | **No** | KeyError / NotFoundError |
| `FileNotFound` | Path passed to `load_file` doesn't exist | **No** | FileNotFoundError / IOError |
| `FileFormat` | File exists but isn't a valid `.kgl` (wrong magic, version, checksum) | **No** | Format / parse error |
| `FileIo` | Filesystem error during read/write | **Maybe** — retry on transient (disk full, network FS) | IO error |
| `InvalidArgument` | Function argument was wrong (out of range, malformed) | **No** | ValueError / IllegalArgumentException |
| `MissingArgument` | Required argument was None / null | **No** | TypeError / NullPointerException |
| `Internal` | Should-never-happen invariant violation | **No** — report as bug | InternalError |

`KgError::Display` already produces a human-readable message; bindings
typically expose both the code (for programmatic dispatch) and the
message (for the user). The pyapi crate's `crates/kglite-py/src/error_py.rs` is the
canonical mapping — every Python exception type kglite raises is a
mechanical projection of `KgErrorCode`.

The bolt-server's `crates/kglite-bolt-server/src/error_map.rs` maps each variant into a
Bolt `ClientError` with a Neo4j-style status code prefix
(`Neo.ClientError.Schema.SchemaNotFound`, etc.). Use it as a
reference if your binding needs wire-protocol-shaped errors.

## Implementing the Embedder trait

KGLite's `text_score()` Cypher function needs to embed user queries
at lookup time. The engine doesn't ship a specific embedder — you
plug in your own via the `kglite::api::Embedder` trait.

The trait has two required methods plus three optional ones:

```rust
pub trait Embedder: Send + Sync {
    fn dimension(&self) -> usize;
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
    fn model_id(&self) -> Option<String> { None }       // optional (provenance)
    fn load(&self) -> Result<(), String> { Ok(()) }     // optional
    fn unload(&self) {}                                  // optional
}
```

The contract:

- **`dimension`** — the output vector size (e.g. 384 for
  all-MiniLM, 1024 for bge-m3). The engine validates this against
  user-supplied vectors at `set_embeddings()` time, so it has to be
  fixed for the embedder's lifetime.
- **`embed(texts)`** — embed a batch. Return one vector per input
  text, each of length `dimension()`. Errors go back to the user via
  `KgError::CypherExecution`.
- **`model_id`** — optional (defaults to `None`). The model's stable id
  (e.g. `"BAAI/bge-m3"`); when present it's stamped onto the embedding
  store as provenance and surfaced via `embedding_info()`. Implement it
  if your embedder can name its model.
- **`load` / `unload`** — optional lifecycle hooks. Called before /
  after each embedding pass. Use these for lazy model loading + idle
  cooldown if your embedder is expensive to keep resident.

### Two existing implementations to copy from

**`FastEmbedAdapter`** (`crates/kglite/src/graph/embedder/fastembed.rs`)
— wraps the `fastembed-rs` crate for local ONNX inference. ~200 lines,
gated on the `fastembed` Cargo feature. Concrete implementation of
lazy `load` + idle `unload` with a cooldown timer.

**`PyEmbedderAdapter`** (in `crates/kglite-py/src/graph/embedder/py_adapter.rs`) — wraps
a user-supplied Python class implementing the embedder protocol. The
adapter acquires the GIL for `embed()` calls and translates between
Python lists and Rust vectors. Showed up as the pattern for any
binding that wants to let users plug in their own embedder.

### Pattern: HTTP-API-backed embedder

If your binding wants to wrap a hosted embedding API (OpenAI,
Cohere, Voyage, etc.):

```rust
use kglite::api::Embedder;
use std::sync::Mutex;

pub struct OpenAiEmbedder {
    api_key: String,
    model: String,                     // "text-embedding-3-small" → 1536
    dimension: usize,
    client: reqwest::blocking::Client,
}

impl Embedder for OpenAiEmbedder {
    fn dimension(&self) -> usize { self.dimension }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        // POST to /v1/embeddings, parse the response, return.
        // The session module passes one batch at a time; you don't
        // need to chunk further unless your API limits batch size.
        // …
    }
    // No load/unload — HTTP client is cheap to keep around.
}
```

Your binding then exposes a constructor in its native language
that returns `Box<dyn Embedder>` (or wraps it in the binding's
own embedder type, with `Arc<dyn Embedder>` underneath).

## Loading data

Four ways to populate a `DirGraph`. Bindings expose whichever subset
fits their audience.

### 1. Read a `.kgl` file written by any other binding

```rust
use kglite::api::load_file;

let graph = load_file("snapshot.kgl")?;  // → Arc<DirGraph>
```

`.kgl` is the cross-binding portable format. A graph saved by Python
(`kg.save("snapshot.kgl")`) reads cleanly here. A graph saved by your
Go binding will read in Python. The format is versioned (`v3`, `v4`)
and the engine handles version negotiation transparently.

### 2. Build from a blueprint + CSVs

The blueprint is a JSON/YAML schema spec; the engine reads it,
streams the referenced CSVs, materializes nodes + connections.

```rust
use kglite::api::blueprint::{load_blueprint_file, build, Blueprint};
use kglite::api::DirGraph;
use std::path::Path;

let blueprint: Blueprint = load_blueprint_file(Path::new("schema.json"))?;
let mut graph = DirGraph::new();
let report = build(&mut graph, &blueprint, Path::new("./data/csv/"))?;
println!("built {} nodes, {} connections", report.nodes_in, report.connections_in);
```

`Blueprint` carries the full schema spec; `BuildReport` carries
counts + validation errors. The Python wheel's `from_blueprint`
wraps these two functions with path resolution + `lock_schema` + a
PyO3-flavored progress callback (see
`crates/kglite-py/src/graph/pyapi/blueprint.rs:23`); your binding
will likely want the same shape in its own language idiom.

### 3. Build from a source tree (code intelligence)

```rust
use codingest::build_code_tree; // the standalone builder crate

let graph = build_code_tree(Path::new("./my_project/"), /* options */)?;
```

This walks a source tree with tree-sitter parsers and produces a
code-intelligence graph (Function / Class / Module / etc. nodes,
CALLS / DEFINES / IMPORTS edges). The kglite MCP server uses this
to answer "what functions call X" queries about a codebase.

### 4. Use a dataset loader (feature-gated)

```toml
[dependencies]
kglite = { version = "0.11", features = ["sec", "sodir", "wikidata"] }
```

Each loader is opt-in via its Cargo feature; the building blocks
land in `kglite::api::datasets::{sec, sodir, wikidata}`:

```rust
use kglite::api::datasets::sec::{
    SecClient, Workdir, YearRange,
    fetch_quarterly_master_idx, fetch_submissions_bulk,
    run_all, SliceSpec, predict_graph_size_gb, pick_storage_mode,
};

let wd = Workdir::new("/tmp/sec");
let client = SecClient::new("YourBinding/1.0 contact@example.com")?;

// All fetch_* entries are async — driven from a tokio runtime
// you manage in your binding.
fetch_quarterly_master_idx(
    &client, &wd, YearRange::new(2020, 2024), /*current_year*/ 2024, /*current_quarter*/ 4,
).await?;

// Plan storage mode from a size estimate:
let gb = predict_graph_size_gb(5, 0, None, true, true, true);
let mode = pick_storage_mode(gb);  // "memory" | "mapped" | "disk"
```

The full lifecycle orchestration (cache management, mode selection,
retry policies, ticker resolution) lives in each binding's wrapper.
See the next section for the pattern.

## Wrapping a dataset for your binding

### The boundary rule

Before any code: there is one rule about where things go.

> **A wrapper only contains code that is specific to its environment
> and cannot be used by any other sibling wrapper.** Anything that
> two or more wrappers would write identically belongs in
> `kglite::api::*`.

That rule is the lens for every decision below. PyO3 marshalling
goes in the Python wrapper because no Go binding would use it. A
JSON parser for SEC's `company_tickers.json` goes in core because
every binding parses the same JSON. A `tqdm` progress display goes
in the Python wrapper because Go uses channels and JS uses event
emitters for the same purpose. The cache-freshness check ("is the
local dump older than the remote one") goes in core because every
binding asks the same question the same way.

If you find yourself writing logic in a wrapper that another
binding would copy verbatim, stop and file it as a core lift.

### The shared lifecycle shape

The three reference wrappers (`kglite/datasets/sec/wrapper.py`,
`kglite/datasets/sodir/wrapper.py`, `kglite/datasets/wikidata.py`)
all follow the same six-step lifecycle, even though their per-step
implementations differ:

1. **Workdir layout** — decide where raw / processed / built files
   live on disk. Each dataset has a `Workdir` type in core
   (`kglite::api::datasets::sec::Workdir`, etc.) — your binding
   wraps it.
2. **Cache short-circuit** — if a fresh build exists for the
   requested storage mode, load and return early without
   re-fetching. The freshness rules live in core (Wikidata: mtime
   + remote HEAD probe; Sodir: cooldown_days; SEC: presence of
   `graph/{mode}/`); your binding decides only when to bypass via
   a `force_rebuild` flag.
3. **Fetch** — call the engine's `fetch_*` functions for whatever
   raw payloads are missing. All `fetch_*` entries are async; if
   your binding doesn't manage a tokio runtime, use the
   `*_blocking` variants in core (they spin up a single-thread
   runtime per call).
4. **Extract / preprocess** — for SEC, call `run_all`; for Sodir,
   `fetch_all` does fetch + preprocess in one shot; for Wikidata,
   the dump is the processed form (no separate extract).
5. **Build** — for SEC + Sodir, call `kglite::api::blueprint::build`
   on the loaded blueprint; for Wikidata, call
   `KnowledgeGraph::load_ntriples` on the dump.
6. **Cache + return** — save the built graph (`save_graph`),
   stamp build-time metadata into the graph dir, return the handle.

### Reference implementations

- **SEC** (largest, most complex): `kglite/datasets/sec/wrapper.py`
  (~600 LOC after the 2026-05-25 lifts). Three-tier cache
  (`raw/` → `processed/` → `graph/{mode}/`). Three storage modes
  (memory / mapped / disk). Per-form-type dispatch reads
  `processed/filing_index.csv` and groups filings into buckets via
  `kglite::api::datasets::sec::resolve_fetch_buckets` (which uses
  the canonical `_FORM_BUCKETS` table in core, NOT a wrapper-side
  copy). Ticker resolution uses
  `kglite::api::datasets::sec::parse_tickers_json`.
- **Sodir** (medium): `kglite/datasets/sodir/wrapper.py` (~280 LOC).
  Two-tier cache (`csv/` + `graph/`). Two modes (memory / disk).
  Blueprint with optional user complement file persisted in workdir.
- **Wikidata** (smallest): `kglite/datasets/wikidata.py` (~300 LOC).
  Single-file dump cache. Process-local cache for Jupyter rerun-
  cell ergonomics — that part stays in the Python wrapper because
  it's Python-specific (Go would use `sync.Map`, JS a module-level
  `Map`, etc.).

### A Go binding wrapping SEC would have…

The same six lifecycle steps, ~150 LOC of Go calling into the
same Rust building blocks. The form-type bucketing is the
canonical table in `kglite::api::datasets::sec::all_buckets`
(materialized once at binding-load time); the ticker parser is
`kglite::api::datasets::sec::parse_tickers_json`. Your Go
binding's contribution is the Go-idiomatic ergonomics around them:

- `kglite::api::Workdir::new(p)` wrapped in your binding's
  `kglite.OpenSEC(p, opts)` constructor
- A progress event channel (`chan ProgressEvent`) instead of tqdm
- Go-native error wrapping (`fmt.Errorf("sec fetch: %w", err)`)
  around the `KgError` you receive
- `sync.Map` for any process-local caching equivalent to Python's
  `_PROCESS_CACHE` (Wikidata)
- cgo / napi / JNI marshalling of `Workdir`, `SecClient`, etc.
  across the FFI boundary

What you **don't** write: the form-type table, the ticker JSON
parser, the cache-freshness rules, the cooldown semantics, the
HTTP rate-limit token bucket (built into `SecClient` /
`ArcGISClient` / `WikidataClient`), the resumable download logic,
the blueprint loader, the build pipeline, the Cypher pipeline.
All of those live in `kglite::api::*` and are shared across every
binding.

### The audit history

The `kglite::api` surface was sized against the wheel's PyO3 methods
in May 2026 (the binding-framework sprint that produced 0.10.0
through 0.10.3). The current shape reflects which items every
binding would write identically vs which are tailored to one
environment.

## Binding-side patterns cookbook

Patterns that surface in every binding, with notes on how the
existing bindings handle them.

### Process-local cache

Loading a large graph (Wikidata at 1.4B triples) takes minutes.
In Jupyter or any REPL-like environment, the user wants
`kg = wikidata.open(...)` to return the *same instance* on
re-execution instead of reloading.

The Python wheel does this via a module-level dict in
`kglite/datasets/wikidata.py:_PROCESS_CACHE` keyed on
`(workdir, entity_limit)`. Each binding has its own idiom:

| Language | Idiom |
|---|---|
| Python | Module-level `dict` (used by wikidata.py) |
| Go | `sync.Map` in a package-level var |
| JS / Node | Module-level `Map` |
| JVM | `ConcurrentHashMap` in a singleton |
| Rust binary | `once_cell::sync::Lazy<DashMap<…>>` |

Cache invalidation key should include anything that affects what
gets loaded — workdir path, entity limits, source-file mtime if
relevant.

### Lazy result materialization

Cypher queries can return millions of rows. The session module's
`execute_read` returns a fully-materialized `CypherResult` by
default — fine for small results, ruinous for large ones.

For large results, the engine has a lazy path: set
`ExecuteOptions.lazy_eligible = true` and the returned
`CypherResult` will have a `lazy: Some(LazyResultDescriptor)`.
Your binding's result-iteration code then calls back into the
engine row-by-row instead of materializing upfront.

The Python wheel's `ResultView`
(`crates/kglite-py/src/graph/pyapi/result_view.rs`) is the canonical
lazy materializer. Bolt-server's `ResultStream`
does the same thing for the wire protocol. If your binding doesn't
have a lazy materializer, pass `lazy_eligible = false` — otherwise
you'll silently see empty `result.rows` for any non-`ORDER BY`
query.

### Value type conversion

`kglite::api::Value` is the engine's universal value type (scalar
variants + `List` + `Map` + graph-specific `Node` / `Relationship` /
`Path`). Your binding needs to translate at the FFI boundary.

The pattern in every existing binding:

1. Define a binding-native value type (`PreProcessedValue` in
   pyapi, `BoltValue` in bolt-server, JSON in mcp-server).
2. Write a `From<Value> for YourValue` impl (and reverse if your
   binding accepts user-supplied params).
3. Pattern-match every `Value` variant. Forgetting one is a
   runtime error.

The pyapi crate's `crates/kglite-py/src/datatypes/py_out.rs` is the
densest example — it handles NumPy and Pandas type coercion on top
of basic conversion.

### Progress callbacks

Long-running operations (dataset fetch, blueprint build) take
seconds-to-minutes. Users want progress.

The engine's `BuildReport` carries final counts but doesn't emit
per-row progress. Bindings emit their own progress by:

- Wrapping the fetch / build call in a thread that periodically
  reads a shared counter
- Or calling small batches in a loop and reporting between them

The Python wheel uses `tqdm` with explicit batch-size control on
the fetch side and a `verbose=True` flag for build-time prints. A
Go binding might use a `chan struct{}` for progress events with
a separate goroutine consuming them.

This is genuinely binding-specific — no shared abstraction is
likely to help. Pick the idiom your audience expects.

### Cancellation / interruptible queries

`ExecuteOptions.cancel: Option<&AtomicBool>` is the engine-agnostic
cancellation primitive. The engine polls it at the same checkpoints it
polls the query deadline (pattern-matcher scans + expansions); once the
flag is set, the run aborts and the call returns `KgError::Cancelled`
(`KgErrorCode::Cancelled` → HTTP 499, Neo
`Neo.ClientError.Transaction.Terminated`). Leave it `None`
(`ExecuteOptions::eager` does) and queries are deadline-bounded only.

How you *flip* the flag is binding-specific — it's part of the
async/threading/signal model each binding owns:

- The **Python wheel** installs a scoped `SIGINT` (Ctrl-C) handler for
  the duration of a query that flips a process-global `AtomicBool`, then
  restores the previous handler; `KgError::Cancelled` is mapped to
  Python's builtin `KeyboardInterrupt`. (Read paths only; mutations stay
  deadline-bounded.)
- A **server** binding typically wires the flag to a request-cancellation
  token (client disconnect, gRPC `tokio::select!` on the cancel future, a
  deadline-exceeded watcher) and maps `Cancelled` onto its protocol's
  cancel/abort status.

Because `cancel` is a `&AtomicBool`, the flag must outlive the
`execute_*` call — a `'static` global (the wheel's choice) or a flag
owned by the request scope both work. The previous SIGINT disposition,
or whatever you replaced, is yours to restore.

## What you don't need to write

The work that's already done. Your binding inherits all of this
by depending on `kglite`:

| Component | Where it lives | LOC |
|---|---|---|
| Cypher parser + planner + executor | `kglite::api::cypher` | ~15,000 |
| Snapshot/working CoW + OCC | `kglite::api::session` | ~1,500 |
| Schema validation pipeline | `kglite::api::cypher::validate_schema` | ~800 |
| Blueprint loader + builder | `kglite::api::blueprint` | ~5,000 |
| `.kgl` format reader + writer (v3, v4) | `kglite::api::{load_file, save_graph}` | ~3,000 |
| Code-graph builder (tree-sitter) | the `codingest` crate | ~4,000 |
| Dataset fetchers (SEC, Sodir, Wikidata) | `kglite::datasets::*` (feature-gated) | ~8,000 |
| Embedder trait + FastEmbed adapter | `kglite::api::Embedder`, `FastEmbedAdapter` | ~600 |
| Cypher result types + path/node/rel | `kglite::api::{Value, NodeValue, RelValue, PathValue}` | ~2,000 |

Your binding contributes:

| Component | Notes |
|---|---|
| Language-native graph handle wrapping `Arc<DirGraph>` | The thin facade your users hold |
| Value type marshalling (`Value` ↔ your native type) | Mechanical but error-prone — write tests |
| Error mapping (`KgError` → your error type) | Use the table above |
| Lifecycle (open, close, save, snapshot) | Thin wrapper around the engine functions |
| Optional: lazy result iterator | If your binding has any |
| Optional: embedder wrapper for user-provided embedders | If your binding lets users plug in custom embedders |
| Optional: dataset wrapper layer | If your binding cares about specific datasets |

The hard parts — correctness of Cypher, snapshot isolation,
OCC conflict detection, format portability — you inherit.

## Cross-binding portability checklist

A `.kgl` file written by your binding should load cleanly in any
other binding. To stay portable:

- **Use `kglite::api::save_graph`** (not your own format). It's
  versioned, checksummed, and bumps along with the engine.
- **Don't bundle binding-ergonomic state** in the graph itself.
  Selection caches, default timeouts, progress callbacks — these
  are per-binding overlays. Save the `DirGraph`, let each binding
  add its own state on load.
- **If you write custom values into Map / List**, stick to the
  scalar variants `Value` already supports. Don't smuggle a JSON
  string in expecting another binding to parse it; use a proper
  variant.
- **Validate after save.** Round-trip your test fixtures
  (your-binding → `.kgl` → Python → `.kgl` → your-binding) at
  least once per release. The `tests/test_phase4_parity.py`
  suite is the model.

## Roadmap

What the binding-author experience needs that isn't done yet,
roughly in priority order. File issues for the ones that block
your specific binding:

- **Phase H — a `kglite-c` crate** exposing the engine through a
  stable C ABI. Unlocks bindings in any FFI-capable language
  without each one rolling its own `extern "C"` layer.
- **Selection / fluent-API in core.** Today's wheel exposes a
  fluent `select() / where() / sort()` builder; unclear how much
  of the builder lives in core vs. the PyO3 layer. Punted unless
  a binding asks for it.
- **Result streaming via the C ABI.** If/when Phase H lands,
  there's a design question about how to stream the lazy
  materializer across a C boundary.
- **Graph algorithms exposed in `api::*`.** Shortest path,
  centrality, community detection — what's implemented internally
  vs. aspirational needs verification (audit punchlist item #8).

If you build a binding and discover a real gap, file an issue —
that's the right signal for promoting an item from the maintainer's
deferred-items list into actual work.

## See also

- [`embedding.md`](embedding.md) — full embedder guide, polars-
  style split rationale, quick start.
- [`session.md`](session.md) — canonical Cypher pipeline + CoW
  transaction reference.
- [`api-reference.md`](api-reference.md) — manifest of stable
  surface items with semver guarantees.
- [Cypher reference](../reference/cypher-reference.md) — the
  Cypher subset kglite supports.
- `CLAUDE.md` (repo root) — engineering conventions for changes
  to the engine itself.
