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
kglite = "0.10"
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
[embedding.md](embedding.md#cgo-sketch-go).

### Option 3 — A C ABI middle crate (the Phase H aspiration)

Both options above tie you to Rust. A future `kglite-c` crate would
expose `kglite::api::*` through stable C function signatures. Any
language with FFI (which is "every language") could then bind to
kglite without writing Rust.

```c
// kglite.h (illustrative — does not exist yet)
typedef struct KgliteDirGraph KgliteDirGraph;
typedef struct KgliteSession  KgliteSession;
typedef struct KgliteResult   KgliteResult;

KgliteDirGraph* kglite_load_file(const char* path);
void kglite_dir_graph_drop(KgliteDirGraph* g);

KgliteResult* kglite_session_execute_read(
    KgliteSession* s, const char* query, const KgliteExecuteOptions* opts);
```

This is Phase H on the roadmap. Until it lands, you have three real
choices:

- **Use the existing PyO3 wrapper indirectly** (your binding calls
  Python which calls kglite — slow and weird, but works today).
- **Roll your own C ABI** in your binding crate. ~200 lines of
  `extern "C"` wrappers. Brittle (no shared maintenance) but
  unblocks you.
- **Use the network protocols.** If your binding is HTTP/RPC-shaped,
  the Bolt server or MCP server are already canonical wire formats —
  your "binding" becomes a Bolt/MCP client in your target language,
  zero Rust code required.

The honest recommendation: if Phase H matters for your work, file
an issue. Two non-Rust bindings asking for it is what pulls it
forward on the roadmap.

## Error mapping

Every kglite API call returns `Result<T, KgError>`. `KgError` is a
typed enum with 16 variants; each variant has a stable
`KgErrorCode` discriminant. Your binding maps these to its target
language's idiomatic error types.

The table below is the recommended mapping. The "Recoverable?"
column is from the agent's POV — should the binding's caller
retry? rewrite? give up?

| `KgErrorCode` | When it fires | Recoverable? | Suggested language idiom |
|---|---|---|---|
| `CypherSyntax` | Tokenizer / parser rejected the query string | **No** — the query is malformed | Type/usage error (`TypeError`, `IllegalArgumentException`, `SyntaxError`) |
| `CypherTimeout` | Query exceeded its `timeout_ms` budget | **Maybe** — retry with longer budget or rewrite | Timeout error (`TimeoutError`, `DeadlineExceeded`) |
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

The trait has three methods:

```rust
pub trait Embedder: Send + Sync {
    fn dimension(&self) -> usize;
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
    fn load(&self) -> Result<(), String> { Ok(()) }    // optional
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
use kglite::api::build_code_tree;
use kglite::code_tree::CodeTreeOptions;

let graph = build_code_tree(Path::new("./my_project/"), &CodeTreeOptions::default())?;
```

This walks a source tree with tree-sitter parsers and produces a
code-intelligence graph (Function / Class / Module / etc. nodes,
CALLS / DEFINES / IMPORTS edges). The kglite MCP server uses this
to answer "what functions call X" queries about a codebase.

### 4. Use a dataset loader (feature-gated)

```toml
[dependencies]
kglite = { version = "0.10", features = ["sec", "sodir", "wikidata"] }
```

Each loader is opt-in. The building blocks are exposed (HTTP
clients, batch fetchers, extract orchestration); the full lifecycle
orchestration (cache management, mode selection, retry policies)
lives in each binding's wrapper. See the next section for the
pattern.

## Wrapping a dataset for your binding

The three dataset loaders (SEC EDGAR, Sodir, Wikidata) each ship as
a set of building blocks in `kglite::datasets::*`. The lifecycle
orchestration — "fetch what's missing, build if cache is stale,
return a ready-to-query graph" — is **not** in the engine. It lives
in each binding's wrapper layer.

**Reference implementation:** the Python wheel's wrappers at
`kglite/datasets/sec/wrapper.py` (779 lines), `wrapper.py`
(279 lines), `kglite/datasets/wikidata.py` (298 lines). Each
follows the same structure:

1. **Workdir layout** — decide where raw / processed / built files
   live on disk.
2. **Cache short-circuit** — if a fresh build exists, load and
   return early.
3. **Fetch** — call the engine's `fetch_*` functions for whatever's
   missing.
4. **Extract / preprocess** — for SEC, call `run_all`; for Sodir,
   call `fetch_all`; for Wikidata, the dump is the processed form.
5. **Build** — call `from_blueprint` for SEC/Sodir, or
   `load_ntriples` for Wikidata.
6. **Cache + return** — save the built graph, return the handle.

A Go binding wrapping SEC would have the same six steps, ~150
lines of Go calling into the same Rust building blocks. The
business logic (form-type bucketing, cooldown decisions, retry
budgets) is identical across languages; only the surrounding
ergonomics (Jupyter process cache, tqdm progress bars,
language-native filesystem APIs) differ.

The audit at `docs/internal/api-audit-2026-05-25.md` documents
which building blocks are in `kglite::api::*` today and which need
re-exporting. Phase 3 of the prep work is closing those gaps.

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
| Code-tree builder (tree-sitter) | `kglite::api::build_code_tree` | ~4,000 |
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
- **`kglite::api::datasets`** — re-export the per-dataset
  fetch/extract/build primitives through the stable surface
  (today they're at `kglite::datasets::*`, semantically stable
  but not formally in `api::*`). Tracked as Phase 3 of the
  current prep work.
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

If you build a binding and discover a real gap, the audit doc
(`docs/internal/api-audit-2026-05-25.md`) is the right place to
read the prior survey; the punchlist there is what we're working
through.

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
