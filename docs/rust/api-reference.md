# API reference

The stable surface — `kglite::api::*` — that gets semver
guarantees. Items outside this surface (`kglite::graph::*`,
`kglite::datatypes::*`, etc.) are implementation details and may
move between minor releases.

For per-symbol API docs (function signatures, struct fields,
trait method docs), use **[docs.rs/kglite](https://docs.rs/kglite)**.
This page is the curated inventory.

## Engine types

| Item | Path | Purpose |
|---|---|---|
| `DirGraph` | `kglite::api::DirGraph` | The in-memory graph. Owned by your binding's "graph handle". |
| `Value` | `kglite::api::Value` | Every value Cypher can return: scalars, `List`, `Map`, `Node`, `Relationship`, `Path`, …. |
| `NodeValue` / `PathValue` / `RelValue` | `kglite::api::*` | Per-variant carriers; pattern-match into them without deriving accessors. |
| `KgError` / `KgErrorCode` | `kglite::api::KgError`, `KgErrorCode` | Typed error enum (16 variants). Map to your binding's error idiom. File I/O: `FileNotFound` / `FileFormat` (corrupt/wrong-format `.kgl`) / `FileIo` (→ Python `FileError`/`FileFormatError`/`FileIoError`). |
| `Embedder` (trait) | `kglite::api::Embedder` | Pluggable text-embedding backend for `text_score()` Cypher. |
| `FastEmbedAdapter` (feature `fastembed`) | `kglite::api::FastEmbedAdapter` | Rust-native ONNX embedder. |
| `SourceLocation` / `SourceLookup` | `kglite::api::*` | Code-entity location lookup result types. |
| `ExploreOptions` / `explore_markdown` | `kglite::api::*` | Codebase exploration as a markdown report. |

## I/O

| Item | Path | Purpose |
|---|---|---|
| `load_file(path)` | `kglite::api::load_file` | Read a `.kgl` file (or disk dir) → `io::Result<Arc<DirGraph>>`. |
| `load_kgl_bytes(&[u8])` | `kglite::api::load_kgl_bytes` | Load an in-memory graph from a `.kgl` byte buffer (counterpart of `write_kgl_to`). |
| `save_graph(&mut arc, path)` | `kglite::api::save_graph` | Write an `Arc<DirGraph>` → `Result<(), String>`. |
| `write_kgl` / `write_kgl_with(..., fsync)` | `kglite::api::write_kgl*` | Atomic (temp+rename) + durable (`fsync`) `.kgl` write. `write_kgl_with` toggles the flush. |
| `write_kgl_to(&graph, &mut writer)` | `kglite::api::write_kgl_to` | Serialize the `.kgl` byte stream into any `Write` (backs `to_bytes`). |

`DirGraph::copy_embeddings_from(&src)` carries embedding stores across a rebuild
by node id (the core behind the Python `copy_embeddings_from`). The other new
0.11.0 methods — `embedding_info` / `embedding_dim`, `replace_connections`,
`embed_texts(mode=…)`, `freeze` — are binding-surface (Python `KnowledgeGraph`)
methods, documented in the Python track, not raw `kglite::api` functions.

## Schema introspection

| Item | Path | Purpose |
|---|---|---|
| `compute_description(...)` | `kglite::api::compute_description` | XML schema description for agent system prompts. |
| `compute_schema(&dir)` | `kglite::api::compute_schema` | Structured `SchemaOverview` (node types, edge types, indexes). |
| `SchemaOverview`, `ConnectionDetail`, `CypherDetail`, `FluentDetail` | `kglite::api::*` | Structured introspection types. |

## Cypher pipeline (`kglite::api::cypher`)

For building custom pipelines. For the canonical pipeline, use
the `session` module instead.

| Item | Purpose |
|---|---|
| `parse_cypher(query)` | Parse a query string → `CypherQuery`. Uses the global cache. |
| `validate_schema(&parsed, &graph)` | Schema-check a parsed query against a graph. |
| `rewrite_text_score(...)` | Lower `text_score()` references into vector lookups. |
| `mark_lazy_eligibility(&mut parsed)` | Mark queries eligible for streaming materialization. |
| `optimize(...)` / `planner::*` | Run the optimizer passes; introspect the pipeline. |
| `CypherExecutor` | Execute a planned query against a graph. |
| `execute_mutable(...)` | Mutation execution path. |
| `is_mutation_query(&parsed)` | Heuristic: does this query mutate? |
| `generate_explain_result(...)` | Build an EXPLAIN-style plan as a CypherResult. |
| `CypherQuery`, `CypherResult`, `OutputFormat` | Data types. |

## Session (canonical query + transaction surface)

The Phase E "single source of truth" — all bindings flow through
here.

| Item | Path | Purpose |
|---|---|---|
| `Session` | `kglite::api::session::Session` | Shared graph state with commit-swap semantics. |
| `Transaction` | `kglite::api::session::Transaction` | Snapshot/working CoW transaction state. |
| `CommitOutcome` | `kglite::api::session::CommitOutcome` | `NoWritesNoOp` / `Committed{new_version}` / `ConflictDetected{current_version, base_version}`. |
| `ExecuteOptions` | `kglite::api::session::ExecuteOptions` | Params + deadline + max_rows + lazy hint + embedder. |
| `ExecuteOutcome` | `kglite::api::session::ExecuteOutcome` | `result: CypherResult` + `is_mutation: bool` + `output_format: OutputFormat`. |
| `execute_read(&graph, query, &opts)` | `kglite::api::session::execute_read` | Run a read query. |
| `execute_mut(&mut graph, query, &opts)` | `kglite::api::session::execute_mut` | Run a mutation. |

## Dataset loaders (feature-gated)

```toml
kglite = { features = ["sec", "sodir", "wikidata"] }
```

| Feature | Module | What it loads |
|---|---|---|
| `sec` | `kglite::api::datasets::sec` | SEC EDGAR filings (quarterly index, bulk submissions, Form 4 / 13F, Exhibit 21, 8-K). |
| `sodir` | `kglite::api::datasets::sodir` | Norwegian Continental Shelf petroleum data (Sodir FactMaps REST). |
| `wikidata` | `kglite::api::datasets::wikidata` | Wikimedia `latest-truthy.nt.bz2` dump fetcher (resumable). |

Each submodule re-exports the same surface the Python wheel uses
via `_sec_internal` / `_sodir_internal` / `_wikidata_internal`:
workdir + storage-mode types, error + Result aliases, the HTTP
client, the async fetch entry points, and (for SEC) the extract
pipeline + size-prediction helpers. All `fetch_*` entries are
`async` — bindings need a tokio runtime to drive them.

**Lifecycle orchestration (cache short-circuit, mode selection,
retry budgets) is NOT in core.** Each binding wraps the building
blocks in its own language idiom; the Python wheel's wrappers
(`kglite/datasets/*/wrapper.py`) are the reference implementation.
See [`implementing-a-binding.md`](implementing-a-binding.md) →
"Wrapping a dataset for your binding".

Polars-io style: opt in only to what you use.

## Semver

`kglite::api::*` items above get semver guarantees within a minor
release. Anything outside that surface — `kglite::graph::*`,
`kglite::datatypes::*`, raw module paths — is internal and may
move freely between minor releases.

| Change kind | Bumps |
|---|---|
| New item in `api::*` | Minor (`0.11.0` → `0.12.0`) |
| Breaking change to an item already in `api::*` | Major (`0.x.y` → `1.0.0` once we hit 1.0; for now any 0.x bump may break, but we try to keep additions additive within a `0.x` line) |
| Internal rearrangement (non-api items) | Patch (`0.11.0` → `0.11.1`) |

The `.kgl` file format has its own version (`v3`, `v4`, …)
tracked separately. Format bumps land with their decoder and
require a graph rebuild from source.

## Where to find each item

```
kglite::                    (the crate root)
├── api::                   (this stable surface)
│   ├── cypher::            (parse / plan / execute primitives)
│   └── session::           (canonical Cypher pipeline + transactions)
├── datasets::              (sec / sodir / wikidata, feature-gated)
├── datatypes::             (internal — use api::Value)
├── error                   (internal — use api::KgError)
├── graph::                 (internal — engine submodules)
```
