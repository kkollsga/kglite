# API reference

The curated, documented surface — `kglite::api::*`. Items outside
this surface (`kglite::graph::*`, `kglite::datatypes::*`, etc.) are
implementation details and may move in any release. See
[Stability policy](#stability-policy) for what pre-1.0 guarantees.

For per-symbol API docs (function signatures, struct fields,
trait method docs), use **[docs.rs/kglite](https://docs.rs/kglite)**.
This page is the curated inventory. If you're building a library
that *produces* kglite graphs, see [Building on kglite](building-on-kglite.md).

## Stability policy

`kglite::api::*` and the `kglite-mcp-server` library surface are
**exact-baseline-locked in CI** (cargo-public-api, pinned nightly). Accidental
drift cannot merge, because the generated public-API listing is diffed against a
committed baseline on every PR. The `include/kglite.h` C header is drift-checked
the same way (cbindgen vs the committed header).

Pre-1.0, the policy is:

- **Any release, including a PATCH, may ship a documented breaking change.**
  KGLite's crates ship in lockstep and deliberately ship breaking engine
  changes in patch bumps. The version number is not a compatibility signal.
  - Every intentional break ships with a `CHANGELOG.md` entry naming the removed items and their replacements.
  - The changelog, not the bump size, is the migration contract.
  - Embedders should pin an exact version (`kglite = "=X.Y.Z"`) and upgrade against the changelog.

  *(This paragraph previously promised "patch releases never break the API".
  Release 0.15.9 removed public items in a patch, per the actual policy, and this page
  was the outlier.)*
- New options land under **0.14's options-struct convention** (`*Options`
  structs, `#[non_exhaustive]` + `Default`), so adding an option is a
  non-breaking change rather than a signature break.

**1.0 criterion:** the 0.14 surface, after the options-struct pass on
`api::algorithms`, soaks across releases without needing a breaking correction.
When the curated facade proves stable in the field, we cut 1.0. The pre-1.0
"any release may break" latitude then ends.

## Engine types

| Item | Path | Purpose |
|---|---|---|
| `DirGraph` / `KnowledgeGraph` | `kglite::api::*` | Core graph handle/storage wrapper spanning memory, mapped, and disk modes. |
| `GraphRead` / `GraphWrite` | `kglite::api::*` | Storage-independent read/mutation traits; `GraphRead` is GAT-based and not object-safe. |
| Storage/config types | `kglite::api::storage::*` | Mode selection, construction, and lifecycle configuration. |
| `Value` | `kglite::api::Value` | Every value Cypher can return: scalars, `List`, `Map`, `Node`, `Relationship`, `Path`, …. |
| `NodeValue` / `PathValue` / `RelValue` | `kglite::api::*` | Per-variant carriers; pattern-match into them without deriving accessors. |
| `KgError` / `KgErrorCode` | `kglite::api::KgError`, `KgErrorCode` | Typed errors with 17 stable codes including cancellation. Map code/message/status through the binding. |
| `Embedder` (trait) | `kglite::api::Embedder` | Pluggable text-embedding backend for `text_score()` Cypher. |
| `FastEmbedAdapter` (feature `fastembed`) | `kglite::api::FastEmbedAdapter` | Rust-native ONNX embedder. |
| `SourceLocation` / `SourceLookup` | `kglite::api::*` | Code-entity location lookup result types. |
| `ExploreOptions` / `explore_markdown` | `kglite::api::*` | Codebase exploration as a markdown report. |

## I/O

| Item | Path | Purpose |
|---|---|---|
| `load_file(path)` | `kglite::api::io::load_file` | Read a `.kgl` file (or disk dir) → `io::Result<Arc<DirGraph>>`. |
| `load_kgl_bytes(&[u8])` | `kglite::api::io::load_kgl_bytes` | Load an in-memory graph from a `.kgl` byte buffer (counterpart of `write_kgl_to`). |
| `save_graph(&mut arc, path)` | `kglite::api::io::save_graph` | Write an `Arc<DirGraph>` → `Result<(), String>`. |
| `write_kgl` / `write_kgl_with(..., fsync)` | `kglite::api::io::write_kgl*` | Atomic (temp+rename) + durable (`fsync`) `.kgl` write. `write_kgl_with` toggles the flush. |
| `write_kgl_to(&graph, &mut writer)` | `kglite::api::io::write_kgl_to` | Serialize the `.kgl` byte stream into any `Write` (backs `to_bytes`). |
| `export_embeddings_to_file(&graph, path, filter, &keys)` | `kglite::api::io::export_embeddings_to_file` | Write node and relationship embedding stores to a standalone `.kgle` file → `ExportStats`. Node-only exports are `.kgle` version 3; an export carrying relationship stores is version 4. |
| `import_embeddings_from_file(&mut graph, path, &keys)` | `kglite::api::io::import_embeddings_from_file` | Install a `.kgle` file's stores by node id and relationship address → `ImportStats` (its `relationships` field is an `RelationshipCarryStats`). |
| `RelationshipKeys`, `RelationshipCarryStats`, `EmbeddingCopyReport` | `kglite::api::io::*` | The relationship carry: `RelationshipKeys` maps a relationship type to the key property that tells a parallel group's members apart. |

`DirGraph::copy_embeddings_from(&src)` carries node embedding stores across a
rebuild by node id. `DirGraph::copy_embeddings_with_relationships_from(&src,
&keys)` also carries relationship stores and returns an `EmbeddingCopyReport`;
it is the core behind the Python `copy_embeddings_from`. A relationship vector
is addressed by relationship type plus the `(type, id)` of both endpoints. When
several relationships of one type connect the same two nodes, the carry
requires a key property named in `RelationshipKeys` whose value is unique
within the group. It refuses an ambiguous group by name, and the export,
import or copy then writes nothing. `embedding_dim`, `replace_connections`,
`embed_texts(mode=…)` and `freeze` are binding-surface (Python
`KnowledgeGraph`) methods, documented in the Python track, not raw
`kglite::api` functions.

## Embeddings (`kglite::api::embeddings`)

| Item | Purpose |
|---|---|
| `set_embeddings` / `add_embeddings` / `embed_property` | Write node vectors: replace a store, upsert into one, or compute them through a bound `Embedder`. |
| `build_vector_index` / `refresh_vector_index` / `drop_vector_index` / `has_vector_index` / `list_vector_indexes` | Node HNSW index lifecycle. A built index is saved in `.kgl` (node and relationship alike); disk generations keep neither. `refresh_vector_index` returns `Result<usize, String>`: the vectors it folded in (`0` when current or read-only), or an error when the store or its index does not exist — it never builds one. |
| `list_embeddings(&graph)` → `Vec<EmbeddingStoreInfo>` | Node stores only. The C ABI publishes its `node_type` field verbatim, so relationship stores are not folded in. |
| `list_relationship_embeddings(&graph)` → `Vec<RelationshipEmbeddingStoreInfo>` | Relationship stores, sorted by type and store. |
| `embedding_info(&graph, EmbeddingEntity, type, column)` | Provenance for one store (dimension, count, model, effective metric, hashed). `EmbeddingEntity::{Node, Relationship}` is explicit because a node type and a relationship type may share a name. |
| `relationship_embeddings(&graph, relationship_type, text_column, &RelationshipKeys)` → `Result<Vec<RelationshipEmbedding>, String>` | Every vector in a relationship store, addressed by endpoint `(type, id)` and ordered by source, target, key, slot — the edge-list plus edge-feature shape. Parallel relationships are told apart by the key property named for their type in `RelationshipKeys`; a named key missing on a member, or repeated within a group, is refused by name. |
| `set_relationship_embeddings` / `add_relationship_embeddings` `(&mut graph, relationship_type, text_column, rows, &RelationshipKeys, metric)` → `Result<RelationshipIngestReport, String>` | Write relationship vectors, each `RelationshipVector` addressed as `relationship_embeddings` reads it back: endpoint `(type, id)` pairs (the types may be `None` when the relationship type has one source and one target node type) plus a key for a parallel-group member. `set_` **replaces** the store (old vectors, metric, provenance and HNSW index discarded), as `set_embeddings` does for nodes; `add_` **upserts**, as `add_embeddings` and `db.relationship_embeddings.set` do (same store path, dimension, metric and provenance rules as the procedure). Every row is resolved first; a row naming no relationship, an ambiguous parallel group, a key a member lacks or repeats, or two rows naming one relationship is refused by its position. `From<RelationshipEmbedding>` makes read-modify-write a round trip. |
| `embed_relationship_texts(&mut graph, relationship_type, text_column, EmbedMode, &dyn Embedder, &EmbedHooks, metric)` → `Result<EmbedOutcome, EmbedError>` | Embed every relationship of a type through a bound `Embedder` — the relationship twin of `embed_property`, and the pass `db.relationship_embeddings.embed` runs over every relationship of the type. Records the model id and per-relationship text hashes. |
| `embedding_diagnostics(&graph, node_type, relationship_type)` | Coverage rows (`EmbeddingDiagnostic`: an `EmbeddingCoverage` of embedded / embeddable / store-orphan, with `LengthStats`) for node and relationship types. With no filter, every node type and every relationship type is scanned. |

Relationship vectors are queried through Cypher (`db.relationship_embeddings.query`,
`vector_score(r, …)`, `text_score(r, …)`), which every binding reaches through
the query pipeline. They are written either in a query
(`db.relationship_embeddings.set` / `.embed` over bound relationships) or in bulk
through the two writers above, which share the procedures' store path.

## Schema introspection

| Item | Path | Purpose |
|---|---|---|
| `compute_description(...)` | `kglite::api::introspection::compute_description` | XML schema description for agent system prompts. |
| `compute_schema(&dir)` | `kglite::api::introspection::compute_schema` | Structured `SchemaOverview` (node types, edge types, indexes). |
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
| `may_invoke_embedder(&parsed)` | Whether the AST contains a callback-capable embedding procedure, including nested query forms. |
| `generate_explain_result(...)` | Build an EXPLAIN-style plan as a CypherResult. |
| `CypherQuery`, `CypherResult`, `OutputFormat` | Data types. |

## Session (canonical query + transaction surface)

The canonical query and transaction pipeline for Rust-side bindings.

| Item | Path | Purpose |
|---|---|---|
| `Session` | `kglite::api::session::Session` | Shared graph state with commit-swap semantics. |
| `Transaction` | `kglite::api::session::Transaction` | Snapshot/working CoW transaction state. |
| `CommitOutcome` | `kglite::api::session::CommitOutcome` | `NoWritesNoOp` / `Committed{new_version}` / `ConflictDetected{current_version, base_version}`. |
| `ExecuteOptions` | `kglite::api::session::ExecuteOptions` | Params, deadline, row/work budget, lazy hint, disabled passes, embedder, value codecs, cancellation, write scope, and provenance. |
| `ExecuteOutcome` | `kglite::api::session::ExecuteOutcome` | `result: CypherResult` + `is_mutation: bool` + `output_format: OutputFormat`. |
| `execute_read(&graph, query, &opts)` | `kglite::api::session::execute_read` | Run a read query. |
| `execute_mut(&mut graph, query, &opts)` | `kglite::api::session::execute_mut` | Run a mutation. |

## Dataset loaders

The pre-packaged dataset loaders (SEC EDGAR, Wikidata) are no
longer part of the kglite core API surface. They live in the
separate kglite-datasets project. `kglite::api::datasets::*` and
the dataset-loader Cargo features have been removed.

kglite loads the graphs those loaders produce via the ordinary
lifecycle API. To ingest RDF directly, use the kept RDF/N-Triples
loaders.

## Relationship identity migration

Relationship values now carry an executor-only statement incarnation so a
collected relationship cannot silently target a different edge after slot
reuse. Rust callers that constructed `RelValue` with a struct literal should
use `RelValue::new(id, start_id, end_id, rel_type, properties)`; the constructor
sets the internal identity to absent. Its five arguments match the former
public fields.

The identity is invisible to every comparison. `PartialEq`, `Eq`, `Hash`,
`PartialOrd` and `Ord` are hand-written over the five public fields only. Two
`RelValue`s describing the same edge are therefore one key in a `HashSet`, one
group under `DISTINCT` and one position under `ORDER BY`, whatever identity they
carry. Serde omits the field, and public result publication clears it.

The identity does gate the small set of operations that write through a
relationship value: the `db.relationship_embeddings.*` procedures and `DELETE`.
They compare the token explicitly and refuse a value this statement did not
bind or whose slot it has since retired.

Low-level callers constructing `api::cypher::EdgeBinding` directly must add
`incarnation: None` to the literal. Match execution supplies a statement token
internally. Callers that only read `source`, `target`, and `edge_index` require
no change.

## Semver

`kglite::api::*` items above are the documented surface: locked against
accidental drift by the CI API baseline, and every intentional break is
announced in `CHANGELOG.md`. Pre-1.0 that break may arrive in **any** release,
including a patch — see [Stability policy](#stability-policy). Anything outside
that surface — `kglite::graph::*`, `kglite::datatypes::*`, raw module paths —
is internal and may move freely in any release.

| Change kind | Bumps |
|---|---|
| Additive item/options field in `api::*` | Patch or minor, with API baseline update |
| Intentional breaking change before 1.0 | Any release, patch included, plus changelog/migration guidance |
| Internal rearrangement (non-api items) | Patch |

The `.kgl` format is versioned separately from the source API. The current
writer emits RGF v9 above the size floors and RGF v8 otherwise, both with the Postcard codec. The reader accepts v9, v8, v7, v6 and v5. RGF
v4/bincode and older containers are
rejected with a clear migration/rebuild path. Convert pre-0.14 artifacts with
kglite 0.13.4 before handing them to a current binding.

## Where to find each item

```
kglite::                    (the crate root)
├── api::                   (this stable surface)
│   ├── cypher::            (parse / plan / execute primitives)
│   └── session::           (canonical Cypher pipeline + transactions)
├── datatypes::             (internal — use api::Value)
├── error                   (internal — use api::KgError)
├── graph::                 (internal — engine submodules)
```
