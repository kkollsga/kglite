# Architecture

KGLite is an embedded Rust graph engine with several thin delivery surfaces.
This page describes the current source layout and runtime boundaries. The
[generated project facts](../_generated/project-facts.md) page records facts
that come directly from Cargo metadata, packaging metadata, active workflows,
and source constants.

## Layer diagram

```text
 Python                Rust applications          Non-Rust applications
 kglite-py (PyO3)      kglite-mcp-server          kglite-c (C ABI)
                       kglite-bolt-server                 │
          └──────────────────┬────────────────────────────┘
                             ▼
                    kglite::api boundary
           lifecycle · query execution · errors · storage
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     Cypher parser/planner/executor   Shared graph primitives
              └──────────────┬──────────────┘
                             ▼
              DirGraph metadata and indexes
                             │
                       GraphBackend
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        MemoryGraph      MappedGraph      DiskGraph
        StableDiGraph    graph + mmap     CSR + mmap
```

The core crate stays synchronous. Python owns GIL handling and Python object
conversion; protocol servers own their async runtimes, wire formats, logging,
and connection lifecycle. Rust-side wrappers call `kglite::api` directly.
Other languages use the C ABI rather than reaching into graph internals. See
the [boundary principle](../rust/boundary-principle.md) for the full rule.

## Workspace

The six crates have distinct responsibilities:

- `kglite` — engine, storage, Cypher, and shared API. (Code-graph parsing moved to the standalone codingest project.)
- `kglite-py` — PyO3 classes and Python-specific conversion.
- `kglite-c` — stable C ABI for non-Rust bindings.
- `kglite-mcp-server` — MCP protocol adapter over the core.
- `kglite-bolt-server` — Bolt protocol adapter over the core.
- `kglite-cli` — command-line client.

The exact member paths and shared version are generated on the
[project-facts page](../_generated/project-facts.md).

## Graph container and storage backends

`DirGraph` owns graph-wide metadata: schemas, string interning, index
definitions, secondary labels, embeddings, temporal/spatial configuration,
transaction versioning, and caches. Its topology is behind `GraphBackend`,
which implements the `GraphRead` and `GraphWrite` traits.

The three user-selectable modes are:

| Mode | Topology and properties | Persistence role |
|---|---|---|
| `memory` | Heap-resident `StableDiGraph`; fastest default | Saves to a `.kgl` snapshot |
| `mapped` | Petgraph topology with property columns forced to mmap spill during build | Saves to a `.kgl` snapshot |
| `disk` | CSR topology and mmap-backed stores in a directory | The directory is the graph |

`RecordingGraph` is an internal test wrapper, not a fourth user mode. It logs
trait-path operations to verify that new backends can stay behind the storage
interface.

In-memory performance is the product gate. Disk-specific safety or scale work
must remain gated to disk mode or large graphs rather than slowing ordinary
in-memory queries.

### Indexes

`DirGraph` coordinates several index families. Their physical representation
can differ by backend:

| Index | Purpose |
|---|---|
| Type and secondary-label indexes | Candidate discovery by label |
| ID index | Lookup by `(primary type, id)` |
| Property index | Equality and `IN` lookup |
| Range index | Ordered comparisons and ranges |
| Composite index | Multi-property lookup |
| Disk prefix/property bundles | Persistent disk-mode candidate routing |

Index definitions are explicit. The planner estimates and chooses among
available routes; a missing optional index falls back to scanning candidates.

## Query execution

The Cypher path is:

```text
query → tokenizer → parser/AST → ordered planner passes → executor → CypherResult
```

- The ordered optimizer registry in
  `crates/kglite/src/graph/languages/cypher/planner/mod.rs` is the source of
  truth for pass order and stable pass names.
- Shared matching, filtering, and traversal primitives live under
  `crates/kglite/src/graph/core/` so Cypher and the fluent API do not grow
  separate engines.
- The executor runs synchronously against `GraphRead`/`GraphWrite` and enforces
  query deadline, cancellation, and work/row budgets.

`ResultView` is lazy at the Python boundary for eligible projections: the
executor can retain row bindings and materialize cells on demand. Other query
shapes hold already-computed Rust values. Either way, conversion into Python
objects is deferred until rows are accessed; this is not a promise that every
query operator streams without intermediate Rust rows.

## Concurrency and snapshots

Graph state is shared through `Arc<DirGraph>`:

- `KnowledgeGraph` is a single-owner Python handle. Concurrent reads are safe;
  a method call that overlaps a mutation on the same Python object waits
  for it.
- `FrozenGraph` is an immutable O(1) snapshot for parallel readers.
- `Session` is the shareable server handle. Reads take momentary snapshots and
  run without holding the session lock; writes serialize and atomically swap
  the committed `Arc`.
- Mutating a graph while another snapshot exists can trigger an
  `Arc::make_mut` copy. This is the cost of snapshot isolation.

Disk readers resolve `CURRENT` once and retain the selected immutable mmap
generation. A cross-process writer lease prevents two writers from publishing
concurrently; it is held from the first mutation until the publish, not for the
writer's lifetime. Readers do not take that writer lock. The detailed contract
and verification matrix live in [Concurrency](concurrency.md).

## Spatial execution

Spatial values are represented with the `geo`/WKT stack. The fused spatial
join builds an `rstar::RTree` for one side of the current query, queries its
envelopes for candidates, and then applies the precise geometry predicate.
The tree is a per-query acceleration structure, not a persistent graph index;
other spatial expressions can still use bounding-box rejection and cached WKT
parsing.

## Persistence

`.kgl` snapshots use the RGF v9 container above the size floors and the RGF v8 container otherwise. This is the v8 layout; a v9 file adds a data region of raw sections after it, described in [The `.kgl` container](../rust/kgl-container.md):

```text
magic RGF\x08 (RGF\x09 for a v9 file)
codec tag (Postcard)
core-data version (u32 LE)
JSON metadata length + metadata + header/metadata CRC32 (v8)
zstd-compressed topology section
zstd-compressed column sections by node type
optional node-embedding, relationship-embedding (core data version 4 only),
  timeseries and secondary-label sections
optional rebuildable index sections: node and relationship HNSW, BM25 text
```

Container, codec, and core-data versions are separate. The current reader
accepts RGF v9, v8, v7, v6 and v5. RGF v4/bincode and older containers are detected
and refused with a 0.13.4 conversion or rebuild message. Within v5, metadata
additions use serde defaults where compatible, while incompatible embedded
cache layouts are detected explicitly. Index definitions are stored so
non-persistent index structures can be rebuilt on load.

Disk mode uses a different lifecycle: writers build a staged generation,
write completion metadata, rename it into `generations/`, then atomically
replace `CURRENT`. A failed or incomplete stage is never selected by a new
reader. Existing readers keep their old immutable generation alive.

A save reuses unchanged files instead of rewriting them:

- A type whose column file did not change is hard-linked into the new generation rather than rewritten.
- `id_indices.bin` and `type_indices.bin` are hard-linked the same way while the live index is an unchanged view of the previous file.
- The `title`/`nid` lookup bundles are carried instead of rebuilt while nothing that writes a title or an id has happened since they were built.

A save then deletes generations older than the previous one
(`KGLITE_KEEP_GENERATIONS`), except any generation a live value in the process
still maps.

A directory carries its own layout revision: `disk_graph_meta.json` holds a
`disk_format` number, `columns_meta.json` is an envelope
(`{"format": 2, "types": [...], "files": {...}, "sidecars": {...}}`), and
`id_indices.bin` has a version.

Column data is one immutable file per node type under
`seg_000/type_columns/`, named by a hash of the type name and listed in the
envelope. Every reader maps those files read-only. A type whose columns
the mmap layout cannot hold (a column mixing kinds) is a zstd sidecar under
`columns/<hex>/`. The envelope's `sidecars` map records which directory
holds which type. A type or property name never becomes a path component.

Writes to a reopened graph work as follows:

- Rows appended to a type that the graph serves from its column file go into a tail store beside the mapped file. The save writes the file's regions and the tail's into the next generation.
- A `SET` over a mapped column is written as the file's cells with the changed cells laid over them, without copying the type onto the heap.

A type whose ids are all `Int64` has its id index stored as a sorted
`(i64, node)` array, 12 bytes an id, searched in the mapping. A write after a
reopen layers a small delta over it, and the next save merges the delta into the
new file as it streams. A save re-points the live graph at the files it just
published, so a saved graph holds no heap copy of the columns or the id index it
wrote.

This build reads directories written before the
revision existed; their shared `columns.bin` is split per type on the next
save. It refuses one newer than it knows, naming both numbers. kglite 0.19.0
and earlier cannot read a directory this build saves.

## Code-graph ingestion

Code-graph parsing and construction live in the standalone codingest project,
a sibling workspace that links this engine as a crate and emits ordinary
kglite graphs. kglite itself carries no tree-sitter grammars. It serves,
queries, and reads code graphs: `api::code_entities`, `read_code_source`,
`explore`, and the rev procedures all operate on any code-schema graph.
