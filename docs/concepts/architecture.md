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

- `kglite` — engine, storage, Cypher, code-tree parsing, and shared API.
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
  overlapping mutation on the same Python object is rejected.
- `FrozenGraph` is an immutable O(1) snapshot for parallel readers.
- `Session` is the shareable server handle. Reads take momentary snapshots and
  run without holding the session lock; writes serialize and atomically swap
  the committed `Arc`.
- Mutating a graph while another snapshot exists can trigger an
  `Arc::make_mut` copy. This is the cost of snapshot isolation.

Disk readers resolve `CURRENT` once and retain the selected immutable mmap
generation. A retained cross-process writer lease prevents two writers from
publishing concurrently. Readers do not take that writer lock. The detailed
contract and verification matrix live in [Concurrency](concurrency.md).

## Spatial execution

Spatial values are represented with the `geo`/WKT stack. The fused spatial
join builds an `rstar::RTree` for one side of the current query, queries its
envelopes for candidates, and then applies the precise geometry predicate.
The tree is a per-query acceleration structure, not a persistent graph index;
other spatial expressions can still use bounding-box rejection and cached WKT
parsing.

## Persistence

`.kgl` snapshots use the RGF v4 container:

```text
magic RGF\x04
core-data version (u32 LE)
JSON metadata length + metadata
zstd-compressed topology section
zstd-compressed column sections by node type
optional embeddings, timeseries, and secondary-label sections
```

Container and core-data versions are separate. RGF v3 is detected and refused
with a rebuild message; it is not silently interpreted as v4. Within v4,
metadata additions use serde defaults where compatible, while incompatible
embedded cache layouts are detected explicitly. Index definitions are stored
so non-persistent index structures can be rebuilt on load.

Disk mode uses a different lifecycle: writers build a staged generation,
write completion metadata, rename it into `generations/`, then atomically
replace `CURRENT`. A failed or incomplete stage is never selected by a new
reader. Existing readers keep their old immutable generation alive.

## Code-tree ingestion

Code-tree parsers and graph construction are Rust-core functionality under
`crates/kglite/src/code_tree/`. Tree-sitter grammars are compiled into the
extension/core build. The Python package supplies the ergonomic entry points,
not a second Python parser implementation.
