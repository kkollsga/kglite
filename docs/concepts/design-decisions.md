# Design decisions

These are the current product and architecture choices. Historical phase
documents explain how the repository arrived here, but they are not the
authority for present behavior.

## Embedded first, with protocol adapters

The primary experience is still an in-process library: `pip install kglite`
or `cargo add kglite`, load a graph, and call it directly. This removes a
mandatory service hop and keeps graph state deployable with the application.

Embedded-first does not mean “Python-only” or “no protocols.” The same core is
wrapped by bundled MCP and Bolt servers, and non-Rust bindings can use the C
ABI. Those adapters are useful integration boundaries, but they do not turn
KGLite into a replicated, horizontally scaled database service.

The storage modes also change the old “must fit in RAM” limit. `mapped` spills
property columns during construction, and `disk` uses CSR plus mmap for large
graphs. The remaining tradeoffs are explicit: no built-in replication or high
availability, writes serialize, and in-memory performance remains the design
centre.

## A core boundary shared by every wrapper

Logic that is reusable across bindings belongs in `kglite::api`; only
environment-specific conversion, async, lifecycle, display, and wire behavior
stays in a wrapper. This prevents the Python, C, MCP, and Bolt surfaces from
quietly implementing different graph semantics.

The core is synchronous. An async protocol server can move a core call onto
its chosen runtime without imposing Tokio on an embedded Rust or C consumer.

## Cypher first

Cypher is the common query surface for people, agents, and every binding. New
per-query capabilities normally land as Cypher functions or procedures, which
makes them immediately available through Python, Rust, C, MCP, and Bolt.
Direct API methods remain appropriate for lifecycle, storage configuration,
dataset loading, embedder registration, and other operations Cypher cannot
express.

KGLite implements a documented subset rather than pretending to implement all
of openCypher. Unsupported syntax should fail clearly. Supported semantics are
protected by an independently authored differential corpus and optional Neo4j
comparison. Silent wrong rows are worse than a declared gap.

## One primary type plus secondary labels

A node has one immutable primary type and may have secondary labels. The
primary type anchors schema sharing, ID lookup, property layout, and the most
important candidate indexes. Secondary labels are additive tags with their
own index. See [Multi-label rationale](multi-label-rationale.md) for the full
contract.

## Backend traits instead of one concrete graph

The memory backend uses petgraph's `StableDiGraph`, whose stable indices are a
good fit for incremental mutation and algorithms. Mapped mode retains a graph
topology while moving property storage toward mmap. Disk mode cannot pretend
to be a `StableDiGraph`: it uses CSR and persistent mmap stores.

`GraphRead` and `GraphWrite` are therefore the reusable contract. Algorithms
that truly require a concrete petgraph representation must gate that path and
provide a backend-appropriate alternative. This keeps large-graph storage
from distorting the default in-memory design.

## Explicit indexes and planner fallbacks

Indexes cost memory and mutation work, so property/range/composite indexes are
explicit rather than created for every field. The planner uses index and
cardinality information when present and retains correct scan paths when it is
not. Type and ID routing remain foundational because the data model makes them
high-value across workloads.

## Result values are lazy where the shape permits it

The executor may return a lazy descriptor for eligible read projections. A
Python `ResultView` then materializes and caches cells row by row while keeping
the graph snapshot alive. More complex operators still materialize Rust rows
as required. The contract is bounded Python conversion and stable result
ownership, not universal operator streaming.

## Copy-on-write snapshots and a separate shared session

An `Arc<DirGraph>` makes immutable snapshots cheap. `FrozenGraph` exposes that
model directly for readers. A mutable `KnowledgeGraph` stays a single-owner
handle; copying on mutation preserves existing snapshots.

Servers need composable writes as well as parallel reads, so `Session` adds a
small synchronization boundary: reads clone the current `Arc` and release the
lock, while writes serialize through a writer lock and atomically publish the
next state. This avoids lost updates without putting a global lock around
query execution.

## Per-query R-tree for spatial joins

The spatial join builds an R-tree from the current query's indexed side. This
provides subquadratic candidate discovery without maintaining another graph
index through every mutation or persistence transition. Precise `geo`
predicates still decide the result after envelope filtering.

The tradeoff is rebuild cost per eligible query. A persistent spatial index
would be a different design with mutation, format, and backend implications;
it should be justified by measurements rather than assumed to be free.

## Two persistence products, two lifecycles

Memory and mapped graphs save portable `.kgl` snapshots. The RGF v5 container
selects Postcard explicitly and uses JSON metadata plus compressed sections so topology,
columns, embeddings, and timeseries data can evolve with explicit version
checks.

Disk graphs are directories of immutable generations. A writer completes a
new generation before atomically replacing `CURRENT`; readers keep using the
generation they opened. This provides crash-safe publication and stable
reader snapshots, not multi-writer transactions or replication.

Persisted user data outlives the code that wrote it. Consequently, explicit
read compatibility or a clear hard-break/rebuild error is required even
though obsolete source APIs are removed rather than deprecated.

## In-memory performance wins conflicts

Mapped and disk modes exist for exploration beyond RAM, but the in-memory
engine is the core product. Shared planner or executor changes must be checked
on small in-memory graphs. Disk workarounds belong behind storage-mode or
scale gates when a general solution would slow the default path.
