# Design Decisions

Why KGLite makes the choices it does. Each section explains a design
tradeoff — what was chosen, what was rejected, and why.

## Why embedded (no server)

KGLite runs inside your Python process. There is no server, no network
protocol, no connection pool, no Docker container.

**What this enables:**
- Zero infrastructure — `pip install` and you're done
- No latency — function calls, not network round-trips
- No deployment — the graph lives in your process and persists to a file
- Reproducible — the `.kgl` file is a complete snapshot

**What this costs:**
- Single-process only (no concurrent access from multiple processes)
- Memory-bound (the graph must fit in RAM)
- No built-in replication or high availability

**When this breaks down:** If you need multi-process access, horizontal
scaling, or a always-on service. Use Neo4j, ArangoDB, or a similar
server-based graph database for those cases. KGLite ships a
Bolt-protocol server (Neo4j's wire protocol) so that external tools can
talk to a KGLite process when they need to, without changing the
embedded-by-default design.

## Why an LLM-agent surface

The library is designed so that handing a graph to an agent is one
step, not a project. Three concrete decisions follow from this:

- **`describe()` returns an XML schema** shaped for system prompts —
  node types, edge types, sample properties, structural rules — so the
  agent can write correct Cypher on its first turn instead of
  discovering the schema by trial and error.
- **The bundled `kglite-mcp-server` binary** exposes `cypher_query`,
  `graph_overview`, structural validators, and source-file tools over
  MCP stdio. Any `.kgl` file becomes a Claude / Cursor / MCP-capable
  workspace tool with one command.
- **Honest Cypher semantics over coverage maximalism.** Silent wrong
  rows would poison an agent's reasoning. The
  [conformance runner](cypher-conformance.md) cross-checks KGLite
  against Neo4j on demand, and the
  [test-fortification work](https://github.com/kkollsga/kglite/blob/main/CHANGELOG.md#0952)
  pinned 30+ NULL / boundary / unicode behaviours into the suite. The
  preference is to say "we don't support that yet" rather than to
  return a misleading row.

This is the differentiator: every embedded graph DB stores triples.
KGLite is the one whose default consumer is an LLM.

## Why Rust + PyO3

The graph operations (traversal, filtering, pattern matching) are
CPU-intensive and benefit from native code. Python is too slow for
large-graph operations; C/C++ extensions are painful to write and
distribute.

**Why Rust specifically:**
- Memory safety without garbage collection (no GC pauses during queries)
- `maturin` makes building and distributing Python wheels trivial
- Cross-platform builds (macOS, Linux, Windows) from a single CI pipeline
- `petgraph` provides a mature, well-tested graph library

**Why PyO3 specifically:**
- First-class Python integration (classes, iterators, exceptions)
- GIL management for thread safety
- Direct access to Python objects (DataFrames, dicts) without serialization

## Why a primary type (plus secondary labels)

Each node has exactly one **primary** type label (e.g., `Person`,
`Company`), set at creation and immutable via label operations. Since
0.10.5 nodes can also carry optional **secondary** labels
(`CREATE (n:A:B)`, `SET n:B`, `REMOVE n:B`); `labels(n)` returns the
full list, primary first. See
[multi-label rationale](multi-label-rationale.md) for the complete
design.

**Why the primary type stays load-bearing:**
- Enables type-indexed storage — `HashMap<String, Vec<NodeIndex>>` gives
  O(1) lookup by type, which is the most common access pattern
- Schema is clearer — each type has a fixed set of properties
- String interning and `TypeSchema` sharing key off the primary type

Secondary labels are additive tags resolved through a separate
`secondary_label_index` — they don't participate in property-schema
sharing, which is why retyping a node is `SET n.type = 'NewType'`
rather than a label swap.

## Why a Cypher subset

KGLite implements a subset of Cypher (the query language used by Neo4j).
Not the full spec — a practical subset covering the operations most
applications need.

**What's included:**
- MATCH, WHERE, RETURN, ORDER BY, LIMIT, SKIP
- CREATE, SET, DELETE, MERGE
- OPTIONAL MATCH, WITH, UNWIND, UNION
- Aggregation, CASE, list comprehensions, shortestPath()
- Window functions, date arithmetic, CALL...YIELD

**What's excluded:**
- Variable-length path patterns with filters
- APOC procedures
- Full-text indexing (replaced by `text_score()` / `search_text()`)
- Schema constraints (CREATE CONSTRAINT)

**Why Cypher at all?** It's the most widely known graph query language.
AI agents can write it without learning a custom API. The fluent API
(`select().where().traverse()`) remains available for Python code that
benefits from it, but it is not a peer surface — new query
capabilities land in Cypher first, and consumers should reach for
`cypher()` by default.

## Why petgraph

[petgraph](https://github.com/petgraph/petgraph) is the standard graph
library in Rust. KGLite uses `StableDiGraph` specifically:

- **Stable indices**: Node indices survive deletions (critical for
  incremental updates — you don't want to re-index after every delete)
- **Directed**: Relationships have direction (`a → b`), which is
  natural for domain modeling
- **Weighted edges**: Edge data (properties) stored inline
- **Mature**: Well-tested, widely used, good performance characteristics

**What petgraph doesn't provide** (and KGLite adds):
- Property storage (petgraph only stores one weight per node/edge)
- Indexes (type, property, range, composite)
- Query language (Cypher)
- Persistence (serialization)

## Why lazy evaluation

`cypher()` returns a `ResultView`, not a list of dicts. Rows convert
from Rust to Python one at a time as you iterate.

**Why this matters:**
- A `MATCH (n:Item) RETURN n` over 100k nodes doesn't allocate 100k
  Python dicts upfront
- If you only need the first 10 results, only 10 rows are materialized
- Memory usage stays bounded even for large result sets

**When to materialize:** Call `to_df=True` to get a pandas DataFrame
(useful for analysis), or `list(result)` to get all rows at once.

## Why copy-on-write

The `KnowledgeGraph` struct wraps its data in `Arc<DirGraph>`. Cloning
is cheap (just increments a reference count). Mutations use
`Arc::make_mut`, which only copies the data if there are other references.

**What this enables:**
- **Transactions**: `begin()` clones the Arc for a snapshot. If the
  transaction is rolled back, the original is untouched.
- **Fluent chains**: Each chained method returns a new `KnowledgeGraph`
  with a different selection but the same underlying graph data.
- **Thread safety**: Multiple readers can share the same graph via Arc
  without locks.

## Why no R-tree for spatial

Spatial queries use bounding-box pre-filtering followed by precise
geometry checks (point-in-polygon, distance). There is no R-tree.

**Why this is good enough:**
- Most spatial queries filter by type first (which is O(1) via type index),
  reducing candidates to hundreds or thousands — not millions
- Bounding-box rejection is cheap and eliminates most non-matching nodes
- An R-tree would add complexity, memory overhead, and maintenance cost
  for a marginal speedup on KGLite's typical graph sizes

**When this might change:** If KGLite needs to support millions of
spatial nodes with complex polygon queries, an R-tree would be worth
the complexity.
