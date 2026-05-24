# Architecture

How KGLite works under the hood. This page explains the internal design —
you don't need any of this to *use* KGLite, but it helps if you're
contributing, debugging performance, or curious about the tradeoffs.

## Layer diagram

```text
┌─────────────────────────────────────────────────────────────┐
│  Python API                                                 │
│  kglite.KnowledgeGraph  (thin PyO3 wrapper)                 │
├─────────────────────────────────────────────────────────────┤
│  Rust core                                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │  Cypher engine           │  │  Fluent API              │ │
│  │  tokenizer → parser →    │  │  select → where →        │ │
│  │  AST → planner →         │  │  traverse → collect      │ │
│  │  executor                │  │                          │ │
│  └──────────┬───────────────┘  └──────────┬───────────────┘ │
│             │                             │                 │
│  ┌──────────▼─────────────────────────────▼───────────────┐ │
│  │  DirGraph                                              │ │
│  │  petgraph::StableDiGraph + indexes + metadata          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Graph storage

The core data structure is `DirGraph`, which wraps a
[petgraph](https://github.com/petgraph/petgraph) `StableDiGraph`.
"Stable" means node indices remain valid even after deletions — important
for long-lived graphs with incremental updates.

Each node stores:
- **Type** (string-interned for memory efficiency)
- **Title** (display name)
- **ID** (unique within type)
- **Properties** as a `Vec<Value>` keyed by a shared `TypeSchema`

The `TypeSchema` pattern means that nodes of the same type share a single
schema reference (`Arc<TypeSchema>`), so adding 100k nodes of the same
type only stores the property names once.

### Indexes

DirGraph maintains several index structures, all rebuilt automatically on load:

| Index type | Data structure | Purpose |
|---|---|---|
| Type index | `HashMap<String, Vec<NodeIndex>>` | O(1) lookup by node type |
| ID index | `HashMap<String, HashMap<Value, NodeIndex>>` | O(1) lookup by (type, id) |
| Property index | `HashMap<IndexKey, HashMap<Value, Vec<NodeIndex>>>` | Equality filters |
| Range index | `BTreeMap<Value, Vec<NodeIndex>>` | Range queries (`>`, `<`, `BETWEEN`) |
| Composite index | Multi-key hash map | Multi-property lookups |

Indexes are opt-in via `create_index()`. The Cypher planner automatically
uses them when available.

## Cypher engine

The Cypher engine is a 4-stage pipeline:

```text
Query string → Tokenizer → Parser → AST → Planner → Executor → ResultView
```

1. **Tokenizer** (`tokenizer.rs`): Splits the query into tokens (keywords,
   operators, literals, identifiers).

2. **Parser** (`parser.rs`): Recursive descent parser that builds a
   `CypherQuery` AST. Pattern matching (e.g., `(a:Person)-[:KNOWS]->(b)`)
   is delegated to a specialized pattern parser.

3. **Planner** (`planner.rs`): Rewrites and optimizes the AST:
   - Predicate pushdown (move WHERE conditions closer to MATCH)
   - Fused operations (OPTIONAL MATCH + COUNT, RETURN + ORDER BY + LIMIT)
   - `text_score()` → `vector_score()` rewriting (triggers embedding)

4. **Executor** (`executor.rs`): Walks the optimized plan, evaluates
   pattern matches, filters rows, computes aggregations, and streams
   results into a `ResultView`.

### ResultView (lazy evaluation)

`ResultView` is a lazy iterator. Rows are converted from Rust to Python
one at a time, so a `MATCH...RETURN` over 100k nodes doesn't allocate
100k Python dicts upfront. Call `to_df=True` to materialize into a
DataFrame, or iterate row-by-row.

## PyO3 bridge

KGLite uses [PyO3](https://pyo3.rs) to expose Rust types to Python:

- **`KnowledgeGraph`**: The main class. Read methods take `&self`;
  mutations use `Arc::make_mut` for copy-on-write semantics.
- **`ResultView`**: Lazy result container. Implements `__iter__` and
  `__len__` for Python iteration.
- **`Transaction`**: Snapshot isolation via `Arc` cloning — the
  transaction gets a cheap copy of the graph state.

All value conversion (Python → Rust → Python) goes through
`py_in::value_from_py()` and `py_out::value_to_py()`.

## Vector search

Embeddings are stored in a columnar `EmbeddingStore` — a dense `Vec<f32>`
per (node_type, text_column) pair. The search is brute-force with
optimizations:

- **Top-k via min-heap**: O(n log k) instead of O(n log n) sort
- **Parallel search**: Uses Rayon when candidates exceed 10k
- **SIMD-friendly cosine**: 4-way parallel accumulators, chunks of 8 for
  auto-vectorization (SSE2/AVX2/NEON)

Supported metrics: cosine (default), dot product, euclidean.

The embedding model itself lives in Python (any object with an `embed()`
method). KGLite calls it via PyO3 when `embed_texts()` is invoked,
then stores the resulting vectors in Rust.

## Spatial operations

Spatial data uses the `geo` crate for geometry operations:

- **Points**: Lat/lon pairs stored as node properties
- **Polygons**: WKT strings parsed and cached as `geo::Geometry`
- **Filtering**: Bounding-box rejection first, then precise
  point-in-polygon or distance checks

There is no R-tree — spatial queries use config-driven resolution
(which properties hold lat/lon or WKT) with bounding-box pre-filtering.

## Persistence

Graphs serialize to `.kgl` files with a versioned binary format:

```text
[0..4]    Magic bytes: "RGF\x02"
[4..8]    Format version (u32)
[8..12]   Metadata length (u32)
[12..N]   JSON metadata (schema, configs, index keys)
[N..]     Gzip-compressed bincode (graph + embeddings + timeseries)
```

Key design choices:
- **String interning** during serialization reduces file size
- **Indexes are NOT persisted** — only index keys are saved; indexes
  rebuild on load (faster than deserializing large hash maps)
- **Backward compatible** — metadata uses `#[serde(default)]` so older
  files load in newer versions
- The embedding model is NOT serialized — call `set_embedder()` again
  after loading

## Memory model

KGLite uses copy-on-write (`Arc::make_mut`) throughout:

- **Cheap cloning**: `graph.clone()` shares the underlying data via `Arc`
- **Transactions**: `begin()` creates a snapshot by cloning the `Arc`
- **Fluent chains**: Each method returns a new `KnowledgeGraph` with a
  modified selection but the same underlying graph data
- **Mutations**: Only the mutating operation pays the copy cost (and only
  if there are other references)
