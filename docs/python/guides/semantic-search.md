# Semantic Search

Store embedding vectors alongside nodes and query them with fast similarity search. Embeddings are stored separately from node properties — they don't appear in `collect()`, `to_df()`, or regular Cypher property access.

## Text-Level API (Recommended)

Register an embedding model once, then embed and search using text column names. The model runs on the Python side — KGLite only stores the resulting vectors.

```python
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._timer = None
        self.dimension = 384  # set in load() if unknown

    def load(self):
        """Called automatically before embedding. Loads model on demand."""
        import threading
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()

    def unload(self, cooldown=60):
        """Called automatically after embedding. Releases after cooldown."""
        import threading
        def _release():
            self._model = None
            self._timer = None
        self._timer = threading.Timer(cooldown, _release)
        self._timer.start()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

# Register once on the graph
graph.set_embedder(Embedder())

# Embed a text column — stores vectors as "summary_emb" automatically
graph.embed_texts("Article", "summary")
# Embedding Article.summary: 100%|████████| 1000/1000 [00:05<00:00]
# → {'embedded': 1000, 'skipped': 3, 'skipped_existing': 0, 'dimension': 384}

# Search with text — resolves "summary" → "summary_emb" internally
results = graph.select("Article").search_text("summary", "machine learning", top_k=10)
# [{'id': 42, 'title': '...', 'type': 'Article', 'score': 0.95, ...}, ...]
```

**Key details:**

- **Auto-naming:** text column `"summary"` → embedding store key `"summary_emb"` (auto-derived)
- **Incremental, three modes:** `embed_texts(mode=…)` — `'missing'` (default) embeds only nodes without a vector; `'changed'` also re-embeds nodes whose **text changed** since the last pass (a per-node content hash is stored to detect this); `'all'` rebuilds the whole store.
- **Progress bar:** shows a tqdm progress bar by default. Disable with `show_progress=False`.
- **Load/unload lifecycle:** if the model has optional `load()` / `unload()` methods, they are called automatically before and after each embedding operation.
- **Provenance:** if the embedder exposes a `model_id` / `model_name` attribute, it's stamped onto the store; `embedding_info()` surfaces it so a model swap is detectable. The model object itself is **not** saved with `save()` — call `set_embedder()` again after deserializing.

```python
# Add new articles, then re-embed — only new ones are processed
graph.embed_texts("Article", "summary")
# → {'embedded': 50, 'skipped': 0, 'skipped_existing': 1000, 'reembedded_changed': 0, ...}

# Edit some article summaries, then re-embed ONLY what changed:
graph.embed_texts("Article", "summary", mode="changed")
# → {'embedded': 12, 'reembedded_changed': 12, 'skipped_existing': 1038, ...}

# Inspect provenance (dimension, count, model id, metric, #hashed):
graph.embedding_info("Article", "summary")
# → {'dimension': 384, 'count': 1050, 'model': 'all-MiniLM-L6-v2', 'metric': 'cosine', 'hashed': 1050}
# Or just the dimension (cheap; None if no store) — handy to detect a model swap:
graph.embedding_dim("Article", "summary")   # → 384

# Combine with filters
results = (graph
    .select("Article")
    .where({"category": "politics"})
    .search_text("summary", "foreign policy", top_k=10))
```

### Carrying vectors across a rebuild

The common "rebuild a fresh graph from a source of truth on each load" workflow
needs the vectors carried forward. `copy_embeddings_from` does it in one call,
matched by node id (carrying dimension, metric, model id, and the per-node text
hashes — so a following `mode='changed'` only re-embeds genuinely new text):

```python
new_graph = build_from_source()              # fresh, no vectors yet
new_graph.copy_embeddings_from(old_graph)    # carry every store by node id
new_graph.embed_texts("Article", "summary", mode="changed")  # fill only the new/changed
# → {'stores_copied': 1, 'vectors_copied': 1050, 'vectors_skipped': 0}  (from copy_embeddings_from)
```

## Low-Level Vector API

If you manage vectors yourself, use the low-level API:

### Storing Embeddings

```python
# Explicit: pass a dict of {node_id: vector}.
# set_embeddings REPLACES the whole store for ('Article', 'summary_emb').
graph.set_embeddings('Article', 'summary', {
    1: [0.1, 0.2, 0.3, ...],
    2: [0.4, 0.5, 0.6, ...],
})

# Or auto-detect during add_nodes with column_types
df = pd.DataFrame({
    'id': [1, 2, 3],
    'title': ['A', 'B', 'C'],
    'text_emb': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
})
graph.add_nodes(df, 'Doc', 'id', 'title', column_types={'text_emb': 'embedding'})
```

### Incremental ingest — `add_embeddings`

`set_embeddings` is a **full replace**: each call discards the existing
store for `(node_type, '{text_column}_emb')`. When you ingest documents in
batches — embed chunks for doc A, then doc B, then doc C — a second
`set_embeddings` call would wipe doc A's vectors.

Use `add_embeddings` for that. It **upserts** into the existing store
(creating it on the first call), so batches coexist without a
read-merge-write cycle in your own code:

```python
graph.add_embeddings('Chunk', 'text', {  # doc A's chunks
    'a:1': [0.1, 0.2, ...],
    'a:2': [0.3, 0.4, ...],
})
graph.add_embeddings('Chunk', 'text', {  # doc B's chunks — A's survive
    'b:1': [0.5, 0.6, ...],
})
# -> {'embeddings_stored': int, 'dimension': int, 'skipped': int, 'store_created': bool}
```

Reach for `set_embeddings` only when you genuinely want to replace the
entire store (e.g. re-embedding everything with a new model).

### Vector Search

Each hit is a dict with `id`, `title`, `type`, `score`, **and all node
properties**. `score` is always present (every metric), and properties are
read live from the node — so a hit carries the same fields before and after
`save()` + reload. You don't need a follow-up `MATCH ... WHERE id IN [...]`
to recover properties.

```python
# Basic search — each hit carries id, title, type, score AND every node
# property (read live, so no follow-up MATCH...WHERE id IN [...] hydrate needed).
results = graph.select('Article').vector_search('summary', query_vec, top_k=10)
# [{'id': 5, 'title': '...', 'type': 'Article', 'score': 0.95, ...all props...}, ...]

# Trim the payload with returning= → id + score + only the named fields
# (ranking-heavy or wide-node workloads):
ranked = graph.select('Article').vector_search(
    'summary', query_vec, top_k=50, returning=['title'])   # → {'id', 'score', 'title'}

# Filtered search — only search within a subset
results = (graph
    .select('Article')
    .where({'category': 'politics'})
    .vector_search('summary', query_vec, top_k=10))

# DataFrame output
df = graph.select('Article').vector_search('summary', query_vec, top_k=10, to_df=True)

# Distance metrics: 'cosine' (default), 'dot_product', 'euclidean', 'poincare'
results = graph.select('Article').vector_search(
    'summary', query_vec, top_k=10, metric='dot_product')
```

### Scaling search with an index (HNSW)

By default vector search is an **exact** brute-force scan: every candidate is
scored. That's the right thing for small/medium stores and for filtered
searches — but on a large corpus, scoring every vector on every query doesn't
scale. Build an **HNSW** approximate-nearest-neighbour index once, and
whole-corpus queries become sub-linear:

```python
graph.embed_texts('Article', 'summary')          # produce vectors
graph.build_vector_index('Article', 'summary')    # opt in (like create_index)

# vector_search / search_text now auto-use the index for whole-corpus queries:
hits = graph.select('Article').search_text('summary', 'machine learning', top_k=10)

# Force an exact scan when you need guaranteed-exact results:
hits = graph.select('Article').vector_search('summary', query_vec, top_k=10, exact=True)
```

It behaves like `create_index`: **opt-in**, and once built it's used
automatically. Key points:

- **Auto-use, with an escape hatch.** A whole-corpus query on a large indexed
  store (≥256 candidates) uses the index; `exact=True` always forces the exact
  scan. The scores returned are on the exact same scale as the brute-force path
  (the index only narrows *which* nodes get scored).
- **Filtered queries stay exact.** A selective `.where(...)` before the search
  falls back to an exact scan automatically — correctness over speed when a
  filter is tight. (So an index helps "search the whole corpus", not "search a
  small filtered slice".)
- **Approximate.** Recall depends on your data and `ef_search`: well-structured
  embeddings (sentence-transformers, bge, OpenAI, etc.) typically get ≥0.99
  recall@10 at the defaults; raise `ef_search` for higher recall at some latency
  cost, or use `exact=True` when you can't tolerate any miss.

  > **Benchmark HNSW on *real* embeddings, not random vectors.** Random unit
  > vectors in high dimensions have no neighbourhood structure — every pair is
  > nearly orthogonal (all cosine sims ≈ 0) — so *any* ANN scores terribly on
  > them (recall can look like ~0.2). That's the curse of dimensionality, not an
  > engine defect: on real embeddings the same index hits ~0.99. If you must
  > sanity-check on synthetic data, query with *stored* vectors (which have a
  > true nearest neighbour) rather than fresh random ones.
- **Metrics.** cosine / dot_product / euclidean are indexable; `poincare` always
  uses the exact path.
- **Lifecycle.** The index is **dropped automatically** whenever the store's
  vectors change (`add_embeddings`, `embed_texts`) or slots are remapped
  (`vacuum`) — rebuild it after such changes. Check with
  `has_vector_index(...)`, remove with `drop_vector_index(...)`.
- **Persisted.** The index is saved inside the `.kgl` (and `to_bytes()`), so a
  reloaded graph keeps it — no rebuild on load.

```python
graph.build_vector_index(
    'Article', 'summary',
    m=16,                 # neighbours per node (higher → better recall, larger index)
    ef_construction=200,  # build-time search width
    ef_search=64,         # default query-time width (higher → better recall, slower)
)
graph.has_vector_index('Article', 'summary')   # True
graph.save('articles.kgl')                       # index travels with the file
```

> The Cypher `text_score()` / `vector_score()` whole-corpus top-k
> (`RETURN vector_score(n, prop, q) AS s ORDER BY s DESC LIMIT k`) auto-uses the
> index too — so agent/MCP semantic search via Cypher benefits as well. The
> end-to-end win is smaller than the fluent API's, though: Cypher's fixed
> per-query cost (parse + plan + projection) is a bigger share of the total, so
> the index saving shows through less at small/medium corpus sizes and widens as
> the corpus grows. A heavily-filtered Cypher query (selective `WHERE`) stays
> exact.

### Choosing a Distance Metric

| Metric | Best for | Why |
|--------|----------|-----|
| `cosine` | General-purpose text/semantic embeddings (OpenAI, Sentence-Transformers, Cohere) | Compares direction, ignores magnitude. Works well when embeddings are normalized or you only care about semantic similarity. |
| `dot_product` | Embeddings where magnitude encodes relevance (MIPS) | Like cosine but magnitude matters — a longer vector scores higher. Useful when the model encodes "importance" in the norm. |
| `euclidean` | Spatial/geometric data, clustering, k-means style lookups | Raw geometric distance. Best when absolute position in the space matters, not just angle. |
| `poincare` | Hierarchical/taxonomic data (ontologies, org charts, category trees) | Hyperbolic geometry naturally encodes tree structure. Nodes near the origin are roots; nodes near the boundary are leaves. 5D Poincaré can outperform 200D Euclidean on hierarchy tasks. |

**Rule of thumb:** If you're using off-the-shelf text embeddings, use `cosine`. If your data has inherent hierarchy and you've trained Poincaré embeddings, use `poincare`.

### Stored Metric

When embeddings are trained for a specific geometry, store the intended metric alongside them so it becomes the default at query time:

```python
# Store Poincaré embeddings with their intended metric
graph.set_embeddings('Concept', 'title', poincare_vectors, metric='poincare')

# Queries now default to poincaré distance — no need to pass metric= each time
results = graph.select('Concept').vector_search('title', query_vec, top_k=10)

# You can still override explicitly
results = graph.select('Concept').vector_search(
    'title', query_vec, top_k=10, metric='cosine')

# list_embeddings() shows the stored metric
graph.list_embeddings()
# [{'node_type': 'Concept', 'text_column': 'title', 'dimension': 5,
#   'count': 500, 'metric': 'poincare'}]
```

Metric resolution order: explicit `metric=` argument > stored metric > `cosine` default.

### Semantic Search in Cypher

`text_score()` enables semantic search directly in Cypher queries. It automatically embeds the query text using the registered model (via `set_embedder()`) and computes similarity:

```python
# Requires: set_embedder() + embed_texts()
graph.cypher("""
    MATCH (n:Article)
    RETURN n.title, text_score(n, 'summary', 'machine learning') AS score
    ORDER BY score DESC LIMIT 10
""")

# With parameters
graph.cypher("""
    MATCH (n:Article)
    WHERE text_score(n, 'summary', $query) > 0.8
    RETURN n.title
""", params={'query': 'artificial intelligence'})

# With explicit metric
graph.cypher("""
    MATCH (n:Article)
    RETURN n.title, text_score(n, 'summary', 'machine learning', 'poincare') AS score
    ORDER BY score DESC LIMIT 10
""")

# Combine with graph filters
graph.cypher("""
    MATCH (n:Article)-[:CITED_BY]->(m:Article)
    WHERE n.category = 'politics'
    RETURN m.title, text_score(m, 'summary', 'foreign policy') AS score
    ORDER BY score DESC LIMIT 5
""")
```

To score against a **pre-computed vector** instead of a text query (no
`set_embedder()` needed), use `vector_score(n, 'summary_emb', $vec)` — the Cypher
counterpart of the fluent `vector_search()` method. Note the surfaces differ:
`text_score()`/`vector_score()` are **Cypher functions** (used in `RETURN`/`WHERE`);
`search_text()`/`vector_search()` are **fluent methods** on a selection.

### Embedding Norm in Cypher

`embedding_norm()` returns the L2 norm of a node's embedding vector. In Poincaré space, norm indicates hierarchy depth: values near 0 are roots, values near 1 are leaves.

```python
# Find the most "root-like" concepts (lowest norm = highest in hierarchy)
graph.cypher("""
    MATCH (n:Concept)
    RETURN n.name, embedding_norm(n, 'title') AS depth
    ORDER BY depth ASC LIMIT 10
""")

# Find leaf nodes (high norm = deep in hierarchy)
graph.cypher("""
    MATCH (n:Concept)
    WHERE embedding_norm(n, 'title') > 0.8
    RETURN n.name, embedding_norm(n, 'title') AS depth
""")
```

## Embedding Utilities

```python
graph.list_embeddings()
# [{'node_type': 'Article', 'text_column': 'summary', 'dimension': 384, 'count': 1000, 'metric': None}]

graph.remove_embeddings('Article', 'summary')

# Retrieve all embeddings for a type (no selection needed)
embs = graph.embeddings('Article', 'summary')
# {1: [0.1, 0.2, ...], 2: [0.4, 0.5, ...], ...}

# Retrieve embeddings for current selection only
embs = graph.select('Article').where({'category': 'politics'}).embeddings('summary')

# Get a single node's embedding (O(1) lookup, returns None if not found)
vec = graph.embedding('Article', 'summary', node_id)
```

Embeddings persist across `save()`/`load()` cycles automatically.

## Embedding Export / Import

Export embeddings to a standalone `.kgle` file so they survive graph rebuilds:

```python
# Export all embeddings
stats = graph.export_embeddings("embeddings.kgle")
# {'stores': 2, 'embeddings': 5000}

# Export only specific node types
graph.export_embeddings("embeddings.kgle", ["Article", "Author"])

# Import into a fresh graph — matches by (node_type, node_id)
result = graph.import_embeddings("embeddings.kgle")
# {'stores': 2, 'imported': 4800, 'skipped': 200}
```

A `.kgle` carries each store's **provenance** — its `metric`, the embedder
`model_id`, and per-node text hashes — so a rebuild-from-`.kgle` pipeline keeps
it: after import, `embedding_info()` reports the model/metric, and
`embed_texts(mode='changed')` re-embeds only genuinely-changed text instead of
everything. Current releases import `.kgle` v3/Postcard only. Convert v1/v2
files with kglite 0.13.4 by importing them into the matching graph and
re-exporting them before upgrading.
