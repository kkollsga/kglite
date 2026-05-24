# Core Concepts

## Nodes, Relationships, and Selections

**Nodes** have three built-in fields — `type` (label), `title` (display name), `id` (unique within type) — plus arbitrary properties. Each node has exactly one type.

**Relationships** connect two nodes with a type (e.g., `:KNOWS`) and optional properties. The Cypher API calls them "relationships"; the fluent API calls them "connections" — same thing.

**Selections** (fluent API) are lightweight views — a set of node indices that flow through chained operations like `select().where().traverse()`. They don't copy data.

**Atomicity.** Each `cypher()` call is atomic — if any clause fails, the graph remains unchanged. For multi-statement atomicity, use `graph.begin()` transactions. Durability only via explicit `save()`.

## How It Works

KGLite stores nodes and relationships in a Rust graph structure ([petgraph](https://github.com/petgraph/petgraph)). Python only sees lightweight handles — data converts to Python objects on access, not on query.

- **Cypher queries** parse, optimize, and execute entirely in Rust, then return a `ResultView` (lazy — rows convert to Python dicts only when accessed)
- **Fluent API** chains build a *selection* (a set of node indices) — no data is copied until you call `collect()`, `to_df()`, etc.
- **Persistence** is via `save()`/`load()` binary snapshots — there is no WAL or auto-save

## Storage Modes

KGLite has three storage backends. The Python API is identical
across all three; the trade-off is in-memory speed vs. on-disk
scalability.

| Mode | Construct | Where data lives | Best for |
|---|---|---|---|
| **Default (in-memory)** | `KnowledgeGraph()` | Heap | Small / medium graphs (<5 M nodes), prototyping, fastest queries |
| **Mapped** | `KnowledgeGraph(storage="mapped")` | mmap-backed columnar files | RAM-friendly as the graph grows; same query speed as in-memory for typed lookups (O(log N) property index) |
| **Disk** | `KnowledgeGraph(storage="disk", path="/data/g")` | mmap CSR + segments | 100 M+ nodes (Wikidata-scale); kept lazy-loaded so the OS pages in only what queries touch |

Save/load works for all three. For disk mode, `save()` consolidates
segment artifacts into a top-level `disk_graph_meta.json` so
`kglite.load(path)` can reconstitute the graph.

**When optimizing**, in-memory wins. Disk and mapped exist for
data that's too big to keep on the heap; they're not "faster"
backends. For Wikidata-scale workflows, see {doc}`guides/datasets`.

## Return Types

All node-related methods use a consistent key order: **`type`, `title`, `id`**, then other properties.

### Cypher

| Query type | Returns |
|-----------|---------|
| Read (`MATCH...RETURN`) | `ResultView` — lazy container, rows converted on access |
| Read with `to_df=True` | `pandas.DataFrame` |
| Mutation (`CREATE`, `SET`, `DELETE`, `MERGE`) | `ResultView` with `.stats` dict |
| `EXPLAIN` prefix | `str` (query plan, not executed) |

**Spatial return types:** `point()` values are returned as `{'latitude': float, 'longitude': float}` dicts.

### ResultView

`ResultView` is a lazy result container returned by `cypher()`, centrality methods, `collect()`, and `sample()`. Data stays in Rust and is only converted to Python objects when you access it — making `cypher()` calls fast even for large result sets.

```python
result = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")

len(result)        # row count (O(1), no conversion)
result[0]          # single row as dict (converts that row only)
result[-1]         # negative indexing works

for row in result: # iterate rows as dicts (one at a time)
    print(row)

result.head()      # first 5 rows → new ResultView
result.head(3)     # first 3 rows → new ResultView
result.tail(2)     # last 2 rows → new ResultView

result.to_list()   # all rows as list[dict] (full conversion)
result.to_df()     # pandas DataFrame (full conversion)

result.columns     # column names: ['n.name', 'n.age']
result.stats       # mutation stats (None for read queries)
```

Because `ResultView` supports iteration and indexing, it works anywhere you'd use a list of dicts — existing code that iterates over `cypher()` results continues to work unchanged.

### Node dicts

Every method that returns node data uses the same dict shape:

```python
{'type': 'Person', 'title': 'Alice', 'id': 1, 'age': 28, 'city': 'Oslo'}
#  ^^^^             ^^^^^             ^^^       ^^^ other properties
```

### Retrieval methods (cheapest to most expensive)

| Method | Returns | Notes |
|--------|---------|-------|
| `len()` | `int` | No materialization |
| `indices()` | `list[int]` | Raw graph indices |
| `ids()` | `list[Any]` | Flat list of IDs |
| `titles()` | `list[str]` | Flat list (see below) |
| `get_properties(['a','b'])` | `list[tuple]` | Flat list (see below) |
| `collect()` | `ResultView` or grouped dict | Full node dicts |
| `to_df()` | `DataFrame` | Columns: `type, title, id, ...props` |
| `node(type, id)` | `dict \| None` | O(1) hash lookup |

### Flat vs. grouped results

`titles()`, `get_properties()`, and `collect()` automatically flatten when there is only one parent group (the common case). After a traversal with multiple parent groups, they return grouped dicts instead:

```python
# No traversal (single group) → flat list
graph.select('Person').titles()
# ['Alice', 'Bob', 'Charlie']

# After traversal (multiple groups) → grouped dict
graph.select('Person').traverse('KNOWS').titles()
# {'Alice': ['Bob'], 'Bob': ['Charlie']}

# Override with flatten_single_parent=False to always get grouped
graph.select('Person').titles(flatten_single_parent=False)
# {'Root': ['Alice', 'Bob', 'Charlie']}
```

### Centrality methods

All centrality methods (`pagerank`, `betweenness_centrality`, `closeness_centrality`, `degree_centrality`) return:

| Mode | Returns |
|------|---------|
| Default | `ResultView` of `{type, title, id, score}` sorted by score desc |
| `as_dict=True` | `{id: score}` — keyed by node ID (unique per type) |
| `to_df=True` | `DataFrame` with columns `type, title, id, score` |
