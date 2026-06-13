# Import and Export

## Saving and Loading

```python
graph.save("my_graph.kgl")
loaded_graph = kglite.load("my_graph.kgl")
```

Save files (`.kgl`) use a pinned binary format (bincode with explicit little-endian, fixed-int encoding). Files are forward-compatible within the same major version. For sharing across machines or long-term archival, prefer a portable format (GraphML, CSV).

### `open()` — load-or-create lifecycle

For an app that persists to one file, `kglite.open(path)` is the ergonomic
entry point: it loads the graph if the file exists and creates a fresh one if
it doesn't, and the returned graph **remembers the path**.

```python
g = kglite.open("app.kgl")          # loads if present, else creates
g.cypher("CREATE (:Person {name: 'Alice'})")
g.save()                             # no path needed — writes back to app.kgl
```

Use it as a context manager to auto-save on clean exit:

```python
with kglite.open("app.kgl") as g:
    g.cypher("CREATE (:Person {name: 'Bob'})")
# snapshotted to app.kgl on block exit
```

- `save()` with no argument writes to the remembered path; passing a path
  (`save("other.kgl")`) updates the remembered target ("save as"). A graph built
  in memory with no path raises `ValueError` if you call `save()` with no path.
- `kglite.load(path)` also remembers its path, so bare `save()` works after a load.
- The context manager **skips the save if the block raised** — the on-disk file
  keeps its last good state. `close()` persists explicitly.

> **Not crash safety.** Auto-save-on-close is a *clean-exit* checkpoint — a hard
> crash (`kill -9`, power loss) mid-session writes nothing. Durable-on-commit
> with crash recovery is a separate capability.

## Export Formats

```python
graph.export('my_graph.graphml', format='graphml')  # Gephi, yEd
graph.export('my_graph.gexf', format='gexf')        # Gephi native
graph.export('my_graph.json', format='d3')           # D3.js
graph.export('my_graph.csv', format='csv')           # creates _nodes.csv + _edges.csv

graphml_string = graph.export_string(format='graphml')
```

## NetworkX Interop

Round-trip with [NetworkX](https://networkx.org/) for graph algorithms.
KGLite is a directed multigraph with typed nodes/edges, so the lossless
target is `networkx.MultiDiGraph`: each node's `id` is the networkx node
key (with `node_type`, `title`, and every property as node attributes),
and each edge's `connection_type` is the edge key (so parallel edges of
different types between the same pair stay distinct).

Requires the `networkx` extra: `pip install kglite[networkx]`.

```python
import networkx as nx

# Export, run an algorithm, write the scores back.
nxg = graph.to_networkx()              # -> nx.MultiDiGraph
scores = nx.pagerank(nxg)               # {node_id: rank} (pagerank needs scipy)

import pandas as pd
df = pd.DataFrame(
    [{'id': nid, 'pagerank': rank} for nid, rank in scores.items()]
)
# Update existing nodes in place (matched by id), or with Cypher SET:
graph.add_nodes(df, 'Person', 'id', conflict_handling='update')
# graph.cypher("MATCH (n) WHERE n.id = $id SET n.pagerank = $r", ...)

# Import a plain networkx graph (defaults applied where attrs are absent).
g2 = kglite.from_networkx(nxg, default_node_type='Node', default_edge_type='RELATED')
```

`from_networkx()` accepts `Graph` / `DiGraph` / `MultiGraph` /
`MultiDiGraph`; undirected edges become a single directed edge each.
`to_networkx()` exports the full graph (the active selection is ignored
in v1).

## Neo4j Export

Push a graph (or the active selection) to a live Neo4j database over Bolt,
using batched `UNWIND` writes. Requires the `neo4j` driver:
`pip install neo4j`.

```python
import kglite

g = kglite.load("graph.kgl")
report = kglite.to_neo4j(
    g,
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    clear=False,    # set True to wipe the target DB first
    merge=False,    # set True for MERGE (upsert) instead of CREATE
    batch_size=5000,
)
# {'nodes_created': ..., 'relationships_created': ..., 'elapsed': ..., 'database': 'neo4j'}
```

Pass `selection_only=True` to export just the current selection (otherwise
the full graph is written). Use `merge=True` for idempotent re-runs against
an existing dataset; `clear=True` for a clean reload.

## Merging Graphs (multi-source ingest)

`extend()` folds one in-memory graph into another in place — the native
alternative to round-tripping through CSV when you build a graph
incrementally from several sources or merge two loaded `.kgl` files.

```python
g1 = kglite.load("source_a.kgl")
g2 = kglite.load("source_b.kgl")

report = g1.extend(g2)              # g2 folded into g1; g2 untouched
report = g1.extend(g2, "preserve")  # on conflict, existing g1 values win
```

Node identity is `(node_type, id)`. The `conflict_handling` argument shares
the `add_nodes` vocabulary — `'update'` (default, *other* wins), `'replace'`,
`'skip'`, `'preserve'` (existing wins), `'sum'` (adds numeric **edge**
properties). Secondary labels are unioned (never removed); edges dedup on
`(connection_type, source, target)` so a merge never silently doubles shared
edges. Scope limits (v1): **in-memory storage only**, and **embeddings are
not merged** — re-run `set_embeddings` / `add_embeddings` after the merge.

## Subgraph Extraction

```python
subgraph = (
    graph.select('Company')
    .where({'title': 'Acme Corp'})
    .expand(hops=2)
    .to_subgraph()
)
subgraph.export('acme_network.graphml', format='graphml')
```

## Embedding Snapshots

Export embeddings to a standalone `.kgle` file so they survive graph rebuilds. Embeddings are keyed by node ID — import resolves IDs against the current graph, skipping any that no longer exist.

```python
# Export all embeddings
stats = graph.export_embeddings("embeddings.kgle")
# {'stores': 2, 'embeddings': 5000}

# Export only specific node types
graph.export_embeddings("embeddings.kgle", ["Article"])

# Export specific (node_type, property) pairs
graph.export_embeddings("embeddings.kgle", {
    "Article": ["summary", "title"],
    "Author": [],                     # all embedding properties for Author
})

# Import into a fresh graph — matches by (node_type, node_id)
graph2 = kglite.KnowledgeGraph()
graph2.add_nodes(articles_df, 'Article', 'id', 'title')
result = graph2.import_embeddings("embeddings.kgle")
# {'stores': 2, 'imported': 4800, 'skipped': 200}
```

## Schema and Indexes

### Schema Definition

```python
graph.define_schema({
    'nodes': {
        'Prospect': {
            'required': ['npdid_prospect', 'prospect_name'],
            'optional': ['prospect_status'],
            'types': {'npdid_prospect': 'integer', 'prospect_name': 'string'}
        }
    },
    'connections': {
        'HAS_ESTIMATE': {'source': 'Prospect', 'target': 'ProspectEstimate'}
    }
})

errors = graph.validate_schema()
schema = graph.schema_text()
```

### Indexes

Two index types:

| Method | Accelerates | Use for |
|--------|-------------|---------|
| `create_index()` | Equality (`= value`) | Exact lookups |
| `create_range_index()` | Range (`>`, `<`, `>=`, `<=`) | Numeric/date filtering |

Both also accelerate Cypher `WHERE` clauses. Composite indexes support multi-property equality.

```python
graph.create_index('Prospect', 'prospect_geoprovince')        # equality index
graph.create_range_index('Person', 'age')                      # B-Tree range index
graph.create_composite_index('Person', ['city', 'age'])        # composite equality

graph.list_indexes()
graph.drop_index('Prospect', 'prospect_geoprovince')
```

Indexes are maintained automatically by all mutation operations.

## Performance Tips

1. **Batch operations** — add nodes/connections in batches, not individually
2. **Specify columns** — only include columns you need to reduce memory
3. **Filter by type first** — `select()` before `filter()` for narrower scans
4. **Create indexes** — on frequently filtered equality conditions (~3x on 100k+ nodes)
5. **Use lightweight methods** — `len()`, `indices()`, `node()` skip property materialization
6. **Cypher LIMIT** — use `LIMIT` to avoid scanning entire result sets

## Threading

The Python GIL is released during heavy Rust operations, allowing other Python threads to run concurrently:

| Operation | GIL Released? | Notes |
|-----------|:---:|-------|
| `save()` | Yes | Serialization + compression + file write |
| `load()` | Yes | File read + decompression + deserialization |
| `cypher()` (reads) | Yes | Query parsing, optimization, and execution |
| `vector_search()` | Yes | Similarity computation (uses rayon internally) |
| `search_text()` | Partial | Model embedding needs GIL; vector search releases it |
| `add_nodes()` | No | DataFrame conversion requires GIL throughout |
| `cypher()` (mutations) | No | Must hold exclusive lock on graph |

## Graph Maintenance

After heavy mutation workloads (DELETE, REMOVE), internal storage accumulates tombstones. Monitor with `graph_info()`.

```python
info = graph.graph_info()
# {'node_count': 950, 'node_capacity': 1000, 'node_tombstones': 50,
#  'edge_count': 2800, 'fragmentation_ratio': 0.05, ...}

if info['fragmentation_ratio'] > 0.3:
    result = graph.vacuum()
    print(f"Reclaimed {result['tombstones_removed']} slots")
```

`vacuum()` rebuilds the graph with contiguous indices and rebuilds all indexes. **Resets the current selection.**

## Common Gotchas

- **One primary type per node.** Secondary labels (multi-label, 0.10.5+) are preserved; `labels(n)` returns a list, primary type first.
- **`id` and `title` are canonical.** `add_nodes(unique_id_field='user_id')` stores the column as `id`. The original name works as an alias.
- **Save files use a pinned binary format.** Compatible across OS/architecture within the same major version.
- **Indexes:** `create_index()` accelerates equality only. For range queries, use `create_range_index()`.
- **Flat vs. grouped results.** After traversal with multiple parents, `titles()` and `collect()` return grouped dicts.
- **No auto-persistence.** The graph lives in memory. `save()` is manual.
