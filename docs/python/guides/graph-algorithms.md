# Graph Algorithms

## Shortest Path

```python
result = graph.shortest_path(source_type='Person', source_id=1, target_type='Person', target_id=100)
if result:
    for node in result["path"]:
        print(f"{node['type']}: {node['title']}")
    print(f"Connections: {result['connections']}")
    print(f"Path length: {result['length']}")
```

Lightweight variants when you don't need full path data:

```python
graph.shortest_path_length(...)    # → int | None (hop count only)
graph.shortest_path_ids(...)       # → list[id] | None (node IDs along path)
graph.shortest_path_indices(...)   # → list[int] | None (raw graph indices, fastest)
```

**Which options each method takes.** Every member of the family accepts the
same traversal controls, so they all answer the same question:

| Method | `connection_types` / `via_types` / `direction` / `timeout_ms` | `weight_property` |
| --- | --- | --- |
| `shortest_path` | yes | yes |
| `shortest_path_ids` | yes | no |
| `shortest_path_indices` | yes | no |
| `shortest_path_length` | yes | yes |
| `shortest_path_lengths_batch` | yes | no |
| `shortest_path_lengths_from` | yes | no |
| `are_connected` | yes | no |
| `all_paths` | yes | no |

**`source_type` / `target_type` / `node_type` are an ID namespace.** They say
which type to look the endpoint id up in — they never restrict which node
types the path may pass through. So this:

```python
graph.shortest_path_length("Person", 3, "Person", 4)
```

happily answers `2` through a `City` node, because two people who live in the
same city are two hops apart. If you meant "how far apart through *people*",
say so:

```python
graph.shortest_path_length("Person", 3, "Person", 4, via_types=["Person"])
# → None
graph.shortest_path_length("Person", 3, "Person", 4, connection_types=["KNOWS"])
# → None
```

`via_types` restricts the *intermediate* nodes only — the two endpoints are
always allowed, whatever their type.

**These methods are undirected by default.** Edges are traversed both ways
whatever direction they were created in, unless you pass `direction`:

```python
graph.shortest_path_length("Person", 1, "Person", 4)                        # 2 (either way)
graph.shortest_path_length("Person", 1, "Person", 4, direction="outgoing")  # 3 (forwards only)
graph.shortest_path_length("Person", 1, "Person", 4, direction="incoming")  # None
```

`direction` accepts `'outgoing'` / `'out'`, `'incoming'` / `'in'`, and
`'any'` / `'both'` / `None` (the default). Anything else raises — it is never
silently ignored. This is the same vocabulary `traverse()` and
`where_connected()` use. Cypher's `shortestPath()` expresses the same thing
with the arrow in the pattern (`(a)-[:KNOWS*..10]->(b)` is directed,
`(a)-[:KNOWS*..10]-(b)` is not); see the `shortestPath()` section of
`CYPHER.md`.

### Weighted shortest path

Pass `weight_property` to switch from BFS (hop count) to Dijkstra (sum of edge weights). Edges missing the property default to weight 1.0; negative weights cause the path to be reported as missing. The weighted search honours `connection_types`, `via_types` and `direction` exactly as the unweighted one does.

```python
# Cheapest path by edge.cost
result = graph.shortest_path(
    "Stop", "A", "Stop", "Z",
    weight_property="cost",
)
# {'path': [...], 'connections': [...], 'length': 3, 'weight': 4.7}

# Length-only variant returns float when weighted, int otherwise
graph.shortest_path_length("Stop", "A", "Stop", "Z", weight_property="cost")  # → 4.7
```

Batch variant for computing many distances at once — it builds the adjacency
once for the whole batch, so it is far cheaper than a loop:

```python
distances = graph.shortest_path_lengths_batch('Person', [(1, 5), (2, 8), (3, 10)])
# → [2, None, 5]  (None where no path exists, same order as input)

# Same filters as the rest of the family
graph.shortest_path_lengths_batch(
    'Person', [(1, 5), (2, 8)],
    connection_types=['KNOWS'], direction='outgoing',
)
```

A pair whose endpoint the filters exclude entirely (a person with no `KNOWS`
edge, under `connection_types=['KNOWS']`) answers `None` — the same "no path"
a disconnected pair gets, never an error.

### One source, many targets

`shortest_path_lengths_from()` walks outward from a single source once and
returns `{node id: hop count}` — the one-to-many shape, where the batch is the
many-to-many one:

```python
# Every Person within 3 hops of Alice.
graph.shortest_path_lengths_from('Person', 1, 'Person', max_hops=3)
# → {1: 0, 2: 1, 5: 2, 4: 2}

# An answer for exactly these ids, None where unreachable.
graph.shortest_path_lengths_from('Person', 1, target_ids=[4, 6])
# → {4: 2, 6: None}
```

The two shapes are deliberately different, and the difference is the whole
contract:

* **With `target_ids`** you get one entry per *requested* id, in the order you
  gave them, and an unreachable target maps to `None`. You asked about it, so
  you get an answer for it.
* **Without `target_ids`** (discovery mode) you get only the nodes actually
  reached. **Absent means unreachable** — or beyond `max_hops`. There are no
  `None` values, because listing every unreached node in the graph is exactly
  the footgun this mode exists to avoid.

For the same reason, at least one of `target_ids`, `target_type` or `max_hops`
is required; an unbounded one-to-all walk is refused with a message naming the
three bounds.

`target_type` filters the **result** (and is the id namespace `target_ids` are
looked up in). It does not restrict the walk — `via_types` does that, and a
node `via_types` excludes is still reported with its own distance while
nothing is reached *through* it, exactly as the pair members exempt their
endpoints.

Unlike the pair members, a `timeout_ms` expiry **raises** here rather than
answering `None`: a dict silently missing its far half is a wrong answer, not
a missing one.

## All Paths

```python
paths = graph.all_paths(
    source_type='Play', source_id=1,
    target_type='Wellbore', target_id=100,
    max_hops=4,
    max_results=100  # Prevent OOM on dense graphs
)
```

## Connected Components

```python
components = graph.connected_components()
# Returns list of lists: [[node_dicts...], [node_dicts...], ...]
print(f"Found {len(components)} connected components")
print(f"Largest component: {len(components[0])} nodes")

graph.are_connected(source_type='Person', source_id=1, target_type='Person', target_id=100)

# True exactly when shortest_path_length() with the same arguments returns a
# distance — so it takes the same filters and direction.
graph.are_connected('Person', 1, 'Person', 100, connection_types=['KNOWS'])
```

## Cypher procedures: scoped subgraph algorithms

Several algorithms are also exposed as Cypher `CALL` procedures so you can
run them over a *subgraph* — one node type and one (or several) relationship
types — instead of the whole graph. This is the idiomatic way to ask
"components among `Person` nodes connected by `KNOWS`" without first
extracting a separate graph.

All three share the same optional `{node_type, relationship}` scoping. Each
field accepts a string or a list of strings; omit the map to run over the
whole graph.

> **Edge-scope key:** `relationship` and `connection_types` are interchangeable
> on every algorithm procedure — the centrality/community procedures historically
> read `connection_types` and the components/k-core ones read `relationship`, but
> either term now works anywhere. **Unknown config keys are rejected** with a
> did-you-mean (`CALL pagerank(): unknown config key 'connection_typ'. Did you
> mean 'connection_types'?`) rather than silently producing an empty result.
> (`where` predicate-scoping is supported by the centrality + community
> procedures; the components/k-core/clustering group scopes by `node_type` +
> `relationship` only.)

### Connected components

```cypher
-- Whole graph
CALL connected_components() YIELD node, component
RETURN component, count(*) AS size ORDER BY size DESC

-- Scoped to one node type + relationship
CALL connected_components({node_type: 'Person', relationship: 'KNOWS'})
YIELD node, component
RETURN component, collect(node.name) AS members

-- Multiple relationship types
CALL connected_components({node_type: ['Person'], relationship: ['KNOWS', 'OWNS']})
YIELD node, component
RETURN count(DISTINCT component) AS num_components
```

### K-core decomposition (coreness)

The *coreness* of a node is the largest `k` for which it survives in the
`k`-core (the maximal subgraph where every node has degree ≥ `k`). High
coreness marks structurally central, resilient nodes. `k_core` and `coreness`
are aliases.

```cypher
CALL k_core() YIELD node, coreness
RETURN node.name AS name, coreness ORDER BY coreness DESC LIMIT 10

-- Scoped
CALL k_core({node_type: 'Person', relationship: 'KNOWS'})
YIELD node, coreness
RETURN coreness, count(*) AS n ORDER BY coreness DESC
```

### Local clustering coefficient

The fraction of a node's neighbour pairs that are themselves connected — the
local triangle-closure rate (0.0 = no neighbours linked, 1.0 = neighbourhood
is a clique). `clustering_coefficient` and `local_clustering_coefficient` are
aliases.

```cypher
CALL clustering_coefficient() YIELD node, coefficient
RETURN node.name AS name, coefficient ORDER BY coefficient DESC

-- Scoped, then averaged
CALL clustering_coefficient({node_type: 'Person', relationship: 'KNOWS'})
YIELD node, coefficient
RETURN avg(coefficient) AS global_avg
```

Scoping is computed lazily over the live graph (no copy), so these run
identically across the in-memory, mapped, and disk storage modes.

## Centrality Algorithms

All centrality methods return a `ResultView` of `{type, title, id, score}` rows, sorted by score descending.

```python
graph.betweenness_centrality(top_k=10)
graph.betweenness_centrality(normalized=True, sample_size=500)
graph.pagerank(top_k=10, damping_factor=0.85)
graph.degree_centrality(top_k=10)
graph.closeness_centrality(top_k=10)

# Alternative output formats
graph.pagerank(to_df=True)        # → DataFrame with type, title, id, score columns

# An {id: score} mapping, when you want one (ids are unique per type only, so
# key by (type, id) whenever the selection spans more than one node type)
scores = {r["id"]: r["score"] for r in graph.pagerank().to_dicts()}
```

## Community Detection

```python
# Louvain modularity optimization (recommended)
result = graph.louvain_communities()
# {'communities': {0: [{type, title, id}, ...], 1: [...]},
#  'modularity': 0.45, 'num_communities': 2}

for comm_id, members in result['communities'].items():
    names = [m['title'] for m in members]
    print(f"Community {comm_id}: {names}")

# With edge weights and resolution tuning
result = graph.louvain_communities(weight_property='strength', resolution=1.5)

# Label propagation (faster, less precise)
result = graph.label_propagation(max_iterations=100)
```

## Clustering

General-purpose clustering via Cypher `CALL cluster()`. Reads nodes from a preceding MATCH clause.

```python
# Spatial DBSCAN — auto-detects lat/lon from set_spatial() config
result = graph.cypher("""
    MATCH (f:Field)
    CALL cluster({method: 'dbscan', eps: 50000, min_points: 2})
    YIELD node, cluster
    RETURN cluster, count(*) AS n, collect(node.name) AS fields
    ORDER BY n DESC
""")

# Property-based K-means — cluster on explicit numeric properties
result = graph.cypher("""
    MATCH (w:Wellbore)
    CALL cluster({
        properties: ['totalDepth', 'bottomHoleTemp'],
        method: 'kmeans', k: 5, normalize: true
    })
    YIELD node, cluster
    RETURN cluster, count(*) AS n
""")
```

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `method` | string | `"dbscan"` | `"dbscan"` or `"kmeans"` |
| `properties` | list | (none) | If omitted, uses spatial config |
| `eps` | float | 0.5 | DBSCAN neighborhood radius (meters for spatial, raw units for properties) |
| `min_points` | int | 3 | DBSCAN minimum neighbors for core point |
| `k` | int | 5 | K-means cluster count |
| `max_iterations` | int | 100 | K-means iteration limit |
| `normalize` | bool | false | Min-max scale features to [0,1] before clustering |

Noise points (DBSCAN only) get `cluster = -1`. Filter with `WHERE cluster >= 0`.

## Analytics

### Statistics

```python
price_stats = graph.select('Product').statistics('price')
unique_cats = graph.select('Product').unique_values(property='category', max_length=10)

# Group by a property — like SQL GROUP BY
graph.select('Person').count(group_by='city')
# → {'Oslo': 42, 'Bergen': 15, 'Trondheim': 8}

graph.select('Person').statistics('age', group_by='city')
# → {'Oslo': {'count': 42, 'mean': 35.2, 'std': 8.1, 'min': 22, 'max': 65, 'sum': 1478},
#    'Bergen': {'count': 15, ...}, ...}
```

### Calculations

```python
graph.select('Product').calculate(expression='price * 1.1', store_as='price_with_tax')

graph.select('User').traverse('PURCHASED').calculate(
    expression='sum(price * quantity)', store_as='total_spent'
)

graph.select('User').traverse('PURCHASED').count(store_as='product_count', group_by_parent=True)
```

### Node Degrees

```python
degrees = graph.select('Person').degrees()
# Returns: {'Alice': 5, 'Bob': 3, ...}
```
