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

All path methods support `connection_types`, `via_types`, and `timeout_ms` for filtering and safety.

### Weighted shortest path

Pass `weight_property` to switch from BFS (hop count) to Dijkstra (sum of edge weights). Edges missing the property default to weight 1.0; negative weights cause the path to be reported as missing.

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

Batch variant for computing many distances at once:

```python
distances = graph.shortest_path_lengths_batch('Person', [(1, 5), (2, 8), (3, 10)])
# → [2, None, 5]  (None where no path exists, same order as input)
```

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
```

## Centrality Algorithms

All centrality methods return a `ResultView` of `{type, title, id, score}` rows, sorted by score descending.

```python
graph.betweenness_centrality(top_k=10)
graph.betweenness_centrality(normalized=True, sample_size=500)
graph.pagerank(top_k=10, damping_factor=0.85)
graph.degree_centrality(top_k=10)
graph.closeness_centrality(top_k=10)

# Alternative output formats
graph.pagerank(as_dict=True)      # → {1: 0.45, 2: 0.32, ...} (keyed by id)
graph.pagerank(to_df=True)        # → DataFrame with type, title, id, score columns
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
