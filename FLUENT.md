# Fluent API Reference

Full fluent (method-chaining) API supported by KGLite. For Cypher queries, see the [Cypher Reference](https://kglite.readthedocs.io/en/latest/reference/cypher-reference.html). For a quick overview, see the [documentation](https://kglite.readthedocs.io).

**Start with:** [Getting Started](https://kglite.readthedocs.io/en/latest/python/getting-started.html) ·
[Data loading](#data-loading) · [Filtering](#selection--filtering) ·
[Traversal](#traversal) · [Retrieval](#data-retrieval) ·
[Algorithms](#graph-algorithms) · [Mutations](#mutation) ·
[Transactions](#transactions) · [Persistence](#persistence) ·
[Introspection](#schema--introspection) ·
[Python API](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html)

> **Selection model:** The fluent API is selection-based. Most methods return a new `KnowledgeGraph` with an updated selection — no data is materialised until you call a retrieval method (`collect()`, `to_df()`, etc.). This makes query chains fast even on large graphs.

---

## Data Loading

DataFrame methods use the optional pandas integration:

```bash
pip install "kglite[pandas]"
```

```python
import kglite
import pandas as pd

graph = kglite.KnowledgeGraph()
people_df = pd.DataFrame({"person_id": [1, 2], "name": ["Alice", "Bob"], "age": [30, 35], "city": ["Oslo", "Bergen"]})
companies_df = pd.DataFrame({"company_id": [10], "name": ["Acme"]})
works_df = pd.DataFrame({"person_id": [1, 2], "company_id": [10, 10]})
df = people_df

# Load nodes from a DataFrame
graph.add_nodes(df, 'Person', 'person_id', 'name')

# With column selection and conflict handling
graph.add_nodes(df, 'Person', 'person_id', 'name',
    columns=['name', 'age', 'city'],
    conflict_handling='update')  # 'update' | 'replace' | 'skip' | 'preserve'

# Spatial columns — declare via column_types
graph.add_nodes(df, 'City', 'city_id', 'name',
    column_types={
        'lat': 'location.lat',
        'lon': 'location.lon',
    })

# Geometry columns (WKT polygons)
graph.add_nodes(df, 'Field', 'field_id', 'name',
    column_types={'wkt_geometry': 'geometry'})

# Named points and shapes
graph.add_nodes(df, 'Pipeline', 'id', 'name',
    column_types={
        'start_lat': 'point.start.lat',
        'start_lon': 'point.start.lon',
        'end_lat': 'point.end.lat',
        'end_lon': 'point.end.lon',
        'route_wkt': 'shape.route',
    })

# Inline timeseries — multiple rows per ID, auto-deduplicated
graph.add_nodes(df, 'Production', 'field_id', 'field_name',
    timeseries={
        'time': 'date',                          # or {'year': 'yr', 'month': 'mo'}
        'channels': ['oil', 'gas', 'condensate'],
        'resolution': 'month',                   # auto-detected if omitted
        'units': {'oil': 'MSm3', 'gas': 'BSm3'},
    })

# Load connections (edges)
graph.add_connections(df, 'WORKS_AT',
    source_type='Person', source_id_field='person_id',
    target_type='Company', target_id_field='company_id')

# Bulk loading — multiple types at once
graph.add_nodes_bulk([
    {'node_type': 'Person', 'unique_id_field': 'id', 'data': people_df},
    {'node_type': 'Company', 'unique_id_field': 'id', 'data': companies_df},
])
graph.add_connections_bulk([
    {'source_type': 'Person', 'target_type': 'Company',
     'connection_name': 'WORKS_AT', 'data': works_df},
])

# Auto-filtering — silently skips connections whose types are not loaded
connection_specs = [{
    'connection_name': 'WORKS_AT', 'source_type': 'Person',
    'target_type': 'Company', 'source_id_field': 'person_id',
    'target_id_field': 'company_id', 'data': works_df,
}]
graph.add_connections_from_source(connection_specs)
```

### Blueprint Loading

```python
# Build from a JSON blueprint + CSVs (round-trips with export_csv)
graph = kglite.from_blueprint("blueprint.json", verbose=True)

# Load from binary file
graph = kglite.load("graph.kgl")
```

---

## Selection & Filtering

### Type Selection

```python
# Select all nodes of a type
people = graph.select('Person')

# With sort and limit
top10 = graph.select('Person', sort='age', limit=10)

# Multi-column sort
graph.select('Person', sort=[('city', True), ('age', False)])  # city ASC, age DESC
```

### Property Filtering

```python
# Exact match
graph.select('Person').where({'city': 'Oslo'})

# Comparison operators
graph.select('Person').where({'age': {'>': 25}})
graph.select('Product').where({'price': {'>=': 100, '<=': 500}})

# String predicates
graph.select('Person').where({'name': {'contains': 'ali'}})
graph.select('Person').where({'name': {'starts_with': 'A'}})
graph.select('Person').where({'email': {'ends_with': '@example.com'}})
graph.select('Person').where({'name': {'regex': '^A.*'}})

# Negated variants
graph.select('Person').where({'status': {'not_in': ['inactive', 'banned']}})
graph.select('Person').where({'name': {'not_contains': 'test'}})

# IN list
graph.select('Person').where({'city': {'in': ['Oslo', 'Bergen']}})

# Null checks
graph.select('Person').where({'email': {'is_not_null': True}})
graph.select('Person').where({'nickname': {'is_null': True}})

# Combined conditions (AND logic within a single dict)
graph.select('Person').where({
    'age': {'>': 25},
    'city': 'Oslo',
    'name': {'regex': '^A.*'},
})
```

### OR Filtering

```python
# OR logic across condition sets
graph.select('Person').where_any([
    {'city': 'Oslo'},
    {'city': 'Bergen'},
    {'age': {'>': 60}},
])
```

### Connection-Based Filtering

```python
# Keep only nodes that have a KNOWS connection
graph.select('Person').where_connected('KNOWS')

# Direction-specific
graph.select('Person').where_connected('KNOWS', direction='outgoing')
graph.select('Person').where_connected('KNOWS', direction='incoming')

# Orphan filtering
graph.select('Person').where_orphans(include_orphans=True)   # only disconnected nodes
graph.select('Person').where_orphans(include_orphans=False)  # only connected nodes
```

### Sorting & Pagination

```python
# Sort
graph.select('Person').sort('age')
graph.select('Person').sort('age', ascending=False)
graph.select('Person').sort([('city', True), ('age', False)])

# Limit
graph.select('Person').limit(100)

# Pagination (skip + limit)
graph.select('Person').sort('name').offset(20).limit(10)  # page 3 of 10
```

---

## Temporal Filtering

Date-range filtering on node properties. NULL semantics: NULL `from` = valid since beginning, NULL `to` = still valid.

```python
# Nodes valid at a specific date
graph.select('Employee').valid_at('2024-01-15')
# Uses default fields: date_from, date_to

# Custom field names
graph.select('Contract').valid_at('2024-06-01',
    date_from_field='start_date',
    date_to_field='end_date')

# Nodes valid during a range (overlap check)
graph.select('Regulation').valid_during('2020-01-01', '2022-12-31',
    date_from_field='effective_from',
    date_to_field='effective_to')
```

---

## Spatial Filtering

### Point-Based Filters

```python
# Bounding box
graph.select('City').within_bounds(
    min_lat=59.0, max_lat=61.0,
    min_lon=10.0, max_lon=12.0)

# With custom field names
graph.select('City').within_bounds(59.0, 61.0, 10.0, 12.0,
    lat_field='lat', lon_field='lon')

# Distance filter (degrees — fast, approximate)
graph.select('City').near_point(59.91, 10.75, max_distance=1.0)

# Distance filter (meters — geodesic, WGS84)
graph.select('City').near_point_m(59.91, 10.75, max_distance_m=100_000)
# Falls back to geometry centroid when lat/lon fields are missing
```

### Geometry Filters (WKT)

```python
# Point-in-polygon: which fields contain a point?
graph.select('Field').contains_point(60.5, 3.5)

# Custom geometry field
graph.select('Field').contains_point(60.5, 3.5, geometry_field='wkt_geometry')

# Geometry intersection: which fields overlap a query polygon?
graph.select('Field').intersects_geometry(
    'POLYGON((3.0 60.0, 4.0 60.0, 4.0 61.0, 3.0 61.0, 3.0 60.0))')

# Also accepts shapely geometry objects
from shapely.geometry import box
graph.select('Field').intersects_geometry(box(3.0, 60.0, 4.0, 61.0))
```

### Spatial Configuration

```python
# Declare spatial properties (alternative to column_types in add_nodes)
graph.set_spatial('City',
    location=('latitude', 'longitude'))

graph.set_spatial('Field',
    geometry='wkt_geometry')

# Named points and shapes
graph.set_spatial('Pipeline',
    points={'start': ('start_lat', 'start_lon'), 'end': ('end_lat', 'end_lon')},
    shapes={'route': 'route_wkt'})

# Query spatial config
graph.spatial('City')     # config for one type
graph.spatial()           # all types
```

### Spatial Aggregations

```python
# Geographic bounds of selection
bounds = graph.select('City').bounds()
# {'min_lat': 58.1, 'max_lat': 71.1, 'min_lon': 5.3, 'max_lon': 31.0}

# As shapely polygon
poly = graph.select('City').bounds(as_shapely=True)

# Centroid (average lat/lon)
center = graph.select('City').centroid()
# {'latitude': 63.4, 'longitude': 10.4}

# WKT centroid (static method — does not require selection)
graph.wkt_centroid('POLYGON((3 60, 4 60, 4 61, 3 61, 3 60))')
# {'latitude': 60.5, 'longitude': 3.5}
```

---

## Timeseries

### Configuration

```python
# Declare timeseries metadata for a node type
graph.set_timeseries('Sensor',
    resolution='day',
    channels=['temperature', 'pressure'],
    units={'temperature': '°C', 'pressure': 'bar'},
    bin_type='sample')  # 'total' | 'mean' | 'sample'

graph.timeseries_config('Sensor')
graph.timeseries_config()  # all types
```

### Bulk Loading from DataFrame

```python
# Bulk load timeseries from a DataFrame
graph.add_timeseries('Field',
    data=production_df,
    fk='field_id',                       # foreign key → node ID
    time_key=['year', 'month'],          # or ['date'] for date strings
    channels=['oil', 'gas', 'condensate'],
    resolution='month',                  # auto-detected if omitted
    units={'oil': 'MSm3'})
```

### Manual Loading

```python
# Set time index for a node
graph.set_time_index('sensor_1', ['2024-01-01', '2024-01-02', '2024-01-03'])

# Add channel data (length must match time index)
graph.add_ts_channel('sensor_1', 'temperature', [20.1, 21.3, 19.8])
graph.add_ts_channel('sensor_1', 'pressure', [1.01, 1.02, 0.99])
```

### Retrieval

```python
# Get all channels for a node
data = graph.timeseries('sensor_1')
# {'keys': ['2024-01-01', '2024-01-02', ...],
#  'channels': {'temperature': [20.1, 21.3, ...], 'pressure': [1.01, ...]}}

# Single channel
data = graph.timeseries('sensor_1', channel='temperature')
# {'keys': ['2024-01-01', ...], 'values': [20.1, ...]}

# Date range
data = graph.timeseries('sensor_1', channel='temperature',
    start='2024-01-01', end='2024-01-02')

# Just the time index
keys = graph.time_index('sensor_1')
# ['2024-01-01', '2024-01-02', '2024-01-03']
```

### Timeseries Aggregation via Cypher

The fluent API provides data loading and extraction. For aggregation (`ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_at`, `ts_delta`, `ts_series`, etc.), use Cypher — see the [Cypher Reference](https://kglite.readthedocs.io/en/latest/reference/cypher-reference.html#timeseries-functions).

```python
# Example: top producers in 2020
graph.cypher("""
    MATCH (f:Field)
    RETURN f.title, ts_sum(f.oil, '2020') AS prod
    ORDER BY prod DESC LIMIT 10
""")
```

---

## Embedding / Vector Search

### Setup

```python
# Register an embedding model (must have .dimension and .embed())
graph.set_embedder(my_model)

# Compute embeddings for a text column
graph.embed_texts('Article', 'summary', batch_size=256, show_progress=True)
# mode='missing' (default): only nodes without an embedding yet.
# mode='changed':            re-embed nodes whose text changed since last time
#                            (a per-node content hash is stored to detect this).
# mode='all' (= replace=True): re-embed every node, rebuilding the store.
graph.embed_texts('Article', 'summary', mode='changed')

# Inspect a store's provenance (dimension, count, model id, metric, #hashed):
graph.embedding_info('Article', 'summary')
# -> {'dimension': 1024, 'count': 5000, 'model': 'BAAI/bge-m3', 'metric': 'cosine', 'hashed': 5000}
# `model` is populated when the embedder exposes a `model_id`/`model_name` attribute.

# Carry vectors across a fresh rebuild (the disposable-cache workflow) in one call,
# matched by node id — instead of snapshot → add_embeddings → embed_texts:
new_graph.copy_embeddings_from(old_graph)

# Or provide pre-computed embeddings
graph.set_embeddings('Article', 'summary', {
    'article_1': [0.1, 0.2, ...],
    'article_2': [0.3, 0.4, ...],
})

# Store embeddings with an intended metric (becomes the default at query time)
graph.set_embeddings('Concept', 'title', poincare_vectors, metric='poincare')
```

### Text Search (auto-embeds query)

```python
# Search using a text query — auto-embeds via set_embedder()
results = graph.select('Article').search_text(
    'summary', 'machine learning advances', top_k=10)

# With explicit metric: 'cosine', 'dot_product', 'euclidean', 'poincare'
results = graph.select('Article').search_text(
    'summary', 'climate change', top_k=5, metric='dot_product')

# As DataFrame
df = graph.select('Article').search_text(
    'summary', 'AI safety', top_k=10, to_df=True)

# By default each hit carries id, title, type, score AND every node property
# (read live — no follow-up query needed). Trim the payload with returning=:
ranked = graph.select('Article').search_text(
    'summary', 'AI safety', top_k=50, returning=['title'])   # -> id, score, title only
```

### Vector Search (pre-computed query vector)

```python
# Search with an explicit vector
results = graph.select('Article').vector_search(
    'summary', query_vector=[0.1, 0.2, ...], top_k=10)

# Combine with property filters
results = (graph
    .select('Article')
    .where({'category': 'politics'})
    .vector_search('summary', query_vec, top_k=10, metric='cosine'))

# Metrics: 'cosine' (default), 'dot_product', 'euclidean', 'poincare'
# If a stored metric was set via set_embeddings(..., metric=), it is used as default
# `returning=[...]` trims each hit to id + score + the named fields (see search_text).
```

### Index for scale (HNSW)

By default search is an **exact** brute-force scan. For large corpora, build an
opt-in HNSW approximate-nearest-neighbour index (like `create_index`); once built
it's used automatically for whole-corpus queries, and `exact=True` forces the
exact scan.

```python
graph.build_vector_index('Article', 'summary')           # opt in (persists in .kgl)
graph.build_vector_index('Article', 'summary',
                         m=16, ef_construction=200, ef_search=64, metric='cosine')

# auto-used for whole-corpus queries on large stores:
graph.select('Article').search_text('summary', 'AI', top_k=10)
# force exact (guaranteed-exact results):
graph.select('Article').vector_search('summary', query_vec, top_k=10, exact=True)

graph.has_vector_index('Article', 'summary')   # -> True
graph.drop_vector_index('Article', 'summary')   # revert to exact
```

- Auto-use applies to whole-corpus queries (≥256 candidates); a selective
  `.where(...)` falls back to an exact scan automatically.
- cosine / dot_product / euclidean are indexable; `poincare` always uses the
  exact path. Recall depends on data + `ef_search` (raise it for higher recall).
- The index is **dropped automatically** when the store's vectors change
  (`add_embeddings`, `embed_texts`) or slots are remapped (`vacuum`) — rebuild
  after. It **persists in the `.kgl`** (and `to_bytes()`).
- The Cypher `text_score()` / `vector_score()` whole-corpus top-k
  (`... ORDER BY score DESC LIMIT k`) also auto-uses the index; a heavily-
  filtered Cypher query stays exact.

### Semantic Search via Cypher

```python
# text_score() in Cypher queries (requires set_embedder)
graph.cypher("""
    MATCH (n:Article)
    RETURN n.title, text_score(n, 'summary', 'machine learning') AS score
    ORDER BY score DESC LIMIT 10
""")

# With threshold in WHERE
graph.cypher("""
    MATCH (n:Article)
    WHERE text_score(n, 'summary', $query) > 0.8
    RETURN n.title
""", params={'query': 'artificial intelligence'})

# embedding_norm() — L2 norm (hierarchy depth in Poincaré space)
graph.cypher("""
    MATCH (n:Concept)
    RETURN n.name, embedding_norm(n, 'title') AS depth
    ORDER BY depth ASC LIMIT 10
""")
```

### Embedding Management

```python
# List all embedding stores
graph.list_embeddings()
# [{'node_type': 'Article', 'text_column': 'summary', 'dimension': 384, 'count': 1000, 'metric': None}]

# Retrieve all embeddings
vecs = graph.embeddings('Article', 'summary')       # by type
vecs = graph.select('Article').embeddings('summary')  # from selection

# Single embedding
vec = graph.embedding('Article', 'summary', 'article_1')

# Remove an embedding store
graph.remove_embeddings('Article', 'summary')
```

---

## Traversal

```python
# Follow outgoing connections
graph.select('Person').traverse('WORKS_AT')

# Direction control
graph.select('Person').traverse('KNOWS', direction='incoming')

# Filter to specific target node type (when a connection goes to multiple types)
graph.select('Field').traverse('OF_FIELD', direction='incoming',
    target_type='ProductionProfile')

# Multiple target types
graph.select('Field').traverse('OF_FIELD', direction='incoming',
    target_type=['ProductionProfile', 'FieldReserves'])

# Filter target nodes by properties
graph.select('Person').traverse('WORKS_AT',
    where={'city': 'Oslo'})

# Filter edge properties
graph.select('Person').traverse('RATED',
    where_connection={'score': {'>': 4}})

# Sort and limit targets
graph.select('Person').traverse('KNOWS',
    sort_target='name', limit=5)

# Multi-hop traversal
companies = (graph
    .select('Person')
    .where({'city': 'Oslo'})
    .traverse('WORKS_AT')
    .traverse('LOCATED_IN'))

# Combine target_type + where + temporal
graph.select('Field').traverse('OF_FIELD', direction='incoming',
    target_type='Wellbore', where={'wlbTotalDepth': {'>': 5000}})
```

### Comparison Operations (`compare()`)

`compare()` finds related nodes by spatial proximity, semantic similarity, or
clustering — without needing explicit graph edges.

`method` accepts a **string** for simple cases or a **dict** with method-specific settings:

```python
method='contains'                                        # string shorthand
method={'type': 'contains', 'resolve': 'geometry'}       # dict with settings
```

#### Spatial Containment

```python
# Find all wells within each structural element's geometry
# Default: target resolved via location fields → geometry centroid fallback
graph.select('Structure').compare('Well', 'contains')

# Force polygon-in-polygon containment (target as full geometry)
graph.select('Structure').compare('Field',
    {'type': 'contains', 'resolve': 'geometry'})

# Force geometry centroid (even if target has location fields)
graph.select('Structure').compare('Well',
    {'type': 'contains', 'resolve': 'centroid'})

# Override geometry field name
graph.select('Zone').compare('Well',
    {'type': 'contains', 'geometry': 'wkt_geometry'})
```

#### Spatial Intersection

```python
# Find licences whose geometry overlaps each field (always geometry-to-geometry)
graph.select('Field').compare('Licence', 'intersects')

# With custom geometry field
graph.select('Field').compare('Licence',
    {'type': 'intersects', 'geometry': 'wkt_field'})
```

#### Distance

```python
# Find wells within 5 km of each platform (point-to-point, default resolution)
graph.select('Platform').compare('Well',
    {'type': 'distance', 'max_m': 5000})

# Force geometry centroid (even if nodes have location fields)
graph.select('Structure').compare('Well',
    {'type': 'distance', 'max_m': 5000, 'resolve': 'centroid'})

# Closest boundary point (min edge-to-edge distance)
graph.select('Structure').compare('Well',
    {'type': 'distance', 'max_m': 5000, 'resolve': 'closest'})

# With filter and limit
graph.select('Platform').compare('Well',
    {'type': 'distance', 'max_m': 10000},
    filter={'status': 'active'}, limit=20)
```

#### Semantic Similarity

```python
# Find articles with similar abstracts (cosine > 0.85)
graph.select('Article').compare('Article',
    {'type': 'text_score', 'property': 'abstract', 'threshold': 0.85},
    limit=5)

# Different similarity metric
graph.select('Doc').compare('Doc',
    {'type': 'text_score', 'property': 'summary',
     'threshold': 0.7, 'metric': 'dot_product'})
```

#### Clustering

```python
# Group wells into clusters by location
graph.select('Well').compare('Well',
    {'type': 'cluster', 'algorithm': 'kmeans', 'k': 5,
     'features': ['latitude', 'longitude']})

# DBSCAN with distance threshold
graph.select('Well').compare('Well',
    {'type': 'cluster', 'algorithm': 'dbscan',
     'eps': 5000, 'min_samples': 3,
     'features': ['latitude', 'longitude']})

# Chain: per-cluster statistics
graph.select('Well').compare('Well',
    {'type': 'cluster', 'algorithm': 'kmeans', 'k': 10,
     'features': ['latitude', 'longitude', 'depth']}) \
    .statistics('production')
```

#### Resolve modes

The `resolve` key controls how polygon geometries are spatially interpreted.
When omitted, the default is: **location fields → geometry centroid fallback**.

| `resolve` | Behavior | Use case |
|-----------|----------|----------|
| *(omitted)* | location lat/lon → geometry centroid fallback | Default |
| `'centroid'` | Geometry centroid (skips location fields) | Force use of geometry |
| `'closest'` | Nearest point on geometry boundary | Min boundary-to-boundary distance |
| `'geometry'` | Full polygon shape | Polygon-in-polygon containment |

### Add Properties from Traversal Chain

After any traversal (edge-based or comparison-based), `add_properties()` enriches
the selected (leaf) nodes with properties from ancestor nodes in the hierarchy.

Use the `Agg` and `Spatial` helper classes for discoverable autocomplete:

```python
from kglite import Agg, Spatial
```

```python
# Copy properties from parent type
graph.select('Structure').compare('Well', 'contains') \
    .add_properties({'Structure': ['name', 'status']})

# Copy all properties
graph.select('Structure').compare('Well', 'contains') \
    .add_properties({'Structure': []})

# Rename properties
graph.select('Structure').compare('Well', 'contains') \
    .add_properties({'Structure': {'struct_name': 'name', 'struct_status': 'status'}})

# Aggregate with Agg helpers
graph.select('Structure').compare('Well', 'contains') \
    .add_properties({'Well': {
        'well_count': Agg.count(),
        'avg_depth': Agg.mean('depth'),
        'max_depth': Agg.max('depth'),
        'total_prod': Agg.sum('production'),
    }})

# Spatial compute with Spatial helpers
graph.select('Structure').compare('Well', 'contains') \
    .add_properties({'Structure': {
        'dist_to_center': Spatial.distance(),
        'parent_area': Spatial.area(),
        'parent_perimeter': Spatial.perimeter(),
    }})

# Combined: rename + spatial in one call
graph.select('Structure').compare('Well', 'contains') \
    .add_properties({
        'Structure': {
            'struct_name': 'name',
            'struct_area': Spatial.area(),
            'dist_to_center': Spatial.distance(),
        },
    })

# Copy from intermediate node in A → B → C chain
graph.select('A').traverse('REL_AB').traverse('REL_BC') \
    .add_properties({'B': ['score']})
```

**Aggregate helpers (`Agg`):** `count()`, `sum(prop)`, `mean(prop)`, `min(prop)`, `max(prop)`, `std(prop)`, `collect(prop)`

**Spatial helpers (`Spatial`):** `distance()`, `area()`, `perimeter()`, `centroid_lat()`, `centroid_lon()`

> **Note:** The raw string forms (`'count(*)'`, `'mean(depth)'`, `'distance'`, etc.) still work — the helpers simply return those strings.

### Breadth-First Expansion

```python
# Expand selection by N hops (undirected)
expanded = graph.select('Person').where({'name': 'Alice'}).expand(hops=2)
```

### Create Connections from Traversal

```python
# After A → B → C traversal, create direct A → C edges
graph.select('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C')

# Copy properties from intermediate B nodes onto the new edges
graph.select('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C', properties={'B': ['score', 'weight']})

# Empty list = copy ALL properties from that type
graph.select('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C', properties={'B': []})

# Copy from multiple node types
graph.select('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C', properties={'A': ['name'], 'B': ['score']})

# Override source/target (connect B → C instead of A → C)
graph.select('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('B_TO_C', source_type='B', target_type='C')

# Conflict handling
graph.select('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C', conflict_handling='skip')
```

---

## Data Retrieval

### Nodes

```python
# Flat ResultView (lazy) — always returns ResultView
result = graph.select('Person').where({'age': {'>': 25}}).collect()
for node in result:
    print(node['title'], node['age'])

# ResultView supports indexing, len(), bool()
print(len(result))
print(result[0])

# As list of dicts (full materialisation)
nodes = result.to_list()

# Grouped by parent type — always returns dict
grouped = graph.select('Field').traverse('HAS_WELL') \
    .collect_grouped('Field')
# → {'TROLL': [...], 'EKOFISK': [...]}

# Include parent metadata in grouped output
grouped = graph.select('Field').traverse('HAS_WELL') \
    .collect_grouped('Field', parent_info=True)

# Lightweight: id + title + type only
ids = graph.select('Person').ids()

# O(1) lookup by type + ID
person = graph.node('Person', 'alice')

# Titles only
titles = graph.select('Person').titles()

# Specific properties as tuples
props = graph.select('Person').get_properties(['name', 'age'])
# [('Alice', 30), ('Bob', 25)]
```

### Count & Indices

```python
# Count without materialising (O(1))
n = graph.select('Person').len()

# Raw graph indices
idx = graph.select('Person').indices()
```

### DataFrame Export

```python
# Current selection as DataFrame
df = graph.select('Person').to_df()

# Without type/id columns
df = graph.select('Person').to_df(include_type=False, include_id=False)

# GeoDataFrame (from ResultView)
result = graph.cypher("MATCH (n:Field) RETURN n.name, n.wkt_geometry AS geometry")
gdf = result.to_gdf(geometry_column='geometry', crs='EPSG:4326')
```

### Human-Readable String Export

`to_str(limit=50)` formats the selection as a multi-line string — each node as a `[Type] title (id: x)` block with indented properties, one per line. Useful for quick inspection in a REPL or for logging.

```python
print(graph.select('Person').to_str(limit=5))
# [Person] Alice (id: 1)
#     age: 30
#     city: Oslo
# [Person] Bob (id: 2)
#     age: 35
#     city: Bergen
# ...
```

`show(columns=['id', 'title'], limit=200)` is the more compact alternative — one node per line as `Type(val1, val2)`, traversal-aware.

### ResultView

All `cypher()` calls, `collect()` (flat), centrality methods, and `sample()` return a `ResultView`:

```python
result = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")

len(result)           # row count (O(1))
bool(result)          # True if non-empty
result[0]             # single row as dict
result.columns        # ['n.name', 'n.age']
result.head(5)        # first 5 rows as new ResultView
result.tail(5)        # last 5 rows
result.to_list()      # list[dict] (full conversion)
result.to_df()        # pandas DataFrame
result.to_gdf()       # GeoDataFrame (with WKT geometry column)
result.stats          # mutation stats (CREATE/SET/DELETE only)
result.profile        # PROFILE stats (PROFILE queries only)

for row in result:
    print(row)
```

---

## Statistics & Calculations

```python
# Descriptive statistics for a numeric property
stats = graph.select('Person').statistics('age')
# {count, mean, std, min, max, sum}

# Group by a property
stats = graph.select('Person').statistics('age', group_by='city')
# {'Oslo': {count, mean, ...}, 'Bergen': {count, mean, ...}}

# Count nodes
n = graph.select('Person').count()

# Count grouped by property
counts = graph.select('Person').count(group_by='city')
# {'Oslo': 150, 'Bergen': 80}

# Math expressions
graph.select('Product').calculate('price * quantity')

# Aggregate functions
graph.select('Person').calculate('mean(age)')

# Store results as new properties
graph.select('Product').calculate('price * 1.25', store_as='price_with_tax')
```

### Unique Values

```python
cities = graph.select('Person').unique_values('city')

# Store as comma-separated list on parent nodes
graph.select('Company').traverse('EMPLOYS').unique_values(
    'skill', store_as='employee_skills')
```

### Children Properties to List

```python
# Collect child titles into comma-separated strings on parents
graph.select('Company').traverse('EMPLOYS').collect_children(
    property='name', sort='name', store_as='employees')
```

---

## Graph Algorithms

### Path Finding

```python
# Shortest path
path = graph.shortest_path('Person', 'alice', 'Person', 'dave')
# {'path': [{id, title, type}, ...], 'connections': ['KNOWS', 'KNOWS'], 'length': 2}

# With edge type and node type filters
path = graph.shortest_path('Person', 'alice', 'Person', 'dave',
    connection_types=['KNOWS', 'WORKS_WITH'],
    via_types=['Person'],
    timeout_ms=5000)

# Just the hop count (faster)
dist = graph.shortest_path_length('Person', 'alice', 'Person', 'dave')

# Just the IDs
ids = graph.shortest_path_ids('Person', 'alice', 'Person', 'dave')

# Just raw indices (fastest)
indices = graph.shortest_path_indices('Person', 'alice', 'Person', 'dave')

# All paths up to max hops
paths = graph.all_paths('Person', 'alice', 'Person', 'dave',
    max_hops=5, max_results=100, timeout_ms=10000)

# Weighted shortest path (Dijkstra) — pass weight_property to minimise
# total edge weight rather than hop count. Edges missing the property
# default to 1.0; negative weights cause the path to be reported as
# missing.
path = graph.shortest_path('Stop', 'A', 'Stop', 'Z', weight_property='cost')
# {'path': [...], 'connections': [...], 'length': 3, 'weight': 4.7}

# Length-only variant returns float when weighted, int otherwise
graph.shortest_path_length('Stop', 'A', 'Stop', 'Z', weight_property='cost')  # → 4.7
```

### Connectivity

```python
# Boolean connectivity check
connected = graph.are_connected('Person', 'alice', 'Person', 'dave')

# Connected components
components = graph.connected_components(weak=True)   # weakly connected (default)
components = graph.connected_components(weak=False)  # strongly connected
# Returns list of components (largest first)

# Degree counts for selected nodes
degrees = graph.select('Person').degrees()
# {'Alice': 5, 'Bob': 3, ...}
```

### Centrality

All centrality methods return `ResultView` by default, with optional `as_dict` or `to_df` output:

```python
# Betweenness centrality
result = graph.betweenness_centrality(top_k=10)
for row in result:
    print(row['title'], row['score'])

# As DataFrame
df = graph.betweenness_centrality(top_k=10, to_df=True)

# With sampling for large graphs
result = graph.betweenness_centrality(sample_size=100, timeout_ms=5000)

# PageRank
result = graph.pagerank(damping_factor=0.85, top_k=10)

# Degree centrality
result = graph.degree_centrality(normalized=True, top_k=10)

# Closeness centrality
result = graph.closeness_centrality(top_k=10)

# As dict
scores = graph.pagerank(as_dict=True)
# {'alice': 0.15, 'bob': 0.12, ...}
```

### Community Detection

```python
# Louvain
result = graph.louvain_communities(resolution=1.0)
# {'communities': {0: [{id, title, type}, ...], 1: [...]},
#  'modularity': 0.45, 'num_communities': 3}

# With edge type filtering and weights
result = graph.louvain_communities(
    weight_property='strength',
    connection_types=['KNOWS'],
    timeout_ms=10000)

# Label propagation
result = graph.label_propagation(max_iterations=100)
```

---

## Set Operations

Combine selections from different query chains on the same graph:

```python
young = graph.select('Person').where({'age': {'<': 25}})
oslo = graph.select('Person').where({'city': 'Oslo'})

# Union — nodes in either selection
young.union(oslo).collect()

# Intersection — nodes in both selections
young.intersection(oslo).collect()

# Difference — in young but not in oslo
young.difference(oslo).collect()

# Symmetric difference — in exactly one selection
young.symmetric_difference(oslo).collect()
```

---

## Mutation

### Fluent Update

```python
# Batch-update all selected nodes
result = graph.select('Person').where({'city': 'Oslo'}).update({
    'region': 'Eastern Norway',
    'updated': True,
})
print(result['nodes_updated'])
```

### Cypher Mutations

For CREATE, SET, DELETE, REMOVE, and MERGE — see the [Cypher Reference](https://kglite.readthedocs.io/en/latest/reference/cypher-reference.html#create--set--delete--remove--merge).

---

## Subgraph Extraction

```python
# Extract selected nodes + their inter-edges into a new independent graph
sub = graph.select('Person').where({'city': 'Oslo'}).to_subgraph()

# Preview what would be extracted
stats = graph.select('Person').expand(2).subgraph_stats()
# {'node_count': 42, 'edge_count': 78, 'node_types': ['Person', 'Company'], ...}
```

---

## Pattern Matching

```python
# Cypher-like pattern matching without full Cypher
matches = graph.match_pattern('(a:Person)-[:KNOWS]->(b:Person)', max_matches=100)
# [{'a': {id, title, type, ...}, 'b': {id, title, type, ...}}, ...]

# Undirected and incoming
graph.match_pattern('(a:Person)-[:KNOWS]-(b:Person)')
graph.match_pattern('(a:Person)<-[:KNOWS]-(b:Person)')

# With inline properties
graph.match_pattern("(a:Person {city: 'Oslo'})-[:KNOWS]->(b:Person)")
```

---

## Indexes

### Equality Indexes

```python
graph.create_index('Person', 'city')
graph.has_index('Person', 'city')        # True
graph.index_stats('Person', 'city')      # {type, property, unique_values}
graph.list_indexes()                     # [{type, property}, ...]
graph.drop_index('Person', 'city')
graph.rebuild_indexes()                  # rebuild all
```

### Range Indexes (B-Tree)

```python
# Fast >, >=, <, <=, BETWEEN queries
graph.create_range_index('Person', 'age')
graph.drop_range_index('Person', 'age')
```

### Composite Indexes

```python
graph.create_composite_index('Person', ['city', 'age'])
graph.has_composite_index('Person', ['city', 'age'])
graph.composite_index_stats('Person', ['city', 'age'])
graph.list_composite_indexes()
graph.drop_composite_index('Person', ['city', 'age'])
```

### Unified Index View

```python
graph.indexes()
# [{'node_type': 'Person', 'property': 'city', 'type': 'equality'},
#  {'node_type': 'Person', 'properties': ['city', 'age'], 'type': 'composite'}]
```

---

## Transactions

```python
# Read-write transaction (snapshot isolation + optimistic concurrency)
with graph.begin() as tx:
    tx.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    tx.cypher("CREATE (:Person {name: 'Bob', age: 25})")
    tx.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    # Auto-commits on success, auto-rolls-back on exception

# Manual control
tx = graph.begin()
tx.cypher("CREATE (:Person {name: 'Charlie'})")
tx.commit()   # or tx.rollback()

# Read-only transaction (O(1) cost, zero memory overhead)
with graph.begin_read() as tx:
    result = tx.cypher("MATCH (n:Person) RETURN n.name")
    # Mutations rejected with RuntimeError
```

---

## Export

```python
# File export (format inferred from extension)
graph.export('graph.graphml')              # GraphML
graph.export('graph.gexf')                 # GEXF
graph.export('graph.json')                 # D3 JSON
graph.export('graph.csv')                  # CSV

# Selection-only export
graph.select('Person').export('people.graphml', selection_only=True)

# String export (no file)
xml = graph.export_string('graphml')

# CSV directory tree with blueprint (round-trip with from_blueprint)
graph.export_csv('output/')
# output/
# ├── nodes/
# │   ├── Person.csv
# │   └── Company.csv
# ├── connections/
# │   └── WORKS_AT.csv
# └── blueprint.json
```

---

## Persistence

```python
graph.save('graph.kgl')          # atomic (temp + rename) + fsync by default —
                                 # a crash mid-save can't leave a torn .kgl
graph.save('graph.kgl', fsync=False)   # skip the flush for speed (still atomic)
graph = kglite.load('graph.kgl')

# In-memory serialization (no filesystem path) — for object storage, a pipe,
# a checksum, or a custom write:
blob = graph.to_bytes()
graph = kglite.from_bytes(blob)
```

A corrupt / truncated / wrong-format file raises a typed, classifiable
`kglite.FileFormatError` (a `kglite.KgError`); a missing file raises
`kglite.FileError` — so a consumer can branch "corrupt → rebuild" vs
"missing → create" without a broad `except`.

---

## Concurrency

A `KnowledgeGraph` is **single-owner**: it is not safe to share one instance
across threads while any thread mutates it (touching a shared graph mid-mutation
raises a clear `RuntimeError`, not a crash). Three safe patterns:

```python
# Per-worker: give each thread its own graph (copy() is cheap; builds are fast)
worker_graph = graph.copy()

# Shared reads: freeze() → an immutable snapshot, lock-free for concurrent readers
snapshot = graph.freeze()        # O(1) — shares data, no deep copy
snapshot.cypher('MATCH (n:Doc) RETURN count(n)')   # safe from many threads at once
# Mutating the source afterwards leaves the snapshot unchanged (copy-on-write).
# Build → freeze → serve readers → swap in a new freeze() when data changes.

# Shared reads AND writes: session() → a thread-safe handle
store = graph.session()          # or kglite.open_session('graph.kgl')
store.execute('CREATE (n:Doc {id: 1})')   # serialized write, composes (no lost updates)
store.cypher('MATCH (n:Doc) RETURN count(n)')  # reads stay lock-free
store.cursor().select('Doc').where({'team': 'A'}).to_df()  # per-thread fluent chains
```

`FrozenGraph` is read-only — mutations (`CREATE`/`SET`/`DELETE`/`REMOVE`/`MERGE`)
raise; semantic search works via `text_score()`/`vector_score()` in `cypher()`.
For the full model — write composition, snapshot isolation, cost — see
[concurrency guide](https://github.com/kkollsga/kglite/blob/main/docs/concepts/concurrency.md).

---

## Schema & Introspection

```python
# Text summary
print(graph.schema_text())

# Full schema dict
schema = graph.schema()
# {'node_types': {'Person': {count, properties}}, 'connection_types': {...},
#  'indexes': [...], 'node_count': 1000, 'edge_count': 2000}

# Node type counts
graph.node_type_counts()
# {'Person': 500, 'Company': 100}

# Property statistics for a type
graph.properties('Person', max_values=20)
# {'age': {'type': 'int', 'non_null': 500, 'unique': 50, 'values': [...]}, ...}

# Connection topology
graph.neighbors_schema('Person')
# {'outgoing': [{'connection_type': 'KNOWS', 'target_type': 'Person', 'count': 800}],
#  'incoming': [...]}

# Connection types with counts
graph.connection_types()

# Quick sample
graph.sample('Person', n=5)

# Selection state
print(graph.selection())
graph.clear()  # reset selection

# Execution plan for current chain
print(graph.select('Person').where({'age': {'>': 25}}).explain())
# SELECT Person (500 nodes) -> WHERE (42 nodes)
```

### Schema Definition & Validation

```python
graph.define_schema({
    'nodes': {
        'Person': {
            'properties': {'name': 'string', 'age': 'integer'},
        },
    },
    'connections': {
        'KNOWS': {'source': 'Person', 'target': 'Person'},
    },
})

errors = graph.validate_schema(strict=True)
graph.has_schema()       # True
graph.clear_schema()
```

### AI Agent Introspection

```python
# XML description for AI agents (progressive disclosure)
print(graph.describe())                              # inventory overview
print(graph.describe(types=['Field', 'Well']))        # focused detail
print(graph.describe(connections=True))               # all connection types
print(graph.describe(connections=['BELONGS_TO']))      # deep-dive
print(graph.describe(cypher=True))                    # Cypher reference
print(graph.describe(cypher=['cluster', 'MATCH']))    # detailed topic docs

# Declare child types (bubbles capabilities into parent descriptor)
graph.set_parent_type('ProductionProfile', 'Field')

# MCP server quickstart
print(kglite.KnowledgeGraph.explain_mcp())
```

---

## Graph Maintenance

```python
# Rebuild all indexes
graph.reindex()

# Compact graph (remove tombstones from deletions)
result = graph.vacuum()
# {'nodes_remapped': 50, 'tombstones_removed': 50}

# Auto-vacuum after DELETE operations
graph.set_auto_vacuum(0.3)     # trigger at 30% fragmentation (default)
graph.set_auto_vacuum(0.2)     # more aggressive
graph.set_auto_vacuum(None)    # disable

# Storage health diagnostics
info = graph.graph_info()
# {'node_count': 1000, 'node_capacity': 1050, 'node_tombstones': 50,
#  'edge_count': 2000, 'fragmentation_ratio': 0.048, ...}

# Read-only mode (blocks all Cypher mutations)
graph.read_only(True)
graph.read_only()   # → True
graph.read_only(False)
```

---

## Code Entity Methods

Methods for code knowledge graphs (built by e.g. the codingest project):

```python
# Find code entities by name
matches = graph.find('execute')
matches = graph.find('execute', node_type='Function')
matches = graph.find('exec', match_type='contains')       # substring
matches = graph.find('get_', match_type='starts_with')     # prefix

# Get source location
loc = graph.source('MyClass.my_method')
# {'file_path': 'src/main.py', 'line_number': 42, 'end_line': 55, ...}

# Batch source lookup
locs = graph.source(['func_a', 'func_b'])

# Neighborhood context
ctx = graph.context('MyClass', hops=2)
# {'node': {...}, 'defined_in': 'src/main.py', 'HAS_METHOD': [...], ...}

# Table of contents for a file
toc = graph.toc('src/main.py')
# {'file': 'src/main.py', 'entities': [...], 'summary': {'Function': 5, 'Class': 2}}
```

---

## Operation Reports

```python
graph.last_report()       # most recent operation report
graph.operation_index()   # sequential operation counter
graph.report_history()    # all reports
```

---

## Fluent vs Cypher Feature Matrix

| Feature | Fluent API | Cypher |
|---------|-----------|--------|
| **Spatial — point filters** | `within_bounds`, `near_point`, `near_point_m`, `contains_point` | `distance()`, `contains()` in WHERE |
| **Spatial — geometry** | `intersects_geometry`, `set_spatial`, `spatial` | `intersects()`, `centroid()`, `area()`, `perimeter()` |
| **Spatial — bounds/centroid** | `bounds()`, `centroid()`, `wkt_centroid()` | Manual via `latitude()`, `longitude()` |
| **Temporal — point-in-time** | `valid_at()` | `valid_at(e, date, 'from', 'to')` |
| **Temporal — range overlap** | `valid_during()` | `valid_during(e, start, end, 'from', 'to')` |
| **Timeseries — load** | `add_timeseries`, `add_ts_channel`, `set_time_index` | N/A (load via fluent) |
| **Timeseries — extract** | `timeseries()`, `time_index()` | `ts_series()` |
| **Timeseries — aggregate** | N/A (use Cypher) | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_at`, `ts_delta` |
| **Vector — load** | `set_embeddings`, `embed_texts`, `set_embedder` | N/A (load via fluent) |
| **Vector — search** | `vector_search()`, `search_text()` (`exact=` to force brute) | `text_score()` in RETURN/WHERE |
| **Vector — index (HNSW)** | `build_vector_index()`, `drop_vector_index()`, `has_vector_index()` | N/A (auto-used by fluent search) |
| **Path finding** | `shortest_path`, `all_paths` + 3 variants | `shortestPath()` in MATCH |
| **Centrality** | `betweenness_centrality`, `pagerank`, `degree_centrality`, `closeness_centrality` | `CALL pagerank() YIELD ...` etc. |
| **Community** | `louvain_communities`, `label_propagation` | `CALL louvain() YIELD ...` etc. |
| **Pattern matching** | `match_pattern()`, `traverse()`, `compare()`, `expand()` | Full MATCH with patterns |
| **Filtering** | `where`, `where_any`, `where_connected`, etc. | WHERE clause |
| **Aggregation** | `statistics()`, `count()`, `calculate()` | `count()`, `sum()`, `avg()` in RETURN |
| **Mutations** | `update()` (batch property update) | CREATE, SET, DELETE, REMOVE, MERGE |
| **Transactions** | `begin()`, `begin_read()`, `session()` | `Session.execute()` / `Transaction` provide rollback; direct `cypher()` writes execute in place |
| **Schema** | `define_schema()`, `validate_schema()` | N/A |
| **Set operations** | `union`, `intersection`, `difference`, `symmetric_difference` | `UNION` (result-level only) |
| **Indexes** | `create_index`, `create_range_index`, `create_composite_index` | Auto-maintained |
