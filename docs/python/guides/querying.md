# Querying (Fluent API)

> For most queries, prefer [Cypher](cypher.md). The fluent API is for building reusable query chains or when you need `explain()` and selection-based workflows.

## Filtering

```python
graph.select('Product').where({'price': 999.99})
graph.select('Product').where({'price': {'<': 500.0}, 'stock': {'>': 50}})
graph.select('Product').where({'id': {'in': [101, 103]}})
graph.select('Product').where({'category': {'is_null': True}})

# Regex matching
graph.select('Person').where({'name': {'regex': '^A.*'}})   # or {'=~': '^A.*'}
graph.select('Person').where({'name': {'regex': '(?i)^alice'}})  # case-insensitive

# Negated conditions
graph.select('Person').where({'city': {'not_in': ['Oslo', 'Bergen']}})
graph.select('Person').where({'name': {'not_contains': 'test'}})
graph.select('Person').where({'name': {'not_regex': '^[A-C].*'}})

# OR logic — where_any keeps nodes matching ANY condition set
graph.select('Person').where_any([
    {'city': 'Oslo'},
    {'city': 'Bergen'},
])

# Connection existence — filter without changing the selection target
graph.select('Person').where_connected('KNOWS')                        # any direction
graph.select('Person').where_connected('KNOWS', direction='outgoing')  # outgoing only

# Orphan nodes (no connections)
graph.where_orphans(include_orphans=True)
```

## Sorting and Pagination

```python
graph.select('Product').sort('price')
graph.select('Product').sort('price', ascending=False)
graph.select('Product').sort([('stock', False), ('price', True)])

# Pagination with offset + limit
graph.select('Person').sort('name').offset(20).limit(10)  # page 3 of 10
```

## Traversing the Graph

```python
alice = graph.select('User').where({'title': 'Alice'})
alice_products = alice.traverse('PURCHASED', direction='outgoing')

# Filter and sort traversal targets
expensive = alice.traverse(
    'PURCHASED',
    where={'price': {'>=': 500.0}},
    sort_target='price',
    limit=10
)

# Get connection information
alice.connections(include_node_properties=True)
```

### Comparison Operations

For spatial, semantic, or clustering operations — where nodes are related
by proximity rather than explicit edges — use `compare()`:

```python
# Spatial: find wells inside structure polygons
graph.select('Structure').compare('Well', 'contains')

# Distance: wells within 5km of each platform
graph.select('Platform').compare('Well', {'type': 'distance', 'max_m': 5000})

# Semantic: similar documents by embedding
graph.select('Doc').compare('Doc',
    {'type': 'text_score', 'property': 'summary', 'threshold': 0.7})
```

See the [Traversal Hierarchy](traversal-hierarchy.md) guide for details
on multi-level chains, property enrichment, and grouped collection.

### Grouped Results

After traversal, `collect_grouped()` groups leaf nodes by a parent type:

```python
grouped = graph.select('Field').traverse('HAS_WELL') \
    .collect_grouped('Field')
# → {'TROLL': [{...}, ...], 'EKOFISK': [{...}, ...]}
```

### Enriching with `add_properties()`

Copy or aggregate properties from ancestor levels onto leaf nodes:

```python
from kglite import Agg, Spatial

graph.select('Structure').compare('Well', 'contains') \
    .add_properties({
        'Structure': {'struct_name': 'name'},     # copy + rename
        'Well': {'n_wells': Agg.count()},         # aggregate
    })
```

See the [Traversal Hierarchy](traversal-hierarchy.md) guide for the full
enrichment API.

## Set Operations

```python
n3 = graph.select('Prospect').where({'geoprovince': 'N3'})
m3 = graph.select('Prospect').where({'geoprovince': 'M3'})

n3.union(m3)                    # all nodes from both (OR)
n3.intersection(m3)             # nodes in both (AND)
n3.difference(m3)               # nodes in n3 but not m3
n3.symmetric_difference(m3)     # nodes in exactly one (XOR)
```

## Retrieving Results

```python
people = graph.select('Person')

# Lightweight (no property materialization)
people.len()                     # → 3
people.indices()                        # → [0, 1, 2]
people.ids()                      # → [1, 2, 3]

# Medium (partial materialization)
people.titles()                     # → ['Alice', 'Bob', 'Charlie']
people.get_properties(['age', 'city'])  # → [(28, 'Oslo'), (35, 'Bergen'), (42, 'Oslo')]

# Full materialization
people.collect()                      # → [{'type': 'Person', 'title': 'Alice', 'id': 1, 'age': 28, ...}, ...]
people.to_df()                          # → DataFrame with columns type, title, id, age, city, ...

# Single node lookup (O(1))
graph.node('Person', 1)       # → {'type': 'Person', 'title': 'Alice', ...} or None
```

## Schema Introspection

Methods for exploring graph structure — what types exist, what properties they have, and how they connect.

### `schema()` — Full graph overview

```python
s = graph.schema()
# {
#   'node_types': {
#     'Person': {'count': 500, 'properties': {'age': 'Int64', 'city': 'String'}},
#     'Company': {'count': 50, 'properties': {'founded': 'Int64'}},
#   },
#   'connection_types': {
#     'KNOWS': {'count': 1200, 'source_types': ['Person'], 'target_types': ['Person']},
#     'WORKS_AT': {'count': 500, 'source_types': ['Person'], 'target_types': ['Company']},
#   },
#   'indexes': ['Person.city', 'Person.(city, age)'],
#   'node_count': 550,
#   'edge_count': 1700,
# }
```

### `properties(node_type)` — Property details

```python
graph.properties('Person')
# {
#   'type':  {'type': 'str', 'non_null': 500, 'unique': 1, 'values': ['Person']},
#   'title': {'type': 'str', 'non_null': 500, 'unique': 500},
#   'id':    {'type': 'int', 'non_null': 500, 'unique': 500},
#   'city':  {'type': 'str', 'non_null': 500, 'unique': 3, 'values': ['Bergen', 'Oslo', 'Stavanger']},
#   'age':   {'type': 'int', 'non_null': 500, 'unique': 45},
# }
```

### `neighbors_schema(node_type)` — Connection topology

```python
graph.neighbors_schema('Person')
# {
#   'outgoing': [
#     {'connection_type': 'KNOWS', 'target_type': 'Person', 'count': 1200},
#     {'connection_type': 'WORKS_AT', 'target_type': 'Company', 'count': 500},
#   ],
#   'incoming': [
#     {'connection_type': 'KNOWS', 'source_type': 'Person', 'count': 1200},
#   ],
# }
```

### `sample(node_type, n=5)` — Quick data peek

```python
result = graph.sample('Person', n=3)
result[0]          # {'type': 'Person', 'title': 'Alice', 'id': 1, 'age': 28, 'city': 'Oslo'}
result.to_list()   # all rows as list[dict]
result.to_df()     # as DataFrame
```

### `describe()` — AI agent context

Progressive-disclosure schema description designed for AI agents. See [AI Agents](ai-agents.md) for details.

## Debugging Selections

```python
result = graph.select('User').where({'id': 1001})
print(result.explain())
# SELECT User (1000 nodes) -> WHERE (1 nodes)
```

## Pattern Matching

For simpler pattern-based queries without full Cypher clause support:

```python
results = graph.match_pattern(
    '(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)'
)

for match in results:
    print(f"Play: {match['p']['title']}, Discovery: {match['d']['title']}")

# With property conditions
graph.match_pattern('(u:User)-[:PURCHASED]->(p:Product {category: "Electronics"})')

# Limit results for large graphs
graph.match_pattern('(a:Person)-[:KNOWS]->(b:Person)', max_matches=100)
```
