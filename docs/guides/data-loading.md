# Data Loading

> For most use cases, use [Cypher queries](cypher.md). The fluent API is for bulk operations from DataFrames or complex data pipelines.

## Adding Nodes

```python
import pandas as pd

products_df = pd.DataFrame({
    'product_id': [101, 102, 103],
    'title': ['Laptop', 'Phone', 'Tablet'],
    'price': [999.99, 699.99, 349.99],
    'stock': [45, 120, 30]
})

report = graph.add_nodes(
    data=products_df,
    node_type='Product',
    unique_id_field='product_id',
    node_title_field='title',
    columns=['product_id', 'title', 'price', 'stock'],       # whitelist columns (None = all)
    column_types={'launch_date': 'datetime'},                  # explicit type hints
    conflict_handling='update'  # 'update' | 'replace' | 'skip' | 'preserve'
)
print(f"Created {report['nodes_created']} nodes in {report['processing_time_ms']}ms")
```

## Property Mapping

When adding nodes, `unique_id_field` and `node_title_field` are **mapped** to `id` and `title`. The original column names become **aliases** — they work in Cypher queries and `filter()`, but results always use the canonical names.

| Your DataFrame Column | Stored As | Alias? |
|-----------------------|-----------|--------|
| `unique_id_field` (e.g., `user_id`) | `id` | `n.user_id` resolves to `n.id` |
| `node_title_field` (e.g., `name`) | `title` | `n.name` resolves to `n.title` |
| All other columns | Same name | — |

```python
# After adding with unique_id_field='user_id', node_title_field='name':
graph.cypher("MATCH (u:User) WHERE u.user_id = 1001 RETURN u")  # OK — alias resolves to id
graph.select('User').where({'user_id': 1001})              # OK — alias works here too
graph.select('User').where({'id': 1001})                   # Also OK — canonical name

# Results always use canonical names:
# {'id': 1001, 'title': 'Alice', 'type': 'User', ...}  — NOT 'user_id' or 'name'
```

## Creating Connections

```python
purchases_df = pd.DataFrame({
    'user_id': [1001, 1001, 1002],
    'product_id': [101, 103, 102],
    'date': ['2023-01-15', '2023-02-10', '2023-01-20'],
    'quantity': [1, 2, 1]
})

graph.add_connections(
    data=purchases_df,
    connection_type='PURCHASED',
    source_type='User',
    source_id_field='user_id',
    target_type='Product',
    target_id_field='product_id',
    columns=['date', 'quantity']
)
```

> `source_type` and `target_type` each refer to a single node type. To connect nodes of the same type, set both to the same value (e.g., `source_type='Person', target_type='Person'`).

## Working with Dates

```python
graph.add_nodes(
    data=estimates_df,
    node_type='Estimate',
    unique_id_field='estimate_id',
    node_title_field='name',
    column_types={'valid_from': 'datetime', 'valid_to': 'datetime'}
)

graph.select('Estimate').where({'valid_from': {'>=': '2020-06-01'}})
graph.select('Estimate').valid_at('2020-06-15')
graph.select('Estimate').valid_during('2020-01-01', '2020-06-30')
```

## Batch Property Updates

```python
result = graph.select('Prospect').where({'status': 'Inactive'}).update({
    'is_active': False,
    'deactivation_reason': 'status_inactive'
})

updated_graph = result['graph']
print(f"Updated {result['nodes_updated']} nodes")
```

## Operation Reports

Operations that modify the graph return detailed reports:

```python
report = graph.add_nodes(data=df, node_type='Product', unique_id_field='product_id')
# report keys: operation, timestamp, nodes_created, nodes_updated, nodes_skipped,
#              processing_time_ms, has_errors, errors

graph.last_report()       # most recent operation report
graph.operation_index()   # sequential index of last operation
graph.report_history()    # all reports
```

## Blueprints

Build a complete graph from CSV files using a declarative JSON blueprint — see the [Blueprints guide](blueprints.md) for a full walkthrough.

```python
graph = kglite.from_blueprint("blueprint.json", verbose=True)
```
