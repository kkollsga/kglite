# Traversal Hierarchy

This guide explains how KGLite builds a multi-level hierarchy as you
chain `traverse()` and `compare()` calls, and how to use that hierarchy
for property enrichment and grouped collection.

## How the hierarchy works

Every fluent chain starts with `select()`, which creates **level 0** —
a flat set of nodes of a single type. Each subsequent `traverse()` or
`compare()` call adds a new level, building a parent → child tree:

```
Level 0:  select('Field')         → [Field_A, Field_B]
Level 1:  .traverse('HAS_WELL')   → [Well_1, Well_2, Well_3, Well_4]
Level 2:  .traverse('HAS_LOG')    → [Log_X, Log_Y, Log_Z]
```

Internally, each level tracks which parent produced which children:

```
Field_A ──→ Well_1 ──→ Log_X
                   ──→ Log_Y
        ──→ Well_2
Field_B ──→ Well_3 ──→ Log_Z
        ──→ Well_4
```

The **current selection** is always the leaf level (the most recent).
When you call `collect()` or `to_df()`, you get the leaf nodes.

## Edge traversal vs comparison

KGLite provides two ways to move between levels:

| Method | What it does | When to use |
|--------|-------------|-------------|
| `traverse(conn_type)` | Follow graph edges | Your data has explicit connections |
| `compare(target_type, method)` | Spatial, semantic, or clustering match | Find related nodes without edges |

```python
# Edge-based: follow HAS_WELL connections
graph.select('Field').traverse('HAS_WELL')

# Comparison-based: find wells inside structure polygons
graph.select('Structure').compare('Well', 'contains')
```

Both produce the same kind of hierarchy — the enrichment and grouping
features below work identically regardless of how the level was created.

## Enriching with `add_properties()`

Once you have a multi-level hierarchy, `add_properties()` copies or
computes values from ancestor levels onto the leaf nodes.

```python
from kglite import Agg, Spatial
```

### Copy properties from ancestors

```python
# Copy 'name' and 'status' from the Field level onto Wells
graph.select('Field').traverse('HAS_WELL') \
    .add_properties({'Field': ['name', 'status']})

# Rename while copying
graph.select('Field').traverse('HAS_WELL') \
    .add_properties({'Field': {'field_name': 'name'}})
```

### Aggregate across leaves

Aggregate functions compute summary statistics per ancestor group:

```python
graph.select('Structure').compare('Well', 'contains') \
    .add_properties({'Well': {
        'well_count': Agg.count(),         # count of wells per structure
        'avg_depth': Agg.mean('depth'),    # mean depth per structure
        'max_depth': Agg.max('depth'),     # deepest well per structure
        'all_names': Agg.collect('name'),  # comma-separated well names
    }})
```

Available aggregations: `Agg.count()`, `Agg.sum(prop)`, `Agg.mean(prop)`,
`Agg.min(prop)`, `Agg.max(prop)`, `Agg.std(prop)`, `Agg.collect(prop)`.

### Spatial computed properties

When ancestors have geometry, compute spatial relationships:

```python
graph.select('Structure').compare('Well', 'contains') \
    .add_properties({'Structure': {
        'dist_to_center': Spatial.distance(),    # meters from well to structure centroid
        'struct_area': Spatial.area(),            # structure area in m²
        'struct_perim': Spatial.perimeter(),      # perimeter in m
    }})
```

Available: `Spatial.distance()`, `Spatial.area()`, `Spatial.perimeter()`,
`Spatial.centroid_lat()`, `Spatial.centroid_lon()`.

### Multi-hop enrichment

In a three-level chain (A → B → C), you can pull properties from any
ancestor — not just the immediate parent:

```python
graph.select('Field').traverse('HAS_BLOCK').traverse('HAS_WELL') \
    .add_properties({
        'Block': ['block_name'],       # from level 1 (immediate parent)
        'Field': ['field_status'],     # from level 0 (grandparent)
    })
```

## Grouped collection with `collect_grouped()`

By default, `collect()` returns a flat `ResultView` of the leaf nodes.
To see how leaves are grouped by an ancestor type, use `collect_grouped()`:

```python
# Flat: all wells regardless of parent
wells = graph.select('Field').traverse('HAS_WELL').collect()

# Grouped by field
grouped = graph.select('Field').traverse('HAS_WELL') \
    .collect_grouped('Field')
# → {'TROLL': [{...}, {...}], 'EKOFISK': [{...}]}

# Include parent metadata
grouped = graph.select('Field').traverse('HAS_WELL') \
    .collect_grouped('Field', parent_info=True)
```

## Common patterns

### Aggregate-then-export

```python
df = graph.select('Structure').compare('Well', 'contains') \
    .add_properties({'Well': {
        'n_wells': Agg.count(),
        'avg_depth': Agg.mean('depth'),
    }}) \
    .to_df()
```

### Multi-hop with intermediate enrichment

```python
graph.select('Field').traverse('HAS_BLOCK').traverse('HAS_WELL') \
    .add_properties({
        'Block': {'block_name': 'name'},
        'Field': {'field_name': 'name'},
    }) \
    .collect()
```

### Compare then group

```python
graph.select('Structure').compare('Well', 'contains') \
    .collect_grouped('Structure')
```
