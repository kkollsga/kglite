# Data Loading

> For most use cases, use [Cypher queries](cypher.md). The fluent API is for bulk operations from DataFrames or complex data pipelines.

## End-to-end walkthrough — DataFrames in, queries out

This is the full path from raw `pandas` tables to a queryable graph
on disk. Most domain graphs land somewhere on this template.

### 1. Shape your tables

Two flat tables — one row per node, one row per edge — is enough.
The columns you'll point `add_nodes` / `add_connections` at:

```python
import pandas as pd
import kglite

users = pd.DataFrame({
    "user_id": [1001, 1002, 1003],
    "name":    ["Alice", "Bob",   "Carol"],
    "country": ["US",    "UK",    "US"],
})

products = pd.DataFrame({
    "sku":   ["P-101", "P-102", "P-103"],
    "title": ["Laptop", "Phone", "Tablet"],
    "price": [999.99,  699.99,  349.99],
})

orders = pd.DataFrame({
    "user_id":  [1001, 1001, 1002, 1003],
    "sku":      ["P-101", "P-103", "P-102", "P-101"],
    "date":     ["2024-01-15", "2024-02-10", "2024-01-20", "2024-03-04"],
    "quantity": [1, 2, 1, 1],
})
```

### 2. Load nodes

```python
graph = kglite.KnowledgeGraph()

graph.add_nodes(users,    "User",    "user_id", "name")
graph.add_nodes(products, "Product", "sku",     "title")
```

`unique_id_field` (3rd arg) is what makes a row identifiable;
`node_title_field` (4th, optional) is the human-readable label.
Both get [aliased](#property-mapping) so queries can use either
the original column name or the canonical `id` / `title`.

### 3. Load edges

```python
graph.add_connections(
    orders,
    connection_type="ORDERED",
    source_type="User",      source_id_field="user_id",
    target_type="Product",   target_id_field="sku",
    columns=["date", "quantity"],   # extra props go on the edge
)
```

The two `(type, id_field)` pairs tell `add_connections` how to
look up endpoints in the existing nodes. Any other columns ride
along as edge properties.

### 4. Query

```python
# How many distinct products has each user ordered?
graph.cypher("""
    MATCH (u:User)-[:ORDERED]->(p:Product)
    RETURN u.name AS user, count(DISTINCT p) AS unique_products
    ORDER BY unique_products DESC
""")

# Total revenue by country, layering arithmetic on edge + node props.
graph.cypher("""
    MATCH (u:User)-[r:ORDERED]->(p:Product)
    RETURN u.country, sum(r.quantity * p.price) AS revenue
    ORDER BY revenue DESC
""")
```

### 5. Save & reload

```python
graph.save("orders.kgl")
g2 = kglite.load("orders.kgl")
```

That's the whole loop — three `add_*` calls, one Cypher query.
Everything below this section is reference detail for when the
template needs to bend (timeseries, dates, batch updates, RDF,
declarative blueprints).

## `add_nodes` — parameter reference

The walkthrough covered the four positional arguments. The optional
keyword arguments worth knowing:

| Parameter | Purpose |
|---|---|
| `columns=[...]` | Whitelist DataFrame columns to ingest. Default `None` = all columns. |
| `skip_columns=[...]` | Inverse: drop these columns before ingest. |
| `column_types={'col': 'datetime'}` | Force a column's storage type. Most common values: `'datetime'`, `'point'`, `'embedding'`. |
| `conflict_handling='update'` | What to do when a row's id already exists. See [Loading in passes](#loading-in-passes) for the full table. |
| `timeseries={...}` | Inline timeseries declaration — see the [Timeseries guide](timeseries.md). |
| `nullable_int_downcast=True` | Convert pandas nullable ints (`Int64`) to native `int64` even when the column has no nulls. |

Every call returns a report dict:

```python
report = graph.add_nodes(products_df, 'Product', 'sku', 'title')
print(report)
# {'operation': 'add_nodes', 'nodes_created': 3, 'nodes_updated': 0,
#  'nodes_skipped': 0, 'has_errors': False, 'processing_time_ms': 0.4, ...}
```

A `UserWarning` fires automatically when the report has skipped
rows or `has_errors=True` — silent partial successes are surfaced
without you needing to inspect the report.

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

## `add_connections` — parameter reference

Past the six required positional arguments
(`data, connection_type, source_type, source_id_field, target_type,
target_id_field`), the keyword surface:

| Parameter | Purpose |
|---|---|
| `columns=[...]` | Whitelist DataFrame columns to attach as edge properties. |
| `skip_columns=[...]` | Inverse: drop these columns. |
| `conflict_handling='update'` | What to do when an edge with the same endpoints already exists. Same modes as `add_nodes`. |
| `query=...` | Alternative to `data=df`: a Cypher query whose `RETURN` columns supply the source/target ids. Lets you stamp edges from results of a query. |
| `extra_properties={...}` | Static properties to attach to every edge (handy with `query=`). |

`source_type` and `target_type` each refer to a single node type.
To connect same-type nodes (org charts, taxonomies), set both to
the same value — see the [Hierarchies](#hierarchies-explicit-edges-vs-set_parent_type)
section for when this is the right move.

## Loading in passes

Real graphs rarely come from one DataFrame. Two patterns dominate:

**Static rows, then timeseries.** Load the structural columns
once, then layer timeseries observations on top. Common for
sessions with daily registration counts, sensors with hourly
readings, products with weekly inventory.

```python
# Pass 1 — one row per node, all static columns
graph.add_nodes(sessions_df, "Session", "session_id", "title")

# Pass 2 — many rows per node, just (id, time, value) shape.
# Notice: no node_title_field. Titles set in pass 1 are preserved.
graph.add_nodes(
    snapshots_df,
    "Session",
    "session_id",
    timeseries={"time": "snapshot_date", "channels": ["registrants"]},
    conflict_handling="update",
)
```

**Schema, then enrichment.** Initial pass establishes the node
inventory; later passes add columns from joins, scores, or
external lookups. Same shape — call `add_nodes` again with the
same `node_type` and `unique_id_field`.

```python
graph.add_nodes(users_df,        "User", "user_id", "name")
graph.add_nodes(pagerank_scores, "User", "user_id")  # adds .pagerank
graph.add_nodes(geocoded,        "User", "user_id")  # adds .lat / .lon
```

### What carries over between calls

A second `add_nodes(..., node_type="X", ...)` doesn't reset the
type — it merges:

| | Carried from prior calls |
|---|---|
| Existing node titles | ✅ Preserved unless you pass `node_title_field` again |
| `id_alias` / `title_alias` | ✅ Preserved (no need to re-pass `node_title_field` to keep the alias) |
| Properties on existing nodes | Merged per `conflict_handling` (below) |
| Spatial / temporal / embedding configs | ✅ Preserved; new ones merge in |
| `column_types` declared once | ⚠️ Re-declare for any new columns; must not contradict prior types |

### Conflict handling cheatsheet

`conflict_handling=` controls what happens when a row's id matches
an existing node:

| Mode | Behavior on existing nodes | When to use |
|---|---|---|
| `"update"` *(default)* | Merge properties; new values overwrite, nulls leave existing alone | Layering enrichment; the usual choice |
| `"preserve"` | Merge properties; existing values win | Backfilling defaults without trampling earlier truth |
| `"replace"` | Reset properties to the new row | A reload that should fully redefine the node |
| `"skip"` | Don't touch existing nodes; only insert new ids | Idempotent appends |
| `"sum"` | Add numeric values; same as `"update"` for non-numeric | Accumulating counters across batches |

Inspect `graph.last_report()` after each pass to confirm the
expected `nodes_created` / `nodes_updated` split.

## Hierarchies — explicit edges vs `set_parent_type`

These two APIs sound similar but solve different problems.
Picking the wrong one usually doesn't break anything — it just
makes `describe()` and your queries less ergonomic than they
could be.

**Use an explicit edge** when one *instance* is the parent of
another, typically same-type:

```python
# Org chart — Company nodes parent other Company nodes.
companies_df = pd.DataFrame({"id": ["alphabet", "google", "youtube"], ...})
parent_of    = pd.DataFrame({
    "parent": ["alphabet", "alphabet", "google"],
    "child":  ["google",   "calico",   "youtube"],
})
graph.add_nodes(companies_df, "Company", "id", "name")
graph.add_connections(parent_of, "PARENT_OF",
                      "Company", "parent", "Company", "child")

# Now Cypher walks the tree:
graph.cypher("""
    MATCH (root:Company {id: 'alphabet'})-[:PARENT_OF*]->(c:Company)
    RETURN c.name
""")
```

This is what you want for org charts, threaded comments, taxonomy
trees with arbitrary depth, geographic containment chains — any
relationship where the same type points at itself and depth is
unbounded.

**Use `set_parent_type`** when a *whole node type* is a
structural child of another type — the child instances exist only
because their parent does, and an LLM seeing `describe()` is
better off thinking of them as "facets of the parent" than as
peer types:

```python
graph.add_nodes(fields_df,             "Field",             "id", "name")
graph.add_nodes(production_profiles,   "ProductionProfile", "id")
graph.add_nodes(reserves,              "FieldReserves",     "id")

# Tell describe() these are supporting children of Field
graph.set_parent_type("ProductionProfile", "Field")
graph.set_parent_type("FieldReserves",     "Field")
```

This affects only `describe()` output: the supporting types drop
out of the top-level inventory and reappear inside the `<type
name="Field">` block with their capabilities (timeseries, spatial,
…) bubbled up to the parent. Cypher still treats them as
ordinary node types — `MATCH (p:ProductionProfile) ...` works
exactly as before.

| Question | Answer |
|---|---|
| "Is Company A above Company B in the org chart?" | Edge: `PARENT_OF` |
| "How deep is this taxonomy tree?" | Edge (variable-length `*` works on edges, not type-tiers) |
| "Show the agent that `ProductionProfile` is really part of `Field`" | `set_parent_type` |
| "Hide noisy supporting types from the inventory but keep them queryable" | `set_parent_type` |

You can use both at once — they don't interact. Edges shape
queries; `set_parent_type` shapes the LLM's mental model.

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

## N-Triples and RDF

`load_ntriples()` streams an RDF/N-Triples file directly into the
graph — designed for Wikidata `latest-truthy` dumps but works with
any N-Triples input:

```python
graph = kglite.KnowledgeGraph(storage="disk", path="/data/wd")
graph.load_ntriples("latest-truthy.nt.bz2", languages=["en"], verbose=True)
```

Compressed inputs (`.bz2`, `.gz`, `.zst`, `.zstd`) are decoded
on-the-fly. Multistream `.bz2` files (the format Wikidata ships)
use a parallel decoder under the hood — ~3× faster than the
single-threaded `MultiBzDecoder`.

`verbose=True` emits one `[Phase N]` line at every gate
(streaming → columnar build → edges → CSR → finalising) plus
periodic in-phase progress for the long Phase 1. Sub-step timings
(CSR step 1/4, peer-count histograms, mmap layout, …) live behind
an env var:

```bash
KGLITE_BUILD_DEBUG=1 python build_graph.py
```

For Wikidata or Sodir specifically, prefer the bundled lifecycle
wrappers — they handle download, cooldown, and resume on top of
`load_ntriples`:

```python
from kglite.datasets import wikidata
g = wikidata.open("/data/wd")
```

See the {doc}`datasets` guide for full coverage.

## Blueprints

Build a complete graph from CSV files using a declarative JSON blueprint — see the [Blueprints guide](blueprints.md) for a full walkthrough.

```python
graph = kglite.from_blueprint("blueprint.json", verbose=True)
```
