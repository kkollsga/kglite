# Structured data (tables, nested values, shapes)

Nodes and relationships carry scalar properties naturally; real data also
arrives as **tables and nested records** — an order's line items, a config
blob, a list of tags. KGLite stores these on the existing `list`/`map`
value substrate (there is deliberately no separate "table" value or file
format) and layers four things on top: DataFrame fidelity, declared
shapes, atomic nested mutations, and an easy path to the normalized
row-nodes form when embedding stops being right.

## Table properties (DataFrame in, DataFrame out)

```python
import pandas as pd

items = pd.DataFrame({
    "sku":   ["a-1", "b-2", "c-3"],
    "qty":   [1, 2, 8],
    "price": [9.5, 12.0, 3.25],
})
g.set_table_property("Order", "order-1", "line_items", items)

df = g.get_table_property("Order", "order-1", "line_items")
# column order, dtypes, and nullability restored — including pandas
# nullable dtypes (Int64) for columns that held nulls
```

The stored value is a plain **list of maps**, so Cypher sees ordinary
data — no special reader required:

```cypher
MATCH (o:Order {id: 'order-1'})
UNWIND o.line_items AS r
RETURN r.sku, r.qty

MATCH (o:Order {id: 'order-1'}) RETURN o.line_items[2].price
```

What a bare list of maps loses is DataFrame fidelity — map keys are stored
sorted, so column order would be gone — which is why the helper records
column order, dtypes, and nullability in a per-`(type, property)` registry
persisted with the graph. Only `get_table_property` reads it; Cypher never
does. Writes route through Cypher `SET` internally, so write scope,
constraints, declared shapes, WAL, and CDC all apply. Unsupported cell
values are rejected up front, never silently dropped.

## Declared shapes (validating nested values)

KGLite lists are heterogeneous by design, and the `IS :: TYPE` constraint
vocabulary deliberately excludes `LIST`/`MAP`. Structured **shapes** fill
the gap, declared through `define_schema`'s `types` values:

```python
g.define_schema({"nodes": {"Order": {"types": {
    "line_items": "list<map{sku: string!, qty: int!, price: float}>",
    "metadata":   "map{status: string!, note: string}",
    "tags":       "list<string>",
}}}})
```

`!` marks a required key; unmarked keys may be absent or null. Scalars use
the `define_schema` spellings (`string|str`, `int|integer`, `float`,
`bool|boolean`, `date|datetime`).

A declared shape is enforced **before anything is written** at every write
path — `add_nodes`/`from_records` (whole-frame: one bad row aborts the
batch with nothing loaded), Cypher `SET`, and `CREATE` — with the exact
cell named:

```text
add_nodes row 37: line_items[37].qty: expected integer, got String
```

Plain scalar type strings (`"qty": "int"` outside a shape) remain advisory
exactly as before — declaring a *shape* is the opt-in to enforcement. A
structured-looking declaration that does not parse fails `define_schema`
rather than silently becoming advisory. Recovery does not repeat structured
shape checks on values already admitted to the log. It does validate the final
state against declared uniqueness, required-property and property-type
constraints, preserving unchanged violations from the loaded checkpoint.

`describe()` renders the declared shape per property (or, without a
declaration, a shape inferred from one sampled value, flagged
`shape_inferred="true"`), so agents see the contract.

## Atomic nested mutations

Updating one cell no longer means rebuilding the collection in application
code — the read-modify-write happens inside the engine, atomically per
statement, which also removes the lost-update window between an
application-level read and write:

```cypher
SET o.line_items[2].qty = 8          -- one cell
SET o.metadata.status = 'approved'   -- creates the map if absent
SET o.line_items = o.line_items + [$row]   -- append (the list `+` operator)
```

Errors name the path (`line_items[7]: index out of bounds (list has 3
entries)`); negative indexes count from the end; a declared shape
re-validates the whole collection after the edit. Nested paths are
read-side too: `o.line_items[2].price` in any expression.

For keyed row operations there are two write procedures:

```cypher
CALL table.upsert({type: 'Order', id: 'order-1', property: 'line_items',
                   key: 'sku', row: {sku: 'b-2', qty: 42}})
  YIELD action, rows       -- 'updated' or 'inserted', and the new row count

CALL table.delete({type: 'Order', id: 'order-1', property: 'line_items',
                   key: 'sku', value: 'a-1'})
  YIELD removed, rows
```

`table.upsert` replaces the first row whose `key` cell matches `row[key]`
(whole-row replace) or appends; `table.delete` removes every match. Both
report `mode: WRITE` in `SHOW PROCEDURES` and honor read-only mode and
write scope.

## Embedded table vs row nodes

Structured payloads have two homes, and choosing deliberately matters more
than the mechanics:

| | Embedded table (`set_table_property`) | Row nodes (`attach_rows`) |
|---|---|---|
| Storage | one `list<map>` property on the parent | one node per row + one edge |
| Read | `get_table_property` → DataFrame; `UNWIND o.line_items`; `o.line_items[2].qty` | ordinary `MATCH` |
| Update one cell | `SET o.line_items[2].qty = 8` or `CALL table.upsert(...)` | `SET r.qty = 8` |
| Validation | declared shape | per-property constraints |
| Indexing | none on inner cells (a lookup scans the parent's rows) | full property/range/composite indexes |
| Memory | lives in a heap-only Mixed column — cannot spill or mmap | normal columnar storage |
| Best for | small, always-read-together payloads | rows queried independently, joined, indexed, or large |

Rules of thumb: reach for row nodes as soon as a row needs its own edges
or an index, or tables grow past a few hundred rows per parent; keep
tables embedded when they travel with the parent as one unit.

`attach_rows` makes the normalized form one call:

```python
kglite.attach_rows(g, "Order", "order-1", items,
                   row_type="LineItem", edge_type="HAS_LINE", key="sku")
# one :LineItem node per row (id "order-1:<sku>"), HAS_LINE edges from the parent
```

## See also

- {doc}`data-loading` — bulk-loading the row-nodes form directly.
- {doc}`inline-records` — nested JSON through `from_records` (which
  produces the same `list`/`map` values).
- {doc}`ontology` — declared semantics for the *types and relationships*
  themselves.
