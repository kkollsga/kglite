# Inline records

`kglite.from_records()` is the JSON-native loader for agents, APIs, and small
in-memory payloads. It builds nodes and relationships without temporary CSVs.

```python
import kglite

graph = kglite.from_records({
    "nodes": [{
        "type": "Person",
        "id_field": "id",
        "title_field": "name",
        "records": [{"id": 1, "name": "Alice", "tags": ["reviewer"]}],
    }],
    "connections": [{
        "type": "KNOWS",
        "source_type": "Person",
        "source_id_field": "source",
        "target_type": "Person",
        "target_id_field": "target",
        "records": [{"source": 1, "target": 2, "since": 2024}],
    }],
}, on_missing_endpoint="vivify")
```

JSON arrays/maps become native list/map values. `on_missing_endpoint` is:

- `"vivify"` (default) — create provisional endpoint stubs.
- `"drop"` — skip relationships whose source/target is absent and report them.
- `"error"` — validate the complete input and fail atomically before applying
  any block when an endpoint is absent.

Use `from_blueprint()` for repeatable CSV pipelines with compute operations;
use DataFrame bulk loaders for already-tabular/high-volume Python data.


## Embedded table vs row nodes

Structured payloads have two homes, and choosing deliberately matters more
than the mechanics:

| | Embedded table (`set_table_property`) | Row nodes (`attach_rows`) |
|---|---|---|
| Storage | one `list<map>` property on the parent | one node per row + one edge |
| Read | `get_table_property` → DataFrame (column order + dtypes restored); `UNWIND o.line_items`; `o.line_items[2].qty` | ordinary `MATCH` |
| Update one cell | `SET o.line_items[2].qty = 8` (atomic engine-side read-modify-write) or `CALL table.upsert(...)` | `SET r.qty = 8` |
| Validation | declared shape: `types: {line_items: "list<map{sku: string!, qty: int!}>"}` | per-property constraints/indexes |
| Indexing | none on inner cells (a lookup scans the parent's rows) | full property/range/composite indexes |
| Memory | lives in a heap-only Mixed column — cannot spill or mmap | normal columnar storage |
| Best for | small, always-read-together payloads (an order's line items, a config blob) | rows queried independently, joined against, indexed, or large |

Rules of thumb: reach for `attach_rows` as soon as a row needs its own
edges or an index, or the tables grow past a few hundred rows per parent;
keep tables embedded when they travel with the parent as one unit.
`kglite.attach_rows(g, "Order", "order-1", df, row_type="LineItem",
edge_type="HAS_LINE", key="sku")` makes the normalized form one call.
