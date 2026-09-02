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

The key sets above are closed — `nodes`, `connections` and
`on_missing_endpoint` at the top level, and the per-spec keys shown. An unknown
key raises a `ValueError` naming a near-miss where there is one, because a key
the loader does not read would otherwise be dropped in silence: a spec written
with `"relationships"` builds no relationships at all.

A node spec may also carry `"labels": ["Human", "Agent"]` — secondary labels
stamped on every node of that type, including endpoint stubs `vivify` created
for it, so `MATCH (:Human)` sees the whole type. Listing the type's own name is
a no-op, not a duplicate.

JSON arrays/maps become native list/map values. `on_missing_endpoint` is:

- `"vivify"` (default) — create provisional endpoint stubs.
- `"drop"` — skip relationships whose source/target is absent and report them.
- `"error"` — validate the complete input and fail atomically before applying
  any block when an endpoint is absent.

Use `from_blueprint()` for repeatable CSV pipelines with compute operations;
use DataFrame bulk loaders for already-tabular/high-volume Python data.


## Embedded table vs row nodes

Nested lists/maps loaded through `from_records` can also live as embedded
table properties or as row nodes — the decision table, the declared-shape
validation, and the `attach_rows` helper are in {doc}`structured-data`.
