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
