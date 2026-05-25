---
name: graph_overview
description: "Read the active graph's schema — node types, property shapes (with example values), edge connectivity. TRIGGER as the FIRST call against any unfamiliar graph before writing Cypher; the cost is one round trip and the value is correctly-shaped queries on the next attempt. ALSO TRIGGER when you've been getting zero rows from cypher_query and suspect the schema differs from your mental model. ALSO TRIGGER when narrowing into one node type (`types=['Function']`) to see its full property catalogue with sample values. SKIP when you already have the schema cached from an earlier call in this session AND the graph hasn't been mutated. SKIP for file-system layout (use list_source) — graph_overview describes nodes/edges, not directories."
applies_to:
  mcp_methods: ">=0.3.36"
  kglite_mcp_server: ">=0.9.31"
references_tools:
  - graph_overview
references_arguments:
  - graph_overview.types
  - graph_overview.connections
  - graph_overview.cypher
references_properties: []
auto_inject_hint: true
---

# `graph_overview` methodology

## Overview

`graph_overview` returns an XML schema description of the active graph — every node type with its property catalogue (including one example value per property), every edge type with source/target shapes and counts. It is the **schema discovery tool**: call it before reasoning about a graph you don't already know. The output is small (< 4 KB for typical graphs, configurable for huge ones) and forms the basis of every well-shaped Cypher query.

## Quick Reference

| Task | Call |
|---|---|
| First contact with an unfamiliar graph | `graph_overview()` — no args |
| Drill into one node type's property catalogue | `graph_overview(types=['Function'])` |
| See connectivity (which types connect via which edges) | `graph_overview(connections=True)` |
| Sample a custom subset via Cypher | `graph_overview(cypher='MATCH (n:Function {is_test: true}) RETURN n LIMIT 200')` |
| Pre-flight a hypothesis | drill into the type you're about to query, look at `vals=` / `sample=` for properties |

## Reading the output

The XML is structured as `<graph>` containing one `<type>` per node label, each with `<properties>` and `<connections>` blocks:

```xml
<type name="Function" count="341" id_alias="qualified_name" title_alias="name">
  <properties>
    <prop name="branch_count" type="Int64" unique="2" vals="0|3"/>
    <prop name="docstring" type="String" unique="16" sample="Return a new ResultView..."/>
    <prop name="module" type="String" unique="2" vals="kglite|kglite.datasets.sodir.catalog"/>
    ...
  </properties>
  <connections>
    <out type="CALLS" target="Function" count="144"/>
    <in type="DEFINES" source="File" count="341"/>
    ...
  </connections>
</type>
```

The key columns to read carefully:

- **`count`** on `<type>` — how many nodes of this label exist. Sets your scale expectations for queries.
- **`vals="a|b|c"`** on `<prop>` — every distinct value when cardinality is low (≤15). Lets you write `WHERE n.kind IN ['class', 'protocol']` correctly the first time.
- **`sample="..."`** on `<prop>` (since 0.9.30) — one example value when cardinality is too high to enumerate. Tells you what the property *looks like* without guessing.
- **`<connections>`** — which edge types exist and what they connect. Don't guess; the agent often invents edge names that don't exist.

## Drill-down patterns

```
# Bare overview — start here every time
graph_overview()

# Just one type — when bare overview pointed you here and you need the
# full property catalogue with sample values
graph_overview(types=['Function'])

# Multiple types — when you're going to traverse between them
graph_overview(types=['Function', 'Class'])

# Connectivity-focused — when the question is "how does X connect to Y"
graph_overview(connections=True)

# Targeted Cypher inside the overview surface — useful when you want
# overview + a specific extract in one call
graph_overview(cypher='MATCH (n:Function) WHERE n.is_test=true RETURN n.module LIMIT 50')
```

The `cypher:` parameter inside `graph_overview` runs the query through the same schema-rendering layer — the row results are formatted alongside the schema context. Convenient when you want "here's the shape AND here's an example."

## Common Pitfalls

❌ Skipping `graph_overview` and writing Cypher directly against a graph the agent has never seen. The first query returns zero rows or the wrong shape; the agent re-queries; the round-trip count balloons. Always overview first.

❌ Reading the schema XML once at session start and forgetting it 10 calls later. Re-overview cheaply when the schema-shaped reasoning gets fuzzy — it's < 4 KB.

❌ Ignoring the `sample="..."` attribute on properties. The agent guesses `WHERE n.path CONTAINS 'foo'` when the actual property is `file_path` (with example shown in `sample=`). Read the schema; don't reconstruct it.

❌ Using `connections=True` only — without bare overview first you don't have property context. Bare → drill → connections is the natural escalation, not connections-first.

✅ Bare `graph_overview()` first thing every session. Cache mentally; re-call if the shape stops making sense.

✅ Read `vals=` and `sample=` carefully — they're the agent's authoritative source for "what does this property's value actually look like?"

✅ Drill into `types=['<one type>']` when you're about to write Cypher against that specific label. The full property catalogue + sample values is what makes the next query correct on the first attempt.

## Behaviour notes

- **Sticky overview prefix.** Manifests can declare `overview_prefix:` text that prepends to bare `graph_overview()` output. This is the operator's "first-impression" guidance for the agent — read it.
- **temp_cleanup on overview.** When `builtins.temp_cleanup: on_overview` is set in the manifest, every bare overview call wipes the `temp/` directory. Used in conjunction with `csv_http_server` to bound disk usage from `FORMAT CSV` exports. Side-effect on read; deliberate; document-able.
- **Disk-backed graphs** (storage="disk", e.g. Wikidata-scale) work the same way but full scans inside `graph_overview` are throttled — the property-stats pass samples 200 nodes per type rather than all of them. The output shape is identical.

## When `graph_overview` is the wrong tool

- **You already have the schema cached** and the graph hasn't been mutated — just write the Cypher.
- **Reading a directory tree** — that's `list_source`, not `graph_overview`. Graphs describe entities and relationships, not filesystems.
- **Wanting one specific entity's properties** — that's `cypher_query` with a precise `MATCH`. Overview gives the schema; Cypher gives the row.
