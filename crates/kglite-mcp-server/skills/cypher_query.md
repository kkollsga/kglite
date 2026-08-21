---
name: cypher_query
description: "Run Cypher against the active knowledge graph to answer structural questions — what calls what, what's connected to what, what types exist, what matches a predicate. TRIGGER when the user asks about graph entities by label, property, or relationship (e.g. \"functions that call X\", \"cases citing this statute\", \"wells in this field\"), needs counts/aggregations across a type, or wants to traverse multi-hop paths. ALSO TRIGGER when text search via grep would be the obvious move BUT the question is structural (\"where is X defined?\" is a graph question, not a regex question — exact, not fuzzy). SKIP for whole-file reads (use read_source), file-tree exploration (use list_source), or symbol lookups when you already have a qualified_name (use read_code_source). SKIP when you don't know the schema yet — call graph_overview first."
applies_to:
  mcp_methods: ">=0.3.36"
  kglite_mcp_server: ">=0.9.31"
references_tools:
  - cypher_query
references_arguments:
  - cypher_query.query
auto_inject_hint: true
---

# `cypher_query` methodology

## Overview

`cypher_query` runs a Cypher query against the active graph and returns up to 15 rows inline (append `FORMAT CSV` for larger result sets — see below). It is the **structural read tool** — use it when the question can be expressed as labels, properties, and relationships. Always call `graph_overview` first if you don't yet know the schema; the saved round trip in the next query is worth more than the cost of one schema scan.

Standard openCypher works — `MATCH` / `OPTIONAL MATCH` / `WHERE` / `WITH` / `UNWIND` / `RETURN`, `ORDER BY` / `SKIP` / `LIMIT`, `DISTINCT`, variable-length paths, and the usual aggregates (`count`, `sum`, `avg`, `min`, `max`, `collect`). Write ordinary Cypher; what `graph_overview` lists under `<extensions>` are KGLite additions on top, not a substitute dialect.

## Quick Reference

| Task | Pattern |
|---|---|
| Find nodes by label + predicate | `MATCH (n:Vessel) WHERE n.flag STARTS WITH 'NO' RETURN n.title LIMIT 10` |
| Count by label | `MATCH (n:Vessel) RETURN count(n) AS n` |
| Traverse a relationship | `MATCH (a:Vessel)-[:VISITED]->(b:Port) WHERE a.title = 'Nordkapp' RETURN b.title` |
| Multi-hop with variable length | `MATCH (a)-[:VISITED*1..3]->(b) WHERE a.id = ... RETURN b.id` |
| Bind a value instead of inlining it | `params={"flag": "NO"}` with `MATCH (n:Vessel {flag: $flag}) RETURN n.title` |
| Aggregate by group | `MATCH (n:Vessel) RETURN n.flag AS flag, count(n) AS ships ORDER BY ships DESC LIMIT 20` |
| Larger result set | append `FORMAT CSV` — a CSV body capped at 200 rows (or a fetch URL when the operator enabled `csv_http_server`) |

## Returning rows, not whole nodes

The 15-row inline cap applies to rows, not properties — but a single `RETURN n` row carries every property on the node, which inflates fast. Two anti-patterns to avoid:

```cypher
MATCH (n:Vessel) RETURN n LIMIT 5              -- 5 rows, but each row is a 20-property blob
MATCH (n:Vessel) RETURN n.title, n.flag        -- 5 rows, 2 columns each — far smaller
```

The agent's context budget appreciates the second form. Reach for the first only when you genuinely need the whole node and have narrowed via `WHERE` to a single match.

## `FORMAT CSV` for larger result sets

If the inline 15-row cap is going to truncate, append `FORMAT CSV` to the query.

**`FORMAT CSV` is not uncapped.** The inline CSV body carries at most **200 data rows**; past that it is trimmed and a trailing notice names the true row count and the full byte size. It is a wider window, not an export. When the operator has set `extensions.csv_http_server: true` in the manifest, the complete result is instead written to a temp file and returned as a `http://127.0.0.1:<port>/<hash>.csv` URL to fetch over HTTP — that, and only that, is the full-export path, and it is an operator setting no query can turn on.

Use FORMAT CSV when:

- The result set is moderately large (15–200 rows of structured data to scan).
- The server returns a fetch URL and you're exporting for downstream analysis.

Don't use FORMAT CSV when:

- You expect <15 rows. The inline preview is faster.
- You want the whole of a large result. Aggregate it in the query, or paginate with `LIMIT` + `SKIP`; re-running the same query cannot widen the cap.
- You want to drill into specific entities — narrow the `WHERE` instead.

## Property shape

When in doubt about a property's name or value shape, look at `graph_overview`'s `<prop sample="..." />` output — every property carries one example value to pattern-match on, and a `coverage="…%"` attribute appears when the property is missing from some nodes of its type.

## Common Pitfalls

❌ Writing Cypher without calling `graph_overview` first — the agent guesses at node labels and property names, gets zero rows, and re-queries. Always start with an overview when entering an unfamiliar graph.

❌ `MATCH (n) RETURN n LIMIT 5` against a graph with millions of nodes — the planner may scan before applying LIMIT depending on shape. Always include a label filter and use specific properties in the RETURN.

❌ Re-querying with slightly different shapes when the first query returned 0 rows — usually means the property name is wrong (look at the schema, sample values are right there) or the label is wrong.

❌ Matching on a display name when the type's identifier is more specific. `MATCH (n:Vessel {title: "Nordkapp"})` matches every vessel sharing that name; matching on the type's `id` (see `graph_overview`'s `id_alias=`) matches one.

✅ `graph_overview` first if you don't know the schema. The cost is one round trip; the value is correctly-shaped Cypher on the second attempt.

✅ Return specific properties (`RETURN n.title, n.flag`) not whole nodes. Smaller context, faster cognition.

✅ Use `LIMIT` aggressively while exploring. Drop or raise it once you know the result shape.

## When `cypher_query` is the wrong tool

- **Don't know what types exist?** Call `graph_overview` first — same server, different surface. Cypher against unknown schema is shadow-boxing.
- **Question is textual, not structural?** `grep` is the regex sweep across the server's source roots, when it has any. Cypher can find an entity by property but not by "the string 'TODO' appears nearby."
- **Reading a file, or browsing a directory tree?** `read_source` / `list_source`, when the server registers them; Cypher answers about the graph, not the filesystem.

## Format quirks

- Single-line queries are easiest to read; multi-line is fine when the WHERE/RETURN combination gets long.
- Bind values with the `params` argument rather than inlining them: `params={"flag": "NO"}` fills `$flag` in both an inline property map (`MATCH (n:Vessel {flag: $flag})`) and a `WHERE` clause. A `$name` the query uses but `params` omits is an error, never an empty result.
- `RETURN n, m AS alias` works; aliasing improves readability when joining tables in the head of your reasoning.
