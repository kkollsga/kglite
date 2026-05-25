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

## Quick Reference

| Task | Pattern |
|---|---|
| Find nodes by label + predicate | `MATCH (n:Function) WHERE n.module STARTS WITH 'foo' RETURN n.name LIMIT 10` |
| Count by label | `MATCH (n:Function) RETURN count(n) AS n` |
| Traverse a relationship | `MATCH (a:Function)-[:CALLS]->(b:Function) WHERE a.name = 'parse' RETURN b.name` |
| Multi-hop with variable length | `MATCH (a)-[:CALLS*1..3]->(b) WHERE a.id = ... RETURN b.id` |
| Aggregate by group | `MATCH (n:Function) RETURN n.module AS module, count(n) AS funcs ORDER BY funcs DESC LIMIT 20` |
| Large CSV export | append `FORMAT CSV` — returns a localhost URL the agent fetches (when `csv_http_server` is enabled in the manifest) |

## Returning rows, not whole nodes

The 15-row inline cap applies to rows, not properties — but a single `RETURN n` row carries every property on the node, which inflates fast. Two anti-patterns to avoid:

```cypher
MATCH (n:Function) RETURN n LIMIT 5            -- 5 rows, but each row is a 20-property blob
MATCH (n:Function) RETURN n.name, n.module     -- 5 rows, 2 columns each — far smaller
```

The agent's context budget appreciates the second form. Reach for the first only when you genuinely need the whole node and have narrowed via `WHERE` to a single match.

## `FORMAT CSV` for large result sets

If the inline 15-row cap is going to truncate, append `FORMAT CSV` to the query. With `extensions.csv_http_server: true` set in the manifest, the result is written to a temp file and returned as a `http://127.0.0.1:<port>/<hash>.csv` URL the agent can fetch via standard HTTP. Without `csv_http_server` configured, the CSV body is inlined (which works for thousands of rows; painful for millions).

Use FORMAT CSV when:

- The result set is genuinely large (>15 rows of structured data the agent needs to scan).
- You're exporting for downstream analysis (the URL is reachable from the agent's tool-use context).

Don't use FORMAT CSV when:

- You expect <15 rows. The inline preview is faster.
- You want to drill into specific entities — paginate by adding `LIMIT` + `SKIP` or narrow the `WHERE` instead.

## Property shape across types

In kglite's code-tree graphs, every entity carries a `module` property (the dotted module path of its file). This is uniform across `File`, `Module`, `Function`, `Class`, `Constant`, `Enum`, `Interface`, `Trait`, `Protocol`, `Struct` (since 0.9.30). So `WHERE n.module STARTS WITH 'foo.bar'` works against any node label without branching.

When in doubt about a property's name or value shape, look at `graph_overview`'s `<prop sample="..." />` output (since 0.9.30) — every property carries one example value the agent can pattern-match on.

## Common Pitfalls

❌ Writing Cypher without calling `graph_overview` first — the agent guesses at node labels and property names, gets zero rows, and re-queries. Always start with an overview when entering an unfamiliar graph.

❌ `MATCH (n) RETURN n LIMIT 5` against a graph with millions of nodes — the planner may scan before applying LIMIT depending on shape. Always include a label filter and use specific properties in the RETURN.

❌ Re-querying with slightly different shapes when the first query returned 0 rows — usually means the property name is wrong (look at the schema, sample values are right there) or the label is wrong.

❌ Querying for entities by `name` when `id` (the qualified_name) is more specific. `MATCH (f:Function {name: "init"})` matches every `init` across every module; `MATCH (f:Function {id: "foo.bar.init"})` matches one.

✅ `graph_overview` first if you don't know the schema. The cost is one round trip; the value is correctly-shaped Cypher on the second attempt.

✅ Return specific properties (`RETURN n.name, n.module`) not whole nodes. Smaller context, faster cognition.

✅ Use `LIMIT` aggressively while exploring. Drop or raise it once you know the result shape.

## When `cypher_query` is the wrong tool

- **Don't know what types exist?** Call `graph_overview` first — same server, different surface. Cypher against unknown schema is shadow-boxing.
- **Want to read a function's source code?** `read_code_source(qualified_name=...)` collapses cypher-then-read into one call when you know the symbol; `read_source(file_path=...)` is the right shape for whole files.
- **Question is textual, not structural?** `grep` is the regex sweep across source roots. Cypher can find a function by name but not by "the string 'TODO' appears nearby."
- **Browsing a directory tree?** `list_source` returns a tree; Cypher doesn't.

## Format quirks

- Single-line queries are easiest to read; multi-line is fine when the WHERE/RETURN combination gets long.
- Parameters via `$name` aren't currently exposed through the MCP tool — inline values directly for now.
- `RETURN n, m AS alias` works; aliasing improves readability when joining tables in the head of your reasoning.
