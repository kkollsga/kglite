# Guides

KGLite has a set of how-to guides. Most projects only need three.

## Start here (the load-bearing path)

Every project that loads its own data and queries it goes through these
three, in this order:

| | |
|---|---|
| {doc}`data-loading` | Shape DataFrames, bulk-load with `add_nodes` / `add_connections`, conflict handling, hierarchies. The day-1 "I have a CSV, now what?" answer. |
| {doc}`cypher` | The query surface — MATCH/WHERE/RETURN, aggregations, subqueries, mutations. Every other guide leans on this one. |
| {doc}`mcp-servers` | Ship the graph to Claude / Cursor / any MCP-capable agent. The bundled `kglite-mcp-server` CLI + the YAML manifest for adding custom tools without forking. |
| {doc}`mcp-skills` | Teach agents *how and when* to use each tool with bundled/operator **skills** — methodology that injects into tool descriptions, gated per-graph. Use this instead of hand-rolling `instructions:`. |

## Add as needed

Domain-specific surfaces — pull them in when your data has the shape:

| Guide | Read this if… |
|---|---|
| {doc}`durable-apps` | …the graph is long-lived state your app reopens across runs. `open()` load-or-create lifecycle, checkpoint-on-close, and crash-safe `durable=True` write-ahead-log writes. |
| {doc}`okf` | …your "data" is a markdown knowledge base — an OKF bundle, a Claude memory dir, a skills folder, an Obsidian vault. Frontmatter → nodes, links → typed edges. |
| {doc}`spatial` | …your nodes have coordinates. R-tree indexing, distance-based filters, GeoJSON I/O. |
| {doc}`timeseries` | …property values change over time. Snapshot history, valid_at / valid_during temporal filters. |
| {doc}`semantic-search` | …you want fuzzy / meaning-based lookup. `text_score()` in Cypher, embedding model registration. |
| {doc}`graph-algorithms` | …you need PageRank, community detection, shortest paths, centrality. |
| {doc}`traversal-hierarchy` | …your graph has parent-child / ancestor structure. `set_parent_type`, `*` walks, hierarchical Cypher. |
| {doc}`datasets` | …you want pre-built graphs of public sources (Wikidata, Sodir). One-call lifecycle wrappers. |

## Power-user / less common

| Guide | When |
|---|---|
| {doc}`querying` | The fluent-API alternative to Cypher (`select` / `where` / `traverse` / `collect`). Useful for programmatic graph construction. |
| {doc}`blueprints` | Declarative graph schemas — nodes/edges defined once in a CSV-driven config. Best for repeated builds of the same shape. |
| {doc}`import-export` | Round-trip with Neo4j, JSON, N-Triples; CSV bulk export. |
| {doc}`ai-agents` | `describe()` XML schema for system prompts. Read this if you're building agent stacks beyond MCP. |
| {doc}`recipes` | Short snippets for "how do I do X" patterns that span multiple guides. |

## If you want to know *why*

Background reading — not required, but the design decisions explain
why APIs look the way they do:

- {doc}`/python/core-concepts` — storage modes (memory / mapped / disk), return types, the fluent / Cypher split.
- {doc}`/concepts/architecture` — Rust core + PyO3 bindings + petgraph, where each subsystem lives.
- {doc}`/concepts/design-decisions` — the label model, columnar storage, Cypher subset choices.
