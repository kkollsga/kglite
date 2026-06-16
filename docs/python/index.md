# Python guide

The Python track. `pip install kglite`, then `import kglite`. This
is the headline distribution path — the wheel ships a compiled
extension (PyO3 wrapper over the pure-Rust `kglite` engine) plus a
pure-Python `kglite.mcp_server` and a console script
`kglite-mcp-server`.

If you're embedding the engine directly in a Rust binary, the
[Rust guide](../rust/index.md) is for you.

## Start here

- **[Getting started](getting-started.md)** — install, build your
  first graph, run a Cypher query, save / load a `.kgl`.
- **[Core concepts](core-concepts.md)** — nodes, relationships,
  storage modes, the selection model.

## How-to guides

```{toctree}
:maxdepth: 1
:caption: How-to guides

guides/index
guides/data-loading
guides/cypher
guides/mcp-servers
guides/mcp-skills
guides/durable-apps
guides/datasets
guides/blueprints
guides/querying
guides/traversal-hierarchy
guides/semantic-search
guides/spatial
guides/timeseries
guides/graph-algorithms
guides/import-export
guides/ai-agents
guides/code-tree
guides/okf
guides/recipes
guides/sec
```

## Python-specific topics

- **[Transactions](transactions.md)** — `begin()` / `commit()` /
  `rollback()`, snapshot isolation, OCC.
- **[Error handling](error-handling.md)** — typed exception
  hierarchy (`KgError` + 16 subclasses).
- **[Value projection](value-projection.md)** — NULL handling,
  CASE branches, optional property semantics.

## Migrations

- **[Neo4j → KGLite](migrations/neo4j-to-kglite.md)** — evaluate or
  adopt KGLite from an existing Neo4j database / driver code.
- **[MCP 0.6 → 0.9](migrations/mcp-0.6-to-0.9.md)** — older MCP
  server users.
- **[MCP pre-0.9.20](migrations/mcp-pre-0.9.20.md)** — the
  bundled-binary → Python-implementation switch.

```{toctree}
:hidden:

getting-started
core-concepts
transactions
error-handling
value-projection
migrations/neo4j-to-kglite
migrations/mcp-0.6-to-0.9
migrations/mcp-pre-0.9.20
```
