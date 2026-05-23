# KGLite

The embedded openCypher engine for LLM-agent workloads. A knowledge
graph that runs inside your Python process — load data, query with
Cypher, hand the graph to an agent via the bundled MCP server. No
server to run, no infrastructure to manage, one `.kgl` file to ship.

```{rubric} Cypher first
```

**Cypher** is the primary query surface — agents already know it, and
the engine targets honest openCypher semantics (three-valued NULL
logic, conformance-tested against Neo4j on demand). DataFrame
loaders `add_nodes()` / `add_connections()` exist to get bulk data
in; once it's in, you query with Cypher.

| | |
|---|---|
| Embedded, in-process | No server, no network; `import` and go |
| LLM-agent surface | Bundled MCP server + `describe()` schema for system prompts |
| Cypher subset, honest semantics | Querying + mutations + `text_score()` for semantic search |
| In-memory by default | Mapped + disk modes for Wikidata-scale; in-memory is the design centre |
| Single-label nodes | Each node has exactly one type — see [design decisions](explanation/design-decisions.md) |
| One-file persistence | `.kgl` snapshots — copy, share, reload elsewhere |

See [`ROADMAP.md`](https://github.com/kkollsga/kglite/blob/main/ROADMAP.md) for what's next (Bolt protocol, multi-language bindings).

**Requirements:** Python 3.10+ (CPython) | macOS (ARM), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

```bash
pip install kglite
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

getting-started
```

```{toctree}
:maxdepth: 2
:caption: How-to Guides

guides/index
guides/data-loading
guides/cypher
guides/mcp-servers
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
guides/recipes
```

```{toctree}
:maxdepth: 2
:caption: Explanation

core-concepts
explanation/architecture
explanation/design-decisions
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/cypher-reference
reference/fluent-api
autoapi/index
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
adding-a-storage-backend
adding-a-query-language
changelog
migrations/mcp-0.6-to-0.9
migrations/mcp-pre-0.9.20
```
