# KGLite

A knowledge graph that runs inside your Python process. Load data, query with Cypher, do semantic search — no server, no setup, no infrastructure.

```{rubric} Two APIs
```

Use **Cypher** for querying, mutations, and semantic search. Use the **fluent API** (`add_nodes` / `add_connections`) for bulk-loading DataFrames. Most agent and application code only needs `cypher()`.

| | |
|---|---|
| Embedded, in-process | No server, no network; `import` and go |
| In-memory | Persistence via `save()`/`load()` snapshots |
| Cypher subset | Querying + mutations + `text_score()` for semantic search |
| Single-label nodes | Each node has exactly one type |
| Fluent bulk loading | Import DataFrames with `add_nodes()` / `add_connections()` |

**Requirements:** Python 3.10+ (CPython) | macOS (ARM/Intel), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

```bash
pip install kglite
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started
core-concepts
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/cypher
guides/data-loading
guides/blueprints
guides/querying
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
:caption: Reference

reference/cypher-reference
reference/fluent-api
autoapi/index
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
changelog
```
