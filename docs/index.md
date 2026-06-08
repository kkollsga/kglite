# KGLite

The embedded openCypher engine for LLM-agent workloads. A knowledge
graph that runs inside your Python process — load data, query with
Cypher, hand the graph to an agent via the bundled MCP server. No
server to run, no infrastructure to manage, one `.kgl` file to ship.

The engine is a pure-Rust crate (`kglite`); the wheel
(`pip install kglite`) is a PyO3 wrapper around it. Bolt and MCP
protocol servers are standalone Rust binaries that wrap the same
engine. The `.kgl` file format is portable across all bindings.

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
| Single-label nodes | Each node has exactly one type — see [design decisions](concepts/design-decisions.md) |
| One-file persistence | `.kgl` snapshots — copy, share, reload elsewhere |
| Rust-embeddable | Pure-Rust core; embed without PyO3 — see [Rust track](rust/index.md) |

```{rubric} Pick your track
```

- **[Python guide](python/index.md)** — `pip install kglite`, then
  `import kglite`. The headline track; covers data loading, Cypher,
  the MCP server, datasets, agents.
- **[Rust guide](rust/index.md)** — embed the engine in a Rust
  binary (`cargo add kglite`). For graph-as-a-library use cases
  without the Python wheel.
- **[Operators](operators/index.md)** — running the Bolt server
  (Neo4j wire compat for cluster-aware drivers).
- **[Reference](reference/cypher-reference.md)** — Cypher subset
  reference + fluent API reference + auto-generated Python API.
- **[Concepts](concepts/architecture.md)** — architecture +
  design decisions + contributor docs.

```{toctree}
:maxdepth: 2
:caption: Python guide
:hidden:

python/index
```

```{toctree}
:maxdepth: 2
:caption: Rust guide
:hidden:

rust/index
```

```{toctree}
:maxdepth: 1
:caption: Operators
:hidden:

operators/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/cypher-reference
reference/fluent-api
autoapi/index
```

```{toctree}
:maxdepth: 1
:caption: Concepts
:hidden:

concepts/index
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

contributing
changelog
```
