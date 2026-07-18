# KGLite

An embedded Cypher dialect for LLM-agent workloads. A knowledge graph that
runs inside your process — load data, query with Cypher, and hand the graph to
an agent via the bundled MCP server. The embedded path needs no database
service; one `.kgl` file can move between Python and Rust bindings.

The engine is a pure-Rust crate (`kglite`); the wheel
(`pip install kglite`) is a PyO3 wrapper around it. Bolt and MCP
protocol servers are standalone Rust binaries that wrap the same
engine. The `.kgl` file format is portable across all bindings.

## Start here

1. Install `kglite` (Python) or add the `kglite` crate (Rust).
2. Build a graph with inline records, DataFrames, Cypher, or a companion
   project such as codingest/kglite-datasets.
3. Query with Cypher or the fluent API; use `Session`/`Transaction` when a
   failed mutation must roll back.
4. Save a `.kgl`, or serve it through the CLI, MCP, or Bolt binary.

**[Python quickstart](python/getting-started.md)** ·
**[Cypher reference](reference/cypher-reference.md)** ·
**[Fluent API](reference/fluent-api.md)** ·
**[Rust quickstart](rust/index.md)** ·
**[Operators and deployment](operators/index.md)** ·
**[Reference](reference/index.md)** ·
**[0.13 → 0.14 migration](python/migrations/0.13-to-0.14.md)**

```{rubric} Cypher first
```

**Cypher** is the primary query surface — agents already know it, and
the engine targets an explicitly documented openCypher-compatible subset
(including three-valued NULL logic), checked with independently authored local
contracts and optional Neo4j differential runs. DataFrame
loaders `add_nodes()` / `add_connections()` exist to get bulk data
in; once it's in, you query with Cypher.

| | |
|---|---|
| Embedded, in-process | No database service; `import` and go |
| LLM-agent surface | Bundled MCP server + `describe()` schema for system prompts |
| Cypher subset, honest semantics | Querying + mutations + `text_score()` for semantic search |
| In-memory by default | Mapped + disk modes for Wikidata-scale; in-memory is the design centre |
| Label model | One primary type + optional secondary labels — see [multi-label rationale](concepts/multi-label-rationale.md) |
| One-file persistence | `.kgl` snapshots — copy, share, reload elsewhere |
| Rust-embeddable | Pure-Rust core; embed without PyO3 — see [Rust track](rust/index.md) |

```{rubric} Ecosystem
```

kglite is the engine. Two companion projects build graphs it serves — each
released and versioned on its own cadence:

- **[kglite](https://github.com/kkollsga/kglite)** — the embedded Cypher
  knowledge-graph engine (this project): graph + Cypher + fluent API + bundled
  MCP server.
- **[codingest](https://codingest.readthedocs.io)** — parses codebases into
  code graphs (14 languages, web-framework route detection). Build with it,
  query the `.kgl` here. Requires kglite ≥ 0.14.
- **[kglite-datasets](https://kglite-datasets.readthedocs.io)** —
  fetch-build-cache loaders for public registries (SEC EDGAR, Wikidata, Sodir).
- **[sonagram](https://sonagram.readthedocs.io)** — turns a local music
  library into a kglite knowledge graph via sonara audio analysis (tempo,
  energy, mood, key); AI agents curate playlists over it through a simple
  bundled skill and CLI (`pip install sonagram`).

**Coming from 0.13?** The code-graph builder and dataset loaders moved out of
the wheel in 0.14 — see the [0.13 → 0.14 migration guide](python/migrations/0.13-to-0.14.md).
Pin back anytime with `pip install "kglite<0.14"`.

```{rubric} Pick your track
```

- **[Python guide](python/index.md)** — `pip install kglite`, then
  `import kglite`. The headline track; covers data loading, Cypher,
  the MCP server, agents.
- **[Rust guide](rust/index.md)** — embed the engine in a Rust
  binary (`cargo add kglite`). For graph-as-a-library use cases
  without the Python wheel.
- **[Operators](operators/index.md)** — choose and run the CLI, MCP, or Bolt
  binary; storage, auth/TLS, and deployment guidance.
- **[Reference](reference/index.md)** — Python, Cypher, fluent, Rust, C ABI,
  and CLI reference surfaces.
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

reference/index
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
explanation/dependency-licenses
```
