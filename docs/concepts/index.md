# Concepts

Architecture, design rationale, and contributor deep-dives.
Audience: contributors, curious users wondering "why is it built
this way," embedders trying to predict behavior.

For day-to-day usage see the [Python guide](../python/index.md)
or [Rust guide](../rust/index.md).

## Architecture + design

- **[Architecture](architecture.md)** — layered diagram, the
  petgraph storage, indexes, the query pipeline.
- **[Design decisions](design-decisions.md)** — the label
  model, why Cypher over SQL, transaction model.
- **[Multi-label rationale](multi-label-rationale.md)** — why KGLite uses an
  immutable primary type plus optional secondary labels.
- **[Cypher conformance](cypher-conformance.md)** — what subset
  of Cypher kglite supports + three-valued logic + the
  conformance harness.
- **[Concurrency](concurrency.md)** — direct graph ownership, shared sessions,
  optimistic commits, snapshots, and disk-backed generation semantics.
- **[Generated project facts](../_generated/project-facts.md)** — workspace,
  package, CI, storage, format, and benchmark facts derived from source.

## Extending

- **[Adding a storage backend](adding-a-storage-backend.md)** —
  trait surface + the in-memory / mapped / disk patterns.
- **[Adding a query language](adding-a-query-language.md)** —
  Cypher and the fluent API are peers under `languages/`; the
  same shape works for SPARQL or a custom DSL.

```{toctree}
:hidden:

architecture
design-decisions
multi-label-rationale
cypher-conformance
concurrency
adding-a-storage-backend
adding-a-query-language
../_generated/project-facts
```
