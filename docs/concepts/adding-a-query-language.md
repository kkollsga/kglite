# Adding a query language

KGLite currently has two mature query surfaces: the Cypher implementation under
`graph/languages/cypher/` and the Rust fluent surface under
`graph/languages/fluent/` plus `api::fluent`. A new language must reuse their
storage-independent primitives and enter through the supported API boundary.

## Architecture

Cypher is organized as tokenizer/parser → AST → planner passes → executor. The
stable planner pass order and public names live in `planner/mod.rs::PASSES`.
Execution uses shared pattern/filter/traversal primitives under `graph/core/`
and reads/writes through `GraphRead`/`GraphWrite`, so one implementation spans
memory, mapped, and disk storage.

The binding-independent entry point is the canonical session pipeline:

```text
binding / protocol adapter
  → kglite::api::session::{execute_read, execute_mut}
  → parse + validate + optimize + execute
  → CypherResult / ExecuteOutcome
  → binding-specific result conversion
```

Python methods live under `crates/kglite-py/src/graph/pyapi/`; conversion and
language idioms stay in that wrapper. Generic pipeline logic belongs in
`kglite::api`, following [the boundary principle](../rust/boundary-principle.md).

## Checklist for a peer language

1. Define its grammar/AST and execution semantics in a new focused module.
2. Reuse `graph/core/` and storage traits; never branch on concrete backends in
   the language executor.
3. Decide whether translation into existing Cypher/shared AST shapes is simpler
   than a separate executor. Do not invent a generic `QueryLanguage` trait until
   two real implementations share a stable shape.
4. Add a core `api::*` entry point/options/result contract before exposing any
   Python, C ABI, MCP, or Bolt wrapper.
5. Keep parsing, errors, cancellation, deadlines, max-row budgets, write scope,
   provenance, and transaction behavior aligned with `api::session`.
6. Update API baselines, introspection/tool surfaces, stubs, reference docs, and
   `[Unreleased]` when the language is user-visible.

## Testing

- Parser/AST unit tests and executor tests near the implementation.
- End-to-end API tests plus memory/mapped/disk parity.
- Differential tests against the existing surface when two languages express
  the same query.
- For Cypher planner changes, register the pass in `PASSES`, document its
  precondition/rewrite/bailout, and add a triggering query to
  `tests/test_cypher_differential.py`.
- Run `scripts/cypher_pass_bisect.py` for optimizer divergences, then the full
  build/lint/test/performance gates.

Useful starting points are `languages/cypher/mod.rs`, `planner/mod.rs`,
`executor/mod.rs`, `graph/core/`, `api/session`, and `api/fluent`.
