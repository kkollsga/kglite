# Adding a query language

KGLite's query surface lives under `src/graph/languages/`. Today there
is one concrete implementation — **Cypher** — plus scaffolding for a
second (`languages/fluent/`). This guide walks through how Cypher is
put together and how a hypothetical peer language (SPARQL, GraphQL,
GQL, a custom DSL) would slot into the same tree.

This document is the companion to [adding a storage backend] — storage
layers handle where bytes live; the languages layer handles how a user
asks about them.

[adding a storage backend]: adding-a-storage-backend.md

## TL;DR

A new query language is a new directory under `src/graph/languages/`
containing its own `parser/`, `planner/` (optional), `executor/`, and a
public entry function callable from `pyapi/`. Most of the heavy work
— pattern matching, filtering, traversal, aggregation, value coercion
— already exists in `src/graph/core/` and is designed to be shared
across languages.

The integration checklist:

1. **Pick a name** — `languages/<name>/`.
2. **Parse** — take the query string, emit an AST.
3. **Plan** (optional) — rewrite the AST for cost or push predicates.
4. **Execute** — walk the AST against `&impl GraphRead`, calling into
   `core::pattern_matching`, `core::filtering_methods`,
   `core::traversal_methods`, and the other shared primitives.
5. **Expose** — add a `#[pyo3] fn kg_<name>(...)` to `pyapi/kg_core.rs`
   (or a dedicated `pyapi/kg_<name>.rs`) that calls the language's
   entry point and returns a `ResultView`.

No storage-layer changes should be needed — that's the whole point of
the `GraphRead` / `GraphWrite` split.

## The `languages/` umbrella

```
src/graph/languages/
├── mod.rs
├── cypher/            # The concrete implementation
│   ├── mod.rs         # Re-exports + estimate_match_rows + explain helper
│   ├── tokenizer.rs
│   ├── ast.rs
│   ├── parser/        # String → AST
│   ├── planner/       # AST → optimised AST (join order, index choice, fusion)
│   ├── executor/      # AST → rows (the work)
│   ├── window.rs      # Window-function evaluator
│   ├── result.rs      # CypherResult shape
│   └── py_convert.rs  # Value → Python object converters
│
└── fluent/            # Scaffolding only — no Rust-side code today
```

`languages/fluent/` is deliberately empty: the fluent chain (the
`kg.select(type).where(...).traverse(...)` API) is implemented entirely
in `pyapi/kg_fluent.rs` on top of `CowSelection`. The directory exists
to document that fluent *is* a query-interface language, and to reserve
the slot for a future Rust-side extraction if a non-Python fluent
caller materialises (say, a SPARQL frontend that reuses the chain as
its execution model).

There is **no `QueryLanguage` trait** yet — Cypher is the single
concrete example, and a trait invented for one impl is premature
abstraction. The first candidate for a trait would be an AST-level
common shape (pattern, filter, project), with `core/` primitives as
the shared execution substrate.

## Anatomy of `languages/cypher/`

### `tokenizer.rs`

Lexer: string → stream of `Token`s. Handles the usual Cypher syntax
— identifiers, keywords, literals, punctuation, comments. No graph
knowledge.

### `ast.rs`

AST node definitions: `Query`, `MatchClause`, `WhereClause`,
`ReturnClause`, and the expression tree (`Expr`, `Value`, `Pattern`,
`RelationshipPattern`). Pure data. Also hosts small helpers that
operate on the AST shape — `is_aggregate_expression`,
`collect_node_types`, etc.

### `parser/`

Submodules: `mod.rs` (entry + token helpers), `match_pattern`,
`predicate`, `expression`, `clauses`. All five share a single
`CypherParser` struct via multiple `impl CypherParser` blocks that
Rust merges at codegen — this keeps per-clause parsing local to its
own file without duplicating state.

The MATCH clause **delegates pattern parsing to
`core::pattern_matching::parse_pattern()`** — the same parser the
fluent API uses. Anything that matches a pattern in KGLite goes
through that one parser.

### `planner/`

Optional AST → AST rewrite pass. Submodules: `join_order`,
`index_selection`, `cost_model`, `fusion`, `simplification`.
`fusion.rs` is the heaviest — it rewrites chains of MATCH/WHERE/RETURN
into fused evaluators that skip intermediate materialisation.
`simplification.rs::rewrite_text_score` converts `ORDER BY
text_score(...)` into an index-backed scan.

A new language may skip the planner entirely and go directly from
parser to executor. The Cypher planner exists because Cypher queries
benefit from join-order + index-choice optimisations; a simpler
language may not.

### `executor/`

Nine files (`mod.rs`, `match_clause.rs`, `where_clause.rs`,
`expression.rs`, `return_clause.rs`, `call_clause.rs`, `write.rs`,
`helpers.rs`, `tests.rs`) sharing a single `CypherExecutor<'a>` struct
via merged `impl` blocks. Each clause file adds its own methods on the
executor. Shared helpers (arithmetic, property resolution, type
coercion, `return_item_column_name`) live in `helpers.rs`.

The executor takes `&impl GraphRead` for reads and `&mut impl
GraphWrite` for writes — it doesn't match on the backend enum. Every
`MATCH`/`RETURN` call goes through the trait surface.

### `window.rs`

Window-function evaluator (`SUM() OVER (...)`, `LAG()`, `LEAD()`,
`ROW_NUMBER()`, etc.). Lives outside the main executor because window
evaluation runs over post-projection row buffers, not against the
graph.

### `result.rs` + `py_convert.rs`

`CypherResult` is the internal row-buffer + metadata carrier.
`py_convert.rs` converts `Value`s to Python objects at the PyO3
boundary.

## How Cypher integrates with `core/`

The `core/` module is KGLite's shared query substrate. Every language
is expected to call into it rather than re-implement primitives.

| Core module | What Cypher uses it for |
|---|---|
| `pattern_matching/` | MATCH-clause pattern parser + `PatternExecutor` evaluation |
| `filtering_methods.rs` | Property predicates, value comparisons (shared with fluent) |
| `traversal_methods.rs` | Outgoing / incoming / undirected edge walks |
| `graph_iterators.rs` | Node and edge iteration primitives with per-backend fast paths |
| `data_retrieval.rs` | Per-node property fetch, typed accessors |
| `calculations.rs` | Arithmetic, math functions |
| `value_operations.rs` | Type coercion, comparison, null handling |
| `statistics_methods.rs` | Aggregation primitives (count/sum/avg/min/max) |

A peer language writes its executor against this palette. The shared
primitives are the reason Cypher and fluent agree on semantics —
they call the same filtering / traversal code.

## Entry point — how `kg.cypher(...)` flows

```
kg.cypher("MATCH (p:Person) RETURN p.name")
    │
    ▼
pyapi/kg_core.rs::cypher (PyO3 method on KnowledgeGraph)
    │
    ▼
languages/cypher::parse_cypher  →  Ast::Query
    │
    ▼
languages/cypher::planner::optimize  →  Ast::Query (rewritten)
    │
    ▼
languages/cypher::executor::CypherExecutor::execute
    │  ├─ match_clause::execute_match          (calls core::pattern_matching)
    │  ├─ where_clause::evaluate_where         (calls core::filtering_methods)
    │  ├─ return_clause::project_return        (calls core::value_operations)
    │  └─ helpers::return_item_column_name
    ▼
CypherResult → ResultView  (pyapi/result_view.rs)
    │
    ▼
Python: list of dicts, pandas DataFrame, GeoDataFrame
```

`cypher` is a single entry point. A peer language adds its own — e.g.
`pyapi/kg_core.rs::sparql`, calling `languages::sparql::parse` →
`execute` → `ResultView`. The `ResultView` shape is language-agnostic
(rows + column names), so the Python-facing return type stays uniform.

## Adding a peer language (speculative)

Suppose you want to add SPARQL. Concrete steps:

1. `src/graph/languages/sparql/` with `mod.rs`, `parser.rs`,
   `executor.rs`. Optional `planner.rs` if SPARQL-specific optimisations
   are worth it.
2. In `parser.rs`, convert SPARQL into an AST. Wherever possible,
   translate patterns (`?s ?p ?o`) into the same `Pattern` /
   `PropertyMatcher` types Cypher uses — the `core::pattern_matching`
   parser can then be reused for the hot path.
3. In `executor.rs`, walk the AST against `&impl GraphRead`,
   calling the same `core/` primitives the Cypher executor uses.
4. Add `#[pyo3] fn sparql(&self, query: &str, ...) -> PyResult<Py<ResultView>>`
   to `pyapi/kg_core.rs`.
5. Add a parity oracle alongside `tests/test_storage_parity.py` that
   runs the same graph through Cypher and SPARQL for an equivalent query
   and asserts row-set equivalence.

**What you don't need**: storage-backend changes, new Python `ResultView`
types, or changes to the cypher parser. The `languages/` boundary is
designed so a peer language lands as a new sibling directory with no
churn in its neighbours.

## Testing

Each language ships its own test module. For Cypher, tests live in
`src/graph/languages/cypher/executor/tests.rs` (unit) +
`tests/test_cypher.py` (Python end-to-end) + the `test_phaseN_parity.py`
files (cross-storage oracle).

A new language should mirror this layout:

- Unit tests alongside the executor (`src/graph/languages/<name>/tests.rs`
  or inline `#[cfg(test)] mod tests`).
- Python end-to-end tests exercising the full parser → executor →
  `ResultView` path.
- A cross-storage parity oracle if the language is read-only —
  memory / mapped / disk must agree.

## Reading more

- `src/graph/languages/cypher/mod.rs` — the Cypher dispatcher + re-exports.
- `src/graph/languages/cypher/executor/mod.rs` — the executor entry point; scroll the file for the clause-submodule glob imports.
- `src/graph/core/` — the shared primitives; read `mod.rs` files for each sub-module's public API.
- `src/graph/pyapi/kg_core.rs` — where `cypher` is exposed to Python.
- `dev-documentation/todo.md` Phase 8 + Phase 9 Report-outs (gitignored,
  repo-checkout only) — the decisions behind the `languages/` umbrella
  and the executor's clause-per-file split.
