---
name: recipe_queries
description: "Use boot-validated named Cypher recipes for exact, repeated graph operations. TRIGGER when a domain skill names a recipe and query, or when the catalog may contain an exact operation matching the request. SKIP recipes for broader, differently scoped, or unmatched structural questions; write the needed read-only Cypher with cypher_query instead."
auto_inject_hint: true
references_tools:
  - list_recipe_queries
  - run_recipe_query
  - cypher_query
references_arguments:
  - list_recipe_queries.recipe
  - run_recipe_query.recipe
  - run_recipe_query.query
  - run_recipe_query.variables
  - run_recipe_query.include_cypher
  - cypher_query.query
applies_when:
  tool_registered: run_recipe_query
---

# Named Cypher recipe methodology

Recipe queries are boot-validated, parameterized, read-only operations. They
are shortcuts for exact common graph questions, not a replacement for Cypher
or a workflow engine.

## Choose the shortest correct path

1. **A domain skill names an exact recipe and query:** call
   `run_recipe_query` directly with that recipe, query, and its variables. Do
   not call `list_recipe_queries` first; the domain skill already selected the
   operation.
2. **You suspect a recipe exists but do not know its name:** call
   `list_recipe_queries()` once for compact recipe summaries. If one is a
   plausible exact match, call `list_recipe_queries(recipe=...)` to inspect
   only that recipe's query names and parameter schemas.
3. **The stored operation exactly matches the requested scope:** call
   `run_recipe_query`. Treat its structured columns, positional rows, and
   errors as the operation's complete contract.
4. **The request is broader, differently scoped, or unmatched:** fall back to
   raw `cypher_query` and express the requested read-only traversal directly.
   Call `graph_overview` first if the graph schema is not already known.

## Exact-match boundary

Recipe names and descriptions define closed operations. Do not treat a
Function-only caller query as callers of every entity kind, a bounded call
path as comprehensive impact analysis, or an empty result as evidence about
something the operation did not test. Do not modify recipe variables to carry
Cypher fragments, labels, property names, or wider traversal semantics.

Entity-oriented workflows may need to distinguish a missing target from a
real target with no matching neighbors. The owning domain skill decides that
sequence and interpretation, including any mandatory `resolve_*` preflight.
The generic recipe methodology must not guess target existence from an empty
row set or invent a preflight query.

Use `include_cypher=true` only when the stored query and bound parameters need
to be audited. It is not required for ordinary execution.
