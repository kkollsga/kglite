---
name: initial_code_review
description: "TRIGGER for the first structural review of a known Function qualified_name. Use the code_review recipe's exact Function-only caller and bounded test paths. SKIP for other entity kinds, non-call dependencies, paths beyond five calls, or comprehensive impact analysis; express those questions with raw cypher_query."
references_tools:
  - run_recipe_query
  - cypher_query
applies_when:
  tool_registered: run_recipe_query
---

# Initial Function review

Use the `code_review` recipe directly; its query names and schemas are already
known here, so do not list the catalog first.

1. Call `run_recipe_query` with `recipe="code_review"`,
   `query="resolve_function"`, `variables={"qualified_name": name}`.
2. If `resolve_function` returns zero rows, report that the Function is
   missing. Do not interpret empty caller or test rows for an unresolved
   target.
3. Once resolved, call `direct_callers` and `affected_tests` with the same
   variables. Empty rows now mean that the exact stored relationship query
   matched nothing.

`direct_callers` covers only `Function-[:CALLS]->Function` edges.
`affected_tests` covers only test Functions with `is_test=true` that reach the
target through `CALLS*1..5`; it is not comprehensive impact analysis. For
other entity kinds, non-call dependencies, longer or unbounded traversals, or
broader impact questions, inspect `graph_overview` as needed and use raw
`cypher_query`.

Treat `stale_graph` as unusable evidence: correct the workspace build problem
and retry. On `result_limit_exceeded`, use the reported `limit` and
`observed_count` to design a genuinely narrower raw Cypher query; never treat
the error as a partial result.
