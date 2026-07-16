---
name: code_graph_views
description: "TRIGGER when a code-graph query needs a library-only view (exclude tests/benchmarks/external code), or needs to predicate on a function's parameters / a class's fields. Use the is_test / is_benchmark / is_external boolean filters, scope graph algorithms with {where:'...'}, and parse_json() to query JSON-string properties. SKIP for plain structural queries that don't need provenance filtering (a simple callers/members lookup is just cypher_query)."
auto_inject_hint: true
references_tools:
  - cypher_query
applies_when:
  graph_has_node_type:
    - Function
    - Class
---

> **Code-graph workflow — do these in order:** **1** `graph_overview` (map the schema) · **2** `cypher_query` (structure: defs, callers, members, types, counts, paths) · **3** `grep` (literal text ONLY — log strings, comments, config keys) · **4** `read_source` / `read_code_source(qualified_name=…)` (read bodies, cite lines).
>
> **Never `grep` for a definition, caller, or call site — that is a `cypher_query` question.** The graph already resolved the cross-file relationships grep can't see.

# Code-graph views: provenance filters & structured fields

A code graph mixes library code with test, benchmark, generated, and
external/stdlib nodes. These properties let you carve out the view you want
(all are real booleans on in-repo nodes — `false`, not null — so the obvious
filter works):

## Library-only views

| Want | Predicate |
|------|-----------|
| Exclude test code | `WHERE n.is_test = false` (on `Function` / `File` / `Module` / `Class`) |
| Exclude benchmarks | `WHERE n.is_benchmark = false` (path-based: `asv_bench/`, `benchmarks/`, `bench/`) |
| Internal classes only | `WHERE c.is_external = false` (external stdlib/3rd-party bases are `is_external = true`) |
| Skip generated files | `WHERE f.is_generated = false` (on `File`; generated/minified files) |
| Library functions, no noise | `WHERE n.is_test = false AND n.is_benchmark = false` |

```cypher
-- The real library hubs, test helpers excluded
MATCH (caller:Function)-[:CALLS]->(f:Function)
WHERE f.is_test = false AND f.is_benchmark = false
RETURN f.qualified_name, count(caller) AS callers
ORDER BY callers DESC LIMIT 15
```

## Scope graph algorithms to a subgraph

`pagerank` / `degree` / `betweenness` / `closeness` / `louvain` / `leiden` /
`label_propagation` take `{node_type, where}` so test/external nodes don't
pollute centrality and community results:

```cypher
CALL pagerank({node_type:'Function', connection_types:'CALLS',
               where:'n.is_test = false AND n.is_external = false'})
YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 15
```

`where` is a predicate over the node variable `n` (full WHERE grammar); only
edges with both endpoints in scope are traversed.

## Query structured fields (parameters / fields)

`Function.parameters`, `Function.signature`, and `Class.fields` are stored as
JSON strings. `parse_json(s)` parses one into a structured map/list so you can
predicate over it:

```cypher
-- Functions that take a parameter typed `Dataset`
MATCH (f:Function)
WHERE any(p IN parse_json(f.parameters) WHERE p.type_annotation = 'Dataset')
RETURN f.qualified_name
```

Each parsed parameter is a map with keys `name` / `type_annotation` /
`default` / `kind`. Bracket-subscript reaches a field after a list index:
`parse_json(f.parameters)[0]['name']`.

## Identifiers

`qualified_name` and `module` are the canonical dotted paths (e.g.
`xarray.core.dataset.open_dataset`) and round-trip into
`read_code_source(qualified_name=...)`. `file_path` (`xarray/core/dataset.py`)
is the clean key for path-prefix scoping (`WHERE n.file_path STARTS WITH
'xarray/core/'`).
