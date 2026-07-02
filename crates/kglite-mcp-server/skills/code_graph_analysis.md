---
name: code_graph_analysis
description: "TRIGGER for any structural question about a codebase — what calls / defines / extends / imports X, where the mass is, how a subsystem hangs together. Map structure with the graph FIRST (graph_overview → cypher_query → explore), then drop to grep/read_source only to confirm a detail and cite a line. SKIP only for free-text content searches the graph can't answer (log strings, comments, TODOs) — those are grep. Never grep to discover what the graph already encodes (callers, callees, inheritance, members, imports)."
auto_inject_hint: true
references_tools:
  - cypher_query
  - graph_overview
  - explore
  - grep
  - read_source
applies_when:
  graph_has_node_type:
    - Function
    - Class
---

> **Code-graph workflow — do these in order:** **1** `graph_overview` (map the schema) · **2** `cypher_query` (structure: defs, callers, members, types, counts, paths) · **3** `grep` (literal text ONLY — log strings, comments, config keys) · **4** `read_source` / `read_code_source(qualified_name=…)` (read bodies, cite lines).
>
> **Never `grep` for a definition, caller, or call site — that is a `cypher_query` question.** The graph already resolved the cross-file relationships grep can't see.

# Code-graph analysis: the sequencing strategy

This is a **code-tree graph** — functions, classes, calls, imports, type
references and inheritance are all first-class nodes and edges. The graph
already encodes the structure you'd otherwise reconstruct by hand from grep
output. Use it in this order:

1. **`graph_overview`** — one call to learn the node types, their properties
   (look at `vals=` / `sample=`), and how they connect. Do this before
   reasoning about a graph you don't already know.
2. **`cypher_query`** — answer the structural question directly:
   - callers: `MATCH (c:Function)-[:CALLS]->(f:Function {name:'parse'}) RETURN c.qualified_name`
   - callees: `MATCH (f:Function {name:'parse'})-[:CALLS]->(c) RETURN c.name`
   - members: `MATCH (cls:Class {name:'Dataset'})-[:HAS_METHOD]->(m) RETURN m.name`
   - subclasses / impls: `MATCH (s)-[:EXTENDS|IMPLEMENTS]->(b:Class {name:'Base'}) RETURN s.name`
   - where the mass is: `MATCH (f:Function) RETURN f.module, count(f) ORDER BY count(f) DESC`
3. **`explore`** — when the question is "how does X work / where is Y / trace
   the Z flow" and you want ranked entry points + a 2-hop neighborhood +
   source in one call, instead of a grep→read chain.
4. **`grep` / `read_source`** — last, and only to (a) confirm a detail the
   graph pointed you at, (b) read the exact lines to cite, or (c) search
   free text the graph doesn't model (a log message, a comment, a config
   string). `read_code_source(qualified_name=...)` is the graph-native way
   to pull a symbol's body.

**The anti-pattern:** reaching for `grep "def foo"` / `grep "class Bar"` /
`grep "foo("` to discover definitions, callers, or call sites. Those are
`cypher_query`/`explore` questions — grep returns text lines with no
structure and misses cross-file relationships the graph has already resolved.

**Scope to real code.** Test, benchmark, and external/stdlib nodes share the
graph with library code. For "what matters" questions, filter them out — see
the `code_graph_views` methodology on the `cypher_query` tool for the exact
`is_test` / `is_benchmark` / `is_external` predicates and how to scope graph
algorithms with `{where: '...'}`.
