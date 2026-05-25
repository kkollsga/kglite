---
name: explore
description: "TRIGGER when the user (or an Explore subagent) asks a `how does X work` / `where is Y` / `trace the Z flow` question against a code-tree graph. ONE call returns entry points (ranked by name + signature + docstring match), a 2-hop neighborhood, and grouped source slices — replacing the typical grep + read chain. SKIP for graph-schema questions (use graph_overview), exact symbol lookups when you already have a qualified_name (use read_code_source), or non-code graphs (no Function/Class nodes)."
applies_to:
  mcp_methods: ">=0.3.36"
  kglite_mcp_server: ">=0.9.34"
references_tools:
  - explore
references_arguments:
  - explore.query
  - explore.max_entities
  - explore.max_depth
  - explore.include_source
auto_inject_hint: true
applies_when:
  graph_has_node_type:
    - Function
    - Class
---

# `explore` methodology

## Overview

`explore` is the **one-call codebase exploration tool** for code-tree graphs.
It ranks Function/Class/Interface/Struct/Trait/Protocol/Enum nodes by lexical
match against your query (name > signature > docstring), takes the top
entries, 2-hop traverses CALLS / USES_TYPE / HAS_METHOD / DEFINES /
REFERENCES_FN, and returns a markdown report with three sections:

- **Entry points** — top-ranked symbols with location + signature.
- **Related** — neighborhood reachable within `max_depth` hops.
- **Source** — grouped, contiguous source slices for the entry points only
  (the related list deliberately omits source to keep the response sized).

Designed for the "how does X work in this codebase" question that would
otherwise turn into a chain of grep + read calls.

## Quick Reference

| Task | Call |
|---|---|
| "How does auth work?" | `explore(query="authenticate")` |
| "Where is the route handler for /users?" | `explore(query="users", max_entities=5)` |
| "What calls `parse_query`?" | `explore(query="parse_query")` — neighbors include callers |
| Smaller, no-source response | `explore(query="X", include_source=false)` |
| Single-hop only | `explore(query="X", max_depth=1)` |

## Writing good queries

- **Short, specific terms beat sentences.** `explore("authenticate")`
  beats `explore("how does the system verify user credentials")`. The
  ranker is lexical, not semantic; multi-word queries match substrings,
  not concepts.
- **A symbol name you've already seen always works.** If `cypher_query`
  returned a Function called `parse_query`, then `explore("parse_query")`
  pivots to its neighborhood.
- **Stack queries:** start broad (`explore("auth")`), then narrow once
  you have a target symbol (`explore("authenticate_user")`).

## When NOT to use

- **Graph-schema questions:** "what node types exist" → `graph_overview`.
- **Exact symbol lookup:** if you already have a qualified_name and want
  just that one body, `read_code_source` is one fewer hop.
- **Non-code graphs:** explore only ranks code-tree node types. On a
  domain graph (legal, sodir, wikidata) it emits a "no match" message
  rather than degrade silently.
- **Structural queries:** "find every Function that returns Result<T>" is
  a Cypher question, not an explore question.

## Tuning knobs

- `max_entities` (default 10): raise to widen the entry-point list; lower
  for tighter focus.
- `max_depth` (default 2): 1 = direct neighbors only (CALLS/USES_TYPE),
  2 = neighbor's neighbors. Past 3 the neighborhood explodes on
  well-connected code.
- `include_source` (default true): turn off when you just want the
  symbol+location list (faster, smaller response).

## Source-budget cap

`explore` truncates the source block at ~32 KB to keep the response
from blowing the LLM context. If you see the `… truncated (source budget
reached) …` marker, narrow the query or set `include_source=false` and
follow up with `read_code_source` for specific entities.
