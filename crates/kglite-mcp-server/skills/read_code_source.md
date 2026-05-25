---
name: read_code_source
description: "Read source code by qualified_name — resolves a graph-indexed symbol (Function / Class / Method) to its source slice with a file_path:line header. TRIGGER when you have a qualified name from a Cypher query (e.g. `kglite.cypher.parse`) and want the body, the agent needs to inspect a function it just located, or you're tracing the implementation of a symbol whose location you already know. ALSO TRIGGER as the second leg of cypher-then-read — `cypher_query` locates by predicate, `read_code_source` fetches the body. SKIP when you only have a file path (use `read_source` — that's the file-path-keyed tool, not this one). SKIP when the graph has no Function / Class types — most non-code-tree graphs (legal-corpus, o&g) don't have code, and this tool will return errors against them. The framework auto-gates this skill via `applies_when` on those deployments."
applies_to:
  mcp_methods: ">=0.3.36"
  kglite_mcp_server: ">=0.9.31"
references_tools:
  - read_code_source
references_arguments:
  - read_code_source.qualified_name
references_properties:
  - Function.id
  - Class.id
auto_inject_hint: true
applies_when:
  graph_has_node_type: [Function, Class]
---

# `read_code_source` methodology

## Overview

`read_code_source` is the **symbol-to-slice tool**: given a qualified name (e.g. `foo.bar.MyClass.method`), it resolves to the source file, returns the slice covering the symbol's body, and prepends a `file_path:line_start-line_end` header. It collapses the "Cypher query found a Function; now I want to see its code" workflow into one call — saving the agent one round trip vs. `cypher_query` for the file_path + `read_source` for the slice.

The tool only works against graphs that have **Function** and **Class** nodes (i.e. code-tree graphs built via `kglite.code_tree.build(...)`). It is not registered when those types are absent — and the framework auto-gates this skill via `applies_when: graph_has_node_type: [Function, Class]` so it doesn't appear in `prompts/list` for legal/o&g/data-only deployments.

## Quick Reference

| Task | Approach |
|---|---|
| Read a known symbol's body | `read_code_source(qualified_name='foo.bar.baz')` |
| Locate symbols, then read | `cypher_query` → list of qualified_names → `read_code_source(...)` per |
| Read a file by path (not a symbol) | Use `read_source` instead, not this tool |
| Read a method on a class | Same qualified-name shape: `pkg.module.ClassName.method_name` |
| Read just a few lines around a known symbol | `read_code_source(qualified_name=..., context=10)` (if supported) or fall back to `read_source` with `start_line`/`end_line` |

## Qualified-name resolution

Qualified names are the **node id** in code-tree graphs. Look at a Function or Class node's `id` property to see the canonical form:

```cypher
MATCH (f:Function) WHERE f.name = 'parse' RETURN f.id, f.file_path, f.line_number LIMIT 5
```

Typical shapes by language (kglite normalises across them):

- Python: `package.module.Class.method` or `package.module.function`
- Rust: `crate::module::Type::method` or `crate::module::function`
- TypeScript: `module/file.ClassName.method` or `module/file.function`
- Go: `package.Type.Method` or `package.Function`

Pass the exact id; partial / fuzzy matches are not supported. If you only have a name (not a qualified name), find it via Cypher first:

```cypher
MATCH (f:Function {name: 'parse'}) RETURN f.id, f.module
```

Pick the `id` that matches the module you want, then `read_code_source(qualified_name=that_id)`.

## When to prefer this over `read_source`

The split: `read_source` is **file-path-keyed**, `read_code_source` is **qualified-name-keyed**. Prefer this tool when:

- You found the symbol via a Cypher query (you already have the qualified_name).
- You're tracing call graphs and want each Function's body without computing file paths.
- The codebase has multiple files defining functions with the same short name — qualified-name resolution disambiguates correctly; grep + file-path doesn't.

Prefer `read_source` when:

- The file isn't a Function/Class/Method/Module entity (configs, build scripts, docs).
- You have the file path from outside the graph (e.g. a stack trace).
- You want a window that doesn't align with any single symbol.

## Common Pitfalls

❌ Calling `read_code_source` with a file path instead of a qualified name. The tool takes qualified_name, not file_path; this is `read_source`'s argument. Symptom: "no node found with id 'src/foo.py'".

❌ Calling it against a non-code graph (legal, o&g) — the tool isn't registered, and the call returns an error. The `applies_when:` gate suppresses this skill from `prompts/list` for those graphs, so the agent shouldn't see it; if you do see it, the framework's predicate evaluator returned the wrong answer.

❌ Searching for a partial qualified name (`MyClass.method` without the leading package). Resolution is exact; provide the full id.

❌ Calling `read_code_source` repeatedly for every Function in a result set to read 50 bodies. That's 50 round trips; consider whether Cypher returning each Function's `docstring` property is enough for your current question.

✅ Cypher first to find the qualified_name, then `read_code_source`. The two-step is the design contract.

✅ Use `f.id` from a Cypher result directly. The `id` property on graph nodes is the qualified_name kglite indexes by.

✅ When a qualified-name lookup returns "not found", check whether the entity exists at all via `cypher_query`. If it does exist but the resolver missed it, the entity's `id` might differ from what you typed (case sensitivity, leading-underscore differences) — re-check the exact `id` value.

## When `read_code_source` is the wrong tool

- **Graph has no Function / Class types.** Tool isn't registered; skill is filtered out via `applies_when`. Legal-corpus, o&g, and other non-code graphs fall here.
- **You only have a file path.** Use `read_source(file_path=...)` — the framework's file-system reader.
- **You want to read the whole file (not just a symbol).** `read_source(file_path=...)` with no line range, or with the full file's range.
- **You want regex search inside the file.** `grep(pattern=...)` for the search, then `read_source` or `read_code_source` for the targeted read of any matches.

## Behaviour notes

- **Return shape.** A `file_path:line_start-line_end` header line, then the source slice. The header lets the agent quote a specific position in further messages without re-resolving.
- **Workspace mode.** When the active code-tree was built from a repo cloned via `repo_management`, `read_code_source` resolves against that repo's source files. After `repo_management('other/repo')`, the resolver switches to the new clone — the qualified-name namespace also changes (it's tied to the repo's module structure).
- **Disk-backed graphs.** Code-tree graphs aren't typically stored as disk-backed graphs (those exist for wikidata-scale entity data). If you somehow see a code-tree graph in disk mode, the resolver works the same; the underlying lookup goes through the index, not a full scan.
