---
name: save_graph
description: "Persist the active graph to its source `.kgl` file after mutating Cypher (CREATE / SET / DELETE / MERGE / REMOVE). TRIGGER after a chain of mutations the user explicitly wants kept — the in-memory graph won't survive a server restart otherwise. ALSO TRIGGER when the user says \"save\" or \"commit\" in the context of graph edits. SKIP for read-only sessions; the tool won't register unless the manifest sets `builtins.save_graph: true`. SKIP for exploratory mutations the user is iterating on — let them say \"save\" explicitly. SKIP entirely on disk-backed graphs (`storage=\"disk\"`) — those persist on every column flush; save_graph isn't registered for them."
applies_to:
  mcp_methods: ">=0.3.36"
  kglite_mcp_server: ">=0.9.31"
references_tools:
  - save_graph
references_arguments: []
auto_inject_hint: true
applies_when:
  tool_registered: save_graph
---

# `save_graph` methodology

## Overview

`save_graph` writes the in-memory graph back to the `.kgl` file the server booted from. It is the **persistence tool** — call it once after a coherent chain of mutations the user wants kept. The tool is **opt-in per manifest**: it doesn't register at all unless the YAML declares `builtins.save_graph: true`. This is deliberate — most kglite deployments are read-only, and exposing save by default would let agents make changes the operator didn't intend.

## Quick Reference

| Task | Approach |
|---|---|
| After CREATE/SET/DELETE the user wants kept | `save_graph()` — no args |
| User said "save" / "commit" | `save_graph()` |
| Each mutation in a long chain | NO — save once at the end, not per-statement |
| Read-only exploration | Don't call. Tool likely isn't registered anyway. |
| Disk-backed graph | Don't call. Tool isn't registered for disk graphs. |

## When the tool isn't registered

If the manifest has `builtins.save_graph: false` (or doesn't declare it at all), `save_graph` won't appear in `tools/list`. If the user asks to save changes and the tool isn't available, surface the manifest gate clearly:

> "The server doesn't have save enabled — the operator's manifest needs `builtins.save_graph: true` for me to persist changes."

Don't try to write the file directly via `read_source` / shell tools; the active graph lives in memory and the on-disk format is binary. The persistence path is `save_graph` or nothing.

## What gets saved

The entire active graph at the moment of the call. Specifically:

- All nodes (types, properties, including any added since boot)
- All edges (types and properties)
- All schema metadata (type-level introspection caches)

What does NOT get saved:

- Embedder state (those load lazily; if you've called `text_score` recently, the model is in memory but doesn't persist)
- Source-tool bindings (`source_roots`, watch handles — these are session state)
- Workspace state (clone inventory, active repo path — those live in their own files)

## When mutations don't need save_graph

If the operator's intent is **try-it-and-see** mutations (a Cypher CREATE to see what the schema looks like with a hypothetical node, or a SET to test a query against modified data), don't call `save_graph` proactively. The next server restart will discard the changes, which is the right behaviour. Save only when the user explicitly says "save" / "commit" / "make this permanent."

## Common Pitfalls

❌ Calling `save_graph` after every CREATE statement. Each call does a full file write; chain three CREATEs and you've done three full writes. Save once at the end.

❌ Calling `save_graph` proactively after a read query. Read queries don't mutate; save is a no-op but signals intent the user didn't have.

❌ Trying to use `save_graph` for an output you want at a different path. The tool saves to the boot path only — there's no `to_path` argument. For an alternate output, the agent would need to call the user's own scripting; not in scope for this tool.

❌ Expecting `save_graph` to register on a graph booted with `storage="disk"`. Disk-backed graphs persist column-by-column on every flush; there's no in-memory state to save explicitly.

✅ Save after a chain. CREATE → SET → SET → DELETE → `save_graph()`. One write, one persistent change.

✅ Surface the manifest gate when save isn't available. The operator can flip `builtins.save_graph: true` and restart; that's a clean recovery path.

## Error modes

- **"save_graph requires --graph mode (no source path bound)."** — the server booted in workspace mode (`--workspace dir/`) or source-root mode (`--source-root path/`); no `.kgl` to write back to. Expected.
- **OSError on write** — disk full, permission denied, file removed. Surface to the user verbatim; the tool returns the underlying error message.
- **Read-only graph** — if the operator booted with a graph marked read-only (rare; via `KnowledgeGraph(read_only=True)`), the in-memory mutations would have failed earlier. Save can't fix that.

## When `save_graph` is the wrong tool

- **Disk-backed mode** — disk graphs persist incrementally; no save call needed.
- **Workspace mode** — the active graph is a code-tree built from cloned source, not a `.kgl` file. The graph is rebuilt every time the workspace is re-activated; persistence isn't the right model here.
- **Read-only session** — if the operator's manifest doesn't enable save, the tool won't appear, and the session is read-only by design. Respect it.
