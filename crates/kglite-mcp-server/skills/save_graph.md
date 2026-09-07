---
name: save_graph
description: "Persist the active graph to its bound `.kgl` file after mutating Cypher (CREATE / SET / DELETE / MERGE / REMOVE). TRIGGER after a chain of mutations the user explicitly wants kept. ALSO TRIGGER when the user says \"save\" or \"commit\" in the context of graph edits. SKIP for exploratory mutations the user is iterating on. The tool registers when `builtins.save_graph: true` or writable mode is enabled."
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

`save_graph` writes the active graph back to its bound `.kgl` file. It is the **persistence tool** — call it once after a coherent chain of mutations the user wants kept. The tool registers when the manifest declares `builtins.save_graph: true` or the server is write-enabled with `--writable` / `extensions.writable: true`. The `builtins.save_graph` switch alone exposes save for already-dirty or boot-configured graph state; it does not authorize Cypher mutations.

## Quick Reference

| Task | Approach |
|---|---|
| After CREATE/SET/DELETE the user wants kept | `save_graph()` — no args |
| User said "save" / "commit" | `save_graph()` |
| Each mutation in a long chain | NO — save once at the end, not per-statement |
| Read-only exploration | Don't call. Tool likely isn't registered anyway. |
| Any storage mode with changes to publish | `save_graph()` — persistence still requires an explicit save |

## When the tool isn't registered

If neither `builtins.save_graph: true` nor writable mode enables the route, `save_graph` won't appear in `tools/list`. If the user asks to save changes and the tool isn't available, surface the gate clearly:

> "The server doesn't have save enabled — the operator can set `builtins.save_graph: true`, or enable writable mode when graph mutations are intended."

Don't try to write the file directly via `read_source` / shell tools; the active graph lives in memory and the on-disk format is binary. Use `save_graph` for the bound path or `save_graph_as` on a write-enabled server for another path.

## What gets saved

The entire active graph at the moment of the call. Specifically:

- All nodes (types, properties, including any added since boot)
- All edges (types and properties)
- All schema metadata (type-level introspection caches)

What does NOT get saved:

- Embedder state (those load lazily; if you've called `text_score` recently, the model is in memory but doesn't persist)
- Source-tool bindings (`source_roots`, watch handles — these are session state)
- Workspace state (clone inventory, active repo path — those live in their own files)

## When not to publish mutations

If the operator's intent is **try-it-and-see** mutations (a Cypher CREATE to see what the schema looks like with a hypothetical node, or a SET to test a query against modified data), don't call `save_graph` proactively. The next server restart will discard the changes, which is the right behaviour. Save only when the user explicitly says "save" / "commit" / "make this permanent."

## Common Pitfalls

❌ Calling `save_graph` after every CREATE statement. Each call does a full file write; chain three CREATEs and you've done three full writes. Save once at the end.

❌ Calling `save_graph` proactively after a read query. Read queries don't mutate; save is a no-op but signals intent the user didn't have.

❌ Trying to pass an output path to `save_graph`. The tool publishes to the bound path and has no `to_path` argument. On a write-enabled server, use `save_graph_as` for another path.

❌ Assuming a disk-backed mutation is already published. Disk storage changes session state; call `save_graph` to publish the coherent change set just as you would for other storage modes.

✅ Save after a chain. CREATE → SET → SET → DELETE → `save_graph()`. One write, one persistent change.

✅ Surface the manifest gate when save isn't available. The operator can flip `builtins.save_graph: true` and restart; that's a clean recovery path.

## Sharing the file with other servers

The server holds the cross-process writer lease only **between your first unsaved change and the save that publishes it**. Outside that window the `.kgl` is lockable by anybody — other MCP clients on the same file, an external rebuilder, the `kglite` CLI. Two consequences for how you work:

- Don't sit on unsaved changes. While you hold them, every other client's write is refused by name. Save (or discard) when the chain is done rather than leaving the graph dirty across a long exploration.
- A refused write is never a lost write. Both refusals below say so explicitly, because the reflex on reading "refused" is to assume the mutation half-landed. It did not: the refusals happen before anything changes, or roll back to the state before the attempt.

## Error modes

- **"save_graph requires --graph mode (no source path bound)."** — the server booted in workspace mode (`--workspace dir/`) or source-root mode (`--source-root path/`); no `.kgl` to write back to. Expected.
- **"…is open for writing by …"** (a write, not a save) — another client is mid-write on the same file. Nothing changed here and the graph is still fully readable; keep querying. Retry once that client saves, and call `reload_graph` to pick up what it wrote. Never delete the `.lock` file to clear this: the lock lives in the OS, and deleting the file only removes the record of who holds it.
- **"…changed on disk since you loaded it"** (a save) — somebody else republished the file after this server read it, so saving would overwrite their version. Your unsaved changes are intact and still queryable. Two ways out, and they are alternatives — **there is no merge**:
  - `save_graph_as` to a different path keeps your work (and releases the original file, so the other writer is unblocked);
  - `reload_graph(discard_unsaved=true)` throws your work away and serves the file as it is on disk.
  Choose deliberately, and tell the user which one you took.
- **"…has unsaved changes"** (from `reload_graph` / `load_graph` / `create_graph`) — you asked to replace the active graph while holding work that only exists in memory. `save_graph` first to keep it, or `reload_graph(discard_unsaved=true)` to drop it. That flag is the *only* spelling for "throw it away"; the other two tools deliberately have no discard argument.
- **OSError on write** — disk full, permission denied, file removed. Surface to the user verbatim; the tool returns the underlying error message.
- **Read-only graph** — if the operator booted with a graph marked read-only (rare; via `KnowledgeGraph(read_only=True)`), the in-memory mutations would have failed earlier. Save can't fix that.

## When `save_graph` is the wrong tool

- **Workspace mode** — the active graph is a code graph built from cloned source, not a `.kgl` file. The graph is rebuilt every time the workspace is re-activated; persistence isn't the right model here.
- **Read-only session** — if the operator's manifest doesn't enable save, the tool won't appear, and the session is read-only by design. Respect it.
