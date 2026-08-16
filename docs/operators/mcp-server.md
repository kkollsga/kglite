# MCP server

`kglite-mcp-server` exposes a KGLite graph over MCP stdio. The same Rust server
is available from `cargo install kglite-mcp-server` and inside the `kglite`
Python wheel.

```bash
kglite-mcp-server --graph /data/graph.kgl
kglite-mcp-server --selftest --graph /data/graph.kgl
```

The default is read-only and registers `ping`, `graph_overview`, and
`cypher_query`. A manifest can add source-root tools, parameterized Cypher,
skills, value codecs, an embedder, and CSV-over-localhost export. Point MCP
clients at the absolute executable path to avoid an older PATH-shadowing
installation.

## Pinning the tool surface

What a server exposes is the union of everything that registered: framework
builtins, the source tools the mode binds, KGLite's graph tools, manifest Cypher
tools, and routes that appear from a dependency or a mode change without the
manifest ever naming them. `extensions.tools_allow` inverts that: name the tools
the deployment is meant to expose, and everything else is hidden.

One long-standing case of this was closed upstream in mcp-methods 0.4.5: an
ambient `GITHUB_TOKEN` exported for unrelated reasons used to add the GitHub
tools to a server whose manifest never mentions GitHub. They now register only
when the manifest opts in with `builtins.github: true`. That removes one route
at the source; the allowlist is what bounds the rest.

```yaml
# /data/graph_mcp.yaml
name: My Graph
extensions:
  tools_allow:
    - cypher_query
    - graph_overview
    - ping
```

That server lists exactly those three tools, in every environment, and a route
arriving later — from a new dependency, an exported credential, or a mode change
— cannot widen the surface without an edit to the list. Hidden tools are
unlisted and rejected when called by name.

Details worth knowing before writing one:

- **Names are the final, agent-visible ones.** A `tools:` override that renames
  `ping` to `domain_ping` means the allowlist must say `domain_ping`.
- **Naming a tool that is not registered in this boot is harmless.** Conditional
  routes (`github_api` without the `builtins.github` opt-in or without a token,
  `load_graph` without `--writable`,
  `explore` on a non-code graph) can be listed safely, so one manifest works
  across environments.
- **It only removes.** Listing a tool some other rule hid — `repo_management` in
  a local workspace, a `hidden: true` override — does not bring it back.
- **The list is the whole surface**, not an addition to a default set: omit
  `ping` and the server has no `ping`. An explicit `tools_allow: []` is taken
  literally and leaves no tools at all.
- A manifest that configures `extensions.cypher_recipes` must list
  `list_recipe_queries` and `run_recipe_query`; omitting them is refused at boot
  rather than serving a catalog no agent can reach.
- A malformed value (not a list, or an element that is not a string) fails boot
  instead of being ignored — an allowlist that silently fails open is worse than
  none.

## Refreshing a rebuilt graph

The graph is read once at boot and served from memory, so a `.kgl` rebuilt by
another process — a nightly ingest, an external producer — does not reach a
running server on its own. `--graph` mode registers a no-argument
`reload_graph` tool (read-only servers included) that re-reads the served path
and reports the new node/edge counts. A failed re-read keeps the current graph
serving and returns the error; on a `--writable` server a reload discards
unsaved in-memory changes, so call `save_graph` first.

To make that automatic, put `graph_watch` in the manifest beside the `.kgl`:

```yaml
# /data/graph_mcp.yaml, next to /data/graph.kgl
name: My Graph
extensions:
  graph_watch: true
```

The server then watches the served file and re-reads it on the next graph tool
call after an external rewrite — no `reload_graph` call needed. It is off by
default and applies to `--graph` mode only. The reload is lazy (a query, not
the filesystem event, pays for it) and single-flight, so a producer that writes
several times between two queries costs one re-read. A re-read that fails keeps
the previous graph serving and attaches a warning to tool results; after three
consecutive failures the watcher stops retrying until a `reload_graph` call
succeeds. Single-file graphs only — a disk-graph *directory* logs a boot warning
and starts no watcher, and `reload_graph` remains its refresh path.

A read-only server does not hold the graph's single-writer lock, so a rebuilder
can open the served `.kgl` with `kglite.open(path)` and republish it in place
while the server keeps answering queries — and several read-only servers can
serve one file at once. Servers that can write the file (`--writable`, or
`builtins.save_graph: true`) keep the exclusive lock for their lifetime, as do
disk-graph directories in every mode, because their columns stay memory-mapped
while served.

## Writable workbench

```bash
kglite-mcp-server --graph /data/work.kgl --writable
kglite-mcp-server --graph /data/new.kgl --storage memory --writable
```

`--writable` enables mutation and the `load_graph`, `create_graph`, and
`save_graph_as` lifecycle tools. `--storage memory|mapped|disk` is required when
the `--graph` target does not yet exist, and on an existing graph it *converts*:
a memory-saved graph booted with `--storage mapped` comes up mapped. A disk
graph is a directory rather than a file, so converting into or out of disk mode
has no in-place form and is refused at boot naming `enable_disk_mode()`. Omit
the flag to serve whatever mode the graph recorded.
Keep read-only mode for untrusted agents and scope filesystem access with
manifest `source_root`/`source_roots`.

## Code intelligence

The generic KGLite server serves and queries code graphs but does not build
them. Use **codingest-mcp** for repository cloning, parsing, local watch mode,
and multi-revision code-graph construction; it embeds this same graph-serving
surface with the builder injected.

The complete manifest, skill, tool-gating, and client-registration reference is
the [MCP servers guide](../python/guides/mcp-servers.md).
