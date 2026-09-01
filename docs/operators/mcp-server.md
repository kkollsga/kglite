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

A `--graph` server serving a regular `.kgl` file re-reads it by itself. Every
graph tool call first `stat`s the served path, and when the file's identity
(length, mtime, device/inode) differs from the one the in-memory graph was
loaded — or last saved — from, the server re-reads it through the normal open
path before answering. There is nothing to configure and no manifest key, and what an agent
can rely on is the strong property: a clean server never answers from, and
never writes onto, a snapshot older than the file was at the time of the call.

`reload_graph` is still registered in `--graph` mode, read-only servers
included. It forces the re-read instead of waiting for the next call, reports
the new node/edge counts and the graph generation, and is the refresh path for
the cases the automatic one declines. A failed re-read keeps the current graph
serving and returns the error.

What that costs, and where it stops:

- **Every save by another process costs each other server one full re-read** on
  its next tool call — seconds on a ~100 MB graph, paid inside whichever tool
  call happens to be first, with concurrent calls waiting behind it. The
  re-read is single-flight and lazy: calls that all saw the change queue behind
  one load rather than starting several, and a producer that writes ten times
  between two queries costs one re-read, not ten.
- **The `stat` runs on the calling thread**, so a `.kgl` on a hung network
  volume stalls tool calls in the freshness check itself. Serve graphs from
  local storage.
- **A failed re-read keeps the previously loaded graph serving** and attaches a
  warning to tool results. It is retried only when the file's identity changes
  *again* **and** at least five seconds have passed since the failure — so a
  producer republishing torn bytes cannot cost every call a doomed load, and a
  file that stays broken is never retried automatically at all. An explicit
  `reload_graph` always tries.
- **A file written by a newer kglite than this binary** cannot be read at all,
  and the warning says to restart this server on a newer kglite rather than
  offering a retry that can never succeed.
- **A server holding unsaved changes never auto-reloads** — the re-read would
  discard them silently. It warns on every response instead and leaves the
  choice to `save_graph_as` or `reload_graph(discard_unsaved=true)` (see *The
  writer lease* below).
- **Disk-graph *directories* are never auto-refreshed.** A disk graph is a tree
  of live memory maps behind a `CURRENT` pointer rather than an atomically
  replaced file, so `reload_graph` remains its refresh path.
- **`extensions.graph_watch` is retired.** The key is still parsed — a
  non-boolean value still fails boot — but any boolean now only logs a
  retirement warning and arms nothing, because the refresh it used to opt into
  is unconditional. Remove it from the manifest.

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

### The writer lease, and several servers on one file

A `.kgl` has one writer at a time, guarded by an advisory lock on a
`<name>.kgl.lock` sidecar. A server takes that lease **at its first unsaved
change**, not at boot:

- A **read-only** server serving a `.kgl` that already exists never takes it.
  Any number of them can serve one file while a rebuilder republishes it in
  place.
- A **write-enabled** server (`--writable`, or `builtins.save_graph: true`)
  boots lease-free as well. The first mutating `cypher_query` acquires the
  lease, and it is held until `save_graph` writes the changes back,
  `save_graph_as` moves them elsewhere,
  `reload_graph(discard_unsaved=true)` drops them, or the process exits.
  Outside that window the server is an ordinary reader.

So several write-enabled servers — four MCP clients booted from one manifest,
say — can serve the same graph and arbitrate per *write* rather than per
process. The first to mutate holds the lease; a peer that writes while it is
held waits about a quarter of a second and is then refused, by name:

```
cypher_query refused: /data/work.kgl is open for writing by "Claude Desktop"
(pid 4711, since 2026-09-01T09:12:04+02:00); only one process may write a graph
at a time. […] Nothing was changed here, and this graph is still readable —
keep querying it.
```

That name is `--lease-label`, else the `KGLITE_LEASE_LABEL` environment
variable, else the name of the process that spawned this server — usually the
MCP client itself, which is how four clients sharing one manifest still name
themselves apart. A refused write changes nothing, the graph stays readable,
and this server picks up what the holder wrote on its next call.

Writes that reach disk cannot silently overwrite each other either:

- `save_graph` refuses if the file changed on disk since this server loaded or
  last saved it. There is no merge between the two versions: `save_graph_as` to
  another path keeps this server's work, and
  `reload_graph(discard_unsaved=true)` drops it and serves the file as it is.
- `save_graph_as` **to the bound path** is `save_graph` under another name,
  that lost-update check included. To a *different* path it also releases the
  source file's lease — the graph is not going back there, and this is the call
  an agent reaches for to get out of the jam.
- `reload_graph` refuses to discard unsaved changes silently, and `load_graph`
  / `create_graph` refuse outright while the server is dirty. All three name
  `reload_graph(discard_unsaved=true)`: throwing work away has one spelling.
- Every `cypher_query` result footer — reads and writes alike — carries the
  graph generation and either `clean` or `unsaved changes — lease held since
  <T>`, as do the `<active_graph>` header on `graph_overview` and the
  activation summary. A lease parked by a write that died mid-call is
  therefore visible on every query instead of only to whoever writes next.

Two targets keep the lock from the open instead, because waiting is not safe
for them: a path that does not exist yet (this open is creating it, and locking
first is what stops two servers from both creating it), and a disk-graph
*directory*, whose columns stay memory-mapped while served, so an external
writer mutating one is memory corruption rather than a stale read. A created
file joins the lazy lifecycle once its first `save_graph` has published it; a
disk-graph directory keeps its lock for as long as the server serves it.

Operating notes:

- **Never delete `<name>.kgl.lock` from a build script or a cleanup job.**
  Deleting it does not release a live lock and does nothing for a dead one —
  the operating system releases the lease when the holder exits, crash
  included. All the deletion removes is the `<name>.kgl.lock-owner` record that
  lets the next refusal name the holder.
- **A peer that merely *inspects* the graph with `kglite.open(path)` rewrites
  it.** `open()` is the writer's entry point: it takes the lease, and its
  `close()` (or `with`-block exit) writes the whole graph back even when
  nothing was mutated. That rewrite costs every serving server one full re-read
  on its next call, and a server that was holding unsaved changes has its
  `save_graph` refused from then on. Inspect with `kglite.load(path)` or
  `kglite.open_session(path)`, which take neither the lease nor the save-back
  binding.

### The source root `--graph` binds by default

In `--graph` mode a manifest that declares no `source_root`/`source_roots` does
not leave the server without one: the parent directory of the `.kgl` file is
auto-bound as the sole static source root, so the file-reading tools serve the
files sitting next to the graph with no configuration. That is the default, not
a fallback for a missing manifest — a manifest that configures Cypher tools and
skills but says nothing about roots still gets it. Reads stay confined to the
bound root, so the directory the graph lives in is exactly the blast radius:
a `.kgl` at the top of a home directory or a shared volume binds all of it.

An explicit declaration wins outright — the auto-bind applies only when the
manifest names no roots at all:

```yaml
# serve the graph from /data but read files only from /srv/project
source_roots: [/srv/project]
```

To scope it, name the narrower directory; to move it, name a different one; to
serve no files from a wide graph directory, keep the graph in a directory of its
own, or drop the source tools from `extensions.tools_allow` (above), which is
the closed-by-default surface. `--source-root`/`--watch` mode has no auto-bind
question: the directory is the argument.

### Pinning the write scope

`cypher_query`'s `write_scope` argument is set by the agent, so by itself it is
role hygiene rather than access control. The operator's counterpart is
`--write-scope` (comma-separated) or `extensions.write_scope`:

```bash
kglite-mcp-server --graph /data/work.kgl --writable --write-scope Plan,Task
```

```yaml
extensions:
  write_scope: [Plan, Task]
```

The pin is a ceiling, and it never falls open:

- the agent omits `write_scope` → the pinned scope applies (not unrestricted);
- the agent supplies one → the two are **intersected**, so it can narrow but
  never widen;
- nothing left in scope → the write is refused, with a message naming the
  server's scope so the agent can tell a policy refusal from a typo;
- flag *and* manifest key set → those two are intersected as well, and the
  effective scope is logged at boot.

A malformed `extensions.write_scope` — anything but a list of strings — fails
the boot rather than being dropped, on the same reasoning as
`extensions.tools_allow`. An explicit `[]` is honoured literally: a
write-enabled server that permits no writes.

The scope covers node writes (by the node's **stored** type, so a pattern label
cannot widen it) and relationship writes (at least one endpoint's stored type in
scope). Outside it, deliberately: relationship *constraint* DDL, `db.cdc.*`, and
the graph-lifecycle tools, which replace or persist the whole graph rather than
writing nodes in it — an agent that must not swap the served graph should not
have `load_graph`/`create_graph`/`save_graph_as` in `extensions.tools_allow`.

Those tools are also the ones that *end* a lease window, so an allowlist that
hides both `save_graph` and `reload_graph` from a server that can still mutate
leaves it holding the writer lease from its first write until the process
exits — locking every peer out of the file for the session. Hide the mutation
route (`cypher_query` write scope, or read-only mode) rather than the way back
out of one.

## Code intelligence

The generic KGLite server serves and queries code graphs but does not build
them. Use **codingest-mcp** for repository cloning, parsing, local watch mode,
and multi-revision code-graph construction; it embeds this same graph-serving
surface with the builder injected.

The complete manifest, skill, tool-gating, and client-registration reference is
the [MCP servers guide](../python/guides/mcp-servers.md).
