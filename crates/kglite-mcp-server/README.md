# kglite-mcp-server

[![crates.io](https://img.shields.io/crates/v/kglite-mcp-server)](https://crates.io/crates/kglite-mcp-server)
[![License: MIT](https://img.shields.io/crates/l/kglite-mcp-server)](https://github.com/kkollsga/kglite/blob/main/LICENSE)

**MCP (Model Context Protocol) server for
[kglite](https://crates.io/crates/kglite) knowledge graphs.**
Pure-Rust single binary exposing `cypher_query`, `graph_overview`,
`save_graph`, `read_code_source` plus the generic source / GitHub
surface from
[`mcp-methods`](https://crates.io/crates/mcp-methods). No libpython
link.

```bash
cargo install kglite-mcp-server

kglite-mcp-server --graph my-graph.kgl
```

Drop into Claude Desktop / Cursor / any MCP-capable client and your
graph is queryable.

## Agent graph workbench (write-enabled)

By default the server is **read-only** (`cypher_query` rejects mutations) —
the safe default for analysis / code-review. Add `--writable` to turn it into
a graph workbench an agent can **plan and work inside**:

```bash
kglite-mcp-server --graph contract.kgl --writable
```

**Write-enabled** means exactly two things: `--writable` on the command line, or
`extensions.writable: true` in the manifest. They are the same statement and
either alone is enough. `builtins.save_graph: true` is not a third spelling — on
its own it registers `save_graph` and nothing else, leaving `cypher_query`
read-only.

Now `cypher_query` accepts `CREATE`/`SET`/`DELETE`/`MERGE`, and three
runtime tools appear:

- `load_graph(path)` / `create_graph(path, storage)` — open or create a graph
  and make it active; swap between graphs within one session.
- `save_graph` / `save_graph_as(path)` — persist (mutations are in-memory until
  you save).

**Role-scoped writes.** Pass `write_scope` to restrict mutations to a set of
node types — e.g. a coding agent that may write its own board but not the
research-owned data:

```jsonc
cypher_query({ "query": "CREATE (:Task {id:'t1', status:'todo'})",
               "write_scope": ["Plan","Task","Decision","Question","Artifact"] })
```

Every **node** write (`CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE`,
`DETACH DELETE`, node-type index/constraint DDL) is judged by the node's
*stored* type — a pattern label cannot widen the scope — and a **relationship**
write (edge `CREATE`, `DELETE r`, `SET r.p`, `REMOVE r.p`) needs at least one
endpoint's stored type in the list.

**Operator-pinned write scope.** The `write_scope` above is the *agent's* own
choice, so on its own it is role hygiene rather than access control. Pin the
ceiling from outside the agent's reach with `--write-scope`, or
`extensions.write_scope` in the manifest:

```bash
kglite-mcp-server --graph contract.kgl --writable --write-scope Plan,Task
```

```yaml
extensions:
  write_scope: [Plan, Task]
```

The pin **always applies**: an agent that omits `write_scope` gets the pinned
scope (never unrestricted), an agent that supplies one gets the **intersection**
of the two, and a write with nothing left in scope is refused with a message
naming the server's scope. Setting both the flag and the manifest key
intersects them too, and the effective scope is logged at boot. A malformed
`extensions.write_scope` (anything but a list of strings) fails the boot rather
than being ignored — an allowlist that silently fails open is worse than no
allowlist. An empty list is honoured literally: a write-enabled server that
permits no writes.

Outside the perimeter, deliberately: relationship *constraint* DDL,
`db.cdc.*`, and graph-lifecycle tools (`load_graph` / `create_graph` /
`save_graph_as`), which replace or persist the whole graph rather than writing
nodes in it.

**Multiple agents, one graph.** Run **one** server process and point all
agents at it — writes serialize through the active graph's lock, so concurrent
agents can't lose each other's work. (Don't run several processes over the
same `.kgl` file: each loads its own copy and the last `save` wins.)

## When to use this binary

The Python wheel (`pip install kglite`) ships a `kglite-mcp-server`
console script too — same protocol surface, Python-flavored
extensibility (YAML manifests, skills, Python tool plugins).

Reach for the **Rust binary** when:
- You want a single static binary with no Python runtime.
- You're shipping kglite-as-MCP-server in a container or system
  that doesn't have Python installed.
- You want to embed kglite-MCP serving inside a larger Rust binary
  (the MCP server framework + tools are linkable as a library).

Reach for the **Python script** when you want the YAML-manifest
+ skills + Python tool plugin ecosystem (most kglite users).

## Documentation

- **[MCP servers guide](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html)**
  — protocol details, manifest schema, skill conventions. Python-flavored
  but the protocol details apply to both implementations.
- **[Named Cypher recipe catalogs](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html#extensions-cypher-recipes-grouped-structured-read-queries)**
  — strict manifest configuration and structured list/run contracts.
- **[kglite Rust API](https://docs.rs/kglite)** — for embedders.

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE).
