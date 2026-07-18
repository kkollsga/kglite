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

## Writable workbench

```bash
kglite-mcp-server --graph /data/work.kgl --writable
kglite-mcp-server --graph /data/new.kgl --storage memory --writable
```

`--writable` enables mutation and the `load_graph`, `create_graph`, and
`save_graph_as` lifecycle tools. `--storage memory|mapped|disk` is required when
the `--graph` target does not yet exist; it does not convert an existing graph.
Keep read-only mode for untrusted agents and scope filesystem access with
manifest `source_root`/`source_roots`.

## Code intelligence

The generic KGLite server serves and queries code graphs but does not build
them. Use **codingest-mcp** for repository cloning, parsing, local watch mode,
and multi-revision code-graph construction; it embeds this same graph-serving
surface with the builder injected.

The complete manifest, skill, tool-gating, and client-registration reference is
the [MCP servers guide](../python/guides/mcp-servers.md).
