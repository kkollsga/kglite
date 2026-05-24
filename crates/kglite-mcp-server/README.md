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
- **[kglite Rust API](https://docs.rs/kglite)** — for embedders.

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE).
