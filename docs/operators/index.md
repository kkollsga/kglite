# Operators

Running the kglite protocol servers in production. The Python
package + wheel are out of scope here — those are the [Python
guide](../python/index.md). This track is for operators deploying
the **standalone binaries**.

## Bolt server (Neo4j wire protocol)

[`kglite-bolt-server`](bolt-server.md) speaks the Bolt v5.x wire
protocol so any Neo4j-aware client (Neo4j Browser, BloodHound, the
official drivers in Python / JS / Java / Go / .NET, LangChain's
`Neo4jGraph`) plugs in unchanged.

After Phase F (2026-05-24):
- `neo4j://` URIs work via single-server routing (`--advertise-addr`
  flag for reverse-proxy deployments).
- TLS via `--tls-cert` / `--tls-key`. Drivers connect with
  `bolt+s://` or `neo4j+s://`.
- `db.labels()` / `db.relationshipTypes()` yield Neo4j-conventional
  column names (`label` / `relationshipType`).
- OCC enforced on commit (concurrent writers get
  `Neo.ClientError.Transaction.ConflictDetected`).

```bash
cargo install --path crates/kglite-bolt-server
kglite-bolt-server --graph my-graph.kgl --bind 127.0.0.1 --port 7687
```

## MCP server (Model Context Protocol)

The `kglite-mcp-server` binary exposes Cypher queries + schema
introspection over MCP for use with Claude Code / other MCP
clients. There are two implementations:

- **Rust binary** (`crates/kglite-mcp-server/`) — pure Rust, no
  Python runtime needed. Built via `cargo install --path
  crates/kglite-mcp-server`.
- **Python implementation** in the wheel — `pip install kglite`
  installs `kglite-mcp-server` console script pointing at
  `kglite.mcp_server.server:main`. Same protocol surface; uses
  the Python ecosystem for tool framework + extensions.

Setup, manifest, and tool customization details live in the
existing [MCP servers guide](../python/guides/mcp-servers.md).
The guide is Python-flavored (matches the wheel's distribution
path) but the protocol details apply to both implementations.

```{toctree}
:hidden:

bolt-server
```
