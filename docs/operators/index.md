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
clients. It's a **single, pure-Rust implementation**:

```bash
cargo install kglite-mcp-server
kglite-mcp-server --graph my-graph.kgl --mcp-config my_graph_mcp.yaml
```

No Python runtime is required to run the server. (Through 0.10.24 the
wheel also shipped a Python `kglite-mcp-server` console script; that
second implementation was retired in 0.10.25 to consolidate on one
server and stop the two surfaces drifting. `pip install kglite` now
installs the engine + `code_tree` only.)

Setup, manifest, and tool customization details live in the
[MCP servers guide](../python/guides/mcp-servers.md); skill authoring
(teaching agents how/when to use each tool) is in
[MCP skills](../python/guides/mcp-skills.md).

```{toctree}
:hidden:

bolt-server
```
