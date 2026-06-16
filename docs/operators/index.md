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
clients. It's a **single, pure-Rust implementation** — reachable two
ways, both running the identical server:

```bash
# Standalone binary (no Python at all — this operators track's default):
cargo install kglite-mcp-server
kglite-mcp-server --graph my-graph.kgl --mcp-config my_graph_mcp.yaml

# Or, for Python users, the same server ships inside the wheel (0.10.26+):
pip install kglite
kglite-mcp-server --graph my-graph.kgl --mcp-config my_graph_mcp.yaml
```

No Python runtime is required for the `cargo install` binary. The
`pip install` path bundles the Rust server *inside* the wheel
(statically linked into the extension, sharing the one engine — no
separate wheel, no duplicated engine) and exposes the same
`kglite-mcp-server` command via a thin console-script shim. (History:
through 0.10.24 the wheel shipped a *Python* server; 0.10.25 retired it
for cargo-only to stop two implementations drifting; 0.10.26 brought the
command back to `pip` as the bundled Rust server.)

Setup, manifest, and tool customization details live in the
[MCP servers guide](../python/guides/mcp-servers.md); skill authoring
(teaching agents how/when to use each tool) is in
[MCP skills](../python/guides/mcp-skills.md).

```{toctree}
:hidden:

bolt-server
```
