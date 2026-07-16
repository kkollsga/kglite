# Operators

Running kglite's standalone binaries in production or automation. The
Python package + wheel are out of scope here — those are the [Python
guide](../python/index.md). This track is for operators deploying the
**standalone binaries**.

## CLI

[`kglite-cli`](cli.md) ships the standalone `kglite` command for
one-shot Cypher queries, scoped writes, agent JSONL sessions, graph
description output, text export, diffs, and the interactive shell.

## Bolt server (Neo4j wire protocol)

[`kglite-bolt-server`](bolt-server.md) speaks the Bolt v5.x wire protocol. The
official Python driver path is regression-tested; other Bolt clients may
connect, subject to KGLite's documented protocol and Cypher limits.

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

### Two workspace archetypes for code intelligence

Beyond serving a single pre-built `--graph` file, the two most-wanted
deployments both build a code graph over source on the fly (builder injected by codingest-mcp). Both
ship as copy-pasteable manifests in the repo's `examples/`:

- **Clone-and-explore GitHub repos** (`workspace.kind: github`) —
  [`open_source_workspace_mcp.yaml`](https://github.com/kkollsga/kglite/blob/main/examples/open_source_workspace_mcp.yaml).
  The agent calls `repo_management('org/repo')` to clone and graph a repo
  on demand.
- **Review a local directory** (`workspace.kind: local` + `set_root_dir` +
  `watch`) —
  [`local_code_review_mcp.yaml`](https://github.com/kkollsga/kglite/blob/main/examples/local_code_review_mcp.yaml).
  Point it at a checked-out tree; swap roots at runtime with
  `set_root_dir(path)`; watch-mode auto-rebuilds on file change.

### Register with the absolute binary path, then verify

Point the client's `command` at the **absolute path** to the binary you
mean (`which kglite-mcp-server` inside your active env), not a bare
`kglite-mcp-server` — a bare name resolves against `$PATH` and can
silently launch an older PATH-shadowing install with a different tool set.
Then confirm the deployment stands up:

```bash
kglite-mcp-server --selftest --mcp-config my_graph_mcp.yaml
```

`--selftest` re-spawns the server with the same flags, drives a real MCP
handshake, and prints green/red per capability (tools registered, github
tools present when a token is set, graph hydrates). It exits non-zero on
any failure, so it also works as a CI / deployment smoke gate. Remember to
**restart** the server or client after editing any manifest / config —
those are read once at boot.

Setup, manifest, and tool customization details live in the
[MCP servers guide](../python/guides/mcp-servers.md); skill authoring
(teaching agents how/when to use each tool) is in
[MCP skills](../python/guides/mcp-skills.md).

```{toctree}
:hidden:

bolt-server
cli
```
