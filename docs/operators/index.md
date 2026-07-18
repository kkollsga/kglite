# Operators

Choose the smallest deployment surface that matches the client:

| Need | Surface | Start |
|---|---|---|
| Local shell, scripts, JSONL agent loop | [`kglite` CLI](cli.md) | `cargo install kglite-cli` or `pip install kglite-cli` |
| MCP-capable agent over stdio | [MCP server](mcp-server.md) | `cargo install kglite-mcp-server` or `pip install kglite` |
| Neo4j driver / Bolt v5.x | [Bolt server](bolt-server.md) | `cargo install kglite-bolt-server` |
| In-process application | [Python](../python/index.md) or [Rust](../rust/index.md) | import/add the library; no protocol server |

## Deployment checklist

1. Choose storage: memory for fastest small/medium graphs, mapped for mmap
   columns, disk for directory-backed CSR at very large scale.
2. Decide read-only vs writable operation. MCP writes require `--writable`;
   Bolt uses `--readonly` to reject writes.
3. Use absolute graph/config paths and bind network listeners to loopback unless
   a trusted reverse proxy or host firewall provides the boundary.
4. Configure Bolt authentication/TLS where exposed beyond localhost. MCP uses
   stdio; source roots and manifests define its filesystem/tool boundary.
5. Back up the complete `.kgl`/disk directory before upgrades and read the
   migration notes. Portable CSV exports are intentionally not full backups.
6. Run `--help` for the installed version and a startup/self-test before routing
   production traffic.

For persistence semantics, backup limitations, and durable in-memory WAL use,
see [Import and Export](../python/guides/import-export.md) and
[Durable apps](../python/guides/durable-apps.md).

```{toctree}
:hidden:

mcp-server
bolt-server
cli
```
