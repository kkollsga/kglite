# kglite-bolt-server

Bolt v5.x protocol server for [kglite](https://github.com/kkollsga/kglite)
knowledge graphs. Pure-Rust single binary, no libpython link.

**Status: skeleton only (Phase B of [`bolt_implementation.md`](../../bolt_implementation.md)).**
Compiles, binds a port, panics on first Bolt message. The 11 `BoltBackend`
trait methods in `src/backend.rs` are wired up with `unimplemented!()`
bodies tagged to the Phase C sub-phase that fills them in:

| Method | Phase |
|---|---|
| `create_session` / `close_session` / `set_session_auth` / `configure_session` / `reset_session` / `get_server_info` | C.1 |
| `execute` (read-only RUN + scalars) | C.2 → C.3 → C.4 |
| `begin_transaction` / `commit` / `rollback` | C.5 |
| `route` (single-server) | C.1 |

The protocol seam (PackStream framing, message dispatch, session state
machine, handshake, auth scheme parsing) is provided by the
[`boltr`](https://crates.io/crates/boltr) crate; this server is the
backend implementation.

## Usage

```bash
kglite-bolt-server --graph path/to/fixture.kgl \
    --bind 127.0.0.1 --port 7687 \
    [--readonly] \
    [--auth none|basic --auth-user X --auth-pass Y]
```

Any Neo4j driver (Python `neo4j`, Cypher Shell, BloodHound, LangChain's
`Neo4jGraph`) can then point at `bolt://127.0.0.1:7687` and run Cypher
against the loaded graph.

## See also

- [`bolt_implementation.md`](../../bolt_implementation.md) — umbrella plan
- [`crates/kglite-mcp-server/`](../kglite-mcp-server) — structural sibling
  (MCP stdio protocol; same crate pattern)
