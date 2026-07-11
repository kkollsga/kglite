# kglite-bolt-server

[![crates.io](https://img.shields.io/crates/v/kglite-bolt-server)](https://crates.io/crates/kglite-bolt-server)
[![License: MIT](https://img.shields.io/crates/l/kglite-bolt-server)](https://github.com/kkollsga/kglite/blob/main/LICENSE)

**Bolt v5.x protocol server for [kglite](https://crates.io/crates/kglite)
knowledge graphs.** A pure-Rust single binary speaking the Neo4j wire protocol.
The official Python driver is regression-tested; other Bolt v5 clients may
connect but can rely on features outside KGLite's documented wire and Cypher
contracts.

```bash
cargo install kglite-bolt-server

kglite-bolt-server --graph my-graph.kgl --bind 127.0.0.1 --port 7687
```

Then point a Bolt v5 client at `bolt://localhost:7687` and run KGLite's
documented Cypher dialect against the loaded `.kgl` graph.

## Features (Phase F)

- **Bolt v5.x handshake + PackStream framing** (handshake versions
  5.0 / 5.1 / 5.2 / 5.3 / 5.4 advertised).
- **`neo4j://` routing URIs** via single-server routing table
  (`--advertise-addr` for reverse-proxy deployments).
- **TLS** via `--tls-cert` + `--tls-key` (drivers connect with
  `bolt+s://` or `neo4j+s://`).
- **`db.labels()` / `db.relationshipTypes()`** yield
  Neo4j-conventional column names (`label`, `relationshipType`).
- **Optimistic concurrency control** on commit — concurrent
  writers whose snapshots become stale see
  `Neo.ClientError.Transaction.ConflictDetected`. Retry on the
  client side.
- **Zero PyO3 in the binary** — no libpython link, no Python
  runtime required. `cargo tree -p kglite-bolt-server | grep
  pyo3` returns empty.

## CLI

```
kglite-bolt-server [OPTIONS] --graph <PATH>

Options:
  --graph <PATH>               .kgl graph file to serve
  --bind <ADDR>                Bind address [default: 127.0.0.1]
  --port <PORT>                Port [default: 7687]
  --readonly                   Reject mutations
  --auth <USER:PASS>           Basic auth credentials
  --idle-timeout <SECS>        Per-session idle timeout
  --max-sessions <N>           Max concurrent sessions
  --advertise-addr <HOST:PORT> Address advertised in routing table (for neo4j:// URIs)
  --tls-cert <PATH>            PEM-encoded TLS certificate chain
  --tls-key <PATH>             PEM-encoded TLS private key
```

## Documentation

- **[Bolt server operator guide](https://kglite.readthedocs.io/en/latest/operators/bolt-server.html)**
  — deployment patterns, driver compatibility, OCC retry shape.
- **[kglite Rust API](https://docs.rs/kglite)** — for embedders
  who want the engine directly without the Bolt frontend.

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE).
