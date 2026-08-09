# kglite-bolt-server

[![crates.io](https://img.shields.io/crates/v/kglite-bolt-server)](https://crates.io/crates/kglite-bolt-server)
[![License: MIT](https://img.shields.io/crates/l/kglite-bolt-server)](https://github.com/kkollsga/kglite/blob/main/LICENSE)

**Bolt v5.x protocol server for [kglite](https://crates.io/crates/kglite)
knowledge graphs.** A pure-Rust single binary speaking the Bolt wire protocol.
The official **Python, JavaScript, and Java** drivers are regression-tested in
CI — session and explicit-transaction lifecycle, managed `executeWrite` retry,
PackStream type round-trips, Node/Relationship/Path values, `Neo.*` error
codes, and OCC conflict detection (`tests/conformance/js`,
`tests/conformance/java`, driven by
`tests/test_bolt_driver_conformance.py`). Other Bolt v5 clients may connect but
can rely on features outside KGLite's documented wire and Cypher contracts.

```bash
cargo install kglite-bolt-server

kglite-bolt-server --graph my-graph.kgl --bind 127.0.0.1 --port 7687
```

Then point a Bolt v5 client at `bolt://localhost:7687` and run KGLite's
documented Cypher dialect against the loaded `.kgl` graph.

## Features

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
- **Cross-process writer lease** — a writable server takes the
  graph's exclusive writer lease before it reads the path and holds
  it until shutdown, so a second writer (server, CLI, MCP server or
  `kglite.open()`) is refused at startup by name instead of
  overwriting its work at save time. `--readonly` takes no lease.
- **Zero PyO3 in the binary** — no libpython link, no Python
  runtime required. `cargo tree -p kglite-bolt-server | rg
  pyo3` returns empty.

## Transaction metadata (`write_scope` / `git_sha` / `modified_by`)

KGLite's write-scope and write-provenance options ride on Bolt
transaction metadata, at parity with the CLI and MCP surfaces:

```python
with driver.session() as session:
    tx = session.begin_transaction(metadata={
        "write_scope": ["Plan", "Task"],   # node types CREATE/SET may touch
        "git_sha": "0f3a9c1",              # provenance stamped on writes
        "modified_by": "planning-agent",   # actor stamped alongside git_sha
    })
    tx.run("CREATE (:Plan {id: 1})")
    tx.commit()
```

Drivers send this under the `tx_metadata` key of BEGIN's `extra`
dict (auto-commit runs: RUN's `extra`); hand-rolled Bolt clients may
also place the same keys at the top level of `extra`. A `CREATE`/`SET`
touching a node type outside `write_scope` fails the query; `git_sha` /
`modified_by` are stamped on writes to `auto_timestamp` types. All
three are ignored by reads. Malformed values (non-list `write_scope`,
non-string `git_sha`) fail the BEGIN/RUN with a client error.

## CLI

```
kglite-bolt-server [OPTIONS] --graph <PATH>

Options:
  --graph <PATH>               .kgl graph file to serve
  --bind <ADDR>                Bind address [default: 127.0.0.1]
  --port <PORT>                Port [default: 7687]
  --readonly                   Reject mutations
  --neo4j-compat               Report a Neo4j-compatible agent in the handshake
  --auth <USER:PASS>           Basic auth credentials
  --idle-timeout <SECS>        Per-session idle timeout
  --max-sessions <N>           Max concurrent sessions
  --advertise-addr <HOST:PORT> Address advertised in routing table (for neo4j:// URIs)
  --tls-cert <PATH>            PEM-encoded TLS certificate chain
  --tls-key <PATH>             PEM-encoded TLS private key
```

## Driver identity and `--neo4j-compat`

By default the server identifies honestly in the Bolt handshake:

```
kglite-bolt-server/0.14.5
```

The official **Python** and **JavaScript** drivers accept this — they do not
inspect the server agent. The official **Java** driver does: it requires the
agent to begin with `Neo4j/` and otherwise aborts the connection before running
a single query.

```
UntrustedServerException: Server does not identify as a genuine Neo4j
instance: 'kglite-bolt-server/0.14.5'
```

Compatibility mode makes the server present an agent that satisfies that check,
while keeping the real product in the string:

```
Neo4j/5.26.0 (kglite-bolt-server/0.14.5)
```

Two equivalent ways to turn it on — the flag wins if both are set:

```bash
kglite-bolt-server --graph graph.kgl --neo4j-compat

KGLITE_BOLT_NEO4J_COMPAT=1 kglite-bolt-server --graph graph.kgl
```

The environment variable accepts `1`, `true`, `yes` or `on` (any case), for
Docker, systemd and CI where editing an argv is awkward.

**Why it is off by default.** Claiming to be Neo4j is a claim about a different
product, so it is the operator's decision, not the server's. When a client whose
driver enforces the check connects while compatibility mode is off, the server
logs a warning naming both activation routes — so the fix is discoverable from
your own log rather than only from a client stack trace. Detection is a hint
only: the identity is never switched automatically on the strength of a
client-supplied string.

Compatibility mode changes only the handshake's `server` field. The `bolt_agent`
metadata keeps reporting `kglite-bolt-server/<version>` either way, because no
driver gates on it and the claim should not travel further than it must.

## Documentation

- **[Bolt server operator guide](https://kglite.readthedocs.io/en/latest/operators/bolt-server.html)**
  — deployment patterns, driver compatibility, OCC retry shape.
- **[kglite Rust API](https://docs.rs/kglite)** — for embedders
  who want the engine directly without the Bolt frontend.

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE).
