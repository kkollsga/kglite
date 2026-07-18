# Bolt server

`kglite-bolt-server` speaks Bolt v5.x and is backed by
`Arc<kglite::api::session::Session>`. The official Neo4j Python driver path is
regression-tested; other Bolt v5 clients are subject to the documented protocol
and [Cypher dialect](../reference/cypher-reference.md) limits.

## Install and start

```bash
cargo install kglite-bolt-server
kglite-bolt-server --graph /data/app.kgl
```

An existing `.kgl` or disk-graph directory is auto-detected. A missing path is
an error unless creation is explicit:

```bash
kglite-bolt-server --graph /data/new.kgl --storage memory
# --storage mapped|disk selects the other creation modes
```

Important options (run `--help` on the installed version for the authority):

| Option | Purpose |
|---|---|
| `--bind`, `--port` | listener, default `127.0.0.1:7687` |
| `--storage memory|mapped|disk` | create a missing graph; ignored for existing graphs |
| `--readonly` | reject mutations at execution |
| `--auth none|basic`, `--auth-user`, `--auth-pass` | Bolt LOGON policy |
| `--idle-timeout`, `--max-sessions`, `--max-message-size` | resource bounds |
| `--advertise-addr HOST:PORT` | address returned to `neo4j://` routing clients |
| `--tls-cert`, `--tls-key` | PEM TLS pair for `bolt+s://` / `neo4j+s://` |

## Driver example

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=None)
with driver.session() as session:
    rows = session.run("MATCH (n) RETURN count(n) AS n").data()
```

With basic auth, pass the configured `(user, password)`. Use `neo4j://` only
when routing behavior is wanted; set `--advertise-addr` to an address reachable
by the client, especially behind a proxy or when binding `0.0.0.0`.

## Transactions and errors

The backend uses native KGLite sessions/transactions, not Python or the GIL.
Auto-commit and explicit driver transactions both run through the canonical
session pipeline. Concurrent writers compose through session serialization;
stale explicit transactions surface a mapped conflict status. KGLite typed
errors map to Neo4j status codes for syntax, schema, timeout, access-mode,
conflict, and execution failures.

The supported behavior is locked by the standing Bolt correctness and
differential suites. Avoid relying on an exact test/query count or a particular
driver patch version; CI exercises the complete current corpus.

## Operations and security

- Loopback is the safe default. If exposed remotely, enable basic auth and TLS
  or terminate TLS/auth at a trusted proxy/firewall boundary.
- Set `--max-message-size`, `--max-sessions`, and an idle timeout for untrusted
  or multi-tenant clients.
- Use `--readonly` for analytical replicas and agent connections that do not
  need writes.
- Back up the complete graph before upgrades; see
  [Import and Export](../python/guides/import-export.md).
- Use release benchmarks/CI reports for performance claims; this operator page
  intentionally avoids unversioned hardware-specific numbers.
