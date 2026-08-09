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

An existing `.kgl` opens in the storage mode it was saved in; a disk-graph
directory opens disk-backed. A missing path is an error unless creation is
explicit:

```bash
kglite-bolt-server --graph /data/new.kgl --storage memory
# --storage mapped|disk selects the other creation modes
```

`--storage` on an *existing* graph is a conversion request, not a no-op: a
memory-saved graph served with `--storage mapped` is converted to mapped before
the listener binds, and the startup log records `converted_from`. A disk graph
is a directory rather than a file, so converting into or out of disk mode has no
in-place form — those requests fail startup naming `enable_disk_mode()` instead
of serving a mode nobody asked for. Omit the flag to serve whatever the graph
recorded.

Important options (run `--help` on the installed version for the authority):

| Option | Purpose |
|---|---|
| `--bind`, `--port` | listener, default `127.0.0.1:7687` |
| `--storage memory|mapped|disk` | create a missing graph in this mode, or convert an existing one to it (memory ⇄ mapped; disk directions refused) |
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

## Driver identity (`--neo4j-compat`)

The handshake reports `kglite-bolt-server/<version>` by default. The official
Python and JavaScript drivers accept that; the official **Java** driver requires
a `Neo4j/` prefix and refuses the connection outright without one:

```
UntrustedServerException: Server does not identify as a genuine Neo4j
instance: 'kglite-bolt-server/0.14.5'
```

Enable compatibility mode to serve JVM clients. Either route works, and the flag
wins if both are set:

```bash
kglite-bolt-server --graph graph.kgl --neo4j-compat
KGLITE_BOLT_NEO4J_COMPAT=1 kglite-bolt-server --graph graph.kgl
```

The agent then becomes `Neo4j/5.26.0 (kglite-bolt-server/<version>)` — enough of
a Neo4j spelling to pass the driver's check, with the real product retained so
the server stays identifiable in logs, in driver errors, and through
`ServerInfo.agent()`. Only the `server` field changes; `bolt_agent` keeps
reporting kglite.

The variable accepts `1`, `true`, `yes` or `on` (any case) — the useful form for
container images and unit files, where adding an argument means rebuilding or
editing a unit.

Off by default on purpose: presenting as a different product is the operator's
call. When a driver that enforces the check connects with compatibility off, the
server logs a warning naming both activation routes, so an operator can diagnose
it from the server log instead of a client stack trace. The identity is never
switched automatically.

## Operations and security

- Loopback is the safe default. If exposed remotely, enable basic auth and TLS
  or terminate TLS/auth at a trusted proxy/firewall boundary.
- Set `--max-message-size`, `--max-sessions`, and an idle timeout for untrusted
  or multi-tenant clients.
- Use `--readonly` for analytical replicas and agent connections that do not
  need writes.
- One writable server per graph. A server started without `--readonly` takes
  the same cross-process writer lease as `kglite.open()` *before* it reads the
  graph, and holds it until shutdown, so a second writable server — or a CLI
  write, MCP server, or `kglite.open()` on that path — fails at startup naming
  the holding process instead of racing it to overwrite at save time. The
  refusal is immediate rather than a wait, so a supervisor's restart policy
  governs the retry. `--readonly` servers take no lease and start alongside a
  live writer.
- Back up the complete graph before upgrades; see
  [Import and Export](../python/guides/import-export.md).
- Use release benchmarks/CI reports for performance claims; this operator page
  intentionally avoids unversioned hardware-specific numbers.
