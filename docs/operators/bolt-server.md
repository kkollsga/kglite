# Bolt server

`kglite-bolt-server` speaks Bolt v5.x and is backed by
`Arc<kglite::api::session::Session>`. The official Neo4j Python driver path is
regression-tested; other Bolt v5 clients are subject to the documented protocol
and [Cypher dialect](../reference/cypher-reference.md) limits.

## What this server is (and is not)

**It is** a driver-compatible Bolt front-end over the embedded engine: one
process owns one graph and serves Neo4j-aware clients over the wire. Use it for
trusted or loopback access, or behind a proxy that owns authentication and
authorization. Reads run against snapshots and scale across concurrent sessions.

**It is not** a Neo4j server replacement:

- **No user directory and no RBAC.** `--auth basic` configures a single shared
  credential; the authenticated principal is validated at LOGON and not stored,
  so there is no per-session identity to authorize against. `--auth none`
  accepts any LOGON.
- **No high availability and no replication.** A single process serves a single
  graph — it is a single point of failure, and there is no failover, no cluster,
  and no bookmark/causal-consistency protocol.
- **One writer.** Writes serialize at commit within the process, and one
  writable server per graph is enforced by a cross-process lease (see
  *Operations and security* below).

If you need per-user access control, the supported shape is not this server: it
is the [derived-index / traversal-component pattern](../python/guides/derived-index.md#an-embedded-traversal-component-behind-your-api),
where the engine is embedded behind your own API and that API owns
authentication, authorization, and write policy.

Single-writer and no-HA are design decisions, not gaps: KGLite is deliberately
an embedded single-graph engine, and this server publishes that engine over
Bolt rather than layering a distributed database on top of it. For the
feature-by-feature carry-over table — routing URIs, auth, auto-commit
mutations, OCC, multi-database — see
[Migrating from Neo4j to KGLite](../python/migrations/neo4j-to-kglite.md).

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
Reads may auto-commit; **all writes must be explicit driver transactions** —
an auto-commit `CREATE`/`SET`/`DELETE`/`MERGE` is rejected rather than run
(drivers wrap writes in a transaction anyway). Concurrent writers serialize at
commit, and a transaction committing against a stale snapshot conflicts with a
retriable status code, so driver-managed transactions (`execute_write` and its
per-language equivalents) retry the unit of work by themselves; hand-rolled
`begin_transaction` code needs its own retry loop. KGLite typed errors map to
Neo4j status codes for syntax, schema, timeout, access-mode, conflict, and
execution failures.

The supported behavior is locked by the standing Bolt correctness and
differential suites. Avoid relying on an exact test/query count or a particular
driver patch version; CI exercises the complete current corpus.

### Write concurrency

Reads run against snapshots and do not block each other or writers. Writes are
the serialized resource: every write is an explicit transaction, transactions
work independently, and they order at commit. A transaction whose snapshot was
overtaken loses the race and conflicts with the retriable status code, so a
driver-managed transaction re-runs the unit of work on a fresh snapshot without
your code seeing the conflict at all.

What that means for capacity, measured under contending managed writers on one
graph:

- **Committed throughput is flat.** Adding writers does not add write
  throughput — the commit point is single — but it does not lose it either.
  Eight contending writers commit at roughly the rate one writer does.
- **Latency is where contention shows up.** The median committed transaction
  stays as fast as the uncontended one; the *average* grows about in proportion
  to the number of writers, because a transaction now waits behind others.
- **Conflicts stay rare and self-clearing.** The share of attempts that had to
  be retried stays in the low single digits at eight writers, no writer
  exhausts the driver's retry budget, and every committed write lands exactly
  once.
- **The tail belongs to the retry policy, not the server.** An unlucky writer
  can lose several conflict rounds in a row, and each loss pays the driver's
  compounding backoff, so worst-case *end-to-end* time for one unit of work
  reaches seconds while the underlying transaction still takes under a
  millisecond. Tune `max_transaction_retry_time` and the retry-delay settings
  if that tail matters to you — that is a client-side dial.

Size a deployment by write *rate*, therefore, not by writer count: more
concurrent clients do not raise the ceiling, and past it, latency rather than
error rate is what degrades. If a workload needs more write throughput than one
commit point provides, batch more work into each transaction rather than adding
writers.

None of this makes the server highly available: one process owns the graph, and
losing it loses the endpoint. That is the design (see *What this server is (and
is not)* above), not a tuning problem.

The measured curve is produced by
`tests/benchmarks/test_bench_bolt_writers.py` (opt-in:
`-m "benchmark and bolt_stress"`), which sweeps the writer count and records
committed throughput, retry rate, and latency percentiles; captured numbers
land in the repository's benchmark results record. The correctness half —
that conflicts are retriable and managed transactions actually retry them —
is pinned by `tests/test_bolt_server_transactions.py` and
`tests/test_bolt_server_concurrency.py`.

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
- Set `--max-message-size`, `--max-sessions`, and an idle timeout whenever the
  listener is reachable from an untrusted network — they bound resource use per
  connection, they are not an access-control boundary.
- Use `--readonly` for read-only analytical instances sharing the same graph
  file, and for agent connections that do not need writes. A `--readonly`
  server is a second process opening the same graph, not a replica: it serves
  what the file contained when it opened it.
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
