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
- **The graph file is not rewritten continuously.** Every commit is appended to
  a write-ahead log as it is acknowledged, but the `.kgl` itself changes only
  when a checkpoint runs — `CALL db.checkpoint()`, `--checkpoint-interval`, or
  `--save-on-exit`. Turn the log off with `--durability off` and a commit is
  process-local until one of those runs: the file is whatever it was when the
  server opened it. See *Durability* below for what each level costs and what
  it leaves behind.

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
| `--storage memory\|mapped\|disk` | create a missing graph in this mode, or convert an existing one to it (memory ⇄ mapped; disk directions refused) |
| `--readonly` | reject mutations at execution |
| `--durability full\|normal\|off` | what an acknowledged commit survives, default `normal` (see *Durability*) |
| `--save-on-exit` | checkpoint the served graph back to `--graph` on `SIGINT`/`SIGTERM` |
| `--checkpoint-interval SECS` | checkpoint the served graph on a timer |
| `--auth none\|basic`, `--auth-user`, `--auth-pass` | Bolt LOGON policy |
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
writers — that is the dial that moves the ceiling. Measured on the same graph,
raising a transaction from a single write to a hundred (one `UNWIND $rows`
query instead of a hundred round-trips) multiplied committed writes per second
by roughly an order of magnitude, contended and uncontended alike: the cost
that dominates a small write is per-*transaction*, and batching amortizes it.
The trade is at the tail — a longer transaction is a wider window in which to
be overtaken, so at a hundred writes per transaction roughly one attempt in ten
is retried, against under one in fifty unbatched, and a single unit of work
correspondingly takes longer end to end.

None of this makes the server highly available: one process owns the graph, and
losing it loses the endpoint. That is the design (see *What this server is (and
is not)* above), not a tuning problem.

The measured curve is produced by
`tests/benchmarks/test_bench_bolt_writers.py` (opt-in:
`-m "benchmark and bolt_stress"`), which sweeps the writer count and the
writes-per-transaction batch size, recording committed throughput, retry rate,
and latency percentiles; captured numbers
land in the repository's benchmark results record. The correctness half —
that conflicts are retriable and managed transactions actually retry them —
is pinned by `tests/test_bolt_server_transactions.py` and
`tests/test_bolt_server_concurrency.py`.

## Durability

Two independent mechanisms decide what a stopped or killed server leaves
behind. A **write-ahead log** records each commit to a sidecar file as it is
acknowledged, so an interruption costs at most what the log does not hold. A
**checkpoint** rewrites the whole `.kgl` from the committed graph and truncates
the log. They are complements, not alternatives: the log bounds the window
while the server runs, the checkpoint is what folds the log back into the file.

### Levels (`--durability`)

`--durability full|normal|off` — or `KGLITE_BOLT_DURABILITY=<level>`, the flag
winning if both are set — selects what an *acknowledged* commit survives. The
frame is written **before** the client is told the commit succeeded, so a
commit whose frame cannot be written is not applied at all and is reported as a
failure, rather than acknowledged over a write the server then discards.

| Level | An acknowledged commit survives | It does not survive |
|---|---|---|
| `full` | the server process dying, and — by asking the device for a write barrier before acknowledging — an OS crash or power loss | media failure, or anything the filesystem itself loses |
| `normal` (default) | the server process dying: `SIGKILL`, an OOM kill, a panic. The frame is in the kernel's page cache | an OS crash or power loss before the kernel writes that page out |
| `off` | nothing by itself — commits stay in this process until a checkpoint rewrites the file | the process ending at all, unless a checkpoint ran first |

What is pinned by test is the process-kill case: at `full` and at `normal`,
committing over Bolt and then `SIGKILL`ing the server with no checkpoint of any
kind leaves the `.kgl` byte-identical, and the restarted server replays the
commit out of the log. `off` is the control — the same write is gone. Nothing
in a user-space test can take the page cache or the power away, so the
`full`-versus-`normal` distinction above is a statement about the barrier each
level takes, not a measured one.

**The default is `normal`, and it was chosen by measurement.** Under contended
managed writers on one graph, `normal` cost nothing distinguishable from `off`
— the two comparison runs straddled zero, inside the cell's own noise — while
`full` cost roughly seven-eighths of committed throughput, about an eightfold
drop. One device barrier is taken per commit, inside the lock every Bolt commit
already serializes on, which is also why `full`'s committed rate does not
change between one writer and four, why its p95 unit-of-work latency grew by
nearly two orders of magnitude at four writers, and why its transaction-conflict
rate rose about tenfold: contenders lose the optimistic-concurrency race far
more often when the winner holds the lock across a barrier. Power-loss safety
is therefore opt-in rather than on by default. The sweep is
`tests/benchmarks/test_bench_bolt_writers.py::test_durability_sweep`
(`-m "benchmark and bolt_stress"`).

### Recovery on startup

Recovery is unconditional and runs before the listener binds, at every level —
opening a path is a decision about that path's *data*, not only about how
future writes will be logged. At `full` and `normal` a sidecar holding commits
the `.kgl` does not contain is replayed into the graph. At `off` the same
sidecar is a startup error naming both ways out: restart at `full` or `normal`
to replay the commits, or move the sidecar aside to discard them deliberately.
A server that would otherwise serve a graph missing acknowledged writes does
not start.

Frames the checkpoint already contains — the harmless residue of a crash
between a checkpoint's file write and its log truncation — are not grounds to
refuse, and open at every level.

### Checkpoints

Three routes rewrite the served `.kgl`, and all three are the same operation:
flush the log, stamp the checkpoint position, write the file, truncate the log.

- **`CALL db.checkpoint()`** — on demand, over the wire. It is a *bolt-server
  verb*, not an engine procedure: it exists only over Bolt, and embedded
  bindings keep their own save calls. It answers in Neo4j's `success, message`
  shape, and an optional `YIELD` of either or both columns is honoured. A
  checkpoint whose graph has not changed since the last one *in this process*
  is skipped and says so; the first call of a process always writes, because
  the file may predate the process.
- **`--checkpoint-interval SECS`** (`KGLITE_BOLT_CHECKPOINT_INTERVAL`) — on a
  timer. The interval task and the verb share one recorded version, so a
  checkpoint by either makes the next tick a skip; an idle server does not
  rewrite its file. A failed tick is logged as an error and the server keeps
  serving. The interval is validated at startup rather than starting a server
  that silently never checkpoints.
- **`--save-on-exit`** (`KGLITE_BOLT_SAVE_ON_EXIT`) — once, on `SIGINT` or
  `SIGTERM`, after periodic checkpointing has been stopped. The saved graph
  version is logged; a failed exit save is logged as an error *and* exits
  non-zero, so a supervisor sees it. Connections are not drained, so a commit
  racing shutdown can land after the save — the logged version is how that is
  told apart from a save that never ran. Under a log the commit is still in the
  sidecar and the next start replays it.

A checkpoint pauses writers and new snapshots for its duration — `Session::save`
mutates the graph, so it holds the session lock for the whole write — while
readers already holding a snapshot are unaffected. That pause is bounded by
what a full save of the graph costs, so it is the graph's size that sets it. At
the benchmark's ten-thousand-node scale a checkpoint every second under four
contended writers cost **no measured committed throughput** (the checkpointed
arm in fact committed more than its own baseline in every pairing, and a
sham-thread control ruled out the extra session as the explanation; the
mechanism is not established). Size the interval for a large graph by timing
one `CALL db.checkpoint()` on it.

Retention is one: each checkpoint atomically replaces the previous file. Use
filesystem tooling — a snapshot, a copy, a backup job — if you want history.

### The sidecar file

At `full` and `normal` the server keeps `<graph>-wal` beside the graph file,
and every checkpoint truncates it back to its header. Between checkpoints it
grows by under a hundred bytes per single-node commit — multiply that by your
commit rate to size it — so a busy server with no checkpointing configured
accumulates a sidecar in proportion to how long it has been running, and a
restart pays a replay proportional to the same thing. `--checkpoint-interval`
bounds both at once, which is the reason to set it: not to make commits safer
(the log already did that) but to keep replay time and sidecar size bounded.

Back up the sidecar with the graph, or checkpoint before copying the `.kgl`
alone — a `.kgl` copied while a sidecar runs ahead of it is missing the commits
the sidecar holds. The engine refuses the dangerous half of this by itself: a
non-durable open, and a save, over a path whose sidecar runs ahead are errors
rather than silent data loss.

### Refusal matrix

Two configurations cannot carry a log or a checkpoint: `--readonly` (a server
that never commits has nothing to log, and nothing to write back) and
disk-mode graphs (a disk graph commits by publishing an immutable generation,
so it keeps no logical log, and every disk save publishes a *new* generation
that nothing prunes — repeated checkpoints would grow the directory without
bound).

An explicitly requested level or feature is refused there. The *default* level
degrades instead, so that flipping the default did not turn every read-only
and disk-mode server into a startup error:

| Configuration | `--durability full`/`normal` (asked for) | `--durability` (default) | `--save-on-exit`, `--checkpoint-interval` | `CALL db.checkpoint()` |
|---|---|---|---|---|
| `.kgl`, writable | serves at that level | serves at `normal` | supported | supported |
| `--readonly` | startup error | serves at `off`, logged | startup error | `Neo.ClientError.Security.Forbidden` |
| disk-mode graph | startup error | serves at `off`, logged | startup error | `Neo.ClientError.Security.Forbidden` |

The one refusal that is about *data* rather than configuration is `off` over a
sidecar that runs ahead of the file, above: a level nobody asked for replays it
instead, which is what makes the default safe to inherit.

Environment mirrors are refused exactly as the flags are; a mistyped level or
interval is a startup error rather than a server that silently logs nothing.
`CALL db.checkpoint()` is also refused inside an explicit transaction — it
writes the *committed* graph, which by definition excludes that transaction's
uncommitted writes — so commit first and call it in auto-commit.

The behavior above is pinned by `tests/test_bolt_server_durability.py`
(`-m bolt`), including the `SIGKILL`-and-restart tests behind each level, the
checkpoint-truncates-the-log test, and every row of this matrix.

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
  what the file contained when it opened it — which, beside a durable writer,
  excludes whatever that writer has committed to its sidecar since the last
  checkpoint (see *Durability*). A read-only server keeps no log itself and
  serves at `--durability off`.
- One writable server per graph. A server started without `--readonly` takes
  the same cross-process writer lease as `kglite.open()` *before* it reads the
  graph, and holds it until shutdown, so a second writable server — or a CLI
  write, MCP server, or `kglite.open()` on that path — fails at startup naming
  the holding process instead of racing it to overwrite at save time. The
  refusal is immediate rather than a wait, so a supervisor's restart policy
  governs the retry. `--readonly` servers take no lease and start alongside a
  live writer. Because the lease is exclusive, the graph's write-ahead sidecar
  has exactly one writer too.
- Back up the complete graph before upgrades — including the `<graph>-wal`
  sidecar, or after a `CALL db.checkpoint()` that folds it in; see
  [Import and Export](../python/guides/import-export.md) and *Durability*.
- Use release benchmarks/CI reports for performance claims; this operator page
  intentionally avoids unversioned hardware-specific numbers.
