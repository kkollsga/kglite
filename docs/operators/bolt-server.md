# Bolt server

`kglite-bolt-server` exposes the embedded KGLite engine over Bolt v5.x. The
official Neo4j Python, JavaScript, and Java drivers are regression-tested.
Other Bolt v5 clients are untested. They must stay within the documented
protocol and [Cypher dialect](../reference/cypher-reference.md) limits.

## What this server is (and is not)

One process owns one graph and serves Bolt clients over the wire. Use it for
trusted or loopback access, or behind a proxy that owns authentication and
authorization. Reads run against snapshots and scale across concurrent
sessions.

- **No user directory and no RBAC.** `--auth basic` configures a single shared
  credential. The authenticated principal is validated at LOGON and not
  stored, so there is no per-session identity to authorize against. `--auth
  none` accepts any LOGON.
- **No high availability and no replication.** There is no failover, cluster,
  or bookmark/causal-consistency protocol.
- **One writer.** Writes serialize at commit within the process. One writable
  server per graph is enforced by a cross-process lease (see *Operations and
  security* below).
- **The graph file is not rewritten continuously.** Every commit is appended
  to a write-ahead log as it is acknowledged. The `.kgl` itself changes only
  when a checkpoint runs: `CALL db.checkpoint()`, `--checkpoint-interval`, or
  `--save-on-exit`. Turn the log off with `--durability off` and a commit is
  process-local until one of those runs. The file is then whatever it was when
  the server opened it. See *Durability* below for what each level costs and
  what it leaves behind.

If you need per-user access control, use a different shape. The supported
pattern is the
[derived-index / traversal-component pattern](../python/guides/derived-index.md#an-embedded-traversal-component-behind-your-api).
The engine is embedded behind your own API, and that API owns authentication,
authorization, and write policy.

For the feature-by-feature carry-over table (routing URIs, auth, auto-commit
mutations, OCC, and multi-database), see
[Migrating from Neo4j to KGLite](../python/migrations/neo4j-to-kglite.md).

## Install and start

```bash
cargo install kglite-bolt-server
kglite-bolt-server --graph /data/app.kgl
```

An existing `.kgl` opens in the storage mode it was saved in. A disk-graph
directory opens disk-backed. A missing path is an error unless creation is
explicit:

```bash
kglite-bolt-server --graph /data/new.kgl --storage memory
# --storage mapped|disk selects the other creation modes
```

`--storage` on an *existing* graph is a conversion request, not a no-op. A
memory-saved graph served with `--storage mapped` is converted to mapped before
the listener binds, and the startup log records `converted_from`.

A disk graph is a directory rather than a file, so converting into or out of
disk mode has no in-place form. Those requests fail startup naming
`enable_disk_mode()` instead of serving a mode nobody asked for. Omit the flag
to serve whatever the graph recorded.

### Durability defaults

The default is `--durability normal`. The server keeps a write-ahead log (`<graph>-wal`) beside the `.kgl` and appends each commit before it acknowledges it.

| Level | Promise |
|---|---|
| `normal` (default) | An acknowledged commit survives the server process dying. An OS crash or power loss can lose commits since the last checkpoint. |
| `full` | An acknowledged commit also survives power loss. Each commit waits for a device barrier. |
| `off` | No log. Commits stay in the process until a checkpoint writes them back. |

- Use `--durability full` when a power cut must not cost an acknowledged commit.
- The log is folded into the `.kgl` and truncated by a checkpoint. The size trigger is on by default at 16 MiB.
- Read-only servers and disk-mode graphs serve at `off`. See *Durability* below for the details.

### Memory sizing

The default `memory` storage mode holds the whole graph in RAM. Plan the server's memory for the graph plus working headroom.

- A running `db.backup()` raises memory use by roughly 10-30% of the graph's size (see *Writer cost* under *Backups*).
- A write transaction forks its working copy on the first mutation, so a large write can need extra memory while it runs.
- A very large single result (about 1M rows) peaks at several GB (see *Known limitations*).
- If the graph outgrows RAM, serve it with `--storage mapped`. Mapped mode spills property columns to mmap and keeps the per-commit log.
- `--storage disk` is for Wikidata-scale exploration. It keeps no per-commit log, so it serves at `--durability off`.

Important options (run `--help` on the installed version for the authority):

| Option | Purpose |
|---|---|
| `--bind`, `--port` | listener, default `127.0.0.1:7687` |
| `--storage memory\|mapped\|disk` | create a missing graph in this mode, or convert an existing one to it (memory ⇄ mapped; disk directions refused) |
| `--readonly` | reject mutations at execution |
| `--durability full\|normal\|off` | what an acknowledged commit survives, default `normal` (see *Durability*) |
| `--save-on-exit` | checkpoint the served graph back to `--graph` on `SIGINT`/`SIGTERM` |
| `--checkpoint-interval SECS` | checkpoint the served graph on a timer |
| `--checkpoint-wal-mib MIB` | checkpoint when the log passes this size, default `16` while a log is kept, `0` disables |
| `--auth none\|basic`, `--auth-user`, `--auth-pass` | Bolt LOGON policy |
| `--idle-timeout`, `--max-sessions`, `--max-message-size` | resource bounds |
| `--query-timeout MS`, `--max-work-units N`, `--max-rows N` | per-statement limits (see *Query limits and transaction timeouts*) |
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
when you want routing behavior. Set `--advertise-addr` to an address reachable
by the client, especially behind a proxy or when binding `0.0.0.0`.

## Valid time

On a graph that declares validity intervals, a statement with no
`FOR VALID_TIME` prefix reads as of today (UTC). `FOR VALID_TIME ALL` reads
every version.

- `--valid-time-default {today|all|YYYY-MM-DD}` sets the instant an unprefixed
  statement reads.
- A statement's own prefix wins.
- The setting is never written into the `.kgl` file. Without the flag, a
  default the graph stored in its file applies, else today.
- Each result's `kglite.temporal` summary key reports the `source` (`default`,
  `explicit`, `all` or `skipped:<reason>`), the instant and the rows the
  context hid.

## Transactions and errors

The backend uses native KGLite sessions and transactions, not Python or the
GIL. A plain `session.run` is auto-commit and may read, write data or change
schema. Each auto-commit write is a transaction of its own and commits before
the result is returned. Explicit transactions (`execute_write`,
`begin_transaction`) group several statements.

### Auto-commit writes

- A `CREATE`/`INSERT`, `SET`/`REMOVE`, delete form or `MERGE` in `session.run`
  commits as one transaction. The summary reports type `w`, or `rw` when the
  statement returns rows, plus the usual `stats`.
- The write is applied or not as a whole. A failed statement, a lost conflict
  or a log failure sends no rows and changes nothing.
- A session in read mode (`default_access_mode=READ`) is refused with
  `Neo.ClientError.Statement.AccessMode`. `--readonly` refuses every write.
- Drivers never retry `session.run`, so the server absorbs contention. In
  `queue` mode the write waits for the writer slot and obeys the wait timeout.
  In `optimistic` mode a lost commit race is retried up to three times, then
  fails with `Neo.TransientError.Transaction.Outdated`.
- The commit happens at RUN, before the result is pulled. Neo4j commits when
  the result is consumed, so a RESET or disconnect between RUN and PULL rolls
  back there. Here the write has already committed. A RESET or disconnect
  *during* the statement cancels it first (see *Cancelling a running query*).
- The server returns no bookmarks. It is a single process, and a commit is
  visible to every later query, so a session reads its own writes.
- `USING PERIODIC COMMIT` and `CALL { … } IN TRANSACTIONS` are unsupported.

### Schema statements

`CREATE INDEX`, `DROP INDEX`, `CREATE CONSTRAINT` and `DROP CONSTRAINT` run
through a plain `session.run`, as they do on Neo4j, and publish as a
transaction of their own. The result summary reports query type `s`. A refused
form, such as a `FULLTEXT` index, publishes nothing.

| Where | Neo4j | KGLite |
|---|---|---|
| `session.run` (auto-commit) | runs | runs |
| `execute_write` / `begin_transaction`, schema only | runs | runs, commits with the transaction |
| Same transaction writes data and schema | refused | **runs**, commits atomically |
| `CALL db.checkpoint()` | auto-commit only | auto-commit only, refused inside a transaction |

The mixed case is the one deliberate difference. A script that works on Neo4j
works here unchanged, and a transaction that groups an index with the data it
serves commits or rolls back as one.

Concurrent writers queue for a writer slot at BEGIN (see
[Write concurrency](#write-concurrency)). A writer that waits too long fails
with a retriable status code. Driver-managed transactions (`execute_write` and
its per-language equivalents) retry the unit of work by themselves.
Hand-rolled `begin_transaction` code needs its own retry loop.

Error codes:

- A `--readonly` server answers a write with
  `Neo.ClientError.General.ReadOnly`. A disk-mode graph answers a write with
  `Neo.ClientError.Security.Forbidden`. No rewrite of the request helps there.
- A write in a read-mode session or transaction is
  `Neo.ClientError.Statement.AccessMode`.
- KGLite typed errors map to Neo4j status codes for syntax, schema, timeout,
  access-mode, conflict, and execution failures.

### Query limits and transaction timeouts

By default the server has **no query deadline of its own and applies no limits**. That is a declared divergence from the Python API and the MCP server, which both apply the shared 180,000 ms default. Three flags set server-wide limits for every statement, auto-commit and inside an explicit transaction.

| Flag | Effect on overrun |
|---|---|
| `--query-timeout MS` | The statement fails with `Neo.ClientError.Transaction.TransactionTimedOut`. `0` or absent means none. |
| `--max-work-units N` | The statement fails with an execution error naming the `max_work_units` budget. It never truncates a result. |
| `--max-rows N` | The result is truncated to the first `N` rows. The summary carries `kglite.row_limit` as `{limit, total_rows}`. Writes still happen in full. |

A client `tx_timeout` is honoured:

- A top-level `tx_timeout` in RUN or BEGIN extra is a per-statement timeout in milliseconds. It applies to every statement of that transaction, not to the transaction as a whole.
- With `--query-timeout` set, the smaller of the two wins: a client can tighten the server limit and never loosen it.
- Zero, NULL, or absent means no client timeout. A negative or non-integer value is `Neo.ClientError.Request.Invalid`.
- A `tx_timeout` key nested inside `tx_metadata` remains ordinary user metadata.

### Cancelling a running query

A RESET or a dropped connection stops the query the session is running.

- **RESET** interrupts the running statement. The RUN fails with `Neo.ClientError.Transaction.Terminated`, then the RESET succeeds and the session is usable again.
- **A dropped connection** (end of input or a socket error without a preceding GOODBYE) stops the statement and releases its writer slot. A client that pipelines `RUN`, `GOODBYE` and half-closes still gets its statement run to completion.
- **Auto-commit writes** are cancelled before anything is published. A cancelled statement changes nothing.
- **Explicit transactions**: the cancelled statement fails and the transaction is rolled back by the RESET, or by the connection closing.
- **Limits**: the stop is cooperative, so it takes effect at the engine's next cancellation check, not instantly. A write waiting for the writer slot is cancelled once it is admitted. A RESET that arrives before its RUN starts, or after it finished, cancels nothing.

`boltr` reads one message at a time and does not read the socket while a RUN executes. The server therefore reads each connection's input itself and forwards it to `boltr` unchanged. It watches the message framing for RESET and for the end of input. A RESET sent behind a running RUN is seen at once, and `boltr` still answers the RUN and then the RESET in order. At most 256 KiB of input is buffered ahead of `boltr`. A client that floods more than that during one query delays the signals behind it until `boltr` catches up.

## Timezone-aware datetime parameters

A zoned PackStream temporal (`DateTime`, `DateTimeZoneId`, `Time`) is
**refused** with `Neo.ClientError.Request.Invalid`. KGLite's temporal values
are zoneless, so there is no lossless translation, and silently dropping the
zone would corrupt a driver's round-trip. Send `LocalDateTime`, `LocalTime` or
`Date` instead.

The Python API deliberately differs. It *converts* an aware `datetime` to
naive UTC. Its own Cypher `datetime()` constructor already does that to an
offset-bearing literal, and there is no wire type on that side to corrupt. See
{doc}`../python/value-projection` for the Python half.

The standing Bolt correctness and differential suites lock the supported
behavior. Avoid relying on an exact test/query count or a particular driver
patch version. CI exercises the complete current corpus.

### Write concurrency

A write transaction waits for the single writer slot when it begins, so
concurrent writers queue instead of conflicting at commit. Reads never wait.

- A write transaction is any explicit transaction not begun in read mode.
  Drivers begin in read mode for `execute_read` (`session.begin_transaction()`
  on a read-access session).
- The transaction takes the slot at BEGIN and holds it until commit, rollback
  or disconnect. It then runs on the latest graph, so its commit cannot
  conflict.
- Waiting writers are served in arrival order.
- Reads (auto-commit and read-mode transactions) run on snapshots and never
  take or wait for the slot. A snapshot never shows an uncommitted write.
- An auto-commit write, data or schema (`CREATE`/`DROP INDEX`, `CREATE`/`DROP
  CONSTRAINT`), takes the slot for its one-shot transaction and obeys the wait
  timeout.
- Automatic and periodic checkpoints never take the slot. They save only
  committed state, so an open writer's uncommitted work is never written, and
  they neither wait for nor delay a queued writer.
- A read-mode transaction cannot write. A mutation in one is refused with
  `Neo.ClientError.Statement.AccessMode`.

Keep write transactions short. The slot is held while your code runs between
statements, so a write transaction that only reads blocks every other writer.
Use `execute_read` for reads. To get more out of the single slot:

- Batch related writes into one transaction, for example with `UNWIND $rows`.
- Open the transaction, write, and commit without work in between.

#### Timeouts

| Flag | Default | Effect |
|---|---|---|
| `--writer-wait-timeout SECS` | `20` | A BEGIN that cannot get the slot in this time fails with the retriable `Neo.TransientError.Transaction.LockAcquisitionTimeout`. `0` waits indefinitely. |
| `--writer-idle-timeout SECS` | `10` | A holder with no activity for this long is rolled back once another writer is waiting for the slot. `0` never reclaims. |

- Both accept fractions, for example `0.5`.
- The wait default is below the 30 s default transaction retry budget of the
  Python, Java and JavaScript drivers, so a driver-managed transaction that
  times out still retries inside its budget.
- The idle default is below the wait default, so a stuck holder is reclaimed
  before the writers behind it give up.
- A holder nobody waits on is never reclaimed.
- A reclaimed transaction is rolled back whole. The client's next message gets
  `Neo.ClientError.Transaction.TransactionTimedOut`, and a ROLLBACK succeeds.
- A query that is running is never counted as idle.
- A dropped connection, a RESET, or a failed commit releases the slot at once.

A client that holds one write transaction open and waits for a second one it
opened itself blocks until the wait timeout. Run the two on one transaction.

#### Optimistic mode

`--write-concurrency optimistic` restores the earlier behaviour:

- Every write transaction runs on its BEGIN snapshot with no slot.
- A transaction whose snapshot was overtaken conflicts at commit with the
  retriable `Neo.TransientError.Transaction.Outdated`, even on disjoint keys.
- A read-mode transaction may write, and commits under the same rule.
- Driver-managed transactions re-run the unit of work on a conflict.
  Hand-rolled `begin_transaction` code needs its own retry loop.

Choose it only when writers rarely overlap and you want transactions to run
in parallel until commit. Under contention it spends the writers' time on
retries.

The opt-in `tests/benchmarks/test_bench_bolt_writers.py` measures writer count,
batch size, and durability on the current code. Correctness is pinned by
`tests/test_bolt_server_writer_queue.py`,
`tests/test_bolt_server_transactions.py` and
`tests/test_bolt_server_concurrency.py`.

## Durability

Two independent mechanisms decide what a stopped or killed server leaves
behind:

- A **write-ahead log** records each commit to a sidecar file as it is
  acknowledged. An interruption costs at most what the log does not hold.
- A **checkpoint** rewrites the whole `.kgl` from the committed graph and
  truncates the log.

They are complements, not alternatives. The log bounds the window while the
server runs. The checkpoint folds the log back into the file.

### Levels (`--durability`)

`--durability full|normal|off` selects what an *acknowledged* commit survives.
You can also set `KGLITE_BOLT_DURABILITY=<level>`; the flag wins if both are
set.

The frame is written **before** the client is told the commit succeeded. A
commit whose frame cannot be written is therefore not applied at all, and is
reported as a failure. The server never acknowledges a write it then discards.

| Level | An acknowledged commit survives | It does not survive |
|---|---|---|
| `full` | the server process dying, and — by asking the device for a write barrier before acknowledging — an OS crash or power loss | media failure, or anything the filesystem itself loses |
| `normal` (default) | the server process dying: `SIGKILL`, an OOM kill, a panic. The frame is in the kernel's page cache | an OS crash or power loss before the kernel writes that page out |
| `off` | nothing by itself — commits stay in this process until a checkpoint rewrites the file | the process ending at all, unless a checkpoint ran first |

What is pinned by test is the process-kill case. At `full` and at `normal`,
committing over Bolt and then `SIGKILL`ing the server with no checkpoint of any
kind leaves the `.kgl` byte-identical. The restarted server replays the commit
out of the log. `off` is the control: the same write is gone.

A user-space test cannot take the page cache or the power away. Power loss is
therefore pinned by crash images, not by cutting power: the engine's tests
rebuild the files a cut can leave and run recovery over them (see *Power loss*
below). They check the order of the barriers, not the drive.

### Power loss

A power cut keeps what was barriered and loses an arbitrary part of the rest.
The server restarts in every case below and never loads a half-applied commit.

| Level | After a power cut |
|---|---|
| `full` | Every acknowledged commit is present. At most the one commit in flight is lost, and only if it was not yet acknowledged. |
| `normal` | The log keeps a prefix of the commits since the last checkpoint. Later commits are lost, including acknowledged ones. |
| `off` | Everything since the last checkpoint is lost. |

Rules recovery follows:

- A frame is whole or discarded. A partial last frame is cut off at the next
  open.
- At `normal`, a cut can leave later log pages on disk and an earlier page
  unwritten. A frame that spans the gap fails its checksum with bytes after it.
- Damage with bytes after it is quarantined, never dropped and never guessed
  at. This covers a power-cut gap and any other checksum or length failure.
  The server copies the whole log to `<graph>.kgl-wal.quarantine-<UTC time>`,
  makes the copy durable, and keeps serving from the frames before the damage.
  The copy is never deleted. The server logs an error at startup with the copy's
  path, the byte offset of the damage and how many bytes and frames were set
  aside, and `graph_info()` lists a `wal_quarantined` advisory. Commits inside
  the set-aside bytes are not in the served graph; inspect the copy to recover
  them.
- If the copy cannot be written, the server refuses to open the log and says
  why. Free space or fix the directory and restart.
- A torn tail is cut off at the next open, and any non-zero byte in it is saved
  first. The server writes exactly the cut bytes to
  `<graph>.kgl-wal.torn-<UTC time>-at-<offset>`, logs the path, offset and byte
  count, and `graph_info()` lists a `wal_tail_saved` advisory. A crash
  mid-append leaves such a tail. A tail of zeros (space that was extended but
  never written) is cut without a copy. If the copy cannot be written, the
  server refuses to open the log.
- A commit whose frame cannot be written (full disk, failing barrier) is not
  applied, and the log is cut back to its last whole frame. If the cut-back
  itself fails, the log refuses further commits until the server restarts.

The guarantee holds only if the device honours write barriers. A drive or SD
card that acknowledges a flush it has not performed can lose barriered data at
any level.

On a device that loses power without warning, run `--durability full`.

The default is `normal`. `full` takes a device barrier for every commit while
holding the commit lock, so it can sharply reduce write throughput and increase
contention. Enable it when acknowledged commits must survive an OS crash or
power loss, and measure the effect on representative storage and workload. The
corresponding sweep is
`tests/benchmarks/test_bench_bolt_writers.py::test_durability_sweep`
(`-m "benchmark and bolt_stress"`).

### Recovery on startup

Recovery is unconditional and runs before the listener binds, at every level.
Opening a path is a decision about that path's *data*, not only about how
future writes will be logged.

- At `full` and `normal`, a sidecar holding commits the `.kgl` does not
  contain is replayed into the graph.
- At `off`, the same sidecar is a startup error naming both ways out. Restart
  at `full` or `normal` to replay the commits, or move the sidecar aside to
  discard them deliberately. A server that would otherwise serve a graph
  missing acknowledged writes does not start.
- Frames the checkpoint already contains are not grounds to refuse, and open at
  every level. They are the harmless residue of a crash between a checkpoint's
  file write and its log truncation.

### Server-facts verbs

Besides `db.checkpoint()` (below), three read-only verbs are answered at the
Bolt layer. They report server state the engine does not hold. They are the
introspection calls Neo4j clients make on connect:

- **`CALL dbms.components()`** — name / versions / edition, following
  `--neo4j-compat` (see *Driver identity*). Edition is always `community`.
- **`CALL dbms.showCurrentUser()`** — the configured `--auth-user`
  (or `neo4j` under `--auth none`); roles and flags are empty. Answered from
  server config: there is deliberately no per-session principal.
- **`SHOW DATABASES`** — one row named `neo4j` (matching the routing
  default), `default`/`home` true, `access`/`writer` reflecting
  `--readonly`. The session `database=` field remains accepted-and-ignored;
  this row is informational.

All three take an optional `YIELD` naming a subset of their declared columns,
like `db.checkpoint()`. They exist only over Bolt, because in-process bindings
have no server to describe. To see what any client sends on connect, run the
server at `RUST_LOG=debug`. Every incoming query is logged.

### Checkpoints

Four routes rewrite the served `.kgl`, and all four are the same operation:
flush the log, stamp the checkpoint position, write the file, truncate the log.

- **`CALL db.checkpoint()`** — on demand, over the wire. It is a *bolt-server
  verb*, not an engine procedure. It exists only over Bolt, and embedded
  bindings keep their own save calls. It answers in Neo4j's `success, message`
  shape, and an optional `YIELD` of either or both columns is honoured. A
  checkpoint whose graph has not changed since the last one *in this process*
  is skipped and says so. The first call of a process always writes, because
  the file may predate the process.
- **`--checkpoint-interval SECS`** (`KGLITE_BOLT_CHECKPOINT_INTERVAL`) — on a
  timer. The interval task and the verb share one recorded version, so a
  checkpoint by either makes the next tick a skip. An idle server does not
  rewrite its file. A failed tick is logged as an error and the server keeps
  serving. The interval is validated at startup rather than starting a server
  that silently never checkpoints.
- **`--checkpoint-wal-mib MIB`** (`KGLITE_BOLT_CHECKPOINT_WAL_MIB`) — on log
  size, and **on by default** at `full` and `normal`. Every 10 seconds the
  server asks the session whether the log has passed the threshold (default
  16 MiB) and is at least as large as the `.kgl`, because a log bigger than the
  file it extends costs more to replay than to rewrite. It then runs an online
  checkpoint: the session locks are held only to fix the snapshot and to trim
  the log, so writers keep committing during the file write.
  `0` disables it. An explicit value is refused with `--readonly` and for
  disk-mode graphs. The default does not apply there, or at `off`, where
  there is no log. It shares the recorded version with the verb and the
  interval task, so an unchanged graph is skipped.
- **`--save-on-exit`** (`KGLITE_BOLT_SAVE_ON_EXIT`) — once, on `SIGINT` or
  `SIGTERM`, after periodic checkpointing has been stopped. The saved graph
  version is logged. A failed exit save is logged as an error *and* exits
  non-zero, so a supervisor sees it. Connections are not drained, so a commit
  racing shutdown can land after the save. The logged version is how you tell
  that apart from a save that never ran. Under a log, the commit is still in
  the sidecar and the next start replays it.

A checkpoint is power-safe at every level. It orders its steps so that a cut
between any two leaves a graph that loads:

1. The log is barriered (`normal` needs this; `full` already did it).
2. The new `.kgl` is written to a temporary file and barriered.
3. The temporary file is renamed over the `.kgl`, and the directory is synced.
4. The log is truncated and barriered.

A cut before step 3 completes keeps the old `.kgl` and the whole log. A cut
before step 4 completes keeps the new `.kgl` and the whole log; frames the
`.kgl` already contains are skipped on replay. If the directory sync fails,
the checkpoint reports the error and leaves the log untouched. A stray
temporary file from a cut is removed at the next start.

A checkpoint pauses writers and new snapshots for its duration, because
`Session::save` holds the session lock for the full save. Readers already
holding a snapshot are unaffected. Time one `CALL db.checkpoint()` on a
representative graph before choosing the interval.

Retention is one: each checkpoint atomically replaces the previous file. Use
`db.backup()` or `--backup-interval` (see *Backups*) if you want history.

### The sidecar file

At `full` and `normal` the server keeps `<graph>-wal` beside the graph file.
Every checkpoint truncates it back to its header. Between checkpoints it grows
by under a hundred bytes per single-node commit. Multiply that by your commit
rate to size it.

The log is bounded by default: `--checkpoint-wal-mib` checkpoints once it
passes 16 MiB (and the size of the `.kgl`). Set `--checkpoint-wal-mib 0` and
nothing else, and the sidecar grows with uptime and a restart replays all of
it. Replay folds the log frame by frame, so its memory follows the distinct
nodes the log names, not the log's length. On macOS a 6 MB and a 56 MB log over
a bounded set of nodes each restarted 7-9 MiB above an idle server, while a log
that keeps creating and deleting new nodes costs up to five times its size. `--checkpoint-interval` adds a timer on top.
Neither makes commits safer, because the log already did that. They keep
replay time and sidecar size bounded.

Use `CALL db.backup()` (see *Backups*) for a live copy. If you copy files
yourself, copy the sidecar with the graph, or checkpoint before copying the
`.kgl` alone. A `.kgl` copied while a sidecar runs ahead of it is missing the
commits the sidecar holds. The engine refuses the dangerous half of this by itself. A
non-durable open, and a save, over a path whose sidecar runs ahead are errors
rather than silent data loss.

### Refusal matrix

Two configurations cannot carry a log or a checkpoint:

- `--readonly`: a server that never commits has nothing to log and nothing to
  write back.
- Disk-mode graphs: a disk graph commits by publishing an immutable generation,
  so it keeps no logical log. Every disk save publishes a *new* generation that
  nothing prunes, so repeated checkpoints would grow the directory without
  bound.

An explicitly requested level or feature is refused there. The *default* level
degrades instead, so flipping the default did not turn every read-only and
disk-mode server into a startup error:

| Configuration | `--durability full`/`normal` (asked for) | `--durability` (default) | `--save-on-exit`, `--checkpoint-interval`, `--checkpoint-wal-mib` (asked for) | `CALL db.checkpoint()` |
|---|---|---|---|---|
| `.kgl`, writable | serves at that level | serves at `normal` | supported | supported |
| `--readonly` | startup error | serves at `off`, logged | startup error | `Neo.ClientError.General.ReadOnly` |
| disk-mode graph | startup error | serves at `off`, logged | startup error | `Neo.ClientError.Security.Forbidden` |

The one refusal that is about *data* rather than configuration is `off` over a
sidecar that runs ahead of the file, above. A level nobody asked for replays it
instead, which is what makes the default safe to inherit.

Environment mirrors are refused exactly as the flags are. A mistyped level or
interval is a startup error rather than a server that silently logs nothing.

`CALL db.checkpoint()` is also refused inside an explicit transaction. It
writes the *committed* graph, which by definition excludes that transaction's
uncommitted writes. Commit first and call it in auto-commit.

`tests/test_bolt_server_durability.py` (`-m bolt`) pins the behavior above. It
includes the `SIGKILL`-and-restart tests behind each level, the
checkpoint-truncates-the-log test, and every row of this matrix.

## Enforced ontology

`--ontology FILE` declares an ontology when the server starts and enforces it on every write, whichever client sends the write.

```bash
kglite-bolt-server graph.kgl --ontology ontology.json
```

`FILE` is the JSON document `define_ontology()` takes (see the [ontology guide](../python/guides/ontology.md#write-time-enforcement)):

```json
{
  "classes": {
    "Person": {"required_properties": ["name"], "property_types": {"name": "string"},
               "enforcement": "error"},
    "Company": {}
  },
  "relationships": {
    "WORKS_AT": {"domain": "Person", "range": "Company", "enforcement": "error"}
  }
}
```

### What a client sees

- **`error`.** A write that breaks a rule fails with `Neo.ClientError.Schema.ConstraintValidationFailed`. The message names the rule, the label or relationship type and the property. In an explicit transaction the failing statement rolls back the whole transaction, including the statements before it.
- **`warn`.** The write succeeds. The result summary carries `kglite.ontology = {warnings: [...]}` and the server logs one line per violation.
- **`SHOW ONTOLOGY`** returns the declaration to any client.

### The lock

An ontology given with `--ontology` is **locked**. `CALL db.ontology.declare()` and `CALL db.ontology.clear()` are refused for every client, with an error naming `--ontology`. Changing the ontology takes an operator action: restart with a new file.

The server has one credential and no roles. "Authenticated" means the client has the password, so without `--ontology` any authenticated client may declare or clear an ontology. Use the flag whenever the shape of the data must not depend on a client.

### Restart

- The declaration persists with the graph at the next checkpoint (`CALL db.checkpoint()`, `--checkpoint-interval` or `--save-on-exit`), and in the write-ahead log at `--durability normal` and above. The lock does not persist: it applies for the life of the process.
- A stored ontology equal to the file starts normally.
- A stored ontology that **differs** from the file stops the start. The message shows a short diff and names `--ontology-replace`.
- `--ontology-replace` replaces the stored ontology with the file's. It requires `--ontology`.
- A file whose `error` rules the stored data already breaks stops the start with the per-rule report: the same refusal as declaring over existing data. Fix the data or lower the rule to `warn`.
- A missing or malformed file stops the start.

A backup taken with `CALL db.backup()` carries the ontology. A server restored from it enforces the same rules once started with the same `--ontology` file.

## Backups

`CALL db.backup('<name>')` writes a consistent single-file `.kgl` of the committed graph while the server keeps serving.

```bash
kglite-bolt-server graph.kgl --backup-dir /var/backups/kglite
```

```cypher
CALL db.backup('nightly.kgl')
```

The verb is off until the server starts with `--backup-dir`. Without it, `db.backup()` is refused.

### Flags

| Flag | Meaning |
|---|---|
| `--backup-dir DIR` | Enables `db.backup()`. Clients pass a bare file name and the server writes `DIR/<name>`. `DIR` is created if missing. |
| `--backup-allow-any-path` | Lets clients name any path the server can write. Refused at startup with `--auth none`. With `--backup-dir` also set, bare names still land in `DIR` and only absolute paths go elsewhere. On Windows a path with a root but no drive (`\x.kgl`) is refused as ambiguous. |
| `--backup-interval SECS` | Writes a backup into `--backup-dir` every `SECS` seconds. Requires `--backup-dir`. |
| `--backup-keep N` | Keeps only the newest `N` scheduled backups. Requires `--backup-interval`; `N` is at least 1. |

### Result columns

`db.backup()` yields one row: `success`, `path`, `lsn`, `nodes`, `relationships`, `bytes`, `lock_hold_ms`, `elapsed_ms`, `graph_version`, `prepared_copy`. `graph_version` is the version of the snapshot written; `prepared_copy` is true when the snapshot needed a private prepared copy first. `lsn` is null when the server keeps no write-ahead log. Like the other verbs, it accepts a `YIELD` naming a subset of these columns.

### Path policy

- **Bare names only by default.** A name containing `/`, `\`, `..` or an absolute path is refused.
- **Credentials.** The server has one credential and no roles. "Authenticated" means the client has the password. Anyone who can log on can write a backup into `--backup-dir`.
- **Why `--backup-allow-any-path` needs auth.** Under `--auth none` every client that can connect would gain a file-write primitive, so the server refuses to start.

### What is refused

- A call inside an explicit transaction.
- A disk-mode graph.
- A name that is the served graph itself.
- A second call while one runs: "backup already in progress".

`--readonly` servers may back up. The file is written to a temp name, fsynced and renamed, so a killed server leaves no partial destination, and an existing backup of the same name stays intact until the new one is complete.

### Scheduled backups

With `--backup-interval`, the server writes `<graph-stem>-YYYYMMDDTHHMMSSZ.kgl` (UTC) into `--backup-dir`.

- The first backup lands one interval after startup.
- A tick is skipped when the graph is unchanged since the last scheduled backup, or when a `db.backup()` is still running.
- A failure is logged and the server keeps serving.
- With `--backup-keep N`, the oldest files matching this name pattern are deleted after each successful scheduled backup. Files from `db.backup()` and any other file in the directory are never touched.

### Writer cost

`backup()` holds the commit path for under a millisecond regardless of graph size. A writer that only adds or updates nodes is not slowed.

- A writer that creates or deletes relationships sees at most one commit up to about 2x slower than usual during the backup.
- Measured worst case at 1 million nodes and 3 million relationships: 0.22 s in memory mode and 0.25 s in mapped mode (one 0.46 s outlier), against 0.1 s for such a commit without a backup.
- Memory use rises by roughly 10-30% of the graph's size while the backup runs.

### Restoring a backup

A backup is an ordinary `.kgl`: serve it, or put it in place of the live graph.

1. Stop the server.
2. To restore over a live path, move the old `<graph>-wal` sidecar aside first. A log that runs ahead of the restored file is refused on startup, correctly.
3. Copy the backup to the graph path and start the server.

```bash
mv graph.kgl-wal graph.kgl-wal.old
cp /var/backups/kglite/nightly.kgl graph.kgl
kglite-bolt-server graph.kgl
```

Cross-architecture portability of a backup file is not yet tested; restore on the architecture that wrote it. For the in-process equivalent, see [Backups and restore](../python/guides/durable-apps.md#backups-and-restore).

## Driver identity (`--neo4j-compat`)

By default the handshake `server` agent and the `CALL dbms.components()` row
(name, versions, edition—always `community`) report
`kglite-bolt-server/<version>`. Under `--neo4j-compat` they report the
Neo4j-compatible spelling. The separate `bolt_agent` metadata always names
`kglite-bolt-server/<version>` honestly. Compatibility mode changes only the
fields clients use for their Neo4j product gate.

Two client families need the compatible spelling:

- the official **Java** driver requires a `Neo4j/` agent prefix and refuses
  the connection outright without one:

  ```
  UntrustedServerException: Server does not identify as a genuine Neo4j
  instance: 'kglite-bolt-server/<version>'
  ```

- **GUI clients** (Neo4j Browser, G.V(), and other IDEs) read
  `dbms.components()` to decide product and feature support, so they need
  the flag too. Under it the row reports `Neo4j Kernel` / `5.26.0` /
  `community`. The official Python and JavaScript drivers accept the honest
  default.

Enable compatibility mode to serve those clients. Either route works, and the
flag wins if both are set:

```bash
kglite-bolt-server --graph graph.kgl --neo4j-compat
KGLITE_BOLT_NEO4J_COMPAT=1 kglite-bolt-server --graph graph.kgl
```

The agent then becomes `Neo4j/5.26.0 (kglite-bolt-server/<version>)`. That is
enough of a Neo4j spelling to pass the driver's check. The real product is
retained, so the server stays identifiable in logs, in driver errors, and
through `ServerInfo.agent()`. The handshake `server` field and
`dbms.components()` row change; `bolt_agent` keeps reporting kglite.

The variable accepts `1`, `true`, `yes` or `on` (any case). That is the useful
form for container images and unit files, where adding an argument means
rebuilding or editing a unit.

The mode is off by default on purpose. Presenting as a different product is the
operator's call, and the identity is never switched automatically. When a
driver that enforces the check connects with compatibility off, the server logs
a warning naming both activation routes. An operator can then diagnose it from
the server log instead of a client stack trace.

## Known limitations

The server speaks Bolt through the `boltr` library (0.2.0). These defects of
that version are visible to drivers.
A `boltr` release that fixes them will remove the limits.

| Behaviour | What a driver sees | Workaround |
|---|---|---|
| One open result per explicit transaction. A second RUN while the first result has unread rows is not tracked by query id. | A `ValueError` about keys and values of different length (neo4j Python driver), once a result is longer than the driver's `fetch_size`. | Read each result to the end before the next `tx.run`, or raise `fetch_size`. |
| A very large single result (about 1M rows) peaks at several GB of server memory. | The server's resident memory grows with the result until the `boltr` 0.2.1 bump. | Paginate, or use `LIMIT`. |
| A message that is invalid in the current state is answered IGNORED, not FAILURE. | None from the official drivers, which do not send such messages. | None needed. |

The server guards the other `boltr` 0.2.0 defects itself: authentication
before any query, nesting depth, message size before LOGON, session cleanup on
disconnect, idle reaping of paging clients and the result summary of DISCARD.

## Operations and security

- **Loopback is the safe default.** If you expose the server remotely, enable
  basic auth and TLS, or terminate TLS/auth at a trusted proxy/firewall
  boundary.
- **Bound resources on untrusted networks.** Set `--max-message-size`,
  `--max-sessions`, and an idle timeout whenever the listener is reachable from
  an untrusted network. They bound resource use per connection; they are not an
  access-control boundary.
- **Use `--readonly` for read-only analytical instances** sharing the same
  graph file, and for agent connections that do not need writes. A `--readonly`
  server is a second process opening the same graph, not a replica. It serves
  what the file contained when it opened it. Beside a durable writer, that
  excludes whatever the writer has committed to its sidecar since the last
  checkpoint (see *Durability*). A read-only server keeps no log itself and
  serves at `--durability off`.
- **One writable server per graph.** A server started without `--readonly`
  takes the same cross-process writer lease as `kglite.open()` *before* it
  reads the graph, and holds it until shutdown.
  - A second writable server, a CLI write, or `kglite.open()` on that path
    fails at startup naming the holding process, instead of racing it to
    overwrite at save time.
  - The refusal is immediate rather than a wait, so a supervisor's restart
    policy governs the retry.
  - A write-enabled MCP server on that path boots fine and is refused at its
    first mutation instead, because it takes the lease lazily. A library
    `load()` + `save()` still opts out of leases (pending WAL recovery can
    separately refuse it). See
    [the MCP server's operating notes](mcp-server.md#the-writer-lease-and-several-servers-on-one-file).
  - `--readonly` servers take no lease and start alongside a live writer.
  - Because the lease is exclusive, the graph's write-ahead sidecar has exactly
    one writer too.
- **Back up the complete graph before upgrades.** Use `CALL db.backup()` (see
  *Backups*), or include the `<graph>-wal` sidecar, or copy after a
  `CALL db.checkpoint()` that folds it in. See
  [Import and Export](../python/guides/import-export.md) and *Durability*.
- **Use release benchmarks/CI reports for performance claims.** This operator
  page intentionally avoids unversioned hardware-specific numbers.
