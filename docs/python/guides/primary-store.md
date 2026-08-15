# KGLite as a primary store: scope and limits

Most KGLite graphs are a {doc}`derived index <derived-index>` over data owned
somewhere else. This page is about the other case: the graph *is* the
authoritative copy, and losing it means losing the data.

That is a much stronger promise, so this page is deliberately conservative. It
states what holds, what the defaults are, and what KGLite does not do — with
enough detail that you can decide without running an experiment first. Where a
limit exists, it is named rather than softened.

## What holds

**A mutating statement is all-or-nothing.** A Cypher statement that fails after
its first write leaves the graph exactly as it found it — node and relationship
identity, properties, labels, index ordering, schema metadata, and the version
counter. Rollback replays a statement-scoped journal of inverse operations
backwards.

**The cost of a write scales with the change, not with the graph.** Because the
journal records only what the statement touched, a single `SET` on a
million-node graph costs about what it costs on a thousand-node graph. This is
the property that separates a store you can write to continuously from an index
you rebuild. It is measured, not assumed:
`tests/benchmarks/test_bench_write_scaling.py` runs the same statements at 1 k,
100 k, and 1 M nodes, and a reading that grows with size is a regression.

**One** case keeps the older whole-graph checkpoint, and there the write cost is
still proportional to graph size: the **`disk`** backend. A disk graph has no
petgraph slot identity for an inverse edit to name, so every mutating statement
on it opens an O(V+E) checkpoint instead. `memory` and `mapped` both take the
journal, and a durable graph over either of them does too — durability and
rollback strategy are independent concerns.

Two cases that used to be on that list are not any more, because the journal
grew to cover them:

- **A graph that has been saved, loaded, or opened from a file.** It carries the
  same property shape a freshly built graph does, and a `SET` journals the
  individual cells it overwrote rather than a copy of the type's whole column
  store. A single-row `SET` measures 4.3–5.0 µs at 50 k nodes × 12 declared
  properties and 4.3 µs at 100 k — parity with a graph that has never touched a
  file, and flat in node count. Inside an explicit transaction the same
  statements cost 14–45 µs each. A graph whose columns have spilled to disk
  under `set_memory_limit` measures 4.4 µs and keeps its spill; `mapped`
  measures 5.0 µs unlogged and 7.5 µs at `durable="normal"`.
- **A graph carrying user-created property, range, or composite indexes.** Their
  bucket edits are journalled with the position they occupied, so `CREATE INDEX`
  and the `create_index` API no longer move a graph's writes back onto the
  whole-graph checkpoint.

Uniqueness constraints are the one structure still rebuilt wholesale, and only
on the *failure* path: a statement that rolls back recomputes the occupancy map
of each type it touched, which scales with that type's node count. Successful
writes never pay it. If your workload both writes continuously and fails
statements often, measure it rather than assuming flat cost.

### What a write costs in memory

Properties live in per-type columns, from the first node onward — building,
saving, loading and reopening all produce the same shape, so there is no
conversion step to plan around and no second cost profile to discover after the
first `save()`. Two consequences are worth knowing before you rely on this as a
primary store:

- **A column is allocated per declared property, per row, whether or not the row
  has a value.** Memory is therefore proportional to schema width rather than to
  the properties actually set, so a type with many optional properties costs
  more at rest than the same data in a narrow type.
  `graph_info()['columnar_heap_bytes']` reports how much the stores hold, and
  `set_memory_limit()` spills columns to disk when they exceed a budget — the
  limit is re-enforced after every statement, not only at load time.
- **`DELETE` tombstones rows; `vacuum()` reclaims them.** Deleted rows keep
  their space until you compact, so a write-heavy primary store should call
  `vacuum()` periodically (it also fires automatically once fragmentation
  crosses `auto_vacuum_threshold`). Scans are unaffected — measured at
  0.975–1.043× with 40% of rows tombstoned — so the cost of putting this off is
  memory, not time. **`vacuum()` is a no-op on `storage="disk"`** — its node
  numbering is frozen mmap, so there is no in-place rebuild to do. `save()`
  reclaims instead: a disk save rewrites the columns without the rows no live
  node points at, so the published directory and the graph that reloads from it
  carry live rows only (measured: a 20k-node graph with half its nodes deleted
  wrote 2.00x the column bytes of the same graph built from the survivors, and
  now writes the same bytes). What a save does *not* reclaim is node slots: a
  deleted node's 16-byte slot and its free-list entry are kept, so a disk
  graph's node capacity only shrinks when the directory is rebuilt from a fresh
  ingest. `compact()` is a separate, edge-only operation — it merges overflow
  edges into the CSR and touches no rows.

Batching mutations into multi-row statements is still worth doing — it
amortises per-statement parsing, planning and checkpoint overhead — but it is
now a throughput optimisation rather than a workaround. See
{doc}`data-loading`'s throughput ladder.

**Crash safety is the default.** `kglite.open(path)` opens in write-ahead-log
mode wherever the storage mode supports it — the default in-memory backend and
`storage="mapped"`. Each committed mutation appends one frame to a `<path>-wal`
sidecar and `fsync`s it before the call returns; on open, the engine loads the
`.kgl` checkpoint and replays every frame newer than it. A frame carries a
CRC32, and a crash mid-append leaves a torn trailing frame that replay discards
rather than half-applying. Every way of changing a graph is logged, not only
Cypher — `add_nodes`, `add_connections`, label changes, and committed
transactions included. `save()` is separately atomic and `fsync`ed, so a reader
never observes a torn file.

`storage="disk"` is the exception, and not because it was overlooked: a disk
graph commits by publishing an immutable generation, so a logical write-ahead
log is not its durability boundary. A disk graph opens non-durable and takes
`save()` checkpoints instead. Asking for any logging level there —
`durable=True`/`"full"` *or* `durable="normal"` — raises `ValueError`
explaining that, since the blocker is the commit boundary rather than barrier
strength; only `durable="off"` is supported. The *default* does not raise, so
disk callers are unaffected by the default being on elsewhere.

**Not everything is logged.** State the log cannot express is *checkpoint-only*
and is persisted by `save()` rather than by the log: schema and config metadata,
user-created indexes, embeddings, and timeseries. If those matter to you, a
`save()` is still part of your durability story, not an optimisation.

Three consequences worth internalising before you rely on this:

- **It costs one barrier per committed mutation.** Writes now wait on physical
  storage, so the cost is device latency rather than graph size — most visible
  in loops of many small writes, negligible for a few large ones. Reads are
  unaffected. `durable="normal"` keeps the log and drops only that barrier: a
  committed mutation still survives the process dying, but an OS crash or power
  cut loses work since the last `save()`, and `sync()` gives you a power-safe
  point on demand. `durable=False` (no log at all) remains fully supported and
  is the right choice for bulk loading and for graphs you can rebuild from
  source. Batching writes into one statement, or one `begin()` transaction,
  buys throughput *and* the strongest guarantee.
- **A `with` block is not a transaction.** Mutations commit as they run, so an
  exception inside the block does not discard them — they are recovered on the
  next `open()`. What the clean or failed exit controls is whether a *checkpoint*
  is written. Use `begin()` when you want discard-on-error.
- **`Session` refuses write queries on a durable graph.** Its writes land on a
  working copy that neither the log nor `save()` can reach, so rather than
  silently losing them it raises. Use `cypher()` or `begin()`. Because durable is
  now the default, code that used `Session.execute()` for writes against an
  `open()`ed graph has to change — or pass `durable=False`.

Two smaller sharp edges: `save(fsync=False)` is ignored on a durable graph and
warns, because the checkpoint truncates the log and so must itself reach disk;
and a log written by this version is refused by older builds with a clear message
rather than silently truncated.

**Readers see a consistent graph.** `freeze()` hands out an immutable, lock-free
snapshot; a `session()` serializes writers, begins each from the last committed
state, and publishes with a pointer swap only on success — though note the
durable-graph restriction on `Session` writes below. Explicit transactions
are optimistically concurrent: a commit against a state that moved underneath
raises `TransactionConflictError` rather than winning silently.

Be precise about how coarse that check is, because it is coarser than most
databases you have used. It compares a **whole-graph version counter**, not the
read/write sets of the two transactions — a commit publishes the transaction's
working copy by pointer swap, so a transaction that began before *any* other
commit is working from a stale snapshot regardless of which nodes it touched.
Two transactions editing entirely unrelated nodes therefore conflict, and the
second one loses. That is not over-caution: its working copy genuinely does not
contain the first one's write, so applying it would silently revert that write.

The practical consequence is that conflicts are ordinary rather than rare, and
every concurrent writer needs a retry loop. Use `kglite.retry_on_conflict`
rather than writing one:

```python
def signup(tx):
    tx.cypher("CREATE (u:User {email: $email})", params={"email": email})

kglite.retry_on_conflict(graph, signup)
```

If your workload has many short concurrent writers, prefer `session()` — it
serializes writers and begins each from the last committed state, so they queue
instead of colliding. {doc}`/concepts/concurrency` is the full model, and worth
reading before you rely on any of it.

**Failures are typed.** Errors arrive as a `KgError` hierarchy with stable
codes, not as strings to match on — see {doc}`/python/error-handling`.

**Integrity constraints are enforced on every write path.** Declared through
`define_schema`, and checked on Cypher `CREATE` / `MERGE` / `SET` / `REMOVE` and
on the bulk loaders alike — `add_nodes`, and therefore blueprints,
`from_records`, OKF ingestion, WAL replay, and `extend_graph`:

```python
graph.define_schema({"nodes": {"Person": {
    "primary_key": "email",            # unique *and* present (NODE KEY)
    "unique": [["first", "last"]],     # composite UNIQUE
    "required": ["email"],             # NOT NULL, at write time
}}})
```

Three things make this real rather than advisory. `primary_key` may name any
property, not just `id` — a key on `id` routes through the O(1) identity index,
any other key is backed by a unique secondary index that persists and rebuilds on
load. `required` is enforced at write time, so a `CREATE` that omits the
property, a `SET` that nulls it, and a `REMOVE` that drops it all raise, rather
than surfacing later in `validate_schema()`. And declaring a constraint the
stored data already violates is refused outright, so you cannot install a
constraint that quietly lies about the rows already present.

A composite `unique` tuple constrains only nodes carrying *every* property in it,
and NULL is exempt throughout — a node sits outside a uniqueness constraint
unless every property in the tuple is present and non-null, so many nodes may
share "no email" while `email` is `UNIQUE`.

Two gaps to know, because both are the kind that look like guarantees until they
are not. **A large bulk load is not all-or-nothing.** `add_nodes` gates each row
before queueing it, so a violation aborts the call — but rows are flushed to the
graph in chunks of 1000, so on an input larger than that, chunks already flushed
stay written. Detection is unaffected; the atomicity is what is chunk-bounded.
Treat a failed large load as needing cleanup, not as a no-op. **And two write
paths bypass enforcement entirely:** the RDF and N-Triples loaders, and the
embedding-carry path. A graph filled through those can hold data that violates a
declared constraint; `verify_unique_constraints()` exists to audit exactly that
case.

One thing about the error surface is worth knowing before you write `except`
clauses. A violation raises `ConstraintViolationError` and a declaration that
cannot be installed raises `ConstraintCreationError`; both subclass
`ConstraintError`, so `except ConstraintError` catches either. This holds on
every write path — `cypher()` and the bulk writers alike — so the duplicate-signup
handler is a type check, not a substring match:

```python
try:
    graph.cypher("CREATE (u:User {email: $email})", params={"email": email})
except kglite.ConstraintViolationError:
    raise Conflict("that email is already registered")
```

Each carries a stable `.code` (`"ConstraintViolation"` /
`"ConstraintCreationFailed"`) for logging and cross-binding dispatch. Note that
`define_schema` *can* fail this way, because installing a schema installs the
constraints it declares — nothing is changed when it does, so you can fix the data
and retry. The message still names the constraint, the property, and the
offending value, and is worth logging; the type and code are the contract.

## Defaults, and how to change them

| | Default | To change |
|---|---|---|
| Crash safety | **On** (`"full"` — survives power loss) for in-memory and `mapped`; `disk` opens non-durable | `durable="normal"` to keep the log without the per-commit barrier, `durable="off"` to opt out entirely |
| Schema | No schema; any property on any node | `define_schema(...)` |
| UNIQUE / NOT NULL / node key | Permissive — a type declaring none keeps the old behaviour | `unique` / `required` / `primary_key` in `define_schema` |
| Freshness stamps | Off, so writes stay deterministic | `auto_timestamp: True` per type |

All of the constraint machinery is opt-in, and older graphs load unchanged.
{doc}`durable-apps` covers the `open()` lifecycle and the per-commit `fsync`
cost in more detail.

## What KGLite does not do

**One process writes — and this is now enforced, not just documented.**
`kglite.open(path)` takes an exclusive cross-process writer lease for as long
as the graph can write back to `path`, so a second process opening the same
path fails immediately with the holder's pid rather than quietly overwriting
its work at `save()`:

```
KgError: app.kgl is open for writing by pid 4711 (since 2026-07-26T09:15:03+02:00)
```

Readers are unaffected: `load()` and `open_session()` take no lease, so any
number of processes can read a graph while one writes. The lease is an OS-owned
lock, so a writer killed with `SIGKILL` releases it immediately — the leftover
`<path>.lock` (the lock, always empty) and `<path>.lock-owner` (the pid and
acquisition time, used to name a holder) are records, not the lock itself, and
deleting them achieves nothing. `open(..., lock=False)` opts out for callers
that coordinate writers some other way.

There is still no shared live multi-process transaction handle and no
replication protocol. Disk mode publishes immutable generations behind the same
kind of lease — that is stable-reader/single-writer publication, not concurrent
multi-process write access. When several processes need to read and write one
graph, `kglite-bolt-server` is the coordination point: that one process owns
the graph while clients connect over the Bolt protocol. It does not lift the
single-writer model — it centralises it. All writes go through explicit
transactions (auto-commit mutations are refused), they serialize at commit, and
a commit against a stale snapshot conflicts with a retriable status code so
driver-managed transactions retry on their own; that retry is contention-tested,
not merely lifecycle-tested
(`tests/test_bolt_server_transactions.py::test_managed_transaction_retries_after_conflict`).
The official Python, JavaScript, and Java drivers are regression-tested in CI —
session and explicit-transaction lifecycle, managed retry, PackStream type
round-trips, `Neo.*` error codes, and OCC conflict detection. Read that as a 22-check
conformance suite per driver rather than a full protocol sweep, and note that
every *other* driver — Go, .NET — remains untested: those clients may connect but
can rely on features outside the documented wire and Cypher contracts.

**Constraints cover uniqueness and presence, not arbitrary rules.** There is no
`CHECK` constraint, and no standing referential-integrity constraint between node
types: a relationship to an unknown endpoint auto-vivifies a provisional stub
rather than being rejected. Stubs are *deferred*, not exempt — the `add_nodes`
upsert that promotes one is a normal, fully-enforced write, an unpromoted stub
stays reportable via `validate_schema()`, and `purge_provisional()` sweeps them.
Individual loads can also be strict up front with
`from_records(..., on_missing_endpoint="error")`, which validates the whole input
and fails atomically. If your correctness argument needs a rule that is not
uniqueness or presence, it still belongs in your application.

**Schema setup is expressible in Cypher, with two asymmetries to know.**
`CREATE [RANGE] INDEX` / `DROP INDEX` / `SHOW INDEXES` and
`CREATE CONSTRAINT` / `DROP CONSTRAINT` / `SHOW CONSTRAINTS` both work, so schema
setup no longer has to happen in Python or Rust. What to watch:

- **Bare `CREATE INDEX` is equality-only.** One property builds a hash equality
  index; two or more build a composite index. `CREATE RANGE INDEX` builds *two*
  structures — the equality index **and** a B-tree range index — and reports
  `indexes_added` of 2. The bare form stays equality-only deliberately, since
  building both for every statement in a ported script would double index memory.
  Add `RANGE` when you need range scans; a multi-property `RANGE` index is
  rejected, because the B-tree is single-property.
- **Index names are not persisted; constraint names are.** An index name is
  accepted for portability and then discarded — index names here are canonical
  and derived (`Label.property`, `Label.(a,b)`). So `DROP INDEX` wants that dotted
  canonical name, or the descriptor form `DROP INDEX FOR (n:Label) ON (n.prop)`.
  **The trap:** dropping by a name you chose fails, and adding `IF EXISTS` to that
  same statement turns the failure into a silent no-op that leaves your index in
  place. `SHOW INDEXES` prints the canonical name, and its output pastes straight
  in. Constraint names, by contrast, persist across save/load and are unique per
  graph, so `CREATE CONSTRAINT person_email_unique …` followed by
  `DROP CONSTRAINT person_email_unique` works as written.
- **Uniqueness on the identity field is refused, not silently accepted.** A
  `REQUIRE … IS UNIQUE` (or `IS NODE KEY`) that resolves to the structural `id` —
  including the node type's own id column name — is rejected, because a unique
  secondary index would never observe those writes and the constraint would admit
  duplicates while reporting success. Declare identity uniqueness as
  `primary_key` in `define_schema` instead, which probes the per-type id index on
  every write path, or use `MERGE` as the idempotent alternative to `CREATE`.
  `IS NOT NULL` on the id field *is* accepted, since it is present by
  construction.

Forms KGLite cannot serve — `TEXT`, `POINT`, `FULLTEXT`, `VECTOR`, `LOOKUP`,
relationship indexes, `OPTIONS { … }`, and `IS :: TYPE` constraints — fail with a
specific unsupported-feature error naming the construct and the route that does
work. That is the deliberate choice: a type constraint accepted and silently
unenforced would be worse than an error, since it is exactly the kind of promise
data-integrity assumptions get built on. Full grammar in
{doc}`/reference/cypher-reference`.

**`LOAD CSV` works, and file access is a capability you grant.** `LOAD CSV [WITH
HEADERS] FROM <source> AS row [FIELDTERMINATOR <sep>]` runs for local files and
`file://` URLs, and must lead the query. Fields stay strings — CSV carries no
types, and inferring them would corrupt leading-zero identifiers — so conversion
is explicit (`toInteger(row.id)`). `http(s)://` is refused, naming the
network-free design: the engine ships no HTTP client.

The security model deserves stating plainly, because it is default-deny:

- **In-process callers** — the Python API, the Rust library, the CLI — get
  **unrestricted** read access to any path the process can read. That is
  deliberate, on the grounds that they already have the host process's
  filesystem access, but it is not a sandbox and should not be read as one.
- **A Bolt client gets nothing** unless an operator passes
  `--allow-csv-import <DIR>` to `kglite-bolt-server` — a single directory, not a
  repeatable flag. Imports are then confined to that directory *after* symlink
  and `..` resolution, and a relative path resolves against the import root
  rather than the server's working directory. Without the gate, anyone who could
  open a Bolt connection could read `file:///etc/passwd`.
- **The MCP server never grants the capability**, so an agent cannot use
  `LOAD CSV` to read the filesystem. Note this holds by construction — the MCP
  server simply never sets the field, inheriting the deny default — rather than
  by an explicit test.

Loading streams: the executor reads 1000 rows at a time, so peak memory does not
track file size for row-local pipelines. A downstream clause that must see the
whole result — an aggregate, `ORDER BY`, `DISTINCT`, `UNION`, `CALL` — cannot be
batched without changing the answer, so those queries take a single capped pass
and fail at 1,000,000 rows naming the clause that forced it, rather than
exhausting memory. `add_nodes` / `add_connections`, {doc}`blueprints`, and the
CLI's `.import` remain the higher-throughput routes.

**Migrations are a convention plus a CLI verb, not a framework.** There is a
user-schema version stamp — your own data-model revision, persisted with the
graph, distinct from the engine's `.kgl` format version and never interpreted by
the engine. Read or set it via `graph.schema_version` / `set_schema_version(n)`,
`graph_info()['user_schema_version']`, or `kglite schema-version <graph>
[--set N]`, and `describe()` reports it once set, so an agent opening a graph cold
sees which generation it holds.

`kglite migrate <graph> <dir>` applies ordered `<version>_<name>.cypher` files —
ascending by parsed integer, so `010` runs after `002`, and gaps are fine — and
advances the stamp. `--dry-run` prints the plan without applying or saving.
Re-running is a no-op. Everything executes against an in-memory copy and the
`.kgl` is written **once**, at the end, only if every statement succeeded: the run
is all-or-nothing, so a failure at migration 3 of 5 saves nothing at all, not even
1 and 2. Version `0` is reserved for the unversioned baseline. A stamp the
migration set cannot explain, a duplicate version, and a `.cypher` file with no
version prefix are all refused rather than guessed at.

Three things it deliberately does not do. **No downgrades** — reversing a
migration means writing the inverse as a new one, since inferring the inverse of
arbitrary Cypher would be a guess. **No detection of an edited migration** —
change one after it has been applied and nothing notices, so treat applied
migrations as immutable and append. **No per-migration ledger** — the stamp is a
high-water mark, so a migration inserted *behind* it is treated as already
applied. Always append with a higher number.

And a node's primary type is still immutable, so a type change means recreating
the node: create the replacement, copy the properties, re-wire the edges, delete
the original. Watch out for `SET n:NewType`, which *appears* to work — it adds a
**secondary label**, leaves `n.type` unchanged, and still matches
`MATCH (n:NewType)`, so a migration written that way looks successful while every
node keeps its original type. See {doc}`import-export` for the round-trip paths a
rebuild would use.

**The large-graph modes are still the weaker ones.** `mapped` now gets the same
per-commit crash safety as in-memory — the kill-9 suites are parametrised over
both — but `disk` does not: its durability boundary is the generation publish, so
it relies on `save()` checkpoints. That boundary is a real primitive rather than
an absence of one, and it is kill-9 tested in its own right
(`crates/kglite/tests/disk_crash_guarantee.rs`): a crash loses exactly the
mutations made since the last `save()`, and the last published generation always
reopens complete — never half-written, and never with a partially-applied commit.
What `disk` does not give you is a *smaller* unit of durability than a whole
`save()`. `disk` also keeps the whole-graph write checkpoint described above;
`mapped` does not — its statements take the same O(changes) journal in-memory
graphs take.
In-memory is the product; the disk modes are for exploring graphs too big for it,
and that is the trade-off you are accepting.

**Three bindings are maintained here; the rest you write.** Python and Rust are
first-class, and Java is official since 0.15.9 (Panama/FFM over the C ABI, on
Maven Central as `io.github.kkollsga:kglite`). Everything else — Go, JavaScript,
.NET — goes through the C ABI in `crates/kglite-c`: a supported boundary with a
generated header, but you are writing the binding. See {doc}`/rust/c-abi`.

## Deciding

Reach for KGLite as a primary store when a single process owns the writes, the
data fits the storage mode you picked, and uniqueness and presence cover the
invariants you need the store itself to hold. That describes a large class of
real applications: desktop and CLI tools, single-node services, agent state,
embedded analytics.

Look elsewhere when you need several processes writing concurrently without a
server in front, integrity rules beyond uniqueness and presence enforced by the
store, or a migration tool with a downgrade path. And if the data's real home is
another system, the {doc}`derived-index` pattern is both cheaper and better
tested.

## See also

- {doc}`durable-apps` — `open()` lifecycle, checkpoints, and `durable=True`.
- {doc}`derived-index` — the pattern to prefer when the graph is a projection.
- {doc}`/concepts/concurrency` — the three concurrency models, stated precisely.
- {doc}`/python/transactions` — `begin()` / `commit()` / `rollback()`, snapshot
  isolation, and OCC conflicts.
- {doc}`/python/error-handling` — the typed exception hierarchy and error codes.
