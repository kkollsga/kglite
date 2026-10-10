# KGLite as a primary store: scope and limits

Most KGLite graphs are a {doc}`derived index <derived-index>` over data owned
somewhere else. This page is about the other case: the graph *is* the
authoritative copy, and losing it means losing the data.

This page states the guarantees, defaults, and limits that matter when KGLite
owns the authoritative copy.

## What holds

### Atomic statements

**A mutating statement is all-or-nothing.** A Cypher statement that fails after
its first write leaves the graph exactly as it found it. That covers:

- node and relationship identity
- properties and labels
- index ordering
- schema metadata
- the version counter

Rollback replays a statement-scoped journal of inverse operations backwards.

### Write cost scales with the change

**The cost of a write scales with the change, not with the graph.** The journal
records only what the statement touched. A single `SET` on a million-node graph
therefore costs about what it costs on a thousand-node graph. This property
separates a store you can write to continuously from an index you rebuild.

It is measured, not assumed. `tests/benchmarks/test_bench_write_scaling.py` runs
the same statements at 1 k, 100 k, and 1 M nodes. A reading that grows with size
is a regression.

#### Which backends use the journal

`memory` and `mapped` take the journal for every statement. A durable graph over
either of them does too, because durability and rollback strategy are
independent concerns.

For `disk`, the journal covers only the statements a register ingest runs. A
disk graph has no petgraph slot identity for an inverse edit to name, so the
journal that undoes a memory or `mapped` statement does not cover it as a whole.

- **Journalled statements.** A `MATCH … SET` of plain properties, a property
  `REMOVE`, `CREATE`, `MERGE` and `DELETE` journal the cells, titles and
  appended rows they write. They are undone from that journal.
- **Cost.** It follows the rows they touch. Closing 2,000 rows of a 2-million-row
  register costs 1.2 to 1.7 µs a row, whatever the table's size.
- **Checkpointed statements.** A statement outside that set opens a checkpoint
  instead. The set is `FOREACH`, `CALL`, `LOAD CSV`, a label change,
  `SET n += {…}`, and a nested-path `SET`.
- **What a checkpoint is.** A copy of the disk graph's mapped arrays and
  overlays, not of its rows. Its column stores are shared until written.
- **Cost of a checkpointed statement.** The first write to a property column
  copies that column, and every later row of the statement writes in place. The
  cost is one copy per column it writes plus a constant per row.

Two other graph shapes also use the journal:

- **A graph that has been saved, loaded, or opened from a file.** It carries the
  same property shape a freshly built graph does. A `SET` journals the
  individual cells it overwrote rather than a copy of the type's whole column
  store. The write-scaling benchmark covers freshly built, loaded, spilled, and
  mapped graphs.
- **A graph carrying user-created property, range, or composite indexes.** Their
  bucket edits are journalled with the position they occupied. `CREATE INDEX`
  and the `create_index` API no longer move a graph's writes back onto the
  whole-graph checkpoint.

Uniqueness constraints are the one structure still rebuilt wholesale, and only
on the *failure* path. A statement that rolls back recomputes the occupancy map
of each type it touched, which scales with that type's node count. Successful
writes never pay it. If your workload both writes continuously and fails
statements often, measure it rather than assuming flat cost.

### What a write costs in memory

Properties live in per-type columns from the first node onward. Building,
saving, loading and reopening all produce the same shape. There is no conversion
step to plan around and no second cost profile after the first `save()`.

Two consequences matter before you rely on this as a primary store:

- **A column is allocated per declared property, per row, whether or not the row
  has a value.** Memory is proportional to schema width rather than to the
  properties actually set. A type with many optional properties costs more at
  rest than the same data in a narrow type.
  - `graph_info()['columnar_heap_bytes']` reports how much the stores hold.
  - `set_memory_limit()` spills columns to disk when they exceed a budget. The
    limit is re-enforced after every statement, not only at load time.
- **`DELETE` tombstones rows; `vacuum()` reclaims them.** Deleted rows keep their
  space until you compact. A write-heavy primary store should call `vacuum()`
  periodically. It also fires automatically once fragmentation crosses
  `auto_vacuum_threshold`.

On `storage="disk"`, reclaiming works differently:

- **`vacuum()` is a no-op on `storage="disk"`.** Its node numbering is frozen
  mmap, so there is no in-place rebuild to do.
- **`save()` reclaims instead.** A disk save rewrites the columns without the
  rows no live node points at. The published directory and the graph that
  reloads from it carry live rows only.
- **A save does not reclaim node slots.** A deleted node's 16-byte slot and its
  free-list entry are kept. A disk graph's node capacity only shrinks when the
  directory is rebuilt from a fresh ingest.
- **`compact()` is a separate, edge-only operation.** It merges overflow edges
  into the CSR and touches no rows.

Batching mutations into multi-row statements is still worth doing. It amortises
per-statement parsing, planning and checkpoint overhead, but it is now a
throughput optimisation rather than a workaround. See {doc}`data-loading`'s
throughput ladder.

### Crash safety

**Crash safety is the default.** `kglite.open(path)` opens in write-ahead-log
mode wherever the storage mode supports it: the default in-memory backend and
`storage="mapped"`.

- **Commit.** Each committed mutation appends one frame to a `<path>-wal`
  sidecar and `fsync`s it before the call returns.
- **Open.** The engine loads the `.kgl` checkpoint and replays every frame newer
  than it.
- **Integrity.** A frame carries a CRC32. A crash mid-append leaves a torn
  trailing frame that replay discards rather than half-applying.
- **What recovery reports.** It says which of two cases it found:
  - A torn tail is reported as the ordinary aftermath of a crash.
  - A corrupt frame with bytes *after* it is named as mid-file damage. A
    durable open copies the whole log to a `.quarantine-` sibling, continues on
    the frames before the damage, and lists a `wal_quarantined` advisory in
    `graph_info()`. The copy is never deleted, and the open is refused if it
    cannot be written. The log cannot be trusted past corruption, so that is a
    storage problem to investigate rather than a routine restart.
- **Coverage.** Every way of changing a graph is logged, not only Cypher:
  `add_nodes`, `add_relationships`, label changes, and committed transactions
  included.
- **`save()`** is separately atomic and `fsync`ed, so a reader never observes a
  torn file.

#### Disk graphs

`storage="disk"` is the exception. A disk graph commits by publishing an
immutable generation, so a logical write-ahead log is not its durability
boundary. A disk graph opens non-durable and takes `save()` checkpoints instead.

- Asking for any logging level there raises `ValueError` explaining that. That
  covers `durable=True`/`"full"` *or* `durable="normal"`. The blocker is the
  commit boundary rather than barrier strength.
- Only `durable="off"` is supported.
- The *default* does not raise, so disk callers are unaffected by the default
  being on elsewhere.

#### What the log carries

The log carries:

- nodes, edges and labels
- every declaration you make about them: identity-field spellings,
  `set_parent_type`, ontologies, constraints, user-created indexes,
  `set_spatial` and `set_schema_version`
- the two bulk payloads: **timeseries channels** and **embeddings**. Embeddings
  carry the model id and per-node text hashes that `embed_texts(mode='changed')`
  reads.

A crash before your first `save()` therefore loses none of it.

Replay rebuilds one thing rather than reading it: the HNSW vector index. Only the
`build_vector_index` declaration is logged. The topology is rebuilt from the
replayed vectors, because the index addresses store slots that replay renumbers.

#### Consequences of the log

Four consequences matter before you rely on this:

- **It costs one barrier per committed mutation.** Writes wait on physical
  storage, so the cost is device latency rather than graph size. It is most
  visible in loops of many small writes and negligible for a few large ones.
  Reads are unaffected.
  - `durable="normal"` keeps the log and drops only that barrier. A committed
    mutation still survives the process dying. An OS crash or power cut loses
    work since the last `save()`. `sync()` gives you a power-safe point on
    demand.
  - `durable=False` (no log at all) remains fully supported. It is the right
    choice for bulk loading and for graphs you can rebuild from source.
  - Batching writes into one statement, or one `begin()` transaction, buys
    throughput *and* the strongest guarantee.
- **A `with` block is not a transaction.** Mutations commit as they run. An
  exception inside the block does not discard them; they are recovered on the
  next `open()`. A clean or failed exit controls only whether a *checkpoint* is
  written. Use `begin()` when you want discard-on-error.
- **No handle derived from a durable graph may write.** `Session.execute()`, and
  equally a selection or view (`g.select(...).update(...)`, `view.cypher("CREATE
  …")`, `view.save()`), land on a working copy that neither the log nor `save()`
  can reach. Rather than silently losing the write, they raise:
  - `kglite.ArgumentError` from `Session`.
  - `ValueError` from a derived handle.

  Use `cypher()` or `begin()` on the graph itself. Durable is now the default, so
  code that used `Session.execute()` for writes against an `open()`ed graph has to
  change, or pass `durable=False`.
- **A refused log append poisons the handle.** If the log cannot take a frame, the
  statement raises `kglite.FileIoError`. A full disk is the case this was
  measured on.
  - Every later write, `save()` and `sync()` on that handle is refused with the
    same class. The graph in memory now contains a statement the caller was told
    did not happen, and `save()` would commit it.
  - The message names the exit: reopen the path.
  - The failed statement is not in the recovered graph.

Two additional constraints:

- `save(fsync=False)` is ignored on a durable graph and warns, because the
  checkpoint truncates the log and so must itself reach disk.
- A log written by this version is refused by older builds with a clear message
  rather than silently truncated.

### Consistent readers and concurrent writers

**Readers see a consistent graph.**

- `freeze()` hands out an immutable, lock-free snapshot.
- A `session()` serializes writers, begins each from the last committed state,
  and publishes with a pointer swap only on success. Note the durable-graph
  restriction on `Session` writes above.
- Explicit transactions are optimistically concurrent. A commit against a state
  that moved underneath raises `TransactionConflictError` rather than winning
  silently.

The conflict check is coarser than in most databases. It compares a
**whole-graph version counter**, not the read/write sets of the two
transactions. A commit publishes the transaction's working copy by pointer swap,
so a transaction that began before *any* other commit is working from a stale
snapshot regardless of which nodes it touched. Two transactions editing entirely
unrelated nodes therefore conflict, and the second one loses because its working
copy does not contain the first write.

Conflicts are ordinary rather than rare, so every concurrent writer needs a retry
loop. Use `kglite.retry_on_conflict` rather than writing one:

```python
def signup(tx):
    tx.cypher("CREATE (u:User {email: $email})", params={"email": email})

kglite.retry_on_conflict(graph, signup)
```

If your workload has many short concurrent writers, prefer `session()`. It
serializes writers and begins each from the last committed state, so they queue
instead of colliding. {doc}`/concepts/concurrency` is the full model, and worth
reading before you rely on any of it.

### Typed failures

**Failures are typed.** Errors arrive as a `KgError` hierarchy with stable codes,
not as strings to match on. See {doc}`/python/error-handling`. That includes the
write path: a `save()`, `sync()` or `to_bytes()` that fails on I/O raises
`kglite.FileIoError` (`.code == "FileIo"`), not a bare `OSError`.

### Integrity constraints

**Integrity constraints are enforced on every write path.** You declare them
through `define_schema`. They are checked on Cypher `CREATE` / `INSERT` /
`MERGE` / `SET` / `REMOVE` and on the bulk loaders alike. The bulk loaders are
`add_nodes`, and therefore blueprints, `from_records`, OKF ingestion, WAL replay,
and `extend_graph`:

```python
graph.define_schema({"nodes": {"Person": {
    "primary_key": "email",            # unique *and* present (NODE KEY)
    "unique": [["first", "last"]],     # composite UNIQUE
    "required": ["email"],             # NOT NULL, at write time
}}})
```

Three things make this real rather than advisory:

- **`primary_key` may name any property, not just `id`.** A key on `id` routes
  through the identity index. Any other key is backed by a unique secondary index
  that persists and rebuilds on load.
- **`required` is enforced at write time.** A `CREATE` that omits the property, a
  `SET` that nulls it, and a `REMOVE` that drops it all raise, rather than
  surfacing later in `validate_schema()`.
- **A constraint the stored data already violates is refused outright.** You
  cannot install a constraint that quietly lies about the rows already present.

A composite `unique` tuple constrains only nodes carrying *every* property in it.
NULL is exempt throughout: a node sits outside a uniqueness constraint unless
every property in the tuple is present and non-null. Many nodes may therefore
share "no email" while `email` is `UNIQUE`.

#### Property types are not enforced by `define_schema`

**`types` is the one part of `define_schema` that is *not* enforced at write
time.** `required`, `unique` and `primary_key` reject the offending write. A
`types` declaration is advisory.

- `validate_schema()` checks it when you call it, and reports every row that
  disagrees (`error_type: "type_mismatch"`).
- Nothing rejects `CREATE (:Item {age: "not a number"})` on a type whose schema
  says `age` is `int`.

To enforce a property type on the way in, you have two options:

- **A constraint.** `CREATE CONSTRAINT FOR (n:Item) REQUIRE n.age IS :: INTEGER`
  raises `ConstraintViolationError` on the write, on every write path, for the
  property types KGLite can check.
- **`lock_schema()`.** It rejects a write whose value disagrees with the property
  type the node type has actually recorded.

#### Bulk loads are all-or-nothing

**A large bulk load *is* all-or-nothing for everything the loader can refuse.**
`add_nodes` and `add_relationships` decide every refusal in a single pass before
the first row is written.

- The constraint gate checks the whole input up front.
- `on_invalid="error"` scans the input for rows with an unusable id in the same
  way.
- Both raise with nothing written, at any input size.

Rows are still flushed to the graph in chunks of 1000, but that is a memory bound
rather than an atomicity boundary. The flush loop has no failure path of its own,
so no error these loaders raise can leave half a load behind.

Chunking still bounds a call that never returns. A process killed mid-load leaves
the chunks folded in so far in memory. On a durable graph that costs nothing,
because the load reaches the log as one frame at the end. A crash mid-load
recovers to the pre-load state.

#### Paths that bypass constraints

The N-Triples loader and embedding-carry path bypass constraint enforcement. A
graph filled through those can hold violations, which
`verify_unique_constraints()` can audit.

The general RDF loader is a fresh-graph bootstrap operation:

- Python and C return a new in-memory graph.
- Rust `load_rdf(&mut graph, ...)` refuses populated or configured targets before
  writing.
- Load RDF first, then declare and validate its constraints.

#### Catching constraint errors

Two exception classes cover constraints:

- A violation raises `ConstraintViolationError`.
- A declaration that cannot be installed raises `ConstraintCreationError`.

Both subclass `ConstraintError`, so `except ConstraintError` catches either. This
holds on every write path, `cypher()` and the bulk writers alike. The
duplicate-signup handler is therefore a type check, not a substring match:

```python
try:
    graph.cypher("CREATE (u:User {email: $email})", params={"email": email})
except kglite.ConstraintViolationError:
    raise Conflict("that email is already registered")
```

Each carries a stable `.code` (`"ConstraintViolation"` /
`"ConstraintCreationFailed"`) for logging and cross-binding dispatch.

`define_schema` *can* fail this way, because installing a schema installs the
constraints it declares. Nothing is changed when it does, so you can fix the data
and retry. The message still names the constraint, the property, and the
offending value, and is worth logging. The type and code are the contract.

## Defaults, and how to change them

| | Default | To change |
|---|---|---|
| Crash safety | **On** (`"full"` — survives power loss) for in-memory and `mapped`; `disk` opens non-durable | `durable="normal"` to keep the log without the per-commit barrier, `durable="off"` to opt out entirely |
| Schema | No schema, but a node type's property set is fixed by its first write (below) | `define_schema(...)` |
| UNIQUE / NOT NULL / node key | Permissive — a type declaring none keeps the old behaviour | `unique` / `required` / `primary_key` in `define_schema` |
| Freshness stamps | Off, so writes stay deterministic | `auto_timestamp: True` per type |

All of the constraint machinery is opt-in, and older graphs load unchanged.
{doc}`durable-apps` covers the `open()` lifecycle and the per-commit `fsync`
cost in more detail.

### The property set of a node type

**"No schema" does not mean "any property".** The first write to a node type
establishes that type's property set. A later `CREATE` naming a property outside
it is refused:

```python
graph.cypher("CREATE (:Item {sku: 'A1'})")
graph.cypher("CREATE (:Item {sku: 'A2', colour: 'red'})")
# kglite.SchemaError: Schema error: Unknown property 'colour' on Item.
#   Valid properties: sku
```

This is a **typo guard, not a schema**. `CREATE (:Item {sk: 'A3'})` is far more
often a misspelling than a new field. Silently storing it produces a graph where
half the rows answer a query and half do not. The guard fires whether or not
`lock_schema()` was called, so `schema_locked` being `False` is not a reason to
expect otherwise.

Four things widen the set. Any of them is the way to add a property deliberately:

- **`SET`.** `MATCH (i:Item) SET i.colour = 'red'` is never refused, and the
  property is part of the type afterwards. This is the shortest route when you
  are adding a field to existing data.
- **`define_schema`.** A property named in `required`, `optional`, `types`,
  `unique` or `primary_key` is accepted by `CREATE` immediately, before any node
  carries it. Declare the shape up front and the guard never gets in the way.
- **A bulk load.** `add_nodes` with a new column widens the type, so a loader that
  is the source of truth for the schema does not need a declaration.
- **A fresh node type.** The set is per type, so `:ItemV2` starts over.

The guard covers the node-creating patterns: `CREATE`, `INSERT`, and `MERGE`'s
match/create pattern. Relationship properties are not guarded at all, and neither
is a `SET` clause attached to a `MERGE`.

## What KGLite does not do

### One process writes

**One process writes, and `kglite.open()` enforces it between openers.**
`kglite.open(path)` takes an exclusive cross-process writer lease for as long as
the graph can write back to `path`. A second process opening the same path fails
immediately with the holder's pid rather than quietly overwriting its work at
`save()`:

```
KgError: app.kgl is open for writing by pid 4711 (since 2026-07-26T09:15:03+02:00)
```

Readers are unaffected. `load()` and `open_session()` take no lease, so any number
of processes can read a graph while one writes.

The lease files work as follows:

- The lease is an OS-owned lock, so a writer killed with `SIGKILL` releases it
  immediately.
- `<path>.lock` (the lock, always empty) and `<path>.lock-owner` (the pid and
  acquisition time, used to name a holder) are records, not the lock itself.
  Deleting them achieves nothing.
- A lease handed back cleanly appends a `released=<timestamp>` line to the
  `.lock-owner` record. A record without one was left by a holder that died still
  holding the lease.
- That line is forensics after the fact, not a liveness signal, since the lock is
  what decides whether a writer waits.
- `open(..., lock=False)` opts out for callers that coordinate writers some other
  way.

**`save()` checks the lease; it does not take it.**

- These take it: `kglite.open()`, the CLI's eager save paths,
  `kglite-bolt-server` and `kglite-mcp-server`.
- `KnowledgeGraph.save()` probes the target's lease. While another process holds
  it, the save raises `WriterLeaseHeldError` naming the holder and writes nothing.
  A lease held by this process is not foreign.
- `open(..., lock=False)` opts out of the probe as well.
- The Rust `kglite::api::io::save_graph` and the C ABI save entries do not probe.

A graph obtained from `kglite.load(path)`, mutated in memory and saved back to a
path a lease holder is mid-write on, therefore fails in Python. Through the Rust
and C entries it publishes straight over the holder's work: the file that results
is a complete, valid graph, but whatever the holder had not saved is not in it. A
serving MCP server then refuses its own `save_graph` because the file changed
under it.

The rule the lease encodes: **any caller that may save to a path holds the lease
across the whole read-modify-save interval.** That is exactly what `open(path)` is
for. `load()` + `save(path)` takes no lease, so it can still write a path
between the probe and the rename; the probe catches the holder who is already there.

Taking the lease is also when `open()` cleans up after a writer that died
mid-`save()`. A save writes a sibling `<name>.tmp.<pid>.<n>` and renames it into
place, so a process killed part-way through leaves a full-size copy of the graph
behind. A crash-looping writer used to fill the volume with them. `open()` now
deletes the ones whose owning process is gone, and only those. A temp belonging to
a running process is never touched, so a concurrent save is safe.

#### Multi-process access

There is no shared live multi-process transaction handle and no replication
protocol. Disk mode publishes immutable generations behind the same kind of
lease. That is stable-reader/single-writer publication, not concurrent
multi-process write access.

When several processes need to read and write one graph, `kglite-bolt-server` is
the coordination point. That one process owns the graph while clients connect over
the Bolt protocol. It does not lift the single-writer model; it centralises it.

- Auto-commit writes (`session.run`) commit as a transaction of their own; `execute_write` groups several statements.
- Writes serialize at commit.
- A commit against a stale snapshot conflicts with a retriable status code, so
  driver-managed transactions retry on their own. That retry is contention-tested,
  not merely lifecycle-tested
  (`tests/test_bolt_server_transactions.py::test_managed_transaction_retries_after_conflict`).

The official Python, JavaScript, and Java drivers are regression-tested in CI:
session and explicit-transaction lifecycle, managed retry, PackStream type
round-trips, `Neo.*` error codes, and OCC conflict detection. This is focused
contract coverage rather than a full protocol sweep. Other drivers, including Go
and .NET, are untested.

### Constraints cover uniqueness, presence and property type

**Constraints cover uniqueness, presence and property type, not arbitrary
rules.** Nodes carry all three. Relationships carry presence and property type
(`FOR ()-[r:T]-() REQUIRE r.p IS NOT NULL` / `IS :: TYPE`) but not uniqueness.

KGLite does not offer:

- a `CHECK` constraint
- a standing referential-integrity constraint between node types

A relationship to an unknown endpoint auto-vivifies a provisional stub rather than
being rejected. Stubs are *deferred*, not exempt:

- The `add_nodes` upsert that promotes one is a normal, fully-enforced write.
- An unpromoted stub stays reportable via `validate_schema()`.
- `purge_provisional()` sweeps them.
- Individual loads can be strict up front with
  `from_records(..., on_missing_endpoint="error")`, which validates the whole
  input and fails atomically.

If your correctness argument needs a rule that is not uniqueness or presence, it
still belongs in your application.

### Schema setup in Cypher

**Schema setup is expressible in Cypher, with two asymmetries to know.**
`CREATE [RANGE] INDEX` / `DROP INDEX` / `SHOW INDEXES` and `CREATE CONSTRAINT` /
`DROP CONSTRAINT` / `SHOW CONSTRAINTS` both work, so schema setup no longer has to
happen in Python or Rust. What to watch:

- **Bare `CREATE INDEX` is equality-only.** One property builds a hash equality
  index. Two or more build a composite index.
  - `CREATE RANGE INDEX` builds *two* structures, the equality index **and** a
    B-tree range index, and reports `indexes_added` of 2.
  - The bare form stays equality-only deliberately, since building both for every
    statement in a ported script would double index memory.
  - Add `RANGE` when you need range scans.
  - A multi-property `RANGE` index is rejected, because the B-tree is
    single-property.
- **Index names are not persisted; constraint names are.** An index name is
  accepted for portability and then discarded. Index names here are canonical and
  derived: `Label.property`, `Label.(a,b)`, and `relationship:TYPE.property` for a
  relationship vector index.
  - `DROP INDEX` wants that derived canonical name, or, for a node index, the
    descriptor form `DROP INDEX FOR (n:Label) ON (n.prop)`.
  - **The trap:** dropping by a name you chose fails. Adding `IF EXISTS` to that
    same statement turns the failure into a silent no-op that leaves your index in
    place.
  - `SHOW INDEXES` prints the canonical name, and its output pastes straight in.
  - Constraint names persist across save/load and are unique per graph. So
    `CREATE CONSTRAINT person_email_unique …` followed by
    `DROP CONSTRAINT person_email_unique` works as written.
- **Uniqueness on the identity field is refused, not silently accepted.** A
  `REQUIRE … IS UNIQUE` (or `IS NODE KEY`) that resolves to the structural `id` is
  rejected. That includes the node type's own id column name. A unique secondary
  index would never observe those writes, so the constraint would admit
  duplicates while reporting success.
  - Declare identity uniqueness as `primary_key` in `define_schema` instead. It
    probes the per-type id index on every write path.
  - Or use `MERGE` as the idempotent alternative to `CREATE`.
  - `IS NOT NULL` on the id field *is* accepted, since it is present by
    construction.

Some forms KGLite cannot serve:

- `TEXT`, `POINT`, `FULLTEXT`, `VECTOR`, `LOOKUP`
- relationship indexes
- `OPTIONS { … }`
- a property type outside the accepted names
- `IS UNIQUE` / `IS RELATIONSHIP KEY` on a relationship

These fail with a specific unsupported-feature error naming the construct and the
route that does work. Unsupported forms are rejected rather than recorded without
enforcement. Full grammar in {doc}`/reference/cypher-reference`.

### LOAD CSV

**`LOAD CSV` works, and file access is a capability you grant.** `LOAD CSV [WITH
HEADERS] FROM <source> AS row [FIELDTERMINATOR <sep>]` runs for local files and
`file://` URLs, and must lead the query.

- Fields stay strings. CSV carries no types, and inferring them would corrupt
  leading-zero identifiers. Conversion is explicit (`toInteger(row.id)`).
- `http(s)://` is refused, naming the network-free design: the engine ships no
  HTTP client.

The security model is default-deny:

- **In-process callers** (the Python API, the Rust library, the CLI) get
  **unrestricted** read access to any path the process can read. That is
  deliberate, on the grounds that they already have the host process's filesystem
  access. It is not a sandbox and should not be read as one.
- **A Bolt client gets nothing** unless an operator passes
  `--allow-csv-import <DIR>` to `kglite-bolt-server`.
  - It is a single directory, not a repeatable flag.
  - Imports are confined to that directory *after* symlink and `..` resolution.
  - A relative path resolves against the import root rather than the server's
    working directory.
  - Without the gate, anyone who could open a Bolt connection could read
    `file:///etc/passwd`.
- **The MCP server never grants the capability**, so an agent cannot use
  `LOAD CSV` to read the filesystem. This holds by construction rather than by an
  explicit test: the MCP server simply never sets the field, inheriting the deny
  default.

Loading streams. The executor reads 1000 rows at a time, so peak memory does not
track file size for row-local pipelines such as `MATCH`, `FILTER`,
`CREATE`/`INSERT`, the delete forms, ordinary procedures, and terminal `FINISH`.

A downstream clause that must see the whole result cannot be batched without
changing the answer:

- an aggregate
- `ORDER BY`
- `SKIP`/`OFFSET`/`LIMIT`
- `DISTINCT`
- a set operation
- `cluster()`
- a `CALL` subquery

Those queries take a single capped pass. They fail at 1,000,000 rows naming the
clause that forced it, rather than exhausting memory. `add_nodes` /
`add_relationships`, {doc}`blueprints`, and the CLI's `.import` remain the
higher-throughput routes.

### Migrations

**Migrations are a convention plus a CLI verb, not a framework.** There is a
user-schema version stamp. It is your own data-model revision, persisted with the
graph, distinct from the engine's `.kgl` format version and never interpreted by
the engine.

- Read or set it via `graph.schema_version` / `set_schema_version(n)`,
  `graph_info()['user_schema_version']`, or `kglite schema-version <graph>
  [--set N]`.
- `describe()` reports it once set, so an agent opening a graph cold sees which
  generation it holds.

`kglite migrate <graph> <dir>` applies ordered `<version>_<name>.cypher` files and
advances the stamp.

- Files run ascending by parsed integer, so `010` runs after `002`, and gaps are
  fine.
- `--dry-run` prints the plan without applying or saving.
- Re-running is a no-op.
- Everything executes against an in-memory copy. The `.kgl` is written **once**,
  at the end, only if every statement succeeded. The run is all-or-nothing, so a
  failure at migration 3 of 5 saves nothing at all, not even 1 and 2.
- Version `0` is reserved for the unversioned baseline.
- A stamp the migration set cannot explain, a duplicate version, and a `.cypher`
  file with no version prefix are all refused rather than guessed at.

Three things it deliberately does not do:

- **No downgrades.** Reversing a migration means writing the inverse as a new one,
  since inferring the inverse of arbitrary Cypher would be a guess.
- **No detection of an edited migration.** Change one after it has been applied and
  nothing notices, so treat applied migrations as immutable and append.
- **No per-migration ledger.** The stamp is a high-water mark, so a migration
  inserted *behind* it is treated as already applied. Always append with a higher
  number.

A node's primary type is still immutable, so a type change means recreating the
node: create the replacement, copy the properties, re-wire the edges, delete the
original. Watch out for `SET n:NewType`. It *appears* to work, but it adds a
**secondary label**, leaves `n.type` unchanged, and still matches
`MATCH (n:NewType)`. A migration written that way looks successful while every
node keeps its original type. See {doc}`import-export` for the round-trip paths a
rebuild would use.

### Large-graph modes

**The large-graph modes are still the weaker ones.** In-memory is the primary
mode; the disk modes are for exploring graphs too big for it.

- **`mapped`** gets the same per-commit crash safety as in-memory. The kill-9
  suites are parametrised over both. Its statements take the same O(changes)
  journal in-memory graphs take.
- **`disk`** does not get per-commit crash safety. Its durability boundary is the
  generation publish, so it relies on `save()` checkpoints. It also keeps the
  whole-graph write checkpoint described above.

The `disk` boundary is a real primitive rather than an absence of one, and it is
kill-9 tested in its own right (`crates/kglite/tests/disk_crash_guarantee.rs`).
A crash loses exactly the mutations made since the last `save()`. The last
published generation always reopens complete: never half-written, and never with a
partially-applied commit. What `disk` does not give you is a *smaller* unit of
durability than a whole `save()`.

### Bindings

**Three bindings are maintained here.** Python and Rust are first-class. Java is
official since 0.15.9 (Panama/FFM over the C ABI, on Maven Central as
`io.github.kkollsga:kglite`). Everything else (Go, JavaScript, .NET) can use the C
ABI in `crates/kglite-c`; KGLite does not ship those bindings. See
{doc}`/rust/c-abi`.

## Deciding

Reach for KGLite as a primary store when all of these hold:

- A single process owns the writes.
- The data fits the storage mode you picked.
- Uniqueness and presence cover the invariants you need the store itself to hold.

That describes a large class of real applications: desktop and CLI tools,
single-node services, agent state, embedded analytics.

Look elsewhere when you need any of these:

- several processes writing concurrently without a server in front
- integrity rules beyond uniqueness and presence enforced by the store
- a migration tool with a downgrade path

If the data's real home is another system, the {doc}`derived-index` pattern is both
cheaper and better tested.

## See also

- {doc}`durable-apps` — `open()` lifecycle, checkpoints, and `durable=True`.
- {doc}`derived-index` — the pattern to prefer when the graph is a projection.
- {doc}`/concepts/concurrency` — the three concurrency models, stated precisely.
- {doc}`/python/transactions` — `begin()` / `commit()` / `rollback()`, snapshot
  isolation, and OCC conflicts.
- {doc}`/python/error-handling` — the typed exception hierarchy and error codes.
