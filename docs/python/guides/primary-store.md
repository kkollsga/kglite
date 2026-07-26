# KGLite as a primary store: scope and limits

Most KGLite graphs are a {doc}`derived index <derived-index>` over data owned
somewhere else. This page is about the other case: the graph *is* the
authoritative copy, and losing it means losing the data.

That is a much stronger promise, so this page is deliberately conservative. It
states what holds, what is opt-in, and what KGLite does not do — with enough
detail that you can decide without running an experiment first. Where a limit
exists, it is named rather than softened.

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

Three cases keep the older whole-graph checkpoint, and there the write cost is
still proportional to graph size:

- the `mapped` and `disk` backends,
- columnar mode, whose `SET` writes a shared column store the journal cannot
  observe,
- graphs carrying user-created property, range, or composite indexes, whose
  delete path needs per-bucket position undo the journal does not record yet.

**Crash safety is available, and is opt-in.** `kglite.open(path, durable=True)`
puts the graph in write-ahead-log mode: each committed mutation appends one
frame to a `.kgl-wal` sidecar and `fsync`s it. A frame carries a CRC32; a crash
mid-append leaves a torn trailing frame, which replay discards rather than
half-applying. On open, the engine loads the `.kgl` checkpoint and replays every
frame newer than it. `save()` is separately atomic and `fsync`ed, so a reader
never observes a torn file.

**Readers see a consistent graph.** `freeze()` hands out an immutable, lock-free
snapshot; a `session()` serializes writers, begins each from the last committed
state, and publishes with a pointer swap only on success. Explicit transactions
are optimistically concurrent: independent work proceeds, and a commit against a
state that moved underneath returns a conflict rather than winning silently.
{doc}`/concepts/concurrency` is the full model, and worth reading before you rely
on any of it.

**Failures are typed.** Errors arrive as a `KgError` hierarchy with stable
codes, not as strings to match on — see {doc}`/python/error-handling`.

## What is opt-in, and off by default

| | Default | To enable |
|---|---|---|
| Crash safety | `open()` is `durable=False` — work reaches disk on clean close, not on a hard crash mid-session | `kglite.open(path, durable=True)` |
| Schema | No schema; any property on any node | `define_schema(...)` |
| Uniqueness on `id` | Permissive | declare `primary_key` on the type |
| Freshness stamps | Off, so writes stay deterministic | `auto_timestamp: True` per type |

The durability default is the one to think hardest about. If the graph is your
authoritative copy, `durable=False` is not the setting you want, and the
`open()` docstring says so directly. {doc}`durable-apps` covers the lifecycle
and the per-commit `fsync` cost you are trading for.

## What KGLite does not do

**One process writes.** There is no shared live multi-process transaction
handle and no replication protocol. Disk mode publishes immutable generations
behind a cross-process writer lease, which stops two processes from publishing
at once — that is stable-reader/single-writer publication, not concurrent
multi-process write access. When several processes need to read and write one
graph, the answer is to run `kglite-bolt-server` and let that one process own
the graph while clients connect over the Bolt protocol. Note the coverage limit
there too: the official Python driver is regression-tested, and other Bolt v5
clients may work but are not.

**Integrity constraints stop at the identity field.** `primary_key` is enforced
on the write path, but it must be `id`; an enforced unique constraint on an
arbitrary property is not supported. `required_fields` is checked at validation
time rather than on every write. There is no `CHECK` constraint and no standing
referential-integrity constraint between node types: a relationship to an
unknown endpoint auto-vivifies a provisional stub by default. Individual loads
can be strict — `from_records(..., on_missing_endpoint="error")` validates the
whole input and fails atomically, and `purge_provisional()` sweeps stubs that
were never promoted — but that is a per-load choice, not an invariant the store
holds for you. If your correctness argument depends on the store refusing bad
data, check it against this list first.

**Schema management is not a Cypher surface.** Indexes and constraints are
created through the Python and Rust APIs. Cypher `CREATE INDEX` is not
supported, so a Neo4j schema script does not run unedited. `LOAD CSV` is
likewise unsupported — bulk loading goes through `add_nodes` /
`add_connections`, {doc}`blueprints`, or the CLI's `.import`.

**There is no migration framework.** No schema-version stamp, no migration
runner, no equivalent of Flyway or Alembic. Worse for a store that has to
evolve: a node's primary type is immutable, so changing it means recreating the
node — which is exactly the operation a migration performs. Plan your own
ordered-scripts convention, and see {doc}`import-export` for the round-trip
paths a rebuild would use.

**The large-graph modes are the weaker ones.** `mapped` and `disk` exist for
graphs that do not fit in RAM, and they reject `durable=True`. They also keep
the whole-graph write checkpoint described above. In-memory is the product;
the disk modes are for exploring graphs too big for it, and that is the
trade-off you are accepting.

**There is no JVM or .NET binding.** Python and Rust are first-class. Everything
else goes through the C ABI in `crates/kglite-c` — a supported boundary with a
generated header, but you are writing the binding. See {doc}`/rust/c-abi`.

## Deciding

Reach for KGLite as a primary store when a single process owns the writes, the
data fits the storage mode you picked, you can turn `durable=True` on, and you
are content to enforce most invariants in your application rather than in the
store. That describes a large class of real applications: desktop and CLI tools,
single-node services, agent state, embedded analytics.

Look elsewhere when you need several processes writing concurrently without a
server in front, declarative constraints as your integrity guarantee, or a
migration framework you did not write. And if the data's real home is another
system, the {doc}`derived-index` pattern is both cheaper and better tested.

## See also

- {doc}`durable-apps` — `open()` lifecycle, checkpoints, and `durable=True`.
- {doc}`derived-index` — the pattern to prefer when the graph is a projection.
- {doc}`/concepts/concurrency` — the three concurrency models, stated precisely.
- {doc}`/python/transactions` — `begin()` / `commit()` / `rollback()`, snapshot
  isolation, and OCC conflicts.
- {doc}`/python/error-handling` — the typed exception hierarchy and error codes.
