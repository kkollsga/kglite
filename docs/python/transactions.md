# Transactions and sessions

KGLite has three mutation styles. Choose by the failure and concurrency
contract you need:

| Surface | Use when | Failure behavior |
|---|---|---|
| `graph.cypher(...)` | one-owner, simple direct work | executes in place; a late error/timeout may leave earlier mutations visible |
| `graph.begin()` | several operations must commit or roll back together | isolated copy-on-write transaction with OCC at commit |
| `graph.session()` | threads/tasks share one live graph | reads use snapshots; writes serialize and atomically swap on success |

## Explicit transactions

```python
import kglite

graph = kglite.KnowledgeGraph()

with graph.begin(timeout_ms=30_000) as tx:
    tx.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    tx.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
    # auto-commit on clean exit; auto-rollback on exception

read_tx = graph.begin_read()  # O(1) immutable snapshot
rows = read_tx.cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name")
read_tx.commit()              # no-op; releases the snapshot
```

`begin()` is O(1). It captures an `Arc` snapshot and creates a
backend-specific working fork only on the first mutation. Memory/mapped modes
clone then; disk mode shares immutable bases and copies mutation overlays.
Outside readers continue to see the pre-commit snapshot.

`commit()` checks the graph version. If another writer committed after
`begin()`, the commit raises a typed `kglite.KgError`; the application decides
whether to retry. `rollback()` discards the working fork. A transaction cannot
be reused after commit or rollback.

Transaction deadlines and query deadlines raise
`kglite.CypherTimeoutError`. Read-only transactions reject mutations. Nested
transactions/savepoints, last-writer-wins, and multi-graph atomic commits are
not supported.

## Shared sessions

Use `Session` when concurrent callers need one evolving graph:

```python
store = graph.session()  # or kglite.open_session("graph.kgl")
store.execute("CREATE (:Person {id: 3, name: 'Carol'})")
rows = store.cypher("MATCH (p:Person) RETURN p.name")
snapshot = store.snapshot()
```

`Session.execute()` serializes writers. Each writer begins from the previous
committed state and publishes with an atomic pointer swap, so failed execution
does not expose a partial working copy. Readers take stable snapshots; readers
already in flight keep seeing their prior snapshot while a write lands.

## Storage and protocol bindings

Transactions and sessions use the same core implementation for memory, mapped,
and disk storage. Rust protocol servers consume the native session/transaction
surface directly; they do not depend on Python or the GIL. Bolt maps KGLite's
typed error codes to Neo4j status codes and manages one transaction per Bolt
session. The C ABI currently exposes atomic mutation batches rather than an
explicit begin/commit handle.

## See also

- [Concurrency](../concepts/concurrency.md) — ownership, snapshots, and shared sessions.
- [Error handling](error-handling.md) — typed exceptions and stable codes.
- [Durable apps](guides/durable-apps.md) — WAL-backed in-memory persistence.
- [Rust session abstraction](../rust/session.md) — binding-level execution contract.

## Captured query defaults and ownership

`freeze()`, `session()`, `begin()` and `begin_read()` capture the source graph's
query timeout, work budget and row cap. Session snapshots/cursors inherit that
captured policy. Changing the original graph's defaults affects newly created
handles, not existing snapshots or transactions.

| Option | Omitted or `None` | Explicit zero |
|---|---|---|
| Query `timeout_ms` | captured default, then180000ms | no per-query deadline |
| Query `max_work_units` | captured budget, then engine backstop | literal zero budget |
| Query `row_limit` | captured row cap, then no cap | retain no result rows |
| `begin(timeout_ms=...)` | no transaction lifetime deadline | no transaction lifetime deadline |

An explicit per-call value overrides the captured value. A positive transaction
lifetime still bounds every query; per-query zero cannot bypass it. Clear a graph
default with its setter before deriving a handle when no inherited row/work limit
is wanted; passing `None` to a query means inherit.

Closing a persisted graph ends its write-back authority. Its retained data stays
mutable as a detached graph, but an earlier transaction with writes cannot commit
into that ended owner. Read-only snapshots and rollback remain available. This is
separate from optimistic data-version conflicts. A with-block around `open()` is
not a multi-statement transaction; see {doc}`guides/durable-apps`.

Fluent views of a CDC-owning graph and Session cursors sharing change-data
capture (CDC) are read-only for captured mutations. Their descendants inherit
that restriction. Use `cursor.copy()`
for an independent graph and change stream, or write through the original owner.
