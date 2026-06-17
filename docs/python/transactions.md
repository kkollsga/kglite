# Transactions and Bolt

This document explains kglite's transaction surface and how a Bolt
server (or any other binding) consumes it. Phase A.3 / 0.9.53 hardened
the existing implementation with tests and this guide; the API itself
has been in place since well before 0.9.53.

## The transaction surface

```python
import kglite

graph = kglite.KnowledgeGraph()

# Read-write transaction — clones the graph for snapshot isolation.
tx = graph.begin()
tx.cypher("CREATE (:Person {name: 'Alice'})")
tx.cypher("CREATE (:Person {name: 'Bob'})")
tx.commit()          # apply atomically
# OR
tx.rollback()        # discard everything

# Read-only transaction — Arc snapshot, O(1) cost, zero memory overhead.
tx = graph.begin_read()
rows = tx.cypher("MATCH (p:Person) RETURN p.name")
# tx.commit() is a no-op for read-only.

# Context manager — auto-commit on success, auto-rollback on exception.
with graph.begin() as tx:
    tx.cypher("CREATE (:Person {name: 'Carol'})")
    tx.cypher("CREATE (:Person {name: 'Dan'})")
    # commit happens here automatically.
```

## Isolation semantics

- **Snapshot isolation.** `begin()` deep-clones the graph; the
  transaction sees a frozen view of the state at `begin()` time.
  Outside reads continue to see the pre-`begin()` state until
  `commit()`.
- **Write isolation.** Mutations inside the transaction touch only
  the working copy; the original graph is untouched until `commit()`.
- **Atomic commit.** `commit()` swaps the working copy into the
  owning `KnowledgeGraph` via an `Arc` pointer swap; other Python
  references see the new state on their next operation.
- **Read-only transactions are free.** `begin_read()` takes an `Arc`
  reference (no clone). Use them liberally for read-heavy sessions.

## Optimistic concurrency control (OCC)

`DirGraph` carries a monotonically incremented `version: u64`. Every
mutation bumps it. `begin()` captures the version at start; `commit()`
checks the version hasn't changed:

- **First-to-commit wins.** If two transactions race, the first
  `commit()` succeeds; the second raises a typed
  `kglite.KgError` with message starting `"Transaction conflict:
  graph was modified since begin(). Retry the transaction."`.
- **Outside mutations also trigger conflicts.** A direct
  `graph.cypher("CREATE ...")` between a transaction's `begin()` and
  `commit()` also bumps the version, so the transaction's commit
  fails.
- **Last-writer-wins is NOT supported.** Bindings must catch the
  conflict and retry (Bolt: send a `FAILURE` with code
  `Neo.TransientError.Transaction.ConflictDetected` and let the
  client retry).

## Auto-commit vs explicit transactions

Calling `graph.cypher(query)` **without** an enclosing
`begin()`/`commit()` is **auto-commit per call**:

```python
graph.cypher("CREATE (:Person {name: 'Alice'})")  # committed immediately
graph.cypher("CREATE (:Person {name: 'Bob'})")    # committed immediately
```

This has a contract caveat: **multi-statement queries that fail
partway through leave earlier statements visible.** Example:

```python
graph.cypher(
    "CREATE (:Person {name: 'A'}) "
    "CREATE (:Person {name: 'B'}) "
    "MATCH (x:NoSuchType) RETURN x"   # later clause fails
)
# The 2 CREATEs are already in the graph; only the MATCH errored.
```

Bolt servers MUST wrap each session's statements in `begin()` /
`commit()` to prevent clients from observing this contract. The
pattern is:

```rust
// Pseudocode for the Bolt server's RUN handler:
match message {
    Bolt::Begin => connection.tx = Some(graph.begin()?),
    Bolt::Run(query, params) => match &mut connection.tx {
        Some(tx) => tx.cypher(query, params),
        None => graph.cypher(query, params),  // auto-commit
    },
    Bolt::Commit => connection.tx.take().unwrap().commit(),
    Bolt::Rollback => connection.tx.take().unwrap().rollback(),
}
```

## Error mapping for Bolt FAILURE codes

All transaction errors are now typed `kglite.KgError` subclasses (the
Phase A.3 / 0.9.53 sweep brought `transaction.rs` in line with the
A.2 typed-exception migration). The Bolt server's `FAILURE` mapping:

| KgError class | Trigger | Suggested Bolt FAILURE code |
|---|---|---|
| `kglite.CypherTimeoutError` | `begin(timeout_ms=N)` expired | `Neo.ClientError.Transaction.TransactionTimedOut` |
| `kglite.KgError` ("Transaction conflict: …") | OCC conflict on `commit()` | `Neo.TransientError.Transaction.ConflictDetected` |
| `kglite.KgError` ("Transaction already committed or rolled back") | Double-commit / use-after-rollback | `Neo.ClientError.Transaction.TransactionAccessedConcurrently` |
| `kglite.KgError` ("Read-only transaction does not support mutations …") | Mutation inside `begin_read()` | `Neo.ClientError.Statement.AccessMode` |
| `kglite.CypherSyntaxError` | Bad Cypher inside `tx.cypher(...)` | `Neo.ClientError.Statement.SyntaxError` |
| `kglite.CypherExecutionError` | Anything else mid-execution | `Neo.DatabaseError.Statement.ExecutionFailed` |

The full KgError taxonomy is in `error-handling.md`.

## Storage-backend implications

All three backends (`Memory`, `Mapped`, `Disk`) support `begin()`:

- **Memory / Mapped** — clone is fast (heap copy of in-memory
  structures). Suitable for high transaction rates.
- **Disk** — clone copies the in-memory overlay (mmap-backed columns
  are shared, not cloned). Mutation rates are typically low on
  disk-mode graphs, but be aware that very-mutation-heavy workloads
  on multi-GB disk graphs will pay a perceptible clone cost.

## Concurrency

For thread-safety guarantees and the WKT-cache / `Arc::make_mut` CoW
contract, see [`concurrency.md`](../concepts/concurrency.md).

A `KnowledgeGraph` is single-owner: don't share one instance across threads while
a thread mutates it (that raises a clear `RuntimeError`). For concurrent *reads*,
the cleanest pattern is **not** a per-session transaction but a `graph.freeze()`
snapshot — an immutable, lock-free read view shared across all reader threads;
build/reload and `freeze()` again when the data changes (see
[`concurrency.md`](../concepts/concurrency.md)).

For a Bolt server running multiple sessions in parallel:

- **One `KnowledgeGraph` Arc shared across all tasks.**
- **One `Transaction` per active session.** Sessions are independent
  — each holds its own working copy or Arc snapshot.
- **Commits serialize through the GIL** (PyO3 + Python boundary).
  This is fine at Bolt scale (tens of connections, sparse writes).
  If profiling reveals GIL contention as a bottleneck, a future
  release can add a Rust-native `TransactionHandle` that bypasses the
  GIL — flag for Phase B/C if it becomes blocking.

## What's NOT supported (and why)

- **Nested transactions / savepoints.** Neo4j Bolt v5 doesn't expose
  these either. Out of scope.
- **Multi-graph atomic commits.** kglite is single-graph by design.
- **Last-writer-wins.** OCC is deliberate; force the binding to
  decide on retry behavior rather than silently overwrite.
- **Pre-commit conflict detection.** Conflicts surface at
  `commit()` time, not at operation time. Operations on a stale
  working copy succeed locally; only the commit fails. This matches
  Neo4j semantics.

## Performance reference

Phase A.3 / 0.9.53 reworked `begin()` to defer the DirGraph clone until
the first mutation lands (Issue #1 of the pre-Bolt audit). Numbers
below are from `scripts/perf_audit.py` on Apple M4 macOS — Linux
servers will differ but the shapes hold.

| Operation | 1k nodes | 10k | 100k |
|---|---:|---:|---:|
| `begin() + commit()` *(no writes)* | **166 ns** | 166 ns | 166 ns |
| `begin() + commit()` *(no writes), pre-0.9.53* | 40 µs | 391 µs | 4.16 ms |
| `begin_read() + commit()` | 125 ns | 125 ns | 125 ns |
| `begin() + 1 mutating cypher() + commit()` | ~30 µs | ~300 µs | ~3 ms |

**The Bolt-relevant takeaway:** `begin()` is now effectively free —
the deep clone only happens on the first mutation. Read-only-then-
commit transactions pay no clone cost regardless of graph size. The
Bolt server can wrap every session in `begin() / RUN.../ commit()`
without performance concern as long as most sessions don't mutate.

A mutating transaction still pays the clone cost on the first
`tx.cypher("CREATE ...")`. Magnitude is proportional to graph size
(~30 µs/k nodes). For sustained-write Bolt workloads on large graphs,
batch many mutations per transaction to amortize the clone.

### Cypher per-call overhead (0.9.53 post-audit)

| Call shape | min |
|---|---:|
| `cypher("RETURN 1 AS n")` | 375 ns |
| `cypher("MATCH (p {id: X}) RETURN p.title")` *(cache hit)* | 708 ns |
| `cypher("MATCH (p {id: X}) RETURN p.title")` *(unique each)* | 1.0 µs |
| `tx.cypher(...)` inside `begin_read()/commit()` | 792 ns |
| `tx.cypher(...)` inside `begin()/commit()` *(no writes)* | 917 ns |

Phase A.3 added an LRU parse cache (256 entries, FIFO eviction) that
roughly halves the per-call cost for repeated/parameterized queries —
the Bolt agent's typical hot-loop pattern.

## See also

- [`error-handling.md`](error-handling.md) — the `kglite.KgError`
  taxonomy.
- [`concurrency.md`](../concepts/concurrency.md) — multi-thread / multi-session
  contracts.
- `tests/test_transaction_bolt_patterns.py` — the executable
  contract this document explains.
- `scripts/perf_audit.py` — re-runnable audit harness for the
  numbers above.
