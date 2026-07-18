# Concurrency

KGLite has three distinct concurrency models. Choose one deliberately; claims
about one model do not apply to the others.

## Bare `KnowledgeGraph`

A graph handle has single-owner mutation semantics. Concurrent reads may run,
but a read that overlaps mutation on the same handle is rejected rather than
silently racing. Direct writes mutate in place.

Use a separate `copy()` for independent evolution or `freeze()` for a stable
shared read snapshot:

```python
snapshot = graph.freeze()
# share snapshot across reader threads; it never changes
```

Snapshots are cheap Arc clones and remain stable after the owner publishes a
new graph. They are the simplest pattern for rebuild-and-swap services.

## Shared `Session`

Use `graph.session()` / `kglite.open_session(...)` when threads or tasks need
one evolving graph:

```python
store = graph.session()
store.execute("CREATE (:Task {id: 1})")  # serialized, atomic publication
rows = store.cypher("MATCH (n) RETURN count(n)")
```

Readers take stable snapshots. Writers serialize, begin from the previous
committed state, execute against a working graph, and publish with an Arc swap
only on success. Snapshot acquisition or unique-owner mutation may briefly
wait; once a reader has its snapshot, a later commit does not change it.

`Session::begin` also supports explicit optimistic transactions. Concurrent
transactions may work independently, but a stale commit returns a conflict;
production bindings should not use last-writer-wins.

## Disk generations and processes

Disk mode publishes immutable generations. Readers resolve `CURRENT` once and
keep that generation mmaped. One retained cross-process writer lease prevents
two processes from publishing concurrently; readers do not take the writer
lease and can keep using an older generation after a new one lands.

This is stable-reader/single-writer publication, not a shared live
multi-process transaction handle or replication protocol. Applications still
coordinate which process owns writes and how readers learn that a newer
generation exists.

## Bindings and servers

- Heavy Python operations release the GIL where conversion is not required,
  but correctness is provided by the Rust ownership/session model, not by the
  GIL.
- Bolt owns `Arc<kglite::api::session::Session>` and per-connection transaction
  state.
- MCP uses the native session pipeline; writable workbench mode is explicit.
- A new binding owns async/runtime scheduling but should reuse
  `kglite::api::session` rather than creating a second lock/transaction model.

## Verification

The standing gates include Rust session/OCC tests, Python concurrency and
lifecycle tests, disk writer-lease/generation regressions, Loom session models,
native lock checks on macOS/Windows, Miri unsafe-loader checks, and scheduled
sanitizer/stress workflows. ThreadSanitizer is a manual/scheduled diagnostic,
not a substitute for the deterministic model tests.

See [Python transactions](../python/transactions.md),
[Rust session](../rust/session.md), and [Architecture](architecture.md).
