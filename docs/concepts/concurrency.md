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
keep that generation mmaped. A cross-process writer lease prevents two
processes from publishing concurrently, held from a writer's first mutation
until the publish that ends it and re-taken by the next mutation, so a process
that has finished publishing excludes nobody. Readers do not take the writer
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
  state. Its client-facing contract — reads on snapshots, every write an
  explicit transaction ordering at commit, conflicts retriable and absorbed by
  driver-managed transactions, and what that does to throughput and latency as
  writers are added — is stated in
  [Bolt server → Write concurrency](../operators/bolt-server.md#write-concurrency).
- A session can also be **durable**: `Session::open_durable(graph, path, level)`
  recovers the path's write-ahead sidecar before the session serves anyone, and
  every later commit appends its frame between the OCC check and the Arc swap,
  so a frame that cannot be written blocks the publish instead of following it.
  Concurrency is unchanged — one owner per path, readers still on snapshots —
  but the commit point now includes an append, and at `full` a device barrier
  taken inside the lock all writers serialize on. Bolt exposes the level as
  `--durability`; see
  [Bolt server → Durability](../operators/bolt-server.md#durability) for the
  per-level loss windows and the measured cost.
- MCP uses the native session pipeline; writable workbench mode is explicit.
- A new binding owns async/runtime scheduling but should reuse
  `kglite::api::session` rather than creating a second lock/transaction model.

## Verification

The standing gates include Rust session/OCC tests, Python concurrency and
lifecycle tests, disk writer-lease/generation regressions, Loom session models,
native lock checks on macOS/Windows, Miri unsafe-loader checks, and scheduled
sanitizer/stress workflows. ThreadSanitizer is a manual/scheduled diagnostic,
not a substitute for the deterministic model tests.

At the Bolt level specifically: `tests/test_bolt_server_concurrency.py`
(concurrent readers, readers against a writer, competing transactions, session
teardown under load), the managed-retry contention test in
`tests/test_bolt_server_transactions.py` (two transactions made to collide
deterministically, then required to both land via driver retry), and the
contended-writer load test `tests/benchmarks/test_bench_bolt_writers.py`
(writer-count sweep producing the throughput/retry/latency curve behind the
operator contract). All three are opt-in (`-m bolt_stress`).

Durability under that same model is pinned by
`tests/test_bolt_server_durability.py` (`-m bolt`): child-process
`SIGKILL`-and-restart tests at each `--durability` level, with `off` as the
control that loses the commit; the unconditional recovery-on-open behaviour and
its `off`-over-an-unreplayed-log refusal; and that a checkpoint truncates the
log while post-checkpoint commits still recover. `tests/test_durability.py` and
`tests/test_durable_save.py` cover the same engine surface from the embedded
side. The cost of each level — why the Bolt default is `normal` — is the
`test_durability_sweep` cell of the benchmark above, alongside
`test_checkpoint_under_contention`.

See [Python transactions](../python/transactions.md),
[Rust session](../rust/session.md), and [Architecture](architecture.md).
