# Concurrency

kglite is designed for embedded use, but the read path is genuinely
parallel and the Bolt server (Phase B/C of `docs/history/bolt-implementation.md`)
runs N concurrent sessions against one shared `KnowledgeGraph`. This
document explains the thread-safety contract bindings can rely on and
the two documented contention quirks.

## TL;DR

- **Reads parallelize.** `graph.cypher()` releases the GIL via
  `py.detach()` during execution. Multiple Python threads (or async
  tasks calling into Python) can read in parallel against the same
  graph without serialization.
- **Mutations serialize.** `Arc::make_mut` in the mutation path
  guarantees write isolation but pays a copy-on-write (CoW) clone
  cost whenever a mutation runs with other references to the graph
  open (e.g. read-only transactions, ResultViews, cloned references).
- **Read-only transactions are free.** `begin_read()` takes an `Arc`
  reference (no clone). Use them for long-running read sessions to
  guarantee a stable snapshot.
- **A `KnowledgeGraph` is single-owner (Python).** Sharing one instance
  across threads is safe for concurrent *reads*, but a read that overlaps
  a *mutation* on the same object raises a clear `RuntimeError` (it never
  panics or corrupts). For shared concurrent reads, take an immutable
  `graph.freeze()` snapshot; for per-worker mutation, give each thread its
  own cheap `copy()`. See the next section.
- **A `Session` is shareable for reads *and* writes.** When you need many
  threads to read **and** mutate one graph (an MCP / request server),
  `graph.session()` returns a `Session` whose methods are all `&self`:
  `cypher()` reads run lock-free against momentary snapshots, and
  `execute()` writes serialize behind an internal writer lock so they
  *compose* (each write begins from the prior writer's committed state — no
  lost updates). This is the supported alternative to wrapping a single
  mutable `KnowledgeGraph` in a global lock. See "The `Session` handle".

## How the read path parallelizes

The `cypher()` method on `KnowledgeGraph` (and on `Transaction`):

1. Clones the `Arc<DirGraph>` reference (O(1), atomic refcount bump).
2. Releases the GIL via `py.detach()`.
3. Runs the executor against the cloned `Arc` reference. Inside the
   executor, the graph is shared-immutable; multiple threads can
   traverse it concurrently without locking the structural data
   (nodes, edges, types).
4. Reacquires the GIL only to convert the result back to Python
   objects.

Existing test coverage at `tests/test_concurrency.py`:

- `test_concurrent_cypher_reads` — 4 threads with disjoint filters.
- `test_concurrent_reads_produce_correct_results` — 8 threads with
  the same query, asserting all return the sequential baseline.
- `test_concurrent_reads_result_equivalence` — 4 ThreadPoolExecutor
  workers, full result-set equality.
- `test_16_concurrent_readers_complete_correctly` — 16 threads × 4
  queries = 64 reads, all return baseline. Pins Bolt-scale read
  parallelism.
- `test_no_panic_under_high_contention` — 32 threads (16 readers +
  16 mutators) for 500 ms over a shared `Session`. Asserts zero panics
  and zero errors.

## Free-threading (no-GIL / 3.14t) support

The extension declares `gil_used = false` (PyO3 0.29), so it imports and
runs under a free-threaded CPython build with the GIL genuinely disabled
— validated in CI (the `free-threading` job builds against `3.14t` and
asserts `sys._is_gil_enabled()` is `False` under a threaded query load).
Because the read path already releases the GIL and runs against a
shared-immutable `Arc<DirGraph>`, true parallelism is the same model
described above — there's just no GIL serializing the Python-side glue.

The shareable handles (`Session`, `FrozenGraph`, the Cypher `ResultView`)
are `#[pyclass(frozen)]` — immutable and `Sync` — which is what lets the
free-threaded build accept them. `tests/test_freethreading.py` exercises
concurrent `Session` reads, composable `Session.execute` writes, and
concurrent `FrozenGraph` readers; it runs on every build and adds the
no-GIL assertion when the interpreter is free-threaded.

One thing genuinely changes under no-GIL: sharing a **bare**
`KnowledgeGraph` across threads where any thread mutates is no longer
masked by GIL serialization, so the single-owner guard fires on every
real overlap (a clear `RuntimeError`, never memory corruption). This was
always unsupported — use `session()` / `freeze()` / `cursor()`. See
*What's NOT supported* below.

## The single-owner contract & `freeze()` snapshots (0.11.0)

The parallelism above is about the *engine*. At the **Python binding** there
is one more rule: a `KnowledgeGraph` object is **single-owner**. PyO3 guards
each `#[pyclass]` with a `RefCell`-style borrow; a method that mutates
(`add_nodes`, `add_connections`, `replace_connections`, `embed_texts`, a
`CREATE`/`SET`/`DELETE`/`MERGE` query, `save`) holds the exclusive borrow for
its duration. Concurrent *reads* coexist fine (each `cypher()` borrows only
momentarily to clone the `Arc`, then releases and runs GIL-free); a read or
mutation that overlaps another thread's in-flight mutation on the **same
object** now raises a clear `RuntimeError` naming the contract — never a panic,
never silent corruption.

Two safe shared-graph patterns, both leaning on kglite's cheap builds:

- **Per-worker (`copy()`).** Give each thread its own graph. `graph.copy()`
  is a deep copy, but builds are fast and the copies are independent — fine for
  worker pools that each mutate.

- **Frozen snapshot (`freeze()`) — for concurrent reads.** `graph.freeze()`
  returns a `FrozenGraph`: an immutable view that shares the source graph's
  data via an **O(1) `Arc` clone** (no deep copy) and exposes *only* read
  methods. Because it has no mutating method, no exclusive borrow can ever
  fire, so any number of threads can run `FrozenGraph.cypher()` (including
  `text_score()` / `vector_score()` semantic search) against the same snapshot
  in parallel, lock-free, with the GIL released. The snapshot is stable under
  copy-on-write: mutating the *source* graph afterwards leaves the frozen view
  on the original data.

  ```python
  snapshot = graph.freeze()          # O(1); share across reader threads
  snapshot.cypher("MATCH (n:Doc) RETURN count(n)")   # safe from N threads at once
  # build → freeze → serve readers → swap in a fresh freeze() when data changes
  ```

This is the recommended model for a read-heavy server that occasionally
rebuilds: build a new graph, `freeze()` it, serve readers off the snapshot, and
atomically swap in the next snapshot when the data changes — rather than a
global lock around a single mutable instance.

## The `Session` handle — shared reads *and* writes (0.11.3)

`freeze()` solves *shared reads*. When a server must also **mutate** one graph
from many threads — the shape that forces consumers to wrap every call in a
global lock — use `graph.session()`. A `Session` wraps the engine's
`Mutex<Arc<DirGraph>>` and exposes only `&self` methods, so it is safe to share
across a thread pool:

- **Reads (`cypher`, `snapshot`)** take a momentary snapshot (O(1) `Arc`
  clone), release the lock, and run GIL-free. Unlimited concurrent readers; no
  blocking during execution.
- **Writes (`execute`)** take an internal **writer lock** held across the whole
  `begin → mutate → commit`: a copy-on-write working copy is mutated, then the
  graph's `Arc` is swapped atomically. The writer lock is what makes concurrent
  writes **compose** — writer B's `begin()` snapshots writer A's *committed*
  state, so increments and read-modify-write updates build on each other
  instead of racing into a lost update. Readers never block on the writer; they
  keep seeing the pre-commit snapshot until the swap lands.

```python
store = graph.session()                       # share across the thread pool
# ...or load a saved graph straight into a shared handle in one call:
store = kglite.open_session("graph.kgl")      # == kglite.load(...).session()
store.cypher("MATCH (n:Doc) RETURN count(n)") # lock-free reads, N threads
store.execute("CREATE (n:Doc {id: 1})")       # serialized writes, compose
fz = store.snapshot()                          # stable FrozenGraph view (cypher only)
cur = store.cursor()                           # per-thread FULL fluent handle
cur.select("Doc").where({"team": "A"}).to_df() # ...select/where/traverse/to_df
```

**`snapshot()` vs `cursor()`.** Both hand out a per-call handle bound to the
session's current state, so both are safe to fan across threads. `snapshot()`
returns a read-only `FrozenGraph` exposing just `cypher()`. `cursor()` returns a
`KnowledgeGraph` bound to the same kind of snapshot but with the **whole fluent
surface** (`select`/`where`/`sort`/`traverse`/`to_df`/`collect`/…) — the
per-thread fluent analogue. Each `cursor()` call is an independent single-owner
handle, so N threads run fluent chains in parallel without the borrow conflict a
*shared* live `KnowledgeGraph` raises. Mutating a cursor is copy-on-write
isolated (it does not write back to the session); take a fresh `cursor()` to see
later session writes. Pinned by `tests/test_session.py` (per-thread cursor
fluent concurrency).

A `Session` is an **independent owner** seeded from the graph's state at
`session()` time: it shares the `Arc` at creation, but once either side
mutates, copy-on-write forks them. Build / load with a `KnowledgeGraph`, then
`.session()` and serve every thread through the `Session` — don't keep mutating
the original graph afterward.

**Cost model.** Reads add one momentary mutex acquire (nanoseconds). A write
with no reader snapshot outstanding mutates in place (refcount 1) — same as a
`KnowledgeGraph` write. A write that overlaps a held `snapshot()` deep-clones
the graph (the inherent price of snapshot isolation; ~O(graph size), transient
~2× memory). Keep snapshots short-lived to stay on the in-place path. Pinned by
`tests/test_session.py` (mixed reader/writer, concurrent-write compose).

**How well it scales depends on what the query does.** Independent
validation across a `freeze()` snapshot measured near-linear scaling for
**CPU-bound** queries (traversal + aggregation: ~1.95× / 3.6× / 6.3× at
2 / 4 / 8 threads) and sub-linear scaling for **memory-bandwidth-bound** ones
(a full-scan `count_edges` over ~1M edges: ~1.5× at 4 threads, tailing off at 8)
— the latter is the memory bus saturating, not a `freeze()` limit. Practical
guidance: fan out `freeze()` readers aggressively for compute-heavy work
(traversals, scoring, aggregation over filtered sets); expect diminishing
returns when the bottleneck is one big sequential scan of the whole graph.

## How the mutation path is isolated

`graph.cypher("CREATE ...")` (and any other mutation) needs an
exclusive write reference to the `DirGraph`. The implementation
uses `Arc::make_mut`, which gives copy-on-write semantics:

- If the `Arc` refcount is 1 (no other live references), the
  mutation happens in-place: O(1) write isolation.
- If the refcount is > 1, `Arc::make_mut` deep-clones the entire
  `DirGraph` to give the mutator a private copy; other refs continue
  to see the original (pre-mutation) state.

This means **reads and mutations never block each other** — readers
get a stable snapshot, mutators get isolation. The trade-off: a
mutation that races with active read-only transactions or held
ResultView references pays a clone cost proportional to the graph
size.

At Bolt scale (tens of connections, sparse write patterns), this
is fine. At sustained-write-heavy workloads on large graphs (10M+
nodes), the clone cost shows up. If you find yourself in that
shape, see "Phase C performance considerations" below.

## The two documented quirks

### Quirk 1: WKT cache write-lock on first encounter

`DirGraph.wkt_cache: Arc<RwLock<HashMap<String, Arc<geo::Geometry>>>>`
caches parsed WKT geometries to avoid re-parsing on every spatial
predicate. The cache is read-locked for hits and write-locked
on misses (first encounter with a given WKT string).

- **Per-WKT contention.** N threads simultaneously parsing the
  *same* novel WKT string will serialize through the write lock
  for the first thread; the rest hit the (now-warmed) read path.
- **Microsecond-scale.** WKT parsing is fast; the contention
  window is per-query, not per-row.
- **Bolt-scale impact: negligible.** At ≤100 concurrent sessions
  with realistic WKT diversity, the cache warms within the first
  few requests and then operates entirely on the read path.

Pinned by `test_wkt_cache_warmup_is_safe_under_contention` — 8
threads concurrently evaluating the same `contains(area, point)`
query complete without panic or wrong results.

### Quirk 2: `Arc::make_mut` CoW clone on mutation with held refs

When `begin_read()` is active (or any other code holds an
`Arc<DirGraph>` reference), an outside mutation triggers a full
graph clone before mutating. The reader's snapshot is preserved
intact; the outside view reflects the new state.

- **Per-mutation cost.** Proportional to graph size: ~10 ns/node +
  ~5 ns/edge + property data. For a 100k-node / 500k-edge graph,
  that's ~5 ms; for a 10M-node graph, ~500 ms.
- **Bolt-scale impact: depends on write pattern.** Bolt sessions
  that mostly read with sparse writes don't notice. Sustained-write
  workloads with many open read sessions multiply the cost.
- **Mitigation:** consolidate writes into batch transactions
  (`begin()` / many `cypher()` / one `commit()`) to amortize the
  clone cost across many mutations.

Pinned by `test_arc_makemut_cow_isolates_reader_from_mutation` — a
read-only transaction keeps reading the pre-mutation state while
an outside mutation succeeds and changes the post-mutation view.

## Bolt server recipe

```rust
// One Arc<KnowledgeGraph> shared across all tokio tasks.
let graph = Arc::new(KnowledgeGraph::load("...")?);

// Per-connection task:
tokio::spawn(async move {
    let graph = Arc::clone(&graph);
    handle_bolt_session(graph).await;
});

async fn handle_bolt_session(graph: Arc<KnowledgeGraph>) {
    // Reads parallelize across sessions; mutations serialize via
    // Python's GIL on the transaction.cypher() call.
    loop {
        match bolt_recv().await {
            Bolt::Begin => let tx = graph.begin()?,
            Bolt::Run(query) => /* tx.cypher(query) or graph.cypher(query) */,
            Bolt::Commit => tx.commit(),
            Bolt::Rollback => tx.rollback(),
        }
    }
}
```

## What's NOT supported (and why)

- **Lock-free MVCC.** kglite uses `Arc::make_mut` for isolation, not
  a versioned multi-snapshot store. Each mutation that races a held
  reference pays a clone; we accept that for simplicity. (`freeze()`
  gives a *single* immutable, lock-free read snapshot via copy-on-write
  — not a continuously-versioned multi-snapshot store.)
- **Cross-process concurrency.** kglite is embedded. For multi-
  process workloads, use the Bolt server as the coordination point.
- **Lock-free reads of mutating fields.** Caches like `wkt_cache`
  and `edge_type_counts_cache` are `RwLock`-protected, not lock-free.
  The contention is bounded but not zero.

## Phase C performance considerations

Two concerns that aren't blockers for first-cut Bolt but are worth
profiling once the server is live:

1. **GIL contention on write commits.** Every `tx.commit()` reacquires
   the GIL. If profiling reveals contention, a future release can
   add a Rust-native `TransactionHandle` that bypasses the Python
   boundary. Not in scope for Phase B/C.
2. **`Arc::make_mut` clone cost on write-heavy + many-reader
   workloads.** If you see ms-scale write latencies with many
   open read sessions, the `Session` handle (see "The `Session`
   handle") implements exactly the write-mutex-to-a-single-committer
   strategy: writes serialize behind one lock and compose, while
   reads stay lock-free off snapshots.

## Performance reference

Phase A.3 / 0.9.53 measured concurrent-read scaling on Apple M4 (4
performance cores + 6 efficiency cores = 10 logical) using the
audit script `scripts/perf_audit.py`. Bench query is a 5k-Person
WHERE + count.

| Threads | µs/query | Speedup | Efficiency | Note |
|---:|---:|---:|---:|---|
| 1 | 489 µs | 1.00× | 100% | sequential baseline |
| 2 | 246 µs | 1.99× | 99% | |
| 4 | 129 µs | 3.80× | 95% | **fills perf cores** |
| 8 | 94 µs | 5.19× | 65% | efficiency cores active |
| 10 | 84 µs | 5.82× | 58% | full CPU saturation |
| 16 | 85 µs | 5.74× | 36% | hardware-bound plateau |
| 32 | 86 µs | 5.69× | 18% | no additional throughput |

The plateau past 8-10 threads is **hardware-bound on M-series CPUs**
— Apple Silicon has heterogeneous cores (perf cores ~2× faster than
efficiency cores). Theoretical max on M4: 4 + 6 × 0.5 = ~7×; we hit
5.8×, which is ~83% of theoretical. On homogeneous x86 server CPUs
(EPYC, Xeon) the linear scaling region extends further; expect
near-linear speedup up to the physical core count.

### What Phase A.3 / 0.9.53 changed

- Issue #3 fix: moved `resolve_noderefs` into the executor's
  `py.detach` block so it runs GIL-free along with the rest of
  Cypher execution. Improved per-thread efficiency by 2-3 percentage
  points across all thread counts.
- The remaining inefficiency at 4-thread scale (~5%) splits between
  heap allocator contention and GIL re-acquisition on PyObject
  result construction — both system-wide concerns rather than
  kglite-specific bottlenecks. A future release can revisit if
  profiling reveals headroom.

## See also

- [`transactions.md`](../python/transactions.md) — `begin()` / `commit()` /
  OCC semantics + per-call cost reference.
- [`error-handling.md`](../python/error-handling.md) — typed exception
  hierarchy.
- `tests/test_concurrency.py` — the executable contract for the
  guarantees above.
- `scripts/perf_audit.py` — re-runnable audit harness for the
  scaling numbers.
