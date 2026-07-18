# Concurrency verification — `Session` / commit path

This records how the `Session` concurrency model is verified, what each layer
guarantees, and the one real bug the verification found. It exists because a
concurrency primitive needs more than "the functional tests pass" — races are
timing-dependent and the GIL can mask them at the Python layer.

## The verification stack

| Layer | Catches | Where | How to run |
|---|---|---|---|
| Functional concurrency tests | **logical** races (atomicity violations, lost updates) | `crates/kglite/src/graph/session/transaction.rs` (`tests::concurrent_*`) — real OS threads, no GIL | `cargo test -p kglite --lib session::transaction::tests` |
| Python stress harness | empirical races end-to-end under sustained load | `tests/test_session_stress.py` (`-m stress`) | `pytest tests/test_session_stress.py -m stress` |
| ThreadSanitizer | **data** races (unsynchronized memory access) | the same Rust `concurrent_*` tests | see below |
| `unsafe impl` audit | soundness of the `Send`/`Sync` claims | this doc | review |
| loom model | **exhaustive** interleaving proof of the lock pattern | `crates/kglite/tests/loom_session.rs` | `RUSTFLAGS="--cfg loom" cargo test -p kglite --test loom_session` |

The two race classes are different and both matter: TSan finds *data* races
(two threads touching memory without synchronization); the functional tests
find *logical* races (each access synchronized, but the check-then-act sequence
isn't atomic). **The bug below was a logical race — every memory access was
already under the `Mutex` — so TSan would not have caught it; the functional
concurrency tests did.**

## The bug this found (fixed 0.11.3)

`Session::commit` had a **TOCTOU race**. The optimistic-concurrency version
check read the version under one lock acquisition (`self.version()` → lock,
read, *unlock*) and then swapped the graph `Arc` under a *separate* lock. Two
threads committing at once could both pass the check and both swap — losing one
commit, and (because the new version derived from the transaction's stale base)
leaving the monotonic version counter **non-monotonic** (it could go backwards).

Surfaced immediately by `concurrent_writers_compose_with_occ_retry` (final
version `724 != 1600`) and `concurrent_snapshots_consistent_under_commits`
("version went backwards: 81 < 83").

**Who was exposed:** the **bolt-server**, which drives the core `Session` from
many connection threads with no serializing lock. The Python `Session` was
*not* exposed — its writer lock already serializes committers, which is exactly
why the Python stress harness passed and only the true-parallel Rust tests
caught it. (A good reminder that GIL/lock-masking can hide a core bug behind a
green binding test suite.)

**The fix:** hold one lock guard across both the OCC check and the swap (atomic
compare-and-swap), and bump the version from the *current* value, so commits
are atomic and the version is monotonic even in last-writer-wins mode —
monotonicity is required for OCC soundness ("version changed ⇒ graph changed").

## Running ThreadSanitizer

TSan needs a nightly toolchain and instrumented std (`-Zbuild-std`). On
aarch64-apple-darwin:

```bash
RUSTFLAGS="-Zsanitizer=thread" \
  rustup run nightly cargo test -p kglite --lib \
  session::transaction::tests::concurrent \
  -Zbuild-std --target aarch64-apple-darwin
```

The documented verification run on the fixed code reported no data races.
ThreadSanitizer is a manual maintainer check, not a CI or default `make test`
gate; it needs nightly plus `build-std`, and the first instrumented build is
slow.

## `unsafe impl Send/Sync` audit

- **In-memory graphs (the default backend, and what `Session` targets): no
  `unsafe`.** `DirGraph` over the petgraph backend derives `Send + Sync`
  safely. The entire Session concurrency story rests on zero unsafe code.
- **Disk mode** (`crates/kglite/src/graph/storage/disk/graph.rs:305`): two
  `unsafe impl Send/Sync` on `DiskGraph`. Sound for the shared-read path —
  `node_arena` / `edge_arena` are `Mutex<Vec<Box<…>>>` (the `Box` gives stable
  heap pointers across `Vec` realloc; the `Mutex` serializes pushes from
  concurrent `node_weight` callers), and the mmap-backed columns are read-only.
  The one remaining `UnsafeCell` (`pending_edges`) is only accessed through
  `&mut self` (`build_csr_from_pending` / `compact`), never during shared
  reads, so the borrow checker enforces exclusivity. The historical
  `UnsafeCell<Vec<NodeData>>` races (0.9.2 disk regression — silent wrong-row
  reads + use-after-free) were fixed in 0.9.3 by switching to the `Mutex`
  pattern.
- **Caveat (functional, not soundness):** a `Session` *write* on a disk-mode
  graph goes through `DirGraph` copy-on-write (clone the working copy). That is
  a cost/support concern for disk + Session writes, not a memory-safety one.

## loom — exhaustive interleaving proof (done)

[loom](https://github.com/tokio-rs/loom) exhaustively explores every legal
thread interleaving of a bounded concurrent model — the strongest possible
signal for exactly the bug class fixed above. It's wired at
`crates/kglite/tests/loom_session.rs`.

loom requires the synchronization primitives under test to be loom's
instrumented `loom::sync::{Arc, Mutex}`, and `Arc<DirGraph>` is too pervasive to
swap under `#[cfg(loom)]`. So the model is a **hand-written model of the
algorithm**: the monotonic version counter — the value the TOCTOU race
corrupted — stands in for the graph, and the commit logic mirrors the real
`Session::commit` exactly (one guard held across the version read and the swap,
new version derived from current). Three models pass under loom:
`occ_retry_commits_compose`, `lww_commits_are_monotonic`,
`snapshot_never_torn_under_commit`. Changing the model to release the lock
between the read and the swap (the pre-fix shape) makes loom find the
commit-dropping interleaving and the assertions fail — i.e. loom would have
caught the original bug.

It's gated behind `--cfg loom` (loom is a `cfg(loom)`-only dev-dependency), so
normal `cargo test` / `make lint` never compile it. Run:

```bash
RUSTFLAGS="--cfg loom" cargo test -p kglite --test loom_session
```

The one caveat is inherent to the approach: it proves the *algorithm*, and
must be kept faithful to `Session::commit` by hand (there's no compiler link
between the two). If the real locking logic is extended (finer-grained or
node-level locks), update the model to match.
