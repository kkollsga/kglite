# Adding a storage backend

This is contributor guidance for a new production storage mode. Read
[Architecture](architecture.md) and [Concurrency](concurrency.md) first: a
backend participates in query execution, persistence, lifecycle, and the
single-writer/stable-reader contract—not only trait dispatch.

## Core traits

`crates/kglite/src/graph/storage/mod.rs` defines:

- `GraphRead` for counts, lookup, iteration, properties, and traversal.
- `GraphWrite: GraphRead` for mutations.

`GraphRead` uses generic associated iterator types and is not object-safe. Use
`&impl GraphRead`, never `&dyn GraphRead`. Add a genuinely shared operation to
the trait first, then implement it for every backend.

## Integration checklist

1. Implement the storage type and its `GraphRead`/`GraphWrite` behavior under
   `graph/storage/`.
2. Add dispatch in `storage/backend.rs` and exports in `storage/mod.rs`.
3. Wire construction/configuration in `storage/config.rs` and the API storage
   facade. Decide what `is_memory`/`is_mapped`/`is_disk` means for the mode.
4. Integrate open/save/copy/compaction semantics. Portable `.kgl` snapshots and
   disk-generation directories have different publication lifecycles.
5. Preserve indexes, schema, interner identity, tombstones, overlays, writer
   leases, and stable readers across save/reopen.
6. Add mode construction at every binding surface only if it is a user-facing
   backend; keep binding-specific parsing outside the core.

## RecordingGraph is a decorator, not a fourth storage mode

`storage/recording.rs` is the production WAL write-capture wrapper. Reads
forward without logging or locking. Mutation methods append `RawOp` values to a
buffer so durable in-memory lifecycle code can publish/replay them. It is useful
for learning GAT forwarding and wrapper transparency, but it does not teach the
construction/persistence work required by a new backend.

## Required verification

- Rust trait/unit tests for every operation and iterator lifetime.
- `tests/test_storage_parity.py` plus `tests/test_phase{1,2,3}_parity.py` for
  memory/mapped/disk equivalence.
- Persistence, copy-independence, lifecycle, concurrency, and sidecar-integrity
  suites relevant to the mode.
- Small in-memory benchmarks before changing shared planner/executor paths;
  in-memory remains the performance gate.
- `make gate`, targeted `kglite` tests, and the matching storage parity suites.
  GitHub CI owns workspace clippy, the broad test matrix, and C-ABI/API profile
  checks described in `AGENTS.md`.

Start at `storage/mod.rs`, `storage/backend.rs`, and the closest existing
backend. Do not copy internal implementation details across modes when a trait
operation or mode-specific strategy can state the contract directly.
