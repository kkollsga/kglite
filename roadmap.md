# KGLite Architecture Roadmap — `KnowledgeGraph` handle decomposition

> Created 2026-06-18. This is a **down-the-line** architectural roadmap, not
> scheduled work. It records the root-cause design smell behind the
> concurrency work, the target layering, and the decision triggers for when
> it becomes worth paying down. The near-term concurrency fix (the `Session`
> pyclass) is deliberately designed to be the *seam* this decomposition would
> build on, so shipping it now is a step toward this plan, not away from it.

> _(Filename note: requested as `roadmap.me`; created as `roadmap.md` so it
> renders on GitHub / readthedocs. Rename on request.)_

## The smell: `KnowledgeGraph` is a god-object

`crates/kglite-py/src/graph/mod.rs` — `KnowledgeGraph` conflates **three
responsibilities with three different lifetimes and ownership models** in one
`#[pyclass]`:

1. **Shared storage** — `inner: Arc<DirGraph>`. Genuinely shareable,
   thread-safe (`DirGraph` is `Send + Sync`).
2. **Session / lifecycle** — `durable: DurableState` (owns a WAL `File`
   handle), `embedder`, `source_path`, version/commit coordination.
   Per-process, long-lived.
3. **Query cursor** — `selection: CowSelection`, `default_timeout_ms`,
   `default_max_rows`, `temporal_context`, `last_mutation_stats`,
   `reports`. Per-**caller**, disposable, thread-affine.

Responsibility (3) is *why* the handle cannot be shared across threads even
though (1) is begging to be. The fluent cursor (`selection`) and the shared
graph live in the same struct, so `.select().where_(...).sort()` chains on a
shared handle would corrupt each other's cursor — a **semantic** hazard no
amount of locking fixes. This conflation is the root cause that the near-term
`Session` pyclass works *around* rather than *through*.

## Target layering (greenfield ideal)

Three types, each with one responsibility and one ownership model:

| Layer | Type | Owns | Sharing model |
|---|---|---|---|
| Storage | `DirGraph` | nodes/edges/indices | already `Send + Sync`; shared via `Arc` |
| Session | `Session` | shared graph + lifecycle (durability, embedder, path, version/commit) | **the unit of sharing**; thread-safe (`Mutex<Arc<DirGraph>>`) |
| Cursor | `Cursor` / `View` | `selection`, per-call defaults, `temporal_context`, last-result state + an `Arc` snapshot | **never shared**; cheap, per-caller, disposable, thread-affine |

Under this split the concurrency problem dissolves structurally: **share the
`Session`, each thread spins up its own throwaway `Cursor`.** No shared mutable
cursor, no locking gymnastics, no `selection`-corruption hazard. The fluent API
operates on the `Cursor`; the `Session` is the long-lived shared anchor.

`DirGraph` already sits correctly at the bottom. The core `Session` type
(`crates/kglite/src/graph/session/transaction.rs`) already exists and is the
right shape — it just needs to grow into the full lifecycle home (see below).
The missing piece is extracting the `Cursor` out of `KnowledgeGraph`.

## Why this is deferred (not done now)

- **Large breaking change.** Splitting the cursor out of `KnowledgeGraph`
  touches every fluent method, the entire `kglite/__init__.pyi` stub surface,
  `describe()` introspection, and every downstream consumer. It is a
  major-version event.
- **Payoff is architectural, not user-facing.** No user is asking for a
  decomposed handle; they are asking for safe concurrency — which the
  near-term `Session` pyclass delivers without the decomposition. By the
  CLAUDE.md "test the use case" bar, doing the decomposition as a prerequisite
  is over-engineering.
- **The seam can be installed cheaply now.** The `Session` pyclass (near-term
  work) introduces the shared unit that this decomposition would build on.
  Once it exists, the eventual cursor-split is "move `selection` + defaults out
  of `KnowledgeGraph` into a `Cursor` that borrows a `Session`" rather than a
  ground-up redesign.

## Prerequisite (near-term, separate work)

The `Session` pyclass + the Phase-E transaction consolidation (folding
`pyapi::Transaction` onto core `Session`) land first. They establish:

- `Session` as the canonical shared, thread-safe anchor in the wheel.
- A single transaction engine (no more parallel `pyapi::Transaction` copy).
- `FrozenGraph` as the read snapshot a `Session` hands out.

This roadmap assumes those are in place.

## Phased decomposition plan (down the line)

Each phase is independently shippable and green; ordered to keep the public
fluent API working until the final cutover.

1. **Enrich core `Session` into the lifecycle home.** Migrate the
   binding-agnostic parts of lifecycle into core `Session`: `version`,
   embedder registration, source-path tracking, and the durability
   coordination that is *not* the OS `File` handle. The `File` handle in
   `DurableState` legitimately stays per-binding (OS lifecycle). Lift follows
   the CLAUDE.md LIFT posture: generic-and-useful into core, tailored shapes
   stay in the wrapper.

2. **Introduce `Cursor` internally (non-breaking).** Add a `Cursor` struct
   carrying `selection`, `default_timeout_ms`, `default_max_rows`,
   `temporal_context`, `last_mutation_stats`, `reports`, plus an `Arc<DirGraph>`
   snapshot. Re-implement `KnowledgeGraph`'s fluent methods as thin delegates to
   an internal `Cursor`. No public API change yet — `KnowledgeGraph` still owns
   one `Cursor` + one `Session` reference behind the scenes.

3. **Expose `Session` + `Cursor` as the primary surface.** `Session.cursor()`
   (or `Session.query()`) returns a fresh `Cursor`; the fluent chain lives on
   `Cursor`. `KnowledgeGraph` becomes a thin convenience facade =
   `Session` + a default `Cursor`, retained for single-owner ergonomics and
   back-compat.

4. **Migrate stubs, introspection, docs.** Update `kglite/__init__.pyi`,
   `describe()` output, `FLUENT.md`, and guides to teach the
   `Session`/`Cursor` model. Per the five-place checklist.

5. **(Major version) Decide `KnowledgeGraph`'s fate.** Either keep it as the
   ergonomic single-owner facade indefinitely (likely — it is a good default
   handle) or deprecate the conflated parts. No back-compat shims per CLAUDE.md;
   if anything is removed, it is removed in the same PR as its replacement.

## Risks

- **Fluent API churn.** The fluent surface is large; the internal-`Cursor`
  step (Phase 2) must be a pure refactor with the fluent test suite green
  before any surface change.
- **Snapshot semantics of a `Cursor`.** A `Cursor` holding an `Arc` snapshot
  observes a point-in-time graph; document that a long-lived `Cursor` will not
  see concurrent commits (same CoW deep-clone-on-write cost as `FrozenGraph` —
  see the concurrency benchmark).
- **Type proliferation.** The wheel already exposes `KnowledgeGraph` +
  `FrozenGraph` + `Session` + `Transaction`. Adding `Cursor` must come with
  consolidation (e.g. `FrozenGraph` becomes a read-only `Cursor`, or the two
  unify) so net surface does not balloon.
- **`DurableState` `File` handle.** Cannot be cloned/shared; the decomposition
  must keep durability bound to exactly one owner (the `Session`), never a
  `Cursor`.

## Decision triggers — when this becomes worth doing

Pay this down when **any** of:

- A second non-Rust binding (Go/JVM) is built and re-implements cursor/session
  separation by hand — the drift the CLAUDE.md "centrally maintained" north
  star exists to prevent.
- Concurrency demand outgrows the `Session` pyclass — e.g. users want
  per-thread fluent chains against a shared store, which the facade cannot
  express cleanly.
- A major version is already being cut for other breaking reasons (amortize
  the API churn).

Until a trigger fires, the `Session` pyclass is the supported concurrency
story and this remains recorded debt, not scheduled work.
