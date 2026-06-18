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

## Why this was deferred until 0.11.3 (historical rationale, now superseded by Status below)

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

## Status (2026-06-18)

**Prerequisite shipped (0.11.3).** The `Session` pyclass + the Phase-E
transaction consolidation are in: `Session` is the canonical shared,
thread-safe anchor; there is a single transaction engine (no parallel
`pyapi::Transaction` copy); `FrozenGraph` is the read snapshot a `Session`
hands out.

**Decomposition: IN PROGRESS.** Full scope (Stage A + B) approved 2026-06-18.
The grounding investigation found the seams are unusually clean — the ~25–30
fluent methods funnel through one uniform `self.clone()` → mutate-`selection`
pattern, `selection` never escapes into the core query engine (core
`filtering::*` already takes `&mut CowSelection`), and `FrozenGraph`/`Session`
already demonstrate "graph handle without a cursor". So the public API need
not change until a deliberate, opt-in final step.

### End-state correction (de-risks the whole effort)

We do **not** wrap the live `KnowledgeGraph` in the `Session`'s
`Mutex<Arc<DirGraph>>` — that would add lock overhead to the single-owner hot
path (a perf regression CLAUDE.md forbids). The live `KnowledgeGraph` keeps
`inner: Arc<DirGraph>` + `make_mut` CoW exactly as today. The decomposition is
**cursor extraction + lifecycle-field grouping**, not re-homing storage. The
shareable `Session` (already shipped) remains the multi-thread variant.

### Cross-cutting risk controls (every phase)

1. Characterization/golden tests **before** any production change.
2. Mechanical (compiler-checked field moves) **before** semantic (logic).
3. Internal **before** public — `KnowledgeGraph` behaviour is byte-identical
   through all of Stage A; no breaking change anywhere in this plan.
4. One commit per phase, each green (lint + full suite + golden + benchmark).
5. Benchmark gate on structural phases — field indirection must be zero-cost.
6. Each phase independently valuable.

## Stage A — internal restructure (non-breaking; shippable as patches)

- **Phase 0 — Characterization/golden tests.** No production code. Pin: `Clone`
  vs `copy()` field preservation (selection/reports/temporal/last_stats/
  source_path kept; `durable=None`; copy resets selection+source_path); fluent
  chain selection inheritance; `temporal_context` via `date()`; `reports`
  accumulation; `last_mutation_stats` lifecycle; `source_path` on derived views;
  the `update()` special case (mutates `inner` *and* returns a derived handle).

- **Phase 1 — Extract `CursorState` struct (mechanical grouping).** Group
  `selection`, `temporal_context`, `last_mutation_stats`, `reports` into
  `cursor: CursorState`. Pure `self.X` → `self.cursor.X`; update `Clone`/
  `copy()`. High LOC, lowest risk (compiler-checked, behaviour identical).

- **Phase 2 — Funnel fluent construction through one factory.** Replace the
  ~25–30 copy-pasted `let mut new_kg = self.clone(); …mutate
  new_kg.cursor.selection…` with a single `derive_with` helper — the seam the
  public `Cursor` reuses.

- **Phase 3 — Group lifecycle fields.** Group `source_path`, `durable`,
  `default_timeout_ms`, `default_max_rows` (and decide `embedder`) into a
  `GraphLifecycle` struct. Clarifies storage / lifecycle / cursor and sets up
  the deferred core-`Session` lifecycle lift.

- **Phase 4 — Promote `CursorState` → `Cursor` type; `KnowledgeGraph`
  delegates.** `Cursor` owns the fluent ops against a borrowed `&Arc<DirGraph>`,
  returning a new `Cursor`. `KnowledgeGraph` = `Arc<DirGraph>` + `Cursor` +
  `GraphLifecycle`; fluent methods delegate then rewrap. Public API still
  returns `KnowledgeGraph` (non-breaking). Fold in `&mut self` → `&self` on
  fluent methods (they already clone; the `&mut` is vestigial).

## Stage B — public surface (additive; a minor feature)

- **Phase 5 — Expose `Cursor` + `Session.cursor()`.** Additive: `Session.cursor()`
  / `graph.cursor()` returns a `Cursor` for per-thread fluent chains against one
  shared `Session` — the "flexible" goal `Session` alone can't express.
  `KnowledgeGraph` stays the convenience facade. New concurrency test: N threads,
  each its own `Cursor` off one shared `Session`.

- **Phase 6 — Docs / stubs / introspection + roadmap update.** Teach the
  storage → `Session` → `Cursor` model (`FLUENT.md`, `concurrency.md`,
  `__init__.pyi`, `describe()`), mark phases done here.

## Open decisions (safe defaults; revisit if they read wrong)

- **`reports` home** — default: keep in `CursorState` (preserves clone-on-derive).
- **`default_timeout_ms/max_rows` home** — default: `GraphLifecycle` (query
  config, not chain state); Phase 0 pins current inheritance first.

## Deferred (separate major version — NOT in this plan)

- **Enrich core `Session` as the cross-binding lifecycle home** (Go/JVM share
  durability/embedder logic) — invasive (`durable` owns an OS `File`); do it when
  a second binding forces it.
- **Deprecate any conflated `KnowledgeGraph` surface** — no back-compat shims,
  so a deliberate major-version event.

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
