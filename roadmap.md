# KGLite Architecture Roadmap — `KnowledgeGraph` handle decomposition

> Created 2026-06-18. **Status: the decomposition is COMPLETE (shipped in
> 0.11.3).** This file now reads as a record: the root-cause smell, the layering
> that was implemented, the plan as executed, and the small set of genuinely
> future items (trigger-gated). The headline `Status` section is the current
> truth; the `Background` and `Executed plan` sections are archival rationale.

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

**Decomposition: Stage A + B SHIPPED (0.11.3, 2026-06-18).** `KnowledgeGraph`
now decomposes into labeled concerns instead of 10 flat fields: `inner`
(shared `DirGraph` storage) + `cursor: CursorState` (per-query selection /
temporal / stats / reports) + `lifecycle: GraphLifecycle` (save target +
durability `File`) + `embedder` + 2 query-default scalars. `derive_with`
funnels fluent derivation through one choke point. The public capability ships
as **`Session.cursor()`** — a per-thread, snapshot-bound `KnowledgeGraph` with
the full fluent surface (see the design note below). All phases behaviour-
identical: the 10 characterization tests + full suite stayed green byte-for-
byte; perf showed no regression (most paths faster).

Phase map as executed: P0 characterization tests · P1 `CursorState` ·
P2 `derive_with` funnel · P3 `GraphLifecycle` · **P4+P5 folded** →
`Session.cursor()`.

**Design note on P4/P5 (the public Cursor).** The original framing was
"promote `CursorState` to a distinct `Cursor` pyclass; KG delegates." Building
that as a separate type with a *full fluent mirror* would mean ~50 hand-written
delegating methods — re-duplicating the exact monolithic surface this
decomposition removes, and forcing a Cursor twin for every future KG method. So
the full mirror is delivered as a snapshot-bound `KnowledgeGraph` (the real
fluent type) via `Session.cursor()`: per-thread, lock-free, zero surface
duplication. A nominal distinct `Cursor` type remains an easy follow-up if a
concrete need for the separate name appears.

---

## Executed plan (shipped 0.11.3, behaviour-identical throughout)

Two design decisions shaped the implementation and are worth preserving:

- **No Mutex on the single-owner path.** The live `KnowledgeGraph` keeps
  `inner: Arc<DirGraph>` + `make_mut` CoW — it is **not** wrapped in the
  `Session`'s `Mutex<Arc<DirGraph>>` (that would add lock overhead to the hot
  path). The decomposition is **cursor extraction + lifecycle grouping**, not
  re-homing storage. The shareable `Session` is the multi-thread variant.
- **Golden-tests-first, mechanical-before-semantic, internal-before-public.**
  Every phase stayed compiler-verified and byte-for-byte behaviour-identical
  (10 characterization tests + full suite green; perf no-regression).

Phase map (one commit each — see git history for detail):

| Phase | What shipped |
|---|---|
| P0 | Characterization/golden tests pinning Clone-vs-copy, selection inheritance, temporal/reports/stats lifecycle, `update()` |
| P1 | `CursorState` struct — grouped `selection`/`temporal_context`/`last_mutation_stats`/`reports` (~170 sites, compiler-verified) |
| P2 | `derive_with` — single funnel for the fluent clone-and-mutate-cursor pattern |
| P3 | `GraphLifecycle` struct — grouped `source_path` + `durable` (the un-shareable identity/WAL fields) |
| P4+P5 | `Session.cursor()` — per-thread, snapshot-bound full-fluent handle (folded; see design note) |
| P6 | Docs (`concurrency.md`) + this roadmap + perf gate |

Result: `KnowledgeGraph` decomposes into labeled concerns — `inner`
(storage) + `cursor: CursorState` + `lifecycle: GraphLifecycle` + `embedder`
+ 2 query-default scalars — instead of 10 flat fields.

## What remains (future, trigger-gated — none currently active)

The decomposition is done. The remaining items are deliberately *not* built;
each is gated on a concrete trigger that has not fired.

1. **Nominal distinct `Cursor` type — CLOSED (satisfied).** The original plan
   imagined promoting `CursorState` to a public `Cursor` pyclass. That is
   delivered functionally by **`Session.cursor()`** (a per-thread snapshot-bound
   `KnowledgeGraph` with the full fluent surface). A *separate* type would be
   either ~50 hand-written delegating methods (re-duplicating the monolithic
   surface this effort removed) or fragile `__getattr__` delegation — net-negative
   for a cosmetic rename. Reopen only if a concrete need for the distinct *name*
   appears, and only with consolidation (e.g. unifying with `FrozenGraph`) so the
   handle-type count doesn't balloon.

2. **Enrich core `Session` as the cross-binding lifecycle home.** Lift the
   binding-agnostic lifecycle coordination (embedder registration, version,
   source-path) into core `Session` so future bindings share it. **Trigger: a
   second non-Rust binding (Go/JVM) exists.** Premature before then — it can't
   even apply to today's `KnowledgeGraph` without the rejected Mutex refactor,
   and `DurableState` owns an OS `File` that stays per-binding regardless. By
   CLAUDE.md's "test the use case" bar, building it now is speculative.

3. **Deprecate any conflated `KnowledgeGraph` surface.** A breaking,
   **major-version** event (no back-compat shims, by project policy). **Trigger:
   a major version is already being cut** for other reasons — amortize the churn
   then, not as a standalone break.

### Trigger summary

- *Second non-Rust binding built* → do item 2.
- *Next major version cut* → consider item 3.
- *Concurrency demand outgrows `Session`* → **already satisfied** by
  `Session.cursor()` (this was the original third trigger; per-thread fluent
  chains against a shared store now work).

Until a trigger fires, `Session` + `Session.cursor()` is the supported
concurrency story and the above stays recorded-but-unscheduled.
