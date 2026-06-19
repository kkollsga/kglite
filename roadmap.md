# KGLite Architecture Roadmap — sealing `kglite::api` as the single chokepoint

> Created 2026-06-19, replacing the completed `KnowledgeGraph` handle-decomposition
> record (shipped 0.11.3; see git history of this file for that archival content).
> **Status: Piece 1 in progress.** This roadmap breaks the api-sealing effort into
> independently-shippable pieces, ordered by leverage.

## The smell: the Python wheel reaches deep below `kglite::api`

The two-tier binding architecture (see `CLAUDE.md` → "Two-tier standardization
architecture") says every downstream consumer reaches the engine through the
curated, semver-stable `kglite::api::*` surface — never the raw module tree. The
Rust-side servers honour this today: **`kglite-bolt-server`, `kglite-mcp-server`,
and `kglite-c` all go through `kglite::api`** (verified 2026-06-19, zero
`kglite_core::graph::` reaches).

**The Python wheel (`kglite-py`) does not.** Its `graph/mod.rs` opens with

```rust
pub use kglite_core::graph::*;   // pulls the ENTIRE engine graph subtree in
```

and `pyapi/` then leans on that glob via **~344 `crate::graph::…` references**
reaching **~80 distinct engine items** — many of them deep internals, not curated
api:

| Cluster | Examples | Nature |
|---|---|---|
| `core::*` (~52) | `filtering::filter_nodes/sort_nodes`, `traversal::make_traversal/MethodConfig`, `calculations::process_equation`, `pattern_matching::MatchBinding`, `statistics::*`, `data_retrieval::*` | **fluent-API implementation primitives** |
| `algorithms::*` (~36) | `graph_algorithms::shortest_path/CentralityResult`, `vector::DistanceMetric`, `hnsw::HnswParams` | engine algorithms |
| `features::*` (~39) | `timeseries::parse_date_query`, `validate_resolution`, `date_from_ymd` | timeseries internals |
| `mutation::*` / `storage::*` (~52) | `maintain::add_connections`, `set_ops::*`, `subgraph::*`, `storage::backend::GraphBackend`, `interner::InternedKey`, `disk::graph` | mutation + storage internals |

This is not "a handful of public types not yet lifted." The wheel's **fluent API
is implemented across the crate boundary**: `pyapi/` orchestrates fine-grained
engine primitives directly. The root cause is historical — the wheel predates the
api-curation effort (Phase G, 2026-05-24) and was never migrated onto the curated
surface the way the later server crates were.

## Why it matters

- **Stability contract.** Today any refactor of `core::graph::*` internals can
  break the wheel. A sealed api means core internals can move freely behind a
  stable facade — the whole point of the two-tier design.
- **Future bindings.** Every item the wheel reaches below api but *should* be
  public is a capability a future Go/JS/JVM binding can't get without us lifting
  it. Sealing forces those lifts, expanding the shared standard (goal 2:
  "standardized"; goal 3: "centrally maintained").
- **`feedback_kglite_py_python_only`.** The wheel is supposed to hold *only* PyO3
  marshalling + Python type conversion. Engine logic living in `pyapi/` (the
  fluent orchestration) is logic in the wrong crate.

## The strict-posture constraint (why this isn't one big `pub use`)

The naïve fix — lift all ~80 internals into `api` and rewrite all 344 call sites
— is **wrong**. It would bless the engine's entire internal surface as
semver-stable public api, directly violating the DOWNGRADE posture in `CLAUDE.md`
("burden of proof is on *keeping* an item; the test is the *shape*"). The
fine-grained fluent primitives (`filter_nodes`, `make_traversal`,
`process_equation`, …) are not api material — they're the fluent *implementation*,
which belongs consolidated in core, not exposed.

So the effort splits into: **lift what's genuinely generic, consolidate what's
fluent-implementation, then seal.**

---

## Pieces (ordered by leverage)

### Piece 1 — Soft-seal foundation: safe lifts + CI grep freeze · **IN PROGRESS**

**Goal.** Stop the erosion immediately and deliver the clearly-generic lifts, at
near-zero risk to the deeply-coupled wheel.

**Scope.**
- Lift the unambiguously-generic, future-binding-useful below-api items into
  `kglite::api::*` (re-export, never re-wrap — provably zero-cost):
  - `graph::storage::GraphRead` — the canonical read trait.
  - `graph::introspection::reporting::{OperationReport, OperationReports}` —
    mutation operation reports; every binding returns these.
  - `graph::handle::{resolve_code_entity, CODE_TYPES}` — code-tree graph helpers
    (`source_location` + `discover_property_keys_from_data` are already in api).
  - `graph::languages::cypher::planner::all_pass_names` — *verify first*; api
    already re-exports the `planner` module, so this may already be reachable as
    `api::cypher::planner::all_pass_names` (switch the import, no lift needed).
- Switch the wheel's already-in-api reaches (`io::file::load_file`,
  `dir_graph::DirGraph`, `io::file::load_kgl_bytes`, `handle::source_location`,
  `features::timeseries::*`, `make_dir_graph_mut`, …) onto the `api::` path.
- Add `scripts/check_api_chokepoint.sh` (wired into `make lint` / CI):
  freeze the current set of below-api reaches as an allowlist; **fail on any new
  `kglite_core::graph::` reach** in any wrapper crate. This is the regression
  ratchet — the set can only shrink from here.

**Deliberately deferred to later pieces** (not safe lifts):
- `Selection` / `CowSelection` / `PlanStep` — the fluent cursor types. Blessing
  them as stable api is a design decision that belongs with Piece 3.
- `wal` / `recording` / `GraphBackend` / `apply_frames` — durable-transaction
  internals. A future lift should expose a *high-level* durable-transaction api
  (cf. the C ABI's `kglite_save_graph_durable`), not raw WAL frames.

**Effort.** 1–2 phases. **Risk.** Low. **Depends on.** Nothing.

---

### Piece 2 — Lift the generic engine capabilities · **2a/2b/2c DONE**

**Status (2026-06-19).** Phases 2a (graph algorithms), 2b (bulk mutation +
operation-report types), 2c (timeseries helpers + `InternedKey` + GraphRead
migration) shipped. Ratchet baseline **253 → 153**. Remaining Piece-2 candidate
— the storage-backend internals (`storage::backend::GraphBackend`,
`storage::disk::DiskGraph`, `storage::lookups::TypeLookup`) — is **deferred**:
it's entangled with the recording/WAL path and storage-mode construction, so it
needs a storage-mode api-shape decision, not a mechanical lift. Folded into the
Piece 3 design work (or a dedicated storage sub-phase).

**Goal.** Move the below-api reaches that *are* genuine engine capabilities (not
fluent-impl glue) onto the curated surface, shrinking the frozen allowlist.

**Scope (each judged against the use-case + shape test before lifting).**
- `algorithms::graph_algorithms::{shortest_path, shortest_path_weighted,
  shortest_path_cost_weighted, weakly_connected_components, CentralityResult,
  get_node_info, get_path_connections}`, `algorithms::vector::{DistanceMetric,
  vector_search}`, `algorithms::hnsw::HnswParams`.
- High-level `mutation::maintain::{add_nodes, add_connections,
  replace_connections, update_node_properties, …}`, `mutation::set_ops::*`,
  `mutation::subgraph{,_streaming}::*`, `mutation::validation::validate_graph`.
- `storage::interner::InternedKey`, public `storage` types as needed.
- `features::timeseries::*` public config + the date helpers
  (`parse_date_query`, `date_from_ymd`, `validate_resolution`, …).

**Effort.** 2–4 phases, mostly mechanical `pub use` + import rewrites, each
independently green. **Risk.** Low–medium. **Depends on.** Piece 1 (the grep
freeze keeps the rewrite honest).

---

### Piece 3 — Consolidate fluent orchestration into core · **3a/3b/3c DONE**

**Status (2026-06-19).** The architecture map (Explore sweep) found
`core::languages::fluent` is **empty** (doc-comment placeholders only) — the
fluent orchestration lives entirely in `pyapi/` as **31 distinct `core::*`
primitive reaches** (kg_introspection.rs 23, kg_fluent.rs 11), 30–65 % genuine
orchestration per method tangled with PyO3 marshalling. This let Piece 3 split:

- **3a DONE** — the **Selection api-type decision**: `CowSelection` /
  `CurrentSelection` / `PlanStep` / `SelectionLevel` / `SelectionOperation` are
  clean generic core types (petgraph `NodeIndex` + maps, no binding coupling) →
  lifted into `kglite::api` root.
- **3b DONE** — with Selection in api, lifted the generic capabilities that
  merely *take* a selection (not fluent-impl glue): `vector_search` +
  `VectorSearchResult` (api::algorithms), `create_connections` (api::mutation),
  set-ops + subgraph extract/expand/stats (new `api::fluent`),
  `infer_selection_node_type`. Ratchet **153 → 137**.
- **3c DONE — reframed after the architecture map.** The map showed
  `core::languages::fluent` was empty and the wheel's fluent methods are
  `marshal → derive_with → call ONE core::* primitive`: there is **no
  extractable orchestration layer to consolidate** — the `core::*` primitives
  are already the correctly-grained shared operations (CLAUDE.md: "shared query
  primitives … used by both Cypher and the fluent API"), not glue to hide. So
  rather than a pass-through-wrapper rewrite, the shared selection-based
  query-primitive layer (filtering / traversal / calculations / statistics /
  data_retrieval / pattern_matching / value_operations) is **exposed via
  `api::fluent`** — primitives stay *defined* in `core::graph::core`, re-exported
  as their stable binding surface. Pure aliasing → behaviour byte-identical.
  Ratchet **137 → 85**. *Trade-off:* a larger but honest api layer vs. a risky
  rewrite; true high-level-op consolidation remains an optional future refinement
  (the small per-method branching in `select`/`traverse` could be hoisted later).

  **Long-tail cleanup DONE (85 → 27).** Batch 1 migrated the already-in-api
  clusters (session / explore / dir_graph / handle / io::file / blueprint) onto
  api paths. Batch 2 lifted the generic clusters: the schema data-type family
  (`NodeData`/`NodeInfo`/`StringInterner`/`SchemaDefinition`*/configs/
  `ValidationError` + parse helpers), `api::introspection` (compute primitives +
  `bug_report`/`mcp_quickstart`/`debugging`), `api::io` (exporters + N-Triples
  loader), `api::fluent` spatial/temporal predicates, and
  `api::mutation::{validate_graph, add_properties, …}`. All pure aliasing,
  golden digest unchanged.

  **Remaining 27 = the storage-backend + durability + embedding cluster**
  (`GraphBackend` / `DiskGraph` / `MappedGraph` / `EmbeddingStore` /
  `recording` / `wal` / `wal_replay` / `subgraph_streaming` + the embedding-file
  io). These are the storage/WAL *implementation* internals — exposing them raw
  would bless the whole backend + durability mechanism as stable api (wrong per
  the strict posture). They need a **high-level api design** — a
  durable-transaction / storage-mode-open surface that hides them — which is the
  real gateway to Piece 4. This is the one genuine design decision left.

**Goal.** The big architectural piece. Move the fine-grained fluent-orchestration
logic out of `pyapi/` (`core::{filtering,traversal,calculations,statistics,
data_retrieval,pattern_matching}` callers) into core's
`graph::languages::fluent`, exposing **one high-level fluent surface** in
`kglite::api`. The wheel's `pyapi/kg_fluent.rs` is then reduced to thin PyO3
marshalling over that surface.

This is where the `Selection` / `CowSelection` / `PlanStep` stable-api-type
decision is made (the fluent cursor is the thing being lifted). It matches both
`feedback_kglite_py_python_only` (engine logic belongs in core) and the strategic
ROADMAP's "de-emphasize the fluent API as a parallel surface" stance — by giving
it a single owned home rather than a cross-crate split.

**Effort.** Largest — multi-phase, per-method, behaviour-preserving. Golden /
characterization tests first (same discipline as the 0.11.3 handle
decomposition). **Risk.** Medium–high (touches the biggest pyapi files).
**Depends on.** Piece 2 (so only fluent-impl primitives remain below api).

---

### Piece 4 — Hard seal

**Goal.** Make the chokepoint compiler-enforced.

**Scope.**
- Delete the `pub use kglite_core::graph::*` glob from the wheel.
- Demote core `graph` (and any other still-leaking top-level module) from
  `pub mod` to `pub(crate) mod` in `crates/kglite/src/lib.rs`. Because `kglite-py`
  is a separate crate, it then *physically cannot* reach below `api` — the
  `api::` `pub use` re-exports still resolve internally (re-exporting a `pub` item
  out of a `pub(crate)` module is legal), so this is zero runtime cost.
- Flip `check_api_chokepoint.sh` from "no new reaches" to "zero reaches."

**Effort.** 1 phase (the lifts/consolidation in Pieces 1–3 did the real work).
**Risk.** Low at this point (compiler catches everything). **Depends on.** Pieces
1–3 complete.

---

## Sequencing

| Order | Piece | Leverage | Risk | Status |
|---|---|---|---|---|
| 1 | Soft-seal foundation (safe lifts + grep freeze) | High (stops erosion, future-binding value now) | Low | **done (0.11.4)** |
| 2 | Lift generic engine capabilities | Medium (shrinks frozen set) | Low–med | **2a/2b/2c done (253→153); storage-backend internals deferred** |
| 3 | Consolidate fluent into core | High (correct end-state) | Med–high | **3a/3b/3c + long-tail done (153→27); 27 storage/durability remain for Piece 4** |
| 4 | Hard seal (pub(crate) + delete glob) | High (compiler-enforced) | Low | queued |

## Invariants for every piece

- **Re-export, don't re-wrap.** Lift the *same concrete type/function* into
  `api::` via `pub use`. Never introduce a wrapper struct that copies/converts;
  never box a hot type (`DirGraph`, `Selection`, `Value`) behind `dyn` at the
  boundary. `pub use` is a compile-time alias — provably zero perf cost.
- **Judge before lifting.** Each candidate passes the use-case + shape test
  (`CLAUDE.md` → boundary principle). Generic-and-useful → lift; tailored / fluent
  -impl → consolidate (Piece 3), don't bless as api.
- **One commit per phase, each green** (`cargo build --lib` + `make lint` + the
  relevant test suite, incl. `-m parity` and `-m bolt` which `make test`
  deselects). No `Cargo.toml` version bump until a release commit.
- **In-memory perf is the gate** (`CLAUDE.md` → performance protocol).
