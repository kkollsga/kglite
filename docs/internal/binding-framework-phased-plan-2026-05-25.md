# Binding-framework phased plan — 2026-05-25

After the 2026-05-25 broad scan + binding-framework audit + agent re-scan
against the upgraded North Star (CLAUDE.md, committed in `0e8edad`), this
doc captures the phased plan to execute the findings.

## Status (as of release 0.10.3)

**Plan complete except H.4.** All four phases shipped into the
`release(0.10.3)` commit (5a8e069). H.4 (Go PoC consumer) was
deliberately deferred — the first real non-Rust binding author
validates the surface better than a synthetic 500-LOC sketch.

| Phase | Status | Commits |
|---|---|---|
| **Boundary principle in CLAUDE.md** | ✅ shipped | `0e8edad`, `08c1836` |
| **A.1 — `text_match_regex`** | ✅ shipped | `dcbc198` |
| **A.2 — `shortest_path_length`** | ✅ shipped | `8832eca` |
| **A.3 — `mode(x)` aggregation** | ✅ shipped | `3fa0c0c` |
| **A.4 — `db.*` schema procedures** | ✅ shipped | `22e22c5` |
| **B — `http_status_code` + `json_value_to_kglite_value`** | ✅ shipped | `e4955fa` |
| **H.1 — C ABI design doc** | ✅ shipped | `3fc651a` |
| **H.2 — `kglite-c` skeleton + cbindgen** | ✅ shipped | `61e800a` |
| **H.3 — Sodir + embedder ABI** | ✅ shipped | `b8f75f8` |
| **H.3a — SEC + Wikidata ABI** | ✅ shipped | `61f335e` |
| **H.4 — Go PoC consumer** | ⏸ deferred | — |
| **H.5 — release coordination** | ✅ shipped | `5a8e069` |

The document below is preserved as a historical artifact of the
plan. For the up-to-date status of the C ABI surface, see
[`docs/rust/c-abi.md`](../rust/c-abi.md) and the
[`[0.10.3]` CHANGELOG entry](../../CHANGELOG.md).

---

## Original plan snapshot

(Preserved verbatim from the planning session — outdated as of 0.10.3.)

- **Boundary principle in CLAUDE.md**: expanded with 4 explicit goals,
  two-tier architecture, runtime model, negative-space table,
  worked examples from today's sweep. (`0e8edad`)
- **Already shipped this session**:
  - Batch 1: `parse_with_mutation_check`, `ExecuteOptions::eager`,
    `KgErrorCode::neo4j_status_code` (`4959c3d`)
  - Batch 2: `add_days` / `add_months` / `add_years` /
    `date_truncate` Cypher fns (`271d25c`)

## Plan

Four phases, each shippable independently. Phases A and B are bounded
( <1 week each); Phase C (Phase H — the C ABI crate) is the multi-week
investment that makes non-Rust bindings cheap.

### Phase A — Finish the original broad-scan Cypher batches (~6-7 hours)

Items already designed + use-case-validated. Each is its own commit.

| # | Item | Effort | Notes |
|---|---|---|---|
| A.1 | `text_match_regex(text, pattern[, flags])` Cypher fn | ~2 hrs | Needs `regex` crate (already a dep), pattern cache (`once_cell::sync::Lazy<DashMap>` or similar), case-insensitive flag handling. Tests for common patterns + escape handling. |
| A.2 | `shortest_path_length(a: Node, b: Node)` Cypher fn | ~1-2 hrs | Wraps `graph_algorithms::shortest_path_cost`. Scalar function dispatch + Node param handling. Tests for connected / disconnected / self-loop cases. |
| A.3 | `mode(x)` aggregation Cypher fn | ~1 hr | Most-frequent-value per group. Aggregation registration in `return_clause`. Tests for ties (deterministic tiebreak) + null handling. |
| A.4 | `db.property_stats` / `db.property_uniqueness` / `db.graph_stats` Cypher procedures | ~2 hrs | CALL procedure registration. Schema introspection logic. Tests against a small graph fixture. |

**Why Phase A first**: lowest risk, highest signal-to-noise. Each batch is small, well-scoped, and the use-case test passed. Shipping unblocks them for binding authors via `cypher_query` immediately.

### Phase B — New small lifts from today's agent sweep (~2 hours)

| # | Item | Effort | Notes |
|---|---|---|---|
| B.1 | `KgErrorCode::http_status_code() -> u16` | ~30 min | Match arm per variant: 400 / 404 / 408 / 422 / 500. Tests for round-trip semantics. |
| B.2 | `kglite::api::param::json_value_to_kglite_value(&serde_json::Value)` | ~45 min | Lift mcp-server's private `json_to_value`. Tests for scalar / list / map / null variants. |
| B.3 | Cleanup: mcp-server `push_value_repr` deletion | ~30 min | Use core's `kglite::datatypes::values::format_value` instead. Verify mcp-server tests still pass. |

**Why Phase B second**: small additions that complete the Rust-side
api surface for the next two future bindings types (REST, gRPC). Land
before Phase C so the C ABI in Phase H wraps the full set.

### Phase C — Phase H: the `kglite-c` crate (~1-2 weeks)

This is the genuine "make non-Rust bindings cheap" workstream — the
standardization layer for Go / JS / JVM / .NET bindings. Five
sub-phases as documented in CLAUDE.md:

#### Phase H.1 — C ABI design (~1-2 days)

- Settle wrapping conventions:
  - Opaque-handle pattern for `DirGraph` / `Session` / `Transaction`
    / `KnowledgeGraph` / `Embedder` → `typedef void* kglite_*`
  - Owned-string-out via `*const c_char` + `kglite_free_string`
  - Error pattern: return code + out-param for message string
    (errno-style), with `KgErrorCode` mapping to small ints
  - Async boundary: only `*_blocking` companions exposed; raw async
    skipped in v1 (Go/JS/JVM bindings have their own runtimes)
- Decide on JSON-at-boundary for complex types (Value, CypherResult,
  CypherQuery AST): bindings serialize/deserialize on their side
- Document in `docs/rust/c-abi.md` + `docs/rust/c-abi-conventions.md`

**Deliverable**: design doc + sample C function signatures for the
top 10 entry points (`load_file`, `save_graph`, `execute_read`,
`execute_mut`, `parse_cypher`, etc.).

#### Phase H.2 — `kglite-c` skeleton crate + cbindgen (~2-3 days)

- New workspace member `crates/kglite-c/`. `cargo new --lib`.
- `crate-type = ["cdylib", "staticlib"]` — both shared and static
- Depend on `kglite = { path = "../kglite" }`
- `#[no_mangle] extern "C"` wrappers for the highest-leverage
  entry points (the top 10 from H.1):
  - `kglite_load_file`, `kglite_save_graph`, `kglite_graph_free`
  - `kglite_execute_cypher_read`, `kglite_execute_cypher_mut`
  - `kglite_result_get_columns`, `kglite_result_get_row_count`,
    `kglite_result_get_value`, `kglite_result_free`
  - `kglite_error_get_code`, `kglite_error_get_message`,
    `kglite_error_neo4j_status_code`, `kglite_error_http_status_code`
- `cbindgen.toml` config; `build.rs` runs cbindgen at build time
- CI publishes `include/kglite.h` as a release artifact
- Unit tests via `cbindgen --output-format=test` for header correctness

**Deliverable**: `crates/kglite-c/` builds, produces
`target/release/libkglite_c.{so,dylib,a}` and `include/kglite.h`.

#### Phase H.3 — datasets + embedder C ABI (~2-3 days)

- Wrap dataset fetch entries (using `*_blocking` companions):
  - `kglite_sec_workdir_new`, `kglite_sec_fetch_*_blocking`,
    `kglite_sec_run_all`
  - `kglite_sodir_*_blocking`, `kglite_wikidata_*_blocking`
- Wrap embedder registration:
  - `kglite_embedder_fastembed_new(model_name)` (concrete impl;
    trait objects deferred to v2)
  - `kglite_graph_set_embedder`
- Wrap blueprint:
  - `kglite_blueprint_load_file`, `kglite_blueprint_build`

**Deliverable**: full feature parity with `kglite::api::*` reachable
via C functions (modulo trait objects, which are concrete-impl-only
in v1).

#### Phase H.4 — Go PoC consumer (~3-5 days)

- New repo or workspace dir `crates/kglite-go-poc/` (or external
  `kkollsga/kglite-go`)
- ~500 LOC of Go + cgo over `kglite-c`'s header
- Demonstrates the four binding-author goals:
  - Quick + easy (~500 LOC achieved — target <1500)
  - Standardized (Go binding uses `cypher_query`, gets the same
    surface as Python/MCP/Bolt)
  - Centrally maintained (Go binding pin-bumps `kglite-c`; no
    per-binding source change)
  - Flexible (Go binding doesn't restrict adding new Cypher fns,
    api items, or datasets in core)
- Smoke test: load `.kgl` file → query → save → close

**Deliverable**: Working Go binding + PoC user repo. Validates the
C ABI surface against a real consumer.

#### Phase H.5 — release coordination (~1-2 days)

- Publish `kglite-c` to crates.io alongside `kglite`,
  `kglite-bolt-server`, `kglite-mcp-server`
- Publish `include/kglite.h` as a GitHub release artifact +
  via the `kglite-c` crate's `[package.metadata.docs.rs]`
- Rewrite `docs/rust/implementing-a-binding.md` — replace the
  cgo / napi / JNI *sketches* with worked examples calling the
  real C ABI
- CHANGELOG entries: `kglite-c` 0.1.0 in the next release; flag the
  Phase H landing in CLAUDE.md as `delivered`

**Deliverable**: Phase H is "shipped, documented, has at least one
consumer (Go PoC), and the framework goals are met for non-Rust
bindings."

### Phase D — Future bindings (post-Phase H, ongoing)

Once Phase H lands, future binding work is unblocked:

- **Go binding** (1-2 weeks after PoC). Polish PoC into a published
  package.
- **JavaScript binding** via napi (2-3 weeks). Slightly more involved
  due to V8 type marshalling.
- **JVM binding** via JNI (2-3 weeks). Same shape as Go but with
  Kotlin/Java idioms.
- **.NET binding** (lower priority — only if user demand exists).

Each binding is its own external workstream; the C ABI standardizes
the framework so each one is ~500-1500 LOC of language-native glue.

## Schedule

| Phase | Effort | Cumulative |
|---|---|---|
| A — Finish Cypher batches | ~6-7 hrs | week 1 |
| B — Small lifts + cleanup | ~2 hrs | week 1 |
| C.1 — Phase H design | 1-2 days | week 2 |
| C.2 — `kglite-c` skeleton | 2-3 days | week 2-3 |
| C.3 — datasets + embedder C | 2-3 days | week 3-4 |
| C.4 — Go PoC | 3-5 days | week 4-5 |
| C.5 — release coordination | 1-2 days | week 5 |
| D — Future bindings | open | post-Phase-C |

Total: ~1 week for Phases A+B (small, high-confidence wins); ~3-4 weeks
for Phase C (the strategic unlock). Phases A and B can run in parallel
with C.1 design work; once H.2 starts, A and B should be wrapped.

## Risks + mitigations

- **C ABI lock-in**: the API we expose in `kglite-c` becomes the
  contract every non-Rust binding depends on. Bad signatures are
  expensive to change. Mitigation: H.1 design doc reviewed before
  implementation; first release is 0.1.x explicitly experimental.
- **cbindgen edge cases**: some Rust types don't lower cleanly to C
  (lifetimes, generics, trait objects). Mitigation: H.1 catalogs
  these upfront; v1 restricts to clean shapes; v2 adds the rest.
- **Trait-object embedders**: `Arc<dyn Embedder>` doesn't cross C.
  Mitigation: v1 ships concrete impls only (FastEmbedAdapter);
  user-supplied embedder via function-pointer + context pattern in
  v2.
- **Cross-platform builds**: `libkglite_c.{so,dylib,dll}` across
  Linux / macOS / Windows. Mitigation: existing CI already builds
  `kglite` across these; extend to `kglite-c`.

## Open questions to settle before C.1 starts

- **Crate naming**: `kglite-c` or `kglite-ffi` or `kglite-capi`?
  (Convention varies — polars uses `polars-c`, ringbuf uses
  `ringbuf-c`. `kglite-c` is fine.)
- **Header packaging**: ship `kglite.h` in-tree (committed) or
  generated only at build time? Generated-only is cleaner but
  requires consumers to run `cargo build` to get the header.
  In-tree is easier for non-Rust consumers but risks drift.
- **Embedder v1 surface**: concrete-impl-only (FastEmbedAdapter) or
  add a C function-pointer pattern from day 1? Concrete-impl-only is
  simpler; function-pointer pattern unlocks user-supplied embedders
  (HTTP API, custom local model, etc.). v1 vote: concrete-impl-only;
  user embedder via function pointer is v2.
- **Async boundary**: only `*_blocking` companions exposed in v1, or
  also raw async via thread-spawn? Vote: only blocking. v2 can add a
  callback-based async if a binding needs it.

## Recommendation

Sequence: **A → B → C.1 → C.2 → C.3 → C.4 → C.5**.

A and B are small enough to land this week (~1 day total of focused
work). C is the strategic investment — start C.1 design in week 2
while A and B are still in CI. By end of week 5, the binding
framework is feature-complete and Go PoC validates it.

The four-goal scorecard (CLAUDE.md) becomes the lens for every C ABI
decision: would a Go author writing 500 LOC have a good experience?
Would users switching between Go and Python see consistent UX? Would
a core change reach every binding by pin-bump? Does the C ABI shape
let us add features without breaking?
