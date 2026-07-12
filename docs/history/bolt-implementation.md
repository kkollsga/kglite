---
orphan: true
---

# Bolt protocol — implementation plan

> **Historical record** — archived to `docs/history/` on 2026-07-12 (it
> lived at the repo root as `bolt_implementation.md`). Source paths in this
> document reflect the pre-workspace layout (root `src/`); the engine now
> lives at `crates/kglite/src/` and the PyO3 wrapper at `crates/kglite-py/src/`.
>
> Umbrella plan for §1 of the since-retired public `ROADMAP.md` (the roadmap
> went internal in 0.10.14 and the root file was removed). The Bolt
> implementation decomposes into discrete phase-loops (A → B → C →
> robustness → E → D) that are planned, implemented, and committed
> independently. Each future plan loop opens by saying *"this is
> Phase X of bolt_implementation.md"* and writes its detail plan
> against the frame here.
>
> **Status: ✅ SHIPPED.** All phases are complete. Phases A, B, C (all 6
> sub-phases), the robustness pass, Phase E (session abstraction), and
> Phase F (the limitation fixes — OCC enforcement, `neo4j://` routing,
> TLS, Neo4j-aligned `db.*` key naming) landed in the 0.10.1 line. Phase D
> (conformance script, reference examples, docs) shipped in 0.10.14,
> finalizing the feature. The `kglite-bolt-server` binary passes 236
> tests. This document is retained as the design/rationale record
> (referenced from source comments); the release history lives in
> `CHANGELOG.md`.

## Vision

Every Neo4j-aware client — BloodHound, the Neo4j Browser, LangChain's
`Neo4jGraph`, llama-index, every official driver (Python, JS, Java,
Go, .NET) and Cypher Shell — talks to a graph DB via the Bolt binary
protocol. If KGLite speaks Bolt, the entire Neo4j ecosystem plugs in
with zero consumer-side changes. This is the single highest-leverage
move available to the project.

Three architectural decisions are locked from prior planning:

- **Target Bolt v5.x.** Current Neo4j wire protocol; cleaner HELLO /
  LOGON split for re-auth; matches what the only Rust server-side
  Bolt crate (`boltr`) targets. v4.x clients fall back via Bolt's
  version-negotiation handshake.
- **Depend on `boltr` v0.2** (GrafeoDB/boltr). Implements protocol
  framing, message dispatch, session state, and exposes a
  `BoltBackend` trait designed for exactly this embedding pattern.
  ~3-week-old crate; we accept the upstream-churn risk in exchange
  for not hand-rolling the protocol layer. Vendoring is the fallback
  if upstream stalls.
- **Post-execution Node/Relationship materialization.** KGLite's
  `Value` enum has no Node, Relationship, Path, List, or Map variants
  today — `RETURN n` collapses to the node's title string. We fix
  this at the *library* level (Phase A.1), not as a Bolt-server-only
  shim. The library-level fix benefits Python `cypher()`, the MCP
  server's agent surface, and future bindings — not just Bolt.

The "fix it at the library, not the server" pattern is the *meta-
decision*: where Bolt's design pressure points to a superior shape,
the answer goes in the core. The Bolt server is the forcing function;
the wins are everyone's.

---

## Phase summary

| Phase | Name | Output | Estimate (orig) | Actual | Plan-loop boundary | Status |
|---|---|---|---|---|---|---|
| **A** | Core preparations | Library-level changes that Bolt depends on but also benefit non-Bolt consumers (Value enum, error codes, db.* procedures) | ~2.5–3 weeks | shipped in 0.10.0 | 3 plan loops (A.1, A.2, A.3) | ✅ Shipped (0.10.0) |
| **B** | Pre-implementation test contract + perf baselines | `crates/kglite-bolt-server/` skeleton, failing `test_bolt_server_smoke.py`, perf baselines re-captured | ~2-3 days | ~1 day | 1 plan loop | ✅ Shipped |
| **C** | Bolt interface implementation | The protocol code itself, in 6 sub-phases each retiring a slice of the 8 failing tests | ~3-4 weeks | ~6 hours (boltr did the protocol work) | 6 plan loops (C.1–C.6) | ✅ Shipped (8/8 smoke tests pass) |
| **Robustness pass** | Production-grade hardening | Per-tx mutex split, mutex poison recovery, structured error gates, max-message-size, NaN/Inf rejection, string→typed-error heuristic, operator docs, lazy-RETURN bug fix; **242 tests** (was 8) including the 27-query differential corpus over the wire | (un-planned) | ~1 day | 1 plan loop | ✅ Shipped |
| **E** | Session abstraction (standardization) | Extract `kglite::api::session::{Session, Transaction}` as the single canonical query surface; rewrite pyapi + mcp-server + bolt-server to wrap it; prepare foundation for future Go/TypeScript bindings | ~1-2 days, ~8 commits | ~1 day, 6 commits | 1 plan loop | ✅ Shipped |
| **D** | End-to-end test program + release | `scripts/bolt_conformance.py` + reference clients in `examples/` + version bump + ROADMAP ✅ Shipped flip | ~1 week | — | 1 plan loop | Pending |

**Dependency arrows** (must land in this order):

```
A.1 (Value enum) ────┐
A.2 (KgErrorCode) ───┼──→ B ──→ C.1 → C.2 → C.3 → C.4 → C.5 → C.6 → Robustness → E → D
A.3 (db.* procs)  ───┘     (C.4 needs A.1; C.6 needs A.2 + A.3; E touches all 3 downstream
                           consumers and gates D's release cleanliness)
```

Total realistic wall-clock so far: **~2 working days** for the
bolt-server end-to-end (B + C + robustness). Plus the multi-week
Phase A library work shipped earlier in 0.10.0. The original "~7-9
week" estimate badly overcounted the protocol implementation
because the upstream `boltr` crate does the wire framing,
PackStream, session state machine, etc. Most of the visible work
was test design + a few hundred lines of trait method bodies.

---

## Phase A — Core preparations

Three library-level changes that the Bolt scoping work surfaced as
*the right shape*. Each lands in its own commit, has its own
non-Bolt beneficiaries, and gates a later Phase C sub-phase.

### A.1 — `Value::Node` / `Relationship` / `Path` / `List` / `Map` enum variants — ✅ Shipped (0.10.0)

**Why now.** Today's `Value` enum (`src/datatypes/values.rs`) lacks
all five. `RETURN n` collapses to `Value::String(node.title)`;
collections are stringified as JSON. The Python boundary
(`PreProcessedValue` in `src/graph/languages/cypher/py_convert.rs`)
has an inference hack to re-parse JSON strings back into Python
dicts/lists. This is the largest single piece of technical debt in
the engine.

**What changes.**
- Add the 5 variants to `Value` in `src/datatypes/values.rs`.
- Executor's RETURN projection (`src/graph/languages/cypher/executor/`)
  populates them instead of flattening.
- Every `match value` site across the codebase updates (this is the
  bulk of the work — there are dozens in the planner, executor,
  storage layers).
- `py_out::value_to_py` returns proper dicts/lists/Node structs.
- The `.kgl` serialization layer (`src/io/`) handles the new variants
  with a version-gated back-compat path so old files keep loading.
- The `PreProcessedValue` JSON-inference hack at the Python boundary
  is removed (no longer needed).

**Beneficiaries beyond Bolt.**
- Python `cypher()` returns `RETURN n` as `{id, labels, properties}`
  dicts immediately — biggest single agent-UX improvement we can
  ship.
- MCP `cypher_query` tool returns richer node data to agents
  automatically.
- Future Arrow/Polars exporters (ROADMAP §6) work on a sensible
  Value model.
- The differential conformance vs Neo4j (`scripts/cypher_conformance.py`)
  starts converging cleanly because results actually match shape.

**Tests.**
- New `tests/test_value_node_returns.py` — pin the new RETURN shape
  for Node/Rel/Path/List/Map. ~10-15 tests.
- The differential corpus (`tests/test_cypher_differential.py`)
  gains 5-10 queries exercising the new shape.
- Phase 4 parity (the `.kgl` golden digest) updates because
  serialized files now carry richer Value variants.

**Gates.** Phase C.4 (Node/Relationship RETURN over Bolt) cannot
start until this is shipped.

**Estimate.** ~1.5 weeks. The enum addition is half a day; the
match-site sweep is a week; the serialization back-compat is 2 days;
the test pinning is 2 days.

### A.2 — `KgErrorCode` enum + typed Python exception hierarchy — ✅ Shipped (0.10.0)

**Why now.** Today's Cypher errors flatten to `PyErr(msg: String)` at
the boundary. The Bolt server needs typed codes for FAILURE messages
(`Neo.ClientError.Statement.SyntaxError` and friends). Python
consumers grep error message strings to distinguish "your query is
wrong" from "the DB ran out of memory" — fragile and silent.

**What changes.**
- Introduce `KgErrorCode` enum in `src/error.rs` (or new
  `src/graph/languages/cypher/errors.rs`) — variants like
  `SyntaxError { line, col }`, `TypeMismatch { expected, found }`,
  `Timeout`, `ConstraintViolation`, etc.
- Cypher executor and parser return `KgError { code: KgErrorCode,
  message: String, position: Option<Position> }` instead of bare
  strings.
- PyO3 boundary in `src/graph/pyapi/` raises typed subclasses:
  `kglite.CypherSyntaxError`, `CypherTimeoutError`,
  `CypherConstraintError`, etc. — all subclass a common
  `kglite.CypherError`.
- `kglite/__init__.pyi` adds the exception class declarations;
  stubtest pins them.

**Beneficiaries beyond Bolt.**
- Python consumers `except CypherSyntaxError` instead of pattern-
  matching strings.
- MCP server's tool error responses gain structured codes.
- Bolt server's FAILURE messages map directly via a lookup table
  rather than parsing internal English.

**Tests.**
- New `tests/test_error_types.py` — assert each `KgErrorCode`
  variant produces the right typed Python exception with the right
  position info. ~15-20 tests.

**Gates.** Phase C.6 (Bolt FAILURE mapping) cannot start until this
is shipped.

**Estimate.** ~3-5 days.

### A.3 — `CALL db.labels()` / `db.relationshipTypes()` / `db.indexes()` procedures — ✅ Shipped (0.10.0)

**Why now.** These are the canonical Neo4j schema-introspection
procedures. Every Bolt client uses them. KGLite already has
equivalent surface via `describe()` (XML) and the MCP server's
`graph_overview` tool — but they're parallel implementations. Adding
the procedures at the core means there's one source of truth.

**What changes.**
- Implement the three procedures in the existing CALL-procedure
  infrastructure under `src/graph/languages/cypher/procedures/`.
- `describe()` becomes a derived view over the same data (or stays
  as a wrapper that calls the procedures internally).
- Documented in `CYPHER.md`.

**Beneficiaries beyond Bolt.**
- MCP `cypher_query` users get the standard Neo4j discovery surface
  immediately.
- `describe()` and `graph_overview` no longer drift from each other.

**Tests.**
- Differential corpus entries (`tests/test_cypher_differential.py`)
  for each procedure.
- A new `tests/test_db_procedures.py` covering edge cases.

**Gates.** Phase C.6 (Bolt server's `db.*` pass-through) cannot
start until this is shipped.

**Estimate.** ~3-5 days.

---

## Phase B — Pre-implementation test contract + perf baselines

One plan loop. Produces the scaffolding that lets Phase C work test-
driven. **No Bolt protocol code yet** — only the skeleton, the
failing contract, and the perf baselines that gate later regression
detection.

### Crate skeleton — `crates/kglite-bolt-server/`

Mirrors `crates/kglite-mcp-server/` exactly:

```
crates/kglite-bolt-server/
├── Cargo.toml          # deps: kglite (path), boltr 0.2, tokio,
│                       #       clap, tracing, tracing-subscriber, anyhow
├── README.md           # one-pager pointing at this doc
└── src/main.rs         # clap CLI; prints "not yet implemented"; exit 1
```

Workspace `Cargo.toml` adds `"crates/kglite-bolt-server"` to the
members list. CI (`.github/workflows/ci.yml`) adds the new binary to
the existing `cargo build --release -p kglite-mcp-server -p kglite`
line.

### Failing test contract — `tests/test_bolt_server_smoke.py`

Uses the official `neo4j` Python driver (already a `[neo4j]` extra
for the conformance runner). Module-level `pytest.importorskip` +
`skipif(not BINARY.exists())` so the suite cleanly skips in
unsupported environments. Marker `pytest.mark.bolt`, excluded from
the default pytest run (matches the existing `binary_size` marker
pattern).

Eight tests, each mapped to the Phase C sub-phase that retires it:

| # | Test | Retired by |
|---|---|---|
| 1 | `test_bolt_handshake_and_verify_connectivity` | C.1 |
| 2 | `test_bolt_run_returns_scalar_rows` | C.2 |
| 3 | `test_bolt_run_supports_parameters` | C.3 |
| 4 | `test_bolt_return_node_yields_node_struct` | C.4 (needs A.1) |
| 5 | `test_bolt_return_relationship_yields_rel_struct` | C.4 (needs A.1) |
| 6 | `test_bolt_transaction_commit_and_rollback` | C.5 |
| 7 | `test_bolt_rejects_writes_when_readonly` | C.5 |
| 8 | `test_bolt_returns_failure_on_parse_error` | C.6 (needs A.2) |

Fixture: build a small Person+KNOWS graph (mirrors
`tests/test_mcp_server_smoke.py::_build_fixture_graph`), save to
tmp `.kgl`, pass via `--graph`. Spin the binary on an ephemeral
port (`socket.bind((host, 0))`), poll-retry the connect, yield the
`bolt://...` URL, tear down on test exit.

### Perf baselines

Two captures:

1. **Re-capture** the existing 11 tracked benchmarks
   (`tests/benchmarks/test_bench_core.py`) with Phase A landed.
   These become the stable "pre-Bolt" baseline so we can detect
   any regression introduced by the Bolt work touching shared
   executor paths. Save to
   `tests/benchmarks/baselines/<version>_pre_bolt.json` on each
   platform.
2. **Add 2 new benchmarks** specifically covering the new
   `Value::Node` projection path (A.1 added this — we need to be
   sure C.4's Bolt-side consumption doesn't regress it). Cover:
   - `RETURN n` over 10k nodes (eager projection)
   - `MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100` (multi-binding
     projection)

Both new benchmarks land via `make refresh-release-constants` per
the captured-constant refresh discipline in `CLAUDE.md`.

### CI integration

- The binary build line gains `-p kglite-bolt-server`.
- The Python install line in CI gains the `[neo4j]` extra:
  `pip install -e .[mcp,neo4j]`.
- A new pytest step `pytest tests/ -m bolt -v` runs the failing
  contract suite and is **expected to fail** initially —
  documented in CHANGELOG `[Unreleased]` and gated as informational
  (continue-on-error) until C.6 lands.

### CHANGELOG entry

`[Unreleased]` gains an `Internal — Bolt protocol scaffolding`
section explaining the failing-by-design contract.

---

## Phase C — Bolt interface implementation

Six plan loops. Each retires a slice of the 8 failing tests.

### C.1 — Handshake + session lifecycle — ✅ Shipped

**Scope correction from boltr-internals exploration.** The bullets
below described work boltr v0.2.0 already does for us: TCP listener
(`BoltServer::serve` → `tokio::net::TcpListener::bind`), magic
preamble + version negotiation (`server_handshake`), PackStream
framing + message dispatch (`Connection::handle_message`), state
machine (`ConnectionState`: Negotiation → Authentication → Ready
→ Streaming → ...), per-connection task spawn (`tokio::spawn`),
RESET / GOODBYE message handling. We don't write any of that.

**What we actually shipped** (~80-line diff in
`crates/kglite-bolt-server/src/backend.rs`, ~1.5 hours):

- 6 backend method bodies (out of 11): `create_session`,
  `get_server_info`, `set_session_auth`, `close_session`,
  `reset_session`, `configure_session`.
- 1 tightened method: `route` returns
  `BoltError::Protocol("connect with bolt:// not neo4j://")`
  instead of `unimplemented!()`.
- Added `session_counter: AtomicU64` to the `KgliteBackend`
  struct for monotonic `bolt-{N}` session IDs.

`set_session_auth` is currently a debug-log no-op — boltr only
calls it when an `AuthValidator` is wired into the builder, which
is Phase C.6's job.

Retires: `test_bolt_handshake_and_verify_connectivity`. **Actual
time: ~1.5 hours** (the original "~1 week" estimate pre-dated the
boltr-internals exploration that revealed how much of the protocol
the upstream crate already handles).

### C.2 — Read-only RUN / PULL with scalar values — ✅ Shipped

**What shipped** (~150-line diff across backend.rs + value_adapter.rs,
~1.5 hours):

- `execute` body mirrors the canonical kglite Cypher pipeline
  from `kg_core.rs::cypher` (parse → rewrite_text_score → optimize
  → mark_lazy_eligibility → mutation gate → executor) using the
  shared `kglite::api::cypher::*` surface. `.with_streaming(false)`
  forces eager row materialization; the lazy-descriptor path is a
  Phase D perf concern.
- All 10 `Value` scalar variants implemented in `to_bolt`
  (Null/Bool/Int64/UniqueId/Float64/String + recursive List/Map
  + Date/Duration/Point). Graph-structure variants return a
  structured `Err(BoltError::Backend("phase C.4 ..."))` rather than
  panicking (which would orphan tokio tasks).
- Defensive error gates for slices that haven't shipped yet —
  parameters (C.3), explicit transactions (C.5), mutations (C.5),
  text_score queries (D). All map to `Neo.DatabaseError.General.UnknownError`
  on the wire, which the Python driver raises as `DatabaseError`,
  not `ClientError` — so tests #3-#8 stay XFAIL.
- SUCCESS summary metadata: `{ type: "r", t_last: elapsed_ms }`.
  `bookmark` / `stats` / `db` keys omitted (all optional per the
  driver's ResultSummary parser).
- `chrono` becomes a direct dep of the bolt-server crate for
  `Value::DateTime` → `BoltDate` arithmetic (was transitive via
  kglite; making it direct keeps the dep surface honest).

Retires: `test_bolt_run_returns_scalar_rows`. **Actual time:
~1.5 hours** (original "~3-5 days" estimate pre-dated knowing
boltr handles PackStream framing + PULL pagination internally).

### C.3 — Parameters — ✅ Shipped

**What shipped** (~80-line diff in `value_adapter.rs` + 6-line gate
removal in `backend.rs`, ~1 hour):

- `from_bolt` (the inverse of `to_bolt`) is real for all the variant
  classes drivers actually send: scalars, recursive List/Dict,
  Date/Duration/Point2D (SRID 4326 only).
- Non-representable inbound types reject with `BoltError::Protocol`
  (maps to `Neo.ClientError.Request.Invalid` — genuine client errors,
  distinct from C.2's `BoltError::Backend` "feature pending" pattern):
  Bytes, Time/LocalTime/DateTime/etc (kglite has date-only precision
  per A.1 deferral), Point3D, Node/Relationship/Path (drivers
  shouldn't pass these as parameters).
- `execute`'s empty-params gate dropped — parameters flow through
  `from_bolt` into the executor's `&HashMap<String, Value>`.

Retires: `test_bolt_run_supports_parameters`. **Actual time ~1 hour.**

### C.4 — Node / Relationship / Path RETURN — ✅ Shipped

**What shipped** (~140-line diff in `value_adapter.rs`, ~1.5 hours):

- `Value::Node` → `BoltNode { id: i64, labels, properties, element_id }`.
- `Value::Relationship` → `BoltRelationship { id, start_node_id,
  end_node_id, rel_type, properties, element_id, start_element_id,
  end_element_id }`. All `*element_id` fields stringify the
  numeric ids — stable within one server lifetime, which is the
  contract drivers care about.
- `Value::Path` → `BoltPath { nodes, rels: Vec<UnboundRel>, indices }`.
  The `indices` field encodes the Neo4j path scheme: pairs of
  (signed-1-based-rel-index, 0-based-next-node-index) where sign
  is direction (+ outgoing relative to traversal, - incoming).
  Direction inferred by comparing `rel.start_id` / `rel.end_id`
  against the surrounding node ids.
- `kglite::api` gained `pub use NodeValue / RelValue / PathValue`
  so downstream Rust consumers can pattern-match the carriers
  without re-deriving accessors.

Retires: `test_bolt_return_node_yields_node_struct` +
`test_bolt_return_relationship_yields_rel_struct`. **Actual time
~1.5 hours** (the A.1 work and the established `to_bolt` shape made
this much faster than the original "~3-5 days" estimate).

### C.5 — BEGIN / COMMIT / ROLLBACK + `--readonly` — ✅ Shipped

**What shipped** (~500-line diff in `backend.rs`, ~3 hours):

- `KgliteBackend` storage restructured: `Arc<KnowledgeGraph>` →
  `Arc<Mutex<Arc<DirGraph>>>` so commits can swap the inner Arc.
- New `transactions: Arc<Mutex<HashMap<String, TxState>>>` map
  (RA-1 of the robustness pass split this to per-tx mutexes).
- `TxState` mirrors `src/graph/pyapi/transaction.rs` CoW shape:
  `snapshot: Option<Arc<DirGraph>>` + `working: Option<DirGraph>`.
  First mutation materializes working via `Arc::try_unwrap`-or-clone.
- `begin_transaction` rejects under `--readonly`; mints `tx-{N}`
  handle, snapshots the graph, stores.
- `commit` swaps working into shared graph (no-op if no mutations).
- `rollback` drops TxState.
- `close_session` / `reset_session` roll back any in-flight tx for
  the session.
- SUCCESS metadata gains `stats` dict when `MutationStats` are set.
- `--readonly` rejects begin_transaction outright + auto-commit
  mutations with `BoltError::Forbidden` → `Neo.ClientError.Security.Forbidden`.

OCC version checking deferred (`DirGraph::version` is `pub(crate)`;
needs api exposure). Listed as one of the 7 known limitations
([see below](#known-limitations-as-of-shipped-c6--robustness-pass)).

Retires: `test_bolt_transaction_commit_and_rollback`,
`test_bolt_rejects_writes_when_readonly`. **Actual time ~3 hours.**

### C.6 — Auth + typed FAILURE codes + db.* pass-through — ✅ Shipped

**What shipped** (~250 lines across 3 new/modified files, ~2 hours):

- **`crates/kglite-bolt-server/src/error_map.rs`** (new): typed
  `kg_to_bolt(KgError) -> BoltError::Query { code, message }` with
  a 16-arm mapping from `KgErrorCode` to `Neo.{Class}.{Category}.{Title}`
  status codes. The robustness pass added a `string_to_bolt` helper
  for the executor's String-returning paths (RB-3).
- **`crates/kglite-bolt-server/src/auth.rs`** (new):
  `BasicAuthValidator` impl of boltr's `AuthValidator` trait. Checks
  scheme + principal + credentials against `--auth-user` /
  `--auth-pass`. Rejects with `BoltError::Authentication` →
  `Neo.ClientError.Security.Unauthorized`.
- **`crates/kglite-bolt-server/src/main.rs`**: wires the validator
  into `BoltServer::builder().auth(...)` when `--auth basic`.
- **`db.*` procs**: confirmed to work via the standard Cypher CALL
  pipeline — no bolt-server code needed; Phase A.3 routed the procs
  through the executor and Phase C.2's `to_bolt` scalar arms handle
  the result rows directly. **Caveat**: the procs yield `name`, not
  Neo4j's `label` / `relationshipType` — one of the 7 limitations.
- `kglite::api` exposes `{KgError, KgErrorCode}` for downstream
  use (was internal-only).

Retires: `test_bolt_returns_failure_on_parse_error`. **All 8 smoke
tests now PASS.** Actual time ~2 hours.

---

## Robustness pass — ✅ Shipped

After Phase C the bolt-server was contractually complete (8/8 smoke
tests pass) but only happy-path verified. The robustness pass
expanded test coverage from 8 → 242 tests, fixed 1 critical kglite-
core bug + 1 critical concurrency bottleneck, and added 7
hardening fixes informed by broad-probe testing.

**Tests added** (`tests/test_bolt_server_*.py`):

| File | Tests | Coverage |
|---|---|---|
| `test_bolt_server_correctness.py` | 59 | Value roundtrip (every BoltValue variant both directions), error paths (each KgErrorCode → Neo4j code), edge cases (empty/multi-stmt/very-long queries, unicode, NaN/Inf, deeply nested) |
| `test_bolt_server_transactions.py` | 19 | Ports `test_transaction_bolt_patterns.py`'s 18 pyapi contracts to the Bolt wire (snapshot isolation, double-commit error, OCC pin, readonly enforcement, etc.) |
| `test_bolt_server_concurrency.py` | 9 (opt-in via `-m bolt_stress`) | 16 concurrent readers, 8r+1w, 4 concurrent writers, session disconnect mid-PULL, RESET mid-tx, 100 sequential conns, 5s sustained load |
| `test_bolt_server_robustness.py` | 16 | Raw garbage bytes, premature handshake disconnect, zero-byte scanners, null bytes in strings, deep predicate nesting, --help, missing graph, invalid port, --readonly enforcement |
| `test_bolt_server_differential.py` | 124 (3 skipped) | Every entry in `DIFFERENTIAL_QUERIES` runs both via direct `cypher()` AND via Bolt; row sets must match. The strongest correctness gate. |

**Bugs fixed:**

- 🔴 **lazy-RETURN-returns-no-rows** (RA-4): `RETURN x AS y` queries
  WITHOUT ORDER BY returned 0 rows from bolt-server because
  `mark_lazy_eligibility` flagged the RETURN, executor populated
  `result.lazy: Some(LazyResultDescriptor)` and left `result.rows`
  empty. The bolt-server iterated `rows.iter()` → 0 records. The
  ORDER BY-only smoke test (`test_bolt_run_returns_scalar_rows`)
  worked because sort forces materialization. Fix: don't call
  `mark_lazy_eligibility` in the bolt pipeline. Lazy materialization
  helper lives in `src/graph/pyapi/result_view.rs`, isn't exposed
  through `kglite::api` — boltr buffers PULL responses anyway so
  eager materialization is the right shape for the wire.

- 🔴 **per-tx mutex held across entire Cypher pipeline** (RA-1):
  Global `transactions` mutex was acquired in `execute_in_tx` and
  held during parse + plan + execute. One slow query blocked all
  other sessions' tx operations (head-of-line). Fix: split to
  `Arc<Mutex<HashMap<String, Arc<Mutex<TxState>>>>>` — outer mutex
  brief-acquire-only for lookup; per-tx mutex for the actual work.

**Hygiene fixes** (RA-2, RA-3, RB-1, RB-2, RB-3, RB-4):

- Mutex poison recovery: `.lock().unwrap_or_else(|p| p.into_inner())`
- Invariant `expect`s → structured `BoltError::Backend` errors
- `--max-message-size` CLI flag (default 16 MiB)
- Empty / multi-statement query gates → `BoltError::Protocol`
- String error → typed Neo4j code heuristic (timeout / type
  mismatch / constraint / etc.)
- NaN / ±Infinity float parameters → `BoltError::Protocol`

**Operator documentation:**
[`docs/explanation/bolt-server.md`](docs/explanation/bolt-server.md)
ships ~220 lines covering CLI reference, connection URLs, auth
modes, tracing, known limitations, driver compatibility matrix,
common error symptoms, and performance shape from the 6 Bolt-
specific benchmarks in `tests/benchmarks/test_bench_bolt.py`.

---

## Known limitations as of shipped C.6 + robustness pass

After all of the above, the bolt-server has 7 documented
limitations vs a full Neo4j server. Triaged for Phase F (post-E):

| # | Limitation | Triage |
|---|---|---|
| 1 | No OCC version checking on commit (last-writer-wins under concurrent writes) | **Fix in Phase F** (~1 hr) — expose `DirGraph::version` accessor + wire into commit. Real value: prevents silent data loss. |
| 2 | No auto-commit mutations (must wrap in BEGIN/COMMIT) | **Keep** — drivers always wrap writes in BEGIN/COMMIT; supporting auto-commit adds surface for no real win. |
| 3 | Single-graph only (no multi-database) | **Keep** — would require rethinking the backend's data model. |
| 4 | No causal consistency / bookmarks | **Keep** — Neo4j cluster feature; doesn't apply to single-server. |
| 5 | No `neo4j://` routing | **Fix in Phase F** (~2 hr) — return a single-server self-pointing routing table; cluster-aware drivers work. |
| 6 | No TLS | **Optional Phase F** (~30 min) — boltr ships a `tls` feature; wire `--tls-cert` / `--tls-key` flags. Reverse proxy is the alternative. |
| 7 | `db.labels()` / `db.relationshipTypes()` yield `name` not Neo4j's `label` / `relationshipType` | **Fix in Phase F** (~1 hr) — kglite engine change; aligns 3 downstream consumers (Python, MCP, Bolt) with Neo4j convention. |

Total Phase F: ~5 hours for the must-do fixes (#1 + #5 + #7), plus
optional ~30 min for TLS. Lands cleanly AFTER Phase E (which makes
the OCC fix touch fewer files).

---

## Phase E — Session abstraction (standardization) — ✅ Shipped

> Implemented across commits E1–E4 + E6. ~1 day, ~6 commits. The
> below is the original design + a "what shipped" coda; see
> `docs/explanation/session.md` for the binding-implementer guide.

**Why now.** kglite now has two production consumers of the same
Cypher pipeline (Python `cypher()`, Bolt server `execute`) plus a
third near-clone (`kglite-mcp-server::cypher_query`). The pipeline
orchestration — parse → validate_schema → rewrite_text_score →
optimize → mark_lazy_eligibility (or not) → mutation gate →
executor — is duplicated three times. The transaction CoW pattern
is duplicated twice. **This duplication has already cost us twice
in this session**:

1. `validate_schema` was missing from mcp-server + bolt-server until
   user-flagged
2. `mark_lazy_eligibility` was wrongly included in bolt-server,
   causing the lazy-RETURN bug (T4 of the robustness pass surfaced
   it)

Adding future bindings (Go via cgo, TypeScript via napi, etc.)
without fixing this would multiply the drift.

**What changes.** Extract `kglite::api::session::{Session,
Transaction}` as the **single canonical query/tx surface**:

```rust
pub mod session {
    pub struct Session { /* Arc<Mutex<Arc<DirGraph>>> + readonly */ }
    pub struct Transaction { /* snapshot/working CoW */ }

    impl Session {
        pub fn new(dir: DirGraph, readonly: bool) -> Self;
        pub fn snapshot(&self) -> Arc<DirGraph>;

        pub fn execute_read(&self, query: &str, params: &HashMap<String, Value>)
            -> Result<CypherResult, KgError>;

        pub fn begin(&self) -> Result<Transaction, KgError>;
        pub fn execute_in_tx(&self, tx: &mut Transaction, query: &str,
                              params: &HashMap<String, Value>)
            -> Result<CypherResult, KgError>;
        pub fn commit(&self, tx: Transaction) -> Result<CommitOutcome, KgError>;
        pub fn rollback(&self, tx: Transaction);
    }
}
```

Pure-Rust, no PyO3, no async, no transport. The three consumers
become thin wrappers:

- **pyapi `Transaction` class**: PyO3 wrapper around
  `session::Transaction`. Drops ~150 lines of CoW code.
- **bolt-server `KgliteBackend`**: async glue + value_adapter +
  error_map + per-tx mutex (concurrency state for many concurrent
  sessions). Drops the pipeline orchestration entirely (~150 lines).
- **mcp-server `cypher_query`**: tool router + GraphState. Drops
  the pipeline (~50 lines).

**Stays binding-specific** (correctly):

- Wire encoding: PackStream for Bolt, PreProcessedValue for Python,
  JSON for MCP
- Transport: async TCP for Bolt, GIL release for Python, stdio for
  MCP
- Idiomatic error types: PyErr subclass, BoltError variant, JSON
  error object

**Beneficiaries.**

- **Single source of truth** for the pipeline; future drift impossible
- **Single source of truth** for the snapshot/working CoW; OCC
  fix lands in one place
- **Testable in pure Rust** without async or PyO3
- **Future Go / TypeScript bindings** become thin cgo / napi wrappers
  around `Session::execute_*` — the hard part is solved once

**Tests.**

- New `tests/test_session_api.rs` — pure-Rust unit tests for the
  Session surface (~20 tests pinning the contract).
- All ~3000+ Python tests + 242 bolt tests + bolt_stress + bolt
  differential pass unchanged.

**Gates.** Phase F (the 3 limitation fixes) lands cleanly after E.
Phase D (conformance + release) wants E done so the release commits
the standardized shape, not the duplicated one.

**Estimate.** ~1-2 days, ~8 commits. Detail plan goes in a separate
plan loop after this doc-update commit.

### What shipped

| Commit | Subject | Scope |
|---|---|---|
| E1 | `feat(api): introduce kglite::api::session module` | `Session`, `Transaction`, `CommitOutcome`, `ExecuteOptions`, `execute_read`, `execute_mut` + 13 unit tests. Pure addition. |
| E2 | `refactor(pyapi): cypher() + Transaction delegate to session` | `kg_core::cypher` and `Transaction` class become thin wrappers; ~330 lines deleted. |
| E3 | `refactor(mcp-server): cypher_query delegates to session` | `run_cypher_inner` becomes a thin wrapper; ~80 lines deleted. |
| E4 | `refactor(bolt-server): backend delegates to session` | `KgliteBackend` wraps `Arc<Session>`; OCC enforcement enabled — closes limitation #1 of 7. ~325 lines deleted. |
| E6 | `docs(api): session abstraction for binding implementers` | `docs/explanation/session.md`; CHANGELOG entry; this status flip. |

E5 (lift `materialise_lazy_row` from pyapi to session) was
explicitly optional in the plan and deferred — the lift requires
moving `PreProcessedValue` helpers out of pyapi, no current
consumer benefits (bolt-server is eager), and the deletion would
mostly serve future bindings that don't yet exist.

**Final API surface** (`src/lib.rs::api::session`):

```rust
pub use self::execute::{execute_mut, execute_read, ExecuteOptions, ExecuteOutcome};
pub use self::transaction::{CommitOutcome, Session, Transaction};
```

The actual `ExecuteOptions` shape (slightly evolved from the design
sketch — uses `Cow<HashMap<String, Value>>` for params so the
text_score embedder can inject vectors without forcing a clone on
the common case):

```rust
pub struct ExecuteOptions<'a> {
    pub params: Cow<'a, HashMap<String, Value>>,
    pub deadline: Option<Instant>,
    pub max_rows: Option<usize>,
    pub lazy_eligible: bool,
    pub disabled_passes: Option<HashSet<String>>,
    pub embedder: Option<Arc<dyn Embedder>>,
}
```

`CommitOutcome` adds `NoWritesNoOp` for the read-only-or-no-writes
fast path (avoids a needless Arc swap when the tx didn't mutate).

Total: ~440 lines new (session module + ~150 lines docs), ~735
lines deleted (across pyapi + mcp + bolt) — net ~295 line reduction
plus the single-source-of-truth win.

---

## Phase D — End-to-end test program + release

One plan loop. Two artifacts plus the release boundary.

### `scripts/bolt_conformance.py`

Extends the existing `scripts/cypher_conformance.py`. Today that
script runs `tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES`
against KGLite and Neo4j and diffs the row sets — Neo4j is the
oracle. New mode `--target kglite-bolt-server` runs the same corpus
through our Bolt server (started locally on an ephemeral port) and
validates that the results match what direct-Rust `KnowledgeGraph
.cypher()` returns. Catches any wire-encoding round-trip bug the
unit tests miss. Documented in `docs/explanation/cypher-conformance.md`.

Makefile target: `make bolt-conformance`. Not part of CI (same
discipline as `make neo4j-conformance`); on-demand correctness oracle.

### Reference client examples — `examples/bolt_*`

Three artifacts that prove ecosystem compatibility:

- **`examples/bolt_client_neo4j_python.py`** — minimal `neo4j`
  driver session against `kglite-bolt-server`; build a graph, save,
  start server, query.
- **`examples/bolt_client_langchain.py`** — point LangChain's
  `Neo4jGraph` chain at `kglite-bolt-server` and answer a natural-
  language question. Demonstrates the ecosystem unlock.
- **`examples/bolt_neo4j_browser.md`** — walkthrough for pointing
  Neo4j Browser at the server. Mostly just configuration, but
  proves the GUI works.

### Release boundary — what actually happened

The original plan assumed a single `0.11.0` minor that bundled B + C +
robustness + E + F + D. In practice the protocol shipped piecemeal across
the 0.10.x line — Phase E, the C.1–C.6 implementation, and Phase F (TLS,
`neo4j://` routing, `db.*` key naming) all landed in **0.10.1** — so there
was nothing left to bundle. Phase D (this section: conformance script,
reference examples, docs) shipped as a patch, **0.10.14**, which finalizes
the feature. The public roadmap's §1 was flipped to shipped + the section
removed (`ROADMAP.md` itself was later retired when the roadmap went internal);
this doc is retained as the design record. The `kglite-bolt-server` crate
tracks the workspace version (it never reset to `0.1.0` — it was already
publishing on crates.io at 0.10.x).

---

## Value → PackStream mapping table

Reference for Phase C.2 / C.3 / C.4 implementers. Post-A.1 the
table is clean:

| KGLite `Value` | PackStream | Notes |
|---|---|---|
| `Null` | NULL (`0xC0`) | |
| `Boolean(b)` | `0xC2`/`0xC3` | |
| `Int64(n)` / `UniqueId(n)` | INT (sized) | UniqueId cast `u32 → i64` |
| `Float64(f)` | FLOAT (`0xC1`) | IEEE 754 double |
| `String(s)` | STRING (sized) | UTF-8 |
| `List(items)` | LIST (sized) | Recursive |
| `Map(entries)` | MAP (sized) | Recursive |
| `Node { id, labels, properties }` | Struct `0x4E` (3 fields) | id: INT, labels: LIST<STRING>, props: MAP |
| `Relationship { id, start, end, type, properties }` | Struct `0x52` (5 fields) | All INTs except type:STRING + props:MAP |
| `Path { nodes, rels, sequence }` | Struct `0x50` (3 fields) | nodes:LIST<Node>, rels:LIST<UnboundRel>, sequence:LIST<INT> |
| `DateTime(NaiveDate)` | Struct `0x44` (Date) | epoch-days INT |
| `Duration { months, days, seconds }` | Struct `0x45` (Duration) | All INT |
| `Point { lat, lon }` | Struct `0x58` (Point2D) | srid INT + x:FLOAT + y:FLOAT (srid=4326 for lat/lon) |
| `NodeRef(idx)` | **bug if it reaches the boundary** | Should be resolved upstream during projection |

---

## Glossary / external references

- **Bolt protocol spec** — https://neo4j.com/docs/bolt/current/
- **PackStream spec** — https://neo4j.com/docs/bolt/current/packstream/
- **boltr crate** (server-side Bolt v5 in Rust) —
  https://crates.io/crates/boltr · https://github.com/GrafeoDB/boltr
- **packs crate** (PackStream primitives) —
  https://crates.io/crates/packs
- **neo4j Python driver** —
  https://github.com/neo4j/neo4j-python-driver
- **`kglite-mcp-server` crate** (workspace precedent) —
  `crates/kglite-mcp-server/`
- **`tests/test_mcp_server_smoke.py`** (test-harness precedent)
- **`scripts/cypher_conformance.py`** (Phase D extension target)
- **Cypher executor entry** (Phase C.2 consumer):
  `src/graph/languages/cypher/executor/mod.rs::CypherExecutor`
- **`Value` enum** (Phase A.1 target): `src/datatypes/values.rs`
