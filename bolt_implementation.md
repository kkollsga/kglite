# Bolt protocol — implementation plan

> Umbrella plan for [`ROADMAP.md`](ROADMAP.md) §1. The actual Bolt
> implementation decomposes into four discrete phase-loops (A → B →
> C → D) that are planned, implemented, and committed independently.
> Each future plan loop opens by saying *"this is Phase X of
> bolt_implementation.md"* and writes its detail plan against the
> frame here.

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

| Phase | Name | Output | Estimate | Plan-loop boundary | Status |
|---|---|---|---|---|---|
| **A** | Core preparations | Library-level changes that Bolt depends on but also benefit non-Bolt consumers (Value enum, error codes, db.* procedures) | ~2.5–3 weeks total across 3 sub-phases | 3 plan loops (A.1, A.2, A.3) | ✅ Shipped (0.10.0) |
| **B** | Pre-implementation test contract + perf baselines | `crates/kglite-bolt-server/` skeleton, failing `test_bolt_server_smoke.py`, perf baselines re-captured | ~2-3 days | 1 plan loop | ✅ Shipped |
| **C** | Bolt interface implementation | The protocol code itself, in 6 sub-phases each retiring a slice of the failing tests | ~3-4 weeks total across 6 sub-phases | 6 plan loops (C.1–C.6) | C.1 ✅ Shipped · C.2–C.6 pending |
| **D** | End-to-end test program + release | `scripts/bolt_conformance.py` + reference clients in `examples/` + version bump + ROADMAP ✅ Shipped flip | ~1 week | 1 plan loop | Pending |

**Dependency arrows** (must land in this order):

```
A.1 (Value enum) ────┐
A.2 (KgErrorCode) ───┼──→ B (skeleton + failing tests) ──→ C.1 → C.2 → C.3 → C.4 → C.5 → C.6 → D
A.3 (db.* procs)  ───┘                                     (C.4 needs A.1; C.6 needs A.2 + A.3)
```

Total realistic wall-clock with focused work and CI iteration:
**~7-9 weeks**. The ROADMAP §1 estimate said 3-5 weeks; that was
optimistic about Phase A. The library-level elevation work is the
correct thing to do and it adds real time — the per-phase shape
keeps progress visible while the longer total bakes.

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

### C.2 — Read-only RUN / PULL with scalar values

- RUN message: extract Cypher string + (empty) params + (empty)
  metadata. Call `cypher::CypherExecutor::with_params(&inner,
  &param_map, deadline).execute(&parsed)` from
  `src/graph/languages/cypher/executor/mod.rs`.
- Wrap `Arc<DirGraph>` in connection state for concurrent reads
  (per `CLAUDE.md`: read-only sharing is free).
- PULL_ALL: send SUCCESS with field metadata, then one RECORD per
  row, then SUCCESS with summary.
- Scalar `Value` → PackStream mapping per the table at the bottom
  of this doc.

Retires: `test_bolt_run_returns_scalar_rows`. **Estimate ~3-5 days.**

### C.3 — Parameters

- Decode the `$param` PackStream Map on RUN, build the
  `HashMap<String, Value>` that `with_params` expects.
- Edge cases: nested maps/lists (A.1 ensures these round-trip
  cleanly), nulls, large strings.

Retires: `test_bolt_run_supports_parameters`. **Estimate ~2 days.**

### C.4 — Node / Relationship / Path RETURN

**Depends on A.1.** Map `Value::Node { id, labels, properties }` →
PackStream Node struct (signature byte `0x4E`, three fields).
Similarly for Relationship (`0x52`) and Path (`0x50`).

Retires: `test_bolt_return_node_yields_node_struct`,
`test_bolt_return_relationship_yields_rel_struct`. **Estimate ~3-5
days** (much shorter than the original BOLT.md estimate because
A.1 did the heavy lifting).

### C.5 — BEGIN / COMMIT / ROLLBACK + mutations

- Per-connection transaction state machine: `Ready → TxReady →
  TxStreaming → Ready` etc.
- BEGIN: clone the graph snapshot via `DirGraph::begin()`; bind to
  the connection's tx state.
- Write path: single-writer mutex around mutations (may elevate to
  `KnowledgeGraph` per elevation candidate #4 if a second consumer
  is visible by this point).
- COMMIT: `Arc::make_mut` swap. ROLLBACK: drop the working clone.
- `--read-only` CLI flag rejects mutations at server boot with a
  Bolt FAILURE message.

Retires: `test_bolt_transaction_commit_and_rollback`,
`test_bolt_rejects_writes_when_readonly`. **Estimate ~3-5 days.**

### C.6 — Auth + error mapping + db.* pass-through

**Depends on A.2 + A.3.**

- Auth: `"none"` (already in C.1) + `"basic"` (username/password
  validated against CLI args or env var; no persistence).
- Error mapping: lookup table from `KgErrorCode` (from A.2) to
  Neo4j codespace strings (`Neo.ClientError.Statement.SyntaxError`,
  etc.). FAILURE message includes the typed code + the human
  message + position info.
- `db.labels()` / `db.relationshipTypes()` / `db.indexes()` calls
  pass straight through to the core procedures (from A.3) — Bolt
  server is a thin transport.

Retires: `test_bolt_returns_failure_on_parse_error`. **Estimate
~2-3 days.**

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

### Release boundary

- Parent `Cargo.toml` version bump (next minor — likely `0.10.0`).
- `crates/kglite-bolt-server/Cargo.toml` bumps to `0.1.0` (first
  user-facing release).
- Full CHANGELOG `[0.10.0]` block summarising the Bolt work.
- `ROADMAP.md` §1 flipped to ✅ Shipped, sequencing table updated,
  this doc moves into an archive section (or is deleted, since the
  CHANGELOG carries the record).

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
