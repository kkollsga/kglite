# Consider For Future

A parking lot for work that has been **deliberately deferred** — not
forgotten, not promised, but written down so it doesn't get rediscovered
later as a surprise.

Items here have been explicitly scoped out of current work. Each entry
should answer:

- **What it is** — the concrete change being deferred
- **Why deferred now** — what triggered the de-scope decision
- **When to revisit** — concrete signal that would make this work
- **Estimated effort** — rough size when we do come back

Items get added during in-flight work (as we discover something is bigger
than expected and choose not to grow scope) and removed when either
landed or definitively dropped.

---

## From the 2026-05-25 "prepare kglite for future-language wrappers" sequence

### Full dataset-lifecycle lift to core

- **What it is.** Move all of `SEC.open` / `Sodir.open` /
  `wikidata.open` orchestration (~1356 LOC across
  `kglite/datasets/*/wrapper.py`) into pure-Rust
  `kglite::datasets::*::open()` functions returning
  `Arc<DirGraph>`. Python wheel wrappers become 1-line delegates.
- **Why deferred now.** "Build on what we have, don't redesign"
  guidance (2026-05-25). The lift was originally ~5-6 weeks. The
  evolutionary alternative — expose the fetch/extract/build
  building blocks cleanly via `kglite::api`, let each binding
  write its own short orchestrator in its own idiom — covers the
  binding-portability goal with ~1 week of api work and zero
  Python deletion. Phase 3 now does the smaller version.
- **When to revisit.** If 2+ non-Python bindings independently
  reimplement the same orchestration glue, that's the duplication
  signal; lift then. Or if `cargo install kglite-cli` becomes a
  goal (CLI that opens datasets without Python).
- **Estimated effort.** 5-6 weeks (SEC: 4 weeks, Sodir: 2, Wikidata: 1).
  Per-dataset breakdown in
  `docs/internal/api-audit-2026-05-25.md` (Phase 3 deferred
  framing).

### Retiring the Python MCP server

- **What it is.** Delete `kglite/mcp_server/` (~3500 LOC across 12
  modules) and make `crates/kglite-mcp-server` (the Rust binary
  shipped via `cargo install`) the canonical MCP server for all
  audiences. Wheel users would get the binary bundled or installed
  via a `pip install kglite[mcp]`-time hook.
- **Why deferred now.** Same "build on what we have" framing. The
  Python MCP server works, is used by all 0.9.x and 0.10.x wheel
  users, and has features the Rust binary doesn't yet (skills
  loader, semantic search via fastembed, multi-graph routing). The
  Rust binary is the right canonical for `cargo install` users.
  Keeping both, with a periodic feature-parity audit, beats a
  forced consolidation.
- **When to revisit.** If maintenance cost of keeping both in sync
  becomes evident (e.g. a feature ships in one and lingers
  un-ported in the other for 2+ releases). Or if the Rust binary
  fully absorbs the Python server's features and the duplication
  becomes pure waste.
- **Estimated effort.** ~1 week if dropped wholesale (audit +
  feature copy + Python deletion + skills loader port +
  bundling). Don't do this without clear user demand.

### `from_blueprint` lift to core

- **What it is.** Move `kglite-py/src/graph/pyapi/blueprint.rs`
  (176 lines, `from_blueprint_rust`) into core as
  `kglite::api::blueprint::from_blueprint()`. Wheel wraps with PyO3
  marshalling only.
- **Why deferred now.** The HEAVY logic (load_blueprint_file +
  build) is already in core and is now re-exported through
  `kglite::api::blueprint` (commit 7153080). The wheel's
  `from_blueprint_rust` adds 176 lines of path resolution + verbose
  printing + `lock_schema` application. Bindings can replicate this
  themselves (~30-50 lines of glue per binding) — there's no
  duplication problem yet.
- **When to revisit.** If 2+ bindings need the same path-resolve +
  lock-schema + build pattern. Then it's a real shared utility.
- **Estimated effort.** ~30 minutes (mostly a move, plus making
  `lock_schema` public).

### Ticker resolution lift (SEC)

- **What it is.** `_resolve_companies()` in
  `kglite/datasets/sec/wrapper.py:~440-510` resolves string
  tickers to int CIKs via SEC's `company_tickers.json`. ~70 lines
  of Python, fetches + parses JSON, caches in workdir.
- **Why deferred now.** Bindings that wrap SEC can call SEC's API
  directly with their own HTTP client (Go's net/http, JS's fetch,
  etc.). No need to centralize.
- **When to revisit.** If we're already lifting the full SEC
  lifecycle (see first item above), this gets folded in.
- **Estimated effort.** ~80 lines of Rust + a small JSON struct.

### Process-local cache for Wikidata

- **What it is.** `_PROCESS_CACHE` dict in
  `kglite/datasets/wikidata.py:~30-100`. Caches loaded graphs by
  workdir + entity_limit so Jupyter "rerun cell" returns the same
  instance instead of re-loading the 1.4B-triple dump.
- **Why deferred now.** Genuinely Python-specific (Jupyter
  ergonomics). Each language has its own process-cache
  conventions (Go: `sync.Map`, JS: module-level Map, etc.) —
  centralizing in Rust would be a weird shape.
- **When to revisit.** Probably never. Document as a per-binding
  ergonomics pattern in the binding guide.

### Selection fluent-API lift

- **What it is.** The wheel's `Selection` PyClass exposes
  `select()`, `where()`, `filter()`, `sort()` as a fluent builder.
  Unclear from the audit whether the underlying builder lives in
  core or only in PyO3 code.
- **Why deferred now.** The audit (#2 on the punchlist) called for
  a 2-hour verification + possible lift. Punted because Cypher is
  the canonical query interface for bindings (more portable than
  fluent-API), and bindings can build their own fluent-API on top
  of the Cypher executor if needed.
- **When to revisit.** If a binding author actually asks for it.
  Fluent-API is more idiomatic in Python than in Go/JVM where
  builder-pattern is heavier.
- **Estimated effort.** 2-4 hours verification + 1-3 days lift if
  needed.

### Graph algorithms exposure

- **What it is.** Punchlist item #8: surface shortest path,
  centrality, communities through `kglite::api`. The audit was
  unclear what's actually implemented vs. aspirational.
- **Why deferred now.** Need to know what exists before scoping a
  lift. Verification alone is ~3 hours.
- **When to revisit.** When someone asks for graph algos from a
  non-Python binding, OR when we do the Phase 2 binding guide
  drafting and discover this is a frequently-needed shape.

### Result-streaming verification

- **What it is.** Punchlist item #7: confirm `CypherResult` exposes
  a streaming iterator for bindings dealing with results too large
  to materialize.
- **Why deferred now.** Same — needs verification before scoping.
- **When to revisit.** First time a binding hits an OOM on a
  large Cypher result. Then it's urgent.

### Phase 1 audit top-10 punchlist (items 2-10)

- **What it is.** 9 remaining items on the audit's punchlist
  (selection, reporting, mutation methods, ntriples loader,
  schema definition, streaming, storage backend enum, bulk
  mutation, lock_schema lift).
- **Why deferred now.** "Lazy expansion" — fix each one when
  Phase 2 guide drafting hits it as a real need, not preemptively.
- **When to revisit.** Each one becomes Phase 2's natural
  fix-as-you-go work. Strike from this list as they land.
- **Estimated effort.** ~14 hours total if done all at once;
  realistically spread over a week as Phase 2 surfaces each.

---

## How to use this file

When you're tempted to grow scope mid-stream:

1. Stop, write what you're tempted to do into this file with the
   four bullets (what / why-deferred / when-to-revisit / effort).
2. Continue with the original scope.
3. After the current work lands, re-read this file. If something
   here is the natural next move, promote it to a real task.

When you start a new work-stream:

1. Re-read this file. Anything in here that the new work-stream
   would naturally absorb? Pull it in.
2. Anything in here that the new work-stream would conflict with?
   Resolve up front.
