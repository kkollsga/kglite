# Changelog

All notable changes to KGLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Cypher queries are interruptible with Ctrl-C.** A long-running `cypher()`
  read — large scans / cross-products *and* `CALL` graph algorithms (pagerank,
  betweenness, louvain, …) — plus **`Session.execute` mutations** can now be
  stopped with `Ctrl-C`, which raises `KeyboardInterrupt` instead of blocking
  until the deadline. A scoped SIGINT handler (installed only while a query
  runs, then restored) flips a cooperative-cancel flag the engine polls at the
  same checkpoints as the query deadline. `Session.execute` mutations are
  **atomic** when cancelled (they run on a copy-on-write working copy that's
  discarded on abort — the graph is fully mutated or unchanged, never partial).
  POSIX only; on other platforms the deadline still bounds queries. Targets the
  interactive single-query case (notebook / REPL). Live `KnowledgeGraph`
  in-place mutations and `Transaction` mutations remain deadline-bounded (not
  Ctrl-C-cancellable) — they mutate in place / don't reliably roll back mid-run,
  so interrupting them could leave partial state; use `Session.execute` for
  cancellable + atomic mutations.

### Internal

- New `kglite::api::session::ExecuteOptions.cancel` (`Option<&AtomicBool>`) —
  the engine-agnostic cancellation primitive bindings flip from their own
  signal model; threaded through the executor and pattern matcher. Servers
  pass `None` (unchanged behaviour). New `KgError::Cancelled` /
  `KgErrorCode::Cancelled` (HTTP 499, `Neo.ClientError.Transaction.Terminated`).
  The graph algorithms now take an `algorithms::Interrupt` (deadline + cancel
  bundle) in place of a bare `deadline: Option<Instant>`, polled at their
  iteration/scan checkpoints so `CALL` procedures are interruptible too.
- GIL-release + error-mapping + cancellation consolidated into one
  `EnterKg::enter_kg` helper in the Python wrapper (replaces scattered
  `py.detach(...).map_err(kg_to_pyerr)` call sites on the Cypher paths).
- **Free-threading (no-GIL / 3.13t) readiness.** The `kglite` extension
  module now declares `gil_used = false`, and the shareable read pyclasses
  (`Session`, `FrozenGraph`, the Cypher `ResultView`) are `#[pyclass(frozen)]`
  — immutable + `Sync`, removing the runtime borrow-flag and matching how the
  concurrent `Session` path already shares state. No API change.

## [0.11.5] — 2026-06-20 — `kglite::api` hard-seal + dataset surface curation + Cypher plan cache

### Changed

- **`kglite::graph` is now `pub(crate)` — the engine is reachable only through
  the curated `kglite::api` facade** (roadmap Piece 4 completed the 253→0
  below-api-reach sweep; the `api` surface was also reorganized into
  one-home-per-concern clusters). The Python wheel, the bolt/mcp/C servers,
  and the Cypher / `kglite::api` surfaces are unaffected. *Potentially
  breaking only for external Rust consumers of the `kglite` engine crate that
  reached `kglite::graph::*` directly — move those to `kglite::api::*`.* A CI
  grep plus the `pub(crate)` compile boundary keep wrappers honest.
- **`kglite::api::datasets` slimmed ~65 → 38 items.** The dataset module is
  now sealed behind `api::datasets` (single, gate-enforced path, the same
  treatment as `graph`); the per-function `*_blocking` twins collapsed to one
  `kglite::api::datasets::block_on` bridge; the dataset surface was curated to
  the items bindings actually consume; and ~530 lines of dead code the seal
  unmasked were removed. The surface every binding actually uses is unchanged.
- **Single mode-aware durable save dispatch** (`kglite::api::io::save_graph_with`)
  now backs the wheel, the MCP server, and the C ABI, replacing three copies
  of the disk-vs-in-memory / columnar / fsync logic. Fixes the C
  `kglite_save_graph_durable`, which previously bypassed disk-mode dispatch
  and columnar consolidation (and whose fsync docs were inverted). Saves are
  byte-identical and remain durable (fsync) by default.

### Added

- **Mapped / disk storage mode reaches every binding.** Creating a graph in
  a specific backend (`memory` / `mapped` / `disk`) is now available across
  all wrappers through one shared core builder
  (`kglite::api::storage::StorageMode` + `new_dir_graph_in_mode`), so the
  mode vocabulary can't drift:
  - **C ABI:** new `kglite_graph_new_in_mode(mode, path, …)` — non-Rust
    bindings can create mapped/disk graphs, not just in-memory ones.
  - **bolt + mcp servers:** new `--storage memory|mapped|disk` flag. An
    existing `--graph` (a `.kgl` file or disk-graph directory) is loaded in
    its saved mode (auto-detected); a `--graph` path that does *not* exist
    errors by default (typo guard) and is created fresh **only when
    `--storage` is given** (opt-in build-and-serve).

### Performance

- **Cypher plan cache.** A param-less, codec-free query re-run against an
  unchanged graph now reuses its fully-optimized plan, skipping parse +
  schema-validate + optimize (the parse cache already covered parse; the
  optimizer was the bigger uncached cost). Keyed on `(graph_id, version)` so
  it is invalidated by any mutation and never leaks across graphs; parameter
  binding still happens fresh at execute time. Biggest win for repeated
  queries against a stable/served graph (bolt/mcp). To make the key sound,
  `DirGraph` version now bumps on every mutation path (Cypher writes via
  `execute_mut`, bulk ingest, and `make_dir_graph_mut`), not only on handle
  acquisition.

## [0.11.4] — 2026-06-19 — C ABI completeness + `kglite::api` soft-seal foundation

### Added

- **C ABI surface completed (`kglite-c`): 32 → 45 `extern "C"` functions.** The
  C ABI (the entry point for every non-Rust binding — Go/cgo, JS/napi, JVM/JNI,
  …) now covers the full lifecycle, not just query. New entry points:
  - `kglite_graph_new` — create an empty in-memory graph (previously the C ABI
    could only load a graph from a file).
  - `kglite_session_execute_read_batch` / `kglite_session_execute_mut_batch` —
    run a batch of queries against one snapshot / inside one transaction (the
    mut batch is atomic).
  - `kglite_session_execute_read_opts` — read with a timeout + max-rows guard
    (max-rows *errors* when exceeded, it does not truncate).
  - `kglite_create_edges_batch` — DataFrame-free bulk edge ingest by stable
    id + type (wraps the new core `add_edges_from_specs`).
  - `kglite_graphgen_to_dir` — synthetic-graph generator.
  - `kglite_blueprint_build` — declarative graph construction from a blueprint.
  - `kglite_save_graph_durable` (fsync) + `kglite_graph_to_bytes` /
    `kglite_graph_from_bytes` / `kglite_free_bytes` — durable save +
    in-memory bytes round-trip.
  - `kglite_compute_schema_json` — schema introspection at the ABI boundary.
  - `kglite_memory_stats` — backed by a tracking global allocator.
- **Core: `add_edges_from_specs` — DataFrame-free bulk edge ingest** (exposed
  via `kglite::api::mutation`, reusing the same engine as the Python
  `add_connections` DataFrame path). The one genuine library gap that the C ABI
  needed; available to every Rust-side binding too.
- **`kglite::api` surface expanded** (api-sealing roadmap Piece 1): `GraphRead`
  (the canonical read trait), `OperationReport` / `OperationReports` (structured
  mutation reports), `resolve_code_entity` + `CODE_TYPES` (code-tree graph
  helpers) are now reachable through the curated `kglite::api` namespace for
  downstream and future bindings. Zero-cost `pub use` re-exports — no behaviour
  or perf change.

### Changed

- **C ABI result rows are now natural untagged JSON.** A scalar comes back as
  `{"n": 2}` instead of the enum-tagged `{"n": {"Int64": 2}}`. The shared
  `kglite_value_to_json` converter was lifted into `kglite::api::param` so every
  binding (and the MCP server) emits the same shape.
- **`kglite_abi_version` now derives from the crate version** (was hard-coded
  and stale at `0.10.5`).

### Fixed

- **JSON array / object query parameters were stringified instead of converted
  to `Value::List` / `Value::Map`.** A live data-corruption bug:
  `UNWIND $rows AS r CREATE {id: r.id}` wrote null ids (an unmatchable graph),
  so subsequent `SET` / `DELETE` silently no-oped. Same class as the 0.11.2 PyO3
  fix, but in the *shared* `kglite::param` converter
  (`json_value_to_kglite_value`) used by the C ABI, the MCP server, and every
  future binding — the PyO3 path was already fixed in 0.11.2; this fixes
  everyone else.

### Packaging

- **aarch64 Linux (gnu) wheel now builds on `manylinux_2_28`** instead of the
  ancient `manylinux2014` cross image. The 2014 cross gcc (4.8.5) could not
  cross-build the wheel's C deps for aarch64 — it failed on `ring`'s `.S` asm
  (fixed in 0.11.1) and then on `libmimalloc-sys`'s `-Wno-error=date-time`
  (gcc <4.9 has no `-Wdate-time`). Building on gcc 12 clears the whole class.
  Trade-off: the gnu-aarch64 wheel's glibc floor rises 2.17 → 2.28
  (RHEL 8 / Ubuntu 18.10+); the musllinux aarch64 and x86_64 wheels are
  unchanged.

## [0.11.3] — 2026-06-18 — thread-safe `Session` handle (shared reads + serialized writes)

### Added

- **`KnowledgeGraph.session()` → `Session` — thread-safe, shareable graph
  handle.** A live `KnowledgeGraph` is single-owner: sharing one across a
  thread pool and mutating it concurrently trips a borrow guard (the failure
  mode that forces server consumers to wrap every call in a global lock).
  `Session` is the fix — it wraps the engine's `Mutex<Arc<DirGraph>>` and
  exposes only `&self` methods, so it can be shared across threads:
  concurrent `cypher()` reads take a momentary snapshot and run lock-free,
  while `execute()` writes serialise behind a writer lock held across
  `begin → mutate → commit` (copy-on-write working copy + atomic swap). The
  writer lock makes concurrent writes **compose** — each `execute()` begins
  from the prior writer's committed state, so increments and read-modify-write
  updates don't clobber each other (the lost-update failure mode that forces
  naive shared-handle consumers to wrap every call in a global lock).
  `snapshot()` hands out a stable `FrozenGraph` for held multi-query views;
  `version()` exposes the monotonic commit counter. Build or load with a
  `KnowledgeGraph`, then `.session()` and serve every thread through the
  `Session`.
- **`Session.cursor()` — per-thread fluent query handle.** Returns a
  `KnowledgeGraph` bound to a snapshot of the session's current state with a
  fresh cursor. Where `snapshot()` gives a read-only `FrozenGraph` (just
  `cypher()`), `cursor()` gives the **full fluent surface**
  (`select`/`where`/`sort`/`traverse`/`to_df`/…) as an independent
  single-owner handle, so N threads can each take a cursor off one shared
  `Session` and run fluent chains in parallel, lock-free. Mutations on a
  cursor are copy-on-write isolated (they don't write back to the session).
  Part of the `KnowledgeGraph` internal decomposition (storage / cursor /
  lifecycle now separated into `CursorState` + `GraphLifecycle`; see
  `roadmap.md`).

- **`kglite.open_session(path)` — one-call shared handle.** Loads a saved
  graph directly as a thread-safe `Session` (equivalent to
  `kglite.load(path).session()`), so the concurrent-serving path is as easy to
  reach as the single-owner one. Paired with a clearer single-owner error:
  when a `KnowledgeGraph` is shared+mutated across threads, the `RuntimeError`
  now names the fix (`session()` / `freeze()` / `cursor()`) instead of only
  suggesting `copy()`/a lock.

### Fixed

- **Core `Session::commit` TOCTOU race (concurrent committers could lose a
  commit or move the version backwards).** The optimistic-concurrency version
  check read the graph version under one lock acquisition and then swapped the
  graph under a *separate* one, so two threads committing at once could both
  pass the check and both swap — losing one commit, and (because the new
  version was derived from the transaction's possibly-stale base) leaving the
  monotonic version counter non-monotonic. The check and swap now happen under
  a single lock guard, and the version bumps from the *current* value, so
  commits are atomic and the version is monotonic even in last-writer-wins
  mode. Affects the **bolt-server** (which drives the core `Session` from many
  connection threads with no serializing lock); the Python `Session` was
  unaffected (its writer lock already serialized committers). Found by new
  true-parallel Rust concurrency tests + an opt-in Python stress harness
  (`-m stress`).

## [0.11.2] — 2026-06-18 — bundled synthetic-graph generator + public benchmark

### Added

- **`kglite.graphgen()` — bundled synthetic-graph generator.** Generate a
  seed-deterministic org/social knowledge graph (Person/Company/Project/Skill/
  City + 7 edge types) in one call — for demos, tests, and benchmarks, with no
  extra dependency or Rust toolchain (it's compiled into the wheel, like
  `code_tree`). `kglite.graphgen("medium")` returns a ready-to-query
  `KnowledgeGraph`; `kglite.graphgen("huge", out=DIR)` streams one CSV per type
  + a `manifest.json` in **bounded memory** (millions of nodes at flat RAM), so
  any engine that reads the same bytes gets the same graph. Scales
  `tiny`…`xhuge` (or an exact `persons=`), `degree_dist='zipf'` for realistic
  high-degree hubs. The generator moved from the standalone `benchmarks/graphgen`
  crate into `crates/kglite/src/graphgen/` (core) and is re-exported from
  `kglite::api` for other bindings. Nodes now also carry **geometry** (City
  `latitude`/`longitude`) and a per-Person **embedding vector** (`embedding_dim`
  in the manifest), so the generated graph exercises geospatial and
  vector-search workloads out of the box.

### Fixed

- **Cypher `UNWIND $list AS i MATCH (n {id:i})` silently returned no rows once
  the list exceeded ~64 elements.** A query-local equality index — meant for
  cross-MATCH joins on *stored* properties — was also being built over the `id`
  node-identity virtual (which resolves to identity, not a stored column),
  yielding an empty map so every probe missed and the bare point-MATCH dropped
  all rows above the index's activation threshold. The index now skips the
  `id`/`title` virtuals (identity has its own fast seek path), so batched
  id-lookups via `UNWIND` are correct at any list size. Found via the new
  cross-engine benchmark parity check.
- **Subgraph-scoped community detection now works on `mapped`/`disk` graphs.**
  `CALL louvain/leiden/label_propagation({node_type, relationship})` previously
  errored on disk/mapped storage ("scoping is in-memory-only"), even though the
  scoped subgraph is bounded and `connected_components` scoping already worked
  there. Scoped runs now route through the materialised (storage-agnostic)
  adjacency path on every mode — identical results across memory/mapped/disk —
  while unscoped whole-graph runs keep the bounded-memory streaming path.
- **Python `dict` and list-of-`dict` Cypher params now marshal to native maps
  and lists instead of `null`.** The PyO3 param converter had no `dict` branch
  (a dict param became `Value::Null`, so `$m.prop` and `UNWIND $rows AS r …
  r.key` returned null) and flattened lists into a JSON *string*. The common
  batch shape `UNWIND $rows AS r CREATE (:T {id: r.id, …})` therefore wrote
  nodes with null ids — unmatchable, so a following `SET`/`DELETE` silently
  no-oped and the in-memory/mapped graph diverged from disk (phantom rows, a
  duplicate-id warning). Params now convert recursively to `Value::Map` /
  `Value::List`; `vector_score`/`UNWIND`/`IN` over a list are unaffected
  (`extract_float_list` already accepts the native list). Found via the
  cross-storage-mode mutation-parity benchmark; storage modes are now
  byte-identical on the mutation suite.

### Benchmarks / docs

- **A linkable, reproducible benchmark table — [`BENCHMARKS.md`](BENCHMARKS.md).**
  Wall-to-wall time per category — **26 sub-benchmarks across 9 categories**:
  scan & lookup / filter & aggregate / traversal / pathfinding / **multi-type
  queries** / graph algorithms / **community detection** / mutations, plus the
  KG-specialized **vector search** and **geospatial** tiers — for kglite vs Kùzu,
  Neo4j, NetworkX, rustworkx, igraph, and DuckDB on one shared synthetic graph
  every engine loads from identical bytes. A capability matrix ("can it do your
  workload?") makes the breadth picture instant — **kglite is the only engine
  covering all 10 categories** (and the only one with vector search), while a
  `—` shows a real gap (DuckDB has no pathfinding/WCC/community/vector; the
  algorithm libraries have no query language for the multi-type joins). A
  *partial* category is percentile-estimated so a within-category skip can't
  flatter a total, and a sub-bench slower than 10 s is marked `⏱` and excluded
  (so one ~11 s pure-Python Louvain can't dominate a sum). Regenerate with
  **`python benchmarks/benchmark.py`** — it stages the dataset with the bundled
  `kglite.graphgen` (no Rust needed) and runs every installed backend; a
  cross-storage-mode result-parity check (`tests/test_benchmark_parity.py`) gates
  every kglite mode to identical results. The public `graphsuite` comparison is
  tracked in the repo; one-off dev scripts moved to `tests/benchmarks/internal/`
  (the perf *gates* stay in `tests/benchmarks/`).
- **Opt-in server backends for the comparison.** Heavy, externally-provisioned
  backends are requestable via `--libs` and skip cleanly when their prerequisite
  is absent: **Neo4j** in two deploy flavors — an auto-managed native server
  (`neo4j-native`, higher-performance) and a `neo4j:5-community` container
  (`neo4j-docker`) — plus **kglite served over Bolt from a container**
  (`kglite-bolt-docker`), backed by a new `crates/kglite-bolt-server/Dockerfile`
  so the Bolt server is one `docker build` away.

## [0.11.1] — 2026-06-17 — HNSW in Cypher, faster index build, embedding-provenance papercuts

Follow-up to 0.11.0, driven by the mcp-servers operator's independent
validation (exact-search parity + 1.83× speedup, near-linear concurrent-read
scaling, HNSW 0.997 recall @ 5.6× on a real 46k×1024 store — 15/16 reported
painpoints verified resolved). This closes the remaining items.

### Added

- **Cypher `vector_score()` / `text_score()` top-k now auto-uses the HNSW
  index.** A whole-corpus `RETURN vector_score(n, prop, q) AS s ORDER BY s DESC
  LIMIT k` (and the `text_score` form) dispatches through a built index instead
  of scoring every row — so agent/MCP semantic search done via Cypher benefits
  too. Opt-in (only fires when `build_vector_index` was called), re-scores
  survivors with the exact `Scorer` (identical score scale), and falls back to
  the exact scan for any shape it can't faithfully serve (ASC order,
  mixed/unbound types, duplicate node bindings, Poincaré, dimension mismatch, or
  a selective `WHERE` whose survivors underfill the limit). Independently
  validated at recall@10 0.994 with exact score parity on a real 46k×1024 store.
  The end-to-end Cypher speedup is more modest than the fluent API's (~2.3× at
  46k vs ~5.6×): Cypher's fixed per-query cost (parse + plan + projection) is a
  larger share of the total at this corpus size, so the index saving shows
  through less — the gap widens on larger corpora where the scan dominates.

### Changed

- **`embedding_info()` / `list_embeddings()` report the *effective* metric.** A
  store created by `embed_texts` (which sets no explicit metric) used to report
  `metric: None` even though search applies cosine. Both methods now report the
  metric search actually uses — the explicit one if set, else `'cosine'` — and
  never `None` for an existing store. Pure reporting; no stored-data or format
  change.
- **`.kgle` export/import carries embedding provenance (format v2).**
  `export_embeddings` / `import_embeddings` now round-trip each store's `metric`
  + embedder `model_id` + per-node text hashes, so a rebuild-from-`.kgle`
  pipeline keeps provenance and `embed_texts(mode='changed')` re-embeds only
  changed text instead of everything. Older v1 `.kgle` files still import (they
  carry no provenance — `mode='changed'` treats every node as new).

### Performance

- **HNSW index build is ~5–6× faster (concurrent).** Build was single-threaded
  (~43s on a 46k×1024 store — the new engine's one rough edge). Inserts now run
  on rayon: the vectors are immutable during a build, so each insert reads the
  growing graph through per-node `RwLock` read locks and writes only its own +
  its neighbours' link lists (one lock at a time → deadlock-free). Measured on
  10 cores: 10k×128 1.9s→0.3s (6.3×), 50k×128 16s→2.8s (5.7×), 100k×256 77s→14s
  (5.4×). Recall and query latency are unchanged. The seeded level assignment
  stays deterministic; the link graph now differs run-to-run (recall is
  statistically equivalent — the index is a rebuildable cache).

### Documentation

- semantic-search guide: the Cypher index path, `.kgle` provenance, and a
  "benchmark HNSW on *real* embeddings, not random vectors" note (random
  high-dim vectors have no neighbourhood structure → any ANN looks bad; ~0.99 on
  real data). Concurrency guide: a `freeze()` fan-out scaling note (near-linear
  for CPU-bound queries, sub-linear for bandwidth-bound full scans).

### CI

- Wheel builds now run in parallel with CI (publish still gated on CI passing) —
  cuts the release pipeline wall-clock roughly in half — and the
  aarch64-unknown-linux-gnu (manylinux2014) wheel build is fixed (`ring`'s ARM
  asm needs `__ARM_ARCH` defined in that cross image).

## [0.11.0] — 2026-06-17 — concurrency snapshots, durable save, embedding provenance, edge upsert, portable wheels

Cut as a **minor** (0.11.0), not a patch: this release adds new public API
(`freeze()`/`FrozenGraph`, `to_bytes()`/`from_bytes()`, `replace_connections()`,
`embedding_info()`/`embedding_dim()`, `embed_texts(mode=…)`, `copy_embeddings_from()`,
`search_text`/`vector_search` `returning=`, public `build_code_tree`), changes a
default (`save()` is now atomic + `fsync`), and makes a scoped on-disk format
break (embeddings section — core-data-version 3). See **Migration** below.

Folds three operator-feedback rounds (2026-06-17): graph-engine limitations &
Cypher footguns (edge upsert, algorithm-config robustness, portable wheels); a
concurrency/durability/embedding roadmap (freeze snapshot, durable save,
embedding provenance + incremental re-embed); and the way-forward shortlist
(search-hit projection, cross-graph vector carry, public code-tree API, typed
load error).

### Migration (0.10.x → 0.11.0)

- **`save()` is atomic + `fsync` by default.** No code change needed; you get
  crash-safety for free. If you do high-frequency saves where durability isn't
  required, pass `save(path, fsync=False)` (still atomic, just no flush).
- **`.kgl` with embeddings from an older binary won't load** (embeddings section
  format changed — core-data-version 3). The graph's nodes/edges/columns are
  unaffected; only the (rebuildable) vector cache broke. **Action:** reload the
  graph, re-run `embed_texts()` / `add_embeddings()`, and `save()` again — or use
  `new.copy_embeddings_from(old)` once both are on 0.11.0. A `.kgl` *without*
  embeddings loads unchanged.
- **`load()` / `from_bytes()` raise `kglite.FileFormatError` (not `IOError`) on a
  corrupt file.** Code with `except IOError:` around a load should catch
  `kglite.FileError` / `kglite.FileFormatError` (both subclass `kglite.KgError`).
- **Sharing one graph across threads** raises a clear `RuntimeError` instead of
  panicking. Give each worker its own `copy()`, serialize access, or share a
  read-only `freeze()` snapshot for concurrent reads.
- **MCP/binary consumers:** rebuild/republish `kglite-mcp-server` against 0.11.0
  (the format bump means an old binary can't read a 0.11.0 `.kgl`).

Concurrency / durability / embeddings, at a glance:

- **Concurrency** — a clear cross-thread error (no more borrow panics) and
  `freeze()` → `FrozenGraph`, an immutable O(1) snapshot with lock-free
  concurrent reads (the thread-safety item, addressed via the "build → freeze
  → share → swap" model rather than a global lock).
- **Durability** — atomic + `fsync` save (no torn `.kgl`), `to_bytes()` /
  `from_bytes()`, typed `FileFormatError` on corrupt load.
- **Embeddings** — model + text-hash provenance, `embed_texts(mode='changed')`
  incremental re-embedding, `embedding_info()`, `copy_embeddings_from()`,
  `search_text`/`vector_search` `returning=` projection.

### Added

- **`build_vector_index()` — opt-in HNSW index for scalable vector search.**
  Brute-force vector search is exact but O(n·d) per query; on large stores that
  doesn't scale. `g.build_vector_index(node_type, text_column, m=…,
  ef_construction=…, ef_search=…, metric=…)` builds a hand-rolled HNSW
  approximate-nearest-neighbour index (cosine / dot-product / Euclidean —
  Poincaré stays exact). Opt-in like `create_index`: once built,
  `vector_search` / `search_text` **auto-use** it for whole-corpus queries on
  large stores (≥256 candidates); pass `exact=True` to force the exact scan.
  Heavily-filtered selections fall back to exact automatically (correctness over
  speed when a filter is selective). The index is **dropped automatically**
  whenever the store's vectors change (`add_embeddings` / `embed_texts`) or slots
  are remapped (`vacuum`) — rebuild it afterward. Companion methods
  `drop_vector_index()` and `has_vector_index()`. The index **persists in the
  `.kgl`** (and `to_bytes()`): it rides in a dedicated, self-describing,
  *skippable* section (own magic + format version), so it's restored on load
  with byte-identical search results, the on-disk index format can evolve
  without a core-data-version bump, and an unrecognised/corrupt index is silently
  dropped (it's a rebuildable cache, never a correctness dependency). *(The
  Cypher `vector_score()` / `text_score()` path still uses the exact scan — a
  follow-up.)*
- **Public `code_tree` build API.** Code-graph building now has a stable public
  entry point — top-level `kglite.build_code_tree(path, …)` and
  `kglite.code_tree.build` — and `kglite._kglite_code_tree` is documented as an
  internal implementation detail (consumers were importing from the
  underscore-prefixed module directly). The top-level `from_bytes`,
  `build_code_tree`, and `FrozenGraph` are now advertised in `kglite.__all__`
  (operator note #3).
- **`copy_embeddings_from(other)`** — one-call, id-keyed cross-graph vector
  carry. The dominant embedding workflow rebuilds a *fresh* graph from a source
  of truth on each load; `embed_texts(mode='changed')` can't help an empty fresh
  graph, so vectors had to be hand-carried (`embeddings()` snapshot →
  `add_embeddings()` → `embed_texts`). Now: build the new graph, then
  `new.copy_embeddings_from(old)` — vectors land on the nodes that share an id,
  carrying dimension, metric, model id, and per-node text hashes (so a following
  `embed_texts(mode='changed')` re-embeds only genuinely new/changed text).
  Implemented in core (`DirGraph::copy_embeddings_from`), so every binding
  reaches it (operator embedding note #2).
- **`search_text` / `vector_search` gain a `returning=[...]` field projection.**
  By default a hit already carries `id`, `title`, `type`, `score`, **and every
  node property** (read live — identical before/after save/reload, so no
  follow-up `MATCH … WHERE id IN […]` hydrate is needed). `returning=['title']`
  trims a hit to `id` + `score` + the named fields, for ranking-heavy or
  wide-node workloads. Documents the default hit contract (operator note: search
  hits + harvest N1).
- **`embed_texts(mode='changed')` + per-node text-hash + model provenance.**
  `embed_texts` now records, per node, a content hash of the embedded text and
  (when the embedder exposes a `model_id`/`model_name`) the model identity.
  `mode='changed'` re-embeds exactly the nodes whose text changed since the last
  pass (or are missing), instead of all (`mode='all'`, = `replace=True`) or only
  the missing ones (`mode='missing'`, the default, = `replace=False`). This
  subsumes the per-node `text_hash` machinery consumers were hand-rolling for
  the rebuild-from-source-cache workflow (operator embedding note #1). The new
  `Embedder::model_id()` trait method defaults to `None`, so any bring-your-own
  embedder works unchanged; a Python embedder can opt in with a `model_id` /
  `model_name` attribute. The result dict gains `reembedded_changed`.
- **`embedding_info(node_type, text_column)`** — provenance for an embedding
  store: `{dimension, count, model, metric, hashed}`. Detect a model swap or a
  partially-hashed store without a sidecar (operator embedding note #2).
- **`KnowledgeGraph.freeze()` → `FrozenGraph`** — an immutable, concurrently-
  readable snapshot. Sharing a live `KnowledgeGraph` across threads is unsafe
  (single-owner; a mutation mid-read trips the borrow guard). `freeze()` returns
  a read-only view that shares the graph's data via an O(1) `Arc` clone — no
  deep copy — and has *no* mutating method, so any number of threads can run
  `FrozenGraph.cypher()` against the same snapshot in parallel, lock-free
  (the GIL is released during execution). The snapshot is stable under
  copy-on-write: mutating the source graph afterwards leaves the frozen view on
  the original data. This is the "build → freeze → share → swap" model the
  operator's concurrency note recommends (Tier 2). `FrozenGraph.cypher` is
  read-only — `CREATE`/`SET`/`DELETE`/`REMOVE`/`MERGE` raise; semantic search
  works via `text_score()`/`vector_score()` in the query.
- **`KnowledgeGraph.to_bytes()` + `kglite.from_bytes(data)`** — serialise an
  in-memory graph to a `.kgl` byte buffer and load it back, without going
  through a filesystem path. Lets a caller own the write (object storage, a
  pipe, a checksum, a custom atomic-write) instead of being limited to
  `save(path)`. `from_bytes` raises a classifiable error on a corrupt/truncated
  or non-`.kgl` buffer (operator durability note §4). Default/mapped modes only
  (a disk graph is a directory, not a byte stream). The Rust api surface gains
  the backing serializers — `kglite::api::{write_kgl, write_kgl_with, write_kgl_to,
  load_kgl_bytes}` — so non-Python bindings get the same atomic-write + byte
  round-trip.
- **`replace_connections(...)`** — an atomic edge upsert. For every source node
  present in the input (`data` DataFrame or `query` result), its existing edges
  *of that connection type* are pruned, then the supplied edges are added — in
  one call. Edges from sources not in the input, and edges of other types from
  the same sources, are untouched. Use it to re-sync a derived edge set
  idempotently ("the current `MENTIONS` of exactly these documents is this
  list") without the race-prone manual `DELETE`-then-re-add. Accepts every
  argument `add_connections` does (including query mode and `extra_properties`);
  validates the id columns before pruning, so a malformed input leaves the graph
  intact (operator B3). Implemented in core (`maintain::replace_connections`), so
  every binding reaches it.
- **`embedding_dim(node_type, text_column)`** — returns the vector dimension of
  an embedding store (or `None`). A cheap, direct way to detect an embedder/
  model change without iterating `list_embeddings` (operator B4).

### Changed

- **`.kgl` embedding section format bumped (core-data-version 3).** The embedding
  store now persists per-vector `model_id` + per-node `text_hashes` (positional
  bincode fields), so a `.kgl` *with embeddings* saved by an older version can't
  be loaded by this binary — it's rejected with a clear "reload, re-embed, save
  again" message. The graph's **nodes/edges/columns are unaffected** and a `.kgl`
  *without* embeddings loads unchanged; only the rebuildable vector cache breaks.
  This is a deliberate, contained break (embeddings are a rebuildable cache),
  not a whole-graph format break.
- **`save()` is now atomic and durable by default.** The `.kgl` is written to a
  sibling temp file and atomically renamed over the target, so a crash mid-save
  can never leave a torn/truncated file — a reader always sees either the old
  file or the complete new one. With the new `fsync=True` default, the file and
  its parent directory are flushed to physical storage before returning (durable
  against an OS/power crash); pass `save(path, fsync=False)` to skip the flush
  for speed (still atomic, just not guaranteed on-disk at return). The temp name
  is unique per process so two writers to one path can't corrupt each other's
  in-flight write. Removes the temp-file + `os.replace` + dir-fsync dance
  consumers were hand-rolling (operator durability note §4). Every existing
  caller (including `to_subgraph().save()` and code-graph builds) gets this for
  free. Cost: the atomic temp+rename adds a fixed ~one-file-create+rename per
  save (negligible on real graphs; the `fsync` flush is the larger, optional
  cost — `fsync=False` for the hot-loop case). Serialization throughput itself
  is unchanged.
- **`embed_texts(replace=False)` now rejects a model/store dimension mismatch**
  instead of silently mixing dimensions (which corrupts similarity search). On a
  model swap, re-embed the whole column with `replace=True` (deterministic —
  rebuilds the store at the new dimension) or `remove_embeddings` first. (B4/B5;
  `add_embeddings` already rejected mismatches.)
- **Graph-algorithm procedures: `relationship` and `connection_types` are now
  interchangeable, and unknown config keys are rejected.** The edge-scope key
  was inconsistent (centrality/community read `connection_types`; components/
  k-core read `relationship`); either term now works on any procedure. A
  genuinely-unknown key (`CALL pagerank({…, bogus:'x'})`) now errors with a
  did-you-mean instead of silently no-op'ing (operator feedback A2/A2b). The
  `where` predicate-scope (added 0.10.25) was already working — it was the key
  name, not the feature, that tripped callers up.

### Performance

- **Cosine vector search ~1.25× faster (≈21% on a 50k×128 top-10 scan).** Each
  `EmbeddingStore` now caches a per-vector L2 norm alongside the vectors, so
  cosine scoring no longer recomputes the stored vector's norm (plus a `sqrt`)
  on every query — the per-candidate work collapses from "dot + two norm sweeps
  + sqrt" to a single dot product and one divide, with the query norm computed
  once per query. Shared by both the fluent `vector_search` path and the Cypher
  `vector_score()` / `text_score()` scalar (so the fused top-K semantic-search
  path benefits too), and by the all-pairs `link_similar` traversal. Results are
  unchanged (exact within floating-point epsilon). The cache is derived from the
  vectors and **not** persisted — it's rebuilt on load, so the `.kgl` format and
  on-disk bytes are identical. Dot-product / Euclidean / Poincaré are unaffected
  (they need the raw magnitudes and fall through to the existing kernels).
- **HNSW vector index scales whole-corpus search sub-linearly** (see
  `build_vector_index` under Added). Stored-vector-query benchmark (cosine,
  top-10, exact vs indexed): 10k×128 → 4.1× faster (recall@10 0.99); 50k×128 →
  5.6× (0.92); 100k×256 → 8.8× (0.72 at default `ef_search`, raise it for higher
  recall). The speedup widens with corpus size; the index is built once and
  persists in the `.kgl`. Repro: `python tests/benchmarks/bench_vector_index.py`.

### Fixed

- **`load()` / `from_bytes()` raise a classifiable `FileFormatError` on a corrupt
  file/buffer**, not a generic `IOError`. A caller can now reliably distinguish
  "this `.kgl` is corrupt → rebuild from source" (`FileFormatError`) from "it
  isn't there" (`FileError`) or a genuine IO fault (`FileIoError`), instead of a
  broad `except IOError`. (Both already detected corruption; the type is now
  typed end-to-end — operator note #4.)
- **A cross-thread borrow conflict now raises a clear, actionable error
  instead of panicking.** Sharing one `KnowledgeGraph` across threads while a
  thread mutates it (`add_nodes` / `embed_texts` / a `CREATE` query / `save`)
  trips PyO3's `RefCell` guard; the hand-written read paths previously *panicked*
  (`borrow()`), and mutations surfaced a cryptic `Already borrowed`. The read
  paths now raise a `RuntimeError` explaining the single-owner contract and
  pointing to the fix (give each worker its own `copy()` — cheap — or serialize
  access; or share a read-only `freeze()` snapshot — see Added). This is the
  operator's concurrency note Tier 1; the `freeze()` snapshot (Tier 2) ships in
  the same release.
- **`create_index` now reports `created` honestly.** Re-creating an existing
  index is still idempotent (no error), but the returned dict now carries
  `created=false` when an index for `(node_type, property)` already existed and
  `created=true` only when this call made a new one — previously it was always
  `true`, so callers couldn't tell "I made it" from "it was already there"
  (operator B6).
- **A WHERE predicate on a property absent from the matched label now warns**
  (non-fatal, stderr — same channel as the unknown-label/relationship warnings)
  instead of silently filtering out every row. `MATCH (f:Function) WHERE
  f.nonexistent = false …` previously returned `No results` indistinguishably
  from a genuine empty match (`null = false` → false); now it emits
  *"WHERE references property 'nonexistent' which no Function node has …"* with a
  did-you-mean. A warning, not an error, so legitimately-sparse (sometimes-null)
  properties — which are in the type's metadata — never trip it (operator A1b).
- **Code-graph: `is_external` is now emitted on `Function` (= `false`), not just
  `Class`/`File`.** Previously `f.is_external` was null on functions, so the
  documented library-only filter `WHERE n.is_external = false` silently matched
  zero rows on `Function` (`null = false` → false). Now uniform across labels.

### Documentation

- **Identifier charset & escaping** is now documented in `CYPHER.md` (operator
  B2). Hyphenated/dotted/spaced relationship types and labels (`supports-claim`,
  `refines-idea`, `Legal Document`) have always worked; they just need
  backtick-quoting *inside a Cypher query* (`[r:`supports-claim`]`). The
  string-typed Python APIs (`add_connections`/`replace_connections`/`add_nodes`/
  `create_index`) accept arbitrary characters directly — no escaping. The bare-
  identifier rule (`[A-Za-z_][A-Za-z0-9_]*`, else backtick) is now spelled out.

## [0.10.28] — 2026-06-16 — bring-your-own embedder (library + model); `[embed]` extra removed

### Changed

- **MCP server embedder is now `library`-based and bring-your-own.** Replaced
  the `extensions.embedder.backend` field with `library` (the engine you name):
  `sentence-transformers` / `fastembed` (Python, wheel-hosted) or `fastembed-rs`
  (Rust, cargo `--features fastembed`), plus a `factory: module:attr` escape for
  any custom embedder. The Rust server hands the whole config to the Python side
  (`kglite._mcp_embed`), so adding a library never touches Rust. This **unlocks
  `bge-m3` on the pip server** via `library: sentence-transformers` — fastembed-py
  (the previous hardwired choice) doesn't have bge-m3; fastembed-rs and
  sentence-transformers do. Unknown library / not-installed / host-mismatch all
  produce actionable boot errors. (Supersedes the `backend: python` shape added
  hours earlier in 0.10.27.)

### Removed

- **The `kglite[embed]` extra.** Embedding is bring-your-own: `pip install kglite`
  pins no embedding library; install whichever you name (`pip install fastembed`
  / `sentence-transformers`), matching the engine's `g.set_embedder(...)`
  philosophy. `pip install 'kglite[embed]'` no longer resolves.

## [0.10.27] — 2026-06-16 — value_codecs (safe literal conversions); cypher_preprocessor removed

### Added

- **`extensions.value_codecs` — position-scoped, bidirectional literal codecs.**
  An operator declares a transform (`prefix` / `map` / `regex`) bound to a
  stored property; the engine decodes query-side literals in that property's
  position (`{id:'Q42'}`, `WHERE n.id = 'Q42'`, `n.id IN [...]`, `CREATE/SET`)
  and encodes direct result-column projections back (`RETURN n.id` → `'Q42'`).
  Applied **after parsing** by a new `cypher::value_codec` pass, reached via
  `ExecuteOptions::value_codecs` and configured from the MCP manifest. Five
  safety invariants: position-scoped (a `'Q42'` in `CONTAINS` / a different
  property / a `RETURN` alias is untouched), full-match (never query-text
  substitution), decode-is-total (any non-match leaves the literal as-is, so
  the 0.10.10 over-eager coercion stays dead), bidirectional, and typed (decode
  lands a real `Value`, hitting the same index path as a native literal). No
  trust gate — a Tier-1 codec is pure declarative data transformation. New
  `kglite::api::cypher::{ValueCodec, CodecKind, StoredType}`. See
  `docs/python/examples/manifest_value_codecs.md`.

### Removed

- **`extensions.cypher_preprocessor` (both `rules:` and `command:`) — removed.**
  Introduced in 0.10.26, it rewrote the raw query *text* before parsing — blind
  substitution that could mangle string literals, `RETURN` aliases, or anything
  that merely contained the pattern (re-creating the over-eager-match failure
  0.10.10 deliberately killed). `value_codecs` does the conversion at a safe,
  post-parse, position-scoped site instead. No deprecation window (0.10.26 had
  no released consumers). `trust.allow_query_preprocessor` is now unused by
  kglite.

## [0.10.26] — 2026-06-16 — MCP server bundled into the wheel + native query preprocessor

### Added

- **`pip install kglite` now ships the `kglite-mcp-server` command.** The
  pure-Rust MCP server moved into the wheel: its server body lives in the
  `kglite-mcp-server` *library* (`run`), is statically linked into the
  compiled extension (sharing the one `kglite` engine — no separate wheel, no
  duplicated engine, ~6 MB added to the extension), and is exposed to Python as
  `kglite._run_mcp_server`. A thin `kglite/mcp_server.py` console-script shim
  forwards argv into it, so `pip install kglite && kglite-mcp-server …` runs the
  identical server as `cargo install kglite-mcp-server`. This restores the
  pip-native entry point retired in 0.10.25, now backed by the single Rust
  implementation rather than a parallel Python server. The standalone cargo
  binary is unchanged (it's now a thin `main.rs` over the same library).
  (Semantic search via `extensions.embedder` still needs the fastembed cargo
  feature, which neither the default wheel nor the default cargo binary ships —
  build the standalone binary with `--features fastembed`.)
- **`extensions.cypher_preprocessor` — rewrite agent Cypher before execution.**
  Trust-gated (`trust.allow_query_preprocessor: true`, mirroring
  `trust.allow_embedder`), in two shapes: declarative `rules:` (ordered regex
  substitutions with `$1` backrefs) and a `command:` subprocess hook (query on
  stdin → rewritten query on stdout, run with the manifest dir as cwd). Applies
  to every `cypher_query` and manifest `tools[].cypher` invocation. The
  motivating case is Wikidata Q-number rewriting (`'Q42'` → `42`, since the
  engine stopped auto-coercing prefixed ids in 0.10.10); it natively replaces
  the bespoke FastMCP rewriting server an operator would otherwise hand-roll.
  A boot error (not a silent no-op) when declared without the trust gate. See
  `docs/python/examples/manifest_cypher_preprocessor.md`.
- **`extensions.embedder.backend: python` — semantic search in the bundled
  server with no Rust toolchain.** The pip-hosted server can now back
  `text_score()` with a fastembed-**py** model (`pip install 'kglite[embed]'`)
  instead of the fastembed-**rs** cargo feature. `kglite._run_mcp_server` takes
  an embedder factory; when a manifest declares `backend: python`, the server
  builds the Python model and wraps it in the existing `PyEmbedderAdapter`,
  which re-acquires the GIL only for the (once-per-query) embed — the ranking
  over the graph still runs in Rust with the GIL released, so non-`text_score`
  queries are unaffected. This closes the one gap from the wheel-bundling work:
  embedder MCP servers (e.g. semantic-search corpora) no longer need
  `cargo install --features fastembed` — `pip install 'kglite[embed]'` suffices.
  The standalone cargo binary has no Python, so it rejects `backend: python`
  with a clear message and keeps using `backend: fastembed` (fastembed-rs). See
  `docs/python/examples/manifest_with_embedder.md`.

## [0.10.25] — 2026-06-16 — code-graph ergonomics, algorithm scoping, single Rust MCP server

### Added

- **MCP server: bundled `code_graph_analysis` + `code_graph_views` skills.**
  Cross-tool skills (attached via `references_tools`, gated
  `graph_has_node_type: [Function, Class]`) that teach graph-first analysis —
  map structure with `graph_overview`/`cypher_query`/`explore`, drop to
  grep/read only to confirm — and library-only views (the `is_test` /
  `is_benchmark` / `is_external` filters, `{where:'…'}` algorithm scoping, and
  `parse_json` for `parameters`/`fields`). This is the guidance operators
  previously hand-rolled into `instructions:`. Requires mcp-methods **0.3.42**
  (the `serve_prompts` pass that injects a skill's `description` under
  `## When to use` and honors `references_tools`); the pin was bumped this
  release.
- **MCP skill-authoring guide** (`docs/python/guides/mcp-skills.md`): the
  frontmatter schema, the `<basename>.skills/` project-layer convention, the
  `skills:` value shapes, `applies_when` gating, the three text channels
  (`instructions:` vs `overview_prefix:` vs skills), the injection size caps,
  and which frontmatter keys are load-bearing vs decorative.
- **Code-graph provenance flags `is_benchmark` and `is_generated`.**
  `is_benchmark` (path-based — `asv_bench/`, `benchmarks/`, `bench/`) joins the
  existing `is_test` on `File` / `Module` / `Function` / `Class`, and `is_test`
  is now also emitted on `Class` (so test classes like `PlotTestCase` can be
  excluded from fan-out / centrality queries). `File` nodes carry
  `is_generated` (true for machine-produced files skipped as generated /
  minified). Lets analysis queries scope to library-only code:
  `WHERE c.is_test = false AND c.is_benchmark = false`.
- **`parse_json(s)` Cypher function** (alias `from_json`). Recursively parses a
  JSON string into a structured map / list / scalar (null on invalid input), so
  Cypher can predicate over data stored as a JSON string — notably the code
  graph's `Function.parameters` and `Class.fields`:
  `WHERE any(p IN parse_json(f.parameters) WHERE p.type_annotation = 'Dataset')`.
- **Subgraph scoping for the centrality + community procedures.** `pagerank`,
  `degree`, `betweenness`, `closeness`, `louvain`, `leiden`, and
  `label_propagation` now accept optional `{node_type, where}` parameters that
  restrict the algorithm to a property-filtered subgraph, so test / benchmark /
  external nodes no longer pollute centrality and community results:
  `CALL pagerank({node_type:'Function', connection_types:'CALLS', where:'n.is_test = false'})`.
  `where` is a predicate over the node variable `n` (full WHERE grammar); only
  edges with both endpoints in scope are traversed, and an explicit scope lifts
  the large-graph refusal guard. In-memory graphs only (disk/mapped reject
  scope — filter with a preceding `MATCH`).

### Fixed

- **Code-graph: `is_external` is now `false` on internal nodes, not null.**
  Internal `Class` / `Struct` / `Trait` / `Interface` nodes left `is_external`
  unset, so only external stubs carried the property and the intuitive filter
  `WHERE c.is_external = false` silently matched nothing. Internal definitions
  now emit `is_external = false` explicitly, sharing one boolean column with the
  external stubs (which stay `true`).
- **Code-graph: `qualified_name` / `module` no longer double the package name.**
  In the common clone layout where the directory the graph is built from shares
  its name with the top-level package (`<repo>/xarray/xarray/core/...`), the
  module path was prepended twice (`xarray.xarray.core`). The package prefix is
  now skipped when the relative path already begins with it, so
  `qualified_name` round-trips with the obvious module path (`xarray.core...`)
  and `read_code_source(qualified_name=...)` takes the un-doubled form.

### Changed

- **MCP server consolidated on a single pure-Rust binary.** `kglite-mcp-server`
  is now exclusively the Rust binary (`cargo install kglite-mcp-server`). The
  parallel Python MCP server was retired — the two implementations had begun to
  drift (duplicate skill directories, tool descriptions, and `applies_when`
  logic), and the Rust binary was already the more complete one. `pip install
  kglite` is now the engine + `code_tree` only. (Anti-drift regression tests
  now fail the build if a second MCP server or a second skill source reappears.)
- **`graph_overview` / `describe()` no longer pads the schema with
  uniformly-`false` boolean columns.** On a single-language code graph the
  other frontends' flags (`flutter_build`, `is_ffi`, `is_pymethod`,
  `is_pymodule`, `is_factory`, …) were emitted false on every node and printed
  in both the `<properties>` and `<samples>` sections. They're now suppressed
  from the overview (a boolean that is actually mixed still shows); the columns
  remain present in the graph and queryable via Cypher.
- **`graph_overview` / `describe()` never truncates identifier columns.** The
  node `id` and its alias (e.g. `qualified_name`) are the join key copied into
  follow-up tool calls, so they're now emitted in full regardless of the
  `sample_truncate` setting; other long string values still truncate.

### Removed

- **The Python MCP server (`kglite.mcp_server`) and its `pip`-installed
  `kglite-mcp-server` console script.** Install the server with `cargo install
  kglite-mcp-server`. **Breaking for users who ran `pip install kglite` and
  relied on the bundled `kglite-mcp-server` command** — switch to the cargo
  install (the agent-facing tool surface is unchanged).
- **The wheel's MCP runtime default dependencies** — `mcp`, `pyyaml`,
  `aiohttp`, `watchdog` — plus the internal `kglite._mcp_internal` mcp-methods
  bridge. `pip install kglite` no longer pulls any of these; the wheel is the
  engine + `code_tree` extension only. (The optional `[embed]` extra
  — `fastembed` for engine-level `set_embedder`/`text_score` — is unchanged
  and still available.)

## [0.10.24] — 2026-06-16 — smaller .kgl files, faster CREATE

### Performance

- **Bulk Cypher `CREATE` is ~30% faster — now beats the 0.10.15 baseline.**
  Two per-node redundancies in the node-create path were removed:
  1. `insert_node_routed` registered node-type metadata for *every* created
     node (a `HashMap<String,String>` of property types), and `create_node`
     *also* ran `ensure_type_metadata` per node — duplicate, type-level work
     per row (the regression introduced in 0.10.17). The upsert is only needed
     on disk (where the node can't be read back), so it's gated to disk mode.
  2. `ensure_type_metadata` now skips its read-back + HashMap build + upsert
     when the type's metadata already covers the node's property keys — the
     common homogeneous case after the first node. Heterogeneous nodes (a new
     key) still fall through to the full upsert, so behaviour is unchanged.

  A 50k-node `UNWIND … CREATE` drops from ~49 ms (0.10.23) to ~34 ms — below
  the ~42 ms 0.10.15 baseline. Metadata, MERGE, and save/load round-trip are
  all byte-identical.

### Fixed

- **Smaller, faster-loading `.kgl` files for in-memory builds.** `enable_columnar()`
  (run on every in-memory `save()`) moved id/title into the column store but left
  the inline copies in each node, so they were serialized twice — once in the
  topology section and once in the column section. A fresh build now nulls the
  inline copies (as a load already did), so a freshly-built graph is byte-for-byte
  identical to a load→save round-trip. Eliminates ~27 bytes/node of duplicated
  topology (e.g. a 557k-node graph sheds ~15 MB uncompressed / ~2.6 MB compressed
  topology), shrinking files ~1.5–2% and speeding load ~2–3%. Existing `.kgl`
  files are unaffected and still load correctly.
- **`disable_columnar()` no longer drops node ids/titles.** Rebuilding per-node
  storage from the column store omitted the reserved `__id__`/`__title__`
  columns, so calling `disable_columnar()` on a loaded graph (whose nodes hold
  the columnar null-sentinel) wiped every id and title. It now restores them
  from the store.
- **Deterministic `.kgl` output for Cypher-`CREATE` graphs.** The schema slot
  order was derived from a `properties` HashMap whose iteration order is
  randomized per process, so saving the same CREATE-built graph could produce
  different column orderings — and, because zstd's ratio is order-sensitive,
  different compressed bytes — run to run. The CREATE path now sorts schema keys,
  so identical input always yields identical output (`save` is reproducible).

## [0.10.23] — 2026-06-15 — code_tree docs pass: link a repo's prose to its code

### Added

- **`code_tree` can ingest a repo's docs and link them to its code.** Pass
  `include_docs=True` to `code_tree.build(...)` / `code_tree.repo_tree(...)` to
  ingest every `.md` **and `.rst`** as a `:Doc` node *and* link it to the rest of
  the graph:
  - `(:Doc)-[:MENTIONS]->(:Function|:Class|:Struct|:Enum|:Trait|:Interface|:Constant)`
    — symbols named in the prose, resolved conservatively from strong code
    signals only (Markdown backtick spans / `::`-qualified names; reStructuredText
    `:func:`/`:class:`/… cross-reference roles and double-backtick literals).
    Resolution tries, most precise first: exact `qualified_name`; a segment-
    aligned **dotted-suffix** match when the doc gives a path (`Dataset.mean`);
    a *unique* bare `name`; and — when a bare name is overloaded — a unique
    **module-level** definition (a free function beats class methods, recovering
    re-exported top-level API like `concat` / `apply_ufunc`). Ambiguous names,
    common words, and private / dunder names never link.
  - `(:Doc)-[:DOCUMENTS]->(:Doc|:File)` — links to another doc (Markdown
    `[..](other.md)` / RST `:doc:`other``, by `concept_id`) or a source file
    (by unique basename).
  Each `:Doc` carries a `kind` (readme / changelog / contributing / license /
  adr / guide / doc), a `headings` outline, and a `file_path` pointer. Markdown
  reuses the OKF loader; reStructuredText (the Sphinx format across scientific-
  Python — numpy / pandas / xarray) has a dedicated extractor. `kg_skip: true`
  markers and the code walk's directory pruning are honored. Off by default
  (existing code-only graphs are unchanged); the open-source MCP server enables
  it by default for cloned GitHub repos.

- **OKF wikilink anchors resolved.** `[[Note#Heading]]` now targets `Note` (the `#anchor` is stripped, mirroring path-link fragment handling), so section links no longer create phantom dangling references.
- **OKF `skip_dirs` directory ignore.** `okf.build(..., skip_dirs=[...])` prunes whole directories (and their subtrees) from the walk — gitignore-style: a bare name matches a directory at any depth, a `path/with/slashes` is an anchored bundle-relative subtree. For excluding cloned / vendored trees you don't own.
- **OKF `kg_skip` opt-out marker.** A file with `kg_skip: true` in its
  frontmatter is excluded from `okf.build` sweeps by default — drop it into
  scratch notes or a project you don't want in a cross-project graph. Honored by
  default; pass `respect_skip=False` to ingest skip-marked files anyway.

## [0.10.22] — 2026-06-15 — OKF: structured-only sweeps + memory-aware labels

### Changed

- **`okf.build` now ingests only structured `.md` by default** — files with a
  YAML frontmatter block (`require_frontmatter=True`). This is the discriminator
  between *structured* knowledge (OKF concepts, Claude memories) and plain
  markdown (READMEs, notes), so you can point at a **parent of many projects**
  and sweep out only the structured knowledge across all of them in one pass —
  each project's tree becomes `Folder` nodes; concept ids stay path-relative.
  (Measured: ~2,000 nodes from a whole multi-project code tree in ~4 s.) Pass
  `require_frontmatter=False` for vault-style ingestion of every `.md`.
- **Memory-aware labels and titles.** Node label falls back `type` →
  `metadata.type` → `Concept`, and title falls back `title` → `name` → file
  stem. Claude memory files (which carry `metadata.type` and `name`, not a
  top-level `type`/`title`) therefore land as `:feedback` / `:project` /
  `:user` / `:reference` nodes titled by their `name`, queryable as
  `MATCH (m:feedback) …`.

### Fixed

- **Dangling-reference stubs now carry `concept_id`** (and `_provisional`),
  matching real concepts, so "references not yet written" are queryable
  uniformly via `MATCH (n {_provisional: true}) RETURN n.concept_id` regardless
  of whether the bundle has any bare `Concept` nodes.

## [0.10.21] — 2026-06-15 — richer OKF graphs (folders, tags, sources)

### Changed

- **OKF ingestion now extracts more meaning from the bundle, by default.**
  `okf.build` synthesizes three structural node types beyond bare concepts, so
  the result is a dense, well-clustering graph instead of a sparse author-link
  one — all from data already in the bundle (no embeddings, no new dependency):
  - **`Folder` nodes** — the directory hierarchy, `(:Folder)-[:CONTAINS]->`
    concepts and subfolders. Each directory's `index.md` (previously discarded)
    now enriches its Folder's title/description.
  - **`Tag` nodes** — `(:Concept)-[:TAGGED]->(:Tag)` from frontmatter `tags`,
    so concepts sharing a tag are connected through a hub.
  - **`Source` nodes** — external `http(s)` links (previously dropped) become
    `(:Concept)-[:CITES|REFERENCES]->(:Source {url})`, enabling co-citation.
  - **Forgiving link resolution** — `[[wikilinks]]` and paths resolve by exact
    id → file stem → normalized slug (case- and `_`/`-`-insensitive) → title, so
    `[[my-note]]` / `[[My Note]]` / `my_note.md` all reach the same concept (cuts
    false-dangling references substantially on real memory dirs).

  On the reference Google bundles this densification roughly tripled node/edge
  counts and turned fragmented clustering into useful communities (stackoverflow
  `CALL leiden`: 19 communities with 12 singletons → 6 communities). Note: with
  structural `CONTAINS`/`TAGGED` edges, an "orphan" query should exclude those
  edge types. Each enrichment can be disabled via `BuildOptions`.

## [0.10.20] — 2026-06-15 — Native OKF (Open Knowledge Format) ingestion

### Added

- **Native OKF (Open Knowledge Format) ingestion — `from kglite import okf`.**
  Loads a directory of markdown files with YAML frontmatter, cross-linked by
  markdown links — Google's [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog),
  and equally Claude memory dirs, skills folders, and Obsidian vaults — into a
  `KnowledgeGraph`. Conceptually `code_tree` for prose knowledge: read-only and
  **partial** (each concept becomes a node carrying its frontmatter as
  properties plus a `file_path` pointer; the body is read on demand via
  `okf.source(path)`, not stored unless `with_body=True`). Markdown links become
  typed edges via an inference ladder — explicit link title (`[x](/y.md
  "JOINS_WITH")`) → enclosing section header (`# Citations` → `CITES`) →
  `LINKS_TO` — plus structural `CONTAINS` edges; links to not-yet-written
  concepts become `_provisional` stub nodes (`MATCH (n {_provisional:true})`).
  A `dialect="obsidian"` mode also resolves `[[wikilinks]]` and tolerates
  frontmatter without a `type`. OKF ships no query engine of its own, so the
  result composes with everything KGLite already has — `CALL leiden`/`pagerank`
  to cluster/rank a knowledge corpus, the `orphan_node` rule to find
  unreferenced notes, temporal filters for staleness. Feature-gated behind the
  engine's `okf` Cargo feature (pulls only `yaml-rust2`); enabled in the wheel,
  off in the bare crate so non-OKF builds pay nothing.

## [0.10.19] — 2026-06-14 — Leiden + multilevel Louvain + bounded-memory algorithms

### Added

- **Leiden community detection — `CALL leiden(...)`.** The algorithm the
  GraphRAG ecosystem standardised on: like Louvain but a refinement phase
  guarantees every returned community is **well-connected** (Louvain can return
  internally-disconnected communities). `CALL leiden({resolution,
  weight_property, connection_types}) YIELD node, community [, level]`.
  Deterministic variant — refinement splits communities into connected
  components (guaranteeing connectivity) without the reference implementation's
  randomised modularity sub-refinement, so results are reproducible. Reaches
  every interface via `cypher_query`; documented in `describe()` /
  `graph_overview(cypher=['leiden'])` and CYPHER.md.

### Changed

- **Louvain is now multilevel (hierarchical).** `CALL louvain(...)` /
  `louvain_communities()` previously ran only the local-moving phase (single
  level) — no aggregation, no hierarchy, and lower modularity than the full
  algorithm. It now runs the complete multilevel loop (local-move → aggregate →
  repeat), finding higher-modularity partitions. **The returned flat partition
  may differ from prior versions** (coarser, higher modularity) — community
  detection results were never a stable contract; existing `louvain_communities()`
  dict shape is unchanged.
- **`CALL louvain` / `CALL leiden` expose the community hierarchy** via an
  optional `level` column: `YIELD node, community, level` emits one row per
  (node, level), finest (0) → coarsest. Omitting `level` returns the flat best
  partition as before. Useful for GraphRAG-style tiered community summaries.

### Performance

- **Bounded-memory graph algorithms on mapped/disk.** Community detection
  (`louvain` / `leiden` / `label_propagation`) and `k_core` previously
  materialised the whole graph into an in-memory `O(edges)` adjacency before
  running — defeating the point of the mmap-backed `mapped` / `disk` modes,
  which keep the graph off-heap so you can explore a larger-than-RAM graph.
  On those modes they now **stream neighbours on demand from the CSR**, holding
  only `O(nodes)` resident state (edges stay page-cached on mmap). Louvain/Leiden
  stream level 0 (the bulk); the aggregated super-graph at levels ≥1 is tiny and
  stays materialised. The in-memory (`Default`) hot path is unchanged — the
  streaming path is gated on storage mode. This makes "cluster a graph too big
  for RAM" — the GraphRAG indexing bottleneck — a real capability. Streaming
  community detection is also exempt from the per-query deadline (bounded but
  slower than in-memory).

### Fixed

- **Disk backend dropped the last node's overflow edges in three fast paths.**
  On a disk graph whose edges live in the overflow maps (a fresh in-memory disk
  graph, or nodes appended after the last CSR rebuild), `iter_peers_filtered`,
  `count_edges_filtered`, and `neighbors_directed_iter` returned early when the
  CSR offset table didn't cover `node + 1` — skipping the overflow scan entirely.
  The highest-index node therefore appeared to have **no** matching/incoming
  edges (e.g. `MATCH`-free count queries undercounted, and the streaming graph
  algorithms above saw it as isolated). Now mirrors the correct
  `edges_directed_filtered_iter` path: empty CSR range, then always scan overflow.

## [0.10.18] — 2026-06-14 — cyclic pattern-match optimisation (matcher + planner)

### Performance

- **Cycle-closing pattern segments no longer expand-then-filter.** When a node
  variable reappears later in the same pattern (a cycle, e.g.
  `(p)-[:WORKS_AT]->(c)-[:OWNS]->(pr)<-[:CONTRIBUTES_TO]-(p)`) or is pre-bound
  by an `UNWIND` seed, the matching segment only needs to confirm the edge to
  that one already-bound node. The matcher previously expanded *every* neighbour
  of the source and discarded all but the matching one — O(degree) work, plus a
  per-result binding allocation and an intra-pattern bound-variable scan. It now
  passes the bound node as a `target_hint` and rejects non-matching peers before
  any of that — turning the closing segment into a targeted check. Measured on a
  hub-skewed graph: an anchored triangle count dropped **2.2×** (41.3 ms →
  18.9 ms); a 4-way cyclic join (`pattern_match`) ~10% on a uniform graph (the
  win scales with the cycle-close target's degree). Variable-length segments are
  unaffected (they still expand). Results are identical — verified by new
  `TestCyclicPatternCorrectness` cases (exact cycle counts + no over-match) and
  a `knows_triangle_cycle` entry in the differential corpus.

- **Cyclic patterns are re-rooted at their most-selective node** (new planner
  pass `reorder_cyclic_pattern_edges`). A cycle like
  `(p:Person)-[:WORKS_AT]->(c:Company)-[:OWNS]->(pr:Project)<-[:CONTRIBUTES_TO]-(p)`
  was evaluated from the written start (`p` — every Person), materialising a
  huge intermediate set before the cycle closed; `optimize_pattern_start_node`
  couldn't help (a cycle's two ends are the same variable, so its reverse is a
  no-op). The pass rotates the ring so the smallest-cardinality node starts the
  walk (here `Company`, ~25× fewer start rows) and — when the edge-type-count
  cache is warm — orients it so the cheaper incident edge drives first; the
  closing segment then lands on the bound start node (the O(1) check above).
  On the 25k-node embedded-app graph this took the 4-way cyclic `pattern_match`
  join from ~16.8 ms to ~6.2 ms (**2.7×**, matching kùzu). Strictly shape-gated
  — fires only on a simple ring of clean single-typed edges and only on a clear
  (≥4×) selectivity win, so every acyclic pattern is left byte-identical. The
  pass is disable-able via `disabled_passes=["reorder_cyclic_pattern_edges"]`;
  a `TestCyclicPatternCorrectness` case asserts optimised == naive when it fires.

## [0.10.17] — 2026-06-13 — durable WAL writes (`durable=True`), disk Cypher CREATE/MERGE, embedded-app perf

### Performance

- **Durable `SET`/property-update no longer scales with graph size.** On a
  `durable=True` graph loaded from a checkpoint (columnar storage), a Cypher
  `SET` ran in O(N-of-type), not O(rows-updated): ~113 ms to set one property on
  one node in a 127k-node graph. Two coupled causes, both fixed: (1) the
  columnar `SET` fast path writes through the master `ColumnStore`, bypassing the
  WAL capture wrapper, so the actual mutation wasn't recorded directly; (2) the
  per-node `Arc<ColumnStore>` handle-refresh sweep that follows touches *every*
  node of the type via `node_weight_mut`, which the wrapper captured as N
  spurious mutations — logging (and re-serialising) the whole type per `SET`.
  Now the fast path records the one mutated node explicitly
  (`note_recorded_node_upsert`), and the refresh sweep uses a new
  `GraphWrite::node_weight_mut_silent` that bypasses capture (it's internal
  storage bookkeeping, not a logical mutation). Durable 1-node `SET` dropped from
  ~5/24/113 ms (2.5k/25k/127k nodes) to a flat ~3 ms; a 500-node `SET` on a
  127k-node graph from ~120 ms to ~5 ms. Crash recovery still captures the `SET`
  (verified). Surfaced by the embedded-app benchmark + a kùzu source-level
  comparison (its per-column in-place update has no such sweep).

- **WAL crash recovery is no longer quadratic.** Replaying recovered WAL frames
  routed each frame through `add_nodes`, which rebuilds the type's id-index per
  call — so recovering N un-checkpointed single-row commits was O(N · graph).
  Replay now **folds all frames into net per-entity state** (last write wins per
  `(node_type, id)` / `(conn, src, tgt)`) and applies it in a handful of bulk
  calls, rebuilding each index once. Sound because the ops are identity-keyed and
  idempotent. Replaying 1,000 un-checkpointed commits onto a 21k-node checkpoint
  dropped from ~932 ms to ~42 ms (~22×); the win grows with the un-checkpointed
  frame count. (Only matters between checkpoints — `save()` truncates the WAL.)

- **`WHERE n.id IN $ids RETURN count(n)` now anchors on the id index instead
  of full-scanning.** The `fuse_node_scan_aggregate` planner pass fused
  `MATCH (n) WHERE … RETURN count(n)` into a streaming node sweep that applied
  the predicate per node — correct for a non-indexable filter like `age > 30`,
  but ~40× too slow when the filter is an `id` equality / `id IN …` that should
  seed from the always-present id index. The pass now **bails on an
  id-anchorable WHERE**, leaving the MATCH+WHERE+RETURN for the eq/IN-anchoring
  passes to drive from the index, then counting the small anchored set. On a
  21k-node graph, `WHERE n.id IN $ids` (500 ids) `count(n)` dropped from ~27 ms
  to ~0.6 ms; `WHERE n.id = $x RETURN count(n)` is now an O(1) index hit.
  Non-id filters keep fusing (the streaming scan is the right plan there).
  Trigger shapes added to the differential corpus (`id_in_count_bails_fusion`,
  `id_eq_count_bails_fusion`). Surfaced by the embedded-app benchmark, where
  batch point-lookup-by-id was the one phase kùzu won.

### Added

- **`kglite.open(path, durable=True)` — crash-safe durable graphs (write-ahead
  log).** A committed Cypher mutation is now appended to a `<path>-wal` sidecar
  and `fsync`'d *before the call returns*, so it survives a hard crash
  (`kill -9` / power loss) — not just a clean close. On open, any WAL frames are
  replayed onto the loaded `.kgl` checkpoint to recover work committed since the
  last `save()`; a durable graph that was never saved recovers entirely from its
  WAL. `save()` writes a full checkpoint and truncates the WAL. The log is
  logical and identity-keyed (`(node_type, id)` / `(conn_type, src, tgt)`) with
  idempotent upsert/remove ops + per-frame CRC, so a torn trailing frame from a
  crash mid-append is discarded and replay is safe to repeat. This is the first
  half of contesting the embedded-Cypher-database use case: a committed mutation
  is durable without an explicit `save()`. **In-memory graphs only** in this
  release (`storage="mapped"/"disk"` raise `ValueError`); the columnar disk
  modes keep their explicit-`save()` checkpoint model. **In-memory non-durable
  performance is unchanged** — a non-durable graph never enters the capture
  path (verified: tracked mutation/read benchmarks flat vs the prior baseline).

- **`kglite.open(path)` — load-or-create embedded-database lifecycle.** Opens a
  graph at `path`, loading it if the file/directory exists or creating a fresh
  one if it doesn't. The returned graph *remembers* `path`, so:
  - **`save()` takes no argument** when the graph was opened (or previously
    saved) — it writes back to the origin file. Passing a path still works and
    updates the remembered target ("save as"). A graph built purely in memory
    with no path raises a clear `ValueError` rather than failing silently.
  - **Context-manager auto-save-on-close**: `with kglite.open("app.kgl") as g:
    …` snapshots to the file on clean block exit. On an exception the save is
    skipped so the on-disk file keeps its last good state. A new `close()`
    method does the same explicitly.
  - `kglite.load(path)` now also remembers `path` for bare `save()`.

  This is lifecycle ergonomics ("open, mutate, close → persisted"), the first
  step toward contesting the embedded-Cypher-DB use case. It is **not** crash
  safety: a hard crash mid-session writes nothing (durable-on-commit is a
  separate, upcoming capability). `storage=` applies only when creating a new
  graph; opening an existing file keeps its saved mode.

- **Cypher `CREATE` / `MERGE` now work on `storage="disk"` graphs.** Previously
  rejected with a loud guard, because the disk `add_node` writes only a slot
  (node_type + row_id) and drops the node's properties/title/id — a naive CREATE
  would silently lose data. Node insertion now routes through one mode-aware
  choke point (`DirGraph::insert_node_routed`): on disk it pushes id/title/
  properties into the per-type `ColumnStore` (the same mechanism `add_nodes`
  uses) and registers the property types in the schema, so created nodes carry
  their properties and **survive save/reload**. `MERGE` (whose create branch
  reuses CREATE) works too. Reached by every interface (Python, Bolt, MCP) since
  they share the executor. memory/mapped behaviour is unchanged; `SET`/`DELETE`/
  `REMOVE` already worked on disk. The cross-mode parity oracle
  (`test_phase2_parity.py`) and `test_cypher_id_semantics.py` now exercise disk
  CREATE/MERGE.

### Documentation

- **New guide: "Durable embedded apps"** (`docs/python/guides/durable-apps.md`).
  Covers the embedded-app lifecycle — `open()` load-or-create, the
  remembered-path + context-manager checkpoint-on-close ergonomics, and
  crash-safe `durable=True` write-ahead-log writes (fsync per commit, replay on
  reopen, checkpoint-and-truncate). Includes mode selection (in-memory durable
  vs non-durable vs `storage="disk"`), the fsync-bound cost model, and batching
  guidance. Fills the gap where `durable=True` existed only in the API-reference
  stub. Linked from the guides index + toctree.

## [0.10.16] — 2026-06-13 — scoped graph algorithms, IN-param anchoring, disk lazy-edge, docs sweep

A capability-discoverability release: the in-place graph procedures
(`k_core` / `coreness` / `clustering_coefficient`, scoped
`connected_components`) shipped alongside an `IN $param` planner-anchoring
fix and a disk-mode lazy-edge traversal rewrite (~4–20× on disk pattern
match / shortest path), then a documentation sweep that surfaced a batch of
previously CHANGELOG-only capabilities in the user guides with verified
examples. Plus schema "did you mean?" warnings now readable on
`diagnostics["warnings"]` for agent callers.

### Added

- **"Did you mean?" warnings for MATCH typos.** A `MATCH` against a node label
  or relationship type the graph has never seen now emits a non-fatal
  `warning:` to stderr with an edit-distance hint (e.g. unknown label
  `'Persn'` → "Did you mean 'Person'?"). The query still runs and returns zero
  rows (unknown types are legal Cypher — a valid existence check), so this is
  a warning, not an error. It catches the single most common "why is my query
  empty?" foot-gun. Emitted from the shared execute path, so every binding
  benefits. The same messages are now also exposed programmatically on
  `result.diagnostics["warnings"]` (a `list[str]`), so MCP/agent callers that
  never see stderr can read why a query came back empty.

- **`CALL k_core` / `coreness` and `CALL clustering_coefficient`.** Two graph
  procedures, in-place over the knowledge graph (no export to an external
  graph-algorithm library): k-core decomposition (coreness per node, via
  O(V+E) Batagelj–Zaversnik) and local clustering coefficient per node. Both
  take the same optional `{node_type, relationship}` scoping as
  `connected_components` (analyse a single-relationship projection rather
  than the whole graph) and are reached by every binding through
  `cypher_query`. Filter `WHERE coreness >= k` for the k-core itself.

- **Scoped weakly-connected-components.** `CALL connected_components()` now
  accepts an optional parameter map — `{node_type, relationship}`, each a
  string or a list of strings — to restrict the analysis to a subgraph
  instead of the whole graph. `relationship` limits which edge types union
  their endpoints; `node_type` sets the component universe (nodes of other
  types are excluded, even as singletons). With neither, behaviour is
  unchanged (every node, every edge type). This is the "components of a
  single-relationship projection" query — e.g.
  `CALL connected_components({node_type: 'Person', relationship: 'KNOWS'})`
  for the social-graph components — that a graph-algorithm library computes
  on an edge-type-projected view. Backed by the new
  `weakly_connected_components_scoped` core function; reached by every
  binding through `cypher_query`.

- **`k_core` / `clustering_coefficient` now surface in `describe()`.** The
  agent-facing schema introspection lists the new procedures (with the
  scoping note) so an LLM can discover them without reading the docs.

### Documentation

- **Surfaced previously CHANGELOG-only capabilities in the guides.** A
  docs-vs-capabilities audit found several shipped, tested features that were
  invisible in the user guides (so under-discovered, including by evaluating
  agents). Now documented with verified examples: scoped `CALL` algorithms
  (graph-algorithms.md + CYPHER.md), `to_neo4j()` export and `extend()`
  multi-source merge (import-export.md), query tuning & diagnostics —
  `PROFILE`/`EXPLAIN`/`timeout_ms`/`max_rows`/`disabled_passes`/
  `diagnostics["warnings"]` (cypher.md), spatial constructive geometry
  (spatial.md), and temporal time-travel `valid_at`/`valid_during`
  (timeseries.md). Plus a hybrid RAG-over-a-graph retrieval recipe (CYPHER.md)
  locked with tests.

### Performance

- **Disk-mode traversal no longer materialises an `EdgeData` per edge.**
  On `storage="disk"`, every edge crossed during pattern matching,
  variable-length / `shortestPath` traversal, or a relationship-scoped
  `connected_components` used to allocate a heap `EdgeData` (with a cloned
  property vector) and take a per-edge arena mutex — *just to read the
  edge's connection type*, which is available for free from the CSR
  endpoint table. `GraphEdgeRef` now carries the connection type directly
  and materialises the full edge lazily, only when a query actually reads
  edge properties; the traversal hot paths read the cheap
  `connection_type()` accessor instead. On the 25k-node / 266k-edge
  comparative benchmark, disk-mode `pattern_match` dropped ~394 ms → ~19 ms,
  `shortest_path` (100 pairs) ~747 ms → ~120 ms, and scoped
  `connected_components` ~20 ms → ~2 ms — now on par with in-memory and
  mapped. In-memory and mapped are unaffected (they keep their borrowed
  `&EdgeData` path; the connection type is a field they already had).

- **`WHERE x.prop IN $param` now anchors on the index instead of a full
  scan.** The planner's predicate-pushdown only recognised an `IN` list
  written as a literal (`IN [1, 2, 3]`); the parameterised form
  (`IN $ids`, an `InExpression`) fell through to a full type scan +
  post-filter. It now resolves the parameter (and the JSON-array string
  form the Python binding uses for list params) at plan time, pushes an
  `IN` matcher into the MATCH pattern — anchoring on the id index when
  the property is `id` — and rewrites the surviving WHERE to the O(1)
  `InLiteralSet` form so the safety-net re-filter is cheap. On a 2k-node
  graph, `MATCH (p:Person)-[:KNOWS]-(f) WHERE p.id IN $ids` (200 seeds)
  dropped from ~89 ms to ~1.2 ms (1-hop) and ~266 ms to ~1.3 ms (2-hop),
  matching the hand-written `UNWIND $ids … MATCH (p {id:sid})` form.
  Trigger query added to the differential corpus as
  `id_in_param_anchored`.

## [0.10.15] — 2026-06-10 — CALL { } subqueries, graph interop, hot-path perf sweep

A user-experience release driven by a roadmap audit: the most-requested
missing Cypher construct (`CALL { }`), ecosystem bridges
(NetworkX round-trip, in-place graph merge), a profile-gated
performance sweep (-5 to -17 % on the measured hot shapes), a Neo4j
migration guide, and a batch of papercut/correctness fixes — including
an MCP-server boot blocker on clean installs and several
stale-docs purges.

### Performance

- **Whole-node materialization (`RETURN n`, `collect(n)`) skips the
  per-node schema walk on in-memory storage.** Materializing a node
  walked every node-type metadata key per node, paying alias + spatial
  resolution per key, although for in-memory storage that pass can only
  ever recover the hoisted id/title field-alias columns — now fetched
  with two O(1) alias lookups instead (the full walk is kept for the
  columnar disk/mapped backends, which need it). `collect(n)` ~8%
  faster, wide `RETURN n` ~4%, and the tracked `return_node_10k` /
  `return_node_rel_node_100` benchmarks ~4% (min). Python-visible
  output is unchanged, including alias-recovered properties.

- **DISTINCT dedup structures use FxHash.** The `RETURN DISTINCT` /
  `WITH DISTINCT` row-dedup sets (plus the `count(DISTINCT)` /
  `collect(DISTINCT)` / `mode()` / streaming-aggregate DISTINCT sets and
  the `distinct_node_hint` pre-dedup) still ran on the default SipHasher
  — missed by the 0.10.7 FxHash sweep and the top symbol in a samply
  profile of the DISTINCT shape. Same exact-equality dedup, faster
  hasher: `RETURN DISTINCT` over 50k rows ~14-17% faster,
  `collect(DISTINCT)` ~15% (min, 50k-node hot-path suite).

- **Per-row property access skips alias resolution when the property
  can't be an alias.** Every in-memory `n.prop` in WHERE/RETURN paid two
  string-keyed HashMap lookups in `resolve_alias` per row, even for
  properties that are plainly not id/title aliases (the hottest symbol
  in a samply profile of five query shapes). A per-query lock-free
  `OnceLock` set of alias-name hashes now fast-rejects non-alias
  properties — semantics unchanged (id/title virtuals,
  stored-property-wins, spatial fallbacks all preserved; hash collisions
  can only route to the slower full path, never change results).
  Measured on the 50k-node hot-path suite (min): multi-property WHERE
  filters ~12-13% faster, `count(DISTINCT)` ~10-11%, `collect(DISTINCT)`
  ~10%, `ORDER BY … LIMIT` ~9%. Tracked core benchmarks flat.

### Added

- **`KnowledgeGraph.extend(other, conflict_handling='update')`.** Merge
  another in-memory graph in place — multi-source ingest no longer
  round-trips through CSV. Nodes match on `(node_type, id)` and resolve
  per the same `conflict_handling` vocabulary as `add_nodes`
  (`update` / `replace` / `skip` / `preserve` / `sum`); property schemas
  extend automatically; secondary labels union; edges dedup on
  `(connection_type, src, tgt)` with property merge (mirroring
  `add_connections`); id/title field-aliases carry over for new types.
  Returns an `add_nodes`-style report dict. The source graph is never
  mutated; embedding stores are not merged (a warning points at
  `set_embeddings`/`add_embeddings`); v1 requires in-memory `Default`
  storage on both sides. 50k-into-50k with 50% overlap merges in ~21 ms.

- **Cypher: `CALL { … }` subqueries.** Both uncorrelated (body executes
  once; results combine with the outer rows as a cartesian product) and
  correlated via an importing `WITH` (body plans once, executes per
  outer row seeded with *only* the imported variables — node/edge/path
  bindings anchor body patterns, including variables left `null` by an
  `OPTIONAL MATCH` miss). Cardinality follows Neo4j: a non-aggregating
  body with zero rows drops the outer row; an aggregating body always
  returns one row, so per-row counts preserve `0` rows —
  `MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN
  count(f) AS c } RETURN p.name, c`. Scoping is strict: the body sees
  no outer variables beyond the imports, only its `RETURN` columns
  escape, and column collisions with outer scope raise a clear error.
  Writes inside a body route through the mutation classifier and are
  rejected in this version (as are unit subqueries and `UNION` inside a
  body). Planner passes treat the clause as an optimization barrier
  (audited pass-by-pass, documented above the `PASSES` registry); body
  optimization respects `disabled_passes` / `disable_optimizer`.
  Covered by 22 differential-corpus entries, Bolt round-trip
  conformance (172 queries, 0 failures), and Neo4j-conformance queries;
  documented in CYPHER.md with the v1 limitation table and surfaced to
  agents via a `CALL_SUBQUERY` introspection topic. The work also fixed
  a pre-existing hole: a write inside a `UNION` arm was classified as a
  read.

- **NetworkX interop.** `KnowledgeGraph.to_networkx()` exports the graph
  as a lossless `nx.MultiDiGraph` (node key = node id; `node_type`,
  `title`, and all properties as attributes; `connection_type` as the
  edge key, so parallel typed edges stay distinct), and
  `kglite.from_networkx(nx_graph, *, default_node_type='Node',
  default_edge_type='RELATED')` builds a graph from any
  Graph/DiGraph/MultiGraph/MultiDiGraph via the bulk DataFrame fast
  paths. Round-trips preserve ids, types, titles, node/edge properties,
  and parallel typed edges; undirected edges become one directed edge.
  `networkx` stays optional — `pip install 'kglite[networkx]'`; a clear
  ImportError points there when it's missing. 10k nodes / 30k edges
  round-trip in ~0.15 s.

- **Cypher: trigonometry + UUID + local-temporal functions.**
  `sin` / `cos` / `tan` / `asin` / `acos` / `atan` / `atan2(y, x)` /
  `cot` / `haversin` / `degrees` / `radians` (null/non-numeric → null,
  matching the existing math functions); `randomUUID()` (RFC 4122 v4,
  excluded from constant folding so it stays unique per row — no new
  dependency, generated from the existing PRNG); `localdatetime()` /
  `localtime()` / `time()` returning ISO-8601 strings (no-arg = local
  now; 1-arg parse form mirrors `datetime(str)`; strings because
  KGLite's DateTime value is date-only — documented in CYPHER.md).
  Reaches every binding through `cypher()` per the cypher-first policy.

- **`ResultView.one()` / `.scalar()` / `.column(name)`.** The three most
  common result shapes get first-class accessors: `one()` returns the
  first row as a dict (or `None`), `scalar()` the first cell by `RETURN`
  order (or `None`) — `g.cypher("… RETURN count(n)").scalar()` — and
  `column(name)` one named column as a plain list without a DataFrame
  round-trip (`KeyError` listing available columns on a miss). All three
  materialize only what they return; row indexing stays integer-only.
- **`KnowledgeGraph.exists(node_type, unique_id) -> bool`.** O(1)
  existence check via the same id-index as `node()`, with identical
  id-coercion semantics — replaces the `node(...) is not None` idiom
  without materializing the node.

### Fixed

- **`properties(n)`, `keys(n)`, and `n {.*}` now match `RETURN n` on
  graphs loaded with non-literal id/title columns.** When `add_nodes`
  hoists e.g. `npdid`/`name` into the node's id/title, `RETURN n`
  recovered those columns into the property map but `properties(n)`,
  `keys(n)`, and the `n {.*}` map projection each had their own
  enumeration that omitted them (`keys(n)` additionally dropped real
  columns on the disk/mapped backends). All three now delegate to the
  same materializer `RETURN n` uses, so the shapes stay in lockstep
  across every storage mode. The materializer also honours the KG-1
  soft-alias rule for `type`: a stored property named `type` wins over
  the structural type string in all four shapes (matching `n.type`);
  `id`/`title` remain canonical virtuals.

- **Map subscript by string key works.** `{x: 1}['x']`,
  `properties(n)['title']`, dynamic keys (`{x: 1}[k]`), nested access
  (`{a: {b: 2}}['a']['b']`), and dynamic property access on bound nodes
  and relationships (`n['title']`, `r['since']`) previously failed with
  `Index must be an integer`. Missing keys and `null` keys now resolve to
  `null` (Neo4j semantics), never an error. List indexing (including the
  integer fast path, negative indices, and out-of-range → `null`) is
  unchanged.

- **`kglite-mcp-server` no longer refuses to boot on a default install.**
  The startup dependency check demanded `fastembed`, which belongs to the
  opt-in `[embed]` extra — so a plain `pip install kglite` (without
  `[embed]`) exited at launch with advice to install a `[mcp]` extra that
  no longer exists. The boot check now verifies only the deps that ship
  in the default install (`mcp`, `pyyaml`, `aiohttp`, `watchdog`), and
  the embedder paths (fastembed + bge-m3) raise an actionable
  `pip install 'kglite[embed]'` error at point of use instead.

### Documentation

- **Removed instructions to install nonexistent extras.** Getting-started,
  the MCP server guide, the code-tree guide, CONTRIBUTING, and the
  conformance doc all still told users to `pip install 'kglite[mcp]'` /
  `'kglite[code-tree]'` — neither extra exists (the MCP server runtime is
  in the default install since 0.9.41; tree-sitter grammars are bundled).
  Docs now point at plain `pip install kglite` and surface the real
  `[embed]` / `[neo4j]` extras.

## [0.10.14] — 2026-06-08 — Bolt conformance tooling + ResultView.to_dicts() + doc clarity

Finalizes the Bolt server (Phase D) — the conformance oracle, reference
clients, and docs that let us call the feature done. Also folds in a small
API addition and a documentation pass driven by a downstream field report:
most flagged items were already shipped or never broken, so the bulk of
that work is making existing behaviour discoverable.

### Added

- **`scripts/bolt_conformance.py` + `make bolt-conformance`.** On-demand
  oracle that runs the differential corpus through `kglite-bolt-server`
  over the wire and compares against direct in-process `cypher()` —
  catches PackStream round-trip bugs. No Neo4j / Docker needed; it spawns
  its own server. Documented in `docs/concepts/cypher-conformance.md`.
- **Reference examples.** `examples/bolt_client_neo4j_python.py` (drive
  the server with the standard `neo4j` driver) and
  `examples/bolt_neo4j_browser.md` (point Neo4j Browser at it).
- **`ResultView.to_dicts()`** — alias for `to_list()` (returns all rows as
  `list[dict]`). Matches the polars `.to_dicts()` name (pandas calls the
  equivalent `.to_dict(orient="records")`), so consumers coming from either
  library reach the right method without a coercion shim.

### Fixed

- **`scripts/cypher_conformance.py`** passed query parameters as
  `cypher(query, **params)` instead of `cypher(query, params=...)`, so
  the Neo4j conformance run errored on every parameterized query. Now
  fixed (same convention the differential test harness uses).

### Documentation

- **`add_embeddings` surfaced for incremental ingest.** The semantic-search
  guide now has an "Incremental ingest" section: `set_embeddings` is a full
  replace; `add_embeddings` upserts into the existing store (no
  read-merge-write at the call site). `set_embeddings`' docstring cross-refs it.
- **`vector_search` hit contract documented.** Each hit carries `id`,
  `title`, `type`, `score`, and all node properties; `score` is always
  present (every metric); properties are read live, so a hit round-trips
  through `save()` + reload without a follow-up id-join.
- **`ResultView` indexing clarified.** Indexing is row-wise; there is no
  `result["col"]` column accessor (use `to_df()["col"]` or a comprehension).

## [0.10.13] — 2026-06-06 — mcp local-mode github repo auto-detect

### Fixed

- **MCP server (local-workspace mode): `github_issues` / `github_api`
  now auto-detect the repo from the active root's git remote.**
  Previously, calls without an explicit `repo_name` defaulted to the
  `local/<dir>` inventory key and 404'd, even when the active root was
  a checkout of a real GitHub repo. Bumped `mcp-methods` to 0.3.41,
  which derives the default from the root's `origin` remote (falling
  back to "ask for `repo_name`" when there's no GitHub remote).
  `github`-mode behaviour is unchanged.

## [0.10.12] — 2026-06-01 — fluent select/filter perf + count(DISTINCT) fusion

### Performance

- **Fluent `select(sort=…, limit=k)` is now a bounded top-K, not a full sort.**
  Previously it sorted the whole selection then truncated; now it partitions
  (`select_nth`) + sorts only `k`. Combined with a rewrite of the sort path from
  a per-comparison HashMap lookup to a precomputed key vector, top-10/top-100
  over 100k nodes dropped **~25 ms → ~1.9 ms** (~13×). The no-limit full sort is
  also faster (~61 → ~40 ms).
- **Fluent `where()` on a full single-type selection uses the property index
  directly** instead of building an O(N) membership set, and stops allocating a
  `String` per node when deriving candidate types. Indexed-property lookups via
  the fluent API are ~2× faster.
- **`count(DISTINCT <property>)` now fuses into the node scan-aggregate.**
  It previously materialized one result row per scanned node and de-duplicated
  afterward; it now tracks a per-group value set inline during the scan.
  `count(DISTINCT n.prop)` (and the grouped `RETURN k, count(DISTINCT n.prop)`
  form) over 100k nodes dropped **~12.7 ms → ~4.9 ms** (~2.6×). Results
  unchanged.

## [0.10.11] — 2026-06-01 — count(node) no longer materializes per row

### Performance

- **`count(node)` no longer materializes the node per row.** `count(n)` (or
  `count(c)`, …) over a bound node/edge variable evaluated the variable each
  row, building a full node value (every property cloned into a map) just to
  test non-null — the dominant cost of scan-, group-, and traversal-counts. It
  is now treated as `count(*)`. Measured on a 110k-node / 830k-edge graph
  (in-memory): `WHERE … RETURN count(n)` filters ~4× faster, `RETURN k,
  count(n)` group-by ~5×, reverse rel-type counts ~3.5×, and deep fixed-length
  path counts (`…->(n5) RETURN count(n5)`) ~2× — shared across the default,
  mapped, and disk backends. Results, column names, and `count(DISTINCT …)`
  semantics are unchanged.

## [0.10.10] — 2026-05-30 — reserved-name papercuts + cross-mode id parity

Reserved-name Cypher papercuts (KG-1/KG-2) plus a node-`id` correctness sweep
that makes the query interface identical across storage modes.

### Changed

- **Node `id` is the same integer in every storage mode.** For prefixed-id
  datasets (Wikidata `Q42`, …) the loader previously stored `id` as the string
  `"Q42"` in memory/mapped but the compact integer `42` on disk, bridged by a
  too-eager string→int coercion. Now `id` is the **integer** (`n.id == 42`) in
  memory, mapped, *and* disk — identical results everywhere — and the string
  form lives in the `nid` property (`n.nid == "Q42"`). **Breaking** (pre-1.0):
  memory/mapped `n.id` for Wikidata changes `"Q42"` → `42`; query the string
  form via `{nid: 'Q42'}` (or the integer via `{id: 42}`). `{id: 'Q42'}` no
  longer matches. `nid`/`qid` are no longer id-aliases — `{nid: X}` is a plain
  (indexed) string-property lookup. See CYPHER.md → "Naming".

### Fixed

- **A node property named `label` (also `type`/`node_type`/`name`) is readable
  (KG-1).** Property-first resolution across every read path — `RETURN`,
  `WHERE`, inline-map, `EXISTS`, disk fast path, map projection. The
  count-by-type fusion is gated when such a property shadows the type, and
  `n.type` projects a scalar there (matching the un-fused path) while
  `labels(n)` stays a list.
- **`{id: 'a1'}` no longer returns the wrong node.** The string→int id coercion
  (`'a1'`/`'x1'`/`'Q1'` → `UniqueId(1)`) is removed; a `String` id matches only
  by exact value. Numeric (Int64↔UniqueId↔Float) coercions are retained.
- **Cypher `CREATE (n {id: X})` honours `X` as the node identity** (was
  auto-assigned), consistent with `add_nodes(unique_id_field='id')`; string /
  int / float ids round-trip and survive save → load.
- Duplicate ids now emit a rate-limited warning (`MATCH (n {id: X})` returns one
  node per id); detected at id-index build so bulk ingest stays O(n).

### Added

- **Reserved keywords usable as relationship types / node labels / property
  keys / property access (KG-2).** `CREATE (s)-[:CONTAINS]->(c)`,
  `MATCH (n:CONTAINS)`, `{contains: 1}`, `n.contains` parse. The safe set
  (operator/sort/set/mutation keywords) works in every name-position across
  MATCH / CREATE / MERGE / SET / REMOVE / WHERE and EXISTS subqueries;
  load-bearing + value keywords stay reserved with a clear error and the
  backtick escape hatch.

### Internal

- Split the `fusion.rs` optimiser god-file into a `fusion/` module directory.
- Conformance/golden test layer for id semantics (`tests/test_id_parity.py`,
  `tests/test_cypher_id_semantics.py`) + N1–N4 regression locks — the layer the
  differential corpus (optimised-vs-naive) and parity oracles (set-equality)
  structurally can't provide.

## [0.10.9] — 2026-05-29 — self-healing id-index (issue #20)

### Fixed

- **id-equality lookups are O(1) regardless of how the graph was built
  (issue #20).** `MATCH (n {id: X})` and the `MERGE` match go through a
  read-only lookup path that never built the id-index; whenever the index
  was absent for a type — the state `add_nodes`, `CREATE`, and `DELETE`
  all leave it in — every id-equality lookup fell back to an
  O(node-position) linear scan (e.g. ~26 µs for a high-id node on a 30k
  graph vs ~1.1 µs once indexed, with the cost growing as the node's
  position grew). The read path now **self-heals**: on a miss it builds
  and caches the id-index once, so that lookup and every subsequent one is
  O(1) — no matter whether the type was populated via `add_nodes`,
  `CREATE`, or had its index invalidated by `DELETE`. Lookups measure a
  uniform ~1.1–1.5 µs across low and high positions after each of those
  paths. This also explains the "repeated-MERGE 36× slower" report — the
  slow case was always a high-position id; varying-id benchmarks masked it
  via `min` (which caught the cheap low-position samples).

## [0.10.8] — 2026-05-29 — openCypher dialect fixes

### Fixed

openCypher dialect gaps reported by kglite-docs (2026-05-29), all with
regression tests:

- **`labels()` / `keys()` / `properties()` / `id()` on a node VALUE.**
  These returned `NULL` silently when given a node that arrived as a value
  (`collect(a)[0]`, `head(collect(a))`, a WITH projection) rather than a
  bound variable — so the standard "latest per group"
  (`… collect(x)[0] AS latest`) idiom read wrong data with no error. They
  now resolve the node value. Relatedly, a materialised node (`RETURN n`,
  collected nodes) now carries its **full** label set (primary +
  secondary), not just the primary type.
- **`DETACH DELETE` ignores NULL variables.** The idiomatic single-
  statement cascade `MATCH (root) OPTIONAL MATCH (root)-->(child) DETACH
  DELETE root, child` no longer errors when a branch is empty (openCypher
  treats NULL in DELETE as a no-op).
- **Parameter inside an `EXISTS {}` pattern.** `EXISTS { MATCH (a)-[:R]->
  (:T {id:$id}) }` now parses (was a syntax error; the literal form
  worked).
- **Node-property expression as an inline-map value.** `MATCH (b {id:
  other.id})` now parses and resolves `other.id` at match time (against a
  bound node or a projected node value), instead of a parse error.

## [0.10.7] — 2026-05-29 — in-memory query perf (SipHash → FxHash)

A samply profile of the in-memory query hot path found ~23% of engine
CPU in the std default **SipHasher**: maps keyed by `InternedKey` (an
already-well-distributed FNV `u64`) re-hashed it through a cryptographic
hash on every property access, and the GROUP BY path did the same to
group keys per row. Swept those to FxHash (`rustc-hash`, already a
dependency). Semantics-identical, `.kgl`-compatible, correctness verified
by the differential corpus + parity oracles.

### Changed

- **Faster property access** — `TypeSchema::key_to_slot` and
  `StringInterner.strings` (both `InternedKey`-keyed, hit per property
  per row on the Compact read path) now use FxHash.
- **Faster alias resolution** — the id/title field-alias maps
  (`resolve_alias`) now use FxHash.
- **Faster GROUP BY** — the five group-key → group-index maps across the
  streaming, materialized, and fused aggregation paths now use FxHash.

Single-label query A/B vs 0.10.6 (50k nodes, release, min-over-rounds):
`multi_where` −38%, `group_by` −23%, `where_scan` −21%, `proj5` −17%;
aggregate queries an additional few percent. Traversal-bound queries are
roughly flat. No single-label regression (every change is a no-op when no
secondary labels / aliases exist).

- **`mcp-methods` 0.3.40** — picks up the merged watch skip-patterns PR
  plus `graph_overview`/`cypher_query` fastmcp fixes. No API change.

## [0.10.6] — 2026-05-29 — multi-label read-path correctness

0.10.5 shipped secondary labels but only taught the slow matcher path
about them; the optimiser's fused fast-paths and several other
candidate-selection sites still assumed `type_indices[T]` was *all*
nodes labelled `:T`. On a multi-label graph that silently over/under-
counted. One root defect, many surfaces — swept and fixed end-to-end,
with single-label performance provably unchanged (every change is a
no-op when no node carries a secondary label; verified against 0.10.3).

### Fixed

- **Multi-label read paths now consult secondary labels everywhere.**
  0.10.5 shipped secondary labels but only the slow matcher path was
  taught about them — the optimiser's fused fast-paths and several
  candidate-selection sites still assumed `type_indices[T]` was *all*
  nodes labelled `:T`, so on a multi-label graph they over/under-counted.
  Reported by `kglite-docs`: `MATCH (n:Item:Pending) RETURN count(n)`
  over-reported after `remove_label`. (The secondary-label index itself
  was never stale — the bug was entirely read-side, so no data is
  corrupted and existing `.kgl` files read correctly once upgraded.)
  Fixed across:
  - `count(n)` of a typed/secondary label (`FusedCountTypedNode`),
    single-pass scan + top-K aggregation, and `labels(n)` grouping.
  - Edge-expansion endpoint filtering — `MATCH (a:Person)-[:KNOWS]->(b:VIP)`
    returned nothing when `:VIP` was a secondary label.
  - The `n:Label` / `WHERE n:Label` predicate, `MERGE (n:Label {...})`
    (no longer creates a duplicate when matching a secondary-labelled
    node), and the transient equality index.
  - Aggregate and spatial-join fusions that can't express the
    secondary-label union now bail to the correct general path when the
    graph has secondary labels (single-label graphs keep every fast-path).
- **Node deletion evicts secondary labels.** `DETACH DELETE` (and
  provisional purge) now remove the deleted node from the secondary-label
  index instead of leaving a dangling entry that over-counted
  `MATCH (n:SecLabel)`. Loading a graph saved by 0.10.5 after such a
  delete is self-healed (dangling indices are dropped on load).

### Added

- **`select(node_type, ..., include_secondary=True)`** (fluent) — select
  nodes carrying `node_type` as a primary *or* secondary label, the
  fluent equivalent of Cypher `MATCH (n:node_type)`. Default `False`
  preserves primary-type-only selection.

## [0.10.5] — 2026-05-28 — multi-label nodes (Track C)

A node can now wear multiple labels: a primary type (set at
creation, immutable via label mutation) plus an optional list of
secondary labels added through Cypher or the new
`add_label` / `remove_label` pymethods. Triggered by
`kglite-docs`'s 2026-05-28 feature request — agent role taxonomies
(`(:Agent:LLM:Reviewer)`), lifecycle status as label
(`(:Chunk:NeedsOcr)`), cross-type predicates (`MATCH (n:Disputed)`).

Non-breaking by construction. Single-label workloads pay zero
overhead — a `has_secondary_labels: bool` graph-level flag
short-circuits every label-keyed read when no node uses secondary
labels. Sodir / Wikidata / code-tree benchmarks unchanged vs 0.10.4.

### Added

- **Multi-label CREATE syntax** — `CREATE (n:Person:Director
  {name: 'Alice'})` stores `Person` as the primary type and
  `Director` as a secondary label.
- **`SET n:Label` and `REMOVE n:Label`** — add or remove
  secondary labels on existing nodes. Multi-colon syntax
  (`SET n:A:B`) parses as multiple items. `REMOVE n:Primary`
  errors with a clear message (use `SET n.type = 'NewType'` to
  retype).
- **`MATCH (n:A:B)`** — AND-intersect across labels. Returns
  nodes that wear every listed label.
- **`labels(n)` returns the full list** — `[primary, ...secondaries]`
  in insertion order. The single-element behavior since 0.9.52
  was the forward-compat placeholder.
- **`g.add_label(node_type, ids, label)`** and
  **`g.remove_label(node_type, ids, label)`** — direct pymethods
  for batch label mutation by id. Returns `{labelled / removed,
  skipped}`. Idempotent.
- **`add_nodes(..., labels=['X'])`** — new kwarg applies a uniform
  set of secondary labels to every row in the batch.
- **`GraphRead::node_labels_of(idx) -> Vec<InternedKey>`** — new
  trait method returning `[primary, ...extras]`. Default impl
  emits a 1-element vec; Memory + Mapped backends override to
  emit the full list.

### Changed

- **`DirGraph.secondary_label_index` is the canonical store** for
  secondary labels (it was already the runtime fast-path index;
  now it's also the persistence source of truth). `NodeData`
  layout is unchanged from 0.10.4 — pre-0.10.5 `.kgl` files load
  cleanly. Secondary labels persist via a new optional section
  in the `.kgl` v4 envelope (in-memory backend) and via the
  `secondary_labels.bin.zst` sidecar in the disk-graph directory.
  Single-label graphs skip both — zero extra bytes.
- **`DirGraph` gains `secondary_label_index` + `has_secondary_labels`**
  — both `#[serde(skip)]`, rebuilt on load from
  `NodeData.extra_labels`.
- **`graph_overview(cypher=True)` updated** — the "Multi-label
  nodes" limitation note now documents the new syntax instead of
  flagging it as unsupported.

### Internal

- **Single-choke-point label-mutation API** on `DirGraph`:
  `add_node_label` / `remove_node_label` / `node_labels`. Every
  mutation site (Cypher executor, pymethods) routes through these
  so `extra_labels` and `secondary_label_index` can never drift
  apart.
- **`secondary_labels.bin.zst` disk sidecar** — the disk backend's
  columnar layout has no slot for `NodeData.extra_labels`, so the
  inverted index is persisted as a separate zstd-compressed file
  in the disk graph directory. Single-label disk graphs skip the
  write entirely (zero bytes, zero cost). Older 0.10.4 disk
  binaries ignore the unknown file and load with single-label
  semantics; older disk graphs without the sidecar load fine into
  0.10.5+ with an empty secondary index.
- **Linux CI perf baseline refreshed** — the Linux runner baseline
  (`tests/benchmarks/baselines/current.linux.json`) had been
  captured at 0.9.52 (2026-05-23) and accumulated ~+16% of
  measurement drift on `test_bench_add_nodes` across 0.10.0
  through 0.10.4, leaving no headroom for 0.10.5's normal noise
  margin. The hot path for `add_nodes`
  (`mutation/maintain.rs::add_nodes` / `apply_node_batch`) is
  byte-identical between 0.10.4 and 0.10.5 — no code change in
  that path. Refreshed against the CI run for the 0.10.5
  perf-fix commit; new `0_10_5.linux.json` archived alongside
  `current.linux.json`.

## [0.10.4] — 2026-05-28 — kglite-docs feedback round

A downstream library author (`kglite-docs`) sent a 280-line bug
report identifying one silent data-loss bug, one storage panic,
one reload regression, and a small cluster of API papercuts.
This release fixes all of them, ships a new `add_embeddings`
API to make incremental ingest first-class, and folds the
audit-flagged `add_nodes` refactor in on the way through.

### Fixed

- **`set_embeddings` silently dropped embeddings after `add_nodes`
  on a loaded graph.** `BatchProcessor` wrote new ids into
  `id_indices` incrementally, creating a partial entry that
  subsequent `build_id_index` calls trusted as complete. The 50-LOC
  `load() → add_nodes(one row) → set_embeddings(merged)` repro from
  `kglite-docs` now reports `skipped: 0` instead of `skipped: N`.
- **Updating a String property on a columnar-backed node panicked
  with `slice index starts at N but ends at M` (N > M).** Mutating
  `offsets[idx+1]` in `TypedColumn::Str::set` corrupted the start
  of row `idx+1`. String updates now park in a relocated overlay;
  the canonical buffers are rebuilt on save.
- **`vector_search` dropped non-core properties after `save()` +
  `load()`.** Switched to `properties_cloned()` (which handles
  `PropertyStorage::Columnar`) on both result-materialization
  paths.

### Added

- **`add_embeddings(node_type, text_column, dict)`** — upserts into
  an existing embedding store instead of replacing it. Sidesteps
  the read-merge-write workflow that triggered the silent-drop
  bug. Behaves like `set_embeddings` on first call;
  `store_created` in the return dict tells callers which mode ran.
- **`embedding_diagnostics()` rows carry a `length_stats` dict** —
  `mean_length`, `max_length`, `distinct_count`, `distinct_ratio`.
  Callers can filter out short-string and fully-unique columns
  themselves rather than getting every String property reported
  uniformly as `embeddable`.

### Changed

- **`set_embedder(None)` now unbinds** the currently-registered
  embedder instead of raising `AttributeError`. Symmetric with
  `set_embedder(model)`.
- **`describe()` docstring** explicitly notes there is no `limit`
  kwarg — `sample_truncate` is the modern name.

### Documentation

- **Cypher guide** expanded with a new "Why semantic search in
  Cypher matters" subsection covering vector ranking + structural
  filters + graph traversal in one query, with worked examples
  for the kglite-docs document-corpus use case.
- **Cypher guide** new "Edge provenance via reified nodes" section
  explaining when to model relationships as reified `Tagging`
  nodes to recover per-application provenance (and when the
  at-most-one-edge constraint is the right shape).

### Internal

- **`add_nodes` refactored** into eight per-phase private helpers
  (`parse_inline_config`, `extract_embedding_pairs`,
  `convert_dataframe`, `apply_node_batch`,
  `register_feature_configs`, `store_extracted_embeddings`,
  `apply_timeseries`, `build_node_report_dict`). Addresses the
  2026-05-27 codebase-health audit's Hotspot 2; the
  `set_embeddings` fix above lands at the clean seam exposed by
  the refactor.

## [0.10.3] — 2026-05-25 — `kglite-c` C ABI ships; Phase B api lifts

Single release theme: every shipped capability of `kglite` is now
reachable from any language with FFI through a stable C ABI. The
new `crates/kglite-c/` workspace member exposes `kglite::api::*`
through `extern "C"` functions plus a cbindgen-generated header.
Future Go / JS / JVM / .NET bindings link against
`libkglite_c.{so,dylib,dll}` and include the shipped `kglite.h`
instead of re-implementing wrappers in their host language.

Companion to the boundary principle codified in 0.10.2: that
release made `kglite::api::*` rich enough to support future
bindings from Rust; this release adds the C ABI layer that makes
those bindings reach kglite without compiling Rust at all.

### Added — new crate `kglite-c`

New publishable workspace member at `crates/kglite-c/`. Exposes
30 `extern "C"` functions covering the full lifecycle / Cypher
pipeline / dataset / embedder surface. Stable C ABI with
`kglite_` naming convention, opaque-handle types
(`KgliteGraph` / `KgliteSession` / `KgliteCypherResult` /
`KgliteEmbedder` / `KgliteSecClient`), errno-style errors mapping
1:1 to `KgErrorCode`, and feature gating via
`KGLITE_FEATURE_*` preprocessor defines.

- **Lifecycle**: `kglite_load_file`, `kglite_save_graph`,
  `kglite_graph_free`.
- **Session**: `kglite_session_new`, `kglite_session_execute_read`,
  `kglite_session_execute_mut`, `kglite_session_free`,
  `kglite_session_set_embedder`.
- **Result accessors**: `kglite_cypher_result_columns_json`,
  `kglite_cypher_result_rows_json`,
  `kglite_cypher_result_row_count`,
  `kglite_cypher_result_free`. JSON-at-boundary for nested
  `Value` shapes — callers parse with their language's stdlib.
- **Error introspection**: `kglite_status_code_name`,
  `kglite_status_code_neo4j_status`,
  `kglite_status_code_http_status` (wrap the api-level lifts).
- **Datasets** (feature-gated): Sodir (`kglite_datasets_sodir_fetch_all`),
  SEC EDGAR (`_sec_client_new`, `_fetch_quarterly_master_idx`,
  `_fetch_submissions_bulk`, `_fetch_company_tickers`,
  `_fetch_company_facts`, `_resolve_fetch_buckets`,
  `_parse_tickers_json`, `_run_all`, `_client_free`), Wikidata
  (`_ensure_dump`, `_remote_last_modified`, `_decide_cache`).
- **Embedder** (feature-gated): `kglite_embedder_fastembed_new`,
  `kglite_session_set_embedder`, `kglite_embedder_free`.
- **String teardown**: single `kglite_free_string` for every
  owned out-string the library returns.
- **ABI version**: `kglite_abi_version()` returns `{major: 0,
  minor: 10, patch: 3}` for binding startup checks.

`crate-type = ["cdylib", "staticlib", "rlib"]` — consumers can
link statically (Go cgo's `#cgo LDFLAGS: -lkglite_c`) or
dynamically (`libkglite_c.{so,dylib,dll}`). cbindgen runs in
`build.rs` and writes `crates/kglite-c/include/kglite.h` (952
lines, committed; CI verifies the committed copy matches a fresh
cbindgen run).

### Added — `KgErrorCode::http_status_code()` (Phase B)

Companion to `KgErrorCode::neo4j_status_code()` (added in 0.10.2).
Maps each error variant to its canonical HTTP status code:

```rust
let kg_err: KgError = /* … */;
let http_status: u16 = kg_err.code().http_status_code();
// 400 for CypherSyntax/InvalidArgument/etc., 404 for NodeNotFound,
// 408 for CypherTimeout, 422 for Schema/Validation/Expr, 500 for
// CypherExecution/FileIo/Internal.
```

Future REST / gRPC bindings call this rather than re-deciding "is
node-not-found a 404 or a 422?" per binding.

### Added — `kglite::api::param::json_value_to_kglite_value` (Phase B)

New module `crates/kglite/src/param/mod.rs` with the canonical
JSON-to-Value converter. Lifted from
`kglite-mcp-server::tools::json_to_value` (a private helper the
mcp server had been carrying since first ship); the mcp server
now delegates to the core function in one line. Future REST /
gRPC / OpenAPI bindings reach for the same canonical converter
rather than each re-implementing the JSON-shaped boundary.

### Added — design + binding docs

- **`docs/rust/c-abi.md`** (581 lines) — the C ABI design
  conventions: naming, opaque-handle pattern, errno-style errors,
  JSON-at-boundary, sync-only ABI, versioning. Source of truth
  for the `kglite-c` surface and the reference for binding
  authors.
- **`docs/rust/implementing-a-binding.md`** — Option 3 rewritten
  with real cgo / napi / JNI worked examples calling the shipped
  C ABI (was sketches against an unbuilt aspiration). The
  "Phase H aspiration" framing is gone; kglite-c is real.

### Added — CI

- **Header drift gate** in `.github/workflows/ci.yml` (new
  `kglite-c` job): clippy + tests with default features, clippy +
  tests with `sec,sodir,wikidata` features, plus a cbindgen
  regen-and-diff check that fails CI if the committed
  `include/kglite.h` doesn't match what a fresh cbindgen run
  would produce. Catches both forgotten regens and hand-edits.
- **Publish workflow** extended with a 4th publish step
  (`kglite-c`) following the same pattern as bolt-server /
  mcp-server. Version-check job reads kglite-c's Cargo.toml +
  probes crates.io; publish runs after kglite has propagated.

### Changed — internal

- `crates/kglite-c/src/datasets.rs` (single-file) →
  `crates/kglite-c/src/datasets/{mod,sodir,sec,wikidata}.rs`
  (directory). Per-loader files keep each at ~250-650 LOC.
- `SessionState` in `kglite-c` gained an
  `embedder: Option<Arc<dyn Embedder>>` slot; execute_read /
  execute_mut clone it into `ExecuteOptions` per call so
  `text_score()` works through the C ABI.

### Test stats

  cargo test -p kglite-c                                       # 4 + 4 default
  cargo test -p kglite-c --features sec,sodir,wikidata --lib   # 29 unit
  cargo test -p kglite-c --features sec,sodir,wikidata         # 9 integration
  cargo test -p kglite-c --features sec,sodir,wikidata,fastembed
                                                                # 38 total

Full workspace `make lint` and `pytest tests/` remain green.

## [0.10.2] — 2026-05-25 — Dataset-wrapper preparation for future-language bindings (boundary principle landed)

Single release theme: get `kglite` core ready to host future
non-Python bindings (Go via cgo, JS via napi, JVM via JNI, etc.)
without each one re-implementing the same orchestration glue the
Python wheel had to write.

The work is anchored by a new explicit **boundary principle** in
`CLAUDE.md` (Architecture section):

> A wrapper only contains code that is specific to its environment
> and cannot be used by any other sibling wrapper. Anything two or
> more wrappers would write identically belongs in `kglite::api`.

Applied in both directions: lift wheel-side code to core when any
binding would write it identically; demote core code back to a
wrapper when only that wrapper can use it. Eight commits totalling
~2,300 lines of net change in service of one goal.

### Added — `kglite::api` surface

- **`kglite::api::blueprint`** — pure-Rust blueprint loader +
  builder is now part of the curated stable API. Re-exports
  `build`, `load_blueprint_file`, `Blueprint`, `Settings`,
  `NodeSpec`, `Connections`, `FkEdge`, `JunctionEdge`, `TimeKey`,
  `TimeseriesSpec`, `ComputeOp`, `CalendarLink`, `AggregateEdge`,
  `BuildReport`, `FlatSpec`. The Python wheel's `from_blueprint`
  has always been a thin wrapper around these — now any binding
  can call them directly without going through PyO3.

- **`kglite::api::datasets::{sec, sodir, wikidata}`** — dataset
  fetch + extract building blocks now reachable through the
  curated stable API. Each submodule is feature-gated (matches
  the existing `sec` / `sodir` / `wikidata` Cargo features) and
  re-exports the same surface the Python wheel uses via
  `_sec_internal` / `_sodir_internal` / `_wikidata_internal`:
  workdir + storage-mode types, error + `Result` aliases, the
  HTTP client, the async `fetch_*` entry points, and (for SEC)
  the extract pipeline + size-prediction helpers. Lifecycle
  orchestration (cache short-circuit, mode selection, retry
  budgets) stays in each binding's wrapper — the Python wheel's
  wrappers at `kglite/datasets/*/wrapper.py` are the reference
  implementation.

- **Sync wrappers for every async `fetch_*` entry point** —
  `kglite::api::datasets::*::*_blocking` for Wikidata, Sodir, SEC
  (13 functions total). Each spins up a single-thread tokio
  runtime via the new `kglite::datasets::blocking::run` helper.
  Bindings with their own async runtime drive the async variants;
  bindings without one use the blocking variants and let core
  manage the runtime per call.

- **`kglite::api::datasets::sec` — generic helpers lifted from
  the wheel**:
  - `SecFormBucket`, `ALL_BUCKETS`, `LEAN_FETCH_BUCKETS`,
    `resolve_fetch_buckets`, `all_buckets` — the SEC form-type →
    per-filing-fetcher bucket mapping is now canonical in core.
    The Python wheel's `_FORM_BUCKETS` table is sourced from
    Rust at import time.
  - `parse_tickers_json` — parses SEC's `company_tickers.json`
    into a `TICKER → CIK` HashMap. Lifted from the wheel's
    `_resolve_companies` helper.
  - `prepare_dispatch_plan` + `DispatchScope` + `DispatchPlan`
    + `FilingTask` — read `processed/filing_index.csv`, apply
    company / year / form filters, group by bucket. The
    planning half of the wheel's
    `_dispatch_per_filing_fetches` is now in core; execution
    half stays in the wrapper for now (see
    `docs/internal/consider-for-future.md`).

- **`kglite::api::datasets::wikidata::{decide, CacheDecision,
  FreshnessInputs}`** + 3 helpers — the disk-cache freshness
  decision tree (force-rebuild flag → graph age → remote HEAD
  probe → cooldown comparison). 5 outcomes (`Build`, `Load`,
  `Rebuild`) with human-readable reason strings so bindings
  don't re-derive the comparisons for verbose prints. Lifted
  from `kglite/datasets/wikidata.py::open`.

### Changed — `kglite::api` discipline

- **`infer_selection_node_type` demoted** from `kglite::api`
  re-exports. Identified by the reverse audit (see Docs below)
  as taking `&CowSelection` — a type only the wheel uses
  externally, so no other binding could meaningfully call this.
  Stays `pub` in `crates/kglite/src/graph/handle.rs` so the
  wheel reaches it via `kglite_core::graph::handle::
  infer_selection_node_type`. When Selection gets lifted to a
  stable api type, both should move together.

- `discover_property_keys_from_data` doc comment rewritten to
  remove Python-flavored language ("DataFrame-style exporters"
  → "any row-oriented exporter (CSV, Parquet, DataFrame,
  JSON-lines)"). No code change; the function signature is
  generic so the doc shouldn't have claimed otherwise.

### Docs

- **`CLAUDE.md`** — boundary principle now formalised under
  Architecture (see "The boundary principle (north star for
  wrappers vs core)"). Applies in both directions; concrete
  examples; cross-references the binding-implementer guide.

- **`docs/rust/implementing-a-binding.md`** — deep-dive companion
  to `embedding.md` and `session.md` for anyone publishing a new-
  language binding. Covers the bridge-layer choice (Rust direct
  vs language FFI vs the Phase H C ABI aspiration), the full
  `KgErrorCode` mapping table with recommended idioms per
  language family, an Embedder trait implementation walkthrough
  with an OpenAI-API-backed example, blueprint / `.kgl` /
  code_tree / dataset loading patterns, the binding-side
  cookbook (process cache, lazy materialization, value
  conversion, progress callbacks), and a cross-binding
  portability checklist. References the three existing
  reference implementations (`kglite-py`, `kglite-bolt-server`,
  `kglite-mcp-server`) as the canonical worked examples. New
  "Wrapping a dataset for your binding" chapter opens with the
  boundary rule and walks the six-step lifecycle common to all
  three datasets.

- **`docs/internal/api-audit-2026-05-25.md`** — Phase 1 audit of
  the `kglite::api` surface ahead of the binding-implementer
  guide. Inventories the 29 + 2-submod current surface,
  classifies every `#[pymethods]` gap (Class A/B/C/D), and
  ranks a top-10 punchlist.

- **`docs/internal/mcp-server-parity-2026-05-25.md`** — Phase 4
  feature-parity audit between the Python MCP server
  (`kglite/mcp_server/`, 3548 LOC) and the Rust MCP server
  (`crates/kglite-mcp-server/`, 1974 LOC). Both ship; neither
  is being retired. 12 of 13 tools at full parity; the one tool
  gap is `explore` (Rust-native, Python lacks). Two "should
  converge" items deferred to `consider-for-future.md`; the
  rest are acceptable design intent. Includes an audience map
  for which server to pick for which deployment.

- **`docs/internal/reverse-audit-2026-05-25.md`** — applies the
  boundary principle in reverse: which `kglite::api::*` items
  actually only one wrapper can use? Method, findings,
  decision rule for future api additions. One demotion
  (`infer_selection_node_type`), one doc-only cleanup
  (`discover_property_keys_from_data`); the other 9 audited
  items had generic signatures and stayed.

- **`docs/internal/consider-for-future.md`** — parking-lot
  pattern for work that's been deliberately deferred. Covers
  the full dataset-lifecycle lift, retiring the Python MCP
  server, `from_blueprint` lift, SEC ticker resolution lift,
  Wikidata process cache, Selection fluent-API lift, graph
  algorithms, result streaming, the Phase 1 audit's items 2-10,
  the SEC dispatch execution loop lift, porting `explore` to
  Python MCP, lazy-load embedder in Rust MCP. Each entry has
  what / why-deferred / when-to-revisit / effort.

### Internal

- New Rust module `crates/kglite/src/datasets/blocking.rs`
  (shared `tokio::runtime::block_on` helper for sync bindings).
- New Rust modules `crates/kglite/src/datasets/sec/{blocking,
  buckets, tickers, dispatch}.rs` housing the lifted SEC
  helpers + 27 new unit tests covering all variant cases.
- New Rust module `crates/kglite/src/datasets/wikidata/
  freshness.rs` with 6 unit tests covering the decision tree.
- Python wrappers (`kglite/datasets/{sec/wrapper.py,
  wikidata.py}`) updated to delegate to the lifted core
  helpers instead of carrying their own copies. Net ~80 LOC of
  Python deleted; net ~700 LOC of core code added (more than a
  1:1 trade because the core versions carry doc + tests).

### Removed — none.

No public API removals. Items demoted from `kglite::api` stay
reachable through deeper paths (`kglite_core::graph::handle::*`,
etc.) — the demotion is a stability claim adjustment, not a code
removal.

## [0.10.1] — 2026-05-25 — Polars-style crate split (Phase G), Bolt Phase F driver-compat fixes, two-track docs, crates.io publish (kglite + kglite-bolt-server + kglite-mcp-server)

The headline of 0.10.1 is the **polars-style core split**: kglite is
now a pure-Rust crate (`crates/kglite/`, zero pyo3 in the dep tree)
publishable to crates.io as a standalone library. The Python wheel
(`pip install kglite`) is unchanged — same install line, same
Python API, same `kglite-mcp-server` console script. The wheel is
now built by a sibling PyO3 wrapper crate (`crates/kglite-py/`).

Three landings in one release:

- **Phase G** — crate split + workspace reorganization. `cargo
  install kglite-bolt-server` (no Python) now works; embedders
  depend on `kglite = "0.10"` with no PyO3 inherited.
- **Phase F** — three Bolt driver-compatibility fixes: TLS via
  `--tls-cert`/`--tls-key` (so `bolt+s://` and `neo4j+s://` work),
  `neo4j://` routing URIs via single-server routing table
  (`--advertise-addr` for reverse-proxy deploys), and Neo4j-
  conventional `db.labels()` / `db.relationshipTypes()` yield
  column names (`label`, `relationshipType`).
- **Two-track docs reorganization** — `docs/python/` and `docs/rust/`
  now live alongside `docs/operators/`, `docs/concepts/`, and the
  existing `docs/reference/`. URL breakage from the old
  `/explanation/X` paths is acknowledged; ReadTheDocs per-path
  redirects will be configured post-deploy.

Plus everything tracked in `[Unreleased]` below (validate_schema
exposure, the kglite::api::session standardization landed in 0.10.0
Phase E, the Bolt protocol C.1–C.6 implementation, polars-style
core split itself, two-track docs + cleanup of stale kglite-core
references, crates.io publish prep with parallel-bz2 decoupling).

`pip install kglite` users see no behavior change. Rust embedders
get a new option: `cargo add kglite`. Bolt-server operators get
production-grade TLS + Neo4j-driver compatibility. Wheel + Bolt-
server CHANGELOG details are below; the Rust crate's docs.rs page
will go live with the crates.io publish.

### Added — Bolt server Phase F (TLS, neo4j:// routing, db.* yield naming)

Three driver-compatibility fixes captured during the C.5 robustness
pass that needed real work to close. All land in `kglite-bolt-server`.

#### `--tls-cert` / `--tls-key` — Bolt over TLS (Phase F #6)

`bolt+s://` and `neo4j+s://` URIs now work. Drivers that require
TLS (production Neo4j deployments + most cloud setups) connect
unchanged. PEM-encoded cert chain and private key files via two
new CLI flags:

```bash
kglite-bolt-server --graph my.kgl \
    --tls-cert ./cert.pem --tls-key ./key.pem \
    --bind 0.0.0.0 --port 7687
```

Implementation: `boltr` 0.2 with the `tls` feature, `rustls = 0.23`
with the `ring` crypto provider explicitly installed at startup
(rustls 0.23+ requires the consumer to choose the provider; we
install `ring::default_provider()`), `boltr::server::TlsConfig::from_pem(...)`
wired into the existing `BoltServer::builder(...)`.

#### `neo4j://` routing URIs (Phase F #5)

Drivers that prefer the `neo4j://` scheme (most cluster-aware
clients default to it; Neo4j Browser uses it; some LangChain
flows hardcode it) make a `ROUTE` call on connect to discover the
cluster topology. We were returning `Neo.ClientError.Routing.RoutingTableNotFound`,
which made `neo4j://` URIs fail unless the user explicitly switched
to `bolt://`.

Now `BoltBackend::route()` returns a single-server routing table
with the configured address in WRITE / READ / ROUTE roles. New
`--advertise-addr HOST:PORT` flag for reverse-proxy deployments
(the address advertised in the routing table may differ from
`--bind`). When omitted, falls back to the bind address.

```bash
# Behind a reverse proxy at public.example.com:
kglite-bolt-server --graph my.kgl \
    --bind 0.0.0.0 --port 7687 \
    --advertise-addr public.example.com:7687
```

#### `db.labels()` / `db.relationshipTypes()` column names (Phase F #7)

These procedures previously yielded a column named `name` for
both, while Neo4j convention (which all drivers + dashboards
expect) is `label` for `db.labels()` and `relationshipType` for
`db.relationshipTypes()`. Tools that pre-fill schema panels from
those columns silently broke against kglite-bolt-server. Now
matches Neo4j's naming exactly. The `valid_yields` table in the
CALL clause executor was split per-procedure so the planner can
validate the right column names per call site.

### Fixed — `create_index` / `create_range_index` / `create_composite_index` returned 0 entries on reloaded `.kgl` graphs

A user-facing perf foot-gun surfaced during competitive benchmarking:
calling `create_index("Person", "ssn")` on a graph loaded from a
`.kgl` file silently returned 0 entries, even when the graph had
500k Person nodes. The `auto-rebuild on load`
(`rebuild_indices_from_keys`, called from
`crates/kglite/src/graph/io/file.rs:1818`) suffered the same bug,
so users who created indexes pre-save lost them on reload without
any error or warning. The result: `MATCH` patterns that should
hit the index fell back to full scans, with no signal.

Two root causes, both addressed:

1. **`NodeData::get_property()` only reads the in-memory snapshot.**
   For mapped/disk graphs loaded from `.kgl`, property values live
   in a backend-managed column store; `NodeData.properties` is the
   stripped-and-restored shell. The matcher's hot path
   (`core/pattern_matching/matcher.rs::node_matches_properties_columnar`)
   uses `GraphBackend::get_node_property()` instead, which dispatches
   per-backend to the right storage. The three `create_*_index`
   methods on `DirGraph` (`dir_graph.rs:815, 940, 985`) now mirror
   that path.

2. **`id` / `title` are special-cased — not in `properties` at all.**
   Their values live on dedicated `NodeData` fields and on the
   per-type id_index. Indexes on title-aliases (e.g. `name`) or
   id-aliases (e.g. `starId`) need alias resolution + the
   `get_node_id` / `get_node_title` accessors to populate
   correctly. The fix adds `resolve_alias()` + special-cased
   reads in each `create_*_index` (mirroring the matcher pattern
   at `matcher.rs:1177-1182`).

The `property_indices` HashMap is still keyed by the user-facing
property name (e.g. `name`, not `title`) because the matcher's
`try_index_lookup` (`matcher.rs:850`) looks up by the unresolved
key — keeping storage / lookup / auto-maintenance keys in lockstep.

A small follow-on fix in
`crates/kglite/src/graph/languages/cypher/executor/write.rs:558`
makes the index-maintenance old-value capture also alias-aware
for the `name` / `title` cases, so SET-on-title-alias correctly
updates an index that was built from `get_node_title`. Without
this, the auto-maintenance silently drifts.

#### Regression coverage

Six new tests in `tests/test_indexes.py::TestIndexRebuildAfterReload`
exercise the round-trip (build → save → reload → create_index/
create_range_index/create_composite_index) for: id-alias names,
the canonical `id` key, a plain property, range index, composite
index, and a composite mixing an id-alias with a plain property.
The 13 existing index-auto-maintenance tests continue to pass
(one regressed during the fix iteration; root-caused to the SET
path's old-value capture).

#### Bench infrastructure

A NornicDB shortestPath competitive bench landed at
`benchmarks/competitive/nornic/` (untracked; the new
`/benchmarks/competitive/` area is gitignored for competitive
library comparisons). It exercises both surfaces — `kglite-py`
(wheel) and `kglite` (Rust core) — against the same `.kgl` graph
on all three storage modes. The wheel side reuses the existing
`bench/nornicdb_compare.py` builder via a thin driver; the Rust
side is a small standalone Cargo project calling
`kglite::api::session::execute_read`.

Initial runs surfaced an unexplained measurement inversion (the
wheel measuring faster than the Rust core it wraps), which is
physically impossible given the architecture. Tracked as a
post-0.10.1 investigation — likely a bench harness issue
(LTO scope / first-call lazy init / per-iteration overhead)
rather than a shipped-code defect. The headline "shortestPath
~0.07-0.5ms on a 500k-node Wikidata-scale graph" numbers from
0.10.0's release notes were against the freshly-built graph
(warm caches + live index data); reload-state perf for
shortestPath specifically is ~10× slower because BFS traversal
cost is distinct from endpoint-lookup cost — endpoint lookups
for `MATCH (n:T {alias: $x})` remain fast (~1ms) on reloaded
graphs via the id_indices auto-rebuild path.

### Internal — Batch 2 kglite-py → kglite lifts (LIFT-low + HYBRID extracts)

Continuing the polars-style cleanup, lifted every remaining
non-Python-specific item from `kglite-py` into `kglite` (core).
With Batch 1 (the earlier section below) this brings the wheel
to **as thin as it can be**: everything left is genuinely
PyO3-specific (`#[pyclass]`/`#[pymethods]`, PyDict/PyList
extraction, GIL handling, Python embedder bridging).

**LIFT-low — pure-Rust items moved to core with the wheel
delegating in 1-line wrappers:**

| Item | New home | Notes |
|---|---|---|
| `field_contains_ci`, `field_starts_with_ci` | Methods on `NodeData` (`kglite::graph::schema`) | Now natural API: `node.field_contains_ci("name", &needle_lower)`. Call sites in `pyapi/kg_fluent.rs` updated. Wheel's static-method wrappers deleted entirely. |
| `discover_property_keys_from_data` | `kglite::api` (in `handle` module) | Generic property-key discovery for DataFrame/Arrow exporters. |
| `infer_selection_node_type` | `kglite::api` (free function) | Takes `(&CowSelection, &Arc<DirGraph>)` — no longer wired to the wheel's `KnowledgeGraph` struct. |
| `build_slice` (SEC) | `SliceSpec::from_optional_filters` | Constructor on the existing public type. |
| `disk_graph_age_days` (Sodir) | `Workdir::disk_graph_age_days` method | Natural API: `wd.disk_graph_age_days()`. |

**HYBRID — pure-Rust cores extracted; wheel keeps only the
PyDict/PyAny → typed-args extraction layer:**

| Item | Core API | Wheel becomes |
|---|---|---|
| `parse_inline_timeseries` | `InlineTimeseriesConfig::from_components(time_col, time_components, channels, resolution, units)` | PyDict extraction + 1-line constructor call |
| `parse_spatial_column_types` | `kglite::api::parse_spatial_column_types_from_pairs(pairs)` | PyDict → `Vec<(String, String)>` + delegation |
| `parse_temporal_column_types` | `kglite::api::parse_temporal_column_types_from_pairs(pairs)` | Same shape |
| `parse_method_param` | `MethodConfig::from_components(...)` | Per-field `extract()` + constructor call |

**Two items deliberately stay in `kglite-py`** even though they
look like LIFT candidates:
- `preprocess_values_owned` — single-variant enum wrapper
  scaffolding for the Cypher → PyAny conversion phase. No core
  to lift; the wrapper does nothing useful outside the Python
  conversion pipeline.
- `json_value_to_py` (in `mcp_tools.rs`) — the function's job
  IS `serde_json::Value` → `PyAny` conversion. The "pure-Rust
  core" would be a no-op since `serde_json::Value` already
  exists in core; only the PyO3 conversion is wheel-specific.

### Internal — Batch 1 kglite-py → kglite lifts

Architectural cleanup continuing the polars-style split. Per the
"`kglite-py` holds ONLY Python-specific code" principle, five
pure-Rust items that had been trapped in the wheel crate moved to
core. The wheel keeps 1-line `pub(crate) use ... as ...`
re-exports so the existing call sites in `graph/pyapi/*.rs` compile
unchanged.

| Item | New home | Why |
|---|---|---|
| `resolve_noderefs(graph, rows)` | `kglite::api::session::resolve_noderefs` | Post-execute cleanup — replaces `Value::NodeRef` with node titles. Every Cypher-emitting binding needs this; previously trapped in the wheel. |
| `TimeSpec` enum | `kglite::api::TimeSpec` (re-export of `graph::features::timeseries::TimeSpec`) | Pure-Rust data shape for inline timeseries config. |
| `InlineTimeseriesConfig` struct | `kglite::api::InlineTimeseriesConfig` | Same. Includes `all_columns()` helper. |
| `get_graph_mut(arc)` → `make_dir_graph_mut(arc)` | `kglite::api::make_dir_graph_mut` (defined in `graph::dir_graph`) | `Arc::make_mut` + version increment. Generic Arc<DirGraph> mutation helper for any binding. Renamed during the lift to match Rust naming conventions; wheel keeps the old name via `use ... as get_graph_mut`. |
| `merge_blueprint(base, complement, overrides)` | `kglite::datasets::sodir::merge_blueprint_json` | JSON-string deep-merge wrapper for Sodir blueprints. CLI tools and other Rust consumers that work with on-disk JSON now have a single entry point. |

All five are 100% pyo3-free; verified by `cargo tree -p kglite |
grep pyo3` (empty) and confirmed by the audit Explore agent's
transitive-dep trace. The wheel's call sites stay unchanged; the
engine logic moves once and stays in one place.

Saved as memory ([[feedback-kglite-py-python-only]]): the principle
is "kglite-py contains ONLY PyO3 bindings + Python type conversions";
engine logic belongs in `kglite` (core). Future work batches the
remaining HYBRID candidates (`parse_spatial_column_types`,
`parse_temporal_column_types`, `parse_inline_timeseries`,
`parse_method_param`, `json_value_to_py`) into 0.11.0 — each
needs a real refactor to split the pure-Rust core from the
PyDict-extraction wrapper.

### Fixed — kglite-mcp-server can now publish to crates.io (pyo3 lifted out of its dep tree)

`kglite-mcp-server`'s `Cargo.toml` previously depended on `kglite-py`
(the wheel crate, which has `pyo3` in its dep tree) because it
used `KnowledgeGraph::source_location`,
`KnowledgeGraph::set_embedder_native`, and other methods that
lived on the wheel's `#[pyclass]`-decorated `KnowledgeGraph`
struct. The binary's README claimed "No libpython link" — that
claim was false (verified via `cargo tree -p kglite-mcp-server |
grep pyo3` → showed pyo3 v0.28.3 and friends).

The lift moves the heavy logic — `source_location` (50 lines) +
`resolve_code_entity` (78-line helper) + `CODE_TYPES` const —
into a new pure-Rust module at `crates/kglite/src/graph/handle.rs`
as free functions. A new thin `kglite::api::KnowledgeGraph` struct
(2 fields: `Arc<DirGraph>` + `Option<Arc<dyn Embedder>>`) bundles
the binding-side convenience surface (`from_arc`, `dir`,
`set_embedder_native`, `embedder`, `source_location`) without the
wheel's full state (selection / reports / mutation stats /
temporal context / default timeout / max-rows).

Two types now share the `KnowledgeGraph` name across the
workspace, in different crates with different audiences — mirrors
the polars pattern (`polars::DataFrame` vs `polars.DataFrame`):

| Crate | Type | Audience |
|---|---|---|
| `kglite` (core) | `kglite::api::KnowledgeGraph` | Rust embedders. Pure-Rust. 2 fields. No pyo3. |
| `kglite-py` (wheel) | `kglite_py::KnowledgeGraph` | Python users via `pip install kglite`. PyO3-decorated. 8 fields. The wheel's heavy methods now delegate to the core's free functions for single-source-of-truth. |

mcp-server's `Cargo.toml` switches its `kglite` dep from
`{ package = "kglite-py", path = "../kglite-py" }` to
`{ version = "0.10", path = "../kglite" }`. Mcp-server's source
needed **zero changes** — every `kglite::api::*` import resolves
to the same item under the new dep, just routed through core's
api mod instead of the wheel's re-exports. Verified:

```bash
cargo tree -p kglite-mcp-server | grep pyo3     # → empty
```

`publish = false` removed from
`crates/kglite-mcp-server/Cargo.toml`. The crate publishes to
crates.io alongside `kglite` and `kglite-bolt-server` in this
release — three crates in the same 0.10.1 publish cycle,
orchestrated by `.github/workflows/publish_crates.yml`.

Net effect: `cargo install kglite-mcp-server` works without a
Python runtime, matching the binary's README claim.

The wheel's `KnowledgeGraph::source_location` /
`KnowledgeGraph::resolve_code_entity` methods stay on the
PyO3-decorated struct for back-compat with Python callers via
`#[pymethods]`, now implemented as 1-line delegates to the
core's free functions. `kg_fluent::find_one`'s
`Self::CODE_TYPES` reference now points at
`kglite_core::graph::handle::CODE_TYPES`. No behavior change.

### Added — crates.io publish prep for the Rust crates

The pure-Rust `kglite` core crate (and the standalone
`kglite-bolt-server` binary) are now metadata-complete for
publishing to [crates.io](https://crates.io).

- **`crates/kglite/`** — added `readme = "README.md"`, `repository`,
  `homepage`, `documentation = "https://docs.rs/kglite"`,
  `keywords = ["graph", "knowledge-graph", "cypher", "petgraph",
  "database"]`, `categories = ["database", "data-structures"]`.
  New `crates/kglite/README.md` (~140 lines) tailored for the
  crates.io audience.
- **`crates/kglite-bolt-server/`** — same metadata pattern, with
  Bolt/Neo4j-flavored keywords. Version bumped `0.0.1 → 0.10.1` to
  align with the wheel's 0.10.x line. New
  `crates/kglite-bolt-server/README.md`.
- **`crates/kglite-mcp-server/`** — metadata + README written but
  marked `publish = false` for now. The crate still depends on
  `kglite-py` (for `KnowledgeGraph::set_embedder_native` /
  `source_location` methods that live on the PyO3 wrapper type);
  publishing must wait until those methods get lifted into the
  core. Version bumped to 0.10.1 anyway for local consistency.
- **`crates/kglite-py/`** — explicitly marked `publish = false`
  with a comment explaining: the wheel is the right artifact for
  Python users (`pip install kglite`), and Rust users want the
  no-pyo3 path via the sibling `kglite` crate.

`cargo publish -p kglite --dry-run` passes: 267 files, 5.1 MiB
(1.1 MiB compressed). `cargo publish -p kglite-bolt-server
--dry-run` succeeds once `kglite` is actually published (chicken-
and-egg; the bolt server depends on the core crate that has to
land first).

#### parallel-bz2 decoupling

In the process, the single-stream bz2 parallel-decode path was
restructured so `cargo publish -p kglite` resolves cleanly
against crates.io.

Background: the path uses the paolobarbolini bzip2-rs git fork
(adds `ParallelDecoderReader` + `ThreadPool` trait + a
`RayonThreadPool` helper gated on the fork's `rayon` Cargo
feature). The crates.io `bzip2-rs = 0.1.x` release has none of
these — only the sequential `DecoderReader`. Cargo's publish-time
manifest resolver requires every declared dep feature to be
satisfiable against crates.io versions, which previously broke
`cargo publish`.

Fix:
- New optional Cargo feature `kglite/parallel-bz2`. Default-off;
  enables the optional `bzip2-rs` dep.
- Implemented `bzip2_rs::ThreadPool` ourselves
  (`graph::io::ntriples::parallel_bz2::KglRayonPool`) on top of
  kglite's existing `rayon` dep instead of using
  `bzip2_rs::RayonThreadPool`. Removes the requirement for the
  fork's `rayon` cargo feature from the published manifest.
- Workspace `[patch.crates-io]` pulls the fork during local
  development. The patch is stripped on `cargo publish`; crates.io
  consumers who enable `kglite/parallel-bz2` need their own
  matching patch until upstream bzip2-rs publishes a 0.2.x with
  these APIs.
- Single-stream fallback when `parallel-bz2` is off is sequential
  `bzip2::read::MultiBzDecoder`. Multi-stream pbzip2 parallelism
  is unaffected.
- The `wikidata` Cargo feature implies `parallel-bz2` (Wikidata
  ingest is the workload that needs it).

All 11 `parallel_bz2` unit tests pass on both feature-on and
feature-off configurations. The `bz2_bench` binary requires
`--features parallel-bz2`.

### Changed — Docs reorganized into two-track Python / Rust layout

[kglite.readthedocs.io](https://kglite.readthedocs.io) now
groups content by audience instead of by document type. The
existing `docs/explanation/` is gone; everything moved to one
of five top-level tracks:

| Track | What's there | Audience |
|---|---|---|
| `docs/python/` | `getting-started`, `core-concepts`, `transactions`, `error-handling`, `value-projection`, all of `guides/`, `examples/`, `migrations/` | Wheel users (`pip install kglite`) |
| `docs/rust/` | `index` (Rust quickstart), `embedding`, `session`, `api-reference` | Rust embedders depending on the `kglite` crate directly |
| `docs/operators/` | `bolt-server` | Operators deploying `kglite-bolt-server` or `kglite-mcp-server` |
| `docs/concepts/` | `architecture`, `design-decisions`, `concurrency`, `cypher-conformance`, `multi-label-rationale`, `adding-a-storage-backend`, `adding-a-query-language` | Contributors, curious users wondering "why is it built this way" |
| `docs/reference/` | `cypher-reference`, `fluent-api`, auto-generated Python API (unchanged location) | Cross-binding reference |

`docs/index.md` rewrites as a track selector (one entry per
track) and the per-track `index.md` files act as navigators
into their own contents.

**URL breakage warning.** Bookmarks of the form
`kglite.readthedocs.io/en/latest/explanation/<X>.html` no longer
resolve. The new paths are:

| Old path | New path |
|---|---|
| `/explanation/transactions` | `/python/transactions` |
| `/explanation/error-handling` | `/python/error-handling` |
| `/explanation/value-projection` | `/python/value-projection` |
| `/explanation/embedding-kglite` | `/rust/embedding` |
| `/explanation/session` | `/rust/session` |
| `/explanation/bolt-server` | `/operators/bolt-server` |
| `/explanation/architecture` | `/concepts/architecture` |
| `/explanation/design-decisions` | `/concepts/design-decisions` |
| `/explanation/concurrency` | `/concepts/concurrency` |
| `/explanation/cypher-conformance` | `/concepts/cypher-conformance` |
| `/explanation/multi-label-rationale` | `/concepts/multi-label-rationale` |
| `/getting-started` | `/python/getting-started` |
| `/core-concepts` | `/python/core-concepts` |
| `/guides/<X>` | `/python/guides/<X>` |
| `/examples/<X>` | `/python/examples/<X>` |
| `/migrations/<X>` | `/python/migrations/<X>` |
| `/adding-a-storage-backend` | `/concepts/adding-a-storage-backend` |
| `/adding-a-query-language` | `/concepts/adding-a-query-language` |

ReadTheDocs per-path redirects will be configured via the RTD
admin UI post-deploy to keep stale bookmarks working.

All file moves used `git mv` for blame/history preservation. The
Sphinx build (with `-W` warnings-as-errors at warning gates
that already cleared in CI) is green; pytest 3013+1, bolt 236+3,
and `make lint` are unaffected.

### Fixed — Stale `kglite-core` / `kglite_core` references after the G.4 rename

G.4 renamed the core crate from `kglite-core` to `kglite` but
the in-flight G.5 work (examples + embedder doc) was authored
before that rename and never updated. This swept up the
leftovers:

- **`crates/kglite/examples/embedded_{basic,session,blueprint}.rs`**
  and **`crates/kglite/tests/datasets_{sec_idx_parser,sec_fetch_live,sodir_fetch_live}.rs`**
  — `kglite_core::*` imports and `cargo run -p kglite-core`
  doc-comment invocations updated to `kglite::*` / `-p kglite`.
  Examples now compile (`cargo build -p kglite --release
  --examples`) and run; the `embedded_session` OCC sequence
  correctly rejects Transaction B with `ConflictDetected`.
- **In-code module doc comments** in
  `crates/kglite/src/{lib,datatypes/mod,graph/io/file,graph/languages/cypher/mod,code_tree/mod,code_tree/builder/mod,code_tree/builder/load}.rs`
  no longer describe the crate as "kglite-core" or "Currently
  named `kglite-core` to avoid a workspace conflict" (the
  conflict was resolved by the rename — that paragraph was
  historical noise).
- **`crates/kglite-py/src/**`** — clarifying comments on the
  `kglite_core = { package = "kglite", ... }` dep alias (the
  alias dodges the extern-crate collision with kglite-py's
  own `[lib] name = "kglite_py"`; the engine itself is the
  `kglite` crate).
- **`ROADMAP.md`** + **`bolt_implementation.md`** updated to
  reference `kglite::api::*` (the post-G.4 surface) rather than
  the historical `kglite_core::*`.

CHANGELOG entries for the Phase G journey itself are preserved
unchanged with an upstream "Note on naming" caveat — the
references to `kglite-core` in those entries describe the
journey faithfully.

### Internal — Polars-style core split (Phase G of `bolt_implementation.md`)

> **Note on naming.** The first commits of Phase G used the
> temporary package name `kglite-core` to avoid a workspace
> conflict with the then-existing root `kglite` crate. The G.4
> commit (`5eecf51`) renamed it to `kglite` and relocated the
> pyo3 wrapper to `crates/kglite-py/`. The references to
> `kglite-core` below describe the journey faithfully; the
> end-state crate is named `kglite` everywhere now.

The Rust core moves out of the wheel crate into a pure-Rust
sibling crate at `crates/kglite/` (initially package-named
`kglite-core`, renamed to `kglite` in G.4). The PyO3 wrapper
sits at `crates/kglite-py/` and depends on the core via
`kglite = { path = "../kglite", ... }`.

**Why** — Polars precedent: kglite's engine has always been pure
Rust, but pyo3 was an unconditional dep of the only crate that
held it. Rust embedders (anyone wanting kglite as a graph
library without the Python wheel) inherited pyo3's build
complexity. The split fixes that.

**End-state verified by `cargo tree`:**
- `cargo tree -p kglite-core | grep pyo3` → **empty** ✓
- `cargo tree -p kglite-bolt-server | grep pyo3` → **empty** ✓ (switched to `kglite = { package = "kglite-core" }` direct dep)
- `cargo tree -p kglite-mcp-server | grep pyo3` → still present (uses `KnowledgeGraph::source_location` etc. that live in the pyo3 wrapper; cleanup deferred)
- `cargo tree -p kglite | grep pyo3` → present (this is the wheel — expected)

**Highlights:**
- **Dataset crates merged** — `crates/kglite-{sec,sodir,wikidata}/`
  folded into `crates/kglite/src/datasets/{sec,sodir,wikidata}/`
  behind features (`sec`, `sodir`, `wikidata`). Workspace down
  from 7 → 4 members. Polars-io pattern: opt in to dataset
  loaders only when you use them.
- **117 KgError→PyErr sites converted** to use a
  `kg_to_pyerr()` helper, fixing the orphan-rule violation that
  would otherwise block the move (`impl From<KgError> for PyErr`
  becomes invalid once KgError lives outside the wrapper crate).
- **Visibility bumps on `DirGraph`** — ~23 `pub(crate)` fields
  + 6 helpers (`resolve_node_property`, `MethodConfig`, etc.)
  promoted to `pub` for cross-crate access. Pragmatic "wide
  public" choice over a ~25-method accessor refactor; tracked
  as a follow-up.
- **Embedder examples + binding-implementer guide** —
  `crates/kglite/examples/embedded_{basic,session,blueprint}.rs`
  run cleanly with `cargo run -p kglite-core --example …`. New
  `docs/explanation/embedding-kglite.md` walks through the
  surface, the .kgl portability story, and sketches cgo / napi
  / JNI wrappers for future bindings.
- **`pip install kglite` unchanged for Python users.** Same
  wheel, same Python API, same `kglite-mcp-server` console
  script. The split is invisible from PyPI's side.

**Verification (~12 minutes wall-clock):**
- `cargo build --workspace --release` green (~3min)
- `cargo run -p kglite-core --example embedded_session` works
- `cargo run -p kglite-core --example embedded_blueprint` works
- `pytest tests/` → 3013 + 1 skipped (unchanged)
- `pytest tests/ -m bolt` → 233 + 3 skipped (unchanged)
- `pytest tests/ -m bolt_stress` → 9 (unchanged)
- `make lint` clean

### Internal — `kglite::api::session` standardization (Phase E of `bolt_implementation.md`)

Single source of truth for the canonical Cypher pipeline and the
snapshot/working CoW transaction model. The same orchestration
previously lived in three places (pyapi `cypher()`, mcp-server
`run_cypher_inner`, bolt-server `KgliteBackend`) and the CoW
machinery in two — drift had already cost correctness twice in
this sprint (`validate_schema` missing from mcp/bolt; bolt's lazy-
RETURN bug from accidentally including `mark_lazy_eligibility`).
Phase E extracts both into `kglite::api::session::*`, then rewrites
all three consumers as thin wrappers.

- **New module `src/graph/session/`** — `Session`, `Transaction`,
  `CommitOutcome`, `ExecuteOptions`, `ExecuteOutcome`,
  `execute_read`, `execute_mut`. Pure Rust, no PyO3, no async. 13
  unit tests pin the contract (snapshot isolation, working CoW
  materialization, OCC conflict detection, read-only enforcement,
  no-writes commit fast path).
- **OCC now enforced in bolt-server** (closes limitation #1 of the
  7 captured during the C.5 robustness pass). Concurrent writers
  whose snapshots become stale see
  `ClientError("Transaction conflict: graph was modified by
  another committer ... Retry the transaction.")` on commit. The
  bolt-stress concurrency tests now use a retry-on-conflict
  pattern that mirrors what real clients should do.
- **pyapi, mcp-server, bolt-server consumers rewritten** to wrap
  the session module. `kg_core::cypher` body shrank from ~280 to
  ~80 lines; `Transaction.cypher` from ~150 to ~40; mcp's
  `run_cypher_inner` from ~80 to ~30; bolt-server's backend
  pipeline from ~325 to ~150. Net ~440 lines new (session +
  docs), ~735 lines deleted across consumers.
- **Foundation for future bindings**. A Go binding (cgo) or
  TypeScript (napi) or JVM (JNI) is now a marshalling layer over
  `session::execute_*` + `Session` / `Transaction` handles — the
  pipeline + CoW + OCC are solved once.
- **`docs/explanation/session.md`** — binding-implementer guide
  covering the API surface, snapshot isolation guarantees, the
  per-binding concurrency model, and a sketch of how to wrap from
  cgo. Read this if you're integrating kglite into a new language
  runtime or transport.

No user-visible behavior change for the Python `cypher()` /
`Transaction` surface (3013 tests pass unchanged). The bolt
contract change — OCC-enforced commits returning ClientError on
conflict — is intentional (the pre-E behavior was last-writer-wins
with no warning, captured as limitation #1).

### Internal — Bolt protocol scaffolding (Phase B of `bolt_implementation.md`)

Pre-implementation scaffolding for the Bolt v5.x wire-protocol server.
No user-visible surface yet; this lands the crate skeleton, the failing-by-design
test contract, and the perf baseline that Phase C sub-phases will retire.

- **New crate `crates/kglite-bolt-server/`** with `clap` CLI (`--graph`,
  `--bind`, `--port`, `--readonly`, `--auth`, `--idle-timeout`,
  `--max-sessions`) and a `BoltBackend` impl whose 11 trait methods all
  panic with `unimplemented!("phase C.X — ...")`. The binary boots,
  loads a `.kgl` graph, binds a TCP port via `boltr::BoltServer::serve`,
  and panics the per-connection task on the first real Bolt message.
  Compiles green; the trait signatures pin the boltr v0.2.0 API
  surface against kglite's `kglite::api::*` types.
- **New `tests/test_bolt_server_smoke.py`** — 8 `xfail(strict=True)`
  tests using the `neo4j` Python driver. Each is tagged with the
  Phase C sub-phase that retires it (C.1 handshake, C.2 scalars,
  C.3 parameters, C.4 Node/Rel, C.5 transactions/readonly, C.6 auth
  + FAILURE mapping). Strict mode means accidentally fixing a test
  out-of-phase turns XFAIL into XPASS → FAIL — the alert that the
  decorator should be removed.
- **New `pyproject.toml` marker `bolt`** — Bolt-protocol smoke tests
  excluded from the default `pytest tests/` run via `addopts`; opt-in
  via `pytest -m bolt`. Mirrors the existing `binary_size` /
  `parity` pattern.
- **New benchmarks `test_bench_return_node_10k` +
  `test_bench_return_node_rel_node_100`** in
  `tests/benchmarks/test_bench_core.py`. Cover the `Value::Node`
  projection paths that Phase A.1 added and Phase C.4 will route
  over Bolt PackStream. Baseline capture deferred to the next
  release commit (Phase D, or whichever 0.10.x ships first) via
  `make refresh-release-constants`.
- **CI**: `cargo build --release` now also builds `-p kglite-bolt-server`,
  the Python install line gains the `[neo4j]` extra, and a dedicated
  `pytest -m bolt` step runs the failing contract on every push.

This is **prep work**, not a feature release. No `Cargo.toml`
version bump.

### Added — `validate_schema` extended to CREATE/MERGE patterns + exposed in api

Two changes folded into one user-visible improvement:

**Fortification.** `src/graph/languages/cypher/planner/schema_check.rs`
previously skipped CREATE and MERGE clauses (explicit `=> {}` arm),
even though those clauses' pattern-literal property names are exactly
the same "unambiguously a property name" shape that MATCH validates.
Now extended:

- `CREATE (:Person {ttle: 'Alice'})` — catches the typo with
  "Unknown property 'ttle' on Person.<did_you_mean>".
- `CREATE (a:Person {age: 30})-[:KNOWS]->(b:Person {agee: 25})` —
  walks multi-element paths.
- `MERGE (:Person {agee: 30})` — same path via `MergeClause.pattern`.
- ON CREATE SET / ON MATCH SET still skip (use `SetItem`, deferred).

Zero false positives preserved — same gate as the MATCH path:
`validate_property` skips when the type has no declared metadata.
Tests grew from 13 to 21.

**Exposure.** User flagged a real gap: the Python boundary
(`src/graph/pyapi/kg_core.rs`) has called `validate_schema` between
parse and optimize since 0.9.x to catch property typos in pattern
literals — but the pure-Rust `kglite-mcp-server` and (newly added)
`kglite-bolt-server` were both missing the pass. The function was
internal-only.

- **`kglite::api::cypher::validate_schema`**: new `pub use` in
  `src/lib.rs`.
- **`crates/kglite-mcp-server/src/tools.rs`**: adds the call right
  after `parse_cypher`. Error mapped to `String` (matches the
  existing mcp-server error pipeline).
- **`crates/kglite-bolt-server/src/backend.rs::execute`**: adds the
  call as pipeline step 2 (renumbered the rest). Error mapped to
  `BoltError::Protocol` (genuine client error — bad property name
  → `Neo.ClientError.Request.Invalid` on the wire). Distinct from
  the `BoltError::Backend`-mapped "feature pending" errors C.2/C.3
  use for slices we haven't shipped yet.

All four downstream Cypher consumers (Python `cypher()`, MCP
`cypher_query`, Bolt `execute`, the `tests/test_schema.py` agent
helper via `KnowledgeGraph.validate_schema()`) now share the same
hardened pre-flight check.

### Added — `kglite::api::cypher::validate_schema` exposed; both pure-Rust servers now run it (superseded)

(See the section above; this entry kept as a pointer for git
log archaeology — superseded by the fortification work that
landed in the same commit.)

User flagged a real gap: the Python boundary (`src/graph/pyapi/kg_core.rs`)
has called `validate_schema` between parse and optimize since 0.9.x
to catch property typos in pattern literals (`{ttle: 'Alice'}` when
only `title` exists on `Person`) — so users see "Unknown property
`ttle` on type `Person` — did you mean `title`?" instead of silently
getting zero rows. The pure-Rust `kglite-mcp-server` and (newly
added) `kglite-bolt-server` were both missing this pass.

- **`kglite::api::cypher::validate_schema`**: new `pub use` in
  `src/lib.rs` (was only reachable via the internal `crate::graph::
  languages::cypher::*` path, which the api docs explicitly warn
  downstream consumers away from).
- **`crates/kglite-mcp-server/src/tools.rs`**: adds the call right
  after `parse_cypher`. Error mapped to `String` (matches the
  existing mcp-server error pipeline).
- **`crates/kglite-bolt-server/src/backend.rs::execute`**: adds the
  call as pipeline step 2 (renumbered the rest). Error mapped to
  `BoltError::Protocol` (genuine client error — bad property name
  → `Neo.ClientError.Request.Invalid` on the wire). Distinct from
  the `BoltError::Backend`-mapped "feature pending" errors C.2/C.3
  use for slices we haven't shipped yet.

All three downstream Cypher consumers (Python `cypher()`, MCP
`cypher_query`, Bolt `execute`) now behave identically wrt schema
validation. Bolt smoke contract still reports `3 passed, 5 xfailed`;
MCP smoke still reports 32 passed.

### Internal — Bolt protocol C.6 (typed FAILURE codes + `--auth basic` + `db.*` verified)

Final sub-phase of Phase C. All 8 smoke tests now pass; the bolt-
server can stand in for a Neo4j instance for the broad happy-path
contract (handshake, auth, scalar reads, parameterized queries,
graph-structure returns, explicit transactions, --readonly, typed
FAILURE codes, schema-introspection procs).

- **New module `crates/kglite-bolt-server/src/error_map.rs`**:
  `kg_to_bolt(KgError) -> BoltError::Query { code, message }` with a
  16-arm mapping table from `KgErrorCode` to `Neo.{Class}.{Category}.{Title}`
  status codes:
  - `CypherSyntax` → `Neo.ClientError.Statement.SyntaxError`
  - `CypherTimeout` → `Neo.ClientError.Transaction.TransactionTimedOut`
  - `CypherTypeMismatch` → `Neo.ClientError.Statement.TypeError`
  - `CypherExecution` → `Neo.DatabaseError.Statement.ExecutionFailed`
  - `Schema` → `Neo.ClientError.Schema.ConstraintValidationFailed`
  - `Validation` / `Expr` / `InvalidArgument` → `Neo.ClientError.Statement.ArgumentError`
  - `NodeNotFound` / `ConnectionNotFound` / `PropertyNotFound` → `Neo.ClientError.Statement.EntityNotFound`
  - `MissingArgument` → `Neo.ClientError.Statement.ParameterMissing`
  - `FileNotFound` / `FileFormat` / `FileIo` / `Internal` → `Neo.DatabaseError.General.UnknownError`
  - 2 unit tests pin the table shape (every code maps to a
    4-segment `Neo.*` string; the SyntaxError case is asserted
    verbatim).
- **`crates/kglite-bolt-server/src/backend.rs::execute`**:
  `parse_cypher` errors now route through `kg_to_bolt` instead of
  the `BoltError::Backend(e.to_string())` fallback. Other error
  sources (`rewrite_text_score`, `CypherExecutor::execute`,
  `execute_mutable`) still return `String` from kglite — when those
  paths gain typed errors in a future refactor, they'll auto-flow
  through the same mapper.
- **New module `crates/kglite-bolt-server/src/auth.rs`**:
  `BasicAuthValidator` implements the boltr `AuthValidator` trait.
  Checks scheme + principal + credentials against the CLI's
  `--auth-user` / `--auth-pass`; rejects with `BoltError::Authentication`
  (maps to `Neo.ClientError.Security.Unauthorized`).
- **`crates/kglite-bolt-server/src/main.rs`**: wires
  `BasicAuthValidator` into `BoltServer::builder().auth(...)` when
  `--auth basic` is selected. `--auth none` (default) leaves no
  validator wired — boltr handles LOGON SUCCESS itself, accepting
  any credentials.
- **`kglite::api::{KgError, KgErrorCode}`** newly exposed
  (`src/lib.rs`). The Python boundary already used them; bolt-
  server now does too.
- **`db.*` schema-introspection procs (`db.labels`, `db.relationshipTypes`,
  `db.indexes`)**: verified to work via the standard Cypher CALL
  pipeline. No bolt-server changes needed — Phase A.3 added the
  procs to kglite's executor, they're routed through the existing
  parse-plan-execute path, and they return scalar rows that the
  C.2 `to_bolt` arms handle directly.
- **Test contract**: `xfail` removed from
  `test_bolt_returns_failure_on_parse_error`. `pytest -m bolt -v`
  now reports **`8 passed, 0 xfailed`** (exit code 0). Strict-mode
  contract retired cleanly — every test turned green on exactly
  the sub-phase it was tagged for.

### Internal — Bolt protocol C.5 (BEGIN/COMMIT/ROLLBACK + --readonly)

Fifth sub-phase of Phase C. Explicit transactions work end-to-end:
`session.begin_transaction()` → `tx.run("CREATE ...")` →
`tx.commit()` (or `tx.rollback()`). The `--readonly` CLI flag rejects
mutations at both the auto-commit boundary and the begin_transaction
boundary.

- **`crates/kglite-bolt-server/src/backend.rs`**: significant
  restructure of `KgliteBackend`:
  - Storage changed from `Arc<KnowledgeGraph>` to
    `Arc<Mutex<Arc<DirGraph>>>` — outer mutex allows commits to
    swap the inner Arc; inner Arc allows readers to cheaply
    snapshot via Arc::clone.
  - New `transactions: Arc<Mutex<HashMap<String, TxState>>>` map
    tracks per-transaction state.
  - `TxState` mirrors `src/graph/pyapi/transaction.rs`'s
    snapshot/working CoW shape: `snapshot: Option<Arc<DirGraph>>`
    + `working: Option<DirGraph>`. First mutation materializes
    working via `Arc::try_unwrap` (free when this tx holds the
    only ref) or deep clone.
  - `begin_transaction` snapshots the current Arc<DirGraph>, mints
    a `tx-{N}` handle, stores TxState. Rejects under `--readonly`.
  - `commit` swaps the working Arc into the backend's shared graph
    (no-op if no mutations occurred — read-only-then-commit
    transactions are cheap).
  - `rollback` drops TxState (working copy discarded).
  - `close_session` / `reset_session` roll back all in-flight
    transactions for the session.
- **Auto-commit mutations remain rejected** with a `BoltError::Backend`
  error pointing at "wrap in BEGIN/COMMIT". Drivers always wrap
  writes in explicit transactions in practice; adding auto-commit
  mutations would broaden the surface for no real win.
- **`--readonly` enforcement**: `begin_transaction` returns
  `BoltError::Forbidden` ("server is read-only — explicit
  transactions rejected"), which maps to
  `Neo.ClientError.Security.Forbidden` on the wire → driver raises
  `ClientError`. Auto-commit mutations also return `Forbidden` when
  `--readonly` is on (vs `Backend` when it isn't).
- **`execute` pipeline refactored** into `plan` + `execute_auto_commit`
  + `execute_in_tx` helpers on `KgliteBackend`. Per-query mutex
  hold is bounded; reads outside tx are wait-free apart from a
  single Arc::clone.
- **SUCCESS metadata**: now includes `stats` dict (nodes-created,
  relationships-created, properties-set, etc.) when the result
  carries `MutationStats`. Reads still emit `type: "r"`; mutations
  emit `type: "w"`.
- **OCC version checking deferred**. `DirGraph.version` is
  `pub(crate)` and not exposed via `kglite::api`. The Python
  `Transaction` class uses it; bolt-server gets it when the
  accessor is added. For C.5 the test scenarios are sequential so
  no conflict is possible; concurrent-writer stress is a Phase D
  consideration.
- **`kglite::api::cypher::CypherQuery`** newly exposed
  (`src/lib.rs`) — bolt-server needs the type to write its
  pipeline helper. The mcp-server and Python boundary already
  use it via the internal path.
- **`crates/kglite-bolt-server/src/main.rs`**: backend constructor
  takes `DirGraph` (not `Arc<KnowledgeGraph>`); `Arc::try_unwrap`
  on the loaded KG's inner Arc is free in the typical boot path.
- **Test contract**: `xfail` removed from
  `test_bolt_transaction_commit_and_rollback` and
  `test_bolt_rejects_writes_when_readonly`. The latter test gains
  a dedicated `bolt_server_readonly` fixture that spawns its own
  `--readonly` server instance. `pytest -m bolt -v` now reports
  `7 passed, 1 xfailed` (exit code 0). Only test #8 (parse-error
  → ClientError mapping) remains — scoped to C.6.

### Internal — Bolt protocol C.4 (Node / Relationship / Path RETURN)

Fourth sub-phase of Phase C. Cypher queries returning graph
structures now round-trip over Bolt — `RETURN n` materializes as
a `neo4j.graph.Node` in the Python driver, `RETURN r` as
`neo4j.graph.Relationship`, and (via the Path encoding scheme)
`RETURN p = (...)-[*]-(...)` returns a `neo4j.graph.Path`.

- **`crates/kglite-bolt-server/src/value_adapter.rs::to_bolt`**:
  graph-structure arms become real, replacing the Phase C.2
  `Err(BoltError::Backend("phase C.4 ..."))` stubs:
  - `Value::Node(node)` → `BoltNode { id: i64, labels, properties,
    element_id: id.to_string() }`. `element_id` is the stringified
    integer id (stable within one server lifetime — the contract
    drivers care about; drivers shouldn't persist `element_id`
    long-term).
  - `Value::Relationship(rel)` → `BoltRelationship { id, start_node_id,
    end_node_id, rel_type, properties, element_id, start_element_id,
    end_element_id }`. All `*element_id` fields stringify the
    numeric ids.
  - `Value::Path(p)` → `BoltPath { nodes, rels: Vec<BoltUnboundRelationship>,
    indices }`. The `indices` field encodes the Neo4j path-traversal
    scheme: pairs of (signed-1-based-rel-index, 0-based-next-node-index)
    where sign = direction (+ outgoing, - incoming relative to path
    traversal). Direction inferred by comparing `rel.start_id` /
    `rel.end_id` against the surrounding node ids.
- New helpers in `value_adapter.rs`: `props_to_bolt_dict`
  (recursive via `to_bolt`); `path_to_bolt_path` (handles the
  indices arithmetic + tracing-logs corrupt paths where a rel
  doesn't connect its surrounding nodes).
- **`kglite::api`**: now exposes `NodeValue`, `RelValue`, `PathValue`
  alongside `Value` (`src/lib.rs`). Downstream Rust consumers (the
  bolt-server's path encoder, future Arrow/Polars exporters) can
  pattern-match the carriers without re-deriving accessors.
- **Test contract**: `xfail` removed from
  `test_bolt_return_node_yields_node_struct` and
  `test_bolt_return_relationship_yields_rel_struct`. `pytest -m bolt -v`
  now reports `5 passed, 3 xfailed` (exit code 0). Only tests #6
  (BEGIN/COMMIT), #7 (`--readonly` enforcement), and #8 (parse-error
  → ClientError mapping) remain — all scoped to C.5 and C.6.

`Value::NodeRef(_)` still returns an error (it's an internal executor
placeholder that should never reach the boundary; leaking it
indicates a kglite bug).

### Internal — Bolt protocol C.3 (parameter PackStream decoding)

Third sub-phase of Phase C. The Bolt server now accepts parameterized
queries — `session.run("MATCH (n:Person {city: $c}) RETURN n.title", c="Oslo")`
works against a `bolt://` driver.

- **`crates/kglite-bolt-server/src/value_adapter.rs::from_bolt`**:
  replaces the `unimplemented!()` stub. Scalar arms
  (Null/Bool/Integer/Float/String) + recursive List/Dict + temporal
  (Date → `Value::DateTime` via epoch arithmetic) + Duration + Point2D
  (SRID 4326 only). Non-representable inbound types surface as
  `BoltError::Protocol` (which maps to `Neo.ClientError.Request.Invalid`
  on the wire — these are genuine client errors, distinct from the
  `BoltError::Backend` / `Neo.DatabaseError.*` "feature pending"
  pattern that C.2 established).
- **`crates/kglite-bolt-server/src/backend.rs::execute`**: the
  empty-params gate is gone; parameters now flow through
  `value_adapter::from_bolt` into the executor's `&kg_params` map.
- **Rejected inbound types** (each with a structured error message):
  Bytes (no kglite `Value` variant), Time/LocalTime/DateTime/
  DateTimeZoneId/LocalDateTime (kglite has date-only precision —
  Phase A.1 deferred time precision), Point3D (kglite is 2D only),
  Node/Relationship/Path/UnboundRelationship (drivers shouldn't pass
  these as params anyway).
- **Test contract**: `xfail` removed from
  `test_bolt_run_supports_parameters`; `pytest -m bolt -v` now
  reports `3 passed, 5 xfailed` (exit code 0).

### Internal — Bolt protocol C.2 (read-only RUN/PULL with scalar values)

Second sub-phase of Phase C. The Bolt server now runs real Cypher
queries end-to-end for scalar-returning reads. `verify_connectivity`
+ `session.run("MATCH (n:Person) RETURN n.title AS name")` works
against a `bolt://` driver; mutations, parameters, and Node/Rel
returns still fail by design.

- **`crates/kglite-bolt-server/src/backend.rs::execute`**: replaces
  `unimplemented!()` with the canonical kglite Cypher pipeline
  (mirrors `kg_core.rs::cypher` / `kglite-mcp-server/src/tools.rs`):
  parse → rewrite_text_score → optimize_with_disabled →
  mark_lazy_eligibility → mutation gate → `CypherExecutor::with_params
  (dir, &params, None).with_streaming(false).execute(&parsed)`.
- **`crates/kglite-bolt-server/src/value_adapter.rs::to_bolt`**:
  signature changed from `BoltValue` to `Result<BoltValue, BoltError>`
  (graph-structure arms must not panic mid-connection — they
  orphan tokio tasks). All 10 scalar variants now real:
  Null/Bool/Int64/UniqueId/Float64/String + recursive List/Map +
  Date/Duration/Point. Node/Relationship/Path return a structured
  `Err(BoltError::Backend("phase C.4 ..."))`.
- **CLI surface**: gates non-empty parameters (`Phase C.3`), explicit
  transactions (`Phase C.5`), Cypher mutations (`Phase C.5`), and
  text_score queries (`Phase D`) with clean `BoltError::Backend`
  messages — each maps to `Neo.DatabaseError.General.UnknownError`
  on the wire, so tests #3-#8's `pytest.raises(ClientError)` checks
  don't catch them and the strict-xfail contract holds.
- **`crates/kglite-bolt-server/Cargo.toml`**: adds `chrono` as a
  direct dep (was transitive via kglite); needed for `Value::DateTime`
  → `BoltDate` arithmetic (days-since-Unix-epoch).
- **Test contract**: `xfail` removed from
  `test_bolt_run_returns_scalar_rows`; `pytest -m bolt -v` now
  reports `2 passed, 6 xfailed` (exit code 0).
- **bolt_implementation.md**: Phase C summary row updated to
  `C.1, C.2 ✅ Shipped · C.3–C.6 pending`; C.2 sub-section heading
  flipped + body rewritten to reflect what shipped.

The server is now a usable thin Bolt frontend for scalar-only
read-only Cypher. SUCCESS metadata includes `type: "r"` + `t_last`
(elapsed ms). The lazy-result-descriptor streaming path is forced
off (`.with_streaming(false)`) for simplicity; revisit in Phase D
if profiling demands it.

### Internal — Bolt protocol C.1 (handshake + session lifecycle)

First sub-phase of Phase C. The Bolt server is now *connectable* —
the neo4j Python driver's `verify_connectivity()` runs end-to-end
against `kglite-bolt-server`. Queries still panic per the strict-
xfail contract; that's C.2 onward.

- **`crates/kglite-bolt-server/src/backend.rs`**: replace 6 of the 11
  `unimplemented!()` stubs:
  - `create_session` — generates `bolt-{N}` handles via an
    `AtomicU64` counter (no UUID dep needed; SessionManager only
    needs uniqueness within one server process).
  - `get_server_info` — returns honest `server: "kglite-bolt-server/{version}"`
    + `bolt_agent` dict; boltr auto-injects `connection_id` + `hints`.
  - `set_session_auth` — no-op (only called once C.6 wires an
    `AuthValidator`; right now boltr handles LOGON SUCCESS itself).
  - `close_session` / `reset_session` / `configure_session` — no-op
    + debug log. No per-session state until C.5 brings transactions.
  - `route` — tightened from `unimplemented!()` to a structured
    `BoltError::Protocol` ("connect with `bolt://` not `neo4j://`")
    so accidental routed-client connections fail cleanly instead of
    panicking the connection task.
- **Test contract**: the `xfail(strict=True)` decorator on
  `test_bolt_handshake_and_verify_connectivity` is removed; the test
  now PASSES. The other 7 stay XFAIL — they exercise RUN / BEGIN
  which still trigger panicked `execute` / `begin_transaction`.
  `pytest -m bolt -v` now reports `1 passed, 7 xfailed` (exit code 0).
- **bolt_implementation.md**: Phase C row gains a C.1 ✅ sub-status.

Server identity is honest, not Neo4j-mimicking. The `server` field
reads `kglite-bolt-server/0.0.1`; if any Phase D ecosystem tool turns
out to require a `Neo4j/<x.y>` prefix, we'll add a `--neo4j-compat`
CLI flag then — pre-emptive lying isn't on the menu.

## [0.10.0] — Phase A (Bolt prep): Value variants + KgError + db.* procedures + audit

The foundation for the Bolt protocol server (`ROADMAP.md` §1). Three
sub-phases shipped as bisectable commits; user-visible surface area is
Cypher type-system completeness, a typed Python exception hierarchy,
and the Neo4j-canonical schema-introspection procedures. Plus the
post-A.3 "are we ready?" audit: fixture rebuild for the v3→v4 hard
break, transaction + concurrency hardening tests, and binding-implementer
docs for Phase B/C.

### Added — `Value::{Node, Relationship, Path, List, Map}` variants (A.1)

The Cypher `Value` enum now carries the full openCypher type lattice
natively, replacing the prior JSON-string round-trip for compound
values. Lists no longer serialise through `String` shapes on the way
out; UNWIND fast-paths a native `Value::List`; the columnar exporter
gets per-row typing without re-parsing.

Per-row execution shaves a few µs on list-heavy queries
(test_bench_cypher_where -16%, columnar -17.5%). The
`Box<NodeValue>`-tax on common-case matches shows up as
test_bench_cypher_match +14.4%; within ±20% policy budget.

### Added — `kglite.KgError` taxonomy + typed Python exceptions (A.2)

Pre-0.10.0 every error from kglite arrived as a built-in Python
exception with a `format!` string body. Now:

- New `kglite.KgError` base class + 17 typed subclasses
  (`CypherSyntaxError`, `CypherTimeoutError`,
  `CypherExecutionError`, `SchemaError`, `ValidationError`,
  `FileError`, `ArgumentError`, etc.). Hierarchy descends from
  `kglite.KgError → Exception`; the Cypher subtree extends
  `kglite.CypherError` for narrower catches.
- Cypher syntax errors carry `line` and `col` as struct fields
  (preserved through the parser → boundary route rather than
  embedded only in the message).
- ~80 `PyErr` sites across the Python surface migrated to the
  typed exceptions (kg_core, kg_mutation, kg_introspection,
  algorithms, kg_fluent, py_in, mcp-server).
- See `docs/explanation/error-handling.md` for the full reference.

**Breaking.** PyO3's `create_exception!` is single-inheritance, so the
typed exceptions extend `KgError`, NOT also the built-in equivalents.
`except ValueError:` / `except RuntimeError:` no longer catches kglite
errors — migrate to `except kglite.CypherSyntaxError:` (or the
universal `except kglite.KgError:`). See the migration table in
`docs/explanation/error-handling.md`.

### Added — `CALL db.labels()` / `db.relationshipTypes()` / `db.indexes()` (A.3)

The Neo4j-canonical schema introspection procedures, callable from any
Bolt-compatible client (cypher-shell, Neo4j Browser, Python `neo4j`
driver):

- `CALL db.labels() YIELD name` — every node-type name, sorted.
- `CALL db.relationshipTypes() YIELD name` — every connection-type
  name, sorted.
- `CALL db.indexes() YIELD name, type, entityType, labelsOrTypes,
  properties, state` — every installed index with structured columns.

The parser now accepts namespaced procedure names (`db.labels`,
`apoc.coll.sum`); previously only single-identifier names parsed.
Procedure names are case-insensitive on dispatch (Neo4j convention).

A KGLite-specific extension: `db.indexes()` returns `type='RANGE'`
for B-tree range indexes (Neo4j collapses them under `'PROPERTY'`).
The planner uses the distinction — an equality index can't serve a
range query — so the procedure surfaces it for index advisors.

Behind the scenes, the new procedures share `pub(crate)` Rust
helpers (`collect_labels`, `collect_relationship_types`,
`collect_indexes_structured`) with `describe()` and `schema()` so
all introspection surfaces report identical data.

### Performance — Pre-Bolt audit + targeted fixes

The pre-Bolt audit identified five performance issues. Four shipped
fixes; the fifth was investigated and documented (root cause is the
A.1 Value enum expansion — a known trade-off).

**Issue #1 — Deferred clone on `begin()`.** The biggest Bolt win in
this release. `begin()` used to deep-clone the entire DirGraph up
front (O(graph_size)); now it takes an Arc snapshot (O(1)) and
defers the clone until the first mutation lands. Read-only-then-
commit transactions pay no clone cost regardless of graph size.

| Graph | `begin() + commit()` *(no writes)* | Before | After |
|---|---:|---:|---:|
| 1k nodes | 40 µs | → | **166 ns** (~240× faster) |
| 10k nodes | 391 µs | → | **166 ns** (~2,400× faster) |
| 100k nodes | 4.16 ms | → | **166 ns** (~25,000× faster) |

A mutating transaction still pays the clone cost on first
`tx.cypher("CREATE ...")` (~30 µs / k nodes), unchanged.

**Issue #2 — Query parse cache.** `cypher()` used to re-parse every
input string from scratch. Added an LRU cache (256-entry, FIFO
eviction, RwLock-protected) at
`src/graph/languages/cypher/parse_cache.rs`. Cache HIT is ~700 ns
end-to-end vs ~1.4 µs uncached — a 50% reduction for the
Bolt-typical "agent re-issues the same parameterized query in a
hot loop" pattern.

**Issue #3 — Concurrent-read scaling.** Audit found the read path
plateaus at ~5.8× speedup beyond 8 threads. On Apple M4 this is
hardware-bound (4 perf + 6 efficiency cores). Targeted fix: moved
`resolve_noderefs` (pure-Rust post-execution step) into the
`py.detach` block so it runs GIL-free, improving per-thread
efficiency by 2-3 percentage points. The remaining inefficiency is
heap allocator contention + minor GIL re-acquisition on PyObject
construction, both system-wide rather than kglite-specific. On
homogeneous x86 server CPUs, scaling extends further than M4
permits.

**Issue #4 — `columnar_enable` regression investigated.** The A.3
release reported a +27% capture variance; rigorous re-measurement
shows the actual regression vs 0.9.52 is +2.7% — within noise.
The minor slowdown is consistent with the documented A.1 Value
enum expansion (`cypher_match` +1.9%). `enable_columnar` is a
one-time setup operation, not in the hot Bolt path.

**Issue #5 — Documentation.** New "Performance reference" sections
in `docs/explanation/transactions.md` and
`docs/explanation/concurrency.md` document the post-fix numbers,
the M-series CPU plateau, and the Bolt-server design implications.
`scripts/perf_audit.py` is the re-runnable audit harness for these
numbers.

### Performance — NornicDB shortestPath benchmark: kglite is now 10–250× faster

Replicated NornicDB's published 500K-node `shortestPath` benchmark
(`scripts/nornicdb_compare.py`, faithful port of
`pkg/cypher/demo_shortest_path_largescale_test.go`). Pre-fix run on the
500K Star / 4M HYPERLANE fixture exposed two compounded bottlenecks
in the hot Cypher path: a quiet **35 ms floor on every shortestPath
call** + a **500 K × 500 K cartesian product when prior bindings
weren't propagated to the shortestPath executor.** Combined with the
trivial-but-disastrous "did you forget to call create_index()?" trap
when the indexed field is the id-alias, the unfixed pipeline timed
out at 10 s per call past depth 3.

**Three targeted fixes shipped:**

1. **Pre-bind propagation in `execute_shortest_path_match`** —
   `MATCH (a {id: X}), (b {id: Y}) MATCH p = shortestPath((a)-[*]-(b))`
   used to re-resolve `(a)` and `(b)` as bare patterns inside the
   shortestPath executor, which (correctly) returned all 500K nodes
   for each, then ran a 250-billion-pair cartesian-product BFS. Now
   the executor reads bindings from the input ResultSet on the fast
   path; only bare-pattern shortestPath callers fall through to the
   pre-fix cartesian behaviour. Extracted into a new
   `executor/shortest_path.rs` submodule.

2. **Id-alias routing in `try_index_lookup`** — when the user calls
   `add_nodes(df, "Star", "starId", "title")`, `starId` becomes the
   ID-field alias for the canonical `id`. Pre-fix, parameterized
   `MATCH (s:Star {starId: $a})` queries fell through to a full
   500K-node type scan because the matcher only special-cased the
   literal property names `id` / `nid` / `qid`. The matcher now
   consults the type's declared id-alias and routes through
   `lookup_by_id_readonly` — O(1) lookup on the auto-maintained
   per-type id_index. No `create_index` call needed.

3. **HashMap-backed BFS state in `reconstruct_path_bfs`** — the BFS
   used to allocate `Vec<bool>` (500 KB) + `Vec<u32>` (2 MB) +
   `VecDeque` (~1 MB) per call sized to `node_bound`, regardless of
   actual traversal scope. For a 1-hop visit (~16 nodes) the
   alloc/init cost (~30 ms) dominated the operation. Now uses a
   `HashMap<usize, u32>` for parent tracking (presence ⇔ visited);
   shallow paths pay µs of alloc, deep paths pay O(visited_nodes).
   Tradeoff: ~50% slower per-node visit cost for very deep BFS
   (HashMap hash vs Vec index); on the NornicDB benchmark at d=60
   this still beats the pre-fix code by 7× and beats NornicDB by
   >100×.

A `VecDeque + HashMap` (no overhead at small N) is a saner default
than `Vec[node_bound]` for the realistic mix of shallow + medium
BFS that real Bolt clients issue. Kept the `Vec<bool>` path on
`shortest_path_directed` (less exercised; reverting the directed
version's HashMap change had no measurable benefit).

**Reverted (no impact / regression):** an earlier id-alias fix to
`create_index` itself was reverted after breaking
`test_set_name_updates_index` — the matcher-side routing covers the
agent-typical case without needing the index-builder change.

**Final NornicDB vs kglite (Apple M4 vs M3 Max, 500K nodes / 4M edges):**

| Depth | kglite med | NornicDB med | Speedup |
|---:|---:|---:|---:|
| 1   | 3.9 µs     | 94.8 µs       | **24×** |
| 5   | 122.9 µs   | 1.23 ms       | **10×** |
| 10  | 418.7 µs   | ~15.5 ms      | **37×** |
| 20  | 1.27 ms    | ~31.1 ms      | **24×** |
| 40  | 2.97 ms    | 612.7 ms      | **206×** |
| 60  | 5.70 ms    | >612.7 ms     | **>100×** |

Graph build is **49× faster** on top of the per-query wins (4.25 s
vs NornicDB's 3 m 32 s, BadgerDB-flush-dominated).

### Refactor — Code-rot cleanup (pre-Bolt Tier 1 audit)

Post-Bolt-audit code-rot review found light rot concentrated in
duplicated helpers + one stale dead-code marker + an inbox folder
holding 41 processed interproject messages. All cleaned in this
release; god-file refactors (Tier 2) deferred — the existing files
are working code, just large.

- **Duplicated helpers consolidated.** Five copies of
  `file_to_module_path` (dart/html/php/swift/css) and three copies of
  `make_qualified` (dart/php/swift) — all byte-identical except for
  the separator character — moved to
  `code_tree/parsers/shared.rs` with a `separator: char` parameter.
  Three byte-identical copies of `sanitize_filename` in
  `blueprint/compute/{derive,chain,filter}.rs` consolidated into the
  shared `blueprint/compute/mod.rs`. The `yield_alias` helper in
  `cypher/executor/{affected_tests,refresh_stats}.rs` moved to the
  shared `executor/helpers.rs`. Three slightly-different copies of
  `value_to_string` (`graph/mod.rs`, `graph/explore.rs`,
  `graph/io/export.rs`) consolidated into a single canonical
  `crate::datatypes::values::raw_string` using the most complete of
  the three impls (rich variant coverage for `DateTime`, `Point`,
  `Duration`, A.1 collection variants). Two copies of
  `default_auto_vacuum_threshold` consolidated as `pub(crate)` in
  `dir_graph.rs`. Net: ~80 LoC deleted, no behavior change.

- **Stale dead-code marker removed.** `subgraph_streaming.rs`'s
  `#[allow(dead_code)] // Phase 4 consumes every method.` marker is no
  longer needed — Phase 4 ships and `pyapi/algorithms.rs` consumes
  `RankIndex::from_bitset` + `kept_count`. Removed the marker and the
  one method that genuinely was unused (`Bitset::bitset` accessor).
  (`unified_columns::WriteResult`'s marker is legitimately still
  needed; its fields are reserved for a future caller and the marker
  stays with a refreshed comment.)

- **Inbox cleanup.** `inbox/read/` (41 messages, 476 KB of processed
  MCP-methods / MCP-servers interproject correspondence) deleted —
  git history preserves them. `tests/fixtures/build_fixtures.py`'s
  origin-attribution comment updated to drop the dead URL reference.

- **God-file size gate.** New `tests/test_god_files.py` rejects any
  `src/**/*.rs` file over 3000 LoC unless it has an entry in an
  explicit `ALLOWLIST` with a pinned ceiling and justification.
  Current state: one allowlisted file
  (`cypher/planner/fusion.rs` at 3028 LoC, pinned at 3050) — the
  optimizer-fusion pass registry, on the deferred Tier 2 split list.
  Companion `test_allowlist_is_not_stale` ensures the allowlist gets
  pruned when files shrink below the default. The gate exists to
  catch the NEXT file that drifts past 3000 — making
  CLAUDE.md's *"each pass should leave it more compartmentalised"*
  guidance mechanical.

### Net Phase A regression matrix (0.9.52 → 0.10.0 with audit fixes)

| Benchmark | 0.9.52 | 0.10.0 | Δ |
|---|---:|---:|---:|
| `shortest_path` | 2.6 µs | 1.3 µs | **−49.7%** |
| `cypher_where` | 251 µs | 185 µs | **−26.3%** |
| `columnar_cypher_where` | 261 µs | 200 µs | **−23.1%** |
| `columnar_cypher_match` | 4.9 µs | 4.6 µs | −5.6% |
| `save_v3` | 401 µs | 394 µs | −1.7% |
| `traversal` | 348 ns | 339 ns | −2.5% |
| `cypher_match` | 4.5 µs | 4.6 µs | +1.9% |
| `columnar_enable` | 196 µs | 201 µs | +2.7% |
| `add_connections` | 489 µs | 500 µs | +2.3% |
| `add_nodes` | 243 µs | 253 µs | +4.2% |

Phase A net: 6 benchmarks meaningfully faster, 4 marginally slower
(all within noise / known A.1 trade-offs). No tracked benchmark
regresses past +5%.

### Added — Pre-Bolt audit: fixtures + tests + docs

The "are we really done preparing kglite core for Bolt?" audit
surfaced three working-but-unverified concerns; the audit work itself
ships here.

- **Test fixtures rebuilt for v4 format.** Phase A.1's `.kgl` v3→v4
  hard break invalidated 4 committed binary fixtures
  (`spatial_graph.kgl`, `timeseries_graph.kgl`,
  `graph_with_orphans.kgl`, `graph_with_duplicates.kgl`). New
  `tests/fixtures/build_fixtures.py` regenerates them deterministically
  (`random.seed(42)`); 8 previously-xfailed MCP tests in
  `tests/test_mcp_server_python_entry.py` (j1/j2/j3/k1/k2/k3/l1/l2)
  now pass. 1 spurious "empty parametrize" SKIPPED in
  `tests/test_cypher_differential.py` is now an intentional
  `@pytest.mark.skipif` with a self-documenting reason.
- **Transaction class typed-exception sweep.** A.2 missed
  `src/graph/pyapi/transaction.rs`; this release migrated 15 PyErr
  sites to typed `kglite.KgError` subclasses. Bolt server bindings
  now see uniform error types from transaction operations (timeout
  → `kglite.CypherTimeoutError`; OCC conflict + read-only mutation
  + double-commit → typed `kglite.KgError`).
- **Bolt-shaped tests pinned.** New
  `tests/test_transaction_bolt_patterns.py` (18 tests) pins the
  BEGIN → cypher × N → COMMIT/ROLLBACK flow, snapshot isolation,
  OCC conflict semantics, context-manager auto-commit/rollback,
  read-only enforcement, and timeout behavior.
- **Bolt-scale concurrency stress.** New tests in
  `tests/test_concurrency.py` (`TestBoltScaleConcurrency`,
  `TestDocumentedQuirks`) cover 16-thread parallel readers and
  32-thread reader+mutator contention without panic, plus pinned
  contracts for the WKT cache write-lock and Arc::make_mut CoW
  isolation quirks.
- **Binding-implementer documentation.** New
  `docs/explanation/transactions.md` and
  `docs/explanation/concurrency.md` document the surface Bolt's
  Phase C will consume — error → FAILURE-code mapping table,
  per-session Arc<KnowledgeGraph> recipe, the two documented
  contention quirks with rationale.

### Verification

- Test suite: 3010 passed (+61 from 0.9.52's 2949), 1 skipped, 8
  warnings. The 0.9.52 baseline had 1 skipped + 8 xfailed; this
  release flips all 8 xfailed → pass and converts the 1 skipped
  to an intentional self-documenting skipif.
- Cross-mode parity (memory/mapped/disk) preserved for all 3 phases.
- `make lint` green across the release (fmt + clippy + ruff +
  stubtest).
- Benchmark gate: 11 tracked benchmarks within the ±20% policy
  budget.

### What this unlocks

Phase B of `bolt_implementation.md` (Bolt server skeleton + failing
test contract) can now start. Phase C.6 (Bolt FAILURE-code mapping
+ db.* pass-through) gates on this release's typed exceptions and
procedures, both now shipped. The deferred streaming wrapper for
Bolt PULL will land in Phase B/C alongside the protocol code itself.

## [0.9.52] — Cypher NULL semantics, batch dedup, shortestPath dedup

### Fixed — Three-valued NULL logic in WHERE predicates

KGLite's WHERE evaluator collapsed Cypher three-valued NULL logic
to boolean at the predicate boundary, producing silent wrong rows
in two patterns. Both are openCypher violations.

- `WHERE x <> 'literal'` now correctly excludes rows where `x` is
  missing. Before this fix, `NotEquals(NULL, 'literal')` returned
  `true`, so missing-property rows were kept.
- `WHERE NOT (x CONTAINS 'lit')` (and the `STARTS WITH` / `ENDS WITH`
  variants) now correctly excludes rows where `x` is missing.
  Before this fix, `NULL CONTAINS x` was `false` and `NOT false` was
  `true`, keeping the rows.
- Kleene `AND` / `OR` / `XOR` composition is correct: NULL only
  propagates when no absorbing element is present.
- `Predicate::Not(None)` is `None`, not flipped to `true`.

External `evaluate_predicate` callers (HAVING, OPTIONAL MATCH filter,
list comprehensions, spatial joins) keep their `Result<bool, _>`
contract — the internal `evaluate_predicate_tristate` does the
NULL-aware composition and the wrapper collapses `None` to `false`
(which every external caller already treated as "drop the row").

### Fixed — `labels(n)` JSON escape

The hand-rolled escape in `scalar_functions::"labels"` covered only
`\\` and `"`. Labels containing control characters or non-ASCII
escapes could produce invalid JSON, breaking the Python deserializer.
Switched to `serde_json::to_string` for the encoding side; the
consumer (`parse_list_value`) was already symmetric. No user-visible
change for ASCII-only labels — this hardens the edge cases.

The two call sites (`scalar_functions.rs::"labels"` and
`helpers.rs::parse_list_value`) carry a Track-C swap-point note: when
`Value::List` lands they're the first sites to migrate. See
`docs/explanation/multi-label-rationale.md`.

### Fixed — Undirected `shortestPath` neighbour dedup

`filtered_neighbors_undirected` concatenated outgoing and incoming
neighbours into one `Vec` without deduplicating. Bidirectional edge
pairs (a→b plus b→a) and parallel edges of the same type each
surfaced the same neighbour twice. BFS-based `shortestPath` was
saved by its visited bitmap, but `all_paths` (DFS) paid duplicate
work per visit.

In-place sort + dedup after collection. Insertion order isn't
load-bearing for any caller (BFS / DFS use set-membership, not order).

### Performance — Per-chunk HashMap dedup in batch flush

`ConnectionBatchProcessor::flush_chunk` called
`graph.edges_connecting(src, tgt).find(...)` per edge to detect
existing edges of the same connection type — O(degree(src)) per
edge. For hub-source fan-out into an *existing* connection type
(the `skip_existence_check=false` path), a chunk of N edges from
a hub of degree D ran in O(N·D).

`flush_chunk` now builds a single
`HashMap<(NodeIndex, NodeIndex), EdgeIndex>` at the top of the chunk
keyed on the unique source set's outgoing edges of the target
connection type, then probes the map per edge. The map is mutated
as edges are created (preserving within-chunk consolidation
semantics) and updated on Replace (so later iterations hit the new
edge id, not the removed one).

Microbenchmark on 5 hubs × 10k targets = 50k edges added to an
existing `:R` connection type: scales linearly with N (≈1.4s
wall-clock for the fan-out add, ≈0.5s for a re-add in Update or
Skip mode), instead of the prior O(N·D) curve.

### Added — Multi-label decision doc

`docs/explanation/multi-label-rationale.md` captures the
investigation: KGLite stays single-label by design, the v3
columnar layout is keyed by primary type, and the motivating
multi-label use cases (Wikidata, code-tree refinements) are
already served by `INSTANCE_OF` / `KIND_OF` edges. Lists three
stepping-stone helpers (`Value::List`, subtype-edge planner
rewrite, `GraphRead::node_types_of` shim) that lower the future
cost without committing to the full multi-label implementation.

### Added — Differential corpus coverage

`tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES` gains 12
entries covering the shapes the above fixes address (NULL
comparisons, NULL through string predicates under NOT, Kleene
composition, `labels()` consumers, undirected `shortestPath`).
The corpus is the regression guard against silent wrong-row bugs;
the additions lock in the new behaviour.

### Internal — Cleared 9 pre-existing parity-test failures

Housekeeping pass on `pytest -m parity` — every failure pre-dated
0.9.52 and none were caused by the work above. Fixed in bulk so
the release lands a clean gate:

- Phase 2 — restrict CREATE/MERGE-via-cypher tests to memory+mapped
  (disk lockout has been intentional since 0.9.26); replaced the
  legacy "disk MERGE works" test with two dedicated lockout-message
  guards.
- Phase 4 — refresh the `.kgl` v3 golden digest. The hash embeds the
  version string in the header, so every release shifts it; the
  allowlist hadn't been updated since 0.9.7.
- Phase 5/6 — extend the `GraphBackend::` enum-match whitelist to
  cover `mutation/subgraph_streaming.rs` (disk-to-disk streaming
  filter) and `pyapi/algorithms.rs` (disk-only PyO3 entry points).
  Both are structural peers of existing whitelist entries.
- Phase 7 — extract `column_store.rs`'s 293-line test block to a
  sibling `column_store_tests.rs` via `#[path]`, dropping the
  production file from 2515 to 2228 lines (under the 2500-line
  god-file cap). Add `// SAFETY:` comments to 5 mmap fadvise /
  env-var-cleanup blocks (each was correct as written; only the
  justification was missing).
- Phase 5 — bump `test_binary_size_regression` baseline from the
  stale 0.9.0 value (23.5 MB) to the current 0.9.52 size (35.9 MB).
  Gate stays +10% on the new baseline. The docstring carries an
  explicit "what grew" breakdown so the next bump is grounded —
  primary growth contributors over the 0.9.0 → 0.9.52 window were
  the 14 tree-sitter grammars, the fastembed feature default-on
  for the kglite-mcp-server binary, mcp-methods evolution, and
  the sodir / wikidata workspace crates.

All 98 parity-marked tests now pass.

### Fixed — Four more Cypher correctness fixes (from the fortified suite)

A test-suite fortification pass (see Internal section below) surfaced
three additional bugs in the Cypher engine plus one discoverability
inconsistency. All four are fixed:

- `IN` predicates now propagate NULL per openCypher. Completes the
  tri-state work the 0.9.52 B1/B2 fix started: `WHERE x IN
  [literal, ...]`, `WHERE x IN $param`, and the `InLiteralSet`
  fast-path all return NULL on NULL LHS or no-match-with-NULL-element.
  Pre-fix, those rows leaked through `NOT (x IN [...])` and similar
  shapes.
- `list[..]` parses as a full-range slice (both ends omitted). Was a
  parser asymmetry — `[start..]` and `[..end]` worked, `[..]` errored.
- `Int64::MIN` (`-9223372036854775808`) is now expressible as a
  literal. Tokenizer-level lookback: when the digit string overflows
  i64 *and* is exactly `9223372036854775808` *and* the previous token
  is `Dash`, the pair collapses to a single `IntLit(i64::MIN)`.
- `keys(n)` enumerates the user-set unique-id-field and title-field
  column names (e.g. `person_id`, `name`) alongside the virtual
  aliases (`id`, `title`, `type`). Discoverability fix: `n.person_id`
  was readable but absent from `keys(n)`.

### Internal — Test-suite fortification

The 0.9.52 release surfaced a structural gap in the test
infrastructure: parity gates were excluded from default CI, perf
benchmarks tracked but didn't gate, and several captured constants
drifted silently across releases. Six-commit overhaul:

- **Structural parity gates → default CI** (god-file cap,
  unsafe-needs-SAFETY, mod_rs purity, enum-match audit, recording
  symbol export). Previously opt-in via `-m parity`; 10 had drifted
  to red on `main` before this session cleared them.
- **Perf-regression gate** (`scripts/compare_bench.py`,
  `tests/benchmarks/baselines/0_9_52.json`, CI `perf-regression`
  job). Blocks PRs on >20% min-time regression against the
  versioned baseline. Threshold matches the CLAUDE.md performance
  protocol.
- **Captured-constants refresh ritual**
  (`scripts/refresh_release_constants.py`,
  `make refresh-release-constants`, CLAUDE.md section). One script
  refreshes the `.kgl` golden digest, binary-size baseline, and perf
  baseline at release time. Idempotent; only the version-tagged
  baseline triggers re-capture. Pre-existing stale `.kgl` golden
  digest fixed in the same commit.
- **+25 openCypher edge-case tests** + 1 differential corpus query
  covering NULL semantics, unicode strings, numeric boundaries,
  collection edges, aggregate NULL handling, and pattern-matching
  edges (self-loops, zero-length paths, parallel edges). Surfaced
  the three Cypher bugs above + one pre-existing parser bug
  (`Int64::MIN` literal) that's also now fixed.
- **On-demand Neo4j conformance runner**
  (`scripts/cypher_conformance.py`, `make neo4j-{up,down,conformance}`,
  `docs/explanation/cypher-conformance.md`). Standalone — not in CI,
  no external service dependency for the regular test run. Reuses
  the differential corpus + shared fixtures.
- **Property-naming round-trip pins** (`test_export.py`). Six tests
  documenting `d3` export flattening, alias-table semantics, and
  `to_neo4j` renaming. Caught the `keys(n)` discoverability bug
  fixed above.

Test count: `pytest tests/` goes from ~2786 (pre-session) to **2826
passed / 1 skipped / 0 xfailed**. `pytest -m parity` stays 98/98
green. `make lint` clean.

## [0.9.51] — Dart / Flutter code-tree support

### Added — Dart language parser

`kglite.code_tree` now parses `.dart` sources, bringing the supported-
language count to 14. A Flutter repository — previously indexed as a
periphery-only graph (native-runner Swift/C++/C scaffolding, the
marketing site) with its entire Dart application core silently dropped
— now produces a queryable graph of the real app.

- Classes, mixins, extensions, enums, top-level and member functions,
  named & factory constructors, getters/setters/operators, constants
  and typedefs — emitted with the same node/edge vocabulary as the
  other languages, so cross-language Cypher patterns just work.
- `extends` / `with` / `implements` → `EXTENDS` / `IMPLEMENTS` edges;
  call sites → `CALLS`; methods → `HAS_METHOD`; cyclomatic branch
  counts and structured parameters as for every other language.
- Named and factory constructors resolve to distinct, addressable
  qualified names (`Owner.Owner`, `Owner.Owner.named`).
- `import` / `export` directives → `IMPORTS` edges (relative and
  same-package URIs); `part` / `part of` files collapse into one
  logical module.

### Added — `Mixin` node label

Dart `mixin` declarations land as a dedicated `Mixin` graph node,
beside the existing `Class` / `Struct` / `Trait` / `Protocol` family.
`extension` / `extension type` are `Class` nodes tagged by `kind`.

### Added — Flutter widget pass

`StatelessWidget` / `StatefulWidget` / `State` subclasses carry a
`flutter_widget` property, and their `build` methods a `flutter_build`
flag — "show me the screens" is one hop away. New queryable `Function`
columns: `is_constructor`, `is_factory`, `accessor`.

### Fixed — `github_api` leading-slash 404

Bumped the `mcp-methods` dependency to 0.3.39, which fixes `github_api`
malforming its URL for a path written with a leading slash
(`/repos/owner/repo` → doubled `/repos/` → 404). A leading slash is now
optional and accepted on either path form.

### Fixed — comment-annotation char-boundary panic

`extract_comment_annotations` panicked when a TODO/FIXME comment body
exceeded 200 bytes with a multi-byte character straddling the
truncation boundary — reachable from every language parser, surfaced
by Dart comments that use `────` rules.

## [0.9.50] — Lossless edge loading: auto-vivified provisional stub nodes

### Lossless edge loading — auto-vivified provisional stub nodes

- An edge loaded against a node that doesn't exist is no longer
  silently dropped. The missing endpoint is auto-vivified as a
  *provisional* stub node (marked `_provisional`) so the edge always
  connects — across blueprint fk-edges, blueprint junction-edges and
  the imperative `add_connections` API. This removes a load-order
  hazard: loading edges before some of their nodes (e.g. `Friends`
  before `Class B`) previously lost every edge into the not-yet-loaded
  nodes.
- A later load of the real node row **promotes** the stub — the
  `_provisional` marker is cleared on node upsert.
- New `KnowledgeGraph.purge_provisional()` deletes any stub never
  promoted (a genuinely dangling reference) and its incident edges,
  returning `{nodes_purged, edges_removed}`.
- A blueprint can set `settings.auto_purge: true` to run that purge
  automatically at the end of `from_blueprint` (default `false` —
  stubs are kept so no edge is lost).
- `_provisional` is a reserved property name — a blueprint node spec
  declaring it is rejected.

### SEC loader — ownership / 13F / Exhibit 21 extraction scoped to filing_index

- These three extractors walked `raw/filings/` directly, parsing every
  cached document regardless of the build's `form_types` / year scope
  — whereas `Filing` nodes come scope-filtered from `filing_index.csv`.
  On a re-scoped rebuild of an existing workdir the mismatch produced
  detail rows (`InsiderTransaction`, `Holding`, `InstitutionalHolding`,
  `Subsidiary`) referencing out-of-scope filings. They now walk via
  `walk_filings_in_index` — only documents whose filing is in
  `filing_index.csv` — so extraction and the `Filing` node set agree.

## [0.9.49] — SEC loader: form-typed extraction + 3-phase build progress

### SEC loader — per-filing documents selected by form type, not filename

- The form extractors (DEF 14A, SC 13D/G, S-1, 424B, 10-K — and 8-K)
  picked their documents with filename predicates that required a
  form-type token (`def14a`, `sc13d`, …). Modern inline-XBRL filings
  are named `{ticker}-{date}.htm` and carry no such token, so on
  recent filings those extractors silently produced nothing
  (`Compensation`, `Proposal`, `ActivistFiling`, … came out empty).
  They now resolve each document's form type by accession against
  `filing_index.csv` (`walk_filings_of_form`) — reliable regardless of
  filename. Verified live: TSLA 8-K events (37) and DEF 14A
  compensation + proposals now extract from 2025-2026 filings.

### SEC loader — minimalist 3-phase build progress

- `SEC.open` / `SEC.fetch` render a minimalist 3-phase tqdm display —
  **Fetch** (per-filing download, live count), **Process** (extraction)
  and **Build** (graph assembly) — replacing the per-form-bucket bars.
  The verbose `[SEC] …` lines are muted whenever a progress display
  (or a caller-supplied `progress` callback) is active, so the terminal
  shows the three phases and nothing else. Falls back to the plain
  `[SEC]` prints when `tqdm` isn't installed.

## [0.9.48] — SEC loader: cold-start, Jupyter progress, 8-K extraction

- `SEC.fetch` / `SEC.open` on a fresh workdir now collect per-filing
  detail in one call. The per-filing dispatcher reads
  `processed/filing_index.csv` — emitted by the extractor's identity
  pre-pass — which didn't exist yet on a cold workdir, so the dispatch
  fetched nothing and the graph had `Company`/`Filing` nodes but no
  insider transactions, events, or roles. The wrapper now builds the
  filing index before the dispatch and re-extracts after it.
- The per-filing fetch releases the GIL during the rate-limited
  download loop, re-acquiring it only to fire each progress event.
  Holding it for the whole batch starved a Jupyter kernel's IOPub
  thread, so tqdm progress couldn't render until the call returned.
- 8-K events are now extracted from modern inline-XBRL filings. The
  extractor's file predicate required `8k` in the filename, but
  recent 8-K primary documents are named `{ticker}-{date}.htm` — so
  `CorporateEvent` nodes were silently empty. The predicate is now
  loose (the `Item N.NN` parser self-gates), and the event
  description stops at the heading sentence instead of running on
  into the filing body (inline-XBRL has no newlines).

## [0.9.47] — SEC EDGAR value-prop upgrade + blueprint compute pipeline

Two big additions land in this release. The first (J0–J7) overhauls
the SEC loader so the detailed-payload extracts actually work and
the resulting graph rewards SQL-person traversal patterns. The
second (K1–K7) introduces a top-level `compute:` block in blueprints
— a small ETL pipeline that runs as a CSV-shaping pre-phase, so
loaders can do unit conversions, conditional flags, temporal chains,
calendar joins, and summary nodes declaratively instead of via
Python pre-scripts or post-build Cypher passes.

### Blueprint compute pipeline (K1–K7)

- **`compute:` block in blueprints** — top-level ordered list of named
  primitives that runs as a CSV-shaping pre-phase before the existing
  5-phase loader. Each primitive writes its outputs to
  `computed/*.csv` and the declarative loader consumes them as if
  they were ordinary inputs.
- **Expression language** — hand-rolled Pratt parser + tree-walking
  evaluator: arithmetic, comparison, logical, membership, function
  calls, list literals. Built-in functions: math (`abs/round/ceil/
  floor/sqrt/log/exp/pow/min/max`), string (`concat/lower/upper/
  contains/starts_with/ends_with/len`), conditional (`if/coalesce`),
  type conversion (`int/float/string`), date components (`year/
  month/day/quarter`). No expression-crate dependency.
- **Five compute primitives**:
  - `derive` — row-level expressions on an existing node type; new
    property columns appended or overwriting existing ones.
  - `filter` — `where` predicate; produces a new derived type
    (`into:`) or rewrites the source destructively.
  - `chain` — group + sort + emit consecutive-pair junction edges
    with `step_index` property.
  - `calendar` — synthesises `Date` nodes for `[start, end]` plus
    `NEXT_DAY` chain edges and `ON_DATE`-style link edges to source
    types' date columns.
  - `aggregate` — group-by + per-group aggregate expressions
    (`sum/avg/min/max/count/count_distinct/first(...,by=)/
    last(...,by=)`), emitting one summary node per group plus
    optional FK edges to the group-key targets.
- **Validation up-front** at blueprint load: dangling type/column
  references, malformed expressions, aggregate-only functions
  outside `aggregate.agg`, calendar date ordering — all caught
  before any CSV is touched.
- **Performance**: aggregate uses `HashMap<String, ...>` with a
  reused String buffer for the group key (one allocation per *new*
  group, not per row). 100K rows / 1K groups / 6 aggregates runs in
  68.7 ms end-to-end (full blueprint load including Phase 1-5).
- **Sub-node resolution** — compute primitives target both top-level
  types and sub-nodes (e.g. SEC's `Transaction` at
  `nodes.Person.sub_nodes.Transaction`). The resolver walks
  `blueprint.nodes` first, then each parent's `sub_nodes`.
- **SEC blueprint showcase**: the dataset's packaged blueprint
  ships with a `compute:` block exercising all five primitives —
  `derive` (filing_year, form-type flags on Filing; total_value,
  is_buy, is_sell on Transaction), `filter` (AnnualRevenue from
  MetricFact), `chain` (NEXT_FILING per company; NEXT_TX per
  person+issuer), `calendar` (2020-2030 + ON_FILED_DATE and
  ON_TX_DATE links), and `aggregate` — chained in two stages for
  insider positions:
  - **`PositionLedger`** (per ledger, group_by `[person_nid,
    issuer_cik, security_title, direct_indirect]`) captures the
    fact that Form 4's `shares_owned_after` is a per-(security,
    direct/indirect) balance, not a global one. Has
    `current_shares=last(shares_owned_after, by=transaction_date)`
    plus shares_acquired/disposed, first/last_tx_date,
    n_transactions, and filed-price total_buy/sell_value. Edges
    LEDGER_OF_PERSON / LEDGER_AT_COMPANY.
  - **`Position`** rolls PositionLedger up to one row per
    (person, issuer) by summing across ledgers — current_shares,
    shares_acquired, shares_disposed, n_transactions, n_ledgers,
    total_buy/sell_value — and taking min/max of first/last_tx_date.
    POSITION_OF / AT_COMPANY edges. `MATCH (p:Person)-[:POSITION_OF]
    -(pos:Position)-[:AT_COMPANY]-(c:Company) RETURN
    pos.current_shares` returns the true total in one hop.
  - **`FilingYear`** (per (cik, year)) summarises filing activity
    with FILINGS_BY edge to Company.
  Chained aggregates demonstrate the compute pipeline composing —
  Stage A's `into` becomes Stage B's `from` automatically via the
  sub-node resolver.
- **Expression engine — null-propagating arithmetic & comparisons**.
  `null * 5`, `null + 3`, `null < x` all yield `null` (SQL
  semantics) instead of erroring. Real-world CSV data routinely
  has nulls (e.g. SEC insider grants with no `price_per_share`);
  the previous "error on null operand" behaviour forced
  `coalesce(x, 0)` wrapping on every arithmetic expression. Sum/
  avg already skip nulls, so propagation composes cleanly with
  aggregates.

### SEC EDGAR value-proposition upgrade (J0–J7)

The shipped SEC loader through 0.9.45 produced a Filing-index
graph but the detailed-payload extracts (Form 4, 13F, 8-K, SC
13D, DEF 14A, Exhibit 21) silently returned zero rows because
the per-filing fetcher loop was never wired. This release closes
that gap AND reshapes the schema so a SQL person looking at the
graph sees an immediate win — same-person multi-role queries,
typed insider edges, sector-cohort traversal via the new
SicCode node, fund-as-issuer bridge.

### Added (foundation — J0–J2)

- **Ticker support in `SEC.open(cik_list=...)`** (J0). Accepts
  string tickers (case-insensitive), int CIKs, or a mix:
  `cik_list=["AAPL", "BRK-B", 1318605]`. Resolves via the SEC
  `company_tickers.json` map (~1 MB, cached after first fetch).
- **Generic per-filing fetcher** (J1):
  `kglite_sec::fetch_filing_primary_doc` for 8-K / SC 13D / DEF
  14A primary docs; `kglite_sec::fetch_exhibit21_attachment` for
  10-K Exhibit 21 discovery via `index.json`. Exposed to Python
  as `_sec_internal.fetch_filing_batch` and
  `_sec_internal.fetch_exhibit21_batch`.
- **Wrapper batch dispatch** (J2):
  `_dispatch_per_filing_fetches` reads `processed/filing.csv`
  after extract_processed, groups by form type, calls the
  fetchers. `include_subsidiaries` and `include_8k_events` are
  no longer parsed-but-ignored — they gate the fetch.
- **Fetcher bug fixes** (J7-prep):
  - Form 4 was downloading XSL-rendered HTML instead of XML
    because `primaryDocument` points inside `xslF345X*/`; we now
    strip the directory and fetch the raw XML at the filing
    root. Without this, every Form 4 file errored at parse time.
  - 13F-HR `index.json` sometimes labels every document as
    `type: "text.gif"` (observed on Berkshire); the info-table
    discovery now falls back to "any non-primary_doc XML" when
    the type-label heuristic finds nothing.

### Changed (graph value — J3–J6)

These are breaking schema changes. Existing graphs cached under
`graph/{mode}/` won't load — call `SEC.open(...,
force_rebuild=True)` to re-derive.

- **Person unification — drops the `Director` node type** (J3).
  Form 4 reporters and DEF 14A directors now project onto a
  single `:Person` node with role edges back to Company. Exact
  token-sorted name match merges aligned references
  ("COOK TIMOTHY D" ↔ "Timothy D. Cook"); unmatched DEF 14A
  directors get a synthetic negative-i63 person_nid so the
  column stays Int64-typed (mixing string and int nids would
  downgrade the column to String and break FK lookups).
  `age` and `since_year` move from Director properties to
  edge properties on `SERVES_ON_BOARD` (per-filing facts, not
  per-Person facts).
- **Typed insider edges replace boolean-flag `HAS_INSIDER`** (J4):
  `Company -[:IS_DIRECTOR_OF]-> Person`,
  `Company -[:IS_OFFICER_OF]-> Person` (with `officer_title`),
  `Company -[:IS_BENEFICIAL_OWNER_OF]-> Person` (10%-owner +
  the rare `is_other` catch-all). Pattern matching becomes
  idiomatic instead of property-filtered, and the planner can
  use edge-type indexing.
- **Industry as a graph node** (J5):
  `processed/sic.csv` aggregates distinct
  (sic_code, sic_description) pairs. New `:SicCode` node + new
  `Company -[:IN_INDUSTRY]-> SicCode` fk edge. Sector cohort
  queries lose the `GROUP BY sic` ceremony.
- **Manager ↔ Company link** (J6): new
  `InstitutionalManager -[:IS_COMPANY]-> Company` fk edge.
  When a 13F filer's manager_cik matches a Company.cik
  (Berkshire, BlackRock, Vanguard), the same legal entity now
  materialises as one bridged node group across all three role
  views (issuer, 13F filer, board member).

### Schema migration

Existing graphs cached under
`<workdir>/graph/{mode}/` from 0.9.45 or earlier won't load
post-J3+J4. Re-derive:

```python
g = SEC.open(workdir, ..., force_rebuild=True)
```

`raw/` and `processed/` tiers are unaffected (well, processed/
regenerates from raw/ as usual); only the built graph file
needs rebuilding.

### Showcase notebook

[`examples/sec_to_claude_mcp.ipynb`](examples/sec_to_claude_mcp.ipynb)
demonstrates the schema with 7 queries that have no clean SQL
equivalent — multi-role insider unification, board interlocks,
sector-cohort sells, fund-as-issuer, 8-K → Form 4 proximity,
voting-power concentration, subsidiary depth. Ends by
registering the graph as a Claude Desktop MCP server (the
agent-first framing).

### Extraction performance + segfault fix

- **Parallel feature extraction** — each raw filing parses independently,
  so the form extractors now parse a chunk of files across all cores and
  emit rows single-threaded (lock-free CSV sinks). Combined with a unified
  ownership-XML pass — Form 3/4/5/144/D are walked and read once and
  dispatched by detected form type instead of five redundant re-parses —
  and larger (512 KB) CSV write buffers, SEC feature extraction is ~3.6×
  faster (14.8 s → 4.1 s on a 6,750-filing cache).
- **Fixed a `SEC.open()` segfault on macOS** — `kglite.datasets`
  submodules now import lazily (PEP 562), so `kglite.datasets.sec` no
  longer pulls `sodir`/`wikidata` → pandas → pyarrow into the process.
  Loading pyarrow after the kglite native extension triggered a
  dynamic-linker crash; keeping it off the SEC import path resolves it.
- **Identity pre-pass scoped to the corpus** — extraction without an
  explicit `cik_list` no longer scans the full ~900K-company
  `submissions.zip`. The CIKs to load are derived from the filings
  already on disk (`raw/filings/`, `raw/financials/`), so the identity
  pre-pass uses the direct-lookup fast path instead of an
  EDGAR-wide scan — ~20 s → tens of ms on a 100-filing corpus, where it
  had been 99.8% of total extraction wall time.

### SEC graph schema rebuilt for the info-row layout (F20)

The dataset blueprint (`kglite/datasets/sec/blueprint.json`) is rebuilt
from scratch against the info-row CSV layout the F-phase extractors now
emit. It is **fully node-centric** — every reported fact is a node that
carries its own data; edges are thin foreign-key connectors with no
properties:

- **Entity-hub nodes** — `Company`, `Person`, `Security`,
  `InstitutionalManager`, `SicCode`.
- **Fact nodes** — `Filing`, `InsiderTransaction`, `Holding`,
  `InstitutionalHolding`, `Role`, `CorporateEvent`, `MetricFact`,
  `Subsidiary`. Each row of an info-row CSV is one node; its attributes
  (role title, share counts, holding value, transaction price, …) live
  as node properties, not on edges.
- **Thin edges** — every connection is a foreign-key edge: a fact node
  links to its participant entities plus a `REPORTED_IN` edge to the
  `Filing` it came from, so provenance is always a traversal.
- **Compute layer** — a `Day`/`Month`/`Quarter` calendar with
  `FILED_ON`/`TRADED_ON`/`HELD_ON`/`OCCURRED_ON` links, `NEXT_FILING`
  and `NEXT_TX` temporal chains, and an `InsiderActivity` per-(person,
  company) rollup node.
- **Unified `insider_transaction.csv`** — the ownership extractor now
  emits one transaction table with a `direction` ("purchase"/"sale")
  column instead of separate `purchase.csv` + `sale.csv`, so an
  insider's whole trading history is one node type (`NEXT_TX` chains,
  net-position rollups).

Verified against a 100-filing corpus: a 31,615-node / 60,398-edge
graph across 17 node types and 19 edge types, no junction edges.

### DEF 14A compensation + governance (F8, F9)

- **`compensation.csv` + `Compensation` node** (F8) — the DEF 14A
  extractor now parses the proxy statement's Summary Compensation
  Table (Item 402): one node per named-executive-officer / fiscal
  year, with salary / bonus / stock + option awards / non-equity
  incentive / pension change / other / total, edged to `Person`,
  `Company`, and the `Filing` it came from.
- **`proposal.csv` / `ceo_pay_ratio.csv` / `audit_fees.csv` + their
  nodes** (F9) — the DEF 14A pass also extracts ballot proposals
  (number, description, board recommendation, company vs shareholder),
  the Item 402(u) CEO pay-ratio disclosure, and the Item 9(e)
  independent-auditor fee table. Heuristic scans — they drop rather
  than guess: pay-ratio values in the 1900-2100 range (mis-read
  dates) and sub-$50k "audit fees" (footnote noise) are rejected.
- **person id scheme unified** — Form 3/4/5/144 person ids are now
  `cik-{N}` (non-numeric), so they no longer collide-type with the
  name-keyed proxy/10-K person ids and break FK edges into `Person`.
- **ownership-table parser hardened** — footnote sentences, table
  captions and city/state address lines no longer leak through as
  beneficial-holder rows.

### Related-party transactions (F12)

- **`related_party_transaction.csv` + `RelatedPartyTransaction`
  node** — a new parser reads the related-party section: 10-K Item 13,
  but since most 10-Ks delegate that item to the proxy, it also
  locates a DEF 14A's "Related Person Transactions" heading (where the
  detail actually lives). Conservative — only sentences carrying an
  explicit dollar amount become rows, with the amount, a year, and a
  relationship hint (director / officer / family member / affiliate).
- **shared `parsers/html_text.rs`** — the `extract_item_text`
  Item-section scanner is lifted out of the SC 13D parser into a
  shared module so the 10-K / DEF 14A section extractors reuse it.

### 8-K officer changes (F13)

- **`officer_change.csv` + `OfficerChange` node** — 8-Ks carrying an
  Item 5.02 are scanned for officer / director changes. The change
  detail is frequently cross-referenced into Item 8.01, so the whole
  (short) 8-K is scanned: each "Mr./Ms./Mrs./Dr." person mention
  yields a change typed from the surrounding verb (resignation /
  retirement / appointment / election / departure) with a title and
  effective date. The lowest-precision extractor in the set — a
  person without a recoverable name or a change verb is skipped.

### 8-K earnings releases (F14)

- **`earnings_release.csv` + `EarningsRelease` node** — a new parser
  reads an earnings press release (8-K Item 2.02 body or its
  Exhibit 99 attachment) and pulls the headline figures: revenue,
  net income, and basic / diluted per-share earnings (parenthesised
  losses read negative, million/billion multipliers honoured). It
  self-gates on the earnings vocabulary — any non-earnings document
  yields nothing. Wired into `forms/eightk.rs`, scanning both 8-K
  covers and `ex-99` attachments; the blueprint gains an
  `EarningsRelease` node edged to `Company` and `Filing`. Verified by
  parser fixtures — the benchmark corpus carries no Exhibit 99
  attachments, so end-to-end coverage awaits an exhibit fetch.

### S-1 / 424B securities offerings (F15)

- **`offering.csv` / `selling_stockholder.csv` / `underwriter.csv` /
  `use_of_proceeds.csv` + their nodes** — a new parser reads a
  registration statement (S-1) or prospectus (424B): the offering
  summary (type, shares, price, gross/net proceeds), the
  selling-stockholder table (per-seller share breakdown), the
  underwriting syndicate, and the use-of-proceeds narrative. The
  previously stubbed `forms::s1` and `forms::prospectus` extractors
  share one walk/parse/emit routine. Verified by parser fixtures —
  the benchmark corpus carries no S-1 / 424B documents, so the four
  nodes load cleanly with zero rows pending a registration-statement
  fetch.

### SC 13D/G amendment + group refinements (F18)

- **Amendment detection** — the SC 13D/G parser reads the cover
  page's "(Amendment No. N)" marker, so `activist_filing.is_amendment`
  is now set from the filing itself instead of hardcoded `0`.
- **`holder_group.csv` + `HolderGroup` node** — when one SC 13D/G
  carries multiple reporting persons they are a § 13(d) group;
  `schedule13` now links each joint filer to the first.
- **`ActivistFiling` node** — `activist_filing.csv` (long the
  SC 13D/G output) finally enters the blueprint, edged to `Company`
  and `Filing`. Verified by parser fixtures — the benchmark corpus
  carries no SC 13D/G documents, so both nodes load with zero rows
  pending a Schedule 13 fetch.

### Deferred-placeholder sinks documented (F19)

- The eight CSV sinks with no extractor yet — `auditor`,
  `auditor_change` (8-K Item 4.01), `restatement` (8-K Item 4.02),
  `ma_event` (8-K Item 2.01), `vote_result` (8-K Item 5.07),
  `pay_vs_performance` (DEF 14A Item 402(v)), `fund_vote` (Form N-PX)
  and `merger` (Form S-4) — now each carry a `PLACEHOLDER (deferred)`
  doc comment naming the form/item that will populate them. The
  headers were already written; this closes the F-phase program by
  making every still-empty sink an intentional, documented
  placeholder rather than an unexplained gap.

### `SEC.open` — lean-core default fetch scope

- `form_types` now scopes the **per-filing fetch**, not just the
  extract step. Previously a sliced call (`form_types=["4"]`) still
  downloaded every form bucket — 13F info tables, DEF 14A proxies, 8-K
  cover pages, Exhibit 21 attachments, XBRL company-facts — and only
  filtered them out at extract time.
- When `form_types` is unset, the per-filing fetch defaults to a lean
  core set — insider ownership (Forms 3/4/5) + 8-K cover pages. The
  heavy payloads are now opt-in: name the form in `form_types`
  (`["13F-HR"]`, `["DEF 14A"]`, `["SC 13D"]`, `["144"]`, `["10-K"]`
  for Exhibit 21), or set the matching `include_*` flag.
- **Default change:** `include_subsidiaries` and `include_xbrl_metrics`
  now default to `False` (were `True`) — Exhibit 21 and XBRL
  company-facts are the most expensive fetches (2 requests per 10-K;
  5-50 MB JSON per company) and are opt-in under the lean default.
  `include_8k_events` stays `True` — 8-K is part of the lean core.

### SEC loader — `SEC.fetch` shortcut, `companies` argument, progress bars

- New `SEC.fetch(path, forms, companies, *, years=2, user_agent=...)`
  — an ergonomic shortcut that fetches *and* builds a graph for a
  focused slice: name a form, a company, and a span (`SEC.fetch(path,
  "13F-HR", "TSLA", years=2, user_agent=UA)`). `forms`/`companies`
  accept a single value or a list; `years` drives both the filing
  index and the per-filing payload depth. A `force_rebuild` flag
  rebuilds when re-running with a changed scope (the graph cache is
  keyed by workdir, not by scope). `SEC.open` remains the
  full-control entry point.
- `SEC.open`'s `cik_list` parameter is renamed to `companies` — it has
  always accepted int CIKs, string tickers, and mixed lists, and the
  new name reflects that. No back-compat alias; update call sites
  (`SEC.open(..., companies=[...])`).
- New `progress` parameter on `SEC.open` — a callable receiving
  structured progress events from the per-filing fetch. The
  rate-limited download of Form 4 / 13F / 8-K / DEF 14A / Exhibit 21
  documents now drives a tqdm progress bar (one per fetch phase),
  auto-enabled on `verbose` runs when `tqdm` is installed and falling
  back to the previous `[SEC]` prints otherwise. `tqdm` stays an
  optional dependency. Ctrl+C during a fetch now aborts cleanly.

### SEC loader — dead FSNDS bulk-feed fetch removed

- `SEC.open` no longer downloads the legacy FSNDS (Financial Statement
  and Notes Data Set) quarterly bulk ZIPs. F17 replaced that feed with
  per-company XBRL company-facts JSON, and nothing read the FSNDS
  `num.tsv` files anymore — the fetch was pure dead weight. XBRL
  financial metrics are unaffected; they still come from the
  company-facts fetch gated by `include_xbrl_metrics`.

### Sodir loader ported to Rust — `pandas` dropped

- The Sodir FactMaps dataset loader is now a pure-Rust crate
  (`kglite-sodir`) behind a thin Python wrapper, matching the SEC
  loader's architecture: ArcGIS REST fetch + GeoJSON→CSV, the
  two-tier cooldown index, the FK preprocessing, and the blueprint
  deep-merge all moved out of Python.
- **`pandas` is no longer a dependency.** It was used only by the old
  Sodir Python modules; with those gone, `pandas` (and transitively
  `numpy`/`pyarrow`) is removed from `pyproject.toml`. `kglite.datasets.
  sodir.open()` / `fetch_all()` keep their signatures.
- The Wikidata dump loader's download orchestration also moved to a
  pure-Rust crate (`kglite-wikidata`) — the resumable download and the
  staleness/cooldown cache no longer shell out to a `curl` subprocess.
  `kglite.datasets.wikidata.open()` keeps its signature; the N-Triples
  graph build is unchanged.

## [0.9.45] — `save_graph` mode-aware dispatch in `kglite-mcp-server`

Correctness fix for an MCP-server-only regression latent since
0.9.20: the `save_graph` tool errored on in-memory `.kgl` graphs
because the Rust crate's `run_save` only handled the disk
branch. Ships with the dispatch extracted into
`kglite::api::save_graph` (single source of truth shared with
the Python wrapper) and the CI gap that hid the regression
closed.

### Fixed

- **`kglite-mcp-server`'s `save_graph` tool errored on in-memory
  `.kgl` graphs** with `"save_disk requires disk mode"`. The Rust
  crate's `run_save` at `crates/kglite-mcp-server/src/tools.rs:599`
  called `dir.save_disk(path)` unconditionally; `save_disk` is the
  disk-mode-only path. The Python `KnowledgeGraph.save()` at
  `src/graph/pyapi/kg_core.rs:505` has always dispatched correctly
  via `is_disk()`, but the Rust crate never got the equivalent.
  Latent since the 0.9.20 architecture change (May 11) — undetected
  because `tests/test_mcp_server_smoke.py` is
  `pytest.mark.skipif(not BINARY.exists())` and CI didn't build
  the binary.

  Fix: new `kglite::api::save_graph(graph, path)` in
  `src/graph/io/file.rs` performs the same dispatch as the Python
  wrapper (disk → `save_disk`; in-memory → `prepare_save` →
  `enable_columnar` → `write_graph_v3`). The MCP crate now calls
  it, removing the duplicated dispatch surface.

### Added

- **CI builds the `kglite-mcp-server` binary** before the pytest
  step, so `tests/test_mcp_server_smoke.py` runs in CI instead of
  silently skipping. This is what would have caught the
  `save_graph` regression at 0.9.20.
- **Disk-mode `save_graph` round-trip test**
  (`test_c8b_save_graph_persists_disk_mode` in
  `tests/test_mcp_server_python_entry.py`) locks in the
  disk-branch of the dispatch — the complement to the existing
  in-memory `test_c8`.
- **`kglite::api::save_graph`** and `kglite::api::save_inmemory`
  (via `kglite::graph::io::file`) for non-pyo3 Rust consumers.

### Changed

- `tests/test_mcp_server_smoke.py` opts into `save_graph` via the
  manifest (`builtins.save_graph: true`) — catching up with the
  opt-in design `ff5cc91` introduced for the canonical
  `tests/test_mcp_server_python_entry.py` fixtures.

## [0.9.44] — Streaming node + FK-edge loaders (F1–F4)

Completes the streaming-CSV refactor started in 0.9.43. The
junction-edge loader was the warm-up (E1–E4); 0.9.44 brings the
same per-chunk dispatch to node specs and their FK edges, so
peak RAM during `from_blueprint` is now bounded by chunk size
rather than total CSV size for the dominant SEC + Wikidata
shapes.

### Added

- **Streaming node-loader for simple specs** in
  `src/graph/blueprint/build.rs` (F1). Specs that are CSV-backed
  and *not* manual / timeseries / spatial flow through a
  per-chunk `read_csv_chunks → typed_dataframe → add_nodes` loop.
  `add_nodes` is upsert-by-id so successive chunks accumulate
  cleanly into the same node type. Buffered path still owns
  timeseries / spatial / manual specs (they need random access
  to the full row set for grouping, in-place geometry conversion,
  or FK-target discovery).

- **Auto-pk threading across chunks** (F2). `pk: "auto"` specs
  stream via a per-spec `u64` counter that advances by each
  chunk's post-filter row count. Synthesised ids remain dense
  `1..=N` matching the buffered path's behaviour. Sub-nodes with
  `pk:"auto"` + parent FK (the dominant SEC + Sodir shape) now
  stream end-to-end on both the node and FK sides (F3).

- **FK-edge streaming for streaming-eligible specs** (F3). FK
  edges from streamed parents emit one `connect()` call per
  (chunk, declared edge) pair, built on the same
  `build_fk_columns` + `build_edge_df` + `connect()` primitives
  as the buffered path. The cache pre-parse step
  (`parse_in_parallel`) skips streamed specs — their CSVs are
  read on demand by the streaming loaders.

- **Adaptive streaming via a file-size gate** (F4). Per-spec
  CSV size is checked at build start: files at or above
  `KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB` (default 100 MB)
  flow through the streaming path; smaller files stay on the
  buffered path. The default threshold keeps Sodir / SEC-1yr
  blueprints on the fast path (zero regression vs 0.9.43) while
  triggering streaming for the SEC full-universe / Wikidata-scale
  cases where the RAM bound matters.

### Tuning knobs

- `KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB` — file-size gate
  (default 100 MB). Set to `0` to force streaming on all
  eligible specs; set higher to keep more on the buffered path.
- `KGLITE_BLUEPRINT_NODE_CHUNK_SIZE` — rows per chunk for node
  + FK streaming (default 250K). ~110 MB peak per chunk at
  typical row widths; reduce for RAM-tight hosts.
- `KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE` — junction-edge
  streaming chunk size (default 100K, unchanged from 0.9.43).

### Performance

Synthetic 500K-row Employee + 1000-row Company + WORKS_AT FK
(13 MB employees.csv), 5 cold rounds, `min` (per CLAUDE.md
perf protocol):

| Path | Time | vs 0.9.43 |
|---|---:|---:|
| 0.9.43 buffered (baseline) | 0.373 s | — |
| 0.9.44 default (file < 100 MB → buffered) | 0.370 s | -0.8% |
| 0.9.44 forced streaming (threshold=0) | 0.431 s | +15.5% |

Synthetic 5M-row Employee + 5K-row Company + WORKS_AT FK
(145 MB employees.csv), 3 cold rounds, `min`:

| Path | Time |
|---|---:|
| 0.9.44 default (file ≥ 100 MB → stream) | 9.91 s |
| 0.9.44 forced buffered (threshold=999) | 7.10 s |

The streaming path carries ~40% wall-time overhead on
in-RAM-fits-anyway sizes — the cost of per-chunk dispatch and
loss of off-thread parallel prep. This is the explicit tradeoff
the size gate manages: streaming earns its keep when buffering
would push the process toward OOM (multi-GB CSVs).

### Notes for v0.9.45+

- **Streaming for timeseries / spatial specs**. Both currently
  require multi-pass access (timeseries: grouping by pk; spatial:
  in-place geometry conversion). A two-pass streaming design is
  feasible but more invasive — deferred until a real Wikidata-scale
  timeseries graph asks for it.
- **Per-chunk type-inference stability**. `build_edge_df` infers
  FK column types per-chunk; chunks with all-int rows + one chunk
  with a string row would disagree. Real-world FK columns are
  consistently typed; if this surfaces in production, move FK
  types to explicit blueprint declarations.

## [0.9.43] — Streaming CSV for junction-edge loader (E1–E4)

### Added

- **Streaming junction-edge loader** in `src/graph/blueprint/build.rs`.
  Previously every junction CSV was eagerly parsed into a `CsvCache`
  before `load_junction_edges` could process it. For multi-million-
  row junction tables (e.g. SEC HOLDS at full-universe scale, ~30M
  rows) that peaked RAM at 5–10 GB during the prep phase.

  The new path streams each junction CSV in chunks of 100K rows
  (configurable via `KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE`),
  building a per-chunk `DataFrame` + dispatching to `connect()`
  before the chunk is dropped. Peak RAM during junction loading is
  now bounded at ~20 MB per chunk regardless of total file size.

  Trade-off: junction loading is now sequential per-spec instead of
  parallel across specs. Negligible on small graphs (Sodir); the
  win is large graphs.

- **`read_csv_chunks(path, chunk_size)`** in
  `src/graph/blueprint/csv_loader.rs`: streaming chunked CSV reader
  that yields `RawCsv` chunks of configurable size. Foundation for
  future node-loader streaming (deferred to a later phase since
  node CSVs interact with multi-pass operations like dedup_by_pk
  and timeseries grouping that need more careful migration).

- **`CsvStream`** in `src/graph/blueprint/csv_stream.rs`: low-level
  per-row streaming iterator. Currently used internally as a design
  reference; can be picked up by future consumers that need per-row
  dispatch (e.g. one-shot mutations against a graph).

### Notes for v0.9.44+

- **Node-loader streaming** (deferred E2 work) — `prep_node_spec`
  still uses the buffered `CsvCache` path. For node CSVs that grow
  past a few million rows (full-universe SEC `MetricFact` would be
  ~50M), that's the next memory hotspot. The migration needs to
  thread an auto-pk counter across chunks and handle timeseries /
  dedup as multi-pass operations.
- **FK-edge streaming** — `prep_fk_edges` re-uses the node spec's
  CSV from cache, so streaming there only helps when paired with
  node-loader streaming. Bundle with the v0.9.44 node migration.

### Build/test status

- 19/19 `tests/test_blueprint.py` parity tests green.
- All SEC smoke + use-case-v2 tests green.
- 5 new chunk-reader unit tests, 6 new CsvStream unit tests.
- `make lint` green per phase commit.

### Documentation

- **README — SEC EDGAR loader promoted to a top-of-page use case.**
  Three placements: a new 🏦 callout right after the codebase →
  Claude callout, a 🏦 bullet at the top of the Use cases list,
  and an SEC EDGAR entry as the first item under Bundled datasets.
  New "Why Cypher?" section between Use cases and How it compares —
  one concrete example (insider sells at a specific CIK) plus a
  hint at how the same pattern shape composes into harder questions
  (swap `:HAS_INSIDER` → `:HOLDS`, add `:SERVES_ON_BOARD`).
- Comparison table updated to list SEC EDGAR alongside Wikidata and
  Sodir under "Bundled public datasets".

## [0.9.42] — SEC EDGAR loader deepening (D1–D10)

### Added

- **D10 — Use-case tests v2** —
  `kglite/datasets/sec/tests/test_usecases_v2.py` runs 10 SQL-style
  queries (UC11–UC20) against a fully-deepened synth graph,
  exercising Subsidiary / MetricFact / Event / Stake / Director on
  top of the v1 Company/Filing/Person/Transaction/Holds. Records
  min/avg query timing and prints a summary table.
- **D9 — DEF 14A board parser** — new `parsers/def14a.rs` (11 unit
  tests) extracts directors via heuristic HTML scanning of
  "DIRECTORS AND EXECUTIVE OFFICERS" sections. Requires age or
  "since YYYY" marker to register a name. Expected 50–70% accuracy.
  `extract_directors` walks raw/filings/ for def14a/proxy filenames.
  Blueprint adds Director + SERVES_ON_BOARD edge to Company.
- **D8 — SC 13D activist-stake parser** — new `parsers/sc13d.rs` (8
  unit tests) extracts Item 4 purpose text + Item 5 percent owned
  from 13D HTML via Item-anchor scanning + percent regex.
  `extract_13d_stakes` emits Stake nodes linked to Filing.
- **D7 — Storage mode auto-escalation** —
  `_predict_graph_size_gb` + `_pick_storage_mode` together pick
  memory / mapped / disk based on years × detailed × CIK-fraction
  × per-deepening cost. SEC.open() default `mode=None` is now auto.
- **D6 — Form 4 + 13F batch fetchers** — `fetch_form4_batch` and
  `fetch_13f_batch` pyo3 functions take a list of (cik, accession,
  ...) and process the whole batch with ONE shared SecClient so the
  10 req/s governor token bucket applies across the entire batch.
- **D5 — 13F info-table fetcher** — `fetch_13f_info_table` hits the
  filing's index.json, discovers the info-table XML filename
  (type='INFORMATION TABLE'), and downloads it into raw/filings/.
- **D4 — 8-K Item codes** — `extract_8k_events` walks raw/filings/
  HTM for `Item N.NN` patterns via the existing parsers::eightk
  parser. Blueprint adds `Event` sub-node + `OF_FILING` fk_edge.
  `include_8k_events` flag in wrapper.
- **D3 — FSNDS XBRL** — `fetch_fsnds_quarterly` downloads quarterly
  ZIPs and extracts NUM.tsv (bulk path, no rate limit).
  `extract_xbrl_metrics` filters via the existing
  DEFAULT_TAG_WHITELIST and emits `processed/metric_fact.csv`.
  Blueprint adds `MetricFact` + `REPORTED_IN_FILING` fk_edge.
  CIK is reached via Filing -> FILED_BY -> Company traversal.
- **D2 — Exhibit 21 subsidiaries deepening** —
  `extract_subsidiaries(workdir, slice, force)` walks
  `raw/filings/{cik}/{accession}/*ex21*.htm` (and `exhibit21`,
  `ex-21` variants), parses via the existing `parsers::exhibit21`,
  and emits `processed/subsidiary.csv` with composite
  `subsidiary_nid = "{parent_cik}_{name_normalized}"` for dedup
  across years. Blueprint adds `Subsidiary` node + `OF_COMPANY`
  fk_edge. Python wrapper gains `include_subsidiaries` flag.
  User-story test: M&A analyst stages Exhibit 21 HTML → 3
  subsidiaries land + Apple → OF_COMPANY → Subsidiary edges work.
- **D1 — Slice grammar wired end-to-end** —
  `kglite_sec::SliceSpec { cik_list, form_types, year_range }` is now
  applied uniformly across `extract_companies_and_filings`,
  `extract_insider_transactions`, and `extract_holdings`. The
  `SEC.open()` Python wrapper exposes `cik_list`, `form_types`, and
  `year_range` kwargs that turn a 5-hour full-universe build into a
  ~5-minute S&P-500-scoped build. User-story test in `test_smoke.py`
  validates that `cik_list=[789019]` produces a Microsoft-only graph
  regardless of submissions.zip size.
- **`kglite.datasets.sec.SEC.open(path, *, years, detailed, mode,
  user_agent, ...)`** — first end-to-end SEC EDGAR loader (phase 3 of
  the planned loader work). Builds a knowledge graph with Company +
  Filing nodes connected by `FILED_BY` edges from a three-tier
  workdir cache (`raw/`, `processed/`, `graph/{mode}/`). Modes
  `memory` and `mapped` work; `disk` lands in a later phase.
  Coexisting per-mode subdirs mean opening with one mode never
  touches another's graph. Default behaviour reuses a cached graph
  on reopen without rebuilding.
- **`crates/kglite-sec/` extended** — `SecClient` (10 req/s token
  bucket, mandatory User-Agent, retry-with-backoff), `fetch.rs`
  orchestrator for quarterly master.idx + bulk submissions.zip +
  company_tickers.json, `parsers::submissions` streaming parser for
  the bulk submissions ZIP, `extract.rs` orchestrator that emits
  `processed/company.csv` + `processed/filing.csv` with dedup across
  sources.
- **PyO3 wrappers in `src/sec.rs`** — exposes the Rust loader as the
  `kglite._sec_internal` submodule. Single-threaded tokio runtime per
  call; Python callers see plain blocking functions.
- **Phase 9 — live SEC integration test** —
  `kglite/datasets/sec/tests/test_integration_live.py` does an
  end-to-end build against live SEC (env-gated via
  `KGLITE_SEC_INTEGRATION=1`). Builds a 1-year graph in ~4.5s on a
  dev box: ~388K Filing nodes from live master.idx fetches, with
  the top-5 form types matching real 2024 SEC volumes (Form 4 80K,
  424B2 46K, 8-K 26K, 13F-HR 18K, NPORT-P 17K). Validates the cached
  graph reopen path too. Fixed an accession-number extraction bug
  uncovered by the test — real SEC master.idx file paths end in
  `.txt` not `-index.htm`; the parser now accepts either.
- **Phase 8 — disk mode + docs** — `SEC.open(mode="disk")` now works
  via `from_blueprint(storage="disk", path=graph/disk/)`. Disk graphs
  are loaded on subsequent opens via the cache reuse path. Adds
  `docs/guides/sec.md` covering the workdir layout, schema, storage
  modes, sizing, and caveats (CIK-as-int, CUSIP edge cases, per-filing
  fetch rate limits).
- **Phase 7 — per-filing detail parsers** — `parsers/eightk.rs`
  extracts standardized Item codes from 8-K cover pages (1.01 = entry
  into material agreement, 5.02 = officer departure, etc.) via
  regex-light scanning of stripped HTML. `parsers/exhibit21.rs`
  extracts subsidiary lists from 10-K Exhibit 21 documents using a
  permissive line-by-line heuristic (Exhibit 21 has no SEC-mandated
  schema). Both parser-only; schema wiring deferred to Phase 8.
- **Phase 6 — FSNDS XBRL parser** — `parsers/fsnds.rs` streaming
  reader for the quarterly Financial Statement and Notes Data Set
  `num.tsv` (tab-separated XBRL numeric facts). Whitelist-based
  filtering with a `DEFAULT_TAG_WHITELIST` covering 20 high-value
  us-gaap tags (Revenues, NetIncomeLoss, Assets, etc.). Schema
  wiring deferred to Phase 8 polish so this phase is parser-only.
- **Phase 5 — 13F institutional holdings** — `parsers/f13f.rs`
  streaming XML parser for Form 13F-HR information tables;
  `extract_holdings` orchestrator walks `raw/filings/{cik}/{accession}/*.xml`
  and emits `processed/{institutional_manager,security,holds}.csv`.
  Schema gains `InstitutionalManager` + `Security` node types and the
  `HOLDS` junction edge with shares / value / voting authority
  properties. PyO3 surface gains `extract_holdings_py`.
- **Phase 4 — Form 4 insider transactions** — `parsers/form4.rs`
  streaming XML parser for Form 4 / 4/A (XSD schemaVersion X0508);
  `extract_insider_transactions` walks `raw/filings/{cik}/{accession}/*.xml`
  and emits `processed/{person,transaction,has_insider}.csv`. Schema
  extended with `Person` node + `Transaction` sub-node + `HAS_INSIDER`
  junction edge (Company → Person, with director/officer/10%-owner
  flags) + `OF_PERSON` / `INVOLVES_ISSUER` / `REPORTED_IN_FILING`
  fk_edges. `fetch_form4_filing` per-accession fetcher for the
  rate-limited Form 4 ingest path. PyO3 surface gains `extract_insider`.

### Changed

- **README: new top-level `Serve it to an agent` section** between
  Quick Start and Bundled datasets. Three subsections covering the
  progressive disclosure of MCP capability: one-command (`kglite-mcp-server --graph X.kgl`),
  YAML manifest customisation (`source_root`, `extensions.embedder`,
  inline Cypher tools — with a worked example), and bundled skills
  (`.skills/*.md` files that teach agents how to use the tools, with
  `applies_when:` predicates so only relevant methodology activates).
- **README intro now opens with a concrete "first graph in seconds"
  hook** — `pip install kglite` + `kglite.code_tree.build(".")`. Sells
  the embedded, zero-setup pitch before the reader sees any prose
  about MCP servers or validators.
- **README use-case fixes**: "Your pandas DataFrames" widened to
  *"Your structured data"* (covers SQL / CSV / Parquet / REST → graph);
  Wikidata claim corrected (the headline is *operate/query a
  billion-edge graph on a 16 GB laptop*, not "loads in 7 minutes" —
  build is ~90 min, reload is <10 s); RAG bullet rewritten with a
  concrete legal-corpus example (laws + court decisions + citations,
  semantic-similar cases → walk to related precedents) instead of a
  generic "document corpus."

- **README refocused around the agent-first pitch.** Tagline now reads
  *"Knowledge graph for Python, built for LLM agents."* The four use
  cases (codebase / DataFrames / public datasets / RAG corpus) are
  reframed as a single `Use cases` section (five bullets covering
  domain knowledge for agents, business data, public datasets, RAG,
  and codebase analysis) instead of competing pitches scattered
  across `Why KGLite?` + `Use Cases` + `Key Features`. Notebook callout promoted to a banner-style H3 and
  also linked inline under the codebase use case (visible twice in the
  first 30 lines). `Use Cases` consolidated into a tighter `Recipes`
  section (MCP serve, hybrid retrieval, structural validators, graph
  algorithms). Stale 0.9.18 → 0.9.20 migration block removed. Broken
  `#public-datasets` anchor fixed. 15 inline links to docs guides
  (`code-tree`, `data-loading`, `semantic-search`, `mcp-servers`,
  `graph-algorithms`, `traversal-hierarchy`, `recipes`, `datasets`,
  `blueprints`, `spatial`, `timeseries`, `import-export`, `ai-agents`,
  `cypher`, `querying`) sprinkled next to the content they describe
  rather than dumped only at the bottom. Documentation section
  reorganised into five themed buckets. Net: 408 → 358 lines.

## [0.9.41] — 2026-05-18

### Changed

- **`[mcp]` extras removed. MCP server runtime is now a default dep.**
  `pip install kglite` ships everything needed to run `kglite-mcp-server`
  out of the box: `mcp`, `pyyaml`, `aiohttp`, `watchdog` (~6 MB
  combined). No more extras-dance for Claude Desktop / Cursor / any
  MCP client. The old `[mcp]` name was confusing (it bundled the MCP
  runtime with the embedder), and after Phase 2's `mcp-methods` wheel
  drop the remaining runtime footprint was small enough to default-
  ship. **Breaking**: `pip install 'kglite[mcp]'` no longer resolves;
  use `pip install kglite`. People who used `[mcp]` for semantic
  search should now use `[embed]` (see below).
- **New `[embed]` extra for semantic search.** `pip install 'kglite[embed]'`
  pulls `fastembed>=0.4` (and ~97 MB of transitives: onnxruntime,
  tokenizers, pillow, huggingface-hub). Required for `text_score()`
  semantic Cypher and the `extensions.embedder` manifest extension.
  Niche use case, hence opt-in. Same ONNX backend as before, same
  `~/.cache/fastembed/` model cache.
- Notebook + README install instructions updated to the new flat
  `pip install kglite` shape. Old `[mcp]` references in
  `dev-documentation/mediumpost.md` swapped too.

- **`mcp-methods` PyPI wheel no longer a runtime dependency.** Skill
  loading routes through new `kglite._mcp_internal.SkillRegistry` /
  `kglite._mcp_internal.Skill` pyo3 wrappers (in `src/mcp_tools.rs`),
  which delegate to `mcp_methods::server::SkillRegistry::from_manifest`
  added in the upstream Rust crate at 0.3.38. Drops ~16 MB of upstream
  wheel + bundled binary from `[mcp]` extras (~93 MB → ~77 MB). No
  orchestration logic on kglite's side — upstream stays canonical.
  Behaviour is byte-identical against
  `tests/test_mcp_server_python_entry.py -k skill` (5 passing) and
  against `examples/open_source_workspace_mcp.yaml` end-to-end (5
  framework skills load, `provenance` strings format exactly as the
  prior pyo3 wheel did: `"project"` / `"bundled"` /
  `"domain_pack:<path>"`). `kglite/mcp_server/skills_loader.py`
  swapped the `import mcp_methods` to `from kglite import _mcp_internal`;
  pyproject.toml dropped `mcp-methods>=0.3.36` from `[mcp]` extras;
  Cargo.toml pinned `mcp-methods` floor to `0.3.38` for the new
  `Registry::from_manifest` helper.

- **`examples/codebase_to_claude_mcp.ipynb` polish (no API change):**
  - Drop the gratuitous `str(ws.root)` — `code_tree.build()` already
    accepts `os.PathLike`. The notebook now reads `build(ws.root)`.
  - The `REPO =` comment clarifies storage modes: in-memory (default)
    handles repos up to millions of LoC; Wikidata-scale graphs need
    `kglite.KnowledgeGraph(storage="disk", path=…)`.
  - Requirements line corrected to `pip install kglite` — the notebook
    itself doesn't pull any `[mcp]`-only deps (verified by import).
    `pip install 'kglite[mcp]'` is now framed as the env Claude Desktop
    uses to spawn `kglite-mcp-server`, not a notebook requirement.

## [0.9.40] — 2026-05-18

### Added

- **`KnowledgeGraph.shape` property and human-readable `__repr__`.**
  `g.shape` returns `(node_count, edge_count)` pandas-style — O(1) via
  the storage backend, no per-type breakdown computed. `repr(g)` and
  `print(g)` now produce `KnowledgeGraph(1,245 nodes, 2,996 edges)`
  instead of the default `<builtins.KnowledgeGraph object at 0x…>`.
  Use `schema()` / `describe()` for full per-type structure when needed.

### Changed

- **`examples/codebase_to_claude_mcp.ipynb` cell 1** now uses
  `kglite.mcp_server.workspace.Workspace` (the built-in clone +
  auto-prune system with `stale_after_days`) instead of `subprocess.run`
  for git clone, and uses `print(graph)` (the new `__repr__`) instead
  of manual `schema()` lookup. Same workspace dir is reachable from the
  Claude Desktop MCP server registered in cell 4, so the demo state
  is continuous.
- **README:** notebook callout promoted to immediately after the lead
  paragraph (was buried in the Examples section near the bottom). Same
  link still exists in Examples for completeness.

## [0.9.39] — 2026-05-18

### Fixed

- **`code_tree` route detector: tuple-form `methods=(...)` no longer
  leaks parens into `Route.method` and `Route.id`.** Flask's own
  tutorial uses `@app.route('/x', methods=('GET', 'POST'))` (tuple),
  not the list form. Before the fix, `parse_methods_list` only stripped
  `[`/`]` brackets, so methods came out as `("GET` / `POST")` and ids
  as `flask::("GET::/register`. Now accepts list `[...]`, tuple `(...)`,
  and bare-string forms. Regression covered by
  `tests/test_code_tree_routes.py::test_flask_route_methods_tuple_form`
  and Rust unit tests in `src/code_tree/builder/routes/mod.rs`.

### Added

- **`kglite.mcp_server.claude_config`** — standalone Python helpers for
  managing MCP server entries across Claude clients:
  `list_mcps` / `get_mcp` / `add_mcp` / `edit_mcp` / `delete_mcp`, plus
  `default_path(client)`. Supports `client="claude_desktop"` (platform-
  aware path to `claude_desktop_config.json`), `client="claude_code"`
  (`~/.claude.json`), `client="vscode"` (`./.vscode/mcp.json` — writes
  the `servers` key with `type: stdio` instead of `mcpServers`), and
  arbitrary `path="/custom/config.json"`. Mutations are atomic
  (write-tmp + `os.replace`) and preserve every other top-level key
  in the config — important because `claude_desktop_config.json` also
  stores `preferences`, scheduled-task flags, etc. that must not be
  clobbered. `dry_run=True` returns the would-be entry without
  touching disk. `add_mcp` / `edit_mcp` default
  `resolve_command=True`: bare binary names are resolved to absolute
  paths via `shutil.which` at write time, so the entry survives Claude
  Desktop's minimal-PATH subprocess environment (avoids the silent
  "server doesn't start" mode where the bare name isn't on Claude's
  launch PATH). Pass `resolve_command=False` for Docker shims or
  wrapper scripts.
- **`examples/codebase_to_claude_mcp.ipynb`** — end-to-end notebook:
  clone a famous open-source repo, parse it into a code knowledge
  graph, run a few Cypher queries, then register a workspace MCP
  server in Claude Desktop (via the new `claude_config` helpers) so
  the agent can `repo_management('org/repo')` any GitHub repo on
  demand.

### Changed

- **README: new "How it compares" section** positioning KGLite vs Kuzu,
  NetworkX, rustworkx, and Neo4j Embedded — install, query language,
  storage, pandas bulk-load, MCP server, `describe()`, code_tree,
  bundled datasets, and license.
- **LICENSE: normalised to the canonical MIT template** (3-paragraph
  grant, straight ASCII quotes). No legal change; the previous text
  was condensed enough that GitHub's licensee auto-detector misread it
  as MIT-0. Detection should flip back to `mit` on the next push.

## [0.9.38] — 2026-05-17

### Added

- **Mode banner in the MCP server's `instructions` block and
  `graph_overview()` preamble.** Operators running several MCP servers
  in parallel previously had no clean way to tell which conditional
  tools (`repo_management`, `set_root_dir`, `save_graph`) were
  registered in the current mode — agents were fingerprinting mode by
  trial calls, burning context and turns. The Python entry
  (`kglite.mcp_server.server`) now prepends a per-mode banner to both:

  - the `instructions` block returned during MCP `initialize` (read
    once at handshake), and
  - the bare `graph_overview()` response preamble (re-read on each
    call, survives context aging).

  Banner names every conditional tool — both the registered ones AND
  the unregistered ones — across all six modes (`graph`, `workspace`,
  `local_workspace`, `source_root`, `watch`, `bare`), and flips the
  `save_graph` line based on `builtins.save_graph`. Marker
  `[kglite-mode]` identifies the segment for downstream tooling.

## [0.9.37] — 2026-05-17

Post-0.9.36 operator-feedback batch. Four independent fixes from the
sodir-prospect / legal / open-source MCP session:

### Added

- **`kglite_version` in `graph_overview()` header.** Every `<graph>`
  opening tag now carries a `kglite_version="…"` attribute sourced at
  compile time from `Cargo.toml`. Makes client-side ↔ server-side
  version skew obvious at first inspection (previously a silent failure
  mode — schema rendered by one version while subsequent queries
  routed to another). Visible in all four graph-overview shapes:
  small / medium / large / extreme inventories *and* the focused-detail
  XML returned by `graph_overview(types=[…])`.

### Changed

- **Agent-facing hints now name the MCP tool, not the Python method.**
  The XML emitted by `describe()` / `graph_overview()` is overwhelmingly
  consumed by AI agents via the `graph_overview` MCP tool, but every
  inline hint pointed at `describe(connections=…)`,
  `describe(types=…)`, etc. — agents following the hint hit a wall
  because there is no `describe` MCP tool. Renamed all agent-facing
  hints from `describe(…)` to `graph_overview(…)` in `describe.rs`
  and `topics.rs`. The Python method `KnowledgeGraph.describe(…)` is
  unchanged; the single doc-entry that documents that Python signature
  also stays as-is.

### Fixed

- **`is_test` no longer false-positives on names like `latest.html`,
  `contest.css`, `protest.swift`.** The HTML / CSS / Swift / PHP
  parsers used `rel_path.to_lowercase().contains("test")` — a loose
  substring check that misclassified every file containing the four
  letters anywhere in its path. Introduced
  `parsers::shared::is_test_path(rel_path, filename, suffix_patterns)`
  which (a) checks language-specific filename suffixes, (b) checks
  for full path segments equal to `test` / `tests` / `__tests__` /
  `spec` / `specs`. No substring matches. TypeScript also gained
  recognition of `test/` and `tests/` directories (previously only
  `__tests__/` was honoured). Go and Python keep their own narrow
  detectors (`*_test.go`, `test_*.py` / `*_test.py`) — too specific
  for the shared helper.

- **`exists(n.prop)` now steers callers to `IS NOT NULL`.** KGLite
  implements the modern pattern-existence forms
  (`EXISTS { (n)-[:R]->() }` and `EXISTS((n)-[:R]->())`) but not the
  Neo4j legacy property-existence form `exists(n.prop)`. The previous
  error message pointed at the pattern syntax — sending operators down
  the wrong rabbit hole when they actually wanted
  `WHERE n.prop IS NOT NULL`. Parser now peeks the three tokens after
  `exists(`; when they look like `<ident> . <ident>`, the error
  explicitly labels the legacy syntax, recommends `IS NOT NULL`, and
  also names the supported pattern-existence alternatives. Other
  malformed `exists(…)` calls keep an expanded generic message
  covering both alternatives.

## [0.9.36] — 2026-05-17

Web-stack language expansion: closes the biggest remaining language
gap by adding PHP, HTML, and CSS — taking KGLite to 13 supported
languages. HTML's "god-file" workflow (single-page-app HTML holding
the whole app's outline + inline JS + forms) is first-class.

Plus two docs-only commits that landed on the unpushed branch earlier:
the "avoid double version bumps" rule and the CLAUDE.md dedupe pass
(199 → 120 lines). Per the new "One version bump per push" rule, both
ride this release.

Release-mode bench gate (release_0936 vs release_0935_v3, same
binary built with --release on commit 4a83328 vs 2ecca3e):

| Bench | 0.9.35 min | 0.9.36 min |
|---|---|---|
| label_pair_counts_compute | 82.8 µs | 84.5 µs |
| planner_two_match_skewed | 6.9 µs | 7.0 µs |
| cypher_where | 170.8 µs | 205.5 µs |
| columnar_enable | 220.8 µs | 219.5 µs |
| add_nodes | 266.0 µs | 259.6 µs |
| add_connections | 430.2 µs | 429.9 µs |
| save_v3 | 418.6 µs | 416.8 µs |

Min-times stable across the board within ±5% (well inside noise per
the pytest-benchmark hygiene rule). The new parsers don't touch any
existing hot paths; their cost shows up only when a PHP/HTML/CSS file
is parsed. The new bench `code_tree_build` (god-HTML fixture)
captures the end-to-end cost of HTML parsing + embedded JS extraction:
1.16 ms min on the synthetic Flask-shaped package.

### Added (code_tree — languages)

- **PHP language parser** (`.php`). Full coverage of classes, interfaces,
  traits, methods, functions, constants, `use` imports, namespace
  declarations (backslash separator), and PHP-8 attributes
  (`#[Route('/x')]`) → DECORATES edges via the 0.9.34 pass. Trait
  declarations land as ClassInfo `kind="trait"` (matching the Rust-trait
  encoding). The `resolve_owner` helper in `builder/type_edges.rs`
  gained `\` to its separator list so HAS_METHOD edges resolve correctly
  on PHP qnames.

- **HTML language parser** (`.html`/`.htm`). God-HTML-file ready:
  emits new `Element` nodes for headings (h1-h6), elements with `id`,
  and `<form action=...>` shapes. Restraint built in — decorative
  `<div>`/`<span>`/`<p>` elements without `id` stay parse noise.
  `Element -[HAS_CHILD]-> Element` edges form the document outline.
  Inline `<script>...</script>` blocks are parsed by the existing JS
  sub-parser; resulting Functions get full CALLS-edge analysis with
  qnames scoped to `<file>:script_<n>.` so multi-block helpers don't
  collide. `<script src="...">` and `<link rel="stylesheet" href="...">`
  populate FileInfo.imports → File→File IMPORTS edges.

- **CSS language parser** (`.css`). Emits `Selector` nodes (one per
  `rule_set` regardless of selector-list count — `.foo, .bar, .baz` is
  ONE node, not three), CSS custom properties (`--my-color: red`) as
  ConstantInfo with `kind="css_custom_property"`, and `@import url(...)`
  / `@import "..."` → FileInfo.imports. `@media` / `@supports` / `@layer`
  / generic at-rules are unwrapped — their nested rule_sets emit normal
  Selector nodes. Regression guard: CSS files never emit Function or
  Class nodes.

- **`Element` and `Selector` node types**. Two new graph-schema node
  types introduced alongside the HTML and CSS parsers. The graph's
  dynamic schema picks them up automatically; planner caches
  (`label_pair_counts`, `refresh_stats`) enumerate them like any other
  type.

Language count: 10 → 13. See `docs/guides/code-tree.md` for the full
node-type list, the god-HTML-file workflow, and the CSS design-token
discovery query.

## [0.9.35] — 2026-05-17

AgensGraph-inspired planner/lookup improvements. Three commits land:

- **Label-pair edge-count cache (planner selectivity).** Generalises the
  pre-existing `type_connectivity_cache` into a lazy, mutation-invalidated
  authority on `(src_type, edge_type, tgt_type) → count`. The planner's
  `reorder_match_clauses` pass now uses per-triple counts when both
  endpoints carry a label — typically 10–100× tighter on label-skewed
  graphs than the old "all R edges" proxy.
- **`refresh_stats()` Cypher procedure.** Operator-callable cardinality
  recomputation, mostly useful as a "what does the planner see?"
  diagnostic.
- **`nodes(p)` dicts now include every node property.** Lets agents
  `UNWIND nodes(p) AS n RETURN n.age` without re-MATCHing each node.
  Wire shape unchanged.

Side fix: `maintain::add_connections` (the Python bulk-mutation path)
now invalidates the edge-cardinality caches. Pre-0.9.35 only Cypher
CREATE/DELETE did; bulk inserts left the existing `edge_type_counts_cache`
stale.

Release-mode bench gate (release_0935 vs release_0934_v2):
no consistent regressions across the 11 core benches. Min-times stable
or marginally better on 0.9.35; median noise within prior 0.9.34
variance. The planner's new selectivity branch shows up in
`test_bench_planner_two_match_with_skewed_labels` at 7.3 µs median.

**Deferred:** the node-label cache (`Vec<InternedKey>` indexed by
NodeIndex) flagged in the AgensGraph review. Profiling didn't surface
`node_type_of()` as a bottleneck against the planner-perf bench, so the
~150 lines of maintenance + parity-test code didn't penciled out. Will
revisit if it shows up as a real hot frame in a future profile.

### Changed (Cypher)

- **`nodes(path)` dicts now include every node property, not just
  `{id, title, type}`.** Lets agents use `UNWIND nodes(p) AS n RETURN
  n.age` without re-MATCHing each node to fetch property values.
  Previous dict keys are unchanged; this is purely additive — code
  that explicitly checked `set(dict.keys()) == {"id","title","type"}`
  will see extra keys. Storage and wire shape unchanged (still a
  JSON-string list under KGLite's `Value::String` convention).

### Added (Cypher)

- **`CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count`** —
  operator-callable recomputation of the label-pair edge-count
  cardinality cache. Forces a fresh O(E) walk of every edge and yields
  one row per `(src_type, edge_type, tgt_type)` triple with its current
  count. Useful for bulk-load workflows that bypass the mutation paths
  the cache invalidator listens on, and as a "what does the planner
  think the schema looks like right now?" diagnostic.

  All four YIELD columns are optional individually — the caller may
  request any subset. Output rows are sorted by
  `(src_type, edge_type, tgt_type)` for stable diffing between calls.

### Added (planner)

- **Label-pair edge-count cardinality cache.** Generalises the
  pre-existing `type_connectivity_cache` (which had only been populated
  by the n-triples loader) into a lazy, mutation-invalidated cache that
  authoritatively records `(src_type, edge_type, tgt_type) → count` for
  every graph. The Cypher planner's `reorder_match_clauses` pass now
  uses these triple counts instead of the broader edge-type totals
  when both endpoints carry a label — typically 10–100× tighter on
  label-skewed graphs (the AgensGraph-inspired pattern).

  Exposed via `KnowledgeGraph.label_pair_counts()` returning
  `[(src_type, edge_type, tgt_type, count), ...]`. Computed O(E) on
  first access; subsequent reads are essentially free
  (release-mode bench: warm read 185 ns).

### Fixed

- **Python `add_connections` now invalidates the edge-cardinality
  caches.** Pre-0.9.35 the existing `edge_type_counts_cache` would
  go stale after bulk inserts via the Python API — only Cypher
  `CREATE`/`DELETE` triggered invalidation. Sequences like
  `cypher("CREATE …") → add_connections(…) → planner-cost-driven query`
  could read stale cardinalities. Fixed alongside the new label-pair
  cache wiring.

## [0.9.34] — 2026-05-17

Code-graph expansion release: closes the feature gap with
colbymchenry/codegraph in six commits — File→File IMPORTS edges,
the `affected_tests` Cypher procedure, DECORATES edges, web-framework
route extraction (Flask/FastAPI/Django), the `explore()` one-call
codebase tool (pymethod + MCP), and a minimal Swift parser.

Release-mode benchmark snapshot on a synthetic Flask-shaped package
(saved as `release_0934` under `.benchmarks/`):

| Surface | Median |
|---|---|
| `CALL affected_tests` | 2.3 µs |
| `MATCH (Function)-[:DECORATES]->(Function)` | 6.0 µs |
| `MATCH (Route)-[:HANDLES]->(Function)` | 8.8 µs |
| `kg.explore(query)` with source slicing | 17.4 µs |
| `code_tree.build()` end-to-end | 600 µs |

Core in-memory benchmarks (`tests/benchmarks/test_bench_core.py`)
were re-run release-mode against 0.9.33 under tightened conditions
(`--benchmark-min-rounds=200 --benchmark-warmup=on`, 30-s thermal
settle between runs) and show **no real regressions**. The original
perf-gate single-run flagged three benches with apparent regressions
of +44% to +127%; the tightened runs show those deltas were
pytest-benchmark variance on M-series macOS — repeating the same
0.9.34 measurement gave `columnar_enable` medians of 516 µs and
229 µs minutes apart on the same binary, and `cypher_match` swung
between 5.6 µs and 11.6 µs across runs of identical code. The
`min`-time across all measurements stayed flat (cypher_match
~5.1 µs, columnar_enable ~219 µs), which is the cleaner signal at
these microsecond scales.

Future regressions in any of these surfaces will surface via
`make bench-compare` against the saved `release_0934` baseline.
Bench-hygiene rule lives in `CLAUDE.md`'s Performance Work Protocol:
release-mode always, min-rounds ≥ 100, thermal settle between
versions, and trust `min` over `median` for sub-millisecond
benches.

### Added (code_tree — Swift)

- **Swift parser.** New `src/code_tree/parsers/swift.rs` via
  `tree-sitter-swift = "0.7.2"`. Coverage in 0.9.34:

  - `class` / `struct` / `actor` / `enum` declarations — emitted as
    Class/Struct nodes with kind tagged from the grammar's
    `declaration_kind` field (Swift's grammar collapses all five
    into one AST node).
  - `protocol` declarations → Interface nodes with `kind="protocol"`.
  - Top-level and method `func` declarations → Function nodes with
    HAS_METHOD edges from their owning type.
  - `import Foundation` → FileInfo.imports → File→Module IMPORTS.
  - Visibility (`public` / `internal` / `fileprivate` / `private`).
  - CALLS edges resolve via the existing 5-tier name resolver.

  Follow-up scope (separate PRs): `extension` IMPLEMENTS edges,
  `init` / `subscript` / computed properties, `@objc` / `@MainActor`
  attributes as decorators, `async` / `throws` flags. The parser
  structure leaves clean slots for each.

### Added (code_tree)

- **File → File IMPORTS edges.** Sibling to the existing File → Module
  IMPORTS, resolved via a `module_path → file_path` reverse index using
  the same longest-prefix walk as the module resolver. Multiple imports
  from one source to the same target collapse into a single edge whose
  `import_count` property records the multiplicity. Enables direct
  file-level impact analysis in one Cypher hop —
  `MATCH (changed:File {path: 'src/foo.py'})<-[:IMPORTS*1..]-(impacted:File)`
  — without joining through Module nodes.

### Added (explore)

- **`KnowledgeGraph.explore(query)` and matching `explore` MCP tool.**
  One-call codebase exploration over a code-tree graph: lexically
  ranks Function/Class/Interface/Struct/Trait/Protocol/Enum nodes
  against a free-text query (name > signature > docstring), takes
  the top `max_entities`, 2-hop traverses CALLS / USES_TYPE /
  HAS_METHOD / DEFINES / REFERENCES_FN, and returns a markdown
  report with entry points, a relationship-map neighborhood, and
  grouped source slices for the entry points. Designed for the
  "how does X work in this codebase" Explore-agent question that
  would otherwise turn into chained grep + read calls — closes the
  feature gap with colbymchenry/codegraph's `codegraph_explore` tool
  while composing existing primitives rather than building a parallel
  index.

  Pyfunction signature:
  ```python
  kg.explore(query, max_entities=10, max_depth=2,
             include_source=True, source_roots=None)
  ```
  MCP tool ships with bundled methodology
  (`kglite/mcp_server/skills/explore.md`) gated on
  `graph_has_node_type: [Function, Class]` so it only activates on
  code-tree graphs.

### Added (code_tree — routes)

- **Web-framework Route extraction.** New `Route` node type plus
  `Route -[HANDLES]-> Function` edges, synthesized from decorators and
  `urlpatterns` constants. Three frameworks in v1: **Flask** (`@app.route`,
  `@app.get`/`@app.post`/..., blueprints), **FastAPI** (`@router.get`,
  `@app.get`, all HTTP verbs), and **Django** (`urlpatterns = [path('x/', view)]`
  in any urls.py-shaped file). `@app.route` decorators with `methods=[...]`
  fan out to one Route per method so `WHERE r.method = 'DELETE'` queries
  match correctly. Express, Axum, Rails, Spring, Laravel land as follow-up
  PRs — they need parser-side capture of call arguments which the
  parser model doesn't preserve today; the per-framework module layout
  under `builder/routes/` makes each subsequent framework a single-file
  addition.

  ```cypher
  MATCH (r:Route)-[:HANDLES]->(f:Function)
  WHERE r.framework = 'fastapi' AND r.method = 'POST'
  RETURN r.path, f.qualified_name
  ```

- **`urlpatterns` is now extracted as a top-level constant on Python
  files.** The constant-extraction filter previously required
  SCREAMING_SNAKE_CASE; a narrow framework-allowlist now also lets
  Django's lowercase `urlpatterns` through so the route extractor can
  read the list literal. Other lowercase names remain filtered out.

### Added (code_tree — DECORATES)

- **Function → Function DECORATES edges.** The Python/TS/Java/C# parsers
  already extracted `FunctionInfo.decorators` as raw strings; a new
  builder pass now resolves each to a target Function via the same
  bare-name lookup CALLS uses. Strips call-args (`@app.route('/x')` →
  `app.route`) and the namespace prefix (`functools.wraps` → `wraps`).
  Edge property `decorator_name` preserves the original literal so
  downstream queries don't have to re-parse the Function.decorators
  property. Unresolved (third-party) and ambiguous decorators are
  silently dropped — same stance as the call-edge resolver.

  ```cypher
  MATCH (d:Function)-[:DECORATES]->(f:Function)
  WHERE d.name = 'cache' RETURN f.qualified_name
  ```

### Added (Cypher)

- **`CALL affected_tests({files: [...], max_depth?}) YIELD test_file, depth`**
  — given a seed set of changed file paths, BFS over inbound IMPORTS
  edges and yield reachable File nodes whose `is_test` property is
  true. Either yield column is optional individually (the common case
  is `YIELD test_file` to get just a list of paths). Builds directly
  on the 0.9.34 File → File IMPORTS edges; closes parity with the
  `codegraph affected` CLI feature from colbymchenry/codegraph but as
  pure Cypher rather than a separate command.

  ```cypher
  CALL affected_tests({files: ['src/utils.rs', 'src/api.rs']})
  YIELD test_file
  RETURN test_file ORDER BY test_file
  ```

## [0.9.33] — 2026-05-14

mcp-methods 0.3.37 adopted both operator-reported fixes from the
0.9.31 deployment audit, including kglite's stop-gap full-body
skill-inject shape verbatim as the framework canonical. This is a
pin-bump release: the Rust binary path picks up the canonical
behaviour automatically; the Python entry's `_apply_skill_hint`
stays — same shape, no longer "stop-gap", now aligned with the
framework's `serve_prompts` auto-inject pass.

Folds in the unpublished 0.9.32 work (overview_prefix plumbing +
auto-inject full-body) so the cumulative diff against 0.9.31 is
one coherent release rather than two adjacent versions on PyPI.

### Changed

- **mcp-methods pin bumped to 0.3.37** (was 0.3.36). Picks up:
  - The framework's `serve_prompts` auto-inject pass now embeds
    the full skill body under a `## Methodology` header instead
    of the dangling `[See prompts/get NAME ...]` pointer.
    Operators running the standalone Rust binary
    (`crates/kglite-mcp-server/`) get the canonical behaviour
    without any kglite-side change.
  - `ResolvedRegistry.parse_warnings()` Rust getter and
    `SkillRegistry.parse_warnings` Python getter for the
    silent-skill-drop visibility (mcp-methods bug 1). The
    framework's `tracing::warn!` channel also continues to
    fire. Operator-visible boot summary integration in the
    Python entry queued for 0.9.34 (the PyPI publish of the
    0.3.37 wheel is still in flight at release-cut time).

### Fixed (MCP server)

These fixes were drafted as 0.9.32 commits but folded into 0.9.33
so a single coherent cut publishes to PyPI:

- **`overview_prefix:` from the manifest is now prepended to bare
  `graph_overview()` output.** Pre-0.9.32 the field was parsed by
  `manifest.py` but never read by `tools.py::run_overview` in the
  Python entry path. The FastMCP path at
  `mcp_methods/fastmcp/_overview.py` had honoured it correctly;
  kglite's Python entry didn't. Operators authoring documented
  `overview_prefix:` blocks were getting silently-dropped content.
  `run_overview` now accepts an optional `overview_prefix` keyword,
  prepended only on bare-overview calls (no `types=...` /
  `connections=...` / `cypher=...` drill-down args), matching the
  framework's behaviour and the documented contract.
- **`_apply_skill_hint` injects the full skill body, not a dangling
  `prompts/get` pointer.** Operator empirically confirmed that
  agents in Claude Code, Claude Desktop, Cursor, and Continue
  don't expose `prompts/get` to the model — the MCP `prompts/*`
  plane was designed for human slash commands in chat UIs, not
  agentic retrieval. The pre-0.9.33 bracketed pointer
  (`[See prompts/get NAME for full methodology.]`) was a dangling
  reference in those clients. 0.9.33 embeds the skill body under
  a `## Methodology` header in the matching tool's description so
  it reaches the agent via `tools/list`, which every MCP client
  exposes. Capped at the framework's 16 KB hard limit / 4 KB soft
  target. Operators can still set `auto_inject_hint: false`
  per-skill to suppress the embed.
- **`mcp-methods` Python wheel added to `[mcp]` extras** (`pyproject.toml`).
  Without the framework's Python wheel, `SkillRegistry.from_manifest`
  is unavailable and `skills_loader.py::load_framework_skills`
  silently returns the empty list — project-layer
  `<basename>.skills/` overrides and operator-declared domain packs
  silently don't load. CI surfaced the gap when test_o3 (the
  `auto_inject_hint: false` escape-hatch test) failed against an
  environment without the wheel manually installed.

### Added (regression tests)

- **O1**: `overview_prefix:` is prepended to bare `graph_overview()`.
- **O2**: `overview_prefix:` is NOT prepended to drill-down calls
  (`types=[...]` etc.).
- **O3**: `auto_inject_hint: false` per-skill suppresses the body
  embed in the matching tool's description.
- **O4**: Re-calling `list_tools` doesn't double-inject (the
  `## Methodology` header is the idempotency marker).
- **SK4** updated to assert the full-body embed semantics.

85/85 mcp Python tests green (was 81 in 0.9.31).

### Sequencing notes

The mcp-methods 0.3.37 wheel publish to PyPI is still in flight at
release-cut time. CI's `pip install kglite[mcp]` resolves
`mcp-methods>=0.3.36` against PyPI's current state — once 0.3.37
lands there, `pip install --upgrade` on kglite will pull it
through. The Cargo dep already resolves to 0.3.37 on crates.io, so
the Rust binary path (and the wheel's bundled Rust extension) get
the canonical inject behaviour today.

### Yanked

The 0.9.32 git commits (`50a41c6...54bd095`) describe the same
fixes that ship in 0.9.33; 0.9.32 was never published to PyPI.
Its CI workflow was cancelled in favour of the single 0.9.33 cut
so PyPI's release history stays clean. Operators who pulled
0.9.32 wheels from the in-flight build artifacts (if any) should
upgrade directly to 0.9.33.

## [0.9.32] — 2026-05-14 — unpublished

Reserved version: the `Cargo.toml` line was bumped to `0.9.32`
mid-day on 2026-05-14 in commits `50a41c6` and `54bd095`. CI was
cancelled mid-flight in favour of a single 0.9.33 cut bundling
the same fixes plus the 0.3.37 framework pin. No 0.9.32 wheel
exists on PyPI. See 0.9.33 above for the actual operator-facing
changes.

## [0.9.31] — 2026-05-14

Two same-day operator bug reports from the 0.9.31 deployment.
Both confirmed root-cause; one is kglite-side and shipped here,
the other is a framework-design issue forwarded to mcp-methods
with our reading of the trade-offs.

### Fixed (MCP server)

- **`overview_prefix:` from the manifest is now prepended to bare
  `graph_overview()` output.** Pre-0.9.32 the field was parsed by
  `manifest.py` but never read by `tools.py::run_overview` in the
  Python entry path. The FastMCP path at
  `mcp_methods/fastmcp/_overview.py` had honoured it correctly;
  kglite's Python entry didn't. Operators authoring documented
  `overview_prefix:` blocks were getting silently-dropped content.
  `run_overview` now accepts an optional `overview_prefix` keyword,
  prepended only on bare-overview calls (no `types=...` /
  `connections=...` / `cypher=...` drill-down args), matching the
  framework's behaviour and the documented contract.
- **`_apply_skill_hint` injects the full skill body, not a dangling
  `prompts/get` pointer.** Operator empirically confirmed that
  agents in Claude Code, Claude Desktop, Cursor, and Continue
  don't expose `prompts/get` to the model — the MCP `prompts/*`
  plane was designed for human slash commands in chat UIs, not
  agentic retrieval. The pre-0.9.32 bracketed pointer
  (`[See prompts/get NAME for full methodology.]`) was a dangling
  reference in those clients. 0.9.32 embeds the skill body under
  a `## Methodology` header in the matching tool's description so
  it reaches the agent via `tools/list`, which every MCP client
  exposes. Capped at the framework's 16 KB hard limit / 4 KB soft
  target. Operators can still set `auto_inject_hint: false`
  per-skill to suppress the embed (useful for clients that DO
  expose prompts/get, or where context cost matters more than
  reachability).

  This is a kglite Python-entry stop-gap; the framework's
  canonical fix is being discussed with the mcp-methods maintainer
  (their `serve_prompts` auto-inject pass would benefit from the
  same upgrade for the Rust binary path + every other framework
  consumer). When that lands, this kglite-side implementation
  will converge.

### Added (regression tests)

- **O1**: `overview_prefix:` is prepended to bare `graph_overview()`.
- **O2**: `overview_prefix:` is NOT prepended to drill-down calls
  (`types=[...]` etc.).
- **O3**: `auto_inject_hint: false` per-skill suppresses the body
  embed in the matching tool's description.
- **O4**: Re-calling `list_tools` doesn't double-inject (the
  `## Methodology` header is the idempotency marker).
- **SK4** updated to assert the full-body embed semantics
  (previously asserted the dangling-pointer shape).

85/85 mcp Python tests green (was 81 in 0.9.31).

### Acknowledged but not yet fixed (forwarded to mcp-methods)

- **`SkillRegistry.from_manifest` silently drops files with YAML
  frontmatter parse errors.** Operator hit this with a
  colon-in-value in an unquoted `description:` field; spent 25
  minutes debugging because no log line surfaces. Framework owns
  the parser; mcp-methods inbox has the bug report with operator-
  ranked fixes (log.warning on each skipped file; return
  parse_warnings on the registry; scaffold helper hints).
- **Skills via `prompts/get` are unreachable in real MCP clients.**
  Forwarded as a framework-wide design issue with the operator's
  ranked fixes (auto-inject full body — adopted here as a
  stop-gap; expose `get_skill` as a tool; document the
  limitation). Maintainer decides the canonical shape; we'll
  converge once they ship.

## [0.9.31] — 2026-05-14

Skills-aware MCP. Ships kglite-authored methodology for the four
custom tools (`cypher_query`, `graph_overview`, `save_graph`,
`read_code_source`) plus the wiring to compose them with framework
defaults + operator-side layers + predicate-gated filtering. Opt-in
per manifest via `skills: true` (or a path list); existing
deployments without that declaration see no behavioural change.

### Added (MCP server)

- **Four bundled skills** under `kglite/mcp_server/skills/`,
  authored against mcp-methods 0.3.35's `writing-effective-skills.md`
  guide (TRIGGER/SKIP descriptions, Overview → Quick Reference →
  Common Pitfalls → When wrong body anatomy, ~150-220 lines each).
  Shipped as Python package data; `include_str!`'d into the Rust
  binary at `crates/kglite-mcp-server/src/main.rs`. One source of
  truth across both shipping paths.
- **`SkillRegistry` wiring in the standalone Rust binary**
  (`crates/kglite-mcp-server/src/main.rs`): `add_bundled` for each
  of the four kglite skills, `merge_framework_defaults`,
  `auto_detect_project_layer`, `layer_dirs(manifest.skills)`,
  predicate evaluator (`KglitePredicateEvaluator` consults
  `graph_state.has_node_type` / `has_property` for the
  `graph_has_node_type:` / `graph_has_property:` clauses),
  `finalise`. Wired into `serve_prompts(&registry, &mut server)`
  before the stdio loop.
- **Python entry point prompts handlers** at
  `kglite/mcp_server/server.py`. The lowlevel `mcp.server.Server`
  surface doesn't have the framework's FastMCP-shaped
  `register_skills_as_prompts` helper, so we hand-roll
  `@server.list_prompts()` and `@server.get_prompt()` backed by
  `skills_loader.build_active_skill_set(...)`. Predicate gating is
  re-evaluated at request time so post-boot graph state changes
  (workspace activation, watch rebuild) reflect immediately.
- **`kglite/mcp_server/skills_loader.py`** — minimal frontmatter
  parser, `Skill`/`AppliesWhen` dataclasses, three-layer merge
  (kglite-bundled + framework + operator), and runtime
  `applies_when:` evaluation. Lives entirely in Python; talks to
  the framework via `mcp_methods.SkillRegistry.from_manifest(...)`
  for the framework+operator layers.
- **Auto-inject hint pass.** When a skill's `name` matches a
  registered tool and `auto_inject_hint: true` (default), the tool's
  `description` gains a `[See prompts/get <name> for full
  methodology.]` pointer in `tools/list`. Agents that scan tools
  first still discover the methodology surface.
- **`Manifest.skills` field** on the Python dataclass
  (`kglite/mcp_server/manifest.py`) parsed from the framework's
  polymorphic JSON shape (`false` / array of `true`/path entries).

### Changed

- **mcp-methods pin bumped to 0.3.36** (was 0.3.34). Picks up
  `applies_when:` predicate gating (0.3.36) and skills foundation
  (0.3.35). Both shipped by the maintainer in response to our
  design feedback within the same day; no design changes during
  review.

### Added (regression tests)

- **SK1**: Manifest without `skills:` exposes no prompts (opt-in
  property; pre-0.9.31 behaviour preserved by default).
- **SK2**: `skills: true` exposes kglite-bundled skills via
  `prompts/list`. read_code_source filtered out via
  applies_when on non-code graph fixtures.
- **SK3**: `prompts/get cypher_query` returns the bundled
  markdown body with description.
- **SK4**: Auto-inject hint appends `[See prompts/get ...]` to
  matching tool descriptions when skills are enabled.
- **SK5**: `read_code_source` skill ACTIVE on a code-tree graph
  (Function/Class present); proves the predicate evaluator
  consults live graph state.
- **SK6**: `prompts/get` with unknown name returns a clean
  JSON-RPC error.

81/81 mcp Python tests green (was 75 in 0.9.30). Standalone Rust
binary builds clean; `cargo fmt --check` + `cargo clippy -- -D
warnings` clean across the workspace.

### Example

`examples/open_source_workspace_mcp.yaml` opts in with `skills:
true` and carries explanatory comments covering the three value
shapes (`true` / single path / list form) plus the `applies_when:`
predicate behaviour for the legal / o&g / code deployment shape.

## [0.9.30] — 2026-05-14

Operator-reported friction from the 0.9.29 deployment audit:
agent-facing schema clarity (Item 2), MCP tool-search round-trip
counts (Item 1), and identical-tool-surface ambiguity across
multiple kglite servers (Item 3). All four items fixed in one
release.

### Fixed (code_tree)

- **`module` property is now populated on every code-tree entity
  type, not just File and Module.** Operator reported
  `MATCH (f:Function) WHERE f.module STARTS WITH 'xarray.core'
  RETURN f` returned zero rows — the property only existed on
  File/Module nodes. The code_tree builder now looks up each
  Function/Class/Constant/Enum/Interface/Trait/Protocol/Struct's
  file_path in a file → module_path map and populates a `module`
  property derived from the parent file's module. Module nodes
  also get a `module` alias of their `qualified_name` for cross-
  type uniformity. Result: `WHERE n.module STARTS WITH '...'`
  works against any node label without branching.

### Added (introspection)

- **`<prop sample="..." />` attribute on high-cardinality
  properties in `describe()` / `graph_overview` output.**
  Pre-0.9.30 the schema XML showed `vals="..."` for properties
  with ≤15 unique values (low-cardinality enums); high-
  cardinality properties (docstring, signature, file_path with
  hundreds of values) showed only `unique=N` with no example.
  Now one example value is emitted as a `sample="..."` attribute
  whenever `vals=` would be omitted, so the agent always sees
  what the property *looks like* (e.g. `signature` becomes
  `sample="def to_list (self) -> list[dict[str, ..."` instead
  of just `unique="16"`). Same logic applied to edge property
  stats in `<connections>` blocks.

### Added (MCP server)

- **Auto-injected ToolSearch batch-load hint into the server's
  `instructions:` field.** Operator reported deferred-tool loaders
  (Claude Code's ToolSearch) gating each tool family per-tool,
  forcing N round trips to load N tools from one server. The
  server now prepends a one-paragraph hint to operator-declared
  `instructions:` describing the
  `ToolSearch(query='+<server-slug>', max_results=20)` batch-load
  pattern (one round trip per server). Idempotent: composing
  twice does not duplicate the hint. Operators who want full
  control can include the literal marker `[kglite-batch-load-hint]`
  in their instructions text to suppress auto-injection.
- **`tools[].bundled: <name>` overrides accept a `rename:` field
  (mcp-methods 0.3.34+).** Operator reported that running three
  kglite servers exposing identical bundled surfaces produced
  six near-identical entries in ToolSearch results, ambiguous
  to rank. `rename:` lets operators expose a bundled tool under
  a per-deployment name (e.g. `legal_cypher_query`,
  `prospect_cypher_query`) while the canonical handler still
  runs the body. Composes with the existing `description:` and
  `hidden:` overrides. Boot-time validation refuses renames
  that shadow another bundled tool, another rename, or a
  manifest-declared cypher tool.

### Changed

- **mcp-methods pin bumped to 0.3.34** (was 0.3.33). Picks up
  the `tools[].bundled: rename:` extension. The framework
  patch was applied locally to mcp-methods, tested (16/16
  bundled tests green; +6 new entries covering rename
  validation and JSON shape), and proposed to the maintainer
  via inbox note.

### Added (regression tests)

- **B14**: `bundled: cypher_query` + `rename: legal_cypher_query`
  exposes the renamed identifier in `tools/list` and removes the
  canonical name from the listing.
- **B15**: Call to renamed tool dispatches through the canonical
  handler (proves rename isn't visible-only).
- **B16**: Renaming a bundled tool to a name that shadows
  another bundled tool fails at boot with a clear collision
  error.
- **B17**: Renaming to a name that shadows a cypher tool fails
  at boot.
- **I1 / I2 / I3**: Batch-load hint appears in composed
  instructions, idempotent across repeated composition,
  suppressible by operator-supplied marker.
- **S1**: `module` property is populated on Function / Class /
  Constant / Module / File nodes uniformly (operator's literal
  reproducer).
- **S2**: `describe()` emits `sample="..."` for high-cardinality
  properties.

75/75 mcp tests green (was 66 in 0.9.29). 603/603 Rust unit
tests green.

## [0.9.29] — 2026-05-14

Two operator-reported fixes from the post-0.9.28 deployment audit:
a hardcoded port-collision default that made parallel server boot
impossible, and a workspace manifest layout that forced explicit
`--mcp-config` for the natural folder shape. The second is fixed
upstream in mcp-methods 0.3.33 via an opt-in `workspace.applies_to`
declaration — kglite 0.9.29 bumps the pin to pick it up.

### Fixed (MCP server)

- **`csv_http_server` default port changed from 8765 to 0
  (OS-assigned).** When Claude Desktop launches multiple
  kglite-mcp-servers concurrently at startup, the first server
  to boot used to grab port 8765 and every subsequent server
  crashed with `OSError: address already in use` — surfaced to
  the user as "Server disconnected" with no actionable detail.
  The default now binds to port 0 (kernel-assigned); the actual
  bound port is captured back into the config so `url_for()`
  produces correct URLs (via `runner.addresses`, aiohttp's
  public API for this). Operators who need a stable port for
  external integrations can still set `port: 9000` explicitly.
- **Workspace manifest auto-discovery walks one level up when
  the parent manifest opts in via `workspace.applies_to`.**
  Operators with the natural layout

  ```text
  open_source/
  ├── workspace_mcp.yaml      # declares `workspace.applies_to: ./*`
  └── repos/                  # --workspace points here
  ```

  no longer have to pass `--mcp-config` explicitly. The opt-in
  declaration accepts a literal name (`./repos`), a glob pattern
  (`./prod-*`), or a list of patterns (`[./repos, ./clones]`).
  Patterns match the workspace dir's basename; the parent walk
  is bounded to one level. Without `applies_to` declared, the
  parent-walk is refused — a deliberate safety property to
  prevent silent-wrong-manifest if `--workspace` points at any
  unrelated sibling under a workspace-manifest parent.

  Implementation lives upstream in mcp-methods 0.3.33's
  `server::manifest::find_workspace_manifest`. kglite's Python
  wrapper is now a thin pass-through; the unconditional
  parent-walk fallback we briefly considered for the wrapper
  was abandoned after the maintainer flagged its silent-wrong-
  manifest failure mode.

### Changed

- **mcp-methods pin bumped to 0.3.33** (was 0.3.31). Picks up
  the `workspace.applies_to` opt-in (above) plus the cumulative
  framework changes from 0.3.32 (initial `applies_to` design as
  a single literal, superseded by 0.3.33's glob + list shape).
- **`WorkspaceCfg` dataclass gains `applies_to: str | list[str]
  | None`** (`kglite/mcp_server/manifest.py`) — parsed from the
  framework's polymorphic JSON shape and passed through to
  consumers verbatim.

### Added (MCP server)

- **C9 / C10 / E6 / E7 regression tests.** C9:
  `csv_http_server: true` produces a URL with a non-zero port.
  C10: two concurrent servers booted with the default both come
  up alive without port collision. E6: parent-walk discovery
  with `applies_to: ./*` resolves the parent manifest. E7
  (safety property): parent-walk is refused when the parent
  manifest has no `applies_to` AND when an `applies_to` literal
  doesn't match the child's basename.
- **Migration guide extended** (`docs/migrations/mcp-0.6-to-0.9.md`):
  new "Translation cheat-sheet" section maps common pre-0.9.x
  patterns to the new mode flags ("loads a .kgl at boot →
  `--graph`", "uses set_root_dir → `workspace.kind: local`",
  etc.). New ".kgl format compatibility" section calls out
  that 0.6.x – 0.8.x graphs cannot be loaded by 0.9.x and must
  be rebuilt. Workspace manifest auto-detection section
  describes `applies_to` opt-in.
- **Example manifest updated** (`examples/open_source_workspace_mcp.yaml`):
  declares `workspace.applies_to: ./*` to demonstrate the
  layout-B (manifest beside workspace dir) pattern. Header
  comment shows both layouts side-by-side.

## [0.9.28] — 2026-05-14

Fixes three bugs the mcp-servers operator surfaced in their 0.6.18 →
0.9.27 deployment-verification audit: workspace mode wasn't actually
building code-tree graphs on activate, `local_workspace` mode booted
with an empty graph, and `kglite.code_tree` attribute-chain access
raised at runtime. Two of the four servers they were migrating
couldn't work end-to-end before this release; they can now.

### Fixed (MCP server)

- **`--workspace` mode now actually builds graphs on activate.** The
  workspace's `post_activate` hook was registered as a Python wrapper
  in 0.9.24 but never wired into `_build_server` — so
  `repo_management('org/repo')` would clone the repo, no code-tree
  build would fire, and the next `cypher_query` returned
  `No active graph.` The hook is now wired and fires on both
  `repo_management` activate and `set_root_dir`. Triggers
  `graph_state.build_code_tree(active_path)` + `source_roots[:] =
  [active_path]` so source tools (`read_source`, `grep`,
  `list_source`) target the active clone.
- **`workspace.kind: local` mode builds the code-tree at boot.**
  Previously local-workspace booted with an empty graph until the
  agent issued the first `set_root_dir`. Mirrors watch mode's
  boot-time `build_code_tree(mode_path)` so the first
  `cypher_query` against a freshly-booted local-workspace server
  sees a populated graph.
- **`kglite.code_tree` attribute-chain access works again.**
  `kglite/mcp_server/tools.py::GraphState.build_code_tree` was
  calling `kglite.code_tree.build(...)` as an attribute chain on the
  kglite package, but `kglite/__init__.py` doesn't import the
  submodule eagerly — so the call raised `AttributeError` at runtime
  the first time the workspace post-activate hook tried to fire.
  Now uses `from kglite import code_tree` to force the submodule
  load, with a clean error if the bundled tree-sitter grammars
  aren't available.

### Added (MCP server)

- **Migration guide for operators upgrading from 0.6.x – 0.8.x**
  (`docs/migrations/mcp-0.6-to-0.9.md`). Covers the shift from
  custom Python MCP scripts to the bundled manifest-driven
  `kglite-mcp-server`: operating modes, tool surface differences
  (`read_source` split, `grep_source` → `grep`, `ripgrep` is not
  a bundled name), embedder transition
  (sentence-transformers/torch/MPS → fastembed/ONNX), manifest
  cheat-sheets, and common gotchas. Linked from `docs/index.md`.
- **F6/F7/F8 regression tests.** F6: `local_workspace` mode builds
  code-tree at boot. F7: `set_root_dir(child)` rebuilds the
  code-tree for the new root via the post-activate hook. F8:
  `from kglite import code_tree` loads successfully. 62/62 mcp
  tests green (was 59 in 0.9.27).

## [0.9.27] — 2026-05-13

Picks up the `tools[].bundled:` override shape from mcp-methods
0.3.31 and the cross-binary `repo_management` gating fix that
landed in the same framework release. Closes a customisation gap
that had been forcing operators to stuff per-tool guidance into
the global `instructions:` block.

### Added

- **`tools[].bundled:` override shape** — manifests can now
  customise the agent-facing surface of bundled tools without
  declaring them inline. Two override types:
  - `description: "..."` — replaces the bundled tool's default
    agent-facing description (what shows in `tools/list`).
    Lets operators teach agents that `repo_management` is the
    FIRST STEP, or that `cypher_query` returns a specific
    dataset shape, without burying the guidance in the global
    instructions blob.
  - `hidden: true` — drops the tool from `tools/list` AND rejects
    direct call attempts with `Error: tool 'X' is hidden by
    manifest configuration.` Useful for narrowing the agent
    surface (e.g. hiding `ping` on a production server, or
    suppressing source tools when the auto-bound `source_root`
    is wider than the operator wants).

  Both validate against the kglite bundled-tool catalogue at
  boot: a typo in the `bundled:` name exits 3 with
  `ERROR: unknown bundled tool name(s) ... Valid names: [...]`
  listing the full catalogue.

  Example:

  ```yaml
  tools:
    - bundled: repo_management
      description: |
        FIRST STEP for this server. Call repo_management('org/repo')
        to clone + build a repo before any other tool.

    - bundled: ping
      hidden: true                # narrow the agent surface

    - name: similar_sessions      # existing cypher-tool shape unchanged
      cypher: ...
  ```

  Cypher tools (`tools[].cypher` entries) carry their own
  description in the manifest entry and are NOT affected by
  bundled overrides — those apply to the fixed bundled catalogue
  only (`cypher_query`, `graph_overview`, `ping`,
  `read_code_source`, `save_graph`, `read_source`, `grep`,
  `list_source`, `repo_management`, `set_root_dir`,
  `github_issues`, `github_api`).

- **Four regression tests** in
  `tests/test_mcp_server_python_entry.py`:
  - **B10** — `bundled: cypher_query` with `description:` appears
    in `tools/list` with the override text.
  - **B11** — `bundled: ping` with `hidden: true` drops `ping`
    from `tools/list` while leaving other tools intact.
  - **B12** — calling a hidden bundled tool by name returns the
    `hidden by manifest configuration` error rather than
    falling through to "unknown tool."
  - **B13** — an unknown bundled name (`bundled: cipher_query`)
    fails at boot with an error listing the valid catalogue.

  103/103 mcp + extensions_schemas tests green (was 99 in 0.9.26
  + 4 new).

### Changed

- **mcp-methods pin: 0.3.30 → 0.3.31** (auto-resolved via
  Cargo.toml's `version = "0.3"` constraint; Cargo.lock locks
  the exact 0.3.31). The framework half of the bundled-override
  work landed in 0.3.31 alongside our implementation; we did the
  Rust changes ourselves and the maintainer reviewed + released
  with no revisions. The same release also fixed the
  `repo_management` cross-binary gating drift we'd flagged from
  the operator's post-0.9.25 verification — `mcp-server` (bare
  framework) and `kglite-mcp-server` now register the tool with
  the same gating rules.

### Internal

- `BUNDLED_TOOL_NAMES` frozenset at `kglite/mcp_server/server.py`
  defines the catalogue against which `tools[].bundled:` names
  are validated. Adding or removing a bundled tool requires
  updating this set; manifest overrides will surface "unknown
  bundled tool" errors otherwise.

## [0.9.26] — 2026-05-13

Operator-driven release combining a CLI fix that unblocks the
wikidata pure-YAML migration, the Cat G-N fixture acceptance
that closes the 0.9.16 → 0.9.25 arc, a disk-storage write-path
guard that turns silent data loss into a clean error, and a
significant docs sweep across `docs/guides/mcp-servers.md`.

### Fixed

- **`kglite-mcp-server --graph` now accepts disk-backed graph
  directories**, not just single `.kgl` files. Pre-0.9.26 the
  validator (`server.py::_validate_mode_paths`) used
  `Path.is_file()` which silently rejected any directory — even
  though `kglite.load(path)` (the Python API) accepts both shapes
  fine. The error message even read "does not exist" when the
  path was demonstrably a valid graph directory, which was
  misleading. The new validator accepts a path if EITHER it's a
  regular file (the `.kgl` case) OR a directory containing the
  `disk_graph_meta.json` sentinel (the disk-graph case, same
  marker the Rust loader at `src/graph/io/file.rs::load_file`
  uses). Reported by the mcp-servers operator after they shipped
  the wikidata preprocessor migration against 0.9.25; the bug
  blocked the last 84 lines of `wikidata_mcp_server.py` from
  being deleted (their disk-backed 124M-node Wikidata graph
  couldn't boot via the CLI). Anyone deploying a
  `storage="disk"` graph (the documented kglite path for
  >50M nodes) hit this immediately.
- **Cypher `CREATE` / `MERGE` on `storage="disk"` graphs now
  fails loudly** (returns a clear `Cypher error` pointing at
  `add_nodes()` / `to_disk()` workarounds) instead of silently
  succeeding-with-no-data. Pre-0.9.26 the disk `add_node` path
  only stored a slot (type + row_id) and dropped the
  `NodeData.properties` / `title` / `id` fields, so `CREATE
  (:Marker {title: 'x'})` against a disk graph completed without
  error but every property and the auto-title vanished — both
  in-memory (the column store was never told) and after
  save/reload. Discovered while writing the B6 regression test;
  not on any reported issue list but a silent-corruption class
  worth surfacing. Affects only `CREATE` and the create-path of
  `MERGE`; `SET` / `DELETE` on disk-backed graphs work
  correctly. The proper disk write-path implementation is on
  the roadmap.

- **Cypher `REMOVE n.prop` on `storage="disk"` graphs now works
  correctly.** Discovered as a silent no-op during the
  disk-CREATE guard investigation: the disk staged-write flush
  (`flush_node_mut_cache`) only persisted property keys *present*
  in the staged Map, so a bare `properties.remove(key)` from
  `NodeData::remove_property` left the column store untouched
  and reads returned the original value. Fix: a new
  `NodeData::clear_property` helper inserts `Value::Null` for
  the key instead, which the flush writes through to the column
  store. `execute_remove` now routes to `clear_property` on
  disk-backed graphs via an `is_disk()` branch; memory and
  mapped backends keep the prior in-place `remove_property`
  behaviour (no change). Verified by B9 (regression test) +
  parity with the documented `SET n.prop = null` path.
- **`DiskGraph::node_weight` debug-assertion no longer fires on
  false-positive cases.** The 0.9.0 Cluster 6 hygiene check at
  `disk/graph.rs::node_weight` previously fired whenever
  `node_mut_cache` had ANY entry for the index being read. That
  included `PropertyStorage::Columnar { row_id, .. }` scratch
  entries left by `batch.rs::flush_chunk` (the `add_nodes`
  path) — those are "already persisted via full-Arc
  replacement, safe to discard" and not a missed-flush concern.
  The check now filters to non-empty `PropertyStorage::Map`
  entries only (the actual Cypher-style staged writes that
  WOULD be shadowed by a column-store read). Removes the
  warning noise that appeared during normal `maturin develop`
  test runs. Debug-only assertion; never appeared in release
  builds.

### Added

- **B6 / B7 / B8 / B9 regression tests** in
  `tests/test_mcp_server_python_entry.py`:
  - B6 — `--graph <disk-graph-dir>` boots the server and serves
    a `cypher_query` against the persisted nodes (built via
    `add_nodes()`, the supported disk-mode write path).
  - B7 — `--graph <arbitrary-directory-without-meta>` is still
    rejected with the new error message, so the validator
    isn't too permissive.
  - B8 — disk-mode Cypher `CREATE` / `MERGE` returns the new
    loud-failure error pointing at `add_nodes` / `to_disk`,
    not a silent no-op.
  - B9 — disk-mode Cypher `SET` and `DELETE` still work normally
    (the guard is narrow by design; existing mutation paths
    must not regress).
- **Cat J / K / L test fixtures and 8 forward-looking tests** —
  the mcp-servers operator delivered the fixture bundle that was
  the last open thread from the 0.9.16 → 0.9.25 arc. Four tiny
  `.kgl` files (5.9 KB total) + paired manifests under
  `tests/fixtures/{spatial_graph,timeseries_graph,graph_with_orphans,graph_with_duplicates}.kgl`,
  with the fixture catalog at `tests/fixtures/CAT_G_N_FIXTURES.md`
  documenting what each one anchors. The wired tests:
  - **J1-J3** (spatial Cypher) — `contains(area, point(lat, lon))`,
    `centroid(polygon)`, query-side `point(...)` literal lookup.
  - **K1-K3** (timeseries Cypher) — `ts_sum(channel, 'YYYY')`,
    `ts_at(channel, 'YYYY-M')`, and `ts_sum` across multiple
    matched nodes. Asserts against the random-seeded values
    in the fixture (TROLL oil 2019 sums to 1563.55; March 2019
    spot is 177.12).
  - **L1-L2** (procedures) — `CALL orphan_node({type:'Wellbore'})`
    returns 3 isolated nodes; `CALL duplicate_title({type:'Prospect'})`
    returns 4 duplicate-set members across two pairs.
  97/97 mcp + schemas tests green (was 89 in this release before
  the fixture wire-up).

### Changed

- **`docs/guides/mcp-servers.md` — significant sweep on top of
  yesterday's six accuracy fixes** (`51436df`). 0.9.26 adds:
  - Quick Start renumbered 1–4 (was 1, 2, 2½, 3); manifest
    teaser moved after Claude registration so the install-and-
    point happy path reads top-to-bottom.
  - Stale `--embedder` CLI-flag reference (line 58) removed —
    that flag doesn't exist; the supported path is
    `extensions.embedder` in a manifest.
  - "Custom embedders" subsection under Built-in patterns
    rewritten as a 4-line pointer to the
    `extensions.embedder` reference + worked example (was
    using the removed-in-0.9.18 `embedder: { module, class }`
    shape with `--trust-tools`).
  - `read_source` / `grep` / `list_source` parameter docs
    reformatted from dense single-line prose to per-tool
    parameter tables.
  - New "Deployment shapes" section with a "Large graphs
    (disk-backed)" subsection covering the canonical Wikidata-
    scale flow (ntriples loader → `storage="disk"` → CLI
    pointed at the directory).
  - New "Known limitations" section documenting the disk-
    `CREATE`/`MERGE` refusal, the disk-`REMOVE` silent no-op,
    and the `repo_management` cross-binary gating drift between
    `kglite-mcp-server` and the bare `mcp-server` CLI.
  - New "Troubleshooting" section gathering common post-boot
    pitfalls (GITHUB_TOKEN discovery, `text_score()` returning
    zero, warm-call slowness, conda PATH shadowing, tools
    missing from `tools/list`, PyPI simple-index lag).
  - Pre-0.9.20 migration sections (90 LoC) moved to
    `docs/migrations/mcp-pre-0.9.20.md`, leaving a one-line
    pointer in the main guide.
- **`mcp-methods` dependency switched from git+rev to crates.io**
  (commit `f53e8f1`, also between releases). mcp-methods 0.3.30
  was the first crates.io publish; library binary surface is
  functionally identical to 0.3.29's `71f7ba6`. The switch is
  cosmetic Cargo.toml tidy — no behaviour change. Cargo.lock
  locks the exact version for reproducible builds.
- **`mcp-methods` dependency switched from git+rev to crates.io**
  (commit `f53e8f1`, also between releases). mcp-methods 0.3.30
  was the first crates.io publish; library binary surface is
  functionally identical to 0.3.29's `71f7ba6`. The switch is
  cosmetic Cargo.toml tidy — no behaviour change. Cargo.lock
  locks the exact version for reproducible builds.

## [0.9.25] — 2026-05-12

Doc + feature release driven entirely by the mcp-servers operator's
end-of-arc audit (`inbox/read/2026-05-12-from-mcp-servers-end-of-arc-audit.md`).
The operator flagged that 0.9.24 was "genuinely solid" but they'd
hesitate to recommend `kglite-mcp-server` to a third party because
the docs left them inbox-thread-dependent for edge cases. 0.9.25
addresses every gap in the audit (eight reference doc sections + four
worked examples + machine-readable JSON schemas) and ships the one
feature that retires their last custom Python MCP server
(`extensions.cypher_preprocessor`).

### Added

- **`extensions.cypher_preprocessor`** — manifest-declarable Python
  hook that fires before every `cypher_query` and `tools[].cypher`
  invocation. The hook can rewrite the query string and/or params
  before they reach `graph.cypher(...)`. Gated by
  `trust.allow_query_preprocessor: true`. The motivating use case
  is Wikidata Q-number rewriting (`{nid: 'Q42'}` →
  `{id: 42}` against the integer-id graph), but the hook generalises
  to date normalisation, multi-tenant scoping, parameter validation,
  and any "rewrite agent input before query execution" shape that
  pure-declarative regex can't express. Class-based loaders thread
  `kwargs:` through to `__init__`; free-function loaders work for
  state-free rewriters. Boot-time errors (trust gate, missing module,
  missing class/function) exit 3 with the operator-facing message;
  runtime exceptions surface as `preprocessor: <message>` in the
  tool body without leaking a traceback. ~50 LOC implementation in
  `kglite/mcp_server/preprocessor.py`; 9 regression tests
  (`test_o1`-`test_o9` in `tests/test_mcp_server_python_entry.py`)
  cover the full contract from in-process unit dispatch to
  end-to-end YAML round-trip through MCP stdio.
- **Eight reference doc sections in `docs/guides/mcp-servers.md`** —
  fills every gap the operator's end-of-arc audit called out:
  mode × YAML-field acceptance matrix; tool gating rules; tool
  response formats (with stability tags so the 0.9.21 row-formatter
  regression class can't recur silently); `extensions:` schema
  reference; `tools[].cypher` template reference ($param semantics,
  JSON Schema flavour, error envelope, FORMAT CSV inheritance);
  embedder backend × model catalog; path resolution + manifest
  discovery rules; operator notes (pip-index lag workaround,
  conda guidance, watch-mode rebuild costs).
- **Four worked manifest examples under `docs/examples/`** —
  `manifest_cypher_tool.md`, `manifest_with_embedder.md`,
  `manifest_workspace.md`, `manifest_cypher_preprocessor.md`. Wired
  into the docs guide via a toctree.
- **Machine-readable JSON Schema (Draft 2020-12)** for each
  first-class `extensions.*` block under `docs/schemas/extensions/`.
  Linked from the reference docs. Anchored to the Python parsers
  by `tests/test_extensions_schemas.py` (44 tests) — schema/parser
  drift fails loudly in CI.
- **`Manifest.trust` dataclass field** — `allow_python_tools`,
  `allow_embedder`, `allow_query_preprocessor` populated from
  `mcp_methods::server::Manifest::to_json()` output. Available to
  the rest of `kglite/mcp_server/` (and to tests).

### Changed

- **mcp-methods pin: 0.3.28 → 0.3.29** (rev `1ba9469` → `71f7ba6`).
  Adds `allow_query_preprocessor` to `ALLOWED_TRUST_KEYS` and
  `TrustConfig`, plus emits it under the `trust` object in
  `Manifest::to_json()`. Non-breaking JSON shape addition.
- **`kglite.mcp_server.tools.run_cypher` signature** — adds an
  optional `preprocessor: Preprocessor | None = None` parameter.
  Existing callers (every prior release plus all current internal
  call sites) work unchanged via the default. Same for
  `kglite.mcp_server.cypher_tools.call_cypher_tool`.

### Internal

- Pre-release suite expanded to 39 default-mode tests (was 34):
  added the 9 new cypher_preprocessor tests (O1-O9) plus 44 schema
  drift tests in `tests/test_extensions_schemas.py`.
- W1 watch-callback assertion loosened to accept either the changed
  file path or the parent directory — macOS FSEvents coalesces
  depending on rate, and the contract we care about is "the
  callback receives a `list[str]` within the debounce window," not
  the path-granularity decision the OS makes.

## [0.9.24] — 2026-05-12

Architectural cleanup: kglite's MCP server framework is now a thin
shim over `mcp-methods` Rust rather than a parallel Python
re-implementation. ~600 LOC of Python deleted; the validated Rust
behaviour replaces it. As a side effect, the 0.9.23 set_root_dir
sandbox-narrowing regression the operator flagged is fixed by
construction.

### Fixed
- **`set_root_dir` no longer narrows the sandbox with each swap.**
  The pre-0.9.24 Python `Workspace.set_root_dir_tool` mutated
  `self.root` after each successful swap, so the next sandbox check
  compared against the narrower active root rather than the
  manifest's declared `workspace.root`. After one swap, lateral
  swaps to sibling projects under the configured root failed with
  "escapes the workspace root." Fix: the new wrapper inherits
  `mcp_methods::server::Workspace`'s atomic-swap RwLock + immutable
  configured `workspace_dir`, so the sandbox check always validates
  against the manifest's declared root. Workspaces remain swappable
  to any sibling under the configured root for the lifetime of the
  server — no restart required.

### Changed
- **`kglite/mcp_server/{manifest,workspace,watch}.py` are now thin
  pyo3 wrappers** around `mcp_methods::server::{Manifest,Workspace,
  watch_dir}`. The Python surface stays the same (`Manifest`
  dataclass, `Workspace.root` / `.kind` / `.repo_management_tool` /
  `.set_root_dir_tool`, `watch.start`), but the implementation is
  ~600 LOC of Rust behind a single passthrough each. The Python
  manifest dataclass is populated from `Manifest::to_json()` (new
  in mcp-methods 0.3.27) so field drift between the framework and
  downstream consumers is a non-issue.
- **`.env` walk-up delegated to mcp-methods Rust**
  (`load_env_walk`). Same parse rules (skip blanks / `#` comments,
  strip outer quotes, no-overwrite-existing-env), same result —
  one fewer parallel implementation to keep in sync. Explicit
  `env_file:` paths still loaded inline.
- **File watcher uses `notify-debouncer-mini` via Rust** instead of
  the pure-Python `watchdog` + threading debounce. The
  `watchdog` extra remains in `[mcp]` optional-deps for now —
  downstream tooling may depend on it — but kglite's own watch path
  no longer uses it.
- **Pin: `mcp-methods` rev `1ba9469` (0.3.28).** Three same-day
  releases against this cleanup: 0.3.26 (three-crate split), 0.3.27
  (`Manifest::to_json`), 0.3.28 (local-mode `set_root_dir` no longer
  clobbers `active_repo_path`). Two of those bumped specifically to
  unblock the 0.9.24 pyo3 wrapper; the third was a bug found during
  the wrap pass.

### Added
- **`kglite._mcp_internal.{Manifest, Workspace, WatchHandle,
  start_watch, load_env_walk}`** — pyo3 wrappers around
  `mcp_methods::server::*`. Internal surface (the public Python
  entry point is still `kglite.mcp_server.server:main`), but
  importable for tests and for downstream tools that want the
  validated mcp-methods behaviour without a Python re-implementation.
- **4 new regression tests** anchoring the pyo3-wrapper boundary:
  F4 (sandbox lateral swap — operator's bug repro), E5 (manifest
  extensions passthrough — recursive `serde_json::Value` → dict),
  F5 (workspace post-activate hook GIL dispatch), W1 (watch
  callback receives changed paths). 34/34 mcp_server tests green
  (was 30/30).

### Internal
- `anyhow` added as a direct dep — required by
  `mcp_methods::server::PostActivateHook`'s `Result<(), anyhow::Error>`
  signature.
- `MEMORY.md` of "thin Python shim" intent updated for future
  sessions (see `CLAUDE.md`'s new "Standard plan procedure" section
  for the per-phase commit rhythm we followed for 0.9.24).

## [0.9.23] — 2026-05-12

### Fixed
- **`extensions.csv_http_server` returned HTTP 500 on every GET.**
  aiohttp's `web.Response` rejects `content_type` strings that
  contain a charset directive — we were passing
  `"text/csv; charset=utf-8"`. Fix: pass `content_type="text/csv"` and
  `charset="utf-8"` as separate kwargs. Operator workaround was to
  disable the csv_http_server block; 0.9.23 makes it usable again.
- **`github_issues` now auto-defaults to the workspace's active repo
  when `repo_name` isn't supplied.** Previously `repo_management(name)`
  activated the repo correctly but `github_issues` without
  `repo_name` hit the "could not auto-detect from git remote" error
  path. Workspaces now track `active_repo` and the github_issues
  dispatcher uses it as the fallback. Agents no longer need to
  repeat `repo_name='org/repo'` on every call.
- **`set_root_dir` now actually rebinds the source tools.** Previously
  the tool registered, the workspace's `root` field updated, but the
  `source_roots` list captured at server build time wasn't refreshed
  — so `list_source` / `grep` / `read_source` kept hitting the old
  root. Tests/test_f2 was designed to catch exactly this and did.
- **`BgeM3Embedder.unload()` is now a no-op** (formerly dropped the
  ONNX session). kglite's `kg_core.rs::cypher` does
  `load → embed → unload` around every text_score call, so dropping
  the session meant every cypher paid the full ~1s ORT session init.
  Warm-call latency drops from ~1.1s to ~50ms; ~2 GB RAM stays
  resident while the embedder is in use.

### Added
- **27-test pre-release suite per the operator's spec.** Every test
  maps to a specific bug from the 0.9.16 → 0.9.22 release arc.
  Categories: A (install/boot), B (per-mode tool registration),
  C (tool output content), D (embedder + semantic search), E
  (manifest/.env), F (workspace state propagation). The
  tool-output content assertions (Cat C) are the gate that would
  have caught 0.9.21's row formatter and 0.9.22's csv_http 500
  before release.
- **bge-m3 cool-down timer**: configurable via
  `extensions.embedder.cooldown` (default 900s = 15 min;
  `0` = never release). Active sessions hold the ONNX session
  resident for fast queries; long-idle servers release ~2 GB of
  RAM. Cool-down check fires on each `embed()` call — no background
  threads.
- `BgeM3Embedder.release()` — explicit counterpart to the no-op
  `unload()`. Drops the ORT session + tokenizer when the caller
  really wants the memory back.
- `tests/fixtures/build_tiny_graph.py` — programmatic 50-node-per-
  type fixture (Person + Company + Article) with semantically
  clustered article bodies (quantum / baking / programming) for
  embedder relevance tests.

### Changed
- **`extensions.embedder` YAML now accepts `cooldown:`** (seconds).
  Falls through to the BgeM3Embedder for `BAAI/bge-m3`; FastEmbed
  adapter ignores the field for other models (their lifecycle
  follows fastembed-python's defaults).

## [0.9.22] — 2026-05-12

### Fixed
- **`cypher_query` row formatter now returns row values, not column
  names.** 0.9.21 regression: `_format_inline` in
  `kglite/mcp_server/tools.py` iterated `for v in row` against a dict,
  yielding the column names as values. Every non-CSV `cypher_query`
  call produced rows like `'f.name'\t'f.line_number'` instead of the
  actual data. Operator caught it on redeploy and rolled back to
  0.9.18. Fix: index the row dict by column (`row[col]`) so the
  preview shows real values. The Rust-side Cypher engine + the
  `FORMAT CSV` path were always correct — only the inline preview
  formatter was wrong.

### Added
- `test_cypher_query_returns_actual_row_data` integration test —
  asserts the inline preview contains the computed value (e.g. `2`)
  and **not** the column name (`'sum'`). The 0.9.21 regression class
  ("tool registers but returns garbage") can no longer reach release
  without breaking the build. Same shape as the existing per-tool
  content assertions (ping returns `pong`, read_source returns the
  file slice, grep returns matches).

## [0.9.21] — 2026-05-12

Fixes the two 0.9.20 regressions the operator caught on redeploy:
8 of 11 tools were silently missing, and bge-m3 embedder was broken
because fastembed-python doesn't carry that model in its catalog.

### Added
- **All 8 framework tools restored**: `ping`, `read_source`, `grep`,
  `list_source`, `repo_management`, `set_root_dir`, `github_issues`,
  `github_api`. Same output format as the 0.9.18 binary. Implementation
  comes from the pure-Rust `mcp-methods` crate (0.3.26+, three-crate
  split with zero pyo3 in the library half) wrapped via pyo3 in
  `src/mcp_tools.rs` and exposed as `kglite._mcp_internal`. The
  Python `kglite.mcp_server.server` entry point dispatches each
  tool to the wrapped Rust function — `cypher()`-style GIL release
  preserves the original Rust performance.
- **`BgeM3Embedder`** (`kglite/mcp_server/bge_m3.py`): direct
  onnxruntime + huggingface_hub implementation for `BAAI/bge-m3`
  because fastembed-python's catalog doesn't include it. Same ONNX
  weights as fastembed-rs, same CLS pooling, same
  `~/.cache/fastembed/` cache directory — operator's existing
  downloaded weights reused without re-download. Other models
  (bge-small/base/large, all-MiniLM-L6-v2, multilingual-e5) continue
  through fastembed-python.
- **CI gate against tool-surface regression**:
  `tests/test_mcp_server_python_entry.py` boots the server in every
  supported mode and asserts `tools/list` matches
  `tests/fixtures/tool_baseline.json` exactly. Any added or removed
  tool fails the build. Adopted in response to the 0.9.20 failure mode
  where the regression was caught after release.

### Changed
- `kglite::api` Rust facade adds `mcp-methods 0.3.26` as a curated
  dep (`default-features = false, features = ["server"]`, no pyo3 in
  its tree). Downstream Rust consumers can use `mcp_methods::*`
  directly without going through us.
- `extensions.embedder` dispatcher routes `BAAI/bge-m3` to the new
  `BgeM3Embedder`; all other models continue to fastembed-python.
- Auto-bind manifest's directory as fallback source root when no
  `source_root:` is declared. Matches the 0.9.18 binary's behaviour
  — without this fallback, sodir-style manifests (no source_root)
  silently lost `read_source`/`grep`/`list_source`.

### Fixed
- 0.9.20's tool-surface regression (8 missing tools per manifest).
- 0.9.20's bge-m3 catalog regression (`text_score()` broken on every
  embedder-enabled deployment).

## [0.9.20] — 2026-05-11

### Changed
- **`kglite-mcp-server` is now a Python console-script entry point,
  not a bundled Rust binary.** The 0.9.18/0.9.19 binary bundling
  forced a 12-wheel (3 OS × 4 Python) build matrix because any
  Rust binary that transitively depends on pyo3 links libpython at
  a specific version — no abi3 escape for binaries. The Python
  entry point bypasses that entirely. Wheel matrix back to 3 abi3
  wheels per release (same as pre-0.9.18). Performance unchanged:
  kglite's Python `cypher()` already releases the GIL inside
  `py.detach()`, so the wrapping layer is sub-microsecond.
- The 0.9.18 conda install_name regression and the 0.9.19
  `install_name_tool` / `patchelf` / mold post-build surgery are
  gone — there's no binary to mis-link.
- Wheel deps: install via `pip install 'kglite[mcp]'` to pull the
  server-time deps (mcp, pyyaml, fastembed, aiohttp, watchdog).
  Plain `pip install kglite` skips them — for users who just want
  the graph engine.

### Removed
- `kglite/_bin/` directory inside the wheel (was per-Python binary
  drop site).
- `kglite/_cli.py` (was the launcher that exec'd the bundled
  binary).
- Per-Python-version wheel build matrix axis.
- `install_name_tool` / `patchelf` / mold steps from the CI
  workflow.
- `[project.optional-dependencies] embeddings` (torch +
  sentence-transformers) — superseded by `[mcp]` which uses
  fastembed natively.

### Internal
- `crates/kglite-mcp-server/` stays in the repo for direct Rust
  consumers (Wikidata-scale deployments, Docker-vendored binaries)
  but is no longer bundled into the wheel. Build via
  `cargo build -p kglite-mcp-server` if you want it.

## [0.9.19] — 2026-05-11

### Changed
- **Wheel build is substantially faster.** Three workflow changes:
  - Drop fastembed's `image-models` default feature (jpeg/png/webp
    decoders we don't use) — ~3-4 min/wheel saved.
  - Add `Swatinem/rust-cache` to the wheel-build workflow with a
    per-target shared key so most deps (mcp-methods, hyper,
    fastembed, tokio) are reused across the four Python-version
    cells within each OS. Warm builds reuse the cache.
  - Use the `mold` linker on Linux. The bundled ld spent ~1-2 min
    linking the cdylib + binary at end-of-build; mold does the same
    work in ~10s. macOS and Windows keep their platform linkers.

### Fixed
- **`pip install kglite` now works on conda Python.** The 0.9.18 wheel
  shipped the bundled `kglite-mcp-server` binary with an absolute
  install_name pointing at `/Library/Frameworks/Python.framework/...`
  (the actions/setup-python build path); conda installations don't
  have that path and the binary failed to launch with a dyld error.
  The wheel-build workflow now rewrites the install_name to
  `@rpath/libpython3.X.dylib` and adds an rpath relative to the
  binary's wheel install location so dyld finds the env-local
  libpython under conda, venv, virtualenv, and Python.org installs
  uniformly. Linux gets the same treatment via `patchelf --set-rpath
  '$ORIGIN/../../../..'`.
- **`builtins.temp_cleanup: on_overview` now actually wipes the
  configured directory.** The 0.9.18 implementation hardcoded the
  cleanup target to `./temp` (cwd-relative) which only worked when
  the server was launched from the manifest's parent directory.
  0.9.19 resolves the temp directory against the manifest base — and
  reuses `extensions.csv_http_server.dir` when configured, so the
  same place CSVs are written is also the place that gets swept.
- **`FORMAT CSV` row-count status no longer reports `0 row(s) written`
  for queries with `LIMIT N`.** The status counter read
  `result.rows.len()`, which is empty when the planner's lazy
  materialisation kicks in — even though the CSV body has the right
  data. The count now comes from the CSV body itself.

## [0.9.18] — 2026-05-11

### Changed
- **MCP server is now pure-Rust at the source level.** `kglite-mcp-server`
  no longer calls PyO3 anywhere — every tool handler goes through the
  new `kglite::api` façade (Cypher pipeline, `compute_description`,
  `build_code_tree`, `source_location`). The mcp-methods Python
  feature is off, so the framework's Python tool surface isn't on
  the binary's dep graph either. The bundled binary still links to
  libpython transitively through kglite's own PyO3 layer (kglite is a
  Python library — that's by design), so the wheel matrix stays at
  3 OS × 4 Python = 12 wheels per release.
- **Embedder backend switched to fastembed-rs.** The framework-level
  `embedder:` Python factory is gone. Configure with
  `extensions.embedder: { backend: fastembed, model: BAAI/bge-m3 }`
  instead — bge-m3, bge-small/base/large-en-v1.5, all-MiniLM-L6-v2,
  and multilingual-e5 supported out of the box. ONNX weights are
  downloaded to `~/.cache/fastembed/` on first use; no `torch` /
  `sentence-transformers` install needed.

### Added
- `kglite::api` — curated Rust façade for downstream binaries. Exposes
  `KnowledgeGraph` + `DirGraph` + `Embedder` + `FastEmbedAdapter` +
  the Cypher parse/plan/execute surface + `compute_description` +
  `compute_schema` + `load_file` + `build_code_tree` +
  `SourceLocation`/`SourceLookup`.
- `extensions.csv_http_server` — opt-in localhost HTTP listener that
  serves `FORMAT CSV` exports as URLs instead of inline strings.
  Useful for million-row exports that would blow the MCP response
  budget. Bound to 127.0.0.1, path-traversal hardened, no write
  surface, CORS-enabled.
- `KnowledgeGraph::set_embedder_native(Arc<dyn Embedder>)` — pure-Rust
  counterpart to the `set_embedder` pymethod; lets downstream Rust
  binaries bind embedders without a `Py<PyAny>`.
- `KnowledgeGraph::source_location(name, node_type)` — pure-Rust
  counterpart to `graph.source()` used by the `read_code_source` tool.

### Removed
- `tools[].python` manifest entries — Python tool hooks no longer
  loadable. Move tool logic into a `tools[].cypher` template or a
  downstream Rust binary that embeds the kglite crate directly.
- `embedder:` top-level manifest key — replaced by
  `extensions.embedder:` (see Changed).
- Pre-0.9.18 install-UX workarounds (`PYO3_PYTHON=`,
  `install_name_tool -add_rpath`, conda-env symlinks) are no longer
  needed; `pip install kglite` ships `kglite-mcp-server` on `PATH`
  directly.

## [0.9.17] — 2026-05-11

### Added

- **`read_code_source(qualified_name=...)` MCP tool** — kglite-side
  companion to the framework's `read_source(file_path=...)`. Resolves a
  fully-qualified entity name through the active graph's
  `graph.source()` (which uses the code-tree node attributes), then
  reads the corresponding file slice from the configured source root(s).
  Equivalent to cypher → graph.source → read_source in a single MCP
  call. Same `start_line` / `end_line` / `grep` / `max_chars` filters
  as `read_source`. Restores the qualified-name flow operators relied
  on pre-0.9.14; reported by the MCP-servers operator after the
  0.9.14 framework take-over trimmed `read_source` to `file_path`-only.
- Boot-summary line on stderr now names the `.env` file actually
  loaded (or reports `(no .env found)` when walk-up came up empty),
  closing the gap between "token missing" and "token present but
  unreadable".

### Fixed

- **`kglite-mcp-server` now actually loads `.env` files.** The
  shim's `main.rs` never invoked mcp-methods'
  `load_env_for_mode()` — so the framework's walk-up + `env_file:`
  YAML key support, although present in mcp-methods 0.3.22+, never
  fired under the kglite binary. Operators ran into "GITHUB_TOKEN
  not set" with a `.env` one directory up from their workspace,
  which should have been auto-discovered. Now wired: walk-up from
  `--graph` parent / `--source-root` / `--workspace` /
  `--watch` / `workspace.kind: local` root / cwd-in-bare, with
  explicit `env_file:` in the manifest as override.
- **`embedding_diagnostics()` now sees columnar properties.** The
  0.9.16 implementation iterated `NodeData::property_iter()`, which
  yields nothing for `PropertyStorage::Columnar` — the variant nodes
  use after save+reload. As a result, diagnostics on a freshly-loaded
  graph reported `nodes_with_property: 0` for properties that
  actually existed (Cypher `WHERE x IS NOT NULL` confirmed), flipping
  the status to `store_orphan` on a healthy steady-state graph. Same
  root cause hid the `embeddable` status when a `node_type` filter
  was passed for a type with a string property but no store yet.
  Fixed by switching to `properties_cloned()`, which dispatches
  across all `PropertyStorage` variants. Two new regression tests
  cover the save+reload and filter-by-type paths.

### Documentation

- **Python linkage policy for `kglite-mcp-server`** — PyO3 picks one
  interpreter at build time. New section in `docs/guides/mcp-servers.md`
  ("Where does the binary find Python? — read this before
  `pip install`") covers the discovery one-liners (`otool -L` /
  `ldd`) and the `PYO3_PYTHON=...` install-time override. README has
  a short callout pointing to the long version. Reported after an
  operator landed 2 GB of `torch` in base conda because the binary
  linked to base Python rather than the sub-env where they pip'd
  `kglite`.

## [0.9.16] — 2026-05-10

### Added

- **YAML `tools[].cypher` entries are now wired into MCP.** The
  `kglite-mcp-server` shim adds a `cypher_tools` module that
  registers each manifest-declared parameterised Cypher tool as a
  first-class MCP tool, dispatching to `graph.cypher(template,
  params=args)` on the active graph. Closes the gap that left all
  five of the MCP-servers project's production manifests'
  `tools:` sections invisible to agents after the Python
  `kglite.mcp_server` was retired. Schema is taken from the YAML
  `parameters:` block when present, otherwise an empty object schema.
- **`manifest.workspace.kind: local` is now honored.** The shim
  promotes a manifest-declared local workspace into a new internal
  `Mode::LocalWorkspace` before mode-specific binding, with
  `set_root_dir` registered for runtime root swap and an optional
  debounced watch loop on `watch: true`. Manifest declaration wins
  over the `--workspace` CLI flag, mirroring the framework's own
  binary. Lets users retire `code_review_mcp_server.py`-style
  custom Python servers in favour of a YAML manifest.
- **`graph.embedding_diagnostics(node_type=None)`** — companion to
  `list_embeddings()` that surfaces per-`(node_type, text_column)`
  coverage with three states: `"embedded"` (store and property both
  present), `"embeddable"` (property present, no store), and
  `"store_orphan"` (store present, no node has the property — the
  symptom an `import_embeddings()` warning indicates). Use it after
  a silent-drop warning to see which stores are affected.
- Type stubs for `import_embeddings()` and `export_embeddings()` —
  both methods existed but were missing from `kglite/__init__.pyi`.
- Documentation of the `code_tree` qualified-name format per
  language with a stability commitment within minor releases —
  `docs/guides/code-tree.md`.
- Recipe: `SET` → `add_properties` migration for hub aggregations,
  with a worked example showing `Agg.count()` / `Spatial.distance()`
  helpers replacing imperative-Cypher `WITH ... SET ...` chains —
  `docs/guides/recipes.md`.
- End-to-end smoke suite for `kglite-mcp-server` over JSON-RPC stdio
  (`tests/test_mcp_server_smoke.py`) — 25 tests covering every tool
  the binary exposes (`cypher_query`, `graph_overview`, `save_graph`,
  `read_source`, `grep`, `list_source`, `github_issues`, `github_api`,
  `set_root_dir`, `ping`, plus YAML-declared parameterised Cypher
  tools). Auto-skips when the binary isn't built; runs in ~3 s.

### Changed

- **`kglite-mcp-server` now pins mcp-methods 0.3.23** (rev `e45a282`,
  bumped from 0.3.21). Brings, in order: `.env` auto-loading,
  GitHub-tool drill-down via `element_id`, honest tool listing gated
  on `GITHUB_TOKEN`, `inventory.json` `last_built_sha` + auto-rebuild
  gating on `repo_management(update=True)`, framework parsing of
  `workspace.kind: local`, the `mcp_methods.fastmcp` Python helper
  submodule (`register_overview` / `register_cypher_query` /
  `register_source_tools` / `register_save_graph` / `serve_csv_via_http`),
  a public `build_tool_attr` for downstream cypher-tool registration,
  and an empty-string filter in `auth_token`. Existing YAML manifests
  parse unchanged — every schema addition is optional.
- The framework's embedder factory now hands back an
  `Arc<EmbedderHandle>` (load/unload/embed/touch + idle tracking)
  instead of a raw `Py<PyAny>`. The shim extracts the underlying
  Python instance via `handle.instance()` and binds that to the
  active graph; kglite's per-batch `set_embedder` lifecycle drives
  the same instance the framework's idle-watch task observes.
- `README.md` migration note: replaces the old `pip install
  "kglite[mcp]"` flow with `cargo install --path crates/kglite-mcp-server`.

### Fixed

- **`import_embeddings()` no longer silently drops mismatched files.**
  When `imported == 0` but the `.kgle` file contained data, or when
  a per-type store had zero matches, the call now emits a
  `UserWarning` describing the mismatch (file path, counts, likely
  cause). The result dict gains a `dropped_stores` key so callers
  can detect partial-drop cases programmatically. Reported via the
  MCP-servers wishlist after a 7 MB embedding file silently became
  `{stores: 0, imported: 0, skipped: 1923}` against a graph whose
  `code_tree` qualified-name format had drifted.
- **`save_disk` no longer fails with `OSError: Invalid argument
  (os error 22)` on disk-backed graphs.** The 0.9.15 unified
  mega-file writer had an early-return gate that required *both*
  `total_bytes == 0` *and* `unhandled.is_empty()` — but unhandled
  types only need sidecar fallback, never bytes in the mega-file.
  With non-zero `unhandled` and zero planned bytes, the code fell
  through to `mmap::map_mut` on a 0-byte file, which returns EINVAL
  on every Unix. Triggered on every fresh disk graph
  (`KnowledgeGraph(storage="disk", ...).add_nodes(...).save(...)`) —
  the entire `tests/test_disk_property_index.py` suite was failing.

## [0.9.15] — 2026-05-10

### Added

- `KnowledgeGraph._save_subset_induced_by_edge_type(path, edge_types)`
  — variant of `_save_subset_filtered_by_edge_type` that produces the
  *induced* subgraph: the kept-node set is still derived from edges
  matching `edge_types` (Pass A), but the output keeps **every** edge
  between any two kept nodes, not just the filter edge type. On the
  Wikidata `articles_authors` carve this expands the result from a
  single P50 layer to ~174 M edges across 20 distinct types
  (P2860 citations, P2093 stated authors, P98 editor, etc.) while
  still pinning the node set to "articles + their authors". Disk
  source only.

### Fixed

- **Streaming subgraph carve now round-trips non-schema properties.**
  The disk-to-disk `save_subset_streaming_disk` writer dropped any
  property whose key wasn't in the type's schema — Wikidata stores
  most low-cardinality properties (P356 DOI, P577 publication date,
  P304 page numbers, …) in a per-row overflow bag, and those were
  silently lost on save. `TypeWriter` now accumulates an overflow
  blob in the same wire format the source uses
  (`[u16 num_entries] + [u64 key | u8 type_tag | value]`), and
  `RowVisitor` routes non-schema keys into it instead of dropping
  them. `ColumnStore::replace_overflow_bag` is the new setter.

- **Saved DiskGraphs now load with mmap-fast-path speed.** A graph
  built in memory and persisted via `save_disk` previously emitted
  per-type zstd sidecars under `columns/<type>/columns.zst`, which
  the loader rebuilt eagerly on every open — ~70 s on a 17 M-node
  carve vs. ~150 ms for the same data when produced by the ntriples
  builder. `save_disk` now emits the unified `seg_000/columns.bin`
  mega-file format the loader's mmap fast path consumes (new
  `crate::graph::io::unified_columns` module), so saved subgraphs
  load in tens of milliseconds. Existing sidecar-format graphs still
  load via the legacy path.

### Performance

- Legacy sidecar column loader (still used for pre-mega-file files)
  parallelised via rayon — read + zstd decode + `load_packed` now
  run per-type concurrently. ~2.3× faster on a 16-core machine for
  the rare case of opening a sidecar-format graph.

## [0.9.14] — 2026-05-09

### Added

- **`kglite-mcp-server` is now a Rust-native single binary**, built
  on top of the `mcp-server` framework (rmcp + manifest-driven tool
  registration) shipped from the sibling `mcp-methods` workspace.
  The binary lives at `crates/kglite-mcp-server/`. Full mode coverage
  matches the previous Python server: `--graph X.kgl`, `--workspace
  DIR`, `--watch DIR`, `--source-root DIR`, plus bare framework. The
  manifest YAML schema is unchanged, so any `<basename>_mcp.yaml`
  written for the Python server boots unchanged on the new binary.
- Workspace mode auto-builds a code-tree graph for each cloned repo
  via a `PostActivateHook` calling `kglite.code_tree.build()`; watch
  mode re-runs the same path on debounced file changes; embedder
  factories declared in the manifest are bound to the active graph
  via `graph.set_embedder()`.

### Removed

- **`kglite/mcp_server/` Python package + the `kglite[mcp]` extras
  dependency group + the `kglite-mcp-server` console script entry +
  `examples/mcp_server.py` + 11 `tests/test_mcp_*.py` modules**
  (~4,150 LoC). All replaced by the Rust binary above. The manifest
  schema and tool surface are 1:1 compatible — agents see the same
  `cypher_query` / `graph_overview` / `save_graph` / `read_source` /
  `grep` / `list_source` / `github_issues` / `github_api` /
  `repo_management` / `ping` tools as before.
- `mcp` optional-dependency group removed from `pyproject.toml`. To
  install the new server: `cargo install --path crates/kglite-mcp-server`
  from a kglite source clone.

### Changed

- `kglite` is now a Cargo workspace (root crate + `crates/kglite-mcp-server`).
  The Python wheel build via maturin is unchanged; a new
  `python-extension` Cargo feature gates `pyo3/extension-module` so
  `cargo build` can share the rlib with the new sibling binary.
- `CLAUDE.md` "When changing a `#[pymethods]` function" checklist
  step 4 now points at `crates/kglite-mcp-server/src/tools.rs`
  (instead of the deleted `examples/mcp_server.py`).

## [0.9.13] — 2026-05-09

### Added

- **`kglite-mcp-server --workspace DIR`: multi-graph workspace mode.**
  Boots the server without a graph; the agent activates one with
  `repo_management('org/repo')`, which clones the GitHub repo, builds
  a code-graph via `kglite.code_tree.build`, and pins it as the active
  graph for `cypher_query` / `graph_overview` / `read_source` / `grep`
  / `list_source`. Inventory tracks `last_accessed` / `access_count`
  per repo in `<workspace>/inventory.json`. Idle repos auto-sweep
  after `--stale-after-days` (default 7); the active repo is exempt
  and stale entries preserve their access history. Layout:
  `<workspace>/{repos,graphs,temp,inventory.json,workspace_mcp.yaml}`.
- **Manifest `embedder:` section** for project-supplied embedder
  factories. Declare `module: ./embedder.py` + `class: GraphEmbedder`
  + `kwargs: {...}` and the CLI imports + instantiates via
  `Class(**kwargs)` and binds with `graph.set_embedder()`. Trust-gated
  by `trust.allow_embedder: true` plus `--trust-tools` (both signals
  required, mirrors the `python:` tool gate). Replaces the
  always-loaded `--embedder MODEL_NAME` shortcut for users who need
  cooldown-based unload (e.g. BAAI/bge-m3 on consumer hardware).
- **Manifest `overview_prefix:` field.** Sticky preamble prepended to
  `graph.describe()` output on bare `graph_overview()` calls. Skipped
  for focused drill-downs (`types=[...]`, `connections=[...]`,
  `cypher=[...]`) so they stay terse. Lets agents re-discover
  load-bearing context (validator hints, baseline counts, hidden
  invariants) deep into a session without competing with the
  conversation for slot in the system instructions.
- **Manifest `builtins:` section** for pre-blessed tools that don't
  need `--trust-tools`. `save_graph: true` registers a `save_graph()`
  MCP tool that calls `graph.save(graph_path)` — for persisting
  CREATE/SET/DELETE Cypher mutations. `temp_cleanup: on_overview`
  clears the CSV-export `temp/` directory on bare `graph_overview()`
  calls so it doesn't grow unbounded across long sessions; `never`
  (default) keeps the existing behaviour.
- **`read_source(qualified_name=...)`** for code-aware servers. When
  the bound graph carries `qualified_name` + `file_path` properties on
  code nodes, the agent can pass a name like `MyClass.my_method` and
  the tool resolves through `graph.source()` to a file slice in one
  round-trip (was: cypher-then-read, two round-trips). Suffix
  fallback handles short bare names (e.g. `helper_fn`) via Cypher
  `ENDS WITH` against `qualified_name`. Available in both single-
  graph and workspace modes.

### Changed

- `kglite-mcp-server` `--graph` and `--workspace` are now mutually
  exclusive flags. Default behaviour (no flag, no `graph.kgl` in cwd)
  is unchanged — single-graph mode looking for `./graph.kgl`.
- `examples/conference_graph_mcp.yaml` annotated with the new
  manifest fields (`overview_prefix`, `builtins`, `embedder`,
  `trust.allow_embedder`).

## [0.9.12] — 2026-05-09

### Added

- **`KnowledgeGraph.save_subset(path)` on the fluent selection chain.**
  Equivalent to `kg.to_subgraph().save(path)` in a single call —
  produces an independent v3 binary file that reloads via
  `kglite.load(path)` (or `load(path, storage='disk')` for disk mode).
  All edges between selected nodes are included; node and edge
  properties round-trip byte-for-byte. Works on any source storage
  mode.
- **`_save_subset_filtered_by_edge_type(path, [edge_types])` —
  disk-to-disk streaming subgraph filter** (Wikidata-scale path).
  Single-pass over the source's `edge_endpoints.bin` builds a kept-
  nodes bitset; a per-type `TypeWriter` then streams kept rows
  directly to dest column files via `BufWriter`s — no intermediate
  in-memory `ColumnStore`, no chunk-and-merge step. End-to-end on the
  full Wikidata graph (124M nodes / 861M edges, P50 + endpoints)
  produces a 17,364,495-node / 35,448,243-edge subgraph in 349s wall
  time (down from 550s on the v1 in-memory path). Working set stays
  in the hundreds of MB regardless of subset size — peak RSS is
  largely soft mmap pages.
- **`BorrowedValue<'a>` zero-copy view of `Value`** in
  `datatypes::values`. `String(&'a str)` borrows from the source
  buffer (typically an mmap region) instead of cloning into a
  `String`. Used by the streaming subgraph filter; available as a
  general read-path primitive. Convert with `to_value()`.
- **`MmapColumnStore::id_borrowed` / `title_borrowed` /
  `try_for_each_property_borrowed`** — allocation-free reads. The
  property visitor decodes overflow-bag bincode entries in place
  and yields `BorrowedValue::String` views into the mmap; previously
  every overflow row allocated a `Vec<(InternedKey, Value)>` plus a
  `String` per entry. ColumnStore wrapper delegates to mmap_store
  for disk graphs.

### Changed

- **`MmapColumnStore::read_str` skips UTF-8 validation.** Source
  bytes were always written through `String::as_bytes()` (Rust's
  UTF-8 invariant), so the `from_utf8` validator was walking ~25 GB
  of source data per Wikidata save for nothing. `from_utf8_unchecked`
  is now used. Saves ~70 s of wall time on the streaming subgraph
  filter; no observable behavior change.
- **Streaming subgraph filter uses borrowed-value writes
  end-to-end.** `TypeWriter::push_row_borrowed` accepts
  `BorrowedValue<'_>` and writes `&[u8]` straight to per-column
  `BufWriter`s without ever materializing `Value::String`. Combined
  with the read-side borrowing above, the Wikidata save phase drops
  from 550s to 349s (-37%). Node-walk sub-phase: 446s → 241s
  (-46%); the `props` portion 330s → 145s (-56%); `id+title` 87s →
  17s (-80%). Output is byte-identical to the prior path —
  `tests/test_subgraph_streaming.py`'s round-trip equality suite
  (14 cases) stays green.
- **Sub-phase timers in `save_subset_streaming_disk` gated on
  `KGLITE_STREAMING_TIMING=1`.** Off by default (zero overhead);
  when enabled, prints per-phase wall times plus per-million-row
  progress so a future optimizer can iterate in 30-second chunks
  rather than 10-minute round-trips. Useful for finding bottlenecks
  without committing to a full bench cycle.

### Changed

- **Edge-driven group-by aggregations with a typed target node now use
  the fast `lookup_peer_counts` path.** Queries shaped
  `MATCH (a)-[:E]->(b:T) RETURN b, count(a) [ORDER BY count(a) DESC LIMIT k]`
  were correctly routed to `FusedMatchReturnAggregate` but BOTH executor
  branches (top-K and non-top-K) bailed when the planner reversed the
  pattern to start at the typed node — the resulting `group_elem_idx == 0`
  short-circuit forced the slow node-centric scan. The fast path now
  detects "group is semantic target" via `(group_elem_idx, edge_direction)`
  and applies a binary-search type filter against `type_indices[T]`
  (sorted by construction).
  On Wikidata:
    - museums-by-works (with ORDER BY): 15 s → 108 ms (140×)
    - most-eponymed-globally: 122 s timeout → 169 ms (~720×)
    - top-influencers: 122 s timeout → 26 ms (~4700×)
    - typed-target without ORDER BY (LIMIT only): 13.3 s → 110 ms (~120×)
    - untyped-target without ORDER BY (LIMIT only): 14.3 s → 535 ms (~27×)
  No on-disk format change; all-Wikidata graph rebuild not required.
  Differential-test queries `edge_groupby_typed_target_top_k` and
  `edge_groupby_typed_target_no_orderby` added to the corpus to gate
  both branches.
- **`MATCH...WITH count(...)` aggregations now use the fast path
  post-pattern-reversal too.** The third instance of the same
  position-only `group_elem_idx == 2` bug was in
  `try_fast_with_aggregate_via_histogram` (the executor's fast path
  for `FusedMatchWithAggregate`). After
  `optimize_pattern_start_node` reverses `(a)-[:E]->(b:T) WITH b,
  count(a)` to start at the typed node, group_elem_idx becomes 0
  and the histogram path silently bailed despite
  `lookup_peer_counts` serving both shapes. Same direction-aware
  predicate fix as the RETURN-aggregate paths, plus
  `count(<edge-var>)` now fuses through the WITH-aggregate gate too.
  c7's `MATCH (n)<-[r]-() WITH n, count(r)` now reaches
  `FusedMatchWithAggregate` (still bounded by the absence of a
  global-in-degree histogram for untyped edges; that's a follow-up
  workstream). Differential test
  `edge_groupby_match_with_aggregate_typed_target` added.
- **`count(<edge-variable>)` now fuses into `FusedMatchReturnAggregate`.**
  `MATCH (paper)<-[r:CITES]-(citing) RETURN paper.title, count(r)` is
  the natural shape for the Wikidata citation graph, but the gate at
  `fuse_match_return_aggregate` only accepted `count(<other-node-var>)`
  — `count(r)` for the edge variable bailed silently. Semantically
  equivalent for a 3-element pattern (each edge is one peer binding),
  so the fix accepts both. On Wikidata: most-cited scholarly articles
  drops from 198s timeout to 28s (~7×, bounded by `lookup_peer_counts`
  HashMap construction over P2860's hundreds-of-millions of edges).
  Differential test `edge_groupby_count_edge_variable` added.
- **`ORDER BY <agg-expr>` now fuses equivalently to `ORDER BY <alias>`.**
  `fuse_match_return_aggregate`'s top-K absorption matched only
  `ORDER BY <alias-name>` (a Variable expression matching a RETURN
  alias). Writing the same query as `ORDER BY count(x)` (an expression
  duplicating a RETURN item's expression) left ORDER BY + LIMIT
  unfused in the pipeline, so the fused MATCH-RETURN-aggregate
  produced every distinct peer's row (245k for `:P138` on Wikidata)
  before downstream OrderBy + Limit trimmed to k. On Wikidata this
  cost 8 s vs the alias-form's 175 ms for the same query. Absorption
  now also matches via `expression_to_column_name` so both forms
  fuse. Differential test
  `edge_groupby_orderby_expression_form` added.
- **Group-by-source aggregations now use a fast path too.** Queries
  shaped `MATCH (h:T)-[:E]->(other) RETURN h, count(other) ORDER BY
  count(other) DESC LIMIT k` (e.g. "humans with most awards") were
  hitting the slow node-centric scan, which on Wikidata's
  `:human` (13.4M nodes) timed out at 30s/75s with random mmap reads
  thrashing the page cache. The fast path now detects "group is
  semantic source" via `(group_elem_idx, edge_direction)` —
  `(0, Outgoing)` for the user-written form, `(2, Incoming)` for the
  post-reversal form — and computes source-keyed counts on the fly via
  `count_edges_grouped_by_peer(.., Direction::Incoming)`, a sequential
  scan of `edge_endpoints`. Sequential I/O is the right shape for this
  workload (see `feedback_disk_io_patterns.md`). On Wikidata,
  `humans-with-most-awards` drops from 30-75s timeout to 54s answer —
  bounded by the sequential edge-scan I/O ceiling, not by query-engine
  inefficiency. Smaller graphs (social_graph) see sub-millisecond
  results. Differential test
  `edge_groupby_source_typed` added.

## [0.9.11] — 2026-05-07

### Docs

- **Getting Started rewritten** to lead with bulk-load
  (`add_nodes` / `add_connections` from DataFrames) instead of
  three single-row `cypher("CREATE ...")` statements. The old
  ordering misrepresented the day-1 workflow — every real project
  loads data through the columnar path. Single-CREATE demoted to
  an "Ad-hoc inserts" callout. Adds the missing
  `pip install "kglite[mcp]"` line and a preview of the bundled
  CLI + `source_root:` one-liner.
- **New audience-ranked guide index** at `docs/guides/index.md`
  groups the 14 how-to guides by intent: load-bearing path
  (data-loading → cypher → mcp-servers), domain-specific
  (code-tree, spatial, timeseries, etc.), power-user, and
  "if you want to know why". Sidebar toctree reordered to match.
- **MCP Servers guide polished**: 3-line "What's MCP?" intro
  with link to modelcontextprotocol.io; "Five tools from one yaml
  line" preview pulled into the Quick Start so the source_root:
  ROI lands before the Claude Desktop config; new "Common boot
  errors" subsection with eight error→fix mappings + exit-code
  reference; two new rows in the manifest-vs-fork decision table
  for the 15-row output cap and FORMAT CSV constraints.
- **README examples reordered**: `conference_graph_mcp.yaml`
  promoted to the first example as the canonical zero-Python
  starter post-0.9.10. `legal_graph.py` reframed as the
  imperative-API alternative; `mcp_server.py` demoted to
  fork-only-when-manifest-can't.
- **`recipes.md` "Top-K Nodes by Centrality"** now shows the
  `CALL pagerank() YIELD node, score` Cypher form alongside the
  inherent `graph.pagerank(top_k=10)` Python form — manifest /
  MCP / agent contexts all reach KGLite through `cypher()`, so
  the Cypher form is the agent-friendly default.

## [0.9.10] — 2026-05-07

### Added

- **YAML manifest for `kglite-mcp-server`.** Drop a
  `<graph_basename>_mcp.yaml` next to your graph file (or pass
  `--mcp-config FILE`) and the bundled CLI auto-loads it at
  startup. Three tiers, all optional:
  - `source_root: ./data` (or `source_roots: [./a, ../b]`)
    auto-registers `read_source` / `grep` / `list_source` tools
    sandboxed to those directories. Backed by the
    [`mcp-methods`](https://github.com/kkollsga/mcp-methods)
    Rust-extension package — ripgrep crates, gitignore-aware,
    parallel walker, with internal grep so agents can search
    files too large to dump into context. Paths resolve relative
    to the yaml's directory; `../` is allowed.
  - `tools: cypher: |` blocks register parameterised Cypher as
    named MCP tools. JSON Schema `parameters:` drives the
    synthesised input schema, every `$param` reference is
    validated at server startup against the schema, and
    function signatures get built dynamically so FastMCP's
    introspection produces clean tool schemas on the wire.
  - `tools: python: ./tools.py` + `function: name` loads custom
    Python hooks. Two-signal trust gate: requires both
    `trust.allow_python_tools: true` in the yaml AND
    `--trust-tools` on the CLI. Either alone refuses to load.
- **`mcp-methods` and `PyYAML` added to the `[mcp]` extras** —
  `pip install "kglite[mcp]"` now pulls them automatically.
- **`--mcp-config FILE`** explicit-override flag and
  **`--trust-tools`** Python-hook authorisation flag added to
  `kglite-mcp-server`.
- **`name:` and `instructions:` manifest fields** override the
  default FastMCP server-info values when set.

### Docs

- MCP Servers guide rewritten around the manifest as the primary
  customisation path. Forking `examples/mcp_server.py` is now
  framed as the escape hatch for needs the manifest can't cover
  (custom CSV-export logic, FastMCP middleware, alternative
  transports). Includes a complete manifest example for a
  conference-catalog graph.
- `KnowledgeGraph.explain_mcp()` (the agent-facing XML quickstart)
  rewritten for the bundled CLI + manifest path. Previously
  recommended forking a server file and pointed agents at the wrong
  install / import.
- New `examples/conference_graph_mcp.yaml` — copy-paste-ready
  reference manifest demonstrating all three tiers (`source_root`,
  inline cypher, python hooks) with comments explaining each.

## [0.9.9] — 2026-05-07

### Added

- **`kglite-mcp-server` console script.** The MCP server that
  exposes any `.kgl` graph as a Cypher tool now ships as part
  of the package — `pip install "kglite[mcp]"` and run
  `kglite-mcp-server --graph my.kgl`. Same surface as
  `examples/mcp_server.py`, which is now a thin wrapper around
  the new `kglite.mcp_server.main` entry point and lives on as
  the fork-this template for adding custom tools.

### Changed

- `add_nodes` and `add_connections` now emit a `UserWarning`
  whenever the report flags **any** errors, not just when rows
  were skipped. Previously, follow-up loads with type mismatches
  set `has_errors=True` on the report but stayed silent; you had
  to inspect `last_report()` to notice. Silent partial successes
  were a recurring footgun.

### Docs

- New "Loading in passes" section in the Data Loading guide
  covering the second-`add_nodes` contract (static-then-
  timeseries, schema-then-enrichment), what carries over between
  calls, and a `conflict_handling` cheatsheet.
- New "Hierarchies" section disambiguating `set_parent_type`
  (type-level disclosure for `describe()`) from explicit
  `PARENT_OF`-style edges (instance-level tree structure that
  Cypher `*` walks). They look similar; they aren't.
- MCP Servers guide rewritten to lead with the bundled CLI:
  `pip install "kglite[mcp]"` → `kglite-mcp-server --graph X.kgl`.
  Claude Desktop / Claude Code configs use `"command":
  "kglite-mcp-server"` directly. Tutorial body is now framed as
  the customisation path for forks of `examples/mcp_server.py`.
- `add_nodes` / `add_connections` reference sections in the Data
  Loading guide reframed as parameter tables (no longer redundant
  with the walkthrough).

## [0.9.8] — 2026-05-07

### Fixed — `add_nodes` no longer clobbers the title alias on follow-up calls

A repeated `add_nodes(...)` on an existing node type without
`node_title_field` (the canonical pattern when layering
timeseries onto static rows, the example in the docstring)
silently rebound `title_field_aliases[node_type]` to the
`unique_id_field`. Later Cypher queries for `s.id` then resolved
to the stored title, returning the title string in place of the
id. The only visible signal was `title_alias="id"` in
`describe()` output.

The alias map is now written only when the caller explicitly
passes `node_title_field`. Surfaced by a real-world graph build
where the "static rows once, timeseries on top" pattern hit it.

### Added — `describe(sample_truncate=…)` to control title truncation in the XML

Sample values, sample node titles, and sample edge attributes
emitted by `describe()` get truncated at 40 chars by default to
keep prompts compact. Pass `describe(sample_truncate=None)` to
emit them in full when you want full titles in an LLM context
and have the budget for it; pass an integer for a custom
threshold. The knob only affects rendering — stored data is
always full-precision and accessible via Cypher.

### Docs

- New "End-to-end walkthrough" section at the top of the Data
  Loading guide — shape tables → `add_nodes` → `add_connections`
  → Cypher → save/load — so the README's "DataFrames in" pitch
  lands on a single connected story instead of scattered
  reference snippets.
- AI Agents guide now documents the `id_alias` / `title_alias`
  attributes on `<type>` elements and the new `sample_truncate`
  knob.

## [0.9.7] — 2026-05-04

## [0.9.7] — 2026-05-04

### Jupyter ergonomics — `wikidata.open()` is now process-cached

`wikidata.open(workdir)` previously did a fresh disk-graph load
(~350 MB in-memory state on the 124M-node truthy graph) on every
call, even when the same workdir had already been opened in the
same process. Repeating the call in a Jupyter notebook (the typical
"rerun-cell" workflow) accumulated RSS until the kernel ran out of
room and started swapping or hung after a dozen iterations.

`open()` now holds a process-local cache keyed by
`(canonical workdir path, entity_limit_millions)` →
`(KnowledgeGraph, disk_graph_meta.mtime)`. Cache hits return the
same `KnowledgeGraph` instance the prior call handed back. The
cache invalidates automatically when:

- the on-disk graph is rebuilt (mtime advances)
- `force_rebuild=True` is passed
- the user calls the new `wikidata.cache_clear()` (mirrors
  `functools.lru_cache`'s pattern; returns count of entries
  dropped)

Memory-mode opens skip the cache entirely — they're meant to be
reproducible rebuilds.

Verified: 3 × `wikidata.open(WORKDIR)` in one process → 426 MB
once, then flat. Same instance returned (`g1 is g2 is g3`).

### Examples — `examples/wikidata_disk.py` rewritten

Replaces the 259-line build-plus-bench harness with a 35-line
realistic walkthrough: download/load the dump via
`wikidata.datasets.wikidata.open()`, print graph size + a
"name+type lookup → awards" demo for Albert Einstein. Each step
shows its wall time so users see what each operation costs. Full
benchmark version preserved in `dev-documentation/`.

### Known issue (not fixed in 0.9.7)

`MATCH ({nid: $param})-[:T]->()` on the 124M-node Wikidata graph
runs ~12,000× slower than the literal-form
`MATCH ({nid: 'Q937'})-[:T]->()` (~65 seconds vs ~5 ms) and
allocates ~3 GB of RSS per call. The index-lookup planner pass
treats `Expression::Parameter` as a non-indexable predicate when
the property name is the global id alias. Workaround: inline the
literal value (Cypher injection-safe when the value came from the
graph itself) or fold into a single multi-MATCH query that anchors
once on a typed node pattern. Tracked for a future release.

585 cargo, 2345 pytest, 97/97 parity, lint clean.

## [0.9.6] — 2026-05-03

### Cypher correctness fix — `collect()[slice]` over `OPTIONAL MATCH` raised a spurious aggregate-context error

Cypher of the shape

```cypher
MATCH (n {id: $id})
OPTIONAL MATCH (n)-[:T]-(x)
WITH n, collect(DISTINCT x.title)[0..3] AS first_three
RETURN n.title, first_three
```

failed at runtime with `Aggregate function 'collect' cannot be used
outside of RETURN/WITH`, even though the `collect` call was clearly
inside a `WITH` projection. The same expression on a non-`OPTIONAL`
`MATCH` worked fine. Caused users to rewrite the query as
`WITH ... collect(...) AS xs RETURN xs[0..3]`, which is identical
semantically but unobvious as a workaround.

**Root cause.** `aggregates_only_count` in
`planner/fusion.rs` — the gate that decides whether the OPTIONAL-MATCH
count fusion can absorb a projection — recursed into arithmetic and
function-call nodes but fell through to `_ => true` on `ListSlice`,
`IndexAccess`, `ListComprehension`, and `Case`. So
`collect(x)[0..3]` (a `ListSlice` wrapping a `FunctionCall`) was
wrongly classified as "all aggregates inside are count-shaped" and
the count-fusion accepted it. The fused executor then ran
`evaluate_expression` per row on the substituted-but-still-
containing-`collect` projection, and the runtime correctly rejected
the per-row aggregate call.

**Fix.** `aggregates_only_count` now recurses into the same wrapper
expression variants that `ast::is_aggregate_expression` walks — slice,
index, list comprehension, case, expression-property-access,
map-literal. `collect()[…]`, `collect()[i]`, `collect()` inside a
`CASE`, and other "aggregate inside a wrapper" shapes all bail
fusion correctly and route through the materialised aggregate
evaluator.

The same fix incidentally closes a related class of broken queries:
`sum(x.prop)`, `min(...)`, `max(...)`, etc. wrapped by
`ListSlice`/`IndexAccess`/etc. over an `OPTIONAL MATCH` were also
silently broken pre-0.9.6 — never explicitly tested but caught by
the new corpus entries.

Differential corpus regressions (`tests/test_cypher_differential.py`):
`collect_slice_over_optional`, `collect_index_over_optional`,
`sum_over_optional`.

### Cypher perf fix — `LIMIT N` not pushed into grouping aggregator

Hub-anchored `OPTIONAL MATCH + collect/aggregate + LIMIT N` queries
on the 124M-node Wikidata graph were unnecessarily slow:

```cypher
MATCH (x)-[:P31]->(hub {nid: 'Q11424'})        -- film hub, 340k inbound
OPTIONAL MATCH (x)-[:P27]->(country)
RETURN x.title AS x, collect(DISTINCT country.title) AS countries
LIMIT 15
```

Materialised the full 340k MATCH expansion + 340k OPTIONAL P27
expansions + 309k group buckets, then truncated to 15 at the very
end. Cold: 64s. Warm: 547ms.

**Root cause.** The materialised aggregator drained every group
key before any downstream `LIMIT` clause looked at the rows.
There was no path for a literal `LIMIT N` to inform the grouping
loop that it could stop creating new groups after `N` distinct
keys.

**Fix.** New planner pass `push_limit_into_aggregate` (registered
in `PASSES` between `push_limit_into_match` and
`push_distinct_into_match`). When the projection clause has both
group keys and aggregates AND the next clause is a literal
`LIMIT N` (no intervening `ORDER BY`, no `DISTINCT`, no `HAVING`),
the pass stamps a `group_limit_hint` on the `ReturnClause` /
`WithClause`. The aggregator then uses a 2× safety margin during
the surrogate-key grouping pass (NodeIndex→Value collisions can
collapse groups during the resolve step) and truncates to the
exact `N` after resolve. Rows for already-collected keys
continue to feed their aggregates so `collect()` / `sum()`
complete correctly for the surviving groups.

ORDER BY between projection and LIMIT correctly disables the
optimisation — needed every group to find the top N. The
existing LIMIT clause stays in the plan as a hard safety cap.

**Verification on the 124M-node graph (warm steady state):**

| State | Latency | Profile shape |
|-------|---------|---------------|
| Pre-fix  | 547 ms | `Return rows_in=340688 rows_out=309004 → Limit 15` |
| Post-fix | 257 ms | `Return rows_in=340688 rows_out=15 → Limit 15` |

The remaining 257ms is genuine MATCH + OPTIONAL fanout work
(340k inbound `:P31` + 340k OPTIONAL `:P27` expansions). True
streaming through MATCH itself would need a larger refactor and
is left for a future release.

Differential corpus regressions: `limit_into_aggregate_collect`,
`limit_into_aggregate_count`, `limit_with_order_by_no_pushdown`.

### Investigated, not fixed — first-MATCH cold-mmap warmup spike

User reported ~700ms latency on the first `cypher` call after
process start against the same 124M-node graph; subsequent queries
sub-100ms. Reproduced locally at ~1010ms on the same graph; <5ms
on a 16M-node graph regardless of whether we'd touched it that
session. Not a code bug — the first MATCH page-faults the
`id_indices`, `node_slots`, and column-store mmap regions for the
queried type, and the cost scales with graph size + how cold the
OS page cache is.

Existing mitigation: `KGLITE_PREFETCH=1` env var triggers
`madvise(MADV_WILLNEED)` against hot regions at load time, paid
upfront instead of on the first user query. Recommended for MCP
servers that load multi-GB graphs at startup and want predictable
first-query latency.

585 cargo, 2345 pytest, 97/97 parity, lint clean.

## [0.9.5] — 2026-05-02

### Disk-mode regression fix — `register_connection_type` flipped the conn-type cache into a half-built state

On a disk graph that's been loaded from disk (`kglite.load(path)`),
the very first `add_connections` call after load broke MATCH queries
on every other edge type. `MATCH ()-[:OF_DISCOVERY]->()` returned 0
rows even though the edges were still on disk and unanchored
`MATCH ()-[r]->() RETURN type(r)` still listed them. Surfaced on
the Sodir prospect graph's load-then-enhance flow: the moment the
re-enhance pass added its first new edge type, every subsequent
FK-edge join (`OF_DISCOVERY`, `OF_FIELD`, `OF_PROSPECT`, `IN_PLAY`,
`HAS_DEPOSIT_PROSPECT`, `HAS_PLAY`) silently produced 0 rows. The
single-pass `build_sodir_graph.py --storage disk` flow was unaffected
because the cache was in the right state during initial build.

**Root cause.** `DirGraph.connection_types` is a `HashSet<InternedKey>`
that powers the O(1) fast path of `has_connection_type`. The set is
built from `connection_type_metadata.keys()` via
`build_connection_types_cache()` — but that function was only called
from `read_graph_v3` (the `.kgl` v3 loader), never from
`load_disk_dir` (the disk-graph loader). On a freshly-loaded disk
graph the set was therefore empty, and `has_connection_type` correctly
fell through to the metadata-fallback branch which returns true for
every edge type in the loaded metadata.

`register_connection_type(NEW)` then unconditionally inserted NEW
into the empty set. The next `has_connection_type(EXISTING)` call
hit the fast path: "set non-empty? consult cache" — which now only
contained NEW, so it returned false for every existing edge type.
The pattern matcher's early-exit ("skip iteration when the conn type
doesn't exist") fired for every typed MATCH on existing edges.

**Fix.** Two complementary patches:

1. `load_disk_dir` now calls `build_connection_types_cache()` after
   loading metadata, mirroring the v3 loader. Keeps the cache
   authoritative throughout the lifetime of any loaded disk graph.

2. `register_connection_type` lazy-builds the cache from
   `connection_type_metadata` when called against an empty set.
   Closes the same hole defensively for any future code path that
   could leave the cache empty before calling it.

**Verification.** New `bench/cross_mode_pipeline.py` harness runs an
enhance-style sequence (load → create_index → add_connections of a
new edge type → SET → re-read baseline queries at every step) on
legal + sodir × {memory, mapped, disk}. Pre-fix:

    legal disk:  step3_after_add_connections → cites_total: 0 (baseline 592305)
    legal disk:  step3_after_add_connections → section_of_total: 0 (baseline 23585)
    sodir disk:  step3_after_add_connections → of_discovery_join: 0 (baseline 107)
    sodir disk:  step3_after_add_connections → of_field_join: 0 (baseline 2280)
    sodir disk:  step3_after_add_connections → of_prospect_join: 0 (baseline 21857)

Post-fix all per-step counts match baseline across all three modes
on both graphs. Re-running the Sodir `enhance(g)` on a loaded disk
graph (which was the user-facing surface of this bug) now matches
the standalone-build counts: 97 discoveries enriched, 139 fields,
126 production profiles, 4345 prospects tagged, 48 plays calibrated,
2406 prospects calibrated.

Regression test:
`tests/test_disk_mutation_roundtrip.py::test_register_new_conn_type_preserves_existing_type_lookups`
loads a small disk graph with WORKS_AT edges, runs `add_connections`
of a new FRIENDS_WITH type, and asserts MATCH on WORKS_AT still
finds all 5 edges. Test fails with the fix reverted.

### Test harness — pipeline-shaped consistency

`bench/cross_mode_pipeline.py` (new). Closes the gap that let the
sequence-of-operations bug class slip through the earlier single-op
harnesses (`cross_mode_table.py`, `cross_mode_consistency.py`):
record per-step row counts during a multi-op pipeline, compare
each step to a fresh-load baseline, flag the exact step + query
where any cell drifts. Same code shape that caught the bug in this
release; held in `bench/` (gitignored) for ad-hoc use.

585 cargo, 2338 pytest, 97/97 parity, lint clean.

## [0.9.4] — 2026-05-02

### Disk- and mapped-mode regression fix — Cypher SET silently no-oped on mmap-backed ColumnStores

Cypher `SET` on nodes whose property storage was an mmap-backed
`ColumnStore` (the path `load_ntriples` always takes for mapped /
disk targets, regardless of input size) reported success at the
clause boundary — `MATCH (a) SET a.x = 1 RETURN count(a)` returned
the expected `count = 1` — but a follow-up `RETURN a.x` came back
`None`. The clause matched, the writer reported success, but the
write was invisible on read. Same pattern for `SET a.title = …`:
the title column held the old value forever.

**Root cause.** `ColumnStore::from_mmap_store` builds a store with
`schema = TypeSchema::new()` (empty), `columns = Vec::new()`,
`title_column = None`, and `mmap_store = Some(...)` carrying the
real data. Every read method (`get`, `get_title`, `str_prop_eq`,
`row_properties`) short-circuited to the mmap-backed read at the
top of the function — but `set` and `set_title` wrote into the
local `self.columns` / `self.title_column` fields the readers
bypassed. The fix to 0.9.2's disk-mode SET visibility regression
(per-clause `flush_pending_writes` + `sync_column_stores_from_disk`)
landed the writes in the right place; readers just couldn't see
them once the store was mmap-backed.

The bug only surfaced for `load_ntriples`-built graphs: `add_nodes`
followed by save+reload produces ColumnStores whose blobs reload as
in-memory `columns: Vec<TypedColumn>` with `mmap_store: None`, so
the read short-circuit never fires there. Sodir's
`from_blueprint` build → save → reload path passes the existing
SET-roundtrip parity tests for the same reason.

**Fix.** `get` / `get_title` / `str_prop_eq` / `row_properties` now
consult the in-memory overlay first and only fall through to the
mmap-backed read when no override exists for that (row, key).
`set_title` lazy-promotes the mmap-backed title column into a
Mixed in-memory column on first override, so the dense title column
isn't allocated up-front (avoiding the multi-million-row
materialisation that an eager promotion would force on Wikidata).

**Verification on the cross-mode consistency harness**
(`bench/cross_mode_consistency.py`, runs the same edit + read
queries against every (graph, mode) cell and SHA-256s the rows):

| graph | mode coverage | pre-fix `verify_edit` digest | post-fix |
|-------|---------------|------------------------------|----------|
| legal      | memory + mapped + disk | identical | identical ✓ |
| sodir      | memory + mapped + disk | identical | identical ✓ |
| wiki100m   | memory + mapped + disk | memory=`278f…9516`, mapped/disk=∅ | all `278f…9516` ✓ |
| wiki500m   | mapped + disk          | mapped/disk=∅ vs absent canonical | both `278f…9516` ✓ |

Read-only queries (simple / medium / complex) were already
identical pre-fix; this release closes the SET-visibility gap.

Regression test `test_cypher_set_visible_on_mmap_backed_columnstore`
in `tests/test_disk_mutation_roundtrip.py` builds a tiny inline
`.nt` with 5 Q-entities, loads it under mapped + disk, applies SET
to both a new property name AND `title`, and asserts the writes are
visible. Test fails with the fix reverted (confirmed by
`git stash`-ing `column_store.rs` only).

585 cargo, 2337 pytest, 97/97 parity, lint clean.

## [0.9.3] — 2026-05-02

### Disk-mode regression fix — parallel projection data race on `node_arena`

Disk-mode queries that reached `node_weight` materialization through
the Cypher executor's projection phase produced **non-deterministic
results across runs**, ranging from silently wrong row counts (Bug A
in the 0.9.2 disk regression report — ~13% of `NEAREST_AFEX_HUB`
edges and ~2% of `IN_AFEX_AREA` edges silently dropped on the Sodir
prospect graph) to use-after-free segfaults with `BUG: InternedKey N
not found in StringInterner` lines on stderr (Bug B in the same
report).

**Root cause.** `DiskGraph::node_arena` was an
`UnsafeCell<Vec<NodeData>>` and its module-level SAFETY block claimed
the single-threaded-query contract guaranteed exclusive access. In
reality the Cypher executor's projection phase
(`return_clause::project_row`) runs `evaluate_expression` under
`par_iter_mut` once `result_set.rows.len() >= RAYON_THRESHOLD` (256),
and any expression that reaches `node_weight` — spatial functions
(`centroid` / `contains` / `distance`), the spatial-fallback branch
of `resolve_property`, the `NodeRef`-in-projected branch — pushed
onto the unguarded `Vec` from sibling Rayon tasks. A `push` that
triggered realloc invalidated the `&NodeData` references already
returned to other tasks; downstream reads either (a) saw the wrong
row's properties (Bug A — sometimes the polygon parsed cleanly to a
near-but-different centroid and the row simply missed its hub) or
(b) followed a dangling pointer into the freed allocation (Bug B —
sometimes the dangling slot decoded to an `InternedKey` the interner
didn't know, surfacing the BUG line; with worse timing, SIGSEGV).

The non-determinism explains why the fresh build → save → reload →
read flow showed different counts than the in-process build: the
in-process flow used the warm in-memory column stores, while the
load-then-read flow re-materialized through `node_weight` under the
parallel projection.

**Fix.** `node_arena` is now `Mutex<Vec<Box<NodeData>>>`, mirroring
the long-standing pattern used by `edge_arena`. The Box gives stable
heap pointers that survive Vec growth; the Mutex serialises pushes.
The `&NodeData` references handed back are valid for the lifetime of
`&self` because the arena is only cleared via `clear_arenas`
(`&mut self`) or `reset_arenas` (called between top-level queries).

**Verification.** Fresh disk build of the Sodir prospect graph
(557k nodes) now produces edge counts that match the in-memory
build byte-for-byte:

| Edge type             | default | disk (pre) | disk (post) |
|-----------------------|---------|------------|-------------|
| `IN_AFEX_AREA`        |    886  |       866  |        886  |
| `NEAREST_AFEX_HUB`    |  5,881  |     5,134  |      5,881  |
| `IN_BLOCK`            |  7,983  |     7,983  |      7,983  |
| `IN_STRUCTURAL_ELEMENT`|  6,763  |     6,763  |      6,763  |

The `load → enhance → save` workflow that was reported as
segfaulting on disk now completes cleanly with no `BUG: InternedKey`
lines on stderr.

Two regression tests cover the race surface:
`test_disk_parallel_projection_node_weight_is_race_free` (8 repeated
runs of `centroid` projection on a 600-node disk graph must report
identical counts) and
`test_disk_parallel_projection_no_interner_corruption` (asserts no
`BUG: InternedKey N not found` lines appear on stderr across
repeated parallel projections).

585 cargo, 2333 pytest, 97/97 parity, lint clean.

## [0.9.2] — 2026-05-02

### Disk-mode regression fix — property visibility after blueprint build

Disk-mode `from_blueprint(...)` left freshly-built graphs in a
state where every Cypher property read returned NULL.
`MATCH (p) RETURN count(p)` worked (slot enumeration), but
`RETURN p.x` for any property — even `id` and `title` — returned
None. The Sodir prospect-graph build on disk surfaced the failure
shape: every Stage 2.x derived edge was 0, every Stage 3 SET
cascade collapsed because the read-back of the previous stage's
writes returned 0 rows. Single-process build → enhance → save was
unusable on disk; default and mapped modes were unaffected.

Two related sync gaps between `DirGraph.column_stores` and
`DiskGraph.column_stores`:

- **batch.rs deferred-columnar pass after creates.** Disk's
  `add_nodes` populated `graph.column_stores` (DirGraph-side) and
  updated each slot's `row_id`, but never mirrored to
  `disk_graph.column_stores` (where disk reads through
  `node_weight` / `get_node_id` / `get_node_title`). The existing
  `sync_disk_column_stores()` only fired when the chunk had
  UPDATEs, never on creates-only. Fix: after the deferred-columnar
  loop, call `graph.sync_disk_column_stores()` when disk-backed.

- **write.rs after Cypher mutation flushes.** Each SET / REMOVE /
  MERGE clause calls `flush_pending_writes` (0.8.41) which drains
  `node_mut_cache` into `disk_graph.column_stores`. But
  `graph.column_stores` stayed stale. A subsequent `add_nodes`
  (which now calls `sync_disk_column_stores` per the first fix)
  would clobber disk's post-flush state with the pre-flush
  DirGraph snapshot — silently losing the SET's effects on the
  multi-stage `SET → add_nodes → read` pipeline. Fix: after every
  `flush_pending_writes` (per-clause + end-of-query), also call
  `graph.sync_column_stores_from_disk()` to mirror the post-flush
  disk state back to DirGraph.

**Verification on the Sodir prospect-graph build (557k nodes, 47
edge types):**

- All Stage 2.x derived edges populated (was all 0): IN_BLOCK
  7,983, IN_STRUCTURAL_ELEMENT 6,763, IN_AFEX_AREA 779,
  NEAREST_AFEX_HUB 5,560.
- All Stage 7 derived edges populated (was all 0):
  HC_IN_FORMATION 850, ENCLOSES 10,588, PLAY_HAS_FORMATION 248,
  DRILLED_IN_PLAY 2,627.
- Stage 3.6 value_basis distribution matches the in-memory
  baseline byte-for-byte: estimate=2754, realized=546, dry=1437,
  deflagged=1577, unscored=460. Pre-fix every prospect collapsed
  into `unscored`.

All three storage modes — default, mapped, disk — now produce
identical results on a 16-case integration smoke test that exercises
every 0.9.x gate item plus the multi-stage SET / `add_connections`
/ save+reload pipeline.

585 cargo, 2333 pytest, 97/97 parity, lint clean.

## [0.9.1] — 2026-05-02

### Breaking change in 0.9.0 (Rust crate API) — late-breaking notice

`MemoryGraph` and `MappedGraph` no longer implement
`std::ops::DerefMut` to their inner `StableDiGraph`. Code that
mutated through auto-deref now fails to compile and must use one
of:

- the explicit accessor: `g.inner_mut().add_node(data)`, OR
- trait dispatch: `use kglite::graph::storage::GraphWrite;
  g.add_node(data)`.

**Why**: auto-deref-via-DerefMut shadowed `GraphWrite::add_node`
(and peer mutation methods), causing calls to bypass
`MappedGraph::invalidate_property_index()` that the trait impl
runs first. Removing `DerefMut` converts a silent stale-index bug
into a compile-time error. Read-only `Deref` is retained — read
calls (`g.node_count()`, etc.) are unchanged.

**Python users**: no impact. The PyO3 boundary is unchanged.

This was already in 0.9.0 (commit `bcb0bb7`, "chore(storage): drop
DerefMut on Memory/MappedGraph") under "Hygiene & test coverage" —
this entry simply re-flags it for downstream Rust embedders who
might miss it under that subhead.

### Blueprint diagnostics

- **`from_blueprint(verbose=True)` reports actual graph edge counts.**
  Pre-fix, the verbose log printed an accumulated input-row count
  from the blueprint pipeline; with default Update conflict
  handling, repeated `(src, tgt)` pairs across multiple section
  ingests over-counted vs `MATCH ()-[r]->() RETURN count(r)`. Easy
  to mistake for a save regression. Now queries
  `graph.get_edge_type_counts()` (the post-build truth) and reports
  those numbers; when input and graph counts diverge, the line is
  annotated `[T]: N edges (M input rows, K deduped)` so users see
  both. Backward-compatible for zero-duplicate datasets.
- **Warning capture documented.** Blueprint warnings (target node
  not found, null FK, etc.) hit stderr by default. Python's
  standard `logging.captureWarnings(True)` routes the Rust-emitted
  UserWarnings into the `py.warnings` logger where any standard
  handler (file, rotating, stream) can catch them — no `2>&1`
  shell redirect needed. The `from_blueprint` docstring now
  documents the pattern with a copy-paste-ready file-capture
  example. Pinned by `test_logging_capture_warnings_pipeline`.

### Documentation

- **CYPHER.md** gained a "Duration semantics" subsection explaining
  the calendar-vs-clock component split (months/years stay
  separate from days/hours/minutes/seconds), with worked examples
  for `duration.between()` and `DateTime ± Duration`. Postgres
  `interval` users will want this. Pinned by an anchor test
  (`test_duration_between_cypher_md_example`).

## [0.9.0] — 2026-05-02

### Cypher dialect — gate items

- **§5 Integer division (Neo4j-standard).** `1967 / 10` now returns
  `196` (truncated `Int64`), matching openCypher / Neo4j. Float
  promotion only when at least one operand is a float. Negatives
  truncate toward zero (`-7 / 2 → -3`). Modulo behavior preserved.
- **§2 NULLS FIRST / NULLS LAST in ORDER BY.** New
  `ast::NullsPlacement` + parser support. Default placement is
  Neo4j 5+: NULLS LAST for ASC, NULLS FIRST for DESC. Plumbed
  through both the in-memory sort path and the `heap_top_k`
  streaming operator.
- **§3 Stable date function set + Cluster 2 proper Value::Duration.**
  Datetime field accessors `.year/.month/.day/.dayOfWeek/.dayOfYear/.epochSeconds`
  on `Value::DateTime`. New `Value::Duration { months: i32, days: i32,
  seconds: i64 }` variant — calendar units (months/years) and clock
  units (days/hours/minutes/seconds) stay separate, so
  `duration({months: 1, days: 5}).months` returns 1 (not 35
  collapsed to days). `duration()` constructor, `duration.between()`,
  `DateTime ± Duration`, `Duration ± Duration` arithmetic. Sub-day
  precision wired in `seconds`; `Value::DateTime` is still
  `NaiveDate`, so DateTime + Duration discards the seconds
  component for now (Cluster 1 deferred). Duration variant is the
  LAST enum variant — old `.kgl` files load unchanged.
- **§4 Polygon-vs-polygon `contains()` in WHERE.** The fast-path
  spatial filter for `MATCH (a), (b) WHERE contains(a, b)` now
  handles geometry-vs-geometry when neither side has a `Location`
  point. Pre-fix the path silently returned `false` for every
  outer-contains-inner match in polygon-only graphs. Bundled
  MULTIPOLYGON dedupe — single boolean answer per `(a, b)` pair
  regardless of how many components match.
- **§1 Better Cypher error messages.** Every parse error now
  carries `(line N col M)` plus a single-line source excerpt with
  a `^` caret. Position is byte-precise — the tokenizer attaches
  char offsets to every token; parser threads them through;
  `format_parse_error` walks `input.chars()` to compute (line,
  col) on the error path. New `intent_level_rewrite` hook in
  `parser/mod.rs` for "feature not yet implemented" detection
  (currently empty — all named candidates parse successfully).
- **§6 size() over pattern expressions.** `size((:A)-[:R]->(:B))`,
  `size((a)-[:R]->(:B))` (per-row binding), `size((:A)-[:R]->(:B)) >= 2`
  in WHERE. Wraps the existing 0.8.16 count-subquery code path.
  Refactored `parse_exists_patterns` into
  `parse_pattern_subquery_patterns` with a caller-supplied
  delimiter (RBrace for EXISTS/count, RParen for size).

### Hygiene & test coverage

- **Cluster 6: dropped `DerefMut` on `MemoryGraph` / `MappedGraph`.**
  Auto-deref-via-DerefMut shadowed `GraphWrite` trait methods, so
  e.g. `g.add_node(data)` on `&mut MappedGraph` reached petgraph's
  inherent method directly, bypassing
  `MappedGraph::invalidate_property_index()` that the trait impl
  runs first. Removing DerefMut forces explicit `.inner_mut()` or
  trait dispatch — compile-time catch instead of silently-stale-
  index runtime bug. Read-only `Deref` retained.
- **Cluster 6: `node_weight_mut` staging contract documented.**
  Trait method on `GraphWrite` now carries explicit doc that disk
  buffers writes in `node_mut_cache` and callers must call
  `flush_pending_writes()` before any subsequent `&self` read.
  Debug-only assertion in `DiskGraph::node_weight` warns when a
  staged write is shadowed by a read (catches future code paths
  that forget the flush).
- **Cluster 7: deep-traversal path-materialization coverage.** New
  `test_long_chain_traversal_path_materialization_100_hops`
  exercises actual path enumeration across memory + mapped + disk
  via `RETURN b.id` instead of the planner-short-circuited
  `RETURN count(b)` shape that the prior 1,000-hop test used.
- **Cluster 7: datetime accessor golden round-trip.** Two new
  queries in the golden corpus pin `joined_at.year/.month/.day`
  extraction against the social-graph fixture so §3 accessor
  behaviour can't drift unnoticed.

### Internal — pre-existing parity audits cleared

- `.kgl` v3 fixture digest updated (no format change —
  `CURRENT_FORMAT_VERSION` still 3). Fixture-graph hash drifted
  across the 0.8.x → 0.9.0 line as save-path / interner
  refinements landed; backward compatibility verified by loading
  pre-0.9.0 `.kgl` files cleanly.
- `GraphBackend` enum-match audit whitelist refreshed for the
  pre-existing leaks (`column_builder.rs`, `match_clause.rs`,
  `blueprint.rs`, `indexes.rs`).
- Binary-size baseline reset to 0.9.0 (~22.4 MB) with a +10%
  gate. Phase 4 baseline (6.67 MB) was 3.4× off the current build
  — accumulated growth from multi-mode storage + spatial +
  timeseries + code-tree + MCP + Cypher dialect work.
- `god_file_gate`: `match_clause.rs` (2,679 lines) and `fusion.rs`
  (2,923 lines) documented in `GOD_FILE_EXCEPTIONS` with concrete
  0.9.x split plans.
- `mod_rs_purity`: caps bumped on `executor/`, `parser/`,
  `planner/` `mod.rs` files to match the dispatch surface they
  carry; 0.9.x cleanup intent documented.

## [0.8.41] — 2026-05-02

### Cypher executor — Bug 8 followup

- **Fix: SET on an in-memory graph dropped new properties on save.**
  The 0.8.39 master-path fix that routed Columnar SET writes through
  `graph.column_stores` to dodge the per-node Arc-clone storm computed
  the property's `InternedKey` via `from_str()` (just hashing) without
  registering the source string in `graph.interner`. As a result,
  Cypher SET that introduced a *new* property name on an in-memory or
  mapped graph survived in-memory queries but vanished on save+reload,
  printing `BUG: InternedKey N not found in StringInterner` to stderr
  and silently corrupting the saved file. Now registers via
  `graph.interner.get_or_intern(property)` before borrowing
  `column_stores`. Disk mode is unaffected (gated path, separate
  pre-existing bug for that backend).
- **Defense in depth: debug-only invariant in `write_graph_v3`.** Walks
  every `column_store`'s schema before serialization and panics with a
  clear, actionable message if any `InternedKey` doesn't resolve in
  `graph.interner`. Catches the entire class of bug ("writer
  synthesizes an InternedKey without first registering the source
  string") at write time, not at load time on the user's machine. Zero
  release-build cost.
- **Regression**: `tests/test_disk_mutation_roundtrip.py::test_cypher_set_new_property_persists_through_save_reload`
  parameterised over `memory` + `mapped` + `disk` (all three modes
  now covered after the disk-side fix below).

### Cypher executor — disk SET visibility

- **Fix: Cypher SET on a disk-backed graph appeared to no-op on
  in-session reads** until the next `&mut self` op (e.g. `save()`)
  flushed the staged writes. Disk's `node_weight_mut` stages writes in
  `node_mut_cache` to dodge the `Arc<ColumnStore>` share-clone storm
  per row; `node_weight` (the read path) reads `column_stores`
  directly and ignored the cache. Save+reload happened to recover the
  data because `clear_arenas` runs as part of save, but every
  in-session read between SET and save returned the pre-SET value —
  silently corrupting any analysis that read SET-staged columns
  back. Affects both new-property SET (`SET n.value_score = …`) and
  existing-property SET (`SET n.age = n.age + 100`). Same query
  shape: `MATCH … SET … RETURN n.prop` returned NULLs for the just-
  written column.
- **Fix shape**: new `GraphWrite::flush_pending_writes(&mut self)`
  trait method, default no-op, overridden on the disk backend to
  call its existing `clear_arenas` (which already does the
  clone-apply-replace flush of `node_mut_cache` /
  `edge_mut_cache` into `column_stores` / `edge_properties`).
  `execute_mutable` calls it after every write clause (SET / REMOVE /
  MERGE) and once at end-of-query so any trailing RETURN's property
  projection — and any next-query read — sees the writes. Memory
  and mapped backends are unaffected (their `node_weight_mut`
  mutates `StableDiGraph` in place, so reads see writes
  immediately).
- **Doesn't affect Sodir** — the prospect graph runs in-memory — but
  unblocks the disk-mode "SET as cached column" pattern for any
  future workstream that uses storage="disk".

## [0.8.40] — 2026-05-01

### Cypher planner

- **Spatial-join fusion: multi-MATCH + `centroid()` probe.** Extends
  `fuse_spatial_join` to recognise the
  `MATCH (a:T1) MATCH (b:T2) WHERE contains(b, centroid(a))` shape (or
  the inverse `contains(a, centroid(b))` — the call decides which MATCH
  is container vs. probe, not pattern position). Previously only the
  single-MATCH cartesian form fired; the multi-MATCH form fell back to
  cross-product + post-filter. Sodir's `IN_STRUCTURAL_ELEMENT` enrichment
  (Prospect → StructuralElement via centroid) drops from ~2 s to ~0.6 s
  on a 6,775-prospect graph; any project doing point-in-polygon
  enrichment via `centroid()` benefits.
- New `Clause::SpatialJoin::probe_kind: SpatialProbeKind` carries
  whether the probe-side point comes from the spatial-config `location`
  (single-MATCH cartesian) or the centroid of the probe's geometry
  (multi-MATCH centroid). The spatial-join executor honours the kind
  when sourcing the per-probe point.
- An optional pre-WHERE between the two MATCHes (e.g.
  `MATCH (p:Prospect) WHERE p.wkt_geometry IS NOT NULL MATCH (s:StructuralElement) ...`)
  is folded into the SpatialJoin's residual predicate so per-pattern
  filters still apply after fusion.
- Regression coverage: `tests/test_cypher_spatial.py::TestSpatialJoin`
  picks up `test_multi_match_with_centroid_probe` (correctness vs. the
  brute-force two-pattern path) and `test_multi_match_centroid_fires_fusion`
  (EXPLAIN must show `SpatialJoin`).

## [0.8.39] — 2026-05-01

### Cypher executor

- **Fixed scalar projection from spatial-function results
  (`centroid(n).latitude` and `WITH centroid(n) AS c RETURN
  c.latitude`).** Previously returned the entire `{latitude,
  longitude}` dict — or `Null` on in-memory graphs — instead of the
  float. Property access on `Value::Point` now extracts the named
  field via a new `point_field()` helper, applied in both
  `Expression::ExprPropertyAccess` and the `resolve_property`
  projected fallback. Accepted aliases: `latitude`/`lat`/`y` and
  `longitude`/`lon`/`lng`/`long`/`x`. The canonical
  `point(centroid(n).lat, centroid(n).lon)` idiom now composes in a
  single Cypher query.
- **Fixed spatial-predicate failure on partial-coverage typed sets.**
  When a typed node set has spatial config but only a fraction of
  rows have geometry data populated (real-world example: 312 of 469
  AfexAreas in the Sodir graph have `wkt_geometry IS NULL`),
  `WHERE contains(a, point(lat, lon))` errored on the missing rows
  instead of treating them as predicate-false. `contains()` now
  NULL-propagates: row-level missing geometry returns
  `Boolean(false)` so the predicate filters the row out cleanly.
  Same NULL-propagation for `centroid()` / `area()` / `perimeter()`
  (return `Value::Null`). `intersects()` retains the loud error for
  the type-level "no spatial config anywhere" case (preserves
  existing diagnostic test coverage).
- **Fixed superlinear `SET` cost on typed nodes with shared
  columnar storage; OOM at ~1k rows on the Sodir Prospect set.**
  Per-node `Arc::make_mut(store).set(...)` cloned the entire shared
  `ColumnStore` on every write — for 6,775 Prospect nodes sharing
  one store, refcount=N+1 meant a full clone per row, giving
  O(N²) work and ~11 GB transient allocations. SET now routes
  Columnar writes through `graph.column_stores[type]` once per
  batch, then refreshes per-node `Arc<ColumnStore>` handles in a
  single end-of-statement sweep. Verified on the real Sodir graph:
  100 → 4 ms (was 0.81 s, 200×), 500 → 4.4 ms (was 13.1 s,
  3000×), 1000 / 2000 → 3 ms (were OOM). Linear scaling restored.
  Disk-mode graphs use a separate write path and are gated out
  via `graph.graph.is_disk()`.

### Fluent API

- **`create_connections()` now emits a `UserWarning` when called on
  a chained graph view.** The fluent `g.select(...).traverse(...)
  .create_connections(...)` pattern returns a NEW `KnowledgeGraph`
  whose mutations live on a temporary clone (Arc COW); discarding
  the return loses the writes. The warning fires when
  `Arc::strong_count(self.inner) > 1` and points to the two
  workarounds: capture the return (`g = g.select(...)
  .create_connections(...)`) or use the equivalent
  `add_connections(data=cypher_result, ...)`. Docstring also
  updated. Proper structural fix (shared interior mutability)
  deferred — would touch ~hundreds of read-path call sites.

## [0.8.38] — 2026-05-01

### code_tree

- **Re-add `_build` to the ignored-dirs list.** Caught while
  perf-testing 0.8.37 against the kglite repo itself: Sphinx
  `docs/_build/_static/*.js` was getting indexed (26 JS files), even
  though the build artifacts aren't user source. The leading-
  underscore convention is a strong signal of "tool-generated build
  output" (Sphinx, mkdocs, mdBook), distinct from the more ambiguous
  `build` / `dist` / `out` (which stay off the list because
  `dist/bundle.js` may be the user's webpack output they want
  flagged-as-too-large). Verified against `code_tree.build()` on
  the kglite repo: 337 files → 310 files, matching the 0.8.34
  baseline exactly. Build time 0.21s.

## [0.8.37] — 2026-05-01

### code_tree

- **Fixed `code_tree.build()` over-indexing on projects with nested
  `.venv` / `node_modules` / `target` / etc.** Reported by an MCP
  consumer whose codebase graph ballooned 7,620 → 70,605 nodes after
  upgrading to 0.8.36. Root cause: the mixed-language safety net
  added in 0.8.36 walked first-level subdirs recursively to detect
  undeclared languages — but the walk filter for ignored directory
  names (`.venv`, `node_modules`, `target`, `__pycache__`,
  `site-packages`, `venv`, `env`, plus all `.dot` dirs) only applied
  at the top level. A single C-extension source inside a nested
  `.venv` (e.g. numpy in a subprojects venv) attracted the parent
  dir as a supplemental source root and then `parse_directory`
  walked the full venv. The filter is now applied at every depth in
  both the safety-net detection walk and the actual parser walk.
- **Conservative trim of the ignored-names list.** `build`, `dist`,
  `out`, `_build` removed — those names are tooling-dependent (e.g.
  `dist/bundle.js` is sometimes the user's webpack output they want
  indexed-and-flagged as too-large rather than excluded outright).
  Use `max_loc_per_file` to handle oversized build artifacts.

## [0.8.36] — 2026-05-01

### Cypher planner

- **Fixed `mark_fast_var_length_paths` per-target row drop** — closes
  the lone open xfail in the differential harness. The pass set
  `needs_path_info=false` on any unnamed variable-length edge, which
  triggered a target-node-deduping BFS that returned fewer rows than
  Cypher's per-path semantic (e.g., 2 rows where Neo4j returns 3).
  The fix gates the pass on `downstream_is_dedup_safe` — the next
  RETURN/WITH must be `DISTINCT` or its projections must be entirely
  dedup-safe aggregates (`min/max/count(DISTINCT)/collect(DISTINCT)`).
  Plain `RETURN q.name` over var-length now uses the slow per-path
  BFS (correct); users who want the fast path opt in via `RETURN
  DISTINCT q.name` or `count(DISTINCT q)`.
- **Differential harness corpus extended to ~95 query shapes** plus 9
  mutations + 26 per-pass bisection tests. Probed three additional
  rounds (CALL/list-comp/path-ops/multi-WITH/HAVING/coalesce/CASE-in-
  agg/expr-filter/etc.) and surfaced no further divergences after
  fixing the var-length pass. The corpus also now includes
  `var_length_no_var_per_path`, `var_length_no_var_distinct`, and
  `var_length_no_var_count_distinct` as permanent regression tests
  for the fix.
- **Performance cleanups in the new code paths.** `optimize()` now
  returns a process-lifetime empty `HashSet<String>` via `OnceLock`
  instead of allocating a fresh `HashSet::new()` on every call; the
  PyAPI's `cypher()` short-circuits the disabled-passes set
  construction when both `disable_optimizer=False` and
  `disabled_passes=None` (the default), bypassing the validation
  loop entirely on the hot path.
- **Third debug-mode IR invariant**: literal `LIMIT` and `SKIP`
  values must be non-negative. Catches passes that synthesize a
  literal limit hint (e.g. fusion top-K) and forget to clamp at
  zero. Zero release cost (`#[cfg(debug_assertions)]`-gated).
- **Makefile bench targets force `maturin develop --release`.**
  Saved baselines are release-built; running `make bench-compare`
  against a dev build previously showed false ~15× regressions
  across every benchmark. New baseline `0007_post_robustness_pass`
  saved with all the planner refactor + bug fixes in place.
- **Optimizer is now a registry of named passes (`kglite.cypher_pass_names()`).**
  The 25-pass orchestrator at `src/graph/languages/cypher/planner/mod.rs`
  has been refactored from a 40-line inline body into a single
  `const PASSES: &[(&str, PassFn)]` source of truth. Each pass has a
  stable name and a doc-comment with precondition / pattern / rewrite
  / why-bail. Adding a new pass is now: write the impl, write a
  one-line wrapper, register it in `PASSES`, add a corpus entry — no
  hidden ordering dependencies.
- **New `cypher(disable_optimizer=True, disabled_passes=[...])` kwargs.**
  Diagnostic / testing knob: skip every optimizer pass, or skip a
  specific subset by name. Validated against the registry — typos
  raise `ValueError`. Used by the new differential test harness and
  the bisection script.
- **Fixed `push_limit_into_match` multi-pattern row drop.** A query
  with a single MATCH containing multiple comma-separated patterns
  plus WHERE plus LIMIT (e.g. self-joins:
  `MATCH (p)-[:T]->(q), (p)-[:T]->(r) WHERE q <> r RETURN ... LIMIT 5`)
  silently dropped rows. The 0.8.27 fix narrowed the pushdown to
  single-MATCH but didn't check single-pattern; the pattern executor's
  `max_matches` hint applied per-pattern and the cartesian
  cross-product fell short of the requested LIMIT. The pass now also
  bails when the MATCH has more than one pattern.
- **Fixed `fuse_node_scan_top_k` empty-result on alias-sorted top-K.**
  Queries of the form `MATCH (p:T) RETURN <expr> AS h ORDER BY h LIMIT k`
  silently produced zero rows when the ORDER BY referenced a RETURN
  alias — the fused executor's sort-key evaluator only knows graph
  variables, not RETURN-alias bindings. The pass now bails when the
  sort expression references any RETURN alias, falling back to the
  materializing path which handles aliases correctly.
- **Fixed `desugar_multi_match_return_aggregate` over-grouping bug**
  surfaced by the new differential harness on first run.
  `MATCH (p:Person) MATCH (c:Company) RETURN p.city, count(c)` was
  producing 20 rows of `(city, n=5)` instead of 4 rows of
  `(city, n=25)`: the rewrite introduced a `WITH p, count(c)` (group
  by source variable) when the user's RETURN was grouping by
  `p.city`. The rewrite now generates `WITH p.city AS <internal>,
  count(c) AS n` so GROUP BY matches Cypher's standard semantic
  (the set of non-aggregate RETURN expressions).
- **New differential test harness `tests/test_cypher_differential.py`.**
  Every query in a curated corpus runs twice (optimized vs.
  optimizer-off) and asserts identical rows. Includes 9 mutation
  tests that compare the cypher result *and* the post-mutation graph
  state (node + edge counts) across the two modes. Surfaced 4
  divergences across two probing rounds, all now fixed and tracked as
  permanent regression tests:
    - `desugar_multi_match_return_aggregate` over-grouping
      (fixed, see above)
    - `push_limit_into_match` multi-pattern row drop (fixed)
    - `fuse_node_scan_top_k` empty-result on alias-sorted top-K (fixed)
    - `mark_fast_var_length_paths` per-target vs. per-path semantics
      (lone `KNOWN_DIVERGENT` entry; pending design call).
- **New `scripts/cypher_pass_bisect.py`.** Given a query that diverges,
  runs each pass disabled in isolation and reports which pass's
  absence resolves the divergence. Works against `.kgl` files or
  `tests/conftest.py` fixtures.
- **Debug-mode IR invariant checks** run after every pass in debug
  builds. Catches passes that produce empty MATCH patterns, empty
  RETURN/WITH item lists, or splice clauses after a terminal RETURN.
  Zero cost in release.

### code_tree

- **Manifest discovery: declared-package strategies and mixed-language
  safety net.** `read_pyproject` was previously the only Python source-root
  finder, and it only matched the `<name>/__init__.py` / `src/<name>/__init__.py`
  conventions. Several common configurations silently parsed only a slice
  of the repo:
  - **Workspace path collisions:** Cargo workspaces with two crates each
    containing `src/lib.rs` collapsed to a single `File` node because every
    parser stripped `rel_path` against its per-root walk directory. Paths
    are now project-root-relative (any two source roots that share a
    same-named file at matching depth survive as distinct nodes).
  - **Explicit `[tool.poetry].packages` declarations** with a custom `from`
    directory (e.g. `from = "lib"`) are now respected. Previously these
    only worked by accident when the project also lacked a conventionally-
    placed package — adding a stub `<name>/__init__.py` would silently
    suppress the lib/ packages.
  - **`[tool.setuptools].packages = [...]`** explicit lists, including
    dotted names resolved against `[tool.setuptools].package_dir`.
  - **`[tool.setuptools.packages.find].where = [...]`** is honoured —
    each `where` directory becomes a source root.
  - **`[tool.hatch.build.targets.wheel].packages = [...]`** is honoured.
  - **`[tool.poetry].name`** is now a valid name source. Pure-poetry
    pyprojects without a `[project]` table previously left `name` as the
    parent directory, breaking name-keyed package discovery.
  - **Mixed-language safety net:** when a pyproject finds a Python package
    next to first-level directories that contain code in a language NOT
    declared by the manifest (e.g. tooling pyproject + huge `src/*.c`),
    those directories are auto-supplemented as source roots labelled
    `auto:<dirname>`. The "undeclared language" gate keeps it surgical:
    sibling `.py` directories are not pulled in for pure-Python repos.

  The manifest module was also refactored into a list of small named
  strategy fns (one per declaration shape) instead of one 150-line
  function — adding a new build backend is now one fn, not a new branch.

## [0.8.35] — 2026-05-01

### Ingestion

- **`add_nodes(nullable_int_downcast=True)` recovers integer columns
  that pandas auto-promoted to float64.** Pandas turns nullable int
  columns into `float64` whenever nulls are present, which surfaces in
  queries as `"2.0"` instead of `2`. The new opt-in flag scans Float64
  columns post-ingestion: when every non-null value is integer-valued
  and within `i64` range, the column is downcast to `Int64`. Default
  `False` so existing callers see no change.

### code_tree

- **`code_tree.build(max_loc_per_file=N)` skips oversized files.**
  Files whose newline count exceeds `N` are recorded as `File`
  nodes with `skip_reason="too_large"` but never sent to the parser.
  Default `None` preserves existing behavior. Targets autogenerated
  multi-thousand-LOC files (dotnet/runtime's JIT regression tests at
  85k+ LOC each) that dominate parse time without contributing
  structural information. Also threaded through `repo_tree(...)`.
- **Module nodes with purely-numeric names are no longer synthesized.**
  Repos with numeric directory components (dotnet/runtime's
  `tests/JIT/Regression/Runtime_<bug-id>/...` test layout, in
  particular) used to produce thousands of `Module {title="125042"}`
  nodes when the parser fell back to file-path-derived module names.
  `build_modules` now skips path segments that are pure ASCII digits
  while keeping legitimate alphanumeric ancestors and descendants.

### Cypher engine

- **Map-typed list-comprehensions now access fields correctly.**
  `[x IN collect({h: a.title, k: km}) WHERE x.k = min_km | x.h]` used
  to silently drop every row because `Value` has no Map variant — the
  collected items round-tripped through a JSON-encoded `Value::String`,
  and `x.k` returned the entire string, never matching the aggregated
  `min_km`. Property access on a map-shaped projected string now parses
  the map on demand and returns the field. `parse_value_token` and
  `extract_map_field` were factored out of `parse_list_value`.
- **Spatial functions infer config from conventional property names.**
  `intersects()` / `contains()` / `centroid()` / `distance()` etc. now
  accept nodes that store WKT under `wkt_geometry` / `geometry` / `geom`
  / `wkt`, or lat/lon under `latitude`+`longitude` / `lat`+`lon`, even
  when no `SpatialConfig` was registered at ingestion. The fallback
  inference is per-query and never mutates `graph.spatial_configs`;
  explicit configs always win. Also: clearer error message when a
  spatial argument truly can't be resolved.
- **Cross-MATCH equality joins on non-id properties now run in O(N+M).**
  When a subsequent MATCH joins on a non-canonical property (e.g.
  `WITH a.key AS k MATCH (b) WHERE b.key = k`), the executor builds a
  query-local hash index over the target type once and probes it per
  outer row, instead of running the full pattern matcher per row. Only
  fires when the outer row count is at least 64 and no persistent
  property index already covers the type+property; below that, the
  existing path runs unchanged. Internal API: new
  `executor::transient_index::TransientEqIndex`. A 5,000×5,000 join
  that previously degraded to ~25M property reads now finishes in tens
  of milliseconds.

## [0.8.34] — 2026-05-01

### Code-tree graph quality

- **`File DEFINES Function` now includes methods.** The previous
  `if !is_method` filter dropped every C# method (the language has no
  top-level functions), so an entire codebase of methods looked
  edge-less when joined through File. Class HAS_METHOD edges still
  carry the logical hierarchy.
- **Kind-aware target resolution for IMPLEMENTS / EXTENDS.** When a
  bare base-type name matches multiple namespaces, `implements` now
  prefers Interface candidates and `extends` prefers class-like
  candidates. On dotnet/runtime the `Class -[IMPLEMENTS]-> Class`
  noise (mis-typed because of name collisions) dropped from 1,869 to
  90 rows, while the correct `Class -[IMPLEMENTS]-> Interface` rows
  rose from 447 to 7,696.
- **Auto-reroute `extends → implements` when the target is an
  Interface.** Fixes the C# parser's "first base is always extends"
  assumption for `class Foo : IDisposable` (no base class).
- **`using`-directive scope as a CALLS resolution tier.** Calls like
  `Assert.True` now pin to the `Assert` class actually imported by
  the caller's file. On dotnet/runtime, `Xunit.Assert.True` collapses
  from four collision-cloned entries (~11 k each, false equals) to a
  single 11 k entry; `IDisposable` implementer count goes from 0 to
  236, `IEnumerable<T>` from 0 to 305, `IEquatable` from 0 to 469.
- **C# `get_base_types` captures every secondary base.** The hardcoded
  list of accepted node kinds dropped any base type whose grammar
  kind wasn't on it (in practice every base after the first), so
  `class Foo : Bar, IDisposable` lost the IDisposable edge entirely.
- **C# generic args are stripped from base type names** so
  `IEnumerable<int>` resolves against the `IEnumerable` index entry.
- **`is_test` propagates from File to defined Functions.** Previously
  `meta_bool(f, "is_test")` returned `false` for every Function in
  every language except Rust `#[test]`, so the codebase-level test
  filter was unusable.
- **`build(save_to=...)` now persists the full property graph.** The
  build path skipped the `prepare_save` + `enable_columnar` steps
  `KnowledgeGraph.save()` does, so everything except `id`/`title`/
  `type` was stripped from the file. Round-tripped graphs now match
  in-memory ones.

### Performance

- `code_tree.build` is ~15% faster on polyglot codebases. Two changes:
  the orchestrator walks the source tree once and partitions files by
  language instead of re-walking per parser (was N+1 traversals — 8 walks
  of ~57k entries on dotnet/runtime); and a byte-level aho-corasick
  pre-check skips the full-AST comment walk for files that contain no
  TODO/FIXME/HACK/etc. keywords at all (the vast majority). On
  dotnet/runtime: 20.3 s → 17.4 s wall-clock, mostly from the C# parse
  phase (12.8 s → 11.2 s).
- New per-language and per-phase timings printed under `verbose=True`.

### Fixed

- `code_tree.build` no longer SIGBUSes on deeply-nested expressions in
  source files (e.g. dotnet/runtime's `JIT/Regression/JitBlue/GitHub_10215.cs`,
  a regression test that is literally a chain of thousands of `+`
  operators). Tree-sitter is recursive-descent and was overflowing the
  rayon worker thread stack (~2 MB on macOS); parsers now share a
  dedicated rayon pool with a 16 MB stack, so pathological-but-valid
  inputs parse cleanly across all languages.

## [0.8.33] — 2026-04-30

### Tooling

- New `bench/bench_cohort_cold.py` — spawns a fresh Python subprocess
  per (query, iteration) and runs `sudo purge` between each so the
  OS page cache is dropped before the kglite mmap is re-faulted.
  Settles "is plan X actually faster than plan Y?" questions that
  warm-cache timings can't answer.
- Planner regression test pinning that the user's exact cohort top-K
  shape (with explicit `WITH p` between MATCHes) absorbs `top_k`
  after the fold + desugar + fuse pipeline.

## [0.8.32] — 2026-04-30

### Performance — cohort top-K with PropertyAccess RETURN now absorbs LIMIT

`fuse_match_with_aggregate_top_k` previously required every RETURN
item to be a plain alias of a WITH-projected column. Cohort queries
of the form

```cypher
MATCH (p)-[:P27]->({id: 20})
WITH p
MATCH (p)-[r]-(other)
WHERE NOT (type(r) = 'P50' AND startNode(r) = other)
RETURN p.title, p.description, count(r) AS d
ORDER BY d DESC LIMIT 10
```

include `p.title` and `p.description` (PropertyAccess) in the RETURN,
so the absorber bailed and the fused operator emitted *every* cohort
row before LIMIT — paying property-column I/O for ~73K Norwegians on
Wikidata even though only 10 survived. The relaxed gate now accepts
PropertyAccess on the WITH's group variable (the executor already
preserves `node_bindings[group_var]` for K-winner rows). Warm-cache
runtime on the Norwegians cohort fell from ≈0.9s to ≈0.4s; cold runs
that previously exceeded the MCP 20s ceiling now stay under it.

## [0.8.31] — 2026-04-30

### Performance — fused OPTIONAL MATCH widens to derived aggregates and edge-var counts

Two follow-on fixes after the 0.8.30 release. Both close gaps in
`fuse_optional_match_aggregate` that kept the cohort-style query in
the issue from picking up its existing fast path:

- **Edge-variable counts now fuse.** The fusion gate's local-binding
  set was built from `collect_pattern_variables`, which only returns
  *node* variables. `count(r)` over an OPTIONAL MATCH edge variable
  failed the local-to-OPT check and fell back to the materialized
  per-row expansion. Replaced with a local walk that includes edge
  variables in the OPTIONAL pattern's binding set, minus any names
  already bound by prior MATCH/WITH/UNWIND.

- **Derived `total - count(rp)` expressions now fuse.** Previously
  the gate accepted only pure `count(...)` aggregates; arithmetic
  involving them blocked fusion entirely. The gate now recognizes
  any expression whose only aggregate is `count(...)` and the
  executor substitutes the per-row count into each `count(...)`
  sub-tree before evaluating the surrounding arithmetic. Same row
  cost as the pure-count path.

- **Output columns now reflect the fused operator's own RETURN/WITH
  items.** Previously the fused operator silently inherited the
  upstream's column names, so a downstream consumer reading
  `row["p50_in"]` got a `KeyError`. Visible only when the OPTIONAL
  MATCH adds new RETURN columns the upstream WITH didn't have.

`startNode(r)` / `endNode(r)` returning the matcher's anchor side
instead of the actual graph endpoints (caught while testing) now
look up the edge endpoints via `edge_index`. This was a pre-existing
bug, exposed by Phase 3 actually exercising the predicate.

**Wikidata cohort impact** (warm cache, 124M-node graph; queries
from the issue's "narrow then enrich" report):
- fm1 (`WHERE NOT (type(r) = 'P50' AND startNode(r) = other)`):
  656ms → 624ms (~unchanged; the win was already in 0.8.30)
- fm2 (post-aggregate OPTIONAL MATCH + `total - count(rp)`):
  1290ms → **403ms** (~3.2× faster, fully fused)

## [0.8.30] — 2026-04-30

### Performance — relationship-predicate pushdown (Phase 3)

Pushes `WHERE` sub-predicates that reference only the edge variable
(and the structural peer endpoint of that edge in the pattern) into
the matcher's expansion loop, *before* per-edge bindings allocate.
Selective edge filters that previously forced the materialized 100M+
row path now run during expansion via the disk CSR sorted-edge-type
binary search.

- **`extract_pushable_rel_predicates`** in
  `src/graph/languages/cypher/planner/rel_predicate_pushdown.rs`
  recognizes `type(r) = 'X'` / `type(r) IN […]`, `r.<prop> OP <lit>`
  for `=`/`<>`/`<`/`<=`/`>`/`>=`, and `startNode(r) = peer` /
  `endNode(r) = peer` against the structural peer in the same
  pattern. AND/OR/NOT compositions of those leaves push as a unit
  when the entire subtree is pushable; partial pushdowns over OR /
  NOT correctly leave the predicate alone.
- **`EdgePattern::edge_filter`** carries the compiled
  `RelEdgePredicate` to the matcher. The hot loop in
  `pattern_matching/matcher.rs` evaluates it after the existing
  connection-type check and before the property check — single
  branch-predicted `if let Some` for the no-filter path.
- **Fused count phase honors the filter.** The fused
  `FusedMatch{Return,With}Aggregate` operators run their own count
  loop via `try_count_simple_pattern` /
  `try_count_distinct_peers`; both now apply the inline filter
  during edge iteration so fused queries with a pushed predicate
  produce correct results without falling back to the materialized
  path. `try_fast_with_aggregate_via_histogram` bails when a filter
  is set (the histogram counts every edge of a type by definition).

`startNode(r)` / `endNode(r)` previously returned the
`EdgeBinding`'s pattern-anchor side instead of the actual graph
source/target — silently wrong when the planner anchored on the
right-hand pattern endpoint and walked incoming edges. Fixed by
looking up the edge endpoints via `edge_index`.

**Wikidata cohort impact** (warm cache, 124M-node graph):
- baseline `MATCH (p:Q20-citizen)-[r]-()` aggregate: 224ms
- `WHERE type(r) = 'P19'` (selective): 47s → **667ms** (~70× faster)
- `WHERE type(r) IN ['P19', 'P569', 'P570']`: 47s → **670ms**
- `WHERE NOT (type(r) = 'P50' AND startNode(r) = other)`: 60s → **656ms** (~90× faster, correct result)

### Performance — streaming aggregate + heap top-K (Phase 1)

First slice of a multi-phase rework that lifts streaming primitives
into the Cypher generic execution path. Falling out of the fused fast
path used to cost ~1000× on cohort-scale queries; this slice closes
the gap on shapes that combine `WITH(group, agg)` with `ORDER BY …
LIMIT k` decorations the existing fused operators don't cover.

- **`RowStream` operator pipeline** in
  `src/graph/languages/cypher/executor/stream/`. The driver in
  `executor::execute` tries to absorb a contiguous clause run
  (`WITH/RETURN(group, agg) [→ ORDER BY → LIMIT]`) into a single
  streaming pipeline before the materialized executor sees the
  clauses. On no match the driver falls through with the input
  `ResultSet` unchanged.

- **`StreamingAggregate`**: hash aggregate that builds per-group state
  inline as upstream rows arrive — same I/O profile as the
  materialized path (NodeIndex surrogate keys, deferred property
  reads, re-bucket-by-resolved-value at finalization). Supports
  `count(*)`, `count[(DISTINCT) expr]`,
  `sum/avg/min/max[(DISTINCT) expr]`. Other aggregates (`collect`,
  `std`, percentiles, arithmetic on aggregates) bail to the
  materialized executor unchanged.

- **`HeapTopK`**: `BinaryHeap` of capacity K replaces the full sort +
  truncate path for streaming pipelines that end in `ORDER BY <expr>
  [ASC|DESC] LIMIT k`. O(n log k) instead of O(n log n).

- **`streaming` kwarg on `kg.cypher`** (default `True`). Pass
  `streaming=False` to force the materialized executor — useful for
  parity debugging.

Phases 2-4 will widen recognized shapes (multi-MATCH, OPTIONAL MATCH
streaming, post-aggregate `WHERE`), inline relationship predicates
into pattern expansion, and retire the shape-matched fused operators
once benchmarks confirm streaming parity.

## [0.8.29] — 2026-04-30

### Performance — cohort + multi-MATCH planner improvements

Three planner passes turn cohort top-K and multi-MATCH joins on
Wikidata-scale graphs from "borderline timing out" into "sub-second
warm." Validated end-to-end on the 124M-node / 861M-edge Wikidata
graph; no regressions on point lookups, 1-hop / 2-hop, aggregates, or
load.

- **`reorder_match_clauses`** — orders consecutive id-anchored MATCH
  clauses by edge-type total-count cost. Drives from the rarer side
  first.
  ```cypher
  MATCH (p)-[:P31]->({id:5}) MATCH (p)-[:P27]->({id:183}) RETURN p.title LIMIT 20
  ```
  458s cold / 497s warm → 49s cold / **0.5s warm**.

- **`fold_pass_through_with`** — strips a `WITH x [, y, ...]` clause
  that's a pure projection (no DISTINCT/aggregate/WHERE/ORDER BY) when
  every variable referenced downstream is in the projection list. Lets
  later fusion passes see a contiguous Match-Match span when the user
  wrote `Match WITH p Match …`.

- **`desugar_multi_match_return_aggregate`** — rewrites
  `Match-Match-Return(group, aggregate)` into
  `Match-Match-With(group_var, aggregate)-Return(project)`. Lets the
  existing aggregate fusion fire on the natural "RETURN with
  aggregate" form.

Together, the two simplification passes turn cohort top-K queries
from per-row materialization into the streaming aggregate path:
- Norwegians outgoing-degree top 10: 34s → **0.07s** (490×)
- Norwegians total-degree top 10: 38s → **1.35s** (28×)

The reorder pass is gated to avoid in-memory regressions
(edge-type-counts cache must be populated, id-anchors required,
shared variable required). The simplifications are pure AST rewrites —
no executor changes, O(1) planner overhead.

## [0.8.28] — 2026-04-30

### Performance — slice-built graph load (round 2)

Continuation of the disk-graph load optimisation that landed earlier in
this version. Three further changes drop slice-built Wikidata graph
loads from seconds to ~100 ms:

- **`metadata.json` heavy fields → binary sidecars.** The two
  HashMap-of-HashMap fields (`node_type_metadata`,
  `connection_type_metadata`) move into dedicated
  `node_type_metadata.bin.zst` and `connection_type_metadata.bin.zst`
  files with a hand-rolled length-prefixed format. On the 1B-triple
  Wikidata slice, `metadata.json` shrinks from 5.0 MB to 23 KB, and
  the parse drops from ~1 s to ~10 ms. (The custom `Serialize`/
  `Deserialize` impls on `ConnectionTypeInfo` make bincode
  round-tripping unsafe — hence the hand-rolled binary.)

- **`type_connectivity_cache` is lazy.** Skipped both the
  cartesian-product derive in `apply_to` (clones tens of millions of
  String triples on slice-built graphs) and the eager
  `type_connectivity.bin.zst` read at load. Existing read sites in
  `introspection/describe.rs` already fall through to a bounded edge
  scan when the cache is missing; first `describe()` triggers a
  `compute_type_connectivity` populate. Set
  `KGLITE_EAGER_TYPE_CONNECTIVITY=1` to opt back into eager loading
  for workloads that immediately call `describe()`.

- **`apply_to_with(graph, derive_type_connectivity: bool)`.** Splits
  the implicit derive out of the metadata-application path so the
  caller can opt out. The original `apply_to` is now a thin wrapper
  that defaults to `true` for the in-memory `.kgl` load path (which
  has no separate sidecar).

**Round-2 measurements** (warm load, M2 macOS, external SSD):

| graph        | nodes | round-1 | round-2 | total speedup vs original |
|--------------|-------|---------|---------|---------------------------|
| graph_500.0  | 6 M   | 770 ms  | **62 ms**  | (built fresh) |
| graph_1000.0 | 16 M  | 4.89 s  | **113 ms** | 43× |
| main wikidata| 124 M | 178 ms  | **179 ms** | 51× vs 9.08 s baseline |

The remaining 100-180 ms is dominated by `column_stores_load`
(72-124 ms) — a per-type ColumnStore wrapper construction that's
already mmap-backed.

### Fixed — seg_000 corruption on save of legacy disk graphs

Pre-existing bug exposed by the migration round-trip benchmark: re-saving
a disk graph whose `disk_graph_meta.json` carries `sealed_nodes_bound: 0`
(the serde default for pre-phase-8 graphs) AND has a non-empty
`seg_manifest.json` would call `seal_to_new_segment` with `tail_lo=0`,
`tail_hi=node_count`. That writes a fresh empty `seg_001` AND truncates
`seg_000/out_offsets.bin` and `seg_000/in_offsets.bin` to one entry via
`reconcile_seg0_csr` — the on-disk CSR loses every edge offset, and the
graph reloads with zero traversable edges (the edge data files survive,
but without offsets nothing is reachable).

`DiskGraph::load_from_dir` now bumps `sealed_nodes_bound` to `node_count`
when it detects the legacy zero with a populated segment manifest. Fresh
phase-8+ graphs persist the correct watermark, so the bump is a no-op
for them.

### Performance — disk-graph load

`kglite.load(path)` on the 124M-node Wikidata graph drops from ~9.0s to
~5.5s warm-cache today, and to ~1s once the graph is re-saved with this
version. Three changes:

- **`prefetch_hot_regions` no longer runs by default.** It called
  `madvise(MADV_WILLNEED)` on the 1.9 GB out_offsets + in_offsets arrays.
  On macOS that syscall synchronously schedules readahead and blocks even
  on warm pages — costing ~2.7s of every load on Wikidata-scale graphs.
  The kernel pages in offsets on first query anyway, so the upfront cost
  was a tax with no payoff for typical anchored-MATCH workloads. Set
  `KGLITE_PREFETCH=1` to opt in for first-query latency-sensitive use
  cases.

- **`id_indices` is now mmap-resident.** New raw `id_indices.bin`
  layout: header + sorted-by-type-key directory + per-type sorted u32
  keys / u32 NodeIndex arrays. Lookups are O(log N) binary search on a
  cache-friendly contiguous slice instead of O(1) HashMap probe with a
  cache miss; cost in practice is parity (~50-100 ns either way for
  13M-entry types). Eliminates the 124M `HashMap::insert` rebuild that
  cost ~5.3s on Wikidata. New struct `IdIndexStore` in
  `storage/disk/id_index.rs` with overlay for post-load mutations.

- **`type_indices` is now mmap-resident.** New raw `type_indices.bin`
  layout: header + directory + contiguous `[u32]` slices per type. Reads
  return a `TypeNodesRef` view that yields `NodeIndex` either directly
  from the overlay `Vec` or by reinterpreting the mmap'd `u32` slice.
  Eliminates the 124M `Vec::push` rebuild that cost ~890ms on Wikidata.
  New struct `TypeIndexStore` in `storage/disk/type_index.rs` with
  overlay for post-load mutations (delete paths promote to overlay on
  first mutation).

- **Sub-stage instrumentation** for `DiskGraph::load_from_dir`. Set
  `KGLITE_LOAD_TIMING=1` and stages emit `[TIMING] dg.<name> dur_ms=N`
  lines to stderr (segment_csr, edge_properties, overflow_edges,
  segment_manifest). Useful for measuring the next round of load
  optimizations.

**Backward compatibility.** Loaders try the new `id_indices.bin` /
`type_indices.bin` first, fall back to the old `.bin.zst` legacy
formats, then to a node_slots scan. Existing graphs continue to load —
slower until re-saved. Re-save once with `g.save(path)` to migrate.

**RSS reduction.** Wikidata peak RSS post-load drops from 3.6 GB to
~500 MB (the Integer-variant id_indices + type_indices were the bulk
of heap). General-variant id_indices entries still materialize on
first access.

## [0.8.27] — 2026-04-29

### Changed

- **Default Cypher query timeout is now 180_000 ms (3 min) for every storage
  mode** (memory, mapped, disk). Previously memory had no default deadline,
  mapped was 60s and disk was 10s. The old per-mode defaults left memory
  queries unbounded and disk's 10s ceiling tripped legitimate cold queries
  on large graphs. Override per-call with `timeout_ms=N` (or `0` to disable),
  or globally via `set_default_timeout(ms)`.

### Fixed

- **Cypher property-anchored single-MATCH fusion returned empty results.**
  `MATCH (m {id: X})<-[:R]-(p) RETURN m.title, count(p)` (and the directed
  / DISTINCT / undirected variants) returned zero rows on any graph with
  matching data. `try_count_simple_pattern` bailed with `Ok(None)` when
  the bound node carried property filters; the fused executor's
  `count_for_node` closures `.unwrap_or(0)` that None into a zero count,
  which the row-skip guard then dropped. The bail-out has been removed —
  the bound `NodeIndex` already satisfies its property filter by virtue of
  being selected upstream, so re-checking is unnecessary. Also fixed an
  in-memory-backend miss in `try_count_distinct_peers` (the new helper
  added in the count(DISTINCT) work this release): `edges_directed_filtered`
  is a hint on the in-memory backend and returns every edge regardless of
  connection type, so the function now post-filters by `connection_type`.
- **Cypher fusion machinery now accepts `count(DISTINCT v)` on a node
  variable.** Previously rejected at fusion time, forcing all distinct-count
  queries — the canonical "top-N by relationship count" Cypher shape —
  through the materializing executor (full intermediate cross-product, then
  group-by, then sort). The planner now propagates a `distinct_count` flag
  into `FusedMatchReturnAggregate` and `FusedMatchWithAggregate`; the
  executor uses a per-group `HashSet<NodeIndex>` of peers instead of an
  edge counter, naturally collapsing multi-edges. Edge-centric fast paths
  (which count edges, not distinct peers) are bypassed in distinct mode.
  Fusion is gated on the group node being type- or property-constrained,
  since the fused per-node enumeration only beats the materializing path
  when the group set is small (the materializing path's single sequential
  edge scan wins on unconstrained 124M-node groups). Documented limitation:
  3+ MATCH queries still don't fuse and continue to use the materializing
  path.
- **Cypher planner's selectivity estimator now considers variables bound by
  earlier clauses.** Previously `MATCH (p:Person) MATCH (p)-[:KNOWS]->(c:Type)`
  treated `(p)` as statically unconstrained — worst-possible selectivity — and
  reversed the second pattern to start scanning all `:Type` nodes (millions on
  large graphs) instead of expanding from the pre-bound `p`. The estimator
  now walks clauses in order, accumulates node variables introduced by each
  MATCH/OPTIONAL MATCH, and treats already-bound variables as selectivity 1
  (effectively-anchored). On a 124M-node Wikidata graph, a 3-MATCH query
  with two `{id: …}` anchors now completes in ~1s (was 20s+ timeout).
- **Cypher LIMIT pushdown was unsafe for multi-MATCH queries with WHERE on a
  late-bound variable.** `MATCH a MATCH b MATCH c WHERE c.id = X RETURN ... LIMIT N`
  was rewritten to push `limit_hint = N` into the last MATCH clause. The
  per-row pattern executor's `max_matches = remaining` then interacted
  incorrectly with the outer row loop, causing fewer matching rows than
  expected to surface (e.g. `LIMIT 10` returning 8, `LIMIT 5` returning 3 — or
  zero rows on Wikidata-scale graphs where the WHERE bucket happened to be at
  the tail of the row stream). The planner now only pushes LIMIT into MATCH
  for queries with a single MATCH/OPTIONAL MATCH clause; multi-MATCH queries
  retain LIMIT as a separate clause and apply it after the full pattern
  matching completes. Reproduced and confirmed against the user's
  124M-node Wikidata disk graph (Issue 1 of the 2026-04-29 bug report).
- **Cypher planner now reverses undirected and variable-length patterns by
  selectivity.** `MATCH (other)-[r]-(p {title: 'X'})` previously left
  `other` (no constraints) as the start node, causing a full-graph scan
  with edge expansion from every node. The over-conservative bail-out on
  `EdgeDirection::Both` and `var_length` has been removed — `Both` reverses
  to `Both` (no semantic change) and var-length reversal is symmetric for
  patterns without a path assignment (path-bound patterns are already
  protected by a separate guard). On a 124M-node Wikidata graph this turns
  a multi-minute scan into a sub-second anchored lookup.

## [0.8.26] — 2026-04-28

### Fixed

- **`code_tree` Rust parser missed calls inside macro invocations.** Calls
  inside `format!`, `vec!`, `json!`, `Err(format!(…))`, custom derive
  macros, etc. were silently dropped because tree-sitter-rust represents
  them as `identifier` + `token_tree` siblings rather than `call_expression`
  nodes. The walker now dives into `macro_invocation` token-trees and
  reconstructs synthetic call sites. Resolves the dominant source of
  false-positive orphan-function reports on Rust codebases.
- **`code_tree` Rust parser dropped turbofish call expressions.** Calls of
  the form `path::with::<T>(...)` are wrapped in a `generic_function`
  AST node that the previous match arm didn't handle, so e.g.
  `reconcile_seg0_csr::<DiskNodeSlot>(arg)` produced no CALLS edge. The
  type arguments are now stripped and the inner identifier/scoped path is
  recursively recorded.
- **`code_tree` Rust parser misresolved `Self::method(...)` calls.** The
  `"Self"` segment was emitted as an explicit receiver hint, which matched
  no function's owner type and broke disambiguation for non-unique method
  names. The parser now strips the `Self::` prefix so the resolver's
  implicit caller-owner hint kicks in, yielding the same behaviour as
  bare `self.method()` calls inside an impl block.
- **`code_tree` IMPLEMENTS edge schema excluded `Enum -> Trait`.** A Rust
  enum implementing a trait (e.g. `impl Clone for GraphBackend`) yielded
  no `IMPLEMENTS` edge because the IMPLEMENTS routing only mapped Class /
  Struct sources. `Enum` is now a recognised source label.

### Removed

- **macOS x86_64 wheels.** `x86_64-apple-darwin` is no longer built or
  published to PyPI. Apple Silicon (`aarch64-apple-darwin`) remains. Intel
  Mac users on existing installs are unaffected; new installs will need to
  build from source.

## [0.8.25] — 2026-04-27

### Fixed

- **`code_tree.build` silently parsed only `tests/` for repos with a tooling-only
  `pyproject.toml`.** Manifests that declared no primary source roots (e.g.
  llama.cpp's `pyproject.toml` for poetry-managed scripts, with no
  `<name>/__init__.py` package and no maturin) yielded `source_roots = []`
  and `test_roots = ["tests"]`. The builder then parsed only `tests/`, set
  `parsed_any = true`, and skipped the whole-repo fallback — silently
  producing an undersized graph (e.g. 58 files instead of thousands).
  When a manifest declares zero source roots, the builder now logs the
  situation in verbose mode and falls through to the whole-repo scan
  instead of trusting the test-only roots.

## [0.8.24] — 2026-04-27

C++ parser robustness round driven by analysing nlohmann/json (24% → 12%
signature fallback) and llama.cpp's `src/` (32% → 5.9% fallback, function
count 742 → 1,263 because previously-`unknown`-named functions can now be
called targets, CALLS edges 352 → 2,971).

### Fixed

- **C-style `struct T *` / `enum T` / `union T` parameter types** —
  `struct_specifier`, `enum_specifier`, `union_specifier`, `class_specifier`,
  `sized_type_specifier` were missing from C++ `extract_parameters`'s
  type-recognition list. C-heavy headers like llama.cpp's
  `void llama_grammar_free(struct llama_grammar * grammar)` were losing the
  parameter type entirely (`type_annotation=None`).

- **Out-of-class C++ method names** — `bool Foo::bar() const` produces a
  `qualified_identifier` (`Foo::bar`) child in tree-sitter-cpp, which the
  existing `get_name` walk missed. Now drills into `qualified_identifier`
  and returns the trailing segment (`bar`).

- **Reference-return functions** — `T & foo()` wraps the real
  `function_declarator` in a `reference_declarator`. `parse_function` now
  unwraps it just like it already did for `pointer_declarator`. Functions
  returning references no longer show as `name="unknown"`.

- **C++ template-typed methods, destructors, qualifier-stripped return types,
  and macro-decorated constructors.** Four targeted parser fixes that drop
  nlohmann/json's signature-fallback rate from 24% → 12% and capture 39
  previously-missed template methods.
  - `template_type` and `qualified_identifier` now in `TYPE_NODES` so generic
    return types like `iteration_proxy<int>` and `std::vector<T>` are captured.
  - `type_qualifier`, `storage_class_specifier`, `virtual_specifier` skipped
    in `get_return_type` — fixes `constexpr int foo()` returning "constexpr"
    instead of "int".
  - `destructor_name` recognized in `get_name` — `~Widget` now returns
    `"~Widget"` instead of `"unknown"`.
  - In-class `field_declaration` items containing `function_declarator` are
    routed to `parse_function` (template-typed methods were being treated as
    fields). New `find_buried_function_declarator` walker unwraps
    `parenthesized_declarator` wrappers that tree-sitter-cpp emits around
    macro-decorated constructors (e.g. `JSON_HEDLEY_NON_NULL(3) Foo(int x)`)
    so the real `function_declarator` and its parameters are captured.

## [0.8.23] — 2026-04-27

C++ macro-aware parsing, Go method return-type fix, and receiver attribution
across Go and Rust. Cross-library validation: testify signature-fallback
49% → 0%, KGLite self-graph USES_TYPE edges 2078 → 3197 (+54%), spdlog
macro names eliminated from top return-type list.

### Added

- **`ParameterKind::Receiver`** — Go method receivers (`(c *Call)`) and Rust
  `&self`/`&mut self`/`self` are now captured as structured parameters with
  `kind: "receiver"`, distinct from positional/variadic/kw_variadic. Excluded
  from `param_count` (receivers aren't user-supplied arguments). Cypher
  consumers can filter via `parameters` JSON column.

- **`USES_TYPE` `position="receiver"`** — receivers contribute USES_TYPE edges
  with their own position label. A method `(c *Call) Once() *Call` (receiver +
  return) collapses to `position="both"` consistent with existing aggregation.
  On testify, this drops `position="signature"` fallback from 49% → 0% and
  raises total USES_TYPE edges 251 → 365 (+45%). On KGLite self-graph,
  USES_TYPE edges go 2078 → 3197 (+54%) — Rust `&self` methods now surface
  their owner type as a receiver USES_TYPE edge.

### Fixed

- **C++ parser ignores macro decorators** (`SPDLOG_INLINE`, `FMT_API`,
  `FMT_BEGIN_NAMESPACE`, `Q_INVOKABLE`, etc.). Without this, tree-sitter-cpp
  parses `SPDLOG_INLINE void foo()` so that `SPDLOG_INLINE` looks like a
  type and `foo` becomes the return type — producing `name="unknown"` and
  `return_type="SPDLOG_INLINE"`. Heuristic in `parsers/shared.rs::looks_like_macro_decorator`
  matches all-caps identifiers (length ≥ 2, optional underscores/digits) and
  is applied in `cpp.rs::get_return_type`, `get_name`, and parameter
  extraction. `get_return_type` also recovers from tree-sitter `ERROR` wrappers
  around primitive type keywords (`void`, `int`, `bool`, etc.) — common when
  a macro decorator has confused the parser. On spdlog, macro names are
  eliminated from the top return-type frequency list.

## [0.8.22] — 2026-04-27

A code-graph quality round driven by analysing KGLite's own self-graph and
closing every concrete gap that surfaced. Five new node/edge primitives,
seven new properties on existing nodes, and one small dead-code cleanup.

### Added

- **`BINDS` edges — Python wrapper to Rust pymethod.** Closes the cross-language
  gap where `kglite.KnowledgeGraph.add_nodes` (the Python class method) and
  `crate::graph::pyapi::*::KnowledgeGraph::add_nodes` (the Rust `#[pymethods]`
  impl) lived as disconnected `Function` nodes. The resolver indexes Rust
  functions with `is_pymethod = true` by `(parent_struct_short_name,
  method_name)` and emits `Function -[BINDS]-> Function` for each Python
  method that finds a unique match. Cypher: `MATCH (py)-[:BINDS]->(rs)
  -[:CALLS*]->(impl)` traces a request from the Python entry point to deep
  Rust impl. On the KGLite codebase: ~184 BINDS edges; closes the false-positive
  dead-code finding for `load_ntriples` and other pyapi-exposed functions.

- **Promoted metadata flags as typed Function/Class properties.** Eight booleans
  (`is_pymethod`, `is_pymodule`, `is_ffi`, `is_static`, `is_abstract`,
  `is_property`, `is_classmethod`) plus the `ffi_kind` string are now
  Function-node columns; `is_pyclass` is a Class/Struct column. Replaces
  `f.metadata.get("is_pymethod") == true` JSON-parsing gymnastics with
  `MATCH (f:Function {is_pymethod: true})` direct filters.

- **`USES_TYPE` edges carry a `position` property** (`parameter` | `return` |
  `both` | `signature`). Distinguishes consumers from producers — a function
  that takes `Widget` as a parameter and a function that returns `Widget` no
  longer collapse to the same edge shape. Aggregated per `(function, type)`
  so a single transformation `fn f(w: Widget) -> Widget` emits one edge with
  `position: "both"`. Cypher: `WHERE r.position IN ['parameter','both']` to
  find consumers; `IN ['return','both']` for producers.

- **`Module HAS_FILE File` edges** — closes the natural top-down walk from
  Module → File → Function. Was string-prefix gymnastics on `qualified_name`;
  now `MATCH (m:Module)-[:HAS_SUBMODULE*0..]->(:Module)-[:HAS_FILE]->(f:File)
  -[:DEFINES]->(fn:Function)` returns "what's in this module" in one query.
  Edge name avoids `CONTAINS` (a reserved Cypher keyword for substring matching).

- **`Procedure` nodes — annotation-driven, language-agnostic.** Functions
  whose docstring/leading comment contains `@procedure: NAME` (or
  `@cypher_procedure: NAME`) at the start of a line synthesize a `Procedure`
  node with an `IMPLEMENTED_BY` edge to the function. A single function can
  carry multiple annotations to register under aliases (e.g. both
  `betweenness` and `betweenness_centrality` dispatching to the same impl).
  Generic mechanism for surfacing project-specific registries (Cypher CALL
  procedures, RPC method catalogs, command-bus dispatchers) as first-class
  graph entities. Anchored to line start so prose mentions in docs/tests
  don't false-positive.

- **Annotated all 22 KGLite Cypher CALL procedures** in
  `src/graph/algorithms/graph_algorithms.rs`,
  `src/graph/languages/cypher/executor/rule_procedures.rs`, and
  `executor/call_clause.rs::execute_call_cluster`. Activates the `Procedure`
  node mechanism on the KGLite self-graph: `MATCH (p:Procedure {name: 'pagerank'})
  -[:IMPLEMENTED_BY]->(f:Function) RETURN f.qualified_name` resolves a Cypher
  procedure name to its Rust impl in one query. 27 Procedure nodes (including
  aliases) → 22 implementing functions.

- **Function complexity counters** — `branch_count`, `param_count`,
  `max_nesting`, `is_recursive` now populate on every `Function` node
  produced by `kglite.code_tree.build(...)`. Computed from the
  tree-sitter AST in the same walk that gathers `CALLS` edges, so
  there's no extra parse pass. Per-language branch tables in
  `parsers/shared.rs` cover `if`/`for`/`while`/`case`/`catch`/ternary
  and short-circuit `&&`/`||` forms (cyclomatic-style). Enables direct
  Cypher queries for high-complexity hotspots:

  ```cypher
  MATCH (f:Function)
  WHERE f.branch_count > 30 AND NOT EXISTS { ()-[:CALLS]->(f) }
  RETURN f.qualified_name, f.branch_count, f.max_nesting
  ORDER BY f.branch_count DESC
  ```

- **Generated and minified file skipping during ingestion.** The
  builder now content-sniffs each source file's first 2 KiB before
  dispatching to the per-language parser. Files matching codegen
  markers (`auto-generated`, `DO NOT EDIT`, `code generated by`,
  `<auto-generated>`, `@generated`) are skipped, as are minified
  bundles (one extreme line, or average line width above 500 chars
  across the first 50 lines). Skipped files emit only a `File` node
  with `skip_reason: "generated"` or `skip_reason: "minified"` — no
  Function/Class/Constant nodes — so phantom CALLS edges from
  protobuf stubs and webpack bundles no longer pollute the graph.

  ```cypher
  MATCH (f:File) WHERE f.skip_reason IS NOT NULL
  RETURN f.path, f.skip_reason
  ```

- **Structured `parameters` on `Function` nodes** — JSON-serialised
  list of `{name, type_annotation, default, kind}` per declared
  parameter, with `kind ∈ {positional, variadic, kw_variadic}`.
  Implicit receivers (`self`/`cls`/`&self`/`&mut self`) are excluded.
  Promoting parameters out of the signature string also extends
  `USES_TYPE` resolution: parameter type annotations are now scanned
  alongside the signature and return type, so a function that takes a
  `Widget` argument but doesn't return one now emits the expected
  `Function -[USES_TYPE]-> Widget` edge.

## [0.8.21] — 2026-04-27

Code-analysis tooling round. Closes seven issues filed against KGLite's
own MCP server after a self-analysis session surfaced them — all
visible to users running `kglite.code_tree.build(...)` or
`g.cypher("CALL ...")` against a Rust codebase.

### Added

- **`REFERENCES_FN` edge type — Function → Function** for bare or
  scoped identifiers passed as arguments to higher-order calls
  (`iter.and_then(some_fn)`, `Option::map(my_helper)`). Distinct from
  `CALLS` because the referenced function isn't necessarily invoked
  at the reference site. Dead-code analysis can union the two:

  ```cypher
  MATCH (f:Function)
  WHERE NOT EXISTS { ()-[:CALLS]->(f) }
    AND NOT EXISTS { ()-[:REFERENCES_FN]->(f) }
  RETURN f.qualified_name
  ```

- **`REFERENCES` edge type — Function → Constant** for bare or scoped
  identifiers in function bodies that resolve to a known constant.
  The Rust parser uses `SCREAMING_SNAKE_CASE` as the parse-time
  filter, so local variables don't pollute the edge set. Enables
  detecting unreferenced constants directly from the graph rather
  than via ripgrep.

- **`orphan_node` accepts `link_type` and `direction` parameters.**
  The default behaviour (zero edges in any direction) is unchanged;
  the new params let queries express "no inbound matching edge of a
  specific connection type" — the natural shape for "functions never
  called", "files never imported", etc.:

  ```cypher
  CALL orphan_node({type: 'Function', link_type: 'CALLS', direction: 'in'})
  YIELD node RETURN node.qualified_name
  ```

- **`EXISTS { MATCH ... MATCH ... [WHERE ...] }` multi-clause
  subqueries.** The bare-pattern form already worked; the full
  subquery form with multiple `MATCH` clauses (sharing variables)
  and a `WHERE` predicate evaluated against the merged bindings now
  parses and executes. Multi-hop existence checks no longer have to
  be rewritten as `MATCH ... WITH collect(...) AS xs ... AND NOT y IN xs`.

- **`Project.crate_type` column** captured from `[lib] crate-type` in
  `Cargo.toml`. Lets downstream queries distinguish a regular `lib`
  crate (where `pub fn` is a real export) from a `cdylib` PyO3 crate
  (where only `#[pyfunction]` / `#[pymethods]` matter).

- **`Function.is_test` column** surfaced as a queryable property on
  Function nodes (previously only stored in metadata).

### Fixed

- **`CALLS` edges now include calls inside closure bodies.**
  `closure_expression` was on the parser's NESTED_SCOPES skip-list,
  so `.map(|x| foo(x))` / `.and_then(|x| bar(x))` produced zero
  CALLS edges to the inner function. Closures are expressions in
  Rust, not items — calls inside them belong to the enclosing
  function semantically.

- **`self.method()` receiver-type disambiguation.** When the same
  method name exists on multiple structs, a bare `self.method()`
  call inside a method of `Foo` now narrows to `Foo::method` ahead
  of `Bar::method`, even when both are candidates and live in
  different files. Uses the caller's owner short name as an implicit
  receiver hint when no explicit one is present.

- **`is_test` propagates into inline `#[cfg(test)] mod tests` blocks.**
  Previously only `#[test]` / `#[bench]` annotated functions were
  flagged; helpers inside the test mod weren't, inflating every
  dead-code query against a Rust codebase. Files literally named
  `tests.rs` are also flagged at the file level.

- **`CALL` rule procedures list accepted parameters in error
  messages.** Missing-required-parameter errors now show the full
  schema (required + optional names), so first-time use of a
  procedure doesn't cost three error rounds before guessing the
  parameter name.

### Removed

- **`ARCHITECTURE.md`.** A refactor-time artifact from the 0.8.0
  storage refactor; framing was past-tense and the parity test that
  validated its file-path references was removed alongside it. The
  three other parity gates (god-file cap, unsafe-SAFETY comments,
  mod.rs purity) are evergreen and stay.

- **Dead disk-build infrastructure** — `block_pool.rs`,
  `block_column.rs`, `memory/build_column_store.rs`,
  `AsyncPropertyLogWriter`, plus a sweep of unused methods, fields,
  and enum variants. v3 disk pipeline replaced these; ~3000 lines net.

- **Legacy benchmark suites** — `test_nx_comparison.py` (NetworkX
  comparison, required scipy that wasn't in the venv) and
  `test_performance.py` (used the old `result["stats"][...]`
  subscript API, every Cypher-mutation test failed). Superseded by
  `test_bench_core.py` and `test_bench_memory.py`.

The "completeness round" — six phases that round out the Cypher and
Fluent surface so no primitive a user would reasonably expect is
missing. Every domain (legal, code, sodir, Wikidata) gets value from
each addition; none are domain-specific.

### Added

- **Cypher INTERSECT / EXCEPT (Phase 6 of the completeness round).**
  Cypher now exposes the standard set operators:

  ```cypher
  MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
  INTERSECT
  MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name

  MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
  EXCEPT
  MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
  ```

  `INTERSECT` keeps rows present in both sides; `EXCEPT` keeps rows in
  left but not in right. Both always dedupe, matching SQL/openCypher
  conventions. Internals: new `SetOpKind` enum on `UnionClause` (the
  same `Clause::Union` variant carries all three operators). The
  executor dispatches on `kind`: UNION uses the existing concat-and-
  dedup path; INTERSECT/EXCEPT pre-build a row-hash set from the
  right-side result and filter the left.

  Brings Cypher in line with the fluent-API set ops (`union`,
  `intersection`, `difference`, `symmetric_difference`).

- **Geospatial primitives (Phase 5 of the completeness round).** Round
  out the spatial surface with the standard GIS toolkit operations.

  Cypher scalar functions on WKT/node geometries:
  - `geom_buffer(geom, meters)` — planar buffer.
  - `geom_convex_hull(geoms)` — variadic or list arg.
  - `geom_union(g1, g2)` / `geom_intersection(g1, g2)` /
    `geom_difference(g1, g2)` — boolean ops.
  - `geom_is_valid(geom)` — OGC validity.
  - `geom_length(geom)` — geodesic length for LineStrings; perimeter
    for polygons (sum of rings); 0 for points.

  Cypher CALL procedure:
  - `kg_knn({lat, lon, target_type, k})` YIELD `node, distance_m` —
    *k* nearest nodes of a target type to a coordinate (geodesic;
    location-first, falls back to geometry centroid).

  ```cypher
  CALL kg_knn({lat: 60.4, lon: 5.3, target_type: 'City', k: 5})
  YIELD node, distance_m
  RETURN node.title, round(distance_m / 1000.0, 1) AS km
  ```

  Backed by `geo = "0.33"` (`Buffer`, `BooleanOps`, `ConvexHull`,
  `Validation`, `LengthMeasurable` traits). New helpers in
  `src/graph/features/spatial.rs`; new `geom_arg` resolver in
  `executor/expression.rs` accepts WKT strings, Points, and
  spatial-configured node/property variables.

- **Weighted shortest path (Phase 4 of the completeness round).**
  `graph.shortest_path()` and `graph.shortest_path_length()` now
  accept an optional `weight_property` parameter. When set, the
  search switches from BFS (hop count) to Dijkstra (sum of edge
  weights). Edges missing the property fall back to weight 1.0
  (matching Louvain's existing weighted-adjacency convention);
  negative weights cause the path to be reported as missing.

  ```python
  result = graph.shortest_path(
      "Stop", "A", "Stop", "Z",
      weight_property="cost",
  )
  # {'path': [...], 'connections': [...], 'length': 3, 'weight': 4.7}

  graph.shortest_path_length(
      "Stop", "A", "Stop", "Z",
      weight_property="cost",
  )  # → 4.7 (float; int when unweighted)
  ```

  Internals: new `shortest_path_weighted()` and
  `shortest_path_cost_weighted()` in `algorithms/graph_algorithms.rs`,
  Dijkstra with a `BinaryHeap<State>` keyed on `(distance, node_idx)`.
  The existing BFS path remains the default — no overhead for
  unweighted callers.

- **Structural validators v2 — rule packs extension (Phase 3 of the
  completeness round).** Seven new CALL procedures complete the
  rule-pack family from 0.8.19 by covering *n*-ary and declarative
  checks:

  - `inverse_violation({rel_a, rel_b}) YIELD a, b` — declared-inverse
    relations not symmetric (e.g. `parent_of` without matching
    `child_of`).
  - `transitivity_violation({rel}) YIELD a, b, c` — `(a)-[rel]->(b)
    -[rel]->(c)` chains where the direct `(a)-[rel]->(c)` is absent.
    Generalizes the OCTF subclass-fold audit pattern.
  - `cardinality_violation({type, edge[, min, max]}) YIELD node, count`
    — declarative cardinality. Setting `max:1` catches functional-
    property violations; `min:1` catches missing-required-edge.
  - `type_domain_violation({edge, expected_source}) YIELD source, target`
  - `type_range_violation({edge, expected_target}) YIELD source, target`
    — schema integrity checks on edge endpoints.
  - `parallel_edges({edge}) YIELD a, b, count` — pairs connected by
    more than one edge of the same type (almost always an ETL bug).
  - `null_property({type, property}) YIELD node` — property side of
    `missing_required_edge`.

  All seven follow the existing rule-procedure pattern and surface
  via `CALL list_procedures()` and `describe(cypher=True)`.

- **Lexical text predicates (Phase 2 of the completeness round).** Six
  string-similarity primitives now expressible in Cypher without
  dropping to Python:

  - `text_edit_distance(a, b)` — Levenshtein, UTF-8 aware (uses
    minimum-row DP for O(min(n,m)) memory).
  - `text_normalize(s)` — lowercase, drop punctuation, collapse
    whitespace. The thing every fuzzy-match pipeline reaches for first.
  - `text_jaccard(a, b [, sep])` — token-set Jaccard, default whitespace
    separator.
  - `text_ngrams(s, n)` — character n-grams as a list.
  - `text_contains_any(s, needles)` / `text_starts_with_any(s, prefixes)`
    — variadic or list-argument forms; short-circuit on first match.

  ```cypher
  MATCH (a:Person), (b:Person) WHERE a.id < b.id
  WITH a, b, text_edit_distance(
      text_normalize(a.title), text_normalize(b.title)
  ) AS d
  WHERE d <= 2 RETURN a.title, b.title, d
  ```

- **Expression-engine fundamentals (Phase 1 of the completeness round).**
  The Cypher engine gains the standard scalar/aggregate/list-fold
  primitives that were missing:

  - `properties(n)` / `properties(r)` — full property map of a node or
    relationship (returns a JSON-formatted map; works alongside
    `keys()`).
  - `start_node(r)` / `end_node(r)` — endpoint access on a bound edge
    variable. `start_node(r).name` works via the existing dotted
    property accessor.
  - `reduce(acc = init, x IN list | body)` — list fold with
    accumulator. New `Expression::Reduce` AST variant; mirrors
    openCypher.
  - `percentile_cont(expr, p)` — continuous percentile via linear
    interpolation; `p ∈ [0,1]`.
  - `percentile_disc(expr, p)` — discrete percentile via nearest rank.
  - `median(expr)` — sugar for `percentile_cont(expr, 0.5)`.
  - `variance(expr)` / `var_samp(expr)` — sample variance, n-1
    denominator (matching the existing `std` convention).

  ```cypher
  MATCH (n:Person)
  RETURN median(n.age), percentile_cont(n.age, 0.9), variance(n.age)

  MATCH (n:Person) WITH collect(n.age) AS ages
  RETURN reduce(s = 0, x IN ages | s + x) AS total
  ```

### Changed

- **Example MCP server simplified to two tools.**
  `examples/mcp_server.py` now exposes only `graph_overview` and
  `cypher_query`. The convenience tools (`search`, `find_entity`,
  `read_source`, `entity_context`, `bug_report`) are removed — every
  operation is reachable from Cypher via `MATCH (n) WHERE n.title =
  $text` etc., and the docstring shows the equivalent patterns. Same
  simplification philosophy as the 0.8.19 rule-procedure refactor:
  lean on the Cypher surface.

## [0.8.19] — 2026-04-26

### Changed

- **Rule packs rebuilt as native Cypher CALL procedures.** The
  Python-layer `kglite.rules` package (`g.rules.run(...)`,
  `RuleReport`, YAML packs, ~1,200 lines) is removed; six
  structural-validator procedures live inside the Cypher engine
  alongside `pagerank` / `connected_components`:

  ```cypher
  CALL orphan_node({type: 'Wellbore'}) YIELD node RETURN node
  CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node ...
  CALL missing_inbound_edge({type: 'Discovery', edge: 'IN_DISCOVERY'}) YIELD node ...
  CALL self_loop({type: 'Person', edge: 'KNOWS'}) YIELD node ...
  CALL cycle_2step({type: 'Person', edge: 'KNOWS'}) YIELD node_a, node_b ...
  CALL duplicate_title({type: 'Prospect'}) YIELD node ...
  ```

  Direct graph iteration in Rust replaces the YAML→Cypher→parse round
  trip — single rule on sodir (564k nodes) runs in **<2 ms** vs.
  ~5 ms for the legacy Python pack runner. Composability with
  surrounding Cypher (WHERE / ORDER BY / aggregation) collapses the
  previous two-step `rules_run + cypher_query` flow into a single
  pass.

  Direction validation, anchored type-by-type iteration, and the
  `DirectionMismatch` error survive — ported to Rust. Same agent
  protection without the parallel API.

  Discovery surface: rule procedures appear in
  `describe(cypher=True)` topic list, in the
  `<rules hint="..."/>` extension hint of `describe()`, and in
  `CALL list_procedures() YIELD name`. Per-procedure docs via
  `describe(cypher=['orphan_node'])`. No `<rule_packs>` block. No
  opt-in `advertise()` function. No separate `rules_run` MCP tool —
  agents invoke via `cypher_query`.

  **Breaking change.** Code using `g.rules.run(...)` or any
  `kglite.rules.*` import from 0.8.16–0.8.18 must migrate to the
  CALL syntax. The migration is mechanical: one `CALL` per rule
  with map-syntax parameters and `YIELD node` (or `YIELD node_a,
  node_b` for `cycle_2step`).

  Removed: `kglite/rules/` package, `g.rules` accessor on
  `KnowledgeGraph`, `Rule`/`RulePack`/`RuleReport`/`_RulesAccessor`
  classes, `kglite.rules.advertise()`, `_set_default_rule_pack_xml`
  PyO3 function, `_set_rule_pack_xml` PyO3 method, `rule_packs_xml`
  field on `KnowledgeGraph`, `inject_rule_packs` helper in
  describe.rs, the `<rule_packs>` block in `describe()`, the
  `rules_run` MCP tool from `examples/mcp_server.py` and
  `prospect_mcp_server.py`, `pyyaml>=6.0` runtime dependency.

## [0.8.18] — 2026-04-26

### Changed

- **Rule-pack discovery via `describe()` is now opt-in.** A fresh
  `kglite.load(...)` produces a `describe()` with no `<rule_packs>`
  block — graphs that don't use rule packs incur no agent-facing
  noise. Activation is explicit:
  - `g.rules.run(...)` or `g.rules.load(...)` activates per-graph
    advertising (existing behaviour, unchanged).
  - New `kglite.rules.advertise()` publishes a module-level default
    visible to every subsequent `describe()` across all graphs. Use
    this for MCP servers that expose a rule-pack tool. Idempotent.
  - The `examples/mcp_server.py` `rules_run` tool is now commented
    out; the file documents how to re-enable it for users who want
    rule packs in their MCP surface. The default MCP example is
    rule-pack-free.
- **Per-rule Cypher timeout via `default_timeout_ms`.** Rules can
  declare a YAML-level `default_timeout_ms` and the runner passes it
  as the `timeout_ms` to `g.cypher()` per rule. A caller-supplied
  `timeout_ms` to `g.rules.run(...)` always wins. Lets full-Wikidata
  users set realistic budgets on rules that scan dense node types
  (13M humans, 45M scholarly articles) without affecting the global
  graph timeout.
- **Rule-pack `describe()` integration moved into Rust.** Slice 1.1
  shipped agent-discovery via a Python monkey-patch that wrapped
  `KnowledgeGraph.describe`, called the Rust method (preserved as
  `_describe_native`), and post-processed the XML to splice in a
  `<rule_packs>` block. That dispatch is now native: `describe()`
  is the Rust method again. Per-instance pack XML lives on the
  `KnowledgeGraph` struct (`Mutex<Option<String>>`); a module-level
  default holds the cold bundled-pack inventory. Python's role
  shrinks to rendering the XML on pack `load()` / `run()` and
  pushing it via the new `_set_rule_pack_xml` PyO3 method (and the
  module-level `_set_default_rule_pack_xml`). User-visible XML and
  behaviour are byte-compatible; the wrapper indirection and its
  per-call `str.rfind`/slice/concat are gone.

### Fixed

- **`LIMIT` was applied before `WHERE` filtering, returning fewer rows
  than expected.** A query like
  `MATCH (n:T) WHERE NOT EXISTS { (n)-[:E]->() } RETURN n.id LIMIT 5`
  could return 0 rows when the first 5 candidate nodes all failed the
  WHERE predicate. Root cause: the planner pushed the LIMIT hint into
  `PatternExecutor`, which capped *candidates* before the inline
  WHERE filter ran. Fix: skip the limit hint at pattern-execution
  time when an inline WHERE is present, and apply LIMIT after the
  WHERE filter (as the surrounding executor already attempts to).
  Affects any filtered query with LIMIT, not just `NOT EXISTS`.

### Added

- **Rule packs** — agent-discoverable structural validators. New
  `g.rules` sub-namespace exposes `list()`, `load()`, `run()`, and
  `describe()` for named YAML packs that compile to Cypher and emit a
  structured `RuleReport`. The bundled `structural_integrity` pack
  (v1.1) ships six universal cross-graph rules: orphan nodes,
  self-loops, short cycles, missing-required-edge (outbound),
  missing-inbound-edge, and duplicate titles. `g.describe()` surfaces
  a `<rule_packs>` block so agents discover packs through the same
  XML they consume for schema. See
  [`docs/guides/rules.md`](docs/guides/rules.md). Reports are lazy:
  `.summary` returns counts without materialising rows, and runs are
  cached per `(pack_name, params, graph)`. New runtime dependency:
  `pyyaml>=6.0`.
- **Rule-pack ergonomics:**
  - `summary["any_truncated"]` — top-level boolean so agents can
    one-glance check if any rule hit its LIMIT.
  - `report.is_suspect(node_id)` — O(1) cross-reference helper that
    returns `[(rule_name, severity), ...]` for rules that flagged the
    node. Built lazily; accepts string or int ids.
  - `g.rules.list()` now reads bundled-YAML headers lazily so cold
    inventory shows real version + rule_count + description (no
    placeholders) before any pack has been loaded.
  - Optional `usage_hint:` field on a pack — surfaced via
    `g.rules.describe(name)` and as an XML attribute in
    `g.describe()` so agents can read "use this pack when…" guidance
    inline with the schema.
  - `to_markdown()` truncates list-typed cells (e.g. the `ids`
    column in `duplicate_title`) to 3 elements + " (+N more)" so
    agent-pasted output stays readable.
  - **Direction-aware `missing_*_edge` rules.** New optional
    `validates_direction:` rule field (`"outbound"` or `"inbound"`).
    The runner inspects `g.connection_types()` and refuses to execute
    when the `(type, edge)` pair flows the wrong way in the graph's
    actual schema, surfacing a `DirectionMismatch` error that
    suggests the right rule. The bundled `missing_required_edge` and
    `missing_inbound_edge` opt in. Prevents trivial rule firing where
    e.g. asking for incoming `IN_LICENCE` on a `Wellbore` would have
    matched every wellbore meaninglessly.

## [0.8.17] — 2026-04-26

### Performance

- **Two-MATCH count fusion: top-K-by-degree filtered queries now run
  ~20× faster.** The shape
  `MATCH (w)-[:T]->(b {nid:'X'}) MATCH (w)-[r]-() WITH ...
  count(r) ... ORDER BY count DESC LIMIT k`
  used to materialise one row per edge for every group key (e.g. 4 M
  edge rows for 416 k Wikidata writers — 494 s on the full graph).
  The aggregation-fusion pass at
  `src/graph/languages/cypher/planner/fusion.rs` now also recognises
  `[Match, Match, With(count)]` and folds it into a single
  `FusedMatchWithAggregate` whose secondary pattern drives the
  per-group-key degree count via the existing `count_edges_filtered`
  fast-path. Measured: top-10-by-degree on Wikidata writers
  **494 s → 24 s (20×)**. The remaining time is one degree lookup
  per group key (832 k mmap reads) — further wins live in storage,
  out of scope for this session.
- **`count_edges_filtered` fast-path now handles undirected `[r]-`
  edges.** Previously the fast-path returned `None` for
  `EdgeDirection::Both`, forcing the slow per-edge enumeration. It
  now sums incoming + outgoing `count_edges_filtered` calls — the
  canonical "total degree" pattern. Both the new two-MATCH fusion and
  the existing single-MATCH `WITH count` benefit.
- **Per-group-key count phase in `execute_fused_match_with_aggregate`
  now runs in parallel** above 4 096 group keys. Each
  `count_edges_filtered` call is a read-only mmap lookup, so rayon's
  par_iter overlaps the per-call I/O instead of serialising it.
  Measured on the same Wikidata top-10-by-degree query
  (124 M nodes / 861 M edges): **24 s → 8.5 s (~2.5×)** on top of
  the fusion win. End-to-end wall on `top_writers.py` is now 73 s,
  vs. 510 s before any of this session's work (~7× total).
- **Top-K hint absorbed into `FusedMatchWithAggregate`.** A new
  planner pass `fuse_match_with_aggregate_top_k` recognises the
  shape `[FusedMatchWithAggregate, Return, OrderBy(count_alias),
  Limit(k)]` (where the RETURN is a pure pass-through projection)
  and pushes the K-bound into the fused stage. The executor sorts
  by count first and then evaluates the group-key projection
  expressions only for the K winners — saves N×P
  `evaluate_expression` calls when N is large and K is small. The
  Wikidata top-10-by-degree query goes from materialising 416 k
  rows to 10; per-row property reads stop being the tail cost
  (modest 5% gain on top of the parallel-count win, but principled:
  "only do necessary work" — projection-heavy queries get a much
  larger benefit).
- **Lazy RETURN — defer per-row property evaluation until Python
  reads each cell.** The planner's new `mark_lazy_eligibility`
  pass annotates the terminal RETURN with `lazy_eligible = true`
  when the query is `MATCH … (WHERE …) RETURN <prop access>` and
  there's no downstream operator that needs row values
  (DISTINCT/HAVING/ORDER BY/aggregate/WITH/UNWIND/CALL/UNION/
  mutation all force the eager path). The executor skips
  `execute_return_projection`'s per-row loop and hands the
  pending rows + return items to the Python `ResultView` via a
  side-channel `LazyResultDescriptor`. `ResultView` materialises
  cells on access (memoised via a `Mutex<Vec<Option<…>>>` so
  repeat reads are free), and `__len__` becomes O(1). Measured
  on the same Wikidata script: the find-writers query
  (`MATCH … RETURN nid, title`, used only for `len()` in the
  caller) **57 s → 35 s (~1.6×)**. End-to-end on
  `top_writers.py` is now **49 s** — down from the original
  **510 s** before this session — **~10× total**.

### Performance

- **Phase 1 N-Triples loader is ~1.7× faster.** Steady-state on
  Wikidata's `latest-truthy.nt.bz2` went from ~2.4 M tri/s to
  **~4.1 M tri/s** (`--size 50` build dropped 20.83 s → 15.96 s; full
  Wikidata Phase 1 projects from ~2 h to **~70 min**). Profiling with
  `samply` showed the loader thread spending ~32% of CPU in
  `libsystem_malloc` and ~10.7% in `core::str::pattern::TwoWaySearcher`
  (used by `str::find`). Four targeted changes:
  - **Byte-level `parse_line`** (`src/graph/io/ntriples/parser.rs`):
    swap `line.find("> ")` for `memchr::memchr(b'>')`. URIs in
    N-triples cannot contain `>`, so a single byte scan is sufficient.
  - **`EntityAccumulator` capacity preallocation**
    (`HashMap::with_capacity(32)`, `Vec::with_capacity(8)`):
    eliminates `RawVecInner::finish_grow` reallocs in the per-entity
    accumulator.
  - **`scratch_props` reuse in `flush_entity`**: hoist the
    `Vec<(InternedKey, Value)>` out of the function so the alloc cost
    is paid once per build instead of per entity.
  - **`mimalloc` as global allocator** (`src/lib.rs`): pure Rust /
    build-time-only dependency; ~10% wall-time win on the loader on
    top of the parser-side changes.
- **Reader-thread channel batches 50k → 200k.** Reduces per-batch
  sync overhead 4× without growing peak RSS meaningfully.

### Added

- **Block-level parallel decoder for single-stream `.bz2` files.** Wikidata
  ships `latest-truthy.nt.bz2` as a single bz2 stream, so the existing
  stream-level scanner in `parallel_bz2.rs` was falling through to a
  single-threaded `MultiBzDecoder` (~1 M triples/s ceiling). The new
  single-stream path delegates to `bzip2_rs::ParallelDecoderReader`
  (paolobarbolini/bzip2-rs, MIT/Apache-2.0), which finds bit-aligned
  block magics inside one stream and decodes blocks on rayon workers.
  Measured Phase 1 throughput on Wikidata: **~1.0 M tri/s → ~3.3 M tri/s
  (3.3× speedup)** at the same memory ceiling. Multistream files still
  use the existing stream-level path. Pinned to a git rev because the
  published 0.1.2 crate ships an older `Cargo.toml` without the `rayon`
  feature flag.
- **Phase 1 progress bar now shows ETA when `max_entities` is set.**
  When the caller has set an entity cap, the loader emits the bar
  position as `entities_created` against `total = max_entities`, so
  tqdm can compute ETA from the entity rate. Without a cap the bar
  still tracks triples (no total → no ETA, just rate). The unused
  counter ships in the event's `fields` dict either way.
- **Ctrl+C cancellation of `load_ntriples` builds.** Phase 1 runs
  inside `py.detach()` (GIL released so Python heartbeat threads can
  run), which previously meant SIGINT couldn't reach Python until the
  Rust call returned — i.e. Ctrl+C did nothing during a multi-hour
  Wikidata build. The progress sink now reacquires the GIL on every
  update event and calls `Python::check_signals`; a pending SIGINT
  flows back through a new `Cancelled` marker on `ProgressSink::emit`,
  unwinds the loader cleanly, and surfaces as `KeyboardInterrupt` on
  the Python side. Cancellation requires a `progress=` callback (which
  is the default in the bench script and dataset wrappers).

### Changed

- **`bench/wikidata_e2e.py` CLI overhaul.** `--progress` is now the
  default (use `--legacy-progress` for the old `[Phase X]` stderr
  output). Removed `--quiet` — wrapper-level status messages
  (cooldown checks, cache hits) print regardless; the loader's
  per-phase eplog lines are auto-silenced when tqdm is active so they
  don't fight the bar. Renamed `--size` to `--entities-m` to make it
  clear the cap is in millions of *entities*, not triples (`--size`
  remains as a deprecated alias).
- **`kglite.datasets.wikidata.open` auto-silences the loader when
  `progress=` is set.** Wrapper-level `verbose=True` controls the
  cache-hit / cooldown messages; loader `verbose` is forced off when
  a progress callback is wired so tqdm owns the terminal.

### Added (continued)

- **Structured build-phase progress callback for `load_ntriples`.** New
  `progress=` kwarg accepts a Python callable that receives one dict
  per phase event (`start` / `update` / `complete`) for each of
  `phase1` (streaming), `phase1b` (columnar build), `phase2` (edges),
  `phase3` (CSR), and `finalising`. Phase 1 fires updates every 5M
  triples (decoupled from the 60s stderr gate) so a UI driven by the
  callback stays live. Errors raised by the callback are swallowed so
  a broken UI cannot kill a multi-hour build.
  Pure-Rust trait `ProgressSink` lives in
  `src/graph/io/ntriples/mod.rs`; the PyO3 adapter that translates a
  Python callable into a sink lives in `src/graph/pyapi/kg_core.rs`,
  keeping the loader free of `pyo3` types.
- **`kglite.progress.TqdmBuildProgress`** — drop-in tqdm-backed
  reporter. One bar per phase, with RSS (via `psutil`) and per-phase
  counters in the postfix. `pip install tqdm psutil` to use.
- **`bench/wikidata_e2e.py --progress`** — opt-in flag that wires
  `TqdmBuildProgress` into the e2e benchmark.

### Fixed

- **`load_ntriples` no longer panics with `slice index starts at A
  but ends at B` on large disk/mapped builds.** Wikidata builds past
  ~450 M triples crashed in `MmapColumnStore::read_str` on reload.
  Root cause: `flush_entity` for entities whose ID didn't parse as a
  Q-code wrote `Value::String(acc.id)` into the `nid` column, which
  flipped that column's `id_is_string=true`. Subsequent entities with
  `Value::UniqueId` left their string offsets uninitialised (zero), so
  reload hit `start > end` decoding them. Fix: in disk/mapped mode,
  skip entities whose ID is not a parseable Q-code at the top of
  `flush_entity` (these were unreachable in the canonical Wikidata
  query surface anyway). In-memory mode is unaffected.
- **Cypher `DETACH DELETE` no longer breaks subsequent typed-edge
  traversals.** Pre-fix, after a Cypher `DETACH DELETE`, fluent
  `g.select(t).traverse(conn_type, ...)` (and any `make_traversal`
  caller) would throw *"Connection type 'X' does not exist in
  graph"* even when `X` still had millions of live edges. The
  Cypher executor's `execute_delete` invalidated the
  `edge_type_counts_cache` but left the `connection_types`
  HashSet alone — `has_connection_type()` consults the HashSet
  first and returned a stale negative. Fix: clear the
  `connection_types` cache on Cypher delete, and add a final
  fall-through in `has_connection_type()` to the disk backend's
  authoritative `conn_type_index_*` arrays.
  Surfaced by `bench/benchmark_full.py` against every disk row.

### Performance

- **Parallel multistream `.bz2` decoder for `load_ntriples` —
  closes the gap with `.zst`.** Wikidata / pbzip2 dumps are a
  concatenation of independent bz2 streams; the previous
  `bzip2::read::MultiBzDecoder` walked them sequentially on a
  single core. New `parallel_bz2::open()` (in
  `src/graph/io/ntriples/parallel_bz2.rs`) scans the file for
  `BZh[1-9]` + 6-byte block magic, dispatches streams to a
  worker pool sized by a memory budget (256 MB default, after
  pbzip2's `NumBufferedBlocksMax`), and re-orders the
  decompressed chunks behind a single `Read` surface. Single-
  stream `.bz2` files take a fast path through `MultiBzDecoder`
  with no thread-pool overhead. Workers join before
  `load_ntriples` exits Phase 1, so no parallelism leaks into
  the Phase-2/3 rayon pool.
  Measured: wiki100m bz2 **99 s → 34 s (2.9×)**, wiki200m
  **199 s → 71 s (2.8×)**. bz2/zst ratio 3.4× → 1.19×.
- **`enable_columnar()` is now idempotent on the already-columnar
  fast path.** Previously, every `g.save()` re-ran the full
  per-node columnar rebuild — even when the graph was already
  columnar and unmodified. At wiki100m memory mode this cost
  ~257 s of pure waste on consecutive saves. Now `enable_columnar`
  walks every node once (O(N) cheap matches) and short-circuits
  if all nodes are `PropertyStorage::Columnar` AND their
  `Arc<ColumnStore>` matches `graph.column_stores` for the type
  (the Arc-pointer check catches the common
  `add_nodes(conflict_handling="update")` fork pattern that would
  otherwise lose updates on save). Measured wiki5m: consecutive
  `save()` 455 ms → **177 ms (2.6×)**.

### Changed

- **`verbose=True` on `load_ntriples` is now phase-oriented.** Previous
  output was a mix of `[T+30s]` timestamps and ad-hoc sub-step prints;
  Phase 2/3 output in particular was developer-grade noise. The new
  output is a small set of `[Phase N]` gate messages — open/close
  pairs around each major stage of the build:
  ```
  [Phase 1] Streaming and parsing N-triples (...)
  [Phase 1] 12.3M triples, 2.8M entities, 8.5M edges buffered — 205k triples/s
  [Phase 1] Complete: ... in 47m18s
  [Phase 1b] Building columnar storage (...)
  [Phase 1b] Complete in 8m42s
  [Phase 2] Creating edges
  [Phase 2] Complete: ... edges in 11m22s
  [Phase 3] Building CSR edge index
  [Phase 3] Complete in 2m04s
  [Finalising] Building auxiliary indexes + saving metadata
  [Finalising] Complete in 32s
  [Build] Total elapsed: 1h09m54s
  ```
  Sub-step timings (CSR step 1/4, peer-count histogram, mmap layout,
  per-type flush logs, interner save timings, Q-code resolution
  timings, …) move behind `KGLITE_BUILD_DEBUG=1`. The legacy
  `KGLITE_CSR_VERBOSE` env var is replaced by `KGLITE_BUILD_DEBUG`
  (one flag for all build sub-step output).

### Added

- **`kglite.datasets.sodir.open(workdir, ...)` — one-call lifecycle for
  Sodir factmaps petroleum data.** Resolves CSVs from the public
  ArcGIS FeatureServer at
  `https://factmaps.sodir.no/api/rest/services/DataService`, applies
  the FK pre-processing the existing build script does, and builds
  the graph via the packaged blueprint. Default storage is
  ``memory`` — Sodir is small enough that disk caching adds little
  on top of CSV caching:

      g = sodir.open("/data/sodir")  # memory; index_cooldown_days=14, dataset_cooldown_days=30
      g = sodir.open("/data/sodir", storage="disk")  # opt-in for cross-process reuse

  Workdir layout: `csv/` (fetched datasets, flat), `sodir_index.json`
  (per-dataset row count + timestamps), `graph/` (disk-mode only).
  Index sweep cheaply re-checks remote row counts every 14 days; only
  changed datasets re-download. Hard cooldown forces full per-dataset
  refresh every 30 days even if counts match.

  **Complement blueprints**: pass
  ``complement_blueprint=path/to/extra.json`` to add new node types
  / edges on top of the packaged baseline. The file is persisted to
  ``workdir/blueprint_complement.json`` on first call and auto-loaded
  on subsequent calls. Pass ``use_complement=False`` to skip it for a
  single call, or ``sodir.remove_complement(workdir)`` to drop it
  permanently. Deep merge with **base-wins on key collisions by
  default** (the packaged baseline tracks the canonical Sodir REST
  catalog and stays authoritative); set
  ``complement_overrides=True`` to flip when the complement should
  win.

  The blueprint walker auto-detects which datasets are referenced and
  fetches only those — adding new node types to the blueprint
  triggers fetches the next time `open()` runs. The packaged
  baseline ships only the 33 node types whose CSVs are fetchable
  from REST (no sideloaded prospect / play / ocean data); use a
  complement to layer those in.

  **Parallel fetcher**: ``workers`` parameter (default 4) drives a
  thread-pool that pulls dataset jobs off a shared backlog. tqdm
  progress bar replaces per-dataset prints so verbose output stays
  one line tall. Geometry handler is defensive against empty
  ``coordinates: []`` (some pre-1970 Sodir wellbores) — drops those
  features' geometry without aborting the fetch. KGLite is independent of Sodir / the
  Norwegian Offshore Directorate; see module docstring. Catalog
  (LAYERS / TABLES / FACTMAPS_LAYERS) vendored from
  `kkollsga/factpages-py`.

- **`kglite.datasets.wikidata.open(workdir, ...)` — one-call lifecycle
  for Wikidata `latest-truthy` graphs.** Resolves the dump (download,
  resume `.part`, refresh on cooldown), builds the disk or in-memory
  graph, and returns it. Subsequent calls cache-hit on the saved
  graph at `workdir/graph[_<N>m]/`. Two storage backends:

      g_full = wikidata.open("/data/wd")                        # disk graph
      g_100m = wikidata.open("/data/wd", entity_limit_millions=100)
      g_mem  = wikidata.open("/data/wd", storage="memory",      # rebuild every call
                              entity_limit_millions=10)

  Sized slices (`entity_limit_millions=100/200/...`) live alongside
  the full graph (`graph_100m/`, `graph_200m/`, `graph/`) and all
  share the same `latest-truthy.nt.bz2` dump under `workdir`.

  Also exports `fetch_truthy(workdir, cooldown_days=31)` for the
  dump-only path. KGLite is independent of the Wikimedia Foundation;
  see module docstring.

- **`load_ntriples` releases the GIL.** Multi-minute loads no longer
  block Python threads — heartbeat / progress monitors and other
  background workers run on schedule throughout the build. Required
  for the new minute-cadence reporting in
  `examples/wikidata_disk.py`.
- **`bench/benchmark_full.py`** — full-stack lifecycle benchmark
  (build / save / load / mutate / resave / Cypher / fluent) across
  every storage mode × Wikidata subset. Wide-pivot CSV output —
  one row per (run, mode, dataset). Errors preserved both in the
  `errors` column (truncated, semicolon-separated) and in a
  sidecar `bench/benchmark_full.errors.log` (full text,
  tab-separated for `cut`-friendly inspection).
- **`bench/results.py`** — pandas-backed analysis tool over the
  bench CSV. Three commands: `latest` (most recent measurement
  per cell), `trends` (per-run time-series for filtered cells),
  `deltas` (consecutive-run deltas — surfaces regressions). Use
  `--mode` / `--dataset` / `--cols` filters.
- **`--languages` CLI flag on `bench/wiki_benchmark.py` and
  `bench/api_benchmark.py`** (default `en`). Matches the existing
  flag on `bench/wikidata_e2e.py`. Threads through subprocess
  scenarios via argv (wiki_benchmark) and `KGLITE_BENCH_LANGUAGES`
  env (api_benchmark, which uses positional args). Pass
  `--languages ""` to keep all languages, but the canonical query
  suite expects English type names like `:human` and will return
  zero rows otherwise.
- **Post-build SANITY PROBE in `bench/wikidata_e2e.py`.** Quick Q42
  title/description lookup + top-5 type histogram printed after
  every build, even with `--no-queries`. Catches language-filter or
  auto-type-rename regressions immediately, before query-suite time.

## [0.8.15] — 2026-04-25

### Performance

- **Mapped-mode property index — `MATCH (n:Type {prop: val})` in O(log N).**
  `MappedGraph` now carries a lazy per-`(node_type, property)` and
  cross-type property index alongside the 0.8.15 conn_type index. On
  first `lookup_by_property_eq` / `lookup_by_property_prefix` /
  `*_any_type` hit the backend iterates nodes once, emits a sorted
  `(key, NodeIndex)` array — same layout as disk's persistent
  `PropertyIndex` — and caches it behind an `Arc<RwLock<…>>`;
  subsequent queries binary-search that array. Alias handling
  matches disk (reads from `node.title` for `title`/`label`/`name`
  and `node.id` for `id`/`nid`/`qid` so the `add_nodes(...,
  node_title_field=...)` pipeline "just works"). Invalidated on
  `add_node`/`remove_node`/`node_weight_mut`. Measured on a 938 k
  node / 212 k Q5 subset (wiki100m Wikidata humans):
  `MATCH (n:Q5 {title: 'Douglas Adams'})` — **37 ms first call
  (builds the index), 0.1 ms warm** (index hit). Prefix scans stay
  in the single-digit-ms range. Correctness pinned by
  `tests/test_mapped_property_index.py` against both memory and disk.

### Added

- **Cypher `count { <pattern> }` subquery expression.** `count { ... }`
  in `WITH` / `RETURN` / `ORDER BY` / `WHERE` now evaluates to the
  number of matches of the inner pattern, scoped to the outer row's
  bindings. Previously the parser rejected the shape with
  *"Expected property name or .property in map projection, got
  Some(LParen)"* because the identifier-followed-by-brace dispatch
  routed to map projection (`n { .prop1, .prop2 }`) unconditionally;
  the parser now special-cases `count` and routes to a new
  `parse_count_subquery` that mirrors the existing `EXISTS { ... }`
  grammar. New AST variant `Expression::CountSubquery`; the executor
  runs the pattern via the shared `PatternExecutor`, bindings-
  compatible with the outer row, with optional inline `WHERE`.
  Parity across all three storage modes verified by
  `tests/test_cypher_count_subquery.py`. Cypher shapes like
  `WITH a, count{(a)-[:REL]->()} AS n` now work out of the box.

### Performance

- **Mapped-mode query acceleration — lazy per-connection-type index.**
  `MappedGraph` was a bare `StableDiGraph` wrapper with none of the
  inverted indexes that make disk-mode fast (`conn_type_index_*`,
  `peer_count_*`, CSR sorted by type). Cypher queries that depend on
  those structures on disk did full-graph scans on mapped — 2-10×
  slower than disk despite every byte being in RAM. 0.8.15 adds a lazy
  `MappedTypeIndex` populated on first typed-edge query per connection
  type: CSR-style sorted source lists, per-peer count histograms, and
  per-source edge slices. Overrides `sources_for_conn_type_bounded`,
  `lookup_peer_counts`, and `count_edges_grouped_by_peer` on the
  mapped `GraphRead` impl; `filter_by_connection` (powering
  `where_connected`) now hoists the source list into a `HashSet` once
  per call instead of probing `edges_directed_filtered` per node.
  Measured on wiki1000m (1 B triples):
  - Cypher `P31 class counts`: **150 ms → 0.5 ms (300×)**.
  - Cypher `2-hop P31 + P279`: **180 ms → 0.9 ms (200×)**.
  - Cypher `P31 LIMIT 50`: **5.6 s → 0.8-1.1 s (5-7×)**, now **beats
    disk at every subset** (disk wiki1000m was 1.40 s).
  - Cypher `Q5 (human) lookup`: 77 ms → 50-54 ms (1.5×).
  - Fluent `traverse P31 out unlimited`: unchanged (74 ms; bare
    petgraph scan is already optimal for sparse-degree nodes and
    avoids the ~100 ns/call index-lookup overhead).
  Correctness preserved: same row counts across storage modes on
  every benchmarked query. Index is built lazily per conn_type,
  amortised across subsequent queries of the same type, and
  invalidated on any edge mutation.

- **Mapped-mode `load_ntriples` routes through the disk fast path.**
  Previously `storage="mapped"` fell through to `DirGraph::enable_columnar`,
  which iterates every node once, clones each property map into a `Vec`,
  and pushes row-by-row into per-column `MmapOrVec` instances that grow
  via `set_len` + remap; each schema extension additionally triggered
  `Arc::make_mut` store clones. On wiki50m (377 k nodes / 282 k edges)
  this ran at ~430 k triples/s with a 5.1 GB peak RSS. Mapped now shares
  the disk path's property-log + single-`columns.bin` pipeline:
  properties stream to a zstd-compressed log during Phase 1, Phase 1b
  replays the log once into a pre-allocated mmap, and a new second-pass
  links each node's `PropertyStorage` to the shared
  `Arc<ColumnStore>` by row_id. Measured (bench/wiki_benchmark_mapped):
  - wiki50m build: **116 s → 13.5 s (8.6× faster)**, peak RSS
    **5073 MB → 828 MB (6× less memory)**.
  - wiki100m build: **? → 29 s**, peak RSS **? → 1.5 GB** — now
    within 1% of disk-mode build time.
  - Same Cypher + fluent rowsets across modes (round-trip oracle in
    `tests/test_incremental_columnar.py::TestNTriplesColumnar`).
  Memory-mode N-Triples load (`storage=default`) is unchanged — still
  goes through the non-columnar `PropertyStorage::Map/Compact` path.

- **Fluent traverse + `where_connected` use CSR-filtered edge iterator.**
  `core/traversal.rs` (`make_traversal_fast`, `make_traversal_full`) and
  `core/filtering.rs` (`filter_by_connection`, i.e. `.where_connected()`)
  previously called `graph.edges_directed(node, dir)` and post-filtered on
  `connection_type`. On disk-mode graphs with `csr_sorted_by_type=true`
  (the `merge_sort` algo that the `wikidata_disk.py` example uses), we now
  pass the connection key into `edges_directed_filtered` so the DiskEdges
  iterator can binary-search the CSR range — O(log D) instead of O(D) plus
  per-edge `EdgeData` materialisation. This is the same fast path the
  Cypher executor has used since 0.8.0 and targets the shape that the
  Wikidata fluent suite regressed on: `select("Entity").traverse("P31",
  limit=100)` was ~71 s, `.where_connected("P31")` was ~887 s, and
  `traverse P31 in limit 50` was ~2171 s. Heap backends ignore the hint;
  correctness is preserved by the existing post-filter.

### Changed

- **`load_ntriples` verbose output is no longer per-type.** Phase 1b used
  to print one `N dense cols, M overflow cols` line per type with any
  overflow and one `overflow bag X MB for N sparse cols` line per type
  with a non-empty overflow bag. On Wikidata this was ~90 k lines. Both
  are now collapsed into a single Phase 1b summary per pass (`columns — N
  dense, M overflow across K types with sparse cols` and `overflow bags —
  X.X MB across N types, M sparse cols total`).
- **Load-phase progress lines throttled to time (≥ 15 s), not bucket.**
  The loader used to emit a progress line every 5 M triples, which was
  ~2× per second on a fast machine and scrolled the interesting phase-
  timing lines off screen. Now the 5 M bucket is still a cheap fast-loop
  counter but the line only fires when ≥ 15 s have passed since the last
  one. Lines now prefix `[T+NNNNs]` so they interleave cleanly with the
  existing phase-timing output.

### Fixed

- **`examples/wikidata_disk.py`** — the fluent cases called `.where_(...)`,
  but the PyO3 binding exposes the method as `.where(...)`; on a large
  graph this masked the real traversal slowness behind an
  `AttributeError`. Expanded the Cypher/fluent suites with 23 + 15 more
  diverse queries (typed 1-hop, 2-hop chains, parameter binding,
  `ORDER BY` on bounded scans, and string-prefix/contains filters). The
  fluent suite now introspects via `g.node_type_counts()` and skips
  unavailable types instead of raising.

## [0.8.14] — 2026-04-24

### Performance — disk-graph `kglite.load()` fast-load series

Four independent on-disk format changes aimed at the serde overhead
that dominates `kglite.load()` on large disk-mode graphs. Profiling
the 124 M-node, 863 M-edge Wikidata graph
(`wikidata_disk_graph_0.8.11`, 81 GB on disk) showed zstd
decompression + mmap setup account for only ~5–6 s of a ~77 s cold
load; the remaining ~70 s is serde rebuild cost on three bulk
structures, plus a 266 MB JSON array inside `metadata.json`. Each
format change replaces that cost with flat packed slices + exact
`HashMap::with_capacity` sizing — same in-memory representation,
zero consumer surface change.

- **`type_connectivity` out of `metadata.json` into a packed binary.**
  On the 81 GB graph this field was 266 MB of a 415 MB JSON file
  (3,176,503 `ConnectivityTriple` entries). The new
  `type_connectivity.bin.zst` at the graph root is:

  ```
  [ 0.. 8]  magic       = b"KGLTCN1\0"
  [ 8..12]  version     = u32 LE (= 1)
  [12..16]  num_entries = u32 LE
  [16..n*32+16]  entries: (u64 src_key, u64 conn_key, u64 tgt_key, u64 count) × n
  ```

  Keys are interner hashes (`InternedKey::as_u64()`). Disk-mode save
  strips the field from metadata.json; in-memory `.kgl` saves keep
  embedding it for single-file portability.

- **`type_indices.bin.zst` flat CSR binary, interner-keyed.** Replaces
  bincode `HashMap<String, Vec<NodeIndex>>` with three packed slices:

  ```
  [ 0.. 8]  magic       = b"KGLTIDX1"
  [ 8..12]  version     = u32 LE (= 1)
  [12..16]  num_types   = u32 LE
  [16..24]  total_nodes = u64 LE
  [24..24 + 8·num_types]        type_keys: [u64]
  [next..next + 8·(num_types+1)]  offsets:  [u64]   (CSR)
  [next..next + 4·total_nodes]   nodes:    [u32]
  ```

  HashMap capacity is sized exactly from `num_types`, and each type's
  `Vec<NodeIndex>` is built from a contiguous u32 slice rather than
  bincode's per-field serde calls.

- **`id_indices.bin.zst` per-variant flat binary.** Replaces bincode
  `HashMap<String, TypeIdIndex>` with:

  ```
  [ 0.. 8]  magic     = b"KGLIIDX1"
  [ 8..12]  version   = u32 LE (= 1)
  [12..16]  num_types = u32 LE
  per-type block:
    [ 0.. 8]  type_key:    u64 LE
    [ 8.. 9]  variant_tag: u8  (0 = Integer, 1 = General)
    [ 9..16]  padding:     [u8; 7]
    [16..24]  num_entries: u64 LE
    payload:
      Integer (tag=0):  keys: [u32], node_idxs: [u32]
      General (tag=1):  blob_len: u64 + bincode HashMap<Value, NodeIndex>
  ```

  The Integer variant dominates Wikidata-style graphs (Q-number ids
  strip to u32), so the bulk of the 997 MB decompressed bincode blob
  collapses to two flat u32 arrays per type.

- **`interner.bin.zst` replaces `interner.json`.** The hash→string
  JSON map becomes a zstd-compressed bincode `Vec<String>` of just
  the originals; hashes are re-derived deterministically on load via
  `get_or_intern`. On the 81 GB graph this drops from ~7 MB JSON to
  ~3 MB bincode and eliminates one JSON parse on the critical path.

- **`KGLITE_LOAD_TIMING=1` stage instrumentation.** Gated per-stage
  wall-clock timing in `load_disk_dir`; off by default (zero
  overhead). Emits one `[TIMING] stage=<name> dur_ms=<ms>` line to
  stderr per major phase (`metadata_json`, `interner_load`,
  `disk_graph_load`, `type_indices_load`, `column_stores_load`,
  `id_indices_load`, `type_connectivity_load`). Used as the
  measurement harness for the four format changes.

**Backward compatibility.** All four loaders fall back to the old
format on a missing file or magic-byte mismatch:

- Missing `type_connectivity.bin.zst` → loader reads embedded JSON
  from `metadata.json`, then derives from `connection_type_metadata`.
- `type_indices.bin.zst` without `KGLTIDX1` magic → old bincode
  `HashMap<String, Vec<NodeIndex>>` path.
- `id_indices.bin.zst` without `KGLIIDX1` magic → old bincode
  `HashMap<String, TypeIdIndex>` path.
- Missing `interner.bin.zst` → old `interner.json` path.

Graphs saved by 0.8.11 and 0.8.12 continue to load without a rewrite.
Re-saving an old-format graph with 0.8.13 produces all four new files
automatically.

**Non-goals / out of scope.** No change to query execution, pattern
matcher, mutation paths, `node_mut_cache` / F1 / F2 flow, segmented
CSR layout (`seg_NNN/`), or the `KGLCOLv1` sidecar format. In-memory
representation of every touched structure is byte-identical to
0.8.12 — zero possibility of query-side regression.

## [0.8.12] — 2026-04-24

### Fixed

Seven disk-backend correctness fixes stacked on top of the 0.8.11
segmented-CSR foundation. Five cover latent save/reload regressions
that slipped through 0.8.11's phase-1–8 coverage; two (F1 + F2) close
the pre-existing Cypher `SET` and `DETACH DELETE` mutation holes on
disk graphs.

- **`save_disk` no longer compacts overflow away before seal.**
  The previous `dg.has_overflow()` gate unconditionally compacted
  before `save_to_dir`, which cleared `overflow_out`/`overflow_in`
  and made the phase-6 seal path see empty overflow. Every edge
  added between saves was silently lost on reload. Gating the
  compact on "won't take the seal path" (manifest empty OR no
  tail above the sealed watermark) preserves overflow for seal
  and keeps the compact-rewrite semantics for the non-seal case.

- **Segment-local seals merge `conn_type_index_sources` with global ids.**
  `write_conn_type_index` walks the segment's segment-local
  `out_offsets` (indices 0..tail_len) and stored those local
  indices as source ids. Reload's merge needed to shift each
  entry by `node_lo[seg]` for segment-local seals (full-range
  seals already store global ids). Without this, post-reload
  `MATCH (a)-[:T]->(b) RETURN a.id, b.id` returned no rows even
  though `count(*)` via the histogram reported the correct total.

- **Compact-rewrite after a prior seal cleans up stale `seg_NNN`.**
  When `save_to_dir` falls to compact-rewrite (tombstones, edits,
  or pure edge mutations between existing nodes), it now removes
  every `seg_NNN > 0` under the target dir before rewriting
  `seg_000`. Without this, `enumerate_segment_dirs` picked up the
  stale sealed segments on reload and concat'd them against the
  fresh seg_000, double-counting nodes and edges.

- **Compact-rewrite persists heap-backed core arrays.**
  `reconcile_seg0_csr` (called inside seal) replaces
  `self.{node_slots, out_offsets, in_offsets, edge_endpoints}`
  with heap-backed `MmapOrVec::Heap` copies. The same-dir
  compact-rewrite previously relied on mmap persistence and
  skipped explicit writes, which left the on-disk files at the
  pre-seal trimmed sizes; reload errored with "File too small".
  `save_to_file` is now called unconditionally for every core
  array — it handles both backings.

- **New types added via `add_nodes` persist on disk save.**
  `DirGraph::save_disk` gated the per-type
  `columns/<type>/columns.zst` sidecar write on the absence of
  `columns.bin`. For disk graphs built via `load_ntriples`,
  `columns.bin` is always present, so the sidecar branch was dead
  code and every type added after the initial build lost its
  column data on reload (properties read back as `None`). Save
  now reads `columns_meta.json`/`.bin.zst` to identify the types
  already covered by `columns.bin` and emits sidecars for the
  remainder. Load path additively walks `columns/` after the mmap
  fast-path to pick them up.

- **Cypher `SET n.prop = X` on disk-backed graphs persists through save + reload.**
  Pre-fix, `DiskGraph::node_weight_mut` materialised a `NodeData`
  into `self.node_arena`; the Cypher executor mutated that arena
  copy; `clear_arenas` dropped it without writing back to the
  canonical `ColumnStore`. Affected SET + save + reload was a silent
  no-op for persistence across 0.8.10 and 0.8.11.

  Fix: mirror the proven `batch.rs::flush_chunk` full-`Arc`
  replacement pattern for exact-row mutations.
  `DiskGraph::node_weight_mut` now stages writes in a
  `node_mut_cache` (Map-backed `NodeData`); `clear_arenas` groups
  cached entries by type, deep-clones each affected `ColumnStore`
  **once**, applies every staged title / property write + DELETE
  tombstone to the clone, and replaces both
  `DiskGraph.column_stores[ty]` and (via
  `DirGraph::sync_column_stores_from_disk`)
  `DirGraph.column_stores[ty]` atomically. Avoids the
  `Arc::make_mut` → per-row clone + Arc divergence that doomed the
  earlier attempt. Title writes diff against the current stored
  value before calling `set_title` so that `TypedColumn::Str`'s
  in-place-update offset-corruption bug (pre-existing) doesn't
  trigger on unchanged titles.

- **`DETACH DELETE` on disk preserves surviving nodes' property
  values across save + reload.** Pre-fix, a disk-graph delete cycle
  corrupted `title` reads (garbage bytes) and returned `None` for
  some `age` values on reload. The surviving count and id set were
  always correct — only the column-store-backed property columns
  were affected. Root cause was in the sidecar load path, not in
  mutation routing: `load_column_sidecars` derived `row_count` from
  `type_indices[type].len()` (live rows only), while the sidecar
  blob retains tombstoned rows alongside live ones. The mismatch
  made `ColumnStore::load_packed` walk column blobs at the wrong
  offsets and decode offset bytes as string data.

  Fix: the sidecar `columns.zst` file now starts with an 8-byte
  `KGLCOLv1` magic tag followed by `ColumnStore::row_count` (u32
  LE) before the existing `write_packed` payload, and the loader
  uses that stored count. Old-format sidecars (no magic tag) fall
  through to the `type_indices.len()` derivation for backward
  compat — best effort for legacy graphs, correct for any graph
  saved by 0.8.12+. Locked by
  `test_detach_delete_property_persistence_disk` (was `_xfail`).

### Deferred to 0.8.13+

- **Planner pruning using `SegmentManifest`.** Summaries are
  populated and persisted since 0.8.11; the pattern matcher
  doesn't yet consult them. Initial exploration showed the win
  is small under the current concat-at-load architecture (typed-
  edge queries at wiki1000m already run at sub-ms when the
  histogram path applies, and concat'd reads are uniform).
  Kept as a future option for 200-segment workloads once those
  exist in practice.

## [0.8.11] — 2026-04-23

### Added (disk-graph-improvement-plan PR1, phases 1–8)

This release lands the segmented-CSR foundation on the disk backend
and then fills in the incremental-save and auxiliary-index work on
top of it. Net result: write amplification on incremental ingest
drops from 5–25× to ~2× (target from
`dev-documentation/disk-graph-improvement-plan.md`) while load/save
on Wikidata-scale graphs now beats 0.8.10 across every subset. Every
existing `.kgl` directory still loads byte-for-byte identically.

- **Segment manifest (`seg_manifest.json`).** On-disk JSON listing
  per-segment `node_id_range`, `edge_count`, `conn_types`,
  `node_type_counts`, and `indexed_prop_ranges` summaries. Future
  planner pruning consults this before scanning. Today populated
  as a single-segment descriptor on every save. New module
  `src/graph/storage/disk/segment_summary.rs`.

- **Segmented CSR directory layout (`seg_NNN/`).** The CSR
  binaries, ColumnStore, and per-(type,prop) property indexes now
  live under a per-segment subdirectory. Graph-level metadata
  (`disk_graph_meta.json`, `seg_manifest.json`, DirGraph
  metadata) stays at the graph root. Gated by
  `csr_layout_version` (`#[serde(default)] == 0`) so legacy flat-
  layout `.kgl` directories still load.

- **`segment_subdir(id)` + `enumerate_segment_dirs(root)`.** The
  directory name is now id-parameterised, and load walks a sorted
  `seg_NNN/` enumeration instead of a hardcoded path.

- **Multi-segment read path.** `SegmentCsr` bundles one segment's
  six core CSR arrays (`node_slots`, `out_offsets`, `out_edges`,
  `in_offsets`, `in_edges`, `edge_endpoints`);
  `concat_segment_csrs` stitches them by shifting segment-local
  `edge_idx` onto combined `edge_endpoints`, concatenating
  per-segment `node_slots` and `edge_endpoints`, and welding the
  offset arrays. Single-segment load stays on the direct-mmap
  path for zero overhead vs 0.8.10.

- **Multi-segment write path (`DiskGraph::seal_to_new_segment`).**
  Flushes the still-mutable tail
  (`[sealed_nodes_bound, node_count)` + overflow edges between
  those nodes) to a fresh `seg_NNN/` — with per-segment
  `conn_type_index_*`, `peer_count_*`, and `edge_prop_*`
  alongside the core CSR — appends a `SegmentSummary` to the
  manifest, clears consumed overflow, advances the watermark,
  and rewrites `disk_graph_meta.json`.

- **Full-range-offset mode for cross-segment seal.** The seal
  path accepts overflow whose source or target is below the
  watermark by emitting offsets that span every global node id
  rather than only the new segment's tail. `concat_segment_csrs`
  distinguishes the two modes per segment via
  `out_offsets.len() > node_slots.len() + 1` and unions
  contributions per-node. Lets general incremental ingest (not
  just new-nodes-only batches) take the seal path.

- **Automatic incremental save.** `save_to_dir` now delegates
  to `seal_to_new_segment` whenever a graph has a prior segment
  manifest and a tail above the watermark — the typical
  incremental-ingest shape. Second save on a 10-chunk build
  produces 10 segments instead of rewriting the entire tree
  each time.

- **Per-segment auxiliary indexes survive seal+reload.**
  Multi-segment reload now merges `conn_type_index_*`,
  `peer_count_*`, and `edge_prop_*` across all segments so
  typed-edge matches, `edge_weight()`, and `peer_count`-backed
  aggregates return correct results on sealed segments.

### Performance

- **Save/load regression on Wikidata-scale graphs undone.**
  0.8.11's initial segmented-CSR work regressed `save`/`load`
  6–22× on wiki100m–wiki500m because the
  `dir.join("columns.bin").exists()` guard in `DirGraph` and
  `io::file` didn't know about the phase-4 `seg_000/`
  relocation. Checking both locations restores the 0.8.10
  baseline — and beats it once the rest of the phase work lands
  (wiki500m build −23 %, save −10 %, load −12 % vs 0.8.10;
  wiki100m build −22 %, save −16 %, load −15 %).

- **`MATCH ()-[:T]->(c) RETURN c, count(*)` aggregations
  sub-millisecond at every scale.** The fused MATCH+RETURN
  aggregate path was running
  `PatternExecutor::execute(MATCH (c))` unconditionally to
  enumerate the group target before dispatching to the
  histogram top-K fast path — for an untyped group target,
  a 14.7 M-node full-graph scan on wiki1000m that the fast
  path never reads. Enumeration is now deferred to the one
  node-centric fallback branch that actually needs it.
  `P31 class counts` on wiki1000m: 3702 ms → 0.3 ms (12 300×).
  Same fix applied to `FusedMatchWithAggregate`:
  `WITH P27 count` on wiki1000m 5387 ms → 13 ms (408×), on
  wiki500m 426 ms → 6 ms (71×).

- **Edge-centric fast path for
  `MATCH (src:T1)-[:T]->(tgt) WITH tgt, count(src)`.** Phase
  3 routes this shape through the pre-built
  `peer_count_histogram` when the source-type filter is a
  no-op, or walks `conn_type_index` source lists otherwise —
  both O(|T-sources|) instead of O(|all nodes| × in-degree).
  On wiki500m the query drops from 1210 ms to 445 ms (the
  result is a stepping stone; the full win lands via the
  histogram-routing fix above).

- **Multi-key ORDER BY LIMIT on aggregated counts.** Phase 4
  extends the fused MATCH+RETURN aggregate path so
  `ORDER BY k DESC, c.title` no longer disables fusion.
  Fusion sets a `candidate_emit` descriptor; the executor
  picks the primary-key threshold via a heap and emits the
  qualifying superset (any tie-breaking on secondary keys
  happens in the unchanged downstream `OrderBy + Limit`).
  `P31 class counts` warm-cache on wiki500m dropped from
  1477 ms to 450 ms as a secondary effect, before the
  group-target-scan fix made it sub-millisecond.

### Fixed

- **CSR mmap files now trim in-place on `save_to_dir`.**
  `MmapOrVec::mapped(path, cap)` has a 64-element minimum, so
  small graphs left trailing zeros on disk. The single-segment
  load path masked this by using `meta.*_len`, but the new
  multi-segment load path relies on file-size inference. All six
  core CSR arrays now pass through `save_to_file` on the
  same-dir save path, triggering the `file.set_len(byte_len)`
  truncation. Same bug pattern as the 0.8.10 conn_type_index trim.

## [0.8.10] — 2026-04-20

### Performance

- **GROUP BY aggregation defers property materialization.** Queries of the
  shape `RETURN x.prop, count(*)` now hash by NodeIndex during the per-row
  pass and resolve the property once per resulting group, rather than once
  per input row. For high-fanout aggregations on disk graphs (e.g., walking
  Wikidata's 439K country=Norway entities and grouping by their `instance_of`
  type) this drops O(rows) random-I/O column reads to O(distinct groups).
  Cypher semantics are preserved by re-bucketing on resolved values after —
  two distinct nodes that share a property value still collapse into one
  group. Implementation in `src/graph/languages/cypher/executor/return_clause.rs`.

### Fixed

- **OPTIONAL MATCH + RETURN with PropertyAccess group keys no longer silently
  returns NULL groups.** The fused OPTIONAL MATCH + aggregation path
  evaluated group-key expressions against the source row (pre-OPTIONAL),
  so a query like `OPTIONAL MATCH (p)-[:OWNS]->(pet) RETURN pet.name,
  count(*)` would resolve `pet.name` to NULL for all rows, collapsing every
  result into one wrong group. The fusion check now rejects PropertyAccess
  on variables only bound by the OPTIONAL MATCH itself, falling through to
  the correct (non-fused) aggregation path. `is_fusable_return_clause` in
  `src/graph/languages/cypher/planner/fusion.rs` now takes the OPTIONAL
  MATCH variable set and rejects matching property accesses.

- **Multi-MATCH re-bind no longer full-scans the graph.** When a second
  MATCH clause re-bound a variable from a prior clause
  (`MATCH (f {id: X}) MATCH (f)-[:R]->(c)`), the pattern matcher's
  inverted-index fast path ignored the existing binding and returned
  every source node for the edge type — 20s+ timeouts on Wikidata-scale
  graphs. The fast path now skips when the first node is already bound,
  falling through to `find_matching_nodes` which resolves the variable
  to a single node. `{Gjøa, Norway}` goes from >20s timeout to ~36ms
  on the 124M-node Wikidata graph.

### Changed

- **Graph algorithm procedures (`CALL pagerank/degree/betweenness/
  closeness/louvain/label_propagation/connected_components`) now error
  on timeout instead of silently returning partial results.** Algorithm
  signatures changed to `Result<_, String>`; the `break`-on-deadline
  branches now return `Err`, and the new `algorithm_timeout_err()`
  message points users at `timeout_ms=N` / `timeout_ms=0`. Fixes silent
  half-converged results that looked successful.
- **`CALL` on graphs over 2M nodes now refuses unscoped procedure
  runs up front.** Prior to this, `CALL degree()` on Wikidata (124M
  nodes) ignored its `_deadline` parameter entirely and ran for
  minutes — long enough to exhaust MCP transport timeouts and appear
  to wedge the server. The new guard errors in <1ms with "would scan
  the whole graph. Subgraph scoping is not yet supported — try a
  smaller graph, or pass timeout_ms=0 to override this guard."
- **`degree_centrality` and `weakly_connected_components` now honor
  the 20s Cypher deadline.** Both previously ignored deadlines (the
  former via an unused `_deadline` parameter, the latter by not
  accepting one). Periodic checks every ~1M edges.

## [0.8.9] — 2026-04-20

### Changed

- **Streaming label journal replaces in-memory `label_cache` during
  `load_ntriples`.** The previous `HashMap<u32, String>` grew to ~10GB
  on Wikidata's 124M entities, collapsing streaming throughput from
  1.8M triples/s to 450K/s via swap pressure. Labels now spill to a
  sequential on-disk journal (`{spill_dir}/labels.bin`) — zero heap
  growth during Phase 1. The post-Phase-1 rename pass reads the
  journal once, filtering to the ~tens-of-thousands of Q-numbers
  that actually appear as type names (~3MB final footprint). New
  module: `src/graph/io/ntriples/label_spill.rs`.

### Fixed

- **Typed `MATCH (n:Type {title: 'X'})` now takes the cross-type
  global-index fast path.** Previously only untyped patterns consulted
  the global index; typed patterns fell through to a full-type scan —
  10–14s (and frequent timeouts) on 13M-row types like Wikidata
  `human`. The matcher now consults the global index and post-filters
  by `node_type_of(idx)`, dropping `MATCH (n:human {title: 'Barack
  Obama'})` from 14s to ~25ms. Alias-aware (title↔label↔name).
- **Per-type `{nid: ...}` / `{qid: ...}` anchors hit the id index.**
  Both typed and untyped paths previously only checked the literal
  `"id"` key, so alias queries fell through to full scans. Now
  `id`/`nid`/`qid` all anchor via the same per-type id_index.
- **String-form id anchors (`{nid: 'Q76'}`) hit the id index.**
  `TypeIdIndex::get` now coerces `"Q76"` → `UniqueId(76)` by stripping
  the leading alpha prefix. Works for any `[A-Za-z]+[0-9]+` id
  scheme (Wikidata Q-codes, P-codes, E-codes, ...). Previously the
  lookup fell through to a full-type scan, so
  `MATCH (a:human {nid: 'Q76'})-[r]-(b:human {nid: 'Q13133'})`
  dropped from ~14s to ~300ms on Wikidata. Also fixes the
  correctness bug where `MATCH (a {id: 'Q76'})` silently returned
  `0` rows instead of the matching node.

## [0.8.8] — 2026-04-19

### Fixed

- **EXISTS inline-property filters on target nodes were silently
  dropped.** `WHERE EXISTS { (a)-[:REL]->({id: 20}) }` used the fast
  path's `get_property("id")` which missed the special id_column,
  producing silent zero-row results even when the pattern genuinely
  matched. Ported the same alias resolution that
  `node_matches_properties` uses — `title`/`name`/`label`/`id`/`type`
  all route to the right column via `resolve_alias`. The fast path
  now behaves identically to the slow path for these literal-property
  checks. Regression tests added to `test_where_exists.py`.

### Added

- **Cross-type global property index.** New `create_global_index(property)`
  builds a single mmap'd sorted-string index covering every live node,
  not just one type. On a disk graph, `save()` now auto-builds a global
  title index so `MATCH (n {title: 'X'})` — without a type label — is
  O(log N) out of the box. Solves the "title-to-ID without guessing
  the type" problem that agents hit repeatedly on Wikidata-scale graphs.
  Files: `global_index_{property}_{meta,keys,offsets,ids}.bin`.
- **`g.search(text, property='title', limit=10)` helper.** Returns
  the top-k nodes whose `property` matches `text` (exact match first,
  then prefix fallback) as `[{id, type, title, id_value}]`. Backed by
  the global index. Also exposed as a new MCP tool so agents can skip
  the "guess the type" ceremony entirely: `search('Equinor')` returns
  the right Q-number without `MATCH` or a type label.
- **Alias-aware cross-type lookups.** When the untyped matcher sees
  `{title: 'X'}` and the literal `title` index doesn't exist, it also
  tries the hardcoded title family (`title/label/name`) AND any
  per-type aliases declared via `node_title_field=` at `add_nodes`
  time. Same for `id/nid/qid`. An agent who built the index as
  `create_global_index('label')` still hits the fast path when
  querying `{title: 'X'}`, and vice versa. Derivation is automatic
  from the graph's existing schema — no new config API.

### Changed

- **`save_disk` auto-builds the global title index.** Every call to
  `save()` on a disk graph now produces `global_index_title_*.bin`
  files. Adds a one-pass sweep over `node_slots` at save time —
  negligible on small graphs, ~single-digit minutes on Wikidata-scale
  (124M nodes). Opt-out: delete the files after save.

## [0.8.7] — 2026-04-19

### Added

- **`WHERE n.prop STARTS WITH 'prefix'` now pushes down into the MATCH
  pattern** and routes through the persistent disk prefix index when
  available. New `PropertyMatcher::StartsWith(String)` variant, new
  `apply_prefix_to_patterns` helper in
  `src/graph/languages/cypher/planner/index_selection.rs`, new path in
  `matcher.rs::try_index_lookup` that calls
  `GraphRead::lookup_by_property_prefix`. String indexes are annotated
  `indexed="eq,prefix"` in `describe()` output (previously just `eq`);
  numeric indexes remain `indexed="eq"` only.
- **Deadline polling inside unanchored matcher scans.** Three hot
  loops in `matcher.rs` that used `.filter().collect()` over
  13M+-node type lists now poll the deadline every 4096 rows (via a
  new `check_scan_deadline()` helper with a structured hint message).
  Worst-case overshoot past the deadline drops from 20-60+ s to under
  a few ms. Other scan loops (variable-length paths, CSR edge
  counting, column stats) already polled; this closes the final gaps.
- **MCP `cypher_query` tool accepts `timeout_ms`.** `examples/mcp_server.py`'s
  tool signature now exposes the override so agents can deliberately
  extend or disable the deadline (`timeout_ms=0`) per call after an
  EXPLAIN confirms the plan is anchored. Previously the MCP agent was
  stuck with the backend-aware default.
- **`ResultView.diagnostics` — lightweight execution diagnostics.** Every
  `cypher()` call now attaches an always-on diagnostics dict to the
  returned `ResultView` with `elapsed_ms`, `timed_out`, and the
  `timeout_ms` that was in effect. Gives agents immediate feedback on
  query cost and timeout state without requiring `PROFILE`. The field
  is ``None`` for mutation paths, EXPLAIN, and transaction queries.
- **`describe()` indexed-property annotations.** Properties covered by
  an index (in-memory `property_indices` *or* the new persistent disk
  `PropertyIndex`) are now emitted with an `indexed="eq"` attribute in
  the `<properties>` detail block. A new `<indexing>` hint inside
  `<extensions>` explains the annotation and reminds agents to prefer
  anchored queries over unanchored scans on disk-backed graphs. New
  helper `DirGraph::has_any_index(node_type, property)` consolidates
  the "in-memory or persistent" check.
- **Persistent disk-backed property index.** `create_index('T', 'label')`
  on a `storage='disk'` graph now writes four mmap'd files
  (`property_index_{type}_{property}_{meta,keys,offsets,ids}.bin`)
  next to the CSR instead of rebuilding a `HashMap<Value, Vec<NodeIndex>>`
  on every `load()`. The previous in-memory path consumed ~1-3 GB of
  heap on 13M-row types and made `create_index` effectively unusable
  on Wikidata-scale disk graphs. The new persistent index is lazy-loaded
  on first query after reopen, keys are sorted lexicographically (so
  both equality and prefix can share the same structure), and the
  Cypher planner consults it via a new `GraphRead::lookup_by_property_eq`
  trait method. `MATCH (n {label: 'X'})` now hits the index on disk in
  O(log N + k). Supports string columns and title aliases
  (`node_title_field` at `add_nodes` time — `label`, `name`, etc.).
  Numeric equality and `STARTS WITH` pushdown are follow-ups. In-memory
  graphs are unchanged (keep the existing `property_indices` HashMap).
  The `create_index` return dict grows a `persistent: bool` field
  indicating whether the disk path was taken.
- **Cypher schema validation** at plan time — catches typos in
  pattern-literal property names (`MATCH (n:Person {agee: 30})`) before
  the executor commits to a scan. Returns a `Did you mean 'age'?` hint.
  Runs in O(clauses) against `node_type_metadata`; skipped when a graph
  has no declared schema. Pattern-literal properties are the only v1
  target — unknown node types, connection types, and WHERE/RETURN
  `n.prop` accesses deliberately pass through (existence-check queries
  and virtual columns would otherwise false-positive). Phase 3 will
  surface those as non-fatal diagnostics.

### Changed

- **`cypher()` default timeout is now backend-aware.** Disk-backed
  graphs default to 10 s, Mapped to 60 s, Memory to no deadline. Users
  can override per-call via `timeout_ms=N` or globally via
  `set_default_timeout(ms)`. The documented escape hatch
  `timeout_ms=0` disables the deadline entirely. Previously,
  disk-backed queries without an explicit `timeout_ms` ran until the
  harness killed them; the new default returns a structured timeout
  error after 10 s with hints pointing at anchoring / index usage.
  (Also applies to transaction-level `cypher()`.)
- **Cypher timeout error message now carries remediation hints.**
  Replaces the bare string `Query timed out` with guidance on
  anchoring queries, raising `timeout_ms`, or using the `timeout_ms=0`
  escape hatch.
- **`set_default_timeout(None)` behaviour updated.** Passing `None`
  now falls through to the backend-aware default rather than meaning
  "no timeout". Pass `0` for the old behaviour explicitly.

## [0.8.6] — 2026-04-19

### Performance

- **`describe(connections=['T'])` fast path on disk graphs.** Rewrote
  `write_connections_detail` to use the persisted `conn_type_index_*`
  inverted index instead of three full `edge_references()` sweeps. The
  previous path materialised every visited edge into a per-query
  `edge_arena` that was never cleared mid-call, growing VSZ linearly
  with scanned edges — on Wikidata (863 M edges) a single
  `describe(connections=['P31'])` call was SIGKILLed by the kernel
  after exhausting VM. The new path:
  - Reads pair counts from `type_connectivity_cache` when populated
    (zero edge I/O).
  - Skips the property-stats scan entirely when the connection type's
    metadata declares no edge properties.
  - Walks only matching edges via the inverted index, capped at two
    samples via an early-exit callback.
  - Measured on Wikidata (`wikidata_disk_graph_p12rebuild`, 122 M
    nodes, 863 M edges, cold page cache):
    - `describe(connections=['P170'])` (1.3 M edges): 108 s → **0.24 s**
      (~450× faster; previous in-flight code held VSZ at +27 GB
      after 90 s without completing).
    - `describe(connections=['P31'])` (122 M edges): **0.25 s** (was
      SIGKILLed by OOM killer before this change).
    - `describe(connections=True)` unchanged at ~0.15 s.

### Changed

- **`describe(connections=['T'])` pair-breakdown now capped at 50
  entries by default** (sorted by count desc), overridable via a new
  `max_pairs` keyword argument. Wide fan-out connection types like
  Wikidata's `P31` have tens of thousands of distinct
  `(src_type, tgt_type)` pairs — P31 alone has 191 k — which produced
  ~13 MB of XML that overshot typical MCP response budgets. The cap
  emits `<endpoints total="N" shown="…">` plus a trailing
  `<more pairs="…" edges="…"/>` marker so agents see both the
  dominant relationships and the exact size of the tail. P31 output
  drops 13 MB → ~4 KB by default; pass `max_pairs=500` (or similar)
  to drill into the full distribution on demand.

### Added

- `GraphBackend::for_each_edge_of_conn_type` — monomorphic closure
  iterator yielding `(src, tgt, edge_idx, properties)` per match. On
  disk uses the inverted index and never materialises `EdgeData`; on
  Memory/Mapped filters petgraph's resident `edge_references`. The
  callback returns `bool` so callers can stop after a bounded prefix.
- `DiskGraph::edge_properties_at(edge_idx)` — borrow an edge's
  property slice without going through the `materialize_edge` arena.
- `describe(..., max_pairs=<int>)` keyword argument — controls the
  pair-breakdown cap described above. `None` (default) resolves to 50.

## [0.8.5] — 2026-04-19

Internal: test coverage, SAFETY docs, storage module reorganization.

## [0.8.4] — 2026-04-19

### Performance

- **Correlated-equality pushdown in the Cypher planner.** `WHERE cur.prop =
  prior.other_prop` — where `prior` is a node bound by an earlier MATCH —
  now pushes onto the current MATCH's pattern as a new `EqualsNodeProp`
  matcher that the executor resolves per-row via the bound node's
  property. When the probe-side property is indexed, the pattern executor
  then picks an indexed lookup instead of scanning all nodes of that
  type. Also pushes `cur.prop = scalar_var` (where `scalar_var` is
  projected by a prior WITH/UNWIND) as `EqualsVar`. WHERE stays as a
  safety-net filter. Fallback: unchanged behavior when no index exists.
- **`add_connections(query=...)` now runs the planner.** Previously, the
  query path in `add_connections` went straight from parse → execute,
  skipping the entire planner — so no pushdowns (equality, IN,
  comparison), no spatial-join fusion, no LIMIT/DISTINCT pushdown. It
  now calls `cypher::optimize` like `g.cypher()` does. Combined with
  the correlated-equality pushdown, the Sodir prospect graph's derived
  connections (Phase 7) now build **~8.5× faster** — 29.6 s → 3.5 s:
  - 7a HC_IN_FORMATION (3 UNION ranks): 11.0 s → 1.2 s (~9×)
  - 7b/7c StructuralElement ENCLOSES: 0.6 s → 0.05 s (fuse_spatial_join
    now fires here)
  - 7d PLAY_HAS_FORMATION (primary + fallback): 17.3 s → 1.5 s (~11×)

## [0.8.3] — 2026-04-19

### Performance

- **Spatial-join operator for `MATCH (s:A), (w:B) WHERE contains(s, w)`.**
  A new planner pass (`fuse_spatial_join`) rewrites this two-pattern
  containment shape into `Clause::SpatialJoin`, bypassing the cartesian
  product. The executor builds an R-tree over the container side (via
  the new `rstar` dependency), iterates the probe side once, and emits
  only matching (container, probe) pairs — `O((N+M) log N + K)` rather
  than `O(N·M)`. Speedups on `tests/bench_spatial.py` (release build):
  - `contains 500K pairs` (500 polygons × 1 K points): 86.96 ms → 0.52 ms (**~167×**)
  - `contains 2.6M prospect_shape` (263 complex polygons × 10 K points):
    480.51 ms → 3.32 ms (**~145×**)
  - `contains 100K pairs`: 17.65 ms → 0.55 ms (~32×)
  - Complex polygons (50 vertices): 18.29 ms → 0.24 ms (~76×)

  Fires when both types have `SpatialConfig` (container needs `geometry`,
  probe needs `location`), the two patterns are disjoint typed nodes
  with no edges, and the WHERE is `contains(var, var)` optionally ANDed
  with a residual predicate. Other shapes (`NOT contains`, constant-point
  `contains(a, point(…))`, intra-pattern edges, three-plus patterns,
  disjunctions) fall back to the existing per-row fast path unchanged.

## [0.8.2] — 2026-04-19

### Changed

- **Blueprint loader rewritten in Rust.** `kglite.from_blueprint()` now runs
  entirely in a new `src/graph/blueprint/` module (schema + CSV reader +
  filter DSL + geometry + timeseries + build orchestrator). `pandas` is no
  longer touched during ingestion — CSVs are parsed with the `csv` crate
  straight into the internal columnar `DataFrame`, then handed to
  `mutation::maintain::add_nodes` / `add_connections`.
  - Every parallelisable phase is pipelined: CSV pre-parse, per-spec
    prep (filter + geometry + typed-column build), FK edge DataFrames,
    and junction edge DataFrames all run across threads via `rayon`.
    Only the graph mutation calls (`add_nodes` / `add_connections`)
    stay serial, because the graph is `&mut`. GeoJSON→WKT centroid
    extraction is also parallelised per row.
  - The Python shim (`kglite/blueprint/__init__.py`, ~60 lines) now
    only handles optional save + schema lock on top of the native
    build. The old 831-line `kglite/blueprint/loader.py` is deleted.
  - Sodir blueprint (564 K nodes, 759 K edges): **9.87 s → 1.6 s** (~6×).
  - Node / edge counts match the previous Python loader exactly (parity
    verified per-type across all 90 node types and all 93 edge types).
  - New runtime deps: `csv`, `geojson`, `indexmap` (the last so node /
    sub-node iteration preserves blueprint JSON order, which in turn
    keeps edge counts byte-identical to the old loader).
  - Set `KGLITE_BLUEPRINT_PROFILE=1` for a per-phase / per-sub-phase
    ms breakdown on stderr.

## [0.8.1] — 2026-04-19

### Changed

- **`code_tree` rewritten in Rust.** The polyglot codebase parser previously
  implemented in Python (`kglite/code_tree/*.py`, ~7,500 LOC) is now a
  first-class Rust module (`src/code_tree/`) exposed via PyO3. All eight
  language parsers (Python, Rust, TypeScript/JavaScript, Go, Java, C#, C,
  C++) plus the builder and manifest readers run natively. Tree-sitter
  grammars are bundled into the native extension — no optional dependency
  needed. **`pip install kglite[code-tree]` is no longer required**; the
  `[code-tree]` extras entry has been removed.
- **abi3 wheel — one wheel per platform, Python 3.10+.** PyO3's
  `abi3-py310` stable-ABI target is now enabled, collapsing the CI wheel
  matrix from 20 wheels (5 Python versions × 4 platforms) to 4. Users on
  any Python ≥ 3.10 install the same wheel.
- **Parallel parsing via rayon + thread-local parsers.** File-level
  parsing runs across CPU cores with one tree-sitter `Parser` per thread
  (via `thread_local!`) — no `Mutex` contention.
- **Parallel CALLS-edge tier resolution.** The 5-tier name-matching pass
  (84 K functions → 200 K edges) now runs in parallel via rayon; each
  function's edges are independent.
- **Aho-Corasick for USES_TYPE.** The multi-pattern type-name scan
  replaces a giant regex alternation with an Aho-Corasick automaton,
  yielding ~2.5× faster `USES_TYPE` edge building on Java-scale corpora.
- **End-to-end performance on real repos:**
  - duckdb C++ (2,805 files): **29 s (Python) → 0.63 s (Rust)** — ~46×
  - neo4j Java (7,966 files, 84 K functions): **crashed in Python →
    1.69 s in Rust**
  - KGLite mixed Py+Rust (248 files): 0.17 s
- **New `code_tree` module shape.** `kglite/code_tree/__init__.py` is a
  4-line shim importing from the native `kglite._kglite_code_tree`
  submodule. The previous Python modules under `kglite/code_tree/` have
  been removed.

### Fixed

- **`build()` no longer crashes on pure-Java repos** (e.g. `neo4j/neo4j`)
  with `Source type 'Struct' does not exist in graph`. Edge routing now
  picks source/target node types per-row from the graph schema rather
  than defaulting to hardcoded names.

## [0.8.0] — 2026-04-18

Internal-only storage-architecture refactor plus a handful of disk-mode
bug fixes and large performance wins. **No Python API signature changes.**
`kglite/__init__.pyi` signatures are byte-identical to v0.7.17 (`git diff
v0.7.17 HEAD -- kglite/__init__.pyi` contains only docstring additions).
Users upgrading from 0.7.x will see no behavioural differences other
than the fixes and performance gains listed below.

### Fixed

- **Concurrent `load_ntriples` calls no longer wipe each other's spill
  directories.** The previous cleanup logic deleted *all other*
  `kglite_build_*` directories in `/tmp` at every ingest start. Two
  `load_ntriples` calls running at the same time (e.g. a long Wikidata
  build and a small test suite) would kill each other's property-log
  files, causing the in-flight build to crash at Phase 1b with `No
  such file or directory`. The cleanup now only removes spill dirs
  whose contents haven't been modified in the last hour, so active
  builds are always safe.
- **`save()` on Wikidata-scale disk graphs no longer appears to hang.**
  On large disk graphs (e.g. 124 M nodes / 88 K types from N-Triples
  ingest), `save()` was iterating `self.column_stores` and writing
  each type's columnar data as a separate `columns/<type>/columns.zst`
  zstd file — a multi-hour serial loop, redundant because the v3
  single-file `columns.bin` (written during Phase 1b of the N-Triples
  builder) already contains everything the loader needs. `save_disk`
  now skips the per-type loop when `columns.bin` exists on disk, and
  reduces to the metadata flush it was always meant to be.
  **Measured on Wikidata (124 230 686 nodes, 862 810 243 edges,
  88 931 types): `save()` went from ≥ 60 min → 5.52 s.** In-memory
  graphs persisted to disk still write the per-type files as before
  (that path never produces `columns.bin`).
- **Disk-mode `add_nodes(conflict_handling="update")` now applies
  property updates.** Previously on disk graphs, re-inserting an
  existing node via `add_nodes(..., conflict_handling="update")`
  silently dropped the new values — `node_weight_mut` materialised
  `NodeData` into a per-query arena that `clear_arenas` discarded
  before the next read, so the mutation never reached
  `DiskGraph::column_stores` where reads happen. The batch-update path
  now mutates the per-type column store directly via `Arc::make_mut`
  and re-syncs with `sync_disk_column_stores` at the end of the chunk.
  Memory and mapped graphs are unaffected (they already worked).
- **Disk-mode `add_nodes(conflict_handling="replace")` now clears
  omitted properties.** Same root cause as above; Replace now nulls
  out every previously-set property on the row before writing the new
  set, matching the `PropertyStorage::replace_all` semantics of the
  heap backends.
- **Disk-mode `MERGE` edges are visible to subsequent `MATCH`
  queries.** `DiskGraph` used to default `defer_csr = true` so every
  `add_edge` on a fresh graph queued into `pending_edges`, which
  `edges_directed` never reads. One-off Cypher mutations now route
  directly to the overflow buffer (visible immediately); bulk loaders
  (`add_connections`, ntriples) still use the pending+rebuild path via
  `build_csr_from_pending`.

### Performance

- **Cypher query primitives faster across the board vs v0.7.17** (N=20
  trials, macOS dev box). Memory and mapped modes both win:
  - `pattern_match` at 10 k nodes: −60 %
  - `two_hop_10x`: −24 %
  - `describe()`: −21 % (memory), −24 % (mapped)
  - `pagerank`: −17 %
  - `find_20x`: −13 % (mapped)
  - Construction sweep (1 k / 10 k / 50 k nodes): −11 % to −22 %
  - No memory-mode query regressed above the +5 % gate; only four cells
    flagged under 5 % (`find_20x_memory` +4.7 %, `simple_filter` /
    `multi_predicate` minor noise).
- **N-Triples disk-graph build is 2.5 % faster on Wikidata.** Added
  `#[inline(always)]` on the hot `GraphBackend` → `GraphRead`/`GraphWrite`
  trampolines (`node_type_of`, `edge_endpoint_keys`, `edge_endpoints`,
  `node_weight`) and a new closure-based
  `GraphBackend::for_each_edge_endpoint_key` that bypasses the
  boxed-iterator virtual-dispatch on hot edge iteration. Phase 1b
  (columnar write) −86 s, Phase 2 (edge creation) −32 s, Phase 3
  (CSR build) −55 s on a 7.65 B-triple / 862.8 M-edge build. Total
  `load_ntriples`: 4747 s → 4627 s.
- **`rebuild_caches()` is 28 % faster on large disk graphs.** Two
  fixes: (a) `compute_type_connectivity` is now Rayon-parallel on the
  disk backend — shards the edge range across all cores and merges
  per-shard HashMaps serially, matching `build_peer_count_histogram`'s
  pattern; (b) removed a `madvise(DONT_NEED)` call at the end of
  `build_peer_count_histogram` that was evicting the 13.8 GB
  `edge_endpoints` from page cache right before
  `compute_type_connectivity` had to re-read it. Also reordered
  `rebuild_caches` to run `compute_type_connectivity` first so its
  sequential sweep warms the cache for the histogram builder. Measured
  on Wikidata (862.8 M edges): 235 s → 169 s. Memory and mapped modes
  unaffected (serial path retained).

### Changed

- **Deterministic `.kgl` v3 saves.** `save()` now produces byte-identical
  output for identical graphs regardless of per-process HashMap
  randomisation. `write_graph_v3` iterates `column_stores` in sorted
  order and canonicalises the metadata JSON (object keys sorted). Old
  `.kgl` files load unchanged — the format on the wire is a strict
  subset of the previous format's possible outputs. Enables byte-level
  golden-hash format-drift tests.
- **`ConnectionTypeInfo` serialises with sorted keys.** `source_types`
  and `target_types` (HashSet<String>) and `property_types`
  (HashMap<String, String>) now emit in lexicographic order, hardening
  the v3 golden-hash invariant for fixtures richer than single-element
  sets. Existing `.kgl` files load unchanged.

### Changed (internal, not user-visible)

- **Internal reorganization — `src/graph/` split into domain
  subdirectories.** Code previously flat in `src/graph/` now lives
  under `algorithms/`, `languages/cypher/`, `features/`,
  `introspection/`, `io/`, `mutation/`, `pyapi/`, `core/` (shared
  primitives, was `query/`), and `storage/`. `storage/` further splits
  into `memory/`, `mapped/`, and `disk/` per-backend folders. Pure
  file moves via `git mv` (rename similarity 97–100 %; git blame
  preserved). Filenames cleaned of redundant prefixes / suffixes
  (`pymethods_*` → `*`, `filtering_methods` → `filtering`, etc.). See
  `ARCHITECTURE.md` for the final layout.
- **Every `.rs` under `src/graph/` is now at or under the 2,500-line
  hard cap.** The Phase 9 split carved nine god files (12,144-line
  `executor.rs` down through the 2,610-line `pattern_matching.rs`)
  into themed submodules. `GOD_FILE_EXCEPTIONS` is empty;
  `test_god_file_gate` passes unconditionally.
- **`MappedGraph` promoted to a distinct struct** (was a type alias
  for `MemoryGraph` pre-Phase 5). Per-backend `impl GraphRead` /
  `impl GraphWrite` land in `src/graph/storage/impls.rs`, setting up
  future backend-specific optimizations without breaking callers.
- **`RecordingGraph<G>` ships as a Rust-only validation wrapper.**
  Generic over any `G: GraphRead`, logs every read-path method call.
  Used internally to prove the architecture is actually open/closed —
  adding a new backend is a 3-src-file change. Not exposed to Python.
  See `docs/adding-a-storage-backend.md` for the worked example.
- **Testing envelope hardened.** New parity tests cover zero-node /
  single-edge / 1 000-hop / Unicode / type-promotion / null-NaN /
  100 000-row cypher results across memory / mapped / disk
  (`tests/test_edge_cases_parity.py`). Golden-fixture regression suite
  (`tests/test_golden.py` + `tests/golden/`) pins byte-exact output
  for a deterministic 1 000-node / 3 000-edge graph across every
  storage mode. New `@pytest.mark.stress` tier for the 30 GB mapped
  bench and 10 k-hop traversal.
- **Unsafe-block hygiene.** All 40 `unsafe { ... }` blocks in `src/`
  carry `// SAFETY:` justifications. A module-level invariants block
  at the top of `src/graph/storage/mapped/mmap_vec.rs` documents the
  shared mmap safety contract.
- **Python API docstring clarification.** The `find()` docstring now
  warns that it searches only code-entity node types (`Function`,
  `Struct`, `Class`, `Enum`, `Trait`, `Protocol`, `Interface`,
  `Module`, `Constant`). The signature is unchanged.
- **Deprecated `TempDir::into_path()` calls** migrated to
  `TempDir::keep()` per tempfile 3.14+ API.
- **`pub type Graph = GraphBackend` alias dropped.** Every call site
  uses `GraphBackend` directly. Removes a hygiene wart flagged in the
  Phase 9 report-out.
- **RecordingGraph audit methods (`log`, `log_len`, `drain_log`) are
  now `#[cfg(test)]` rather than `#[allow(dead_code)]`.** Release
  builds no longer compile these helpers at all.

## [0.7.17] — 2026-04-17

### Added
- **Python 3.14 wheels**. CI test matrix and `build_wheels.yml` now cover
  3.14 across Linux/macOS (Intel + arm64)/Windows, alongside 3.10–3.13.
  Full test suite passes on 3.14 (1758 tests, same as 3.12 minus the
  optional `code-tree` tests that require tree-sitter wheels). pyo3 0.28
  (shipped in 0.7.16) enables this via `ABI3_MAX_MINOR = 14`.

## [0.7.16] — 2026-04-17

CI-fix release on top of 0.7.15. No functional changes to the Cypher
engine; dependency bumps + clippy 1.95 compatibility only.

### Dependencies
- **pyo3 0.27 → 0.28**, **geo 0.29 → 0.33**, **wkt 0.11 → 0.14**,
  **bzip2 0.5 → 0.6**. API changes absorbed: `#[pyclass(skip_from_py_object)]`
  on `KnowledgeGraph` (pyo3 0.28 opt-in); `Geodesic` is now a static value
  (call as `Geodesic.distance(...)` / `length(&Geodesic)`) with the
  `LengthMeasurable` trait imported from `geo::line_measures`.
- Clippy 1.95 compat: `sort_by` → `sort_by_key(Reverse)`, collapsed `if`/
  `match` guard patterns, `file_len.checked_div(elem_size)`, removed
  redundant `.into_iter()` in `IntoIterator` args.

## [0.7.15] — 2026-04-17

### Added
- **`WHERE n:Label` predicate**. Cypher now supports label checks as boolean
  predicates (not just MATCH-level filters). Composes with `AND`/`OR`/`NOT`
  and chained `n:A:B` form (`n:A AND n:B`). Example:
  `MATCH (n) WHERE n:Person OR n:Org RETURN count(n)`.
- **`Value::as_str() -> Option<&str>`**. Borrowing companion to the existing
  `as_string()`. Prefer when ownership is not required — avoids the per-call
  `String` clone.

### Changed
- **Function names lowercased at parse time** instead of per-row during
  dispatch. Every Cypher scalar/aggregate dispatch used to call
  `.to_lowercase()` on the function name each time it evaluated a row
  (21+ sites); names are now normalized once in `parse_function_call` and
  compared directly. Pure CPU win on function-heavy queries.
- **`count(DISTINCT n)` uses typed identity sets** — `HashSet<usize>` keyed
  on node/edge indices (with a `HashSet<Value>` fallback for non-binding
  expressions) instead of per-row `format!("n:{}", idx.index())` string
  formatting. ~20–26% faster on DISTINCT-count queries.
- **`substring()` skips intermediate `Vec<char>`** — uses `chars().skip(start)
  .take(len).collect()` instead of materializing the full char vector.
  ~10–18% faster on substring-heavy queries.
- **Zero-allocation property iterators**. `PropertyStorage::keys()` and
  `::iter()` return explicit `PropertyKeyIter` / `PropertyIter` enums instead
  of `Box<dyn Iterator>`. Saves one heap allocation per `keys(n)` /
  `RETURN n {.*}` / property-scan call. ~10% faster on `keys(n)` over
  all nodes.

### Fixed
- **`HAVING` with aggregate expressions**. `HAVING count(m) > 1` was silently
  returning zero rows when the RETURN item was aliased (`count(m) AS c`).
  Root cause: the aggregate function call fell through to per-row scalar
  dispatch, which errored with "Aggregate function cannot be used outside
  of RETURN/WITH", and the error was swallowed by `unwrap_or(false)`,
  dropping every row. Now `HAVING count(m)` and `HAVING c` both resolve
  to the pre-computed aggregate value regardless of aliasing. Unaliased,
  `DISTINCT`, and no-group-by forms all covered.
- **`rand()` / `random()` correctness under tight loops**. The previous
  SystemTime-per-call seeding could return identical values for adjacent
  rows when the system clock resolved two calls to the same nanosecond,
  and constant folding could collapse `rand()` to a single value for the
  whole query. Replaced with a thread-local xorshift64 PRNG, seeded once
  per thread with a splitmix64-avalanched counter (so parallel Rayon
  workers don't collide), and marked as row-dependent so it bypasses
  constant folding. Also uses the top 53 bits of state for full f64
  mantissa precision.

## [0.7.14] — 2026-04-17

### Added
- **Per-(conn_type, peer) edge-count histogram** as a persistent disk cache. Built once at CSR-build time (parallelised via Rayon, single sequential scan of `edge_endpoints.bin`), stored as three flat `peer_count_*.bin` files. Unanchored aggregate queries like `MATCH (a)-[:TYPE]->(b) RETURN b.title, count(a) ORDER BY cnt DESC LIMIT N` now return in ~ms instead of scanning the full 13 GB `edge_endpoints` array. Rebuildable on existing disk graphs via `g.rebuild_caches()` without a full graph rebuild.
- **`FusedCountAnchoredEdges` planner rule + executor**. `MATCH (var)-[r:TYPE?]->({id: V}) RETURN count(var)` (and the three symmetric variants) is now fused into O(log D) CSR offset arithmetic. The anchor is resolved to a `NodeIndex` at plan time via `graph.id_indices`. Combined with the tombstone short-circuit (below) this turns hub-node count queries (e.g. ~40 M incoming edges on Q5) from 100 s TIMEOUTs into sub-second lookups.
- **Tombstone-free short-circuit in `count_edges_filtered`**. When no nodes/edges have been removed and no peer-type filter is set, the function returns `end - start + overflow_count` directly after binary-searching for the connection-type range — skipping the per-edge tombstone check on hot hubs. Adds a `has_tombstones: bool` flag to `DiskGraph` and `DiskGraphMeta` (defaults to conservative `true` on legacy graphs so correctness is preserved; new builds flip it false).
- **Bounded `sources_for_conn_type`**. `DiskGraph::sources_for_conn_type_bounded(conn_type, max)` stops copying source node IDs after `max` entries, avoiding the ~400 MB eager heap allocation on cold-cache `LIMIT`-bounded pattern-matching queries. `pattern_matching.rs` now passes the `source_cap` through so e.g. `LIMIT 10` queries only read 1 000 sources from `conn_type_index_sources.bin` on first access.
- **`FusedCountTypedEdge` uses cached edge-type counts**. A one-liner that had been missed in v0.7.12: `MATCH (_)-[:TYPE]->(_) RETURN count(*)` now returns `edge_type_counts[TYPE]` in O(1) instead of scanning `edge_weights()` (64 s → sub-millisecond on Wikidata's 862 M edges).
- **`rebuild_caches` refreshes the peer-count histogram** on existing disk graphs, so users don't need to rebuild from scratch to get the v0.7.14 aggregate speedups.

### Fixed
- **DataFrame / blueprint disk builds now rebuild indexes at save time**. Previously, the first `add_connections` batch triggered a CSR build (via `ensure_disk_edges_built`) which wrote `conn_type_index` and `peer_count_histogram` reflecting only that first batch's edges. Subsequent batches added edges to overflow but never refreshed those indexes. Fix: `save_disk` now calls `compact()` once when overflow has accumulated, merging overflow back into CSR and rebuilding the indexes from all live edges. The per-batch `ensure_disk_edges_built` is now a no-op for overflow purposes (no O(E²) cost during multi-batch builds).
- **`lookup_peer_counts` returns `None` on type miss**. Previously returned `Some(empty_map)`, which blocked the caller from falling back to the sequential-scan path when the histogram was stale. Now returns `None` so callers see a clean cache miss.
- **Deadline checks in anchored-count paths**. `try_count_simple_pattern` / `count_edges_filtered` now accept an `Option<Instant>` deadline and check it every 1 M iterations. Closes the bypass that let `Q5_count_P31_incoming` run to 100 s past the 20 s default timeout.
- **Deadline check in `expand_var_length_fast` inner loop**. The outer queue loop was already checked every 512 pops, but the per-edge inner loop was unbounded — a single hub expansion could process 100 M+ edges without checking. Added an inner check every 1 M iterations.
- **Benchmark metric: Wikidata `unanchored_P31_count` now returns in 0.7 ms** (was 64 s with a wrong answer on cold deadline checks), `Q5_count_P31_incoming` 615 ms (was 100 s TIMEOUT), `Q5_incoming_all_count` 670 ms (was 20 s TIMEOUT), `cross_type_limited` 3 ms (was 2.5 s), `limit_10_P31` 10 ms cold-cache (was 2.6 s).

## [0.7.12] — 2026-04-16

### Added
- **Parallel Phase 3 CSR build**: The per-node `out_edges` sort-by-connection-type and the `conn_type_index` inverted-index build are now Rayon-parallelised. On a 124 M-node / 862 M-edge Wikidata build, this cuts combined CSR-build wall-clock on the parallelised portions from ~1000 s serial to ~100-200 s on 8+ P-cores. Build output is bit-identical to the serial version (index source lists are sorted post-reduce for determinism).
- **Deadline enforcement on long edge scans**: `count_edges_grouped_by_peer` (used by fused aggregate top-K and streaming HAVING paths) now accepts an optional deadline and checks it every 1 M edges. Pattern-matching's parallel expansion short-circuits when any thread detects a timeout. Together these stop unanchored aggregate queries from running unbounded past the default 20 s timeout.
- **Disk mode iterative updates**: Loaded disk graphs now support `add_connections()` — new edges go directly to overflow and are immediately visible to queries without CSR rebuild.
- **`compact()` method**: Merges overflow edges back into CSR arrays via full rebuild. Call after accumulating significant overflow (e.g., >10% of edges) to restore optimal query performance.
- **Connection-type inverted index for overflow**: `sources_for_conn_type()` now includes nodes with overflow edges, so Cypher queries on new edge types work immediately.
- **Partitioned CSR build parity**: Out-edges now sorted by connection type (enables binary search), and connection-type inverted index built for both CSR algorithms.

### Fixed
- **Streaming HAVING aggregate no longer OOMs**: `MATCH ...-[...]->(...) RETURN group_key, count(...) HAVING ... ORDER BY ...` (without LIMIT) used to materialise all edge rows before grouping — a 10 GB materialisation on Wikidata-scale graphs that triggered macOS OOM kill. The planner now fuses this shape into `FusedMatchReturnAggregate`; the executor's non-top-k path uses edge-centric `count_edges_grouped_by_peer` and applies HAVING post-aggregation on the small group-by map. On 16 GB hosts, queries that previously SIGKILL'd the Python process now return a clean "Query timed out" error.
- **Benchmark CSV preserved across crashes**: `bench/benchmark_wikidata_cypher.py` now streams each query's result to the CSV row-by-row with per-row flushes, instead of batching the write at the end. SIGKILL / OOM / Ctrl-C mid-run leaves every completed row on disk.
- **Mmap lifecycle during CSR build**: CSR build now writes to a temporary directory, then atomically swaps files into place. Fixes panics on large DataFrame builds where CSR output overwrote mmap'd files.
- **Overflow edges missing from `edges_directed`**: The `edges_directed_filtered_iter` iterator now correctly includes overflow edges (was passing `None` for overflow parameter).
- **Column store corruption on save→load→save cycle**: `write_packed()` now handles mmap-backed column stores (from loaded disk graphs) by materializing data from the MmapColumnStore. Also skips writing empty schema columns that duplicate id/title columns.
- **`defer_csr` not reset after CSR build**: After the first CSR build, `defer_csr` stayed `true`, causing all subsequent `add_edge()` calls to route to `pending_edges` instead of overflow. Each CSR rebuild then lost all previous edges. Fixed by setting `defer_csr = false` in `build_csr_from_pending()`.
- **`edge_weight_mut` for disk mode**: Implemented mutable edge property access for disk graphs, required by `add_connections` with duplicate edge handling (e.g., blueprint builds with temporal edge properties).
- **Disk graph `save_to_dir` missing metadata**: `disk_graph_meta.json` and conn_type_index files were only written to `data_dir`, not to `target_dir` when saving to a different directory. Fixed `save_to_dir` to write metadata to the target.
- **N-Triples mapped mode used compact IDs**: Mapped mode incorrectly used disk-style compact integer IDs instead of string IDs. Now matches memory mode behavior.
- **`enable_columnar()` title column mismatch**: Columnar nodes with missing titles in old stores got no title pushed, causing title column length < row_count. Save→load then failed with "blob too small". Fixed by always using node.title as fallback.
- **`edge_weight_mut` arena offset bug**: The flush logic assumed all edge_weight_mut entries were contiguous at the end of the arena, but read-only `edge_weight` calls interspersed between writes caused wrong offsets. Replaced arena-based tracking with a dedicated `edge_mut_cache` HashMap.
- **N-Triples mapped mode used compact edge path**: `use_compact` was true for mapped mode, sending it through `create_edges_compact()` instead of `create_edges_strings()`. Mapped now uses the memory-mode path for everything.
- **`InternedKey` hash is now deterministic across processes**: `InternedKey::from_str` previously used `DefaultHasher` (SipHash with a per-process random seed). Since `DiskNodeSlot.node_type` persists this as raw u64 on disk and the loader resolves it via the freshly-built interner's hashes, disk graphs built in one process couldn't be reliably loaded in another. Replaced with FNV-1a 64-bit (zero-alloc, zero new deps, deterministic). **Breaking change for existing disk graphs** saved with an older kglite: their `node_type` u64 values were hashed with a random SipHash seed and will not resolve against the new interner. Rebuild affected disk graphs.
- **Disk mode save/load loses embeddings, timeseries_store, and parent_types**: `save_disk` and `load_disk_dir` only persisted the `FileMetadata` struct, which didn't include `parent_types` and omitted embeddings/timeseries entirely. Describe() output on reloaded disk graphs was missing the "core vs supporting" tier split and `<embeddings>` section. Fix: added `parent_types` to `FileMetadata`, and save/load `embeddings.bin.zst` and `timeseries.bin.zst` alongside the other disk artifacts.
- **`describe()` non-deterministic across processes**: `compute_join_candidates` iterated `node_type_metadata` (HashMap) and broke `sort_by` ties with insertion order. Different HashMap `RandomState` seeds produced different candidate orderings, making checksums unstable. Property iteration is now sorted by name, and the candidate sort uses `(overlap desc, left_type, right_type, left_prop)` as a stable key.
- **Disk mode DataFrame/blueprint builds: wrong node titles/properties after multiple types**: `batch_operations` assigned `DiskNodeSlot.row_id` by slot index (set in `add_node`) instead of the per-type column store row returned by `push_row`. Pass 2 tried to fix this via `node_weight_mut`, but that call materializes into an arena that gets cleared on the next call — so the correction never persisted. Once a second node type was added, slot indices diverged from column store rows, causing `n.title`/`n.id` to read wrong rows (and `None` for out-of-bounds slots). Fix: batch_operations now also calls `DiskGraph::update_row_id` after each deferred assignment. Raises `api_benchmark.py` from 38/51 to 49/51 across all 3 modes.
- **Column store schema rebuild drops titles**: When batch_operations rebuilds a column store due to schema growth, titles for existing rows could be lost if `get_title()` returned None. Fixed by always pushing Null fallback.
- **`save_disk()` now persists `type_indices.bin.zst` and `id_indices.bin.zst`**: Previously only written by the N-Triples builder, DataFrame/blueprint-built disk graphs now also persist these files for correct and fast reload.
- **`write_packed()` preserves all schema columns**: Empty schema columns are now written with null padding instead of being skipped, ensuring lossless metadata round-trip through save→load cycles.

## [0.7.10] - 2026-04-16

### Added
- **Connection-type inverted index**: Built during CSR construction, maps edge types to source node IDs. Enables instant lookup of "which nodes have P31 outgoing edges" for unanchored edge queries. Cold-cache `MATCH (a)-[:P31]->(b) LIMIT 50` improved from 14.5s to 4.6s.
- **madvise hints for edge scans**: Sequential/DontNeed advisories on edge_endpoints during full-graph aggregation to reduce page cache pollution.

### Changed
- **FusedNodeScanTopK**: New fused clause for `MATCH (n:Type) RETURN n.prop ORDER BY n.prop LIMIT K` — single-pass scan with inline top-K selection, avoids materializing all rows. String sort keys supported.
- **Streaming top-K for FusedMatchReturnAggregate**: Iterates group nodes directly from type_indices instead of materializing all PatternMatch objects.
- **Edge-centric aggregation**: For untyped group nodes, scans edge_endpoints sequentially with HashMap accumulation instead of per-node iteration.
- **Lightweight peer iteration**: `expand_from_node` skips edge_endpoints reads when edge variable is unnamed (disk-only, reduces I/O by ~50%).

## [0.7.9] - 2026-04-16

### Changed
- **Zero-allocation edge counting**: `count()` queries on edge patterns use a new fast path that iterates CSR edges without materializing `EdgeData`. With sorted CSR, uses binary search to narrow to matching edge type. Result: "count instances of City" dropped from 2.3s to 37ms (63x faster).
- **WHERE-MATCH fusion**: The executor detects MATCH followed by WHERE and evaluates the WHERE predicate inline during pattern expansion. Non-matching rows are skipped immediately, and expansion stops after finding exactly LIMIT matching rows. Previously stuck queries (>10 min) now complete within timeout.
- **LIMIT push-down through WHERE**: Extended `push_limit_into_match` to handle `MATCH → WHERE → RETURN → LIMIT` pattern. The executor enforces exact LIMIT during fused WHERE evaluation.
- **Pre-computed edge type counts**: Edge type counts are computed during CSR build (zero overhead — counted inline during endpoint materialization). Persisted to metadata so `FusedCountEdgesByType` is O(1) on reload.

### Fixed
- **Wikidata type merge**: Q-code types (e.g., "Q5") now properly merge into human-readable labels ("human") during N-Triples build. Previously, when both "Q5" and "human" existed as types, the merge was skipped — now indices and column stores are merged correctly.
- **Column store key remapping**: Property log entries with old Q-code InternedKeys are remapped to merged label keys during Phase 1b, ensuring column stores have correct data after type merges.

## [0.7.8] - 2026-04-15

### Added
- **`set_default_timeout(timeout_ms)`**: Set a default per-query timeout (milliseconds) applied to all `cypher()` calls. Per-query `timeout_ms` overrides it.
- **`set_default_max_rows(max_rows)`**: Set a default cap on intermediate result rows. Queries exceeding this return an error with guidance to add LIMIT. Per-query `max_rows` overrides it.
- **`cypher(max_rows=N)`**: Per-query max rows limit parameter.

### Changed
- **Cypher LIMIT push-down**: Tightened source candidate cap from 10,000× to 100× the LIMIT value. Queries like `MATCH (n:Type)-[:EDGE]->(m) RETURN ... LIMIT 10` on large types are now ~100x faster (avoids allocating the full type index).
- **Cypher pattern start-node optimization**: Improved selectivity estimation for property-filtered nodes (equality filters now estimate /100 instead of /10). Lowered reversal threshold from 10× to 5×. Queries with filters on the target node (e.g., `WHERE b.prop = 'X'`) are now 2-3× faster.
- **DiskGraph edge iteration**: DiskEdges iterator now reads CSR edges lazily from the mmap instead of pre-collecting into a Vec. Eliminates O(degree) allocation per iterator — critical for high-degree nodes at Wikidata scale.
- **DiskGraph direct columnar property access**: Property checks in Cypher WHERE clauses and pattern matching now read individual column values directly from the ColumnStore on disk graphs, bypassing full `NodeData` materialization. Eliminates arena allocation and unnecessary id/title reads — ~3x fewer mmap reads per property check.
- **DiskGraph CSR sorted by connection type**: CSR edges are now sorted by `(node, connection_type)` during build. Edge-type filtering uses binary search instead of linear scan — O(log D + matching) instead of O(D) for high-degree nodes. Metadata flag `csr_sorted_by_type` ensures backward compatibility with older graphs.
- **Fused aggregation with WHERE clauses**: `FusedNodeScanAggregate` now activates for queries with property filters (e.g., `MATCH (n:Entity) WHERE n.pop > 1M RETURN n.continent, count(n)`). `FusedMatchReturnAggregate` now supports property filters on the unbound (counted) node. Both avoid materializing intermediate result rows.

## [0.7.7] - 2026-04-15

### Added
- **Schema locking**: `lock_schema()` / `unlock_schema()` enforce the graph's known schema on Cypher mutations (CREATE, SET, MERGE). Invalid writes return descriptive errors with "did you mean?" suggestions via edit-distance matching. Works on any graph — locks against `node_type_metadata` and `connection_type_metadata`.
- **`from_blueprint(lock_schema=True)`**: Convenience parameter to lock the schema immediately after blueprint loading.
- **`schema_locked` property**: Check whether the schema is currently locked.
- **`describe()` schema-locked notice**: When schema is locked, `describe()` includes a `<schema-locked>` element so agents know writes will be validated.

## [0.7.6] - 2026-04-12

### Fixed
- **Silent data loss on incremental save/load**: Loading a `.kgl` file, adding or updating nodes, then saving and loading again would silently lose properties for the new/updated nodes. The v3 column writer now always consolidates all node properties (Compact, Map, and Columnar) into column stores before writing.
- **Corrupt `.kgl` file on re-save**: Simply loading and re-saving a `.kgl` file (with no changes) could produce a corrupt file that failed to load with `blob too small for offsets`. The v3 column loader was building the ColumnStore schema from `node_type_metadata` (which includes id/title fields) instead of from the column section metadata (which only has property columns), creating empty placeholder columns that corrupted on write.
- **`enable_columnar()` dropped Columnar nodes on rebuild**: When rebuilding column stores, nodes already using Columnar storage were skipped, then their old stores were replaced — losing their properties. Now reads properties from old Columnar stores during rebuild and preserves mapped-mode id/title columns.

## [0.7.5] - 2026-04-10

### Added
- **`describe()` extreme-scale support**: Adaptive output for graphs with thousands or millions of types. Four scale tiers: Small (≤15 types, inline detail), Medium (16-200, compact listing), Large (201-5000, top-50 + search hint), Extreme (5001+, statistical summary).
- **`describe(type_search='...')`**: Find types by name with 1-layer neighborhood fan-out. Returns matching types with their connections plus connected types — enables domain discovery in a single call.
- **`rebuild_caches()`**: Force computation of type connectivity, edge type counts, and connection endpoint types in a single O(E) pass. Caches are persisted by `save()` and restored by `load()`.
- **Type connectivity cache**: Pre-computed type-level graph `(src_type, conn_type, tgt_type, count)` triples. Makes `type_search` and `describe(types=[...])` instant on any scale.
- **Lazy connectivity compute**: For Large/Extreme graphs, type connectivity is computed on first `type_search` call and cached for the session.

### Changed
- **`describe()` on Wikidata**: Output reduced from 2.9MB/508s to 3KB/0.15s. `type_search` with warm cache is sub-millisecond (was 2082s).
- **Performance guards**: Sampled neighbor schema for types >50K nodes, skip join candidates for >200 types, bounded error messages for large graphs.
- **Connection overview capping**: `describe(connections=True)` caps at 50 connection types for graphs with >500 connection types.
- **Empty endpoint resolution**: Disk-imported graphs with empty `connection_type_metadata` endpoints get source/target types resolved via bounded edge scan.
- **CSR build: in_edges merge sort**: Replaced scatter-write with merge sort for in_edges (1407s → 259s, 5.4× faster). Power-law target distributions caused page cache thrashing with scatter; merge sort uses only sequential I/O.
- **CSR build: zero-fill elimination**: `mapped_zeroed` creates mmap files at full size without writing zeros — OS lazy-fills pages on demand. Saves 13.8 GB of writes.
- **CSR build: edge_endpoints reuse**: Steps 3-4 read from edge_endpoints (written in step 1) instead of re-reading pending_edges. Eliminates one redundant 13.8 GB copy.
- **Wikidata build: 93 min → 73 min** (Phase 3 CSR: 1842s → 631s).
- **Cypher backtick type names**: `MATCH (n:\`programming language\`)` now works. Pattern parser re-adds backticks when reconstructing identifiers with spaces.
- **Q-code type resolution**: Post-Phase-1 pass resolves raw Q-code type names (e.g., Q13442814 → "scholarly article") using the complete label cache. Single sequential scan of node_slots.
- **NTriples type connectivity**: Type connectivity triples accumulated inline during edge creation, eliminating separate O(E) rebuild pass for freshly built graphs.
- **`get_edge_type_counts()` memory-safe**: Fallback path uses `edge_endpoint_keys()` (mmap reads) instead of `edge_weights()` (which materialized all EdgeData → OOM on disk graphs).

## [0.7.4] - 2026-04-08

### Changed
- **CsrEdge 16 → 8 bytes**: Removed `conn_type` from CSR edge records. Connection type stored only in `EdgeEndpoints`. Saves ~14 GB on Wikidata (out_edges + in_edges halved).
- **MergeSortEntry 24 → 12 bytes**: Removed `conn_type` from sort entries. 2× more edges per sort chunk during CSR build.
- **Edge conn_type pre-filter**: `DiskEdges` iterator checks `edge_endpoints` before `materialize_edge()`, skipping arena allocation and property HashMap lookup for non-matching edges.
- **Arena clearing at query boundaries**: `reset_arenas()` called at start of every Cypher execution. Prevents unbounded memory growth across queries (was the OOM cause on Wikidata).
- **`node_type_of()` — zero-materialization type check**: Reads directly from mmap'd `node_slots` (16-byte struct). Used in all Cypher executor fast paths and pattern matching hot loops instead of `node_weight()`.
- **Edge properties fast path**: `materialize_edge()` skips HashMap lookup when `edge_properties` is empty (common for Wikidata — 862M edges, zero properties).
- **Source node cap with LIMIT**: Multi-hop patterns with LIMIT N only allocate PatternMatch objects for `N × 10,000` source nodes instead of the full type.
- **`expand_from_node` limit propagation**: Edge expansion stops after collecting enough results instead of eagerly materializing all matching edges.
- **`id_indices` built on load**: Disk graphs build id_indices from column stores during load (no node materialization). Enables O(1) cross-type id lookup.
- **`lookup_by_id_normalized` trusts id_indices**: When id_indices exist for a type, the O(1) lookup result is trusted without falling through to linear scan.

### Added
- `DiskGraph::node_type_of()` — O(1) node type lookup from mmap'd node_slots.
- `DiskGraph::reset_arenas()` — public arena clearing for query boundary use.
- `DiskGraph::edges_directed_filtered_iter()` — pre-filtered edge iteration by connection type.
- `GraphBackend::node_type_of()`, `edges_directed_filtered()`, `reset_arenas()` — backend-agnostic wrappers.
- `DirGraph::build_id_index_from_columns()` — builds id_indices directly from mmap'd column stores without node materialization.
- `WHERE id(n) = X` pushdown in planner — converts `id()` function calls to inline `{id: X}` pattern properties.
- Cross-type id lookup in `find_matching_nodes` — untyped `{id: X}` patterns try all types via id_indices (O(types) × O(1)).
- `estimate_node_selectivity` returns 1 for any `{id: X}` pattern regardless of type.

### Fixed
- **Typed edge queries on disk graphs returning 0 rows**: `has_connection_type()` returned false when `connection_type_metadata` was empty (disk graphs skip O(types²) metadata). Fixed by falling back to interner check.
- **N-Triples build not registering connection type names**: Added lightweight connection type metadata registration (names only, no type×type matrix).

## [0.7.3] - 2026-04-08

### Changed
- **Single-file mmap column storage**: Column stores written to a single `columns.bin` file with mmap-backed reads. Replaces per-type `columns/<type>/columns.zst` layout. Near-instant load via mmap (no decompression).
- **Property log (disk mode)**: Phase 1 serializes properties to a zstd-compressed log file instead of building ColumnStores in-memory. Phase 1b replays the log to build columns in bulk — avoids O(n²) column rebuilds.
- **Partitioned CSR build**: Default CSR algorithm switched to hash-partitioned (Kuzu pattern). Merge-sort still available via `KGLITE_CSR_ALGO=merge_sort`.
- **File-backed pending edges**: `pending_edges` buffer uses mmap-backed `MmapOrVec` instead of heap `Vec`, avoiding ~14 GB heap allocation at Wikidata scale.
- **Auto-typing from P31**: N-Triples loader automatically derives node types from `P31` (instance-of) values, resolving Q-codes to labels. Entities without P31 default to "Entity".
- **Sparse property overflow**: Properties with <5% fill rate stored in a compact overflow bag instead of dense columns, reducing file size for wide schemas.

### Added
- `MmapColumnStore` — mmap-backed column reader for disk mode.
- `BuildColumnStore` — direct column writer that streams to the mmap file.
- `PropertyLogWriter`/`PropertyLogReader` — zstd-compressed property spill log for disk builds.
- `BlockPool`/`BlockColumn` — block-allocated typed column storage.
- `TypeBuildMeta` — per-type metadata for build-time column schema discovery.
- `MmapOrVec::load_mapped_region()`, `from_vec()`, `as_mut_bytes()` — new helpers for region-mapped and bulk byte access.
- `DiskGraph::update_row_id()` — fix per-type row_id mapping after column conversion.
- `ColumnStore::from_mmap_store()`, `from_raw_columns()` — constructors for mmap-backed and direct-built stores.
- `ColumnStore` id/title column accessors now work in disk mode.

### Fixed
- **code_tree stack overflow**: `extract_comment_annotations` switched from recursive to iterative traversal, fixing crashes on deeply nested ASTs.

## [0.7.2] - 2026-04-07

### Fixed
- **code_tree stack overflow**: `extract_comment_annotations` switched from recursive to iterative traversal, fixing crashes on deeply nested ASTs.

## [0.7.1] - 2026-04-06

### Changed
- **CSR build: external merge sort** (DuckDB-inspired). Replaced random-I/O scatter with external merge sort — sort chunks in memory, merge sequentially. All disk I/O is sequential. Phase 3 at Wikidata scale (862M edges, 16 GB RAM): ~16 min vs 90+ min previously.
- **Disk graph auto-persistence**: CSR arrays and metadata written directly to graph dir during build. No separate `save()` step needed. Mutations (`add_node`, `add_edge`, etc.) auto-flush metadata.
- **Disk graph raw storage**: Save/load uses raw `.bin` files (direct mmap) instead of zstd compression. Load is near-instant (mmap, no decompression). Legacy `.bin.zst` files still supported for loading.
- **Mmap-backed edge buffer**: N-Triples loader streams edges to mmap during Phase 1 (0 heap for edge buffer). Eliminates 13.8 GB heap allocation at Wikidata scale.

### Fixed
- **Memory leak in N-Triples loader**: `edge_buffer` (13.8 GB at Wikidata scale) was kept alive during Phase 3 CSR build, doubling peak memory. Now dropped immediately after Phase 2.
- **Disk thrashing during CSR build**: Random writes to mmap caused SSD thrashing. All writes are now sequential.
- **Temp file cleanup**: CSR build temp files cleaned up immediately after merge. Drop impl flushes metadata as safety net.

## [0.7.0] - 2026-04-05

### Added
- **Disk storage mode**: `KnowledgeGraph(storage="disk", path="./my_graph")` — fully disk-backed graph for very large datasets (100M+ nodes, 1B+ edges). Data lives on disk via mmap, using ~10% of equivalent in-memory RAM. The directory IS the graph — no separate save step needed.
- **GraphBackend abstraction**: Unified API across InMemory (petgraph), Mapped, and Disk backends. All Cypher queries, fluent API, and graph algorithms work identically across all three storage modes.
- **CSR edge storage**: Disk mode uses cache-friendly Compressed Sparse Row format. 3-4x faster than default on WHERE filters, SELECT, and SET operations at 100k scale.
- **zstd N-Triples support**: `load_ntriples()` now accepts `.nt.zst` files — 30x faster decompression than bz2.
- **`enable_disk_mode()`** method: Convert existing in-memory graph to disk-backed CSR.
- **`path` parameter** on constructor: Required for `storage="disk"`.

### Changed
- **Mapped mode**: Fixed O(n²) Arc clone bug — 50-300x faster `add_nodes` in mapped mode.
- **N-Triples loader**: 81x faster via bulk columnar conversion, pipeline parallelism, zero-copy parsing, byte-level filtering, and dense Vec edge lookup.

### Fixed
- Schema extension bug in mapped mode incremental `add_nodes`.
- `add_connections()` in disk mode auto-builds CSR so queries work immediately.

## [0.6.18] - 2026-03-30

### Fixed
- **Cypher LIMIT**: 16x faster multi-hop traversals with LIMIT. `MATCH (a)-[:R]->(b)-[:R]->(c) RETURN ... LIMIT 20` now pushes the limit into the pattern matcher — early termination at the last hop, overcommit budgets at intermediate hops. Benchmarks show parity with Neo4j on 2-hop queries.

## [0.6.17] - 2026-03-30

### Added
- `kglite.to_neo4j(graph, uri, ...)` — push graph data directly to a Neo4j database using batched UNWIND operations. Supports `clear`/`merge` modes, selection export, and verbose progress. Requires the `neo4j` package (`pip install neo4j` or `pip install kglite[neo4j]`).
- **ResultView**: Polars-style table display — `repr()` and `print()` now show a bordered table with column headers. Large results show first 10 + last 5 rows with `…` separator.
- **ResultView**: Improved `help(ResultView)` with quick-reference cheat sheet and examples on all methods.

### Fixed
- **code_tree**: Parse output (`Found N files`) now respects `verbose=False` — silent by default.

## [0.6.16] - 2026-03-30

### Changed
- **ResultView**: Polars-style table display — `repr()` and `print()` now show a bordered table with column headers instead of `ResultView(N rows, columns=[...])`. Large results show first 10 + last 5 rows with `…` separator.
- **ResultView**: Improved `help(ResultView)` with quick-reference cheat sheet and examples on all methods.
- **code_tree**: Parse output (`Found N files`) now respects `verbose=False` — silent by default.

## [0.6.15] - 2026-03-30

### Added
- `kglite.repo_tree(repo)` / `code_tree.repo_tree(repo)` — clone a GitHub repository and build a knowledge graph in one call. Cloned files are cleaned up by default; pass `clone_to=` to keep them locally. Supports private repos via `token=` or `GITHUB_TOKEN` env var.

### Fixed
- **code_tree**: Auto-create stub nodes for external base classes, enums, and traits referenced in EXTENDS, IMPLEMENTS, and HAS_METHOD edges — eliminates all "rows skipped: node not found" warnings during graph building.

## [0.6.12] - 2026-03-30

### Fixed
- **BUG-21**: Window functions (`row_number`, `rank`, `dense_rank`) crash with "Window function must appear in RETURN/WITH clause" when query has `WITH` aggregation + `ORDER BY` + `LIMIT`. The planner's `fuse_order_by_top_k` optimization now skips fusion when RETURN contains window functions.

### Changed
- Extracted window function execution into `window.rs` module (~240 lines out of executor.rs)
- Moved `is_aggregate_expression` / `is_window_expression` from executor.rs to ast.rs for cross-module reuse

## [0.6.11] - 2026-03-29

### Fixed

- **19 Cypher engine bugs resolved** — systematic fix of all bugs discovered via legal knowledge graph testing (BUG-01 through BUG-20, except BUG-04 which requires large-graph validation).

#### Critical — Silent wrong results
- **BUG-01**: Equality filter + GROUP BY no longer returns empty results. WHERE clause is now preserved after predicate pushdown to guarantee correctness when fusion fails.
- **BUG-02**: ORDER BY + LIMIT preserves integer types. `count()`, `size()`, `sum()` on integers no longer convert to float through the top-K heap path.
- **BUG-03**: HAVING clause is now propagated when the planner converts RETURN to WITH in fused optional-match aggregation.
- **BUG-05**: `RETURN *` expands to all bound variables (nodes, edges, paths, projected) instead of returning `{'*': 1}`.
- **BUG-06**: Path variable on explicit multi-hop patterns (`p = (a)-[]->(b)-[]->(c)`) now captures all intermediate nodes and relationships. `length(p)`, `nodes(p)`, `relationships(p)` return correct results.
- **BUG-17**: `MATCH (n) WHERE n.type = 'X'` on unlabeled nodes now works. Pattern matcher recognizes `type`/`node_type`/`label` as virtual properties.
- **BUG-18**: `labels()` returns consistent list format in both plain RETURN and GROUP BY contexts. Single-element list comparison (`labels(n) = 'Person'`) now works.

#### High — Errors on valid syntax
- **BUG-07**: `stDev()` / `stdev()` recognized as alias for `std()` aggregate function.
- **BUG-08**: `datetime('2024-03-15T10:30:00')` parses correctly instead of crashing on the time portion.
- **BUG-09**: `date()` returns null on invalid input (`''`, `'2016-00-00'`, `'2016-13-01'`) instead of crashing.
- **BUG-10**: `date('...').year`, `.month`, `.day` property access on function results now works.
- **BUG-11**: `[:TYPE1|TYPE2|TYPE3]` pipe syntax for multiple relationship types in MATCH patterns.
- **BUG-12**: `XOR` logical operator implemented with correct precedence (between OR and AND).
- **BUG-13**: `%` modulo operator implemented for both integer and float operands.
- **BUG-14**: `head()` and `last()` list functions implemented.
- **BUG-15**: `IN` operator accepts variable references, parameters, and function results — not just literal `[...]` lists.

#### Medium — Less common patterns
- **BUG-16**: Boolean/comparison expressions (`STARTS WITH`, `CONTAINS`, `>`, `=~`, etc.) work in RETURN/WITH clauses, evaluating to boolean values.
- **BUG-19**: `null = null` and `null <> null` return null (Cypher three-valued logic) instead of syntax error.
- **BUG-20**: Map all-properties projection `n {.*}` supported.

### Added

- **`Expression::PredicateExpr`** — AST variant bridging the expression/predicate boundary, enabling boolean predicates in RETURN/WITH items.
- **`Expression::ExprPropertyAccess`** — property access on arbitrary expression results (e.g. `date().year`).
- **`Expression::Modulo`** — modulo arithmetic operator.
- **`Predicate::Xor`** — exclusive-or logical operator.
- **`Predicate::InExpression`** — IN with runtime-evaluated list expressions.
- **`MapProjectionItem::AllProperties`** — wildcard map projection.
- **`EdgePattern.connection_types`** — multi-type edge matching for pipe syntax.
- **Performance benchmark suite** (`bench/benchmark_bugs.py`) — 70 targeted benchmarks covering all affected code paths, with CSV output for version-to-version comparison.

## [0.6.10] - 2026-03-29

### Fixed

- **Multi-MATCH empty propagation** — when the first MATCH in a multi-MATCH query returns 0 rows, subsequent MATCH/OPTIONAL MATCH clauses now correctly return 0 rows instead of matching against the entire graph.
- **Planner fusion guard** — MATCH fusion optimizations (FusedNodeScanAggregate, FusedMatchReturnAggregate, FusedMatchWithAggregate) are now restricted to first-clause position, preventing incorrect results when fused clauses ignored pipeline state from prior clauses.

### Changed

- **Retired legacy pytest/ test suite** — migrated unique test coverage (edge cases, subgraph extraction, pattern matching property filters, connection aggregation, connector API) into the official tests/ suite. Test count grew from 1,573 to 1,609.

## [0.6.9] - 2026-03-22

### Added

- **`'poincare'` distance metric** — new metric for `vector_search()`, `text_score()`, `compare()`, and `search_text()`. Computes hyperbolic distance in the Poincaré ball model, ideal for hierarchical data (taxonomies, ontologies). Based on Nickel & Kiela (2017).
- **`embedding_norm()` Cypher function** — returns the L2 norm of a node's embedding vector. In Poincaré embeddings, norm encodes hierarchy depth (0 = root/general, ~1 = leaf/specific).
- **Stored metric on embeddings** — `set_embeddings(..., metric='poincare')` stores the intended distance metric alongside vectors. Queries default to the stored metric when no explicit `metric=` is passed.

## [0.6.8] - 2026-03-19

### Added

- **`compare()` method** — dedicated API for spatial, semantic, and clustering operations. Replaces the overloaded `traverse(..., method=...)` pattern with a clearer `compare(target_type, method)` signature.
- **`collect_grouped()` method** — materialise nodes grouped by parent type as a dict. `collect()` now always returns a flat `ResultView`.
- **`Agg` helper class** — discoverable aggregation expression builders for `add_properties()`: `Agg.count()`, `Agg.sum(prop)`, `Agg.mean(prop)`, `Agg.min(prop)`, `Agg.max(prop)`, `Agg.std(prop)`, `Agg.collect(prop)`.
- **`Spatial` helper class** — spatial compute expression builders for `add_properties()`: `Spatial.distance()`, `Spatial.area()`, `Spatial.perimeter()`, `Spatial.centroid_lat()`, `Spatial.centroid_lon()`.
- **Traversal hierarchy guide** — new conceptual documentation explaining levels, property enrichment, and grouped collection.

### Breaking

- **`traverse()` no longer accepts `method=`** — use `compare(target_type, method)` instead.
- **`collect()` no longer accepts `parent_type`, `parent_info`, `flatten_single_parent`, or `indices`** — use `collect_grouped(group_by)` for grouped output. `collect()` always returns `ResultView`.

## [0.6.7] - 2026-03-18

### Performance

- **31% faster `.kgl` load** — large files are now memory-mapped directly instead of buffered read; small columns (< 256 KB) skip temp file creation and load into heap.
- **28% faster Cypher queries** — `PropertyStorage::get_value()` returns `Value` directly, avoiding `Cow` wrapping/unwrapping overhead on every property access.
- **Zero-alloc string column access** — `TypedColumn::get_str()` returns `&str` slices into mmap'd data without heap allocation, benefiting all WHERE string comparisons.
- **23% faster save** — reduced overhead from mmap threshold optimizations.

## [0.6.6] - 2026-03-18

### Breaking

- **`.kgl` format upgraded to v3** — files saved with older versions (v1/v2) cannot be loaded; rebuild the graph from source data and re-save.
- **`save_mmap()` and `kglite.load_mmap()` removed** — the v3 `.kgl` format replaces the mmap directory format with a single shareable file that supports larger-than-RAM loading.
- `save()` now leaves the graph in columnar mode after saving (previously restored non-columnar state). This avoids an expensive O(N×P) disable step.

### Added

- **v3 unified columnar file format** — `save()` now writes a single `.kgl` file with separated topology and per-type columnar sections (zstd-compressed). On load, column sections are decompressed to temp files and memory-mapped, keeping peak memory to topology + one type's data at a time.
- `save()` automatically enables columnar storage if not already active — no need to call `enable_columnar()` before saving.
- Loaded v3 files are always columnar (`is_columnar` returns `True`).

### Fixed

- **Temp directory leak** — `/tmp/kglite_v3_*` and `/tmp/kglite_spill_*` directories created during `load()` and `enable_columnar()` are now automatically cleaned up when the graph is dropped.
- Reduced save-side memory usage by eliminating double buffering in column packing.

### Removed

- `save_mmap(path)` method — use `save(path)` instead.
- `kglite.load_mmap(path)` function — use `kglite.load(path)` instead.
- v1 and v2 `.kgl` format support (load and save).
- Dead code: `StringInterner::len()`.

## [0.6.5] - 2026-03-18

### Added

- **Columnar property storage** — `enable_columnar()` / `disable_columnar()` convert node properties to per-type column stores, reducing memory usage for homogeneous typed columns (int64, float64, string, etc.). `is_columnar` property reports current storage mode.
- **Memory-mapped directory format** — `save_mmap(path)` / `kglite.load_mmap(path)` persist graphs as mmap-backed column files, enabling instant startup and out-of-core (larger-than-RAM) workloads. Directory layout: `manifest.json` + `topology.zst` + per-type column files.
- **Automatic memory-pressure spill** — `set_memory_limit(limit_bytes)` configures a heap-byte threshold; `enable_columnar()` automatically spills the largest column stores to disk when the limit is exceeded. `graph_info()` now reports `columnar_heap_bytes`, `columnar_is_mapped`, and `memory_limit`.
- **`unspill()`** — move mmap-backed columnar data back to heap memory (e.g., after deleting nodes to free space).
- **`memmap2` dependency** for memory-mapped file I/O.
- Columnar and mmap benchmarks in `test_bench_core.py` (5 new benchmarks).
- Comprehensive test suite for columnar storage and mmap format (28 new Python tests, 30+ new Rust tests).

### Fixed

- **`vacuum()` now rebuilds columnar stores** — previously, deleting nodes left orphaned rows in columnar storage that were never reclaimed. Now `vacuum()` (and auto-vacuum) automatically rebuilds column stores from only live nodes, eliminating the memory leak.
- `graph_info()` reports `columnar_total_rows` and `columnar_live_rows` for diagnosing columnar fragmentation.
- Boolean columns now correctly persist in mmap directory format (`from_type_str` now matches `"boolean"` in addition to `"bool"`).

### Performance

- 4-11x speedup for columnar/mmap operations: eliminated unnecessary full graph clone in `save_mmap()`, bulk memcpy in `materialize_to_heap()`, async flush, aligned pointer reads, direct push in `push_row()`, and skipped UTF-8 re-validation for string columns.

## [0.6.1] - 2026-03-08

### Changed

- **`describe()` default output now shows edge property names** in the `<connections>` section, improving agent discoverability of edge data without requiring `describe(connections=True)`.
- Improved hint text in describe output to guide agents toward `describe(connections=['CONN_TYPE'])` for edge property stats.
- `write_connections_overview` now reuses pre-computed metadata instead of scanning all edges (performance improvement).

## [0.6.0] - 2026-03-07

### Added

- **Python linting (ruff)** — format + lint enforcement for all Python files. `make lint` now checks both Rust and Python. `make fmt-py` auto-fixes.
- **Coverage reporting** — pytest-cov + Codecov integration in CI (informational, not blocking). `make cov` for local reports.
- **Stubtest** — `mypy.stubtest` verifies `.pyi` stubs match the compiled Rust extension. Runs in CI (py3.12). `make stubtest` for local checks.
- **Property-based testing** — Hypothesis tests for graph invariants (node count, filter correctness, index transparency, Cypher-fluent parity, delete consistency, sort correctness, type roundtrip).
- **Historical benchmark tracking** — pytest-benchmark with `github-action-benchmark` for performance regression detection. `make bench-save` / `make bench-compare` for local use.
- **Diátaxis documentation** — restructured docs into Tutorials, How-to Guides, Explanation, and Reference sections. New architecture and design-decisions explanation pages.
- **GitHub scaffolding** — issue templates (YAML forms), PR template, dependabot, security policy, `.editorconfig`, `.codecov.yml`.
- **PEP 561 `py.typed` marker** — type checkers now recognize KGLite's type stubs.
- **`connection_types` parameter** on `betweenness_centrality()`, `pagerank()`, `degree_centrality()` (stub fix — parameter existed at runtime).
- **`titles_only` parameter** on `connected_components()` (stub fix).
- **`timeout_ms` parameter** on `cypher()` (stub fix).

### Changed

- **Tree-sitter is now an optional dependency** — `pip install kglite[code-tree]` for codebase parsing. Core install reduced to just `pandas`.
- **README rewritten** as a keyword-optimized landing page for discoverability.
- Benchmarks CI job now runs on every push to main (was manual dispatch only).

## [0.5.88] - 2026-03-04

### Added

- **MCP Servers guide** — new docs page covering server setup, core tools, FORMAT CSV export, security, semantic search, and a minimal template

## [0.5.87] - 2026-03-04

### Added

- **`FORMAT CSV` Cypher clause** — append `FORMAT CSV` to any query to get results as a CSV string instead of a ResultView. Good for large data transfers and token-efficient output in MCP servers.

## [0.5.86] - 2026-03-03

### Added

- **`add_connections` query mode** — `add_connections(None, ..., query='MATCH ... RETURN ...')` creates edges from Cypher query results instead of a DataFrame. `extra_properties=` stamps static properties onto every edge.
- **`'sum'` conflict handling mode** — `conflict_handling='sum'` adds numeric edge properties on conflict (Int64+Int64, Float64+Float64, mixed promotes to Float64). Non-numeric properties overwrite like `'update'`. For nodes, `'sum'` behaves identically to `'update'`.

### Fixed

- **`add_connections` query-mode param validation** — `columns`, `skip_columns`, and `column_types` now raise `ValueError` in query mode (previously silently ignored)
- **`describe()` incomplete `add_connections` signature** — now shows `query`, `extra_properties`, `conflict_handling` params and query-mode example

## [0.5.84] - 2026-03-03

### Fixed

- **Cypher edge traversal without ORDER BY** — queries like `MATCH (a)-[r:REL]->(b) RETURN ... LIMIT N` returned wrong row count, NULL target/edge properties, and ignored LIMIT. Root cause: `push_limit_into_match` pushed LIMIT into the pattern executor for edge patterns, causing early termination before edge expansion. Now only pushes for node-only patterns.
- **`create_connections()` silently creating 0 edges** — two sub-bugs: (1) `ConnectionBatchProcessor.flush_chunk` used `find_edge()` which matches ANY edge type, so creating PERSON_AT edges would update existing WORKS_AT edges instead. Now uses type-aware `edges_connecting` lookup. (2) Parent map in `maintain_graph::create_connections` used `HashMap<NodeIndex, NodeIndex>` (single parent per child), losing multi-parent relationships. Now uses `Vec<NodeIndex>` per child and iterates group parents directly.
- **`describe(fluent=['loading'])` wrong parameter name** — documented `properties=` for `add_connections()`, actual parameter is `columns=`
- **`traverse()` with `method='contains'` ignoring `target_type=`** — when spatial method was specified, `target_type=` keyword was ignored and only the first positional arg was used as target type. Now prefers explicit `target_type=` over positional arg.
- **`geometry_contains_geometry` missing combinations** — added `(MultiPolygon, LineString)` and `(MultiPolygon, MultiPolygon)` match arms that previously fell through to `false`

## [0.5.83] - 2026-03-03

### Added

- **`fold_or_to_in` optimizer pass** — folds `WHERE n.x = 'A' OR n.x = 'B' OR n.x = 'C'` into `WHERE n.x IN ['A', 'B', 'C']` for pushdown and index acceleration
- **`InLiteralSet` AST node** — pre-evaluated literal IN with HashSet for O(1) membership testing instead of per-row list evaluation
- **TypeSchema-based fast property key discovery** — `to_df()`, `ResultView`, and `describe()` use TypeSchema for O(1) key lookup when all nodes share a type (>50 nodes)
- **Sampled property stats** — `describe()` and `properties()` sample large types (>1000 nodes) for faster response
- **`StringInterner::try_resolve()`** — fallible key resolution for TypeSchema-based paths
- **`rebuild_type_indices_and_compact` metadata fallback** — scans nodes to build TypeSchemas when metadata is empty (loaded from file)

### Fixed

- **FusedMatchReturnAggregate output columns** — built from return clause items instead of reusing pre-existing columns, fixing wrong column names in fused aggregation results
- **FusedMatchReturnAggregate top-k sort order** — removed erroneous `top.reverse()` calls that inverted DESC/ASC order in `ORDER BY ... LIMIT` queries
- **FusedMatchReturnAggregate zero-count rows** — exclude nodes with zero matching edges (MATCH semantics require at least one match)
- **Path binding variable lookup** — path assignments now find the correct variable-length edge variable instead of grabbing the first available binding
- **UNWIND null produces zero rows** — `UNWIND null AS x` now correctly produces no rows per Cypher spec instead of emitting a null row
- **InLiteralSet cross-type equality** — `WHERE n.id IN [1, 2, 3]` now matches float values via `values_equal` fallback
- **NULL = NULL returns false in WHERE** — implements Cypher three-valued logic where NULL comparisons are falsy; grouping/DISTINCT unaffected
- **Property push-down no longer overwrites** — `apply_property_to_patterns` uses `entry().or_insert()` to preserve earlier matchers
- **Pattern reversal skips path assignments** — `optimize_pattern_start_node` no longer reverses patterns bound to path variables
- **Fuse guard: HAVING clause** — `fuse_match_return_aggregate` bails out when HAVING is present
- **Fuse guard: vector score aggregation** — `fuse_vector_score_order_limit` bails out when return items contain aggregate functions
- **Fuse guard: bidirectional edge count** — `fuse_count_short_circuits` skips undirected patterns that could produce wrong counts
- **Fuse guard: dead SKIP check removed** — `fuse_order_by_top_k` no longer checks wrong clause index for SKIP
- **Parallel expansion error propagation** — errors in rayon parallel edge expansion are now propagated instead of silently returning empty results
- **Variable-length paths min_hops=0** — source node is now yielded at depth 0 when `min_hops=0` (e.g., `[*0..2]`)
- **Parallel distinct target dedup** — parallel expansion path now applies `distinct_target_var` deduplication matching the serial path
- **Unterminated string/backtick detection** — tokenizer now returns errors for unclosed string literals and backtick identifiers
- **String reconstruction preserves escapes** — `CypherToken::StringLit` re-escapes quotes and backslashes during reconstruction

## [0.5.82] - 2026-03-03

### Changed

- **zstd compression for save/load** — replaced gzip level 3 with zstd level 1; Save 9.5s → 1.1s (8.6×), Load 2.3s → 1.0s (2.2×), file 7% smaller. Backward-compatible: old gzip files load transparently
- **Vectorized pandas series extraction** — `convert_pandas_series()` uses `series.tolist()` + `PyList.get_item()` instead of per-cell `Series.get_item()`, plus batch `extract::<Vec<Option<T>>>()` for Float64/Boolean/String. Build 24.7s → 19.3s
- **Fast lookup constructors** — `TypeLookup::from_id_indices()` and `CombinedTypeLookup::from_id_indices()` reuse pre-built `DirGraph.id_indices` instead of scanning all nodes
- **Skip edge existence check on initial load** — `ConnectionBatchProcessor.skip_existence_check` flag bypasses `find_edge()` when no edges of that type exist yet
- **Pre-interned property keys** — intern column name strings once before the row loop, use `Vec<(InternedKey, Value)>` instead of per-row `HashMap<String, Value>` for node creation
- **Single-pass load finalize** — `rebuild_type_indices_and_compact()` combines type index rebuild + Map→Compact property conversion in one pass, with TypeSchemas built from metadata instead of scanning nodes
- **Zero-alloc InternedKey deserialization** — custom serde Visitor hashes borrowed `&str` from the decompressed buffer, eliminating ~5.6M String allocations per load
- **Remove unnecessary `.copy()` on first CSV read** in blueprint loader

## [0.5.81] - 2026-03-02

### Added

- **Comparison pushdown into MATCH** — `WHERE n.prop > val` (and `>=`, `<`, `<=`) is now pushed from WHERE into MATCH patterns, filtering during node scan instead of post-expansion. Includes range merging (`year >= 2015 AND year <= 2022` → single Range matcher). Benchmark: filtered 2-hop query 109ms → 14ms (7.6×), property filter 2.5ms → 0.8ms (3×)
- **Range index acceleration** — pushed comparisons now use `create_range_index()` B-Tree indexes via `lookup_range()` for O(log N + k) scans instead of O(N) type scans
- **Reverse fused aggregation** — `MATCH (:A)-[:REL]->(b:B) RETURN b.prop, count(*)` (group by target node) now fuses into a single pass like source-node grouping. In-degree benchmark: 26ms → 9ms (2.8×)
- **EXISTS/NOT EXISTS fast path** — direct edge-existence check for simple EXISTS patterns instead of instantiating a full PatternExecutor per row. NOT EXISTS: 2372ms → 0.3ms (7400×)
- **FusedMatchReturnAggregate top-k** — BinaryHeap-based top-k selection during edge counting, avoiding full materialization + sort. In-degree top-20: 10.5ms → 5.0ms
- **FusedOrderByTopK external sort expression** — ORDER BY on expressions not in RETURN items now fuses into the top-k heap, projecting only surviving rows. UNION per-arm: 5.4ms → 2.3ms
- **FusedNodeScanAggregate** — single-pass node scan with inline accumulators (count/sum/avg/min/max) for `MATCH (n:Type) RETURN group_keys, aggs(...)`, avoiding intermediate ResultRow creation
- **FusedMatchWithAggregate** — fuse `MATCH...WITH count()` into single pass (same as MATCH+RETURN fusion but for pipeline continuation)
- **DISTINCT push-down into MATCH** — when `RETURN DISTINCT` references a single node variable, pre-deduplicate by NodeIndex during pattern matching. Includes intermediate-hop dedup for anonymous nodes. Filtered 2-hop DISTINCT: 15ms → 10ms
- **UNION hash-based dedup** — replace `HashSet<Vec<Value>>` with hash-of-values approach for UNION (non-ALL) deduplication
- **35-query DuckDB/SQLite comparison benchmark** (`bench_graph_traversal.py`)

## [0.5.80] - 2026-03-02

### Added

- **`closeness_centrality(sample_size=…)`** — stride-based node sampling for closeness centrality, matching the existing betweenness pattern; reduces O(N²) to O(k×(N+E)) for approximate results on large graphs
- **`copy()` / `__copy__` / `__deepcopy__`** — deep-copy a `KnowledgeGraph` in memory without disk I/O, useful for running mutations on an independent copy

### Changed

- **`compute_property_stats` value-set cap** — stop cloning values into the uniqueness `HashSet` once `max_values+1` entries are collected, avoiding O(N) clones for high-cardinality properties
- **Closeness centrality Cypher `CALL`** — `CALL closeness({sample_size: 100})` now supported alongside `normalized` and `connection_types`
- **Regex cache in fluent filtering** — pre-compile `Regex` patterns before filter loops (was compiling per-node); `fluent_where_regex` 302 ms → 1 ms
- **Single-pass property stats** — replaced O(N×P) two-pass scan with O(N×avg_props) single-pass accumulator
- **Pre-computed neighbor schemas** — `describe()` scans all edges once instead of per-type

## [0.5.79] - 2026-03-02

### Added

- **Window functions** — `row_number()`, `rank()`, `dense_rank()` with `OVER (PARTITION BY ... ORDER BY ...)` syntax for ranking within result partitions
- **HAVING clause** — post-aggregation filtering on RETURN and WITH (`RETURN n.type, count(*) AS cnt HAVING cnt > 5`)
- **Date arithmetic** — DateTime ± Int64 (add/subtract days), DateTime − DateTime (days between), `date_diff()` function
- **Window function performance** — pre-computed column names, constant folding, OVER spec deduplication, rayon parallelism, fast path for unpartitioned windows

## [0.5.78] - 2026-03-02

### Changed

- **Betweenness BFS inner loop** — merged redundant `dist[w_idx]` loads into cached `if/else if` branch, eliminating a second memory access per edge in both parallel and sequential paths
- **Pre-intern connection types in algorithms** — betweenness, pagerank, degree, closeness, louvain, and label propagation now pre-intern connection type filters once per call instead of hashing per-edge
- **Adjacency list dedup** — undirected adjacency lists are now sorted and deduplicated to prevent double-counting from bidirectional edges (A→B + B→A)
- **3-way traversal benchmark** — added DuckDB (columnar/vectorized) alongside SQLite and KGLite with optimized batch queries

## [0.5.77] - 2026-03-02

### Changed

- **Edge data optimization** — `EdgeData.connection_type` changed from `String` (24 bytes) to `InternedKey` (8 bytes), reducing per-edge overhead by 16 bytes
- **Edge properties compacted** — `EdgeData.properties` changed from `HashMap<InternedKey, Value>` (48 bytes) to `Vec<(InternedKey, Value)>` (24 bytes), saving 24 bytes per edge
- **BFS connection type comparison** — pre-intern connection type before edge loops for `u64 == u64` comparison instead of string equality
- **Static slice in BFS** — `expand_from_node` changed `vec![Direction]` heap allocation to `&[Direction]` static slice
- **Save/load performance** — save time -70% (2,253 → 682 ms), load time -93% (1,676 → 119 ms) on 50k node / 150k edge benchmark
- **Deep traversal speedup** — 8-20 hop citation queries 16-28% faster from interned comparison and eliminated heap allocations

## [0.5.76] - 2026-03-01

### Changed

- **BFS traversal optimization** — replaced `HashSet` visited set with `Vec<bool>` for cache-friendly O(1) lookups during variable-length path expansion
- **Skip redundant node type checks** — planner now marks edges where the connection type guarantees the target node type, avoiding unnecessary `node_weight()` loads during BFS
- **Skip edge data cloning** — unnamed edge variables no longer clone `connection_type` and `properties`, eliminating thousands of heap allocations per traversal
- **DISTINCT dedup optimization** — uses `Value` hash keys instead of `format_value_compact()` string allocation per row

### Added

- **Graph traversal benchmark suite** — SQLite recursive CTE vs KGLite across 15 query types (citation chains, shortest path, reachability, triangles, neighborhood aggregation)

## [0.5.75] - 2026-03-01

### Added

- **`keys()` function** — `keys(n)` / `keys(r)` returns property names of nodes and relationships as a JSON list
- **Math functions** — `log`/`ln`, `log10`, `exp`, `pow`/`power`, `pi`, `rand`/`random` (previously documented but not implemented)
- **`datetime()` alias** — `datetime('2020-01-15')` works identically to `date()`
- **DateTime property accessors** — `d.year`, `d.month`, `d.day` on DateTime values (via WITH alias)
- **Scientific notation** — tokenizer now parses `1e6`, `1.5e-3`, `2E+10` as float literals

### Fixed

- **String function auto-coercion** — `substring`, `left`, `right`, `split`, `replace`, `trim`, `reverse` now auto-coerce DateTime/numeric/boolean values to strings instead of returning NULL
- **`describe()` algorithm hint** — fixed misleading `YIELD node, score|community|cluster` that didn't mention `component`; now shows which yield name belongs to which procedure
- **Spatial coordinate order note** — added documentation clarifying WKT uses (longitude latitude) while `point()` uses (latitude, longitude)

## [0.5.74] - 2026-03-01

### Added

- **Multi-hop traversal benchmarks** — scale-free graph benchmarks at 1K/10K/50K/100K nodes with hop depths 1–8, comparable to TuringDB/Neo4j multi-hop benchmarks
- **Blueprint documentation** — standalone guide page with step-by-step walkthrough, real CSV examples, and troubleshooting

### Changed

- **Variable-length path BFS** — global dedup mode skips path tracking when path info isn't needed (no `p = ...` assignment, no named edge variable), reducing memory and redundant exploration (~4x faster)
- **WHERE IN predicate pushdown** — `WHERE n.id IN [list]` is now pushed into the MATCH pattern and resolved via id-index O(1) lookups instead of post-filtering all nodes (~1,400x faster on 10K 8-hop traversals)

## [0.5.73] - 2026-02-27

### Changed

- **README** — added blueprint loading and code review examples to Quick Start, doc links on each section
- **CLAUDE.md** — simplified and consolidated conventions

## [0.5.72] - 2026-02-27

### Added

- **Documentation site** — Sphinx + Furo docs with auto-generated API reference from `.pyi` stubs, hosted on Read the Docs. Guide pages for Cypher, data loading, querying, semantic search, spatial, timeseries, graph algorithms, import/export, AI agents, and code tree.

## [0.5.71] - 2026-02-27

### Added

- **`traverse()` API improvements:**
  - `target_type` parameter — filter targets to specific node type(s): `traverse('OF_FIELD', direction='incoming', target_type='ProductionProfile')` or `target_type=['ProductionProfile', 'FieldReserves']`
  - `where` parameter — alias for `filter_target`, consistent with the fluent API: `traverse('HAS_LICENSEE', where={'title': 'Equinor'})`
  - `where_connection` parameter — alias for `filter_connection`: `traverse('RATED', where_connection={'score': {'>': 4}})`
  - `help(g.traverse)` now shows a comprehensive docstring with args, examples, and usage patterns
- **Temporal awareness** — first-class support for time-dependent nodes and connections:
  - Declare temporal columns via `column_types={"fldLicenseeFrom": "validFrom", "fldLicenseeTo": "validTo"}` on `add_nodes()` or `add_connections()` — auto-configures temporal filtering behind the scenes (same pattern as spatial `"geometry"` / `"location.lat"`)
  - `date("2013")` sets a temporal context for the entire chain — all subsequent `select()` and `traverse()` calls filter to that date instead of today
  - `date("2010", "2015")` — range mode: include everything valid at any point during the period (overlap check)
  - `date("all")` — disable temporal filtering entirely (show all records regardless of validity dates)
  - `select()` auto-filters temporal nodes to "currently valid" (or the `date()` context). Pass `temporal=False` to include all historic records
  - `traverse()` auto-filters temporal connections to "currently valid". Override with `at="2015"`, `during=("2010", "2020")`, or `temporal=False`
  - `valid_at()` / `valid_during()` auto-detect field names from temporal config; NULL `date_to` treated as "still active"
  - Display (`sample()`, `collect()`) filters connection summaries to temporally valid edges
  - `describe()` includes `temporal_from`/`temporal_to` attributes on configured types and connections
  - Blueprint loader: use `"validFrom"` / `"validTo"` property types to auto-configure temporal filtering
  - `set_temporal(type_name, valid_from, valid_to)` available as low-level API for manual configuration
  - Temporal configs persist through `save()`/`load()` round-trips
- **`show(columns, limit=200)`** — compact display of selected nodes with chosen properties. Single-level shows `Type(val1, val2)` per line; after `traverse()` walks the full chain as `Type1(vals) -> Type2(vals) -> Type3(vals)`. Resolves field aliases and truncates long values

## [0.5.70] - 2026-02-26

### Added

- **`to_str(limit=50)`** — format current selection as a human-readable string with `[Type] title (id: x)` headers and indented properties
- **`print(ResultView)` smart formatting** — `ResultView.__str__` uses multiline card format (properties + connection arrows) for ≤3 rows, compact one-liner for >3. Connections show direction with `◆` as the current node: `◆ --WORKS_AT--> Company(id, title)` for outgoing, `Person(id, title) --WORKS_AT--> ◆` for incoming. Long values (WKT geometries, etc.) are truncated with middle ellipsis
- **`sample()` selection-aware** — `sample()` now works on the current selection (`graph.select('Person').sample(3)`) in addition to the existing `sample('Person', 3)` form
- **`head()`/`tail()` preserve connections** — slicing a ResultView carries connection summaries through

## [0.5.67] - 2026-02-26

### Changed

- **BREAKING: Fluent API method renames** — modernized the fluent API surface to match common query DSL conventions:
  - `type_filter()` → `select()`
  - `filter()` → `where()`
  - `filter_any()` → `where_any()`
  - `filter_orphans()` → `where_orphans()`
  - `has_connection()` → `where_connected()`
  - `max_nodes()` → `limit()`
  - `get_nodes()` → `collect()`
  - `node_count()` → `len()` (also adds `__len__` for `len(graph)`)
  - `id_values()` → `ids()`
  - `max_nodes=` parameter → `limit=` everywhere (select, where, traverse, collect, etc.)
- **BREAKING: Retrieval method renames** — dropped inconsistent `get_` prefix and shortened verbose methods:
  - `get_titles()` → `titles()`
  - `get_connections()` → `connections()`
  - `get_degrees()` → `degrees()`
  - `get_bounds()` → `bounds()`
  - `get_centroid()` → `centroid()`
  - `get_selection()` → `selection()`
  - `get_schema()` → `schema_text()`
  - `get_schema_definition()` → `schema_definition()`
  - `get_last_report()` → `last_report()`
  - `get_operation_index()` → `operation_index()`
  - `get_report_history()` → `report_history()`
  - `get_spatial()` → `spatial()`
  - `get_timeseries()` → `timeseries()`
  - `get_time_index()` → `time_index()`
  - `get_timeseries_config()` → `timeseries_config()`
  - `get_embeddings()` → `embeddings()`
  - `get_embedding()` → `embedding()`
  - `get_node_by_id()` → `node()`
  - `children_properties_to_list()` → `collect_children()` (also `filter=` param → `where=`)

### Removed

- **`get_ids()`** — removed; use `ids()` for flat ID list or `collect()` for full node dicts

## [0.5.66] - 2026-02-26

### Changed

- **Blueprint loader output** — quiet by default (only warnings/errors + summary); verbose mode for per-type detail. Warnings from `add_connections` skips are now tracked in the loader instead of surfacing as raw `UserWarning`s
- **Blueprint settings** — `root` renamed to `input_root`, `output` split into `output_path` (optional directory) + `output_file` (filename or relative path with `../` support). Old keys still accepted for backwards compatibility

### Fixed

- **Float→Int ID coercion** — FK columns with nullable integers (read as float64 by pandas, e.g. `260.0`) are now auto-coerced to int before edge matching. The Rust lookup layer also gained Float64 → Int64/UniqueId fallback as a safety net
- **Timeseries FK edge filtering** — FK edges for timeseries node types now apply the same time-component filter as node creation (e.g. dropping month=0 aggregate rows), preventing "source node not found" warnings for carriers that only have aggregate data

## [0.5.65] - 2026-02-26

### Added

- **`FLUENT.md`** — comprehensive fluent API reference documenting all method-chaining operations: data loading, selection & filtering, spatial, temporal, timeseries, vector search, traversal, algorithms, set operations, indexes, transactions, export, and a fluent-vs-Cypher feature matrix
- **`create_connections()`** — renamed from `selection_to_new_connections` with new capabilities: `properties` dict copies node properties onto new edges (e.g. `properties={'B': ['score']}`), `source_type`/`target_type` override which traversal levels to connect (defaults to first→last level)
- **Comparison-based `traverse(method=...)`** — discover relationships without pre-existing edges. Five methods: `'contains'` (spatial containment), `'intersects'` (geometry overlap), `'distance'` (geodesic proximity), `'text_score'` (semantic similarity via embeddings), `'cluster'` (kmeans/dbscan grouping). `method` accepts a string shorthand (`method='contains'`) or a dict with settings (`method={'type': 'distance', 'max_m': 5000, 'resolve': 'centroid'}`). The `resolve` key controls polygon geometry interpretation: `'centroid'` (force geometry centroid), `'closest'` (nearest boundary point), `'geometry'` (full polygon shape). Produces the same selection hierarchy as edge-based traversal, so all downstream methods work unchanged
- **`add_properties()`** — enrich selected nodes with properties from ancestor nodes in the traversal chain. Supports copy (`['name']`), copy-all (`[]`), rename (`{'new': 'old'}`), aggregate expressions (`'count(*)'`, `'mean(depth)'`, `'sum(production)'`, `'min()'`, `'max()'`, `'std()'`, `'collect()'`), and spatial compute (`'distance'`, `'area'`, `'perimeter'`, `'centroid_lat'`, `'centroid_lon'`)

### Changed

- **`selection_to_new_connections` → `create_connections`** — renamed for brevity. Now defaults to connecting the top-level ancestor to leaf nodes (was parent→child at last level only)

## [0.5.64] - 2026-02-25

### Added

- **List quantifier predicates** — `any(x IN list WHERE pred)`, `all(...)`, `none(...)`, `single(...)` for filtering over lists in WHERE, RETURN, and WITH clauses
- **Exploration hints in `describe()`** — inventory views now surface disconnected node types and join candidates (property value overlaps between unconnected type pairs) to suggest enrichment opportunities
- **Temporal Cypher functions** — `valid_at(entity, date, 'from_field', 'to_field')` and `valid_during(entity, start, end, 'from_field', 'to_field')` for date-range filtering on both nodes and relationships in WHERE clauses. NULL fields treated as open-ended boundaries

### Changed

- **Rewritten examples** — new domain examples: `legal_graph.py` (index-based loading), `code_graph.py` (code tree parsing), `spatial_graph.py` (blueprint loading), `mcp_server.py` (generic MCP server with auto-detected code tools)

## [0.5.63] - 2026-02-25

### Added

- **`export_csv(path)`** — bulk export to organized CSV directory tree with one file per node type and connection type, sub-node nesting, full properties, and a `blueprint.json` for round-trip re-import via `from_blueprint()`
- **Variable binding in MATCH pattern properties** — bare variables from `WITH`/`UNWIND` can now be used in inline pattern properties: `WITH "Oslo" AS city MATCH (n:Person {city: city}) RETURN n`
- **Map literals in Cypher expressions** — `{key: expr, key2: expr}` syntax in `RETURN`/`WITH` for constructing map objects: `RETURN {name: n.name, age: n.age} AS m`
- **WHERE clause inside EXISTS subqueries** — `EXISTS { MATCH (n:Type) WHERE n.prop = expr }` now supports arbitrary WHERE predicates including cross-scope variable references and regex

### Changed

- **Cypher query performance** — eliminated `type_indices` Vec clone on every MATCH (iterate by reference), move-on-last-match optimization to reduce row cloning in joins, pre-allocated result vectors, eliminated unnecessary clone in composite index lookups
- **MERGE index acceleration** — MERGE now uses `id_indices`, `property_indices`, and `composite_indices` for O(1) pattern matching instead of linear scan through all nodes of a type. Orders-of-magnitude faster for batch `UNWIND + MERGE` workloads
- **UNWIND/MERGE clone reduction** — UNWIND moves (instead of cloning) the row for the last unwound item; MERGE iterates source rows by value to avoid per-row cloning

## [0.5.61] - 2026-02-24

### Added

- **PROFILE** prefix for Cypher queries — executes query and collects per-clause statistics (rows_in, rows_out, elapsed_us). Access via `result.profile`
- **Structured EXPLAIN** — `EXPLAIN` now returns a `ResultView` with columns `[step, operation, estimated_rows]` instead of a plain string. Cardinality estimates use type_indices counts
- **Read-only transactions** — `begin_read()` creates an O(1) Arc-backed snapshot (zero memory overhead). Mutations are rejected
- **Optimistic concurrency control** — `commit()` detects graph modifications since `begin()` and raises `RuntimeError` on conflict
- **Transaction timeout** — `begin(timeout_ms=...)` and `begin_read(timeout_ms=...)` set a deadline for all operations within the transaction
- `Transaction.is_read_only` property
- `describe(cypher=['EXPLAIN'])` and `describe(cypher=['PROFILE'])` topic detail pages
- Expanded `<limitations>` section in `describe(cypher=True)` with workarounds for unsupported features
- openCypher compatibility matrix in CYPHER.md

## [0.5.60] - 2026-02-24

### Added

- `describe(cypher=True)` tier 1 hint now highlights KGLite-specific features (||, =~, coalesce, CALL procedures, distance/contains)
- `describe(cypher=True)` tier 2 includes `<not_supported>` section and spatial functions group
- `describe()` overview connection map includes `count` attribute per connection type
- `describe()` connections hint only shown when graph has edges
- `describe(cypher=['spatial'])` topic with distance, contains, intersects, centroid, area, perimeter docs

## [0.5.59] - 2026-02-24

### Added

- `bug_report(query, result, expected, description)` — file Cypher bug reports to `reported_bugs.md`. Timestamped, version-tagged entries prepended to top of file. Input sanitised against HTML/code injection
- `KnowledgeGraph.explain_mcp()` — static method returning a self-contained XML quickstart for setting up a KGLite MCP server (server template, core/optional tools, Claude registration config)

### Fixed

- `collect(node)[0].property` now returns the actual property value instead of the node's title. Previously, `WITH f, collect(fr)[0] AS lr RETURN lr.oil` would return the node title for every property access. Node identity is now preserved through collect→index→WITH pipelines via internal `Value::NodeRef` references

## [0.5.58] - 2026-02-24

### Added

- `CALL cluster()` procedure — general-purpose clustering via Cypher. Supports DBSCAN and K-means methods. Reads nodes from preceding MATCH clause. Spatial mode auto-detects lat/lon from `set_spatial()` config with geometry centroid fallback; property mode clusters on explicit numeric properties with optional normalization. YIELD node, cluster (noise = -1 for DBSCAN)
- `round(x, decimals)` — optional second argument for decimal precision (e.g. `round(3.14159, 2)` → 3.14). Backward compatible: `round(x)` still rounds to integer
- `||` string concatenation operator — concatenates values in expressions (e.g. `n.first || ' ' || n.last`). Null propagates. Non-string values auto-converted
- `describe(cypher=True)` — 3-tier Cypher language reference: compact `<cypher hint/>` in overview (tier 1), full clause/operator/function/procedure listing with `cypher=True` (tier 2), detailed docs with params and examples via `cypher=['cluster','MATCH',...]` (tier 3)
- `describe(connections=True)` — connection type progressive disclosure: overview with `connections=True` (all types, counts, endpoints, property names), deep-dive with `connections=['BELONGS_TO']` (per-pair counts, property stats, sample edges)

## [0.5.56] - 2026-02-23

### Added

- `near_point_m()` — geodesic distance filter in meters (SI units), replaces `near_point_km()` and `near_point_km_from_wkt()`
- Geometry centroid fallback: fluent API spatial methods (`near_point_m`, `within_bounds`, `get_bounds`, `get_centroid`) now fall back to WKT geometry centroid when lat/lon fields are missing but a geometry is configured via `set_spatial` or `column_types`

### Changed

- Cypher `distance(a, b)` returns Null (instead of erroring) when a node has no spatial data, so `WHERE distance(a, b) < X` simply filters those nodes out
- Cypher comparison operators (`<`, `<=`, `>`, `>=`) now follow three-valued logic: comparisons involving Null evaluate to false (previously Null sorted as less-than-everything)

### Removed

- `near_point_km()` — use `near_point_m()` with meters instead (e.g. `max_distance_m=50_000.0` for 50 km)
- `near_point_km_from_wkt()` — subsumed by `near_point_m()` which auto-falls back to geometry centroid

## [0.5.55] - 2026-02-23

### Changed

- Cypher spatial functions now return SI units: `distance()` → meters, `area()` → m², `perimeter()` → meters (were km/km²). Distance uses WGS84 geodesic (Karney algorithm) instead of spherical haversine

### Removed

- `agent_describe()` — replaced by `describe()`. Migration: `graph.agent_describe()` → `graph.describe()`, `graph.agent_describe(detail='full')` → `graph.describe()` (auto-selects detail level)

## [0.5.54] - 2026-02-23

### Added

- `describe(types=None)` — progressive disclosure schema description for AI agents. Inventory mode returns node types grouped by size with property complexity markers and capability flags, connection map, and Cypher extensions. Focused mode (`types=['Field']`) returns detailed properties, connections, timeseries/spatial config, and sample nodes. Automatically inlines full detail for graphs with ≤15 types
- `set_parent_type(node_type, parent_type)` — declare a node type as a supporting child of a core type. Supporting types are hidden from the `describe()` inventory and appear in the `<supporting>` section when the parent is inspected. The `from_blueprint()` loader auto-sets parent types for sub-nodes
- Cypher math functions: `abs()`, `ceil()` / `ceiling()`, `floor()`, `round()`, `sqrt()`, `sign()` — work with Int64 and Float64 values, propagate Null
- String coercion on `+` operator: when one operand is a string, the other is automatically converted (e.g. `2024 + '-06'` → `'2024-06'`). Null still propagates

### Changed

- `describe()` inventory now uses compact descriptor format `TypeName[size,complexity,flags]` instead of size bands. Types listed as flat comma-separated list sorted by count descending. Core types with supporting children show `+N` suffix. Capability flags from supporting types bubble up to their parent descriptor
- `describe()` now shows a `<read-only>` notice listing unsupported Cypher write commands (CREATE, SET, DELETE, REMOVE, MERGE) when the graph is in read-only mode

## [0.5.53] - 2026-02-23

### Added

- `from_blueprint()` — build a complete KnowledgeGraph from a JSON blueprint and CSV files. Supports core nodes, sub-nodes, FK edges, junction edges, timeseries, geometry conversion, filters, manual nodes (from FK values), and auto-generated IDs
- Cypher `date()` function — converts date strings to DateTime values: `date('2020-01-15')`
- `property_types` on blueprint junction edges for automatic type conversion (e.g. epoch millis → DateTime)
- Temporal join support: `ts_*()` functions accept DateTime edge properties and null values as date range arguments
- Cypher `IS NULL` / `IS NOT NULL` now supported as expressions in RETURN/WITH (e.g. `RETURN x IS NULL AS flag`)
- `agent_describe(detail, include_fluent)` — optional detail level adapts output to graph complexity. Graphs with >15 types auto-select compact mode (~5-8x smaller output). Fluent API docs excluded by default (opt-in via `include_fluent=True`)

### Changed

- **Performance**: `agent_describe()` 27x faster (1.3s → 48ms) via property index fast path and scan capping
- **Performance**: `MATCH (n) RETURN count(n)` short-circuits to O(1) via `FusedCountAll` (was ~266ms, now sub-ms)
- **Performance**: `MATCH (n) RETURN n.type, count(n)` short-circuits to O(types) via `FusedCountByType` (was ~727ms, now sub-ms)
- **Performance**: `MATCH ()-[r]->() RETURN type(r), count(*)` short-circuits to O(E) single-pass via `FusedCountEdgesByType` (was ~822ms, now ~3ms)
- **Performance**: `MATCH (n:Type) RETURN count(n)` short-circuits to O(1) via `FusedCountTypedNode` (reads type index length directly)
- **Performance**: `MATCH ()-[r:Type]->() RETURN count(*)` short-circuits via `FusedCountTypedEdge` (single-pass edge filter)
- **Performance**: Edge type counts cached in DirGraph with lazy invalidation on mutations
- **Performance**: Multi-hop fused aggregation for 5-element patterns (e.g. `MATCH (a)-[]->(b)<-[]-(c) RETURN a.x, count(*)`) traverses without materializing intermediate rows
- **Performance**: Regex `=~` operator caches compiled patterns per query execution (compile once, match many)
- **Performance**: PageRank uses pull-based iteration with rayon parallelization for large graphs (3-4x speedup)
- **Performance**: Louvain community detection precomputes loop-invariant division terms
- Timeseries keys stored as `NaiveDate` instead of composite integer arrays (`Vec<Vec<i64>>`)
- `set_time_index()` now accepts date strings (`['2020-01', '2020-02']`) in addition to integer lists
- `get_time_index()` returns ISO date strings (`['2020-01-01', '2020-02-01']`) instead of integer lists
- `get_timeseries()` keys returned as ISO date strings
- `ts_series()` output uses ISO date strings for time keys (e.g. `"2020-01-01"` instead of `[2020, 1]`)
- Null date arguments to `ts_*()` treated as open-ended ranges (no bound)
- Timeseries data format bumped (v2); legacy files skip timeseries loading with a warning

### Fixed

- `MATCH (a)-[]->(b) RETURN count(*)` with all-aggregate RETURN (no group keys) now correctly returns a single row instead of per-node rows
- `ORDER BY` on DateTime properties with `LIMIT` now returns correct results (FusedOrderByTopK optimization extended to handle DateTime, UniqueId, and Boolean sort keys)
- `ORDER BY` on String/Point properties with `LIMIT` now falls back to standard sort instead of returning empty results

## [0.5.52] - 2026-02-22

### Added

- `add_nodes()` now accepts a `timeseries` parameter for inline timeseries loading from flat DataFrames — automatically deduplicates rows per ID and attaches time-indexed channels
- Timeseries resolution extended to support `hour` (depth 4) and `minute` (depth 5) granularity
- `parse_date_string` now handles `'yyyy-mm-dd hh:mm'` and ISO `'yyyy-mm-ddThh:mm'` formats
- **Timeseries support**: per-node time-indexed data channels with resolution-aware date-string queries
- `set_timeseries()` with `resolution` ("year", "month", "day"), `units`, and `bin_type` metadata
- `set_time_index()` / `add_ts_channel()` for per-node timeseries construction
- `add_timeseries()` for bulk DataFrame ingestion with FK-based node matching and resolution validation
- `get_timeseries()` / `get_time_index()` for data extraction with date-string range filters
- Cypher `ts_*()` functions with date-string arguments: `ts_sum(f.oil, '2020')`, `ts_avg(f.oil, '2020-2', '2020-6')`, etc.
- Query precision validation: errors when query detail exceeds data resolution (e.g. `'2020-2-15'` on month data)
- Channel units (e.g. "MSm3", "°C") and bin type ("total", "mean", "sample") metadata
- Timeseries data persisted as a separate section in `.kgl` files (backward compatible)
- `agent_describe()` includes timeseries metadata, resolution, units, and function reference
- Cypher `range(start, end [, step])` function — generates integer lists for use with `UNWIND`

## [0.5.51] - 2026-02-21

### Added

- Fluent API: `filter()` now supports `regex` (or `=~`) operator for pattern matching, e.g. `filter({'name': {'regex': '^A.*'}})`
- Fluent API: `filter()` now supports negated operators: `not_contains`, `not_starts_with`, `not_ends_with`, `not_in`, `not_regex`
- Fluent API: `filter_any()` method for OR logic — keeps nodes matching any of the provided condition sets
- Fluent API: `offset(n)` method for pagination — combine with `max_nodes()` for page-based queries
- Fluent API: `has_connection(type, direction)` method — filter nodes by edge existence without changing the selection target
- Fluent API: `count(group_by='prop')` and `statistics('prop', group_by='prop')` — group by arbitrary property instead of parent hierarchy

## [0.5.50] - 2026-02-21

### Added

- Shapely/geopandas integration for spatial methods — `intersects_geometry()` and `wkt_centroid()` now accept shapely geometry objects as input in addition to WKT strings
- `as_shapely=True` parameter on `get_centroid()`, `get_bounds()`, and `wkt_centroid()` to return shapely geometry objects instead of dicts
- `ResultView.to_gdf()` — converts lazy results to a geopandas GeoDataFrame, parsing a WKT column into shapely geometries with optional CRS
- Spatial type system via `column_types` in `add_nodes()` — declare `location.lat`/`location.lon`, `geometry`, `point.<name>.lat`/`.lon`, and `shape.<name>` types for auto-resolution in Cypher and fluent API methods
- `set_spatial()` / `get_spatial()` for retroactive spatial configuration
- Cypher `distance(a, b)` now auto-resolves via spatial config (location preferred, geometry centroid fallback)
- Virtual spatial properties in Cypher: `n.location` → Point, `n.geometry` → WKT, `n.<point_name>` → Point, `n.<shape_name>` → WKT
- Spatial methods (`within_bounds`, `near_point_km`, `get_bounds`, `get_centroid`, etc.) auto-resolve field names from spatial config when not explicitly provided
- Node-aware spatial Cypher functions: `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)` — auto-resolve geometry via spatial config, also accept WKT strings
- Geometry-aware `distance()` — `distance(a.geometry, b.geometry)` returns 0 if touching; `distance(point(...), n.geometry)` returns 0 if inside, closest boundary distance otherwise

### Removed

- Cypher functions `wkt_contains()`, `wkt_intersects()`, `wkt_centroid()` — replaced by node-aware `contains()`, `intersects()`, `centroid()` which also accept raw WKT strings

### Fixed

- Betweenness centrality now uses undirected BFS — previously only traversed outgoing edges, causing nodes bridging communities via incoming edges to get zero scores

### Performance

- `RETURN ... ORDER BY expr LIMIT k` fused into single-pass top-k heap — O(n log k) instead of O(n log n) sort + O(n) full projection. **5.4x speedup** on `distance()` ORDER BY LIMIT queries (1M pairs: 2627ms → 486ms)
- `WHERE contains(a, b)` fast path (`ContainsFilterSpec`) — extracts contains() patterns and evaluates directly from spatial cache, bypassing expression evaluator chain
- Spatial Cypher functions 6-8x faster for contains/intersects via per-node spatial cache + bounding box pre-filter:
  - Per-node cache (`NodeSpatialData`): resolves each node's spatial data once per query, cached for all cross-product rows (N×M → N+M lookups)
  - Bounding box pre-filter: computes `geo::Rect` alongside cached geometry; rejects non-overlapping pairs in O(1) before expensive polygon tests
  - `resolve_spatial()` skips redundant expression evaluation for Variable/PropertyAccess — goes directly to cached node data
- Spatial resolution uses WKT geometry cache for centroid fallback path — previously re-parsed WKT on every row
- `intersects()` and `centroid()` avoid deep-cloning `Arc<Geometry>` — use references directly
- `geometry_contains_geometry()` uses `geo::Contains` trait instead of point-by-point boundary check

## [0.5.49] - 2026-02-20

### Added

- Python type stub (`.pyi`) files now included in code graph — enables graph coverage of stub-only packages, compiled extensions, and authoritative type contracts

### Fixed

- Cypher parser now accepts reserved words (e.g. `optional`, `match`, `type`) as alias names after `AS` — previously failed with "Expected alias name after AS"
- Betweenness centrality `sample_size` now uses stride-based sampling across the full node range — previously sampled only the first k nodes, which could be non-participating node types (Module/Class) yielding all-zero scores

## [0.5.46] - 2026-02-20

### Fixed

- Decorator property stored as JSON array instead of comma-separated string — fixes fragmentation of decorators with comma-containing arguments (e.g. `@functools.wraps(func, assigned=(...))`)
- `is_test`, `is_async`, `is_method` boolean properties now explicitly `false` on non-matching entities instead of `null` — enables `WHERE f.is_test = false` queries
- Dynamic project versions (setuptools-scm etc.) now stored as `"dynamic"` instead of `null` on the Project node
- CALLS edges now scope-aware — calls inside nested functions, lambdas, and closures are no longer attributed to the enclosing function (fixes over-counted fan-out in all 7 language parsers)
- `collect(x)[0..N]`, `count(x) + 1` and other aggregate-wrapping expressions in RETURN now work — previously errored with "Aggregate function cannot be used outside of RETURN/WITH"
- `size(collect(...))` and other non-aggregate functions wrapping aggregates now evaluate correctly — previously silently returned `null` because the expression was misclassified as non-aggregate

## [0.5.43] - 2026-02-20

### Added

- List slicing in Cypher: `expr[start..end]`, `expr[..end]`, `expr[start..]` — works on `collect()` results and list literals, supports negative indices

### Fixed

- `size()` and `length()` functions on lists now return element count instead of JSON string length — e.g. `size(collect(n.name))` returns 5 instead of 29
- Duplicate nodes when test directory overlaps with source root (e.g. `root/tests/` inside `root/`) — test roots already covered by a parent source root are now skipped, with `is_test` flags applied to the existing entities instead
- Duplicate Dependency ID collision when same package appears in multiple optional groups — IDs now include the group name (e.g. `matplotlib::viz`)

## [0.5.42] - 2026-02-19

### Added

- `connection_types` parameter for `louvain` and `label_propagation` procedures — filter edges by type, matching the existing support in centrality algorithms

### Fixed

- `CALL pagerank({connection_types: ['CALLS']})` list literal syntax now works correctly — was silently serialized as JSON string causing zero edge matches and uniform scores
- Document list comprehension patterns as unsupported in Cypher reference

## [0.5.41] - 2026-02-19

### Added

- Cypher string functions: `split(str, delim)`, `replace(str, search, repl)`, `substring(str, start [, len])`, `left(str, n)`, `right(str, n)`, `trim(str)`, `ltrim(str)`, `rtrim(str)`, `reverse(str)`

### Fixed

- Duplicate File nodes when source and test roots overlap in code_tree (e.g. `xarray/` source root containing `xarray/tests/` + separate test root)
- Empty `Module.path` properties for declared submodules in code_tree — now resolved from parsed files or inferred from parent directory
- Boolean properties (`is_test`, `is_abstract`, `is_async`, etc.) stored as string `'True'` instead of actual booleans — improved pandas `object` dtype detection to recognize boolean-only columns

## [0.5.39] - 2026-02-19

### Added

- `read_only(True/False)` method to disable Cypher mutations (CREATE, SET, DELETE, REMOVE, MERGE). When enabled, `agent_describe()` omits mutation documentation, simplifying the agent interface for read-only use cases

## [0.5.38] - 2026-02-19

### Added

- Cypher `CALL procedure({params}) YIELD columns` for graph algorithms: pagerank, betweenness, degree, closeness, louvain, label_propagation, connected_components. YIELD `node` is a node binding enabling `node.title`, `node.type` etc. in downstream WHERE/RETURN/ORDER BY clauses
- Inline pattern predicates in WHERE clauses — `WHERE (a)-[:REL]->(b)` and `WHERE NOT (a)-[:REL]->(b)` now work as shorthand for `EXISTS { ... }`, matching standard Cypher behavior
- `CALL list_procedures() YIELD name, description, yield_columns` — introspection procedure listing all available graph algorithm procedures with their parameters and descriptions

### Changed

- `build()` now includes test directories by default (`include_tests=True`)
- CALL procedure error message now hints at the correct map syntax when keyword arguments are used instead of `{key: value}` maps

### Fixed

- CALLS edge resolution in code_tree now uses tiered scope-aware matching (same owner > same file > same language > global) instead of flat bare-name lookup — eliminates false cross-class and cross-language edges
- Rust parser now detects test files at the File level (`_test.rs`, `test_*`, `tests/`, `benches/` conventions) — previously only function-level `#[test]` attributes were detected, leaving File nodes untagged

## [0.5.36] - 2026-02-18

### Changed

- Split `mod.rs` (6,742 LOC) into 5 thematic `#[pymethods]` files: algorithms, export, indexes, spatial, vector — mod.rs reduced to 4,005 LOC
- Enabled PyO3 `multiple-pymethods` feature for multi-file `#[pymethods]` blocks
- Documented transaction isolation semantics (snapshot isolation, last-writer-wins)

### Fixed

- `[n IN nodes(p) | n.name]` now correctly extracts node properties in list comprehensions over path functions — previously returned serialized JSON fragments instead of property values
- `parse_list_value` is now brace-aware — splits at top-level commas only, preserving JSON objects and nested structures
- `EXISTS { MATCH (pattern) }` syntax now accepted — the optional `MATCH` keyword inside EXISTS braces is silently skipped, matching standard Cypher behavior

## [0.5.35] - 2026-02-18

### Added

- CALLS edges now carry `call_lines` and `call_count` properties — line numbers where each call occurs in the caller function
- Comment annotation extraction (TODO/FIXME/HACK/NOTE/etc.) for all non-Rust parsers (Python, TypeScript, JavaScript, Java, Go, C, C++, C#)
- Test file detection (`is_test`) for all parsers based on language naming conventions
- Generic/type parameter extraction for Go 1.18+ and Python 3.12+ (PEP 695) parsers

## [0.5.34] - 2026-02-18

### Added

- `toc(file_path)` method: get a table of contents for any source file — all code entities sorted by line number with a type summary
- `find()` now accepts `match_type` parameter: `"exact"` (default), `"contains"` (case-insensitive substring), `"starts_with"` (case-insensitive prefix)
- `file_toc` MCP tool in `examples/mcp_server.py` for file-level exploration
- `find_entity` MCP tool now supports `match_type` parameter
- Qualified name format documented in `agent_describe()` output (Rust: `crate::module::Type::method`, Python: `package.module.Class.method`)
- Block doc comment support (`/** */`) in Rust parser — previously only `///` line comments were captured
- `call_trace` MCP tool in `examples/mcp_server.py` for tracing function call chains (outgoing/incoming, configurable depth)
- Call trace Cypher pattern documented in `agent_describe()` output
- CHANGELOG.md, CONTRIBUTING.md, and CLAUDE.md for project governance

### Changed

- Doc comments added to all critical Rust structs (`KnowledgeGraph`, `DirGraph`, `CypherExecutor`, `PatternExecutor`, `CypherParser`, and 15+ supporting types)
- Rust parser now captures all `use` declarations, not just `crate::` prefixed imports
- MCP tool descriptions improved with workflow guidance (`graph_overview` says "ALWAYS call this first", `cypher_query` mentions label-optional MATCH, etc.)
- GitHub Release workflow now uses CHANGELOG.md content instead of auto-generated notes

## [0.5.31] - 2025-05-15

### Added

- `find(name, node_type=None)` method: search code entities by name across all types
- `source(name)` method: resolve entity names to file paths and line ranges (supports single string or list)
- `context(name, hops=None)` method: get full neighborhood of a code entity grouped by relationship type
- `find_entity`, `read_source`, `entity_context` MCP tools in `examples/mcp_server.py`
- Label-optional MATCH documented in `agent_describe()` — `MATCH (n {name: 'x'})` searches all node types

### Changed

- Code entity helpers (`find`, `context`) moved from Python (`kglite/code_tree/helpers.py`) to native Rust methods for performance
- `agent_describe()` now conditionally shows code entity methods and notes when code entities are present in the graph

### Removed

- `kglite/code_tree/helpers.py` — replaced by native Rust methods on `KnowledgeGraph`

## [0.5.28] - 2025-05-10

### Added

- Manifest-based building: `build(".")` auto-detects `pyproject.toml` / `Cargo.toml` and reads project metadata (name, version, dependencies)
- `Project` and `Dependency` node types with `DEPENDS_ON` and `HAS_SOURCE` edges
- `USES_TYPE` edges: Function → type references in signatures
- `EXPOSES` edges: FFI boundary tracking (PyO3 modules → exposed items)

### Fixed

- Various code tree parser fixes for Rust trait implementations and method resolution

## [0.5.22] - 2025-04-28

### Added

- `kglite.code_tree` module: parse multi-language codebases into knowledge graphs using tree-sitter
- Supported languages: Rust, Python, TypeScript, JavaScript, Go, Java, C++, C#
- Node types: File, Module, Function, Struct, Class, Enum, Trait, Protocol, Interface, Constant
- Edge types: DEFINES, CALLS, HAS_METHOD, HAS_SUBMODULE, IMPLEMENTS, EXTENDS, IMPORTS
- Embedding export support

---

*For versions prior to 0.5.22, see [GitHub Releases](https://github.com/kkollsga/kglite/releases).*

[0.5.35]: https://github.com/kkollsga/kglite/compare/v0.5.34...v0.5.35
[0.5.34]: https://github.com/kkollsga/kglite/compare/v0.5.31...v0.5.34
[0.5.31]: https://github.com/kkollsga/kglite/compare/v0.5.28...v0.5.31
[0.5.28]: https://github.com/kkollsga/kglite/compare/v0.5.22...v0.5.28
[0.5.22]: https://github.com/kkollsga/kglite/releases/tag/v0.5.22
