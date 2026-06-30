# The boundary principle (wrappers vs core)

> This document is the full reference for KGLite's wrapper/core boundary
> doctrine and the Phase H C-ABI history. `CLAUDE.md` carries a short
> summary and links here. Read this when working on the `kglite::api::*`
> surface, the C ABI, or a new binding.

## The boundary principle (north star for wrappers vs core)

When deciding where a piece of code belongs:

> **A wrapper only contains code that is specific to its environment
> and cannot be used by any other sibling wrapper. Anything two or
> more wrappers would write identically belongs in `kglite::api`.**

Concrete examples:

- PyO3 marshalling (`#[pyfunction]`, `Py<PyAny>`, NumPy/Pandas
  conversion) → Python wrapper (`crates/kglite-py/`). A Go binding
  doesn't use any of it.
- `tqdm` progress display, `_PROCESS_CACHE` dict for Jupyter rerun-
  cell ergonomics → Python wrapper. Go uses channels, JS uses a
  module-level `Map`.
- SEC form-string → bucket mapping, ticker JSON parser, cache-
  freshness decision tree, blocking/async runtime bridge → core
  (`kglite::api::*`). Every binding asks the same questions the
  same way.

The principle applies in both directions, with **different
postures for each**:

## Wrapper → core (the LIFT direction): active-design posture, cypher-first, use-case-checked

We are actively designing the api surface for future bindings (Go
via cgo, JS via napi, JVM via JNI, …). Default-generous about
lifting generic-and-useful capabilities — don't wait for a second
binding to discover the gap, file a request, and wait for us to
ship it. The cost of speculative-but-useful lifts is small; the
cost of not-lifting is that every new binding author hits "wait, I
have to reinvent this from the wheel" on day one.

**But "generic" isn't enough — test the use case.** Before lifting
any helper or proposing any new Cypher function/procedure, ask:
*who would actually call this, and in what query / workflow?* If
the only honest answers are "validation that should happen at load
time anyway" or "type introspection that fights a data-modeling
smell" or "syntactic sugar over an existing function" — drop it.
Generic-and-pointless adds api surface to maintain without
delivering value.

Concrete use-case test examples (worked through 2026-05-25):

- `wkt_is_valid` as a Cypher function — DROPPED. The only honest
  use cases (pre-CREATE validation, find-malformed-data audit) are
  better addressed at load time where Rust-level `parse_wkt` is
  already directly callable.
- `add_days(date, n)` — KEPT. Real query: "events scheduled in
  the next 30 days": `WHERE e.date <= add_days(date(), 30)`.
- `shortest_path_length(a, b)` — KEPT. Real query: "how many hops
  from X to Y" without materializing the path.
- `quartile(x)` aggregation — DROPPED. Syntactic sugar over
  existing `percentile_cont(x, 0.25)`; no extra value.

**But: lift to the right surface.** kglite has two surfaces that
bindings reach:

1. **Cypher (the universal per-query surface).** Every wrapper
   exposes a `cypher_query` tool/method. New Cypher functions and
   procedures are reached automatically by every binding through
   that one entry point — no per-wrapper wiring required.
2. **Direct Rust api (the bootstrap / lifecycle / error surface).**
   Items in `kglite::api::*` that bindings call directly during
   open / build / save / error-mapping / embedder-registration.

Cypher-first is the default for any per-query feature: WKT helpers,
date/time helpers, string formatters, graph algorithms, statistics,
aggregations. A new binding running `cypher_query("WHERE
wkt_within(geom, $box)")` gets WKT for free. Wiring those as direct
Rust helpers (`kglite::api::geometry::validate_wkt`) forces every
binding to expose them through their own FFI layer.

Direct Rust api is for things Cypher can't express:

- The Cypher pipeline itself (`session::execute_*`, `cypher::parse_*`)
- Lifecycle: `load_file`, `save_graph`, `from_blueprint`
- Error types and codes (errors fire before/after Cypher)
- Embedder registration (bindings hand kglite an `Arc<dyn Embedder>`)
- Storage backend configuration
- Dataset loaders (the fetched data isn't a graph yet)

When in doubt, ask: "is this a per-query feature or a setup/error
concern?" Per-query → Cypher function or procedure. Setup/error →
direct Rust api.

## Core → wrapper (the DOWNGRADE direction): strict posture

Default-suspicious of items in `kglite::api::*`. Burden of proof
is on *keeping* an item, not on removing it. The question to ask
of every item is: "Is this *tailored for one specific binding's
environment*? If I were writing it for a Go binding from scratch,
would I write the same thing — or differently?" If "differently"
→ demote.

Consumer count is **not** the test (we ship one major wrapper
today, of course most items have one consumer). The test is the
*shape*: does the signature take a wrapper-specific type
(`Bound<PyAny>`, `BoltValue`, `&CowSelection`) or encode a
wrapper's input idiom (duck-typed Python objects, language-
specific display conventions)? If yes → tailored, demote.

## Combined: lift generously, demote rigorously

The two postures sound contradictory but aren't. Generic-and-useful
logic lifts proactively (don't wait); tailored-to-one-binding shapes
get demoted rigorously (don't keep speculatively). The boundary
between the two is the *signature* of the lifted thing — generic
core types in, generic core types out.

## Four explicit goals for the binding framework

The principle + postures above exist in service of four goals for
future-language wrappers. Any api shape decision should be
checked against all four:

1. **Quick + easy** — A new wrapper is small (target: Rust-side
   wrapper < 1000 LOC of glue; non-Rust wrapper < 1500 LOC total
   of FFI shim + language-native idioms). A new binding author
   sets up a "hello, query a graph" example in under a day.

2. **Standardized** — Users switching between wrappers see the
   same data model (`Value` variants, error categories), the same
   query language (Cypher), and the same lifecycle vocabulary
   (`open` / `save` / `from_blueprint`). The look-and-feel of
   binding-specific idioms differs (Python's PyDict vs Go's map vs
   JVM's HashMap), but the *concepts* match across wrappers.

3. **Centrally maintained** — When we add a feature in core, every
   binding gets it without per-wrapper code changes — either
   automatically (a new Cypher function reaches all bindings via
   `cypher_query`) or via a single pin-bump (a new api function
   becomes available after the binding's next dependency update).
   We don't have to ship N PRs across N bindings for one feature.

4. **Flexible** — The interface shape doesn't restrict us from
   adding crucial functionality later. We can ship a new Value
   variant, a new ExecuteOptions field, a new Cypher function, a
   new dataset, without breaking existing bindings or forcing
   them to fork. Non-breaking additions are the dominant change
   mode.

Score every proposed lift against these four. Anything that fails
two or more is the wrong shape; redesign or skip.

## Two-tier standardization architecture

Different binding types reach kglite through different layers:

| Binding type | Standardization layer | Examples |
|---|---|---|
| **Rust-side wrappers** | `kglite::api::*` — Rust types, traits, functions | `kglite-py` (PyO3), `kglite-bolt-server`, `kglite-mcp-server`; future `kglite-grpc-server`, `kglite-rest-server` |
| **Non-Rust wrappers** | C ABI — `extern "C" fn` over `kglite::api::*` (the `kglite-c` crate) | Future Go (cgo), JavaScript (napi), JVM (JNI), .NET (P/Invoke) |

**A "framework helper" in `kglite::api::*` is reachable only by
Rust-side wrappers.** Non-Rust wrappers won't see a `ParamUnmarshaller`
trait or a `GraphHandle` struct directly — they see a C function
signature in `kglite.h`. For *those* bindings, the standardization
is the C ABI shape itself.

**Phase H — the `kglite-c` crate — shipped in 0.10.3.** What landed
across H.1–H.5:

1. **H.1 — C ABI design** (`docs/rust/c-abi.md`). Conventions:
   `kglite_*` naming, opaque-handle pattern (empty `#[repr(C)]`
   facade + private `XState` sidecar), errno-style errors mapping
   1:1 to `KgErrorCode`, owned out-strings freed via a single
   `kglite_free_string`, JSON-at-boundary for nested `Value`
   shapes, sync-only ABI (bindings own their own async).

2. **H.2 — `kglite-c` skeleton + cbindgen.** Workspace member at
   `crates/kglite-c/`. Top-12 entry points: lifecycle / session /
   Cypher / result accessors / error introspection / ABI version.
   cbindgen runs in `build.rs` and writes
   `include/kglite.h`.

3. **H.3 — Sodir + embedder ABI.** First dataset wrapper +
   fastembed factory + `kglite_session_set_embedder`. Locked in
   the feature-gating convention (cbindgen `[defines]` maps
   `feature = X` to `KGLITE_FEATURE_X` preprocessor define).

4. **H.3a — SEC + Wikidata ABI.** Completed the dataset surface
   symmetrically. Total surface: 30 `extern "C"` functions, 6
   opaque-handle types, 952-line generated header.

5. **H.5 — release coordination.** Header-drift CI gate (fresh
   cbindgen run vs committed header). `publish_crates.yml`
   extended with a 4th publish step. `implementing-a-binding.md`
   rewritten with cgo / napi / JNI worked examples.

H.4 (Go PoC consumer) was **deferred** — the first real non-Rust
binding author validates the surface better than a synthetic
500-LOC sketch. The cgo / napi / JNI examples in
`implementing-a-binding.md` give them a starting point.

The boundary-principle posture above (active-design + cypher-first +
use-case-checked) applies to the Rust `api::*` surface AND the C
ABI we expose through it. Same rules: per-query features go via
Cypher (no C ABI exposure needed — bindings call
`kglite_session_execute_read(...)`); lifecycle/error/embedder go
via direct C functions; tailored-to-one-language shapes never
appear in the C ABI.

## The runtime model — core is sync, bindings own async

`kglite::api::session::execute_read` / `execute_mut` are
**synchronous**. The Cypher pipeline runs to completion on the
calling thread. Async fetchers (`fetch_*` in `kglite::api::datasets::*`)
have `*_blocking` companions for callers without a tokio runtime.

This is deliberate. Each binding chooses its own async/threading
model on top:

- Python wheel: releases the GIL via `py.detach()` for parallel readers
- Bolt server: drives the sync pipeline from a `tokio::task::spawn_blocking`
- MCP server: same; runs on tokio but `execute_read` itself is sync
- Future Go binding: goroutines wrapping the sync C ABI
- Future JS binding: napi async with `.spawn_blocking` equivalent
- Future JVM binding: thread pool + sync JNI calls

Never force tokio on a binding. If we make the canonical Cypher
entry async, Go/JVM bindings either drag a tokio runtime into their
language's runtime (painful) or fork the function. Sync-by-default
is the cross-language-friendly choice.

## What's INTENTIONALLY per-binding (the negative space)

These are deliberately NOT in `kglite::api::*`. They're per-binding
because each one has language-idiom or protocol-shape concerns:

| Concern | Where it lives | Why |
|---|---|---|
| Value ↔ native type marshalling | Each binding's `value_adapter` / `py_in` / etc. | `PyDict` / `BoltValue` / protobuf / `js::Object` are language-specific |
| Error formatting / wrapping into protocol types | Each binding's `error_*` module | `PyErr`, `BoltError`, `tonic::Status`, etc. |
| Wire format (JSON / CSV / BoltValue / protobuf / Arrow) | Each binding's `result_format` / serializer | Each protocol has its preferred encoding |
| Display protocols (`__repr__`, `Debug`, JSON debug) | Each binding's `_repr_*` | Language-specific protocols |
| Tool registration mechanism | Each binding's `tools::register` / manifest YAML / route table | Protocol-specific (MCP tool YAML, REST route registration, gRPC `Service` impl) |
| Result iteration style (eager / lazy / streaming) | Each binding's `ResultView` / `ResultStream` / iterator | Protocol-shape-specific; Python supports lazy, Bolt streams, MCP is eager |
| Async / threading model | Each binding | See "runtime model" above |
| CLI / config-file parsing | Each binding's own | mcp-server uses clap + YAML manifest; bolt-server uses clap + flags; wheel uses argparse; a future Go binding would use Go's `flag` or `cobra` |
| Logging / observability | Each binding's native logger | Rust binaries → `tracing`; Python → stdlib `logging`; Go would use `slog`; JVM would use `slf4j`. Don't unify — each ecosystem has its own conventions. |
| Lifecycle / teardown semantics | Each binding's native idiom | Python → `__del__` + context managers (`with`); Rust → `Drop`; JVM → finalizers + try-with-resources; JS → explicit `.close()`. Different cleanup contracts per language. |

If you find yourself wanting to "unify" any of these, that's a
yellow flag. They're per-binding *by design* — unifying forces all
bindings into one language's idiom or one protocol's shape.

## Worked examples from the 2026-05-25 sweep

To anchor the abstract rules:

**Lifts that PASSED the use-case + cypher-first tests** (shipped or
queued):
- `parse_with_mutation_check` — direct api, every binding's pipeline-bootstrap pattern
- `ExecuteOptions::eager` — direct api, factory for the conservative-defaults shape
- `KgErrorCode::neo4j_status_code` — direct api, every Neo4j-wire-compatible binding shares
- `add_days` / `add_months` / `add_years` / `date_truncate` — Cypher fns, real "events in next N days" query
- `shortest_path_length` — Cypher fn, real "how many hops" query
- `mode(x)` — Cypher aggregation, real "most common value per group" query
- `db.property_stats` / `db.property_uniqueness` / `db.graph_stats` — Cypher procedures, real schema-introspection queries

**Lifts that FAILED the tests** (dropped):
- `wkt_is_valid` — only honest use cases (pre-CREATE validation, audit) belong at load time
- `wkt_type` — fights mixed-geometry-types data smell
- `lpad`, `rpad` — display formatting is binding concern
- `quartile`, `decile` — syntactic sugar over `percentile_cont`
- Standalone `cosine_similarity` — already inside `vector_score`
- `GraphHandle` struct — too generic to add value; each binding's state genuinely differs
- `ParamUnmarshaller` trait — Rust-side trait that non-Rust bindings can't see; helps only future Rust-side wrappers (not yet)
- `QueryContext` — `temporal_context` is wheel-only today

See `docs/rust/implementing-a-binding.md` → "Wrapping a dataset for
your binding" for the worked dataset example. (The reverse-audit
methodology — strict posture, test the signature not the
consumer count — is recorded in the maintainer's local audit
under `dev_workfolder/dev-documentation/audits/`.)
