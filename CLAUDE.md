# KGLite — Claude Code Conventions

## Build & test

```bash
source .venv/bin/activate && unset CONDA_PREFIX
maturin develop --release    # release build; required for any perf measurement
make test                    # Rust + Python tests
make lint                    # fmt + clippy + ruff format/check + mypy stubtest; run before pushing
```

`make lint` covers the full CI gate. If you run pieces by hand, both
`cargo fmt --check` and `ruff format --check` matter — CI gates on the
`--check` variants separately from the auto-fix variants.

## Architecture

- **Rust core** (`src/`): `KnowledgeGraph` exposed to Python via PyO3, `petgraph` storage.
- **Cypher engine** (`src/graph/languages/cypher/`): parser → AST → planner → executor.
- **Shared query primitives** (`src/graph/core/`): pattern matching, filtering, traversal — used by both Cypher and the fluent API.
- **Python package** (`kglite/`): thin wrapper + `code_tree/` (tree-sitter codebase parsing).
- **Type stubs** (`kglite/__init__.pyi`): source of truth for API docs.
- **Introspection** (`src/graph/introspection/`): `describe()` XML schema for agents.

### The boundary principle (north star for wrappers vs core)

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

### Wrapper → core (the LIFT direction): active-design posture, cypher-first, use-case-checked

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

### Core → wrapper (the DOWNGRADE direction): strict posture

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

### Combined: lift generously, demote rigorously

The two postures sound contradictory but aren't. Generic-and-useful
logic lifts proactively (don't wait); tailored-to-one-binding shapes
get demoted rigorously (don't keep speculatively). The boundary
between the two is the *signature* of the lifted thing — generic
core types in, generic core types out.

### Four explicit goals for the binding framework

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

### Two-tier standardization architecture

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

### The runtime model — core is sync, bindings own async

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

### What's INTENTIONALLY per-binding (the negative space)

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

### Worked examples from the 2026-05-25 sweep

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
under `dev-documentation/audits/`.)

## In-memory is the core product

Three storage modes: `Default` (in-memory petgraph), `Mapped` (mmap-backed columns), `Disk` (CSR + mmap). The disk modes are addons for large-graph exploration (Wikidata-scale). When optimisation conflicts arise, **in-memory wins** — never regress in-memory perf to protect disk safety. Add disk-specific workarounds gated on storage mode or graph size instead.

The Cypher planner/executor is shared across all modes. Changes to `core/pattern_matching.rs` or `languages/cypher/executor.rs` affect everyone — benchmark on small in-memory graphs before merging.

## Code health

Each pass through a file should leave it more compartmentalised than you found it.

- Factor a function when it grows past ~80 lines or starts handling 3+ unrelated concerns. Prefer small named strategy fns dispatched by the caller over long if/else chains.
- Fixing a bug — scan for the *class* of bug. The reported symptom is rarely the only one; probe with scratch fixtures before declaring scope.
- A new feature is a chance to extract a helper that's been wanted elsewhere. Don't over-design, don't pass it up either.
- Don't add a parameter/branch/flag without checking whether the existing structure should be reshaped to absorb it.

### Cypher planner passes

The optimiser pipeline lives at `src/graph/languages/cypher/planner/mod.rs` as `const PASSES: &[(&str, PassFn)]` — single source of truth for order and naming. When adding or changing a pass:

1. **Implement** in the appropriate sub-module (`fusion.rs`, `simplification.rs`, …) or a new file for fresh concerns.
2. **Register** in `PASSES` with a unique stable name (user-facing via `disabled_passes=[...]`).
3. **Doc-comment** the wrapper fn: precondition, pattern matched, rewrite, why-bail.
4. **Add a query** to `tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES` exercising the trigger shape. Passes not in the corpus aren't trusted.
5. **Bisect divergences** with `scripts/cypher_pass_bisect.py` before assuming a query is wrong.

The differential corpus is *the* mechanism preventing silent correctness regressions — every fix to an optimiser bug lands its triggering query into the corpus as part of the fix commit.

## Performance protocol

Before any perf-related change:

1. **Baseline first** — write/extend a benchmark covering touched code paths. Run it, record numbers.
2. **Release mode only.** `maturin develop --release`. Never trust debug-build numbers; per-test variance is unbounded.
3. **Trust `min` over `median`** for sub-millisecond benches. Median pulls upward with system load; min reflects best-case throughput.
4. **Tighten the harness for noisy benches**:
   - `--benchmark-min-rounds=100` (200 for sub-10-µs benches).
   - `--benchmark-warmup=on --benchmark-warmup-iterations=20`.
   - 30-second sleep between baseline and comparison runs (thermal settle).
   - Re-measure twice on the suspect commit. If runs disagree, you're seeing variance, not a regression.
5. **In-memory is the gate.** Disk-mode benchmarks are nice-to-have but never at the cost of in-memory.

## Key patterns

- **PyO3**: `&self` for read-only methods; return `PyResult<Py<PyAny>>`; wrap blocking work in `Python::attach()`. Use `.cast::<T>()`, not `.downcast::<T>()` (deprecated in pyo3 0.27+).
- **`#[pymethods]` location**: all method blocks live under `src/graph/pyapi/`. Private helpers stay in `src/graph/mod.rs` as `pub(crate)`. The `#[pyclass]` *struct attribute* may stay with the struct definition.
- **Value conversion**: `py_out::value_to_py()` and `py_out::nodeinfo_to_pydict()`.
- **Storage traits**: reads on `GraphRead`, mutations on `GraphWrite: GraphRead` (both in `src/graph/storage/mod.rs`). Add new storage ops to the trait first. `GraphRead` is non-object-safe (GATs on iterator methods) — use `&impl GraphRead` everywhere, never `&dyn`. Iterator-returning trait methods declare an associated type (`type FooIter<'a>: Iterator<…> where Self: 'a;`).
- **Transactions stay on `DirGraph`**, not in the trait surface (`version`, `read_only`, `schema_locked`, validation helpers).
- **No back-compat shims, no `#[deprecated]`.** Obsoleted code is deleted in the same PR as its replacement.
- **Parity oracles** at `tests/test_storage_parity.py`, `tests/test_phase{1,2,3}_parity.py` (gated by `pytest -m parity`) must stay green after any backend-touching change.

## When changing a `#[pymethods]` function — the five-place checklist

1. `src/graph/pyapi/*.rs` — implementation.
2. `kglite/__init__.pyi` — type stub + docstring.
3. `src/graph/introspection/*.rs` — `describe()` output, if agent-facing.
4. `crates/kglite-mcp-server/src/tools.rs` — MCP tool wrapper, if agent-facing.
5. `CHANGELOG.md` `[Unreleased]` — user-visible changes only.

## Documentation

Docs auto-rebuild at [kglite.readthedocs.io](https://kglite.readthedocs.io) on every push to `main`.

- **API reference**: auto-generated from `kglite/__init__.pyi` docstrings.
- **Cypher reference**: `CYPHER.md`.
- **Fluent API reference**: `FLUENT.md`.
- **Guide content**: `docs/guides/*.md`.
- **README.md**: landing page only — don't duplicate guide content.

## Commits & releases

Commit format: `type: short description` (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`). Update `CHANGELOG.md` `[Unreleased]` for user-visible changes; skip for internal refactors, CI, test-only, formatting.

**Pushing requires explicit, in-the-moment approval.** Default is *don't push*. The user runs `git push` manually unless they tell you, *in the same turn you'd run it*, to push for them — e.g. "go ahead and push now", "push it", "yes, push". Approval is one-shot: it covers exactly that one `git push` invocation and does not carry across to any later commit, amend, or branch.

**Exception — the CI fix-and-push loop.** When an approved push triggers CI that fails, and you diagnose the failure as a bug in shipped code or test/CI infra (not a feature gap), you may push subsequent `fix(...)` / `ci(...)` commits *for that same loop* without re-asking, until CI on the most recent push is fully green. This covers the common case where the first push surfaces a flaky dep / missing fixture / linter-only issue and you'd otherwise need to ping the user every iteration just to type "push" again.

The exception **stops applying** the moment any of these are true:
- All required workflows on the latest push reach `conclusion: success` → loop converged, fresh approval needed for the next push
- A fix would change the release shape (new version, new feature, scope expansion, removal of declared functionality) → ask, don't push
- More than ~3 fix-and-push iterations happen on the same loop without progress → likely a deeper problem, surface it and ask
- The user pivots the conversation away from the CI loop → context shift means fresh approval needed

The loop's pushes are still subject to the same rigor as any release push (lint clean, tests green, dry-runs pass before pushing). The exception removes the "ask first" step, not the "build with care" step.

Conversational phrasing from earlier in the session ("ship it", "looks good", "you may push", "we're ready") **does not** carry over to a later moment outside the fix-and-push loop, even within the same turn if other actions intervene. When in doubt, prepare the commit, stop, and ask. The cost of a re-prompt is small; an unapproved push to `main` is not.

Version source of truth: `Cargo.toml` line 3 (post-G.4: `crates/kglite-py/Cargo.toml` for the wheel version, `crates/kglite/Cargo.toml` for the engine — both should match at release time).

### One version bump per push

A version isn't "released" until the user pushes. If a `release(x.y.z): ...` commit is already local, fold any follow-up work into the same `[x.y.z]` CHANGELOG block — amend or extend the release commit, don't add a new `release(x.y.z+1): ...` on top.

Check before bumping:

```bash
git log origin/main..HEAD --oneline | grep -E "^\w+ release\("
```

If that returns a commit, keep the version it picked. Only mint a new version after a clean push to origin.

### Captured-constant refresh at release time

Three captured values drift across releases and need a version-paired refresh as part of the release commit. The gates that check them are otherwise reliable — see Test infrastructure → Phase 4 / Phase 5 / perf-regression — but they go stale silently when nobody updates the captured constants. `make refresh-release-constants` does all three in one pass and prints a `git diff --stat` so the maintainer can stage them into the release commit.

- `tests/test_phase4_parity.py::GOLDEN_V3_DIGEST` (and demote the prior value into `ACCEPTABLE_DIGESTS`). The version string lives in the `.kgl` header, so every release shifts the digest.
- `tests/test_phase5_parity.py::test_binary_size_regression` baseline. Update the docstring's "what grew" note with each bump — the script adds a `TODO: describe what grew since the prior baseline` line for the maintainer to fill in.
- `tests/benchmarks/baselines/<version>.json` and `current.json`. Captured by re-running the 11 tracked core benchmarks. The script is idempotent — if `<version>.json` already exists, recapture is skipped (delete the file to force a fresh capture; benchmark numbers are inherently noisy so we don't want to overwrite on every script run).

The script requires a fresh release build (`maturin develop --release`) for steps 2 and 3.

### Multi-phase plans

When a plan has Steps 1 / 2 / 3 / …:

1. **One commit per phase.** Bisectability beats batched commits. Each phase's code + tests in its own `feat:` / `refactor:` / etc.
2. **Each phase must be green before its commit** — `cargo build --lib`, `make lint`, and the relevant test suite all pass.
3. **Keep going to the end.** Once a plan is approved, don't pause between phases. The only mid-plan stops are genuine blockers (failing test you can't fix, architectural surprise invalidating a later step).
4. **End with a perf gate.** Before the final release commit, run new + existing benchmarks per the Performance protocol above. Record numbers in the release commit message or `[x.y.z]` CHANGELOG block. Fix regressions before the release commit, not in a follow-up.
5. **Final commit is the version bump + CHANGELOG promotion.** No earlier phase touches `Cargo.toml`. User pushes once.
