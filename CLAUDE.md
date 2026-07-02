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

### The boundary principle (wrappers vs core) — summary

Full doctrine + Phase H C-ABI history: **`docs/rust/boundary-principle.md`**.
Read it before working on the `kglite::api::*` surface, the C ABI, or a new
binding. The essentials:

> **A wrapper only contains code that is specific to its environment and
> cannot be used by any other sibling wrapper. Anything two or more wrappers
> would write identically belongs in `kglite::api`.**

- **Lift generously, demote rigorously.** Lift generic-and-useful logic
  proactively (don't wait for a second binding); demote anything whose
  *signature* is tailored to one binding (takes `Bound<PyAny>`/`BoltValue`,
  encodes a language idiom). The test is the shape, not the consumer count.
- **Cypher-first.** Per-query features (WKT/date/string helpers, graph algos,
  stats, aggregations) go in as Cypher functions/procedures — every binding
  gets them free via `cypher_query`. Direct `kglite::api::*` is only for what
  Cypher can't express: the pipeline itself, lifecycle (`load_file`/
  `save_graph`/`from_blueprint`), error types/codes, embedder registration,
  storage config, dataset loaders.
- **Use-case test before lifting.** Ask "who calls this, in what query?" Drop
  load-time validation, data-smell introspection, and sugar over existing fns.
- **Core is sync; bindings own async.** `execute_read`/`execute_mut` run to
  completion on the calling thread; `fetch_*` has `*_blocking` companions.
  Never force tokio on a binding.
- **Two tiers:** Rust-side wrappers reach `kglite::api::*` directly; non-Rust
  wrappers reach the C ABI (`kglite-c`, shipped 0.10.3). Marshalling, error
  formatting, wire format, display, tool registration, iteration style,
  logging, lifecycle/teardown are **intentionally per-binding** — don't unify.

## In-memory is the core product

Three storage modes: `Default` (in-memory petgraph), `Mapped` (mmap-backed columns), `Disk` (CSR + mmap). The disk modes are addons for large-graph exploration (Wikidata-scale). When optimisation conflicts arise, **in-memory wins** — never regress in-memory perf to protect disk safety. Add disk-specific workarounds gated on storage mode or graph size instead.

The Cypher planner/executor is shared across all modes. Changes to `core/pattern_matching.rs` or `languages/cypher/executor.rs` affect everyone — benchmark on small in-memory graphs before merging.

## Code health

Each pass through a file should leave it more compartmentalised than you found it.

- **No bugs left behind.** When you encounter a pre-existing bug while working — even one unrelated to your task — fix it in the same change, or if it's genuinely out of scope, surface it explicitly (file an issue / call it out) rather than silently stepping over it. Don't leave known bugs in the codebase. Before "fixing", confirm it's actually a bug and not deliberate behaviour: read the surrounding code and tests, check whether it's consistent across versions, and distinguish a real defect from an intentional design choice (e.g. the planner schema-check rejecting unknown CREATE properties is a deliberate typo-guard, not a bug). A measured performance change is only a "fix" if it measurably improves performance — never ship a perf change that doesn't.
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
- **No back-compat shims, no `#[deprecated]` — this is about *code/APIs*, not *data*.** Obsoleted code/API paths are deleted in the same PR as their replacement: no deprecated public surface, no dual old-vs-new-API codepaths, no compat wrappers for renamed/replaced functions. **Data-format compatibility is a separate, legitimate concern and is NOT a "shim".** Persisted files (`.kgl`, disk graphs) outlive the binary that wrote them, so *reading* an older on-disk/serialized format (read-compat), or *detecting* one and refusing it with a clear "rebuild your graph" message (a deliberate hard-break, e.g. the `.kgl` v3→v4 break or the embeddings-provenance break), is expected format-lifecycle handling — keep or migrate it, don't delete it to satisfy this rule. The test when unsure: *would deleting this break a caller's **code** (shim → remove) or an existing user's **saved file** (data-compat → keep/migrate)?* Examples that are NOT shims and stay: `EdgePropertyStore` legacy-format detection, `ConnectionTypeInfo`'s old-field deserializer, the v3-magic rejection in `io/file.rs`.
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

## Inbox hygiene

`inbox/unread/` (at the repo root) holds incoming feedback/bug/coordination
notes (named `YYYY-MM-DD-from-<sender>-<topic>.md`); `inbox/read/` is the
archive. The inbox is gitignored (`/inbox/`) — it's local working state, not
committed.

**When a message has been actioned, move it from `inbox/unread/`
to `inbox/read/`.** "Actioned" means the work shipped, the bug was verified
fixed, or it's a no-action acknowledgement — not merely read. `unread/`
must reflect only what still needs doing, so a stale "you still have unread
mail" never hides a genuinely open item among resolved ones. Append a
one-line `## Status (kglite, <date>): …` footer to substantive work-items
before moving, so `inbox/read/` carries the resolution record.

**Route to the party who can act.** A note only belongs in another project's
inbox (e.g. `../mcp-servers/inbox/`, `../mcp-methods/inbox/`) if it carries an
*actionable* task for them. If there's nothing for them to do, don't file it —
their `unread/` should hold only things that need their action.

## Commits & releases

Commit format: `type: short description` (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`). Update `CHANGELOG.md` `[Unreleased]` for user-visible changes; skip for internal refactors, CI, test-only, formatting.

**Commit messages are public — keep sensitive intent out of them.** The
message is part of the permanent, externally-visible history. Don't let it
spell out anything we'd rather keep subtle (competitive positioning, who or
what a change targets, internal motivations, security-sensitive details).
Describe the *mechanical* change in neutral terms — what the diff does to the
code/docs — not the strategy behind it. When a change touches something
delicate, default to the plainest accurate phrasing (e.g. "generalize
benchmark wording", "tidy CHANGELOG") over anything that narrates the reason.

**Pushing requires explicit, in-the-moment approval.** Default is *don't push*. The user runs `git push` manually unless they tell you, *in the same turn you'd run it*, to push for them — e.g. "go ahead and push now", "push it", "yes, push". Approval is one-shot: it covers exactly that one `git push` invocation and does not carry across to any later commit, amend, or branch.

**Exception — the CI fix-and-push loop.** When an approved push triggers CI that fails, and you diagnose the failure as a bug in shipped code or test/CI infra (not a feature gap), you may push subsequent `fix(...)` / `ci(...)` commits *for that same loop* without re-asking, until CI on the most recent push is fully green. This covers the common case where the first push surfaces a flaky dep / missing fixture / linter-only issue and you'd otherwise need to ping the user every iteration just to type "push" again.

The exception **stops applying** the moment any of these are true:
- All required workflows on the latest push reach `conclusion: success` → loop converged, fresh approval needed for the next push
- A fix would change the release shape (new version, new feature, scope expansion, removal of declared functionality) → ask, don't push
- More than ~3 fix-and-push iterations happen on the same loop without progress → likely a deeper problem, surface it and ask
- The user pivots the conversation away from the CI loop → context shift means fresh approval needed

The loop's pushes are still subject to the same rigor as any release push (lint clean, tests green, dry-runs pass before pushing). The exception removes the "ask first" step, not the "build with care" step.

Conversational phrasing from earlier in the session ("ship it", "looks good", "you may push", "we're ready") **does not** carry over to a later moment outside the fix-and-push loop, even within the same turn if other actions intervene. When in doubt, prepare the commit, stop, and ask. The cost of a re-prompt is small; an unapproved push to `main` is not.

Version source of truth: **`[workspace.package] version` in the root `Cargo.toml`**. Every member crate (engine, wheel, `kglite-c`, servers, cli) sets `version.workspace = true` and inherits it, so a release bumps this one line and all published crates ship in lockstep.

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
