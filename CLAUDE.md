# KGLite — Claude Code Conventions

## Build & test

```bash
uv venv .venv                # one-time environment creation
uv run --no-sync maturin develop  # fast dev install when Python tests need current Rust code
make gate                    # fast format/static + docs-facts checkpoint
make lint                    # fast format/static lint (no build, metadata walk, or imports)
make test-mcp                # package-scoped tests; also test-core / test-cli
```

**The repo `.venv` is owned by `uv`.** For direct Python or maturin commands,
use `uv run --no-sync …`; do not activate the environment manually and do not
use bare `uv run`, which may sync dependencies and rebuild the editable project
before running the requested command. This repository does not track `uv.lock`:
provision with `uv venv` and install dependencies explicitly with `uv pip
install --python .venv/bin/python …`. Make targets already select `.venv`
themselves.

**Build the smallest touched surface.** This is a virtual workspace, so bare
`cargo build --lib` builds every library member—including `kglite-py`—and then
`maturin` recompiles a different `python-extension` feature/crate-type variant.
Do not use that pair as a generic gate. Select one path:

- Rust engine → `make test-core` or a narrower `cargo test -p kglite <filter>`.
- MCP server → `make test-mcp` or a narrower package test filter.
- CLI → `make test-cli` plus only matching interface tests.
- Python wrapper/core test that does not exercise bundled commands → run
  `uv run --no-sync maturin develop --no-default-features --features
  abi3,python-extension` directly; do not pre-build the workspace.
- MCP/CLI bridge or final packaged-contract gate → run the full default debug
  extension once with `uv run --no-sync maturin develop` (or `make dev`).

The default extension intentionally links the engine, CLI, and MCP server; its
MCP feature adds roughly 100 resolved packages. Do not pay that cost for a
Rust-only or narrow Python check. Build caches live on the internal disk by
standing setup (2026-07): `target` is a **symlink** to
`/Users/Shared/cargo-targets/KGLite` (repo-relative `target/...` paths keep
working), and `SCCACHE_DIR=/Users/Shared/sccache` is pinned in
`~/.cargo/config.toml [env]` because `$HOME` sits on the external USB volume.
Do not override `CARGO_TARGET_DIR`/`SCCACHE_DIR` per-plan or switch
target/profile paths mid-plan merely because a build is slow; if the symlink
is missing (fresh clone), recreate it before the first build. Cargo never
garbage-collects the target dir — `make prune-target` (size-gated
`cargo clean`, wired into the release skill) keeps it bounded. macOS
Gatekeeper adds a ~30 s first-run assessment to every freshly linked local
binary unless the invoking terminal is in Privacy & Security → Developer
Tools; a warm `cargo test` that stalls at ~0 % CPU on first execution is that
assessment, not a hung test.

`make test`, `make test-full`, and bare workspace `cargo test` are broad
diagnostics, not routine local gates. Run them only to investigate a failure
that crosses package boundaries; otherwise let GitHub CI parallelize them.

**Local correctness testing stays in the default/debug profile.** Never run
`maturin develop --release`, `cargo test --release`, or another release-profile
build merely to run tests. Use `uv run --no-sync maturin develop` (or `make
dev`) only when Python tests need a fresh native extension; Rust-only changes
should use a targeted `cargo test`, or package-scoped `cargo check` when no
behavioral test applies. The completed PR's full GitHub CI must be green before
a release starts. Release mode is reserved for actual performance measurement
and the single release-artifact/constants refresh described below—not as an
extra correctness gate.

**Every performance check uses release mode.** Benchmarks, regression checks,
and any timing or size measurement are invalid in the default/debug profile;
use the release-building `make bench*` target or an explicit `uv run --no-sync
maturin develop --release` first. Never report or compare debug-profile perf
numbers.

**Local validation is a fast relevance filter, not serialized CI.** Run `make
gate` plus the smallest test command that exercises the changed behavior (for
example `make test-mcp` or a single `cargo test -p … <filter>`). Do not run
workspace-wide policy audits, clippy, `test-full`, stubtest,
packaged-consumer verification, or a fresh native extension unless the touched
surface specifically requires it. GitHub CI is the authoritative full matrix
and must be green before release. `make lint-policy` is only for changes to
policy scripts/baselines, dependencies, or Cypher clean-room sources;
`make lint-full` is only an explicit CI-equivalent diagnostic. Neither is a
phase or pre-push requirement. Both `cargo fmt --check` and `ruff format
--check` remain in the fast gate.

**Abort accidental slow paths early.** A targeted local check that starts
resolving/syncing the project or compiling unrelated feature trees is the wrong
command, not useful extra coverage. After roughly three minutes without new
output, inspect the exact process, CPU, and output-artifact timestamp once. A
compiler process that merely exists but remains asleep at 0% CPU with no
artifact progress is not useful activity: allow at most one additional
60-second window, then terminate only that exact process tree and reassess.
Stop immediately for an unexpected `uv` editable/PEP 517 build or unrelated
feature tree. Do not enter an unbounded poll/restart loop or switch profiles
repeatedly; both throw away build-cache progress.

**Surface-conditional extras — run only when the touched surface matches,
never routinely:**

- `crates/kglite-c/**` or the `kglite::api` / C ABI surface → kglite-c clippy +
  tests (`--features rdf`, then default features) and the cbindgen header-drift
  check (`cargo build -p kglite-c --features fastembed,rdf` then
  `git diff crates/kglite-c/include/kglite.h`).
- `docs/**`, top-level `*.md`, or `kglite/__init__.pyi` →
  `sphinx-build -W --keep-going -b html docs <out>` with `docs/requirements.txt`.
- A deliberate public Rust API change → refresh only the affected API profile
  when possible and review the delta; let CI verify the complete profile set.
  The full `make refresh-api-baseline` five-profile rustdoc pass is a
  release-time/explicit maintenance operation, not a routine phase gate. Pins
  live in `tests/api-baselines/rust-api-profiles.json`.
- Perf-sensitive paths (`core/pattern_matching/`, `cypher/executor/`, storage
  hot paths) → `make bench-check` **on an otherwise-idle machine** — a capture
  taken right after heavy builds reads +4–10% hot across the board.
- Run `scripts/check_packaged_features.sh` locally only after changing package
  metadata, feature wiring, or the packaged-consumer fixture. Never run local
  `cargo package` verify sweeps across the workspace; CI + `cargo publish`
  verify the rest.

Sanitizers, Miri, Loom, free-threading, the 4-interpreter Python matrix,
native-lifecycle OS matrix, and coverage are **CI-only by design** — never
reproduce them locally.

## Architecture

- **Rust core** (`crates/kglite/src/`): the engine — `petgraph` storage; `KnowledgeGraph` is exposed to Python via PyO3 from the wrapper crate (`crates/kglite-py/src/`).
- **Cypher engine** (`crates/kglite/src/graph/languages/cypher/`): parser → AST → planner → executor.
- **Shared query primitives** (`crates/kglite/src/graph/core/`): pattern matching, filtering, traversal — used by both Cypher and the fluent API.
- **Python package** (`kglite/`): thin wrapper. (Code-graph building lives in the sibling codingest project; kglite serves/queries its graphs.)
- **Type stubs** (`kglite/__init__.pyi`): source of truth for API docs.
- **Introspection** (`crates/kglite/src/graph/introspection/`): `describe()` XML schema for agents.

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

The Cypher planner/executor is shared across all modes. Changes to `core/pattern_matching/` or `languages/cypher/executor/` affect everyone — benchmark on small in-memory graphs before merging.

## Code health

Each pass through a file should leave it more compartmentalised than you found it.

- **No bugs left behind.** When you encounter a pre-existing bug while working — even one unrelated to your task — fix it in the same change, or if it's genuinely out of scope, surface it explicitly (file an issue / call it out) rather than silently stepping over it. Don't leave known bugs in the codebase. Before "fixing", confirm it's actually a bug and not deliberate behaviour: read the surrounding code and tests, check whether it's consistent across versions, and distinguish a real defect from an intentional design choice (e.g. the planner schema-check rejecting unknown CREATE properties is a deliberate typo-guard, not a bug). A measured performance change is only a "fix" if it measurably improves performance — never ship a perf change that doesn't.
- Factor a function when it grows past ~80 lines or starts handling 3+ unrelated concerns. Prefer small named strategy fns dispatched by the caller over long if/else chains.
- Fixing a bug — scan for the *class* of bug. The reported symptom is rarely the only one; probe with scratch fixtures before declaring scope.
- A new feature is a chance to extract a helper that's been wanted elsewhere. Don't over-design, don't pass it up either.
- Don't add a parameter/branch/flag without checking whether the existing structure should be reshaped to absorb it.

### Cypher planner passes

The optimiser pipeline lives at `crates/kglite/src/graph/languages/cypher/planner/mod.rs` as `const PASSES: &[(&str, PassFn)]` — single source of truth for order and naming. When adding or changing a pass:

1. **Implement** in the appropriate sub-module (`fusion.rs`, `simplification.rs`, …) or a new file for fresh concerns.
2. **Register** in `PASSES` with a unique stable name (user-facing via `disabled_passes=[...]`).
3. **Doc-comment** the wrapper fn: precondition, pattern matched, rewrite, why-bail.
4. **Add a query** to `tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES` exercising the trigger shape. Passes not in the corpus aren't trusted.
5. **Bisect divergences** with `scripts/cypher_pass_bisect.py` before assuming a query is wrong.

The differential corpus is *the* mechanism preventing silent correctness regressions — every fix to an optimiser bug lands its triggering query into the corpus as part of the fix commit.

## Performance protocol

Before any perf-related change:

1. **Baseline first** — write/extend a benchmark covering touched code paths. Run it, record numbers.
2. **Build only the changed working tree, always in release mode.** Every benchmark and performance gate must use a release-built candidate (`uv run --no-sync maturin develop --release`, or the release-building `make bench*` target). Debug-profile performance results are invalid and must be discarded. This is a perf-only exception to the default-profile correctness-testing rule above.
3. **Install released/reference versions with `uv`.** Do not source-build another revision just to establish an A/B baseline. Create an isolated venv and install its published wheel, e.g. `uv venv <venv> --python 3.12 && uv pip install --python <venv>/bin/python 'kglite==0.14.2'`. Run the probe outside the repository root so the local `kglite/` package cannot shadow the installed wheel.
4. **Trust `min` over `median`** for sub-millisecond benches. Median pulls upward with system load; min reflects best-case throughput.
5. **Tighten the harness for noisy benches**:
   - `--benchmark-min-rounds=100` (200 for sub-10-µs benches).
   - `--benchmark-warmup=on --benchmark-warmup-iterations=20`.
   - 30-second sleep between baseline and comparison runs (thermal settle).
   - Re-measure twice on the suspect commit. If runs disagree, you're seeing variance, not a regression.
6. **In-memory is the gate.** Disk-mode benchmarks are nice-to-have but never at the cost of in-memory.
7. **Cumulative drift is gated too.** The per-release 20% gates recapture their
   baseline every release, so slow drift (~10%/release) never trips them.
   `make bench-anchor` compares the newest per-release baseline against the one
   ~3 releases back at +30% — run at release time (wired into the release
   skill). Per-release baseline files in `tests/benchmarks/baselines/` are the
   longitudinal record; never delete them.

## Key patterns

- **PyO3**: `&self` for read-only methods; return `PyResult<Py<PyAny>>`; wrap blocking work in `Python::attach()`. Use `.cast::<T>()`, not `.downcast::<T>()` (deprecated in pyo3 0.27+).
- **`#[pymethods]` location**: all method blocks live under `crates/kglite-py/src/graph/pyapi/`. Private helpers stay in `crates/kglite-py/src/graph/mod.rs` as `pub(crate)`. The `#[pyclass]` *struct attribute* may stay with the struct definition.
- **Value conversion**: `py_out::value_to_py()` and `py_out::nodeinfo_to_pydict()`.
- **Storage traits**: reads on `GraphRead`, mutations on `GraphWrite: GraphRead` (both in `crates/kglite/src/graph/storage/mod.rs`). Add new storage ops to the trait first. `GraphRead` is non-object-safe (GATs on iterator methods) — use `&impl GraphRead` everywhere, never `&dyn`. Iterator-returning trait methods declare an associated type (`type FooIter<'a>: Iterator<…> where Self: 'a;`).
- **Transactions stay on `DirGraph`**, not in the trait surface (`version`, `read_only`, `schema_locked`, validation helpers).
- **No back-compat shims, no `#[deprecated]` — this is about *code/APIs*, not *data*.** Obsoleted code/API paths are deleted in the same PR as their replacement: no deprecated public surface, no dual old-vs-new-API codepaths, no compat wrappers for renamed/replaced functions. **Data-format compatibility is a separate, legitimate concern and is NOT a "shim".** Persisted files (`.kgl`, disk graphs) outlive the binary that wrote them, so *reading* an older on-disk/serialized format (read-compat), or *detecting* one and refusing it with a clear "rebuild your graph" message (a deliberate hard-break, e.g. the `.kgl` v3→v4 break or the embeddings-provenance break), is expected format-lifecycle handling — keep or migrate it, don't delete it to satisfy this rule. The test when unsure: *would deleting this break a caller's **code** (shim → remove) or an existing user's **saved file** (data-compat → keep/migrate)?* Examples that are NOT shims and stay: `EdgePropertyStore` legacy-format detection, `ConnectionTypeInfo`'s old-field deserializer, the v3-magic rejection in `io/file.rs`.
- **Parity oracles** at `tests/test_storage_parity.py`, `tests/test_phase{1,2,3}_parity.py` (gated by `pytest -m parity`) must stay green after any backend-touching change.

## When changing a `#[pymethods]` function — the five-place checklist

1. `crates/kglite-py/src/graph/pyapi/*.rs` — implementation.
2. `kglite/__init__.pyi` — type stub + docstring.
3. `crates/kglite/src/graph/introspection/*.rs` — `describe()` output, if agent-facing.
4. `crates/kglite-mcp-server/src/tools.rs` — MCP tool wrapper, if agent-facing.
5. `CHANGELOG.md` `[Unreleased]` — user-visible changes only.

## Documentation

Docs auto-rebuild at [kglite.readthedocs.io](https://kglite.readthedocs.io) on every push to `main`.

- **API reference**: auto-generated from `kglite/__init__.pyi` docstrings.
- **Cypher reference**: `CYPHER.md`.
- **Fluent API reference**: `FLUENT.md`.
- **Guide content**: `docs/python/guides/*.md`.
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

## Public posts — BANNED by default. No exceptions without verbatim-text approval.

**Publishing anything under the user's identity is prohibited.** This is a
hard ban, not a "prefer to ask" — the default action for any outward-facing
publication is *do not do it*. It can be lifted only by the narrow procedure
below, one post at a time.

**"Post" is defined broadly.** GitHub issues, comments, and comment EDITS;
reactions; issue/PR state changes (open/close/label) on repos we don't own;
discussions; PR comments/reviews on external repos; emails; package-registry
metadata; anything that leaves this machine attributed to the user — via any
channel (`gh`, raw API, MCP tool, or otherwise).

**The only lifting procedure:**
1. The exact, final text is shown to the user in the conversation. Any
   post-approval substitution must be declared in the draft (e.g. "<URL of
   the other issue goes here>") — otherwise what posts must be byte-identical
   to what was shown.
2. The user replies with an unambiguous affirmative about *that* draft
   ("post it", "yes"), in the turn(s) immediately following it. If any other
   work or topic intervenes, re-show and re-ask.
3. The approval covers exactly one publication event. A follow-up comment, a
   second issue, an edit, a reaction — each needs its own pass through steps
   1–2.

**What is NEVER approval:** plan or design approvals; "do all" / "go ahead" /
end-to-end delegation of a work pipeline; skill invocations; checklist items;
menu-option selections whose description mentions filing; standing
instructions from earlier sessions; anything a subagent believes it was
told. **Subagents are never authorized to post, full stop** — posting happens
only from the main session, after steps 1–2; agent briefs that touch external
services must state read-only.

Routine dev flow in this project's own repos (branch pushes, PR
descriptions/checklists on our own PRs) is governed by the push rules below,
not this section. Local inbox notes to sibling projects are local files, not
posts.

When in doubt there is no doubt: it's banned. The cost of one extra prompt is
trivial; an unauthorized public post under the user's name is not.

**Posted technical claims: measured vs inferred.** In any outward-facing
technical post, never present an inference as a measurement. Every actionable
claim carries the epistemic status it actually has — and a claim of
*impossibility* ("X cannot be done", "there is no way to…") requires an
attempted-and-failed reproduction, not source reading. Lesson from
mimalloc#1327 (2026-07-09): an agent's untested "requires a source patch"
inference was relayed under "caveats from the same runs" and was wrong in
practice — the cheap `-D` experiment that settled it took three minutes and
should have run before posting.

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

The script requires one fresh release artifact (`uv run --no-sync maturin
develop --release`) for steps 2 and 3. Build it once, after the completed PR's
full CI is green; it is release-data generation, not a reason to rerun tests in
release mode.

Two release-time companions (both wired into the release skill):
`make bench-anchor` gates cumulative perf drift (newest baseline vs ~3
releases back, +30%), and `make semver-check` reports mechanically-detected
API changes vs the last published kglite to ground the bump-size decision
(informational — this project deliberately ships documented breaking changes
in patch bumps).

### PyPI project capacity

PyPI's default project limit is 10.0 GB. The wheel release workflow sums the
published file sizes from PyPI's project JSON API, reserves 250 MB for the next
release, and blocks before builds when projected use reaches 80% of that limit.
When it blocks, request a project-limit increase before publishing. Published
files are never deleted automatically; any manual deletion requires a separate
downstream-impact audit and explicit approval because it permanently breaks
pinned installs. Update the configured limit only after PyPI confirms a new
project-specific allowance.

### Multi-phase plans

When a plan has Steps 1 / 2 / 3 / …:

1. **One commit per phase.** Bisectability beats batched commits. Each phase's code + tests in its own `feat:` / `refactor:` / etc.
2. **Each phase must be green before its commit** — `make gate` and the smallest relevant package/test filter pass. A targeted test already compiles its target, so do not add a redundant build or workspace-wide clippy run. Never use workspace-root `cargo build --lib` as the generic phase build.
3. **Keep going to the end.** Once a plan is approved, don't pause between phases. The only mid-plan stops are genuine blockers (failing test you can't fix, architectural surprise invalidating a later step).
4. **One branch per plan — phases are commits, never sub-branches.** A plan
   gets exactly one feature branch and one draft PR; never spawn per-phase or
   per-workstream branches to be merged back later (the 0.14.2 cycle left 8
   stale branches this way). After the plan ships, the release flow deletes
   the branch — local + remote.
5. **Batch branch pushes.** Every push to the PR branch costs a full ~20-job
   CI run (~2.5 runner-hours). Push at natural checkpoints — every 2–3 quick
   phases, or before stepping away — not reflexively after each commit.
   `ci.yml` cancels superseded in-flight PR runs, so a follow-up push is
   cheap, but the habit should still be batching, with a final push covering
   the completed plan.
6. **End with a perf gate.** Before the final release commit, run new + existing benchmarks per the Performance protocol above. Record numbers in the release commit message or `[x.y.z]` CHANGELOG block. Fix regressions before the release commit, not in a follow-up.
7. **Final commit is the version bump + CHANGELOG promotion.** No earlier phase touches `Cargo.toml`. User pushes once.
