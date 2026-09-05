# KGLite — Claude Code Conventions

**Authority:** `CLAUDE.md` is the authority this repo's agent instructions are
regenerated from; `AGENTS.md` and `.agents/` are generated adapters (and
`.claude/skills/` is the authority for `.agents/skills/`). Edit the authority
and regenerate in the same action — never edit an adapter.

## Doctrine adoption

Follow `../doctrine/learn-from-us.md` → Doctrine sync procedure at planning or
release entry. Versioned corrections apply to KGLite even when made in doctrine
first. Merge into the declared authority and regenerate adapters; preserve local
improvements. A missing marker requires an adoption audit. Planned/deferred work
does not advance `dev-docs/.doctrine-synced`. Snapshotting never performs sync.

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
Rust-only or narrow Python check. Build caches live on the external volume
beside the repos by standing setup (2026-08-31, superseding the 2026-07
internal-disk setup after unbounded target dirs filled the 228 GB system disk
to zero mid-build): `target` is a **symlink** to
`/Volumes/EksternalHome/coding-cache/cargo-targets/KGLite` (repo-relative
`target/...` paths keep working), and
`SCCACHE_DIR=/Volumes/EksternalHome/coding-cache/sccache` is pinned in
`~/.cargo/config.toml [env]`. The old "external USB is slow" premise was
re-measured at the move: ~973 MB/s sequential write (SSD) vs ~2.1 GB/s
internal — not a build bottleneck.
Do not override `CARGO_TARGET_DIR`/`SCCACHE_DIR` per-plan or switch
target/profile paths mid-plan merely because a build is slow; if the symlink
is missing (fresh clone), recreate it before the first build. Cargo never
garbage-collects the target dir — `make prune-target` (`cargo clean` when the
build volume drops below 40 GB free, or the dir meters 40+ GB, wired into the
release skill) keeps it bounded, and every building target carries a
`check-free-space` prerequisite that warns below 40 GB free and refuses below
15 GB. Alternating
`make dev` and a workspace `cargo test` leaves TWO full-size kglite rlib
variants (maturin's single-manifest scope narrows the package set, which
re-unifies one transitive dep's flags and forks the metadata hash — no
feature difference; evidenced 2026-08-31): expected cache cost, not a bug —
don't chase the duplicate. Dep debuginfo is trimmed to `line-tables-only`
via `[profile.dev.package]` overrides in the root manifest (rlib −43%;
debug assertions unaffected — see the comment there before touching them). macOS
Gatekeeper adds a ~30 s first-run assessment to every freshly linked local
binary unless the invoking terminal is in Privacy & Security → Developer
Tools; a warm `cargo test` that stalls at ~0 % CPU on first execution is that
assessment, not a hung test.

`make test`, `make test-full`, and bare workspace `cargo test` are broad
diagnostics, not routine local gates. Run them only to investigate a failure
that crosses package boundaries; otherwise let GitHub CI parallelize them.

**Testing discipline (locked 2026-07-22).**

- **The installed extension must be a debug build during correctness work.** A
  release-built `kglite.abi3.so` silently disables the debug-only assertions
  (parser stack-depth behavior, the disk arena-guard protocol) that are the
  suite's real detectors — the 2026-07-21 audit found latent bugs those
  assertions catch immediately. After any release-profile build (benchmarks,
  release constants), rebuild debug (`uv run --no-sync maturin develop`)
  before the next test run.
- **Any harness that loads a built artifact resolves newest-of-profile.** The
  Python suites do it through `tests/conftest.py::workspace_binary` — newest of
  release/debug, and a skip-with-rebuild-command when the binary predates the
  root `Cargo.toml`; the Java harness does it in `Abi.resolveLibrary` (added
  2026-08). Never hard-code *or prefer* a profile path; a stale release
  artifact shadowing fresh code produces contract failures that reveal nothing,
  and "release if present" is the same bug wearing a default.
- **Targeted tests in flight; the full battery once, at the program's
  completion.** A landing's gate is `make gate` plus the suites *chosen to
  catch what that change could break* — its touched surface and that surface's
  direct consumers — not a fixed list and not everything. The full default
  pytest run, the parity and differential corpora, and workspace clippy run
  once over the **union** of a multi-phase program's changes, at its end. That
  is the stated default for programs (2026-08), not a per-phase requirement.
- **Every Python test carries a 120 s hang ceiling** (`pytest-timeout`,
  configured in `pyproject.toml`; opt-in heavy markers are exempted in
  `tests/conftest.py::pytest_collection_modifyitems`). A test that hits the
  ceiling is a FAILED test — fix the hang; never raise the default, never
  wait out a stuck run. The default suite's slowest test is ~2 s, so the
  ceiling is pure hang detection.
- **A reported status is not the result — check the primitive (added
  2026-07-28).** Every instance below failed in the *reassuring* direction,
  and two of them put a false claim into a committed file:
  - `cargo check … | tail` reports **tail's** exit code. Use `set -o pipefail`
    or read `$?` from the command itself. Never claim "verified" from a
    pipeline's status.
  - A GitHub step with `continue-on-error: true` reports
    `conclusion: success` **after exiting 101**. Read `.outcome`, never
    `.conclusion` — and check for a **job-level** `continue-on-error`, which
    makes the whole job unable to fail and any gate step inside it decorative.
  - `cargo metadata --no-deps` skips resolution entirely and passes on a
    workspace that cannot resolve (use `make bump-version`, which resolves).
  - `gh run list` straight after a push returns a partial set, so "0
    incomplete" is true and meaningless. Require the expected run **count**
    first, then wait for completion.
  - **"It compiles" is a weak test for a dependency floor.** Below `anyhow`
    1.0.47, `anyhow!("{e}")` compiles with a warning and prints the literal
    `{e}` to users. Where a version can compile yet misbehave, *run* it.
  - `git add` with **one** bad pathspec stages **nothing** — the good paths in
    the same invocation are discarded with it, and the only complaint is on
    stderr. A dotted-vs-underscored filename voided an entire release staging
    that way (2026-08). Read back `git status --porcelain`, never the silence.
  - `grep -c` **exits 1 on a count of zero**, so it silently breaks the `&&`
    chain of a command that was only ever asking "how many?".
  - A backgrounded shell can fail its `cd` restore and leave nothing behind but
    the echo of a dead eval. Read the output artifact the command was supposed
    to write, not the message it printed about writing it.

  Same rule as the non-vacuity doctrine, applied to observability: ask "can
  this green go red?" and prove it. Prefer an **expected-failure contract**
  (tolerate one named failure; red on anything else, *including* the failure
  disappearing) over a blanket `continue-on-error`, which silences a job
  instead of scoping what it tolerates.
- **When a committed claim is retracted, grep for every place it was
  written.** The 2026-07-28 minimal-versions claim lived in three files; two
  were corrected and `dev-docs/todos.md` — which advertises itself as enough
  to brief a fresh agent — stayed wrong for hours.

**Dev-environment cleanliness — every file accumulation needs a gate.** Any
path the tooling writes outside git must have a bound and an owner: `target/`
→ `make prune-target` (free-space gate: cleans below 40 GB free on the build
volume, or above a 40 GB metered size — `du` undercounts APFS clone-shared
cargo artifacts ~2×, so free space is the meter and `du` only a diagnostic);
regenerable artifacts and tool caches
(`.bench-current.json`, `docs/_build`, `.mypy_cache`, `.ruff_cache`,
`.pytest_cache`, `.uv-cache`, stale ABI-variant extensions, `.DS_Store`) →
`make prune-dev` (wired into the release skill); sccache → its 30 GiB config
cap; `dev-docs/` and `inbox/` → their skills; `../KGLite-worktrees/` → the
release flow (next paragraph). Never add a new file-writing step (bench
capture, fixture dump, scratch graph) without pointing it at the session
scratchpad / `tmp_path`, or adding it to `prune-dev` in the same change.
`.hypothesis/` is deliberately exempt — it is the found-counterexample
regression corpus, not a cache.

**Agent worktrees live in `KGLite-worktrees/<name>` — never loose in the
parent folder.** An agent that needs an isolated tree creates it as
`git worktree add ../KGLite-worktrees/<name> <branch>`: a sibling *directory*
of the repo, holding all of them, not a scattered `../KGLite-<name>` beside
the real projects in `Rust/`. Release cleanup reclaims inactive trees through
`.claude/skills/dev-docs-cleanup/SKILL.md` §6. Preserve and verify commits,
staged/unstaged changes, untracked contents and valuable ignored files before
removal; a bare `git diff` is incomplete. A detached HEAD needs a recovery ref
or bundle. Keep active, dirty or ambiguous trees and remove the parent directory
only if empty. Note that a
fresh worktree does **not** inherit `target` — that is a symlink to
`/Volumes/EksternalHome/coding-cache/cargo-targets/KGLite` in this repo (see
"Build the smallest touched surface"), and a worktree missing it cold-builds
into its own local `target/` outside the shared cache. Recreate the symlink
before the first build in any new worktree.

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
surface specifically requires it — or you are running a multi-phase program's
completion union, where the full pytest run, the parity/differential corpora,
and workspace clippy run once ("Testing discipline"). GitHub CI is the
authoritative full matrix
and must be green before release. `make lint-policy` is only for changes to
policy scripts/baselines, dependencies, or Cypher clean-room sources; neither
it nor `make lint-full` is a per-phase requirement. Both `cargo fmt --check`
and `ruff format --check` remain in the fast gate.

**The one exception: run `make lint-full` once before the first push of a
long-lived branch.** `make gate` runs *none* of the CI-only policy gates —
`scripts/check_source_quality.py` (centralisation, god files, complexity
ceilings), `scripts/check_lint_allowances.py`, workspace `cargo clippy
--all-targets -- -D warnings`, stubtest — so a program branch can accumulate
weeks of work that CI rejects on contact. The 0.15.8 branch reached its first
push carrying four such blockers (god files, complexity drift, clippy debt,
unreviewed `#[allow]`s), every one invisible to `make gate`. Once per branch,
not once per phase.

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
  `git diff crates/kglite-c/include/kglite.h`). Note that *any* workspace-wide
  `cargo check`/`build` runs this build script and rewrites the header — an
  unrelated command can leave `kglite.h` dirty (a non-default cbindgen, e.g.
  the one a minimal-versions resolve picks, rewrites it into an older form).
  Check `git status` for it before staging.
- `.github/workflows/**` → `pytest tests/test_ci_workflow.py`. `make gate`
  compiles no Rust and runs no Python tests, so a workflow edit is
  **completely ungated locally**: bumping two action majors once failed all
  four Python-matrix jobs on `main` because a guard test pinned the old
  versions. The file is fast (~0.2 s) and there is no excuse for skipping it.
- `docs/**`, top-level `*.md`, or `kglite/__init__.pyi` →
  `sphinx-build -W --keep-going -b html docs <out>` with `docs/requirements.txt`.
- A deliberate public Rust API change → refresh only the affected API profile
  when possible and review the delta; let CI verify the complete profile set.
  The full `make refresh-api-baseline` five-profile rustdoc pass is a
  release-time/explicit maintenance operation, not a routine phase gate. Pins
  live in `tests/api-baselines/rust-api-profiles.json`.
- Perf-sensitive paths (`core/pattern_matching/`, `cypher/executor/`, storage
  hot paths) → `make bench-check`, **under whatever load the machine has** —
  don't wait for an idle machine and don't defer the check because a build is
  running. Read the verdict the way "Performance protocol" items 8–9 require:
  against its control cells, two agreeing runs, one retake near a threshold.
- Run `scripts/check_packaged_features.sh` locally only after changing package
  metadata, feature wiring, or the packaged-consumer fixture. Never run local
  `cargo package` verify sweeps across the workspace; CI + `cargo publish`
  verify the rest.

Sanitizers, Miri, Loom, free-threading, the 4-interpreter Python matrix,
native-lifecycle OS matrix, and coverage are **CI-only by design** — never
reproduce them locally.

## Architecture

Crate and module layout is derivable from `ls crates/` and the manifests; the
two things it does *not* tell you:

- **`kglite/__init__.pyi` is the source of truth for API docs** — not the Rust
  docstrings, not the `.md` files.
- **Code-graph building lives in the sibling codingest project**, not here.
  kglite serves and queries those graphs; it does not build them.

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
  storage config.
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

## Code review — report what is broken, not what you would have written

**This section is addressed to review agents. It overrides any default
reviewer instinct to produce a list of improvements.**

**Design critique has a stage, and review is not it.** Work here runs
investigation → plan approval → implementation → review. The **planning** stage
is where "I would have designed this differently" belongs: it is invited there,
argued there, and settled there — that is what plan approval *is*. Once the plan
is approved, review measures the implementation against exactly two things: the
plan it agreed to, and correctness. It does not measure it against the
reviewer's alternative design, however better that design may be. A reviewer who
forms a design opinion while reading a diff has not found a defect; they have
found **input for the next plan**, and that is where it goes — raised at the
next investigation phase, where it gets argued on its merits, rather than
arriving as an unanswerable objection attached to code that already works.

A review's output is not a to-do list. It is an answer to one question: *what
here is wrong?* If nothing is wrong, the correct review is "no findings", and
that is a good review, not a lazy one. A reviewer that always returns an action
list is not measuring the code — it is measuring its own appetite for
restructuring, and it trains the reader to skim past real defects sitting in
the same list as preferences.

**A finding requires a concrete failure.** Name the inputs or state, and the
wrong behaviour they produce: a wrong result, a crash, data loss or corruption,
a security hole, a broken contract with a caller or a persisted file, a
*measured* performance regression, a gate that cannot fail, or a claim in the
code or docs that the code contradicts. If you cannot write down the case that
breaks, you do not have a finding.

**Not findings — do not report these, at any confidence. Every one of them is
planning-stage material: if it is worth saying, say it at the next plan, not
here:**

- Structure and organisation preferences: "extract this", "split this file",
  "this belongs in another module", "invert this conditional", "this would read
  better as X".
- Naming, ordering, formatting, comment density, or idiom preferences.
- "Could be simplified", "is a bit repetitive", "consider using <pattern>" —
  absent a defect the duplication or complexity actually causes.
- Inconsistency with surrounding code, unless the inconsistency itself produces
  a defect.
- Speculative futures: "this won't scale", "this will be hard to extend", "what
  if someone later…", without a present, reachable failure.
- Performance opinions without a measurement. Reading a loop is not a
  benchmark; this project treats an unmeasured perf change as not a fix.
- Anything a formatter, linter, type checker or compiler already decides. Those
  have gates; a human-readable duplicate of them is noise.

**The one exception is a rule this project already declared.** Citing a
documented constraint — the god-file ceiling, the boundary principle, the
five-place `#[pymethods]` checklist, "no back-compat shims", the non-vacuous
gate rule — is legitimate *when you name the rule and the specific line that
violates it*. That is enforcing an agreed standard, not expressing taste. The
distinction is whether the rule existed before you read the diff — the same
stage test as above, applied to rules instead of designs: a constraint settled
in advance is enforceable at review; one you formed while reading is not.

**Severity is not a workaround.** Filing a preference as "minor" or "nit" does
not make it a finding; it makes it a preference with a label. Severity labels
are where preferences get laundered into review output — "minor: consider
extracting this" is not a small finding, it is not a finding. Drop it, or route
it to planning. A finding that cannot state its failure case is removed, never
downgraded.

**When you are unsure, apply the failing-input test.** Can you state a concrete
input, sequence, or state that produces a wrong outcome? Yes → report it, with
that case. No → say nothing here. **And if it fails the test but still bothers
you, it is planning input, not a finding** — hold it for the next plan's
investigation phase and raise it there. A reviewer's value is in the defects it
catches, and every preference in the list dilutes the ones that matter.

## Code health

Each pass through a file should leave it more compartmentalised than you found it.

- **No bugs left behind.** When you encounter a pre-existing bug while working — even one unrelated to your task — fix it in the same change, or if it's genuinely out of scope, surface it explicitly (file an issue / call it out) rather than silently stepping over it. Don't leave known bugs in the codebase. Before "fixing", confirm it's actually a bug and not deliberate behaviour: read the surrounding code and tests, check whether it's consistent across versions, and distinguish a real defect from an intentional design choice (e.g. the planner schema-check rejecting unknown CREATE properties is a deliberate typo-guard, not a bug). A measured performance change is only a "fix" if it measurably improves performance — never ship a perf change that doesn't.
- **A comment is a claim, and a false claim is a defect (R17).** A comment its own function contradicts is a bug of the same class as a wrong value — it is what the next maintainer, human or agent, acts on. Two standing duties keep the population true and lean *continuously*, so the tree never needs another whole-repo audit. (1) **A change that falsifies a nearby comment corrects that comment in the same change** — the falsehood does not exist until the code moves, and whoever moved it is the only party who knows it moved; repairing it later is an audit, and audits are the incident, not the mechanism. (2) **A change through commented code applies the information test to the comments it touches**: a comment earns its lines by information the reader cannot get from the code or an earlier comment — zero information (restating the next line or the signature, banners, narration of the journey) is deleted, low density is compressed to what it carries. The unit is information, not fact-count: an edit doctrine of "keep the fact, drop the label" moved 231 files by 0.2%, the information test moved 104 files by 12.4% while correctly sparing the specification files. Deletion has a floor — why-not-what, invariants and safety preconditions, lock ordering, data-format lifecycle, regression rationale in tests, bail reasons in planner/executor code (the differential corpus cannot catch that class), and anything under R18 below — kept regardless of how worthless they look. A comment that predicts a future ("a later phase will…") is a claim with an expiry date and nothing notices when it passes: word it so the work landing retires it, or don't write it. And a doc block is attached only by **adjacency**, which is editable — a `///` separated from its item by a blank line, or an item inserted mid-block, silently documents the *next* item while the compiler stays quiet and the renderer renders confidently. After inserting or moving items near a doc block, verify it in the **rendered** surface (`help()`, the generated header, the stubs), not the source. `/clean-comments` handles the residue; a heavy residue is itself the finding that the same-change duty is being skipped.
- **A comment the tooling parses is load-bearing — check what reads one before deleting it (R18).** Nothing at the comment site says so, and the reader is never discoverable from the comment itself. In this repo: `scripts/check_lint_allowances.py` requires every `#[allow]` to carry a `//` reason ≥ 12 chars within the 3 preceding lines — *including* a `///` doc comment, so a signature-restating one-liner can be the only thing keeping the gate green; clippy's `collapsible_if` / `collapsible_else_if` are suppressed by a comment's mere *presence* inside the block, so deleting the only comment there turns `-D warnings` red; `///` on `#[no_mangle]` fns in `crates/kglite-c` is copied verbatim by cbindgen into `crates/kglite-c/include/kglite.h`; and `#[pymethods]` docs are mirrored into `kglite/__init__.pyi`. The last two are published contracts answering to the C-ABI and five-place rules, not to comment hygiene. `/clean-comments` carries the maintained enumeration — extend it there when a new reader appears. Corollary: a checker whose identities derive from proximity is coupled to comment volume (ours keyed allowances by searching the next 2000 characters for a `fn`, and deleting ~45 comment lines silently re-keyed a `dead_code` allowance) — key such identities to the *nearest following item*, and until one is fixed, re-run it after any comment deletion.
- Factor a function when it grows past ~80 lines or starts handling 3+ unrelated concerns. Prefer small named strategy fns dispatched by the caller over long if/else chains.
- Fixing a bug — scan for the *class* of bug. The reported symptom is rarely the only one; probe with scratch fixtures before declaring scope.
- A new feature is a chance to extract a helper that's been wanted elsewhere. Don't over-design, don't pass it up either.
- Don't add a parameter/branch/flag without checking whether the existing structure should be reshaped to absorb it.

### Cypher planner passes

The optimiser pipeline lives at `crates/kglite/src/graph/languages/cypher/planner/mod.rs` as `const PASSES: &[(&str, PassFn)]` — single source of truth for order and naming. When adding or changing a pass:

1. **Implement** in the appropriate sub-module (`fusion/`, `simplification.rs`, …) or a new file for fresh concerns.
2. **Register** in `PASSES` with a unique stable name (user-facing via `disabled_passes=[...]`).
3. **Doc-comment** the wrapper fn: precondition, pattern matched, rewrite, why-bail.
4. **Add a query** to `tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES` exercising the trigger shape. Passes not in the corpus aren't trusted.
5. **Bisect divergences** with `scripts/cypher_pass_bisect.py` before assuming a query is wrong.

The differential corpus catches **optimizer divergence** — a pass producing a different result from the unoptimised path — so every fix to an optimiser bug lands its triggering query into the corpus as part of the fix commit. It cannot see a defect the two paths *share*: an executor or parser bug returns the same wrong answer with the passes on and off, and the comparison stays green. Those need absolute golden expected-value tests.

## Performance protocol

Before any perf-related change:

1. **Baseline first** — write/extend a benchmark covering touched code paths. Run it, record numbers.
2. **Build only the changed working tree, always in release mode.** Every benchmark and performance gate must use a release-built candidate (`uv run --no-sync maturin develop --release`, or the release-building `make bench*` target). Debug-profile performance results are invalid and must be discarded. This is a perf-only exception to the default-profile correctness-testing rule above.
3. **Install released/reference versions with `uv`.** Do not source-build another revision just to establish an A/B baseline. Create an isolated venv and install its published wheel, e.g. `uv venv <venv> --python 3.12 && uv pip install --python <venv>/bin/python 'kglite==0.14.2'`. Run the probe outside the repository root so the local `kglite/` package cannot shadow the installed wheel.
4. **Trust `min` over `median`** for sub-millisecond benches. Median pulls upward with system load; min reflects best-case throughput. **Two cell classes break this rule — judge them by mean/median and say in the write-up which statistic you used.** (a) *Once-per-event* costs: when the expensive work happens on the first call after a state change (the first write after a held view forks the store), `min` only ever reports the cheap repeats and is structurally blind to the thing being measured — use the mean of first-writes. (b) *Heavy-tailed* cells whose min sits 30%+ below their own median are reporting a lucky round, not a rate; `columnar_cypher_where` raised three false regression alarms on separate landings that way (2026-08).
5. **Tighten the harness for noisy benches**:
   - `--benchmark-min-rounds=100` (200 for sub-10-µs benches).
   - `--benchmark-warmup=on --benchmark-warmup-iterations=20`.
   - 30-second sleep between baseline and comparison runs (thermal settle).
   - Re-measure twice on the suspect commit. If runs disagree, you're seeing variance, not a regression.
   - **Retake any verdict that lands near its threshold.** A pass at 19% or a failure at 21% against a 20% gate is a coin-flip on machine load; one confirmation run settles which it was. A verdict far from its threshold needs no retake.
6. **In-memory is the gate.** Disk-mode benchmarks are nice-to-have but never at the cost of in-memory.
7. **Cumulative drift is gated too.** The per-release 20% gates recapture their
   baseline every release, so slow drift (~10%/release) never trips them.
   `make bench-anchor` compares the newest per-release baseline against the one
   ~3 releases back at +30% — run at release time (wired into the release
   skill). Per-release baseline files in `tests/benchmarks/baselines/` are the
   longitudinal record; never delete them. Because they are compared *across
   sessions*, every longitudinal capture (per-release baselines, `bench-anchor`
   inputs) notes the machine state it was taken under — metadata in the release
   commit or the capture record, never a gate on taking it. The 0.15.7 baseline
   was captured hot, nothing said so, and `bench-anchor` read the resulting
   offset as real drift.
8. **Read the distribution, and cross-check the instrument** (2026-08):
   - A **30×+ median-to-max spread on a deterministic operation** is a rare
     expensive branch, not noise. Distribution shape is a diagnostic — chase
     the outlier instead of averaging it away.
   - **Every capture carries unchanged-path control cells**; they are the
     machine-drift meter. A *control* that regresses means the instrument
     moved, not the code — re-measure, don't bisect.
   - **A control needs margin over its own resolution, and immunity from what
     it anchors** (doctrine R11 corollary, 2026-08-13). Require the control's
     median at **≥2×** the capture's noise floor — a control at ~1× the floor
     measures nothing — and prefer the slowest query *measured* stable across
     the last dependency move: a control chosen because "our source can't
     touch it" silently expires when the thing that moves is a dependency.
     A control that moves **deterministically** across repeated re-measures is
     not instrument wander — "re-measure" cannot resolve it; the control's
     premise is void and the control must be replaced.
   - **Measure a claim two independent ways** (two holders, two call routes)
     before believing it. A `WHERE` clause silently disqualifying the lazy path
     was invisible to either route alone and showed up only as disagreement
     between them; a single measurement cannot detect its own instrument bug.
9. **Run benchmarks under the load the machine has** (relaxed 2026-08-09). The
   old "otherwise-idle machine" precondition cost far more in stalled work and
   stretched development time than it ever bought in precision — waiting for
   quiet is not a step. Measure now, under whatever else is running, and accept
   the wider uncertainty: the control cells in item 8 are the drift meter, two
   agreeing runs are the confirmation, and a threshold-adjacent verdict gets one
   retake. The one thing machine load still changes is the *record* — see item 7
   for longitudinal captures.

## Key patterns

- **PyO3**: `&self` for read-only methods; return `PyResult<Py<PyAny>>`; wrap blocking work in `Python::attach()`. Use `.cast::<T>()`, not `.downcast::<T>()` (deprecated in pyo3 0.27+).
- **`#[pymethods]` location**: all method blocks live under `crates/kglite-py/src/graph/pyapi/`. Private helpers stay in `crates/kglite-py/src/graph/mod.rs` as `pub(crate)`. The `#[pyclass]` *struct attribute* may stay with the struct definition.
- **Value conversion**: `py_out::value_to_py()` and `py_out::nodeinfo_to_pydict()`.
- **Storage traits**: reads on `GraphRead`, mutations on `GraphWrite: GraphRead` (both in `crates/kglite/src/graph/storage/mod.rs`). Add new storage ops to the trait first. `GraphRead` is non-object-safe (GATs on iterator methods) — use `&impl GraphRead` everywhere, never `&dyn`. Iterator-returning trait methods declare an associated type (`type FooIter<'a>: Iterator<…> where Self: 'a;`).
- **Transactions stay on `DirGraph`**, not in the trait surface (`version`, `read_only`, `schema_locked`, validation helpers).
- **No back-compat shims, no `#[deprecated]` — this is about *code/APIs*, not *data*.** Obsoleted code/API paths are deleted in the same PR as their replacement: no deprecated public surface, no dual old-vs-new-API codepaths, no compat wrappers for renamed/replaced functions. **Data-format compatibility is a separate, legitimate concern and is NOT a "shim".** Persisted files (`.kgl`, disk graphs) outlive the binary that wrote them, so *reading* an older on-disk/serialized format (read-compat), or *detecting* one and refusing it with a clear "rebuild your graph" message (a deliberate hard-break, e.g. the `.kgl` v3→v4 break or the embeddings-provenance break), is expected format-lifecycle handling — keep or migrate it, don't delete it to satisfy this rule. The test when unsure: *would deleting this break a caller's **code** (shim → remove) or an existing user's **saved file** (data-compat → keep/migrate)?* Examples that are NOT shims and stay: `EdgePropertyStore` legacy-format detection, `ConnectionTypeInfo`'s old-field deserializer, the v3-magic rejection in `io/file.rs`.
- **The published C ABI is additive-only within a major.** Before changing any exported signature in `crates/kglite-c`, diff it against the shipped header — `git show vX.Y.Z:crates/kglite-c/include/kglite.h` — because a signature that has been published never changes within an ABI major; new behaviour arrives as a *new symbol*. Prebuilt consumers link the old one, so this is data-compat's sibling, not a shim. Full rules in `docs/rust/c-abi.md`.
- **Parity oracles** at `tests/test_storage_parity.py`, `tests/test_phase{1,2,3}_parity.py` (gated by `pytest -m parity`) must stay green after any backend-touching change.

## When changing a `#[pymethods]` function — the five-place checklist

1. `crates/kglite-py/src/graph/pyapi/*.rs` — implementation.
2. `kglite/__init__.pyi` — type stub + docstring.
3. `crates/kglite/src/graph/introspection/*.rs` — `describe()` output, if agent-facing.
4. `crates/kglite-mcp-server/src/tools/` — MCP tool wrapper, if agent-facing (router wiring in `tools/register.rs`; the pre-split `tools.rs` monolith is gone).
5. `CHANGELOG.md` `[Unreleased]` — user-visible changes only.

**Which docstring says what — one line in Rust, the contract in the stub.** The
same method is documented in three places with three readers, and they are not
copies of each other. The Rust `///` on the `#[pymethods]` fn is a **one-line
summary only** — its readers are `help()`/`pydoc` at an interactive prompt and
rustdoc, so it is plain prose (convert Sphinx markup: ``x`` → `x`,
:func:`f` → `f()`, :class:`C` → `C`) and never repeats Args/Returns.
`kglite/__init__.pyi` carries the **full contract** — signature, Args, Returns,
Raises, examples — and remains the source of truth the published API docs are
generated from. `introspection/topics.rs` carries an **independent
agent-facing one-liner** written for `describe()`, not derived from either.
A missing `///` is a defect (an empty `help()`), but so is a `///` that
duplicates the stub: it will drift, and the stub is the one that ships.
Watch the adjacency — the `///` goes above the `#[pyo3(...)]` attributes with
no blank line before the item, and it must never become the nearest `//`
comment above an `#[allow]` (see R18: that would satisfy the allowance gate's
reason requirement vacuously).

## Documentation

Docs auto-rebuild at [kglite.readthedocs.io](https://kglite.readthedocs.io) on every push to `main`.

- **API reference**: auto-generated from `kglite/__init__.pyi` docstrings.
- **Cypher reference**: `CYPHER.md`.
- **Fluent API reference**: `FLUENT.md`.
- **Guide content**: `docs/python/guides/*.md`.
- **README.md**: landing page only — don't duplicate guide content.

## dev-docs steers the sprint; commits are the durable record

`dev-docs/todos.md` is read at the start of every phase and by every steering
agent, so detail there is load-bearing — an entry recording what was tried,
what was rejected and why, stops a fresh agent burning a phase relitigating a
settled decision. The test is "would an agent act differently for having read
it?", not length. Entries whose action has shipped are dead weight; prune those.

It is gitignored and unbacked, so anything that must survive the machine also
goes somewhere tracked: the commit message that implements it, a comment at the
code it constrains, or here. And never cite a `dev-docs/` path from committed
code — the citation outlives the file, silently.

## Inbox hygiene

`inbox/unread/` (at the repo root) holds incoming feedback/bug/coordination
notes (named `YYYY-MM-DD-from-<sender>-<topic>.md`); `inbox/read/` is the
archive. The inbox is gitignored (`/inbox/`) — it's local working state, not
committed.

**Triage before archiving.** Preserve durable evidence and track unresolved
work through `read-inbox` and `add-todo`. Archival means triage is recorded, not
that every action shipped. Verify destination records, append a completed Status
record with an explicit UTC archive time, then move without overwriting. Keep
messages with no durable owner unread. Purge only completed, verified records
whose archive-based grace period expired; retain legacy/unmarked records.

**Route to the party who can act.** A note only belongs in another project's
inbox (e.g. `../mcp-methods/inbox/`, `../../mcp-servers/inbox/`) if it carries an
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
   ("post it", "yes"). That authorization remains valid for the same draft
   and action until completed, revoked or materially changed; intervening work
   alone does not require another approval.
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
not this section. Local inbox notes also require authorization for their
recipient and purpose; a review or incoming message alone does not authorize
sending. Existing authorization need not be repeated.

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

**How this interacts with `/release`.** Invoking the skill authorizes the
entire release run, **including the `main` push that fires the publish**. No
separate prompt. The run still *reports* — version, semver-check findings, perf
numbers, anything it learned that the user did not know at invocation — but
immediately before pushing, not as a gate on it.

That distinction was got wrong on 2026-07-30 and corrected on 2026-07-31. Making
the report a blocking confirmation sounded safer and was not: it fired *after*
the irreversible decision had already been made, so it added no information to
the choice, and it broke unattended releases — 0.15.4 sat at a staged commit
while the user was away, and they noticed it had not landed before the agent
did. The safety that matters is upstream and unchanged: green branch CI, the ten
`release-preflight` preconditions, refreshed constants, artifact-set
verification, surgical staging, ff-merge clean. Those can fail. A prompt cannot.

**Exception — the CI fix-and-push loop.** When an approved push triggers CI that fails, and you diagnose the failure as a bug in shipped code or test/CI infra (not a feature gap), you may push subsequent `fix(...)` / `ci(...)` commits *for that same loop* without re-asking, until CI on the most recent push is fully green. This covers the common case where the first push surfaces a flaky dep / missing fixture / linter-only issue and you'd otherwise need to ping the user every iteration just to type "push" again.

The exception **stops applying** the moment any of these are true:
- All required workflows on the latest push reach `conclusion: success` → loop converged, fresh approval needed for the next push
- A fix would change the release shape (new version, new feature, scope expansion, removal of declared functionality) → ask, don't push
- More than ~3 fix-and-push iterations happen on the same loop without progress → likely a deeper problem, surface it and ask
- The user pivots the conversation away from the CI loop → context shift means fresh approval needed

The loop's pushes are still subject to the same rigor as any release push (lint clean, tests green, dry-runs pass before pushing). The exception removes the "ask first" step, not the "build with care" step.

Conversational phrasing from earlier in the session ("ship it", "looks good", "you may push", "we're ready") **does not** carry over to a later moment outside the fix-and-push loop, even within the same turn if other actions intervene. When in doubt, prepare the commit, stop, and ask. The cost of a re-prompt is small; an unapproved push to `main` is not.

Version source of truth: **`[workspace.package] version` in the root
`Cargo.toml`**. Every member crate (engine, wheel, `kglite-c`, servers, cli)
sets `version.workspace = true` and inherits it, so all published crates ship
in lockstep.

**But the version lives in five files, not one.** `[workspace.package]` covers
each crate's *own* `package.version`; it does not cover the internal
dependency requirements. Four member manifests — `kglite-bolt-server`,
`kglite-c`, `kglite-cli`, `kglite-mcp-server` — declare
`kglite = { version = "X.Y.Z", path = "../kglite", … }`, because `cargo
publish` rejects a `path`-only dependency. Bumping only the workspace table
leaves the workspace unresolvable as soon as the bump crosses a minor
(`cargo metadata`: *failed to select a version for the requirement
`kglite = ^0.14`*), and every release step that shells out to cargo dies on
its first call — that is exactly what broke `make refresh-release-constants`
during the 0.15.0 release.

So: **never hand-edit the version. Run `make bump-version VERSION=X.Y.Z`.**
It rewrites all five places and verifies with a resolving `cargo metadata`
that every member resolved (`--no-deps` skips resolution entirely and passes
on exactly the broken tree). The internal requirement carries the full `X.Y.Z`
rather than the `X.Y` series, because these crates ship in lockstep and this
project deliberately ships breaking engine changes in patch bumps — `^0.15`
would advertise that `kglite-cli 0.15.3` builds against `kglite 0.15.0`, which
is often false. `make gate` fails if any of the five drifts apart.

**The bump size is always patch unless the release command said otherwise.**
`/release` with no size means `x.y.Z+1`, with no clarification prompt — a
breaking change is not a reason to stop and ask, because this project
deliberately ships documented breaking changes in patch bumps, so asking
spends the user's attention re-confirming a default they already set. A minor
or major happens only when the invocation specified one (`/release minor`,
`/release 0.16.0`). `make semver-check` still runs, and its findings still get
quoted — into the CHANGELOG and the downstream notes. It is evidence for what
to *write*, never a gate on what to *number*. The target only applies the
version you hand it.

Release-run procedure — captured-constant refresh, preflight and CI polling,
ecosystem version consistency, and PyPI capacity — lives in the `release`
skill, which loads when you invoke it. It is not repeated here.

**One prohibition from it stays resident, because it is irreversible and
must not depend on a skill being loaded: never delete published files from
PyPI or crates.io.** Published artifacts are never removed automatically, and
any manual deletion permanently breaks every pinned install that depends on
them — it requires a downstream-impact audit and explicit approval first.
### One version bump per push

A version isn't "released" until the user pushes. If a `release(x.y.z): ...` commit is already local, fold any follow-up work into the same `[x.y.z]` CHANGELOG block — amend or extend the release commit, don't add a new `release(x.y.z+1): ...` on top.

Check before bumping:

```bash
git log origin/main..HEAD --oneline | grep -E "^\w+ release\("
```

If that returns a commit, keep the version it picked. Only mint a new version after a clean push to origin.

### Multi-phase plans

When a plan has Steps 1 / 2 / 3 / …:

1. **One commit per phase.** Bisectability beats batched commits. Each phase's code + tests in its own `feat:` / `refactor:` / etc.
2. **Each phase must be green before its commit** — `make gate` and the targeted suites that would catch what the phase could break (its surface + that surface's direct consumers). A targeted test already compiles its target, so do not add a redundant build or workspace-wide clippy run. Never use workspace-root `cargo build --lib` as the generic phase build. The full battery runs once over the plan's union at the end, per "Testing discipline".
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
   the completed plan. **Run `make gate-push` once immediately before each
   push** — it adds the CI-only checks that historically reddened first in CI
   (public-API baselines, source quality, lint allowances, workspace clippy,
   the `RUST_STABLE` toolchain-pin match). Per push, not per phase; a red
   there costs seconds locally and a full 20-job round-trip in CI.
6. **End with a perf gate — only if the plan touched perf-sensitive paths** (`core/pattern_matching/`, `cypher/executor/`, storage hot paths). Then run new + existing benchmarks per the Performance protocol above before the final release commit, and record the numbers in the release commit message or `[x.y.z]` CHANGELOG block. Fix regressions before the release commit, not in a follow-up. A plan that touched none of those paths skips this — CI's Linux perf gate and the release-time baseline capture cover it.
7. **Shipping is a separate release request.** Implementation phases may edit manifest dependencies/features, but do not bump package versions or promote the release CHANGELOG. The release request authorizes its publish push.
