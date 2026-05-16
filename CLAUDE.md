# KGLite — Claude Code Conventions

## Build & Test

```bash
source .venv/bin/activate && unset CONDA_PREFIX
maturin develop              # build Rust extension into venv
make test                    # Rust + Python tests
make lint                    # fmt --check + clippy (always run before pushing)
```

## Architecture

- **Rust core** (`src/`): `KnowledgeGraph` with `#[pymethods]` via PyO3, `petgraph` storage.
- **Cypher engine** (`src/graph/languages/cypher/`): parser → AST → executor. Phase 8 moved this under the `languages/` umbrella; a future `languages/fluent/` is a peer for the Rust-side fluent-chain surface.
- **Shared query primitives** (`src/graph/core/`): pattern matching, filtering, traversal, iterators — used by Cypher and the fluent API.
- **Python package** (`kglite/`): thin wrapper + `code_tree/` (tree-sitter codebase parsing).
- **Type stubs** (`kglite/__init__.pyi`): source of truth for API docs — update when changing `#[pymethods]`.
- **Introspection** (`src/graph/introspection.rs`): `describe()` XML schema for AI agents.

## Storage Modes & Optimization Priority

KGLite has three storage modes: `Default` (in-memory petgraph), `Mapped` (mmap-backed columns), and `Disk` (CSR + mmap). **In-memory is the core product.** Disk/mapped are addons for large-graph exploration (e.g., Wikidata).

**When optimizing, in-memory wins every time.** If an optimization helps both in-memory and disk, great. If there's a conflict (e.g., adding overhead to the hot path to protect against disk-scale explosions), find a disk-specific workaround instead. Never regress in-memory performance for disk safety.

- Small graphs (legal, code, domain): ~100K–500K nodes, all in-memory, full scans are fast (<10 ms). Don't add guardrails that penalize these.
- Large graphs (Wikidata): 100M+ nodes, disk-backed, full scans are catastrophic. Safeguards needed but must be gated behind storage mode or graph-size thresholds.
- Cypher query planner/executor is shared across all modes. Any changes to `core/pattern_matching.rs` or `languages/cypher/executor.rs` affect everyone — benchmark on small graphs before merging.

## Code health — incremental compartmentalization

Each pass through a file should leave it more robust, more compartmentalized,
and easier to extend than you found it. The codebase should get **better** as
we work, not just bigger.

- When a function grows past ~80 lines or starts handling 3+ unrelated
  concerns, factor it before adding more. Prefer a list of small named
  strategy fns dispatched by the caller over one long if/else chain — adding
  a new case becomes one new fn, not a new branch in a god function.
- When fixing a bug, scan for the *class* of bug — same root cause in nearby
  code paths. The reported symptom is rarely the only one. Probe with scratch
  fixtures before declaring scope.
- A new feature/fix is a chance to extract a helper that's been wanted
  elsewhere. Don't over-design, but don't pass up the opportunity either.
- Don't add a parameter, branch, or flag without checking whether the
  existing structure should be reshaped to absorb it cleanly.

### Cypher planner: when adding or changing a pass

The optimizer pipeline lives at `src/graph/languages/cypher/planner/mod.rs`
as a `const PASSES: &[(&str, PassFn)]` registry — one source of truth
for ordering, naming, and what runs in what order. When adding or
changing a pass:

1. **Implement** the pass in its appropriate sub-module (`fusion.rs`,
   `simplification.rs`, etc.) or a new file if it's a fresh concern.
2. **Register** it in `PASSES` with a unique stable name (the name is
   user-facing via `disabled_passes=[...]`).
3. **Doc-comment** the wrapper fn with: precondition, pattern matched,
   rewrite, why-bail. The doc-comments next to the existing wrappers
   are the format to follow.
4. **Add at least one query** to
   `tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES` that
   exercises the pass's trigger shape. The differential harness is the
   regression mechanism — passes not in the corpus aren't trusted.
5. **Bisect any divergence** the harness reports with
   `scripts/cypher_pass_bisect.py` before assuming a query is wrong.

The differential corpus is *the* mechanism to prevent silent
correctness regressions. Every fix to an optimizer bug lands its
triggering query into the corpus as part of the fix commit.

## Performance Work Protocol

Before starting any performance-related code changes:
1. **Baseline first** — write a benchmark covering all code paths being touched. Run it, record numbers.
2. **Benchmark on in-memory graphs** — small/medium graphs (legal-scale) must not regress. This is the gate.
3. **Measure after** — re-run the same benchmark after changes. Report before/after.
4. **Disk benchmarks are secondary** — nice to show improvement, but never at the cost of in-memory.

## Key Patterns

- PyO3: `&self` for read-only, return `PyResult<Py<PyAny>>`, use `Python::attach()`.
- Use `.cast::<T>()` not `.downcast::<T>()` (deprecated in pyo3 0.27+).
- **All `#[pymethods] impl` blocks live under `src/graph/pyapi/`.** Private `impl KnowledgeGraph` helpers stay in `src/graph/mod.rs` (pub(crate) when called from pyapi). The `#[pyclass]` *struct attribute* may stay with the struct definition where the data model lives — this applies to `KnowledgeGraph` itself (in `src/graph/mod.rs`) and `ResultView` (in `src/graph/languages/cypher/result_view.rs` / `pyapi/result_view.rs`). The rule is about where PyO3 *method implementations* live, not struct declarations.
- Value conversion: `py_out::value_to_py()` and `py_out::nodeinfo_to_pydict()`.

## Storage-backend work

- **Reads go on `GraphRead`, mutations on `GraphWrite: GraphRead`.** Add new storage ops to the trait first, not as inherent `GraphBackend` methods. Both traits live in `src/graph/storage/mod.rs`.
- **Transactions stay on `DirGraph`.** OCC `version`, `read_only`, `schema_locked`, and validation helpers are not trait surface — they're DirGraph-inherent.
- **No shims / no `#[deprecated]`.** Obsoleted code gets deleted in the same PR as its replacement. Clean break for 0.8.0.
- **`&impl GraphRead` / `&mut impl GraphWrite` everywhere.** Phase 3 added GATs to every iterator-returning method on `GraphRead`, which makes the trait non-object-safe. `&dyn GraphRead` does not compile; use `&impl GraphRead` (monomorphised) for every consumer. Iterator-returning methods must declare an associated type (`type FooIter<'a>: Iterator<…> where Self: 'a;`).
- **Parity oracles**: `tests/test_storage_parity.py`, `tests/test_phase1_parity.py`, `tests/test_phase2_parity.py`, `tests/test_phase3_parity.py`. Gated behind `pytest -m parity`. Must stay green after any backend-touching change.

## When Changing a `#[pymethods]` Function

1. `src/graph/pyapi/kg_methods.rs` (or domain file under `pyapi/`) — implementation
2. `kglite/__init__.pyi` — type stub + docstring
3. `src/graph/introspection/*.rs` — `describe()` output (if agent-facing)
4. `crates/kglite-mcp-server/src/tools.rs` — MCP tool wrapper (if agent-facing)
5. `CHANGELOG.md` — `[Unreleased]` section

## Documentation

Docs auto-rebuild at [kglite.readthedocs.io](https://kglite.readthedocs.io) on every push to `main`.

- **API reference**: auto-generated from docstrings in `kglite/__init__.pyi`
- **Cypher reference**: edit `CYPHER.md`
- **Fluent API reference**: edit `FLUENT.md`
- **Guide content**: edit `docs/guides/*.md`
- **README.md**: landing page only — do not duplicate guide content here

## Commits & Releases

Commit format: `type: short description` (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`)

Update `CHANGELOG.md` `[Unreleased]` for user-visible changes. Skip for internal refactors, CI, test-only, formatting.

**NEVER push — the user pushes manually.** Before a release:
1. Confirm version number with user (bump patch +0.0.1)
2. Update `Cargo.toml` version + promote changelog
3. Commit, then let the user push

Version source of truth: `Cargo.toml` line 3.

### Standard plan procedure (multi-phase work)

When a plan has multiple phases (Step 1 / 2 / 3 / …), follow this rhythm:

1. **Commit after each phase.** Each completed phase is its own commit
   (`feat: ...`, `refactor: ...`, etc.). Don't bundle Phase 1's code with
   Phase 2's tests — that defeats bisectability when something regresses.
2. **Version bump and push happen together, at the very end.** Don't bump
   `Cargo.toml` after each phase. After all phases land as separate
   commits, the FINAL commit promotes `[Unreleased]` → `[x.y.z]` in the
   CHANGELOG and bumps `Cargo.toml`. The user pushes once.
3. **Each phase must compile + lint + test green before its commit.**
   `cargo build --lib`, `make lint`, and the relevant test suite all pass
   before phase commit; if a phase is "scaffolding only" (e.g. adding
   pyo3 classes that aren't called from Python yet), at minimum the
   build + lint gate still applies.
4. **Don't pause for review between phases — keep going to the end.**
   Once the plan is approved, execute every phase straight through:
   commit, mark the task done, start the next phase. The final commit
   is the version bump + push. Don't ask "should I continue?" after a
   phase — that defeats the point of an approved plan. The only reason
   to stop mid-plan is a genuine blocker (failing test you can't fix,
   architectural surprise that invalidates a later step).

Rationale: clean per-phase commits make `git bisect` useful when the
operator (or CI) flags a regression days later. Bundling phases into one
big commit is the wrong default — the only exception is when phases are
genuinely indivisible (e.g. a rename that touches both definition and
all call sites).
