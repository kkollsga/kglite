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

**NEVER push — the user pushes manually.** Version source of truth: `Cargo.toml` line 3.

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
2. **Each phase must be green before its commit** — `cargo build --lib`, `make lint`, and the relevant test suite all pass.
3. **Keep going to the end.** Once a plan is approved, don't pause between phases. The only mid-plan stops are genuine blockers (failing test you can't fix, architectural surprise invalidating a later step).
4. **End with a perf gate.** Before the final release commit, run new + existing benchmarks per the Performance protocol above. Record numbers in the release commit message or `[x.y.z]` CHANGELOG block. Fix regressions before the release commit, not in a follow-up.
5. **Final commit is the version bump + CHANGELOG promotion.** No earlier phase touches `Cargo.toml`. User pushes once.
