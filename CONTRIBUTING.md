# Contributing to KGLite

## Development setup

KGLite is a Rust workspace with a PyO3 wheel. Use a virtual environment so the
extension under test is always the one built from your checkout.

```bash
git clone https://github.com/kkollsga/kglite.git
cd kglite
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest pandas hypothesis networkx neo4j ruff mypy
pip install -r requirements/coverage.txt
cargo install cargo-llvm-cov --version 0.8.7 --locked
maturin develop --release
```

Release builds are required for performance measurements. Ordinary correctness
work may use `make dev`, but rebuild release mode before trusting timings.

The generated [project facts](https://github.com/kkollsga/kglite/blob/main/docs/_generated/project-facts.md) list the exact
workspace members, supported Python metadata, active wheel targets, storage
modes, and captured benchmark environment. Refresh them with `make docs-facts`;
`make check-docs-facts` fails if they drift.

## Repository structure

```text
crates/kglite/                 Rust engine and shared API
  src/graph/core/              Shared matching/filtering/traversal primitives
  src/graph/languages/cypher/  Tokenizer, parser, planner, executor
  src/graph/storage/           Memory, mapped, and disk backends
  src/code_tree/               Rust tree-sitter parsers and graph builder
crates/kglite-py/              PyO3 wrapper
  src/graph/pyapi/             All #[pymethods] blocks
crates/kglite-c/               C ABI and generated public header
crates/kglite-mcp-server/      MCP adapter and bundled server
crates/kglite-bolt-server/     Bolt adapter
crates/kglite-cli/             Command-line client
kglite/                        Python package, stubs, and thin helpers
tests/                         Python, parity, contract, and benchmark suites
docs/                          Sphinx/MyST documentation
```

Read [the boundary principle](https://github.com/kkollsga/kglite/blob/main/docs/rust/boundary-principle.md) before changing
`kglite::api`, the C ABI, or adding a binding. Reusable graph behavior belongs
in the core. PyO3 values, protocol wire values, async runtimes, and display
behavior stay in their wrapper.

## Build and test commands

```bash
make test        # Rust + default Python markers
make test-full   # Rust + Python parity and Bolt markers
make lint        # local format, lint, clean-room, license, and stub gates
make cov         # pinned line + branch coverage for Python production modules
make cov-rust-core  # report-only default-feature kglite core coverage
make source-quality  # production-only structure and complexity ratchet
```

The default Python configuration skips benchmark, parity, stress,
model-download, binary-size, Bolt, and Bolt-stress markers. `make test-full`
adds parity and Bolt while keeping expensive/host-specific markers excluded.

Python coverage uses the versions pinned in `requirements/coverage.txt` and
the settings in `pyproject.toml`. The report measures `kglite/` production
modules with branch coverage, excludes package-local test fixtures under
`kglite/**/tests/`, and uploads the Python component separately from Rust.

Rust coverage is intentionally component-scoped. `make cov-rust-core` measures
the `kglite` core crate's library and tests with default features and writes
`rust-core-coverage.lcov`; CI uploads it under the `rust-core` flag without a
threshold. It does not claim coverage for the PyO3 Rust wrapper, C ABI,
MCP/Bolt/CLI crates, or optional dataset, RDF, parallel-decoder, and embedder
features. Those components need their own representative harnesses before
their denominators can be reported honestly.

`make source-quality` is the single production-source structural gate. It
checks Rust file size, backend enum dispatch, unsafe-block justification,
module-file caps, required recording symbols, and per-function line/branch/
nesting growth. Existing complex functions are identified in
`tests/api-baselines/source-quality.json`; improvements must tighten that
baseline with `python scripts/check_source_quality.py --refresh-functions`.
The dedicated CI job runs the analysis once rather than repeating it in every
Python-version leg.

Rust `#[allow(...)]` attributes are separately inventoried by stable
module/item + lint identity in `tests/api-baselines/lint-allowances.json`.
The inventory classifies API-shape, binding-boundary, test, transitional, and
dead-code cases; dead code remains a separate list so style cleanup cannot
hide unused implementation. A new allowance needs a nearby explanatory
comment and an explicit `python scripts/check_lint_allowances.py --refresh`;
deleting an allowance also tightens the exact baseline. To re-audit a broad
suppression, use Clippy's `--force-warn <lint>` so crate/module `allow`
attributes cannot hide its current trigger sites.

CI also checks surfaces not covered by `make lint`:

- public Rust API against `tests/api-baselines/kglite.txt` on the pinned nightly;
- `kglite-c` all-feature clippy/tests and generated-header drift;
- storage parity, disk concurrency, Loom, Miri, sanitizers, free-threading,
  binary size, and release performance contracts.

Run the focused suite for the code you touched before the broad gates. Storage
changes must keep `tests/test_storage_parity.py` and the phase parity suites
green. Planner changes must add a triggering query to
`tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES` and register their
stable pass name in the planner's `PASSES` table.

## Python API changes

When a `#[pymethods]` function changes, inspect all five consumers:

1. implementation under `crates/kglite-py/src/graph/pyapi/`;
2. `kglite/__init__.pyi`, which is the API-doc source of truth;
3. agent introspection under `crates/kglite/src/graph/introspection/`;
4. MCP tool exposure in `crates/kglite-mcp-server/src/tools.rs`;
5. `[Unreleased]` in `CHANGELOG.md` when users can observe the change.

The project removes obsolete code APIs with their replacement. Do not add
deprecated aliases or compatibility wrappers. Persisted data is different:
keep a read-compatible path or detect an incompatible format and return a
specific rebuild/migration error.

## Code health

- Leave a touched file more compartmentalized than you found it. Extract
  functions that exceed roughly 80 lines or mix three unrelated concerns.
- Confirm a suspected defect against current source, docs, and tests before
  changing it. When fixing one instance, scan for the same bug class.
- Preserve in-memory performance when large-graph storage needs a special
  path. Benchmark the default mode before merging shared executor/planner work.
- Rust formatting and Clippy, Ruff format/check, and stubtest are enforced.
- Do not refresh a baseline merely to hide a regression.

## Documentation

API reference is generated from `kglite/__init__.pyi`. Cypher and fluent
references live in `CYPHER.md` and `FLUENT.md`; guides live under `docs/`.
Keep `README.md` as a landing page rather than duplicating guides.

Machine-derived facts belong in `docs/_generated/project-facts.md`, generated
from authoritative files. Prefer executable behavior tests for semantics and
link the prose to those contracts instead of copying implementation details.

## Commits and changelog

Use `type: short description`, where the usual types are `feat`, `fix`,
`docs`, `refactor`, `test`, and `chore`. Keep messages mechanical and public:
describe what the diff does, not sensitive motivation.

Add user-visible changes to the top `[Unreleased]` section of `CHANGELOG.md`.
Internal refactors, CI-only work, tests, and formatting do not need entries.

For a multi-phase change, make one green commit per phase. Each phase must pass
`cargo build --lib`, its focused tests, and `make lint` before committing.

## Maintainer release checklist

The workspace version in the root `Cargo.toml` is the single version source.
A release commit promotes `[Unreleased]`, bumps that one version, and refreshes
the version-coupled constants:

```bash
maturin develop --release
make refresh-release-constants
make test-full
make lint
```

Review retained-format, interface, and performance gates before the release
commit. A push to `main` triggers CI-gated crate and wheel publication; never
publish a new version merely to account for unpushed follow-up work.
