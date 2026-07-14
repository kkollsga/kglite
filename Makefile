# kglite development tasks
# All targets handle CONDA_PREFIX and .venv activation automatically.

SHELL := /bin/bash
ACTIVATE := unset CONDA_PREFIX && source .venv/bin/activate

.PHONY: dev dev-with-bin bundle-bin test test-full test-rust test-py bench bench-save bench-compare bench-check bench-check-v090 refresh-release-constants refresh-api-baseline docs-facts check-docs-facts neo4j-up neo4j-down neo4j-conformance bolt-conformance check clean fmt fmt-py clippy lint lint-py cov stubtest

## Build and install the package into the local .venv
dev:
	$(ACTIVATE) && maturin develop

## Dev install with the bundled `kglite-mcp-server` binary on PATH.
## Builds the binary via cargo, copies it into kglite/_bin/, then runs
## `maturin develop`. The same sequence is what CI runs at wheel-build
## time. Use this if you want `which kglite-mcp-server` to resolve to
## the wheel-installed binary during local development.
dev-with-bin: bundle-bin
	$(ACTIVATE) && maturin develop --release

## Build the kglite-mcp-server binary and copy it into kglite/_bin/.
## Idempotent. Used by `dev-with-bin` locally and by the wheel-build
## workflow in CI.
bundle-bin:
	cargo build --release -p kglite-mcp-server
	mkdir -p kglite/_bin
	cp target/release/kglite-mcp-server kglite/_bin/kglite-mcp-server

## Run all tests (Rust + Python)
test: test-rust test-py

## Run Rust unit tests only
test-rust:
	$(ACTIVATE) && cargo test

## Run Python tests only (excludes benchmarks)
test-py:
	$(ACTIVATE) && pytest tests/ -v

## Full local suite: Rust tests + Python tests INCLUDING the parity and
## bolt marker suites that `make test` skips (pyproject addopts deselects
## benchmark/parity/stress/model_download/binary_size/bolt/bolt_stress).
## Still excludes benchmark (needs pytest-benchmark), stress (30GB-scale),
## model_download (multi-GB weights), binary_size (needs the release
## cdylib) and bolt_stress (slow, opt-in). The bolt tests skip silently
## unless target/release/kglite-bolt-server exists — build it first via
## `cargo build --release -p kglite-bolt-server`.
test-full: test-rust
	$(ACTIVATE) && pytest tests/ -v -m "not benchmark and not stress and not model_download and not binary_size and not bolt_stress"

## Run performance benchmarks (forces release build — saved baselines
## are release-built, so a dev-profile comparison shows ~15× false
## regressions across every test).
bench:
	$(ACTIVATE) && maturin develop --release --quiet && pytest tests/benchmarks/ -v -m benchmark -s

## Save benchmark baseline for comparison (release build).
bench-save:
	$(ACTIVATE) && maturin develop --release --quiet && pytest tests/benchmarks/test_bench_core.py -m benchmark --benchmark-save=baseline

## Compare current performance against saved baseline (release build).
bench-compare:
	$(ACTIVATE) && maturin develop --release --quiet && pytest tests/benchmarks/test_bench_core.py -m benchmark --benchmark-compare

## Compare against the v0.9.0 baseline and fail on >5% mean regression (release build).
## Run after each 0.9.0 gate-item lands to enforce the no-regression rule.
bench-check-v090:
	$(ACTIVATE) && maturin develop --release --quiet && pytest tests/benchmarks/test_bench_core.py -m benchmark --benchmark-compare=v0_9_0_baseline --benchmark-compare-fail=mean:5%

## Perf regression gate: compare the tracked core benchmarks against
## the current platform's baseline and fail on >20% regression on
## `min` time. This is the gate CI runs (on Linux); local
## developers usually hit the macOS arm. Baselines are platform-
## specific because Linux GitHub runners are ~2-3x slower than
## Apple Silicon for these benchmarks (same source, different
## hardware).
## Refresh the baseline at release time via `make refresh-release-constants`.
bench-check:
	$(ACTIVATE) && maturin develop --release --quiet \
		&& pytest tests/benchmarks/test_bench_core.py -m benchmark \
			--benchmark-min-rounds=100 --benchmark-warmup=on --benchmark-warmup-iterations=20 \
			--benchmark-json=.bench-current.json \
		&& BASELINE=tests/benchmarks/baselines/current$$( [ "$$(uname)" = "Linux" ] && echo ".linux" )$$( [ "$$(uname)" = "Darwin" ] && echo "" ).json \
		&& python scripts/compare_bench.py $$BASELINE .bench-current.json \
			--metric min --threshold 20

## Refresh the three captured constants that drift across releases:
## the .kgl golden digest, the binary-size baseline, and the perf
## baseline. Run as part of every release commit — see CLAUDE.md
## under "Captured-constant refresh at release time".
refresh-release-constants:
	$(ACTIVATE) && maturin develop --release --quiet
	$(ACTIVATE) && python scripts/refresh_release_constants.py

## Refresh the committed public-API baseline (tests/api-baselines/kglite.txt)
## via cargo-public-api on the pinned nightly. Keep KGLITE_API_NIGHTLY in sync
## with the public-api job in .github/workflows/ci.yml. One-time setup:
##   rustup toolchain install $(KGLITE_API_NIGHTLY)
##   cargo install cargo-public-api --locked --version 0.49.0
KGLITE_API_NIGHTLY ?= nightly-2026-07-01
refresh-api-baseline:
	RUSTUP_TOOLCHAIN=$(KGLITE_API_NIGHTLY) cargo public-api -p kglite -ss > tests/api-baselines/kglite.txt
	@echo "refreshed tests/api-baselines/kglite.txt ($(KGLITE_API_NIGHTLY))"

docs-facts:
	$(ACTIVATE) && python scripts/render_docs_facts.py

check-docs-facts:
	$(ACTIVATE) && python scripts/render_docs_facts.py --check

## On-demand openCypher conformance check vs Neo4j. Not part of CI.
## See docs/concepts/cypher-conformance.md for the full workflow.
neo4j-up:
	docker run -d --name kglite-neo4j-conformance \
		-p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/conformance \
		neo4j:5

neo4j-down:
	-docker rm -f kglite-neo4j-conformance

neo4j-conformance:
	$(ACTIVATE) && pip install -q -e '.[neo4j]'
	$(ACTIVATE) && python scripts/cypher_conformance.py \
		--uri bolt://localhost:7687 --user neo4j --password conformance

## On-demand Bolt wire round-trip check: runs the differential corpus through
## kglite-bolt-server and compares against direct cypher(). Spawns its own
## server on an ephemeral port — no Neo4j / Docker needed. Not part of CI.
bolt-conformance:
	cargo build -p kglite-bolt-server --release
	$(ACTIVATE) && pip install -q -e '.[neo4j]'
	$(ACTIVATE) && python scripts/bolt_conformance.py

## Fast compilation check (no codegen)
check:
	cargo check

## Format Rust code
fmt:
	cargo fmt

## Format Python code
fmt-py:
	$(ACTIVATE) && ruff format . && ruff check --fix .

## Run clippy lints
clippy:
	cargo clippy --all-targets -- -D warnings

## Run Python lint checks
lint-py:
	$(ACTIVATE) && ruff format --check . && ruff check .

## Enforce the kglite::api single-chokepoint boundary (docs/history/roadmap-2026H1.md)
check-api-chokepoint:
	./scripts/check_api_chokepoint.sh

## Run all lint checks (Rust + Python + stubs) — use before pushing
lint: check-api-chokepoint
	$(ACTIVATE) && python scripts/check_cypher_clean_room.py
	$(ACTIVATE) && python scripts/check_dependency_licenses.py
	cargo fmt -- --check
	cargo clippy --all-targets -- -D warnings
	$(ACTIVATE) && ruff format --check . && ruff check .
	$(ACTIVATE) && python -m mypy.stubtest kglite --ignore-missing-stub --ignore-unused-allowlist --mypy-config-file mypy_stubtest.ini --allowlist stubtest_allowlist.txt

## Run tests with coverage report
cov:
	$(ACTIVATE) && pytest tests/ -v --cov=kglite --cov-branch --cov-config=pyproject.toml --cov-report=term-missing

## Verify type stubs match runtime (requires built extension)
stubtest:
	$(ACTIVATE) && python -m mypy.stubtest kglite --ignore-missing-stub --ignore-unused-allowlist --mypy-config-file mypy_stubtest.ini --allowlist stubtest_allowlist.txt

## Remove build artifacts
clean:
	cargo clean
