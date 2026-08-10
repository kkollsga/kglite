# kglite development tasks
# All targets handle CONDA_PREFIX and .venv activation automatically.

SHELL := /bin/bash
ACTIVATE := unset CONDA_PREFIX && source .venv/bin/activate

.PHONY: dev dev-with-bin bundle-bin build-bolt-server test test-full test-rust test-core test-mcp test-cli test-py test-parity bench bench-save bench-compare bench-check bump-version check-release-hygiene release-preflight refresh-release-constants refresh-api-baseline docs-facts check-docs-facts neo4j-up neo4j-down neo4j-conformance bolt-conformance check clean fmt fmt-py clippy gate lint lint-policy lint-full lint-py source-quality rustsec-policy cov stubtest

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

## Fast package-scoped Rust suites for normal local development.
test-core:
	cargo test -p kglite --lib

test-mcp:
	cargo test -p kglite-mcp-server

test-cli:
	cargo test -p kglite-cli

## Cross-storage-mode parity oracles + the .kgl golden-digest tripwire.
##
## Exists because `pytest tests/test_phase4_parity.py` WITHOUT `-m parity`
## exits 5 with "8 deselected" — the marker is in pyproject addopts. That is a
## green that cannot go red, and it silently covers `test_kgl_golden_hash`, the
## only byte-level check on the .kgl format. It was written into a sprint brief
## on 2026-07-30 and only caught because the agent following the brief noticed
## the exit code. A named target means no one has to remember the flag.
test-parity:
	$(ACTIVATE) && pytest tests/ -v -m parity

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
		&& EXACT_SET=$$( [ "$$(uname)" = "Linux" ] && echo "--require-exact-set" || true ) \
		&& python scripts/compare_bench.py $$BASELINE .bench-current.json \
			--metric min --threshold 20 $$EXACT_SET

## Cumulative perf-drift gate: newest per-release baseline vs the anchor
## ~3 releases back at +30% (min). Catches slow drift the per-release 20%
## gates structurally can't (the baseline ratchets forward every release).
## Release-time companion to bench-check; runs in seconds (no benchmarks).
bench-anchor:
	$(ACTIVATE) && python scripts/check_perf_anchor.py

## Rust semver report for the kglite crate vs the last published release.
## INFORMATIONAL by convention: this project deliberately ships documented
## breaking changes in patch bumps (0.x), so the release skill surfaces this
## report in the bump-size decision instead of hard-gating on it.
semver-check:
	cargo semver-checks check-release -p kglite --baseline-version $$(curl -s -H "User-Agent: kglite-semver-check" https://crates.io/api/v1/crates/kglite | python3 -c "import json,sys; print(json.load(sys.stdin)['crate']['max_stable_version'])") || true

## Refresh the three captured constants that drift across releases:
## the .kgl golden digest, the binary-size baseline, and the perf
## baseline. Run as part of every release commit — see CLAUDE.md
## under "Captured-constant refresh at release time".
refresh-release-constants:
	$(ACTIVATE) && maturin develop --release --quiet
	$(ACTIVATE) && python scripts/refresh_release_constants.py

## Refresh every feature-profiled Rust public-API baseline. Toolchain, tool
## version, feature classifications, and output paths are single-sourced in
## tests/api-baselines/rust-api-profiles.json.
refresh-api-baseline:
	python3 scripts/rust_api_profiles.py refresh

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

## Build the release kglite-bolt-server binary the `-m bolt` suites resolve.
## `tests/conftest.py` and `scripts/bolt_conformance.py` both point users here.
build-bolt-server:
	cargo build -p kglite-bolt-server --release

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

check-lint-allowances:
	python scripts/check_lint_allowances.py

## Bump the workspace version everywhere it is written down: the
## `[workspace.package] version` in the root Cargo.toml AND the internal
## `kglite = { version = ... }` requirement in the four member manifests that
## publish against the engine. Editing only the workspace table leaves the
## workspace unresolvable across a minor bump (`cargo metadata`: "failed to
## select a version for the requirement `kglite = ^0.14`"), which broke the
## 0.15.0 release. Verifies with a RESOLVING `cargo metadata` afterwards --
## `--no-deps` skips resolution and passes on exactly the broken tree.
## The bump SIZE is a human decision — see the release skill's `make
## semver-check` step; this target only applies the version you give it.
bump-version:
	@test -n "$(VERSION)" || { echo "usage: make bump-version VERSION=X.Y.Z"; exit 1; }
	python3 scripts/bump_version.py --set $(VERSION)

## Report whether the tree is ready for the release commit. Asserts workspace
## coherence -- every member inherits [workspace.package] version, the
## workspace RESOLVES (which is what a stale internal `kglite = ...` pin
## breaks), and all five publishable crates are at one version -- plus
## version/CHANGELOG agreement, captured constants, server-binary freshness,
## formatting, and fast-forwardability. Run it after promoting the CHANGELOG
## and before writing the release commit.
##
## The three coherence assertions are not redundant with `make bump-version`:
## the bump tool fixes the forward path, but a merge, hand-edit, or a rebase
## that resolves a manifest conflict wrongly can drift the state anyway. And
## resolution alone cannot see a member carrying its own stale explicit
## version -- that resolves fine and still ships a broken publish set.
##
## CHECKER, NOT A DRIVER. It performs no release step and has no --fix: it
## prints the command for each unmet precondition and leaves running it — and
## deciding whether to release at all — to the maintainer. A tool that quietly
## performs the steps it checks is how gates stop gating.
release-preflight:
	python3 scripts/release_preflight.py

## Release-paperwork gates. Both are pure file reads (no cargo, no imports)
## so they stay inside the fast-gate budget:
##  - every internal kglite dependency requirement matches the workspace version
##  - CHANGELOG [Unreleased] has no duplicate/bespoke `###` headings, and no
##    `TODO:` marker written by refresh_release_constants.py survived
check-release-hygiene:
	python3 scripts/bump_version.py --check
	python3 scripts/check_release_hygiene.py

## Cross-repo version-consistency audit over the sibling checkouts under
## ../ (codingest, kglite-datasets, sonagram, sonara, mcp-methods).
##
## DELIBERATELY NOT IN `gate` OR CI. Its whole subject is disagreement
## *between* repos, and CI checks out one repo — there, every sibling is
## absent and the check would either pass vacuously or need five extra
## clones per run. It is a release-time and on-demand target instead, and it
## skips-with-a-message rather than failing when a sibling is missing, so it
## stays usable from a partial checkout. Exits non-zero on contradictions.
check-ecosystem-versions:
	$(ACTIVATE) && python scripts/check_version_consistency.py

## Same audit, plus the full inventory of declaration sites that live outside
## package metadata — the places that can drift with nothing watching them.
check-ecosystem-versions-verbose:
	$(ACTIVATE) && python scripts/check_version_consistency.py --show-sites

## Preview the downstream release notes for the current version without
## writing anything. Run before `notify-downstream` at release time.
notify-downstream-dry-run:
	$(ACTIVATE) && python scripts/check_version_consistency.py --notify --dry-run

## Write a release note into inbox/unread/ of every AFFECTED downstream —
## and only those (see the affected-only rule in the script's docstring).
## Run after publish is verified. inbox/ is gitignored working state; the
## notes are never committed, here or there.
notify-downstream:
	$(ACTIVATE) && python scripts/check_version_consistency.py --notify

## Fast local checkpoint. Pair this with the smallest package/test filter
## covering the change. Policy audits, workspace clippy, stubtest, packaged-
## consumer verification, and the broad test matrix run in CI.
gate: lint check-docs-facts check-release-hygiene check-skill-mirrors check-dev-docs

## Fast formatting/static lint. Intentionally performs no Rust compilation,
## metadata walk, or runtime import.
lint: check-api-chokepoint
	cargo fmt -- --check
	$(ACTIVATE) && ruff format --check . && ruff check .

## Slower repository-wide policy audits. CI runs these; invoke locally only
## when changing their rules, baselines, dependency policy, or Cypher sources.
lint-policy: check-lint-allowances source-quality rustsec-policy
	$(ACTIVATE) && python scripts/check_cypher_clean_room.py
	$(ACTIVATE) && python scripts/check_dependency_licenses.py

## Explicit CI-equivalent lint sweep. Not part of routine local development.
lint-full: gate lint-policy
	cargo clippy --all-targets -- -D warnings
	$(ACTIVATE) && python -m mypy.stubtest kglite --ignore-missing-stub --ignore-unused-allowlist --mypy-config-file mypy_stubtest.ini --allowlist stubtest_allowlist.txt

## Run tests with coverage report
cov:
	$(ACTIVATE) && pytest tests/ -v --cov=kglite --cov-branch --cov-config=pyproject.toml --cov-report=term-missing

## Report-only coverage for the kglite core crate with default features
cov-rust-core:
	cargo llvm-cov --package kglite --lib --tests --ignore-filename-regex 'src/bin/' --lcov --output-path rust-core-coverage.lcov

## Check centralized production-source structure, complexity ceilings, and
## lint-allowance drift — both halves the CI `source-quality` job runs. Keep
## these in step: this target ran only the first script, so a tree with an
## unreviewed or stale #[allow] passed local verification and failed CI.
## The `--self-test` runs come first, and they are the reason this target is
## worth running at all when the scripts change: each one feeds a synthetic
## known-bad tree to the detectors and asserts they fire. A gate whose
## detectors have gone blind reports a clean tree, so proving they still
## trigger must not be reachable only through CI — which is where these two
## invocations previously lived alone.
source-quality:
	python scripts/check_source_quality.py --self-test
	python scripts/check_source_quality.py
	python scripts/check_lint_allowances.py --self-test
	python scripts/check_lint_allowances.py

## Validate that every temporary RustSec exception is justified and unexpired
rustsec-policy:
	python scripts/check_rustsec_advisories.py --policy-only

## Verify type stubs match runtime (requires built extension)
stubtest:
	$(ACTIVATE) && python -m mypy.stubtest kglite --ignore-missing-stub --ignore-unused-allowlist --mypy-config-file mypy_stubtest.ini --allowlist stubtest_allowlist.txt

## Remove build artifacts
clean:
	cargo clean

## Size-gated cargo clean. Cargo never garbage-collects target/: incremental
## session dirs and feature-variant artifacts accumulate without bound (a
## 2026-07 audit found 503 GB / 1.27M files, making an incremental
## `maturin develop` take 8 minutes). Run at release time (wired into the
## release skill); no-ops while target/ stays under the threshold. With
## sccache configured, the post-clean rebuild is cheap.
## Dev-environment cleanliness sweep: the size-gated target prune plus
## removal of regenerable local artifacts that otherwise accumulate without
## any bound or owner (bench captures, sphinx output, tool caches, stale
## ABI-variant extensions, .DS_Store litter). Everything removed here is
## re-creatable by the tool that made it. `.hypothesis/` is deliberately
## KEPT — it is the found-counterexample regression corpus, not a cache.
## Wired into the release skill; safe to run any time.
prune-dev: prune-target
	rm -f .bench-current.json
	rm -rf docs/_build .mypy_cache .ruff_cache .pytest_cache .uv-cache
	find kglite -maxdepth 1 -name "kglite.*.so" ! -name "kglite.abi3.so" -delete
	rm -rf tests/conformance/js/node_modules tests/conformance/java/target tests/conformance/java/.m2
	rm -rf kglite-java/build kglite-java/.gradle
	find . \( -path ./target -o -path ./.venv \) -prune -o -name ".DS_Store" -type f -print0 | xargs -0 rm -f

PRUNE_TARGET_GB := 40
prune-target:
	@link_target=$$(readlink target 2>/dev/null || true); \
	dir=$${CARGO_TARGET_DIR:-$${link_target:-target}}; \
	size_gb=$$(du -sg "$$dir" 2>/dev/null | cut -f1); \
	if [ "$${size_gb:-0}" -ge $(PRUNE_TARGET_GB) ]; then \
		echo "target/ is $${size_gb} GB (>= $(PRUNE_TARGET_GB) GB) — running cargo clean"; \
		mkdir -p "$$dir"; \
		printf '%s\n' \
			'Signature: 8a477f597d28d172789f06886806bc55' \
			'# This file is a cache directory tag created by cargo.' \
			'# For information about cache directory tags see https://bford.info/cachedir/' \
			> "$$dir/CACHEDIR.TAG"; \
		cargo clean --target-dir "$$dir" || exit $$?; \
		mkdir -p "$$dir"; \
		printf '%s\n' \
			'Signature: 8a477f597d28d172789f06886806bc55' \
			'# This file is a cache directory tag created by cargo.' \
			'# For information about cache directory tags see https://bford.info/cachedir/' \
			> "$$dir/CACHEDIR.TAG"; \
		if [ -n "$$link_target" ] && [ ! -L target ]; then ln -s "$$link_target" target; fi; \
	else \
		echo "target/ is $${size_gb:-0} GB — under the $(PRUNE_TARGET_GB) GB prune threshold"; \
	fi

.PHONY: check-skill-mirrors
check-skill-mirrors:
	python3 scripts/check_skill_mirrors.py

## Mechanical bound on the gitignored dev-docs/ working folder — the one
## accumulation with no reviewer, no CI and no remote watching it grow. Unlike
## prune-target this NEVER deletes: which tier a file belongs in, and whether
## it is reproducible, is a judgement call, so the gate FAILS and hands the
## decision back. Stale purge-tier entries are reported as a warning (temp/bin
## churn is normal working state; failing on it would only teach people to
## bypass the gate). Tier lifecycles: dev-docs/README.md.
DEV_DOCS_MAX_MB := 256
.PHONY: check-dev-docs
check-dev-docs:
	@[ -d dev-docs ] || { echo "no dev-docs/ — nothing to bound"; exit 0; }; \
	mb=$$(du -sm dev-docs | cut -f1); \
	stale=$$( { find dev-docs/bench/out -mindepth 1 -maxdepth 1 -mtime +14; \
	            find dev-docs/temp      -mindepth 1 -maxdepth 1 -mtime +1;  \
	            find dev-docs/bin       -mindepth 1 -maxdepth 1 -mtime +7;  \
	          } 2>/dev/null ); \
	if [ "$${mb:-0}" -ge $(DEV_DOCS_MAX_MB) ]; then \
		echo "FAIL: dev-docs/ is $${mb} MB (>= $(DEV_DOCS_MAX_MB) MB)"; \
		echo "  largest tiers:"; \
		du -sm dev-docs/* dev-docs/bench/* 2>/dev/null | sort -rn | head -8 | sed 's/^/    /'; \
		[ -z "$$stale" ] || { echo "  past their documented lifetime:"; echo "$$stale" | sed 's/^/    /'; }; \
		echo "  -> reclaim, or move anything irreproducible to a durable tier (dev-docs/README.md)"; \
		exit 1; \
	fi; \
	echo "dev-docs/ is $${mb} MB (limit $(DEV_DOCS_MAX_MB) MB)"; \
	[ -z "$$stale" ] || { echo "WARN: past their documented lifetime (dev-docs/README.md):"; \
	                      echo "$$stale" | sed 's/^/    /'; }
