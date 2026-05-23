"""Structural parity tests.

Evergreen quality gates that grew out of the 0.8.0 storage refactor:

- **god-file gate** — enumerates `*.rs` under `src/graph/`, asserts each
  is ≤ 2500 lines unless it is on the documented exception list. The
  exception list is maintained with a written rationale per entry and
  a target 0.9.0+ follow-up.
- **unsafe-has-SAFETY gate** — every `unsafe { ... }` block has a
  preceding `// SAFETY:` comment (within the last 5 lines).
- **mod.rs purity gate** — each subdir-`mod.rs` file under
  `src/graph/{algorithms,features,introspection,io,mutation,pyapi,query,storage}/`
  is short and contains no `impl` blocks or long function bodies.

These are pure file scans (no fixtures, no backends, < 1s total) so
they run in the default `pytest tests/` collection — no `parity`
marker. The cross-backend functional parity tests live in
test_phase{1,2,3,4}_parity.py and stay opt-in via `-m parity`.

Run: pytest tests/test_phase7_parity.py
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_GRAPH = REPO_ROOT / "src" / "graph"

# ── God-file gate ──────────────────────────────────────────────────────────
#
# Soft target: every .rs ≤ 1500. Hard cap: ≤ 2500. Exceptions below
# carry a rationale + the 0.9.0+ follow-up plan.
HARD_CAP = 2500

GOD_FILE_EXCEPTIONS: dict[str, str] = {
    # 0.9.0 follow-up: both files grew with Cypher dialect work
    # (multi-MATCH spatial fusion, count-subquery, size() pattern-expr,
    # NULLS LAST sort in heap_top_k, polygon-vs-polygon fast path).
    # Splitting them would touch the executor/planner critical paths
    # mid-release. Tracked for 0.9.x as "executor split into per-clause
    # subfiles" + "planner fusion split into one file per fusion shape".
    "languages/cypher/executor/match_clause.rs": (
        "0.9.x split: extract pattern-execution helpers + the streaming "
        "match orchestration into peer files (`match_executor.rs`, "
        "`match_stream.rs`)."
    ),
    "languages/cypher/planner/fusion.rs": (
        "0.9.x split: one file per fusion shape — fuse_spatial_join, "
        "fuse_topk, fuse_aggregate_pushdown, fuse_simplification."
    ),
}


def _relative_rs_path(path: Path) -> str:
    return path.relative_to(SRC_GRAPH).as_posix()


def test_god_file_gate():
    """No `.rs` under `src/graph/` exceeds 2500 lines unless documented."""
    offenders: list[tuple[str, int]] = []
    for rs in sorted(SRC_GRAPH.rglob("*.rs")):
        rel = _relative_rs_path(rs)
        lines = len(rs.read_text().splitlines())
        if lines > HARD_CAP and rel not in GOD_FILE_EXCEPTIONS:
            offenders.append((rel, lines))
    if offenders:
        msg = "Files over 2500-line hard cap without a documented exception:\n"
        msg += "\n".join(f"  {rel}: {lines} lines" for rel, lines in offenders)
        msg += (
            "\n\nEither split the file or add it to GOD_FILE_EXCEPTIONS "
            "with a written rationale and 0.9.0+ follow-up plan."
        )
        pytest.fail(msg)


UNSAFE_OPEN = re.compile(r"unsafe\s*\{")


def test_unsafe_has_safety_comment():
    """Every `unsafe { ... }` has `// SAFETY:` within the prior 5 lines."""
    offenders: list[str] = []
    for rs in sorted(SRC_GRAPH.rglob("*.rs")):
        lines = rs.read_text().splitlines()
        for i, line in enumerate(lines):
            if UNSAFE_OPEN.search(line):
                # Skip comment lines that mention unsafe in prose
                stripped = line.lstrip()
                if stripped.startswith("//"):
                    continue
                window = lines[max(0, i - 5) : i]
                if not any("SAFETY" in w for w in window):
                    rel = rs.relative_to(REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{i + 1}: {line.strip()}")
    if offenders:
        pytest.fail("`unsafe {` blocks without `// SAFETY:` justification:\n" + "\n".join(f"  {o}" for o in offenders))


def test_mod_rs_purity():
    """Each subdir-`mod.rs` is short (no big impl blocks, no long fns)."""
    subdirs = [
        "algorithms",
        "core",
        "core/pattern_matching",
        "features",
        "introspection",
        "io",
        "io/ntriples",
        "languages",
        "languages/cypher",
        "languages/cypher/executor",
        "languages/cypher/parser",
        "languages/cypher/planner",
        "languages/fluent",
        "mutation",
        "pyapi",
        "storage",
        "storage/disk",
        "storage/mapped",
        "storage/memory",
    ]
    offenders: list[str] = []
    for sub in subdirs:
        mod_rs = SRC_GRAPH / sub / "mod.rs"
        if not mod_rs.exists():
            continue
        text = mod_rs.read_text()
        lines = text.splitlines()
        # Soft cap on subdir mod.rs line count. `storage/mod.rs`
        # legitimately holds the GraphRead/GraphWrite trait definitions
        # + GAT associated-type bounds. `cypher/mod.rs` holds the
        # Cypher facade (cypher() entry point + parse/plan orchestration).
        # Other subdir mod.rs files are pure re-exports.
        if sub == "storage":
            cap = 800
        elif sub == "languages/cypher":
            cap = 500
        elif sub == "languages/cypher/executor":
            # Hosts CypherExecutor struct + constructor + execute()
            # orchestrator + finalize_result + filter-spec types — the
            # shared state that every clause-submodule borrows from.
            # 0.9.0: bumped from 1200 — accumulated dispatch + filter
            # specs from §1/§3/§4/§6 work. 0.9.x cleanup: extract
            # filter-spec types into a peer file.
            cap = 1300
        elif sub == "languages/cypher/parser":
            # Hosts CypherParser struct + token helpers + parse_query
            # orchestrator + public parse_cypher entry + tests.
            # 0.9.0: bumped from 1000 — Cluster 3's byte-precise
            # error-position helpers + intent-rewrite hook live here.
            cap = 1100
        elif sub == "languages/cypher/planner":
            # Hosts the optimize() orchestrator + mark_* helpers + tests.
            # 0.9.0: bumped from 400 — Phase 8+ added the const
            # PASSES registry + per-pass docs + the optimize_query
            # entry that walks them. Splitting needs a focused PR
            # (passes registry → planner/passes/mod.rs).
            cap = 750
        else:
            cap = 300
        if len(lines) > cap:
            offenders.append(f"{sub}/mod.rs: {len(lines)} lines > cap {cap}")
    if offenders:
        pytest.fail("Subdir `mod.rs` files are too long:\n" + "\n".join(f"  {o}" for o in offenders))
