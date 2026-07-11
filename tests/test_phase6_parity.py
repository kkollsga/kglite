"""Phase 6 crunch-point parity tests.

Guards the validation-backend phase of the 0.8.0 storage refactor.
Phase 6 ships `RecordingGraph<G: GraphRead>` — a thin wrapper that
delegates every trait call to the wrapped backend while logging the
method name — and a `GraphBackend::Recording(Box<RecordingGraph<
GraphBackend>>)` variant that lets the enum dispatcher exercise the
wrapper end-to-end. The backend is Rust-only (no Python constructor
reaches it); the cross-mode parity matrix for it lives in
`src/graph/storage/recording.rs::tests`.

The three tests here are gates, not functional coverage:

- **Enum-match audit still holds** — re-runs the Phase 5 whitelist
  check to confirm no new enum-match site leaked into Phase 6's
  `recording.rs` or anywhere else.
- **Symbol smoke** — asserts `pub use recording::RecordingGraph;`
  stays in `src/graph/storage/mod.rs` so downstream consumers see the
  type at the documented path.
- **File-count budget** — the Phase 6 crunch-point gate: the phase
  should touch at most three src files beyond the test file. Computed
  against the last on-disk `Phase 5` commit.

Both gates are pure file scans (no fixtures, no backends), so they run
in the default `pytest tests/` collection — no `parity` marker. The
cross-backend functional parity tests live in test_phase{1,2,3,4}_parity.py
and stay opt-in via `-m parity`.

Run: pytest tests/test_phase6_parity.py
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest  # noqa: F401  (kept for downstream marker decorators if added)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files permitted to carry `GraphBackend::<Variant>` enum-match patterns.
# Phase 6 does not add any new whitelisted file — Recording dispatch
# lives in `schema.rs` alongside the existing enum dispatcher, and
# `recording.rs` uses GraphRead / GraphWrite traits (no variant
# matching).
ENUM_MATCH_WHITELIST = {
    "dir_graph/mod.rs": "DirGraph index maintenance (petgraph-only fast paths)",
    "dir_graph/disk_persistence.rs": "DirGraph disk lifecycle and durable-save dispatch",
    "introspection/connectivity.rs": "compute_type_connectivity disk-mode Rayon fast path",
    "io/ntriples/writer.rs": "disk-internal bulk-build (ntriples edge writer)",
    "mutation/batch.rs": "disk-internal update-path row_id lookup",
    "mutation/subgraph_streaming.rs": "disk-internal streaming subgraph filter (Pass A/B)",
    "storage/mode.rs": "explicit storage-mode transition constructor",
    "io/ntriples/column_builder.rs": "ntriples columnar-build hot path",
    "languages/cypher/executor/match_clause/fused_match.rs": (
        "MATCH executor inspects backend variant to pick storage-mode-specific traversal primitives"
    ),
}

ENUM_MATCH_PATTERN = re.compile(r"GraphBackend::[A-Z]")


def _list_rs_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.rs"))


def _strip_test_modules(src: str) -> str:
    """Drop any `#[cfg(test)] mod …` block. The audit checks production
    dispatch, not in-source test fixtures that legitimately construct
    `GraphBackend::Memory(...)` / `::Mapped(...)` / `::Disk(...)`.
    """

    marker = "#[cfg(test)]"
    idx = src.find(marker)
    production = src if idx < 0 else src[:idx]
    return "\n".join(line for line in production.splitlines() if not line.lstrip().startswith("//"))


def test_enum_match_audit_still_holds():
    """`GraphBackend::<Variant>` matches only appear in whitelisted files."""

    src_graph = REPO_ROOT / "crates" / "kglite" / "src" / "graph"
    assert src_graph.is_dir(), f"structural audit root is missing: {src_graph}"
    rs_files = _list_rs_files(src_graph)
    assert rs_files, f"structural audit found no Rust sources under {src_graph}"
    offenders: dict[Path, int] = {}
    for rs in rs_files:
        if rs.name.endswith("_tests.rs"):
            continue
        rel = rs.relative_to(src_graph).as_posix()
        if rel in ENUM_MATCH_WHITELIST:
            continue
        # Strip `#[cfg(test)] mod …` blocks — the audit is about the
        # production dispatch path, not fixtures inside unit tests that
        # legitimately construct `GraphBackend::Memory(…)` etc.
        text = _strip_test_modules(rs.read_text())
        hits = ENUM_MATCH_PATTERN.findall(text)
        if hits:
            offenders[rs] = len(hits)

    assert not offenders, (
        "GraphBackend:: enum matches leaked outside the whitelist:\n"
        + "\n".join(f"  {p.relative_to(REPO_ROOT)}: {n} hit(s)" for p, n in offenders.items())
        + "\n\nAdd the file to ENUM_MATCH_WHITELIST (with a written justification) "
        + "or route the call through GraphRead / GraphWrite."
    )


def test_enum_match_whitelist_is_not_stale():
    src_graph = REPO_ROOT / "crates" / "kglite" / "src" / "graph"
    stale = []
    for rel in ENUM_MATCH_WHITELIST:
        path = src_graph / rel
        if not path.is_file():
            stale.append(f"{rel}: file no longer exists")
        elif not ENUM_MATCH_PATTERN.search(_strip_test_modules(path.read_text())):
            stale.append(f"{rel}: no production GraphBackend variant match remains")
    assert not stale, "Stale enum-match whitelist entries:\n  " + "\n  ".join(stale)


def test_recording_graph_symbol_exported():
    """`pub use recording::RecordingGraph;` stays in storage/mod.rs."""

    mod_rs = (REPO_ROOT / "crates" / "kglite" / "src" / "graph" / "storage" / "mod.rs").read_text()
    assert "pub mod recording;" in mod_rs, (
        "`storage/mod.rs` lost the `pub mod recording;` declaration — "
        "downstream consumers (schema.rs) import `RecordingGraph` via this path."
    )
    assert "pub use recording::RecordingGraph;" in mod_rs, (
        "`storage/mod.rs` lost the `pub use recording::RecordingGraph;` re-export."
    )


# test_file_count_budget was a Phase-6-specific gate that self-skipped
# under any Phase-7+ commit. It was deleted in Phase 12 — once the
# code tree passed the Phase-7 structural reorg the gate's purpose
# (enforcing RecordingGraph's 3-file PR shape) was permanently
# superseded by the god-file gate in test_phase7_parity.py.
