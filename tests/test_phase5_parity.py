"""Phase 5 crunch-point parity tests.

Guards the columnar-cleanup + per-backend-impls phase of the 0.8.0
storage refactor. Tests here:

- **enum-match audit** — confirms that `GraphBackend::<Variant>` match
  sites are confined to the documented whitelist (the dispatcher in
  `schema.rs`, the PyO3 boundary in `mod.rs`, the trait declarations in
  `storage/mod.rs`, and the three disk-internal boundary files
  `ntriples.rs`, `io_operations.rs`, `batch_operations.rs` which reach
  into `DiskGraph` internals for bulk-path performance).
- **graph.copy() CoW correctness** — mutating a copy leaves the
  original unchanged on every backend. This is the Phase 0 crunch-point
  re-asserted after Phase 5's per-backend impls + ColumnStore split.
- **binary-size regression gate** — the release `.dylib` stays under
  the +20% budget relative to the Phase 4 baseline.

Marker assignment is per-function so the structural gate runs in
default CI while the rest stay opt-in:

  - `test_enum_match_audit` — unmarked (pure file scan, < 1s).
  - `test_graph_copy_cow_correctness_*` — `@pytest.mark.parity`
    (functional, needs backend setup).
  - `test_binary_size_regression` — `@pytest.mark.binary_size`
    (needs the release `.dylib` built; CI's `python-tests` job
    already does `cargo build --release`, so it plugs in there).
  - `test_dead_code_check` — `@pytest.mark.parity` (runs
    `cargo clippy --release`, ~30s).

Run: pytest tests/test_phase5_parity.py                  (structural only)
     pytest tests/test_phase5_parity.py -m parity        (functional)
     pytest tests/test_phase5_parity.py -m binary_size   (release-build gate)
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
import pytest

from kglite import KnowledgeGraph

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files allowed to carry `GraphBackend::<Variant>` enum-match patterns.
# Everything else should dispatch through the `GraphRead` / `GraphWrite`
# traits. Each whitelist entry has a justification — if the list grows,
# revisit the design instead of adding another file.
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
    """Drop any `#[cfg(test)] mod …` block. The audit is about the
    production dispatch path; in-source test fixtures may legitimately
    construct `GraphBackend::Memory(…)` etc. (Phase 6's
    `storage/recording.rs` tests do this.)
    """

    marker = "#[cfg(test)]"
    idx = src.find(marker)
    production = src if idx < 0 else src[:idx]
    return "\n".join(line for line in production.splitlines() if not line.lstrip().startswith("//"))


def test_enum_match_audit():
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
        # storage/ subdir files MUST NOT carry enum matches — they
        # exist to provide trait-based alternatives. Test-module
        # fixtures (`#[cfg(test)]`) are stripped before scanning.
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


@pytest.mark.parity
def test_graph_copy_cow_correctness_memory():
    """Mutating the copy does not affect the original (in-memory backend)."""

    kg = KnowledgeGraph()
    df = pd.DataFrame([{"pid": 1, "name": "Alice", "age": 30}, {"pid": 2, "name": "Bob", "age": 25}])
    kg.add_nodes(df, "Person", "pid", "name")

    kg2 = kg.copy()
    kg2.add_nodes(
        pd.DataFrame([{"pid": 1, "name": "Alice Updated", "age": 99}]),
        "Person",
        "pid",
        "name",
        conflict_handling="update",
    )

    orig = kg.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")
    mod = kg2.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")

    orig_rows = [dict(r) for r in orig]
    mod_rows = [dict(r) for r in mod]

    assert orig_rows == [{"age": 30}], f"original mutated unexpectedly: {orig_rows}"
    assert mod_rows == [{"age": 99}], f"copy update did not apply: {mod_rows}"


@pytest.mark.parity
def test_graph_copy_cow_correctness_mapped():
    """Mutating the copy does not affect the original (mapped backend)."""

    kg = KnowledgeGraph(storage="mapped")
    df = pd.DataFrame([{"pid": 1, "name": "Alice", "age": 30}, {"pid": 2, "name": "Bob", "age": 25}])
    kg.add_nodes(df, "Person", "pid", "name")

    kg2 = kg.copy()
    kg2.add_nodes(
        pd.DataFrame([{"pid": 1, "name": "Alice Updated", "age": 99}]),
        "Person",
        "pid",
        "name",
        conflict_handling="update",
    )

    orig = [dict(r) for r in kg.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")]
    mod = [dict(r) for r in kg2.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")]

    assert orig == [{"age": 30}], f"mapped original mutated: {orig}"
    assert mod == [{"age": 99}], f"mapped copy update lost: {mod}"


#: Per-platform release-wheel library size baseline. The Linux ELF
#: (libkglite.so) is ~65% larger than the macOS Mach-O (.dylib) for the
#: same source — different linker behaviour around debug info, lazy
#: binding, and the absence of macOS-style `strip` defaults. CI runs on
#: Linux; most local development happens on macOS; both pin separately.
#: Update both at release time via `make refresh-release-constants`
#: (run on each platform; the script writes whichever entry matches the
#: current host).
BINARY_SIZE_BASELINES = {
    "darwin": 41_234_896,  # 0.12.15 darwin baseline
    "linux": 64_656_000,  # 0.10.26 estimate: 0.9.52 Linux .so (59,529,016) scaled by
    # the same +8.6% as the macOS recapture. Linux has no strip so the bundled
    # server likely adds more in absolute terms — refresh with the real value on
    # the next CI run (the 0.10.0–0.10.25 Linux baseline was never recaptured).
}


@pytest.mark.binary_size
def test_binary_size_regression():
    """Release library size stays under a +10% budget over the per-platform
    baseline.

    Baseline history:
      - Phase 4 exit:  6,996,688 bytes (≈6.67 MB, macOS).
      - 0.9.0:        23,535,664 bytes (≈22.4 MB, macOS). Multi-mode
                      storage, spatial, timeseries, code-tree, MCP,
                      Cypher dialect coverage all landed in the 0.8.x sweep.
      - 0.9.52:       35,925,104 bytes (≈34.3 MB, macOS .dylib).
                      59,529,016 bytes (≈56.8 MB, Linux .so) —
                      added when the first CI run on 0.9.52 surfaced
                      the platform divergence. Growth between 0.9.0
                      and 0.9.52 (~52% on macOS) is concentrated in:
                        * 14 tree-sitter grammars (Dart added 0.9.51,
                          Swift 0.9.40, PHP/HTML/CSS in the 0.9.2x
                          range — each grammar is ~0.5-1 MB);
                        * fastembed feature default-on for the
                          kglite-mcp-server binary build (ort runtime
                          + hf-hub native TLS path, ~3-4 MB);
                        * mcp-methods 0.3.x server-feature evolution;
                        * sodir / wikidata workspace crates with
                          their own dependency closures.
      - 0.10.26:      39,319,984 bytes (≈37.5 MB, macOS .dylib).
                      The kglite-mcp-server *library* is now bundled
                      into the wheel (its `run` statically linked into
                      the extension, so `pip install kglite` ships the
                      MCP server). It shares the one kglite engine — no
                      duplication — but pulls the server's own closure
                      (mcp-methods, rmcp, hyper/hyper-util, clap,
                      tracing-subscriber) into the cdylib: ~3 MB net on
                      macOS after strip, more on Linux (no strip).


      - 0.12.15:       41,234,896 bytes (≈39.3 MB, macOS .dylib).
                      Hardening added checked persistence decoders, disk
                      generation/session ownership paths, complete C-ABI
                      panic boundaries, and bounded executor guards (+4.9%).

    Raising the baseline is a deliberate act — every bump should
    be accompanied by an updated growth note above. For a precise
    drilldown, run `cargo bloat --release --crates --filter kglite`.
    """

    # Post-G.4 the wheel cdylib is the kglite-py crate's output —
    # `libkglite_py.{dylib,so}` — not the engine's `libkglite.{dylib,so}`
    # (which is now an rlib + dylib pair for downstream Rust crates).
    # The wheel artifact is what users `pip install`, so that's what
    # the size gate should track. Pre-G.4 candidates kept for stale
    # leftover compatibility, but listed second so the cdylib wins.
    candidates = [
        REPO_ROOT / "target" / "release" / "libkglite_py.dylib",
        REPO_ROOT / "target" / "release" / "libkglite_py.so",
        REPO_ROOT / "target" / "release" / "libkglite.dylib",
        REPO_ROOT / "target" / "release" / "libkglite.so",
    ]
    bin_path = next((p for p in candidates if p.exists()), None)
    if bin_path is None:
        pytest.skip("release build not present — run `cargo build --release` first")

    size = bin_path.stat().st_size
    platform_key = sys.platform if sys.platform in BINARY_SIZE_BASELINES else "linux"
    baseline = BINARY_SIZE_BASELINES[platform_key]
    gate = int(baseline * 1.10)
    assert size <= gate, (
        f"{bin_path.name} = {size:,} bytes > gate {gate:,} "
        f"(+10% over 0.12.15 {platform_key} baseline {baseline:,}). "
        "Investigate what grew before raising the gate — see the "
        "growth note in this test's docstring for the breakdown shape."
    )


@pytest.mark.parity
def test_dead_code_check():
    """`cargo clippy -- -D dead_code` flags nothing in the graph module."""

    result = subprocess.run(
        ["cargo", "clippy", "--release", "--", "-D", "dead_code"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail("cargo clippy found dead-code warnings:\n" + (result.stdout or "") + "\n" + (result.stderr or ""))
