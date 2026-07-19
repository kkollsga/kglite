"""Phase 5 crunch-point parity tests.

Guards the columnar-cleanup + per-backend-impls phase of the 0.8.0
storage refactor. Tests here:

- **graph.copy() CoW correctness** — mutating a copy leaves the
  original unchanged on the memory and mapped backends. This is the Phase 0
  crunch-point re-asserted after Phase 5's per-backend impls + ColumnStore
  split. Disk copy ownership has a separate storage-lifecycle follow-up.
- **binary-size regression gate** — the release extension stays under
  the +10% budget relative to the current per-platform baseline.

Marker assignment is per-function so the expensive checks stay opt-in:

  - `test_graph_copy_cow_correctness_*` — `@pytest.mark.parity`
    (functional, needs backend setup).
  - `test_binary_size_regression` — `@pytest.mark.binary_size`
    (needs the release extension built; CI's `python-tests` job
    already builds a release wheel with maturin, so it plugs in there).
  - `test_dead_code_check` — `@pytest.mark.parity` (runs
    `cargo clippy --release`, ~30s).

Run: pytest tests/test_phase5_parity.py -m parity        (functional)
     pytest tests/test_phase5_parity.py -m binary_size   (release-build gate)
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from kglite import KnowledgeGraph

REPO_ROOT = Path(__file__).resolve().parent.parent


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
#: (`libkglite_py.so`) is ~65% larger than the macOS Mach-O (`.dylib`) for the
#: same source — different linker behaviour around debug info, lazy
#: binding, and the absence of macOS-style `strip` defaults. CI runs on
#: Linux; most local development happens on macOS; both pin separately.
#: Update both at release time via `make refresh-release-constants`
#: (run on each platform; the script writes whichever entry matches the
#: current host).
BINARY_SIZE_BASELINES = {
    "darwin": 18_940_768,  # 0.14.3 darwin baseline
    "linux": 28_810_000,  # estimate: the post-code_tree Linux estimate (30.2 MB)
    # scaled by the same −4.6% the macOS loader removal measured. Both
    # removals deliberately recaptured DOWNWARD so the +10% budget guards
    # the real binary. Refresh with the real value on the next CI run.
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
      - 0.13.0:       41,334,128 bytes (≈39.4 MB, macOS .dylib).
                      Checked persistence decoders, disk generation and
                      snapshot ownership, writer leases, complete C-ABI panic
                      boundaries, bounded executor guards, and guarded lazy
                      result materialization account for the growth since the
                      prior published baseline.
      - post-0.13.4:  19,767,648 bytes (≈18.9 MB, macOS .dylib) — a 53%
                      SHRINK: the in-tree code_tree builder and all 15
                      tree-sitter grammars moved to the standalone codingest
                      project. Baseline deliberately recaptured downward so
                      the +10% budget guards the new, smaller binary.
      - post-0.13.4b: 18,857,776 bytes (≈18.0 MB, macOS .dylib) — the
                      sec/sodir/wikidata dataset loaders moved to the
                      kglite-datasets project (zip/quick-xml gone; the
                      remaining ureq/rustls belong to the bundled MCP
                      server via mcp-methods, not the engine).

      - 0.13.1:       41,367,232 bytes (≈39.5 MB). The 33 KB increase adds
                      fused grouped/global count operators, mutation-safe
                      in-memory peer histograms, and fixed-path trail pruning.

      - 0.13.2:       41,400,304 bytes (≈39.5 MB). Added node and relationship
                      text-predicate matchers plus index-aware optimizer routing.


      - 0.13.3:       41,466,176 bytes (≈39.5 MB). Added independent graph-copy
                      identity and disk writer-lineage transfer; refreshed native
                      dependencies. Net growth: 65,872 bytes (0.16%).


      - 0.13.4:       42,310,576 bytes (≈40.4 MB). Bundled the shared Rust
                      CLI (including code-review skill and code-tree commands)
                      into the Python extension, and added Postcard alongside
                      the retained legacy bincode reader. Net growth: 844,400
                      bytes (2.04%).


      - 0.14.2:       18,907,376 bytes (≈18.0 MB). Inline-record endpoint
                      validation, write-provenance propagation, and nested
                      value preservation added 49,600 bytes (0.26%).


      - 0.14.3:       18,940,768 bytes (≈18.1 MB). Postcard-only
                      persistence and legacy-reader removal mostly offset
                      the full dependency-graph upgrade (serde, chrono,
                      rayon, csv, and friends), for a net growth of 33,392
                      bytes (0.18%).

    Raising the baseline is a deliberate act — every bump should
    be accompanied by an updated growth note above. For a precise
    drilldown, run `cargo bloat --release --crates --filter kglite`.
    """

    # Post-G.4 the wheel cdylib is the kglite-py crate's output —
    # `libkglite_py.{dylib,so}` — not the engine's `libkglite.{dylib,so}`
    # (which is now an rlib + dylib pair for downstream Rust crates).
    # The wheel artifact is what users `pip install`, so only kglite-py's
    # cdylib is a valid measurement. Falling back to the core library can make
    # this gate pass while the shipped extension is absent or oversized.
    candidates = [
        REPO_ROOT / "target" / "release" / "libkglite_py.dylib",
        REPO_ROOT / "target" / "release" / "libkglite_py.so",
        REPO_ROOT / "target" / "release" / "kglite_py.dll",
    ]
    bin_path = next((p for p in candidates if p.exists()), None)
    if bin_path is None:
        pytest.fail("kglite-py release cdylib is missing — run `maturin build --release`")

    size = bin_path.stat().st_size
    platform_key = sys.platform if sys.platform in BINARY_SIZE_BASELINES else "linux"
    baseline = BINARY_SIZE_BASELINES[platform_key]
    gate = int(baseline * 1.10)
    assert size <= gate, (
        f"{bin_path.name} = {size:,} bytes > gate {gate:,} "
        f"(+10% over 0.14.3 {platform_key} baseline {baseline:,}). "
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
