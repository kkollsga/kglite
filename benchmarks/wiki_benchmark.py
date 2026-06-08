"""Disk-mode wiki benchmark across KGLite versions.

Runs build / save / load / cypher against each wiki .nt.zst subset on
disk and appends rows to ``benchmarks/wiki_benchmark.csv`` with a ``version``
column so multiple versions accumulate in the same file.

Usage:
    # Against whatever kglite is currently installed.
    python benchmarks/wiki_benchmark.py --version 0.8.11

    # Restrict to a subset of the datasets.
    python benchmarks/wiki_benchmark.py --version 0.8.11 --datasets wiki50m,wiki100m

Typical multi-version invocation:
    maturin develop --release                       # build current tree
    python benchmarks/wiki_benchmark.py --version 0.8.11

    git worktree add /tmp/kg-0810 v0.8.10
    ( cd /tmp/kg-0810 && maturin develop --release )
    cp benchmarks/wiki_benchmark.py /tmp/kg-0810/bench/   # v0.8.10 tree used bench/
    python /tmp/kg-0810/bench/wiki_benchmark.py --version 0.8.10
    git worktree remove /tmp/kg-0810 --force

Measurements per dataset:
  - build: wall_ms, peak_rss_mb, bytes_written, triples/s, node/edge counts
  - save : wall_ms, bytes_written (post-save data_dir size)
  - load : wall_ms, peak_rss_mb (re-measured in the reload subprocess)
  - each cypher query: wall_ms, row_count, status

Each dataset runs in its own subprocess so peak RSS is scoped to that
dataset only and no heap from a smaller run pollutes a later one. Disk
mode only — this benchmark targets the storage engine, not the in-memory
backend.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

# IMPORTANT: do NOT prepend the repo root to sys.path. When running this
# benchmark under a per-version venv (e.g. `kglite==0.8.0` installed via
# pip) we want `import kglite` to resolve to the venv's site-packages,
# not the dev `kglite/` directory that lives next to this file. The
# matching maturin-develop install already exposes the dev build for
# the current-tree run.

# ---------------------------------------------------------------------------
# Dataset registry — path + nominal triple count used for triples/s.
# Subsets whose file is absent are silently skipped.
# ---------------------------------------------------------------------------
DATA_DIR = "/Volumes/EksternalHome/Data/Wikidata"
WIKIDATA_SUBSETS = [
    ("wiki500k", f"{DATA_DIR}/test_500k.nt.zst", 500_000),
    ("wiki5m", f"{DATA_DIR}/test_5M.nt.zst", 5_000_000),
    ("wiki50m", f"{DATA_DIR}/test_50M.nt.zst", 50_000_000),
    ("wiki100m", f"{DATA_DIR}/test_100M.nt.zst", 100_000_000),
    ("wiki200m", f"{DATA_DIR}/test_200M.nt.zst", 200_000_000),
    ("wiki500m", f"{DATA_DIR}/test_500M.nt.zst", 500_000_000),
    ("wiki1000m", f"{DATA_DIR}/test_1000M.nt.zst", 1_000_000_000),
]

TIMEOUT_MS = 30_000

# Same Cypher suite as benchmarks/api_benchmark.py WIKIDATA_QUERIES (kept in
# sync so per-query times are directly comparable across the two
# benchmarks).
#
# Queries assume the build was run with `languages=["en"]` so the
# auto_type rename produces English type names (e.g. Q5 → "human").
# Unanchored `ORDER BY a.title` patterns timed out at 10s on the full
# 124M-node graph, so the larger queries are anchored to a Q-code
# (Q42 = Douglas Adams) — gives a deterministic, finite-cost benchmark
# at every scale instead of a random-walk-style full scan.
WIKIDATA_QUERIES = [
    (
        "P31 from Q42 LIMIT 50",
        "MATCH (a {nid: 'Q42'})-[:P31]->(b) RETURN a.title, b.title LIMIT 50",
    ),
    (
        "human (Q5) lookup",
        "MATCH (h:human) RETURN h.title ORDER BY h.title LIMIT 10",
    ),
    (
        "P31 class counts",
        "MATCH ()-[:P31]->(c) RETURN c.title, count(*) AS k ORDER BY k DESC, c.title LIMIT 10",
    ),
    (
        "2-hop P31 + P279 from Q42",
        "MATCH (a {nid: 'Q42'})-[:P31]->(b)-[:P279]->(c) RETURN a.title, c.title LIMIT 10",
    ),
    (
        "Human citizenship (P27)",
        "MATCH (h:human)-[:P27]->(c) RETURN h.title, c.title ORDER BY h.title, c.title LIMIT 10",
    ),
    (
        "OPTIONAL citizenship",
        "MATCH (h:human) OPTIONAL MATCH (h)-[:P27]->(c) RETURN h.title, c.title ORDER BY h.title LIMIT 10",
    ),
    (
        "EXISTS P27",
        "MATCH (h:human) WHERE EXISTS { MATCH (h)-[:P27]->() } RETURN h.title ORDER BY h.title LIMIT 10",
    ),
    (
        "WITH P27 count",
        "MATCH (h:human)-[:P27]->(c) WITH c, count(h) AS k RETURN c.title, k ORDER BY k DESC, c.title LIMIT 10",
    ),
]

CSV_COLUMNS = [
    "timestamp",
    "version",
    "dataset",
    "test",
    "wall_ms",
    "peak_rss_mb",
    "bytes_written",
    "nominal_triples",
    "triples_per_sec",
    "node_count",
    "edge_count",
    "row_count",
    "status",
    "error",
]

DEFAULT_CSV = Path(__file__).resolve().parent / "wiki_benchmark.csv"
RESULT_PREFIX = "RESULT:"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _dir_size(path):
    total = 0
    for entry in Path(path).rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _maxrss_mb():
    # ru_maxrss is bytes on darwin, kilobytes on linux.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def _append_row(csv_path, row):
    new_file = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        # Fill missing columns so DictWriter doesn't choke.
        w.writerow({k: row.get(k) for k in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Scenario (runs inside a subprocess)
# ---------------------------------------------------------------------------
def _emit(version, label, test, wall_ms, **extra):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": version,
        "dataset": label,
        "test": test,
        "wall_ms": round(wall_ms, 1),
    }
    row.update(extra)
    print(RESULT_PREFIX + json.dumps(row), flush=True)


def _run_scenario(version, label, nt_path, n_triples, data_dir, languages):
    """Disk-only build + save + load + cypher. Emits one RESULT: line per step."""
    import kglite  # noqa: F401 — surface import failures clearly
    from kglite import KnowledgeGraph, load

    # Fresh directory (caller removes + recreates, but be defensive).
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── build (all phases: construct + load_ntriples + rebuild_caches) ──
    # Matches the end-to-end time a user waits to go from an empty
    # disk-backed KG to a fully queryable one, minus save. Save is
    # measured separately below.
    try:
        t0 = time.perf_counter()
        g = KnowledgeGraph(storage="disk", path=str(data_dir))
        load_kwargs: dict = {}
        if languages:
            load_kwargs["languages"] = languages
        g.load_ntriples(nt_path, **load_kwargs)
        # rebuild_caches populates type connectivity and edge counts that
        # the Cypher planner consults — omitting it makes later query
        # numbers incomparable across versions. If the installed version
        # doesn't expose rebuild_caches (pre-0.7.x), fall through.
        if hasattr(g, "rebuild_caches"):
            g.rebuild_caches()
        wall_ms = (time.perf_counter() - t0) * 1000
        info = g.graph_info()
        _emit(
            version,
            label,
            "build",
            wall_ms,
            peak_rss_mb=round(_maxrss_mb(), 1),
            bytes_written=_dir_size(data_dir),
            nominal_triples=n_triples,
            triples_per_sec=round(n_triples / (wall_ms / 1000.0), 1) if wall_ms > 0 else 0,
            node_count=info["node_count"],
            edge_count=info["edge_count"],
            status="ok",
        )
    except BaseException as e:
        _emit(version, label, "build", 0, status="ERROR", error=str(e)[:120])
        return

    # ── save ──
    try:
        t0 = time.perf_counter()
        g.save(str(data_dir))
        wall_ms = (time.perf_counter() - t0) * 1000
        _emit(
            version,
            label,
            "save",
            wall_ms,
            bytes_written=_dir_size(data_dir),
            status="ok",
        )
    except BaseException as e:
        _emit(version, label, "save", 0, status="ERROR", error=str(e)[:120])

    # ── load ──
    try:
        del g
        t0 = time.perf_counter()
        g = load(str(data_dir))
        wall_ms = (time.perf_counter() - t0) * 1000
        info = g.graph_info()
        _emit(
            version,
            label,
            "load",
            wall_ms,
            peak_rss_mb=round(_maxrss_mb(), 1),
            node_count=info["node_count"],
            edge_count=info["edge_count"],
            status="ok",
        )
    except BaseException as e:
        _emit(version, label, "load", 0, status="ERROR", error=str(e)[:120])
        return

    # ── queries ──
    for name, query in WIKIDATA_QUERIES:
        try:
            t0 = time.perf_counter()
            r = g.cypher(query, timeout_ms=TIMEOUT_MS)
            wall_ms = (time.perf_counter() - t0) * 1000
            rc = len(r) if r is not None else 0
            _emit(version, label, name, wall_ms, row_count=rc, status="ok")
        except BaseException as e:
            err = str(e)
            status = "TIMEOUT" if "timed out" in err.lower() else "ERROR"
            _emit(version, label, name, 0, status=status, error=err[:120])


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------
def _spawn_scenario(version, label, nt_path, n_triples, languages):
    """Run _run_scenario in a subprocess, stream its RESULT lines back."""
    data_dir = Path(tempfile.gettempdir()) / f"wikibench_{label}"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # `-P` suppresses Python's automatic cwd injection into sys.path —
    # without it the repo-root `kglite/` directory would shadow the
    # venv-installed kglite whenever we run this from the repo root.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-P",
            __file__,
            "--_scenario",
            "--version",
            version,
            "--label",
            label,
            "--path",
            nt_path,
            "--triples",
            str(n_triples),
            "--data-dir",
            str(data_dir),
            "--languages",
            ",".join(languages),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    rows = []
    # Live-stream subprocess output so long builds show progress.
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith(RESULT_PREFIX):
            rows.append(json.loads(line[len(RESULT_PREFIX) :]))
        else:
            # Echo non-RESULT stderr/stdout so the user sees progress
            # (e.g., verbose load_ntriples output if the version prints it).
            sys.stdout.write(line)
            sys.stdout.flush()
    rc = proc.wait()
    if rc != 0:
        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "version": version,
                "dataset": label,
                "test": "build",
                "status": "SUBPROC_CRASH",
                "error": f"rc={rc}",
            }
        )

    # Scrub the per-run data dir once results are captured.
    shutil.rmtree(data_dir, ignore_errors=True)
    return rows


def _print_row(r):
    """Human-readable one-liner for the console while running."""
    test = r.get("test", "?")
    wall = r.get("wall_ms")
    status = r.get("status", "?")
    tag = f"{wall:>9.1f}ms" if isinstance(wall, (int, float)) and status == "ok" else f"{status:>11s}"
    extras = []
    if r.get("triples_per_sec"):
        extras.append(f"{r['triples_per_sec']:,.0f} triples/s")
    if r.get("peak_rss_mb"):
        extras.append(f"peak_rss={r['peak_rss_mb']:.1f} MB")
    if r.get("bytes_written") is not None:
        mb = r["bytes_written"] / (1024 * 1024)
        extras.append(f"{mb:.1f} MB on disk")
    if r.get("row_count") is not None and test not in ("build", "save", "load"):
        extras.append(f"rows={r['row_count']}")
    if r.get("node_count") is not None and test == "build":
        extras.append(f"nodes={r['node_count']}, edges={r['edge_count']}")
    extra_str = f"  ({', '.join(extras)})" if extras else ""
    err = f"  err={r.get('error', '')}" if r.get("error") else ""
    print(f"    {test:<30s} {tag}{extra_str}{err}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Label written into the CSV's version column.")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated subset labels (e.g. wiki50m,wiki100m). Default: every subset whose file exists.",
    )
    parser.add_argument("--out", default=str(DEFAULT_CSV), help="Output CSV path.")
    parser.add_argument(
        "--languages",
        default="en",
        metavar="CODES",
        help=(
            "Comma-separated language codes for label/description filtering "
            "(default: en). Pass an empty string to keep all languages — "
            "warning: that yields non-deterministic node-type names because "
            "the auto_type rename picks whichever label was seen first, so "
            "the canonical Cypher suite (which expects English type names "
            "like :human) will return 0 rows."
        ),
    )
    args = parser.parse_args()

    languages = [c.strip() for c in args.languages.split(",") if c.strip()]
    wanted = {s.strip() for s in args.datasets.split(",") if s.strip()}
    subsets = WIKIDATA_SUBSETS
    if wanted:
        subsets = [s for s in subsets if s[0] in wanted]

    import kglite

    print(f"KGLite v{kglite.__version__} — wiki disk benchmark")
    print(f"version label: {args.version}")
    print(f"languages:     {languages or '(all)'}")
    print(f"output CSV:    {args.out}")

    for label, path, n_triples in subsets:
        if not os.path.exists(path):
            print(f"\n  [{label}] skipped — {path} not present")
            continue
        print(f"\n  [{label}]  ({n_triples:,} nominal triples, source={Path(path).name})")
        rows = _spawn_scenario(args.version, label, path, n_triples, languages)
        for r in rows:
            _print_row(r)
            _append_row(args.out, r)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_scenario":
        # Internal entrypoint invoked by the parent.
        ap = argparse.ArgumentParser()
        ap.add_argument("--_scenario", action="store_true")
        ap.add_argument("--version", required=True)
        ap.add_argument("--label", required=True)
        ap.add_argument("--path", required=True)
        ap.add_argument("--triples", type=int, required=True)
        ap.add_argument("--data-dir", required=True)
        ap.add_argument("--languages", default="en")
        a = ap.parse_args()
        langs = [c.strip() for c in a.languages.split(",") if c.strip()]
        _run_scenario(a.version, a.label, a.path, a.triples, a.data_dir, langs)
    else:
        main()
