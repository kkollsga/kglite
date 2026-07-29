"""Full-stack benchmark across every storage mode and dataset size.

Covers the end-to-end lifecycle of a KGLite graph:

  1. build        load_ntriples from a Wikidata .nt.zst subset
  2. save         graph.save() to disk
  3. load         kglite.load() from disk
  4. mutate       bulk add_nodes / add_connections + Cypher SET + DELETE
  5. resave       graph.save() after mutations
  6. cypher_*     curated Cypher queries (point lookup, 1/2-hop,
                  aggregations, OPTIONAL/EXISTS, count subquery,
                  STARTS WITH)
  7. fluent_*     curated fluent-API operations (select / traverse /
                  where / where_connected / 2-hop chain)

Per (mode, dataset) runs in its own subprocess so peak RSS is scoped.
A single CSV row per (mode, dataset, test) lands in `--out`. Final
console output: per-dataset comparison table + totals.

Usage:
    python bench/benchmark_full.py                         # all modes, all sizes
    python bench/benchmark_full.py --modes mapped,disk
    python bench/benchmark_full.py --datasets wiki5m,wiki50m
    python bench/benchmark_full.py --memory-max wiki50m   # skip memory >50m
    python bench/benchmark_full.py --out /tmp/full.csv

Each phase logs per-test RESULT lines so the parent can stream a CSV
without the child needing the path passed in. Status column is `ok`,
`SKIP`, `ERROR`, or `TIMEOUT`. Rows with non-`ok` status are
included in totals as 0 ms (so missing data doesn't inflate
comparisons).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "/Volumes/EksternalHome/Data/Wikidata"

DATASETS = [
    ("wiki500k", f"{DATA_DIR}/test_500k.nt.zst", 500_000),
    ("wiki5m", f"{DATA_DIR}/test_5M.nt.zst", 5_000_000),
    ("wiki50m", f"{DATA_DIR}/test_50M.nt.zst", 50_000_000),
    ("wiki100m", f"{DATA_DIR}/test_100M.nt.zst", 100_000_000),
    ("wiki200m", f"{DATA_DIR}/test_200M.nt.zst", 200_000_000),
    ("wiki500m", f"{DATA_DIR}/test_500M.nt.zst", 500_000_000),
    ("wiki1000m", f"{DATA_DIR}/test_1000M.nt.zst", 1_000_000_000),
]

MODES = ["default", "mapped", "disk"]

# Default cap for memory mode — wiki100m is ~1 GB heap, wiki200m+ would
# OOM 16-32 GB machines. Override with --memory-max.
DEFAULT_MEMORY_MAX = "wiki100m"

QUERY_TIMEOUT_MS = 60_000

# Cypher suite — labels are stable column keys. Each query is sized so
# wiki1000m runs in <30 s on disk.
FLUENT_TESTS = [
    "fl/select.len",
    "fl/where score>500",
    "fl/traverse P31 out l50",
    "fl/traverse P31 in l50",
    "fl/where_connected P31",
    "fl/2-hop P31->P279",
]

CYPHER_QUERIES = [
    (
        "cy/point lookup",
        "MATCH (n {nid:'Q42'}) RETURN n.title",
    ),
    (
        "cy/typed P31 LIMIT 50",
        "MATCH (a)-[:P31]->(b) RETURN a.title, b.title ORDER BY a.title, b.title LIMIT 50",
    ),
    (
        "cy/instance count Q5",
        "MATCH (n)-[:P31]->({nid:'Q5'}) RETURN count(n) AS c",
    ),
    (
        "cy/class counts top10",
        "MATCH (a)-[:P31]->(b) RETURN b.title AS cls, count(a) AS c ORDER BY c DESC LIMIT 10",
    ),
    (
        "cy/2-hop P31+P279",
        "MATCH (a)-[:P31]->(b)-[:P279]->(c) RETURN a.title, c.title LIMIT 10",
    ),
    (
        "cy/citizenship join",
        "MATCH (a)-[:P31]->(b {nid:'Q5'}) MATCH (a)-[:P27]->(c) RETURN a.title, c.title LIMIT 10",
    ),
    (
        "cy/OPTIONAL P27",
        "MATCH (a)-[:P31]->(b {nid:'Q5'}) OPTIONAL MATCH (a)-[:P27]->(c) RETURN a.title, c.title LIMIT 10",
    ),
    (
        "cy/EXISTS P27",
        "MATCH (a)-[:P31]->(b {nid:'Q5'}) WHERE EXISTS ((a)-[:P27]->()) RETURN a.title LIMIT 10",
    ),
    (
        "cy/count{} subquery",
        "MATCH (a)-[:P31]->(b {nid:'Q5'}) WITH a, count{(a)-[:P27]->()} AS n RETURN a.title, n ORDER BY n DESC LIMIT 10",
    ),
    (
        "cy/title STARTS WITH",
        "MATCH (n) WHERE n.title STARTS WITH 'Albert ' RETURN n.title LIMIT 20",
    ),
]


# ---------------------------------------------------------------------------
# Subprocess scenario (the child process actually runs the tests)
# ---------------------------------------------------------------------------

RESULT_PREFIX = "RESULT:"


# Wide CSV schema — one row per (run, mode, dataset). Stable column
# order so subsequent runs append cleanly without reshuffling. Columns
# are derived from the test names above so adding a test means adding
# a column at the end (you'll need to migrate the CSV manually if the
# new column isn't last; the easy escape is to delete the old CSV
# and start fresh).
def _column_name(test: str, kind: str = "ms") -> str:
    """Sanitize a test name into a valid CSV column suffix."""
    s = (
        test.replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("{}", "subq")
        .replace("->", "_to_")
        .replace(">", "gt")
        .replace("<", "lt")
    )
    s = "".join(c if c.isalnum() or c == "_" else "" for c in s)
    return f"{s.lower()}_{kind}"


def _csv_columns() -> list[str]:
    cols = [
        "run_started_at",
        "version",
        "mode",
        "dataset",
        # Build / save / load / mutate / resave: ms + key metadata.
        "build_ms",
        "build_nodes",
        "build_edges",
        "build_rss_mb",
        "save_ms",
        "save_mb",
        "load_ms",
        "load_rss_mb",
        "mutate_ms",
        "mutate_nodes",
        "mutate_edges",
        "resave_ms",
        "resave_mb",
    ]
    cols += [_column_name(name) for name, _ in CYPHER_QUERIES]
    cols += [_column_name(name) for name in FLUENT_TESTS]
    cols += [
        "total_ms",
        "errors",  # semicolon-separated list of failed test labels
    ]
    return cols


CSV_COLUMNS = _csv_columns()


def _maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def _emit(mode: str, dataset: str, test: str, wall_ms: float, **extra) -> None:
    """Child-side: emit a per-test record to stdout for the parent to
    pivot. Stays in tall shape over the pipe; the parent collects all
    of one (mode, dataset)'s records and writes a single wide CSV row."""
    row = {
        "mode": mode,
        "dataset": dataset,
        "test": test,
        "wall_ms": round(wall_ms, 1),
    }
    row.update(extra)
    print(RESULT_PREFIX + json.dumps(row), flush=True)


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    if p.is_file():
        return p.stat().st_size
    for f in p.rglob("*"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def _open_graph(mode: str, work_dir: Path):
    import kglite

    if mode == "default":
        return kglite.KnowledgeGraph()
    if mode == "mapped":
        return kglite.KnowledgeGraph(storage="mapped")
    if mode == "disk":
        return kglite.KnowledgeGraph(storage="disk", path=str(work_dir / "graph"))
    raise ValueError(mode)


def _save_path(mode: str, work_dir: Path) -> Path:
    if mode == "disk":
        return work_dir / "graph"
    return work_dir / "graph.kgl"


def _run_scenario(mode: str, dataset: str, nt_path: str, n_triples: int, work_dir: Path) -> None:
    """Execute every phase for one (mode, dataset). Emits one RESULT
    line per phase to stdout for the parent to capture."""
    from kglite import load as kg_load

    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. build ─────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        g = _open_graph(mode, work_dir)
        g.load_ntriples(nt_path)
        if hasattr(g, "rebuild_caches"):
            g.rebuild_caches()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        info = g.graph_info()
        _emit(
            mode,
            dataset,
            "build",
            wall_ms,
            peak_rss_mb=round(_maxrss_mb(), 1),
            node_count=info.get("node_count"),
            edge_count=info.get("edge_count"),
            status="ok",
        )
    except BaseException as e:
        _emit(mode, dataset, "build", 0, status="ERROR", error=str(e)[:200])
        return

    save_path = _save_path(mode, work_dir)

    # ── 2. save ──────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        g.save(str(save_path))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        _emit(
            mode,
            dataset,
            "save",
            wall_ms,
            bytes_written=_dir_size(save_path),
            status="ok",
        )
    except BaseException as e:
        _emit(mode, dataset, "save", 0, status="ERROR", error=str(e)[:200])

    # ── 3. load (reload from saved) ──────────────────────────────────────
    try:
        del g
        gc.collect()
        t0 = time.perf_counter()
        g = kg_load(str(save_path))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        info = g.graph_info()
        _emit(
            mode,
            dataset,
            "load",
            wall_ms,
            peak_rss_mb=round(_maxrss_mb(), 1),
            node_count=info.get("node_count"),
            edge_count=info.get("edge_count"),
            status="ok",
        )
    except BaseException as e:
        _emit(mode, dataset, "load", 0, status="ERROR", error=str(e)[:200])
        return

    # ── 4. mutate ────────────────────────────────────────────────────────
    # Add 1 000 nodes + 500 edges, set 100 props, delete 50 nodes.
    try:
        import pandas as pd

        t0 = time.perf_counter()
        new_nodes = pd.DataFrame(
            {
                "nid": [f"BENCH_N_{i}" for i in range(1000)],
                "name": [f"BenchNode{i}" for i in range(1000)],
                "score": [float(i) for i in range(1000)],
            }
        )
        g.add_nodes(new_nodes, node_type="BenchType", unique_id_field="nid", node_title_field="name")

        new_edges = pd.DataFrame(
            {
                "src": [f"BENCH_N_{i}" for i in range(500)],
                "tgt": [f"BENCH_N_{i + 500}" for i in range(500)],
            }
        )
        g.add_connections(
            new_edges,
            connection_type="BENCH_LINK",
            source_type="BenchType",
            source_id_field="src",
            target_type="BenchType",
            target_id_field="tgt",
        )

        # Property updates
        g.cypher("MATCH (n:BenchType) WHERE n.score < 100 SET n.tag = 'low'")
        # Delete a chunk
        g.cypher("MATCH (n:BenchType) WHERE n.score >= 950 DETACH DELETE n")

        wall_ms = (time.perf_counter() - t0) * 1000.0
        info = g.graph_info()
        _emit(
            mode,
            dataset,
            "mutate",
            wall_ms,
            peak_rss_mb=round(_maxrss_mb(), 1),
            node_count=info.get("node_count"),
            edge_count=info.get("edge_count"),
            status="ok",
        )
    except BaseException as e:
        _emit(mode, dataset, "mutate", 0, status="ERROR", error=str(e)[:200])

    # ── 5. resave ────────────────────────────────────────────────────────
    try:
        resave_path = _save_path(mode, work_dir / "resave")
        (work_dir / "resave").mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        g.save(str(resave_path))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        _emit(
            mode,
            dataset,
            "resave",
            wall_ms,
            bytes_written=_dir_size(resave_path),
            status="ok",
        )
    except BaseException as e:
        _emit(mode, dataset, "resave", 0, status="ERROR", error=str(e)[:200])

    # ── 6. cypher queries ────────────────────────────────────────────────
    for name, query in CYPHER_QUERIES:
        try:
            t0 = time.perf_counter()
            r = g.cypher(query, timeout_ms=QUERY_TIMEOUT_MS)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            rc = len(r) if r is not None else 0
            _emit(mode, dataset, name, wall_ms, row_count=rc, status="ok")
        except BaseException as e:
            err = str(e)
            status = "TIMEOUT" if "timed out" in err.lower() else "ERROR"
            _emit(mode, dataset, name, 0, status=status, error=err[:200])

    # ── 7. fluent queries ────────────────────────────────────────────────
    # Discover a source type with outgoing P31 edges (otherwise the
    # fluent traversals all return 0). Probe top 5 by node count.
    try:
        types = g.node_type_counts() if hasattr(g, "node_type_counts") else {}
    except BaseException:
        types = {}
    candidates = sorted(types.items(), key=lambda kv: -kv[1])[:5]
    biggest = None
    for t, _ in candidates:
        try:
            hit = g.select(t).traverse("P31", direction="outgoing", limit=1).len()
            if hit > 0:
                biggest = t
                break
        except BaseException:
            continue
    if biggest is None and candidates:
        biggest = candidates[0][0]

    fluent_cases = [
        ("fl/select.len", lambda: g.select(biggest).len()),
        ("fl/where score>500", lambda: g.select("BenchType").where({"score": {">": 500}}).len()),
        ("fl/traverse P31 out l50", lambda: g.select(biggest).traverse("P31", direction="outgoing", limit=50).len()),
        ("fl/traverse P31 in l50", lambda: g.select(biggest).traverse("P31", direction="incoming", limit=50).len()),
        ("fl/where_connected P31", lambda: g.select(biggest).where_connected("P31").len()),
        ("fl/2-hop P31->P279", lambda: g.select(biggest).traverse("P31", limit=10).traverse("P279", limit=10).len()),
    ]
    if biggest is None:
        for name, _ in fluent_cases:
            _emit(mode, dataset, name, 0, status="SKIP", error="no usable type")
    else:
        for name, fn in fluent_cases:
            try:
                t0 = time.perf_counter()
                rc = fn()
                wall_ms = (time.perf_counter() - t0) * 1000.0
                _emit(
                    mode,
                    dataset,
                    name,
                    wall_ms,
                    row_count=int(rc) if isinstance(rc, int) else None,
                    status="ok",
                )
            except BaseException as e:
                err = str(e)
                status = "TIMEOUT" if "timed out" in err.lower() else "ERROR"
                _emit(mode, dataset, name, 0, status=status, error=err[:200])


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------


def _pivot_to_wide(mode: str, dataset: str, rows: list[dict], run_started: str, version: str) -> dict:
    """Collapse one (mode, dataset)'s tall rows into a single wide row
    matching `CSV_COLUMNS`. Tests that succeeded contribute their
    `wall_ms`; failures leave the cell empty and are recorded in the
    `errors` column (semicolon-separated labels). Build/save/load/
    mutate/resave also surface their key metadata (rss, bytes, node/
    edge count) into the wide row's dedicated columns."""
    out: dict[str, object] = {
        "run_started_at": run_started,
        "version": version,
        "mode": mode,
        "dataset": dataset,
    }
    errors: list[str] = []
    err_log_path = Path(os.environ.get("KGLITE_BENCH_ERRLOG", "")) if os.environ.get("KGLITE_BENCH_ERRLOG") else None
    total_ms = 0.0
    for r in rows:
        test = r.get("test")
        status = r.get("status", "ok")
        wall = float(r.get("wall_ms") or 0)
        if status != "ok":
            err_text = (r.get("error") or "").replace(";", ",").replace("\n", " ")
            # CSV column gets a truncated, semicolon-safe form.
            errors.append(f"{test}={status}: {err_text[:120]}")
            # Sidecar log captures the full error text (no truncation,
            # tabs as separators so it stays grep/cut friendly).
            if err_log_path is not None:
                try:
                    with open(err_log_path, "a") as fh:
                        fh.write(
                            "\t".join(
                                [
                                    run_started,
                                    version,
                                    mode,
                                    dataset,
                                    str(test),
                                    status,
                                    (r.get("error") or "").replace("\n", " "),
                                ]
                            )
                            + "\n"
                        )
                except OSError:
                    pass
            continue
        total_ms += wall
        if test == "build":
            out["build_ms"] = round(wall, 1)
            out["build_nodes"] = r.get("node_count")
            out["build_edges"] = r.get("edge_count")
            out["build_rss_mb"] = r.get("peak_rss_mb")
        elif test == "save":
            out["save_ms"] = round(wall, 1)
            bw = r.get("bytes_written")
            if bw is not None:
                out["save_mb"] = round(bw / (1 << 20), 1)
        elif test == "load":
            out["load_ms"] = round(wall, 1)
            out["load_rss_mb"] = r.get("peak_rss_mb")
        elif test == "mutate":
            out["mutate_ms"] = round(wall, 1)
            out["mutate_nodes"] = r.get("node_count")
            out["mutate_edges"] = r.get("edge_count")
        elif test == "resave":
            out["resave_ms"] = round(wall, 1)
            bw = r.get("bytes_written")
            if bw is not None:
                out["resave_mb"] = round(bw / (1 << 20), 1)
        elif test:
            col = _column_name(test)
            out[col] = round(wall, 1)
    out["total_ms"] = round(total_ms, 1)
    out["errors"] = ";".join(errors)
    return out


def _spawn(mode: str, dataset: str, nt_path: str, n_triples: int, csv_writer, run_started: str, version: str) -> dict:
    """Run one (mode, dataset) scenario in a subprocess, stream RESULT
    lines back, pivot into a single wide CSV row, append it, and
    return the wide-row dict."""
    work_dir = Path(tempfile.gettempdir()) / f"kglite_bench_{mode}_{dataset}_{os.getpid()}"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-P",
            os.path.abspath(__file__),
            "--child-scenario",
            "--mode",
            mode,
            "--dataset",
            dataset,
            "--nt-path",
            nt_path,
            "--n-triples",
            str(n_triples),
            "--work-dir",
            str(work_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    rows: list[dict] = []
    for line in proc.stdout or []:
        line = line.rstrip()
        if line.startswith(RESULT_PREFIX):
            try:
                row = json.loads(line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError:
                continue
            rows.append(row)
            _print_row(row)
        else:
            print(line, flush=True)
    proc.wait()
    shutil.rmtree(work_dir, ignore_errors=True)
    wide = _pivot_to_wide(mode, dataset, rows, run_started, version)
    csv_writer.writerow({k: wide.get(k) for k in CSV_COLUMNS})
    return wide


def _print_row(row: dict) -> None:
    test = row["test"]
    status = row.get("status", "ok")
    if status == "ok":
        wall = float(row["wall_ms"])
        wall_s = f"{wall / 1000:.2f}s" if wall >= 1000 else f"{wall:.1f}ms"
        tag = f"{wall_s:>10s}"
    else:
        tag = f"[{status}]".rjust(10)
    extras = []
    if row.get("peak_rss_mb"):
        extras.append(f"rss={row['peak_rss_mb']:.0f}MB")
    if row.get("bytes_written"):
        b = row["bytes_written"]
        extras.append(f"{b / (1 << 20):.0f}MB on disk")
    if row.get("node_count") is not None and test in ("build", "load", "mutate"):
        extras.append(f"nodes={row['node_count']:,}, edges={row.get('edge_count', '?'):,}")
    if row.get("row_count") is not None and test not in ("build", "save", "load", "mutate", "resave"):
        extras.append(f"rows={row['row_count']:,}")
    extras_str = f"  ({', '.join(extras)})" if extras else ""
    err = f"  err={row.get('error', '')}" if row.get("error") else ""
    print(f"      {test:<28s} {tag}{extras_str}{err}", flush=True)


def _summary(wide_rows: list[dict], modes: list[str], datasets: list[tuple[str, str, int]]) -> None:
    """Per-dataset totals + grand total table. Reads `total_ms` from
    each pivoted wide row."""
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)

    # (mode, dataset) -> total wall_ms (already aggregated per-row).
    totals: dict[tuple[str, str], float] = {}
    for r in wide_rows:
        try:
            totals[(r["mode"], r["dataset"])] = float(r.get("total_ms") or 0)
        except (TypeError, ValueError):
            continue

    label_w = max(len(d[0]) for d in datasets) + 2
    col_w = 14
    header = "  " + " " * label_w + "".join(f"{m:>{col_w}}" for m in modes)
    print(header)
    print("  " + " " * label_w + "".join("-" * col_w for _ in modes))

    grand = 0.0
    for label, _, _ in datasets:
        line = f"  {label:<{label_w}}"
        for m in modes:
            t = totals.get((m, label))
            if t is None:
                line += f"{'—':>{col_w}}"
            else:
                grand += t
                if t >= 1000:
                    line += f"{t / 1000:>{col_w - 1}.2f}s"
                else:
                    line += f"{t:>{col_w - 2}.0f}ms"
        print(line)

    print("  " + " " * label_w + "".join("-" * col_w for _ in modes))
    # Per-mode totals
    line = f"  {'TOTAL':<{label_w}}"
    for m in modes:
        s = sum(t for (mm, _ds), t in totals.items() if mm == m)
        if s >= 1000:
            line += f"{s / 1000:>{col_w - 1}.2f}s"
        else:
            line += f"{s:>{col_w - 2}.0f}ms"
    print(line)
    print("\n" + "=" * 88)
    if grand >= 1000:
        print(f"  GRAND TOTAL: {grand / 1000:.2f}s ({grand / 60000:.1f} min)")
    else:
        print(f"  GRAND TOTAL: {grand:.0f}ms")
    print("=" * 88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated subset of {default,mapped,disk}.",
    )
    ap.add_argument(
        "--datasets",
        default="",
        help="Comma-separated subset (e.g. wiki5m,wiki50m). Default: every dataset whose source file exists.",
    )
    ap.add_argument(
        "--memory-max",
        default=DEFAULT_MEMORY_MAX,
        help="Skip 'default' (memory) mode for datasets larger than this. Default: wiki100m.",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "benchmark_full.csv"),
        help="Output CSV path.",
    )
    ap.add_argument("--child-scenario", action="store_true", help="(internal) run the child scenario.")
    ap.add_argument("--mode")
    ap.add_argument("--dataset")
    ap.add_argument("--nt-path")
    ap.add_argument("--n-triples", type=int)
    ap.add_argument("--work-dir")
    args = ap.parse_args()

    if getattr(args, "child_scenario", False):
        _run_scenario(
            args.mode,
            args.dataset,
            args.nt_path,
            args.n_triples,
            Path(args.work_dir),
        )
        return

    requested_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = set(requested_modes) - set(MODES)
    if invalid:
        print(f"unknown modes: {sorted(invalid)}; valid: {MODES}", file=sys.stderr)
        sys.exit(2)

    requested_datasets = {s.strip() for s in args.datasets.split(",") if s.strip()}
    selected = [d for d in DATASETS if d[0] in requested_datasets] if requested_datasets else DATASETS
    selected = [d for d in selected if os.path.exists(d[1])]
    if not selected:
        print("No datasets selected (or none of the requested files exist).", file=sys.stderr)
        sys.exit(2)

    # Memory mode capacity gate
    memory_max_idx = next(
        (i for i, (label, _, _) in enumerate(DATASETS) if label == args.memory_max),
        len(DATASETS) - 1,
    )

    import kglite

    # Fix the run identity once for every row written by this
    # invocation. Children inherit via env vars.
    run_started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.environ["KGLITE_BENCH_RUN_STARTED"] = run_started
    os.environ["KGLITE_BENCH_VERSION"] = kglite.__version__

    # Sidecar error log: one tab-separated row per failed test. The
    # wide CSV's `errors` column is truncated to 120 chars and unsafe
    # for newlines/semicolons; the sidecar preserves the full text.
    err_log_path = Path(args.out).with_suffix(".errors.log")
    os.environ["KGLITE_BENCH_ERRLOG"] = str(err_log_path)

    print(f"KGLite v{kglite.__version__} — full-stack benchmark")
    print(f"  run started: {run_started}")
    print(f"  modes:    {', '.join(requested_modes)}")
    print(f"  datasets: {', '.join(d[0] for d in selected)}")
    print(f"  memory cap: {args.memory_max}")
    print(f"  output:   {args.out}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not out_path.exists() or out_path.stat().st_size == 0
    overall_start = time.perf_counter()
    wide_rows: list[dict] = []

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if fresh:
            writer.writeheader()
        for label, path, n_triples in selected:
            print(f"\n──── {label} ────────────────────────────────────────────────")
            for mode in requested_modes:
                if mode == "default":
                    ds_idx = next(i for i, d in enumerate(DATASETS) if d[0] == label)
                    if ds_idx > memory_max_idx:
                        print(f"  [{mode}]   skipped — exceeds --memory-max ({args.memory_max})")
                        continue
                print(f"  [{mode}]")
                wide = _spawn(mode, label, path, n_triples, writer, run_started, kglite.__version__)
                f.flush()
                wide_rows.append(wide)

    elapsed = time.perf_counter() - overall_start
    _summary(wide_rows, requested_modes, selected)
    print(f"\n  wall-clock end-to-end: {elapsed:.1f}s ({elapsed / 60:.1f} min)\n")


if __name__ == "__main__":
    main()
