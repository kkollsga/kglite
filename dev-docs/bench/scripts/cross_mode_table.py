"""4-column cross-mode benchmark across legal / sodir / wiki100m / wiki500m.

Per (graph, mode) measures:
  - build: from source data (legal CSVs / sodir blueprint / wiki bz2 with
           max_triples clip).
  - edit:  representative mutation (1k node SET via Cypher).
  - q_simple:  single-type lookup with LIMIT.
  - q_medium:  1-hop traversal with WHERE + ORDER BY + LIMIT.
  - q_complex: aggregation with WITH pipeline.

Each (graph, mode) runs in its own subprocess so peak RSS is scoped and
an OOM in one cell doesn't take down the rest of the run.

Wikidata builds clip the 40GB latest-truthy.nt.bz2 source via
`max_triples` — no pre-clipped files needed. Memory mode skips wiki500m
unconditionally (~30GB+ heap, infeasible on 16GB hosts) and falls
through with a fail-safe OOM marker on wiki100m if heap exhausts.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SODIR_DIR = "/Volumes/EksternalHome/Koding/MCP servers/prospect_mcp"
LEGAL_DIR = "/Volumes/EksternalHome/Koding/MCP servers/legal"
SODIR_BLUEPRINT = str(REPO / "bench/sodir_graph_config.json")
WIKI_BZ2 = "/Volumes/EksternalHome/Data/Wikidata/latest-truthy.nt.bz2"

# Modes to bench. memory = in-memory, mapped = mmap'd columns, disk = CSR + mmap
MODES = ("memory", "mapped", "disk")

# Datasets and their build size (None for legal/sodir; max_triples for wiki).
DATASETS = (
    ("legal", None, ("memory", "mapped", "disk")),
    ("sodir", None, ("memory", "mapped", "disk")),
    ("wiki100m", 100_000_000, ("memory", "mapped", "disk")),
    ("wiki500m", 500_000_000, ("mapped", "disk")),  # memory skipped: needs ~30GB+ heap
)


def _run_subprocess(code: str, timeout_s: int = 7200) -> dict:
    """Run a Python snippet in a subprocess with strict timeout. Returns dict
    with 'ok' bool, 'result' (parsed JSON from last line) or 'error' string."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            tail = "\n".join(err[-5:]) if err else f"exit={proc.returncode}"
            return {"ok": False, "error": tail}
        # Last non-empty line of stdout must be the JSON result.
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return {"ok": False, "error": "no output"}
        try:
            return {"ok": True, "result": json.loads(lines[-1])}
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"non-JSON last line: {lines[-1][:120]} ({e})"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout_s}s"}


# ---------------------------------------------------------------------------
# Per-dataset bench code (runs in subprocess). Each prints a single JSON line
# at the end with all measured wall times in milliseconds.
# ---------------------------------------------------------------------------

LEGAL_BENCH = r"""
import json, sys, time, resource, os, importlib
sys.path.insert(0, %(legal_dir)r)
import kglite

mode = %(mode)r
storage = 'default' if mode == 'memory' else mode
sys.argv = ['build_legal_graph.py', '--storage', storage]

t0 = time.perf_counter()
mod = importlib.import_module('build_legal_graph')
g = mod.main()
build_ms = (time.perf_counter() - t0) * 1000

# Edit: SET a fresh property on a chunk of nodes.
t0 = time.perf_counter()
g.cypher('MATCH (n:Law) SET n.bench_marker = 1 RETURN count(n) AS n', timeout_ms=60_000)
edit_ms = (time.perf_counter() - t0) * 1000

# Simple: single-type ordered lookup.
t0 = time.perf_counter()
list(g.cypher('MATCH (n:Law) RETURN n.id, n.title ORDER BY n.id LIMIT 10', timeout_ms=60_000))
simple_ms = (time.perf_counter() - t0) * 1000

# Medium: 1-hop traversal w/ WHERE.
t0 = time.perf_counter()
list(g.cypher(
    "MATCH (d:CourtDecision {court_level: 'hoyesterett'})-[:CITES]->(s:LawSection) "
    "RETURN d.title, s.title ORDER BY d.title, s.title LIMIT 20",
    timeout_ms=60_000,
))
medium_ms = (time.perf_counter() - t0) * 1000

# Complex: WITH pipeline + aggregation.
t0 = time.perf_counter()
list(g.cypher(
    "MATCH (d:CourtDecision)-[:CITES]->(s) WITH s, count(d) AS citations "
    "WHERE citations > 5 RETURN s.title, citations ORDER BY citations DESC LIMIT 20",
    timeout_ms=60_000,
))
complex_ms = (time.perf_counter() - t0) * 1000

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
print(json.dumps({
    'build_ms': build_ms, 'edit_ms': edit_ms,
    'simple_ms': simple_ms, 'medium_ms': medium_ms, 'complex_ms': complex_ms,
    'rss_mb': rss_mb,
}))
"""

SODIR_BENCH = r"""
import json, sys, time, resource, os
import kglite

mode = %(mode)r
blueprint = %(blueprint)r
disk_path = %(disk_path)r

kwargs = {'verbose': False, 'save': False}
if mode == 'mapped':
    kwargs['storage'] = 'mapped'
elif mode == 'disk':
    kwargs['storage'] = 'disk'
    kwargs['path'] = disk_path

t0 = time.perf_counter()
g = kglite.from_blueprint(blueprint, **kwargs)
build_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
g.cypher('MATCH (p:Prospect) SET p.bench_marker = 1 RETURN count(p) AS n', timeout_ms=120_000)
edit_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
list(g.cypher('MATCH (w:Wellbore) RETURN w.title ORDER BY w.title LIMIT 10', timeout_ms=60_000))
simple_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
list(g.cypher(
    'MATCH (f:Field)<-[:IN_FIELD]-(w:Wellbore) RETURN f.title, w.title '
    'ORDER BY f.title, w.title LIMIT 20',
    timeout_ms=60_000,
))
medium_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
list(g.cypher(
    'MATCH (l:Licence)-[:HAS_LICENSEE]->(c:Company) '
    'WITH c, count(l) AS n WHERE n > 5 '
    'RETURN c.title, n ORDER BY n DESC, c.title LIMIT 20',
    timeout_ms=60_000,
))
complex_ms = (time.perf_counter() - t0) * 1000

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
print(json.dumps({
    'build_ms': build_ms, 'edit_ms': edit_ms,
    'simple_ms': simple_ms, 'medium_ms': medium_ms, 'complex_ms': complex_ms,
    'rss_mb': rss_mb,
}))
"""

WIKI_BENCH = r"""
import json, time, resource, os, tempfile, shutil
import kglite

mode = %(mode)r
max_triples = %(max_triples)d
nt_path = %(nt_path)r
disk_path = %(disk_path)r

t0 = time.perf_counter()
if mode == 'memory':
    g = kglite.KnowledgeGraph()
    g.load_ntriples(nt_path, max_triples=max_triples, languages=['en'], verbose=False)
elif mode == 'mapped':
    g = kglite.KnowledgeGraph(storage='mapped')
    g.load_ntriples(nt_path, max_triples=max_triples, languages=['en'], verbose=False)
elif mode == 'disk':
    g = kglite.KnowledgeGraph(storage='disk', path=disk_path)
    g.load_ntriples(nt_path, max_triples=max_triples, languages=['en'], verbose=False)
build_ms = (time.perf_counter() - t0) * 1000

# Edit: SET on a sampled subset (don't SET on all to avoid the bench
# itself dominating the timing in larger graphs — pick anchored Q42).
t0 = time.perf_counter()
g.cypher("MATCH (a {nid: 'Q42'}) SET a.bench_marker = 1 RETURN count(a) AS n", timeout_ms=60_000)
edit_ms = (time.perf_counter() - t0) * 1000

# Simple: anchored 1-hop (Q42 = Douglas Adams, P31 = instance of).
t0 = time.perf_counter()
list(g.cypher("MATCH (a {nid: 'Q42'})-[:P31]->(b) RETURN a.title, b.title LIMIT 50", timeout_ms=30_000))
simple_ms = (time.perf_counter() - t0) * 1000

# Medium: anchored 2-hop traversal (instance-of → subclass-of).
t0 = time.perf_counter()
list(g.cypher(
    "MATCH (a {nid: 'Q42'})-[:P31]->(b)-[:P279]->(c) RETURN a.title, c.title LIMIT 10",
    timeout_ms=30_000,
))
medium_ms = (time.perf_counter() - t0) * 1000

# Complex: aggregation over a P31 class — counts instance-of-X edges and
# returns the top 10 by frequency. Doesn't depend on the auto type rename.
t0 = time.perf_counter()
list(g.cypher(
    "MATCH ()-[:P31]->(c) RETURN c.title, count(*) AS k ORDER BY k DESC, c.title LIMIT 10",
    timeout_ms=120_000,
))
complex_ms = (time.perf_counter() - t0) * 1000

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
print(json.dumps({
    'build_ms': build_ms, 'edit_ms': edit_ms,
    'simple_ms': simple_ms, 'medium_ms': medium_ms, 'complex_ms': complex_ms,
    'rss_mb': rss_mb,
}))
"""


def run_legal(mode: str) -> dict:
    return _run_subprocess(
        LEGAL_BENCH % {"legal_dir": LEGAL_DIR, "mode": mode}, timeout_s=600
    )


def run_sodir(mode: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="sodir_bench_") as tmp:
        disk_path = os.path.join(tmp, "sodir_disk")
        return _run_subprocess(
            SODIR_BENCH % {
                "blueprint": SODIR_BLUEPRINT,
                "mode": mode,
                "disk_path": disk_path,
            },
            timeout_s=900,
        )


def run_wiki(mode: str, max_triples: int, label: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"wiki_{label}_") as tmp:
        disk_path = os.path.join(tmp, f"{label}_disk")
        timeout = 7200  # 2h cap per cell — wiki500m mapped builds may take ~30-60min.
        return _run_subprocess(
            WIKI_BENCH % {
                "mode": mode,
                "max_triples": max_triples,
                "nt_path": WIKI_BZ2,
                "disk_path": disk_path,
            },
            timeout_s=timeout,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="legal,sodir,wiki100m,wiki500m",
                        help="Comma-separated subset of datasets to run.")
    parser.add_argument("--out", default="bench/cross_mode_table.csv",
                        help="CSV output path (relative to repo root).")
    args = parser.parse_args()

    requested = set(args.datasets.split(","))
    out_path = REPO / args.out
    rows: list[dict] = []

    for label, max_triples, modes in DATASETS:
        if label not in requested:
            continue
        for mode in modes:
            print(f"\n[{label}/{mode}] running...", flush=True)
            t0 = time.perf_counter()
            if label == "legal":
                res = run_legal(mode)
            elif label == "sodir":
                res = run_sodir(mode)
            else:
                res = run_wiki(mode, max_triples, label)
            elapsed = time.perf_counter() - t0
            row = {"dataset": label, "mode": mode, "wall_s": round(elapsed, 1)}
            if res["ok"]:
                row.update(res["result"])
                row["status"] = "ok"
                print(
                    f"  ok in {elapsed:.0f}s — "
                    f"build={row['build_ms']:.0f}ms edit={row['edit_ms']:.0f}ms "
                    f"simple={row['simple_ms']:.0f}ms medium={row['medium_ms']:.0f}ms "
                    f"complex={row['complex_ms']:.0f}ms rss={row['rss_mb']:.0f}MB",
                    flush=True,
                )
            else:
                row["status"] = "fail"
                row["error"] = res["error"][:200]
                print(f"  FAIL in {elapsed:.0f}s — {res['error'][:200]}", flush=True)
            rows.append(row)
            # Persist after every cell so partial results survive a crash.
            _write_csv(out_path, rows)

    print(f"\nResults written to {out_path}")
    print_table(rows)
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = ["dataset", "mode", "status", "build_ms", "edit_ms", "simple_ms",
            "medium_ms", "complex_ms", "rss_mb", "wall_s", "error"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def print_table(rows: list[dict]) -> None:
    """4-col table per dataset: op, memory, mapped, disk."""
    by_ds: dict[str, dict] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["mode"]] = r

    ops = [
        ("build_ms", "build"),
        ("edit_ms", "edit"),
        ("simple_ms", "simple"),
        ("medium_ms", "medium"),
        ("complex_ms", "complex"),
    ]

    for ds, modes in by_ds.items():
        print(f"\n{ds}")
        print(f"  {'op':<10}{'memory':>14}{'mapped':>14}{'disk':>14}")
        print(f"  {'-' * 10}{'-' * 14}{'-' * 14}{'-' * 14}")
        for op_key, op_name in ops:
            cells = []
            for m in ("memory", "mapped", "disk"):
                row = modes.get(m)
                if not row or row.get("status") != "ok":
                    cells.append("—" if not row else "FAIL")
                else:
                    cells.append(f"{row[op_key]:.0f}ms")
            print(f"  {op_name:<10}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}")


if __name__ == "__main__":
    sys.exit(main())
