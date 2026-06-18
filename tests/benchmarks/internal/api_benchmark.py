"""KGLite API Benchmark — tests API compatibility, performance, and correctness
across memory, mapped, and disk storage modes on 4 datasets.

Run all modes + compare:     python benchmarks/api_benchmark.py
Run single mode:             python benchmarks/api_benchmark.py disk

Ingest-measurement mode (PR0 baseline for disk-graph-improvement-plan):
    maturin develop --release         # debug builds are 7–8× slower
    python benchmarks/api_benchmark.py --measure-ingest

Records bytes_written + peak_rss + wall_time for three scenarios
(one-shot, incremental-10-save, mutation-stream) on the disk backend,
streaming results to benchmarks/ingest_baseline.csv. This is the
measurement gate for PR1 (segmented CSR) and PR2 (edge-props off heap).
Always use a release build — debug-mode numbers are not comparable to
the PyPI-released baseline.

Memory results are ground truth. Mismatches in mapped/disk are flagged as errors.
"""

import csv
from datetime import datetime, timezone
import gc
import hashlib  # noqa: I001
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kglite  # noqa: E402
from kglite import KnowledgeGraph, load  # noqa: E402

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
LEGAL_BUILD_SCRIPT = Path("/Volumes/EksternalHome/Koding/MCP servers/legal")
SODIR_BUILD_SCRIPT = Path("/Volumes/EksternalHome/Koding/MCP servers")
SODIR_BLUEPRINT = str(Path(__file__).resolve().parent / "sodir_graph_config.json")
WIKIDATA_500K = "/Volumes/EksternalHome/Data/Wikidata/test_500k.nt.zst"
WIKIDATA_5M = "/Volumes/EksternalHome/Data/Wikidata/test_5M.nt.zst"
WIKIDATA_50M = "/Volumes/EksternalHome/Data/Wikidata/test_50M.nt.zst"
WIKIDATA_100M = "/Volumes/EksternalHome/Data/Wikidata/test_100M.nt.zst"
WIKIDATA_200M = "/Volumes/EksternalHome/Data/Wikidata/test_200M.nt.zst"
WIKIDATA_500M = "/Volumes/EksternalHome/Data/Wikidata/test_500M.nt.zst"
WIKIDATA_1000M = "/Volumes/EksternalHome/Data/Wikidata/test_1000M.nt.zst"

# (label, path, nominal triple count)
# Nominal count is the number of lines the subset was sliced at via
# `make_wikidata_subset.sh`; the actual graph-level node/edge counts
# are much smaller because load_ntriples collapses labels/types into
# node attributes. Used for triples/s reporting on build.
WIKIDATA_SUBSETS = [
    ("wiki500k", WIKIDATA_500K, 500_000),
    ("wiki5m", WIKIDATA_5M, 5_000_000),
    ("wiki50m", WIKIDATA_50M, 50_000_000),
    ("wiki100m", WIKIDATA_100M, 100_000_000),
    ("wiki200m", WIKIDATA_200M, 200_000_000),
    ("wiki500m", WIKIDATA_500M, 500_000_000),
    ("wiki1000m", WIKIDATA_1000M, 1_000_000_000),
]

# Language filter for the auto_type rename pass (see WIKIDATA_QUERIES note).
# Override at the CLI with `--languages en,de` or `--languages ""` for all.
WIKIDATA_LANGUAGES: list[str] = ["en"]

TIMEOUT_MS = 30_000


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
def checksum(result):
    """Hash first 20 rows of a query result for cross-mode comparison."""
    if result is None:
        return None
    rows = result[:20] if hasattr(result, "__getitem__") else []
    return hashlib.md5(repr(sorted(str(r) for r in rows)).encode()).hexdigest()[:12]


def timed(fn):
    """Call fn(), return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Build functions — same API for all modes
# ---------------------------------------------------------------------------
def _make_graph(mode, disk_path=None):
    """Create a KnowledgeGraph with the right storage kwargs."""
    kwargs = {}
    if mode == "disk":
        kwargs["storage"] = "disk"
        kwargs["path"] = disk_path
    elif mode == "mapped":
        kwargs["storage"] = "mapped"
    return KnowledgeGraph(**kwargs)


def _save_path(dataset, mode):
    """Temp path for save/load cycle."""
    base = Path(tempfile.gettempdir()) / f"kglite_apibench_{dataset}_{mode}"
    if mode == "disk":
        base.mkdir(parents=True, exist_ok=True)
        return str(base)
    return str(base.with_suffix(".kgl"))


def build_legal(mode):
    """Build legal graph by importing the existing build script."""
    old_argv = sys.argv
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(LEGAL_BUILD_SCRIPT))
        sys.argv = ["build_legal_graph.py", "--storage", mode if mode != "memory" else "default"]
        import importlib

        mod = importlib.import_module("build_legal_graph")
        importlib.reload(mod)  # ensure fresh state on repeated calls
        return mod.main()
    finally:
        sys.argv = old_argv
        sys.path = old_path


def build_prospect(mode):
    """Build prospect graph from blueprint.
    Assumes CSVs are already preprocessed (run build_sodir_graph.py once manually).
    """
    kwargs = {"verbose": False, "save": False}
    if mode == "disk":
        disk_path = _save_path("prospect", mode)
        kwargs["storage"] = "disk"
        kwargs["path"] = disk_path
    elif mode == "mapped":
        kwargs["storage"] = "mapped"

    return kglite.from_blueprint(SODIR_BLUEPRINT, **kwargs)


def build_wikidata(mode, nt_path, label):
    """Build wikidata graph from N-Triples."""
    disk_path = _save_path(label, mode) if mode == "disk" else None
    g = _make_graph(mode, disk_path)
    load_kwargs: dict = {}
    if WIKIDATA_LANGUAGES:
        load_kwargs["languages"] = list(WIKIDATA_LANGUAGES)
    g.load_ntriples(nt_path, **load_kwargs)
    return g


# ---------------------------------------------------------------------------
# Query suites per dataset
# ---------------------------------------------------------------------------
# Query tuples: (name, query, deterministic)
# deterministic=True: has ORDER BY → compare checksums (exact row match)
# deterministic=False: no ORDER BY → compare row counts only (order is arbitrary)

SHARED_QUERIES = [
    ("count(n)", "MATCH (n) RETURN count(n) AS cnt", True),
    (
        "count by type",
        "MATCH (n) RETURN n.type AS t, count(n) AS c ORDER BY c DESC, t LIMIT 10",
        True,
    ),
    (
        "edge count by type",
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC, t LIMIT 10",
        True,
    ),
]

LEGAL_QUERIES = [
    ("Law lookup", "MATCH (n:Law) RETURN n.id, n.title ORDER BY n.id LIMIT 5", True),
    (
        "Court cites LawSection",
        "MATCH (d:CourtDecision {court_level: 'hoyesterett'})-[:CITES]->(s:LawSection) "
        "RETURN d.title, s.title ORDER BY d.title, s.title LIMIT 10",
        True,
    ),
    (
        "2-hop cites->law",
        "MATCH (d:CourtDecision {court_level: 'hoyesterett'})-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) "
        "RETURN d.title, l.title ORDER BY d.title, l.title LIMIT 10",
        True,
    ),
    (
        "cases per keyword",
        "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) "
        "RETURN k.title, count(d) AS c ORDER BY c DESC, k.title LIMIT 10",
        True,
    ),
    (
        "OPTIONAL MATCH",
        "MATCH (l:Law) OPTIONAL MATCH (l)-[:GOVERNED_BY]->(d:Department) "
        "RETURN l.title, d.title ORDER BY l.title LIMIT 10",
        True,
    ),
    (
        "EXISTS subquery",
        "MATCH (d:CourtDecision {court_level: 'hoyesterett'}) "
        "WHERE EXISTS { MATCH (d)-[:CITES]->() } RETURN d.title ORDER BY d.title LIMIT 5",
        True,
    ),
    (
        "WITH pipeline",
        "MATCH (d:CourtDecision)-[:CITES]->(s) WITH s, count(d) AS citations "
        "WHERE citations > 5 RETURN s.title, citations ORDER BY citations DESC LIMIT 10",
        True,
    ),
    (
        "CONTAINS search",
        "MATCH (n:Law) WHERE n.title CONTAINS 'straff' RETURN n.title ORDER BY n.title LIMIT 20",
        True,
    ),
]

PROSPECT_QUERIES = [
    ("Wellbore lookup", "MATCH (w:Wellbore) RETURN w.title ORDER BY w.title LIMIT 5", True),
    ("Field lookup", "MATCH (f:Field) RETURN f.title ORDER BY f.title LIMIT 5", True),
    (
        "Licence connections",
        "MATCH (l:Licence)-[r]->(n) RETURN l.title, type(r), n.title ORDER BY l.title, n.title LIMIT 10",
        True,
    ),
    ("Discovery lookup", "MATCH (d:Discovery) RETURN d.title ORDER BY d.title LIMIT 5", True),
    (
        "Field->Wellbore hop",
        "MATCH (f:Field)<-[:IN_FIELD]-(w:Wellbore) RETURN f.title, w.title ORDER BY f.title, w.title LIMIT 10",
        True,
    ),
]

# Queries assume the build was run with `languages=["en"]` so the
# auto_type rename produces English type names (Q5 → "human", etc.).
# Larger queries are anchored to Q42 (Douglas Adams) so the cost is
# bounded at every dataset scale instead of doing a full-graph sort.
WIKIDATA_QUERIES = [
    # Anchored 1-hop: bounded cost regardless of graph size.
    (
        "P31 from Q42 LIMIT 50",
        "MATCH (a {nid: 'Q42'})-[:P31]->(b) RETURN a.title, b.title LIMIT 50",
        True,
    ),
    # Node-type lookup — uses the English label `human` (was Q5 pre-rename).
    (
        "human (Q5) lookup",
        "MATCH (h:human) RETURN h.title ORDER BY h.title LIMIT 10",
        True,
    ),
    # Aggregation with LIMIT — counts every :P31 edge and returns top 10 classes.
    (
        "P31 class counts",
        "MATCH ()-[:P31]->(c) RETURN c.title, count(*) AS k ORDER BY k DESC, c.title LIMIT 10",
        True,
    ),
    # Two-hop through the type hierarchy, anchored.
    (
        "2-hop P31 + P279 from Q42",
        "MATCH (a {nid: 'Q42'})-[:P31]->(b)-[:P279]->(c) RETURN a.title, c.title LIMIT 10",
        True,
    ),
    # Different edge type (P27 = citizenship) anchored on type human.
    (
        "Human citizenship (P27)",
        "MATCH (h:human)-[:P27]->(c) RETURN h.title, c.title ORDER BY h.title, c.title LIMIT 10",
        True,
    ),
    # OPTIONAL MATCH path — left-joins P27 onto every human.
    (
        "OPTIONAL citizenship",
        "MATCH (h:human) OPTIONAL MATCH (h)-[:P27]->(c) RETURN h.title, c.title ORDER BY h.title LIMIT 10",
        True,
    ),
    # EXISTS subquery — humans with any citizenship edge.
    (
        "EXISTS P27",
        "MATCH (h:human) WHERE EXISTS { MATCH (h)-[:P27]->() } RETURN h.title ORDER BY h.title LIMIT 10",
        True,
    ),
    # WITH + post-aggregation sort — countries by number of citizens.
    (
        "WITH P27 count",
        "MATCH (h:human)-[:P27]->(c) WITH c, count(h) AS k RETURN c.title, k ORDER BY k DESC, c.title LIMIT 10",
        True,
    ),
]


# ---------------------------------------------------------------------------
# Fluent API tests
# ---------------------------------------------------------------------------
def fluent_tests(g, dataset):
    """Run fluent API tests, return list of result dicts."""
    results = []
    info = g.graph_info()
    first_type = next((t for t in info.get("node_types", {}).keys()), None)

    def record(name, fn):
        try:
            val, ms = timed(fn)
            cs = hashlib.md5(repr(val).encode()).hexdigest()[:12] if val is not None else None
            results.append(
                {
                    "dataset": dataset,
                    "test": f"fluent:{name}",
                    "time_ms": round(ms, 1),
                    "status": "ok",
                    "row_count": None,
                    "checksum": cs,
                    "error": None,
                }
            )
        except BaseException as e:
            results.append(
                {
                    "dataset": dataset,
                    "test": f"fluent:{name}",
                    "time_ms": 0,
                    "status": "ERROR",
                    "row_count": None,
                    "checksum": None,
                    "error": str(e)[:80],
                }
            )

    record("graph_info", lambda: g.graph_info())
    record("describe", lambda: g.describe())

    if first_type:
        record(f"select({first_type}).len", lambda: g.select(first_type).len())
        record(f"select({first_type}).to_df[:5]", lambda: g.select(first_type).to_df().head(5).to_dict())

    return results


# ---------------------------------------------------------------------------
# Run one dataset
# ---------------------------------------------------------------------------
def run_dataset(mode, dataset, build_fn, extra_queries, build_metadata=None):
    """Build, save, load, query one dataset. Returns list of result dicts.

    ``build_metadata`` optionally carries extra numbers to attach to the
    ``build`` result row. Only key currently used is ``nominal_triples``
    for wiki datasets, which the comparison output reads to compute
    triples/s.
    """
    results = []
    build_metadata = build_metadata or {}

    def record(test_name, time_ms, status="ok", row_count=None, cs=None, error=None, extra=None):
        tag = f"{time_ms:>8.1f}ms" if status == "ok" else f"{status:>8s}"
        row_str = f" rows={row_count}" if row_count is not None else ""
        err_str = f"  {error[:50]}" if error else ""
        # Inline triples/s on the build row for wiki datasets.
        rate_str = ""
        if extra and "triples_per_sec" in extra:
            rate_str = f"  ({extra['triples_per_sec']:,.0f} triples/s)"
        print(f"    {test_name:40s} {tag}{row_str}{rate_str}{err_str}", flush=True)
        row = {
            "dataset": dataset,
            "test": test_name,
            "time_ms": round(time_ms, 1),
            "status": status,
            "row_count": row_count,
            "checksum": cs,
            "error": error,
        }
        if extra:
            row.update(extra)
        results.append(row)

    # ── Build ──
    g = None  # noqa: F841 — set in try, used in later blocks
    nc, ec = 0, 0
    try:
        g, ms = timed(build_fn)
        info = g.graph_info()
        nc, ec = info["node_count"], info["edge_count"]
        extra = {}
        if "nominal_triples" in build_metadata and ms > 0:
            n_triples = build_metadata["nominal_triples"]
            extra["nominal_triples"] = n_triples
            extra["triples_per_sec"] = n_triples / (ms / 1000.0)
        record("build", ms, row_count=nc, extra=extra or None)
    except BaseException as e:
        record("build", 0, status="ERROR", error=str(e)[:120])
        return results

    # ── Save ──
    save_path = _save_path(dataset, mode)
    try:
        _, ms = timed(lambda: g.save(save_path))  # noqa: B023, F821
        record("save", ms)
    except BaseException as e:
        record("save", 0, status="ERROR", error=str(e)[:120])

    # ── Load ──
    try:
        g2, ms = timed(lambda: load(save_path))
        info2 = g2.graph_info()
        nc2, ec2 = info2["node_count"], info2["edge_count"]
        if nc2 != nc or ec2 != ec:
            record("load", ms, status="MISMATCH", error=f"nodes {nc}->{nc2}, edges {ec}->{ec2}")
        else:
            record("load", ms, row_count=nc2)
        g = g2  # use loaded graph for queries
    except BaseException as e:
        record("load", 0, status="ERROR", error=str(e)[:80])

    # ── Cypher queries ──
    all_queries = SHARED_QUERIES + extra_queries
    for name, query, deterministic in all_queries:
        try:
            r, ms = timed(lambda q=query: g.cypher(q, timeout_ms=TIMEOUT_MS))  # noqa: B023, F821
            rc = len(r) if r is not None else 0
            cs = checksum(r) if deterministic else None
            record(name, ms, row_count=rc, cs=cs)
        except BaseException as e:
            err = str(e)
            status = "TIMEOUT" if "timed out" in err.lower() else "ERROR"
            record(name, 0, status=status, error=err[:80])

    # ── describe() ──
    try:
        d, ms = timed(lambda: g.describe())  # noqa: B023, F821
        record("describe()", ms, row_count=len(d), cs=hashlib.md5(d.encode()).hexdigest()[:12])
    except BaseException as e:
        record("describe()", 0, status="ERROR", error=str(e)[:80])

    # ── Fluent API ──
    results.extend(fluent_tests(g, dataset))

    # ── Cleanup ──
    del g
    try:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path, ignore_errors=True)
        elif os.path.exists(save_path):
            os.remove(save_path)
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Run all datasets for one mode
# ---------------------------------------------------------------------------
def run(mode):
    """Run all 4 datasets for the given storage mode. Returns list of result dicts."""
    storage = mode if mode != "memory" else "default"
    print(f"\n{'=' * 70}")
    print(f"  MODE: {mode}")
    print(f"{'=' * 70}")

    all_results = []

    # 1. Legal
    print(f"\n  [legal] ({mode})")
    all_results.extend(run_dataset(mode, "legal", lambda: build_legal(storage), LEGAL_QUERIES))

    # 2. Prospect
    print(f"\n  [prospect] ({mode})")
    all_results.extend(run_dataset(mode, "prospect", lambda: build_prospect(storage), PROSPECT_QUERIES))

    # 3+. Wikidata subsets. Each entry auto-skipped if the .nt.zst file
    # is absent on the current machine, so developers without the full
    # corpus can still run the core legal/prospect benchmarks. Nominal
    # triple counts (from `make_wikidata_subset.sh`) drive the
    # triples/s display in the comparison output.
    for label, path, nominal_triples in WIKIDATA_SUBSETS:
        if not os.path.exists(path):
            print(f"\n  [{label}] ({mode}) — skipped (file not found: {path})")
            continue
        print(f"\n  [{label}] ({mode})")
        all_results.extend(
            run_dataset(
                mode,
                label,
                lambda p=path, lbl=label: build_wikidata(storage, p, lbl),
                WIKIDATA_QUERIES,
                build_metadata={"nominal_triples": nominal_triples},
            )
        )

    return all_results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare(all_results):
    """Compare results across modes. Memory is ground truth."""
    modes = list(all_results.keys())
    if "memory" not in modes:
        print("WARNING: no memory results to use as ground truth")
        return

    # Index results by (dataset, test)
    indexed = {}
    for mode in modes:
        for r in all_results[mode]:
            key = (r["dataset"], r["test"])
            indexed.setdefault(key, {})[mode] = r

    # Print comparison table
    print(f"\n{'=' * 100}")
    print("  COMPARISON (memory = ground truth)")
    print(f"{'=' * 100}")

    header_modes = [m for m in modes if m != "memory"]
    print(f"  {'dataset':<12s} {'test':<30s} {'memory':>10s}", end="")
    for m in header_modes:
        print(f" {m:>10s}", end="")
    print(f"  {'match':>6s}")
    print(f"  {'-' * 12} {'-' * 30} {'-' * 10}", end="")
    for _ in header_modes:
        print(f" {'-' * 10}", end="")
    print(f"  {'-' * 6}")

    total = 0
    passed = 0
    mismatches = []

    for key in sorted(indexed.keys()):
        dataset, test = key
        mem = indexed[key].get("memory")
        if not mem:
            continue

        total += 1
        mem_time = f"{mem['time_ms']:.0f}ms" if mem["status"] == "ok" else mem["status"]
        print(f"  {dataset:<12s} {test:<30s} {mem_time:>10s}", end="")

        match = True
        for m in header_modes:
            other = indexed[key].get(m)
            if not other:
                print(f" {'---':>10s}", end="")
                continue

            other_time = f"{other['time_ms']:.0f}ms" if other["status"] == "ok" else other["status"]
            print(f" {other_time:>10s}", end="")

            # Skip checksum for: graph_info (mode-specific metadata),
            # and non-deterministic queries (checksum=None means no ORDER BY)
            skip_checksum = "graph_info" in test or not mem.get("checksum")
            if mem["status"] == "ok" and other["status"] == "ok":
                if mem["row_count"] != other["row_count"]:
                    match = False
                    mismatches.append((dataset, test, m, f"rows: {mem['row_count']} vs {other['row_count']}"))
                elif (
                    not skip_checksum and mem["checksum"] and other["checksum"] and mem["checksum"] != other["checksum"]
                ):
                    match = False
                    mismatches.append((dataset, test, m, f"checksum: {mem['checksum']} vs {other['checksum']}"))
            elif mem["status"] != other["status"]:
                match = False
                mismatches.append((dataset, test, m, f"status: {mem['status']} vs {other['status']}"))

        if match:
            passed += 1
        print(f"  {'OK':>6s}" if match else f"  {'FAIL':>6s}")

    print(f"\n  SUMMARY: {passed}/{total} pass", end="")
    if mismatches:
        print(f", {len(mismatches)} mismatches:")
        for ds, test, mode, detail in mismatches:
            print(f"    {ds}/{test} [{mode}]: {detail}")
    else:
        print(" -- all results match")

    # Build throughput — triples/s per wiki dataset × mode.
    # Rows are sparse: only wiki datasets record nominal_triples.
    wiki_rows = []
    for mode in modes:
        for r in all_results[mode]:
            if r["test"] == "build" and r.get("nominal_triples") and r["status"] == "ok":
                wiki_rows.append((r["dataset"], mode, r["nominal_triples"], r["time_ms"]))
    if wiki_rows:
        print(f"\n{'=' * 100}")
        print("  BUILD THROUGHPUT (N-Triples ingest)")
        print(f"{'=' * 100}")
        print(f"  {'dataset':<12s} {'mode':<8s} {'triples':>12s} {'wall_s':>10s} {'triples/s':>14s}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 14}")
        for ds, mode, n_tri, ms in sorted(wiki_rows):
            rate = n_tri / (ms / 1000.0) if ms > 0 else 0
            print(f"  {ds:<12s} {mode:<8s} {n_tri:>12,} {ms / 1000.0:>10.2f} {rate:>14,.0f}")


# ---------------------------------------------------------------------------
# Ingest-measurement harness (PR0 of disk-graph-improvement-plan)
#
# Captures bytes_written + peak_rss + wall_time across three scenarios
# (one-shot, incremental, mutation-stream) on disk-backed storage.
# Streams to benchmarks/ingest_baseline.csv — the measurement gate PR1 and PR2
# compare against. See dev-documentation/disk-graph-improvement-plan.md.
#
# Scenarios use a synthetic graph (500k nodes, 1M edges, 10 chunks) for
# reproducibility. Optional --ingest-dataset wiki5m runs on the N-Triples
# corpus instead (requires /Volumes/EksternalHome/Data/Wikidata/test_5M.nt.zst).
# ---------------------------------------------------------------------------

BASELINE_CSV = Path(__file__).resolve().parent / "ingest_baseline.csv"
BASELINE_COLUMNS = [
    "timestamp",
    "commit_sha",
    "scenario",
    "dataset",
    "storage_mode",
    "n_nodes",
    "n_edges",
    "n_chunks",
    "bytes_written",
    "peak_rss_mb",
    "rss_delta_mb",
    "wall_time_ms",
]


def _rss_mb() -> float:
    """Peak RSS in MB. ru_maxrss is bytes on macOS, KB on Linux."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def _dir_size_bytes(path: str) -> int:
    """Sum of file sizes under path (or size of single file). 0 if missing."""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(p):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _git_sha() -> str:
    """Short commit SHA; 'unknown' if not in a git tree."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _append_baseline_row(row: dict) -> None:
    """Stream one result row to benchmarks/ingest_baseline.csv. Creates header if new."""
    new_file = not BASELINE_CSV.exists()
    with BASELINE_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BASELINE_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Synthetic data for reproducible scenarios
# ---------------------------------------------------------------------------

SYNTH_NODES = 500_000
SYNTH_EDGES = 1_000_000
SYNTH_CHUNKS = 10


def _synthetic_chunks(n_nodes=SYNTH_NODES, n_edges=SYNTH_EDGES, n_chunks=SYNTH_CHUNKS, seed=42):
    """Yield n_chunks of (nodes_df, edges_df). Deterministic given seed.

    Nodes: Person(id, title, age, category). Edges: KNOWS(src_id, tgt_id, weight).
    Edges in chunk i only reference nodes in chunks 0..=i so source/target always exist.
    """
    import pandas as pd

    rng = random.Random(seed)
    nodes_per_chunk = n_nodes // n_chunks
    edges_per_chunk = n_edges // n_chunks

    for ci in range(n_chunks):
        node_start = ci * nodes_per_chunk
        node_end = node_start + nodes_per_chunk
        node_ids = list(range(node_start, node_end))
        nodes_df = pd.DataFrame(
            {
                "id": node_ids,
                "title": [f"P_{i:07d}" for i in node_ids],
                "age": [rng.randint(18, 90) for _ in node_ids],
                "category": [f"cat_{rng.randint(0, 19)}" for _ in node_ids],
            }
        )

        # Edges: source in this chunk, target in any chunk 0..=ci
        cap = node_end  # exclusive upper bound on node ids seen so far
        src = [rng.randint(node_start, node_end - 1) for _ in range(edges_per_chunk)]
        tgt = [rng.randint(0, cap - 1) for _ in range(edges_per_chunk)]
        edges_df = pd.DataFrame(
            {
                "src_id": src,
                "tgt_id": tgt,
                "weight": [round(rng.random(), 3) for _ in range(edges_per_chunk)],
            }
        )
        yield nodes_df, edges_df


def _write_synth_chunk(g, nodes_df, edges_df):
    g.add_nodes(nodes_df, "Person", "id", "title")
    g.add_connections(edges_df, "KNOWS", "Person", "src_id", "Person", "tgt_id")


def _clean(path: str) -> None:
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists():
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


def _scenario_one_shot(mode: str, dataset: str) -> dict:
    """Build whole synthetic graph in memory, save once. Measures cold build+save."""
    save_path = str(Path(tempfile.gettempdir()) / f"ingest_oneshot_{dataset}_{mode}")
    _clean(save_path)

    gc.collect()
    pre_rss = _rss_mb()
    t0 = time.perf_counter()

    g = _make_graph(mode, save_path if mode == "disk" else None)
    for nodes_df, edges_df in _synthetic_chunks():
        _write_synth_chunk(g, nodes_df, edges_df)
    g.save(save_path)

    wall_ms = (time.perf_counter() - t0) * 1000
    post_rss = _rss_mb()
    bytes_written = _dir_size_bytes(save_path)

    del g
    gc.collect()
    _clean(save_path)

    return {
        "scenario": "one_shot",
        "dataset": dataset,
        "storage_mode": mode,
        "bytes_written": bytes_written,
        "peak_rss_mb": round(post_rss, 1),
        "rss_delta_mb": round(post_rss - pre_rss, 1),
        "wall_time_ms": round(wall_ms, 1),
    }


def _scenario_incremental(mode: str, dataset: str) -> dict:
    """Add one chunk, save, repeat. Measures write amplification on disk mode.

    bytes_written = sum of post-save directory sizes across all saves,
    which equals cumulative work if each save does a full rewrite. This
    is the PR1 gate: target ≤2× of final graph size.
    """
    save_path = str(Path(tempfile.gettempdir()) / f"ingest_incremental_{dataset}_{mode}")
    _clean(save_path)

    gc.collect()
    pre_rss = _rss_mb()
    t0 = time.perf_counter()

    g = _make_graph(mode, save_path if mode == "disk" else None)
    total_bytes = 0
    for nodes_df, edges_df in _synthetic_chunks():
        _write_synth_chunk(g, nodes_df, edges_df)
        g.save(save_path)
        total_bytes += _dir_size_bytes(save_path)

    wall_ms = (time.perf_counter() - t0) * 1000
    post_rss = _rss_mb()

    del g
    gc.collect()
    _clean(save_path)

    return {
        "scenario": "incremental",
        "dataset": dataset,
        "storage_mode": mode,
        "bytes_written": total_bytes,
        "peak_rss_mb": round(post_rss, 1),
        "rss_delta_mb": round(post_rss - pre_rss, 1),
        "wall_time_ms": round(wall_ms, 1),
    }


def _scenario_build_only(mode: str, dataset: str) -> dict:
    """Internal: build synthetic graph and save. Used to pre-populate mutation_stream input.

    Uses the PREBUILD_OUT env var to decide the destination path so the
    orchestrator can pass a stable location across subprocess invocations.
    Emits bytes_written so the orchestrator can report something useful
    if called directly (not used by the three official scenarios).
    """
    out = os.environ.get("INGEST_PREBUILD_OUT")
    if not out:
        raise RuntimeError("build_only requires INGEST_PREBUILD_OUT env var")
    _clean(out)

    t0 = time.perf_counter()
    g = _make_graph(mode, out if mode == "disk" else None)
    for nodes_df, edges_df in _synthetic_chunks():
        _write_synth_chunk(g, nodes_df, edges_df)
    g.save(out)
    wall_ms = (time.perf_counter() - t0) * 1000
    post_rss = _rss_mb()

    return {
        "scenario": "build_only",
        "dataset": dataset,
        "storage_mode": mode,
        "bytes_written": _dir_size_bytes(out),
        "peak_rss_mb": round(post_rss, 1),
        "rss_delta_mb": 0.0,
        "wall_time_ms": round(wall_ms, 1),
    }


def _scenario_mutation_stream(mode: str, dataset: str, n_mutations: int = 100) -> dict:
    """Load pre-built graph, run N mutations via add_nodes, save. Measures overflow-compact path.

    Expects ``INGEST_PREBUILT_SOURCE`` env var pointing at a graph the
    orchestrator already built in a separate subprocess (so peak RSS in
    this process reflects only reload + mutate + save, not the build).
    """
    import pandas as pd

    source = os.environ.get("INGEST_PREBUILT_SOURCE")
    if not source or not Path(source).exists():
        raise RuntimeError("mutation_stream requires INGEST_PREBUILT_SOURCE env var pointing at a pre-built graph")

    # Copy pre-built graph to a working path so the pristine source can be
    # reused (or cleaned up) by the orchestrator. Copy time is outside
    # the measurement window on purpose — it's orchestration, not ingest.
    working = str(Path(tempfile.gettempdir()) / f"ingest_mut_work_{dataset}_{mode}_{os.getpid()}")
    _clean(working)
    if Path(source).is_dir():
        shutil.copytree(source, working)
    else:
        shutil.copy2(source, working)

    gc.collect()
    pre_rss = _rss_mb()
    t0 = time.perf_counter()

    g = load(working)
    mut_nodes = pd.DataFrame(
        {
            "id": list(range(SYNTH_NODES, SYNTH_NODES + n_mutations)),
            "title": [f"M_{i:07d}" for i in range(n_mutations)],
            "age": [30] * n_mutations,
            "category": ["mutated"] * n_mutations,
        }
    )
    g.add_nodes(mut_nodes, "Person", "id", "title")
    g.save(working)

    wall_ms = (time.perf_counter() - t0) * 1000
    post_rss = _rss_mb()
    bytes_written = _dir_size_bytes(working)

    del g
    gc.collect()
    _clean(working)

    return {
        "scenario": "mutation_stream",
        "dataset": dataset,
        "storage_mode": mode,
        "bytes_written": bytes_written,
        "peak_rss_mb": round(post_rss, 1),
        "rss_delta_mb": round(post_rss - pre_rss, 1),
        "wall_time_ms": round(wall_ms, 1),
    }


_RESULT_SENTINEL = "<<SCENARIO_RESULT>>"
_SCENARIO_FNS = {
    "one_shot": _scenario_one_shot,
    "incremental": _scenario_incremental,
    "mutation_stream": _scenario_mutation_stream,
    "build_only": _scenario_build_only,  # helper, not an official scenario
}


def _run_scenario_in_subprocess(scenario: str, mode: str, dataset: str, env_extra: dict = None) -> dict:
    """Run one scenario in a fresh subprocess so RSS HWM isn't polluted by prior scenarios."""
    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--_run-scenario", scenario, mode, dataset],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"scenario subprocess failed (rc={proc.returncode}):\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_SENTINEL):
            return json.loads(line[len(_RESULT_SENTINEL) :])
    raise RuntimeError(f"scenario subprocess produced no result marker. stdout:\n{proc.stdout}")


def run_ingest_measurement(mode: str = "disk", dataset: str = "synth500k") -> None:
    """Run all three ingest scenarios in isolated subprocesses, stream rows to benchmarks/ingest_baseline.csv.

    Each scenario runs in its own Python process so ``ru_maxrss`` (a
    high-water mark, non-decreasing) reflects only that scenario's peak.
    Same-process runs previously let scenario-2's peak bleed into
    scenario-3's measurement.
    """
    print(f"\n{'=' * 70}")
    print(f"  INGEST MEASUREMENT — mode={mode} dataset={dataset}")
    print(f"  Output: {BASELINE_CSV}")
    print(f"{'=' * 70}\n")

    sha = _git_sha()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Pre-build the source graph that mutation_stream will reload from.
    # Runs in its own subprocess so the mutation_stream measurement
    # subprocess inherits a clean ru_maxrss baseline (the pre-build's
    # peak would otherwise pollute the measurement).
    prebuild_dest = str(Path(tempfile.gettempdir()) / f"ingest_mut_source_{dataset}_{mode}")
    _clean(prebuild_dest)
    print(f"  [prebuild] building mutation_stream source graph at {prebuild_dest}...", flush=True)
    _run_scenario_in_subprocess("build_only", mode, dataset, env_extra={"INGEST_PREBUILD_OUT": prebuild_dest})

    try:
        scenarios_to_run = ["one_shot", "incremental", "mutation_stream"]
        for name in scenarios_to_run:
            print(f"  [{name}] running in subprocess...", flush=True)
            env_extra = {"INGEST_PREBUILT_SOURCE": prebuild_dest} if name == "mutation_stream" else None
            try:
                result = _run_scenario_in_subprocess(name, mode, dataset, env_extra=env_extra)
                result["timestamp"] = ts
                result["commit_sha"] = sha
                result["n_nodes"] = SYNTH_NODES
                result["n_edges"] = SYNTH_EDGES
                result["n_chunks"] = SYNTH_CHUNKS
                _append_baseline_row(result)
                print(
                    f"    bytes={result['bytes_written']:>12,}  "
                    f"peak_rss={result['peak_rss_mb']:>6.1f} MB  "
                    f"d_rss={result['rss_delta_mb']:>+6.1f} MB  "
                    f"wall={result['wall_time_ms']:>8.0f} ms"
                )
            except BaseException as e:
                print(f"    FAILED: {e}")
                raise
    finally:
        _clean(prebuild_dest)

    print(f"\n  Baseline rows appended to {BASELINE_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global WIKIDATA_LANGUAGES

    print(f"KGLite v{kglite.__version__} API Benchmark")

    args = sys.argv[1:]

    # Pull --languages=<csv> out of argv before mode parsing. Subprocess
    # paths (--_run-scenario / --measure-ingest) don't see this flag, but
    # they inherit WIKIDATA_LANGUAGES via the parent process's mutation
    # only when running in-process; subprocesses re-import the module and
    # use the default. Forward via env var for those paths.
    cli_langs: list[str] | None = None
    rest: list[str] = []
    for a in args:
        if a.startswith("--languages="):
            cli_langs = [c.strip() for c in a.split("=", 1)[1].split(",") if c.strip()]
        else:
            rest.append(a)
    args = rest

    env_langs = os.environ.get("KGLITE_BENCH_LANGUAGES")
    if cli_langs is not None:
        WIKIDATA_LANGUAGES = cli_langs
    elif env_langs is not None:
        WIKIDATA_LANGUAGES = [c.strip() for c in env_langs.split(",") if c.strip()]
    print(f"  wikidata languages: {WIKIDATA_LANGUAGES or '(all)'}")
    # Propagate to subprocess scenarios (run_ingest_measurement spawns).
    os.environ["KGLITE_BENCH_LANGUAGES"] = ",".join(WIKIDATA_LANGUAGES)

    # Internal: run a single scenario and emit its result as a sentinel-prefixed
    # JSON line. Used by run_ingest_measurement() for subprocess isolation.
    if args and args[0] == "--_run-scenario":
        if len(args) != 4:
            print("usage: --_run-scenario <name> <mode> <dataset>", file=sys.stderr)
            sys.exit(2)
        _, scenario, smode, sdataset = args
        fn = _SCENARIO_FNS.get(scenario)
        if fn is None:
            print(f"unknown scenario: {scenario}", file=sys.stderr)
            sys.exit(2)
        result = fn(smode, sdataset)
        print(_RESULT_SENTINEL + json.dumps(result))
        return

    # --measure-ingest: PR0 ingest-measurement mode (disk-graph-improvement-plan)
    if "--measure-ingest" in args:
        rest = [a for a in args if a != "--measure-ingest"]
        mode = next((a for a in rest if a in ("memory", "mapped", "disk")), "disk")
        run_ingest_measurement(mode=mode)
        return

    valid_modes = ["memory", "mapped", "disk"]
    modes = [a for a in args if a in valid_modes] or valid_modes

    # Run disk last — its Rayon thread panics can SIGABRT the process.
    # Print comparison after non-disk modes so results aren't lost.
    safe_modes = [m for m in modes if m != "disk"]
    disk_modes = [m for m in modes if m == "disk"]

    all_results = {}
    for mode in safe_modes:
        all_results[mode] = run(mode)

    if len(all_results) > 1:
        compare(all_results)

    # Now attempt disk mode (may crash on some queries)
    for mode in disk_modes:
        all_results[mode] = run(mode)

    if disk_modes and len(all_results) > 1:
        compare(all_results)


if __name__ == "__main__":
    main()
