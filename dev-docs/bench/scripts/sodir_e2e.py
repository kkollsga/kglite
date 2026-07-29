"""End-to-end Sodir benchmark.

Times the full lifecycle: fetch (or cache-hit) → build → optional
queries. Wraps `kglite.datasets.sodir.open()`.

Examples
--------
First run, in-memory build (downloads CSVs, builds, runs queries):

    python bench/sodir_e2e.py --workdir /Volumes/EksternalHome/Data/Sodir

Force a disk build (persistent + cached):

    python bench/sodir_e2e.py --workdir /Volumes/EksternalHome/Data/Sodir --storage disk

Build with a complement blueprint that adds your sideloaded extras:

    python bench/sodir_e2e.py --workdir /Volumes/EksternalHome/Data/Sodir \
        --complement /path/to/sodir_extras.json

Skip the query suite (just measure fetch + build):

    python bench/sodir_e2e.py --workdir /Volumes/EksternalHome/Data/Sodir --no-queries
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from kglite.datasets import sodir


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _run_table(title, cases):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"\n  {'Query':50s} {'Time':>9s}  {'Result':>14s}")
    print("  " + "-" * 78)
    total = 0.0
    for label, fn in cases:
        t0 = time.perf_counter()
        try:
            result = fn()
            rendered = f"{result:,}" if isinstance(result, int) else str(result)[:30]
        except Exception as e:
            rendered = f"ERR: {e}"[:30]
        elapsed = time.perf_counter() - t0
        total += elapsed
        print(f"  {label:50s} {elapsed:>8.3f}s  {rendered:>14s}")
    print(f"\n  Total: {total:.2f}s across {len(cases)} queries")


def _run_cypher(g):
    """Petroleum-domain Cypher coverage: counts, typed traversals,
    aggregates, parameter binding."""

    def q(cql, params=None):
        return len(g.cypher(cql, params=params).to_df())

    def first(cql, params=None):
        df = g.cypher(cql, params=params).to_df()
        return df.iloc[0, 0] if len(df) else None

    cases = [
        # ── Counts by type ───────────────────────────────────────────────────
        ("count all nodes", lambda: first("MATCH (n) RETURN count(n) AS c")),
        ("count fields", lambda: first("MATCH (n:Field) RETURN count(n) AS c")),
        ("count wellbores", lambda: first("MATCH (n:Wellbore) RETURN count(n) AS c")),
        ("count discoveries", lambda: first("MATCH (n:Discovery) RETURN count(n) AS c")),
        ("count companies", lambda: first("MATCH (n:Company) RETURN count(n) AS c")),
        ("count licences", lambda: first("MATCH (n:Licence) RETURN count(n) AS c")),
        # ── Property scans ───────────────────────────────────────────────────
        (
            "active fields",
            lambda: first("MATCH (n:Field) WHERE n.fldCurrentActivitySatus = 'Producing' RETURN count(n)"),
        ),
        (
            "discoveries with HC type",
            lambda: q("MATCH (n:Discovery) WHERE n.dscHcType IS NOT NULL RETURN n.title LIMIT 20"),
        ),
        (
            "wellbores 2020+",
            lambda: q("MATCH (n:Wellbore) WHERE n.wlbEntryYear >= 2020 RETURN n.title LIMIT 50"),
        ),
        # ── 1-hop typed ──────────────────────────────────────────────────────
        (
            "field → operator company",
            lambda: q("MATCH (f:Field)-[:HAS_OPERATOR]->(c:Company) RETURN f.title, c.title LIMIT 20"),
        ),
        (
            "discovery → field (incl-hst)",
            lambda: q("MATCH (f:Field)-[:INCLUDES_DISCOVERY]->(d:Discovery) RETURN f.title, d.title LIMIT 20"),
        ),
        # ── Parameter binding ────────────────────────────────────────────────
        (
            "params: field by name",
            lambda: q(
                "MATCH (n:Field) WHERE n.fldName = $name RETURN n.title",
                params={"name": "EKOFISK"},
            ),
        ),
        # ── Aggregations ─────────────────────────────────────────────────────
        (
            "fields by main area",
            lambda: q("MATCH (n:Field) RETURN n.fldMainArea AS area, count(n) AS c ORDER BY c DESC LIMIT 10"),
        ),
        (
            "discoveries by year",
            lambda: q(
                "MATCH (n:Discovery) WHERE n.dscDiscoveryYear IS NOT NULL "
                "RETURN n.dscDiscoveryYear AS year, count(n) AS c "
                "ORDER BY year DESC LIMIT 10"
            ),
        ),
    ]
    _run_table("CYPHER QUERIES", cases)


def _run_fluent(g):
    """Fluent select / traverse / where coverage."""
    types = g.node_type_counts() or {}

    def have(t):
        return t in types

    def safe(op):
        def run():
            try:
                return op()
            except Exception as e:
                return f"skipped: {e}"[:30]

        return run

    cases = [
        ("select('Field').len()", safe(lambda: g.select("Field").len())),
        ("select('Wellbore').len()", safe(lambda: g.select("Wellbore").len())),
        ("select('Discovery').len()", safe(lambda: g.select("Discovery").len())),
        (
            "select('Field').where(area=North sea, limit=20)",
            safe(lambda: g.select("Field").where({"fldMainArea": "North sea"}, limit=20).len()),
        ),
        (
            "select('Wellbore').where(year>=2020, limit=50)",
            safe(lambda: g.select("Wellbore").where({"wlbEntryYear": {">=": 2020}}, limit=50).len()),
        ),
        (
            "select('Field').traverse(HAS_OPERATOR)",
            safe(lambda: g.select("Field").traverse("HAS_OPERATOR").len() if have("Field") else "no Field"),
        ),
        (
            "select('Field').where_connected(INCLUDES_DISCOVERY)",
            safe(lambda: g.select("Field").where_connected("INCLUDES_DISCOVERY").len()),
        ),
        ("len(graph.describe())", lambda: len(g.describe())),
        ("len(graph_info)", lambda: len(g.graph_info())),
    ]
    _run_table("FLUENT API QUERIES", cases)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, help="Sodir workdir (cached CSVs + index + optional graph)")
    ap.add_argument(
        "--storage",
        choices=["memory", "disk"],
        default="memory",
        help="memory (default — Sodir is small) or disk (persistent + cached)",
    )
    ap.add_argument(
        "--index-cooldown",
        type=int,
        default=14,
        help="Days between cheap row-count probes (default 14)",
    )
    ap.add_argument(
        "--dataset-cooldown",
        type=int,
        default=30,
        help="Days before forcing full per-dataset re-fetch (default 30)",
    )
    ap.add_argument(
        "--blueprint",
        default=None,
        help="Path to a custom blueprint that fully replaces the packaged one",
    )
    ap.add_argument(
        "--complement",
        default=None,
        help="Path to a complementary blueprint that's persisted into the workdir",
    )
    ap.add_argument(
        "--complement-overrides",
        action="store_true",
        help="Let the complement override base on key collisions (default: base wins)",
    )
    ap.add_argument(
        "--no-complement",
        action="store_true",
        help="Skip any saved complement blueprint for this call (file untouched)",
    )
    ap.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force graph rebuild even if a cached disk graph exists (memory mode always rebuilds)",
    )
    ap.add_argument("--no-queries", action="store_true", help="Skip the cypher + fluent query suites")
    ap.add_argument("--quiet", action="store_true", help="Suppress phase output from fetch + build")
    args = ap.parse_args()

    print("=" * 70)
    print("SODIR E2E BENCHMARK")
    print(f"  workdir:           {args.workdir}")
    print(f"  storage:           {args.storage}")
    print(f"  index_cooldown:    {args.index_cooldown}d")
    print(f"  dataset_cooldown:  {args.dataset_cooldown}d")
    if args.blueprint:
        print(f"  blueprint:         {args.blueprint}")
    if args.complement:
        print(f"  complement:        {args.complement}")
    if args.complement_overrides:
        print("  complement_overrides: True")
    if args.no_complement:
        print("  use_complement:    False")
    if args.force_rebuild:
        print("  force_rebuild:     True")
    print("=" * 70)
    print()

    wall_start = time.time()
    open_start = time.time()
    g = sodir.open(
        args.workdir,
        storage=args.storage,
        index_cooldown_days=args.index_cooldown,
        dataset_cooldown_days=args.dataset_cooldown,
        blueprint_path=args.blueprint,
        complement_blueprint=args.complement,
        use_complement=not args.no_complement,
        complement_overrides=args.complement_overrides,
        force_rebuild=args.force_rebuild,
        verbose=not args.quiet,
    )
    open_elapsed = time.time() - open_start

    info = g.graph_info()
    print()
    print(f"  Graph ready in {_fmt_dur(open_elapsed)}")
    print(f"    nodes: {info.get('node_count', 0):,}")
    print(f"    edges: {info.get('edge_count', 0):,}")

    queries_elapsed = 0.0
    if not args.no_queries:
        q_start = time.time()
        _run_cypher(g)
        _run_fluent(g)
        queries_elapsed = time.time() - q_start

    total = time.time() - wall_start
    print()
    print("=" * 70)
    print("TIMING SUMMARY")
    print(f"  fetch + build:  {_fmt_dur(open_elapsed):>10}")
    if not args.no_queries:
        print(f"  queries:        {_fmt_dur(queries_elapsed):>10}")
    print(f"  total wall:     {_fmt_dur(total):>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
