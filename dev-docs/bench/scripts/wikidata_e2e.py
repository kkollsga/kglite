"""End-to-end Wikidata bz2 benchmark.

Loads ``latest-truthy.nt.bz2`` directly through the parallel-bz2
decoder, capping at ``--size <N>`` million triples. Times the full
pipeline (decompress + parse + columnar build + edges + CSR + save)
and emits tqdm progress bars per phase.

Examples
--------

    # 5M-triple slice of the live bz2 dump, disk mode
    python bench/wikidata_e2e.py --workdir /Volumes/EksternalHome/Data/Wikidata --size 5

    # 50M triples, memory mode, skip queries
    python bench/wikidata_e2e.py --workdir ... --size 50 --storage memory --no-queries

    # Force a clean rebuild (wipes <workdir>/graph_<size>/ first)
    python bench/wikidata_e2e.py --workdir ... --size 5 --force-rebuild
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from kglite import KnowledgeGraph

DUMP_FILENAME = "latest-truthy.nt.bz2"


def _start_watchdog(seconds: float) -> threading.Thread:
    """Start a daemon thread that SIGINTs the process after ``seconds``.

    A clean SIGINT routes through the loader's progress sink → Rust
    `check_signals` → `KeyboardInterrupt`, so the build cancels at its
    next safe point (every 5M triples on Phase 1, every 1M edges on
    Phase 2). If the build is genuinely wedged before the first
    cancel-check fires (e.g. during the bz2 stream-boundary scan), we
    follow up with a hard SIGTERM after a 10s grace window.
    """

    def _bark() -> None:
        time.sleep(seconds)
        sys.stderr.write(f"\n[watchdog] timeout after {seconds}s — sending SIGINT\n")
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(10)
        sys.stderr.write("[watchdog] still alive after grace — sending SIGTERM\n")
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGTERM)

    t = threading.Thread(target=_bark, daemon=True, name="wikidata-e2e-watchdog")
    t.start()
    return t


def _fmt_dur(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--workdir",
        required=True,
        help=f"directory holding {DUMP_FILENAME} (built graphs go in <workdir>/graph_<size>/)",
    )
    ap.add_argument(
        "--size",
        type=float,
        default=None,
        metavar="N",
        help=(
            "Cap the build at N million triples scanned (no cap if omitted). "
            "Wikidata is ~1.25%% entities, so e.g. --size 5 gives ~62k entities."
        ),
    )
    ap.add_argument(
        "--storage",
        choices=["disk", "memory"],
        default="disk",
        help="disk = persisted graph dir, memory = in-RAM only (default disk)",
    )
    ap.add_argument(
        "--languages",
        default="en",
        metavar="CODES",
        help=(
            "Comma-separated language codes for label/description filtering "
            "(default: en). Pass an empty string to keep all languages — "
            "warning: that yields non-deterministic node-type names because "
            "the auto_type rename picks whichever label was seen first."
        ),
    )
    ap.add_argument(
        "--no-queries",
        action="store_true",
        help="Skip the cypher + fluent query suites",
    )
    ap.add_argument(
        "--legacy-progress",
        action="store_true",
        help="Use the old [Phase X] stderr lines instead of tqdm bars (default: tqdm)",
    )
    ap.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Wipe the existing graph dir at <workdir>/graph_<size>/ before building",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Kill the build after SEC seconds (sends SIGINT, then SIGTERM "
            "after a 10s grace). Useful for test sweeps that shouldn't "
            "stall on a wedged run. Default: no timeout."
        ),
    )
    args = ap.parse_args()

    if args.timeout is not None:
        _start_watchdog(args.timeout)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dump_path = workdir / DUMP_FILENAME
    if not dump_path.exists():
        sys.exit(f"error: {dump_path} not found")

    max_triples = int(args.size * 1_000_000) if args.size is not None else None
    size_label = f"{args.size}M triples" if args.size is not None else "full"
    graph_subdir = f"graph_{args.size}" if args.size is not None else "graph"

    print("=" * 70)
    print("WIKIDATA E2E BENCHMARK (bz2 → graph)")
    print(f"  workdir:  {workdir}")
    print(f"  source:   {dump_path.name}")
    print(f"  size:     {size_label}")
    print(f"  storage:  {args.storage}")
    print("=" * 70)
    print()

    progress_cb = None
    if not args.legacy_progress:
        from kglite.progress import TqdmBuildProgress

        progress_cb = TqdmBuildProgress()

    graph_dir = workdir / graph_subdir
    if args.storage == "disk":
        if args.force_rebuild and graph_dir.exists():
            print(f"  force_rebuild=True — deleting {graph_dir}")
            shutil.rmtree(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        g = KnowledgeGraph(storage="disk", path=str(graph_dir))
    else:
        g = KnowledgeGraph()

    # Build temp files (~40 GB property log, edge buffer overflow, label
    # spill) default to the system temp dir, which lands on the boot SSD
    # — not enough room for a full-Wikidata build. Pin the spill dir
    # under workdir so all heavy I/O stays on the data volume.
    spill_dir = workdir / "build_temp"
    spill_dir.mkdir(parents=True, exist_ok=True)
    g.set_memory_limit(None, str(spill_dir))

    # Language filter — defaults to ["en"]. Without it, the auto_type
    # rename picks the first label seen per Q-code (non-deterministic
    # across runs, often non-English), which corrupts type names. Pass
    # `--languages ""` to disable and keep all languages.
    languages = [code.strip() for code in args.languages.split(",") if code.strip()]

    wall_start = time.time()
    build_start = time.time()
    load_kwargs: dict = {"verbose": args.legacy_progress}
    if languages:
        load_kwargs["languages"] = languages
    if max_triples is not None:
        load_kwargs["max_triples"] = max_triples
    if progress_cb is not None:
        load_kwargs["progress"] = progress_cb
    g.load_ntriples(str(dump_path), **load_kwargs)
    build_elapsed = time.time() - build_start

    if args.storage == "disk":
        save_start = time.time()
        g.save(str(graph_dir))
        save_elapsed = time.time() - save_start
    else:
        save_elapsed = 0.0

    info = g.graph_info()
    print()
    print("=" * 70)
    print(f"GRAPH COMPLETE in {_fmt_dur(build_elapsed)}")
    print(f"  nodes: {info.get('node_count', 0):,}")
    print(f"  edges: {info.get('edge_count', 0):,}")
    if args.storage == "disk":
        print(f"  save:  {_fmt_dur(save_elapsed)}")
    print("=" * 70)

    # SANITY PROBE — quick + always run, even with --no-queries. Catches
    # regressions in type rename / language filter / property storage on
    # the next build immediately rather than at query time.
    print()
    print("=" * 70)
    print("SANITY PROBE")
    try:
        sample = g.cypher(
            "MATCH (n {nid: 'Q42'}) RETURN n.title, n.description LIMIT 1",
            timeout_ms=5_000,
        )
        if sample:
            row = sample[0]
            print(f"  Q42 title:        {row.get('n.title')!r}")
            print(f"  Q42 description:  {row.get('n.description')!r}")
        else:
            print("  Q42:              (not found — language filter or rename issue?)")
    except Exception as e:
        print(f"  Q42 probe failed: {e}")
    try:
        # Use the cached type-count map instead of a full-scan Cypher
        # aggregation — at 124M nodes the scan times out at 10s, and
        # graph_info already maintains this counter authoritatively.
        type_counts = g.node_type_counts()
        top = sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("  Top 5 node types:")
        for t, k in top:
            print(f"    {t!r:<35s} {k:>12,}")
    except Exception as e:
        print(f"  Type probe failed: {e}")
    print("=" * 70)

    queries_elapsed = 0.0
    if not args.no_queries:
        # Reuse the curated query suite from the example so we benchmark
        # the same surface users see. Imported lazily so the script still
        # runs with --no-queries when examples/wikidata_disk.py is absent.
        sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
        from wikidata_disk import run_cypher, run_fluent  # noqa: E402

        q_start = time.time()
        run_cypher(g)
        run_fluent(g)
        queries_elapsed = time.time() - q_start

    total = time.time() - wall_start
    print()
    print("=" * 70)
    print("TIMING SUMMARY")
    print(f"  build:          {_fmt_dur(build_elapsed):>10}")
    if args.storage == "disk":
        print(f"  save:           {_fmt_dur(save_elapsed):>10}")
    if not args.no_queries:
        print(f"  queries:        {_fmt_dur(queries_elapsed):>10}")
    print(f"  total wall:     {_fmt_dur(total):>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
