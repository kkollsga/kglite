"""Mapped-mode niche stress test — graphs that don't fit comfortably in RAM
but aren't Wikidata-scale either (roughly 1M–30M nodes).

This targets the user story for ``storage="mapped"``: you have a heterogeneous
knowledge graph with rich properties (think a few million entities with
descriptions, categories, and multi-hop relationships) that would take
~30 GB in-memory but can sit happily on an SSD while keeping RAM bounded.

The default --target-gb is modest so this runs cleanly on a dev laptop;
bump it to the real niche (20–30 GB) to exercise the intended use case.

Usage (standalone — not part of pytest auto-collection)::

    python tests/benchmarks/test_mapped_scale.py                       # ~1 GB default
    python tests/benchmarks/test_mapped_scale.py --target-gb 5         # 5 GB
    python tests/benchmarks/test_mapped_scale.py --target-gb 30        # the real niche
    python tests/benchmarks/test_mapped_scale.py --target-gb 2 --compare-memory

    # Pinned output directory (kept after completion):
    python tests/benchmarks/test_mapped_scale.py --target-gb 5 --dir /tmp/kg_mapped

Reports: build time, peak/steady RSS, on-disk size, query latencies on a set
of representative Cypher queries. With --compare-memory it also builds the
same shape in default in-memory mode at 1/10th the scale to sanity-check API
parity and show relative timings.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import kglite  # noqa: E402

from tests.conftest import rss_mb  # noqa: E402


def fmt_time(s):
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.2f} s"
    return f"{s / 60:.1f} min"


def fmt_mb(mb):
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


def section(title):
    print(f"\n{'─' * 68}\n  {title}\n{'─' * 68}")


def dir_size_mb(path):
    total = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)


# ── Synthetic graph parameters ───────────────────────────────────────────────

# Per-node budget in memory mode (measured empirically on a similar shape):
#   id(u64) + title(~40 char) + 3×int64 + category(~20 char) + description(~400 char)
#   plus HashMap overhead ≈ ~700 B / node
# Plus ~3 edges/node × ~100 B ≈ 300 B
# ⇒ ~1 KB per node → 1M nodes ≈ 1 GB memory-equivalent.
BYTES_PER_NODE_MEMORY_EST = 1024
EDGES_PER_NODE = 3
BATCH_SIZE = 50_000

# Node type mix — mimics a real knowledge graph where one type dominates
# (e.g. Documents or Events) with a few satellite types.
TYPE_MIX = [
    ("Entity", 0.60, ("main entity with description + category")),
    ("Document", 0.25, ("documents linked to entities")),
    ("Topic", 0.15, ("topical categories")),
]


def _make_entity_batch(start, count):
    """Entity: id + title + description + category + 3 numerics."""
    ids = list(range(start, start + count))
    return pd.DataFrame(
        {
            "eid": ids,
            "title": [f"Entity_{i}" for i in ids],
            # Description ~ 400 chars each, varying to avoid interner collapsing
            "description": [
                f"Entity {i} is a synthetic node used for mapped-mode stress testing. "
                f"Its numeric payload is {i * 3:.2f} and it belongs to cluster {i % 1000}. "
                f"Additional filler text to reach ~400 bytes per node: {'x' * 200}"
                for i in ids
            ],
            "category": [f"cat_{i % 100}" for i in ids],
            "score": [float(i % 10000) / 100.0 for i in ids],
            "rank": [i % 5000 for i in ids],
            "year": [2000 + (i % 25) for i in ids],
        }
    )


def _make_document_batch(start, count):
    ids = list(range(start, start + count))
    return pd.DataFrame(
        {
            "did": ids,
            "title": [f"Doc_{i}" for i in ids],
            "body": [f"Document body {i}. " + ("lorem ipsum " * 20) for i in ids],
            "pages": [(i % 500) + 1 for i in ids],
            "published": [2000 + (i % 25) for i in ids],
        }
    )


def _make_topic_batch(start, count):
    ids = list(range(start, start + count))
    return pd.DataFrame(
        {
            "tid": ids,
            "name": [f"Topic_{i}" for i in ids],
            "domain": [f"domain_{i % 20}" for i in ids],
            "popularity": [float(i % 1000) for i in ids],
        }
    )


def _make_edges(src_range, tgt_range, count, prop_offset=0):
    """Deterministic pseudo-random edges using linear-congruential mixing."""
    src_lo, src_hi = src_range
    tgt_lo, tgt_hi = tgt_range
    src_span = src_hi - src_lo
    tgt_span = tgt_hi - tgt_lo
    return pd.DataFrame(
        {
            "src": [src_lo + (i * 2654435761) % src_span for i in range(count)],
            "dst": [tgt_lo + ((i + prop_offset) * 40503) % tgt_span for i in range(count)],
            "weight": [float(i % 1000) / 10.0 for i in range(count)],
        }
    )


# ── Build ────────────────────────────────────────────────────────────────────


def _resolve_scale(target_gb):
    total_nodes = int(target_gb * 1024 * 1024 * 1024 / BYTES_PER_NODE_MEMORY_EST)
    # Keep at least 10k nodes per type so the mix stays heterogeneous at
    # small --target-gb; at bigger scales the fractions dominate.
    total_nodes = max(30_000, total_nodes)

    per_type = []
    assigned = 0
    for i, (name, frac, _desc) in enumerate(TYPE_MIX):
        if i == len(TYPE_MIX) - 1:
            count = total_nodes - assigned
        else:
            count = max(1, int(total_nodes * frac))
        per_type.append((name, count))
        assigned += count
    return total_nodes, per_type


def build_graph(target_gb, mode, path=None):
    """Build the synthetic graph. Returns (kg, stats)."""
    total_nodes, per_type = _resolve_scale(target_gb)
    section(
        f"Build: {total_nodes:,} nodes (~{fmt_mb(total_nodes * BYTES_PER_NODE_MEMORY_EST / 1024 / 1024)} memory-eq), "
        f"mode={mode}"
    )
    peak_rss = rss_mb()
    print(f"  RSS at start: {fmt_mb(peak_rss)}")

    kwargs = {}
    if mode == "mapped":
        kwargs["storage"] = "mapped"
    elif mode == "disk":
        if path is None:
            raise ValueError("disk mode needs path")
        kwargs["storage"] = "disk"
        kwargs["path"] = path
    kg = kglite.KnowledgeGraph(**kwargs)

    t_start = time.perf_counter()

    # --- Nodes per type ---
    type_offsets = {}
    running = 0
    for ntype, count in per_type:
        type_offsets[ntype] = running
        t0 = time.perf_counter()
        for batch_start in range(0, count, BATCH_SIZE):
            batch_n = min(BATCH_SIZE, count - batch_start)
            global_start = running + batch_start
            if ntype == "Entity":
                df = _make_entity_batch(global_start, batch_n)
                kg.add_nodes(df, "Entity", "eid", "title")
            elif ntype == "Document":
                df = _make_document_batch(global_start, batch_n)
                kg.add_nodes(df, "Document", "did", "title")
            elif ntype == "Topic":
                df = _make_topic_batch(global_start, batch_n)
                kg.add_nodes(df, "Topic", "tid", "name")
            peak_rss = max(peak_rss, rss_mb())
        elapsed = time.perf_counter() - t0
        print(f"  {ntype:10s}: {count:>10,} in {fmt_time(elapsed):>10s}  RSS={fmt_mb(rss_mb())}")
        running += count

    # --- Edges ---
    # Entity -> Document (mentions), Entity -> Topic (about),
    # Document -> Topic (classified), Entity -> Entity (related).
    entity_count = dict(per_type).get("Entity", 0)
    doc_count = dict(per_type).get("Document", 0)
    topic_count = dict(per_type).get("Topic", 0)

    entity_range = (type_offsets["Entity"], type_offsets["Entity"] + entity_count)
    doc_range = (type_offsets["Document"], type_offsets["Document"] + doc_count)
    topic_range = (type_offsets["Topic"], type_offsets["Topic"] + topic_count)

    edge_plan = [
        ("MENTIONS", "Entity", "Document", entity_range, doc_range, entity_count * 2, 0),
        ("ABOUT", "Entity", "Topic", entity_range, topic_range, entity_count, 1),
        ("CLASSIFIED", "Document", "Topic", doc_range, topic_range, doc_count, 2),
        ("RELATED", "Entity", "Entity", entity_range, entity_range, entity_count, 3),
    ]

    total_edges = 0
    for etype, src_type, tgt_type, src_range, tgt_range, n_edges, offset in edge_plan:
        # Skip edge types where either endpoint is empty — happens at very
        # small --target-gb when a type gets 0 nodes.
        if n_edges == 0 or src_range[1] <= src_range[0] or tgt_range[1] <= tgt_range[0]:
            continue
        t0 = time.perf_counter()
        for batch_start in range(0, n_edges, BATCH_SIZE):
            batch_n = min(BATCH_SIZE, n_edges - batch_start)
            df = _make_edges(src_range, tgt_range, batch_n, prop_offset=offset + batch_start)
            id_col_src = {"Entity": "eid", "Document": "did", "Topic": "tid"}[src_type]
            id_col_tgt = {"Entity": "eid", "Document": "did", "Topic": "tid"}[tgt_type]
            _ = (id_col_src, id_col_tgt)  # column names informational only
            kg.add_connections(df, etype, src_type, "src", tgt_type, "dst", columns=["weight"])
            peak_rss = max(peak_rss, rss_mb())
        elapsed = time.perf_counter() - t0
        total_edges += n_edges
        print(
            f"  {etype:10s}: {n_edges:>10,} {src_type}→{tgt_type} in {fmt_time(elapsed):>10s}  RSS={fmt_mb(rss_mb())}"
        )

    total_time = time.perf_counter() - t_start
    steady_rss = rss_mb()

    stats = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "build_time_s": total_time,
        "peak_rss_mb": peak_rss,
        "steady_rss_mb": steady_rss,
        "mode": mode,
    }
    print(
        f"\n  total: {total_nodes:,} nodes, {total_edges:,} edges "
        f"in {fmt_time(total_time)}; peak RSS={fmt_mb(peak_rss)}  steady={fmt_mb(steady_rss)}"
    )
    return kg, stats


# ── Queries ──────────────────────────────────────────────────────────────────


def run_queries(kg, label, total_nodes):
    section(f"Queries ({label})")
    queries = [
        ("count_entities", "MATCH (n:Entity) RETURN count(n) AS c"),
        ("filter_category", "MATCH (n:Entity) WHERE n.category = 'cat_42' RETURN count(n) AS c"),
        ("range_scan", "MATCH (n:Entity) WHERE n.rank < 100 RETURN n.title LIMIT 1000"),
        (
            "contains",
            "MATCH (n:Entity) WHERE n.description CONTAINS 'cluster 7' RETURN count(n) AS c",
        ),
        (
            "aggregation",
            "MATCH (n:Entity) RETURN n.year AS year, count(n) AS cnt, avg(n.score) AS mean_score",
        ),
        (
            "two_hop",
            "MATCH (a:Entity)-[:MENTIONS]->(d:Document)-[:CLASSIFIED]->(t:Topic) "
            "WHERE t.domain = 'domain_5' RETURN count(*) AS c",
        ),
        ("describe_overview", None),  # describe() is Python method, not Cypher
    ]

    timings = {}
    for name, query in queries:
        if query is None:
            # Use graph.describe() directly
            t0 = time.perf_counter()
            out = kg.describe()
            dt = time.perf_counter() - t0
            timings[name] = dt
            print(f"  {name:22s} {fmt_time(dt):>10s}  ({len(out):,} chars)")
        else:
            kg.cypher(query)  # warm
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                result = kg.cypher(query)
                # Materialise if iterable
                try:
                    _ = list(result)
                except TypeError:
                    _ = result
                times.append(time.perf_counter() - t0)
            best = min(times)
            timings[name] = best
            print(f"  {name:22s} {fmt_time(best):>10s}  RSS={fmt_mb(rss_mb())}")

    _ = total_nodes
    return timings


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-gb",
        type=float,
        default=1.0,
        help="Target graph size in GB (memory-mode equivalent). Default 1.0.",
    )
    parser.add_argument(
        "--mode",
        choices=("mapped", "memory", "disk"),
        default="mapped",
        help="Primary storage mode. Use 'mapped' for the real niche test.",
    )
    parser.add_argument(
        "--compare-memory",
        action="store_true",
        help="Also build in memory mode at 1/10th scale as a parity check.",
    )
    parser.add_argument("--dir", type=str, default=None, help="Working directory (default: tempdir).")
    parser.add_argument("--keep", action="store_true", help="Keep the working directory.")
    args = parser.parse_args()

    tmpdir = args.dir or tempfile.mkdtemp(prefix="kglite_mapped_scale_")
    os.makedirs(tmpdir, exist_ok=True)

    print(
        f"Target: ~{args.target_gb} GB (memory-mode equivalent), mode={args.mode}, "
        f"workdir={tmpdir}, start RSS={fmt_mb(rss_mb())}"
    )

    try:
        # --- Primary build in the requested mode ---
        disk_path = os.path.join(tmpdir, "kg_dir") if args.mode == "disk" else None
        if disk_path:
            os.makedirs(disk_path, exist_ok=True)
        kg, stats = build_graph(args.target_gb, args.mode, path=disk_path)

        # --- Save to .kgl (or report disk dir size for disk mode) ---
        if args.mode == "disk":
            disk_mb = dir_size_mb(disk_path)
            print(f"\n  On-disk footprint (graph dir): {fmt_mb(disk_mb)}")
            stats["disk_mb"] = disk_mb
        else:
            save_path = os.path.join(tmpdir, f"graph_{args.mode}.kgl")
            section("save()")
            t0 = time.perf_counter()
            kg.save(save_path)
            save_time = time.perf_counter() - t0
            disk_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"  save(): {fmt_time(save_time)}; file={fmt_mb(disk_mb)}")
            stats["save_time_s"] = save_time
            stats["disk_mb"] = disk_mb

        # --- Query battery ---
        query_timings = run_queries(kg, f"{args.mode} mode", stats["total_nodes"])
        stats["queries"] = query_timings

        del kg

        # --- Optional memory-mode comparison at 1/10th scale ---
        if args.compare_memory and args.mode != "memory":
            section(f"Parity check: memory mode at {args.target_gb / 10:.2f} GB")
            small_kg, small_stats = build_graph(args.target_gb / 10, "memory")
            small_timings = run_queries(small_kg, "memory mode (small)", small_stats["total_nodes"])
            stats["memory_compare"] = {"stats": small_stats, "queries": small_timings}
            del small_kg

        # --- Summary ---
        section("SUMMARY")
        print(f"  Mode:            {stats['mode']}")
        print(f"  Nodes / edges:   {stats['total_nodes']:,}  /  {stats['total_edges']:,}")
        print(f"  Build time:      {fmt_time(stats['build_time_s'])}")
        print(f"  Peak RSS build:  {fmt_mb(stats['peak_rss_mb'])}")
        print(f"  Steady RSS:      {fmt_mb(stats['steady_rss_mb'])}")
        print(f"  On-disk:         {fmt_mb(stats['disk_mb'])}")
        if "save_time_s" in stats:
            print(f"  Save time:       {fmt_time(stats['save_time_s'])}")
        print("\n  Query timings:")
        for name, t in stats["queries"].items():
            print(f"    {name:22s} {fmt_time(t):>10s}")

        if "memory_compare" in stats:
            mc = stats["memory_compare"]
            print(f"\n  Memory-mode parity check ({mc['stats']['total_nodes']:,} nodes):")
            for name, t in mc["queries"].items():
                ratio = t / stats["queries"][name] if stats["queries"][name] > 0 else 0
                print(f"    {name:22s} {fmt_time(t):>10s}  (vs {args.mode}: {ratio:.2f}×)")

    finally:
        if args.keep or args.dir:
            print(f"\n  Working dir kept: {tmpdir}")
        else:
            print(f"\n  Cleaning up {tmpdir}")
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
