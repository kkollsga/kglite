"""Build graph_1000.0 from latest-truthy.nt.bz2 with max_triples=1_000_000_000.

Matches the `wiki1000m` slice in wiki_benchmark.py — 1B triples scanned.
Roughly 2× the size of graph_500.0 (which used 509M triples, 6M entities).

Run from repo root:
    python bench/rebuild_graph_1000.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

import kglite

DUMP = "/Volumes/EksternalHome/Data/Wikidata/latest-truthy.nt.bz2"
GRAPH_DIR = "/Volumes/EksternalHome/Data/Wikidata/graph_1000.0"
TARGET_TRIPLES = 1_000_000_000

os.environ.setdefault("KGLITE_CSR_ALGO", "merge_sort")
os.environ.setdefault("KGLITE_CSR_VERBOSE", "1")


def main() -> int:
    if not Path(DUMP).exists():
        print(f"error: dump not found at {DUMP}", file=sys.stderr)
        return 2

    if Path(GRAPH_DIR).exists():
        print(f"removing existing {GRAPH_DIR}…")
        shutil.rmtree(GRAPH_DIR)

    print("=" * 70)
    print(f"BUILDING {GRAPH_DIR}")
    print(f"  dump:           {DUMP}  ({Path(DUMP).stat().st_size / 1024**3:.1f} GB)")
    print(f"  max_triples:    {TARGET_TRIPLES:,}")
    print("=" * 70)
    print()

    t0 = time.perf_counter()
    g = kglite.KnowledgeGraph(storage="disk", path=GRAPH_DIR)
    stats = g.load_ntriples(
        DUMP,
        languages=["en"],
        max_triples=TARGET_TRIPLES,
        verbose=True,
    )
    t_parse = time.perf_counter() - t0
    print()
    print(f"parse done in {t_parse:.1f}s ({t_parse / 60:.1f} min)")
    print(f"  triples_scanned: {stats.get('triples_scanned', 0):,}")
    print(f"  entities:        {stats.get('entities', 0):,}")
    print(f"  edges:           {stats.get('edges', 0):,}")
    print()

    if hasattr(g, "rebuild_caches"):
        t0 = time.perf_counter()
        g.rebuild_caches()
        print(f"rebuild_caches in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    g.save(GRAPH_DIR)
    print(f"save in {time.perf_counter() - t0:.1f}s")

    info = g.graph_info()
    total_bytes = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(GRAPH_DIR)
        for f in files
    )
    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  nodes:  {info['node_count']:,}")
    print(f"  edges:  {info['edge_count']:,}")
    print(f"  disk:   {total_bytes / 1024**3:.2f} GB")

    for f in ("id_indices.bin", "type_indices.bin"):
        p = Path(GRAPH_DIR) / f
        if p.exists():
            print(f"  {f}: {p.stat().st_size / 1024**2:.1f} MB")

    del g
    print("\nreloading + smoke-testing…")
    t0 = time.perf_counter()
    g = kglite.load(GRAPH_DIR)
    print(f"  reload in {(time.perf_counter() - t0) * 1000:.0f}ms")
    rows = list(g.cypher("MATCH ()-[r]->() RETURN count(r) AS c"))
    expected = info["edge_count"]
    actual = rows[0]["c"] if rows else 0
    if actual == expected:
        print(f"  count(r) = {actual:,} (matches edge_count) ✓")
        return 0
    print(f"  FAIL: count(r) = {actual:,} but edge_count = {expected:,}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
