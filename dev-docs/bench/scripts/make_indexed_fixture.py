#!/usr/bin/env python3
"""Build the load-memory program's *indexed* baseline fixture — and its
index-free twin.

WHY. The dominant modeled term in a `.kgl` load is the eager rebuild of user
indexes (`storage/columns.rs` -> `dir_graph/mod.rs::rebuild_type_indices_and_schemas`),
which scales with `row_count x index keys`, NOT with file bytes. Neither
`sodir_graph.kgl` nor `kglite_codebase.kgl` carries user indexes, so neither can
show that term at all. This script produces a graph that does, plus a
byte-for-byte identical dataset saved with **no** indexes, so the difference
between the two loads *is* the index-rebuild cost — the direct measurement of
the P2 lever.

WHAT IT BUILDS (both variants, same nodes/edges/properties):

    dev-docs/bench/out/indexed_500k.kgl    unique constraint on Item.sku (string)
                                           + property index Item.category
                                           + property index Item.region
                                           + composite index Item.(category, region)
    dev-docs/bench/out/noindex_500k.kgl    the same data, zero indexes/constraints

Run it with the RELEASE wheel installed (`uv run --no-sync maturin develop
--release`) — this is a measurement fixture, and CLAUDE.md's debug-profile rule
is a *correctness*-testing rule, not a build rule for perf artifacts.

Re-runnable: it overwrites both outputs and writes nothing outside
`dev-docs/bench/out/` (that tier is auto-purged after 14 days, so regenerating
is the expected recovery, not a fallback).

    .venv/bin/python dev-docs/bench/scripts/make_indexed_fixture.py \
        [--nodes 500000] [--edges-per-node 1] [--out-dir DIR] [--seed 20260829]

Use the repo venv's interpreter explicitly — a bare `./` would run whichever
python is on PATH, which is not the one the release wheel was installed into.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "dev-docs" / "bench" / "out"

NODE_TYPE = "Item"
# The node type's identity column. A unique constraint cannot go here: the
# engine refuses `REQUIRE n.<id-field> IS UNIQUE` because that name resolves to
# the structural id rather than a stored property, so the secondary index would
# never see writes. Identity uniqueness is a primary-key declaration instead.
UID_FIELD = "uid"
# A second high-cardinality string property, stored like any other — this is
# what carries the unique *index* the fixture exists to measure.
SKU_FIELD = "sku"
CATEGORIES = [f"cat_{i:03d}" for i in range(200)]
REGIONS = [f"region_{i:02d}" for i in range(40)]


def build_frame(n: int, seed: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    return pd.DataFrame(
        {
            UID_FIELD: [f"item-{i:09d}" for i in idx],
            # Distinct per row: the unique index over it must hash and store
            # one entry per node, which is the whole point of the fixture.
            SKU_FIELD: [f"SKU-{i:09d}" for i in idx],
            "title": [f"Item {i}" for i in idx],
            "category": rng.choice(CATEGORIES, size=n),
            "region": rng.choice(REGIONS, size=n),
            "score": rng.random(n) * 100.0,
            "count": rng.integers(0, 10_000, size=n),
        }
    )


def build_graph(kglite, frame, edges_per_node: int, seed: int):
    import numpy as np
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(frame, node_type=NODE_TYPE, unique_id_field=UID_FIELD,
                node_title_field="title")

    if edges_per_node > 0:
        rng = np.random.default_rng(seed + 1)
        n = len(frame)
        src = np.repeat(np.arange(n, dtype=np.int64), edges_per_node)
        dst = rng.integers(0, n, size=n * edges_per_node)
        edges = pd.DataFrame(
            {
                "src": [f"item-{i:09d}" for i in src],
                "dst": [f"item-{i:09d}" for i in dst],
            }
        )
        g.add_connections(
            edges,
            connection_type="LINKS_TO",
            source_type=NODE_TYPE,
            source_id_field="src",
            target_type=NODE_TYPE,
            target_id_field="dst",
        )
    return g


def add_indexes(g) -> None:
    """The four index structures the fixture exists to exercise.

    A unique constraint, two single-property equality indexes, and one
    composite — deliberately different structures, because each is rebuilt by a
    different branch at load and they do not cost the same per row.
    """
    g.cypher(f"CREATE CONSTRAINT item_sku FOR (n:{NODE_TYPE}) REQUIRE n.{SKU_FIELD} IS UNIQUE")
    g.cypher(f"CREATE INDEX FOR (n:{NODE_TYPE}) ON (n.category)")
    g.cypher(f"CREATE INDEX FOR (n:{NODE_TYPE}) ON (n.region)")
    g.cypher(f"CREATE INDEX FOR (n:{NODE_TYPE}) ON (n.category, n.region)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=500_000)
    ap.add_argument("--edges-per-node", type=int, default=1)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    import kglite

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    indexed = out_dir / "indexed_500k.kgl"
    noindex = out_dir / "noindex_500k.kgl"

    print(f"kglite {kglite.__version__} -> {out_dir}")
    t0 = time.perf_counter()
    frame = build_frame(args.nodes, args.seed)
    print(f"  frame          {len(frame):>9,} rows   {time.perf_counter() - t0:6.1f}s")

    # Two independent builds rather than one graph saved twice: dropping
    # indexes off a graph that had them can leave residue the no-index variant
    # must not carry, and the point of the pair is that the *only* difference
    # is index presence.
    for path, want_indexes in ((noindex, False), (indexed, True)):
        t = time.perf_counter()
        g = build_graph(kglite, frame, args.edges_per_node, args.seed)
        if want_indexes:
            add_indexes(g)
        g.save(str(path))
        size = path.stat().st_size
        print(f"  {path.name:<20} indexes={str(want_indexes):<5} "
              f"{size / 1048576:8.1f} MB  {time.perf_counter() - t:6.1f}s")
        del g

    return 0


if __name__ == "__main__":
    sys.exit(main())
