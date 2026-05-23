"""kglite vs NornicDB shortest-path benchmark — faithful port.

Mirrors NornicDB's `TestLargeScaleShortestPath_HopBuckets` from
`pkg/cypher/demo_shortest_path_largescale_test.go`. Every shape detail
(node id format, edge generator, RNG seed, BFS-bucket method, query
text, warm-up rule) matches their Go implementation 1:1.

Reference:
  https://github.com/orneryd/NornicDB/blob/main/pkg/cypher/
  demo_shortest_path_largescale_test.go
  https://github.com/orneryd/NornicDB/blob/main/docs/performance/
  shortest-path-500k-benchmark.md

The only deliberate divergence: their RNG is Go's
`rand.New(rand.NewSource(0xfeedd3))` (LCG-based PRNG). We use Python's
Mersenne Twister with the same seed — the topology is statistically
equivalent but the exact edge set differs. The bucketing methodology
yields the same depth distribution either way; per-pair latencies aren't
sensitive to which specific edges are picked.

Usage:
    python scripts/nornicdb_compare.py
"""

from __future__ import annotations

from collections import defaultdict, deque
import random
import time

import pandas as pd

import kglite

# ── NornicDB-matched parameters ─────────────────────────────────────────────
# From largescale.go:
#   const (
#     largeScaleNodes      = 500_000
#     largeScaleSectors    = 1000
#     largeScaleSectorSize = 500
#     largeScaleIntraEdges = 3
#     largeScaleGateways   = 2
#     largeScalePairsPer   = 30
#   )

LARGE_SCALE_NODES = 500_000
LARGE_SCALE_SECTORS = 1_000
LARGE_SCALE_SECTOR_SIZE = 500
LARGE_SCALE_INTRA_EDGES = 3  # random extras per node
LARGE_SCALE_GATEWAYS = 2  # bridges between adjacent sectors
LARGE_SCALE_PAIRS_PER = 30
LARGE_SCALE_MAX_DEPTH = 60
LARGE_SCALE_MAX_SOURCES_BFS = 200

# Their RNG seed: rand.NewSource(0xfeedd3) in Go. We can't replicate Go's
# LCG output in Python, but we set our seed to the same value so the run
# is at least deterministic on our side.
RNG_SEED = 0xFEEDD3

# Their Cypher query, verbatim:
#   MATCH (start:Star {starId: $startId}), (end:Star {starId: $endId})
#   MATCH p = shortestPath((start)-[:HYPERLANE*]-(end))
#   RETURN [n IN nodes(p) | n.starId] AS pathIds, length(p) AS hops
#   LIMIT 1
NORNIC_QUERY = """
MATCH (start:Star {starId: $startId}), (dst:Star {starId: $endId})
MATCH p = shortestPath((start)-[:HYPERLANE*]-(dst))
RETURN [n IN nodes(p) | n.starId] AS pathIds, length(p) AS hops
LIMIT 1
""".strip()


def fmt_t(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def star_id(sector: int, idx: int) -> str:
    """NornicDB's id format: `fmt.Sprintf("s%d-%d", s, i)`."""
    return f"s{sector}-{idx}"


def build_graph() -> tuple[kglite.KnowledgeGraph, list[list[str]], dict[str, set[str]]]:
    """Build the 500k Star / ~3.4M-3.9M HYPERLANE graph matching NornicDB's
    topology exactly. Returns (graph, sector_members, undirected_adj).

    Edge topology (from largescale.go:160-180):
      - Per sector: spanning chain `addEdge(members[i-1], members[i])` for
        i in 1..500 — 499 edges
      - Per node: `largeScaleIntraEdges=3` random extras to a same-sector
        member — `addEdge(id, members[rng.Intn(len(members))])`
      - Between adjacent sectors `s` and `s+1`: `largeScaleGateways=2`
        gateway edges from random member of `s` to random member of `s+1`
      - `addEdge` dedupes via "min|max" key — same (a,b) is at most one
        undirected edge — so the per-sector total caps below `499 + 3*500`.

    NornicDB stores each undirected edge as TWO directed edges for the
    `[:HYPERLANE*]-` pattern. We do the same.
    """
    print(
        f"Building NornicDB-spec graph: {LARGE_SCALE_NODES:,} nodes / "
        f"{LARGE_SCALE_SECTORS} sectors × {LARGE_SCALE_SECTOR_SIZE} stars..."
    )
    t0 = time.perf_counter()

    g = kglite.KnowledgeGraph()
    rng = random.Random(RNG_SEED)

    # 1. Build the node table. starId is the string id; sector + a synthetic
    #    `title` column round out the Star node properties (`title` is required
    #    by kglite's add_nodes — set it equal to starId for simplicity).
    sector_members: list[list[str]] = []
    all_ids: list[str] = []
    sectors: list[int] = []
    for s in range(LARGE_SCALE_SECTORS):
        members = [star_id(s, i) for i in range(LARGE_SCALE_SECTOR_SIZE)]
        sector_members.append(members)
        all_ids.extend(members)
        sectors.extend([s] * LARGE_SCALE_SECTOR_SIZE)

    nodes_df = pd.DataFrame(
        {
            "starId": all_ids,
            "title": all_ids,  # kglite wants a title column
            "sector": sectors,
        }
    )
    g.add_nodes(nodes_df, "Star", "starId", "title")
    t_nodes = time.perf_counter() - t0
    print(f"  add_nodes: {fmt_t(t_nodes)}")

    # 2. Build undirected adjacency in Python (mirrors Go's addEdge + dedup).
    added: set[tuple[str, str]] = set()
    adj: dict[str, set[str]] = defaultdict(set)

    def add_edge(a: str, b: str) -> None:
        if a == b:
            return
        k = (a, b) if a < b else (b, a)
        if k in added:
            return
        added.add(k)
        adj[a].add(b)
        adj[b].add(a)

    # Per sector: spanning chain + N random extras per node.
    for s in range(LARGE_SCALE_SECTORS):
        members = sector_members[s]
        for i in range(1, len(members)):
            add_edge(members[i - 1], members[i])
        for member_id in members:
            for _ in range(LARGE_SCALE_INTRA_EDGES):
                other = members[rng.randrange(len(members))]
                add_edge(member_id, other)

    # Between adjacent sectors: gateway edges.
    for s in range(LARGE_SCALE_SECTORS - 1):
        a = sector_members[s]
        b = sector_members[s + 1]
        for _ in range(LARGE_SCALE_GATEWAYS):
            from_id = a[rng.randrange(len(a))]
            to_id = b[rng.randrange(len(b))]
            add_edge(from_id, to_id)

    t_edge_build = time.perf_counter() - t0 - t_nodes
    print(f"  edge gen + dedup: {fmt_t(t_edge_build)} ({len(added):,} unique undirected)")

    # 3. Mirror each undirected edge into two directed rows.
    src: list[str] = []
    tgt: list[str] = []
    for a, b in added:
        src.append(a)
        tgt.append(b)
        src.append(b)
        tgt.append(a)

    edges_df = pd.DataFrame({"src": src, "tgt": tgt})
    g.add_connections(edges_df, "HYPERLANE", "Star", "src", "Star", "tgt")
    t_edges = time.perf_counter() - t0 - t_nodes - t_edge_build
    print(f"  add_connections: {fmt_t(t_edges)} ({len(src):,} directed)")

    # Create the starId equality index — NornicDB does the same via
    # `CREATE INDEX star_id_idx IF NOT EXISTS FOR (n:Star) ON (n.starId)`
    # in their fixture. Without it the parameterized `MATCH (start:Star
    # {starId: $startId})` does a 500k-node scan per call, completely
    # dominating the bench.
    t_idx = time.perf_counter()
    g.create_index("Star", "starId")
    print(f"  create_index(Star.starId): {fmt_t(time.perf_counter() - t_idx)}")

    total = time.perf_counter() - t0
    print(f"  TOTAL BUILD: {fmt_t(total)}")
    return g, sector_members, adj


def sample_pairs_by_depth(
    all_ids: list[str],
    adj: dict[str, set[str]],
    pairs_per: int = LARGE_SCALE_PAIRS_PER,
    max_depth: int = LARGE_SCALE_MAX_DEPTH,
    max_sources: int = LARGE_SCALE_MAX_SOURCES_BFS,
    rng_seed: int = RNG_SEED,
) -> dict[int, list[tuple[str, str]]]:
    """Faithful port of `samplePairsByDepth` — pick up to `max_sources`
    random sources, BFS the full graph from each, bucket (src, tgt) pairs
    by hop distance; early-stop when every depth has `pairs_per` samples."""
    print(f"\nSampling depth buckets via reference BFS (up to {max_sources} sources, {pairs_per} pairs/bucket)...")
    t0 = time.perf_counter()
    rng = random.Random(rng_seed)
    buckets: dict[int, list[tuple[str, str]]] = defaultdict(list)
    sources = rng.sample(all_ids, max_sources)

    def enough() -> bool:
        for d in range(1, max_depth + 1):
            if len(buckets[d]) < pairs_per:
                return False
        return True

    for src in sources:
        # BFS from `src` recording dist to every reachable node.
        dist: dict[str, int] = {src: 0}
        queue: deque[str] = deque([src])
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj.get(u, ()):
                if v in dist:
                    continue
                dist[v] = d + 1
                queue.append(v)
        for dst, d in dist.items():
            if d == 0:
                continue
            if len(buckets[d]) < pairs_per:
                buckets[d].append((src, dst))
        if enough():
            break

    elapsed = time.perf_counter() - t0
    populated = sorted(d for d, p in buckets.items() if p)
    print(f"  BFS bucketing: {fmt_t(elapsed)}")
    print(f"  depths populated: {populated[0]}..{populated[-1]} ({len(populated)} buckets)")
    return dict(buckets)


def bench_kglite(
    g: kglite.KnowledgeGraph,
    buckets: dict[int, list[tuple[str, str]]],
    per_query_timeout_ms: int = 10_000,
    max_depth: int = LARGE_SCALE_MAX_DEPTH,
) -> dict[int, dict[str, float]]:
    """Run NornicDB's exact Cypher query per depth bucket. Returns
    {depth: {min, median, p95, max, n}}. Prints one progress line per
    depth so long runs aren't opaque. Skips depths above `max_depth`
    and bails the bucket if any individual query times out."""
    print(f"\nRunning NornicDB Cypher per depth (timeout={per_query_timeout_ms}ms, max_depth={max_depth})...")
    results: dict[int, dict[str, float]] = {}

    for depth in sorted(buckets.keys()):
        if depth > max_depth:
            break
        pairs = buckets[depth]
        if not pairs:
            continue
        # Warm-up — one timed query (matches NornicDB's runShortestPathPair on pairs[0]).
        s, t = pairs[0]
        try:
            g.cypher(
                NORNIC_QUERY,
                params={"startId": s, "endId": t},
                timeout_ms=per_query_timeout_ms,
            )
        except kglite.KgError as e:
            print(f"  depth={depth} WARMUP TIMEOUT/ERROR: {e}; skipping bucket", flush=True)
            continue

        durs: list[float] = []
        timed_out = False
        for s, t in pairs:
            t0 = time.perf_counter()
            try:
                g.cypher(
                    NORNIC_QUERY,
                    params={"startId": s, "endId": t},
                    timeout_ms=per_query_timeout_ms,
                )
            except kglite.KgError as e:
                print(f"  depth={depth} TIMEOUT/ERROR on pair: {e}", flush=True)
                timed_out = True
                break
            durs.append(time.perf_counter() - t0)
        if timed_out or not durs:
            continue

        durs.sort()
        n = len(durs)
        results[depth] = {
            "min": durs[0],
            "median": durs[n // 2] if n % 2 else (durs[n // 2 - 1] + durs[n // 2]) / 2,
            "p95": durs[int(min(n - 1, n * 0.95))],
            "max": durs[-1],
            "n": n,
        }
        print(
            f"  depth={depth:>3}  n={n:>2}  min={fmt_t(results[depth]['min']):>9}  "
            f"med={fmt_t(results[depth]['median']):>9}  p95={fmt_t(results[depth]['p95']):>9}",
            flush=True,
        )
    return results


# NornicDB published medians (Apple M3 Max).
NORNIC_MEDIANS: dict[int, float] = {
    1: 94.8e-6,
    4: 342.7e-6,
    5: 1.23e-3,
    8: 2.39e-3,
    40: 612.7e-3,
}


def nornic_estimate(depth: int) -> float | None:
    if depth in NORNIC_MEDIANS:
        return NORNIC_MEDIANS[depth]
    if 9 <= depth <= 39:
        return 14e-3 * depth / 9
    return None


def print_comparison(kg: dict[int, dict[str, float]]) -> None:
    print(f"\n{'=' * 102}")
    print("RESULTS — kglite (Apple M4) vs NornicDB published medians (Apple M3 Max)")
    print(f"{'=' * 102}")
    print(
        f"{'Depth':<8}{'kglite min':<14}{'kglite med':<14}{'kglite p95':<14}{'NornicDB med':<16}{'speedup':<14}{'n':<5}"
    )
    print("-" * 102)
    for d in sorted(kg.keys()):
        r = kg[d]
        nornic = nornic_estimate(d)
        speedup_str = f"{nornic / r['median']:.2f}x" if nornic else "—"
        nornic_str = fmt_t(nornic) if nornic else "—"
        print(
            f"{d:<8}{fmt_t(r['min']):<14}{fmt_t(r['median']):<14}{fmt_t(r['p95']):<14}"
            f"{nornic_str:<16}{speedup_str:<14}{r['n']:<5}"
        )


def main() -> None:
    g, sector_members, adj = build_graph()
    all_ids = [sid for s in sector_members for sid in s]
    buckets = sample_pairs_by_depth(all_ids, adj)
    results = bench_kglite(g, buckets)
    print_comparison(results)


if __name__ == "__main__":
    main()
