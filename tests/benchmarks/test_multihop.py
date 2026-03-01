"""Multi-hop traversal benchmarks.

Measures query latency at increasing hop depths (1, 2, 4, 7, 8) from seed
nodes on graphs of varying sizes.  Comparable to TuringDB/Neo4j multi-hop
benchmarks.

Run:
    pytest tests/benchmarks/test_multihop.py -v -s -m benchmark
    pytest tests/benchmarks/test_multihop.py -v -s -m benchmark -k "1k"
    pytest tests/benchmarks/test_multihop.py -v -s -m benchmark -k "Summary"
"""

import random
import time

import pandas as pd
import pytest

from kglite import KnowledgeGraph

pytestmark = pytest.mark.benchmark

HOP_DEPTHS = [1, 2, 4, 7, 8]

# Per-query timeout in seconds.  If a query exceeds this, we record "timeout"
# instead of a time and skip larger hops for that graph size.
QUERY_TIMEOUT_S = 30.0

# ============================================================================
# Graph builders
# ============================================================================


def _build_scale_free_graph(n_nodes: int, edges_per_node: int = 4, seed: int = 42):
    """Build a scale-free-ish graph with controllable density.

    Uses preferential attachment (Barabasi-Albert style) so a few hub nodes
    have high degree -- realistic for biological/social networks and the kind
    of graph TuringDB benchmarks against.

    Returns (graph, seed_node_ids, n_edges) where seed_node_ids are the 15
    highest-degree nodes (matching TuringDB's "set of 15 seed nodes").
    """
    rng = random.Random(seed)

    # --- Build edges via preferential attachment ---
    core_size = max(edges_per_node + 1, 5)
    edge_set: set[tuple[int, int]] = set()
    repeated_nodes: list[int] = []

    for i in range(core_size):
        for j in range(i + 1, core_size):
            edge_set.add((i, j))
            repeated_nodes.extend([i, j])

    for new_node in range(core_size, n_nodes):
        chosen: set[int] = set()
        attempts = 0
        while len(chosen) < edges_per_node and attempts < edges_per_node * 10:
            t = rng.choice(repeated_nodes)
            if t != new_node:
                chosen.add(t)
            attempts += 1
        for t in chosen:
            edge_set.add((new_node, t))
            repeated_nodes.extend([new_node, t])

    # --- Load into KGLite ---
    graph = KnowledgeGraph()
    node_df = pd.DataFrame({
        "id": list(range(n_nodes)),
        "name": [f"N{i}" for i in range(n_nodes)],
        "group": [i % 20 for i in range(n_nodes)],
    })
    graph.add_nodes(node_df, "Node", "id", "name")

    edge_list = [{"src": s, "tgt": t} for s, t in edge_set]
    edge_df = pd.DataFrame(edge_list)
    graph.add_connections(edge_df, "LINKED", "Node", "src", "Node", "tgt")

    # 15 highest-degree nodes (by outgoing edges)
    degree_counts: dict[int, int] = {}
    for s, _t in edge_set:
        degree_counts[s] = degree_counts.get(s, 0) + 1
    top_seeds = sorted(degree_counts, key=degree_counts.get, reverse=True)[:15]

    return graph, top_seeds, len(edge_set)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def graph_1k():
    return _build_scale_free_graph(1_000, edges_per_node=4)


@pytest.fixture(scope="module")
def graph_10k():
    return _build_scale_free_graph(10_000, edges_per_node=4)


@pytest.fixture(scope="module")
def graph_50k():
    return _build_scale_free_graph(50_000, edges_per_node=4)


@pytest.fixture(scope="module")
def graph_100k():
    return _build_scale_free_graph(100_000, edges_per_node=4)


# ============================================================================
# Multi-hop query helper
# ============================================================================


def _run_multihop(graph, seed_ids, hops, timeout_s=QUERY_TIMEOUT_S):
    """Run a multi-hop count query.  Returns (count, elapsed_ms) or (None, None) on timeout."""
    seed_list = ", ".join(str(s) for s in seed_ids)

    if hops == 1:
        query = (
            f"MATCH (n:Node)-[:LINKED]->(m:Node) "
            f"WHERE n.id IN [{seed_list}] "
            f"RETURN count(DISTINCT m) AS cnt"
        )
    else:
        query = (
            f"MATCH (n:Node)-[:LINKED*1..{hops}]->(m:Node) "
            f"WHERE n.id IN [{seed_list}] "
            f"RETURN count(DISTINCT m) AS cnt"
        )

    # Use KGLite's built-in timeout
    timeout_ms = int(timeout_s * 1000)
    start = time.perf_counter()
    try:
        result = graph.cypher(query, timeout_ms=timeout_ms)
        elapsed_ms = (time.perf_counter() - start) * 1000
        cnt = result[0]["cnt"]
        return cnt, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            return None, elapsed_ms
        raise


# ============================================================================
# Parametrized benchmarks per graph size
# ============================================================================


class TestMultiHop1K:
    """Multi-hop traversal on 1K-node scale-free graph."""

    @pytest.mark.parametrize("hops", HOP_DEPTHS)
    def test_hop(self, graph_1k, hops):
        graph, seeds, n_edges = graph_1k
        cnt, elapsed = _run_multihop(graph, seeds, hops)
        if cnt is None:
            print(f"  1K nodes, {n_edges:,} edges | {hops}-hop: TIMEOUT (>{QUERY_TIMEOUT_S}s)")
        else:
            print(f"  1K nodes, {n_edges:,} edges | {hops}-hop: {elapsed:.1f}ms, {cnt:,} reached")


class TestMultiHop10K:
    """Multi-hop traversal on 10K-node scale-free graph."""

    @pytest.mark.parametrize("hops", HOP_DEPTHS)
    def test_hop(self, graph_10k, hops):
        graph, seeds, n_edges = graph_10k
        cnt, elapsed = _run_multihop(graph, seeds, hops)
        if cnt is None:
            print(f"  10K nodes, {n_edges:,} edges | {hops}-hop: TIMEOUT (>{QUERY_TIMEOUT_S}s)")
        else:
            print(f"  10K nodes, {n_edges:,} edges | {hops}-hop: {elapsed:.1f}ms, {cnt:,} reached")


class TestMultiHop50K:
    """Multi-hop traversal on 50K-node scale-free graph."""

    @pytest.mark.parametrize("hops", HOP_DEPTHS)
    def test_hop(self, graph_50k, hops):
        graph, seeds, n_edges = graph_50k
        cnt, elapsed = _run_multihop(graph, seeds, hops)
        if cnt is None:
            print(f"  50K nodes, {n_edges:,} edges | {hops}-hop: TIMEOUT (>{QUERY_TIMEOUT_S}s)")
        else:
            print(f"  50K nodes, {n_edges:,} edges | {hops}-hop: {elapsed:.1f}ms, {cnt:,} reached")


class TestMultiHop100K:
    """Multi-hop traversal on 100K-node scale-free graph."""

    @pytest.mark.parametrize("hops", HOP_DEPTHS)
    def test_hop(self, graph_100k, hops):
        graph, seeds, n_edges = graph_100k
        cnt, elapsed = _run_multihop(graph, seeds, hops)
        if cnt is None:
            print(f"  100K nodes, {n_edges:,} edges | {hops}-hop: TIMEOUT (>{QUERY_TIMEOUT_S}s)")
        else:
            print(f"  100K nodes, {n_edges:,} edges | {hops}-hop: {elapsed:.1f}ms, {cnt:,} reached")


# ============================================================================
# Summary table — all sizes x all hops with timeout protection
# ============================================================================


class TestMultiHopSummary:
    """Builds all graph sizes and prints a summary table."""

    SIZES = [
        (1_000, 4),
        (10_000, 4),
        (50_000, 4),
        (100_000, 4),
    ]

    def test_summary_table(self):
        print("\n" + "=" * 90)
        print("MULTI-HOP TRAVERSAL BENCHMARK")
        print(f"15 seed nodes (highest degree), scale-free graph, timeout={QUERY_TIMEOUT_S}s")
        print("=" * 90)

        # Header
        header = f"{'Graph':>15s} | {'Edges':>8s} | {'Build':>8s}"
        for h in HOP_DEPTHS:
            header += f" | {h}-hop{'':>8s}"
        print(header)
        print("-" * len(header))

        for n_nodes, epr in self.SIZES:
            build_start = time.perf_counter()
            graph, seeds, n_edges = _build_scale_free_graph(n_nodes, epr)
            build_ms = (time.perf_counter() - build_start) * 1000

            row = f"{n_nodes:>12,} n | {n_edges:>7,}e | {build_ms:>6.0f}ms"
            timed_out = False

            for hops in HOP_DEPTHS:
                if timed_out:
                    row += f" | {'--':>14s}"
                    continue

                cnt, elapsed = _run_multihop(graph, seeds, hops)
                if cnt is None:
                    row += f" | {'TIMEOUT':>14s}"
                    timed_out = True
                else:
                    row += f" | {elapsed:>6.0f}ms ({cnt:>4,})"

            print(row)

        print("=" * 90)
        print("(N) = distinct nodes reached from 15 seeds | -- = skipped after timeout")
        print()


# ============================================================================
# Single-seed benchmarks for deeper hops
# ============================================================================


class TestMultiHopSingleSeed:
    """Use a single seed node to test deeper hops without combinatorial blowup."""

    SIZES = [1_000, 10_000]

    @pytest.mark.parametrize("hops", HOP_DEPTHS)
    @pytest.mark.parametrize("n_nodes", [1_000, 10_000])
    def test_single_seed(self, n_nodes, hops):
        graph, seeds, n_edges = _build_scale_free_graph(n_nodes, edges_per_node=4)
        # Use only the top-1 seed
        cnt, elapsed = _run_multihop(graph, [seeds[0]], hops)
        if cnt is None:
            print(f"  {n_nodes//1000}K, 1 seed | {hops}-hop: TIMEOUT")
        else:
            print(f"  {n_nodes//1000}K, 1 seed | {hops}-hop: {elapsed:.1f}ms, {cnt:,} reached")


# ============================================================================
# Graph construction benchmark
# ============================================================================


class TestGraphConstruction:
    """Measure graph building time (node loading + edge loading)."""

    @pytest.mark.parametrize("n_nodes", [1_000, 10_000, 50_000, 100_000])
    def test_build_time(self, n_nodes):
        start = time.perf_counter()
        _graph, _seeds, n_edges = _build_scale_free_graph(n_nodes, edges_per_node=4)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {n_nodes:>7,} nodes, {n_edges:>7,} edges: {elapsed:.0f}ms")
