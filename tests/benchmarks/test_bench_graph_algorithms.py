"""Deterministic performance cells and accuracy oracles for graph algorithms.

The benchmark cells measure only the public algorithm call and result
materialisation.  Graph construction lives in module-scoped fixtures.  The
unmarked tests are independent correctness gates that remain active when the
default pytest marker expression deselects benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@dataclass(frozen=True)
class GraphCase:
    graph: KnowledgeGraph
    node_count: int
    edges: tuple[tuple[int, int], ...]


def _build_case(node_count: int, edges: list[tuple[int, int]]) -> GraphCase:
    """Build one in-memory graph and assert its benchmark trigger shape."""
    unique_edges = tuple(sorted(set(edges)))
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(node_count),
                "name": [f"n{i}" for i in range(node_count)],
            }
        ),
        "Node",
        "id",
        "name",
    )
    if unique_edges:
        graph.add_connections(
            pd.DataFrame(unique_edges, columns=["source", "target"]),
            "LINK",
            "Node",
            "source",
            "Node",
            "target",
        )
    assert graph.shape == (node_count, len(unique_edges))
    assert graph.node_type_counts() == {"Node": node_count}
    return GraphCase(graph, node_count, unique_edges)


def _clique_blocks(groups: int, group_size: int) -> GraphCase:
    edges: list[tuple[int, int]] = []
    for group in range(groups):
        start = group * group_size
        edges.extend(
            (start + left, start + right) for left in range(group_size) for right in range(left + 1, group_size)
        )
        # One sparse bridge per dense block makes the planted partition
        # connected without obscuring its community boundary.
        next_start = ((group + 1) % groups) * group_size
        edges.append((start + group_size - 1, next_start))
    case = _build_case(groups * group_size, edges)
    assert len(case.edges) == groups * (math.comb(group_size, 2) + 1)
    return case


def _sparse_hub_case(node_count: int) -> GraphCase:
    edges: list[tuple[int, int]] = []
    for node in range(1, node_count):
        # A deterministic skewed-degree graph: the global hub and binary-tree
        # parent create long sorted prefixes, while the third edge adds local
        # triangles and prevents the coefficient workload from being vacuous.
        for peer in {0, (node - 1) // 2, (node * 17 + 11) % node}:
            if peer != node:
                edges.append((peer, node))
    case = _build_case(node_count, edges)
    assert 2 * node_count < len(case.edges) < 3 * node_count
    return case


def _pagerank_case(node_count: int) -> GraphCase:
    edges = [
        edge
        for source in range(node_count)
        for edge in (
            (source, (source + 1) % node_count),
            (source, (source * 37 + 101) % node_count),
        )
        if edge[0] != edge[1]
    ]
    case = _build_case(node_count, edges)
    assert len(case.edges) >= 2 * node_count - 2
    return case


def _sampled_brandes_case(node_count: int = 1_024) -> GraphCase:
    edges = [
        edge
        for node in range(node_count)
        for edge in (
            (node, (node + 1) % node_count),
            (node, (node + 17) % node_count),
            (node, (node * 13 + 97) % node_count),
        )
        if edge[0] != edge[1]
    ]
    case = _build_case(node_count, edges)
    assert 2 * node_count < len(case.edges) <= 3 * node_count
    return case


def _coreness_case(core_nodes: int = 1_792, leaves: int = 256) -> GraphCase:
    edges = [(node, (node + offset) % core_nodes) for node in range(core_nodes) for offset in (1, 2, 4)]
    edges.extend((leaf, leaf % core_nodes) for leaf in range(core_nodes, core_nodes + leaves))
    case = _build_case(core_nodes + leaves, edges)
    assert len(case.edges) == 3 * core_nodes + leaves
    return case


@pytest.fixture(scope="module")
def dense_clustering_case() -> GraphCase:
    return _clique_blocks(groups=32, group_size=24)


@pytest.fixture(scope="module")
def sparse_clustering_case() -> GraphCase:
    return _sparse_hub_case(2_000)


@pytest.fixture(scope="module")
def pagerank_below_case() -> GraphCase:
    case = _pagerank_case(4_095)
    assert case.node_count < 4_096
    return case


@pytest.fixture(scope="module")
def pagerank_above_case() -> GraphCase:
    case = _pagerank_case(4_097)
    assert case.node_count >= 4_096
    return case


@pytest.fixture(scope="module")
def brandes_case() -> GraphCase:
    return _sampled_brandes_case()


@pytest.fixture(scope="module")
def coreness_case() -> GraphCase:
    return _coreness_case()


def _undirected_adjacency(case: GraphCase) -> list[set[int]]:
    adjacency = [set() for _ in range(case.node_count)]
    for source, target in case.edges:
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def _clustering_oracle(case: GraphCase) -> dict[int, float]:
    adjacency = _undirected_adjacency(case)
    coefficients: dict[int, float] = {}
    for node, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        if degree < 2:
            coefficients[node] = 0.0
            continue
        links = sum(1 for left in neighbors for right in neighbors if left < right and right in adjacency[left])
        coefficients[node] = 2.0 * links / (degree * (degree - 1))
    return coefficients


def _pagerank_oracle(
    case: GraphCase,
    *,
    damping: float = 0.85,
    max_iterations: int = 30,
) -> dict[int, float]:
    incoming: list[list[int]] = [[] for _ in range(case.node_count)]
    out_degree = [0] * case.node_count
    for source, target in case.edges:
        incoming[target].append(source)
        out_degree[source] += 1

    rank = [1.0 / case.node_count] * case.node_count
    teleport = (1.0 - damping) / case.node_count
    for _ in range(max_iterations):
        dangling = sum(score for score, degree in zip(rank, out_degree, strict=True) if degree == 0)
        base = teleport + damping * dangling / case.node_count
        rank = [base + sum(damping * rank[source] / out_degree[source] for source in sources) for sources in incoming]
    return dict(enumerate(rank))


def _slow_coreness_oracle(case: GraphCase) -> dict[int, int]:
    """Compute core numbers by repeated k-core pruning, independent of BZ."""
    adjacency = _undirected_adjacency(case)
    max_degree = max(map(len, adjacency), default=0)
    coreness = {node: 0 for node in range(case.node_count)}
    for k in range(1, max_degree + 1):
        remaining = set(range(case.node_count))
        changed = True
        while changed:
            removed = {node for node in remaining if sum(neighbor in remaining for neighbor in adjacency[node]) < k}
            changed = bool(removed)
            remaining.difference_update(removed)
        for node in remaining:
            coreness[node] = k
    return coreness


def _clustering_rows(graph: KnowledgeGraph) -> list[dict[str, object]]:
    return graph.cypher(
        "CALL clustering_coefficient() YIELD node, coefficient RETURN node.id AS id, coefficient"
    ).to_list()


def _coreness_rows(graph: KnowledgeGraph) -> list[dict[str, object]]:
    return graph.cypher("CALL k_core() YIELD node, coreness RETURN node.id AS id, coreness").to_list()


# ---------------------------------------------------------------------------
# Independent accuracy gates (unmarked: active in normal correctness runs)
# ---------------------------------------------------------------------------


def test_clustering_matches_hashset_oracle() -> None:
    case = _build_case(
        9,
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 3),
            (5, 6),
            (6, 7),
        ],
    )
    expected = _clustering_oracle(case)
    actual = {int(row["id"]): float(row["coefficient"]) for row in _clustering_rows(case.graph)}
    assert actual == pytest.approx(expected, abs=1e-15)


def test_clustering_benchmark_cases_match_hashset_oracle(
    dense_clustering_case: GraphCase,
    sparse_clustering_case: GraphCase,
) -> None:
    """Keep both timed fixture shapes non-vacuous and exact."""
    for case in (dense_clustering_case, sparse_clustering_case):
        expected = _clustering_oracle(case)
        actual = {int(row["id"]): float(row["coefficient"]) for row in _clustering_rows(case.graph)}
        assert actual == pytest.approx(expected, abs=1e-15)
        assert any(coefficient > 0.0 for coefficient in expected.values())


@pytest.mark.parametrize("node_count", [127, 4_097], ids=["sequential", "parallel"])
def test_pagerank_matches_scalar_oracle(node_count: int) -> None:
    case = _pagerank_case(node_count)
    expected = _pagerank_oracle(case)
    actual = case.graph.pagerank(
        damping_factor=0.85,
        max_iterations=30,
        tolerance=0.0,
        connection_types=["LINK"],
        as_dict=True,
    )
    assert set(actual) == set(expected)
    assert actual == pytest.approx(expected, rel=2e-13, abs=2e-15)
    assert sum(actual.values()) == pytest.approx(1.0, abs=2e-13)


def test_louvain_preserves_planted_partition() -> None:
    groups = 4
    group_size = 8
    case = _clique_blocks(groups, group_size)
    result = case.graph.louvain_communities(connection_types=["LINK"])
    actual = frozenset(frozenset(int(member["id"]) for member in members) for members in result["communities"].values())
    expected = frozenset(frozenset(range(group * group_size, (group + 1) * group_size)) for group in range(groups))
    assert actual == expected
    assert result["num_communities"] == groups
    assert result["modularity"] > 0.35


def test_brandes_matches_networkx() -> None:
    nx = pytest.importorskip("networkx")
    case = _build_case(
        12,
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (2, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (5, 8),
            (8, 9),
            (9, 10),
            (10, 11),
        ],
    )
    reference = nx.Graph()
    reference.add_nodes_from(range(case.node_count))
    reference.add_edges_from(case.edges)
    sample_size = 4
    # KGLite samples evenly by compact node index, then scales the subset
    # scores by n/k.  NetworkX supplies an independent Brandes implementation.
    sources = [0, 3, 6, 9]
    expected = nx.betweenness_centrality_subset(
        reference,
        sources=sources,
        targets=list(reference),
        normalized=False,
    )
    expected = {node: score * case.node_count / sample_size for node, score in expected.items()}
    actual = case.graph.betweenness_centrality(
        normalized=False,
        sample_size=sample_size,
        connection_types=["LINK"],
        as_dict=True,
    )
    assert actual == pytest.approx(expected, abs=1e-14)


def test_coreness_matches_slow_peel() -> None:
    case = _build_case(
        14,
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2),
            (1, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 4),
            (6, 7),
            (7, 8),
            (8, 9),
            (10, 11),
        ],
    )
    expected = _slow_coreness_oracle(case)
    actual = {int(row["id"]): int(row["coreness"]) for row in _coreness_rows(case.graph)}
    assert actual == expected


# ---------------------------------------------------------------------------
# Release-only performance cells
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_clustering_dense(benchmark, dense_clustering_case: GraphCase) -> None:
    rows = benchmark(_clustering_rows, dense_clustering_case.graph)
    assert len(rows) == dense_clustering_case.node_count
    assert all(0.0 < float(row["coefficient"]) <= 1.0 for row in rows)


@pytest.mark.benchmark
def test_bench_clustering_sparse(benchmark, sparse_clustering_case: GraphCase) -> None:
    rows = benchmark(_clustering_rows, sparse_clustering_case.graph)
    assert len(rows) == sparse_clustering_case.node_count
    assert any(float(row["coefficient"]) > 0.0 for row in rows)


@pytest.mark.benchmark
def test_bench_louvain_planted(benchmark, dense_clustering_case: GraphCase) -> None:
    result = benchmark(
        dense_clustering_case.graph.louvain_communities,
        connection_types=["LINK"],
    )
    assert result["num_communities"] == 32
    assert sum(map(len, result["communities"].values())) == dense_clustering_case.node_count
    assert result["modularity"] > 0.45


@pytest.mark.benchmark
def test_bench_pagerank_below_parallel_threshold(benchmark, pagerank_below_case: GraphCase) -> None:
    result = benchmark(
        pagerank_below_case.graph.pagerank,
        connection_types=["LINK"],
        as_dict=True,
    )
    assert len(result) == pagerank_below_case.node_count
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.benchmark
def test_bench_pagerank_above_parallel_threshold(benchmark, pagerank_above_case: GraphCase) -> None:
    result = benchmark(
        pagerank_above_case.graph.pagerank,
        connection_types=["LINK"],
        as_dict=True,
    )
    assert len(result) == pagerank_above_case.node_count
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.benchmark
def test_bench_brandes_sampled(benchmark, brandes_case: GraphCase) -> None:
    result = benchmark(
        brandes_case.graph.betweenness_centrality,
        normalized=True,
        sample_size=64,
        connection_types=["LINK"],
        as_dict=True,
    )
    assert len(result) == brandes_case.node_count
    assert max(result.values()) > 0.0


@pytest.mark.benchmark
def test_bench_coreness_nonuniform(benchmark, coreness_case: GraphCase) -> None:
    rows = benchmark(_coreness_rows, coreness_case.graph)
    by_id = {int(row["id"]): int(row["coreness"]) for row in rows}
    assert len(by_id) == coreness_case.node_count
    assert by_id[0] == 6
    assert by_id[1_791] == 6
    assert by_id[1_792] == 1
