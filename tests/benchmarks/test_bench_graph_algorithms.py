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

COMMUNITY_GROUPS = 32
COMMUNITY_GROUP_SIZE = 24
DBSCAN_EUCLIDEAN_CLUSTERS = 640
DBSCAN_EUCLIDEAN_NOISE = 128
DBSCAN_EUCLIDEAN_POINTS = DBSCAN_EUCLIDEAN_CLUSTERS * 3 + DBSCAN_EUCLIDEAN_NOISE
DBSCAN_HAVERSINE_POINTS = 1_024


@dataclass(frozen=True)
class GraphCase:
    graph: KnowledgeGraph
    node_count: int
    edges: tuple[tuple[int, int], ...]
    weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DbscanCase:
    graph: KnowledgeGraph
    query: str
    expected_clusters: tuple[int, ...]


def _build_case(
    node_count: int,
    edges: list[tuple[int, int]],
    edge_weights: dict[tuple[int, int], float] | None = None,
) -> GraphCase:
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
    stored_weights = None
    if unique_edges:
        frame = pd.DataFrame(unique_edges, columns=["source", "target"])
        if edge_weights is not None:
            stored_weights = tuple(edge_weights[edge] for edge in unique_edges)
            frame["weight"] = stored_weights
        graph.add_connections(
            frame,
            "LINK",
            "Node",
            "source",
            "Node",
            "target",
        )
    assert graph.shape == (node_count, len(unique_edges))
    assert graph.node_type_counts() == {"Node": node_count}
    return GraphCase(graph, node_count, unique_edges, stored_weights)


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


def _weighted_clique_blocks(groups: int, group_size: int) -> GraphCase:
    edges: list[tuple[int, int]] = []
    weights: dict[tuple[int, int], float] = {}
    for group in range(groups):
        start = group * group_size
        for left in range(group_size):
            for right in range(left + 1, group_size):
                edge = (start + left, start + right)
                edges.append(edge)
                weights[edge] = 2.0
        next_start = ((group + 1) % groups) * group_size
        bridge = (start + group_size - 1, next_start)
        edges.append(bridge)
        weights[bridge] = 0.05
    case = _build_case(groups * group_size, edges, weights)
    assert len(case.edges) == groups * (math.comb(group_size, 2) + 1)
    assert case.weights is not None
    return case


def _weighted_cycle_case() -> GraphCase:
    """Six-cycle whose weighted optimum differs from its unweighted optimum."""
    weighted_edges = [
        (0, 1, 10.0),
        (1, 2, 0.1),
        (2, 3, 10.0),
        (3, 4, 0.1),
        (4, 5, 10.0),
        (5, 0, 0.1),
    ]
    edges = [(source, target) for source, target, _ in weighted_edges]
    weights = {(source, target): weight for source, target, weight in weighted_edges}
    return _build_case(6, edges, weights)


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
    assert {source for source, _ in case.edges} == set(range(node_count))
    return case


def _pagerank_sink_heavy_case(node_count: int = 4_097) -> GraphCase:
    """Parallel PageRank fixture with three quarters of nodes dangling.

    The active quarter has eight outgoing links per node.  This keeps total
    edge work comparable to the existing no-sink fixture while making the
    dangling-rank redistribution a material part of every iteration.
    """
    active_sources = node_count // 4
    offsets = (1, 7, 31, 127, 509, 1_021, 2_053, 3_079)
    edges = [(source, (source + offset) % node_count) for source in range(active_sources) for offset in offsets]
    case = _build_case(node_count, edges)
    sources = {source for source, _ in case.edges}
    assert sources == set(range(active_sources))
    assert case.node_count - len(sources) >= 3 * case.node_count // 4
    assert len(case.edges) == active_sources * len(offsets)
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


def _brandes_path_case(node_count: int) -> GraphCase:
    """Path graph with closed-form sampled Brandes scores."""
    case = _build_case(node_count, [(node, node + 1) for node in range(node_count - 1)])
    assert len(case.edges) == node_count - 1
    return case


def _coreness_case(core_nodes: int = 1_792, leaves: int = 256) -> GraphCase:
    edges = [(node, (node + offset) % core_nodes) for node in range(core_nodes) for offset in (1, 2, 4)]
    edges.extend((leaf, leaf % core_nodes) for leaf in range(core_nodes, core_nodes + leaves))
    case = _build_case(core_nodes + leaves, edges)
    assert len(case.edges) == 3 * core_nodes + leaves
    return case


@pytest.fixture(scope="module")
def dense_clustering_case() -> GraphCase:
    return _clique_blocks(groups=COMMUNITY_GROUPS, group_size=COMMUNITY_GROUP_SIZE)


@pytest.fixture(scope="module")
def weighted_community_case() -> GraphCase:
    return _weighted_clique_blocks(groups=COMMUNITY_GROUPS, group_size=COMMUNITY_GROUP_SIZE)


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
def pagerank_sink_heavy_case() -> GraphCase:
    case = _pagerank_sink_heavy_case()
    assert case.node_count >= 4_096
    return case


@pytest.fixture(scope="module")
def brandes_case() -> GraphCase:
    return _sampled_brandes_case()


@pytest.fixture(scope="module", params=[4_095, 4_097], ids=["sequential", "parallel"])
def brandes_threshold_case(request: pytest.FixtureRequest) -> GraphCase:
    case = _brandes_path_case(request.param)
    assert (case.node_count >= 4_096) is (request.param == 4_097)
    return case


@pytest.fixture(scope="module")
def coreness_case() -> GraphCase:
    return _coreness_case()


@pytest.fixture(scope="module")
def dbscan_euclidean_case() -> DbscanCase:
    """Eight-dimensional triplets plus isolated points (2048 total)."""
    expected = tuple(
        point // 3 if point < DBSCAN_EUCLIDEAN_CLUSTERS * 3 else -1 for point in range(DBSCAN_EUCLIDEAN_POINTS)
    )
    features = {f"f{dimension}": [] for dimension in range(8)}
    for point in range(DBSCAN_EUCLIDEAN_POINTS):
        if point < DBSCAN_EUCLIDEAN_CLUSTERS * 3:
            group, member = divmod(point, 3)
            base = float(group * 100)
            offset = member * 0.05
        else:
            base = float((DBSCAN_EUCLIDEAN_CLUSTERS + point) * 100)
            offset = 0.0
        for dimension in range(8):
            features[f"f{dimension}"].append(base + offset + dimension * 0.001)

    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(DBSCAN_EUCLIDEAN_POINTS),
                "name": [f"point_{point}" for point in range(DBSCAN_EUCLIDEAN_POINTS)],
                **features,
            }
        ),
        "Point",
        "id",
        "name",
        columns=list(features),
    )
    assert graph.shape == (DBSCAN_EUCLIDEAN_POINTS, 0)
    assert graph.node_type_counts() == {"Point": DBSCAN_EUCLIDEAN_POINTS}
    query = """
        MATCH (point:Point)
        CALL cluster({
            properties: ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'],
            method: 'dbscan', eps: 0.3, min_points: 2
        })
        YIELD node, cluster
        RETURN node.id AS id, cluster ORDER BY id
    """
    return DbscanCase(graph, query, expected)


@pytest.fixture(scope="module")
def dbscan_haversine_case() -> DbscanCase:
    """Two three-point geographic clusters plus a coarse isolated grid."""
    coordinates = [
        (59.91000, 10.75000),
        (59.91001, 10.75000),
        (59.91002, 10.75000),
        (41.90000, 12.50000),
        (41.90001, 12.50000),
        (41.90002, 12.50000),
    ]
    noise_count = DBSCAN_HAVERSINE_POINTS - len(coordinates)
    coordinates.extend((-60.0 + 3.0 * (point // 40), -150.0 + 7.0 * (point % 40)) for point in range(noise_count))
    expected = (0, 0, 0, 1, 1, 1, *([-1] * noise_count))

    graph = KnowledgeGraph()
    graph.set_spatial("GeoPoint", location=("lat", "lon"))
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(DBSCAN_HAVERSINE_POINTS),
                "name": [f"geo_{point}" for point in range(DBSCAN_HAVERSINE_POINTS)],
                "lat": [lat for lat, _ in coordinates],
                "lon": [lon for _, lon in coordinates],
            }
        ),
        "GeoPoint",
        "id",
        "name",
        columns=["lat", "lon"],
    )
    assert len(coordinates) == DBSCAN_HAVERSINE_POINTS
    assert graph.shape == (DBSCAN_HAVERSINE_POINTS, 0)
    assert graph.node_type_counts() == {"GeoPoint": DBSCAN_HAVERSINE_POINTS}
    query = """
        MATCH (point:GeoPoint)
        CALL cluster({method: 'dbscan', eps: 3.0, min_points: 2})
        YIELD node, cluster
        RETURN node.id AS id, cluster ORDER BY id
    """
    return DbscanCase(graph, query, expected)


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


def _path_brandes_oracle(case: GraphCase, sample_size: int) -> dict[int, float]:
    """Closed-form normalized Brandes scores for an evenly sampled path."""
    node_count = case.node_count
    step = node_count / sample_size
    sources = [int(sample * step) for sample in range(sample_size)]
    denominator = (node_count - 1) * (node_count - 2)
    scores: dict[int, float] = {}
    for node in range(node_count):
        sources_left = sum(source < node for source in sources)
        sources_right = sum(source > node for source in sources)
        dependencies = sources_left * (node_count - node - 1) + sources_right * node
        scores[node] = dependencies * node_count / (sample_size * denominator)
    return scores


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


def _centrality_scores(method, **kwargs) -> dict[object, float]:
    """`{id: score}` from a centrality ResultView.

    Replaces the removed `as_dict=True`, which keyed by bare id and silently
    dropped rows on cross-type id collisions. These fixtures are single-type,
    so the mapping is lossless; the cell still materializes every row.
    """
    return {row["id"]: row["score"] for row in method(**kwargs).to_dicts()}


def _clustering_rows(graph: KnowledgeGraph) -> list[dict[str, object]]:
    return graph.cypher(
        "CALL clustering_coefficient() YIELD node, coefficient RETURN node.id AS id, coefficient"
    ).to_list()


def _dbscan_rows(case: DbscanCase) -> list[dict[str, object]]:
    return case.graph.cypher(case.query, timeout_ms=0).to_list()


def _canonical_dbscan_result(
    clusters: tuple[int, ...],
) -> tuple[frozenset[frozenset[int]], frozenset[int]]:
    groups: dict[int, set[int]] = {}
    noise: set[int] = set()
    for node_id, cluster in enumerate(clusters):
        if cluster == -1:
            noise.add(node_id)
        else:
            assert cluster >= 0, f"invalid DBSCAN cluster label: {cluster}"
            groups.setdefault(cluster, set()).add(node_id)
    return frozenset(frozenset(group) for group in groups.values()), frozenset(noise)


def _assert_dbscan_rows(case: DbscanCase, rows: list[dict[str, object]]) -> None:
    assert all(set(row) == {"id", "cluster"} for row in rows)
    assert [row["id"] for row in rows] == list(range(len(case.expected_clusters)))
    actual_clusters = tuple(int(row["cluster"]) for row in rows)
    assert all(cluster == row["cluster"] for cluster, row in zip(actual_clusters, rows, strict=True))
    assert _canonical_dbscan_result(actual_clusters) == _canonical_dbscan_result(case.expected_clusters)


def _community_rows(
    graph: KnowledgeGraph,
    procedure: str,
    weight_property: str | None,
) -> list[dict[str, object]]:
    weight_option = ", weight_property: 'weight'" if weight_property else ""
    return graph.cypher(
        f"CALL {procedure}({{connection_types: ['LINK']{weight_option}}}) "
        "YIELD node, community RETURN node.id AS id, community"
    ).to_list()


def _canonical_partition(rows: list[dict[str, object]]) -> frozenset[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for row in rows:
        groups.setdefault(int(row["community"]), set()).add(int(row["id"]))
    return frozenset(frozenset(group) for group in groups.values())


def _louvain_partition(result: dict[str, object]) -> frozenset[frozenset[int]]:
    communities = result["communities"]
    assert isinstance(communities, dict)
    return frozenset(frozenset(int(member["id"]) for member in members) for members in communities.values())


def _expected_community_partition() -> frozenset[frozenset[int]]:
    return frozenset(
        frozenset(range(group * COMMUNITY_GROUP_SIZE, (group + 1) * COMMUNITY_GROUP_SIZE))
        for group in range(COMMUNITY_GROUPS)
    )


def _assert_partition_connected(case: GraphCase, partition: frozenset[frozenset[int]]) -> None:
    adjacency = _undirected_adjacency(case)
    for community in partition:
        pending = [next(iter(community))]
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            pending.extend(adjacency[node] & community)
        assert visited == set(community)


def _partition_modularity(
    case: GraphCase,
    partition: frozenset[frozenset[int]],
    weighted: bool,
) -> float:
    weights = case.weights if weighted else (1.0,) * len(case.edges)
    assert weights is not None
    membership = {node: community for community, nodes in enumerate(partition) for node in nodes}
    degrees = [0.0] * case.node_count
    internal_weight = [0.0] * len(partition)
    for (source, target), weight in zip(case.edges, weights, strict=True):
        degrees[source] += weight
        degrees[target] += weight
        if membership[source] == membership[target]:
            internal_weight[membership[source]] += weight
    total_weight = sum(weights)
    two_m = 2.0 * total_weight
    degree_sum = [0.0] * len(partition)
    for node, degree in enumerate(degrees):
        degree_sum[membership[node]] += degree
    return sum(
        internal / total_weight - (community_degree / two_m) ** 2
        for internal, community_degree in zip(internal_weight, degree_sum, strict=True)
    )


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


def test_dbscan_benchmark_cases_match_exact_oracle(
    dbscan_euclidean_case: DbscanCase,
    dbscan_haversine_case: DbscanCase,
) -> None:
    for case in (dbscan_euclidean_case, dbscan_haversine_case):
        rows = _dbscan_rows(case)
        _assert_dbscan_rows(case, rows)
        assert any(cluster >= 0 for cluster in case.expected_clusters)
        assert any(cluster == -1 for cluster in case.expected_clusters)


def test_dbscan_oracle_rejects_membership_and_noise_mutations(
    dbscan_euclidean_case: DbscanCase,
) -> None:
    expected = dbscan_euclidean_case.expected_clusters
    arbitrarily_relabelled = [
        {"id": node_id, "cluster": cluster + 10_000 if cluster >= 0 else -1} for node_id, cluster in enumerate(expected)
    ]
    _assert_dbscan_rows(dbscan_euclidean_case, arbitrarily_relabelled)

    wrong_membership = [dict(row) for row in arbitrarily_relabelled]
    wrong_membership[3]["cluster"] = wrong_membership[0]["cluster"]
    with pytest.raises(AssertionError):
        _assert_dbscan_rows(dbscan_euclidean_case, wrong_membership)

    wrong_noise = [dict(row) for row in arbitrarily_relabelled]
    wrong_noise[-1]["cluster"] = wrong_noise[0]["cluster"]
    with pytest.raises(AssertionError):
        _assert_dbscan_rows(dbscan_euclidean_case, wrong_noise)


@pytest.mark.parametrize("node_count", [127, 4_097], ids=["sequential", "parallel"])
def test_pagerank_matches_scalar_oracle(node_count: int) -> None:
    case = _pagerank_case(node_count)
    expected = _pagerank_oracle(case)
    # `as_dict=True` removed: same mapping, built from the ResultView.
    actual = _centrality_scores(
        case.graph.pagerank,
        damping_factor=0.85,
        max_iterations=30,
        tolerance=0.0,
        connection_types=["LINK"],
    )
    assert set(actual) == set(expected)
    assert actual == pytest.approx(expected, rel=2e-13, abs=2e-15)
    assert sum(actual.values()) == pytest.approx(1.0, abs=2e-13)


def test_pagerank_sink_heavy_matches_scalar_oracle(
    pagerank_sink_heavy_case: GraphCase,
) -> None:
    """Pin dangling redistribution on the parallel benchmark fixture."""
    expected = _pagerank_oracle(pagerank_sink_heavy_case)
    # `as_dict=True` removed: same mapping, built from the ResultView.
    actual = _centrality_scores(
        pagerank_sink_heavy_case.graph.pagerank,
        damping_factor=0.85,
        max_iterations=30,
        tolerance=0.0,
        connection_types=["LINK"],
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


def test_planted_louvain_and_leiden_oracles(
    dense_clustering_case: GraphCase,
    weighted_community_case: GraphCase,
) -> None:
    expected = _expected_community_partition()
    for case, weight_property in (
        (dense_clustering_case, None),
        (weighted_community_case, "weight"),
    ):
        louvain = case.graph.louvain_communities(
            connection_types=["LINK"],
            weight_property=weight_property,
        )
        louvain_partition = _louvain_partition(louvain)
        leiden_partition = _canonical_partition(_community_rows(case.graph, "leiden", weight_property))

        assert louvain_partition == expected
        assert leiden_partition == expected
        _assert_partition_connected(case, leiden_partition)

        weighted = weight_property is not None
        louvain_modularity = _partition_modularity(case, louvain_partition, weighted)
        leiden_modularity = _partition_modularity(case, leiden_partition, weighted)
        assert float(louvain["modularity"]) == pytest.approx(louvain_modularity, abs=1e-12)
        assert leiden_modularity >= louvain_modularity - 1e-12


def test_weighted_community_options_change_partition() -> None:
    """Ensure both public weighted paths actually consume the weight property."""
    case = _weighted_cycle_case()
    expected = frozenset((frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5))))
    unweighted = _canonical_partition(_community_rows(case.graph, "leiden", None))
    weighted_leiden = _canonical_partition(_community_rows(case.graph, "leiden", "weight"))
    weighted_louvain = _louvain_partition(
        case.graph.louvain_communities(
            connection_types=["LINK"],
            weight_property="weight",
        )
    )

    assert unweighted != expected
    assert weighted_leiden == expected
    assert weighted_louvain == expected


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
    # `as_dict=True` removed: same mapping, built from the ResultView.
    actual = _centrality_scores(
        case.graph.betweenness_centrality,
        normalized=False,
        sample_size=sample_size,
        connection_types=["LINK"],
    )
    assert actual == pytest.approx(expected, abs=1e-14)


def test_brandes_threshold_cases_match_path_oracle(
    brandes_threshold_case: GraphCase,
) -> None:
    """Exercise the closed-form oracle on both sides of the Rayon threshold."""
    sample_size = 64
    expected = _path_brandes_oracle(brandes_threshold_case, sample_size)
    # `as_dict=True` removed: same mapping, built from the ResultView.
    actual = _centrality_scores(
        brandes_threshold_case.graph.betweenness_centrality,
        normalized=True,
        sample_size=sample_size,
        connection_types=["LINK"],
    )
    assert actual == pytest.approx(expected, rel=2e-13, abs=2e-15)


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
def test_bench_dbscan_euclidean_distance_matrix(
    benchmark,
    dbscan_euclidean_case: DbscanCase,
) -> None:
    rows = benchmark(_dbscan_rows, dbscan_euclidean_case)
    _assert_dbscan_rows(dbscan_euclidean_case, rows)
    benchmark.extra_info["points"] = DBSCAN_EUCLIDEAN_POINTS
    benchmark.extra_info["dense_matrix_bytes"] = 8 * DBSCAN_EUCLIDEAN_POINTS**2


@pytest.mark.benchmark
def test_bench_dbscan_haversine_distance_matrix(
    benchmark,
    dbscan_haversine_case: DbscanCase,
) -> None:
    rows = benchmark(_dbscan_rows, dbscan_haversine_case)
    _assert_dbscan_rows(dbscan_haversine_case, rows)
    benchmark.extra_info["points"] = DBSCAN_HAVERSINE_POINTS
    benchmark.extra_info["dense_matrix_bytes"] = 8 * DBSCAN_HAVERSINE_POINTS**2


@pytest.mark.benchmark
def test_bench_louvain_planted(benchmark, dense_clustering_case: GraphCase) -> None:
    result = benchmark(
        dense_clustering_case.graph.louvain_communities,
        connection_types=["LINK"],
    )
    assert result["num_communities"] == 32
    assert sum(map(len, result["communities"].values())) == dense_clustering_case.node_count
    assert result["modularity"] > 0.45
    assert _louvain_partition(result) == _expected_community_partition()


@pytest.mark.benchmark
def test_bench_louvain_weighted_planted(benchmark, weighted_community_case: GraphCase) -> None:
    result = benchmark(
        weighted_community_case.graph.louvain_communities,
        weight_property="weight",
        connection_types=["LINK"],
    )
    assert _louvain_partition(result) == _expected_community_partition()


@pytest.mark.benchmark
def test_bench_leiden_planted(benchmark, dense_clustering_case: GraphCase) -> None:
    rows = benchmark(_community_rows, dense_clustering_case.graph, "leiden", None)
    partition = _canonical_partition(rows)
    assert partition == _expected_community_partition()
    _assert_partition_connected(dense_clustering_case, partition)


@pytest.mark.benchmark
def test_bench_leiden_weighted_planted(benchmark, weighted_community_case: GraphCase) -> None:
    rows = benchmark(_community_rows, weighted_community_case.graph, "leiden", "weight")
    partition = _canonical_partition(rows)
    assert partition == _expected_community_partition()
    _assert_partition_connected(weighted_community_case, partition)


@pytest.mark.benchmark
def test_bench_pagerank_below_parallel_threshold(benchmark, pagerank_below_case: GraphCase) -> None:
    # `as_dict=True` removed: the cell still times full row materialization.
    result = benchmark(
        _centrality_scores,
        pagerank_below_case.graph.pagerank,
        connection_types=["LINK"],
    )
    assert len(result) == pagerank_below_case.node_count
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.benchmark
def test_bench_pagerank_above_parallel_threshold(benchmark, pagerank_above_case: GraphCase) -> None:
    # `as_dict=True` removed: the cell still times full row materialization.
    result = benchmark(
        _centrality_scores,
        pagerank_above_case.graph.pagerank,
        connection_types=["LINK"],
    )
    assert len(result) == pagerank_above_case.node_count
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-8)
    benchmark.extra_info["dangling_nodes"] = 0


@pytest.mark.benchmark
def test_bench_pagerank_sink_heavy_parallel(benchmark, pagerank_sink_heavy_case: GraphCase) -> None:
    # `as_dict=True` removed: the cell still times full row materialization.
    result = benchmark(
        _centrality_scores,
        pagerank_sink_heavy_case.graph.pagerank,
        connection_types=["LINK"],
    )
    dangling_nodes = pagerank_sink_heavy_case.node_count - len({source for source, _ in pagerank_sink_heavy_case.edges})
    assert len(result) == pagerank_sink_heavy_case.node_count
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-8)
    assert dangling_nodes >= 3 * pagerank_sink_heavy_case.node_count // 4
    benchmark.extra_info["dangling_nodes"] = dangling_nodes


@pytest.mark.benchmark
def test_bench_brandes_sampled(benchmark, brandes_case: GraphCase) -> None:
    # `as_dict=True` removed: the cell still times full row materialization.
    result = benchmark(
        _centrality_scores,
        brandes_case.graph.betweenness_centrality,
        normalized=True,
        sample_size=64,
        connection_types=["LINK"],
    )
    assert len(result) == brandes_case.node_count
    assert max(result.values()) > 0.0
    benchmark.extra_info["queue_shape"] = "mixed_sparse"


@pytest.mark.benchmark
@pytest.mark.parametrize("sample_size", [1, 64, 256], ids=["sample_1", "sample_64", "sample_256"])
def test_bench_brandes_work_threshold(
    benchmark,
    brandes_threshold_case: GraphCase,
    sample_size: int,
) -> None:
    # `as_dict=True` removed: the cell still times full row materialization.
    result = benchmark(
        _centrality_scores,
        brandes_threshold_case.graph.betweenness_centrality,
        normalized=True,
        sample_size=sample_size,
        connection_types=["LINK"],
    )
    expected = _path_brandes_oracle(brandes_threshold_case, sample_size)
    expected_max = max(expected.values())
    expected_top = {node for node, score in expected.items() if score == expected_max}

    assert len(result) == brandes_threshold_case.node_count
    assert result == pytest.approx(expected, rel=2e-13, abs=2e-15)
    assert next(iter(result)) in expected_top
    assert result[next(iter(result))] == pytest.approx(expected_max, rel=2e-13, abs=2e-15)
    benchmark.extra_info["nodes"] = brandes_threshold_case.node_count
    benchmark.extra_info["sample_size"] = sample_size
    benchmark.extra_info["parallel"] = brandes_threshold_case.node_count >= 4_096
    benchmark.extra_info["queue_shape"] = "narrow_path"


@pytest.mark.benchmark
def test_bench_coreness_nonuniform(benchmark, coreness_case: GraphCase) -> None:
    rows = benchmark(_coreness_rows, coreness_case.graph)
    by_id = {int(row["id"]): int(row["coreness"]) for row in rows}
    assert by_id == {node: 6 if node < 1_792 else 1 for node in range(coreness_case.node_count)}
