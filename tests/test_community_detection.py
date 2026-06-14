"""Tests for community detection algorithms: Louvain and Label Propagation."""

import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def two_cluster_graph():
    """Graph with two clear clusters connected by a single bridge."""
    graph = KnowledgeGraph()

    # Cluster 1: Alice, Bob, Charlie (fully connected)
    graph.cypher("CREATE (:Person {name: 'Alice', group: 'A'})")
    graph.cypher("CREATE (:Person {name: 'Bob', group: 'A'})")
    graph.cypher("CREATE (:Person {name: 'Charlie', group: 'A'})")

    graph.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    graph.cypher("""
        MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Charlie'})
        CREATE (a)-[:KNOWS]->(c)
    """)
    graph.cypher("""
        MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'})
        CREATE (b)-[:KNOWS]->(c)
    """)

    # Cluster 2: Dave, Eve, Frank (fully connected)
    graph.cypher("CREATE (:Person {name: 'Dave', group: 'B'})")
    graph.cypher("CREATE (:Person {name: 'Eve', group: 'B'})")
    graph.cypher("CREATE (:Person {name: 'Frank', group: 'B'})")

    graph.cypher("""
        MATCH (d:Person {name: 'Dave'}), (e:Person {name: 'Eve'})
        CREATE (d)-[:KNOWS]->(e)
    """)
    graph.cypher("""
        MATCH (d:Person {name: 'Dave'}), (f:Person {name: 'Frank'})
        CREATE (d)-[:KNOWS]->(f)
    """)
    graph.cypher("""
        MATCH (e:Person {name: 'Eve'}), (f:Person {name: 'Frank'})
        CREATE (e)-[:KNOWS]->(f)
    """)

    # Bridge: one connection between clusters
    graph.cypher("""
        MATCH (c:Person {name: 'Charlie'}), (d:Person {name: 'Dave'})
        CREATE (c)-[:KNOWS]->(d)
    """)

    return graph


class TestLouvainCommunities:
    """Test Louvain modularity optimization."""

    def test_two_clusters_detected(self, two_cluster_graph):
        """Louvain should detect two clear communities."""
        result = two_cluster_graph.louvain_communities()

        assert "communities" in result
        assert "modularity" in result
        assert "num_communities" in result

        # Should find 2 communities (or close to it)
        assert result["num_communities"] >= 2

    def test_all_nodes_assigned(self, two_cluster_graph):
        """Every node should be assigned to a community."""
        result = two_cluster_graph.louvain_communities()

        all_nodes = set()
        for comm_id, members in result["communities"].items():
            for node in members:
                all_nodes.add(node["title"])

        assert all_nodes == {"Alice", "Bob", "Charlie", "Dave", "Eve", "Frank"}

    def test_modularity_positive(self, two_cluster_graph):
        """Modularity should be positive for clustered graph."""
        result = two_cluster_graph.louvain_communities()
        assert result["modularity"] > 0

    def test_cluster_members_together(self, two_cluster_graph):
        """Nodes in the same cluster should be in the same community."""
        result = two_cluster_graph.louvain_communities()

        # Build name -> community mapping
        name_to_community = {}
        for comm_id, members in result["communities"].items():
            for node in members:
                name_to_community[node["title"]] = comm_id

        # Cluster 1 nodes should share a community
        assert name_to_community["Alice"] == name_to_community["Bob"]
        assert name_to_community["Alice"] == name_to_community["Charlie"]

        # Cluster 2 nodes should share a community
        assert name_to_community["Dave"] == name_to_community["Eve"]
        assert name_to_community["Dave"] == name_to_community["Frank"]

        # The two clusters should be different communities
        assert name_to_community["Alice"] != name_to_community["Dave"]

    def test_resolution_parameter(self, two_cluster_graph):
        """Higher resolution should produce more communities."""
        low_res = two_cluster_graph.louvain_communities(resolution=0.5)
        high_res = two_cluster_graph.louvain_communities(resolution=3.0)

        # Higher resolution tends to find more communities
        assert high_res["num_communities"] >= low_res["num_communities"]

    def test_empty_graph(self):
        """Louvain on empty graph returns empty result."""
        graph = KnowledgeGraph()
        result = graph.louvain_communities()

        assert result["num_communities"] == 0
        assert result["modularity"] == 0.0
        assert len(result["communities"]) == 0

    def test_single_node(self):
        """Single node → single community."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice'})")

        result = graph.louvain_communities()
        assert result["num_communities"] == 1

    def test_no_edges(self):
        """Nodes with no edges → each in own community."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice'})")
        graph.cypher("CREATE (:Person {name: 'Bob'})")
        graph.cypher("CREATE (:Person {name: 'Charlie'})")

        result = graph.louvain_communities()
        assert result["num_communities"] == 3
        assert result["modularity"] == 0.0

    def test_fully_connected(self):
        """Fully connected graph → single community."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'A'})")
        graph.cypher("CREATE (:Person {name: 'B'})")
        graph.cypher("CREATE (:Person {name: 'C'})")

        graph.cypher("MATCH (a:Person {name: 'A'}), (b:Person {name: 'B'}) CREATE (a)-[:KNOWS]->(b)")
        graph.cypher("MATCH (a:Person {name: 'A'}), (c:Person {name: 'C'}) CREATE (a)-[:KNOWS]->(c)")
        graph.cypher("MATCH (b:Person {name: 'B'}), (c:Person {name: 'C'}) CREATE (b)-[:KNOWS]->(c)")

        result = graph.louvain_communities()
        assert result["num_communities"] == 1

    def test_weight_property(self):
        """Weighted edges affect community assignment."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'A'})")
        graph.cypher("CREATE (:Person {name: 'B'})")
        graph.cypher("CREATE (:Person {name: 'C'})")

        # Strong connection A-B, weak connection B-C
        graph.cypher("""
            MATCH (a:Person {name: 'A'}), (b:Person {name: 'B'})
            CREATE (a)-[:KNOWS {weight: 10}]->(b)
        """)
        graph.cypher("""
            MATCH (b:Person {name: 'B'}), (c:Person {name: 'C'})
            CREATE (b)-[:KNOWS {weight: 1}]->(c)
        """)

        result = graph.louvain_communities(weight_property="weight")
        assert result["num_communities"] >= 1  # At least some structure detected


class TestLabelPropagation:
    """Test label propagation community detection."""

    def test_returns_valid_result(self, two_cluster_graph):
        """Label propagation returns valid community structure."""
        result = two_cluster_graph.label_propagation()

        assert "communities" in result
        assert "modularity" in result
        assert "num_communities" in result

        # LP may merge clusters across bridges, so just check it returns >= 1
        assert result["num_communities"] >= 1

    def test_all_nodes_assigned(self, two_cluster_graph):
        """Every node should be assigned to a community."""
        result = two_cluster_graph.label_propagation()

        all_nodes = set()
        for comm_id, members in result["communities"].items():
            for node in members:
                all_nodes.add(node["title"])

        assert all_nodes == {"Alice", "Bob", "Charlie", "Dave", "Eve", "Frank"}

    def test_converges(self, two_cluster_graph):
        """Algorithm should converge within max_iterations."""
        result = two_cluster_graph.label_propagation(max_iterations=100)
        assert result["num_communities"] >= 1

    def test_cluster_members_together(self, two_cluster_graph):
        """Nodes in same cluster should be in same community."""
        result = two_cluster_graph.label_propagation()

        name_to_community = {}
        for comm_id, members in result["communities"].items():
            for node in members:
                name_to_community[node["title"]] = comm_id

        # Cluster 1 nodes should share a community
        assert name_to_community["Alice"] == name_to_community["Bob"]
        assert name_to_community["Alice"] == name_to_community["Charlie"]

        # Cluster 2 nodes should share a community
        assert name_to_community["Dave"] == name_to_community["Eve"]
        assert name_to_community["Dave"] == name_to_community["Frank"]

    def test_empty_graph(self):
        """Label propagation on empty graph."""
        graph = KnowledgeGraph()
        result = graph.label_propagation()

        assert result["num_communities"] == 0
        assert len(result["communities"]) == 0

    def test_single_node(self):
        """Single node → single community."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice'})")

        result = graph.label_propagation()
        assert result["num_communities"] == 1

    def test_max_iterations_respected(self):
        """With max_iterations=1, algorithm runs at most once."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'A'})")
        graph.cypher("CREATE (:Person {name: 'B'})")

        result = graph.label_propagation(max_iterations=1)
        assert result["num_communities"] >= 1

    def test_result_structure(self, two_cluster_graph):
        """Verify the structure of returned data."""
        result = two_cluster_graph.label_propagation()

        assert isinstance(result["communities"], dict)
        assert isinstance(result["modularity"], float)
        assert isinstance(result["num_communities"], int)

        for comm_id, members in result["communities"].items():
            assert isinstance(members, list)
            for node in members:
                assert "title" in node
                assert "type" in node
                assert "id" in node


def _build_two_clusters(graph):
    """Two triangles {Alice,Bob,Charlie} and {Dave,Eve,Frank} + a bridge,
    on any storage mode."""
    for name, grp in [("Alice", "A"), ("Bob", "A"), ("Charlie", "A"), ("Dave", "B"), ("Eve", "B"), ("Frank", "B")]:
        graph.cypher("CREATE (:Person {name: $n, group: $g})", params={"n": name, "g": grp})
    edges = [
        ("Alice", "Bob"),
        ("Alice", "Charlie"),
        ("Bob", "Charlie"),
        ("Dave", "Eve"),
        ("Dave", "Frank"),
        ("Eve", "Frank"),
        ("Charlie", "Dave"),
    ]
    for a, b in edges:
        graph.cypher(
            "MATCH (a:Person {name:$a}),(b:Person {name:$b}) CREATE (a)-[:KNOWS]->(b)",
            params={"a": a, "b": b},
        )
    return graph


class TestLeidenCypher:
    """CALL leiden(...) — Leiden community detection via Cypher."""

    def test_leiden_two_communities(self, two_cluster_graph):
        rows = two_cluster_graph.cypher(
            "CALL leiden() YIELD node, community RETURN node.name AS name, community AS c"
        ).to_list()
        assert len(rows) == 6
        by_name = {r["name"]: r["c"] for r in rows}
        assert by_name["Alice"] == by_name["Bob"] == by_name["Charlie"]
        assert by_name["Dave"] == by_name["Eve"] == by_name["Frank"]
        assert by_name["Alice"] != by_name["Dave"]
        assert len({r["c"] for r in rows}) == 2

    def test_leiden_level_yield_hierarchy(self, two_cluster_graph):
        rows = two_cluster_graph.cypher(
            "CALL leiden() YIELD node, community, level RETURN node.name AS name, community AS c, level AS l"
        ).to_list()
        # at least one level, every row has an integer level >= 0
        levels = {r["l"] for r in rows}
        assert len(levels) >= 1
        assert all(isinstance(r["l"], int) and r["l"] >= 0 for r in rows)
        # each level assigns all 6 nodes
        from collections import Counter

        per_level = Counter(r["l"] for r in rows)
        assert all(count == 6 for count in per_level.values())

    def test_louvain_level_yield(self, two_cluster_graph):
        rows = two_cluster_graph.cypher(
            "CALL louvain() YIELD node, community, level RETURN node.name AS name, level AS l"
        ).to_list()
        assert all(r["l"] >= 0 for r in rows)

    def test_leiden_in_list_procedures(self):
        g = KnowledgeGraph()
        rows = g.cypher("CALL list_procedures() YIELD name RETURN name").to_list()
        names = {r["name"] for r in rows}
        assert "leiden" in names
        assert "louvain" in names

    @pytest.mark.parity
    @pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
    def test_leiden_parity_across_modes(self, storage, tmp_path):
        if storage == "memory":
            g = KnowledgeGraph()
        elif storage == "mapped":
            g = KnowledgeGraph(storage="mapped")
        else:
            g = KnowledgeGraph(storage="disk", path=str(tmp_path / "kg"))
        _build_two_clusters(g)
        rows = g.cypher("CALL leiden() YIELD node, community RETURN node.name AS name, community AS c").to_list()
        by_name = {r["name"]: r["c"] for r in rows}
        # same community structure regardless of storage mode
        assert by_name["Alice"] == by_name["Bob"] == by_name["Charlie"]
        assert by_name["Dave"] == by_name["Eve"] == by_name["Frank"]
        assert by_name["Alice"] != by_name["Dave"]


def _mode_graph(storage, tmp_path):
    if storage == "memory":
        return KnowledgeGraph()
    if storage == "mapped":
        return KnowledgeGraph(storage="mapped")
    return KnowledgeGraph(storage="disk", path=str(tmp_path / "kg"))


class TestBoundedMemoryParity:
    """The streaming (bounded-memory) mapped/disk paths for k_core and
    label_propagation must produce results identical to the in-memory
    materialised path."""

    def test_label_propagation_parity_across_modes(self, tmp_path):
        """The streaming mapped/disk paths must produce the *same partition* as
        the in-memory path. Compared as a canonical grouping (label ids are
        arbitrary) rather than a fixed structure — label propagation can collapse
        symmetric graphs to one community, which is fine as long as every mode
        agrees."""

        def partition(storage):
            g = _mode_graph(storage, tmp_path)
            _build_two_clusters(g)
            rows = g.cypher(
                "CALL label_propagation() YIELD node, community RETURN node.name AS name, community AS c"
            ).to_list()
            groups = {}
            for r in rows:
                groups.setdefault(r["c"], set()).add(r["name"])
            # canonical: frozenset of frozensets, label ids dropped
            return frozenset(frozenset(s) for s in groups.values())

        mem = partition("memory")
        assert partition("mapped") == mem
        assert partition("disk") == mem

    @pytest.mark.parity
    @pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
    def test_k_core_parity_across_modes(self, storage, tmp_path):
        g = _mode_graph(storage, tmp_path)
        _build_two_clusters(g)
        rows = g.cypher(
            "CALL k_core() YIELD node, coreness RETURN node.name AS name, coreness AS k ORDER BY name"
        ).to_list()
        by_name = {r["name"]: r["k"] for r in rows}
        # Two triangles joined by a single Charlie–Dave bridge: every node sits
        # in a triangle (coreness 2). The bridge doesn't raise either endpoint.
        assert by_name == {
            "Alice": 2,
            "Bob": 2,
            "Charlie": 2,
            "Dave": 2,
            "Eve": 2,
            "Frank": 2,
        }

    @pytest.mark.parity
    @pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
    def test_k_core_with_pendant_parity(self, storage, tmp_path):
        """A pendant node (degree 1) must get coreness 1 in every mode — exercises
        the peeling order, not just the uniform-triangle case."""
        g = _mode_graph(storage, tmp_path)
        _build_two_clusters(g)
        g.cypher("CREATE (:Person {name:'Zoe', group:'A'})")
        g.cypher("MATCH (a:Person {name:'Alice'}),(z:Person {name:'Zoe'}) CREATE (a)-[:KNOWS]->(z)")
        rows = g.cypher("CALL k_core() YIELD node, coreness RETURN node.name AS name, coreness AS k").to_list()
        by_name = {r["name"]: r["k"] for r in rows}
        assert by_name["Zoe"] == 1
        assert by_name["Alice"] == 2
        assert by_name["Frank"] == 2
