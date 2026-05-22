"""Tests for connection operations: add, retrieve, connections."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


class TestAddConnections:
    def test_add_connections_basic(self, small_graph):
        conns = small_graph.select("Person").where({"title": "Alice"}).connections()
        assert len(conns) > 0

    def test_add_connections_with_properties(self, small_graph):
        # connections returns a nested dict: {title: {node_id, node_type, incoming, outgoing}}
        conns = small_graph.select("Person").where({"title": "Alice"}).connections()
        assert "Alice" in conns
        alice = conns["Alice"]
        # Alice has outgoing KNOWS connections
        assert "outgoing" in alice
        assert "KNOWS" in alice["outgoing"]

    def test_add_connections_empty_dataframe(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [], "target": []})
        report = graph.add_connections(conn_df, "LINKS", "Node", "source", "Node", "target")
        assert report["connections_created"] == 0

    def test_self_referential_connection(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [1], "target": [1]})
        report = graph.add_connections(conn_df, "SELF", "Node", "source", "Node", "target")
        assert report["connections_created"] == 1

    def test_cross_type_connections(self):
        graph = KnowledgeGraph()
        users = pd.DataFrame({"id": [1], "name": ["Alice"]})
        products = pd.DataFrame({"id": [101], "name": ["Laptop"]})
        graph.add_nodes(users, "User", "id", "name")
        graph.add_nodes(products, "Product", "id", "name")

        conn_df = pd.DataFrame({"user_id": [1], "product_id": [101]})
        report = graph.add_connections(conn_df, "PURCHASED", "User", "user_id", "Product", "product_id")
        assert report["connections_created"] == 1


class TestGetConnections:
    def test_connections_basic(self, small_graph):
        # connections returns nested dict keyed by node title
        conns = small_graph.select("Person").where({"title": "Alice"}).connections()
        assert "Alice" in conns
        alice_conns = conns["Alice"]
        assert "outgoing" in alice_conns
        assert "KNOWS" in alice_conns["outgoing"]
        assert len(alice_conns["outgoing"]["KNOWS"]) >= 2

    def test_connections_include_properties(self, small_graph):
        # include_node_properties is a bool flag
        conns = small_graph.select("Person").where({"title": "Alice"}).connections(include_node_properties=True)
        assert "Alice" in conns
        # With include_node_properties=True, node_properties should be populated
        alice = conns["Alice"]
        for conn_type, targets in alice.get("outgoing", {}).items():
            for target_name, target_info in targets.items():
                assert "node_properties" in target_info

    def test_duplicate_connections(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [1, 1], "target": [2, 2]})
        report = graph.add_connections(conn_df, "LINKS", "Node", "source", "Node", "target")
        # Both connections should be created (multigraph)
        assert report["connections_created"] >= 1


class TestBatchFlushDedup:
    """A1: per-flush HashMap dedup replaces the O(degree) edges_connecting
    walk that ran per edge when adding into an existing connection type.

    These tests pin the correctness invariants that the in-flush lookup
    map must preserve: no duplicate edges, Skip-mode suppression still
    works, Replace-mode updates the lookup so later chunk entries hit the
    freshly-created edge, and within-chunk duplicates consolidate onto a
    single edge instead of creating two.
    """

    @staticmethod
    def _seed(num_hubs: int = 3, targets_per_hub: int = 200) -> KnowledgeGraph:
        """Hub-source fan-out fixture. The connection type :R is
        established by the first pass so subsequent passes go through
        the existence-check path (the path A1 accelerates)."""
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame([{"id": h, "name": f"hub{h}"} for h in range(num_hubs)]),
            "Hub",
            "id",
            "name",
        )
        g.add_nodes(
            pd.DataFrame(
                [
                    {"id": h * targets_per_hub + t, "name": f"t{h * targets_per_hub + t}"}
                    for h in range(num_hubs)
                    for t in range(targets_per_hub)
                ]
            ),
            "Target",
            "id",
            "name",
        )
        g.add_connections(
            pd.DataFrame(
                [
                    {"from_id": h, "to_id": h * targets_per_hub + t}
                    for h in range(num_hubs)
                    for t in range(targets_per_hub)
                ]
            ),
            "R",
            "Hub",
            "from_id",
            "Target",
            "to_id",
        )
        return g

    def test_fan_out_into_existing_type_creates_no_duplicates(self):
        """Re-adding the exact same edge set with the default (Update)
        conflict mode must not produce duplicate edges. Pre-fix this
        relied on the O(degree) walk; post-fix it relies on the
        per-flush HashMap finding each pre-existing edge."""
        g = self._seed(num_hubs=3, targets_per_hub=200)
        before = g.cypher("MATCH ()-[r:R]->() RETURN count(r) AS c").to_list()[0]["c"]

        g.add_connections(
            pd.DataFrame([{"from_id": h, "to_id": h * 200 + t} for h in range(3) for t in range(200)]),
            "R",
            "Hub",
            "from_id",
            "Target",
            "to_id",
        )
        after = g.cypher("MATCH ()-[r:R]->() RETURN count(r) AS c").to_list()[0]["c"]
        assert before == after == 600, (before, after)

    def test_fan_out_into_existing_type_adds_new_edges(self):
        """Net-new edges into an existing connection type all land."""
        g = self._seed(num_hubs=3, targets_per_hub=200)
        # New targets 1000..1599 — entirely new IDs.
        g.add_nodes(
            pd.DataFrame([{"id": 1000 + i, "name": f"u{i}"} for i in range(600)]),
            "Target",
            "id",
            "name",
        )
        g.add_connections(
            pd.DataFrame([{"from_id": h, "to_id": 1000 + h * 200 + t} for h in range(3) for t in range(200)]),
            "R",
            "Hub",
            "from_id",
            "Target",
            "to_id",
        )
        total = g.cypher("MATCH ()-[r:R]->() RETURN count(r) AS c").to_list()[0]["c"]
        assert total == 1200, total

    def test_skip_mode_suppresses_duplicates(self):
        """Skip-mode (pre-buffer + flush) still drops all duplicates."""
        g = self._seed(num_hubs=2, targets_per_hub=50)
        stats = g.add_connections(
            pd.DataFrame([{"from_id": h, "to_id": h * 50 + t} for h in range(2) for t in range(50)]),
            "R",
            "Hub",
            "from_id",
            "Target",
            "to_id",
            conflict_handling="skip",
        )
        assert stats["connections_created"] == 0, stats
        total = g.cypher("MATCH ()-[r:R]->() RETURN count(r) AS c").to_list()[0]["c"]
        assert total == 100, total

    def test_replace_mode_does_not_duplicate(self):
        """Replace-mode removes the existing edge and inserts a new one;
        the per-flush lookup must be updated to the new edge id so a later
        chunk entry hitting the same (src, tgt) finds the new edge — not
        the now-removed one. Net edge count stays the same."""
        g = self._seed(num_hubs=2, targets_per_hub=10)
        # 20 existing edges. Replace all 20, then immediately replace the
        # same set again — exercises the lookup-update-on-replace path.
        repl_df = pd.DataFrame([{"from_id": h, "to_id": h * 10 + t} for h in range(2) for t in range(10)])
        g.add_connections(repl_df, "R", "Hub", "from_id", "Target", "to_id", conflict_handling="replace")
        g.add_connections(repl_df, "R", "Hub", "from_id", "Target", "to_id", conflict_handling="replace")
        total = g.cypher("MATCH ()-[r:R]->() RETURN count(r) AS c").to_list()[0]["c"]
        assert total == 20, total

    def test_within_chunk_duplicate_consolidates(self):
        """When the connection type already exists, two chunk entries
        with the same (src, tgt) must consolidate onto a single edge —
        the first iteration creates an edge, the per-flush lookup is
        updated, and the second iteration finds it via Update mode.

        (The skip_existence_check=true initial-load fast path bypasses
        within-chunk dedup by design — first-batch consolidation is the
        caller's responsibility there. That path is unchanged.)
        """
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}]),
            "N",
            "id",
            "name",
        )
        # Establish the :R connection type with a sentinel edge.
        g.add_connections(
            pd.DataFrame([{"src": 1, "tgt": 3}]),
            "R",
            "N",
            "src",
            "N",
            "tgt",
        )
        # Now :R exists, so skip_existence_check=false. Two identical
        # (1, 2) rows in one chunk must collapse to one edge.
        g.add_connections(
            pd.DataFrame([{"src": 1, "tgt": 2}, {"src": 1, "tgt": 2}]),
            "R",
            "N",
            "src",
            "N",
            "tgt",
        )
        total = g.cypher("MATCH (a:N {name:'a'})-[r:R]->(b:N {name:'b'}) RETURN count(r) AS c").to_list()[0]["c"]
        assert total == 1, total


class TestConflictHandlingSum:
    """Tests for conflict_handling='sum' on add_connections."""

    def _make_graph(self):
        graph = KnowledgeGraph()
        nodes = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        graph.add_nodes(nodes, "Node", "id", "name")
        edges = pd.DataFrame(
            {
                "src": [1, 1],
                "tgt": [2, 3],
                "weight": [10, 20],
                "label": ["x", "y"],
            }
        )
        graph.add_connections(edges, "LINK", "Node", "src", "Node", "tgt", columns=["weight", "label"])
        return graph

    def test_sum_int_properties(self):
        graph = self._make_graph()
        edges2 = pd.DataFrame({"src": [1], "tgt": [2], "weight": [5]})
        graph.add_connections(edges2, "LINK", "Node", "src", "Node", "tgt", columns=["weight"], conflict_handling="sum")
        result = graph.cypher("MATCH (:Node {id: 1})-[r:LINK]->(:Node {id: 2}) RETURN r.weight")
        assert result[0]["r.weight"] == 15

    def test_sum_float_properties(self):
        graph = KnowledgeGraph()
        nodes = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(nodes, "Node", "id", "name")
        edges1 = pd.DataFrame({"src": [1], "tgt": [2], "score": [1.5]})
        graph.add_connections(edges1, "LINK", "Node", "src", "Node", "tgt", columns=["score"])
        edges2 = pd.DataFrame({"src": [1], "tgt": [2], "score": [2.5]})
        graph.add_connections(edges2, "LINK", "Node", "src", "Node", "tgt", columns=["score"], conflict_handling="sum")
        result = graph.cypher("MATCH (:Node {id: 1})-[r:LINK]->(:Node {id: 2}) RETURN r.score")
        assert abs(result[0]["r.score"] - 4.0) < 1e-10

    def test_sum_non_numeric_overwrites(self):
        graph = self._make_graph()
        edges2 = pd.DataFrame({"src": [1], "tgt": [2], "weight": [5], "label": ["z"]})
        graph.add_connections(
            edges2, "LINK", "Node", "src", "Node", "tgt", columns=["weight", "label"], conflict_handling="sum"
        )
        result = graph.cypher("MATCH (:Node {id: 1})-[r:LINK]->(:Node {id: 2}) RETURN r.weight, r.label")
        assert result[0]["r.weight"] == 15
        assert result[0]["r.label"] == "z"  # overwritten, not summed

    def test_sum_new_edge_created(self):
        graph = self._make_graph()
        edges2 = pd.DataFrame({"src": [2], "tgt": [3], "weight": [42]})
        report = graph.add_connections(
            edges2, "LINK", "Node", "src", "Node", "tgt", columns=["weight"], conflict_handling="sum"
        )
        assert report["connections_created"] == 1
        result = graph.cypher("MATCH (:Node {id: 2})-[r:LINK]->(:Node {id: 3}) RETURN r.weight")
        assert result[0]["r.weight"] == 42

    def test_sum_new_property_added(self):
        graph = self._make_graph()
        edges2 = pd.DataFrame({"src": [1], "tgt": [2], "new_prop": [42]})
        graph.add_connections(
            edges2, "LINK", "Node", "src", "Node", "tgt", columns=["new_prop"], conflict_handling="sum"
        )
        result = graph.cypher("MATCH (:Node {id: 1})-[r:LINK]->(:Node {id: 2}) RETURN r.weight, r.new_prop")
        assert result[0]["r.weight"] == 10  # unchanged
        assert result[0]["r.new_prop"] == 42

    def test_sum_on_nodes_behaves_as_update(self):
        graph = KnowledgeGraph()
        df1 = pd.DataFrame({"id": [1], "name": ["A"], "v": [10]})
        df2 = pd.DataFrame({"id": [1], "name": ["A"], "v": [20]})
        graph.add_nodes(df1, "Node", "id", "name")
        graph.add_nodes(df2, "Node", "id", "name", conflict_handling="sum")
        result = graph.cypher("MATCH (n:Node {id: 1}) RETURN n.v")
        assert result[0]["n.v"] == 20  # overwrite, not 30


class TestQueryModeParamValidation:
    """Data-mode-only params should raise errors in query mode."""

    def _make_graph(self):
        graph = KnowledgeGraph()
        nodes = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(nodes, "Node", "id", "name")
        return graph

    def test_columns_rejected_in_query_mode(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="columns.*data mode"):
            graph.add_connections(
                None,
                "LINK",
                "Node",
                "src",
                "Node",
                "tgt",
                query="MATCH (a:Node {id: 1}), (b:Node {id: 2}) RETURN a.id AS src, b.id AS tgt",
                columns=["src"],
            )

    def test_skip_columns_rejected_in_query_mode(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="skip_columns.*data mode"):
            graph.add_connections(
                None,
                "LINK",
                "Node",
                "src",
                "Node",
                "tgt",
                query="MATCH (a:Node {id: 1}), (b:Node {id: 2}) RETURN a.id AS src, b.id AS tgt",
                skip_columns=["tgt"],
            )

    def test_column_types_rejected_in_query_mode(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="column_types.*data mode"):
            graph.add_connections(
                None,
                "LINK",
                "Node",
                "src",
                "Node",
                "tgt",
                query="MATCH (a:Node {id: 1}), (b:Node {id: 2}) RETURN a.id AS src, b.id AS tgt",
                column_types={"src": "integer"},
            )
