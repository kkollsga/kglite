"""Tests for graph algorithms: shortest path, all paths, connected components, centrality."""

import pandas as pd
import pytest

from kglite import KgError, KnowledgeGraph


class TestShortestPath:
    def test_shortest_path_basic(self, social_graph):
        path = social_graph.shortest_path(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=5,
        )
        assert path is not None
        assert len(path) >= 2

    def test_shortest_path_length(self, social_graph):
        length = social_graph.shortest_path_length(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=5,
        )
        assert length >= 1

    def test_shortest_path_ids(self, social_graph):
        ids = social_graph.shortest_path_ids(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=5,
        )
        assert len(ids) >= 2

    def test_shortest_path_indices(self, social_graph):
        indices = social_graph.shortest_path_indices(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=5,
        )
        assert len(indices) >= 2

    def test_shortest_path_not_found(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")
        # No connections, no path
        path = graph.shortest_path(
            source_type="Node",
            source_id=1,
            target_type="Node",
            target_id=2,
        )
        assert path is None or len(path) == 0

    def test_shortest_path_same_node(self, social_graph):
        path = social_graph.shortest_path(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=1,
        )
        # Path to self is either length 0 or 1 (just the node)
        assert path is not None


class TestAllPaths:
    def test_all_paths_basic(self, social_graph):
        paths = social_graph.all_paths(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=5,
            max_hops=4,
        )
        assert len(paths) >= 1

    def test_all_paths_limited_hops(self, small_graph):
        paths = small_graph.all_paths(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=3,
            max_hops=1,
        )
        # Direct path Alice->Charlie exists
        assert len(paths) >= 1


class TestConnectedComponents:
    def test_connected_components(self, social_graph):
        components = social_graph.connected_components()
        assert len(components) >= 1

    def test_are_connected(self, social_graph):
        result = social_graph.are_connected(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=5,
        )
        assert result is True

    def test_are_not_connected(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")
        result = graph.are_connected(
            source_type="Node",
            source_id=1,
            target_type="Node",
            target_id=2,
        )
        assert result is False


class TestCentrality:
    def test_betweenness_centrality(self, social_graph):
        result = social_graph.betweenness_centrality()
        assert result is not None
        assert len(result) > 0

    def test_degree_centrality(self, social_graph):
        result = social_graph.degree_centrality()
        assert result is not None

    def test_degrees(self, social_graph):
        # degrees needs a selection — use type_filter first
        degrees = social_graph.select("Person").degrees()
        assert degrees is not None
        assert isinstance(degrees, dict)
        assert len(degrees) > 0

    def test_pagerank(self, social_graph):
        result = social_graph.pagerank()
        assert result is not None
        assert len(result) > 0

    def test_pagerank_directed_ranking(self):
        """PageRank on a directed chain A->B->C: C should rank highest (most indirectly linked)."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Node {name: 'A'})")
        g.cypher("CREATE (:Node {name: 'B'})")
        g.cypher("CREATE (:Node {name: 'C'})")
        g.cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:LINK]->(b)")
        g.cypher("MATCH (b:Node {name: 'B'}), (c:Node {name: 'C'}) CREATE (b)-[:LINK]->(c)")

        result = g.pagerank()
        scores = {r["title"]: r["score"] for r in result}
        # In directed PageRank: C receives rank from B, B receives from A, A is a dangling start
        # C should have highest score, A lowest
        assert scores["C"] > scores["B"]
        assert scores["B"] > scores["A"]

    def test_closeness_centrality(self, social_graph):
        result = social_graph.closeness_centrality()
        assert result is not None

    @pytest.mark.parametrize("method_name", ["betweenness_centrality", "closeness_centrality"])
    def test_zero_sample_size_is_rejected(self, social_graph, method_name):
        with pytest.raises(KgError, match="sample_size must be greater than 0"):
            getattr(social_graph, method_name)(sample_size=0)

    @pytest.mark.parametrize(
        "method_name",
        [
            "betweenness_centrality",
            "pagerank",
            "degree_centrality",
            "closeness_centrality",
        ],
    )
    def test_as_dict_kwarg_is_rejected(self, social_graph, method_name):
        """`as_dict` was removed: it keyed by bare node id and silently dropped
        rows whose id collided across node types. Callers build the mapping
        themselves from the ResultView, which carries `type` alongside `id`."""
        with pytest.raises(TypeError):
            getattr(social_graph, method_name)(as_dict=True)


class TestDegreesTitleCollision:
    """`degrees()` keys its dict by node TITLE, and titles are not unique —
    not even within one type. Duplicate titles used to silently overwrite one
    another (fewer rows out than nodes in, with no signal); the call now
    refuses and points at the id-keyed `degree_centrality()`."""

    @staticmethod
    def _duplicate_title_graph():
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Alice", "Bob"]}),
            "Person",
            "id",
            "name",
        )
        g.add_connections(
            pd.DataFrame([{"src": 1, "tgt": 3}, {"src": 2, "tgt": 3}]),
            "KNOWS",
            "Person",
            "src",
            "Person",
            "tgt",
        )
        return g

    def test_duplicate_titles_raise(self):
        g = self._duplicate_title_graph()
        with pytest.raises(KgError, match="sharing title"):
            g.select("Person").degrees()

    def test_error_names_degree_centrality(self):
        g = self._duplicate_title_graph()
        with pytest.raises(KgError) as excinfo:
            g.select("Person").degrees()
        message = str(excinfo.value)
        assert "degrees()" in message
        assert "degree_centrality()" in message

    def test_degree_centrality_is_the_working_drop_in(self):
        """The recipe the error names returns a row per node, not per title."""
        g = self._duplicate_title_graph()
        rows = g.select("Person").degree_centrality().to_dicts()
        assert sorted(r["id"] for r in rows) == [1, 2, 3]

    def test_unique_titles_unchanged(self):
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Cleo"]}),
            "Person",
            "id",
            "name",
        )
        g.add_connections(
            pd.DataFrame([{"src": 1, "tgt": 2}, {"src": 2, "tgt": 3}]),
            "KNOWS",
            "Person",
            "src",
            "Person",
            "tgt",
        )
        assert g.select("Person").degrees() == {"Alice": 1, "Bob": 2, "Cleo": 1}


CENTRALITIES = [
    "betweenness_centrality",
    "pagerank",
    "degree_centrality",
    "closeness_centrality",
]


def _two_reltype_graph():
    """Two relationship types over the same nodes, so a `connection_types`
    filter is observable: KNOWS is a Person chain, WORKS_AT is a star into
    one Company. Restricting to either one changes every centrality score."""
    g = KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1, 2, 3, 4], "name": ["Alice", "Bob", "Cleo", "Dan"]}),
        "Person",
        "id",
        "name",
    )
    g.add_nodes(pd.DataFrame({"id": [100], "name": ["Acme"]}), "Company", "id", "name")
    g.add_connections(
        pd.DataFrame([{"src": 1, "tgt": 2}, {"src": 2, "tgt": 3}, {"src": 3, "tgt": 4}]),
        "KNOWS",
        "Person",
        "src",
        "Person",
        "tgt",
    )
    g.add_connections(
        pd.DataFrame([{"src": i, "tgt": 100} for i in (1, 2, 3, 4)]),
        "WORKS_AT",
        "Person",
        "src",
        "Company",
        "tgt",
    )
    return g


def _scores(view):
    return {(row["type"], row["id"]): round(row["score"], 10) for row in view.to_dicts()}


class TestCentralityConnectionTypes:
    """`connection_types` is documented as ``str | list[str]`` in the stub
    (and the Cypher twin accepts a bare scalar), but the four centralities
    only ever extracted a list — a bare string raised
    ``TypeError: Can't extract 'str' to 'Vec'``."""

    @pytest.fixture
    def graph(self):
        return _two_reltype_graph()

    @pytest.mark.parametrize("method_name", CENTRALITIES)
    def test_string_matches_single_element_list(self, graph, method_name):
        method = getattr(graph, method_name)
        assert _scores(method(connection_types="KNOWS")) == _scores(method(connection_types=["KNOWS"]))

    @pytest.mark.parametrize("method_name", CENTRALITIES)
    def test_filter_is_observable(self, graph, method_name):
        """The two relationship types give different scores, so a passing
        string form cannot be a no-op filter that silently ignores the arg."""
        method = getattr(graph, method_name)
        knows = _scores(method(connection_types="KNOWS"))
        works_at = _scores(method(connection_types="WORKS_AT"))
        assert knows != works_at

    @pytest.mark.parametrize("method_name", CENTRALITIES)
    def test_wrong_type_names_the_parameter(self, graph, method_name):
        with pytest.raises(KgError, match="connection_types must be a string or list of strings"):
            getattr(graph, method_name)(connection_types=123)

    @pytest.mark.parametrize("method_name", CENTRALITIES)
    def test_empty_list_is_no_filter(self, graph, method_name):
        method = getattr(graph, method_name)
        assert _scores(method(connection_types=[])) == _scores(method())
