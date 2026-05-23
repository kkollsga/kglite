"""Tests for CALL cluster() — general-purpose clustering, plus round(x,d) and || operator."""

import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture
def spatial_graph():
    """Graph with spatial nodes in two clear geographic clusters."""
    g = KnowledgeGraph()

    # Configure spatial
    g.set_spatial("City", location=("lat", "lon"))

    # Cluster 1: Nordic cities (lat ~59-60, lon ~5-11)
    g.cypher("CREATE (:City {name: 'Oslo', lat: 59.91, lon: 10.75})")
    g.cypher("CREATE (:City {name: 'Bergen', lat: 60.39, lon: 5.32})")
    g.cypher("CREATE (:City {name: 'Stavanger', lat: 58.97, lon: 5.73})")

    # Cluster 2: Mediterranean cities (lat ~37-41, lon ~12-24)
    g.cypher("CREATE (:City {name: 'Rome', lat: 41.90, lon: 12.50})")
    g.cypher("CREATE (:City {name: 'Athens', lat: 37.98, lon: 23.73})")
    g.cypher("CREATE (:City {name: 'Naples', lat: 40.85, lon: 14.27})")

    # Outlier
    g.cypher("CREATE (:City {name: 'Reykjavik', lat: 64.13, lon: -21.90})")

    return g


@pytest.fixture
def property_graph():
    """Graph with numeric properties for property-based clustering."""
    g = KnowledgeGraph()

    # Cluster 1: deep, hot wells
    g.cypher("CREATE (:Well {name: 'W1', depth: 4000.0, temperature: 150.0})")
    g.cypher("CREATE (:Well {name: 'W2', depth: 4200.0, temperature: 155.0})")
    g.cypher("CREATE (:Well {name: 'W3', depth: 3800.0, temperature: 145.0})")

    # Cluster 2: shallow, cool wells
    g.cypher("CREATE (:Well {name: 'W4', depth: 500.0, temperature: 30.0})")
    g.cypher("CREATE (:Well {name: 'W5', depth: 600.0, temperature: 35.0})")
    g.cypher("CREATE (:Well {name: 'W6', depth: 450.0, temperature: 28.0})")

    return g


class TestClusterSpatialDbscan:
    """Test CALL cluster() with spatial auto-detection (DBSCAN)."""

    def test_two_clusters_plus_noise(self, spatial_graph):
        # eps=1200km groups Med cities (Rome-Athens ~1050km), min_points=1 for pairs
        result = spatial_graph.cypher("""
            MATCH (c:City)
            CALL cluster({method: 'dbscan', eps: 1200000, min_points: 1})
            YIELD node, cluster
            RETURN node.name AS name, cluster
            ORDER BY cluster, name
        """)
        assert len(result) == 7

        # Group by cluster
        clusters = {}
        for r in result:
            cid = r["cluster"]
            clusters.setdefault(cid, []).append(r["name"])

        # Should have 2 real clusters and possibly noise (-1)
        real_clusters = {k: v for k, v in clusters.items() if k >= 0}
        assert len(real_clusters) == 2, f"Expected 2 clusters, got {real_clusters}"

        # Nordic cities should be in one cluster, Mediterranean in another
        nordic = {"Oslo", "Bergen", "Stavanger"}
        med = {"Rome", "Athens", "Naples"}
        cluster_sets = [set(v) for v in real_clusters.values()]
        assert nordic in cluster_sets, f"Nordic cities not grouped: {cluster_sets}"
        assert med in cluster_sets, f"Mediterranean cities not grouped: {cluster_sets}"

        # Reykjavik should be noise (>1200km from both clusters)
        assert -1 in clusters, "Reykjavik should be noise"
        assert "Reykjavik" in clusters[-1]

    def test_all_one_cluster_with_large_eps(self, spatial_graph):
        result = spatial_graph.cypher("""
            MATCH (c:City)
            CALL cluster({method: 'dbscan', eps: 5000000, min_points: 1})
            YIELD node, cluster
            RETURN cluster, count(*) AS n
        """)
        # With a 5000km eps, everything should be in one cluster
        assert len(result) == 1
        assert result[0]["n"] == 7
        assert result[0]["cluster"] >= 0


class TestClusterSpatialKmeans:
    """Test CALL cluster() with spatial K-means."""

    def test_kmeans_two_groups(self, spatial_graph):
        result = spatial_graph.cypher("""
            MATCH (c:City)
            CALL cluster({method: 'kmeans', k: 2})
            YIELD node, cluster
            RETURN node.name AS name, cluster
        """)
        assert len(result) == 7

        # Group by cluster
        clusters = {}
        for r in result:
            clusters.setdefault(r["cluster"], []).append(r["name"])

        # Nordic cities (+ Reykjavik) should land in one group,
        # Mediterranean in another — k-means has no noise concept
        assert len(clusters) == 2


class TestClusterPropertyDbscan:
    """Test CALL cluster() with explicit numeric properties."""

    def test_two_property_clusters(self, property_graph):
        result = property_graph.cypher("""
            MATCH (w:Well)
            CALL cluster({
                properties: ['depth', 'temperature'],
                method: 'dbscan',
                eps: 0.3,
                min_points: 2,
                normalize: true
            })
            YIELD node, cluster
            RETURN node.name AS name, cluster
            ORDER BY cluster, name
        """)
        assert len(result) == 6

        clusters = {}
        for r in result:
            clusters.setdefault(r["cluster"], []).append(r["name"])

        real = {k: v for k, v in clusters.items() if k >= 0}
        assert len(real) == 2

        deep = {"W1", "W2", "W3"}
        shallow = {"W4", "W5", "W6"}
        cluster_sets = [set(v) for v in real.values()]
        assert deep in cluster_sets
        assert shallow in cluster_sets


class TestClusterPropertyKmeans:
    """Test CALL cluster() with K-means on explicit properties."""

    def test_kmeans_properties(self, property_graph):
        result = property_graph.cypher("""
            MATCH (w:Well)
            CALL cluster({
                properties: ['depth', 'temperature'],
                method: 'kmeans',
                k: 2,
                normalize: true
            })
            YIELD node, cluster
            RETURN node.name AS name, cluster
        """)
        assert len(result) == 6

        clusters = {}
        for r in result:
            clusters.setdefault(r["cluster"], []).append(r["name"])

        assert len(clusters) == 2

        deep = {"W1", "W2", "W3"}
        shallow = {"W4", "W5", "W6"}
        cluster_sets = [set(v) for v in clusters.values()]
        assert deep in cluster_sets
        assert shallow in cluster_sets


class TestClusterWithGeometryCentroid:
    """Test cluster() spatial mode with WKT geometry centroid fallback."""

    def test_geometry_centroid_clustering(self):
        g = KnowledgeGraph()
        g.set_spatial("Area", geometry="wkt")

        # Two areas near each other (Nordic) and one far away (Mediterranean)
        g.cypher("""CREATE (:Area {name: 'NorthSea', wkt: 'POLYGON((4 58, 8 58, 8 62, 4 62, 4 58))'})""")
        g.cypher("""CREATE (:Area {name: 'NorwegianSea', wkt: 'POLYGON((3 62, 7 62, 7 66, 3 66, 3 62))'})""")
        g.cypher("""CREATE (:Area {name: 'Mediterranean', wkt: 'POLYGON((10 35, 25 35, 25 42, 10 42, 10 35))'})""")

        result = g.cypher("""
            MATCH (a:Area)
            CALL cluster({method: 'dbscan', eps: 1000000, min_points: 1})
            YIELD node, cluster
            RETURN node.name AS name, cluster
            ORDER BY cluster
        """)
        assert len(result) == 3

        clusters = {}
        for r in result:
            clusters.setdefault(r["cluster"], []).append(r["name"])

        # NorthSea and NorwegianSea should cluster together
        real = {k: v for k, v in clusters.items() if k >= 0}
        assert len(real) >= 1
        north_cluster = None
        for cid, names in real.items():
            if "NorthSea" in names:
                north_cluster = cid
                break
        assert north_cluster is not None
        assert "NorwegianSea" in real[north_cluster]


class TestClusterErrors:
    """Test error handling for cluster()."""

    def test_no_match_gives_error(self):
        g = KnowledgeGraph()
        with pytest.raises(Exception, match="requires a preceding MATCH"):
            g.cypher("""
                CALL cluster({method: 'dbscan'}) YIELD node, cluster
                RETURN node, cluster
            """)

    def test_unknown_method(self, spatial_graph):
        with pytest.raises(Exception, match="Unknown clustering method"):
            spatial_graph.cypher("""
                MATCH (c:City)
                CALL cluster({method: 'spectral'})
                YIELD node, cluster
                RETURN node, cluster
            """)

    def test_no_spatial_config_gives_error(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Thing {name: 'A', x: 1.0})")
        g.cypher("CREATE (:Thing {name: 'B', x: 2.0})")
        with pytest.raises(Exception, match="spatial"):
            g.cypher("""
                MATCH (t:Thing)
                CALL cluster({method: 'dbscan'})
                YIELD node, cluster
                RETURN node, cluster
            """)


class TestClusterAggregation:
    """Test cluster() combined with aggregation (the dream query pattern)."""

    def test_cluster_with_count_and_collect(self, spatial_graph):
        result = spatial_graph.cypher("""
            MATCH (c:City)
            CALL cluster({method: 'dbscan', eps: 1200000, min_points: 1})
            YIELD node, cluster
            RETURN cluster, count(*) AS n, collect(node.name) AS cities
            ORDER BY n DESC
        """)
        assert len(result) >= 2  # at least 2 groups (real clusters + noise)

        # The two real clusters should each have 3 cities
        real = [r for r in result if r["cluster"] >= 0]
        assert len(real) == 2
        for r in real:
            assert r["n"] == 3


class TestRoundPrecision:
    """Test round(x, decimals) precision argument."""

    def test_round_two_decimals(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {val: 3.14159})")
        result = g.cypher("MATCH (n:N) RETURN round(n.val, 2) AS r")
        assert result[0]["r"] == 3.14

    def test_round_zero_decimals(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {val: 3.14159})")
        result = g.cypher("MATCH (n:N) RETURN round(n.val, 0) AS r")
        assert result[0]["r"] == 3.0

    def test_round_backward_compat(self):
        """round() with no precision argument should still work."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {val: 3.7})")
        result = g.cypher("MATCH (n:N) RETURN round(n.val) AS r")
        assert result[0]["r"] == 4.0

    def test_round_null_propagation(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {val: 3.14})")
        result = g.cypher("MATCH (n:N) RETURN round(n.missing, 2) AS r")
        assert result[0]["r"] is None


class TestStringConcat:
    """Test || string concatenation operator."""

    def test_basic_concat(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {a: 'hello', b: 'world'})")
        result = g.cypher("MATCH (n:N) RETURN n.a || ' ' || n.b AS msg")
        assert result[0]["msg"] == "hello world"

    def test_concat_with_number(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {name: 'block', num: 35})")
        result = g.cypher("MATCH (n:N) RETURN n.name || '-' || n.num AS label")
        assert result[0]["label"] == "block-35"

    def test_concat_null_propagation(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {name: 'test'})")
        result = g.cypher("MATCH (n:N) RETURN n.name || n.missing AS r")
        assert result[0]["r"] is None

    def test_concat_in_return(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {first: 'A', last: 'B'})")
        result = g.cypher("MATCH (n:N) RETURN n.first || n.last AS code")
        assert result[0]["code"] == "AB"


class TestDescribeCypherTiers:
    """Test 3-tier progressive Cypher documentation."""

    # -- Tier 1: hint in extensions --

    def test_tier1_cypher_hint(self):
        g = KnowledgeGraph()
        desc = g.describe()
        assert "cypher" in desc.lower()
        assert "cluster" in desc.lower()

    # -- Tier 2: overview (cypher=True) --

    def test_tier2_has_clauses_and_procedures(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=True)
        assert "<cypher>" in desc
        assert "<clauses>" in desc
        assert "MATCH" in desc
        assert "WHERE" in desc
        assert "<procedures>" in desc
        assert "cluster" in desc
        assert "pagerank" in desc

    def test_tier2_has_operators_and_functions(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=True)
        assert "<operators>" in desc
        assert "<functions>" in desc
        assert "||" in desc
        assert "CONTAINS" in desc
        assert "coalesce" in desc

    def test_tier2_no_graph_schema(self):
        """cypher=True should NOT include node types or inventory."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Field {name: 'Troll'})")
        desc = g.describe(cypher=True)
        assert "<cypher>" in desc
        # Should not have graph inventory
        assert "<graph" not in desc
        assert "Field" not in desc

    def test_tier2_has_hint_for_tier3(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=True)
        assert "graph_overview(cypher=[" in desc

    # -- Tier 3: topic detail (cypher=list) --

    def test_tier3_cluster_detail(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=["cluster"])
        assert "<cluster>" in desc
        assert "<params>" in desc
        assert "<examples>" in desc
        assert "spatial" in desc.lower()
        assert "property" in desc.lower()

    def test_tier3_match_detail(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=["MATCH"])
        assert "<MATCH>" in desc
        assert "<examples>" in desc

    def test_tier3_multiple_topics(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=["MATCH", "cluster", "WHERE"])
        assert "<MATCH>" in desc
        assert "<cluster>" in desc
        assert "<WHERE>" in desc

    def test_tier3_case_insensitive(self):
        g = KnowledgeGraph()
        desc1 = g.describe(cypher=["match"])
        desc2 = g.describe(cypher=["MATCH"])
        assert "<MATCH>" in desc1
        assert "<MATCH>" in desc2

    def test_tier3_unknown_topic_error(self):
        g = KnowledgeGraph()
        with pytest.raises(kglite.KgError, match="Unknown Cypher topic"):
            g.describe(cypher=["nonexistent"])

    def test_tier3_error_lists_available(self):
        g = KnowledgeGraph()
        try:
            g.describe(cypher=["bogus"])
        except kglite.KgError as e:
            msg = str(e)
            assert "MATCH" in msg
            assert "cluster" in msg

    def test_tier3_empty_list_gives_overview(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=[])
        assert "<clauses>" in desc
        assert "<procedures>" in desc

    def test_tier3_operators_detail(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=["operators"])
        assert "<operators>" in desc
        assert "concatenation" in desc

    def test_tier3_functions_detail(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=["functions"])
        assert "<functions>" in desc
        assert "round" in desc

    def test_tier3_pagerank_detail(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=["pagerank"])
        assert "<pagerank>" in desc
        assert "<params>" in desc
        assert "damping_factor" in desc

    def test_tier3_spatial_detail(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=["spatial"])
        assert "<spatial>" in desc
        assert "distance" in desc
        assert "contains" in desc
        assert "<examples>" in desc

    def test_tier2_has_limitations(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=True)
        assert "<limitations>" in desc
        assert "FOREACH" in desc
        assert "subqueries" in desc.lower()

    def test_tier2_has_spatial_functions(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=True)
        assert "distance" in desc
        assert "spatial" in desc.lower()

    # -- Backward compat --

    def test_cypher_false_same_as_none(self):
        g = KnowledgeGraph()
        r1 = g.describe()
        r2 = g.describe(cypher=False)
        assert r1 == r2

    def test_cypher_invalid_type_raises(self):
        g = KnowledgeGraph()
        with pytest.raises(TypeError):
            g.describe(cypher=42)

    def test_no_connections_hint_without_edges(self):
        """Connections hint should not appear in graphs with no edges."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Thing {name: 'A'})")
        desc = g.describe()
        assert "connections hint" not in desc

    def test_connections_hint_with_edges(self):
        """Connections hint should appear when graph has edges."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:A {name: 'a'})")
        g.cypher("CREATE (:B {name: 'b'})")
        g.cypher("MATCH (a:A), (b:B) CREATE (a)-[:KNOWS]->(b)")
        desc = g.describe()
        assert "connections hint" in desc or "graph_overview(connections=" in desc

    def test_overview_connection_map_has_counts(self):
        """Overview connection map should include count attribute."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:A {name: 'a1'})")
        g.cypher("CREATE (:A {name: 'a2'})")
        g.cypher("CREATE (:B {name: 'b'})")
        g.cypher("MATCH (a:A), (b:B) CREATE (a)-[:LINKS]->(b)")
        desc = g.describe()
        assert 'count="2"' in desc


class TestDescribeConnections:
    """Test describe(connections=...) for connection type progressive disclosure."""

    @pytest.fixture
    def connected_graph(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Field {name: 'Troll'})")
        g.cypher("CREATE (:Field {name: 'Ekofisk'})")
        g.cypher("CREATE (:Well {name: 'W1'})")
        g.cypher("CREATE (:Well {name: 'W2'})")
        g.cypher("CREATE (:Company {name: 'Equinor'})")
        g.cypher("""
            MATCH (w:Well {name: 'W1'}), (f:Field {name: 'Troll'})
            CREATE (w)-[:BELONGS_TO {since: 1995}]->(f)
        """)
        g.cypher("""
            MATCH (w:Well {name: 'W2'}), (f:Field {name: 'Ekofisk'})
            CREATE (w)-[:BELONGS_TO {since: 2001}]->(f)
        """)
        g.cypher("""
            MATCH (c:Company {name: 'Equinor'}), (f:Field {name: 'Troll'})
            CREATE (c)-[:OPERATES]->(f)
        """)
        return g

    # -- connections=True: overview --

    def test_connections_overview(self, connected_graph):
        desc = connected_graph.describe(connections=True)
        assert "<connections>" in desc
        assert "BELONGS_TO" in desc
        assert "OPERATES" in desc
        assert "count=" in desc

    def test_connections_overview_has_endpoints(self, connected_graph):
        desc = connected_graph.describe(connections=True)
        assert "Well" in desc
        assert "Field" in desc

    def test_connections_overview_has_properties(self, connected_graph):
        desc = connected_graph.describe(connections=True)
        assert "since" in desc

    def test_connections_overview_no_graph_schema(self, connected_graph):
        desc = connected_graph.describe(connections=True)
        assert "<graph" not in desc

    # -- connections=list: deep-dive --

    def test_connections_detail(self, connected_graph):
        desc = connected_graph.describe(connections=["BELONGS_TO"])
        assert "<BELONGS_TO" in desc
        assert "<endpoints>" in desc
        assert "<properties>" in desc
        assert "<samples>" in desc

    def test_connections_detail_has_pair_counts(self, connected_graph):
        desc = connected_graph.describe(connections=["BELONGS_TO"])
        assert "pair" in desc
        assert "Well" in desc
        assert "Field" in desc

    def test_connections_detail_unknown_error(self, connected_graph):
        with pytest.raises(kglite.KgError, match="not found"):
            connected_graph.describe(connections=["BOGUS"])

    def test_connections_detail_error_lists_available(self, connected_graph):
        try:
            connected_graph.describe(connections=["BOGUS"])
        except kglite.KgError as e:
            msg = str(e)
            assert "BELONGS_TO" in msg

    def test_connections_invalid_type_raises(self, connected_graph):
        with pytest.raises(TypeError):
            connected_graph.describe(connections=42)

    # -- Combined --

    def test_connections_and_cypher_combined(self, connected_graph):
        desc = connected_graph.describe(connections=True, cypher=True)
        assert "<connections>" in desc
        assert "<cypher>" in desc


# ── Bug Report ───────────────────────────────────────────────────────────────


class TestBugReport:
    """Tests for bug_report() method."""

    @pytest.fixture()
    def graph(self):
        return KnowledgeGraph()

    @pytest.fixture()
    def tmp_report_path(self, tmp_path):
        return str(tmp_path / "reported_bugs.md")

    def test_creates_file(self, graph, tmp_report_path):
        msg = graph.bug_report(
            "MATCH (n) RETURN n",
            "empty",
            "5 rows",
            "no results",
            path=tmp_report_path,
        )
        assert "saved" in msg.lower()
        import pathlib

        content = pathlib.Path(tmp_report_path).read_text()
        assert "# KGLite Bug Reports" in content
        assert "### Bug Report" in content
        assert "MATCH (n) RETURN n" in content
        assert "no results" in content

    def test_has_timestamp_and_version(self, graph, tmp_report_path):
        graph.bug_report("q", "r", "e", "d", path=tmp_report_path)
        import pathlib

        content = pathlib.Path(tmp_report_path).read_text()
        assert "UTC" in content
        assert "KGLite v" in content

    def test_prepends_new_reports(self, graph, tmp_report_path):
        graph.bug_report("q1", "r1", "e1", "first report", path=tmp_report_path)
        graph.bug_report("q2", "r2", "e2", "second report", path=tmp_report_path)
        import pathlib

        content = pathlib.Path(tmp_report_path).read_text()
        pos_second = content.index("second report")
        pos_first = content.index("first report")
        assert pos_second < pos_first, "new report should appear before old"

    def test_has_all_sections(self, graph, tmp_report_path):
        graph.bug_report("my query", "my result", "my expected", "my desc", path=tmp_report_path)
        import pathlib

        content = pathlib.Path(tmp_report_path).read_text()
        assert "**Query:**" in content
        assert "**Result:**" in content
        assert "**Expected:**" in content
        assert "**Description:**" in content

    def test_has_separators(self, graph, tmp_report_path):
        graph.bug_report("q1", "r1", "e1", "d1", path=tmp_report_path)
        graph.bug_report("q2", "r2", "e2", "d2", path=tmp_report_path)
        import pathlib

        content = pathlib.Path(tmp_report_path).read_text()
        assert content.count("---") >= 2, "each report should have a separator"

    def test_sanitizes_html(self, graph, tmp_report_path):
        graph.bug_report(
            "<script>alert('xss')</script>",
            "result",
            "expected",
            "desc",
            path=tmp_report_path,
        )
        import pathlib

        content = pathlib.Path(tmp_report_path).read_text()
        assert "<script>" not in content

    def test_sanitizes_triple_backticks(self, graph, tmp_report_path):
        graph.bug_report(
            "normal query",
            "```break out",
            "expected",
            "desc",
            path=tmp_report_path,
        )
        import pathlib

        content = pathlib.Path(tmp_report_path).read_text()
        # Triple backticks in user input should be escaped
        assert "\\`\\`\\`" in content

    def test_sanitizes_javascript_protocol(self, graph, tmp_report_path):
        graph.bug_report(
            "query",
            "[click](javascript:alert(1))",
            "expected",
            "desc",
            path=tmp_report_path,
        )
        import pathlib

        content = pathlib.Path(tmp_report_path).read_text()
        assert "javascript:" not in content

    def test_default_path(self, graph, monkeypatch, tmp_path):
        """Without path= argument, writes to reported_bugs.md in cwd."""
        monkeypatch.chdir(tmp_path)
        graph.bug_report("q", "r", "e", "d")

        assert (tmp_path / "reported_bugs.md").exists()

    def test_describe_mentions_bug_report(self, graph):
        desc = graph.describe()
        assert "bug_report" in desc
