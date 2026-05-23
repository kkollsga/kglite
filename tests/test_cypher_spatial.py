"""Tests for Cypher spatial functions: point(), distance(), contains(), etc."""

import pandas as pd
import pytest

import kglite


@pytest.fixture
def geo_graph():
    """Graph with cities that have lat/lon coordinates and areas with WKT polygons."""
    g = kglite.KnowledgeGraph()

    cities = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "title": ["Oslo", "Bergen", "Stavanger", "Trondheim", "Drammen"],
            "latitude": [59.91, 60.39, 58.97, 63.43, 59.74],
            "longitude": [10.75, 5.32, 5.73, 10.40, 10.20],
        }
    )
    g.add_nodes(cities, "City", "id", "title")

    # Add connections between nearby cities
    edges = pd.DataFrame(
        {
            "source": [1, 1, 2, 4],
            "target": [5, 2, 3, 1],
            "type": ["NEAR", "ROUTE", "ROUTE", "ROUTE"],
        }
    )
    g.add_connections(edges, "NEAR", "City", "source", "City", "target")

    # Add areas with WKT polygons
    areas = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "title": ["SouthNorway", "WestNorway", "SmallBox"],
            "geometry": [
                "POLYGON((5 58, 12 58, 12 62, 5 62, 5 58))",
                "POLYGON((4 58, 7 58, 7 62, 4 62, 4 58))",
                "POLYGON((10 59, 11 59, 11 60, 10 60, 10 59))",
            ],
        }
    )
    g.add_nodes(areas, "Area", "id", "title")

    return g


# ============================================================================
# point() constructor
# ============================================================================


class TestPoint:
    def test_point_constructor(self, geo_graph):
        rows = geo_graph.cypher("MATCH (n:City) WHERE n.title = 'Oslo' RETURN point(59.91, 10.75) AS p").to_list()
        assert len(rows) == 1
        p = rows[0]["p"]
        assert isinstance(p, dict)
        assert p["latitude"] == 59.91
        assert p["longitude"] == 10.75

    def test_point_from_property(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' RETURN point(n.latitude, n.longitude) AS p"
        ).to_list()
        assert len(rows) == 1
        p = rows[0]["p"]
        assert abs(p["latitude"] - 59.91) < 0.01
        assert abs(p["longitude"] - 10.75) < 0.01

    def test_point_equality(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "WITH point(n.latitude, n.longitude) AS p "
            "WHERE p = point(59.91, 10.75) "
            "RETURN p"
        ).to_list()
        assert len(rows) == 1

    def test_point_wrong_args(self, geo_graph):
        with pytest.raises(kglite.KgError, match="point.*requires 2"):
            geo_graph.cypher("MATCH (n:City) RETURN point(1.0)")

    def test_point_non_numeric(self, geo_graph):
        with pytest.raises(kglite.KgError, match="numeric"):
            geo_graph.cypher("MATCH (n:City) RETURN point('hello', 10.0)")


# ============================================================================
# distance()
# ============================================================================


class TestDistance:
    def test_distance_two_points(self, geo_graph):
        """Oslo to Bergen should be ~305 km (~305,000 m)."""
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' RETURN distance(point(59.91, 10.75), point(60.39, 5.32)) AS d"
        ).to_list()
        d = rows[0]["d"]
        assert 300_000 < d < 310_000

    def test_distance_four_args(self, geo_graph):
        """4-arg shorthand should give same result."""
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' RETURN distance(59.91, 10.75, 60.39, 5.32) AS d"
        ).to_list()
        d = rows[0]["d"]
        assert 300_000 < d < 310_000

    def test_distance_same_point(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' RETURN distance(point(60.0, 10.0), point(60.0, 10.0)) AS d"
        ).to_list()
        assert rows[0]["d"] == 0.0

    def test_distance_in_where(self, geo_graph):
        """Filter cities within 50km of Oslo."""
        rows = geo_graph.cypher(
            "MATCH (n:City) "
            "WHERE distance(point(n.latitude, n.longitude), point(59.91, 10.75)) < 50000.0 "
            "RETURN n.title AS name ORDER BY name"
        ).to_list()
        names = [r["name"] for r in rows]
        assert "Oslo" in names
        assert "Drammen" in names  # ~40km from Oslo
        assert "Bergen" not in names  # ~305km away

    def test_distance_in_return(self, geo_graph):
        """Return distance values for each city."""
        rows = geo_graph.cypher(
            "MATCH (n:City) "
            "RETURN n.title AS name, "
            "distance(point(n.latitude, n.longitude), point(59.91, 10.75)) AS dist "
            "ORDER BY dist"
        ).to_list()
        assert rows[0]["name"] == "Oslo"
        assert rows[0]["dist"] < 1000.0

    def test_distance_with_traversal(self, geo_graph):
        """Combine graph traversal with spatial filtering."""
        rows = geo_graph.cypher(
            "MATCH (a:City)-[]->(b:City) "
            "RETURN a.title AS a_name, b.title AS b_name, "
            "distance(point(a.latitude, a.longitude), "
            "point(b.latitude, b.longitude)) AS dist "
            "ORDER BY dist"
        ).to_list()
        assert len(rows) >= 1
        # Closest pair should be Oslo-Drammen (~40km)
        assert rows[0]["dist"] < 50_000.0

    def test_distance_wrong_arg_count(self, geo_graph):
        with pytest.raises(kglite.KgError, match="distance.*requires"):
            geo_graph.cypher("MATCH (n:City) RETURN distance(point(1,2), point(3,4), point(5,6))")


# ============================================================================
# contains()
# ============================================================================


class TestContains:
    def test_contains_with_point(self, geo_graph):
        """Oslo (59.91, 10.75) is inside SouthNorway polygon."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) "
            "WHERE a.title = 'SouthNorway' "
            "AND contains(a.geometry, point(59.91, 10.75)) "
            "RETURN a.title AS title"
        ).to_list()
        assert len(rows) == 1
        assert rows[0]["title"] == "SouthNorway"

    def test_contains_with_wkt_and_point(self, geo_graph):
        """contains() with WKT string property and point literal."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) WHERE contains(a.geometry, point(59.91, 10.75)) RETURN a.title AS title ORDER BY title"
        ).to_list()
        titles = [r["title"] for r in rows]
        assert "SouthNorway" in titles
        assert "SmallBox" in titles

    def test_contains_false(self, geo_graph):
        """Trondheim (63.43, 10.40) is outside SouthNorway (max lat 62)."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) WHERE a.title = 'SouthNorway' AND contains(a.geometry, point(63.43, 10.40)) RETURN a.title"
        ).to_list()
        assert len(rows) == 0

    def test_contains_with_node_properties(self, geo_graph):
        """Find which areas contain each city."""
        rows = geo_graph.cypher(
            "MATCH (c:City), (a:Area) "
            "WHERE contains(a.geometry, point(c.latitude, c.longitude)) "
            "RETURN c.title AS city, a.title AS area ORDER BY city, area"
        ).to_list()
        assert len(rows) >= 1
        oslo_areas = [r["area"] for r in rows if r["city"] == "Oslo"]
        assert "SouthNorway" in oslo_areas


# ============================================================================
# intersects()
# ============================================================================


class TestIntersects:
    def test_intersects_overlapping(self, geo_graph):
        """SouthNorway and WestNorway polygons overlap."""
        rows = geo_graph.cypher(
            "MATCH (a:Area), (b:Area) "
            "WHERE a.title = 'SouthNorway' AND b.title = 'WestNorway' "
            "AND intersects(a.geometry, b.geometry) "
            "RETURN a.title"
        ).to_list()
        assert len(rows) == 1

    def test_intersects_non_overlapping(self, geo_graph):
        """SmallBox (lon 10-11) and WestNorway (lon 4-7) don't overlap."""
        rows = geo_graph.cypher(
            "MATCH (a:Area), (b:Area) "
            "WHERE a.title = 'SmallBox' AND b.title = 'WestNorway' "
            "AND intersects(a.geometry, b.geometry) "
            "RETURN a.title"
        ).to_list()
        assert len(rows) == 0


# ============================================================================
# centroid()
# ============================================================================


class TestCentroid:
    def test_centroid_polygon(self, geo_graph):
        """Centroid of SmallBox should be (59.5, 10.5)."""
        rows = geo_graph.cypher("MATCH (a:Area) WHERE a.title = 'SmallBox' RETURN centroid(a.geometry) AS c").to_list()
        c = rows[0]["c"]
        assert isinstance(c, dict)
        assert abs(c["latitude"] - 59.5) < 0.01
        assert abs(c["longitude"] - 10.5) < 0.01

    def test_centroid_in_distance(self, geo_graph):
        """Use centroid in distance calculation."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) "
            "RETURN a.title AS title, "
            "distance(centroid(a.geometry), point(59.91, 10.75)) AS dist "
            "ORDER BY dist"
        ).to_list()
        assert len(rows) == 3


# ============================================================================
# latitude() / longitude()
# ============================================================================


class TestAccessors:
    def test_latitude(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' RETURN latitude(point(59.91, 10.75)) AS lat"
        ).to_list()
        assert abs(rows[0]["lat"] - 59.91) < 0.001

    def test_longitude(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' RETURN longitude(point(59.91, 10.75)) AS lon"
        ).to_list()
        assert abs(rows[0]["lon"] - 10.75) < 0.001

    def test_accessors_with_centroid(self, geo_graph):
        """Extract lat/lon from centroid."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) WHERE a.title = 'SmallBox' "
            "RETURN latitude(centroid(a.geometry)) AS lat, "
            "longitude(centroid(a.geometry)) AS lon"
        ).to_list()
        assert abs(rows[0]["lat"] - 59.5) < 0.01
        assert abs(rows[0]["lon"] - 10.5) < 0.01


# ============================================================================
# Spatial aggregation
# ============================================================================


class TestSpatialAggregation:
    def test_avg_distance(self, geo_graph):
        """Average distance from Oslo to all cities."""
        rows = geo_graph.cypher(
            "MATCH (n:City) RETURN avg(distance(point(n.latitude, n.longitude), point(59.91, 10.75))) AS avg_dist"
        ).to_list()
        assert rows[0]["avg_dist"] > 0

    def test_min_max_distance(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) "
            "RETURN min(distance(point(n.latitude, n.longitude), point(59.91, 10.75))) AS min_d, "
            "max(distance(point(n.latitude, n.longitude), point(59.91, 10.75))) AS max_d"
        ).to_list()
        assert rows[0]["min_d"] < 1000.0  # Oslo to itself ~ 0
        assert rows[0]["max_d"] > 300_000  # Farthest city > 300km


# ============================================================================
# Spatial-join fusion: MATCH (a:A), (b:B) WHERE contains(a, b)
# ============================================================================


@pytest.fixture
def spatial_join_graph():
    """Graph with SpatialConfig on both Area (geometry) and City (location),
    which is the exact shape that triggers `fuse_spatial_join`."""
    g = kglite.KnowledgeGraph()
    areas = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "title": ["SouthNorway", "WestNorway", "SmallBox"],
            "geometry": [
                "POLYGON((5 58, 12 58, 12 62, 5 62, 5 58))",
                "POLYGON((4 58, 7 58, 7 62, 4 62, 4 58))",
                "POLYGON((10 59, 11 59, 11 60, 10 60, 10 59))",
            ],
        }
    )
    g.add_nodes(areas, "Area", "id", "title", column_types={"geometry": "geometry"})
    cities = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "title": ["Oslo", "Bergen", "Stavanger", "Trondheim", "Drammen"],
            "latitude": [59.91, 60.39, 58.97, 63.43, 59.74],
            "longitude": [10.75, 5.32, 5.73, 10.40, 10.20],
        }
    )
    g.add_nodes(
        cities,
        "City",
        "id",
        "title",
        column_types={"latitude": "location.lat", "longitude": "location.lon"},
    )
    return g


class TestSpatialJoin:
    def _brute_force(self, graph):
        """Reference implementation using the pre-existing non-fused path
        (`contains(a.geometry, point(c.latitude, c.longitude))` goes through
        expression evaluation, not the spatial-join executor)."""
        rows = graph.cypher(
            "MATCH (a:Area), (c:City) "
            "WHERE contains(a.geometry, point(c.latitude, c.longitude)) "
            "RETURN a.title AS area, c.title AS city ORDER BY area, city"
        ).to_list()
        return {(r["area"], r["city"]) for r in rows}

    def test_two_pattern_matches_brute_force(self, spatial_join_graph):
        """The fused spatial-join must produce the same (area, city) pairs
        as the non-fused brute-force path."""
        fused = spatial_join_graph.cypher(
            "MATCH (a:Area), (c:City) WHERE contains(a, c) RETURN a.title AS area, c.title AS city ORDER BY area, city"
        ).to_list()
        fused_set = {(r["area"], r["city"]) for r in fused}
        reference = self._brute_force(spatial_join_graph)
        assert fused_set == reference
        # Sanity: the reference isn't empty — we expect at least Oslo∈SouthNorway.
        assert ("SouthNorway", "Oslo") in fused_set

    def test_with_and_remainder(self, spatial_join_graph):
        """The ANDed remainder predicate must still filter correctly after
        the spatial-join emits pairs."""
        rows = spatial_join_graph.cypher(
            "MATCH (a:Area), (c:City) WHERE contains(a, c) AND c.title = 'Oslo' RETURN a.title AS area ORDER BY area"
        ).to_list()
        areas = [r["area"] for r in rows]
        assert "Oslo" not in areas  # areas only — not cities
        assert "SouthNorway" in areas  # Oslo is inside SouthNorway

    def test_explain_shows_spatial_join(self, spatial_join_graph):
        """EXPLAIN output should mention the fused SpatialJoin operator
        so that we can confirm the fusion actually fires in production."""
        result = spatial_join_graph.cypher(
            "EXPLAIN MATCH (a:Area), (c:City) WHERE contains(a, c) RETURN a.title"
        ).to_list()
        text = "\n".join(str(r) for r in result)
        assert "SpatialJoin" in text

    def test_negated_falls_back_to_existing_path(self, spatial_join_graph):
        """`NOT contains(a, c)` must not trigger the fusion (the executor
        wouldn't handle negation) and must still return correct results via
        the existing per-row fast path."""
        rows = spatial_join_graph.cypher(
            "MATCH (a:Area), (c:City) "
            "WHERE NOT contains(a, c) AND a.title = 'SmallBox' "
            "RETURN c.title AS city ORDER BY city"
        ).to_list()
        # SmallBox (lon 10-11, lat 59-60) contains Oslo and Drammen;
        # Bergen, Stavanger, Trondheim are outside.
        assert [r["city"] for r in rows] == ["Bergen", "Stavanger", "Trondheim"]
        # Confirm the fallback path was taken — no SpatialJoin in EXPLAIN.
        explain = spatial_join_graph.cypher(
            "EXPLAIN MATCH (a:Area), (c:City) WHERE NOT contains(a, c) AND a.title = 'SmallBox' RETURN c.title"
        ).to_list()
        assert "SpatialJoin" not in "\n".join(str(r) for r in explain)

    def test_empty_when_one_type_missing(self):
        """A graph with Area but no City must return zero rows without crashing."""
        g = kglite.KnowledgeGraph()
        areas = pd.DataFrame(
            {
                "id": [1],
                "title": ["Only"],
                "geometry": ["POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"],
            }
        )
        g.add_nodes(areas, "Area", "id", "title", column_types={"geometry": "geometry"})
        rows = g.cypher("MATCH (a:Area), (c:City) WHERE contains(a, c) RETURN a.title").to_list()
        assert rows == []

    def test_multi_match_with_centroid_probe(self, spatial_join_graph):
        """Multi-MATCH form: `MATCH (a:Area) MATCH (b:Area) WHERE contains(a, centroid(b))`.
        Sodir's IN_STRUCTURAL_ELEMENT shape — fusion must wire `centroid()`
        through `SpatialProbeKind::Centroid` and produce the same pairs as
        the brute-force two-pattern path."""
        fused = spatial_join_graph.cypher(
            "MATCH (outer:Area) MATCH (inner:Area) "
            "WHERE contains(outer, centroid(inner)) "
            "RETURN outer.title AS o, inner.title AS i ORDER BY o, i"
        ).to_list()
        fused_pairs = {(r["o"], r["i"]) for r in fused}
        reference = spatial_join_graph.cypher(
            "MATCH (outer:Area), (inner:Area) "
            "WHERE contains(outer.geometry, centroid(inner.geometry)) "
            "RETURN outer.title AS o, inner.title AS i ORDER BY o, i"
        ).to_list()
        ref_pairs = {(r["o"], r["i"]) for r in reference}
        assert fused_pairs == ref_pairs
        assert ("SouthNorway", "SmallBox") in fused_pairs  # SmallBox ⊂ SouthNorway
        assert ("SmallBox", "SmallBox") in fused_pairs  # self-containment

    def test_multi_match_centroid_fires_fusion(self, spatial_join_graph):
        """EXPLAIN must show SpatialJoin for the multi-MATCH centroid shape
        so the planner's `try_fuse_spatial_multi_match` extension is exercised."""
        explain = spatial_join_graph.cypher(
            "EXPLAIN MATCH (outer:Area) MATCH (inner:Area) WHERE contains(outer, centroid(inner)) RETURN outer.title"
        ).to_list()
        assert "SpatialJoin" in "\n".join(str(r) for r in explain)


# ── 0.8.20: Geometry primitives + KNN ──────────────────────────


class TestGeomPrimitives:
    def test_geom_is_valid_true(self):
        g = kglite.KnowledgeGraph()
        rows = list(g.cypher("RETURN geom_is_valid('POINT(10.7 59.9)') AS v"))
        assert rows[0]["v"] is True

    def test_geom_length_linestring(self):
        g = kglite.KnowledgeGraph()
        # Oslo (10.75, 59.91) → Bergen (5.32, 60.39) ≈ 305 km
        rows = list(g.cypher("RETURN geom_length('LINESTRING(10.75 59.91, 5.32 60.39)') AS m"))
        assert 290_000 < rows[0]["m"] < 320_000

    def test_geom_length_point_is_zero(self):
        g = kglite.KnowledgeGraph()
        rows = list(g.cypher("RETURN geom_length('POINT(10 60)') AS m"))
        assert rows[0]["m"] == 0.0

    def test_geom_convex_hull_list(self):
        g = kglite.KnowledgeGraph()
        rows = list(g.cypher("RETURN geom_convex_hull(['POINT(0 0)', 'POINT(1 0)', 'POINT(1 1)', 'POINT(0 1)']) AS h"))
        assert "POLYGON" in rows[0]["h"]

    def test_geom_union_overlapping(self):
        g = kglite.KnowledgeGraph()
        rows = list(
            g.cypher("RETURN geom_union('POLYGON((0 0,2 0,2 2,0 2,0 0))', 'POLYGON((1 1,3 1,3 3,1 3,1 1))') AS u")
        )
        assert "MULTIPOLYGON" in rows[0]["u"]

    def test_geom_intersection_disjoint_returns_empty(self):
        g = kglite.KnowledgeGraph()
        rows = list(
            g.cypher(
                "RETURN geom_intersection('POLYGON((0 0,1 0,1 1,0 1,0 0))', "
                "'POLYGON((10 10,11 10,11 11,10 11,10 10))') AS i"
            )
        )
        # Empty MULTIPOLYGON when no intersection
        assert "MULTIPOLYGON" in rows[0]["i"]

    def test_geom_buffer_returns_multipolygon(self):
        g = kglite.KnowledgeGraph()
        rows = list(g.cypher("RETURN geom_buffer('POINT(10.7 59.9)', 1000) AS b"))
        assert "MULTIPOLYGON" in rows[0]["b"]

    def test_geom_buffer_negative_errors(self):
        g = kglite.KnowledgeGraph()
        with pytest.raises(kglite.KgError, match="negative"):
            list(g.cypher("RETURN geom_buffer('POINT(0 0)', -10) AS b"))


class TestKgKnn:
    @pytest.fixture
    def cities(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame(
                {
                    "id": ["oslo", "bergen", "tromso", "tokyo"],
                    "name": ["Oslo", "Bergen", "Tromso", "Tokyo"],
                    "lat": [59.9, 60.4, 69.6, 35.7],
                    "lon": [10.7, 5.3, 18.9, 139.7],
                }
            ),
            "City",
            "id",
            "name",
        )
        g.set_spatial("City", location=("lat", "lon"))
        return g

    def test_knn_from_bergen(self, cities):
        rows = list(
            cities.cypher(
                "CALL kg_knn({lat: 60.4, lon: 5.3, target_type: 'City', k: 3}) "
                "YIELD node, distance_m RETURN node.title AS t, distance_m"
            )
        )
        names = [r["t"] for r in rows]
        assert names[0] == "Bergen"
        assert names[1] == "Oslo"
        assert names[2] == "Tromso"
        assert rows[0]["distance_m"] == 0.0
        assert 290_000 < rows[1]["distance_m"] < 320_000  # Oslo-Bergen ≈ 305km

    def test_knn_k_limits_results(self, cities):
        rows = list(
            cities.cypher(
                "CALL kg_knn({lat: 60.4, lon: 5.3, target_type: 'City', k: 1}) "
                "YIELD node, distance_m RETURN node.title AS t"
            )
        )
        assert len(rows) == 1
        assert rows[0]["t"] == "Bergen"

    def test_knn_missing_lat_errors(self, cities):
        with pytest.raises(kglite.KgError, match="missing required parameter"):
            list(cities.cypher("CALL kg_knn({lon: 5.3, target_type: 'City', k: 3}) YIELD node RETURN node"))


# ============================================================================
# Implicit spatial-config inference (P1)
# ============================================================================


class TestSpatialConfigInference:
    """Spatial functions should accept nodes that store WKT or lat/lon under
    conventional property names without an explicit `set_spatial` /
    `column_types` registration. Explicit configs always take precedence."""

    def test_intersects_uses_wkt_geometry_property(self):
        import pandas as pd

        g = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["P1", "P2"],
                "wkt_geometry": [
                    "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                    "POLYGON((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))",
                ],
            }
        )
        g.add_nodes(df, "Prospect", "id", "name")  # no spatial column_types
        rows = list(
            g.cypher("MATCH (a:Prospect), (b:Prospect) WHERE a.id = 1 AND b.id = 2 RETURN intersects(a, b) AS overlaps")
        )
        assert rows == [{"overlaps": True}]

    def test_centroid_uses_geometry_property(self):
        import pandas as pd

        g = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1],
                "name": ["P1"],
                "geometry": ["POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))"],
            }
        )
        g.add_nodes(df, "Block", "id", "name")
        rows = list(g.cypher("MATCH (a:Block) RETURN centroid(a) AS c"))
        assert abs(rows[0]["c"]["latitude"] - 1.0) < 1e-6
        assert abs(rows[0]["c"]["longitude"] - 1.0) < 1e-6

    def test_distance_uses_lat_lon_property(self):
        import pandas as pd

        g = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "lat": [59.91, 60.39], "lon": [10.75, 5.32]})
        g.add_nodes(df, "City", "id", "name")
        rows = list(g.cypher("MATCH (a:City), (b:City) WHERE a.id = 1 AND b.id = 2 RETURN distance(a, b) AS d"))
        # Oslo-Bergen ≈ 305 km
        assert 290_000 < rows[0]["d"] < 320_000

    def test_explicit_config_overrides_inference(self):
        """If both wkt_geometry (conventional) and a registered field exist,
        the explicit config wins — inference only fires when no config
        is registered."""
        import pandas as pd

        g = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1],
                "name": ["P1"],
                "wkt_geometry": ["POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"],
                "shape_a": ["POLYGON((10 10, 11 10, 11 11, 10 11, 10 10))"],
            }
        )
        g.add_nodes(
            df,
            "Prospect",
            "id",
            "name",
            column_types={"shape_a": "geometry"},
        )
        rows = list(g.cypher("MATCH (a:Prospect) RETURN centroid(a) AS c"))
        # Explicit config picks shape_a (centered at 10.5, 10.5), not wkt_geometry (0.5, 0.5)
        assert abs(rows[0]["c"]["latitude"] - 10.5) < 1e-6
        assert abs(rows[0]["c"]["longitude"] - 10.5) < 1e-6

    def test_no_spatial_data_still_errors_helpfully(self):
        """A node with no geometry/lat-lon properties of any kind should
        still produce the help-bearing error."""
        import pandas as pd

        g = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["P1"], "color": ["red"]})
        g.add_nodes(df, "Item", "id", "name")
        with pytest.raises(kglite.KgError, match="conventional property name"):
            list(g.cypher("MATCH (a:Item), (b:Item) RETURN intersects(a, b) AS x"))
