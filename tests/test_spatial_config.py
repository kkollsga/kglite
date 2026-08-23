"""Tests for spatial configuration (location, geometry, point, shape types)."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def field_graph():
    """Graph with Field nodes that have lat/lon and WKT geometry."""
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Troll", "Draugen", "Asgard"],
            "latitude": [60.6, 64.4, 65.1],
            "longitude": [3.7, 7.8, 6.8],
            "wkt_polygon": [
                "POLYGON((3 60, 4 60, 4 61, 3 61, 3 60))",
                "POLYGON((7 64, 8 64, 8 65, 7 65, 7 64))",
                "POLYGON((6 65, 7 65, 7 66, 6 66, 6 65))",
            ],
        }
    )
    graph.add_nodes(
        df,
        "Field",
        "id",
        "name",
        column_types={
            "latitude": "location.lat",
            "longitude": "location.lon",
            "wkt_polygon": "geometry",
        },
    )
    return graph


@pytest.fixture
def well_graph():
    """Graph with Well nodes that have surface + bottom-hole points and a boundary shape."""
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Well_A", "Well_B"],
            "surface_lat": [60.5, 64.3],
            "surface_lon": [3.5, 7.7],
            "bh_lat": [60.6, 64.5],
            "bh_lon": [3.6, 7.9],
            "boundary_wkt": [
                "POLYGON((3 60, 4 60, 4 61, 3 61, 3 60))",
                "POLYGON((7 64, 8 64, 8 65, 7 65, 7 64))",
            ],
        }
    )
    graph.add_nodes(
        df,
        "Well",
        "id",
        "name",
        column_types={
            "surface_lat": "location.lat",
            "surface_lon": "location.lon",
            "bh_lat": "point.bottom_hole.lat",
            "bh_lon": "point.bottom_hole.lon",
            "boundary_wkt": "shape.boundary",
        },
    )
    return graph


# ── set_spatial / spatial ────────────────────────────────────────


class TestSetGetSpatial:
    def test_set_spatial_basic(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1],
                "name": ["A"],
                "lat": [60.0],
                "lon": [3.0],
            }
        )
        graph.add_nodes(df, "Field", "id", "name")
        graph.set_spatial("Field", location=("lat", "lon"))

        config = graph.spatial("Field")
        assert config is not None
        assert config["location"] == ("lat", "lon")

    def test_set_spatial_full(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Field", "id", "name")
        graph.set_spatial(
            "Field",
            location=("lat", "lon"),
            geometry="wkt_col",
            points={"drill": ("d_lat", "d_lon")},
            shapes={"boundary": "bnd_wkt"},
        )

        config = graph.spatial("Field")
        assert config["location"] == ("lat", "lon")
        assert config["geometry"] == "wkt_col"
        assert config["points"] == {"drill": ("d_lat", "d_lon")}
        assert config["shapes"] == {"boundary": "bnd_wkt"}

    def test_spatial_none(self):
        graph = KnowledgeGraph()
        assert graph.spatial("Nonexistent") is None

    def test_spatial_all(self, field_graph):
        result = field_graph.spatial()
        assert result is not None
        assert "Field" in result

    def test_spatial_empty_graph(self):
        graph = KnowledgeGraph()
        assert graph.spatial() is None


# ── column_types parsing ─────────────────────────────────────────────


class TestColumnTypesParsing:
    def test_location_via_column_types(self, field_graph):
        """Location config is set via column_types during add_nodes."""
        config = field_graph.spatial("Field")
        assert config is not None
        assert config["location"] == ("latitude", "longitude")

    def test_geometry_via_column_types(self, field_graph):
        """Geometry config is set via column_types during add_nodes."""
        config = field_graph.spatial("Field")
        assert config["geometry"] == "wkt_polygon"

    def test_named_point_via_column_types(self, well_graph):
        """Named points are set via column_types during add_nodes."""
        config = well_graph.spatial("Well")
        assert config is not None
        assert config["points"] == {"bottom_hole": ("bh_lat", "bh_lon")}

    def test_named_shape_via_column_types(self, well_graph):
        """Named shapes are set via column_types during add_nodes."""
        config = well_graph.spatial("Well")
        assert config["shapes"] == {"boundary": "boundary_wkt"}

    def test_incomplete_location_raises(self):
        """Missing one of location.lat/location.lon raises ValueError."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "lat": [60.0]})
        with pytest.raises(ValueError, match="Incomplete location"):
            graph.add_nodes(
                df,
                "Field",
                "id",
                "name",
                column_types={
                    "lat": "location.lat",
                },
            )

    def test_incomplete_named_point_raises(self):
        """Missing one of point.<name>.lat/lon raises ValueError."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "x": [60.0]})
        with pytest.raises(ValueError, match="Incomplete point"):
            graph.add_nodes(
                df,
                "Field",
                "id",
                "name",
                column_types={
                    "x": "point.drill.lat",
                },
            )

    def test_data_stored_as_natural_types(self, field_graph):
        """Spatial columns are stored as their natural types (float/str)."""
        rv = field_graph.cypher("MATCH (n:Field {name: 'Troll'}) RETURN n.latitude AS lat, n.wkt_polygon AS wkt")
        row = rv[0]
        assert isinstance(row["lat"], float)
        assert isinstance(row["wkt"], str)


# ── Cypher distance(a, b) auto-resolution ────────────────────────────


class TestDistanceAutoResolution:
    def test_distance_with_location_config(self, field_graph):
        """distance(a, b) auto-resolves via spatial config location."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Draugen'})
            RETURN distance(a, b) AS dist
        """)
        dist = rv[0]["dist"]
        assert isinstance(dist, float)
        assert dist > 0

    def test_distance_geometry_fallback(self):
        """distance(a, b) falls back to geometry centroid when no location."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["FieldA", "FieldB"],
                "wkt": [
                    "POLYGON((3 60, 4 60, 4 61, 3 61, 3 60))",
                    "POLYGON((7 64, 8 64, 8 65, 7 65, 7 64))",
                ],
            }
        )
        graph.add_nodes(
            df,
            "Field",
            "id",
            "name",
            column_types={
                "wkt": "geometry",
            },
        )
        rv = graph.cypher("""
            MATCH (a:Field {name: 'FieldA'}), (b:Field {name: 'FieldB'})
            RETURN distance(a, b) AS dist
        """)
        dist = rv[0]["dist"]
        assert isinstance(dist, float)
        assert dist > 0

    def test_distance_named_points(self, well_graph):
        """distance(a.bottom_hole, b.bottom_hole) works with named points."""
        rv = well_graph.cypher("""
            MATCH (a:Well {name: 'Well_A'}), (b:Well {name: 'Well_B'})
            RETURN distance(a.bottom_hole, b.bottom_hole) AS dist
        """)
        dist = rv[0]["dist"]
        assert isinstance(dist, float)
        assert dist > 0

    def test_distance_no_config_returns_null(self):
        """distance(a, b) without spatial config returns None (Null)."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Thing", "id", "name")
        rv = graph.cypher("""
            MATCH (a:Thing {name: 'A'}), (b:Thing {name: 'B'})
            RETURN distance(a, b) AS dist
        """)
        assert rv[0]["dist"] is None


# ── Virtual properties in Cypher ─────────────────────────────────────


class TestVirtualProperties:
    def test_location_virtual_property(self, field_graph):
        """n.location returns a Point value."""
        rv = field_graph.cypher("MATCH (n:Field {name: 'Troll'}) RETURN n.location AS loc")
        loc = rv[0]["loc"]
        # Point is returned as a dict with latitude/longitude
        assert isinstance(loc, dict)
        assert abs(loc["latitude"] - 60.6) < 0.01
        assert abs(loc["longitude"] - 3.7) < 0.01

    def test_geometry_virtual_property(self, field_graph):
        """n.geometry returns the WKT string from the configured geometry field."""
        rv = field_graph.cypher("MATCH (n:Field {name: 'Troll'}) RETURN n.geometry AS geom")
        geom = rv[0]["geom"]
        assert isinstance(geom, str)
        assert "POLYGON" in geom

    def test_named_point_virtual_property(self, well_graph):
        """n.bottom_hole returns a Point from the named point config."""
        rv = well_graph.cypher("MATCH (n:Well {name: 'Well_A'}) RETURN n.bottom_hole AS pt")
        pt = rv[0]["pt"]
        assert isinstance(pt, dict)
        assert abs(pt["latitude"] - 60.6) < 0.01
        assert abs(pt["longitude"] - 3.6) < 0.01

    def test_named_shape_virtual_property(self, well_graph):
        """n.boundary returns the WKT string from the named shape config."""
        rv = well_graph.cypher("MATCH (n:Well {name: 'Well_A'}) RETURN n.boundary AS shp")
        shp = rv[0]["shp"]
        assert isinstance(shp, str)
        assert "POLYGON" in shp

    def test_distance_with_virtual_location(self, field_graph):
        """distance(a.location, b.location) uses the explicit location virtual property."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Draugen'})
            RETURN distance(a.location, b.location) AS dist
        """)
        dist = rv[0]["dist"]
        assert isinstance(dist, float)
        assert dist > 0


# ── pymethods auto-resolution ────────────────────────────────────────


class TestPymethodsAutoResolution:
    def test_near_point_m_auto_resolve(self, field_graph):
        """near_point_m() works without explicit lat_field/lon_field."""
        result = field_graph.select("Field").near_point_m(
            center_lat=60.5,
            center_lon=3.5,
            max_distance_m=100_000.0,
        )
        assert result.len() >= 1

    def test_within_bounds_auto_resolve(self, field_graph):
        """within_bounds() works without explicit lat_field/lon_field."""
        result = field_graph.select("Field").within_bounds(
            min_lat=59.0,
            max_lat=62.0,
            min_lon=2.0,
            max_lon=5.0,
        )
        assert result.len() >= 1

    def test_bounds_auto_resolve(self, field_graph):
        """bounds() works without explicit lat_field/lon_field."""
        bounds = field_graph.select("Field").bounds()
        assert bounds is not None
        assert "min_lat" in bounds

    def test_centroid_auto_resolve(self, field_graph):
        """centroid() works without explicit lat_field/lon_field."""
        centroid = field_graph.select("Field").centroid()
        assert centroid is not None
        assert "latitude" in centroid

    def test_intersects_geometry_auto_resolve(self, field_graph):
        """intersects_geometry() works without explicit geometry_field."""
        result = field_graph.select("Field").intersects_geometry(
            "POLYGON((2 59, 5 59, 5 62, 2 62, 2 59))",
        )
        assert result.len() >= 1

    def test_contains_point_auto_resolve(self, field_graph):
        """contains_point() works without explicit geometry_field."""
        result = field_graph.select("Field").contains_point(
            lat=60.5,
            lon=3.5,
        )
        assert result.len() >= 1


# ── Geometry centroid fallback (no lat/lon) ──────────────────────────


class TestGeometryCentroidFallback:
    """Fluent API methods fall back to geometry centroid when location is missing."""

    @pytest.fixture
    def prospect_graph(self):
        """Graph with Prospect nodes that have only WKT geometry (no lat/lon)."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alpha", "Beta", "Gamma"],
                "wkt_geometry": [
                    "POLYGON((3 60, 4 60, 4 61, 3 61, 3 60))",  # centroid ~(60.5, 3.5)
                    "POLYGON((7 64, 8 64, 8 65, 7 65, 7 64))",  # centroid ~(64.5, 7.5)
                    "POLYGON((6 65, 7 65, 7 66, 6 66, 6 65))",  # centroid ~(65.5, 6.5)
                ],
            }
        )
        graph.add_nodes(
            df,
            "Prospect",
            "id",
            "name",
            column_types={
                "wkt_geometry": "geometry",
            },
        )
        return graph

    def test_near_point_m_fallback(self, prospect_graph):
        """near_point_m() uses geometry centroid when no lat/lon."""
        result = prospect_graph.select("Prospect").near_point_m(
            center_lat=60.5,
            center_lon=3.5,
            max_distance_m=50_000.0,
        )
        assert result.len() == 1  # only Alpha
        nodes = result.collect()
        assert nodes[0]["title"] == "Alpha"

    def test_within_bounds_fallback(self, prospect_graph):
        """within_bounds() uses geometry centroid when no lat/lon."""
        result = prospect_graph.select("Prospect").within_bounds(
            min_lat=60.0,
            max_lat=61.0,
            min_lon=3.0,
            max_lon=4.0,
        )
        assert result.len() == 1  # only Alpha

    def test_bounds_fallback(self, prospect_graph):
        """bounds() uses geometry centroids when no lat/lon."""
        bounds = prospect_graph.select("Prospect").bounds()
        assert bounds is not None
        # All three centroids should be included
        assert bounds["min_lat"] < 61.0
        assert bounds["max_lat"] > 65.0

    def test_centroid_fallback(self, prospect_graph):
        """centroid() uses geometry centroids when no lat/lon."""
        centroid = prospect_graph.select("Prospect").centroid()
        assert centroid is not None
        # Average of ~(60.5, 3.5), ~(64.5, 7.5), ~(65.5, 6.5)
        assert 63.0 < centroid["latitude"] < 64.0


# ── Save/Load roundtrip ──────────────────────────────────────────────


class TestSaveLoadRoundtrip:
    def test_spatial_config_persists(self, field_graph, tmp_path):
        """Spatial config survives save/load cycle."""
        path = str(tmp_path / "test.kgl")
        field_graph.save(path)

        from kglite import load

        loaded = load(path)

        config = loaded.spatial("Field")
        assert config is not None
        assert config["location"] == ("latitude", "longitude")
        assert config["geometry"] == "wkt_polygon"

    def test_distance_works_after_load(self, field_graph, tmp_path):
        """Cypher distance(a, b) works after save/load."""
        path = str(tmp_path / "test.kgl")
        field_graph.save(path)

        from kglite import load

        loaded = load(path)

        rv = loaded.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Draugen'})
            RETURN distance(a, b) AS dist
        """)
        assert rv[0]["dist"] > 0


# ── Node-aware spatial Cypher functions ──────────────────────────────


class TestNodeAwareSpatialFunctions:
    def test_contains_node_node(self, field_graph):
        """contains(a, b) where a has geometry and b has location."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Troll'})
            RETURN contains(a, b) AS result
        """)
        # Troll's location (60.6, 3.7) is inside Troll's polygon (3 60, 4 60, 4 61, 3 61)
        assert rv[0]["result"] is True

    def test_contains_node_point(self, field_graph):
        """contains(a, point(lat, lon)) works."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'})
            RETURN contains(a, point(60.5, 3.5)) AS result
        """)
        assert rv[0]["result"] is True

    def test_contains_outside_point(self, field_graph):
        """contains() returns false for point outside geometry."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'})
            RETURN contains(a, point(70.0, 10.0)) AS result
        """)
        assert rv[0]["result"] is False

    def test_contains_wkt_point(self):
        """contains('POLYGON(...)', point(lat, lon)) backward compat."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Thing", "id", "name")
        rv = graph.cypher("""
            MATCH (n:Thing)
            RETURN contains('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))', point(5.0, 5.0)) AS result
        """)
        assert rv[0]["result"] is True

    def test_intersects_node_node(self, field_graph):
        """intersects(a, b) with geometry configs."""
        # Troll and Draugen have non-overlapping polygons
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Draugen'})
            RETURN intersects(a, b) AS result
        """)
        assert rv[0]["result"] is False

    def test_intersects_self(self, field_graph):
        """intersects(a, a) should be true (same geometry)."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Troll'})
            RETURN intersects(a, b) AS result
        """)
        assert rv[0]["result"] is True

    def test_intersects_wkt_wkt(self):
        """intersects('POLYGON(...)', 'POLYGON(...)') backward compat."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Thing", "id", "name")
        rv = graph.cypher("""
            MATCH (n:Thing)
            RETURN intersects(
                'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))',
                'POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))'
            ) AS result
        """)
        assert rv[0]["result"] is True

    def test_centroid_node(self, field_graph):
        """centroid(n) returns Point from geometry config."""
        rv = field_graph.cypher("""
            MATCH (n:Field {name: 'Troll'})
            RETURN centroid(n) AS c
        """)
        c = rv[0]["c"]
        assert isinstance(c, dict)
        assert "latitude" in c
        assert "longitude" in c
        # Troll polygon: (3 60, 4 60, 4 61, 3 61) → centroid ~(60.5, 3.5)
        assert abs(c["latitude"] - 60.5) < 0.1
        assert abs(c["longitude"] - 3.5) < 0.1

    def test_centroid_wkt(self):
        """centroid('POLYGON(...)') backward compat."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Thing", "id", "name")
        rv = graph.cypher("""
            MATCH (n:Thing)
            RETURN centroid('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))') AS c
        """)
        c = rv[0]["c"]
        assert isinstance(c, dict)
        assert abs(c["latitude"] - 5.0) < 0.1
        assert abs(c["longitude"] - 5.0) < 0.1

    def test_area_node(self, field_graph):
        """area(n) returns m² from geometry config."""
        rv = field_graph.cypher("""
            MATCH (n:Field {name: 'Troll'})
            RETURN area(n) AS a
        """)
        a = rv[0]["a"]
        assert isinstance(a, float)
        assert a > 0  # ~1 degree × 1 degree polygon (area in m²)

    def test_area_wkt(self):
        """area('POLYGON(...)') backward compat."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Thing", "id", "name")
        rv = graph.cypher("""
            MATCH (n:Thing)
            RETURN area('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))') AS a
        """)
        assert rv[0]["a"] > 0

    def test_perimeter_node(self, field_graph):
        """perimeter(n) returns meters from geometry config."""
        rv = field_graph.cypher("""
            MATCH (n:Field {name: 'Troll'})
            RETURN perimeter(n) AS p
        """)
        p = rv[0]["p"]
        assert isinstance(p, float)
        assert p > 0  # perimeter of ~1° × 1° polygon (meters)

    def test_contains_named_shape(self, well_graph):
        """contains(a.boundary, point(...)) with named shapes."""
        rv = well_graph.cypher("""
            MATCH (w:Well {name: 'Well_A'})
            RETURN contains(w.boundary, point(60.5, 3.5)) AS result
        """)
        assert rv[0]["result"] is True


# ── Geometry-aware distance ──────────────────────────────────────────


class TestGeometryAwareDistance:
    def test_distance_point_to_geometry_inside(self, field_graph):
        """distance(point(inside), a.geometry) returns 0."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'})
            RETURN distance(point(60.5, 3.5), a.geometry) AS dist
        """)
        assert rv[0]["dist"] == 0.0

    def test_distance_point_to_geometry_outside(self, field_graph):
        """distance(point(outside), a.geometry) returns distance > 0."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'})
            RETURN distance(point(62.0, 5.0), a.geometry) AS dist
        """)
        assert rv[0]["dist"] > 0

    def test_distance_geometry_to_geometry_self(self, field_graph):
        """distance(a.geometry, a.geometry) returns 0 (same geometry intersects itself)."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Troll'})
            RETURN distance(a.geometry, b.geometry) AS dist
        """)
        assert rv[0]["dist"] == 0.0

    def test_distance_geometry_to_geometry_separate(self, field_graph):
        """distance(a.geometry, b.geometry) returns centroid distance when separate."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Draugen'})
            RETURN distance(a.geometry, b.geometry) AS dist
        """)
        assert rv[0]["dist"] > 0

    def test_distance_node_to_geometry_mixed(self, field_graph):
        """distance(a, b.geometry) — point (location) to body (geometry)."""
        rv = field_graph.cypher("""
            MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Troll'})
            RETURN distance(a, b.geometry) AS dist
        """)
        # Troll's location (60.6, 3.7) is inside Troll's polygon → distance = 0
        assert rv[0]["dist"] == 0.0

    def test_distance_geometry_touching(self):
        """distance(a.geometry, b.geometry) returns 0 when polygons share an edge."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["A", "B"],
                "wkt": [
                    "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                    "POLYGON((1 0, 2 0, 2 1, 1 1, 1 0))",  # shares edge at x=1
                ],
            }
        )
        graph.add_nodes(
            df,
            "Box",
            "id",
            "name",
            column_types={
                "wkt": "geometry",
            },
        )
        rv = graph.cypher("""
            MATCH (a:Box {name: 'A'}), (b:Box {name: 'B'})
            RETURN distance(a.geometry, b.geometry) AS dist
        """)
        assert rv[0]["dist"] == 0.0


# ── One-sided lat_field / lon_field overrides ────────────────────────


class TestOneSidedCoordinateOverride:
    """Naming one coordinate field must not discard the config for the other."""

    @pytest.fixture
    def renamed_graph(self):
        """Sites whose coordinate properties are named nothing like the literals."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["Near", "Far"],
                "lat_wgs84": [60.5, 10.0],
                "lon_wgs84": [3.5, 40.0],
            }
        )
        graph.add_nodes(df, "Site", "id", "name")
        graph.set_spatial("Site", location=("lat_wgs84", "lon_wgs84"))
        return graph

    def test_near_point_m_explicit_lat_keeps_configured_lon(self, renamed_graph):
        result = renamed_graph.select("Site").near_point_m(
            center_lat=60.5,
            center_lon=3.5,
            max_distance_m=50_000.0,
            lat_field="lat_wgs84",
        )
        assert result.len() == 1
        assert result.collect()[0]["title"] == "Near"

    def test_within_bounds_explicit_lon_keeps_configured_lat(self, renamed_graph):
        result = renamed_graph.select("Site").within_bounds(
            min_lat=59.0,
            max_lat=62.0,
            min_lon=2.0,
            max_lon=5.0,
            lon_field="lon_wgs84",
        )
        assert result.len() == 1
        assert result.collect()[0]["title"] == "Near"

    def test_bounds_explicit_lat_keeps_configured_lon(self, renamed_graph):
        bounds = renamed_graph.select("Site").bounds(lat_field="lat_wgs84")
        assert bounds is not None
        assert bounds["min_lon"] == pytest.approx(3.5)
        assert bounds["max_lon"] == pytest.approx(40.0)

    def test_explicit_argument_still_overrides_the_config(self, renamed_graph):
        """An explicit name wins over the config for its own side."""
        result = renamed_graph.select("Site").near_point_m(
            center_lat=60.5,
            center_lon=3.5,
            max_distance_m=50_000.0,
            lat_field="not_a_property",
        )
        assert result.len() == 0
