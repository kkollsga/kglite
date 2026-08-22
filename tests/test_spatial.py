"""Tests for spatial/geometry operations."""

import pandas as pd
import pytest

from kglite import ArgumentError, KgError, KnowledgeGraph


@pytest.fixture
def geo_graph():
    """Graph with spatial data."""
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "name": [f"Location_{i}" for i in range(1, 11)],
            "latitude": [58.0 + i * 0.5 for i in range(10)],
            "longitude": [2.0 + i * 0.3 for i in range(10)],
        }
    )
    graph.add_nodes(df, "Location", "id", "name")
    return graph


@pytest.fixture
def wkt_graph():
    """Graph with WKT geometry data (no lat/lon, geometry only)."""
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["FieldA", "FieldB", "FieldC"],
            "wkt_geometry": [
                "POLYGON((1 58, 5 58, 5 62, 1 62, 1 58))",
                "POLYGON((3 60, 7 60, 7 64, 3 64, 3 60))",
                "POLYGON((10 50, 15 50, 15 55, 10 55, 10 50))",
            ],
        }
    )
    graph.add_nodes(
        df,
        "Field",
        "id",
        "name",
        column_types={
            "wkt_geometry": "geometry",
        },
    )
    return graph


class TestWithinBounds:
    def test_basic(self, geo_graph):
        result = geo_graph.select("Location").within_bounds(
            lat_field="latitude",
            lon_field="longitude",
            min_lat=59.0,
            max_lat=61.0,
            min_lon=2.0,
            max_lon=4.0,
        )
        assert result.len() > 0
        nodes = result.collect()
        for n in nodes:
            assert 59.0 <= n["latitude"] <= 61.0
            assert 2.0 <= n["longitude"] <= 4.0

    def test_empty_result(self, geo_graph):
        result = geo_graph.select("Location").within_bounds(
            lat_field="latitude",
            lon_field="longitude",
            min_lat=0.0,
            max_lat=1.0,
            min_lon=0.0,
            max_lon=1.0,
        )
        assert result.len() == 0

    def test_all_nodes(self, geo_graph):
        result = geo_graph.select("Location").within_bounds(
            lat_field="latitude",
            lon_field="longitude",
            min_lat=50.0,
            max_lat=70.0,
            min_lon=0.0,
            max_lon=10.0,
        )
        assert result.len() == 10


class TestNearPoint:
    def test_near_point_m(self, geo_graph):
        result = geo_graph.select("Location").near_point_m(
            center_lat=60.0,
            center_lon=3.0,
            max_distance_m=100_000.0,
            lat_field="latitude",
            lon_field="longitude",
        )
        assert result.len() > 0

    def test_near_point_no_matches(self, geo_graph):
        result = geo_graph.select("Location").near_point_m(
            center_lat=0.0,
            center_lon=0.0,
            max_distance_m=1000.0,
            lat_field="latitude",
            lon_field="longitude",
        )
        assert result.len() == 0

    def test_near_point_large_radius(self, geo_graph):
        result = geo_graph.select("Location").near_point_m(
            center_lat=60.0,
            center_lon=3.0,
            max_distance_m=1_000_000.0,
            lat_field="latitude",
            lon_field="longitude",
        )
        assert result.len() == 10


class TestWKTOperations:
    def test_intersects(self, wkt_graph):
        # intersects_geometry(query_wkt, geometry_field=None)
        result = wkt_graph.select("Field").intersects_geometry(
            "POLYGON((2 59, 6 59, 6 63, 2 63, 2 59))",
            geometry_field="wkt_geometry",
        )
        assert result.len() >= 1

    def test_contains_point(self, wkt_graph):
        result = wkt_graph.select("Field").contains_point(
            lat=60.0,
            lon=3.0,
            geometry_field="wkt_geometry",
        )
        assert result.len() >= 1

    def test_wkt_centroid(self, wkt_graph):
        centroid = wkt_graph.wkt_centroid("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
        assert centroid is not None

    def test_near_point_m_geometry_fallback(self, wkt_graph):
        """near_point_m falls back to geometry centroid when no lat/lon."""
        result = wkt_graph.select("Field").near_point_m(
            center_lat=60.0,
            center_lon=3.0,
            max_distance_m=500_000.0,
        )
        assert result.len() >= 1


class TestBounds:
    def test_bounds(self, geo_graph):
        bounds = geo_graph.select("Location").bounds(
            lat_field="latitude",
            lon_field="longitude",
        )
        assert bounds is not None
        assert "min_lat" in bounds
        assert "max_lat" in bounds

    def test_centroid(self, geo_graph):
        centroid = geo_graph.select("Location").centroid(
            lat_field="latitude",
            lon_field="longitude",
        )
        assert centroid is not None


@pytest.fixture
def region_site_graph():
    """One WKT region plus two point sites — the minimal shape `compare()`
    needs, used here to pin its ``target_type`` argument shape."""
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame([{"id": 1, "name": "RegionA", "wkt_geometry": "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"}]),
        "Region",
        "id",
        "name",
    )
    graph.set_spatial("Region", geometry="wkt_geometry")
    graph.add_nodes(
        pd.DataFrame([{"id": 2, "name": "SiteX", "lat": 5.0, "lon": 5.0}]),
        "Site",
        "id",
        "name",
    )
    graph.set_spatial("Site", location=("lat", "lon"))
    return graph


class TestCompareTargetTypeShape:
    """`compare()`'s ``target_type`` accepts a bare string or a list; the
    parsing is shared with `traverse()` and must stay byte-identical in
    behaviour after the two hand-rolled blocks were unified."""

    def test_string_form(self, region_site_graph):
        assert len(region_site_graph.select("Region").compare("Site", "contains").collect()) == 1

    def test_list_form_matches_string_form(self, region_site_graph):
        as_string = region_site_graph.select("Region").compare("Site", "contains").collect()
        as_list = region_site_graph.select("Region").compare(["Site"], "contains").collect()
        assert [r["title"] for r in as_list] == [r["title"] for r in as_string]

    def test_invalid_type_raises_naming_the_parameter(self, region_site_graph):
        with pytest.raises(KgError, match="target_type must be a string or list of strings"):
            region_site_graph.select("Region").compare(123, "contains")

    def test_multi_element_list_refuses_instead_of_truncating(self, region_site_graph):
        """The comparison traversal is single-target; a two-element list used to
        silently compare against element 0 only. It must refuse, not truncate."""
        with pytest.raises(ArgumentError, match=r"compare\(\) supports one target_type; got 2"):
            region_site_graph.select("Region").compare(["Site", "Region"], "contains")

    def test_empty_list_behaves_like_omitted_target(self, region_site_graph):
        """An empty list carries no type, so it reaches the method's own
        missing-target error rather than silently comparing against nothing."""
        with pytest.raises(ValueError, match="requires a target_type"):
            region_site_graph.select("Region").compare([], "contains")
