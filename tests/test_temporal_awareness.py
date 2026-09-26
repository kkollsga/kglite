"""Tests for temporal awareness — auto-filtering nodes and connections by date."""

import os
import tempfile

import pandas as pd
import pytest

import kglite


@pytest.fixture
def temporal_graph():
    """Graph with temporal connections configured via column_types."""
    g = kglite.KnowledgeGraph()

    # Fields
    df_fields = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["DRAUGEN", "ORMEN LANGE"],
        }
    )
    g.add_nodes(df_fields, "Field", "id", "title")

    # Companies
    df_companies = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "title": ["Equinor", "Shell", "BP"],
        }
    )
    g.add_nodes(df_companies, "Company", "id", "title")

    # HAS_LICENSEE connections — temporal via column_types
    df_lic = pd.DataFrame(
        {
            "field_id": [1, 1, 2],
            "company_id": [10, 20, 30],
            "fldLicenseeFrom": ["2000-01-01", "2011-01-01", "1990-01-01"],
            "fldLicenseeTo": ["2010-12-31", "2099-12-31", None],  # None = still active
        }
    )
    g.add_connections(
        df_lic,
        "HAS_LICENSEE",
        source_type="Field",
        source_id_field="field_id",
        target_type="Company",
        target_id_field="company_id",
        column_types={
            "fldLicenseeFrom": "validFrom",
            "fldLicenseeTo": "validTo",
        },
    )

    return g


@pytest.fixture
def temporal_node_graph():
    """Graph with temporal nodes configured via column_types."""
    g = kglite.KnowledgeGraph()

    # FieldStatus nodes with temporal validity via column_types
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Producing", "Shut down", "Producing again"],
            "date_from": ["2000-01-01", "2010-06-01", "2020-01-01"],
            "date_to": ["2010-05-31", "2019-12-31", None],  # None = still active
        }
    )
    g.add_nodes(
        df,
        "FieldStatus",
        "id",
        "title",
        column_types={
            "date_from": "validFrom",
            "date_to": "validTo",
        },
    )

    return g


class TestAutoTemporalConfig:
    """Temporal auto-configuration via column_types."""

    def test_auto_config_edge_type(self, temporal_graph):
        """column_types validFrom/validTo auto-configures connection temporal."""
        result = temporal_graph.select("Field").traverse("HAS_LICENSEE").collect()
        # Should filter to current connections only
        assert len(result) > 0

    def test_auto_config_node_type(self, temporal_node_graph):
        """column_types validFrom/validTo auto-configures node temporal."""
        result = temporal_node_graph.select("FieldStatus").collect()
        # Only the currently valid status (id=3, still active) should remain
        assert len(result) == 1
        assert result[0]["title"] == "Producing again"

    def test_set_temporal_unknown_type(self):
        """set_temporal raises error for unknown types."""
        g = kglite.KnowledgeGraph()
        with pytest.raises(kglite.KgError, match="not a known"):
            g.set_temporal("NonExistent", "from", "to")


class TestDateContext:
    """date() temporal context shifting."""

    def test_date_shift_connections(self, temporal_graph):
        """date('2005') shifts traversal to 2005 context."""
        result = (
            temporal_graph.date("2005").select("Field").where({"title": "DRAUGEN"}).traverse("HAS_LICENSEE").collect()
        )
        # In 2005, only Equinor (2000-2010) was licensee
        titles = [r["title"] for r in result]
        assert "Equinor" in titles
        assert "Shell" not in titles

    def test_date_shift_nodes(self, temporal_node_graph):
        """date('2015') selects status valid in 2015."""
        result = temporal_node_graph.date("2015").select("FieldStatus").collect()
        assert len(result) == 1
        assert result[0]["title"] == "Shut down"

    def test_date_year_only(self, temporal_graph):
        """date('2005') is interpreted as 2005-01-01."""
        result = (
            temporal_graph.date("2005").select("Field").where({"title": "DRAUGEN"}).traverse("HAS_LICENSEE").collect()
        )
        assert len(result) >= 1

    def test_date_reset(self, temporal_node_graph):
        """date(None) resets to today."""
        g = temporal_node_graph.date("2005")
        g2 = g.date()  # reset
        result = g2.select("FieldStatus").collect()
        # Back to today — only currently active status
        assert len(result) == 1
        assert result[0]["title"] == "Producing again"


class TestTemporalTraverse:
    """Temporal filtering on traverse()."""

    def test_auto_filter_current(self, temporal_graph):
        """Default traverse filters to current connections."""
        result = temporal_graph.select("Field").where({"title": "DRAUGEN"}).traverse("HAS_LICENSEE").collect()
        # Today > 2011, so only Shell (2011-2099)
        titles = [r["title"] for r in result]
        assert "Shell" in titles
        assert "Equinor" not in titles

    def test_temporal_false_disables(self, temporal_graph):
        """temporal=False disables filtering."""
        result = (
            temporal_graph.select("Field")
            .where({"title": "DRAUGEN"})
            .traverse("HAS_LICENSEE", temporal=False)
            .collect()
        )
        titles = [r["title"] for r in result]
        assert "Shell" in titles
        assert "Equinor" in titles

    def test_at_override(self, temporal_graph):
        """at='2005' overrides to point-in-time."""
        result = (
            temporal_graph.select("Field").where({"title": "DRAUGEN"}).traverse("HAS_LICENSEE", at="2005").collect()
        )
        titles = [r["title"] for r in result]
        assert "Equinor" in titles
        assert "Shell" not in titles

    def test_null_valid_to_means_active(self, temporal_graph):
        """NULL valid_to = still active (open-ended)."""
        result = temporal_graph.select("Field").where({"title": "ORMEN LANGE"}).traverse("HAS_LICENSEE").collect()
        # BP has null valid_to → always current
        titles = [r["title"] for r in result]
        assert "BP" in titles


class TestTemporalSelect:
    """Temporal filtering on select()."""

    def test_auto_filter_nodes(self, temporal_node_graph):
        """select() auto-filters temporal nodes to current."""
        result = temporal_node_graph.select("FieldStatus").collect()
        assert len(result) == 1
        assert result[0]["title"] == "Producing again"

    def test_temporal_false_all_nodes(self, temporal_node_graph):
        """temporal=False includes all historic nodes."""
        result = temporal_node_graph.select("FieldStatus", temporal=False).collect()
        assert len(result) == 3

    def test_non_temporal_type_unaffected(self, temporal_graph):
        """Types without temporal config are unaffected."""
        result = temporal_graph.select("Company").collect()
        assert len(result) == 3


class TestValidAtAutoDetect:
    """valid_at() / valid_during() auto-detect field names."""

    def test_valid_at_auto_fields(self, temporal_node_graph):
        """valid_at with temporal config auto-detects fields."""
        result = (
            temporal_node_graph.select("FieldStatus", temporal=False)  # get all first
            .valid_at("2015-06-15")
            .collect()
        )
        assert len(result) == 1
        assert result[0]["title"] == "Shut down"

    def test_valid_at_no_args_uses_today(self, temporal_node_graph):
        """valid_at() with no args uses reference date (today)."""
        result = temporal_node_graph.select("FieldStatus", temporal=False).valid_at().collect()
        assert len(result) == 1
        assert result[0]["title"] == "Producing again"

    def test_valid_during_auto_fields(self, temporal_node_graph):
        """valid_during with temporal config auto-detects fields."""
        result = (
            temporal_node_graph.select("FieldStatus", temporal=False).valid_during("2005-01-01", "2015-01-01").collect()
        )
        # Overlaps with "Producing" (2000-2010) and "Shut down" (2010-2019)
        assert len(result) == 2


class TestSaveLoadTemporal:
    """Temporal configs persist through save/load."""

    def test_save_load_preserves_edge_config(self, temporal_graph):
        """Temporal edge config survives save/load."""
        with tempfile.NamedTemporaryFile(suffix=".kglite", delete=False) as f:
            path = f.name
        try:
            temporal_graph.save(path)
            g2 = kglite.load(path)

            # Config should be preserved — traverse should still auto-filter
            result = g2.select("Field").where({"title": "DRAUGEN"}).traverse("HAS_LICENSEE").collect()
            titles = [r["title"] for r in result]
            assert "Shell" in titles
            assert "Equinor" not in titles
        finally:
            os.unlink(path)

    def test_save_load_preserves_node_config(self, temporal_node_graph):
        """Temporal node config survives save/load."""
        with tempfile.NamedTemporaryFile(suffix=".kglite", delete=False) as f:
            path = f.name
        try:
            temporal_node_graph.save(path)
            g2 = kglite.load(path)

            result = g2.select("FieldStatus").collect()
            assert len(result) == 1
            assert result[0]["title"] == "Producing again"
        finally:
            os.unlink(path)


class TestDateRange:
    """date('start', 'end') range mode — overlap check."""

    def test_range_includes_overlapping_connections(self, temporal_graph):
        """date('2005', '2015') includes both Equinor (2000-2010) and Shell (2011-2099)."""
        result = (
            temporal_graph.date("2005", "2015")
            .select("Field")
            .where({"title": "DRAUGEN"})
            .traverse("HAS_LICENSEE")
            .collect()
        )
        titles = [r["title"] for r in result]
        assert "Equinor" in titles
        assert "Shell" in titles

    def test_range_excludes_outside_connections(self, temporal_graph):
        """date('1990', '1999') excludes connections that don't overlap."""
        result = (
            temporal_graph.date("1990", "1999")
            .select("Field")
            .where({"title": "DRAUGEN"})
            .traverse("HAS_LICENSEE")
            .collect()
        )
        titles = [r["title"] for r in result]
        # Equinor starts 2000, Shell starts 2011 — neither overlaps 1990-1999
        assert "Equinor" not in titles
        assert "Shell" not in titles

    def test_range_nodes(self, temporal_node_graph):
        """date('2005', '2015') includes overlapping node statuses."""
        result = temporal_node_graph.date("2005", "2015").select("FieldStatus").collect()
        # "Producing" (2000-2010) and "Shut down" (2010-2019) overlap 2005-2015
        assert len(result) == 2
        titles = [r["title"] for r in result]
        assert "Producing" in titles
        assert "Shut down" in titles

    def test_range_year_expansion(self, temporal_graph):
        """End date '2010' expands to 2010-12-31 (includes Equinor ending 2010-12-31)."""
        result = (
            temporal_graph.date("2010", "2010")
            .select("Field")
            .where({"title": "DRAUGEN"})
            .traverse("HAS_LICENSEE")
            .collect()
        )
        titles = [r["title"] for r in result]
        # Equinor valid_to=2010-12-31, Shell valid_from=2011-01-01
        assert "Equinor" in titles


class TestDateAll:
    """date('all') disables temporal filtering entirely."""

    def test_all_disables_node_filtering(self, temporal_node_graph):
        """date('all') returns all nodes regardless of validity."""
        result = temporal_node_graph.date("all").select("FieldStatus").collect()
        assert len(result) == 3

    def test_all_disables_connection_filtering(self, temporal_graph):
        """date('all') returns all connections regardless of validity."""
        result = (
            temporal_graph.date("all").select("Field").where({"title": "DRAUGEN"}).traverse("HAS_LICENSEE").collect()
        )
        titles = [r["title"] for r in result]
        assert "Equinor" in titles
        assert "Shell" in titles

    def test_all_then_reset(self, temporal_node_graph):
        """date('all').date() resets back to today filtering."""
        result = temporal_node_graph.date("all").date().select("FieldStatus").collect()
        assert len(result) == 1
        assert result[0]["title"] == "Producing again"

    def test_end_without_start_raises(self, temporal_graph):
        """date(None, '2015') raises an error."""
        with pytest.raises(kglite.KgError, match="requires a start"):
            temporal_graph.date(None, "2015")


class TestDescribeTemporal:
    """Temporal info in describe() output."""

    def test_describe_shows_temporal_node(self, temporal_node_graph):
        """describe() includes temporal_from/temporal_to for node types."""
        xml = temporal_node_graph.describe()
        assert 'temporal_from="date_from"' in xml
        assert 'temporal_to="date_to"' in xml

    def test_describe_shows_temporal_edge(self, temporal_graph):
        """describe() includes temporal_from/temporal_to for connection types."""
        xml = temporal_graph.describe()
        assert 'temporal_from="fldLicenseeFrom"' in xml
        assert 'temporal_to="fldLicenseeTo"' in xml


@pytest.fixture
def bound_kinds_graph():
    """Three `:T` nodes valid 2015-01-01..2016-12-31, their bounds stored as
    a date, a datetime and an ISO string, configured with set_temporal."""
    g = kglite.KnowledgeGraph()
    g.cypher(
        """
        CREATE (:T {id: 'date_bounded', title: 'date_bounded', vf: date('2015-01-01'), vt: date('2016-12-31')}),
               (:T {id: 'ts_bounded', title: 'ts_bounded',
                    vf: datetime('2015-01-01T00:00:00'), vt: datetime('2016-12-31T00:00:00')}),
               (:T {id: 'str_bounded', title: 'str_bounded', vf: '2015-01-01', vt: '2016-12-31'})
        """
    )
    g.set_temporal("T", "vf", "vt")
    return g


def _titles(result):
    return sorted(row["title"] for row in result.collect())


class TestFluentBoundKinds:
    """Fluent filters honour date, datetime and ISO-string bounds alike."""

    def test_date_context_before_the_interval(self, bound_kinds_graph):
        assert _titles(bound_kinds_graph.date("2009").select("T")) == []

    def test_default_context_after_the_interval(self, bound_kinds_graph):
        assert _titles(bound_kinds_graph.select("T")) == []

    def test_date_context_inside_the_interval(self, bound_kinds_graph):
        assert _titles(bound_kinds_graph.date("2016").select("T")) == ["date_bounded", "str_bounded", "ts_bounded"]

    def test_valid_at_before_the_interval(self, bound_kinds_graph):
        assert _titles(bound_kinds_graph.select("T", temporal=False).valid_at("2009-06-30")) == []

    def test_valid_at_inside_the_interval(self, bound_kinds_graph):
        got = _titles(bound_kinds_graph.select("T", temporal=False).valid_at("2015-06-30"))
        assert got == ["date_bounded", "str_bounded", "ts_bounded"]

    def test_valid_during_outside_and_inside(self, bound_kinds_graph):
        base = bound_kinds_graph.select("T", temporal=False)
        assert _titles(base.valid_during("2009-01-01", "2010-12-31")) == []
        assert _titles(base.valid_during("2016-06-01", "2020-01-01")) == ["date_bounded", "str_bounded", "ts_bounded"]

    def test_range_context_outside(self, bound_kinds_graph):
        assert _titles(bound_kinds_graph.date("2009", "2010").select("T")) == []

    def test_cypher_agrees(self, bound_kinds_graph):
        rows = bound_kinds_graph.cypher("MATCH (n:T) WHERE valid_at(n, date('2009'), 'vf', 'vt') RETURN n.id").to_list()
        assert rows == []

    def test_unreadable_bound_raises(self):
        g = kglite.KnowledgeGraph()
        g.cypher("CREATE (:T {id: 'bad', title: 'bad', vf: 'someday', vt: date('2016-12-31')})")
        # set_temporal validates the stored bounds...
        with pytest.raises(kglite.ArgumentError, match="node 'bad'.*someday"):
            g.set_temporal("T", "vf", "vt")
        # ...but a write onto a declared type is not validated, so the fluent
        # filters report a bound written afterwards.
        g.cypher("MATCH (n:T) SET n.vf = date('2015-01-01')")
        g.set_temporal("T", "vf", "vt")
        g.cypher("MATCH (n:T) SET n.vf = 'someday'")
        with pytest.raises(ValueError, match="someday"):
            g.date("2015").select("T")
        with pytest.raises(ValueError, match="bad"):
            g.select("T", temporal=False).valid_at("2015-06-30")

    def test_traverse_edge_bounds_of_every_kind(self):
        g = kglite.KnowledgeGraph()
        g.cypher(
            """
            CREATE (f:F {id: 1, title: 'f'}),
                   (f)-[:L {vf: datetime('2015-01-01T00:00:00'), vt: datetime('2016-12-31T00:00:00')}]->
                       (:C {id: 10, title: 'ts'}),
                   (f)-[:L {vf: '2015-01-01', vt: '2016-12-31'}]->(:C {id: 20, title: 'str'})
            """
        )
        g.set_temporal("L", "vf", "vt")
        assert _titles(g.select("F").traverse("L", at="2009-06-30")) == []
        assert _titles(g.select("F").traverse("L", at="2015-06-30")) == ["str", "ts"]


class TestSelectTemporalRequired:
    def test_temporal_true_on_unconfigured_type_raises(self, temporal_graph):
        with pytest.raises(ValueError, match="Company"):
            temporal_graph.select("Company", temporal=True)

    def test_temporal_true_on_configured_type_filters(self, temporal_node_graph):
        result = temporal_node_graph.select("FieldStatus", temporal=True).collect()
        assert [r["title"] for r in result] == ["Producing again"]
