"""Tests for auto-vacuum feature (Phase 2).

Auto-vacuum automatically compacts the graph after DELETE operations
when fragmentation exceeds a configurable threshold.
"""

import os
import tempfile

import pandas as pd
import pytest

import kglite


def make_large_graph(n=500):
    """Create a graph with n Person nodes."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": list(range(n)),
            "title": [f"Person_{i}" for i in range(n)],
            "age": [20 + (i % 50) for i in range(n)],
        }
    )
    g.add_nodes(df, "Person", "id", "title")
    return g


class TestAutoVacuumDefault:
    """Auto-vacuum is enabled by default at 0.3 threshold."""

    def test_default_threshold_in_graph_info(self):
        """New graphs should have auto-vacuum enabled, at 0.3."""
        g = kglite.KnowledgeGraph()
        assert g.graph_info()["auto_vacuum_threshold"] == 0.3
        g.set_auto_vacuum(0.3)
        assert g.graph_info()["auto_vacuum_threshold"] == 0.3

    def test_set_auto_vacuum_none_disables(self):
        """Setting None disables auto-vacuum."""
        g = kglite.KnowledgeGraph()
        g.set_auto_vacuum(None)
        # Should not error

    def test_set_auto_vacuum_float(self):
        """Setting a float between 0 and 1 works."""
        g = kglite.KnowledgeGraph()
        g.set_auto_vacuum(0.1)
        g.set_auto_vacuum(0.5)
        g.set_auto_vacuum(1.0)
        g.set_auto_vacuum(0.0)

    def test_set_auto_vacuum_invalid_raises(self):
        """Setting a value outside [0, 1] raises ValueError."""
        g = kglite.KnowledgeGraph()
        with pytest.raises(Exception):
            g.set_auto_vacuum(1.5)
        with pytest.raises(Exception):
            g.set_auto_vacuum(-0.1)


class TestAutoVacuumTrigger:
    """Auto-vacuum triggers after DELETE when threshold is exceeded."""

    def test_auto_vacuum_triggers_on_heavy_deletion(self):
        """Deleting many nodes should trigger auto-vacuum, removing tombstones."""
        g = make_large_graph(500)
        g.set_auto_vacuum(0.2)  # 20% threshold

        # Delete ~60% of nodes (300 out of 500)
        g.cypher("MATCH (n:Person) WHERE n.age < 40 DETACH DELETE n")

        # After auto-vacuum, fragmentation should be low
        info = g.graph_info()
        # Auto-vacuum should have fired (300 tombstones > 100, ratio > 0.2)
        # After vacuum, tombstones should be 0
        assert info["node_tombstones"] == 0
        assert info["fragmentation_ratio"] == 0.0

    def test_auto_vacuum_does_not_trigger_below_threshold(self):
        """Deleting few nodes should NOT trigger auto-vacuum."""
        g = make_large_graph(500)
        g.set_auto_vacuum(0.5)  # 50% threshold — very lenient

        # Delete only ~10 nodes (age == 20 in a modular pattern)
        g.cypher("MATCH (n:Person) WHERE n.age = 20 DETACH DELETE n")
        stats = g.last_mutation_stats
        deleted = stats["nodes_deleted"]

        info = g.graph_info()
        # With only ~10 deleted out of 500, ratio is ~2% — well below 50%
        # Also tombstones ≤ 100, so auto-vacuum should NOT trigger
        assert info["node_tombstones"] == deleted
        assert info["node_tombstones"] > 0

    def test_auto_vacuum_disabled_no_compaction(self):
        """With auto-vacuum disabled, tombstones remain after DELETE."""
        g = make_large_graph(500)
        g.set_auto_vacuum(None)  # Disable

        # Delete many nodes
        g.cypher("MATCH (n:Person) WHERE n.age < 40 DETACH DELETE n")

        info = g.graph_info()
        # Tombstones should still be present since auto-vacuum is off
        assert info["node_tombstones"] > 0

    def test_auto_vacuum_respects_100_tombstone_minimum(self):
        """Auto-vacuum requires > 100 tombstones even if ratio is high."""
        g = make_large_graph(150)
        g.set_auto_vacuum(0.1)  # Very aggressive threshold

        # Delete 50 nodes — high ratio (~33%) but tombstones ≤ 100
        g.cypher("MATCH (n:Person) WHERE n.age < 30 DETACH DELETE n")
        stats = g.last_mutation_stats
        deleted = stats["nodes_deleted"]

        info = g.graph_info()
        if deleted <= 100:
            # Tombstones ≤ 100, auto-vacuum should NOT trigger
            assert info["node_tombstones"] == deleted
        else:
            # If more than 100 deleted, it may have triggered
            assert info["node_tombstones"] <= deleted


class TestAutoVacuumDataIntegrity:
    """Data should be intact after auto-vacuum fires."""

    def test_data_survives_auto_vacuum(self):
        """Remaining nodes should be queryable after auto-vacuum compaction."""
        g = make_large_graph(500)
        g.set_auto_vacuum(0.2)

        # Count nodes with age >= 40 before deletion
        before = g.cypher("MATCH (n:Person) WHERE n.age >= 40 RETURN count(n) AS cnt")
        count_before = before[0]["cnt"]

        # Delete nodes with age < 40 (triggers auto-vacuum)
        g.cypher("MATCH (n:Person) WHERE n.age < 40 DETACH DELETE n")

        # Surviving nodes should still be queryable
        after = g.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        count_after = after[0]["cnt"]
        assert count_after == count_before

        # Specific queries should still work
        result = g.cypher("MATCH (n:Person) WHERE n.age = 50 RETURN n.title ORDER BY n.title")
        assert len(result) > 0

    def test_filter_works_after_auto_vacuum(self):
        """Fluent API should work after auto-vacuum reindexes."""
        g = make_large_graph(500)
        g.set_auto_vacuum(0.2)

        g.cypher("MATCH (n:Person) WHERE n.age < 40 DETACH DELETE n")

        # Filter should still work (indices were rebuilt)
        result = g.where({"type": "Person"}).collect()
        assert len(result) > 0
        for node in result:
            assert node["age"] >= 40


class TestAutoVacuumPersistence:
    """Auto-vacuum threshold survives save/load."""

    def test_threshold_persists(self):
        """Custom threshold should survive save/load."""
        g = kglite.KnowledgeGraph()
        # Single add_nodes with interleaved ages — deletions leave mid-graph tombstones
        ids = list(range(600))
        ages = [25 if i % 2 == 0 else 50 for i in range(600)]  # even=remove, odd=keep
        df = pd.DataFrame(
            {
                "id": ids,
                "title": [f"P_{i}" for i in range(600)],
                "age": ages,
            }
        )
        g.add_nodes(df, "Person", "id", "title")
        g.set_auto_vacuum(0.15)

        with tempfile.NamedTemporaryFile(suffix=".kglite", delete=False) as f:
            path = f.name

        try:
            g.save(path)
            g2 = kglite.load(path)

            g2.cypher("MATCH (n:Person) WHERE n.age = 25 DETACH DELETE n")

            # With threshold 0.15, should have auto-vacuumed (300 tombstones > 100, ratio ~50% > 15%)
            info = g2.graph_info()
            assert info["node_tombstones"] == 0
        finally:
            os.unlink(path)

    def test_disabled_threshold_persists(self):
        """Disabled (None) threshold should survive save/load."""
        g = kglite.KnowledgeGraph()
        # Use a single add_nodes with interleaved ages so deletions leave mid-graph tombstones
        ids = list(range(600))
        ages = [25 if i % 2 == 0 else 50 for i in range(600)]  # even=remove, odd=keep
        df = pd.DataFrame(
            {
                "id": ids,
                "title": [f"P_{i}" for i in range(600)],
                "age": ages,
            }
        )
        g.add_nodes(df, "Person", "id", "title")
        g.set_auto_vacuum(None)

        with tempfile.NamedTemporaryFile(suffix=".kglite", delete=False) as f:
            path = f.name

        try:
            g.save(path)
            g2 = kglite.load(path)

            g2.cypher("MATCH (n:Person) WHERE n.age = 25 DETACH DELETE n")

            # With auto-vacuum disabled, tombstones should remain
            info = g2.graph_info()
            assert info["node_tombstones"] > 0
        finally:
            os.unlink(path)


class TestManualVacuumStillWorks:
    """Manual vacuum() should still work alongside auto-vacuum."""

    def test_manual_vacuum_after_auto_disabled(self):
        """Manual vacuum works when auto-vacuum is off."""
        g = make_large_graph(500)
        g.set_auto_vacuum(None)

        g.cypher("MATCH (n:Person) WHERE n.age < 40 DETACH DELETE n")
        info = g.graph_info()
        assert info["node_tombstones"] > 0

        result = g.vacuum()
        assert result["tombstones_removed"] > 0

        info = g.graph_info()
        assert info["node_tombstones"] == 0
