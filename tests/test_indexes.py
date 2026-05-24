"""Tests for index management: create, drop, list, composite, performance."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def indexed_graph():
    """Graph with data suitable for indexing tests."""
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": list(range(1, 101)),
            "name": [f"Node_{i}" for i in range(1, 101)],
            "category": [f"Cat_{i % 5}" for i in range(100)],
            "value": [i * 10 for i in range(100)],
        }
    )
    graph.add_nodes(df, "Item", "id", "name")
    return graph


class TestCreateDrop:
    def test_create_index(self, indexed_graph):
        indexed_graph.create_index("Item", "category")
        assert indexed_graph.has_index("Item", "category")

    def test_drop_index(self, indexed_graph):
        indexed_graph.create_index("Item", "category")
        indexed_graph.drop_index("Item", "category")
        assert not indexed_graph.has_index("Item", "category")

    def test_has_index_false(self, indexed_graph):
        assert not indexed_graph.has_index("Item", "nonexistent")

    def test_list_indexes(self, indexed_graph):
        indexed_graph.create_index("Item", "category")
        indexed_graph.create_index("Item", "value")
        indexes = indexed_graph.list_indexes()
        assert len(indexes) >= 2


class TestIndexUsage:
    def test_filter_uses_index(self, indexed_graph):
        indexed_graph.create_index("Item", "category")
        result = indexed_graph.select("Item").where({"category": "Cat_0"})
        assert result.len() == 20

    def test_filter_no_index_same_result(self, indexed_graph):
        # Without index
        result_no_idx = indexed_graph.select("Item").where({"category": "Cat_1"})
        count_no_idx = result_no_idx.len()

        # With index
        indexed_graph.create_index("Item", "category")
        result_idx = indexed_graph.select("Item").where({"category": "Cat_1"})
        count_idx = result_idx.len()

        assert count_no_idx == count_idx

    def test_index_stats(self, indexed_graph):
        indexed_graph.create_index("Item", "category")
        stats = indexed_graph.index_stats("Item", "category")
        assert stats is not None

    def test_rebuild_indexes(self, indexed_graph):
        indexed_graph.create_index("Item", "category")
        indexed_graph.rebuild_indexes()
        assert indexed_graph.has_index("Item", "category")


class TestCompositeIndex:
    def test_create_composite(self, indexed_graph):
        indexed_graph.create_composite_index("Item", ["category", "value"])
        assert indexed_graph.has_composite_index("Item", ["category", "value"])

    def test_drop_composite(self, indexed_graph):
        indexed_graph.create_composite_index("Item", ["category", "value"])
        indexed_graph.drop_composite_index("Item", ["category", "value"])
        assert not indexed_graph.has_composite_index("Item", ["category", "value"])

    def test_list_composite(self, indexed_graph):
        indexed_graph.create_composite_index("Item", ["category", "value"])
        composites = indexed_graph.list_composite_indexes()
        assert len(composites) >= 1


class TestEmptyGraph:
    def test_index_on_empty(self):
        graph = KnowledgeGraph()
        # Should not error
        indexes = graph.list_indexes()
        assert len(indexes) == 0


class TestIndexRebuildAfterReload:
    """Regression guard for the 0.10.1 fix where `create_index` /
    `create_range_index` / `create_composite_index` returned 0 entries
    when called on a graph loaded from .kgl in memory/mapped storage.

    Root cause: `NodeData::get_property()` only reads the in-memory
    `PropertyStorage::Map`/`Compact` snapshot, which is stripped during
    save. The matcher's hot path uses
    `GraphBackend::get_node_property()` (backend-aware) plus alias
    resolution + id/title special-casing. The fix made `create_index`
    use the same path.
    """

    @pytest.fixture
    def reloaded_graph(self, tmp_path):
        # Build, save, reload — round-trips properties through the .kgl
        # format so loaded NodeData sees column-stored values rather
        # than the in-memory snapshot the build path populates.
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "starId": [f"s{i}" for i in range(100)],
                "title": [f"Star {i}" for i in range(100)],
                "sector": [i // 10 for i in range(100)],
                "lum": [float(i) * 1.5 for i in range(100)],
            }
        )
        graph.add_nodes(df, "Star", "starId", "title")
        kgl = tmp_path / "stars.kgl"
        graph.save(str(kgl))
        from kglite import load

        return load(str(kgl))

    def test_create_index_on_alias_after_reload(self, reloaded_graph):
        # `starId` is an id-alias declared via add_nodes; create_index
        # must resolve the alias to "id" and read via get_node_id() —
        # without this, the matcher's column path is used but
        # create_index would see 0 values.
        result = reloaded_graph.create_index("Star", "starId")
        assert result["unique_values"] == 100, (
            f"create_index(Star, starId) on reloaded .kgl returned "
            f"{result['unique_values']} entries — expected 100. "
            f"Regression in alias-resolving column-store read path."
        )

    def test_create_index_on_id_after_reload(self, reloaded_graph):
        # Same as above but via the canonical "id" key directly.
        result = reloaded_graph.create_index("Star", "id")
        assert result["unique_values"] == 100

    def test_create_index_on_plain_property_after_reload(self, reloaded_graph):
        # Non-id, non-title property — exercises the
        # get_node_property() backend dispatch.
        result = reloaded_graph.create_index("Star", "sector")
        assert result["unique_values"] == 10  # 100 stars / 10 per sector

    def test_create_range_index_after_reload(self, reloaded_graph):
        result = reloaded_graph.create_range_index("Star", "lum")
        assert result["unique_values"] == 100

    def test_create_composite_index_after_reload(self, reloaded_graph):
        result = reloaded_graph.create_composite_index("Star", ["sector", "lum"])
        assert result["unique_combinations"] == 100  # every (sector, lum) pair is unique

    def test_create_composite_with_alias_after_reload(self, reloaded_graph):
        # Composite index that mixes an id-alias with a plain property —
        # alias resolution must happen per-column.
        result = reloaded_graph.create_composite_index("Star", ["starId", "sector"])
        assert result["unique_combinations"] == 100
