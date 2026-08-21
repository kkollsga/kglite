"""State-based contracts for what ``graph_info()`` says about disk mode.

Deliberately **no RSS assertions**. ``enable_disk_mode()``'s docstring
promised a ~90% memory reduction for years; measurement found the opposite
in-process (the conversion adds the CSR pages and clones every edge's
properties into a heap overlay) and the promise only true of the *saved
directory reopened in a fresh process*. RSS is the wrong instrument for a
regression test — allocator retention makes it lie in both directions — so
what is pinned here is the graph's observable *shape*:

* ``edges_mapped`` — the CSR is file-backed,
* ``edge_property_overlay_rows`` — how much edge-property data is still on
  the heap rather than paged,
* ``columnar_is_mapped`` — property-column spilling, which is a *different*
  question and answers ``False`` on a perfectly healthy disk graph.

That last one was misread as a disk-mode regression fingerprint by an
external evaluation; the meaning is pinned below so a future reader can see
which knob it tracks.
"""

import pandas as pd
import pytest

import kglite

NODES = 200
# Every edge carries properties, so the conversion's edge-property overlay is
# exactly the edge count until E2 streams it to the mapped blob.
EDGES = NODES


@pytest.fixture
def frame():
    nodes = pd.DataFrame(
        {
            "nid": list(range(NODES)),
            "name": [f"n{i}" for i in range(NODES)],
            "value": [float(i) for i in range(NODES)],
        }
    )
    edges = pd.DataFrame(
        {
            "src": list(range(EDGES)),
            "dst": [(i + 1) % NODES for i in range(EDGES)],
            "weight": [float(i) for i in range(EDGES)],
            "note": [f"edge-{i}" for i in range(EDGES)],
        }
    )
    return nodes, edges


def _fill(graph, frame):
    nodes, edges = frame
    graph.add_nodes(nodes, "Item", "nid", "name")
    graph.add_connections(edges, "LINKS", "Item", "src", "Item", "dst")
    return graph


class TestEnableDiskModeShape:
    def test_conversion_maps_the_edge_csr(self, frame):
        """``enable_disk_mode()`` materializes a file-backed CSR."""
        graph = _fill(kglite.KnowledgeGraph(), frame)

        before = graph.graph_info()
        assert before["storage_mode"] == "memory"
        assert before["edges_mapped"] is False, "a memory graph has no CSR to map"
        assert before["edge_property_overlay_rows"] == 0

        graph.enable_disk_mode()

        after = graph.graph_info()
        assert after["storage_mode"] == "disk"
        assert after["edges_mapped"] is True
        assert after["edge_count"] == EDGES

    def test_conversion_clones_edge_properties_onto_the_heap(self, frame):
        """The overlay reading, and the number Phase E2 is measured by.

        ``enable_disk_mode()`` copies every property-bearing edge into the
        disk backend's heap mutation overlay instead of streaming it to the
        mapped blob — the dominant term in the conversion's in-process memory
        growth. **E2 streams it, which drives this to 0**; when it lands,
        tighten the marked assertion to ``== 0``. The bound above it holds
        either way.
        """
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.enable_disk_mode()

        rows = graph.graph_info()["edge_property_overlay_rows"]
        assert isinstance(rows, int)
        assert 0 <= rows <= graph.graph_info()["edge_count"]
        assert rows == EDGES  # E2 marker: becomes 0 once properties stream

        # Whatever the overlay holds, the properties read back.
        result = graph.cypher("MATCH (:Item {nid: 3})-[r:LINKS]->() RETURN r.note AS note").to_list()
        assert result == [{"note": "edge-3"}]

    def test_saving_drains_the_overlay_into_the_mapped_base(self, frame, tmp_path):
        """A save writes the overlay through to the columnar base.

        True today and after E2 (which only removes the reason there was
        anything to drain), so this is the durable half of the contract.
        """
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.enable_disk_mode()
        graph.save(str(tmp_path / "converted.kgl"))

        info = graph.graph_info()
        assert info["edge_property_overlay_rows"] == 0
        assert info["edges_mapped"] is True

        reopened = kglite.open(str(tmp_path / "converted.kgl"))
        reopened_info = reopened.graph_info()
        assert reopened_info["storage_mode"] == "disk"
        assert reopened_info["edges_mapped"] is True
        assert reopened_info["edge_property_overlay_rows"] == 0
        assert reopened.cypher("MATCH (:Item {nid: 3})-[r:LINKS]->() RETURN r.note AS note").to_list() == [
            {"note": "edge-3"}
        ]


class TestFromScratchDiskGraph:
    def test_reports_the_same_fields(self, frame, tmp_path):
        """A ``storage="disk"`` graph answers both fields, and honestly.

        Its edges land in the mutation overflow, not a CSR, so ``edges_mapped``
        is ``False`` until the first save builds one — the field reports the
        structure that exists, not the mode that was requested.
        """
        path = str(tmp_path / "fresh.kgl")
        graph = _fill(kglite.KnowledgeGraph(storage="disk", path=path), frame)

        info = graph.graph_info()
        assert info["storage_mode"] == "disk"
        assert isinstance(info["edges_mapped"], bool)
        assert isinstance(info["edge_property_overlay_rows"], int)
        assert info["edges_mapped"] is False, "no CSR has been built yet"

        graph.save(path)

        saved = graph.graph_info()
        assert saved["edges_mapped"] is True
        assert saved["edge_property_overlay_rows"] == 0


class TestColumnarIsMappedMeaning:
    """``columnar_is_mapped`` tracks property-column spilling. Only that."""

    def test_disk_mode_leaves_it_false(self, frame):
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.enable_disk_mode()

        info = graph.graph_info()
        assert info["edges_mapped"] is True
        assert info["columnar_is_mapped"] is False, (
            "columnar_is_mapped reports column spilling, not disk-mode health; "
            "False is disk mode's permanent shape and not a regression"
        )

    def test_a_memory_limit_flips_it_without_touching_the_edges(self, frame):
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.set_memory_limit(0)
        graph.cypher("MATCH (n:Item {nid: 1}) SET n.name = 'spilled'")

        info = graph.graph_info()
        assert info["columnar_is_mapped"] is True
        assert info["storage_mode"] == "memory"
        assert info["edges_mapped"] is False, "a spill moves columns, never edges"

    def test_mapped_mode_flips_it_without_touching_the_edges(self, frame):
        graph = _fill(kglite.KnowledgeGraph(storage="mapped"), frame)

        info = graph.graph_info()
        assert info["storage_mode"] == "mapped"
        assert info["columnar_is_mapped"] is True
        assert info["edges_mapped"] is False
        assert info["edge_property_overlay_rows"] == 0
