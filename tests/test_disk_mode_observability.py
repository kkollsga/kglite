"""State-based contracts for what ``graph_info()`` says about disk mode.

Deliberately **no RSS assertions**. ``enable_disk_mode()``'s docstring
promised a ~90% memory reduction for years; measurement found the opposite
in-process (the conversion adds the CSR pages on top of a graph that is
already resident) and the promise only true of the *saved directory reopened
in a fresh process*. RSS is the wrong instrument for a
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
# Every edge carries properties, so a conversion that kept them on the heap
# would show an overlay of exactly this size.
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

    def test_conversion_streams_edge_properties_to_the_mapped_blob(self, frame):
        """The overlay reading, and the contract Phase E2 landed.

        ``enable_disk_mode()`` used to copy every property-bearing edge into
        the disk backend's heap mutation overlay — the dominant term in the
        conversion's in-process memory growth, and pure duplication: the
        in-memory graph it copies from is dropped moments later. The
        properties are now written into the columnar blob beside the CSR as
        the edges are walked, so nothing lands on the heap.
        """
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.enable_disk_mode()

        rows = graph.graph_info()["edge_property_overlay_rows"]
        assert isinstance(rows, int)
        assert 0 <= rows <= graph.graph_info()["edge_count"]
        assert rows == 0  # streamed, not cloned

        result = graph.cypher("MATCH (:Item {nid: 3})-[r:LINKS]->() RETURN r.note AS note").to_list()
        assert result == [{"note": "edge-3"}]

    def test_saving_drains_the_overlay_into_the_mapped_base(self, frame, tmp_path):
        """A save writes the overlay through to the columnar base.

        True before E2 and after it (E2 only removed the reason there was
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


class TestConvertedEdgePropertiesReadBack:
    """The correctness half of streaming the conversion's edge properties.

    ``edge_property_overlay_rows == 0`` above says the properties are not on
    the heap; these say they are still *there* — through the mapped base the
    conversion writes, through the save that rewrites it into the published
    generation, and through a reopen that maps it fresh. A streaming writer
    that emitted a subtly different blob would satisfy the overlay reading and
    lose the data, so the two halves are pinned together.
    """

    @pytest.fixture
    def typed_frame(self):
        nodes = pd.DataFrame({"nid": list(range(NODES)), "name": [f"n{i}" for i in range(NODES)]})
        edges = pd.DataFrame(
            {
                "src": list(range(EDGES)),
                "dst": [(i + 1) % NODES for i in range(EDGES)],
                # Float64 is the shape the external evaluation's graph used;
                # the others cover the remaining scalar encodings.
                "weight": [i + 0.5 for i in range(EDGES)],
                "count": list(range(EDGES)),
                "note": [f"edge-{i}" for i in range(EDGES)],
                "flag": [i % 2 == 0 for i in range(EDGES)],
            }
        )
        return nodes, edges

    QUERY = (
        "MATCH (:Item {nid: 7})-[r:LINKS]->() "
        "RETURN r.weight AS weight, r.count AS count, r.note AS note, r.flag AS flag"
    )
    EXPECTED = [{"weight": 7.5, "count": 7, "note": "edge-7", "flag": False}]

    def test_every_scalar_type_survives_conversion_save_and_reopen(self, typed_frame, tmp_path):
        graph = _fill(kglite.KnowledgeGraph(), typed_frame)
        assert graph.cypher(self.QUERY).to_list() == self.EXPECTED

        graph.enable_disk_mode()
        assert graph.cypher(self.QUERY).to_list() == self.EXPECTED, "lost in the conversion"

        path = str(tmp_path / "typed.kgl")
        graph.save(path)
        assert graph.cypher(self.QUERY).to_list() == self.EXPECTED, "lost in the save"

        reopened = kglite.open(path)
        assert reopened.cypher(self.QUERY).to_list() == self.EXPECTED, "lost in the reopen"

    def test_all_edges_keep_their_properties_not_just_the_probed_one(self, typed_frame, tmp_path):
        """A per-edge offset that drifted would leave *some* edge readable."""
        graph = _fill(kglite.KnowledgeGraph(), typed_frame)
        graph.enable_disk_mode()
        path = str(tmp_path / "all.kgl")
        graph.save(path)

        for source in (graph, kglite.open(path)):
            rows = source.cypher(
                "MATCH (a:Item)-[r:LINKS]->() RETURN a.nid AS nid, r.note AS note, r.weight AS weight"
            ).to_list()
            assert len(rows) == EDGES
            for row in rows:
                assert row["note"] == f"edge-{row['nid']}"
                assert row["weight"] == row["nid"] + 0.5

    def test_a_write_after_conversion_wins_and_persists(self, typed_frame, tmp_path):
        """``SET r.p`` lands in the overlay on top of the streamed base."""
        graph = _fill(kglite.KnowledgeGraph(), typed_frame)
        graph.enable_disk_mode()

        graph.cypher("MATCH (:Item {nid: 7})-[r:LINKS]->() SET r.note = 'rewritten', r.extra = 1.25")
        assert graph.graph_info()["edge_property_overlay_rows"] == 1, "the write is the only overlay row"

        updated = [{"note": "rewritten", "extra": 1.25, "weight": 7.5}]
        probe = "MATCH (:Item {nid: 7})-[r:LINKS]->() RETURN r.note AS note, r.extra AS extra, r.weight AS weight"
        assert graph.cypher(probe).to_list() == updated

        path = str(tmp_path / "mutated.kgl")
        graph.save(path)
        assert graph.graph_info()["edge_property_overlay_rows"] == 0
        assert graph.cypher(probe).to_list() == updated

        reopened = kglite.open(path)
        assert reopened.cypher(probe).to_list() == updated
        # The untouched neighbours came through the same rewrite unharmed.
        assert reopened.cypher("MATCH (:Item {nid: 8})-[r:LINKS]->() RETURN r.note AS note").to_list() == [
            {"note": "edge-8"}
        ]

    def test_a_large_conversion_answers_immediately(self):
        """Correctness at a size where the blob spans many pages."""
        count = 12_000
        nodes = pd.DataFrame({"nid": list(range(count)), "name": [f"n{i}" for i in range(count)]})
        edges = pd.DataFrame(
            {
                "src": list(range(count)),
                "dst": [(i + 1) % count for i in range(count)],
                "weight": [float(i) for i in range(count)],
                "note": [f"edge-{i}" for i in range(count)],
            }
        )
        graph = kglite.KnowledgeGraph()
        graph.add_nodes(nodes, "Item", "nid", "name")
        graph.add_connections(edges, "LINKS", "Item", "src", "Item", "dst")
        totals = "MATCH ()-[r:LINKS]->() RETURN count(r) AS n, sum(r.weight) AS total"
        before = graph.cypher(totals).to_list()

        graph.enable_disk_mode()

        assert graph.graph_info()["edge_property_overlay_rows"] == 0
        assert graph.cypher(totals).to_list() == before
        assert graph.cypher("MATCH (:Item {nid: 11999})-[r:LINKS]->() RETURN r.note AS note").to_list() == [
            {"note": "edge-11999"}
        ]
