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

The pathless ``enable_disk_mode()`` calls below deliberately keep using the
no-argument form: it is the scratch conversion (temp directory, removed when
the graph drops), and since E3 it warns about exactly that. The warning is
expected here, not incidental — ``TestPathlessConversionWarns`` pins it, and
``TestEnableDiskModeAtPath`` covers the ``path=`` form that publishes instead.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import warnings

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


class TestPathlessConversionWarns:
    """The scratch conversion says where it put the data, and how to choose.

    Before E3 it silently materialized the CSR under the system temp directory:
    a conversion bigger than a small (or RAM-backed) ``/tmp`` failed there with
    nothing naming the location, and a caller who assumed the conversion had
    persisted something found nothing after the process exited. The form is
    kept — a throwaway conversion is a real use — but it is no longer silent.
    """

    def test_it_names_the_location_and_the_remedy(self, frame):
        graph = _fill(kglite.KnowledgeGraph(), frame)

        with pytest.warns(UserWarning, match="enable_disk_mode\\(path=") as record:
            graph.enable_disk_mode()

        message = str(record[0].message)
        assert "do not survive" in message, message
        assert tempfile.gettempdir().rstrip("/") in message, ("the warning must name the location", message)
        assert graph.graph_info()["storage_mode"] == "disk", "the conversion still happens"

    def test_the_path_form_does_not_warn(self, frame, tmp_path):
        """Nothing to warn about: the caller chose the location."""
        graph = _fill(kglite.KnowledgeGraph(), frame)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            graph.enable_disk_mode(str(tmp_path / "quiet.kgl"))


class TestEnableDiskModeAtPath:
    """``enable_disk_mode(path=...)`` converts *and publishes* in one call.

    The end state is the one a fresh ``kglite.open(path)`` reads — mapped
    edges, no overlay — which is where the documented small resident footprint
    actually lives. Pinned as state, never as RSS (module docstring).
    """

    def test_the_live_handle_lands_in_the_published_state(self, frame, tmp_path):
        target = tmp_path / "published.kgl"
        graph = _fill(kglite.KnowledgeGraph(), frame)

        graph.enable_disk_mode(str(target))

        info = graph.graph_info()
        assert info["storage_mode"] == "disk"
        assert info["edges_mapped"] is True
        assert info["edge_property_overlay_rows"] == 0
        assert info["edge_count"] == EDGES
        assert graph.cypher("MATCH (:Item {nid: 3})-[r:LINKS]->() RETURN r.note AS note").to_list() == [
            {"note": "edge-3"}
        ]

    def test_the_directory_is_a_published_generation(self, frame, tmp_path):
        target = tmp_path / "generation.kgl"
        graph = _fill(kglite.KnowledgeGraph(), frame)

        graph.enable_disk_mode(str(target))

        assert (target / "CURRENT").is_file(), "no publish happened"
        generations = sorted(child.name for child in (target / "generations").iterdir())
        assert [name for name in generations if name.startswith("gen_")], generations
        # The conversion's scratch lives *inside* the destination (never the
        # system temp directory) and is removed once the publish has rebased
        # every mapping onto the generation.
        leftovers = [child.name for child in target.iterdir() if child.name.startswith(".converting-")]
        assert leftovers == [], f"conversion scratch survived the publish: {leftovers}"

    def test_a_fresh_process_reopens_it(self, frame, tmp_path):
        target = tmp_path / "reopened.kgl"
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.enable_disk_mode(str(target))
        del graph

        script = textwrap.dedent(
            f"""
            import json
            import kglite

            graph = kglite.open({str(target)!r})
            info = graph.graph_info()
            rows = graph.cypher(
                "MATCH (:Item {{nid: 3}})-[r:LINKS]->() RETURN r.note AS note, r.weight AS weight"
            ).to_list()
            fields = ("storage_mode", "edges_mapped", "node_count", "edge_count")
            print(json.dumps({{"info": {{k: info[k] for k in fields}}, "rows": rows}}))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=90, check=False
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        assert payload["info"] == {
            "storage_mode": "disk",
            "edges_mapped": True,
            "node_count": NODES,
            "edge_count": EDGES,
        }
        assert payload["rows"] == [{"note": "edge-3", "weight": 3.0}]

    def test_a_later_bare_save_goes_to_the_directory(self, frame, tmp_path):
        target = tmp_path / "home.kgl"
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.enable_disk_mode(str(target))

        # `value` rather than `name`: a SET of the *title* field (or of a
        # brand-new key) after a disk graph's first publish is lost on reopen —
        # a pre-existing disk-mode defect, unrelated to the conversion, pinned
        # as a strict xfail in tests/test_disk_mutation_roundtrip.py
        # (test_disk_set_after_first_publish_*). Using an ordinary existing
        # property keeps this test measuring what it is about: where the save
        # went.
        graph.cypher("MATCH (n:Item {nid: 5}) SET n.value = 99.5")
        graph.save()  # no argument: the directory is this graph's home now

        reopened = kglite.open(str(target))
        assert reopened.cypher("MATCH (n:Item {nid: 5}) RETURN n.value AS value").to_list() == [{"value": 99.5}]

    def test_it_rebinds_a_graph_that_already_had_a_source_path(self, frame, tmp_path):
        """Same 'save as' rule ``save(path)`` follows: the new target wins.

        There is no separate ``save_as`` here — an explicit path *is* the
        rebind — so a converted graph's later bare ``save()`` writes to the
        directory, and the file it came from is left exactly as it was.
        """
        origin = tmp_path / "origin.kgl"
        _fill(kglite.KnowledgeGraph(), frame).save(str(origin))
        before = origin.stat().st_mtime_ns

        # `durable=False`: a write-ahead log has no place in a disk directory,
        # so a durable handle refuses the conversion outright (pinned by
        # ``test_a_durable_graph_refuses_the_conversion``). This is the
        # write-back entry point without one.
        graph = kglite.open(str(origin), durable=False)
        target = tmp_path / "converted-dir.kgl"
        graph.enable_disk_mode(str(target))
        # An ordinary existing property, for the reason given in
        # test_a_later_bare_save_goes_to_the_directory.
        graph.cypher("MATCH (n:Item {nid: 5}) SET n.value = -1.0")
        graph.save()

        assert origin.is_file() and origin.stat().st_mtime_ns == before, "the origin file was rewritten"
        # Read the origin with ``load`` rather than ``open``: rebinding the save
        # target does not hand back the writer lease the ``open`` above still
        # holds on the file — exactly as ``save(other_path)`` leaves it held.
        assert kglite.load(str(origin)).cypher("MATCH (n:Item {nid: 5}) RETURN n.value AS value").to_list() == [
            {"value": 5.0}
        ]
        assert kglite.open(str(target)).cypher("MATCH (n:Item {nid: 5}) RETURN n.value AS value").to_list() == [
            {"value": -1.0}
        ]

    def test_the_conversion_survives_a_write_and_a_second_publish(self, frame, tmp_path):
        """The published generation is immutable; a second save makes a new one."""
        target = tmp_path / "twice.kgl"
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.enable_disk_mode(str(target))

        graph.cypher("MATCH (:Item {nid: 7})-[r:LINKS]->() SET r.note = 'rewritten'")
        assert graph.graph_info()["edge_property_overlay_rows"] == 1
        graph.save()

        info = graph.graph_info()
        assert info["edges_mapped"] is True
        assert info["edge_property_overlay_rows"] == 0
        generations = sorted(
            child.name for child in (target / "generations").iterdir() if child.name.startswith("gen_")
        )
        assert len(generations) >= 2, generations
        assert kglite.open(str(target)).cypher(
            "MATCH (:Item {nid: 7})-[r:LINKS]->() RETURN r.note AS note"
        ).to_list() == [{"note": "rewritten"}]


class TestConversionRefusals:
    def test_a_durable_graph_refuses_the_conversion(self, frame, tmp_path):
        """Disk mode keeps no logical log, so a WAL-backed handle cannot convert.

        Core refuses it (the backend is a recording wrapper it must not unwrap),
        and the refusal is stated in the caller's vocabulary: ``kglite.open()``
        attaches a log by default, so this is the shape a user actually meets.
        """
        path = str(tmp_path / "durable.kgl")
        _fill(kglite.KnowledgeGraph(), frame).save(path)
        graph = kglite.open(path)  # durable by default

        for argument in ((), (str(tmp_path / "converted"),)):
            with pytest.raises(ValueError, match="durable=False"):
                graph.enable_disk_mode(*argument)

    def test_an_already_converted_graph_refuses_and_names_save(self, frame, tmp_path):
        """Nothing left to convert — and the refusal is not an I/O error."""
        graph = _fill(kglite.KnowledgeGraph(), frame)
        graph.enable_disk_mode(str(tmp_path / "once.kgl"))

        for argument in ((), (str(tmp_path / "twice.kgl"),)):
            with pytest.raises(ValueError, match="already disk-backed"):
                graph.enable_disk_mode(*argument)

        # The operation it names does work.
        graph.save(str(tmp_path / "twice.kgl"))
        assert kglite.open(str(tmp_path / "twice.kgl")).graph_info()["node_count"] == NODES
